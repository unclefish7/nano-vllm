import pickle
import warnings
from datetime import timedelta
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models import check_model_contract, resolve_model_class
from nanovllm.layers.request_state import RequestStatePool
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import can_use_flash_attention
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.model_cls = resolve_model_class(hf_config)
        check_model_contract(self.model_cls)
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            f"tcp://127.0.0.1:{config.master_port}",
            world_size=self.world_size,
            rank=rank,
            timeout=timedelta(seconds=120),
        )
        default_dtype = torch.get_default_dtype()
        legacy_gpu = torch.cuda.get_device_capability(rank)[0] < 8
        if legacy_gpu and hf_config.dtype != torch.float32:
            warnings.warn("Using float32 on pre-Ampere GPU for compatibility.")
            hf_config.dtype = torch.float32
        if not can_use_flash_attention(torch.device("cuda", rank), hf_config.dtype):
            warnings.warn("Using PyTorch math attention; CUDA graphs are disabled.")
            self.enforce_eager = True
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = self.model_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_request_states()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i, seq in enumerate(seqs):
            seq.num_scheduled_tokens = seq_len
            if self.request_states.nbytes:
                seq.state_slot = i
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_request_states(self):
        config = self.config
        max_bs = min(config.max_num_seqs, 512)
        self.graph_bs = sorted({b for b in [1, 2, 4, 8, *range(16, max_bs, 16), max_bs] if b <= max_bs})
        # 图补齐最多需要两个相邻批大小之差减一行。每行独占一个槽，避免并发写冲突。
        self.num_padding_slots = 0 if self.enforce_eager else max(
            (b - a - 1 for a, b in zip(self.graph_bs, self.graph_bs[1:])), default=0)
        free, total = torch.cuda.mem_get_info()
        budget = int(total * config.gpu_memory_utilization) - (total - free)
        self.request_states = RequestStatePool(
            self.model, config.hf_config.dtype, config.max_num_seqs + self.num_padding_slots, budget)
        config.prefix_cache_disabled_reason = (
            "固定请求状态没有前缀块快照，无法从共享 KV 前缀恢复。"
            if self.request_states.nbytes else None)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        kv_modules = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        # 固定状态在 warmup 前分配，已包含在 used 中。额外预留 warmup 瞬时峰值。
        budget = int(total * config.gpu_memory_utilization - used
                     - stats["allocated_bytes.all.peak"] + stats["allocated_bytes.all.current"])
        if not kv_modules:
            if not self.request_states.nbytes:
                raise ValueError("模型没有声明 KV cache 或固定请求状态")
            if budget < 0:
                raise ValueError("显存预算不足以容纳请求状态和 warmup 峰值")
            config.num_kvcache_blocks = 0
            self.kv_cache = None
            return
        cache_shapes = set()
        for module in kv_modules:
            # Attention 声明的是本 rank 实际写入的头数和头宽。
            num_kv_heads = getattr(module, "num_kv_heads", None)
            head_dim = getattr(module, "head_dim", None)
            if num_kv_heads is None:
                num_kv_heads = hf_config.num_key_value_heads // self.world_size
            if head_dim is None:
                head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
            cache_shapes.add((num_kv_heads, head_dim))
        if len(cache_shapes) != 1:
            raise ValueError(f"共享 KV 缓存要求各层的本地 KV 头数和头宽一致，实际为 {sorted(cache_shapes)}")
        num_kv_heads, head_dim = cache_shapes.pop()
        block_bytes = 2 * len(kv_modules) * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        num_blocks = budget // block_bytes
        if self.world_size > 1:
            # 调度器的物理块号在每个 rank 都必须可用。
            capacity = torch.tensor(num_blocks, dtype=torch.int64)
            dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
            num_blocks = capacity.item()
        if num_blocks < 1:
            raise ValueError(f"单个 KV 块需要 {block_bytes} 字节，显存预算只剩 {budget} 字节")
        if num_blocks == 1 and not self.enforce_eager:
            warnings.warn("显存仅容纳一个 KV 块，使用 eager 推理以保留请求页。")
            self.enforce_eager = True
        # 图的虚拟 decode 行共享末尾的只读零页。调度器只分配前面的请求页。
        config.num_kvcache_blocks = num_blocks if self.enforce_eager else num_blocks - 1
        self.kv_cache = torch.empty(2, len(kv_modules), num_blocks, self.block_size,
                                    num_kv_heads, head_dim, dtype=hf_config.dtype)
        if not self.enforce_eager:
            self.kv_cache[:, :, config.num_kvcache_blocks].zero_()
        for idx, module in enumerate(kv_modules):
            module.k_cache = self.kv_cache[0, idx]
            module.v_cache = self.kv_cache[1, idx]

    @staticmethod
    def _batch_tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).cuda(non_blocking=True)

    def prepare_request_slots(self, seqs, is_prefill):
        if not self.request_states.nbytes:
            return None
        slots = self._batch_tensor([seq.state_slot for seq in seqs], torch.int64)
        if is_prefill:
            fresh = [seq.state_slot for seq in seqs if seq.num_cached_tokens == 0]
            if fresh:
                fresh_slots = slots if len(fresh) == len(seqs) else self._batch_tensor(fresh, torch.int64)
                self.request_states.reset(fresh_slots)
        return slots

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = self._batch_tensor(block_tables, torch.int32)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup 或仅使用固定状态
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if seqs[0].block_table and cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = self._batch_tensor(input_ids, torch.int64)
        positions = self._batch_tensor(positions, torch.int64)
        cu_seqlens_q_list, cu_seqlens_k_list = cu_seqlens_q, cu_seqlens_k
        cu_seqlens_q = self._batch_tensor(cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._batch_tensor(cu_seqlens_k, torch.int32)
        slot_mapping = self._batch_tensor(slot_mapping, torch.int32)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    cu_seqlens_q_list=cu_seqlens_q_list, cu_seqlens_k_list=cu_seqlens_k_list,
                    request_slots=self.prepare_request_slots(seqs, True))
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            if seq.block_table:
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = self._batch_tensor(input_ids, torch.int64)
        positions = self._batch_tensor(positions, torch.int64)
        slot_mapping = self._batch_tensor(slot_mapping, torch.int32)
        context_lens_list = context_lens
        context_lens = self._batch_tensor(context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs) if seqs[0].block_table else None
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    context_lens_list=context_lens_list, request_slots=self.prepare_request_slots(seqs, False))
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = self._batch_tensor(temperatures, torch.float32)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            if context.block_tables is not None:
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if context.request_slots is not None:
                graph_vars["request_slots"][:bs] = context.request_slots
            if bs < graph_bs:
                padding = slice(bs, graph_bs)
                graph_vars["positions"][padding].zero_()
                graph_vars["input_ids"][padding].zero_()
                graph_vars["context_lens"][padding].fill_(1)
                if context.block_tables is not None:
                    graph_vars["block_tables"][padding].fill_(self.config.num_kvcache_blocks)
                if context.request_slots is not None:
                    padding_slots = graph_vars["padding_slots"][:graph_bs - bs]
                    self.request_states.reset(padding_slots)
                    graph_vars["request_slots"][padding] = padding_slots
            graph_vars["context_lens"][:bs] = context.context_lens
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.ones(max_bs, dtype=torch.int32)
        block_tables = torch.full((max_bs, max_num_blocks), config.num_kvcache_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        request_slots = padding_slots = None
        if self.request_states.nbytes:
            request_slots = torch.arange(max_bs, dtype=torch.int64)
            padding_slots = torch.arange(config.max_num_seqs, config.max_num_seqs + self.num_padding_slots,
                                         dtype=torch.int64)
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs] if config.num_kvcache_blocks else None,
                        request_slots=request_slots[:bs] if request_slots is not None else None)
            if request_slots is not None:
                self.request_states.reset(request_slots[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            if request_slots is not None:
                self.request_states.reset(request_slots[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            request_slots=request_slots,
            padding_slots=padding_slots,
        )
