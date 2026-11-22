"""用 CPU 张量验证请求状态、缓存预算及调度生命周期。"""

import pickle
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


class History(nn.Module):
    def __init__(self):
        super().__init__()
        from nanovllm.layers.request_state import RequestStateSpec
        self.request_state_specs = {
            "window": RequestStateSpec((2,)),
            "count": RequestStateSpec((), torch.int64),
        }

    def forward(self, tokens, positions):
        from nanovllm.layers.request_state import read_request_state, write_request_state
        window = read_request_state(self, "window")
        count = read_request_state(self, "count")
        ctx = get_context()
        if ctx.is_prefill:
            rows = []
            offsets = ctx.cu_seqlens_q_list
            for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
                values = torch.cat((window[i], tokens[start:end]))
                rows.append(values[-2:])
                count[i] += end - start
            window = torch.stack(rows)
        else:
            window = torch.stack((window[:, -1], tokens), dim=1)
            count = count + 1
        write_request_state(self, "window", window)
        write_request_state(self, "count", count)
        return torch.cat((window, count[:, None]), dim=1)

    def compute_logits(self, hidden):
        return hidden


class HybridHistory(History):
    """零查询/键使注意力输出恰为前缀均值，可从原始词元独立重算。"""

    def __init__(self):
        super().__init__()
        self.attn = Attention(1, 3, 1, 1)

    def forward(self, tokens, positions):
        history = super().forward(tokens, positions)
        values = tokens.float()[:, None, None].repeat(1, 1, 3)
        zeros = torch.zeros_like(values)
        means = self.attn(zeros, zeros, values).reshape(-1, 3)
        ctx = get_context()
        if ctx.is_prefill:
            means = means[ctx.cu_seqlens_q[1:] - 1]
        return torch.cat((history, means[:, :1]), dim=-1)


class RequestStateTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(reset_context)
        self.capacity = 65536
        self.used = 0
        self.peak_extra = 0
        self.enter_patch(torch.cuda, "mem_get_info",
                         side_effect=lambda: (self.capacity - self.used, self.capacity))
        self.enter_patch(torch.cuda, "memory_stats", side_effect=lambda: {
            "allocated_bytes.all.peak": self.used + self.peak_extra,
            "allocated_bytes.all.current": self.used,
        })

    def enter_patch(self, obj, name, **kwargs):
        p = patch.object(obj, name, **kwargs)
        self.addCleanup(p.stop)
        return p.start()

    def runner(self, model, slots=2, chunk=2, eager=True):
        r = ModelRunner.__new__(ModelRunner)
        r.rank = 0
        r.world_size = 1
        r.block_size = 2
        r.enforce_eager = eager
        r.model = model
        r.config = SimpleNamespace(
            hf_config=SimpleNamespace(dtype=torch.float32, num_hidden_layers=16,
                                      hidden_size=3, num_attention_heads=1, num_key_value_heads=1),
            max_num_seqs=slots, max_num_batched_tokens=chunk, max_model_len=12,
            eos=-1, gpu_memory_utilization=1.0, kvcache_block_size=r.block_size,
        )
        # 只替代 H2D 传输和显存计量，状态池、调度器、前向均实际执行。
        r._batch_tensor = lambda values, dtype: torch.tensor(values, dtype=dtype)
        r.allocate_request_states()
        self.used += r.request_states.nbytes
        r.allocate_kv_cache()
        return r

    def test_actual_kv_modules_and_state_bytes(self):
        layers = nn.ModuleList([History() for _ in range(10)] +
                               [Attention(1, 3, 1.0, 1) for _ in range(6)])
        self.peak_extra = 256
        r = self.runner(layers)
        self.assertEqual(r.request_states.nbytes, 2 * 10 * (2 * 4 + 8))
        expected = (self.capacity - self.used - self.peak_extra) // (2 * 6 * 2 * 3 * 4)
        self.assertEqual(r.kv_cache.shape, (2, 6, expected, 2, 1, 3))
        self.assertEqual(r.config.num_kvcache_blocks, expected)
        self.assertTrue(r.config.prefix_cache_disabled_reason)
        for index, layer in enumerate(layers[10:]):
            self.assertEqual(layer.v_cache.data_ptr(), r.kv_cache[1, index].data_ptr())
        from nanovllm.layers.request_state import read_request_state, write_request_state
        set_context(False, request_slots=torch.tensor([1, 0]))
        write_request_state(layers[0], "window", torch.ones(2, 2))
        self.assertEqual(read_request_state(layers[1], "window").tolist(), [[0, 0], [0, 0]])
        self.assertEqual(read_request_state(layers[0], "count").dtype, torch.int64)

    def test_single_kv_page_stays_available_for_requests(self):
        self.capacity = 48  # 只够一页时仍可进行 eager 推理。
        with self.assertWarnsRegex(UserWarning, "eager"):
            r = self.runner(Attention(1, 3, 1, 1), eager=False)
        self.assertTrue(r.enforce_eager)
        self.assertEqual(r.config.num_kvcache_blocks, 1)
        self.assertEqual(r.kv_cache.size(2), 1)

    def test_budget_rejects_fixed_states_or_insufficient_kv_space(self):
        self.capacity = 30
        with self.assertRaisesRegex(ValueError, "状态"):
            self.runner(History())
        self.capacity = 64
        with self.assertRaisesRegex(ValueError, "KV"):
            self.runner(nn.ModuleList([History(), Attention(1, 3, 1.0, 1)]))

    def test_prefix_hits_only_for_pure_kv(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), patch.object(Sequence, "block_size", 2):
                modules = [Attention(1, 3, 1.0, 1)]
                if hybrid:
                    modules.append(History())
                r = self.runner(nn.ModuleList(modules), chunk=6)
                sched = Scheduler(r.config)
                for repeat in range(2):
                    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
                    sched.add(seq)
                    batch, prefill = sched.schedule()
                    expected = 2 if repeat and not hybrid else 0
                    self.assertEqual(seq.num_cached_tokens, expected)
                    sched.postprocess(batch, [9], prefill)
                self.assertEqual(bool(sched.block_manager.hash_to_block_id), not hybrid)
                self.assertEqual(bool(r.config.prefix_cache_disabled_reason), hybrid)

    def step(self, runner, scheduler, output, reverse=False):
        batch, prefill = scheduler.schedule()
        if reverse:
            batch = list(reversed(batch))
        # 模拟 TP 工作进程按相同的槽位和历史执行。
        wire = pickle.loads(pickle.dumps(batch))
        inputs = runner.prepare_prefill(wire) if prefill else runner.prepare_decode(wire)
        actual = runner.run_model(*inputs, prefill).tolist()
        for seq, row in zip(batch, actual):
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            self.assertEqual(row, ([0] + seq.token_ids[:end])[-2:] + [end])
        scheduler.postprocess(batch, output, prefill)
        return batch

    def test_state_only_reorder_chunk_preempt_finish_and_recycle(self):
        r = self.runner(History(), chunk=3)
        self.assertEqual(r.config.num_kvcache_blocks, 0)
        scheduler = Scheduler(r.config)
        a = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=3))
        b = Sequence([6], SamplingParams(max_tokens=3))
        c = Sequence([8], SamplingParams(max_tokens=1))
        for seq in (a, b, c):
            scheduler.add(seq)
        self.assertEqual(self.step(r, scheduler, [9]), [a])
        slot = a.state_slot
        self.assertEqual(self.step(r, scheduler, [5, 7]), [a, b])
        self.assertEqual(a.state_slot, slot)
        self.assertEqual(self.step(r, scheduler, [8, 9], reverse=True), [b, a])
        scheduler.running.remove(a)
        scheduler.preempt(a)
        self.assertIsNone(a.state_slot)
        self.assertEqual(a.num_cached_tokens, 0)
        self.assertEqual(self.step(r, scheduler, [0]), [a])
        self.assertEqual(self.step(r, scheduler, [6]), [a])
        self.assertIsNone(a.state_slot)
        self.assertEqual(self.step(r, scheduler, [1]), [c])
        self.assertEqual(self.step(r, scheduler, [0]), [b])
        self.assertTrue(scheduler.is_finished())

    def test_hybrid_attention_and_window_match_full_history_after_automatic_preemption(self):
        # 两个固定槽、三个 KV 块。第二条请求首次 decode 时必定被抢占。
        self.capacity = 2 * 16 + 3 * 48
        with patch.object(Sequence, "block_size", 2):
            r = self.runner(HybridHistory(), chunk=2)
            scheduler = Scheduler(r.config)
            self.assertEqual(r.config.num_kvcache_blocks, 3)
            first = Sequence([1, 2, 3], SamplingParams(max_tokens=3))
            second = Sequence([7, 8], SamplingParams(max_tokens=3))
            for seq in (first, second):
                scheduler.add(seq)
            starts = []
            for _ in range(16):
                batch, prefill = scheduler.schedule()
                inputs = r.prepare_prefill(batch) if prefill else r.prepare_decode(batch)
                actual = r.run_model(*inputs, prefill)
                for seq, row in zip(batch, actual):
                    end = seq.num_cached_tokens + seq.num_scheduled_tokens
                    prefix = seq.token_ids[:end]
                    expected = ([0] + prefix)[-2:] + [end, sum(prefix) / end]
                    torch.testing.assert_close(row, torch.tensor(expected))
                    if seq is second and prefill:
                        starts.append(seq.num_cached_tokens)
                scheduler.postprocess(batch, [9] * len(batch), prefill)
                if scheduler.is_finished():
                    break
            self.assertTrue(scheduler.is_finished())
            self.assertEqual(starts, [0, 0, 2])
            self.assertIsNone(first.state_slot)
            self.assertIsNone(second.state_slot)
            self.assertEqual(sorted(scheduler.free_state_slots), [0, 1])

    def test_warmup_uses_independent_initialized_rows(self):
        r = self.runner(History(), slots=3, chunk=6)
        r.config.max_model_len = 2
        r.sampler = lambda logits, temperatures: logits[:, -1].long()
        for name in ("empty_cache", "reset_peak_memory_stats"):
            self.enter_patch(torch.cuda, name)
        seen = []
        hook = r.model.register_forward_hook(lambda m, args, out: seen.append(out.tolist()))
        self.addCleanup(hook.remove)
        r.warmup_model()
        self.assertEqual(seen, [[[0, 0, 2]] * 3])
        s = Scheduler(r.config)
        s.add(Sequence([4], SamplingParams(max_tokens=1)))
        self.step(r, s, [0])

    def test_graph_capture_and_padded_replay_isolate_live_rows(self):
        r = self.runner(History(), slots=5, chunk=10, eager=False)
        self.enter_patch(torch.cuda, "CUDAGraph")
        self.enter_patch(torch.cuda, "graph", side_effect=lambda *args: nullcontext())
        self.enter_patch(torch.cuda, "synchronize")
        seen = []
        def check_window_position(module, args):
            ctx = get_context()
            if not ctx.is_prefill:
                # 窗口通过 context_lens - 1 选取当前位置，负索引会直接报错。
                positions = torch.arange(r.config.max_model_len)
                selected = positions.index_select(0, ctx.context_lens.long() - 1)
                torch.testing.assert_close(selected, args[1])
                seen.append((ctx, *args))

        hook = r.model.register_forward_pre_hook(check_window_position)
        self.addCleanup(hook.remove)
        from nanovllm.layers.request_state import read_request_state
        capture_counts = []
        counter_hook = r.model.register_forward_pre_hook(
            lambda m, args: capture_counts.append(read_request_state(m, "count").tolist()))
        r.capture_cudagraph()
        counter_hook.remove()
        self.assertTrue(all(all(c == 0 for c in counts) for counts in capture_counts), capture_counts)
        self.assertEqual(r.graph_bs, [1, 2, 4, 5])
        ctx, ids, positions = next(t for t in seen if t[1].numel() == 4)
        scheduler = Scheduler(r.config)
        for i in range(5):
            scheduler.add(Sequence([i + 10], SamplingParams(max_tokens=2)))
        self.step(r, scheduler, [1, 2, 3, 4, 5])

        def replay():
            from nanovllm.utils import context
            with patch.object(context, "_CONTEXT", ctx):
                self.assertEqual(read_request_state(r.model, "count")[-1].item(), 0)
                r.graph_vars["outputs"][:4].copy_(r.model(ids, positions))

        r.graphs[4].replay.side_effect = replay
        batch, _ = scheduler.schedule()
        for order in ((4, 0, 2), (1, 3, 0)):
            selected = [batch[i] for i in order]
            inputs = r.prepare_decode(selected)
            actual = r.run_model(*inputs, False)
            expected = [[i + 10, i + 1, 2] if i != 0 or order[0] == 4
                        else [1, 1, 3] for i in order]
            self.assertEqual(actual.tolist(), expected)

    def test_padded_decode_attention_reads_an_isolated_zero_page(self):
        class DecodeAttention(Attention):
            def forward(self, tokens, positions):
                values = tokens.float()[:, None, None].repeat(1, 1, 3)
                zeros = torch.zeros_like(values)
                return super().forward(zeros, zeros, values).reshape(-1, 3)

            def compute_logits(self, hidden):
                return hidden

        self.enter_patch(torch.cuda, "CUDAGraph")
        self.enter_patch(torch.cuda, "graph", side_effect=lambda *args: nullcontext())
        self.enter_patch(torch.cuda, "synchronize")
        r = self.runner(DecodeAttention(1, 3, 1, 1), slots=4, eager=False)
        live_pages = r.kv_cache[:, :, :r.config.num_kvcache_blocks]
        live_pages.fill_(3)
        captured = {}

        def check_capture(module, args, output):
            ctx = get_context()
            bs = len(args[0])
            self.assertEqual(ctx.context_lens.tolist(), [1] * bs)
            self.assertEqual(ctx.slot_mapping.tolist(), [-1] * bs)
            self.assertTrue((ctx.block_tables[:, 0] >= r.config.num_kvcache_blocks).all())
            torch.testing.assert_close(output, torch.zeros_like(output))
            captured[bs] = (ctx, *args)

        with r.model.register_forward_hook(check_capture):
            r.capture_cudagraph()
        self.assertTrue((live_pages == 3).all())
        ctx, ids, positions = captured[4]

        def replay():
            from nanovllm.utils import context
            with patch.object(context, "_CONTEXT", ctx):
                r.graph_vars["outputs"][:4].copy_(r.model(ids, positions))

        r.graphs[4].replay.side_effect = replay
        with patch.object(Sequence, "block_size", 2):
            seqs = [Sequence([3, 11 + i]) for i in range(4)]
            for i, seq in enumerate(seqs):
                seq.block_table = [i]
            # 先把第 4 行指向真实页，再把它作为填充行重放同一张图。
            r.run_model(*r.prepare_decode(seqs), False)
            before = r.kv_cache.clone()
            actual = r.run_model(*r.prepare_decode(seqs[:3]), False)
        self.assertEqual(ctx.context_lens.tolist(), [2, 2, 2, 1])
        self.assertEqual(ctx.slot_mapping.tolist(), [1, 3, 5, -1])
        self.assertEqual(positions.tolist(), [1, 1, 1, 0])
        self.assertEqual(ids[-1].item(), 0)
        self.assertGreaterEqual(ctx.block_tables[-1, 0].item(), r.config.num_kvcache_blocks)
        torch.testing.assert_close(r.graph_vars["outputs"][3], torch.zeros(3))
        torch.testing.assert_close(actual[:, 0], torch.tensor([7., 7.5, 8.]))
        torch.testing.assert_close(r.kv_cache, before)
