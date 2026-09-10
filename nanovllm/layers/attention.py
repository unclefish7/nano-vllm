import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except (ImportError, OSError):
    flash_attn_varlen_func = flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


def can_use_flash_attention(device, dtype):
    return (
        flash_attn_varlen_func is not None
        and torch.device(device).type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and dtype in (torch.float16, torch.bfloat16)
    )


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    if key.device.type != "cuda" or torch.cuda.get_device_capability(key.device)[0] < 8:
        valid = slot_mapping >= 0
        slots = slot_mapping[valid].long()
        k_cache.view(-1, num_heads, head_dim)[slots] = key[valid]
        v_cache.view(-1, num_heads, head_dim)[slots] = value[valid]
        return
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if not can_use_flash_attention(q.device, q.dtype):
            return self.forward_torch(q, k, v)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def forward_torch(self, q, k, v):
        # Eager only: lengths/block tables are read on the host per batch.
        context = get_context()
        if context.is_prefill:
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            lengths = [b - a for a, b in zip(k_offsets, k_offsets[1:])]
        else:
            lengths = context.context_lens.tolist()
            q_offsets = list(range(len(lengths) + 1))
        outputs = []
        for i, length in enumerate(lengths):
            qi = q[q_offsets[i]:q_offsets[i + 1]]
            if context.block_tables is not None:
                block_size = self.k_cache.size(1)
                blocks = context.block_tables[i, :(length + block_size - 1) // block_size].long()
                ki = self.k_cache[blocks].flatten(0, 1)[:length]
                vi = self.v_cache[blocks].flatten(0, 1)[:length]
            else:
                ki = k[k_offsets[i]:k_offsets[i + 1]]
                vi = v[k_offsets[i]:k_offsets[i + 1]]
            # GQA: each KV head serves a consecutive group of query heads.
            repeats = self.num_heads // self.num_kv_heads
            ki = ki.repeat_interleave(repeats, dim=1).transpose(0, 1)
            vi = vi.repeat_interleave(repeats, dim=1).transpose(0, 1)
            # Bound math-backend score memory even for long prefills.
            for start in range(0, qi.size(0), 128):
                chunk = qi[start:start + 128]
                positions = torch.arange(start, start + chunk.size(0), device=q.device) + length - qi.size(0)
                mask = torch.arange(length, device=q.device)[None, :] <= positions[:, None]
                with sdpa_kernel(SDPBackend.MATH):
                    out = F.scaled_dot_product_attention(
                        chunk.transpose(0, 1), ki, vi,
                        attn_mask=mask, dropout_p=0.0, scale=self.scale,
                    )
                outputs.append(out.transpose(0, 1))
        out = torch.cat(outputs, dim=0)
        return out if context.is_prefill else out.unsqueeze(1)
