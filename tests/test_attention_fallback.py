import unittest
from unittest.mock import patch

import torch

from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.utils.context import set_context, reset_context


class AttentionFallbackTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def check_attention(self, device, dtype):
        torch.manual_seed(7)
        heads, kv_heads, dim, block_size = 4, 2, 8, 4
        layer = Attention(heads, dim, 0.31, kv_heads)
        layer.k_cache = torch.full((6, block_size, kv_heads, dim), float('nan'), device=device, dtype=dtype)
        layer.v_cache = torch.full_like(layer.k_cache, float('nan'))
        # Noncontiguous physical blocks, padding, and a partially filled block.
        tables = torch.tensor([[3, 0, -1], [2, 4, -1]], device=device, dtype=torch.int32)
        keys = [torch.randn(n, kv_heads, dim, device=device, dtype=dtype) for n in (7, 6)]
        values = [torch.randn_like(k) for k in keys]

        def reference(q, k, v, offset):
            result = []
            for row in range(q.size(0)):
                per_head = []
                for head in range(heads):
                    kh = k[:offset + row + 1, head // 2].double()
                    vh = v[:offset + row + 1, head // 2].double()
                    probs = (kh @ q[row, head].double() * layer.scale).softmax(0)
                    per_head.append(probs @ vh)
                result.append(torch.stack(per_head))
            return torch.stack(result).to(dtype)

        for starts, ends, prefill in [([0, 0], [3, 2], True), ([3, 2], [6, 5], True), ([6, 5], [7, 6], False)]:
            qs = [torch.randn(end - start, heads, dim, device=device, dtype=dtype) for start, end in zip(starts, ends)]
            k = torch.cat([x[start:end] for x, start, end in zip(keys, starts, ends)])
            v = torch.cat([x[start:end] for x, start, end in zip(values, starts, ends)])
            slots = [int(tables[i, pos // block_size]) * block_size + pos % block_size for i in range(2) for pos in range(starts[i], ends[i])]
            cuq = [0, len(qs[0]), len(qs[0]) + len(qs[1])]
            cuk = [0, ends[0], sum(ends)]
            set_context(prefill,
                        cu_seqlens_q=torch.tensor(cuq, device=device),
                        cu_seqlens_k=torch.tensor(cuk, device=device),
                        slot_mapping=torch.tensor(slots, device=device),
                        context_lens=torch.tensor(ends, device=device),
                        block_tables=tables if any(starts) else None)
            with patch('nanovllm.layers.attention.can_use_flash_attention', return_value=False):
                actual = layer(torch.cat(qs), k, v)
            expected = torch.cat([reference(q, k, v, start) for q, k, v, start in zip(qs, keys, values, starts)])
            self.assertEqual(actual.shape, expected.shape if prefill else (2, 1, heads, dim))
            torch.testing.assert_close(actual.reshape_as(expected), expected, atol=2e-3 if dtype == torch.float16 else 2e-6, rtol=2e-3 if dtype == torch.float16 else 2e-5)
            for i, end in enumerate(ends):
                restored = layer.k_cache[tables[i, :2].long()].flatten(0, 1)[:end]
                torch.testing.assert_close(restored, keys[i][:end])
                restored_v = layer.v_cache[tables[i, :2].long()].flatten(0, 1)[:end]
                torch.testing.assert_close(restored_v, values[i][:end])

        # Graph padding sentinel must leave cache untouched.
        before = layer.k_cache.clone()
        store_kvcache(keys[0][:1], values[0][:1], layer.k_cache, layer.v_cache, torch.tensor([-1], device=device))
        torch.testing.assert_close(layer.k_cache, before, equal_nan=True)

    def test_cpu(self):
        self.check_attention('cpu', torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                self.check_attention('cuda', dtype)

    def test_warmup_chunk_boundary(self):
        torch.manual_seed(1)
        q = torch.randn(133, 2, 8)
        k, v = torch.randn(133, 1, 8), torch.randn(133, 1, 8)
        layer = Attention(2, 8, 0.4, 1)
        set_context(True, cu_seqlens_q=torch.tensor([0, 133]), cu_seqlens_k=torch.tensor([0, 133]))
        actual = layer(q, k, v)
        scores = torch.einsum('qhd,khd->hqk', q.double(), k.double().expand(-1, 2, -1)) * 0.4
        scores.masked_fill_(~torch.ones(133, 133, dtype=torch.bool).tril(), float('-inf'))
        expected = torch.einsum('hqk,khd->qhd', scores.softmax(-1), v.double().expand(-1, 2, -1)).float()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
