import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


class RMSNormTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp32_preserves_residual(self):
        torch.manual_seed(2)
        norm = RMSNorm(16).cuda()
        x, residual = torch.randn(5, 16, device="cuda"), torch.randn(5, 16, device="cuda")
        original_x, original_residual = x.clone(), residual.clone()
        with torch.inference_mode():
            out = norm(x)
            expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(x, original_x)
            out, summed = norm(x, residual)
            expected_sum = original_x + original_residual
            expected = expected_sum * torch.rsqrt(expected_sum.square().mean(-1, keepdim=True) + norm.eps)
            torch.testing.assert_close(out, expected)
            torch.testing.assert_close(summed, expected_sum)
            torch.testing.assert_close(x, original_x)
            torch.testing.assert_close(residual, original_residual)


if __name__ == "__main__":
    unittest.main()
