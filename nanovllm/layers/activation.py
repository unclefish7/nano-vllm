import torch
from torch import nn

from nanovllm.utils.compilation import compile_on_supported_gpu
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @compile_on_supported_gpu
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
