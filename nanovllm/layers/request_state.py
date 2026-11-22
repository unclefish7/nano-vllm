"""模型模块声明、读取和写回每请求固定形状的状态。"""

from dataclasses import dataclass
from math import prod

import torch
from torch import nn

from nanovllm.utils.context import get_context


@dataclass(frozen=True)
class RequestStateSpec:
    """shape 是单个请求在本 rank 的形状；dtype 默认取模型 dtype。"""

    shape: tuple[int, ...]
    dtype: torch.dtype | None = None

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("请求状态 shape 应为正整数元组，标量使用 ()")
        if self.dtype is not None and not isinstance(self.dtype, torch.dtype):
            raise TypeError("请求状态 dtype 应为 torch.dtype 或 None")


def read_request_state(module: nn.Module, name: str) -> torch.Tensor:
    """按当前请求顺序返回 [B, *shape] 的独立张量。"""
    return module._request_state_buffers[name].index_select(0, get_context().request_slots)


def write_request_state(module: nn.Module, name: str, value: torch.Tensor) -> None:
    """按当前请求顺序写回 [B, *shape]，要求 dtype 和设备与声明一致。"""
    module._request_state_buffers[name].index_copy_(0, get_context().request_slots, value)


class RequestStatePool:
    def __init__(self, model, model_dtype, num_rows, budget):
        declarations = []
        self.nbytes = 0
        self.buffers = []
        for path, module in model.named_modules():
            specs = getattr(module, "request_state_specs", {})
            if not isinstance(specs, dict):
                raise TypeError(f"{path}.request_state_specs 必须是 dict")
            for name, spec in specs.items():
                if not isinstance(name, str) or not name or not isinstance(spec, RequestStateSpec):
                    raise TypeError(f"{path}.request_state_specs 需要非空状态名和 RequestStateSpec")
                dtype = spec.dtype or model_dtype
                self.nbytes += num_rows * prod(spec.shape) * dtype.itemsize
                declarations.append((module, name, spec.shape, dtype))
        if self.nbytes and self.nbytes > budget:
            raise ValueError(f"请求状态需要 {self.nbytes} 字节，显存预算只剩 {budget} 字节")
        bindings = {}
        for module, name, shape, dtype in declarations:
            buf = torch.zeros((num_rows, *shape), dtype=dtype)
            bindings.setdefault(module, {})[name] = buf
            self.buffers.append(buf)
        for module, buffers in bindings.items():
            module._request_state_buffers = buffers

    def reset(self, slots):
        for buf in self.buffers:
            buf.index_fill_(0, slots, 0)
