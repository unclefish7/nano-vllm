from functools import wraps

import torch


def compile_on_supported_gpu(fn):
    compiled = torch.compile(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        tensor = next((arg for arg in (*args, *kwargs.values()) if isinstance(arg, torch.Tensor)), None)
        if tensor is not None and tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 8:
            return fn(*args, **kwargs)
        return compiled(*args, **kwargs)

    return wrapper
