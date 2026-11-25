import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


@torch.no_grad()
def default_weight_loader(target: torch.Tensor, loaded_weight: torch.Tensor):
    if target.shape != loaded_weight.shape:
        raise ValueError(f"checkpoint 形状 {tuple(loaded_weight.shape)} 与目标形状 {tuple(target.shape)} 不符")
    target.copy_(loaded_weight)


def _find_packed_rule(weight_name: str, mapping: dict) -> str | None:
    """找出 checkpoint 名字命中的 packed 规则；命中多条视为配置错误。"""
    hits = [key for key in mapping if key in weight_name]
    if len(hits) > 1:
        raise ValueError(
            f"权重名 {weight_name!r} 同时命中多条 packed_modules_mapping 规则 {hits}，无法判断走哪条"
        )
    return hits[0] if hits else None


def _loadable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters(remove_duplicate=False))
    for prefix, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            if name not in module._non_persistent_buffers_set:
                tensors[f"{prefix}.{name}" if prefix else name] = buffer
    return tensors


def load_model(model: nn.Module, path: str):
    """把 checkpoint 装进模型，装完核对两边是否一一对上。

    三种关系：同名直接装；packed_modules_mapping 声明的多对一按 shard 装；
    ignored_weight_keys 命中的直接跳过。装完要求每个参数和持久缓冲区都被填过，
    否则抛错并列出是谁——静默的随机初始化比报错难查得多。
    """
    mapping = getattr(model, "packed_modules_mapping", {})
    ignored = tuple(getattr(model, "ignored_weight_keys", ()))
    targets = _loadable_tensors(model)
    loaded: set[int] = set()
    skipped: list[str] = []

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"{path} 下没有 *.safetensors 文件")

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if any(pattern in weight_name for pattern in ignored):
                    skipped.append(weight_name)
                    continue

                rule = _find_packed_rule(weight_name, mapping)
                name = weight_name if rule is None else weight_name.replace(rule, mapping[rule][0])
                if name not in targets:
                    hint = f"（packed_modules_mapping 规则 {rule!r}）" if rule else ""
                    raise ValueError(f"checkpoint 里的 {weight_name!r} 找不到对应参数或持久缓冲区 {name!r}{hint}")
                tensor = targets[name]
                if rule is None:
                    loader = getattr(tensor, "weight_loader", default_weight_loader)
                    loader(tensor, f.get_tensor(weight_name))
                else:
                    tensor.weight_loader(tensor, f.get_tensor(weight_name), mapping[rule][1])
                loaded.add(id(tensor))

    missing = sorted(name for name, tensor in targets.items() if id(tensor) not in loaded)
    if missing:
        preview = "、".join(missing[:10])
        if len(missing) > 10:
            preview += f" 等共 {len(missing)} 个"
        raise ValueError(
            f"以下参数或持久缓冲区没有被 checkpoint 填充：{preview}。"
            f"通常是 packed_modules_mapping 少了规则，或模型多写了 checkpoint 里没有的参数。"
        )

    return {"loaded": len(loaded), "skipped": skipped}
