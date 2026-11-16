import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _find_packed_rule(weight_name: str, mapping: dict) -> str | None:
    """找出 checkpoint 名字命中的 packed 规则；命中多条视为配置错误。"""
    hits = [key for key in mapping if key in weight_name]
    if len(hits) > 1:
        raise ValueError(
            f"权重名 {weight_name!r} 同时命中多条 packed_modules_mapping 规则 {hits}，无法判断走哪条"
        )
    return hits[0] if hits else None


def _get_parameter(model: nn.Module, name: str, weight_name: str, rule: str | None):
    try:
        return model.get_parameter(name)
    except AttributeError:
        if rule is not None:
            hint = f"（packed_modules_mapping 规则 {rule!r} 把它映射到了 {name!r}）"
        else:
            hint = "（模型里没有这个参数；如果它确实用不上，把名字加进模型的 ignored_weight_keys）"
        raise ValueError(f"checkpoint 里的 {weight_name!r} 找不到对应参数 {name!r}{hint}") from None


def load_model(model: nn.Module, path: str):
    """把 checkpoint 装进模型，装完核对两边是否一一对上。

    三种关系：同名直接装；packed_modules_mapping 声明的多对一按 shard 装；
    ignored_weight_keys 命中的直接跳过。装完要求模型的每个参数都被填过，
    否则抛错并列出是谁——静默的随机初始化比报错难查得多。
    """
    mapping = getattr(model, "packed_modules_mapping", {})
    ignored = tuple(getattr(model, "ignored_weight_keys", ()))
    expected = {name for name, _ in model.named_parameters()}
    loaded: set[str] = set()
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
                if rule is None:
                    param_name = weight_name
                    param = _get_parameter(model, param_name, weight_name, None)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                else:
                    target, shard_id = mapping[rule]
                    param_name = weight_name.replace(rule, target)
                    param = _get_parameter(model, param_name, weight_name, rule)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                loaded.add(param_name)

    missing = sorted(expected - loaded)
    if missing:
        preview = "、".join(missing[:10])
        if len(missing) > 10:
            preview += f" 等共 {len(missing)} 个"
        raise ValueError(
            f"以下参数没有被 checkpoint 填充，会保持随机初始化：{preview}。"
            f"通常是 packed_modules_mapping 少了规则，或模型多写了 checkpoint 里没有的参数。"
        )

    return {"loaded": len(loaded), "skipped": skipped}
