"""模型注册表与契约检查。

引擎手里只有 config.json，需要据此决定用哪个模型类（找到它），
并在构造之前确认这个类满足引擎的基本要求（验它）。

新增一个模型只需要两步：写 nanovllm/models/<model>.py，在 ARCH 里登记一行。
"""

import inspect
import os

from torch import nn

from nanovllm.models.qwen3 import Qwen3ForCausalLM

# config.architectures -> 模型类。键是 HF config.json 里的 architectures 字符串。
ARCH: dict[str, type[nn.Module]] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}


def register_candidate_file(path: str) -> str:
    """以本地模型类的名称登记候选，返回文件中第一个入口的名称。

    引擎用 spawn 起其他 rank，子进程会重新 import 本模块。只在父进程里改
    ARCH 是传不过去的，所以这里提供一个“从文件重放登记”的入口，
    由环境变量 NANOVLLM_MODEL_FILE 在子进程导入时自动触发。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("nanovllm_candidate_plugin", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法把 {path} 当成 Python 模块加载")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = []
    for attr, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and attr == obj.__qualname__
            and issubclass(obj, nn.Module)
            and callable(getattr(obj, "compute_logits", None))
        ):
            ARCH[attr] = obj
            names.append(attr)
    if not names:
        raise ValueError(f"{path} 里没有定义提供 compute_logits 的模型类")
    return names[0]


_candidate_file = os.environ.get("NANOVLLM_MODEL_FILE")
if _candidate_file:
    register_candidate_file(_candidate_file)


def resolve_model_class(hf_config) -> type[nn.Module]:
    """按 config.architectures 找到对应的模型类。"""
    architectures = list(getattr(hf_config, "architectures", None) or [])
    for name in architectures:
        if name in ARCH:
            return ARCH[name]
    raise ValueError(
        f"config.architectures={architectures} 里没有任何一个已注册的模型类；"
        f"已注册的有 {sorted(ARCH)}。"
        f"新模型请在 nanovllm/models/__init__.py 的 ARCH 里登记。"
    )


def check_model_contract(cls) -> None:
    """构造之前检查模型类是否满足引擎契约；不满足就把缺什么一次说清楚。"""
    problems = []

    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        raise TypeError(f"{cls!r} 不是 nn.Module 的子类")

    forward = getattr(cls, "forward", None)
    if not callable(forward):
        problems.append("缺少 forward 方法")
    else:
        names = [p for p in inspect.signature(forward).parameters if p != "self"]
        if names[:2] != ["input_ids", "positions"]:
            problems.append(
                f"forward 的前两个参数必须是 (input_ids, positions)，实际是 {tuple(names[:2])}"
            )

    if not callable(getattr(cls, "compute_logits", None)):
        problems.append("缺少 compute_logits(hidden_states) 方法")

    mapping = getattr(cls, "packed_modules_mapping", {})
    if not isinstance(mapping, dict):
        problems.append(f"packed_modules_mapping 必须是 dict，实际是 {type(mapping).__name__}")

    if problems:
        raise TypeError(f"模型类 {cls.__name__} 不满足引擎契约：" + "；".join(problems))
