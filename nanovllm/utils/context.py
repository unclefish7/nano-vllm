from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # 主机侧的同一份数据。prepare_* 本来就是从 Python 列表构造张量的，这里把列表
    # 一并带下来：参考实现（无 flash attention 时的 forward_torch）需要按请求切分
    # 长度，若逐层调用 .tolist() 会每层同步一次设备，TP 下这个同步还要等对端集合通信。
    cu_seqlens_q_list: list[int] | None = None
    cu_seqlens_k_list: list[int] | None = None
    context_lens_list: list[int] | None = None
    request_slots: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, *,
                cu_seqlens_q_list=None, cu_seqlens_k_list=None, context_lens_list=None, request_slots=None):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables,
        cu_seqlens_q_list, cu_seqlens_k_list, context_lens_list, request_slots,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
