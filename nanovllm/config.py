import os
from dataclasses import dataclass, field
from transformers import AutoConfig, GenerationConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int | list[int] | None = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    master_port: int = 0  # 0 = 由引擎自动挑一个空闲端口
    prefix_cache_disabled_reason: str | None = field(default=None, init=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def resolve_eos(self, tokenizer_eos: int | None) -> None:
        eos = getattr(self.hf_config, "eos_token_id", None)
        if os.path.isfile(os.path.join(self.model, "generation_config.json")):
            generation = GenerationConfig.from_pretrained(self.model, local_files_only=True)
            if generation.eos_token_id is not None:
                eos = generation.eos_token_id
        self.eos = tokenizer_eos if eos is None else eos
