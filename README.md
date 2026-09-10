<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

从零实现的轻量级 vLLM。本 fork 增加了旧 NVIDIA GPU 的兼容路径，用于在当前工作机上运行推理并支持后续算子生成实验。

## 主要特性

- 🚀 **快速离线推理**：上游基准中的推理速度与 vLLM 接近。
- 📖 **代码易读**：上游核心实现约 1,200 行 Python 代码。
- ⚡ **推理优化**：支持前缀缓存、张量并行、Torch 编译和 CUDA graph 等。
- **旧 GPU 兼容**：提供 PyTorch 注意力兼容路径，无需安装 FlashAttention 即可运行。

## 安装

开发本 fork 时，在 `nano-vllm/` 目录中安装当前代码：

```bash
python -m pip install -e . --no-build-isolation
python -c "import nanovllm; print('ok')"
```

需要 Python 3.10–3.12 和可用的 PyTorch CUDA 环境。默认安装不会构建 FlashAttention。

如需安装上游原版，可执行以下命令；上游版本不包含本 fork 的兼容修改：

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## 注意力后端与旧 NVIDIA GPU

FlashAttention 是可选依赖。在支持的 GPU 上，可安装：

```bash
python -m pip install -e '.[flash]' --no-build-isolation
```

当扩展成功导入、CUDA 计算能力为 8.0 及以上，且数据类型为 FP16/BF16 时，使用原有 FlashAttention 路径。其他情况使用 PyTorch SDPA 的 math 后端，通过显式因果掩码和分页缓存读取计算注意力。

兼容路径支持普通预填充（prefill）、分块预填充、缓存前缀、分组查询注意力（GQA）和逐 token 解码（decode）。由于需要在 CPU 端读取序列元数据，该路径会关闭 CUDA graph。

在 GTX 1080 Ti（sm_61）等 Ampere 之前的 GPU 上：

- 模型使用 FP32，算子使用 eager 模式，KV cache 通过 PyTorch 写入，避免调用 Triton 编译器。
- RMSNorm 避免原地归一化 FP32 输入和残差，防止破坏残差连接；采样在 inference mode 中执行，确保原地操作合法。
- `example.py` 将上下文限制为 1024 个 token，单批最多处理 512 个 token、4 条序列。
- 示例关闭 Qwen3 thinking，将 256-token 输出预算用于最终回答。

这些设置优先保证正确性。使用其他入口时，应向 `LLM` 传入类似的上下文和批量限制；math attention 的计算量仍随序列长度呈平方增长。

## 模型下载

下载官方示例模型到默认路径：

```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

如果模型位于其他目录，请修改 `example.py` 或 `bench.py` 中的 `path`。

## 快速开始

运行完整示例：

```bash
python example.py
```

API 与 vLLM 接口相似，但 `LLM.generate` 的返回格式略有不同：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/你的模型路径",
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=1024,
    max_num_batched_tokens=512,
    max_num_seqs=4,
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["你好，请介绍一下自己。"]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

聊天模板的使用方式见 `example.py`。

## 正确性验证

在 `nano-vllm/` 目录中执行：

```bash
python -m unittest discover -s tests -v
# 可选的真实模型对照，也会在 CPU 上加载官方 Transformers 权重：
python tests/check_qwen3_reference.py
```

注意力测试将兼容路径输出与显式双精度注意力计算进行比较，覆盖 GQA、变长序列、不连续缓存块、带前缀或分块的预填充、解码、填充槽位和查询分块边界。有可用 GPU 时，还会执行 FP32 和 FP16 CUDA 测试。

FP32 RMSNorm 测试检查输入和残差是否保持正确。可选集成测试将真实模型的 logits 与 Transformers 对照，并重复运行跨越多个缓存块的提示词。这些测试不验证 FlashAttention 的性能。

## 性能基准

基准脚本为 `bench.py`，运行方式如下：

```bash
python bench.py
```

该脚本默认使用较大的批量和上下文，旧 GPU 上应先调整配置。以下为上游 README 提供的基准数据，不代表 GTX 1080 Ti 兼容路径的性能。

测试配置：

- 硬件：RTX 4070 Laptop（8GB）
- 模型：Qwen3-0.6B
- 请求总数：256 条序列
- 输入长度：在 100–1024 个 token 之间随机采样
- 输出长度：在 100–1024 个 token 之间随机采样

性能结果：

| 推理引擎 | 输出 token 数 | 耗时（秒） | 吞吐量（token/秒） |
|----------|---------------|------------|-------------------|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

## Star 历史

[![Star 数量历史图](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
