"""可选的真实模型对照；需要已下载的 Qwen3-0.6B 和 CUDA。"""
import os
from unittest.mock import patch
import torch
from transformers import AutoModelForCausalLM
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

def main():
    path = os.path.expanduser('~/huggingface/Qwen3-0.6B')
    torch.manual_seed(0)
    llm = LLM(path, max_model_len=512, max_num_batched_tokens=128, max_num_seqs=2, gpu_memory_utilization=0.7)
    runner = llm.model_runner
    ids = llm.tokenizer.encode('The capital of France is')
    seq = Sequence(ids)
    seq.num_scheduled_tokens = len(ids)
    seq.block_table = [0]
    with torch.inference_mode():
        inputs, positions = runner.prepare_prefill([seq])
        actual = runner.run_model(inputs, positions, True).cpu()
        reset_context()
        reference = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation='eager').eval()
        expected = reference(torch.tensor([ids])).logits[:, -1, :]
        print('HF max absolute logit error:', (actual - expected).abs().max().item(), flush=True)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        print('HF top token matches:', actual.argmax(-1).item() == expected.argmax(-1).item(), flush=True)
        del reference
    prompt = llm.tokenizer.encode('A short story about a cat. ' * 40)
    assert 256 < len(prompt) < 480, len(prompt)
    params = SamplingParams(temperature=0.01, max_tokens=8, ignore_eos=True)
    # 使用确定性采样，避免缓存命中前后采样次数不同导致随机误报。
    with patch.object(runner.sampler, "forward", side_effect=lambda logits, temperatures: logits.argmax(dim=-1)):
        first = llm.generate([prompt], params, use_tqdm=False)
        second = llm.generate([prompt], params, use_tqdm=False)
    assert first[0]['token_ids'] == second[0]['token_ids'], (first, second)
    print('Chunked prefill + reused prefix match:', len(prompt), first[0]['text'], flush=True)


if __name__ == "__main__":
    main()
