"""模型注册表、契约检查与权重装载的单元测试（纯 CPU，不需要 GPU）。"""

import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from torch import nn

from nanovllm.models import ARCH, check_model_contract, resolve_model_class
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


class _Config:
    def __init__(self, architectures):
        self.architectures = architectures


class TestRegistry(unittest.TestCase):

    def test_qwen3_is_registered(self):
        self.assertIs(ARCH["Qwen3ForCausalLM"], Qwen3ForCausalLM)

    def test_resolves_qwen3(self):
        self.assertIs(resolve_model_class(_Config(["Qwen3ForCausalLM"])), Qwen3ForCausalLM)

    def test_unknown_architectures_reports_both_sides(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_model_class(_Config(["LlamaForCausalLM"]))
        message = str(ctx.exception)
        self.assertIn("LlamaForCausalLM", message)
        self.assertIn("Qwen3ForCausalLM", message)

    def test_empty_architectures(self):
        with self.assertRaises(ValueError):
            resolve_model_class(_Config([]))


class TestContract(unittest.TestCase):

    def test_qwen3_satisfies_contract(self):
        check_model_contract(Qwen3ForCausalLM)

    def test_missing_compute_logits(self):
        class Bad(nn.Module):
            def forward(self, input_ids, positions):
                return input_ids

        with self.assertRaises(TypeError) as ctx:
            check_model_contract(Bad)
        self.assertIn("compute_logits", str(ctx.exception))

    def test_wrong_forward_signature(self):
        class Bad(nn.Module):
            def forward(self, hidden_states):
                return hidden_states

            def compute_logits(self, hidden_states):
                return hidden_states

        with self.assertRaises(TypeError) as ctx:
            check_model_contract(Bad)
        self.assertIn("input_ids", str(ctx.exception))


class _TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(2, 3))
        self.b = nn.Parameter(torch.zeros(4))


class _PackedModel(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(6, 2))
        self.gate_up_proj.weight_loader = _packed_weight_loader


def _packed_weight_loader(param, loaded_weight, shard_id):
    half = param.shape[0] // 2
    param.data.narrow(0, int(shard_id) * half, half).copy_(loaded_weight)


class TestLoader(unittest.TestCase):
    def test_checkpoint_loads_persistent_buffer_and_preserves_computed_cache(self):
        model = nn.Module()
        model.layer = nn.Module()
        model.layer.register_buffer("scale", torch.tensor([1., 1.]))
        model.layer.register_buffer("lookup", torch.tensor([9]), persistent=False)
        self._save({"layer.scale": torch.tensor([2., 3.])})
        stats = load_model(model, self.tmp)
        self.assertEqual(stats["loaded"], 1)
        self.assertEqual(model.layer.scale.tolist(), [2., 3.])
        self.assertEqual(model.layer.lookup.tolist(), [9])
        self._save({"layer.lookup": torch.tensor([10])})
        with self.assertRaisesRegex(ValueError, "layer.lookup"):
            load_model(model, self.tmp)

    def test_missing_persistent_buffer_and_incorrect_shape_are_reported(self):
        model = _TinyModel()
        model.register_buffer("scale", torch.ones(2))
        weights = {"a": torch.zeros_like(model.a), "b": torch.zeros_like(model.b)}
        self._save(weights)
        with self.assertRaisesRegex(ValueError, "scale"):
            load_model(model, self.tmp)
        self._save({**weights, "scale": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "shape|形状"):
            load_model(model, self.tmp)


    def setUp(self):
        # 用系统临时目录（可用 TMPDIR 改写）；容器里源码目录是只读挂载，不能就地建目录。
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, tensors):
        save_file(tensors, os.path.join(self.tmp, "model.safetensors"))

    def test_loads_every_tensor(self):
        model = _TinyModel()
        self._save({"a": torch.ones(2, 3), "b": torch.arange(4.0)})
        stats = load_model(model, self.tmp)
        self.assertEqual(stats["loaded"], 2)
        self.assertTrue(torch.equal(model.a, torch.ones(2, 3)))
        self.assertTrue(torch.equal(model.b, torch.arange(4.0)))

    def test_unfilled_parameter_raises(self):
        model = _TinyModel()
        self._save({"a": torch.ones(2, 3)})
        with self.assertRaises(ValueError) as ctx:
            load_model(model, self.tmp)
        self.assertIn("b", str(ctx.exception))

    def test_unknown_checkpoint_key_raises(self):
        model = _TinyModel()
        self._save({"a": torch.ones(2, 3), "b": torch.zeros(4), "c": torch.zeros(1)})
        with self.assertRaises(ValueError) as ctx:
            load_model(model, self.tmp)
        self.assertIn("'c'", str(ctx.exception))

    def test_ignored_weight_keys_are_skipped(self):
        class WithIgnored(_TinyModel):
            ignored_weight_keys = ("rotary_emb.inv_freq",)

        model = WithIgnored()
        self._save({
            "a": torch.ones(2, 3),
            "b": torch.zeros(4),
            "layers.0.rotary_emb.inv_freq": torch.zeros(8),
        })
        stats = load_model(model, self.tmp)
        self.assertEqual(stats["skipped"], ["layers.0.rotary_emb.inv_freq"])

    def test_packed_mapping_fills_slices(self):
        model = _PackedModel()
        self._save({"gate_proj": torch.ones(3, 2), "up_proj": torch.full((3, 2), 2.0)})
        load_model(model, self.tmp)
        self.assertTrue(torch.equal(model.gate_up_proj[:3], torch.ones(3, 2)))
        self.assertTrue(torch.equal(model.gate_up_proj[3:], torch.full((3, 2), 2.0)))


if __name__ == "__main__":
    unittest.main()
