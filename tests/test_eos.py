import json
import tempfile
import unittest
from pathlib import Path

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestEndOfSequence(unittest.TestCase):
    def test_generation_settings_take_precedence_and_match_every_stop_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": 1}))
            (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [0, 7]}))
            config = Config(directory, num_kvcache_blocks=8)
            config.resolve_eos(1)
            for token, ignore, finished in [(0, False, True), (7, False, True),
                                            (1, False, False), (7, True, False)]:
                with self.subTest(token=token, ignore=ignore):
                    scheduler = Scheduler(config)
                    seq = Sequence([23], SamplingParams(max_tokens=2, ignore_eos=ignore))
                    scheduler.add(seq)
                    batch, prefill = scheduler.schedule()
                    scheduler.postprocess(batch, [token], prefill)
                    self.assertEqual(seq.is_finished, finished)
                    if not finished:
                        batch, prefill = scheduler.schedule()
                        scheduler.postprocess(batch, [23], prefill)
                        self.assertTrue(seq.is_finished)

    def test_fallback_to_single_model_or_tokenizer_stop_id(self):
        with tempfile.TemporaryDirectory() as directory:
            for model_eos, tokenizer_eos, expected in [(0, 7, {0}), (None, 7, {7}), (None, None, set())]:
                with self.subTest(model_eos=model_eos, tokenizer_eos=tokenizer_eos):
                    (Path(directory) / "config.json").write_text(json.dumps(
                        {"model_type": "llama", "eos_token_id": model_eos}))
                    config = Config(directory, num_kvcache_blocks=8)
                    config.resolve_eos(tokenizer_eos)
                    self.assertEqual(Scheduler(config).eos_token_ids, expected)
