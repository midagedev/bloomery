#!/usr/bin/env python3
"""tools/ref/qwen4exp-nvfp4-oracle.py --resume continues only the run it paused, on the same cut generation.

Run with --self-test (also registered in check-recipes). torch, numpy, safetensors and transformers are stand-ins and the
cut is an empty first shard with a config, so no weight is read: a run that passes the resume check stops where it would
open its set writers. QWEN4EXP_NVFP4_ORACLE selects another copy of the oracle for a FAIL-first run.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ORACLE = Path(os.environ.get("QWEN4EXP_NVFP4_ORACLE", Path(__file__).with_name("qwen4exp-nvfp4-oracle.py")))


class Accepted(Exception):
    """The run passed the resume check and reached its set writers."""


class Capture:
    """A saved identity that records the identity the run compares it with, and differs from it."""

    def __ne__(self, other):
        self.seen = other
        return True


def stand_ins():
    """The oracle's third-party modules: enough for main() up to its set writers."""
    torch = mock.MagicMock(name="torch", __version__="stand-in")
    torch.nn.Module = object
    hf = ModuleType("modeling_qwen4_exp")
    hf.__file__ = str(ORACLE)
    configuration = ModuleType("configuration_qwen4_exp")
    configuration.Qwen4ExpTextConfig = lambda **kw: SimpleNamespace(**kw)
    transformers = mock.MagicMock(name="transformers", __version__="stand-in")
    transformers.models.qwen4_exp.modeling_qwen4_exp = hf
    safetensors = ModuleType("safetensors")
    safetensors.safe_open = lambda path, framework: SimpleNamespace(metadata=lambda: json.loads(Path(path).read_text()))
    return {
        "torch": torch,
        "numpy": mock.MagicMock(name="numpy"),
        "safetensors": safetensors,
        "transformers": transformers,
        "transformers.masking_utils": transformers.masking_utils,
        "transformers.models": transformers.models,
        "transformers.models.qwen4_exp": transformers.models.qwen4_exp,
        "transformers.models.qwen4_exp.modeling_qwen4_exp": hf,
        "transformers.models.qwen4_exp.configuration_qwen4_exp": configuration,
    }


class Resume(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, stand_ins())
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location("qwen4exp_nvfp4_oracle", ORACLE)
        self.oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.oracle)
        self.oracle.SetWriter = mock.Mock(side_effect=Accepted)
        tmp = tempfile.TemporaryDirectory(prefix="qwen4exp-nvfp4-oracle-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cut = self.root / "cut"
        self.cut.mkdir()
        meta = {"bloomery.fixture.version": "1", "bloomery.fixture.source": self.oracle.REPO,
                "bloomery.fixture.rev": self.oracle.MODEL_REV}
        (self.cut / "model-00001-of-00002.safetensors").write_text(json.dumps(meta))
        # The config a cut of layers [0, 1, 2, 3] and one of [0, 1, 6, 7] both write.
        (self.cut / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 4, "vocab_size": 64}}))
        self.prose = self.root / "prose.txt"
        self.prose.write_text(" ".join(str(i % 64) for i in range(3001)))
        self.out = self.root / "out"
        self.out.mkdir()
        (self.out / "resume.pt").write_bytes(b"")

    def resume(self, line, saved):
        """Resume the paused run on the cut whose `# fixture` line is `line`, with `saved` its state's identity."""
        self.oracle.torch.load = lambda path: {"identity": saved}
        argv = ["oracle", "--model", str(self.cut), "--out", str(self.out), "--prose", str(self.prose),
                "--fixture-line", line, "--resume"]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            self.oracle.main()

    def test_a_resume_continues_only_the_cut_generation_it_paused_on(self):
        first, regenerated = "# fixture\tlayers=0,1,2,3", "# fixture\tlayers=0,1,6,7"
        paused = Capture()
        with self.assertRaises(SystemExit) as captured:
            self.resume(first, paused)
        self.assertIn("resume.pt", str(captured.exception))
        with self.assertRaises(Accepted):
            self.resume(first, paused.seen)
        with self.assertRaises(SystemExit) as refused:
            self.resume(regenerated, paused.seen)
        self.assertIn("resume.pt", str(refused.exception))


def self_test():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Resume))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        print("usage: qwen4exp-nvfp4-oracle-test.py --self-test", file=sys.stderr)
        sys.exit(64)
    sys.exit(self_test())
