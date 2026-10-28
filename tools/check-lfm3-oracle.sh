#!/usr/bin/env bash
# The lfm3 logits oracle's shell contracts, against copies of the scripts in temp directories with stub binaries: no box, no
# model, no card, no network (the fetch reads file:// URLs). It holds what tools/ref/models/lfm3.sh, tools/ref/fetch-lfm3.sh,
# tools/ref/dump-lfm3.sh, the lfm3 arm of tools/ref/dump-nemotron.sh and the lfm3 arm of tools/ref/build-nemotron-ref.sh say:
# a file that is not the pinned one, or a directory not at the pinned revision, ends the run by name before a dump starts; each
# file's dump names its own set and binary and the architecture the file must state; a failed dump ends the run and keeps the
# sets before it; a tree at another commit than the profile pins is refused before the harness is linked.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 - "$@" <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path.cwd()
REF = ROOT / "tools/ref"
# Every run below is waited on by its exit; this bound only catches a hang.
FAILSAFE = 120
LDD = (REF / "emb2/ldd-stub.sh").read_text()

# The harness a dump runs, as a shell script: it records its arguments and the build record it was given, fails when the
# model's name ends in $FIXTURE_FAIL, and writes the manifest a finished dump has (a row a --prompt, then the trailer).
STUB_REF = r'''#!/bin/sh
out=; model=; arch=; names=
while [ $# -gt 0 ]; do
  case $1 in
    -m) model=$2; shift 2 ;;
    --arch) arch=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --prompt) names="$names ${2%%:*}"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s|%s|%s|%s\n' "$model" "$arch" "$BLOOMERY_REF_BUILD" "$names" >> "$FIXTURE_CALLS"
if [ -n "${FIXTURE_FAIL:-}" ] && [ "${model%"$FIXTURE_FAIL"}" != "$model" ]; then exit 1; fi
mkdir -p "$out"
n=0
for p in $names; do
  n=$((n + 1))
  printf 'prompt\t%s\n' "$p" >> "$out/MANIFEST.tsv"
done
printf '# complete\t%s\t0\n' "$n" >> "$out/MANIFEST.tsv"
'''

STUB_TOKENIZE = '''#!/bin/sh
printf '['
seq -s', ' 1 5000 | tr -d '\\n'
printf ']\\n'
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class Box:
    """A temp root holding copies of the scripts, a profile whose files are the temp directory's, three stub GGUF files, a stub
    llama.cpp tree and the stub harness the dump runs."""

    FILES = (("ref-lfm3-logits", "q4.gguf", b"q4-bytes"), ("ref-lfm3-q8-logits", "q8.gguf", b"q8-bytes-longer"),
             ("ref-lfm3-bf16-logits", "bf16.gguf", b"bf16-bytes-longest"))

    def __init__(self, tmp):
        self.root = Path(tmp).resolve()
        self.models = self.root / "models"
        self.data = self.root / "data"
        self.tree = self.root / "llama.cpp-lfm3"
        self.served = self.root / "served"
        self.calls = self.root / "calls"
        ref = self.root / "tools/ref"
        for name in ("ref-paths.sh", "dspark-path.sh", "dump-nemotron.sh", "dump-lfm3.sh", "fetch-lfm3.sh",
                     "build-nemotron-ref.sh", "nemotron_ref.cpp", "ref_io.h"):
            ref.mkdir(parents=True, exist_ok=True)
            shutil.copy(REF / name, ref / name)
        profile = (REF / "models/lfm3.sh").read_text()
        entries = " ".join(f'"{s}:{self.models / f}:{len(b)}:{sha256(b)}"' for s, f, b in self.FILES)
        write(ref / "models/lfm3.sh", profile + f"LOGITS_SETS=({entries})\n")
        self.revision = next(l.split("=", 1)[1] for l in profile.splitlines() if l.startswith("LFM3_REVISION="))
        for _, f, b in self.FILES:
            self.models.mkdir(exist_ok=True)
            (self.models / f).write_bytes(b)
            self.served.mkdir(exist_ok=True)
            (self.served / f).write_bytes(b)
        bin_dir = self.tree / "build/bin"
        write(bin_dir / "llama-tokenize", STUB_TOKENIZE, True)
        write(bin_dir / "libllama.so", "library")
        write(self.data / "ref-tokenizer-nemotron/code.txt", "code\n" * 3000)
        write(self.data / "ref-tokenizer-nemotron/prose.txt", "prose\n" * 3000)
        self.bin = self.data / "bin/nemotron_ref_lfm3"
        write(self.bin, STUB_REF, True)
        cpp_io = (ref / "nemotron_ref.cpp").read_bytes() + (ref / "ref_io.h").read_bytes()
        md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
        self.build_id = f"llama-tokenize {md5(bin_dir / 'llama-tokenize')} libllama.so {md5(bin_dir / 'libllama.so')}"
        write(Path(f"{self.bin}.build"),
              f"source_sha256 {sha256(cpp_io)}\nlcpp_commit fixture\nbuild {self.build_id}\n")
        shim = self.root / "shim"
        write(shim / "ldd", LDD, True)
        self.env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "BLOOMERY_MODEL": "lfm3",
                    "BLOOMERY_DATA": str(self.data), "LCPP": str(self.tree), "FIXTURE_CALLS": str(self.calls)}
        for var in ("BLOOMERY_REF_MODEL", "BLOOMERY_REF_MODEL_PROFILE", "BLOOMERY_TIER", "BLOOMERY_FIXTURE_VARIANT"):
            self.env.pop(var, None)

    def revision_file(self, text=None):
        (self.models / "REVISION").write_text((self.revision if text is None else text) + "\n")

    def run(self, script, **env):
        result = subprocess.run(["bash", str(self.root / "tools/ref" / script)], cwd=self.root, env={**self.env, **env},
                                capture_output=True, text=True, timeout=FAILSAFE)
        return result.returncode, result.stdout + result.stderr

    def recorded(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lfm3-fetch-")
        self.addCleanup(self.tmp.cleanup)
        self.box = Box(self.tmp.name)
        # The files are served from a directory and absent from the models directory.
        self.url = {"LFM3_BASE_URL": f"file://{self.box.served}"}
        for _, f, _ in Box.FILES:
            (self.box.models / f).unlink()

    def test_downloads_checks_and_writes_the_revision_last(self):
        rc, out = self.box.run("fetch-lfm3.sh", **self.url)
        self.assertEqual(rc, 0, out)
        for _, f, b in Box.FILES:
            self.assertEqual((self.box.models / f).read_bytes(), b)
            self.assertFalse((self.box.models / f"{f}.part").exists())
        self.assertEqual((self.box.models / "REVISION").read_text(), self.box.revision + "\n")
        rc, out = self.box.run("fetch-lfm3.sh", **self.url)
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count("is the pinned file already"), 3, out)

    def test_a_file_that_is_not_the_pinned_one_ends_the_run_and_clears_the_revision(self):
        self.box.revision_file()
        (self.box.served / "q8.gguf").write_bytes(b"q8-bytes-LONGER")  # the table's size, other bytes
        rc, out = self.box.run("fetch-lfm3.sh", **self.url)
        self.assertEqual(rc, 65, out)
        self.assertIn("q8.gguf is 15 bytes", out)
        self.assertTrue((self.box.models / "q4.gguf").exists(), "the file before it stays")
        self.assertFalse((self.box.models / "q8.gguf").exists(), out)
        self.assertFalse((self.box.models / "q8.gguf.part").exists(), out)
        self.assertFalse((self.box.models / "REVISION").exists(), "a directory that failed has no revision")
        self.assertFalse((self.box.models / "bf16.gguf").exists(), "the run ended at the bad file")


class Dump(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lfm3-dump-")
        self.addCleanup(self.tmp.cleanup)
        self.box = Box(self.tmp.name)
        self.box.revision_file()

    def refused(self, why):
        rc, out = self.box.run("dump-lfm3.sh")
        self.assertEqual(rc, 66, out)
        self.assertIn(why, out)
        self.assertEqual(self.box.recorded(), [], "a refused directory starts no dump")
        self.assertFalse(self.box.data.joinpath("ref-lfm3-logits").exists(), out)

    def test_a_missing_file_is_named_before_any_dump(self):
        (self.box.models / "q8.gguf").unlink()
        self.refused("no " + str(self.box.models / "q8.gguf"))

    def test_a_file_of_another_size_is_named_before_any_dump(self):
        (self.box.models / "bf16.gguf").write_bytes(b"short")
        self.refused("bf16.gguf is 5 bytes")

    def test_a_directory_with_no_revision_is_named_before_any_dump(self):
        (self.box.models / "REVISION").unlink()
        self.refused("names 'nothing'")

    def test_a_directory_at_another_revision_is_named_before_any_dump(self):
        self.box.revision_file("0" * 40)
        self.refused("names '" + "0" * 40 + "'")

    def test_each_file_has_its_set_its_binary_and_the_architecture_it_must_state(self):
        rc, out = self.box.run("dump-lfm3.sh")
        self.assertEqual(rc, 0, out)
        self.assertEqual([c.split("|")[:3] for c in self.box.recorded()],
                         [[str(self.box.models / f), "lfm3", self.box.build_id] for _, f, _ in Box.FILES])
        for set_name, _, _ in Box.FILES:
            manifest = (self.box.data / set_name / "MANIFEST.tsv").read_text()
            self.assertEqual(manifest.count("prompt\t"), 10, set_name)
            self.assertIn("# complete\t10\t0", manifest)
            self.assertFalse((self.box.data / f"{set_name}.staging").exists(), set_name)
        names = self.box.recorded()[0].split("|")[3].split()
        self.assertEqual(names, "code5 code29 code4096 prose1 prose64 prose255 prose256 prose257 prose512 prose1024".split())

    def test_a_failed_dump_ends_the_run_and_keeps_the_sets_before_it(self):
        rc, out = self.box.run("dump-lfm3.sh", FIXTURE_FAIL="q8.gguf")
        self.assertEqual(rc, 1, out)
        self.assertIn("ref-lfm3-q8-logits from", out)
        self.assertTrue((self.box.data / "ref-lfm3-logits/MANIFEST.tsv").exists(), out)
        self.assertFalse((self.box.data / "ref-lfm3-q8-logits").exists(), out)
        self.assertFalse((self.box.data / "ref-lfm3-q8-logits.staging").exists(), "an incomplete stage is removed")
        self.assertEqual(len(self.box.recorded()), 2, "the dump of the third file never started")

    def test_the_dump_script_refuses_a_file_the_profile_names_no_set_for(self):
        other = self.box.models / "other.gguf"
        other.write_bytes(b"other")
        rc, out = self.box.run("dump-nemotron.sh", BLOOMERY_REF_MODEL=str(other))
        self.assertEqual(rc, 2, out)
        self.assertIn("is none of the lfm3 profile's LOGITS_SETS files", out)
        self.assertEqual(self.box.recorded(), [], out)

    def test_the_binary_must_be_the_one_built_against_the_live_libraries(self):
        (self.box.tree / "build/bin/libllama.so").write_text("a rebuilt library")
        rc, out = self.box.run("dump-lfm3.sh")
        self.assertEqual(rc, 2, out)
        self.assertIn("differs from the build", out)
        self.assertEqual(self.box.recorded(), [], out)


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lfm3-build-")
        self.addCleanup(self.tmp.cleanup)
        self.box = Box(self.tmp.name)
        git = ["git", "-C", str(self.box.tree)]
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(self.box.tree)], check=True, env=env)
        subprocess.run([*git, "commit", "--allow-empty", "-qm", "fixture"], check=True, env=env)
        self.head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, env=env).strip()

    def test_a_tree_at_another_commit_than_the_profile_pins_is_refused_before_the_link(self):
        rc, out = self.box.run("build-nemotron-ref.sh")
        self.assertEqual(rc, 2, out)
        self.assertIn(f"is at {self.head}, the lfm3 profile pins 19cce6919992fdffd92e0ac152caaf1e2b8f254e", out)
        self.assertFalse((self.box.data / "bin/nemotron_ref_lfm3.tmp").exists(), out)

    def test_a_tree_at_the_pinned_commit_goes_on_to_the_tokenizer_it_needs(self):
        (self.box.tree / "build/bin/llama-tokenize").unlink()
        profile = self.box.root / "tools/ref/models/lfm3.sh"
        profile.write_text(profile.read_text() + f"LCPP_COMMIT={self.head}\n")
        rc, out = self.box.run("build-nemotron-ref.sh")
        self.assertEqual(rc, 66, out)
        self.assertIn("no llama-tokenize at", out)


unittest.main(argv=["check-lfm3-oracle"], verbosity=2)
PY
