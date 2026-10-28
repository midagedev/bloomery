#!/usr/bin/env bash
# EmbeddingGemma writer contracts, with local shell fixtures and no model or card.
# --dumper PATH also checks the linked dumper's thread option before it opens a model.
# No argument: the shell contracts; a missing Linux utility is a named skip.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 - "$@" <<'PY'
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--dumper", type=Path)
args = parser.parse_args()
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
DUMPER = args.dumper.resolve() if args.dumper else None
LDD = (ROOT / "tools/ref/emb2/ldd-stub.sh").read_text()
# Every run below is waited on by its pid, and every wait for a writer's step by its file or its exit; this bound only
# catches a hang: a bound a run must fit in reads the machine's load, not the contract.
FAILSAFE = 120
GGUF_STUB = ("#include <cstdint>\nstruct gguf_context;\nstruct gguf_init_params { bool no_alloc; void * ctx; };\n"
             "static gguf_context * gguf_init_from_file(const char *, gguf_init_params) { return nullptr; }\n"
             "static int64_t gguf_find_key(const gguf_context *, const char *) { return -1; }\n"
             'static const char * gguf_get_val_str(const gguf_context *, int64_t) { return ""; }\n'
             "static void gguf_free(gguf_context *) {}\n")


def write(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class Writer(unittest.TestCase):
    def setUp(self):
        for name in ("flock", "timeout", "sha256sum", "readlink"):
            if not shutil.which(name):
                self.skipTest(f"{name} is required for the Linux writer fixtures")
        self.root = Path(tempfile.mkdtemp(prefix="bloomery-emb2-")).resolve()
        self.addCleanup(shutil.rmtree, self.root)
        for name in ("tools/ref/emb2/emb2.sh", "tools/ref/emb2/oracle-tree.sh", "tools/gate-bound.sh",
                     "tools/ref/emb2/dump_emb2.cpp", "tools/ref/clefvis/set_writer.h"):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.data = self.root / "data"
        self.data.mkdir()
        self.final = self.data / "ref_fixture_tower.cpu"
        self.oracle = self.root / "oracle.tree"
        self.oracle.mkdir()
        self.pin = "a" * 40
        model, projector = self.root / "text.gguf", self.root / "projector.gguf"
        model.write_bytes(b"text fixture")
        projector.write_bytes(b"projector fixture")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        variables = {
            "DATA": self.data, "MODEL": model, "MMPROJ": projector,
            "MODEL_SHA256": digest(model), "MMPROJ_SHA256": digest(projector),
            "LCPP_PIN": self.pin, "MODEL_NAME": "gemma-embedding2",
            "ORACLE": self.oracle, "BIN": self.root / "dump", "FAMILY": "emb2",
            "DUMPER": "tools/ref/emb2/dump_emb2.cpp", "SETPFX": "ref_fixture", "THREADS": "1",
            "EMBED_VOCAB": "fixture", "EMBED_CASES": "cases-fixture.txt", "EMBED_CONTEXT": "8192",
        }
        write(self.root / "tools/ref/ref-paths.sh", "\n".join(
            f"{key}={shlex.quote(str(value))}" for key, value in variables.items()) + "\n")
        write(self.root / "tools/ref/clefvis/profile.sh", "IMAGES=()\nEMBED_STEPS=(text tower server text.cpu tower.cpu)\n")
        write(self.oracle / "build/bin/llama-server", f"#!/bin/bash\necho 'version: fixture (commit {self.pin})'\n", True)
        write(self.oracle / "build/bin/libllama.so", "")
        write(self.root / "bin/ldd", LDD, True)
        self.flock = shutil.which("flock")
        write(self.root / "bin/flock", "#!/bin/bash\n"
              'if [ "$1" = -w ]; then touch "$FIXTURE_ATTEMPT/$FIXTURE_TAG"; fi\n'
              + "exec " + shlex.quote(self.flock) + ' "$@"\n', True)
        (self.root / "attempts").mkdir()
        write(self.root / "bin/git", """#!/bin/bash
case " $* " in
  *' rev-parse HEAD '*) printf '%s\\n' "$FIXTURE_PIN" ;;
  *' status '*) exit "${FIXTURE_STATUS_RC:-0}" ;;
  *' ls-files '*) exit 0 ;;
  *) exit 91 ;;
esac
""", True)
        write(self.root / "dump", """#!/bin/bash
set -eu
out=
while [ "$#" -gt 0 ]; do
  if [ "$1" = --out ]; then out=$2; shift; fi
  shift
done
printf '%s:start\\n' "$FIXTURE_TAG" >> "$FIXTURE_EVENTS"
if [ -n "${FIXTURE_RELEASE:-}" ]; then
  for _ in {1..200}; do
    [ ! -e "$FIXTURE_RELEASE" ] || break
    sleep 0.02
  done
  [ -e "$FIXTURE_RELEASE" ] || exit 92
fi
[ "${FIXTURE_FAIL:-0}" = 0 ] || exit 42
mkdir -p "$out"
printf '%s\\n' "$FIXTURE_TAG" > "$out/MANIFEST.tsv"
printf '%s:end\\n' "$FIXTURE_TAG" >> "$FIXTURE_EVENTS"
""", True)
        record = [
            "source_sha256 " + digest(self.root / "tools/ref/emb2/dump_emb2.cpp"),
            "set_writer_sha256 " + digest(self.root / "tools/ref/clefvis/set_writer.h"),
            "lcpp_commit " + self.pin[:9], "lcpp " + str(self.oracle.resolve()),
        ]
        write(self.root / "dump.build", "\n".join(record) + "\n")
        self.env = dict(os.environ, PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
                        EMB2_BOUND=str(FAILSAFE), FIXTURE_ATTEMPT=str(self.root / "attempts"), FIXTURE_PIN=self.pin, FIXTURE_EVENTS=str(self.root / "events"))
        self.jobs = []
        self.addCleanup(self.stop_jobs)

    def stop_jobs(self):
        for process in self.jobs:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=FAILSAFE)

    def start(self, tag, **extra):
        env = dict(self.env, FIXTURE_TAG=tag, **extra)
        process = subprocess.Popen(["bash", "tools/ref/emb2/emb2.sh", "tower.cpu"],
                                   cwd=self.root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        self.jobs.append(process)
        return process

    def finish(self, process, rc=0):
        output, _ = process.communicate(timeout=FAILSAFE)
        self.assertEqual(process.returncode, rc, output)
        return output

    def wait_for(self, path, process, what):
        """Until PATH exists: a writer that ends first fails naming its rc and output, as does the failsafe."""
        end = time.monotonic() + FAILSAFE
        while not path.exists():
            if process.poll() is not None:
                self.fail(f"{what}: the writer ended rc {process.returncode} first\n{process.communicate()[0]}")
            if time.monotonic() > end:
                self.fail(f"{what}: not within the {FAILSAFE} s failsafe")
            time.sleep(0.02)

    def wait_attempt(self, tag, process):
        self.wait_for(self.root / "attempts" / tag, process, f"{tag} did not attempt the set lock")

    def wait_started(self, process):
        self.wait_for(self.root / "events", process, "the first writer did not reach generation")

    def test_text_only_profile_refuses_tower_and_joined_steps_before_the_runner(self):
        env = os.environ.copy()
        env.pop("BLOOMERY_MODEL", None)
        env["BLOOMERY_REF_MODEL_PROFILE"] = "gemma-embedding"
        for step in ("tower", "text server"):
            with self.subTest(step=step):
                result = subprocess.run(["bash", str(ROOT / "tools/ref/emb2/emb2.sh"), step],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn(f"unknown step '{step}'", result.stderr)

    def test_lock_precedes_recovery_and_staging(self):
        old = Path(str(self.final) + ".old")
        stage = Path(str(self.final) + ".staging")
        write(old / "MANIFEST.tsv", "old\n")
        write(stage / "sentinel", "staging\n")
        with Path(str(self.final) + ".lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            process = self.start("new")
            self.wait_attempt("new", process)
            time.sleep(0.1)
            self.assertIsNone(process.poll())
            self.assertFalse(self.final.exists())
            self.assertEqual((old / "MANIFEST.tsv").read_text(), "old\n")
            self.assertEqual((stage / "sentinel").read_text(), "staging\n")
            self.assertFalse((self.root / "events").exists())
            fcntl.flock(held, fcntl.LOCK_UN)
        self.finish(process)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "new\n")
        self.assertFalse(old.exists())

    def test_contended_lock_names_the_set_and_returns_75(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        with Path(str(self.final) + ".lock").open("w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            output = self.finish(self.start("blocked", EMB2_BOUND="1"), 75)
        self.assertIn("set lock for " + str(self.final), output)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        self.assertFalse((self.root / "events").exists())

    def test_two_writers_generate_and_publish_in_order(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        release = self.root / "release"
        first = self.start("first", FIXTURE_RELEASE=str(release))
        self.wait_started(first)
        second = self.start("second")
        self.wait_attempt("second", second)
        time.sleep(0.1)
        self.assertEqual((self.root / "events").read_text(), "first:start\n")
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        release.touch()
        self.finish(first)
        self.finish(second)
        self.assertEqual((self.root / "events").read_text(),
                         "first:start\nfirst:end\nsecond:start\nsecond:end\n")
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "second\n")
        self.assertFalse(Path(str(self.final) + ".staging").exists())
        self.assertFalse(Path(str(self.final) + ".old").exists())

    def test_failed_writer_keeps_the_set_and_releases_the_lock(self):
        write(self.final / "MANIFEST.tsv", "old\n")
        self.finish(self.start("failed", FIXTURE_FAIL="1"), 42)
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "old\n")
        self.finish(self.start("next"))
        self.assertEqual((self.final / "MANIFEST.tsv").read_text(), "next\n")

    def test_failed_status_check_stops_before_generation(self):
        output = self.finish(self.start("failed-status", FIXTURE_STATUS_RC="128"), 2)
        self.assertIn("cannot inspect tracked changes in " + str(self.oracle), output)
        self.assertFalse((self.root / "events").exists())

    def test_build_record_from_another_tree_is_refused(self):
        record = self.root / "dump.build"
        record.write_text(record.read_text().replace("lcpp " + str(self.oracle),
                                                     "lcpp " + str(self.oracle).replace("oracle.tree", "oracleXtree")))
        output = self.finish(self.start("wrong-tree"), 2)
        self.assertIn("no 'lcpp " + str(self.oracle), output)
        self.assertFalse((self.root / "events").exists())


class Builders(unittest.TestCase):
    PIN = "a" * 40
    STAGES = ["build", "mel", "compile", "gpu", "cpu", "check-gpu", "check-cpu"]

    def audio_root(self, tmp, failure, server_commit):
        """A tree holding audio.sh, its oracle tree (stub binaries and libraries) and every stub the script calls."""
        root = Path(tmp).resolve()
        for name in ("tools/ref/emb2/audio.sh", "tools/ref/emb2/oracle-tree.sh", "tools/host-gate.sh", "tools/gate-bound.sh"):
            write(root / name, (ROOT / name).read_text())
        shutil.copytree(ROOT / "tools/bloomery", root / "tools/bloomery")
        from bloomery.manifest import dump_file_name
        mel_dir = root / "data/ref_gemma_embedding2_mel"
        rows = []
        for case in ("minimum/0", "sine/0", "silence/0", "before-30s/0", "after-30s/0", "after-30s/1"):
            name = case + "/mel"
            write(mel_dir / dump_file_name(name, 0, "output", "contig", "f32"), "\0" * (511 if failure == "size" else 512))
            rows.append(f"tensor\t{name}\t0\tf32\t1\t{127 if failure == 'shape' else 128}\t1\t1\t512\t0\tNONE\t1\t0\t-\t-")
        write(mel_dir / "MANIFEST.tsv", "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n" + "\n".join(rows) + "\n# complete\t6\t0\n")
        projector = root / "projector.gguf"
        projector.write_bytes(b"audio fixture")
        write(root / "tools/ref/models/gemma-embedding2.sh",
              f'LCPP={shlex.quote(str(root / "oracle"))}\nLCPP_PIN={self.PIN}\n'
              f'MMPROJ={shlex.quote(str(projector))}\n'
              f'MMPROJ_SHA256={hashlib.sha256(projector.read_bytes()).hexdigest()}\n')
        bin_dir = root / "oracle/build/bin"
        write(bin_dir / "llama-server", f"#!/bin/bash\necho 'version: 0.6.0-dev (build 1, commit {server_commit})'\n", True)
        write(root / "bin/ldd", LDD, True)
        for lib in ("libmtmd.so", "libllama.so", "libggml.so", "libggml-base.so"):
            write(bin_dir / lib, "")
        write(root / "bin/git", '#!/bin/bash\ncase " $* " in\n'
              '  *" rev-parse HEAD "*) echo "${FIXTURE_HEAD:-' + self.PIN + '}" ;;\n'
              '  *" status "*) exit 0 ;;\n  *" ls-files "*) exit 0 ;;\n  *) exit 91 ;;\nesac\n', True)
        # The script builds refset-check with cargo, then runs it through tools/host-gate.sh; the stub binary is what that build leaves.
        write(root / "bin/cargo", '#!/bin/bash\n'
              '[ "$*" = "build --release -p bloomery-refset --bin refset-check" ] || exit 91\n'
              'printf "build\\n" >> "$FIXTURE_EVENTS"\n'
              '[ "$FIXTURE_FAIL" != build ] || exit 42\n'
              'mkdir -p target/release && cp "$FIXTURE_ROOT/bin/refset-check" target/release/refset-check\n', True)
        write(root / "bin/refset-check", '#!/bin/bash\n'
              'case "$1" in\n'
              '  mel-gemma4a) stage=mel ;;\n'
              '  tower-gemma4a) stage=check-gpu ;;\n'
              '  tower-gemma4a-cpu) stage=check-cpu ;;\n'
              '  *) exit 91 ;;\nesac\n'
              'printf "%s\\n" "$stage" >> "$FIXTURE_EVENTS"\n'
              '[ "$FIXTURE_FAIL" != "$stage" ] || exit 42\n', True)
        # No build step may touch the oracle tree: a call of cmake, make or ninja is an event the expected list lacks.
        for tool in ("cmake", "make", "ninja"):
            write(root / "bin" / tool, f'#!/bin/bash\nprintf "{tool}\\n" >> "$FIXTURE_EVENTS"\nexit 91\n', True)
        write(root / "bin/dump-audio", '#!/bin/bash\n'
              'printf "%s\\n" "$3" >> "$FIXTURE_EVENTS"\n'
              '[ "$FIXTURE_FAIL" != "$3" ] || exit 42\n'
              '[ ! -e "$2" ] || exit 91\nmkdir "$2"\nprintf "complete\\n" > "$2/MANIFEST.tsv"\n', True)
        write(root / "bin/c++", '#!/bin/bash\n'
              'printf "compile\\n" >> "$FIXTURE_EVENTS"\n'
              '[ "$FIXTURE_FAIL" != compile ] || exit 42\n'
              'cp "$FIXTURE_ROOT/bin/dump-audio" target/dump_audio\n', True)
        return root

    def run_audio(self, root, failure, **extra):
        env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                   BLOOMERY_DATA=str(root / "data"), CXX="c++", FIXTURE_ROOT=str(root),
                   FIXTURE_EVENTS=str(root / "events"), FIXTURE_FAIL=failure, **extra)
        result = subprocess.run(["bash", str(root / "tools/ref/emb2/audio.sh")], env=env, capture_output=True,
                                text=True, timeout=FAILSAFE)
        events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
        return result.returncode, result.stdout + result.stderr, events

    def test_audio_reuses_the_oracle_tree_and_checks_the_mel_family(self):
        if sys.platform != "linux" or not shutil.which("sha256sum") or not shutil.which("timeout"):
            self.skipTest("audio builder fixture requires Linux, sha256sum and timeout")
        for failure in ("", "mel", "build", "compile", "gpu", "cpu", "check-gpu", "check-cpu", "shape", "size"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="audio-build-") as tmp:
                root = self.audio_root(tmp, failure, self.PIN[:9])
                rc, output, events = self.run_audio(root, failure)
                self.assertEqual(rc, (1 if failure in ("shape", "size") else 42) if failure else 0, output)
                expected = self.STAGES
                if failure in ("shape", "size"):
                    expected = expected[:3]
                    self.assertIn("malformed mel row or file", output)
                elif failure:
                    expected = expected[:expected.index(failure) + 1]
                self.assertEqual(events, expected, output)
                for suffix in ("", ".cpu"):
                    self.assertEqual((root / ("data/ref_gemma_embedding2_audio" + suffix)).exists(), not bool(failure))

    def test_audio_refuses_an_oracle_that_is_missing_or_another_build_before_anything_runs(self):
        if sys.platform != "linux" or not shutil.which("sha256sum") or not shutil.which("timeout"):
            self.skipTest("audio builder fixture requires Linux, sha256sum and timeout")
        refusals = {
            "no-server": lambda root: f"audio: no {root}/oracle/build/bin/llama-server",
            "rebuilt-without-git": lambda root: f"audio: {root}/oracle/build/bin/llama-server is not built at aaaaaaa: version: 0.6.0-dev (build 1, commit unknown)",
            "no-library": lambda root: f"audio: no {root}/oracle/build/bin/libggml-base.so",
            "other-commit": lambda root: f"audio: {root}/oracle is at {'b' * 40}, the profile pins {self.PIN}",
        }
        for name, message in refusals.items():
            with self.subTest(refusal=name), tempfile.TemporaryDirectory(prefix="audio-oracle-") as tmp:
                root = self.audio_root(tmp, "", "unknown" if name == "rebuilt-without-git" else self.PIN[:9])
                extra = {}
                if name == "no-server":
                    (root / "oracle/build/bin/llama-server").unlink()
                if name == "no-library":
                    (root / "oracle/build/bin/libggml-base.so").unlink()
                if name == "other-commit":
                    extra["FIXTURE_HEAD"] = "b" * 40
                rc, output, events = self.run_audio(root, "", **extra)
                self.assertEqual(rc, 2, output)
                self.assertIn(message(root), output)
                self.assertEqual(events, [], output)
                self.assertFalse((root / "data/ref_gemma_embedding2_audio").exists())

    def test_unreadable_index_stops_before_compilation(self):
        scripts = (
            ("tools/ref/build-hidden.sh", "qwen35"),
            ("tools/ref/build-nemotron-ref.sh", "nemotron_h"),
            ("tools/ref/build-nemotron-ref.sh", "lfm3"),
            ("tools/ref/clefvis/build-clefvis.sh", "qwen35"),
            ("tools/ref/qvis/build-lcpp-qvis.sh", "qwen35"),
            ("tools/ref/vision/build-fork.sh", "deepseek41"),
            ("tools/ref/build-ik-cand.sh", "deepseek41"),
            ("tools/ref/build-ik-mimo2.sh", "mimo2"),
            ("tools/ref/build-dump-draft.sh", "deepseek41"),
            ("tools/ref/build-dump-mtp.sh", "glm5next"),
        )
        for script, profile in scripts:
            with self.subTest(script=script), tempfile.TemporaryDirectory(prefix="oracle-status-") as tmp:
                root = Path(tmp).resolve()
                target = root / script
                target.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / script, target)
                oracle = root / "oracle"
                subprocess.run(["git", "init", "-q", str(oracle)], check=True)
                if script == "tools/ref/build-dump-draft.sh":
                    if not shutil.which("sha256sum"):
                        self.skipTest("sha256sum is required for the draft builder fixture")
                    write(oracle / "src/fixture", "original\n")
                    subprocess.run(["git", "-C", str(oracle), "add", "src/fixture"], check=True)
                subprocess.run(["git", "-C", str(oracle), "-c", "user.name=fixture",
                                "-c", "user.email=fixture@example.invalid", "commit",
                                "--allow-empty", "-qm", "fixture"], check=True)
                pin = subprocess.check_output(["git", "-C", str(oracle), "rev-parse", "HEAD"],
                                              text=True).strip()
                if script in ("tools/ref/build-dump-draft.sh", "tools/ref/build-dump-mtp.sh"):
                    subprocess.run(["git", "-C", str(oracle), "config", "status.relativePaths", "invalid"],
                                   check=True)
                    write(root / "tools/ref/ref-build-common.sh",
                          f'REF_BIN={shlex.quote(str(root / "bin"))}\nHERE={shlex.quote(str(root))}\n')
                    write(oracle / "build/CMakeCache.txt", "")
                    if script == "tools/ref/build-dump-draft.sh":
                        patch = root / "patch.py"
                        write(patch, 'ROOT = ""\nfrom pathlib import Path\nPath(ROOT + "fixture").write_text("patched\\n")\n')
                        source = target.read_text().replace("BASE=/home/user/ik-idxkey", f"BASE={shlex.quote(str(oracle))}")
                        source = source.replace("BASE_SHA=db517b69", f"BASE_SHA={pin}")
                        source = source.replace("PATCH=/home/user/ik-dsv41-draft.py", f"PATCH={shlex.quote(str(patch))}")
                        source = source.replace("PATCH_SHA256=4eeb56aa1d75b470b556bab87ab90a467c8e4942b3269c9c15a3c78db0b1bc60",
                                                "PATCH_SHA256=" + hashlib.sha256(patch.read_bytes()).hexdigest())
                        source = source.replace("PATCHED=(src/llama-hparams.h src/llama-hparams.cpp src/llama-load-tensors.cpp src/graphs/build_deepseek4.cpp)",
                                                "PATCHED=(src/fixture)")
                        target.write_text(source)
                else:
                    (oracle / ".git/index").write_bytes(b"invalid index")
                write(root / "tools/ref/ref-paths.sh",
                      f"MODEL_NAME={profile}\nLCPP={shlex.quote(str(oracle))}\n"
                      f"IK=$LCPP\nCAND_IK=$LCPP\nGLM_MTP_IK=$LCPP\nGLM_MTP_SHA={pin[:8]}\nREF_BIN_NAME=bin-fixture\n"
                      f"BLOOMERY_DATA={shlex.quote(str(root / 'data'))}\n")
                write(root / "tools/ref/clefvis/profile.sh",
                      'ORACLE=$LCPP\nDATA=$BLOOMERY_DATA\nDUMPER=fixture.cpp\n')
                write(root / "tools/ref/lease.sh", 'lease_free() { return 0; }\nnow() { printf fixture; }\n'
                      'lease_take() { return 0; }\nlease_bounded() { shift; "$@"; }\n')
                marker = root / "compiled"
                for compiler in ("g++", "cmake"):
                    write(root / "cuda/bin" / compiler,
                          '#!/bin/bash\n: > "$FIXTURE_COMPILED"\nexit 88\n', True)
                # Exercise source integrity without qvis' root-only user switch.
                write(root / "cuda/bin/id", "#!/bin/bash\necho 501\n", True)
                write(root / "cuda/bin/sudo", '#!/bin/bash\nshift 5\nexec "$@"\n', True)
                if script in ("tools/ref/vision/build-fork.sh", "tools/ref/build-ik-cand.sh",
                              "tools/ref/build-ik-mimo2.sh"):
                    write(root / "tools/ref/lease-probe.sh", 'lease_free() { return 0; }\nnow() { printf fixture; }\n')
                    source = root / "vision/src"
                    source.mkdir(parents=True)
                    (source / ".git").mkdir()
                    # Commit pins and the root-only user switch are fixture boundaries; status uses real Git.
                    write(root / "cuda/bin/git", f'#!{sys.executable}\n'
                          'import subprocess, sys\n'
                          'args = sys.argv[1:]\n'
                          'if args[:1] == ["-C"]: args = args[2:]\n'
                          f'if args[:1] == ["status"]: sys.exit(subprocess.call([{shutil.which("git")!r}, "-C", {str(oracle)!r}] + args))\n'
                          'if args[:1] == ["rev-parse"]: print("cfd8adcf6fb86ca523ad72fe21e8d2ac3860cd8e")\n', True)
                    write(root / "cuda/bin/find", '#!/bin/bash\nexit 0\n', True)
                    write(root / "shell-env", 'cd() { if [ "$1" = /home/user ]; then builtin cd "$FIXTURE_HOME"; else builtin cd "$@"; fi; }\n')
                env = dict(os.environ, PATH=str(root / "cuda/bin") + os.pathsep + os.environ["PATH"],
                           LCPP_BUILD_TREE=str(oracle), LCPP_BUILD_PIN=pin,
                           LCPP_BUILD_DIR=str(root / "run"), LCPP_BUILD_CUDA=str(root / "cuda"),
                           FIXTURE_COMPILED=str(marker), FIXTURE_HOME=str(root),
                           BASH_ENV=str(root / "shell-env"), VISREF_FORK_ROOT=str(root / "vision"),
                           DRAFT_IK=str(oracle))
                result = subprocess.run(["bash", str(target)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=FAILSAFE)
                output = result.stdout + result.stderr
                if script == "tools/ref/qvis/build-lcpp-qvis.sh":
                    output += (root / "run/log").read_text()
                    self.assertEqual((root / "run/rc").read_text().strip(), "2", output)
                self.assertEqual(result.returncode, 2, output)
                self.assertIn("cannot inspect tracked changes", output)
                self.assertFalse(marker.exists(), output)


class AudioDumper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cxx = shutil.which("c++") or shutil.which("g++")
        if not cxx:
            raise RuntimeError("a C++ compiler is required for the audio backend checks")
        cls.tmp = Path(tempfile.mkdtemp(prefix="bloomery-audio-dumper-"))
        write(cls.tmp / "clip.h", r'''
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
struct ggml_backend_device {};
using ggml_backend_dev_t = ggml_backend_device *;
enum { GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU };
inline ggml_backend_dev_t ggml_backend_dev_by_type(int type) {
    static ggml_backend_device gpu;
    const char * value = getenv(type == GGML_BACKEND_DEVICE_TYPE_GPU ? "FIXTURE_GPU" : "FIXTURE_IGPU");
    return value && !strcmp(value, "1") ? &gpu : nullptr;
}
inline void ggml_backend_load_all() {}
enum { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_MUL, GGML_TYPE_F32 };
struct ggml_tensor {
    int op = 0, type = GGML_TYPE_F32;
    const char * name = "";
    ggml_tensor * src[2] = {};
    int64_t ne[4] = {1, 1, 1, 1};
};
inline bool ggml_is_contiguous(const ggml_tensor *) { return true; }
inline int64_t ggml_nelements(const ggml_tensor *) { return 1; }
inline void ggml_backend_tensor_get(const ggml_tensor *, void * data, size_t, size_t size) { memset(data, 0, size); }
struct clip_context_params {
    bool use_gpu = false, warmup = false;
    ggml_backend_dev_t device = nullptr;
    bool (*cb_eval)(ggml_tensor *, bool, void *) = nullptr;
    void * cb_eval_user_data = nullptr;
};
struct clip_ctx { clip_context_params params; };
struct clip_init_result { clip_ctx * ctx_a = nullptr, * ctx_v = nullptr, * ctx_gen_a = nullptr; };
inline clip_init_result clip_init(const char *, clip_context_params p) {
    if (p.use_gpu && p.device && getenv("FIXTURE_GPU_INIT_FAIL")) return {};
    return {new clip_ctx{p}, nullptr, nullptr};
}
inline void clip_free(clip_ctx * ctx) { delete ctx; }
struct clip_image_f32 { void set_size(std::pair<int, int>, bool, bool) {} void cpy_buf(const std::vector<float> &) {} };
struct clip_image_f32_batch { bool is_audio; std::vector<clip_image_f32> entries; };
inline bool clip_image_batch_encode(clip_ctx * ctx, int, clip_image_f32_batch *, std::vector<float> &) {
    auto p = ctx->params;
    if (!p.cb_eval) return true;
    for (int i = 0; i < 16; ++i) {
        char name[64];
        ggml_tensor source, t;
        t.src[0] = &source;
        if (i == 0) { source.name = "a.input_projection.weight"; t.op = GGML_OP_MUL_MAT; }
        else if (i == 1) { source.name = "a.pre_encode.out.bias"; t.src[1] = &source; t.op = GGML_OP_ADD; }
        else if (i < 14) {
            snprintf(name, sizeof(name), "a.blk.%d.ln2.weight", i - 2);
            source.name = name; t.src[1] = &source; t.op = GGML_OP_MUL;
        } else { source.name = i == 14 ? "pos_emb" : "kq_mask"; }
        if (p.cb_eval(&t, true, p.cb_eval_user_data)) p.cb_eval(&t, false, p.cb_eval_user_data);
    }
    return true;
}
''' + GGUF_STUB)
        for header in ("clip-impl.h", "ggml-backend.h"):
            write(cls.tmp / header, '#include "clip.h"\n')
        cls.bin = cls.tmp / "dump_audio"
        subprocess.run([cxx, "-std=c++17", "-include", "cstdio", "-I", str(cls.tmp),
                        str(ROOT / "tools/ref/emb2/dump_audio.cpp"), "-o", str(cls.bin)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_backend_selection_and_malformed_inputs_never_complete_a_set(self):
        import struct
        cases = [
            ("gpu", {}, "case/0", b"\0" * 512, "visible GPU backend"),
            ("cpu", {}, "case/0", b"\0" * 512, None),
            ("gpu", {"FIXTURE_GPU": "1"}, "case/0", b"\0" * 512, None),
            ("gpu", {"FIXTURE_IGPU": "1"}, "case/0", b"\0" * 512, None),
            ("gpu", {"FIXTURE_GPU": "1", "FIXTURE_GPU_INIT_FAIL": "1"}, "case/0", b"\0" * 512, "projector did not open"),
            ("other", {}, "case/0", b"\0" * 512, "device must be"),
            ("cpu", {"BLOOMERY_REF_BUILD": ""}, "case/0", b"\0" * 512, "oracle commit"),
            ("cpu", {"BLOOMERY_REF_BUILD": "6868a52cb\n"}, "case/0", b"\0" * 512, "oracle commit"),
            ("cpu", {}, "bad\tname", b"\0" * 512, "case name"),
            ("cpu", {}, "", b"\0" * 512, "case name"),
            ("cpu", {}, "a" * 129, b"\0" * 512, "case name"),
            ("cpu", {}, ["case/0", "case/0"], b"\0" * 512, "case name"),
            ("cpu", {}, ["case/0", "case_0"], b"\0" * 512, "case name"),
            ("cpu", {}, "case/0", b"", "bad mel file"),
            ("cpu", {}, "case/0", b"\0" * 511, "bad mel file"),
            ("cpu", {}, "case/0", struct.pack("f", float("nan")) * 128, "non-finite mel"),
            ("cpu", {}, "case/0", struct.pack("f", float("inf")) * 128, "non-finite mel"),
        ]
        for device, extra, name, data, error in cases:
            with self.subTest(device=device, extra=extra, error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                mel, out = root / "mel.f32", root / "out"
                mel.write_bytes(data)
                env = {k: v for k, v in os.environ.items() if not k.startswith("FIXTURE_")}
                env.update({"BLOOMERY_REF_BUILD": "6868a52cb", **extra})
                inputs = [item for case in ([name] if isinstance(name, str) else name) for item in (case, str(mel))]
                result = subprocess.run([str(self.bin), "projector", str(out), device, *inputs],
                                        env=env, capture_output=True, text=True, timeout=FAILSAFE)
                manifest = out / "MANIFEST.tsv"
                if error:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertFalse(manifest.exists())
                    if error == "visible GPU backend":
                        self.assertFalse(out.exists(), "GPU refusal must precede the output writer")
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"# device\t{device}\n", manifest.read_text())
                    self.assertIn("# complete\t18\t0\n", manifest.read_text())
                    again = subprocess.run([str(self.bin), "projector", str(out), device, name, str(mel)],
                                           env=env, capture_output=True, text=True, timeout=FAILSAFE)
                    self.assertNotEqual(again.returncode, 0)
                    self.assertIn("output path must be new", again.stderr)


class Options(unittest.TestCase):
    def test_nonroot_build_recipe_names_the_failure_before_detaching(self):
        recipe = (ROOT / "justfile").read_text().split("\nbuild-emb2-lcpp:\n", 1)[1].split("\n\n", 1)[0]
        words = shlex.split(recipe.strip())
        command = words[words.index("./tools/box.sh") + 1]
        with tempfile.TemporaryDirectory(prefix="bloomery-emb2-build-") as tmp:
            tmp = Path(tmp)
            write(tmp / "id", "#!/bin/bash\necho 501\n", True)
            marker = tmp / "detached"
            write(tmp / "setsid", "#!/bin/bash\ntouch \"$FIXTURE_DETACHED\"\n", True)
            env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ["PATH"], FIXTURE_DETACHED=str(marker))
            result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, timeout=FAILSAFE)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("build-emb2-lcpp: this recipe requires root", result.stderr)
            self.assertFalse(marker.exists())

    @unittest.skipUnless(DUMPER, "--dumper PATH is needed for the linked C++ option checks")
    def test_thread_counts_are_checked_before_model_loading(self):
        for value in ("", "garbage", "16junk", "-1", "2147483648", "999999999999999999999999999999999"):
            with self.subTest(value=value):
                result = subprocess.run([str(DUMPER), "text", "-t", value], capture_output=True, text=True, timeout=FAILSAFE)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("-t requires an integer from 0 to INT_MAX", result.stderr)
        for value in ("0", "1", "16", "2147483647"):
            with self.subTest(value=value):
                result = subprocess.run([str(DUMPER), "text", "-t", value], capture_output=True, text=True, timeout=FAILSAFE)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("usage: dump_emb2", result.stderr)
                self.assertNotIn("-t requires", result.stderr)


class Ids(unittest.TestCase):
    """set_writer.h's parse_ids, built alone: an id file is digits only, one id from 0 to INT32_MAX a line."""

    @classmethod
    def setUpClass(cls):
        cxx = shutil.which("c++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("a C++ compiler is required for the id reader checks")
        cls.tmp = Path(tempfile.mkdtemp(prefix="bloomery-emb2-ids-"))
        # gguf_arch needs gguf.h's four calls; this build never calls it, so they are stubs and llama.cpp is not needed.
        write(cls.tmp / "ids.cpp", GGUF_STUB + '#define DUMP_TOOL "ids"\n#include "' + str(ROOT / "tools/ref/clefvis/set_writer.h") + '"\n'
              "int main(int, char ** argv) {\n    std::vector<uint8_t> raw;\n    if (!read_file(argv[1], raw)) return 2;\n"
              '    for (int32_t id : parse_ids(argv[1], raw)) printf("%d\\n", id);\n}\n')
        cls.bin = cls.tmp / "ids"
        subprocess.run([cxx, "-std=c++17", "-o", str(cls.bin), str(cls.tmp / "ids.cpp")], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def read(self, data):
        path = self.tmp / "case.ids"
        path.write_bytes(data)
        return path, subprocess.run([str(self.bin), str(path)], capture_output=True, timeout=FAILSAFE)

    def test_malformed_lines_name_the_file_and_line(self):
        for data in (b"7\n-1\n", b"7\n+1\n", b"7\n 1\n", b"7\n1 \n", b"7\n1\x002\n", b"7\n\n", b"7\n2147483648\n", b"7\n0x1\n"):
            with self.subTest(data=data):
                path, result = self.read(data)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(f"ids: {path}:2 is not a token id".encode(), result.stderr)
                self.assertEqual(result.stdout, b"")

    def test_digit_lines_are_read(self):
        _, result = self.read(b"0\n42\n2147483647\n")
        self.assertEqual((result.returncode, result.stdout), (0, b"0\n42\n2147483647\n"), result.stderr)


unittest.main(argv=["check-emb2"], verbosity=2)
PY
