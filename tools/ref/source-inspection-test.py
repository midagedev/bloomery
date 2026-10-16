#!/usr/bin/env python3
"""Timing runners reject a corrupt index before publishing source provenance.

Run with --self-test (also registered in check-recipes). These are dry runner
processes with real Git repositories and preflight-only engine/card stand-ins:
no model is loaded, no lease is taken, and no timing result is produced.
SOURCE_INSPECTION_RUNNER_ROOT selects another checkout for a FAIL-first run.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def self_test():
    root = Path(os.environ.get("SOURCE_INSPECTION_RUNNER_ROOT", Path(__file__).resolve().parents[2])).resolve()
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("source-inspection-test: git is required")
    checks = 0
    with tempfile.TemporaryDirectory(prefix="source-inspection-") as name:
        tmp = Path(name)
        work = tmp / "work"
        ref = tmp / "ref"
        bindir = tmp / "bin"
        for directory in (work, ref, bindir, tmp / "home"):
            directory.mkdir()
        env = {
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(tmp / "home"),
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "BLOOMERY_DRY": "1",
            "BLOOMERY_PREHEAT": "0",
            "BLOOMERY_AB_ROUNDS": "1",
            "BLOOMERY_AB_WARMUP": "0",
            "BLOOMERY_CPU_BUSY_COMMS": "none",
            "BLOOMERY_REF_MODEL": str(tmp / "model.gguf"),
            "BLOOMERY_DATA": str(tmp / "data"),
            "IK": str(ref),
            "LCPP": str(ref),
            "MRS": str(ref),
            "MRSBIN": str(ref / "mistralrs"),
            "LCPP27752": str(ref),
            "LCPP27754": str(ref),
            "EXL3": str(ref),
            "EXL3_PY": str(ref / "venv/bin/python"),
            "EXL3_MODEL": str(tmp / "exl3-model"),
            "EXL3_WIKITEXT": str(tmp / "wiki.test.raw"),
            "BLOOMERY_TOKENIZE_BIN": str(tmp / "tokenize"),
            "BLOOMERY_GEN_BIN": str(tmp / "plan"),
            # This synthetic model has no profile-sized placement; only dry preflight runs.
            "IK_NCMOE": "0",
            "LCPP_NCMOE": "0",
        }

        def put(path, text, executable=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            if executable:
                path.chmod(0o755)

        def git_run(*args):
            return subprocess.run(
                [git, "-C", str(ref), *args], env=env, check=True,
                text=True, capture_output=True,
            ).stdout.strip()

        def runner(script, model, *arms, extra=None):
            run_env = dict(env, BLOOMERY_MODEL=model)
            if extra:
                run_env.update(extra)
            return subprocess.run(
                ["bash", str(root / "tools/ref" / script), *arms],
                cwd=work, env=run_env, text=True, capture_output=True, timeout=30,
            )

        def expect(name, result, rc, witness=None):
            nonlocal checks
            output = result.stdout + result.stderr
            assert result.returncode == rc, (
                f"{name}: rc {result.returncode}, want {rc}\n{output}"
            )
            if witness is not None:
                assert witness in result.stdout, f"{name}: missing {witness!r}\n{output}"
            if rc == 2:
                for detail in ("source-inspection:", str(ref), "128"):
                    assert detail in result.stderr, f"{name}: missing error context {detail!r}\n{output}"
                assert "dirty_files=" not in result.stdout, f"{name}: published failed inspection\n{output}"
            checks += 1
            print(f"ok {name}")

        # Only the runner preflight may invoke these stand-ins. Any attempt to run
        # a measured engine fails instead of manufacturing a benchmark row.
        engine = """#!/usr/bin/env bash
case ${1:-} in
  --help)
    echo '-m -ngl --n-cpu-moe -fa -t --no-op-offload -fit -np -ctxcp --cache-ram -c --host --port -lzm'
    ;;
  --version) echo 'inspection-fixture 1' ;;
  *) echo 'unexpected measured engine invocation' >&2; exit 90 ;;
esac
"""
        for binary in ("llama-bench", "llama-cli", "llama-completion", "llama-server"):
            put(ref / "build/bin" / binary, engine, True)
        put(ref / "mistralrs", engine, True)
        put(ref / "target/release/generate", engine, True)
        put(ref / "eval/perf.py", "# inspection fixture\n")
        put(ref / "venv/bin/python", engine, True)
        (ref / "venv/lib/python3.12/site-packages/exllamav3-1.dist-info").mkdir(parents=True)
        put(tmp / "exl3-model/config.json", "{}\n")
        put(tmp / "wiki.test.raw", "inspection fixture\n")
        put(bindir / "nvidia-smi", """#!/usr/bin/env bash
case $* in
  *--query-gpu=name,uuid*)
    echo 'NVIDIA RTX A6000, GPU-00000000-0000-0000-0000-000000000001'
    echo 'NVIDIA GeForce RTX 3090, GPU-00000000-0000-0000-0000-000000000002'
    ;;
  *--query-gpu=name*) echo 'NVIDIA RTX A6000' ;;
  *) exit 90 ;;
esac
""", True)
        # The dry server preflight needs --help and --plan, never a timed child.
        put(bindir / "timeout", """#!/usr/bin/env bash
[ "${1:-}" != --kill-after=10 ] || shift
shift
exec "$@"
""", True)
        put(bindir / "curl", "#!/usr/bin/env bash\nexit 90\n", True)
        put(tmp / "plan", "#!/usr/bin/env bash\necho 'plan place=a ctx_max=32768 (inspection fixture)'\n", True)
        # ik-draft's real prompt round-trip preflight still executes in dry mode.
        ids = "\n".join(str(i) for i in range(512)) + "\n"
        put(tmp / "data/engram/corpus-prose.ids", ids)
        put(ref / "build/bin/llama-tokenize", "#!/usr/bin/env bash\nprintf '%s\\n' '" + ",".join(str(i) for i in range(512)) + "'\n", True)
        put(work / "Cargo.toml", "# freshness fixture\n")
        put(work / "Cargo.lock", "# freshness fixture\n")
        put(work / "crates/fixture/Cargo.toml", "# freshness fixture\n")
        put(tmp / "tokenize", "#!/usr/bin/env bash\necho 'inspection prompt'\n", True)
        put(ref / "tracked-a", "a\n")
        put(ref / "tracked-b", "b\n")
        git_run("init", "-q")
        git_run("add", ".")
        git_run("-c", "user.name=Inspection Test", "-c", "user.email=inspection@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "inspection fixture")
        index = ref / ".git/index"
        good_index = index.read_bytes()
        head9 = git_run("rev-parse", "--short=9", "HEAD")
        head10 = git_run("rev-parse", "--short=10", "HEAD")
        cases = [
            ("draft-ik", "ik-draft.sh", "deepseek41", ["prose", "4", "plain"]),
            ("draft-lcpp", "ik-draft.sh", "deepseek41", ["prose", "4", "lcpp"]),
            ("qwen-ik", "depth-qwen3moe.sh", "qwen3moe", ["ik:6"]),
            ("qwen-lcpp", "depth-qwen3moe.sh", "qwen3moe", ["lcpp:6"]),
            ("qwen-mrs", "depth-qwen3moe.sh", "qwen3moe", ["mrs:6"]),
            ("qwen-bin", "depth-qwen3moe.sh", "qwen3moe", [f"bin:{ref}/target/release/generate:6"]),
            ("qwen-server", "depth-qwen3moe.sh", "qwen3moe", ["lcppsrv:6"]),
            ("ds41-ik", "depth-ds41.sh", "deepseek41", ["ik:6"]),
            ("ds41-lcpp", "depth-ds41.sh", "deepseek41", ["lcpp:6"]),
            ("ds41-bin", "depth-ds41.sh", "deepseek41", [f"bin:{ref}/target/release/generate:6"]),
            ("ds41-server", "depth-ds41.sh", "deepseek41", ["lcppsrv:6"]),
            ("glm-27752", "depth-glm5next.sh", "glm5next", ["lcpp27752:6"]),
            ("glm-27754", "depth-glm5next.sh", "glm5next", ["lcpp27754:6"]),
            ("glm-exl3", "depth-glm5next.sh", "glm5next", ["exl3:256"]),
        ]
        for state, count in (("clean", 0), ("dirty", 2), ("corrupt-index", None)):
            if state == "dirty":
                put(ref / "tracked-a", "changed a\n")
                put(ref / "tracked-b", "changed b\n")
                put(ref / "untracked", "not counted\n")
            if state == "corrupt-index":
                index.write_bytes(b"bad index\n")
            for label, script, model, arms in cases:
                result = runner(script, model, *arms)
                test_name = f"{label}-{state}"
                if count is None:
                    expect(test_name, result, 2)
                else:
                    head = head10 if model == "glm5next" else head9
                    expect(test_name, result, 0, f"head={head} dirty_files={count}")
        index.write_bytes(good_index)

        # box.sh's no-Git provenance belongs only to DS41's own synced tree.
        own_bin = work / "target/release/generate"
        put(own_bin, engine, True)
        for commit, expected in (("abc123-dirty", "abc123-dirty(box.sh)"), ("", "?")):
            result = runner("depth-ds41.sh", "deepseek41", f"bin:{own_bin}:6",
                            extra={"BLOOMERY_GIT_COMMIT": commit})
            expect(f"ds41-no-git-{commit or 'unknown'}", result, 0,
                   f"head={expected} dirty_files=?")
        other = tmp / "other/target/release/generate"
        put(other, engine, True)
        result = runner("depth-ds41.sh", "deepseek41", f"bin:{other}:6",
                        extra={"BLOOMERY_GIT_COMMIT": "abc123"})
        expect("ds41-no-git-other-tree", result, 0, "head=? dirty_files=?")
    print(f"source-inspection-test: {checks} checks, 0 failed (dry runners, no timing)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        print("usage: source-inspection-test.py --self-test", file=sys.stderr)
        sys.exit(64)
    sys.exit(self_test())
