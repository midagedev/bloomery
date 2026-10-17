#!/usr/bin/env python3
"""Build/run immutable dense-Qwen forced-logit diagnostics; compare without an invented accuracy band.

The bundles use the existing MANIFEST.tsv reader, but are diagnostic captures, not a registered
reference family or a passing dense GPU gate. `compare --bitwise` checks native eager/graph equality
only. `--self-test` checks malformed/incomplete inputs and comparison contracts without a GPU.
"""
import argparse
import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/ref/dense_logits_ref.cpp"
_spec = importlib.util.spec_from_file_location("manifest", ROOT / "tools/bloomery/manifest.py")
manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(manifest)

# Each diagnostic target is an immutable, complete file. The small target is explicitly opt-in.
TARGETS = {
    "qwen38-27b": {
        "vocab": 248320,
        "size": 16810714336,
        "sha256": "e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520",
        "repository": "lmstudio-community/Qwen3.8-27B-GGUF",
        "revision": "5a7da681f60570ab5b439a587e912d2e5eddb582",
        "file": "Qwen3.8-27B-Q4_K_M.gguf",
    },
    "qwen35-0.8b-candidate": {
        "vocab": 248320,
        "size": 590880672,
        "sha256": "be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988",
        "repository": "notschmee/Qwen3.5-0.8B-GGUF",
        "revision": "1870603ac8ab59c7ddb9d94cccf3b297e379538f",
        "file": "Qwen3.5-0.8B-Q4_K_M.gguf",
    },
}


def digest(path):
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def file_signature(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def immutable_model(path, target):
    before = file_signature(path)
    if before[2] != target["size"] or digest(path) != target["sha256"] or file_signature(path) != before:
        raise ValueError("model differs from the selected immutable target; no engine was opened")
    return before


def model_unchanged(path, before):
    if file_signature(path) != before:
        raise ValueError("model file changed during capture")


def command(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, **kwargs).strip()


def git(tree, *args):
    return command(["git", "-C", tree, *args])


def clean_head(tree):
    if git(tree, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"{tree}: source changes or untracked files, refusing commit attribution")
    return git(tree, "rev-parse", "HEAD")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def token_ids(path, vocab=248320):
    text = Path(path).read_text()
    lines = text.splitlines()
    if not lines or len(lines) > 65536:
        raise ValueError(f"{path}: requires 1..65536 token ids")
    if any(not re.fullmatch(r"[0-9]+", line) or int(line) >= vocab for line in lines):
        raise ValueError(f"{path}: each line must be one decimal id in this vocabulary [0,{vocab})")
    return [int(line) for line in lines]


def build(args):
    tree, folder = args.llama_tree.resolve(), args.llama_build.resolve()
    commit = clean_head(tree)
    remote = git(tree, "remote", "get-url", "origin")
    if not re.fullmatch(r"(?:https://github.com/|git@github.com:)(?:ggml-org|ggerganov)/llama\.cpp(?:\.git)?", remote):
        raise ValueError("reference source origin is not the official llama.cpp repository")
    cache = (folder / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={tree}" not in cache:
        raise ValueError("llama build directory belongs to a different source tree")
    info = (folder / "common/build-info.cpp").read_text()
    built_commit = re.search(r'LLAMA_COMMIT\s*=\s*"([0-9a-f]+)"', info)
    if not built_commit or not commit.startswith(built_commit[1]):
        raise ValueError("llama build-info commit differs from the clean source tree")
    build_command = [args.cmake, "--build", str(folder), "--parallel", "2", "--target", "llama", "ggml"]
    if not args.cached_libraries:
        subprocess.run(build_command, check=True, timeout=args.build_bound)
        if clean_head(tree) != commit:
            raise ValueError("reference source changed during incremental build")
    libs = sorted({p.resolve() for p in (folder / "bin").glob("lib*.so*") if p.is_file()})
    if not libs:
        raise ValueError("no shared reference libraries in build/bin")
    output = args.out.resolve()
    if output.exists() or Path(str(output) + ".build.json").exists():
        raise ValueError(f"build output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output) + ".partial")
    if tmp.exists():
        raise ValueError(f"incomplete build already exists: {tmp}")
    library = folder / "bin"
    source_sha = digest(SOURCE)
    library_hashes = {str(p): digest(p) for p in libs}
    argv = [args.cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(SOURCE),
            f"-I{tree / 'include'}", f"-I{tree / 'ggml/include'}", f"-L{library}",
            "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{library}", "-o", str(tmp)]
    subprocess.run(argv, check=True, timeout=300)
    if digest(SOURCE) != source_sha or clean_head(tree) != commit:
        raise ValueError("reference harness or llama source changed during link")
    if any(digest(path) != expected for path, expected in library_hashes.items()):
        raise ValueError("reference shared libraries changed during link")
    record = {"llama_tree": str(tree), "llama_build": str(folder), "llama_commit": commit, "remote": remote,
              "source_sha256": source_sha, "binary_sha256": digest(tmp), "command": argv,
              "libraries": library_hashes,
              "cuda_build": "GGML_CUDA:BOOL=ON" in cache,
              "hip_build": "GGML_HIP:BOOL=ON" in cache,
              "libraries_rebuilt": not args.cached_libraries,
              "library_build_command": build_command if not args.cached_libraries else None,
              "library_source_proof": ("successful_incremental_build" if not args.cached_libraries
                                       else "cached_library_hashes_only_source_build_not_proven"),
              "cmake_cache_sha256": digest(folder / "CMakeCache.txt")}
    tmp.rename(output)
    write_json(str(output) + ".build.json", record)
    print(json.dumps(record, indent=2))


def check_build(binary, backend):
    data = json.loads(Path(str(binary) + ".build.json").read_text())
    if digest(binary) != data["binary_sha256"] or digest(SOURCE) != data["source_sha256"]:
        raise ValueError("reference binary or harness source differs from its build record")
    if clean_head(data["llama_tree"]) != data["llama_commit"]:
        raise ValueError("reference source moved after its build")
    if git(data["llama_tree"], "remote", "get-url", "origin") != data["remote"]:
        raise ValueError("reference source origin changed after its build")
    for path, expected in data["libraries"].items():
        if digest(path) != expected:
            raise ValueError(f"reference shared library moved: {path}")
    if backend == "cuda" and not data["cuda_build"]:
        raise ValueError("CUDA diagnostic requested from a build without GGML_CUDA")
    return data


def native_build(args):
    head = clean_head(ROOT)
    outputs = [ROOT / "target/release" / name for name in ("generate_qwen3moe", "bloomery-serve")]
    # A failed rebuild must not leave an old stamp claiming a new executable.
    for output in outputs:
        stamp = Path(str(output) + ".build.json")
        if stamp.exists():
            stamp.unlink()
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / "target")
    argv = ["cargo", "oxide", "build", "--arch", args.arch, "--", "-p", "bloomery-gpu-gates",
            "--features", "glm5next,clef,vision,mimo2,gemma-embedding2", "--release",
            "--bin", "generate_qwen3moe", "--bin", "bloomery-serve"]
    subprocess.run(argv, cwd=ROOT, env=env, check=True, timeout=1800)
    if clean_head(ROOT) != head:
        raise ValueError("native source moved during cargo oxide build")
    for output in outputs:
        record = {"source_tree": str(ROOT), "source_head": head, "binary_sha256": digest(output),
                  "command": argv, "arch": args.arch, "compiler": command(["rustc", "--version"])}
        write_json(str(output) + ".build.json", record)
        print(json.dumps(record, indent=2))


def check_native_build(binary, head):
    record = json.loads(Path(str(binary) + ".build.json").read_text())
    if record["source_head"] != head or Path(record["source_tree"]).resolve() != ROOT:
        raise ValueError("native binary build stamp belongs to another source checkout or commit")
    if digest(binary) != record["binary_sha256"]:
        raise ValueError("native binary changed after its cargo oxide build")
    return record


def library_environment(binary, build_info, env):
    if env.get("LD_PRELOAD") or env.get("LD_AUDIT"):
        raise ValueError("reference provenance requires LD_PRELOAD and LD_AUDIT to be unset")
    lib = str(Path(build_info["llama_build"]) / "bin")
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    resolved = command(["ldd", binary], env=env)
    for line in resolved.splitlines():
        if re.match(r"\s*lib(?:llama|ggml)", line):
            match = re.search(r"=>\s+(\S+)", line)
            if not match or str(Path(match[1]).resolve()) not in build_info["libraries"]:
                raise ValueError("a loaded llama/ggml library differs from the recorded build: " + line)
    return resolved


def session_members(session):
    """Snapshot Linux process identities; a timeout may create additional process groups."""
    members = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == session and fields[0] != "Z":
                members[int(path.name)] = fields[19]  # start time guards against PID reuse
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop_owned_process(process, grace=10):
    """Stop only this start_new_session=True launch, including timeout's other groups."""
    if process.pid == os.getsid(0):
        raise ValueError("refusing to stop the caller's own session")

    def send(sig):
        # Snapshot before reaping the leader so its session number cannot be reused.
        for pid, started in session_members(process.pid).items():
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[3]) == process.pid and fields[19] == started:
                    os.kill(pid, sig)
            except (FileNotFoundError, ProcessLookupError):
                pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while session_members(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    send(signal.SIGKILL)
    process.wait(timeout=10)


def bounded_run(argv, env, log, seconds):
    with subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            stop_owned_process(process)
            raise
        except BaseException:
            stop_owned_process(process)
            raise


def read_meta(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        pair = line.split("\t")
        if len(pair) != 2 or pair[0] in fields:
            raise ValueError(f"{path}: duplicate or malformed metadata line")
        fields[pair[0]] = pair[1]
    if fields.get("format") != "bloomery-forced-logits-v1" or fields.get("dtype") != "float32-le":
        raise ValueError(f"{path}: unsupported raw-logit format")
    if fields.get("complete") != "true" or fields.get("last_step") != "true":
        raise ValueError(f"{path}: incomplete rows or a different prompt cut")
    return fields


def raw_rows(path, rows, vocab):
    if rows < 1 or vocab < 1 or Path(path).stat().st_size != rows * vocab * 4:
        raise ValueError(f"{path}: truncated or excess raw logits, expected {rows} x {vocab} f32")
    values = array.array("f")
    with Path(path).open("rb") as stream:
        values.fromfile(stream, rows * vocab)
    if values.itemsize != 4:
        raise ValueError("host array float is not f32")
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"{path}: nonfinite logit")
    return values


def finish_bundle(stage, identity, meta):
    raw = stage / "logits.0.f32"
    n, vocab = int(meta["rows"]), int(meta["vocab"])
    raw_rows(raw, n, vocab)
    identity.update({"rows": str(n), "vocab": str(vocab), "raw_sha256": digest(raw),
                     "mode": meta["mode"], "prefill": meta["prefill"], "last_step": "true"})
    lines = ["# Dense forced-logit diagnostic; no numerical accuracy gate"]
    for key, value in identity.items():
        text = str(value)
        if any(c in text for c in "\r\n\t"):
            raise ValueError(f"metadata {key} contains a line separator")
        lines.append(f"# {key}\t{text}")
    lines += ["# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\top",
              f"tensor\tlogits\t0\tf32\t{vocab}\t{n}\t1\t1\t{n * vocab * 4}\tDIAGNOSTIC",
              "# complete\t1\t0"]
    (stage / "MANIFEST.tsv").write_text("\n".join(lines) + "\n")


def run(args):
    target = TARGETS[args.target]
    model, binary, output = args.model.resolve(), args.binary.resolve(), args.out.resolve()
    model_stat = immutable_model(model, target)
    prompt, forced = token_ids(args.prompt), token_ids(args.forced)
    if len(forced) > 4096 or len(prompt) + len(forced) > args.ctx:
        raise ValueError("context does not hold prompt plus continuation, or more than 4096 rows")
    if args.ctx < 1 or args.ubatch < 1:
        raise ValueError("context and ubatch must be positive")
    stage = Path(str(output) + ".staging")
    if output.exists() or stage.exists():
        raise ValueError("output or staging directory already exists; use a new output directory")
    source_head = git(ROOT, "rev-parse", "HEAD")
    source_dirty = bool(git(ROOT, "status", "--porcelain", "--untracked-files=normal"))
    if args.kind == "native" and source_dirty:
        raise ValueError("native source has tracked changes; commit and rebuild before attribution")
    driver_sha = digest(__file__)
    binary_sha = digest(binary)
    build_info = (check_build(binary, args.backend) if args.kind == "reference"
                  else check_native_build(binary, source_head))
    if args.kind == "native" and args.backend != "cuda":
        raise ValueError("the current native generator is CUDA only")
    if args.kind == "reference" and args.prefill not in ("step", "batch"):
        raise ValueError("reference prefill is step or batch")
    if args.kind == "native" and args.prefill not in ("pass", "auto", "gemm"):
        raise ValueError("native prefill is pass, auto or gemm")
    if args.backend == "cuda" and binary.parent != (ROOT / "target/release").resolve():
        raise ValueError("CUDA binaries must be in this checkout's target/release for gpu-gate.sh")
    stage.mkdir(parents=True)
    shutil.copyfile(args.prompt, stage / "prompt.ids")
    shutil.copyfile(args.forced, stage / "forced.ids")
    # Re-read the snapshots: the source files may have moved while the model digest was checked.
    if token_ids(stage / "prompt.ids") != prompt or token_ids(stage / "forced.ids") != forced:
        raise ValueError("input ids changed before their snapshot")
    raw = stage / "logits.0.f32"
    env = os.environ.copy()
    env.update({"BLOOMERY_REF_MODEL": str(model), "BLOOMERY_DRAFT": "off",
                "BLOOMERY_QWEN3_UBATCH": str(args.ubatch), "BLOOMERY_QWEN3_KV": "f16",
                "BLOOMERY_GATE_BOUND": "900"})
    if args.kind == "reference":
        argv = [str(binary), "--model", str(model), "--prompt-file", str(stage / "prompt.ids"),
                "--forced-file", str(stage / "forced.ids"), "--out", str(raw),
                "--ctx", str(args.ctx), "--ubatch", str(args.ubatch), "--prefill", args.prefill,
                "--gpu-layers", "99" if args.backend == "cuda" else "0"]
        env["BLOOMERY_REF_WRITE"] = "1"
        (stage / "resolved-libraries.log").write_text(library_environment(binary, build_info, env) + "\n")
    else:
        argv = [str(binary), "--tokens", ",".join(map(str, prompt)), "-n", str(len(forced)),
                "--ctx", str(args.ctx), "--mode", args.mode, "--prefill", args.prefill,
                "--last-step", "--logit-rows", str(raw)]
        env["BLOOMERY_FORCE_IDS"] = str(stage / "forced.ids") + ":0"
        env["BLOOMERY_GEN_SLOTS"] = "1"
    if args.backend == "cuda":
        argv = ["bash", str(ROOT / "tools/gpu-gate.sh"), binary.name, *argv[1:]]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
    write_json(stage / "invocation.json", {"command": argv, "kind": args.kind,
               "backend": args.backend, "target": args.target, "build": build_info,
               "source_head": source_head, "binary_sha256": binary_sha,
               "source_dirty": source_dirty, "driver_sha256": driver_sha,
               "model_stat": model_stat,
               "effective_environment": {key: value for key, value in env.items() if key.startswith("BLOOMERY_")},
               "CUDA_VISIBLE_DEVICES_before_gate": env.get("CUDA_VISIBLE_DEVICES"),
               "BLOOMERY_GATE_CARD": env.get("BLOOMERY_GATE_CARD")})
    try:
        with (stage / "run.log").open("w") as log:
            result = bounded_run(argv, env, log, 2800 if args.backend == "cuda" else 900)
        write_json(stage / "exit.json", {"exit_code": result})
        if result:
            raise ValueError(f"diagnostic exited {result}; incomplete evidence retained at {stage}")
    except subprocess.TimeoutExpired:
        write_json(stage / "exit.json", {"exit_code": 124, "timeout": True})
        raise ValueError(f"diagnostic timed out; incomplete evidence retained at {stage}") from None
    meta = read_meta(str(raw) + ".meta")
    if meta["mode"] != (args.mode if args.kind == "native" else "reference") or meta["prefill"] != args.prefill:
        raise ValueError("writer returned a different mode or prefill schedule")
    if meta["prompt_ids"] != ",".join(map(str, prompt)) or meta["forced_ids"] != ",".join(map(str, forced)):
        raise ValueError("writer returned different prompt or forced ids")
    if int(meta["rows"]) != len(forced) or int(meta["vocab"]) != target["vocab"]:
        raise ValueError("writer returned wrong row count or vocabulary")
    model_unchanged(model, model_stat)
    if git(ROOT, "rev-parse", "HEAD") != source_head or digest(binary) != binary_sha or digest(__file__) != driver_sha:
        raise ValueError("source or binary changed during diagnostic")
    if args.kind == "reference":
        check_build(binary, args.backend)
    else:
        if clean_head(ROOT) != source_head:
            raise ValueError("native source became dirty during diagnostic")
        check_native_build(binary, source_head)
    identity = {"format": "bloomery-forced-logits-v1", "model": str(model),
                "model_sha256": target["sha256"], "model_revision": target["revision"],
                "arch": "qwen35", "target": args.target, "engine": args.kind,
                "backend": args.backend, "build": build_info["llama_commit"] if args.kind == "reference" else source_head,
                "source_head": source_head, "binary_sha256": binary_sha,
                "source_dirty": str(source_dirty).lower(), "driver_sha256": driver_sha,
                "prompt_sha256": digest(stage / "prompt.ids"),
                "forced_sha256": digest(stage / "forced.ids"), "ctx": str(args.ctx),
                "ubatch": str(args.ubatch), "kv": "f16"}
    finish_bundle(stage, identity, meta)
    # mkdir is exclusive, unlike rename which can replace a concurrently created empty dir.
    # The existing manifest completion marker is published after all of its files.
    output.mkdir()
    for path in stage.iterdir():
        if path.name != "MANIFEST.tsv":
            path.rename(output / path.name)
    (stage / "MANIFEST.tsv").rename(output / "MANIFEST.tsv")
    stage.rmdir()
    print(f"complete diagnostic bundle: {output}; GPU correctness acceptance remains separate")


def bundle(path):
    path = Path(path)
    m = manifest.read(str(path))
    if m.complete != "1\t0" or m.header.get("format") != "bloomery-forced-logits-v1":
        raise ValueError(f"{path}: incomplete or foreign diagnostic bundle")
    target = TARGETS.get(m.header.get("target"))
    if target is None or m.header.get("model_sha256") != target["sha256"]:
        raise ValueError(f"{path}: wrong immutable model identity")
    rows = m.rows("tensor")
    if len(rows) != 1 or rows[0]["name"] != "logits" or rows[0]["type"] != "f32":
        raise ValueError(f"{path}: requires one f32 logits tensor")
    n, vocab = rows[0].int("ne1"), rows[0].int("ne0")
    if vocab != target["vocab"] or m.header.get("arch") != "qwen35":
        raise ValueError(f"{path}: architecture or vocabulary differs from the immutable target")
    if rows[0].int("occurrence") != 0 or rows[0].int("ne2") != 1 or rows[0].int("ne3") != 1 or rows[0].int("bytes") != n * vocab * 4:
        raise ValueError(f"{path}: inconsistent tensor dimensions or declared bytes")
    if (n, vocab) != (int(m.header["rows"]), int(m.header["vocab"])):
        raise ValueError(f"{path}: row shape differs from metadata")
    for filename, key in (("prompt.ids", "prompt_sha256"), ("forced.ids", "forced_sha256"),
                          ("logits.0.f32", "raw_sha256")):
        if digest(path / filename) != m.header[key]:
            raise ValueError(f"{path}: digest mismatch for {filename}")
    token_ids(path / "prompt.ids", vocab)
    if len(token_ids(path / "forced.ids", vocab)) != n:
        raise ValueError(f"{path}: forced-id count differs from rows")
    return m.header, raw_rows(path / "logits.0.f32", n, vocab)


def compare(left, right, bitwise=False):
    a, x = bundle(left)
    b, y = bundle(right)
    for key in ("model_sha256", "prompt_sha256", "forced_sha256", "rows", "vocab", "ctx", "kv", "last_step"):
        if a[key] != b[key]:
            raise ValueError(f"comparison differs at {key}: {a[key]} versus {b[key]}")
    if bitwise:
        if a["engine"] != "native" or b["engine"] != "native":
            raise ValueError("bitwise acceptance is only defined for native eager/graph diagnostics")
        for key in ("build", "binary_sha256", "prefill", "ubatch", "backend"):
            if a[key] != b[key]:
                raise ValueError(f"bitwise comparison differs at {key}")
        if {a["mode"], b["mode"]} != {"eager", "graph"}:
            raise ValueError("bitwise acceptance needs one eager and one graph run")
    n, vocab = int(a["rows"]), int(a["vocab"])
    output = []
    for i in range(n):
        p, q = x[i * vocab:(i + 1) * vocab], y[i * vocab:(i + 1) * vocab]
        delta = math.fsum((float(u) - float(v)) ** 2 for u, v in zip(p, q))
        denom = math.fsum(float(v) ** 2 for v in q)
        output.append({"row": i, "max_abs": max(abs(float(u) - float(v)) for u, v in zip(p, q)),
                       "relative_l2": math.sqrt(delta / denom) if denom else (0.0 if not delta else None),
                       "reference_norm_zero": denom == 0,
                       "argmax_left": max(range(vocab), key=p.__getitem__),
                       "argmax_right": max(range(vocab), key=q.__getitem__)})
    equal = a["raw_sha256"] == b["raw_sha256"]
    return {"finite_shape_identity_checks": "pass", "bitwise_equal": equal,
            "acceptance": ("pass" if equal else "fail") if bitwise else "diagnostic_only_no_numeric_band",
            "rows": output}, 1 if bitwise and not equal else 0


def self_test():
    TARGETS["_selftest"] = {"sha256": "selftest-identity", "vocab": 2}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "model"
        model.write_bytes(b"original")
        target = {"size": 8, "sha256": digest(model)}
        before = immutable_model(model, target)
        model_unchanged(model, before)
        model.write_bytes(b"replaced")
        for check in (lambda: model_unchanged(model, before), lambda: immutable_model(model, target)):
            try:
                check()
                raise AssertionError("accepted a model replaced during capture")
            except ValueError:
                pass
        ids_file = root / "ids"
        for bad in ("", "-1\n", "248320\n", "1 x\n", "1\n\n"):
            ids_file.write_text(bad)
            try:
                token_ids(ids_file)
                raise AssertionError(f"accepted malformed ids {bad!r}")
            except ValueError:
                pass
        ids_file.write_text("1\n2\n")
        assert token_ids(ids_file) == [1, 2]
        for name, mode in (("a", "eager"), ("b", "graph")):
            folder = root / name
            folder.mkdir()
            (folder / "prompt.ids").write_text("1\n")
            (folder / "forced.ids").write_text("0\n")
            values = array.array("f", [1.0, -2.0])
            if sys.byteorder != "little":
                values.byteswap()
            (folder / "logits.0.f32").write_bytes(values.tobytes())
            identity = {"target": "_selftest", "model_sha256": TARGETS["_selftest"]["sha256"], "arch": "qwen35",
                        "format": "bloomery-forced-logits-v1", "engine": "native", "build": "same",
                        "binary_sha256": "same", "backend": "cuda", "ctx": "8", "kv": "f16", "ubatch": "8",
                        "prompt_sha256": digest(folder / "prompt.ids"), "forced_sha256": digest(folder / "forced.ids")}
            finish_bundle(folder, identity, {"rows": "1", "vocab": "2", "mode": mode, "prefill": "pass"})
        result, code = compare(root / "a", root / "b", True)
        assert code == 0 and result["bitwise_equal"]
        mp = root / "b/MANIFEST.tsv"
        complete = mp.read_text()
        raw_path = root / "b/logits.0.f32"
        original = raw_path.read_bytes()
        changed = array.array("f", [1.5, -2.0])
        if sys.byteorder != "little":
            changed.byteswap()
        before = digest(raw_path)
        raw_path.write_bytes(changed.tobytes())
        mp.write_text(complete.replace(before, digest(raw_path)))
        result, code = compare(root / "a", root / "b", True)
        assert code == 1 and result["acceptance"] == "fail" and result["rows"][0]["max_abs"] == 0.5
        result, code = compare(root / "a", root / "b")
        assert code == 0 and result["acceptance"] == "diagnostic_only_no_numeric_band"
        raw_path.write_bytes(original)
        mp.write_text(complete)
        for bad, want in ((complete.replace("# complete\t1\t0\n", ""), "incomplete"),
                          (complete.replace(TARGETS["_selftest"]["sha256"], "wrong"), "identity"),
                          (complete.replace("tensor\tlogits\t0\tf32\t2", "tensor\tlogits\t0\tf32\t3"), "vocabulary"),
                          (complete.replace("\t1\t1\t8\tDIAGNOSTIC", "\t2\t1\t8\tDIAGNOSTIC"), "dimensions")):
            mp.write_text(bad)
            try:
                bundle(root / "b")
                raise AssertionError("accepted malformed bundle")
            except ValueError as error:
                assert want in str(error), error
        mp.write_text(complete)
        (root / "b/logits.0.f32").write_bytes(b"\0\0\0")
        try:
            raw_rows(root / "b/logits.0.f32", 1, 2)
            raise AssertionError("accepted truncated rows")
        except ValueError:
            pass
        # A child can create another process group while remaining in the owned session.
        child = "import os,signal,time; os.setpgrp(); signal.signal(signal.SIGTERM,signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
        leader = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(60)"
        process = subprocess.Popen([sys.executable, "-c", leader, child], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            pid = int(process.stdout.readline())
            assert pid in session_members(process.pid)
            stop_owned_process(process, grace=0.1)
            deadline = time.monotonic() + 2
            while session_members(process.pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not session_members(process.pid)
        finally:
            if process.returncode is None:
                stop_owned_process(process, grace=0.1)
            process.stdout.close()
        nan = array.array("f", [math.nan, 0.0])
        if sys.byteorder != "little":
            nan.byteswap()
        (root / "b/logits.0.f32").write_bytes(nan.tobytes())
        try:
            raw_rows(root / "b/logits.0.f32", 1, 2)
            raise AssertionError("accepted NaN")
        except ValueError:
            pass
        meta = root / "raw.meta"
        meta.write_text("format\tbloomery-forced-logits-v1\nformat\tbloomery-forced-logits-v1\ndtype\tfloat32-le\nlast_step\ttrue\ncomplete\ttrue\n")
        try:
            read_meta(meta)
            raise AssertionError("accepted duplicate metadata")
        except ValueError:
            pass
    del TARGETS["_selftest"]
    print("dense-logits self-test: malformed ids, provenance, incomplete/truncated/NaN rows, eager/graph and owned-session cleanup contracts passed")
    return 0


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--llama-tree", type=Path, required=True)
    b.add_argument("--llama-build", type=Path, required=True)
    b.add_argument("--out", type=Path, default=ROOT / "target/release/dense_logits_ref")
    b.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    b.add_argument("--cmake", default="cmake")
    b.add_argument("--build-bound", type=int, default=900)
    b.add_argument("--cached-libraries", action="store_true", help="explicit cached-library smoke only; source/build correspondence is not proven")
    nb = sub.add_parser("build-native")
    nb.add_argument("--arch", default="sm_86")
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=("reference", "native"), required=True)
    r.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--target", choices=TARGETS, default="qwen38-27b")
    for key in ("model", "binary", "prompt", "forced", "out"):
        r.add_argument("--" + key, type=Path, required=True)
    r.add_argument("--prefill", choices=("step", "batch", "pass", "auto", "gemm"), required=True)
    r.add_argument("--mode", choices=("eager", "graph"), default="graph")
    r.add_argument("--ctx", type=int, default=4352)
    r.add_argument("--ubatch", type=int, default=512)
    c = sub.add_parser("compare")
    c.add_argument("left", type=Path)
    c.add_argument("right", type=Path)
    c.add_argument("--bitwise", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "build":
            build(args)
        elif args.action == "build-native":
            native_build(args)
        elif args.action == "run":
            run(args)
        else:
            result, code = compare(args.left, args.right, args.bitwise)
            print(json.dumps(result, indent=2, allow_nan=False))
            return code
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"dense-logits: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
