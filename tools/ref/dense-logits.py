#!/usr/bin/env python3
"""Build/run immutable dense forced-logit sets (Qwen3.5-architecture and Llama-architecture files); compare them without an
invented accuracy band.

A set is a node-dump `MANIFEST.tsv` (tools/bloomery/manifest.py here, crates/refset there) with the
rows in `logits.0.f32` and the ids it was fed. A CUDA `--kind reference` run is the one writer of the
refset families `logits-qwen38d`, `logits-qwen38nv` (crates/refset/src/arch/qwen35/dense.rs; `just dump-logits-qwen38d`,
`just dump-logits-qwen38nv`) and `logits-nanbeige43` (crates/refset/src/arch/llama/dense.rs; `just dump-logits-nanbeige43`),
each driven through tools/ref/dense-logits.sh: its `# model`, `# build` and `# arch` lines are what the family checks,
and the `# arch` line is the target's own `arch`. HIP reference runs are lab diagnostics with full build
hashes, not those families' sets.
A `--kind native` run is a comparison capture and is no family's set. `compare --bitwise` checks native
eager/graph equality only; `floor` prints the oracle's own spread (mainline CUDA against its CPU twin)
and the top-2 margins the gates' bands are derived from; `identity` measures the size and sha256 a TARGETS entry takes for a
file the tree makes (the NVFP4 twin's). `--self-test` checks the
malformed/incomplete inputs and the comparison contracts without a GPU.
"""
import argparse
import array
import hashlib
import heapq
import importlib.util
import io
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
        "arch": "qwen35",
    },
    "nanbeige43-33b": {
        "vocab": 166912,
        "size": 20788825216,
        "sha256": "2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37",
        "repository": "bartowski/Nanbeige_Nanbeige4.3-33B-GGUF",
        "revision": "9beb68421fdf7be095093a6888325fa2c83ddd38",
        "file": "Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf",
        "arch": "llama",
    },
    # The GGUF twin of the Qwen3.8-27B NVFP4 fixture cut (tools/ref/models/qwen38nv.sh holds the conversion): mainline's converter
    # at the oracle pin turns the cut folder into NVFP4 feed-forward and head blocks (36 bytes per 64 values), Q8_0 attention and
    # delta-rule projections and BF16 for the rest. Its bytes are made on the box, so size and sha256 stay None, a run refuses by
    # name, until `dense-logits.py identity <file>` has measured them and they are written here.
    "qwen38nv-twin": {
        "vocab": 248320,
        "size": None,
        "sha256": None,
        "repository": "nvidia/Qwen3.8-27B-NVFP4",
        "revision": "482ca0f3832238542f8f5295dde86b5f22711d80",
        "file": "qwen38nv-twin.gguf",
        "arch": "qwen35",
    },
    "qwen35-0.8b-candidate": {
        "vocab": 248320,
        "size": 590880672,
        "sha256": "be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988",
        "repository": "notschmee/Qwen3.5-0.8B-GGUF",
        "revision": "1870603ac8ab59c7ddb9d94cccf3b297e379538f",
        "file": "Qwen3.5-0.8B-Q4_K_M.gguf",
        "arch": "qwen35",
    },
    "hip-lab-hybrid": {
        "vocab": 258,
        "size": 685984,
        "sha256": "420f680e27e9ec75c7c29d39298574f7fd2aff8808e9b2566f3915664016ed06",
        "repository": "midagedev/bloomery",
        "revision": "03372d339f99996ec5103484f6e917412191d661",
        "file": "smoke.gguf",
        "arch": "qwen35",
    },
}

# A set's MANIFEST.tsv format, and the raw rows' sidecar format the two writers (dense_logits_ref.cpp,
# generate_qwen3moe --logit-rows) leave beside logits.0.f32; the runner turns the second into the first.
MANIFEST_FORMAT = "bloomery-forced-logits-v2"
RAW_FORMAT = "bloomery-forced-logits-v1"
# The `op` the set's one tensor row carries: the rows are no graph node of any engine.
LOGITS_OP = "FORCED_LOGITS"
# What the library check of a pinned build and a run records of mainline's build directory: the libraries the harness loads.
LIBRARY_GLOBS = ("libllama.so*", "libggml*.so*")


def digest(path):
    with Path(path).open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def file_signature(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def unpinned(target):
    return target.get("size", 0) is None or target.get("sha256", "") is None


def target_identity(path):
    """The `size` and `sha256` a TARGETS entry spells for the file at `path`: what pins a target whose bytes the tree makes."""
    return {"size": Path(path).stat().st_size, "sha256": digest(path)}


def immutable_model(path, target):
    if unpinned(target):
        raise ValueError("the target is unpinned: its file was never measured; run `dense-logits.py identity <file>` "
                         "and write its size and sha256 into TARGETS; no engine was opened")
    before = file_signature(path)
    if before[2] != target["size"] or digest(path) != target["sha256"] or file_signature(path) != before:
        raise ValueError("model differs from the selected immutable target; no engine was opened")
    return before + [target["sha256"]]


def model_unchanged(path, before):
    signature = before[:-1]
    if file_signature(path) != signature or digest(path) != before[-1] or file_signature(path) != signature:
        raise ValueError("model file changed during capture")


def command(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, **kwargs).strip()


def git(tree, *args):
    # safe.directory: a tree another account owns (the box's mainline checkouts) is read, never trusted blind.
    return command(["git", "-c", f"safe.directory={tree}", "-C", tree, *args])


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


def oracle_tree_proof(tree, pin):
    """tools/ref/emb2/oracle-tree.sh's check of a pinned oracle tree: HEAD is the pin, no tracked change, and the
    libllama.so its llama-tokenize loads lies under the tree's build and is newer than every tracked source. It reads
    the tree and builds nothing in it."""
    script = ROOT / "tools/ref/emb2/oracle-tree.sh"
    done = subprocess.run(["bash", "-c", 'source "$1" && oracle_tree dense-logits "$2" "$3" llama-tokenize', "_",
                           str(script), str(tree), pin], capture_output=True, text=True, timeout=300)
    if done.returncode:
        raise ValueError(done.stderr.strip() or f"oracle_tree refused {tree} (rc {done.returncode})")


def build(args):
    tree, folder = args.llama_tree.resolve(), args.llama_build.resolve()
    if args.pin and args.cached_libraries:
        raise ValueError("--pin proves the build itself; it does not combine with --cached-libraries")
    if args.pin and folder != tree / "build":
        # The proof reads <tree>/build; the link must read the same libraries, not a stale build elsewhere.
        raise ValueError("--pin requires --llama-build <llama-tree>/build")
    commit = clean_head(tree)
    remote = git(tree, "remote", "get-url", "origin")
    if args.pin:
        # A pin is a full commit hash: the content, whatever the checkout's remote is called.
        if not re.fullmatch(r"[0-9a-f]{40}", args.pin) or commit != args.pin:
            raise ValueError(f"{tree} is at {commit}, the profile pins {args.pin}")
    elif not re.fullmatch(r"(?:https://github.com/|git@github.com:)(?:ggml-org|ggerganov)/llama\.cpp(?:\.git)?", remote):
        raise ValueError("reference source origin is not the official llama.cpp repository")
    cache = (folder / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={tree}" not in cache:
        raise ValueError("llama build directory belongs to a different source tree")
    info = (folder / "common/build-info.cpp").read_text()
    built_commit = re.search(r'LLAMA_COMMIT\s*=\s*"([0-9a-f]+)"', info)
    if not built_commit or not commit.startswith(built_commit[1]):
        raise ValueError("llama build-info commit differs from the clean source tree")
    build_command = [args.cmake, "--build", str(folder), "--parallel", "2", "--target", "llama", "ggml"]
    if args.pin:
        # The pinned tree is the one the depth runners time: nothing is built in it, its build is proven.
        oracle_tree_proof(tree, args.pin)
    elif not args.cached_libraries:
        subprocess.run(build_command, check=True, timeout=args.build_bound)
        if clean_head(tree) != commit:
            raise ValueError("reference source changed during incremental build")
    libs = sorted({p.resolve() for pattern in LIBRARY_GLOBS for p in (folder / "bin").glob(pattern) if p.is_file()})
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
              "pin": args.pin,
              "libraries_rebuilt": not (args.cached_libraries or args.pin),
              "library_build_command": build_command if not (args.cached_libraries or args.pin) else None,
              "library_source_proof": ("oracle_tree: HEAD is the pin, no tracked change, the loaded libllama.so newer than every source"
                                       if args.pin else
                                       "successful_incremental_build" if not args.cached_libraries
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
    if backend == "hip" and not data.get("hip_build"):
        raise ValueError("HIP diagnostic requested from a build without GGML_HIP")
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
            "--features", "glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint", "--release",
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


def process_identities():
    """pid -> (session, live, start identity) for every process; the start guards against PID reuse."""
    found = {}
    if Path("/proc/self/stat").exists():
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                found[int(path.name)] = (int(fields[3]), fields[0] != "Z", fields[19])
            except (FileNotFoundError, ProcessLookupError):
                continue
        return found
    # No procfs (macOS): one ps snapshot for pid, state and start time, the session from getsid.
    listing = subprocess.run(["ps", "-A", "-o", "pid=,stat=,lstart="], check=True, capture_output=True,
                             text=True, env={**os.environ, "LC_ALL": "C"}).stdout
    for line in listing.splitlines():
        pid, state, started = line.split(None, 2)
        try:
            found[int(pid)] = (os.getsid(int(pid)), not state.startswith("Z"), started.strip())
        except ProcessLookupError:
            continue
    return found


def session_members(session):
    """Snapshot process identities of a session; a timeout may create additional process groups."""
    return {pid: started for pid, (sid, live, started) in process_identities().items()
            if sid == session and live}


def signal_members(session, members, sig):
    """Signal each snapshot member still in the session with the same start identity."""
    now = process_identities()
    for pid, started in members.items():
        if pid in now and now[pid][0] == session and now[pid][2] == started:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def stop_owned_process(process, grace=10):
    """Stop only this start_new_session=True launch, including timeout's other groups."""
    if process.pid == os.getsid(0):
        raise ValueError("refusing to stop the caller's own session")

    def send(sig):
        # Snapshot before reaping the leader so its session number cannot be reused.
        signal_members(process.pid, session_members(process.pid), sig)

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
    if fields.get("format") != RAW_FORMAT or fields.get("dtype") != "float32-le":
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


def corpus_slice(path, sha256, count, vocab):
    """The first `count` ids of a corpus file, one decimal id a line and any length; `sha256` is the file's pinned digest."""
    if digest(path) != sha256:
        raise ValueError(f"{path}: sha256 differs from the pinned corpus digest {sha256}")
    ids = []
    with Path(path).open() as stream:
        for line in stream:
            text = line.rstrip("\n")
            if not re.fullmatch(r"[0-9]+", text) or int(text) >= vocab:
                raise ValueError(f"{path}:{len(ids) + 1}: not one decimal id in [0,{vocab})")
            ids.append(int(text))
            if len(ids) == count:
                return ids
    raise ValueError(f"{path}: holds {len(ids)} ids, {count} wanted")


def fit_reserve(source=None):
    """Read the production owner; unsupported expressions are a refusal, not a new default."""
    text = (ROOT / "crates/gpu/src/model/fit.rs").read_text() if source is None else source
    declarations = re.findall(r"^pub const FIT_RESERVE:\s*usize\s*=\s*(.+);$", text, re.MULTILINE)
    if len(declarations) != 1:
        raise ValueError("requires exactly one production FIT_RESERVE declaration")
    match = re.fullmatch(r"([1-9][0-9]*)\s*<<\s*([0-9]+)", declarations[0])
    if not match or int(match[2]) > 30:
        raise ValueError("unsupported production FIT_RESERVE expression")
    value = int(match[1]) << int(match[2])
    if value > 2147483647:
        raise ValueError("production FIT_RESERVE exceeds reference byte argument range")
    return value


def verify_repeat(folder, meta, rows, vocab):
    repeats = meta.get("reset_repeats", "1")
    if repeats not in ("1", "2"):
        raise ValueError("unsupported reset repeat count")
    if repeats == "1":
        return {}
    name = meta.get("repeat_file", "")
    if name != "logits.0.f32.repeat-1.f32" or meta.get("reset_repeat_bitwise_equal") != "true":
        raise ValueError("missing or unsafe reset-repeat artifact metadata")
    repeat = Path(folder) / name
    raw_rows(repeat, rows, vocab)
    sha = digest(repeat)
    if sha != digest(Path(folder) / "logits.0.f32"):
        raise ValueError("reset-repeat raw artifact differs bitwise")
    return {"repeat_file": name, "reset_repeat_bitwise_equal": "true", "repeat_sha256": sha}


def hip_metadata(meta, reserve, maximum):
    required = ("hip_device", "hip_description", "hip_trunk_layers", "hip_model_layers",
                "hip_output", "hip_math_nodes", "hip_cpu_math_nodes", "hip_reserve_bytes",
                "hip_cpu_input_nodes",
                "hip_max_model_bytes", "model_tensor_bytes", "hip_free_before_model",
                "hip_free_after_load", "hip_free_after_context", "hip_free_after_run",
                "hip_context_allowance_bytes", "hip_registry", "hip_memory_guard",
                "hip_estimate_exclusions")
    if any(not meta.get(key) for key in required) or meta.get("backend") != "hip":
        raise ValueError("missing HIP execution evidence")
    if (meta["hip_registry"] != "ROCm" or meta["hip_output"] != "true" or
            int(meta["hip_trunk_layers"]) != int(meta["hip_model_layers"]) or
            int(meta["hip_model_layers"]) <= 0 or int(meta["hip_math_nodes"]) <= 0 or
            int(meta["hip_cpu_math_nodes"]) < 0 or int(meta["hip_reserve_bytes"]) != reserve or
            int(meta["hip_cpu_input_nodes"]) != int(meta["hip_cpu_math_nodes"]) or
            int(meta["hip_max_model_bytes"]) != maximum or
            not 0 < int(meta["model_tensor_bytes"]) <= maximum):
        raise ValueError("HIP placement or admission evidence disagrees with request")
    allowance = int(meta["hip_context_allowance_bytes"])
    if allowance <= 0 or any(int(meta[key]) < minimum for key, minimum in (
            ("hip_free_before_model", reserve + allowance + int(meta["model_tensor_bytes"])),
            ("hip_free_after_load", reserve + allowance),
            ("hip_free_after_context", reserve), ("hip_free_after_run", reserve))):
        raise ValueError("HIP memory evidence does not retain production reserve")
    return {key: meta[key] for key in required}


def finish_bundle(stage, identity, meta):
    """Write the set's MANIFEST.tsv last: the node dumps' header (`# model`, `# build`, `# arch` are what a family
    checks), the one tensor row, and the `# complete` trailer."""
    raw = stage / "logits.0.f32"
    n, vocab = int(meta["rows"]), int(meta["vocab"])
    values = raw_rows(raw, n, vocab)
    identity.update(verify_repeat(stage, meta, n, vocab))
    if "reset_repeats" in meta:
        identity["reset_repeats"] = meta["reset_repeats"]
    identity.update({"rows": str(n), "vocab": str(vocab), "raw_sha256": digest(raw),
                     "mode": meta["mode"], "schedule": meta["prefill"], "last_step": "true"})
    title = {"reference": "# dense_logits_ref - llama.cpp mainline logits after the prompt and after each forced id",
             "native": "# generate_qwen3moe - bloomery logits after the prompt and after each forced id; a comparison capture, no family's set"}
    lines = [title[identity["engine"]]]
    for key, value in identity.items():
        text = str(value)
        if any(c in text for c in "\r\n\t"):
            raise ValueError(f"metadata {key} contains a line separator")
        lines.append(f"# {key}\t{text}")
    lines += ["# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1",
              f"tensor\tlogits\t0\tf32\t{vocab}\t{n}\t1\t1\t{n * vocab * 4}\t{math.fsum(values)!r}\t{LOGITS_OP}\t1\t0\t-\t-",
              "# complete\t1\t0"]
    (stage / "MANIFEST.tsv").write_text("\n".join(lines) + "\n")


def run(args):
    target = TARGETS[args.target]
    native = args.kind == "native"
    source_bound = native or args.backend == "hip"
    model, binary, output = args.model.resolve(), args.binary.resolve(), args.out.resolve()
    corpus = {}
    if args.corpus:
        if (args.prompt or args.forced or not args.corpus_sha256
                or (args.prompt_count or 0) < 1 or (args.forced_count or 0) < 1):
            raise ValueError("--corpus takes --corpus-sha256, --prompt-count and --forced-count, and no --prompt or --forced")
        ids = corpus_slice(args.corpus, args.corpus_sha256, args.prompt_count + args.forced_count, target["vocab"])
        prompt, forced = ids[:args.prompt_count], ids[args.prompt_count:]
        corpus = {"tokens_file": str(args.corpus.resolve()), "tokens_file_sha256": args.corpus_sha256,
                  "tokens_count": str(len(ids))}
    else:
        if (not args.prompt or not args.forced or args.prompt_count is not None or args.forced_count is not None
                or args.corpus_sha256):
            raise ValueError("a run takes --prompt and --forced, or --corpus with its digest and counts")
        prompt, forced = token_ids(args.prompt, target["vocab"]), token_ids(args.forced, target["vocab"])
    if len(forced) > 4096 or len(prompt) + len(forced) > args.ctx:
        raise ValueError("context does not hold prompt plus continuation, or more than 4096 rows")
    if args.ctx < 1 or args.ubatch < 1:
        raise ValueError("context and ubatch must be positive")
    if not 0 < args.hip_max_model_bytes <= 2147483647:
        raise ValueError("hip-max-model-bytes must be positive and fit the reference byte argument range")
    if args.reset_repeats not in (1, 2):
        raise ValueError("reset-repeats is 1 or 2")
    if native and args.reset_repeats != 1:
        raise ValueError("reset-repeats is supported by the reference writer only")
    model_stat = immutable_model(model, target)
    reserve = fit_reserve() if args.backend == "hip" else None
    reserve_source_sha = digest(ROOT / "crates/gpu/src/model/fit.rs") if reserve is not None else None
    stage = Path(str(output) + ".staging")
    if output.exists() or stage.exists():
        raise ValueError("output or staging directory already exists; use a new output directory")
    # Native and lab HIP captures bind this checkout. Standalone CPU/CUDA references
    # bind mainline's build record and may run from a deployment without .git.
    source_head = git(ROOT, "rev-parse", "HEAD") if source_bound else None
    source_dirty = bool(git(ROOT, "status", "--porcelain", "--untracked-files=normal")) if source_bound else None
    if native and source_dirty:
        raise ValueError("native source has tracked changes; commit and rebuild before attribution")
    driver_sha = digest(__file__)
    binary_sha = digest(binary)
    build_info = check_native_build(binary, source_head) if native else check_build(binary, args.backend)
    if native and args.backend != "cuda":
        raise ValueError("the current native generator is CUDA only")
    if not native and args.prefill not in ("step", "batch"):
        raise ValueError("reference prefill is step or batch")
    if native and args.prefill not in ("pass", "auto", "gemm"):
        raise ValueError("native prefill is pass, auto or gemm")
    if args.backend == "cuda" and binary.parent != (ROOT / "target/release").resolve():
        raise ValueError("CUDA binaries must be in this checkout's target/release for gpu-gate.sh")
    stage.mkdir(parents=True)
    if corpus:
        for name, values in (("prompt.ids", prompt), ("forced.ids", forced)):
            (stage / name).write_text("".join(f"{i}\n" for i in values))
    else:
        shutil.copyfile(args.prompt, stage / "prompt.ids")
        shutil.copyfile(args.forced, stage / "forced.ids")
    # Re-read the snapshots: the source files may have moved while the model digest was checked.
    if token_ids(stage / "prompt.ids", target["vocab"]) != prompt or token_ids(stage / "forced.ids", target["vocab"]) != forced:
        raise ValueError("input ids changed before their snapshot")
    raw = stage / "logits.0.f32"
    env = os.environ.copy()
    env.update({"BLOOMERY_REF_MODEL": str(model), "BLOOMERY_DRAFT": "off",
                "BLOOMERY_QWEN3_UBATCH": str(args.ubatch), "BLOOMERY_QWEN3_KV": "f16",
                "BLOOMERY_GATE_BOUND": "900"})
    if not native:
        argv = [str(binary), "--model", str(model), "--arch", target["arch"], "--prompt-file", str(stage / "prompt.ids"),
                "--forced-file", str(stage / "forced.ids"), "--out", str(raw),
                "--ctx", str(args.ctx), "--ubatch", str(args.ubatch), "--prefill", args.prefill,
                "--backend", args.backend, "--reset-repeats", str(args.reset_repeats),
                "--gpu-layers", "0" if args.backend == "cpu" else "99"]
        if args.backend == "hip":
            argv += ["--hip-max-model-bytes", str(args.hip_max_model_bytes),
                     "--hip-reserve-bytes", str(reserve)]
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
    elif args.backend == "cpu":
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
               "HIP_VISIBLE_DEVICES": env.get("HIP_VISIBLE_DEVICES"),
               "ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"),
               "BLOOMERY_GATE_CARD": env.get("BLOOMERY_GATE_CARD")})
    bound = args.bound or (2800 if args.backend == "cuda" else 900)
    try:
        with (stage / "run.log").open("w") as log:
            result = bounded_run(argv, env, log, bound)
        write_json(stage / "exit.json", {"exit_code": result})
        if result:
            raise ValueError(f"diagnostic exited {result}; incomplete evidence retained at {stage}")
    except subprocess.TimeoutExpired:
        write_json(stage / "exit.json", {"exit_code": 124, "timeout": True})
        raise ValueError(f"diagnostic timed out; incomplete evidence retained at {stage}") from None
    meta = read_meta(str(raw) + ".meta")
    if meta["mode"] != (args.mode if native else "reference") or meta["prefill"] != args.prefill:
        raise ValueError("writer returned a different mode or prefill schedule")
    if meta["prompt_ids"] != ",".join(map(str, prompt)) or meta["forced_ids"] != ",".join(map(str, forced)):
        raise ValueError("writer returned different prompt or forced ids")
    if int(meta["rows"]) != len(forced) or int(meta["vocab"]) != target["vocab"]:
        raise ValueError("writer returned wrong row count or vocabulary")
    if args.kind == "reference" and (meta.get("backend") != args.backend or
                                    meta.get("reset_repeats") != str(args.reset_repeats)):
        raise ValueError("writer returned a different backend or reset schedule")
    model_unchanged(model, model_stat)
    if (source_bound and git(ROOT, "rev-parse", "HEAD") != source_head) or digest(binary) != binary_sha or digest(__file__) != driver_sha:
        raise ValueError("source or binary changed during diagnostic")
    if native:
        if clean_head(ROOT) != source_head:
            raise ValueError("native source became dirty during diagnostic")
        check_native_build(binary, source_head)
    else:
        check_build(binary, args.backend)
    identity = {"format": MANIFEST_FORMAT, "model": str(model),
                # Canonical CPU/CUDA sets retain the family's nine-digit build identity.
                # HIP lab diagnostics keep the full hash and cannot impersonate that family.
                "build": (source_head if native else build_info["llama_commit"] if args.backend == "hip"
                          else build_info["llama_commit"][:9]),
                "arch": target["arch"], "model_sha256": target["sha256"], "model_revision": target["revision"],
                "target": args.target, "engine": args.kind, "backend": args.backend,
                "binary_sha256": binary_sha, "driver_sha256": driver_sha,
                "prompt_sha256": digest(stage / "prompt.ids"), "forced_sha256": digest(stage / "forced.ids"),
                "ctx": str(args.ctx), "ubatch": str(args.ubatch), "kv": "f16", **corpus}
    if source_bound:
        identity.update({"source_head": source_head, "source_dirty": str(source_dirty).lower()})
    if args.backend == "hip":
        identity.update(hip_metadata(meta, reserve, args.hip_max_model_bytes))
        if fit_reserve() != reserve or digest(ROOT / "crates/gpu/src/model/fit.rs") != reserve_source_sha:
            raise ValueError("production FIT_RESERVE source changed during capture")
        identity["hip_fit_reserve_source_sha256"] = reserve_source_sha
    finish_bundle(stage, identity, meta)
    # mkdir is exclusive, unlike rename which can replace a concurrently created empty dir.
    # The existing manifest completion marker is published after all of its files.
    output.mkdir()
    for path in stage.iterdir():
        if path.name != "MANIFEST.tsv":
            path.rename(output / path.name)
    (stage / "MANIFEST.tsv").rename(output / "MANIFEST.tsv")
    stage.rmdir()
    print(f"complete set: {output}; GPU correctness acceptance remains separate")


def bundle(path):
    path = Path(path)
    m = manifest.read(str(path))
    if m.complete != "1\t0":
        raise ValueError(f"{path}: incomplete or foreign diagnostic bundle")
    if m.header.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path}: not a {MANIFEST_FORMAT} set (an older capture: dump it again)")
    target = TARGETS.get(m.header.get("target"))
    if target is not None and unpinned(target):
        raise ValueError(f"{path}: the target {m.header.get('target')} is unpinned (no size and sha256 in TARGETS)")
    if target is None or m.header.get("model_sha256") != target["sha256"]:
        raise ValueError(f"{path}: wrong immutable model identity")
    rows = m.rows("tensor")
    if len(rows) != 1 or rows[0]["name"] != "logits" or rows[0]["type"] != "f32" or rows[0]["op"] != LOGITS_OP:
        raise ValueError(f"{path}: requires one f32 logits tensor")
    n, vocab = rows[0].int("ne1"), rows[0].int("ne0")
    if vocab != target["vocab"] or m.header.get("arch") != target["arch"]:
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
    values = raw_rows(path / "logits.0.f32", n, vocab)
    if repr(math.fsum(values)) != rows[0]["sum"]:
        raise ValueError(f"{path}: the tensor row's sum differs from the rows")
    repeat = verify_repeat(path, m.header, n, vocab)
    if repeat and repeat["repeat_sha256"] != m.header.get("repeat_sha256"):
        raise ValueError("repeat artifact differs from manifest identity")
    return m.header, values


def compare(left, right, bitwise=False):
    a, x = bundle(left)
    b, y = bundle(right)
    for key in ("model_sha256", "prompt_sha256", "forced_sha256", "rows", "vocab", "ctx", "kv", "last_step"):
        if a[key] != b[key]:
            raise ValueError(f"comparison differs at {key}: {a[key]} versus {b[key]}")
    if bitwise:
        if a["engine"] != "native" or b["engine"] != "native":
            raise ValueError("bitwise acceptance is only defined for native eager/graph diagnostics")
        for key in ("build", "binary_sha256", "schedule", "ubatch", "backend"):
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


def quantiles(values):
    """{point: value} at 0, 10, 25, 50, 75, 90 and 100 % of the sorted values, linear between ranks; an empty list has none."""
    if not values:
        raise ValueError("quantiles of no values")
    ordered = sorted(values)
    found = {}
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        position = q * (len(ordered) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(ordered) - 1)
        found[q] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return found


def top2(row):
    """(argmax, top-1 minus top-2) of one logits row; ties are a margin of 0."""
    (first, _), (second, _) = heapq.nlargest(2, enumerate(row), key=lambda pair: pair[1])
    return first, float(row[first]) - float(row[second])


def stable_margin(l2, max_abs):
    """[derived] The margin above which a row's argmax cannot move when every logit moves by a vector of L2 norm `l2` and
    largest component `max_abs`: two logits swap only if their margin is within the difference of their two moves, at most
    2 * max_abs (each move is at most max_abs) and at most sqrt(2) * l2 (Cauchy-Schwarz on the pair); the smaller holds."""
    return min(2.0 * max_abs, math.sqrt(2.0) * l2)


def floor(references, twin_suffix=".cpu"):
    """The oracle's own spread of each reference set: the CPU twin's rows beside it (`<set>.cpu`, one binary, one prompt,
    one ubatch, no card in view) against mainline's CUDA rows, the set being the reference as in tools/ref/hidden-diff.py:
    per row the relative L2 `|cpu - cuda| / |cuda|`, the largest absolute difference, whether the argmax agrees, and the
    top-2 margin of both. A row's own [derived] bound (stable_margin) says which rows could flip: those whose CUDA margin
    is at or under it."""
    sets = []
    for reference in references:
        twin = Path(str(reference) + twin_suffix)
        a, x = bundle(reference)
        b, y = bundle(twin)
        if (a["engine"], b["engine"]) != ("reference", "reference") or (a["backend"], b["backend"]) != ("cuda", "cpu"):
            raise ValueError(f"{reference}: floor takes a reference CUDA set and its reference CPU twin")
        for key in ("model_sha256", "build", "binary_sha256", "prompt_sha256", "forced_sha256", "rows", "vocab", "ctx",
                    "ubatch", "schedule", "kv"):
            if a[key] != b[key]:
                raise ValueError(f"{reference} and its twin differ at {key}: {a[key]} versus {b[key]}")
        n, vocab = int(a["rows"]), int(a["vocab"])
        rows = []
        for i in range(n):
            p, q = x[i * vocab:(i + 1) * vocab], y[i * vocab:(i + 1) * vocab]
            delta = math.sqrt(math.fsum((float(u) - float(v)) ** 2 for u, v in zip(p, q)))
            norm = math.sqrt(math.fsum(float(u) ** 2 for u in p))
            if not norm:
                raise ValueError(f"{reference}: row {i} is all zero")
            cuda_arg, cuda_margin = top2(p)
            cpu_arg, cpu_margin = top2(q)
            largest = max(abs(float(u) - float(v)) for u, v in zip(p, q))
            rows.append({"row": i, "rel_l2": delta / norm, "l2": delta, "cuda_norm": norm, "max_abs": largest,
                         "stable_margin": stable_margin(delta, largest),
                         "argmax_cuda": cuda_arg, "argmax_cpu": cpu_arg,
                         "margin_cuda": cuda_margin, "margin_cpu": cpu_margin})
        sets.append({"set": str(reference), "twin": str(twin), "rows": rows,
                     "prompt_len": len(token_ids(Path(reference) / "prompt.ids", vocab)),
                     "backends": [a["backend"], b["backend"]]})
    pooled = [r for s in sets for r in s["rows"]]
    summary = {"rows": len(pooled), "rel_l2": quantiles([r["rel_l2"] for r in pooled]),
               "max_abs": quantiles([r["max_abs"] for r in pooled]),
               "margin_cuda": quantiles([r["margin_cuda"] for r in pooled]),
               "margin_cpu": quantiles([r["margin_cpu"] for r in pooled]),
               "argmax_equal": sum(r["argmax_cuda"] == r["argmax_cpu"] for r in pooled),
               "stable_margin": quantiles([r["stable_margin"] for r in pooled]),
               "rows_within_bound": sum(r["margin_cuda"] <= r["stable_margin"] for r in pooled)}
    return {"sets": sets, "pooled": summary}


def print_floor(report, stream=sys.stdout):
    def q_line(label, found):
        names = ("min", "p10", "p25", "median", "p75", "p90", "max")
        print(f"  {label}: " + "  ".join(f"{name} {value:.6g}" for name, value in zip(names, found.values())), file=stream)

    for s in report["sets"]:
        rows = s["rows"]
        print(f"floor {s['set']}: {len(rows)} rows after a prompt of {s['prompt_len']} ids; cpu twin vs cuda", file=stream)
        q_line("rel_l2 |cpu-cuda|/|cuda|", quantiles([r["rel_l2"] for r in rows]))
        q_line("max_abs", quantiles([r["max_abs"] for r in rows]))
        q_line("top-2 margin, cuda rows", quantiles([r["margin_cuda"] for r in rows]))
        q_line("top-2 margin, cpu rows", quantiles([r["margin_cpu"] for r in rows]))
        flips = [r for r in rows if r["argmax_cuda"] != r["argmax_cpu"]]
        print(f"  argmax cuda == cpu: {len(rows) - len(flips)}/{len(rows)}"
              + "".join(f"; row {r['row']} flips at margins {r['margin_cuda']:.6g}/{r['margin_cpu']:.6g}" for r in flips), file=stream)
    p = report["pooled"]
    print(f"floor pooled: {p['rows']} rows", file=stream)
    q_line("rel_l2", p["rel_l2"])
    q_line("max_abs", p["max_abs"])
    q_line("top-2 margin, cuda rows", p["margin_cuda"])
    q_line("top-2 margin, cpu rows", p["margin_cpu"])
    q_line("[derived] stable margin min(2 max_abs, sqrt(2) l2)", p["stable_margin"])
    print(f"  argmax cuda == cpu: {p['argmax_equal']}/{p['rows']}; {p['rows_within_bound']} of {p['rows']} rows have a cuda "
          f"margin at or under their own stable margin (the only rows that could flip)", file=stream)


def self_test():
    TARGETS["_selftest"] = {"sha256": "selftest-identity", "vocab": 2, "arch": "qwen35"}
    TARGETS["_selftest_llama"] = {"sha256": "selftest-llama-identity", "vocab": 2, "arch": "llama"}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model = root / "model"
        model.write_bytes(b"original")
        target = {"size": 8, "sha256": digest(model)}
        before = immutable_model(model, target)
        model_unchanged(model, before)
        assert target_identity(model) == target and unpinned(TARGETS["qwen38nv-twin"]) and not unpinned(target)
        for pinned_by_hand in ({"size": None, "sha256": "x"}, {"size": 8, "sha256": None}):
            try:
                immutable_model(model, pinned_by_hand)
                raise AssertionError("opened a target with a missing identity half")
            except ValueError as error:
                assert "unpinned" in str(error), error
        model.write_bytes(b"replaced")
        # Same metadata must not hide a content change.
        before[:-1] = file_signature(model)
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
        ids_file.write_text("258\n")
        try:
            token_ids(ids_file, TARGETS["hip-lab-hybrid"]["vocab"])
            raise AssertionError("accepted out-of-vocabulary small-model input")
        except ValueError:
            pass
        repeat_folder = root / "repeat"
        repeat_folder.mkdir()
        raw = repeat_folder / "logits.0.f32"
        repeat = repeat_folder / "logits.0.f32.repeat-1.f32"
        finite = array.array("f", [1.0, -2.0])
        if sys.byteorder != "little":
            finite.byteswap()
        raw.write_bytes(finite.tobytes())
        repeat.write_bytes(finite.tobytes())
        repeat_meta = {"reset_repeats": "2", "reset_repeat_bitwise_equal": "true",
                       "repeat_file": repeat.name}
        evidence = verify_repeat(repeat_folder, repeat_meta, 1, 2)
        assert evidence["repeat_sha256"] == digest(raw)
        for bad in ({**repeat_meta, "repeat_file": "../logits.0.f32"},
                    {**repeat_meta, "reset_repeat_bitwise_equal": "false"},
                    {**repeat_meta, "reset_repeats": "3"}):
            try:
                verify_repeat(repeat_folder, bad, 1, 2)
                raise AssertionError("accepted unsafe/incomplete repeat evidence")
            except ValueError:
                pass
        for bad_bytes in (finite.tobytes()[:-1], b"\0" * 8,
                          bytes.fromhex("0000c07f00000000")):
            repeat.write_bytes(bad_bytes)
            try:
                verify_repeat(repeat_folder, repeat_meta, 1, 2)
                raise AssertionError("accepted truncated, unequal, or nonfinite reset repeat")
            except ValueError:
                pass
        assert fit_reserve("pub const FIT_RESERVE: usize = 256 << 20;") == 268435456
        for bad in ("", "pub const FIT_RESERVE: usize = 1 + 2;",
                    "pub const FIT_RESERVE: usize = 1 << 63;",
                    "pub const FIT_RESERVE: usize = 1 << 20;\n" * 2):
            try:
                fit_reserve(bad)
                raise AssertionError("accepted unsupported/ambiguous reserve source")
            except ValueError:
                pass
        hip = {"backend": "hip", "hip_device": "ROCm0", "hip_description": "selected GPU",
               "hip_registry": "ROCm", "hip_output": "true", "hip_trunk_layers": "2",
               "hip_model_layers": "2", "hip_math_nodes": "3", "hip_cpu_math_nodes": "1",
               "hip_cpu_input_nodes": "1",
               "hip_reserve_bytes": "256", "hip_max_model_bytes": "16", "model_tensor_bytes": "8",
               "hip_context_allowance_bytes": "32", "hip_memory_guard": "stage_preflight_not_allocation_cap",
               "hip_estimate_exclusions": "driver", "hip_free_before_model": "296",
               "hip_free_after_load": "288", "hip_free_after_context": "256", "hip_free_after_run": "256"}
        assert hip_metadata(hip, 256, 16)["hip_cpu_math_nodes"] == "1"
        for key, value in (("backend", "cpu"), ("hip_registry", "CUDA"), ("hip_output", "false"),
                           ("hip_trunk_layers", "1"), ("hip_math_nodes", "0"),
                           ("hip_cpu_math_nodes", "-1"), ("model_tensor_bytes", "17"),
                           ("hip_cpu_input_nodes", "0"),
                           ("hip_reserve_bytes", "255"), ("hip_free_before_model", "295"),
                           ("hip_free_after_load", "287"), ("hip_free_after_context", "255"),
                           ("hip_free_after_run", "255")):
            try:
                hip_metadata({**hip, key: value}, 256, 16)
                raise AssertionError("accepted incomplete placement or unsafe HIP memory evidence")
            except ValueError:
                pass
        for name, mode in (("a", "eager"), ("b", "graph")):
            folder = root / name
            folder.mkdir()
            (folder / "prompt.ids").write_text("1\n")
            (folder / "forced.ids").write_text("0\n")
            values = array.array("f", [1.0, -2.0])
            if sys.byteorder != "little":
                values.byteswap()
            (folder / "logits.0.f32").write_bytes(values.tobytes())
            identity = {"format": MANIFEST_FORMAT, "model": "/models/selftest.gguf", "build": "same",
                        "arch": TARGETS["_selftest"]["arch"],
                        "target": "_selftest", "model_sha256": TARGETS["_selftest"]["sha256"], "engine": "native",
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
        # The digest and the row's sum both follow the rows, as a writer's would.
        mp.write_text(complete.replace(before, digest(raw_path)).replace("\t-1.0\t", "\t-0.5\t"))
        result, code = compare(root / "a", root / "b", True)
        assert code == 1 and result["acceptance"] == "fail" and result["rows"][0]["max_abs"] == 0.5
        result, code = compare(root / "a", root / "b")
        assert code == 0 and result["acceptance"] == "diagnostic_only_no_numeric_band"
        raw_path.write_bytes(original)
        mp.write_text(complete)
        for bad, want in ((complete.replace("# complete\t1\t0\n", ""), "incomplete"),
                          (complete.replace(TARGETS["_selftest"]["sha256"], "wrong"), "identity"),
                          (complete.replace("# arch\tqwen35", "# arch\tllama"), "architecture"),
                          (complete.replace("tensor\tlogits\t0\tf32\t2", "tensor\tlogits\t0\tf32\t3"), "vocabulary"),
                          (complete.replace("\t1\t1\t8\t-1.0\t" + LOGITS_OP, "\t2\t1\t8\t-1.0\t" + LOGITS_OP), "dimensions"),
                          (complete.replace("\t-1.0\t" + LOGITS_OP, "\t-1.5\t" + LOGITS_OP), "sum"),
                          (complete.replace(MANIFEST_FORMAT, RAW_FORMAT), "older capture")):
            mp.write_text(bad)
            try:
                bundle(root / "b")
                raise AssertionError("accepted malformed bundle")
            except ValueError as error:
                assert want in str(error), error
        mp.write_text(complete)
        # Each target states its own architecture: a Llama-architecture target's set reads, and the Qwen3.5 one's
        # `# arch` on it (or its own on a Qwen3.5 target) is refused by name.
        folder = root / "l"
        folder.mkdir()
        for name in ("prompt.ids", "forced.ids", "logits.0.f32"):
            shutil.copyfile(root / "a" / name, folder / name)
        finish_bundle(folder, {"format": MANIFEST_FORMAT, "model": "/models/selftest.gguf", "build": "same",
                               "arch": TARGETS["_selftest_llama"]["arch"], "target": "_selftest_llama",
                               "model_sha256": TARGETS["_selftest_llama"]["sha256"], "engine": "native",
                               "binary_sha256": "same", "backend": "cuda", "ctx": "8", "kv": "f16", "ubatch": "8",
                               "prompt_sha256": digest(folder / "prompt.ids"), "forced_sha256": digest(folder / "forced.ids")},
                      {"rows": "1", "vocab": "2", "mode": "eager", "prefill": "pass"})
        header, _ = bundle(folder)
        assert header["arch"] == "llama" and header["target"] == "_selftest_llama"
        llama_set = (folder / "MANIFEST.tsv").read_text()
        (folder / "MANIFEST.tsv").write_text(llama_set.replace("# arch\tllama", "# arch\tqwen35"))
        try:
            bundle(folder)
            raise AssertionError("accepted a Llama target's set that states the Qwen3.5 architecture")
        except ValueError as error:
            assert "architecture" in str(error), error
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
            signal_members(process.pid, {pid: "other start"}, signal.SIGKILL)  # reused-pid identity
            time.sleep(0.2)
            assert pid in session_members(process.pid), "signalled a pid with another start identity"
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
        # The set a writer leaves is a node dump a refset family reads: the header names the file, the build and the
        # architecture; no key is one refset types differently or takes for a column line; the row has the v2 columns;
        # the trailer is the last line.
        m = manifest.read(str(root / "a"))
        assert m.complete == "1\t0" and (root / "a/MANIFEST.tsv").read_text().splitlines()[-1] == "# complete\t1\t0"
        assert m.header["model"] == "/models/selftest.gguf" and m.header["build"] == "same" \
            and m.header["arch"] == TARGETS["_selftest"]["arch"]
        assert not {"prefill", "kind", "input", "int", "flags", "tokens"} & set(m.header), sorted(m.header)
        assert m.columns["tensor"] == ["kind", "name", "occurrence", "type", "ne0", "ne1", "ne2", "ne3", "bytes", "sum", "op",
                                       "contig", "logical", "src0", "src1"]
        row = m.rows("tensor")[0]
        assert (row["sum"], row["op"], row["contig"], row["logical"], row["src0"]) == ("-1.0", LOGITS_OP, "1", "0", "-")
        # A corpus slice is the file's first ids under its pinned digest; run() refuses a wrong call before the model is read.
        corpus_file = root / "corpus"
        corpus_file.write_text("0\n1\n0\n1\n0\n")
        corpus_sha = digest(corpus_file)
        assert corpus_slice(corpus_file, corpus_sha, 4, 2) == [0, 1, 0, 1]
        for call, want in (((corpus_file, "0" * 64, 2, 2), "pinned corpus digest"), ((corpus_file, corpus_sha, 6, 2), "holds 5 ids"),
                           ((corpus_file, corpus_sha, 4, 1), "not one decimal id")):
            try:
                corpus_slice(*call)
                raise AssertionError(f"accepted a corpus slice {call}")
            except ValueError as error:
                assert want in str(error), error

        def asked(**changes):
            call = dict(target="_selftest", kind="reference", model=root / "model", binary=root / "none", out=root / "out",
                        corpus=None, prompt=None, forced=None, prompt_count=None, forced_count=None, corpus_sha256=None,
                        ctx=8, ubatch=8, backend="cpu", prefill="batch", mode="graph", bound=None,
                        hip_max_model_bytes=16 * 1024 * 1024, reset_repeats=1)
            call.update(changes)
            run(argparse.Namespace(**call))

        for changes, want in (({"corpus": corpus_file, "prompt": root / "p", "prompt_count": 2, "forced_count": 1,
                                "corpus_sha256": corpus_sha}, "--corpus takes"),
                              ({"corpus": corpus_file, "prompt_count": 2, "corpus_sha256": corpus_sha}, "--corpus takes"),
                              ({"corpus": corpus_file, "prompt_count": -1, "forced_count": 33, "corpus_sha256": corpus_sha},
                               "--corpus takes"),
                              ({"corpus": corpus_file, "prompt_count": 3, "forced_count": -1, "corpus_sha256": corpus_sha},
                               "--corpus takes"),
                              ({}, "a run takes --prompt and --forced"),
                              ({"prompt": root / "p", "forced": root / "f", "forced_count": 1}, "a run takes --prompt and --forced"),
                              ({"prompt": root / "p", "forced": root / "f", "prompt_count": 0}, "a run takes --prompt and --forced"),
                              ({"corpus": corpus_file, "prompt_count": 3, "forced_count": 2, "corpus_sha256": corpus_sha, "ctx": 4},
                               "context does not hold")):
            try:
                asked(**changes)
                raise AssertionError(f"accepted a run call {changes}")
            except ValueError as error:
                assert want in str(error), error
        try:
            build(argparse.Namespace(pin="0" * 40, cached_libraries=True, llama_tree=root, llama_build=root))
            raise AssertionError("accepted --pin with --cached-libraries")
        except ValueError as error:
            assert "does not combine" in str(error), error
        try:
            build(argparse.Namespace(pin="0" * 40, cached_libraries=False, llama_tree=root, llama_build=root / "elsewhere"))
            raise AssertionError("accepted --pin with a build directory the proof did not read")
        except ValueError as error:
            assert "--pin requires --llama-build" in str(error), error

        # The floor: a reference CUDA set and the CPU twin beside it, per row rel_l2, the argmax and the top-2 margins.
        def make_set(folder, flat, backend):
            folder.mkdir()
            (folder / "prompt.ids").write_text("1\n")
            (folder / "forced.ids").write_text("0\n1\n")
            rows = array.array("f", flat)
            if sys.byteorder != "little":
                rows.byteswap()
            (folder / "logits.0.f32").write_bytes(rows.tobytes())
            finish_bundle(folder, {"format": MANIFEST_FORMAT, "model": "/models/selftest.gguf", "build": "53ed051ce",
                                   "arch": TARGETS["_selftest"]["arch"], "target": "_selftest",
                                   "model_sha256": TARGETS["_selftest"]["sha256"],
                                   "engine": "reference", "backend": backend, "binary_sha256": "same", "ctx": "8", "kv": "f16",
                                   "ubatch": "8", "prompt_sha256": digest(folder / "prompt.ids"),
                                   "forced_sha256": digest(folder / "forced.ids")},
                          {"rows": "2", "vocab": "2", "mode": "reference", "prefill": "batch"})

        make_set(root / "c", [3.0, 1.0, 0.5, 0.25], "cuda")
        make_set(root / "c.cpu", [3.5, 1.0, 0.25, 0.5], "cpu")
        report = floor([root / "c"])
        first, second = report["sets"][0]["rows"]
        assert abs(first["rel_l2"] - 0.5 / math.sqrt(3.0 ** 2 + 1.0)) < 1e-9 and first["max_abs"] == 0.5 and first["l2"] == 0.5
        assert abs(second["rel_l2"] - math.sqrt(0.125) / math.sqrt(0.5 ** 2 + 0.25 ** 2)) < 1e-9
        assert (first["argmax_cuda"], first["argmax_cpu"], second["argmax_cuda"], second["argmax_cpu"]) == (0, 0, 0, 1)
        assert (first["margin_cuda"], first["margin_cpu"], second["margin_cuda"], second["margin_cpu"]) == (2.0, 2.5, 0.25, 0.25)
        pooled = report["pooled"]
        assert pooled["argmax_equal"] == 1 and pooled["rows_within_bound"] == 1 and report["sets"][0]["prompt_len"] == 1
        assert abs(first["stable_margin"] - math.sqrt(2) * 0.5) < 1e-9 and abs(second["stable_margin"] - 0.5) < 1e-9
        assert pooled["margin_cuda"][0.0] == 0.25 and abs(pooled["stable_margin"][1.0] - math.sqrt(2) * 0.5) < 1e-9
        # Past two coordinates the max-component branch can be the smaller one.
        assert stable_margin(3.0, 1.0) == 2.0 and abs(stable_margin(0.5, 0.5) - math.sqrt(2) * 0.5) < 1e-12
        shown = io.StringIO()
        print_floor(report, shown)
        assert "argmax cuda == cpu: 1/2; row 1 flips" in shown.getvalue() and "at or under their own stable margin" in shown.getvalue()
        for refused, want in ((root / "a", "cannot read"), ):
            try:
                floor([refused])
                raise AssertionError("floor read a set with no twin")
            except ValueError as error:
                assert want in str(error), error
        make_set(root / "d", [3.0, 1.0, 0.5, 0.25], "cuda")
        make_set(root / "d.cpu", [3.5, 1.0, 0.25, 0.5], "cuda")
        try:
            floor([root / "d"])
            raise AssertionError("floor took a twin that ran on the card")
        except ValueError as error:
            assert "floor takes a reference CUDA set" in str(error), error
    del TARGETS["_selftest"], TARGETS["_selftest_llama"]
    print("dense-logits self-test: malformed ids, provenance, the set's family-shaped manifest, corpus slices, run and build refusals, "
          "incomplete/truncated/NaN rows, HIP placement and reserve, reset repeats, eager/graph, the oracle floor, the unpinned-target refusal "
          "and owned-session cleanup contracts passed")
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
    b.add_argument("--pin", help="the full commit hash the tree must be at: the build is proven by tools/ref/emb2/oracle-tree.sh and nothing is built in the tree")
    nb = sub.add_parser("build-native")
    nb.add_argument("--arch", default="sm_86")
    r = sub.add_parser("run")
    r.add_argument("--kind", choices=("reference", "native"), required=True)
    r.add_argument("--backend", choices=("cuda", "cpu", "hip"), default="cuda")
    r.add_argument("--target", choices=TARGETS, default="qwen38-27b")
    for key in ("model", "binary", "out"):
        r.add_argument("--" + key, type=Path, required=True)
    for key in ("prompt", "forced", "corpus"):
        r.add_argument("--" + key, type=Path)
    r.add_argument("--corpus-sha256", help="the corpus file's pinned sha256, with --corpus")
    r.add_argument("--prompt-count", type=int, help="with --corpus: the prompt is its first N ids")
    r.add_argument("--forced-count", type=int, help="with --corpus: the M ids after the prompt are forced")
    r.add_argument("--bound", type=int, help="seconds the whole call may take (default 2800 on the card, 900 on the CPU)")
    r.add_argument("--prefill", choices=("step", "batch", "pass", "auto", "gemm"), required=True)
    r.add_argument("--mode", choices=("eager", "graph"), default="graph")
    r.add_argument("--ctx", type=int, default=4352)
    r.add_argument("--ubatch", type=int, default=512)
    r.add_argument("--hip-max-model-bytes", type=int, default=16 * 1024 * 1024)
    r.add_argument("--reset-repeats", type=int, choices=(1, 2), default=1)
    c = sub.add_parser("compare")
    c.add_argument("left", type=Path)
    c.add_argument("right", type=Path)
    c.add_argument("--bitwise", action="store_true")
    i = sub.add_parser("identity", help="the size and sha256 a TARGETS entry takes for a file")
    i.add_argument("file", type=Path)
    f = sub.add_parser("floor", help="the oracle's own spread: each reference CUDA set against the CPU twin beside it")
    f.add_argument("sets", type=Path, nargs="+", help="reference CUDA set directories; the twin of S is S.cpu")
    f.add_argument("--twin-suffix", default=".cpu")
    f.add_argument("--json", type=Path, help="also write the per-row numbers here")
    args = parser.parse_args()
    try:
        if args.action == "build":
            build(args)
        elif args.action == "build-native":
            native_build(args)
        elif args.action == "run":
            run(args)
        elif args.action == "identity":
            found = target_identity(args.file)
            print(f'        "size": {found["size"]},\n        "sha256": "{found["sha256"]}",')
        elif args.action == "floor":
            report = floor(args.sets, args.twin_suffix)
            print_floor(report)
            if args.json:
                write_json(args.json, report)
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
