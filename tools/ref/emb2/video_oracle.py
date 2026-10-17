#!/usr/bin/env python3
"""Collect pinned llama-server card and CPU-twin sets for three decoded RGB clips."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from server_vectors import sha256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--oracle", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8097)
    a = ap.parse_args()
    if not 1024 <= a.port <= 65535:
        ap.error("port must be between 1024 and 65535")
    a.out = a.out.resolve()
    a.oracle = a.oracle.resolve()
    here = Path(__file__).resolve().parent
    profile = here.parent / "models/gemma-embedding2.sh"
    values = subprocess.check_output([
        "bash", "-eu", "-c", 'source "$1"; printf "%s\\n" "$MODEL" "$MODEL_SHA256" "$MMPROJ" "$MMPROJ_SHA256" "$LCPP_PIN" "$EMBED_CONTEXT"',
        "video_oracle", str(profile),
    ], text=True).splitlines()
    model, model_sha, projector, projector_sha, pin, ctx = values
    for path, want in ((model, model_sha), (projector, projector_sha)):
        got = sha256(Path(path))
        if got != want:
            raise SystemExit(f"{path}: sha256 {got}, expected {want}")
        print("MODEL", path, got, flush=True)
    revision = subprocess.check_output(["git", "-C", str(a.oracle), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(a.oracle), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != pin or dirty:
        raise SystemExit(f"oracle must be clean at {pin}, got {revision}: {dirty}")
    server = a.oracle / "build/bin/llama-server"
    print("ORACLE", pin, flush=True)
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    for call, args in ((cuda.cuInit, (0,)), (cuda.cuDeviceGetCount, (ctypes.byref(count),))):
        status = call(*args)
        if status:
            raise SystemExit(f"video_oracle: {call.__name__} failed with CUDA status {status}")
    if count.value == 0:
        raise SystemExit("video_oracle: CUDA exposes no GPU; check CUDA_VISIBLE_DEVICES")
    cards = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        for call, args in ((cuda.cuDeviceGet, (ctypes.byref(device), ordinal)),
                           (cuda.cuDeviceGetName, (name, len(name), device))):
            status = call(*args)
            if status:
                raise SystemExit(f"video_oracle: {call.__name__} failed with CUDA status {status}")
        cards.append(name.value.decode("utf-8"))
    card = ", ".join(cards)
    subprocess.run([str(server), "--version"], check=True)
    subprocess.run(["python3", str(here / "video_clips.py"), str(a.out)], check=True)
    clips = json.loads((a.out / "clips.json").read_text())
    record = a.out / "oracle.build"
    record.write_text(f"lcpp_commit {pin}\nserver_sha256 {sha256(server)}\n")
    url = f"http://127.0.0.1:{a.port}"
    try:
        urllib.request.urlopen(url + "/health", timeout=2).close()
    except urllib.error.URLError:
        pass
    else:
        raise SystemExit(f"a server already listens at {url}")
    for cpu in (False, True):
        env = dict(os.environ, BLOOMERY_REF_BUILD=pin[:9], OMP_NUM_THREADS="8")
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        flags = ["-m", model, "--mmproj", projector, "--embeddings", "--pooling", "mean",
                 "-c", ctx, "-b", ctx, "-ub", ctx, "-np", "1", "-t", "8", "-ngl", "0" if cpu else "99",
                 "--host", "127.0.0.1", "--port", str(a.port), "--no-webui",
                 "--image-max-tokens", "140", "--video-fps", "1", "--video-timestamp-interval", "0"]
        if cpu:
            flags.append("--no-mmproj-offload")
        log = a.out / ("server.cpu.log" if cpu else "server.cuda.log")
        print("COMMAND", str(server), *flags, flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen([str(server), *flags], env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise SystemExit(f"server exited {process.returncode}, see {log}")
                    try:
                        urllib.request.urlopen(url + "/health", timeout=2).close()
                        break
                    except urllib.error.URLError:
                        time.sleep(1)
                else:
                    raise SystemExit(f"server did not become ready, see {log}")
                for clip in clips:
                    dest = a.out / clip["name"] / ("ref_gemma_embedding2_video.cpu" if cpu else "ref_gemma_embedding2_video")
                    args = ["python3", str(here / "server_vectors.py"), "--url", url, "--pid", str(process.pid),
                            "--model", model, "--model-sha256", model_sha, "--mmproj", projector,
                            "--mmproj-sha256", projector_sha, "--arch", "gemma-embedding2",
                            "--card", "cpu" if cpu else card, "--build-record", str(record),
                            "--ids-dir", str(a.out), "--frames", "--out", str(dest), "--server-flags", " ".join(flags)]
                    if cpu:
                        args.append("--cpu")
                    for index, image in enumerate(clip["images"]):
                        args += ["--image", f"frame_{index}={image}"]
                    print("CLIP", clip["name"], "cpu" if cpu else "cuda", "source indices", clip["indices"], flush=True)
                    subprocess.run(args, env=env, check=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise SystemExit("server required KILL during cleanup")
        if process.returncode != 0:
            raise SystemExit(f"server cleanup returned {process.returncode}, see {log}")
    print("VIDEO ORACLE COLLECTED", a.out, flush=True)


if __name__ == "__main__":
    main()
