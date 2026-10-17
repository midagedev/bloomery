#!/usr/bin/env python3
"""Capture dense-Qwen CLI/HTTP token diagnostics on the same native qwen3 seat.

Requires the immutable model and both cargo-oxide binaries stamped by dense-logits.py
build-native. This is plain-text, fixed-ID diagnostic support, not a numeric reference
or a replacement for the serving GPU gates. No timing result is computed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dense_logits", Path(__file__).with_name("dense-logits.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def ids(value, vocab):
    if not isinstance(value, list) or not value or any(type(x) is not int or not 0 <= x < vocab for x in value):
        raise ValueError("tokens must be a nonempty array of integer vocabulary IDs")
    return value


def cli_tokens(text, count, vocab):
    lines = [line[7:] for line in text.splitlines() if line.startswith("tokens ")]
    if len(lines) != 1:
        raise ValueError(f"CLI requires exactly one tokens line, got {len(lines)}")
    value = ids(json.loads(lines[0]), vocab)
    if len(value) != count:
        raise ValueError(f"CLI returned {len(value)} IDs, expected {count}")
    return value


def answer(value, count, vocab, prompt_len, slots, cache):
    if not isinstance(value, dict) or "error" in value:
        raise ValueError("completion is not a successful JSON object")
    tokens = ids(value.get("tokens"), vocab)
    if len(tokens) > count or type(value.get("tokens_predicted")) is not int or value["tokens_predicted"] != len(tokens):
        raise ValueError("completion token count differs from tokens_predicted or request bound")
    if value.get("truncated") is not False or value.get("stop") is not True:
        raise ValueError("completion was truncated or has no final stop")
    stop = value.get("stop_type")
    if stop not in ("limit", "eos") or value.get("stopped_eos") is not (stop == "eos") or value.get("stopped_limit") is not (stop == "limit") or value.get("stopped_word") is not False:
        raise ValueError("completion has an unexpected or inconsistent stop reason")
    if stop == "limit" and len(tokens) != count:
        raise ValueError("limit completion returned fewer IDs than requested")
    slot = value.get("id_slot")
    if type(slot) is not int or not 0 <= slot < slots:
        raise ValueError("completion returned an invalid slot")
    timing = value.get("timings")
    if not isinstance(timing, dict):
        raise ValueError("completion has no timings object")
    kept = timing.get("cache_n")
    if type(kept) is not int or not 0 <= kept <= prompt_len or (not cache and kept != 0):
        raise ValueError("completion returned an invalid kept-prefix count")
    return {"tokens": tokens, "stop_type": stop, "id_slot": slot, "cache_n": kept}


def agreement(actual, expected, count):
    if actual["tokens"] != expected[:len(actual["tokens"])]:
        return "mismatch"
    # HTTP stops at EOG; ignore_eos would mask that logit and change greedy selection.
    # A matching early prefix is retained but cannot pass this fixed-count diagnostic.
    return "pass" if len(actual["tokens"]) == count else "incomplete_early_eos"


def endpoint(log):
    found = re.findall(r"listening on http://([^\s]+)", log)
    if len(found) > 1:
        raise ValueError("server logged more than one listening address")
    if not found:
        return None
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]+)", found[0])
    if not match or not 0 < int(match[1]) < 65536:
        raise ValueError("server did not bind its own requested loopback address")
    return "http://" + found[0]


def request(url, route, body, folder, name, bound):
    # The wire protocol is serve_client::completion_raw: IDs, temperature 0,
    # return_tokens and cache_prompt. curl owns an absolute request deadline.
    payload = None if body is None else json.dumps(body)
    if body is not None:
        common.write_json(folder / (name + ".request.json"), body)
    argv = ["curl", "--noproxy", "*", "--silent", "--show-error", "--max-time", str(bound),
            "--max-filesize", "8388608", "--write-out", "\n%{http_code}"]
    if body is not None:
        argv += ["--header", "Content-Type: application/json", "--data-binary", "@-"]
    result = subprocess.run(argv + [url + route], input=payload, text=True, capture_output=True, timeout=bound + 5)
    (folder / (name + ".response.txt")).write_text(result.stdout)
    (folder / (name + ".curl.err")).write_text(result.stderr)
    if result.returncode:
        raise ValueError(f"{name}: curl failed ({result.returncode}); see retained logs")
    raw, separator, status = result.stdout.rpartition("\n")
    if not separator or status != "200":
        raise ValueError(f"{name}: HTTP {status or 'missing'}; see retained response")
    return raw


def completion(url, prompt, count, folder, name, bound, vocab, slots, cache=False):
    body = {"prompt": prompt, "n_predict": count, "temperature": 0, "return_tokens": True,
            "cache_prompt": cache, "stream": False, "ignore_eos": False}
    raw = request(url, "/completion", body, folder, name, bound)
    return answer(json.loads(raw), count, vocab, len(prompt), slots, cache)


def metrics(text):
    output = {}
    for key in ("n_decode_total", "n_busy_slots_per_decode", "swaps_total"):
        values = re.findall(r"^llamacpp:" + key + r" ([^\n]+)$", text, re.MULTILINE)
        if not values and key == "swaps_total":
            output[key] = None  # Only engines whose slots take turns export this metric.
            continue
        if len(values) != 1:
            raise ValueError(f"metrics requires one {key}")
        value = float(values[0])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"metrics has an invalid {key}")
        if key != "n_busy_slots_per_decode" and not value.is_integer():
            raise ValueError(f"metrics counter {key} is not an integer")
        if key == "n_busy_slots_per_decode" and value > 2:
            raise ValueError("busy-slot mean exceeds the two requested slots")
        output[key] = value
    return output


def overlap(before, after):
    rounds = after["n_decode_total"] - before["n_decode_total"]
    busy = (after["n_decode_total"] * after["n_busy_slots_per_decode"]
            - before["n_decode_total"] * before["n_busy_slots_per_decode"])
    if (before["swaps_total"] is None) != (after["swaps_total"] is None):
        raise ValueError("server changed the presence of its slot-turn metrics")
    swaps = None if before["swaps_total"] is None else after["swaps_total"] - before["swaps_total"]
    if rounds < 0 or busy < 0 or (swaps is not None and swaps < 0):
        raise ValueError("server counters decreased during the requests")
    # Rounded exported means are diagnostic evidence only, not a new occupancy gate.
    return {"decode_rounds": rounds, "busy_slots_sum_from_exported_mean": busy, "swaps": swaps,
            "concurrent_round_observed": rounds > 0 and round(busy) > rounds and swaps in (None, 0),
            "acceptance": "diagnostic_only_no_occupancy_band"}


class Server:
    def __init__(self, model, folder, env, args, slots):
        self.folder, self.env, self.args, self.slots = folder, env.copy(), args, slots
        self.env.pop("BLOOMERY_REF_MODEL", None)
        self.env.pop("BLOOMERY_GEN_SLOTS", None)  # The CLI's parsed lever is refused by the serving seat.
        self.env["BLOOMERY_GATE_BOUND"] = str(args.startup_bound + args.request_bound * 12)
        self.argv = ["bash", str(ROOT / "tools/gpu-gate.sh"), "bloomery-serve",
                     "--model", "qwen3", "-m", str(model), "--host", "127.0.0.1", "--port", "0",
                     "--parallel", str(slots), "--ctx", str(args.ctx * slots), "--cache-type-k", "f16"]
        self.process = None
        self.log = None

    def __enter__(self):
        self.folder.mkdir()
        common.write_json(self.folder / "launch.json", {"command": self.argv, "slots": self.slots,
                          "effective_environment": {key: value for key, value in self.env.items() if key.startswith("BLOOMERY_")}})
        self.log = (self.folder / "server.log").open("w")
        self.process = subprocess.Popen(self.argv, cwd=ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + self.args.startup_bound
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ValueError(f"server exited {self.process.returncode} before listening")
                self.url = endpoint((self.folder / "server.log").read_text())
                if self.url:
                    return self
                time.sleep(0.1)
            raise ValueError("server startup deadline expired")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if self.process is not None:
                common.stop_owned_process(self.process)
                self.process = None
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None

    def props(self, model):
        value = json.loads(request(self.url, "/props", None, self.folder, "props", self.args.request_bound))
        if value.get("total_slots") != self.slots or value.get("n_ctx") != self.args.ctx or value.get("model_path") != str(model):
            raise ValueError("/props does not identify the requested model, slots and per-slot context")
        engine = value.get("engine", {})
        if engine.get("name") != "bloomery" or engine.get("server_pid") not in common.session_members(self.process.pid):
            raise ValueError("/props does not identify the Bloomery process in this runner's own session")
        return value


def environment(args):
    env = os.environ.copy()
    # These diagnostics own generation policy; inherited forced rows must not replace argmax.
    for key in ("BLOOMERY_FORCE_IDS", "BLOOMERY_ROUTE_TRACE", "BLOOMERY_REF_WRITE"):
        env.pop(key, None)
    env.update({"BLOOMERY_DRAFT": "off", "BLOOMERY_GEN_SLOTS": "1", "BLOOMERY_QWEN3_KV": "f16",
                "BLOOMERY_QWEN3_UBATCH": str(args.ubatch), "BLOOMERY_GATE_BOUND": str(args.cli_bound)})
    return env


def run(args):
    target = common.TARGETS[args.target]
    model, output = args.model.resolve(), args.out.resolve()
    head = common.clean_head(ROOT)
    binaries = [ROOT / "target/release" / name for name in ("generate_qwen3moe", "bloomery-serve")]
    builds = [common.check_native_build(binary, head) for binary in binaries]
    for key in ("command", "arch", "compiler"):
        if builds[0][key] != builds[1][key]:
            raise ValueError(f"native binaries have different build {key}")
    model_stat = common.immutable_model(model, target)
    short, long = common.token_ids(args.short), common.token_ids(args.long)
    if not 1 <= len(short) <= 256 or len(long) != 4096:
        raise ValueError("requires a 1..256-ID short prompt and an exactly 4096-ID long prompt")
    if not 1 <= args.predict <= 256 or args.ctx < 4096 + args.predict or args.ctx > 65536 or not 1 <= args.ubatch <= 4096:
        raise ValueError("requires 1..256 output IDs, context holding prompt+output (at most 65536), ubatch 1..4096")
    if any(not 1 <= n <= 3600 for n in (args.cli_bound, args.startup_bound, args.request_bound)):
        raise ValueError("each deadline must be 1..3600 seconds")
    edited = long.copy()
    edit_at = len(long) - 32
    edited[edit_at] = (edited[edit_at] + 1) % target["vocab"]
    prompts = {"short": short, "long": long, "edited": edited}
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    for name, prompt in prompts.items():
        (fixtures / (name + ".ids")).write_text("".join(f"{token}\n" for token in prompt))
    env = environment(args)
    env["BLOOMERY_REF_MODEL"] = str(model)
    invocation = {"format": "bloomery-dense-serve-v1", "target": args.target, "model": str(model),
                  "model_sha256": target["sha256"], "model_stat": model_stat, "source_head": head, "builds": builds,
                  "driver_sha256": common.digest(__file__), "helper_sha256": common.digest(common.__file__),
                  "ctx_per_slot": args.ctx, "ubatch": args.ubatch, "predict": args.predict,
                  "prefill": "auto", "mode": "graph", "kv": "f16", "last_step": True,
                  "prompt_sha256": {name: common.digest(fixtures / (name + ".ids")) for name in prompts},
                  "prefix_edit_at": edit_at, "two_slot": args.two_slot,
                  "effective_environment": {key: value for key, value in env.items() if key.startswith("BLOOMERY_")},
                  "CUDA_VISIBLE_DEVICES_before_gate": env.get("CUDA_VISIBLE_DEVICES"),
                  "bounds_seconds": {"cli": args.cli_bound, "startup": args.startup_bound, "request": args.request_bound}}
    common.write_json(output / "invocation.json", invocation)
    expected = {}
    # CLI captures finish before any server starts, so only one model is resident.
    for name, prompt in prompts.items():
        argv = ["bash", str(ROOT / "tools/gpu-gate.sh"), "generate_qwen3moe", "--tokens", ",".join(map(str, prompt)),
                "-n", str(args.predict), "--ctx", str(args.ctx), "--prefill", "auto", "--mode", "graph", "--last-step"]
        common.write_json(output / (name + ".cli.command.json"), argv)
        logpath = output / (name + ".cli.log")
        with logpath.open("w") as log:
            code = common.bounded_run(argv, env, log, args.cli_bound + 10)
        if code:
            raise ValueError(f"{name} CLI exited {code}; retained incomplete output at {output}")
        expected[name] = cli_tokens(logpath.read_text(), args.predict, target["vocab"])
    checks = []
    with Server(model, output / "single", env, args, 1) as server:
        server.props(model)
        # cache_prompt=false is the existing API's full reset, including recurrent state.
        cases = (("short_fresh", "short", False), ("short_repeat_reset", "short", False),
                 ("long_fresh", "long", False), ("long_repeat_reset", "long", False),
                 ("edited_cached", "edited", True), ("edited_fresh", "edited", False),
                 ("short_after_long_reset", "short", False))
        for name, prompt_name, cache in cases:
            got = completion(server.url, prompts[prompt_name], args.predict, server.folder, name,
                             args.request_bound, target["vocab"], 1, cache)
            result = {"case": name, "status": agreement(got, expected[prompt_name], args.predict), **got}
            if cache:
                if got["cache_n"] > edit_at:
                    raise ValueError("edited request reused IDs past its common prefix")
                result["cache_reuse"] = "observed" if got["cache_n"] > 0 else "not_observed"
                if not got["cache_n"] and result["status"] == "pass":
                    result["status"] = "incomplete_no_prefix_reuse"
            checks.append(result)
    concurrent = None
    if args.two_slot:
        with Server(model, output / "two", env, args, 2) as server:
            server.props(model)
            selected = ("short", "edited")
            alone = [completion(server.url, prompts[name], args.predict, server.folder, name + "_alone",
                                args.request_bound, target["vocab"], 2) for name in selected]
            before = metrics(request(server.url, "/metrics", None, server.folder, "metrics_before", args.request_bound))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(completion, server.url, prompts[name], args.predict, server.folder,
                                       name + "_together", args.request_bound, target["vocab"], 2) for name in selected]
                together = [future.result() for future in futures]
            after = metrics(request(server.url, "/metrics", None, server.folder, "metrics_after", args.request_bound))
            concurrent = {"before": before, "after": after, **overlap(before, after),
                          "distinct_slots": together[0]["id_slot"] != together[1]["id_slot"]}
            for name, baseline, got in zip(selected, alone, together):
                checks.append({"case": name + "_two_slot_alone_cli", "status": agreement(baseline, expected[name], args.predict), **baseline})
                checks.append({"case": name + "_together_alone", "status": agreement(got, baseline["tokens"], args.predict), **got})
            if not concurrent["distinct_slots"] or not concurrent["concurrent_round_observed"]:
                checks.append({"case": "resident_overlap", "status": "incomplete_overlap_not_observed"})
    if common.clean_head(ROOT) != head or common.digest(__file__) != invocation["driver_sha256"] or common.digest(common.__file__) != invocation["helper_sha256"]:
        raise ValueError("source or diagnostic tooling changed during capture")
    common.model_unchanged(model, model_stat)
    for binary, built in zip(binaries, builds):
        if common.check_native_build(binary, head) != built:
            raise ValueError("native binary or build stamp changed during capture")
    result = {"complete": True, "format": "bloomery-dense-serve-v1", "checks": checks,
              "two_slot_observation": concurrent, "acceptance": "pass" if all(c["status"] == "pass" for c in checks) else "not_passed",
              "scope": "same-seat plain-text fixed-ID diagnostics; no numeric accuracy or performance claim"}
    common.write_json(output / "results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["acceptance"] == "pass" else 1


def self_test():
    valid = {"tokens": [4, 5], "tokens_predicted": 2, "truncated": False, "stop": True,
             "stop_type": "limit", "stopped_limit": True, "stopped_eos": False, "stopped_word": False,
             "id_slot": 0, "timings": {"cache_n": 0}}
    parsed = answer(valid, 2, 10, 4, 1, False)
    assert agreement(parsed, [4, 5], 2) == "pass"
    assert agreement(parsed, [4, 6], 2) == "mismatch"
    early = dict(valid, tokens=[4], tokens_predicted=1, stop_type="eos", stopped_eos=True, stopped_limit=False)
    assert agreement(answer(early, 2, 10, 4, 1, False), [4, 5], 2) == "incomplete_early_eos"
    for patch in ({"tokens": [True, 5]}, {"tokens": [4, 10]}, {"tokens": []}, {"tokens_predicted": 1},
                  {"truncated": True}, {"stop": False}, {"stop_type": "word"}, {"stopped_eos": True},
                  {"id_slot": 1}, {"timings": {"cache_n": 1}}, {"timings": []}, {"error": "engine fault"}):
        try:
            answer(dict(valid, **patch), 2, 10, 4, 1, False)
            raise AssertionError(f"accepted invalid completion: {patch}")
        except ValueError:
            pass
    assert cli_tokens("noise\ntokens [4, 5]\n", 2, 10) == [4, 5]
    for text in ("", "tokens [4]\n", "tokens [true, 5]\n", "tokens [4, 5] slot=0\n", "tokens [4, 5]\ntokens [4, 5]\n"):
        try:
            cli_tokens(text, 2, 10)
            raise AssertionError("accepted malformed CLI tokens")
        except ValueError:
            pass
    assert endpoint("loading\nlistening on http://127.0.0.1:12345\n") == "http://127.0.0.1:12345"
    for text in ("listening on http://0.0.0.0:12345", "listening on http://127.0.0.1:0",
                 "listening on http://127.0.0.1:99999", "listening on http://127.0.0.1:4\nlistening on http://127.0.0.1:5"):
        try:
            endpoint(text)
            raise AssertionError("accepted foreign/ambiguous server address")
        except ValueError:
            pass
    base = metrics("llamacpp:n_decode_total 2\nllamacpp:n_busy_slots_per_decode 1\n")
    together = metrics("llamacpp:n_decode_total 4\nllamacpp:n_busy_slots_per_decode 1.5\n")
    assert overlap(base, together)["concurrent_round_observed"]
    assert not overlap(base, base)["concurrent_round_observed"]
    for bad in ("", "llamacpp:n_decode_total NaN\nllamacpp:n_busy_slots_per_decode 1\n"):
        try:
            metrics(bad)
            raise AssertionError("accepted invalid metrics")
        except ValueError:
            pass
    transport_self_test(valid)
    print("dense-serve self-test: strict tokens, HTTP errors/truncation/EOG, reset counts, loopback and resident metrics contracts passed")
    return 0


def transport_self_test(valid):
    """Exercise the actual curl/body/status path against a bounded host-only HTTP fixture."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(valid if self.path == "/completion" else {"error": "fixture engine error"}).encode()
            self.send_response(200 if self.path == "/completion" else 500)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                got = completion(url, [1, 2, 3, 4], 2, folder, "good", 2, 10, 1)
                assert got["tokens"] == [4, 5]
                assert received == [{"prompt": [1, 2, 3, 4], "n_predict": 2, "temperature": 0,
                                     "return_tokens": True, "cache_prompt": False, "stream": False, "ignore_eos": False}]
                try:
                    request(url, "/error", {}, folder, "bad", 2)
                    raise AssertionError("accepted HTTP engine error")
                except ValueError as error:
                    assert "HTTP 500" in str(error)
            finally:
                server.shutdown()
                thread.join(timeout=3)
                assert not thread.is_alive()


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=common.TARGETS, default="qwen38-27b")
    for name in ("model", "short", "long", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=4352, help="per-slot context; server receives this times its slots")
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--two-slot", action="store_true", help="also compare two resident requests together versus alone")
    parser.add_argument("--cli-bound", type=int, default=900)
    parser.add_argument("--startup-bound", type=int, default=300)
    parser.add_argument("--request-bound", type=int, default=120)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"dense-serve: {error}; incomplete evidence has no results.json completion marker", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
