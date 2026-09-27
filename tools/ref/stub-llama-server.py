#!/usr/bin/env python3
"""A stand-in llama-server for the depth runners' stub tests (depth-ds41-stub.sh, depth-qwen3moe-stub.sh):
the process lcpp-warm.sh starts, with no model and no card.

--help lists the flags the server arms pass (less STUB_SRV_HELP_MISSING). Otherwise it logs its argv
($TMPDIR/stub-srv-argv) and pid ($TMPDIR/stub-srv-pids), a build and a device line, and under `-fit on -v`
two model loads (the fit's measuring one, then the real one with two expert tensors of blk 1 on the host);
binds 127.0.0.1 at a port the kernel picks and logs `listening on`; answers /health, and /completion with
prompt_n the ids, prompt_ms 100, prompt_per_second 10 an id, predicted_ms 50 a step, predicted_per_second
20.0, cache_n 0 and tokens 1000, 1001, ... Every request appends `<number in this process> <ids> <n_predict>
<ids, comma-separated>` to $TMPDIR/stub-srv-reqs. STUB_SRV_COLD, a comma list of request numbers (1 the
warm-up, 2 the timed request, 3 a retry), adds 1000000 to the fault counter file $STUB_MAJFLT at those;
STUB_SRV_BADN answers request 2 with one predicted token too few; STUB_SRV_EXIT exits 5 before it listens;
STUB_SRV_HANG never listens. A TERM ends it at once, as llama-server's own handler does.
"""
import http.server, json, os, signal, sys, time
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
tmp = os.environ.get("TMPDIR", "/tmp")
if "--help" in sys.argv:
    for f in ("-m", "-ngl", "--n-cpu-moe", "-ncmoe", "-fa", "-t", "-ub", "-b", "-fit", "-fitt", "-v", "-np", "-ctxcp",
              "--cache-ram", "-c", "--host", "--port", "--no-op-offload", "-lzm"):
        if f != os.environ.get("STUB_SRV_HELP_MISSING"):
            print(f"{f}, --x   stub")
    sys.exit(0)
with open(os.path.join(tmp, "stub-srv-argv"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
with open(os.path.join(tmp, "stub-srv-pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
print("build: 1 (stub)", flush=True)
print("  Device 0: Stub Card, compute capability 0.0, VMM: yes", flush=True)
if "on" == (sys.argv[sys.argv.index("-fit") + 1] if "-fit" in sys.argv else "") and "-v" in sys.argv:
    for l in ("llama_model_loader: loaded meta data with 3 key-value pairs and 6 tensors from stub",
              "load_tensors: offloaded 3/3 layers to GPU", "load_tensors:        CUDA0 model buffer size =    12.00 MiB",
              "llama_model_loader: loaded meta data with 3 key-value pairs and 6 tensors from stub",
              "tensor blk.1.ffn_up_exps.weight (1 MiB q4_K) buffer type overridden to CPU",
              "tensor blk.1.ffn_down_exps.weight (1 MiB q4_K) buffer type overridden to CPU",
              "load_tensors: offloaded 3/3 layers to GPU", "load_tensors:        CUDA0 model buffer size =    10.00 MiB",
              "load_tensors:   CPU_Mapped model buffer size =     2.00 MiB"):
        print(l, flush=True)
if os.environ.get("STUB_SRV_EXIT"):
    sys.exit(5)
if os.environ.get("STUB_SRV_HANG"):
    time.sleep(3600)
cold = {int(x) for x in os.environ.get("STUB_SRV_COLD", "").split(",") if x}
n_req = [0]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, {"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_req[0] += 1
        with open(os.path.join(tmp, "stub-srv-reqs"), "a") as f:
            f.write(f"{n_req[0]} {len(req['prompt'])} {req['n_predict']} {','.join(map(str, req['prompt']))}\n")
        if n_req[0] in cold:
            m = os.environ["STUB_MAJFLT"]
            v = int(open(m).read())
            open(m, "w").write(f"{v + 1000000}\n")
        n, p = len(req["prompt"]), req["n_predict"]
        if os.environ.get("STUB_SRV_BADN") and n_req[0] == 2: p -= 1
        self.reply(200, {"tokens": [1000 + i for i in range(p)], "timings": {
            "cache_n": 0, "prompt_n": n, "prompt_ms": 100.0, "prompt_per_second": n * 10.0,
            "predicted_n": p, "predicted_ms": 50.0 * max(p - 1, 0), "predicted_per_second": 20.0 if p > 1 else 0.0}})
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(f"srv  main: listening on http://127.0.0.1:{s.server_address[1]}", flush=True)
s.serve_forever()
