#!/usr/bin/env python3
"""A stand-in llama-server for the depth runners' stub tests (depth-ds41-stub.sh, depth-qwen3moe-stub.sh,
depth-glm5next-stub.sh):
the process lcpp-warm.sh starts, with no model and no card.

--help lists the flags the server arms pass (less STUB_SRV_HELP_MISSING). Otherwise it logs its argv
($TMPDIR/stub-srv-argv) and pid ($TMPDIR/stub-srv-pids), a build and a device line, under `-fit on -v`
two model loads (the fit's measuring one, then the real one with two expert tensors of blk 1 on the host),
and llama_context's n_ctx line: the -c it was given, STUB_SRV_NCTX in its place;
binds 127.0.0.1 at a port the kernel picks and logs `listening on`; answers /health, and /completion with
prompt_n the ids, prompt_ms 100, prompt_per_second 10 an id, predicted_ms 50 a step, predicted_per_second
20.0, cache_n 0 and tokens 1000, 1001, ... (the second STUB_SRV_TOKEN1 in its stead) Every request appends `<number in this process> <ids> <n_predict>
<ids, comma-separated>` to $TMPDIR/stub-srv-reqs. STUB_SRV_COLD, a comma list of request numbers (1 the
warm-up, 2 the timed request, 3 a retry), adds 1000000 to the fault counter file $STUB_MAJFLT at those;
STUB_SRV_BADN answers request 2 with one predicted token too few; STUB_SRV_EXIT exits 5 before it listens;
STUB_SRV_HANG never listens. Under --spec-type (the MTP arm) each answer carries draft_n 4 and
draft_n_accepted 3 and the log a `draft acceptance` line; STUB_SRV_NODRAFT drafts nothing. A TERM ends it
at once, as llama-server's own handler does.
Under a CUDA_VISIBLE_DEVICES of two cards (the two-card mode) its device lines are ggml_cuda_init's for the
A6000 and the 3090, as the stub llama-bench's (depth-stub-cards.sh); STUB_SRV_SEE_ONE=1 makes them one card,
and STUB_SRV_XID=1 leaves an Xid 79 line of the 3090 in $TMPDIR/stub-xid, the stub kernel journal.
"""
import http.server, json, os, signal, sys, time
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
tmp = os.environ.get("TMPDIR", "/tmp")
if "--help" in sys.argv:
    for f in ("-m", "-ngl", "--n-cpu-moe", "-ncmoe", "-fa", "-t", "-ub", "-b", "-fit", "-fitt", "-v", "-np", "-ctxcp",
              "--cache-ram", "-c", "--host", "--port", "--no-op-offload", "--op-offload", "-lzm", "-ts", "--spec-type",
              "--spec-draft-n-max"):
        if f != os.environ.get("STUB_SRV_HELP_MISSING"):
            print(f"{f}, --x   stub")
    sys.exit(0)
with open(os.path.join(tmp, "stub-srv-argv"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
with open(os.path.join(tmp, "stub-srv-pids"), "a") as f:
    f.write(f"{os.getpid()}\n")
print("build: 1 (stub)", flush=True)
if "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
    if os.environ.get("STUB_SRV_SEE_ONE"):
        print("ggml_cuda_init: found 1 CUDA devices (Total VRAM: 48539 MiB):", flush=True)
        print("  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB", flush=True)
    else:
        print("ggml_cuda_init: found 2 CUDA devices (Total VRAM: 72663 MiB):", flush=True)
        print("  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB", flush=True)
        print("  Device 1: NVIDIA GeForce RTX 3090 (stub), compute capability 8.6, VMM: yes, VRAM: 24124 MiB", flush=True)
    if os.environ.get("STUB_SRV_XID"):
        with open(os.path.join(tmp, "stub-xid"), "a") as f:
            f.write("1790428806.570394 ws kernel: NVRM: Xid (PCI:0000:41:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.\n")
else:
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
ctx = sys.argv[sys.argv.index("-c") + 1] if "-c" in sys.argv else "0"
print(f"llama_context: n_ctx                 = {os.environ.get('STUB_SRV_NCTX', ctx)}", flush=True)
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
        t = {"cache_n": 0, "prompt_n": n, "prompt_ms": 100.0, "prompt_per_second": n * 10.0,
             "predicted_n": p, "predicted_ms": 50.0 * max(p - 1, 0), "predicted_per_second": 20.0 if p > 1 else 0.0}
        if "--spec-type" in sys.argv and not os.environ.get("STUB_SRV_NODRAFT"):
            t["draft_n"], t["draft_n_accepted"] = 4, 3
            print("slot print_timing: id  0 | task 0 | draft acceptance = 0.75000 (    3 accepted /     4 generated), mean len = 1.75", flush=True)
        toks = [1000 + i for i in range(p)]
        if os.environ.get("STUB_SRV_TOKEN1") and p > 1: toks[1] = int(os.environ["STUB_SRV_TOKEN1"])
        self.reply(200, {"tokens": toks, "timings": t})
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(f"srv  main: listening on http://127.0.0.1:{s.server_address[1]}", flush=True)
s.serve_forever()
