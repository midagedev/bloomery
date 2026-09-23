# Plan (a) tok/s re-derived from measured inputs (v41-placement.md, 「재유도」).
# Tags: M = measured (source), D = derived here, A = assumed.
# Run: python3 docs/research/v41-placement/measured.py

# M: `just time-gpu-v41`, A6000, after B11c (lead ABAB 09-23 13:12-13:20, three B arms; graph_us_mean per
# site x launches). After B11b (11:08) these were 20.873 / 20.062 / 6.335 / 2.666 and the wkv 0.508.
TOKEN_TODAY_MS = 19.637      # token graph, 784 nodes (grouped wo_a as 8 launches per layer)
TOKEN_WOA_MS = 18.930        # same bytes with wo_a as one launch per layer, 504 nodes
SEL_MS = 6.294               # q3k/q4k `_sel`, 6 slots x 38 layers in the token graph
SHARED_MS = 2.403            # shared expert gate/up/down, 40 layers
EXP_ALL = 3_824_271_360      # M inventory: routed expert bytes of layers 2..39 at 6 slots
L01 = 2 * 6 * (2 * 5_068_800 + 8_110_080)  # layers 0,1: all six experts on the host (q5_K down)
HOST_A, GPU_A = 3.411e9, 0.633e9            # placement (a): prefix n_l = 63-64 per layer after the f16 q8_0 scales (b11c)
# M: b2h sweep, CPU lease, T = 16..32, engine group dispatches (rig-log 09-23#v41-host-leg)
HOST_GBPS = (127.7, 129.9)
# D: the B5 node table (docs/research/v41-b5-plan/nodes.py, n = 1, D = 4096). It replaced two
# assumptions of the first re-derivation: non-gemv nodes 0.9-1.4 ms and KV 0.16 ms at 195 GB/s.
NONGEMV = (1.59, 2.19)       # D: non-gemv nodes on the critical path, 1.89 ms, small kernels +-0.3 [A]
ATTN = (0.50, 0.68)          # D: 64-head attention 0.39-0.57 ms (compute-bound, 131 kFLOP/key) + merge 0.11
EXP_QUANT = 0.08             # D: the GPU experts' activation quant, 38 layers, in the host-leg shadow under R1
HC_PRE_FFN = 0.164           # D: HC_PRE(ffn) after `go`, in the host-leg shadow
ENGRAM_WKV = 0.476           # M: 2 x 238.0 us inside the dense graph; B5 runs it in layer 0's host-leg shadow
LOOKUP = 0.31                # M: engram row lookup on the step thread (v41-placement.md, engram)
JOIN = (0.7, 0.9)            # D: memop counter join, 40 layers, not hidden (b2a)
B11B = (0.0, 0.0)            # M: B11b is in the token graph above (its prize is spent)
B11B_SHARED = 0.0

r_sel = EXP_ALL / 1e9 / SEL_MS                    # GB per ms on the GPU `_sel` path
dense_today, dense_woa = TOKEN_TODAY_MS - SEL_MS, TOKEN_WOA_MS - SEL_MS


def compose(dense, host_bytes, gpu_bytes, gbps, nongemv, attn, join, b11b):
    host = host_bytes / 1e9 / gbps * 1e3
    gexp = gpu_bytes / 1e9 / r_sel + EXP_QUANT
    # R1: the critical path is the dense graph without the shared expert and the engram wkv, plus
    # the non-gemv critical nodes, attention, the joins, the lookup and the host leg.
    r1 = dense - b11b - SHARED_MS - ENGRAM_WKV + nongemv + attn + join + LOOKUP + host
    hidden = SHARED_MS + gexp + ENGRAM_WKV + HC_PRE_FFN
    assert host > hidden, "R1 assumes the host leg is the longer one in every layer"
    # serial: the shared and GPU experts no longer overlap the host leg; the wkv and HC_PRE(ffn)
    # placements do not depend on R1
    serial = r1 + SHARED_MS + gexp
    return serial, r1, host


def row(label, dense, host_bytes, gpu_bytes, levers):
    lo = compose(dense, host_bytes, gpu_bytes, HOST_GBPS[0], NONGEMV[1], ATTN[1], JOIN[1], levers[0])
    hi = compose(dense, host_bytes, gpu_bytes, HOST_GBPS[1], NONGEMV[0], ATTN[0], JOIN[0], levers[1])
    mid = compose(dense, host_bytes, gpu_bytes, sum(HOST_GBPS) / 2, sum(NONGEMV) / 2, ATTN[0],
                  sum(JOIN) / 2, sum(levers) / 2)
    f = lambda t: 1000.0 / t
    print(f"| {label} | {mid[2]:.2f} | {f(lo[0]):.1f}–{f(hi[0]):.1f} ({f(mid[0]):.1f}) | "
          f"{f(lo[1]):.1f}–{f(hi[1]):.1f} ({f(mid[1]):.1f}) |")


print(f"dense today {dense_today:.2f} ms, wo_a one launch {dense_woa:.2f} ms, sel {r_sel * 1e3:.0f} GB/s")
print("| 경우 | 호스트 ms | 직렬 tok/s | R1 tok/s |")
print("|---|---:|---|---|")
row("오늘 코드(B11c 포함), 접두 배치", dense_today, HOST_A, GPU_A, (0, 0))
row("+ B4 묶음 `wo_a` 한 런치", dense_woa, HOST_A, GPU_A, (0, 0))
for s, tag in ((0.401, "합친 정적 목록 40.1 %"), (0.489, "합친 정적 목록 48.9 %"),
               (0.576, "온라인 갱신 57.6 %(상한)"), (0.735, "온라인 갱신 73.5 %(상한)")):
    row(f"+ B12 {tag}", dense_woa, L01 + EXP_ALL * (1 - s), EXP_ALL * s, (0, 0))
