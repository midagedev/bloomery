#!/usr/bin/env python3
"""The Qwen3 prompt under Nsight Compute: one launch of one layer in a P-token prompt (the ncu q3pp form).

  q3pp.py plan <gguf> <src tree> <P> <layer> <kernel> <plan out> [--ubatch U] [--unit K] [--ctx C]
      the launch the form profiles, derived from the code: the prompt's units, the launch skip that
      reaches <kernel> in layer <layer> of GEMM ubatch K (default the last), the launch's grid and
      block, and the counters that prove it was that launch; written to <plan out> as key=value lines
  q3pp.py summary <csv> <plan> <run log>
      the proof against the profiled run, then the step's cycles and each unit's demand beside them
  q3pp.py metrics
      the --metrics list the summary reads, comma-joined: this file owns the names
  q3pp.py source <source csv> [--limit R]
      the source page (ncu-gpu.sh's <out>.source.csv, BLOOMERY_NCU_SOURCE=1, the q3pp and gemm forms): per
      global memory instruction (LDG, LDGSTS, STG, ...) of each kernel, ncu's L2 Theoretical Sectors
      Global against its Ideal, summed over the page's launches, flagged above R (default 1.05) by address,
      offset from the kernel's first instruction (the offsets tools/sass-scan.sh lists) and instruction.
      A ratio above 1 on a 16-byte copy is a layout symptom (a source or destination 32-byte sector split
      in two); on a narrower access it is the access size alone. No GPU; exit 3 when the page holds no row

The profiled command is tools/ref/depth-qwen3moe.sh's `<P>` arm, `generate_qwen3moe --tokens
<lcg_prompt P> --ctx C` with -n 1 (the smallest the binary takes) and without --time, which -n 1
refuses and the prefill path does not read. The kernel table below holds the launch order and the
proof of each kernel the form can profile; a kernel not in it is refused, and adding one means
writing its derivation there.

Derivation of the skip (gqa_prefill_flash). Every term comes from the source tree the binary was
built from (<src tree>, the directory above its target/), and a line the derivation reads that is
gone is a named refusal, not a guess:
  - before the prompt: 0 launches. Qwen3moeModel::load_full loads the module only
    (arch/qwen3moe/body.rs `Ubatch::new`, flash_gqa_prefill.rs `FlashGqaPrefill::load`); in graph mode
    capture_step and capture_prefill record graphs and execute nothing (graph.rs `Graph::capture`), and
    a captured pass attends through the decode flash (arch/qwen3moe/dispatch.rs `k.flash.enqueue_pass`,
    the gqa_flash_seg kernels), never this one.
  - in the prompt: PrefillPlan::new(P, Auto, U) (arch/qwen3moe/prefill.rs): ubatches of U, a tail of
    at most MAX_TOKENS as one pass (decode flash again), a longer tail as one more ubatch. U is
    BLOOMERY_QWEN3_UBATCH, else UBATCH = GEMM_MAX_SLOTS / N_USED (arch/qwen3moe/ubatch.rs).
  - per ubatch: one launch per layer, layers in order (ubatch.rs `Ubatch::enqueue` -> `attention` ->
    `flash.enqueue`, the only call).
  so skip = K x n_layer + layer, count 1.
The launch: grid blocks_for(t, n_kv) = ceil(t / POSITIONS) x n_kv, block (POSITIONS / 2) x 32.
Neither depends on the ubatch's first position, so the proof is the tensor pipe's count: a warp runs
128 mma.m16n8k16 (HMMA.16816.F32) per key tile it is live on — (HEAD/16)(KEY_TILE/8) for the scores,
(KEY_TILE/16)(HEAD/8) for the values — and is live on ceil(warp_hi / KEY_TILE) tiles, warp_hi the
larger live key count of its two rows (p0 + r + 1 for row r of a ubatch at position p0). The sum over
the grid depends on t and p0 both. At P <= U every layer's launch has the same t, p0 and so the same
shape and work: the counters prove the ubatch, not the layer, which rests on the zero above.

Microseconds here are not records: ncu serializes and replays the kernel (and the q3pp form lets the
clock float). The tables are counts, cycles and shares.

Exit status: 0 with the plan or the tables; 3 a named refusal (a source line, a header key, a proof
that does not hold); 64 a usage error.
"""
import csv
import heapq
import os
import re
import struct
import sys

# Stall reasons as ncu 2025.3.1 lists them for ga102 (`--query-metrics --chip ga102`,
# smsp__warp_issue_stalled_*_per_warp_active without the _pipe_ breakdowns). Each is a share of the
# active warps' cycles; together they partition them, so the list is complete on purpose.
STALLS = ("barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss", "lg_throttle",
          "long_scoreboard", "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction",
          "not_selected", "selected", "short_scoreboard", "sleeping", "tex_throttle", "wait")

# The counters the summary reads beyond the sections (SpeedOfLight, LaunchStats and Occupancy give the
# duration, the SM count, the registers and the block limits).
METRICS = (
    "sm__cycles_elapsed.avg", "sm__cycles_elapsed.avg.per_second", "sm__cycles_active.avg",
    "smsp__inst_executed.sum", "smsp__issue_active.avg", "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_issued.avg.per_cycle_active", "smsp__warps_active.avg.per_cycle_active",
    "sm__inst_executed_pipe_tensor_op_hmma.sum", "smsp__pipe_tensor_op_hmma_cycles_active.avg",
    "smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_alu.sum", "sm__inst_executed_pipe_fma.sum", "sm__inst_executed_pipe_fmaheavy.sum",
    "sm__inst_executed_pipe_fmalite.sum", "sm__inst_executed_pipe_xu.sum", "sm__inst_executed_pipe_lsu.sum",
    "sm__inst_executed_pipe_cbu.sum", "sm__inst_executed_pipe_adu.sum", "sm__inst_executed_pipe_uniform.sum",
    "smsp__pipe_alu_cycles_active.avg", "smsp__pipe_fma_cycles_active.avg",
    "smsp__pipe_fmaheavy_cycles_active.avg", "smsp__pipe_fmalite_cycles_active.avg",
    "l1tex__data_pipe_lsu_wavefronts.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
) + tuple(f"smsp__warp_issue_stalled_{s}_per_warp_active" for s in STALLS)


def refuse(msg, code=3):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def opt(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


# ---- the source tree ----

def src_const(tree, rel, pattern, what):
    """The integer a line of <tree>/<rel> states (pattern's first group, `_` separators allowed)."""
    path = os.path.join(tree, rel)
    try:
        text = open(path).read()
    except OSError as e:
        refuse(f"[derive] {path}: {e.strerror}: the derivation reads {what} there")
    m = re.search(pattern, text, re.M)
    if m is None:
        refuse(f"[derive] {path} no longer states {what} (/{pattern}/): the derivation must be re-read")
    return int(m.group(1).replace("_", "")) if m.groups() else m.group(0)


def src_count(tree, rel, pattern, want, what):
    path = os.path.join(tree, rel)
    n = len(re.findall(pattern, open(path).read()))
    if n != want:
        refuse(f"[derive] {path} holds {n} {what} (/{pattern}/), the derivation counts {want}: re-read it")


def flash_consts(tree):
    fp = "crates/gpu/src/flash_gqa_prefill.rs"
    c = dict(
        key_tile=src_const(tree, fp, r"^pub const KEY_TILE: usize = ([\d_]+);", "KEY_TILE"),
        positions=src_const(tree, fp, r"^pub const POSITIONS: usize = ([\d_]+);", "POSITIONS"),
        head=src_const(tree, "crates/gpu/src/flash_gqa.rs", r"^pub const HEAD: usize = ([\d_]+);", "HEAD"),
        group=src_const(tree, "crates/gpu/src/flash_gqa.rs", r"^pub const GROUP: usize = ([\d_]+);", "GROUP"),
        max_tokens=src_const(tree, "crates/gpu/src/arch/qwen3moe/router.rs",
                             r"^pub const MAX_TOKENS: usize = ([\d_]+);", "MAX_TOKENS"),
        n_used=src_const(tree, "crates/gpu/src/arch/qwen3moe/router.rs",
                         r"^pub const N_USED: usize = ([\d_]+);", "N_USED"),
        gemm_max_slots=src_const(tree, "crates/gpu/src/gemm.rs",
                                 r"^pub const GEMM_MAX_SLOTS: usize = ([\d_]+);", "GEMM_MAX_SLOTS"),
    )
    # The shapes the grid, the block, the tile walk and the order are written in.
    src_const(tree, fp, r"^const WARPS: usize = POSITIONS / 2;", "WARPS = POSITIONS / 2")
    src_const(tree, fp, r"^const THREADS: usize = WARPS \* 32;", "THREADS = WARPS * 32")
    src_const(tree, fp, r"t\.div_ceil\(POSITIONS\) \* n_kv", "blocks_for = ceil(t / POSITIONS) * n_kv")
    src_const(tree, fp, r"let qt = n_tiles - 1 - b / nkv;", "the deepest-first block order")
    src_const(tree, fp, r"let live = KEY_TILE(?:_U32)? \* kb < warp_hi;", "a warp's live tiles")
    src_const(tree, fp, r"let warp_hi = cnt\[0\]\.max\(cnt\[1\]\);", "warp_hi, its two rows' larger count")
    src_const(tree, "crates/gpu/src/arch/qwen3moe/ubatch.rs",
              r"^pub const UBATCH: usize = GEMM_MAX_SLOTS / N_USED;", "UBATCH = GEMM_MAX_SLOTS / N_USED")
    src_const(tree, "crates/gpu/src/arch/qwen3moe/ubatch.rs",
              r"self\.host\[o_nk \+ i\] = pos \+ 1;", "a row's live key count, its position + 1")
    src_count(tree, "crates/gpu/src/arch/qwen3moe/ubatch.rs", r"\bflash\.enqueue\(", 1,
              "prefill-flash enqueues a layer")
    # The two mma loops: QK_STEPS x KEY_NT/2 pairs, PV_STEPS x DIM_PAIRS pairs, two mma each.
    src_count(tree, fp, r"wmma::mma_m16n8k16_f32_f16\(", 4, "mma call sites (two per loop)")
    c["ubatch_max"] = c["gemm_max_slots"] // c["n_used"]
    c["warps"] = c["positions"] // 2
    c["threads"] = c["warps"] * 32
    c["hmma_per_tile"] = (c["head"] // 16) * (c["key_tile"] // 8) + (c["key_tile"] // 16) * (c["head"] // 8)
    return c


# ---- the model file ----

def gguf_meta(path, keys):
    """The values of `keys` in a GGUF header (scalars only)."""
    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            refuse(f"[derive] {path} is not a GGUF file")
        f.read(4)
        _n_t, n_kv = struct.unpack("<QQ", f.read(16))

        def string():
            (ln,) = struct.unpack("<Q", f.read(8))
            return f.read(ln)

        def value(t):
            if t in scal:
                return struct.unpack(scal[t], f.read(struct.calcsize(scal[t])))[0]
            if t == 8:
                return string()
            if t == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                if et in scal:
                    f.seek(struct.calcsize(scal[et]) * cnt, 1)
                else:
                    for _ in range(cnt):
                        value(et)
                return None
            refuse(f"[derive] {path}: a metadata value of type {t}")

        for _ in range(n_kv):
            k = string().decode()
            (t,) = struct.unpack("<I", f.read(4))
            v = value(t)
            if k in keys:
                out[k] = v
    missing = [k for k in keys if k not in out]
    if missing:
        refuse(f"[derive] {path} has no {', '.join(missing)}")
    return out


# ---- the plan ----

def prefill_plan(tokens, ub, max_tokens):
    """arch/qwen3moe/prefill.rs PrefillPlan::new(tokens, PrefillPath::Auto, ub): [(kind, tokens)]."""
    def chunked(n, w, kind):
        return [(kind, min(w, n - i * w)) for i in range((n + w - 1) // w)]
    if tokens <= max_tokens:
        return chunked(tokens, max_tokens, "pass")
    tail = tokens % ub
    steps = chunked(tokens - tail, ub, "ubatch")
    if tail == 0:
        pass
    elif tail <= max_tokens:
        steps.append(("pass", tail))
    else:
        steps.append(("ubatch", tail))
    return steps


def plan_str(steps):
    """PrefillPlan's Display: `ubatch:512x2,276 pass:1`."""
    parts = []
    for kind in ("ubatch", "pass"):
        runs = []
        for k, t in steps:
            if k != kind:
                continue
            if runs and runs[-1][0] == t:
                runs[-1][1] += 1
            else:
                runs.append([t, 1])
        if runs:
            parts.append(kind + ":" + ",".join(str(s) if n == 1 else f"{s}x{n}" for s, n in runs))
    return " ".join(parts)


def flash_walk(t, p0, n_kv, c):
    """Per block, in launch order: (key tiles walked, live warp-tiles); rows of a ubatch at p0."""
    kt, pos = c["key_tile"], c["positions"]
    n_tiles = -(-t // pos)
    out = []
    for b in range(n_tiles * n_kv):
        qt = n_tiles - 1 - b // n_kv
        t0 = qt * pos
        cnt = [p0 + r + 1 if r < t else 0 for r in range(t0, t0 + pos)]
        tiles = -(-max(cnt) // kt)
        live = sum(-(-max(cnt[2 * w], cnt[2 * w + 1]) // kt) for w in range(c["warps"]))
        out.append((tiles, live))
    return out


def list_schedule(costs, slots):
    """Blocks in launch order, each to the slot that frees first: the makespan in block-steps."""
    heap = [0] * slots
    for x in costs:
        heapq.heapreplace(heap, heap[0] + x)
    return max(heap)


def cmd_plan(argv):
    if len(argv) < 6:
        print(__doc__)
        raise SystemExit(64)
    gguf, tree, P, layer, kernel, out = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4], argv[5]
    if kernel != "gqa_prefill_flash":
        refuse(f"[derive] the form knows the launch order and the proof of gqa_prefill_flash only, got "
               f"'{kernel}': another kernel needs its own entry in tools/ref/q3pp.py (its launches before "
               f"the prompt, per layer, its grid and the counter that proves the position)", 64)
    c = flash_consts(tree)
    arch = "qwen3moe"
    meta = gguf_meta(gguf, [f"{arch}.block_count", f"{arch}.attention.head_count",
                            f"{arch}.attention.head_count_kv", f"{arch}.attention.key_length"])
    n_layer = meta[f"{arch}.block_count"]
    n_head = meta[f"{arch}.attention.head_count"]
    n_kv = meta[f"{arch}.attention.head_count_kv"]
    if meta[f"{arch}.attention.key_length"] != c["head"]:
        refuse(f"[derive] the file's key_length is {meta[f'{arch}.attention.key_length']}, the kernel's HEAD {c['head']}")
    if n_head != n_kv * c["group"]:
        refuse(f"[derive] {n_head} query heads over {n_kv} key heads, the kernel groups {c['group']}")
    U = int(opt(argv, "--ubatch", c["ubatch_max"]))
    if not 1 <= U <= c["ubatch_max"]:
        refuse(f"[derive] a ubatch of {U} tokens (the binary takes 1..={c['ubatch_max']})", 64)
    ctx = int(opt(argv, "--ctx", 0)) or None
    if not 0 <= layer < n_layer:
        refuse(f"[derive] layer {layer} of {n_layer}", 64)
    if ctx is not None and P + 1 > ctx:
        refuse(f"[derive] a {P}-token prompt and one generated token pass --ctx {ctx}", 64)
    steps = prefill_plan(P, U, c["max_tokens"])
    ubs = [i for i, (k, _) in enumerate(steps) if k == "ubatch"]
    if not ubs:
        refuse(f"[derive] a {P}-token prompt runs as {plan_str(steps)}: no GEMM ubatch, so no {kernel} launch", 64)
    unit = opt(argv, "--unit", None)
    k = len(ubs) - 1 if unit is None else int(unit)
    if not 0 <= k < len(ubs):
        refuse(f"[derive] ubatch {k} of {len(ubs)} ({plan_str(steps)})", 64)
    p0 = sum(t for _, t in steps[:ubs[k]])
    t = steps[ubs[k]][1]
    walk = flash_walk(t, p0, n_kv, c)
    grid = len(walk)
    block_steps = sum(x for x, _ in walk)
    warp_tiles = sum(w for _, w in walk)
    hmma = warp_tiles * c["hmma_per_tile"]
    skip = k * n_layer + layer
    lines = dict(kernel=kernel, P=P, ubatch=U, ctx=ctx or "", plan_line=plan_str(steps), unit=k, units=len(ubs),
                 n_layer=n_layer, n_head=n_head, n_kv=n_kv, head=c["head"], layer=layer, t=t, p0=p0,
                 skip=skip, count=1, grid=f"{grid},1,1", block=f"{c['threads']},1,1", block_steps=block_steps,
                 warp_tiles=warp_tiles, hmma_per_tile=c["hmma_per_tile"], hmma=hmma, key_tile=c["key_tile"],
                 positions=c["positions"], warps=c["warps"], costs=",".join(str(x) for x, _ in walk))
    with open(out, "w") as f:
        for key, v in lines.items():
            f.write(f"{key}={v}\n")
    print(f"[derive] source {tree}: KEY_TILE {c['key_tile']}, POSITIONS {c['positions']} (warps {c['warps']}, "
          f"threads {c['threads']}), HEAD {c['head']}, GROUP {c['group']}, MAX_TOKENS {c['max_tokens']}, "
          f"UBATCH {c['ubatch_max']} = GEMM_MAX_SLOTS {c['gemm_max_slots']} / N_USED {c['n_used']}")
    print(f"[derive] file: {n_layer} layers, {n_head} query heads over {n_kv} key heads of {c['head']}")
    print(f"[derive] P = {P}, ubatch {U}: plan={plan_str(steps)}; the profiled unit is ubatch {k} of {len(ubs)} "
          f"(t = {t} rows at positions {p0}..{p0 + t - 1})")
    print(f"[derive] skip = before the prompt 0 + ubatches before it {k} x {n_layer} layers x 1 launch + layer "
          f"{layer} = {skip}, count 1")
    print(f"[derive] grid ceil({t} / {c['positions']}) x {n_kv} = {grid}, block {c['warps']} x 32 = {c['threads']}; "
          f"key tiles walked (block-steps) {block_steps}, live warp-tiles {warp_tiles}, x {c['hmma_per_tile']} "
          f"= {hmma} HMMA (the position proof: sm__inst_executed_pipe_tensor_op_hmma.sum)")
    return 0


# ---- the summary ----

def read_plan(path):
    plan = {}
    for line in open(path):
        key, _, v = line.rstrip("\n").partition("=")
        plan[key] = v
    return plan


def read_csv(path):
    rows = list(csv.reader(open(path)))
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if hdr is None:
        refuse(f"[proof] {path} holds no ncu CSV header: no table")
    col = {n: i for i, n in enumerate(rows[hdr])}
    need = ("ID", "Kernel Name", "Block Size", "Grid Size", "Metric Name", "Metric Unit", "Metric Value")
    if any(n not in col for n in need):
        refuse(f"[proof] the CSV lacks one of {need}: no table")
    per = {}
    for r in rows[hdr + 1:]:
        if len(r) <= col["Metric Value"] or not r[col["Metric Name"]]:
            continue
        d = per.setdefault(int(r[col["ID"]]), dict(
            name=r[col["Kernel Name"]].split("(")[0],
            block=",".join(re.findall(r"\d+", r[col["Block Size"]])),
            grid=",".join(re.findall(r"\d+", r[col["Grid Size"]])), m={}))
        try:
            v = float(r[col["Metric Value"]].replace(",", ""))
        except ValueError:
            continue
        d["m"][(r[col["Metric Name"]], r[col["Metric Unit"]])] = v
    return per


def cmd_summary(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(64)
    csvp, planf, runlog = argv
    plan = read_plan(planf)
    log = open(runlog, errors="replace").read()
    bad = []
    load = re.search(r"^load .*$", log, re.M)
    if load is None:
        bad.append("the run printed no load line")
    else:
        for key, want in (("layers", plan["n_layer"]), ("ubatch", plan["ubatch"]), ("ctx", plan["ctx"]),
                          ("ubatch_attn", plan["kernel"])):
            got = re.search(rf"\b{key}=(\S+)", load.group(0))
            if got is None or (want and got.group(1) != want):
                bad.append(f"load line {key}={got and got.group(1)}, the plan's {want}")
    step0 = re.search(r"^step 0 .*plan=(.*), [0-9.]+ s, runtime value\)$", log, re.M)
    if step0 is None or step0.group(1) != plan["plan_line"]:
        bad.append(f"step 0 plan={step0 and step0.group(1)!r}, the derived {plan['plan_line']!r}")
    per = read_csv(csvp)
    if len(per) != 1:
        bad.append(f"{len(per)} profiled launches, the plan names 1 (skip {plan['skip']})")
    if bad:
        refuse("[proof] MISMATCH, no table:\n    " + "\n    ".join(bad))
    d = next(iter(per.values()))
    if (d["name"], d["grid"], d["block"]) != (plan["kernel"], plan["grid"], plan["block"]):
        refuse(f"[proof] MISMATCH, no table: profiled {d['name']} grid ({d['grid']}) block ({d['block']}), the plan's "
               f"{plan['kernel']} grid ({plan['grid']}) block ({plan['block']})")

    def met(name, unit=None):
        vs = [v for (n, u), v in d["m"].items() if n == name and (unit is None or u == unit)]
        return vs[0] if vs else None

    hmma, want = met("sm__inst_executed_pipe_tensor_op_hmma.sum"), int(plan["hmma"])
    wt = int(plan["warp_tiles"])
    if hmma is None or int(hmma) != want:
        per_tile = "?" if hmma is None else f"{hmma / wt:.3f}"
        refuse(f"[proof] MISMATCH, no table: {hmma} HMMA, the plan's {want} = {wt} live warp-tiles x "
               f"{plan['hmma_per_tile']} (t = {plan['t']} at p0 = {plan['p0']}); measured / warp-tiles = {per_tile}: "
               f"an integer other than {plan['hmma_per_tile']} is another SASS granularity, anything else another "
               f"launch")
    print(f"--- proof: load layers={plan['n_layer']} ubatch={plan['ubatch']} ctx={plan['ctx']}, step 0 plan="
          f"{plan['plan_line']}, one {plan['kernel']} launch grid ({plan['grid']}) block ({plan['block']}), "
          f"{int(hmma)} HMMA = {wt} warp-tiles x {plan['hmma_per_tile']}: ubatch {plan['unit']} (t = {plan['t']}, "
          f"p0 = {plan['p0']}), skip {plan['skip']}. The layer ({plan['layer']}) is the skip's: at one ubatch every "
          f"layer's launch has this shape and work")

    n_sm = met("launch__sm_count") or met("# SMs")
    if n_sm is None:
        refuse("[proof] the CSV has no # SMs (LaunchStats): no table")
    lim = [met(n) for n in ("Block Limit Registers", "Block Limit Shared Mem", "Block Limit Warps", "Block Limit SM")]
    lim = [v for v in lim if v is not None]
    if not lim:
        refuse("[proof] the CSV has no Block Limit rows (Occupancy): no table")
    resident = int(min(lim))
    need = ("sm__cycles_elapsed.avg", "sm__cycles_elapsed.avg.per_second", "smsp__inst_executed.sum",
            "smsp__pipe_tensor_op_hmma_cycles_active.avg")
    gone = [n for n in need if met(n) is None]
    if gone:
        refuse(f"[proof] the CSV lacks {', '.join(gone)}: no table")
    costs = [int(x) for x in plan["costs"].split(",")]
    bs = int(plan["block_steps"])
    slots = int(n_sm) * resident
    span = list_schedule(costs, slots)
    elapsed = met("sm__cycles_elapsed.avg")
    clock = met("sm__cycles_elapsed.avg.per_second")
    step = elapsed * n_sm / bs
    slot_step = elapsed / span
    print("--- the launch (cycles are SM cycles at the card's own clock; microseconds are not records)")
    dur = [(u, v) for (n, u), v in d["m"].items() if n == "Duration"]
    print(f"    clock {clock / 1e9 if clock else float('nan'):.3f} GHz (sm__cycles_elapsed.avg.per_second), elapsed "
          f"{elapsed:,.0f} cycles an SM, active {met('sm__cycles_active.avg') or float('nan'):,.0f}; duration "
          + (" ".join(f"{v:,.0f} {u}" for u, v in dur) or "not reported") + " under ncu")
    print(f"    {plan['grid'].split(',')[0]} blocks of {plan['block'].split(',')[0]} threads, {resident} resident an SM "
          f"(limits {', '.join(f'{v:.0f}' for v in lim)}) over {n_sm:.0f} SMs = {slots} slots; key tiles walked "
          f"(block-steps) {bs} = {bs / n_sm:.1f} an SM; the blocks' walks are 1..{max(costs)} tiles, deepest first, "
          f"so a list schedule over the slots ends at {span} block-steps a slot (the mean {bs / slots:.1f})")
    print(f"    step: {step:,.1f} SM cycles a block-step (elapsed x SMs / block-steps; the unit of every demand below), "
          f"{slot_step:,.1f} cycles a block-step in one slot (elapsed / makespan: {resident} blocks share an SM)")
    print(f"    achieved warps a scheduler {met('smsp__warps_active.avg.per_cycle_active') or float('nan'):.2f}, "
          f"registers {met('Registers Per Thread') or float('nan'):.0f} a thread, shared "
          f"{met('Static Shared Memory Per Block') or float('nan'):,.0f} B a block")

    dem = []

    def pipe(label, metric, cost, per_smsp=True):
        v = met(metric)
        dem.append((label, None if v is None else v * cost / ((4 if per_smsp else 1) * bs), metric))

    def busy(label, metric):
        v = met(metric)
        dem.append((label, None if v is None else v * n_sm / bs, metric))

    def share(label, metric):
        v = met(metric)
        dem.append((label, None if v is None else v / 100 * step, metric))

    pipe("tensor (HMMA x 16)", "sm__inst_executed_pipe_tensor_op_hmma.sum", 16)
    busy("tensor (pipe active)", "smsp__pipe_tensor_op_hmma_cycles_active.avg")
    pipe("issue (count)", "smsp__inst_executed.sum", 1)
    busy("issue (active)", "smsp__issue_active.avg")
    pipe("fmaheavy (count x 2)", "sm__inst_executed_pipe_fmaheavy.sum", 2)
    pipe("fmalite (count x 2)", "sm__inst_executed_pipe_fmalite.sum", 2)
    busy("fma (pipe active)", "smsp__pipe_fma_cycles_active.avg")
    pipe("alu (count x 2)", "sm__inst_executed_pipe_alu.sum", 2)
    busy("alu (pipe active)", "smsp__pipe_alu_cycles_active.avg")
    pipe("xu / MUFU (count x 8)", "sm__inst_executed_pipe_xu.sum", 8)
    pipe("lsu (count x 2, a SM)", "sm__inst_executed_pipe_lsu.sum", 2, per_smsp=False)
    pipe("L1TEX data pipe", "l1tex__data_pipe_lsu_wavefronts.sum", 1, per_smsp=False)
    pipe("  of it shared memory", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", 1, per_smsp=False)
    share("L2 -> SM (share of peak)", "lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed")
    share("DRAM (share of peak)", "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    print("--- each unit's demand a block-step, in the step's SM cycles (issue 1 a cycle a scheduler; HMMA.16816 16 "
          "cycles on a scheduler's tensor core at full rate; fmaheavy, fmalite, alu 16 lanes a scheduler; xu 4; lsu "
          "and the L1TEX data pipe as ncu's peaks; L2 and DRAM as ncu's share of their peak over the elapsed "
          "cycles; `active` rows are ncu's own busy cycles)")
    for label, v, metric in dem:
        if v is None:
            print(f"    {label:26s} {'NOT REPORTED':>10s}   {metric}")
        else:
            print(f"    {label:26s} {v:10.1f} cycles  {100 * v / step:5.1f} % of the step   {metric}")
    missing = [m for _, v, m in dem if v is None]
    top = max((x for x in dem if x[1] is not None), key=lambda x: x[1], default=None)
    if top:
        print(f"    busiest {top[0]} at {100 * top[1] / step:.1f} % of the step")
    l2 = met("lts__t_sectors_srcunit_tex_op_read.sum")
    dr, dw = met("dram__bytes_read.sum"), met("dram__bytes_write.sum")
    print(f"    bytes: L2 -> SM reads {fmt_bytes(l2 and l2 * 32)} ({fmt_bytes(l2 and l2 * 32 / bs)} a block-step), DRAM "
          f"read {fmt_bytes(dr)}, write {fmt_bytes(dw)}; shared bank conflicts "
          f"{met('l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum') or 0:,.0f}")

    inst = met("smsp__inst_executed.sum")
    t_act = met("smsp__pipe_tensor_op_hmma_cycles_active.avg")
    print("--- the two readings of the tail design (docs/research/q3tail-design-report.md 3.2): per block-step")
    if inst is not None:
        tens, rest = hmma * 16 / (4 * bs), (inst - hmma) / (4 * bs)
        print(f"    sum model  (tensor at full rate + the non-HMMA issue) {tens + rest:8.1f} cycles  "
              f"{100 * (tens + rest) / step:5.1f} % of the step")
        print(f"    max model  (the larger of the two)                    {max(tens, inst / (4 * bs)):8.1f} cycles  "
              f"{100 * max(tens, inst / (4 * bs)) / step:5.1f} % of the step")
        mix = [(k, met(f"sm__inst_executed_pipe_{k}.sum")) for k in
               ("alu", "fma", "fmaheavy", "fmalite", "xu", "lsu", "cbu", "adu", "uniform")]
        print(f"    warp-instructions a live warp-tile (prologue and epilogue included): all {inst / wt:.1f}, HMMA "
              f"{hmma / wt:.1f}, the rest {(inst - hmma) / wt:.1f}; by pipe " +
              ", ".join(f"{k} {v / wt:.1f}" for k, v in mix if v is not None))
    if t_act is not None:
        print(f"    tensor pipe active cycles an HMMA (ncu) {t_act * 4 * n_sm / hmma:.2f}: 16 is the full-rate reading, "
              f"32 the half-rate one")
    iss = met("smsp__issue_active.avg.pct_of_peak_sustained_active")
    ten = met("smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")
    print(f"    tensor pipe {ten if ten is not None else float('nan'):.1f} % of peak (active), issue "
          f"{(iss or float('nan')) / 100:.3f} a scheduler a cycle, warp-instructions a scheduler a cycle "
          f"{met('smsp__inst_issued.avg.per_cycle_active') or float('nan'):.3f}")

    st = {s: met(f"smsp__warp_issue_stalled_{s}_per_warp_active.pct") for s in STALLS}
    got = {s: v for s, v in st.items() if v is not None}
    if not got:
        refuse("[proof] no smsp__warp_issue_stalled_*_per_warp_active.pct in the CSV: no stall table")
    print(f"--- warp states (% of the active warps' cycles; listed {len(got)} of {len(STALLS)}, sum "
          f"{sum(got.values()):.1f} %): " + ", ".join(f"{s} {v:.1f}" for s, v in sorted(got.items(), key=lambda x: -x[1])))
    stalls = {s: v for s, v in got.items() if s != "selected"}
    chain = got.get("short_scoreboard", 0) + got.get("wait", 0)
    other = max((v for s, v in stalls.items() if s not in ("short_scoreboard", "wait")), default=0)
    rules = (
        ("math_pipe_throttle >= 25 %: the lockstep reading (1), FS", got.get("math_pipe_throttle", 0) >= 25,
         f"{got.get('math_pipe_throttle', float('nan')):.1f}"),
        ("short_scoreboard + wait above every other reason: the chain reading (2), FA", chain > other,
         f"{chain:.1f} against {other:.1f}"),
        ("no_instruction > 10 %: the loop body misses the instruction cache", got.get("no_instruction", 0) > 10,
         f"{got.get('no_instruction', float('nan')):.1f}"),
        ("tensor pipe >= 90 %: the full-rate reading was wrong", ten is not None and ten >= 90,
         f"{ten if ten is not None else float('nan'):.1f}"),
    )
    print("--- q3tail's tests (section 4), as numbers; the reading is the lead's")
    for text, hit, val in rules:
        print(f"    {'MEETS' if hit else 'no   '} {text} [{val}]")
    gone = [s for s in STALLS if s not in got]
    if missing or gone:
        print(f"[proof] NOT REPORTED: {', '.join(missing + [f'stall {s}' for s in gone])}: the table above is "
              f"short of them (rc 3)")
        return 3
    return 0


def cmd_source(argv):
    if not argv:
        refuse("usage: q3pp.py source <source csv> [--limit R]", 64)
    path, limit = argv[0], float(opt(argv, "--limit", "1.05"))
    try:
        rows = list(csv.reader(open(path)))
    except OSError as e:
        refuse(f"[source] {path}: {e.strerror}")
    agg, order, pages, first = {}, {}, {}, {}
    kernel, col = None, None
    for r in rows:
        if r and r[0] == "Kernel Name":
            kernel, col = r[1], None
            pages[kernel] = pages.get(kernel, 0) + 1
            continue
        if r and r[0] == "Address":
            col = {n: i for i, n in enumerate(r)}
            need = ("Source", "Access Size", "L2 Theoretical Sectors Global", "L2 Theoretical Sectors Global Ideal")
            if any(n not in col for n in need):
                refuse(f"[source] {path}: the page lacks one of {need}")
            continue
        if kernel is None or col is None or len(r) < len(col):
            continue
        first.setdefault(kernel, int(r[0], 16))
        try:
            th = float(r[col["L2 Theoretical Sectors Global"]].replace(",", ""))
            ideal = float(r[col["L2 Theoretical Sectors Global Ideal"]].replace(",", ""))
        except ValueError:
            continue
        if th <= 0:
            continue
        key = (kernel, int(r[0], 16))
        a = agg.setdefault(key, [0.0, 0.0, " ".join(r[col["Source"]].split()), r[col["Access Size"]]])
        a[0] += th
        a[1] += ideal
        order.setdefault(kernel, []).append(key)
    if not agg:
        refuse(f"[source] {path}: no global memory instruction with L2 theoretical sectors")
    print(f"[source] {path}: L2 Theoretical Sectors Global / Ideal per global memory instruction, summed over "
          f"the page's launches; FLAG above {limit:g}")
    for kernel in order:
        keys = list(dict.fromkeys(order[kernel]))
        print(f"  [{kernel}]  launch pages {pages[kernel]}")
        print(f"    {'address':16s} {'offset':>8s} {'ratio':>7s} {'theoretical':>12s} {'ideal':>12s} {'bits':>5s}  instruction")
        flagged, th_all, ideal_all = 0, 0.0, 0.0
        for key in keys:
            th, ideal, src, size = agg[key]
            ratio = th / ideal if ideal > 0 else float("inf")
            flag = ratio > limit
            flagged += flag
            th_all += th
            ideal_all += ideal
            print(f"    0x{key[1]:x} {'+0x%x' % (key[1] - first[kernel]):>8s} {ratio:7.3f} {th:12.0f} {ideal:12.0f} "
                  f"{size:>5s}  {src}{'   FLAG' if flag else ''}")
        print(f"    {len(keys)} instructions, {flagged} above {limit:g}; sectors {th_all:.0f} against ideal "
              f"{ideal_all:.0f} ({th_all / ideal_all if ideal_all else float('inf'):.3f}x)")
    return 0


def fmt_bytes(v):
    if v is None:
        return "?"
    for unit, s in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if v >= s:
            return f"{v / s:.2f} {unit}"
    return f"{v:.0f} B"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("plan", "summary", "metrics", "source"):
        print(__doc__)
        raise SystemExit(64)
    if sys.argv[1] == "metrics":
        print(",".join(METRICS))
        raise SystemExit(0)
    raise SystemExit({"plan": cmd_plan, "summary": cmd_summary, "source": cmd_source}[sys.argv[1]](sys.argv[2:]))


if __name__ == "__main__":
    main()
