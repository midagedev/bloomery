#!/usr/bin/env python3
"""The sensitivity table `bloomery-quantize --fit` reads, measured through the engine's own loss tool.

A unit is a tensor class over a layer range (docs/research/quant-fit.md). `bloomery-quantize --fit-ladder` lists every unit
of a model with the types it may take and a rules token for each; this tool takes that list as the one owner of units, types
and tokens, and measures: the base mix (every unit at its largest type) and, for every unit and every smaller type, the base
with that one unit retyped, each quantized (`bloomery-quantize`), walked by the loss tool (a command that writes ik's KLD base
file) and compared with the base's by `kld_diff`. The table's row is the mean KLD of the variant against the base; the base's
own types are anchor rows at 0.

  qfit.py measure --src=ARGS --quant-args=ARGS --fixed-rules FILE --ppl-cmd CMD --work DIR --out TABLE [options]
  qfit.py verify  --table TABLE (--budget BYTES | --budget-of NAME) --base-kld FILE [--mix NAME=RULES]... [same arguments]
  qfit.py --self-test

  --quantize BIN      the quantizer (default target/release/bloomery-quantize)
  --src=ARGS          the input part of its command line: `--hf DIR`, or the first shard (one shell word list)
  --quant-args=ARGS   every other flag every run takes (imatrix, flag types, --allow-requantize, --keep-split, --cpu);
                      both are written with `=`, since a value that begins with `--` is no argument to argparse otherwise
  --ftype TYPE        its <ftype> argument (default Q4_K_M)
  --fixed-rules FILE  the rules of what no unit holds; the units' tokens come first in every variant's rules
  --group N           layers to a unit (default 0: a class over all its layers)
  --ppl-cmd CMD       the loss tool's command, `{kld}` the file it writes; the model's first shard goes in its environment as
                      BLOOMERY_REF_MODEL (a ppl_exaone command through tools/gpu-gate.sh)
  --kld-diff BIN      default target/release/kld_diff
  --work DIR          where the variants, their rules and KLD files live; each variant's model and KLD file is removed once
                      read (`--keep` keeps them); the base's KLD file stays as DIR/base.kld
  --ahead N           variants quantized ahead of the one walked (default 1; 0 runs them one after the other)
  --dry-run           list the variants and the disk they need, run nothing but the ladder
  --resume            continue a stopped run from OUT.part (one line a finished variant under a key of the ladder, the
                      arguments and the loss tool; a key that differs, or a missing base kld, is refused); the base's
                      KLD file stays in --work for it

`measure` writes TABLE (class, layers, type, dkld, se, positions; `#` lines name how it was made). `verify` quantizes what
`--fit` picks at the budget, and every `--mix NAME=RULES` (a hand-written mix, the fixed rules not appended), walks them and
prints each one's measured KLD against the base's, with the sum the table predicted for the pick. `--budget-of NAME` takes the
byte size of the mix NAME as the budget.

Exit: 0 done; 1 a step refused, named (the step's command and the end of its output); 64 a usage error.
"""

import argparse
import glob
import hashlib
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time

TABLE_HEAD = "class\tlayers\ttype\tdkld\tse\tpositions"


class Refused(Exception):
    """A step that ended wrong: named in the message with what ran."""


# The children this tool started, so an abort signals those and no other process.
_children = []
_children_lock = threading.Lock()


def run(cmd, what, env=None, timeout=None):
    """Run cmd (a list), return its stdout; a nonzero exit or a timeout is Refused naming `what` and the output's end."""
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    with _children_lock:
        _children.append(p)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise Refused(f"{what}: no end in {timeout} s: {shlex.join(cmd)}")
    finally:
        with _children_lock:
            _children.remove(p)
    if p.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        raise Refused(f"{what}: rc={p.returncode}: {shlex.join(cmd)}\n{tail}")
    return out


def abort_children():
    with _children_lock:
        for p in _children:
            if p.poll() is None:
                p.terminate()


class Cand:
    def __init__(self, ty, nbytes, rule):
        self.ty, self.bytes, self.rule = ty, nbytes, rule


def parse_ladder(text):
    """The ladder's rows by unit (class, layers), in the order printed, and the base plan's bytes (None when unnamed)."""
    units, order, base = {}, [], None
    head = False
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.search(r"base plan (\d+) B", line)
            if m:
                base = int(m.group(1))
            continue
        if not line.strip():
            continue
        if not head:
            if line != "class\tlayers\ttype\tbytes\trule":
                raise Refused(f"the ladder's header is {line!r}")
            head = True
            continue
        f = line.split("\t")
        if len(f) != 5:
            raise Refused(f"the ladder row {line!r} has {len(f)} columns, not 5")
        key = (f[0], f[1])
        if key not in units:
            units[key] = []
            order.append(key)
        units[key].append(Cand(f[2], int(f[3]), f[4]))
    if not head:
        raise Refused("the ladder has no header row: the quantizer printed no ladder")
    return units, order, base


def top(cands):
    return max(cands, key=lambda c: c.bytes)


def variants(units, order):
    """The base, then every unit at every type below its largest: dicts with the name and the override token."""
    out = [{"name": "v00", "unit": None, "type": None, "rule": None, "bytes": None}]
    for key in order:
        t = top(units[key])
        for c in sorted(units[key], key=lambda c: -c.bytes):
            if c is t:
                continue
            out.append(
                {
                    "name": f"v{len(out):02d}",
                    "unit": key,
                    "type": c.ty,
                    "rule": c.rule,
                    "bytes": c.bytes - t.bytes,
                }
            )
    return out


def rule_tokens(text):
    return text.split()


def read_tokens(path):
    with open(path) as f:
        return rule_tokens(f.read())


def variant_rules(var, units, order, fixed):
    """Tokens of a variant: its override, every unit at its largest type, then the fixed rules."""
    toks = [] if var["rule"] is None else [var["rule"]]
    toks += [top(units[k]).rule for k in order]
    return toks + fixed


class Config:
    def __init__(self, a):
        self.quantize = a.quantize
        self.src = shlex.split(a.src)
        self.quant_args = shlex.split(a.quant_args)
        self.ftype = a.ftype
        self.fixed = a.fixed_rules
        self.group = a.group
        self.ppl_cmd = shlex.split(a.ppl_cmd)
        self.kld_diff = a.kld_diff
        self.work = a.work
        self.keep = a.keep
        self.ahead = a.ahead
        self.timeout = a.timeout

    def quant_cmd(self, out, rules, extra=()):
        return [
            self.quantize,
            *self.src,
            out,
            self.ftype,
            "--tensor-type-file",
            rules,
            *self.quant_args,
            *extra,
        ]


def get_ladder(cfg):
    out = run(
        cfg.quant_cmd(
            os.path.join(cfg.work, "ladder"),
            cfg.fixed,
            ["--fit-ladder", "--fit-group", str(cfg.group)],
        ),
        "the quantizer's --fit-ladder",
        timeout=cfg.timeout,
    )
    return parse_ladder(out)


def shard_bytes(d):
    files = glob.glob(os.path.join(d, "*.gguf"))
    return sum(os.path.getsize(f) for f in files)


def first_shard(d):
    f = glob.glob(os.path.join(d, "*-00001-of-*.gguf"))
    if len(f) != 1:
        raise Refused(f"{d} holds {len(f)} first shards, not 1")
    return f[0]


def done_line(out):
    """The quantizer's own closing line (its wall and weights/s), or a note that it printed none."""
    for l in reversed(out.splitlines()):
        if l.startswith("bloomery-quantize: done in"):
            return l.removeprefix("bloomery-quantize: ")
    return "quantizer printed no done line"


def quantize_one(cfg, name, rules_toks):
    """Write variant `name`'s model; return (its directory, its bytes)."""
    d = os.path.join(cfg.work, name)
    rules = os.path.join(cfg.work, f"{name}.rules")
    with open(rules, "w") as f:
        f.write("\n".join(rules_toks) + "\n")
    os.makedirs(d, exist_ok=True)
    out = run(
        cfg.quant_cmd(os.path.join(d, "model"), rules),
        f"quantize {name}",
        timeout=cfg.timeout,
    )
    print(f"qfit: {name} {done_line(out)}", flush=True)
    return d, shard_bytes(d)


def walk(cfg, name, d):
    """The loss tool's KLD file of model directory `d`."""
    kld = os.path.join(cfg.work, f"{name}.kld")
    if os.path.exists(kld):
        raise Refused(f"{kld} exists: a base file is never overwritten")
    env = dict(os.environ, BLOOMERY_REF_MODEL=first_shard(d))
    cmd = [w.replace("{kld}", kld) for w in cfg.ppl_cmd]
    if not any("{kld}" in w for w in cfg.ppl_cmd):
        raise Refused("--ppl-cmd names no {kld}")
    t = time.time()
    run(cmd, f"the loss tool on {name}", env=env, timeout=cfg.timeout)
    print(f"qfit: {name} walked in {time.time() - t:.0f} s", flush=True)
    if not os.path.isfile(kld):
        raise Refused(f"the loss tool wrote no {kld}")
    return kld


def diff(cfg, base, var):
    """kld_diff of the variant against the base: mean KLD, its standard error, the positions."""
    out = run([cfg.kld_diff, base, var], f"kld_diff {os.path.basename(var)}", timeout=cfg.timeout)
    m = re.search(r"mean KLD\s+([-0-9.eE+]+) ± ([-0-9.eE+]+)", out)
    n = re.search(r"compared, (\d+) positions", out)
    t = re.search(r"same top\s+([-0-9.eE+]+)", out)
    if not (m and n):
        raise Refused(f"kld_diff {var}: no mean KLD line in its output")
    return {
        "dkld": float(m.group(1)),
        "se": float(m.group(2)),
        "positions": int(n.group(1)),
        "same_top": float(t.group(1)) if t else None,
    }


def remove_tree(d):
    for f in glob.glob(os.path.join(d, "*")):
        os.remove(f)
    os.rmdir(d)


def pipeline(jobs, produce, consume, ahead):
    """`produce(job)` runs in a worker thread, `consume(job, item)` here, both in order; at most ahead + 1 items are
    produced and not yet consumed (one being consumed, `ahead` ready or being made). `ahead` 0 runs them one by one."""
    if ahead == 0:
        for j in jobs:
            consume(j, produce(j))
        return
    slots = threading.Semaphore(ahead + 1)
    ready = queue.Queue()
    stop = threading.Event()

    def worker():
        try:
            for j in jobs:
                slots.acquire()
                if stop.is_set():
                    return
                ready.put((j, produce(j), None))
        except Exception as e:  # handed to the consumer, which names it
            ready.put((None, None, e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        for _ in jobs:
            j, item, err = ready.get()
            if err is not None:
                raise err
            consume(j, item)
            slots.release()
    except BaseException:
        stop.set()
        abort_children()
        slots.release()
        t.join()
        raise
    t.join()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def measure(a):
    cfg = Config(a)
    os.makedirs(cfg.work, exist_ok=True)
    if os.path.exists(a.out):
        raise Refused(f"{a.out} exists: a table is never overwritten")
    fixed = read_tokens(cfg.fixed)
    units, order, base_bytes = get_ladder(cfg)
    if not order:
        raise Refused("the ladder holds no unit: this model has no class with two legal types")
    vs = variants(units, order)
    peak = 0
    for v in vs:
        size = (base_bytes or 0) + (v["bytes"] or 0)
        peak = max(peak, size)
        label = "base" if v["unit"] is None else f"{v['unit'][0]}@{v['unit'][1]} -> {v['type']}"
        print(f"qfit: {v['name']} {label}: file {size} B" if base_bytes else f"qfit: {v['name']} {label}", flush=True)
    print(
        f"qfit: {len(vs)} variants, {peak * (cfg.ahead + 1)} B of disk at the peak ({cfg.ahead + 1} live), under {cfg.work}",
        flush=True,
    )
    if a.dry_run:
        return
    results, t0 = {}, time.time()
    base = os.path.join(cfg.work, "base.kld")
    # The checkpoint: one line a finished variant, under a key of everything that decides its result. A run that
    # stops (a hung step, the box's reboot) continues with --resume from the variants not yet in it.
    part = a.out + ".part"
    key = hashlib.sha256(
        repr(
            ([variant_rules(v, units, order, fixed) for v in vs], cfg.src, cfg.quant_args, cfg.ftype, cfg.ppl_cmd, cfg.group)
        ).encode()
    ).hexdigest()[:16]
    done = {}
    if os.path.exists(part):
        if not a.resume:
            raise Refused(f"{part} exists: --resume continues that run, or remove it")
        lines = open(part).read().splitlines()
        if not lines or lines[0] != f"# key {key}":
            raise Refused(f"{part} is from another run (its key is not {key}): remove it, or pass the arguments it was made with")
        for ln in lines[1:]:
            f = ln.split("\t")
            done[f[0]] = {"dkld": float(f[1]), "se": float(f[2]), "positions": int(f[3])}
        if "v00" in done and not os.path.isfile(base):
            raise Refused(f"{part} holds the base, and {base} is gone: remove the checkpoint")
    elif a.resume:
        raise Refused(f"--resume: no {part}")
    else:
        with open(part, "w") as f:
            f.write(f"# key {key}\n")
    for v in vs:
        if v["name"] in done and v["unit"] is not None:
            results[(v["unit"], v["type"])] = done[v["name"]]
    todo = [v for v in vs if v["name"] not in done]
    if done:
        print(f"qfit: resuming: {len(done)} of {len(vs)} variants in {part}", flush=True)

    def produce(v):
        return quantize_one(cfg, v["name"], variant_rules(v, units, order, fixed))

    def consume(v, item):
        d, nbytes = item
        kld = walk(cfg, v["name"], d)
        if v["unit"] is None:
            os.rename(kld, base)
            r = {"dkld": 0.0, "se": 0.0, "positions": 0}
            print(f"qfit: {v['name']} base walked, {nbytes} B", flush=True)
        else:
            r = diff(cfg, base, kld)
            results[(v["unit"], v["type"])] = r
            os.remove(kld)
            print(
                f"qfit: {v['name']} {v['unit'][0]}@{v['unit'][1]} -> {v['type']}: dkld {r['dkld']:.6f} ± {r['se']:.6f} over {r['positions']} positions, {nbytes} B",
                flush=True,
            )
        with open(part, "a") as f:
            f.write(f"{v['name']}\t{r['dkld']:.9f}\t{r['se']:.9f}\t{r['positions']}\n")
        n_done = len(vs) - len(todo) + todo.index(v) + 1
        print(f"qfit: {n_done}/{len(vs)} variants, {time.time() - t0:.0f} s", flush=True)
        if not cfg.keep:
            remove_tree(d)

    try:
        pipeline(todo, produce, consume, cfg.ahead)
    except BaseException:
        if not cfg.keep:
            for f in glob.glob(os.path.join(cfg.work, "v[0-9][0-9]")):
                if os.path.isdir(f):
                    remove_tree(f)
            for f in glob.glob(os.path.join(cfg.work, "v[0-9][0-9].kld")):
                os.remove(f)
        raise
    n = {r["positions"] for r in results.values()}
    if len(n) != 1:
        raise Refused(f"the variants were scored over different positions: {sorted(n)}")
    pos = n.pop()
    version = run([cfg.quantize, "--version"], "the quantizer's --version").strip()
    with open(a.out, "w") as f:
        f.write("# qfit sensitivity table: the mean KLD (kld_diff) of the base mix with one unit retyped, against the base\n")
        f.write(f"# quantizer: {version}\n")
        f.write(f"# src: {' '.join(cfg.src)} ftype {cfg.ftype}; quant args: {' '.join(cfg.quant_args)}\n")
        f.write(f"# fixed rules: {cfg.fixed} sha256:{sha(cfg.fixed)}\n")
        f.write(f"# loss tool: {' '.join(cfg.ppl_cmd)}\n")
        f.write(f"# base: every unit at its largest type, its kld at {base}; {pos} positions; group {cfg.group}\n")
        f.write(f"# measured in {time.time() - t0:.0f} s over {len(vs)} variants\n")
        f.write(TABLE_HEAD + "\n")
        for key in order:
            t = top(units[key])
            f.write(f"{key[0]}\t{key[1]}\t{t.ty}\t0\t0\t{pos}\n")
            for c in sorted(units[key], key=lambda c: -c.bytes):
                if c is t:
                    continue
                r = results[(key, c.ty)]
                f.write(f"{key[0]}\t{key[1]}\t{c.ty}\t{r['dkld']:.9f}\t{r['se']:.9f}\t{r['positions']}\n")
    os.remove(part)
    print(f"qfit: table written to {a.out}", flush=True)


def verify(a):
    cfg = Config(a)
    os.makedirs(cfg.work, exist_ok=True)
    if not os.path.isfile(a.base_kld):
        raise Refused(f"--base-kld {a.base_kld}: no such file")
    mixes = []
    for m in a.mix:
        name, _, rules = m.partition("=")
        if not name or not os.path.isfile(rules):
            raise Refused(f"--mix {m}: NAME=RULES names an existing rules file")
        mixes.append((name, rules))
    results = {}

    def measure_mix(name, rules):
        d = os.path.join(cfg.work, f"mix-{name}")
        os.makedirs(d, exist_ok=True)
        out = run(cfg.quant_cmd(os.path.join(d, "model"), rules), f"quantize {name}", timeout=cfg.timeout)
        print(f"qfit verify: {name}: {done_line(out)}", flush=True)
        nbytes = shard_bytes(d)
        kld = walk(cfg, f"mix-{name}", d)
        r = diff(cfg, a.base_kld, kld)
        os.remove(kld)
        if not cfg.keep:
            remove_tree(d)
        results[name] = (nbytes, r)
        print(f"qfit verify: {name}: {nbytes} B, KLD against the base {r['dkld']:.6f} ± {r['se']:.6f} over {r['positions']} positions", flush=True)

    for name, rules in mixes:
        measure_mix(name, rules)
    if a.budget_of:
        if a.budget_of not in results:
            raise Refused(f"--budget-of {a.budget_of}: no such --mix")
        budget = results[a.budget_of][0]
    elif a.budget:
        budget = int(a.budget)
    else:
        raise Refused("--budget BYTES or --budget-of NAME")
    fit_rules = os.path.join(cfg.work, "fit.rules")
    if os.path.exists(fit_rules):
        os.remove(fit_rules)
    out = run(
        cfg.quant_cmd(
            os.path.join(cfg.work, "fit-plan"),
            cfg.fixed,
            ["--fit", str(budget), "--sensitivity", a.table, "--fit-out", fit_rules],
        ),
        "the quantizer's --fit",
        timeout=cfg.timeout,
    )
    print(out, end="", flush=True)
    m = re.search(r"predicted sum of the units' dkld ([-0-9.eE+]+)", out)
    if not m:
        raise Refused("--fit printed no predicted sum")
    predicted = float(m.group(1))
    measure_mix("fit", fit_rules)
    nbytes, r = results["fit"]
    ratio = r["dkld"] / predicted if predicted > 0 else float("nan")
    print(
        f"qfit verify: the fit's pick at {budget} B: file {nbytes} B, predicted sum {predicted:.6f}, measured {r['dkld']:.6f} ± {r['se']:.6f}, measured / predicted {ratio:.3f}",
        flush=True,
    )


def self_test():
    """The tool's own tests with its progress lines held back: shown only when one fails."""
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _self_test_body()
    except BaseException:
        sys.stderr.write(buf.getvalue())
        raise
    print("qfit.py self-test ok")


def _self_test_body():
    """The tool against stub programs in a temporary directory: the ladder's reading, the variants, each variant's rules, the
    table, the pipeline's overlap (the next quantize starts before the walk ends) and the refusals; no card, no box."""
    ladder = (
        "# bloomery-quantize --fit-ladder: 2 units, 0 layer(s) a unit (0: all), base plan 1000 B; one legal type: attn\n"
        "class\tlayers\ttype\tbytes\trule\n"
        "routed_gate_up\t1-4\tQ4_K\t600\t^A$=q4_k\n"
        "routed_gate_up\t1-4\tIQ2_XXS\t300\t^A$=iq2_xxs\n"
        "routed_gate_up\t1-4\tIQ1_S\t200\t^A$=iq1_s\n"
        "routed_down\t1-4\tQ4_K\t300\t^B$=q4_k\n"
        "routed_down\t1-4\tIQ1_S\t100\t^B$=iq1_s\n"
    )
    units, order, base = parse_ladder(ladder)
    assert base == 1000 and order == [("routed_gate_up", "1-4"), ("routed_down", "1-4")]
    vs = variants(units, order)
    assert [(v["name"], v["type"], v["bytes"]) for v in vs] == [
        ("v00", None, None),
        ("v01", "IQ2_XXS", -300),
        ("v02", "IQ1_S", -400),
        ("v03", "IQ1_S", -200),
    ], vs
    assert variant_rules(vs[1], units, order, ["F=q8_0"]) == ["^A$=iq2_xxs", "^A$=q4_k", "^B$=q4_k", "F=q8_0"]
    assert variant_rules(vs[0], units, order, []) == ["^A$=q4_k", "^B$=q4_k"]
    for bad in ["", "class\tlayers\n", "class\tlayers\ttype\tbytes\trule\nx\ty\n"]:
        try:
            parse_ladder(bad)
        except Refused:
            pass
        else:
            raise AssertionError(f"a ladder {bad!r} was read")

    with tempfile.TemporaryDirectory() as tmp:
        stubs = os.path.join(tmp, "stubs")
        os.makedirs(stubs)
        log = os.path.join(tmp, "events.log")

        def script(name, body):
            p = os.path.join(stubs, name)
            with open(p, "w") as f:
                f.write("#!/usr/bin/env bash\nset -eu\n" + body)
            os.chmod(p, 0o755)
            return p

        # the quantizer: --fit-ladder prints the ladder, --version a line, --fit a pick, and a write makes a shard that holds
        # the rules it was given
        quant = script(
            "quant",
            f"""case " $* " in
*" --version "*) echo 'bloomery-quantize stub'; exit 0;;
*" --fit-ladder "*) cat <<'EOF'
{ladder}EOF
exit 0;;
*" --fit "*)
  while [ $# -gt 0 ]; do case $1 in --fit-out) o=$2; shift;; esac; shift; done
  printf '%s\\n' '^A$=iq2_xxs' '^B$=q4_k' 'F=q8_0' > "$o"
  echo "fit: 1 of 3 steps; the file is 700 B; predicted sum of the units' dkld 0.050000"
  exit 0;;
esac
out=$3; rules=""
while [ $# -gt 0 ]; do case $1 in --tensor-type-file) rules=$2; shift;; esac; shift; done
n=$(basename "$(dirname "$out")")
echo "start $n $(date +%s.%N)" >> {log}
[ "${{STUB_FAIL:-}}" = "$n" ] && {{ echo "stub quantize failed" >&2; exit 3; }}
sleep 0.4
mkdir -p "$(dirname "$out")"; cp "$rules" "$out-00001-of-00001.gguf"
echo "end $n $(date +%s.%N)" >> {log}
""",
        )
        ppl = script(
            "ppl",
            f"""n=$(basename "$(dirname "$BLOOMERY_REF_MODEL")")
echo "walk $n $(date +%s.%N)" >> {log}
sleep 0.4
cp "$BLOOMERY_REF_MODEL" "$1"
echo "walked $n $(date +%s.%N)" >> {log}
""",
        )
        # the stub's KLD names the first rule of the walked mix: 0 for the base's
        kdiff = script(
            "kdiff",
            """b=$(head -1 "$2")
case "$b" in
  '^A$=iq2_xxs') k=0.05;;
  '^A$=iq1_s') k=0.2;;
  '^B$=iq1_s') k=0.12;;
  *) k=0;;
esac
echo "kld_diff: P $1 (1 chunk(s)), Q $2 (1 chunk(s)) — n_ctx 8, n_vocab 4, 1 chunk(s) compared, 40 positions"
echo "  mean KLD           $k ± 0.001000   (whole vocabulary)"
echo "  same top           99.000 ± 0.1 %   (40 of 40)"
""",
        )
        fixed = os.path.join(tmp, "fixed.rules")
        with open(fixed, "w") as f:
            f.write("F=q8_0\n")

        def args(extra=()):
            p = parser()
            return p.parse_args(
                [
                    "measure",
                    "--quantize", quant,
                    "--src=--hf /nowhere",
                    "--quant-args=--cpu",
                    "--fixed-rules", fixed,
                    "--ppl-cmd", f"{ppl} {{kld}}",
                    "--kld-diff", kdiff,
                    "--work", os.path.join(tmp, "work"),
                    "--out", os.path.join(tmp, "table.tsv"),
                    *extra,
                ]
            )

        measure(args())
        table = open(os.path.join(tmp, "table.tsv")).read().splitlines()
        rows = [l for l in table if not l.startswith("#")]
        assert rows[0] == TABLE_HEAD, rows
        assert rows[1:] == [
            "routed_gate_up\t1-4\tQ4_K\t0\t0\t40",
            "routed_gate_up\t1-4\tIQ2_XXS\t0.050000000\t0.001000000\t40",
            "routed_gate_up\t1-4\tIQ1_S\t0.200000000\t0.001000000\t40",
            "routed_down\t1-4\tQ4_K\t0\t0\t40",
            "routed_down\t1-4\tIQ1_S\t0.120000000\t0.001000000\t40",
        ], rows
        assert os.path.isfile(os.path.join(tmp, "work", "base.kld"))
        assert not glob.glob(os.path.join(tmp, "work", "v[0-9][0-9]")), "a variant's model stays"
        assert not glob.glob(os.path.join(tmp, "work", "v*.kld")), "a variant's kld stays"
        # the pipeline: the next variant's quantize starts before the walk of the one before it ends
        ev = {}
        for l in open(log):
            k, n, t = l.split()
            ev[(k, n)] = float(t)
        assert ev[("start", "v02")] < ev[("walked", "v01")], "ahead 1 overlaps the next quantize with this walk"
        # a table is never overwritten
        try:
            measure(args())
        except Refused as e:
            assert "never overwritten" in str(e), e
        else:
            raise AssertionError("a table was overwritten")

        # ahead 0: no overlap
        os.remove(os.path.join(tmp, "table.tsv"))
        os.remove(os.path.join(tmp, "work", "base.kld"))
        os.remove(log)
        measure(args(["--ahead", "0"]))
        ev = {}
        for l in open(log):
            k, n, t = l.split()
            ev[(k, n)] = float(t)
        assert ev[("start", "v02")] >= ev[("walked", "v01")], "ahead 0 runs one after the other"

        # resume: a run that stops at v03 leaves its checkpoint and the base's kld; --resume does only v03
        os.remove(os.path.join(tmp, "table.tsv"))
        os.remove(os.path.join(tmp, "work", "base.kld"))
        os.remove(log)
        os.environ["STUB_FAIL"] = "v03"
        try:
            measure(args())
        except Refused as e:
            assert "stub quantize failed" in str(e), e
        else:
            raise AssertionError("a failing quantize did not stop the run")
        finally:
            del os.environ["STUB_FAIL"]
        part = os.path.join(tmp, "table.tsv.part")
        got = open(part).read().splitlines()
        assert got[0].startswith("# key ") and [l.split("\t")[0] for l in got[1:]] == ["v00", "v01", "v02"], got
        assert os.path.isfile(os.path.join(tmp, "work", "base.kld")) and not os.path.exists(os.path.join(tmp, "table.tsv"))
        assert not glob.glob(os.path.join(tmp, "work", "v*.kld")) and not glob.glob(os.path.join(tmp, "work", "v[0-9][0-9]"))
        try:
            measure(args())
        except Refused as e:
            assert "--resume continues" in str(e), e
        else:
            raise AssertionError("a checkpoint was overwritten")
        try:
            measure(args(["--resume", "--group", "1"]))
        except Refused as e:
            assert "another run" in str(e), e
        else:
            raise AssertionError("a checkpoint of another run was continued")
        os.remove(log)
        measure(args(["--resume"]))
        assert [l.split()[1] for l in open(log) if l.startswith("start")] == ["v03"], "--resume redid a finished variant"
        assert not os.path.exists(part)
        again = open(os.path.join(tmp, "table.tsv")).read().splitlines()
        assert [l for l in again if not l.startswith("#")] == rows, "the resumed table differs from the one-run table"

        # verify: a hand mix and the fit's pick against the base, the pick's measured KLD against the table's prediction
        import contextlib
        import io

        hand = os.path.join(tmp, "hand.rules")
        with open(hand, "w") as f:
            f.write("^B$=iq1_s\nF=q8_0\n")
        vargs = parser().parse_args(
            [
                "verify",
                "--quantize", quant,
                "--src=--hf /nowhere",
                "--quant-args=--cpu",
                "--fixed-rules", fixed,
                "--ppl-cmd", f"{ppl} {{kld}}",
                "--kld-diff", kdiff,
                "--work", os.path.join(tmp, "work"),
                "--table", os.path.join(tmp, "table.tsv"),
                "--base-kld", os.path.join(tmp, "work", "base.kld"),
                "--mix", f"hand={hand}",
                "--budget-of", "hand",
            ]
        )
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            verify(vargs)
        out = sink.getvalue()
        hand_b = os.path.getsize(hand)
        fit_b = len("^A$=iq2_xxs\n^B$=q4_k\nF=q8_0\n")
        assert f"qfit verify: hand: {hand_b} B, KLD against the base 0.120000" in out, out
        assert f"the fit's pick at {hand_b} B: file {fit_b} B, predicted sum 0.050000, measured 0.050000" in out, out
        assert "measured / predicted 1.000" in out, out
        vargs.base_kld = os.path.join(tmp, "nope.kld")
        try:
            verify(vargs)
        except Refused as e:
            assert "no such file" in str(e), e
        else:
            raise AssertionError("a missing base kld was accepted")

        # a quantize that fails ends the run by name, and leaves no variant model behind
        os.remove(os.path.join(tmp, "table.tsv"))
        os.remove(os.path.join(tmp, "work", "base.kld"))
        os.environ["STUB_FAIL"] = "v02"
        try:
            measure(args())
        except Refused as e:
            assert "quantize v02: rc=3" in str(e) and "stub quantize failed" in str(e), e
        else:
            raise AssertionError("a failed quantize did not end the run")
        finally:
            del os.environ["STUB_FAIL"]
        assert not os.path.exists(os.path.join(tmp, "table.tsv"))
        assert not glob.glob(os.path.join(tmp, "work", "v[0-9][0-9]")), "a variant's model stays after a failure"


def parser():
    p = argparse.ArgumentParser(prog="qfit.py", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("measure", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--quantize", default="target/release/bloomery-quantize")
        s.add_argument("--src", required=True)
        s.add_argument("--quant-args", default="")
        s.add_argument("--ftype", default="Q4_K_M")
        s.add_argument("--fixed-rules", required=True)
        s.add_argument("--group", type=int, default=0)
        s.add_argument("--ppl-cmd", required=True)
        s.add_argument("--kld-diff", default="target/release/kld_diff")
        s.add_argument("--work", required=True)
        s.add_argument("--keep", action="store_true")
        s.add_argument("--ahead", type=int, default=1)
        s.add_argument("--timeout", type=int, default=1700)
        if name == "measure":
            s.add_argument("--out", required=True)
            s.add_argument("--dry-run", action="store_true")
            s.add_argument("--resume", action="store_true")
        else:
            s.add_argument("--table", required=True)
            s.add_argument("--budget")
            s.add_argument("--budget-of")
            s.add_argument("--base-kld", required=True)
            s.add_argument("--mix", action="append", default=[])
    return p


def main(argv):
    if argv == ["--self-test"]:
        self_test()
        return 0
    a = parser().parse_args(argv)
    if a.ahead < 0 or a.group < 0:
        print("qfit.py: --ahead and --group are counts of at least 0", file=sys.stderr)
        return 64
    try:
        {"measure": measure, "verify": verify}[a.cmd](a)
    except Refused as e:
        print(f"qfit.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
