#!/usr/bin/env python3
"""Verdict diff: run the same recipes in two trees and compare what they print, noise masked.

The proof of a change that must not move a verdict ("move" and "integer-path" classes in AGENTS.md)
is that every gate prints the same lines in the base tree and in the changed one. Two raw logs never
compare equal: builds, timestamps, pids, thread ids and the parallel order of `cargo test` differ
from run to run. This drops build lines, masks those fields, sorts what is left (unless --ordered)
and diffs it, per recipe.

  verdict-diff.py run [--dry-run] --out DIR TREE_A TREE_B STEP...
  verdict-diff.py compare [--ordered] A B [STEP...]

run      runs each STEP with `just` in TREE_A and then in TREE_B, each tree on its own box directory
         (BLOOMERY_REMOTE=~/repo/<tree name>), into DIR/A/<step>.out|.rc and DIR/B/<step>.out|.rc, then
         compares. A STEP is a recipe with its arguments joined by ':' (`ptx-scan:gate_p5` runs `just
         ptx-scan gate_p5`). After a step that drives a card, both cards must still answer (nvidia-smi
         on the box); if one does not, the set stops there — no retry. A TREE is a path, or a
         directory name beside this repository's own. --dry-run prints the commands and runs nothing.
compare  A and B are two such directories (every STEP both have, or the ones named) or two log files.
         Prints one line per step, `identical` or `DIFFERS` with the differing lines, and keeps the
         masked texts in B/_compare/ (or beside file B). Exit 0 when every step is identical with
         equal exit codes, 1 otherwise.

What counts as noise is the MASKS and DROPS lists below; a verdict line is anything else. Numbers
are compared as printed except a duration with its unit, which is masked like a timestamp: a gate
that prints its own timings shows them in neither text. A source location (`file.rs:line:col`, as
in a panic message) is masked too — a change that moves code moves it. Two parallel tests that
write at once can splice a line; such a step reads DIFFERS on the spliced line, and the masked
texts beside the report show it.
"""
import difflib
import os
import re
import subprocess
import sys
import time

# Build and harness lines, dropped whole.
DROPS = [re.compile(p) for p in (
    r'^\s*(Compiling|Finished|Running|Checking|Blocking|Fresh|Doc-tests|Downloading|Downloaded|'
    r'Updating|Locking|Adding|Waiting)\b',
    r'^warning: ', r'^\s*= ', r'RUSTC-CODEGEN', r'^=+$', r'Running cargo build', r'Cargo build succeeded',
    r'^\s*$', r'^\./tools/box\.sh ', r'^[A-Z_]+=\S+ \./tools/box\.sh ', r'^\s*(-->|\||[0-9]+ \|)',
    r'^\s*\^',
    # generate_ds41's BLOOMERY_STEP_STATS lines: host-tier counters and page faults, a load reading.
    r'^stat (step|summary) ',
)]
# Fields that differ between two runs of the same binary, masked in place.
MASKS = [
    (re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?'), '<time>'),
    (re.compile(r'\b\d{2}:\d{2}:\d{2}Z?\b'), '<time>'),
    (re.compile(r'\bepoch \d+'), 'epoch <n>'),
    (re.compile(r'\b(\w*(?:_s|_ms|_us|secs|seconds))=[0-9.]+'), r'\1=<t>'),
    # The hybrid gate's load counters, which it labels a load and not a timing: they move run to run.
    (re.compile(r'\b(\w*_us_(?:mean|max|p\d+)|go_early|parks_in_service)=[0-9.]+'), r'\1=<n>'),
    (re.compile(r'\b\d+(\.\d+)?\s?(ns|us|µs|ms|s)\b'), r'<t>\2'),
    (re.compile(r'\b(pid|PID|lock-holder-pid)([ =:]+)\d+'), r'\1\2<pid>'),
    (re.compile(r'ThreadId\(\d+\)'), 'ThreadId(<n>)'),
    (re.compile(r"thread '[^']*' \(\d+\)"), "thread '<thread>' (<tid>)"),
    (re.compile(r"thread '[^']*'"), "thread '<thread>'"),
    (re.compile(r'/tmp/[^\s\'"]*'), lambda m: re.sub(r'\d{3,}', '<n>', m.group(0))),
    (re.compile(r'(/root/repo/|~/repo/)bloomery[\w.-]*'), r'\1<tree>'),
    (re.compile(r'\bsha256=[0-9a-f]{8,}'), 'sha256=<sha>'),
    (re.compile(r'(\.rs):\d+:\d+'), r'\1:<line>:<col>'),
]
GPU_STEP = re.compile(r'gpu|^ptx-scan|^sass-scan')


def normalize(text, ordered):
    out = []
    for line in text.splitlines():
        if any(d.search(line) for d in DROPS):
            continue
        for pat, rep in MASKS:
            line = pat.sub(rep, line)
        out.append(line.rstrip())
    return out if ordered else sorted(out)


def read(path):
    try:
        return open(path, errors='replace').read()
    except FileNotFoundError:
        return None


def compare_pair(name, a_text, b_text, a_rc, b_rc, keep, ordered, width=12):
    a, b = normalize(a_text, ordered), normalize(b_text, ordered)
    if keep:
        os.makedirs(keep, exist_ok=True)
        open(os.path.join(keep, f'{name}.A'), 'w').write('\n'.join(a) + '\n')
        open(os.path.join(keep, f'{name}.B'), 'w').write('\n'.join(b) + '\n')
    rc = f'rc A={a_rc} B={b_rc}'
    if a == b and a_rc == b_rc:
        print(f'{name:28s} {rc}  identical ({len(a)} lines)')
        return True
    diff = [d for d in difflib.unified_diff(a, b, lineterm='', n=0) if d[:1] in '+-' and d[:3] not in ('+++', '---')]
    what = 'DIFFERS' if a != b else 'same lines, DIFFERENT exit codes'
    print(f'{name:28s} {rc}  {what} ({len(diff)} lines)')
    for d in diff[:width]:
        print(f'    {"A" if d[0] == "-" else "B"}| {d[1:]}')
    if len(diff) > width:
        print(f'    … {len(diff) - width} more; the masked texts are in {keep}')
    return False


def compare(args):
    ordered = '--ordered' in args
    args = [a for a in args if a != '--ordered']
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    a, b, steps = args[0], args[1], args[2:]
    if os.path.isfile(a) and os.path.isfile(b):
        name = os.path.basename(b)
        ok = compare_pair(name, read(a), read(b), '-', '-', os.path.join(os.path.dirname(b), '_compare'), ordered)
        return 0 if ok else 1
    if not steps:
        have_a = {f[:-4] for f in os.listdir(a) if f.endswith('.out')}
        have_b = {f[:-4] for f in os.listdir(b) if f.endswith('.out')}
        steps = sorted(have_a | have_b)
    ok = True
    for s in steps:
        s = s.replace(':', '-')
        ta, tb = read(os.path.join(a, f'{s}.out')), read(os.path.join(b, f'{s}.out'))
        if ta is None or tb is None:
            print(f'{s:28s} missing in {"A" if ta is None else ""}{"B" if tb is None else ""}')
            ok = False
            continue
        ra = (read(os.path.join(a, f'{s}.rc')) or '?').strip()
        rb = (read(os.path.join(b, f'{s}.rc')) or '?').strip()
        ok &= compare_pair(s, ta, tb, ra, rb, os.path.join(b, '_compare'), ordered)
    return 0 if ok else 1


def tree_path(t):
    if '/' in t:
        return os.path.abspath(t)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(here), t)


def healthy(host):
    probe = "nvidia-smi --query-gpu=index,name,pstate --format=csv,noheader 2>&1; echo n=$(nvidia-smi -L 2>/dev/null | grep -c GPU)"
    out = subprocess.run(['ssh', host, probe], capture_output=True, text=True).stdout
    ok = not re.search(r'ERR!|No devices|Unable to determine', out) and 'n=2' in out
    return ok, out.strip().replace('\n', ';')


def run(args):
    dry = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']
    if '--out' not in args or len(args) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    i = args.index('--out')
    out = os.path.abspath(args[i + 1])
    rest = args[:i] + args[i + 2:]
    trees, steps = [tree_path(t) for t in rest[:2]], rest[2:]
    host = os.environ.get('BLOOMERY_BOX', 'ws')
    plan = []
    for s in steps:
        name = s.replace(':', '-')
        for arm, tree in zip('AB', trees):
            remote = f'~/repo/{os.path.basename(tree)}'
            cmd = ['just'] + s.split(':')
            plan.append((arm, tree, remote, name, cmd))
    for arm, tree, remote, name, cmd in plan:
        print(f"{arm} (cd {tree} && BLOOMERY_REMOTE='{remote}' {' '.join(cmd)}) > {out}/{arm}/{name}.out")
    if dry:
        return 0
    for t in trees:
        if not os.path.isdir(t):
            print(f'verdict-diff: no tree at {t}', file=sys.stderr)
            return 2
    for arm in 'AB':
        os.makedirs(os.path.join(out, arm), exist_ok=True)
    timeline = open(os.path.join(out, '_timeline.txt'), 'a')
    stamp = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    timeline.write(f'set start {stamp()} A={trees[0]} B={trees[1]}\n')
    for arm, tree, remote, name, cmd in plan:
        env = dict(os.environ, BLOOMERY_REMOTE=remote)
        t0 = time.time()
        with open(os.path.join(out, arm, f'{name}.out'), 'w') as log:
            rc = subprocess.run(cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        open(os.path.join(out, arm, f'{name}.rc'), 'w').write(f'{rc}\n')
        timeline.write(f'{arm} {name} rc={rc} secs={time.time() - t0:.0f} end {stamp()}\n')
        timeline.flush()
        print(f'{arm} {name} rc={rc} ({time.time() - t0:.0f} s)', flush=True)
        if GPU_STEP.search(name):
            ok, state = healthy(host)
            timeline.write(f'health after {arm} {name}: {state}\n')
            if not ok:
                timeline.write(f'STOP: card health failed after {arm} {name}\n')
                print(f'verdict-diff: STOP — a card did not answer after {arm} {name}: {state}', file=sys.stderr)
                return 3
    timeline.write(f'set end {stamp()}\n')
    return compare([os.path.join(out, 'A'), os.path.join(out, 'B')] + [s.replace(':', '-') for s in steps])


def main(argv):
    if argv[:1] == ['run']:
        return run(argv[1:])
    if argv[:1] == ['compare']:
        return compare(argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
