#!/usr/bin/env bash
# Lever check — runs on the Mac (text only, no build).
# Blocks: an environment read in crates/*/src outside the lever registry (crates/levers/src) that
# tools/levers-direct.txt does not name — a lever parsed where it is used, which is how a value the
# code does not understand became its default without a word. A read is a call of env::var or
# env::var_os (std::env::… too), or an import that brings var/var_os in bare; its variable is the
# string literal, or a same-file `const NAME: &str = "…"`. A read through a helper that takes the
# name as an argument (`flag(name)`) cannot be resolved: its file must still be listed, and each
# variable the list gives it must appear in that file as a string literal.
# Also red: a list line whose file has no read left, or whose variable the file no longer names —
# the list only shrinks, so a converted read takes its line with it.
# The registry's in-place rows and the list are held to each other by the levers crate's test
# registry_and_allow_list_agree (just gate-levers), which reads this list at build time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$ROOT" << 'PY'
import os, re, sys

root = sys.argv[1]
allow_path = os.path.join(root, 'tools', 'levers-direct.txt')
bad = []

allow = {}
with open(allow_path) as f:
    for n, line in enumerate(f, 1):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 3 or not all(c.strip() for c in cols):
            bad.append(f'tools/levers-direct.txt:{n}: not file<TAB>variables<TAB>round: {line!r}')
            continue
        allow.setdefault(cols[0], []).append((n, set(cols[1].split(','))))

CALL = re.compile(r'\benv::var(?:_os)?\s*\(\s*([^()]*?)\s*\)')
IMPORT = re.compile(r'\buse\s+std::env::(?:\{[^}]*\bvar(?:_os)?\b[^}]*\}|var(?:_os)?\b)')
CONST = re.compile(r'\bconst\s+([A-Z_][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]*)"')
LITERAL = re.compile(r'^"([^"]*)"$')

reads = {}
crates = os.path.join(root, 'crates')
for crate in sorted(os.listdir(crates)):
    src = os.path.join(crates, crate, 'src')
    if crate == 'levers' or not os.path.isdir(src):
        continue
    for dirpath, _, files in os.walk(src):
        for name in sorted(files):
            if not name.endswith('.rs'):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            text = open(path, encoding='utf-8').read()
            consts = dict(CONST.findall(text))
            for n, line in enumerate(text.split('\n'), 1):
                if line.lstrip().startswith('//'):
                    continue
                for m in CALL.finditer(line):
                    arg = m.group(1)
                    lit = LITERAL.match(arg)
                    var = lit.group(1) if lit else consts.get(arg)
                    reads.setdefault(rel, []).append((n, var, line.strip()))
                if IMPORT.search(line):
                    reads.setdefault(rel, []).append((n, None, line.strip()))

for rel in sorted(reads):
    lines = allow.get(rel)
    if lines is None:
        for n, var, code in reads[rel]:
            bad.append(f'{rel}:{n}: reads {var or "a variable"} in place and tools/levers-direct.txt '
                       f'does not list the file: {code}')
        continue
    listed = set().union(*(v for _, v in lines))
    for n, var, code in reads[rel]:
        if var is not None and var not in listed:
            bad.append(f'{rel}:{n}: reads {var} in place, which the file\'s lines of '
                       f'tools/levers-direct.txt do not list: {code}')

for rel in sorted(allow):
    path = os.path.join(root, rel)
    for n, vars_ in allow[rel]:
        if rel not in reads:
            bad.append(f'tools/levers-direct.txt:{n}: {rel} reads nothing in place any more: '
                       'remove the line')
            continue
        text = open(path, encoding='utf-8').read()
        for var in sorted(vars_):
            if f'"{var}"' not in text:
                bad.append(f'tools/levers-direct.txt:{n}: {rel} no longer names {var}: '
                           'take it off the line')

if bad:
    for b in bad:
        print(b, file=sys.stderr)
    print(f'check-levers: {len(bad)} problem(s) — a lever is parsed by the registry '
          '(crates/levers) at a binary\'s main; a read left in place is a line of '
          'tools/levers-direct.txt with the round that converts it', file=sys.stderr)
    sys.exit(1)
count = sum(len(v) for v in reads.values())
print(f'check-levers: ok ({count} reads in place in {len(reads)} files, all listed)')
PY
