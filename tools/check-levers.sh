#!/usr/bin/env bash
# Lever check — runs on the Mac (text only, no build). Five holds:
#
# 1. Every environment read in the crates — crates/*/src outside the lever registry
#    (crates/levers/src), crates/*/tests and crates/*/build.rs — is a line of
#    tools/levers-direct.txt: a lever parsed where it is used is how a value the code does not
#    understand became its default without a word. A read is a call of env::var or env::var_os
#    (std::env::… too), or an import that brings var/var_os in bare. Its variable is the string
#    literal, or a same-file `const NAME: &str = "…"`; or, when the argument is a parameter of the
#    function the call sits in, that function is a helper that takes the name, and each of its calls
#    in the file is the read, its argument there the variable. A helper must be private to its file
#    (its calls elsewhere are not read), and a read or a helper call whose variable is none of these
#    is red. A list line holds only while its file reads each variable it gives — a string anywhere
#    else in the file (a message) does not count — so a converted read takes its line with it.
# 2. Every BLOOMERY_* name under tools/, in the justfile and under .cargo/ is a row of the registry
#    (crates/levers/src/registry.rs): a binary's `at_main` refuses a name no row names, so a runner's
#    new variable is red here, on the Mac, before a sitting meets it. A name ending in `_` is a glob
#    or a prefix, not a name.
# 3. A registry row that is no lever names its owner: a script under tools/ that names the variable,
#    or, with no script, a harness whose read is a line of the list.
# 4. A lever read in place in crates/*/src sits in a function something in the crates calls — `main`
#    and a `#[test]` count as reached: a registered reader nothing calls is a lever that reaches no
#    binary, so a value set for it runs the default without a word. A caller is found by name: an
#    associated function of an inherent impl as `Type::name` (or `Self::name` in its file), a method
#    as `.name(` or `Type::name`, a free function as its name anywhere but its own `fn`. By name it
#    can take another item of the same name for a caller, never miss one.
# 5. A Phase::Runtime row's setter resolves: the row names, inline, the file from the repository
#    root and the fn — a free function or a method — that changes the lever between calls of one
#    loaded model (crates/levers/src/lib.rs's Phase), and that file is in the tree and defines
#    `fn <item>` in code, not in a comment or a string. A Phase::Runtime stated any other way, a
#    file that is not there or a fn that file does not define is red, naming the row.
# The registry's in-place rows and the list are held to each other by the levers crate's test
# registry_and_allow_list_agree (just gate-levers), which reads this list at build time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$ROOT" << 'PY'
import os, re, sys

root = sys.argv[1]
allow_path = os.path.join(root, 'tools', 'levers-direct.txt')
registry_path = os.path.join(root, 'crates', 'levers', 'src', 'registry.rs')
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
IDENT = re.compile(r'^[a-z_][a-z0-9_]*$')
FN = re.compile(r'\bfn\s+([a-z_][a-z0-9_]*)\s*(?:<(?:[^<>{]|<[^<>]*>)*>)?\s*\(')


def balanced(text, open_at):
    """The text between the parenthesis at open_at and the one that closes it; string literals
    are skipped whole."""
    depth, i = 0, open_at
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == '\\' else 1
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
        i += 1
    return None


def split_args(inner):
    """Top-level comma-separated items of an argument or parameter list."""
    out, depth, cur, i = [], 0, '', 0
    while i < len(inner):
        c = inner[i]
        if c == '"':
            j = i + 1
            while j < len(inner) and inner[j] != '"':
                j += 2 if inner[j] == '\\' else 1
            cur += inner[i:j + 1]
            i = j + 1
            continue
        if c in '([{<':
            depth += 1
        elif c in ')]}>':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(cur.strip())
            cur, i = '', i + 1
            continue
        cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def line_of(text, at):
    return text.count('\n', 0, at) + 1


def commented(lines, n):
    return lines[n - 1].lstrip().startswith('//')


def name_of(arg, consts):
    lit = LITERAL.match(arg)
    if lit:
        return lit.group(1)
    return consts.get(arg)


def scan(rel, text):
    """(reads, problems): reads is [(line, variable, code)] of every resolved read."""
    lines = text.split('\n')
    consts = dict(CONST.findall(text))
    fns = [(m.start(), m.group(1), m.end() - 1) for m in FN.finditer(text)]
    reads, problems, helpers = [], [], {}
    for m in CALL.finditer(text):
        n = line_of(text, m.start())
        if commented(lines, n):
            continue
        arg = m.group(1)
        var = name_of(arg, consts)
        if var is not None:
            reads.append((n, var, lines[n - 1].strip()))
            continue
        encl = [f for f in fns if f[0] < m.start()]
        params = []
        if encl:
            start, fname, paren = encl[-1]
            params = [p.split(':')[0].strip() for p in split_args(balanced(text, paren) or '')]
        if IDENT.match(arg) and arg in params:
            sig_line = lines[line_of(text, start) - 1]
            head = sig_line[:sig_line.find('fn ')]
            if re.search(r'\bpub\b', head):
                problems.append(f'{rel}:{line_of(text, start)}: {fname} takes the variable\'s name '
                                'and is not private to its file: its calls elsewhere are not read')
            helpers[fname] = params.index(arg)
            continue
        problems.append(f'{rel}:{n}: reads a variable whose name is not a literal, a same-file '
                        f'const or the parameter of a helper: {lines[n - 1].strip()}')
    for fname, index in helpers.items():
        for m in re.finditer(r'(?<![\w.:])' + fname + r'\s*\(', text):
            n = line_of(text, m.start())
            if commented(lines, n) or re.search(r'\bfn\s+$', text[:m.start()][-8:]):
                continue
            args = split_args(balanced(text, m.end() - 1) or '')
            var = name_of(args[index], consts) if index < len(args) else None
            if var is None:
                problems.append(f'{rel}:{n}: calls {fname}, which reads the variable its argument '
                                f'names, with no literal or same-file const: {lines[n - 1].strip()}')
                continue
            reads.append((n, var, lines[n - 1].strip()))
    for n, line in enumerate(lines, 1):
        if not line.lstrip().startswith('//') and IMPORT.search(line):
            problems.append(f'{rel}:{n}: imports var/var_os bare, whose reads this check does not '
                            f'see; call env::var: {line.strip()}')
    return reads, problems


def sources():
    crates = os.path.join(root, 'crates')
    for crate in sorted(os.listdir(crates)):
        base = os.path.join(crates, crate)
        if not os.path.isdir(base):
            continue
        dirs = [d for d in ('src', 'tests') if not (crate == 'levers' and d == 'src')]
        for d in dirs:
            for dirpath, _, files in os.walk(os.path.join(base, d)):
                for name in sorted(files):
                    if name.endswith('.rs'):
                        yield os.path.join(dirpath, name)
        build = os.path.join(base, 'build.rs')
        if os.path.isfile(build):
            yield build


reads = {}
for path in sources():
    rel = os.path.relpath(path, root)
    got, problems = scan(rel, open(path, encoding='utf-8').read())
    bad.extend(problems)
    if got:
        reads[rel] = got

for rel in sorted(reads):
    lines = allow.get(rel)
    if lines is None:
        for n, var, code in reads[rel]:
            bad.append(f'{rel}:{n}: reads {var} in place and tools/levers-direct.txt does not list '
                       f'the file: {code}')
        continue
    listed = set().union(*(v for _, v in lines))
    for n, var, code in reads[rel]:
        if var not in listed:
            bad.append(f'{rel}:{n}: reads {var} in place, which the file\'s lines of '
                       f'tools/levers-direct.txt do not list: {code}')

for rel in sorted(allow):
    read_vars = {var for _, var, _ in reads.get(rel, [])}
    for n, vars_ in allow[rel]:
        if not read_vars:
            bad.append(f'tools/levers-direct.txt:{n}: {rel} reads nothing in place any more: '
                       'remove the line')
            continue
        for var in sorted(vars_ - read_vars):
            bad.append(f'tools/levers-direct.txt:{n}: {rel} no longer reads {var}: '
                       'take it off the line')

# 2. Every BLOOMERY_* name the tools, the justfile and .cargo/ give is a row.
registry = open(registry_path, encoding='utf-8').read()
rows = set(re.findall(r'"(BLOOMERY_[A-Z0-9_]+)"', registry))
# Self-test fixtures: each tool's test hands a made-up name to the code under test (spelled in two
# pieces here, or this file would give it too).
FIXTURE = 'BLOOMERY' + '_X'
FIXTURES = {('tools/recipes.py', FIXTURE), ('tools/ref/card-tests/run.sh', FIXTURE)}
NAME = re.compile(rb'BLOOMERY_[A-Z0-9_]+')
given = [os.path.join(root, 'justfile')]
for top in ('tools', '.cargo'):
    for dirpath, dirs, files in os.walk(os.path.join(root, top)):
        dirs[:] = [d for d in dirs if d != '__pycache__']  # Python's byte-code cache repeats the sources' names
        given += [os.path.join(dirpath, name) for name in sorted(files)]
names = set()
for path in given:
    rel = os.path.relpath(path, root)
    data = open(path, 'rb').read()
    for m in NAME.finditer(data):
        name = m.group(0).decode()
        if name.endswith('_') or (rel, name) in FIXTURES:
            continue
        names.add(name)
        if name not in rows:
            n = data.count(b'\n', 0, m.start()) + 1
            bad.append(f'{rel}:{n}: {name} is no row of the lever registry, and a binary refuses a '
                       'name no row names: add its row (crates/levers/src/registry.rs) or fix the name')

# 3. A row that is no lever names its owner, and the owner names it.
ENV_SHORT = re.compile(r'\b(?:path|runner)\(\s*"(BLOOMERY_[A-Z0-9_]+)",\s*(?:Some\("([^"]+)"\)|None)')
ENV_FULL = re.compile(r'name:\s*"(BLOOMERY_[A-Z0-9_]+)",(?:(?!LeverSpec \{).)*?'
                      r'site:\s*Site::Env\s*\{\s*script:\s*(?:Some\("([^"]+)"\)|None)', re.S)
listed_anywhere = set().union(*(v for lines in allow.values() for _, v in lines)) if allow else set()
env_rows = ENV_SHORT.findall(registry) + ENV_FULL.findall(registry)
for name, script in env_rows:
    if not script:
        if name not in listed_anywhere:
            bad.append(f'{registry_path[len(root) + 1:]}: {name} names a harness as its owner, and no '
                       'line of tools/levers-direct.txt reads it')
        continue
    path = os.path.join(root, 'tools', script)
    if not os.path.isfile(path) or name.encode() not in open(path, 'rb').read():
        bad.append(f'{registry_path[len(root) + 1:]}: {name} names tools/{script} as its owner, '
                   'which does not name it')

# 4. Every lever read in place in crates/*/src is reached.
lever_names = rows - {name for name, _ in env_rows}


def block_end(text, open_at):
    """The index of the brace that closes the one at open_at; strings, char literals and comments
    are skipped whole."""
    depth, i, n = 0, open_at, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == '\\' else 1
        elif c == "'":
            m = re.match(r"'(?:\\.|[^\\'])'", text[i:])
            if m:
                i += m.end()
                continue
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


IMPL = re.compile(r'\bimpl\b(?:\s*<(?:[^<>{]|<[^<>{]*>)*>)?\s+([^{;]+?)\s*(?:where\b[^{]*)?\{')


def fn_spans(text):
    """(start, end, name, takes self) of every fn with a body."""
    out = []
    for m in FN.finditer(text):
        params = balanced(text, m.end() - 1)
        if params is None:
            continue
        brace = text.find('{', m.end() - 1 + len(params) + 2)
        semi = text.find(';', m.end() - 1 + len(params) + 2)
        if brace < 0 or (0 <= semi < brace):
            continue
        first = split_args(params)[:1]
        takes_self = bool(first) and re.search(r'\bself\b', first[0]) is not None
        out.append((m.start(), block_end(text, brace), m.group(1), takes_self))
    return out


def impl_spans(text):
    """(start, end, type, is a trait impl) of every impl block."""
    out = []
    for m in IMPL.finditer(text):
        head = m.group(1)
        trait = re.search(r'\bfor\b', head) is not None
        ty = head.split(' for ')[-1].strip()
        ty = re.sub(r'<.*', '', ty).split('::')[-1].strip()
        out.append((m.start(), block_end(text, m.end() - 1), ty, trait))
    return out


def code_only(text):
    """`text` with every comment and the inside of every string and char literal blanked, lines
    and offsets kept: a name in a message or a doc is no call."""
    out, i, n = list(text), 0, len(text)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != '\n':
                out[k] = ' '
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            blank(i + 1, j)
            i = j + 1
        elif text[i] == "'" and re.match(r"'(?:\\.|[^\\'])'", text[i:]):
            j = i + re.match(r"'(?:\\.|[^\\'])'", text[i:]).end()
            blank(i + 1, j - 1)
            i = j
        else:
            i += 1
    return ''.join(out)


code_files = {}
for path in sources():
    code_files[os.path.relpath(path, root)] = open(path, encoding='utf-8').read()
code_lines = {rel: text.split('\n') for rel, text in code_files.items()}
bare = {rel: code_only(text) for rel, text in code_files.items()}


def called(pattern, same_file=None, self_pattern=None):
    rx = re.compile(pattern)
    for rel, text in bare.items():
        pats = [rx] + ([re.compile(self_pattern)] if self_pattern and rel == same_file else [])
        for p in pats:
            for m in p.finditer(text):
                if re.search(r'\bfn\s+$', text[max(0, m.start() - 8):m.start()]):
                    continue
                return True
    return False


lever_reads = 0
for rel in sorted(reads):
    if not re.match(r'crates/[^/]+/src/', rel):
        continue
    text = code_files[rel]
    fns = fn_spans(text)
    impls = impl_spans(text)
    for n, var, code in reads[rel]:
        if var not in lever_names:
            continue
        lever_reads += 1
        at = sum(len(l) + 1 for l in code_lines[rel][:n - 1])
        encl = [f for f in fns if f[0] <= at <= f[1]]
        if not encl:
            bad.append(f'{rel}:{n}: reads lever {var} outside any function: {code}')
            continue
        start, _, name, takes_self = max(encl, key=lambda f: f[0])
        head = text[:start].rstrip()
        if name == 'main' or re.search(r'#\[test\]\s*(?:#\[[^\]]*\]\s*)*$', head):
            continue
        owner = [i for i in impls if i[0] <= start <= i[1]]
        if owner:
            _, _, ty, trait = max(owner, key=lambda i: i[0])
            if trait or takes_self:
                pattern = r'(?:\.' + name + r'\s*(?:::<[^>]*>)?\s*\(|\b' + ty + r'::' + name + r'\b)'
            else:
                pattern = r'\b' + ty + r'::' + name + r'\b'
            reached = called(pattern, rel, r'\bSelf::' + name + r'\b')
            who = f'{ty}::{name}'
        else:
            reached = called(r'(?<![\w])' + name + r'\b')
            who = name
        if not reached:
            bad.append(f'{rel}:{n}: reads lever {var} in place in {who}, which nothing in crates/ '
                       'calls: the lever reaches no binary and a value set for it runs the default '
                       'without a word; call the reader, or make the row Parsed so every binary '
                       'refuses it by name until one acts on it')

# 5. A Phase::Runtime row's setter resolves: the file is in the tree and defines `fn <item>` in
#    code (not in a comment or a string), and the row's lever is named with the failure.
RUNTIME = re.compile(
    r'Phase::Runtime\s*\{\s*setter\s*:\s*Setter\s*\{\s*file\s*:\s*"([^"]*)"\s*,\s*'
    r'item\s*:\s*"([^"]*)"')
ROW_NAME = re.compile(r'\bname\s*:\s*("[^"]*"|[A-Za-z_][A-Za-z0-9_]*)')
registry_rel = os.path.relpath(registry_path, root)
consts = dict(CONST.findall(registry))
runtime_rows = 0
for m in re.finditer(r'\bPhase::Runtime\b', registry):
    n = registry.count('\n', 0, m.start()) + 1
    shape = RUNTIME.match(registry, m.start())
    if shape is None:
        bad.append(f'{registry_rel}:{n}: a Phase::Runtime stated any other way than '
                   'Phase::Runtime { setter: Setter { file: "…", item: "…" } }')
        continue
    file, item = shape.groups()
    bindings = list(ROW_NAME.finditer(registry, 0, m.start()))
    binding = bindings[-1].group(1) if bindings else None
    if binding is None:
        bad.append(f'{registry_rel}:{n}: a Phase::Runtime whose row names no lever')
        continue
    lever = binding.strip('"') if binding.startswith('"') else consts.get(binding, binding)
    setter_path = os.path.join(root, file)
    if not os.path.isfile(setter_path):
        bad.append(f'{registry_rel}:{n}: {lever} is Phase::Runtime; its setter names {file}, '
                   'which is no file under the repository root')
        continue
    if not re.search(r'\bfn\s+' + re.escape(item) + r'(?![A-Za-z0-9_])',
                     code_only(open(setter_path, encoding='utf-8').read())):
        bad.append(f'{registry_rel}:{n}: {lever} is Phase::Runtime; its setter {file} defines '
                   f'no fn {item} in code')
        continue
    runtime_rows += 1

if bad:
    for b in bad:
        print(b, file=sys.stderr)
    print(f'check-levers: {len(bad)} problem(s) — a lever is parsed by the registry '
          '(crates/levers) at a binary\'s main; a read left in place is a line of '
          'tools/levers-direct.txt with the round that converts it; every BLOOMERY_* name is a '
          'registry row', file=sys.stderr)
    sys.exit(1)
count = sum(len(v) for v in reads.values())
print(f'check-levers: ok ({count} reads in place in {len(reads)} files, all listed; '
      f'{len(names)} BLOOMERY_* names in tools/, the justfile and .cargo/, all rows; '
      f'{len(env_rows)} rows that are no lever, each named by its owner; {lever_reads} lever '
      'reads in crates/*/src, each in a function something calls; '
      f'{runtime_rows} Phase::Runtime rows, each setter a fn in the tree)')
PY
