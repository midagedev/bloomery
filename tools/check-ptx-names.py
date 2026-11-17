#!/usr/bin/env python3
"""The ptx-shapes name check as a static check (Mac, text only, no build).

    tools/check-ptx-names.py check [--root DIR]
    tools/check-ptx-names.py --self-test

tools/ptx-spill-check.sh holds every entry a pinned binary carries to the row of tools/ref/ptx-shapes.tsv, but
only on the box, after the pinned binaries are built and scanned: a kernel deleted without its row, or added
without one, passed every Mac check and went red inside a landing batch (train0211i, 2026-10-11, carried R5u's
first commit without its pin commit: ds41_ffn_card_acc_10 unpinned, ds41_ffn_places_10_cols, q38_card_acc and
qwen3moe_places stale). This check reads the same two sets from source:

  rows      the table's `<binary> <entry> <spill> <jit_local>` lines;
  declared  the entries the binary's device crates declare, which are the PTX entry names (the table's entries
            are the `fn` names: `_8`, `_10`, `_10_cols` are part of the name, no suffix is generated).

and fails a row no source declares (`stale`) and a declared entry with no row (`unpinned`), per binary:

  <bin> <entry> stale                         (in the table, declared by none of the binary's crates)
  <bin> <entry> unpinned (<file>:<line>)      (declared, not in the table)

A row is `<bin> <entry> <spill> <jit_local>` and may carry driver exceptions after them, `@<driver>=<jit_local>`, under a
`# jit-driver: <version>...` header line (tools/ptx-spill-check.sh's header says what they pin); a malformed, unlisted, repeated or
plain-valued exception is a refusal, so a contributor on any driver sees it here, not at the card.

Every table the recipe names, one per PTX arch and card capability (an argument that ends in `/` is every
`*.tsv` in that directory, and one with none is a refusal), is checked for stale and unpinned entries:

  <bin> <entry> stale (<table>)               (in that table, declared by none of the binary's crates)
  <bin> <entry> unpinned (<file>:<line>; <table>) (declared for that target, not in that table)

A target-only module may instead be pinned in a target table with a `# cfg: NAME` line,
matching its sibling mod.rs or lib.rs's `#[cfg(NAME)] mod name;` declaration. Its target's
table requires those entries and every unconditional entry; the other tables require only
their own target entries and the unconditional ones. Resource values come from a scan on each target card.

What declares an entry. A `#[kernel]` attribute (its attribute stack may span lines: `#[allow(..)]`,
`#[launch_bounds(..)]`, `#[launch_contract(..)]` and doc lines sit between it and the `fn`) on a line of its own,
not in a comment; the entry is the `fn` that follows. The one generated family is a `macro_rules!` whose kernel
is `fn $name`: each invocation binds the names on its `KEY: a, b, c,` line positionally to the macro's own
pattern line `KEY: $x:ident, $y:ident, $z:ident,` (`sigmoid_router_instance!`, `entries:`: the glm5next and
mimo2 routers). A macro that repeats a group (`quant_sel_module!`, one line a type) matches the same pattern
line inside the repetition, and its invocation repeats the `KEY:` line: every line binds its own names. A `$name` the macro's pattern does not bind, a `#[kernel]` that no `fn` follows,
a kernel under a `#[cfg(..)]` attribute or inside a `#[cfg(test)]` module, and a binary or recipe this file does
not know are each a named refusal, never a skipped entry.

Which tables, binaries and crates. The tables and binaries come from the `gate-ptx-spill` recipe's `ptx-spill-check.sh
tools/ref/ptx-shapes.tsv [<table>...] <bin>[:<substring>]...` call (the one owner of what is pinned); BIN_CRATES
names the device crates each links under the features the recipe builds it with (`generate_ds41`: gpu + deepseek41,
`gate_qwen3vl_tower`: vision = gpu + gpu-vision). A substring after `:` is ptx-scan's own entry
filter: the binary's expected entries are the crates' kernels that hold it. A binary in the recipe or the
table that BIN_CRATES lacks is a refusal that says what to add.

The covered binary. A binary the recipe does not scan, whose kernels the scanned ones carry, is pinned by the premise that
it carries the same device code (COVERED: `gate_e2e`, whose entries are `generate_ds41`'s bloomery-gpu bundle again). The
premise is derived, not listed: from the manifests, by a small resolver
of cargo's feature rules over the workspace's path crates (`dep:`, `crate/feature`, optional dependencies, default
features; a weak `crate?/feature` is a refusal), for the covered binary under its `[[bin]]` `required-features` and for every
scanned binary under the `--features` of its `cargo oxide build` in the recipe. A device crate is a workspace crate whose
source has a `#[kernel]` or `#[cuda_module]` line, or invokes a `macro_rules!` whose body has one. It fails:

  <bin> <crate> not scanned (...)            the covered binary links a device crate no unfiltered scanned binary links (a
                                             binary scanned for the entries holding a substring pins only those entries)
  <bin> <crate> features differ (...)        every scanned binary that links it gives the crate other features
  <bin> <file>:<line> declares a kernel ...  a `#[kernel]` or `#[cuda_module]` in the covered binary's own sources (its
                                             bin file and the modules it names with `mod x;` / `#[path]`), which no
                                             scanned binary compiles

What it cannot see (the box scan still does):
  - which crates a recipe's feature set links, for the scanned binaries' own pins: BIN_CRATES is a map written beside the
    recipe; a feature change that links one more kernel crate into a pinned binary shows on the box as `unpinned` (the
    covered binary's guard above does read the manifests);
  - the covered binary's guard compares the workspace crates' features, not those of an external crate (cuda-*, serde_json),
    code behind `#[cfg(..)]` outside a kernel's own attributes, an `include!` (refused where it names a source of the
    covered binary), or a build of the covered binary with more features than its `required-features` (the recipe never
    scanned one);
  - a macro other than the one bound by a `KEY:` line, a `#[macro_export]` macro invoked from another crate, a
    kernel written by a proc macro (all named refusals where the text shows them, otherwise invisible);
  - the spill and jit_local values (the ratchet's own job) and whether the compiler emits an entry.

Exit: 0 every binary's rows equal its declared entries; 1 a violation or a refusal (printed, one line each, every
binary read before the verdict); 2 a usage error or an unreadable input.
"""

import glob
import os
import re
import shutil
import sys
import tempfile
from recipes import rust_lex

TABLE = "tools/ref/ptx-shapes.tsv"
JUSTFILE = "justfile"
RECIPE = "gate-ptx-spill"

# Pinned binary -> the device crates (directories under crates/) whose kernels it carries, under the features
# the gate-ptx-spill recipe builds it with.
BIN_CRATES = {
    "generate_ds41": ("gpu", "gpu-deepseek41"),
    "gate_qwen3vl_tower": ("gpu", "gpu-vision"),
}

# The package that owns every pinned binary, and the binaries of it the recipe does not scan because the scanned ones carry
# their device code (header: the covered binary).
GATES_PKG = "bloomery-gpu-gates"
COVERED = ("gate_e2e",)

KERNEL_RE = re.compile(r"^#\[kernel(?:\(.*\))?\]")
FN_RE = re.compile(
    r'^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+(\$?[A-Za-z_][A-Za-z0-9_]*)'
)
MACRO_RE = re.compile(r"^macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)")
STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
PATTERN_RE = re.compile(r"^\s*([A-Za-z_]\w*):\s*((?:\$\w+:ident\s*,?\s*)+)$")
BIND_RE = re.compile(r"^\s*([A-Za-z_]\w*):\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*,?\s*$")


class Refuse(Exception):
    """A named refusal: the check cannot tell, so it says so instead of passing."""


def bal(line):
    """Open minus close of ( and [ on a line, strings and a trailing // comment taken out."""
    s = STR_RE.sub('""', line)
    s = s.split("//", 1)[0]
    return s.count("(") + s.count("[") - s.count(")") - s.count("]")


def cfg_test_spans(lines):
    """Line ranges (0-based, inclusive) of `#[cfg(test)] mod x { ... }` blocks, by indentation."""
    spans = []
    for i, raw in enumerate(lines):
        if not raw.strip().startswith("#[cfg(test)]"):
            continue
        indent = len(raw) - len(raw.lstrip())
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("#["):
            j += 1
        if j >= len(lines) or not re.match(r"(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{", lines[j].strip()):
            continue
        end = j
        while end < len(lines) and lines[end].rstrip() != " " * indent + "}":
            end += 1
        spans.append((i, end))
    return spans


def attr_stack_before(lines, i):
    """The attribute lines directly above line i (doc and plain comments skipped, multi-line attributes joined
    by their opening line)."""
    out, b, depth = [], i - 1, 0
    while b >= 0:
        t = lines[b].strip()
        if t.startswith("//"):
            b -= 1
            continue
        if depth > 0 or t.startswith("#["):
            depth -= bal(t)
            out.append(t)
            b -= 1
            continue
        break
    return out


def scan_file(text, rel):
    """(kernels, macros) of one file: kernels are (name, rel, line, macro or None); macros maps a macro_rules
    name to its body lines."""
    lines = text.split("\n")
    spans = cfg_test_spans(lines)
    kernels, macros, cur, body = [], {}, None, []
    for i, raw in enumerate(lines):
        m = MACRO_RE.match(raw)
        if m:
            cur, body = m.group(1), []
        elif cur is not None:
            if raw.rstrip() == "}":
                macros[cur] = body
                cur = None
            else:
                body.append(raw)
        t = raw.strip()
        if t.startswith("//") or not KERNEL_RE.match(t):
            continue
        where = f"{rel}:{i + 1}"
        if any(a <= i <= b for a, b in spans):
            raise Refuse(f"{where}: a #[kernel] inside a #[cfg(test)] module: the release scan never sees it")
        attrs = attr_stack_before(lines, i)
        k, depth, name = i + 1, 0, None
        while k < len(lines):
            u = lines[k].strip()
            if depth == 0 and (u == "" or u.startswith("//")):
                k += 1
                continue
            if depth > 0 or u.startswith("#["):
                if depth == 0:
                    attrs.append(u)
                depth += bal(u)
                k += 1
                continue
            fm = FN_RE.match(u)
            if not fm:
                raise Refuse(f"{where}: #[kernel] is followed by `{u[:60]}`, not a fn")
            name = fm.group(1)
            break
        if name is None:
            raise Refuse(f"{where}: #[kernel] at the end of the file, no fn follows")
        if any(re.match(r"#\[cfg(?:_attr)?\(", a) for a in attrs):
            raise Refuse(f"{where}: kernel `{name}` carries a #[cfg(..)] attribute: whether the build compiles it is not text")
        kernels.append((name, rel, i + 1, cur if name.startswith("$") else None))
    return kernels, macros


def invocations(lines, macro):
    """The `KEY: ident, ident` binding lines of every `macro! { .. }` invocation in lines: a list, one dict a
    call, KEY -> [([idents], line), ..] (a key repeats in a call whose macro repeats a group)."""
    out, i = [], 0
    start = re.compile(r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*" + re.escape(macro) + r"!\s*\{")
    while i < len(lines):
        if not start.match(lines[i]):
            i += 1
            continue
        depth, call = 0, {}
        while i < len(lines):
            t = lines[i]
            if not t.strip().startswith("//"):
                s = STR_RE.sub('""', t).split("//", 1)[0]
                b = BIND_RE.match(s)
                if b:
                    call.setdefault(b.group(1), []).append(([x.strip() for x in b.group(2).split(",")], i + 1))
                depth += s.count("{") - s.count("}")
            i += 1
            if depth <= 0:
                break
        out.append(call)
    return out


def crate_scan(root, crate):
    """The crate's kernel sites, its macro_rules! definitions and its file texts, the shared scan of
    [`bin_kernels`]'s per-binary union below."""
    base = os.path.join(root, "crates", crate, "src")

    def unlisted(e):  # a directory the walk cannot list is an unreadable input (main_check's exit 2), never skipped
        if isinstance(e, FileNotFoundError) and e.filename == base:
            raise Refuse(f"crates/{crate}/src is not there (BIN_CRATES names it)")
        raise e
    files = []
    for d, _, fs in os.walk(base, onerror=unlisted):
        files += [os.path.join(d, f) for f in fs if f.endswith(".rs")]
    found, macros, texts = {}, {}, {}
    for f in sorted(files):
        rel = os.path.relpath(f, root)
        with open(f, encoding="utf-8") as fh:
            texts[rel] = fh.read()
        ks, ms = scan_file(texts[rel], rel)
        macros.update(ms)
        for name, r, ln, mac in ks:
            if name.startswith("$"):
                found.setdefault(("$", mac, name[1:]), []).append((r, ln))
            else:
                found.setdefault(name, []).append((r, ln))
    return found, macros, texts


def bin_kernels(scans):
    """name -> (file, line) of every entry a binary's device crates declare; macro-bound names resolved over the
    union of those crates' files. A macro family may be defined in one of the binary's crates and invoked from
    another (sigmoid_router_instance! lives in gpu, its glm5next and mimo2 instances in gpu-deepseek41), so a
    `$name` family binds over every crate the binary links, not the defining crate's alone."""
    found, macros, texts = {}, {}, {}
    for f, m, t in scans:
        for key, sites in f.items():
            found.setdefault(key, []).extend(sites)
        macros.update(m)
        texts.update(t)
    out = {}
    for key, sites in found.items():
        if not isinstance(key, tuple):
            out[key] = sites[0]
            continue
        _, mac, var = key
        r, ln = sites[0]
        if mac is None or mac not in macros:
            raise Refuse(f"{r}:{ln}: kernel `fn ${var}` is not inside a macro_rules! this check can read")
        keys = {}
        for line in macros[mac]:
            p = PATTERN_RE.match(line)
            if p:
                keys[p.group(1)] = re.findall(r"\$(\w+):ident", p.group(2))
        bound = [k for k, vs in keys.items() if var in vs]
        if len(bound) != 1:
            raise Refuse(f"{r}:{ln}: `${var}` of macro {mac}! is bound by {len(bound)} `KEY: $x:ident, ..` pattern lines, want 1")
        k = bound[0]
        for rel, text in texts.items():
            for call in invocations(text.split("\n"), mac):
                if k not in call:
                    raise Refuse(f"{rel}: an invocation of {mac}! has no `{k}:` line")
                for names, at in call[k]:
                    if len(names) != len(keys[k]):
                        raise Refuse(f"{rel}:{at}: {mac}!'s `{k}:` binds {len(names)} names, the pattern has {len(keys[k])}")
                    out.setdefault(names[keys[k].index(var)], (rel, at))
    return out


DRIVER_SHAPE = re.compile(r"^[0-9]+(\.[0-9]+)+$")
EXCEPTION_SHAPE = re.compile(r"^@([0-9]+(?:\.[0-9]+)+)=([0-9]+)$")


def table_drivers(path, text):
    """The driver versions of a table's one `# jit-driver:` line, [] for a table with none (it pins every driver). The shape rules are
    tools/ptx-spill-check.sh's, which reads the same line and the same row exceptions."""
    lines = re.findall(r"^# jit-driver:(.*)$", text, re.M)
    if not lines:
        return []
    if len(lines) > 1:
        raise Refuse(f"{path}: {len(lines)} `# jit-driver:` lines, a table names its drivers on one")
    drivers = lines[0].split()
    if not drivers or not all(DRIVER_SHAPE.match(d) for d in drivers):
        raise Refuse(f"{path}: `# jit-driver:{lines[0]}` is not driver versions as in `# jit-driver: 595.99.02 615.71.09`")
    if len(set(drivers)) != len(drivers):
        raise Refuse(f"{path}: `# jit-driver:{lines[0]}` lists a driver twice")
    return drivers


def read_table(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    drivers = table_drivers(path, text)
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        if len(f) < 4:
            raise Refuse(f"{path}:{no}: want `<binary> <entry> <spill> <jit_local>`, got {len(f)} fields")
        seen = set()
        for tok in f[4:]:
            m = EXCEPTION_SHAPE.match(tok)
            if not m:
                raise Refuse(f"{path}:{no}: want `@<driver>=<jit_local>` after the four columns, got `{tok}`")
            if m.group(1) not in drivers:
                raise Refuse(f"{path}:{no}: the exception `{tok}` names a driver the table's `# jit-driver:` line does not list")
            if m.group(1) in seen:
                raise Refuse(f"{path}:{no}: the driver {m.group(1)} has two exceptions")
            if int(m.group(2)) == int(f[3]):
                raise Refuse(f"{path}:{no}: the exception `{tok}` repeats the plain jit_local")
            seen.add(m.group(1))
        rows.setdefault(f[0], {})[f[1]] = no
    return rows


def recipe_call(justfile):
    """([the other tables], [(bin, filter or '')]) from the gate-ptx-spill recipe's ptx-spill-check.sh call."""
    with open(justfile, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    at = [i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(RECIPE)}\s*:", ln)]
    if len(at) != 1:
        raise Refuse(f"{justfile}: want one `{RECIPE}:` recipe, found {len(at)}")
    body = []
    for ln in lines[at[0] + 1:]:
        if ln and not ln[0].isspace():
            break
        body.append(ln)
    calls = re.findall(r"ptx-spill-check\.sh\s+" + re.escape(TABLE) + r"\s+([^'\"\n]+)", "\n".join(body))
    if len(calls) != 1:
        raise Refuse(f"{justfile}: recipe {RECIPE} has {len(calls)} `ptx-spill-check.sh {TABLE} <bins>` calls, want 1")
    args = calls[0].split()
    n = next((i for i, a in enumerate(args) if "/" not in a), len(args))
    return args[:n], [tuple(a.split(":", 1)) if ":" in a else (a, "") for a in args[n:]]


def recipe_builds(justfile):
    """bin -> the features of the `cargo oxide build` (or `cargo build`) that builds it in the gate-ptx-spill recipe: the
    comma-separated words after `--features`, every `--bin` of the segment taking them. A bin built twice with other
    features is a refusal (the second build would replace the first's file)."""
    with open(justfile, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    at = [i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(RECIPE)}\s*:", ln)]
    if len(at) != 1:
        raise Refuse(f"{justfile}: want one `{RECIPE}:` recipe, found {len(at)}")
    body = []
    for ln in lines[at[0] + 1:]:
        if ln and not ln[0].isspace():
            break
        body.append(ln)
    out = {}
    for seg in re.findall(r"cargo (?:oxide )?build\b([^&']*)", "\n".join(body)):
        words = seg.split()
        feats = []
        for i, w in enumerate(words):
            if w == "--features" and i + 1 < len(words):
                feats = [f for f in words[i + 1].split(",") if f]
        for i, w in enumerate(words):
            if w == "--bin" and i + 1 < len(words):
                b = words[i + 1]
                if b in out and out[b] != feats:
                    raise Refuse(f"{justfile}: recipe {RECIPE} builds {b} with features {out[b]} and again with {feats}")
                out[b] = feats
    return out


def read_toml(path):
    import tomllib

    with open(path, "rb") as fh:
        try:
            return tomllib.load(fh)
        except tomllib.TOMLDecodeError as e:
            raise Refuse(f"{path}: does not parse as TOML ({e})")


def load_workspace(root):
    """package name -> {dir, features, deps, bins} for every crates/*/Cargo.toml. A dependency is `pkg` (the workspace
    crate it names by `path`, directly or through [workspace.dependencies]; None for an external one), `optional`,
    `features` and `default`, keyed by the name the manifest and its [features] use. Dev and build dependencies are not
    linked into a binary and are left out; a `[target.*.dependencies]` table counts whole."""
    top = read_toml(os.path.join(root, "Cargo.toml"))
    wdeps = top.get("workspace", {}).get("dependencies", {})
    docs = {}
    for path in sorted(glob.glob(os.path.join(root, "crates", "*", "Cargo.toml"))):
        docs[os.path.dirname(path)] = read_toml(path)
    by_dir = {}
    for d, doc in docs.items():
        if "name" not in doc.get("package", {}):
            raise Refuse(f"{os.path.relpath(d, root)}/Cargo.toml has no [package] name")
        by_dir[d] = doc["package"]["name"]
    ws = {}
    for d, doc in docs.items():
        deps = {}
        tables = [doc.get("dependencies", {})] + [v.get("dependencies", {}) for v in doc.get("target", {}).values()]
        for table in tables:
            for key, spec in table.items():
                spec = {"version": spec} if isinstance(spec, str) else spec
                base = wdeps.get(key, {}) if spec.get("workspace") else {}
                base = {"version": base} if isinstance(base, str) else base
                path = spec.get("path") or base.get("path")
                anchor = d if spec.get("path") else root
                deps[key] = {
                    "pkg": by_dir.get(os.path.normpath(os.path.join(anchor, path))) if path else None,
                    "optional": bool(spec.get("optional")),
                    "features": [*base.get("features", []), *spec.get("features", [])],
                    "default": spec.get("default-features", base.get("default-features", True)),
                }
        ws[by_dir[d]] = {
            "dir": d,
            "features": doc.get("features", {}),
            "deps": deps,
            "bins": {b["name"]: b for b in doc.get("bin", []) if "name" in b},
        }
    return ws


def link_closure(ws, root_pkg, requested):
    """{workspace crate: sorted features} for every crate a build of root_pkg with the `requested` features links, by cargo's
    rules for a feature list: `feat` (a feature, or an optional dependency's implicit one), `dep:x`, `x/feat` (turns x on),
    and an edge's own `features` and default features. `x?/feat` is a refusal."""
    feats, linked = {}, set()

    def link(name):
        if name in linked:
            return
        linked.add(name)
        feats.setdefault(name, set())
        for dep in ws[name]["deps"].values():
            if dep["pkg"] and not dep["optional"]:
                use(dep)

    def use(dep):
        link(dep["pkg"])
        for f in dep["features"]:
            enable(dep["pkg"], f)
        if dep["default"] and "default" in ws[dep["pkg"]]["features"]:
            enable(dep["pkg"], "default")

    def enable(name, f):
        link(name)
        p = ws[name]
        if f in feats[name]:
            return
        if f in p["features"]:
            feats[name].add(f)
            for item in p["features"][f]:
                if item.startswith("dep:"):
                    activate(name, item[4:])
                elif "/" in item:
                    dk, sub = item.split("/", 1)
                    if dk.endswith("?"):
                        raise Refuse(f"{name}: feature `{f}` names `{item}`, a weak dependency feature this check does not resolve")
                    activate(name, dk)
                    if p["deps"][dk]["pkg"]:
                        enable(p["deps"][dk]["pkg"], sub)
                else:
                    enable(name, item)
        elif f in p["deps"] and p["deps"][f]["optional"] and not any(f"dep:{f}" in v for v in p["features"].values()):
            feats[name].add(f)
            activate(name, f)
        else:
            raise Refuse(f"{name}: feature `{f}` is declared by no [features] entry and no optional dependency")

    def activate(name, key):
        dep = ws[name]["deps"].get(key)
        if dep is None:
            raise Refuse(f"{name}: a feature names the dependency `{key}`, which its manifest does not declare")
        if dep["pkg"]:
            use(dep)

    link(root_pkg)
    if "default" in ws[root_pkg]["features"]:
        enable(root_pkg, "default")
    for f in requested:
        enable(root_pkg, f)
    return {n: sorted(feats[n]) for n in sorted(linked)}


DEVICE_RE = re.compile(r"^#\[(?:kernel|cuda_module)\b")
INCLUDE_RE = re.compile(r"\binclude!\s*\(")
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")
PATH_RE = re.compile(r'^\s*#\[path\s*=\s*"([^"]+)"\s*\]')


def device_lines(text):
    """The 1-based lines that open a device item: a #[kernel] or #[cuda_module] line of its own, not in a comment."""
    return [i for i, ln in enumerate(text.split("\n"), 1) if not ln.strip().startswith("//") and DEVICE_RE.match(ln.strip())]


def crate_sources(root, ws, name):
    """{relative path: text} of crates/<dir>/src/**/*.rs of workspace crate `name`."""
    base = os.path.join(ws[name]["dir"], "src")
    out = {}

    def unlisted(e):  # a directory the walk cannot list is an unreadable input (main_check's exit 2), never skipped
        raise e
    for d, _, fs in os.walk(base, onerror=unlisted):
        for f in sorted(fs):
            if f.endswith(".rs"):
                p = os.path.join(d, f)
                with open(p, encoding="utf-8") as fh:
                    out[os.path.relpath(p, root)] = fh.read()
    return out


def device_crates(root, ws, names):
    """The crates among `names` that carry device code: a source with a #[kernel] or #[cuda_module] line, or one that
    invokes a macro_rules! (of any crate in `names`) whose body has one (a family instantiated from another crate)."""
    srcs = {n: crate_sources(root, ws, n) for n in names}
    kernel_macros = set()
    for files in srcs.values():
        for text in files.values():
            cur, body = None, []
            for ln in text.split("\n"):
                m = MACRO_RE.match(ln)
                if m:
                    cur, body = m.group(1), []
                elif cur is not None:
                    if ln.rstrip() == "}":
                        if device_lines("\n".join(body)):
                            kernel_macros.add(cur)
                        cur = None
                    else:
                        body.append(ln)
    out = set()
    for n, files in srcs.items():
        for text in files.values():
            if device_lines(text) or any(re.search(rf"\b{re.escape(m)}!\s*\{{", text) for m in kernel_macros):
                out.add(n)
                break
    return out


def bin_sources(root, ws, pkg, bin_name):
    """The files of one bin target: its root (the [[bin]] `path`, else src/bin/<name>.rs or src/bin/<name>/main.rs) and
    every file it declares with `mod x;` (x.rs or x/mod.rs beside it) or `#[path = ".."] mod x;` (relative to the
    declaring file). A `mod x;` that names no file, and an include!, are refusals."""
    p = ws[pkg]
    entry = p["bins"].get(bin_name, {})
    base = p["dir"]
    cands = [os.path.join(base, entry["path"])] if "path" in entry else [
        os.path.join(base, "src", "bin", bin_name + ".rs"), os.path.join(base, "src", "bin", bin_name, "main.rs")]
    top = next((c for c in cands if os.path.exists(c)), None)
    if top is None:
        raise Refuse(f"{pkg}: no source file for the bin `{bin_name}` (looked at {', '.join(os.path.relpath(c, root) for c in cands)})")
    seen, order = set(), []

    def walk(path, mod_dir):
        if path in seen:
            return
        seen.add(path)
        order.append(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        rel = os.path.relpath(path, root)
        attr = None
        for no, ln in enumerate(text.split("\n"), 1):
            if ln.strip().startswith("//"):
                continue
            if INCLUDE_RE.search(ln):
                raise Refuse(f"{rel}:{no}: an include! in a source of the covered binary `{bin_name}`: what it pulls in is not text this check reads")
            m = PATH_RE.match(ln)
            if m:
                attr = m.group(1)
                continue
            m = MOD_RE.match(ln)
            if m:
                if attr:
                    child = os.path.normpath(os.path.join(os.path.dirname(path), attr))
                    nested = os.path.dirname(child)
                else:
                    near = [os.path.join(mod_dir, m.group(1) + ".rs"), os.path.join(mod_dir, m.group(1), "mod.rs")]
                    child = next((c for c in near if os.path.exists(c)), None)
                    if child is None:
                        raise Refuse(f"{rel}:{no}: `mod {m.group(1)};` names no file ({', '.join(os.path.relpath(c, root) for c in near)})")
                    nested = os.path.dirname(child) if os.path.basename(child) == "mod.rs" else os.path.splitext(child)[0]
                if not os.path.exists(child):
                    raise Refuse(f"{rel}:{no}: `mod {m.group(1)};` names the file {os.path.relpath(child, root)}, which is not there")
                walk(child, nested)
            if ln.strip():
                attr = None

    walk(top, os.path.dirname(top))
    return [os.path.relpath(f, root) for f in order]


def covered_problems(root, bins, builds):
    """(violation lines, summary) of the covered binaries' premise (header: the covered binary)."""
    root = os.path.abspath(root)
    ws = load_workspace(root)
    if GATES_PKG not in ws:
        raise Refuse(f"{GATES_PKG}: no crates/*/Cargo.toml names this package")
    scanned = {}
    for b, flt in bins:
        if flt:
            continue  # scanned for the entries holding a substring: it pins those, not the crates' other kernels
        if b not in builds:
            raise Refuse(f"{b}: the {RECIPE} recipe scans this binary and has no `cargo oxide build ... --features .. --bin {b}` for it")
        scanned[b] = link_closure(ws, GATES_PKG, builds[b])
    covered = {}
    for b in COVERED:
        entry = ws[GATES_PKG]["bins"].get(b)
        if entry is None:
            raise Refuse(f"{GATES_PKG}: COVERED names the bin `{b}`, which crates/*/Cargo.toml does not declare")
        covered[b] = link_closure(ws, GATES_PKG, entry.get("required-features", []))
    names = {n for m in [*scanned.values(), *covered.values()] for n in m} - {GATES_PKG}
    carriers = device_crates(root, ws, sorted(names))
    problems, shown = [], []
    for b, linked in covered.items():
        for crate in sorted(c for c in linked if c in carriers):
            mine = linked[crate]
            same = [s for s, m in scanned.items() if m.get(crate) == mine]
            if same:
                shown.append(f"{crate} by {same[0]}")
                continue
            others = {s: m[crate] for s, m in scanned.items() if crate in m}
            if others:
                theirs = "; ".join(f"{s}: [{','.join(f)}]" for s, f in sorted(others.items()))
                problems.append(f"{b} {crate} features differ ({b}: [{','.join(mine)}]; {theirs})")
            else:
                problems.append(f"{b} {crate} not scanned ({b} links it with features [{','.join(mine)}]; "
                                f"no unfiltered binary of the recipe links it)")
        for rel in bin_sources(root, ws, GATES_PKG, b):
            with open(os.path.join(root, rel), encoding="utf-8") as fh:
                for no in device_lines(fh.read()):
                    problems.append(f"{b} {rel}:{no} declares a kernel (a source only {b} compiles: no scanned binary carries it)")
    return problems, f"{', '.join(COVERED)} covered ({', '.join(shown) or 'no device crate'})"


def check(root):
    """(violation lines, summary) for the tree under root. Raises Refuse for a thing it cannot read."""
    others, bins = recipe_call(os.path.join(root, JUSTFILE))
    rows = read_table(os.path.join(root, TABLE))
    tables = []
    for rel in others:
        if not rel.endswith("/"):
            tables.append(rel)
            continue
        found = sorted(glob.glob(os.path.join(root, rel, "*.tsv")))
        if not found:
            raise Refuse(f"recipe {RECIPE} names the directory {rel}, which holds no table")
        tables += [os.path.relpath(p, root) for p in found]
    tables = [(rel, read_table(os.path.join(root, rel))) for rel in tables]
    # PIN(2026-10-23): target-only entries use each table's cfg; another target's rows cannot supply a missing pin.
    table_cfgs = {}
    for rel, _ in [(TABLE, rows), *tables]:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            table_cfgs[rel] = set(re.findall(r"^# cfg: ([A-Za-z_]\w*)$", fh.read(), re.M))
    cfg_names = set().union(*table_cfgs.values())
    problems = []
    scan, nrows, ndecl, decl, guards = {}, 0, 0, {}, {}
    for b, flt in bins:
        if b not in BIN_CRATES:
            raise Refuse(f"{b}: the {RECIPE} recipe pins this binary and BIN_CRATES (tools/check-ptx-names.py) does not "
                         f"say which device crates it links: add the crates of the features the recipe builds it with")
    for b, flt in bins:
        for c in BIN_CRATES[b]:
            if c not in scan:
                scan[c] = crate_scan(root, c)
        declared = {}
        for name, site in bin_kernels([scan[c] for c in BIN_CRATES[b]]).items():
            if flt in name:
                declared.setdefault(name, site)
        decl[b] = declared
        for source in {site[0] for site in declared.values()} - guards.keys():
            guards[source] = set()
            parent = os.path.dirname(os.path.join(root, source))
            module = os.path.splitext(os.path.basename(source))[0]
            for owner in ("mod.rs", "lib.rs"):
                path = os.path.join(parent, owner)
                if not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as fh:
                    text = rust_lex(fh.read())[1]
                for cfg in cfg_names:
                    pattern = r"^\s*#\[cfg\(" + re.escape(cfg) + r"\)\]\s*mod\s+" + re.escape(module) + r"\s*;"
                    if re.search(pattern, text, re.M):
                        guards[source].add(cfg)
        have = rows.get(b, {})
        nrows += len(have) + sum(bool(guards[site[0]]) for e, site in declared.items() if e not in have)
        ndecl += len(declared)
    for rel, brows in [(TABLE, rows), *tables]:
        for b in sorted(set(brows) - set(decl)):
            label = "the table" if rel == TABLE else rel
            problems.append(f"{b}: {label} pins this binary, the {RECIPE} recipe does not run it")
        for b in decl:
            expected = {e for e, site in decl[b].items() if not guards[site[0]] or guards[site[0]] & table_cfgs[rel]}
            have = set(brows.get(b, {}))
            for e in sorted(have - expected):
                problems.append(f"{b} {e} stale" + (f" ({rel})" if rel != TABLE else ""))
            for e in sorted(expected - have):
                source, line = decl[b][e]
                where = f"{source}:{line}" + (f"; {rel}" if rel != TABLE else "")
                problems.append(f"{b} {e} unpinned ({where})")
    tail = f"; {len(tables)} tables beside it checked for stale and unpinned entries" if tables else ""
    more, covered = covered_problems(root, bins, recipe_builds(os.path.join(root, JUSTFILE)))
    problems += more
    return problems, f"{len(bins)} binaries, {nrows} rows = {ndecl} declared entries{tail}; {covered}"


def main_check(root):
    try:
        problems, summary = check(root)
    except Refuse as e:
        print(f"check-ptx-names: refused: {e}")
        return 1
    except OSError as e:
        print(f"check-ptx-names: cannot read {e.filename}: {e.strerror}", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    if problems:
        print(f"check-ptx-names: {len(problems)} violation(s) against {TABLE} — a deleted kernel's row goes with it, "
              f"an added kernel's row is pinned in the same change (tools/ptx-spill-check.sh header); a line that names a covered "
              f"binary is the premise of this file's header, «The covered binary»")
        return 1
    print(f"check-ptx-names: ok ({summary})")
    return 0


# ---- self-test ------------------------------------------------------------------------------------------------

GPU_RS = """\
//! A `#[kernel]` in a doc line and a // #[kernel]
// #[kernel]
// pub fn commented(a: u32) {}

#[cuda_module]
pub mod kernels {
    /// The first. A #[kernel] mention inside a doc line is not an entry.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= n)
    )]
    pub fn k_one(a: &[u32], n: u32) {}

    #[kernel]
    pub fn k_two_8(a: &[u32]) {}
}

/// A family defined in this crate and invoked from another of a binary's crates (the gpu-side
/// sigmoid_router_instance! shape).
macro_rules! x_router_instance {
    (
        pub mod $m:ident {
            entries: $xr:ident, $xs:ident,
        }
    ) => {
        pub mod $m {
            #[kernel]
            pub fn $xr(a: &[u32]) {}

            #[kernel]
            pub fn $xs(a: &[u32]) {}
        }
    };
}

#[cfg(test)]
mod tests {
    fn helper() {}
}
"""

DS_RS = """\
macro_rules! router_instance {
    (
        pub mod $m:ident {
            experts: $e:tt,
            entries: $router:ident, $scores:ident,
            prepare: $p_router:ident, $p_scores:ident,
        }
    ) => {
        pub mod $m {
            #[kernel]
            #[launch_bounds(128)]
            pub fn $router(a: &[f32]) {}

            #[kernel]
            pub fn $scores(a: &[f32]) {}
        }
    };
}

router_instance! {
    /// A doc line with `entries: not, names,` that must not bind.
    pub mod alpha {
        experts: 288,
        entries: alpha_router, alpha_scores,
        prepare: prepare_alpha_router, prepare_alpha_scores,
    }
}

router_instance! {
    pub mod beta {
        experts: 256,
        entries: beta_router, beta_scores,
        prepare: prepare_beta_router, prepare_beta_scores,
    }
}

#[kernel]
pub fn ds_plain(a: &[f32]) {}
"""

MULTI_RS = """
macro_rules! multi_instance {
    (
        $(
            $t:ident {
                $(
                    run: $r:ident, $p:ident,
                )?
            }
        )*
    ) => {
        $(
            $(
                #[kernel]
                pub fn $r(a: &[f32]) {}
            )?
        )*
    };
}

multi_instance! {
    one {
        run: multi_one, prepare_multi_one,
    }
    two {
        run: multi_two, prepare_multi_two,
    }
}
"""

VIS_RS = """\
#[kernel]
pub fn vis_one(a: &[f32]) {}

#[kernel]
pub fn vis_two(a: &[f32]) {}

#[kernel]
pub fn not_filtered(a: &[f32]) {}
"""

JUSTFILE_TXT = """\
gate-other:
    echo other

gate-ptx-spill:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41 && cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features vision --release --bin gate_qwen3vl_tower --bin oxart_jit && bash tools/ptx-spill-check.sh tools/ref/ptx-shapes.tsv generate_ds41 gate_qwen3vl_tower:vis_'

gate-after:
    echo after
"""

ROOT_TOML = """\
[workspace]
members = ["crates/gpu", "crates/gpu-deepseek41", "crates/gpu-vision", "crates/app", "crates/gpu-gates"]

[workspace.dependencies]
cuda-device = "0.2"
"""

GPU_TOML = """\
[package]
name = "bloomery-gpu"

[dependencies]
cuda-device.workspace = true
"""

DS_TOML = """\
[package]
name = "bloomery-gpu-deepseek41"

[dependencies]
cuda-device.workspace = true
bloomery-gpu = { path = "../gpu" }
"""

VIS_TOML = DS_TOML.replace("deepseek41", "vision")

APP_TOML = """\
[package]
name = "bloomery-app"

[dependencies]
bloomery-gpu = { path = "../gpu" }
bloomery-gpu-deepseek41 = { path = "../gpu-deepseek41", optional = true }

[features]
dspark = []
deepseek41 = ["dep:bloomery-gpu-deepseek41", "dspark"]
"""

GATES_TOML = """\
[package]
name = "bloomery-gpu-gates"

[dependencies]
bloomery-gpu = { path = "../gpu", optional = true }
bloomery-gpu-deepseek41 = { path = "../gpu-deepseek41", optional = true }
bloomery-gpu-vision = { path = "../gpu-vision", optional = true }
bloomery-app = { path = "../app", optional = true }

[features]
gpu = ["dep:bloomery-gpu", "dep:bloomery-app", "bloomery-app/dspark"]
deepseek41 = ["gpu", "dep:bloomery-gpu-deepseek41", "bloomery-app/deepseek41"]
vision = ["gpu", "dep:bloomery-gpu-vision"]

[[bin]]
name = "generate_ds41"
required-features = ["deepseek41"]

[[bin]]
name = "gate_e2e"
required-features = ["gpu"]

[[bin]]
name = "gate_qwen3vl_tower"
required-features = ["vision"]
"""

E2E_RS = "fn main() {}\n"

# A crate that carries kernels only through the macro `x_router_instance!` that GPU_RS defines (the glm5next and mimo2 shape).
MIMO_TOML = DS_TOML.replace("deepseek41", "mimo2")
MIMO_RS = "bloomery_gpu::x_router_instance! {\n    pub mod xi {\n        entries: m_router, m_scores,\n    }\n}\n"

BESIDE_FIXTURE = "tools/ref/ptx-shapes/sm_121-cc12.1.tsv"
BESIDE_DIR = "tools/ref/ptx-shapes/"

TABLE_ROWS = {
    "generate_ds41": ["k_one", "k_two_8", "alpha_router", "alpha_scores", "beta_router", "beta_scores", "ds_plain"],
    "gate_qwen3vl_tower": ["vis_one", "vis_two"],
}


def line_of(text, needle):
    """1-based line of the first line of text that holds needle."""
    return next(i for i, ln in enumerate(text.split("\n"), 1) if needle in ln)


def table_text(rows):
    out = ["# fixture", "# arch: sm_86", ""]
    for b, es in rows.items():
        out += [f"{b}   {e}   0   0" for e in es]
    return "\n".join(out) + "\n"


def write_fixture(root, gpu=GPU_RS, ds=DS_RS, vis=VIS_RS, justfile=JUSTFILE_TXT, rows=TABLE_ROWS, table=None, beside=None,
                  gates=GATES_TOML, gpu_toml=GPU_TOML, app=APP_TOML, e2e=E2E_RS, extra=()):
    if beside is not None:
        justfile = justfile.replace(f"{TABLE} ", f"{TABLE} {BESIDE_DIR} ")
    for rel, text in (*(((BESIDE_FIXTURE, beside),) if beside is not None else ()),
        ("crates/gpu/src/lib.rs", gpu),
        ("crates/gpu-deepseek41/src/router.rs", ds),
        ("crates/gpu-vision/src/vis.rs", vis),
        ("Cargo.toml", ROOT_TOML),
        ("crates/gpu/Cargo.toml", gpu_toml),
        ("crates/gpu-deepseek41/Cargo.toml", DS_TOML),
        ("crates/gpu-vision/Cargo.toml", VIS_TOML),
        ("crates/app/Cargo.toml", app),
        ("crates/app/src/lib.rs", "pub fn app() {}\n"),
        ("crates/gpu-gates/Cargo.toml", gates),
        ("crates/gpu-gates/src/lib.rs", "pub fn gates() {}\n"),
        ("crates/gpu-gates/src/bin/gate_e2e.rs", e2e),
        (JUSTFILE, justfile),
        (TABLE, table if table is not None else table_text(rows)),
        *extra,
    ):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)


def self_test():
    fails = []
    top = tempfile.mkdtemp(prefix="check-ptx-names-")

    def run(name, want_rc, want_lines, **kw):
        root = tempfile.mkdtemp(dir=top)
        write_fixture(root, **kw)
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main_check(root)
        out = buf.getvalue()
        got = [ln for ln in out.split("\n") if ln and not ln.startswith("check-ptx-names: ")]
        if rc != want_rc:
            fails.append(f"{name}: rc {rc}, want {want_rc}\n{out}")
        elif sorted(got) != sorted(want_lines):
            fails.append(f"{name}: lines {sorted(got)}, want {sorted(want_lines)}\n{out}")
        return out

    try:
        out = run("green: every row declared, every declared entry pinned (macro names and the filter included)", 0, [])
        if "2 binaries, 9 rows = 9 declared entries" not in out or "gate_e2e covered (bloomery-gpu by generate_ds41)" not in out:
            fails.append(f"green summary: {out!r}")
        native = "#[kernel]\npub fn k_native() {}\n"
        guarded_gpu = GPU_RS + "\n#[cfg(native)]\nmod native;\n"
        native_table = "# cfg: native\n" + table_text(dict(TABLE_ROWS, generate_ds41=TABLE_ROWS["generate_ds41"] + ["k_native"]))
        fixture = {"gpu": guarded_gpu, "extra": (("crates/gpu/src/native.rs", native),)}
        run("target-gated: its target table pins the entry", 0, [], beside=native_table, **fixture)
        run("target-gated: the target table must pin its conditional entry", 1,
            [f"generate_ds41 k_native unpinned (crates/gpu/src/native.rs:1; {BESIDE_FIXTURE})"],
            beside=native_table.replace("generate_ds41   k_native   0   0\n", ""), **fixture)
        run("target-gated: the target table must also pin unconditional entries", 1,
            [f"generate_ds41 k_two_8 unpinned (crates/gpu/src/lib.rs:{line_of(GPU_RS, 'pub fn k_two_8') - 1}; {BESIDE_FIXTURE})"],
            beside=native_table.replace("generate_ds41   k_two_8   0   0\n", ""), **fixture)
        run("target-gated: another target does not need this conditional entry", 0, [],
            beside=native_table, **dict(fixture, extra=fixture["extra"] +
                (("tools/ref/ptx-shapes/sm_86-cc12.1.tsv", table_text(TABLE_ROWS)),)))
        run("target-gated: an unannotated target table cannot hide the missing default pin", 1,
            ["generate_ds41 k_native unpinned (crates/gpu/src/native.rs:1)"],
            beside=native_table.replace("# cfg: native\n", ""), **fixture)
        run("target-gated: the source guard must match the table", 1,
            ["generate_ds41 k_native unpinned (crates/gpu/src/native.rs:1)"],
            beside=native_table.replace("cfg: native", "cfg: different"), **fixture)
        run("target-gated: a commented guard is not a build condition", 1,
            ["generate_ds41 k_native unpinned (crates/gpu/src/native.rs:1)"],
            beside=native_table, gpu=guarded_gpu.replace("#[cfg(native)]", "// #[cfg(native)]"),
            extra=fixture["extra"])
        # stale: a kernel deleted from source, its row left.
        run("stale: a deleted kernel's row", 1, ["generate_ds41 k_two_8 stale"],
            gpu=GPU_RS.replace("    #[kernel]\n    pub fn k_two_8(a: &[u32]) {}\n", ""))
        # unpinned: a kernel added without its row (in each of the binary's crates).
        gpu2 = GPU_RS.replace("pub fn k_two_8(a: &[u32]) {}\n", "pub fn k_two_8(a: &[u32]) {}\n\n    #[kernel]\n    pub fn k_new() {}\n")
        at = f"crates/gpu/src/lib.rs:{line_of(gpu2, 'pub fn k_new') - 1}"
        run("unpinned: an added kernel with no row", 1, [f"generate_ds41 k_new unpinned ({at})"], gpu=gpu2)
        ds2 = DS_RS + "\n#[kernel]\npub fn ds_new() {}\n"
        run("unpinned: a kernel of the V4.1 crate lands on generate_ds41 only", 1,
            [f"generate_ds41 ds_new unpinned (crates/gpu-deepseek41/src/router.rs:{line_of(ds2, 'pub fn ds_new') - 1})"], ds=ds2)
        # the macro path: one instance dropped leaves its two names stale, a third instance two unpinned.
        run("macro: a dropped invocation leaves its names stale", 1,
            ["generate_ds41 beta_router stale", "generate_ds41 beta_scores stale"],
            ds=DS_RS.replace("router_instance! {\n    pub mod beta {\n        experts: 256,\n        entries: beta_router, beta_scores,\n"
                             "        prepare: prepare_beta_router, prepare_beta_scores,\n    }\n}\n", ""))
        ds3 = (DS_RS + "\ncrate::router_instance! {\n    pub mod gamma {\n        experts: 8,\n        entries: gamma_router, gamma_scores,\n"
               "        prepare: p_g_router, p_g_scores,\n    }\n}\n")
        at = f"crates/gpu-deepseek41/src/router.rs:{line_of(ds3, 'entries: gamma_router')}"
        run("macro: an added invocation's names are unpinned", 1,
            [f"generate_ds41 gamma_router unpinned ({at})", f"generate_ds41 gamma_scores unpinned ({at})"], ds=ds3)
        # a macro that repeats a group: the invocation repeats its KEY line, and every line binds its own names.
        ds5 = DS_RS + MULTI_RS
        at1 = f"crates/gpu-deepseek41/src/router.rs:{line_of(ds5, 'run: multi_one')}"
        at2 = f"crates/gpu-deepseek41/src/router.rs:{line_of(ds5, 'run: multi_two')}"
        run("macro: a repeated KEY line binds every line's names", 1,
            [f"generate_ds41 multi_one unpinned ({at1})", f"generate_ds41 multi_two unpinned ({at2})"], ds=ds5)
        out = run("refusal: a repeated KEY line that binds the wrong count", 1, [],
                  ds=ds5.replace("run: multi_two, prepare_multi_two,", "run: multi_two,"))
        if "binds 1 names, the pattern has 2" not in out:
            fails.append(f"repeated key, wrong count: {out!r}")
        # a family defined in gpu, invoked from gpu-deepseek41: per-crate binding never sees the invocation, and
        # the names go stale or stay unseen (train0211i's mode, a red landing batch after green Mac checks).
        ds4 = (DS_RS + "\ngpu_crate::x_router_instance! {\n    pub mod xi {\n        entries: x_router, x_scores,\n    }\n}\n")
        at = f"crates/gpu-deepseek41/src/router.rs:{line_of(ds4, 'entries: x_router')}"
        run("macro: a family defined in another crate of the binary binds its names", 1,
            [f"generate_ds41 x_router unpinned ({at})", f"generate_ds41 x_scores unpinned ({at})"], ds=ds4)
        # the filter: a vision kernel without the substring is outside the pin, one with it must have a row.
        vis2 = VIS_RS + "\n#[kernel]\npub fn vis_three(a: &[f32]) {}\n"
        run("filter: a vis_ kernel with no row", 1,
            [f"gate_qwen3vl_tower vis_three unpinned (crates/gpu-vision/src/vis.rs:{line_of(vis2, 'pub fn vis_three') - 1})"], vis=vis2)
        # refusals, each by name.
        out = run("refusal: a macro name the pattern does not bind", 1, [],
                  ds=DS_RS.replace("pub fn $scores(a", "pub fn $unbound(a"))
        if "`$unbound` of macro router_instance! is bound by 0" not in out:
            fails.append(f"unbound macro name: {out!r}")
        out = run("refusal: a kernel under #[cfg]", 1, [], gpu=GPU_RS.replace("    #[kernel]\n    pub fn k_two_8", "    #[cfg(feature = \"x\")]\n    #[kernel]\n    pub fn k_two_8"))
        if "carries a #[cfg(..)] attribute" not in out:
            fails.append(f"cfg kernel: {out!r}")
        out = run("refusal: a kernel in a cfg(test) module", 1, [],
                  gpu=GPU_RS.replace("    fn helper() {}", "    #[kernel]\n    pub fn in_test() {}"))
        if "inside a #[cfg(test)] module" not in out:
            fails.append(f"cfg(test) kernel: {out!r}")
        out = run("refusal: #[kernel] not followed by a fn", 1, [], gpu=GPU_RS + "\n#[kernel]\nstruct S;\n")
        if "is followed by `struct S;`, not a fn" not in out:
            fails.append(f"kernel then struct: {out!r}")
        out = run("refusal: a pinned binary BIN_CRATES lacks", 1, [],
                  justfile=JUSTFILE_TXT.replace(" gate_qwen3vl_tower:vis_'", " gate_new'"))
        if "gate_new: the gate-ptx-spill recipe pins this binary and BIN_CRATES" not in out:
            fails.append(f"unknown binary: {out!r}")
        out = run("table: a binary the recipe does not run", 1, ["gate_old: the table pins this binary, the gate-ptx-spill recipe does not run it"],
                  rows=dict(TABLE_ROWS, gate_old=["k_one"]))
        # Every table holds the unconditional entries and the entries of its own target.
        lag = {b: [e for e in es if e != "k_two_8"] for b, es in TABLE_ROWS.items()}
        out = run("beside: a missing row is unpinned", 1,
            [f"generate_ds41 k_two_8 unpinned (crates/gpu/src/lib.rs:{line_of(GPU_RS, 'pub fn k_two_8') - 1}; {BESIDE_FIXTURE})"],
            beside=table_text(lag))
        if "1 violation(s)" not in out:
            fails.append(f"beside summary: {out!r}")
        run("beside: a missing filtered entry is unpinned", 1,
            [f"gate_qwen3vl_tower vis_two unpinned (crates/gpu-vision/src/vis.rs:{line_of(VIS_RS, 'pub fn vis_two') - 1}; {BESIDE_FIXTURE})"],
            beside=table_text(dict(TABLE_ROWS, gate_qwen3vl_tower=["vis_one"])))
        run("beside: an explicit table path is checked too", 1,
            [f"generate_ds41 k_two_8 unpinned (crates/gpu/src/lib.rs:{line_of(GPU_RS, 'pub fn k_two_8') - 1}; {BESIDE_FIXTURE})"],
            extra=((BESIDE_FIXTURE, table_text(lag)),), justfile=JUSTFILE_TXT.replace(f"{TABLE} ", f"{TABLE} {BESIDE_FIXTURE} "))
        run("beside: a deleted kernel's row there is stale", 1,
            [f"generate_ds41 k_old stale ({BESIDE_FIXTURE})"],
            beside=table_text(dict(TABLE_ROWS, generate_ds41=TABLE_ROWS["generate_ds41"] + ["k_old"])))
        out = run("refusal: a directory of tables that holds none", 1, [],
                  justfile=JUSTFILE_TXT.replace(f"{TABLE} ", f"{TABLE} {BESIDE_DIR} "))
        if f"names the directory {BESIDE_DIR}, which holds no table" not in out:
            fails.append(f"empty directory: {out!r}")
        out = run("refusal: no ptx-spill-check call in the recipe", 1, [], justfile=JUSTFILE_TXT.replace("ptx-spill-check.sh", "other.sh"))
        if "has 0 `ptx-spill-check.sh" not in out:
            fails.append(f"no call: {out!r}")
        out = run("refusal: a malformed table row", 1, [], table="generate_ds41 k_one 0\n")
        if "want `<binary> <entry> <spill> <jit_local>`, got 3 fields" not in out:
            fails.append(f"malformed row: {out!r}")
        # driver exceptions (the header's «A row»): read under a `# jit-driver:` line, refused by name otherwise.
        base = table_text(TABLE_ROWS)
        drv = base.replace("# arch: sm_86\n", "# arch: sm_86\n# jit-driver: 595.99.02 615.71.09\n")
        row = "generate_ds41   k_one   0   0\n"
        out = run("green: a listed driver's exception", 0, [], table=drv.replace(row, row.rstrip() + " @615.71.09=8\n"))
        if "2 binaries, 9 rows = 9 declared entries" not in out:
            fails.append(f"exception summary: {out!r}")
        for name, text, want in (
            ("an exception under no driver line", base.replace(row, row.rstrip() + " @615.71.09=8\n"),
             "the exception `@615.71.09=8` names a driver the table's `# jit-driver:` line does not list"),
            ("an exception of a driver the line does not list", drv.replace(row, row.rstrip() + " @600.1.2=8\n"),
             "the exception `@600.1.2=8` names a driver the table's `# jit-driver:` line does not list"),
            ("an exception that repeats the plain column", drv.replace(row, row.rstrip() + " @615.71.09=0\n"),
             "the exception `@615.71.09=0` repeats the plain jit_local"),
            ("two exceptions of one driver", drv.replace(row, row.rstrip() + " @615.71.09=8 @615.71.09=16\n"),
             "the driver 615.71.09 has two exceptions"),
            ("an exception with no value", drv.replace(row, row.rstrip() + " @615.71.09\n"),
             "want `@<driver>=<jit_local>` after the four columns, got `@615.71.09`"),
            ("a fifth field that is no exception", drv.replace(row, row.rstrip() + " junk\n"),
             "want `@<driver>=<jit_local>` after the four columns, got `junk`"),
            ("two driver lines", drv.replace("# jit-driver: 595.99.02 615.71.09\n", "# jit-driver: 595.99.02\n# jit-driver: 615.71.09\n"),
             "2 `# jit-driver:` lines, a table names its drivers on one"),
            ("a driver line with no version", drv.replace("# jit-driver: 595.99.02 615.71.09\n", "# jit-driver:\n"),
             "is not driver versions as in `# jit-driver: 595.99.02 615.71.09`"),
            ("a driver line with a version of another shape", drv.replace("595.99.02 615.71.09", "595 615.71.09"),
             "is not driver versions as in `# jit-driver: 595.99.02 615.71.09`"),
            ("a driver listed twice", drv.replace("595.99.02 615.71.09", "615.71.09 615.71.09"), "lists a driver twice"),
        ):
            out = run(f"refusal: {name}", 1, [], table=text)
            if want not in out:
                fails.append(f"{name}: {out!r}")
        # the covered binary (header): its device crates, their features and its own sources against the scanned binaries'.
        # A feature on bloomery-gpu that only gate_e2e's feature set turns on: its build compiles a kernel set no scanned binary does.
        wide_gpu = GPU_TOML + "\n[features]\nwide = []\n"
        wide_e2e = GATES_TOML.replace('name = "gate_e2e"\nrequired-features = ["gpu"]', 'name = "gate_e2e"\nrequired-features = ["e2e-wide"]'
                                      ).replace('vision = ["gpu", "dep:bloomery-gpu-vision"]',
                                                'vision = ["gpu", "dep:bloomery-gpu-vision"]\ne2e-wide = ["gpu", "bloomery-gpu/wide"]')
        run("covered: a feature only gate_e2e's set turns on", 1,
            ["gate_e2e bloomery-gpu features differ (gate_e2e: [wide]; generate_ds41: [])"], gates=wide_e2e, gpu_toml=wide_gpu)
        # ...and the other way: a scanned binary's set turns it on and gate_e2e's does not, so gate_e2e's kernels are not the pinned ones.
        wide_ds = GATES_TOML.replace('deepseek41 = ["gpu",', 'deepseek41 = ["gpu", "bloomery-gpu/wide",')
        run("covered: a feature only the scanned binary's set turns on", 1,
            ["gate_e2e bloomery-gpu features differ (gate_e2e: []; generate_ds41: [wide])"], gates=wide_ds, gpu_toml=wide_gpu)
        # A device crate gate_e2e links that the one scanned binary pinning only a substring of its entries links: not covered.
        vis_e2e = GATES_TOML.replace('name = "gate_e2e"\nrequired-features = ["gpu"]', 'name = "gate_e2e"\nrequired-features = ["vision"]')
        run("covered: a device crate only a filtered scan links", 1,
            ["gate_e2e bloomery-gpu-vision not scanned (gate_e2e links it with features []; no unfiltered binary of the recipe links it)"],
            gates=vis_e2e)
        # A crate with no #[kernel] line of its own that instantiates a kernel macro of another crate is a device crate too.
        mimo_gates = GATES_TOML.replace('bloomery-app = { path = "../app", optional = true }\n',
                                        'bloomery-app = { path = "../app", optional = true }\nbloomery-gpu-mimo2 = { path = "../gpu-mimo", optional = true }\n'
                                        ).replace('name = "gate_e2e"\nrequired-features = ["gpu"]', 'name = "gate_e2e"\nrequired-features = ["mimo2"]'
                                                  ).replace('vision = ["gpu", "dep:bloomery-gpu-vision"]',
                                                            'vision = ["gpu", "dep:bloomery-gpu-vision"]\nmimo2 = ["gpu", "dep:bloomery-gpu-mimo2"]')
        mimo = (("crates/gpu-mimo/Cargo.toml", MIMO_TOML), ("crates/gpu-mimo/src/lib.rs", MIMO_RS))
        run("covered: a crate whose kernels come from a macro of another crate", 1,
            ["gate_e2e bloomery-gpu-mimo2 not scanned (gate_e2e links it with features []; no unfiltered binary of the recipe links it)"],
            gates=mimo_gates, extra=mimo)
        # A kernel in a source only gate_e2e compiles: the bin file itself, and a file it names with #[path].
        run("covered: a #[kernel] in gate_e2e's own file", 1,
            ["gate_e2e crates/gpu-gates/src/bin/gate_e2e.rs:2 declares a kernel (a source only gate_e2e compiles: no scanned binary carries it)"],
            e2e="fn main() {}\n#[kernel]\npub fn e2e_only() {}\n")
        run("covered: a #[cuda_module] in a module gate_e2e names with #[path]", 1,
            ["gate_e2e crates/gpu-gates/src/bin/shared/e2e_k.rs:2 declares a kernel (a source only gate_e2e compiles: no scanned binary carries it)"],
            e2e='#[path = "shared/e2e_k.rs"]\nmod e2e_k;\nfn main() {}\n',
            extra=(("crates/gpu-gates/src/bin/shared/e2e_k.rs", "// shared\n#[cuda_module]\nmod m {}\n"),))
        run("covered: a #[kernel] in a comment of gate_e2e's file is no kernel", 0, [], e2e="// #[kernel]\nfn main() {}\n")
        out = run("refusal: a `mod x;` of gate_e2e that names no file", 1, [], e2e="mod gone;\nfn main() {}\n")
        if "`mod gone;` names no file" not in out:
            fails.append(f"missing module: {out!r}")
        out = run("refusal: an include! in gate_e2e", 1, [], e2e='include!("x.rs");\nfn main() {}\n')
        if "an include! in a source of the covered binary `gate_e2e`" not in out:
            fails.append(f"include!: {out!r}")
        out = run("refusal: a weak dependency feature", 1, [], gates=GATES_TOML.replace('"bloomery-app/dspark"', '"bloomery-app?/dspark"'))
        if "a weak dependency feature this check does not resolve" not in out:
            fails.append(f"weak feature: {out!r}")
        out = run("refusal: a scanned binary the recipe has no build for", 1, [],
                  justfile=JUSTFILE_TXT.replace("--bin generate_ds41", "--bin generate_other"))
        if "generate_ds41: the gate-ptx-spill recipe scans this binary and has no `cargo oxide build" not in out:
            fails.append(f"no build: {out!r}")
        out = run("refusal: a binary built twice with other features", 1, [],
                  justfile=JUSTFILE_TXT.replace("--bin oxart_jit", "--bin oxart_jit && cargo build --features gpu --bin gate_qwen3vl_tower"))
        if "builds gate_qwen3vl_tower with features ['vision'] and again with ['gpu']" not in out:
            fails.append(f"built twice: {out!r}")
        out = run("refusal: a covered binary the manifest does not declare", 1, [],
                  gates=GATES_TOML.replace('name = "gate_e2e"', 'name = "gate_e2f"'))
        if "COVERED names the bin `gate_e2e`" not in out:
            fails.append(f"undeclared covered bin: {out!r}")
        # an unreadable tree is rc 2, not a pass.
        empty = tempfile.mkdtemp(dir=top)
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = main_check(empty)
        if rc != 2:
            fails.append(f"an empty root: rc {rc}, want 2")
        # a directory under a crate's src that the walk cannot list is rc 2, not a pass (a mode does not bind root).
        if os.geteuid() != 0:
            root = tempfile.mkdtemp(dir=top)
            write_fixture(root)
            hidden = os.path.join(root, "crates", "gpu", "src", "hidden")
            os.makedirs(hidden)
            with open(os.path.join(hidden, "k.rs"), "w", encoding="utf-8") as fh:
                fh.write("#[kernel]\npub fn k_hidden() {}\n")
            os.chmod(hidden, 0)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    rc = main_check(root)
            finally:
                os.chmod(hidden, 0o755)
            if rc != 2:
                fails.append(f"a directory it cannot list: rc {rc}, want 2\n{buf.getvalue()}")
    finally:
        shutil.rmtree(top, ignore_errors=True)
    if fails:
        for f in fails:
            print(f"check-ptx-names self-test FAIL: {f}", file=sys.stderr)
        return 1
    print("check-ptx-names: self-test ok")
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if argv[:1] == ["check"]:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rest = argv[1:]
        if rest[:1] == ["--root"] and len(rest) == 2:
            root = rest[1]
        elif rest:
            print(__doc__.split("\n\n")[0], file=sys.stderr)
            return 2
        return main_check(root)
    print("usage: tools/check-ptx-names.py check [--root DIR] | --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
