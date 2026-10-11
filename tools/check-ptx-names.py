#!/usr/bin/env python3
"""The ptx-shapes name check as a static check (Mac, text only, no build).

    tools/check-ptx-names.py check [--root DIR]
    tools/check-ptx-names.py --self-test

tools/ptx-spill-check.sh holds every entry a pinned binary carries to the row of tools/ref/ptx-shapes.tsv, but
only on the box, after the three binaries are built and scanned: a kernel deleted without its row, or added
without one, passed every Mac check and went red inside a landing batch (train0211i, 2026-10-11, carried R5u's
first commit without its pin commit: ds41_ffn_card_acc_10 unpinned, ds41_ffn_places_10_cols, q38_card_acc and
qwen3moe_places stale). This check reads the same two sets from source:

  rows      the table's `<binary> <entry> <spill> <jit_local>` lines;
  declared  the entries the binary's device crates declare, which are the PTX entry names (the table's entries
            are the `fn` names: `_8`, `_10`, `_10_cols` are part of the name, no suffix is generated).

and fails a row no source declares (`stale`) and a declared entry with no row (`unpinned`), per binary:

  <bin> <entry> stale                         (in the table, declared by none of the binary's crates)
  <bin> <entry> unpinned (<file>:<line>)      (declared, not in the table)

What declares an entry. A `#[kernel]` attribute (its attribute stack may span lines: `#[allow(..)]`,
`#[launch_bounds(..)]`, `#[launch_contract(..)]` and doc lines sit between it and the `fn`) on a line of its own,
not in a comment; the entry is the `fn` that follows. The one generated family is a `macro_rules!` whose kernel
is `fn $name`: each invocation binds the names on its `KEY: a, b, c,` line positionally to the macro's own
pattern line `KEY: $x:ident, $y:ident, $z:ident,` (today `sigmoid_router_instance!`, `entries:`: the
glm5next and mimo2 routers). A `$name` the macro's pattern does not bind, a `#[kernel]` that no `fn` follows,
a kernel under a `#[cfg(..)]` attribute or inside a `#[cfg(test)]` module, and a binary or recipe this file does
not know are each a named refusal, never a skipped entry.

Which binaries and crates. The binaries come from the `gate-ptx-spill` recipe's `ptx-spill-check.sh
tools/ref/ptx-shapes.tsv <bin>[:<substring>]...` call (the one owner of what is pinned); BIN_CRATES names the
device crates each links under the features the recipe builds it with (`generate_ds41`: gpu + deepseek41,
`gate_e2e`: gpu, `gate_qwen3vl_tower`: vision = gpu + gpu-vision). A substring after `:` is ptx-scan's own entry
filter: the binary's expected entries are the crates' kernels that hold it. A binary in the recipe or the
table that BIN_CRATES lacks is a refusal that says what to add.

What it cannot see (the box scan still does):
  - which crates a feature set links: BIN_CRATES is a map written beside the recipe, not derived from
    crates/gpu-gates/Cargo.toml; a feature change that links one more kernel crate into a pinned binary shows
    on the box as `unpinned`;
  - a macro other than the one bound by a `KEY:` line, a `#[macro_export]` macro invoked from another crate, a
    kernel written by a proc macro (all named refusals where the text shows them, otherwise invisible);
  - the spill and jit_local values (the ratchet's own job) and whether the compiler emits an entry.

Exit: 0 every binary's rows equal its declared entries; 1 a violation or a refusal (printed, one line each, every
binary read before the verdict); 2 a usage error or an unreadable input.
"""

import os
import re
import shutil
import sys
import tempfile

TABLE = "tools/ref/ptx-shapes.tsv"
JUSTFILE = "justfile"
RECIPE = "gate-ptx-spill"

# Pinned binary -> the device crates (directories under crates/) whose kernels it carries, under the features
# the gate-ptx-spill recipe builds it with.
BIN_CRATES = {
    "generate_ds41": ("gpu", "gpu-deepseek41"),
    "gate_e2e": ("gpu",),
    "gate_qwen3vl_tower": ("gpu", "gpu-vision"),
}

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
    call, KEY -> ([idents], line)."""
    out, i = [], 0
    start = re.compile(r"^\s*" + re.escape(macro) + r"!\s*\{")
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
                    call[b.group(1)] = ([x.strip() for x in b.group(2).split(",")], i + 1)
                depth += s.count("{") - s.count("}")
            i += 1
            if depth <= 0:
                break
        out.append(call)
    return out


def crate_kernels(root, crate):
    """name -> (file, line) of every entry the crate's src declares; macro-bound names resolved."""
    base = os.path.join(root, "crates", crate, "src")
    if not os.path.isdir(base):
        raise Refuse(f"crates/{crate}/src is not there (BIN_CRATES names it)")
    files = []
    for d, _, fs in os.walk(base):
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
                names, at = call[k]
                if len(names) != len(keys[k]):
                    raise Refuse(f"{rel}:{at}: {mac}!'s `{k}:` binds {len(names)} names, the pattern has {len(keys[k])}")
                out.setdefault(names[keys[k].index(var)], (rel, at))
    return out


def read_table(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            f = s.split()
            if len(f) != 4:
                raise Refuse(f"{path}:{no}: want `<binary> <entry> <spill> <jit_local>`, got {len(f)} fields")
            rows.setdefault(f[0], {})[f[1]] = no
    return rows


def recipe_bins(justfile):
    """[(bin, filter or '')] from the gate-ptx-spill recipe's ptx-spill-check.sh call."""
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
    return [tuple(a.split(":", 1)) if ":" in a else (a, "") for a in calls[0].split()]


def check(root):
    """(violation lines, summary) for the tree under root. Raises Refuse for a thing it cannot read."""
    bins = recipe_bins(os.path.join(root, JUSTFILE))
    rows = read_table(os.path.join(root, TABLE))
    problems = []
    for b in sorted(set(rows) - {b for b, _ in bins}):
        problems.append(f"{b}: the table pins this binary, the {RECIPE} recipe does not run it")
    kern, nrows, ndecl = {}, 0, 0
    for b, flt in bins:
        if b not in BIN_CRATES:
            raise Refuse(f"{b}: the {RECIPE} recipe pins this binary and BIN_CRATES (tools/check-ptx-names.py) does not "
                         f"say which device crates it links: add the crates of the features the recipe builds it with")
    for b, flt in bins:
        declared = {}
        for c in BIN_CRATES[b]:
            if c not in kern:
                kern[c] = crate_kernels(root, c)
            for name, site in kern[c].items():
                if flt in name:
                    declared.setdefault(name, site)
        have = rows.get(b, {})
        for e in sorted(set(have) - set(declared)):
            problems.append(f"{b} {e} stale")
        for e in sorted(set(declared) - set(have)):
            problems.append(f"{b} {e} unpinned ({declared[e][0]}:{declared[e][1]})")
        nrows += len(have)
        ndecl += len(declared)
    return problems, f"{len(bins)} binaries, {nrows} rows = {ndecl} declared entries"


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
              f"an added kernel's row is pinned in the same change (tools/ptx-spill-check.sh header)")
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
    ./tools/box.sh 'cargo oxide build && bash tools/ptx-spill-check.sh tools/ref/ptx-shapes.tsv generate_ds41 gate_e2e gate_qwen3vl_tower:vis_'

gate-after:
    echo after
"""

TABLE_ROWS = {
    "generate_ds41": ["k_one", "k_two_8", "alpha_router", "alpha_scores", "beta_router", "beta_scores", "ds_plain"],
    "gate_e2e": ["k_one", "k_two_8"],
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


def write_fixture(root, gpu=GPU_RS, ds=DS_RS, vis=VIS_RS, justfile=JUSTFILE_TXT, rows=TABLE_ROWS, table=None):
    for rel, text in (
        ("crates/gpu/src/lib.rs", gpu),
        ("crates/gpu-deepseek41/src/router.rs", ds),
        ("crates/gpu-vision/src/vis.rs", vis),
        (JUSTFILE, justfile),
        (TABLE, table if table is not None else table_text(rows)),
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
        if "3 binaries, 11 rows = 11 declared entries" not in out:
            fails.append(f"green summary: {out!r}")
        # stale: a kernel deleted from source, its row left.
        run("stale: a deleted kernel's row", 1, ["generate_ds41 k_two_8 stale", "gate_e2e k_two_8 stale"],
            gpu=GPU_RS.replace("    #[kernel]\n    pub fn k_two_8(a: &[u32]) {}\n", ""))
        # unpinned: a kernel added without its row (in each of the binary's crates).
        gpu2 = GPU_RS.replace("pub fn k_two_8(a: &[u32]) {}\n", "pub fn k_two_8(a: &[u32]) {}\n\n    #[kernel]\n    pub fn k_new() {}\n")
        at = f"crates/gpu/src/lib.rs:{line_of(gpu2, 'pub fn k_new') - 1}"
        run("unpinned: an added kernel with no row", 1, [f"generate_ds41 k_new unpinned ({at})", f"gate_e2e k_new unpinned ({at})"], gpu=gpu2)
        ds2 = DS_RS + "\n#[kernel]\npub fn ds_new() {}\n"
        run("unpinned: a kernel of the V4.1 crate lands on generate_ds41 only", 1,
            [f"generate_ds41 ds_new unpinned (crates/gpu-deepseek41/src/router.rs:{line_of(ds2, 'pub fn ds_new') - 1})"], ds=ds2)
        # the macro path: one instance dropped leaves its two names stale, a third instance two unpinned.
        run("macro: a dropped invocation leaves its names stale", 1,
            ["generate_ds41 beta_router stale", "generate_ds41 beta_scores stale"],
            ds=DS_RS.replace("router_instance! {\n    pub mod beta {\n        experts: 256,\n        entries: beta_router, beta_scores,\n"
                             "        prepare: prepare_beta_router, prepare_beta_scores,\n    }\n}\n", ""))
        ds3 = (DS_RS + "\nrouter_instance! {\n    pub mod gamma {\n        experts: 8,\n        entries: gamma_router, gamma_scores,\n"
               "        prepare: p_g_router, p_g_scores,\n    }\n}\n")
        at = f"crates/gpu-deepseek41/src/router.rs:{line_of(ds3, 'entries: gamma_router')}"
        run("macro: an added invocation's names are unpinned", 1,
            [f"generate_ds41 gamma_router unpinned ({at})", f"generate_ds41 gamma_scores unpinned ({at})"], ds=ds3)
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
                  justfile=JUSTFILE_TXT.replace("gate_e2e", "gate_new"))
        if "gate_new: the gate-ptx-spill recipe pins this binary and BIN_CRATES" not in out:
            fails.append(f"unknown binary: {out!r}")
        out = run("table: a binary the recipe does not run", 1, ["gate_old: the table pins this binary, the gate-ptx-spill recipe does not run it"],
                  rows=dict(TABLE_ROWS, gate_old=["k_one"]))
        out = run("refusal: no ptx-spill-check call in the recipe", 1, [], justfile=JUSTFILE_TXT.replace("ptx-spill-check.sh", "other.sh"))
        if "has 0 `ptx-spill-check.sh" not in out:
            fails.append(f"no call: {out!r}")
        out = run("refusal: a malformed table row", 1, [], table="generate_ds41 k_one 0\n")
        if "want `<binary> <entry> <spill> <jit_local>`, got 3 fields" not in out:
            fails.append(f"malformed row: {out!r}")
        # an unreadable tree is rc 2, not a pass.
        empty = tempfile.mkdtemp(dir=top)
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = main_check(empty)
        if rc != 2:
            fails.append(f"an empty root: rc {rc}, want 2")
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
