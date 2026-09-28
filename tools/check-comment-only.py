#!/usr/bin/env python3
"""주석만 바뀌었는지 판정한다(맥에서, 빌드 없이).

    tools/check-comment-only.py <base-ref> <path>...
    tools/check-comment-only.py --keep <base-ref> [--allow FILE] <path>...
    tools/check-comment-only.py --comments <file>...
    tools/check-comment-only.py --self-test

각 파일의 <base-ref> 판과 작업 트리 판에서 주석을 걷어내고 비교한다(공백은 아래 모드별 규칙대로).
같으면 코드는 한 글자도 안 바뀐 것이므로 비트 불변이 구성상 성립한다 — 주석 다이어트
라운드의 완료 조건. 문자열·raw 문자열·문자 리터럴 안의 `//`는 주석이 아니다.

A path may be a directory: every file under it at the base or in the tree. The mode follows the file:
  .rs       comments are `//` and nested `/* */`; literals (strings, raw strings, chars) are compared
            verbatim, whitespace outside them is ignored; a doc comment inside a Rust code fence
            (a doctest: no info, or only rustdoc's words such as compile_fail) is code.
  .sh       a `#` starts a comment only at the start of a word, outside quotes (a line start, after a
            blank or after one of |&;()<>), which is bash's own rule, `[[ ]]` and case patterns
            included: `${x#y}`, `$#`, `a#b` are code. Line 1's `#!` is code. Heredoc bodies are
            compared verbatim, except a quoted-delimiter heredoc of Python, which is compared as
            Python (the heredoc's line names python, or its delimiter starts with PY). `"$( )"`, backticks and `${ }` are compared verbatim. Whitespace runs outside
            quotes count as one blank; blank lines are dropped.
  .py       `ast.dump` with docstrings removed; a file that reads `__doc__` keeps its docstrings in the
            comparison (the docstring is then output, not a comment).
  justfile  `just --dump --dump-format json` without the `doc` fields (the `#` line above a recipe),
            each recipe body compared as shell (a non-sh shebang recipe verbatim).
Where a construct is unclear the lexer falls to "code": a false CODE CHANGED is loud, a false
comment-only is not. Input it cannot read (an unterminated literal, comment or heredoc, a file of
another kind) is refused by name (65), never judged. A changed .md under a directory is listed as a
document, not judged. Comments the compiler reads (`// SAFETY:` under undocumented_unsafe_blocks,
`# Safety` under missing_safety_doc) are not code here: the lint is still part of the proof.

--keep counts, for base and tree, per file, the lines a cleanup must not lose: numeric-contract
comment lines, SAFETY blocks, `PIN(` lines, named-refusal and fault lines, "this order is the gate"
lines (the scan of each is in KEEP). Any count that drops in a file is red unless the allow file
lists it exactly: `<category> <path> <base> <tree> <reason>`, one per line; an allow line that
matches no drop is an error.

--comments prints every comment line as `path:line:text` (a block comment one line each); it is
tools/check-comments.sh's reader.

Exit: 0 comment-only / no count dropped, 1 code changed / a count dropped, 64 usage, 65 unreadable
input.
"""
import ast
import bisect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter


class InputError(Exception):
    """Input this tool cannot read; the message names the file and the line."""


# ----------------------------------------------------------------------------------------------
# Rust
# ----------------------------------------------------------------------------------------------

_RUSTDOC_WORDS = {"rust", "ignore", "should_panic", "no_run", "compile_fail", "test_harness", "standalone_crate"}


def _doctest_fence(info: str) -> bool:
    """rustdoc compiles a fence with no info, or whose every word is one of its own."""
    words = [w for w in re.split(r"[\s,]+", info.strip()) if w]
    return all(w in _RUSTDOC_WORDS or re.fullmatch(r"edition\d+|E\d{4}|ignore-\S+|\{.*\}", w) for w in words)


def _ident(c: str) -> bool:
    return c.isalnum() or c == "_"


def lex_rust(src: str, path: str = "<rust>"):
    """(tokens, comments): tokens are (text, line) with whitespace outside literals dropped and each
    literal kept whole behind a NUL; comments are (line, text), a block comment one entry a line."""
    nl = [k for k, ch in enumerate(src) if ch == "\n"]

    def line_at(k: int) -> int:
        return bisect.bisect_left(nl, k) + 1

    toks, comments = [], []
    run, run_line = [], 0
    fence = None  # (doc marker, doctest?) while a doc comment's code fence is open

    def flush():
        nonlocal run
        if run:
            toks.append(("".join(run), run_line))
            run = []

    def literal(text: str, at: int):
        flush()
        toks.append(("\0" + text + "\0", line_at(at)))

    i, n = 0, len(src)
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            text = src[i:j]
            comments.append((line_at(i), text))
            doc = text[:3] if text[:3] in ("///", "//!") and text[:4] != "////" else None
            if doc:
                body = text[3:].strip()
                if fence and fence[0] == doc and body.startswith("```"):
                    if fence[1]:
                        literal(text, i)
                    fence = None
                elif fence:
                    if fence[1]:
                        literal(text, i)
                elif body.startswith("```"):
                    fence = (doc, _doctest_fence(body[3:]))
                    if fence[1]:
                        literal(text, i)
            else:
                fence = None
            i = j
            continue
        if two == "/*":
            depth, j = 1, i + 2
            while j < n and depth:
                if src[j : j + 2] == "/*":
                    depth, j = depth + 1, j + 2
                elif src[j : j + 2] == "*/":
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            if depth:
                raise InputError(f"{path}:{line_at(i)}: unterminated block comment")
            text = src[i:j]
            first = line_at(i)
            for k, part in enumerate(text.split("\n")):
                comments.append((first + k, part))
            if re.match(r"/\*[*!](?![*/])", text) and "```" in text:
                literal(text, i)  # a doc block with a fence may hold a doctest: code
            i = j
            continue
        if not c.isspace():
            fence = None
        m = re.compile(r"(?:b|c)?r(#*)\"").match(src, i) if c in "bcr" else None
        if m and (i == 0 or not _ident(src[i - 1])):
            close = '"' + m.group(1)
            k = src.find(close, m.end())
            if k < 0:
                raise InputError(f"{path}:{line_at(i)}: unterminated raw string")
            k += len(close)
            literal(src[i:k], i)
            i = k
            continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            if j >= n:
                raise InputError(f"{path}:{line_at(i)}: unterminated string")
            literal(src[i : j + 1], i)
            i = j + 1
            continue
        if c == "'":
            # char literal ('x', '\n', '\'', '\u{..}') vs lifetime or label ('a)
            if src[i + 1 : i + 2] == "\\":
                j = src.find("'", i + 3)
                if j < 0 or "\n" in src[i:j]:
                    raise InputError(f"{path}:{line_at(i)}: unterminated char literal")
                literal(src[i : j + 1], i)
                i = j + 1
                continue
            if src[i + 2 : i + 3] == "'":
                literal(src[i : i + 3], i)
                i += 3
                continue
        if c.isspace():
            flush()
        else:
            if not run:
                run_line = line_at(i)
            run.append(c)
        i += 1
    flush()
    return toks, comments


# ----------------------------------------------------------------------------------------------
# shell
# ----------------------------------------------------------------------------------------------

_META = set("|&;()<>")


def lex_shell(src: str, path: str = "<shell>", line0: int = 0):
    """(lines, comments): lines are (normalized text, line) with comments removed, whitespace runs
    outside quotes as one blank and blank lines dropped; literals are kept whole behind a NUL (their
    newlines as \\x01). comments are (line, text), heredoc Python comments included."""
    n = len(src)
    toks, comments = [], []  # toks: (kind, text, line) kind c|w|l
    line = line0 + 1

    def err(what: str, at_line: int):
        raise InputError(f"{path}:{at_line}: {what}")

    def scan_sq(k: int) -> int:  # after the opening '
        j = src.find("'", k)
        if j < 0:
            err("unterminated single quote", line)
        return j + 1

    def scan_bt(k: int) -> int:  # after the opening `
        while k < n and src[k] != "`":
            k += 2 if src[k] == "\\" else 1
        if k >= n:
            err("unterminated backtick", line)
        return k + 1

    def scan_group(k: int, open_: str, close: str, what: str) -> int:  # after $( or ${
        depth = 1
        while k < n:
            ch = src[k]
            if ch == "\\":
                k += 2
            elif ch == "'":
                k = scan_sq(k + 1)
            elif ch == '"':
                k = scan_dq(k + 1)
            elif ch == "`":
                k = scan_bt(k + 1)
            elif ch == open_:
                depth, k = depth + 1, k + 1
            elif ch == close:
                depth, k = depth - 1, k + 1
                if not depth:
                    return k
            else:
                k += 1
        err(f"unterminated {what}", line)

    def scan_dq(k: int) -> int:  # after the opening "
        while k < n:
            ch = src[k]
            if ch == "\\":
                k += 2
            elif ch == '"':
                return k + 1
            elif src[k : k + 2] == "$(":
                k = scan_group(k + 2, "(", ")", "$( inside double quotes")
            elif src[k : k + 2] == "${":
                k = scan_group(k + 2, "{", "}", "${ inside double quotes")
            elif ch == "`":
                k = scan_bt(k + 1)
            else:
                k += 1
        err("unterminated double quote", line)

    def emit_literal(text: str):
        nonlocal line
        toks.append(("l", text, line))
        line += text.count("\n")

    i, prev = 0, "\n"
    pending = []  # heredocs opened on this line: (delimiter, strip tabs, python body)
    line_start = 0
    if src.startswith("#!"):
        j = src.find("\n")
        j = n if j < 0 else j
        toks.append(("c", src[:j], line))
        i = j
    while i < n:
        c = src[i]
        if c == "\n":
            toks.append(("w", "\n", line))
            line += 1
            i += 1
            prev = "\n"
            line_start = i
            for delim, tabs, python in pending:
                body_start, body_line = i, line
                while True:
                    if i >= n:
                        err(f"unterminated heredoc (no line {delim!r})", body_line - 1)
                    j = src.find("\n", i)
                    j = n if j < 0 else j
                    text = src[i:j]
                    if (text.lstrip("\t") if tabs else text) == delim:
                        break
                    i = j + 1
                body = src[body_start:i]
                code = None
                if python:
                    try:
                        code = "<python>" + py_code(body, f"{path}:{body_line}")[0]
                        comments.extend(py_comments(body, path, body_line - 1))
                    except InputError:
                        code = None
                toks.append(("l", code if code is not None else body, body_line))
                line += body.count("\n")
                toks.append(("c", src[i:j], line))  # the delimiter line
                i = j
            pending = []
            continue
        if c in " \t":
            toks.append(("w", " ", line))
            prev = c
            i += 1
            continue
        if c == "#" and (prev in " \t\n" or prev in _META):
            j = src.find("\n", i)
            j = n if j < 0 else j
            comments.append((line, src[i:j]))
            i = j
            continue
        if c == "\\":
            emit_literal(src[i : i + 2])
            prev = "\\"
            i += 2
            continue
        if c == "'":
            j = scan_sq(i + 1)
        elif src[i : i + 2] == "$'":
            j = i + 2
            while j < n and src[j] != "'":
                j += 2 if src[j] == "\\" else 1
            if j >= n:
                err("unterminated $'", line)
            j += 1
        elif c == '"':
            j = scan_dq(i + 1)
        elif c == "`":
            j = scan_bt(i + 1)
        elif src[i : i + 2] == "${":
            j = scan_group(i + 2, "{", "}", "${")
        else:
            j = None
        if j is not None:
            emit_literal(src[i:j])
            prev = src[j - 1]
            i = j
            continue
        if src[i : i + 3] == "<<<":  # a here-string, not a heredoc
            toks.append(("c", "<<<", line))
            prev = "<"
            i += 3
            continue
        if src[i : i + 2] == "<<":
            m = re.compile(r"<<(-?)[ \t]*((?:[^\s|&;()<>'\"\\]|'[^']*'|\"[^\"]*\"|\\.)+)").match(src, i)
            if not m:
                err("a heredoc operator with no delimiter", line)
            word = m.group(2)
            quoted = any(q in word for q in "'\"\\")
            delim = re.sub(r"['\"\\]", "", word)
            python = quoted and (re.search(r"\bpython3?\b", src[line_start:i]) is not None or re.match(r"PY", delim) is not None)
            pending.append((delim, m.group(1) == "-", python))
            toks.append(("c", m.group(0), line))
            prev = src[m.end() - 1]
            i = m.end()
            continue
        toks.append(("c", c, line))
        prev = c
        i += 1
    if pending:
        err(f"unterminated heredoc (no line {pending[0][0]!r})", line)

    lines, cur, cur_line = [], [], None
    for kind, text, ln in toks + [("w", "\n", line)]:
        if kind == "w" and text == "\n":
            s = "".join(cur).strip(" ")
            if s:
                lines.append((s, cur_line))
            cur, cur_line = [], None
            continue
        if cur_line is None and kind != "w":
            cur_line = ln
        if kind == "w":
            if cur and cur[-1] != " ":
                cur.append(" ")
        elif kind == "l":
            cur.append("\0" + text.replace("\n", "\x01") + "\0")
        else:
            cur.append(text)
    return lines, comments


# ----------------------------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------------------------

_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def py_code(src: str, path: str = "<python>"):
    """(ast.dump with docstrings removed, whether the file reads __doc__ — then they are kept)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise InputError(f"{path}:{e.lineno}: does not parse as Python: {e.msg}") from None
    reads_doc = any(
        (isinstance(x, ast.Name) and x.id == "__doc__") or (isinstance(x, ast.Attribute) and x.attr == "__doc__")
        for x in ast.walk(tree)
    )
    if not reads_doc:
        for x in ast.walk(tree):
            if isinstance(x, _DOC_OWNERS) and x.body and _is_docstring(x.body[0]):
                x.body = x.body[1:]
    return ast.dump(tree), reads_doc


def py_comments(src: str, path: str = "<python>", line0: int = 0):
    """(line, text) of every `#` comment and every docstring line."""
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.COMMENT:
                out.append((line0 + t.start[0], t.string))
        tree = ast.parse(src)
    except (tokenize.TokenError, SyntaxError, IndentationError) as e:
        raise InputError(f"{path}: does not tokenize as Python: {e}") from None
    for x in ast.walk(tree):
        if isinstance(x, _DOC_OWNERS) and x.body and _is_docstring(x.body[0]):
            d = x.body[0]
            src_lines = src.split("\n")[d.lineno - 1 : d.end_lineno]
            out.extend((line0 + d.lineno + k, s.strip()) for k, s in enumerate(src_lines))
    return sorted(out)


# ----------------------------------------------------------------------------------------------
# justfile
# ----------------------------------------------------------------------------------------------


def _just_dump(src: str, path: str) -> dict:
    just = shutil.which("just")
    if not just:
        raise InputError(f"{path}: `just` is not on PATH; the justfile mode reads just's own dump")
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "justfile")
        with open(jf, "w", encoding="utf-8") as fh:
            fh.write(src)
        p = subprocess.run([just, "--justfile", jf, "--working-directory", d, "--dump", "--dump-format", "json"],
                           capture_output=True, text=True)
    if p.returncode:
        raise InputError(f"{path}: `just --dump` failed: {p.stderr.strip()}")
    return json.loads(p.stdout)


def _drop_docs(x):
    if isinstance(x, dict):
        return {k: _drop_docs(v) for k, v in x.items() if k != "doc"}
    if isinstance(x, list):
        return [_drop_docs(v) for v in x]
    return x


def just_code(src: str, path: str = "justfile") -> str:
    """just's dump without docs and source, each recipe body as its shell lines."""
    dump = _drop_docs(_just_dump(src, path))
    dump.pop("source", None)
    for name, rec in dump.get("recipes", {}).items():
        exprs, lines = [], []
        for frags in rec["body"]:
            s = ""
            for f in frags:
                if isinstance(f, str):
                    s += f
                else:
                    s += f"\x02{len(exprs)}\x02"
                    exprs.append(f)
            lines.append(s)
        text = "\n".join(lines)
        if rec.get("shebang") and not re.match(r"#!\S*(?:\bbash|/sh|\benv\s+(?:ba)?sh)\b", text):
            body = [text]  # not shell: verbatim
        else:
            body = [s for s, _ in lex_shell(text, f"{path} recipe {name}")[0]]
        rec["body"] = {"shell": body, "exprs": exprs}
    return json.dumps(dump, sort_keys=True, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------
# one entry per mode
# ----------------------------------------------------------------------------------------------


def just_comments(src: str, path: str = "justfile"):
    """(line, text) of the justfile's comments: a top-level line lexed alone, each recipe body (a run of
    indented lines) dedented as just runs it and lexed as shell."""
    lines = src.split("\n")
    out, k = [], 0
    while k < len(lines):
        if lines[k][:1] in (" ", "\t"):
            j = k
            while j < len(lines) and (lines[j][:1] in (" ", "\t") or (not lines[j].strip() and j + 1 < len(lines) and lines[j + 1][:1] in (" ", "\t"))):
                j += 1
            block = lines[k:j]
            ind = min(len(b) - len(b.lstrip()) for b in block if b.strip())
            out.extend(lex_shell("\n".join(b[ind:] for b in block), path, k)[1])
            k = j
        else:
            out.extend(lex_shell(lines[k], path, k)[1])
            k += 1
    return out


def lang_of(path: str):
    base = os.path.basename(path)
    if base == "justfile":
        return "just"
    ext = os.path.splitext(base)[1]
    return {".rs": "rust", ".sh": "shell", ".py": "python"}.get(ext)


def code_units(lang: str, src: str, path: str):
    """The comparable code of a file: a list of (text, line)."""
    if lang == "rust":
        return lex_rust(src, path)[0]
    if lang == "shell":
        return lex_shell(src, path)[0]
    if lang == "python":
        return [(py_code(src, path)[0], 0)]
    if lang == "just":
        return [(just_code(src, path), 0)]
    raise InputError(f"{path}: no mode for this file (.rs, .sh, .py, justfile)")


def comments_of(lang: str, src: str, path: str):
    if lang == "rust":
        return lex_rust(src, path)[1]
    if lang == "shell":
        return lex_shell(src, path)[1]
    if lang == "just":
        return just_comments(src, path)
    if lang == "python":
        return py_comments(src, path)
    raise InputError(f"{path}: no mode for this file (.rs, .sh, .py, justfile)")


def _show(text: str) -> str:
    return text.replace("\0", "⟦").replace("\x01", "⏎").replace("\x02", "¦")


def compare(lang: str, old: str, new: str, path: str):
    """None when only comments differ, else a message naming where the code moved."""
    a, b = code_units(lang, old, path), code_units(lang, new, path)
    if lang in ("rust", "shell"):
        sa, sb = [t for t, _ in a], [t for t, _ in b]
        if lang == "rust":
            ja, jb = "".join(sa), "".join(sb)
            if ja == jb:
                return None
            k = next((x for x in range(min(len(ja), len(jb))) if ja[x] != jb[x]), min(len(ja), len(jb)))

            def where(units, joined_k):
                ends, tot = [], 0
                for t, _ in units:
                    tot += len(t)
                    ends.append(tot)
                u = min(bisect.bisect_right(ends, joined_k), len(units) - 1) if units else 0
                return units[u][1] if units else 0

            return (f"base line {where(a, k)}, tree line {where(b, k)}\n"
                    f"  base: …{_show(ja[max(0, k - 60) : k + 60])}…\n  now:  …{_show(jb[max(0, k - 60) : k + 60])}…")
        if sa == sb:
            return None
        k = next((x for x in range(min(len(sa), len(sb))) if sa[x] != sb[x]), min(len(sa), len(sb)))
        la = f"{a[k][1]}: {_show(a[k][0])[:160]}" if k < len(a) else "(end)"
        lb = f"{b[k][1]}: {_show(b[k][0])[:160]}" if k < len(b) else "(end)"
        return f"first differing code line\n  base {la}\n  now  {lb}"
    ja, jb = a[0][0], b[0][0]
    if ja == jb:
        return None
    note = ""
    if lang == "python" and (py_code(old, path)[1] or py_code(new, path)[1]):
        note = " (the file reads __doc__, so its docstrings are output and were compared)"
    k = next((x for x in range(min(len(ja), len(jb))) if ja[x] != jb[x]), min(len(ja), len(jb)))
    return (f"the {lang} structure differs{note}\n"
            f"  base: …{_show(ja[max(0, k - 80) : k + 80])}…\n  now:  …{_show(jb[max(0, k - 80) : k + 80])}…")


# ----------------------------------------------------------------------------------------------
# keep counts
# ----------------------------------------------------------------------------------------------

# The survey's two numeric-contract scans, flags unchanged, as one union over comment text: the
# crates/*/src one (cgrep.py numeric-contract, Python, case-insensitive) and the gates one (gates.md
# Conflicts 6, grep -E, case-sensitive).
_NUMERIC_SRC = re.compile(r"(saturat|−128|-128|\bi16\b.*(limit|range|max)|overflow|fits (in|an?)|at most \d|\b2\^\d+|≤ ?\d|<= ?\d{2,}|code domain|max\|)", re.I)
_NUMERIC_GATES = re.compile(r"(\b[0-9.]+u\b|γ|2\^-|ulp|±127|−128|-128|saturat|clamp)", re.A)  # grep's ASCII \b: `4u²` is a hit

# category: (what it scans, how it matches a line of that text)
KEEP = {
    "numeric": ("comment", lambda s: bool(_NUMERIC_SRC.search(s) or _NUMERIC_GATES.search(s))),
    "safety": ("comment", lambda s: bool(re.match(r"(//[/!]?|/\*+!?|#)\s*SAFETY", s.strip()))),
    "pin": ("comment", lambda s: "PIN(" in s),
    "refusal": ("line", lambda s: bool(re.search(r"by name|named error", s, re.I))),
    "fault": ("line", lambda s: bool(re.search(r"FaultSite::|fault word|\.fault\(|take_fault|read_fault", s))),
    "order": ("comment", lambda s: bool(re.search(r"order is the gate", s, re.I))),
}
_KEEP_DOC = {
    "numeric": "comment lines, the survey's src and gates numeric-contract regexes as one union",
    "safety": "SAFETY blocks: comment lines that open with SAFETY",
    "pin": "comment lines with `PIN(`",
    "refusal": "whole lines with `by name` or `named error`",
    "fault": "whole lines with FaultSite::, fault word, .fault(, take_fault, read_fault",
    "order": "comment lines with `order is the gate`",
}


def keep_counts(lang: str, src: str, path: str) -> Counter:
    c = Counter()
    comment_lines = {}
    for ln, text in comments_of(lang, src, path):
        comment_lines.setdefault(ln, []).append(text)
    for cat, (scope, hit) in KEEP.items():
        if scope == "comment":
            c[cat] = sum(1 for texts in comment_lines.values() if any(hit(t) for t in texts))
        else:
            c[cat] = sum(1 for s in src.split("\n") if hit(s))
    return c


def read_allow(path: str):
    out = []
    with open(path, encoding="utf-8") as fh:
        for k, raw in enumerate(fh, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            f = s.split(None, 4)
            if len(f) < 5 or f[0] not in KEEP or not f[2].isdigit() or not f[3].isdigit() or not f[4].strip():
                raise InputError(f"{path}:{k}: an allow line is `<category> <path> <base> <tree> <reason>` "
                                 f"(category one of {', '.join(KEEP)}), got: {s}")
            out.append((f[0], f[1], int(f[2]), int(f[3]), f[4], k))
    return out


def judge(counts: dict, allows):
    """counts: path -> (base Counter, tree Counter). (unallowed drops, allow lines that match none)."""
    drops = [(cat, p, b[cat], t[cat]) for p, (b, t) in sorted(counts.items()) for cat in KEEP if t[cat] < b[cat]]
    allowed = {(a[0], a[1], a[2], a[3]) for a in allows}
    bad = [d for d in drops if d not in allowed]
    stale = [a for a in allows if (a[0], a[1], a[2], a[3]) not in set(drops)]
    return bad, stale


# ----------------------------------------------------------------------------------------------
# git side
# ----------------------------------------------------------------------------------------------


def _git(*args, cwd=None) -> str:
    p = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if p.returncode:
        raise InputError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def _top() -> str:
    return _git("rev-parse", "--show-toplevel").strip()


def _rel(top: str, p: str) -> str:
    return os.path.relpath(os.path.abspath(p), top)


def expand(top: str, base: str, paths):
    """{repo-relative file: named itself?}: each path itself, or every file under a directory at the base
    or in the tree."""
    out = {}
    for p in paths:
        rel = _rel(top, p)
        if os.path.isdir(os.path.join(top, rel)) or _git("ls-tree", "-d", "--name-only", base, "--", rel, cwd=top).strip() == rel:
            at_base = _git("ls-tree", "-r", "--name-only", base, "--", rel, cwd=top).split("\n")
            in_tree = _git("ls-files", "--cached", "--others", "--exclude-standard", "--", rel, cwd=top).split("\n")
            for f in sorted({f for f in at_base + in_tree if f}):
                out.setdefault(f, False)
        else:
            out[rel] = True
    return out


def base_bytes(top: str, base: str, rel: str):
    p = subprocess.run(["git", "show", f"{base}:{rel}"], capture_output=True, cwd=top)
    return p.stdout if p.returncode == 0 else None


def tree_bytes(top: str, rel: str):
    f = os.path.join(top, rel)
    if not os.path.isfile(f):
        return None
    with open(f, "rb") as fh:
        return fh.read()


def text_of(data, rel: str, side: str):
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InputError(f"{rel}: the {side} copy is not UTF-8 ({e})") from None


def check_base(top: str, base: str):
    _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", cwd=top)


def main_proof(base: str, paths) -> int:
    top = _top()
    check_base(top, base)
    bad = same = judged = 0
    for rel, named in expand(top, base, paths).items():
        old, new = base_bytes(top, base, rel), tree_bytes(top, rel)
        if old is None and new is None:
            raise InputError(f"{rel}: neither at {base} nor in the tree")
        if old == new:
            same += 1
            continue
        lang = lang_of(rel)
        if lang is None:
            if rel.endswith(".md"):
                print(f"document, not judged: {rel}")
                continue
            if named:
                raise InputError(f"{rel}: no mode for this file (.rs, .sh, .py, justfile)")
            print(f"NO MODE: {rel} changed and no mode reads it (.rs, .sh, .py, justfile)", file=sys.stderr)
            bad += 1
            continue
        if old is None or new is None:
            print(f"CODE CHANGED: {rel} ({'added' if old is None else 'deleted'})", file=sys.stderr)
            bad += 1
            continue
        judged += 1
        msg = compare(lang, text_of(old, rel, "base"), text_of(new, rel, "tree"), rel)
        if msg is None:
            print(f"comment-only: {rel}")
        else:
            print(f"CODE CHANGED: {rel}: {msg}", file=sys.stderr)
            bad += 1
    print(f"check-comment-only: {judged} changed files judged, {same} unchanged, {bad} not comment-only")
    return 1 if bad else 0


def main_keep(base: str, allow_path, paths) -> int:
    top = _top()
    check_base(top, base)
    allows = read_allow(allow_path) if allow_path else []
    counts, skipped, langs = {}, Counter(), Counter()
    for rel in expand(top, base, paths):
        lang = lang_of(rel)
        if lang is None:
            skipped[os.path.splitext(rel)[1] or os.path.basename(rel)] += 1
            continue
        old, new = text_of(base_bytes(top, base, rel), rel, "base"), text_of(tree_bytes(top, rel), rel, "tree")
        langs[lang] += 1
        zero = Counter({k: 0 for k in KEEP})
        counts[rel] = (keep_counts(lang, old, rel) if old is not None else zero,
                       keep_counts(lang, new, rel) if new is not None else zero)
    print(f"keep-count: base {base}, {len(counts)} files ({', '.join(f'{k} {v}' for k, v in sorted(langs.items()))})"
          + (f"; no mode, not counted: {', '.join(f'{k} {v}' for k, v in sorted(skipped.items()))}" if skipped else ""))
    print("file\t" + "\t".join(KEEP))
    for rel, (b, t) in counts.items():
        if any(b[c] or t[c] for c in KEEP):
            cells = [str(b[c]) if b[c] == t[c] else f"{b[c]}->{t[c]}" for c in KEEP]
            print(rel + "\t" + "\t".join(cells))
    print("category\tbase\ttree\tscans")
    for c in KEEP:
        print(f"{c}\t{sum(b[c] for b, _ in counts.values())}\t{sum(t[c] for _, t in counts.values())}\t{_KEEP_DOC[c]}")
    bad, stale = judge(counts, allows)
    sys.stdout.flush()
    for cat, p, nb, nt in bad:
        print(f"DROP {cat} {p} {nb} {nt} — not in the allow file", file=sys.stderr)
    for a in stale:
        print(f"STALE ALLOW {allow_path}:{a[5]}: {a[0]} {a[1]} {a[2]} {a[3]} matches no drop", file=sys.stderr)
    if bad or stale:
        print(f"keep-count: red — {len(bad)} drop(s) not allowed, {len(stale)} allow line(s) matching nothing", file=sys.stderr)
        return 1
    print(f"keep-count: ok — no count dropped{f' beyond {len(allows)} allowed' if allows else ''}")
    return 0


def main_comments(files) -> int:
    for f in files:
        lang = lang_of(f)
        if lang is None:
            raise InputError(f"{f}: no mode for this file (.rs, .sh, .py, justfile)")
        with open(f, encoding="utf-8") as fh:
            src = fh.read()
        for ln, text in comments_of(lang, src, f):
            print(f"{f}:{ln}:{text}")
    return 0


# ----------------------------------------------------------------------------------------------
# self-test: each mode's comment-only cases, and a mutant of each that must not pass
# ----------------------------------------------------------------------------------------------

_RUST_BASE = r'''//! Module doc.
/// ```compile_fail,E0308
/// let x: u8 = "";
/// ```
/// ```text
/// a picture
/// ```
fn f<'a>(s: &'a str) -> u8 { // trailing
    let t = "a // b  c"; /* block /* nested */ still */
    let r = br#"raw "// x" y"#;
    let c = '"'; let e = '\''; let u = '\u{1F600}';
    b'x' + 1 // one
}
'''
_SHELL_BASE = r'''#!/usr/bin/env bash
# header
set -- 1 2
echo a#b ${x#y} $# "$#" ${#x} # trailing
[[ $x == a#b ]] && echo "${v:-a #b}" $'q\'r' # it's
case $x in a) echo "$(echo a # inside
)";; esac
cat << 'EOF'
# data, not a comment
EOF
python3 - "$1" << 'PY'
x = 1  # py comment
"""not a docstring position"""
PY
echo   spaced   out
'''
_PY_BASE = '''"""Module doc."""
import os  # why


def f(a):
    """Function doc."""
    return a + 1  # sum
'''
_PY_DOC = '''"""Usage: x."""
import sys
print(__doc__)
'''
_JUST_BASE = '''# section header

v := "x"

# the doc of a
a:
    echo {{v}} one  # trailing
    # a body comment

b: a
    #!/usr/bin/env bash
    set -e  # strict
    echo two
'''


def self_test() -> int:
    fails = []

    def expect(lang, old, new, same, name):
        try:
            got = compare(lang, old, new, name) is None
        except InputError as e:
            fails.append(f"{name}: refused: {e}")
            return
        if got != same:
            fails.append(f"{name}: expected {'comment-only' if same else 'CODE CHANGED'}")

    def refused(fn, src, name):
        try:
            fn(src)
        except InputError:
            return
        fails.append(f"{name}: not refused")

    R = _RUST_BASE
    expect("rust", R, R.replace("// trailing", "// another").replace("still */", "gone */").replace("Module doc.", "M."), True, "rust comments")
    expect("rust", R, R.replace("/// a picture", "/// another picture"), True, "rust text fence")
    expect("rust", R, R.replace("fn f<'a>(s: &'a str) -> u8 { // trailing\n", "fn f<'a>(\n    s: &'a str\n) -> u8 {\n"), True, "rust whitespace")
    expect("rust", R, R.replace('"a // b  c"', '"a // b c"'), False, "rust mutant: whitespace inside a string")
    expect("rust", R, R.replace('y"#', 'z"#'), False, "rust mutant: inside a raw byte string")
    expect("rust", R, R.replace("'\\''", "'\\\"'"), False, "rust mutant: an escaped char literal")
    expect("rust", R, R.replace('/// let x: u8 = "";', '/// let x: u16 = "";'), False, "rust mutant: inside a doctest")
    expect("rust", R, R.replace("+ 1 // one", "+ 2 // one"), False, "rust mutant: code after a nested comment")
    refused(lambda s: lex_rust(s), 'fn f() { let s = "open; }', "rust: unterminated string")
    refused(lambda s: lex_rust(s), "fn f() { /* open /* x */ }", "rust: unterminated block comment")
    refused(lambda s: lex_rust(s), "fn f() { let c = '\\", "rust: unterminated escaped char")
    got = [ln for ln, _ in lex_rust("a /* x\ny */ b // c\nlet s = \"// no\";")[1]]
    if got != [1, 2, 2]:
        fails.append(f"rust comments list: lines {got}, expected [1, 2, 2]")

    S = _SHELL_BASE
    expect("shell", S, S.replace("# header", "# changed").replace("# trailing", "# other").replace("# it's", "#"), True, "shell comments")
    expect("shell", S, S.replace("x = 1  # py comment", "x = 1  # another"), True, "shell python heredoc comment")
    expect("shell", S, S.replace("echo   spaced   out", "echo spaced out\n\n"), True, "shell whitespace")
    expect("shell", S, S.replace("echo a#b", "echo a#c"), False, "shell mutant: # inside a word")
    expect("shell", S, S.replace("${x#y}", "${x#z}"), False, "shell mutant: ${x#y}")
    expect("shell", S, S.replace("[[ $x == a#b ]]", "[[ $x == a#c ]]"), False, "shell mutant: # inside [[ ]]")
    expect("shell", S, S.replace('"${v:-a #b}"', '"${v:-a #c}"'), False, "shell mutant: # inside a quoted ${ }")
    expect("shell", S, S.replace("# data, not a comment", "# data changed"), False, "shell mutant: heredoc body")
    expect("shell", S, S.replace("x = 1  # py", "x = 2  # py"), False, "shell mutant: python heredoc code")
    expect("shell", S, S.replace("#!/usr/bin/env bash", "#!/bin/sh"), False, "shell mutant: shebang")
    expect("shell", S, S.replace("echo   spaced   out", "echo spacedout"), False, "shell mutant: a blank between words")
    expect("shell", S, S.replace("# inside", "# changed"), False, "shell mutant: \"$( )\" is compared verbatim")
    refused(lambda s: lex_shell(s), "echo 'open\n", "shell: unterminated single quote")
    refused(lambda s: lex_shell(s), 'echo "a $(b "c")\n', "shell: unterminated double quote")
    refused(lambda s: lex_shell(s), "cat <<EOF\nno end\n", "shell: unterminated heredoc")
    got = [t for _, t in lex_shell(S)[1]]
    if got != ["# header", "# trailing", "# it's", "# py comment"]:
        fails.append(f"shell comments list: {got}")

    P = _PY_BASE
    expect("python", P, P.replace("Module doc.", "Other.").replace("# why", "").replace("Function doc.", "F."), True, "python docstrings and comments")
    expect("python", P, P.replace("a + 1", "a + 2"), False, "python mutant: code")
    expect("python", P, P.replace('"""Function doc."""\n', ""), True, "python: a docstring removed")
    expect("python", _PY_DOC, _PY_DOC.replace("Usage: x.", "Usage: y."), False, "python mutant: a docstring the file prints")
    refused(lambda s: py_code(s), "def f(:\n", "python: does not parse")

    J = _JUST_BASE
    if shutil.which("just"):
        expect("just", J, J.replace("# the doc of a", "# a new doc").replace("# section header", "# other").replace("# trailing", "# x").replace("# a body comment", "# y"), True, "justfile docs and comments")
        expect("just", J, J.replace("# strict", "# strictly"), True, "justfile shebang body comment")
        expect("just", J, J.replace("echo {{v}} one", "echo {{v}} won"), False, "justfile mutant: body")
        expect("just", J, J.replace('v := "x"', 'v := "y"'), False, "justfile mutant: assignment")
        expect("just", J, J.replace("b: a", "b:"), False, "justfile mutant: dependency")
        expect("just", J, J.replace("#!/usr/bin/env bash", "#!/usr/bin/env python3"), False, "justfile mutant: shebang")
    else:
        fails.append("justfile cases: `just` is not on PATH")

    base = Counter(keep_counts("rust", "// saturates at i16 max\n// SAFETY: x\n// PIN(2026-01-01): y\n"
                                       "// this order is the gate\nlet e = named(\"refused by name\");\nFaultSite::A;\n", "t.rs"))
    want = {"numeric": 1, "safety": 1, "pin": 1, "refusal": 1, "fault": 1, "order": 1}
    if dict(base) != want:
        fails.append(f"keep counts: {dict(base)}, expected {want}")
    tree = Counter(base)
    tree["numeric"] -= 1
    bad, stale = judge({"t.rs": (base, tree)}, [])
    if bad != [("numeric", "t.rs", 1, 0)] or stale:
        fails.append(f"keep: a numeric drop not red: {bad} {stale}")
    bad, stale = judge({"t.rs": (base, tree)}, [("numeric", "t.rs", 1, 0, "merged", 1)])
    if bad or stale:
        fails.append("keep: an allowed drop is red")
    bad, stale = judge({"t.rs": (base, base)}, [("numeric", "t.rs", 1, 0, "merged", 1)])
    if not stale:
        fails.append("keep: a stale allow line passed")
    bad, stale = judge({"t.rs": (base, tree)}, [("numeric", "t.rs", 2, 0, "wrong counts", 1)])
    if not bad:
        fails.append("keep: an allow line with other counts covered a drop")
    if keep_counts("rust", 'let s = "// saturates";\n', "t.rs")["numeric"]:
        fails.append("keep: a string counted as a numeric-contract comment")

    for f in fails:
        print(f"check-comment-only self-test FAIL: {f}", file=sys.stderr)
    if fails:
        print(f"check-comment-only: self-test {len(fails)} failed", file=sys.stderr)
        return 1
    print("check-comment-only: self-test ok")
    return 0


def main(argv) -> int:
    try:
        if argv[:1] == ["--self-test"]:
            if len(argv) != 1:
                print("check-comment-only: --self-test takes nothing", file=sys.stderr)
                return 64
            return self_test()
        if argv[:1] == ["--comments"]:
            if len(argv) < 2:
                print(__doc__, file=sys.stderr)
                return 64
            return main_comments(argv[1:])
        if argv[:1] == ["--keep"]:
            rest, allow = argv[1:], None
            if "--allow" in rest:
                k = rest.index("--allow")
                if k + 1 >= len(rest):
                    print("check-comment-only: --allow takes a file", file=sys.stderr)
                    return 64
                allow = rest[k + 1]
                rest = rest[:k] + rest[k + 2 :]
            if len(rest) < 2:
                print(__doc__, file=sys.stderr)
                return 64
            return main_keep(rest[0], allow, rest[1:])
        if len(argv) < 2 or argv[0].startswith("-"):
            print(__doc__, file=sys.stderr)
            return 64
        return main_proof(argv[0], argv[1:])
    except InputError as e:
        print(f"check-comment-only: {e}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
