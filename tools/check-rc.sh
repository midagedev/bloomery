#!/usr/bin/env bash
# `$?` read after a command substitution in the same command is the substitution's status (Mac, no build).
# `echo "$(basename "$c") rc=$?"` prints basename's status, always 0, never the status of the command before it:
# in bash a command's words expand left to right, and each `$( … )` or backtick sets `$?` as it ends.
# Scope: every `*.sh` under tools/ and every recipe body of the justfile, read as a stream (a quoted string, an
# argument list, an assignment list, a continued line or a string that spans lines is one command).
# Red: `$?` or `${?}` after a `$( … )` or a backtick that closed earlier in the same simple command, and a `$?`
# at the start of a substitution that follows one (a substitution reads the shell's status as it was forked).
# Not red: `$?` before the substitution; `$?` after `;` `&&` `||` `|` `&` or a newline (a new command); `$?` inside the
# substitution after its own first command (`$(foo; echo $?)`); comments; single quotes (but for the argument of
# `box.sh`, `eval` or `sh -c`, below); a heredoc with a quoted delimiter; `$(( … ))` and `<( … )`, which leave `$?`
# alone. Inside `[[ … ]]` `&&` and `||` do not end the command.
# A line that is right on purpose ends with `# check-rc: <reason>` (a reason after the colon); the hit's own physical
# line is the one read, and the check prints how many hits it allowed that way.
# Exit codes: 0 clean; 1 hits (each `path:line: …`); 2 the tool's own error (a file it cannot read, or text it cannot
# close: an unterminated quote, substitution or heredoc, named by file and line); 64 usage.
# Known blind spots (not parsed, so not checked): single-quoted text, except a whole word given to `box.sh`, `eval` or
# `sh -c` / `bash -c`, which is read as its own script (so `ssh h '…'`, `trap '…'`, a script held in a variable, a
# composed `'…'"$x"'…'` word and a quoted heredoc are data); `${v:-$(cmd)}` counts as a substitution though the default
# may not run; process substitution is not counted; a status carried through a function return or a variable; a `case`
# whose keywords are quoted or expanded; a backtick nested more than one escape deep; a `$((` that bash would re-read
# as `$( (`; a justfile recipe is read as one script with `{{ … }}` opaque.
# `--self-test` runs fixtures from a temp dir.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
  "" | --self-test) [ "$#" -le 1 ] || { echo "usage: tools/check-rc.sh [--self-test]" >&2; exit 64; } ;;
  *) echo "usage: tools/check-rc.sh [--self-test] ('$1' is not one)" >&2; exit 64 ;;
esac
exec python3 -I - "$ROOT" "${1:-}" << 'PY'
import bisect
import os
import re
import sys
import tempfile
import textwrap

ALLOW_RE = re.compile(r"#\s*check-rc:\s*\S")
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
CMD_KW = {"if", "then", "else", "elif", "while", "until", "do", "!", "{", "time"}
SHELLS = {"bash", "sh", "dash", "zsh", "ksh"}
NAME_CH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


class ParseError(Exception):
    def __init__(self, line, msg):
        Exception.__init__(self, msg)
        self.line = line
        self.msg = msg


class Ctx:
    """One command list: the state of the simple command being read."""

    def __init__(self, taint):
        self.taint = taint  # 0, or the line of the substitution whose status `$?` holds now
        self.cmd_pos = True
        self.wtext = ""
        self.wopen = False
        self.wcomplex = False
        self.last = ""  # the last plain word that ended
        self.words = []  # the words of the simple command so far ("" for one that is not plain)
        self.parens = []  # "g" group, "a" array literal
        self.cases = []  # per open `case`: ["word" | "pat" | "body"]
        self.dbl = False  # inside [[ ... ]]
        self.pending = []  # heredocs whose bodies follow the line

    def start_command(self):
        self.cmd_pos = True
        self.words = []


class Scanner:
    def __init__(self, text, base, hits):
        self.s = text
        self.n = len(text)
        self.i = 0
        self.base = base
        self.hits = hits
        self.nl = [m.start() for m in re.finditer("\n", text)]

    def line(self, pos):
        return self.base + bisect.bisect_left(self.nl, pos)

    def hit(self, pos, tline):
        self.hits.append((self.line(pos), tline))

    # -- words and separators ---------------------------------------------------------------------------

    def touch(self, c, complex_word=True):
        c.wopen = True
        if complex_word:
            c.wcomplex = True

    def end_word(self, c):
        if not c.wopen:
            return
        w = None if c.wcomplex else c.wtext
        c.wopen = False
        c.wtext = ""
        c.wcomplex = False
        was_cmd = c.cmd_pos
        top = c.cases[-1] if c.cases else None
        if w is not None:
            c.last = w
            if was_cmd and w == "case":
                c.cases.append(["word"])
                c.cmd_pos = False
                return
            if top and top[0] == "word" and w == "in":
                top[0] = "pat"
                return
            if top and top[0] == "pat" and w == "esac":
                c.cases.pop()
                c.cmd_pos = False
                return
            if top and top[0] == "body" and was_cmd and w == "esac":
                c.cases.pop()
                c.cmd_pos = False
                return
            if was_cmd and w == "[[":
                c.dbl = True
                c.cmd_pos = False
                return
            if w == "]]" and c.dbl:
                c.dbl = False
                c.cmd_pos = False
                return
            if was_cmd and (w in CMD_KW or ASSIGN_RE.match(w)):
                return
        else:
            c.last = ""
        c.words = [w or ""] if was_cmd else c.words + [w or ""]
        c.cmd_pos = False

    def sep(self, c, reset=True):
        self.end_word(c)
        c.start_command()
        if reset:
            c.taint = 0

    # -- quotes and expansions --------------------------------------------------------------------------

    def parse_dq(self, c, quoted, open_line):
        """Inside double quotes (self.i is past the opening quote), or a heredoc body when not quoted."""
        s = self.s
        while self.i < self.n:
            ch = s[self.i]
            if ch == "\\":
                self.i += 2
            elif quoted and ch == '"':
                self.i += 1
                return
            elif ch == "$":
                self.dollar(c, True)
            elif ch == "`":
                self.backtick(c, quoted)
            else:
                self.i += 1
        if quoted:
            raise ParseError(open_line, "unterminated double quote")

    def dollar(self, c, in_dq):
        s, i = self.s, self.i
        nx = s[i + 1 : i + 2]
        if nx == "?":
            if c.taint:
                self.hit(i, c.taint)
            self.i = i + 2
        elif nx == "(":
            if s.startswith("(", i + 2):
                e = self.arith_end(i + 3)
                if e >= 0:
                    self.arith(c, i + 3, e)
                    return
            open_line = self.line(i)
            self.i = i + 2
            self.parse_list(")", c.taint, open_line)
            c.taint = open_line
        elif nx == "{":
            self.i = i + 2
            self.brace(c, in_dq)
        elif nx == "'" and not in_dq:
            j = i + 2
            while j < self.n and s[j] != "'":
                j += 2 if s[j] == "\\" else 1
            if j >= self.n:
                raise ParseError(self.line(i), "unterminated $'…' string")
            self.i = j + 1
        elif nx and nx in "$!@*#-0123456789":
            self.i = i + 2
        elif nx and nx in NAME_CH:
            j = i + 1
            while j < self.n and s[j] in NAME_CH:
                j += 1
            self.i = j
        else:
            self.i = i + 1

    def brace(self, c, in_dq):
        """After `${`: to the matching `}`; `${?}` reads `$?`; `$(` inside counts as a substitution."""
        s = self.s
        if s.startswith("?", self.i) and c.taint:
            self.hit(self.i, c.taint)
        open_line = self.line(self.i)
        while self.i < self.n:
            ch = s[self.i]
            if ch == "}":
                self.i += 1
                return
            if ch == "\\":
                self.i += 2
            elif ch == "$":
                self.dollar(c, in_dq)
            elif ch == "`":
                self.backtick(c, in_dq)
            elif ch == '"':
                self.i += 1
                self.parse_dq(c, True, self.line(self.i))
            elif ch == "'" and not in_dq:
                j = s.find("'", self.i + 1)
                if j < 0:
                    raise ParseError(self.line(self.i), "unterminated single quote")
                self.i = j + 1
            else:
                self.i += 1
        raise ParseError(open_line, "unterminated ${ … }")

    def backtick(self, c, in_dq):
        s, i = self.s, self.i
        j = i + 1
        while j < self.n and s[j] != "`":
            j += 2 if s[j] == "\\" else 1
        if j >= self.n:
            raise ParseError(self.line(i), "unterminated backtick")
        raw = s[i + 1 : j]
        inner = re.sub(r'\\([$`\\"])' if in_dq else r"\\([$`\\])", r"\1", raw)
        sub = Scanner(inner, self.line(i + 1), self.hits)
        sub.parse_list(None, c.taint, self.line(i))
        c.taint = self.line(i)
        self.i = j + 1

    def arith_end(self, start):
        """The index past the `))` that closes an arithmetic body starting at `start`, or -1."""
        s = self.s
        depth = 2
        j = start
        while j < self.n:
            ch = s[j]
            if ch == "\\":
                j += 2
                continue
            if ch in "'\"":
                k = s.find(ch, j + 1)
                if k < 0:
                    return -1
                j = k + 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return j + 1 if s[j - 1] == ")" and j - 1 >= start else -1
            j += 1
        return -1

    def arith(self, c, start, end):
        s = self.s
        self.i = start
        while self.i < end - 2:
            ch = s[self.i]
            if ch == "$":
                self.dollar(c, False)
            elif ch == "`":
                self.backtick(c, False)
            elif ch == "\\":
                self.i += 2
            else:
                self.i += 1
        self.i = end

    # -- heredocs ---------------------------------------------------------------------------------------

    def heredoc_start(self, c):
        """At `<<` (not `<<<`): read the delimiter word and queue the body."""
        s = self.s
        j = self.i + 2
        strip = s.startswith("-", j)
        if strip:
            j += 1
        while j < self.n and s[j] in " \t":
            j += 1
        delim = ""
        quoted = False
        while j < self.n and s[j] not in " \t\n;&|()<>":
            ch = s[j]
            if ch == "'" or ch == '"':
                k = s.find(ch, j + 1)
                if k < 0:
                    raise ParseError(self.line(j), "unterminated quote in a heredoc delimiter")
                delim += s[j + 1 : k]
                quoted = True
                j = k + 1
            elif ch == "\\":
                delim += s[j + 1 : j + 2]
                quoted = True
                j += 2
            else:
                delim += ch
                j += 1
        if not delim:
            raise ParseError(self.line(self.i), "heredoc with no delimiter")
        c.pending.append((delim, quoted, strip, self.line(self.i)))
        self.i = j

    def read_heredocs(self, c):
        """self.i is past the newline that ends the line the heredocs were declared on."""
        s = self.s
        for delim, quoted, strip, dline in c.pending:
            start = self.i
            while True:
                if self.i >= self.n:
                    raise ParseError(dline, "heredoc <<%s is not terminated" % delim)
                j = s.find("\n", self.i)
                if j < 0:
                    j = self.n
                text = s[self.i : j]
                if (text.lstrip("\t") if strip else text) == delim:
                    body_end = self.i
                    self.i = min(j + 1, self.n)
                    break
                self.i = min(j + 1, self.n)
            if not quoted:
                sub = Scanner(s[start:body_end], self.line(start), self.hits)
                sub.parse_dq(Ctx(c.taint), False, dline)
        c.pending = []

    # -- the command list -------------------------------------------------------------------------------

    def parse_list(self, closer, taint0, open_line):
        s, n = self.s, self.n
        c = Ctx(taint0)
        while self.i < n:
            i = self.i
            ch = s[i]
            if ch == " " or ch == "\t":
                self.end_word(c)
                self.i += 1
            elif ch == "\n":
                self.end_word(c)
                self.i += 1
                if c.pending:
                    self.read_heredocs(c)
                if not (c.parens and c.parens[-1] == "a"):
                    c.taint = 0
                    c.start_command()
            elif ch == "\\":
                if s.startswith("\n", i + 1):
                    self.i += 2
                else:
                    self.touch(c)
                    self.i += 2
            elif ch == "'":
                j = s.find("'", i + 1)
                if j < 0:
                    raise ParseError(self.line(i), "unterminated single quote")
                if not c.wopen and self.carries_shell(c) and (j + 1 >= n or s[j + 1] in " \t\n;&|)<>"):
                    Scanner(s[i + 1 : j], self.line(i + 1), self.hits).parse_list(None, 0, self.line(i))
                self.touch(c)
                self.i = j + 1
            elif ch == '"':
                self.touch(c)
                self.i += 1
                self.parse_dq(c, True, self.line(i))
            elif ch == "$":
                self.touch(c)
                self.dollar(c, False)
            elif ch == "`":
                self.touch(c)
                self.backtick(c, False)
            elif ch == "#" and not c.wopen:
                j = s.find("\n", i)
                self.i = n if j < 0 else j
            elif ch == ";":
                self.end_word(c)
                top = c.cases[-1] if c.cases else None
                if s.startswith(";;", i) or s.startswith(";&", i):
                    if top and top[0] == "body":
                        top[0] = "pat"
                    self.i += 3 if s.startswith(";;&", i) else 2
                else:
                    self.i += 1
                c.taint = 0
                c.start_command()
            elif ch == "&":
                nx = s[i + 1 : i + 2]
                if nx == "&":
                    self.sep(c, not c.dbl)
                    self.i += 2
                elif nx == ">":
                    self.end_word(c)
                    self.i += 3 if s.startswith("&>>", i) else 2
                else:
                    self.sep(c)
                    self.i += 1
            elif ch == "|":
                self.sep(c, not c.dbl)
                self.i += 2 if s[i + 1 : i + 2] in ("|", "&") else 1
            elif ch == "(":
                self.open_paren(c, i)
            elif ch == ")":
                self.end_word(c)
                top = c.cases[-1] if c.cases else None
                if c.dbl:
                    self.touch(c, False)
                    c.wtext += ch
                    self.i += 1
                elif top and top[0] == "pat":
                    top[0] = "body"
                    c.taint = 0
                    c.start_command()
                    self.i += 1
                elif c.parens:
                    kind = c.parens.pop()
                    if kind == "g":
                        c.taint = 0
                    c.start_command()
                    self.i += 1
                elif closer == ")":
                    self.i += 1
                    if c.pending:
                        raise ParseError(c.pending[0][3], "heredoc <<%s is not terminated" % c.pending[0][0])
                    return
                else:
                    c.taint = 0
                    c.start_command()
                    self.i += 1
            elif (ch == "<" or ch == ">") and not c.dbl:
                self.redirect(c, i, ch)
            else:
                self.touch(c, False)
                c.wtext += ch
                self.i += 1
        self.end_word(c)
        if closer is not None:
            raise ParseError(open_line, "unterminated $( … )")
        if c.pending:
            raise ParseError(c.pending[0][3], "heredoc <<%s is not terminated" % c.pending[0][0])

    def carries_shell(self, c):
        """The word about to start is a shell script given whole to `box.sh`, `eval` or `sh -c`."""
        names = [os.path.basename(x) for x in c.words]
        if "box.sh" in names or "eval" in names:
            return True
        return len(names) >= 2 and re.match(r"^-[a-zA-Z]*c$", names[-1]) is not None and any(x in SHELLS for x in names[:-1])

    def open_paren(self, c, i):
        s = self.s
        top = c.cases[-1] if c.cases else None
        if c.dbl:
            self.touch(c, False)
            c.wtext += "("
            self.i += 1
        elif s.startswith("((", i) and not c.wopen and (c.cmd_pos or c.last == "for"):
            e = self.arith_end(i + 2)
            if e >= 0:
                self.arith(c, i + 2, e)
                c.cmd_pos = False
                return
            self.end_word(c)
            c.parens.append("g")
            c.taint = 0
            c.start_command()
            self.i += 1
        elif top and top[0] == "pat":
            self.i += 1
        elif c.wopen and not c.wcomplex and c.wtext.endswith("="):
            self.end_word(c)
            c.parens.append("a")
            self.i += 1
        else:
            self.end_word(c)
            c.parens.append("g")
            c.taint = 0
            c.start_command()
            self.i += 1

    def redirect(self, c, i, ch):
        s = self.s
        nx = s[i + 1 : i + 2]
        if nx == "(":
            self.touch(c)
            open_line = self.line(i)
            self.i = i + 2
            self.parse_list(")", c.taint, open_line)
            return
        self.end_word(c)
        if ch == "<" and s.startswith("<<<", i):
            self.i += 3
        elif ch == "<" and nx == "<":
            self.heredoc_start(c)
        elif nx == "&" or nx == "|" or (ch == ">" and nx == ">") or (ch == "<" and nx == ">"):
            self.i += 2
        else:
            self.i += 1


def scan_text(text, base):
    """The hits `(line, line of the substitution)` of one text, lines counted from `base`."""
    hits = []
    Scanner(text, base, hits).parse_list(None, 0, base)
    return sorted(set(hits))


def recipe_bodies(lines):
    """`(first line number, text)` of each justfile recipe body, dedented as just runs it, `{{ … }}` made opaque."""
    out = []
    k = 0
    n = len(lines)
    while k < n:
        ln = lines[k]
        k += 1
        if not ln or ln[0] in " \t#[" or ln.startswith(("set ", "import ", "mod ", "alias ", "export ")):
            continue
        if re.match(r"^[A-Za-z_][\w-]*\s*:=", ln) or ":" not in ln:
            continue
        start = k
        body = []
        while k < n and (not lines[k].strip() or lines[k][0] in " \t"):
            raw = lines[k].replace("{{{{", "J")
            raw = re.sub(r"\{\{.*?\}\}", "J", raw)
            body.append(re.sub(r"^(\s+)[@-]+", r"\1", raw))
            k += 1
        if any(b.strip() for b in body):
            out.append((start + 1, textwrap.dedent("\n".join(body))))
    return out


def scan_tree(root):
    """`(hits, allowed, errors, shell files, recipe bodies)`; a hit is `(path, line, tline, source line)`."""
    hits, errors = [], []
    allowed = 0
    files = []
    for d, dirs, names in os.walk(os.path.join(root, "tools")):
        dirs.sort()
        for nm in sorted(names):
            if nm.endswith(".sh"):
                files.append(os.path.join(d, nm))
    jf = os.path.join(root, "justfile")
    units = [(f, None) for f in files]
    if os.path.exists(jf):
        units.append((jf, "just"))
    nfiles = nrecipes = 0
    for path, kind in units:
        rel = os.path.relpath(path, root)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as e:
            errors.append("%s: cannot read: %s" % (rel, e))
            continue
        lines = text.split("\n")
        pieces = [(1, text)] if kind is None else recipe_bodies(lines)
        if kind is None:
            nfiles += 1
        else:
            nrecipes += len(pieces)
        found = set()
        for base, piece in pieces:
            try:
                found.update(scan_text(piece, base))
            except ParseError as e:
                errors.append("%s:%d: cannot parse: %s" % (rel, e.line, e.msg))
        seen = set()
        for ln, tline in sorted(found):
            if ln in seen:
                continue
            seen.add(ln)
            src = lines[ln - 1] if ln - 1 < len(lines) else ""
            if ALLOW_RE.search(src):
                allowed += 1
            else:
                hits.append((rel, ln, tline, src.strip()))
    return hits, allowed, errors, nfiles, nrecipes


def run(root):
    hits, allowed, errors, nfiles, nrecipes = scan_tree(root)
    for e in errors:
        print("check-rc: %s" % e, file=sys.stderr)
    if errors:
        return 2
    for rel, ln, tline, src in hits:
        print(
            "check-rc: %s:%d: `$?` is the status of the command substitution that closed at line %d, not of the command before it: %s"
            % (rel, ln, tline, src[:140]),
            file=sys.stderr,
        )
    if hits:
        print(
            "check-rc: %d hits (%d allowed by `# check-rc:`); fix the line, or end it with `# check-rc: <reason>` when the substitution's status is meant"
            % (len(hits), allowed),
            file=sys.stderr,
        )
        return 1
    print("check-rc: %d shell files and %d justfile recipe bodies read, 0 hits (%d allowed by `# check-rc:`)" % (nfiles, nrecipes, allowed))
    return 0


# -- self-test ------------------------------------------------------------------------------------------
# (name, kind, lines, expected hit lines, expected allowed)
def L(*lines):
    return "\n".join(lines) + "\n"


FIXTURES = [
    # red
    ("quoted string with nested quotes", "sh", L('echo "$(basename "$c") rc=$?"'), [1], 0),
    ("argument list", "sh", L("f $(x) $?"), [1], 0),
    ("assignment list", "sh", L("a=$(x) b=$?"), [1], 0),
    ("backtick", "sh", L("echo `x` rc=$?"), [1], 0),
    ("braced ${?}", "sh", L('echo "$(x) ${?}"'), [1], 0),
    ("default word substitution then $?", "sh", L('echo "${u:-$(x)} $?"'), [1], 0),
    ("[[ ]] keeps && inside the command", "sh", L('[[ $(x) == "" && $? -eq 0 ]]'), [1], 0),
    ("a later substitution reads the earlier one's status", "sh", L('echo "$(a)" "$(echo $?)"'), [1], 0),
    ("array literal", "sh", L("a=($(x) $?)"), [1], 0),
    ("continued line", "sh", L('echo "$(x)" \\', '  "rc=$?"'), [2], 0),
    ("string spanning lines", "sh", L('echo "$(x)', 'rc=$?"'), [2], 0),
    ("nested substitution", "sh", L("echo $(echo $(y) $?)"), [1], 0),
    ("after a case inside a substitution", "sh", L("echo $(case x in a) y;; *) z;; esac) $?"), [1], 0),
    ("inside if", "sh", L("if x; then echo \"$(a) $?\"; fi"), [1], 0),
    ("unquoted heredoc body", "sh", L("cat <<EOF", "$(x) $?", "EOF"), [2], 0),
    ("assignment list of two substitutions", "sh", L("a=$(x) y=$(z) w=$?"), [1], 0),
    ("function body", "sh", L("f() {", '  echo "$(basename "$1") rc=$?"', "}"), [2], 0),
    ("bare marker without a reason", "sh", L('echo "$(x) $?" # check-rc:'), [1], 0),
    ("justfile recipe body", "just", L("a := 'x'", "", "go ARG:", '    echo "{{ARG}} $(x) $?"', "", "other:", '    out=$(y) || rc=$?'), [4], 0),
    ("script given to box.sh", "sh", L("./tools/box.sh 'cargo build; echo \"$(x) $?\"'"), [1], 0),
    ("script given to bash -c", "sh", L("timeout 5 bash -c 'echo $(x) $?'"), [1], 0),
    ("multi-line script given to box.sh", "sh", L("DEBUG=1 ./tools/box.sh 'set -e", "a=$(x)", "echo $(y) $?'"), [3], 0),
    ("script given to box.sh in a justfile recipe", "just", L("go:", "    ./tools/box.sh 'x=$(y) z=$?'"), [2], 0),
    # green
    ("$? before the substitution", "sh", L('log "rc=$? $(grep x y)"'), [], 0),
    ("after ||", "sh", L('out=$(cmd) || rc=$?'), [], 0),
    ("after ;", "sh", L('out=$(cmd); rc=$?'), [], 0),
    ("after &&", "sh", L('out=$(cmd) && echo $?'), [], 0),
    ("after a newline", "sh", L("out=$(cmd)", "echo $?"), [], 0),
    ("after |", "sh", L("echo $(x) | cat $?"), [], 0),
    ("after &", "sh", L("echo $(x) & echo $?"), [], 0),
    ("inside the substitution after its first command", "sh", L('echo "$(foo; echo $?)"'), [], 0),
    ("spaced $( ( subshell ; $? )", "sh", L('got=$( (lock_take x) 2>&1; echo "rc=$?")'), [], 0),
    ("comment", "sh", L('# echo "$(x) $?"', 'echo "$(x)" # rc=$?'), [], 0),
    ("quoted heredoc", "sh", L("cat <<'EOF'", "$(x) $?", "EOF"), [], 0),
    ("double-quoted heredoc delimiter", "sh", L('cat <<"EOF"', "$(x) $?", "EOF"), [], 0),
    ("backslash heredoc delimiter", "sh", L("cat <<\\EOF", "$(x) $?", "EOF"), [], 0),
    ("quoted heredoc, tab stripped", "sh", L("cat <<-'EOF'", "\t$(x) $?", "\tEOF"), [], 0),
    ("heredoc then code", "sh", L("cat <<'EOF'", "$(x)", "EOF", "echo $?"), [], 0),
    ("single quotes", "sh", L("echo '$(x) $?'"), [], 0),
    ("escaped", "sh", L('echo "$(x) \\$?"', "echo $(x) \\$?"), [], 0),
    ("arithmetic leaves $? alone", "sh", L("echo $((1 + 2)) $?"), [], 0),
    ("process substitution leaves $? alone", "sh", L("diff <(x) <(y) $?"), [], 0),
    ("here-string", "sh", L('read -r a <<< "$(y)"; echo $?'), [], 0),
    ("length of an array", "sh", L('n=${#a[@]}; echo "$(x)" ${#a[@]} ${x#*|} ${y%%#*}'), [], 0),
    ("case in a substitution, then ;", "sh", L('v=$(case x in a) echo 1;; *) echo 2;; esac); rc=$?'), [], 0),
    ("subshell group", "sh", L("x=$(a); ( echo $? )"), [], 0),
    ("two substitutions, no $?", "sh", L("x=$(a) y=$(b)"), [], 0),
    ("while read from a process substitution", "sh", L("while read -r l; do echo $?; done < <(x)"), [], 0),
    ("[[ with $? before the substitution", "sh", L('[[ $? -eq 0 && -n $(x) ]]'), [], 0),
    ("arithmetic command", "sh", L("(( i = $(x) )); echo $?"), [], 0),
    ("script given to bash -c, $? after ;", "sh", L("bash -c 'echo $(x); echo $?'"), [], 0),
    ("single quotes of a command that is not a shell", "sh", L("grep -c '$(x) $?' f", "ssh h 'echo $(x) $?'"), [], 0),
    ("single-quoted word after a box.sh line, in [[ ]]", "sh", L("./tools/box.sh 'true'", "[[ -n '$(x) $?' ]]"), [], 0),
    ("quote character in [[ ]] after a box.sh line", "sh", L("./tools/box.sh 'true'", "[[ $l == '\"' ]]"), [], 0),
    ("case pattern after a box.sh line", "sh", L("./tools/box.sh 'true'", "case $y in '$(x) $?') ;; esac"), [], 0),
    ("single-quoted word after a box.sh line and ;", "sh", L("./tools/box.sh 'true'; grep -c '$(x) $?' f"), [], 0),
    ("marker with a reason", "sh", L('echo "$(x) $?" # check-rc: the substitution status is the point'), [], 1),
    ("justfile recipe, $? after ||", "just", L("go ARG:", "    out=$(y) || rc=$?"), [], 0),
]
ERROR_FIXTURES = [
    ("unterminated $(", L("echo $(x"), "unterminated"),
    ("unterminated double quote", L('echo "x'), "unterminated"),
    ("unterminated heredoc", L("cat <<EOF", "x"), "heredoc"),
]


def write_tree(root, kind, text):
    os.makedirs(os.path.join(root, "tools"))
    with open(os.path.join(root, "justfile" if kind == "just" else "tools/f.sh"), "w", encoding="utf-8") as fh:
        fh.write(text)


def self_test():
    bad = 0
    red = green = 0
    for name, kind, text, want, want_allowed in FIXTURES:
        with tempfile.TemporaryDirectory() as root:
            write_tree(root, kind, text)
            hits, allowed, errors, _, _ = scan_tree(root)
            code = run_quiet(root)
        path = "justfile" if kind == "just" else "tools/f.sh"
        got = [(h[0], h[1]) for h in hits]
        exp = [(path, w) for w in want]
        if errors or got != exp or allowed != want_allowed or code != (1 if want else 0):
            bad += 1
            print(
                "check-rc: self-test fixture '%s': hits %s allowed %d rc %d, want %s allowed %d rc %d%s"
                % (name, got, allowed, code, exp, want_allowed, 1 if want else 0, " errors %s" % errors if errors else ""),
                file=sys.stderr,
            )
        elif want:
            red += 1
        else:
            green += 1
    for name, text, word in ERROR_FIXTURES:
        with tempfile.TemporaryDirectory() as root:
            write_tree(root, "sh", text)
            code = run_quiet(root)
            _, _, errors, _, _ = scan_tree(root)
        if code != 2 or not errors or word not in errors[0] or not errors[0].startswith("tools/f.sh:"):
            bad += 1
            print("check-rc: self-test fixture '%s': rc %d errors %s, want rc 2 naming tools/f.sh and '%s'" % (name, code, errors, word), file=sys.stderr)
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "tools"))
        with open(os.path.join(root, "tools", "bad.sh"), "wb") as fh:
            fh.write(b"echo \xff\xfe\n")
        code = run_quiet(root)
    if code != 2:
        bad += 1
        print("check-rc: self-test fixture 'a *.sh that cannot be read': rc %d, want 2" % code, file=sys.stderr)
    if bad:
        print("check-rc: self-test %d failed" % bad, file=sys.stderr)
        return 1
    print("check-rc: self-test ok (%d red fixtures red, %d green fixtures green, %d error fixtures rc 2)" % (red, green, len(ERROR_FIXTURES) + 1))
    return 0


def run_quiet(root):
    saved = sys.stdout, sys.stderr
    with open(os.devnull, "w") as dn:
        sys.stdout = sys.stderr = dn
        try:
            return run(root)
        finally:
            sys.stdout, sys.stderr = saved


root_dir, mode = sys.argv[1], sys.argv[2]
sys.exit(self_test() if mode == "--self-test" else run(root_dir))
PY
