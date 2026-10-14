#!/usr/bin/env python3
"""gate-paths-merge.py — git's merge driver for tools/gate-paths.tsv: one glob owns its rows.

    tools/gate-paths-merge.py BASE OURS THEIRS      the driver (`%O %A %B`): the merged file goes to OURS
    tools/gate-paths-merge.py --install             registers the driver in this clone's git config (worktrees share it)
    tools/gate-paths-merge.py --self-test

.gitattributes names it (`tools/gate-paths.tsv merge=gate-paths`); a clone with no `merge.gate-paths.driver` entry
merges the file as text, as before. The driver runs `git merge-file --diff3` first, so every line both sides left
alone, and every hunk only one side touched, merges as git would. The owner of a glob is the tuple of its rows in
file order. A duplicate glob is ordinary table data: its rows stay, and the other globs still merge. A glob with at
most one row on every side merges that row: a gate either side added is in, a gate one side removed and the other
kept is out, and the why and exceptions columns take the side that changed them. A glob with two or more rows on any
side is one value, picked as a whole: the side that changed the tuple, or a conflict when both sides changed it
differently. Deleting the glob is the empty tuple, so a deletion against an edit conflicts the same way, including
when git's text merge is clean. A comment, a why or an exception both sides changed differently, or a glob whose
place among the others changed, stays a conflict: its markers stay in OURS and the exit is 1, git's "not merged".
The message names the glob and its line numbers. The order check lists each glob once, at its first row, so a second
row of the same glob is not a move.

Exit: 0 merged; 1 a conflict left with its markers; 2 git merge-file failed; 64 usage.
"""

import io
import os
import subprocess
import sys
import tempfile

OURS_M, BASE_M, SEP_M, THEIRS_M = "<<<<<<< ", "||||||| ", "=======", ">>>>>>> "
# The registered command. A checkout without this file (a branch from before it) merges the file as text, markers and all,
# rather than failing the merge on a missing script.
DRIVER = ("sh -c '[ -f tools/gate-paths-merge.py ] && exec python3 tools/gate-paths-merge.py \"$1\" \"$2\" \"$3\"; "
          "exec git merge-file -L ours -L base -L theirs \"$2\" \"$1\" \"$3\"' gate-paths %O %A %B")


def is_row(line):
    return bool(line) and not line.startswith("#") and "\t" in line


def rows_of(lines):
    """Rows grouped by glob, in file order, the non-row lines, and each glob once at its first row."""
    rows, order, other = {}, [], []
    for line in lines:
        if is_row(line):
            fields = line.split("\t")
            glob = fields[0]
            if glob not in rows:
                rows[glob] = []
                order.append(glob)
            rows[glob].append(fields)
        else:
            other.append(line)
    return rows, order, other


def line_index(text):
    """The same grouping as rows_of, plus each glob's 1-based line numbers."""
    rows, at, order = {}, {}, []
    for n, line in enumerate(text.splitlines(), 1):
        if not is_row(line):
            continue
        fields = tuple(line.split("\t"))
        glob = fields[0]
        if glob not in rows:
            rows[glob] = []
            at[glob] = []
            order.append(glob)
        rows[glob].append(fields)
        at[glob].append(n)
    return rows, at, order


def pick(base, ours, theirs):
    """Three-way pick of one value: the side that changed it; None when both changed it differently."""
    if ours == theirs or theirs == base:
        return ours
    if ours == base:
        return theirs
    return None


def merge_row(b, o, t):
    """One row, three ways; a row is a field list or None (absent). None back means a conflict."""
    if o is None and t is None:
        return "drop"
    if o is None or t is None:
        kept = o if t is None else t
        if b is None:
            return kept  # added on one side
        return "drop" if kept == b else None  # removed on one side: unchanged on the other drops it
    base_g = set(b[1].split()) if b else set()
    og, tg = o[1].split(), t[1].split()
    gates = [g for g in og if g in tg or g not in base_g]
    gates += [g for g in tg if g not in og and g not in base_g]
    width = max(len(o), len(t), len(b) if b else 0)
    pad = lambda f: f + [""] * (width - len(f))
    bf, of, tf = pad(list(b) if b else [o[0]] + [None] * (width - 1)), pad(list(o)), pad(list(t))
    out = [o[0], " ".join(gates)]
    for k in range(2, width):
        v = pick(bf[k], of[k], tf[k])
        if v is None:
            return None
        out.append(v)
    while len(out) > 2 and out[-1] == "":
        out.pop()
    return out


def decide(b, o, t):
    """One glob's rows, three ways. A tuple of field-tuples, () when the glob drops, None on conflict.

    At most one row on every side merges field by field. Two or more rows on any side is one tuple: the side that
    changed it, including a deletion (the empty tuple) against an unchanged side. Both sides changing it differently
    conflicts, and so does a deletion against an edit.
    """
    b = tuple(tuple(r) for r in b)
    o = tuple(tuple(r) for r in o)
    t = tuple(tuple(r) for r in t)
    if len(b) <= 1 and len(o) <= 1 and len(t) <= 1:
        m = merge_row(list(b[0]) if b else None, list(o[0]) if o else None, list(t[0]) if t else None)
        if m is None:
            return None
        if m == "drop":
            return ()
        return (tuple(m),)
    return pick(b, o, t)


def resolve(ours, base, theirs):
    """A conflicted hunk's lines, or None to leave it a conflict."""
    ro, oo, xo = rows_of(ours)
    rb, bo, _xb = rows_of(base)
    rt, to, xt = rows_of(theirs)
    if xo != xt:
        return None
    merged = {}
    for glob in list(dict.fromkeys(oo + to + bo)):
        m = decide(rb.get(glob, ()), ro.get(glob, ()), rt.get(glob, ()))
        if m is None:
            return None
        merged[glob] = m
    cursors = {glob: 0 for glob in merged}
    out = []
    for line in ours:
        if not is_row(line):
            out.append(line)
            continue
        glob = line.split("\t", 1)[0]
        rows = merged[glob]
        i = cursors[glob]
        if i < len(rows):
            out.append("\t".join(rows[i]))
            cursors[glob] = i + 1
    extra = []
    for glob in list(dict.fromkeys(to + oo)):
        rows = merged[glob]
        extra.extend("\t".join(row) for row in rows[cursors[glob]:])
        cursors[glob] = len(rows)
    if not extra:
        return out
    # theirs' new rows go after the last row of the hunk, in their order
    last = max((i for i, line in enumerate(out) if is_row(line)), default=len(out) - 1)
    return out[: last + 1] + extra + out[last + 1 :]


def merge_text(text):
    """merge-file --diff3 output with its hunks resolved where they can be; (text, conflicts left)."""
    lines, out, left, i = text.split("\n"), [], 0, 0
    while i < len(lines):
        if not lines[i].startswith(OURS_M):
            out.append(lines[i])
            i += 1
            continue
        start, sides, cur, j = i, {"ours": [], "base": [], "theirs": []}, "ours", i + 1
        while j < len(lines) and not lines[j].startswith(THEIRS_M):
            if lines[j].startswith(BASE_M):
                cur = "base"
            elif lines[j] == SEP_M:
                cur = "theirs"
            else:
                sides[cur].append(lines[j])
            j += 1
        r = resolve(sides["ours"], sides["base"], sides["theirs"])
        if r is None:
            out += lines[start : j + 1]
            left += 1
        else:
            out += r
        i = j + 1
    return "\n".join(out), left


def _fmt_lines(nums):
    return "none" if not nums else " ".join(str(n) for n in nums)


def driver(base, ours, theirs):
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs", ours, base, theirs],
                       capture_output=True, text=True)
    if p.returncode < 0 or p.returncode > 127:
        sys.stderr.write("gate-paths-merge: git merge-file failed (rc %d): %s" % (p.returncode, p.stderr))
        return 2
    # Git's text merge is not the row merge. Read every glob either way: a row deleted on one side and edited on
    # the other can miss the edit and git returns 0, and a second row of a glob must not blank the whole file.
    reason = "a comment, a why or an exception both sides changed differently"
    texts, indexed = [], []
    for path in (base, ours, theirs):
        with open(path) as f:
            texts.append(f.read())
        indexed.append(line_index(texts[-1]))
    (rb, lb, bo), (ro, lo, oo), (rt, lt, tho) = indexed
    order_changed = bool(p.returncode) and any(
        [g for g in order if g in bo] != [g for g in bo if g in order]
        for order in (oo, tho)
    )
    globs, bad = [], []
    for glob in oo + bo + tho:
        if glob not in globs:
            globs.append(glob)
    for glob in globs:
        if decide(rb.get(glob, ()), ro.get(glob, ()), rt.get(glob, ())) is None:
            bad.append(glob)
    if bad or order_changed:
        text = p.stdout
        if not p.returncode:
            b, o, t = (s if s.endswith("\n") else s + "\n" for s in texts)
            text = "<<<<<<< ours\n%s||||||| base\n%s=======\n%s>>>>>>> theirs\n" % (o, b, t)
        if bad:
            reason = "; ".join(
                "glob %s (ours line %s, base line %s, theirs line %s)" % (
                    glob, _fmt_lines(lo.get(glob, ())), _fmt_lines(lb.get(glob, ())), _fmt_lines(lt.get(glob, ())))
                for glob in bad)
        else:
            reason = "a row moved"
        left = text.count(OURS_M) or 1
    else:
        text, left = merge_text(p.stdout) if p.returncode else (p.stdout, 0)
    with open(ours, "w") as f:
        f.write(text)
    if left:
        sys.stderr.write("gate-paths-merge: %d hunk(s) of %s left as conflicts (%s)\n" % (
            left, os.environ.get("GIT_MERGE_PATH", "tools/gate-paths.tsv"), reason))
        return 1
    return 0


def self_test():
    head = "# table\n\n"
    base = head + "a/*.rs\tgate-x gate-y\twhy a\nb/*.rs\tgate-x\twhy b\nc/*.rs\tgate-z\twhy c\n"
    cases = [
        # name, ours, theirs, want rc, want text (None: only the rc is held)
        ("both add a gate to one row", base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o\twhy a"),
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-t\twhy a"), 0,
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o gate-t\twhy a")),
        ("one removes a gate, the other adds one", base.replace("gate-x gate-y\twhy a", "gate-y\twhy a"),
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-t\twhy a"), 0,
         base.replace("gate-x gate-y\twhy a", "gate-y gate-t\twhy a")),
        ("adjacent rows", base.replace("gate-x\twhy b", "gate-x gate-o\twhy b"),
         base.replace("gate-z\twhy c", "gate-z gate-t\twhy c"), 0,
         base.replace("gate-x\twhy b", "gate-x gate-o\twhy b").replace("gate-z\twhy c", "gate-z gate-t\twhy c")),
        ("both add a row at one place", base.replace("b/*.rs", "o/*.rs\tgate-o\twhy o\nb/*.rs"),
         base.replace("b/*.rs", "t/*.rs\tgate-t\twhy t\nb/*.rs"), 0,
         base.replace("b/*.rs", "o/*.rs\tgate-o\twhy o\nt/*.rs\tgate-t\twhy t\nb/*.rs")),
        ("one side's why with the other's gate", base.replace("why a", "why a2"),
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-t\twhy a"), 0,
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-t\twhy a2")),
        ("both change one why differently", base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o\twhy o"),
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-t\twhy t"), 1, None),
        ("a comment both change", base.replace("# table", "# table o").replace("gate-x gate-y\t", "gate-x gate-y gate-o\t"),
         base.replace("# table", "# table t"), 1, None),
        ("one removes a row the other changes", base.replace("b/*.rs\tgate-x\twhy b\n", ""),
         base.replace("gate-x\twhy b", "gate-x gate-t\twhy b"), 1, None),
        ("one removes a row the other leaves, beside a change", base.replace("b/*.rs\tgate-x\twhy b\n", "").replace("why a", "why a2"),
         base.replace("gate-z\twhy c", "gate-z gate-t\twhy c"), 0,
         base.replace("b/*.rs\tgate-x\twhy b\n", "").replace("why a", "why a2").replace("gate-z\twhy c", "gate-z gate-t\twhy c")),
        ("an exception column one side adds", base.replace("why c\n", "why c\texcept gate-z: r\n"),
         base.replace("gate-z\twhy c", "gate-z gate-t\twhy c"), 0,
         base.replace("gate-z\twhy c\n", "gate-z gate-t\twhy c\texcept gate-z: r\n")),
        # The other direction of the two cases above. `gates = list(og)` and `pick` returning ours pass the
        # earlier ten and drop a gate theirs removed, or a why theirs changed.
        ("theirs removes a gate while ours adds one", base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o\twhy a"),
         base.replace("gate-x gate-y\twhy a", "gate-y\twhy a"), 0,
         base.replace("gate-x gate-y\twhy a", "gate-y gate-o\twhy a")),
        ("theirs changes why while ours adds a gate", base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o\twhy a"),
         base.replace("why a", "why a2"), 0,
         base.replace("gate-x gate-y\twhy a", "gate-x gate-y gate-o\twhy a2")),
        # Ours deletes a row and edits another; theirs moves the deleted row and edits it. The hunk that holds the
        # edit does not hold the moved row in its base, so a hunk-local merge puts the row back.
        ("deleted row moved and edited on the other side",
         base.replace("a/*.rs\tgate-x gate-y\twhy a\n", "").replace("gate-z\twhy c", "gate-z gate-o\twhy c"),
         base.replace("a/*.rs\tgate-x gate-y\twhy a\n", "").replace("c/*.rs", "a/*.rs\tgate-x gate-y gate-t\twhy a\nc/*.rs"),
         1, None),
        # The same hole across a comment: row-key order is unchanged, and the conflict hunk's base is empty.
        ("deleted row moved across header and edited",
         "o/*.rs\tgate-o\twhy o\n" + base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         "a/*.rs\tgate-x gate-y gate-t\twhy a\n" + base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         1, None),
        # The same row with no text conflict: both sides delete the original line, and theirs inserts the
        # edited row before the header. Git returns 0, so the check has to run on a clean merge too.
        ("deleted row moved across header without a text conflict",
         base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         "a/*.rs\tgate-x gate-y gate-t\twhy a\n" + base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         1, None),
        # The other direction. Dropping either arm of the removed-and-changed check leaves one of the two green.
        ("deleted row moved across header without a text conflict, the other way",
         "a/*.rs\tgate-x gate-y gate-t\twhy a\n" + base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         base.replace("a/*.rs\tgate-x gate-y\twhy a\n", ""),
         1, None),
    ]
    # A glob that already has two rows. The rows sit outside the hunk that merges b, and a one-sided change of
    # the pair is that side's tuple. Both sides changing the pair differently conflicts and names the lines.
    dup = head + "a/*.rs\tgate-x\twhy1\na/*.rs\tgate-y\twhy2\nb/*.rs\tgate-z\twhy b\n"
    ours_a = head + "a/*.rs\tgate-x gate-o\twhy1\na/*.rs\tgate-y\twhy2\nb/*.rs\tgate-z\twhy b\n"
    theirs_a = head + "a/*.rs\tgate-x\twhy1\na/*.rs\tgate-y gate-t\twhy2\nb/*.rs\tgate-z\twhy b\n"
    extra = [
        ("a duplicate glob outside the hunk still merges",
         dup,
         head + "a/*.rs\tgate-x\twhy1\na/*.rs\tgate-y\twhy2\nb/*.rs\tgate-z gate-o\twhy b\n",
         head + "a/*.rs\tgate-x\twhy1\na/*.rs\tgate-y\twhy2\nb/*.rs\tgate-z gate-t\twhy b\n",
         0,
         head + "a/*.rs\tgate-x\twhy1\na/*.rs\tgate-y\twhy2\nb/*.rs\tgate-z gate-o gate-t\twhy b\n",
         None),
        ("a duplicate glob one side changes",
         dup, ours_a, dup, 0, ours_a, None),
        ("a duplicate glob the other side changes",
         dup, dup, ours_a, 0, ours_a, None),
        ("a duplicate glob both sides change differently",
         dup, ours_a, theirs_a, 1, None,
         "glob a/*.rs (ours line 3 4, base line 3 4, theirs line 3 4)"),
        ("a duplicate glob deleted on one side and kept on the other",
         dup, head + "b/*.rs\tgate-z\twhy b\n", dup, 0, head + "b/*.rs\tgate-z\twhy b\n", None),
        ("a duplicate glob deleted on one side and edited on the other",
         dup, head + "b/*.rs\tgate-z\twhy b\n", theirs_a, 1, None,
         "glob a/*.rs (ours line none, base line 3 4, theirs line 3 4)"),
    ]
    fails, n = 0, 0

    def check_resolve(name, ours_lines, base_lines, theirs_lines, want):
        nonlocal fails, n
        n += 1
        got = resolve(ours_lines, base_lines, theirs_lines)
        if got != want:
            fails += 1
            print("FAIL %s: got %s" % (name, got))
        else:
            print("ok   %s" % name)

    check_resolve(
        "an unchanged two-row glob inside a hunk stays, and the other row merges",
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-o\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-t\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-o gate-t\twhy b"])
    check_resolve(
        "one side's two-row glob inside a hunk replaces the tuple",
        ["a/*.rs\tgate-x gate-o\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-o\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-t\twhy b"],
        ["a/*.rs\tgate-x gate-o\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-o gate-t\twhy b"])
    check_resolve(
        "the other side's two-row glob inside a hunk replaces the tuple",
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z gate-o\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y gate-t\twhy2", "b/*.rs\tgate-z gate-t\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y gate-t\twhy2", "b/*.rs\tgate-z gate-o gate-t\twhy b"])
    check_resolve(
        "a two-row glob both sides change inside a hunk stays a conflict",
        ["a/*.rs\tgate-x gate-o\twhy1", "a/*.rs\tgate-y\twhy2"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y gate-t\twhy2"],
        None)
    check_resolve(
        "a two-row glob deleted on one side and kept on the other drops",
        ["b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["b/*.rs\tgate-z\twhy b"])
    check_resolve(
        "a two-row glob deleted on one side and edited on the other conflicts",
        ["b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        ["a/*.rs\tgate-x gate-t\twhy1", "a/*.rs\tgate-y\twhy2", "b/*.rs\tgate-z\twhy b"],
        None)

    with tempfile.TemporaryDirectory() as d:
        def run(name, base_text, ours, theirs, want_rc, want, err_has):
            nonlocal fails, n
            n += 1
            paths = {}
            for k, v in (("base", base_text), ("ours", ours), ("theirs", theirs)):
                paths[k] = os.path.join(d, k)
                with open(paths[k], "w") as f:
                    f.write(v)
            buf = io.StringIO()
            old_err = sys.stderr
            sys.stderr = buf
            try:
                rc = driver(paths["base"], paths["ours"], paths["theirs"])
            finally:
                sys.stderr = old_err
            err = buf.getvalue()
            if err:
                sys.stderr.write(err)
            got = open(paths["ours"]).read()
            ok = rc == want_rc and (want is None or got == want) and (rc == 0 or OURS_M in got)
            if err_has is not None and err_has not in err:
                ok = False
            if not ok:
                fails += 1
                print("FAIL %s: rc %d (want %d), stderr %r, got:\n%s" % (name, rc, want_rc, err, got))
            else:
                print("ok   %s" % name)

        for name, ours, theirs, want_rc, want in cases:
            run(name, base, ours, theirs, want_rc, want, None)
        for item in extra:
            run(*item)
    print("gate-paths-merge: self-test %s (%d cases, %d failed)" % ("ok" if not fails else "FAIL", n, fails))
    return 1 if fails else 0


def install():
    here = os.path.dirname(os.path.abspath(__file__))
    top = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    for k, v in (("merge.gate-paths.name", "tools/gate-paths.tsv rows by glob, gates as sets"),
                 ("merge.gate-paths.driver", DRIVER)):
        subprocess.run(["git", "-C", top, "config", k, v], check=True)
    print("gate-paths-merge: merge.gate-paths.driver set in %s's git config" % top)
    return 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if argv == ["--install"]:
        return install()
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 64
    return driver(*argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
