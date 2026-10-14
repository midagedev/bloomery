#!/usr/bin/env python3
"""gate-paths-merge.py — git's merge driver for tools/gate-paths.tsv: rows merge by their glob, gates as sets.

    tools/gate-paths-merge.py BASE OURS THEIRS      the driver (`%O %A %B`): the merged file goes to OURS
    tools/gate-paths-merge.py --install             registers the driver in this clone's git config (worktrees share it)
    tools/gate-paths-merge.py --self-test

.gitattributes names it (`tools/gate-paths.tsv merge=gate-paths`); a clone with no `merge.gate-paths.driver` entry
merges the file as text, as before. The driver runs `git merge-file --diff3` first, so every line both sides left
alone, and every hunk only one side touched, merges as git would. Inside a conflicted hunk each row (a non-comment
line with a tab) is keyed by its glob and merged three ways against the hunk's base: a gate either side added is in,
a gate one side removed and the other kept is out, and the why and exceptions columns take the side that changed them.
A hunk whose comment and blank lines the two sides changed differently, a row whose why or exceptions both sides
changed to different texts, or a row one side removed and the other changed stays a conflict: its markers stay in
OURS and the exit is 1, git's "not merged". The failure it blocks: two branches that each add a gate to a row, or a
row beside another branch's row, stop a rebase on a file whose right merge is mechanical (three times on one PR in a
day).

Exit: 0 merged; 1 a conflict left with its markers; 2 git merge-file failed; 64 usage.
"""

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
    """The rows of a hunk side by glob, in order, and the side's other lines."""
    rows, order, other = {}, [], []
    for line in lines:
        if is_row(line):
            f = line.split("\t")
            if f[0] in rows:
                return None, None, None
            rows[f[0]] = f
            order.append(f[0])
        else:
            other.append(line)
    return rows, order, other


def pick(base, ours, theirs):
    """Three-way pick of one field: the side that changed it; None when both changed it differently."""
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


def resolve(ours, base, theirs):
    """A conflicted hunk's lines, or None to leave it a conflict."""
    ro, oo, xo = rows_of(ours)
    rb, _, _ = rows_of(base)
    rt, ot, xt = rows_of(theirs)
    if ro is None or rb is None or rt is None or xo != xt:
        return None
    merged = {}
    for key in list(dict.fromkeys(oo + ot + list(rb))):
        m = merge_row(rb.get(key), ro.get(key), rt.get(key))
        if m is None:
            return None
        merged[key] = m
    out, placed = [], set()
    for line in ours:
        if is_row(line):
            key = line.split("\t")[0]
            placed.add(key)
            if merged[key] != "drop":
                out.append("\t".join(merged[key]))
        else:
            out.append(line)
    # theirs' new rows go after the last row of the hunk, in their order
    last = max((i for i, line in enumerate(out) if is_row(line)), default=len(out) - 1)
    new = ["\t".join(merged[k]) for k in ot if k not in placed and merged[k] != "drop"]
    return out[: last + 1] + new + out[last + 1 :]


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


def driver(base, ours, theirs):
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs", ours, base, theirs],
                       capture_output=True, text=True)
    if p.returncode < 0 or p.returncode > 127:
        sys.stderr.write("gate-paths-merge: git merge-file failed (rc %d): %s" % (p.returncode, p.stderr))
        return 2
    text, left = merge_text(p.stdout) if p.returncode else (p.stdout, 0)
    with open(ours, "w") as f:
        f.write(text)
    if left:
        sys.stderr.write("gate-paths-merge: %d hunk(s) of %s left as conflicts (a comment, a why or an exception both sides "
                         "changed differently)\n" % (left, os.environ.get("GIT_MERGE_PATH", "tools/gate-paths.tsv")))
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
    ]
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        for name, ours, theirs, want_rc, want in cases:
            paths = {}
            for k, v in (("base", base), ("ours", ours), ("theirs", theirs)):
                paths[k] = os.path.join(d, k)
                with open(paths[k], "w") as f:
                    f.write(v)
            rc = driver(paths["base"], paths["ours"], paths["theirs"])
            got = open(paths["ours"]).read()
            ok = rc == want_rc and (want is None or got == want) and (rc == 0 or OURS_M in got)
            if not ok:
                fails += 1
                print("FAIL %s: rc %d (want %d), got:\n%s" % (name, rc, want_rc, got))
            else:
                print("ok   %s" % name)
    print("gate-paths-merge: self-test %s (%d cases, %d failed)" % ("ok" if not fails else "FAIL", len(cases), fails))
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
