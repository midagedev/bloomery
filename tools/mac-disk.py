#!/usr/bin/env python3
"""mac-disk.py — this Mac's bloomery worktrees: size, landing state, named-only cleanup.

  tools/mac-disk.py [report]        one line per worktree and a footer
  tools/mac-disk.py clean NAME…     remove <worktree>/target for each NAME
  tools/mac-disk.py retire NAME…    remove the worktree for each NAME (LANDED or
                                    PRUNABLE only; the branch is never deleted)
  --dry-run                         print what clean/retire would do, remove nothing
  --self-test                       a throwaway repo in a temp dir; no real
                                    worktree is touched

The main repo is the worktree whose .git is a directory, found from the current
tree with `git rev-parse --git-common-dir`. A NAME is a worktree basename
(`bloomery-q38s14`) or its absolute path. The report's tree column excludes the
top-level `target/` and `.git` (a main worktree's object store is repo
plumbing, not tree; a worktree's `.git` is a pointer file).

States, in precedence order:
  MAIN      the main worktree
  LIVE      a process has its cwd in it (one `lsof -d cwd -Fn` call for the
            whole machine; when the scan fails, report still prints and marks
            every non-MAIN row LIVE?, while clean and retire refuse)
  PRUNABLE  git lists the worktree prunable, or its directory is gone — a
            prunable worktree cannot be status-checked, so it comes before
            DIRTY, and retiring it runs `git worktree prune`
  DIRTY     `git status --porcelain` is not empty (untracked files count)
  LANDED    HEAD is an ancestor of origin/main, or `git cherry origin/main HEAD`
            prints no `+` line
  AHEAD     otherwise, with the count of unlanded commits

Removal re-reads the worktree list, the cwd scan and the state immediately
before each NAME — the state can change between the listing and the removal,
the rule tools/box-tracks.sh applies to the names it removes. Branches are
never deleted: a branch is a few bytes and may be another session's. There is
no bulk mode: a bare clean or retire is refused, and the report footer prints
(never runs) the command that retires every LANDED worktree by name.

Exit: 0 ok; 1 some NAME refused (the others are still done); 64 usage; 69 a
required tool (git, df, lsof for clean/retire) failed or timed out, or
origin/main is missing (no silent fallback to another ref).

Test seams, used by --self-test only: MAC_DISK_LSOF_PATHS replaces the lsof
call with these cwd paths (newline-separated, may be empty); MAC_DISK_LSOF_FAIL
makes the cwd scan fail before the call; MAC_DISK_LSOF_BIN names the binary
the lsof call runs (point it at one that fails).
"""

import os
import shlex
import shutil
import subprocess
import sys
import time

USAGE = ("usage: tools/mac-disk.py [report] | clean NAME… | retire NAME… "
         "[--dry-run] | --self-test")

EXIT_REFUSED = 1
EXIT_USAGE = 64
EXIT_TOOL = 69

# Every subprocess runs under a bound: a stalled mount can wedge df and the
# whole machine's process table with it, and a tool that waits forever is a
# silent failure of its own.
GIT_BOUND = 60
LSOF_BOUND = 300
DF_BOUND = 30

ZERO_SHA = set("0")


class Usage(Exception):
    """A bad command line; the message is printed with the usage line."""


class Tool(Exception):
    """A required tool failed or timed out, or the repo lacks what is needed."""


def say(msg):
    print(f"mac-disk: {msg}", file=sys.stderr)


def run(cmd, bound):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=bound)
    except FileNotFoundError:
        raise Tool(f"{cmd[0]} is not installed (needed: {shlex.join(cmd)})") from None
    except subprocess.TimeoutExpired:
        raise Tool(f"{shlex.join(cmd)} did not answer within {bound}s") from None


def git_ok(main, *args):
    """git output with rc 0; any other rc is a Tool failure naming the call."""
    proc = run(["git", "-C", main, *args], GIT_BOUND)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"rc {proc.returncode}"
        raise Tool(f"git {' '.join(args)} failed in {main}: {detail}")
    return proc.stdout


def find_main_root():
    """The main worktree of this tree's repo: the one whose .git is a directory."""
    proc = run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], GIT_BOUND)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"rc {proc.returncode}"
        raise Tool(f"git rev-parse --git-common-dir failed here ({detail}): "
                   "run the tool inside a worktree of the repo it should report")
    root = os.path.dirname(proc.stdout.strip())
    if not os.path.isdir(os.path.join(root, ".git")):
        raise Tool(f"{root} is not the main worktree of its repo (.git is not a directory)")
    return root


def list_worktrees(main):
    """`git worktree list --porcelain`, one dict per worktree, in git's order."""
    worktrees, cur = [], None
    for line in git_ok(main, "worktree", "list", "--porcelain").splitlines():
        if not line.strip():
            if cur is not None:
                worktrees.append(cur)
                cur = None
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            cur = {"path": value, "head": None, "branch": None, "detached": False,
                   "prunable": False, "locked": False, "bare": False}
        elif cur is None:
            raise Tool(f"worktree list printed a line outside a stanza: {line!r}")
        elif key == "HEAD":
            cur["head"] = value
        elif key == "branch":
            cur["branch"] = value
        elif key == "detached":
            cur["detached"] = True
        elif key == "prunable":
            cur["prunable"] = True
        elif key == "locked":
            cur["locked"] = True
        elif key == "bare":
            cur["bare"] = True
    if cur is not None:
        worktrees.append(cur)
    return [wt for wt in worktrees if not wt["bare"]]


def cwd_paths():
    """Every process's cwd, one `lsof -d cwd -Fn` call for the whole machine.

    The seams are the self-test's: MAC_DISK_LSOF_PATHS stands in for the scan
    (newline-separated paths, empty allowed), MAC_DISK_LSOF_FAIL fails it
    before the call, MAC_DISK_LSOF_BIN names the binary the call runs.
    """
    if os.environ.get("MAC_DISK_LSOF_FAIL"):
        raise Tool("the cwd scan failed (MAC_DISK_LSOF_FAIL is set)")
    if "MAC_DISK_LSOF_PATHS" in os.environ:
        raw = os.environ["MAC_DISK_LSOF_PATHS"]
        return {os.path.realpath(p) for p in raw.splitlines() if p.startswith("/")}
    proc = run([os.environ.get("MAC_DISK_LSOF_BIN", "lsof"), "-d", "cwd", "-Fn"], LSOF_BOUND)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"rc {proc.returncode}"
        raise Tool(f"lsof -d cwd -Fn failed ({detail})")
    return {os.path.realpath(line[1:]) for line in proc.stdout.splitlines()
            if line.startswith("n") and line[1:].startswith("/")}


def is_live(worktree_real, paths):
    return any(p == worktree_real or p.startswith(worktree_real + os.sep) for p in paths)


def origin_ref(main):
    proc = run(["git", "-C", main, "rev-parse", "--verify", "origin/main"], GIT_BOUND)
    if proc.returncode != 0:
        raise Tool(f"origin/main is missing in {main} (no silent fallback to another ref)")
    return "origin/main"


def landing(main, wt, origin):
    """(landed, unlanded_count) for the worktree's HEAD against origin/main."""
    sha = wt["head"]
    if not sha or set(sha) <= ZERO_SHA:
        return (False, 0)  # an unborn HEAD: nothing has landed
    proc = run(["git", "-C", main, "merge-base", "--is-ancestor", sha, origin], GIT_BOUND)
    if proc.returncode == 0:
        return (True, 0)
    if proc.returncode != 1:
        detail = proc.stderr.strip() or f"rc {proc.returncode}"
        raise Tool(f"git merge-base --is-ancestor {sha[:12]} {origin} failed: {detail}")
    out = git_ok(main, "cherry", origin, sha)
    plus = [line for line in out.splitlines() if line.startswith("+")]
    return (not plus, len(plus))


def is_dirty(wt):
    proc = run(["git", "-C", wt["path"], "status", "--porcelain"], GIT_BOUND)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"rc {proc.returncode}"
        raise Tool(f"git status failed in {wt['path']} ({detail}) — prune or repair it "
                   "by hand; this tool will not guess a state for it")
    return bool(proc.stdout.strip())


def classify(main, main_real, wt, origin, cwd_set):
    """The worktree's state and a one-line detail for refusals.

    cwd_set is None when the cwd scan failed (report marks the row LIVE?);
    clean and retire refuse instead of classifying.
    """
    real = os.path.realpath(wt["path"])
    if real == main_real:
        return "MAIN", "this is the repo's own checkout"
    if cwd_set is None:
        return "LIVE?", "the cwd scan failed, so liveness is unknown"
    if is_live(real, cwd_set):
        return "LIVE", "a process has its cwd inside it"
    if wt["prunable"] or not os.path.isdir(wt["path"]):
        return "PRUNABLE", "git lists it prunable" if wt["prunable"] else "its directory is gone"
    if is_dirty(wt):
        return "DIRTY", "git status is not empty (untracked files count)"
    landed, count = landing(main, wt, origin)
    if landed:
        return "LANDED", "HEAD's patches are all in origin/main"
    return "AHEAD", f"{count} unlanded commit{'s' if count != 1 else ''}"


def paths_bytes(seeds):
    """Real bytes under each seed: st_size of every regular file, symlinked
    directories below the seeds not followed (a symlinked seed itself is
    resolved and walked). A seed that is a plain file counts as its own size.
    Returns (bytes, unreadable paths) — unreadable paths are named, never
    skipped in silence."""
    total, skipped, stack = 0, [], []
    for path in seeds:
        try:
            st = os.lstat(path)
        except OSError:
            skipped.append(path)
            continue
        if os.path.islink(path):
            target = os.path.realpath(path)
            if not os.path.isdir(target):
                total += st.st_size  # a dangling link holds only its own name
            else:
                stack.append(target)
        elif os.path.isdir(path):
            stack.append(path)
        else:
            total += st.st_size
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            skipped.append(directory)
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                else:
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                skipped.append(entry.path)
    return total, skipped


def target_bytes(wt):
    path = os.path.join(wt["path"], "target")
    if not os.path.lexists(path):
        return None, []
    return paths_bytes([path])


def tree_bytes(wt):
    """Bytes of the worktree without its top-level target/ and .git — the
    tree a retire removes, not the repo plumbing a main worktree's .git
    object store is. None when the worktree directory is gone. An unreadable
    root is named in the skipped list, like any unreadable subdirectory."""
    root = wt["path"]
    if not os.path.isdir(root):
        return None, []
    try:
        entries = list(os.scandir(root))
    except OSError as err:
        return 0, [f"{root}: {err}"]
    return paths_bytes([e.path for e in entries if e.name not in ("target", ".git")])


def human(n):
    """One token, no space: every report column stays split()-able."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def volume_free_gib(path):
    proc = run(["df", "-k", path], DF_BOUND)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or len(lines) < 2:
        detail = proc.stderr.strip() or f"rc {proc.returncode}"
        raise Tool(f"df -k {path} failed ({detail})")
    fields = lines[-1].split()
    try:
        avail_kib = int(fields[3])
    except (IndexError, ValueError):
        raise Tool(f"df -k {path} printed a row this tool cannot read: {lines[-1]!r}") from None
    return avail_kib / (1024 * 1024), fields[-1]


def commit_age_days(main, wt):
    sha = wt["head"]
    if not sha or set(sha) <= ZERO_SHA:
        return None
    out = git_ok(main, "show", "-s", "--format=%ct", sha)
    try:
        ts = int(out.strip())
    except ValueError:
        return None
    if ts <= 0:
        return None
    return int((time.time() - ts) // 86400)


def short_branch(wt):
    return wt["branch"][len("refs/heads/"):] if wt["branch"] else "detached"


def resolve_name(name, worktrees):
    """The worktrees a NAME points at: basename match, else absolute-path match."""
    hits = []
    if "/" not in name:
        hits = [wt for wt in worktrees if os.path.basename(wt["path"]) == name]
    if not hits:
        wanted = os.path.realpath(os.path.normpath(name))
        hits = [wt for wt in worktrees if os.path.realpath(wt["path"]) == wanted]
    return hits


def target_verdict(wt):
    """Why <worktree>/target may not be removed, or None when it may."""
    path = os.path.join(wt["path"], "target")
    if os.path.islink(path):
        return f"target is a symlink (to {os.readlink(path)})"
    if os.path.lexists(path):
        real = os.path.realpath(path)
        wt_real = os.path.realpath(wt["path"])
        if real != wt_real and not real.startswith(wt_real + os.sep):
            return f"target ({real}) resolves outside the worktree"
        gate = os.path.join(path, "gate-batch")
        if os.path.isdir(gate):
            try:
                with os.scandir(gate) as it:
                    empty = next(iter(it), None) is None
            except OSError:
                empty = False
            if not empty:
                size, _ = paths_bytes([gate])
                return (f"target/gate-batch/ exists and is not empty "
                        f"({human(size)} at {gate}) — move or delete it by hand first")
    return None


def remove_target(wt, name, dry, freed):
    """clean's and retire's target step; returns the updated freed total."""
    path = os.path.join(wt["path"], "target")
    if not os.path.lexists(path):
        print(f"{name}: no target — nothing to remove")
        return freed
    size, _ = paths_bytes([path])
    if dry:
        print(f"{name}: would remove target ({human(size)})")
        return freed + size
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:  # a degenerate target: a plain file (a symlink is refused before this)
            os.remove(path)
    except OSError as err:
        raise Tool(f"removing {path} failed ({err}) — that worktree's target is left as it is") from None
    print(f"{name}: target removed, {human(size)} freed")
    return freed + size


def cmd_report():
    main = find_main_root()
    worktrees = list_worktrees(main)
    origin = origin_ref(main)
    main_real = os.path.realpath(main)
    try:
        cwds = cwd_paths()
    except Tool as err:
        cwds = None
        say(f"{err}; every non-MAIN row is marked LIVE? and nothing is removed")
    order = ["MAIN", "LIVE?", "LIVE", "PRUNABLE", "DIRTY", "LANDED", "AHEAD"]
    counts, target_by_state, skipped = {}, {}, []
    states = []
    print(f"{'state':<8} {'target':>10} {'tree':>10} {'age':>5}  {'branch':<24} worktree")
    for wt in worktrees:
        state, _detail = classify(main, main_real, wt, origin, cwds)
        states.append((wt, state))
        tbytes, skip1 = target_bytes(wt)
        bbytes, skip2 = tree_bytes(wt)
        skipped += skip1 + skip2
        counts[state] = counts.get(state, 0) + 1
        target_by_state[state] = target_by_state.get(state, 0) + (tbytes or 0)
        age = commit_age_days(main, wt)
        age_col = f"{age}d" if age is not None else "?"
        print(f"{state:<8} "
              f"{human(tbytes) if tbytes is not None else '-':>10} "
              f"{human(bbytes) if bbytes is not None else '-':>10} "
              f"{age_col:>5}  {short_branch(wt):<24} {os.path.basename(wt['path'])}")
    free_gib, mount = volume_free_gib(main)
    print()
    print("states: " + "  ".join(f"{s} {counts[s]}" for s in order if s in counts))
    print("target bytes: " + "  ".join(f"{s} {human(target_by_state[s])}"
                                       for s in order if s in counts))
    print(f"volume free: {free_gib:.1f} GiB ({mount})")
    landed = [(wt, state) for wt, state in states if state == "LANDED"]
    if landed:
        from collections import Counter
        freq = Counter(os.path.basename(wt["path"]) for wt, _ in landed)
        names = [shlex.quote(os.path.basename(wt["path"]))
                 if freq[os.path.basename(wt["path"])] == 1
                 else shlex.quote(wt["path"]) for wt, _ in landed]
        print("retire every LANDED worktree by name (printed, never run):")
        print("  python3 tools/mac-disk.py retire " + " ".join(names))
    else:
        print("no LANDED worktree to retire")
    if skipped:
        say(f"{len(skipped)} paths could not be read and are not counted "
            f"(first: {skipped[0]})")
    return 0


def cmd_clean(names, dry):
    if any(not name.strip() for name in names):
        raise Usage("a NAME is empty — give a worktree basename or an absolute path")
    main = find_main_root()
    main_real = os.path.realpath(main)
    cwd_paths()  # a failed scan refuses the whole command before any output
    refused, freed = 0, 0
    for name in names:
        # Re-read the list and the cwd scan immediately before each NAME: the
        # state can change between the listing and the removal.
        worktrees = list_worktrees(main)
        cwds = cwd_paths()
        hits = resolve_name(name, worktrees)
        if not hits:
            print(f"refused {name}: no worktree matches it")
            refused += 1
            continue
        if len(hits) > 1:
            print(f"refused {name}: matches {len(hits)} worktrees: "
                  + ", ".join(wt["path"] for wt in hits))
            refused += 1
            continue
        wt = hits[0]
        wt_real = os.path.realpath(wt["path"])
        if wt_real == main_real:
            print(f"refused {name}: the main worktree ({wt['path']}) is never cleaned by name")
            refused += 1
            continue
        if is_live(wt_real, cwds):
            print(f"refused {name}: LIVE: a process has its cwd inside it")
            refused += 1
            continue
        reason = target_verdict(wt)
        if reason is not None:
            print(f"refused {name}: {reason}")
            refused += 1
            continue
        freed = remove_target(wt, name, dry, freed)
    free_gib, mount = volume_free_gib(main)
    print(f"total {'would free' if dry else 'freed'}: {human(freed)}")
    print(f"volume free: {free_gib:.1f} GiB ({mount})")
    return EXIT_REFUSED if refused else 0


def cmd_retire(names, dry):
    if any(not name.strip() for name in names):
        raise Usage("a NAME is empty — give a worktree basename or an absolute path")
    main = find_main_root()
    main_real = os.path.realpath(main)
    origin = origin_ref(main)
    cwd_paths()  # a failed scan refuses the whole command before any output
    refused, freed = 0, 0
    for name in names:
        # The same re-read rule as clean: list, cwd scan and state are taken
        # immediately before each NAME.
        worktrees = list_worktrees(main)
        cwds = cwd_paths()
        hits = resolve_name(name, worktrees)
        if not hits:
            print(f"refused {name}: no worktree matches it")
            refused += 1
            continue
        if len(hits) > 1:
            print(f"refused {name}: matches {len(hits)} worktrees: "
                  + ", ".join(wt["path"] for wt in hits))
            refused += 1
            continue
        wt = hits[0]
        if os.path.realpath(wt["path"]) == main_real:
            print(f"refused {name}: the main worktree ({wt['path']}) is never retired")
            refused += 1
            continue
        if is_live(os.path.realpath(wt["path"]), cwds):
            print(f"refused {name}: LIVE: a process has its cwd inside it")
            refused += 1
            continue
        state, detail = classify(main, main_real, wt, origin, cwds)
        if state not in ("LANDED", "PRUNABLE"):
            print(f"refused {name}: state is {state} ({detail}); only LANDED or PRUNABLE retire")
            refused += 1
            continue
        reason = target_verdict(wt)
        if reason is not None:
            print(f"refused {name}: {reason}")
            refused += 1
            continue
        freed = remove_target(wt, name, dry, freed)
        branch = short_branch(wt)
        base = os.path.basename(wt["path"])
        if dry:
            if state == "PRUNABLE":
                print(f"{name}: would prune the worktree entry (its directory is already gone)")
            else:
                print(f"{name}: would remove the worktree (branch {branch} kept)")
            print(f"{name}: box line to run by hand: just box-tracks --remove {base}")
            continue
        if os.path.isdir(wt["path"]):
            proc = run(["git", "-C", main, "worktree", "remove", wt["path"]], GIT_BOUND)
            if proc.returncode != 0:
                detail = proc.stderr.strip() or f"rc {proc.returncode}"
                print(f"refused {name}: git worktree remove failed: {detail}")
                refused += 1
                continue
        git_ok(main, "worktree", "prune")
        if resolve_name(wt["path"], list_worktrees(main)):
            print(f"refused {name}: still listed after removal ({wt['path']}) — "
                  "locked? run `git worktree prune` by hand and look at it")
            refused += 1
            continue
        print(f"{name}: worktree removed (branch {branch} kept)")
        print(f"{name}: box line to run by hand: just box-tracks --remove {base}")
    free_gib, mount = volume_free_gib(main)
    print(f"total target bytes {'would free' if dry else 'freed'}: {human(freed)}")
    print(f"volume free: {free_gib:.1f} GiB ({mount})")
    return EXIT_REFUSED if refused else 0


def cli(argv):
    if "--self-test" in argv:
        return run_self_test()
    if "--help" in argv:
        print(USAGE)
        return 0
    rest = list(argv)
    dry = False
    if "--dry-run" in rest:
        dry = True
        rest.remove("--dry-run")
    if not rest:
        if dry:
            raise Usage("--dry-run belongs to clean or retire")
        return cmd_report()
    sub, rest = rest[0], rest[1:]
    if sub not in ("report", "clean", "retire"):
        raise Usage(f"unknown argument {sub}")
    if sub == "report":
        if rest:
            raise Usage("report takes no NAME")
        if dry:
            raise Usage("--dry-run belongs to clean or retire")
        return cmd_report()
    if not rest:
        raise Usage(f"{sub} takes the NAMEs to remove, one per worktree; "
                    f"a bare {sub} is refused (no bulk mode)")
    return cmd_clean(rest, dry) if sub == "clean" else cmd_retire(rest, dry)


def main(argv):
    try:
        return cli(argv)
    except Usage as err:
        say(str(err))
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    except Tool as err:
        say(str(err))
        return EXIT_TOOL


# ---------------------------------------------------------------------------
# --self-test: a throwaway repo with an origin, worktrees in every state, and
# the cwd scan through the seams. No real worktree is touched.
# ---------------------------------------------------------------------------

def run_self_test():
    import contextlib
    import io
    import tempfile

    fails = 0

    def check(name, got, want):
        nonlocal fails
        if got != want:
            print(f"FAIL {name}: got [{got!r}] want [{want!r}]")
            fails += 1
        else:
            print(f"ok {name}")

    def cli_env(argv, env=None):
        """cli() with captured output and the seam env set only for the call."""
        saved = {}
        try:
            for key, value in (env or {}).items():
                saved[key] = os.environ.get(key)
                os.environ[key] = value
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(argv)
            return rc, out.getvalue(), err.getvalue()
        finally:
            for key, value in saved.items():
                if value is None:
                    del os.environ[key]
                else:
                    os.environ[key] = value

    def G(cwd, *args, check_rc=True, env=None):
        run_env = {**os.environ, **env} if env else None
        proc = subprocess.run(
            ["git", "-c", "user.name=self-test", "-c", "user.email=self-test@example.com",
             "-C", cwd, *args], capture_output=True, text=True, env=run_env)
        if check_rc and proc.returncode != 0:
            raise Tool(f"fixture git {' '.join(args)} in {cwd}: {proc.stderr.strip()}")
        return proc

    def write(path, data):
        with open(path, "wb") as fh:
            fh.write(data)

    keep_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="mac-disk-selftest-")
    try:
        base = os.path.join(tmp, "b")
        origin = os.path.join(base, "origin.git")
        repo = os.path.join(base, "bloomery")
        os.makedirs(base)
        G(tmp, "init", "--bare", "-b", "main", origin)
        G(tmp, "init", "-b", "main", repo)
        G(repo, "remote", "add", "origin", origin)
        write(os.path.join(repo, "README"), b"base\n")
        # the real repo ignores target/; without this every target fixture
        # below (a directory, a symlink, a plain file — and the symin fixture's
        # inner/ dir, which its target symlink points at) would read DIRTY
        # (untracked) instead of LANDED
        write(os.path.join(repo, ".gitignore"), b"target\ninner\n")
        G(repo, "add", ".")
        G(repo, "commit", "-m", "base")
        G(repo, "push", "-q", "-u", "origin", "main")
        base_sha = G(repo, "rev-parse", "HEAD").stdout.strip()

        def worktree(rel, branch, commit_file=None, commit_env=None):
            path = os.path.join(base, *rel.split("/"))
            G(repo, "worktree", "add", "-b", branch, path)
            if commit_file:
                write(os.path.join(path, commit_file), b"x\n")
                G(path, "add", ".")
                G(path, "commit", "-m", commit_file, env=commit_env)
            return path

        wt_landed = worktree("bloomery-landed", "landed-br", "landed.txt")
        G(repo, "merge", "--ff-only", "landed-br")
        G(repo, "push", "-q", "origin", "main")
        # The cherry fixture's branch commit is dated in the past: a pick made
        # in the same second as the branch commit (same tree, same parent, same
        # message, same dates) reproduces the same sha, and the worktree then
        # lands by ancestry — the git cherry branch never runs.
        past = {"GIT_AUTHOR_DATE": "2005-04-07T22:13:13",
                "GIT_COMMITTER_DATE": "2005-04-07T22:13:13"}
        wt_cherry = worktree("bloomery-cherry", "cherry-br", "cherry.txt", commit_env=past)
        cherry_sha = G(wt_cherry, "rev-parse", "HEAD").stdout.strip()
        G(repo, "cherry-pick", cherry_sha)
        G(repo, "push", "-q", "origin", "main")
        wt_ahead = worktree("bloomery-ahead", "ahead-br", "ahead.txt")
        wt_dirty = worktree("bloomery-dirty", "dirty-br")
        write(os.path.join(wt_dirty, "untracked.txt"), b"x\n")
        wt_live = worktree("bloomery-live", "live-br")
        wt_gone = worktree("bloomery-gone", "gone-br", "gone.txt")  # unlanded, dir deleted
        wt_gone2 = worktree("bloomery-gone2", "gone2-br")           # landed HEAD, dir deleted
        wt_target = worktree("bloomery-target", "target-br")
        os.makedirs(os.path.join(wt_target, "target"))
        target_file = os.path.join(wt_target, "target", "f.bin")
        write(target_file, b"\0" * (1024 * 1024))
        wt_gate = worktree("bloomery-gatebatch", "gatebatch-br")
        os.makedirs(os.path.join(wt_gate, "target", "gate-batch"))
        gate_file = os.path.join(wt_gate, "target", "gate-batch", "x.bin")
        write(gate_file, b"\0" * 4096)
        wt_sym = worktree("bloomery-symlink", "symlink-br")
        elsewhere = os.path.join(base, "elsewhere")
        os.makedirs(elsewhere)
        keep_file = os.path.join(elsewhere, "keep.bin")
        write(keep_file, b"\0" * 2048)
        os.symlink(os.path.join("..", "elsewhere"), os.path.join(wt_sym, "target"))
        wt_dup1 = worktree("dups/a/bloomery-dup", "dup-a-br")
        wt_dup2 = worktree("dups/b/bloomery-dup", "dup-b-br")
        wt_locked = worktree("bloomery-locked", "locked-br")
        G(repo, "worktree", "lock", wt_locked)
        # a detached HEAD at a landed commit (LANDED by ancestry, branch column
        # "detached")
        wt_detached = os.path.join(base, "bloomery-detached")
        G(repo, "worktree", "add", "--detach", wt_detached, base_sha)
        # a basename with a space: the row stays readable and the footer's
        # retire command quotes it
        wt_space = worktree("bloomery-sp ace", "space-br")
        # a target symlinked to a directory inside the same worktree — refused
        # by the symlink check alone, not by the resolves-outside check
        wt_symin = worktree("bloomery-symin", "symin-br")
        os.makedirs(os.path.join(wt_symin, "inner"))
        inner_file = os.path.join(wt_symin, "inner", "f.bin")
        write(inner_file, b"\0" * 4096)
        os.symlink("inner", os.path.join(wt_symin, "target"))
        # a degenerate target: a plain file where target/ would be
        wt_filetarget = worktree("bloomery-filetarget", "filetarget-br")
        file_target = os.path.join(wt_filetarget, "target")
        write(file_target, b"\0" * 512)
        shutil.rmtree(wt_gone)
        shutil.rmtree(wt_gone2)
        os.chdir(repo)  # find_main_root() resolves from the current tree

        seam = {"MAC_DISK_LSOF_PATHS": os.path.join(wt_live, "sub")}
        # the cherry fixture must not land by ancestry, or the git cherry
        # branch below proves nothing
        check("cherry-fixture-not-ancestor",
              G(repo, "merge-base", "--is-ancestor", cherry_sha, "origin/main",
                check_rc=False).returncode, 1)

        # report: every state, bytes, ages, the footer counts and command
        rc, out, err = cli_env([], seam)
        check("report-rc", rc, 0)
        rows = {}
        for line in out.splitlines():
            fields = line.split()
            if len(fields) == 6 and fields[0] in ("MAIN", "LIVE?", "LIVE", "PRUNABLE",
                                                  "DIRTY", "LANDED", "AHEAD"):
                rows[fields[5]] = fields
        check("state-main", rows["bloomery"][0], "MAIN")
        check("state-live", rows["bloomery-live"][0], "LIVE")
        check("state-dirty", rows["bloomery-dirty"][0], "DIRTY")
        check("state-landed-by-ancestry", rows["bloomery-landed"][0], "LANDED")
        check("state-landed-by-cherry", rows["bloomery-cherry"][0], "LANDED")
        check("state-ahead", rows["bloomery-ahead"][0], "AHEAD")
        check("state-prunable-unlanded", rows["bloomery-gone"][0], "PRUNABLE")
        check("state-prunable-landed-head", rows["bloomery-gone2"][0], "PRUNABLE")
        check("state-target-row", rows["bloomery-target"][0], "LANDED")
        check("target-bytes", rows["bloomery-target"][1], "1.0MiB")
        check("symlink-target-bytes", rows["bloomery-symlink"][1], "2.0KiB")
        check("prunable-target-dash", rows["bloomery-gone"][1], "-")
        check("prunable-tree-dash", rows["bloomery-gone"][2], "-")
        check("state-detached", (rows["bloomery-detached"][0], rows["bloomery-detached"][4]),
              ("LANDED", "detached"))
        check("space-name-row",
              any(l.startswith("LANDED") and l.endswith("bloomery-sp ace")
                  for l in out.splitlines()), True)
        # tree bytes count plain files at the top level (README 5 + .gitignore
        # 13 + landed.txt 2 + cherry.txt 2, both landed in main before the
        # target worktree was cut) and exclude .git and target
        check("tree-bytes-count-files", (rows["bloomery"][2], rows["bloomery-target"][2]),
              ("22B", "22B"))
        check("main-age-days", rows["bloomery"][3], "0d")
        check("symin-target-bytes", rows["bloomery-symin"][1], "4.0KiB")
        check("filetarget-target-bytes", rows["bloomery-filetarget"][1], "512B")
        check("target-bytes-per-state",
              next(l for l in out.splitlines() if l.startswith("target bytes:")),
              "target bytes: MAIN 0B  LIVE 0B  PRUNABLE 0B  DIRTY 0B  "
              "LANDED 1.0MiB  AHEAD 0B")
        check("states-line",
              next(l for l in out.splitlines() if l.startswith("states:")),
              "states: MAIN 1  LIVE 1  PRUNABLE 2  DIRTY 1  LANDED 12  AHEAD 1")
        retire_line = next(l for l in out.splitlines() if l.startswith("  python3"))
        check("footer-retire-command",
              ("bloomery-landed" in retire_line and "bloomery-cherry" in retire_line
               and retire_line.count(shlex.quote(wt_dup1)) == 1
               and retire_line.count(shlex.quote(wt_dup2)) == 1
               and shlex.quote("bloomery-sp ace") in retire_line
               and "bloomery-gone" not in retire_line and "bloomery-ahead" not in retire_line),
              True)

        # report with a failed cwd scan: still prints, LIVE? everywhere but MAIN
        rc, out, err = cli_env([], {"MAC_DISK_LSOF_FAIL": "1"})
        check("report-lsof-fail-rc", rc, 0)
        check("report-lsof-fail-rows",
              sum(1 for l in out.splitlines() if l.startswith("LIVE?")), 17)
        check("report-lsof-fail-says-why", "MAC_DISK_LSOF_FAIL" in err, True)

        # usage: bare clean/retire, unknown subcommand, --dry-run on report
        check("bare-clean-refused", cli_env(["clean"])[0], 64)
        check("bare-retire-refused", cli_env(["retire"])[0], 64)
        check("unknown-subcommand", cli_env(["bulk"])[0], 64)
        check("report-takes-no-name", cli_env(["report", "x"])[0], 64)

        # clean: refusals by name
        rc, out, err = cli_env(["clean", "nosuchwt"], seam)
        check("clean-no-match", (rc, "no worktree matches" in out), (1, True))
        rc, out, err = cli_env(["clean", "bloomery-dup"], seam)
        check("clean-ambiguous",
              (rc, "matches 2 worktrees" in out, wt_dup1 in out), (1, True, True))
        rc, out, err = cli_env(["clean", "bloomery"], seam)
        check("clean-main", (rc, "the main worktree" in out), (1, True))
        rc, out, err = cli_env(["clean", "bloomery-live"], seam)
        check("clean-live", (rc, "LIVE: a process has its cwd inside it" in out), (1, True))
        rc, out, err = cli_env(["clean", "bloomery-symlink"], seam)
        check("clean-symlink",
              (rc, "symlink" in out, os.path.exists(keep_file)), (1, True, True))
        rc, out, err = cli_env(["clean", "bloomery-gatebatch"], seam)
        check("clean-gatebatch",
              (rc, "gate-batch" in out and "by hand" in out, os.path.exists(gate_file)),
              (1, True, True))
        rc, out, err = cli_env(["clean", "bloomery-symin"], seam)
        check("clean-symin",
              (rc, "target is a symlink" in out, os.path.exists(inner_file)), (1, True, True))
        rc, out, err = cli_env(["clean", "--dry-run", "bloomery-filetarget"], seam)
        check("clean-filetarget-dry",
              (rc, "would remove target (512B)" in out, os.path.exists(file_target)),
              (0, True, True))
        rc, out, err = cli_env(["clean", "bloomery-filetarget"], seam)
        check("clean-filetarget",
              (rc, os.path.exists(file_target), "512B freed" in out), (0, False, True))
        rc, out, err = cli_env(["clean", "bloomery-sp ace"], seam)
        check("clean-space-name", (rc, "no target — nothing to remove" in out), (0, True))
        # The resolves-outside check is a backstop: every way a target can
        # resolve outside the worktree on this filesystem is a symlink, which
        # the islink check already refuses, so it is probed with islink patched
        # out — the one input that reaches it.
        real_islink = os.path.islink
        os.path.islink = lambda _p: False
        try:
            rc, out, err = cli_env(["clean", "bloomery-symlink"], seam)
        finally:
            os.path.islink = real_islink
        check("clean-outside-backstop",
              (rc, "resolves outside" in out, os.path.exists(keep_file)), (1, True, True))

        # clean: dry-run removes nothing, the real run frees the walked bytes
        rc, out, err = cli_env(["clean", "--dry-run", "bloomery-target"], seam)
        check("clean-dry-run",
              (rc, "would remove target" in out, os.path.exists(target_file)), (0, True, True))
        rc, out, err = cli_env(["clean", "bloomery-target"], seam)
        check("clean-real",
              (rc, os.path.exists(target_file), "1.0MiB freed" in out), (0, False, True))
        # a refused NAME does not stop the others
        os.makedirs(os.path.join(wt_cherry, "target"))
        write(os.path.join(wt_cherry, "target", "c.bin"), b"\0" * 512)
        rc, out, err = cli_env(["clean", "nosuchwt", "bloomery-cherry"], seam)
        check("clean-partial",
              (rc, "refused nosuchwt" in out,
               not os.path.exists(os.path.join(wt_cherry, "target"))), (1, True, True))

        # retire: refusals by state and by target, all under --dry-run
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-ahead"], seam)
        check("retire-ahead",
              (rc, "AHEAD" in out, "1 unlanded commit" in out), (1, True, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-dup"], seam)
        check("retire-ambiguous", (rc, "matches 2 worktrees" in out), (1, True))
        check("empty-name-usage", cli_env(["clean", ""])[0], 64)
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-dirty"], seam)
        check("retire-dirty", (rc, "DIRTY" in out), (1, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-live"], seam)
        # the direct LIVE refusal, distinct from the state gate's message (which
        # also contains LIVE), so deleting the direct check goes red
        check("retire-live", (rc, "LIVE: a process has its cwd inside it" in out), (1, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery"], seam)
        check("retire-main", (rc, "the main worktree" in out), (1, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-gatebatch"], seam)
        check("retire-gatebatch",
              (rc, "gate-batch" in out, os.path.exists(gate_file)), (1, True, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-symlink"], seam)
        check("retire-symlink", (rc, "symlink" in out, os.path.exists(keep_file)), (1, True, True))
        rc, out, err = cli_env(["retire", "--dry-run", "bloomery-landed", "bloomery-gone"],
                               seam)
        check("retire-dry-run",
              (rc, "would remove the worktree (branch landed-br kept)" in out,
               "would prune the worktree entry" in out,
               os.path.isdir(wt_landed),
               "just box-tracks --remove bloomery-landed" in out),
              (0, True, True, True, True))

        # retire: the real runs. The list is re-read per NAME: a name given
        # twice is refused the second time, after the first removal took it
        # out of the listing. The prune after a removal retires every other
        # PRUNABLE entry in the repo too (bloomery-gone below) — that is how
        # `git worktree prune` works, and why the branches-kept check covers
        # both.
        rc, out, err = cli_env(["retire", "bloomery-gone2", "bloomery-gone2"], seam)
        # git resolves symlinked ancestors when it registers a worktree, so
        # the listing is compared through realpath (the tool does the same)
        listed = {os.path.realpath(l[len("worktree "):])
                  for l in G(repo, "worktree", "list", "--porcelain").stdout.splitlines()
                  if l.startswith("worktree ")}
        check("retire-prunable-reread",
              (rc, out.count("refused bloomery-gone2"),
               "just box-tracks --remove bloomery-gone2" in out,
               os.path.realpath(wt_gone2) not in listed,
               os.path.realpath(wt_gone) not in listed),
              (1, 1, True, True, True))
        check("retire-keeps-branches",
              (G(repo, "rev-parse", "--verify", "refs/heads/gone2-br",
                 check_rc=False).returncode,
               G(repo, "rev-parse", "--verify", "refs/heads/gone-br",
                 check_rc=False).returncode),
              (0, 0))
        rc, out, err = cli_env(["retire", "bloomery-landed"], seam)
        check("retire-landed", (rc, os.path.isdir(wt_landed)), (0, False))
        check("retire-landed-box-line",
              "just box-tracks --remove bloomery-landed" in out, True)
        check("retire-keeps-branch",
              G(repo, "rev-parse", "--verify", "refs/heads/landed-br",
                check_rc=False).returncode, 0)
        # a locked LANDED worktree: git worktree remove refuses it, and the
        # refusal names git's own message — no --force is added silently
        rc, out, err = cli_env(["retire", "bloomery-locked"], seam)
        still = {os.path.realpath(l[len("worktree "):])
                 for l in G(repo, "worktree", "list", "--porcelain").stdout.splitlines()
                 if l.startswith("worktree ")}
        check("retire-locked",
              (rc, "locked" in out, os.path.isdir(wt_locked),
               os.path.realpath(wt_locked) in still),
              (1, True, True, True))

        # a failed cwd scan refuses clean and retire entirely
        check("clean-lsof-fail", cli_env(["clean", "bloomery-target"],
                                         {"MAC_DISK_LSOF_FAIL": "1"})[0], 69)
        check("retire-lsof-fail", cli_env(["retire", "bloomery-cherry"],
                                          {"MAC_DISK_LSOF_FAIL": "1"})[0], 69)
        # the same refusal through the real lsof call: a binary that fails
        rc, out, err = cli_env(["clean", "bloomery-ahead"],
                               {"MAC_DISK_LSOF_BIN": "/usr/bin/false"})
        check("clean-lsof-bin-fail", (rc, "lsof -d cwd -Fn failed" in err), (69, True))
        rc, out, err = cli_env([], {"MAC_DISK_LSOF_BIN": "/usr/bin/false"})
        # 14 rows, not 17: the retirees above removed gone, gone2 and landed
        check("report-lsof-bin-fail",
              (rc, sum(1 for l in out.splitlines() if l.startswith("LIVE?")),
               "lsof -d cwd -Fn failed" in err), (0, 14, True))

        # a repo without origin/main is refused by name, with no fallback
        norepo = os.path.join(tmp, "norepo")
        G(tmp, "init", "-q", "-b", "main", norepo)
        write(os.path.join(norepo, "f"), b"x\n")
        G(norepo, "add", ".")
        G(norepo, "commit", "-m", "x")
        G(norepo, "worktree", "add", "-b", "w2", os.path.join(tmp, "norepo-wt"))
        os.chdir(norepo)
        rc, out, err = cli_env([], {})
        check("origin-missing", (rc, "origin/main is missing" in err), (69, True))
        os.chdir(repo)
    finally:
        os.chdir(keep_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    if fails == 0:
        print("mac-disk: self-test ok")
        return 0
    print(f"mac-disk: self-test {fails} failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
