# Status board

One pinned GitHub issue whose body `board.py` rewrites from facts that already exist: the Claude session registry,
the gate batches' event logs, git and its tags, `gh`. No LLM runs in it, and nobody has to remember to update it. The
files in the state directory are the lead's and the sessions' optional notes and overrides. Contributors read the same
page: where a PR stands, what is landing.

    python3 tools/board/board.py                  # print the body
    python3 tools/board/board.py --publish ISSUE  # replace the issue's body (what launchd runs)
    python3 tools/board/board.py --self-test      # synthetic state, a stub gh, temp git repos, no network

The body opens with ASCII art and one caption line, then the sections, each with its own "as of": Release, Trains,
Batches, Workers, Contributor PRs and issues, Recently landed. A missing state directory is an empty section. A source that fails (gh, git, an unreadable state file or registry) is a `problem:` line in its
section, a line on stderr, and exit 1 after everything else is rendered and published.

## What is derived, and from what

| Section | Derived from | The hand-written file is |
|---|---|---|
| Workers | `~/.claude/sessions/<pid>.json` (`--sessions`), the sessions named in `--roles` | `workers/<role>.md`, a note |
| Trains | `batch/*.jsonl` (the lead's batches) and `git` in `--tree`: a branch on `origin/main` has landed | `trains/<name>.json`, overrides |
| Release | `git ls-remote --tags origin`: a version with its tag is tagged | `release.json`, the plan |

## State files

The directory is `~/.cache/bloomery/board` (`--dir`). `tools/gate-batch.sh` puts it beside its times file
(`dirname $BLOOMERY_GATE_TIMES`/board), so a test that moves the times file moves the board's batch events with it.

**`release.json`**, written by the lead (no file renders "No release planned."):

    {"version": "0.3.0", "cut": "2026-10-11 22:00–24:00 KST", "state": "collecting|union-batch|user-pass|tagged",
     "items": [{"what": "...", "owner": "...", "state": "landed|in-train|pending|deferred", "eta": "HH:MM"|null, "note": "..."}]}

If `v<version>` is a tag on origin (`git ls-remote --tags origin`, so no local ref is read or moved; a leading `v` in `version` is
the tag's own), the release renders as **tagged** with the tag's date (the annotated tag's, else the commit's, from the
objects this checkout has) whatever its `state` says, followed by `next: not opened`, and flags no item late or overdue.
It stays so until `release.json` names a newer version. A lookup that fails is a `problem:` line and the release shows the
file's own state.

Items are shown in four groups in file order: Landed, In a train now, Pending (with ETA), Deferred to the next release.
An item in a train or pending whose `eta` is later in the day than the cut window's start (the first HH:MM of `cut`;
both are times of day, in the Mac's zone) is flagged **late for the cut**; an ETA equal to the start is not. An item in a train or pending whose `eta` is earlier than
the clock now (a time of today) is flagged **overdue**; an ETA equal to now is not. A `cut` with
no HH:MM–HH:MM window, an unknown state or a bad ETA is a `problem:` line and no release is shown.

**`trains/<name>.json`**, written by the lead, optional. Only what the facts cannot say is the lead's to give:

    {"name": "train3", "branch": "train3", "base": "abc1234",
     "pieces": [{"what": "kv pool", "owner": "kvpool", "pr": 41}, {"what": "spark docs", "owner": "sparkdoc", "pr": null}],
     "state": "building|static|batch|ab|landing|landed", "note": "free text"}

Everything is optional but a readable JSON object; `state` is the lead's word for a stage git and the batches cannot see
(`static`, `ab`, `landing`). A `state` that is none of those words is appended to the note, never a `problem:`.

How a train's row is made:

- **A file with no state** takes its newest batch's: `batch running|green|red|stopped|quiet`.
- **A lead batch with no train file** makes a train of its own, named by its `branch`. Only batches with `mode: lead`
  (`--ledger`) count: a round's `--round-ledger` batch ran on a round's tree and stays in Batches (without this cut
  one day of rounds is about 25 rows). A branch that is `HEAD` (a detached tree), `main` or empty makes no row.
- **Landed by ancestry.** After the `git fetch`, a train (file or derived) whose local branch head is an ancestor of
  `origin/main` renders as **landed** whatever its file says, with `on origin/main at <head>` (landing is
  fast-forward, so the branch head is the main commit it reached). The branch must have carried commits of its own:
  `base..head` is not empty (the file's `base`), or, with no usable base, the branch's reflog moved past its creation.
  A branch still at its base (a train before its first commit) has nothing to land. A branch that no longer exists
  keeps its file state or its last batch state.
- **A landed train** drops off a day after the newest of its file's last write, its head commit and its batch. A
  train landed by the file's word alone, with no branch to test, drops a day after the file was last written.

A batch belongs to a train when the batch's `branch` equals the train's `branch`; the newest such batch shows under the train.

**`batch/<out-dir basename>.jsonl`**, appended by `tools/gate-batch.sh` on a real run (a dry run writes none), one JSON
object per line, whole lines under a flock. Best effort: a failed write costs one stderr line, never the batch's exit code
or output.

    {"ev": "start", "ts": ISO, "total": N, "tier": "real|fixture", "lanes": 1|2, "mode": "lead|round|none",
     "predicted_wall": S, "tree": "<checkout dir name>", "branch": "<git branch>"}
    {"ev": "item", "ts": ISO, "name": "<log stem>", "rc": 0|N|"skip"|"deferred"|"yield", "secs": S, "lane": "A|B|C|X|-"}
    {"ev": "done", "ts": ISO, ...the DONE line's key=value words: "total", "red", "wall", "laneA", ... (seconds as numbers)}
    {"ev": "stop", "ts": ISO}                       # INT/TERM: the batch was stopped

The board shows the last `start` of each file with the events after it: done = item events, green = rc 0, red = any other
number (the batch's own count), predicted end = start + predicted_wall. A batch with no `done` and no `stop` that has been
quiet 3 h reads "no DONE, last event HH:MM" (it died). A batch quiet 24 h drops off. Old files are not pruned; `find
~/.cache/bloomery/board/batch -mtime +14 -delete` does it.

**`workers/<role>.md`**, optional, written by a session that has something to say beyond what the registry shows
(the round, its ETA, what it waits on). The board needs no note: a session may write one, and none is overdue.

**The session registry** is the Workers section's source: `~/.claude/sessions/<pid>.json` (`--sessions`), one file per
live Claude session, with `name`, `status` (`busy`, `idle`, `shell`, ...), `updatedAt` and `statusUpdatedAt` (epoch ms), `cwd`,
`pid`. Only `<pid>.json` is read; the `.key` files beside them hold peer tokens and are never opened. A session is a
worker when its name, less a trailing ` (2)`, is one of `--roles` (default `leader,vision,unitier,common,quant,toktape`).
Of several sessions of a role, the live one with the newest update wins. Each worker shows:

- the state: `busy` (the registry's word, whatever it is, but `idle`), `idle since HH:MM` (from `statusUpdatedAt`), or
  `gone` (the pid is not alive, `kill -0`). A non-idle session whose `updatedAt` is older than 30 min is flagged
  **quiet 30+ min**. The registry's timestamps move with the status, not with the session's output, so this reads
  "one turn or wait has run 30+ min", which is a prompt to look at the session, not proof it is stuck;
- the age of `updatedAt`;
- the train pieces it owns (`owner` in `trains/*.json`; an owner field with several names is split on `, / & +`; a landed
  train's pieces are not listed). `lead` is not `leader`: the names must match;
- the newest batch run from its own tree, when the session's `cwd` (its last component) names that batch's `tree` and
  no other live session shares that cwd. The registry holds no branch. The role sessions all sit in the main checkout, so
  today this maps nothing for them; it is not guessed any other way. A worker with nothing derivable shows state and age only.
- its note, when `workers/<role>.md` exists: newer than 30 min, its first 6 lines nested under `note (updated HH:MM)` (a
  leading `- ` is dropped); older, `(note from HH:MM, stale)` and its first line. A note with no session is headed `no session`.

A session's cwd, pid and id are never rendered, and a session file the board cannot read is only counted
(`sessions: N unreadable file(s)`). No registry directory is no sessions.

## The header art

Seven plain-ASCII pieces (`PIECES`, each at most 10 lines and 56 columns, no backtick), one chosen by `mood()` in this
order: a red in the newest batch (last event within 3 h) is the **rain cloud over a sprout**; else any batch running is
the **furnace with sparks**; else the newest batch ended with no red within 3 h, or a train landed within the last hour,
is the **flower in bloom**; else one of four resting pieces (`IDLE`: anvil, bellows, iron bloom, sleeping furnace)
rotates by date. The caption under it names the state: the train (or the batch's tree) and its green count.

## What the board does with GitHub

- **Open PRs**: every open PR, with its stage: `in train <name> (<state>)` when a train piece names its number; else
  `waiting (draft)` or `waiting (changes requested)`; else `not yet picked`.
- **Issues**: open issues with a comment in the last 7 days from someone who is not a maintainer (`--maintainer`, default
  `midagedev`) and not a bot. The board's own issue is left out (`--publish ISSUE` implies `--board-issue ISSUE`; a preview
  passes `--board-issue` to print exactly what would be sent).
- PR and issue titles and commit subjects are cut to 100 characters and shown as code spans, so an outsider's title cannot
  carry a mention, a link or markup.
- **Recently landed**: `git fetch -q origin main`, then `git log origin/main -10` in the checkout holding `board.py`. The
  fetch is best effort: a failure is a `problem:` line and the list is as of the last fetch that worked.

## Privacy filter

Every rendered line is checked before it is printed or published (same path for both). A line matching a pattern in
`PRIVACY` is dropped and counted; the body ends with `N lines withheld`. The patterns: the home-LAN `192.168.`, tailnet
`100.x` addresses and `.ts.net`, `admin`, the host names `sihyung`, `dopc` and `ws` (whole word), the paths `/Users/`,
`/root/`, `/home/…` and `~/`, and tokens. stderr says how many and by which pattern
names; `--explain` also prints each line. The self-test has one case per pattern, each red with the filter off.
The token patterns are narrow on purpose: `token:` / `token =` and `Bearer <x>` are withheld (with `hf_…` and `ghp_`), while
prose such as "token limits" and "tokenize" in a title is not.

## When it runs

launchd runs it when a state file changes (`WatchPaths` in the plist: `KICK`, `release.json`, `trains/`, `workers/`,
`batch/`), at most once per 15 s, and every 600 s as the fallback for changes on GitHub's side and for the facts nothing
watches: the session registry (its files change with every status, so watching it would publish every few seconds), the
tags and `origin/main`. A worker's state and a landing therefore show within 10 minutes without anyone writing a file. A writer that appends to
a file in place (a batch's event log) also touches `~/.cache/bloomery/board/KICK`, since appending does not change the
directory launchd watches.

## Install

The agent runs `board.py --publish` on those events, as the gh account that owns the repo (`gh auth token --user`, read from the
keyring at run time; launchd runs no shell hook, so it cannot rely on the active account).

    ISSUE=<the pinned issue's number>; REPO=$HOME/repo/bloomery
    mkdir -p ~/.cache/bloomery/board/{trains,batch,workers}
    sed -e "s|@PYTHON@|$(command -v python3)|" -e "s|@REPO@|$REPO|g" -e "s|@ISSUE@|$ISSUE|" \
        -e "s|@GHUSER@|midagedev|" -e "s|@HOME@|$HOME|g" \
        $REPO/tools/board/com.bloomery.board.plist > ~/Library/LaunchAgents/com.bloomery.board.plist
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.bloomery.board.plist
    launchctl kickstart gui/$(id -u)/com.bloomery.board     # run now; the log is ~/.cache/bloomery/board/launchd.log

## Uninstall

    launchctl bootout gui/$(id -u)/com.bloomery.board
    rm ~/Library/LaunchAgents/com.bloomery.board.plist
