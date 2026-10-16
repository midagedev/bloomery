# Status board

One pinned GitHub issue whose body `board.py` rewrites every 10 minutes from state files, `gh` and `git log`. No LLM runs
in it. Contributors read the same page: where a PR stands, what is landing.

    python3 tools/board/board.py                  # print the body
    python3 tools/board/board.py --publish ISSUE  # replace the issue's body (what launchd runs)
    python3 tools/board/board.py --self-test      # synthetic state, a stub gh, no network

The body opens with ASCII art and one caption line, then the sections, each with its own "as of": Release, Trains,
Batches, Workers, Contributor PRs and issues, Recently landed. A missing state directory is an empty section. A source that fails (gh, git, an unreadable state file) is a `problem:` line in its
section, a line on stderr, and exit 1 after everything else is rendered and published.

## State files

The directory is `~/.cache/bloomery/board` (`--dir`). `tools/gate-batch.sh` puts it beside its times file
(`dirname $BLOOMERY_GATE_TIMES`/board), so a test that moves the times file moves the board's batch events with it.

**`release.json`**, written by the lead (no file renders "No release planned."):

    {"version": "0.3.0", "cut": "2026-10-11 22:00–24:00 KST", "state": "collecting|union-batch|user-pass|tagged",
     "items": [{"what": "...", "owner": "...", "state": "landed|in-train|pending|deferred", "eta": "HH:MM"|null, "note": "..."}]}

Items are shown in four groups in file order: Landed, In a train now, Pending (with ETA), Deferred to the next release.
An item in a train or pending whose `eta` is later in the day than the cut window's start (the first HH:MM of `cut`;
both are times of day, in the Mac's zone) is flagged **late for the cut**; an ETA equal to the start is not. An item in a train or pending whose `eta` is earlier than
the clock now (a time of today) is flagged **overdue**; an ETA equal to now is not. A `cut` with
no HH:MM–HH:MM window, an unknown state or a bad ETA is a `problem:` line and no release is shown.

**`trains/<name>.json`**, written by the lead:

    {"name": "train3", "branch": "train3", "base": "abc1234",
     "pieces": [{"what": "kv pool", "owner": "kvpool", "pr": 41}, {"what": "spark docs", "owner": "sparkdoc", "pr": null}],
     "state": "building|static|batch|ab|landing|landed", "note": "free text"}

A `landed` train drops off 24 h after the file was last written (its mtime). A batch belongs to a train when the batch's
`branch` equals the train's `branch`; the newest such batch shows under the train.

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

**`workers/<role>.md`**, written by the sessions. The rule each session follows:

> On a round's start, end, or a landing request, rewrite `~/.cache/bloomery/board/workers/<role>.md` with at most 6 lines:
> the round, its ETA, what it waits on.

The board shows the first 6 lines under the file's mtime; a file quiet for 3 h reads "stale since HH:MM".

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
`PRIVACY` is dropped and counted; the body ends with `N lines withheld`. stderr says how many and by which pattern
names; `--explain` also prints each line. The self-test has one case per pattern, each red with the filter off.
The token patterns are narrow on purpose: `token:` / `token =` and `Bearer <x>` are withheld (with `hf_…` and `ghp_`), while
prose such as "token limits" and "tokenize" in a title is not.

## Install

The agent runs `board.py --publish` every 600 s, as the gh account that owns the repo (`gh auth token --user`, read from the
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
