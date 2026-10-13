# The review bot

Every ready pull request gets one advisory review from Claude (`.github/workflows/review.yml`). It posts a COMMENT
review: it never approves and never blocks a merge. This file says why it is shaped the way it is; the instructions
the model follows are `reviewer.md` and `validator.md`, and the patterns it learns from are `escape-catalog.md`.

## Flow

| Job | Holds | Does |
|---|---|---|
| gate | read token | skips drafts, bots, the `no-review` label and a PR past `REVIEW_CAP` automatic reviews; finds the last reviewed head for an incremental review |
| diagnostics | read token, no key | runs dryer and crapper (`tools/review/diagnostics.py`) on the changed Rust code |
| review | the API key, read token | builds the context (diff, PR text, files, open bot issues), runs the reviewer, then the validator in a fresh context (`tools/review/run-claude.sh`), checks the output (`tools/review/post.py check`) |
| post | pull-requests and issues write, no key | files proven defects outside the PR as `review-bot` issues (at most 2, deduplicated by title), posts the review |

A synchronize event cancels the running review of the older head. A review whose head moved before posting is not
posted. `workflow_dispatch` with a PR number runs a full review past the cap.

## Trust

The workflow runs on `pull_request_target`, so a fork's PR gets the key. In exchange:
- The workflow, the instructions and `tools/review/` come from the base branch; a PR cannot change what reviews it.
- Nothing from the PR is executed. Claude runs with its working directory in an empty output directory (so a PR's
  `.claude/settings.json` hooks, `.mcp.json` and `CLAUDE.md` never load), the PR tree, base tree and context added
  read-only, `--permission-mode dontAsk`, Read only inside those four directories, Write only to the output
  directory, no Bash, no web, no subagents. A local test with a planted hook, a Bash request and reads and writes
  outside the four directories saw all of them refused.
- The key and the write token never share a job; `post.py check` refuses output that carries the key or a
  token-shaped string, in the review, the suggestions or the issue candidates.

## Why advisory, and what that costs

The maintainers chose advisory (2026-10-13). The measured cost of that choice in another codebase: non-blocking
review suggestions were acted on less than 10 % of the time, because the PR merged with the review. So the review is
built to be read: a verdict first, must-fix items with paste-ready fixes and one-click inline suggestions, at most
five non-blocking items, and a length budget (3,500 characters clean, 6,000 with must-fix items).

## Where the method comes from

Most of the review method is carried over from a production review gate that was tuned over several months with
audits of its own output:
- the two kinds of must-fix (a constructed failure path; a proven quality defect with a mechanical fix), because a
  rule that only blocks on failure paths turned every quality finding into an ignored suggestion;
- the adversarial procedure (find, refute, one lens, doubt a clean verdict), because a checklist only sees the code
  that is there;
- a separate validator in a fresh context, because the writer of a finding is lenient with it;
- the anti-speculation rule and a "what not to flag" list, the volume cap, and the escape catalog (the catalog grows
  instead of the prompt).
What is bloomery's own: the focus list (silent failure, gates as the contract, common code first, one owner per fact,
numbers with conditions, comments that state what is true now), the adversarial performance procedure tied to the
change-class table in `AGENTS.md`, the three layers of a fix, and Uncle Bob's dryer and crapper as aiming signals.

## Diagnostics

`tools/review/diagnostics.py` runs two of Uncle Bob's tools, pinned by commit, on source only:
- dryer: structural duplicate pairs, kept when one side overlaps lines the diff adds. `crates/gpu-gates` is never
  searched (its bins share a boilerplate `main` by design, and it alone costs minutes); the searched crates stay under
  300,000 lines because dryer compares every pair. Measured on the Mac: four crates for PR #89, 96 s.
- crapper, complexity only: functions whose cyclomatic complexity rose by 5 or more, or grew above 15. Coverage would
  need a build, which this job must not run.
mutator (mutation testing) is not used: it runs the tests, which means executing the PR. The reviewer does the
mutation check in words for each new or changed test instead.

## Cost and token accounting

GitHub-hosted runners are free for a public repository, so the runner minutes cost nothing; the cost is API tokens.
Every run keeps what a later optimization needs:
- the job summary has one line per role (turns, requests, cost, the largest context, the summed context, output
  tokens, tool calls by tool, denials, distinct files read);
- the posted review ends with `<!-- bloomery-review … -->` (head, verdict, counts, mode) and
  `<!-- bloomery-usage … -->` (the same accounting), so the reviews themselves are the dataset;
- the `review-trace` artifact (30 days) has both stream-json transcripts and `tools/review/usage.py`'s JSON with the
  most-read files.
The measured driver of cost in the earlier gate was the context re-sent each turn (cache reads were 95 % of tokens),
not the prose written. The knobs below act on that.

## Knobs

| Knob | Now | Where |
|---|---|---|
| model, effort | `claude-sonnet-5-5`, high | `REVIEW_MODEL` in the workflow, `run-claude.sh` |
| turns | reviewer 60, validator 30 | `run-claude.sh` |
| automatic reviews per PR | 5 | `REVIEW_CAP` |
| non-blocking items | 5 | `reviewer.md` |
| body length | 3,500 / 6,000 characters (12,000 hard cut) | `reviewer.md`, `post.py` `MAX_BODY` |
| inline suggestions | 10 | `reviewer.md`, `post.py` `MAX_COMMENTS` |
| issues a run | 2 | `reviewer.md`, `post.py` `MAX_ISSUES` |
| dryer score, line budget | 0.85, 300,000 lines | `diagnostics.py` |
| crapper rise, ceiling | 5, 15 | `diagnostics.py` |

## Changing it

The escape catalog is the first place to teach the bot: a pattern that passed review here and turned out to be a
defect goes there, with the incident. Change `reviewer.md` only for a new kind of judgment. Run
`python3 tools/review/post.py --self-test`, `diagnostics.py --self-test` and `usage.py --self-test` after changing
the tools (`tools/check-recipes.sh` runs them).
