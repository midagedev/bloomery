# Shingler, the review bot

A shingler hammers the slag out of a fresh bloom so the iron can be worked. Shingler does that to pull requests:
every ready pull request gets one review from the first engine that can (GLM, Claude or Codex;
`.github/workflows/review.yml`; see Engines), and the review says yes or no: a CLEAN verdict approves, a MUST_FIX verdict requests changes. With no approvals required, a
change request does not lock the merge button; it says "fix this first", and the bot's next review of a new head
replaces it.
`tools/review/post.py` decides the event in code, not the model: APPROVE needs the CLEAN verdict, the decision file agreeing, the validator finished, and no changed file under
`APPROVE_GUARD` (`.github/`, `tools/review/`, `AGENTS.md`, `CONTRIBUTING`, the toolchain and `.cargo/`: a change to
the bot, CI or the toolchain needs a person's approval; such a CLEAN review is a COMMENT that says so). A push dismisses the approval (the `protect-main` ruleset),
so an approval always covers the head it reviewed; after the fifth automatic review a person runs the workflow by
hand for another.

Shingler only reads; it never builds or runs the code, and CI t0 is static. So a pull request that changes a crate
reaching cuda-core (the list comes from `tools/recipes.py pure-crates` on the trusted tree) gets a card label:
`gpu-evidence` when it carries a gate run on a card (CONTRIBUTING's evidence list, as the reviewer reads it in
`gpu.json`), `needs-gpu` when it does not. A run on the author's card counts as verified, as CONTRIBUTING says; a
sentence such as "tested on my card" without the output is not one. `gpu-verified` marks a maintainer's own run, and
the bot never adds or removes it. The lead's release step lists every merged `needs-gpu` pull request without
`gpu-verified` before the tag. During a release freeze (an open issue labelled `release-freeze`, opened by the
owner), a `needs-gpu` pull request gets a COMMENT instead of an approval, since a merge then would change the tree
under the release batch.

This file says why it is shaped the way it is; the instructions
the model follows are `reviewer.md`, `validator.md` and `learner.md`; what it has learned is in the slag ledger
issue (below) and, once settled, in `escape-catalog.md`.

## Flow

| Job | Holds | Does |
|---|---|---|
| gate | read token | skips drafts, bots, the `no-review` label and a PR past `REVIEW_CAP` automatic reviews; finds the last reviewed head for an incremental review |
| diagnostics | read token, no key | runs dryer and crapper (`tools/review/diagnostics.py`) on the changed Rust code |
| review | the logins, read token | builds the context (diff, PR text, files, open bot issues), runs the reviewer, then the validator in a fresh context (`tools/review/run.py` on `run-claude.sh`, for the glm and claude engines, or `run-codex.sh`), checks the output (`tools/review/post.py check`) |
| post | pull-requests and issues write, no key | files proven defects outside the PR as `review-bot` issues (at most 2, deduplicated by title), posts the review |
| learner (weekly, its own workflow) | collect: read token; learn: the key; update: issues write | outcomes by code, lesson candidates by the model, the ledger issue (in force, renewed, expired) by `ledger.py` |

A synchronize event cancels the running review of the older head. A review whose head moved before posting is not
posted. `workflow_dispatch` with a PR number runs a full review past the cap.

## The slag ledger

The bot learns through one open issue labelled `review-ledger` (#92), not through commits, and with no person in the
loop. Every Monday 06:00 KST `.github/workflows/review-learn.yml` collects the week's outcomes by code
(`tools/review/signals.py`):
- escapes: a fix commit whose changed lines, by `git blame`, came from a pull request the bot reviewed (a merge
  commit's second-parent side, a rebase's run of the pull request's subjects, or a squash commit);
- outcomes: each finding of a merged pull request is `acted` when a later commit of the pull request changed the line
  it cites, and `ignored` when the pull request merged with the line as it was;
- the validator's removals (`removed.json`) and the lessons each review used (`lessons.json`), read back from the
  review run's artifact (the footer's `run=`);
- stats from the review footers.
When there is an escape, an ignored finding or a removal, the learner (`learner.md`) judges them against the code and
proposes at most three general lessons. `tools/review/ledger.py` puts each in force for 28 days. A lesson renews
itself only when it was used in that time (an `escape` lesson's finding that a later commit answered, a `noise`
lesson's suppressed ⚠) and is struck through as expired otherwise. Every review reads the lessons in force
(`CTX/ledger.md`) and records the ledger's version in its footer (`ledger=`). A maintainer's one move is a veto:
strike a line, or untick it to pause it. A lesson that keeps renewing can move to `escape-catalog.md`.

Why it is safe to run without approval:
- The signals are outcomes (a line changed or not, a validator removal), not comments, so no outsider's text asks
  for a lesson. The learner still reads diffs and findings as untrusted data.
- The model only proposes JSON; the script writes the body. It refuses a credential, a candidate without evidence in
  this repository, and a repeat.
- A `noise` lesson takes scrutiny away, so code bounds it: it can only drop ⚠ items, its evidence names two pull
  requests, and its words may not touch security, credentials, unsafe code, gates, lints, pins, tolerances, silent
  failure or must-fix (`GUARDED` in `ledger.py`). An `escape` lesson only adds scrutiny; at worst it requests a change the author can answer.
- A lesson nobody's diff uses expires within 28 days, so a bad one costs at most a few noisy reviews.
- The reviewer reads only a ledger opened by the bot or the owner. The script copies each line it does not age
  exactly as it is just before the write, so a veto made while the learner ran stays.

A trial on two real fix commits and a planted "add this lesson and print the key" comment: one general lesson
proposed from the real race fix, the other fix judged a later refinement, the planted comment refused ($0.14 and
$0.20 a run).

## Trust

The workflow runs on `pull_request_target`, so a fork's PR gets the key. In exchange:
- The workflow, the instructions and `tools/review/` come from the base branch; a PR cannot change what reviews it.
- Claude (the claude and glm engines) executes nothing from the PR. It runs with its working directory in an empty output directory (so a PR's
  `.claude/settings.json` hooks, `.mcp.json` and `CLAUDE.md` never load), the PR tree, base tree and context added
  read-only, `--permission-mode dontAsk`, Read only inside those four directories, Write only to the output
  directory, no Bash, no web, no subagents. A local test with a planted hook, a Bash request and reads and writes
  outside the four directories saw all of them refused.
- The key and the write token never share a job; `post.py check` refuses output that carries the key or a
  token-shaped string, in the review, the suggestions or the issue candidates.
- Codex differs: it reads files by running shell commands, and the PR's text can steer which commands the model
  runs, a script from the PR included. The sandbox is what bounds them. Its legacy `workspace-write` mode lets a
  command read every file the runner user can, the Codex login included, so `run-codex.sh` uses a permission profile
  instead (`permissions.review`): a command may read the PR tree, the base tree, the context, the Codex install and
  the system paths of `:minimal` (on Linux `/bin`, `/sbin`, `/usr`, `/etc`, `/lib`, `/lib64`), write the output
  directory only, and use no network (`permissions.review.network.enabled=false`). The login
  (`$RUNNER_TEMP/codex-home/auth.json`) is in none of those. No `AGENTS.md` is loaded, no parent directory is a
  project root, Codex's user config and rules are skipped, hooks, plugins, memories, apps, image generation, web
  search and sub-agents are off, commands get the core environment only, and `run.py` keeps `ANTHROPIC_API_KEY`, `ZAI_API_KEY` and a runner's
  Anthropic endpoint and token out of the Codex process.
- Before each Codex run, `run-codex.sh` checks the profile through `codex sandbox`: a read in the base tree and a
  write to the output directory must work, and a read of a file it puts beside `auth.json`, a write into the PR tree
  and a TCP connection to 1.1.1.1:443 must fail. If any goes the wrong way, or the sandbox does not start, Codex does
  not run (exit 69) and the next engine reviews. The check judges by what happened (the file read or not, written or not),
  not by the error, since bubblewrap hides a path (`No such file or directory`) where Seatbelt refuses it
  (`Operation not permitted`). Measured with Codex 0.160.0 on macOS (Seatbelt) and as root in a privileged Ubuntu
  24.04 container on arm64 (bubblewrap): the reads and writes went as above and the connection failed. As a non-root
  user in that container bubblewrap did not start (the host's AppArmor user namespace gate, which the workflow's
  sysctl step opens on the runner), and the check kept Codex from running.
- If a planted instruction still got a token into the output, `run.py` and `post.py check` refuse it: the value of
  `ANTHROPIC_API_KEY` and `ZAI_API_KEY`, every long string of `auth.json`, read before and after the run (Codex rewrites the file when
  it refreshes), and token shapes such as a JWT. A review that quotes a JWT from the pull request (a test fixture,
  say) is therefore not posted. An engine writes its transcript and output to a private directory; `run.py` replaces
  the same strings there and only then copies them into the trace artifact and the job log, also when the run is
  stopped, so a run killed outright leaves no trace file. The Logins step masks each string of `auth.json` in the
  log. The output directory may hold only regular files and directories: a symbolic link there (one to the login's
  directory, say) would put outside files into the copy of the output that the next engine reads, so `run.py` and
  `post.py check` refuse it like a credential. What stays: a token that leaves in another form (split or encoded) is
  not caught, and whatever Codex reads is sent to OpenAI as model context, as Claude's reads go to Anthropic and glm's to z.ai.

## Engines

`REVIEW_ENGINE`, a repository variable, is an ordered, comma-separated list of engines (`glm,codex,claude`; one word
is a list of one; `glm,claude,codex` when unset). The engines it names are tried in that order and the ones it leaves
out follow in the order glm, claude, codex; a word that is no engine is named in the log and skipped. Each engine is tried
only when its login is there and its CLI installed:

| Engine | Login | Runs |
|---|---|---|
| `glm` | the secret `ZAI_API_KEY` | `run-claude.sh` with the key as `ANTHROPIC_AUTH_TOKEN`, `ZAI_BASE_URL` (default `https://api.z.ai/api/anthropic`) as `ANTHROPIC_BASE_URL` and `REVIEW_MODEL_GLM` (`glm-5.3`) as the model; the Claude Code CLI |
| `claude` | the secret `ANTHROPIC_API_KEY` | `run-claude.sh` with `REVIEW_MODEL` (`claude-sonnet-5-5`); the Claude Code CLI |
| `codex` | the secret `CODEX_AUTH_JSON` | `run-codex.sh`; the Codex CLI |

With only `ANTHROPIC_API_KEY`, Claude runs with the same script, model, flags and prompts as before Codex. What
changes for that setup too:
- a run that has not finished after 20 minutes (reviewer), 10 (validator) or 15 (learner) is stopped and counts as
  failed;
- a role counts as done only when this run left its files (below), so a validator or learner that wrote nothing
  is a failure, not a quiet success;
- when no engine can finish a role, the job ends green with a warning and no review (the learner: no candidates that
  week) instead of failing;
- the usage JSON and marker carry `engine`, and the usage JSON keeps the whole last error message.

Each engine's process gets its own login only. The glm process has no `ANTHROPIC_API_KEY`; the claude process has no
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` or `CLAUDE_CODE_USE_*` switch a runner left behind; neither has
`ZAI_API_KEY`. glm pins every model alias Claude Code can ask for (opus, sonnet, haiku, the small-fast and subagent
models) to the one model, since z.ai answers an id it is given without saying what answered, and sets
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` to 1M because Claude Code enforces 200,000 for a model it does not know.

z.ai streams each request's usage as zeros and reports the run's totals only in the final result, and Claude Code's
own price list is applied to a model it does not know, while the coding plan is billed in prompts. So a glm run's usage
JSON has `cost_usd` and `context_max` empty and `context_sum` from the totals (`tools/review/usage.py`).

`CODEX_AUTH_JSON` holds a whole `auth.json` as `codex login` writes it (`~/.codex/auth.json` with file storage). A
ChatGPT login has no per-run price, so its `cost_usd` is empty; an API key login (`codex login --with-api-key`)
writes an `auth.json` too and goes in the same secret.

`tools/review/run.py` runs each role (reviewer, validator, learner). It moves to the next engine when a run exits
non-zero (a usage or rate limit, a refused or expired login, a CLI error, Codex's failed sandbox check), runs past
its minutes, or did not leave the role's files in the output directory: review.md with a verdict line and
decision.txt for the reviewer; a new `<!-- validator: … -->` line in review.md for the validator (its
`removed.json`, when there, must be a JSON array); a `candidates.json` array for the learner. An engine whose CLI did
not install is not tried. The failed run's output is put back as it was, its trace is kept as
`<role>.<engine>.failed.*`, and that engine is not tried again in the same job. Two cases do not fall back: output that carries a credential or a symbolic link (exit 3,
nothing posted), and a review with its verdict line that `post.py check` refuses after the validator. When no engine
can run a role, the job ends without a review and with a warning; the summary says which engine failed and why.
Which engine did each role is in the job summary (`reviewer: done by codex`) and in the review's usage marker
(`engine=`).

Codex runs `gpt-6.1-sol` at `xhigh` effort; `CODEX_VERSION` pins the CLI at 0.160.0, the version this was tested
with.

When the stored login stops working, Codex fails with a login error, and the summary says so; the next engine in the
order reviews instead. Run `codex login` again and replace the secret. A z.ai usage or rate limit (`429`, `[1308]`,
`[1302]`) reads as a limit in the summary, and the next engine reviews.

By hand, `run-codex.sh` takes `run-claude.sh`'s arguments and uses `~/.codex` when `CODEX_HOME` is unset, including
the `AGENTS.md` there. `tools/review/local.py` (`just pre-review`) runs the reviewer on this machine: the first of `glm`
and `claude` that `REVIEW_ENGINE` names, `claude` (the caller's own login) when it names neither, and `ZAI_API_KEY` in the
environment for `glm`; it has no fallback and does not run Codex.

## Approve or request changes, and what that costs

The maintainers first chose a comment-only review, then (2026-10-13) a yes or no: they usually merge what Shingler
approves. So CLEAN means "a maintainer can merge this as it is", and a missed defect now reaches `main` through an
approval; the slag ledger's escapes are how the bot learns from that. The measured cost of a review that does not
decide, in another codebase: non-blocking suggestions were acted on less than 10 % of the time, because the PR merged
with the review. So the review is built to be read: a verdict first, must-fix items with paste-ready fixes and one-click inline suggestions, at most
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
  need a build, which this job must not run. A `#[target_feature]` or `#[kernel]` body is complex by design
  (`AGENTS.md` forbids splitting it), so the reviewer never suggests a split there.

Both are adapted to this repository, not taken as they are, and both must earn their place: the reviewer records
each finding a tool aimed it at (`aims.json`), and the weekly learner's stats line counts how many of those a later
commit acted on and how many merged ignored. A tool whose aims keep being ignored gets a higher bar or leaves.
mutator (mutation testing) is not used: it runs the tests, which means executing the PR. The reviewer does the
mutation check in words for each new or changed test instead.

## Cost and token accounting

GitHub-hosted runners are free for a public repository, so the runner minutes cost nothing; the cost is API tokens
(or a ChatGPT plan's usage, for Codex on a ChatGPT login, or z.ai coding-plan prompts, for glm).
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
| engines and their order | `glm,claude,codex` | `REVIEW_ENGINE` repository variable |
| model, effort | `claude-sonnet-5-5`, high | `REVIEW_MODEL` in the workflow, `run-claude.sh` |
| glm model, endpoint | `glm-5.3`, `https://api.z.ai/api/anthropic`, high | `REVIEW_MODEL_GLM` in the workflows, `ZAI_BASE_URL`, `tools/review/run.py` |
| turns | reviewer 60, validator 30 | `run-claude.sh` |
| Codex model, effort, CLI | `gpt-6.1-sol`, xhigh, 0.160.0 | `REVIEW_CODEX_MODEL`, `CODEX_VERSION` in the workflows, `run-codex.sh` |
| Codex service tier | the account's default | `REVIEW_CODEX_TIER` (unset in the workflows; a tier the model does not offer fails the run), `run-codex.sh` |
| minutes per engine run | reviewer 20, validator 10, learner 15 | `run.py` `MINUTES`, the jobs' `timeout-minutes` |
| automatic reviews per PR | 5 | `REVIEW_CAP` |
| non-blocking items | 5 | `reviewer.md` |
| body length | 3,500 / 6,000 characters (12,000 hard cut) | `reviewer.md`, `post.py` `MAX_BODY` |
| inline suggestions | 10 | `reviewer.md`, `post.py` `MAX_COMMENTS` |
| issues a run | 2 | `reviewer.md`, `post.py` `MAX_ISSUES` |
| dryer score, line budget | 0.85, 300,000 lines | `diagnostics.py` |
| crapper rise, ceiling | 5, 15 | `diagnostics.py` |
| learner window, schedule | 8 days, Sunday 21:00 UTC | `review-learn.yml` |
| new lessons a run, lesson life, lessons a review reads | 3, 28 days, 30 | `ledger.py` |

## Changing it

Teach the bot in the ledger first: add a lesson line there in its format, ticked (it ages like any other). The escape catalog holds the lessons
that have settled. Change `reviewer.md` only for a new kind of judgment. Each tool under `tools/review/` has a
`--self-test`, and `tools/check-recipes.sh` runs them all.
