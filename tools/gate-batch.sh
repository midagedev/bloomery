#!/usr/bin/env bash
# The lead's gate batch runner, two lanes. Runs on the Mac; every item is `just <recipe> [ARGS]`, so
# the box is reached only through the recipes (tools/box.sh) and each item keeps the bound and the
# exit code its recipe's runner owns (tools/gate.sh, tools/gpu-gate.sh, 900 s). This script adds no
# second bound.
#   tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--tier real|fixture] [--ledger [--trust-rounds] [--no-gpu-hold] | --round-ledger] [--rerun]
#   tools/gate-batch.sh --classes     every recipe's class, the classifier below, and nothing else
#   tools/gate-batch.sh --self-test   the placement rules on a fixture justfile and the disk floor
#                                      against a fake df (check-recipes runs it)
#
# Items. `NAME[@K=V[,K=V…]][:ARGS]` — NAME a recipe in `just --dump`; `@K=V,…` added to
# BLOOMERY_BOX_ENV for that item (values without spaces, commas or colons); `:ARGS` passed to the
# recipe, word-split (shell quoting allowed). Env comes before ARGS, e.g.
#   gate-gpu-ds41-prefill@BLOOMERY_PREFILL_GROUP=1:--cases 512 --no-split --no-extra
# --list FILE: one item per line (blank lines and `#` lines skipped), or the raw output of `just
# affected …` (first line `affected:`, or just's echoed `./tools/affected-gates.sh …` line and then it): then
# only lines starting with two spaces and `gate-` or `weekly-` (a weekly recipe a trigger named) count, the
# first word is the recipe, everything else is ignored — the `always:` checks are not taken from it.
# --smoke: the fixed smoke list (SMOKE below; docs/gates-plan.md 3.1). --weekly: every `weekly-*` recipe in
# `just --dump`, in its order (`just weekly`; a justfile with none is a named error). An unknown recipe, a malformed
# item, ARGS given to a recipe with no parameters, or an empty list is a named error before anything
# runs.
#
# Lanes (--lanes 2, the default), read from each recipe's text and attributes in `just --dump` (and
# its dependencies'), and from the scripts that text runs, followed transitively (walk() below):
#   A  fixed, the 3090: a tools/gpu-gate.sh call, in the recipe or a script it runs, without
#      BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} (V4.1 `--place gate` gates; gate-gpu-hybrid, whose host tier spins
#      the worker pool on the cores a V4.1 gate's pool is pinned to), or device code run with no gate lock at all
#      (`tools/gate.sh --oxide`, `cargo oxide test|run`, a `target/release/` binary of a recipe that
#      runs `cargo oxide`: it lands on the box env's 3090 pin). A gpu-gate.sh recipe here runs with
#      BLOOMERY_GATE_CARD=3090 in BLOOMERY_BOX_ENV, so a mixed recipe's `any` calls stay on the 3090.
#   A|B  balanced: a recipe whose every gpu-gate.sh call carries the `any` form (gate-ptx-spill, whose
#      scan JITs through one in tools/ptx-scan.sh), or that runs no device code (CPU gates, check, lint,
#      fmt-check, check-*). It goes to the lane expected
#      to finish first: the fixed items' expected times are summed first, then the balanced items go
#      longest first (ties in list order), each to the lane with the smaller running sum (lane B on a
#      tie). Its card is its lane's, never `any`: BLOOMERY_GATE_CARD=3090 in lane A, a6000 in lane B
#      (where it never queues on the 3090 lock); a recipe with no device code gets none.
#   X  alone, after both lanes end: a recipe (or a dependency) that carries the just attribute
#      `[group('solo')]`, whose gate pins a host-memory count (page faults, page-cache residency) that
#      another lane's model loads move; and a recipe that uses box.sh's own card pick
#      BLOOMERY_CARD=both|a6000 (stage-gpu-load-v41*, which carry the attribute too): its gpu-gate.sh call
#      takes the gate lock of every card box.sh put in view (both cards: both locks), so it holds them
#      against other tracks' gates, and it still runs alone here. A solo gpu-gate.sh item keeps a card (3090 for a 3090 gate,
#      a6000 for an `any` one); a BLOOMERY_CARD recipe gets none (box.sh picks). Alone means within
#      this batch: another track's box jobs still run. The attribute is read from `just --dump --dump-format json`, where a recipe's
#      "attributes" list holds {"group": "solo"}; a `[group('…')]` value anywhere in the justfile other
#      than solo, solo-real, v41-load and host is a named error — a group is a scheduling class here, and a typo must
#      not drop a gate out of its class.
#   solo-real: `[group('solo')]` in the real tier, balanced in the fixture tier (--tier below). A gate whose reason for running
#      alone is its host set — the whole model loaded beside other lanes, and the box-wide V4.1 load lock — and not its two
#      cards or a host-memory count it pins. Its gpu-gate.sh calls all take the `any` card, which is what the fixture tier
#      balances it on; it carries no `BLOOMERY_CARD=both` pick (two cards keep it alone in both tiers: use solo), no `solo` or
#      `host`, and no `[group('v41-load')]` pin in the fixture tier (below) — each a named error where it can be told from the
#      recipe's text. A box.sh pick of the A6000 alone (`BLOOMERY_CARD=a6000`, one card, a gpu-gate.sh call that takes the pick)
#      is allowed: the fixture tier fixes the recipe to lane B, the A6000's lane, first in the lane's order, and no lane steals
#      it; the real tier keeps it alone, as any card pick does. A solo-real recipe's arm handed through another recipe is
#      refused in both tiers, as a solo one's is.
#   host: a recipe that carries `[group('host')]` — device code built and run by `cargo oxide test` on a
#      device-linked crate whose tests open no card (tools/recipes.py check holds that: the code of every
#      test it runs names no card opener) — is balanced like a recipe with no device code, with no card
#      forced. With solo, v41-load, a tools/gpu-gate.sh call or a box.sh card pick it is a named error.
#   v41-load: a recipe (or a dependency) that carries `[group('v41-load')]` — one whose binary loads the
#      whole V4.1 model (body::open, HostResidency::at_load: the host set, ~190 GB, populated) — runs
#      in lane A even when its gpu-gate.sh call takes `any` (then on the 3090). Not alone: lane B keeps
#      running. solo wins. What keeps two V4.1 loads apart on the box, across trees and batches, is
#      tools/gpu-gate.sh's V4.1 load lock, which a member asks for by exporting
#      BLOOMERY_GATE_V41_LOAD=1 in its box command (group and export must name the same recipes, a named
#      error otherwise). The lane is for the plan: members in two lanes would serialize on that lock
#      anyway, so lane sums that ran them in parallel would be wrong. In the fixture tier the group pins
#      nothing: a fixture load takes no V4.1 load lock (tools/gpu-gate.sh), so a member is placed by its
#      gpu-gate.sh form like any recipe, and a 3090-only call still keeps it in lane A.
#   C  the chain (the real tier, --lanes 2 only; loadchain, 2026-10-09): the v41-load lock is the
#      batch's critical path, so its members are one serial resource both lanes feed, not a lane.
#      Members: every v41-load recipe that is not solo and takes no BLOOMERY_CARD=both pick (the
#      recipes lane A held), plus each solo-real recipe CHAIN_SOLO_REAL (below) names. One member at
#      a time, batch-wide: a mutex (a directory under DIR) around each member's run, whatever its rc —
#      a batch-side mirror of gpu-gate.sh's box-wide lock, so a lane never parks on it with its card
#      idle; a lane that dies holding it ends the batch by name, never a wait without end. A member's
#      card candidates, from its forms and its box pick: both cards when every gpu-gate.sh call takes
#      the `any` form (one ledger key per candidate, as a balanced item); the 3090 when a call does
#      not or it runs device code with no gate lock; the A6000 alone for a BLOOMERY_CARD=a6000 pick,
#      with no card forced (box.sh picks, as the recipe's own key keeps). A member green on one
#      candidate's key skips there at 0 s; green on both, it takes the 3090's. Order, by model family
#      (family() below): the family lane X opens with goes last, so X starts on a warm host set; the
#      chain's other families in the reverse of X's family order, then families X does not hold in
#      first-appearance order; within a family, record order. A lane takes the first unclaimed member
#      in that order whose candidates include its card, within the family of the first unclaimed
#      member — it never opens the next family while the current one holds an unclaimed member, even
#      one only the other lane's card can run. In the fixture tier and --lanes 1 the class does not
#      exist: a member is placed as the v41-load paragraph says. The dry run lists the members as
#      `lane C` with their candidates; the lane and card of each are decided at run time.
#   pool: a recipe (or a dependency) that carries `[group('pool')]` — its host tier spins a worker
#      pool on the cores a big load's pool is pinned to (crates/threads: the pool is process-wide and
#      pins its workers; crates/gpu/src/host/step.rs: the host tier's service is a pool job) — never
#      starts beside a chain item. It changes no placement: the recipe's calls and other groups place
#      it as without the attribute.
#   big-host: a recipe (or a dependency) that carries `[group('big-host')]` — it locks or allocates
#      more host memory than fits beside a chain item's host set — is held to the same rule as pool.
#      No byte table lives here: the group is the one owner of the fact.
#   pack (the X phase, --lanes 2; packsched, 2026-10-09): lane X's items no longer run one at a
#      time. The phase is a pack: an item starts when a card is free, the summed host bytes of the
#      running items fit PACK_BUDGET (below: loadchain's B, a lower bound on the reading every
#      load's own HostNeed check takes), no two [group('pool')] items run at once (batch-wide: the
#      lanes' takes and the pack hold one pool mutex, two pools spinning on the same pinned cores),
#      and an item the resource table flags m — a clause that measures the host: page-fault pins, or
#      a plan-time room read its clauses' premises follow — starts with nothing else running, and
#      nothing starts beside it. A both-cards item still needs both cards free (nothing else runs);
#      a one-card item takes a free card, and a solo item whose gpu-gate.sh call takes the `any`
#      form now has both cards as candidates, its ledger key and times row following the card it
#      took. The table (tools/gate-batch-resources.tsv) holds one row per solo/solo-real recipe:
#      its flags and its host bytes, the plan's HostNeed derived per family (the fixture tier's
#      loads are the tree's small files and reserve nothing — the pack reads 0 bytes there). A solo
#      recipe with no row, or a row for a recipe that is not solo, is a named error: a recipe that
#      runs in the pack declares its bytes. Launches take the longest-est eligible item; a single
#      item always starts (the budget bounds pairs, and one load's own check is its own); the dry
#      run simulates the pack from the expected seconds and the predicted line's laneX= is that
#      wall. --lanes 1 keeps today's serial lane.
#   deferred (the fixture tier only; class D): a recipe the fixture tier cannot run, which the batch defers to the real tier instead
#      of running it or counting it red. Two kinds, two owners. (1) real-only: the gate has no fixture-tier conversion. Its box
#      command opens with `bash tools/ref/real-only.sh <its own name> && …` (every box.sh command of the recipe does), and that call
#      is the whole declaration: this runner reads the recipe's text, the box command runs the same script (it stops a run by hand
#      under BLOOMERY_TIER=fixture by name, exit 66, before a build or a lock). A recipe whose closure holds such a call is real-only
#      too. (2) no-fixture: the recipe's family (the BLOOMERY_MODEL of its box line, else the default profile) has no row in the
#      fixture table of tools/ref/ref-paths.sh — a profile that table does not name falls to its `*)` row, `none`, where box.sh
#      exits 66; this runner reads that table, so the fact has one owner. In the real tier neither kind places anything: the recipe
#      is placed by its groups and calls as without them. In the fixture tier a deferred item gets no lane, no ledger key, no claim
#      and no times row, and the batch never calls the recipe, so it is never a run on the fixture and never a red. Its run.log line
#      is `<item> rc=deferred <real-only|no-fixture> lane=- deferred=1`, the dry run names it (`lane -`), and the batch writes every
#      deferred item to DIR/deferred.list (real-only gates first, then no-fixture families, one item per line in the --list format,
#      comment lines between), so a real-tier batch takes exactly that set with `--list`. The DONE line counts each in `deferred=`
#      (whole gates, where it otherwise counts clauses), then `real_only=<n>`, `no_fixture=<n>` (each when not 0) and
#      `deferred_list=<path>`. Named errors, wherever the recipe sits in the justfile: a real-only call that names another recipe, a
#      box.sh command of the recipe that does not open with its call (or a call outside one), a missing script, a recipe that is
#      both real-only and solo-real (a fixture-tier placement it never uses), and a fixture table this runner cannot read.
#   --tier real|fixture (default real; BLOOMERY_TIER in the environment names the same, and the two must agree): the tier
#      the batch runs in, passed to every item as BLOOMERY_TIER=fixture in its BLOOMERY_BOX_ENV — tools/box.sh resolves each
#      item's model file from it (the family's fixture, tools/ref/ref-paths.sh); a family with none is a deferred item (above,
#      never a run on the real file, never a red). The real tier adds nothing to the environment: its commands, keys and times are
#      today's. In the fixture tier a `[group('solo-real')]` recipe is balanced and `v41-load` pins no lane (above), the keys
#      carry the tier (they carry the box env), and the times file keeps the fixture runs apart from the real ones — a fixture
#      item's row is `name@BLOOMERY_TIER=fixture[…]`, an entry of its own. An item's own `@BLOOMERY_TIER=…` is a named error:
#      the tier is the batch's. Clauses a fixture run leaves to the real tier print `deferred(real)` lines (crates/gpu-gates/
#      src/tier.rs); each item's line in run.log gains `deferred=<n>` when its log holds any, and the DONE line ends with the
#      batch's total (a deferred item counts one; the deferred paragraph above has the rest of the line).
# A recipe that runs a timing runner or takes the timing lease is refused, because a batch's builds
# contaminate a timed run. Two tests: what a script the recipe runs does, followed transitively — a
# tools/…sh that calls lease_take (tools/ref/lease.sh), or that opens the lease lock for a descriptor
# and waits on it with flock -w — and, for what that walk cannot see, the names in timed() below. Also
# refused: a recipe whose text runs this script (`just smoke`: a batch inside a batch runs lanes of its
# own on the same cards), and an item whose ARGS hand a `[group('solo')]` recipe's gpu-gate.sh binary
# one of the --flags that recipe passes it (in the self-test's fixture, `v41-a:--faults` is the solo
# recipe v41-solo's arm, which runs alone and with BLOOMERY_HOST_LOCK=1) — the error names the solo
# recipe to list instead. Lane A runs its items in list order, lane B in the order the balance placed them
# (its skips, then longest first), lane X by model family (the profile box.sh loads: V4.1 first, then each
# family in the order its first item is listed, each in list order); A and B run concurrently (cargo
# serializes their builds by one lock on the box tree's target directory, so a long lane-B build makes a
# lane-A item that starts inside it wait: lane B's order decides when its long builds start, and in a
# landing list whose longest balanced item runs on the Mac, check-recipes, they start after it).
# Work stealing: a lane that empties its own queue takes the other lane's not-yet-started movable
# items, in run order — a balanced item (class F: either card, or no card) that is not a ledger
# skip. A fixed, solo, v41-load or one-lane item never moves, and neither does an item the ledger
# already holds green (its key names one card). The thief runs a stolen item on its own lane's card
# — the item's other candidate: BLOOMERY_GATE_CARD is rewritten, and the ledger key, the ledger
# state and the times row's card label move with it, so the record names the card the item ran on
# and run.log's lane= names the lane that ran it; a no-card item (host group, plain CPU checks)
# changes lanes only. A steal onto a card first probes the timing lease through box.sh (READONLY,
# so no rsync: `. tools/ref/lease-probe.sh && lease_free`, the shared-lock probe — never an
# exclusive flock) and takes no card while the lease is held, re-probing after RETRY_WAIT seconds
# — waited in 1 s slices that end the pass the moment no candidate remains, so a sitting longer
# than the wait costs the lane nothing beyond its own items; a hold, or a sitting that starts after
# the probe, is the stolen item's own box.sh guard, as it is for every item. --lanes 1 steals
# nothing.
# --lanes 1 runs every item in the
# list's order in one lane (labelled A), with no card forced — the recipes' own defaults, as a hand
# batch runs them.
#
# Expected times, the balance's input: the median of the item's last 5 rows in the times file
# ${BLOOMERY_GATE_TIMES:-~/.cache/bloomery/gate-times.tsv} (on the Mac, shared by every tree),
# `item<TAB>lane<TAB>card<TAB>secs<TAB>date`, one row per item that ran, green or red: the item as
# written without its lane's card (a plain item is its recipe name; an item with @env or :ARGS is an
# entry of its own — `gate-gpu-ds41-prefill:--cases 512 …` is not the full prefill), the lane and card
# it ran on (3090 or a6000; both or a6000 for box.sh's own pick; none for no device code; any under
# --lanes 1, where the recipe picks at run time), the seconds of its final try (an rc 75 try before it
# waited on a lock and ran nothing; an item whose last try is still rc 75 writes no row), and the date.
# A final try that built cold — its log shows cargo compiling a crate from outside the tree, a registry
# or git dependency: a new remote directory, a toolchain, flag or pin move — writes its row to the
# sibling file <times file without .tsv>-cold.tsv, same format, which no plan reads, and its item line
# gains `cold=1`: one build must not drag the item's median. The row format stays five fields — every
# tree's copy of this script reads the one shared file.
# Only this script appends, each row with one write(2) on an O_APPEND descriptor under an exclusive
# flock; the plan reads the file under a shared one. A malformed row is a named error before anything
# runs. An item with no row expects DEFAULT_S (45 s, derived: the fixup5 batch's mean item, 2,167 s
# over 48 recipes, docs/gates-plan.md §1); --dry-run names the items that used it. With --ledger an
# item that skips counts 0 s, and a balanced gpu-gate item whose key is green on one card only goes to
# that card's lane, where it skips: the key carries the lane's card, so an item placed by time alone
# would run again on the other card although nothing it reads moved.
#
# Per item: stdout+stderr to DIR/g-<recipe>[-<n>].log (n for the n-th repeat of a recipe; each try
# appends under a `=== try` header). rc 75 (lock contention) retries after 30 s, up to 10 times; any
# other rc is final —
# except in a --round-ledger batch, where a try that ended rc 75 on a GPU hold another batch put up is
# not retried (hold75_owner reads the line tools/gpu-gate.sh prints only at that bound, which names the
# hold's owner): the item ends `rc=yield held by <owner>: not retried (a round batch yields to the
# lead's hold)`, counts in the DONE line's `yielded=` and in no rc class, and writes no times row and no
# ledger record. A plain card-lock rc 75 and the lead's --ledger batch keep today's retry.
# DIR/run.log opens with `plan laneA=<s>s laneB=<s>s laneX=<s>s wall=<s>s
# defaults=<n> …` (the predicted sums, derived from the times file), then gets `<recipe>[-<n>] rc=<n>
# <s>s try=<t> lane=<A|B|X>` per item (plus `cold=1` for a cold build, `times=append-failed` when its
# times row could not be written, and `item=…` last when it carries env or ARGS), then `DONE total=<n> red=<n> wall=<s>s laneA=<s>s laneB=<s>s`
# (`laneX=` when X ran, `lint_warnings=<n>` when lint ran: `grep -c '^warning:'` on its log). A fixture batch's real-only item has
# its own line (`<item> rc=deferred <real-only|no-fixture> lane=- deferred=1`), counted in `total=` and in no other rc class; a
# yielded item (`rc=yield`, above) is counted the same way, in `yielded=` when any did. Exit 0
# iff every rc is 0 (or deferred, yielded, or a ledger skip). DIR defaults to $HOME/.cache/bloomery/batches/<tree>/<stamp> (<tree> the
# basename of this worktree's root): outside every tree, so the logs survive a `rm -rf target/`
# cleanup of the worktrees and neither mark a tree dirty nor reach box.sh's rsync. An explicit
# --out is still taken: under the tree's target/ it is gitignored and accepted, a DIR inside the
# tree but outside target/ is refused (box.sh would ship the logs and mark the tree dirty).
#
# Refuses to start while the timing lease (/root/bloomery-cpu.lock) or a hold
# (/root/bloomery-<owner>-hold, other than BLOOMERY_HOLD_OWNER's) is up, naming what is up (exit 75);
# a lease that cannot be tested is exit 70; no override. The check is tools/box.sh's own guard
# (lease_guard, tools/ref/lease-probe.sh) with BLOOMERY_BOX_WAIT=0. Once the batch runs, every item's box.sh
# call passes that guard with its default wait: a sitting that takes the lease mid-batch gets a quiet
# box — the item waits (up to 30 min, then rc 75, which retries below) and starts after the sitting.
# The wait comes before the item's command, so the runners' 900 s bound does not count it; nor does the
# item's times row: the final try's seconds less the waits its log names (box.sh's guard, gpu-gate.sh's card
# lock and V4.1 load lock), which the item line prints as `waited=<s>s`. --dry-run validates, prints each item's lane, command and
# plan (fixed, balanced or solo, with its expected seconds) and the predicted lane sums, and touches
# neither the box nor DIR (with --ledger it reads the box once, for the manifest below).
#
# Budget refusal. Before anything runs (after the plan validates): every item whose expected
# seconds pass 0.75 × the bound that would kill it — the BLOOMERY_GATE_BOUND of its one bounded
# call (tools/gate.sh, tools/gpu-gate.sh, tools/host-gate.sh; unset 900, tools/gate-bound.sh) — is
# a named refusal, exit 65: a slower gate is a red gate in the making, and a suite whose slowest
# item overflows its bound is an error, not an overflow. --over-budget-ok passes them (the dry run
# names both the refused and the unchecked — no single bounded call, or a bound the text does not
# spell: several calls, an arithmetic, no runner at all).
#
# Disk floor. Before anything starts — after argument parsing, before the log directory and the first
# lane — the free space of the volume that holds the tree's target/ and of the volume that holds OUT
# must each be at least MIN_FREE_GIB (giB). Below the floor: exit 69 and one `gate-batch: disk:` line
# naming the free GiB, the floor and the mount point — not 75, which is lock contention a batch
# retries. A df that fails or whose output does not parse is also 69, naming why. --dry-run prints
# the same verdict lines and does not fail: a dry run shows what a real run would do.
# BLOOMERY_MIN_FREE_GIB overrides the floor; an override that is not a positive integer is a named
# error (64), never silently ignored.
#
# Ledgers. Two files, one format, written only by this script: the lead's (--ledger, the lead's
# batches) and the rounds' (--round-ledger, a delegated round's batches). A round never passes --ledger:
# it writes the rounds' file only. What a batch reads to skip an item:
#   --ledger                  the lead's file (the default landing batch)
#   --ledger --trust-rounds   the lead's, then the rounds'; writes the lead's. The lead passes it only
#                             for a change that moves no behaviour — a move whose `just ptx-scan` equals
#                             the base (user, 2026-09-27: a round runner's recorded green at the same
#                             key is not re-run at landing; an agent saying "green" is not a record).
#                             The tool cannot tell a move from a change, so the flag is that judgment.
#   --round-ledger            the lead's, then the rounds'; writes the rounds'.
# A skip names its source, `src=lead` or `src=round`, in run.log and the dry run. A rebase moves the
# key of every item whose closure the landed commits touched, so a landing still reruns those; the
# saving is in the items the rebase left alone.
# Each item's input key is `tools/recipes.py key` (its section header lists the parts): the files its
# targets read (the walk `just affected` uses; the whole tree for a command run on the Mac, a cargo
# subcommand the parser does not model, or a script that walks the tree), the text of its recipe and
# of every recipe it depends on, the scripts those name, the cargo globals and the cuda-oxide pin, its
# ARGS, its full BLOOMERY_BOX_ENV (the caller's, the item's, its lane's card), the Mac-side variables
# box.sh carries, and a box manifest read once per batch through one box.sh call (`recipes.py
# box-manifest` behind box.sh's guard with BLOOMERY_BOX_WAIT=0: refused while the lease or a hold is
# up): ~/bloomery-env.sh, the kernel, each card's
# name, UUID, driver and VBIOS, rustc, llc, ptxas and cargo-oxide, the cuda-oxide backend, the content
# of $BLOOMERY_DATA (sha256 per file, cached by stat on the box in ~/.cache/bloomery/sha256-cache.tsv,
# so a file rewritten with the same bytes keeps its hash; the first fetch hashes everything, later ones
# only what moved), and the stat of every file in each /models/<name> directory the profiles, the box
# env or the tree name. The whole manifest is in every key but its model rows: the data a gate reads
# is not bounded by its recipe text, so a data change reruns every item — the safe side; a model
# directory's rows enter only the keys of the items that can open it (a /models literal in the item's
# files or recipe text, its profile among them, or a /models value in the box env).
#   Keys are computed before the lease check. An item whose key is in the ledger does not run: run.log
# gets `<stem> rc=skip green-at=<commit> <date> lane=<L>`. Every other item runs as without --ledger;
# one whose final rc is 0 is recorded after its key is computed again (against the same pre-batch box
# manifest) and found unchanged, so a tree-side input that moved while the batch ran is never recorded
# as green. Item lines gain `ledger=<state>`: recorded, changed (an input moved during the batch), red, yield (a --round-ledger
# item that yielded to another batch's GPU hold — nothing ran, nothing recorded),
# never, unkeyed (no key before the batch, or a recheck that failed — named on stderr), append-failed. The DONE line gains `skipped=<n>`; the exit code counts only the items
# that ran. --rerun runs every item and still records. --dry-run --ledger prints each item's status and
# why: `skip` with the green run's commit, date and tree; `run` with what moved since the item's last
# green (the parts of each green key are kept in <ledger>.parts/<key>.parts).
#   Never skipped: a recipe that runs a reference tree's files (ik, llama.cpp, mistral.rs — today
# gate-1-1 and gate-tokenizer), an item whose env or ARGS name an
# absolute path the manifest does not read, and — with --lanes 1, where no card is forced — a recipe
# whose gpu-gate.sh call takes the `any` card, which it picks at run time. A key that cannot be computed
# (a missing file, a failed manifest) is a named error: the item runs and is not recorded.
#   The files: ${BLOOMERY_GATE_LEDGER:-~/.cache/bloomery/gate-ledger.tsv} (the lead's) and
# ${BLOOMERY_GATE_ROUND_LEDGER:-~/.cache/bloomery/gate-ledger-rounds.tsv} (the rounds'), on the Mac,
# outside every tree and shared by every lead's and round's tree, `key<TAB>recipe<TAB>item<TAB>commit<TAB>date<TAB>tree`, one line
# per green item. Only this script appends to it, each line with one write(2) on an O_APPEND descriptor
# while holding an exclusive flock on it: two leads' batches cannot interleave a line.
# Honest limits — what the key cannot see:
#   - a box file or process outside the manifest: the reference trees (hence never-skip), the box's own
#     tools the gates run (curl, crates/gpu-gates/src/bin/gate_ds41_serve.rs:200; sha256sum,
#     gate_e2e.rs:579 and gate_qwen3moe_e2e.rs:1029; nvidia-smi, crates/gpu-gates/src/bind.rs:302;
#     coreutils, python3), the page cache;
#   - hardware and kernel state: the 3090's bus (Xid 79), thermal and power events, and the topology
#     and memory state the gates read from /proc and /sys (crates/threads/src/lib.rs:542,
#     crates/gpu/src/model/launcher.rs:278, crates/engram/src/prefetch.rs:477 and lib.rs:700,
#     crates/gpu-gates/src/bin/gate_load_v41.rs:701 and :712);
#   - a model file's content: model files are keyed by size, mtime, ctime and inode;
#   - a model directory a gate opens through a path no literal spells (one ~/bloomery-env.sh exports,
#     one a program builds at run time): its rows reach no item's key;
#   - a box file that changes during the batch (the manifest is read once, before it): the next batch's
#     key differs, so the item reruns then;
#   - a stale binary: the key trusts cargo's freshness check, so a green run of a binary cargo failed
#     to rebuild would be recorded;
#   - a directory symlink inside $BLOOMERY_DATA, keyed by its target path, not walked (none today).
# $BLOOMERY_DATA/nsys/ and ncu/ are left out of the manifest: only the timing runners write them and no
# gate reads them (recipes.py's self-test fails when a gate names one).
#
# The GPU hold. A --ledger batch (the lead's landing batch) puts a GPU-only hold up on the box for its whole run, so that
# other tracks' GPU gates stop queueing on the cards between its items: /root/bloomery-batch.gpuhold, `owner=<id> since=<epoch>`,
# written and read only by tools/gpu-gate.sh (--gpuhold; its header has the wait, the bound and the stale rule). Each item runs
# with BLOOMERY_BATCH_OWNER=<id> in its box env — added to the call, never to the key, the times row or the ledger — so the
# batch's own gpu-gate.sh runs pass and every other run waits at the top of its poll, holding no lock. Builds and every non-GPU
# box command ignore it (its name does not match the sittings' /root/bloomery-<owner>-hold). The hold goes up after the start
# check, only when some item runs on the box, and is refused (75) while another batch's fresh hold is up; a heartbeat (a
# background job, pid in DIR/gpuhold.pid) refreshes its mtime every 60 s (BLOOMERY_GPU_HOLD_BEAT, for the self-test), through box.sh
# with BLOOMERY_BOX_READONLY=1 and BLOOMERY_BOX_WAIT=0: no sync, no guard wait — a deliberate write of that one control file.
# The box calls a hold stale after 300 s without a beat (this Mac asleep or cut off), so it never outlives its batch by more than
# that. The hold comes down on every exit — the end, a red batch, `die`, INT/TERM — by the EXIT trap, which kills only the
# heartbeat pid written under DIR and removes the hold only if it is still this batch's. DIR/gpuhold.log holds the calls' output.
# --no-gpu-hold (with --ledger) runs without it; --round-ledger and a plain batch never take it. A --round-ledger batch's item
# that ends rc 75 on a hold another batch put up does not wait it out try after try: it yields (rc=yield, the DONE line's
# yielded=), so a round never starts in the first gap the lead's batch leaves — the gap the lead's own next step needs.
# --dry-run prints the hold it
# would take and touches nothing.
# Stopping a batch (INT/TERM): the trap signals only the pids written at spawn under DIR (lane-*.pid,
# lane-*.child). A box process a killed item left behind is `just box-gc`'s to clear.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
JUST=(just --justfile "$ROOT/justfile" --working-directory "$ROOT")

SMOKE=(
  check-recipes check-rustflags check-comments check-levers check-arch check fmt-check lint gate-ptx-spill
  gate-gpu-ds41-step
  "gate-gpu-ds41-prefill:--cases 512 --no-split --no-extra"
  gate-gpu-e2e
)
TRIES_MAX=11
RETRY_WAIT=30
STEAL_WAIT=$RETRY_WAIT # a held timing lease: the steal's re-probe bound, waited in 1 s slices
DEFAULT_S=45 # the expected seconds of an item with no row in the times file (the header says why 45)
MIN_FREE_GIB=3 # the disk floor (the header's «Disk floor»): 2× the larger of one cold check's and one cold combos' growth of a tree's target/ (tools/mac-check.sh), rounded up to a whole GiB
TIMES_FILE=${BLOOMERY_GATE_TIMES:-$HOME/.cache/bloomery/gate-times.tsv}
COLD_FILE=${TIMES_FILE%.tsv}-cold.tsv # a cold build's rows, outside the median (the header)
# The solo-real recipes that join the chain (class C, the header). Each one's own reason was read
# (loadchain's D5 N3, 2026-10-09): the stated reason is the host set under the big-load lock — not
# two cards, not a host-memory pin — and no clause or plan of its gate reads a host-memory value a
# neighbour moves. The Qwen3.8 e2e gates stay out by name: their plan reads the host's room
# (crates/model/src/arch/qwen35moe/place.rs, PlanInputs::of) and its PLE table's tier follows that
# reading. A name here must carry solo-real and v41-load wherever the justfile holds it (a member
# loads under the lock); a name the justfile does not hold is ignored — the self-test's fixture
# justfile holds its own recipes.
CHAIN_SOLO_REAL='gate-gpu-ds41-residency-a gate-gpu-glm5next-e2e gate-gpu-glm5next-stagger gate-gpu-glm5next-mtp gate-gpu-glm5next-residency'
# The pack's host budget (the header's «pack»): the bytes the running X items' loads may sum to.
# Loadchain's B (its D5 N2): the box's lowest observed MemAvailable 263,259,930,624 B (train028b's
# serve logs) less a 1 GiB margin for a reading lower than train028b's lowest = 262,185,889,800 B.
# A set of loads whose HostNeeds sum within it each pass their own check (crates/gpu/src/model.rs,
# host_available: a neighbour's populated page cache counts as available; its locked bytes do not).
# BLOOMERY_PACK_BUDGET overrides it (bytes, a positive integer; the self-test's numbers).
PACK_BUDGET=262185889800

USAGE="usage: tools/gate-batch.sh [--out DIR] [--smoke | --weekly | --list FILE | ITEM…] [--dry-run] [--lanes 1|2] [--tier real|fixture] [--ledger [--trust-rounds] [--no-gpu-hold] | --round-ledger] [--rerun] [--over-budget-ok] | --classes | --self-test"
die() { echo "gate-batch: $*" >&2; exit "${RC:-64}"; }

# The append, for the ledger and the times file: a ledger record's parts file first (its parts exist
# once the record does), then the line, one write(2) on an O_APPEND descriptor under an exclusive flock.
IFS= read -r -d '' PYAPPEND << 'PY' || true
import fcntl, os, shutil, sys

path, line = sys.argv[1:3]
if len(sys.argv) == 5:
    parts_src, parts_dst = sys.argv[3:5]
    if not os.path.exists(parts_dst):
        os.makedirs(os.path.dirname(parts_dst), exist_ok=True)
        tmp = f"{parts_dst}.{os.getpid()}.tmp"
        shutil.copyfile(parts_src, tmp)
        os.replace(tmp, parts_dst)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
data = (line.replace("\n", " ") + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    if os.write(fd, data) != len(data):
        sys.exit(f"gate-batch: a short write to {path}")
finally:
    os.close(fd)
PY

times_record() { # $1 = plan index, $2 = lane, $3 = the final try's seconds, $4 = 1 for a cold build
  local f=$TIMES_FILE
  [ "${4:-0}" = 0 ] || f=$COLD_FILE
  python3 -c "$PYAPPEND" "$f" "$(printf '%s\t%s\t%s\t%s\t%s' "${P_TKEY[$1]}" "$2" "${P_TCARD[$1]}" "$3" "$(date '+%Y-%m-%dT%H:%M:%S%z')")"
}

# cold_of <log> <try>: 1 when the try's section of the item's log compiled a crate from outside the tree
# (cargo's `Compiling <name> v<ver>` with no local ` (/path)`: a registry or git dependency), which a
# build does only in a new remote directory or after the toolchain, the flags or a pinned dependency
# moved; else 0. Such a try's seconds are a one-off build, not the item's.
cold_of() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && /^ *Compiling [A-Za-z0-9_-]+ v[0-9]/ && !/ \(\// { c = 1 }
    END { print c + 0 }' "$1"
}

# deferred_of <log> <try>: the `deferred(real)` lines of the try's section of the item's log — the clauses a fixture-tier
# gate left to the real tier (crates/gpu-gates/src/tier.rs prints them; its test holds this prefix to that file's).
deferred_of() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && /^deferred\(real\) / { n++ }
    END { print n + 0 }' "$1"
}

# try_waits <log> <try>: the seconds the given try spent waiting, not running — box.sh's guard for a sitting
# (`[guard] … after N s: the command starts`) and tools/gpu-gate.sh for a gate lock, for the V4.1 load
# lock and for another batch's GPU hold (`gpu-gate.sh: waited N s for the …`),
# summed over the try's section of the item's log. A times row holds the rest: a lock queue is not a gate's cost.
try_waits() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && match($0, /after [0-9]+ s: the command starts/) { split(substr($0, RSTART + 6), a, " "); w += a[1] }
    on && match($0, /^gpu-gate\.sh: waited [0-9]+ s for the (gate lock|V4\.1 load lock|batch hold)/) { split(substr($0, RSTART + 20), a, " "); w += a[1] }
    END { print w + 0 }' "$1"
}

# hold75_owner <log> <try>: the owner the try's rc 75 names when it ended on a GPU hold another batch put
# up — tools/gpu-gate.sh's `the batch hold <path> (owner <owner>) was still up after N s (CARD_BOUND M s)`
# line, printed only at that hold's bound (its once-a-minute `[batch-hold]` wait lines say `waits:` and
# never match) — or '' when the 75 was a plain lock queue (a card lock, the V4.1 load lock or the timing
# lease, none of which names a hold owner).
hold75_owner() {
  awk -v t="=== try $2 " '
    index($0, t) == 1 { on = 1; next }
    /^=== try / { on = 0 }
    on && match($0, /the batch hold [^ ]+ \(owner [^)]+\) was still up after [0-9]+ s \(CARD_BOUND [0-9]+ s\)/) {
      s = substr($0, RSTART, RLENGTH); sub(/.*\(owner /, "", s); sub(/\).*/, "", s); print s; exit
    }' "$1"
}

# The disk floor, the header's «Disk floor». floor_gib resolves the floor once (the constant, or
# BLOOMERY_MIN_FREE_GIB, which must be a positive integer of GiB or a named 64). df_able walks up to
# the deepest existing ancestor of a path — df errors on a path that does not exist, and the volume
# of the deepest existing ancestor is where mkdir -p creates the rest (a symlink or mount inside the
# missing span cannot change it). disk_ok reads the Available column of `df -Pk` (POSIX output,
# macOS and Linux alike) and prints one verdict line: 0 above the floor, 1 below it (a real run
# turns that into 69; a dry run names it and continues), 69 when df fails or does not parse — a
# named failure in a dry run too, whose verdict line must be a verdict and not a guess.
floor_gib() {
  if [ -n "${BLOOMERY_MIN_FREE_GIB+x}" ]; then
    case $BLOOMERY_MIN_FREE_GIB in
      '' | 0 | *[!0-9]*) RC=64 die "BLOOMERY_MIN_FREE_GIB='$BLOOMERY_MIN_FREE_GIB' is not a positive integer of GiB" ;;
    esac
    printf '%s\n' "$((10#$BLOOMERY_MIN_FREE_GIB))"
  else
    printf '%s\n' "$MIN_FREE_GIB"
  fi
}

df_able() { # the deepest existing ancestor of $1
  local p=$1
  while [ ! -e "$p" ]; do
    case $p in
      / | '') printf '/\n'; return ;;
      *) p=${p%/*} ;;
    esac
  done
  printf '%s\n' "$p"
}

disk_ok() { # $1 a path whose volume must hold the floor free; one verdict line, then 0 / 1 / 69
  local path=$1 probe out rc=0 line
  local floor
  floor=$(floor_gib) || return $?
  probe=$(df_able "$path")
  out=$(df -Pk "$probe" 2>&1) || rc=$?
  [ "$rc" = 0 ] || RC=69 die "disk: df -Pk $probe failed (rc $rc): $(printf '%s\n' "$out" | tail -1)"
  rc=0
  line=$(printf '%s\n' "$out" | awk -v kib=$((10#$floor * 1048576)) -v floor="$floor" -v probe="$probe" '
    END {
      if (NR < 2 || $4 !~ /^[0-9]+$/) {
        printf "gate-batch: disk: df -Pk %s output does not parse (the Available column): %s\n", probe, $0 > "/dev/stderr"
        exit 2
      }
      mount = $6; for (i = 7; i <= NF; i++) mount = mount " " $i
      if ($4 + 0 < kib) {
        printf "gate-batch: disk: %.1f GiB free on %s (for %s), below the %s GiB floor — free space and run again\n", $4 / 1048576, mount, probe, floor > "/dev/stderr"
        exit 1
      }
      printf "gate-batch: disk: %.1f GiB free on %s (for %s), above the %s GiB floor\n", $4 / 1048576, mount, probe, floor
    }') || rc=$?
  [ "$rc" = 2 ] && rc=69
  [ "$rc" = 0 ] || return "$rc"
  printf '%s\n' "$line"
}

# --self-test: the rules of items 1 and 4 on a fixture justfile and fixture times rows, in a temporary
# tree holding a copy of this script (no box, no ssh; `just` and python3 only). One line per case,
# `ok <name>` or `FAIL <name>: <why>` with the output; exit 0 iff none failed. check-recipes runs it.
self_test() {
  local self t h n=0 bad=0 out rc
  unset BLOOMERY_TIER # the tier of a case is the case's own
  self=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/gate-batch-test.XXXXXX") || { echo "gate-batch: self-test: no temporary directory" >&2; return 70; }
  # The fake HOMEs sit outside the tree copy, never under $t: the default OUT is $HOME/.cache/…, and a
  # HOME under the tree is an OUT the --out-inside-the-tree guard (rightly) refuses on a symlink-free host.
  h=$(mktemp -d "${TMPDIR:-/tmp}/gate-batch-home.XXXXXX") || { rm -rf "$t"; echo "gate-batch: self-test: no temporary directory for the fake HOMEs" >&2; return 70; }
  # shellcheck disable=SC2064 # the paths are fixed now
  trap "rm -rf '$t' '$h'" EXIT
  mkdir -p "$t/tools"
  cp "$self" "$t/tools/gate-batch.sh"
  cat > "$t/justfile" << 'JF'
[group('v41-load')]
v41-any:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_x'

[group('v41-load')]
v41-a *ARGS:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_y {{ARGS}}'

[group('solo')]
[group('v41-load')]
v41-solo:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh gen_y --faults'

plain-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'

host:
    ./tools/box.sh 'bash tools/gate.sh -p x'

gate-x:
    ./tools/box.sh 'bash tools/gate.sh -p y'

gate-y:
    ./tools/box.sh 'bash tools/gate.sh -p w'

[group('host')]
hostdev:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p hd'

commented:
    # the timed recipes source tools/ref/timing-card.sh; this one runs none
    ./tools/box.sh 'bash tools/gate.sh -p c'

timedrun:
    ./tools/box.sh 'bash tools/ref/timing-card.sh probe'

nested *ARGS:
    ./tools/gate-batch.sh --smoke {{ARGS}}

weekly-a:
    ./tools/box.sh 'bash tools/gate.sh -p wa'

[group('solo')]
weekly-b:
    ./tools/box.sh 'BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh wb'

[group('solo-real')]
sr-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_sr'

[group('solo-real')]
[group('v41-load')]
sr-v41:
    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_sv'

[group('solo')]
x-glm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/gpu-gate.sh xg'

[group('solo')]
x-q:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/gpu-gate.sh xq'

[group('solo')]
x-v41:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/gpu-gate.sh xv'

[group('solo')]
x-none:
    ./tools/box.sh 'bash tools/gpu-gate.sh xn'

[group('solo')]
x-v41b: x-v41dep

x-v41dep:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/gpu-gate.sh xw'

steal-slow:
    ./tools/box.sh 'bash tools/gpu-gate.sh gen_s'

steal-fix:
    ./tools/box.sh 'bash tools/gpu-gate.sh gen_f'

steal-bal:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_b'

steal-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p sl'

# The real-only class: a call that opens the box command and names the recipe.
ro-any:
    ./tools/box.sh 'bash tools/ref/real-only.sh ro-any && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_ro'

[group('solo')]
[group('v41-load')]
ro-solo:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh ro-solo && export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_rs'

ro-chain: ro-any

ro-glm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/real-only.sh ro-glm && bash tools/gpu-gate.sh xr'

[group('solo-real')]
sr-a6000:
    BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/gpu-gate.sh gen_sa --place a'

# The chain's cases (class C): two members CHAIN_SOLO_REAL names in the real tree — a both-cards
# one and an A6000 pick — plus two same-family members on one card each, and the neighbour groups.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-e2e:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_ge'

[group('solo-real')]
[group('v41-load')]
gate-gpu-ds41-residency-a:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_ra --place a'

[group('v41-load')]
v41-d41-a:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh gen_da'

[group('pool')]
pool-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pl'

[group('big-host')]
big-any:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_bh'

# The pack's cases (the header's «pack»): one-card items that fit the budget, a pair that does not,
# an m-flagged item, a both-cards one; a second pool item beside pool-any; and a recipe whose bound
# lives two scripts deep (the budget scan follows the calls).
[group('solo')]
pk-a:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pa'

[group('solo')]
pk-b:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pb'

[group('solo')]
pk-big:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pg'

[group('solo')]
pk-big2:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pg2'

[group('solo')]
pk-m:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_pm'

[group('solo')]
pk-both:
    BLOOMERY_CARD=both ./tools/box.sh 'bash tools/gpu-gate.sh gen_pt'

[group('pool')]
pool-two:
    ./tools/box.sh 'BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_p2'

bt-deep:
    ./tools/box.sh 'bash tools/bt-one.sh'
JF
  # The pack's resource table: a row per solo/solo-real recipe of the fixture justfile above (a
  # recipe the justfile does not hold — sr-arm arrives in a case below — is ignored), and the pack
  # case rows the cases below read (BLOOMERY_PACK_BUDGET shrinks the budget per case).
  cat > "$t/tools/gate-batch-resources.tsv" << 'RT'
v41-solo	-	1000
weekly-b	-	1000
sr-any	-	1000
sr-v41	-	1000
x-glm	-	1000
x-q	-	1000
x-v41	-	1000
x-none	-	1000
x-v41b	-	1000
ro-solo	-	1000
sr-a6000	-	1000
sr-arm	-	1000
gate-gpu-glm5next-e2e	-	1000
gate-gpu-ds41-residency-a	-	1000
pk-a	-	100
pk-b	-	100
pk-big	-	9000
pk-big2	-	9000
pk-m	m	100
pk-both	-	100
RT
  # A bound two scripts deep: bt-deep -> bt-one.sh -> bt-two.sh holds the bounded call.
  printf '%s\n' '#!/usr/bin/env bash' 'bash tools/bt-two.sh' > "$t/tools/bt-one.sh"
  printf '%s\n' '#!/usr/bin/env bash' 'BLOOMERY_GATE_BOUND=100 bash tools/gpu-gate.sh gen_bt' > "$t/tools/bt-two.sh"
  # The default profile a recipe with no BLOOMERY_MODEL loads.
  mkdir -p "$t/tools/ref"
  # The GPU hold's box-side writer is the real tools/gpu-gate.sh (with the real lease-probe.sh it sources): the fake box below runs
  # its --gpuhold calls on the fixture, so the batch's hold is the real one end to end.
  cp "$(dirname "$self")/gpu-gate.sh" "$t/tools/gpu-gate.sh"
  cp "$(dirname "$self")/ref/lease-probe.sh" "$t/tools/ref/lease-probe.sh"
  cp "$(dirname "$self")/ref/real-only.sh" "$t/tools/ref/real-only.sh"
  cat > "$t/tools/ref/ref-paths.sh" << 'RP'
: "${BLOOMERY_MODEL:=${BLOOMERY_REF_MODEL_PROFILE:-deepseek2}}"
case "${BLOOMERY_TIER:-real}" in
  fixture)
    case "$BLOOMERY_MODEL" in
      qwen4exp) __fixture_dir=qwen38 ;;
      deepseek41) __fixture_dir=v41 ;;
      deepseek2 | qwen3moe | qwen35moe | qwen35) __fixture_dir=self ;;
      *) __fixture_dir=none ;;
    esac
    ;;
esac
RP
  # Unpinned, v41-any (50 s, balanced) would go to lane B: lane A holds v41-a's 100 s.
  printf '%s\t%s\t%s\t%s\t%s\n' v41-a A 3090 100 2026-09-27T10:00:00+0900 v41-any B a6000 50 2026-09-27T10:00:00+0900 \
    plain-any B a6000 10 2026-09-27T10:00:00+0900 host B none 5 2026-09-27T10:00:00+0900 > "$t/times.tsv"
  # A fake df first on PATH for the whole self-test — the disk floor's verdict must not depend on
  # the host's real disk. One POSIX data line whose Available column is DISK_KB KiB on /fake/mount;
  # DISK_RC makes df fail, DISK_BAD output that does not parse.
  mkdir -p "$t/fakebin"
  cat > "$t/fakebin/df" << 'DF'
#!/bin/sh
if [ -n "${DISK_RC:-}" ]; then echo "df: fake failure" >&2; exit "$DISK_RC"; fi
if [ -n "${DISK_BAD:-}" ]; then printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\ngarbage line\n'; exit 0; fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/fake 100000000 1000 %s 1%% /fake/mount\n' "${DISK_KB:-999999999}"
DF
  chmod +x "$t/fakebin/df"
  export PATH="$t/fakebin:$PATH"
  pass() { n=$((n + 1)); echo "ok $1"; }
  fail() {
    n=$((n + 1)) bad=$((bad + 1))
    echo "FAIL $1: ${2:-}"
    printf '%s\n' "$out" | sed 's/^/    | /'
  }
  # check <name> <want rc> <ERE the output must hold> <command…>
  check() {
    local name=$1 want=$2 pat=$3
    shift 3
    rc=0
    out=$("$@" 2>&1) || rc=$?
    if [ "$rc" != "$want" ]; then fail "$name" "rc $rc, want $want"
    elif ! grep -Eq -- "$pat" <<< "$out"; then fail "$name" "no line matches /$pat/"
    else pass "$name"; fi
  }
  local gb=(bash "$t/tools/gate-batch.sh")
  # The rc 75 cases below run a copy of the fixture script whose retry wait is 1 s, not 30 — a sed of
  # the one constant, the way gpu-gate.sh's --test-locks shrinks its bounds. No BLOOMERY_* name exists
  # for it: every such name is a row of the lever registry (tools/check-levers.sh holds tools/ to it).
  sed 's/^RETRY_WAIT=30$/RETRY_WAIT=1/' "$t/tools/gate-batch.sh" > "$t/tools/gate-batch-fast.sh"
  grep -q '^RETRY_WAIT=1$' "$t/tools/gate-batch-fast.sh" || { echo "gate-batch: self-test: the fast-retry copy did not patch (RETRY_WAIT's line moved?)" >&2; return 70; }
  export BLOOMERY_GATE_TIMES=$t/times.tsv
  check 'classes: an any-form v41-load recipe is a chain member over both cards' 0 \
    "^v41-any	C	chain	3090,a6000	v41-any: \[group\('v41-load'\)\]" "${gb[@]}" --classes
  check 'classes: an any-form recipe outside the group stays balanced' 0 '^plain-any	F	balanced	3090,a6000	' "${gb[@]}" --classes
  check 'classes: solo wins over v41-load' 0 '^v41-solo	X	solo	3090	' "${gb[@]}" --classes
  check 'classes: a host recipe with device code is balanced with no card forced' 0 "^hostdev	F	balanced	-	hostdev: \[group\('host'\)\]" "${gb[@]}" --classes
  check 'classes: a timed name in a comment of the body is not a timed recipe' 0 '^commented	F	balanced	-	no device code$' "${gb[@]}" --classes
  check 'classes: a timed script in command position is' 0 '^timedrun	T	timed	-	timedrun runs tools/ref/timing-card\.sh' "${gb[@]}" --classes
  check 'classes: a recipe that runs a batch is refused' 0 '^nested	R	refused	-	nested runs tools/gate-batch\.sh' "${gb[@]}" --classes
  check 'dry run: the chain member pins no card — the lane that takes it decides' 0 \
    "^lane C  v41-any +just v41-any$" "${gb[@]}" --dry-run v41-a v41-any plain-any host
  check 'dry run: both V4.1 loads are the chain' 0 'predicted laneA=5s laneB=10s laneX=0s chain=150s wall=150s' \
    "${gb[@]}" --dry-run v41-a v41-any plain-any host
  # Lane X by model family, V4.1 first, then each family in the order its first item is listed; a
  # dependency's profile is the recipe's, a recipe with none loads ref-paths.sh's default.
  out=$("${gb[@]}" --dry-run x-glm x-none x-q x-v41 x-glm x-v41b 2>&1) || fail 'order: lane X by family' "the dry run failed"
  got=$(printf '%s\n' "$out" | sed -n 's/^lane X  \([^ ]*\) .*/\1/p' | tr '\n' ' ')
  if [ "$got" = 'x-v41 x-v41b x-glm x-glm-2 x-none x-q ' ]; then pass 'order: lane X by family, V4.1 first, then first appearance'
  else fail 'order: lane X by family, V4.1 first, then first appearance' "got '$got'"; fi
  # Lane B runs longest first (the Mac-only check-recipes heads it in a landing batch), not in list order.
  out=$("${gb[@]}" --dry-run steal-fix host plain-any 2>&1) || fail 'order: lane B longest first' "the dry run failed"
  got=$(printf '%s\n' "$out" | sed -n 's/^lane B  \([^ ]*\) .*/\1/p' | tr '\n' ' ')
  if [ "$got" = 'plain-any host ' ]; then pass 'order: lane B longest first, not in list order'
  else fail 'order: lane B longest first, not in list order' "got '$got'"; fi
  check 'items: a batch as an item is refused by name' 65 'nested runs tools/gate-batch\.sh' "${gb[@]}" --dry-run host nested
  check "items: a solo recipe's arm through another recipe is refused, naming the solo recipe" 65 \
    "'v41-a:--faults' hands gen_y --faults, the arm of the solo recipe v41-solo" "${gb[@]}" --dry-run 'v41-a:--faults'
  check 'items: other ARGS of the same binary pass' 0 "^lane C  v41-a-2 " "${gb[@]}" --dry-run v41-a 'v41-a:--sets'
  # `just affected … 2>&1` output: just's echoed recipe line, then the `affected:` header and the recipe lines.
  printf '%s\n' './tools/affected-gates.sh main --no-box' 'affected: a..b — 2 changed files, 2 of 6 gate-recipes selected' \
    '  gate-x                      crates/x/src/lib.rs (+1)' 'unmapped: none' > "$t/aff-echo.txt"
  check "list: just affected output behind just's echoed line takes its recipe lines" 0 '^lane [AB]  gate-x ' \
    "${gb[@]}" --dry-run --list "$t/aff-echo.txt"
  # `just affected --narrow` output: the recipes it keeps run; a `  - gate-y` line it leaves out does not.
  printf '%s\n' './tools/affected-gates.sh main --no-box --narrow --scan b.log n.log' \
    'affected: a..b — 2 changed files, 1 of 6 gate-recipes selected (narrowed from 2 by ptx-scan)' 'narrow: ptx-scan gx: identical' \
    '  gate-x  crates/x/src/lib.rs (+1)' 'recipes: gate-x' 'always: ' 'left out (1), each by ptx-scan gx: identical:' \
    '  - gate-y  (ptx-scan gx: identical)' 'unmapped (0):' > "$t/aff-narrow.txt"
  check 'list: a narrowed affected output runs the recipes it keeps' 0 '^lane [AB]  gate-x ' "${gb[@]}" --dry-run --list "$t/aff-narrow.txt"
  if grep -q 'gate-y' <<< "$out"; then fail 'list: a narrowed affected output does not run a left-out recipe' "gate-y is in the batch"
  else pass 'list: a narrowed affected output does not run a left-out recipe'; fi
  # A weekly recipe a trigger named: its `  weekly-` line runs, beside the gate lines.
  printf '%s\n' 'affected: a..b — 2 changed files, 1 of 6 gate-recipes selected, 1 of 2 weekly-recipes by a trigger' \
    '  gate-x    crates/x/src/lib.rs  [lib x]' '  weekly-a  crates/w/src/lib.rs  [trigger tools/gate-paths.tsv:9 crates/w/**]' \
    'recipes: gate-x weekly-a' 'unmapped (0):' > "$t/aff-weekly.txt"
  check 'list: a weekly recipe a trigger named in affected output runs' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --list "$t/aff-weekly.txt"
  # --weekly: every weekly-* recipe of the justfile, each in its class; none of another name.
  check 'weekly: a weekly recipe with no group is balanced' 0 '^lane [AB]  weekly-a ' "${gb[@]}" --dry-run --weekly
  check 'weekly: a solo weekly recipe runs alone' 0 '^lane X  weekly-b ' "${gb[@]}" --dry-run --weekly
  if grep -Eq '^lane [ABX]  (gate|v41|host|plain)' <<< "$out"; then fail 'weekly: nothing but the weekly-* recipes' "another recipe is in the batch"
  else pass 'weekly: nothing but the weekly-* recipes'; fi
  check 'weekly: with items, a named error' 64 'exclusive' "${gb[@]}" --dry-run --weekly gate-x
  printf '%s\n' 'something else' 'affected: a..b — 1 changed file' '  gate-x  crates/x/src/lib.rs (+1)' > "$t/aff-other.txt"
  check 'list: any other line above the affected: header is a malformed item' 65 "aff-other.txt:1: malformed item 'something else'" \
    "${gb[@]}" --dry-run --list "$t/aff-other.txt"
  # Work stealing (the header's paragraph): real batch runs on the fixture, through a fake box.sh
  # that logs every call and never runs the command, and a key stub that keys an item by its own
  # text (env included: the card names differ). An item's env drives the fake: FAKE_SLEEP holds the
  # call that many seconds, FAKE_UNTIL=<word> ends the hold early once a logged command (or the
  # steal's lease probe, logged as `probe`) ends in <word>, then FAKE_GRACE seconds more — so a
  # case waits on the other lane's progress, not on a race — and FAKE_LEASE_UP=1 raises the lease
  # marker the fake answers the steal's READONLY probe (`. tools/ref/lease-probe.sh && lease_free`)
  # from.
  cat > "$t/tools/box.sh" << 'FB'
#!/bin/sh
here=$(dirname "$0")/..
st=$here/fake-state
mkdir -p "$st"
case "$*" in
  '. tools/ref/lease-probe.sh && lease_free')
    echo probe >> "$st/probe.log"
    [ -e "$st/lease-up" ] || exit 0
    echo '[guard] the fake lease is up' >&2
    exit 1 ;;
  'bash tools/gpu-gate.sh --gpuhold '*)
    # The hold's writer: logged with the box.sh flags the batch passed, then the real script on the fixture, its hold file under
    # fake-state (--test-locks).
    printf 'ro=%s wait=%s %s\n' "${BLOOMERY_BOX_READONLY:-0}" "${BLOOMERY_BOX_WAIT:-}" "$*" >> "$st/hold.log"
    cd "$here" || exit 70
    # shellcheck disable=SC2086 # the verb and the owner: two words
    exec bash tools/gpu-gate.sh --test-locks "$st" ${*#bash tools/gpu-gate.sh } ;;
esac
for kv in ${BLOOMERY_BOX_ENV:-}; do export "$kv"; done
printf '%s\t%s\n' "${BLOOMERY_BOX_ENV:-}" "$*" >> "$st/box.log"
# What an item sees of the batch's hold when it starts: its owner from the env, and the hold file as it stands.
printf '%s\t%s\t%s\n' "${BLOOMERY_BATCH_OWNER:-none}" "$(head -1 "$st/batch.gpuhold" 2> /dev/null || echo nohold)" "$*" >> "$st/seen.log"
n=0
while [ "$n" -lt "${FAKE_DEFER:-0}" ]; do
  echo "deferred(real) oracle: fake clause $n"
  n=$((n + 1))
done
# The rc 75 tools/gpu-gate.sh ends on at a bound, for the retry and yield cases: FAKE_CARD75=1 the plain
# card-lock queue, FAKE_HOLD75=<owner> a GPU hold another batch put up (that bound's line names its owner).
if [ -n "${FAKE_HOLD75:-}" ]; then
  echo "gpu-gate.sh: gen_x: the batch hold /root/bloomery-batch.gpuhold (owner $FAKE_HOLD75) was still up after 4 s (CARD_BOUND 4 s) — contention, not a red gate (rc 75)"
  exit 75
fi
if [ "${FAKE_CARD75:-0}" = 1 ]; then
  echo "gpu-gate.sh: no gate lock (3090) was free within 1800 s — contention, not a red gate"
  exit 75
fi
[ "${FAKE_LEASE_UP:-0}" = 1 ] && : > "$st/lease-up"
s0=$(date +%s)
k=0
while [ "$k" -lt "${FAKE_SLEEP:-0}" ]; do
  if [ -n "${FAKE_UNTIL:-}" ] && cat "$st/box.log" "$st/probe.log" 2> /dev/null | grep -q "${FAKE_UNTIL}\$"; then
    sleep "${FAKE_GRACE:-0}"
    break
  fi
  sleep 1
  k=$((k + 1))
done
echo "$s0 $(date +%s) $*" >> "$st/ran.log"
# …and whether the hold is still up when it ends (a lane that finished first must not have taken it down).
if [ -f "$st/batch.gpuhold" ]; then printf 'present\t%s\n' "$*" >> "$st/end.log"; else printf 'ABSENT\t%s\n' "$*" >> "$st/end.log"; fi
exit "${FAKE_RC:-0}"
FB
  chmod +x "$t/tools/box.sh"
  cat > "$t/tools/recipes.py" << 'FP'
#!/usr/bin/env python3
import sys
a = sys.argv[1:]
if a[:1] == ["box-manifest"]:
    sys.exit(0)
items, i, pd = [], 1, None
while i < len(a):
    if a[i].startswith("--"):
        if a[i] == "--box-env":
            import os
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fake-state")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "keycalls.log"), "a") as fh:
                fh.write(a[i + 1] + "\n")
        if a[i] == "--parts-dir":
            pd = a[i + 1]
        i += 1 if a[i] == "--rerun" else 2
        continue
    items.append(a[i])
    i += 1
import os
green = set()
if os.environ.get("FAKE_GREEN_FILE"):
    with open(os.environ["FAKE_GREEN_FILE"]) as fh:
        green = {ln.strip() for ln in fh if ln.strip()}
for k, it in enumerate(items):
    if os.environ.get("FAKE_SKIP") == "1" or f"stub-{it}" in green:
        print(f"stub-{it}\t{it}\tskip\tgreen-at=abc1234 2026-10-08T00:00:00+0900 tree=/x")
    else:
        print(f"stub-{it}\t{it}\trun\tstub")
    if pd:
        import os
        os.makedirs(pd, exist_ok=True)
        with open(f"{pd}/{k}.parts", "w") as fh:
            fh.write("stub\n")
FP
  want() { # <name> <file> <ERE>: the file holds a matching line
    n=$((n + 1))
    if [ -f "$2" ] && grep -Eq -- "$3" "$2"; then echo "ok $1"
    else bad=$((bad + 1)); echo "FAIL $1: no line matches /$3/ in $2"; fi
  }
  want_not() { # <name> <file> <ERE>: the file holds no matching line
    n=$((n + 1))
    if [ -f "$2" ] && grep -Eq -- "$3" "$2"; then bad=$((bad + 1)); echo "FAIL $1: a line matches /$3/ in $2"
    else echo "ok $1"; fi
  }
  want_row() { # <name> <file> <awk program over $1 (env) and $2 (command)>: a row answers it
    n=$((n + 1))
    if [ -f "$2" ] && [ -n "$(awk -F'\t' "$3" "$2")" ]; then echo "ok $1"
    else bad=$((bad + 1)); echo "FAIL $1: no row of $2 answers {$3}"; fi
  }
  want_no_row() { # <name> <file> <awk program>: no row answers it
    n=$((n + 1))
    if [ -f "$2" ] && [ -n "$(awk -F'\t' "$3" "$2")" ]; then bad=$((bad + 1)); echo "FAIL $1: a row of $2 answers {$3}"
    else echo "ok $1"; fi
  }
  steal_case() { # <tag> <times rows…> -- <items…>: one real batch run on the fixture, rc printed;
    # LEASE_UP=1 puts the fake lease up before the batch starts; GBFAST=1 runs the fast-retry copy above
    local tag=$1 rc=0 runner=("${gb[@]}")
    local times=$t/times-$tag.tsv
    shift
    local rows=()
    while [ "$1" != -- ]; do rows+=("$1"); shift; done
    shift
    printf '%s\n' "${rows[@]}" > "$times"
    rm -rf "$t/fake-state" "$t/target/c-$tag"
    if [ "${LEASE_UP:-0}" = 1 ]; then mkdir -p "$t/fake-state" && : > "$t/fake-state/lease-up"; fi
    # PRE_HOLD=<owner> [PRE_AGE=<s>]: another batch's GPU hold is already up on the fake box, last refreshed <s> seconds ago
    if [ -n "${PRE_HOLD:-}" ]; then
      mkdir -p "$t/fake-state"
      printf 'owner=%s since=%s\n' "$PRE_HOLD" "$(date +%s)" > "$t/fake-state/batch.gpuhold"
      python3 -c 'import os, sys, time; a = time.time() - float(sys.argv[2]); os.utime(sys.argv[1], (a, a))' "$t/fake-state/batch.gpuhold" "${PRE_AGE:-0}"
    fi
    # LFLAG: the ledger flag of the batch (default the round's); MTAG: a mutant copy's tag
    [ "${GBFAST:-}" != 1 ] || runner=(bash "$t/tools/gate-batch-fast.sh")
    [ -z "${MTAG:-}" ] || runner=(bash "$t/tools/gate-batch-$MTAG.sh")
    BLOOMERY_GATE_TIMES=$times BLOOMERY_GATE_LEDGER=$t/lead-$tag.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-$tag.tsv \
      "${runner[@]}" --out "$t/target/c-$tag" "${LFLAG:---round-ledger}" "$@" > "$t/out-$tag.log" 2>&1 || rc=$?
    printf '%s' "$rc"
  }
  rc_ok() { # <name> <tag> <want total>: the case's batch ended green
    if [ "$(cat "$t/rc-$2")" = 0 ] && grep -Eq "^DONE total=$3 red=0 " "$t/out-$2.log"; then pass "$1"
    else fail "$1" "rc=$(cat "$t/rc-$2") $(tail -1 "$t/out-$2.log" 2>/dev/null)"; fi
  }
  local d=2026-09-27T10:00:00+0900
  # A contended 3090: lane A's first item (fixed) holds its lane until the balanced item behind it
  # has run — on today's runner, 20 s and then in lane A. Lane B, idle, steals it onto the A6000;
  # the times row and the ledger key name that card.
  local slow_a='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=gen_b'
  rc=$(steal_case steal-a "$slow_a	A	3090	30	$d" "host	B	none	70	$d" "steal-bal	A	3090	60	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_a" host steal-bal plain-any)
  printf '%s' "$rc" > "$t/rc-steal-a"
  rc_ok 'steal: a contended lane A and an idle lane B end green' steal-a 4
  want 'steal: the balanced item runs in lane B' "$t/target/c-steal-a/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=B( |$)'
  want_row 'steal: its box call carries the A6000 card' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_b$/'
  want 'steal: its times row names lane B and the A6000' "$t/times-steal-a.tsv" "$(printf '^steal-bal\tB\ta6000\t')"
  want 'steal: the ledger records it under the A6000 key' "$t/rounds-steal-a.tsv" "$(printf '^stub-steal-bal@BLOOMERY_GATE_CARD=a6000\tsteal-bal\t')"
  # A fixed item never moves: lane B empties while two fixed items wait behind lane A's first (held
  # until lane B's own item ran, then 3 s more), a 3090 one and a no-card one (device code with no
  # gate lock, which lands on the box env's 3090 pin), and takes neither.
  local slow_f='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=other,FAKE_GRACE=3'
  rc=$(steal_case steal-fix "$slow_f	A	3090	30	$d" "steal-fix	A	3090	30	$d" "steal-lib	A	3090	30	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_f" steal-fix steal-lib plain-any)
  printf '%s' "$rc" > "$t/rc-steal-fix"
  rc_ok 'steal: fixed items end green' steal-fix 4
  want_row 'steal: the fixed 3090 item stays in lane A on the 3090' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=3090/ && $2 ~ /gen_f$/'
  want 'steal: its run.log line names lane A' "$t/target/c-steal-fix/run.log" '^steal-fix rc=0 [0-9]+s try=1 lane=A( |$)'
  want 'steal: the fixed no-card item stays in lane A' "$t/target/c-steal-fix/run.log" '^steal-lib rc=0 [0-9]+s try=1 lane=A( |$)'
  want_no_row 'steal: no fixed call reached the A6000' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_f$/'
  # A held timing lease keeps every item off the A6000: the lease is up from the start, lane A's first
  # item holds its lane until the thief has probed (then 2 s more), and the owner — whose own items a
  # real box would hold in the guard — runs the balanced item on the 3090.
  local slow_l='steal-slow@FAKE_SLEEP=20,FAKE_UNTIL=probe,FAKE_GRACE=2'
  rc=$(LEASE_UP=1 steal_case steal-lease "$slow_l	A	3090	30	$d" "host	B	none	70	$d" "steal-bal	A	3090	60	$d" \
    "plain-any	B	a6000	10	$d" -- "$slow_l" host steal-bal plain-any)
  printf '%s' "$rc" > "$t/rc-steal-lease"
  rc_ok 'steal: a held timing lease ends green' steal-lease 4
  want 'steal: the thief probed the lease' "$t/fake-state/probe.log" '^probe$'
  want_row 'steal: the balanced item stayed off the A6000' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=3090/ && $2 ~ /gen_b$/'
  want_no_row 'steal: no call reached the A6000 under the lease' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_[bf]$/'
  # The dry run names the movable items (its own times rows, so the balance is fixed).
  printf '%s\n' 'steal-slow@FAKE_SLEEP=3	A	3090	30	2026-09-27T10:00:00+0900' \
    'host	B	none	70	2026-09-27T10:00:00+0900' 'steal-bal	A	3090	60	2026-09-27T10:00:00+0900' > "$t/times-dry.tsv"
  out=$(BLOOMERY_GATE_TIMES=$t/times-dry.tsv "${gb[@]}" --dry-run 'steal-slow@FAKE_SLEEP=3' host steal-bal 2>&1) \
    || fail 'steal: the dry run of a movable pair failed' "$out"
  if grep -A3 '^lane A  steal-bal' <<< "$out" | grep -q 'movable'; then pass 'steal: the dry run names the balanced item movable'
  else fail 'steal: the dry run names the balanced item movable' "no movable line under steal-bal"; fi
  if grep -A3 '^lane A  steal-slow' <<< "$out" | grep -q 'movable'; then fail 'steal: the dry run leaves a fixed item unmarked' "a movable line under steal-slow"
  else pass 'steal: the dry run leaves a fixed item unmarked'; fi
  # The chain (class C). A mutant copy per rule: the sed applies (its line greps the copy), the
  # batch runs against the mutant and the defect shows, then the same batch runs green here.
  mutant_of() { # <tag> <sed expr> <ERE proving the sed applied>
    local tag=$1 expr=$2 proof=$3
    n=$((n + 1))
    sed "$expr" "$t/tools/gate-batch.sh" > "$t/tools/gate-batch-$tag.sh"
    if grep -Eq -- "$proof" "$t/tools/gate-batch-$tag.sh"; then echo "mutant $tag: $expr"
    else bad=$((bad + 1)); echo "FAIL mutant $tag: the sed did not apply: $expr"; fi
  }
  # no_overlap <ran.log> <cmd-a> <cmd-b>: the two items' run intervals do not overlap; a batch's
  # chain members run one at a time, so their intervals never cross. A command with no line in the log
  # (or no log) is a FAIL line and status 2, never a pass or a defined answer for the pair.
  no_overlap() {
    local arc=0
    awk -v a="$2" -v b="$3" '
      $0 ~ (" " a "[ ]*$") { as = $1; ae = $2 }
      $0 ~ (" " b "[ ]*$") { bs = $1; be = $2 }
      END { if (as == "" || bs == "") exit 2; exit !(as < bs ? ae <= bs : be <= as) }' "$1" || arc=$?
    [ "$arc" != 2 ] || { n=$((n + 1)) bad=$((bad + 1)); echo "FAIL ${FUNCNAME[0]}: '$2' or '$3' has no line in $1"; }
    return "$arc"
  }
  # ran_before <ran.log> <cmd-a> <cmd-b>: prints yes (a ended before b started), no, or absent (either never
  # ran, or there is no log); a mutant twin passes on `no` only.
  ran_before() {
    awk -v a="$2" -v b="$3" '
      $0 ~ (" " a "[ ]*$") { ae = $2 }
      $0 ~ (" " b "[ ]*$") { bs = $1 }
      END { print (ae == "" || bs == "" ? "absent" : ae <= bs ? "yes" : "no") }' "$1" || echo absent
  }
  # (1) chain exclusion: two any-form members and both lanes free — the mutex keeps them one at a
  # time, whichever lane takes each.
  rc=$(steal_case ch-excl "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	B	a6000	5	$d" "gate-gpu-glm5next-e2e	B	a6000	5	$d" \
    -- steal-fix 'plain-any@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=5' 'gate-gpu-glm5next-e2e@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-excl"
  rc_ok 'chain: two any-form members end green' ch-excl 4
  if no_overlap "$t/fake-state/ran.log" gen_x gen_ge; then pass 'chain: no two members overlap (the mutex)'
  else fail 'chain: no two members overlap (the mutex)' "$(awk '$0 ~ /gen_x$|gen_ge$/' "$t/fake-state/ran.log" | tr '\n' ' ')"; fi
  mutant_of ch-nomutex 's|^  mkdir "$CHAIN_MUTEX" 2> /dev/null \|\| return 1$|  true # MUTANT ch-nomutex|' '# MUTANT ch-nomutex$'
  rc=$(MTAG=ch-nomutex steal_case ch-exclm "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	B	a6000	5	$d" "gate-gpu-glm5next-e2e	B	a6000	5	$d" \
    -- steal-fix 'plain-any@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=5' 'gate-gpu-glm5next-e2e@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-exclm"
  if no_overlap "$t/fake-state/ran.log" gen_x gen_ge; then fail 'chain mutant: without the mutex the members overlap' "the intervals did not cross"
  else pass 'chain mutant: without the mutex the members overlap'; fi
  # (2) a lane-B chain run: the pick member (A6000) holds lane B first, so lane A opens its own slow
  # item; when the pick ends lane B takes the any-form member at once, on the A6000 — its row, its
  # key and its call name that card.
  rc=$(steal_case ch-laneb "steal-slow	A	3090	15	$d" "plain-any	B	a6000	5	$d" "gate-gpu-ds41-residency-a	B	a6000	3	$d" "v41-any	A	3090	2	$d" \
    -- 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=2' x-q)
  printf '%s' "$rc" > "$t/rc-ch-laneb"
  rc_ok 'chain: a lane-B chain run ends green' ch-laneb 5
  want 'chain: the any-form member ran in lane B' "$t/target/c-ch-laneb/run.log" '^v41-any rc=0 [0-9]+s try=1 lane=B( |$)'
  want 'chain: its times row names lane B and the A6000' "$t/times-ch-laneb.tsv" "$(printf '^v41-any@FAKE_SLEEP=2\tB\ta6000\t')"
  want 'chain: the ledger records it under the A6000 key' "$t/rounds-ch-laneb.tsv" "$(printf '^stub-v41-any@FAKE_SLEEP=2,BLOOMERY_GATE_CARD=a6000\tv41-any\t')"
  want_row 'chain: its box call carries the A6000 card' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_x$/'
  mutant_of ch-pina 's|^  local lane=\$1 i card cards$|  local lane=$1 i card cards; [ "$lane" = B ] \&\& return 1 # MUTANT ch-pina|' '# MUTANT ch-pina$'
  rm -rf "$t/fake-state" "$t/target/c-ch-lanebm"
  BLOOMERY_GATE_TIMES=$t/times-ch-laneb.tsv BLOOMERY_GATE_LEDGER=$t/lead-ch-lanebm.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-ch-lanebm.tsv \
    bash "$t/tools/gate-batch-ch-pina.sh" --out "$t/target/c-ch-lanebm" --round-ledger 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=2' x-q > "$t/out-ch-lanebm.log" 2>&1 &
  sp=$!
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  if kill -0 "$sp" 2> /dev/null; then hung=1; else hung=0; fi
  kill -TERM "$sp" 2> /dev/null || true
  wait "$sp" 2> /dev/null || true
  for f in "$t"/target/c-ch-lanebm/lane-*.pid "$t"/target/c-ch-lanebm/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  n=$((n + 1))
  if [ "$hung" = 1 ] && ! grep -q '^v41-any rc=' "$t/target/c-ch-lanebm/run.log" 2> /dev/null; then
    echo 'ok chain mutant: members pinned to lane A never reach lane B (the batch waits without end)'
  else
    bad=$((bad + 1)); echo "FAIL chain mutant: members pinned to lane A never reach lane B (the batch waits without end): $(grep '^v41-any ' "$t/target/c-ch-lanebm/run.log" 2>&1 | tr '\n' ' ')"
  fi
  # (3) the card rule: the pick member goes to lane B, the 3090-only one waits for lane A; no card
  # of the batch reaches the pick member (box.sh picks).
  rc=$(steal_case ch-cards "steal-slow	A	3090	15	$d" "plain-any	B	a6000	5	$d" "gate-gpu-ds41-residency-a	B	a6000	2	$d" "v41-a	A	3090	4	$d" \
    -- 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=4' x-q)
  printf '%s' "$rc" > "$t/rc-ch-cards"
  rc_ok 'chain: a 3090-only and an A6000-pick member end green' ch-cards 5
  want 'chain: the 3090-only member runs in lane A' "$t/target/c-ch-cards/run.log" '^v41-a rc=0 [0-9]+s try=1 lane=A( |$)'
  want 'chain: the A6000-pick member runs in lane B' "$t/target/c-ch-cards/run.log" '^gate-gpu-ds41-residency-a rc=0 [0-9]+s try=1 lane=B( |$)'
  want_no_row 'chain: no card of the batch reached the pick member' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD/ && $2 ~ /gen_ra/'
  mutant_of ch-anywant 's|^  want=$(lane_card "$1")$|  want=3090 # MUTANT ch-anywant|' '# MUTANT ch-anywant$'
  rm -rf "$t/fake-state" "$t/target/c-ch-cardsm"
  BLOOMERY_GATE_TIMES=$t/times-ch-cards.tsv BLOOMERY_GATE_LEDGER=$t/lead-ch-cardsm.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-ch-cardsm.tsv \
    bash "$t/tools/gate-batch-ch-anywant.sh" --out "$t/target/c-ch-cardsm" --round-ledger 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=4' x-q > "$t/out-ch-cardsm.log" 2>&1 &
  sp=$!
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 25 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$sp" 2> /dev/null || true
  wait "$sp" 2> /dev/null || true
  for f in "$t"/target/c-ch-cardsm/lane-*.pid "$t"/target/c-ch-cardsm/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  want_not 'chain mutant: every card reading 3090 lets a lane take a member it may not' "$t/target/c-ch-cardsm/run.log" '^v41-a rc=0 [0-9]+s try=1 lane=A( |$)'
  # (4) solo and both-cards members stay X, and a solo-real the pass list does not name stays X in
  # the real tier and balanced in the fixture tier — the tier cases below hold all of that.
  # (5) the chain's family order: lane X's first family (deepseek41, of x-v41) goes last, the
  # other families in the reverse of X's family order — so v41-any and v41-a (deepseek2) run
  # before glm5next's member although the list names it first, and deepseek41's member (gen_da)
  # ends the chain.
  rc=$(steal_case ch-order "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	A	3090	2	$d" "gate-gpu-glm5next-e2e	B	a6000	2	$d" "v41-a	A	3090	2	$d" "v41-d41-a	A	3090	2	$d" \
    -- steal-fix 'plain-any@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=2' 'gate-gpu-glm5next-e2e@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=2' 'v41-d41-a@FAKE_SLEEP=2' x-v41)
  printf '%s' "$rc" > "$t/rc-ch-order"
  rc_ok 'chain: a two-family chain with lane X ends green' ch-order 7
  if [ "$(ran_before "$t/fake-state/ran.log" gen_y gen_ge)" = yes ] && [ "$(ran_before "$t/fake-state/ran.log" gen_ge gen_da)" = yes ]; then
    pass 'chain: the families group (X'"'"'s first family last, the record order inside one only)'
  else
    fail 'chain: the families group (X'"'"'s first family last, the record order inside one only)' "$(grep -E 'gen_y$|gen_ge$|gen_da$' "$t/fake-state/ran.log" | tr '\n' ' ')"
  fi
  mutant_of ch-recorder 's|key=lambda k: (fam_order.index(fams\[k\]), k))|key=lambda k: k)  # MUTANT ch-recorder|' '^cs = sorted\(ck, key=lambda k: k\)  # MUTANT ch-recorder$'
  rc=$(MTAG=ch-recorder steal_case ch-orderm "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	A	3090	2	$d" "gate-gpu-glm5next-e2e	B	a6000	2	$d" "v41-a	A	3090	2	$d" "v41-d41-a	A	3090	2	$d" \
    -- steal-fix 'plain-any@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=2' 'gate-gpu-glm5next-e2e@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=2' 'v41-d41-a@FAKE_SLEEP=2' x-v41)
  printf '%s' "$rc" > "$t/rc-ch-orderm"
  if [ "$(ran_before "$t/fake-state/ran.log" gen_y gen_ge)" = no ]; then pass 'chain mutant: record order interleaves the families'
  else fail 'chain mutant: record order interleaves the families' "gen_y still ran before gen_ge, or a member never ran: $(grep -E 'gen_y$|gen_ge$' "$t/fake-state/ran.log" 2>&1 | tr '\n' ' ')"; fi
  # The family gate: the chain's first member is the pick (A6000, deepseek41), the second the
  # 3090-only one of the same family; lane A, busy on its own item once the pick ends, must not
  # open deepseek2's member (v41-any) while the 3090-only one is unclaimed — it waits for lane B.
  rc=$(steal_case ch-famgate "steal-slow	A	3090	15	$d" "plain-any	B	a6000	5	$d" "gate-gpu-ds41-residency-a	B	a6000	2	$d" "v41-d41-a	A	3090	2	$d" "v41-any	A	3090	2	$d" \
    -- 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=2' 'v41-d41-a@FAKE_SLEEP=2' 'v41-any@FAKE_SLEEP=2' x-q)
  printf '%s' "$rc" > "$t/rc-ch-famgate"
  rc_ok 'chain: a family held by one-card members ends green' ch-famgate 6
  if [ "$(ran_before "$t/fake-state/ran.log" 'gen_ra --place a' gen_x)" = yes ]; then pass 'chain: a lane does not open the next family while the current one holds an unclaimed member'
  else fail 'chain: a lane does not open the next family while the current one holds an unclaimed member' "gen_x started before gen_ra ended, or a member never ran: $(grep -E 'gen_ra|gen_x$' "$t/fake-state/ran.log" 2>&1 | tr '\n' ' ')"; fi
  # The family gate's runtime half has no deterministic mutant at the fixture's 1 s granularity
  # (the mutex race decides which lane meets the boundary); the order mutant ch-recorder above is
  # the family rule's FAIL-first, and the good case above pins the runtime wait.
  # (6) a pool item never overlaps a member; an untagged one runs beside one.
  rc=$(steal_case ch-pool "steal-fix	A	3090	5	$d" "pool-any	B	a6000	3	$d" "plain-any	B	a6000	3	$d" "v41-any	A	3090	5	$d" \
    -- steal-fix 'pool-any@FAKE_SLEEP=3' 'plain-any@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-pool"
  rc_ok 'chain: a pool item and a plain one end green' ch-pool 4
  if no_overlap "$t/fake-state/ran.log" gen_x gen_pl; then pass 'chain: a pool item never overlaps a member'
  else fail 'chain: a pool item never overlaps a member' "the intervals crossed"; fi
  if no_overlap "$t/fake-state/ran.log" gen_x other; then fail 'chain: an untagged item runs beside a member'
  else pass 'chain: an untagged item runs beside a member' "plain-any never overlapped gen_x"; fi
  mutant_of ch-nopool 's|^tagged_neighbour() { #.*$|tagged_neighbour() { return 1 # MUTANT ch-nopool|' '# MUTANT ch-nopool$'
  rc=$(MTAG=ch-nopool steal_case ch-poolm "steal-fix	A	3090	5	$d" "pool-any	B	a6000	3	$d" "plain-any	B	a6000	3	$d" "v41-any	A	3090	5	$d" \
    -- steal-fix 'pool-any@FAKE_SLEEP=3' 'plain-any@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-poolm"
  if no_overlap "$t/fake-state/ran.log" gen_x gen_pl; then fail 'chain mutant: without the pool rule the item overlaps a member' "the intervals did not cross"
  else pass 'chain mutant: without the pool rule the item overlaps a member'; fi
  # (7) a big-host item never overlaps a member.
  rc=$(steal_case ch-big "steal-fix	A	3090	5	$d" "big-any	B	a6000	3	$d" "v41-any	A	3090	5	$d" \
    -- steal-fix 'big-any@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-big"
  rc_ok 'chain: a big-host item ends green' ch-big 3
  if no_overlap "$t/fake-state/ran.log" gen_x gen_bh; then pass 'chain: a big-host item never overlaps a member'
  else fail 'chain: a big-host item never overlaps a member' "the intervals crossed"; fi
  rc=$(MTAG=ch-nopool steal_case ch-bigm "steal-fix	A	3090	5	$d" "big-any	B	a6000	3	$d" "v41-any	A	3090	5	$d" \
    -- steal-fix 'big-any@FAKE_SLEEP=3' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-bigm"
  if no_overlap "$t/fake-state/ran.log" gen_x gen_bh; then fail 'chain mutant: without the rule a big-host item overlaps a member' "the intervals did not cross"
  else pass 'chain mutant: without the rule a big-host item overlaps a member'; fi
  # (8) the gap: the pick member (A6000) runs first; the 3090-only member waits for lane A, inside
  # its own 15 s item — and while it is unclaimed the big-host item must not start in the gap,
  # though the mutex is free and lane B has nothing else.
  rc=$(steal_case ch-gap "steal-slow	A	3090	15	$d" "plain-any	B	a6000	1	$d" "big-any	B	a6000	3	$d" "gate-gpu-ds41-residency-a	B	a6000	2	$d" "v41-a	A	3090	2	$d" \
    -- 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'big-any@FAKE_SLEEP=3' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=2' x-q)
  printf '%s' "$rc" > "$t/rc-ch-gap"
  rc_ok 'chain: a gap between two members ends green' ch-gap 6
  if [ "$(ran_before "$t/fake-state/ran.log" gen_y gen_bh)" = yes ]; then pass 'chain: a big-host item ready in the gap does not start there'
  else fail 'chain: a big-host item ready in the gap does not start there' "gen_bh started before gen_y ended"; fi
  mutant_of ch-noclaim 's|^    claimed "\$i" \|\| return 1$|    true # MUTANT ch-noclaim|' '# MUTANT ch-noclaim$'
  rc=$(MTAG=ch-noclaim steal_case ch-gapm "steal-slow	A	3090	15	$d" "plain-any	B	a6000	1	$d" "big-any	B	a6000	3	$d" "gate-gpu-ds41-residency-a	B	a6000	2	$d" "v41-a	A	3090	2	$d" \
    -- 'steal-slow@FAKE_SLEEP=15' 'plain-any@FAKE_SLEEP=1' 'big-any@FAKE_SLEEP=3' 'gate-gpu-ds41-residency-a@FAKE_SLEEP=2' 'v41-a@FAKE_SLEEP=2' x-q)
  printf '%s' "$rc" > "$t/rc-ch-gapm"
  if [ "$(ran_before "$t/fake-state/ran.log" gen_y gen_bh)" = no ]; then pass 'chain mutant: without the unstarted half the item starts in the gap'
  else fail 'chain mutant: without the unstarted half the item starts in the gap' "gen_bh still waited, or an item never ran: $(grep -E 'gen_y$|gen_bh$' "$t/fake-state/ran.log" 2>&1 | tr '\n' ' ')"; fi
  # (9) a ledger skip: green on the 3090's key only, the member skips there and never runs on the
  # A6000. The stub's FAKE_GREEN_FILE names the one key.
  printf 'stub-v41-any@FAKE_SLEEP=2,BLOOMERY_GATE_CARD=a6000\n' > "$t/green-a6000.tsv"
  rc=$(FAKE_GREEN_FILE=$t/green-a6000.tsv steal_case ch-skip "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	A	3090	2	$d" \
    -- steal-fix plain-any 'v41-any@FAKE_SLEEP=2')
  printf '%s' "$rc" > "$t/rc-ch-skip"
  rc_ok 'chain: a member green on one key ends green' ch-skip 3
  want 'chain: it skips at 0 s, at its green key' "$t/target/c-ch-skip/run.log" '^v41-any rc=skip green-at=abc1234 2026-10-08T00:00:00\+0900 lane=C( |$)'
  want_no_row 'chain: and nothing ran on the 3090' "$t/fake-state/box.log" '$2 ~ /gen_x$/'
  mutant_of ch-noskip 's|green = \[j for j, s in enumerate(states) if s == "skip"\]|green = [] # MUTANT ch-noskip|' '# MUTANT ch-noskip$'
  rc=$(FAKE_GREEN_FILE=$t/green-a6000.tsv MTAG=ch-noskip steal_case ch-skipm "steal-fix	A	3090	5	$d" "plain-any	B	a6000	5	$d" "v41-any	A	3090	2	$d" \
    -- steal-fix plain-any 'v41-any@FAKE_SLEEP=2')
  printf '%s' "$rc" > "$t/rc-ch-skipm"
  want_not 'chain mutant: without the skip placement the member runs although it is green' "$t/target/c-ch-skipm/run.log" '^v41-any rc=skip '
  # (10) a lane killed while it holds the mutex: the batch ends by name inside a bound, never a
  # wait without end. The deaf-watch mutant hangs instead and the case kills it.
  rm -rf "$t/fake-state" "$t/target/c-ch-kill"
  BLOOMERY_GATE_TIMES=$t/times.tsv BLOOMERY_GATE_LEDGER=$t/lead-ch-kill.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-ch-kill.tsv \
    "${gb[@]}" --out "$t/target/c-ch-kill" --round-ledger 'steal-slow@FAKE_SLEEP=15' 'v41-a@FAKE_SLEEP=20' 'big-any@FAKE_SLEEP=1' > "$t/out-ch-kill.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_y' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$(cat "$t/target/c-ch-kill/lane-A.pid")"
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 30 ]; do
    sleep 1
    k=$((k + 1))
  done
  if ! kill -0 "$sp" 2> /dev/null; then
    wait "$sp" 2> /dev/null || true
    if grep -q 'lane A ended without its sentinel' "$t/out-ch-kill.log"; then pass 'chain: a lane killed with the mutex ends the batch by name'
    else fail 'chain: a lane killed with the mutex ends the batch by name' "$(tail -2 "$t/out-ch-kill.log")"; fi
  else
    kill -TERM "$sp" 2> /dev/null || true
    wait "$sp" 2> /dev/null || true
    fail 'chain: a lane killed with the mutex ends the batch by name' "the batch was still running after ${k} s"
  fi
  for f in "$t"/target/c-ch-kill/lane-*.pid "$t"/target/c-ch-kill/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  mutant_of ch-deaf "s|^      \*) dead_now=\"\\\$dead_now \\\$lane\"; continue ;;\$|      *) continue ;; # MUTANT ch-deaf|" '# MUTANT ch-deaf$'
  rm -rf "$t/fake-state" "$t/target/c-ch-killm"
  BLOOMERY_GATE_TIMES=$t/times.tsv BLOOMERY_GATE_LEDGER=$t/lead-ch-kill.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-ch-kill.tsv \
    bash "$t/tools/gate-batch-ch-deaf.sh" --out "$t/target/c-ch-killm" --round-ledger 'steal-slow@FAKE_SLEEP=15' 'v41-a@FAKE_SLEEP=20' 'big-any@FAKE_SLEEP=1' > "$t/out-ch-killm.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_y' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$(cat "$t/target/c-ch-killm/lane-A.pid")"
  k=0
  hung=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 15 ]; do
    sleep 1
    k=$((k + 1))
  done
  if kill -0 "$sp" 2> /dev/null; then hung=1; fi
  kill -TERM "$sp" 2> /dev/null || true
  wait "$sp" 2> /dev/null || true
  for f in "$t"/target/c-ch-killm/lane-*.pid "$t"/target/c-ch-killm/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  if [ "$hung" = 1 ]; then pass 'chain mutant: a deaf watch leaves the batch waiting without end'
  else fail 'chain mutant: a deaf watch leaves the batch waiting without end' "the mutant batch ended within ${k} s"; fi
  # (10b) both lanes killed at once: each lane's sighting is its own, so the batch still ends by
  # name inside the bound. A watch that kept one list of sightings for both lanes would see them
  # alternate and never twice in a row, and wait without end.
  rm -rf "$t/fake-state" "$t/target/c-ch-kill2"
  BLOOMERY_GATE_TIMES=$t/times.tsv BLOOMERY_GATE_LEDGER=$t/lead-ch-kill2.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-ch-kill2.tsv \
    "${gb[@]}" --out "$t/target/c-ch-kill2" --round-ledger 'steal-slow@FAKE_SLEEP=15' 'v41-a@FAKE_SLEEP=20' 'big-any@FAKE_SLEEP=1' > "$t/out-ch-kill2.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_y' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$(cat "$t/target/c-ch-kill2/lane-A.pid")" "$(cat "$t/target/c-ch-kill2/lane-B.pid")"
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 30 ]; do
    sleep 1
    k=$((k + 1))
  done
  if ! kill -0 "$sp" 2> /dev/null; then
    wait "$sp" 2> /dev/null || true
    if grep -Eq 'lane [AB] ended without its sentinel' "$t/out-ch-kill2.log"; then pass 'chain: both lanes killed at once end the batch by name'
    else fail 'chain: both lanes killed at once end the batch by name' "$(tail -2 "$t/out-ch-kill2.log")"; fi
  else
    kill -TERM "$sp" 2> /dev/null || true
    wait "$sp" 2> /dev/null || true
    fail 'chain: both lanes killed at once end the batch by name' "the batch was still running after ${k} s"
  fi
  for f in "$t"/target/c-ch-kill2/lane-*.pid "$t"/target/c-ch-kill2/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  # (11) the predicted line carries chain= and its phase-1 term.
  check 'chain: the predicted line carries chain= and the phase-1 derivation' 0 \
    'predicted laneA=[0-9]+s laneB=[0-9]+s laneX=0s chain=100s wall=[0-9]+s.*phase 1 = max\(the chain' \
    "${gb[@]}" --dry-run v41-a
  mutant_of ch-nopred 's| chain=\${SUM_C}s||' 'laneX=\$\{SUM_X\}s wall='
  out=$(bash "$t/tools/gate-batch-ch-nopred.sh" --dry-run v41-a 2>&1) || true
  if grep -q 'chain=' <<< "$out"; then fail 'chain mutant: the predicted line without chain=' "chain= is still there"
  else pass 'chain mutant: the predicted line without chain='; fi
  # The take (the free lane's longest-est choice): the balanced item (60 s) goes before the short
  # no-card one (5 s), on whichever lane is free beside the chain's member. The member outlasts the
  # balanced item, so whichever lane the mutex race gives it, the short item can start only after the
  # balanced one ended (the other lane takes the fixed 3090 item first, the longer of the two it may
  # take); the times rows are keyed by the items as written.
  rc=$(steal_case ch-lpt "steal-slow@FAKE_SLEEP=3	A	3090	15	$d" "steal-bal@FAKE_SLEEP=1	A	3090	60	$d" "host@FAKE_SLEEP=1	B	none	5	$d" \
    "v41-any@FAKE_SLEEP=5	B	a6000	1	$d" -- 'steal-slow@FAKE_SLEEP=3' 'steal-bal@FAKE_SLEEP=1' 'host@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-lpt"
  rc_ok 'chain: a longest-first take ends green' ch-lpt 4
  if [ "$(ran_before "$t/fake-state/ran.log" gen_b 'bash tools/gate.sh -p x')" = yes ]; then pass 'chain: the free lane takes the longest-est item first'
  else fail 'chain: the free lane takes the longest-est item first' "$(grep -E 'gen_b|gate.sh -p x' "$t/fake-state/ran.log" | tr '\n' ' ')"; fi
  # The twin: the take reversed (the shortest-est item first) starts the short item at once, so the balanced
  # one cannot have ended before it.
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  mutant_of ch-lptrev 's|^    if \[ "\$exp" -gt "\$be" \]; then best=\$j be=\$exp; fi$|    if [ "$be" -lt 0 ] \|\| [ "$exp" -lt "$be" ]; then best=$j be=$exp; fi # MUTANT ch-lptrev|' '# MUTANT ch-lptrev$'
  rc=$(MTAG=ch-lptrev steal_case ch-lptm "steal-slow@FAKE_SLEEP=3	A	3090	15	$d" "steal-bal@FAKE_SLEEP=1	A	3090	60	$d" "host@FAKE_SLEEP=1	B	none	5	$d" \
    "v41-any@FAKE_SLEEP=5	B	a6000	1	$d" -- 'steal-slow@FAKE_SLEEP=3' 'steal-bal@FAKE_SLEEP=1' 'host@FAKE_SLEEP=1' 'v41-any@FAKE_SLEEP=5')
  printf '%s' "$rc" > "$t/rc-ch-lptm"
  if [ "$(ran_before "$t/fake-state/ran.log" gen_b 'bash tools/gate.sh -p x')" = no ]; then pass 'chain mutant: a reversed take starts the short item first'
  else fail 'chain mutant: a reversed take still starts the longest-est item first, or an item never ran' "$(grep -E 'gen_b|gate.sh -p x' "$t/fake-state/ran.log" 2>&1 | tr '\n' ' ')"; fi
  # The take's card and lease semantics are the steal cases' above and below.
  # The pack (the header's «pack»). overlap is no_overlap's inverse: the two items' intervals cross (a command
  # with no line is a FAIL line and status 2, as there).
  overlap() {
    local arc=0
    awk -v a="$2" -v b="$3" '
      $0 ~ (" " a "[ ]*$") { as = $1; ae = $2 }
      $0 ~ (" " b "[ ]*$") { bs = $1; be = $2 }
      END { if (as == "" || bs == "") exit 2; exit !(as < bs ? ae > bs : be > as) }' "$1" || arc=$?
    [ "$arc" != 2 ] || { n=$((n + 1)) bad=$((bad + 1)); echo "FAIL ${FUNCNAME[0]}: '$2' or '$3' has no line in $1"; }
    return "$arc"
  }
  ivals() { # <ran.log> <cmd-a> <cmd-b>: the two items' ran.log lines (start, end, command), a red's detail
    awk -v a="$2" -v b="$3" '$0 ~ (" " a "[ ]*$") || $0 ~ (" " b "[ ]*$")' "$1" | tr '\n' ' '
  }
  # The times rows are keyed by the item as written (its @FAKE_SLEEP env included), so each case's
  # expected seconds, and the longest-est launch order the case names, are the rows' own.
  # (1) two one-card items whose bytes fit run side by side, one a card; the second take names the
  # free card and its row, key and call follow it.
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-pair "pk-a@FAKE_SLEEP=5	A	3090	5	$d" "pk-b@FAKE_SLEEP=5	B	a6000	5	$d" \
    -- pk-a@FAKE_SLEEP=5 pk-b@FAKE_SLEEP=5)
  printf '%s' "$rc" > "$t/rc-pk-pair"
  rc_ok 'pack: two one-card items that fit end green' pk-pair 2
  if overlap "$t/fake-state/ran.log" gen_pa gen_pb; then pass 'pack: the two items ran side by side, one a card'
  else fail 'pack: the two items ran side by side, one a card' "$(awk '$0 ~ /gen_p[ab]$/' "$t/fake-state/ran.log" | tr '\n' ' ')"; fi
  want_row 'pack: the second take ran on the free card (the 3090)' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=3090/ && $2 ~ /gen_p[ab]$/'
  want_row 'pack: both rows name lane X and the card each took' "$t/times-pk-pair.tsv" '$2 == "X" && ($3 == "3090" || $3 == "a6000") { c++ } END { if (c == 2) print c }'
  # (2) a pair over the budget never overlaps; without the check it does.
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-budget "pk-big@FAKE_SLEEP=3	A	3090	3	$d" "pk-big2@FAKE_SLEEP=3	B	a6000	3	$d" \
    -- pk-big@FAKE_SLEEP=3 pk-big2@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-budget"
  rc_ok 'pack: a pair over the budget ends green (serially)' pk-budget 2
  if no_overlap "$t/fake-state/ran.log" gen_pg gen_pg2; then pass 'pack: the pair over the budget never overlapped'
  else fail 'pack: the pair over the budget never overlapped' "the intervals crossed: $(ivals "$t/fake-state/ran.log" gen_pg gen_pg2)"; fi
  mutant_of pk-nobudget 's|^      sumb=\$((sumb + R_BYTES\[i\]))$|      sumb=0 # MUTANT pk-nobudget|' '# MUTANT pk-nobudget$'
  rc=$(BLOOMERY_PACK_BUDGET=10000 MTAG=pk-nobudget steal_case pk-budgetm "pk-big@FAKE_SLEEP=3	A	3090	3	$d" "pk-big2@FAKE_SLEEP=3	B	a6000	3	$d" \
    -- pk-big@FAKE_SLEEP=3 pk-big2@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-budgetm"
  if overlap "$t/fake-state/ran.log" gen_pg gen_pg2; then pass 'pack mutant: without the budget the pair overlaps'
  else fail 'pack mutant: without the budget the pair overlaps' "the intervals did not cross"; fi
  # (3) an m item runs with nothing else beside it, first (it is the longest) or after a running item.
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-m1 "pk-m@FAKE_SLEEP=5	A	3090	5	$d" "pk-a@FAKE_SLEEP=3	B	a6000	3	$d" \
    -- pk-m@FAKE_SLEEP=5 pk-a@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-m1"
  rc_ok 'pack: an m item first ends green' pk-m1 2
  if no_overlap "$t/fake-state/ran.log" gen_pm gen_pa; then pass 'pack: nothing ran beside the m item'
  else fail 'pack: nothing ran beside the m item' "the intervals crossed: $(ivals "$t/fake-state/ran.log" gen_pm gen_pa)"; fi
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-m2 "pk-m@FAKE_SLEEP=3	B	a6000	3	$d" "pk-a@FAKE_SLEEP=5	A	3090	5	$d" \
    -- pk-m@FAKE_SLEEP=3 pk-a@FAKE_SLEEP=5)
  printf '%s' "$rc" > "$t/rc-pk-m2"
  rc_ok 'pack: an m item behind a longer one ends green' pk-m2 2
  if no_overlap "$t/fake-state/ran.log" gen_pm gen_pa; then pass 'pack: the m item waited for the host alone'
  else fail 'pack: the m item waited for the host alone' "the intervals crossed: $(ivals "$t/fake-state/ran.log" gen_pm gen_pa)"; fi
  mutant_of pk-nom 's|^          \[ "\$mrun" = 0 \] \|\| continue # and nothing starts beside one$|          true # MUTANT pk-nom|' '# MUTANT pk-nom$'
  rc=$(BLOOMERY_PACK_BUDGET=10000 MTAG=pk-nom steal_case pk-m1m "pk-m@FAKE_SLEEP=5	A	3090	5	$d" "pk-a@FAKE_SLEEP=3	B	a6000	3	$d" \
    -- pk-m@FAKE_SLEEP=5 pk-a@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-m1m"
  if overlap "$t/fake-state/ran.log" gen_pm gen_pa; then pass 'pack mutant: without the m rule an item runs beside it'
  else fail 'pack mutant: without the m rule an item runs beside it' "the intervals did not cross"; fi
  mutant_of pk-nomw 's|^          \[ "\$nrun" = 0 \] \|\| continue # an m item starts with nothing else running$|          true # MUTANT pk-nomw|' '# MUTANT pk-nomw$'
  rc=$(BLOOMERY_PACK_BUDGET=10000 MTAG=pk-nomw steal_case pk-m2m "pk-m@FAKE_SLEEP=3	B	a6000	3	$d" "pk-a@FAKE_SLEEP=5	A	3090	5	$d" \
    -- pk-m@FAKE_SLEEP=3 pk-a@FAKE_SLEEP=5)
  printf '%s' "$rc" > "$t/rc-pk-m2m"
  if overlap "$t/fake-state/ran.log" gen_pm gen_pa; then pass 'pack mutant: without the wait the m item starts beside a running one'
  else fail 'pack mutant: without the wait the m item starts beside a running one' "the intervals did not cross: $(ivals "$t/fake-state/ran.log" gen_pm gen_pa)"; fi
  # (4) a both-cards item runs alone (both cards busy for the pack).
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-bothc "pk-both@FAKE_SLEEP=4	X	both	4	$d" "pk-a@FAKE_SLEEP=3	B	a6000	3	$d" \
    -- pk-both@FAKE_SLEEP=4 pk-a@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-bothc"
  rc_ok 'pack: a both-cards item ends green' pk-bothc 2
  if no_overlap "$t/fake-state/ran.log" gen_pt gen_pa; then pass 'pack: a both-cards item ran alone'
  else fail 'pack: a both-cards item ran alone' "the intervals crossed: $(ivals "$t/fake-state/ran.log" gen_pt gen_pa)"; fi
  # (4b) a box-pick item (its recipe names the card: the plan's cards field is `-`, its label the card) takes no
  # card from the pack: its call carries no BLOOMERY_GATE_CARD, so its key holds and its line ends recorded. The
  # twin is the guard removed — the pack switches whatever card its label names, the old behaviour.
  rc=$(BLOOMERY_PACK_BUDGET=10000 steal_case pk-pick "sr-a6000@FAKE_SLEEP=2	B	a6000	2	$d" -- sr-a6000@FAKE_SLEEP=2)
  printf '%s' "$rc" > "$t/rc-pk-pick"
  rc_ok 'pack: a box-pick item ends green' pk-pick 1
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  want_row 'pack: the box-pick item ran' "$t/fake-state/box.log" '$2 ~ /gen_sa/'
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  want_no_row 'pack: the box-pick item took no card env from the pack' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=/ && $2 ~ /gen_sa/'
  want 'pack: the box-pick item'"'"'s line ends ledger=recorded' "$t/target/c-pk-pick/run.log" '^sr-a6000 rc=0 [0-9]+s try=1 lane=X ledger=recorded( |$)'
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  mutant_of pk-noguard 's|^      case " \$cards " in \*" \$card "\*) switch_candidate "\$i" "\$card" ;; esac$|      case $card in 3090 \| a6000) switch_candidate "$i" "$card" ;; esac # MUTANT pk-noguard|' '# MUTANT pk-noguard$'
  rc=$(BLOOMERY_PACK_BUDGET=10000 MTAG=pk-noguard steal_case pk-pickm "sr-a6000@FAKE_SLEEP=2	B	a6000	2	$d" -- sr-a6000@FAKE_SLEEP=2)
  printf '%s' "$rc" > "$t/rc-pk-pickm"
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  want_row 'pack mutant: without the guard the box-pick item takes a card env' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_GATE_CARD=a6000/ && $2 ~ /gen_sa/'
  want 'pack mutant: … and its line ends ledger=changed' "$t/target/c-pk-pickm/run.log" '^sr-a6000 rc=0 [0-9]+s try=1 lane=X ledger=changed( |$)'
  # (5) the predicted line's laneX= is the pack's own wall (two fits beside each other, a pair over
  # the budget serial); the dry run names each item's simulated take.
  printf '%s\n' "pk-a	A	3090	5	$d" "pk-b	B	a6000	5	$d" > "$t/times-pkpred.tsv"
  check 'pack: the predicted laneX= is the pack wall (a fitting pair)' 0 \
    'predicted laneA=0s laneB=0s laneX=5s chain=0s wall=5s' env BLOOMERY_GATE_TIMES=$t/times-pkpred.tsv BLOOMERY_PACK_BUDGET=10000 "${gb[@]}" --dry-run pk-a pk-b
  printf '%s\n' "pk-big	A	3090	3	$d" "pk-big2	B	a6000	3	$d" > "$t/times-pkpred2.tsv"
  check 'pack: … and a pair over the budget serializes (laneX=6s)' 0 \
    'predicted laneA=0s laneB=0s laneX=6s chain=0s wall=6s' env BLOOMERY_GATE_TIMES=$t/times-pkpred2.tsv BLOOMERY_PACK_BUDGET=10000 "${gb[@]}" --dry-run pk-big pk-big2
  out=$(env BLOOMERY_GATE_TIMES=$t/times-pkpred.tsv BLOOMERY_PACK_BUDGET=10000 "${gb[@]}" --dry-run pk-a pk-b 2>&1) || fail 'pack: the dry run names the takes' "$out"
  if [ "$(grep -c 'the pack takes it at [0-9]*s' <<< "$out")" = 2 ]; then pass 'pack: the dry run names each X item'"'"'s simulated take'
  else fail 'pack: the dry run names each X item'"'"'s simulated take' "$(grep 'lane X' <<< "$out")"; fi
  # (6) the fixture tier reads no bytes: a pair over any budget packs there.
  rc=$(BLOOMERY_PACK_BUDGET=1 steal_case pk-fx "pk-big	A	3090	3	$d" "pk-big2	B	a6000	3	$d" \
    -- --tier fixture pk-big@FAKE_SLEEP=3 pk-big2@FAKE_SLEEP=3)
  printf '%s' "$rc" > "$t/rc-pk-fx"
  rc_ok 'pack: a fixture pair over the budget ends green' pk-fx 2
  if overlap "$t/fake-state/ran.log" gen_pg gen_pg2; then pass 'pack: the fixture tier sums no bytes (its loads are the small fixtures)'
  else fail 'pack: the fixture tier sums no bytes (its loads are the small fixtures)' "the intervals did not cross"; fi
  # (7) the table's own errors: a solo recipe with no row, a row for a recipe that is not solo.
  cp "$t/justfile" "$t/justfile.pk"
  printf '%s\n' '' "[group('solo')]" 'pk-norow:' "    ./tools/box.sh 'bash tools/gpu-gate.sh gen_pn'" >> "$t/justfile"
  check 'pack: a solo recipe with no resource row is a named error' 65 \
    'recipe pk-norow: no resource table row in tools/gate-batch-resources.tsv' "${gb[@]}" --classes
  cp "$t/justfile.pk" "$t/justfile"
  cp "$t/tools/gate-batch-resources.tsv" "$t/tools/gate-batch-resources.pk"
  printf 'host\t-\t1000\n' >> "$t/tools/gate-batch-resources.tsv"
  check 'pack: a row for a recipe that is not solo is a named error' 65 \
    'recipe host: a resource table row for a recipe that is not \[group' "${gb[@]}" --classes
  cp "$t/tools/gate-batch-resources.pk" "$t/tools/gate-batch-resources.tsv"
  # (8) the pool mutex (batch-wide): two pool items the balance puts in two lanes never overlap.
  rc=$(steal_case pool-lanes "pool-any	A	3090	4	$d" "pool-two	B	a6000	4	$d" \
    -- pool-any@FAKE_SLEEP=4 pool-two@FAKE_SLEEP=4)
  printf '%s' "$rc" > "$t/rc-pool-lanes"
  rc_ok 'pool: two pool items in two lanes end green' pool-lanes 2
  if no_overlap "$t/fake-state/ran.log" gen_pl gen_p2; then pass 'pool: two pool items never overlap (the mutex)'
  else fail 'pool: two pool items never overlap (the mutex)' "the intervals crossed"; fi
  mutant_of pk-nopool 's|^pool_take() { until mkdir "\$POOL_MUTEX" 2> /dev/null; do sleep 1; done; }$|pool_take() { true; } # MUTANT pk-nopool|' '# MUTANT pk-nopool$'
  rc=$(MTAG=pk-nopool steal_case pool-lanesm "pool-any	A	3090	4	$d" "pool-two	B	a6000	4	$d" \
    -- pool-any@FAKE_SLEEP=4 pool-two@FAKE_SLEEP=4)
  printf '%s' "$rc" > "$t/rc-pool-lanesm"
  if overlap "$t/fake-state/ran.log" gen_pl gen_p2; then pass 'pool mutant: without the mutex the two pool items overlap'
  else fail 'pool mutant: without the mutex the two pool items overlap' "the intervals did not cross"; fi
  # (9) the budget scan follows the calls: a bound two scripts deep refuses the item; one hop deep
  # (the mutant) it reads unchecked and the batch runs it.
  printf '%s\n' 'bt-deep	A	3090	80	2026-09-27T10:00:00+0900' > "$t/times-bt.tsv"
  check 'budget: a bound two scripts deep is read (the refusal names it)' 65 \
    'gate-batch: budget: bt-deep=80s>past-0.75xBLOOMERY_GATE_BOUND=100s' \
    env BLOOMERY_GATE_TIMES=$t/times-bt.tsv "${gb[@]}" bt-deep
  mutant_of bt-nohind 's|^    follow_scripts = True$|    follow_scripts = False # MUTANT bt-nohind|' '# MUTANT bt-nohind$'
  rc=0
  out=$(env BLOOMERY_GATE_TIMES=$t/times-bt.tsv BLOOMERY_PACK_BUDGET=10000 bash "$t/tools/gate-batch-bt-nohind.sh" --dry-run bt-deep 2>&1) || rc=$?
  if [ "$rc" = 0 ] && grep -q 'budget: 1 item(s) unchecked' <<< "$out"; then pass 'budget mutant: one hop deep, the bound is invisible (unchecked)'
  else fail 'budget mutant: one hop deep, the bound is invisible (unchecked)' "rc=$rc $(grep budget <<< "$out")"; fi
  # The budget refusal: an item past 0.75x its bound is a named 65 before anything runs; the flag
  # passes it; the dry run names both lists (budget-two holds two bounded calls — unchecked).
  cp "$t/justfile" "$t/justfile.good"
  printf '%s\n' '' 'budget-long:' "    ./tools/box.sh 'BLOOMERY_GATE_BOUND=100 bash tools/gpu-gate.sh gen_bl'" \
    'budget-two:' "    ./tools/box.sh 'bash tools/gpu-gate.sh gen_t1 && bash tools/gpu-gate.sh gen_t2'" >> "$t/justfile"
  printf '%s\n' 'budget-long	A	3090	80	2026-09-27T10:00:00+0900' 'budget-two	A	3090	10	2026-09-27T10:00:00+0900' > "$t/times-budget.tsv"
  check 'budget: an item past 0.75x its bound is a named 65' 65 \
    'gate-batch: budget: budget-long=80s>past-0.75xBLOOMERY_GATE_BOUND=100s' \
    env BLOOMERY_GATE_TIMES=$t/times-budget.tsv "${gb[@]}" budget-long budget-two
  check 'budget: --over-budget-ok passes it' 0 '^DONE total=2 red=0' \
    env BLOOMERY_GATE_TIMES=$t/times-budget.tsv "${gb[@]}" --over-budget-ok budget-long budget-two
  printf '%s\n' 'budget-long	A	3090	80	2026-09-27T10:00:00+0900' 'budget-two	A	3090	10	2026-09-27T10:00:00+0900' > "$t/times-budget.tsv" # the run above appended its row
  out=$(env BLOOMERY_GATE_TIMES=$t/times-budget.tsv "${gb[@]}" --dry-run budget-long budget-two 2>&1) || fail 'budget: the dry run names both lists' "$out"
  if grep -q 'gate-batch: budget: 1 item(s) past 0.75x' <<< "$out" && grep -q 'unchecked (no single bounded call, or a bound the text does not spell): budget-two' <<< "$out"; then
    pass 'budget: the dry run names the refused and the unchecked'
  else fail 'budget: the dry run names the refused and the unchecked' "$(grep budget <<< "$out")"; fi
  mutant_of ch-nobudget 's|if len(calls) == 1 and calls\[0\]\[1\] > 0 and exp \* 4 > calls\[0\]\[1\] \* 3:|if False: # MUTANT ch-nobudget|' '^    if False: # MUTANT ch-nobudget$'
  printf '%s\n' 'budget-long	A	3090	80	2026-09-27T10:00:00+0900' 'budget-two	A	3090	10	2026-09-27T10:00:00+0900' > "$t/times-budget.tsv"
  rc=0
  out=$(env BLOOMERY_GATE_TIMES=$t/times-budget.tsv bash "$t/tools/gate-batch-ch-nobudget.sh" budget-long budget-two 2>&1) || rc=$?
  if [ "$rc" = 0 ] && grep -q '^budget-long rc=0 ' <<< "$out"; then pass 'budget mutant: without the check the over-budget item runs'
  else fail 'budget mutant: without the check the over-budget item runs' "rc=$rc $(tail -1 <<< "$out")"; fi
  cp "$t/justfile.good" "$t/justfile"
  # The plan record (REC_FIELDS, rec_get): a writer that gains or loses a field, or a call site that names a
  # field the list does not hold, ends the batch by name (70) at the first read — never a field read one place
  # late. A copy of the script per fault; each twin has the check that meets its fault removed, and the
  # case's own assertion must then fail (the count's fault is met later, by the balance, under another name;
  # the name's fault is an unset variable read under set -u, an unbound-variable error that names no field).
  rec_dies() { # <script> <ERE the output must hold>: the dry run of `host` ends 70 and its output names the fault
    rc=0
    out=$(bash "$1" --dry-run host 2>&1) || rc=$?
    [ "$rc" = 70 ] && grep -Eq -- "$2" <<< "$out"
  }
  mutant_of rec-more 's|else "", held,$|else "", held, "extra",  # MUTANT rec-more|' '# MUTANT rec-more$'
  if rec_dies "$t/tools/gate-batch-rec-more.sh" 'rec_get: plan record 0 has 17 fields, REC_FIELDS names 16'; then pass 'record: a writer that gains a field is a named 70 at the first read'
  else fail 'record: a writer that gains a field is a named 70 at the first read' "rc=$rc"; fi
  mutant_of rec-less 's|else "", held,$|else "",  # MUTANT rec-less|' '# MUTANT rec-less$'
  if rec_dies "$t/tools/gate-batch-rec-less.sh" 'rec_get: plan record 0 has 15 fields, REC_FIELDS names 16'; then pass 'record: a writer that loses a field is a named 70 at the first read'
  else fail 'record: a writer that loses a field is a named 70 at the first read' "rc=$rc"; fi
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  mutant_of rec-nocount 's|^  \[ \$((\${#_d} + 1)) = .*$|  true # MUTANT rec-nocount|;s|else "", held,$|else "", held, "extra",  # MUTANT rec-more|' '# MUTANT rec-nocount$'
  want 'record mutant: the no-count copy carries the gaining writer' "$t/tools/gate-batch-rec-nocount.sh" '# MUTANT rec-more$'
  if rec_dies "$t/tools/gate-batch-rec-nocount.sh" 'rec_get: plan record 0 has 17 fields, REC_FIELDS names 16'; then fail 'record mutant: without the count, rec_get still names the gained field' "$out"
  else pass 'record mutant: without the count, rec_get does not name the gained field'; fi
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  mutant_of rec-name 's|^      rec_get "\$i" cls$|      rec_get "$i" clss # MUTANT rec-name|' '# MUTANT rec-name$'
  if rec_dies "$t/tools/gate-batch-rec-name.sh" "rec_get: 'clss' is no field of the plan record"; then pass 'record: a field name the list does not hold is a named 70'
  else fail 'record: a field name the list does not hold is a named 70' "rc=$rc"; fi
  # shellcheck disable=SC2016 # a sed or awk program: its dollars are its own
  mutant_of rec-noname 's|^      rec_get "\$i" cls$|      rec_get "$i" clss # MUTANT rec-name|;s|^    \[ "\$_k" -lt .*$|    true # MUTANT rec-noname|' '# MUTANT rec-noname$'
  want 'record mutant: the no-name-check copy carries the misnamed call' "$t/tools/gate-batch-rec-noname.sh" '# MUTANT rec-name$'
  if rec_dies "$t/tools/gate-batch-rec-noname.sh" "rec_get: 'clss' is no field of the plan record"; then fail 'record mutant: without the name check, rec_get still names the field' "$out"
  else pass 'record mutant: without the name check the misnamed read names no field'; fi
  # The tier. solo-real is solo in the real tier and balanced in the fixture tier; `v41-load` pins lane A in the real tier only.
  check 'tier: a solo-real recipe is alone in the real tier (the pack takes either card)' 0 \
    "^sr-any	X	solo	3090,a6000	sr-any: \[group\('solo-real'\)\]; pack: -, 1000 B" "${gb[@]}" --classes
  check 'tier: … and balanced over both cards in the fixture tier' 0 "^sr-any	F	balanced	3090,a6000	sr-any: \[group\('solo-real'\)\]" \
    "${gb[@]}" --tier fixture --classes
  check 'tier: a solo-real v41-load recipe is alone in the real tier (solo wins)' 0 '^sr-v41	X	solo	3090,a6000	' "${gb[@]}" --classes
  check 'tier: … and balanced in the fixture tier, the v41-load group pinning no lane' 0 '^sr-v41	F	balanced	3090,a6000	' "${gb[@]}" --tier fixture --classes
  check 'tier: a plain v41-load recipe is a chain member in the real tier' 0 '^v41-any	C	chain	3090,a6000	' "${gb[@]}" --classes
  check 'tier: … and is balanced in the fixture tier, which takes no load lock' 0 '^v41-any	F	balanced	3090,a6000	' "${gb[@]}" --tier fixture --classes
  check 'tier: a 3090-only v41-load recipe stays in lane A in the fixture tier' 0 '^v41-a	A	fixed	3090	' "${gb[@]}" --tier fixture --classes
  check 'tier: a solo recipe is alone in both tiers' 0 '^v41-solo	X	solo	3090	' "${gb[@]}" --tier fixture --classes
  check 'tier: the real tier names nothing in the box env: lane X, no BLOOMERY_TIER, no card (the pack decides)' 0 \
    "^lane X  sr-any +just sr-any$" "${gb[@]}" --dry-run sr-any host
  check 'tier: the fixture tier passes BLOOMERY_TIER=fixture to the item, which lane A or B runs' 0 \
    "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_CARD=(3090|a6000)' just sr-any$" "${gb[@]}" --tier fixture --dry-run sr-any host
  out=$("${gb[@]}" --dry-run sr-any host plain-any 2>&1) || fail 'tier: the real dry run failed' "$out"
  if grep -q BLOOMERY_TIER <<< "$out"; then fail 'tier: the real tier leaves BLOOMERY_TIER out of every command' "a line names it"
  else pass 'tier: the real tier leaves BLOOMERY_TIER out of every command'; fi
  check 'tier: BLOOMERY_TIER in the environment is the batch tier' 0 "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture " \
    env BLOOMERY_TIER=fixture "${gb[@]}" --dry-run sr-any
  check 'tier: a BLOOMERY_TIER entry of the caller box env is the batch tier too, once' 0 "^lane [AB]  sr-any +BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_CARD=" \
    env BLOOMERY_BOX_ENV=BLOOMERY_TIER=fixture "${gb[@]}" --dry-run sr-any
  check 'tier: --tier other than real or fixture: 64, named' 64 "^gate-batch: --tier is real or fixture, got 'both'" "${gb[@]}" --tier both --dry-run host
  check 'tier: --tier without a value: 64' 64 '^gate-batch: --tier needs real or fixture' "${gb[@]}" --dry-run host --tier
  check 'tier: the flag and the environment naming two tiers: 64, named' 64 "^gate-batch: --tier names the tier 'fixture' and BLOOMERY_TIER in the environment 'real'" \
    env BLOOMERY_TIER=real "${gb[@]}" --tier fixture --dry-run host
  check 'tier: the flag and the box env naming two tiers: 64, named' 64 "^gate-batch: --tier names the tier 'real' and BLOOMERY_BOX_ENV's BLOOMERY_TIER 'fixture'" \
    env BLOOMERY_BOX_ENV=BLOOMERY_TIER=fixture "${gb[@]}" --tier real --dry-run host
  check "tier: an item's own BLOOMERY_TIER is refused: the tier is the batch's" 65 "sets BLOOMERY_TIER; the tier is the batch's" \
    "${gb[@]}" --dry-run 'plain-any@BLOOMERY_TIER=fixture'
  # The times file keeps the tiers apart: a fixture item's row is an entry of its own.
  printf '%s\n' 'plain-any	B	a6000	10	2026-09-27T10:00:00+0900' 'plain-any@BLOOMERY_TIER=fixture	B	a6000	77	2026-09-27T10:00:00+0900' > "$t/times-tier.tsv"
  check 'tier: a real item expects its real rows' 0 'balanced, expected 10 s \(median of 1 row\)' env "BLOOMERY_GATE_TIMES=$t/times-tier.tsv" "${gb[@]}" --dry-run plain-any
  check 'tier: a fixture item expects its own rows' 0 'balanced, expected 77 s \(median of 1 row\)' env "BLOOMERY_GATE_TIMES=$t/times-tier.tsv" "${gb[@]}" --tier fixture --dry-run plain-any
  # A solo-real recipe the fixture tier could not place is a named error.
  cp "$t/justfile" "$t/justfile.good"
  printf '%s\n' '' "[group('solo-real')]" "[group('solo')]" 'sr-solo:' "    ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with solo: a named error' 65 "recipe sr-solo: \[group\('solo-real'\)\] with \[group\('solo'\)\]" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-3090:' "    ./tools/box.sh 'bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with a 3090-only gpu-gate.sh call: a named error' 65 "recipe sr-3090: \[group\('solo-real'\)\] with its tools/gpu-gate.sh calls are not all .*(forms: 3090)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-none:' "    ./tools/box.sh 'bash tools/gate.sh -p y'" >> "$t/justfile"
  check 'tier: solo-real with no gpu-gate.sh call: a named error' 65 "recipe sr-none: \[group\('solo-real'\)\] with its tools/gpu-gate.sh calls are not all .*(forms: none)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('solo-real')]" 'sr-both:' "    BLOOMERY_CARD=both ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'tier: solo-real with a box.sh card pick: a named error' 65 "recipe sr-both: \[group\('solo-real'\)\] with BLOOMERY_CARD=both .*\(use solo\)" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('host')]" "[group('solo-real')]" 'sr-host:' "    ./tools/box.sh 'bash tools/gate.sh --oxide -p hd'" >> "$t/justfile"
  check 'tier: solo-real with host: a named error' 65 "recipe sr-host: \[group\('host'\)\] with \[group\('solo-real'\)\]" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  # A solo-real recipe's flags are its own arm: another recipe handing them is refused in either tier, and the recipe's own item
  # with them is not.
  printf '%s\n' '' "[group('solo-real')]" 'sr-arm *ARGS:' "    ./tools/box.sh 'BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gen_y --sr-arm {{ARGS}}'" >> "$t/justfile"
  check "tier: a solo-real recipe's arm handed through another recipe is refused in the fixture tier" 65 \
    "'v41-a:--sr-arm' hands gen_y --sr-arm, the arm of the solo recipe sr-arm" "${gb[@]}" --tier fixture --dry-run 'v41-a:--sr-arm'
  check '  … and in the real tier' 65 "'v41-a:--sr-arm' hands gen_y --sr-arm, the arm of the solo recipe sr-arm" "${gb[@]}" --dry-run 'v41-a:--sr-arm'
  check "  … while the recipe's own item with its flag runs in lane A or B in the fixture tier" 0 '^lane [AB]  sr-arm ' "${gb[@]}" --tier fixture --dry-run 'sr-arm:--sr-arm'
  check '  … and alone in the real tier' 0 '^lane X  sr-arm ' "${gb[@]}" --dry-run 'sr-arm:--sr-arm'
  cp "$t/justfile.good" "$t/justfile"
  # Run: a fixture batch's items carry the tier in their box env, every key call (the plan's and the recheck's) sees it, the
  # clauses an item left to the real tier are counted (its line, then the DONE line); the real tier does none of that.
  rc=$(steal_case tier-fx "steal-bal	B	a6000	10	$d" -- --tier fixture 'steal-bal@FAKE_DEFER=2')
  printf '%s' "$rc" > "$t/rc-tier-fx"
  rc_ok 'tier run: a fixture batch ends green' tier-fx 1
  want_row 'tier run: its box call carries BLOOMERY_TIER=fixture' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER=fixture/ && $2 ~ /gen_b$/'
  want 'tier run: its item line counts the deferred clauses' "$t/target/c-tier-fx/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=[AB] deferred=2 ledger=recorded item=steal-bal@FAKE_DEFER=2$'
  want 'tier run: the DONE line totals them' "$t/target/c-tier-fx/run.log" '^DONE total=1 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=2$'
  want 'tier run: the times row is an entry of its own' "$t/times-tier-fx.tsv" "$(printf '^steal-bal@FAKE_DEFER=2,BLOOMERY_TIER=fixture\t')"
  n=$((n + 1))
  if [ "$(grep -c . "$t/fake-state/keycalls.log")" -ge 2 ] && ! grep -qv 'BLOOMERY_TIER=fixture' "$t/fake-state/keycalls.log"; then echo 'ok tier run: the plan'"'"'s key call and the recheck both carry the tier'
  else bad=$((bad + 1)); echo "FAIL tier run: a key call lacks the tier: $(cat "$t/fake-state/keycalls.log" 2>&1)"; fi
  # A fixture batch whose gates left nothing to the real tier (none is converted yet) still ends green, its total 0.
  rc=$(steal_case tier-f0 "steal-bal	B	a6000	10	$d" -- --tier fixture steal-bal)
  printf '%s' "$rc" > "$t/rc-tier-f0"
  rc_ok 'tier run: a fixture batch whose gates defer nothing ends green' tier-f0 1
  want 'tier run: … its DONE line totals 0' "$t/target/c-tier-f0/run.log" '^DONE total=1 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=0$'
  want_not 'tier run: … and its item line counts none' "$t/target/c-tier-f0/run.log" '^steal-bal .* deferred='
  rc=$(steal_case tier-re "steal-bal	B	a6000	10	$d" -- 'steal-bal@FAKE_DEFER=2')
  printf '%s' "$rc" > "$t/rc-tier-re"
  rc_ok 'tier run: the same item in the real tier ends green' tier-re 1
  want_no_row 'tier run: … and none of its calls names the tier' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER/'
  want_not 'tier run: … and the DONE line counts no deferrals' "$t/target/c-tier-re/run.log" '^DONE .* deferred='
  want 'tier run: … though its line does' "$t/target/c-tier-re/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=[AB] deferred=2 ledger=recorded '
  # real-only (the header's paragraph): a no-op in the real tier, a deferral in the fixture tier, never a lane, a key or a call.
  check 'real-only: in the real tier the recipe is placed by its calls, as without the dependency' 0 \
    '^ro-any	F	balanced	3090,a6000	ro-any: real-only call; ro-any: gpu-gate.sh any' "${gb[@]}" --classes
  check '  … and a solo recipe stays alone' 0 "^ro-solo	X	solo	3090	" "${gb[@]}" --classes
  check 'real-only: in the fixture tier the recipe is deferred, in no lane' 0 '^ro-any	D	real-only	-	ro-any: real-only call' \
    "${gb[@]}" --tier fixture --classes
  check '  … a solo one too (the group beats solo there)' 0 '^ro-solo	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  check '  … and one that reaches the guard through another recipe, by closure' 0 '^ro-chain	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  check 'real-only: the real dry run places it as usual' 0 '^lane [AB]  ro-any ' "${gb[@]}" --dry-run ro-any
  out=$("${gb[@]}" --tier fixture --dry-run ro-any ro-solo plain-any 2>&1) || fail 'real-only: the fixture dry run failed' "$out"
  if grep -q '^lane -  ro-any ' <<< "$out" && grep -q '^lane -  ro-solo ' <<< "$out" && grep -q '^lane [AB]  plain-any '  <<< "$out" \
    && grep -q '2 item(s) deferred to the real tier (2 real-only, 0 of a family with no fixture), in no lane (DONE: deferred=+2, and the list .*/deferred.list): ro-any ro-solo' <<< "$out" \
    && grep -q 'dry run — 1 items (and 2 deferred to the real tier: 2 real-only, 0 of a family with no fixture)' <<< "$out"; then pass 'real-only: the fixture dry run names both deferred items beside the one that runs'
  else fail 'real-only: the fixture dry run names both deferred items beside the one that runs' "$out"; fi
  check 'real-only: a deferred item adds nothing to the lane sums (plain-any alone: its 45 s default, no fixture row)' 0 'predicted laneA=0s laneB=45s laneX=0s chain=0s wall=45s' \
    env "BLOOMERY_GATE_TIMES=$t/times.tsv" "${gb[@]}" --tier fixture --dry-run ro-any ro-solo plain-any
  check "real-only: an item's ARGS are still checked by just" 65 'takes no arguments' "${gb[@]}" --tier fixture --dry-run 'ro-any:--x'
  rc=$(steal_case ro-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any 'steal-bal@FAKE_DEFER=2' ro-solo)
  printf '%s' "$rc" > "$t/rc-ro-fx"
  rc_ok 'real-only run: a fixture batch with two deferred items ends green' ro-fx 3
  want 'real-only run: the deferred item has its own line' "$t/target/c-ro-fx/run.log" '^ro-any rc=deferred real-only lane=- deferred=1$'
  want '  … a solo one too' "$t/target/c-ro-fx/run.log" '^ro-solo rc=deferred real-only lane=- deferred=1$'
  want 'real-only run: the DONE line counts them in deferred= (with the clauses) and in real_only=' "$t/target/c-ro-fx/run.log" \
    '^DONE total=3 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=4 real_only=2 deferred_list=[^ ]*/deferred.list$'
  want_no_row 'real-only run: the batch never called a deferred recipe' "$t/fake-state/box.log" '$2 ~ /gen_ro$|gen_rs$/'
  want_no_row '  … and wrote no times row for it' "$t/times-ro-fx.tsv" '$1 ~ /^ro-/'
  want_not '  … nor a ledger record' "$t/rounds-ro-fx.tsv" '^[^	]*ro-(any|solo)'
  rc=$(steal_case ro-all "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any ro-solo)
  printf '%s' "$rc" > "$t/rc-ro-all"
  rc_ok 'real-only run: a batch of deferred items only ends green' ro-all 2
  want 'real-only run: … its DONE line' "$t/target/c-ro-all/run.log" '^DONE total=2 red=0 skipped=0 wall=[0-9]+s laneA=0s laneB=0s deferred=2 real_only=2 deferred_list=[^ ]*/deferred.list$'
  want 'real-only run: … says nothing ran on the box' "$t/out-ro-all.log" '^gate-batch: every item is deferred to the real tier — nothing runs on the box, so no lease check$'
  want_no_row 'real-only run: … and made no box call at all' "$t/fake-state/box.log" '1'
  rc=$(steal_case ro-real "steal-bal	B	a6000	10	$d" -- ro-any)
  printf '%s' "$rc" > "$t/rc-ro-real"
  rc_ok 'real-only run: the real tier runs the recipe' ro-real 1
  want_row '  … through its box call' "$t/fake-state/box.log" '$2 ~ /gen_ro$/'
  want_not '  … and its DONE line has no real_only=' "$t/target/c-ro-real/run.log" '^DONE .*real_only='
  # no-fixture (the header's deferred paragraph, kind 2): a family the fixture table of tools/ref/ref-paths.sh does not name is deferred
  # in the fixture tier, never run and never red; the table is read from that file, so a family it names runs.
  check 'no-fixture: in the real tier a recipe of a family with no fixture is placed as usual' 0 '^x-glm	X	solo	' "${gb[@]}" --classes
  check '  … in the fixture tier it is deferred, named by its family' 0 '^x-glm	D	no-fixture	-	.*family glm5next: no row in the fixture table of tools/ref/ref-paths.sh \(box.sh exits 66\)' \
    "${gb[@]}" --tier fixture --classes
  check '  … a family with a fixture directory runs (qwen4exp)' 0 '^x-q	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … a family whose real file stands runs (the default profile)' 0 '^x-none	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … a family named by closure runs (deepseek41)' 0 '^x-v41b	X	solo	' "${gb[@]}" --tier fixture --classes
  check '  … and a recipe that is both is real-only (the declared fact wins)' 0 '^ro-glm	D	real-only	-	' "${gb[@]}" --tier fixture --classes
  out=$("${gb[@]}" --tier fixture --dry-run x-glm plain-any 2>&1) || fail 'no-fixture: the fixture dry run failed' "$out"
  if grep -q '^lane -  x-glm ' <<< "$out" && grep -q 'deferred to the real tier (no-fixture), never run in the fixture tier — ' <<< "$out" \
    && grep -q 'dry run — 1 items (and 1 deferred to the real tier: 0 real-only, 1 of a family with no fixture)' <<< "$out"; then pass 'no-fixture: the fixture dry run names the family item as deferred (no-fixture)'
  else fail 'no-fixture: the fixture dry run names the family item as deferred (no-fixture)' "$out"; fi
  rc=$(steal_case nf-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any x-glm 'x-glm:--x' steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-fx"
  check_nf=$(cat "$t/out-nf-fx.log" 2>&1)
  if [ "$rc" = 65 ] && grep -q "x-glm takes no arguments" <<< "$check_nf"; then pass 'no-fixture: an ARGS error of a deferred item is still named, and nothing ran'
  else fail 'no-fixture: an ARGS error of a deferred item is still named, and nothing ran' "rc $rc: $check_nf"; fi
  rc=$(steal_case nf-fx "steal-bal	B	a6000	10	$d" -- --tier fixture ro-any x-glm steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-fx"
  rc_ok 'no-fixture run: a fixture batch with a real-only and a no-fixture item ends green' nf-fx 3
  want 'no-fixture run: the family item has its own line' "$t/target/c-nf-fx/run.log" '^x-glm rc=deferred no-fixture lane=- deferred=1$'
  want '  … beside the real-only one' "$t/target/c-nf-fx/run.log" '^ro-any rc=deferred real-only lane=- deferred=1$'
  want 'no-fixture run: the DONE line counts both kinds and names the list' "$t/target/c-nf-fx/run.log" \
    '^DONE total=3 red=0 skipped=0 wall=[0-9]+s laneA=[0-9]+s laneB=[0-9]+s deferred=2 real_only=1 no_fixture=1 deferred_list=[^ ]*/c-nf-fx/deferred.list$'
  want_no_row 'no-fixture run: the batch never called either recipe' "$t/fake-state/box.log" '$2 ~ /gen_ro$|xg$/'
  n=$((n + 1))
  if [ "$(grep -v '^#' "$t/target/c-nf-fx/deferred.list" | paste -sd, -)" = 'ro-any,x-glm' ] && grep -q '^# real-only: ' "$t/target/c-nf-fx/deferred.list" \
    && grep -q '^# no-fixture: ' "$t/target/c-nf-fx/deferred.list"; then echo 'ok no-fixture run: DIR/deferred.list holds the two items, real-only first, with a comment for each kind'
  else bad=$((bad + 1)); echo "FAIL no-fixture run: DIR/deferred.list: $(cat "$t/target/c-nf-fx/deferred.list" 2>&1)"; fi
  check 'no-fixture run: a real-tier batch takes exactly that list (both items placed as usual)' 0 '^lane X  x-glm ' "${gb[@]}" --dry-run --list "$t/target/c-nf-fx/deferred.list"
  out=$("${gb[@]}" --dry-run --list "$t/target/c-nf-fx/deferred.list" 2>&1) || fail 'no-fixture run: the real dry run of the list failed' "$out"
  if grep -q '^lane [AB]  ro-any ' <<< "$out" && grep -q 'dry run — 2 items,' <<< "$out"; then pass 'no-fixture run: … two items, none deferred'
  else fail 'no-fixture run: … two items, none deferred' "$out"; fi
  rc=$(steal_case nf-item "steal-bal	B	a6000	10	$d" -- --tier fixture 'ro-any@FAKE_X=1' steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-item"
  want 'no-fixture run: a deferred item with env keeps its item text in the list' "$t/target/c-nf-item/deferred.list" '^ro-any@FAKE_X=1$'
  rc=$(steal_case nf-none "steal-bal	B	a6000	10	$d" -- --tier fixture steal-bal)
  printf '%s' "$rc" > "$t/rc-nf-none"
  if [ ! -e "$t/target/c-nf-none/deferred.list" ]; then pass 'no-fixture run: a batch that deferred nothing writes no list'
  else fail 'no-fixture run: a batch that deferred nothing writes no list' "$(cat "$t/target/c-nf-none/deferred.list")"; fi
  want_not '  … and its DONE line has none of the deferral fields' "$t/target/c-nf-none/run.log" '^DONE .*(real_only|no_fixture|deferred_list)='
  # The fixture table is ref-paths.sh's: a table this reader cannot read, or whose default is not `none`, is a named error.
  cp "$t/tools/ref/ref-paths.sh" "$t/ref-paths.good"
  sed -i.bak '/__fixture_dir=/d' "$t/tools/ref/ref-paths.sh" && rm -f "$t/tools/ref/ref-paths.sh.bak"
  check 'no-fixture: a ref-paths.sh with no fixture table: a named error' 65 'no fixture table .* tools/gate-batch.sh reads which families have a fixture there' "${gb[@]}" --tier fixture --classes
  check '  … the real tier never reads it' 0 '^x-glm	X	solo	' "${gb[@]}" --classes
  cp "$t/ref-paths.good" "$t/tools/ref/ref-paths.sh"
  sed -i.bak 's/\*) __fixture_dir=none ;;/*) __fixture_dir=v41 ;;/' "$t/tools/ref/ref-paths.sh" && rm -f "$t/tools/ref/ref-paths.sh.bak"
  check "no-fixture: a table whose *) row is not none: a named error" 65 "the fixture table's .\*\). row is 'v41', not none" "${gb[@]}" --tier fixture --classes
  cp "$t/ref-paths.good" "$t/tools/ref/ref-paths.sh"
  # The real table: the tree's own ref-paths.sh parses, and the family it names runs (a converted V4.1 gate is balanced, not deferred).
  check "no-fixture: the tree's own tools/ref/ref-paths.sh is a table this runner reads" 0 '^gate-gpu-ds41-step	F	balanced	' bash "$self" --tier fixture --classes
  # The class's own errors, wherever the recipe sits.
  printf '%s\n' '' 'ro-other:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-any && bash tools/gate.sh -p z'" >> "$t/justfile"
  check "real-only: a call that names another recipe: a named error" 65 "recipe ro-other: its real-only call names ro-any, not itself" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-late:' "    ./tools/box.sh 'cargo build && bash tools/ref/real-only.sh ro-late && bash tools/gate.sh -p z'" >> "$t/justfile"
  check 'real-only: a box command that does not open with the call: a named error' 65 \
    'recipe ro-late: a real-only recipe opens every box.sh command with .bash tools/ref/real-only.sh ro-late && …. — 1 single-quoted box.sh command\(s\), 1 call\(s\), 1 not opening with it' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-two:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-two && bash tools/gate.sh -p a'" "    ./tools/box.sh 'bash tools/gate.sh -p b'" >> "$t/justfile"
  check 'real-only: a second box command with no call: a named error' 65 'recipe ro-two: .* 2 single-quoted box.sh command\(s\), 1 call\(s\), 1 not opening with it' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'ro-bare:' "    bash tools/ref/real-only.sh ro-bare" >> "$t/justfile"
  check 'real-only: a call outside a single-quoted box command: a named error' 65 'recipe ro-bare: .* 0 single-quoted box.sh command\(s\), 1 call\(s\)' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  mv "$t/tools/ref/real-only.sh" "$t/tools/ref/real-only.sh.off"
  check 'real-only: a call to a script that is not in the tree: a named error' 65 'recipe ro-any: calls tools/ref/real-only.sh, which is not in the tree' "${gb[@]}" --classes
  mv "$t/tools/ref/real-only.sh.off" "$t/tools/ref/real-only.sh"
  printf '%s\n' '' "[group('solo-real')]" 'ro-sr:' "    ./tools/box.sh 'bash tools/ref/real-only.sh ro-sr && BLOOMERY_GATE_CARD=\${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'real-only: with solo-real: a named error' 65 'recipe ro-sr: \[group\(.solo-real.\)\] with a real-only call' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  # The guard script itself: it stops the fixture tier by name before any box call, and only that tier.
  local ro_sh
  ro_sh=$(dirname "$self")/ref/real-only.sh
  check 'real-only guard: BLOOMERY_TIER=fixture stops it by name, exit 66' 66 '^gate-x: real-only: .*exit 66' env BLOOMERY_TIER=fixture bash "$ro_sh" gate-x
  check '  … the real tier passes' 0 '^$' env BLOOMERY_TIER=real bash "$ro_sh" gate-x
  check '  … and so does no tier, or an empty one' 0 '^$' env -u BLOOMERY_TIER bash "$ro_sh" gate-x
  check '  … a tier that is neither is refused, 64' 64 "got 'both'" env BLOOMERY_TIER=both bash "$ro_sh" gate-x
  check '  … a call with no recipe name is a usage error, 64' 64 'usage: bash tools/ref/real-only.sh <recipe>' env BLOOMERY_TIER=fixture bash "$ro_sh"
  # One card by box.sh's pick (solo-real, BLOOMERY_CARD=a6000): alone in the real tier, fixed to lane B in the fixture tier; two cards stay refused (sr-both above).
  check 'one card: a solo-real recipe with BLOOMERY_CARD=a6000 is alone in the real tier' 0 '^sr-a6000	X	solo	-	' "${gb[@]}" --classes
  check '  … and fixed to the A6000 lane in the fixture tier' 0 '^sr-a6000	B	fixed	-	' "${gb[@]}" --tier fixture --classes
  out=$("${gb[@]}" --tier fixture --dry-run sr-a6000 plain-any host 2>&1) || fail 'one card: the fixture dry run failed' "$out"
  if grep -q '^lane B  sr-a6000 ' <<< "$out" && grep -q "^lane B  sr-a6000 .*BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture' just sr-a6000$" <<< "$out" \
    && ! grep -A3 '^lane B  sr-a6000 ' <<< "$out" | grep -q movable; then pass 'one card: the fixture dry run puts it in lane B with no card forced, and not movable'
  else fail 'one card: the fixture dry run puts it in lane B with no card forced, and not movable' "$out"; fi
  rc=$(steal_case sr-one "plain-any	B	a6000	10	$d" "host	B	none	5	$d" "sr-a6000	B	a6000	8	$d" -- --tier fixture host plain-any sr-a6000)
  printf '%s' "$rc" > "$t/rc-sr-one"
  rc_ok 'one card: a fixture batch with it ends green' sr-one 3
  want 'one card: it ran in lane B' "$t/target/c-sr-one/run.log" '^sr-a6000 rc=0 [0-9]+s try=1 lane=B( |$)'
  want_row '  … with the tier and no card of the batch in its env' "$t/fake-state/box.log" '$1 ~ /BLOOMERY_TIER=fixture/ && $1 !~ /BLOOMERY_GATE_CARD/ && $2 ~ /gen_sa/'
  want 'one card: its times row names lane B and the A6000' "$t/times-sr-one.tsv" "$(printf '^sr-a6000@BLOOMERY_TIER=fixture\tB\ta6000\t')"
  # The GPU hold (the header). Its writer is the real tools/gpu-gate.sh on the fixture (the fake box above), so these cases cross the
  # batch, box.sh's two flags and the box-side file. A --ledger batch puts the hold up for its whole run — two lanes here: lane B's
  # item ends first and its lane ends, and the hold must still be up when lane A's slow item ends — with a heartbeat (1 s here),
  # each item running with its batch's owner in the call's env only (never the key, the ledger or the times row: the caller's own
  # box env KEEP=1 makes the key calls non-empty), and the hold down at the end.
  local items_only='$3 != "true" && $3 !~ /box-manifest/'
  local slow_h='steal-slow@FAKE_SLEEP=5'
  rc=$(LFLAG=--ledger BLOOMERY_GPU_HOLD_BEAT=1 BLOOMERY_BOX_ENV=KEEP=1 steal_case hold-up "$slow_h	A	3090	30	$d" "plain-any	B	a6000	10	$d" -- "$slow_h" plain-any)
  printf '%s' "$rc" > "$t/rc-hold-up"
  rc_ok 'hold: a --ledger batch of two lanes ends green' hold-up 2
  want 'hold: it names the hold up, and down' "$t/out-hold-up.log" '^gate-batch: gpu hold: up as gb_[A-Za-z0-9_]+ \(.*batch\.gpuhold, refreshed every 1 s, log .*gpuhold\.log\)'
  want '  … and down' "$t/out-hold-up.log" '^gate-batch: gpu hold: down$'
  want_row 'hold: both items ran, each under its own batch'"'"'s hold file' "$t/fake-state/seen.log" "$items_only"' { c++ } END { if (c == 2) print c }'
  want_no_row '  … none without its owner in the env and the hold in place' "$t/fake-state/seen.log" "$items_only"' && index($2, "owner=" $1 " since=") != 1'
  want_no_row '  … every item call carries BLOOMERY_BATCH_OWNER in its box env' "$t/fake-state/box.log" '$2 != "true" && $2 !~ /box-manifest/ && $1 !~ /BLOOMERY_BATCH_OWNER=gb_/'
  want '  … and the caller'"'"'s own box env reached the key calls' "$t/fake-state/keycalls.log" '^KEEP=1$'
  want_not '  … the owner is in no key call' "$t/fake-state/keycalls.log" 'BATCH_OWNER'
  want '  … the ledger recorded both' "$t/lead-hold-up.tsv" '^stub-plain-any'
  want_not '  … the owner is in no ledger line' "$t/lead-hold-up.tsv" 'BATCH_OWNER'
  want_not '  … nor in a times row' "$t/times-hold-up.tsv" 'BATCH_OWNER'
  want_not '  … and run.log carries no hold line' "$t/target/c-hold-up/run.log" 'gpuhold|gpu hold|BATCH_OWNER'
  want_no_row '  … every hold call went read-only, with no guard wait' "$t/fake-state/hold.log" '$0 !~ /^ro=1 wait=0 bash tools\/gpu-gate\.sh --gpuhold /'
  want_row '  … one up' "$t/fake-state/hold.log" '/--gpuhold up gb_/ { c++ } END { if (c == 1) print c }'
  want_row '  … at least two beats (5 s of an item at 1 s)' "$t/fake-state/hold.log" '/--gpuhold beat gb_/ { c++ } END { if (c >= 2) print c }'
  want_row '  … one down' "$t/fake-state/hold.log" '/--gpuhold down gb_/ { c++ } END { if (c == 1) print c }'
  want '  … the heartbeat logged its beats' "$t/target/c-hold-up/gpuhold.log" 'beat ok$'
  want_row '  … the hold was up when each item ended (a finished lane did not take it down)' "$t/fake-state/end.log" '$1 == "present" && $2 != "true" && $2 !~ /box-manifest/ { c++ } END { if (c == 2) print c }'
  want_no_row '  … and absent at none of them' "$t/fake-state/end.log" '$1 != "present" && $2 != "true" && $2 !~ /box-manifest/'
  if [ ! -e "$t/fake-state/batch.gpuhold" ]; then pass '  … and the hold file is gone once the batch ends'
  else fail '  … and the hold file is gone once the batch ends' "$(cat "$t/fake-state/batch.gpuhold")"; fi
  hp=$(cat "$t/target/c-hold-up/gpuhold.pid" 2> /dev/null || echo 0)
  if [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null; then pass '  … and the heartbeat is gone'
  else fail '  … and the heartbeat is gone' "pid $hp answers kill -0"; fi
  # No hold: a --round-ledger batch, and a --ledger batch that opts out. Every item ran with no owner and no hold file, no call
  # carried an owner, and no hold call and no heartbeat pid exist.
  rc=$(steal_case hold-round "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-round"
  rc_ok 'no hold: a --round-ledger batch ends green' hold-round 1
  want_row '  … its item ran' "$t/fake-state/seen.log" "$items_only"' { c++ } END { if (c == 1) print c }'
  want_no_row '  … with no owner and no hold file' "$t/fake-state/seen.log" "$items_only"' && ($1 != "none" || $2 != "nohold")'
  want_not '  … no call carries an owner' "$t/fake-state/box.log" 'BATCH_OWNER'
  if [ ! -e "$t/fake-state/hold.log" ] && [ ! -e "$t/target/c-hold-round/gpuhold.pid" ]; then pass '  … no hold call, no heartbeat pid'
  else fail '  … no hold call, no heartbeat pid' "hold.log or gpuhold.pid exists"; fi
  rc=$(LFLAG=--ledger steal_case hold-off "plain-any	B	a6000	10	$d" -- --no-gpu-hold plain-any)
  printf '%s' "$rc" > "$t/rc-hold-off"
  rc_ok '  … and a --ledger batch with --no-gpu-hold ends green' hold-off 1
  want_no_row '  … with no owner and no hold file' "$t/fake-state/seen.log" "$items_only"' && ($1 != "none" || $2 != "nohold")'
  want_not '  … no call carries an owner' "$t/fake-state/box.log" 'BATCH_OWNER'
  if [ ! -e "$t/fake-state/hold.log" ] && [ ! -e "$t/target/c-hold-off/gpuhold.pid" ]; then pass '  … no hold call, no heartbeat pid'
  else fail '  … no hold call, no heartbeat pid' "hold.log or gpuhold.pid exists"; fi
  # A batch that is red takes the hold down too (the EXIT trap: every exit).
  rc=$(LFLAG=--ledger steal_case hold-red "plain-any	B	a6000	10	$d" -- 'plain-any@FAKE_RC=1')
  printf '%s' "$rc" > "$t/rc-hold-red"
  if [ "$(cat "$t/rc-hold-red")" = 1 ] && grep -Eq '^DONE total=1 red=1 ' "$t/out-hold-red.log" && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a red batch (rc 1) takes it down'
  else fail 'hold: a red batch (rc 1) takes it down' "rc=$(cat "$t/rc-hold-red") hold file: $(cat "$t/fake-state/batch.gpuhold" 2>&1 | head -1)"; fi
  want_row '  … through the down call' "$t/fake-state/hold.log" '/--gpuhold down gb_/ { c++ } END { if (c == 1) print c }'
  # INT/TERM (stop): TERM to the batch's own pid, once its item runs.
  local slow_s='steal-slow@FAKE_SLEEP=20'
  printf '%s\n' "$slow_s	A	3090	30	$d" > "$t/times-hold-stop.tsv"
  rm -rf "$t/fake-state" "$t/target/c-hold-stop"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-stop.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-stop.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-stop" --ledger "$slow_s" > "$t/out-hold-stop.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  held=$(head -1 "$t/fake-state/batch.gpuhold" 2> /dev/null || echo none)
  kill -TERM "$sp"
  rc=0
  wait "$sp" || rc=$?
  hp=$(cat "$t/target/c-hold-stop/gpuhold.pid" 2> /dev/null || echo 0)
  case $held in owner=gb_*) pass 'stop: the hold was up while the item ran' ;; *) fail 'stop: the hold was up while the item ran' "the hold file read '$held'" ;; esac
  if [ "$rc" = 130 ] && [ ! -e "$t/fake-state/batch.gpuhold" ] && [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null; then
    pass '  … TERM to the batch ends it (130), takes the hold down and ends the heartbeat'
  else fail '  … TERM to the batch ends it (130), takes the hold down and ends the heartbeat' "rc=$rc hold: $(cat "$t/fake-state/batch.gpuhold" 2>&1 | head -1) heartbeat pid $hp"; fi
  want '  … and says it stopped' "$t/out-hold-stop.log" '^gate-batch: stopped;'
  # die: a lane killed from outside (its pid, as written under DIR) ends the batch through die (70), and the hold still comes down.
  rm -rf "$t/fake-state" "$t/target/c-hold-die"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-die.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-die.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-die" --ledger "$slow_s" > "$t/out-hold-die.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  kill -TERM "$(cat "$t/target/c-hold-die/lane-A.pid")"
  rc=0
  wait "$sp" 2> /dev/null || rc=$?
  if [ "$rc" = 70 ] && grep -q 'ended without its sentinel' "$t/out-hold-die.log" && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a batch that ends through die (a lane killed: 70) takes it down'
  else fail 'hold: a batch that ends through die (a lane killed: 70) takes it down' "rc=$rc hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  if [ -f "$t/target/c-hold-die/lane-A.child" ]; then kill -TERM "$(cat "$t/target/c-hold-die/lane-A.child")" 2> /dev/null || true; fi
  # An unclean death (KILL runs no trap): the hold is left up, but its heartbeat sees the batch gone and stops beating, so the box ages
  # the hold out instead of it staying fresh for ever.
  rm -rf "$t/fake-state" "$t/target/c-hold-kill"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-kill.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-kill.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-kill" --ledger "$slow_s" > "$t/out-hold-kill.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  hp=$(cat "$t/target/c-hold-kill/gpuhold.pid" 2> /dev/null || echo 0)
  kill -KILL "$sp"
  rc=0
  wait "$sp" 2> /dev/null || rc=$?
  k=0
  while [ "$hp" -gt 1 ] && kill -0 "$hp" 2> /dev/null && [ "$k" -lt 10 ]; do
    sleep 1
    k=$((k + 1))
  done
  if [ "$hp" -gt 1 ] && ! kill -0 "$hp" 2> /dev/null && [ -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'stop: a batch killed outright leaves its hold up, and its heartbeat ends within a beat'
  else fail 'stop: a batch killed outright leaves its hold up, and its heartbeat ends within a beat' "heartbeat pid $hp after ${k} s, hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want '  … and logs why' "$t/target/c-hold-kill/gpuhold.log" 'is gone: the heartbeat stops'
  for f in "$t"/target/c-hold-kill/lane-*.pid "$t"/target/c-hold-kill/lane-*.child; do # the killed batch's lanes: the pids it wrote
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  # A hold that vanishes (taken down by hand): the next beat finds it gone (3), logs it and the heartbeat stops — and no beat
  # creates the file again.
  rm -rf "$t/fake-state" "$t/target/c-hold-gone"
  BLOOMERY_GATE_TIMES=$t/times-hold-stop.tsv BLOOMERY_GATE_LEDGER=$t/lead-hold-gone.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-hold-gone.tsv \
    BLOOMERY_GPU_HOLD_BEAT=1 "${gb[@]}" --out "$t/target/c-hold-gone" --ledger "$slow_s" > "$t/out-hold-gone.log" 2>&1 &
  sp=$!
  k=0
  while ! grep -q 'gen_s' "$t/fake-state/seen.log" 2> /dev/null && [ "$k" -lt 20 ]; do
    sleep 1
    k=$((k + 1))
  done
  rm -f "$t/fake-state/batch.gpuhold"
  # The beat that finds it gone logs the stop; the count of beat calls is read after that line and again 2 s later (beats 1 s apart).
  k=0
  while ! grep -q 'the heartbeat stops' "$t/target/c-hold-gone/gpuhold.log" 2> /dev/null && [ "$k" -lt 15 ]; do
    sleep 1
    k=$((k + 1))
  done
  b1=$(grep -c -- '--gpuhold beat' "$t/fake-state/hold.log" || true)
  sleep 2
  b2=$(grep -c -- '--gpuhold beat' "$t/fake-state/hold.log" || true)
  if [ "$b1" -ge 1 ] && [ "$b1" = "$b2" ] && [ ! -e "$t/fake-state/batch.gpuhold" ]; then
    pass 'hold: a hold taken down by hand: the heartbeat stops after the beat that finds it gone, and creates nothing'
  else fail 'hold: a hold taken down by hand: the heartbeat stops after the beat that finds it gone, and creates nothing' "beats $b1 then $b2, hold file: $(ls "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want '  … and logs why' "$t/target/c-hold-gone/gpuhold.log" "the hold is gone or another batch's: the heartbeat stops"
  kill -TERM "$sp"
  wait "$sp" 2> /dev/null || true
  # Another batch's fresh hold: not starting (75, named), nothing ran, its hold untouched, no run.log; a stale one is taken over.
  rc=$(PRE_HOLD=otherbatch LFLAG=--ledger steal_case hold-foreign "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-foreign"
  if [ "$(cat "$t/rc-hold-foreign")" = 75 ] && grep -q "is up for owner otherbatch" "$t/out-hold-foreign.log" \
    && grep -q "^gate-batch: another batch's GPU hold is up" "$t/out-hold-foreign.log"; then
    pass 'hold: another batch'"'"'s fresh hold: not starting, 75, named'
  else fail 'hold: another batch'"'"'s fresh hold: not starting, 75, named' "rc=$(cat "$t/rc-hold-foreign") $(tail -2 "$t/out-hold-foreign.log")"; fi
  if head -1 "$t/fake-state/batch.gpuhold" | grep -q '^owner=otherbatch ' && [ ! -e "$t/target/c-hold-foreign/run.log" ]; then
    pass '  … its hold stays, and the refused start left no run.log'
  else fail '  … its hold stays, and the refused start left no run.log' "$(head -1 "$t/fake-state/batch.gpuhold" 2>&1)"; fi
  want_no_row '  … and no item ran' "$t/fake-state/seen.log" "$items_only"
  rc=$(PRE_HOLD=otherbatch PRE_AGE=400 LFLAG=--ledger steal_case hold-stale "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-stale"
  rc_ok 'hold: a stale hold of another batch is taken over, and the batch ends green' hold-stale 1
  want '  … the takeover is named' "$t/target/c-hold-stale/gpuhold.log" 'owner otherbatch was not refreshed for 4[0-9][0-9] s: taking it over'
  if [ ! -e "$t/fake-state/batch.gpuhold" ]; then pass '  … and the hold is down at the end'
  else fail '  … and the hold is down at the end' "$(cat "$t/fake-state/batch.gpuhold")"; fi
  # A round batch yields to a hold another batch put up (the header's GPU-hold paragraph): the item whose
  # try ends rc 75 on it is not retried — it ends rc=yield naming the owner, the DONE line counts yielded=,
  # no times row and no ledger record are written, and the lane moves on to its next item. The fake box's
  # FAKE_HOLD75/FAKE_CARD75 end a call as gpu-gate.sh does at its bounds; GBFAST=1 runs the fast-retry
  # copy, so the retries that still happen (the plain queue, the lead's batch) cost 1 s each.
  rc=$(GBFAST=1 steal_case yield-hold "plain-any	B	a6000	10	$d" -- 'plain-any@FAKE_HOLD75=leadbatch' steal-bal)
  printf '%s' "$rc" > "$t/rc-yield-hold"
  rc_ok 'yield: a round batch with a hold-blocked item ends green (not red)' yield-hold 2
  want 'yield: the item ends rc=yield naming the hold'"'"'s owner, not retried' "$t/target/c-yield-hold/run.log" \
    "^plain-any rc=yield held by leadbatch: not retried \(a round batch yields to the lead's hold\) [0-9]+s try=1 lane=[AB] ledger=yield item=plain-any@FAKE_HOLD75=leadbatch$"
  want 'yield: the DONE line counts it in yielded=' "$t/target/c-yield-hold/run.log" '^DONE total=2 red=0 skipped=0 yielded=1 wall=[0-9]+s'
  want_row 'yield: one call, then no retry — it never starts in a gap the lead leaves' "$t/fake-state/box.log" \
    '$1 ~ /FAKE_HOLD75=leadbatch/ { c++ } END { if (c == 1) print c }'
  want '  … and the lane moved on to its next item' "$t/target/c-yield-hold/run.log" '^steal-bal rc=0 [0-9]+s try=1 lane=[AB] ledger=recorded( |$)'
  want_not 'yield: no rc 75 line for it' "$t/target/c-yield-hold/run.log" ' rc=75 '
  want_not 'yield: and no ledger record' "$t/rounds-yield-hold.tsv" 'plain-any'
  want_no_row 'yield: no times row for a yielded item' "$t/times-yield-hold.tsv" '$1 ~ /^plain-any@FAKE_HOLD75/' # the seeded plain-any row is the plan's input
  rc=$(GBFAST=1 steal_case yield-card "plain-any	B	a6000	10	$d" -- 'plain-any@FAKE_CARD75=1')
  printf '%s' "$rc" > "$t/rc-yield-card"
  if [ "$(cat "$t/rc-yield-card")" = 1 ] && grep -Eq '^plain-any rc=75 [0-9]+s try=11 lane=[AB] ' "$t/target/c-yield-card/run.log" \
    && grep -Eq '^DONE total=1 red=1 skipped=0 wall=[0-9]+s' "$t/out-yield-card.log"; then
    pass 'yield: a plain card-lock rc 75 keeps today'"'"'s retry (try=11, red)'
  else fail 'yield: a plain card-lock rc 75 keeps today'"'"'s retry (try=11, red)' "rc=$(cat "$t/rc-yield-card") $(tail -1 "$t/out-yield-card.log")"; fi
  rc=$(GBFAST=1 LFLAG=--ledger steal_case yield-lead "plain-any	B	a6000	10	$d" -- 'plain-any@FAKE_HOLD75=leadbatch')
  printf '%s' "$rc" > "$t/rc-yield-lead"
  if [ "$(cat "$t/rc-yield-lead")" = 1 ] && grep -Eq '^plain-any rc=75 [0-9]+s try=11 lane=[AB] ' "$t/target/c-yield-lead/run.log"; then
    pass "yield: the lead's --ledger batch keeps waiting (try=11, red)"
  else fail "yield: the lead's --ledger batch keeps waiting (try=11, red)" "rc=$(cat "$t/rc-yield-lead") $(tail -1 "$t/out-yield-lead.log")"; fi
  want_not "yield: and its DONE line counts no yielded=" "$t/target/c-yield-lead/run.log" 'yielded='
  # Every item skips: nothing runs on the box, so no hold goes up.
  rc=$(FAKE_SKIP=1 LFLAG=--ledger steal_case hold-skip "plain-any	B	a6000	10	$d" -- plain-any)
  printf '%s' "$rc" > "$t/rc-hold-skip"
  rc_ok 'hold: a batch whose every item skips ends green' hold-skip 1
  if [ ! -e "$t/fake-state/hold.log" ]; then pass '  … and puts no hold up'
  else fail '  … and puts no hold up' "$(cat "$t/fake-state/hold.log")"; fi
  # The dry run names the hold it would take and touches nothing; the options' refusals.
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --ledger plain-any 2>&1) \
    || fail 'hold: the dry run of a --ledger batch failed' "$out"
  if grep -Eq '^gate-batch: gpu hold: would put .*batch\.gpuhold up on the box as gb_[A-Za-z0-9_]+, refreshed every 60 s; each item gets BLOOMERY_BATCH_OWNER=gb_' <<< "$out"; then
    pass 'dry run: a --ledger batch prints the hold it would take'
  else fail 'dry run: a --ledger batch prints the hold it would take' "$out"; fi
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --ledger --no-gpu-hold plain-any 2>&1) || true
  grep -q '^gate-batch: gpu hold: off (--no-gpu-hold)$' <<< "$out" && pass '  … --no-gpu-hold says it is off' || fail '  … --no-gpu-hold says it is off' "$out"
  out=$(BLOOMERY_GATE_TIMES=$t/times-hold-up.tsv BLOOMERY_GATE_LEDGER=$t/lead-dry.tsv BLOOMERY_GATE_ROUND_LEDGER=$t/rounds-dry.tsv "${gb[@]}" --dry-run --round-ledger plain-any 2>&1) || true
  grep -q '^gate-batch: gpu hold: none (only a --ledger batch takes it)$' <<< "$out" && pass '  … a --round-ledger batch says none' || fail '  … a --round-ledger batch says none' "$out"
  check 'hold: --no-gpu-hold without --ledger: 64, named' 64 "^gate-batch: --no-gpu-hold is the lead's: only a --ledger batch takes the GPU hold" "${gb[@]}" --dry-run --no-gpu-hold host
  check 'hold: --no-gpu-hold with --round-ledger: 64, named' 64 "^gate-batch: --no-gpu-hold is the lead's" "${gb[@]}" --dry-run --round-ledger --no-gpu-hold host
  check 'hold: a heartbeat period that is not whole seconds: 64, named' 64 "^gate-batch: BLOOMERY_GPU_HOLD_BEAT is whole seconds from 1 .*got 'abc'" \
    env BLOOMERY_GPU_HOLD_BEAT=abc "${gb[@]}" --dry-run host
  # Cold builds: the final try's section only; a local crate is the tree's, a registry or git one is not.
  printf '%s\n' '=== try 1 x' '   Compiling libc v0.2.155' '=== try 2 x' '   Compiling bloomery-gpu v0.1.0 (/root/repo/bloomery/crates/gpu)' > "$t/warm.log"
  printf '%s\n' '=== try 1 x' '   Compiling cuda-core v0.1.0 (https://github.com/x/cuda-oxide?branch=b#abc)' '    Finished `release`' > "$t/cold.log"
  out=$(cold_of "$t/warm.log" 2)/$(cold_of "$t/warm.log" 1)/$(cold_of "$t/cold.log" 1)
  if [ "$out" = 0/1/1 ]; then pass 'cold: a registry or git crate compiled in the final try, and only there'
  else fail 'cold: a registry or git crate compiled in the final try, and only there' "got $out, want 0/1/1"; fi
  local TIMES_FILE=$t/rows.tsv COLD_FILE=$t/rows-cold.tsv P_TKEY=(k-warm k-cold) P_TCARD=(3090 3090)
  times_record 0 A 12 0 && times_record 1 A 900 1
  out="$(cut -f1,4 "$TIMES_FILE" 2>&1 || true) | $(cut -f1,4 "$COLD_FILE" 2>&1 || true)"
  if [ "$out" = "k-warm	12 | k-cold	900" ]; then pass 'cold: its row goes to the cold file, never into the median'
  else fail 'cold: its row goes to the cold file, never into the median' "got '$out'"; fi
  # The waits a times row leaves out: box.sh's guard, the card lock and the V4.1 load lock, final try only.
  printf '%s\n' '=== try 1 x' 'gpu-gate.sh: waited 99 s for the gate lock (3090)' '=== try 2 x' \
    '[guard] 2026-09-27T00:00:00Z quiet on two polls in a row after 60 s: the command starts' \
    'gpu-gate.sh: waited 7 s for the batch hold' \
    'gpu-gate.sh: waited 15 s for the gate lock (3090)' 'gpu-gate.sh: waited 40 s for the V4.1 load lock' > "$t/waits.log"
  out=$(try_waits "$t/waits.log" 2)
  if [ "$out" = 122 ]; then pass 'waits: the guard, the batch hold, the card lock and the V4.1 load lock of the final try'
  else fail 'waits: the guard, the batch hold, the card lock and the V4.1 load lock of the final try' "got $out, want 122"; fi
  # The hold a try's rc 75 names: only gpu-gate.sh's bound line, never its once-a-minute wait line, a
  # card-lock 75 or another try's section.
  printf '%s\n' '=== try 1 x' '[batch-hold] 2026-10-08T00:00:00Z gen_x waits: a landing batch holds the GPUs (/root/bloomery-batch.gpuhold, owner leadbatch, up 900 s, refreshed 0 s ago), holding no lock; 4 s so far' \
    'gpu-gate.sh: no gate lock (3090) was free within 1800 s — contention, not a red gate' \
    '=== try 2 x' 'gpu-gate.sh: gen_x: the batch hold /root/bloomery-batch.gpuhold (owner leadbatch) was still up after 4 s (CARD_BOUND 4 s) — contention, not a red gate (rc 75)' > "$t/hold75.log"
  out="$(hold75_owner "$t/hold75.log" 1)/$(hold75_owner "$t/hold75.log" 2)/$(hold75_owner "$t/hold75.log" 3)"
  if [ "$out" = "/leadbatch/" ]; then pass 'hold75: the owner of the try'"'"'s hold-bound line, and nothing else'
  else fail 'hold75: the owner of the try'"'"'s hold-bound line, and nothing else' "got '$out', want '/leadbatch/'"; fi
  # The group and gpu-gate.sh's lock request must name the same recipes.
  cp "$t/justfile" "$t/justfile.good"
  sed 's/export BLOOMERY_GATE_V41_LOAD=1 \&\& BLOOMERY_GATE_CARD/BLOOMERY_GATE_CARD/' "$t/justfile.good" > "$t/justfile"
  check 'group without the export: a named error' 65 "recipe v41-any: \[group\('v41-load'\)\] without .export BLOOMERY_GATE_V41_LOAD=1." "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' 'plain-v41:' "    ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'export without the group: a named error' 65 'recipe plain-v41: names BLOOMERY_GATE_V41_LOAD without' "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  printf '%s\n' '' "[group('host')]" "[group('solo')]" 'host-solo:' "    ./tools/box.sh 'bash tools/gpu-gate.sh other'" >> "$t/justfile"
  check 'host with solo and a card gate: a named error' 65 "recipe host-solo: \[group\('host'\)\] with \[group\('solo'\)\] and a tools/gpu-gate.sh call" "${gb[@]}" --classes
  cp "$t/justfile.good" "$t/justfile"
  sed '/^weekly-a:$/,$d' "$t/justfile.good" > "$t/justfile" # the weekly recipes close the fixture
  check 'weekly: a justfile with no weekly-* recipe is a named error' 65 'no weekly-\* recipe' "${gb[@]}" --dry-run --weekly
  cp "$t/justfile.good" "$t/justfile"
  # The disk floor: above the floor a run proceeds on the verdict line; below it a real run stops at
  # 69 with the named line (nothing created — the fake HOME's tree would show it), a dry run shows
  # the same verdict and continues; df failing or unparsable is 69 in a dry run too; the override is
  # honored when positive and a named 64 when not; the default OUT lands outside every tree.
  check 'disk: above the floor, the verdict line and a dry run that proceeds' 0 \
    '^gate-batch: disk: [0-9.]+ GiB free on /fake/mount \(for .*\), above the [0-9]+ GiB floor$' "${gb[@]}" --dry-run host
  check 'disk: below the floor, a real run stops with 69 and the named line' 69 \
    '^gate-batch: disk: [0-9.]+ GiB free on /fake/mount .* below the [0-9]+ GiB floor' \
    env HOME=$h/home-low DISK_KB=1024 "${gb[@]}" host
  check "disk: below the floor, the real run's OUT is not created" 69 '^gate-batch: disk:' \
    env HOME=$h/home-low2 DISK_KB=1024 "${gb[@]}" host
  [ ! -e "$h/home-low2" ] || fail "a refused run created its OUT under $h/home-low2"
  check 'disk: a failing df is a named 69 in a dry run too' 69 \
    '^gate-batch: disk: df -Pk .* failed \(rc 1\): df: fake failure$' env DISK_RC=1 "${gb[@]}" --dry-run host
  check 'disk: df output that does not parse is a named 69' 69 'output does not parse' \
    env DISK_BAD=1 "${gb[@]}" --dry-run host
  check 'disk: BLOOMERY_MIN_FREE_GIB=abc is a named 64' 64 \
    "^gate-batch: BLOOMERY_MIN_FREE_GIB='abc' is not a positive integer of GiB$" \
    env BLOOMERY_MIN_FREE_GIB=abc "${gb[@]}" --dry-run host
  check 'disk: BLOOMERY_MIN_FREE_GIB=0 is a named 64 too' 64 'is not a positive integer of GiB' \
    env BLOOMERY_MIN_FREE_GIB=0 "${gb[@]}" --dry-run host
  check 'disk: a positive override raises the floor (the fake disk is below it)' 0 'below the 999999 GiB floor' \
    env BLOOMERY_MIN_FREE_GIB=999999 "${gb[@]}" --dry-run host
  check 'disk: below the floor, --dry-run prints the verdict and exits 0' 0 'below the [0-9]+ GiB floor' \
    env DISK_KB=1024 "${gb[@]}" --dry-run host
  out=$(env HOME=$h/home-dry "${gb[@]}" --dry-run host 2>&1) || fail "a dry run under a fake HOME failed: $out"
  case $out in
    *"logs would go to $h/home-dry/.cache/bloomery/batches/$(basename "$t")/"*) pass 'disk: the default OUT is $HOME/.cache/bloomery/batches/<tree>/<stamp>' ;;
    *) fail 'disk: the default OUT is $HOME/.cache/bloomery/batches/<tree>/<stamp>' "the dry run says: $out" ;;
  esac
  [ ! -e "$h/home-dry" ] || fail "the dry run created its OUT tree under $h/home-dry"
  echo "gate-batch self-test: $((n - bad)) of $n ok"
  [ "$bad" = 0 ]
}


if [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || die "--self-test takes nothing else; $USAGE"
  self_test
  exit $?
fi
OUT='' SRC='' LIST='' DRY=0 LANES=2 LEDGER=0 RERUN=0 LMODE='' TRUST=0 TIER_FLAG='' NOHOLD=0 BUDGET_OK=0
ITEMS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || die "--out needs a directory; $USAGE"; OUT=$2; shift 2 ;;
    --smoke) [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=smoke; shift ;;
    --weekly) [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=weekly; shift ;;
    --classes) [ -z "$SRC" ] || die "--classes takes no items; $USAGE"; SRC=classes; shift ;;
    --list) [ $# -ge 2 ] || die "--list needs a file; $USAGE"
      [ -z "$SRC" ] || die "--smoke, --weekly, --list and items are exclusive; $USAGE"; SRC=list; LIST=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --ledger) [ "$LMODE" != round ] || die "--ledger and --round-ledger are exclusive; $USAGE"
      LEDGER=1 LMODE=lead; shift ;;
    --round-ledger) [ "$LMODE" != lead ] || die "--ledger and --round-ledger are exclusive; $USAGE"
      LEDGER=1 LMODE=round; shift ;;
    --trust-rounds) TRUST=1; shift ;;
    --no-gpu-hold) NOHOLD=1; shift ;;
    --rerun) RERUN=1; shift ;;
    --over-budget-ok) BUDGET_OK=1; shift ;;
    --lanes) [ $# -ge 2 ] || die "--lanes needs 1 or 2; $USAGE"
      case "$2" in 1 | 2) LANES=$2 ;; *) die "--lanes is 1 or 2, got '$2'" ;; esac; shift 2 ;;
    --tier) [ $# -ge 2 ] || die "--tier needs real or fixture; $USAGE"
      TIER_FLAG=$2; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    --*) die "unknown option '$1'; $USAGE" ;;
    *) [ -z "$SRC" ] || [ "$SRC" = items ] || die "--smoke, --weekly, --list, --classes and items are exclusive; $USAGE"
      SRC=items; ITEMS+=("$1"); shift ;;
  esac
done
[ -n "$SRC" ] || die "no items; $USAGE"
if [ "$SRC" = list ]; then
  [ -f "$LIST" ] && [ -r "$LIST" ] || die "--list $LIST: not a readable file"
fi
[ "$SRC" = smoke ] && ITEMS=("${SMOKE[@]}")
[ "$RERUN" = 0 ] || [ "$LEDGER" = 1 ] || die "--rerun needs --ledger or --round-ledger; $USAGE"
[ "$TRUST" = 0 ] || [ "$LMODE" = lead ] || die "--trust-rounds is the lead's: it goes with --ledger (a round's --round-ledger reads the rounds' file already); $USAGE"
[ "$NOHOLD" = 0 ] || [ "$LMODE" = lead ] || die "--no-gpu-hold is the lead's: only a --ledger batch takes the GPU hold, so it goes with --ledger; $USAGE"
# The GPU hold (the header): the lead's batch takes it; its owner names this batch on the box and in each item's box env.
HOLD_ON=0 HOLD_UP=0 HOLD_OWNER='' HOLD_BEAT=${BLOOMERY_GPU_HOLD_BEAT:-60}
case $HOLD_BEAT in
  '' | 0 | *[!0-9]*) die "BLOOMERY_GPU_HOLD_BEAT is whole seconds from 1 (the heartbeat's period, default 60; the box calls a hold stale after 300), got '$HOLD_BEAT'" ;;
esac
if [ -n "${BLOOMERY_PACK_BUDGET+x}" ]; then
  case $BLOOMERY_PACK_BUDGET in
    '' | 0 | *[!0-9]*) die "BLOOMERY_PACK_BUDGET is the pack's byte budget, a positive integer (the header's «pack»), got '$BLOOMERY_PACK_BUDGET'" ;;
  esac
  PACK_BUDGET=$BLOOMERY_PACK_BUDGET
fi
if [ "$LMODE" = lead ] && [ "$NOHOLD" = 0 ]; then
  HOLD_ON=1
  HOLD_OWNER=gb_$(printf '%s' "$(basename "$ROOT")" | tr -c 'A-Za-z0-9' _)_$$_$(date +%s)
fi
command -v just > /dev/null || die "just is not on PATH"
command -v python3 > /dev/null || die "python3 is not on PATH"

# The tier (the header's --tier paragraph). Three places may name it — the flag, a BLOOMERY_TIER entry of BLOOMERY_BOX_ENV
# (where tools/box.sh reads it for the item's model file) and BLOOMERY_TIER in the environment — and they must agree: two
# tiers named at once would plan one and run the other. The fixture tier reaches every item through BLOOMERY_BOX_ENV, so the
# plan, the keys, the key rechecks and the runs all see the same string; the real tier adds nothing.
TIER_BOX=''
for kv in ${BLOOMERY_BOX_ENV:-}; do
  [ "${kv%%=*}" != BLOOMERY_TIER ] || TIER_BOX=${kv#*=}
done
TIER='' TIER_FROM=''
for src in "--tier:$TIER_FLAG" "BLOOMERY_BOX_ENV's BLOOMERY_TIER:$TIER_BOX" "BLOOMERY_TIER in the environment:${BLOOMERY_TIER:-}"; do
  [ -n "${src#*:}" ] || continue
  case "${src#*:}" in
    real | fixture) ;;
    *) die "${src%%:*} is real or fixture, got '${src#*:}'" ;;
  esac
  if [ -z "$TIER" ]; then
    TIER=${src#*:} TIER_FROM=${src%%:*}
  elif [ "$TIER" != "${src#*:}" ]; then
    die "$TIER_FROM names the tier '$TIER' and ${src%%:*} '${src#*:}': name one"
  fi
done
[ -n "$TIER" ] || TIER=real
if [ "$TIER" = fixture ] && [ -z "$TIER_BOX" ]; then
  BLOOMERY_BOX_ENV="${BLOOMERY_BOX_ENV:+$BLOOMERY_BOX_ENV }BLOOMERY_TIER=fixture"
fi

# The plan, in two passes. The first (PYPLAN) validates and classifies every item: one record per
# item, fields split by \x1f — class (A fixed, F balanced, X alone), log stem, recipe, the item's env
# (space separated, no card), ARGS (shell-quoted for eval), the item as written, the reason, the kind
# (fixed, balanced, solo, both-cards, one-lane), the expected seconds and their source, the times key,
# the card candidates (`-` for no card forced; a balanced gpu-gate item has two, 3090 and a6000), the
# times file's card label of each, the family (X and C items), the neighbour tags (pool, big-host) and
# the pack's resource field (X items): `<flag> <bytes>`. REC_FIELDS (below the writer) names the sixteen
# in order, and rec_get is the one reader of a record. The second pass (PYBAL, once the ledger has keyed
# every candidate) picks each item's lane and card.
# (The sources sit in variables: bash 3.2 misparses a heredoc inside $(…) that holds quotes.)
IFS= read -r -d '' PYPLAN << 'PY' || true
import fcntl, json, os, re, shlex, statistics, subprocess, sys
from datetime import datetime

root, lanes, src, listfile, caller_env, times_path, default_s, tier, chain_list = sys.argv[1:10]
raw_items = sys.argv[10:]
chain_solo_real = frozenset(chain_list.split())
just = ["just", "--justfile", root + "/justfile", "--working-directory", root]


def fail(msg):
    print("gate-batch: " + msg, file=sys.stderr)
    sys.exit(65)


if src == "list":
    with open(listfile, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    head = [ln for ln in lines if ln.strip()][:2]
    # `just affected … 2>&1` puts just's echo of the recipe line above the output's `affected:` line.
    if head and head[0].startswith("./tools/affected-gates.sh ") and len(head) > 1:
        head = head[1:]
    if head and head[0].startswith("affected:"):
        raw_items = [ln.split()[0] for ln in lines if ln.startswith(("  gate-", "  weekly-"))]
        where = [f"{listfile} (just affected output)"] * len(raw_items)
    else:
        raw_items, where = [], []
        for i, ln in enumerate(lines, 1):
            if ln.strip() and not ln.lstrip().startswith("#"):
                raw_items.append(ln.strip())
                where.append(f"{listfile}:{i}")
else:
    where = [src] * len(raw_items)
if not raw_items and src not in ("classes", "weekly"):
    fail(f"the {src} input names no item — nothing to run is not a green batch")

proc = subprocess.run(just + ["--dump", "--dump-format", "json"], capture_output=True, text=True)
if proc.returncode != 0:
    fail(f"`just --dump` failed (exit {proc.returncode}): {proc.stderr.strip()}")
recipes = json.loads(proc.stdout)["recipes"]
if src == "weekly":
    raw_items = [n for n in recipes if n.startswith("weekly-")]
    where = ["--weekly"] * len(raw_items)
    if not raw_items:
        fail("--weekly: the justfile has no weekly-* recipe — nothing to run is not a green batch")


def body(name):
    out = []
    for frags in recipes[name]["body"]:
        out.append("".join(f if isinstance(f, str) else "{{}}" for f in frags))
    return "\n".join(out)


def closure(name, seen):
    if name in seen:
        return []
    seen.add(name)
    names = [name]
    for dep in recipes[name]["dependencies"]:
        names += closure(dep["recipe"], seen)
    return names


GPU_GATE = "tools/gpu-gate.sh"
ANY = "BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}"
# The names the structural rule (walk() and takes_lease below) cannot see: timing-card.sh, a library a
# timing runner sources, which takes no lease itself; gpu-ab.py, a Python runner walk() does not read;
# the lease lock named in a recipe's own text, which walk() does not read either. Every runner that calls
# lease_take is found by the walk. Each counts where it runs, as TAKE's name does: the two scripts in
# command position (after env words and an interpreter, `.`, `source`, `exec` or `timeout N`; the start
# of box.sh's quoted command and of a `bash -c` string is a command position too), the lock as a word of
# a command; never in a comment, and never in a message (echo, printf, die, fail, say).
TIMED_RUN = re.compile(r"^[^#\n]*?(?:^|[;&|({!]|\b(?:then|do|else|if|while|until)(?=\s)|(?:tools/box\.sh|\b(?:ba)?sh\s+-c)\s+['\"])\s*"
                       r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:(?:bash|sh|source|\.|exec|python3?|timeout(?:\s+-\S+)*\s+\S+)\s+)*"
                       r"[^\s;&|#'\"]*(tools/ref/timing-card\.sh|gpu-ab\.py)\b", re.M)
TIMED_LOCK = re.compile(r"(?<![A-Za-z0-9_./-])['\"]?(/root/bloomery-(?:cpu|lease)\.lock)\b")


def timed(text):
    """The timed name a recipe's text runs, or None: TIMED_RUN, or the lease lock in a command that is
    not a message, outside a comment."""
    m = TIMED_RUN.search(text)
    if m:
        return m.group(1)
    for line in text.split("\n"):
        for seg in SEGMENT.split(line.split("#", 1)[0]):
            if not MESSAGE.match(seg):
                m = TIMED_LOCK.search(seg)
                if m:
                    return m.group(1)
    return None
BOX_CARD = re.compile(r"\bBLOOMERY_CARD=(both|a6000)\b")
UNLOCKED = re.compile(r"tools/gate\.sh --oxide|cargo oxide (test|run)\b")
SEGMENT = re.compile(r"&&|\|\||;|\||\n")
# A script a recipe names takes the timing lease when it calls lease_take (tools/ref/lease.sh), or
# opens the lease lock for a descriptor and waits on it with flock -w (the runners that take it by
# hand). A probe (lease_free, `flock -s -n <lock> true`: tools/gpu-gate.sh, box.sh's guard) only tests
# the lock and is no take.
SCRIPT = re.compile(r"(?:crates/[A-Za-z0-9_-]+/)?tools/[A-Za-z0-9_./-]+\.sh\b")
# A word in command position: at a line's start or after ; & | ( { ! or a keyword, before any `#`.
CMD = r"^[^#\n]*?(?:^|[;&|({!]|\b(?:then|do|else|if|while|until)(?=\s))\s*"
TAKE = re.compile(CMD + r"lease_take\s*(?=$|[;&|)#])", re.M)
WAIT = re.compile(CMD + r"flock\s+-w\b", re.M)
LEASE_EXEC = re.compile(r"^[^#\n]*\bexec\s+[0-9]+>\s*['\"]?/root/bloomery-(?:cpu|lease)\.lock", re.M)
LEASE_VAR = re.compile(r"^\s*(?:local\s+)?[A-Za-z_][A-Za-z0-9_]*=['\"]?/root/bloomery-(?:cpu|lease)\.lock", re.M)
TAKES = {}


def takes_lease(path):
    """`path:line what` when the tree script at path takes the timing lease, else None."""
    if path not in TAKES:
        hit, full = None, os.path.join(root, path)
        if os.path.isfile(full):
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()

            def line_of(m):
                return text.count("\n", 0, m.start()) + 1

            take = TAKE.search(text)
            opened = LEASE_EXEC.search(text) or LEASE_VAR.search(text)
            wait = WAIT.search(text)
            if take:
                hit = f"{path}:{line_of(take)} calls lease_take"
            elif opened and wait:
                hit = f"{path}:{line_of(opened)} opens the lease lock, :{line_of(wait)} waits on it"
        TAKES[path] = hit
    return TAKES[path]


# The scripts a recipe runs, followed transitively: a tools/…sh path on a script's non-comment line,
# or a relative `[dir/]name.sh` there resolved against that script's directory (ptx-spill-check.sh
# names "${BASH_SOURCE[0]%/*}/ptx-scan.sh"). Each file is read once (a cycle ends there), at most
# WALK_DEPTH hops from the recipe's text; a chain that would go deeper is a named refusal, never a
# walk cut short. Two questions are asked of every file it reaches, by this one walker: does it take
# the timing lease (takes_lease), and does it call tools/gpu-gate.sh — in command position, with the
# call's card form: `BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any}` (balanced), none (the 3090), or
# any other BLOOMERY_GATE_CARD on the call (a form this runner does not read: refused). A mention in
# a message (echo, printf, die, fail, say) is not a script the file runs.
# tools/gpu-gate.sh itself is the runner a call reaches, and its text is not followed; nor is this file's
# (check-recipes runs its --classes): its patterns and messages name the runner in strings the call pattern
# would read as calls. A stub-test
# harness (a directory named *-tests, tools/ref/card-tests) is named in the reason and not read: it
# runs the lease and runner code against a lock of its own, and its fixtures hold both on purpose.
WALK_DEPTH = 6
BARE = re.compile(r"([A-Za-z0-9_.-][A-Za-z0-9_./-]*\.(?:sh|bash))\b")
GATE_CALL = re.compile(CMD + r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:bash\s+|exec\s+)?[^\s;&|#'\"]*gpu-gate\.sh\b", re.M)
BATCH = "tools/gate-batch.sh"
RUNNER_SELF = {GPU_GATE, BATCH}
MESSAGE = re.compile(r"^[\s{(!]*(?:echo|printf|die|fail|say)\b")


def is_harness(path):
    return any(part.endswith("-tests") for part in path.split("/")[:-1])


def script_refs(path, text):
    """The tree scripts path's non-comment lines name, in order — outside a message: a segment whose
    command is echo, printf or a message function (die, fail, say) names a script it does not run."""
    out, sdir = [], os.path.dirname(path)
    for line in text.split("\n"):
        st = line.strip()
        if not st or st.startswith("#"):
            continue
        for seg in SEGMENT.split(line):
            if MESSAGE.match(seg):
                continue
            for q in SCRIPT.findall(seg) + [os.path.join(sdir, b) for b in BARE.findall(seg)]:
                q = os.path.normpath(q)
                if q.endswith((".sh", ".bash")) and os.path.isfile(os.path.join(root, q)):
                    out.append(q)
    return list(dict.fromkeys(out))


def walk(start):
    """{calls: [(form, where)], take: (chain, hit) or None, harness: [paths], error: str or None} for the
    scripts named in start, breadth first."""
    res = {"calls": [], "take": None, "harness": [], "error": None}
    parent, depth, queue = {}, {}, []
    for s in dict.fromkeys(os.path.normpath(q) for q in start):
        if os.path.isfile(os.path.join(root, s)) and s not in depth:
            depth[s] = 1
            queue.append(s)

    def chain(s):
        out = [s]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        return " <- ".join(out)

    while queue:
        s = queue.pop(0)
        if s in RUNNER_SELF or not s.endswith((".sh", ".bash")):
            continue
        if is_harness(s):
            res["harness"].append(s)
            continue
        with open(os.path.join(root, s), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        hit = takes_lease(s)
        if hit and res["take"] is None:
            res["take"] = (chain(s), hit)
        for m in GATE_CALL.finditer(text):
            where = f"{s}:{text.count(chr(10), 0, m.start()) + 1}"
            call = m.group(0)
            if ANY in call:
                res["calls"].append(("any", where))
            elif "BLOOMERY_GATE_CARD" in call:
                res["calls"].append(("unread", f"{where}: {call.strip()}"))
            else:
                res["calls"].append(("3090", where))
        for q in script_refs(s, text):
            if q in depth:
                continue
            if depth[s] >= WALK_DEPTH:
                res["error"] = (f"the scripts it runs go deeper than WALK_DEPTH = {WALK_DEPTH} hops "
                                f"({q} <- {chain(s)}): raise WALK_DEPTH in tools/gate-batch.sh")
                return res
            depth[q], parent[q] = depth[s] + 1, s
            queue.append(q)
    return res


GATE_FORMS = {}


def classify(name):
    """(lane, reason) for a recipe and its dependencies: lane A, B or X, or T (a timed recipe) or R
    (refused for another reason), which do not run in a batch."""
    lanes_seen, reasons, forms = set(), [], set()
    for n in closure(name, set()):
        text = body(n)
        if BATCH in SCRIPT.findall(text):
            return "R", (f"{n} runs {BATCH}: a batch inside a batch runs lanes of its own beside this batch's, "
                         "on the same cards; list its items instead")
        hit = timed(text)
        if hit:
            return "T", f"{n} runs {hit}: a timed recipe does not run in a gate batch"
        w = walk(SCRIPT.findall(text))
        if w["error"]:
            return "R", f"{n}: {w['error']}"
        if w["take"]:
            path, hit = w["take"]
            return "T", f"{n} runs {path}, which takes the timing lease ({hit}): a timed recipe does not run in a gate batch"
        for form, where in w["calls"]:
            if form == "unread":
                return "R", f"{n}: a script sets BLOOMERY_GATE_CARD in a form this runner does not read: {where}"
            forms.add(form)
            if form == "any":
                lanes_seen.add("B")
                reasons.append(f"{n}: gpu-gate.sh any ({where})")
            else:
                lanes_seen.add("A")
                reasons.append(f"{n}: gpu-gate.sh 3090 ({where})")
        reasons += [f"{n}: {h} not read (a stub-test harness)" for h in w["harness"]]
        m = BOX_CARD.search(text)
        if m:
            lanes_seen.add("X")
            reasons.append(f"{n}: {m.group(0)} (box.sh picks the cards, and the gate runner takes their locks)")
            continue
        oxide = "cargo oxide" in text
        for seg in SEGMENT.split(text):
            if GPU_GATE in seg:
                if ANY in seg:
                    lanes_seen.add("B")
                    forms.add("any")
                    reasons.append(f"{n}: gpu-gate.sh any")
                elif "BLOOMERY_GATE_CARD" in seg:
                    return "R", f"{n}: a gpu-gate.sh call sets BLOOMERY_GATE_CARD in a form this runner does not read: {seg.strip()}"
                else:
                    lanes_seen.add("A")
                    forms.add("3090")
                    reasons.append(f"{n}: gpu-gate.sh 3090")
            elif UNLOCKED.search(seg):
                lanes_seen.add("A")
                reasons.append(f"{n}: {UNLOCKED.search(seg).group(0)} (device code, no gate lock)")
            elif oxide and "target/release/" in seg and "host-gate.sh" not in seg:
                lanes_seen.add("A")
                reasons.append(f"{n}: direct target/release run (box env 3090 pin, no gate lock)")
    GATE_FORMS[name] = forms
    why = "; ".join(dict.fromkeys(r for r in reasons))
    for lane in ("X", "A", "B"):
        if lane in lanes_seen:
            return lane, why
    return "B", "no device code" + (f" ({why})" if why else "")


# The groups this runner reads from the just attribute `[group('…')]`. A value it does not know is a
# named error wherever it sits in the justfile (check-recipes runs `--smoke --dry-run`, so it fails
# there too): a group is a scheduling class here, and a typo must not drop a gate out of its class.
GROUPS = {
    "solo": "alone in lane X, after both lanes",
    "solo-real": "alone in lane X in the real tier, balanced over lanes A and B in the fixture tier",
    "v41-load": "the chain (class C): one member at a time, batch-wide (solo wins)",
    "host": "balanced over lanes A and B, no card forced: device-crate tests that open no card",
    "pool": "never beside a chain item: its host tier spins a worker pool on the cores a big load's pool is pinned to",
    "big-host": "never beside a chain item: it locks or allocates more host memory than fits beside a chain item's host set",
}
unknown = [
    f"recipe {n}: [group({a['group']!r})] is not a group this runner knows ({', '.join(GROUPS)}) — "
    "a group is a scheduling class here: add it to GROUPS in tools/gate-batch.sh with its lane rule, or remove it"
    for n in sorted(recipes)
    for a in recipes[n]["attributes"]
    if isinstance(a, dict) and "group" in a and a["group"] not in GROUPS
]
if unknown:
    for e in unknown:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(unknown)} unknown group attribute(s) in the justfile; nothing ran")

# The v41-load group has two readers: this runner's lane, and tools/gpu-gate.sh's box-wide V4.1 load lock,
# which a recipe asks for by exporting BLOOMERY_GATE_V41_LOAD=1 in its box command. The two must name the
# same recipes: a member without the export loads V4.1 beside another tree's load; an export without the
# group is a load this runner would balance onto lane B.
V41_EXPORT = "export BLOOMERY_GATE_V41_LOAD=1 && "
mismatch = []
for n in sorted(recipes):
    member = {"group": "v41-load"} in recipes[n]["attributes"]
    text = body(n)
    if member and V41_EXPORT not in text:
        mismatch.append(f"recipe {n}: [group('v41-load')] without `{V41_EXPORT.strip(' &')}` in its box command "
                        "— tools/gpu-gate.sh would not take the V4.1 load lock")
    elif not member and "BLOOMERY_GATE_V41_LOAD" in text:
        mismatch.append(f"recipe {n}: names BLOOMERY_GATE_V41_LOAD without [group('v41-load')]")
if mismatch:
    for e in mismatch:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(mismatch)} v41-load recipe(s) whose group and export disagree; nothing ran")

def group_of(name, group):
    """The recipes of name's closure that carry [group('<group>')]."""
    return [n for n in closure(name, set()) if {"group": group} in recipes[n]["attributes"]]


# The real-only class (the header's paragraph) is a call in the box command: `bash tools/ref/real-only.sh <recipe> && …`, the first
# command of every box.sh command of the recipe. The box side runs it (it stops the fixture tier by name); this runner reads it from
# the text. A call that names another recipe would make the refusal name the wrong gate; a box command that does not open with it,
# or a call outside a single-quoted box command, would leave a command the refusal does not guard.
REAL_ONLY = "tools/ref/real-only.sh"
REAL_ONLY_CALL = re.compile(r"\bbash " + re.escape(REAL_ONLY) + r"\s+(\S+)")
BOX_OPEN = "tools/box.sh '"


# The budget refusal (a suite whose slowest single item is past 0.75 × the bound that kills it is a
# named refusal, not an overflow): the plan reads each recipe's bounded calls — a command segment
# of its closure's text, or of a tree script that text names, holding tools/gate.sh,
# tools/gpu-gate.sh or tools/host-gate.sh — with the segment's BLOOMERY_GATE_BOUND (a literal, or
# the literal inside ${BLOOMERY_GATE_BOUND:-n}; unset is the runners' 900, gate-bound.sh). One
# bounded call with a parsed bound: the item is checked against 0.75 × that bound. No call, several
# calls, or a bound the text does not spell (an arithmetic): unchecked, named in the dry run.
BOUND_SET = re.compile(r"BLOOMERY_GATE_BOUND=(?:([1-9][0-9]*)|\$\{BLOOMERY_GATE_BOUND:-([1-9][0-9]*)\})")
_budget_texts = {}


def budget_text(path):
    if path not in _budget_texts:
        try:
            with open(os.path.join(root, path), encoding="utf-8", errors="replace") as fh:
                _budget_texts[path] = fh.read()
        except OSError:
            _budget_texts[path] = ""
    return _budget_texts[path]


def bounded_calls(name):
    """[(where, bound)] of the bounded runner calls in name's closure and the scripts it names,
    followed transitively (script_refs, WALK_DEPTH hops — a bound two scripts deep is as killing as
    one in the recipe's text): the segment's BLOOMERY_GATE_BOUND literal, the runners' 900 when the
    call sets none, or -1 when it sets one the text does not spell (an arithmetic) — the item is
    then unchecked."""
    follow_scripts = True

    def bound_of(seg):
        m = BOUND_SET.search(seg)
        if m:
            return int(m.group(1) or m.group(2))
        return -1 if "BLOOMERY_GATE_BOUND" in seg else 900

    head = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(?:bash\s+|exec\s+|\./)?")
    box = re.compile(r"^[^']*?tools/box\.sh\s+'")

    def calls_of(text, where):
        for line in text.split("\n"):
            for seg in SEGMENT.split(line.split("#", 1)[0]):
                if MESSAGE.match(seg):
                    continue
                stripped = head.sub("", box.sub("", seg.strip()))
                if stripped.startswith(("tools/gpu-gate.sh", "tools/gate.sh", "tools/host-gate.sh")):
                    yield where, bound_of(seg)

    out = []
    for n in closure(name, set()):
        text = body(n)
        out += calls_of(text, n)
        if not follow_scripts:
            continue
        seen, queue = {}, {}
        for s in dict.fromkeys(os.path.normpath(q) for q in SCRIPT.findall(text)):
            if os.path.isfile(os.path.join(root, s)) and s not in seen:
                seen[s] = 1
                queue[s] = 0
        while queue:
            s = next(iter(queue))
            del queue[s]
            if s in RUNNER_SELF or s.endswith("tools/box.sh"):
                continue  # box.sh is the transport, not a runner of gates
            stext = budget_text(s)
            out += calls_of(stext, s)
            for q in script_refs(s, stext):
                if q in seen:
                    continue
                if seen[s] >= WALK_DEPTH:
                    fail(f"{n}: the scripts it runs go deeper than WALK_DEPTH = {WALK_DEPTH} hops ({q} <- {s}) "
                         "in the budget scan: raise WALK_DEPTH in tools/gate-batch.sh")
                seen[q], queue[q] = seen[s] + 1, 0
    return out


def defer_tags(name):
    """What defers a recipe in the fixture tier, for the plan's reason text."""
    tags = [f"{n}: real-only call" for n in real_only_of(name)]
    if tier == "fixture" and not tags and no_fixture(name):
        tags.append(f"family {no_fixture(name)}: no row in the fixture table of tools/ref/ref-paths.sh (box.sh exits 66)")
    return tags


def real_only_of(name):
    """The recipes of name's closure whose own text calls the real-only guard."""
    return [n for n in closure(name, set()) if REAL_ONLY_CALL.search(body(n))]


ro_bad = []
for n in sorted(recipes):
    text = body(n)
    calls = REAL_ONLY_CALL.findall(text)
    if not calls:
        continue
    first = f"bash {REAL_ONLY} {n} && "
    if not os.path.isfile(os.path.join(root, REAL_ONLY)):
        ro_bad.append(f"recipe {n}: calls {REAL_ONLY}, which is not in the tree")
    wrong = sorted({c for c in calls if c != n})
    if wrong:
        ro_bad.append(f"recipe {n}: its real-only call names {', '.join(wrong)}, not itself — write `{first.rstrip(' &')}`")
    opens = [m.end() for m in re.finditer(re.escape(BOX_OPEN), text)]
    loose = [e for e in opens if not text.startswith(first, e)]
    if not opens or loose or len(calls) != len(opens):
        ro_bad.append(f"recipe {n}: a real-only recipe opens every box.sh command with `{first.rstrip(' &')} && …` — "
                      f"{len(opens)} single-quoted box.sh command(s), {len(calls)} call(s), {len(loose)} not opening with it")
if ro_bad:
    for e in ro_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(ro_bad)} real-only recipe(s) the class could not hold; nothing ran")


host_bad = []
for n in sorted(recipes):
    if {"group": "host"} not in recipes[n]["attributes"]:
        continue
    text = "\n".join(body(m) for m in closure(n, set()))
    why = [f"[group('{g}')]" for g in ("solo", "solo-real", "v41-load") if group_of(n, g)]
    why += [w for w, hit in (("a tools/gpu-gate.sh call", GPU_GATE in text), ("a box.sh card pick", BOX_CARD.search(text))) if hit]
    if why:
        host_bad.append(f"recipe {n}: [group('host')] with {' and '.join(why)} — a host recipe opens no card and runs beside other items")
if host_bad:
    for e in host_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(host_bad)} host recipe(s) that open a card or run alone; nothing ran")


# pool and big-host (the header): they hold a lane item off the chain's side, so they cannot sit on
# a recipe whose class owns its placement — solo, solo-real (lane X) or v41-load (the chain itself,
# whose members spin the very pools and hold the very host sets the rule guards).
pb_bad = []
for n in sorted(recipes):
    held = [g for g in ("pool", "big-host") if {"group": g} in recipes[n]["attributes"]]
    if not held:
        continue
    why = [f"[group('{g}')]" for g in ("solo", "solo-real", "v41-load") if group_of(n, g)]
    if why:
        pb_bad.append(f"recipe {n}: [group('{held[0]}')] with {' and '.join(why)} — the group holds a lane item off a chain item's side")
if pb_bad:
    for e in pb_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(pb_bad)} pool/big-host recipe(s) the class could not hold; nothing ran")


# CHAIN_SOLO_REAL (the shell constant above): the solo-real recipes that join the chain. A name the
# justfile holds must carry the groups the chain needs; a name it does not hold is ignored (the
# self-test's fixture justfile holds its own recipes).
csr_bad = []
for n in chain_solo_real:
    if n not in recipes:
        continue
    why = []
    if {"group": "solo-real"} not in recipes[n]["attributes"]:
        why.append("no [group('solo-real')] — the list names the recipes whose own reason let them join the chain")
    if not group_of(n, "v41-load"):
        why.append("no [group('v41-load')] — a chain member loads the whole model under the V4.1 lock")
    if why:
        csr_bad.append(f"recipe {n}: CHAIN_SOLO_REAL names it with {' and '.join(why)}")
if csr_bad:
    for e in csr_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(csr_bad)} CHAIN_SOLO_REAL recipe(s) the class could not hold; nothing ran")


# The pack's resource table (the header's «pack»): one row per solo/solo-real recipe — its flags
# (m: a clause measures the host) and its host bytes (the plan's HostNeed, derived). Read before
# anything is placed: a solo recipe with no row is a named error (a recipe that runs in the pack
# declares its bytes), and so is a row for a recipe that is not solo (stale rows must not linger).
RES_TABLE = "tools/gate-batch-resources.tsv"


def load_resources():
    """{name: (flags, bytes)} of the resource table; a row that does not parse is a named error."""
    path = os.path.join(root, RES_TABLE)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        fail(f"{RES_TABLE} cannot be read ({e.strerror}) — the pack's rows are the solo recipes' bytes and flags")
    got, bad = {}, []
    for n, ln in enumerate(text.split("\n"), 1):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) != 3 or not re.fullmatch(r"[A-Za-z0-9_-]+", f[0]) or f[1] not in ("m", "-") \
                or not re.fullmatch(r"[0-9]+", f[2]):
            bad.append(f"{RES_TABLE}:{n}: a row is '<recipe>\\t<m|->\\t<bytes>', not {ln!r}")
            continue
        got[f[0]] = (f[1], int(f[2]))
    if bad:
        for e in bad:
            print("gate-batch: " + e, file=sys.stderr)
        fail(f"{len(bad)} malformed resource row(s); nothing ran")
    return got


RESOURCES = load_resources()


def chain_member(name, box):
    """Whether the recipe joins the chain (class C): a v41-load recipe that is not solo and takes no
    both-cards pick, or a solo-real recipe CHAIN_SOLO_REAL names. box is its box.sh pick, if any."""
    if box == "both" or group_of(name, "solo"):
        return False
    if group_of(name, "solo-real"):
        return name in chain_solo_real
    return bool(group_of(name, "v41-load"))


def solo_any(name):
    """Whether the recipe is solo or solo-real, whatever the tier: what its own arms are judged by."""
    return group_of(name, "solo") or group_of(name, "solo-real")


def solo_of(name):
    """The recipes that keep name alone in this tier: [group('solo')], and [group('solo-real')] in the real tier."""
    return group_of(name, "solo") + (group_of(name, "solo-real") if tier == "real" else [])


# A solo-real recipe is balanced in the fixture tier, so it must be one the balance can place: every gpu-gate.sh call of it takes
# the `any` card, or the recipe picks the A6000 alone through box.sh (one card: the fixture tier fixes it to lane B), and nothing
# keeps it alone in either tier — a `solo` beside it, a pick of both cards (it stays alone whatever the tier) or a 3090-only call
# would each make the second tier's placement the first's, and the group a name for nothing. A recipe with no fixture conversion
# (real-only) has no second tier's placement to choose.
sr_bad = []
for n in sorted(recipes):
    if {"group": "solo-real"} not in recipes[n]["attributes"]:
        continue
    why = []
    if group_of(n, "solo"):
        why.append("[group('solo')] — a recipe is alone in both tiers (solo) or in the real one (solo-real)")
    if real_only_of(n):
        why.append("a real-only call — a recipe with no fixture conversion has no fixture-tier placement (real-only, or solo-real)")
    text = "\n".join(body(m) for m in closure(n, set()))
    pick = BOX_CARD.search(text)
    if pick and pick.group(1) == "both":
        why.append(f"{pick.group(0)} — a box.sh pick of both cards keeps a recipe alone in both tiers (use solo)")
    if not why:
        lane, reason = classify(n)
        if lane in ("T", "R"):
            why.append(reason)
        elif pick:
            if GPU_GATE not in text or "3090" in GATE_FORMS[n]:
                seen = ", ".join(sorted(GATE_FORMS[n])) or "none"
                why.append(f"{pick.group(0)} with no tools/gpu-gate.sh call that takes the pick (script forms: {seen}) — "
                           "the fixture tier fixes the recipe to the A6000's lane")
        elif GATE_FORMS[n] != {"any"}:
            seen = ", ".join(sorted(GATE_FORMS[n])) or "none"
            why.append(f"its tools/gpu-gate.sh calls are not all `{ANY}` (forms: {seen}) — the fixture tier balances it on the any card")
    if why:
        sr_bad.append(f"recipe {n}: [group('solo-real')] with " + "; ".join(why))
if sr_bad:
    for e in sr_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(sr_bad)} solo-real recipe(s) the fixture tier could not place; nothing ran")


# The pack's table validation (the header's «pack»; after the solo-real placement checks above, so
# a recipe several checks could name fails by its placement error first): every row's recipe is a
# solo one, and every solo recipe holds a row — a recipe that runs in the pack declares its bytes.
res_bad = [f"recipe {n}: a resource table row for a recipe that is not [group('solo')]/[group('solo-real')] — "
           "the table serves the pack, whose items are the solo ones"
           for n in sorted(RESOURCES)
           if n in recipes
           and {"group": "solo"} not in recipes[n]["attributes"]
           and {"group": "solo-real"} not in recipes[n]["attributes"]]
res_bad += [f"recipe {n}: no resource table row in {RES_TABLE} — a solo recipe runs in the pack and declares "
            "its bytes (<recipe>\\t<m|->\\t<bytes>)"
            for n in sorted(recipes)
            if ({"group": "solo"} in recipes[n]["attributes"] or {"group": "solo-real"} in recipes[n]["attributes"])
           and n not in RESOURCES]
if res_bad:
    for e in res_bad:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(res_bad)} resource table row(s) the pack could not hold; nothing ran")


# A solo recipe's arm run through another recipe: a non-solo item whose ARGS hand a solo recipe's
# gpu-gate.sh binary one of the flags that solo recipe passes it would run that arm in a lane beside
# other loads, and without whatever else the solo recipe sets around it. Read from the justfile: per solo recipe, the binary of each
# gpu-gate.sh call and the literal --flags after it.
CALL_ARGS = re.compile(r"gpu-gate\.sh\s+([A-Za-z0-9_-]+)((?:\s+[^\s;&|]+)*)")


def gate_calls(name):
    """[(binary, {literal --flags})] of the gpu-gate.sh calls in name's own text."""
    return [(m.group(1), set(re.findall(r"(?<!\S)(--[A-Za-z0-9][A-Za-z0-9-]*)", m.group(2))))
            for m in CALL_ARGS.finditer(body(name))]


SOLO_ARMS = {}
for _n in recipes:
    if {"group": "solo"} in recipes[_n]["attributes"] or {"group": "solo-real"} in recipes[_n]["attributes"]:
        for _bin, _flags in gate_calls(_n):
            for _f in _flags:
                SOLO_ARMS.setdefault((_bin, _f), _n)


def solo_arm(name, argv):
    """The (flag, solo recipe) whose arm the item's ARGS select, or None."""
    if solo_any(name):
        return None
    for n in closure(name, set()):
        for b, _ in gate_calls(n):
            for a in argv:
                if (b, a) in SOLO_ARMS:
                    return a, b, SOLO_ARMS[(b, a)]
    return None


ITEM = re.compile(r"^([A-Za-z0-9_-]+)(?:@([^:]*))?(?::(.*))?$")
VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TIMES_LANES, TIMES_CARDS = ("A", "B", "X"), ("3090", "a6000", "both", "none", "any")
DEFAULT_S = int(default_s)


def load_times(path):
    """{item: [seconds, …]} in file order. A missing file is empty; a malformed row is a named error."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return {}
    except OSError as e:
        fail(f"the times file {path} cannot be read: {e.strerror}")
    chunks = []
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
    except OSError as e:
        fail(f"the times file {path} cannot be read: {e.strerror}")
    finally:
        os.close(fd)
    fix = "this runner writes the file; fix or remove the row"
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as e:
        fail(f"the times file {path} is not UTF-8 ({e}) — {fix}")
    if text and not text.endswith("\n"):
        fail(f"{path}: the last row has no newline (a torn write) — {fix}")
    rows = {}
    for n, ln in enumerate(text.split("\n")[:-1], 1):
        f, bad = ln.split("\t"), None
        if len(f) != 5:
            bad = f"{len(f)} fields, not 5"
        elif not ITEM.match(f[0]):
            bad = f"item {f[0]!r} is not NAME[@K=V,…][:ARGS]"
        elif f[1] not in TIMES_LANES:
            bad = f"lane {f[1]!r} is not A, B or X"
        elif f[2] not in TIMES_CARDS:
            bad = f"card {f[2]!r} is not one of {', '.join(TIMES_CARDS)}"
        elif not re.fullmatch(r"[0-9]+", f[3]):
            bad = f"seconds {f[3]!r} is not a whole number"
        else:
            try:
                datetime.strptime(f[4], "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                bad = f"date {f[4]!r} is not YYYY-MM-DDTHH:MM:SS+hhmm"
        if bad:
            fail(f"{path}:{n}: a malformed times row ({bad}): {ln!r} — {fix}")
        rows.setdefault(f[0], []).append(int(f[3]))
    return rows


TIMES = load_times(times_path)


def expected(key):
    """(seconds, source): the median of the item's last 5 rows, else the default."""
    got = TIMES.get(key, [])[-5:]
    if not got:
        return DEFAULT_S, "default"
    return int(statistics.median(got) + 0.5), f"median of {len(got)} row{'s' if len(got) > 1 else ''}"


for kv in caller_env.split():
    if lanes == "2" and kv.split("=", 1)[0] == "BLOOMERY_GATE_CARD":
        fail("BLOOMERY_BOX_ENV sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")

# An item's model family: the profile tools/box.sh loads for it — the BLOOMERY_MODEL a recipe of its
# closure sets on its box line, else tools/ref/ref-paths.sh's default. Lane X runs FIRST_FAMILY first:
# lane A's last loads are V4.1's (the v41-load group), so its host set is still in the page cache when lane
# X starts, and every family switch after that refetches a host set once.
FIRST_FAMILY = "deepseek41"
PROFILE_SET = re.compile(r"\bBLOOMERY_MODEL=([A-Za-z0-9_]+)\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*\./tools/box\.sh\b")
DEFAULT_PROFILE = []


def default_profile():
    """ref-paths.sh's default profile, read once, when a recipe sets none."""
    if not DEFAULT_PROFILE:
        path = os.path.join(root, "tools/ref/ref-paths.sh")
        try:
            with open(path, encoding="utf-8") as fh:
                got = re.findall(r'^: "\$\{BLOOMERY_MODEL:=\$\{BLOOMERY_REF_MODEL_PROFILE:-([A-Za-z0-9_]+)\}\}"$', fh.read(), re.M)
        except OSError as e:
            fail(f"{path} cannot be read ({e.strerror}): lane X's family order reads the default profile there")
        if len(got) != 1:
            fail(f"{path} holds {len(got)} default-profile lines, not one: lane X's family order reads the default there")
        DEFAULT_PROFILE.append(got[0])
    return DEFAULT_PROFILE[0]


def family(name):
    for n in closure(name, set()):
        m = PROFILE_SET.search(body(n))
        if m:
            return m.group(1)
    return default_profile()


FIXTURE_TABLE = []


def fixture_families():
    """The profiles with a row in the fixture tier's table of tools/ref/ref-paths.sh (`<profiles> ) __fixture_dir=<dir> ;;`): a
    fixture directory, or `self` (the family's real file is small and stands). Every other profile falls to the table's `*)` row,
    `none`, where tools/box.sh exits 66 under the fixture tier. The table is that file's, read here; a table this reader does not
    understand is a named error."""
    if not FIXTURE_TABLE:
        path = os.path.join(root, "tools/ref/ref-paths.sh")
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            fail(f"{path} cannot be read ({e.strerror}): the fixture tier's family table is read there")
        m = re.search(r'^\s*case "\$BLOOMERY_MODEL" in\n((?:\s*[^\n]*\) __fixture_dir=[a-z0-9_-]+ ;;\n)+)\s*esac$', text, re.M)
        if not m:
            fail(f"{path}: no fixture table (`case \"$BLOOMERY_MODEL\" in … ) __fixture_dir=<dir> ;; … esac`) — tools/gate-batch.sh reads which "
                 "families have a fixture there")
        have, default = set(), None
        for row in m.group(1).strip().split("\n"):
            pats, dirname = re.fullmatch(r"\s*(.*?)\) __fixture_dir=([a-z0-9_-]+) ;;", row).groups()
            for pat in (x.strip() for x in pats.split("|")):
                if pat == "*":
                    default = dirname
                elif dirname != "none":
                    have.add(pat)
        if default != "none":
            fail(f"{path}: the fixture table's `*)` row is {default!r}, not none: a family without a row must be one with no fixture")
        FIXTURE_TABLE.append(have)
    return FIXTURE_TABLE[0]


def no_fixture(name):
    """The profile of name's family when the fixture tier has none for it (box.sh would exit 66), else None."""
    fam = family(name)
    return None if fam in fixture_families() else fam


def placement(name, lane, lanes):
    """(class, kind, card candidates, their times-file labels) of a recipe that runs in a batch."""
    names = closure(name, set())
    uses_gpu_gate = bool(GATE_FORMS[name])
    box = next((m.group(1) for m in (BOX_CARD.search(body(n)) for n in names) if m), None)
    if tier == "fixture" and real_only_of(name):
        # No fixture conversion: the item is deferred to the real tier, in no lane (class D; the shell side keeps it apart).
        return "D", "real-only", ["-"], ["none"]
    if tier == "fixture" and no_fixture(name):
        # The family has no fixture (box.sh would exit 66, a red): deferred in the same way, under its own kind.
        return "D", "no-fixture", ["-"], ["none"]
    if lanes == "1":
        if box:
            labels = [box]
        elif uses_gpu_gate:
            labels = ["any" if "any" in GATE_FORMS[name] else "3090"]
        else:
            labels = ["3090" if lane == "A" else "none"]
        return "A", "one-lane", ["-"], labels
    if tier == "fixture" and box == "a6000" and group_of(name, "solo-real"):
        # One card by box.sh's pick: the fixture tier fixes it to the A6000's lane (class B; no lane steals it).
        return "B", "fixed", ["-"], [box]
    if tier == "real" and lanes == "2" and chain_member(name, box):
        # Class C (the header): the chain, one member at a time batch-wide. The card candidates are
        # decided here; which lane runs the member, on which candidate, is the run's.
        if box == "a6000":
            return "C", "chain", ["-"], ["a6000"]  # box.sh picks the card; no card forced (the recipe's own key)
        if not uses_gpu_gate:
            return "C", "chain", ["-"], ["3090"]  # device code with no gate lock: the box env's 3090 pin
        if GATE_FORMS[name] == {"any"}:
            return "C", "chain", ["3090", "a6000"], ["3090", "a6000"]  # one ledger key per candidate
        return "C", "chain", ["3090"], ["3090"]  # a call without the any form: gpu-gate.sh's default 3090
    if solo_of(name) or lane == "X":
        kind = "solo" if solo_of(name) else "both-cards"
        if box:
            return "X", kind, ["-"], [box]
        if uses_gpu_gate:
            if "any" in GATE_FORMS[name]:
                # The pack takes a free card (the header's «pack»): both cards are candidates, one
                # ledger key each, the item's key and times row following the card it took.
                return "X", kind, ["3090", "a6000"], ["3090", "a6000"]
            return "X", kind, ["3090" if lane == "A" else "a6000"], ["3090" if lane == "A" else "a6000"]
        return "X", kind, ["-"], ["3090" if lane == "A" else "none"]
    if group_of(name, "host"):
        # Device code with no card: its lane is a scheduling choice, and the balance makes it.
        return "F", "balanced", ["-"], ["none"]
    if group_of(name, "v41-load") and tier == "real":
        # One lane: two V4.1 loads at once evict each other's host set from the page cache. Lane A,
        # where the 3090-only (`--place gate`) loads already are; an `any` member runs on the 3090.
        # Not in the fixture tier: a fixture load takes no V4.1 load lock and has no such host set.
        return "A", "fixed", (["3090"] if uses_gpu_gate else ["-"]), ["3090"]
    if lane == "A":
        return "A", "fixed", (["3090"] if uses_gpu_gate else ["-"]), ["3090"]
    if uses_gpu_gate:
        return "F", "balanced", ["3090", "a6000"], ["3090", "a6000"]
    return "F", "balanced", ["-"], ["none"]


# --classes: every recipe of the justfile, one line each, and nothing else runs:
#   <recipe>\t<class>\t<kind>\t<cards>\t<reason>
# class A (fixed, the 3090), F (balanced), X (alone), T (timed: refused in a batch) or R (refused for
# another reason); cards the candidates, `-` for none forced. tools/check-recipes.sh reads the T lines.
if src == "classes":
    for name in sorted(recipes):
        lane, reason = classify(name)
        if lane in ("T", "R"):
            print("\t".join([name, lane, "timed" if lane == "T" else "refused", "-", reason]))
        else:
            cls, kind, cards, labels = placement(name, lane, "2")
            tags = [f"{n}: [group('{g}')]" for g in ("solo", "solo-real", "v41-load", "host", "pool", "big-host") for n in group_of(name, g)]
            tags += defer_tags(name)
            if cls == "C":
                cards = labels  # a chain member's candidates are its labels (a pick forces no card)
            if cls == "X":
                rflags, rbytes = RESOURCES.get(name, ("-", 0))
                if tier == "fixture":
                    rbytes = 0
                tags.append(f"pack: {rflags}, {rbytes} B")
            print("\t".join([name, cls, kind, ",".join(cards), "; ".join(tags + [reason])]))
    sys.exit(0)

counts, errors, out, fams = {}, [], [], []
budget_over, budget_unchecked = [], []
for item, at in zip(raw_items, where):
    m = ITEM.match(item)
    if not m:
        errors.append(f"{at}: malformed item {item!r} (NAME[@K=V,…][:ARGS])")
        continue
    name, env, args = m.group(1), m.group(2), m.group(3)
    if name not in recipes:
        errors.append(f"{at}: {name!r} is not a recipe in the justfile")
        continue
    envs = []
    if env is not None:
        for kv in env.split(","):
            k, eq, v = kv.partition("=")
            if not eq or not VAR.match(k) or re.search(r"\s", v):
                errors.append(f"{at}: env entry {kv!r} of {item!r} is not K=V (no spaces)")
            elif lanes == "2" and k == "BLOOMERY_GATE_CARD":
                errors.append(f"{at}: {item!r} sets BLOOMERY_GATE_CARD; with two lanes the card is the runner's (use --lanes 1)")
            elif k == "BLOOMERY_TIER":
                errors.append(f"{at}: {item!r} sets BLOOMERY_TIER; the tier is the batch's (--tier), as it decides the item's lane")
            else:
                envs.append(f"{k}={v}")
    argv = []
    if args is not None:
        if re.search(r"@[A-Za-z_][A-Za-z0-9_]*=", args):
            errors.append(f"{at}: {item!r}: env goes before ARGS (NAME@K=V:ARGS)")
            continue
        try:
            argv = shlex.split(args)
        except ValueError as e:
            errors.append(f"{at}: ARGS of {item!r}: {e}")
            continue
        if argv and not recipes[name]["parameters"]:
            errors.append(f"{at}: {name} takes no arguments (just would read {argv[0]!r} as another recipe)")
            continue
    # just's own argument check (arity, options), printing the recipe instead of running it.
    chk = subprocess.run(just + ["--dry-run", name] + argv, capture_output=True, text=True)
    if chk.returncode != 0:
        msg = [ln for ln in chk.stderr.splitlines() if ln.startswith("error")] or chk.stderr.splitlines()[-1:]
        errors.append(f"{at}: just rejects {item!r}: {' '.join(msg)}")
        continue
    lane, reason = classify(name)
    if lane in ("T", "R"):
        errors.append(f"{at}: {reason}")
        continue
    arm = solo_arm(name, argv)
    if arm:
        flag, b, solo_n = arm
        errors.append(f"{at}: {item!r} hands {b} {flag}, the arm of the solo recipe {solo_n} — it runs alone in "
                      f"lane X with what that recipe sets around it; list {solo_n} instead")
        continue
    tags = [f"{n}: [group('{g}')]" for g in ("solo", "solo-real", "v41-load", "host", "pool", "big-host") for n in group_of(name, g)]
    tags += defer_tags(name)
    if tags:
        reason = "; ".join(tags + [reason])
    qargs = " ".join(shlex.quote(a) for a in argv)
    # A fixture run is not the real one: its row is an entry of its own, so neither pollutes the other's median.
    tenvs = envs + (["BLOOMERY_TIER=fixture"] if tier == "fixture" else [])
    tkey = name + ("@" + ",".join(tenvs) if tenvs else "") + (":" + qargs if argv else "")
    if re.search(r"[\t\n]", tkey):
        errors.append(f"{at}: {item!r} holds a tab or a newline in its env or ARGS: the times file is one tab-separated row per item")
        continue
    exp, esrc = expected(tkey)
    cls, kind, cards, labels = placement(name, lane, lanes)
    rflags, rbytes = RESOURCES.get(name, ("-", 0))
    if tier == "fixture":
        rbytes = 0  # the fixture tier's loads are the tree's small files: the pack sums nothing there
    if cls == "X":
        reason += f"; pack: {rflags}, {rbytes} B"
    counts[name] = counts.get(name, 0) + 1
    stem = name if counts[name] == 1 else f"{name}-{counts[name]}"
    shown = item if (env is not None or args is not None) else ""
    calls = bounded_calls(name)
    if len(calls) == 1 and calls[0][1] > 0 and exp * 4 > calls[0][1] * 3:
        budget_over.append(f"{shown or name}={exp}s>past-0.75xBLOOMERY_GATE_BOUND={calls[0][1]}s")
    elif len(calls) != 1 or calls[0][1] < 0:
        budget_unchecked.append(shown or name)
    # The neighbour tags (pool, big-host) the run-time rule reads, and the family of a chain member
    # (class C: the chain's order is by family). The pack's resource field (X items only): the
    # table's flags and host bytes, read by the shell side's driver and PYBAL's pack simulation.
    held = " ".join(g for g in ("pool", "big-host") if group_of(name, g))
    # The fields below are the shell side's REC_FIELDS, in this order.
    out.append("\x1f".join([cls, stem, name, " ".join(envs), qargs, shown, reason, kind, str(exp), esrc, tkey, " ".join(cards), " ".join(labels),
                            family(name) if cls in ("X", "C") else "", held,
                            " ".join([rflags, str(rbytes)]) if cls == "X" else ""]))
    fams.append(family(name) if cls in ("X", "C") else "")
if errors:
    for e in errors:
        print("gate-batch: " + e, file=sys.stderr)
    fail(f"{len(errors)} item error(s); nothing ran")
# Lane X by model family: FIRST_FAMILY's items first, then each other family's in the order its first item
# is listed, each family's items in list order. The other lanes keep list order (a record's lane is its
# first field, and lanes run their own records in order).
rank = {FIRST_FAMILY: 0}
for f in fams:
    if f:
        rank.setdefault(f, len(rank))
xs = sorted((k for k, rec in enumerate(out) if rec.startswith("X\x1f")), key=lambda k: (rank[fams[k]], k))
# The chain's order (class C, D3): by family, the family lane X opens with LAST (X then starts on a
# warm host set), the chain's other families in the reverse of X's family order, then families X
# does not hold in first-appearance order; within a family, record order. Computed on the pre-sort
# indices, like xs; the K record names each member's final index after the X records move to the tail.
ck = [k for k, rec in enumerate(out) if rec.startswith("C\x1f")]
chain_fams = list(dict.fromkeys(fams[k] for k in ck))
x_fams = list(dict.fromkeys(fams[k] for k in xs))
x_first = x_fams[0] if x_fams else FIRST_FAMILY
fam_order = ([f for f in reversed(x_fams) if f != x_first and f in chain_fams]
             + [f for f in chain_fams if f not in x_fams and f != x_first]
             + ([x_first] if x_first in chain_fams else []))
cs = sorted(ck, key=lambda k: (fam_order.index(fams[k]), k))
final_of = {k: i for i, k in enumerate(k2 for k2 in range(len(out)) if not out[k2].startswith("X\x1f"))}
out = [rec for rec in out if not rec.startswith("X\x1f")] + [out[k] for k in xs]
print("\n".join(out))
if cs:
    print("\x1f".join(["K", " ".join(str(final_of[k]) for k in cs), " ".join(fams[k] for k in cs)]))
if budget_over or budget_unchecked:
    print("\x1f".join(["W", " ".join(budget_over), " ".join(budget_unchecked)]))  # W, not B: a class-B record starts with B
PY
# The plan record's fields, in the order the writer above joins them (its `out.append`): the one list
# the shell side reads a record by. A writer that gains or loses a field without this list is a named
# error at the first read (rec_get counts), never a field read one place late.
REC_FIELDS=(cls stem name env args item why kind exp esrc tkey cards labels fam held res)
rec_get() { # $1 = plan index, then field names (REC_FIELDS): each name is set, as a variable of that
  # name in the caller's scope, to that field of R_REC[$1]; an unknown name, an index with no record or a
  # record whose field count is not the list's is a named error (70). Pure bash: the lanes call it often.
  local _i=$1 _rec _d _k _n _f
  local -a _vals
  shift
  case $_i in '' | *[!0-9]*) RC=70 die "rec_get: '$_i' is no plan index" ;; esac
  [ "$_i" -lt "${#R_REC[@]}" ] || RC=70 die "rec_get: no plan record $_i (the plan has ${#R_REC[@]})"
  _rec=${R_REC[$_i]}
  _d=${_rec//[!$'\x1f']/}
  [ $((${#_d} + 1)) = "${#REC_FIELDS[@]}" ] || RC=70 die "rec_get: plan record $_i has $((${#_d} + 1)) fields, REC_FIELDS names ${#REC_FIELDS[@]} (${REC_FIELDS[*]})"
  IFS=$'\x1f' read -r -a _vals <<< "$_rec"
  for _n in "$@"; do
    _k=0
    for _f in "${REC_FIELDS[@]}"; do
      [ "$_f" = "$_n" ] && break
      _k=$((_k + 1))
    done
    [ "$_k" -lt "${#REC_FIELDS[@]}" ] || RC=70 die "rec_get: '$_n' is no field of the plan record (${REC_FIELDS[*]})"
    printf -v "$_n" '%s' "${_vals[$_k]-}"
  done
}
PLAN0=$(python3 -c "$PYPLAN" "$ROOT" "$LANES" "$SRC" "$LIST" "${BLOOMERY_BOX_ENV:-}" "$TIMES_FILE" "$DEFAULT_S" "$TIER" "$CHAIN_SOLO_REAL" ${ITEMS[@]+"${ITEMS[@]}"}) \
  || RC=$? die "the list did not validate (above)"
if [ "$SRC" = classes ]; then
  printf '%s\n' "$PLAN0"
  exit 0
fi

item_str() { # NAME ENV ARGS: the item as it runs, for the key — NAME[@ENV with commas for spaces][:ARGS]
  local s=$1
  [ -z "$2" ] || s="$s@${2// /,}"
  [ -z "$3" ] || s="$s:$3"
  printf '%s' "$s"
}

# The candidates: one per card an item may run with. C_ITEM is the item as the ledger keys it (its
# env, then the card); R_C0 is the index of each record's first candidate. C_LST stays `-` without
# --ledger.
R_REC=() R_C0=() C_ITEM=() C_KEY=() C_LST=() C_LDET=()
# The fixture tier's deferred items (class D: real-only gates, and gates of a family with no fixture) are not part of the plan below:
# no lane, no key, no claim, no times row. They are kept apart, named in the dry run and in run.log, listed in DIR/deferred.list and
# counted in the DONE line (the header's deferred paragraph).
DP_STEM=() DP_NAME=() DP_ENV=() DP_ARGS=() DP_ITEM=() DP_WHY=() DP_KIND=()
# The chain's order (PYPLAN's `K` record): plan indices in the order the lanes take them (D3), each
# with its model family. CHAIN_ACTIVE names the batches it exists in: the real tier, two lanes.
CHAIN_ORDER=() CHAIN_FAMS=() CHAIN_ACTIVE=0 CHAIN_K=''
[ "$TIER" = real ] && [ "$LANES" = 2 ] && CHAIN_ACTIVE=1
BUDGET_OVER=() BUDGET_UNCHECKED=() # the plan's B record: items past 0.75x their bound, and unchecked ones
while IFS= read -r rec; do
  case $rec in
    D$'\x1f'*)
      IFS=$'\x1f' read -r _ stem name env args item why kind _ <<< "$rec"
      DP_STEM+=("$stem"); DP_NAME+=("$name"); DP_ENV+=("$env"); DP_ARGS+=("$args"); DP_ITEM+=("$item"); DP_WHY+=("$why"); DP_KIND+=("$kind") ;;
    K$'\x1f'*)
      IFS=$'\x1f' read -r _ ord fams <<< "$rec"
      read -r -a CHAIN_ORDER <<< "$ord"
      read -r -a CHAIN_FAMS <<< "$fams"
      CHAIN_K=$rec ;;
    W$'\x1f'*)
      IFS=$'\x1f' read -r _ bover bunc <<< "$rec"
      read -r -a BUDGET_OVER <<< "$bover"
      read -r -a BUDGET_UNCHECKED <<< "$bunc" ;;
    *) R_REC+=("$rec") ;;
  esac
done <<< "$PLAN0"
# The budget refusal (the header's B record): an item past 0.75 × the bound that kills it is a named
# refusal before anything runs, unless --over-budget-ok passes it.
if [ "${#BUDGET_OVER[@]}" -gt 0 ] && [ "$DRY" = 0 ] && [ "$BUDGET_OK" = 0 ]; then
  for b in "${BUDGET_OVER[@]}"; do echo "gate-batch: budget: $b" >&2; done
  RC=65 die "${#BUDGET_OVER[@]} item(s) past 0.75x the bound that kills them — a slower gate is a red gate in the making; pass --over-budget-ok to run them anyway"
fi
N=${#R_REC[@]}
ND=${#DP_STEM[@]}
NT=$((N + ND)) # every item of the batch: the ones that run and the ones deferred
ND_RO=0 ND_NF=0 # of those, the real-only gates and the gates of a family with no fixture
for ((i = 0; i < ND; i++)); do
  if [ "${DP_KIND[$i]}" = real-only ]; then ND_RO=$((ND_RO + 1)); else ND_NF=$((ND_NF + 1)); fi
done
DEFER_NOTE=''
[ "$ND" = 0 ] || DEFER_NOTE=" (and $ND deferred to the real tier: $ND_RO real-only, $ND_NF of a family with no fixture)"
for ((i = 0; i < N; i++)); do
  rec_get "$i" name env args cards
  R_C0+=("${#C_ITEM[@]}")
  for c in $cards; do
    e=$env
    [ "$c" = - ] || e="${e:+$e }BLOOMERY_GATE_CARD=$c"
    C_ITEM+=("$(item_str "$name" "$e" "$args")"); C_KEY+=(-); C_LST+=(-); C_LDET+=("")
  done
done
NC=${#C_ITEM[@]}

# The second pass: each record, then the ledger state of each of its candidates. It prints per item
# the lane, stem, recipe, env with the card, ARGS, the item as written, the reason, the plan line
# (kind, expected seconds, placement), the seconds it counts in its lane, the times key and card
# label, the candidate it took and the neighbour tags (pool, big-host); then one `=` record: the
# lane sums, the chain's, the wall, and the items whose expectation is the default.
IFS= read -r -d '' PYBAL << 'PY' || true
import sys

lanes = sys.argv[1]


def fail(msg):
    print("gate-batch: balance: " + msg, file=sys.stderr)
    sys.exit(70)


def lane_cand(cards):
    """lane -> candidate index of a balanced item: its two cards, or one no-card candidate for both."""
    if cards == ["3090", "a6000"]:
        return {"A": 0, "B": 1}
    if cards == ["-"]:
        return {"A": 0, "B": 0}
    return None


recs = [ln.split("\x1f") for ln in sys.stdin.read().split("\n") if ln]
k_order = []
if recs and recs[0][0] == "K":  # the chain's order (PYPLAN's K record), fed first by the shell
    k_order = [int(x) for x in recs.pop(0)[1].split()]
sums, placed, flex = {"A": 0, "B": 0, "X": 0, "C": 0}, {}, []
for i, f in enumerate(recs):
    if len(f) != 17:
        fail(f"a plan record has {len(f)} fields, not 17: {f!r}")
    cls, stem, exp, cards, labels, states = f[0], f[1], int(f[8]), f[11].split(), f[12].split(), f[16].split()
    if not cards or not len(cards) == len(labels) == len(states):
        fail(f"{stem}: {len(cards)} card candidates, {len(labels)} labels, {len(states)} ledger states")
    if cls == "C" and lanes == "2":
        # The chain (the header): not a lane's work — its seconds are the chain's own sum. A member
        # green on one candidate's key skips there at 0 s (green on both, it takes the 3090's); a
        # member with both candidates live runs on whichever lane's card takes it, so no candidate
        # is fixed here.
        green = [j for j, s in enumerate(states) if s == "skip"]
        if green:
            c = green[0] if len(green) == 1 else 0
            placed[i] = ("C", c, 0, f"skips at its {labels[c]} key, counted 0 s")
        else:
            placed[i] = ("C", 0, exp, "")
        sums["C"] += placed[i][2]
    elif cls == "X":
        # The pack's item (the header's «pack»): no lane places it — its seconds are the pack
        # simulation's below. A skip on any candidate costs 0 (the pack skips at the green key).
        placed[i] = ("X", 0, 0 if any(s == "skip" for s in states) else exp, "")
    elif cls in ("A", "B") and len(cards) == 1:
        placed[i] = (cls, 0, 0 if states[0] == "skip" else exp, "")
        sums[cls] += placed[i][2]
    elif cls == "F" and lanes == "2" and lane_cand(cards):
        flex.append(i)
    else:
        fail(f"{stem} fits no lane (class {cls!r}, cards {' '.join(cards)}, --lanes {lanes})")
# The ledger first: an item green on one lane's card only skips there, so it goes there at 0 s.
rest, seq = [], []
for i in flex:
    lc, states = lane_cand(recs[i][11].split()), recs[i][16].split()
    green = [ln for ln in ("A", "B") if states[lc[ln]] == "skip"]
    if len(green) == 1:
        placed[i] = (green[0], lc[green[0]], 0, f"to {green[0]}, the one lane whose card has it green")
        seq.append(i)
    elif green:
        ln = "A" if sums["A"] < sums["B"] else "B"
        placed[i] = (ln, lc[ln], 0, f"to {ln}, green in either lane")
        seq.append(i)
    else:
        rest.append(i)
# Then longest first, each to the lane with the smaller running sum (B on a tie).
for i in sorted(rest, key=lambda i: (-int(recs[i][8]), i)):
    ln, lc, exp = "A" if sums["A"] < sums["B"] else "B", lane_cand(recs[i][11].split()), int(recs[i][8])
    placed[i] = (ln, lc[ln], exp, f"to {ln} (lane sums before it: A {sums['A']} s, B {sums['B']} s)")
    sums[ln] += exp
    seq.append(i)
defaults = []
# The take order a chain batch's lanes would run, for the dry run (the run decides: the free lane
# takes the chain's next member its card may run — the K order, the mutex's race approximated —
# else the longest-est item its card may run; the neighbour rule holds while a member is unclaimed,
# and skips cost 0 s). Each simulated take names the lane and its start, "A@123".
sim = {}
if k_order:
    def fits(i, lane):
        labels = recs[i][12].split()
        return labels == ["none"] or (lane == "A" and "3090" in labels) or (lane == "B" and "a6000" in labels)

    clock = {"A": 0, "B": 0}
    chain = [i for i in k_order if placed[i][2] > 0]
    rest = sorted((i for i, f in enumerate(recs) if f[0] in ("A", "B", "F") and placed[i][2] > 0),
                  key=lambda i: (-placed[i][2], i))
    left = len(chain) + len(rest)
    while left:
        lane = "A" if clock["A"] <= clock["B"] else "B"
        pick = next((i for i in chain if i not in sim), None)
        if pick is not None and not fits(pick, lane):
            pick = None  # only the other lane's card runs it; this lane works beside the chain
        if pick is None:
            tagged_hold = bool([i for i in chain if i not in sim])
            pick = next((i for i in rest if i not in sim and fits(i, lane)
                         and not (tagged_hold and recs[i][14])), None)
        if pick is None:
            clock[lane] = 1 << 60  # this lane has nothing it may run
            if clock["A" if lane == "B" else "B"] >= (1 << 60) and left:
                break  # both lanes done; the remainder is not simulatable (should not happen)
            continue
        sim[pick] = f"{lane}@{clock[lane]}"
        clock[lane] += placed[pick][2]
        left -= 1
# The pack's simulation (the header's «pack»): the X items placed onto the two cards by the driver's
# own rule — the longest-est eligible item first (a one-card item takes a free card, the A6000 first
# of the two), the running items' bytes within the budget (a lone item always starts), an m-flagged
# item alone. Its wall is the predicted line's laneX=; each item's simulated take joins the chain's.
budget = int(sys.argv[2])
pack_wall = 0
free, busy = {"3090", "a6000"}, []
left = [i for i, f in enumerate(recs) if f[0] == "X" and placed[i][2] > 0]
while left or busy:
    launched = True
    while launched:
        launched = False
        sumb = sum(b[2] for b in busy)
        mrun = any(b[3] for b in busy)
        for i in sorted(left, key=lambda i: (-placed[i][2], i)):
            rflags, rbytes = recs[i][15].split()
            mflag = rflags == "m"
            if busy and (mflag or mrun or sumb + int(rbytes) > budget):
                continue  # alone when m; nothing beside an m item; pairs within the budget
            lab = recs[i][12].split()
            if lab == ["none"]:
                cards = set()
            elif lab == ["both"]:
                if free != {"3090", "a6000"}:
                    continue
                cards = {"3090", "a6000"}
            elif lab == ["a6000"]:
                if "a6000" not in free:
                    continue
                cards = {"a6000"}
            elif lab == ["3090"]:
                if "3090" not in free:
                    continue
                cards = {"3090"}
            else:
                if not free & {"3090", "a6000"}:
                    continue
                cards = {"a6000"} if "a6000" in free else {"3090"}
            sim[i] = f"X@{pack_wall}"
            busy.append((pack_wall + placed[i][2], cards, int(rbytes), mflag))
            free -= cards
            left.remove(i)
            launched = True
            break
    if not busy:
        break
    pack_wall = min(b[0] for b in busy)
    for b in [b for b in busy if b[0] == pack_wall]:
        busy.remove(b)
        free |= b[1]
for i, f in enumerate(recs):
    cls, stem, name, env, args, shown, why, kind, exp, esrc, tkey, cards, labels, fam, tag, res, states = f
    ln, c, cost, how = placed[i]
    card, label, state = cards.split()[c], labels.split()[c], states.split()[c]
    # A chain member or a pack item with both candidates live pins no card here: the lane that takes
    # the member, or the pack the item, rewrites its placement (switch_candidate) at run time.
    if card != "-" and not (ln in ("C", "X") and len(cards.split()) > 1 and state != "skip"):
        env = (env + " " if env else "") + "BLOOMERY_GATE_CARD=" + card
    if state == "skip" and ln not in ("C", "X"):
        told = "skips (the ledger has it green), counted 0 s"
    elif ln == "C":
        told = how or f"expected {exp} s ({esrc}), family {fam}"
        told += "; the lane and card are decided at run time (one at a time, batch-wide)"
        if not how and esrc == "default":
            defaults.append(stem)
    elif ln == "X":
        if state == "skip" or any(s == "skip" for s in states.split()):
            told = "skips at its green candidate, counted 0 s"
        else:
            told = f"expected {exp} s ({esrc}), {res.split()[0]} flag, {res.split()[1]} B"
            told += "; the pack decides the card" if len(cards.split()) > 1 else "; the pack"
            if esrc == "default":
                defaults.append(stem)
    else:
        told = f"expected {exp} s ({esrc})"
        if esrc == "default":
            defaults.append(stem)
    plan = f"{kind}, {told}"
    print("\x1f".join([ln, stem, name, env, args, shown, why, plan, str(cost), tkey, label, str(c), tag, sim.get(i, "")]))
# The wall. Without a chain it is today's: the longer lane, then the pack. With one, phase 1 is the
# largest of the chain's own sum (one member at a time batch-wide), the whole work over two cards (a
# member lands on either), each card's own-only sum, and the chain plus the tail of the pool and
# big-host items — they wait for the chain's end (the run-time rule) and then pack onto the two
# cards; then the pack.
pinned = {"3090": 0, "a6000": 0}
pool = {"3090": 0, "a6000": 0, "bal": 0}
work = sums["C"]
for i, f in enumerate(recs):
    if f[0] in ("C", "X"):
        continue  # the chain has its own term; the pack runs after phase 1
    cost, cards, labels, tag = placed[i][2], f[11].split(), f[12].split(), f[14]
    work += cost
    if cards == ["3090"] or (cards == ["-"] and labels == ["3090"]):
        pinned["3090"] += cost
    elif cards == ["a6000"] or (cards == ["-"] and labels == ["a6000"]):
        pinned["a6000"] += cost
    if tag and cost:
        if labels == ["3090"]:
            pool["3090"] += cost
        elif labels == ["a6000"]:
            pool["a6000"] += cost
        else:
            pool["bal"] += cost
tail = max(pool["3090"], pool["a6000"], -(-pool["bal"] // 2),
           -(-(pool["3090"] + pool["a6000"] + pool["bal"]) // 2))
phase1 = max(sums["C"], -(-work // 2), pinned["3090"], pinned["a6000"], sums["C"] + tail)
wall = max(phase1, sums["A"], sums["B"]) + pack_wall
print("\x1f".join(["=", str(sums["A"]), str(sums["B"]), str(pack_wall), str(sums["C"]), str(wall), str(len(defaults)), " ".join(defaults)]))
# The run order: lanes A and X in record order, lane B's fixed items (class B: a solo-real recipe that picks the A6000 in the
# fixture tier) first in record order, then its balanced items in the order the balance placed them — its skips, then
# longest first. Every lane-A item opens with a release build, and cargo holds one build lock per target
# directory for the whole box tree: a lane-B build that starts at t = 0 (a device crate's lib tests build
# for ~100 s) makes lane A's first build wait for it. When the longest balanced item runs on the Mac
# (check-recipes, ~130 s, in a narrowed landing list), longest first starts the long builds behind it,
# while lane A runs its first binaries; when it is a box build, lane B opens with that build as before.
# The chain's items are in no lane's order: the lanes take them at run time, one at a time.
order = ([i for i in range(len(recs)) if placed[i][0] == "A"] + [i for i in range(len(recs)) if recs[i][0] == "B"]
         + [i for i in seq if placed[i][0] == "B"] + [i for i in range(len(recs)) if placed[i][0] == "X"])
nchain = sum(1 for f in recs if f[0] == "C")
if sorted(order) != sorted(i for i, f in enumerate(recs) if f[0] != "C"):
    fail(f"the run order is not a permutation of the {len(recs)} items ({nchain} of them the chain's): {order}")
print("\x1f".join(["O", " ".join(map(str, order))]))
PY

balance() { # PYBAL over the records and their candidates' ledger states: the P_* arrays and SUM_*
  local i j k st feed='' lane stem name env args item why plan tkey tcard cand tag sim plan_out
  [ -z "$CHAIN_K" ] || feed="$CHAIN_K"$'\n'
  for ((i = 0; i < N; i++)); do
    j=${R_C0[$i]}
    if [ $((i + 1)) -lt "$N" ]; then k=${R_C0[$((i + 1))]}; else k=$NC; fi
    st=''
    for ((; j < k; j++)); do st="${st:+$st }${C_LST[$j]}"; done
    feed="$feed${R_REC[$i]}"$'\x1f'"$st"$'\n'
  done
  plan_out=$(printf '%s' "$feed" | python3 -c "$PYBAL" "$LANES" "$PACK_BUDGET") || RC=70 die "the lane balance failed (above)"
  i=0 SUM_W=''
  while IFS=$'\x1f' read -r lane stem name env args item why plan _ tkey tcard cand tag sim; do
    if [ "$lane" = "=" ]; then
      SUM_A=$stem SUM_B=$name SUM_X=$env SUM_C=$args SUM_W=$item SUM_NDEF=$why SUM_DEF=$plan
      continue
    fi
    if [ "$lane" = O ]; then
      read -r -a ORDER <<< "$stem"
      continue
    fi
    P_LANE+=("$lane"); P_STEM+=("$stem"); P_NAME+=("$name"); P_ENV+=("$env"); P_ARGS+=("$args")
    P_ITEM+=("$item"); P_WHY+=("$why"); P_PLAN+=("$plan"); P_TKEY+=("$tkey"); P_TCARD+=("$tcard")
    j=$((${R_C0[$i]} + cand))
    P_CI+=("$j"); P_KEY+=("${C_KEY[$j]}"); P_LST+=("${C_LST[$j]}"); P_LDET+=("${C_LDET[$j]}"); P_TAG+=("$tag"); P_SIM+=("$sim")
    i=$((i + 1))
  done <<< "$plan_out"
  [ "$i" = "$N" ] && [ -n "$SUM_W" ] && [ "${#ORDER[@]}" = "$((N - ${#CHAIN_ORDER[@]}))" ] || RC=70 die "the lane balance returned $i of $N items and an order of ${#ORDER[@]}"
}
ORDER=() # the run order (PYBAL's `O` record): plan indices, each lane's items in the order they run
P_LANE=() P_STEM=() P_NAME=() P_ENV=() P_ARGS=() P_ITEM=() P_WHY=() P_PLAN=() P_TKEY=() P_TCARD=()
P_CI=() P_KEY=() P_LST=() P_LDET=() P_TAG=() P_SIM=() # the candidate taken, its key, status, detail, tags, simulated take
# P_FAM: a chain member's model family (aligned with CHAIN_ORDER), for the run-time order.
P_FAM=()
for ((i = 0; i < ${#CHAIN_ORDER[@]}; i++)); do P_FAM[${CHAIN_ORDER[$i]}]=${CHAIN_FAMS[$i]}; done
SUM_A=0 SUM_B=0 SUM_X=0 SUM_C=0 SUM_W=0 SUM_NDEF=0 SUM_DEF=''

box_env_of() { # the item's full BLOOMERY_BOX_ENV: the caller's, then the item's and the lane's
  local e="${BLOOMERY_BOX_ENV:-}"
  [ -z "${P_ENV[$1]}" ] || e="${e:+$e }${P_ENV[$1]}"
  printf '%s' "$e"
}
cmd_of() {
  local e; e=$(box_env_of "$1")
  printf '%sjust %s%s' "${e:+BLOOMERY_BOX_ENV='$e' }" "${P_NAME[$1]}" "${P_ARGS[$1]:+ ${P_ARGS[$1]}}"
}
item_of() { # the item as it runs, for the key: NAME[@its env and its lane's card][:ARGS]
  item_str "${P_NAME[$1]}" "${P_ENV[$1]}" "${P_ARGS[$1]}"
}
def_cmd_of() { # a deferred item's command as it would have run (the dry run names it; the batch never calls it)
  local e="${BLOOMERY_BOX_ENV:-}"
  [ -z "${DP_ENV[$1]}" ] || e="${e:+$e }${DP_ENV[$1]}"
  printf '%sjust %s%s' "${e:+BLOOMERY_BOX_ENV='$e' }" "${DP_NAME[$1]}" "${DP_ARGS[$1]:+ ${DP_ARGS[$1]}}"
}

LEAD_LEDGER=${BLOOMERY_GATE_LEDGER:-$HOME/.cache/bloomery/gate-ledger.tsv}
ROUND_LEDGER=${BLOOMERY_GATE_ROUND_LEDGER:-$HOME/.cache/bloomery/gate-ledger-rounds.tsv}
[ "$LEAD_LEDGER" != "$ROUND_LEDGER" ] || die "BLOOMERY_GATE_LEDGER and BLOOMERY_GATE_ROUND_LEDGER name one file ($LEAD_LEDGER): a round's green would land in the lead's ledger"
# The file this batch writes, and the second file it reads (empty: none).
if [ "$LMODE" = round ]; then LEDGER_FILE=$ROUND_LEDGER; else LEDGER_FILE=$LEAD_LEDGER; fi
ALSO_READ=''
if [ "$LMODE" = round ] || [ "$TRUST" = 1 ]; then ALSO_READ=$ROUND_LEDGER; fi
LEDGER_PARTS=$LEDGER_FILE.parts

ledger_plan() { # the box manifest, then every candidate's key and status (C_*), before the lease check
  local m=$1/box-manifest.txt merr='' rc=0 try i key item status detail kargs=()
  for try in 1 2; do
    rc=0
    BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" "python3 tools/recipes.py box-manifest" > "$m" 2> "$1/box-manifest.err" || rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 75 ]; then break; fi
    [ "$try" = 2 ] || sleep 10
  done
  if [ "$rc" = 75 ]; then
    if [ "$DRY" = 0 ]; then
      grep '^\[' "$1/box-manifest.err" >&2 || true
      RC=75 die "the timing lease or a hold is up (above) — a batch's builds contaminate a sitting; not starting"
    fi
    merr="the timing lease or a hold was up, so the manifest was not read"
  elif [ "$rc" != 0 ]; then
    merr="box.sh rc=$rc: $(tail -1 "$1/box-manifest.err")"
    echo "gate-batch: ledger: the box manifest failed ($merr) — every item runs and none is recorded" >&2
  fi
  kargs=(key --manifest "$m" --box-env "${BLOOMERY_BOX_ENV:-}" --ledger "$LEAD_LEDGER" --parts-dir "$1/parts")
  [ -z "$ALSO_READ" ] || kargs+=(--round-ledger "$ALSO_READ")
  [ -z "$merr" ] || kargs+=(--manifest-error "$merr")
  [ "$RERUN" = 0 ] || kargs+=(--rerun)
  rc=0
  python3 "$ROOT/tools/recipes.py" "${kargs[@]}" "${C_ITEM[@]}" > "$1/keys.tsv" 2> "$1/keys.err" || rc=$?
  [ ! -s "$1/keys.err" ] || sed 's/^/gate-batch: ledger: /' "$1/keys.err" >&2
  i=0
  if [ "$rc" = 0 ]; then
    while IFS=$'\t' read -r key item status detail; do
      if [ "$i" -ge "$NC" ] || [ "$item" != "${C_ITEM[$i]}" ]; then i=-1; break; fi
      C_KEY[$i]=$key C_LST[$i]=$status C_LDET[$i]=$detail
      i=$((i + 1))
    done < "$1/keys.tsv"
  fi
  if [ "$i" != "$NC" ]; then
    echo "gate-batch: ledger: tools/recipes.py key failed (rc=$rc, $i of $NC lines in order) — every item runs and none is recorded" >&2
    for ((i = 0; i < NC; i++)); do C_KEY[$i]=- C_LST[$i]=error C_LDET[$i]="the key tool failed (rc=$rc)"; done
  fi
}

ledger_record() { # $1 = plan index, $2 = final rc: record a green item whose key did not move; print its state
  local i=$1 k2 rec
  case "${P_LST[$i]}" in
    never) echo never; return ;;
    error) echo unkeyed; return ;;
  esac
  if [ "$2" != 0 ]; then echo red; return; fi
  # The recheck's failure is named, never read as a moved input: its stderr and a keyless line
  # (`-<TAB>item<TAB>error<TAB>why`) go to the batch's stderr, and the state is `unkeyed`.
  local out err=$LWORK/recheck-$i.err rc=0
  out=$(python3 "$ROOT/tools/recipes.py" key --manifest "$LWORK/box-manifest.txt" --box-env "${BLOOMERY_BOX_ENV:-}" "$(item_of "$i")" 2> "$err") || rc=$?
  k2=${out%%$'\t'*}
  if [ "$rc" != 0 ] || [ -z "$k2" ] || [ "$k2" = - ]; then
    echo "gate-batch: ledger: the key recheck of $(item_of "$i") failed (rc=$rc): $(tail -1 "$err")${out:+ $out}" >&2
    echo unkeyed
    return
  fi
  if [ "$k2" != "${P_KEY[$i]}" ]; then echo changed; return; fi
  rec=$(printf '%s\t%s\t%s\t%s\t%s\t%s' "${P_KEY[$i]}" "${P_NAME[$i]}" "$(item_of "$i")" "$COMMIT" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$ROOT")
  if python3 -c "$PYAPPEND" "$LEDGER_FILE" "$rec" "$LWORK/parts/${P_CI[$i]}.parts" "$LEDGER_PARTS/${P_KEY[$i]}.parts"; then
    echo recorded
  else
    echo append-failed
  fi
}

if [ -z "$OUT" ]; then
  OUT="$HOME/.cache/bloomery/batches/$(basename "$ROOT")/$(date +%Y%m%d-%H%M%S)"
fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$OUT/" in
  "$ROOT"/target/*) ;;
  "$ROOT"/*) die "--out $OUT is inside the tree but not under target/: box.sh would ship the logs and mark the tree dirty" ;;
esac

# The disk floor (the header): both volumes before the log directory exists and any lane starts. A
# below-floor verdict (rc 1, the line printed by disk_ok) ends a real run at 69; a dry run says so
# and continues, and a df that could not be read (69) fails it too.
rc=0; disk_ok "$ROOT/target" || rc=$?
case $rc in 0) ;; 1) below=1 ;; *) exit "$rc" ;; esac
rc=0; disk_ok "$OUT" || rc=$?
case $rc in 0) ;; 1) below=1 ;; *) exit "$rc" ;; esac
if [ "${below:-0}" = 1 ]; then
  if [ "$DRY" = 0 ]; then exit 69; fi
  echo "gate-batch: dry run — a real run would not start (the disk floor above)" >&2
fi

if [ "$DRY" = 0 ]; then
  mkdir -p "$OUT" || RC=73 die "cannot create the log directory $OUT"
  [ -d "$OUT" ] && [ -w "$OUT" ] || RC=73 die "the log directory $OUT is not a writable directory"
  [ ! -e "$OUT/run.log" ] || RC=73 die "$OUT already holds a batch (run.log) — give --out a new directory"
fi
RUNLOG=$OUT/run.log

SKIP_N=0
if [ "$LEDGER" = 1 ]; then
  if [ "$DRY" = 1 ]; then
    LWORK=$(mktemp -d "${TMPDIR:-/tmp}/gate-batch-ledger.XXXXXX") || RC=73 die "cannot make a scratch directory for the ledger"
    trap 'rm -rf "$LWORK"' EXIT
  else
    LWORK=$OUT/ledger
    mkdir -p "$LWORK" || RC=73 die "cannot create $LWORK"
    COMMIT=$(git -C "$ROOT" rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
    [ -z "$(git -C "$ROOT" status --porcelain 2> /dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
  fi
  [ "$NC" = 0 ] || ledger_plan "$LWORK" # a batch whose items are all deferred has nothing to key
fi
balance
for ((i = 0; i < N; i++)); do [ "${P_LST[$i]}" != skip ] || SKIP_N=$((SKIP_N + 1)); done
if [ "$LEDGER" = 1 ]; then
  echo "gate-batch: ledger: reads $LEAD_LEDGER${ALSO_READ:+ and $ALSO_READ}, writes $LEDGER_FILE: $SKIP_N of $N items skip"
fi
if [ "$LANES" = 1 ]; then
  PREDICTED="laneA=${SUM_A}s wall=${SUM_W}s"
else
  PREDICTED="laneA=${SUM_A}s laneB=${SUM_B}s laneX=${SUM_X}s chain=${SUM_C}s wall=${SUM_W}s"
fi
predicted() { # the plan's lane sums, derived from the times file
  local how="the median of each item's last 5 rows in $TIMES_FILE"
  [ "$LEDGER" = 0 ] || how="$how, a skipped item 0 s"
  if [ "$LANES" = 1 ]; then
    how="$how; wall = the lane, then nothing"
  elif [ "$CHAIN_ACTIVE" = 1 ] && [ "${#CHAIN_ORDER[@]}" -gt 0 ]; then
    how="$how; phase 1 = max(the chain, ⌈the lanes' work with the chain over two cards⌉, the 3090-only sum, the A6000-only sum, the chain plus the pool/big-host tail), then the pack (laneX=: the cards, the ${PACK_BUDGET} B host budget, m items alone)"
  else
    how="$how; wall = the longer of A and B, then the pack (laneX=: the cards, the ${PACK_BUDGET} B host budget, m items alone)"
  fi
  echo "gate-batch: predicted $PREDICTED (derived: $how)"
  [ "$SUM_NDEF" = 0 ] || echo "gate-batch: $SUM_NDEF item(s) expected at the ${DEFAULT_S} s default (no row in the times file): $SUM_DEF"
  [ "$ND" = 0 ] || echo "gate-batch: $ND item(s) deferred to the real tier ($ND_RO real-only, $ND_NF of a family with no fixture), in no lane (DONE: deferred=+$ND, and the list $OUT/deferred.list): ${DP_STEM[*]}"
}

# The GPU hold (the header's «The GPU hold»). The box-side writer is tools/gpu-gate.sh --gpuhold, reached through box.sh with
# BLOOMERY_BOX_READONLY=1 (no sync, so no rsync under a running batch) and BLOOMERY_BOX_WAIT=0 (no guard wait: a heartbeat held in a
# sitting's guard would go stale), stdin closed (the read-only ssh is the one that reads it).
hold_box() { # $1 = up | beat | down: the box-side writer's output and rc
  BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" "bash tools/gpu-gate.sh --gpuhold $1 $HOLD_OWNER" < /dev/null
}
hold_path() { bash "$ROOT/tools/gpu-gate.sh" --gpuhold path 2> /dev/null || echo '(the path tools/gpu-gate.sh --gpuhold path prints)'; }
hold_say() { # the dry run's line: the hold this batch would take
  if [ "$HOLD_ON" = 1 ]; then
    echo "gate-batch: gpu hold: would put $(hold_path) up on the box as $HOLD_OWNER, refreshed every ${HOLD_BEAT} s; each item gets BLOOMERY_BATCH_OWNER=$HOLD_OWNER, and other tracks' tools/gpu-gate.sh runs wait until the batch ends"
  elif [ "$LMODE" = lead ]; then
    echo "gate-batch: gpu hold: off (--no-gpu-hold)"
  else
    echo "gate-batch: gpu hold: none (only a --ledger batch takes it)"
  fi
}
hold_beat_loop() { # the heartbeat, a background job: one refresh of the hold every HOLD_BEAT s until TERM, until the hold is gone,
  # or until the batch is (a batch that dies uncleanly runs no trap: its heartbeat must not keep the hold fresh after it)
  local sp='' rc
  trap - EXIT
  trap '[ -z "$sp" ] || kill "$sp" 2> /dev/null; exit 0' TERM
  while :; do
    sleep "$HOLD_BEAT" &
    sp=$!
    wait "$sp" || true
    if ! kill -0 "$$" 2> /dev/null; then # $$ is the batch's own pid in this subshell too
      echo "$(date '+%F %T') the batch (pid $$) is gone: the heartbeat stops, and the box calls the hold stale after 300 s" >> "$OUT/gpuhold.log"
      exit 0
    fi
    rc=0
    hold_box beat >> "$OUT/gpuhold.log" 2>&1 &
    sp=$!
    wait "$sp" || rc=$?
    case $rc in
      0) echo "$(date '+%F %T') beat ok" >> "$OUT/gpuhold.log" ;;
      3) echo "$(date '+%F %T') the hold is gone or another batch's: the heartbeat stops" >> "$OUT/gpuhold.log"; exit 0 ;;
      *) echo "$(date '+%F %T') beat failed rc=$rc: a hold nobody refreshes goes stale on the box by itself" >> "$OUT/gpuhold.log" ;;
    esac
  done
}
hold_up() { # the hold goes up (HOLD_UP=1), or the batch does not start: 75 while another batch's fresh hold is up, 70 otherwise
  local out rc=0
  out=$(hold_box up 2>&1) || rc=$?
  { echo "$(date '+%F %T') up rc=$rc"; printf '%s\n' "$out"; } >> "$OUT/gpuhold.log"
  case $rc in
    0) HOLD_UP=1 ;;
    75) printf '%s\n' "$out" >&2; RC=75 die "another batch's GPU hold is up (above) — not starting; wait for it, or --no-gpu-hold runs beside it" ;;
    *) printf '%s\n' "$out" >&2; RC=70 die "the GPU hold could not be put up (rc $rc, above; tools/gpu-gate.sh --gpuhold through box.sh) — not starting; --no-gpu-hold runs without it" ;;
  esac
}
hold_down() { # the EXIT trap: the heartbeat's pid from DIR, then the hold, only while it is still this batch's
  [ "$HOLD_UP" = 1 ] || return 0
  HOLD_UP=0
  local p rc=0
  if [ -f "$OUT/gpuhold.pid" ]; then
    p=$(cat "$OUT/gpuhold.pid")
    kill -TERM "$p" 2> /dev/null || true
    wait "$p" 2> /dev/null || true
  fi
  hold_box down >> "$OUT/gpuhold.log" 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then
    echo "gate-batch: gpu hold: down"
  else
    echo "gate-batch: the GPU hold was not taken down (rc $rc, $OUT/gpuhold.log): the box calls it stale after 300 s without a beat" >&2
  fi
}

stop() {
  local f
  # The lanes first, so none starts its next item, then the items they were running.
  for f in "$OUT"/lane-*.pid "$OUT"/lane-*.child; do
    if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
  done
  echo "gate-batch: stopped; a box process a killed item left behind is cleared by 'just box-gc'" >&2
  exit 130
}

if [ "$DRY" = 1 ]; then
  echo "gate-batch: dry run — $N items$DEFER_NOTE, lanes $LANES, logs would go to $OUT"
  [ "${#BUDGET_OVER[@]}" = 0 ] || echo "gate-batch: budget: ${#BUDGET_OVER[@]} item(s) past 0.75x the bound that kills them (a real run refuses them, 65, unless --over-budget-ok): ${BUDGET_OVER[*]}"
  [ "${#BUDGET_UNCHECKED[@]}" = 0 ] || echo "gate-batch: budget: ${#BUDGET_UNCHECKED[@]} item(s) unchecked (no single bounded call, or a bound the text does not spell): ${BUDGET_UNCHECKED[*]}"
  if [ "${#CHAIN_ORDER[@]}" -gt 0 ]; then
    echo "gate-batch: the lanes' items below are simulated from the expected seconds (the chain first, then longest-est); each lane is chosen at run time"
  fi
  for i in ${ORDER[@]+"${ORDER[@]}"}; do
    where=${P_LANE[$i]}
    plan=${P_PLAN[$i]}
    case ${P_SIM[$i]:-} in
      '') ;;
      X@*) plan="$plan; the pack takes it at ${P_SIM[$i]#X@}s" ;; # the lane label stays X: the listing is the record order
      *) where="${P_SIM[$i]}s (simulated)" ;;
    esac
    printf 'lane %s  %-28s %s\n        %s — %s\n' "$where" "${P_STEM[$i]}" "$(cmd_of "$i")" "$plan" "${P_WHY[$i]}"
    [ "$LEDGER" = 0 ] || printf '        ledger: %s — %s\n' "${P_LST[$i]}" "${P_LDET[$i]}"
    if [ "$LANES" = 2 ]; then
      rec_get "$i" cls
      if [ "$cls" = F ] && [ "${P_LST[$i]}" != skip ]; then
        printf '        movable — a free lane takes it on its card, longest-est first (never a fixed, solo or v41-load item)\n'
      fi
    fi
  done
  for i in ${CHAIN_ORDER[@]+"${CHAIN_ORDER[@]}"}; do
    printf 'lane C  %-28s %s\n        %s — %s\n' "${P_STEM[$i]}" "$(cmd_of "$i")" "${P_PLAN[$i]}" "${P_WHY[$i]}"
    [ "$LEDGER" = 0 ] || printf '        ledger: %s — %s\n' "${P_LST[$i]}" "${P_LDET[$i]}"
  done
  for ((i = 0; i < ND; i++)); do
    printf 'lane -  %-28s %s\n        deferred to the real tier (%s), never run in the fixture tier — %s\n' "${DP_STEM[$i]}" "$(def_cmd_of "$i")" "${DP_KIND[$i]}" "${DP_WHY[$i]}"
  done
  predicted
  hold_say
  exit 0
fi

# The start check is box.sh's guard with no wait (the header): the lease or a hold up is 75 at once,
# with the guard's lines naming what is up (the lease's holders through lease_holders, each hold's
# owner and age); a lease that cannot be tested is 70.
if [ "$SKIP_N" -lt "$N" ]; then
  rc=0
  lease=$(BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" true 2>&1) || rc=$?
  case $rc in
    0) ;;
    75) echo "$lease" >&2; RC=75 die "the timing lease or a hold is up (above) — a batch's builds contaminate a sitting; not starting" ;;
    *) echo "$lease" >&2; RC=70 die "the start check through box.sh failed (rc $rc; 70 from the guard: the lease cannot be tested)" ;;
  esac
elif [ "$N" = 0 ]; then
  echo "gate-batch: every item is deferred to the real tier — nothing runs on the box, so no lease check"
else
  echo "gate-batch: every item skips — nothing runs on the box, so no lease check"
fi

# The traps, then the GPU hold (the header): the hold goes up only when some item runs on the box, after the start check, and a
# refusal here leaves no run.log, as the start check's does. Every exit from here on — the end, a red batch, die, INT/TERM (stop) —
# takes it down through the EXIT trap.
trap stop INT TERM
trap hold_down EXIT
if [ "$HOLD_ON" = 1 ] && [ "$SKIP_N" -lt "$N" ]; then
  hold_up
  hold_beat_loop &
  echo $! > "$OUT/gpuhold.pid"
  echo "gate-batch: gpu hold: up as $HOLD_OWNER ($(hold_path), refreshed every ${HOLD_BEAT} s, log $OUT/gpuhold.log): other tracks' GPU gates wait until this batch ends"
fi

# run.log exists from here on (a refused start leaves none, so the same --out can be retried).
: > "$RUNLOG"
mkdir -p "$OUT/claims" # one claim per item: the owner's start and the thief's steal agree on it
T0=$(date +%s)
echo "plan $PREDICTED defaults=$SUM_NDEF (predicted, derived from $TIMES_FILE)" >> "$RUNLOG"
echo "gate-batch: $N items$DEFER_NOTE, lanes $LANES, logs in $OUT"
predicted
for ((i = 0; i < ND; i++)); do # a deferred item is recorded at once: the batch never calls it (nothing to time, key or claim)
  line="${DP_STEM[$i]} rc=deferred ${DP_KIND[$i]} lane=- deferred=1"
  [ -z "${DP_ITEM[$i]}" ] || line="$line item=${DP_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
done
DEFER_LIST=$OUT/deferred.list
if [ "$ND" != 0 ]; then
  { # the deferred items in --list format, so a real-tier batch takes exactly this set (`--list DIR/deferred.list`)
    echo "# gate-batch --tier fixture: the $ND item(s) it deferred to the real tier, one per line (the --list format)"
    for kind in real-only no-fixture; do
      if [ "$kind" = real-only ]; then
        [ "$ND_RO" = 0 ] || echo "# real-only: no fixture-tier conversion (bash tools/ref/real-only.sh in the recipe's box command)"
      else
        [ "$ND_NF" = 0 ] || echo "# no-fixture: the recipe's family has no row in the fixture table of tools/ref/ref-paths.sh (box.sh exits 66)"
      fi
      for ((i = 0; i < ND; i++)); do
        [ "${DP_KIND[$i]}" = "$kind" ] || continue
        echo "${DP_ITEM[$i]:-${DP_NAME[$i]}}"
      done
    done
  } > "$DEFER_LIST" || RC=73 die "cannot write $DEFER_LIST"
fi


run_item() { # $1 = plan index, $2 = lane label; the lane's current child pid goes to lane-<lane>.child
  local i=$1 lane=$2 log="$OUT/g-${P_STEM[$1]}.log" try=0 rc t0 t1 ran s benv line waits=0 cold=0 deferred=0 yowner=''
  local argv=()
  eval "argv=(${P_ARGS[$i]})"
  benv=$(box_env_of "$i")
  # The batch's own runs pass its GPU hold: the owner joins the call's env here, never box_env_of's (which feeds the ledger key).
  [ "$HOLD_UP" != 1 ] || benv="${benv:+$benv }BLOOMERY_BATCH_OWNER=$HOLD_OWNER"
  t0=$(date +%s)
  while :; do
    try=$((try + 1))
    echo "=== try $try $(date '+%F %T') lane=$lane: $(cmd_of "$i")" >> "$log"
    rc=0
    t1=$(date +%s)
    BLOOMERY_BOX_ENV="$benv" "${JUST[@]}" "${P_NAME[$i]}" ${argv[@]+"${argv[@]}"} >> "$log" 2>&1 < /dev/null &
    [ "$lane" = X ] || echo $! > "$OUT/lane-$lane.child" # the pack's driver owns its children's pids
    wait $! || rc=$?
    ran=$(($(date +%s) - t1))
    waits=$(try_waits "$log" "$try")
    if [ "$rc" -eq 75 ]; then
      # A round's batch yields to a GPU hold another batch put up (the header's GPU-hold paragraph): a
      # lane that retried would park on the lead's hold for TRIES_MAX bounds and then start in the first
      # gap the lead's own next step needs. Whatever the try count — the cause is the same at the last
      # one. The lead's --ledger batch and a plain lock queue keep the retry below.
      if [ "$LMODE" = round ]; then
        yowner=$(hold75_owner "$log" "$try")
        if [ -n "$yowner" ]; then
          echo "=== rc 75 (the GPU hold of $yowner still up): a round batch yields to the lead's hold — not retried" >> "$log"
          break
        fi
      fi
      if [ "$try" -lt "$TRIES_MAX" ]; then
        echo "=== rc 75 (lock contention): retry in ${RETRY_WAIT} s" >> "$log"
        sleep "$RETRY_WAIT"
        continue
      fi
    fi
    break
  done
  s=$(($(date +%s) - t0))
  if [ -n "$yowner" ]; then
    line="${P_STEM[$i]} rc=yield held by $yowner: not retried (a round batch yields to the lead's hold) ${s}s try=$try lane=$lane"
  else
    line="${P_STEM[$i]} rc=$rc ${s}s try=$try lane=$lane"
  fi
  [ "$waits" = 0 ] || line="$line waited=${waits}s"
  ran=$((ran > waits ? ran - waits : 0))
  cold=$(cold_of "$log" "$try")
  [ "$cold" = 0 ] || line="$line cold=1"
  deferred=$(deferred_of "$log" "$try")
  [ "$deferred" = 0 ] || line="$line deferred=$deferred"
  if [ -n "$yowner" ]; then
    # A yielded item records nothing and its rc stays 75 below, so no times row either: it ran nothing.
    [ "$LEDGER" = 0 ] || line="$line ledger=yield"
  else
    [ "$LEDGER" = 0 ] || line="$line ledger=$(ledger_record "$i" "$rc")"
  fi
  # The final try's seconds, green or red; a last try still at rc 75 got no lock and ran nothing. A cold
  # build's row goes to COLD_FILE, which no plan reads.
  if [ "$rc" != 75 ] && ! times_record "$i" "$lane" "$ran" "$cold"; then
    line="$line times=append-failed"
    echo "gate-batch: ${P_STEM[$i]}: its row did not reach $([ "$cold" = 0 ] && echo "$TIMES_FILE" || echo "$COLD_FILE") (above)" >&2
  fi
  [ -z "${P_ITEM[$i]}" ] || line="$line item=${P_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
}

skip_item() { # $1 = plan index, $2 = lane label: the ledger holds a green run of these inputs
  local i=$1 line
  line="${P_STEM[$i]} rc=skip ${P_LDET[$i]% tree=*} lane=$2"
  [ -z "${P_ITEM[$i]}" ] || line="$line item=${P_ITEM[$i]}"
  echo "$line" >> "$RUNLOG"
  echo "$line"
}

# Work stealing (the header's paragraph above the lanes): one claim per item, under DIR/claims —
# the owner claims an item before it starts it, the thief before it steals one, and an item runs
# exactly once however the two lanes race.
claim_item() { # $1 = plan index, $2 = the claiming lane: wins the item's claim?
  local f="$OUT/claims/$1"
  ( set -o noclobber; printf '%s\n' "$2" > "$f" ) 2> /dev/null
}

claimed() { [ -e "$OUT/claims/$1" ]; }

# The card this lane would run a balanced item on — its candidate for the lane's card, `-` for a
# no-card item — or empty when the item has no candidate here (a fixed item's single card, or the
# other card only).
steal_card() { # $1 = plan index, $2 = lane
  local cards want
  want=$([ "$2" = A ] && printf 3090 || printf a6000)
  rec_get "$1" cards
  case " $cards " in
    ' - ') printf -- '-' ;;
    *" $want "*) printf '%s' "$want" ;;
  esac
}

# movable: the other lane's item this lane may take — class F (the plan's own word that either
# card, or no card, runs it), not a ledger skip, not yet claimed, and held to the same neighbour
# rule as its own items (below).
stealable() { # $1 = plan index, $2 = lane
  local i=$1 cls other
  other=$([ "$2" = A ] && printf B || printf A)
  [ "${P_LANE[$i]}" = "$other" ] || return 1
  rec_get "$i" cls
  [ "$cls" = F ] || return 1
  [ "${P_LST[$i]}" != skip ] || return 1
  ! claimed "$i" || return 1
  ! tagged_neighbour "$i" || chain_quiet
  ! pool_tagged "$i" || [ ! -e "$POOL_MUTEX" ]
  [ -n "$(steal_card "$i" "$2")" ]
}

# The chain (class C, the header): one member at a time batch-wide, both lanes feeding it.
CHAIN_MUTEX=$OUT/chain.lock # taken (mkdir) before a member's run, released (rmdir) when it returns
lane_card() { [ "$1" = A ] && printf 3090 || printf a6000; }
tagged_neighbour() { # $1 = plan index: a pool or big-host item (D5's groups)
  case " ${P_TAG[$1]:-} " in *' pool '*|*' big-host '*) return 0 ;; *) return 1 ;; esac
}
# The pack's pool mutex (the header's «pack»): two [group('pool')] items spin worker pools pinned to
# the same cores, so at most one runs at a time, batch-wide — the lanes' takes and the pack hold it
# around a pool item's run. A directory mutex like the chain's; a lane that finds it up takes its
# next item instead, and one that claimed a pool item just before the mutex went up waits it out
# (bounded by the running item's BLOOMERY_GATE_BOUND).
POOL_MUTEX=$OUT/pool.lock
pool_tagged() { case " ${P_TAG[$1]:-} " in *' pool '*) return 0 ;; *) return 1 ;; esac; }
pool_take() { until mkdir "$POOL_MUTEX" 2> /dev/null; do sleep 1; done; }
pool_rel() { rmdir "$POOL_MUTEX" 2> /dev/null || true; }
run_pooled() { # $1 = plan index, $2..: run the item under the pool mutex when it is pool-tagged
  local i=$1
  shift
  if pool_tagged "$i"; then
    pool_take
    "$@"
    pool_rel
  else
    "$@"
  fi
}
chain_quiet() { # no chain item runs (the mutex is free) and none is unstarted: the neighbour rule's
  # second half — an item a neighbour could start in a gap between two members would hold the next
  # one off for its whole run, on the critical path
  [ "$CHAIN_ACTIVE" = 1 ] || return 0
  [ -e "$CHAIN_MUTEX" ] && return 1
  local i
  for i in ${CHAIN_ORDER[@]+"${CHAIN_ORDER[@]}"}; do
    claimed "$i" || return 1
  done
  return 0
}
neighbour_blocked() { # $1 = plan index: a tagged item starts only once the whole chain is done
  [ "$CHAIN_ACTIVE" = 1 ] || return 1
  tagged_neighbour "$1" || return 1
  ! chain_quiet
}
card_ok() { # $1 = plan index, $2 = a card: one of the item's candidate labels names it (a pick's
  # label is the card box.sh gives it; no card is forced)
  local labels
  rec_get "$1" labels
  case " $labels " in *" $2 "*) return 0 ;; *) return 1 ;; esac
}
chain_next() { # $1 = lane: the first unclaimed member, in the K order, whose candidates include
  # this lane's card — within the family of the first unclaimed member: a lane never opens the next
  # family while the current one holds an unclaimed member, even one only the other card can run (D3)
  local i fam='' want
  want=$(lane_card "$1")
  for i in ${CHAIN_ORDER[@]+"${CHAIN_ORDER[@]}"}; do
    claimed "$i" && continue
    [ -n "$fam" ] || fam=${P_FAM[$i]}
    [ "${P_FAM[$i]}" = "$fam" ] || break
    if card_ok "$i" "$want"; then
      printf '%s' "$i"
      return 0
    fi
  done
  return 1
}
chain_skip_take() { # $1 = lane: a member green on a candidate's key skips at 0 s — no mutex, it runs nothing
  local i
  for i in ${CHAIN_ORDER[@]+"${CHAIN_ORDER[@]}"}; do
    [ "${P_LST[$i]}" = skip ] || continue
    claimed "$i" && continue
    claim_item "$i" "chain-$1" || continue
    skip_item "$i" C
    return 0
  done
  return 1
}
chain_take() { # $1 = lane: the mutex first (a lane that loses it holds no claim it must honour), then
  # the member D3 gives this lane's card, its placement rewritten onto it (as a steal's is), run, release
  local lane=$1 i card cards
  mkdir "$CHAIN_MUTEX" 2> /dev/null || return 1
  i=''
  i=$(chain_next "$lane") || { rmdir "$CHAIN_MUTEX" 2> /dev/null || true; return 1; }
  claim_item "$i" "chain-$lane" || { rmdir "$CHAIN_MUTEX" 2> /dev/null || true; return 1; }
  card=$(lane_card "$lane")
  rec_get "$i" cards
  case " $cards " in *" $card "*) switch_candidate "$i" "$card" ;; esac
  run_item "$i" "$lane"
  rmdir "$CHAIN_MUTEX" 2> /dev/null || true
  return 0
}
takeable() { # $1 = plan index, $2 = lane: a lane item (class A, B or F — never the chain's C or
  # lane X's) whose card candidates allow this lane's card, or no card at all
  local cls labels
  rec_get "$1" cls labels
  case $cls in
    A | B | F) ;;
    *) return 1 ;;
  esac
  case " $labels " in
    ' none ') return 0 ;;
    *" $(lane_card "$2") "*) return 0 ;;
    *) return 1 ;;
  esac
}
take_next() { # $1 = lane, $2 = the variable for the index: a green skip first (it costs 0 s), then
  # the longest-est unclaimed item this lane may take — its card's fixed items and any balanced one,
  # longest first, the free lane's own choice (no static assignment) — that passes the neighbour
  # rule; an item the rule or the lease holds is skipped for now, not dropped
  local lane=$1 var=$2 j best=-1 be=-1 exp labels cls cards card
  for j in ${ORDER[@]+"${ORDER[@]}"}; do
    claimed "$j" && continue
    takeable "$j" "$lane" || continue
    if [ "${P_LST[$j]}" = skip ]; then
      printf -v "$var" '%s' "$j"
      return 0
    fi
    if neighbour_blocked "$j"; then continue; fi
    if pool_tagged "$j" && [ -e "$POOL_MUTEX" ]; then continue; fi # another pool item runs (the pack's mutex)
    rec_get "$j" exp labels
    if [ "$exp" -gt "$be" ]; then best=$j be=$exp; fi
  done
  [ "$best" -ge 0 ] || return 1
  rec_get "$best" cls cards labels
  # A balanced item taken onto this lane's card is today's steal: the timing lease is probed first
  # (never a card while it is held) and the placement moves with it. A fixed item is its lane's own.
  if [ "$cls" = F ] && [ "$labels" != none ]; then
    lease_free_for_steal || return 1
    card=$(lane_card "$lane")
    case " $cards " in
      *" $card "*) switch_candidate "$best" "$card" ;;
    esac
  fi
  printf -v "$var" '%s' "$best"
  return 0
}
lane_blocked() { # $1 = lane: something this lane still waits for — a chain member its card could
  # run, or any unclaimed item it may take (the neighbour rule or the lease holds it for now)
  local lane=$1 i
  [ "$CHAIN_ACTIVE" = 1 ] || return 1
  for i in ${CHAIN_ORDER[@]+"${CHAIN_ORDER[@]}"}; do
    if ! claimed "$i" && card_ok "$i" "$(lane_card "$lane")"; then return 0; fi
  done
  for i in ${ORDER[@]+"${ORDER[@]}"}; do
    if ! claimed "$i" && takeable "$i" "$lane"; then return 0; fi
  done
  return 1
}

# Rewrite a stolen item's placement onto this lane's card: the env, the times label and the ledger
# key, state and candidate move to the item's other candidate, so the row and the record name the
# card it ran on. A no-card item needs none of it (its one candidate runs anywhere).
switch_candidate() { # $1 = plan index, $2 = the lane's card (or -)
  local i=$1 card=$2 env cards labels c pos=-1 k=0 l='' lc=0 j
  rec_get "$i" env cards labels
  [ "$card" = - ] || env="${env:+$env }BLOOMERY_GATE_CARD=$card"
  P_ENV[$i]=$env
  for c in $cards; do
    [ "$c" = "$card" ] && pos=$k
    k=$((k + 1))
  done
  [ "$pos" -ge 0 ] || return 0
  for c in $labels; do
    [ "$lc" = "$pos" ] && l=$c
    lc=$((lc + 1))
  done
  j=$((${R_C0[$i]} + pos))
  P_TCARD[$i]=$l
  P_CI[$i]=$j P_KEY[$i]=${C_KEY[$j]} P_LST[$i]=${C_LST[$j]} P_LDET[$i]=${C_LDET[$j]}
}

# The timing-lease probe of a steal: lease_free through box.sh, READONLY so there is no rsync and
# no guard wait (the probe is itself the read — the shared-lock probe, never an exclusive flock).
# Free prints 0; held, or a lease that cannot be tested (named), prints nonzero and the steal takes
# no card.
lease_free_for_steal() {
  local out rc=0
  out=$(BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 "$ROOT/tools/box.sh" '. tools/ref/lease-probe.sh && lease_free' 2>&1) || rc=$?
  if [ "$rc" != 0 ] && [ "$rc" != 1 ]; then
    echo "gate-batch: the steal's lease probe through box.sh failed (rc $rc): $out — taking no card until it reads free" >&2
  fi
  [ "$rc" = 0 ]
}

# The steal pass: while the other lane has an unstarted movable item, take the next one in run
# order onto this lane's card. A steal onto a card probes the timing lease first and waits it out
# in 1 s slices (up to RETRY_WAIT) while candidates remain: never a card while the lease is held,
# and no lane sleeps past the last candidate. The loop always ends: every turn runs an item, loses
# a claim, waits a slice, or finds no candidate.
steal_pass() { # $1 = lane
  local lane=$1 i j card k
  while :; do
    i=-1
    for j in ${ORDER[@]+"${ORDER[@]}"}; do
      if stealable "$j" "$lane"; then i=$j; break; fi
    done
    [ "$i" -ge 0 ] || return 0
    card=$(steal_card "$i" "$lane")
    if [ "$card" != - ] && ! lease_free_for_steal; then
      for ((k = 0; k < STEAL_WAIT; k++)); do
        sleep 1
        i=-1
        for j in ${ORDER[@]+"${ORDER[@]}"}; do
          if stealable "$j" "$lane"; then i=$j; break; fi
        done
        [ "$i" -ge 0 ] || return 0
      done
      continue
    fi
    claim_item "$i" "steal-$lane" || continue
    switch_candidate "$i" "$card"
    run_pooled "$i" run_item "$i" "$lane"
  done
}

# The pack (the header's «pack»): the X items placed onto the free cards. The driver runs in the
# main process — its death is the batch's — and launches items as background children whose pids go
# to lane-X-<i>.child (the stop trap's lane-*.child glob). One item a card; the running items' bytes
# within PACK_BUDGET (a lone item always starts: the budget bounds pairs, and one load's own check is
# its own); an m-flagged item alone; a both-cards item needs both cards. Launches take the
# longest-est eligible item (the lanes' own rule), a two-candidate item the free card (the A6000
# first). Green skips claim and record at once at the green candidate's key. The loop always ends:
# every turn launches, reaps, or sleeps a second.
run_pack() {
  local t0 i j k labels exp res card best be bc launched alldone p cards
  t0=$(date +%s)
  local -a XP=()
  for i in ${ORDER[@]+"${ORDER[@]}"}; do [ "${P_LANE[$i]}" = X ] && XP+=("$i"); done
  local -a R_PID=() R_CARD=() R_BYTES=() R_M=()
  for i in "${XP[@]}"; do
    rec_get "$i" res
    # Every X record ends in its resource field, `<flag> <bytes>`; anything else is a misread record,
    # never an item of 0 bytes and no m flag free to run beside anything.
    case $res in
      [-m]' '[0-9]*) ;;
      *) RC=70 die "the pack: ${P_STEM[$i]}'s resource field reads '$res', not '<flag> <bytes>' (the plan record's res field)" ;;
    esac
    R_PID[$i]='' R_CARD[$i]='' R_M[$i]=0
    # shellcheck disable=SC2086 # the resource field: a flag and a byte count, no spaces inside
    set -- $res
    [ "$1" = m ] && R_M[$i]=1
    R_BYTES[$i]=$2
  done
  local nrun=0 mrun=0 sumb=0 b3090=0 ba6000=0
  while :; do
    # the green skips first: at the green candidate's key (the pack runs nothing for them)
    for i in "${XP[@]}"; do
      [ -z "${R_PID[$i]}" ] || continue
      claimed "$i" && continue
      j=${R_C0[$i]}
      if [ $((i + 1)) -lt "$N" ]; then k=${R_C0[$((i + 1))]}; else k=$NC; fi
      card=''
      for ((; j < k; j++)); do
        if [ "${C_LST[$j]}" = skip ]; then
          rec_get "$i" cards
          p=$((j - R_C0[i] + 1))
          # shellcheck disable=SC2086 # the candidates, one word each
          set -- $cards
          eval "card=\${$p}"
          break
        fi
      done
      [ -n "$card" ] || continue # not green anywhere: the launch pass below decides
      [ "$card" = - ] || switch_candidate "$i" "$card"
      claim_item "$i" pack || continue
      skip_item "$i" X
      R_PID[$i]=x
    done
    # launches: the longest-est eligible item, again and again while one fits
    launched=1
    while [ "$launched" = 1 ]; do
      launched=0 best=-1 be=-1 bc=''
      for i in "${XP[@]}"; do
        [ -z "${R_PID[$i]}" ] || continue
        claimed "$i" && continue
        if [ "${R_M[$i]}" = 1 ]; then
          [ "$nrun" = 0 ] || continue # an m item starts with nothing else running
        else
          [ "$mrun" = 0 ] || continue # and nothing starts beside one
          [ "$nrun" = 0 ] || [ $((sumb + ${R_BYTES[$i]})) -le "$PACK_BUDGET" ] || continue
        fi
        rec_get "$i" labels
        case $labels in
          none) card=- ;;
          both)
            [ "$b3090" = 0 ] && [ "$ba6000" = 0 ] || continue
            card=both ;;
          3090)
            [ "$b3090" = 0 ] || continue
            card=3090 ;;
          a6000)
            [ "$ba6000" = 0 ] || continue
            card=a6000 ;;
          '3090 a6000')
            if [ "$ba6000" = 0 ]; then card=a6000; elif [ "$b3090" = 0 ]; then card=3090; else continue; fi ;;
          *) continue ;;
        esac
        rec_get "$i" exp
        if [ "$exp" -gt "$be" ]; then best=$i be=$exp bc=$card; fi # the card this item would take
      done
      [ "$best" -ge 0 ] || break
      i=$best card=$bc
      claim_item "$i" pack || continue
      rec_get "$i" cards
      case " $cards " in *" $card "*) switch_candidate "$i" "$card" ;; esac
      R_CARD[$i]=$card
      run_item "$i" X &
      R_PID[$i]=$!
      echo "${R_PID[$i]}" > "$OUT/lane-X-$i.child"
      case $card in
        3090) b3090=1 ;;
        a6000) ba6000=1 ;;
        both) b3090=1 ba6000=1 ;;
      esac
      sumb=$((sumb + R_BYTES[i]))
      [ "${R_M[$i]}" = 1 ] && mrun=1
      nrun=$((nrun + 1))
      launched=1
    done
    # reaping: a dead child releases its card, its bytes and the m hold
    for i in "${XP[@]}"; do
      case ${R_PID[$i]} in '' | x) continue ;; esac
      kill -0 "${R_PID[$i]}" 2> /dev/null && continue
      wait "${R_PID[$i]}" 2> /dev/null || true
      case ${R_CARD[$i]} in
        3090) b3090=0 ;;
        a6000) ba6000=0 ;;
        both) b3090=0 ba6000=0 ;;
      esac
      sumb=$((sumb - R_BYTES[i]))
      [ "${R_M[$i]}" = 1 ] && mrun=0
      nrun=$((nrun - 1))
      R_PID[$i]=x
      rm -f "$OUT/lane-X-$i.child"
    done
    # the end: every item claimed and done, nothing running
    alldone=1
    for i in "${XP[@]}"; do case ${R_PID[$i]} in x) ;; *) alldone=0 ;; esac; done
    if [ "$alldone" = 1 ] && [ "$nrun" = 0 ]; then break; fi
    sleep 1
  done
  echo $(($(date +%s) - t0)) > "$OUT/lane-X.s"
}

run_lane() { # $1 = lane label; a lane of the chain's batch (the header's class C): when free it
  # takes, in order, a green member's skip, the member D3 gives its card (under the mutex), or the
  # longest-est item it may take (its card's fixed items and any balanced one, the free lane's own
  # choice — today's steal is that take, so the pass is gone here); then it waits in 1 s slices
  # while a member it could run or an item the rule or the lease holds remains, and else ends.
  # Without a chain (lane X, --lanes 1, the fixture tier, a chainless list) it is today's loop: its
  # items in ORDER, then the steal pass.
  local lane=$1 t0 i
  t0=$(date +%s)
  if [ "$CHAIN_ACTIVE" = 1 ] && [ "${#CHAIN_ORDER[@]}" -gt 0 ] && { [ "$lane" = A ] || [ "$lane" = B ]; }; then
    while :; do
      chain_skip_take "$lane" && continue
      chain_take "$lane" && continue
      if take_next "$lane" i; then
        if [ "${P_LST[$i]}" = skip ]; then
          skip_item "$i" "$lane"
        else
          claim_item "$i" "$lane" || continue # the other lane took it while this one was busy
          run_pooled "$i" run_item "$i" "$lane"
        fi
        continue
      fi
      lane_blocked "$lane" || break
      sleep 1
    done
  else
    for i in ${ORDER[@]+"${ORDER[@]}"}; do
      [ "${P_LANE[$i]}" = "$lane" ] || continue
      if [ "${P_LST[$i]}" = skip ]; then
        skip_item "$i" "$lane"
      else
        claim_item "$i" "$lane" || continue # the other lane stole it while this one was busy
        run_pooled "$i" run_item "$i" "$lane"
      fi
    done
    case $lane in
      A | B) [ "$LANES" = 2 ] && steal_pass "$lane" ;;
    esac
  fi
  echo $(($(date +%s) - t0)) > "$OUT/lane-$lane.s"
}

has_lane() { local i; for ((i = 0; i < N; i++)); do [ "${P_LANE[$i]}" = "$1" ] && return 0; done; return 1; }

LANES_RUN=""
for lane in A B; do
  if has_lane "$lane"; then
    run_lane "$lane" &
    echo $! > "$OUT/lane-$lane.pid"
    LANES_RUN="$LANES_RUN $lane"
  fi
done
# Every lane is watched until its sentinel exists: a lane that dies — with the chain's mutex held,
# or anywhere else — must not leave the other spinning on what it holds. A dead-but-unreaped lane
# reads as state Z to ps, and a pid ps itself cannot find reads as ps's own rc (`wait` would block on
# the survivor instead); the sentinel is each lane's last act, so a lane that wrote it is only waiting
# to be reaped below. A lane is declared dead on two sweeps in a row, a second apart, that each
# found it gone: a busy machine can make one ps read fail, and a false death would kill a working
# lane. Each lane's sighting is its own (two lanes that die together are each declared), and a
# sweep that finds a lane alive forgets its last sighting.
dead_prev=''
while :; do
  miss=0
  for lane in $LANES_RUN; do [ -f "$OUT/lane-$lane.s" ] || miss=1; done
  [ "$miss" = 0 ] && break
  dead_now=''
  for lane in $LANES_RUN; do
    [ -f "$OUT/lane-$lane.s" ] && continue
    st=$(ps -o state= -p "$(cat "$OUT/lane-$lane.pid" 2>/dev/null)" 2>/dev/null) && [ -n "$st" ] && [ "$st" != Z ] && continue
    case " $dead_prev " in
      *" $lane "*) ;;
      *) dead_now="$dead_now $lane"; continue ;;
    esac
    for f in "$OUT"/lane-*.pid "$OUT"/lane-*.child; do
      if [ -f "$f" ]; then kill -TERM "$(cat "$f")" 2> /dev/null || true; fi
    done
    RC=70 die "lane $lane ended without its sentinel ($OUT/lane-$lane.s) — see run.log"
  done
  dead_prev=$dead_now
  sleep 1
done
for lane in $LANES_RUN; do
  wait "$(cat "$OUT/lane-$lane.pid")" || true
done
# Every lane is waited for before any is judged: a lane that died must not leave the other running.
for lane in $LANES_RUN; do
  [ -f "$OUT/lane-$lane.s" ] || RC=70 die "lane $lane ended without its sentinel ($OUT/lane-$lane.s) — see run.log"
done
if has_lane X; then run_pack; fi
trap - INT TERM

recorded=$(grep -c ' rc=' "$RUNLOG" || true)
[ "$recorded" -eq "$NT" ] || RC=70 die "$NT items planned, $recorded recorded in $RUNLOG"
green=$(grep -c ' rc=0 ' "$RUNLOG" || true)
skipped=$(grep -c ' rc=skip ' "$RUNLOG" || true)
yielded=$(grep -c ' rc=yield ' "$RUNLOG" || true)
red=$((NT - green - skipped - yielded - ND))
lane_s() { if [ -f "$OUT/lane-$1.s" ]; then cat "$OUT/lane-$1.s"; else echo 0; fi; }
done_line="DONE total=$NT red=$red"
[ "$LEDGER" = 0 ] || done_line="$done_line skipped=$skipped"
[ "$yielded" = 0 ] || done_line="$done_line yielded=$yielded"
done_line="$done_line wall=$(($(date +%s) - T0))s laneA=$(lane_s A)s laneB=$(lane_s B)s"
if has_lane X; then done_line="$done_line laneX=$(lane_s X)s"; fi
for ((i = 0; i < N; i++)); do
  if [ "${P_NAME[$i]}" = lint ]; then
    if [ "${P_LST[$i]}" = skip ]; then
      done_line="$done_line lint_warnings=skip"
    else
      done_line="$done_line lint_warnings=$(grep -c '^warning:' "$OUT/g-${P_STEM[$i]}.log" || true)"
    fi
    break
  fi
done
# The fixture tier's clauses left to the real tier, summed over the items' lines (the lanes ran in subshells).
if [ "$TIER" = fixture ]; then
  done_line="$done_line deferred=$(awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^deferred=[0-9]+$/) n += substr($i, 10) } END { print n + 0 }' "$RUNLOG")"
  [ "$ND_RO" = 0 ] || done_line="$done_line real_only=$ND_RO"
  [ "$ND_NF" = 0 ] || done_line="$done_line no_fixture=$ND_NF"
  [ "$ND" = 0 ] || done_line="$done_line deferred_list=$DEFER_LIST"
fi
echo "$done_line" >> "$RUNLOG"
echo "$done_line"
if [ "$red" -ne 0 ]; then
  grep -v ' rc=0 ' "$RUNLOG" | grep ' rc=' | grep -v ' rc=skip ' | grep -v ' rc=deferred ' | grep -v ' rc=yield ' |
    while read -r stem _; do echo "red: $stem  $OUT/g-$stem.log"; done
  exit 1
fi
