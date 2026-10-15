# Checking a change on your own machine

This page is for a contributor who wants to check a change before opening a pull request. It says what to run on a Mac,
on a Linux machine with no card and on a Linux machine with an NVIDIA card, what a green result shows, and what to paste
into the pull request. How a pull request is then verified and merged is in `CONTRIBUTING.md`, "How a pull request is
verified and merged"; what a number from your own card must carry is in its "Rules", "Numbers carry their conditions".

One line per machine. On any OS, `just contrib-check --tier t0`. On a Mac, also `just mac-static`, once its environment
file exists. On x86_64 or aarch64 Linux, with or without a card, `just contrib-check --tier t1`. On Linux with an NVIDIA
card, also the gates themselves in local mode: `BLOOMERY_BOX_LOCAL=1 BLOOMERY_TIER=fixture just gate-<x>`. A Mac never
runs a gate. Without `BLOOMERY_BOX_LOCAL=1`, a `just gate-*` recipe sends its command over ssh to the maintainers'
workstation (the box), which a contributor cannot reach.

Two pieces are not built yet: T2, one command that runs your gate list on your card, and the command that prints that
list. Section 3 gives the steps to use until then, and section 6 lists everything that is still only designed.

Words this page uses. A *gate* is a `just gate-*` recipe that checks the code against a reference, most of them on a
card. The *real tier* runs a gate on the model's real file; the *fixture tier* (`BLOOMERY_TIER=fixture`) runs it on the
family's *fixture*, a small file of the same architecture with generated weights. A *real-only* gate has no fixture-tier
form, and a *real-file* gate still opens the real file at the fixture tier; neither is on a contributor's list
(section 3). The `fx_` *reference sets* are what the fixture-tier gates compare against, dumped by `ik` (ik_llama.cpp,
the numerical reference). The *lease* is the machine-wide lock a timing run takes, and a *hold* is a file that makes
other box commands wait while it exists. An *arm* is one configuration a timing run measures, and the *ruler* is how far
runs of the same binary scatter on one machine. *DSpark* is V4.1's draft model for speculative decoding.

## 1. What to run on your machine

| Machine | Command | A green result shows | Needs |
|---|---|---|---|
| Any OS, a Mac included | `just contrib-check --tier t0` | formatting, the pure crates' tests, the Python tool self-tests and every `tools/check-*.sh` pass | the pinned nightly through rustup, `python3` 3.11 or later with numpy and PIL, `just` (section 2) |
| Mac | `just mac-static [BASE]` | formatting, clippy and the build shapes `BASE`'s change touches check as x86_64 Linux, and every check script passes; no test or gate runs | the environment file the header of `tools/mac-check.sh` describes (section 2) |
| x86_64 or aarch64 Linux, with or without a card | `just contrib-check --tier t1` | T0, then the tests of every crate that reaches no card | the CUDA headers, libclang, numpy and PIL (section 2) |
| Linux with an NVIDIA card | `BLOOMERY_BOX_LOCAL=1 BLOOMERY_TIER=fixture just gate-<x>`, for each gate of your list | that gate passes on your card at the fixture tier | an environment file, the fixtures and the reference sets (sections 3 and 4) |
| Any machine with the `claude` CLI logged in | `just pre-review` before each push of a pull request branch | Shingler's reviewer, on your committed HEAD with the pull request review's own instructions, finds no must-fix (exit 0) and prints one pull-request body line with the full HEAD sha; exit 1 lists must-fix items; a base behind origin/main is refused by name (64) | `git`, `gh` for an incremental review of an open pull request and the slag ledger, a Claude login |

Paste what each command printed into the pull request: for `contrib-check`, its header block, its unit lines and its
`DONE` line; for a gate, its summary lines and its exit code. Name the machine as `CONTRIBUTING.md` asks ("Pull requests
for your own hardware"). CI runs `just contrib-check --tier t0` on every pull request (`.github/workflows/pr.yml`).

`just pre-review` (`tools/review/local.py`) runs only the reviewer by default, in at most 10 minutes; `--full` adds the
validator as the pull request review runs it. Nothing is posted, and the pull request's own review stays the review of
record. Its output stays in `~/.cache/bloomery/pre-review/<branch>-<sha8>/`. Do not push while that review is running:
a push starts it again. Before the reviewer, a branch that touches `tools/`, the justfile, `.cargo/` or `crates/` runs
`tools/check-levers.sh` (text only, about 10 s; `PRE_CHECKS` in `local.py` lists such checks): red ends the run 1 with its
output, before the reviewer's cost.

## 2. `just contrib-check` (T0 and T1)

`just contrib-check [--tier t0|t1] [--only REGEX]` runs one script, `tools/contrib-check.sh`. The default tier is T1 on
x86_64 Linux and T0 everywhere else, aarch64 Linux included, where `--tier t1` runs T1. `--tier t1` on any other host is
a named exit 64, and `--tier t2` prints `T2 is not built yet` and exits 64. `--only REGEX` runs just the units whose name
matches, for example a red unit again. The script reads every list from its owner and copies none: the pure crates from
`python3 tools/recipes.py pure-crates`, the Python self-tests from the `selftests=(` array of `tools/check-recipes.sh`,
and the host units from `python3 tools/recipes.py host-nightly`.

| Tier | Where | Runs |
|---|---|---|
| T0 | any OS with the pinned nightly (rustup, rustfmt) and `python3` 3.11 or later (`tools/recipes.py` reads `Cargo.lock` with `tomllib`) with numpy and PIL | `cargo fmt --all -- --check`; `cargo test -p <crate>` for each crate of `recipes.py pure-crates`; `python3 tools/recipes.py --self-test`; each Python tool self-test `check-recipes.sh` lists (some import numpy and PIL, which CI installs before T0); every `tools/check-*.sh`. Each check script reads the tree and runs no cargo and no box command, so none is left out |
| T1 | x86_64 or aarch64 Linux, no card needed | T0, then every unit of `recipes.py host-nightly` that T0 did not already run under the same name, each `bash tools/gate.sh --release --locked -p <crate>`. Needs what `tools/nightly/install.sh` lists: the CUDA 13.3 headers under `CUDA_TOOLKIT_PATH`, libclang for bindgen, python3-numpy and python3-pil |

T1 leaves out the six device crates (`python3 tools/recipes.py host-nightly --excluded` names them). A device crate is
built only with `cargo oxide`, never with plain `cargo` (`AGENTS.md`, "Never"), and the gate recipes of section 3 build
them that way.

`RUSTFLAGS`: on x86_64 Linux the script sets `-C target-cpu=native` for every cargo unit unless `RUSTFLAGS` is already
set, because a binary built for the repo's `znver3` may stop with an illegal instruction on another CPU (`docs/BUILD.md`,
"The CPU flag"). On aarch64 it sets nothing: the repo's `.cargo/config.toml` already builds for the native CPU there. The
nightly sets `x86-64-v3` for its own machine (`tools/nightly/run.sh:55`), and both callers share one unit runner,
`tools/nightly/units.sh`.

Output: a header (commit, host, toolchain, `RUSTFLAGS`, bound, log directory), one line per unit,
`<name> rc=<n> <seconds>s`, under a red one its failing tests or first red lines and its log path, then
`DONE tier=… total=… red=… wall=…s`. Each unit's log is in `target/contrib-check/<time>/<name>.log`. Every unit, in
both tiers, runs under a bound of `CONTRIB_UNIT_BOUND` seconds (default 1200) through GNU `timeout`, or `gtimeout` when
only that is installed; a host with neither (macOS as shipped) runs units unbounded, and the header says so. Exit: 0
every unit green, 1 a unit red, 64 usage, 69 a prerequisite is missing (named, with how it is made), 70 a list the
script reads is empty or `--only` matches no unit. `CONTRIB_INJECT_FAIL=1` adds one deliberately failing unit, so the
red path can be shown to fail first.

Exceptions:

- **`just` is a T0 prerequisite.** Three T0 units call it: `recipes-self-test`, `check-recipes` and the self-test of
  `tools/check-comment-only.py`. The script's own preflight checks only `python3` and `rustup`
  ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)).
- **aarch64 Linux.** `tools/nightly/install.sh` installs the x86_64 packages only. One DGX Spark ran T1 with the CUDA 13.3
  headers of the `nvidia/cuda:13.3.1-devel-ubuntu24.04` image, LLVM 21.1.8's libclang, the distribution's numpy and
  PIL, and `just` built from source (same link).
- **A locale other than English.** `check-recipes` runs `tools/ref/lease.sh --self-test`, and one of its cases matches
  the English `strerror` text. On a host whose libc messages are translated (`LANG=ko_KR.utf8`, say) that case fails, on
  main as well. Run under `LC_ALL=C` ([#38](https://github.com/midagedev/bloomery/pull/38), "What I ran into").
- **The Mac's environment file.** `just mac-static` checks the Linux build, so bindgen needs the CUDA 13.3 headers and
  an x86_64 Linux `/usr/include` on the Mac, beside libclang from the Xcode command line tools. The header of
  `tools/mac-check.sh` copies the two directories from the maintainers' workstation. Any x86_64 Linux host with CUDA 13.3
  has the same two, `/usr/local/cuda-13.3/include` and `/usr/include`; without CUDA there, the headers are in the three
  `.deb` files `tools/nightly/install.sh` names.

## 3. Gates on your own card: local mode

`BLOOMERY_BOX_LOCAL=1` makes `tools/box.sh` run, in your tree and with `bash`, the same command line it would send to the
box: the guard, the model profile, the tier and the exports, with no ssh and no rsync. So every `just gate-*` recipe runs
unchanged on a Linux machine with a card of its own. The header of `tools/box.sh` is the full contract; this section is
what a contributor sets up and what to expect.

**Set up once:**

1. **The environment file.** `BLOOMERY_ENV_FILE`, default `~/bloomery-env.sh`, is sourced before the command; a
   missing file exits 69. It exports the toolchain of `docs/BUILD.md` ("Toolchain") in four variables: `PATH` (cargo,
   `cargo-oxide`, LLVM 21 and CUDA 13.3's `bin`), `CUDA_TOOLKIT_PATH`, `LIBCLANG_PATH` and `CUDA_OXIDE_LLC`, the path of
   LLVM 21's `llc`. Without `CUDA_OXIDE_LLC`, cuda-oxide takes the Rust toolchain's `llc` first (`docs/BUILD.md`), and
   the pinned nightly's LLVM is 23. One DGX Spark run exported these four, with LLVM 21.1.8's `llc`
   ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082), "Reproduce"). T1 (section 2) needs
   `PATH` and `CUDA_TOOLKIT_PATH` as well. Local mode unsets `CUDA_OXIDE_BACKEND` after the file, so `cargo oxide`
   builds its own backend from the pinned fork rev.
2. **The pinned `cargo-oxide`.** Before a command that builds with `cargo oxide`, `tools/box.sh` checks with
   `cargo install --list` that the `cargo-oxide` on `PATH` was installed from the pinned rev, and stops with exit 70
   when it was not. The install command is in `docs/BUILD.md` ("Toolchain").
3. **The lock directory.** `BLOOMERY_LOCK_DIR`, an absolute path, default `~/.cache/bloomery/locks`, made when missing.
   It holds the timing lease, the holds and the gate locks. `tools/gpu-gate.sh` takes one lock there,
   `bloomery-gate-local.lock`, for whatever card the machine shows, so on a machine with several cards the gates run one
   at a time.
4. **The card.** `BLOOMERY_CARD` unset, `3090` or `a6000` runs on your card, and the command sees
   `BLOOMERY_BOX_CARD=local`. `tools/box.sh` reads `BLOOMERY_CARD` before its local branch, so any other value is
   refused (below). `CUDA_VISIBLE_DEVICES` is yours, or the environment file's when it sets one. A recipe's own
   `BLOOMERY_GATE_CARD`, the card it asks `tools/gpu-gate.sh` for, may be unset, `3090`, `a6000` or `any` in local mode.
5. **The CPU flag, on x86_64.** The cargo files build for Zen 3 (`znver3`) on x86_64, and local mode does not change
   that, so on another x86_64 CPU a gate binary may stop with an illegal instruction. Set
   `RUSTFLAGS='-C target-cpu=native'` in your shell or the environment file. Plain `cargo` then takes it in place of
   `.cargo/config.toml`'s flags, and `cargo oxide` adds it after `.cargo/cuda-oxide.toml`'s `extra-rustflags`, so the
   later `target-cpu` wins. On an RTX 5090 host (x86_64) at `b289130a`, with cargo-oxide 0.2.1 at the pinned rev, this
   build passed `-C target-cpu=x86-64-v2` after `-C target-cpu=znver3 -C target-feature=-sse4a` in every `rustc`
   call:

       RUSTFLAGS='-C target-cpu=x86-64-v2' cargo oxide build --arch sm_86 -- \
         -p bloomery-gpu-gates --features gpu --release --bin gate_kquant -v

   The other way is to edit both files (`docs/BUILD.md`, "The CPU flag"). On aarch64 the files already build for the
   native CPU.
6. **The inputs.** `BLOOMERY_DATA` names your copy of the reference sets and reaches the command as you set it; unset,
   it is the maintainers' `/root/bloomery-data`. The fixtures sit under `/models/fixtures`. Section 4 says how to make
   both.

**Run one gate:**

    BLOOMERY_BOX_LOCAL=1 BLOOMERY_TIER=fixture BLOOMERY_DATA=<your sets> just gate-<x>

`BLOOMERY_TIER=fixture` makes `tools/ref/ref-paths.sh` resolve the family's fixture and the card budget its header names.
Each gate runs under `BLOOMERY_GATE_BOUND` seconds, default 900 (`tools/gate-bound.sh`); one DGX Spark run set it to 1800
for every item with `BLOOMERY_BOX_ENV='BLOOMERY_TIER=fixture BLOOMERY_GATE_BOUND=1800'` (same link).

A recipe the fixture tier cannot run ends in one of three ways:

- a real-only gate stops before it builds with exit 66 from `tools/ref/real-only.sh`, which names it;
- a gate of a family with no fixture stops with exit 66 from `ref-paths.sh`, which names the family;
- a gate of a family whose fixture row is `self` (no fixture file) runs on the profile's real file, and the gate
  fence, `Tier::check_file` in `crates/gpu-gates/src/tier.rs`, refuses that file, so the gate ends red.

A *variant item* runs a recipe on a variant fixture: the family's fixture requantized to one type map of
`tools/fixture-variants.tsv`. `just affected` lists it as `<recipe>@BLOOMERY_FIXTURE_VARIANT=<tag>`, and
`tools/gate-batch.sh` names a recipe's second and later items `<recipe>-2`, `<recipe>-3`. Run one by adding
`BLOOMERY_FIXTURE_VARIANT=<tag>` to `BLOOMERY_BOX_ENV`. `just fixture-requant` makes the variant fixture with `ik`'s
`llama-quantize`; without the variant fixture the item stops with exit 66. Its reference sets are not published
(section 6).

**Refused (exit 64):** a host that is not Linux (a Mac runs `just mac-static`); a `BLOOMERY_CARD` other than `3090` or
`a6000`: `both`, so every two-card recipe, and any other name; `BLOOMERY_BOX_QUEUE=1`; a relative `BLOOMERY_LOCK_DIR`,
or one that is `/root`; an environment file that moves the lock directory or sets a `BLOOMERY_ARCH` other than the one
the recipe expanded; and the box itself.

**Which gates.** The contributor's list is `just affected` minus the real-file and real-only gates and the timing
runners; a fixture-only gate stays ([`dgx-spark.md`](dgx-spark.md) section 11, answer 4). No command prints it yet
(section 6). Until one does, build it as one DGX Spark run did
([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082), "Reproduce"):

    python3 tools/recipes.py affected <base> --no-box > affected.txt
    bash tools/gate-batch.sh --list affected.txt --tier fixture --dry-run

`<base>` is the upstream commit your branch starts from; `upstream/main` works, because `affected` diffs from its merge
base with `HEAD`, so it needs a git clone with that history. `--no-box` keeps `affected` from reading the box's build
information over ssh. The dry run touches neither the box nor an output directory, and prints one `lane` line per item.
A line `lane -  <item>` followed by `deferred to the real tier (<kind>)` is an item the fixture tier does not run:
real-only, real-file, or of a family with no fixture. A line `lane A`, `lane B` or `lane X` is an item it runs, with its
command. Your list is the items of the second kind. Keep `gate-ptx-spill` on it only where its tables pin your PTX
arch and card capability: `sm_86` on cc 8.6, `sm_86` or `sm_121` on cc 12.1 ([`dgx-spark.md`](dgx-spark.md) section 6).

**What the card name closes.** Two machines whose card is neither the RTX 3090 nor the RTX A6000 ran a fixture-tier list
in local mode. The two runs used different commits and different lists, so their counts do not add up:

- One DGX Spark (GB10), main `697790c4`, the 93 items of the rule above: 14 green items open the card, and the other 15
  green items do not open it ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)).
- One RTX 5090, main `e8e32a38`, 42 fixture-tier gates: 18 passed, host gates included
  ([#54](https://github.com/midagedev/bloomery/issues/54), "The rest of the fixture tier"). Its `gate-gpu-kquant` red
  came from the `llc` that built the PTX; since [#63](https://github.com/midagedev/bloomery/pull/63) the gate passes
  under that `llc` (#63's runs on an RTX 5090 and a DGX Spark).

In both runs, the one gate that stopped on the card's name was `gate-gpu-qwen38-serve`: its `--place gate` opened the
RTX 3090 by name at those commits. `gate` plans the 3090's bytes on the smallest visible card with the room for them
(`ALIASES` in `crates/placement/src/placement/devices.rs`), so the item opens on any such card. A recipe outside these
two lists may still open a card by name. `gate-gpu-ds41-dspark-loop` also passes `--place gate`, but only at the real
tier; at the fixture tier it runs `--place a`. The two-card recipes stop at the `BLOOMERY_CARD` refusal above. Every other closed GPU item stopped
on an input the machine did not have, except `gate-gpu-kquant` on the RTX 5090.

## 4. Fixtures and reference sets

A fixture-tier gate reads two inputs: the family's fixture and the `fx_` reference sets. Three families have a fixture
(`crates/refset/src/fixture.rs`, `DIRS`); MiMo and the older families do not.

1. **Make `/models/fixtures` exist** on your machine; a symlink to where you keep the files is enough, and it needs
   root once ([`dgx-spark.md`](dgx-spark.md) section 11, answer 2). A set's `# model` line states the first shard's
   full path, which `first_shard` in `crates/refset/src/fixture.rs` builds from the root. The published sets were
   dumped under `/models/fixtures`, so they read as stale under another root, even though `BLOOMERY_FIXTURE_ROOT` moves
   the files.
2. **Make the fixture of each family** with the `fixture` binary of `bloomery-model`, one command a family. It runs
   `curl`, which must be on `PATH`, and on x86_64 that is not Zen 3 it needs the `RUSTFLAGS` of section 3:

       cargo run --release -p bloomery-model --bin fixture -- fetch qwen38 /models/fixtures/qwen38
       cargo run --release -p bloomery-model --bin fixture -- fetch v41 /models/fixtures/v41
       cargo run --release -p bloomery-model --bin fixture -- fetch glm5next /models/fixtures/glm5next

   The word after `fetch` is the family's name in `SOURCES` (`crates/model/src/bin/fixture.rs`), whose row gives the
   source repo, quant and revision. The directory is the one `DIRS` maps the family's architecture to (`qwen4exp` to
   `qwen38`, `deepseek41` to `v41`, `glm5next` to `glm5next`), and `fetch` refuses it when it exists already. `fetch`
   reads a few MB of headers per shard by HTTP Range and writes each shard at its listed length as a sparse file. A
   fixture's bytes are a function of the seed and the source header alone (`crates/model/src/fixture/mod.rs`), and the
   loader refuses a file too short to hold its tensors, so a header grown to the file's listed length is enough. On one
   DGX Spark, two `fixture fetch qwen38` runs and `fixture generate` from the real files, whole or cut to their headers
   and grown back, wrote the same bytes; cut and not grown back, generate refused
   ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6117747643)). The maintainers checked on their
   machine that `fixture fetch` makes the qwen38 fixture (with its MTP draft), the v41 fixture (with the r8 sidecar) and
   the glm5next fixture (at its pinned revision) byte-identical to their files. There, `du -sh` on 2026-10-10 measured
   `glm5next` 8.0 GB, `qwen38` 11 GB, `v41` 16 GB and `v41-r8` (the V4.1 sidecar) 7.3 GB. The V4.1 DSpark draft
   fixture cannot be fetched (its source is the maintainers' own copy), and no set of it is published.
3. **Download the reference sets** into `$BLOOMERY_DATA`. They are `ik`'s output, so a contributor cannot make them, and
   they are published as one dataset, <https://huggingface.co/datasets/midagedev/bloomery-refsets>: 6.48 GB in 17 sets
   and 29,941 files (the dry run of 2026-10-10, `just refset-publish-dry`; `du -sh` of the `fx_*` directories said
   6.4 GB). The dataset mirrors that directory, so this one command is the install (`--include` takes one family's
   sets; the dataset README lists them by name):

       hf download --repo-type dataset midagedev/bloomery-refsets --local-dir "$BLOOMERY_DATA"

   `just refset-publish-dry` is the maintainers' dry run of the dataset (`tools/refset-publish.sh`): it lists every
   file with its sha256 and says which sets are left out (the fixture variants and the DSpark draft sets) and why.

## 5. Timing on your own card

Local mode also times your card, with one runner. Under `BLOOMERY_BOX_CARD=local`, `tools/ref/timing-card.sh` times the
one card `nvidia-smi` lists, picked by UUID, or among several cards the one `BLOOMERY_TIMING_GPU` names
([#39](https://github.com/midagedev/bloomery/pull/39)). The lease and its trace, `bloomery-sittings.log`, sit in
`BLOOMERY_LOCK_DIR`.

- **The runner.** Only `tools/ref/depth-qwen3moe.sh` opts in, through `just depth-gpu-qwen3moe`, `depth-gpu-qwen35moe`
  and `depth-gpu-qwen4exp`. Any other runner prints the reason and exits 64 before it takes the lease. These recipes
  time a real model file, the profile's (`BLOOMERY_REF_MODEL` moves it, `tools/ref/ref-paths.sh`), not a fixture. The
  reference engine arms read the profile's `IK`, `LCPP` and `MRS` trees, which you build yourself; no recipe builds
  them.
- **The prediction card.** The lease starts a run only with a card file written before it: what the run measures, the
  band the derivation predicts and the next step for each outcome. Write it as `docs/cards/<slug>.card` in your tree
  (the format is the header of `tools/ref/card.py`; `docs/cards/example-*.card` show each kind and are refused as a
  run's card), and name it through `BLOOMERY_BOX_ENV`:

      BLOOMERY_BOX_LOCAL=1 BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/<slug>.card' just depth-gpu-qwen3moe

  Without one, the lease refuses with exit 66.
- **Placement.** Only placement `a` runs. `gate` is refused because its rows are the 3090's, and `bp` because it loads
  two cards (`tools/ref/arm-place.sh`).
- **Xid.** Before the lease, the run checks that the kernel journal answers. A user outside the `adm` and
  `systemd-journal` groups gets an empty answer, and the run stops with exit 69. After each arm, an NVRM Xid or a card
  that stopped answering fails that arm.
- **The witness.** Before and after every arm the runner prints the card (name, power limit, SM clock, clock-event
  counters), the Xid count, `cpu-freq`, the binary's hash, CPU load, the shared memory pool and the machine (OS, kernel,
  driver, `nvcc`).
- **The ruler.** The same-binary spread in `AGENTS.md` ("Know the ruler", SD 0.6 %) was measured on the maintainers'
  box; your machine's spread is its own. An A/A arm, the same binary twice, shows it, and `AGENTS.md` asks for one
  under any claim below 1 %.

A number from this runner is evidence for your pull request on your card when it carries what `CONTRIBUTING.md` lists
under "Numbers carry their conditions". It can be published in a table of its own for that card; a table never mixes
cards.

## 6. Not built yet

- **T2.** `just contrib-check --tier t2` would run the contributor's list (section 3) serially under
  `BLOOMERY_TIER=fixture`. A recipe the tier refuses by name (exit 66 from `tools/ref/real-only.sh` or `ref-paths.sh`,
  which `tools/gate-batch.sh` counts as `deferred`) would print `DEFER`, never count green. A fixture-only recipe runs
  there, because `tools/ref/fixture-only.sh` stops only the real tier. The recipe must keep `just check-recipes` green:
  it takes no lock itself, runs no bare `cargo test`, and is not a `gate-*` recipe. How it counts a two-card recipe is
  open question 4.
- **The list command.** The maintainers own it, as one derived command ([`dgx-spark.md`](dgx-spark.md) section 11,
  answer 4). Until it exists, section 3 builds the list by hand.
- **`contrib-check` flags and states:** `--base`, `--require T2`, a `report.txt` per run (the header and the log
  directory carry its facts), the `SKIP` and `DEFER` unit states (no unit is skipped in T0 or T1), and a call of
  `mac-static` on a Mac (`just mac-static` stays the Mac's static tier beside this command).
- **T1 in CI.** `.github/workflows/pr.yml` runs T0 only: T1 needs three NVIDIA `.deb` files unpacked by hand, more than
  apt packages (its header).
- **An example environment file** for local mode (reference row 2).
- **One model path resolver** for the path literals of reference row 9.
- **A root-relative set identity,** so the fixtures could sit under another root: it comes as its own change when a
  contributor needs it ([`dgx-spark.md`](dgx-spark.md) section 11, answer 2).
- **Gaps in the fixture tier the maintainers took** ([`dgx-spark.md`](dgx-spark.md) section 11, answer 18), from the
  DGX Spark run's points 1 to 5 ([#13](https://github.com/midagedev/bloomery/issues/13#issuecomment-6124298082)). Until
  they land, these items stay closed for a contributor:
  - The V4.1 DSpark draft fixture is not fetchable, and five of the gates that need it also read
    `engram/corpus-prose.ids`, which is not published.
  - `glm5next/corpus-prose.ids` and `fx_ref_glm5next_vis_taps` are not published.
  - The list rule keeps host gates that need a real file or a `just dump-ref` oracle (the V2-Lite gates) and a public
    real file (the bf16 projector of `gate-gpu-glm5next-media-seat`); a no-input class would keep them off the list.
- **Variant sets.** The dataset publishes no set of a fixture variant: its README lists them under "Left out", because
  the variant file is requantized on the maintainers' machine and cannot be made again byte for byte off it. The
  variant items (section 3) stay closed for a contributor.

## 7. Reference: the box assumptions

The design listed what a run off the box must bypass. This table keeps that list with each row's state at `3bc2a172`.

| # | Assumption | Where | State |
|---|---|---|---|
| 1 | Commands run on the maintainers' workstation after an rsync of the tree and `source ~/bloomery-env.sh` | `tools/box.sh` | **Built** ([#38](https://github.com/midagedev/bloomery/pull/38)): section 3; its tests are in `tools/ref/card-tests/run.sh` |
| 2 | The box's environment file (nightly, LLVM 21, CUDA 13.3) | `tools/box.sh` header | **Not built:** an example file from the toolchain block of `docs/BUILD.md`; what it exports is in section 3 |
| 3 | A timing lease and holds under `/root` gate every command | `tools/ref/lease-probe.sh:17` | **Built** (#38): one `BLOOMERY_LOCK_DIR` moves the lease and the holds (section 3) |
| 4 | Gate locks per card name at `/root/bloomery-gate*.lock`, the V4.1 load lock, the batch hold | `tools/gpu-gate.sh:86-89` | **Built** (#38): every lock under `BLOOMERY_LOCK_DIR`, and one local lock for the machine's card (section 3) |
| 5 | Cards are found by exact name, `NVIDIA GeForce RTX 3090` and `NVIDIA RTX A6000` | `tools/ref/cards.sh:33-34`; the card pick in `tools/box.sh` | **Built** (#38): the `BLOOMERY_CARD` values that run and the ones refused, and the gates the name still closes, are in section 3 |
| 6 | The device arch was `sm_86` in 139 `justfile` lines and `tools/gate.sh` | the justfile's `arch` variable, `tools/arch.sh` | **Done:** `BLOOMERY_ARCH`, default `sm_86`, read by the justfile and by the tools on the box, forwarded by `box.sh`; `check-recipes` fails on a literal. `gate-ptx-spill` reads a table per PTX arch and card capability (`tools/ref/ptx-shapes.tsv` is sm_86 on cc 8.6; sm_86 and sm_121 on cc 12.1 sit in `tools/ref/ptx-shapes/`), `ptx-scan` takes sm_86, sm_120 and sm_121, and `just narrow` and `release-build` refuse another arch by name |
| 7 | The cuda-oxide backend is a prebuilt `.so` per rev, and `cargo oxide` refuses (rc 70) without it | `tools/box.sh` | **Built** (#38): local mode lets `cargo-oxide` build its own backend (section 3) |
| 8 | The CPU flag `znver3` in both cargo files | `.cargo/config.toml`, `.cargo/cuda-oxide.toml` | **aarch64 done** (`f0139df3`): each file has a table per target triple, `target-cpu=native` for aarch64. **x86_64:** the files stay `znver3`, and a caller's `RUSTFLAGS` reaches both plain `cargo` and `cargo oxide` builds (measured, section 3) |
| 9 | Real model files at `/models/...`: ~~133~~ 151 lines with a path literal in `crates/**/*.rs`, comments included (`git grep -n '"/models/\|"/root/\|/home/user' -- 'crates/**/*.rs' \| wc -l` at `3bc2a172`), e.g. `crates/model/tests/mimo2_meta.rs:20` | tests, `tools/ref/models/*.sh` | **Not built:** one resolver (`crates/model/tests/common/model_path.rs` reads `$BLOOMERY_MODEL` already) and `BLOOMERY_DATA` (`tools/ref/ref-paths.sh:75`, overridable) |
| 10 | Fixtures at `/models/fixtures/<dir>` and the sets under `$BLOOMERY_DATA` | `tools/ref/ref-paths.sh`, `crates/refset/src/fixture.rs:31,33` | **Published:** the sets are a public dataset and `fixture fetch` makes the fixtures (section 4). The root-relative identity is not built (section 6) |

Without local mode, every `just` recipe that calls `tools/box.sh` runs on the box: ~~278~~ 291 lines of the `justfile`
outside comments name `tools/box.sh` at `3bc2a172` (`grep -vE '^\s*#' justfile | grep -c 'tools/box\.sh'`).

## 8. Open questions for the maintainers

1. `CONTRIBUTING.md` ("Numbers carry their conditions") makes a number from a contributor's own card evidence for that
   pull request on that card. Does the same hold for a gate result from a card whose arch is not `sm_86`, beyond the
   DGX Spark's Arm evidence ([`dgx-spark.md`](dgx-spark.md) section 11, answer 5)? Which fixture gates the card name
   closed in two such runs is in section 3. For S7's `devmap` clause the answer is yes: a result checked on the
   contributor's card is taken as it is ([`dgx-spark.md`](dgx-spark.md) section 11, answer 19).
2. **Answered:** the `fx_` sets are public, and a contributor regenerates fixtures from headers grown to each file's
   listed length (section 4).
3. **Answered:** local mode lives in `box.sh` as `BLOOMERY_BOX_LOCAL=1` ([`dgx-spark.md`](dgx-spark.md) section 11,
   answer 10), and it is built (section 3).
4. Local mode refuses a two-card recipe with exit 64, a usage error, while T2's design counts a named refusal with exit 66
   as `DEFER`. On a one-card machine a two-card gate is "cannot run here" rather than a caller's mistake, so a T2 runner
   would count these recipes red. Should the code be 66, 69 or another
   ([#38](https://github.com/midagedev/bloomery/pull/38), question 2)?
