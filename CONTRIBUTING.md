# Contributing

Thank you for looking. Issues and pull requests are welcome. A plain statement of where the project stands first:
every gate runs on the maintainers' one workstation, so a pull request is verified by a maintainer there. What you
can check on your own machine, and what you cannot, is below. A Korean version: [`CONTRIBUTING.ko.md`](CONTRIBUTING.ko.md).

## Build

[`docs/BUILD.md`](docs/BUILD.md) has the toolchain (a pinned nightly, a cuda-oxide fork pinned by rev, CUDA 13.3) and one
command block per model. Never build a device crate with plain `cargo`; use `cargo oxide` (`AGENTS.md`, "Never").
The `cargo-oxide` install on a clean host has not been run yet (`docs/BUILD.md:32`).

## What you can do without our hardware

- **Pure crates, any OS.** `python3 tools/recipes.py pure-crates` lists them (`bloomery-serve`, `-tokenizer`, `-jinja`,
  `-hf`, `-refset`, `-placement`, `-levers`, `-runtime`, `-sampler`, `-decision`, `-models`, `-vision`); `cargo test -p
  <crate>` runs one. `bloomery-serve` has a mock engine, so the HTTP API is testable with no model.
- **The host tier, x86_64 Linux, no card.** `python3 tools/recipes.py host-nightly` prints the units the nightly runs
  (`tools/nightly/run.sh`): a `cargo test --release` per crate that reaches no card, then every `tools/check-*.sh`.
  The build targets Zen 3 (`.cargo/config.toml:8`) and may stop with an illegal instruction on another CPU
  (`docs/BUILD.md:51`): set `RUSTFLAGS='-C target-cpu=native'` as the nightly does (`tools/nightly/run.sh:55`). A host with
  no card still needs the CUDA headers, libclang and numpy/PIL (`tools/nightly/install.sh:7-17`).
- **Python and shell tools** under `tools/` carry self-tests (`python3 tools/recipes.py --self-test`,
  `python3 tools/hf-arch-check.py --self-test`).
- **On a Mac,** `just mac-static` is the static tier (format, clippy as an x86_64-linux cross check, every check script);
  it needs the environment the header of `tools/mac-check.sh` describes (`docs/BUILD.md:299`).
- Starter rows that need none of our hardware: [`docs/contrib/good-first-issues.md`](docs/contrib/good-first-issues.md).
- One command for all of this is designed, not built: [`docs/contrib/local-check.md`](docs/contrib/local-check.md).

## How a pull request is verified

The `just gate-*` recipes reach the workstation through `tools/box.sh` (ssh and rsync to one host), so they do not run
on yours. After your change is green on what you can run, a maintainer runs the gates `just affected` selects on the
box and lands the change in a batch. A label on a pull request that starts that run through the box queue is planned,
not built. Say in the pull request which commands you ran and paste their output.

## Two tracks we hope for

- **A new model architecture:** [`docs/contrib/new-model.md`](docs/contrib/new-model.md) (the steps, in order, from the
  last three attaches, and which of them you can verify alone).
- **NVIDIA DGX Spark (aarch64, GB10):** [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md) (the port map).

Open an issue first for either (the "New model support" template for a model), so two people do not take the same one. MiMo-V2.6-Flash is in progress.

## Rules

**The gates are the contract.** Do not relax a band, a tolerance or a lint to make a gate pass. If a threshold must
move, move it with a dated `PIN(YYYY-MM-DD):` comment and the reason. A new gate should first be shown to fail on the
defect it guards.

**Numbers carry their conditions.** Write `tok/s @ n=96, depth 4096, RTX A6000`, not `29 tok/s`; a derived number says
[derived]. Our own numbers come from the lease runners in `tools/ref/`, which run only on the maintainers' box today
(they take a lease under `/root` and pick the timing card by name). Your speed number is your own: give its conditions
and how you measured it, and never mix it into our tables. The Spark map has the plan for timing on another machine
(`docs/contrib/dgx-spark.md`).

**AI-assisted pull requests are welcome.** Much of bloomery was written with AI help. Point your agent at `AGENTS.md`
first. Say in the pull request that AI helped, and with which parts; you answer for the change (you have read the diff
and can explain it); paste the real output of the commands you ran, never output you did not run.

**Pull requests for your own hardware are especially welcome** (other cards, CPUs, memory sizes, native Windows). Name
the machine (`--version`, card, driver, CPU, RAM) and what you ran on it, and let the new path follow from what the
machine reports rather than a flag only you would set.

## Code shape

Comments and anything that goes upstream are in English; a comment states what is true now, and history belongs in the
commit message. Code shape is judged by [`docs/rust-quality.md`](docs/rust-quality.md) (the rules are numbered
R1 to R29 and reviews cite the number). `AGENTS.md` is the full working contract, written for the maintainers and their
AI rounds: read its "Never" list before your first change.
