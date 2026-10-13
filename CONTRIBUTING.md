# Contributing

Thank you for looking. Issues and pull requests are welcome. A plain statement of where the project stands first:
the gates need an NVIDIA card and the model files, and the full set runs on the maintainers' workstation. A run on your
own card counts. What you can check on your own machine, and what you cannot, is below. A Korean version:
[`CONTRIBUTING.ko.md`](CONTRIBUTING.ko.md).

## Build

[`docs/BUILD.md`](docs/BUILD.md) has the toolchain (a pinned nightly, a cuda-oxide fork pinned by rev, CUDA 13.3) and one
command block per model. Never build a device crate with plain `cargo`; use `cargo oxide` (`AGENTS.md`, "Never").
The `cargo-oxide` install on a clean host has not been run yet (`docs/BUILD.md:32`).

## What you can do without our hardware

- **Pure crates, any OS.** `python3 tools/recipes.py pure-crates` lists them (`bloomery-serve`, `-tokenizer`, `-jinja`,
  `-hf`, `-refset`, `-placement`, `-levers`, `-runtime`, `-sampler`, `-decision`, `-models`, `-vision`); `cargo test -p
  <crate>` runs one. `bloomery-serve` has a mock engine, so the HTTP API is testable with no model.
- **The host tier, x86_64 or aarch64 Linux, no card.** `python3 tools/recipes.py host-nightly` prints the units the nightly runs
  (`tools/nightly/run.sh`): a `cargo test --release` per crate that reaches no card, then every `tools/check-*.sh`.
  The build targets Zen 3 (`.cargo/config.toml:8`) and may stop with an illegal instruction on another CPU
  (`docs/BUILD.md:51`): set `RUSTFLAGS='-C target-cpu=native'` (`just contrib-check` does this for you; the nightly's own Zen 1 host sets `x86-64-v3`, `tools/nightly/run.sh:55`). A host with
  no card still needs the CUDA headers, libclang and numpy/PIL (`tools/nightly/install.sh:7-17`).
- **Python and shell tools** under `tools/` carry self-tests (`python3 tools/recipes.py --self-test`,
  `python3 tools/hf-arch-check.py --self-test`).
- **On a Mac,** `just mac-static` is the static tier (format, clippy as an x86_64-linux cross check, every check script);
  it needs the environment the header of `tools/mac-check.sh` describes (`docs/BUILD.md:299`).
- Starter rows that need none of our hardware: [`docs/contrib/good-first-issues.md`](docs/contrib/good-first-issues.md).
- **One command for all of this:** `just contrib-check` (`tools/contrib-check.sh`). `--tier t0` runs on any OS with the
  pinned nightly: `cargo fmt --check`, `cargo test -p` for each pure crate, the Python tool self-tests and every
  `tools/check-*.sh`. `--tier t1` (the default on x86_64 Linux) adds the host-nightly units. It prints one line per unit
  and a `DONE` line; paste them in your pull request. The card tier (T2) is designed, not built:
  [`docs/contrib/local-check.md`](docs/contrib/local-check.md). CI runs `just contrib-check --tier t0` on every pull
  request (`.github/workflows/pr.yml`); that is a smoke test for your change, not landing evidence.

## How a pull request is verified and merged

- **CI.** `just contrib-check --tier t0` runs on every pull request (`.github/workflows/pr.yml`); `main` requires it
  green.
- **Review.** Every ready pull request gets one advisory review from Claude (`.github/workflows/review.yml`; how it
  works: [`.github/review/README.md`](.github/review/README.md)). It comments and never blocks a merge. A draft, the
  `no-review` label or five automatic reviews stop it. A proven defect it finds outside your change becomes a
  `review-bot` issue.
- **Gates.** Run the `just gate-*` recipes `just affected` selects on your own card if you can
  (`BLOOMERY_BOX_LOCAL=1 just gate-<name>` on a Linux machine with a card; the header of `tools/box.sh` has the
  setup), and paste the commands and their output: we trust a run on your card. A maintainer runs what you could not on the workstation.
- **Merge.** `main` takes changes only through a pull request, by rebase (linear history). A contributor with write
  access merges their own pull request once CI is green and the evidence is in it.

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
[derived]. The lease runners in `tools/ref/` produce the maintainers' numbers. A number from your own NVIDIA card counts
when it carries: the `nvidia-smi` line (card and driver), both commit hashes, the model file, and the commands with
their output. For a small difference, running each side twice in turn helps. Such a number decides your PR on that
card, and it can be published (the README, the docs, a release note) in a table of its own for that card: a table never
mixes cards. Where two cards measure a change differently, both stand, and the change is gated by the card fact that
separates them. The Spark map has the plan for timing on another machine (`docs/contrib/dgx-spark.md`).

**Use AI freely, for the code and for talking with us.** Much of bloomery was written with AI help, and the
maintainers write every reply, review and issue with AI help too. Assume everything here is AI-assisted, on both sides;
you do not need to say so. Let an AI draft your issues, comments and pull requests in whatever language is easiest for
you, Korean included. Point your agent at `AGENTS.md` first. What we look at is the result: a simple design, code
quality, and that it works.

**Pull requests for your own hardware are especially welcome** (other cards, CPUs, memory sizes, native Windows). Name
the machine (`--version`, card, driver, CPU, RAM) and what you ran on it, and let the new path follow from what the
machine reports rather than a flag only you would set.

**A lab for hardware we do not have** (today: AMD GPUs, `labs/hip-native/`) lives under `labs/`, outside the workspace,
so the CUDA build, the gates and the release do not change. Your card is its gate. The lab carries a `verify.sh` that
runs its checks and writes a result JSON: the source tree hash, the binary hashes, the card, the runtime version, and
each case's result. Every PR to the lab attaches that JSON from its head commit. We check that the hash matches the
PR, and run the format, lint and comment checks on our side. The lab's owner is the contributor who opened it.

## Code shape

Comments and anything that goes upstream are in English; a comment states what is true now, and history belongs in the
commit message. Code shape is judged by [`docs/rust-quality.md`](docs/rust-quality.md) (the rules are numbered
R1 to R29 and reviews cite the number). `AGENTS.md` is the working contract for everyone, people and agents: read its
"Never" list before your first change.
