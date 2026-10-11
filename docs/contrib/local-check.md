# `just contrib-check`: one off-box verification command (T0 and T1 implemented, T2 designed)

A contributor needs one command that answers "is my change green?" on their own machine. The gates are `just gate-*`
recipes, and 278 lines of the `justfile` run through `tools/box.sh`, which ssh's to the maintainers' one workstation
(`tools/box.sh:42,225`, `docs/BUILD.md:5`). `just contrib-check` runs the tiers that need no workstation: T0 and T1 are
built (`tools/contrib-check.sh`, with the unit runner it shares with the nightly, `tools/nightly/units.sh`), and
`.github/workflows/pr.yml` runs T0 on every pull request. T2, the card and fixture tier, is designed below and not built;
this page lists the box assumptions it must bypass with their size, and who does each change. Every claim cites
`path:line`, and `[unverified]` marks what I could not check. The result of `contrib-check` is the contributor's evidence in
a pull request, never landing evidence: the maintainers' box record still decides (`AGENTS.md`, "Never run a gate on the
Mac").

## What already runs off the box

- **Pure crates:** `python3 tools/recipes.py pure-crates` derives the list (today `bloomery-decision`, `-hf`, `-jinja`,
  `-levers`, `-models`, `-placement`, `-refset`, `-runtime`, `-sampler`, `-serve`, `-tokenizer`, `-vision`);
  `just mac-test` runs `cargo test -p` on each natively on a Mac (`tools/mac-check.sh:1-15`).
- **The Mac static tier:** `just mac-static [BASE]` runs fmt-check, clippy as an x86_64-linux cross check, the scoped
  build shapes (`recipes.py combos`) and every `tools/check-*.sh` in two lanes (`tools/mac-static.sh:1-15`).
- **The host tier, x86_64 Linux, no card, no model file:** `python3 tools/recipes.py host-nightly` prints one unit a
  line, `name<TAB>command` (`bloomery-app`, `-gguf`, `-model`, `-qdot`, `-threads`, ... each `bash tools/gate.sh --release
  --locked -p <crate> --no-fail-fast`, then every `tools/check-*.sh` and `recipes-self-test`); `--excluded` names the six
  device crates it leaves out. `tools/nightly/run.sh` runs that list nightly on a VPS that is not the box and its header
  states that machine's differences: `RUSTFLAGS='-C target-cpu=x86-64-v3'` overrides the repo's `znver3`
  (`run.sh:55`), libclang 18 for bindgen, the CUDA 13.3 headers only, no model file (`run.sh:36-39`;
  `tools/nightly/install.sh:7-17` lists the apt and CUDA header packages).
- **A recipe's inner command:** `python3 tools/recipes.py box-command RECIPE` prints the text inside a recipe's
  `./tools/box.sh '…'` (`tools/recipes.py:15,1907`; `gate-mimo2-meta` prints `bash tools/gate.sh --release -p
  bloomery-model --lib -- arch::mimo2 …`). It refuses a recipe whose box call carries an env prefix such as
  `BLOOMERY_MODEL=qwen4exp` (verified by running it on `gate-gpu-qwen4exp-e2e`), because the prefix is the model
  profile that only `box.sh` applies. `tools/gate.sh` itself is a `cargo test` with a bound and runs on any host
  with GNU `timeout`.
- **The affected list:** `just affected BASE` prints the gate recipes a change selects, on any machine with `python3`
  and `just`, building nothing (`justfile:87`).

## The command

`just contrib-check [--tier t0|t1]` is a recipe that calls one script, `tools/contrib-check.sh` (`just check-recipes`
forbids a test recipe with `||` or a bare `cargo test`, so the logic lives in the script). The default tier is T1 on
x86_64 Linux and T0 elsewhere; `--tier t2` prints `T2 is not built yet` and exits 64, and `--tier t1` on another host is a
named 64. The script reads every list from its owner and copies none: the pure crates from `recipes.py pure-crates`, the
Python self-tests from the `selftests=(` array of `tools/check-recipes.sh`, the host units from `recipes.py host-nightly`.

| Tier | Where | Runs |
|---|---|---|
| T0 | any OS with the pinned nightly (rustup, rustfmt) | `cargo fmt --all -- --check`; `cargo test -p <crate>` for each crate of `recipes.py pure-crates`; `python3 tools/recipes.py --self-test`; each Python tool self-test `check-recipes.sh` lists; every `tools/check-*.sh`. None of the check scripts is left out: each reads the tree and runs no cargo and no box command, `host-nightly` runs the same set, and all ran green on a Mac. The earlier draft of this page excluded `check-rustflags` as the maintainers' after a CPU change; it holds `.cargo/config.toml` against `.cargo/cuda-oxide.toml` and never reads this machine's CPU, so it runs |
| T1 | x86_64 Linux, no card needed | T0, then every unit of `recipes.py host-nightly` that T0 did not already run under the same name (the check scripts and `recipes-self-test` run once), each `bash tools/gate.sh --release --locked -p <crate>` under a bound (`CONTRIB_UNIT_BOUND`, default 1200 s). Needs what `tools/nightly/install.sh` lists: the CUDA 13.3 headers, libclang for bindgen, python3-numpy and python3-pil |
| T2 | T1 plus an NVIDIA card and fixtures | **not built.** The fixture-tier family gates: `just affected BASE`, run serially under `BLOOMERY_TIER=fixture`; a recipe the tier refuses by name (rc 66 from `tools/ref/real-only.sh` or `ref-paths.sh`; `gate-batch.sh` counts such items as `deferred`) is printed `DEFER`, never counted green |

`RUSTFLAGS`: on x86_64 Linux the script sets `-C target-cpu=native` for every cargo unit unless `RUSTFLAGS` is already set
(a binary built for the repo's `znver3` may stop with an illegal instruction on another CPU, `docs/BUILD.md:51`). The nightly
sets `x86-64-v3` for its own VPS (`tools/nightly/run.sh:55`). The two callers share one unit runner, `tools/nightly/units.sh`
(`run_unit`, `unit_failure`): how a unit's exit code, counts and failing tests are read is one piece of code.

Output (built): a header (commit, host, toolchain, `RUSTFLAGS`, bound), one line per unit, `<name> rc=<n> <seconds>s`, under a
red one its failing tests or first red lines and its log path, then `DONE tier=… total=… red=… wall=…s`. Each unit's log is
in `target/contrib-check/<time>/<name>.log`. A host without GNU `timeout` (macOS) runs units unbounded and the header says
so. Exit: 0 every unit green, 1 a unit red, 64 usage, 69 a prerequisite is missing (named, with how it is made), 70 a list
the script reads is empty (75 stays contention, which a retry is for: `AGENTS.md`'s rc 75, so a missing card must not use
it). `CONTRIB_INJECT_FAIL=1` adds one deliberately failing unit, as `NIGHTLY_INJECT_FAIL` does (`run.sh:33`), so the red path
is shown to fail first.

Not built, still designed: the `--base` and `--require T2` flags, a `report.txt` per run (the header and the log directory
carry its facts), the `SKIP`/`DEFER` unit states (no unit is skipped in T0 or T1), a call of `mac-static` on a Mac (it needs
the maintainers' Mac environment file; `just mac-static` stays the Mac's static tier beside this command), and the T2 tier.
aarch64 Linux runs T0 only until the host compiles there (`docs/contrib/dgx-spark.md`).

## The box assumptions T2 must bypass

| # | Assumption | Where | Change | Size | Who |
|---|---|---|---|---|---|
| 1 | Commands run on host `ws` after an rsync of the tree and `source ~/bloomery-env.sh` | `tools/box.sh:42,108,225` | a new `BLOOMERY_BOX_LOCAL=1` (`BLOOMERY_BOX` already names the ssh host, `box.sh:42`): run the already composed command string (profile, tier, env exports, `:144-223`) with `bash -c` in the current tree, skip the rsync, source `$BLOOMERY_ENV` (default `~/bloomery-env.sh`). Every `just gate-*` then works unchanged. Its tests extend `tools/ref/card-tests/run.sh:500-521`, which drives `box.sh` through a fake ssh | S-M | maintainers |
| 2 | The environment file of the box (nightly, LLVM 21, CUDA 13.3) | `tools/box.sh:4` | ship `tools/local-env.example.sh` from the `docs/BUILD.md:34-40` block | XS | contributor |
| 3 | A timing lease and holds under `/root` gate every command | `tools/ref/lease-probe.sh:14,45` | already overridable (`BLOOMERY_LEASE_LOCK`, `BLOOMERY_LEASE_HOLDS`); local mode points both at a file in a user directory so the guard finds no lease | XS | maintainers |
| 4 | Gate locks per card name at `/root/bloomery-gate*.lock`, the V4.1 load lock, the batch hold | `tools/gpu-gate.sh:48-50` | a lock directory variable, and the lock keyed by card UUID so `any` takes the card that exists | M | maintainers |
| 5 | Cards are found by exact name, `NVIDIA GeForce RTX 3090` and `NVIDIA RTX A6000` | `tools/ref/cards.sh:33-34`; `box.sh` `a6000\|both` pick, `:116-134` | local mode accepts only the default pick; a one-card host is the card the census finds. Which fixture gates open on a card that is neither named one is `[unverified]`: `AGENTS.md` ("The 3090 pins today") lists the pinned ones, the first run answers the rest | M | maintainers |
| 6 | The device arch was `sm_86` in 139 `justfile` lines and `tools/gate.sh` | the justfile's `arch` variable, `tools/arch.sh` | **Done:** `BLOOMERY_ARCH`, default `sm_86`, read by the justfile and by the tools on the box, forwarded by `box.sh`; `check-recipes` fails on a literal. The ptx-spill table (`tools/ref/ptx-shapes.tsv`) stays sm_86 only: `gate-ptx-spill`, `ptx-scan`, `just narrow` and `release-build` refuse another arch by name | done | maintainers |
| 7 | The cuda-oxide backend is a prebuilt `.so` per rev, and `cargo oxide` refuses (rc 70) without it | `tools/box.sh:217` | local mode leaves `CUDA_OXIDE_BACKEND` unset so `cargo-oxide` builds from the fork checkout (`docs/BUILD.md:32`) | XS | maintainers |
| 8 | The CPU flag `znver3` in both cargo files | `.cargo/config.toml:8`, `.cargo/cuda-oxide.toml:5` | host units: `RUSTFLAGS` as the nightly does. A `cargo oxide` build on another CPU needs both files edited (`docs/BUILD.md:51`); whether `RUSTFLAGS` reaches `cargo oxide` is `[unverified]` | S | maintainers |
| 9 | Real model files at `/models/...`: 133 path literals in `crates/**/*.rs` (`grep -rn '"/models/\|"/root/\|/home/user'`, comments included), e.g. `crates/model/tests/mimo2_meta.rs:20` | tests, `tools/ref/models/*.sh` | one resolver (`crates/model/tests/common/model_path.rs:4` reads `$BLOOMERY_MODEL` already) and `BLOOMERY_DATA` (`tools/ref/ref-paths.sh:67`, overridable) | M, mechanical | contributor |
| 10 | Fixtures at `/models/fixtures/<dir>` and the sets under `$BLOOMERY_DATA` | `tools/ref/ref-paths.sh:88`, `crates/refset/src/fixture.rs:23,33` | see below | M | maintainers |

## Fixtures: what a contributor would download

The fixture tier is what makes a family's gates runnable without the 100 GB to 350 GB real files
(`tools/ref/ref-paths.sh:33-46`). Three families have a fixture (`crates/refset/src/fixture.rs:33-37`); MiMo and the
older families do not. Measured on the box with `du -sh` on 2026-10-10 (read-only): `glm5next` 8.0 GB, `qwen38`
11 GB, `v41` 16 GB, `v41-r8` (the V4.1 sidecar) 7.3 GB. A contributor does not download those files: `fixture
generate` writes random weights whose bytes are a function of the seed and the source header alone
(`crates/model/src/fixture/mod.rs:29`) and hashes headers only (`plan.rs:224`), so a family's fixture is made again
from the source repo's headers. Measured by the `fxfetch` round: `fixture fetch` made the qwen38 fixture (with its MTP
draft), the v41 fixture (with the r8 sidecar) and the glm5next fixture (at its pinned revision) byte-identical to the
files on the box. What a contributor cannot make is the `fx_` reference sets: they are ik's output, and
they are published as one dataset. The download is those sets: 6.48 GB in 17 sets and 29,941 files (the dry run of 2026-10-10, `just refset-publish-dry`;
`du -sh` of the `fx_*` directories said 6.4 GB). Published after 0.2.10.

1. **Make the fixture of each family** (the verb is added by `fxfetch`):

       fixture fetch <family> /models/fixtures/<dir>

   The family's source (repo, quant, revision) is its row of the fixture table in `crates/model/src/bin/fixture.rs`
   (`SPECS`, or the path `fxfetch` reports); it is not copied here. `<family>` and `<dir>` are the pairs of
   `crates/refset/src/fixture.rs` DIRS (the dataset README lists them). It reads a few MB per shard by HTTP Range.
   The V4.1 DSpark draft fixture cannot be fetched (its source is the maintainers' own copy), and no set of it is
   published.
2. **Download the reference sets** into `$BLOOMERY_DATA`. The dataset mirrors that directory, so this one command is the
   install (`--include` takes one family's sets; the dataset README lists them by name):

       hf download --repo-type dataset midagedev/bloomery-refsets --local-dir "$BLOOMERY_DATA"

   `just refset-publish-dry` is the maintainers' dry run of the dataset (`tools/refset-publish.sh`): it lists every
   file with its sha256 and says which sets are left out (the fixture variants and the DSpark draft sets) and why.
   The upload is a separate step after the 0.2.10 tag.

- **The path is part of the identity.** A set's `# model` line states the first shard's full path `{root}/{dir}/{file}`
  (`crates/refset/src/fixture.rs:96-107`), so sets dumped under `/models/fixtures` are stale under another root even
  though `BLOOMERY_FIXTURE_ROOT` moves the files (`:23`). The contributor makes `/models/fixtures` exist (a symlink;
  needs root once). Keying the identity on `bloomery.fixture.source_header_sha256` (`plan.rs:224`) instead touches
  every family's check: M, the maintainers' design, `[unverified]` beyond that.

## Order and sizes

1. **C1, T0 and T1: built** (`tools/contrib-check.sh`, the shared runner `tools/nightly/units.sh`, the recipe, and the T0
   pull-request workflow `.github/workflows/pr.yml`). Every pure-crate and host-tier change is checkable alone. T1 is not in
   the workflow: it needs three NVIDIA `.deb` files unpacked by hand, more than apt packages (`pr.yml` header).
2. **Rows 1, 2, 3, 7** (the `BLOOMERY_BOX_LOCAL` mode of `box.sh` and the env example): S-M, maintainers. After it `just gate-<x>` runs on a
   host with no card for the gates that open none (`gate-mimo2-meta`'s header test given the file).
3. **Rows 4, 5, 6, 8, 9, 10**: the card and fixture half, M each; row 10's publication is a maintainers' decision first.
   C1 gains T2 when row 1 and 10 land: S.

`just check-recipes` must stay green: the recipe takes no lock itself, runs no bare `cargo test`, and is not a
`gate-*` recipe.

## Open questions for the maintainers

1. Row 5, and row 6's open half (the arch is `BLOOMERY_ARCH` now): do you accept a card that is neither the 3090 nor the A6000 and an arch other than `sm_86` for
   contributor evidence, and which fixture gates must stay on named cards?
2. Row 10: where do fixtures and the `fx_` sets live, and may a contributor regenerate fixtures from headers?
   **Answered 2026-10-11, the first half:** the `fx_` sets are public at
   <https://huggingface.co/datasets/midagedev/bloomery-refsets> (17 sets, 6.48 GB, with each base model's licence;
   `dgx-spark.md` section 5, answer 3). Whether `fixture generate` needs only a source header is still open
   (`dgx-spark.md` section 6).
3. Is a local mode (`BLOOMERY_BOX_LOCAL`) acceptable in the maintainers' own `box.sh`, given the guard and lease rules around it, or
   should the local path be a separate script?
   **Answered 2026-10-11:** in `box.sh`, as `BLOOMERY_BOX_LOCAL=1` (`dgx-spark.md` section 5, answer 10).
