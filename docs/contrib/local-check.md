# `just contrib-check`: one off-box verification command (design, not implemented)

A contributor has no way today to answer "is my change green?" on their own machine with one command. The gates are
`just gate-*` recipes, and 278 lines of the `justfile` run through `tools/box.sh`, which ssh's to the maintainers' one
workstation (`tools/box.sh:42,225`, `docs/BUILD.md:5`). This page designs the command, lists the box assumptions it must
bypass with their size, and says who does each change. Nothing here is built; every claim cites `path:line` and
`[unverified]` marks what I could not check. The result of `contrib-check` is the contributor's evidence in a pull
request, never landing evidence: the maintainers' box record still decides (`AGENTS.md`, "Never run a gate on the Mac").

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

`just contrib-check [--base BASE] [--require T2]` is a recipe that calls one script, `tools/contrib-check.sh`
(`just check-recipes` forbids a test recipe with `||` or a bare `cargo test`, so the logic lives in the script). It
detects its tier, runs it, and never skips silently:

| Tier | Where | Runs |
|---|---|---|
| T0 | any OS with the pinned nightly | `cargo fmt --check`; `cargo test -p` for each `pure` row of `recipes.py pure-crates`; `python3 tools/recipes.py --self-test`; the `tools/check-*.sh` scripts except `check-rustflags` (it holds the repo's `znver3` against itself, so it is the maintainers' after a CPU change, `docs/BUILD.md:51`). On macOS it calls `mac-static` and `mac-check.sh test` |
| T1 | x86_64 Linux, no card needed | T0 plus every unit of `recipes.py host-nightly`, each under `BLOOMERY_GATE_BOUND`, with `RUSTFLAGS` set to the machine's own CPU when it is not Zen 3 (`run.sh:55`) |
| T2 | T1 plus an NVIDIA card and fixtures | the fixture-tier family gates: `just affected BASE`, run serially under `BLOOMERY_TIER=fixture`; a recipe the tier refuses by name (rc 66 from `tools/ref/real-only.sh` or `ref-paths.sh`; `gate-batch.sh` counts such items as `deferred`) is printed `DEFER`, never counted green |

Output is one line per unit, `contrib-check <tier> <unit> PASS|FAIL|SKIP|DEFER rc=<n> <seconds>s`, then a summary of what
ran and what was skipped and why (`T2 skipped: no nvidia-smi`). It writes `target/contrib-check/<time>/` with each
unit's log and a `report.txt` that opens with the commit, `uname -a`, the CPU model, RAM, the toolchain, and the card,
driver and `--version` when there is one: the block a pull request pastes. Exit: 0 every unit that ran is green, 1 a
unit failed, 64 usage, 69 a prerequisite is missing or a tier named by `--require` could not run (the code
`tools/mac-check.sh:35-39` uses for a missing prerequisite; 75 stays contention, which a retry is for: the standing rule
is `AGENTS.md`'s rc 75, so a missing card must not use it). An opt-in
`CONTRIB_INJECT_FAIL=1` adds one deliberately failing unit, as `NIGHTLY_INJECT_FAIL` does (`run.sh:33`), so the red path
is shown to fail first. aarch64 Linux runs T0 only until the host compiles there (`docs/contrib/dgx-spark.md`).

## The box assumptions T2 must bypass

| # | Assumption | Where | Change | Size | Who |
|---|---|---|---|---|---|
| 1 | Commands run on host `ws` after an rsync of the tree and `source ~/bloomery-env.sh` | `tools/box.sh:42,108,225` | a new `BLOOMERY_BOX_LOCAL=1` (`BLOOMERY_BOX` already names the ssh host, `box.sh:42`): run the already composed command string (profile, tier, env exports, `:144-223`) with `bash -c` in the current tree, skip the rsync, source `$BLOOMERY_ENV` (default `~/bloomery-env.sh`). Every `just gate-*` then works unchanged. Its tests extend `tools/ref/card-tests/run.sh:500-521`, which drives `box.sh` through a fake ssh | S-M | maintainers |
| 2 | The environment file of the box (nightly, LLVM 21, CUDA 13.3) | `tools/box.sh:4` | ship `tools/local-env.example.sh` from the `docs/BUILD.md:34-40` block | XS | contributor |
| 3 | A timing lease and holds under `/root` gate every command | `tools/ref/lease-probe.sh:14,45` | already overridable (`BLOOMERY_LEASE_LOCK`, `BLOOMERY_LEASE_HOLDS`); local mode points both at a file in a user directory so the guard finds no lease | XS | maintainers |
| 4 | Gate locks per card name at `/root/bloomery-gate*.lock`, the V4.1 load lock, the batch hold | `tools/gpu-gate.sh:48-50` | a lock directory variable, and the lock keyed by card UUID so `any` takes the card that exists | M | maintainers |
| 5 | Cards are found by exact name, `NVIDIA GeForce RTX 3090` and `NVIDIA RTX A6000` | `tools/ref/cards.sh:33-34`; `box.sh` `a6000\|both` pick, `:116-134` | local mode accepts only the default pick; a one-card host is the card the census finds. Which fixture gates open on a card that is neither named one is `[unverified]`: `AGENTS.md` ("The 3090 pins today") lists the pinned ones, the first run answers the rest | M | maintainers |
| 6 | The device arch is `sm_86` in 136 `justfile` lines and `gate.sh:35` | `tools/gate.sh:35`, `justfile` | one `BLOOMERY_ARCH` default `sm_86`; the ptx-spill table (`tools/ref/ptx-shapes.tsv`) stays sm_86 only | M, mechanical | contributor, after the maintainers say which archs they accept |
| 7 | The cuda-oxide backend is a prebuilt `.so` per rev, and `cargo oxide` refuses (rc 70) without it | `tools/box.sh:217` | local mode leaves `CUDA_OXIDE_BACKEND` unset so `cargo-oxide` builds from the fork checkout (`docs/BUILD.md:32`) | XS | maintainers |
| 8 | The CPU flag `znver3` in both cargo files | `.cargo/config.toml:8`, `.cargo/cuda-oxide.toml:5` | host units: `RUSTFLAGS` as the nightly does. A `cargo oxide` build on another CPU needs both files edited (`docs/BUILD.md:51`); whether `RUSTFLAGS` reaches `cargo oxide` is `[unverified]` | S | maintainers |
| 9 | Real model files at `/models/...`: 133 path literals in `crates/**/*.rs` (`grep -rn '"/models/\|"/root/\|/home/user'`, comments included), e.g. `crates/model/tests/mimo2_meta.rs:20` | tests, `tools/ref/models/*.sh` | one resolver (`crates/model/tests/common/model_path.rs:4` reads `$BLOOMERY_MODEL` already) and `BLOOMERY_DATA` (`tools/ref/ref-paths.sh:67`, overridable) | M, mechanical | contributor |
| 10 | Fixtures at `/models/fixtures/<dir>` and the sets under `$BLOOMERY_DATA` | `tools/ref/ref-paths.sh:88`, `crates/refset/src/fixture.rs:23,33` | see below | M | maintainers |

## Fixtures: what a contributor would download

The fixture tier is what makes a family's gates runnable without the 100 GB to 350 GB real files
(`tools/ref/ref-paths.sh:33-46`). Measured on the box with `du -sh` on 2026-10-10 (read-only): `glm5next` 8.0 GB,
`qwen38` 11 GB, `v41` 16 GB, `v41-r8` (the V4.1 sidecar) 7.3 GB, and the 18 reference-set directories `fx_*` under
`/root/bloomery-data` 6.4 GB: about 49 GB together, not the 12 GB one family suggests. Three families have a fixture
(`crates/refset/src/fixture.rs:33-37`); MiMo and the older families do not.

- **Two ways to get them.** Download the published files, or regenerate: `fixture generate` writes random weights whose
  bytes are a function of the seed and the source header alone (`crates/model/src/fixture/mod.rs:29`) and hashes headers
  only (`plan.rs:224`), so the bulk of the 35 GB is reproducible from the real files' headers. It needs a real split
  file opened, so whether a headers-only partial download suffices is `[unverified]` (`tools/ref/gguf-ranges.py` reads
  headers by range). The `fx_` sets are ik's output and cannot be regenerated without an ik build: they would be
  published, 6.4 GB.
- **The path is part of the identity.** A set's `# model` line states the first shard's full path `{root}/{dir}/{file}`
  (`crates/refset/src/fixture.rs:96-107`), so sets dumped under `/models/fixtures` are stale under another root even
  though `BLOOMERY_FIXTURE_ROOT` moves the files (`:23`). The cheap fix is that the contributor makes `/models/fixtures`
  exist (a symlink; needs root once). Keying the identity on `bloomery.fixture.source_header_sha256` (`plan.rs:224`)
  instead touches every family's check: M, the maintainers' design, `[unverified]` beyond that.
- **Where.** Not decided. Candidates: a Hugging Face dataset under the maintainers' account (`tools/ref` already
  range-reads the hub, `tools/hf-arch-check.py`), or release assets. The maintainers decide (open question).

## Order and sizes

1. **C1, T0 and T1 only** (`tools/contrib-check.sh`, its self-test, the recipe): S. A contributor can write it from
   `run.sh` and `recipes.py host-nightly`, which already hold the logic. Value on day one: every pure-crate and
   host-tier change is checkable alone.
2. **Rows 1, 2, 3, 7** (the `BLOOMERY_BOX_LOCAL` mode of `box.sh` and the env example): S-M, maintainers. After it `just gate-<x>` runs on a
   host with no card for the gates that open none (`gate-mimo2-meta`'s header test given the file).
3. **Rows 4, 5, 6, 8, 9, 10**: the card and fixture half, M each; row 10's publication is a maintainers' decision first.
   C1 gains T2 when row 1 and 10 land: S.

`just check-recipes` must stay green: the recipe takes no lock itself, runs no bare `cargo test`, and is not a
`gate-*` recipe.

## Open questions for the maintainers

1. Row 5 and 6: do you accept a card that is neither the 3090 nor the A6000 and an arch other than `sm_86` for
   contributor evidence, and which fixture gates must stay on named cards?
2. Row 10: where do fixtures and the `fx_` sets live, and may a contributor regenerate fixtures from headers?
3. Is a local mode (`BLOOMERY_BOX_LOCAL`) acceptable in the maintainers' own `box.sh`, given the guard and lease rules around it, or
   should the local path be a separate script?
