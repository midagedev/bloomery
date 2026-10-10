# Adding a model architecture

How a model is attached today, derived from the last three attaches: GLM-5.3-Flash (`glm5next`, 2026-09-27 to 10-02),
Qwen3.8-Flash-Next (`qwen4exp`, 2026-09-27 to 10-08) and MiMo-V2.6-Flash (`mimo2`, 2026-10-07 to 10-09). Read from the
tree at `6b630e5e`; every claim cites `path:line`, and `[unverified]` marks what I could not check. The ladder is
long. A reader and a placement you can finish alone; the card body, the serve seat and every oracle need the
maintainers' box today, and the last section says where they take over.

Open an issue before you start so two people do not take the same model. MiMo-V2.6-Flash is in progress and Kolibri-1
is next (`docs/plan-triage.md:23`, `:696-700`); Mistral Large 4 waits for open weights.

## Two shapes of attach

- **A variant of a family we have.** Qwen3.8's file declares `qwen4exp`; the reader treats it as a variant row of the
  `qwen35moe` reader (`crates/model/src/arch/mod.rs:32-46`, commit `3c5f6b09`), and its card body is `Body38` inside the
  existing `crates/gpu/src/arch/qwen3moe/` (`body38.rs`, 4,862 lines). Cheapest when the new architecture is a small
  change of an existing one.
- **A new family.** GLM and MiMo each have a reader module (`crates/model/src/arch/glm5next/`, `.../mimo2/`), a refset
  family, a card crate (`crates/gpu-glm5next`, `crates/gpu-mimo2`), an app session and workspace-level bookkeeping.
  MiMo's card crate has no device code of its own: it composes `bloomery-gpu` and `bloomery-gpu-deepseek41` kernels
  (`crates/gpu-mimo2/src/lib.rs:1-7`). Sizes today (`wc -l`, 2026-10-10): `gpu-mimo2/src` 1,377 lines against
  `gpu-glm5next/src` 11,469; the reader modules 2,363 (`mimo2`) and 3,823 (`glm5next`).

A new family still touches common files. MiMo's reader commit (`554f1d69`) changed 21 files, 15 of them outside the
`mimo2` module and its own meta test: `crates/models/src/{lib,need,shape}.rs`, `crates/model/src/arch/coverage.rs`,
`arch/mod.rs`, the `qwen35moe` reader, `crates/runtime`. Each such edit is a fact the new model forces into a common
owner. That is the cost we want to shrink (`docs/plan-triage.md:687`).

## The ladder

Order from the three histories: oracle and reader first (MiMo's refset `eaee1c21` and reader `554f1d69` both landed
10-07), then placement (`38b71576`), then the card body (`0c4bc496`, "runs end to end"), then lifts of what the body
copied into common owners (`69ce7cb7`), then the gate rows (`76adc288`); the serve seat came after the body (GLM:
`ae31fea8`, `f8c57c86`). Fixture tier last. MiMo has no fixture row and no serve seat yet.

**Alone?** (can you verify it without our box): **A** any machine with Rust; **L** x86_64 Linux with no card and no
model file (`bloomery-model` links x86_64 code, so not a Mac or aarch64: `python3 tools/recipes.py pure-crates` leaves it
out; the build targets Zen 3, so on another CPU set `RUSTFLAGS='-C target-cpu=native'` as `tools/nightly/run.sh:55` does,
and install what `tools/nightly/install.sh:7-17` lists: CUDA headers, libclang, numpy and PIL); **G** Linux with an NVIDIA card and the `docs/BUILD.md` toolchain; **B** the maintainers' box only (ik oracle
dumps, real-file gates, timed runs).

| # | Step | Owning files (new family) | Alone? |
|---|---|---|---|
| 1 | GGUF arch reading | `crates/model/src/arch/<arch>/{hparams,names,roles,spec,place}.rs`; arms in `arch/mod.rs:32`, `crates/models/src/lib.rs:52` (`Arch`), `coverage.rs:52` (`program_of`) | L; header meta test needs the real file's headers |
| 2 | Tokenizer, chat template | `crates/tokenizer/src/pretok.rs:324`, `ChatSpec` `crates/models/src/lib.rs:589`, `chat_of` `arch/mod.rs:256`, tool parsers `crates/serve/src/{glmxml,qwenxml,dsml,hermes}.rs` | A for the code; the id oracle is B |
| 3 | Refset family and its dump | `crates/refset/src/arch/<arch>/mod.rs` and the table `arch/mod.rs:17-28`; `tools/ref/models/<arch>.sh`, `tools/ref/build-ik-<arch>.sh`, `justfile:667` (`dump-ref-mimo2`) | A for the row's tests; the dump is B |
| 4 | Fixture row | `crates/model/src/arch/<arch>/fixture.rs`, `crates/model/src/bin/fixture.rs:44`, `crates/refset/src/fixture.rs:33`, `tools/ref/ref-paths.sh:73-79` | generate L; its `fx_` oracle sets B |
| 5 | Card body | `crates/gpu-<arch>/` (or `crates/gpu/src/arch/...` for a variant); root `Cargo.toml:3`, `crates/gpu-gates/Cargo.toml` | G to build and run |
| 6 | App session | `crates/app/src/arch/<arch>/mod.rs`, feature in `crates/app/Cargo.toml` | G (feature-gated); L for default-feature tests |
| 7 | Serve seat | `crates/gpu-gates/src/bin/shared/serve_seats/<seat>.rs`, `serve_seats/mod.rs`, `bloomery_serve.rs:206-226` | G, then B for the seat gates |
| 8 | Placement sizing | `crates/model/src/arch/<arch>/place.rs` over `crates/placement/src/placement.rs:1821` | A (placement is pure) and L |
| 9 | Levers | `crates/levers/src/registry.rs` | A |
| 10 | Gates and recipes | `justfile`, `tools/gate-paths.tsv`, `tools/gate-batch-resources.tsv`, `tools/ref/ptx-shapes.tsv`, `tools/unsafe-ratchet.txt` | A for `just check-recipes`; running the gates is G or B |

### 1. GGUF arch reading

The reader turns a file's header into a `models::ModelSpec` (layers, mixers, feed-forward blocks, residual scheme, chat
surface; `crates/models/src/lib.rs:1-12`) and a role for every tensor, and refuses by name what it does not understand.
`arch::spec` is the only place that reads `general.architecture` (`arch/mod.rs:1-7`). Then `coverage::check`
(`coverage.rs:959`) lists, all at once, every part of the file no program in the tree runs (`Program`, `AVAILABLE`:
`coverage.rs:76`). That list is your work list for step 5. Variant names are the GGUF strings themselves, so `grep
<arch>` finds the module, the keys, the profile and the gates.

- Verify, L: `bash tools/gate.sh --release -p bloomery-model --lib -- arch::<arch> --nocapture` (the text inside
  `gate-mimo2-meta`, `justfile:2007`; `tools/gate.sh` runs on any host) runs the refusals on synthetic headers.
- Header test, needs the real file: `crates/model/tests/<arch>_meta.rs`, `hw_`-prefixed, `#[ignore]`, headers only,
  seconds. It pins the description line by line (`mimo2_meta.rs:31`). The MiMo test hard-codes the box path
  (`mimo2_meta.rs:20`); `crates/model/tests/common/model_path.rs:4` reads `$BLOOMERY_MODEL` instead.

### 2. Tokenizer and template

`tokenizer.ggml.pre` must be one of the pre-tokenizers the tokenizer runs (`pretok.rs:324`: `deepseek-v3`, `qwen2`,
`glm4`, `qwen35` and two aliases); `coverage` asks `runs_pre_tokenizer` (`crates/tokenizer/src/lib.rs:53`) at every plan,
so an unknown one is a named refusal, not a wrong id. MiMo needed no new tokenizer code: its spec reads `qwen2`,
`ToolFormat::QwenXml` and the `<think>` span (`crates/model/src/arch/mimo2/spec.rs:25,31`). A new tool-call syntax is a
new `ToolFormat` variant (`models/lib.rs:602`) with its parser in `crates/serve/src/`, tested with the mock engine.
The oracle for ids is ik's `llama-tokenize` (`crates/tokenizer/tools/oracle.sh`, `tests/tokenizer.rs:243`): B.

### 3. Refset family and its reference dump

A reference set is ik_llama.cpp's node dump of the model file, the oracle every card gate compares against. One family
row per oracle (`crates/refset/src/arch/mimo2/mod.rs`, 67 lines: model path, ik build, set names, the consuming gate);
the file's first shard path is the set's identity. The profile `tools/ref/models/<arch>.sh` holds what is a property of
the model, not the machine. MiMo needed an ik tree newer than the shared one with one oracle commit cherry-picked, hence
`tools/ref/build-ik-mimo2.sh` and the dated PIN at `refset/src/arch/mimo2/mod.rs:17-21`. The row's unit tests are pure (A). Dumping needs the ik tree,
the real file in the page cache (167 GB for MiMo) and the box's data directory: B.

### 4. Fixture row

A fixture is a small file with the real file's header, every per-layer shape and type, and random weights, so the whole
weight set fits in host memory (`crates/model/src/fixture/mod.rs:1-30`); the bytes are a function of the seed and the
source header alone (`:29`), and only headers are hashed (`plan.rs:224`). Three families have one
(`refset/src/fixture.rs:33-37`); MiMo does not, so its end-to-end gate is a weekly that the fixture tier refuses by
name (`justfile:2016`). To add one: a `FixtureSpec` in `arch/<arch>/fixture.rs`, a line in `SPECS`
(`bin/fixture.rs:44`), `DIRS` (`refset/src/fixture.rs:33`) and the table in `ref-paths.sh:73-79` (the Rust and shell
twins must agree), then `just dump-ref-fixture` (`justfile:609`) for the `fx_` sets, which needs ik: B.
Whether `fixture generate` works from a partial download of the real file's headers is `[unverified]`.

### 5. Card body

Compose, do not write kernels first: MiMo's body is attention, a dense block and a routed block over the shared kernels,
and the step is a walk of `runtime::sched` over `runtime::layer` programs (`gpu-mimo2/src/lib.rs:1-20`). A new kernel
is a separate round with its own `ptx-scan` proof (the table in `AGENTS.md`, "Derive first"). Add the family to
`Program`/`program_of` and `AVAILABLE` rows (`coverage.rs:30-61,76`) so the coverage list shrinks to empty. Build with
`cargo oxide`, never plain `cargo` (`docs/BUILD.md:55`). `just mac-check` type-checks it without a card.

### 6. App session

Three small traits behind `Session<B>`: `Open` (plan and load), `Prompt` (feed ids), `Keep` (what a cut keeps)
(`crates/app/src/lib.rs:149,183,192`). MiMo's is 113 lines and feeds one id a step with no checkpoint
(`crates/app/src/arch/mimo2/mod.rs:1-8`); GLM's has checkpoints and MTP (`glm5next/mod.rs`, 336 lines). Reuse
`HostCfg` and `PlanLevers` (`mimo2/mod.rs:23-27`) rather than new options.

### 7. Serve seat

A seat is the model's module behind `bind::Seat` (`crates/gpu-gates/src/bind.rs:747`): its flags, records, slots and
prompt cache. Registering it touches `serve_seats/mod.rs`, the `Model` enum and `serves` in `bloomery_serve.rs:206-226`,
the mirror in `tools/hf-arch-check.py` (its self-test greps `Model::serves`, `Arch::from_name` and `spec`), and a row of
the README's Use table (`README.md:83-84`). MiMo has none yet; take the shape from the GLM seat (`serve_seats/glm.rs`).
Gate: `gate-gpu-glm5next-serve` (`justfile:2143`) is the pattern; it needs cards.

### 8. Placement sizing

`place.rs` states which bytes go where on a machine: the reader's `PlanInputs`, the KV bytes per layer, and the common
planner (`crates/placement/src/placement.rs`, `plan_host_routed` at `:1821` for a model whose routed experts all sit on
the host). MiMo's is 407 lines (`38b71576`). The machine description is `model::placement::workstation` (the cards and
RAM of the maintainers' box, `crates/model/src/placement/workstation.rs`); a plan names what holds the bytes, never a
copied shape from another card.

### 9. Levers

Runtime knobs are rows of `crates/levers/src/registry.rs` and nowhere else (`AGENTS.md`, Conventions). A new model
usually needs no row: it reads the existing ones (`BLOOMERY_CARD_BUDGET`, `BLOOMERY_HOST_LOCK`, ...). A model-specific
one (`BLOOMERY_QWEN38_EXPERTS`, `registry.rs:36`) is a row plus a line of `tools/levers-direct.txt` while it is read in
place. `cargo test -p bloomery-levers` is pure.

### 10. Gates and recipes

Per family: a `gate-<fam>-meta` recipe, an e2e (or a weekly while there is no fixture), a `dump-ref-<fam>` recipe.
Bookkeeping that turns a check red when missed (`just check-recipes`, A):

- `tools/gate-paths.tsv` rows (e.g. `:85`, `:156-160`); `tools/gate-batch-resources.tsv:36` for a solo recipe;
- `tools/ref/ptx-shapes.tsv:200-202` for each PTX entry; `tools/unsafe-ratchet.txt:174` for a new crate;
- the hard-coded device-crate list in `tools/recipes.py:8499`; `[[bin]] required-features` in
  `crates/gpu-gates/Cargo.toml`; a new workspace member in the root `Cargo.toml:3`.

## Common code first

Copied from `AGENTS.md`, Conventions:

> **Common code first, so the next model attaches cheaply.** A concept two models share (tier legs, prompt walks, seats,
> placement sizing, drafts, residency glue) has one common owner. A model's own code holds only what an architecture fact forces (tensor shapes, the recurrent state, the
> attention kind, the quant), and its spec names that fact. A per-model copy of shared logic stays only for a large
> measured speed gain, named in the code beside it. Lift what two models already do the same way, not what a future
> model might need. A common owner that grows a branch, flag or trait method per model is the wrong cut; the
> model-specific part goes back to the model.

MiMo shows the rule at work: the body copied helpers, and the next commit gave each one an owner (`69ce7cb7`).

## Where you stop today

Steps 1, 2 (code), 8, 9 and the bookkeeping of 10 you can do and verify on an x86_64 Linux machine. Steps 3 and 4
(reference sets) need ik and, for the real tier, the model file in the places the tools expect (`/models/...`, 133
path literals across `crates/**/*.rs`, `grep -rn '"/models/\|"/root/\|/home/user'`; many are comments). Steps 5 to 7
and every `just gate-*` run need an NVIDIA card plus a way past `tools/box.sh`, which does not exist yet:
[`local-check.md`](local-check.md) designs it. Until then a maintainer takes over at step 3, runs the dumps, and runs
the card gates on the box. Say in your pull request which steps you did and how you verified each.
