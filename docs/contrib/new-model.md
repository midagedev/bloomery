# Adding a model architecture

How a model is attached today, derived from the last three attaches: GLM-5.3-Flash (`glm5next`, 2026-09-27 to 10-02),
Qwen3.8-Flash-Next (`qwen4exp`, 2026-09-27 to 10-08) and MiMo-V2.6-Flash (`mimo2`, 2026-10-07 to 10-09). Read from the
tree at `6b630e5e`; every claim cites `path:line`, and `[unverified]` marks what I could not check. The ladder is
long. A reader and a placement you can finish alone; the card body, the serve seat and every oracle need the
maintainers' box today, and the last section says where they take over.

Open an issue before you start so two people do not take the same model. MiMo-V2.6-Flash is in progress and Kolibri-1
is next (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan-triage.md#L23, https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan-triage.md#L696-L700); Mistral Large 4 waits for open weights.

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
owner. That is the cost we want to shrink (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan-triage.md#L687).

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
| 1 | GGUF arch reading | `crates/model/src/arch/<arch>/{hparams,names,roles,spec,place}.rs`; arms in `arch/mod.rs` (`spec`), `crates/models/src/lib.rs` (`Arch`), `crates/placement/src/accepts.rs` (`program_of`, re-exported from `coverage`) | L; header meta test needs the real file's headers |
| 2 | Tokenizer, chat template | `crates/tokenizer/src/pretok.rs:350`, `ChatSpec` `crates/models/src/lib.rs:589`, `chat_of` `arch/mod.rs:256`, tool parsers `crates/serve/src/{glmxml,qwenxml,dsml,hermes}.rs` | A for the code; the id oracle is B |
| 3 | Refset family and its dump | `crates/refset/src/arch/<arch>/mod.rs` and the table `arch/mod.rs:17-28`; `tools/ref/models/<arch>.sh`, `tools/ref/build-ik-<arch>.sh`, `justfile:667` (`dump-ref-mimo2`) | A for the row's tests; the dump is B |
| 4 | Fixture row | `crates/model/src/arch/<arch>/fixture.rs`, `FIXTURE_SPECS` and each row's `FixtureRow` in `crates/model/src/arch/registry.rs`, `crates/refset/src/fixture.rs:33`, `tools/ref/ref-paths.sh:73-79` | generate L; its `fx_` oracle sets B |
| 5 | Card body | `crates/gpu-<arch>/` (or `crates/gpu/src/arch/...` for a variant); root `Cargo.toml:3`, `crates/gpu-gates/Cargo.toml` | G to build and run |
| 6 | App session | `crates/app/src/arch/<arch>/mod.rs`, feature in `crates/app/Cargo.toml` | G (feature-gated); L for default-feature tests |
| 7 | Serve seat | `crates/gpu-gates/src/bin/shared/serve_seats/<seat>.rs`, `serve_seats/mod.rs`, `bloomery_serve.rs:206-226` | G, then B for the seat gates |
| 8 | Placement sizing | `crates/model/src/arch/<arch>/place.rs` over `plan_host_routed` in `crates/placement/src/placement.rs` | A (placement is pure) and L |
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

A next family calls the common readers instead of copying them. A count above zero is `positive` and an optional
count is `optional_usize`, both beside `meta_usize` in `arch/mod.rs`. The block spelling is `names::blk`, and the
three model-level tensors are `names::token_embd`, `names::output_norm` and `names::output`. Already common, and
still the ones to call: the role walks in `arch/classify.rs` (`classify_groups` when every layer carries the same
stem groups, `classify_layers` when a layer's kind picks them, `role_with_nextn` and `block_role_with_nextn` when
next-token layers close the file, and `Counts::new`; `classify_with` under them takes any role function), `Stem` with
`req` and `opt` and the rows several families spell alike (`DENSE`, `MOE`, `HC` in `arch/stems.rs`), the `meta_*` key
readers, `chat_of`, `n_vocab`, `spec_u32` and `nextn_layers`. A family's `roles.rs` then holds only its own rows, and
its header reader's tensor check reads the same table (`layers_agree` over the kind's groups). Shared or per-layer
counts use `per_layer_count` (`arch/mod.rs`): a scalar expands to every layer; an array must have one count per layer,
zero included. A family that refuses zero checks it after the read. Gemma and MiMo use `check_kv` to reject zero KV-head counts or counts that do not divide the query-head
count. A float array with a scalar fallback (GLM's clamps) stays the family's own read.
`program_of` is defined in `crates/placement/src/accepts.rs` and re-exported from
`coverage`. An architecture with a reader and no program returns `None` from it; the coverage check then names
`Need::Program`.

### 2. Tokenizer and template

`tokenizer.ggml.model` may be `gpt2` (GPT-2 byte-level BPE), `gemma4` (BPE over metaspace text), or `llama`
(SentencePiece). The shared loader is `crates/tokenizer/src/vocab.rs`. For `gpt2`, `tokenizer.ggml.pre` must be one
of the names in `Pre::NAMES` (`pretok.rs`), including `flint`; aliases live in that same table.
A `gemma4` model always runs `gemma4`; SentencePiece uses no regex pre-tokenizer and reports `default`.
Bloomery's optional string key `tokenizer.ggml.normalizer = "NFC"` normalizes BPE fragments after raw special-token matching, with the Unicode 9.0 tables Hugging Face's tokenizers normalizes with; omission preserves unnormalized encoding, and other values are refused.
Model `coverage` asks `runs_pre_tokenizer`: BPE specs use the pre-tokenizer's name, and a SentencePiece spec
uses `llama`, its tokenizer model. `default` alone does not identify a supported BPE pre-tokenizer.
`ignore_merges` is a property of the pre (`Pre::ignore_merges`): `Pre::Llama3` (`lfm2`) and `Pre::Tekken` (`tekken`,
`pixtral`) take a word whose byte-encoded text is itself a vocabulary token. A tree older than mainline's change that
maps `pixtral` to TEKKEN (ik is one) reads that name as LLAMA3, so a `pixtral` vocabulary's oracle is a later mainline
tree (Nemotron's: the `nemotron` profile of `crates/tokenizer/tools/oracle.sh`). SentencePiece instead merges the
highest `tokenizer.ggml.scores` first, with the leftmost pair winning ties. Its `add_space_prefix` flag defaults to true and follows the header when present.
Both metaspace paths share whitespace spelling, byte fallback validation, special-token partition and decoding.
MiMo needed no new tokenizer code: its spec reads `qwen2`,
`ToolFormat::QwenXml` and the `<think>` span (`crates/model/src/arch/mimo2/spec.rs:25,31`). A new tool-call syntax is a
new `ToolFormat` variant (`models/lib.rs:602`) with its parser in `crates/serve/src/`, tested with the mock engine.
The oracle for ids is ik's `llama-tokenize` (`crates/tokenizer/tools/oracle.sh`, `tests/tokenizer.rs:243`): B. A
vocabulary ik cannot load reads through mainline's `llama-tokenize`; its profile in `oracle.sh` sets `REF_FLAGS`
(`--no-escape`: mainline turns a `\n` in a `-f` text into a newline by default; `--no-bos` when the file adds a BOS).

Flint uses the pinned mainline's effective ordered number/CJK/word split, strict UTF-8 decoding without
NFC, and no automatic BOS/EOS. Mainline collapses letter subcategories where the official HF regex
distinguishes them; compare both token IDs and pretokenizer boundaries instead of assuming regex text
means identical execution. With the pinned Flint vocabulary, the exercised Unicode/number/tag corpus
had identical IDs against both references; that is not a claim for arbitrary vocabularies.
`tokenizer.ggml.eog_token_ids` is the authoritative GGUF stop array: nonempty integer IDs, bounds-checked,
sorted and deduplicated, without disabling GGUF attribute overrides or changing BPE mode. The private
`tokenizer.bloomery.eog_token_ids` still selects HF mode; if both occur their canonical sets must agree.
The actual Flint header supplies four stops, including developer/150018; pinned mainline's loaded
heuristic set has only EOS/user/observation. Preserve the explicit four rather than match that heuristic.

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
name (`justfile:2016`). To add one: a `FixtureSpec` in `arch/<arch>/fixture.rs`, a line in `FIXTURE_SPECS`
and the row's fixture source (both `arch/registry.rs`), and a row of `tools/fixture-dirs.tsv` (the one table of the
fixture directories: refset, `ref-paths.sh` and the tools read that file, and the registry's test holds each row's `family` to it), then `just dump-ref-fixture` (`justfile:609`) for the `fx_` sets, which needs ik: B.
`fixture fetch <family> <out>` (`bin/fixture.rs`, the fixture rows of `arch/registry.rs`) regenerates a fixture from the
source's GGUF headers alone, read by HTTP Range: the output is byte-identical to the box's fixtures for V4.1, Qwen3.8
and GLM-5.3-Flash. V4.1's DSpark draft is the exception, made from a local copy with no public header.

### 5. Card body

Compose, do not write kernels first: MiMo's body is attention, a dense block and a routed block over the shared kernels,
and the step is a walk of `runtime::sched` over `runtime::layer` programs (`gpu-mimo2/src/lib.rs:1-20`). A new kernel
is a separate round with its own `ptx-scan` proof (the table in `AGENTS.md`, "Derive first"). Add the family to
`Program`/`program_of` and give it coverage rows so the coverage list shrinks to empty: the rows only this family runs
live in its own module (an `AVAILABLE` table such as `arch/minimaxm3/spec.rs`, named by the family's `coverage` entry
in `registry::ROWS`, so only a file of that row is checked against them), and a shared kernel's row is reused by its
`runs` predicate instead of adding the family's `Program` to the common table in `coverage.rs`, where `check-model-tax`
counts each name. Build with `cargo oxide`, never plain `cargo` (`docs/BUILD.md:55`). `just mac-check` type-checks it
without a card.

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
planner (`plan_host_routed` in `crates/placement/src/placement.rs`, for a model whose routed experts all sit on
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

- `tools/gate-paths.tsv`: one glob row for the new reader, `crates/model/src/arch/<arch>/**<TAB>derive<TAB>why`, so
  `just narrow` bounds the next change to it instead of falling back to the full list. The table derives its gates: `derive`
  stands for every gate whose target reaches a file of the glob or names its module, written out on read
  (`python3 tools/recipes.py rows` prints them), so a new gate or a new edge needs no edit and there is nothing to fill
  (`check --fill` is gone). Write gate names beside `derive` only for a gate that links the file and the derivation does not
  reach; `tools/gate-batch-resources.tsv:36` for a solo recipe;
- `tools/ref/ptx-shapes.tsv:200-202` for each PTX entry (the cc 12.1 tables in `tools/ref/ptx-shapes/` take the entry's row
  from a scan on a cc 12.1 card); `tools/unsafe-ratchet.txt:174` for a new crate;
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

## The common-owner table: fill it in before the body

A model's spec, and its pull request, open with one row for each common owner below. Each row says one of three
things:
- **attaches**: the model plugs into the owner;
- **already attached**: the owner works for the model with no new code;
- **does not attach**: the row names the architecture fact that forces this.

"Later" is not a value. A row left for later needs a maintainer to cut it explicitly, and the cut is written into the
row.

The rule exists because skipping a row costs twice:
- you pay once in a slower first version;
- you pay again in a copy that someone has to lift later.

MiMo's first card-expert design skipped two rows, with these results:
- **The card leg.** The design planned a fourth copy of the leg, when three already existed (GLM's
  `crates/gpu-glm5next/src/ffn.rs:503`, Qwen3.8's `crates/gpu/src/arch/qwen3moe/card38.rs`, V4.1's
  `crates/gpu-deepseek41/src/chain/ffn/batch.rs`).
- **Residency.** The design placed experts statically, by id. The common residency machine never ran:
  `crates/gpu-mimo2/src/body.rs:961` still returns `None`. The predicted gain was smaller than it had to be.

| Owner | Where it lives | How a model attaches |
|---|---|---|
| Placement plan | `placement::plan_routed` in `crates/placement/src/placement.rs` | Pass the model's card rule; no model-owned planner |
| Kernel table | The card columns of each type's `QROWS` row (`crates/gguf/src/qtype.rs`), read by operation in `crates/placement/src/kernels.rs` | A quant type gets its card columns in its row; the body dispatches through `kernels`, never by matching `GgmlType` itself |
| Card expert leg | Each family's own today (the three files above) | Call an existing leg, or lift it to a common owner in `crates/gpu`; never add a copy |
| Residency (hot experts move to the card at run time) | `crates/gpu/src/host/served.rs` (`TierBody::residency_parts`, `ResidencyParts`) | The body's `residency_parts` returns `Some`; glue like GLM's `crates/gpu-glm5next/src/swap.rs` (72 lines) |
| Expert tier (a second card holds experts) | `crates/gpu/src/host/tier.rs` | Glue like `crates/gpu-glm5next/src/tier.rs` or `crates/gpu/src/arch/qwen3moe/tier38.rs` |
| Prompt walk | `crates/runtime/src/prompt.rs` | The body implements the walk's hooks; no per-model prompt loop |
| Resident slots (`--parallel N`) | `crates/gpu/src/model/slots.rs` (`Slots`), `crates/placement/src/slots.rs` | `impl Slots` for the body |
| Serve seat | `crates/gpu-gates/src/bin/shared/serve_seats/` | One seat file per model, shaped like `glm.rs` or `mimo2.rs` |
| Timing | The lease runners in `tools/ref/` | Through the serve seat or an existing runner; no per-model timing driver |

If a row needs a copy of another model's code, stop and say so in the issue: that copy is a lift to do first.

## A new host module

A host module is the CPU step under `crates/model/src/` that a family runs before a card body exists. It refuses by name, the way `embed` and `ops` already do. These four tests land with the module.

- **Size products use `checked_mul`.** A wrapping `usize` product turns a huge extent into a short buffer, and the next index walks off it. `checked_mul`, and `checked_add` when a sum is a length, returns `None` into a named error. The embedding table does this for its row bytes (`crates/model/src/embed.rs:66-69`). The test builds two extents whose product does not fit in `usize` and expects that error.
- **A zero extent the step cannot run is a named error.** Zero rows, zero channels or zero taps are an impossible geometry, so the step returns its error instead of an empty success. The embedding table refuses a file that holds no rows (`embed.rs:63-65`). A count the format allows to be zero, such as a per-layer array in step 1, is a different contract and stays allowed.
- **A NaN is not a defined output.** A non-finite lane is a named error (`AGENTS.md`, No silent failure). Compare and sort floats with `total_cmp` (`docs/rust-quality.md`, R23). Plant one NaN and assert the refusal. A column-wise op reports the column through `first_non_finite_col` (`crates/model/src/ops.rs:853`); the test is `first_non_finite_col_finds_every_position_in_a_chunk`.
- **Pin the FMA order.** Where the host expression is the contract a later kernel matches, write `mul_add` for a fused product and a separate multiply and add where the roundings stay apart (`docs/rust-quality.md`, R29). `v_expf` states that contract: every `mul_add` is the reference's fused op (`crates/model/src/arch/deepseek2/attn.rs:713`). The test uses values where `a.mul_add(b, c)` and `a * b + c` differ in bits and asserts the fused bits. Zeros and exact products leave the two forms equal, so they do not pin the order.

## Where you stop today

Steps 1, 2 (code), 8, 9 and the bookkeeping of 10 you can do and verify on an x86_64 Linux machine. Steps 3 and 4
(reference sets) need ik and, for the real tier, the model file in the places the tools expect (`/models/...`, 133
path literals across `crates/**/*.rs`, `grep -rn '"/models/\|"/root/\|/home/user'`; many are comments). Steps 5 to 7
and every `just gate-*` run need an NVIDIA card plus a way past `tools/box.sh`, which does not exist yet:
[`local-check.md`](local-check.md) designs it. Until then a maintainer takes over at step 3, runs the dumps, and runs
the card gates on the box. Say in your pull request which steps you did and how you verified each.
