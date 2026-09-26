# Session: one `Session` over `GpuModel<B>`, one generation loop, one `bloomery` CLI (round `sessiondesign`)

**Status.** Design memo from the read-only round `sessiondesign` (2026-09-27), for rebuild wave 3's `session`
(`docs/rebuild.md` §6-1: GD4, GC2). It read main `e5cc69a`. Numbers are measured with their source, or [derived].

**The lead's dispositions (2026-09-27).**
- The shape is taken: three crates (`runtime` host-only, `app` GPU with bin `bloomery`, `serve` HTTP); the rounds
  `session` → `oneloop` (chat's batch feed proven bit-identical before opslib turns V4.1's prompt contract banded)
  → `draftserve` ∥ `cli`, with `seqstate` in parallel from `oneloop` on (Q8).
- Wave fit (Open 12): the session work is a sequence in wave 3's `session` slot and spills into wave 4. It lands
  **before** `gatesproc` (GD3), reversing GD4's order: argv shims keep the recipes, runners and records byte-identical,
  and `gatesproc` then writes its in-process cases once, against `Session`.
- Recurrent rollback: `StoreRule::Recurrent` with m lanes and a device-scalar committed lane first (option b); the
  transition log (d) waits for acceptance data on GLM; host snapshots never on the token path (Q3d).
- Serve's penalty history (Open 11): prompt + generated, mainline's rule (`init_sampler`). Penalties are unreachable
  today (`api.rs:480-481`), so nothing a user sees changes now.
- Open 1 (the READONLY rsync): no job ran from `~/repo/bloomery` on the box that morning (both leads' batches and
  sittings ran from track directories); `toolsw2` T2 made READONLY sync nothing.
- The six assumptions about gpumodel ("What depends on details gpumodel leaves open") went into the `gpumodel` round's
  spec: `Rows`' m as an associated const, the graph key by rows alone, an explicit `capture(Rows(m))`, `Prompt` as a
  capability, `Weights` shareable for a same-card MTP draft, one writer of the position.

---

**Base.** I read main `e5cc69a`. Its code is identical to the spec's `f7d6ee0`: the one commit in between changes only `docs/cards/*`, `docs/plan-triage.md` and `docs/research/models-survey.md`.

**Process notes.**
- No message from the user reached this round.
- **Side effect.** Two `BLOOMERY_BOX_READONLY=1 tools/box.sh` calls from the main tree rsynced the clean main tree to `~/repo/bloomery` on the box. That rsync is `--delete`, because READONLY has no no-sync yet (triage `:144`; toolsw2 T2 is in flight). This is Open item 1.
- I did no box build, no box run, no lease and no GPU work. I read the references from local copies, fetched once through READONLY box.sh:
  - llama.cpp-mainline `53ed051ce`
  - mistral.rs
  - exllamav3
  - tabbyAPI

**Units.** MB = 10⁶ B. "674 GB/s" is `bw_card` (`tools/flow/constants.tsv:10`): a *kernel streaming* rate, measured on the A6000 with large launches, band 600–701 (`docs/plan.md:72`). It is not a measured D2D copy rate. Every copy time below is **[derived: bytes / 674 GB/s]**, and a copy of S bytes moves 2S.

---

## 0a. Paper decomposition

| Term | From | Value | Settles it |
|---|---|---|---|
| Product code to move | wc; GC2 `gpucore.md:153-188` | `generate_ds41.rs` 1,732 lines, `bind.rs` 599, `generate.rs` 203, `ds41_dspark.rs` 296, `bloomery_chat.rs` 356, `bloomery_serve_ds41.rs` 236, `genloop.rs` 463, `sampling.rs` 146 (deleted); 895 product lines live in the gates crate [measured, GC2] | code |
| Loads a product-gate batch pays | `gates-ds41.md:36`, `:16`, `:19` | chat 6 + serve 2 + draft 4 + dspark-loop 3 = 15 loads × 30.0–41.9 s = 450–629 s [derived]; the gate-times medians give 615 s | measured medians |
| Chat turn, prompt by steps vs batch | README `:66`, `:71`; AGENTS prefill rows | P 512 ≈ 4.8×, P 4096 ≈ 6.5–7.4× [derived, §Q5] | arithmetic on measured rows; the step-feed rate is a proxy (Open 8) |
| Recurrent rollback bytes per verify | modelspec `:735-737`, `:768-770` | tables in §Q3 | arithmetic; the D2D rate stays a label (Open 2) |
| DSpark feature hop per pair pass | `body.rs:1933-1935` (width = layers × n_embd); n_embd 5,120 (`crates/model/tests/ds41_meta.rs:155`); target layers [37, 38, 39] (`gate_dspark_read.rs:60`) | 2 rows × 3 × 5,120 × 4 B = 122,880 B → D2H 6.4 µs at 19.15 GB/s + H2D 4.7 µs at 26.28 GB/s [derived; `constants.tsv:103`, `:11`] ≈ 0.05 % of a 22.3 ms pass | arithmetic: **not a lever** |
| Chat's full-logits read under greedy | `bloomery_chat.rs:298`; n_vocab 129,280 (`ds41_meta.rs:205-206`) | 517,120 B per token → 27 µs at 19.15 GB/s [derived] = 0.12 % of 22.3 ms | arithmetic: below the ruler; it goes away with the one loop |

**Terms the derivation cannot fix.** Each is a future round's one run, with its expected value:
- the A6000 D2D copy rate at 63–153 MB: expected 0.19–0.22 ms and 0.44–0.51 ms (Open 2);
- draft acceptance on Qwen3.6 and GLM: no expectation, no source (Open 3).

The chat switch needs **no** timed A/B: its predicted effect (≈ 4.8×) is far outside the ruler.

## 0b. Resource timeline

- **V4.1 decode, per token.**
  - Prose 512, plan (a), A6000: 44.8 tok/s (README `:66`, rig-log 09-25#dspark-loop-tps), so 22.3 ms per token [derived].
  - The nsys row (depth 6, n = 8): ~~step 31.6 ms = card critical 8.56 + bridge 24.06 (`plan-triage.md:173`)~~ a generated step 31.6 ms = card critical 8.56 + bridges 22.61 + idle between replays 0.42 (replays 6–12; 24.06 was the mean over all 13 replays, the six prompt-feed replays included — corrected by the lead 09-27 from `tools/ref/nsys-bridge.py` on the same sqlite).
  - Per layer the chain is card → host experts → card. The card's expert shadow sits under the host section, so a layer costs the max of the two, and the wall is the chain (`ds41.md:67-71`).
  - The session's own host work per token (the sampler pick, the stop scan, the UTF-8 decoder) is serial, because the next step needs the token. It costs µs, except the sampler chain over 129,280 logits, which is unmeasured (Open 7).
  - **Flow change for serve.** Today serve runs its gen loop on the connection thread and sends two channel messages per token to the engine thread (`genloop.rs:331-436`, `bind.rs:339-358`). That is µs per token, but a draft loop cannot live there. Move the whole request onto the engine thread (§Q6).
- **DSpark pass.**
  - The target pair pass costs 1.58–1.61× a step, and 1.61–1.65× with DSpark (rig-log `2026-09-25.md:183-189` via DS6).
  - The draft's propose on the 3090 is serial with the target: propose → verify → append.
  - The feature hop is 11 µs (above).
- **Chat prompt.**
  - Steps pay the per-token bridge, which reads each routed expert's weights once per token.
  - The prompt call amortizes those reads over the columns: the host union, SIMD plus bulk.
  - At lcg P 4096 the wall is the host union, 68.6 of 73.8 ms per layer-batch (`ds41.md:77-78`). The session does not move that; chat only gets onto it.
- **Qwen3.6 verify.**
  - The model is whole-card (modelspec `:769`), so the wall is card bytes and every extra recurrent-state byte lands on it.
  - The flow lever is **deferral**: do not write state a reject would discard; fold the accepted rows into the next pass, which reads the state anyway. That beats shortening a copy (§Q3, option d).
- **GLM** [assumed host-served like V4.1; its placement is undecided].
  - The KDA mixer runs on the card chain before the route. It is not in the expert shadow (`ds41.md:71` covers card expert work only).
  - So the extra state bytes are additive there too (Open 9).

---

## Q1. Inventory today

### 1a. Drivers × facts

`gd` = `generate_ds41` `drive`; `chat` = lib `Generator` + `bloomery-chat`; `serve` = `Ds41Engine` + `bloomery-serve-ds41` + serve's `genloop`; `dsp` = the DSpark loop; `q3` = `generate_qwen3moe`.

| Fact | gd | chat | serve | dsp | q3 |
|---|---|---|---|---|---|
| Position | `m.pos()` = `GpuModel.pos` (`model.rs:364`) + `Body.history` (`body.rs:649`) | same | same, plus the mirror `Ds41Engine.pos` (`bind.rs:369`) and `Slot.held` ids (`genloop.rs:151-157`) | its own `committed()` (`ds41_dspark.rs:147-149`) is checked against the target (`:277-283`) | `GpuModel.pos` |
| Rollback | `lookup_pass` → `m.rollback(pos-1)` (`generate_ds41.rs:1403-1411`) | none (single turn) | `keepable` → `keep_point` (`bind.rs:561-567`, `bloomery_serve_ds41.rs:181`); `cut` → `Cmd::Rollback` (`bind.rs:569-576`) | `m.rollback(pos+1)` on reject (`ds41_dspark.rs:292-294`) | none (refusing default) |
| KV and state reset | `m.reset()`, also after the pair-capture pass (`generate_ds41.rs:1314-1315`) | `Generator` | `Cmd::Reset` (`bind.rs:339-350`) | `d.reset()` (`generate_ds41.rs:1428`) | `reset` |
| Sampling chain | greedy only | `crates/sampler`; history = prompt + generated (`bloomery_chat.rs:292`, `:316`) | serve's `Sampler` trait over `sampler_factory`, **silent argmax fallback** (`bind.rs:167-174`); history = generated only (`genloop.rs:379`, `:410`); a second sampler in `serve/src/sampling.rs` (CPU5) | greedy | greedy |
| Stop | `-n` | `tok.eog()` set (`:317`); `pos >= ctx_max` (`:328`) | `vocab.eos()` only (`genloop.rs:371`, `:385`) + `StopScan` + `ctx_max` | `-n` | `-n` |
| UTF-8 decode | none (ids) | `tokenizer::Decoder<'t>` (`decode.rs:76-134`) | copy `VocabDecoder` (`bind.rs:105-146`); mock `Utf8Decoder` (`mock.rs:358-389`) | — | none |
| Prompt feed | batch, steps, checked (`:1131-1174`), dspark = batch with `FeatureRows` (`:1416-1462`) | **steps**: `generate.rs:137-140` → `GpuModel::step` | batch or steps via closure (`bloomery_serve_ds41.rs:182-188`) | `feed` step-per-id (`ds41_dspark.rs:249-259`), or batch inside `feed_dspark` | `prefill` (`arch/qwen3moe/prefill.rs:518`) |
| Draft | plain, lookup, dspark (`Draft` enum `:951-981`; `decode_draft` `:1305-1399`) | none | none (`DraftProps` only describes, `serve/src/engine.rs:210-216`) | `pass` (`:273-296`), WIDTH 1 (`:45`) | none |
| Stats and records | kind list `GENERATE_DS41` (`record.rs:1033-1069`) | `BLOOMERY_CHAT` (`:1072-1083`) | `BLOOMERY_SERVE_DS41` (`:1086-1094`); `LISTENING` (`bloomery_serve_ds41.rs:203-207`) | `DRAFT_SUMMARY` via gd | `println!` outside `record.rs`, no `at_main` (`generate_qwen3moe.rs:148-262`) |
| Plan print | `print_plan` (`:986-997`) called at `:474`, then `body::open` plans again at `:507` (triage `:142`) | `print_plan` (`:269-280`) | `print_plan` (`:215-235`) | — | own lines |
| Levers | every Parsed lever at `main`; the acted-on set differs per binary (triage `:147`, `levers2`); `DRAFT` is gd-only | the same parse; the prefill mode is never used | the same; `env!("BLOOMERY_SERVE_COMMIT")` outside the registry (`api.rs:660`) | `BLOOMERY_DSPARK_CARD` read in place (`ds41_dspark.rs:72-82`), no registry row (triage `:142`), owner `session` in `tools/levers-direct.txt` | `QWEN3_UBATCH` and `GQA_MMA` in place ([03]) |

### 1b. Facts with more than one owner

| Fact | Owners (cited) | Target owner |
|---|---|---|
| Position | `GpuModel.pos` (`model.rs:364`); `Body.history.len()` (`body.rs:649`, cross-checked in `decode_input` `:2010` and `begin_call`); `Ds41Engine.pos` (`bind.rs:369`); `Slot.held` (`genloop.rs:151-157`) | `SeqState` inside `GpuModel` (§Q3); serve reads it and keeps no mirror |
| Prompt feed | `generate.rs:137-140`; `generate_ds41.rs:1131-1174` and `:1416-1462`; `bloomery_serve_ds41.rs:182-188` | `Target::prompt` → `B::prompt` |
| Placement enum | `generate.rs:29-61`; `generate_ds41.rs:230-251` | one `Place` in the app crate |
| Plan print | `bloomery_chat.rs:269`, `bloomery_serve_ds41.rs:215`, `generate_ds41.rs:986`, `gate_deepseek41_load.rs:230`, plus `body::open`'s own plan | `Session::open` prints once |
| Sampler | `crates/sampler` (`lib.rs:125`, `:173`); `serve/src/sampling.rs` (CPU5: other RNG, other NaN rule, no penalties) | `crates/sampler` |
| Sampler params | `sampler::SamplerParams` (`lib.rs:25-41`); `serve::SamplingParams` (`engine.rs:218-241`); `bind.rs:152-163` | `SamplerParams` |
| Penalty history | chat: prompt + generated; serve: generated only (above) | prompt + generated, as mainline's server does (`init_sampler` accepts the prompt tokens, `server-context.cpp:409-427`) |
| Stop set | `eog()` (`bloomery_chat.rs:317`) vs `eos()` (`genloop.rs:371`) | one `Stop` over `eog()` |
| Context limit | `bloomery_chat.rs:328`; serve's `ctx_max` | `Target::ctx` |
| UTF-8 decoder | `decode.rs:76-134`; `bind.rs:105-146`; `mock.rs:358-389` | an owned-vocabulary decoder in `crates/tokenizer` |
| Generation loop | `genloop.rs:331-436`; `bloomery_chat.rs:284-355`; `generate_ds41.rs:1178-1399` | `runtime::generate` |
| Draft loop | `decode_draft` + `lookup_pass`; `ds41_dspark.rs:273-296` | `Speculative<D>` |
| Pair capture | dummy pair + `reset` (`generate_ds41.rs:1314-1315`) | `Session::with_draft` captures `Rows(m)` |
| Load record | `LOAD` (`record.rs:427`, gd) vs `LOAD_GENERATOR` (`:456`, chat and serve); both have head `load` | one kind at the next schema bump (§Q6) |

### 1c. The three surfaces

- **`gpu::Engine`** (`model.rs:252-259`; `step`, `reset`, `seed_depth`, `pos`, `resident_bytes`, `arch`).
  - It is the product surface today by accident. It becomes V2-Lite's instrument surface only; `seed_depth` stays there under `Instrumented` (gpumodel item 4).
  - The product uses `runtime::Target` and `Verify`. v2fence decides whether `gpu::Engine` survives.
- **`serve::Engine`** (`engine.rs:97-148`; token-level `prefill`, `next`, …).
  - Becomes request-level: `generate` once per request plus `keepable`, `cut`, `reset`, `save_state`, `restore_state`, `describe`, `props`.
  - `dyn` is allowed there: one call per request.
- **`AnyEngine`** (`gpu-gates/src/engine.rs:25-28`; it refuses V4.1 at `:37-47`).
  - Becomes the CLI's architecture dispatch on `ModelSpec.arch` (modelspec §6 step 4, `modelspec-design.md:790-796`).

---

## Q2. `Session`

**Layering.**
- `crates/runtime` is **host-only**, with no `crates/gpu` dependency. It holds the generation loop, the target traits, the draft trait, `Lookup`, `Stop`, and later `SeqState`'s rules. It is testable without a card.
- `crates/app` is the GPU side. The spec calls it `crates/cli`; `rebuild.md` §2 calls it `app`; it is one crate, lib plus bin `bloomery`. It holds `Session<B>`, the card drafts and the engine thread.
- `crates/serve` stays host-only: HTTP, templates, request parsing and slot files. It depends on `runtime`, and its mock implements `runtime::Target`, so `gate-serve` keeps exercising the one loop.

```rust
// crates/runtime — host only
pub trait Target {
    fn pos(&self) -> u32;                                                        // reads the model's position; no mirror
    fn ctx(&self) -> u32;
    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, TargetError>; // prompt schedule, from pos()
    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, TargetError>;       // decode schedule, m = 1
    fn keepable(&self, n: u32) -> u32;                                            // longest prefix ≤ n every store keeps
    fn cut(&mut self, n: u32) -> Result<(), TargetError>;                         // named refusal unless keepable(n) == n
    fn reset(&mut self) -> Result<(), TargetError>;
}
pub enum Want { Argmax, Logits }                  // greedy never reads the row back
pub enum Out<'a> { Argmax(u32), Logits(&'a [f32]) }

pub trait Verify: Target {                        // the verify schedule
    const MAX_ROWS: usize;                        // B's Rows statement (V4.1: 2 today)
    fn verify(&mut self, rows: &[u32], want: Want) -> Result<RowsOut<'_>, TargetError>; // 2 ≤ m ≤ MAX_ROWS
    /// Keep rows[..accepted] (1 ≤ accepted ≤ m). Position moves; recurrent stores
    /// take the lane (or the logged rows) of row accepted − 1; the rest is rolled back.
    fn commit(&mut self, accepted: usize) -> Result<(), TargetError>;
}
pub trait Tapped: Verify { fn taps(&self) -> Taps<'_>; }   // hidden rows of the last pass
pub enum TapNeed { None, Final, Layers(&'static [u16]) }

pub trait Draft<T: Verify> {
    const WIDTH: usize;                                     // ids per proposal; the verify runs WIDTH + 1 rows
    const TAPS: TapNeed;
    const FITS: () = assert!(Self::WIDTH + 1 <= T::MAX_ROWS); // referenced by Speculative's impl → compile-time error
    fn begin(&mut self, t: &T, prompt: &[u32]) -> Result<(), DraftError>;
    fn propose(&mut self, t: &T, last: u32, out: &mut Proposal) -> Result<(), DraftError>;
    fn accept(&mut self, t: &T, rows: &[u32], accepted: usize) -> Result<(), DraftError>;
}
pub trait Advance<T: Target> {
    fn pass(&mut self, t: &mut T, last: u32, pick: &mut Pick<'_>) -> Result<Committed, GenError>;
}
pub struct Plain;                   impl<T: Target> Advance<T> for Plain { /* step */ }
pub struct Speculative<D>(pub D);   impl<T: Verify, D: Draft<T>> Advance<T> for Speculative<D> { /* propose → verify → pick per row → commit → accept */ }

/// The one generation loop: the one sampler (crates/sampler), the one Stop, the one decoder.
pub fn generate<T: Target, A: Advance<T>>(
    t: &mut T, a: &mut A, vocab: &Vocab, req: &GenRequest, sink: &mut impl TokenSink,
) -> Result<GenOutcome, GenError>;
```

```rust
// crates/app — GPU side, bin `bloomery`
pub struct Session<B: ChainBody> {
    model: GpuModel<B>,   // one card, the graph cache keyed by rows, the position
    vocab: Arc<Vocab>,    // owned: stop set, decoder, template (ModelSpec.chat)
    stats: Stats,         // → DRAFT_SUMMARY, TIME_PASS, STAT_* kinds
}
pub trait Open: ChainBody + Sized {               // gpumodel item 5: a constructor per model
    type Cfg;                                     // OpenCfg (V4.1), OpenOpts{ctx, ubatch, flash} (Qwen3)
    fn open(card: Card, spec: &ModelSpec, cfg: &Self::Cfg, rec: &mut Records) -> Result<GpuModel<Self>, GpuError>;
}
pub trait Prompt: ChainBody {                     // NOT in gpumodel's list (see "depends on", item 5)
    fn prepare(m: &mut GpuModel<Self>) -> Result<usize, GpuError>;     // prompt buffers → PREFILL_BYTES
    fn prompt(m: &mut GpuModel<Self>, ids: &[u32], taps: Option<TapSink<'_>>) -> Result<u32, GpuError>;
}
pub trait Taps: ChainBody {                       // V4.1: attach_features / read_features / FeatureRows
    fn attach(m: &mut GpuModel<Self>, need: TapNeed) -> Result<(), GpuError>;
}

impl<B: Open + Prompt> Session<B> {
    /// Load, capture Rows(1), B::prepare; LOAD, CAPTURE, PREFILL_BYTES printed once, from one plan.
    pub fn open(a: &OpenArgs<'_>, rec: &mut Records) -> Result<Self, SessionError>;
}
impl<B: ChainBody + Rows + Rollback> Session<B> {
    /// Captures Rows(D::WIDTH + 1) (CAPTURE_PAIR) and arms D::TAPS; replaces the pair + reset trick.
    pub fn with_draft<D: Draft<Self>>(&mut self, d: D, rec: &mut Records) -> Result<Speculative<D>, SessionError>
    where Self: Verify;
}
impl<B: ChainBody + Prompt> runtime::Target for Session<B> { /* step → model.step; prompt → B::prompt */ }
impl<B: ChainBody + Prompt + Rows + Rollback> runtime::Verify for Session<B> { /* verify → model.step_rows */ }
```

**The three schedules.** The session calls, and never names CED, G or ubatch; those are body levers, parsed at `main` and passed in `Cfg`.
- **decode**: `Target::step` → `GpuModel::step`, graph key `Rows(1)`.
- **verify**: `Verify::verify` → `GpuModel::step_rows`, graph key `Rows(m)`, only where `B: Rows`.
- **prompt**: `Target::prompt` → `B::prompt`, eager.

**What is generic, and what a model adds.**
- Generic: the loop, sampler, stop, decoder, drafts' logic, `SeqState` rules, record printing and lever reading.
- A model adds:
  - `ChainBody` plus its capabilities: `Rows` (with its m), `Rollback` (its stores' keep rules), `Prompt`, `Taps`, `HostServed`;
  - `Open`;
  - its `ChatSpec` (template, tools, reasoning — modelspec).

**Refusal at compile time and at runtime.**
- `s.with_draft(Lookup::new(), rec)` on `Session<Deepseek2Body>` does not compile (E0599: the bounds `Rows`, `Rollback` are not satisfied).
- `generate(&mut s, &mut Speculative(Lookup::new()), …)` fails with E0277 (`Session<Deepseek2Body>: Verify`).
- A draft whose `WIDTH + 1 > MAX_ROWS` fails at monomorphization through `FITS`.
- The runtime twin is the CLI's single dispatch `match (spec.arch, draft)`. For example, `(Qwen3moe, Some(_))` becomes `CliError::Draft { arch, draft, why: "qwen3moe has no Rows + Rollback yet (Q3-3)" }`, a named refusal for the user.

**One owner each for the sampler, stop and decode.** Inside `runtime::generate`:
- `Sampler::new(req.sampling)` from `crates/sampler`;
- `Stop::new(vocab.eog(), req.stop, req.max_tokens, t.ctx())`;
- `vocab.decoder()`.

`serve/src/sampling.rs`, serve's param struct, chat's loop, `genloop.rs`'s loop and `bind.rs:105-146` all go. Unsupported request fields become a named 400; the `bind.rs:167-174` fallback goes.

**Sampling with drafts.** Greedy verification compares argmax on the card. Sampled verification needs m logits rows:
- mainline samples each verified row with the full chain on the host and accepts while the sample equals the draft (`server-context.cpp:70-95`);
- mistral.rs verifies on the device (`greedy_device_verify_batch`, `sparse_rejection_device_verify_batch`, `speculative/driver.rs:23-27`).

Until the m-row head exposes rows ("depends on", item 3), a draft with a non-greedy request is a named refusal.

---

## Q3. `SeqState` and rollback with recurrent layers

### 3a. `SeqState`

It goes inside `GpuModel` in place of `pos: u32`, which keeps gpumodel's fixed shape ("GpuModel holds the position"). It also absorbs `Body.history`, `holes`, `holds` and the CED need (DS3, `ds41.md:224-251`). Its crate is host-only and below `crates/gpu`.

```rust
pub struct SeqState {
    history: Vec<u32>,                 // ids; len() is the position — the only one
    holes: Vec<Range<u32>>,            // prompt-call CED holes (V4.1)
    stores: Vec<StoreRule>,            // one rule per state store
    rows: Option<RowsInFlight { first: u32, m: u8 }>,
    checkpoints: Vec<Checkpoint { pos: u32, id: HostSlot }>,  // host copies of non-positional state
}
pub enum StoreRule {
    Positional,                                // KV planes, latent/compressed rows, index keys: any cut ≤ len
    Ratio { ratio: u8 },                       // V4.1 compressor state ring (Holds, body.rs:1765-1842)
    Window { slots: u32, shadow_from: u32 },   // V4.1 raw window ring + pinned shadow (body.rs:1238-1249)
    Recurrent { lanes: u8, committed: u8 },    // GDN/KDA: m lanes and the committed lane (b); or a row log (d)
}
impl SeqState {
    pub fn keepable(&self, n: u32) -> u32;                 // today's keep_point (body.rs:1225-1282), generalized
    pub fn cut(&mut self, n: u32) -> Result<CutPlan, SeqError>;     // per store: restore rows / load checkpoint
    pub fn begin_rows(&mut self, first: u32, m: u8) -> Result<(), SeqError>;
    pub fn commit_rows(&mut self, accepted: u8) -> Result<CommitPlan, SeqError>; // lane index for a device scalar
}
```

The rules are pure functions over this struct, so they get host tests with FAIL-first mutants. That is DS3's "testable without a card".

### 3b. Options, per verify of m rows

Sizes from modelspec, all [derived, f32 state]:
- Qwen3.6: S = 62,914,560 B, C = 2,949,120 B, 30 GDN layers (`modelspec-design.md:768-770`).
- GLM: S = 142,606,336 B, C = 10,027,008 B, 34 KDA layers (`:735-737`).
- A GDN or KDA kernel that keeps the state in registers across the m rows reads S once and writes it once, whatever m is. That is mainline's form, `gated_delta_net.cu:57-61`, `:160-166` (linear-attn §2.2). Everything below is **extra** over that.

**(a) Snapshot before each verify, restore and recompute the accepted prefix on reject.**
- Per verify, on the card: 2(S + C).
  - Qwen3.6: 131.7 MB → **0.195 ms**.
  - GLM: 305.3 MB → **0.453 ms**.
  - Both are independent of m [derived: 2(S + C) / 674 GB/s].
- A partial accept adds the same again for the restore, plus the replay:
  - (a1) a whole-model re-run of the accepted rows (mainline); or
  - (a2) the recurrence alone from stashed projections, about 2S (mistral.rs `GdnLayerRollback::Replay { projected, conv_state, recurrent_state }`, `qwen3_5/text.rs:1894-1960`).
- With a **host** snapshot (mainline's FULL path): D2H of S + C costs 3.44 ms (Qwen3.6) or 7.97 ms (GLM) per verify; the restore costs 2.51 or 5.81 ms [derived at 19.15 and 26.28 GB/s, `constants.tsv:103`, `:11`].

**(b) One state per verify row, m lanes per layer, commit by lane index.**
- Extra: (m − 1)·(S + one conv row per layer); the conv rows are Qwen3.6 30 × 8,192 × 4 = 983,040 B and GLM 34 × 3 × 8,192 × 4 = 3,342,336 B.
- Resident: m·S. The input lane can be reused because the kernel reads it whole before writing; the input state is never needed after the verify, since row 0 is always kept.
- `rebuild.md:151`'s "k = 4: GLM 570 MB, Qwen3.6 252 MB, ≈ 0.8 ms" is this option at m = 5, all four extra writes. It is consistent.

| m | Qwen3.6 extra | ms | resident m·S | GLM extra | ms | resident m·S |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 63.9 MB | 0.095 | 125.8 MB | 145.9 MB | 0.217 | 285.2 MB |
| 3 | 127.8 | 0.190 | 188.7 | 291.9 | 0.433 | 427.8 |
| 4 | 191.7 | 0.284 | 251.7 | 437.8 | 0.650 | 570.4 |
| 5 | 255.6 | 0.379 | 314.6 | 583.8 | 0.866 | 713.0 |
| 6 | 319.5 | 0.474 | 377.5 | 729.7 | 1.083 | 855.6 |
| 7 | 383.4 | 0.569 | 440.4 | 875.7 | 1.299 | 998.2 |
| 8 | 447.3 | 0.664 | 503.3 | 1,021.6 | 1.516 | 1,140.9 |

- Times are [derived: extra / 674 GB/s]; at 600 GB/s multiply by 1.12, at 701 by 0.96.
- Against a Qwen3.6 step of ≈ 4.28 ms [derived: 2,884 MB (linear-attn §2.3) / 674 GB/s], this is +2.2 % at m = 2 and +15.5 % at m = 8.

**(c) Recompute from a checkpoint.**
- For speculation, the best checkpoint is "before this verify", and that is (a).
- An older checkpoint replays more whole-model tokens.
- No reference uses (c) on the token path. Its host form is what all three use for prefix reuse (3e).

**(d) What the references do instead: the transition log (mistral.rs).**
- The verify writes no row states. It logs, per row and layer, what a rank-1 apply needs:
  - the conv input,
  - the key,
  - δ,
  - the decay.
- The accepted rows are applied later (`RecurrentSpeculativeStorage::TransitionLog`, `kv_cache/hybrid_cache.rs:36-47`; pool `:69-140`).
- mistral.rs chooses it whenever the model supports it and 2 ≤ lanes ≤ 8; otherwise FullCheckpoints lanes (b); the replay stash (a2) is the fallback (`pipeline/mod.rs:507-531`).
- Bytes per row per layer:
  - Qwen3.6: conv 8,192 × 4 = 32,768 B (f32 as modelspec assumes; mistral.rs stores it in the activation dtype, bf16 → 16,384 B), key 16 × 128 × 4 = 8,192 B (mistral.rs keeps two key banks, `GDN_PENDING_KEY_BANK_COUNT = 2`, `:59`, which I leave out), δ 32 × 128 × 4 = 16,384 B, decay 32 × 4 = 128 B. Total 57,472 B × 30 layers = **1,724,160 B per row** [derived].
  - GLM (KDA: per-channel decay): conv 3 × 8,192 × 4 = 98,304 B, key 32,768 B, δ 32,768 B, decay 32,768 B. Total 196,608 B × 34 = **6,684,672 B per row** [derived; **no reference implements a KDA log**: `speculative_transitions_supported` requires GDN dims and `GdnValueMajor`, `gdn/layer.rs:256-266`].
- Per verify: write m rows and read at most m → ≤ 2m rows.

| m | Qwen3.6 log | ms | GLM log | ms |
|---|---:|---:|---:|---:|
| 2 | 6.90 MB | 0.010 | 26.74 MB | 0.040 |
| 4 | 13.79 | 0.020 | 53.48 | 0.079 |
| 8 | 27.59 | 0.041 | 106.95 | 0.159 |

- **Reading, not settled.** The "no extra state traffic" claim holds only if the next pass applies the pending rows inside the kernel that reads S anyway.
  - mistral.rs's checkpoint kernel takes `pending_conv` and `pending_recurrence` as inputs (`gdn/layer.rs:803-849`).
  - A separate `apply_pending_recurrent_transitions` path also exists (`qwen3_5/text.rs:1569`).
  - If the apply is a separate pass, add 2S per verify (Open 10).

### 3c. Which option each reference uses

- **mainline:**
  - (b) with kernel-written snapshot slots (`gated_delta_net.cu:145-157`, `keep_rs`, K = `n_rs_seq`) and an index-only commit (`llama-memory-recurrent.cpp:101`, `:193-203`). `n_rs_seq = draft.n_max` for MTP, EAGLE3, DFlash and DSpark (`common.h:394-400`, `common.cpp:1723`).
  - For **n-gram drafts** `n_rs_seq` is 0, so the partial `seq_rm` probe fails (`common.cpp:1583-1615`) and the context reads FULL, "speculative decoding will use checkpoints" (`server-context.cpp:1240-1246`).
  - Every verify then takes a host checkpoint and replays on a partial accept (`:3050-3095`, `:3905-3953`): 3.4 ms (Qwen3.6) or 8.0 ms (GLM) per verify [derived]. This is the strongest argument for card-side lanes that our `Lookup` also uses.
  - The V4 compressor state gets the same snapshot planes (`llama-kv-cache-dsv4.cpp:964`).
- **exllamav3:** (b) with k + 1 slots.
  - `recurrent_state` has shape `(B, max_history + 1, heads, k, v)` (`modules/gated_delta_net.py:182-191`).
  - `max_history` = the draft count (`cache/cache.py:123-124`; tabbyAPI `backends/exllamav3/model.py:617-619`).
  - The verify writes each row's history; a rewind **copies** a history slot into slot 0 (`gated_delta_net.py:233-244`, batched), which costs 2S on a partial accept. It does not ping-pong.
  - KDA is supported through the same history path (`:860-879`).
  - Recurrent *draft* models are refused (`generator/generator.py:162-163`).
- **mistral.rs:** (d) preferred, then (b) FullCheckpoints with a `committed_lane` index (`hybrid_cache.rs:1196-1290`), then the (a2) stash.

### 3d. Recommendation

- **`StoreRule::Recurrent` with lanes (b) first.** It is the common core of mainline and mistral.rs and the shape linear-attn §3.3 designed: a device-scalar lane index, one graph per rows key, the kernel writing lane (r + j) mod m. It costs ≤ 0.66 ms at m = 8 on Qwen3.6 [derived].
- **Keep (d) as the kernel-side upgrade behind the same `commit_rows` API** when GLM runs wide verifies: 0.65–1.52 ms → 0.08–0.16 ms at m = 4–8 [derived].
- **Never use host snapshots on the token path.** `Lookup` gets lanes too, unlike mainline.

### 3e. Prefix reuse across chat turns

- **Append-only multi-turn needs nothing.** The state at `len` is current, and `keepable(len) = len`.
- **Regenerate, edit, or a shared system prompt need host checkpoints** at chosen positions. Mainline puts them at user-message starts and at 4 + n_ubatch and 4 tokens before the prompt end (`server-context.cpp:3550-3570`). exllamav3 stashes at page boundaries in a 4 GiB host LRU (`cache/recurrent.py:20-35`).
  - A checkpoint is S + C: 65.9 MB (Qwen3.6) or 152.6 MB (GLM).
  - D2H costs 3.4 or 8.0 ms, H2D 2.5 or 5.8 ms [derived at the measured PCIe rates].
  - My proposal: one checkpoint at each prompt end, which is linear-attn §3.2's policy, bounded by an LRU.
- **`keepable(n)`** = the largest checkpoint position ≤ n (positional stores always keep), intersected with V4.1's ring rules where they apply.
- **A recurrent slot is valid only at exactly its length.** mistral.rs skips a prefix match whose length differs from the snapshot's (`prefix_cacher.rs:834-838`); mainline loads the nearest checkpoint or re-processes (`server-context.cpp:3369-3380`).
- **`slotfile.rs`** (VERSION 1: header + engine bytes, `:1-26`) needs VERSION 2, carrying (pos, S + C) per checkpoint. Only the mock saves today (`bloomery_serve_ds41.rs:200` `slot_save_path: None`).
- **V4.1 after a prompt call** keeps only 0, P and the call's last few positions (`body.rs:1238-1249`). An edited earlier turn re-feeds from 0 unless a checkpoint sits there.

---

## Q4. Drafts as one interface

| Source | Reads from the target | Runs where | Its own state and rollback | Today |
|---|---|---|---|---|
| **n-gram lookup** | token history only | host, µs | index of the committed tokens; only accepted tokens are pushed (`generate_ds41.rs:1370-1374`), so no rollback | `draft.rs:14-61`, WIDTH 1 |
| **DSpark** (`DraftSpec::Block`: width 5, target_layers [37, 38, 39], mask token 128,799, Markov rank 256; `gate_dspark_read.rs:48-60`) | hidden rows at named layers for every committed position (`attach_features`, `body.rs:209-219`; `FeatureTap` `:1871-1953`; `FeatureRows` in prefill) | second `GpuModel` on the other card (`Dspark::open` with `Gpu::for_card`, `ds41_dspark.rs:100-132`) | its KV: `committed()`, `skip_to`; rolled back with the target | `ds41_dspark.rs`; WIDTH 1 (`:45`) because V4.1 `Rows` is 2 |
| **MTP** (GLM: layer 45, "an ordinary DSA layer, but a plain residual and no hc", carried and not run, `modelspec-design.md:537`, `:742`; **Qwen3.6: our file has no nextn**, `:58`, `:475`) | the final pre-norm hidden row of every position (mainline `h_nextn`, `llama-context.cpp:1613-1620`) plus the target's embed and head | the **same card**, on the target's weights | its one-layer KV, positional → cursor. Mainline keeps it in sync by running the MTP layer over every target batch with h shifted by one (`speculative.cpp:1482-1597`); exllamav3 prefills accepted rows' hidden states into the draft cache (`generator.py:1287-1320`). MTP layers are attention-only, not recurrent (`qwen35moe.cpp:18-19`) | none |

**Does one interface hold all three?** Yes: `Draft<T: Verify>` with `WIDTH`, `TAPS`, `begin`, `propose` and `accept` (§Q2). That is mainline's `begin`/`process`/`draft`/`accept` (`common/speculative.cpp:166-172`), with `process` folded into `accept` for one sequence. Mainline dispatches it virtually over a vector of implementations; mistral.rs drives it generically (`try_sample_speculative_causal_gen<P, C>`, `speculative/driver.rs:145`). Ours is monomorphic:

```rust
impl<T: Verify>  Draft<T> for Lookup           { const WIDTH: usize = 1; const TAPS: TapNeed = TapNeed::None; … }
impl<T: Tapped> Draft<T> for CardDraft<DsparkBody> { const TAPS: TapNeed = TapNeed::Layers(&[37, 38, 39]); … } // second GpuModel, other card
impl<T: Tapped> Draft<T> for MtpDraft          { const TAPS: TapNeed = TapNeed::Final; … }             // same card, target's weights
```

The token path is `generate::<Session<B>, Speculative<D>>`, with no `dyn`. The only `dyn` is the prompt call's tap sink, once per group, as today (`prefill.rs:357-363`).

---

## Q5. Chat's batch prefill

**What changes.**
- `Generator::prefill` (`generate.rs:137-140`, one `GpuModel::step` per id) becomes `Target::prompt` → `B::prompt`.
- `B::prepare` runs at open, as serve does already (`bloomery_serve_ds41.rs:177`); `PREFILL_BYTES` must fit chat's `Place::A` budget (`bloomery_chat.rs:102`; Open 5).
- The session is kept across turns.
- A new turn whose ids extend `history` feeds only the suffix. Otherwise it cuts to `keepable(common)` and re-feeds, which is serve's `reuse` rule (`genloop.rs:172-193`).

**Derived gain.** All rows are A6000, plan (a), with a hot list.

| | step feed (proxy: the decode rate) | batch | gain |
|---|---|---|---|
| prose P = 512 | ≤ 512 / 44.8 = 11.4 s (README `:66`; shallower steps are cheaper, so this is an upper bound) | 512 / 216.1 = 2.37 s (`237321a` cardtile, hot list 384, **before** prefillgroup `eb3e08a`; rig-log 09-26#cardtile-ab) | ≈ **4.8×** [derived] |
| prose P = 4096 | 4096 / 44.8 to 4096 / 39.6 = 91.4–103.4 s (39.6 tok/s after a 4096 prompt, README `:71`) | 4096 / 292.1 = 14.0 s | **6.5–7.4×** [derived] |

- Assumption: a step-feed token costs what a decode step costs at the same depth. The step feed has not been re-measured since `43cd107`, where the batch was 2.97× (`gates-ds41.md:147-148`; Open 8).
- The lcg rows (pp512 144.6 at `2f45a79`; pp4096 247.5 at G = 2, `eb3e08a`) use another prompt and do not share this table.

**Nonzero start: works today, nothing to add.**
- V4.1:
  - `feed` takes `first = m.pos()` (`prefill.rs:377`);
  - `begin_call(first, end, …)` (`:388`, `:1111-1142`);
  - `Ced::need` floors every block to `first` (`ced.rs:255-259`), and a unit case is `c.need(700, 1100, &[700], None)` (`:387`);
  - the prefill gate pins consecutive calls 700 + 400, 1800 + 1000 and 300 + 2700 (`gate_deepseek41_prefill.rs:85-94`).
- Serve already feeds from a cut through `body::prefill` (`bloomery_serve_ds41.rs:185`).
- Qwen3's `prefill_with` starts at `pos0 = self.pos()` (`arch/qwen3moe/prefill.rs:554-647`).
- The call refuses "a card that does not run every layer" (`prefill.rs:1117-1122`). Plan (a) runs every layer.

**Proof.**
- Under V4.1's bit contract, which holds until opslib's GEMM branch lands (fixed input 3), the batch state equals the step state bit for bit. That covers:
  - consecutive calls, a rollback after 1,100 and a take-back;
  - G = 1 and 2 (`gate_deepseek41_prefill.rs:85-111`).
- So `gate-gpu-ds41-chat`'s greedy tokens must be **identical to the base**. That is the runtime proof, with `gate-gpu-ds41-prefill`.
- After opslib, or for Qwen3 (banded), chat's pins re-pin with a dated reason and the prefill band judges them.

---

## Q6. The CLI

**Crates.** `crates/runtime` (host); `crates/app` (GPU session, bin `bloomery run|chat|serve|bench`); `crates/serve` (HTTP, host). This is the mistral.rs split: core, cli, server-core (`mistralrs-cli/src/args/mod.rs`: Serve `:43`, Run `:65`, Bench `:188`).

**Serve's threads.** The target is the mistral.rs shape: one engine thread owns the model, and requests arrive over a channel (`lib.rs:1225-1310`, `engine/mod.rs:200-202`, `:1107`).

```text
connection thread (one per connection, api.rs:174-179): parse → template (ChatSpec) → tokenize
   └─ Cmd::Generate{ids, req, sink} ─► engine thread (one; owns CUDA context, pinned buffers, Session<B> + A — bind.rs:6-11)
                                        runtime::generate(...) — the whole request, drafts included
   ◄─ Event::Token{id, text} … Event::Done{stats}   (per-request channel; a cancel flag checked per pass)
```

- The slot stays single (one user, fixed input 5; the concurrent scheduler is C3, L, later, triage `:52`).
- `SessionThread` implements the request-level `serve::Engine`.

**Bench.**
- `bloomery bench` is the instrument (today `generate_ds41 --time`), with one (P, depth, n, arm) per process.
- It emits the same kinds: `TIME_PROMPT`, `TIME_STEP`/`TIME_PASS`, `SMOKE`, `STEP0`, `FED`, `DRAFT_SUMMARY`.
- The runners keep the lease and the witnesses (AGENTS "Never hand-run a benchmark").
- mistral.rs's bench sweeps in-process (`prompt_len`, `depth`, `iterations`, `warmup`; `commands/bench.rs:80-`). Keep ours one process per arm, so each row sits inside its runner's witness block.

**Binaries.**

| Today | Becomes | Why the old name stays for a while |
|---|---|---|
| `generate_ds41` | `bloomery run` / `bench`; the old name is an argv shim | `depth-ds41.sh`'s `bin:<path>:<D>` arms run base trees' `generate_ds41` (`:63-66`, `:276`); `BIN` defaults (`depth-ds41.sh:216`, `nsys-ds41.sh:316`); the recipes `justfile:891-901` |
| `bloomery-chat` | `bloomery chat`; shim | `gate-gpu-ds41-chat` (`justfile:924`) |
| `bloomery-serve-ds41` | `bloomery serve`; shim | `gate-gpu-ds41-serve` (`:934`) |
| `generate_qwen3moe` | `bloomery run --model qwen3moe` **after** its lines become kinds (03's M item) | read by pattern: `depth-qwen3moe.sh:184`, `:458`; `q3pp.py:389-398`; `nsys-gpu.sh:70` |

The shims go once no runner arm points at a base tree that predates them, about one wave.

**Runners.**
- `depth-ds41.sh` (records via `records.py`, `:143-160`), `nsys-ds41.sh`, `ds41pp.py` (records and `--plan`) and `tools/flow/ds41_prefill.py` (plans plus `--counts` through `records.of_kind`): unchanged while the kinds are unchanged.
- `cstate-ab.sh` keeps `^SMOKE|^time ` lines (`:30`), and the heads are kept.
- `tools/gpu-ab.py` reads only V2-Lite `generate` (`INSTRUMENTS = {'time-gpu-generate': 'generate'}`, `:48`), so it is not a `generate_ds41` reader and is unaffected. GD4 listed it by mistake.
- The Qwen3 runners switch to `records.py` together with 03's kinds.

**Records, byte-compatible.**
- Kind lists become per subcommand: `RUN` = `GENERATE_DS41`, `CHAT` = `BLOOMERY_CHAT`, `SERVE` = `BLOOMERY_SERVE_DS41` (`record.rs:1033-1094`).
- `LOAD` and `LOAD_GENERATOR` both have head `load` (`:427`, `:456`), and `LISTENING`'s head is literally `bloomery-serve-ds41:` (`:1020`). Both stay as they are, per subcommand.
- Proof:
  - `bloomery <sub> --records-schema | diff - tools/bloomery/schema/<bin>.jsonl` is empty;
  - `just records-refresh` (`justfile:884-885`) leaves `git diff` empty, which covers the ten `--plan` files;
  - a runner dry-run on the Mac parses a base log and a new log to equal values.
- Unifying the `load` kinds and renaming the `LISTENING` head is a later VERSION 2 bump (`record.rs:29`), with its readers in the same round.

---

## Q7. Boundaries

| Round | My types it touches | Order |
|---|---|---|
| `gpumodel` (wave 2) | `Session` wraps `GpuModel<B>`; `Rows`/`Rollback`/`Instrumented` bounds; graph key; `Open` = its constructors | before everything here |
| `modelspec` (wave 3) | `DraftSpec::Block` → `CardDraft` (width, taps); `ModelSpec.mtp` → `MtpDraft`; `ChatSpec` → serve's template, tool and reasoning per model (fixes `api.rs:1550-1555`, `reasoning.rs:32`); architecture dispatch | in parallel; the session takes `--model <name>` until the reader lands |
| `opslib` (wave 3) | none of my types; it flips V4.1's prompt contract to banded | land chatbatch **before** the selector flips, so the proof stays bit-identical tokens |
| `v2fence` (wave 3) | `gpu::Engine`, `Instrumented`; V2-Lite is not a `Session` target | independent |
| `layerprog` (wave 4) | the three schedules **under** `step`, `step_rows` and `prompt`; `StoreRule::Recurrent` (the verify kernel writes lanes) | after the session, which fixes the call shape so layerprog touches no driver |
| `gatesproc` (wave 4, GD3) | in-process gate cases call `Session` | **after** the session. This reverses GD4's "after GD3" (`gates-ds41.md:159`): with shims, the shell recipes (grep/sed/cmp, `justfile:891-901`) run unchanged, and gatesproc writes its cases once, against one API |
| `gatestoml`, `batchwide` (wave 4) | `bloomery` subcommands as registered gates; nothing | after cli; independent |
| 03: `hybrid.rs`, `arch/qwen3moe/**`, host tier | Qwen3's `Open`/`Prompt` impls wrap `Body::open` and `prefill.rs:518`; `Rows` at m ≤ 8 (Q3-3, `step_rows`: aa designs, 03 signs, `rebuild.md` §6-1); `HostServed`'s m-row `serve_replay` | 03 signs; the session edits none of 03's files |

## Q8. Build order and size

The box minutes come from `gates-ds41.md`: load 30.0–41.9 s (`:16`); loads per recipe (`:36`); 3090-lane medians (`:19`): body gates 1,037 s (6 gates), product recipes 615 s, faults 61 s.

| # | Round | Change class (AGENTS table) | Proof | Gates, box minutes [derived] | Size |
|---|---|---|---|---|---|
| 1 | `session`: `crates/runtime` + `crates/app` `Session`; `generate_ds41`'s drive onto it; `Lookup` and `Dspark` as `Draft`s; one `Place`, one plan print (plans once) | **move**: `just ptx-scan` of `generate_ds41` and `gate_e2e` identical, plus structural lines (capture node counts, dsloop pins 1172/2344, eager = replay) | tokens identical to base in every feed × decode arm; `--records-schema` and `--plan` byte-identical | narrowed host-path batch: body gates 1,037 + product 615 + faults 61 = 1,713 s ≈ **29 min** | M |
| 2 | `oneloop`: `runtime::generate` for chat and serve; one sampler, param struct, `Stop`, decoder; request-level `serve::Engine`; engine thread runs the request; **chat batch feed** as a separate proof item | host-only + move; the feed switch per the prefill gate's contract | greedy tokens identical for chat and serve; `gate-serve`/`gate-sampler` (a mock seed re-pin, dated, if the RNG changes); `gate-gpu-ds41-prefill` | chat 6 + serve 2 loads = 240–335 s, prefill 190–232 s, host gates → **≈ 7–10 min** | M |
| 2b | `draftserve`: drafts in chat and serve (greedy; named refusal when sampled); `BLOOMERY_DSPARK_CARD` → parsed row | host path | plain = draft tokens for chat and serve (the `gate-gpu-ds41-draft` form) | +4 to 8 loads ≈ **2–6 min** | S |
| 3 | `cli`: bin `bloomery`, shims, per-subcommand kind lists, `records-refresh` on the new bin, bench | move | schema diffs; `--plan` identical; runner dry-run parse (Mac); tokens identical for run, chat, serve | product 615 s + builds ≈ **12 min** | M |
| 4 | `seqstate` (DS3 + `StoreRule`): `SeqState` inside `GpuModel` replaces `pos` + `Body.history`; host tests incl. `Recurrent` lanes | **move** (ptx-scan identical, dsloop pins, eager = replay, prefill gate) | new host tests FAIL-first on mutated keep and commit rules | body 1,037 + faults 61 ≈ **18 min** | M |
| later | `mtp` (wave 5, GLM); sampled speculation (m-row logits or device verify); (d) row log | new ops | plain = draft tokens (lossless pin) | — | M–L |

**Order and wave fit.**
- Order: gpumodel → 1 → 2 → {2b, 3}. Round 4 can run in parallel with 2–3: its files are `model.rs` and `body*.rs`, against the gates, serve and app crates.
- `rebuild.md` §6-1 allows at most four aa rounds per wave, and wave 3 already has four slots. So the session work is a sequence inside its one slot, spilling into wave 4. Placing it there is the lead's call.

---

## What depends on details gpumodel leaves open

1. **How `Rows` states its m.**
   - A const gives `Draft::FITS`, a compile-time width check.
   - A method turns it into a named refusal in `with_draft`.
   - I assume a const; V4.1's `PAIR_ROWS = 2` is one already (`body.rs:113`).
2. **Graph key.**
   - Rows only, with a device-scalar lane index, is what `CommitPlan` assumes (linear-attn §3.3).
   - A rows × parity key doubles the verify captures and their node pins.
3. **The m-row head.**
   - `step_rows → [u32; m]` alone (Q3-3) leaves sampled speculation refused.
   - Sampled speculation needs m logits rows (mainline host pick, `server-context.cpp:70-95`) or a device verify (mistral.rs).
4. **`Weights` ownership.**
   - A same-card `MtpDraft` needs the target's embed, head and MTP tensors, shared (`Arc`) or borrowed. Mainline's MTP context "runs on the weights of the main model" (`common.cpp:1300-1318`); exllamav3 uses `attach_to(model)` (`generator.py:248-249`).
   - If `GpuModel` owns `Weights` exclusively, MTP must become a body schedule, which is the mistral.rs "attach to target" shape.
5. **`prompt` is in neither the core trait nor the four capabilities.**
   - The session needs `Prompt`: V4.1's `prefill`, `prepare_prefill` and `prefill_with`; Qwen3's `prefill`; Q3-7's target (`qwen3.md:260-277`).
6. **Position.** gpumodel keeps `pos: u32`. The session reads it and holds no mirror (`bind.rs:369` goes). Round 4 swaps the field for `SeqState` inside `GpuModel`, so the fixed shape still holds.
7. **Fault-read modes** (Qwen3 at call end, V4.1 at group end).
   - `Session::prompt` maps both to one `SessionError::Fault`.
   - After a failed V4.1 call the body has already taken the call back (`prefill.rs:439-467`), so the session must not re-feed on its own.
8. **`HostServed`'s m-row `serve_replay`.** A host-served verify at m > 2 (GLM) needs it.
9. **An explicit `capture(Rows(m))`** in place of the dummy pair + `reset` (`generate_ds41.rs:1314-1315`).

## Recommendation

Build the three-crate split:
- `runtime`, host-only: one monomorphic `generate<T, A>` owning the sampler, stop and decoder; `Target`/`Verify`/`Draft`/`Advance`; `Lookup`.
- `app`, GPU: `Session<B>` with `Open`/`Prompt`/`Taps`, the engine thread, `bloomery run|chat|serve|bench`.
- `serve`: HTTP, with a request-level `Engine`.

Land it as `session` → `oneloop` (chat's batch feed proven bit-identical before opslib flips the contract) → `draftserve` + `cli`. Keep argv shims so the runners, recipes and records stay byte-identical until gatesproc writes its in-process cases against `Session`. In parallel, give `seqstate` the host-tested ledger with `StoreRule::Recurrent`, as m lanes plus a device-scalar committed lane: the common core of mainline and mistral.rs, costing ≤ 0.66 ms at m = 8 on Qwen3.6 [derived].

Leave for later:
- the transition log (d), until GLM's acceptance data says wide verifies pay;
- sampled speculation, until the m-row head exposes rows;
- MTP, until modelspec and gpumodel say how a same-card draft shares weights;
- prefix-reuse host checkpoints and slotfile VERSION 2, until a recurrent model serves;
- the records VERSION 2 unification.

## Open

1. **The READONLY rsync side effect.** Two calls synced main to `~/repo/bloomery` with `--delete` during this round. Step: the lead confirms no main-remote job was mid-run; toolsw2 T2 lands the no-sync.
2. **The D2D copy rate on the A6000 at 63 and 153 MB.** Expected 0.188–0.220 ms and 0.435–0.509 ms at 600–701 GB/s. Step: one `cuMemcpyDtoD` timing row inside a lease with a card. Needed only if option (a) is considered.
3. **Draft acceptance on Qwen3.6 and GLM.** It decides (b) vs (d) and the m. Step: mainline speculative runs with the published Block drafts once the model runs on our engine; V4.1 lookup rates do not transfer.
4. **The Qwen3.6 verify-pass time at m.** It is the denominator of the overhead tables. Step: 03's m-row path timing after Q3-3.
5. **`PREFILL_BYTES` under chat's `Place::A` and budget.** Step: read the `prefill bytes` kind from an existing `generate_ds41 --place a` log.
6. **A Qwen3.6 GGUF that carries nextn.** Ours has none (`modelspec-design.md:58`); mainline loads MTP-only GGUFs (`qwen35moe.cpp:39-41`). Step: an HF file listing and a header range request.
7. **The sampler chain's cost per token over 129,280 logits.** Step: time `gate-sampler`'s chain on one row (host).
8. **The step-feed rate today.** Only needed for a public chat number. Step: one lease row with `BLOOMERY_PREFILL=steps` against batch.
9. **GLM's placement.** It decides whether the KDA extra bytes sit on the card chain. Step: modelspec §5's plan for GLM.
10. **Whether mistral.rs applies pending transitions inside the next verify kernel or in a separate pass.** It decides whether (d) costs 0 or 2S. Step: read the checkpoint kernel in `mistralrs-core/src/cuda/gdn.cu` against `gdn/layer.rs:803-849` and `qwen3_5/text.rs:1569`.
11. **The penalty-history rule for serve** (generated only → prompt + generated): a user-visible change under nonzero penalties, which today are unreachable (`api.rs:480-481`). Step: the lead decides.
12. **Wave fit.** The session rounds exceed one wave-3 slot, and GD4's "after GD3" is reversed here. Step: the lead decides.

## Beyond the brief

Report only; I touched nothing.

- `crates/gpu-gates/src/generate.rs:1-4`, `:137-140`: the "one real step per id" doc and the step feed; round 2 fixes them (XS).
- `crates/serve/src/sampling.rs:1-3`: false header (CPU5); it dies in round 2 (XS).
- `crates/gpu-gates/src/bind.rs:167-174`: argmax fallback when the sampler factory errs, a silent failure (S).
- `crates/serve/src/api.rs:480-481` with `:546-583`: unknown fields (penalties) are ignored while `generation_settings` reports neutral penalties (S).
- `crates/gpu-gates/src/record.rs:1020`: the `LISTENING` head hard-codes the binary name (XS; schema bump).
- `generate.rs:29-61` / `generate_ds41.rs:230-251`: two `Place` enums. `bloomery_chat.rs:269`, `bloomery_serve_ds41.rs:215`, `generate_ds41.rs:986`, `gate_deepseek41_load.rs:230`: four `print_plan`s (S; round 1).
- `crates/gpu-gates/src/bin/bloomery_chat.rs:298`: a full 517,120 B logits read per token under greedy [derived]; 0.12 % of a step (XS).
- `crates/serve/src/genloop.rs:371`: single-EOS stop against the `eog()` set (S).
- `crates/gpu-gates/src/bind.rs:369`: `Ds41Engine.pos`, a third position mirror (XS).
- `docs/research/audit/gates-ds41.md:159`: lists `gpu-ab.py` as a `generate_ds41` reader; it reads V2-Lite `generate` only (`tools/gpu-ab.py:48`) (XS).
- `crates/gpu-gates/src/bin/generate_qwen3moe.rs:148-262`: `println!` outside `record.rs`, no `at_main` (M; 03's).
- `crates/gpu-gates/src/bin/generate_ds41.rs:1413-1415`: `feed_dspark`'s doc says "one step each", but the batch arm runs `prefill_with` (`:1427-1442`) (XS).
- ~~`docs/plan-triage.md:173`: "31.6 = 8.56 + 24.06" does not add up; the sum is 32.62 (XS doc check).~~ Closed 09-27: 24.06 mixed the prompt-feed replays in; see the corrected line in §0b.
- `docs/research/audit/ds41.md:228-230`: DS3's anchors are stale (fields now `body.rs:614-685`; the history check is `:2010`) (XS).
- `docs/research/audit/cpu.md` CPU5: cites `bloomery_serve_ds41.rs:188` for `slot_save_path`; it is `:200` (XS).
- `docs/plan-triage.md:57` (`q35gdn`): "exllamav3 history slot + ping-pong = k+1 slots". exllamav3 **copies** the history slot back into slot 0 (`gated_delta_net.py:233-244`); the lane-index form is mainline's and mistral.rs's (XS).
- `crates/gpu-deepseek41/src/body.rs:1229-1237` (`Holds`, ratio slots): a V4.1 verify at m > 2 cannot roll back to odd positions below len − 1. exllamav3 keeps a ring of the last PAGE_SIZE + m projected rows, so any rewind ≤ 256 is a cursor move (`cache/dsa.py:15-35`); mainline gives the compressor state snapshot planes (`llama-kv-cache-dsv4.cpp:964`). This is needed before V4.1 rows go past 2, e.g. DSpark block 5 (M).
- mistral.rs's deferred GDN decode (write the state every 4 tokens) is batch ≥ 8 only (`GDN_DEFERRED_DECODE_MIN_BATCH = 8`, `cuda/gdn.rs:51`); the Qwen3.6 round should not chase it for one user (note).
