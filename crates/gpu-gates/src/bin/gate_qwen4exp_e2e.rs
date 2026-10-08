//! The Qwen3.8-Flash-Next (`qwen4exp`) end-to-end gate: the whole program —
//! 36 sigmoid-gated delta-rule layers and 12 selecting attention layers, every
//! block in the four gated-residual streams, the PLE site on layer 1, 48
//! MoE blocks whose 512 routed experts run on the host tier (under (k)'s
//! card plan each eligible layer's id prefix on the card) beside a
//! sigmoid-gated shared expert on the card, the head's mix, the q8_0 head and
//! the argmax — loaded by its placement on the gate card, against ik's
//! CPU oracle sets (`refset::arch::qwen4exp`: the 5-token batch set, the step
//! after a fused 4-token prefill, the same after a prefill run node by node,
//! the steps at positions 1,024 and 3,000 of the prose), every set read
//! through its family.
//!
//! The file opens through `PlanInputs::describe` and `Body38::open_placed`,
//! which refuses by name every coverage item past the two the program
//! allows (`bloomery_gpu::arch::qwen3moe::ALLOWED`, the chat surface's; the
//! plain `PlanInputs::read` refuses those two as well). The coverage check
//! holds qwen4exp to `Body38`'s own rows (`crates/model/src/arch/coverage.rs`,
//! not this gate's to change).
//!
//! Tiers (`BLOOMERY_TIER`, `crates/gpu-gates/src/tier.rs`): the real tier
//! runs every clause on the real file. The fixture tier opens the small file
//! `fixture generate` wrote (`ref_model_path`: a whole fixture or a named
//! error), plans with `a` (the largest visible card) under the card budget
//! its header records, and runs the self-consistency clauses — two arms of the
//! engine under one plan agree, a count against the plan's own derivation —
//! while every comparison with ik's output (the free arm's layer outputs, flips
//! and last argmax, the step sets, each ubatch arm's last argmax) is an oracle
//! clause and prints a `deferred(real)` line. The sets' token ids are read in
//! both tiers: the fixture copies the real file's vocabulary. Every layer
//! count, node count and store byte is derived from the header
//! (`q38_fixture::Shape`); the real tier prints each beside the literal it
//! replaced and requires them equal (`move proof` lines).
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds the header's node count
//!   (the embedding row, each layer's launches by its kind, the PLE site and the
//!   head: 1,151 on the real file),
//!   one pair of memory-operation batches a layer of them stream (each layer's go and
//!   wait) and the rest kernels, and the program's own count
//!   (`Body38::step_launches`) is the same; the selecting layers are every
//!   fourth ([`qsa_layers`]); the stores' bytes equal their derivation from
//!   the header ([`store_bytes`]).
//! - (p) one program: the batch set's five tokens as five graph steps, as
//!   five eager steps (the taps armed) and as one eager pass of five rows
//!   through the batch port (`Prompt38::Pass`), each from a reset with every
//!   selecting store's planes set to [`PLANE_FILL`] (`reset` leaves them, so
//!   a row a path reads without writing reads the pattern, not the previous
//!   run's value), leave the same last token, the same last logits and every
//!   store — the delta states and conv rings, the K/V planes, the raw and
//!   pooled indexer keys, the PLE ring — bit for bit; the graph and eager
//!   steps also every position's token and logits. Then five graph steps
//!   from `reset` after the pass's state leave the first run's everything:
//!   `reset` clears the recurrent state.
//! - (c) free-running on the batch set, each layer's picks read from the
//!   eager run's route taps (our own chain's routing, which is what a
//!   layer-by-layer run would read, so this gate has no layered clause),
//!   each tap's ids the top ten of its own logits ([`picks_its_top`]): a
//!   token whose chosen set differs from ik's `ffn_moe_topk-L` is a flip,
//!   allowed only when every exchanged pair's gap in ik's logits
//!   (`ffn_moe_logits`, whose softmax order is the pick's) lies within our
//!   two logits' error there and that error within [`FLIP_ERR_CAP`], a
//!   measured frontier ([`Flip::allowed`]), named and counted. A flip
//!   at layer `L'` and position `t'` lies on the path of every layer output
//!   from `L'` on at `t'` and at every later position; each layer output off
//!   every flip's path against ik's `l_out-L` within [`FREE_BAND`], the
//!   outputs past a flip printed and counted; the last position's argmax
//!   equal to ik's `result_output` argmax.
//! - (t) each step set: its prefill fed by our own steps from a reset, then
//!   the step; its argmax equal to ik's, or — named and counted — our argmax
//!   ik's runner-up, ik's own margin between the two inside twice the
//!   distance between our logits and ik's at those ids, and our whole
//!   logits row within [`FREE_BAND`] of ik's. The logits row is printed and
//!   bounds only a tie, as in the GLM gate: [`FREE_BAND`] bounds a layer
//!   output off every flip's path, and every step set's last position lies
//!   on a flip's path from its first layers (the router's top-10 margins
//!   over 512 sit under ik's q8_2 spacing). The step's layer outputs
//!   against ik's `l_out-L` are printed, and held to [`FREE_BAND`] on the two
//!   4-token sets only, below the first layer a flip lies on the path of the
//!   batch set's last position as (c) names them. D1K's step reads every
//!   pool (1,024 positions, 256 pools, under the 512 the selector keeps), so
//!   its selection list is the identity; D3K's reads a selection (751 pools
//!   past 512).
//! - (q) the pass in the selector region: D3K's prefill as eager passes of
//!   up to eight positions (`Prompt38::Pass`) from a reset with the planes
//!   filled, then the same
//!   step, leaves the step-fed run's step token, logits, layer outputs and
//!   every store bit for bit.
//! - (v) the verify: after a prefix of two steps, a verify of the first T
//!   of four rows ([`verify_rows`]; T = 2, 3, 4, `step_rows`, captured) — one
//!   row completing pool 0 — returns every row's argmax and logits bit for
//!   bit the steps' of those rows; for every k in 1..=T the commit of k rows
//!   (`rollback` to the first position not kept) leaves the lane word at
//!   `(c + k − 1) mod 4`, computed here from the lane before the verify, and
//!   the live stores bit for bit the k steps' — each delta layer's committed
//!   lane, the conv ring's slots of the eight positions before the count,
//!   the K/V planes and raw keys below it, the pooled plane's pools complete
//!   at the count, the PLE ring's slots of the fourteen positions before the
//!   count (a rejected row wrote position-indexed rows past the count —
//!   among them the pool it completed — which nothing reads until the row
//!   at that position rewrites them first) — and the step after the commit
//!   its token and logits; a second verify from a moved lane (4 rows after a
//!   commit of 3, lanes 2, 3, 0, 1) the same against its steps; an eager
//!   verify bit for bit the replay; and a deep verify in the selector
//!   region, after [`DEEP`] positions of D3K's prefill by passes, its row
//!   completing pool 513 rejected, the same against its steps — with every
//!   pooled row past the count set to NaN after the commit ([`POOL_POISON`]),
//!   the step after and the one after it (which completes pool 513 again)
//!   bit for bit the steps', raising nothing. Structure: each verify's graph holds
//!   the header's verify node count (1,235 on the real file), two a layer of them batch mem ops, one argmax,
//!   one `ds41_ffn_handoff_10_cols` a layer and no one-column handoff; a
//!   replay's host tier serves each layer once with one call into the host
//!   experts (48 services, 48 calls, 48 `Cols` services). Refusals: a step
//!   while a verify waits for its commit, and a commit with no verify
//!   waiting, by name; the lane word planted on a lane no call wrote makes
//!   the next step's first delta layer raise `delta_stamp` there, alone.
//! - (h) the head of m rows: from the prefix, a head of two rows over the
//!   loaded q8_0 lm_head, its input written from the host, reads back two
//!   equal tokens for two equal finite rows, and with a NaN in row 0's
//!   input, then in row 1's, reads back the head's `logit` fault each time,
//!   never a token (built here because the model's own head mix raises
//!   `hc_mix` on a row that is not finite first, so no call of the model
//!   plants one row alone); the word left raised, a verify of two
//!   rows is that fault and poisons the model.
//! - (o) one owner of the position: after the prefix, a graph step, an eager
//!   pass of three rows and a graph verify of two rows (kept whole), each
//!   with a failure planted before its launch (`Body38::plant_before_launch`)
//!   fail by name with the model at the prefix and not poisoned, and the
//!   same call again gives the clean call's token, logits, PLE rows
//!   (`Body38::ple_rows`) and every store bit for bit; each with a failure
//!   planted after its launch (`Body38::plant_after_launch`) fails by name
//!   with the model at the prefix, `Body38::kept` keeps less than the
//!   prefix (no delta state holds it any more), a step there is refused by
//!   name (the recurrent stores hold it already), and after `reset` the
//!   prefix and the same call give the clean bits.
//! - (g) the ubatch walk (`Prompt38::Gemm`), whose Q8_0 projections read q8
//!   activations and so are not the pass's bits, on the batch set from a
//!   reset in two arms. The free arm routes by its own router: its last
//!   argmax equal to ik's; its route taps each the top ten of their own
//!   logits; each flip against the pass's routing (the eager steps' taps)
//!   excused only while every pair's gap lies within our error and that
//!   error within six deviations of the error model's router error
//!   ([`flip_cap`]), and — where no earlier flip lies on its path — only at
//!   a gap within six deviations of the router's error over the logits'
//!   spread ([`margin_cap`]); its logits and stores printed. The forced arm
//!   takes the pass's routes (`Body38::plant_ubatch_routes`), so no flip
//!   can happen: every id the pass's, its last argmax equal to ik's, the
//!   last logits and every layer's live store and the PLE ring within the
//!   error model's band ([`gemm_band`]: the q8 step per 32 values at the
//!   largest crest a block can have, `rounding::q8_32_rel`, four
//!   projections a layer, √(l + 1) over the layers), no layer excused; the
//!   router's own logits against the pass's printed; `auto` at five
//!   positions the pass's bits. Around the free arm:
//!   a read of the route taps past the walk's positions refused by name; a
//!   reset leaves no split; a walk past the armed taps' rows refused by name
//!   before it moves anything, the same call with the taps off then the
//!   free walk's bits. On D3K's prefill: one ubatch and two cut at
//!   [`SPLIT`] (inside the selecting rows, mid-pool, off the eight-row
//!   runs, past the router's first run) leave the last logits and every
//!   store bit for bit — a token's bits do not depend on the ubatch it
//!   lands in; the walk's cut of the selecting layers' rows between the
//!   prefill flash and the selection is the rule's (2,051 and 949, from
//!   KEPT and POOL — a record, checked against the gate's own derivation);
//!   the one ubatch again with the walk's timing armed
//!   (`GpuModel::set_prompt38_stats`) writes the same bits, and its record
//!   holds one ubatch and a row for every layer with every card span read;
//!   the step after the ubatch as (t) holds a step set. A slot map with a
//!   routed expert on a tier card (planted, `Body38::plant_slot_map`) is
//!   refused by name — the walk, the layer and the count — by each of the
//!   four walks (a graph step, an eager pass, a ubatch, a graph verify),
//!   before anything moves, the position kept and the model not poisoned,
//!   and the same call with the plant taken back runs; `auto` is the pass
//!   below `Prompt38::GEMM_FROM` positions and the ubatch from it.
//! - (k) the card leg: after every clause above, the host plan's model
//!   dropped, the card rule's plan (`place::Experts::Card`: each eligible
//!   layer's id prefix on the card) loaded on the same card. Its step graph
//!   holds the header's decode nodes plus the leg's five a card layer (1,391 over
//!   the real file's 48), over the plan's card layers, the
//!   program's count. The pass's places entry, on synthetic ids at the
//!   router's pitch, writes each slot's place, [`HOST`] and `expert_id` for
//!   an id past the experts, and reads no shared expert's word
//!   ([`places_rule`]). (i) The batch set's five eager steps against the
//!   host plan's eager steps of (p): each flip of the routing excused as
//!   (g)'s free arm excuses one; off every flip's path each layer output
//!   within [`card_band`] and each position's logits within the last
//!   layer's band, its argmax the host plan's or a named tie; on a flip's
//!   path printed. (ii) Its
//!   eager steps the graph steps' bits, one pass of the five rows the steps'
//!   last token, logits and every store bit for bit, and verifies of 2, 3
//!   and 4 rows kept whole every row bit for bit the steps' (each graph
//!   holds the verify's nodes plus the same five a card layer). (iii) The ubatch walk — the card route's
//!   — with the pass's routes planted against the pass of the same plan
//!   within [`gemm_band`] (the forced arm's shape: the card route
//!   quantizes the same blocks the pass's card leg quantizes), and the
//!   batch set and D3K's prefill each as two cut ubatches against one, bit
//!   for bit — a token's bits depend neither on the ubatch it lands in nor
//!   on the route's run (D3K's 3,001 tokens run two at the load's ubatch
//!   of 4,096).
//! - (r) refusals: `Prompt38::parse` takes `step`, `pass`, `gemm` and `auto`
//!   and refuses any other name by name; the image placeholder
//!   the header's image placeholder (`ple.image_token_id`) as a step, as a pass and as a ubatch is refused by name
//!   with the position kept and the model not poisoned; a prompt past the
//!   stores is refused by name before any launch, on every path, the
//!   position kept.
//! - (y) resident slots: a load made with a plan of two sequences
//!   (`PlanInputs::plan_with_slots` — its per-load sequence terms counted
//!   twice, and at one slot the plan every other clause loads by) answers
//!   the slot harness's contracts (`slots_gate`: H1 interleave, H2 one pass,
//!   H3 bytes, H4 reset, H6 refusals, H7 captures), stream 0 window A (four
//!   ids) by the
//!   pass path and stream 1 window B (D1K's prefill) by the ubatch path, its
//!   state hash every per-sequence store, the PLE ring and the lane word,
//!   `seq_bytes`' derivation the stores, the lane word and the two arena-row
//!   buffers; and between the harness's halves the body's own: a third
//!   `add_slots` past the plan is refused by name; a select with a verify
//!   waiting for its commit is refused by name, moving nothing; and a
//!   rollback of slot 1 (a verify's commit) leaves slot 0's continuation its
//!   solo run's.
//! - (z) the PLE table on the NVMe tier: the host plan made with the host's
//!   room given ([`PlanInputs::room`]) as the host arm's need less one byte,
//!   so the placement leaves the table on the NVMe tier with its row room
//!   set aside, against the same plan at the reading's room, the table on
//!   the host, both loaded at a ubatch of [`ZUB`]: each plan line names its
//!   tier, and the NVMe arm's batch set as graph steps and as one eager
//!   pass, and D3K's prefill as one call of three ubatches (its later two
//!   read ahead) with the step after it, are the host arm's bit for bit —
//!   every token, logit and store: the same rows decoded, only where they
//!   are read from moves. The step thread's faults across each arm's graph
//!   steps are printed.
//!
//! Named differences, not banded away: ik combines the block as
//! `routed + σ(g)·shared`, ours as `hsum + shared·w` with the sigmoid weight
//! the router's eleventh slot — one f32 rounding of the update; ik's CPU
//! projections quantize their input to q8_2 per 32 values, ours read the f32
//! row; at D3K's position (3,001 keys, 751 pools, the last one of one key)
//! ik cuts the selection by cells and keeps up to three keys of the 513th
//! pool that ours, which keeps 512 whole pools and the tail, does not (the
//! difference q38sel measured), its logits printed.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen4exp_e2e: built without the `gpu` feature; see `just gate-gpu-qwen4exp-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the compare, tap, layer-table and flip-accounting helpers; the clause reporting, the refusal helpers, the selector reader and the forced arm's accumulators serve the other gates"
)]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[path = "shared/q38_arch.rs"]
mod q38_arch;

#[cfg(feature = "gpu")]
mod gate {
    use std::time::Instant;

    use crate::e2e::{
        argmax, first_flip_layers, flips_report, ik_last, layer_rels, layer_table, print_layers,
        rel, same_bits, second, set_open, tap, tie_numbers, worst_off_path,
    };

    use bloomery_gpu::arch::qwen3moe::{
        Body38, LayerKind38, Prompt38, Qwen38Model, RouteTap, Store38Host,
    };
    use bloomery_gpu::head::Head;
    use bloomery_gpu::host::batch::HOT_COLS;
    use bloomery_gpu::host::handoff::{HandoffKernels, Places};
    use bloomery_gpu::hybrid::{HOST, Slot, SlotMap};
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::{Fault, FaultSite, GpuError, LAYER_HEAD};
    use bloomery_gpu_gates::flip::{self, Flip};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::q38_fixture::{self as q38, Shape};
    use bloomery_gpu_gates::rounding::q8_32_rel;
    use bloomery_gpu_gates::slots_gate::{self, Derived, Launches, PassAdapter, SlotsAdapter};
    use bloomery_gpu_gates::tier::Tag;
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, RefManifest, checks_failed, data_dir, ref_model_path,
        topk_ids_logical_within, verdict,
    };
    use bloomery_levers::{CARD_BUDGET, HostCfg};
    use cuda_core::{DeviceBuffer, sys};
    use gguf::Split;
    use gguf::quant::half_to_f32;
    use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
    use model::placement::workstation::{HostNeed, HostRead};
    use model::placement::{Device, PlanLevers};
    use refset::arch::qwen4exp::{BATCH, D1K, D3K, IK, MODEL, STEP4, STEP4_EVERY_NODE};

    /// Cache rows: D3K's step at position 3,000, with room.
    const CTX: usize = 3072;

    /// The file's widths, as the header states them (qwen4arch-design §1) and
    /// the engine's kernels fix them (`plan38.rs` refuses a file that differs,
    /// the fixture keeps them): what the derivations below are written
    /// against. The layer counts are not here: they come from the header
    /// ([`shape`]).
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const N_EXPERT: usize = 512;
    const N_USED: usize = 10;
    /// Delta layers: 48 value heads and 16 key heads of 128, a conv of 4
    /// taps; selecting layers: 2 K/V heads of 256, indexer keys of 128, pools
    /// of 4; the PLE conv: 4 taps at dilation 3 over the four streams.
    const V_HEADS: usize = 48;
    const K_HEADS: usize = 16;
    const HEAD_V: usize = 128;
    const CONV: usize = 4;
    const N_KV: usize = 2;
    const HEAD: usize = 256;
    const IDX_DIM: usize = 128;
    const POOL: usize = 4;
    const PLE_TAPS: usize = 4;
    const PLE_DILATION: usize = 3;
    /// The most positions an eager pass or a ring's tail takes.
    const PASS_ROWS: usize = 8;
    /// The real file's image placeholder, `ple.image_token_id`: the token its
    /// PLE hash refuses as an input.
    const REAL_IMAGE_TOKEN: u32 = 248_056;

    /// The f16 pattern every selecting store's planes hold before a run
    /// (100.0): `reset` leaves those rows as they were, and a path that reads
    /// a row it did not write must read this, not a previous run's value — a
    /// finite value, so a correct path's masked reads stay finite, and far
    /// from any key, value or indexer key the model writes.
    const PLANE_FILL: u16 = 0x5640;

    /// The real file's layer counts and the captured graphs' node counts: the
    /// literals the gate carried before it derived them from the header
    /// ([`shape`], `q38_fixture::Shape`), kept as the witnesses the real tier
    /// prints each derived value beside ([`witnesses`]) and requires equal.
    /// Each node count is the pin's derivation, one launch at a time:
    ///
    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: the embedding row; each layer's two mixes, three
    /// launches each (the grouped norm with the down projection, the up, the
    /// gated mean — the combine folded into the next mix); a delta layer's
    /// 7 mixer launches (q·k·v, z, β·α, the conv, the delta step, the gated
    /// norm, the output projection) and a selecting layer's 14 (q, k, v, the
    /// indexer key's projection, its append and the pool, the indexer query,
    /// the selection's two, the q/k norm, turn and append, the selected
    /// flash's two, the out gate, the output projection); each block's 9 (the
    /// router, the handoff, the go, the shared expert's four, the wait, the
    /// gated sum); the PLE site's 5 on layer 1 (the combine, key and value,
    /// gate, conv); the head's 5 (its mix's three, the q8_0 gemv, the argmax):
    /// 1 + 36·22 + 12·29 + 5 + 5.
    const REAL_N_LAYER: usize = 48;
    const REAL_PLE_LAYER: usize = 1;
    const REAL_N_QSA: usize = 12;
    const REAL_N_GDN: usize = 36;
    const REAL_NODES_DECODE: usize = 1151;

    /// PIN(2026-09-27): each layer's go and wait.
    const REAL_MEMOPS: usize = 2 * REAL_N_LAYER;

    /// PIN(2026-09-28): the captured verify's node count at 2, 3 and 4 rows,
    /// derived before the chain was built: the step's [`REAL_NODES_DECODE`],
    /// plus at more than one row each delta layer's two token-major copies (β
    /// and α out of the joined projection) and each selecting layer's one (the
    /// indexer queries); the handoff (one launch writing every row's image),
    /// the go, the wait and the head (its mix's three, one q8_0 gemv over
    /// every row, one argmax) are one each whatever the rows:
    /// 1151 + 36·2 + 12·1.
    const REAL_NODES_VERIFY: usize = 1235;

    /// PIN(2026-10-05): the nodes one more busy slot adds to a captured pass
    /// of several slots' rows, derived before the pass was built: the
    /// launches bound to one sequence — the embedding row, the PLE site's
    /// conv on layer 1, each delta layer's conv and delta step, each
    /// selecting layer's 8 (the indexer key's projection, its append and the
    /// pool, the selection's two, the q/k norm, turn and append, the
    /// selected flash's two) — and the copy of a parked slot's rows into its
    /// own: 1 + 1 + 36·2 + 12·8 + 1.
    const REAL_NODES_SLOT: usize = 171;

    /// PIN(2026-09-29): the layers the gate card's card plan puts routed
    /// experts on, derived before the leg was built: the 48 less the five
    /// whose routed stacks the card experts do not read (`place::host_only`:
    /// layer 2's q5_K stacks, the q8_0 downs of 640-value rows of layers 4,
    /// 30, 46 and 47), each of the 43 holding over a hundred experts on the
    /// 3090 at this cache (the card rule's plans, `qwen4exp_meta`'s
    /// `CARD_PLANS`: 105/104 at 4,096 positions and a ubatch of 4,096, the
    /// ubatch walk's card route in the budget).
    /// PIN(2026-10-01): all 48: the card experts read layer 2's q5_K gate and
    /// up and the q8_0 downs (`kq_gate_up_act_q5k`, `q8_0_gemv_sel32`), so
    /// `place::host_only` is empty and every layer holds its prefix (93/92 on
    /// the 3090 at 4,096 positions and a ubatch of 4,096, `CARD_PLANS`).
    const REAL_CARD_LAYERS: usize = 48;

    /// PIN(2026-09-29): the captured step's and verify's node counts under
    /// the card plan, derived before the leg was built: each card layer's
    /// shadow adds the leg's five launches (the normed rows' q8_1, the
    /// gate·up, the q8_1 of the card slots' columns, the down, the card sum),
    /// and its back combines the card sum in the gated sum's launch:
    /// 1151 + 5·43 and 1235 + 5·43.
    /// PIN(2026-10-01): over the 48 card layers, the same five launches on
    /// each whatever its types: 1151 + 5·48 and 1235 + 5·48.
    const REAL_NODES_DECODE_CARD: usize = 1391;
    const REAL_NODES_VERIFY_CARD: usize = 1475;

    /// The gate's one reading of what the run is made from: the model file
    /// (`ref_model_path`, a whole fixture under the fixture tier), the host
    /// load config and the plan's levers (the budget the header records in
    /// the fixture tier, [`q38::plan_levers`]), and the layer kinds the
    /// header names. Set once, first thing in `run`, before any load.
    struct Cfg {
        path: std::path::PathBuf,
        host: HostCfg,
        plan_levers: PlanLevers,
        shape: Shape,
        n_expert: usize,
    }

    static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();

    fn cfg() -> &'static Cfg {
        CFG.get().expect("init runs before any clause")
    }

    /// The layer kinds the header names ([`q38::Shape`]): never read off the
    /// loaded body, which the structure clause holds against them.
    fn shape() -> &'static Shape {
        &cfg().shape
    }

    fn n_layer() -> usize {
        shape().n_layer
    }

    fn n_qsa() -> usize {
        shape().n_qsa()
    }

    fn n_gdn() -> usize {
        shape().n_gdn()
    }

    fn nodes_decode() -> usize {
        shape().nodes_decode()
    }

    fn memops() -> usize {
        shape().memops()
    }

    fn nodes_verify() -> usize {
        shape().nodes_verify()
    }

    /// The PLE site's layer: the band of the PLE ring is the error model's
    /// at the layer that writes it.
    fn ple_layer() -> Result<usize, GateError> {
        shape()
            .ple_layer
            .ok_or_else(|| "the header lists no PLE site (ple.layers)".into())
    }

    /// The model file this gate runs on, opened as a split of architecture
    /// `qwen4exp`.
    fn open_split() -> Result<Split, GateError> {
        let path = &cfg().path;
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!(
                "{} is {:?}, not qwen4exp",
                path.display(),
                file.architecture()
            )
            .into());
        }
        Ok(file)
    }

    /// Read the file's header once: the path the tier names, the levers, the
    /// plan's budget and the layer kinds; print each derived value beside
    /// the real file's literal it replaced ([`witnesses`]).
    fn init() -> Result<bool, GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&file)?;
        let shape = crate::q38_arch::shape_of(&inputs.hp)?;
        let plan_levers = q38::plan_levers(&file, &levers, 0)?;
        let n_expert = inputs.hp.n_expert;
        let cfg = Cfg {
            path,
            host: levers.host(),
            plan_levers,
            shape,
            n_expert,
        };
        if CFG.set(cfg).is_err() {
            return Err("the gate's configuration was read twice".into());
        }
        witnesses()
    }

    /// The move proof: each value the header gives, beside the literal the
    /// real file's gate carried. In the real tier they are equal; the
    /// fixture tier prints the fixture's own.
    fn witnesses() -> Result<bool, GateError> {
        let s = shape();
        let model = cfg().path.to_string_lossy().into_owned();
        let mut ok = q38::witness("model file", model, MODEL.to_string());
        ok &= q38::witness("N_LAYER", s.n_layer, REAL_N_LAYER);
        ok &= q38::witness("PLE_LAYER", s.ple_layer.unwrap_or(0), REAL_PLE_LAYER);
        ok &= q38::witness("N_QSA", s.n_qsa(), REAL_N_QSA);
        ok &= q38::witness("N_GDN", s.n_gdn(), REAL_N_GDN);
        ok &= q38::witness("NODES_DECODE", s.nodes_decode(), REAL_NODES_DECODE);
        ok &= q38::witness("MEMOPS", s.memops(), REAL_MEMOPS);
        ok &= q38::witness("NODES_VERIFY", s.nodes_verify(), REAL_NODES_VERIFY);
        ok &= q38::witness("NODES_SLOT", s.nodes_slot(), REAL_NODES_SLOT);
        ok &= q38::witness("IMAGE_TOKEN", s.image_token.unwrap_or(0), REAL_IMAGE_TOKEN);
        ok &= q38::witness(
            "plan levers' card budget",
            cfg().plan_levers.card_budget_bytes,
            PlanLevers::default().card_budget_bytes,
        );
        ok &= q38::witness_card(&card()?);
        Ok(ok)
    }

    /// The move proof of the card plan's counts, printed when the card plan
    /// is loaded: the card layers the plan holds routed experts on, and the
    /// node counts under them.
    fn card_witnesses(card_layers: usize) -> bool {
        let s = shape();
        let mut ok = q38::witness("CARD_LAYERS", card_layers, REAL_CARD_LAYERS);
        ok &= q38::witness(
            "NODES_DECODE_CARD",
            s.nodes_decode_card(card_layers),
            REAL_NODES_DECODE_CARD,
        );
        ok &= q38::witness(
            "NODES_VERIFY_CARD",
            s.nodes_verify_card(card_layers),
            REAL_NODES_VERIFY_CARD,
        );
        ok
    }

    /// Lanes of a delta layer's state, a verify's rows at most.
    const LANES: usize = 4;

    /// Rows of the conv ring and of the PLE ring: the reach of the conv and
    /// a pass.
    const CONV_RING: usize = CONV - 1 + PASS_ROWS;
    const PLE_RING: usize = (PLE_TAPS - 1) * PLE_DILATION + PASS_ROWS;

    /// Positions of the verify's prefix: the verify starts at position 2,
    /// so its row 1 (position 3, count 4) completes pool 0.
    const PREFIX: usize = 2;

    /// Positions of the deep verify's prefix: its rows (counts 2,053 to
    /// 2,056) select, each seeing more complete pools than the 512 a select
    /// keeps, and its row 3 completes pool 513; kept to [`DEEP_KEPT`] rows,
    /// the step after (count 2,055) selects among pools 0 to 512 only, and
    /// the step after that (count 2,056) completes pool 513 again before its
    /// select reads it.
    const DEEP: usize = 2052;
    const DEEP_KEPT: usize = 2;

    /// An f16 NaN: the deep verify's pooled rows past the count after its
    /// commit, so a select that scores one raises `pool_select` by name.
    const POOL_POISON: u16 = 0x7e00;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's, off every flip's path. Borrowed, not measured on
    /// this model: GLM-5.3's gate's derivation (√45 · 1.415e-2, its forced
    /// arm's worst layer error on the streams on ik's inputs) over this
    /// model's 48 layers, √48 · 1.415e-2 ≈ 0.098, rounded up. GLM is the
    /// analog and Qwen3.6 (0.26) is not: at every q8_0 projection our side
    /// reads the f32 row, so the gap is ik's q8_2 activation alone, as in
    /// GLM, where Qwen3.6's inputs are quantized on both sides; the host
    /// experts run the q8_2_x4 rule on both. This gate has no forced arm to
    /// re-derive it on Qwen3.8; (c) prints every layer's distance, the input
    /// a forced arm would take.
    const FREE_BAND: f64 = 0.10;

    /// PIN(2026-09-28): [잠정 — 백로그] the most error, in router logit units, our
    /// two logits may carry at an excused flip — a measured frontier, not a derivation.
    /// In the batch set's free run the clean chain's largest pair error
    /// was 0.942 (gap 0.483); the PLE site skipped (m06) reached 4.69, the other broken
    /// chains 20.15 and 20.66; 2.0 sits 2.1x above clean and 2.3x below m06. The route tap
    /// reading the next layer's logits (m14, 6.99) is left out: that error is the tap's,
    /// not the router's. A flip-aware bound from a Qwen3.8 forced arm replaces it.
    const FLIP_ERR_CAP: f64 = 2.0;

    /// The selecting layers the header's interval names (`(il + 1) % interval
    /// == 0`): every fourth on the real file, every second on the fixture.
    fn qsa_layers() -> Vec<usize> {
        shape().qsa.clone()
    }

    /// The stores' bytes at [`CTX`] rows, derived from the header: each
    /// delta layer's state, 48 heads of 128 × 128 f32, and conv ring,
    /// `CONV − 1 + PASS_ROWS` rows of the conv's `2·16·128 + 48·128`
    /// channels in f32; each selecting layer's K and V rows, `2 · 2 · 256`
    /// f16 a position, its raw indexer key, 128 f16 a position, and its
    /// pooled key, 128 f16 a pool of 4 (the last pool counted whole); the
    /// PLE ring, `(taps − 1)·dilation + PASS_ROWS` rows of the four streams
    /// in f32. PIN(2026-09-28): a delta layer's state is [`LANES`] lanes, a
    /// verify's rows each keeping its own, with a u32 stamp a lane.
    fn store_bytes() -> usize {
        let conv_ch = 2 * K_HEADS * HEAD_V + V_HEADS * HEAD_V;
        let rec = LANES * (V_HEADS * HEAD_V * HEAD_V * 4 + 4) + CONV_RING * conv_ch * 4;
        let sel = CTX * (2 * N_KV * HEAD * 2 + IDX_DIM * 2) + CTX.div_ceil(POOL) * IDX_DIM * 2;
        let ple = ((PLE_TAPS - 1) * PLE_DILATION + PASS_ROWS) * STREAMS * HIDDEN * 4;
        n_gdn() * rec + n_qsa() * sel + ple
    }

    /// Whether our last argmax `top` is ik's `result_output` argmax, and
    /// ik's as the line prints it: an oracle clause, which the fixture tier
    /// leaves to the real one (true, and `not compared`).
    fn ik_argmax(
        man: &RefManifest,
        vocab: usize,
        clause: &str,
        top: u32,
    ) -> Result<(bool, String), GateError> {
        if !q38::clause(&format!("{clause} against ik's"), Tag::Oracle)? {
            return Ok((true, "not compared in this tier".to_string()));
        }
        let ik_top = argmax(&ik_last(man, vocab)?);
        Ok((top == ik_top, ik_top.to_string()))
    }

    /// What a load made: the model, the tier its plan reads the PLE table
    /// from, and what the plan holds on the card.
    struct Opened {
        model: Qwen38Model,
        ple_tier: Option<Device>,
        card: CardPlan,
    }

    /// The plan's routed experts on the card: the layers that hold any, and
    /// each layer's count (`Plan::n_l`) — the plan's own, read before the
    /// body is built, so the structure clause holds the body to it.
    struct CardPlan {
        layers: usize,
        n_l: Vec<u64>,
    }

    /// The card a plan is made on: the real tier's the gate runner's
    /// (the 3090's bytes on the card in view), the fixture tier's `a`
    /// (`q38_fixture::card`).
    fn card() -> Result<model::placement::workstation::CardSpec, GateError> {
        q38::card(crate::gate_card::card)
    }

    /// The model placed on the gate card by its plan, the routed experts
    /// where `experts` says under the placement's levers ([`Cfg`]).
    fn open(experts: Experts) -> Result<Qwen38Model, GateError> {
        open_slots(experts, 1)
    }

    /// [`open`] for a load that serves `slots` resident sequences
    /// ([`Body38::open_placed_slots`]): the plan counts them
    /// ([`PlanInputs::plan_with_slots`]).
    fn open_slots(experts: Experts, slots: usize) -> Result<Qwen38Model, GateError> {
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        Ok(open_at(experts, slots, ub, None)?.model)
    }

    /// [`open_slots`] at a ubatch of `ub`, the host's room the reading's or
    /// `room` given ([`PlanInputs::room`]); with the tier the plan reads the
    /// PLE table from, which the plan line names.
    fn open_at(
        experts: Experts,
        slots: usize,
        ub: usize,
        room: Option<u64>,
    ) -> Result<Opened, GateError> {
        let file = open_split()?;
        let t = Instant::now();
        // `describe`, not `read`: `read` refuses the chat surface's two
        // items too; `open_placed` refuses what `ALLOWED` does not name.
        let mut inputs = PlanInputs::describe(&file)?;
        if let Some(r) = room {
            inputs.room = (r, HostRead::Given);
        }
        let card = card()?;
        let machine =
            machine_for_experts(card, inputs.spec.layers.len(), u64::try_from(ub)?, experts);
        let plan =
            inputs.plan_with_slots(&machine, CTX as u64, &cfg().plan_levers, experts, slots)?;
        let held = plan.n_l.iter().filter(|&&n| n > 0).count();
        let tier = plan.row_tier()?;
        println!(
            "plan card={} experts={experts:?} slots={slots} ctx_max={} ubatch={ub} host_experts={} \
             card_experts={} card_layers={held} ple={} room={} read={} row_reserve={} \
             card_budget={:?}",
            card.name,
            plan.ctx_max,
            plan.host.experts,
            plan.cards[0].experts,
            tier_word(tier),
            inputs.room.0,
            inputs.room.1.word(),
            plan.host.row_reserve_bytes,
            cfg().plan_levers.card_budget_bytes
        );
        let n_l = plan.n_l.clone();
        let mut m = Body38::open_placed_slots(file, &plan, &inputs, 0, cfg().host, ub, slots)?;
        m.set_mode(StepMode::Graph);
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok(Opened {
            model: m,
            ple_tier: tier,
            card: CardPlan { layers: held, n_l },
        })
    }

    /// A plan's PLE tier as the plan line prints it.
    fn tier_word(tier: Option<Device>) -> &'static str {
        match tier {
            Some(Device::Host) => "host",
            Some(Device::Nvme) => "nvme",
            Some(Device::Card(_)) => "card",
            Some(Device::Unused) => "unused",
            None => "none",
        }
    }

    // ------------------------------------------------------ (s) structure

    fn structure(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let qsa: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l] == LayerKind38::Qsa)
            .collect();
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let mut ok = kinds.len() == n_layer() && qsa == qsa_layers() && stores == want_stores;
        println!(
            "structure layers={} selecting at {qsa:?}; store bytes {stores} (want {want_stores}, \
             derived) {}",
            kinds.len(),
            verdict(ok)
        );
        let (counted, memops_counted) = body.step_launches();
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let (want_nodes, want_memops) = (nodes_decode(), memops());
        let pass = nodes == want_nodes
            && counted == want_nodes
            && memops_counted == want_memops
            && b == want_memops
            && k == want_nodes - want_memops
            && other == 0;
        println!(
            "structure decode graph_nodes={nodes} (want {want_nodes}; the program counts \
             {counted}, {memops_counted} of them batch_mem_op) kernel={k} batch_mem_op={b} (want \
             {want_memops}) other={other} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    // ------------------------------------------ (p) one program, (c) free

    /// What a run of tokens left: each position's argmax and logits (only
    /// the last of each for a pass), with the taps armed each position's
    /// layer outputs and route, and every store with the PLE ring.
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
        routes: Vec<Vec<RouteTap>>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
    }

    /// Arm or disarm the layer and route taps.
    fn set_taps(m: &mut Qwen38Model, on: bool) -> Result<(), GateError> {
        m.set_layer_taps(on)?;
        Ok(())
    }

    /// `reset`, then every selecting store's planes set to [`PLANE_FILL`].
    fn fresh(m: &mut Qwen38Model) -> Result<(), GateError> {
        m.reset()?;
        let (gpu, _, b) = m.body_parts("fresh")?;
        b.fill_planes(gpu, PLANE_FILL)?;
        Ok(())
    }

    /// Every store and the PLE ring, read back.
    fn stores(m: &mut Qwen38Model) -> Result<(Vec<Store38Host>, Vec<f32>), GateError> {
        let (gpu, _, b) = m.body_parts("stores")?;
        Ok(b.stores_host(gpu)?)
    }

    /// `toks` as steps from [`fresh`] (without `reset`: from where the
    /// model stands), eager steps with the taps armed.
    fn run_steps(
        m: &mut Qwen38Model,
        toks: &[u32],
        mode: StepMode,
        reset: bool,
    ) -> Result<Run, GateError> {
        if reset {
            fresh(m)?;
        }
        m.set_mode(mode);
        let taps_on = mode == StepMode::Eager;
        set_taps(m, taps_on)?;
        let mut r = Run {
            tokens: Vec::new(),
            logits: Vec::new(),
            taps: Vec::new(),
            routes: Vec::new(),
            stores: Vec::new(),
            ple_ring: Vec::new(),
        };
        for &t in toks {
            r.tokens.push(m.step(&[t])?);
            r.logits.push(m.logits()?);
            if taps_on {
                let (gpu, _, b) = m.body_parts("run_steps")?;
                r.taps.push(b.taps(gpu)?);
                r.routes.push(b.route_taps(gpu)?);
            }
        }
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        (r.stores, r.ple_ring) = stores(m)?;
        Ok(r)
    }

    /// `toks` as one prompt call by eager passes from [`fresh`]: the last
    /// token and logits, and every store.
    fn run_pass(m: &mut Qwen38Model, toks: &[u32]) -> Result<Run, GateError> {
        fresh(m)?;
        let last = m.prompt38(toks, Prompt38::Pass)?;
        let logits = m.logits()?;
        let (stores, ple_ring) = stores(m)?;
        Ok(Run {
            tokens: vec![last],
            logits: vec![logits],
            taps: Vec::new(),
            routes: Vec::new(),
            stores,
            ple_ring,
        })
    }

    /// The layers whose stores differ, and whether the PLE rings do.
    fn store_diff(a: &Run, b: &Run) -> (Vec<usize>, bool) {
        let n = a.stores.len().max(b.stores.len());
        let layers = (0..n)
            .filter(|&l| match (a.stores.get(l), b.stores.get(l)) {
                (Some(x), Some(y)) => !x.same_bits(y),
                _ => true,
            })
            .collect();
        (layers, !same_bits(&a.ple_ring, &b.ple_ring))
    }

    /// `got` against `want`: the tokens and logits (every position, or the
    /// last only) and every store, bit for bit.
    fn same_run(label: &str, got: &Run, want: &Run, last_only: bool) -> bool {
        let (tokens, logits) = if last_only {
            (
                got.tokens.last() == want.tokens.last() && !got.tokens.is_empty(),
                matches!((got.logits.last(), want.logits.last()), (Some(a), Some(b)) if same_bits(a, b)),
            )
        } else {
            (
                got.tokens == want.tokens,
                got.logits.len() == want.logits.len()
                    && got
                        .logits
                        .iter()
                        .zip(&want.logits)
                        .all(|(a, b)| same_bits(a, b)),
            )
        };
        let (layers, ring) = store_diff(got, want);
        let ok = tokens && logits && layers.is_empty() && !ring;
        println!(
            "paths {label}: tokens {:?} vs {:?}; {} logits bit for bit: {logits}; stores \
             differing at layers {layers:?}, PLE ring differing: {ring} {}",
            got.tokens,
            want.tokens,
            if last_only {
                "the last"
            } else {
                "every position's"
            },
            verdict(ok)
        );
        ok
    }

    /// (p): the graph steps, the eager steps (returned, taps armed), the
    /// pass, and the steps again from `reset` alone.
    fn paths(m: &mut Qwen38Model, toks: &[u32]) -> Result<(bool, Run), GateError> {
        let graph = run_steps(m, toks, StepMode::Graph, true)?;
        let eager = run_steps(m, toks, StepMode::Eager, true)?;
        let mut ok = same_run(
            "five eager steps vs five graph replays",
            &eager,
            &graph,
            false,
        );
        let pass = run_pass(m, toks)?;
        ok &= same_run(
            "one pass of five rows vs five graph steps",
            &pass,
            &graph,
            true,
        );
        // After the pass's state: `reset` must clear it.
        let again = run_steps(m, toks, StepMode::Graph, true)?;
        ok &= same_run(
            "after the pass's state and a reset, the same five steps",
            &again,
            &graph,
            false,
        );
        Ok((ok, eager))
    }

    /// ik's routing of one layer over the set's tokens: its logits
    /// ([`N_EXPERT`] a token; the pick ranks their softmax, the same order)
    /// and its chosen ids ([`N_USED`] a token).
    struct IkRoute {
        logits: Vec<f32>,
        ids: Vec<i32>,
    }

    impl IkRoute {
        fn read(man: &RefManifest, l: usize) -> Result<IkRoute, GateError> {
            let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
            let row = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, row, N_EXPERT as u32)?;
            if logits.len() % N_EXPERT != 0 || ids.len() * N_EXPERT != logits.len() * N_USED {
                return Err(format!(
                    "layer {l}: {} logits and {} ids; a token takes {N_EXPERT} and {N_USED}",
                    logits.len(),
                    ids.len()
                )
                .into());
            }
            Ok(IkRoute { logits, ids })
        }

        fn tokens(&self) -> usize {
            self.ids.len() / N_USED
        }

        /// Token `t`'s logits and ids.
        fn at(&self, t: usize) -> (&[f32], &[i32]) {
            (
                &self.logits[t * N_EXPERT..(t + 1) * N_EXPERT],
                &self.ids[t * N_USED..(t + 1) * N_USED],
            )
        }

        /// ik's own margin at token `t`: its tenth pick's logit less the
        /// best it left.
        fn margin(&self, t: usize) -> f64 {
            let (v, ids) = self.at(t);
            flip::margin(v, ids)
        }
    }

    /// The flip at layer `l`, token `t`, if our chosen set is not ik's.
    fn flip_at(l: usize, t: usize, ours: &RouteTap, ik: &IkRoute) -> Option<Flip> {
        let (iv, ids) = ik.at(t);
        Flip::between((l, t), (&ours.ids, &ours.logits), (ids, iv), ik.margin(t))
    }

    /// Whether a route tap's ids are [`N_USED`] distinct experts under
    /// [`N_EXPERT`] and the top of its own logits: none it left above the
    /// lowest it kept (ties either way). A tap of another layer's logits
    /// beside this layer's ids does not hold it.
    fn picks_its_top(r: &RouteTap) -> bool {
        let mut ids = r.ids.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != N_USED
            || r.logits.len() != N_EXPERT
            || ids.iter().any(|&e| e as usize >= N_EXPERT)
        {
            return false;
        }
        let kept = |e: usize| ids.binary_search(&(e as u32)).is_ok();
        let min_in = ids
            .iter()
            .map(|&e| r.logits[e as usize])
            .fold(f32::INFINITY, f32::min);
        (0..N_EXPERT)
            .filter(|&e| !kept(e))
            .all(|e| r.logits[e] <= min_in)
    }

    /// The free clause's verdict, and by position the first layer a flip
    /// lies on the path of ([`n_layer`] where none does). Every route tap
    /// the top ten of its own logits is a self-consistency clause; the
    /// layer outputs against ik's, the flips against ik's routing and the
    /// last argmax against ik's are the oracle's, and the fixture tier
    /// leaves them to the real one (no flip is then on any path).
    fn free(man: &RefManifest, eager: &Run, vocab: usize) -> Result<(bool, Vec<usize>), GateError> {
        let mut inconsistent = Vec::new();
        for (t, routes) in eager.routes.iter().enumerate() {
            for (l, r) in routes.iter().enumerate() {
                if !picks_its_top(r) {
                    inconsistent.push((l, t));
                }
            }
        }
        println!(
            "free: route taps whose ids are not the top {N_USED} of their own logits (distinct, \
             under {N_EXPERT}): {} {:?} {}",
            inconsistent.len(),
            &inconsistent[..inconsistent.len().min(8)],
            verdict(inconsistent.is_empty())
        );
        if !q38::clause(
            "(c) free: each layer's output, each route and the last argmax against ik's",
            Tag::Oracle,
        )? {
            return Ok((inconsistent.is_empty(), vec![n_layer(); eager.taps.len()]));
        }
        let table = layer_table(man, &eager.taps, STREAMS * HIDDEN, n_layer())?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!("free table layer={l} l_out_rel by tap {}", cells.join(" "));
        }
        let mut flips = Vec::new();
        for l in 0..n_layer() {
            let ik = IkRoute::read(man, l)?;
            if ik.tokens() != eager.routes.len() {
                return Err(format!(
                    "layer {l}: ik routes {} tokens, the run {}",
                    ik.tokens(),
                    eager.routes.len()
                )
                .into());
            }
            let margins: Vec<String> = (0..ik.tokens())
                .map(|t| format!("{:.2e}", ik.margin(t)))
                .collect();
            println!("ik margin layer={l} by token {}", margins.join(" "));
            for (t, routes) in eager.routes.iter().enumerate() {
                let ours = routes
                    .get(l)
                    .ok_or_else(|| format!("position {t}: no route tap for layer {l}"))?;
                flips.extend(flip_at(l, t, ours, &ik));
            }
        }
        let flips_ok = flips_report(&flips, "free", FLIP_ERR_CAP);
        let (held, hl, ht, exempt) = worst_off_path(&table, &flips, false);
        let firsts = first_flip_layers(&flips, eager.taps.len(), n_layer());
        println!(
            "free: first layer a flip lies on the path of, by position {}; {exempt} (layer, \
             position) outputs past a flip, printed and counted",
            firsts
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let ik = ik_last(man, vocab)?;
        let ours = eager.logits.last().ok_or("no logits")?;
        let (top, ik_top) = (argmax(ours), argmax(&ik));
        let ok = held <= FREE_BAND && flips_ok && top == ik_top && inconsistent.is_empty();
        println!(
            "free: {} tokens, {} flips ({} allowed), worst l_out_rel off every flip's path \
             {held:.3e} at layer {hl} position {ht} (band {FREE_BAND:.2}); last position argmax \
             ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            eager.tokens.len(),
            flips.len(),
            flips.iter().filter(|f| f.allowed(FLIP_ERR_CAP)).count(),
            rel(ours, &ik),
            verdict(ok)
        );
        Ok((ok, firsts))
    }

    // --------------------------------------------------- (t) step sets

    /// The last token eagerly with the taps armed from where the model
    /// stands: its layer outputs, its logits and argmax, then every store.
    fn run_last(m: &mut Qwen38Model, tok: u32) -> Result<Run, GateError> {
        run_steps(m, &[tok], StepMode::Eager, false)
    }

    /// The step of set `name` after its prefill fed by `path`; its argmax
    /// against ik's, a tie named and counted; with `band` — the batch set's
    /// tokens and the first layer a flip lies on the path of its last
    /// position — its layer outputs below that layer held to the band. The
    /// whole comparison is an oracle clause: the fixture tier leaves it to
    /// the real one, and runs the step only when another clause reads its
    /// run (`feeds`: (q) reads D3K's step-fed run), returning the empty run
    /// when it does not.
    fn step_set(
        m: &mut Qwen38Model,
        name: &str,
        path: Prompt38,
        (band, feeds): (Option<(&[u32], usize)>, bool),
        ties: &mut usize,
    ) -> Result<(bool, Run), GateError> {
        let vs_ik = q38::clause(
            &format!(
                "(t) step set {name} fed by {}: argmax, layer outputs and logits against ik's",
                path.name()
            ),
            Tag::Oracle,
        )?;
        if !vs_ik && !feeds {
            let none = Run {
                tokens: Vec::new(),
                logits: Vec::new(),
                taps: Vec::new(),
                routes: Vec::new(),
                stores: Vec::new(),
                ple_ring: Vec::new(),
            };
            return Ok((true, none));
        }
        let (man, pos, tok, prefill, held) = set_open((name, &IK), band)?;
        fresh(m)?;
        let t = Instant::now();
        if !prefill.is_empty() {
            m.prompt38(&prefill, path)?;
        }
        let r = run_last(m, tok)?;
        if !vs_ik {
            println!(
                "step {name}: position {pos} after {} fed by {} ({:.1} s, runtime value); \
                 compared with ik's in the real tier only",
                prefill.len(),
                path.name(),
                t.elapsed().as_secs_f64()
            );
            return Ok((true, r));
        }
        let vocab = m.body("step_set")?.vocab();
        let ik = ik_last(&man, vocab)?;
        let ours = r.logits.last().ok_or("no logits")?;
        let (top, ik_top, ik_2, margin, dist, logits_rel) = tie_numbers(ours, &ik);
        let tie = top != ik_top && top == ik_2 && margin <= 2.0 * dist && logits_rel <= FREE_BAND;
        *ties += usize::from(tie);
        let rels = layer_rels(&man, &r.taps, STREAMS * HIDDEN, n_layer())?;
        let inside = print_layers(name, &rels, held, FREE_BAND);
        let ok = (top == ik_top || tie) && inside;
        println!(
            "step {name}: position {pos} after {} fed by {} ({:.1} s, runtime value); argmax \
             ours={top} ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, our distance at \
             the two {dist:.4}{}); logits_rel {logits_rel:.3e} (printed; bounds a tie at \
             {FREE_BAND:.2}); worst \
             l_out_rel={:.3e} (band on layers 0..{held}, off every flip's path; the rest \
             printed) {}",
            prefill.len(),
            path.name(),
            t.elapsed().as_secs_f64(),
            if tie { ", a named tie" } else { "" },
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            verdict(ok)
        );
        Ok((ok, r))
    }

    // ------------------------------------ (q) the pass past the dense region

    /// D3K's prefill by passes against `steps`, its step-fed run: the step's
    /// token, logits, layer outputs and every store bit for bit.
    fn pass_selects(m: &mut Qwen38Model, steps: &Run) -> Result<bool, GateError> {
        let (_, _, tok, prefill, _) = set_open((D3K, &IK), None)?;
        fresh(m)?;
        let t = Instant::now();
        m.prompt38(&prefill, Prompt38::Pass)?;
        let fed = t.elapsed().as_secs_f64();
        let r = run_last(m, tok)?;
        let taps = r.taps.len() == steps.taps.len()
            && r.taps.iter().zip(&steps.taps).all(|(a, b)| same_bits(a, b));
        let mut ok = taps;
        println!(
            "pass {D3K}: {} ids by passes of up to {} ({fed:.1} s, runtime value), then the \
             step: layer outputs bit for bit the step-fed run's: {taps}",
            prefill.len(),
            Prompt38::PASS_ROWS
        );
        ok &= same_run(
            "D3K fed by passes vs fed by steps, the step after",
            &r,
            steps,
            false,
        );
        Ok(ok)
    }

    // ------------------------------------------------------- (v) verify

    /// The rows the verifies feed, the first T of them a verify's, and the
    /// token of the step after a commit: the batch set's tokens, rotated
    /// past the prefix.
    fn verify_rows(toks: &[u32]) -> Result<([u32; LANES], u32), GateError> {
        let t = |i: usize| -> Result<u32, GateError> {
            Ok(*toks
                .get(i % toks.len())
                .ok_or("the batch set holds no token")?)
        };
        Ok((
            [t(PREFIX)?, t(PREFIX + 1)?, t(PREFIX + 2)?, t(PREFIX + 3)?],
            t(PREFIX + 4)?,
        ))
    }

    /// A step's token and logits, or the error it failed with.
    type Stepped = Result<(u32, Vec<f32>), String>;

    /// What a run left after its last kept row: each row's token and logits
    /// (a verify's every row, a step run's every step), the stores and the
    /// PLE ring, the lane word, and the step after it.
    struct VRun {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
        lane: u32,
        next: Stepped,
    }

    /// The step of `next` from where the model stands: its token and logits,
    /// or the error it failed with (the model reset behind it).
    fn next_step(m: &mut Qwen38Model, next: u32) -> Stepped {
        let r = m.step(&[next]).and_then(|t| Ok((t, m.logits()?)));
        r.map_err(|e| {
            let text = e.to_string();
            let _ = m.reset();
            text
        })
    }

    /// Where a (v) run stands before its rows: [`fresh`], then the prefix
    /// as graph steps or as one prompt by passes.
    #[derive(Clone, Copy)]
    enum Prefix<'a> {
        Steps(&'a [u32]),
        Pass(&'a [u32]),
    }

    impl Prefix<'_> {
        fn feed(self, m: &mut Qwen38Model) -> Result<(), GateError> {
            fresh(m)?;
            m.set_mode(StepMode::Graph);
            match self {
                Prefix::Steps(t) => m.step(t)?,
                Prefix::Pass(t) => m.prompt38(t, Prompt38::Pass)?,
            };
            Ok(())
        }
    }

    /// Steps of `feeds` (each a run of rows), every row a step, after the
    /// prefix: the rows' tokens and logits, the stores, then the step of
    /// `next`.
    fn ref_run(
        m: &mut Qwen38Model,
        prefix: Prefix<'_>,
        feeds: &[&[u32]],
        next: u32,
    ) -> Result<VRun, GateError> {
        prefix.feed(m)?;
        let (mut tokens, mut logits) = (Vec::new(), Vec::new());
        for &t in feeds.iter().flat_map(|f| f.iter()) {
            tokens.push(m.step(&[t])?);
            logits.push(m.logits()?);
        }
        let (stores, ple_ring) = stores(m)?;
        let lane = m.body("ref_run")?.lane();
        Ok(VRun {
            tokens,
            logits,
            stores,
            ple_ring,
            lane,
            next: next_step(m, next),
        })
    }

    /// A verify of `rows` in `mode` from where the model stands, kept to its
    /// first `k` rows: its rows' tokens and logits, the host tier's services,
    /// host calls and `Cols` services during it, the stores after the
    /// commit, the lane word, then the step of `next` (when given).
    fn verify<const T: usize>(
        m: &mut Qwen38Model,
        rows: [u32; T],
        k: usize,
        mode: StepMode,
        next: Option<u32>,
    ) -> Result<(VRun, [u64; 3]), GateError> {
        m.set_mode(mode);
        let before = m.body("verify")?.hybrid().stats();
        let pos0 = m.pos();
        let tokens = m.step_rows::<T>(rows)?.to_vec();
        let logits = m.rows_logits::<T>()?.to_vec();
        let after = m.body("verify")?.hybrid().stats();
        let served = [
            after.served - before.served,
            after.host_calls - before.host_calls,
            after.cols_served - before.cols_served,
        ];
        m.rollback(pos0 + k as u32)?;
        m.set_mode(StepMode::Graph);
        let (stores, ple_ring) = stores(m)?;
        let lane = m.body("verify")?.lane();
        let next = match next {
            Some(t) => next_step(m, t),
            None => Err("no step after".to_string()),
        };
        Ok((
            VRun {
                tokens,
                logits,
                stores,
                ple_ring,
                lane,
                next,
            },
            served,
        ))
    }

    /// The layers whose live stores differ at count `pos`, and whether the
    /// PLE rings' live slots do ((v)'s rule: the committed lane, the conv
    /// ring's eight positions before the count, the K/V planes' and raw
    /// keys' rows below it, the pools complete at the count, the PLE ring's
    /// fourteen positions before the count).
    fn live_diff(a: &VRun, b: &VRun, pos: usize) -> (Vec<usize>, bool) {
        let conv_ch = 2 * K_HEADS * HEAD_V + V_HEADS * HEAD_V;
        let slots =
            |ring: usize, back: usize| (pos.saturating_sub(back)..pos).map(move |p| p % ring);
        let ring_same = |x: &[f32], y: &[f32], ring: usize, width: usize, back: usize| {
            x.len() == ring * width
                && y.len() == ring * width
                && slots(ring, back).all(|s| {
                    same_bits(
                        &x[s * width..(s + 1) * width],
                        &y[s * width..(s + 1) * width],
                    )
                })
        };
        // A plane of `heads` heads of `ctx` rows of `width` (`ctx` read off
        // its length), compared on the rows below the count.
        let rows_same = |x: &[u16], y: &[u16], heads: usize, width: usize| {
            let ctx = x.len() / (heads * width);
            x.len() == y.len()
                && x.len() == heads * ctx * width
                && pos <= ctx
                && (0..heads).all(|h| {
                    let at = h * ctx * width;
                    x[at..at + pos * width] == y[at..at + pos * width]
                })
        };
        let n = a.stores.len().max(b.stores.len());
        let layers = (0..n)
            .filter(|&l| match (a.stores.get(l), b.stores.get(l)) {
                (
                    Some(Store38Host::Rec { state: s, ring: r }),
                    Some(Store38Host::Rec {
                        state: s2,
                        ring: r2,
                    }),
                ) => !(same_bits(s, s2) && ring_same(r, r2, CONV_RING, conv_ch, 8)),
                (
                    Some(Store38Host::Qsa { k, v, raw, pooled }),
                    Some(Store38Host::Qsa {
                        k: k2,
                        v: v2,
                        raw: raw2,
                        pooled: pooled2,
                    }),
                ) => {
                    // A pool past the count is read only at a count that
                    // completes it, and that row writes it before its select.
                    let live = pos / POOL * IDX_DIM;
                    !(rows_same(k, k2, N_KV, HEAD)
                        && rows_same(v, v2, N_KV, HEAD)
                        && rows_same(raw, raw2, 1, IDX_DIM)
                        && pooled.len() == pooled2.len()
                        && live <= pooled.len()
                        && pooled[..live] == pooled2[..live])
                }
                _ => true,
            })
            .collect();
        let ple = !ring_same(&a.ple_ring, &b.ple_ring, PLE_RING, STREAMS * HIDDEN, 14);
        (layers, ple)
    }

    /// `got` — a verify kept to `k` rows — against `want`, the steps of those
    /// rows: every row's token and logits `got` ran that `want` holds, the
    /// live stores, the lane word the gate computes (`want_lane`), and the
    /// step after.
    fn same_vrun(label: &str, got: &VRun, want: &VRun, pos: usize, want_lane: u32) -> bool {
        let n = got.tokens.len().min(want.tokens.len());
        let tokens = got.tokens[..n] == want.tokens[..n];
        let logits = (0..n).all(|r| same_bits(&got.logits[r], &want.logits[r]));
        let (layers, ple) = live_diff(got, want, pos);
        let lane = got.lane == want_lane;
        let next = match (&got.next, &want.next) {
            (Ok((t, l)), Ok((t2, l2))) => t == t2 && same_bits(l, l2),
            _ => false,
        };
        let ok = tokens && logits && layers.is_empty() && !ple && lane && next;
        println!(
            "verify {label}: rows {:?} vs steps {:?} ({n} compared), logits bit for bit: \
             {logits}; live stores differing at layers {layers:?}, PLE ring: {ple}; lane {} \
             (want {want_lane}, computed here); the step after: {} {}",
            got.tokens,
            want.tokens,
            got.lane,
            match (&got.next, &want.next) {
                (Ok((t, _)), Ok((t2, _))) => format!("{t} vs {t2}, logits bit for bit {next}"),
                (Err(e), _) => format!("FAILED: {e}"),
                (_, Err(e)) => format!("the reference FAILED: {e}"),
            },
            verdict(ok)
        );
        ok
    }

    /// The structure of the captured verify of `T` rows: the node count, its
    /// batch mem ops, its argmax and handoff launches.
    fn verify_structure<const T: usize>(
        m: &mut Qwen38Model,
        want: usize,
    ) -> Result<bool, GateError> {
        let nodes = m.capture_rows::<T>()?;
        let counted = m.body("verify_structure")?.verify_launches(T);
        let list = m.rows_graph_nodes::<T>()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([_, b], other) = count_kinds(&list, [kernel, memop]);
        let named = |f: &dyn Fn(&str) -> bool| {
            list.iter()
                .filter(|n| n.kernel.as_ref().is_some_and(|k| f(&k.name)))
                .count()
        };
        let argmax = named(&|n| n.contains("argmax"));
        let cols = named(&|n| n == "ds41_ffn_handoff_10_cols");
        let one = named(&|n| n == "ds41_ffn_handoff_10");
        let (want_memops, layers) = (memops(), n_layer());
        let ok = nodes == want
            && counted == want
            && b == want_memops
            && other == 0
            && argmax == 1
            && cols == layers
            && one == 0;
        println!(
            "verify structure T={T}: graph_nodes={nodes} (want {want}; the program counts \
             {counted}) batch_mem_op={b} (want {want_memops}) other={other}; argmax launches \
             {argmax} (want 1: one head of {T} rows); ds41_ffn_handoff_10_cols {cols} (want \
             {layers}), ds41_ffn_handoff_10 {one} (want 0) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// One verify of `T` rows kept to `k`, against `refs[k − 1]`: the case,
    /// its structure of services, and a failed call named, not aborting.
    fn verify_case<const T: usize>(
        m: &mut Qwen38Model,
        toks: &[u32],
        rows: [u32; LANES],
        next: u32,
        k: usize,
        refs: &[VRun],
    ) -> Result<bool, GateError> {
        let label = format!("T={T} k={k}");
        let run = (|| -> Result<(VRun, [u64; 3], u32), GateError> {
            Prefix::Steps(&toks[..PREFIX]).feed(m)?;
            let c = m.body("verify_case")?.lane();
            let r: [u32; T] = std::array::from_fn(|i| rows[i]);
            let (v, served) = verify::<T>(m, r, k, StepMode::Graph, Some(next))?;
            Ok((v, served, c))
        })();
        let (v, served, c) = match run {
            Ok(x) => x,
            Err(e) => {
                println!("verify {label}: FAILED: {e} {}", verdict(false));
                m.reset()?;
                return Ok(false);
            }
        };
        let want = refs.get(k - 1).ok_or("a reference for every kept count")?;
        let mut ok = same_vrun(
            &label,
            &v,
            want,
            PREFIX + k,
            (c + k as u32 - 1) % LANES as u32,
        );
        // Rows past k: the verify's own, against the steps of all four.
        let all = refs.last().ok_or("the reference of four rows")?;
        let rows_ok = v.tokens[..] == all.tokens[..T]
            && (0..T).all(|r| same_bits(&v.logits[r], &all.logits[r]));
        let layers = n_layer();
        let services = served == [layers as u64; 3];
        ok &= rows_ok && services;
        println!(
            "verify {label}: every row's token and logits bit for bit the steps' {rows_ok}; the \
             replay's services, host calls and Cols services {served:?} (want {layers} each: \
             one go, one union call and one wait a layer) {}",
            verdict(rows_ok && services)
        );
        Ok(ok)
    }

    fn verify_clause(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let (rows, next) = verify_rows(toks)?;
        let mut ok = true;
        ok &= verify_structure::<2>(m, nodes_verify())?;
        ok &= verify_structure::<3>(m, nodes_verify())?;
        ok &= verify_structure::<4>(m, nodes_verify())?;
        let refs = (1..=LANES)
            .map(|k| ref_run(m, Prefix::Steps(&toks[..PREFIX]), &[&rows[..k]], next))
            .collect::<Result<Vec<_>, _>>()?;
        for k in 1..=2 {
            ok &= verify_case::<2>(m, toks, rows, next, k, &refs)?;
        }
        for k in 1..=3 {
            ok &= verify_case::<3>(m, toks, rows, next, k, &refs)?;
        }
        for k in 1..=4 {
            ok &= verify_case::<4>(m, toks, rows, next, k, &refs)?;
        }
        // A second verify from a moved lane: four rows kept to three (lane 2),
        // then four more kept whole (lanes 2, 3, 0, 1: the word to 1).
        let want = ref_run(
            m,
            Prefix::Steps(&toks[..PREFIX]),
            &[&rows[..3], &rows[..]],
            next,
        )?;
        let chained = (|| -> Result<VRun, GateError> {
            Prefix::Steps(&toks[..PREFIX]).feed(m)?;
            verify::<4>(m, rows, 3, StepMode::Graph, None)?;
            Ok(verify::<4>(m, rows, 4, StepMode::Graph, Some(next))?.0)
        })();
        match chained {
            Ok(v) => {
                // The second verify's rows are the reference's last four.
                let (t, l) = (want.tokens[3..].to_vec(), want.logits[3..].to_vec());
                let tail = VRun {
                    tokens: t,
                    logits: l,
                    stores: want.stores,
                    ple_ring: want.ple_ring,
                    lane: want.lane,
                    next: want.next,
                };
                ok &= same_vrun(
                    "after a commit of 3, T=4 k=4 from lane 2",
                    &v,
                    &tail,
                    PREFIX + 7,
                    (2 + 4 - 1) % LANES as u32,
                );
            }
            Err(e) => {
                println!("verify chained: FAILED: {e} {}", verdict(false));
                m.reset()?;
                ok = false;
            }
        }
        // Eager verify against the replay.
        let eager = (|| -> Result<(VRun, VRun), GateError> {
            fresh(m)?;
            m.step(&toks[..PREFIX])?;
            let g = verify::<4>(m, rows, 4, StepMode::Graph, Some(next))?.0;
            fresh(m)?;
            m.step(&toks[..PREFIX])?;
            let e = verify::<4>(m, rows, 4, StepMode::Eager, Some(next))?.0;
            Ok((e, g))
        })();
        match eager {
            Ok((e, g)) => {
                ok &= same_vrun("eager T=4 k=4 vs its replay", &e, &g, PREFIX + 4, g.lane);
            }
            Err(e) => {
                println!("verify eager: FAILED: {e} {}", verdict(false));
                m.reset()?;
                ok = false;
            }
        }
        ok &= verify_deep(m)?;
        ok &= verify_refusals(m, toks, rows)?;
        Ok(ok)
    }

    /// The deep verify: D3K's first [`DEEP`] prefill ids by passes, then a
    /// verify of the next four kept to [`DEEP_KEPT`], against the steps of
    /// those rows; the pooled rows past the count poisoned, then the two
    /// steps after, the prefill's next ids.
    fn verify_deep(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let ids = prefill
            .get(..DEEP + LANES)
            .ok_or_else(|| format!("{D3K}: a prefill of {} ids", prefill.len()))?
            .to_vec();
        let (prefix, rows) = (Prefix::Pass(&ids[..DEEP]), &ids[DEEP..]);
        let rows: [u32; LANES] = std::array::from_fn(|i| rows[i]);
        let (next, then) = (rows[DEEP_KEPT], rows[DEEP_KEPT + 1]);
        let want = ref_run(m, prefix, &[&rows[..DEEP_KEPT]], next)?;
        let want_then = next_step(m, then);
        let run = (|| -> Result<(VRun, u32, Stepped), GateError> {
            prefix.feed(m)?;
            let c = m.body("verify_deep")?.lane();
            let mut v = verify::<4>(m, rows, DEEP_KEPT, StepMode::Graph, None)?.0;
            {
                let (gpu, _, b) = m.body_parts("verify_deep")?;
                b.poison_pools_from(gpu, (DEEP + DEEP_KEPT) / POOL, POOL_POISON)?;
            }
            v.next = next_step(m, next);
            let got_then = next_step(m, then);
            Ok((v, c, got_then))
        })();
        match run {
            Ok((v, c, got_then)) => {
                let mut ok = same_vrun(
                    &format!(
                        "deep at {DEEP}, T=4 k={DEEP_KEPT}, pool 513's row rejected, the pools \
                         past the count poisoned"
                    ),
                    &v,
                    &want,
                    DEEP + DEEP_KEPT,
                    (c + DEEP_KEPT as u32 - 1) % LANES as u32,
                );
                let then_ok = matches!(
                    (&got_then, &want_then),
                    (Ok((t, l)), Ok((t2, l2))) if t == t2 && same_bits(l, l2)
                );
                ok &= then_ok;
                println!(
                    "verify deep: the step after that (count {}, completing pool 513 again \
                     before its select): {} {}",
                    DEEP + DEEP_KEPT + 2,
                    match (&got_then, &want_then) {
                        (Ok((t, _)), Ok((t2, _))) => {
                            format!("{t} vs {t2}, logits bit for bit {then_ok}")
                        }
                        (Err(e), _) => format!("FAILED: {e}"),
                        (_, Err(e)) => format!("the reference FAILED: {e}"),
                    },
                    verdict(then_ok)
                );
                Ok(ok)
            }
            Err(e) => {
                println!("verify deep: FAILED: {e} {}", verdict(false));
                m.reset()?;
                Ok(false)
            }
        }
    }

    /// (v)'s refusals: a step while a verify waits, a commit with no verify,
    /// and the planted lane's fault.
    fn verify_refusals(
        m: &mut Qwen38Model,
        toks: &[u32],
        rows: [u32; LANES],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        fresh(m)?;
        m.step(&toks[..PREFIX])?;
        m.step_rows::<2>([rows[0], rows[1]])?;
        let got = m.step(&[rows[2]]);
        let named = matches!(&got, Err(e) if e.to_string().contains("waits for its commit"));
        ok &= named;
        println!(
            "verify refusal: a step while the verify waits -> {} {}",
            match &got {
                Ok(t) => format!("accepted, next {t}"),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        m.rollback(PREFIX as u32 + 2)?;
        let got = m.rollback(PREFIX as u32);
        let named = matches!(&got, Err(e) if e.to_string().contains("with no verify waiting"));
        ok &= named;
        println!(
            "verify refusal: a commit back to {PREFIX} with no verify waiting -> {} {}",
            match &got {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        fresh(m)?;
        m.step(&toks[..PREFIX])?;
        {
            let (gpu, _, b) = m.body_parts("plant")?;
            b.plant_lane(gpu, 2)?;
        }
        let got = m.step(&[rows[0]]);
        let want = Fault::at(0, FaultSite::DeltaStamp);
        let raised = matches!(&got, Err(GpuError::Fault { fault, .. }) if *fault == want);
        ok &= raised;
        println!(
            "verify refusal: the lane word planted on lane 2, never written -> {} (want {want}) {}",
            match &got {
                Ok(t) => format!("accepted, next {t}"),
                Err(e) => e.to_string(),
            },
            verdict(raised)
        );
        m.reset()?;
        Ok(ok)
    }

    // ------------------------------------------------- (h) the head of m rows

    /// (h): from the prefix, a head of two rows over the loaded q8_0 lm_head,
    /// its input written from the host: two equal finite rows read back two
    /// equal tokens; then a NaN in row 0's input, then (the word cleared) in
    /// row 1's, makes that row's every logit NaN, and each readback is the
    /// head's [`FaultSite::Logit`], never a token — each launch after one
    /// that must have put its argmax's ticket count back. The head is built
    /// here, not reached through a verify, because the model path cannot
    /// plant one row alone: the head's mix checks every value it writes into
    /// the head's input and raises [`FaultSite::HcMix`] first, so a NaN
    /// upstream never reaches the logits as a Logit fault. Then, the word left
    /// raised, a verify of two rows: its head's readback carries the raised
    /// word, so the call is that fault and the model poisoned.
    fn head_rows(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let want = Fault::at(LAYER_HEAD, FaultSite::Logit);
        let (rows, _) = verify_rows(toks)?;
        fresh(m)?;
        m.step(&toks[..PREFIX])?;
        let mut ok = true;
        {
            let (gpu, w, b) = m.body_parts("head_rows")?;
            let mut head = Head::with_norm(gpu, w, b.head_eps(), 2, b.head_norm())?;
            let k = head.hidden();
            let row: Vec<f32> = (0..k)
                .map(|j| ((j * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            gpu.clear_fault()?;
            head.set_input(gpu, &[&row[..], &row[..]].concat())?;
            head.enqueue(gpu, w)?;
            let clean = head.tokens(gpu);
            let equal = matches!(&clean, Ok(t) if t.len() == 2 && t[0] == t[1]);
            ok &= equal;
            println!(
                "head rows: a head of 2 rows over the q8_0 lm_head, two equal finite rows -> {} \
                 (want two equal tokens) {}",
                match &clean {
                    Ok(t) => format!("tokens {t:?}"),
                    Err(e) => format!("error \"{e}\""),
                },
                verdict(equal)
            );
            for r in 0..2 {
                gpu.clear_fault()?;
                let mut x = [&row[..], &row[..]].concat();
                x[r * k] = f32::NAN;
                head.set_input(gpu, &x)?;
                head.enqueue(gpu, w)?;
                let got = head.tokens(gpu);
                let named = matches!(&got, Err(GpuError::Fault { fault, .. }) if *fault == want);
                ok &= named;
                println!(
                    "head rows: a head of 2 rows over the q8_0 lm_head, a NaN in row {r}'s input \
                     -> {} (want the output head's fault at site {}) {}",
                    match &got {
                        Ok(t) => format!("tokens {t:?}"),
                        Err(e) => format!("error \"{e}\""),
                    },
                    FaultSite::Logit.name(),
                    verdict(named)
                );
            }
        }
        let got = m.step_rows::<2>([rows[0], rows[1]]);
        let poisoned = m.poisoned();
        let carried = matches!(&got, Err(GpuError::Fault { fault, .. }) if *fault == want)
            && poisoned == Some(want);
        ok &= carried;
        println!(
            "head rows: the word left raised, a verify of 2 rows at {PREFIX} -> {}, poisoned {} \
             (want the output head's fault, and poisoned by it) {}",
            match &got {
                Ok(t) => format!("tokens {t:?}"),
                Err(e) => format!("error \"{e}\""),
            },
            poisoned.map_or_else(|| "none".to_string(), |f| f.to_string()),
            verdict(carried)
        );
        m.reset()?;
        Ok(ok)
    }

    // ------------------------------------------- (o) one owner of the position

    /// The calls (o) plants its failures on, each from position [`PREFIX`].
    #[derive(Clone, Copy)]
    enum Call {
        Step,
        Pass,
        Verify,
    }

    impl Call {
        fn name(self) -> &'static str {
            match self {
                Call::Step => "a graph step",
                Call::Pass => "an eager pass of 3 rows",
                Call::Verify => "a graph verify of 2 rows",
            }
        }

        /// The call at [`PREFIX`]: its last row's token.
        fn run(self, m: &mut Qwen38Model, toks: &[u32], rows: [u32; 2]) -> Result<u32, GpuError> {
            match self {
                Call::Step => m.step(&toks[PREFIX..PREFIX + 1]),
                Call::Pass => m.prompt38(&toks[PREFIX..PREFIX + 3], Prompt38::Pass),
                Call::Verify => m.step_rows::<2>(rows).map(|t| t[1]),
            }
        }
    }

    /// What a clean call left: its last row's token and logits, the PLE
    /// rows its fill named, and every store with the PLE ring after it (a
    /// verify's after its commit of both rows).
    struct Owned {
        token: u32,
        logits: Vec<f32>,
        rows: Vec<u32>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
    }

    /// The call `c` that just returned `token`, read back.
    fn owned(m: &mut Qwen38Model, c: Call, token: u32) -> Result<Owned, GateError> {
        let rows = m.body("owned")?.ple_rows().to_vec();
        let logits = match c {
            Call::Verify => {
                let [_, last] = m.rows_logits::<2>()?;
                m.rollback(PREFIX as u32 + 2)?;
                last
            }
            Call::Step | Call::Pass => m.logits()?,
        };
        let (stores, ple_ring) = stores(m)?;
        Ok(Owned {
            token,
            logits,
            rows,
            stores,
            ple_ring,
        })
    }

    /// [`fresh`], the prefix as graph steps, then the call `c`.
    fn clean_call(
        m: &mut Qwen38Model,
        c: Call,
        toks: &[u32],
        rows: [u32; 2],
    ) -> Result<Owned, GateError> {
        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        let token = c.run(m, toks, rows)?;
        owned(m, c, token)
    }

    /// `got` against the clean call `want`, bit for bit: a line's tail.
    fn same_owned(got: &Owned, want: &Owned) -> (bool, String) {
        let logits = same_bits(&got.logits, &want.logits);
        let rows = got.rows == want.rows && !got.rows.is_empty();
        let layers: Vec<usize> = (0..got.stores.len().max(want.stores.len()))
            .filter(|&l| match (got.stores.get(l), want.stores.get(l)) {
                (Some(x), Some(y)) => !x.same_bits(y),
                _ => true,
            })
            .collect();
        let ring = same_bits(&got.ple_ring, &want.ple_ring);
        let ok = got.token == want.token && logits && rows && layers.is_empty() && ring;
        (
            ok,
            format!(
                "token {} (clean {}), logits bit for bit: {logits}, PLE rows equal: {rows}, \
                 stores differing at layers {layers:?}, PLE ring equal: {ring}",
                got.token, want.token
            ),
        )
    }

    fn text<T: std::fmt::Display>(r: &Result<T, GpuError>) -> String {
        match r {
            Ok(t) => format!("accepted ({t})"),
            Err(e) => format!("error \"{e}\""),
        }
    }

    /// (o) for the call `c`: a failure before its launch, then one after.
    fn owner_call(
        m: &mut Qwen38Model,
        c: Call,
        toks: &[u32],
        rows: [u32; 2],
    ) -> Result<bool, GateError> {
        let clean = clean_call(m, c, toks, rows)?;
        let at = PREFIX as u32;

        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        let lane = m.body("owner")?.lane();
        m.body_parts("owner")?.2.plant_before_launch();
        let first = c.run(m, toks, rows);
        let (pos, lane_after) = (m.pos(), m.body("owner")?.lane());
        let failed = matches!(&first, Err(e)
            if e.to_string().contains("the planted failure before the launch"));
        let again = c.run(m, toks, rows);
        let (rerun_ok, rerun) = match &again {
            Ok(t) => same_owned(&owned(m, c, *t)?, &clean),
            Err(_) => (false, String::new()),
        };
        let before_ok = failed
            && pos == at
            && lane_after == lane
            && m.poisoned().is_none()
            && again.is_ok()
            && rerun_ok;
        println!(
            "position owner, {}: a failure before the launch at {at}: {} at position {pos} (want \
             {at}), lane {lane_after} (want {lane}); the call again: {} {rerun} {}",
            c.name(),
            text(&first),
            text(&again),
            verdict(before_ok)
        );

        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        m.body_parts("owner")?.2.plant_after_launch();
        let first = c.run(m, toks, rows);
        let pos = m.pos();
        let failed = matches!(&first, Err(e)
            if e.to_string().contains("the planted failure after the launch"));
        let kept = m.body("owner")?.kept(at, pos);
        let step = m.step(&toks[PREFIX..PREFIX + 1]);
        let refused = matches!(&step, Err(e)
            if e.to_string().contains("failed after its chain was launched"));
        let (pos_step, poisoned) = (m.pos(), m.poisoned());
        let back = clean_call(m, c, toks, rows)?;
        let (back_ok, back_line) = same_owned(&back, &clean);
        let after_ok = failed
            && pos == at
            && kept.at < at
            && refused
            && pos_step == at
            && poisoned.is_none()
            && back_ok;
        println!(
            "position owner, {}: a failure after the launch at {at}: {} at position {pos} (want \
             {at}); a cut keeps {} ({kept}, want under {at}); a step at {at}: {} (want refused \
             by name), position {pos_step}; after reset the call again: {back_line} {}",
            c.name(),
            text(&first),
            kept.at,
            text(&step),
            verdict(after_ok)
        );
        m.reset()?;
        Ok(before_ok && after_ok)
    }

    /// (o): each call's failures, before and after its launch.
    fn position_owner(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let (rows, _) = verify_rows(toks)?;
        let rows = [rows[0], rows[1]];
        let mut ok = true;
        for c in [Call::Step, Call::Pass, Call::Verify] {
            ok &= owner_call(m, c, toks, rows)?;
        }
        Ok(ok)
    }

    // ------------------------------------------------ (g) the ubatch walk

    /// Route tap rows the batch set's walks arm: more than its five, so a
    /// read of more positions than a walk ran finds rows to read.
    const TAP_ROWS: usize = 128;

    /// Positions the free arm asks its taps for, past the five it walked.
    const TAP_ASK: usize = 100;

    /// PIN(2026-09-28): the ubatch walk's distance from the pass per layer, as
    /// the error model gives it: a q8 activation of 32 values moves a
    /// block's RMS by at most `rounding::q8_32_rel` = 1.2858e-2 of it (the
    /// largest crest, √32); a projection passes that relative error to its
    /// output. A layer's stream gains it from four projections in series
    /// whose outputs join the stream — the mixer's input and output
    /// projections, the shared expert's gate·up and down — √4 of it
    /// independent, 2.572e-2 a layer (the mixes' down and up reach the
    /// stream through σ, whose slope is at most ¼, and add under 4 %); layers
    /// add independently, so layer `l`'s output and store sit within
    /// √(l + 1) · 2.572e-2 of the pass's where both take the same experts;
    /// the head reads the last layer's, √48 · 2.572e-2 = 0.178.
    fn gemm_band(l: usize) -> f64 {
        let q = q8_32_rel();
        ((l + 1) as f64).sqrt() * 2.0 * q
    }

    /// PIN(2026-09-28): a flip between the ubatch walk's routing and the
    /// pass's is excused while each exchanged pair's gap in the pass's logits
    /// lies within our two logits' distance there, and that distance within
    /// six standard deviations of the router's error at the layer: a logit
    /// is a dot of the layer's input, which carries `band` of relative error
    /// ([`gemm_band`] at the layer for the ubatch walk, [`card_band`] for the
    /// card leg), so a logit moves by about that times the logits' RMS; two
    /// logits, three deviations each.
    fn flip_cap(band: f64, logits: &[f32]) -> f64 {
        let rms = (logits.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
            / logits.len().max(1) as f64)
            .sqrt();
        6.0 * band * rms
    }

    /// The deviation of `v` about its mean: the spread a ranking reads (a
    /// common offset moves no rank).
    fn spread(v: &[f32]) -> f64 {
        let n = v.len().max(1) as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        (v.iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The widest gap in the pass's logits a flip may exchange where no
    /// earlier flip lies on its path, derived: a logit moves by about the
    /// layer's `band` times the logits' [`spread`] (the error of the layer's
    /// input, read through the router's rows; the logits' common offset,
    /// which [`flip_cap`]'s RMS carries, moves no rank), and a pair's gap is
    /// crossed only by the two logits' errors together: three deviations
    /// each. A flip at a wider gap is a pick the error model does not
    /// explain.
    fn margin_cap(band: f64, logits: &[f32]) -> f64 {
        6.0 * band * spread(logits)
    }

    /// Each layer's relative distance between two runs' live stores at count
    /// `pos` — the committed lane with the conv ring, the K/V planes' and raw
    /// keys' rows below the count and the pools complete at it — and the PLE
    /// rings'. Rows past the count hold [`PLANE_FILL`] in both and would
    /// only dilute the distance.
    fn live_rel(a: &Run, b: &Run, pos: usize) -> (Vec<f64>, f64) {
        let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&h| half_to_f32(h)).collect() };
        let rows = |x: &[u16], heads: usize, width: usize| -> Vec<f32> {
            let ctx = x.len() / (heads * width).max(1);
            (0..heads)
                .flat_map(|h| {
                    let at = h * ctx * width;
                    f(&x[at..at + pos.min(ctx) * width])
                })
                .collect()
        };
        let layers = a
            .stores
            .iter()
            .zip(&b.stores)
            .map(|pair| match pair {
                (
                    Store38Host::Rec { state: s, ring: r },
                    Store38Host::Rec {
                        state: s2,
                        ring: r2,
                    },
                ) => {
                    let x: Vec<f32> = s.iter().chain(r).copied().collect();
                    let y: Vec<f32> = s2.iter().chain(r2).copied().collect();
                    rel(&x, &y)
                }
                (
                    Store38Host::Qsa { k, v, raw, pooled },
                    Store38Host::Qsa {
                        k: k2,
                        v: v2,
                        raw: raw2,
                        pooled: pooled2,
                    },
                ) => {
                    let live = (pos / POOL * IDX_DIM).min(pooled.len()).min(pooled2.len());
                    let x: Vec<f32> = [
                        rows(k, N_KV, HEAD),
                        rows(v, N_KV, HEAD),
                        rows(raw, 1, IDX_DIM),
                        f(&pooled[..live]),
                    ]
                    .concat();
                    let y: Vec<f32> = [
                        rows(k2, N_KV, HEAD),
                        rows(v2, N_KV, HEAD),
                        rows(raw2, 1, IDX_DIM),
                        f(&pooled2[..live]),
                    ]
                    .concat();
                    rel(&x, &y)
                }
                _ => f64::INFINITY,
            })
            .collect();
        (layers, rel(&a.ple_ring, &b.ple_ring))
    }

    /// The ubatch walk of `toks` from [`fresh`] with its route taps armed
    /// at [`TAP_ROWS`] and, when `plant` is given, the routes of its
    /// positions planted (`Body38::plant_ubatch_routes`): the run and the
    /// taps of its positions. The taps and the plant stay as they are.
    fn gemm_walk(
        m: &mut Qwen38Model,
        toks: &[u32],
        plant: Option<&[Vec<RouteTap>]>,
    ) -> Result<(Run, Vec<Vec<RouteTap>>), GateError> {
        fresh(m)?;
        {
            let (gpu, _, b) = m.body_parts("gemm_walk")?;
            b.set_ubatch_route_taps(gpu, TAP_ROWS)?;
            b.plant_ubatch_routes(gpu, plant.map(|r| (0, r)))?;
        }
        let last = m.prompt38(toks, Prompt38::Gemm)?;
        let logits = m.logits()?;
        let routes = {
            let (gpu, _, b) = m.body_parts("gemm_walk")?;
            b.ubatch_route_taps(gpu, 0, toks.len())?
        };
        let (stores, ple_ring) = stores(m)?;
        let run = Run {
            tokens: vec![last],
            logits: vec![logits],
            taps: Vec::new(),
            routes: Vec::new(),
            stores,
            ple_ring,
        };
        Ok((run, routes))
    }

    /// The route taps off and no route planted.
    fn gemm_taps_off(m: &mut Qwen38Model) -> Result<(), GateError> {
        let (gpu, _, b) = m.body_parts("gemm_taps_off")?;
        b.set_ubatch_route_taps(gpu, 0)?;
        b.plant_ubatch_routes(gpu, None)?;
        Ok(())
    }

    /// Two runs' last tokens and logits and every store bit for bit.
    fn same_last(a: &Run, b: &Run) -> bool {
        a.tokens.last() == b.tokens.last()
            && same_bits(
                a.logits.last().map_or(&[][..], |v| &v[..]),
                b.logits.last().map_or(&[][..], |v| &v[..]),
            )
            && store_diff(a, b) == (Vec::new(), false)
    }

    /// Each flip of one run's routes (`ours`, `[token][layer]`) against
    /// another's (`theirs`, the reference), judged as (g)'s free arm judges
    /// them, and the tally.
    struct FlipTally {
        /// Each flip, its pair cap ([`flip_cap`]) and its margin bound
        /// ([`margin_cap`]).
        flips: Vec<(Flip, f64, f64)>,
        refused: usize,
        first: usize,
        wide: usize,
        smallest: f64,
    }

    impl FlipTally {
        /// Every flip allowed by the pair rule, none first on its path at a
        /// wide margin.
        fn ok(&self) -> bool {
            self.refused == 0 && self.wide == 0
        }

        /// Whether a flip lies on the path of position `t`'s last layer: at
        /// any layer, at `t` or before.
        fn on_path(&self, t: usize) -> bool {
            self.flips.iter().any(|(f, _, _)| f.token <= t)
        }
    }

    /// The flips of `ours` against `theirs` (each `[token][layer]`), each
    /// printed with `arm`: excused by the pair rule under [`flip_cap`] and,
    /// where no earlier flip lies on its path, only at a gap within
    /// [`margin_cap`], both at the layer's `band`.
    fn judge_flips(
        arm: &str,
        ours: &[Vec<RouteTap>],
        theirs: &[Vec<RouteTap>],
        band: fn(usize) -> f64,
    ) -> FlipTally {
        let mut flips = Vec::new();
        for (t, (ours, theirs)) in ours.iter().zip(theirs).enumerate() {
            for (l, (o, p)) in ours.iter().zip(theirs).enumerate() {
                let ids: Vec<i32> = p.ids.iter().map(|&e| e as i32).collect();
                let margin = flip::margin(&p.logits, &ids);
                if let Some(f) =
                    Flip::between((l, t), (&o.ids, &o.logits), (&ids, &p.logits), margin)
                {
                    let b = band(l);
                    flips.push((f, flip_cap(b, &p.logits), margin_cap(b, &p.logits)));
                }
            }
        }
        let (mut refused, mut first, mut wide) = (0usize, 0usize, 0usize);
        for (f, cap, bound) in &flips {
            let prior = flips
                .iter()
                .any(|(g, _, _)| g.layer < f.layer && g.token <= f.token);
            let widest = f
                .pairs
                .iter()
                .map(|p| p.2)
                .fold(f64::NEG_INFINITY, f64::max);
            let at_wide = !prior && (widest > *bound || widest.is_nan());
            refused += usize::from(!f.allowed(*cap));
            first += usize::from(!prior);
            wide += usize::from(at_wide);
            println!(
                "{}; {}",
                f.line(arm, *cap),
                if prior {
                    "past an earlier flip on its path (margin rule not applied)".to_owned()
                } else {
                    format!(
                        "first on its path: widest gap {widest:.3e} (margin bound {bound:.3e}) {}",
                        if at_wide {
                            "FAIL: a flip at a wide margin"
                        } else {
                            "within"
                        }
                    )
                }
            );
        }
        let smallest = flips
            .iter()
            .map(|(f, _, _)| f.margin)
            .fold(f64::INFINITY, f64::min);
        FlipTally {
            flips,
            refused,
            first,
            wide,
            smallest,
        }
    }

    /// (g) on the batch set, the free arm: the ubatch walk ([`gemm_walk`],
    /// its own routing), then the pass (by passes from [`fresh`]), the walk
    /// against the pass (`eager`, the eager steps' taps, bit for bit the
    /// pass's routing) and ik — the last argmax equal to ik's; every route
    /// tap the top ten of its own logits; each flip against the pass's
    /// routing excused by the pair rule under [`flip_cap`], and, where no
    /// earlier flip lies on its path, only at a gap within [`margin_cap`]; its
    /// logits and stores printed. Around it: a read of the taps past the
    /// walk's positions refused by name; the reset before the pass leaving no
    /// split behind; a walk of more rows than the armed taps refused before
    /// it moves anything. Returns the pass's run.
    fn gemm_free(
        m: &mut Qwen38Model,
        man: &RefManifest,
        toks: &[u32],
        eager: &Run,
    ) -> Result<(bool, Run), GateError> {
        let n = toks.len();
        let (free, routes) = gemm_walk(m, toks, None)?;
        let (ask, split) = {
            let (gpu, _, b) = m.body_parts("gemm_free")?;
            (b.ubatch_route_taps(gpu, 0, TAP_ASK), b.ubatch_split())
        };
        gemm_taps_off(m)?;
        let pass_run = run_pass(m, toks)?;
        let after = m.body("gemm_free")?.ubatch_split();
        let pass = &pass_run;
        let vocab = m.body("gemm_free")?.vocab();
        let (last, top) = (free.tokens[0], argmax(&free.logits[0]));
        let (ik_ok, ik_text) = ik_argmax(man, vocab, "(g) free: the last argmax", top)?;
        let mut ok = ik_ok && last == top;
        println!(
            "gemm free: last argmax ours={top} (returned {last}) ik={ik_text} (the pass's {:?}) {}",
            pass.tokens,
            verdict(ik_ok && last == top)
        );
        let want = format!("the last ubatch walk ran 0..{n}");
        let named = matches!(&ask, Err(e) if e.to_string().contains(&want));
        ok &= named;
        println!(
            "gemm taps: taps armed at {TAP_ROWS} rows, a walk of {n}, a read of {TAP_ASK} \
             positions -> {} (want an error naming {want:?}) {}",
            match &ask {
                Ok(r) => format!("accepted, {} rows", r.len()),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        let inconsistent: Vec<(usize, usize)> = routes
            .iter()
            .enumerate()
            .flat_map(|(t, ls)| {
                ls.iter()
                    .enumerate()
                    .filter(|(_, r)| !picks_its_top(r))
                    .map(move |(l, _)| (l, t))
            })
            .collect();
        let taps_ok = routes.len() == n
            && routes.iter().all(|ls| ls.len() == n_layer())
            && inconsistent.is_empty();
        ok &= taps_ok;
        println!(
            "gemm free: route taps of {} tokens, those not the top {N_USED} of their own logits: \
             {} {:?} {}",
            routes.len(),
            inconsistent.len(),
            &inconsistent[..inconsistent.len().min(8)],
            verdict(taps_ok)
        );
        let tally = judge_flips("gemm", &routes, &eager.routes, gemm_band);
        let flips_ok = tally.ok();
        ok &= flips_ok;
        println!(
            "gemm free: {} flip(s) against the pass's routing, {} allowed by the pair rule; {} \
             first on their path, {} of them at a wide margin; the smallest pass margin that \
             flipped {:.3e} {}",
            tally.flips.len(),
            tally.flips.len() - tally.refused,
            tally.first,
            tally.wide,
            tally.smallest,
            verdict(flips_ok)
        );
        let (layers, ple) = live_rel(&free, pass, n);
        let shown: Vec<usize> = (0..layers.len())
            .filter(|l| l % 8 == 0 || l + 1 == n_layer())
            .collect();
        println!(
            "gemm free: last logits' distance from the pass's {:.3e}, stores' at layers {shown:?} \
             {:?}, PLE ring {ple:.3e} (printed: the flips move them; the forced arm holds the \
             bands)",
            rel(&free.logits[0], pass.logits.last().ok_or("no pass logits")?),
            shown
                .iter()
                .map(|&l| format!("{:.3e}", layers[l]))
                .collect::<Vec<_>>()
        );
        let split_ok = split.is_some() && after.is_none();
        ok &= split_ok;
        println!(
            "gemm split: the walk's {split:?}, after reset and a pass {after:?} (want Some, then \
             None) {}",
            verdict(split_ok)
        );
        ok &= gemm_taps_first(m, toks, &free)?;
        Ok((ok, pass_run))
    }

    /// (g): with the route taps armed at fewer rows than the batch set's,
    /// its ubatch walk is refused by name before it moves anything — the
    /// position kept, the model not poisoned — and the same call with the
    /// taps off leaves the free walk's (`free`, from [`fresh`]) last token,
    /// logits and every store bit for bit.
    fn gemm_taps_first(m: &mut Qwen38Model, toks: &[u32], free: &Run) -> Result<bool, GateError> {
        let short = toks.len() - 2;
        fresh(m)?;
        {
            let (gpu, _, b) = m.body_parts("gemm_taps_first")?;
            b.set_ubatch_route_taps(gpu, short)?;
        }
        let got = m.prompt38(toks, Prompt38::Gemm);
        let named = matches!(&got, Err(e) if e.to_string().contains("armed route tap rows"));
        let pos = m.pos();
        let kept = pos == 0 && m.poisoned().is_none();
        gemm_taps_off(m)?;
        let (same, again_text) = match m.prompt38(toks, Prompt38::Gemm) {
            Ok(last) => {
                let logits = m.logits()?;
                let (stores, ple_ring) = stores(m)?;
                let r = Run {
                    tokens: vec![last],
                    logits: vec![logits],
                    taps: Vec::new(),
                    routes: Vec::new(),
                    stores,
                    ple_ring,
                };
                (same_last(&r, free), format!("token {last}"))
            }
            Err(e) => (false, format!("error \"{e}\"")),
        };
        let p = named && kept && same;
        println!(
            "gemm taps first: taps armed at {short} rows, a ubatch of {} -> {}; position {pos}, \
             not poisoned (kept: {kept}); the same call with the taps off: {again_text}, the \
             free walk's last token, logits and every store bit for bit: {same} {}",
            toks.len(),
            match &got {
                Ok(t) => format!("accepted, next {t}"),
                Err(e) => e.to_string(),
            },
            verdict(p)
        );
        Ok(p)
    }

    /// (g) on the batch set, the forced arm: the ubatch walk with the pass's
    /// own routes planted (`eager`'s taps, `Body38::plant_ubatch_routes`), so
    /// no token's experts can differ from the pass's and the error model
    /// alone separates the two: every tapped id the pass's; the last argmax
    /// equal to ik's; the last logits within [`gemm_band`] of the head, and
    /// every layer's live store and the PLE ring within [`gemm_band`] of the
    /// pass's, with no layer excused; the router's own logits against the
    /// pass's printed per layer (the input [`margin_cap`] is derived from).
    fn gemm_forced(
        m: &mut Qwen38Model,
        man: &RefManifest,
        toks: &[u32],
        (eager, pass): (&Run, &Run),
    ) -> Result<bool, GateError> {
        let n = toks.len();
        let walk = gemm_walk(m, toks, Some(&eager.routes));
        gemm_taps_off(m)?;
        let (forced, routes) = walk?;
        let mut taken = 0usize;
        let mut other = Vec::new();
        for (t, (ours, theirs)) in routes.iter().zip(&eager.routes).enumerate() {
            for (l, (o, p)) in ours.iter().zip(theirs).enumerate() {
                if o.ids == p.ids {
                    taken += 1;
                } else {
                    other.push((l, t));
                }
            }
        }
        let cells = n * n_layer();
        let ids_ok = routes.len() == n && taken == cells;
        let mut ok = ids_ok;
        println!(
            "gemm forced: the planted routes' ids taken at {taken} of {cells} (layer, position) \
             cells, others {:?} {}",
            &other[..other.len().min(8)],
            verdict(ids_ok)
        );
        let vocab = m.body("gemm_forced")?.vocab();
        let (last, top) = (forced.tokens[0], argmax(&forced.logits[0]));
        let (ik_ok, ik_text) = ik_argmax(man, vocab, "(g) forced: the last argmax", top)?;
        let argmax_ok = ik_ok && last == top;
        ok &= argmax_ok;
        let logits_rel = rel(
            &forced.logits[0],
            pass.logits.last().ok_or("no pass logits")?,
        );
        let head_band = gemm_band(n_layer() - 1);
        let logits_ok = logits_rel <= head_band;
        ok &= logits_ok;
        println!(
            "gemm forced: last argmax ours={top} (returned {last}) ik={ik_text}; last logits' \
             distance from the pass's {logits_rel:.3e} (band {head_band:.3e}) {}",
            verdict(argmax_ok && logits_ok)
        );
        let (layers, ple) = live_rel(&forced, pass, n);
        let mut past = Vec::new();
        for (l, &e) in layers.iter().enumerate() {
            let over = e > gemm_band(l) || e.is_nan();
            if over {
                past.push(l);
            }
            if l % 8 == 0 || l + 1 == layers.len() || over {
                println!(
                    "gemm forced: layer {l} store distance {e:.3e} (band {:.3e})",
                    gemm_band(l)
                );
            }
        }
        let ple_band = gemm_band(ple_layer()?);
        let ple_ok = ple <= ple_band;
        let stores_ok = layers.len() == n_layer() && past.is_empty() && ple_ok;
        ok &= stores_ok;
        println!(
            "gemm forced: every layer's store within its band, past it at {past:?}; PLE ring \
             {ple:.3e} (band {:.3e}) {}",
            ple_band,
            verdict(stores_ok)
        );
        let (mut worst, mut wl, mut peak) = (0.0f64, 0usize, 0.0f64);
        for l in 0..n_layer() {
            let (mut rel_l, mut peak_l) = (0.0f64, 0.0f64);
            for (ours, theirs) in routes.iter().zip(&eager.routes) {
                let (Some(o), Some(p)) = (ours.get(l), theirs.get(l)) else {
                    continue;
                };
                let d: Vec<f32> = o.logits.iter().zip(&p.logits).map(|(a, b)| a - b).collect();
                let s = spread(&p.logits).max(f64::MIN_POSITIVE);
                rel_l = rel_l.max(spread(&d) / s);
                let dm = d.iter().map(|&x| f64::from(x)).sum::<f64>() / d.len().max(1) as f64;
                let pk = d
                    .iter()
                    .map(|&x| (f64::from(x) - dm).abs())
                    .fold(0.0, f64::max);
                peak_l = peak_l.max(pk / s);
            }
            if rel_l / gemm_band(l) > worst {
                (worst, wl) = (rel_l / gemm_band(l), l);
            }
            peak = peak.max(peak_l / (3.0 * gemm_band(l)));
            if l % 8 == 0 || l + 1 == n_layer() {
                println!(
                    "gemm forced: layer {l} router logits' error over their spread, worst over \
                     positions {rel_l:.3e} (the input's band {:.3e}); peak {peak_l:.3e} (the margin \
                     rule's three deviations {:.3e})",
                    gemm_band(l),
                    3.0 * gemm_band(l)
                );
            }
        }
        println!(
            "gemm forced: router logits' error over [`gemm_band`], worst {worst:.3} at layer \
             {wl}; peak error over three deviations, worst {peak:.3} (printed: [`margin_cap`]'s \
             premise, 1 or under where it holds)"
        );
        Ok(ok)
    }

    /// (g) `auto` on the batch set's five positions — below
    /// `Prompt38::GEMM_FROM`, so the pass — from [`fresh`]: the pass's last
    /// token, logits and every store bit for bit (`prompt38` runs the path
    /// `Prompt38::resolve` names; an `auto` it returned would be refused by
    /// name).
    fn gemm_auto_call(m: &mut Qwen38Model, toks: &[u32], pass: &Run) -> Result<bool, GateError> {
        fresh(m)?;
        let (same, text) = match m.prompt38(toks, Prompt38::Auto) {
            Ok(last) => {
                let logits = m.logits()?;
                let (stores, ple_ring) = stores(m)?;
                let r = Run {
                    tokens: vec![last],
                    logits: vec![logits],
                    taps: Vec::new(),
                    routes: Vec::new(),
                    stores,
                    ple_ring,
                };
                (same_last(&r, pass), format!("token {last}"))
            }
            Err(e) => (false, format!("error \"{e}\"")),
        };
        println!(
            "gemm auto call: {} positions by `auto` -> {text}; the pass's last token, logits and \
             every store bit for bit: {same} {}",
            toks.len(),
            verdict(same)
        );
        Ok(same)
    }

    /// (g) on the batch set: the free arm and the pass, `auto`, then the
    /// forced arm.
    fn gemm_batch(
        m: &mut Qwen38Model,
        man: &RefManifest,
        toks: &[u32],
        eager: &Run,
    ) -> Result<bool, GateError> {
        let (mut ok, pass) = gemm_free(m, man, toks, eager)?;
        ok &= gemm_auto_call(m, toks, &pass)?;
        ok &= gemm_forced(m, man, toks, (eager, &pass))?;
        Ok(ok)
    }

    /// (g) on D3K's prefill: one ubatch against two cut inside the selecting
    /// rows ([`SPLIT`], mid-pool, off the eight-row runs), the last logits and
    /// every store bit for bit; the walk's cut of the selecting layers' rows
    /// against the rule's; then the step after the ubatch against ik's.
    fn gemm_d3k(m: &mut Qwen38Model, ties: &mut usize) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let prefill = prefill.to_vec();
        let rows = m.body("gemm_d3k")?.ubatch_rows();
        fresh(m)?;
        let t = Instant::now();
        let one_tok = m.prompt38(&prefill, Prompt38::Gemm)?;
        let fed = t.elapsed().as_secs_f64();
        let split = m.body("gemm_d3k")?.ubatch_split();
        let one = Run {
            tokens: vec![one_tok],
            logits: vec![m.logits()?],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: Vec::new(),
            ple_ring: Vec::new(),
        };
        let (one_stores, one_ring) = stores(m)?;
        let one = Run {
            stores: one_stores,
            ple_ring: one_ring,
            ..one
        };
        // A row selects once it sees more complete pools than the 512 a
        // select keeps (TOP_K 2,048 over pools of 4): at a count of
        // 4 · 513 = 2,052, position 2,051. From position 0 the prefill
        // flash takes the 2,051 rows before it.
        let dense = (POOL * (KEPT + 1) - 1).min(prefill.len());
        let want = (dense, prefill.len() - dense);
        let split_ok = rows >= prefill.len() && split == Some(want);
        println!(
            "gemm {D3K}: {} ids in one ubatch (the load's size {rows}; {fed:.1} s, runtime \
             value); the selecting layers' rows by the prefill flash and by the selection {split:?} \
             (want {want:?}, from KEPT and POOL) {}",
            prefill.len(),
            verdict(split_ok)
        );
        let mut ok = split_ok;
        fresh(m)?;
        m.prompt38(&prefill[..SPLIT], Prompt38::Gemm)?;
        let cut_tok = m.prompt38(&prefill[SPLIT..], Prompt38::Gemm)?;
        let (cut_stores, cut_ring) = stores(m)?;
        let cut = Run {
            tokens: vec![cut_tok],
            logits: vec![m.logits()?],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: cut_stores,
            ple_ring: cut_ring,
        };
        ok &= same_run(
            &format!("D3K by ubatches cut at {SPLIT} vs one ubatch"),
            &cut,
            &one,
            true,
        );
        ok &= gemm_timed(m, &prefill, &one)?;
        ok &= step_set(m, D3K, Prompt38::Gemm, (None, false), ties)?.0;
        Ok(ok)
    }

    /// (g) with the walk's timing armed (`BLOOMERY_STEP_STATS`): the same
    /// ubatch writes `one`'s bits, and its record holds one ubatch and a row
    /// for every layer with every card span read, finite and not negative,
    /// whose routing counts hold: each row's slots are every position's
    /// [`N_USED`] (D3K's experts all host, none left out), its widest expert
    /// at least the mean (`m_max · experts ≥ slots`), its hot experts'
    /// columns a sub-sum of consistent shape (`cols_hot ≤ slots`, none hot
    /// exactly when none summed, each hot expert past [`HOT_COLS`]), and
    /// `Σ m² ≥ slots²/experts` (Cauchy–Schwarz).
    fn gemm_timed(m: &mut Qwen38Model, prefill: &[u32], one: &Run) -> Result<bool, GateError> {
        m.set_prompt38_stats(true)?;
        fresh(m)?;
        let tok = m.prompt38(prefill, Prompt38::Gemm)?;
        let logits = m.logits()?;
        let (st, ring) = stores(m)?;
        let stats = m.take_prompt38_stats()?;
        m.set_prompt38_stats(false)?;
        let timed = Run {
            tokens: vec![tok],
            logits: vec![logits],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: st,
            ple_ring: ring,
        };
        let mut ok = same_run(
            "D3K by one ubatch, timing armed vs unarmed",
            &timed,
            one,
            true,
        );
        let (ubatches, rows, spans) = match &stats {
            Some(s) => (
                s.ubatches,
                s.rows.len(),
                s.rows
                    .iter()
                    .filter(|r| {
                        [r.front_ms, r.down_ms, r.shadow_ms, r.upload_ms, r.back_ms]
                            .iter()
                            .all(|v| v.is_finite() && *v >= 0.0)
                            && r.front_ms > 0.0
                    })
                    .count(),
            ),
            None => (0, 0, 0),
        };
        let layers = n_layer();
        let rec_ok = ubatches == 1 && rows == layers && spans == layers;
        println!(
            "gemm {D3K}: the timed walk's record: {ubatches} ubatch(es) (want 1), {rows} layer rows \
             (want {layers}), {spans} with every card span read (want {layers}) {}",
            verdict(rec_ok)
        );
        ok &= rec_ok;
        // Σ slots over the rows against every position's ten host slots, and
        // each row's counts against the sums they claim to come from.
        let (mut sum, mut bad, mut m_max) = (0u64, 0usize, 0usize);
        if let Some(s) = &stats {
            for r in &s.rows {
                let e = r.experts as u64;
                sum += r.slots;
                m_max = m_max.max(r.m_max);
                bad += usize::from(
                    r.cols != prefill.len()
                        || r.slots != r.cols as u64 * N_USED as u64
                        || r.m_max as u64 * e < r.slots
                        || r.cols_hot as u64 > r.slots
                        || (r.m_hot == 0) != (r.cols_hot == 0)
                        || (r.m_hot > 0
                            && (r.cols_hot as u64) < r.m_hot as u64 * (HOT_COLS as u64 + 1))
                        || r.m_sq * e < r.slots * r.slots,
                );
            }
        }
        let want = rows as u64 * prefill.len() as u64 * N_USED as u64;
        let counts_ok = rec_ok && bad == 0 && sum == want;
        println!(
            "gemm {D3K}: the timed walk's routing counts: {bad} layer row(s) off of {rows} (Σ slots \
             {sum}, want rows·P·N_USED {want}; m_max {m_max}) {}",
            verdict(counts_ok)
        );
        ok &= counts_ok;
        Ok(ok)
    }

    /// Where (g) cuts D3K's prefill in two: inside the selecting rows, in
    /// the middle of pool 650 (positions 2,600 to 2,603), 550 rows past the
    /// first selecting row (not a multiple of the eight-row runs), and past
    /// the router's first 2,048-token run.
    const SPLIT: usize = 2601;
    const KEPT: usize = 512;
    const _: () =
        assert!(!SPLIT.is_multiple_of(POOL) && !(SPLIT - POOL * (KEPT + 1) + 1).is_multiple_of(8));

    /// The walks (g)'s map refusals plant their maps on, each from
    /// position [`PREFIX`].
    #[derive(Clone, Copy)]
    enum Walk {
        Step,
        Pass,
        Gemm,
        Verify,
    }

    impl Walk {
        /// The walk as its refusal names it.
        fn name(self) -> &'static str {
            match self {
                Walk::Step => "step",
                Walk::Pass => "pass",
                Walk::Gemm => "ubatch",
                Walk::Verify => "verify",
            }
        }

        /// The call at [`PREFIX`]: a graph step, an eager pass of three
        /// rows, a ubatch of the rest of `toks`, a graph verify of two rows;
        /// its last row's token.
        fn run(self, m: &mut Qwen38Model, toks: &[u32], rows: [u32; 2]) -> Result<u32, GpuError> {
            match self {
                Walk::Step => m.step(&toks[PREFIX..PREFIX + 1]),
                Walk::Pass => m.prompt38(&toks[PREFIX..PREFIX + 3], Prompt38::Pass),
                Walk::Gemm => m.prompt38(&toks[PREFIX..], Prompt38::Gemm),
                Walk::Verify => m.step_rows::<2>(rows).map(|t| t[1]),
            }
        }
    }

    /// The layer (g)'s map refusals plant their map on: the real file's layer
    /// 7, or the file's last layer where it has fewer.
    const PLANT_LAYER: usize = 7;

    /// (g) the map refusals: with a slot map planted (`Body38::plant_slot_map`)
    /// that holds expert 3 of the planted layer on the tier card, each walk's call after
    /// the prefix is refused by name — the walk, the layer and the count —
    /// before anything moves, the position kept and the model not poisoned.
    /// With the plant taken back the same call runs. The plant is taken back
    /// before anything else can fail, so no line leaves it armed.
    fn map_refusals(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let mode = m.mode();
        let (rows, _) = verify_rows(toks)?;
        let rows = [rows[0], rows[1]];
        let at = PLANT_LAYER.min(n_layer() - 1);
        let planted = |entry: u32| -> Result<SlotMap, GateError> {
            let mut entries = vec![HOST; n_layer() * N_EXPERT];
            entries[at * N_EXPERT + 3] = entry;
            Ok(SlotMap::from_rows(0..n_layer(), N_EXPERT, entries)?)
        };
        let tier = planted(Slot::Tier { tier: 0, slot: 0 }.entry()?)?;
        let cases = [
            (Walk::Step, &tier, "tier card", "tier leg"),
            (Walk::Pass, &tier, "tier card", "tier leg"),
            (Walk::Gemm, &tier, "tier card", "tier leg"),
            (Walk::Verify, &tier, "tier card", "tier leg"),
        ];
        let mut ok = true;
        for (w, map, device, leg) in cases {
            Prefix::Steps(&toks[..PREFIX]).feed(m)?;
            let before = m.pos();
            m.body_parts("map_refusals")?
                .2
                .plant_slot_map(Some(map.clone()))?;
            let got = w.run(m, toks, rows);
            m.body_parts("map_refusals")?.2.plant_slot_map(None)?;
            let want = format!(
                "the {} walk has no {leg}: layer {at} holds 1 routed experts on the {device}",
                w.name()
            );
            let named = matches!(&got, Err(e) if e.to_string().contains(&want));
            let pos = m.pos();
            let kept = pos == before && m.poisoned().is_none();
            let again = w.run(m, toks, rows);
            let line_ok = named && kept && again.is_ok();
            ok &= line_ok;
            println!(
                "map refusal, {}: a slot map with expert 3 of layer {at} on the {device} at \
                 position {before} -> {}; position {pos} (kept, not poisoned: {kept}); the same \
                 call with the plant taken back: {} {}",
                w.name(),
                text(&got),
                text(&again),
                verdict(line_ok)
            );
        }
        m.reset()?;
        m.set_mode(mode);
        Ok(ok)
    }

    /// (g) `auto`: the pass below [`Prompt38::GEMM_FROM`] positions and the
    /// ubatch from it.
    fn gemm_auto() -> bool {
        let at = Prompt38::GEMM_FROM;
        let ok = Prompt38::Auto.resolve(at - 1) == Prompt38::Pass
            && Prompt38::Auto.resolve(at) == Prompt38::Gemm
            && Prompt38::Gemm.resolve(1) == Prompt38::Gemm;
        println!(
            "gemm auto: {} positions -> {}, {at} -> {} {}",
            at - 1,
            Prompt38::Auto.resolve(at - 1).name(),
            Prompt38::Auto.resolve(at).name(),
            verdict(ok)
        );
        ok
    }

    // --------------------------------------------------- (k) the card leg

    /// `f` on `m`, its error a named FAIL line (the model reset behind it)
    /// rather than the gate's end: a clause that failed does not stop the
    /// ones after it.
    fn guarded(
        label: &str,
        m: &mut Qwen38Model,
        f: impl FnOnce(&mut Qwen38Model) -> Result<bool, GateError>,
    ) -> Result<bool, GateError> {
        match f(m) {
            Ok(ok) => Ok(ok),
            Err(e) => {
                println!("card leg {label}: FAILED: {e} {}", verdict(false));
                m.reset()?;
                Ok(false)
            }
        }
    }

    /// (k) structure under the card plan, against the plan's own counts: the
    /// body runs the leg on exactly the layers the plan holds routed experts
    /// on, the captured step holds the header's node count plus the leg's
    /// five launches on each of them, [`memops`] of them batch mem ops, as
    /// the program counts. In the fixture tier the plan is also held to its
    /// contract: the header's budget puts half the experts
    /// (`n_expert / 2`) on the card of every card layer — a budget one
    /// granule off leaves a layer one expert short.
    fn card_structure(m: &mut Qwen38Model, plan: &CardPlan) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let body = m.body("card_structure")?;
        let (layers, (counted, memops_counted)) = (body.card_layers(), body.step_launches());
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let want_nodes = shape().nodes_decode_card(plan.layers);
        let want_memops = memops();
        let ok = layers == plan.layers
            && nodes == want_nodes
            && counted == want_nodes
            && memops_counted == want_memops
            && b == want_memops
            && k == want_nodes - want_memops
            && other == 0;
        println!(
            "card leg structure: card layers {layers} (want {}, the plan's); decode graph_nodes=\
             {nodes} (want {want_nodes}; the program counts {counted}, {memops_counted} of them \
             batch_mem_op) kernel={k} batch_mem_op={b} (want {want_memops}) other={other} {}",
            plan.layers,
            verdict(ok)
        );
        let held = |on: &dyn Fn(u64) -> bool| plan.n_l.iter().filter(|&&n| on(n)).count();
        let fixture = q38::tier()? == bloomery_gpu_gates::tier::Tier::Fixture;
        let want_each = (cfg().n_expert / 2) as u64;
        let contract = !fixture || plan.n_l.iter().all(|&n| n == want_each);
        println!(
            "card leg plan: {} of {} layers hold routed experts on the card, a layer's count \
             {:?}..{:?}{} {}",
            held(&|n| n > 0),
            plan.n_l.len(),
            plan.n_l.iter().min(),
            plan.n_l.iter().max(),
            if fixture {
                format!(" (want {want_each} on every layer: half the experts, the header's budget)")
            } else {
                " (the real file's counts are `qwen4exp_meta`'s CARD_PLANS)".to_string()
            },
            verdict(contract)
        );
        let witnessed = card_witnesses(plan.layers);
        Ok(ok && contract && witnessed)
    }

    /// PIN(2026-09-29): the card plan's layer outputs against the host
    /// plan's where both take the same experts, derived before the leg was
    /// built. The two differ only in the card experts' activations: the
    /// card's gate·up reads its input as q8_1 of 128 values a scale (the
    /// q8_1 quantizer's block), the host's as q8_2 of 32 (`block_q8_2_x4`);
    /// both read the SwiGLU output as 32-value q8 blocks. A q8 block of `n`
    /// values errs by at most `crest/(127·√12)` relative, the crest at most
    /// √n: [`q8_32_rel`] = q at 32 values, 2q at 128. The two sides' errors
    /// are independent, so an expert's output differs by at most
    /// `√((2q)² + q² + q² + q²) = √7·q` relative, the card slots' share of
    /// the routed sum at most all of it, 3.402e-2 a layer; layers add
    /// independently, √(l + 1) of it at layer `l`, and the head reads the
    /// last layer's, √48 · 3.402e-2 = 0.236.
    fn card_band(l: usize) -> f64 {
        ((l + 1) as f64).sqrt() * 7f64.sqrt() * q8_32_rel()
    }

    /// A place the places rule's map holds for expert `e` of its read row:
    /// [`HOST`] for every third expert, a card slot for the rest.
    fn place_of(e: usize) -> u32 {
        if e.is_multiple_of(3) {
            HOST
        } else {
            (e as u32 * 7) % 300
        }
    }

    /// (k) the places entry's own rule (`HandoffKernels::enqueue_places_cols`,
    /// the pass's places), at layer 9's sink: two columns of routed ids at the
    /// router's pitch of eleven, each column's eleventh word the shared
    /// expert's id [`N_EXPERT`], column 1's slot 3 an id past the experts, a
    /// map of two rows read at row 1, a guard word past the twenty places.
    /// Every place is the map's at its id, the bad id's [`HOST`], the guard
    /// untouched, and `expert_id` alone raised at layer 9 — once the pitch's
    /// eleventh word were read it would raise too and move a place.
    fn places_rule(m: &mut Qwen38Model) -> Result<bool, GateError> {
        const PITCH: usize = N_USED + 1;
        const GUARD: u32 = 0xdead_beef;
        let gpu = m.gpu();
        let stream = gpu.stream();
        let k = HandoffKernels::load(gpu.context())?;
        let mut ids = vec![0u32; 2 * PITCH];
        for c in 0..2 {
            for e in 0..N_USED {
                ids[c * PITCH + e] = ((37 * (c * N_USED + e) + 11) % N_EXPERT) as u32;
            }
            ids[c * PITCH + N_USED] = N_EXPERT as u32;
        }
        ids[PITCH + 3] = N_EXPERT as u32 + 88;
        let mut map = vec![7u32; 2 * N_EXPERT];
        for (e, p) in map[N_EXPERT..].iter_mut().enumerate() {
            *p = place_of(e);
        }
        let ids_d = DeviceBuffer::from_host(stream, &ids)?;
        let map_d = DeviceBuffer::from_host(stream, &map)?;
        let mut sel = DeviceBuffer::from_host(stream, &[GUARD; 2 * N_USED + 1])?;
        k.enqueue_places_cols(
            stream,
            &Places {
                ids: &ids_d,
                map: &map_d,
                row_off: N_EXPERT,
                n_expert: N_EXPERT,
            },
            PITCH,
            2,
            gpu.layer_sink(9)?,
            &mut sel,
        )?;
        stream.synchronize()?;
        let got = sel.to_host_vec(stream)?;
        let want: Vec<u32> = (0..2 * N_USED)
            .map(|k| {
                let id = ids[k / N_USED * PITCH + k % N_USED] as usize;
                if id < N_EXPERT { place_of(id) } else { HOST }
            })
            .chain([GUARD])
            .collect();
        let fault = gpu.take_fault()?;
        let raised = fault.is_some_and(|f| {
            f.layer == 9
                && f.site() == Some(FaultSite::ExpertId)
                && f.sites == 1 << FaultSite::ExpertId as u32
        });
        let places_ok = got == want;
        let ok = places_ok && raised;
        println!(
            "card leg places rule: two columns at a pitch of {PITCH}, column 1's slot 3 id {}: \
             places the map's, the bad id's HOST, the guard untouched: {places_ok} (got {got:?}); \
             the fault word {fault:?} (want expert_id alone at layer 9) {}",
            N_EXPERT + 88,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (k) (i): the batch set's tokens as eager steps under the card plan
    /// against the host plan's (`host`, the eager run of (p)): layer 0's
    /// routes bit for bit the host run's (its router reads what the same
    /// launches wrote from the same inputs); each flip of
    /// the card run's routing against the host run's excused as (g)'s free
    /// arm excuses one, at [`card_band`]; off every flip's path each layer
    /// output within [`card_band`] of the host run's, and each position's
    /// logits within the last layer's band with its argmax the host run's —
    /// or, named and counted, the host run's runner-up with the host run's
    /// margin between the two within twice the two runs' distance at those
    /// ids, as (t) excuses a tie; on a flip's path printed.
    fn card_vs_host(m: &mut Qwen38Model, toks: &[u32], host: &Run) -> Result<bool, GateError> {
        let card = run_steps(m, toks, StepMode::Eager, true)?;
        // Layer 0's router reads what the same launches wrote from the same
        // inputs in both plans: its route cannot differ.
        let first_same = card.routes.len() == host.routes.len()
            && card.routes.iter().zip(&host.routes).all(|(a, b)| {
                matches!((a.first(), b.first()), (Some(x), Some(y))
                    if x.ids == y.ids && same_bits(&x.logits, &y.logits))
            });
        println!(
            "card leg vs host: layer 0's routes bit for bit the host plan's at every position: \
             {first_same} {}",
            verdict(first_same)
        );
        let tally = judge_flips("card", &card.routes, &host.routes, card_band);
        let mut ok = first_same && tally.ok() && card.routes.len() == host.routes.len();
        println!(
            "card leg vs host: {} flip(s) of the card plan's routing against the host plan's, {} \
             allowed by the pair rule; {} first on their path, {} of them at a wide margin {}",
            tally.flips.len(),
            tally.flips.len() - tally.refused,
            tally.first,
            tally.wide,
            verdict(tally.ok())
        );
        let row = STREAMS * HIDDEN;
        let path = |l: usize, t: usize| {
            tally
                .flips
                .iter()
                .any(|(f, _, _)| f.layer <= l && f.token <= t)
        };
        let (mut held, mut past, mut worst) = (0usize, Vec::new(), 0.0f64);
        for (t, (a, b)) in card.taps.iter().zip(&host.taps).enumerate() {
            for l in 0..n_layer() {
                let e = match (a.get(l * row..(l + 1) * row), b.get(l * row..(l + 1) * row)) {
                    (Some(x), Some(y)) => rel(x, y),
                    _ => f64::INFINITY,
                };
                if path(l, t) {
                    continue;
                }
                held += 1;
                worst = worst.max(e / card_band(l));
                if e > card_band(l) {
                    past.push((l, t, e));
                }
            }
        }
        let layers_ok = past.is_empty() && card.taps.len() == host.taps.len();
        ok &= layers_ok;
        println!(
            "card leg vs host: {held} layer outputs off every flip's path, their worst distance \
             {worst:.3} of the band (card_band(l)); past it {:?} {}",
            &past[..past.len().min(8)],
            verdict(layers_ok)
        );
        let band = card_band(n_layer() - 1);
        let mut ties = 0usize;
        for (t, (a, b)) in card.logits.iter().zip(&host.logits).enumerate() {
            let (top, want) = (argmax(a), argmax(b));
            let runner = second(b, want);
            let margin = f64::from(b[want as usize]) - f64::from(b[runner as usize]);
            let dist = [want, runner]
                .iter()
                .map(|&i| (f64::from(a[i as usize]) - f64::from(b[i as usize])).abs())
                .fold(0.0, f64::max);
            let tie = top != want && top == runner && margin <= 2.0 * dist;
            let d = rel(a, b);
            let on = tally.on_path(t);
            ties += usize::from(tie && !on);
            let pos_ok = on || ((top == want || tie) && d <= band);
            ok &= pos_ok;
            println!(
                "card leg vs host position {t}: argmax card={top} host={want} (the host's \
                 runner-up {runner}, margin {margin:.4}, the runs' distance at the two \
                 {dist:.4}{}), logits_rel {d:.3e} (band {band:.3e}){} {}",
                if tie { ", a named tie" } else { "" },
                if on {
                    " on a flip's path (printed)"
                } else {
                    ""
                },
                verdict(pos_ok)
            );
        }
        println!("card leg vs host: {ties} named tie(s)");
        ok &= card.logits.len() == host.logits.len() && card.logits.len() == toks.len();
        Ok(ok)
    }

    /// (k) (ii): under the card plan the eager steps are the graph steps'
    /// bits, a pass of the batch set's five rows leaves the steps' last token,
    /// logits and every store bit for bit, and a verify of 2, 3 and 4 rows
    /// kept whole returns every row's token and logits bit for bit the
    /// steps' — the leg's kernels are independent per slot and column; each
    /// verify's graph holds the header's verify nodes plus the leg's five
    /// launches on each of the plan's card layers.
    fn card_rows(m: &mut Qwen38Model, toks: &[u32], plan: &CardPlan) -> Result<bool, GateError> {
        let eager = run_steps(m, toks, StepMode::Eager, true)?;
        let graph = run_steps(m, toks, StepMode::Graph, true)?;
        let mut ok = same_run(
            "card plan: five eager steps vs five graph replays",
            &eager,
            &graph,
            false,
        );
        let pass = run_pass(m, toks)?;
        ok &= same_run(
            "card plan: one pass of five rows vs five graph steps",
            &pass,
            &graph,
            true,
        );
        let want = shape().nodes_verify_card(plan.layers);
        ok &= verify_structure::<2>(m, want)?;
        ok &= verify_structure::<3>(m, want)?;
        ok &= verify_structure::<4>(m, want)?;
        let (rows, next) = verify_rows(toks)?;
        let refs = (1..=LANES)
            .map(|k| ref_run(m, Prefix::Steps(&toks[..PREFIX]), &[&rows[..k]], next))
            .collect::<Result<Vec<_>, _>>()?;
        ok &= verify_case::<2>(m, toks, rows, next, 2, &refs)?;
        ok &= verify_case::<3>(m, toks, rows, next, 3, &refs)?;
        ok &= verify_case::<4>(m, toks, rows, next, 4, &refs)?;
        Ok(ok)
    }

    /// (k) (iii): under the card plan the ubatch walk — the card route's —
    /// against the pass of the same plan, (g)'s forced arm: the pass's own
    /// routes planted (`eager`'s taps), so no flip can separate the two and
    /// the error model alone does — every tapped id the pass's, the last
    /// argmax equal to ik's, the last logits and every layer's live store
    /// and the PLE ring within [`gemm_band`] of the pass's (the card route
    /// quantizes the same blocks the pass's card leg quantizes — a q8_1 of
    /// 128 values for the gate·up's input, a 32-value q8_1 of the SwiGLU —
    /// so the two card legs differ only in their sums' order). Then the
    /// batch set cut in two, and D3K's prefill cut at [`SPLIT`], against
    /// one ubatch of each, bit for bit: a token's bits depend neither on
    /// the ubatch it lands in nor on the route's run (D3K's 3,001 tokens
    /// run two) — the routing free, as (g)'s D3K clause runs it.
    fn card_gemm(m: &mut Qwen38Model, man: &RefManifest, toks: &[u32]) -> Result<bool, GateError> {
        let eager = run_steps(m, toks, StepMode::Eager, true)?;
        let pass = run_pass(m, toks)?;
        let n = toks.len();
        let (forced, routes) = gemm_walk(m, toks, Some(&eager.routes))?;
        gemm_taps_off(m)?;
        let mut taken = 0usize;
        for (ours, theirs) in routes.iter().zip(&eager.routes) {
            for (o, p) in ours.iter().zip(theirs) {
                taken += usize::from(o.ids == p.ids);
            }
        }
        let cells = n * n_layer();
        let ids_ok = routes.len() == n && taken == cells;
        let mut ok = ids_ok;
        println!(
            "card leg gemm: the planted routes' ids taken at {taken} of {cells} (layer, position) \
             cells {}",
            verdict(ids_ok)
        );
        let vocab = m.body("card_gemm")?.vocab();
        let (last, top) = (forced.tokens[0], argmax(&forced.logits[0]));
        let (ik_ok, ik_text) = ik_argmax(man, vocab, "(k)(iii) card gemm: the last argmax", top)?;
        let argmax_ok = ik_ok && last == top;
        ok &= argmax_ok;
        let logits_rel = rel(
            &forced.logits[0],
            pass.logits.last().ok_or("no pass logits")?,
        );
        let head_band = gemm_band(n_layer() - 1);
        let logits_ok = logits_rel <= head_band;
        ok &= logits_ok;
        println!(
            "card leg gemm: last argmax ours={top} (returned {last}) ik={ik_text}; last logits' \
             distance from the pass's {logits_rel:.3e} (band {head_band:.3e}) {}",
            verdict(argmax_ok && logits_ok)
        );
        let (layers, ple) = live_rel(&forced, &pass, n);
        let past: Vec<usize> = layers
            .iter()
            .enumerate()
            .filter(|&(l, &e)| e > gemm_band(l) || e.is_nan())
            .map(|(l, _)| l)
            .collect();
        let ple_band = gemm_band(ple_layer()?);
        let ple_ok = ple <= ple_band;
        let stores_ok = layers.len() == n_layer() && past.is_empty() && ple_ok;
        ok &= stores_ok;
        println!(
            "card leg gemm: every layer's store within its band, past it at {past:?}; PLE ring \
             {ple:.3e} (band {:.3e}) {}",
            ple_band,
            verdict(stores_ok)
        );
        // The batch set, one free walk and two cut, bit for bit.
        let one = gemm_walk(m, toks, None)?.0;
        gemm_taps_off(m)?;
        fresh(m)?;
        m.prompt38(&toks[..2], Prompt38::Gemm)?;
        let last = m.prompt38(&toks[2..], Prompt38::Gemm)?;
        let logits = m.logits()?;
        let (two_stores, two_ring) = stores(m)?;
        let cut = Run {
            tokens: vec![last],
            logits: vec![logits],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: two_stores,
            ple_ring: two_ring,
        };
        ok &= same_run(
            "card plan: the batch set by two ubatches vs one ubatch",
            &cut,
            &one,
            true,
        );
        // D3K's prefill, one walk and two cut inside the selecting rows, bit
        // for bit — the only walk that crosses the route's run boundary.
        let d3k = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, _, prefill) = d3k.step()?;
        let prefill = prefill.to_vec();
        fresh(m)?;
        let last = m.prompt38(&prefill, Prompt38::Gemm)?;
        let (one_stores, one_ring) = stores(m)?;
        let one = Run {
            tokens: vec![last],
            logits: vec![m.logits()?],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: one_stores,
            ple_ring: one_ring,
        };
        fresh(m)?;
        m.prompt38(&prefill[..SPLIT], Prompt38::Gemm)?;
        let last = m.prompt38(&prefill[SPLIT..], Prompt38::Gemm)?;
        let (cut_stores, cut_ring) = stores(m)?;
        let cut = Run {
            tokens: vec![last],
            logits: vec![m.logits()?],
            taps: Vec::new(),
            routes: Vec::new(),
            stores: cut_stores,
            ple_ring: cut_ring,
        };
        ok &= same_run(
            &format!("card plan: {D3K} by ubatches cut at {SPLIT} vs one ubatch"),
            &cut,
            &one,
            true,
        );
        m.reset()?;
        Ok(ok)
    }

    /// (k): the card plan loaded on the gate card (the host plan's model
    /// dropped first), then its structure, the places entry's rule, (i)
    /// against `host`, (ii) and (iii).
    fn card_leg(toks: &[u32], host: &Run, man: &RefManifest) -> Result<bool, GateError> {
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let Opened {
            model: mut m,
            card: plan,
            ..
        } = match open_at(Experts::Card, 1, ub, None) {
            Ok(o) => o,
            Err(e) => {
                println!(
                    "card leg: the card plan's load FAILED: {e} {}",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let mut ok = guarded("structure", &mut m, |m| card_structure(m, &plan))?;
        ok &= guarded("places rule", &mut m, places_rule)?;
        ok &= guarded("vs host", &mut m, |m| card_vs_host(m, toks, host))?;
        ok &= guarded("rows", &mut m, |m| card_rows(m, toks, &plan))?;
        ok &= guarded("gemm", &mut m, |m| card_gemm(m, man, toks))?;
        Ok(ok)
    }

    // ---------------------------------------------------- (r) refusals

    fn refusals(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let mut ok = true;
        // The four paths parse; the refusal is a name none of them is,
        // which the parser answers by naming the four.
        for s in ["step", "pass", "gemm", "auto"] {
            let got = Prompt38::parse(s);
            let named = matches!(&got, Ok(p) if p.name() == s);
            ok &= named;
            println!(
                "refusal: --prefill {s} -> {} {}",
                match &got {
                    Ok(p) => format!("accepted as {}", p.name()),
                    Err(e) => e.to_string(),
                },
                verdict(named)
            );
        }
        let got = Prompt38::parse("wide");
        let named = matches!(&got, Err(e) if e.to_string().contains("`gemm` or `auto`"));
        ok &= named;
        println!(
            "refusal: --prefill wide -> {} {}",
            match &got {
                Ok(p) => format!("accepted as {}", p.name()),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        m.reset()?;
        m.step(&[0])?;
        // The placeholder is the header's (`ple.image_token_id`).
        let image = shape()
            .image_token
            .ok_or("the header lists no image placeholder (ple.image_token_id)")?;
        // A step refuses its one token; a pass refuses the whole pass before
        // any row of it runs.
        for (path, ids) in [
            (Prompt38::Step, &[image][..]),
            (Prompt38::Pass, &[1, image][..]),
            (Prompt38::Gemm, &[1, image][..]),
        ] {
            let before = m.pos();
            let got = m.prompt38(ids, path);
            let named =
                matches!(&got, Err(e) if e.to_string().contains("is the image placeholder"));
            let kept = m.pos() == before && m.poisoned().is_none();
            ok &= named && kept;
            println!(
                "refusal: the image placeholder {image} by {} at position {before} -> {}; \
                 position {} (kept: {kept}) {}",
                path.name(),
                match &got {
                    Ok(t) => format!("accepted, next {t}"),
                    Err(e) => e.to_string(),
                },
                m.pos(),
                verdict(named && kept)
            );
        }
        let before = m.pos();
        let past = vec![0u32; CTX + 1 - before as usize];
        for path in [Prompt38::Step, Prompt38::Pass, Prompt38::Gemm] {
            let got = m.prompt38(&past, path);
            let named = matches!(&got, Err(e) if e.to_string().contains("passes the stores"));
            let kept = m.pos() == before;
            ok &= named && kept;
            println!(
                "refusal: {} ids by {} from position {before} past {CTX} -> {}; position {} \
                 (kept: {kept}) {}",
                past.len(),
                path.name(),
                match &got {
                    Ok(t) => format!("accepted, next {t}"),
                    Err(e) => e.to_string(),
                },
                m.pos(),
                verdict(named && kept)
            );
        }
        Ok(ok)
    }

    // ------------------------------------------------- (y) resident slots

    /// Set `name`'s prefill ids.
    fn prefill_of(name: &str) -> Result<Vec<u32>, GateError> {
        let man = RefManifest::open(&data_dir().join(name), &IK)?;
        let (_, _, prefill) = man.step()?;
        Ok(prefill.to_vec())
    }

    /// The digest of the selected slot's whole resident state: every store
    /// and the PLE ring ([`stores`]) and the lane word — the state hash a
    /// slot's solo run and its interleaved run are read against each other
    /// by ([`SlotsAdapter::state_hash`]).
    fn slot_digest(m: &mut Qwen38Model) -> Result<u64, GateError> {
        let (stores, ring) = stores(m)?;
        let lane = m.body("slots")?.lane();
        let mut h = Fnv1a64::default();
        let u16s = |h: Fnv1a64, v: &[u16]| v.iter().fold(h, |h, &w| h.bytes(&w.to_le_bytes()));
        for s in &stores {
            match s {
                Store38Host::Rec { state, ring } => h = h.f32s(state).f32s(ring),
                Store38Host::Qsa { k, v, raw, pooled } => {
                    h = u16s(u16s(u16s(u16s(h, k), v), raw), pooled);
                }
            }
        }
        Ok(h.f32s(&ring).bytes(&lane.to_le_bytes()).value())
    }

    /// (y)'s body for the slot harness ([`slots_gate`]): the load of
    /// [`open_slots`], stream 0 window A by the pass path and stream 1
    /// window B by the ubatch path, each from [`fresh`] (the selecting
    /// planes filled, which [`slot_digest`] reads whole).
    struct Slots38 {
        experts: Experts,
        a: Vec<u32>,
        b: Vec<u32>,
    }

    impl SlotsAdapter for Slots38 {
        type Body = Body38;

        const STEPS: usize = 8;
        const TAIL: usize = 4;

        fn open(&self, slots: usize) -> Result<Qwen38Model, GateError> {
            open_slots(self.experts, slots)
        }

        fn rewind(&self, m: &mut Qwen38Model) -> Result<(), GateError> {
            fresh(m)
        }

        fn prompt(&self, m: &mut Qwen38Model, stream: usize) -> Result<u32, GateError> {
            Ok(match stream {
                0 => m.prompt38(&self.a, Prompt38::Pass)?,
                1 => m.prompt38(&self.b, Prompt38::Gemm)?,
                _ => return Err(format!("(y) runs streams 0 and 1, not {stream}").into()),
            })
        }

        fn state_hash(&self, m: &mut Qwen38Model) -> Result<u64, GateError> {
            slot_digest(m)
        }

        /// The stores ([`Body38::store_bytes`], held to the header's
        /// derivation by (s)), the lane word (one device word) and the two
        /// arena-row buffers (`(1 + PASS_ROWS) · STREAMS · HIDDEN` f32).
        fn seq_bytes_derived(&self, m: &Qwen38Model) -> Result<Derived, GateError> {
            let stores = m.body("slots")?.store_bytes();
            let rows = (1 + PASS_ROWS) * STREAMS * HIDDEN * 4;
            Ok(Derived {
                bytes: stores + 4 + rows,
                terms: format!("store_bytes {stores} + the lane word 4 + the arena rows {rows}"),
            })
        }

        /// H5's planter ([`SlotsAdapter::plant_refusal`]): the tier through
        /// the body's own mut path.
        fn plant_refusal(&self, m: &mut Qwen38Model) -> Result<bool, GateError> {
            m.body_parts("gate_qwen4exp_e2e slots")?
                .2
                .hybrid_mut()
                .plant_refusal("a planted refusal (the slots harness's seam)");
            Ok(true)
        }

        /// H5's round of several slots: the body's one pass
        /// ([`GpuModel::step_slots`], this body's [`SlotRows`]).
        fn step_all(&self, m: &mut Qwen38Model, last: &[u32]) -> Result<Vec<u32>, GpuError> {
            let ones: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
            let rows: Vec<(usize, &[u32])> = ones.iter().map(|t| &t[..]).enumerate().collect();
            Ok(m.step_slots(&rows)?.ids)
        }

        /// H5's window: the tier's own refusal, read through the body's
        /// tier.
        fn tier_poisoned(&self, m: &mut Qwen38Model) -> Result<bool, GateError> {
            Ok(m.body_parts("gate_qwen4exp_e2e slots")?
                .2
                .hybrid()
                .refuse_if_poisoned("slots H5")
                .is_err())
        }
    }

    impl PassAdapter for Slots38 {
        /// The header's slot nodes ([`Shape::nodes_slot`]).
        fn added_slot_launches(&self, _m: &Qwen38Model) -> Result<Launches, GateError> {
            let (gdn, qsa) = (n_gdn(), n_qsa());
            Ok(Launches {
                n: shape().nodes_slot(),
                terms: format!(
                    "the embedding 1 + the PLE conv 1 + {gdn} delta layers' 2 + {qsa} \
                     selecting layers' 8 + the parked rows' copy 1"
                ),
            })
        }

        fn slots_launches(
            &self,
            m: &Qwen38Model,
            key: &[(usize, usize)],
        ) -> Result<usize, GateError> {
            Ok(m.body("slots")?.slots_launches(key))
        }
    }

    /// (y) the plan's slot count against the owners of the per-sequence
    /// bytes (module doc): the plan of two grows the card's kv class by
    /// exactly the bytes one sequence holds — its stores and PLE ring
    /// (`runtime::stores`'s own terms, the ones the load holds the
    /// allocations to), the lane word (one device word) and the two
    /// arena-row buffers (`(1 + PASS_ROWS) · STREAMS · HIDDEN` f32,
    /// `seq38_bytes`' rows term) — and at one slot is the plan every other
    /// clause loads by.
    fn slots_plan(experts: Experts) -> Result<bool, GateError> {
        let seq = {
            use runtime::stores as st;
            let rec = st::recurrent_bytes(V_HEADS, K_HEADS, HEAD_V, CONV)
                + st::delta_lane_bytes(V_HEADS, HEAD_V, LANES);
            let sel = st::selecting_bytes(N_KV, HEAD, IDX_DIM, POOL, CTX);
            let ple = st::ple_ring_bytes(PLE_TAPS, PLE_DILATION, STREAMS, HIDDEN);
            let rows = (1 + PASS_ROWS) * STREAMS * HIDDEN * 4;
            u64::try_from(n_gdn())? * rec + u64::try_from(n_qsa())? * sel + ple + 4 + rows as u64
        };
        let file = open_split()?;
        let inputs = PlanInputs::describe(&file)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let machine = machine_for_experts(
            card()?,
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            experts,
        );
        let ctx = CTX as u64;
        let plan_levers = &cfg().plan_levers;
        let plain = inputs.plan_with(&machine, ctx, plan_levers, experts)?;
        let one = inputs.plan_with_slots(&machine, ctx, plan_levers, experts, 1)?;
        let two = inputs.plan_with_slots(&machine, ctx, plan_levers, experts, 2)?;
        let totals = |p: &model::placement::Plan| {
            let c = &p.cards[0];
            (
                c.dense_bytes,
                c.expert_bytes,
                c.rounding_bytes,
                c.kv_bytes,
                c.scratch_bytes,
                c.context_bytes,
            )
        };
        let same_at_one = totals(&plain) == totals(&one);
        let grown = two.cards[0].kv_bytes - one.cards[0].kv_bytes;
        let ok = same_at_one && grown == seq;
        println!(
            "slots plan: the plan of 2 grows the card's kv class by {grown} (one sequence's \
             stores, ring, lane word and arena rows: {seq}), and at 1 slot {} the plan the load \
             always makes {}",
            if same_at_one { "is" } else { "is not" },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (y) (module doc): the harness's contracts over the plan of two and,
    /// between its halves, the body's own clauses. The clause's model is
    /// dropped with its second sequence.
    fn slots_two(experts: Experts) -> Result<bool, GateError> {
        let body = Slots38 {
            experts,
            a: prefill_of(STEP4)?,
            b: prefill_of(D1K)?,
        };
        // PIN(2026-10-05): (y)'s interleave and bytes clauses moved to
        // slots_gate H1/H3; H4/H6/H7 (reset, select, captures) are new to (y),
        // the harness's for every body.
        let mut s = slots_gate::interleave(&body)?;
        s.one_pass();
        // A third sequence past the plan is refused by name.
        let past = match s.model().add_slots(3) {
            Err(e) => e.to_string().contains("of a plan that counts 2 slots"),
            Ok(()) => false,
        };
        println!(
            "slots past the plan: add_slots(3) on a plan of 2 named the plan and the ask {past} {}",
            verdict(past)
        );
        // A verify waiting for its commit refuses a select by name, moving
        // nothing; its commit (a rollback of slot 1) then stands.
        let t = s.last(1)?;
        let m = s.model();
        m.select_slot(1)?;
        let at = m.pos();
        m.step_rows::<4>([t, t, t, t])?;
        let before = (m.pos(), m.body("slots")?.lane());
        let named = match m.select_slot(0) {
            Err(e) => e.to_string().contains("waits for its commit"),
            Ok(()) => false,
        };
        let kept = (m.pos(), m.body("slots")?.lane()) == before;
        let pending_ok = named && kept;
        println!(
            "slots pending: a select with slot 1's verify waiting named it {named}, the position \
             and lane kept {kept} {}",
            verdict(pending_ok)
        );
        m.rollback(at + 2)?;
        // The rollback of slot 1 leaves slot 0's continuation its solo run's.
        let iso_ok = s.continues(0)?;
        println!(
            "slots rollback isolation: slot 1's verify committed 2 of its 4 rows; slot 0's next \
             {} ids {} its solo run's continuation {}",
            Slots38::TAIL,
            if iso_ok { "equal" } else { "differ from" },
            verdict(iso_ok)
        );
        let harness_ok = s.finish()?;
        Ok(past && pending_ok && iso_ok && harness_ok)
    }

    /// The (z) arms' ubatch: D3K's 3,001-position prefill runs as one call of
    /// three ubatches (1,024, 1,024, 953), so the NVMe arm's first ubatch
    /// reads its own rows and the two after it are read ahead.
    const ZUB: usize = 1024;

    /// One (z) arm's runs: the batch set as graph steps from a reset (with the
    /// step thread's faults across them) and as one eager pass, and D3K's
    /// prefill as one ubatch call with the graph step after it.
    fn z_runs(
        m: &mut Qwen38Model,
        toks: &[u32],
        prefill: &[u32],
    ) -> Result<([Run; 3], engram::Faults), GateError> {
        let before = engram::faults_thread();
        let steps = run_steps(m, toks, StepMode::Graph, true)?;
        let after = engram::faults_thread();
        let faults = engram::Faults {
            major: after.major - before.major,
            minor: after.minor - before.minor,
        };
        let pass = run_pass(m, toks)?;
        fresh(m)?;
        let last = m.prompt38(prefill, Prompt38::Gemm)?;
        let first = m.logits()?;
        let next = m.step(&[last])?;
        let logits = vec![first, m.logits()?];
        let (stores, ple_ring) = stores(m)?;
        let gemm = Run {
            tokens: vec![last, next],
            logits,
            taps: Vec::new(),
            routes: Vec::new(),
            stores,
            ple_ring,
        };
        Ok(([steps, pass, gemm], faults))
    }

    /// (z) The PLE table read from the NVMe tier: the host plan with the
    /// host's room given as the host arm's need less one byte, so the rule
    /// leaves the table on the NVMe tier, against the same plan at the
    /// reading's room, the table on the host; both at a ubatch of [`ZUB`].
    /// Each plan line names its tier; the NVMe arm's every token, logit and
    /// store is the host arm's bit for bit — the same rows decoded, only
    /// where they are read from moves — on the graph steps, the eager pass
    /// and the ubatch call whose later ubatches are read ahead. The step
    /// thread's faults across each arm's graph steps are printed.
    fn ple_nvme(toks: &[u32]) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let prefill = prefill.to_vec();
        let file = open_split()?;
        let inputs = PlanInputs::describe(&file)?;
        // The host arm's need is read off the plan the loads below make: the
        // same card ([`card`], not a card by name), the same levers.
        let machine = machine_for_experts(
            card()?,
            inputs.spec.layers.len(),
            u64::try_from(ZUB)?,
            Experts::Host,
        );
        let host_need = HostNeed::of(
            &inputs.plan_with_slots(&machine, CTX as u64, &cfg().plan_levers, Experts::Host, 1)?,
            0,
        )
        .bytes();
        drop(inputs);
        drop(file);
        let Opened {
            model: mut m,
            ple_tier: host_tier,
            ..
        } = open_at(Experts::Host, 1, ZUB, None)?;
        let (want, host_faults) = z_runs(&mut m, toks, &prefill)?;
        drop(m);
        let Opened {
            model: mut m,
            ple_tier: nvme_tier,
            ..
        } = open_at(Experts::Host, 1, ZUB, Some(host_need - 1))?;
        let (got, nvme_faults) = z_runs(&mut m, toks, &prefill)?;
        drop(m);
        let tiers = host_tier == Some(Device::Host) && nvme_tier == Some(Device::Nvme);
        println!(
            "ple tiers: the reading's room puts the table on the {} tier, a room of the host \
             arm's need {host_need} B less one byte on the {} tier {}",
            tier_word(host_tier),
            tier_word(nvme_tier),
            verdict(tiers)
        );
        println!(
            "ple step thread faults across the batch set's graph steps: host arm {} major {} \
             minor, NVMe arm {} major {} minor (runtime values)",
            host_faults.major, host_faults.minor, nvme_faults.major, nvme_faults.minor
        );
        let mut ok = tiers;
        for ((label, last_only), (g, w)) in [
            ("ple nvme vs host: five graph steps", false),
            ("ple nvme vs host: one eager pass", true),
            (
                "ple nvme vs host: D3K by one call of three ubatches, and its step",
                false,
            ),
        ]
        .into_iter()
        .zip(got.iter().zip(&want))
        {
            ok &= same_run(label, g, w, last_only);
        }
        Ok(ok)
    }

    /// A self-consistency clause: it runs in both tiers, so a tier that
    /// deferred it would be refused by name here, not skipped.
    fn sc(name: &str) -> Result<(), GateError> {
        if q38::clause(name, Tag::SelfConsistency)? {
            Ok(())
        } else {
            Err(format!("the self-consistency clause {name:?} was deferred").into())
        }
    }

    pub fn run() -> Result<(), GateError> {
        let mut ok = init()?;
        let mut m = open(Experts::Host)?;
        sc("(s) structure: the header's layer kinds, store bytes and node counts")?;
        ok &= structure(&mut m)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        sc("(h) the head of m rows: finite rows, NaN faults")?;
        ok &= head_rows(&mut m, &toks)?;
        sc("(p) one program: graph = eager = pass, reset clears")?;
        let (paths_ok, eager) = paths(&mut m, &toks)?;
        ok &= paths_ok;
        let vocab = m.body("run")?.vocab();
        sc("(c) free: every route tap the top ten of its own logits")?;
        let (free_ok, firsts) = free(&man, &eager, vocab)?;
        ok &= free_ok;
        sc("(g) the ubatch walk on the batch set: taps, flips, forced arm, auto")?;
        ok &= gemm_batch(&mut m, &man, &toks, &eager)?;
        let last = *firsts.last().ok_or("no positions")?;
        let mut ties = 0usize;
        let band = Some((&toks[..], last));
        for (name, band) in [(STEP4, band), (STEP4_EVERY_NODE, band), (D1K, None)] {
            ok &= step_set(&mut m, name, Prompt38::Step, (band, false), &mut ties)?.0;
        }
        let (d3k_ok, d3k) = step_set(&mut m, D3K, Prompt38::Step, (None, true), &mut ties)?;
        ok &= d3k_ok;
        println!(
            "{D3K}: ik cuts its selection by cells and keeps up to three keys of the 513th pool \
             ours does not read (a named difference, its logits printed)"
        );
        println!("step sets: {ties} named tie(s)");
        sc("(q) D3K's prefill by passes = by steps")?;
        ok &= pass_selects(&mut m, &d3k)?;
        drop(d3k);
        sc("(g) D3K by ubatches: the cut, the walk's split, the timed record")?;
        ok &= gemm_d3k(&mut m, &mut ties)?;
        sc("(g) the map refusals and auto")?;
        ok &= map_refusals(&mut m, &toks)?;
        ok &= gemm_auto();
        println!("step sets and the ubatch's D3K step: {ties} named tie(s)");
        sc("(v) the verify: rows = steps, commits, structure, refusals")?;
        ok &= verify_clause(&mut m, &toks)?;
        sc("(o) one owner of the position")?;
        ok &= position_owner(&mut m, &toks)?;
        sc("(r) refusals")?;
        ok &= refusals(&mut m)?;
        drop(m);
        sc("(z) the PLE table on the NVMe tier = on the host")?;
        ok &= ple_nvme(&toks)?;
        sc("(y) resident slots: the plan's bytes and the harness's contracts")?;
        ok &= slots_plan(Experts::Host)?;
        ok &= slots_two(Experts::Host)?;
        sc("(k) the card leg: structure, places rule, vs the host plan, rows, ubatch walk")?;
        ok &= card_leg(&toks, &eager, &man)?;
        let (ran, deferred) = q38::tally();
        println!(
            "clauses: {ran} ran, {deferred} left to the real tier (deferred(real) lines above)"
        );
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
