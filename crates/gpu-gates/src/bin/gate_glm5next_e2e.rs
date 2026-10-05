//! The GLM-5.3-Flash end-to-end gate: the whole program — 34 KDA layers and
//! 11 latent-attention layers, every block in the four hyper-connection
//! streams, three dense blocks and 42 routed blocks whose experts run on the
//! card where the plan puts them (each layer's id prefix) and on the host
//! tier otherwise, the streams' mean, the q8_0 head
//! and the argmax — loaded
//! once by its placement on the gate card (`workstation::plan_gate`), against
//! ik's CPU oracle sets (`refset::arch::glm5next`: the 5-token batch set, the
//! step after a fused 4-token prefill, the same after a prefill run node by
//! node, the step after a fused 1,024-token prefill of the prose, and the
//! step after 3,070 of it with ik's k-pool indexer on, past the 2,051
//! positions a latent layer keeps whole), every set read through its family.
//!
//! What is asserted:
//! The load at [`CTX`] is the verify's (`app::arch::glm5next::open_pair`:
//! two KDA lanes, since (v) verifies on it); the prompt batch's load below
//! is the plain session's, one lane.
//! - (card) right after the load at [`CTX`], the card experts' slot map,
//!   slots and card copy, checks (i)–(iii) of `shared/glm5next_card.rs`.
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes
//!   plus [`CARD_NODES`] for each routed layer the plan gives card experts
//!   (its `n_l`, which the host tier's slot map must hold layer for layer),
//!   [`MEMOPS`] of them stream memory-operation batches (each routed layer's
//!   go and wait) and the rest kernels, and the program's own count
//!   (`step_launches` over the body's layer programs) is the same; the
//!   layers with card experts are routed layers whose three stacks the card
//!   reads — none of [`Q6K_DOWN`], whose downs are Q6_K — and, without a
//!   card budget (`BLOOMERY_CARD_BUDGET`), every one of those; the
//!   latent layers are the ones the file's description names, the dense
//!   blocks the first three; the load holds two KDA lanes, and the stores'
//!   bytes equal their derivation from the header at two ([`store_bytes`]).
//! - (p) one chain: the batch set's five tokens as five graph steps and as
//!   five eager steps (the per-layer taps armed), each from a reset, leave
//!   the same per-position tokens and logits bit for bit.
//! - (l) layer by layer: the batch set's five positions run through one
//!   layer at a time (`Body::forced_row`), each layer on our own previous
//!   layer's streams from the embedding, leave every layer output bit for
//!   bit the eager chain's tap: the forced arm runs what the chain runs.
//! - (c) free-running on the batch set, each routed layer's picks read from
//!   the layered run: a token whose chosen set differs from ik's
//!   `ffn_moe_topk-L` is a flip, allowed only when every exchanged pair's
//!   gap in ik's ranked values (`ffn_moe_probs_biased`) lies within our two
//!   values' error there and that error within [`FLIP_ERR_CAP`], a measured
//!   frontier ([`Flip::allowed`]), named and counted; void — a named FAIL —
//!   when (l) is red, since the picks are the layered run's. A flip at
//!   layer `L'` and position `t'` lies on the path of every layer output from
//!   `L'` on at `t'` and at every later position (the mixers' stores carry
//!   it), and no band here derives how far it moves them: each layer output
//!   off every flip's path against ik's `l_out-L` within [`FREE_BAND`], the
//!   outputs past a flip printed and counted; the last position's argmax
//!   equal to ik's `result_output` argmax. ik's margins, every routed layer
//!   by token, and the worst output off a same-position path are printed.
//! - (f) teacher-forced, per layer, on the batch set: the mixer sub-layer
//!   on ik's layer input (`hc_init`, then `l_out-(L−1)`), the feed-forward
//!   sub-layer on ik's streams after the mixer (`attn_out-L`), from a reset;
//!   the folds, the mixer's output, the streams' updates, the normed block
//!   input, the router's logits and ranked values, the weights matched by
//!   id and the block's output, each tap's error over its input's gap within
//!   [`RATIO_BAND`]; a flip allowed as in (c), its position's block output
//!   and layer output left out. Each layer's error on the streams is
//!   printed: [`FREE_BAND`]'s derivation input.
//! - (t) each step set: its prefill fed by our own steps from a reset, then
//!   the step; its argmax equal to ik's, or — named and counted, never
//!   silently — our argmax ik's runner-up, ik's own margin between the two
//!   inside twice the distance between our logits and ik's at those ids,
//!   and our whole logits row within [`HEAD_RATIO`] times the last layer's
//!   own distance from ik's ([`tie_allowed`]: the head carries its input's
//!   distance). The step's layer
//!   outputs against ik's `l_out-L` are printed, and held to [`FREE_BAND`]
//!   on the two 4-token sets only, below the first layer a flip lies on the
//!   path of the batch set's last position as (c) names them (those sets'
//!   prefill and step are the batch set's tokens, and our steps route them
//!   as (c)'s): ik's prefill is its own fused graph, not our steps, and
//!   after 1,024 positions the two states have drifted by an amount no band
//!   here derives. Each step set's logits print their FNV-1a digest: below
//!   2,051 positions the selector lists every position in order, so a tree
//!   that changes only the selector leaves the three short sets' digests
//!   where its base left them. The 3,070-position set (`--dsa`) is the only
//!   end-to-end run past the dense limit; ik's prefill lists position 0 in
//!   the empty tail slots of its rows past 2,051 whose tail is short, and
//!   ours does not (a named difference, `crate::mla`'s cut), so that set's
//!   drift carries it.
//!
//! - (h) the q8_0 head's fault: `output_norm.weight[0]` set to NaN, a step
//!   is the fault [`FaultSite::Logit`] at the head, not a token; the weight
//!   put back, a reset steps clean.
//! - (a) taps armed after a capture ([`set_taps`]): a graph step, then the
//!   taps armed, then a step, whose taps are the eager run's bit for bit —
//!   arming drops the capture, whose replays would copy nothing.
//! - (o) one owner of the position: a failure planted before a step's launch
//!   ([`Plant::BeforeLaunch`]) leaves the model at that position and the
//!   step then runs it, its logits the plain run's bit for bit; one planted
//!   after the launch leaves the model there too, the next step there
//!   refused by name (the recurrent stores hold it already) and nothing a
//!   cut keeps past the model's position.
//! - (v) the verify of two rows (`Rows`, `bloomery_gpu_glm5next`'s verify):
//!   its capture holds twice the step's nodes (each row the step's
//!   launches, the head included); on the batch set's tokens, in graph mode
//!   and in eager: from a reset
//!   and a step of token 0, a verify of tokens 1 and 2 gives the plain
//!   graph run's tokens and logits at positions 1 and 2 bit for bit; a step
//!   while it waits for its commit is refused by name and moves nothing;
//!   keeping both rows, the stores (each KDA layer's committed lane and
//!   ring, every latent layer's rows and pool plane) digest as three plain
//!   steps', and the step of token 3 gives the plain run's position 3. Then
//!   the draft rejected: a verify of token 1 and a wrong draft, row 0 kept
//!   alone, then the steps of tokens 2 and 3 give the plain run's positions
//!   2 and 3 and the stores after token 2 digest as the plain steps'. A
//!   verify whose row 1 wrote the committed lane in place fails the
//!   reject's step at 2 by its stamp, and one that left row 1's lane
//!   uncommitted fails the accept's step at 3. It runs last on the load
//!   ((pb-long) before it), and an error in it is its FAIL, not the gate's
//!   end.
//! - (k) checkpoints, on the prose set's ids through the session: a prompt
//!   of [`A`] ids takes its checkpoints at 512 and [`A`]; a cut keeps 512 for
//!   600 and nothing for 500, each with its reason's code, and a cut to 600
//!   is refused by name; a cut to 512 and a branch of 200 ids leave the
//!   points at 512 and 712 (the abandoned 640 dropped); a cut to 712 (a
//!   point the restored branch took, then a step: the copy back runs in the
//!   step) and a cut to 512 (below the abandoned point, then a prompt call:
//!   the copy back runs in its first take) then a tail of 64 ids each give
//!   the logits plain steps of the same ids from a reset give, bit for bit;
//!   and a prompt call's takes leave those logits as they are. All of it
//!   twice: on the steps feed, then on the batch feed in groups of two
//!   batches (`set_prefill_group`), where the prompt of [`A`] ids is one
//!   group whose take at 512 is opened before it and filled layer by layer
//!   inside it: the same points, and the logits the same calls give from a
//!   reset with no cut (a batch past a chunk runs the GEMM, whose bits are
//!   not the steps'; the same calls cut the same batches).
//! - (pb-long) the prompt batch past the positions the latent layers attend
//!   whole, on the same load: [`LONG`] lcg ids as one batch call from a
//!   reset, then as two batch calls from a reset, the first of [`LONG_CUT`]
//!   ids so that the second starts inside a pool, the prompt itself ending
//!   inside one, every batch of either past a chunk; every store, the latent
//!   layers' pool planes included, the last logits and the argmax bit for bit
//!   the one call's after the prompt, then [`GREEDY`] greedy steps from there
//!   each giving the one call's token and logits bit for bit, and every store
//!   bit for bit again after them (a greedy step completes the prompt's last
//!   pool). Then after a call of [`LONG_TAIL`] ids, the rest to [`LONG`] — the
//!   selector's positions — by graph steps, and again from the call's
//!   checkpoint in calls of a chunk: every store, the logits and the argmax
//!   bit for bit the steps'.
//!
//! The prompt batch, on a second load at [`CTX_PP`] positions — every
//! position the latent layers attend whole — in its own session fed in
//! batches (`bloomery_gpu_glm5next::prefill`), the clauses above having run
//! on the steps feed:
//! - (l1) the plain load holds one KDA lane and the stores' bytes are their
//!   derivation at one ([`store_bytes`]); after a step, a verify of two rows
//!   is refused by name, eager and in its capture, the model standing where
//!   it stood with every store as it was, and the step after it the plain
//!   run's.
//! - (pb) one run of plain steps over [`CTX_PP`] lcg ids from a reset, every
//!   store digested and the logits and argmax kept after each of [`PP`]
//!   positions; then the reference: the same ids from a reset in batch calls
//!   of at most [`CHUNK`] positions, each ending at the next count of [`PP`],
//!   with the route taps armed (`set_prompt_route_taps`) — every KDA layer's
//!   state and conv ring, every latent layer's latent and index rows and pool
//!   plane, the last logits and the argmax bit for bit the steps' at each
//!   count (a batch of at most a chunk runs the step's gemvs), its stores'
//!   live values and every routed layer's scores and picks kept. At groups
//!   of one, a batch call of each count from a reset: under [`GEMM_FROM`]
//!   the steps' bits as above; from it (the GEMM's projections, int8
//!   activations) against the reference: every routed pick at every position
//!   with no earlier flip on its path judged by the pair rule under
//!   [`route_cap`] and a gap within [`route_margin_cap`], those past an
//!   earlier flip counted against the cap and printed with no margin rule
//!   (the forced arm below holds that case); every layer's live stores within its
//!   stores' band ([`gemm_bands`]) at the layers no flip's path reaches (no flip at a
//!   lower layer), the rest printed; the last logits within the head's band
//!   when no flip lies anywhere, printed otherwise; the argmax the
//!   reference's, or one whose logit in the reference's row lies within six
//!   head bands of the row's RMS below its top. Then the forced arm: the same
//!   calls with the reference's picks and weights planted at every position
//!   (`plant_prompt_routes`), so no flip can happen: every layer's live
//!   stores within its band, no layer excused, the last logits within the
//!   head's band, the argmax as above. At groups of two and four ([`GROUPS`],
//!   `set_prefill_group` on the one load) each call against the call at
//!   groups of one, bit for bit. Every call's checkpoints at the multiples of
//!   512 inside it and its end — the steps feed's marks, as (k) holds them.
//! - (pr) a call one position past the stores is refused by name before any
//!   launch, on either feed: the model stands at 0 and every store digests
//!   as the reset's.
//! - (pf) `blk.0.attn_norm.weight[0]` set to NaN, at groups of one and of
//!   two batches: a step and a batch call of nine ids each end in a fault at
//!   layer 0 — the batch's the step's with the GEMM quantizer's site added —
//!   the model poisoned and the next call refused as such; a call of 513 ids (two batches) ends in it after the group that
//!   holds its first batch (that batch alone at a group of one), no
//!   checkpoint taken or sealed of the faulted state — also with the NaN in
//!   the last KDA layer's norm instead, where the first batch's every store
//!   has reached its take before the fault is read; the weight put back, a
//!   reset and the call give (pb)'s argmax at groups of one.
//!   PIN(2026-10-02): the fault point moved from the first batch to the
//!   group that holds it, as a group reads the fault word once.
//! - (pg) a failure that is no fault, planted in a group's walk at its
//!   second unit (`Plant::Group(1)`), at groups of two: after a call of 513
//!   ids, a call of the next 517 ids (one group of two batches, the mark at
//!   1024 inside it) fails by name; the model stands at 513, the points from
//!   before the call (512, 513) stand, and the call again gives the stores,
//!   logits and argmax of the same two calls unplanted from a reset, its
//!   points 512, 513, 1024, 1030.
//!
//! The resident sequence slots (`bloomery_gpu::model::Slots` over the body):
//! the slot harness's contracts (`slots_gate`: H1 interleave, H3 bytes, H4
//! reset, H6 refusals, H7 captures) over the NextN load at [`SLOT_CTX`]
//! positions a slot on the gate placement, its plan counting the harness's
//! two slots (`PlanInputs::plan_nextn_slots`), two prompts of the prose
//! set's ids fed as the server feeds them (every id but the last in one
//! batch call, the last a step). H3's plan descriptor is one sequence of
//! `PlanInputs::seq_terms` (`SeqTerms::bytes`: 327,090,452 B at 256
//! positions, two lanes and the NextN layer [derived]). Between the
//! interleave and H4, the body's own clauses, moving slot 1 alone and
//! reading slot 0 through the harness:
//! - (sp) plan: a third slot on a plan of two refused by name; the plan of
//!   two counts one sequence's bytes past the plan of one (its stage card's
//!   KV term and the layer's store), the descriptor's.
//! - (sc) cut: slot 1 cut back to its prompt call's checkpoint, slot 0
//!   selected and stepped on its solo run's continuation before slot 1 takes
//!   the step that copies the cut back, then slot 1 stepped again to where it
//!   stood: its last id, position and stores bit for bit as before the cut —
//!   the pending cut travels with its slot.
//!
//! Then, on a load of their own (the MTP draft drives a session, which owns
//! its model; the harness lends its own by reference only), three prompts:
//! - (sd) drafted: under the MTP draft (each slot's side of it parked and put
//!   back around a select, as the seat's table does), the two prompts on
//!   slots 0 and 1, a select between every pass, [`SLOT_PASSES`] passes and
//!   a step each: each slot's ids, every pass's proposal and kept rows, and
//!   the last logits bit for bit its solo run's (slot 0 from a reset); the
//!   solo runs both accept and reject a proposal.
//! - (s3) park/resume: slot 1's state saved with its draft's side
//!   (`seq_save`), the slot reset and run on the third prompt, slot 0 run on
//!   between, then slot 1 put back (`seq_resume`): both slots'
//!   [`SLOT_RESUMED`] passes and a step bit for bit, counts included, their
//!   solo runs'.
//!
//! A pass of two slots (`GpuModel::step_slots`: the pair's walk, each row a
//! plain step of its own slot), residency off, the first two prompts. On the
//! NextN load, between (sc) and H4, (g5)'s first refusal: a pass of both
//! slots refused by name, no position moved. Then on the plain load of one
//! KDA lane, its plan counting [`G_SLOTS`] slots, against each slot's solo run from
//! a reset (slot 0; the stores digested as (slots) digests them):
//! - (g1) bits: both slots prompted, [`G_ROUNDS`] passes of both (row order
//!   0, 1 and 1, 0 in turn), one of slot 1 alone, then a plain step of each
//!   (its own chain over the buffers the passes bound and gave back): every
//!   row's id and logits hash, each slot's last logits bit for bit, stores
//!   and position its solo run's — in graph mode and again in eager mode.
//! - (g1b) parked, after (g1) in each mode: slots 1 and 2 prompted with the
//!   same two prompts, [`G1B_ROUNDS`] passes of both while slot 0, the live
//!   one, stays idle (both rows a parked sequence's): each slot its solo
//!   run's as in (g1), slot 0's position and stores as they stood.
//! - (g2) cut: after (g1) in graph mode, slot 1 cut back to its prompt
//!   call's checkpoint, then [`G_AFTER_CUT`] passes of both: slot 1 its solo
//!   run from its prompt's step, slot 0 its solo run's continuation — the
//!   parked slot's cut carried out from its own checkpoints by the pass.
//!   Slot 1's stores compare up to its position: the cut leaves its rows
//!   past it as they were, which a run from a reset holds zeroed.
//! - (g3) fault: a NaN in the head's gain under a pass of both slots ends it
//!   in the head's fault and poisons both, each slot's step and the pass
//!   refused naming the set; slot 0's reset leaves slot 1 in it, slot 1's
//!   lifts it; prompted again, the next pass both solo runs' first step.
//! - (g5) refusals by name, no position moved, each from both slots
//!   prompted afresh: three rows in one pass naming the body's bound 2, the
//!   taps armed, a slot whose step failed past its launch, a slot of two
//!   rows, a route trace attached.
//!
//! Then on a plain load of two KDA lanes:
//! - (g4) walk: the captured pass of both slots holds the captured verify's
//!   nodes; the verify replayed after the pass's capture (which re-records
//!   the pair's go order) and the pass replayed after the verify's capture
//!   each bit for bit their solo runs; between them, a pass refused by name
//!   while slot 0's verify waits for its commit.
//!
//! An error inside a clause body of (sp), (sc), (sd), (s3) or (g1)-(g5) is
//! that clause's FAIL, not the gate's end; one in a load, a solo run or the
//! draft's open ends the gate.
//!
//! `--only main` runs the clauses on the load at [`CTX`] alone, `--only pp`
//! the prompt batch's load alone, `--only pplong` (pb-long) alone on a load
//! at [`CTX`] of one lane, `--only verify` (s) and (v) alone on a load at
//! [`CTX`] of two (the plain graph run of the batch set they compare against
//! included), `--only keep` (k) alone on a load at [`CTX`] of two, `--only
//! slots` the resident slots' clauses alone on their four loads, `--only
//! stagger` (g1)-(g5) alone on their two plain loads.
//! `--step-sets short` takes (t)'s two 4-token sets only,
//! `--step-sets long` the 1,024- and 3,070-position sets only, `--step-sets
//! all` (the default) all four; it names the sets of the load at [`CTX`], so
//! beside `--only pp` or `--only pplong` it is refused by name.
//!
//! Named differences, not banded away: ik clamps each KDA state to ±1e6
//! after every token, ours raises its fault site where the state stops being
//! finite and clamps nothing; ik renormalizes the router's eight weights by
//! their bare sum, ours adds `1e-20` (under half an ulp of every sum from
//! 2^-42 up); ik's CPU head quantizes the normed row to q8_0 per 32 values
//! for its q8_0 lm_head, ours multiplies the f32 row.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_e2e: built without the `glm5next` feature; see `just gate-gpu-glm5next-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_e2e", gate::run())
}

#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_card.rs"]
mod card;

#[cfg(feature = "glm5next")]
mod gate {
    use crate::card;
    use std::path::PathBuf;
    use std::time::Instant;

    use app::arch::glm5next::{GlmCfg, open_pair};
    use app::mtp::{MtpBody, MtpDraft, Parked};
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::fault::{Fault, FaultSite, LAYER_HEAD};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LATENT, pools_for};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::flip::{self, Flip, tie_allowed};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::rounding::q8_32_rel;
    use bloomery_gpu_gates::slots_gate::{self, Derived, Interleaved, SlotsAdapter};
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, RefManifest, checks_failed, data_dir, ik_q8_2, patch_bytes,
        ref_tensor_logical_in, split_f32, topk_ids_logical_within, verdict,
    };
    use bloomery_gpu_glm5next::forced::{ForcedRoute, ForcedRow};
    use bloomery_gpu_glm5next::{
        Body, CHUNK, GEMM_FROM, Glm5nextModel, GlmArena, Plant, PrefillMode, RouteTapRows,
        StoreDigest, StoreRows, feed, plant_prompt_routes, prefill, prefill_mode,
        prompt_route_taps, seq_resume, seq_save, set_prefill, set_prefill_group,
        set_prompt_route_taps, set_taps, step_launches, store_digests, store_rows,
    };
    use bloomery_levers::CARD_BUDGET;
    use cuda_core::sys;
    use gguf::quant::dequant_row;
    use gguf::{GgmlType, Split};
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{KdaLanes, NextnInputs, NextnPlan, PlanInputs};
    use model::placement::{Machine, Plan, PlanLevers, workstation};
    use refset::arch::glm5next::{BATCH, D1K, D3K_DSA, IK, IK_DSA, MODEL, STEP4, STEP4_EVERY_NODE};
    use refset::family::Family;
    use runtime::layer::{FfnKind, Layer, MixerKind};
    use runtime::swaprule::KeptRows;
    use runtime::{Out, Target, Want};

    /// The router's experts and picks a token.
    const N_EXPERT: usize = 288;
    const N_USED: usize = 8;

    /// Cache rows: the `--dsa` set's step at position 3,070, with room.
    const CTX: usize = 3136;

    /// The prompt batch's load: every position the latent layers attend whole
    /// (`place::dense_positions`: the indexer's top-k in whole pools of 512,
    /// plus a pool's tail).
    const CTX_PP: usize = 2051;

    /// The prompt batch's calls, in positions: one; a chunk short, whole and
    /// one past it; a batch whole and one past it; a mark with a short tail
    /// after it; every position the stores hold.
    const PP: [usize; 8] = [1, 7, 8, 9, 512, 513, 1030, CTX_PP];

    /// (pb)'s prompt groups, in batches: each alone, then two and four
    /// batches a group (513, 1030 and [`CTX_PP`] are two, three and five
    /// batches: a group of two, one of three, a group of four and its lone
    /// last batch joined).
    const GROUPS: [usize; 3] = [1, 2, 4];

    /// (pb-long)'s prompt: past [`CTX_PP`] by more than a batch's chunk, and
    /// not a whole number of pools of four, so it ends inside one.
    const LONG: usize = 2222;
    /// (pb-long)'s first call: not a whole number of pools either, so the
    /// second call's first positions complete a pool the first began; its
    /// last batch, after the mark at 1024, past a chunk, so every batch of
    /// both calls runs the GEMM as the one call's do.
    const LONG_CUT: usize = 1042;
    /// (pb-long)'s tail: from the last mark below [`LONG`], past the
    /// positions the latent layers attend whole, fed by steps and in calls
    /// of a chunk.
    const LONG_TAIL: usize = 2048;
    /// (pb-long)'s greedy steps after the prompt: the second completes the
    /// prompt's last pool.
    const GREEDY: usize = 4;
    const _: () = assert!(
        LONG > CTX_PP + 8
            && LONG + GREEDY <= CTX
            && !LONG.is_multiple_of(4)
            && !LONG_CUT.is_multiple_of(4)
            && (LONG / 4 + 1) * 4 <= LONG + GREEDY
            && LONG_CUT % 512 > 8
            && LONG % 512 > 8
            && LONG_TAIL.is_multiple_of(512)
            && LONG_TAIL < CTX_PP
            && LONG_TAIL + 512 > LONG
    );

    /// The file's shape, as the header states it (glmops-design §1): what
    /// the derivations below are written against.
    const HIDDEN: usize = 4096;
    const STREAMS: usize = 4;
    const N_LAYER: usize = 45;
    const N_LATENT: usize = 11;
    const N_KDA: usize = 34;
    const N_DENSE: usize = 3;
    const N_ROUTED: usize = 42;
    const N_VOCAB: usize = 154_880;

    /// PIN(2026-09-28): the captured decode step's node count, derived before
    /// the chain was built: every layer's two sub-layers take three launches
    /// each for the streams (`hc_pre_q8_0`, the fold, `hc_post`); a KDA mixer
    /// 11 (the norm, four q8_0 projections of the normed row — the q·k·v
    /// joined, the two low-rank halves, β — and two of the halves, the conv
    /// and prep, the delta step, the gated norm, the output projection); a
    /// latent mixer 16 (the norm, the joined projection, the q_a norm, two
    /// appends; the selector's pool, head weights, indexer query, scores and
    /// top-k on the branch, which adds no node; q_b, k_b, the attention's two
    /// launches, v_b, the output projection); a dense block 3 (norm,
    /// gate·up, down); a routed block 8 (norm, router, handoff, the go, the
    /// shared gate·up and down, the wait, the sum); the head 4 (the mean, the
    /// norm, the q8_0 gemv, the argmax): 45·6 + 34·11 + 11·16 + 3·3 + 42·8 + 4.
    const NODES_DECODE: usize = 1169;

    /// PIN(2026-09-27): each routed layer's go and wait.
    const MEMOPS: usize = 2 * N_ROUTED;

    /// PIN(2026-09-27): the shadow's launches more on a routed layer with card
    /// experts: the norm's q8_1, the gate·up `_sel`, the q8_1 of its card
    /// columns, the down `_sel`, the card slots' sum and its add to the
    /// shared expert's output.
    const CARD_NODES: usize = 6;

    /// The routed layers whose downs are Q6_K in the file (the header), which
    /// the card experts do not read.
    const Q6K_DOWN: [usize; 3] = [11, 12, 44];

    const _: () = assert!(
        NODES_DECODE == N_LAYER * 6 + N_KDA * 11 + N_LATENT * 16 + N_DENSE * 3 + N_ROUTED * 8 + 4
            && N_KDA + N_LATENT == N_LAYER
            && N_DENSE + N_ROUTED == N_LAYER
    );

    /// PIN(2026-09-27): the teacher-forced bound on a tap's error ratio — its
    /// relative error over the relative distance between ik's 8-bit
    /// activation of the input that reaches it and that input ([`quant_gap`]:
    /// ik quantizes a q8_0 projection's input to q8_2, 32 values a bf16
    /// scale; our q8_0 gemvs read it in f32, so the gap is ik's alone; inputs
    /// in quadrature). Derivation, the qwen35moe gate's for this model: a
    /// linear map carries its input's relative perturbation unchanged, so the
    /// folds, the projections, the router's f32 logits and its sigmoid scores
    /// read about 1 (a sigmoid's slope is at most 1/4 of its value's scale,
    /// the Sinkhorn mix a normalization, neither amplifies); the KDA output,
    /// a sum over the call's positions the decay and β (both at most 1) do not
    /// amplify, at most √5 times a position's, so at most 5; the latent
    /// attention's softmax over at most 5 positions at most 3; a block's
    /// output at most 6 (qwen3moe measured 6.1 on a layer with one dominant
    /// channel). A wiring fault — another layer's input or weight, a position
    /// or a stream off by one — reads an error of order one over a gap of
    /// order 1e-2, a ratio above 40.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's, off every flip's path. Derivation, the qwen3moe
    /// and qwen35moe gates' form with this gate's own forced arm as its
    /// input: each layer's error on the streams on ik's inputs, the mixer's
    /// and the block's in quadrature, at most 1.415e-2 (layer 0, whose input
    /// is the bare embedding), added in quadrature over 45 layers,
    /// independent: √45 · 1.415e-2 ≈ 0.095, rounded up.
    const FREE_BAND: f64 = 0.10;

    /// PIN(2026-09-28): [잠정 — 백로그] the most error our two ranked values
    /// (a sigmoid score plus the selection bias, so a pair's error is below
    /// about 2) may carry at an excused flip — a measured frontier, not a
    /// derivation, in the rule the Qwen3.8 gate's cap takes. The clean
    /// chain's largest, over the batch set's 56 free flips, is 0.157 (layer
    /// 27, position 0; 0.140 over 42 with the card experts ranked by a
    /// router-frequency list learned from the test corpora); with one layer's delta-rule mixer skipped (layer 5,
    /// its output zeroed) the largest is 0.301 (layer 5, position 4) and the
    /// next 0.170. 0.2 is their geometric mean rounded, 1.27x above clean,
    /// and only that one flip past it: a thin frontier. The forced arm's
    /// ratio and the node count are what catch a skipped sub-layer; a
    /// flip-aware bound replaces this cap.
    const FLIP_ERR_CAP: f64 = 0.2;

    /// PIN(2026-09-28): a named tie's logits row against the head input's
    /// own distance: the row's relative distance from ik's within this many
    /// times the last layer's. The head is the streams' mean, a norm and a
    /// linear map, which carries its input's relative perturbation about
    /// unchanged (the heuristic [`RATIO_BAND`] takes for a linear map, not a
    /// bound); the three step sets read 0.75, 0.67 and 0.83 of it.
    const HEAD_RATIO: f64 = 1.5;

    /// The stores' bytes at `ctx` rows and `lanes` KDA lanes, derived from
    /// the header: each KDA layer's state, `lanes` lanes of 64 heads of 128 ×
    /// 128 f32, a u32 stamp a lane, and conv ring, 11 rows of 3 · 64 · 128
    /// f32; each latent layer's latent and index rows, 512 + 256 f16 a
    /// position, and its pool plane, 128 f16 a pool of four.
    fn store_bytes(ctx: usize, lanes: usize) -> usize {
        N_KDA * ((lanes * 64 * 128 * 128 + 11 * 3 * 64 * 128) * 4 + lanes * 4)
            + N_LATENT * (ctx * (512 + 256) + ctx.div_ceil(4) * 128) * 2
    }

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch, so
    /// neither passes a band.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The first index of the largest value, as the head's argmax breaks ties.
    fn argmax(v: &[f32]) -> u32 {
        let best = v
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b });
        best as u32
    }

    /// The runner-up's index: the largest value other than at `top`.
    fn second(v: &[f32], top: u32) -> u32 {
        let best = v.iter().enumerate().fold(None::<usize>, |b, (i, &x)| {
            if i == top as usize {
                b
            } else {
                match b {
                    Some(j) if v[j] >= x => Some(j),
                    _ => Some(i),
                }
            }
        });
        best.unwrap_or(0) as u32
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// The open's records, as the gate prints them.
    struct Log {
        t: Instant,
        ctx: usize,
        nodes: Option<usize>,
        /// The plan's card experts a layer.
        n_l: Vec<u64>,
        /// The routed experts a layer.
        experts: u64,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            _inputs: &PlanInputs,
            _machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            println!(
                "plan place={place} ctx_max={} host_experts={} card_experts={}",
                plan.ctx_max, plan.host.experts, plan.cards[0].experts
            );
            self.n_l = plan.n_l.clone();
            self.experts = plan.model.experts;
            Ok(true)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            println!(
                "load resident_bytes={} ctx={} layers={} in {:.1} s (runtime value)",
                m.resident_bytes(),
                self.ctx,
                m.layers().len(),
                self.t.elapsed().as_secs_f64()
            );
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            self.nodes = Some(nodes);
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// What the open decided besides the session: the step's captured nodes,
    /// the plan's card experts a layer and routed experts a layer, and whether
    /// it ran under a card budget.
    struct Opened {
        nodes: usize,
        n_l: Vec<u64>,
        experts: u64,
        budgeted: bool,
    }

    /// The session at `ctx` positions on the gate placement, its prompts fed
    /// by `prefill`, each KDA layer's state `lanes` lanes: two opens the
    /// verify's load (`app::arch::glm5next::open_pair`), one the plain
    /// session's.
    fn open(
        levers: &bloomery_levers::Levers,
        ctx: usize,
        prefill: PrefillMode,
        lanes: KdaLanes,
    ) -> Result<(Session<Body>, Opened), GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(levers)?,
            host: levers.host(),
            prefill,
            group: 1,
        };
        let budgeted = cfg.place.card_budget_bytes.is_some();
        let mut log = Log {
            t: Instant::now(),
            ctx,
            nodes: None,
            n_l: Vec::new(),
            experts: 0,
        };
        let args = OpenArgs {
            place: "gate",
            machine: workstation::plan_gate,
            ctx,
            mode: StepMode::Graph,
            cfg,
        };
        let s = match lanes {
            KdaLanes::One => Loaded::<Body>::open(file, args, &mut log)?
                .ok_or("the open stopped at its plan")?
                .ready(&mut log)?,
            KdaLanes::Two => {
                open_pair(file, args, &mut log)?.ok_or("the open stopped at its plan")?
            }
        };
        let nodes = log.nodes.ok_or("graph mode captured no step")?;
        let n_l = log.n_l;
        Ok((
            s,
            Opened {
                nodes,
                n_l,
                experts: log.experts,
                budgeted,
            },
        ))
    }

    // ------------------------------------------------------ (s) structure

    fn structure(m: &Glm5nextModel, o: &Opened) -> Result<bool, GateError> {
        let (nodes, budgeted) = (o.nodes, o.budgeted);
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let at = |f: &dyn Fn(usize) -> bool| -> Vec<usize> {
            (0..kinds.len()).filter(|&l| f(l)).collect()
        };
        let latent = at(&|l| kinds[l].mixer == MixerKind::Latent);
        let dense = at(&|l| kinds[l].ffn == FfnKind::Dense);
        let want_latent: Vec<usize> = (0..N_LAYER).filter(|l| l % 4 == 3).collect();
        let lanes = body.lanes();
        let (stores, want_stores) = (body.store_bytes(), store_bytes(CTX, lanes.count()));
        let mut ok = kinds.len() == N_LAYER
            && latent == want_latent
            && dense == (0..N_DENSE).collect::<Vec<_>>()
            && body.host_run() == (N_DENSE..N_LAYER)
            && lanes == KdaLanes::Two
            && stores == want_stores;
        println!(
            "structure layers={} latent at {latent:?} dense at {dense:?} host run {:?}; {} KDA \
             lanes (want 2: the verify's load); store bytes {stores} (want {want_stores}, derived) \
             {}",
            kinds.len(),
            body.host_run(),
            lanes.count(),
            verdict(ok)
        );
        let mixers: Vec<MixerKind> = kinds.iter().map(|k| k.mixer).collect();
        let ffns: Vec<FfnKind> = kinds.iter().map(|k| k.ffn).collect();
        // The card layers are the plan's: its n_l, which the slot map must hold.
        let slots = body.hybrid().slots();
        let planned = |l: usize| o.n_l.get(l).copied().unwrap_or(0);
        let cards: Vec<bool> = (0..kinds.len()).map(|l| planned(l) > 0).collect();
        let card_layers = at(&|l| cards[l]);
        let mut map_is_plan = o.n_l.len() == kinds.len();
        for (l, k) in kinds.iter().enumerate() {
            // A dense layer has no slot-map row: no card experts, as planned.
            // A routed layer the map has no row for fails the gate by name.
            let on_card = match k.ffn {
                FfnKind::Dense => 0,
                FfnKind::Moe => slots.on_card(l)?,
            };
            map_is_plan &= on_card as u64 == planned(l);
        }
        let readable = at(&|l| kinds[l].ffn == FfnKind::Moe && !Q6K_DOWN.contains(&l));
        let card_ok = map_is_plan
            && card_layers.iter().all(|l| readable.contains(l))
            && (budgeted || card_layers == readable);
        let per_layer: Vec<u64> = card_layers.iter().map(|&l| planned(l)).collect();
        println!(
            "structure the plan's card experts on {} routed layers {card_layers:?}, per layer \
             {per_layer:?}, the slot map the same ({map_is_plan}); none on a dense layer or on \
             {Q6K_DOWN:?}, and {} {}",
            card_layers.len(),
            if budgeted {
                "a card budget set".to_string()
            } else {
                format!("with no card budget on all {} others", readable.len())
            },
            verdict(card_ok)
        );
        ok &= card_ok;
        let want = NODES_DECODE + CARD_NODES * card_layers.len();
        let counted = step_launches(&mixers, &ffns, &cards);
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let pass =
            nodes == want && counted == want && b == MEMOPS && k == want - MEMOPS && other == 0;
        println!(
            "structure decode graph_nodes={nodes} (want {want} = {NODES_DECODE} + {CARD_NODES} x {} \
             card layers; the program counts {counted}) kernel={k} batch_mem_op={b} (want {MEMOPS}) \
             other={other} {}",
            card_layers.len(),
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    // ------------------------------------------ (p) one chain, (c) free

    /// One run of `toks` from a reset: each position's argmax and logits,
    /// and, with the taps armed, each position's layer outputs.
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
    }

    fn run_steps(m: &mut Glm5nextModel, toks: &[u32], mode: StepMode) -> Result<Run, GateError> {
        m.reset()?;
        let taps_on = mode == StepMode::Eager;
        set_taps(m, taps_on)?;
        m.set_mode(mode);
        let mut r = Run {
            tokens: Vec::new(),
            logits: Vec::new(),
            taps: Vec::new(),
        };
        for &t in toks {
            r.tokens.push(m.step(&[t])?);
            r.logits.push(m.logits()?);
            if taps_on {
                let (gpu, _, b) = m.body_parts("run_steps")?;
                r.taps.push(b.taps(gpu)?);
            }
        }
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        Ok(r)
    }

    /// Bit equality of two logits rows.
    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn one_chain(graph: &Run, eager: &Run) -> bool {
        let tokens = graph.tokens == eager.tokens;
        let logits = graph.logits.len() == eager.logits.len()
            && graph
                .logits
                .iter()
                .zip(&eager.logits)
                .all(|(a, b)| same_bits(a, b));
        let ok = tokens && logits;
        println!(
            "one chain: graph tokens {:?}, eager tokens {:?}; logits bit for bit: {logits} {}",
            graph.tokens,
            eager.tokens,
            verdict(ok)
        );
        ok
    }

    /// Each layer's worst relative distance over the positions `taps` holds
    /// against ik's `l_out-L`, whose last rows are those positions' (ik
    /// keeps only the output token's rows past the last attention), and the
    /// index into `taps` it was met at.
    fn layer_rels(man: &RefManifest, taps: &[Vec<f32>]) -> Result<Vec<(f64, usize)>, GateError> {
        Ok(worst_of(&layer_table(man, taps)?))
    }

    /// Each row's worst entry of a [`layer_table`] and the tap it was met at.
    fn worst_of(table: &[Vec<Option<f64>>]) -> Vec<(f64, usize)> {
        table
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(t, e)| e.map(|e| (e, t)))
                    .fold((0.0f64, 0usize), |w, x| if x.0 > w.0 { x } else { w })
            })
            .collect()
    }

    /// Each layer's relative distance at every position `taps` holds
    /// against ik's `l_out-L` (`None` where ik kept no row for it).
    fn layer_table(
        man: &RefManifest,
        taps: &[Vec<f32>],
    ) -> Result<Vec<Vec<Option<f64>>>, GateError> {
        let row = STREAMS * HIDDEN;
        let n = taps.len();
        (0..N_LAYER)
            .map(|l| {
                let ik = tap(man, &format!("l_out-{l}"))?;
                let kept = ik.len() / row;
                Ok(taps
                    .iter()
                    .enumerate()
                    .map(|(t, ours)| {
                        let i = (t + kept).checked_sub(n)?;
                        Some(rel(
                            &ours[l * row..(l + 1) * row],
                            &ik[i * row..(i + 1) * row],
                        ))
                    })
                    .collect())
            })
            .collect()
    }

    /// The worst layer and the first of layers `0..held` past the band,
    /// printed; whether every one of those is inside it.
    fn print_layers(what: &str, rels: &[(f64, usize)], held: usize) -> bool {
        for (l, &(e, t)) in rels.iter().enumerate() {
            if l % 8 == 0 || l == rels.len() - 1 || (l < held && e > FREE_BAND) {
                println!("{what} layer={l} l_out_rel={e:.3e} at tap {t}");
            }
        }
        match rels.iter().take(held).position(|&(e, _)| e > FREE_BAND) {
            Some(l) => {
                println!("{what}: first layer past the band {FREE_BAND:.2}: {l}");
                false
            }
            None => true,
        }
    }

    // ------------------------------------------------ routing and flips

    /// ik's routing of one routed layer over the set's tokens: its ranked
    /// values (`ffn_moe_probs_biased`, [`N_EXPERT`] a token) and its chosen
    /// ids ([`N_USED`] a token).
    struct IkRoute {
        biased: Vec<f32>,
        ids: Vec<i32>,
    }

    impl IkRoute {
        fn read(man: &RefManifest, l: usize) -> Result<IkRoute, GateError> {
            let biased = tap(man, &format!("ffn_moe_probs_biased-{l}"))?;
            let row = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, row, N_EXPERT as u32)?;
            if biased.len() % N_EXPERT != 0 || ids.len() * N_EXPERT != biased.len() * N_USED {
                return Err(format!(
                    "layer {l}: {} ranked values and {} ids",
                    biased.len(),
                    ids.len()
                )
                .into());
            }
            Ok(IkRoute { biased, ids })
        }

        fn tokens(&self) -> usize {
            self.ids.len() / N_USED
        }

        /// Token `t`'s ranked values and ids.
        fn at(&self, t: usize) -> (&[f32], &[i32]) {
            (
                &self.biased[t * N_EXPERT..(t + 1) * N_EXPERT],
                &self.ids[t * N_USED..(t + 1) * N_USED],
            )
        }

        /// ik's own margin at token `t`: its eighth pick's ranked value less
        /// the best it left.
        fn margin(&self, t: usize) -> f64 {
            let (v, ids) = self.at(t);
            flip::margin(v, ids)
        }
    }

    /// Our ranked values: the scores plus the selection bias, the f32 add
    /// the router's pick makes.
    fn ours_ranked(r: &ForcedRoute) -> Vec<f32> {
        r.probs.iter().zip(&r.bias).map(|(&p, &b)| p + b).collect()
    }

    /// The flip at layer `l`, token `t`, if our chosen set is not ik's, in
    /// the ranked values ([`Flip::between`]).
    fn flip_at(l: usize, t: usize, ours: &ForcedRoute, ik: &IkRoute) -> Option<Flip> {
        let (iv, ids) = ik.at(t);
        let ov = ours_ranked(ours);
        Flip::between((l, t), (&ours.ids, &ov), (ids, iv), ik.margin(t))
    }

    /// Every routed layer's ik routing, by layer (`None` for a dense one).
    fn ik_routes(man: &RefManifest) -> Result<Vec<Option<IkRoute>>, GateError> {
        (0..N_LAYER)
            .map(|l| {
                if l < N_DENSE {
                    Ok(None)
                } else {
                    IkRoute::read(man, l).map(Some)
                }
            })
            .collect()
    }

    /// ik's margins, every routed layer by token: printed, the numbers a
    /// free table's jumps are read against.
    fn print_margins(routes: &[Option<IkRoute>]) {
        for (l, r) in routes.iter().enumerate() {
            if let Some(r) = r {
                let cells: Vec<String> = (0..r.tokens())
                    .map(|t| format!("{:.2e}", r.margin(t)))
                    .collect();
                println!("ik margin layer={l} by token {}", cells.join(" "));
            }
        }
    }

    // ---------------------------------------------- (l) layer by layer

    /// Token `t`'s embedding row from the file, dequantized as the body
    /// reads it, in each of the four streams.
    fn embedding(file: &Split, t: u32) -> Result<Vec<f32>, GateError> {
        let name = names::token_embd();
        let (s, info) = file
            .find(&name)
            .ok_or_else(|| format!("{name}: not in the file"))?;
        if info.ty != GgmlType::Q8_0 {
            return Err(format!("{name} is {:?}, not q8_0", info.ty).into());
        }
        let rb = HIDDEN / 32 * 34;
        let data = file.shard(s).ok_or("the embedding's shard")?.data(info)?;
        let src = data
            .get(t as usize * rb..(t as usize + 1) * rb)
            .ok_or_else(|| format!("token {t} past the embedding"))?;
        let mut row = vec![0.0f32; HIDDEN];
        dequant_row(GgmlType::Q8_0, src, &mut row)?;
        Ok(row.repeat(STREAMS))
    }

    /// Layer `l` alone at positions `0..inputs.len()` from a reset, one
    /// forced row each: input `t` its streams, `ffn[t]` the feed-forward
    /// sub-layer's when given.
    fn layer_rows(
        m: &mut Glm5nextModel,
        l: usize,
        inputs: &[&[f32]],
        ffn: Option<&[&[f32]]>,
    ) -> Result<Vec<ForcedRow>, GateError> {
        m.reset()?;
        let rows = {
            let (gpu, w, b) = m.body_parts("layer_rows")?;
            inputs
                .iter()
                .enumerate()
                .map(|(t, x)| b.forced_row(gpu, w, l, t as u32, x, ffn.map(|f| f[t])))
                .collect::<Result<Vec<_>, _>>()?
        };
        m.reset()?;
        Ok(rows)
    }

    /// The batch tokens through every layer, one layer at a time, each on
    /// our own previous layer's streams, from the embedding.
    fn layered(m: &mut Glm5nextModel, toks: &[u32]) -> Result<Vec<Vec<ForcedRow>>, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let mut x: Vec<Vec<f32>> = toks
            .iter()
            .map(|&t| embedding(&file, t))
            .collect::<Result<_, _>>()?;
        let mut all = Vec::with_capacity(N_LAYER);
        for l in 0..N_LAYER {
            let inputs: Vec<&[f32]> = x.iter().map(Vec::as_slice).collect();
            let rows = layer_rows(m, l, &inputs, None)?;
            x = rows.iter().map(|r| r.out.clone()).collect();
            all.push(rows);
        }
        Ok(all)
    }

    /// Every layer output of the layered run against the eager chain's tap,
    /// bit for bit.
    fn layered_is_chain(layered: &[Vec<ForcedRow>], eager: &Run) -> bool {
        let row = STREAMS * HIDDEN;
        let differ: Vec<(usize, usize)> = layered
            .iter()
            .enumerate()
            .flat_map(|(l, rows)| rows.iter().enumerate().map(move |(t, r)| (l, t, r)))
            .filter(|&(l, t, r)| {
                eager
                    .taps
                    .get(t)
                    .and_then(|tap| tap.get(l * row..(l + 1) * row))
                    .is_none_or(|want| !same_bits(&r.out, want))
            })
            .map(|(l, t, _)| (l, t))
            .collect();
        let ok = layered.len() == N_LAYER
            && layered.iter().all(|r| r.len() == eager.taps.len())
            && differ.is_empty();
        println!(
            "layered: {} layers x {} positions one layer at a time on our own streams against the \
             eager chain's taps, bit for bit; (layer, position) differing: {} {:?} {}",
            layered.len(),
            eager.taps.len(),
            differ.len(),
            &differ[..differ.len().min(8)],
            verdict(ok)
        );
        ok
    }

    // --------------------------------------------------- (c) free

    /// Whether a flip at `(l', t')` lies on the path of layer `l`'s output
    /// at position `t`: every layer from `l'` on reads it at `t'`, and every
    /// later position through the mixers' stores.
    fn on_path(flips: &[Flip], l: usize, t: usize, same_position: bool) -> bool {
        flips.iter().any(|f| {
            f.layer <= l
                && if same_position {
                    f.token == t
                } else {
                    f.token <= t
                }
        })
    }

    /// The free clause's verdict, and by position the first layer a flip
    /// lies on the path of ([`N_LAYER`] where none does). The picks come
    /// from the layered run, so the clause is void — a named FAIL — when
    /// `layered_ok` says that run is not the chain's: its flips would excuse
    /// what another chain did.
    fn free(
        man: &RefManifest,
        eager: &Run,
        (layered, layered_ok): (&[Vec<ForcedRow>], bool),
        routes: &[Option<IkRoute>],
    ) -> Result<(bool, Vec<usize>), GateError> {
        let table = layer_table(man, &eager.taps)?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!("free table layer={l} l_out_rel by tap {}", cells.join(" "));
        }
        print_margins(routes);
        let mut flips = Vec::new();
        for (l, (rows, ik)) in layered.iter().zip(routes).enumerate() {
            let Some(ik) = ik else { continue };
            for (t, r) in rows.iter().enumerate() {
                let route = r
                    .route
                    .as_ref()
                    .ok_or_else(|| format!("layer {l}: a routed layer with no route"))?;
                flips.extend(flip_at(l, t, route, ik));
            }
        }
        for f in &flips {
            println!("{}", f.line("free", FLIP_ERR_CAP));
        }
        let flips_ok = flips.iter().all(|f| f.allowed(FLIP_ERR_CAP));
        let worst_held = |same: bool| -> (f64, usize, usize, usize) {
            let mut w = (0.0f64, 0usize, 0usize, 0usize);
            for (l, row) in table.iter().enumerate() {
                for (t, e) in row.iter().enumerate() {
                    let Some(e) = *e else { continue };
                    if on_path(&flips, l, t, same) {
                        w.3 += 1;
                    } else if e > w.0 {
                        (w.0, w.1, w.2) = (e, l, t);
                    }
                }
            }
            w
        };
        let (held, hl, ht, exempt) = worst_held(false);
        let (same, sl, st, same_exempt) = worst_held(true);
        let firsts: Vec<usize> = (0..eager.taps.len())
            .map(|t| {
                (0..N_LAYER)
                    .find(|&l| on_path(&flips, l, t, false))
                    .unwrap_or(N_LAYER)
            })
            .collect();
        println!(
            "free: the band holds before the first flip on a position's path (any flip at an \
             earlier or equal layer and position); first such layer by position {}; {exempt} \
             (layer, position) outputs past a flip, printed and counted",
            firsts
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!(
            "free (same-position path, printed): worst l_out_rel={same:.3e} at layer {sl} \
             position {st}; {same_exempt} outputs past a flip at their own position"
        );
        let ik = tap(man, "result_output")?;
        let ik_last = &ik[ik
            .len()
            .checked_sub(N_VOCAB)
            .ok_or("result_output holds no row")?..];
        let ours = eager.logits.last().ok_or("no logits")?;
        let (top, ik_top) = (argmax(ours), argmax(ik_last));
        if !layered_ok {
            println!(
                "free: FAIL: void — the layered run is not the chain's ((l) is red), so the \
                 picks it read and the flips they excuse are another chain's"
            );
        }
        let ok = layered_ok && held <= FREE_BAND && flips_ok && top == ik_top;
        println!(
            "free: {} tokens, {} flips ({} allowed), worst l_out_rel off every flip's path \
             {held:.3e} at layer {hl} position {ht} (band {FREE_BAND:.2}); last position argmax \
             ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            eager.tokens.len(),
            flips.len(),
            flips.iter().filter(|f| f.allowed(FLIP_ERR_CAP)).count(),
            rel(ours, ik_last),
            verdict(ok)
        );
        Ok((ok, firsts))
    }

    // ------------------------------------------------ (f) teacher-forced

    /// `‖x̂_ik − x‖ / ‖x‖`: how far ik's 8-bit activation of `x` (q8_2,
    /// blocks of 32, a bf16 scale) sits from the f32 `x` our kernels read.
    fn quant_gap(x: &[f32]) -> f64 {
        let whole = x.len() / ik_q8_2::QK * ik_q8_2::QK;
        rel(&ik_q8_2::reconstruct(&x[..whole]), &x[..whole])
    }

    /// The RMS-normed rows of `x` times `gain`, in f64 then rounded: the
    /// input a mixer's projections read, near enough for its gap.
    fn normed(x: &[f32], gain: &[f32]) -> Vec<f32> {
        x.chunks(HIDDEN)
            .flat_map(|r| {
                let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / HIDDEN as f64;
                let s = 1.0 / (ms + 1e-6).sqrt();
                r.iter()
                    .zip(gain)
                    .map(move |(&v, &g)| (f64::from(v) * s * f64::from(g)) as f32)
            })
            .collect()
    }

    /// Worst ratio and its tap, over a layer's taps.
    #[derive(Default)]
    struct Worst {
        ratio: f64,
        tap: String,
        lines: Vec<String>,
    }

    impl Worst {
        fn add(&mut self, tap: &str, e: f64, gap: f64) {
            let r = e / gap.max(f64::MIN_POSITIVE);
            let r = if r.is_nan() { f64::INFINITY } else { r };
            self.lines
                .push(format!("{tap} rel={e:.3e} gap={gap:.3e} ratio={r:.2}"));
            if self.tap.is_empty() || r > self.ratio {
                self.ratio = r;
                self.tap = tap.to_string();
            }
        }
    }

    /// `rows` of `k` values of `v`, concatenated.
    fn pick(v: &[f32], k: usize, rows: &[usize]) -> Vec<f32> {
        rows.iter()
            .flat_map(|&t| v[t * k..(t + 1) * k].to_vec())
            .collect()
    }

    /// `a − b`, value by value.
    fn minus(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(&x, &y)| x - y).collect()
    }

    /// One field of every row, concatenated.
    fn cat(rows: &[ForcedRow], f: impl Fn(&ForcedRow) -> &[f32]) -> Vec<f32> {
        rows.iter().flat_map(|r| f(r).to_vec()).collect()
    }

    /// Every layer alone on ik's own inputs at the batch set's positions:
    /// the mixer sub-layer on ik's layer input (`hc_init`, then `l_out-(L−1)`),
    /// the feed-forward sub-layer on ik's streams after the mixer
    /// (`attn_out-L`); each tap's error over its input's gap within
    /// [`RATIO_BAND`], every flip allowed ([`Flip::allowed`]). Returns the
    /// verdict and each layer's error on the streams (the mixer's and the
    /// block's in quadrature), the input [`FREE_BAND`] is derived from.
    fn forced(
        m: &mut Glm5nextModel,
        man: &RefManifest,
        routes: &[Option<IkRoute>],
    ) -> Result<(bool, Vec<f64>), GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let kinds = m.body("forced")?.kinds();
        let row = STREAMS * HIDDEN;
        let (mut ok, mut worst, mut flips_all) = (true, 0.0f64, 0usize);
        let mut errs = Vec::with_capacity(N_LAYER);
        for (l, kind) in kinds.iter().enumerate() {
            let x = if l == 0 {
                tap(man, "hc_init")?
            } else {
                tap(man, &format!("l_out-{}", l - 1))?
            };
            let f = tap(man, &format!("attn_out-{l}"))?;
            let out = tap(man, &format!("l_out-{l}"))?;
            let t_n = x.len() / row;
            if t_n == 0 || x.len() != t_n * row || f.len() != x.len() || out.len() != x.len() {
                return Err(format!(
                    "layer {l}: streams of {}, {} and {} values",
                    x.len(),
                    f.len(),
                    out.len()
                )
                .into());
            }
            let xs: Vec<&[f32]> = x.chunks(row).collect();
            let fs: Vec<&[f32]> = f.chunks(row).collect();
            let rows = layer_rows(m, l, &xs, Some(&fs))?;
            let mut w = Worst::default();
            // The mixer sub-layer.
            let gap_hca = quant_gap(&tap_at(man, &format!("hc_pre-{l}"), 0)?);
            let fold_a = tap(man, &format!("hc_attn_pre-{l}"))?;
            let gain = split_f32(&file, &names::attn_norm(l), HIDDEN)?;
            let gap_a = quant_gap(&normed(&fold_a, &gain));
            let (y_name, out_name) = match kind.mixer {
                MixerKind::DeltaRule => ("final_output", "linear_attn_out"),
                MixerKind::Latent => ("kqv_2d", "kqv_out"),
                MixerKind::Gqa => return Err(format!("layer {l}: a GQA mixer").into()),
            };
            let gap_y = quant_gap(&tap(man, &format!("{y_name}-{l}"))?);
            let gap_mix = gap_hca.hypot(gap_a).hypot(gap_y);
            w.add(
                "hc_attn_pre (fold)",
                rel(&cat(&rows, |r| &r.mix_in), &fold_a),
                gap_hca,
            );
            w.add(
                &format!("{out_name} (mixer out)"),
                rel(
                    &cat(&rows, |r| &r.mix_out),
                    &tap(man, &format!("{out_name}-{l}"))?,
                ),
                gap_mix,
            );
            let mixed = cat(&rows, |r| &r.mixed);
            w.add(
                "attn_out (update)",
                rel(&minus(&mixed, &x), &minus(&f, &x)),
                gap_mix,
            );
            let e_mix = rel(&mixed, &f);
            // The feed-forward sub-layer on ik's streams.
            let gap_hcf = quant_gap(&tap_at(man, &format!("hc_pre-{l}"), 1)?);
            let ik_normed = tap(man, &format!("ffn_norm-{l}"))?;
            let gap_ffn = gap_hcf.hypot(quant_gap(&ik_normed));
            w.add(
                "hc_ffn_pre (fold)",
                rel(
                    &cat(&rows, |r| &r.ffn_in),
                    &tap(man, &format!("hc_ffn_pre-{l}"))?,
                ),
                gap_hcf,
            );
            w.add(
                "ffn_norm",
                rel(&cat(&rows, |r| &r.ffn_normed), &ik_normed),
                gap_hcf,
            );
            let mut kept: Vec<usize> = (0..t_n).collect();
            let mut flips_ok = true;
            if let Some(ik) = &routes[l] {
                let rs: Vec<&ForcedRoute> = rows
                    .iter()
                    .map(|r| r.route.as_ref().ok_or("a routed layer with no route"))
                    .collect::<Result<_, _>>()?;
                if ik.tokens() != t_n {
                    return Err(format!("layer {l}: ik routes {} tokens", ik.tokens()).into());
                }
                let logits: Vec<f32> = rs.iter().flat_map(|r| r.logits.clone()).collect();
                w.add(
                    "ffn_moe_logits",
                    rel(&logits, &tap(man, &format!("ffn_moe_logits-{l}"))?),
                    gap_hcf,
                );
                let ranked: Vec<f32> = rs.iter().flat_map(|r| ours_ranked(r)).collect();
                w.add("ffn_moe_probs_biased", rel(&ranked, &ik.biased), gap_hcf);
                let ik_w = tap(man, &format!("ffn_moe_weights_scaled-{l}"))?;
                let (mut w_ours, mut w_ik) = (Vec::new(), Vec::new());
                for (t, r) in rs.iter().enumerate() {
                    if let Some(fl) = flip_at(l, t, r, ik) {
                        println!("{}", fl.line("forced", FLIP_ERR_CAP));
                        flips_ok &= fl.allowed(FLIP_ERR_CAP);
                        flips_all += 1;
                        kept.retain(|&k| k != t);
                        continue;
                    }
                    let (_, ids) = ik.at(t);
                    for (s, &e) in r.ids.iter().enumerate() {
                        let j = ids
                            .iter()
                            .position(|&x| x == e as i32)
                            .ok_or("an id the set test found")?;
                        w_ours.push(r.weights[s]);
                        w_ik.push(ik_w[t * N_USED + j]);
                    }
                }
                if !w_ik.is_empty() {
                    w.add("ffn_moe_weights_scaled", rel(&w_ours, &w_ik), gap_hcf);
                }
            }
            let e_ffn = if kept.is_empty() {
                0.0
            } else {
                let ours_out = cat(&rows, |r| &r.ffn_out);
                w.add(
                    "ffn_out",
                    rel(
                        &pick(&ours_out, HIDDEN, &kept),
                        &pick(&tap(man, &format!("ffn_out-{l}"))?, HIDDEN, &kept),
                    ),
                    gap_ffn,
                );
                let got = pick(&cat(&rows, |r| &r.out), row, &kept);
                let (want, base) = (pick(&out, row, &kept), pick(&f, row, &kept));
                w.add(
                    "l_out (update)",
                    rel(&minus(&got, &base), &minus(&want, &base)),
                    gap_ffn,
                );
                rel(&got, &want)
            };
            let e = e_mix.hypot(e_ffn);
            errs.push(e);
            let pass = flips_ok && w.ratio <= RATIO_BAND;
            println!(
                "forced layer={l} {:?}/{:?}: worst ratio {:.2} at {} (band {RATIO_BAND}); \
                 stream error {e:.3e} (mixer {e_mix:.3e}, block {e_ffn:.3e}); {} of {t_n} \
                 positions past a flip {}",
                kind.mixer,
                kind.ffn,
                w.ratio,
                w.tap,
                t_n - kept.len(),
                verdict(pass)
            );
            if !pass || l % 8 == 0 || l == N_LAYER - 1 {
                for line in &w.lines {
                    println!("  forced layer={l} {line}");
                }
            }
            worst = worst.max(w.ratio);
            ok &= pass;
        }
        let e_max = errs.iter().copied().fold(0.0, f64::max);
        let quad = errs.iter().map(|e| e * e).sum::<f64>().sqrt();
        println!(
            "forced: {N_LAYER} layers on ik's inputs; worst ratio {worst:.2} (band {RATIO_BAND}); \
             {flips_all} flips; worst stream error a layer {e_max:.3e}, sqrt(45)·it {:.3e}, the \
             errors in quadrature {quad:.3e} (FREE_BAND's derivation input) {}",
            (N_LAYER as f64).sqrt() * e_max,
            verdict(ok)
        );
        Ok((ok, errs))
    }

    /// A set's tap `name`, occurrence `occ`, in its logical order.
    fn tap_at(man: &RefManifest, name: &str, occ: u32) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, occ)?)?)
    }

    // --------------------------------------------------- (t) step sets

    /// The step of set `name` after its prefill fed by our steps; its
    /// argmax against ik's, a tie named and counted; with `band` — the batch
    /// set's tokens and the first layer a flip lies on the path of its last
    /// position — its layer outputs below that layer held to the band. The
    /// set's prefill and step must then be those tokens, which our steps
    /// route as the batch set's free arm did.
    fn step_set(
        m: &mut Glm5nextModel,
        (name, family): (&str, &Family),
        band: Option<(&[u32], usize)>,
        ties: &mut usize,
    ) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(name), family)?;
        let (pos, step, prefill) = man.step()?;
        let (pos, step, prefill) = (pos, step.to_vec(), prefill.to_vec());
        let [tok] = step[..] else {
            return Err(format!("{name}: a step of {} tokens, not one", step.len()).into());
        };
        let held = match band {
            Some((toks, first)) => {
                if toks.split_last() != Some((&tok, &prefill[..])) {
                    return Err(format!(
                        "{name}: prefill {prefill:?} and step {tok} are not the batch set's {toks:?}"
                    )
                    .into());
                }
                first
            }
            None => 0,
        };
        m.reset()?;
        let t = Instant::now();
        if !prefill.is_empty() {
            m.step(&prefill)?;
        }
        let r = run_last(m, tok)?;
        let ik = tap(&man, "result_output")?;
        let ik_last = &ik[ik
            .len()
            .checked_sub(N_VOCAB)
            .ok_or("result_output holds no row")?..];
        let (top, ik_top) = (argmax(&r.1), argmax(ik_last));
        let ik_2 = second(ik_last, ik_top);
        let margin = f64::from(ik_last[ik_top as usize]) - f64::from(ik_last[ik_2 as usize]);
        let dist = [ik_top, ik_2]
            .iter()
            .map(|&i| (f64::from(r.1[i as usize]) - f64::from(ik_last[i as usize])).abs())
            .fold(0.0, f64::max);
        let logits_rel = rel(&r.1, ik_last);
        let rels = layer_rels(&man, std::slice::from_ref(&r.0))?;
        let input_rel = rels.last().map_or(f64::INFINITY, |r| r.0);
        let tie = tie_allowed(
            (top, ik_top, ik_2),
            (margin, dist),
            (logits_rel, input_rel),
            HEAD_RATIO,
        );
        *ties += usize::from(tie);
        let inside = print_layers(name, &rels, held);
        let ok = (top == ik_top || tie) && inside;
        println!(
            "step {name}: logits digest {:016x} (FNV-1a over the f32 bits)",
            Fnv1a64::default().f32s(&r.1).value()
        );
        println!(
            "step {name}: position {pos} after {} fed ({:.1} s, runtime value); argmax ours={top} \
             ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, our distance at the two \
             {dist:.4}, logits_rel {logits_rel:.3e} against the last layer's {input_rel:.3e}, a \
             tie's bound {HEAD_RATIO:.1}x it{}); worst l_out_rel={:.3e} (band on layers 0..{held}, \
             off every flip's path; the rest printed) {}",
            prefill.len(),
            t.elapsed().as_secs_f64(),
            if tie { ", a named tie" } else { "" },
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The last token eagerly with the taps armed: its layer outputs and
    /// its logits.
    fn run_last(m: &mut Glm5nextModel, tok: u32) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        set_taps(m, true)?;
        m.step(&[tok])?;
        let logits = m.logits()?;
        let taps = {
            let (gpu, _, b) = m.body_parts("run_last")?;
            b.taps(gpu)?
        };
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        Ok((taps, logits))
    }

    // ------------------------------------- (h) head fault, (a) taps, (o) owner

    /// The final norm's gain, whose first value the head clause sets to NaN
    /// and puts back.
    const HEAD_GAIN: &str = "output_norm.weight";

    /// Write `bytes` over the first value of [`HEAD_GAIN`] and return the
    /// bytes it replaced.
    fn patch_head_gain(m: &mut Glm5nextModel, bytes: [u8; 4]) -> Result<[u8; 4], GateError> {
        let (gpu, w, _) = m.body_parts("gate_glm5next_e2e patch_head_gain")?;
        let Some(DevWeight::F32 { w: gain, .. }) = w.get(HEAD_GAIN) else {
            return Err(format!("{HEAD_GAIN} is not resident as F32").into());
        };
        patch_bytes(gpu.stream(), gain.buf(), 0, bytes)
    }

    /// (h): a NaN gain makes every logit NaN; the step is the head's
    /// [`FaultSite::Logit`], the model poisoned, and with the gain put back a
    /// reset steps to the clean token.
    fn head_fault(m: &mut Glm5nextModel, tok: u32, clean: u32) -> Result<bool, GateError> {
        let want = Fault::at(LAYER_HEAD, FaultSite::Logit);
        m.reset()?;
        let old = patch_head_gain(m, f32::NAN.to_le_bytes())?;
        let first = m.step(&[tok]);
        let poisoned = m.poisoned();
        patch_head_gain(m, old)?;
        m.reset()?;
        let again = m.step(&[tok]);
        m.reset()?;
        let named = matches!(&first, Err(GpuError::Fault { fault, .. }) if *fault == want);
        let ok = named && poisoned == Some(want) && matches!(again, Ok(t) if t == clean);
        println!(
            "head fault: NaN in {HEAD_GAIN}[0], a step: {} (want the output head's fault at site {}), poisoned {}; \
             the gain put back and a reset: {} (want token {clean}) {}",
            match &first {
                Ok(t) => format!("token {t}"),
                Err(e) => format!("error \"{e}\""),
            },
            FaultSite::Logit.name(),
            poisoned.map_or_else(|| "none".to_string(), |f| f.to_string()),
            match &again {
                Ok(t) => format!("token {t}"),
                Err(e) => format!("error \"{e}\""),
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (a): a graph step captures the chain; taps armed after it, the next
    /// step's taps are the eager run's at that position, bit for bit.
    fn taps_after_capture(
        m: &mut Glm5nextModel,
        toks: &[u32],
        eager: &Run,
    ) -> Result<bool, GateError> {
        m.reset()?;
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        m.step(&toks[..1])?;
        let captured = m.has_capture();
        set_taps(m, true)?;
        let mode = m.mode();
        m.step(&toks[1..2])?;
        let taps = {
            let (gpu, _, b) = m.body_parts("taps_after_capture")?;
            b.taps(gpu)?
        };
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let same = eager.taps.get(1).is_some_and(|e| same_bits(&taps, e));
        let ok = captured && mode == StepMode::Eager && same;
        println!(
            "taps after a capture: captured {captured}, then armed: mode {mode:?} (want Eager); the \
             next step's taps are the eager run's bit for bit: {same} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// (o): a failure planted on either side of a step's launch, graph
    /// mode, against the plain graph run's logits.
    fn position_owner(m: &mut Glm5nextModel, toks: &[u32], graph: &Run) -> Result<bool, GateError> {
        m.reset()?;
        m.set_mode(StepMode::Graph);
        m.step(&toks[..2])?;
        let plant = |m: &mut Glm5nextModel, p: Plant| -> Result<(), GateError> {
            m.body_parts("position_owner")?.2.plant(p);
            Ok(())
        };
        let text = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        plant(m, Plant::BeforeLaunch)?;
        let before = m.step(&toks[2..3]);
        let at_before = m.pos();
        let rerun = m.step(&toks[2..3]);
        let rerun_bits = rerun.is_ok()
            && graph
                .logits
                .get(2)
                .is_some_and(|g| m.logits().is_ok_and(|l| same_bits(&l, g)));
        let before_ok = matches!(&before, Err(GpuError::State { .. }))
            && m.poisoned().is_none()
            && at_before == 2
            && rerun_bits;
        println!(
            "position owner: a failure before the launch at 2: {} at position {at_before} (want \
             2); the step again: {}, its logits the plain run's bit for bit: {rerun_bits} {}",
            text(&before),
            text(&rerun),
            verdict(before_ok)
        );
        plant(m, Plant::AfterLaunch)?;
        let after = m.step(&toks[3..4]);
        let at_after = m.pos();
        let kept = m.body("position_owner")?.kept(u32::MAX, at_after);
        let again = m.step(&toks[3..4]);
        let refused = matches!(&again, Err(e)
            if e.to_string().contains("failed after its chain was launched"));
        let after_ok = matches!(&after, Err(GpuError::State { .. }))
            && m.poisoned().is_none()
            && at_after == 3
            && kept.at <= at_after
            && refused;
        println!(
            "position owner: a failure after the launch at 3: {} at position {at_after} (want 3); \
             a cut keeps {kept} (want at most 3); the step again: {} (want refused by name) {}",
            text(&after),
            text(&again),
            verdict(after_ok)
        );
        m.reset()?;
        Ok(before_ok && after_ok)
    }

    // ------------------------------------------------ (v) the verify

    /// `m` from a reset, the plain steps of `toks` in graph mode: every
    /// store's digest after them.
    fn digests_after(m: &mut Glm5nextModel, toks: &[u32]) -> Result<Vec<StoreDigest>, GateError> {
        m.reset()?;
        m.set_mode(StepMode::Graph);
        m.step(toks)?;
        Ok(store_digests(m)?)
    }

    /// The first store whose digest differs, by layer and name.
    fn same_stores(got: &[StoreDigest], want: &[StoreDigest]) -> Result<(), String> {
        match first_store_diff(got, want) {
            None => Ok(()),
            Some(d) => Err(d),
        }
    }

    /// Whether the last step's logits are `want`'s bit for bit.
    fn logits_are(m: &Glm5nextModel, want: Option<&Vec<f32>>) -> bool {
        want.is_some_and(|w| m.logits().is_ok_and(|l| same_bits(&l, w)))
    }

    /// (v) in `mode`: the accepted verify, then the rejected one, against
    /// the plain graph run of `toks` and the digests of its first three and
    /// two steps.
    fn verify_mode(
        m: &mut Glm5nextModel,
        mode: StepMode,
        toks: &[u32],
        graph: &Run,
        d3: &[StoreDigest],
    ) -> Result<bool, GateError> {
        let text = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        m.reset()?;
        m.set_mode(mode);
        m.step(&toks[..1])?;
        let rows = m.step_rows::<2>([toks[1], toks[2]])?;
        let logits = m.rows_logits::<2>()?;
        let rows_ok = rows == [graph.tokens[1], graph.tokens[2]]
            && same_bits(&logits[0], &graph.logits[1])
            && same_bits(&logits[1], &graph.logits[2]);
        let at = m.pos();
        let waiting = m.step(&toks[3..4]);
        let refused = matches!(&waiting, Err(e) if e.to_string().contains("waits for its commit"))
            && m.pos() == at;
        m.rollback(3)?;
        m.keep_rows(KeptRows::prefix(2), PassKind::Pair)?;
        let accept_stores = same_stores(&store_digests(m)?, d3);
        let next = m.step(&toks[3..4]);
        let accept_ok = matches!(next, Ok(t) if t == graph.tokens[3])
            && logits_are(m, graph.logits.get(3))
            && accept_stores.is_ok();
        println!(
            "verify ({mode:?}): rows 1, 2 {rows:?} (want {:?}), logits bit for bit: {rows_ok}; a \
             step while it waits: {} (want refused by name, the model left at {at}): {refused}; \
             both kept: stores {} (want three steps'), the step at 3 {} (want token {}, the plain \
             logits) {}",
            [graph.tokens[1], graph.tokens[2]],
            text(&waiting),
            accept_stores
                .as_ref()
                .map_or_else(|d| format!("differ at {d}"), |()| "equal".to_string()),
            text(&next),
            graph.tokens[3],
            verdict(rows_ok && refused && accept_ok)
        );
        m.reset()?;
        m.step(&toks[..1])?;
        let wrong = (toks[2] + 1) % N_VOCAB as u32;
        let rows = m.step_rows::<2>([toks[1], wrong])?;
        let row0 = m.rows_logits::<2>()?;
        let row0_ok = rows[0] == graph.tokens[1] && same_bits(&row0[0], &graph.logits[1]);
        m.rollback(2)?;
        m.keep_rows(KeptRows::prefix(1), PassKind::Pair)?;
        let at2 = m.step(&toks[2..3]);
        let at2_ok =
            matches!(at2, Ok(t) if t == graph.tokens[2]) && logits_are(m, graph.logits.get(2));
        // After the step at 2 the stores are three steps': the state grew from
        // the kept row's lane, and the step wrote over the rejected row's ring
        // slot.
        let reject_stores = if at2.is_ok() {
            same_stores(&store_digests(m)?, d3)
        } else {
            Err("the step at 2 failed".to_string())
        };
        let at3 = m.step(&toks[3..4]);
        let at3_ok =
            matches!(at3, Ok(t) if t == graph.tokens[3]) && logits_are(m, graph.logits.get(3));
        let reject_ok = row0_ok && at2_ok && reject_stores.is_ok() && at3_ok;
        println!(
            "verify ({mode:?}): a wrong draft {wrong} rejected, row 0 kept: row 0 token {} (want \
             {}), logits bit for bit {row0_ok}; the step at 2 {} (want token {}), stores {} (want \
             three steps'); the step at 3 {} (want token {}) {}",
            rows[0],
            graph.tokens[1],
            text(&at2),
            graph.tokens[2],
            reject_stores
                .as_ref()
                .map_or_else(|d| format!("differ at {d}"), |()| "equal".to_string()),
            text(&at3),
            graph.tokens[3],
            verdict(reject_ok)
        );
        m.reset()?;
        m.set_mode(StepMode::Graph);
        Ok(rows_ok && refused && accept_ok && reject_ok)
    }

    /// (v): the captured verify holds the step's nodes twice (each row the
    /// step's launches), then [`verify_mode`] in graph mode and in eager.
    fn verify_rows(
        m: &mut Glm5nextModel,
        toks: &[u32],
        graph: &Run,
        step_nodes: usize,
    ) -> Result<bool, GateError> {
        if toks.len() < 4 || graph.tokens.len() < 4 {
            return Err(format!(
                "the verify clause reads 4 tokens, the set has {}",
                toks.len()
            )
            .into());
        }
        m.reset()?;
        m.set_mode(StepMode::Graph);
        let pair_nodes = m.capture_rows::<2>()?;
        let mut ok = pair_nodes == 2 * step_nodes;
        println!(
            "verify: the captured verify of 2 rows holds {pair_nodes} nodes (want twice the \
             step's {step_nodes}) {}",
            verdict(ok)
        );
        let d3 = digests_after(m, &toks[..3])?;
        for mode in [StepMode::Graph, StepMode::Eager] {
            ok &= verify_mode(m, mode, toks, graph, &d3)?;
        }
        Ok(ok)
    }

    // ------------------------------------------------ (k) checkpoints

    /// The first prompt's ids: past one inner mark (512) and short of the
    /// next.
    const A: usize = 640;

    /// The logits row a session call read back.
    fn row(out: Out<'_>) -> Result<Vec<f32>, GateError> {
        match out {
            Out::Logits { row, .. } => Ok(row.to_vec()),
            Out::Argmax(_) => Err("a call asked for logits read back none".into()),
        }
    }

    /// (k) on the steps feed, then on the batch feed at groups of two
    /// batches; the session left on the steps feed at groups of one.
    fn keep_groups(s: &mut Session<Body>) -> Result<bool, GateError> {
        let mut ok = keep(s, "steps")?;
        let m = s.model_mut();
        set_prefill(m, PrefillMode::Batch)?;
        set_prefill_group(m, 2)?;
        ok &= keep(s, "batches, groups of 2")?;
        let m = s.model_mut();
        set_prefill_group(m, 1)?;
        set_prefill(m, PrefillMode::Steps)?;
        s.reset()?;
        Ok(ok)
    }

    fn keep(s: &mut Session<Body>, feed: &str) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let ids = prefill.to_vec();
        if ids.len() < 964 {
            return Err(format!("{D1K}: {} prefill ids, the clause reads 964", ids.len()).into());
        }
        let (a, d, e) = (&ids[..A], &ids[700..900], &ids[900..964]);
        let points = |s: &Session<Body>| -> Result<Vec<u32>, GateError> {
            Ok(s.model().body("keep")?.checkpoints().positions())
        };
        let mut ok = true;
        // One take into a slot made for it, then one into a reused slot:
        // the copy alone, and the copy with the slot's pinned allocation.
        let mut take_ms = [0.0f64; 2];
        for ms in &mut take_ms {
            s.reset()?;
            s.step(ids[0], Want::Argmax)?;
            let pos = s.model().pos();
            let (gpu, _, b) = s.model_mut().body_parts("keep")?;
            let t = Instant::now();
            b.checkpoint(gpu, pos)?;
            *ms = t.elapsed().as_secs_f64() * 1e3;
        }
        println!(
            "keep ({feed}): a checkpoint's take {:.2} ms with its slot made, {:.2} ms into a reused slot \
             (runtime values)",
            take_ms[0], take_ms[1]
        );
        let t = Instant::now();
        s.reset()?;
        s.prompt(a, Want::Argmax)?;
        let first = t.elapsed().as_secs_f64();
        s.step(ids[A], Want::Argmax)?;
        s.step(ids[A + 1], Want::Argmax)?;
        let p1 = points(s)?;
        let (k600, k500) = (s.kept(600), s.kept(500));
        let refused = match s.cut(600) {
            Err(SessionError::Refused(t)) => t,
            other => format!("not refused: {other:?}"),
        };
        let first_ok = p1 == [512, A as u32]
            && (k600.at, k600.why.code()) == (512, "checkpoint")
            && (k500.at, k500.why.code()) == (0, "no-checkpoint")
            && refused.contains("checkpoint: the recurrent state copied back at 512");
        println!(
            "keep ({feed}): prompt of {A} ({first:.1} s, runtime value) and 2 steps; points {p1:?} (want \
             [512, {A}]); kept(600) = {k600}; kept(500) = {k500}; cut(600): {refused} {}",
            verdict(first_ok)
        );
        ok &= first_ok;
        s.cut(512)?;
        s.prompt(d, Want::Argmax)?;
        let p2 = points(s)?;
        s.step(ids[A], Want::Argmax)?;
        s.step(ids[A + 1], Want::Argmax)?;
        s.cut(712)?;
        // A step first: the restore runs in the step's refresh, not in a
        // prompt call's first take.
        s.step(e[0], Want::Argmax)?;
        let l712 = row(s.prompt(&e[1..], Want::Logits)?)?;
        let k700 = s.kept(700);
        s.cut(k700.at)?;
        let l512 = row(s.prompt(e, Want::Logits)?)?;
        let branch_ok = p2 == [512, 712] && k700.at == 512;
        println!(
            "keep ({feed}): cut to 512, branch of {}: points {p2:?} (want [512, 712]); kept(700) = {k700} \
             {}",
            d.len(),
            verdict(branch_ok)
        );
        ok &= branch_ok;
        let stats = s.model().body("keep")?.checkpoints().stats();
        // The references, from a reset with no cut. On the steps feed plain
        // steps: no mark, no take. On the batch feed the same calls the
        // branches made, so each token runs in a batch of the size it ran in
        // there — a batch's bits depend on its side of GEMM_FROM — and the
        // calls' takes are the references' too.
        let batch = prefill_mode(s.model())? == PrefillMode::Batch;
        let (f512, f712) = if batch {
            s.reset()?;
            s.prompt(&a[..512], Want::Argmax)?;
            let f512 = row(s.prompt(e, Want::Logits)?)?;
            s.reset()?;
            s.prompt(&a[..512], Want::Argmax)?;
            s.prompt(d, Want::Argmax)?;
            s.step(e[0], Want::Argmax)?;
            (f512, row(s.prompt(&e[1..], Want::Logits)?)?)
        } else {
            let plain = |s: &mut Session<Body>, ids: &[u32]| -> Result<Vec<f32>, GateError> {
                s.reset()?;
                let m = s.model_mut();
                m.step(ids)?;
                Ok(m.logits()?)
            };
            (
                plain(s, &[&a[..512], e].concat())?,
                plain(s, &[&a[..512], d, e].concat())?,
            )
        };
        let reference = if batch {
            "the same calls from a reset with no cut"
        } else {
            "plain steps of the same ids from a reset"
        };
        s.reset()?;
        let t512 = row(s.prompt(&[&a[..512], e].concat(), Want::Logits)?)?;
        let read_only = same_bits(&t512, &f512);
        println!(
            "keep ({feed}): a prompt call's takes leave the logits of {}, bit for bit: \
             {read_only} {}",
            if batch {
                "a call of 512 then one of the tail"
            } else {
                "plain steps"
            },
            verdict(read_only)
        );
        ok &= read_only;
        let bits = same_bits(&l712, &f712) && same_bits(&l512, &f512);
        let c = s.model().body("keep")?.checkpoints();
        println!(
            "keep ({feed}): restored at 712 and at 512, a tail of {} each, against {reference}: \
             logits bit for bit {bits} (argmax {} / {} and {} / {}); {} \
             taken, {} restored, {} dropped, {} evicted before the references; {} slots made of \
             {}, {} bytes a checkpoint {}",
            e.len(),
            argmax(&l712),
            argmax(&f712),
            argmax(&l512),
            argmax(&f512),
            stats.taken,
            stats.restored,
            stats.dropped,
            stats.evicted,
            c.slots(),
            c.capacity(),
            c.bytes(),
            verdict(bits)
        );
        ok &= bits;
        Ok(ok)
    }

    // ------------------------------------------------- (slots) resident slots

    /// The resident slots' load: a slot's context, and the slots its plan
    /// counts — the harness's streams, one a slot.
    const SLOT_CTX: usize = 256;
    const SLOTS: usize = slots_gate::STREAMS;
    /// The drafted passes each slot runs after its prompt's step, and (s3)'s
    /// after the resume.
    const SLOT_PASSES: usize = 8;
    const SLOT_RESUMED: usize = 6;
    /// The rows a verify runs: the target's next token and the proposal.
    const PAIR: usize = <Body as MtpBody>::VERIFY_ROWS;

    /// What one slot's drafted run left at a point: its ids from the
    /// prompt's step on, each pass's proposal and kept rows, and the logits
    /// of its last step.
    #[derive(Clone)]
    struct SlotRun {
        ids: Vec<u32>,
        passes: Vec<(bool, usize)>,
        logits: Vec<f32>,
    }

    impl SlotRun {
        fn same(&self, other: &SlotRun) -> bool {
            self.ids == other.ids
                && self.passes == other.passes
                && same_bits(&self.logits, &other.logits)
        }

        fn last(&self) -> Result<u32, GateError> {
            Ok(*self.ids.last().ok_or("a slot run with no id")?)
        }
    }

    /// The rows-log that hears nothing.
    struct Quiet;

    impl app::RowsLog for Quiet {
        fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// The window a drafted slot run drives, `PAIR` rows a verify.
    type Spec = runtime::Speculative<MtpDraft<Body>, PAIR>;

    /// The (slots) body for the slot harness ([`slots_gate`]): the NextN load
    /// at [`SLOT_CTX`] positions on the gate placement, its plan counting the
    /// slots it serves (`PlanInputs::plan_nextn_slots`), stream `i`'s prompt
    /// `prompts[i]` fed as the server feeds it ([`server_start`]); and the
    /// inputs it is planned from.
    struct GlmSlots<'a> {
        levers: &'a bloomery_levers::Levers,
        inputs: PlanInputs,
        nextn: NextnInputs,
        prompts: [&'a [u32]; SLOTS],
    }

    impl GlmSlots<'_> {
        /// The plan of `slots` sequences on `machine` the load runs by.
        fn plan<'p>(
            &'p self,
            machine: &'p Machine,
            slots: usize,
        ) -> Result<NextnPlan<'p>, GateError> {
            let place = PlanLevers::from_levers(self.levers)?;
            let ctx = u64::try_from(SLOT_CTX)?;
            Ok(self
                .inputs
                .plan_nextn_slots(machine, ctx, &place, &self.nextn, slots)?)
        }

        /// One sequence's card bytes as the plan's descriptor counts them
        /// (`PlanInputs::seq_terms`, `SeqTerms::bytes` of one).
        fn seq_terms_one(&self) -> Result<u64, GateError> {
            let kv = self.inputs.kv.with_lanes(KdaLanes::Two);
            let terms = self.inputs.seq_terms(&kv, Some(&self.nextn));
            Ok(terms.bytes(u64::try_from(SLOT_CTX)?, 1))
        }
    }

    impl SlotsAdapter for GlmSlots<'_> {
        type Body = Body;

        const STEPS: usize = 12;
        const TAIL: usize = 6;

        fn open(&self, slots: usize) -> Result<Glm5nextModel, GateError> {
            let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
            let machine = workstation::plan_gate(self.inputs.model.layers);
            let np = self.plan(&machine, slots)?;
            let t = Instant::now();
            let mut m = Body::open_placed_nextn_slots(
                file,
                &np,
                &self.inputs,
                &self.nextn,
                0,
                self.levers.host(),
                Residency::Off,
                slots,
            )?;
            set_prefill(&mut m, PrefillMode::Batch)?;
            println!(
                "slots load resident_bytes={} ctx={SLOT_CTX} slots={slots} in {:.1} s (runtime \
                 value)",
                m.resident_bytes(),
                t.elapsed().as_secs_f64()
            );
            Ok(m)
        }

        /// The reset: it zeroes every store whole and the lane word, so the
        /// stores [`SlotsAdapter::state_hash`] reads hold no other stream's
        /// rows past the position.
        fn rewind(&self, m: &mut Glm5nextModel) -> Result<(), GateError> {
            Ok(m.reset()?)
        }

        fn prompt(&self, m: &mut Glm5nextModel, stream: usize) -> Result<u32, GateError> {
            let p = self
                .prompts
                .get(stream)
                .ok_or_else(|| format!("(slots) runs streams 0 and 1, not {stream}"))?;
            server_start(m, p)
        }

        /// Every trunk layer's stores digested ([`store_digests`]: a KDA
        /// layer's committed lane and its conv ring, a latent layer's rows
        /// and pools, every row of them).
        fn state_hash(&self, m: &mut Glm5nextModel) -> Result<u64, GateError> {
            stores_hash(m)
        }

        /// The stores (`Body::store_bytes`, the allocations the load holds to
        /// its plan), the lane word (one device word), each row's final
        /// streams and the verify's row-0 copy (`(lanes + 1) · streams ·
        /// n_embd` f32), and the next-token layer's store ([`SLOT_CTX`]
        /// latent and index rows and the pools over them in f16, at
        /// `bloomery_gpu::latent`'s widths).
        fn seq_bytes_derived(&self, m: &Glm5nextModel) -> Result<Derived, GateError> {
            let body = m.body("slots")?;
            let stores = body.store_bytes();
            let hp = &self.inputs.hp;
            let rows = (body.lanes().count() + 1) * hp.hc.streams * hp.n_embd * 4;
            let layer = (SLOT_CTX * (LATENT + INDEX_ROW) + pools_for(SLOT_CTX) * INDEX_HEAD) * 2;
            Ok(Derived {
                bytes: stores + 4 + rows + layer,
                terms: format!(
                    "store_bytes {stores} + the lane word 4 + the rows' final streams and the \
                     row-0 copy {rows} + the next-token layer's store {layer}"
                ),
            })
        }

        fn seq_terms_bytes(&self, _m: &Glm5nextModel) -> Result<Option<usize>, GateError> {
            Ok(Some(usize::try_from(self.seq_terms_one()?)?))
        }
    }

    /// The prompt as the server feeds it: every id but the last in one call
    /// by the load's feed ([`feed`]), then the last one step; the step's
    /// argmax.
    fn server_start(m: &mut Glm5nextModel, p: &[u32]) -> Result<u32, GateError> {
        let (&last, head) = p.split_last().ok_or("an empty prompt")?;
        feed(m, head)?;
        Ok(m.step(&[last])?)
    }

    /// (sp) the plan's count (module header): a slot past it refused by name,
    /// and the plan of [`SLOTS`] counting one sequence's bytes past the plan
    /// of one for each slot past the first. Moves no slot.
    fn slots_plan(a: &GlmSlots<'_>, m: &mut Glm5nextModel) -> Result<bool, GateError> {
        let past = m.add_slots(SLOTS + 1);
        let named = matches!(&past, Err(e) if e.to_string().contains(&format!(
            "of a load whose plan counted {SLOTS} resident sequences"
        )));
        let machine = workstation::plan_gate(a.inputs.model.layers);
        let counted = |n: usize| -> Result<u64, GateError> {
            let np = a.plan(&machine, n)?;
            Ok(np.plan.cards[0].kv_bytes + np.nextn.cards[0].kv_bytes)
        };
        let grown = counted(SLOTS)? - counted(1)?;
        let one = a.seq_terms_one()?;
        let counts = grown == one * u64::try_from(SLOTS - 1)?;
        println!(
            "slots (sp) plan: add_slots({}) on a plan of {SLOTS} -> {}; the plan of {SLOTS} counts \
             {grown} B past the plan of one (its stage card's KV term and the layer's store), one \
             sequence {one} B a slot past the first {}",
            SLOTS + 1,
            match &past {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(named && counts)
        );
        Ok(named && counts)
    }

    /// (sc) after the interleave (module header): slot 1 cut back to its
    /// prompt call's checkpoint, slot 0 selected and stepped on its solo
    /// run's continuation ([`Interleaved::continues`]) before slot 1 takes
    /// the step that copies the cut back, then slot 1 stepped again to where
    /// it stood: its last id, position and state hash as before the cut.
    fn slots_cut(
        s: &mut Interleaved<'_, GlmSlots<'_>>,
        a: &GlmSlots<'_>,
    ) -> Result<bool, GateError> {
        let b = a.prompts[1];
        let (&b_last, _) = b.split_last().ok_or("an empty prompt")?;
        let last = s.last(1)?;
        let m = s.model();
        m.select_slot(1)?;
        let (pos, hash) = (m.pos(), a.state_hash(m)?);
        let at = u32::try_from(b.len() - 1)?;
        let kept = m.body("slots")?.kept(at, pos);
        m.rollback(at)?;
        let on = s.continues(0)?;
        let m = s.model();
        m.select_slot(1)?;
        let mut id = m.step(&[b_last])?;
        while m.pos() < pos {
            id = m.step(&[id])?;
        }
        let again = (id, m.pos(), a.state_hash(m)?) == (last, pos, hash);
        let pass = kept.at == at && on && again;
        println!(
            "slots (sc) cut: slot 1 at {pos} cut to {at} (kept({at}) = {kept}), slot 0's next {} \
             ids its solo run's {on}; slot 1 stepped again to {}: last id {id} (before {last}), \
             stores as before the cut {again} {}",
            GlmSlots::TAIL,
            m.pos(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Make `slot` live with its draft's side: the live slot's parked into
    /// `parked`, the target's put back — the seat's table, by hand.
    fn select_drafted(
        s: &mut Session<Body>,
        spec: &mut Spec,
        parked: &mut [Parked<GlmArena>],
        slot: usize,
    ) -> Result<(), GateError> {
        let was = s.selected();
        if was == slot {
            return Ok(());
        }
        let live = spec.draft().park();
        s.select_slot(slot)?;
        let back = parked
            .get(slot)
            .ok_or("a slot past the draft table")?
            .clone();
        *parked.get_mut(was).ok_or("a slot past the draft table")? = live;
        spec.draft_mut().unpark(&back);
        Ok(())
    }

    /// The live slot and its draft from empty.
    fn reset_drafted(s: &mut Session<Body>, spec: &mut Spec) -> Result<(), GateError> {
        s.reset()?;
        spec.draft_mut().restart();
        Ok(())
    }

    /// One step of `last` under the draft, as the seat steps
    /// (`DraftedSeat::step`): the draft's waiting rows walked, the step, the
    /// step told to the draft; its argmax, and its logits when asked.
    fn drafted_step(
        s: &mut Session<Body>,
        spec: &mut Spec,
        last: u32,
        logits: bool,
    ) -> Result<(u32, Option<Vec<f32>>), GateError> {
        spec.draft_mut().before_step(s, last)?;
        let next = s.step(last, Want::Argmax)?.argmax();
        let row = if logits {
            Some(s.model().logits()?)
        } else {
            None
        };
        runtime::Draft::stepped(spec.draft_mut(), s, last, next)?;
        Ok((next, row))
    }

    /// The prompt as the server feeds it under the draft: every id but the
    /// last through the draft's prompt call, then the last one step.
    fn drafted_start(
        s: &mut Session<Body>,
        spec: &mut Spec,
        p: &[u32],
    ) -> Result<SlotRun, GateError> {
        let (&last, head) = p.split_last().ok_or("an empty prompt")?;
        runtime::Advance::prompt(spec, s, head)?;
        let (first, _) = drafted_step(s, spec, last, false)?;
        Ok(SlotRun {
            ids: vec![first],
            passes: Vec::new(),
            logits: Vec::new(),
        })
    }

    /// `n` drafted passes from `r`'s last id, then one step that reads its
    /// logits.
    fn drafted_passes(
        s: &mut Session<Body>,
        spec: &mut Spec,
        r: &mut SlotRun,
        n: usize,
    ) -> Result<(), GateError> {
        let mut out = Vec::new();
        for _ in 0..n {
            out.clear();
            let c = runtime::Advance::pass(spec, s, r.last()?, &mut out)?;
            r.passes.push((c.proposed, c.kept));
            r.ids.extend_from_slice(&out);
        }
        let (next, row) = drafted_step(s, spec, r.last()?, true)?;
        r.ids.push(next);
        r.logits = row.ok_or("a step that read no logits")?;
        Ok(())
    }

    /// `p`'s drafted run on `slot` from a reset: [`SLOT_PASSES`] passes and a
    /// step, then [`SLOT_RESUMED`] passes and a step.
    fn drafted_solo(
        s: &mut Session<Body>,
        spec: &mut Spec,
        parked: &mut [Parked<GlmArena>],
        slot: usize,
        p: &[u32],
    ) -> Result<[SlotRun; 2], GateError> {
        select_drafted(s, spec, parked, slot)?;
        reset_drafted(s, spec)?;
        let mut r = drafted_start(s, spec, p)?;
        drafted_passes(s, spec, &mut r, SLOT_PASSES)?;
        let at = r.clone();
        drafted_passes(s, spec, &mut r, SLOT_RESUMED)?;
        Ok([at, r])
    }

    /// (sd) drafted: `a` on slot 0 and `b` on slot 1 under the draft, a
    /// select between every pass, each slot's ids, passes and last logits
    /// its solo run's; both slots' runs, for (s3).
    fn slots_drafted(
        s: &mut Session<Body>,
        spec: &mut Spec,
        parked: &mut [Parked<GlmArena>],
        (a, b): (&[u32], &[u32]),
        (solo_a, solo_b): (&SlotRun, &SlotRun),
    ) -> Result<(bool, [SlotRun; 2]), GateError> {
        for slot in [1, 0] {
            select_drafted(s, spec, parked, slot)?;
            reset_drafted(s, spec)?;
        }
        let mut ra = drafted_start(s, spec, a)?;
        select_drafted(s, spec, parked, 1)?;
        let mut rb = drafted_start(s, spec, b)?;
        let mut out = Vec::new();
        for _ in 0..SLOT_PASSES {
            for (slot, r) in [(0, &mut ra), (1, &mut rb)] {
                select_drafted(s, spec, parked, slot)?;
                out.clear();
                let c = runtime::Advance::pass(spec, s, r.last()?, &mut out)?;
                r.passes.push((c.proposed, c.kept));
                r.ids.extend_from_slice(&out);
            }
        }
        for (slot, r) in [(0, &mut ra), (1, &mut rb)] {
            select_drafted(s, spec, parked, slot)?;
            drafted_passes(s, spec, r, 0)?;
        }
        let (a_ok, b_ok) = (ra.same(solo_a), rb.same(solo_b));
        let accepts = |r: &SlotRun| r.passes.iter().filter(|&&(p, k)| p && k == PAIR).count();
        println!(
            "slots (sd) drafted: {SLOT_PASSES} passes a slot, a select between every one: slot 0 \
             ids {:?} passes {:?} (solo {:?} {:?}) logits bit for bit {a_ok}; slot 1 ids {:?} \
             passes {:?} (solo {:?} {:?}) bit for bit {b_ok}; accepted windows {} and {} {}",
            ra.ids,
            ra.passes,
            solo_a.ids,
            solo_a.passes,
            rb.ids,
            rb.passes,
            solo_b.ids,
            solo_b.passes,
            accepts(&ra),
            accepts(&rb),
            verdict(a_ok && b_ok)
        );
        Ok((a_ok && b_ok, [ra, rb]))
    }

    /// (s3) after (sd) drafted: slot 1's state saved with its draft's side
    /// (`seq_save`), the slot reset and run on `c`, then put back
    /// (`seq_resume`) and run on, slot 0 running on between: both slots'
    /// continuations their solo runs'.
    fn slots_resume(
        s: &mut Session<Body>,
        spec: &mut Spec,
        parked: &mut [Parked<GlmArena>],
        c: &[u32],
        [mut ra, mut rb]: [SlotRun; 2],
        (solo_a, solo_b): (&SlotRun, &SlotRun),
    ) -> Result<bool, GateError> {
        select_drafted(s, spec, parked, 1)?;
        let state = seq_save(s.model_mut())?;
        let side = spec.draft().park();
        reset_drafted(s, spec)?;
        let mut rc = drafted_start(s, spec, c)?;
        drafted_passes(s, spec, &mut rc, 2)?;
        select_drafted(s, spec, parked, 0)?;
        drafted_passes(s, spec, &mut ra, SLOT_RESUMED)?;
        select_drafted(s, spec, parked, 1)?;
        reset_drafted(s, spec)?;
        seq_resume(s.model_mut(), &state)?;
        spec.draft_mut().unpark(&side);
        drafted_passes(s, spec, &mut rb, SLOT_RESUMED)?;
        let (a_on, b_back) = (ra.same(solo_a), rb.same(solo_b));
        println!(
            "slots (s3) park/resume: slot 1's state of {} positions saved ({} host bytes) with its \
             draft's side, the slot run on {} other ids and put back, then {SLOT_RESUMED} passes: \
             ids {:?} passes {:?} (solo {:?} {:?}) logits bit for bit {b_back}; slot 0 on \
             between, bit for bit its solo run's {a_on} {}",
            state.positions(),
            state.bytes(),
            c.len(),
            rb.ids,
            rb.passes,
            solo_b.ids,
            solo_b.passes,
            verdict(b_back && a_on)
        );
        Ok(b_back && a_on)
    }

    /// A slot clause's verdict, an error it ended in printed as its FAIL
    /// rather than ending the gate.
    fn clause(what: &str, r: Result<bool, GateError>) -> bool {
        r.unwrap_or_else(|e| {
            println!("slots {what}: ended in error \"{e}\" {}", verdict(false));
            false
        })
    }

    /// The (slots) clauses (the module header): the harness's contracts and
    /// the body's own between them on one load, then (sd) and (s3) on a load
    /// of their own.
    fn slots(levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let prefill = slot_prefill()?;
        // Three prompts of distinct ids and lengths: (sc)'s cut restores
        // slot 1's prompt call's checkpoint, which a ledger the slots shared
        // would hold slot 0's in place of.
        let (a, b, c) = slot_prompts(&prefill);
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = PlanInputs::read(&file)?;
        let nextn = NextnInputs::read(&inputs)?;
        drop(file);
        let body = GlmSlots {
            levers,
            inputs,
            nextn,
            prompts: [a, b],
        };
        let mut s = slots_gate::interleave(&body)?;
        let mut ok = clause("(sp) plan", slots_plan(&body, s.model()));
        ok &= clause("(sc) cut", slots_cut(&mut s, &body));
        ok &= clause("(g5) on the NextN load", g5_nextn(s.model()));
        ok &= s.finish()?;
        // (sd) and (s3) drive the MTP draft through a session, which owns
        // its model; the harness lends its own by reference only.
        let mut s = Session::from_model(body.open(SLOTS)?, u32::try_from(SLOT_CTX)?);
        s.model_mut().set_mode(StepMode::Graph);
        s.add_slots(SLOTS)?;
        let draft = MtpDraft::open(s.model(), PrefillMode::Batch, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, PAIR>(draft, &mut Quiet)?;
        let mut parked = vec![spec.draft().park(); SLOTS];
        let solo_a = drafted_solo(&mut s, &mut spec, &mut parked, 0, a)?;
        let solo_b = drafted_solo(&mut s, &mut spec, &mut parked, 0, b)?;
        let solo_passes = || solo_a[1].passes.iter().chain(&solo_b[1].passes);
        let proposed = solo_passes().any(|&(p, k)| p && k == PAIR)
            && solo_passes().any(|&(p, k)| p && k < PAIR);
        println!(
            "slots: the solo drafted runs accept and reject a proposal {}",
            verdict(proposed)
        );
        ok &= proposed;
        let solos = (&solo_a[0], &solo_b[0]);
        match slots_drafted(&mut s, &mut spec, &mut parked, (a, b), solos) {
            Ok((sd, runs)) => {
                ok &= sd;
                let r = slots_resume(
                    &mut s,
                    &mut spec,
                    &mut parked,
                    c,
                    runs,
                    (&solo_a[1], &solo_b[1]),
                );
                ok &= clause("(s3) park/resume", r);
            }
            Err(e) => {
                ok &= clause("(sd) drafted", Err(e));
                ok &= clause("(s3) park/resume", Err("(sd) drafted ended first".into()));
            }
        }
        // The stagger's loads hold the card next: the drafted load goes first.
        drop((spec, parked, s));
        ok &= stagger(levers, &body.inputs, (a, b))?;
        Ok(ok)
    }

    /// The prose set's prefill ids the slot clauses cut their prompts from.
    fn slot_prefill() -> Result<Vec<u32>, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (_, _, prefill) = man.step()?;
        if prefill.len() < 700 {
            return Err(
                format!("{D1K}: {} prefill ids, the clause reads 700", prefill.len()).into(),
            );
        }
        Ok(prefill.to_vec())
    }

    /// The slot clauses' three prompts out of `prefill` ([`slot_prefill`]):
    /// distinct ids, lengths 33, 40 and 30.
    fn slot_prompts(prefill: &[u32]) -> (&[u32], &[u32], &[u32]) {
        (&prefill[..33], &prefill[300..340], &prefill[600..630])
    }

    /// Every trunk layer's stores on the selected slot digested into one
    /// hash ([`store_digests`]: a KDA layer's committed lane and its conv
    /// ring, a latent layer's rows and pools, every row of them).
    fn stores_hash(m: &mut Glm5nextModel) -> Result<u64, GateError> {
        let h = store_digests(m)?
            .iter()
            .fold(Fnv1a64::default(), |h, d| h.bytes(&d.fnv.to_le_bytes()));
        Ok(h.value())
    }

    // ------------------------------------------ (g) a pass of two slots

    /// (g1)'s passes of both slots, and (g2)'s after slot 1's cut.
    const G_ROUNDS: usize = 16;
    const G_AFTER_CUT: usize = 4;

    /// Slot 0's and slot 1's steps past their prompts at the end of (g1):
    /// the passes of both, slot 1's pass alone, a plain step of each.
    const G1_STEPS: [usize; 2] = [G_ROUNDS + 1, G_ROUNDS + 2];

    /// (g1b)'s passes of slots 1 and 2.
    const G1B_ROUNDS: usize = 4;

    /// The one-lane load's slots: (g1b)'s pass of slots 1 and 2 leaves slot
    /// 0, the live one, idle.
    const G_SLOTS: usize = 3;

    /// What `GpuModel::step_slots` names its refusals and its poisoned slot
    /// set by, and the body's refusals.
    const STEP_SLOTS: &str = "GpuModel::step_slots";
    const GLM_BODY: &str = "glm5next Body";

    /// The selected slot's stores digested two ways: every row they hold
    /// ([`stores_hash`]), and only what a step reads ([`live_hash`]).
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct GStores {
        whole: u64,
        live: u64,
    }

    impl GStores {
        fn of(m: &mut Glm5nextModel) -> Result<GStores, GateError> {
            Ok(GStores {
                whole: stores_hash(m)?,
                live: live_hash(m)?,
            })
        }
    }

    /// Which digest of a slot's stores [`g_off`] holds to its solo run's:
    /// every row, or, for a slot cut back since its reset, the live ones. A
    /// cut copies the recurrent state back ([`Body::kept`]) and leaves the
    /// latent rows past its position as they were, which no later step reads
    /// and a run from a reset holds zeroed.
    #[derive(Clone, Copy)]
    enum GCompare {
        Whole,
        Live,
    }

    /// The selected slot's stores up to its position ([`store_rows`]): each
    /// KDA layer's committed lane and conv ring, each latent layer's rows
    /// below the position and the pools they complete, digested in order.
    fn live_hash(m: &mut Glm5nextModel) -> Result<u64, GateError> {
        let live = usize::try_from(m.pos())?;
        let h = store_rows(m, live)?
            .iter()
            .fold(Fnv1a64::default(), |h, r| {
                h.bytes(&r.layer.to_le_bytes())
                    .bytes(r.what.as_bytes())
                    .f32s(&r.values)
            });
        Ok(h.value())
    }

    /// One slot's solo run on the load from a reset, slot 0 selected: its
    /// prompt as the server feeds it ([`server_start`]), then greedy steps.
    /// `ids[0]` is the prompt's argmax and `logits[0]` its last step's
    /// logits hash; `ids[t + 1]` and `logits[t + 1]` the step that fed
    /// `ids[t]`. At each step count of the marks, the stores' digests and
    /// the logits.
    struct GSolo {
        ids: Vec<u32>,
        logits: Vec<u64>,
        marks: Vec<(usize, GStores, Vec<f32>)>,
    }

    impl GSolo {
        /// The stores' digests and the logits after `steps` steps past the
        /// prompt.
        fn at(&self, steps: usize) -> Result<(GStores, &[f32]), GateError> {
            self.marks
                .iter()
                .find(|(t, _, _)| *t == steps)
                .map(|(_, h, l)| (*h, l.as_slice()))
                .ok_or_else(|| format!("no solo mark at {steps} steps").into())
        }
    }

    fn g_solo(
        m: &mut Glm5nextModel,
        p: &[u32],
        steps: usize,
        marks: &[usize],
    ) -> Result<GSolo, GateError> {
        m.select_slot(0)?;
        m.reset()?;
        let mut run = GSolo {
            ids: vec![server_start(m, p)?],
            logits: vec![fnv_row(&m.logits()?)],
            marks: Vec::new(),
        };
        for t in 0..=steps {
            if marks.contains(&t) {
                run.marks.push((t, GStores::of(m)?, m.logits()?));
            }
            if t < steps {
                run.ids.push(m.step(&[run.ids[t]])?);
                run.logits.push(fnv_row(&m.logits()?));
            }
        }
        Ok(run)
    }

    /// One slot's ids and per-call logits hashes in passes of both slots,
    /// laid as its [`GSolo`]'s.
    #[derive(Default)]
    struct GRun {
        ids: Vec<u32>,
        logits: Vec<u64>,
    }

    /// One pass of the slots of `order`, in that row order, each fed its
    /// run's last id; each row's id and logits hash appended to its slot's
    /// run. Each row's logits, in `order`.
    fn g_pass(
        m: &mut Glm5nextModel,
        runs: &mut [GRun],
        order: &[usize],
    ) -> Result<Vec<Vec<f32>>, GateError> {
        let lasts = order
            .iter()
            .map(|&s| runs[s].ids.last().copied().ok_or("a slot run with no id"))
            .collect::<Result<Vec<u32>, _>>()?;
        let rows: Vec<(usize, &[u32])> = order
            .iter()
            .zip(&lasts)
            .map(|(&s, t)| (s, std::slice::from_ref(t)))
            .collect();
        let out = m.step_slots(&rows)?;
        let logits = m.slots_logits()?;
        if out.ids.len() != order.len() || logits.len() != order.len() {
            return Err(format!(
                "a pass of slots {order:?} gave {} ids and {} logits rows",
                out.ids.len(),
                logits.len()
            )
            .into());
        }
        for ((&s, &id), row) in order.iter().zip(&out.ids).zip(&logits) {
            runs[s].ids.push(id);
            runs[s].logits.push(fnv_row(row));
        }
        Ok(logits)
    }

    /// One plain step of `slot`, fed its run's last id; its id and logits
    /// hash appended to its run. Its logits.
    fn g_plain(
        m: &mut Glm5nextModel,
        runs: &mut [GRun],
        slot: usize,
    ) -> Result<Vec<f32>, GateError> {
        let run = &mut runs[slot];
        let last = run.ids.last().copied().ok_or("a slot run with no id")?;
        m.select_slot(slot)?;
        run.ids.push(m.step(&[last])?);
        let logits = m.logits()?;
        run.logits.push(fnv_row(&logits));
        Ok(logits)
    }

    /// Slots 0 and 1 each reset, both before either is prompted (a fault
    /// that poisoned both refuses every call until both are reset), then
    /// prompted ([`server_start`]): their runs.
    fn g_prompted(m: &mut Glm5nextModel, (a, b): (&[u32], &[u32])) -> Result<[GRun; 2], GateError> {
        for slot in [0, 1] {
            m.select_slot(slot)?;
            m.reset()?;
        }
        let mut runs = [GRun::default(), GRun::default()];
        for (slot, p) in [(0, a), (1, b)] {
            m.select_slot(slot)?;
            runs[slot] = GRun {
                ids: vec![server_start(m, p)?],
                logits: vec![fnv_row(&m.logits()?)],
            };
        }
        Ok(runs)
    }

    /// The parts of slot `slot`'s state that differ from its solo run `solo`
    /// after `steps` steps past its prompt `p`: its ids and logits hashes
    /// from the prompt on (`run`), its last row's logits bit for bit
    /// (`last`), its stores' digest as `cmp` names it and its position.
    fn g_off(
        m: &mut Glm5nextModel,
        (slot, cmp): (usize, GCompare),
        (run, last): (&GRun, &[f32]),
        (solo, p, steps): (&GSolo, &[u32], usize),
    ) -> Result<Vec<String>, GateError> {
        let (want, logits) = solo.at(steps)?;
        m.select_slot(slot)?;
        let pos = u32::try_from(p.len() + steps)?;
        let (part, stores) = match cmp {
            GCompare::Whole => ("stores", stores_hash(m)? == want.whole),
            GCompare::Live => ("live stores", live_hash(m)? == want.live),
        };
        let parts = [
            ("ids", solo.ids.get(..=steps) == Some(&run.ids[..])),
            ("logits", solo.logits.get(..=steps) == Some(&run.logits[..])),
            ("last logits", same_bits(last, logits)),
            (part, stores),
            ("position", m.pos() == pos),
        ];
        Ok(parts
            .into_iter()
            .filter(|&(_, same)| !same)
            .map(|(part, _)| format!("slot {slot} {part}"))
            .collect())
    }

    /// The stagger's load: the plain load (no next-token layer) of `lanes`
    /// KDA lanes at [`SLOT_CTX`] positions on the gate placement, residency
    /// off, its plan counting `slots` sequences, every slot made, graph
    /// steps, the prompt fed in batches.
    fn g_open(
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        lanes: KdaLanes,
        slots: usize,
    ) -> Result<Glm5nextModel, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let place = PlanLevers::from_levers(levers)?;
        let plan = inputs.plan_slots(&machine, u64::try_from(SLOT_CTX)?, &place, lanes, slots)?;
        let term = plan.machine.cards[0].scratch_bytes;
        let t = Instant::now();
        let mut m = Body::open_placed_slots(
            file,
            &plan,
            inputs,
            0,
            levers.host(),
            Residency::Off,
            lanes,
            slots,
        )?;
        set_prefill(&mut m, PrefillMode::Batch)?;
        m.set_mode(StepMode::Graph);
        m.add_slots(slots)?;
        let body = m.body("stagger")?;
        println!(
            "stagger load lanes={} resident_bytes={} scratch_bytes={} (the plan's card scratch \
             term {term}) ctx={SLOT_CTX} slots={slots} in {:.1} s (runtime value)",
            body.lanes().count(),
            m.resident_bytes(),
            body.scratch_bytes(),
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    /// Whether `r` is `what`'s shape refusal whose words hold `says`.
    fn refused_by<T>(r: &Result<T, GpuError>, what: &str, says: &str) -> bool {
        matches!(r, Err(GpuError::Shape { what: w, detail }) if *w == what && detail.contains(says))
    }

    /// (g1)-(g5) on their two loads (module doc): the one-lane load's, then
    /// the two-lane load's (g4).
    fn stagger(
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        (a, b): (&[u32], &[u32]),
    ) -> Result<bool, GateError> {
        let mut m = g_open(levers, inputs, KdaLanes::One, G_SLOTS)?;
        let [s0, s1] = G1_STEPS;
        let sa = g_solo(
            &mut m,
            a,
            s0 + G_AFTER_CUT,
            &[s0, s0 + G_AFTER_CUT, G1B_ROUNDS],
        )?;
        let sb = g_solo(&mut m, b, s1, &[s1, G_AFTER_CUT - 1, G1B_ROUNDS])?;
        let mut ok = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            m.set_mode(mode);
            match g1_bits(&mut m, (a, b), (&sa, &sb), mode) {
                Ok((bits, runs)) => {
                    ok &= bits;
                    if mode == StepMode::Graph {
                        ok &= clause("(g2) cut", g2_cut(&mut m, (a, b), runs, (&sa, &sb)));
                    }
                }
                Err(e) => {
                    ok &= clause(&format!("(g1) bits {mode:?}"), Err(e));
                    if mode == StepMode::Graph {
                        ok &= clause("(g2) cut", Err("(g1) ended first".into()));
                    }
                }
            }
            ok &= clause(
                &format!("(g1b) parked {mode:?}"),
                g1b_parked(&mut m, (a, b), (&sa, &sb), mode),
            );
        }
        m.set_mode(StepMode::Graph);
        ok &= clause("(g3) fault", g3_fault(&mut m, (a, b), (&sa, &sb)));
        ok &= clause("(g5) refusals", g5_refusals(&mut m, inputs, (a, b)));
        drop(m);
        ok &= clause("(g4) walk", g4_walk(levers, inputs, (a, b)));
        Ok(ok)
    }

    /// (g1) in `mode`: both slots prompted, [`G_ROUNDS`] passes of both
    /// (slot 0's row first, then slot 1's, in turn), one of slot 1 alone,
    /// then a plain step of each; each slot's ids, logits hashes, last
    /// logits bit for bit, stores and position its solo run's. The runs,
    /// for (g2).
    fn g1_bits(
        m: &mut Glm5nextModel,
        (a, b): (&[u32], &[u32]),
        (sa, sb): (&GSolo, &GSolo),
        mode: StepMode,
    ) -> Result<(bool, [GRun; 2]), GateError> {
        let mut runs = g_prompted(m, (a, b))?;
        let mut last = [Vec::new(), Vec::new()];
        for r in 0..G_ROUNDS {
            let order: &[usize] = if r % 2 == 0 { &[0, 1] } else { &[1, 0] };
            for (&s, row) in order.iter().zip(g_pass(m, &mut runs, order)?) {
                last[s] = row;
            }
        }
        for row in g_pass(m, &mut runs, &[1])? {
            last[1] = row;
        }
        // Each slot's plain step replays its own chain over the buffers the
        // passes bound and gave back.
        for (s, l) in last.iter_mut().enumerate() {
            *l = g_plain(m, &mut runs, s)?;
        }
        let mut off = g_off(
            m,
            (0, GCompare::Whole),
            (&runs[0], &last[0]),
            (sa, a, G1_STEPS[0]),
        )?;
        off.extend(g_off(
            m,
            (1, GCompare::Whole),
            (&runs[1], &last[1]),
            (sb, b, G1_STEPS[1]),
        )?);
        let pass = off.is_empty();
        println!(
            "stagger (g1) bits {mode:?}: one lane, {G_ROUNDS} passes of slots 0 and 1 from \
             positions {} and {} (row order 0,1 then 1,0 in turn), one of slot 1 alone, then a \
             plain step of each: {} {}",
            a.len(),
            b.len(),
            if pass {
                "every row's id and logits, each slot's last logits, stores and position bit for \
                 bit its solo run's"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(pass)
        );
        Ok((pass, runs))
    }

    /// (g1b) in `mode` after (g1): slots 1 and 2 reset and prompted with
    /// (g1)'s two prompts, then [`G1B_ROUNDS`] passes of both (row order 1,
    /// 2 and 2, 1 in turn) while slot 0, the live one, stays idle: both rows
    /// a parked sequence's. Each slot its solo run's as in (g1); slot 0's
    /// position and stores as before.
    fn g1b_parked(
        m: &mut Glm5nextModel,
        (a, b): (&[u32], &[u32]),
        (sa, sb): (&GSolo, &GSolo),
        mode: StepMode,
    ) -> Result<bool, GateError> {
        m.select_slot(0)?;
        let idle = (m.pos(), stores_hash(m)?);
        let mut runs = [GRun::default(), GRun::default(), GRun::default()];
        for (slot, p) in [(1, a), (2, b)] {
            m.select_slot(slot)?;
            m.reset()?;
            runs[slot] = GRun {
                ids: vec![server_start(m, p)?],
                logits: vec![fnv_row(&m.logits()?)],
            };
        }
        let mut last = [Vec::new(), Vec::new(), Vec::new()];
        for r in 0..G1B_ROUNDS {
            let order: &[usize] = if r % 2 == 0 { &[1, 2] } else { &[2, 1] };
            for (&s, row) in order.iter().zip(g_pass(m, &mut runs, order)?) {
                last[s] = row;
            }
        }
        let mut off = g_off(
            m,
            (1, GCompare::Whole),
            (&runs[1], &last[1]),
            (sa, a, G1B_ROUNDS),
        )?;
        off.extend(g_off(
            m,
            (2, GCompare::Whole),
            (&runs[2], &last[2]),
            (sb, b, G1B_ROUNDS),
        )?);
        m.select_slot(0)?;
        if (m.pos(), stores_hash(m)?) != idle {
            off.push("idle slot 0's position or stores".to_string());
        }
        let pass = off.is_empty();
        println!(
            "stagger (g1b) parked {mode:?}: {G1B_ROUNDS} passes of slots 1 and 2 from positions \
             {} and {} (row order 1,2 then 2,1 in turn), slot 0 idle: {} {}",
            a.len(),
            b.len(),
            if pass {
                "every row's id and logits, each slot's last logits, stores and position bit for \
                 bit its solo run's; slot 0 as it stood"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(pass)
        );
        Ok(pass)
    }

    /// (g2) after (g1) in graph mode: slot 1 cut back to its prompt call's
    /// checkpoint, then [`G_AFTER_CUT`] passes of both slots, slot 1 fed its
    /// prompt's last id first: slot 1 its solo run from its prompt's step,
    /// slot 0 its solo run's continuation — the cut carried out from slot
    /// 1's own checkpoints by the pass that plans it, slot 1 parked.
    fn g2_cut(
        m: &mut Glm5nextModel,
        (a, b): (&[u32], &[u32]),
        mut runs: [GRun; 2],
        (sa, sb): (&GSolo, &GSolo),
    ) -> Result<bool, GateError> {
        let (&b_last, _) = b.split_last().ok_or("an empty prompt")?;
        let at = u32::try_from(b.len() - 1)?;
        m.select_slot(1)?;
        let kept = m.body("stagger")?.kept(at, m.pos());
        m.rollback(at)?;
        runs[1] = GRun {
            ids: vec![b_last],
            logits: Vec::new(),
        };
        let mut last = [Vec::new(), Vec::new()];
        for r in 0..G_AFTER_CUT {
            let order: &[usize] = if r % 2 == 0 { &[0, 1] } else { &[1, 0] };
            for (&s, row) in order.iter().zip(g_pass(m, &mut runs, order)?) {
                last[s] = row;
            }
        }
        // Slot 1 restarted at its prompt's last id: past it, its run is the
        // solo run's from the prompt's step.
        let restarted = GRun {
            ids: runs[1].ids.get(1..).unwrap_or_default().to_vec(),
            logits: runs[1].logits.clone(),
        };
        let mut off = g_off(
            m,
            (0, GCompare::Whole),
            (&runs[0], &last[0]),
            (sa, a, G1_STEPS[0] + G_AFTER_CUT),
        )?;
        // Slot 1's rows past its position still hold its run before the cut.
        off.extend(g_off(
            m,
            (1, GCompare::Live),
            (&restarted, &last[1]),
            (sb, b, G_AFTER_CUT - 1),
        )?);
        let pass = kept.at == at && off.is_empty();
        // Not a verdict: whether slot 1's rows past its position still hold
        // its run before the cut, as the cut leaves them.
        m.select_slot(1)?;
        let whole = stores_hash(m)? == sb.at(G_AFTER_CUT - 1)?.0.whole;
        println!(
            "stagger (g2) cut: slot 1 cut back to {at} (kept({at}) = {kept}), then {G_AFTER_CUT} \
             passes of both slots (slot 1's whole stores, rows past its position included, \
             equal its solo run's: {whole}): {} {}",
            if off.is_empty() {
                "slot 1 its solo run from its prompt's step and slot 0 its solo run's \
                 continuation, ids, logits, last logits, stores (slot 1's up to its position) \
                 and positions bit for bit"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(pass)
        );
        Ok(pass)
    }

    /// (g3) after (g2): a NaN in the head's gain under a pass of both
    /// slots ends it in the head's fault and poisons both: each slot's step
    /// and the pass refused naming the set; the gain put back, slot 0's reset
    /// leaves slot 1 standing in the set, slot 1's lifts it; both slots
    /// prompted again, the next pass their solo runs' first steps bit for
    /// bit.
    fn g3_fault(
        m: &mut Glm5nextModel,
        (a, b): (&[u32], &[u32]),
        (sa, sb): (&GSolo, &GSolo),
    ) -> Result<bool, GateError> {
        let want = Fault::at(LAYER_HEAD, FaultSite::Logit);
        let set = "slots 0 and 1 are poisoned";
        let old = patch_head_gain(m, f32::NAN.to_le_bytes())?;
        let read = m.step_slots(&[(0, &[1]), (1, &[1])]);
        patch_head_gain(m, old)?;
        let raised = matches!(&read, Err(GpuError::Fault { fault, .. }) if *fault == want)
            && m.poisoned() == Some(want);
        m.select_slot(0)?;
        let named0 = refused_by(&m.step(&[1]), "GpuModel::step", set);
        m.select_slot(1)?;
        let named1 = refused_by(&m.step(&[1]), "GpuModel::step", set);
        let named_pass = refused_by(&m.step_slots(&[(0, &[1]), (1, &[1])]), STEP_SLOTS, set);
        m.select_slot(0)?;
        m.reset()?;
        let stands = refused_by(&m.step(&[1]), "GpuModel::step", "slot 1 is poisoned");
        m.select_slot(1)?;
        m.reset()?;
        let lifted = m.poisoned().is_none();
        let mut runs = g_prompted(m, (a, b))?;
        g_pass(m, &mut runs, &[0, 1])?;
        let again = (0..2).all(|s| {
            let solo = if s == 0 { sa } else { sb };
            solo.ids.get(..2) == Some(&runs[s].ids[..])
                && solo.logits.get(..2) == Some(&runs[s].logits[..])
        });
        let pass = raised && named0 && named1 && named_pass && stands && lifted && again;
        println!(
            "stagger (g3) fault: NaN in {HEAD_GAIN}[0] under a pass of slots 0 and 1: {}, \
             poisoned {}; both slots and the pass refused naming the set ({named0}, {named1}, \
             {named_pass}); after slot 0's reset still refused naming slot 1 {stands}; after \
             both resets lifted {lifted}; prompted again, the next pass both solo runs' first \
             step bit for bit {again} {}",
            match &read {
                Ok(o) => format!("ids {:?}", o.ids),
                Err(e) => format!("\"{e}\""),
            },
            m.poisoned()
                .map_or_else(|| "none now".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        Ok(pass)
    }

    /// (g5)'s cases on the one-lane load, in the order they run: the ones
    /// whose missing refusal would move a slot last.
    #[derive(Clone, Copy, Debug)]
    enum G5 {
        /// Three rows in one pass: the owner's refusal naming the body's
        /// bound, 2.
        Bound,
        /// The taps armed.
        Taps,
        /// Slot 1's step failed past its launch: its stores hold one more
        /// position than it stands at.
        Held,
        /// Slot 0 with two rows.
        Two,
        /// A route trace attached.
        Traced,
    }

    impl G5 {
        fn says(self) -> &'static str {
            match self {
                G5::Bound => "a pass of three rows (naming the bound 2)",
                G5::Taps => "a pass with the taps armed",
                G5::Held => "a pass beside slot 1's step failed past its launch",
                G5::Two => "a pass of slot 0's two rows",
                G5::Traced => "a pass with a route trace attached",
            }
        }
    }

    /// Slot 0's and slot 1's positions, slot 0 selected after.
    fn slot_positions(m: &mut Glm5nextModel) -> Result<[u32; 2], GateError> {
        m.select_slot(1)?;
        let p1 = m.pos();
        m.select_slot(0)?;
        Ok([m.pos(), p1])
    }

    /// (g5) on the one-lane load after (g3), its last clause: each case
    /// ([`G5`]) from both slots prompted afresh, refused by name with no
    /// slot's position moved, a line a case as it ends.
    fn g5_refusals(
        m: &mut Glm5nextModel,
        inputs: &PlanInputs,
        (a, b): (&[u32], &[u32]),
    ) -> Result<bool, GateError> {
        let mut pass = true;
        for case in [G5::Bound, G5::Taps, G5::Held, G5::Two, G5::Traced] {
            let runs = g_prompted(m, (a, b))?;
            let (named, stood) = g5_case(m, inputs, case, [runs[0].ids[0], runs[1].ids[0]])?;
            println!(
                "stagger (g5) {}: refused by name {named}, both positions unmoved {stood} {}",
                case.says(),
                verdict(named && stood)
            );
            pass &= named && stood;
        }
        Ok(pass)
    }

    /// One (g5) case on both slots prompted, `t` their prompts' argmaxes:
    /// whether the pass was refused by its name, and whether both slots'
    /// positions stood (for [`G5::Held`], as the failed step left them).
    fn g5_case(
        m: &mut Glm5nextModel,
        inputs: &PlanInputs,
        case: G5,
        [t0, t1]: [u32; 2],
    ) -> Result<(bool, bool), GateError> {
        let both: [(usize, &[u32]); 2] = [(0, &[t0]), (1, &[t1])];
        let mut before = slot_positions(m)?;
        let named = match case {
            G5::Bound => refused_by(
                &m.step_slots(&[(0, &[t0, t0]), (1, &[t1])]),
                STEP_SLOTS,
                "3 rows in one pass; this body's pass of several slots holds at most 2",
            ),
            G5::Taps => {
                set_taps(m, true)?;
                let r = m.step_slots(&both);
                set_taps(m, false)?;
                m.set_mode(StepMode::Graph);
                refused_by(&r, GLM_BODY, "a pass of several slots with the taps armed")
            }
            G5::Held => {
                m.select_slot(1)?;
                m.body_parts("stagger")?.2.plant(Plant::AfterLaunch);
                if m.step(&[t1]).is_ok() {
                    return Err("slot 1's planted step ran".into());
                }
                before = slot_positions(m)?;
                let p1 = before[1];
                refused_by(
                    &m.step_slots(&both),
                    GLM_BODY,
                    &format!(
                        "slot 1: position {p1}: the step there failed after its chain was launched"
                    ),
                )
            }
            G5::Two => refused_by(
                &m.step_slots(&[(0, &[t0, t0])]),
                GLM_BODY,
                "slot 0's 2 rows in a pass of several slots",
            ),
            G5::Traced => {
                let dir = std::env::temp_dir()
                    .join(format!("gate_glm5next_e2e-g5-{}", std::process::id()));
                let run = m.body("stagger")?.host_run();
                let trace = RouteTrace::create(
                    &dir,
                    TraceHeader {
                        model: PathBuf::from(MODEL),
                        arch: "glm5next".to_owned(),
                        build: "gate_glm5next_e2e".to_owned(),
                        n_expert: inputs.hp.n_expert,
                        n_used: inputs.hp.n_used,
                        first_layer: run.start,
                        n_layer: run.len(),
                        extra: Vec::new(),
                    },
                )?;
                m.body_parts("stagger")?
                    .2
                    .hybrid_mut()
                    .attach_route_trace(trace)?;
                let r = m.step_slots(&both);
                drop(m.body_parts("stagger")?.2.hybrid_mut().take_route_trace());
                std::fs::remove_dir_all(&dir)?;
                refused_by(
                    &r,
                    GLM_BODY,
                    "a pass of several slots with a route trace attached",
                )
            }
        };
        Ok((named, slot_positions(m)? == before))
    }

    /// (g5) on the NextN load, between (sc) and H4: a pass of both slots
    /// refused by name, every slot's position as it stood.
    fn g5_nextn(m: &mut Glm5nextModel) -> Result<bool, GateError> {
        let at = |m: &mut Glm5nextModel| -> Result<[u32; 2], GateError> {
            m.select_slot(1)?;
            let p1 = m.pos();
            m.select_slot(0)?;
            Ok([m.pos(), p1])
        };
        let before = at(m)?;
        let r = m.step_slots(&[(0, &[1]), (1, &[1])]);
        let named = refused_by(&r, GLM_BODY, "a pass of several slots on a NextN load");
        let stood = at(m)? == before;
        println!(
            "stagger (g5) NextN: a pass of slots 0 and 1 on the NextN load: {}; positions \
             unmoved {stood} {}",
            match &r {
                Ok(o) => format!("ids {:?}", o.ids),
                Err(e) => format!("\"{e}\""),
            },
            verdict(named && stood)
        );
        Ok(named && stood)
    }

    /// (g4) on a load of two KDA lanes: the captured pass of both slots
    /// holds the captured verify's nodes; the verify replayed after the
    /// pass's capture re-recorded the pair's go order, and the pass replayed
    /// after the verify's capture re-recorded it again, each bit for bit
    /// its solo runs (the verify on slot 0 its solo run's next two ids and
    /// logits, the pass each slot's solo step) — the two walks raise their
    /// goes in one order. Between them, (g5)'s refusal of a pass while slot
    /// 0's verify waits for its commit. The pass runs after the verify kept
    /// both rows, so slot 0's committed lane is the second, slot 1's the
    /// first.
    fn g4_walk(
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
        (a, b): (&[u32], &[u32]),
    ) -> Result<bool, GateError> {
        let mut m = g_open(levers, inputs, KdaLanes::Two, SLOTS)?;
        let sa = g_solo(&mut m, a, 3, &[])?;
        let sb = g_solo(&mut m, b, 1, &[])?;
        let runs = g_prompted(&mut m, (a, b))?;
        m.select_slot(0)?;
        let pos = m.pos();
        let verify_nodes = m.capture_rows::<2>()?;
        let pass_nodes = m.capture_slots(&[(0, 1), (1, 1)])?;
        m.select_slot(0)?;
        let verify = m.step_rows([runs[0].ids[0], sa.ids[1]]);
        if let Err(e) = &verify {
            println!(
                "stagger (g4) walk: two lanes; the captured pass of slots 0 and 1 holds \
                 {pass_nodes} nodes, the verify {verify_nodes}; the verify replayed after the \
                 pass's capture ended in \"{e}\" FAIL"
            );
            return Ok(false);
        }
        let verify_logits: Vec<u64> = m.rows_logits::<2>()?.iter().map(|l| fnv_row(l)).collect();
        let waits = refused_by(
            &m.step_slots(&[(0, &[sa.ids[2]]), (1, &[runs[1].ids[0]])]),
            GLM_BODY,
            "a pass of several slots: a call at position",
        );
        m.rollback(pos + 2)?;
        let recaptured = m.capture_rows::<2>()?;
        let out = m.step_slots(&[(0, &[sa.ids[2]]), (1, &[runs[1].ids[0]])])?;
        let pass_logits: Vec<u64> = m.slots_logits()?.iter().map(|l| fnv_row(l)).collect();
        let nodes = verify_nodes == pass_nodes && recaptured == verify_nodes;
        let verify_ok = matches!(&verify, Ok(ids) if ids[..] == sa.ids[1..3])
            && verify_logits[..] == sa.logits[1..3];
        let pass_ok =
            out.ids == [sa.ids[3], sb.ids[1]] && pass_logits == [sa.logits[3], sb.logits[1]];
        let ok = nodes && verify_ok && waits && pass_ok;
        println!(
            "stagger (g4) walk: two lanes; the captured pass of slots 0 and 1 holds \
             {pass_nodes} nodes, the verify {verify_nodes} (again {recaptured}); the verify \
             replayed after the pass's capture: {} (solo {:?}), logits {verify_ok}; while it \
             waits, a pass refused by name {waits}; the pass replayed after the verify's \
             capture: ids {:?} (solo {:?}), logits {pass_ok} {}",
            match &verify {
                Ok(ids) => format!("ids {ids:?}"),
                Err(e) => format!("\"{e}\""),
            },
            &sa.ids[1..3],
            out.ids,
            [sa.ids[3], sb.ids[1]],
            verdict(ok)
        );
        Ok(ok)
    }

    /// Which clauses a run takes.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Only {
        /// Every clause, on both loads.
        All,
        /// The load at [`CTX`], (pb-long) included.
        Main,
        /// The prompt batch's load at [`CTX_PP`].
        Pp,
        /// (pb-long) alone, on a load at [`CTX`].
        PpLong,
        /// (s) and (v) alone, on a load at [`CTX`].
        Verify,
        /// (k) alone, on a load at [`CTX`].
        Keep,
        /// (slots) alone, on their loads at [`SLOT_CTX`].
        Slots,
        /// (g1)-(g5) alone, on their two loads at [`SLOT_CTX`] (the NextN
        /// load's (g5) runs with (slots)).
        Stagger,
    }

    /// Which of (t)'s step sets the load at [`CTX`] runs.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum StepSets {
        /// All four.
        All,
        /// [`STEP4`] and [`STEP4_EVERY_NODE`].
        Short,
        /// [`D1K`] and [`D3K_DSA`].
        Long,
    }

    impl StepSets {
        fn name(self) -> &'static str {
            match self {
                StepSets::All => "all",
                StepSets::Short => "short",
                StepSets::Long => "long",
            }
        }

        /// Whether the set is taken: the 4-token sets are the short ones.
        fn takes(self, name: &str) -> bool {
            let short = name == STEP4 || name == STEP4_EVERY_NODE;
            match self {
                StepSets::All => true,
                StepSets::Short => short,
                StepSets::Long => !short,
            }
        }
    }

    /// `--step-sets short|long|all`, `all` when absent; refused beside an
    /// `--only` that runs no load at [`CTX`]'s clauses.
    fn step_sets(only: Only) -> Result<StepSets, GateError> {
        let args: Vec<String> = std::env::args().collect();
        let sets = match args.iter().position(|a| a == "--step-sets") {
            None => return Ok(StepSets::All),
            Some(i) => match args.get(i + 1).map(String::as_str) {
                Some("all") => StepSets::All,
                Some("short") => StepSets::Short,
                Some("long") => StepSets::Long,
                other => {
                    return Err(format!("--step-sets is short, long or all, not {other:?}").into());
                }
            },
        };
        if !matches!(only, Only::All | Only::Main) {
            return Err(
                "--step-sets names the step sets of the load at CTX: it goes with \
                        --only main or no --only"
                    .into(),
            );
        }
        Ok(sets)
    }

    /// `--only main`, `--only pp`, `--only pplong`, `--only verify`, `--only
    /// keep`, `--only slots`, `--only stagger`, or every clause.
    fn only() -> Result<Only, GateError> {
        let args: Vec<String> = std::env::args().collect();
        match args.iter().position(|a| a == "--only") {
            None => Ok(Only::All),
            Some(i) => match args.get(i + 1).map(String::as_str) {
                Some("main") => Ok(Only::Main),
                Some("pp") => Ok(Only::Pp),
                Some("pplong") => Ok(Only::PpLong),
                Some("verify") => Ok(Only::Verify),
                Some("keep") => Ok(Only::Keep),
                Some("slots") => Ok(Only::Slots),
                Some("stagger") => Ok(Only::Stagger),
                other => Err(format!(
                    "--only is main, pp, pplong, verify, keep, slots or stagger, not {other:?}"
                )
                .into()),
            },
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let only = only()?;
        let sets = step_sets(only)?;
        let mut ok = true;
        if matches!(only, Only::All | Only::Main) {
            ok &= main_clauses(&levers, sets)?;
        }
        if only == Only::Verify {
            ok &= verify_only(&levers)?;
        }
        if only == Only::Keep {
            let (mut s, _) = open(&levers, CTX, PrefillMode::Steps, KdaLanes::Two)?;
            ok &= keep_groups(&mut s)?;
        }
        if only == Only::PpLong {
            let (mut s, _) = open(&levers, CTX, PrefillMode::Steps, KdaLanes::One)?;
            ok &= prompt_long(s.model_mut())?;
        }
        if matches!(only, Only::All | Only::Pp) {
            ok &= prompt_batch(&levers)?;
        }
        if matches!(only, Only::All | Only::Slots) {
            ok &= slots(&levers)?;
        }
        if only == Only::Stagger {
            let prefill = slot_prefill()?;
            let (a, b, _) = slot_prompts(&prefill);
            let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
            let inputs = PlanInputs::read(&file)?;
            drop(file);
            ok &= stagger(&levers, &inputs, (a, b))?;
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// Every clause on the steps feed, on the load at [`CTX`] of two KDA
    /// lanes: (v) verifies on it.
    fn main_clauses(levers: &bloomery_levers::Levers, sets: StepSets) -> Result<bool, GateError> {
        let (mut s, opened) = open(levers, CTX, PrefillMode::Steps, KdaLanes::Two)?;
        let mut ok = card::clauses(&mut s, &opened.n_l, opened.experts, opened.budgeted)?;
        let m = s.model_mut();
        ok &= structure(m, &opened)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        let graph = run_steps(m, &toks, StepMode::Graph)?;
        let eager = run_steps(m, &toks, StepMode::Eager)?;
        ok &= one_chain(&graph, &eager);
        ok &= taps_after_capture(m, &toks, &eager)?;
        ok &= position_owner(m, &toks, &graph)?;
        ok &= head_fault(m, toks[0], graph.tokens[0])?;
        let routes = ik_routes(&man)?;
        let layered = layered(m, &toks)?;
        let layered_ok = layered_is_chain(&layered, &eager);
        ok &= layered_ok;
        let (free_ok, firsts) = free(&man, &eager, (&layered, layered_ok), &routes)?;
        ok &= free_ok;
        drop(layered);
        let last = *firsts.last().ok_or("no positions")?;
        let (forced_ok, _) = forced(m, &man, &routes)?;
        ok &= forced_ok;
        let mut ties = 0usize;
        let band = Some((&toks[..], last));
        let mut ran = Vec::new();
        for (set, band) in [
            ((STEP4, &IK), band),
            ((STEP4_EVERY_NODE, &IK), band),
            ((D1K, &IK), None),
            ((D3K_DSA, &IK_DSA), None),
        ] {
            if sets.takes(set.0) {
                ok &= step_set(m, set, band, &mut ties)?;
                ran.push(set.0);
            }
        }
        println!(
            "step sets ({}): {} ran, {ties} named tie(s)",
            sets.name(),
            ran.join(" ")
        );
        ok &= keep_groups(&mut s)?;
        ok &= prompt_long(s.model_mut())?;
        ok &= verify_clause(s.model_mut(), &toks, &graph, opened.nodes);
        Ok(ok)
    }

    /// (s) and (v) alone on the load at [`CTX`] (`--only verify`), against
    /// the plain graph run of the batch set.
    fn verify_only(levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let (mut s, opened) = open(levers, CTX, PrefillMode::Steps, KdaLanes::Two)?;
        let m = s.model_mut();
        let mut ok = structure(m, &opened)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        let graph = run_steps(m, &toks, StepMode::Graph)?;
        ok &= verify_clause(m, &toks, &graph, opened.nodes);
        Ok(ok)
    }

    /// [`verify_rows`], an error it ends in printed as the clause's FAIL
    /// rather than ending the gate; the gate's last clause, so nothing runs
    /// on the model it leaves.
    fn verify_clause(m: &mut Glm5nextModel, toks: &[u32], graph: &Run, nodes: usize) -> bool {
        match verify_rows(m, toks, graph, nodes) {
            Ok(ok) => ok,
            Err(e) => {
                println!("verify: ended in error \"{e}\" {}", verdict(false));
                false
            }
        }
    }

    // ------------------------------------------- (pb) (pr) (pf) prompt batch

    /// `n` ids of an lcg over the vocabulary: every id a row of the
    /// embedding, the routing spread over the experts.
    fn lcg_ids(n: usize) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % N_VOCAB as u64) as u32
            })
            .collect()
    }

    /// FNV-1a over a row's bits.
    fn fnv_row(row: &[f32]) -> u64 {
        row.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
            v.to_bits().to_le_bytes().iter().fold(h, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        })
    }

    /// What the plain steps leave after `p` positions.
    struct After {
        p: usize,
        stores: Vec<StoreDigest>,
        logits: u64,
        argmax: u32,
    }

    /// The first store of `got` whose digest differs from `want`'s, named.
    fn first_store_diff(got: &[StoreDigest], want: &[StoreDigest]) -> Option<String> {
        if got.len() != want.len() {
            return Some(format!("{} stores against {}", got.len(), want.len()));
        }
        got.iter()
            .zip(want)
            .find(|(g, w)| g != w)
            .map(|(g, _)| format!("layer {} {}", g.layer, g.what))
    }

    /// The checkpoints a call of `p` positions from 0 leaves: the multiples of
    /// 512 inside it and its end.
    fn marks(p: usize) -> Vec<u32> {
        let mut v: Vec<u32> = (1..)
            .map(|k| 512 * k)
            .take_while(|&k| k < p)
            .map(|k| k as u32)
            .collect();
        v.push(p as u32);
        v
    }

    /// What a feed leaves: every store and the last logits and argmax after
    /// the prompt, each greedy step's token and logits, every store after
    /// them.
    #[derive(PartialEq, Eq)]
    struct Long {
        prompt: Vec<StoreDigest>,
        logits: u64,
        argmax: u32,
        greedy: Vec<(u32, u64)>,
        after: Vec<StoreDigest>,
    }

    /// What a feed that ended at `argmax` leaves: every store, the last
    /// logits, then [`GREEDY`] greedy steps from `argmax`, each its token and
    /// its logits' digest, then every store again.
    fn after_prompt(m: &mut Glm5nextModel, argmax: u32) -> Result<Long, GateError> {
        let prompt = store_digests(m)?;
        let logits = fnv_row(&m.logits()?);
        let mut tok = argmax;
        let mut greedy = Vec::with_capacity(GREEDY);
        for _ in 0..GREEDY {
            tok = m.step(&[tok])?;
            greedy.push((tok, fnv_row(&m.logits()?)));
        }
        Ok(Long {
            prompt,
            logits,
            argmax,
            greedy,
            after: store_digests(m)?,
        })
    }

    /// (pb-long) on `m`, a load at [`CTX`]; the model left reset. The taps
    /// disarmed (a batch refuses them) and graph steps, whatever the clauses
    /// before it left. Two references, each the same tokens in batches of the
    /// same side of `GEMM_FROM` (a batch's bits depend on nothing else): one
    /// call of [`LONG`] ids against two cut at [`LONG_CUT`]; and after a call
    /// of [`LONG_TAIL`] ids, the tail to [`LONG`] — past the positions the
    /// latent layers attend whole, through the selector — by steps against
    /// calls of a chunk, from the checkpoint at [`LONG_TAIL`].
    fn prompt_long(m: &mut Glm5nextModel) -> Result<bool, GateError> {
        let ids = lcg_ids(LONG);
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let t = Instant::now();
        let one = prefill(m, &ids)?;
        let one_s = t.elapsed().as_secs_f64();
        let want = after_prompt(m, one)?;
        m.reset()?;
        let t = Instant::now();
        let fed = prefill(m, &ids[..LONG_CUT]).and_then(|_| prefill(m, &ids[LONG_CUT..]));
        let batch_s = t.elapsed().as_secs_f64();
        let argmax = match fed {
            Ok(tok) => tok,
            Err(e) => {
                println!(
                    "prompt batch long: {LONG} ids as {LONG_CUT} + {}: error \"{e}\" {}",
                    LONG - LONG_CUT,
                    verdict(false)
                );
                m.reset()?;
                return Ok(false);
            }
        };
        let pos = m.pos() as usize;
        let points = m.body("prompt batch long")?.checkpoints().positions();
        let got = after_prompt(m, argmax)?;
        m.reset()?;
        let pass = got == want && pos == LONG;
        let tokens = |l: &Long| l.greedy.iter().map(|g| g.0).collect::<Vec<_>>();
        println!(
            "prompt batch long: {LONG} ids as {LONG_CUT} + {} in {batch_s:.2} s against one call \
             in {one_s:.2} s (runtime values), pos {pos}, checkpoints {points:?}: stores after the \
             prompt {} ({} stores), logits {} argmax {} (one call {}); {GREEDY} greedy steps: \
             tokens {:?} (one call {:?}), logits {}; stores after them {} {}",
            LONG - LONG_CUT,
            first_store_diff(&got.prompt, &want.prompt)
                .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
            want.prompt.len(),
            if got.logits == want.logits {
                "bit for bit"
            } else {
                "differ"
            },
            got.argmax,
            want.argmax,
            tokens(&got),
            tokens(&want),
            if got.greedy == want.greedy {
                "bit for bit"
            } else {
                "differ"
            },
            first_store_diff(&got.after, &want.after)
                .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
            verdict(pass)
        );
        Ok(pass & long_tail(m, &ids)?)
    }

    /// (pb-long)'s tail: a call of [`LONG_TAIL`] ids, then the rest to
    /// [`LONG`] by graph steps; back to the call's checkpoint at
    /// [`LONG_TAIL`] and the rest in calls of at most a chunk: every store,
    /// the logits and the argmax bit for bit the steps'. The model left
    /// reset.
    fn long_tail(m: &mut Glm5nextModel, ids: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        prefill(m, &ids[..LONG_TAIL])?;
        let mut tok = 0;
        for &id in &ids[LONG_TAIL..] {
            tok = m.step(&[id])?;
        }
        let (stores, logits) = (store_digests(m)?, fnv_row(&m.logits()?));
        m.rollback(LONG_TAIL as u32)?;
        let mut at = LONG_TAIL;
        let mut chunked = 0;
        while at < LONG {
            let to = (at + CHUNK).min(LONG);
            chunked = prefill(m, &ids[at..to])?;
            at = to;
        }
        let diff = first_store_diff(&store_digests(m)?, &stores);
        let same_logits = fnv_row(&m.logits()?) == logits;
        m.reset()?;
        let pass = diff.is_none() && same_logits && chunked == tok;
        println!(
            "prompt batch long tail: after a call of {LONG_TAIL}, positions {LONG_TAIL}..{LONG} in \
             calls of at most {CHUNK} against steps: stores {} logits {} argmax {chunked} (steps \
             {tok}) {}",
            diff.as_deref()
                .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
            if same_logits { "bit for bit" } else { "differ" },
            verdict(pass)
        );
        Ok(pass)
    }

    /// The GEMM batch's bands per layer ([`gemm_bands`]): each layer's
    /// stores', and its stream's after the layer's mixer and block — what
    /// the next layer, the layer's router and, at the last layer, the head
    /// read.
    struct Bands {
        stores: Vec<f64>,
        stream: Vec<f64>,
    }

    /// PIN(2026-10-02): the GEMM batch's distance from the steps per layer,
    /// as the error model gives it: q = `rounding::q8_32_rel` = 1.2858e-2 per
    /// quantized input, passed to the projection's output; outputs of one
    /// input through different weights' rows are independent, so their
    /// variances add. Each site is counted at the first store or stream that
    /// reads it. A KDA layer's stores (the conv ring of q, k and v, the delta
    /// rule's state) read q, k and v (3) and, through σ, slope <= 1/4, β
    /// (1/16): 3.0625; its stream reads beside them the gated rows through
    /// `out` (1) and, through σ, z twice (2/16): 4.1875 in all. Its decay pair
    /// stays on the one-column gemv (0). A latent layer's stores (its latent
    /// and index rows) read its stack's one input (1); its stream reads beside
    /// it the query's low rank through `q_b` (1) and the heads' outputs
    /// through `out` (1): 3 in all — the query's error reaches the scores
    /// `q·k` as the stack's error in the latent keys does, through the same
    /// softmax, so it is counted as that one is; the absorbed pair `k_b`,
    /// `v_b` stays on the gemv (0). A dense block reaches only the stream:
    /// through the up (1) and the gate, whose relative error reaches
    /// `silu(g)` times its log-slope `1 + g·(1 − σ(g))`, at most 1.28 (at
    /// g = 1.28; the clamp only lowers it): 1.28² = 1.64, and its SwiGLU rows
    /// through the down (1): 3.64. A routed layer's shared expert stays on
    /// the gemv (0). Layers add independently, where both feeds take the same
    /// experts and pools: layer l's stores within q·√(Σ_{j<l} c_j + s_l) (s_l
    /// its stores' part), its stream within q·√(Σ_{j<=l} c_j) — the router
    /// reads the stream after the layer's mixer, a routed block adding 0; the
    /// head reads all 45: q·√(34·4.1875 + 3·3.64 + 11·3) = 0.1755. From
    /// `kinds` (each layer's mixer and feed-forward block).
    fn gemm_bands(kinds: &[Layer]) -> Bands {
        let q = q8_32_rel();
        let mut c = 0.0f64;
        let (mut stores, mut stream) = (Vec::new(), Vec::new());
        for k in kinds {
            let (store, mixer) = if k.mixer == MixerKind::Latent {
                (1.0, 3.0)
            } else {
                (3.0625, 4.1875)
            };
            stores.push(q * (c + store).sqrt());
            c += mixer;
            if k.ffn == FfnKind::Dense {
                c += 3.64;
            }
            stream.push(q * c.sqrt());
        }
        Bands { stores, stream }
    }

    /// PIN(2026-10-02): a flip between the GEMM batch's routing and the
    /// reference's with no earlier flip on its path is excused while each
    /// exchanged pair's gap in the reference's ranked values lies within our
    /// two values' distance there ([`Flip::allowed`]), and that distance
    /// within six deviations of the router's error at the layer: a logit `z`
    /// is a dot of the layer's normed input, which carries `band`
    /// ([`gemm_bands`]' stream at the layer) of relative error, so `z` moves by about
    /// that times the logits' RMS; the ranked value σ(z) + bias moves by at
    /// most ¼ of it; two values, three deviations each: 6 · ¼ · band ·
    /// RMS(z). The logits are the scores' own, `ln(p / (1 − p))` of the
    /// reference's. Such a flip's reference margin — its eighth pick's ranked
    /// value less the best it left, the least gap any exchange crosses —
    /// lies within the cap too (a margin is at most the pair's distance). Past an earlier flip the layer's input also carries that
    /// flip's other experts' output, which no band covers: the distance is
    /// counted and printed against the cap, not held, and no margin rule
    /// applies there (the clean run at 2051 positions has 322 of 47148 such
    /// flips past the cap, the worst at 3.20 of it). That case is held by the
    /// forced arm ([`gemm_forced`]): with the reference's routes planted at
    /// every layer, every layer's stores and the logits lie within their
    /// bands ([`gemm_bands`]).
    fn route_cap(band: f64, probs: &[f32]) -> f64 {
        1.5 * band * logit_moments(probs).0
    }

    /// The widest gap in the reference's ranked values a flip may exchange
    /// where no earlier flip lies on its path: [`route_cap`]'s derivation on
    /// the logits' deviation about their mean (a common offset moves no
    /// rank). A flip at a wider gap is a pick the error model does not
    /// explain.
    fn route_margin_cap(band: f64, probs: &[f32]) -> f64 {
        1.5 * band * logit_moments(probs).1
    }

    /// The RMS and the deviation about the mean of the logits a row of
    /// sigmoid scores was made from; NaN when a score is 0 or 1 (its logit
    /// not finite), so no cap made from it passes.
    fn logit_moments(probs: &[f32]) -> (f64, f64) {
        let z: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let p = f64::from(p);
                (p / (1.0 - p)).ln()
            })
            .collect();
        if z.iter().any(|v| !v.is_finite()) {
            return (f64::NAN, f64::NAN);
        }
        let n = z.len().max(1) as f64;
        let rms = (z.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        let mean = z.iter().sum::<f64>() / n;
        let dev = (z.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        (rms, dev)
    }

    /// The RMS of a row.
    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    }

    /// What a feed leaves after `p` positions with the values a band reads:
    /// the record, the last logits, and from [`GEMM_FROM`] positions every
    /// store's live values ([`store_rows`]).
    struct Held {
        after: After,
        logits: Vec<f32>,
        stores: Vec<StoreRows>,
    }

    /// Each layer's live store values joined, in layer order.
    fn by_layer(rows: &[StoreRows]) -> Vec<Vec<f32>> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); N_LAYER];
        for r in rows {
            if let Some(v) = out.get_mut(r.layer) {
                v.extend_from_slice(&r.values);
            }
        }
        out
    }

    /// Each routed layer's selection bias, the ranked value of expert `e`
    /// being its score plus `bias[e]`; `None` for a dense layer. A routed
    /// layer without its F32 bias resident is refused by name.
    fn biases(m: &mut Glm5nextModel) -> Result<Vec<Option<Vec<f32>>>, GateError> {
        let kinds = m.body("biases")?.kinds();
        let (gpu, w, _) = m.body_parts("biases")?;
        kinds
            .iter()
            .enumerate()
            .map(|(l, k)| {
                if k.ffn != FfnKind::Moe {
                    return Ok(None);
                }
                let name = names::exp_probs_b(l);
                let Some(DevWeight::F32 { w: b, .. }) = w.get(&name) else {
                    return Err(format!("{name} is not resident as F32").into());
                };
                Ok(Some(b.buf().to_host_vec(gpu.stream())?))
            })
            .collect()
    }

    /// The (pb) reference ([`chunk_calls`]): whether it held the steps' bits,
    /// its record at each count, and every routed layer's route taps.
    struct Reference {
        ok: bool,
        held: Vec<Held>,
        routes: Vec<Option<RouteTapRows>>,
    }

    /// The (pb) reference: `ids` fed from a reset in calls of at most
    /// [`CHUNK`] positions, each ending at the next of `after`'s counts — a
    /// batch of at most a chunk runs the step's gemvs, so it writes the
    /// steps' bits — with the route taps armed: at each count its stores,
    /// logits and argmax held to the steps' (`after`) bit for bit, and its
    /// record kept; then every routed layer's taps over the positions fed.
    fn chunk_calls(
        m: &mut Glm5nextModel,
        ids: &[u32],
        after: &[After],
    ) -> Result<Reference, GateError> {
        m.reset()?;
        let t = Instant::now();
        let (mut at, mut argmax, mut ok) = (0usize, 0u32, true);
        let mut held = Vec::with_capacity(after.len());
        let mut calls = 0usize;
        for a in after {
            while at < a.p {
                let to = (at + CHUNK).min(a.p);
                argmax = prefill(m, &ids[at..to])?;
                at = to;
                calls += 1;
            }
            let (digests, logits) = (store_digests(m)?, m.logits()?);
            let diff = first_store_diff(&digests, &a.stores);
            let pass = diff.is_none() && fnv_row(&logits) == a.logits && argmax == a.argmax;
            println!(
                "prompt batch chunks P={}: calls of at most {CHUNK} positions against the steps: \
                 stores {} logits {} argmax {argmax} (steps {}) {}",
                a.p,
                diff.as_deref()
                    .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
                if fnv_row(&logits) == a.logits {
                    "bit for bit"
                } else {
                    "differ"
                },
                a.argmax,
                verdict(pass)
            );
            ok &= pass;
            held.push(Held {
                after: After {
                    p: a.p,
                    stores: digests,
                    logits: fnv_row(&logits),
                    argmax,
                },
                stores: if a.p >= GEMM_FROM {
                    store_rows(m, a.p)?
                } else {
                    Vec::new()
                },
                logits,
            });
        }
        let routes = prompt_route_taps(m, at)?;
        println!(
            "prompt batch chunks: {at} positions in {calls} calls in {:.1} s (runtime value), \
             every routed layer's route taps read",
            t.elapsed().as_secs_f64()
        );
        Ok(Reference { ok, held, routes })
    }

    /// The flips of one run's routing (`ours`) against another's (`theirs`,
    /// the reference) over positions `0 .. p`, and their tally: the flips
    /// first on their path, those past [`route_cap`] and those at a wide
    /// margin (held), and those past an earlier flip, the ones of them past
    /// the cap and the largest distance over it (printed).
    struct Tally {
        flips: usize,
        first: usize,
        refused: usize,
        wide: usize,
        past_over: usize,
        past_worst: f64,
        /// Flips whose reference margin (its eighth pick's ranked value less
        /// the best it left) lies past [`route_cap`], on any path (printed).
        margin_over: usize,
        /// Of them, the flips first on their path (held).
        margin_first: usize,
        /// The flips' reference margins over their caps, sorted.
        margins: Vec<f64>,
        /// The lowest layer a flip lies at: every store past it lies on a
        /// flip's path.
        low: Option<usize>,
    }

    impl Tally {
        /// Every flip first on its path allowed by the pair rule, at a gap
        /// within the margin bound and at a reference margin within the cap.
        fn ok(&self) -> bool {
            self.refused == 0 && self.wide == 0 && self.margin_first == 0
        }

        /// The flips' margin over cap at the quantile `q` (0 to 1).
        fn margin_at(&self, q: f64) -> f64 {
            if self.margins.is_empty() {
                return 0.0;
            }
            let i = ((self.margins.len() - 1) as f64 * q).round() as usize;
            self.margins[i]
        }
    }

    /// The flips of `ours` against `theirs` (each routed layer's taps, `None`
    /// a dense one) over positions `0 .. p`: each with no earlier flip on its
    /// path (a flip at a lower layer and at the same or an earlier position)
    /// excused by the pair rule under [`route_cap`] and only at a gap within
    /// [`route_margin_cap`], both at the layer's band; each past one counted
    /// against the cap, printed. The first few, every refused one and the
    /// first few past ones over the cap printed with `arm`.
    fn judge_routes(
        arm: &str,
        p: usize,
        (ours, theirs): (&[Option<RouteTapRows>], &[Option<RouteTapRows>]),
        bias: &[Option<Vec<f32>>],
        bands: &[f64],
    ) -> Tally {
        let mut flips: Vec<(Flip, f64, f64)> = Vec::new();
        for (l, ((o, r), b)) in ours.iter().zip(theirs).zip(bias).enumerate() {
            let (Some(o), Some(r), Some(b)) = (o, r, b) else {
                continue;
            };
            for t in 0..p {
                let (oi, ri) = (
                    &o.ids[t * N_USED..(t + 1) * N_USED],
                    &r.ids[t * N_USED..(t + 1) * N_USED],
                );
                if oi.iter().all(|e| ri.contains(e)) {
                    continue;
                }
                let ranked = |x: &[f32]| -> Vec<f32> {
                    x[t * N_EXPERT..(t + 1) * N_EXPERT]
                        .iter()
                        .zip(b)
                        .map(|(&s, &c)| s + c)
                        .collect()
                };
                let (ov, rv) = (ranked(&o.probs), ranked(&r.probs));
                let rids: Vec<i32> = ri.iter().map(|&e| e as i32).collect();
                let margin = flip::margin(&rv, &rids);
                if let Some(f) = Flip::between((l, t), (oi, &ov), (&rids, &rv), margin) {
                    let probs = &r.probs[t * N_EXPERT..(t + 1) * N_EXPERT];
                    let band = bands[l];
                    flips.push((f, route_cap(band, probs), route_margin_cap(band, probs)));
                }
            }
        }
        // The earliest position a flip lies at below each layer.
        let mut below = vec![usize::MAX; N_LAYER + 1];
        for (f, _, _) in &flips {
            for e in below.iter_mut().skip(f.layer + 1) {
                *e = (*e).min(f.token);
            }
        }
        let (mut refused, mut first, mut wide, mut shown) = (0usize, 0usize, 0usize, 0usize);
        let (mut past_over, mut past_worst, mut past_shown) = (0usize, 0.0f64, 0usize);
        let (mut margin_over, mut margin_first) = (0usize, 0usize);
        let mut margins = Vec::with_capacity(flips.len());
        for (f, cap, bound) in &flips {
            let wide_margin = f.margin > *cap || f.margin.is_nan() || cap.is_nan();
            margin_over += usize::from(wide_margin);
            margins.push(f.margin / cap);
            let prior = below[f.layer] <= f.token;
            let widest = f
                .pairs
                .iter()
                .map(|p| p.2)
                .fold(f64::NEG_INFINITY, f64::max);
            let over = !f.allowed(*cap);
            let at_wide = !prior && (widest > *bound || widest.is_nan());
            let bad = !prior && (over || at_wide || wide_margin);
            if prior {
                past_over += usize::from(over);
                let err = f.pairs.iter().map(|p| p.3).fold(0.0f64, f64::max);
                past_worst = past_worst.max(err / cap);
            } else {
                first += 1;
                margin_first += usize::from(wide_margin);
                refused += usize::from(over);
                wide += usize::from(at_wide);
            }
            let show_past = prior && over && past_shown < 4;
            if shown < 8 || (bad && shown < 40) || show_past {
                shown += 1;
                past_shown += usize::from(show_past);
                println!(
                    "{}; {}",
                    f.line(arm, *cap),
                    if prior {
                        "past an earlier flip on its path (printed: no band covers its input; \
                         margin rule not applied)"
                            .to_owned()
                    } else if wide_margin {
                        format!(
                            "first on its path: FAIL: the reference's margin {:.3e} past the cap",
                            f.margin
                        )
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
        }
        Tally {
            flips: flips.len(),
            first,
            refused,
            wide,
            past_over,
            past_worst,
            margin_over,
            margin_first,
            margins: {
                margins.sort_by(f64::total_cmp);
                margins
            },
            low: flips.iter().map(|(f, _, _)| f.layer).min(),
        }
    }

    /// (pb) at groups of one, a call of `want.after.p` ids from a reset from
    /// [`GEMM_FROM`] on, against the chunk calls' `want` (the steps' bits):
    /// every routed layer's picks judged ([`judge_routes`]); each layer's live
    /// stores within its band ([`gemm_bands`]) at every layer no flip's path
    /// reaches (no flip at a lower layer), the rest printed; the last logits
    /// within the head's band when no flip lies anywhere, printed otherwise;
    /// the argmax the reference's, or one whose logit in the reference's row
    /// lies within six head bands of the row's RMS below its top; the position and the checkpoints the
    /// call's. Returns the call's record.
    fn gemm_against(
        m: &mut Glm5nextModel,
        ids: &[u32],
        want: &Held,
        routes: &[Option<RouteTapRows>],
        (bias, bands): (&[Option<Vec<f32>>], &Bands),
    ) -> Result<(bool, Option<After>), GateError> {
        let p = want.after.p;
        m.reset()?;
        let t = Instant::now();
        let tok = match prefill(m, &ids[..p]) {
            Ok(tok) => tok,
            Err(e) => {
                println!("prompt batch G=1 P={p}: error \"{e}\" {}", verdict(false));
                return Ok((false, None));
            }
        };
        let secs = t.elapsed().as_secs_f64();
        let points = m.body("prompt batch")?.checkpoints().positions();
        let (digests, logits) = (store_digests(m)?, m.logits()?);
        let ours = by_layer(&store_rows(m, p)?);
        let theirs = by_layer(&want.stores);
        let taps = prompt_route_taps(m, p)?;
        let tally = judge_routes("gemm", p, (&taps, routes), bias, &bands.stream);
        let reach = tally.low.unwrap_or(N_LAYER - 1);
        let (mut held_ok, mut worst, mut worst_l, mut past, mut past_l) =
            (true, 0.0f64, 0usize, 0.0f64, 0usize);
        for (l, (o, w)) in ours.iter().zip(&theirs).enumerate() {
            let r = rel(o, w) / bands.stores[l];
            if l <= reach {
                held_ok &= r <= 1.0;
                if r > worst || r.is_nan() {
                    (worst, worst_l) = (r, l);
                }
                if r > 1.0 || r.is_nan() {
                    println!(
                        "prompt batch G=1 P={p}: layer {l}'s stores {:.3e} from the reference's, \
                         past its band {:.3e} FAIL",
                        r * bands.stores[l],
                        bands.stores[l]
                    );
                }
            } else if r > past {
                (past, past_l) = (r, l);
            }
        }
        let head = bands.stream[N_LAYER - 1];
        let logits_rel = rel(&logits, &want.logits);
        let logits_ok = tally.low.is_some() || logits_rel <= head;
        let top = argmax(&want.logits);
        let runner = second(&want.logits, top);
        let gap = f64::from(want.logits[top as usize]) - f64::from(want.logits[runner as usize]);
        let head_cap = 6.0 * head * rms(&want.logits);
        let tok_gap = f64::from(want.logits[top as usize])
            - want
                .logits
                .get(tok as usize)
                .map_or(f64::NAN, |&v| f64::from(v));
        let argmax_ok = tok == want.after.argmax || tok_gap <= head_cap;
        let pass = tally.ok()
            && held_ok
            && logits_ok
            && argmax_ok
            && m.pos() as usize == p
            && points == marks(p);
        println!(
            "prompt batch G=1 P={p}: against calls of {CHUNK}: {} flip(s); {} first on their path, \
             {} of them past the pair rule's cap, {} at a wide margin; {} past an earlier flip, {} \
             of them past the cap, the worst at {:.2} of it (printed); the reference's margins \
             over the cap at the flips: median {:.3}, 99th percentile {:.3}, largest {:.3}, {} \
             past it, {} of them first on their path (held); the stores of layers 0..={reach} (no flip's path) held to their bands, \
             worst {worst:.3} of it at layer \
             {worst_l}; past them (printed) worst {past:.3} at layer {past_l}; logits {logits_rel:.3e} \
             from the reference's (band {head:.3e}, {}); argmax {tok} (reference {}, {tok_gap:.3e} \
             below its top, cap {head_cap:.3e}; its runner-up {runner} at {gap:.3e}{}); pos {} \
             checkpoints {points:?} (want {:?}), \
             {secs:.2} s (runtime value) {}",
            tally.flips,
            tally.first,
            tally.refused,
            tally.wide,
            tally.flips - tally.first,
            tally.past_over,
            tally.past_worst,
            tally.margin_at(0.5),
            tally.margin_at(0.99),
            tally.margin_at(1.0),
            tally.margin_over,
            tally.margin_first,
            if tally.low.is_some() {
                "printed: a flip lies on its path"
            } else {
                "held"
            },
            want.after.argmax,
            if tok == runner && tok != want.after.argmax {
                ", ours"
            } else {
                ""
            },
            m.pos(),
            marks(p),
            verdict(pass)
        );
        Ok((
            pass,
            Some(After {
                p,
                stores: digests,
                logits: fnv_row(&logits),
                argmax: tok,
            }),
        ))
    }

    /// (pb)'s forced arm at groups of one: a call of `want.after.p` ids from a
    /// reset with the reference's routes planted (`plant_prompt_routes`:
    /// every pick and weight the chunk calls' at every position), so both
    /// take the same experts and weights and no flip can happen: every
    /// layer's live stores within its band ([`gemm_bands`]), no layer
    /// excused; the last logits within the head's band; the argmax the
    /// reference's, or one whose logit in the reference's row lies within six
    /// head bands of the row's RMS below its top.
    fn gemm_forced(
        m: &mut Glm5nextModel,
        ids: &[u32],
        want: &Held,
        bands: &Bands,
    ) -> Result<bool, GateError> {
        let p = want.after.p;
        m.reset()?;
        let tok = match prefill(m, &ids[..p]) {
            Ok(tok) => tok,
            Err(e) => {
                println!(
                    "prompt batch forced P={p}: error \"{e}\" {}",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let logits = m.logits()?;
        let ours = by_layer(&store_rows(m, p)?);
        let theirs = by_layer(&want.stores);
        let (mut held_ok, mut worst, mut worst_l) = (true, 0.0f64, 0usize);
        for (l, (o, w)) in ours.iter().zip(&theirs).enumerate() {
            let r = rel(o, w) / bands.stores[l];
            held_ok &= r <= 1.0;
            if r > worst || r.is_nan() {
                (worst, worst_l) = (r, l);
            }
            if r > 1.0 || r.is_nan() {
                println!(
                    "prompt batch forced P={p}: layer {l}'s stores {:.3e} from the reference's, \
                     past its band {:.3e} FAIL",
                    r * bands.stores[l],
                    bands.stores[l]
                );
            }
        }
        let head = bands.stream[N_LAYER - 1];
        let logits_rel = rel(&logits, &want.logits);
        let top = argmax(&want.logits);
        let tok_gap = f64::from(want.logits[top as usize])
            - want
                .logits
                .get(tok as usize)
                .map_or(f64::NAN, |&v| f64::from(v));
        let head_cap = 6.0 * head * rms(&want.logits);
        let argmax_ok = tok == want.after.argmax || tok_gap <= head_cap;
        let pass = held_ok && logits_rel <= head && argmax_ok;
        println!(
            "prompt batch forced P={p}: the reference's routes planted: every layer's stores held \
             to its band, worst {worst:.3} of it at layer {worst_l}; logits {logits_rel:.3e} from \
             the reference's (band {head:.3e}); argmax {tok} (reference {}, {tok_gap:.3e} below its \
             top, cap {head_cap:.3e}) {}",
            want.after.argmax,
            verdict(pass)
        );
        Ok(pass)
    }

    /// One batch call of `a.p` ids from a reset at the load's group against
    /// `a`, bit for bit: the stores, the logits, the argmax, the position and
    /// the checkpoints. `what` names the reference.
    fn batch_bits(
        m: &mut Glm5nextModel,
        ids: &[u32],
        a: &After,
        g: usize,
        what: &str,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let t = Instant::now();
        let got = prefill(m, &ids[..a.p]);
        let secs = t.elapsed().as_secs_f64();
        let (argmax, stores, logits) = match got {
            Ok(tok) => (tok, store_digests(m)?, fnv_row(&m.logits()?)),
            Err(e) => {
                println!(
                    "prompt batch G={g} P={}: error \"{e}\" {}",
                    a.p,
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let points = m.body("prompt batch")?.checkpoints().positions();
        let diff = first_store_diff(&stores, &a.stores);
        let pass = diff.is_none()
            && logits == a.logits
            && argmax == a.argmax
            && m.pos() as usize == a.p
            && points == marks(a.p);
        println!(
            "prompt batch G={g} P={}: against {what}: stores {} logits {} argmax {argmax} ({what} \
             {}) pos {} checkpoints {points:?} (want {:?}), {secs:.2} s (runtime value) {}",
            a.p,
            diff.as_deref()
                .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
            if logits == a.logits {
                "bit for bit"
            } else {
                "differ"
            },
            a.argmax,
            m.pos(),
            marks(a.p),
            verdict(pass)
        );
        Ok(pass)
    }

    /// (l1), (pb), (pr), (pf), (pg) on a load at [`CTX_PP`] of one KDA lane
    /// whose session feeds in batches.
    fn prompt_batch(levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let (mut s, _) = open(levers, CTX_PP, PrefillMode::Batch, KdaLanes::One)?;
        let m = s.model_mut();
        let ids = lcg_ids(CTX_PP + 1);
        let mut ok = one_lane(m, &ids[..3])?;
        // (pb): the steps' record, then the chunk calls' against it.
        m.reset()?;
        let t = Instant::now();
        let mut after = Vec::new();
        for (i, &id) in ids[..CTX_PP].iter().enumerate() {
            let argmax = m.step(&[id])?;
            if PP.contains(&(i + 1)) {
                after.push(After {
                    p: i + 1,
                    stores: store_digests(m)?,
                    logits: fnv_row(&m.logits()?),
                    argmax,
                });
            }
        }
        println!(
            "prompt batch: {CTX_PP} plain steps in {:.1} s (runtime value), {} stores digested \
             at each of {PP:?}",
            t.elapsed().as_secs_f64(),
            after.first().map_or(0, |a| a.stores.len())
        );
        set_prefill_group(m, 1)?;
        set_prompt_route_taps(m, CTX_PP)?;
        let Reference {
            ok: chunks_ok,
            held: reference,
            routes,
        } = chunk_calls(m, &ids, &after)?;
        ok &= chunks_ok;
        let bands = gemm_bands(&m.body("prompt batch")?.kinds());
        let bias = biases(m)?;
        println!(
            "prompt batch: the GEMM bands: stores at layers 0..=4 {:.3e} {:.3e} {:.3e} {:.3e} \
             {:.3e}; the stream at layers 0, 3, 44 {:.3e} {:.3e} {:.3e}",
            bands.stores[0],
            bands.stores[1],
            bands.stores[2],
            bands.stores[3],
            bands.stores[4],
            bands.stream[0],
            bands.stream[3],
            bands.stream[N_LAYER - 1]
        );
        // Groups of one: a call of at most a chunk the steps' bits, one past
        // it against the chunk calls.
        let mut ones = Vec::with_capacity(after.len());
        for (a, r) in after.iter().zip(&reference) {
            if a.p < GEMM_FROM {
                ok &= batch_bits(m, &ids, a, 1, "steps")?;
                ones.push(Some(After {
                    p: a.p,
                    stores: a.stores.clone(),
                    logits: a.logits,
                    argmax: a.argmax,
                }));
            } else {
                let (pass, got) = gemm_against(m, &ids, r, &routes, (&bias, &bands))?;
                ok &= pass;
                ones.push(got);
            }
        }
        // The forced arm: the reference's routes planted.
        set_prompt_route_taps(m, 0)?;
        plant_prompt_routes(m, Some((&routes, CTX_PP)))?;
        for r in reference.iter().filter(|r| r.after.p >= GEMM_FROM) {
            ok &= gemm_forced(m, &ids, r, &bands)?;
        }
        plant_prompt_routes(m, None)?;
        drop(reference);
        // Groups of two and four: the calls at groups of one, bit for bit.
        for g in GROUPS.into_iter().filter(|&g| g > 1) {
            set_prefill_group(m, g)?;
            for one in &ones {
                let Some(one) = one else {
                    println!("prompt batch G={g}: no call at groups of one to hold it to FAIL");
                    ok = false;
                    continue;
                };
                ok &= batch_bits(m, &ids, one, g, "groups of one")?;
            }
        }
        set_prefill_group(m, 1)?;
        ok &= refused_past(m, &ids)?;
        set_prefill_group(m, 2)?;
        ok &= planted_group(m, &ids)?;
        let clean = ones.iter().flatten().find(|a| a.p == 9).map(|a| a.argmax);
        for g in [1, 2] {
            set_prefill_group(m, g)?;
            ok &= batch_fault(m, &ids[..9], &ids[..513], clean, g)?;
        }
        set_prefill_group(m, 1)?;
        Ok(ok)
    }

    /// (pg) at the load's group of two: the calls of 513 ids and of the next
    /// 517 from a reset, unplanted — the reference, every token in the batch
    /// it lands in below (a batch's bits depend on its side of [`GEMM_FROM`],
    /// so the reference makes the same cut); then after the call of 513, the
    /// call of the next 517 — one group of two batches, the mark at 1024
    /// inside it — with [`Plant::Group`]`(1)` planted fails by name, the model
    /// standing at 513 and the points from before it (512, 513) standing; the
    /// call again gives the reference's stores, logits and argmax, its points
    /// 512, 513, 1024 and 1030. The model left reset.
    fn planted_group(m: &mut Glm5nextModel, ids: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        prefill(m, &ids[..513])?;
        let want_tok = prefill(m, &ids[513..1030])?;
        let (want_stores, want_logits) = (store_digests(m)?, fnv_row(&m.logits()?));
        m.reset()?;
        prefill(m, &ids[..513])?;
        let before = m.body("planted group")?.checkpoints().positions();
        m.body_parts("planted group")?.2.plant(Plant::Group(1));
        let failed = prefill(m, &ids[513..1030]);
        let named = matches!(&failed, Err(GpuError::State { missing, .. })
            if missing.contains("the planted failure in a prompt group's walk"));
        let (pos, points) = (m.pos(), m.body("planted group")?.checkpoints().positions());
        let again = prefill(m, &ids[513..1030]);
        let (stores, logits) = (store_digests(m)?, fnv_row(&m.logits()?));
        let after_points = m.body("planted group")?.checkpoints().positions();
        m.reset()?;
        let diff = first_store_diff(&stores, &want_stores);
        let ok = named
            && before == [512, 513]
            && pos == 513
            && points == before
            && matches!(again, Ok(t) if t == want_tok)
            && diff.is_none()
            && logits == want_logits
            && after_points == [512, 513, 1024, 1030];
        println!(
            "planted group: after 513 ids (points {before:?}) a call of 517 at groups of 2 with \
             a failure planted at unit 1: {}; the model at {pos} (want 513), points {points:?} \
             (want {before:?}); the call again: {}, stores {}, logits {}, points \
             {after_points:?} (want [512, 513, 1024, 1030]), against the same two calls unplanted {}",
            match &failed {
                Ok(t) => format!("token {t}"),
                Err(e) => format!("error \"{e}\""),
            },
            match &again {
                Ok(t) => format!("token {t} (unplanted {want_tok})"),
                Err(e) => format!("error \"{e}\""),
            },
            diff.as_deref()
                .map_or("bit for bit".to_string(), |d| format!("differ at {d}")),
            if logits == want_logits {
                "bit for bit"
            } else {
                "differ"
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (l1) on the plain session's load: one KDA lane, the stores the header's
    /// bytes at that lane; a verify of two rows refused by name in eager
    /// mode and in its capture, the model standing where it stood with the
    /// stores as they were, and the step after it the plain step. The model
    /// left reset, in the step mode it came in.
    fn one_lane(m: &mut Glm5nextModel, toks: &[u32]) -> Result<bool, GateError> {
        const WANT: &str = "on a load of one KDA lane";
        let mode = m.mode();
        m.reset()?;
        m.set_mode(StepMode::Eager);
        let body = m.body("one lane")?;
        let lanes = body.lanes();
        let (stores, want) = (body.store_bytes(), store_bytes(CTX_PP, 1));
        let t0 = m.step(&toks[..1])?;
        let before = store_digests(m)?;
        let pos = m.pos();
        let rows = m.step_rows::<2>([toks[1], toks[2]]);
        let captured = m.capture_rows::<2>();
        let after = store_digests(m)?;
        let stood = m.pos() == pos && after == before;
        fn named<T>(r: &Result<T, GpuError>) -> bool {
            matches!(r, Err(GpuError::Shape { detail, .. }) if detail.contains(WANT))
        }
        let t1 = m.step(&toks[1..2]);
        let reset_ok = m.reset().is_ok();
        let plain = m.step(&toks[..1]).and_then(|_| m.step(&toks[1..2]));
        m.reset()?;
        m.set_mode(mode);
        let ok = lanes == KdaLanes::One
            && stores == want
            && named(&rows)
            && named(&captured)
            && stood
            && reset_ok
            && matches!((&t1, &plain), (Ok(a), Ok(b)) if a == b);
        let text = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        println!(
            "(l1) the plain load: {} KDA lanes, store bytes {stores} (want {want}, derived at one \
             lane); after a step ({t0}) a verify of two rows: {}; its capture: {}; the model at \
             {pos} with its stores as they were ({stood}); the next step {} (plain {}) {}",
            lanes.count(),
            match &rows {
                Ok(r) => format!("tokens {r:?}"),
                Err(e) => format!("error \"{e}\""),
            },
            match &captured {
                Ok(n) => format!("{n} nodes"),
                Err(e) => format!("error \"{e}\""),
            },
            text(&t1),
            text(&plain),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (pr): a call one position past the stores, on each feed.
    fn refused_past(m: &mut Glm5nextModel, ids: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        let zero = store_digests(m)?;
        let mut ok = true;
        for mode in [PrefillMode::Batch, PrefillMode::Steps] {
            set_prefill(m, mode)?;
            let r = feed(m, &ids[..CTX_PP + 1]);
            let named = matches!(&r, Err(GpuError::Shape { detail, .. })
                if detail.contains(&format!("past the {CTX_PP} positions")));
            let same = store_digests(m)? == zero;
            let pass = named && m.pos() == 0 && same;
            println!(
                "prompt past the stores ({} feed): {} ids: {}; pos {} stores as the reset's {same} {}",
                mode.name(),
                CTX_PP + 1,
                match &r {
                    Ok(t) => format!("token {t}"),
                    Err(e) => format!("error \"{e}\""),
                },
                m.pos(),
                verdict(pass)
            );
            ok &= pass;
        }
        set_prefill(m, PrefillMode::Batch)?;
        Ok(ok)
    }

    /// Layer `l`'s attention norm, whose first value (pf) sets to NaN.
    fn patch_attn_norm(
        m: &mut Glm5nextModel,
        l: usize,
        bytes: [u8; 4],
    ) -> Result<[u8; 4], GateError> {
        let name = names::attn_norm(l);
        let (gpu, w, _) = m.body_parts("gate_glm5next_e2e patch_attn_norm")?;
        let Some(DevWeight::F32 { w: gain, .. }) = w.get(&name) else {
            return Err(format!("{name} is not resident as F32").into());
        };
        patch_bytes(gpu.stream(), gain.buf(), 0, bytes)
    }

    /// (pf) at groups of `g`: a NaN in layer 0's norm; a step and a batch
    /// call end in a fault at layer 0, the model poisoned — the batch's the
    /// step's with the GEMM's quantizer's site added ([`FaultSite::QuantColumn`],
    /// the smallest code: a batch of nine runs its projections on the GEMM,
    /// whose quantizer refuses the normed input's NaN block, and the NaN it
    /// leaves in every output row meets the step's sites after it); a call of
    /// two batches ends in the batch's after the group that holds its first,
    /// with no checkpoint taken or sealed of the faulted state, and so does
    /// that call with the NaN in the last KDA layer's norm instead (the fault
    /// at that layer, after every store of the first batch reached its take);
    /// the weight put back, a reset and the batch call give `clean`, (pb)'s
    /// argmax at nine positions at groups of one.
    fn batch_fault(
        m: &mut Glm5nextModel,
        ids: &[u32],
        two: &[u32],
        clean: Option<u32>,
        g: usize,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let old = patch_attn_norm(m, 0, f32::NAN.to_le_bytes())?;
        let step = m.step(&ids[..1]);
        let step_poison = m.poisoned();
        m.reset()?;
        let batch = prefill(m, ids);
        let batch_poison = m.poisoned();
        let again = prefill(m, ids);
        m.reset()?;
        let zero_points = m.body("batch fault")?.checkpoints().positions();
        let long = prefill(m, two);
        let long_points = m.body("batch fault")?.checkpoints().positions();
        patch_attn_norm(m, 0, old)?;
        // The same call with the NaN in the last KDA layer's norm: the fault
        // comes after the first batch's every store reached its take.
        let last_kda = m
            .body("batch fault")?
            .kinds()
            .iter()
            .rposition(|k| k.mixer == MixerKind::DeltaRule)
            .ok_or("no KDA layer")?;
        m.reset()?;
        let old_late = patch_attn_norm(m, last_kda, f32::NAN.to_le_bytes())?;
        let late = prefill(m, two);
        let late_points = m.body("batch fault")?.checkpoints().positions();
        patch_attn_norm(m, last_kda, old_late)?;
        m.reset()?;
        let restored = prefill(m, ids);
        m.reset()?;
        let fault_of = |r: &Result<u32, GpuError>| match r {
            Err(GpuError::Fault { fault, .. }) => Some(*fault),
            _ => None,
        };
        let (fs, fb) = (fault_of(&step), fault_of(&batch));
        let quant = FaultSite::QuantColumn as u32;
        let with_quant = |f: Fault| (f.layer, f.code.min(quant), f.sites | (1 << quant));
        let fb_is_gemm = fs
            .zip(fb)
            .is_some_and(|(s, b)| (b.layer, b.code, b.sites) == with_quant(s));
        let early = fault_of(&long) == fb && long_points == zero_points;
        let late_ok = fault_of(&late).is_some_and(|f| f.layer as usize == last_kda)
            && late_points == zero_points;
        let named = fs.is_some()
            && fb_is_gemm
            && fs.is_some_and(|f| f.layer == 0)
            && step_poison == fs
            && batch_poison == fb;
        let refused = matches!(again, Err(GpuError::Poisoned { .. }));
        let back = clean.is_some() && matches!(restored, Ok(t) if Some(t) == clean);
        let ok = named && early && late_ok && refused && back;
        let show = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        println!(
            "batch fault (groups of {g}): NaN in blk.0.attn_norm.weight[0]: a step: {}; a batch \
             of {}: {}; \
             poisoned {} / {}; the next call: {}; a call of {}: {}, checkpoints {long_points:?} \
             (the reset's {zero_points:?}); with the NaN in layer {last_kda}'s norm instead, a \
             call of {}: {}, checkpoints {late_points:?}; the weight put back and a reset: {} \
             (want token {}) {}",
            show(&step),
            ids.len(),
            show(&batch),
            step_poison.map_or_else(|| "none".to_string(), |f| f.to_string()),
            batch_poison.map_or_else(|| "none".to_string(), |f| f.to_string()),
            show(&again),
            two.len(),
            show(&long),
            two.len(),
            show(&late),
            show(&restored),
            clean.map_or_else(|| "none".to_string(), |t| t.to_string()),
            verdict(ok)
        );
        Ok(ok)
    }
}
