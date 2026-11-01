//! The Qwen3.6-35B-A3B end-to-end gate: the whole chain — 30 gated-delta-rule
//! layers, 10 gated GQA layers at head 256, 40 MoE blocks with the shared
//! expert as a ninth slot, the head and the argmax — on one card, against
//! ik's CPU oracle sets (`refset::arch::qwen35moe`: the 5-token batch set,
//! the step after a 4-token prefill, the step after a 1,024-token prefill),
//! all read in one process over one load.
//!
//! The layer counts, the Q6_K sites and the captured graphs' node counts below
//! are the real file's; the gate derives each from the header of the file it
//! opens (`q36_fixture::Shape`) and prints it beside that literal
//! (`tier::witness`).
//!
//! Tiers (`BLOOMERY_TIER`, `crates/gpu-gates/src/tier.rs`): the real tier runs every clause on the real files. The
//! fixture tier runs on the file `ref_model_path` names: the base fixture (4 layers, every shared expert joined) or one
//! of its variants (`BLOOMERY_FIXTURE_VARIANT=unjoined|unjoinedq6`: the same 4 layers in UD-Q4_K_M's layout, the shared
//! expert apart from the routed stacks, the routed down Q5_K or Q6_K). Every clause carries one tag:
//! - self-consistency, both tiers: (s), (p), (p′), (w), (w′), (k), (r), (e), (v), the shape and the placed load's own
//!   lines. Their ids come from ik's sets of the tier ([`ik_set`]), the fixture copying the real file's vocabulary.
//! - oracle in the real tier, fixture-oracle in the fixture tier ([`ik_oracle`]): (c), (f), (t), (u) with (d), and
//!   (c) and (u) with (d) again on (o). The fixture tier compares with ik's sets dumped on the fixture
//!   (`refset::arch::qwen35moe::fixture`, each opened by `tier::fixture_set`, resolved to the variant's own sets under
//!   `BLOOMERY_FIXTURE_VARIANT`); the real tier prints a `fixture-only(oracle)` line for each, and the gate's end holds
//!   the count it decided to [`FX_ORACLE`] (`tier::expect_fixture_oracle`). The bands are the real tier's: they are
//!   per value, from ik's inputs and our rules. Two forms differ by tier, both read from `cfg().tier`: the
//!   free-running argmax of (c), (o2), (o7) and (t1) is equality with ik's in the real tier and [`flip::head_tie`]
//!   in the fixture tier ([`head_argmax`]), and (t1)'s new-state term is `state_rel / gap` in the real tier and the
//!   larger of that gap and the layer input's own distance in the fixture tier ([`state_ratio`]).
//! - `no-layout-layer` (`tier::layout_clause`): (n) reads joined layers (the node count of a slot,
//!   [`Shape::nodes_slot`]), (j) and (o8) apart ones. (j) and (o8) open [`UNJOINED`] in the real tier and the tier's file
//!   in the fixture tier, so they run on a variant and are left on the base fixture; (n) the other way round.
//! - scale: (m).
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`Shape::nodes_decode`]
//!   nodes, all kernels; a captured pass of `m` rows holds
//!   [`Shape::nodes_pass`] `+ 4m`
//!   (`m` memcpys of a row into its head, the rest kernels), at `m` = 5 and
//!   8; each count also equals the body's own launch count
//!   (`Body35::pass_launches`) plus its heads. The layer stores' bytes equal
//!   their derivation from the header ([`store_bytes`]). The flash
//!   launches' grids: the segment pass `n_kv · SEGMENTS` blocks an
//!   attention layer and the merge `n_head`, one of each an attention
//!   layer, whatever the cache height the load allocated.
//! - (p) one layer body: the batch set's five tokens as five decode steps
//!   in graph mode, as one pass of five rows (`step_rows`) and as five eager
//!   steps, each from zero stores written by the gate, then as five graph
//!   steps from `reset` alone after the eager run's state, leave the same
//!   per-position logits and tokens and every layer's store (K/V planes,
//!   recurrent state, conv ring) bit for bit. No reduction differs between
//!   the decode step and the pass: every launch of a pass computes each row
//!   the way its one-row launch does (the FFN norm inside the router at one
//!   row is `norm_quant` + the router bit for bit, gate_qwen35moe_moe's norm
//!   clause; the flash and rope rows are their one-row launches',
//!   gate_qwen35moe_attn; the q4_K projections run the one-column body;
//!   `q6k_gemv` keeps one accumulator chain per column whatever the column
//!   count). Every other clause starts from the gate's zero stores, so a
//!   `reset` that keeps state reddens the last line of this clause alone.
//! - (c) free-running on the batch set: the five decode steps' every layer
//!   output against ik's `l_out-L` within [`FREE_BAND`], and the last
//!   position's argmax equal to ik's `result_output` argmax in the real tier;
//!   in the fixture tier a tie ([`head_argmax`]): ik's own logit at our argmax
//!   within [`flip::head_cap`] of its top, at the run's last layer output's
//!   relative distance from ik's.
//! - (f) teacher-forced, per layer, on the batch set as one pass of five
//!   rows from a reset: each layer run alone on ik's own input rows
//!   (`inp_embd` for layer 0, `l_out-(L−1)` after), its mixer's taps against
//!   ik's, and its FFN half run alone on ik's own FFN input against ik's MoE
//!   taps; every tap's relative error over the relative distance between
//!   the two sides' 8-bit activations of the input that reaches it
//!   ([`quant_gap_two_sided`]) within [`RATIO_BAND`]; the routed ids equal as sets
//!   except where ik's own margin between its eighth pick and the best
//!   other expert lies inside our logits' measured error (counted,
//!   printed); the decay's underflow sites (exp(g) = 0) equal.
//! - (t) the decode step after each step set's prefill: the prefill's state
//!   loaded from the set's inputs (`cache_s_lL` — conv then ssm —,
//!   `cache_k_lL`, `cache_v_lL`) into every layer's store, the model stood at
//!   the set's position, then (t1) the step free-running: every layer
//!   output within [`FREE_BAND`], the argmax as (c) takes it (equal in the
//!   real tier, a tie in the fixture tier), every delta layer's new state
//!   within [`RATIO_BAND`] of the input gap in the real tier and of the
//!   larger of that gap and the layer input's own distance from ik's in
//!   the fixture tier ([`state_ratio`]); and (t2) each
//!   layer teacher-forced at one row on its reloaded store, its taps as (f).
//! - (p′) the ubatch arena's gemv arm: the batch set's five tokens through
//!   the prompt call (`prefill_with`) from zero stores — on the pass path
//!   (one unit of five rows) and on the GEMM path (a ubatch of five, which
//!   is no more than `GEMV_COLS` and so the same walk) — leave (p)'s five
//!   decode steps' last token, last logits (the last unit's last row
//!   through the head, `Tail::Last`) and every store bit for bit.
//! - (u) the wide arm against the gemv arm: the 1,024 prompt ids of the
//!   step-1,024 set prefilled on the pass path (128 units of eight, bit for
//!   bit the one-token path) and as one ubatch of 1,024 (the wide arm), each
//!   from zero stores: layer 0's recurrent state and conv ring within
//!   [`U_L0_REL`] of the pass run's; every layer's store distance and the
//!   prefill's last logits printed, and (d)
//!   prints each layer's store distance between the two runs beside its
//!   distances from ik.
//! - (d) both prefills against ik's state after its 1,024-token prompt batch
//!   (the set's `cache_s_lL`, `cache_k_lL`, `cache_v_lL`): each layer's
//!   store distance from ik's, the ubatch run's over the pass run's, within
//!   [`PROMPT_RATIO`]; then the step at position 1,024 free-running from each
//!   run's own state: each layer's output distance from ik's `l_out`, and the
//!   logits' from ik's `result_output`, the ubatch run's over the pass run's
//!   within [`STEP_RATIO`]; the three argmaxes printed.
//! - (w) the ubatch size moves no bit: the same 1,024 ids on the GEMM path at
//!   each size of [`W_SIZES`] (ubatches of 512 x 2; 1,000 and 24; 100 x 10
//!   and 24), and at [`U_GATE`] in two calls cut at [`W_CUT`], leave the last
//!   token, the last logits and every store of the one-ubatch run bit for
//!   bit. Every unit of these runs is wide: a unit of at most `GEMV_COLS`
//!   rows is the gemv arm, a band away.
//! - (w′) the wide plan: for the load's ubatch `ub`, every
//!   [`PrefillPath::Wide`] plan of [`WIDE_FROM`]..=`3 · ub` ids — one, two
//!   and three whole ubatches and every tail class of each — holds only
//!   ubatch units of [`WIDE_FROM`]..=`ub` rows that sum to the call's ids,
//!   and none is refused (a ubatch of at least `2 · WIDE_FROM` rows splits
//!   every tail); and a wide call whose tail is past the borrow's floor
//!   (`ub + 16` ids) runs uncut and as two calls cut at the whole ubatch,
//!   leaving the last token, last logits and every store bit for bit — the
//!   schedule's own promise (a row's bits do not depend on the unit it
//!   lands in), not a band.
//! - (k) checkpoints, on a second load at [`KCTX`] (the marks' positions
//!   need the room [`CTX`] does not hold) with the marks armed and an lcg
//!   prompt of [`KP`] ids on the seat's schedule
//!   (`PrefillPath::Wide`): a prompt call takes its checkpoints at the
//!   load's ubatch multiples inside it and [`KP`] (the spacing pinned in
//!   the clause), and one checkpoint's bytes and slots are their derivation
//!   (each delta layer's state of `N_V · HEAD_V²` f32 and conv ring of
//!   `RING_ROWS · C`, the host budget's share); the call runs exactly
//!   ⌈[`KP`]/ub⌉ walks, one a unit of its own plan, the re-fed tail after a
//!   cut one walk; a cut to an ask between the last inner mark and [`KP`]
//!   keeps it, one below that keeps the mark under it, one below the first
//!   keeps nothing (`no-checkpoint`), each with its code, and a cut to a
//!   non-checkpoint is refused by name; a cut to the last inner mark and
//!   the re-fed tail leave the token, the last logits and every store bit
//!   for bit the uncut call's (the call takes its points back, the mark
//!   held); a reset drops every point, and the captured step after it holds
//!   the flash launches at this load's height with (s)'s grids — the
//!   engine-level two-height pin, no second load of its own. FAIL-first
//!   mutants, each red on its
//!   line: the spacing put back at the flat 512 it was (the points and the
//!   walk count red); the restore omitted (the cut a Stay: the re-feed runs
//!   from the fed state, the stores and logits red); `kept` answering the
//!   ask instead of the checkpoint (the neighbours red, and the refused cut
//!   not refused); `reset` keeping the checkpoints (the dropped-points line
//!   red).
//! - (n) resident slots (`add_slots`), on their own load with the
//!   checkpoints armed, as the seat arms them: the slot harness's contracts
//!   (`slots_gate`: H1 interleave, H2 one pass, H3 bytes, H4 reset, H6
//!   refusals, H7 captures) over two lcg streams, A of [`SLOT_A`] ids and B of
//!   [`SLOT_B`] — each past the load's first mark (its ubatch, [`U_GATE`])
//!   by a run the mark keeps — prompted by the session's schedule (the wide
//!   walk), [`SLOT_STEPS`] steps a stream, each slot's state digest every
//!   attention layer's K/V rows below its position and every delta layer's
//!   state and ring, `seq_bytes` held to [`store_bytes`], and H2's pass of
//!   one row a slot holding [`Shape::nodes_pass`] + [`Shape::nodes_slot`] + its head's
//!   five nodes. Between the harness's halves: a pass of nine rows, a slot
//!   twice, a slot out of range, a slot of no token, an id past the
//!   vocabulary and a pass with the layer taps armed are each refused by
//!   name, every slot's position kept; and each slot's checkpoints are its
//!   own: after the interleave and H2 slot 0 holds its points at the mark
//!   and its prompt's end,
//!   slot 1 at the mark and its own end; slot 1 cut back to the mark, slot
//!   0 — selected before slot 1 runs again, so a cut that waited in the
//!   wrong checkpoints is carried out on it — holds its points and digest
//!   and its next ids are its solo run's continuation; slot 1, the rest of
//!   B re-fed from the mark and stepped as far as the interleave and H2
//!   stepped it, stands with H2's last id, digest, points and position;
//!   a reset of slot 1 leaves slot 0's points standing.
//! - (j) the unjoined shared expert, on its own load of [`unjoined_path`] (no
//!   ik set of that file but a fixture variant's): every
//!   layer's shared expert kept apart from its routed stacks exactly where
//!   the header gives a part of another type than its stack's (here every
//!   layer: q8_0 shared parts beside q4_K gate and up and q5_K or q6_K
//!   downs); the captured decode step [`Shape::nodes_decode`] of that file's
//!   header (844 nodes on the real one), all kernels, and a pass of `m` rows
//!   [`Shape::nodes_pass`] (841) `+ 4m`, each the
//!   body's launch count plus its heads; (p) and (p′) on the batch set's
//!   tokens; and (w)'s cut on the step-1,024 set's prompt (one ubatch of
//!   [`U_GATE`] and two calls cut at [`W_CUT`], the wide arm's unjoined
//!   launches), each bit for bit. The FFN half against the host: the first
//!   layer of each of the file's FFN type combinations, run alone on each
//!   arm — the gemv arm (`ffn_rows`) on the rows a decode run of the batch
//!   set's tokens left at the layer before it, the wide arm
//!   (`ffn_rows_wide`) on [`WIDE_ROWS`] of them cycled — against the host's
//!   f64 FFN of the file's own matrices (dequantized as stored) under the
//!   card's routing — each
//!   routed slot's `w · down(silu(gate · x̂) · up · x̂)` and the shared
//!   expert's at the router's sigmoid weight — the update's relative error
//!   within [`UNJOINED_FFN_BAND`]. (s) holds each layer's shared expert to
//!   the header's layout. `--unjoined-only` runs (j) alone.
//! - (r) refusals: a ubatch size of 0 or past `UBATCH` is refused by name
//!   with the size and the resident bytes kept; a prompt past the cache is
//!   refused by name before any launch, the position kept.
//! - (v) the q8_0 cache arm (`BLOOMERY_QWEN3_KV=q8_0`, the seat's
//!   `--cache-type-k q8_0`), on its own models: the store bytes' derived
//!   drop (the attention planes' two-plane layout against f16, the delta
//!   layers' stores f32 either way) carrying the resident bytes' delta
//!   exactly, and the first 96 prompt ids of the step-1,024 set stepped, prefilled on the
//!   pass path and replayed through the captured step graph leaving the
//!   same last logits bit for bit and the same argmax; the distance to the
//!   f16 run prints as the quantization diagnostic.
//! - (o) the placed load: the file planned on device 0 under a card budget
//!   of [`placed_budget`] (`shared/qwen3moe_place.rs`, the CLI's and the
//!   seat's planner), which must leave routed experts both on the card and
//!   on the host tier, then loaded by that plan (`open_placed`, the shared
//!   expert its own one-expert stacks on the card): (o1) the batch set's
//!   five tokens as five graph steps and as five eager steps from zero
//!   stores leave the same logits, tokens and stores bit for bit, and so
//!   does the prompt call on the pass path (one unit of five rows through
//!   the host tier's batch port) against the fifth step — each row of a
//!   unit runs every card launch and every host dot the way its one-row
//!   step does; (o2) (c) on it, the same [`FREE_BAND`] — a host expert runs
//!   ik's own 8-bit rule (q8_2 per 32 values for Q4_K-Q6_K and the 32-block
//!   types, Q8_K per 256 for the rest) where the card runs q8_1, so no
//!   layer's error grows; (o5) (u) and (d) on it ([`wide_arm`]): the step
//!   set's prompt as ubatches of [`PLACED_UBATCH`] rows, whose layers run
//!   the card's experts through the wide GEMMs over a per-layer map and the
//!   host's once a unit, against the pass run, within the same bands; (o3)
//!   a ubatch resize refused by name, the ubatch kept, and two slots of the
//!   placed load in one pass (`GpuModel::step_slots`) give the greedy ids of
//!   the same two prompts stepped in turn, in graph mode and in eager mode:
//!   the batch set's tokens and the lcg walk [`q35_clauses::lcg_ids`] draws
//!   over the load's vocabulary, as many ids as the set's — one construction
//!   on every tier and fixture variant, so the two prompts differ — each
//!   prefilled on the pass path from zero stores and then 4 steps —
//!   FAIL-first: both columns served from one prompt, or the columns swapped,
//!   reddens the id compare; `step_slots` still refused on the placed load
//!   fails the clause by name;
//!   (o4) the host tier served slots on both of its ports;
//!   (o7) the file planned on device 0 under no budget, a plan that keeps
//!   every routed expert on the card (a refusal by name where the card holds
//!   no such plan), loaded by the placed opener (`q3place::open_qwen35`): the
//!   body is the placed one with no host expert, its five graph steps leave
//!   the same greedy tokens as the whole-card load's, and (c) holds on it, the
//!   same [`FREE_BAND`]. FAIL-first: an opener that takes a plan with no host
//!   expert for the whole load (mutant: `open_qwen35` opening
//!   `GpuModel::<Body35>::open` when `plan.host.experts == 0`) leaves no
//!   placed side, and (o7) names it.
//!   `--placed-only` runs (o) and (o7) alone.
//! - (o8) the unjoined shared expert on the placed load, on [`unjoined_path`]
//!   (the real UD file in the real tier; the fixture tier's variants
//!   `unjoined` and `unjoinedq6`, tools/fixture-variants.tsv, carry its layout
//!   on the fixture's layers): the
//!   file whose every layer keeps a q8_0 shared expert beside a K-quant
//!   routed stack, planned on device 0 under the smallest card budget that
//!   leaves at least half its layer count of routed experts on the card (a bisection
//!   over the planner, which is host arithmetic: its expert rule fills the
//!   layers in ascending order, so this budget leaves the later layers no
//!   card expert at all, the mix [`placed_budget`]'s even spread cannot give),
//!   then loaded by that plan (`open_placed`). Asserted on the opened body:
//!   some layer holds no card expert and some holds at least one, the plan
//!   holds host experts, the arena is the plan's count and every shared
//!   expert is kept apart. The shared expert's three parts run as the dense
//!   FFN at their site types on every layer, whatever its card count (the
//!   placed twin of the whole-card load's apart arm), the q8_0 parts reading
//!   the f32 normed rows: the step boundary's on a decode step, the arena's on
//!   a unit. The reference is the same file's whole-card load, dropped before
//!   the placed one opens. One verdict line per arm: the gemv arm, the batch
//!   set's five tokens as five graph steps; the same tokens as a prompt call
//!   (a unit within the gemv arm's cut) against the fifth step, bit for bit
//!   as (o1) holds it, which reads the arena's rows against the boundary's;
//!   and the wide arm, [`PLACED_UBATCH`] ids of the step set's prompt as one
//!   GEMM unit past the gemv arm's cut ([`wide_arm`] is not used: its (d)
//!   half reads ik's sets, which are dumped from the other file). The two
//!   steps and the unit are held to the whole-card load by (e)'s rule, not
//!   bit for bit: every call's logits row within [`FREE_BAND`] of the
//!   whole-card load's (relative), the greedy ids equal or parting at a near
//!   tie ([`GREEDY_MARGIN`]) — a host expert runs ik's 8-bit rule where the
//!   card runs q8_1, so the two loads are two realizations of one rule, and
//!   (o7)'s exact ids hold only for a plan with no host expert.
//!   FAIL-first: a placed side that admits only a K-quant stack of one expert
//!   for the shared expert refuses the opener, `qwen3moe::Body::load:
//!   blk.0.ffn_gate_shexp.weight is not a resident K-quant word plane`.
//!   Mutants: the decode step reading the arena's normed rows in place of the
//!   boundary's (the step's front writes the boundary's, so the rows are stale
//!   and the bit-for-bit line parts from the step), a layer with no card
//!   expert dropping its shared expert or running it on other rows, its
//!   sigmoid weight dropped (each an update of order one: the logits band).
//!   Not seen: a q8_0 part reading the
//!   q8_1 of the normed rows in place of the f32 rows, an 8-bit reading of its
//!   input and well inside [`FREE_BAND`]. `--unjoined-placed-only` runs (o8)
//!   alone.
//! - (e) the residency machine on the placed load (`open_placed_residency`
//!   at the `mid` word [`residency_clauses::mid_of`] makes of (o)'s plan,
//!   through the session's calls: a prompt is one residency pass), against
//!   the same plan with it off, on the step set's prompt: the first 512 ids
//!   as one wide unit, 64 greedy steps (the flips land), the next 256 ids
//!   as a second wide unit, 16 steps — the residency run feeding the off
//!   run's ids. `flips`: at least one flip lands before the second prompt,
//!   and after each wide prompt every layer's map of the wide arm is the
//!   card copy's row followed by the shared expert's HOST entry
//!   (`Placed::stale_maps`), on both runs; every call's logits row within
//!   [`FREE_BAND`] of the off run's (relative), the ids equal or parting at
//!   a near tie ([`GREEDY_MARGIN`]). `c1` (green-only): the history twice gives the same ids and
//!   logits bit for bit. `c7`: a residency reset brings every live set back
//!   to its seed. `admit`: the history once more with the prompt calls'
//!   streaming `admit` (`Body35::set_xstream`): the first call admits
//!   experts on the wide arm's layers (none fails by name), keeps them (each
//!   layer's live set its start set with the pick's admits swapped in, every
//!   wide map the card copy's row), the second (a decode history before it)
//!   returns each layer to its start set, and every call's logits and ids
//!   hold as `flips` holds them. FAIL-first, `admit`: the layer's map not
//!   brought to the pick (the map line and the logits read it), and the
//!   places kept from the front's cut before the pick (the logits read it).
//!   `split`: the history once more with the prompt calls' streaming
//!   `split`: the first call streams experts on the wide arm's layers (none
//!   fails by name), which the card runs from the ring beside its stacks,
//!   every ring pass's union places read back are the stack's row with the
//!   ring's experts at `n_card` plus their slot, and the calls' live sets,
//!   ends and logits hold as `admit`'s. FAIL-first, `split`: the union one
//!   card place short (the combine's fault names the streamed expert past
//!   it), the ring's first expert left out of the union places (the union
//!   read back names its slots; the logits band does not see one expert),
//!   and the combine over the stack's places and card count (the streamed
//!   experts summed by neither side: the logits read it).
//!   FAIL-first, `flips`: the maps kept as the load built them
//!   (`PlacedSide::refresh_maps` returning at once) reads the map line red
//!   (the first prompt's maps name the spares the machine emptied at its
//!   start, the second's miss the admitted experts). The clause opens the
//!   file `ref_model_path` names and is a self-consistency clause.
//!   `--residency-only` runs (e) alone.
//!
//! (m) the memory guard, `--memguard-only` (no model loaded): the census
//!   free reading, the quiet-card whole decision, the held-card refusal by
//!   name and the placed decision, and the live reading once the hold drops
//!   — `gate_qwen3moe_e2e`'s (m) is the clause's full form.
//!
//! Tap maps (ik → ours). Delta layer: `qkv_mixed` → the q·k·v projection;
//! `z` → the gate projection; `beta_in`, `alpha` → β's and α's raw
//! projections; `g_in` → `ln` of our decay (`exp(g)`); the v channels of
//! `conv_output_silu` → the conv's v channels; `q_fused`, `k_fused` (the
//! L2 norm, a PERMUTE's src0 where the set names the permute) → the conv's
//! normed q (over `Q_SCALE`) and k; `attn_output` → the delta output;
//! `new_state` (`[k][v]` per head) → our state (`[v][k]`), transposed;
//! `new_conv_states_cont` (`[C][3]`) → the ring's last three positions;
//! `attn_out_norm` → the gated norm's output; `linear_attn_out` → the
//! residual update. Attention layer: `Qaux` → the `[q | gate]` rows;
//! `Qcur_normed` and `Qcur_roped` → our queries past and inside the 64
//! turned values; `Kcur_*` likewise; `Vcur`; `fa` → the flash output;
//! `qkv_gated` → our flash output times the sigmoid of our gate (a host
//! product: the kernel folds it into the quantizer); `attn_out` → the
//! residual update. MoE: `ffn_moe_logits` → the router's first 256 logits;
//! `ffn_moe_topk` → the first eight slots' ids; `ffn_moe_weights_norm` →
//! their weights, matched by id; `shared_expert_gate_sigmoid` → the ninth
//! slot's weight; `l_out` → the layer output.
//!
//! Known differences, named, not banded away: ik clamps nothing on these
//! sets (q35oracle Q1); layer 37's decay underflows to 0 at one site in 960,
//! and ours must underflow at the same sites; ik combines `(routed + resid)
//! + shexp_gated`, ours `((Σ8 w·d) + w8·d_sh) + resid` — one or two f32
//! roundings of the update, some 1e-7 of it, five orders under the band of
//! `l_out`, which is the only tap past that add.
//!
//! The chain runs the tensor-core flash pass, the engine's.
//!
//! The text path: the gate opens the file `ref_model_path` names (the text recipe's profile names MiniCPM's). In the real tier, on a qwen35moe file that is not
//! [`refset::arch::qwen35moe::MODEL`] (MiniCPM-V 4.7's text model, profile `minicpmv47`) no ik set exists, so
//! `text_run` runs the clauses that read none — (s), (p) on [`TEXT_TOKENS`], (p′), (w′), (r), (k), (n), (v) — and a
//! greedy cross-check (g): 32 greedy ids of three prompts against ik's on a header-corrected copy, equal, or
//! differing only at near ties ([`GREEDY_MARGIN`]). Every clause that reads a set prints `SKIPPED (<clause>)`,
//! and the verdict line counts them. `just gate-gpu-minicpm-text`.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate plans and opens a qwen35moe file only; the other half serves the CLI and the seat"
)]
mod q3place;

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the argmax, the tap reader and the RMS-normed rows; the rest of the runner serves the GLM and Qwen3.8 gates"
)]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/flash_grid.rs"]
mod flash_grid;

#[cfg(feature = "gpu")]
#[path = "shared/q35_clauses.rs"]
mod q35_clauses;

#[cfg(feature = "gpu")]
#[path = "shared/residency_clauses.rs"]
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; this one takes the residency, the seed lists and the c1 and c7 verdicts"
)]
mod residency_clauses;

#[cfg(feature = "gpu")]
mod gate {
    use super::e2e::{argmax, normed, tap};
    use super::flash_grid::flash_grids;
    use super::q3place::{self, PlaceQ3};
    use super::q35_clauses::{
        self as q35, Geometry35, PathRun, Slots35, decode_run, fresh, lcg_ids, refed_after_cut,
        same_run, same_store, stores, uncut_call,
    };
    use super::residency_clauses;
    use app::Session;
    use bloomery_gpu::NodeInfo;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::arch::qwen3moe::{
        Body35, Delta35Run, Gqa35Run, KvQ8, LayerKind35, Mixer35Run, Open35, PrefillPath,
        PrefillStep, Qwen35moeModel, StoreHost, WIDE_FROM, WideRoute,
    };
    use bloomery_gpu::gemm::{GEMM_BN, GemmTile};
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{CallPick, CallReport, PassReport, Residency};
    use bloomery_gpu::host::xstream::{Costs, XLayer, XMode};
    use bloomery_gpu::hybrid::{HOST, SlotMap};
    use bloomery_gpu::linear::{Q_SCALE, RING_ROWS, expf_ik};
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::q36_fixture::{Shape, Site};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GREEDY_MARGIN, GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed,
        data_dir, ik_q8_2, q8_1_dequant, record, ref_ints, ref_model_path, ref_tensor_logical_in,
        split_f32, topk_ids_logical_within, verdict, widened_f16_rows_in,
    };
    use bloomery_levers::{CARD_BUDGET, HostCfg};
    use cuda_core::sys;
    use gguf::Split;
    use gguf::quant::{GgmlType, half_to_f32};
    use model::arch::models::Mixer;
    use model::arch::models::shape::{AttnShape, MoeShape};
    use model::arch::qwen35moe::hparams::Kind;
    use model::arch::qwen35moe::names;
    use model::placement::PlanLevers;
    use model::placement::accepts::{self, LegSplit};
    use refset::arch::qwen35moe::fixture as fx;
    use refset::arch::qwen35moe::{BATCH, D1K, IK, MODEL, STEP4};
    use refset::flip::{RouteFlip, held_rows, rel_base, rel_to, router_tie, worse};
    use runtime::{Out, Target, Want};
    use std::time::Instant;
    use tokenizer::Tokenizer;

    /// Cache rows: the 1,024-token set's step at position 1,024, with room.
    const CTX: usize = 1088;

    /// The ubatch size the gate opens with and every clause but (w) runs at:
    /// the step-1,024 set's prompt as one ubatch, and an arena of 1,024 rows
    /// (about 0.36 GB at some 350 KB a row, derived from the arena's
    /// buffers), which fits a gate lane on either card.
    const U_GATE: usize = 1024;

    /// (w)'s other ubatch sizes, and where its two-call run cuts the prompt.
    const W_SIZES: [usize; 3] = [512, 1000, 100];
    const W_CUT: usize = 24;

    /// A placed load's prompt unit: the host union's columns
    /// (`model::moe::UNION_MAX_COLS`), each host expert read once a unit.
    const PLACED_UBATCH: usize = model::moe::UNION_MAX_COLS;

    /// PIN(2026-09-27): (u)'s band on layer 0's recurrent state and conv ring
    /// against the pass run's (`‖ubatch − pass‖ / ‖pass‖`). Layer 0's input is
    /// the same bits on both paths (the embedding rows; `rms_norm` then the
    /// GEMM quantizer write the bytes `norm_quant` writes, gate_qwen3moe's
    /// premise for the same kernels), so only its four projections differ,
    /// each within its sum's rounding of the exact dot of the same codes:
    /// about 1.2e-6 of `Σ |terms|` at K = 2,048, some four times a value, so
    /// Δ ≤ 1e-5 of a row (gate_qwen3moe's GEMM_L0_REL note). The ring holds
    /// the q·k·v projection's rows (Δ). The state is f32 throughout — no f16
    /// step turns Δ into an ulp, as the K/V rows' band had to — and each
    /// token's update is a product of its normed k, its v, β and the decay,
    /// each carrying Δ, under a map that does not expand (unit k, β and the
    /// decay at most 1); over 1,024 tokens whose updates partly cancel the
    /// sum's relative error is a few Δ. Predicted 1e-6 to 3e-5; a wiring
    /// fault in layer 0 (β read from α's weights, a token's position off)
    /// reads an error of order one.
    const U_L0_REL: f64 = 1e-4;

    /// PIN(2026-10-15): (u)'s layer-0 band when layer 0's delta projections
    /// are Q8_0. [`U_L0_REL`]'s premise,
    /// the same codes on both arms, does not hold there: the gemv arm reads
    /// the f32 rows (`site::gemv`), the wide arm the rows quantized to q8
    /// blocks of 32 (`site::gemm`), so the runs differ by that rounding, not
    /// by sum order [derived: a value's error is uniform within ±amax/254 of
    /// its block; at amax ≈ 2.2 · rms over 32 values, 2.2 / 254 / √3 ≈ 5e-3
    /// of a row, which the projections and the state carry; measured on the
    /// unjoined fixture, state 5.157e-3 and ring 3.539e-3]. The band is twice
    /// that; a wiring fault still reads order one.
    const U_L0_REL_Q32: f64 = 1e-2;

    /// PIN(2026-09-27): (d)'s bound on the step at position 1,024 free-running
    /// from each run's own state — each layer's output distance from ik's and
    /// the logits' — the ubatch run's over the pass run's. Derivation: past
    /// layer 0 the two runs' first difference (Δ, [`U_L0_REL`]) grows at every
    /// q8_1 re-quantization by code flips until the ubatch run is another q8_1
    /// realization of the same rule (gate_qwen3moe's GEMM_SPREAD_RATIO note),
    /// statistically the pass run's twin against ik; one step from each state
    /// reads about 1. A wiring fault — another layer's weight, a token off by
    /// one, a stale route row — adds an error of order one to distances of
    /// 1e-2 to 0.3, a ratio above 3 on every layer it reaches.
    const STEP_RATIO: f64 = 1.5;

    /// PIN(2026-10-11): [`STEP_RATIO`] for the placed load's wide arm: the
    /// stores' [`PROMPT_RATIO`], 2.5. Derivation: the ratio is one step's output
    /// distance from ik after the ubatch run's own state over the pass run's,
    /// and each distance is one excursion of the state it started from, not an
    /// average (the whole-card load's per-layer ratios scatter about 1 on
    /// the log scale, the pass run's distance jumping at single layers). The
    /// placed pass run is the host's 8-bit rule, another realization of the
    /// same rule, so its denominator has its own excursions. The bound is
    /// [`PROMPT_RATIO`]'s: √(1 + (d_wp/d_pi)²) with the two runs' own distance
    /// up to 2.3 d_pi is 2.5. A wiring fault adds an error of order one to
    /// distances of 1e-2 to 0.3, a ratio above 3 on every layer it reaches
    /// ([`STEP_RATIO`]'s note). The last logits keep [`STEP_RATIO`]: they are
    /// the one tensor all layers' excursions average into.
    const PLACED_STEP_RATIO: f64 = 2.5;

    /// PIN(2026-09-28): (d)'s bound on a layer's store distance from ik (a
    /// recurrent state and the conv ring's rows ik keeps, or the K and V rows,
    /// the larger of the two), the ubatch run's over the pass run's. Raised from
    /// 1.5 [잠정 — 백로그]: 1.5 assumed the two runs' own distance d_wp is at
    /// most the pass run's distance from ik d_pi, so the ratio is at most
    /// √(1 + (d_wp/d_pi)²) ≤ √2; over 1,024 positions the recurrent stores
    /// integrate the flips and d_wp/d_pi reads 0.52, 1.03, 1.19, 1.41 at layers
    /// 8, 16, 24, 32, each measured ratio (0.97, 1.14, 1.30, 1.54) at or under
    /// that bound, the worst 1.72 at layer 29. 2.5 holds d_wp up to 2.3 d_pi;
    /// the wiring faults above still read past 3. A band derived from the flip
    /// amplification replaces this pin.
    const PROMPT_RATIO: f64 = 2.5;

    /// The file's widths, as the header states them (q35design §1): what the
    /// derivations below are written against. The fixture keeps the real
    /// widths, and `init` holds the file's embedding width to [`HIDDEN`] by
    /// name.
    const HIDDEN: usize = 2048;
    /// The real file's layer counts and Q6_K sites, as its header states them
    /// (q35design §1): the literals the shape the header gives ([`Cfg`]) is
    /// witnessed against.
    const REAL_N_LAYER: usize = 40;
    const REAL_N_ATTN: usize = 10;
    const REAL_N_DELTA: usize = 30;
    /// Delta layers whose `attn_qkv` is Q6_K, and attention layers whose
    /// `attn_v` is (3, 7, 19, 31, 35, 39).
    const REAL_Q6_QKV: usize = 14;
    const REAL_Q6_V: usize = 6;
    const C: usize = 8192;
    const N_V: usize = 32;
    const HEAD_V: usize = 128;
    const Q_ROWS: usize = 8192;
    const KV_ROW: usize = 512;
    const N_HEAD: usize = 16;
    const N_KV: usize = 2;
    const HEAD: usize = 256;
    const ROT: usize = 64;

    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: the embedding row; each delta layer's 8 mixer
    /// launches (norm+quant, the two projection launches — a Q6_K `attn_qkv`
    /// in its own gemv, `attn_gate`·β·α in one — conv, delta, gated norm,
    /// q8_1, `ssm_out` with the residual) and 5 FFN launches (the norm with
    /// the gated router, gate·up over the joined stacks, q8_1 of the nine
    /// slots, the down `_sel`, the combine); each attention layer's 7 mixer
    /// launches (norm+quant, q·k·v, rope-256, flash segment pass and merge,
    /// the gated q8_1, the output projection with the residual), one more
    /// on the 6 layers whose `attn_v` is Q6_K, and 5 FFN; then the head's
    /// three: 1 + 30·13 + 10·12 + 6 + 3.
    const REAL_NODES_DECODE: usize = 520;

    /// PIN(2026-10-05): a captured pass of `m >= 2` rows without its heads,
    /// derived the same way at more than one row: each delta layer 8 + 5
    /// (the norm with the gated router in one launch, as at one row), one
    /// more on the 14 whose `attn_qkv` is Q6_K (its token-major copy); each
    /// attention layer 7 + 5, two more on the 6 Q6_K `attn_v` layers (gemv
    /// and copy): 1 + 30·13 + 14 + 10·12 + 12. Each row's head adds a copy
    /// of the row's residual and three launches.
    const REAL_NODES_PASS: usize = 537;

    /// PIN(2026-11-01): the launches one more busy slot adds to a captured
    /// pass of several slots (`step_slots`) over f16 planes, derived from the
    /// launches that bind one sequence's stores or record and run once a
    /// sequence: its embedding and each attention layer's flash segment pass
    /// and merge: 1 + 10·2. The delta layers' conv and delta step and the
    /// attention layers' rope with the cache append run once over every
    /// slot's rows (91 before, 1 + 30·2 + 10·3).
    const REAL_NODES_SLOT: usize = 21;

    /// The train-gate file of the unjoined shared expert: unsloth's
    /// `UD-Q4_K_M`, every layer's shared expert q8_0 beside q4_K gate and up
    /// stacks and a q5_K down (37 layers) or a q6_K one (34, 38, 39); every
    /// mixer site q8_0 but β and α, F32.
    const UNJOINED: &str = "/models/qwen36-hf/unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";

    /// PIN(2026-10-10): the unjoined file's captured decode step, derived
    /// from its header before the path ran: the embedding row; each delta
    /// layer's 10 mixer launches (norm+quant, the q8_0 `attn_qkv` and
    /// `attn_gate` and the F32 β and α each its own gemv, conv, delta, gated
    /// norm, the q8_0 `ssm_out` into `normed` and the residual add) and each
    /// attention layer's 10 (norm+quant, q, k and v each its own gemv, rope,
    /// flash segment pass and merge, the f32 out gate, the q8_0
    /// `attn_output` and the add); each layer's 11 FFN launches with its
    /// shared expert kept apart (`norm_quant` and the router on the f32 rows
    /// its q8_0 gate and up read, the card list, the routed gate·up, the
    /// routed slots' quantizer, the routed down `_sel`, the shared gate and
    /// up, the SwiGLU, the shared down, the host-sum combine); then the
    /// head's three: 1 + 40·(10 + 11) + 3. No site here is row-major past
    /// one row (q8_0 and F32 gemvs are token-major), so a pass of `m >= 2`
    /// rows holds the same launches a row: 1 + 40·21.
    const REAL_NODES_DECODE_UD: usize = 844;
    const REAL_NODES_PASS_UD: usize = 841;

    /// PIN(2026-10-10): (j)'s band on the FFN update against the host's f64
    /// FFN (`‖ours − host‖ / ‖host‖`, the update `l_out − ffn_inp`). The
    /// card's routed launches read the normed rows and the SwiGLU rows as
    /// q8_1 (8-bit codes a 32-value block, a step of the block's largest
    /// value over 127): each reading moves a projection by some 0.5 % of its
    /// size, two readings in the routed path, the SwiGLU between them
    /// adding the gate's and the up's; the q8_0 shared expert reads f32 rows
    /// on the gemv arm and q8 blocks of 32 on the wide arm (an 8-bit reading
    /// of each of its three projections' inputs). Predicted 0.5 % to 2 %. The
    /// shared expert at weight 1 in place of its sigmoid, a down read at
    /// another type, or a slot's rows from another expert move the update by
    /// an order of its size.
    const UNJOINED_FFN_BAND: f64 = 5e-2;

    /// The gate's one reading of what the run is made from: the model file
    /// (`ref_model_path`, a whole fixture under the fixture tier), the host
    /// load config and the plan's levers (the budget the header records in the
    /// fixture tier, [`tier::plan_levers`]), the embedding width and the layer
    /// kinds and site types the header names ([`Shape`]). Set once, first
    /// thing in `run`, before any load.
    struct Cfg {
        tier: Tier,
        path: std::path::PathBuf,
        host: HostCfg,
        plan_levers: PlanLevers,
        shape: Shape,
        n_embd: usize,
    }

    static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();

    fn cfg() -> &'static Cfg {
        CFG.get().expect("init runs before any clause")
    }

    /// The layer kinds and site types the header names ([`Shape`]): never read
    /// off the loaded body, which the structure clause holds against them.
    fn shape() -> &'static Shape {
        &cfg().shape
    }

    fn n_layer() -> usize {
        shape().n_layer()
    }

    fn n_attn() -> usize {
        shape().n_attn()
    }

    fn n_delta() -> usize {
        shape().n_delta()
    }

    fn nodes_decode() -> usize {
        shape().nodes_decode()
    }

    fn nodes_pass() -> usize {
        shape().nodes_pass()
    }

    /// The nodes one more busy slot adds to a captured pass: defined for the
    /// joined layers of this gate's file only ([`Shape::nodes_slot`]).
    fn nodes_slot() -> Result<usize, GateError> {
        shape().nodes_slot()
    }

    /// The model file this gate runs on, opened.
    fn open_model() -> Result<Split, GateError> {
        let path = &cfg().path;
        Ok(Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?)
    }

    /// The path of the file the apart-layout clauses (j) and (o8) open:
    /// [`UNJOINED`] in the real tier, the file the tier names in the fixture
    /// tier (a fixture variant carries that layout on the fixture's layers; the
    /// base fixture holds none, and `tier::layout_clause` leaves them there).
    fn unjoined_path() -> &'static std::path::Path {
        match cfg().tier {
            Tier::Real => std::path::Path::new(UNJOINED),
            Tier::Fixture => cfg().path.as_path(),
        }
    }

    /// One of ik's three sets of the qwen35moe family.
    #[derive(Clone, Copy)]
    enum Set {
        /// The five-token batch set.
        Batch,
        /// The step after a 4-token prefill.
        Step4,
        /// The step after a 1,024-token prefill.
        D1k,
    }

    impl Set {
        /// The set's name in the tier's family: the real family's in the real
        /// tier, the fixture family's in the fixture tier.
        fn name(self) -> &'static str {
            let (real, fixture) = match self {
                Set::Batch => (BATCH, fx::BATCH),
                Set::Step4 => (STEP4, fx::STEP4),
                Set::D1k => (D1K, fx::D1K),
            };
            match cfg().tier {
                Tier::Real => real,
                Tier::Fixture => fixture,
            }
        }

        /// What a clause line calls the set, in either tier.
        fn what(self) -> &'static str {
            match self {
                Set::Batch => "the batch set",
                Set::Step4 => "the step-4 set",
                Set::D1k => "the step-1,024 set",
            }
        }
    }

    /// ik's set `set` of this tier, opened for the clause `what`: the real
    /// family's in the real tier; in the fixture tier the fixture family's,
    /// through `tier::fixture_set`, which ends the gate by name for a set that
    /// is missing, stale, foreign or unfinished (the variant's own sets under
    /// `BLOOMERY_FIXTURE_VARIANT`).
    fn ik_set(what: &str, set: Set) -> Result<RefManifest, GateError> {
        match cfg().tier {
            Tier::Real => Ok(RefManifest::open(&data_dir().join(set.name()), &IK)?),
            Tier::Fixture => {
                let path = tier::fixture_set(what, &fx::IK, set.name())?;
                Ok(RefManifest::open(&path, &fx::IK)?)
            }
        }
    }

    /// The clause `what` that compares with ik's output on `set`:
    /// `Tag::Oracle` on the real family's set in the real tier,
    /// `Tag::FixtureOracle` on the fixture family's in the fixture tier. A tier
    /// decides both and prints the other's line (`fixture-only(oracle)`,
    /// `deferred(real)`), so the tally counts the clause on either side; the
    /// set opened is the one of the tier that runs it.
    fn ik_oracle(what: &str, set: Set) -> Result<RefManifest, GateError> {
        tier::run_clause(&format!("{what} on ik's real set"), Tag::Oracle)?;
        let fixture = format!("{what} on ik's fixture set");
        tier::run_clause(&fixture, Tag::FixtureOracle)?;
        ik_set(&fixture, set)
    }

    /// The shape of `split` from its header: the layer kinds the file's
    /// description gives (cross-checked by its reader with the tensors and the
    /// interval) and the type of each site the launch rules read, from the
    /// split's tensor table. Returns the embedding width with it.
    fn header_shape(split: &Split) -> Result<(usize, Shape), GateError> {
        let spec = model::arch::spec(split)?.spec;
        let kinds = spec
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| match &layer.mixer {
                Mixer::DeltaRule(_) => Ok(Kind::DeltaRule),
                Mixer::Gqa(_) => Ok(Kind::Attention),
                other => Err(GateError::from(format!(
                    "layer {l} mixes with {}, which has no launch rules here",
                    other.what()
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = Shape::from_header(&kinds, |l, site| {
            let name = match site {
                Site::AttnQkv => names::attn_qkv(l),
                Site::AttnGate => names::attn_gate(l),
                Site::SsmAlpha => names::ssm_alpha(l),
                Site::SsmBeta => names::ssm_beta(l),
                Site::SsmOut => names::ssm_out(l),
                Site::AttnQ => names::attn_q(l),
                Site::AttnK => names::attn_k(l),
                Site::AttnV => names::attn_v(l),
                Site::AttnOutput => names::attn_output(l),
                Site::GateExps => names::ffn_gate_exps(l),
                Site::UpExps => names::ffn_up_exps(l),
                Site::DownExps => names::ffn_down_exps(l),
                Site::GateShexp => names::ffn_gate_shexp(l),
                Site::UpShexp => names::ffn_up_shexp(l),
                Site::DownShexp => names::ffn_down_shexp(l),
            };
            split
                .find(&name)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| GateError::from(format!("the file holds no {name}")))
        })?;
        Ok((spec.hidden as usize, shape))
    }

    /// Read the file's header once: the path the tier names, the levers, the
    /// plan's budget and the layer kinds; print each derived value beside the
    /// real file's literal it replaced ([`witnesses`]).
    fn init() -> Result<bool, GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let (n_embd, shape) = header_shape(&file)?;
        if n_embd != HIDDEN {
            return Err(format!(
                "{} has an embedding width of {n_embd}, the derivations here are written for {HIDDEN}",
                path.display(),
            )
            .into());
        }
        let plan_levers = tier::plan_levers(&file, &levers, 0)?;
        let cfg = Cfg {
            tier: Tier::from_env()?,
            path,
            host: levers.host(),
            plan_levers,
            shape,
            n_embd,
        };
        if CFG.set(cfg).is_err() {
            return Err("the gate's configuration was read twice".into());
        }
        if text_file()? {
            return Ok(true);
        }
        witnesses()
    }

    /// The real tier on a qwen35moe file that is not [`MODEL`], ik's: no set
    /// was dumped from it, so the gate runs [`text_run`] and holds no move
    /// proof against the oracle file's literals.
    fn text_file() -> Result<bool, GateError> {
        let real = tier::Tier::from_env()? == tier::Tier::Real;
        Ok(real && cfg().path != std::path::Path::new(MODEL))
    }

    /// The move proof: each value the header gives, beside the literal the
    /// real file's gate carried. In the real tier they are equal; the fixture
    /// tier prints the fixture's own.
    fn witnesses() -> Result<bool, GateError> {
        let model = cfg().path.to_string_lossy().into_owned();
        let mut ok = tier::witness("model file", model, MODEL.to_string());
        ok &= tier::witness("HIDDEN", cfg().n_embd, HIDDEN);
        ok &= tier::witness("N_LAYER", n_layer(), REAL_N_LAYER);
        ok &= tier::witness("N_ATTN", n_attn(), REAL_N_ATTN);
        ok &= tier::witness("N_DELTA", n_delta(), REAL_N_DELTA);
        ok &= tier::witness("Q6_QKV", shape().q6_qkv(), REAL_Q6_QKV);
        ok &= tier::witness("Q6_V", shape().q6_v(), REAL_Q6_V);
        ok &= tier::witness("NODES_DECODE", nodes_decode(), REAL_NODES_DECODE);
        ok &= tier::witness("NODES_PASS", nodes_pass(), REAL_NODES_PASS);
        if shape().apart().contains(&true) {
            println!(
                "derived NODES_SLOT: none, a layer keeps its shared expert apart (the slots \
                 clause (n) reads joined layers)"
            );
        } else {
            ok &= tier::witness("NODES_SLOT", nodes_slot()?, REAL_NODES_SLOT);
        }
        ok &= tier::witness("PLACED_BUDGET", placed_budget(), REAL_PLACED_BUDGET);
        Ok(ok)
    }

    /// The shape of [`unjoined_path`]'s file from its header.
    fn unjoined_shape() -> Result<Shape, GateError> {
        let path = unjoined_path();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        Ok(header_shape(&file)?.1)
    }

    /// The apart-layout clause `name`, which reads the layers of [`unjoined_path`]'s
    /// file that keep their shared expert apart (`tier::layout_clause`): the
    /// file's shape and its move proof beside the literals the clauses
    /// carried, its layer count, and with `counts` the node counts of its
    /// captured decode step and pass. `None`: the fixture tier leaves the
    /// clause, the file holding no such layer.
    fn apart_clause(name: &str, counts: bool) -> Result<Option<(Shape, bool)>, GateError> {
        let shape = unjoined_shape()?;
        if !tier::layout_clause(name, shape.apart().contains(&true))? {
            return Ok(None);
        }
        let mut ok = tier::witness("N_LAYER (unjoined)", shape.n_layer(), REAL_N_LAYER);
        if counts {
            ok &= tier::witness(
                "NODES_DECODE_UD",
                shape.nodes_decode(),
                REAL_NODES_DECODE_UD,
            );
            ok &= tier::witness("NODES_PASS_UD", shape.nodes_pass(), REAL_NODES_PASS_UD);
        }
        Ok(Some((shape, ok)))
    }

    /// PIN(2026-09-27): the teacher-forced bound on a tap's error ratio — its
    /// relative error over the relative distance between the two sides'
    /// 8-bit activations of the input that reaches it (ours q8_1 per 128
    /// values, ik's q8_2 per 32: [`quant_gap_two_sided`]; two inputs in quadrature).
    /// Derivation, the qwen3moe gate's: to first order a linear map carries
    /// its input's relative perturbation unchanged, so a projection reads
    /// about 1; the kernels are held to their host rules bit for bit or
    /// within γ(n) of them (gate_linear, gate_qwen35moe_attn,
    /// gate_qwen35moe_moe), terms some 1e-6 of the ~1e-2 gap, so the gap is
    /// the whole prediction. Predicted per class: the projections, β/α, z,
    /// the conv's v channels and the normed q/k about 1 (median), at most 3;
    /// the delta output and the state, a sum over the call's tokens of
    /// per-token errors the decay and β (both at most 1) do not amplify,
    /// at most √5 ≈ 2.2 times a token's, so at most 5; the flash output at
    /// most 3 (the softmax sharpens a score error, qwen3moe measured 2.8);
    /// the FFN at most 6 (qwen3moe measured 6.1 on a layer with one dominant
    /// channel). A wiring fault — another layer's weight, a head or a token
    /// off by one, the state transposed — reads an error of order one over a
    /// gap of about 2e-2, a ratio above 40.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's. Derivation, the qwen3moe gate's: the forced
    /// arm's per-layer errors, up to about 4e-2 of a layer's update, whose
    /// norm is of the residual's order, added in quadrature over 40 layers,
    /// independent: √40 · 4e-2 ≈ 0.25. The step sets' one step starts from
    /// ik's own state, so the same composition bounds it.
    const FREE_BAND: f64 = 0.26;

    /// The layer stores' bytes at [`CTX`] rows, derived from the header:
    /// each attention layer's K and V planes, `2 · n_kv · ctx · 256` f16;
    /// each delta layer's state, `32 · 128 · 128` f32 (one lane), and conv
    /// ring, `RING_ROWS · 8192` f32.
    fn store_bytes() -> usize {
        geometry().store_bytes(CTX)
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32). Not `e2e::quant_gap`, which is ik's q8_2
    /// against the f32 `x` our kernels read.
    fn quant_gap_two_sided(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel_base(&o, &i, |j| f64::from(x[j]))
    }

    /// The node a set dumps as `name`, or when that row only relabels
    /// another (a permute, reshape, view or copy), the node it relabels,
    /// through its `src0`, with that row's `ne`: the step sets name the
    /// permute of the L2 norm's output and leave the output itself unnamed,
    /// the batch set names the output.
    fn tap_or_src(man: &RefManifest, name: &str) -> Result<(Vec<f32>, [u64; 4]), GateError> {
        const RELABEL: [&str; 5] = ["PERMUTE", "RESHAPE", "VIEW", "CONT", "TRANSPOSE"];
        let (mut at, mut row) = man.tensor_at(name, 0)?;
        for _ in 0..4 {
            if !RELABEL.contains(&row.op.as_str()) {
                return Ok((ref_tensor_logical_in(&man.dir, row)?, row.ne));
            }
            (at, row) = man.last_before(at, row.src0.as_deref())?;
        }
        Err(format!("{name}: four relabelling rows and no node under them").into())
    }

    /// The last `n` rows of `k` values of `v`, the rows a tap kept (the
    /// last layer keeps only the output token's rows past its attention):
    /// `None` when `v` holds fewer.
    fn last_rows(v: &[f32], k: usize, n: usize) -> Option<&[f32]> {
        v.len().checked_sub(n * k).map(|s| &v[s..])
    }

    /// Worst ratio and its tap, over a layer's taps. Not `e2e::Worst`, whose
    /// lines also print the gap (this gate's printed lines do not).
    #[derive(Default)]
    struct Worst {
        ratio: f64,
        tap: String,
        lines: Vec<String>,
    }

    impl Worst {
        fn add(&mut self, tap: &str, e: f64, gap: f64) {
            let r = worse(0.0, e / gap.max(f64::MIN_POSITIVE));
            self.lines.push(format!("{tap} rel={e:.3e} ratio={r:.2}"));
            if self.tap.is_empty() || r > self.ratio {
                self.ratio = r;
                self.tap = tap.to_string();
            }
        }
    }

    fn open(ctx: usize, kv: KvQ8) -> Result<Qwen35moeModel, GateError> {
        open_at(&cfg().path, ctx, kv)
    }

    fn open_at(
        path: impl AsRef<std::path::Path>,
        ctx: usize,
        kv: KvQ8,
    ) -> Result<Qwen35moeModel, GateError> {
        let path = path.as_ref();
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!(
                "{} is {:?}, not qwen35moe",
                path.display(),
                file.architecture()
            )
            .into());
        }
        let t = Instant::now();
        let m = Qwen35moeModel::open(
            Gpu::new()?,
            file,
            Open35 {
                ctx,
                mma: true,
                ubatch: U_GATE,
                kv,
                media: false,
            },
        )?;
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={} ubatch={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            m.body("gate_qwen35moe_e2e")?.flash_mma(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    // ------------------------------------------------------ (s) structure

    /// The flash launches' grids in a captured step (module doc, (s) and
    /// (k)): the eight-head `_256` entries, one pack a key head, at one row,
    /// one segment pass and one merge an attention layer ([`flash_grids`]).
    fn flash_grid_ok(nodes: &[NodeInfo]) -> Result<(bool, String), GateError> {
        let names = ["gqa_flash_seg_mma_256", "gqa_flash_merge_256"];
        flash_grids(nodes, (1, N_KV, N_HEAD), n_attn(), names)
    }

    /// The layers `kept` marks, in order.
    fn apart_at(kept: &[bool]) -> Vec<usize> {
        (0..kept.len()).filter(|&l| kept[l]).collect()
    }

    fn structure(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        tier::sc("(s) structure: layer kinds, stores, node counts and flash grids")?;
        let (want_attn, want_layers) = (n_attn(), n_layer());
        let (want_decode, want_pass) = (nodes_decode(), nodes_pass());
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let n_attn = kinds
            .iter()
            .filter(|k| **k == LayerKind35::Attention)
            .count();
        let (launch_1, launch_m) = (body.pass_launches(1), body.pass_launches(2));
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let (apart, want_apart) = (body.shared_apart(), shape().apart());
        let mut ok = n_attn == want_attn
            && kinds.len() == want_layers
            && stores == want_stores
            && apart == want_apart;
        let attn_at: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l] == LayerKind35::Attention)
            .collect();
        println!(
            "structure layers={} attention at {attn_at:?} ({n_attn}, want {want_attn}); store bytes \
             {stores} (want {want_stores}, derived); shared experts kept apart at layers {:?} (want \
             {:?}, the header's) {}",
            kinds.len(),
            apart_at(&apart),
            apart_at(&want_apart),
            verdict(ok)
        );
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memcpy = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY;
        let nodes = m.capture_step()?;
        let ([k, c], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memcpy]);
        let pass = nodes == want_decode && k == nodes && launch_1 + 3 == nodes;
        println!(
            "structure decode graph_nodes={nodes} (want {want_decode}; the body counts {} + 3) \
             kernel={k} memcpy={c} other={other} {}",
            launch_1,
            verdict(pass)
        );
        ok &= pass;
        let (flash_ok, got) = flash_grid_ok(&m.step_graph_nodes()?)?;
        println!("structure flash grid {got} {}", verdict(flash_ok));
        ok &= flash_ok;
        for (rows, nodes, list) in [
            (5usize, m.capture_rows::<5>()?, m.rows_graph_nodes::<5>()?),
            (8, m.capture_rows::<8>()?, m.rows_graph_nodes::<8>()?),
        ] {
            let ([k, c], other) = count_kinds(&list, [kernel, memcpy]);
            let want = want_pass + 4 * rows;
            let pass =
                nodes == want && c == rows && k == nodes - rows && launch_m + 4 * rows == nodes;
            println!(
                "structure pass m={rows} graph_nodes={nodes} (want {want}; the body counts {} + 4·{rows}) \
                 kernel={k} memcpy={c} (want {rows}) other={other} {}",
                launch_m,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// The file's attention and delta shapes, from its spec, against the
    /// constants the derivations above are written with: a file of another
    /// shape fails here by name rather than under a slicing written for this
    /// one.
    fn file_shape() -> Result<bool, GateError> {
        tier::sc("shape: the file's attention and delta shapes against the derivations'")?;
        let split = open_model()?;
        let spec = model::arch::qwen35moe::spec::read(&split)
            .map_err(|e| format!("{}: {e}", cfg().path.display()))?
            .spec;
        let c = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let (mut ok, mut seen) = (true, Vec::new());
        for (l, layer) in spec.layers.iter().enumerate() {
            let (got, want) = match &layer.mixer {
                Mixer::Gqa(g) => {
                    let a = AttnShape::of(g);
                    let got = [a.n_head, a.n_kv, a.head, g.rope.dims];
                    (("gqa", got), [c(N_HEAD), c(N_KV), c(HEAD), c(ROT)])
                }
                Mixer::DeltaRule(d) => {
                    let got = [d.v_heads, d.d, (2 * d.k_heads + d.v_heads) * d.d, 0];
                    (("delta", got), [c(N_V), c(HEAD_V), c(C), 0])
                }
                Mixer::Latent(_) => (("latent", [0; 4]), [u32::MAX; 4]),
                Mixer::ShortConv(_) => (("short-conv", [0; 4]), [u32::MAX; 4]),
                Mixer::Mamba2(_) | Mixer::None => (("mamba2", [0; 4]), [u32::MAX; 4]),
            };
            let pass = got.1 == want;
            ok &= pass;
            if !pass || !seen.contains(&got) {
                println!(
                    "shape layer={l} {} {:?} (the derivations' {want:?}) {}",
                    got.0,
                    got.1,
                    verdict(pass)
                );
                seen.push(got);
            }
        }
        let rows_ok = Q_ROWS == 2 * N_HEAD * HEAD && KV_ROW == N_KV * HEAD;
        ok &= rows_ok;
        println!(
            "shape q rows {Q_ROWS} = 2·{N_HEAD}·{HEAD}, kv row {KV_ROW} = {N_KV}·{HEAD} {}",
            verdict(rows_ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (p) one layer body

    /// (p) (module doc): `q35_clauses::paths` under the self-consistency tag.
    fn paths(m: &mut Qwen35moeModel, toks: &[u32]) -> Result<(bool, PathRun), GateError> {
        tier::sc("(p) one layer body: pass = decode, eager = graph, reset clears")?;
        q35::paths(m, toks)
    }

    // --------------------------------------------------- (c) free-running

    /// A free-running clause's argmax verdict and the text its line prints for
    /// it, `ours` being our last position's argmax over `ik_row` (ik's logits
    /// there) and `head_in` the run's head input's own relative distance from
    /// ik's (its last layer output's). The real tier holds `ours` equal to ik's
    /// argmax. The fixture tier holds [`flip::head_tie`] at `head_in`: ik's own
    /// logit at our argmax within [`flip::head_cap`] of its top, which reads
    /// the same as equality where ik's row has one clear top. A `head_in` that
    /// is not finite is no tie (the layer clause is red on it as well).
    ///
    /// PIN(2026-10-13): the fixture tier's tie form. Derivation: the fixture's
    /// random weights leave ik's step-1,024 row near flat (rms 0.999, five
    /// logits within 0.123 of its top) and our row 0.123 of its norm off ik's,
    /// so a logit carries an error σ_e ≈ 0.123 and ik's logit at our pick lies
    /// 0.138 = 1.1 σ_e under its top: exact equality holds there with
    /// probability 0.3 to 0.5, a coin the weights threw. The real file's row
    /// has its top 1.25 = 13 σ_e clear (rms 2.45, distance 0.038), where
    /// equality is deterministic and stays the assertion. The cap is 6 ·
    /// `head_in` · rms of ik's row, three deviations each for our logit and
    /// ik's: 0.73 on the fixture's row. A fault upstream of the head moves
    /// `head_in` past [`FREE_BAND`], which the layer clause reddens first (and
    /// the cap stays under 1.6 rms); a fault in the head leaves `head_in` as
    /// it is and puts our pick far under ik's top, a gap of several caps.
    fn head_argmax(ik_row: &[f32], ours: u32, head_in: f64) -> (bool, String) {
        let ik_top = argmax(ik_row);
        match cfg().tier {
            Tier::Real => (
                ours == ik_top,
                format!("argmax ours={ours} ik={ik_top} strict (real tier)"),
            ),
            Tier::Fixture => {
                let (gap, cap) = (
                    flip::head_gap(ik_row, ours),
                    flip::head_cap(head_in, ik_row),
                );
                let tie = head_in.is_finite() && flip::head_tie(ik_row, ours, head_in);
                let cmp = if tie { "≤" } else { "past" };
                (
                    tie,
                    format!(
                        "argmax ours={ours} ik={ik_top} tie: gap {gap:.3} {cmp} cap {cap:.3} \
                         (fixture tier: head_tie at the run's head error {head_in:.4})"
                    ),
                )
            }
        }
    }

    /// A delta layer's free-running new-state term. The real tier is
    /// `state_rel / gap` (the two sides' 8-bit activations of ik's normed
    /// input). The fixture tier is `state_rel` over the larger of `gap` and
    /// `in_rel` (our layer input's own relative distance from ik's: 0 at
    /// layer 0, whose input is the exact embedding). The tier is `cfg().tier`,
    /// the same owner [`head_argmax`] reads. In the fixture tier a gap or an
    /// input distance that is not finite makes the ratio infinite.
    ///
    /// PIN(2026-10-13): the fixture tier's form. The fixture's layer 2 input
    /// is 1.48e-1 off, and the ratio over the gap alone read 14.26.
    /// Derivation: to first order the state carries its input's relative
    /// error through a linear update, so `state_rel` ≈ g · `in_rel` with a
    /// gain g of 0.73: the fixture's layer 0, whose input is exact and so
    /// carries the 8-bit gap alone, reads 3.94e-3 / 5.40e-3 = 0.73, and its
    /// layer 2 reads 1.08e-1 / 1.48e-1 = 0.73. A fixture layer whose input is
    /// off by more than the gap is held to that gain, not to the gap. The
    /// real tier keeps `state_rel / gap`. (t2) is the wiring catcher on ik's
    /// own input and is unchanged.
    fn state_ratio(state_rel: f64, gap: f64, in_rel: f64) -> f64 {
        match cfg().tier {
            Tier::Real => state_rel / gap.max(f64::MIN_POSITIVE),
            Tier::Fixture => {
                if !gap.is_finite() || !in_rel.is_finite() {
                    return f64::INFINITY;
                }
                state_rel / gap.max(in_rel).max(f64::MIN_POSITIVE)
            }
        }
    }

    /// The router decisions of a free-running chain, read by a probe: the
    /// chain's own layer outputs `rows` (`[layer][token · HIDDEN]`, from the
    /// layer taps) fed to a whole-card load's layers, each layer's pass over
    /// the five rows from a zero state on the chain's input to it (ik's
    /// embedding for layer 0, the chain's previous output above). The probe's
    /// mixer and router run the chain's kernels on the chain's own input, so
    /// its slots and logits are the chain's; a placed load differs from the
    /// probe in its experts' arithmetic alone.
    struct Routes {
        moe: Vec<MoeShape>,
        t_n: usize,
        /// Per layer, the router's slots `[t][top_k + 1]`.
        ids: Vec<Vec<u32>>,
        /// Per layer, the router's logits `[t][experts + 1]`.
        logits: Vec<Vec<f32>>,
        /// The probe's layer outputs against the chain's: worst distance,
        /// layers bit for bit, layers.
        agree: (f64, usize, usize),
    }

    fn route_probe(t_n: usize, rows: &[Vec<f32>]) -> Result<Routes, GateError> {
        let mut m = open(CTX, KvQ8::F16)?;
        m.set_mode(StepMode::Eager);
        fresh(&mut m)?;
        let man = ik_set("free: our router decisions on the chain's rows", Set::Batch)?;
        let embd = tap(&man, "inp_embd")?;
        let embd =
            last_rows(&embd, HIDDEN, t_n).ok_or("inp_embd holds fewer rows than the batch")?;
        let n = m.layers().len();
        let Gains { moe, .. } = Gains::read(n)?;
        let mut r = Routes {
            moe,
            t_n,
            ids: Vec::new(),
            logits: Vec::new(),
            agree: (0.0, 0, 0),
        };
        for l in 0..n {
            let x_in: &[f32] = if l == 0 { embd } else { &rows[l - 1] };
            let run = m.layer_rows(l, x_in, 0)?;
            let want = &rows[l];
            r.agree.0 = worse(r.agree.0, rel_to(&run.ffn.l_out, want));
            r.agree.1 += usize::from(
                run.ffn
                    .l_out
                    .iter()
                    .zip(want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            );
            r.agree.2 += 1;
            r.ids.push(run.ffn.ids);
            r.logits.push(run.ffn.logits);
        }
        Ok(r)
    }

    /// A chain's router sites against ik's, over the rows ik kept.
    struct RouteSites {
        sites: usize,
        flips: Vec<RouteFlip>,
        /// Sites whose ids are not [`flip::picks_its_top`] of the chain's own logits.
        inconsistent: Vec<(usize, usize)>,
    }

    fn route_sites(man: &RefManifest, c: &Routes) -> Result<RouteSites, GateError> {
        let mut out = RouteSites {
            sites: 0,
            flips: Vec::new(),
            inconsistent: Vec::new(),
        };
        for (l, shape) in c.moe.iter().enumerate() {
            let (n_expert, n_used) = (shape.experts as usize, shape.top_k as usize);
            let (slots, row) = (n_used + 1, n_expert + 1);
            let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
            let kept = logits.len() / n_expert;
            let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, trow, shape.experts)?;
            if kept == 0
                || kept > c.t_n
                || kept * n_expert != logits.len()
                || ids.len() != kept * n_used
            {
                return Err(format!(
                    "layer {l}: {} router logits and {} ids for {} rows",
                    logits.len(),
                    ids.len(),
                    c.t_n
                )
                .into());
            }
            for k in 0..kept {
                let t = c.t_n - kept + k;
                let ours_lg = &c.logits[l][t * row..t * row + n_expert];
                let ours = &c.ids[l][t * slots..t * slots + n_used];
                let theirs = &ids[k * n_used..(k + 1) * n_used];
                let lg = &logits[k * n_expert..(k + 1) * n_expert];
                out.sites += 1;
                let consistent = flip::picks_its_top(ours, ours_lg, n_used);
                if !consistent {
                    out.inconsistent.push((l, t));
                }
                if ours.iter().all(|&e| theirs.contains(&(e as i32))) {
                    continue;
                }
                let margin = flip::margin(lg, theirs) as f32;
                let err = ours_lg
                    .iter()
                    .zip(lg)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                out.flips.push(RouteFlip {
                    layer: l,
                    token: t,
                    ours: ours.to_vec(),
                    theirs: theirs.to_vec(),
                    margin,
                    err,
                    tie: router_tie(margin, err),
                });
            }
        }
        Ok(out)
    }

    /// PIN(2026-10-13): the fixture tier's free clause leaves unheld the rows a
    /// router near-tie flip carries ([`held_rows`]); three bounds keep it a bug
    /// catcher. [`router_tie`] is necessary for any flip the logits' error
    /// explains: for an exchanged pair (a ours only, b ik's only) ik's gap
    /// ik[b] − ik[a] is at least its margin, and at most |Δa| + |Δb| when
    /// ours[a] ≥ ours[b], which holds when our ids are the top set of our
    /// logits; so margin ≤ 2 · our largest error. That proviso is checked: a
    /// site whose ids are not the top set of its own logits is red, which
    /// keeps a lane swap red where the margin is small. Derivation of the
    /// bounds, from
    /// today's nine free runs (the batch set, 4 layers × 5 tokens, 16 rows and
    /// sites ik kept; the base fixture and the variants `unjoined` and
    /// `unjoinedq6`, each at the whole-card, the placed and the placed no-host
    /// load): the base fixture holds 11, 12 and 11 of 16 rows with 5, 4 and 5
    /// flips; both variants hold 4 of 16 (25 %: layer 0 of tokens 1 to 4) with
    /// 8 flips on all three loads, token 0's flip at layer 0 (ik's margin
    /// 1.77e-3 under our logits' error 2.66e-2, a coin the 8-bit/32 rule threw
    /// where the f32-row rule held 15 of 16 with 1 flip) carrying every later
    /// row. A broken router flips about every site (16: a wrong set shares
    /// about 0.25 of its 8 experts with ik's), today's worst is 8, so the cap
    /// is 12, three quarters of the sites. The floor is 20 %: 25 % is the least
    /// the rule leaves (a layer-0 flip of token 0 keeps four rows), 20 % refuses
    /// one row less (3 of 16) and a 25 % floor would leave no room. The cost:
    /// on the variants the clause holds layer 0 alone; layers 1 to 3 are held
    /// by the forced arm, per layer on ik's own input. The real tier holds
    /// every row.
    const FREE_COVER_MIN: f64 = 0.20;
    /// The most near-tie flips a set may carry (see [`FREE_COVER_MIN`]).
    const FREE_FLIPS_MAX: usize = 12;
    /// PIN(2026-10-13): the probe's layer outputs against a card-only load's
    /// chain: the same kernels on the same input, bit for bit on the
    /// whole-card load and 4.5e-8 on the placed no-host load of the base
    /// fixture (its sums run in the placed walk's order) in today's runs; 1e-6
    /// is twenty times the larger.
    const FREE_PROBE_CARD: f64 = 1e-6;
    /// PIN(2026-10-13): the same against a load whose host tier runs experts:
    /// their arithmetic is the host's own, 5.1e-3 to 5.5e-3 in today's six
    /// runs; 2e-2 is four times the larger.
    const FREE_PROBE_HOST: f64 = 2e-2;
    /// PIN(2026-10-22): (o9)'s wide unit's last-row logits against the host
    /// tier's wide unit over the same ids (the card runs the experts through
    /// the route GEMMs on q8_1 rows, the host through ik's 8-bit rule):
    /// 1.90e-2 to 2.26e-2 in today's five runs (the base fixture's dispatch
    /// layers and the variants `q35iq4xs`, `q35q80`, `q35iq4q8` and `q35q6k`);
    /// 1e-1 is four times the larger. A route block whose gate and up weights
    /// are swapped reads 4.8e-1 on `q35q6k`.
    const COMMON_WIDE_REL: f64 = 1e-1;

    /// The free-running clause: every layer's output of the one-token
    /// chain against ik's `l_out-L` within [`FREE_BAND`], and the last
    /// position's argmax. In the fixture tier the rows a near-tie router flip
    /// carries are not held ([`held_rows`]; the flip is read from the chain's
    /// own router decisions by [`route_probe`], tested by [`router_tie`],
    /// printed, and counted against [`FREE_FLIPS_MAX`]); the clause is red when
    /// a flip is no near tie, when a site's ids are not the top set of its own
    /// logits, when the held rows fall below [`FREE_COVER_MIN`] of the
    /// rows ik kept, or when the probe does not replay the chain's layers
    /// within `probe_rel` ([`FREE_PROBE_CARD`], [`FREE_PROBE_HOST`]). The real
    /// tier holds every row.
    fn free(
        m: &mut Qwen35moeModel,
        which: &str,
        toks: &[u32],
        decode: &PathRun,
        probe_rel: f64,
    ) -> Result<bool, GateError> {
        let man = &ik_oracle(
            &format!("{which}free: every layer's output and the last argmax"),
            Set::Batch,
        )?;
        let n = m.layers().len();
        let ik_out: Vec<Vec<f32>> = (0..n)
            .map(|l| tap(man, &format!("l_out-{l}")))
            .collect::<Result<_, _>>()?;
        let ik_logits = tap(man, "result_output")?;
        let vocab = m.body("free")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output holds no row")?;
        m.set_layer_taps(true)?;
        fresh(m)?;
        let t_n = toks.len();
        let mut rel = vec![vec![None::<f64>; t_n]; n];
        let mut chain_rows: Vec<Vec<f32>> = vec![Vec::new(); n];
        for (t, &tok) in toks.iter().enumerate() {
            m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for (l, (ik, got)) in ik_out.iter().zip(&taps).enumerate() {
                chain_rows[l].extend_from_slice(got);
                let kept = ik.len() / HIDDEN;
                let Some(i) = (t + kept).checked_sub(t_n) else {
                    continue;
                };
                rel[l][t] = Some(rel_to(got, &ik[i * HIDDEN..(i + 1) * HIDDEN]));
            }
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ours = decode.logits.last().ok_or("no decode logits")?;
        let top = argmax(ours);
        let all_rows = |l: usize| rel[l].iter().flatten().copied().fold(0.0f64, worse);
        let per_layer: Vec<f64> = (0..n).map(all_rows).collect();
        // The head reads the last position of the last layer's output, the one
        // row ik keeps of that layer.
        let head_in = *per_layer.last().ok_or("the model holds no layer")?;
        let (argmax_ok, argmax_line) = head_argmax(ik_last, top, head_in);
        let chain = match cfg().tier {
            Tier::Real => None,
            Tier::Fixture => Some(route_probe(t_n, &chain_rows)?),
        };
        let sites = chain.as_ref().map(|c| route_sites(man, c)).transpose()?;
        let held = match &sites {
            Some(s) => held_rows(n, t_n, &s.flips),
            None => vec![vec![true; t_n]; n],
        };
        let held_worst: Vec<f64> = (0..n)
            .map(|l| {
                (0..t_n)
                    .filter(|&t| held[l][t])
                    .filter_map(|t| rel[l][t])
                    .fold(0.0f64, worse)
            })
            .collect();
        let counts = |keep: &dyn Fn(usize, usize) -> bool| -> Vec<usize> {
            (0..n)
                .map(|l| {
                    (0..t_n)
                        .filter(|&t| rel[l][t].is_some() && keep(l, t))
                        .count()
                })
                .collect()
        };
        let (all_n, held_n) = (counts(&|_, _| true), counts(&|l, t| held[l][t]));
        for (l, &e) in held_worst.iter().enumerate() {
            if l % 8 == 0 || l == n - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        let worst = held_worst.iter().copied().fold(0.0f64, worse);
        let mut ok = worst <= FREE_BAND && argmax_ok;
        if let (Some(s), Some(c)) = (&sites, &chain) {
            let set = Set::Batch.name();
            for f in &s.flips {
                let only_ours: Vec<u32> = f
                    .ours
                    .iter()
                    .copied()
                    .filter(|&e| !f.theirs.contains(&(e as i32)))
                    .collect();
                let only_ik: Vec<i32> = f
                    .theirs
                    .iter()
                    .copied()
                    .filter(|&e| !f.ours.contains(&(e as u32)))
                    .collect();
                println!(
                    "free {which}flip set={set} layer={} token={}: ours {only_ours:?} ik {only_ik:?} \
                     (sets {:?} / {:?}), ik margin {:.3e}, our logits' error {:.3e}: {}",
                    f.layer,
                    f.token,
                    f.ours,
                    f.theirs,
                    f.margin,
                    f.err,
                    if f.tie {
                        "a near tie (counted)"
                    } else {
                        "FAIL: not a near tie"
                    }
                );
                ok &= f.tie;
            }
            for &(l, t) in &s.inconsistent {
                println!(
                    "free {which}route layer={l} token={t}: the ids are not the top set of the \
                     chain's own logits FAIL"
                );
            }
            ok &= s.inconsistent.is_empty();
            let (held_all, rows_all) = (held_n.iter().sum::<usize>(), all_n.iter().sum::<usize>());
            let cover = held_all as f64 / rows_all.max(1) as f64;
            let ties = s.flips.iter().filter(|f| f.tie).count();
            let (cover_ok, cap_ok, probe_ok) = (
                cover >= FREE_COVER_MIN,
                ties <= FREE_FLIPS_MAX,
                c.agree.0 <= probe_rel,
            );
            println!(
                "free {which}coverage set={set}: held {held_all} of {rows_all} rows ({:.0} %, floor \
                 {:.0} %) {}; per layer held {held_n:?} of {all_n:?}; route sites {}, flips {} \
                 (cap {FREE_FLIPS_MAX}) {}, inconsistent {}; the probe's layer outputs against \
                 the chain's: worst rel {:.3e} (band {probe_rel:.0e}), {} of {} layers bit \
                 for bit {}",
                100.0 * cover,
                100.0 * FREE_COVER_MIN,
                verdict(cover_ok),
                s.sites,
                s.flips.len(),
                verdict(cap_ok),
                s.inconsistent.len(),
                c.agree.0,
                c.agree.1,
                c.agree.2,
                verdict(probe_ok),
            );
            ok &= cover_ok && cap_ok && probe_ok;
        }
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); last \
             position {argmax_line} logits_rel={:.3e} (printed) {}",
            rel_to(ours, ik_last),
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------- (f)/(t2) forced taps

    /// The file's gains a gap is computed through.
    struct Gains {
        attn_norm: Vec<Vec<f32>>,
        ffn_norm: Vec<Vec<f32>>,
        /// Each layer's routed mixture as the file's spec states it: the
        /// taps slice the router's rows and slots by it, the engine's source.
        moe: Vec<MoeShape>,
    }

    impl Gains {
        fn attn(&self, l: usize) -> Result<&[f32], GateError> {
            Ok(self
                .attn_norm
                .get(l)
                .map(Vec::as_slice)
                .ok_or("a layer past the file's gains")?)
        }

        fn read(n: usize) -> Result<Gains, GateError> {
            let split = open_model()?;
            let each = |stem: &str| -> Result<Vec<Vec<f32>>, GateError> {
                (0..n)
                    .map(|l| split_f32(&split, &format!("blk.{l}.{stem}"), HIDDEN))
                    .collect()
            };
            let spec = model::arch::qwen35moe::spec::read(&split)
                .map_err(|e| format!("{}: {e}", cfg().path.display()))?
                .spec;
            let moe = spec
                .layers
                .iter()
                .take(n)
                .enumerate()
                .map(|(l, s)| {
                    s.moe()
                        .map(MoeShape::of)
                        .ok_or_else(|| format!("layer {l}: no routed FFN in the file's spec"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if moe.len() != n {
                return Err(format!("the file's spec has {} layers, want {n}", moe.len()).into());
            }
            Ok(Gains {
                attn_norm: each("attn_norm.weight")?,
                ffn_norm: each("post_attention_norm.weight")?,
                moe,
            })
        }
    }

    /// Our `[t][h][d]` values of `h_n` heads of `d_n` against ik's `ik`
    /// laid out by `ne`: `[t][h][d]` (`ne = [d, h, t]`) or `[h][t][d]` (`ne
    /// = [d, t, h]`, a permuted norm's output), each ik value times `scale`.
    fn heads_rel(
        ours: &[f32],
        ik: &[f32],
        ne: [u64; 4],
        (t_n, h_n, d_n): (usize, usize, usize),
        scale: f32,
    ) -> Result<f64, GateError> {
        let ne = [ne[0] as usize, ne[1] as usize, ne[2] as usize];
        let head_major = ne == [d_n, t_n, h_n] && t_n > 1;
        if !(head_major || ne == [d_n, h_n, t_n] || ik.len() == t_n * h_n * d_n) {
            return Err(format!("a head tap of ne {ne:?} for {t_n} x {h_n} x {d_n}").into());
        }
        let mut want = vec![0.0f32; t_n * h_n * d_n];
        for t in 0..t_n {
            for h in 0..h_n {
                for d in 0..d_n {
                    let src = if head_major {
                        (h * t_n + t) * d_n + d
                    } else {
                        (t * h_n + h) * d_n + d
                    };
                    want[(t * h_n + h) * d_n + d] = ik[src] * scale;
                }
            }
        }
        Ok(rel_to(ours, &want))
    }

    /// Values `lo..hi` of every `width`-value head of `v`.
    fn head_part(v: &[f32], width: usize, lo: usize, hi: usize) -> Vec<f32> {
        v.chunks(width).flat_map(|h| h[lo..hi].to_vec()).collect()
    }

    /// A delta layer's taps against ik's: the worst ratio; the decay's
    /// underflow sites must be ik's.
    #[allow(
        clippy::too_many_arguments,
        reason = "a layer's run, its inputs' gaps, its store and ik's set, as the tap map reads them"
    )]
    fn delta_taps(
        man: &RefManifest,
        l: usize,
        r: &Delta35Run,
        store: &StoreHost,
        (t_n, pos_end): (usize, usize),
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<bool, GateError> {
        let gap_y = quant_gap_two_sided(&r.y, N_V * HEAD_V);
        w.add(
            "qkv_mixed",
            rel_to(&r.x, &tap(man, &format!("qkv_mixed-{l}"))?),
            gap_in,
        );
        w.add("z", rel_to(&r.z, &tap(man, &format!("z-{l}"))?), gap_in);
        w.add(
            "beta_in",
            rel_to(&r.b, &tap(man, &format!("beta_in-{l}"))?),
            gap_in,
        );
        w.add(
            "alpha",
            rel_to(&r.a, &tap(man, &format!("alpha-{l}"))?),
            gap_in,
        );
        // ik dumps g, not exp(g) (the delta op takes g): our g is the log of
        // our decay where it did not underflow, and the underflow sites are
        // those where ik's exp — `expf_ik`, the same algorithm — gives 0.
        let g = tap(man, &format!("g_in-{l}"))?;
        let ik_decay: Vec<f32> = g.iter().map(|&x| expf_ik(x)).collect();
        let under = |v: &[f32]| -> Vec<usize> { (0..v.len()).filter(|&i| v[i] == 0.0).collect() };
        let (ours_u, ik_u) = (under(&r.decay), under(&ik_decay));
        let live: Vec<usize> = (0..g.len().min(r.decay.len()))
            .filter(|&i| r.decay[i] > 0.0)
            .collect();
        let g_ours: Vec<f32> = live
            .iter()
            .map(|&i| f64::from(r.decay[i]).ln() as f32)
            .collect();
        let g_ik: Vec<f32> = live.iter().map(|&i| g[i]).collect();
        w.add(
            "g_in (the log of our decay)",
            rel_to(&g_ours, &g_ik),
            gap_in,
        );
        let silu = tap(man, &format!("conv_output_silu-{l}"))?;
        let v_of =
            |x: &[f32]| -> Vec<f32> { x.chunks(C).flat_map(|t| t[C / 2..].to_vec()).collect() };
        w.add(
            "conv_output_silu (v)",
            rel_to(&v_of(&r.conv), &v_of(&silu)),
            gap_in,
        );
        let (q, q_ne) = tap_or_src(man, &format!("q_fused-{l}"))?;
        let (k, k_ne) = tap_or_src(man, &format!("k_fused-{l}"))?;
        let q_ours: Vec<f32> = r.conv.chunks(C).flat_map(|t| t[..C / 4].to_vec()).collect();
        let k_ours: Vec<f32> = r
            .conv
            .chunks(C)
            .flat_map(|t| t[C / 4..C / 2].to_vec())
            .collect();
        w.add(
            "q_fused",
            heads_rel(&q_ours, &q, q_ne, (t_n, 16, HEAD_V), Q_SCALE)?,
            gap_in,
        );
        w.add(
            "k_fused",
            heads_rel(&k_ours, &k, k_ne, (t_n, 16, HEAD_V), 1.0)?,
            gap_in,
        );
        w.add(
            "attn_output",
            rel_to(&r.o, &tap(man, &format!("attn_output-{l}"))?),
            gap_in,
        );
        w.add(
            "attn_out_norm",
            rel_to(&r.y, &tap(man, &format!("attn_out_norm-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let ik_upd = tap(man, &format!("linear_attn_out-{l}"))?;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        w.add(
            "linear_attn_out",
            rel_to(&got, &ik_upd),
            gap_in.hypot(gap_y),
        );
        let StoreHost::Rec { state, ring } = store else {
            return Err(format!("layer {l} is a delta layer with another store").into());
        };
        let ik_state = tap(man, &format!("new_state-{l}"))?;
        let mut t = vec![0.0f32; state.len()];
        for h in 0..N_V {
            for kk in 0..HEAD_V {
                for v in 0..HEAD_V {
                    t[(h * HEAD_V + v) * HEAD_V + kk] = ik_state[(h * HEAD_V + kk) * HEAD_V + v];
                }
            }
        }
        w.add("new_state (transposed)", rel_to(state, &t), gap_in);
        let ik_conv = tap(man, &format!("new_conv_states_cont-{l}"))?;
        let mut ours_conv = vec![0.0f32; 3 * C];
        for j in 0..3 {
            let p = pos_end - 3 + j;
            let slot = p % RING_ROWS;
            for ch in 0..C {
                ours_conv[ch * 3 + j] = ring[slot * C + ch];
            }
        }
        w.add("new_conv_states_cont", rel_to(&ours_conv, &ik_conv), gap_in);
        let ok = ours_u == ik_u;
        if !ours_u.is_empty() || !ik_u.is_empty() {
            println!(
                "forced layer={l} decay underflow sites ours {ours_u:?} ik {ik_u:?} {}",
                verdict(ok)
            );
        }
        Ok(ok)
    }

    /// An attention layer's taps against ik's.
    fn attn_taps(
        man: &RefManifest,
        l: usize,
        r: &Gqa35Run,
        t_n: usize,
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<(), GateError> {
        w.add(
            "Qaux",
            rel_to(&r.qg, &tap(man, &format!("Qaux-{l}"))?),
            gap_in,
        );
        let q_normed = tap(man, &format!("Qcur_normed-{l}"))?;
        let q_roped = tap(man, &format!("Qcur_roped-{l}"))?;
        let k_normed = tap(man, &format!("Kcur_normed-{l}"))?;
        let k_roped = tap(man, &format!("Kcur_roped-{l}"))?;
        for (name, ours, ik, lo, hi) in [
            ("Qcur_normed (past the turn)", &r.q, &q_normed, ROT, HEAD),
            ("Qcur_roped (the turn)", &r.q, &q_roped, 0, ROT),
            ("Kcur_normed (past the turn)", &r.k, &k_normed, ROT, HEAD),
            ("Kcur_roped (the turn)", &r.k, &k_roped, 0, ROT),
        ] {
            w.add(
                name,
                rel_to(&head_part(ours, HEAD, lo, hi), &head_part(ik, HEAD, lo, hi)),
                gap_in,
            );
        }
        w.add(
            "Vcur",
            rel_to(&r.v, &tap(man, &format!("Vcur-{l}"))?),
            gap_in,
        );
        w.add("fa", rel_to(&r.fa, &tap(man, &format!("fa-{l}"))?), gap_in);
        let gated: Vec<f32> = (0..t_n * N_HEAD * HEAD)
            .map(|i| {
                let (t, h, d) = (i / (N_HEAD * HEAD), (i / HEAD) % N_HEAD, i % HEAD);
                r.fa[i] * sigmoid(r.qg[t * Q_ROWS + h * 2 * HEAD + HEAD + d])
            })
            .collect();
        let gap_g = quant_gap_two_sided(&gated, N_HEAD * HEAD);
        w.add(
            "qkv_gated (host product)",
            rel_to(&gated, &tap(man, &format!("qkv_gated-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        let ik_upd = tap(man, &format!("attn_out-{l}"))?;
        w.add("attn_out", rel_to(&got, &ik_upd), gap_in.hypot(gap_g));
        Ok(())
    }

    /// The FFN half on ik's own FFN input against ik's MoE taps, over the
    /// rows ik kept (`ffn_in` may hold more: the last layer keeps only the
    /// output token's rows past its attention): the router's 257 logits
    /// (`ffn_moe_logits` and `shared_expert_gate`), the ids as sets — a
    /// token whose sets differ is a flip, allowed only where ik's own margin
    /// between its eighth pick and the best expert it left lies inside
    /// twice our logits' error at that token, counted and printed — the
    /// weights matched by id with the shared expert's (ik's sigmoid tap, or
    /// the sigmoid of ik's logit where a one-token graph fused it away), and
    /// the layer output's update on the tokens without a flip.
    fn moe_taps(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        l: usize,
        ffn_in: &[f32],
        gains: &Gains,
        w: &mut Worst,
    ) -> Result<(bool, usize), GateError> {
        let shape = gains.moe[l];
        if !shape.rule.gated {
            return Err(format!("layer {l}: {shape:?} has no gated shared expert").into());
        }
        let (n_expert, n_used) = (shape.experts as usize, shape.top_k as usize);
        let slots = n_used + 1;
        let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
        let gate = tap(man, &format!("shared_expert_gate-{l}"))?;
        let t_n = gate.len();
        if t_n == 0 || logits.len() != t_n * n_expert {
            return Err(format!(
                "layer {l}: {} router logits for {t_n} gate logits",
                logits.len()
            )
            .into());
        }
        let ffn_in = last_rows(ffn_in, HIDDEN, t_n)
            .ok_or_else(|| format!("layer {l}: ik routes {t_n} rows, the input holds fewer"))?;
        let f = m.ffn_rows(l, ffn_in)?;
        let gap = quant_gap_two_sided(&normed(ffn_in, &gains.ffn_norm[l], HIDDEN), HIDDEN);
        let ik_logits: Vec<f32> = (0..t_n)
            .flat_map(|t| {
                let mut row = logits[t * n_expert..(t + 1) * n_expert].to_vec();
                row.push(gate[t]);
                row
            })
            .collect();
        w.add(
            "ffn_moe_logits + shared_expert_gate",
            rel_to(&f.logits, &ik_logits),
            gap,
        );
        let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ids = topk_ids_logical_within(man, trow, shape.experts)?;
        let wn = tap(man, &format!("ffn_moe_weights_norm-{l}"))?;
        if ids.len() != t_n * n_used || wn.len() != t_n * n_used {
            return Err(format!(
                "layer {l}: {} ids and {} weights for {t_n} tokens",
                ids.len(),
                wn.len()
            )
            .into());
        }
        let ik_sg: Vec<f32> = match man.tensor(&format!("shared_expert_gate_sigmoid-{l}"), 0) {
            Ok(row) => ref_tensor_logical_in(&man.dir, row)?,
            Err(_) => gate.iter().map(|&g| sigmoid(g)).collect(),
        };
        if ik_sg.len() != t_n {
            return Err(format!("layer {l}: {} shared gates for {t_n} tokens", ik_sg.len()).into());
        }
        let (mut ok, mut flipped) = (true, Vec::new());
        let (mut w_ours, mut w_ik) = (Vec::new(), Vec::new());
        let row = n_expert + 1;
        for (t, &sg) in ik_sg.iter().enumerate() {
            let ours = &f.ids[t * slots..t * slots + n_used];
            let theirs = &ids[t * n_used..(t + 1) * n_used];
            if !ours.iter().all(|&e| theirs.contains(&(e as i32))) {
                let lg = &ik_logits[t * row..t * row + n_expert];
                let margin = flip::margin(lg, theirs) as f32;
                let err = f.logits[t * row..t * row + n_expert]
                    .iter()
                    .zip(lg)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let tie = router_tie(margin, err);
                ok &= tie;
                flipped.push(t);
                println!(
                    "forced layer={l} token={t}: ids ours {ours:?} ik {theirs:?}, ik margin {:.3e}, \
                     our logits' error {err:.3e}: {}",
                    margin,
                    if tie { "a near tie (counted)" } else { "FAIL" }
                );
                continue;
            }
            for (s, &e) in ours.iter().enumerate() {
                let j = theirs
                    .iter()
                    .position(|&x| x == e as i32)
                    .ok_or("an id the set test found")?;
                w_ours.push(f.weights[t * slots + s]);
                w_ik.push(wn[t * n_used + j]);
            }
            w_ours.push(f.weights[t * slots + n_used]);
            w_ik.push(sg);
        }
        if !w_ik.is_empty() {
            w.add(
                "ffn_moe_weights_norm + shared_expert_gate_sigmoid",
                rel_to(&w_ours, &w_ik),
                gap,
            );
        }
        let out = tap(man, &format!("l_out-{l}"))?;
        let ik_out = last_rows(&out, HIDDEN, t_n).ok_or("l_out rows")?;
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for t in (0..t_n).filter(|t| !flipped.contains(t)) {
            let r = t * HIDDEN..(t + 1) * HIDDEN;
            got.extend(
                f.l_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r.clone()])
                    .map(|(&o, &i)| o - i),
            );
            want.extend(
                ik_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r])
                    .map(|(&o, &i)| o - i),
            );
        }
        if !want.is_empty() {
            w.add("l_out (update)", rel_to(&got, &want), gap);
        }
        Ok((ok, flipped.len()))
    }

    /// Teacher-forced taps on every layer of `man` at `t_n` rows from
    /// position `pos`, each layer's store first set by `load` (a reset's
    /// zero store, or the set's prefill state).
    fn forced(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        label: &str,
        (pos, t_n): (u32, usize),
        gains: &Gains,
        load: &dyn Fn(&mut Qwen35moeModel, usize) -> Result<(), GateError>,
    ) -> Result<bool, GateError> {
        let n = m.layers().len();
        let kinds = m.body("forced")?.kinds();
        let embd = tap(man, "inp_embd")?;
        let mut ok = true;
        let (mut worst, mut flips_all) = (0.0f64, 0usize);
        for (l, &kind) in kinds.iter().enumerate() {
            let x_all = if l == 0 {
                embd.clone()
            } else {
                tap(man, &format!("l_out-{}", l - 1))?
            };
            let x_in = last_rows(&x_all, HIDDEN, t_n).ok_or("a layer input row is missing")?;
            load(m, l)?;
            let run = m.layer_rows(l, x_in, pos)?;
            let gap_in = quant_gap_two_sided(&normed(x_in, gains.attn(l)?, HIDDEN), HIDDEN);
            let mut w = Worst::default();
            let body_ok = match (&run.mixer, kind) {
                (Mixer35Run::Delta(r), LayerKind35::Delta) => {
                    let store = {
                        let (gpu, _, body) = m.body_parts("forced")?;
                        body.store(gpu, l)?
                    };
                    delta_taps(
                        man,
                        l,
                        r,
                        &store,
                        (t_n, pos as usize + t_n),
                        gap_in,
                        (&run.ffn_inp, x_in),
                        &mut w,
                    )?
                }
                (Mixer35Run::Attention(r), LayerKind35::Attention) => {
                    attn_taps(man, l, r, t_n, gap_in, (&run.ffn_inp, x_in), &mut w)?;
                    true
                }
                _ => return Err(format!("layer {l}: a run of another kind than the layer").into()),
            };
            // The FFN half alone on ik's FFN input: the layer input plus
            // ik's mixer update, over the rows ik kept.
            let upd_name = match kind {
                LayerKind35::Delta => format!("linear_attn_out-{l}"),
                LayerKind35::Attention => format!("attn_out-{l}"),
            };
            let ik_upd = tap(man, &upd_name)?;
            let kept = ik_upd.len() / HIDDEN;
            let x_kept = last_rows(x_in, HIDDEN, kept).ok_or("rows")?;
            let ffn_in: Vec<f32> = ik_upd.iter().zip(x_kept).map(|(&a, &x)| a + x).collect();
            let (moe_ok, flips) = moe_taps(m, man, l, &ffn_in, gains, &mut w)?;
            let pass = body_ok && moe_ok && w.ratio <= RATIO_BAND;
            println!(
                "forced {label} layer={l} {:?}: worst ratio {:.2} at {} (band {RATIO_BAND}); \
                 flip sites {flips} {}",
                kind,
                w.ratio,
                w.tap,
                verdict(pass)
            );
            if !pass || l % 8 == 0 {
                for line in &w.lines {
                    println!("  forced {label} layer={l} {line}");
                }
            }
            worst = worse(worst, w.ratio);
            flips_all += flips;
            ok &= pass;
        }
        println!(
            "forced {label}: {n} layers x {t_n} rows from position {pos}; worst ratio {worst:.2} \
             (band {RATIO_BAND}); flip sites {flips_all} (near ties, counted) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (t) the step sets

    /// Layer `l`'s store as a step set's prefill left it: a delta layer's
    /// state (ik's `[k][v]` per head, transposed into ours) and conv ring
    /// (ik's `[C][3]` window of the three positions before `p`, into slots
    /// `(p − 3 + j) mod RING_ROWS`), or an attention layer's K and V rows
    /// `0..p`. ik lays a plane out as rows of 512 in three shapes: `[512,
    /// cells]`, K's `[256, 2·cells]` (a cell's two heads adjacent) and, with
    /// flash attention, V's flat `[512·cells]`; without flash attention V is
    /// transposed, `[cells, 512]`. A flat plane from a set with no
    /// `FLASH_ATTN_EXT` at the layer is refused by name.
    fn prefill_store(
        man: &RefManifest,
        kind: LayerKind35,
        l: usize,
        p: usize,
    ) -> Result<StoreHost, GateError> {
        match kind {
            LayerKind35::Delta => {
                let s = ref_tensor_logical_in(&man.dir, man.input(&format!("cache_s_l{l}"), 0)?)?;
                let conv_len = 3 * C;
                if s.len() != conv_len + N_V * HEAD_V * HEAD_V {
                    return Err(format!("cache_s_l{l} holds {} values", s.len()).into());
                }
                let (conv, ssm) = s.split_at(conv_len);
                let mut state = vec![0.0f32; ssm.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            state[(h * HEAD_V + v) * HEAD_V + k] =
                                ssm[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let mut ring = vec![0.0f32; RING_ROWS * C];
                for j in 0..3 {
                    let Some(pos) = (p + j).checked_sub(3) else {
                        continue;
                    };
                    let slot = pos % RING_ROWS;
                    for ch in 0..C {
                        ring[slot * C + ch] = conv[ch * 3 + j];
                    }
                }
                Ok(StoreHost::Rec { state, ring })
            }
            LayerKind35::Attention => {
                let flash = man
                    .tensor(&format!("fa-{l}"), 0)
                    .is_ok_and(|t| t.op == "FLASH_ATTN_EXT");
                let plane = |name: String| -> Result<Vec<u16>, GateError> {
                    let row = man.input(&name, 0)?;
                    let bits = widened_f16_rows_in(&man.dir, row)?;
                    let (ne0, ne1) = (row.ne[0] as usize, row.ne[1] as usize);
                    let flat = ne1 == 1 && ne0.is_multiple_of(KV_ROW);
                    let rows = (ne0 == KV_ROW && ne1 >= p)
                        || (ne0 == HEAD && ne1.is_multiple_of(N_KV) && ne1 / N_KV >= p)
                        || (flat && flash && ne0 / KV_ROW >= p);
                    let transposed = ne1 == KV_ROW && ne0 >= p;
                    if !rows && !transposed {
                        return Err(format!(
                            "{name} is {:?} (flash attention at the layer: {flash}), \
                             not rows of {KV_ROW} holding {p} positions",
                            row.ne
                        )
                        .into());
                    }
                    let mut out = vec![0u16; N_KV * CTX * HEAD];
                    for r in 0..p {
                        for h in 0..N_KV {
                            for d in 0..HEAD {
                                let c = h * HEAD + d;
                                let src = if rows { r * KV_ROW + c } else { c * ne0 + r };
                                out[(h * CTX + r) * HEAD + d] = bits[src];
                            }
                        }
                    }
                    Ok(out)
                };
                Ok(StoreHost::Kv {
                    k: plane(format!("cache_k_l{l}"))?,
                    v: plane(format!("cache_v_l{l}"))?,
                })
            }
        }
    }

    fn step_set(m: &mut Qwen35moeModel, set: Set, gains: &Gains) -> Result<bool, GateError> {
        let name = set.name();
        let man = ik_oracle(
            &format!(
                "(t) the decode step after {}'s prefill, free-running and teacher-forced",
                set.what()
            ),
            set,
        )?;
        let (p, tail, before) = man.step()?;
        let token = *tail.first().ok_or("the step set holds no token")?;
        let pu = p as usize;
        let kinds = m.body("step_set")?.kinds();
        let loaded: Vec<StoreHost> = kinds
            .iter()
            .enumerate()
            .map(|(l, &k)| prefill_store(&man, k, l, pu))
            .collect::<Result<_, _>>()?;
        println!(
            "step {name}: position {p}, token {token} after {} tokens; every layer's store loaded \
             from the set's inputs",
            before.len()
        );
        let load_all = |m: &mut Qwen35moeModel| -> Result<(), GateError> {
            fresh(m)?;
            m.seed_depth(pu)?;
            let (gpu, _, body) = m.body_parts("step_set")?;
            for (l, s) in loaded.iter().enumerate() {
                body.set_store(gpu, l, s)?;
            }
            Ok(())
        };
        // (t1) the step, free-running from the loaded state.
        load_all(m)?;
        m.set_layer_taps(true)?;
        m.set_mode(StepMode::Eager);
        let top = m.step(&[token])?;
        let taps = m.layer_taps()?;
        let logits = m.logits()?;
        let after = stores(m)?;
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ik_logits = tap(&man, "result_output")?;
        let vocab = m.body("step_set")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output")?;
        let (mut worst, mut worst_state) = (0.0f64, 0.0f64);
        // Layer l's input is layer l − 1's output; layer 0's is the exact
        // embedding. After the loop it holds the last layer's output distance,
        // the head's input.
        let mut in_rel = 0.0f64;
        for (l, got) in taps.iter().enumerate() {
            let want = tap(&man, &format!("l_out-{l}"))?;
            let want = last_rows(&want, HIDDEN, 1).ok_or("l_out")?;
            let e = rel_to(got, want);
            worst = worse(worst, e);
            if let (LayerKind35::Delta, StoreHost::Rec { state, .. }) = (kinds[l], &after[l]) {
                let ik_state = tap(&man, &format!("new_state-{l}"))?;
                let mut t = vec![0.0f32; state.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            t[(h * HEAD_V + v) * HEAD_V + k] =
                                ik_state[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let x_in = if l == 0 {
                    tap(&man, "inp_embd")?
                } else {
                    tap(&man, &format!("l_out-{}", l - 1))?
                };
                let x_in = last_rows(&x_in, HIDDEN, 1).ok_or("input")?;
                let gap = quant_gap_two_sided(&normed(x_in, &gains.attn_norm[l], HIDDEN), HIDDEN);
                let state_rel = rel_to(state, &t);
                let ratio = state_ratio(state_rel, gap, in_rel);
                println!(
                    "step {name} free layer={l} new_state: state_rel={state_rel:.3e} gap={gap:.3e} \
                     in_rel={in_rel:.3e} ratio={ratio:.2}"
                );
                worst_state = worse(worst_state, ratio);
            }
            if l % 8 == 0 || l == taps.len() - 1 || e > FREE_BAND {
                println!("step {name} free layer={l} l_out_rel={e:.3e}");
            }
            in_rel = e;
        }
        let (argmax_ok, argmax_line) = head_argmax(ik_last, top, in_rel);
        let mut ok = worst <= FREE_BAND && argmax_ok && worst_state <= RATIO_BAND;
        println!(
            "step {name} free: worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); worst new_state \
             ratio {worst_state:.2} (band {RATIO_BAND}); {argmax_line} logits_rel={:.3e} \
             (printed) {}",
            rel_to(&logits, ik_last),
            verdict(ok)
        );
        // (t2) each layer teacher-forced at one row on its reloaded store.
        let reload = |m: &mut Qwen35moeModel, l: usize| -> Result<(), GateError> {
            let (gpu, _, body) = m.body_parts("step_set")?;
            body.set_store(gpu, l, &loaded[l])?;
            Ok(())
        };
        load_all(m)?;
        m.set_mode(StepMode::Eager);
        ok &= forced(m, &man, name, (p, 1), gains, &reload)?;
        m.set_mode(StepMode::Graph);
        Ok(ok)
    }

    // ------------------------------------- (p′) the ubatch arena's gemv arm

    fn gemv_arm(m: &mut Qwen35moeModel, toks: &[u32], decode: &PathRun) -> Result<bool, GateError> {
        tier::sc(
            "(p′) the ubatch arena's gemv arm: the pass path and the GEMM path = the decode steps",
        )?;
        let (want_tok, want_logits) = decode
            .tokens
            .last()
            .zip(decode.logits.last())
            .ok_or("no decode run")?;
        let mut ok = true;
        for path in [PrefillPath::Pass, PrefillPath::Gemm] {
            fresh(m)?;
            let plan = m.prefill_plan(toks.len(), path)?;
            let tok = m.prefill_with(toks, path)?;
            let logits = m.logits()?;
            let got = stores(m)?;
            let differ: Vec<usize> = (0..got.len())
                .filter(|&l| !decode.stores.get(l).is_some_and(|w| same_store(&got[l], w)))
                .collect();
            let same_logits = bits_equal(&logits, want_logits);
            let pass = tok == *want_tok
                && same_logits
                && differ.is_empty()
                && got.len() == decode.stores.len();
            println!(
                "gemv arm {path:?} plan={plan}: token {tok} vs the fifth step's {want_tok}, last \
                 logits bit-identical={same_logits}, every store bit-identical (layers differing \
                 {differ:?}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------- (u)/(d) the wide arm

    /// A prompt prefilled from zero stores, and the step after it.
    struct PromptRun {
        next: u32,
        logits: Vec<f32>,
        stores: Vec<StoreHost>,
        step_top: u32,
        step_taps: Vec<Vec<f32>>,
        step_logits: Vec<f32>,
    }

    fn prompt_run(
        m: &mut Qwen35moeModel,
        prompt: &[u32],
        path: PrefillPath,
        token: u32,
    ) -> Result<PromptRun, GateError> {
        fresh(m)?;
        println!(
            "prompt {path:?}: {} ids, plan={}",
            prompt.len(),
            m.prefill_plan(prompt.len(), path)?
        );
        let next = m.prefill_with(prompt, path)?;
        let logits = m.logits()?;
        let stores = stores(m)?;
        m.set_layer_taps(true)?;
        m.set_mode(StepMode::Eager);
        let step_top = m.step(&[token])?;
        let step_taps = m.layer_taps()?;
        let step_logits = m.logits()?;
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        Ok(PromptRun {
            next,
            logits,
            stores,
            step_top,
            step_taps,
            step_logits,
        })
    }

    /// The ring slots of the three positions before `p`, the rows ik's conv
    /// state holds.
    fn ik_slots(p: usize) -> Vec<usize> {
        (1..=3).rev().map(|d| (p - d) % RING_ROWS).collect()
    }

    /// The rows `slots` of a `[RING_ROWS][C]` ring.
    fn ring_rows(ring: &[f32], slots: &[usize]) -> Vec<f32> {
        slots
            .iter()
            .flat_map(|&s| ring[s * C..(s + 1) * C].to_vec())
            .collect()
    }

    /// `(a, b)` of a layer's store against `base`'s, each `rel_to`: the
    /// recurrent state and the ring's rows `slots`, or the K and the V rows.
    fn store_rel(ours: &StoreHost, base: &StoreHost, slots: &[usize]) -> (f64, f64) {
        let f16 = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| half_to_f32(b)).collect() };
        match (ours, base) {
            (StoreHost::Rec { state, ring }, StoreHost::Rec { state: s, ring: r }) => (
                rel_to(state, s),
                rel_to(&ring_rows(ring, slots), &ring_rows(r, slots)),
            ),
            (StoreHost::Kv { k, v }, StoreHost::Kv { k: kb, v: vb }) => {
                (rel_to(&f16(k), &f16(kb)), rel_to(&f16(v), &f16(vb)))
            }
            _ => (f64::INFINITY, f64::INFINITY),
        }
    }

    /// (u) and (d) on `m`, whose ubatch is `ubatch`: the step set's prompt on
    /// the pass path (the one-token path's bits) and as the ubatch walk, whose
    /// units are all wide (`label` prefixes the clause's lines). The whole
    /// load runs the prompt as one ubatch ([`U_GATE`]); the placed load as
    /// ubatches of [`PLACED_UBATCH`] — the host union's columns — whose layers
    /// run the card's experts through the wide GEMMs and the host's once a
    /// unit.
    fn wide_arm(m: &mut Qwen35moeModel, label: &str, ubatch: usize) -> Result<bool, GateError> {
        let man = ik_oracle(
            &format!("{label}(u)(d) the step-1,024 set's prompt by ubatches against the passes"),
            Set::D1k,
        )?;
        let (p, tail, prompt) = man.step()?;
        let token = *tail.first().ok_or("the step set holds no token")?;
        let pu = p as usize;
        if prompt.len() != pu || m.ubatch()? != ubatch {
            return Err(format!(
                "the step set's prompt is {} ids before position {p}, the model's ubatch {}; \
                 the clause runs them as ubatches of {ubatch}",
                prompt.len(),
                m.ubatch()?
            )
            .into());
        }
        let pass = prompt_run(m, prompt, PrefillPath::Pass, token)?;
        let ub = prompt_run(m, prompt, PrefillPath::Gemm, token)?;
        let kinds = m.body("wide_arm")?.kinds();
        let slots: Vec<usize> = (0..RING_ROWS).collect();
        // (u)
        let mut ok = true;
        // (u)'s layer-0 band: U_L0_REL_Q32 where layer 0 is the Q8_0 delta
        // layout (`Shape` admits it only with its shared expert apart).
        let band = if shape().apart()[0] {
            U_L0_REL_Q32
        } else {
            U_L0_REL
        };
        for (l, (a, b)) in ub.stores.iter().zip(&pass.stores).enumerate() {
            let (x, y) = store_rel(a, b, &slots);
            if l == 0 {
                let pass0 = kinds[0] == LayerKind35::Delta && worse(x, y) <= band;
                println!(
                    "{label}wide layer=0 {:?} state_rel={x:.3e} ring_rel={y:.3e} against the pass run \
                     (band {band:.0e}) {}",
                    kinds[0],
                    verdict(pass0)
                );
                ok &= pass0;
            } else {
                println!(
                    "{label}wide layer={l} {:?} rel={x:.3e} / {y:.3e} (printed)",
                    kinds[l]
                );
            }
        }
        println!(
            "{label}wide prefill: token ours ubatch={} pass={}, last logits_rel={:.3e} (printed)",
            ub.next,
            pass.next,
            rel_to(&ub.logits, &pass.logits)
        );
        // (d)
        let ik_slot = ik_slots(pu);
        let mut worst = (0.0f64, 0usize);
        for (l, &kind) in kinds.iter().enumerate() {
            let ik = prefill_store(&man, kind, l, pu)?;
            let (ua, ub_) = store_rel(&ub.stores[l], &ik, &ik_slot);
            let (pa, pb) = store_rel(&pass.stores[l], &ik, &ik_slot);
            let (wa, wb) = store_rel(&ub.stores[l], &pass.stores[l], &ik_slot);
            let d_pi = worse(pa, pb).max(f64::MIN_POSITIVE);
            let r = worse(0.0, worse(ua, ub_) / d_pi);
            if r > worst.0 {
                worst = (r, l);
            }
            let rw = worse(wa, wb) / d_pi;
            println!(
                "{label}prompt store layer={l} {kind:?} from ik: ubatch {:.3e} pass {:.3e} ratio {r:.2}; \
                 ubatch from pass {:.3e}, independent-error bound {:.2} (printed)",
                worse(ua, ub_),
                worse(pa, pb),
                worse(wa, wb),
                (1.0 + rw * rw).sqrt()
            );
        }
        let pass_d = worst.0 <= PROMPT_RATIO;
        println!(
            "{label}prompt stores: worst ratio {:.2} at layer {} (band {PROMPT_RATIO}) {}",
            worst.0,
            worst.1,
            verdict(pass_d)
        );
        ok &= pass_d;
        let mut worst_out = (0.0f64, 0usize);
        for (l, (a, b)) in ub.step_taps.iter().zip(&pass.step_taps).enumerate() {
            let want = tap(&man, &format!("l_out-{l}"))?;
            let want = last_rows(&want, HIDDEN, 1).ok_or("l_out")?;
            let (da, db) = (rel_to(a, want), rel_to(b, want));
            let r = worse(0.0, da / db.max(f64::MIN_POSITIVE));
            println!(
                "{label}prompt step layer={l} l_out from ik: ubatch {da:.3e} pass {db:.3e} ratio \
                 {r:.2} (printed)"
            );
            if r > worst_out.0 {
                worst_out = (r, l);
            }
        }
        let ik_logits = tap(&man, "result_output")?;
        let vocab = m.body("wide_arm")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output")?;
        let (lu, lp) = (
            rel_to(&ub.step_logits, ik_last),
            rel_to(&pass.step_logits, ik_last),
        );
        let r_logits = worse(0.0, lu / lp.max(f64::MIN_POSITIVE));
        let step_band = if label.is_empty() {
            STEP_RATIO
        } else {
            PLACED_STEP_RATIO
        };
        let pass_s = worst_out.0 <= step_band && r_logits <= STEP_RATIO;
        println!(
            "{label}prompt step {p}: worst l_out ratio {:.2} at layer {}, logits from ik ubatch {lu:.3e} \
             pass {lp:.3e} ratio {r_logits:.2} (band {step_band}); argmax ubatch={} pass={} \
             ik={} (printed) {}",
            worst_out.0,
            worst_out.1,
            ub.step_top,
            pass.step_top,
            argmax(ik_last),
            verdict(pass_s)
        );
        Ok(ok && pass_s)
    }

    // ------------------------------------------ (w) the ubatch size

    fn ubatch_bits(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        tier::sc("(w) the ubatch size moves no bit")?;
        let man = ik_set("(w) the step-1,024 set's prompt ids", Set::D1k)?;
        let (_, _, prompt) = man.step()?;
        let run = |m: &mut Qwen35moeModel,
                   u: usize,
                   cut: Option<usize>|
         -> Result<(u32, Vec<f32>, Vec<StoreHost>), GateError> {
            m.set_ubatch(u)?;
            fresh(m)?;
            let next = match cut {
                None => m.prefill_with(prompt, PrefillPath::Gemm)?,
                Some(c) => {
                    m.prefill_with(&prompt[..c], PrefillPath::Gemm)?;
                    m.prefill_with(&prompt[c..], PrefillPath::Gemm)?
                }
            };
            Ok((next, m.logits()?, stores(m)?))
        };
        let (tok0, logits0, stores0) = run(m, U_GATE, None)?;
        let arms = W_SIZES
            .iter()
            .map(|&u| (u, None))
            .chain(std::iter::once((U_GATE, Some(W_CUT))));
        let mut ok = true;
        for (u, cut) in arms {
            let (tok, logits, st) = run(m, u, cut)?;
            let differ: Vec<usize> = (0..st.len())
                .filter(|&l| !stores0.get(l).is_some_and(|w| same_store(&st[l], w)))
                .collect();
            let same_logits = bits_equal(&logits, &logits0);
            let pass = tok == tok0 && same_logits && differ.is_empty();
            let how = match cut {
                None => format!("plan={}", m.prefill_plan(prompt.len(), PrefillPath::Gemm)?),
                Some(c) => format!("two calls of {c} and {} ids", prompt.len() - c),
            };
            println!(
                "ubatch {u} {how}: token {tok} vs {tok0}, last logits bit-identical={same_logits}, \
                 every store bit-identical (layers differing {differ:?}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        m.set_ubatch(U_GATE)?;
        Ok(ok)
    }

    // ------------------------------------ (w′) the wide plan and its cut

    /// (w′) (module doc): every Wide plan's unit bounds over a sweep of tail
    /// classes at the load's ubatch, and one wide tail past the borrow's
    /// floor run uncut and as two calls cut at a whole ubatch, bit for bit.
    fn wide_plan(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        tier::sc("(w′) the wide plan's units and its cut")?;
        let ub = m.ubatch()?;
        // (a) the plans: every call length of one, two and three whole
        // ubatches and every tail class of each (no tail, a tail under the
        // wide floor that borrows, the floor itself, and every tail past it).
        let (mut checked, mut refused) = (0usize, Vec::new());
        let mut bad: Option<String> = None;
        for tokens in WIDE_FROM..=3 * ub {
            let plan = m.prefill_plan(tokens, PrefillPath::Wide);
            let plan = match plan {
                Ok(p) => p,
                Err(e) => {
                    refused.push(format!("{tokens} ids: {e}"));
                    continue;
                }
            };
            checked += 1;
            let units: Vec<usize> = plan
                .steps
                .iter()
                .map(|s| match s {
                    PrefillStep::Ubatch(t) | PrefillStep::Pass(t) => *t,
                })
                .collect();
            if units.iter().sum::<usize>() != tokens
                || units.iter().any(|&t| !(WIDE_FROM..=ub).contains(&t))
            {
                bad.get_or_insert_with(|| format!("{tokens} ids -> {plan}"));
            }
        }
        let plan_ok = refused.is_empty() && bad.is_none();
        println!(
            "wide plans at ub {ub}: {checked} lengths checked, {} refused by name (want 0: a \
             ubatch of at least {} rows splits every tail), first offending plan {}",
            refused.len(),
            2 * WIDE_FROM,
            bad.as_deref().unwrap_or("none")
        );
        if let Some(first) = refused.first() {
            println!("wide plans: first refusal: {first}");
        }
        // (b) one tail past the borrow's floor: the uncut call against the
        // same ids as two calls cut at the whole ubatch. The wide schedule
        // promises the same bits wherever a call is cut (module doc, (w′)):
        // every row of the tail runs the wide arm in either plan.
        let vocab = m.body("wide_plan")?.vocab();
        let ids = lcg_ids(ub + 16, vocab);
        fresh(m)?;
        let uncut = m.prefill_with(&ids, PrefillPath::Wide);
        let run_ok = match uncut {
            Ok(tok0) => {
                let (logits0, stores0) = (m.logits()?, stores(m)?);
                fresh(m)?;
                m.prefill_with(&ids[..ub], PrefillPath::Wide)?;
                let tok = m.prefill_with(&ids[ub..], PrefillPath::Wide)?;
                let (logits, st) = (m.logits()?, stores(m)?);
                let differ: Vec<usize> = (0..st.len())
                    .filter(|&l| !stores0.get(l).is_some_and(|w| same_store(&st[l], w)))
                    .collect();
                let same_logits = bits_equal(&logits, &logits0);
                let pass =
                    tok == tok0 && same_logits && differ.is_empty() && st.len() == stores0.len();
                println!(
                    "wide tail {} of {} ids, uncut and cut at {ub}: token {tok} vs {tok0}, last \
                     logits bit-identical={same_logits}, every store bit-identical (layers \
                     differing {differ:?}) {}",
                    ids.len() - ub,
                    ids.len(),
                    verdict(pass)
                );
                pass
            }
            Err(e) => {
                println!(
                    "wide tail {} of {} ids: the uncut call refused: {e} {}",
                    ids.len() - ub,
                    ids.len(),
                    verdict(false)
                );
                false
            }
        };
        Ok(plan_ok && run_ok)
    }

    // ---------------------------------------------- (k) the checkpoints

    /// (k)'s prompt: past an inner mark of the load's ubatch with a run
    /// left after it, and short of the next whole window.
    const KP: usize = 4200;

    /// (k)'s cache: the prompt, and the marks' positions, with room.
    const KCTX: usize = 4352;

    /// (k) (module doc): the checkpoints of a marked prompt call, the keep
    /// rule around them, the refused non-checkpoint cut, and the re-feed
    /// after a cut bit for bit. The model is dropped before the clause
    /// returns.
    fn checkpoints() -> Result<bool, GateError> {
        tier::sc("(k) checkpoints: the marks, the cut, the restore and the reset")?;
        let mut m = open(KCTX, KvQ8::F16)?;
        {
            let (_, _, body) = m.body_parts("gate_qwen35moe_e2e")?;
            body.set_checkpoints(true);
        }
        let vocab = m.body("gate_qwen35moe_e2e")?.vocab();
        let ids = lcg_ids(KP, vocab);
        let mut ok = true;

        // One checkpoint's bytes and slots against their derivation: each
        // delta layer's state (N_V heads of HEAD_V x HEAD_V f32) and conv
        // ring (RING_ROWS rows of C), and the slots the load's budget holds.
        // PIN(2026-10-11): the slots were "at least 9" of the fixed 4 GiB
        // budget; the budget now follows the load's context: the points one
        // prompt call over KCTX takes at the ubatch's spacing (its multiples
        // below KCTX and its end, KCTX.div_ceil(ubatch)) and the previous
        // call's end, under the copies 4 GiB fits. An exact pin, by the
        // budget's specification, not a relaxation.
        let c = m.body("gate_qwen35moe_e2e")?.checkpoints();
        let want_bytes = geometry().checkpoint_bytes();
        let every = m.ubatch()?;
        let fit = runtime::seqstate::capacity(runtime::seqstate::HOST_BUDGET, want_bytes as u64)
            .map_err(|e| e.to_string())?;
        let want_slots = (KCTX.div_ceil(every) + 1).min(fit);
        let points_ok = c.bytes() == want_bytes as u64 && c.capacity() == want_slots;
        println!(
            "checkpoints: one copy {} bytes (want {want_bytes}, derived: a delta layer's state \
             {} f32 and ring {} f32), {} slots of one copy at {KCTX} positions (want \
             {want_slots}: a call over them at the ubatch {every}'s spacing and the call before's \
             end) {}",
            c.bytes(),
            N_V * HEAD_V * HEAD_V,
            RING_ROWS * C,
            c.capacity(),
            verdict(points_ok)
        );
        ok &= points_ok;

        // The fresh call: its marks, the walks it ran, and what a cut keeps
        // around them.
        let plan = m.prefill_plan(KP, PrefillPath::Wide)?;
        let uncut = uncut_call(&mut m, &ids)?;
        // PIN(2026-10-04): the marks' spacing is the ubatch size the load was
        // asked for — the walk's most positions, the sibling body's rule —
        // re-derived here from the model's own size, where it was GLM's flat
        // 512: a mark every 512 rows cut a seat prompt of one ubatch walk's
        // rows into a walk each, every walk re-reading the expert weights
        // ([derived] pp@4096 at the seat down 27–29 %). At the load's size
        // the marks are its multiples: the points, the keep below an ask
        // (the multiple under it) and the walks follow — a run between
        // marks is one whole ubatch, the tail one unit, so the call runs
        // exactly ⌈P/ub⌉ walks, one more only where a mark falls inside a
        // walk (a continuation's first, or the borrowed tail).
        let ub = u32::try_from(m.ubatch()?).expect("a ubatch the load validated, at most UBATCH");
        let mark_below = |ask: u32| ub * (ask / ub);
        let want_pts: Vec<u32> = (1..)
            .map(|k| ub * k)
            .take_while(|&p| p < KP as u32)
            .chain([KP as u32])
            .collect();
        let pts = m.body("gate_qwen35moe_e2e")?.checkpoints().positions();
        let walks = m.body("gate_qwen35moe_e2e")?.prompt_walks();
        let (k_above, k_below, k_low, k_here) = (
            m.body("gate_qwen35moe_e2e")?.kept(4150, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(4095, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(100, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(KP as u32, KP as u32),
        );
        let marks_ok = pts == want_pts
            && (k_above.at, k_above.why.code()) == (mark_below(4150), "checkpoint")
            && (k_below.at, k_below.why.code()) == (mark_below(4095), "checkpoint")
            && (k_low.at, k_low.why.code()) == (mark_below(100), "no-checkpoint")
            && (k_here.at, k_here.why.code()) == (KP as u32, "current")
            && plan.ubatch_tokens() == KP;
        let walks_ok = walks == (KP as u32).div_ceil(ub) as usize;
        println!(
            "checkpoints: a prompt of {KP} ids (plan {plan}) took its points at {pts:?} (want \
             {want_pts:?}, the ubatch {ub}'s multiples); kept(4150) = {k_above}; kept(4095) = \
             {k_below}; kept(100) = {k_low}; kept({KP}) = {k_here}; {walks} walks (want \
             {}, the plan's own units) {}",
            (KP as u32).div_ceil(ub),
            verdict(marks_ok && walks_ok)
        );
        ok &= marks_ok && walks_ok;

        // A cut to a non-checkpoint is refused by name, the points kept.
        let refused = match m.rollback(4150) {
            Err(e) => e.to_string(),
            Ok(()) => "accepted".to_owned(),
        };
        let cut_ok = refused.contains("no checkpoint restores it")
            && m.body("gate_qwen35moe_e2e")?.checkpoints().positions() == want_pts;
        println!(
            "checkpoints: a cut to 4150: {refused}; the points stand {} ",
            verdict(cut_ok)
        );
        ok &= cut_ok;

        // The cut and the re-fed tail: the same bits the uncut call left.
        let cut_at = mark_below(4150);
        ok &= refed_after_cut(&mut m, &ids, cut_at, &uncut, &want_pts, ub)?;

        // A reset drops every point.
        fresh(&mut m)?;
        let dropped = m
            .body("gate_qwen35moe_e2e")?
            .checkpoints()
            .positions()
            .is_empty();
        println!(
            "checkpoints: a reset drops every point {}",
            verdict(dropped)
        );
        // The flash grid at this load's height (4,352 rows, not the (s)
        // clause's 1,088): the same SEGMENTS-sized segment pass and head
        // merge as (s) — the engine-level two-height pin, on a capture that
        // runs after the reset and so changes nothing above.
        m.capture_step()?;
        let (flash_ok, got) = flash_grid_ok(&m.step_graph_nodes()?)?;
        println!(
            "checkpoints flash grid at ctx {KCTX}: {got} {}",
            verdict(flash_ok)
        );
        Ok(ok && dropped && flash_ok)
    }

    // ------------------------------------------------ (n) resident slots

    /// (n)'s prompts, A and B: each past the load's first mark ([`U_GATE`]) by a run the mark keeps (`marked` drops a
    /// mark with a run of fewer than `WIDE_FROM` = `GEMV_COLS + 1` = 9 rows beside it), of different lengths so the
    /// slots stand at different positions, each with its steps inside [`CTX`] (`Slots35::new` holds both).
    const SLOT_A: usize = 1040;
    const SLOT_B: usize = 1056;

    /// This file's geometry as the derivations above state it.
    fn geometry() -> Geometry35 {
        Geometry35 {
            n_attn: n_attn(),
            n_delta: n_delta(),
            n_kv: N_KV,
            head: HEAD,
            n_v: N_V,
            head_v: HEAD_V,
            conv_c: C,
        }
    }

    /// (n) (module doc): the shared slots clause (`q35_clauses::slots_two`) over two lcg streams of the whole-card load
    /// at [`CTX`] with the checkpoints armed.
    fn slots_two(vocab: usize) -> Result<bool, GateError> {
        // A slot's node count reads every layer joined (`Shape::nodes_slot` refuses an apart one).
        let joined = !shape().apart().contains(&true);
        if !tier::layout_clause(
            "(n) resident slots: the harness's contracts over two lcg streams",
            joined,
        )? {
            return Ok(true);
        }
        let slots = Slots35::new(
            lcg_ids(SLOT_A + SLOT_B, vocab),
            SLOT_A,
            (U_GATE, CTX),
            (geometry(), nodes_slot()?),
            Box::new(|| open(CTX, KvQ8::F16)),
        )?;
        q35::slots_two(&slots)
    }

    // ------------------------------------ (j) the unjoined shared expert

    /// One of the file's matrices as the host reads it: its stored rows of
    /// `k` values, each `row_bytes` long.
    struct HostMat<'a> {
        ty: gguf::GgmlType,
        data: &'a [u8],
        row_bytes: usize,
        k: usize,
    }

    impl<'a> HostMat<'a> {
        /// Tensor `name` of `split`, whose first dim is `k` values.
        fn of(split: &'a Split, name: &str) -> Result<HostMat<'a>, GateError> {
            let (s, t) = split
                .find(name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            let data = split.shard(s).ok_or("shard index out of range")?.data(t)?;
            let k = usize::try_from(*t.dims.first().ok_or("a tensor of no dims")?)?;
            let rows = usize::try_from(t.dims.iter().skip(1).product::<u64>())?;
            if rows == 0 || data.len() % rows != 0 {
                return Err(format!("{name}: {} bytes over {rows} rows", data.len()).into());
            }
            Ok(HostMat {
                ty: t.ty,
                data,
                row_bytes: data.len() / rows,
                k,
            })
        }

        /// Rows `first..first + n`, each dequantized as stored, dotted with
        /// `x` in f64.
        fn dots(&self, first: usize, n: usize, x: &[f64]) -> Result<Vec<f64>, GateError> {
            let mut row = vec![0.0f32; self.k];
            (first..first + n)
                .map(|r| {
                    let bytes = self
                        .data
                        .get(r * self.row_bytes..(r + 1) * self.row_bytes)
                        .ok_or("a row past the tensor")?;
                    gguf::quant::dequant_row(self.ty, bytes, &mut row)
                        .map_err(|e| format!("dequant {}: {e}", self.ty))?;
                    Ok(row.iter().zip(x).map(|(&w, &v)| f64::from(w) * v).sum())
                })
                .collect()
        }
    }

    /// One expert's FFN on the host in f64: `down(silu(gate · x) · up · x)`
    /// over expert `e`'s rows of the three matrices (`ff` gate and up rows
    /// an expert, `hidden` down rows).
    fn host_expert(
        [g, u, d]: [&HostMat<'_>; 3],
        e: usize,
        x: &[f64],
        (ff, hidden): (usize, usize),
    ) -> Result<Vec<f64>, GateError> {
        let (gr, ur) = (g.dots(e * ff, ff, x)?, u.dots(e * ff, ff, x)?);
        let h: Vec<f64> = gr
            .iter()
            .zip(&ur)
            .map(|(&a, &b)| a / (1.0 + (-a).exp()) * b)
            .collect();
        d.dots(e * hidden, hidden, &h)
    }

    /// (j)'s wide rows: the decode run's rows cycled to twice the gemv arm's
    /// widest unit, so the FFN takes the wide arm's launches.
    const WIDE_ROWS: usize = 2 * (WIDE_FROM - 1);

    /// (j)'s FFN clause (module doc): each FFN type combination's first
    /// layer run alone on each arm against the host's f64 FFN under the
    /// card's routing.
    fn unjoined_ffn(
        m: &mut Qwen35moeModel,
        toks: &[u32],
        n_layer: usize,
    ) -> Result<bool, GateError> {
        tier::sc("(j) unjoined ffn: the shared expert's FFN half against the host's")?;
        let path = unjoined_path();
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let parts = |l: usize| {
            ["gate", "up", "down"].map(|p| {
                [
                    format!("blk.{l}.ffn_{p}_exps.weight"),
                    format!("blk.{l}.ffn_{p}_shexp.weight"),
                ]
            })
        };
        let (mut seen, mut layers) = (Vec::new(), Vec::new());
        for l in 0..n_layer {
            let tys: Vec<String> = parts(l)
                .iter()
                .flatten()
                .map(|n| split.find(n).map(|(_, t)| t.ty.to_string()))
                .collect::<Option<_>>()
                .ok_or_else(|| format!("layer {l}: an FFN matrix missing"))?;
            if !seen.contains(&tys) {
                seen.push(tys);
                layers.push(l);
            }
        }
        m.set_layer_taps(true)?;
        fresh(m)?;
        let mut rows: Vec<Vec<f32>> = vec![Vec::new(); n_layer];
        for &t in toks {
            m.step(&[t])?;
            for (r, tap) in rows.iter_mut().zip(m.layer_taps()?) {
                r.extend(tap);
            }
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        for (&l, tys) in layers.iter().zip(&seen) {
            // The rows the decode run left at the layer before (layer 0 its
            // own): rows at the chain's scale, which the FFN reads as any.
            let x = &rows[l.saturating_sub(1)];
            let cycled: Vec<f32> = x
                .chunks(HIDDEN)
                .cycle()
                .take(WIDE_ROWS)
                .flatten()
                .copied()
                .collect();
            let gain = split_f32(
                &split,
                &format!("blk.{l}.post_attention_norm.weight"),
                HIDDEN,
            )?;
            let [g, u, d] = parts(l);
            let mats = |i: usize| -> Result<[HostMat<'_>; 3], GateError> {
                Ok([
                    HostMat::of(&split, &g[i])?,
                    HostMat::of(&split, &u[i])?,
                    HostMat::of(&split, &d[i])?,
                ])
            };
            let (routed, shared) = (mats(0)?, mats(1)?);
            let ff = routed[0].data.len()
                / routed[0].row_bytes
                / (routed[2].data.len() / routed[2].row_bytes / HIDDEN);
            for (arm, x, f) in [
                ("gemv", x.as_slice(), m.ffn_rows(l, x)?),
                ("wide", cycled.as_slice(), m.ffn_rows_wide(l, &cycled)?),
            ] {
                let t_n = x.len() / HIDDEN;
                let slots = f.ids.len() / t_n;
                let xn = normed(x, &gain, HIDDEN);
                let (mut got, mut want) = (Vec::new(), Vec::new());
                for t in 0..t_n {
                    let xt: Vec<f64> = xn[t * HIDDEN..(t + 1) * HIDDEN]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect();
                    let mut acc = vec![0.0f64; HIDDEN];
                    for j in 0..slots {
                        let (id, w) = (
                            f.ids[t * slots + j] as usize,
                            f64::from(f.weights[t * slots + j]),
                        );
                        let (mats, e) = if j + 1 < slots {
                            (&routed, id)
                        } else {
                            (&shared, 0)
                        };
                        let y = host_expert([&mats[0], &mats[1], &mats[2]], e, &xt, (ff, HIDDEN))?;
                        for (a, v) in acc.iter_mut().zip(y) {
                            *a += w * v;
                        }
                    }
                    want.extend(acc.iter().map(|&v| v as f32));
                    got.extend(
                        f.l_out[t * HIDDEN..(t + 1) * HIDDEN]
                            .iter()
                            .zip(&x[t * HIDDEN..(t + 1) * HIDDEN])
                            .map(|(&o, &i)| o - i),
                    );
                }
                let err = rel_to(&got, &want);
                let pass = err <= UNJOINED_FFN_BAND;
                ok &= pass;
                println!(
                    "unjoined ffn layer={l} {tys:?} {arm} arm ({t_n} rows, {slots} slots a row): \
                     update vs the host's f64 FFN rel={err:.3e} (band {UNJOINED_FFN_BAND:.0e}) {}",
                    verdict(pass)
                );
            }
        }
        Ok(ok)
    }

    /// (j) (module doc): the unjoined file's plan, node counts and paths.
    fn unjoined(toks: &[u32]) -> Result<bool, GateError> {
        let Some((ud, proof)) = apart_clause(
            "(j) the unjoined shared expert: plan, decode, pass and the wide arm",
            true,
        )?
        else {
            return Ok(true);
        };
        let want = ud.apart();
        let (want_layers, want_decode, want_pass) =
            (ud.n_layer(), ud.nodes_decode(), ud.nodes_pass());
        let mut m = open_at(unjoined_path(), CTX, KvQ8::F16)?;
        let body = m.body("unjoined")?;
        let got = body.shared_apart();
        let (launch_1, launch_m) = (body.pass_launches(1), body.pass_launches(2));
        let n_apart = got.iter().filter(|a| **a).count();
        let mut ok = got == want && n_apart == want_layers;
        println!(
            "unjoined plan: shared experts kept apart on {n_apart} of {} layers (want {want_layers}, \
             the header's {} layers of a part of another type) {}",
            got.len(),
            want.iter().filter(|a| **a).count(),
            verdict(ok)
        );
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memcpy = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY;
        let nodes = m.capture_step()?;
        let ([k, c], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memcpy]);
        let pass = nodes == want_decode && k == nodes && launch_1 + 3 == nodes;
        println!(
            "unjoined decode graph_nodes={nodes} (want {want_decode}; the body counts {} + 3) \
             kernel={k} memcpy={c} other={other} {}",
            launch_1,
            verdict(pass)
        );
        ok &= pass;
        for (rows, nodes, list) in [
            (5usize, m.capture_rows::<5>()?, m.rows_graph_nodes::<5>()?),
            (8, m.capture_rows::<8>()?, m.rows_graph_nodes::<8>()?),
        ] {
            let ([k, c], other) = count_kinds(&list, [kernel, memcpy]);
            let want = want_pass + 4 * rows;
            let pass =
                nodes == want && c == rows && k == nodes - rows && launch_m + 4 * rows == nodes;
            println!(
                "unjoined pass m={rows} graph_nodes={nodes} (want {want}; the body counts {} + \
                 4·{rows}) kernel={k} memcpy={c} (want {rows}) other={other} {}",
                launch_m,
                verdict(pass)
            );
            ok &= pass;
        }
        let (p_ok, decode) = paths(&mut m, toks)?;
        ok &= p_ok;
        ok &= gemv_arm(&mut m, toks, &decode)?;
        ok &= unjoined_ffn(&mut m, toks, want_layers)?;
        let man = ik_set("(j) the step-1,024 set's prompt ids", Set::D1k)?;
        let (_, _, prompt) = man.step()?;
        let mut run = |cut: Option<usize>| -> Result<(u32, Vec<f32>, Vec<StoreHost>), GateError> {
            fresh(&mut m)?;
            let next = match cut {
                None => m.prefill_with(prompt, PrefillPath::Gemm)?,
                Some(c) => {
                    m.prefill_with(&prompt[..c], PrefillPath::Gemm)?;
                    m.prefill_with(&prompt[c..], PrefillPath::Gemm)?
                }
            };
            Ok((next, m.logits()?, stores(&mut m)?))
        };
        let (tok0, logits0, stores0) = run(None)?;
        let (tok, logits, st) = run(Some(W_CUT))?;
        let differ: Vec<usize> = (0..st.len())
            .filter(|&l| !stores0.get(l).is_some_and(|w| same_store(&st[l], w)))
            .collect();
        let same_logits = bits_equal(&logits, &logits0);
        let pass = tok == tok0 && same_logits && differ.is_empty() && st.len() == want_layers;
        println!(
            "unjoined wide arm: {} ids as one ubatch of {U_GATE} and as two calls of {W_CUT} and \
             {}: token {tok} vs {tok0}, last logits bit-identical={same_logits}, every store \
             bit-identical (layers differing {differ:?}) {}",
            prompt.len(),
            prompt.len() - W_CUT,
            verdict(pass)
        );
        ok &= pass;
        ok &= proof;
        println!("unjoined: {}", verdict(ok));
        Ok(ok)
    }

    // ---------------------------------------------------- (r) refusals

    fn refusals(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        tier::sc("(r) refusals: a ubatch size of 0 or past UBATCH, a prompt past the cache")?;
        let (u0, bytes0) = (m.ubatch()?, m.resident_bytes());
        let mut ok = true;
        for u in [0, UBATCH + 1] {
            let e = m.set_ubatch(u).err();
            let kept = m.ubatch()? == u0 && m.resident_bytes() == bytes0;
            let pass = e.is_some() && kept;
            println!(
                "refuse ubatch {u}: {} (size {u0} and resident bytes kept={kept}) {}",
                e.map_or("accepted".to_string(), |e| e.to_string()),
                verdict(pass)
            );
            ok &= pass;
        }
        fresh(m)?;
        let long = vec![0u32; CTX + 1];
        let e = m.prefill_with(&long, PrefillPath::Auto).err();
        let pass = e.is_some() && m.pos() == 0;
        println!(
            "refuse a prompt of {} ids on a {CTX}-row cache: {} (position {}) {}",
            long.len(),
            e.map_or("accepted".to_string(), |e| e.to_string()),
            m.pos(),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ---------------------------------------------- (o) the placed load

    /// (o)'s card budget in the real tier: a 12 GiB card's, under which the
    /// file's routed experts split between the card and the host tier.
    const REAL_PLACED_BUDGET: u64 = 12 << 30;

    /// (o)'s card budget: the real tier's [`REAL_PLACED_BUDGET`] unless the
    /// caller's plan lever names one; the fixture tier's is the budget the
    /// fixture's header records ([`tier::plan_levers`]), which splits the
    /// routed experts the same way on the fixture's card.
    fn placed_budget() -> u64 {
        cfg()
            .plan_levers
            .card_budget_bytes
            .every()
            .unwrap_or(REAL_PLACED_BUDGET)
    }

    /// What the memguard clause leaves free of the card while it holds the
    /// rest: well under the trunk's smallest term, so the refusal is
    /// certain.
    const MEMGUARD_LEFT: usize = 512 << 20;

    /// The memory guard on this file (`gate_qwen3moe_e2e`'s (m) is the
    /// clause's full form; this one holds the qwen35moe shapes of the same
    /// planner): the quiet-card decision is the whole-card load, holding all
    /// but [`MEMGUARD_LEFT`] of the free bytes refuses the plan on device 0
    /// by name and makes the unset decision the placed one, and the hold
    /// dropped restores the whole-card decision — the reading is live.
    /// FAIL-first: a census that reads the total as the free bytes (mutant:
    /// `raw_device_free_bytes` returning `cuDeviceTotalMem`'s figure) turns
    /// every arm red.
    fn memguard() -> Result<bool, GateError> {
        use bloomery_gpu_gates::gpu_census::census;
        if !tier::run_clause(
            "(m) memguard: the census's decisions on a quiet card and a held one",
            Tag::Scale,
        )? {
            return Ok(true);
        }
        let file = open_model()?;
        let d0 = census()?
            .into_iter()
            .find(|d| d.ordinal == 0)
            .ok_or("the census reads no device 0")?;
        println!(
            "memguard census: cuda0 {} free {} B",
            d0.name, d0.free_bytes
        );
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        let whole = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Whole(_)
        );
        println!("memguard quiet: whole decision {whole} {}", verdict(whole));
        let gpu = bloomery_gpu::Gpu::new()?;
        let (free, _) = gpu.mem_info()?;
        let hold = free
            .checked_sub(MEMGUARD_LEFT)
            .ok_or("the card had less free than the clause leaves")?;
        let held = cuda_core::DeviceBuffer::<u8>::zeroed(gpu.stream(), hold)?;
        let refusal = match q3place::PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?
            .plan(CTX, &PlanLevers::default())
        {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        let refused = refusal.contains("the plan's dense trunk alone needs");
        println!("memguard held: refusal {refused} {refusal}");
        let placed = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Placed(_)
        );
        println!(
            "memguard held: placed decision {placed} {}",
            verdict(placed)
        );
        drop(held);
        drop(gpu);
        let whole_again = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Whole(_)
        );
        println!(
            "memguard released: whole decision {whole_again} {}",
            verdict(whole_again)
        );
        Ok(whole && refused && placed && whole_again)
    }

    /// (o) (module doc): the model is dropped before the clause returns.
    fn placed(toks: &[u32], host: HostCfg) -> Result<bool, GateError> {
        tier::sc(
            "(o) the placed load: plan, scratch, graph steps, prompt call, refusals, host tier",
        )?;
        let file = open_model()?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?;
        let levers = PlanLevers {
            card_budget_bytes: Some(placed_budget()).into(),
        };
        let plan = q.plan(CTX, &levers)?;
        for line in q.records(&plan, None)? {
            line.print();
        }
        let split = plan.host.experts > 0 && plan.cards[0].experts > 0;
        println!(
            "placed plan: card_experts={} host_experts={} (both above 0) {}",
            plan.cards[0].experts,
            plan.host.experts,
            verdict(split)
        );
        if !split {
            return Ok(false);
        }
        // (o6)
        let scratch = GpuModel::<Body35>::placed_arena_bytes(&file, o)?;
        let card = plan.machine.cards.first().ok_or("a plan of no card")?;
        let counted = model::placement::workstation::counted_arena_bytes(card);
        let joined = shape().apart().iter().filter(|&&a| !a).count();
        let t = Instant::now();
        let mut m = GpuModel::<Body35>::open_placed(file, &plan, o, host)?;
        let apart = m.body("gate_qwen35moe_e2e")?.shared_apart();
        let every = apart.iter().all(|&a| a);
        let counts = scratch == counted;
        println!(
            "placed scratch: the plan counts {counted} B, the arena and placed side the planner \
             counts {scratch} B {}; every layer's shared expert kept apart on the placed load \
             {every} ({joined} of {} layers are folded in the file) {}",
            verdict(counts),
            n_layer(),
            verdict(every)
        );
        let mut ok_scratch = counts && every;
        let counts = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .card_counts();
        println!(
            "placed load resident_bytes={} ubatch={} card experts per layer {counts:?} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        m.set_mode(StepMode::Graph);
        println!("placed capture graph_nodes={}", m.capture_step()?);
        // (o1)
        let decode = decode_run(&mut m, toks, false)?;
        m.set_mode(StepMode::Eager);
        let eager = decode_run(&mut m, toks, false)?;
        let mut ok = same_run(
            "placed: five eager steps vs five graph replays",
            &eager,
            &decode,
        );
        // (o2)
        ok &= free(&mut m, "(o2) ", toks, &decode, FREE_PROBE_HOST)?;
        m.set_mode(StepMode::Graph);
        fresh(&mut m)?;
        let plan_p = m.prefill_plan(toks.len(), PrefillPath::Auto)?;
        let tok = m.prefill_with(toks, PrefillPath::Auto)?;
        let call = PathRun {
            logits: vec![m.logits()?],
            tokens: vec![tok],
            stores: stores(&mut m)?,
        };
        let PathRun {
            mut logits,
            mut tokens,
            stores: last_stores,
        } = decode;
        let fifth = PathRun {
            logits: logits.pop().into_iter().collect(),
            tokens: tokens.pop().into_iter().collect(),
            stores: last_stores,
        };
        ok &= same_run(
            &format!("placed: the prompt call ({plan_p}) vs the fifth graph step"),
            &call,
            &fifth,
        );
        // (o5)
        ok &= wide_arm(&mut m, "placed ", PLACED_UBATCH)?;
        // (o3)
        // PIN(2026-10-22): a ubatch resize stays refused (the arena is the
        // plan's count). A pass of two resident slots is the placed body's
        // one pass: each column's greedy ids are that prompt stepped in
        // turn. Was a named refusal of the pass.
        // FAIL-first: both columns served from one prompt, or the columns
        // swapped, reddens the id compare; `step_slots` still refused fails
        // the clause by name.
        let u0 = m.ubatch()?;
        let resize = m.set_ubatch(U_GATE).err();
        let kept = resize.is_some() && m.ubatch()? == u0;
        println!(
            "placed refusals: ubatches of {U_GATE}: {} (ubatch {u0} kept) {}",
            resize.map_or("accepted".to_string(), |e| e.to_string()),
            verdict(kept)
        );
        m.add_slots(2)?;
        // The second slot steps the lcg walk the (n) clause steps, over this
        // load's vocabulary, as many ids as `toks`: one construction on every
        // tier and fixture variant, never the ids of the set the first slot
        // steps (the fixture's batch set holds `TEXT_TOKENS`).
        let walk = lcg_ids(toks.len(), m.body("gate_qwen35moe_e2e")?.vocab());
        let pass = placed_two_slots(&mut m, toks, &walk)?;
        // (o4)
        let stats = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .hybrid()
            .stats();
        let served = stats.host_slots > 0 && stats.batch_host_slots > 0;
        println!(
            "placed host tier: step host_slots={} batch host_slots={} (both above 0) {}",
            stats.host_slots,
            stats.batch_host_slots,
            verdict(served)
        );
        ok_scratch &= kept && pass && served;
        Ok(ok && ok_scratch)
    }

    /// (o3)'s two slots: each prompt stepped alone on slot 0 from zero
    /// stores, then both in one pass. The greedy ids match column for
    /// column, in graph mode and in eager mode. The second prompt is the
    /// caller's lcg walk ([`lcg_ids`] over the load's vocabulary), as many
    /// ids as the first: the two differ on every tier and fixture variant,
    /// so a column swap is not the same ids.
    fn placed_two_slots(m: &mut Qwen35moeModel, a: &[u32], b: &[u32]) -> Result<bool, GateError> {
        const SLOT_STEPS: usize = 4;
        if a == b {
            return Err("(o3): the two prompts are the same".into());
        }
        let mut ok = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            m.set_mode(mode);
            let solo = |m: &mut Qwen35moeModel, prompt: &[u32]| -> Result<Vec<u32>, GateError> {
                m.select_slot(0)?;
                fresh(m)?;
                let mut ids = vec![m.prefill_with(prompt, PrefillPath::Pass)?];
                for t in 0..SLOT_STEPS {
                    ids.push(m.step(&[ids[t]])?);
                }
                Ok(ids)
            };
            let sa = solo(m, a)?;
            let sb = solo(m, b)?;
            m.select_slot(0)?;
            fresh(m)?;
            let mut ga = vec![m.prefill_with(a, PrefillPath::Pass)?];
            m.select_slot(1)?;
            fresh(m)?;
            let mut gb = vec![m.prefill_with(b, PrefillPath::Pass)?];
            let mut same = ga[0] == sa[0] && gb[0] == sb[0];
            for t in 0..SLOT_STEPS {
                let out = m.step_slots(&[(0, &[ga[t]]), (1, &[gb[t]])])?;
                let &[ia, ib] = out.ids.as_slice() else {
                    return Err(format!("(o3): a pass gave {:?}", out.ids).into());
                };
                same &= ia == sa[t + 1] && ib == sb[t + 1];
                ga.push(ia);
                gb.push(ib);
            }
            println!(
                "placed slots ({mode:?}): one pass of two slots vs each prompt stepped in turn {}",
                verdict(same)
            );
            ok &= same;
        }
        Ok(ok)
    }

    /// (o7) (module doc): the whole-card load's five greedy tokens first, its
    /// model dropped before the placed one opens (two whole-sized models
    /// never share a card), then the placed load by a plan with no host
    /// expert.
    fn placed_no_host(toks: &[u32], host: HostCfg) -> Result<bool, GateError> {
        tier::sc("(o7) the placed load with no host expert = the whole-card load")?;
        let whole = {
            let mut m = open(CTX, KvQ8::F16)?;
            decode_run(&mut m, toks, false)?.tokens
        };
        let file = open_model()?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?;
        let plan = q.plan(CTX, &PlanLevers::default())?;
        for line in q.records(&plan, None)? {
            line.print();
        }
        let on_card = plan.host.experts == 0 && plan.cards[0].experts > 0;
        println!(
            "placed plan with no budget: card_experts={} host_experts={} (every expert on the \
             card) {}",
            plan.cards[0].experts,
            plan.host.experts,
            verdict(on_card)
        );
        if !on_card {
            return Ok(false);
        }
        let mut m = q3place::open_qwen35(file, &plan, o, host)?;
        let placed = m.body("gate_qwen35moe_e2e")?.placed().is_some();
        println!(
            "placed no-host load has a placed side {placed} {}",
            verdict(placed)
        );
        if !placed {
            return Ok(false);
        }
        m.set_mode(StepMode::Graph);
        println!("placed no-host capture graph_nodes={}", m.capture_step()?);
        let decode = decode_run(&mut m, toks, false)?;
        let same = decode.tokens == whole;
        println!(
            "placed no-host tokens {:?} vs whole-card {whole:?} {}",
            decode.tokens,
            verdict(same)
        );
        let banded = free(&mut m, "(o7) ", toks, &decode, FREE_PROBE_CARD)?;
        Ok(same && banded)
    }

    /// (o8)'s wide unit is a unit of the wide arm: past the gemv arm's cut, and
    /// within the placed load's ubatch.
    const _: () = assert!(PLACED_UBATCH >= WIDE_FROM);

    /// `got`'s calls (each a greedy id and a logits row) against the
    /// whole-card load's `want` by (e)'s rule ([`e_band`]), a call at a time:
    /// the calls are fed the same ids, so each id stands alone and a part at
    /// any call must be a near tie, not only the first. Every row must also be
    /// finite (the band's distance does not see a NaN). The rule is not bit
    /// identity: a host expert runs ik's 8-bit rule where the card runs q8_1,
    /// so the two loads are two realizations of one rule.
    fn forced_calls(
        label: &str,
        got: (&[u32], &[Vec<f32>]),
        want: (&[u32], &[Vec<f32>]),
    ) -> Result<bool, GateError> {
        let n = got.0.len();
        if n == 0 || got.1.len() != n || want.0.len() != n || want.1.len() != n {
            println!(
                "{label}: {} ids and {} rows against {} ids and {} rows (the same count, above 0) \
                 {}",
                got.0.len(),
                got.1.len(),
                want.0.len(),
                want.1.len(),
                verdict(false)
            );
            return Ok(false);
        }
        let call = |(ids, rows): (&[u32], &[Vec<f32>]), i: usize| Seen {
            tokens: vec![ids[i]],
            rows: vec![rows[i].clone()],
            ..Seen::default()
        };
        let finite = got.1.iter().flatten().all(|v| v.is_finite());
        let mut ok = finite;
        let (mut worst, mut over, mut parted, mut far) = ((0usize, 0.0f64), vec![], vec![], vec![]);
        for i in 0..n {
            let band = e_band(&call(got, i), &call(want, i))?;
            ok &= band.ok();
            if band.worst.1 > worst.1 {
                worst = (i, band.worst.1);
            }
            if band.worst.1 > FREE_BAND {
                over.push(i);
            }
            if band.parted.is_some() {
                parted.push(i);
            }
            if !band.near {
                far.push(i);
            }
        }
        println!(
            "{label}: {n} calls' logits within {FREE_BAND} of the whole-card load's, relative: \
             worst {:.3e} at call {} (calls over the band {over:?}); rows finite {finite}; greedy \
             ids {:?} vs the whole-card load's {:?}, parting at calls {parted:?} (each a near tie, \
             margin under {GREEDY_MARGIN}: calls that are not {far:?}) {}",
            worst.1,
            worst.0,
            got.0,
            want.0,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (o8) (module doc): the whole-card load's references first, its model
    /// dropped before the placed one opens (two whole-sized models never share
    /// a card), then the placed load by the plan [`PlaceQ3::budget_for_card_experts`]
    /// finds.
    fn placed_unjoined(toks: &[u32], host: HostCfg) -> Result<bool, GateError> {
        let Some((ud, proof)) =
            apart_clause("(o8) the unjoined shared expert on the placed load", false)?
        else {
            return Ok(true);
        };
        let man = ik_set("(o8) the step-1,024 set's prompt ids", Set::D1k)?;
        let (_, _, prompt) = man.step()?;
        let wide = prompt.get(..PLACED_UBATCH).ok_or_else(|| {
            format!(
                "the step set's prompt holds {} ids, (o8)'s wide unit takes {PLACED_UBATCH}",
                prompt.len()
            )
        })?;
        let apart = ud.apart();
        let want_layers = ud.n_layer();
        let n_apart = apart.iter().filter(|&&a| a).count();
        let unjoined_file = n_apart == want_layers;
        println!(
            "placed unjoined file: shared experts of another type than their stacks' on {n_apart} \
             of {} layers (want {want_layers}) {}",
            apart.len(),
            verdict(unjoined_file)
        );
        if !unjoined_file {
            return Ok(false);
        }
        let (whole, whole_wide) = {
            let mut m = open_at(unjoined_path(), CTX, KvQ8::F16)?;
            m.set_mode(StepMode::Graph);
            let steps = decode_run(&mut m, toks, false)?;
            fresh(&mut m)?;
            let plan_w = m.prefill_plan(wide.len(), PrefillPath::Gemm)?;
            let next = m.prefill_with(wide, PrefillPath::Gemm)?;
            println!(
                "placed unjoined whole-card reference: tokens {:?}; the prompt of {} ids \
                 (plan={plan_w}) token {next}",
                steps.tokens,
                wide.len()
            );
            (steps, (next, m.logits()?))
        };
        let file = Split::open(unjoined_path())
            .map_err(|e| format!("open {}: {e}", unjoined_path().display()))?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?;
        let (want, top) = (u64::try_from(want_layers / 2)?, placed_budget());
        let budget = q.budget_for_card_experts(CTX, want, placed_budget())?;
        let levers = PlanLevers {
            card_budget_bytes: Some(budget).into(),
        };
        let plan = q.plan(CTX, &levers)?;
        for line in q.records(&plan, None)? {
            line.print();
        }
        let card_experts = plan.cards.first().ok_or("a plan of no card")?.experts;
        let split = plan.host.experts > 0 && card_experts >= want;
        println!(
            "placed unjoined plan: card budget {budget} B (the smallest under {top} B \
             that plans {want} card experts or more) card_experts={card_experts} host_experts={} \
             (both above 0) card experts per layer {:?} {}",
            plan.host.experts,
            plan.n_l,
            verdict(split)
        );
        if !split {
            return Ok(false);
        }
        // (o6)
        let scratch = GpuModel::<Body35>::placed_arena_bytes(&file, o)?;
        let card = plan.machine.cards.first().ok_or("a plan of no card")?;
        let counted = model::placement::workstation::counted_arena_bytes(card);
        let t = Instant::now();
        let mut m = GpuModel::<Body35>::open_placed(file, &plan, o, host)?;
        let every = m
            .body("gate_qwen35moe_e2e")?
            .shared_apart()
            .iter()
            .all(|&a| a);
        let counts = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .card_counts();
        let none = counts.iter().filter(|&&n| n == 0).count();
        let some = counts.len() - none;
        println!(
            "placed unjoined load resident_bytes={} ubatch={} card experts per layer {counts:?} \
             in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        let reason = match (none, some) {
            (0, _) => ": the plan holds no layer with 0 card experts",
            (_, 0) => ": the plan holds no layer with card experts",
            _ => "",
        };
        let mixed = reason.is_empty();
        println!(
            "placed unjoined layers: {none} with 0 card experts and {some} with card experts \
             (both above 0){reason} {}",
            verdict(mixed)
        );
        let counts_ok = scratch == counted;
        println!(
            "placed unjoined scratch: the plan counts {counted} B, the arena and placed side the \
             planner counts {scratch} B {}; every layer's shared expert kept apart on the placed \
             load {every} {}",
            verdict(counts_ok),
            verdict(every)
        );
        if !(mixed && counts_ok && every) {
            return Ok(false);
        }
        m.set_mode(StepMode::Graph);
        println!("placed unjoined capture graph_nodes={}", m.capture_step()?);
        // The gemv arm: the decode step.
        let decode = decode_run(&mut m, toks, false)?;
        let mut ok = forced_calls(
            "placed unjoined gemv arm, the batch set's steps",
            (&decode.tokens, &decode.logits),
            (&whole.tokens, &whole.logits),
        )?;
        // The gemv arm: a unit within its cut, against the fifth step.
        fresh(&mut m)?;
        let plan_p = m.prefill_plan(toks.len(), PrefillPath::Auto)?;
        let tok = m.prefill_with(toks, PrefillPath::Auto)?;
        let call = PathRun {
            logits: vec![m.logits()?],
            tokens: vec![tok],
            stores: stores(&mut m)?,
        };
        let PathRun {
            mut logits,
            mut tokens,
            stores: last_stores,
        } = decode;
        let fifth = PathRun {
            logits: logits.pop().into_iter().collect(),
            tokens: tokens.pop().into_iter().collect(),
            stores: last_stores,
        };
        ok &= same_run(
            &format!("placed unjoined: the prompt call ({plan_p}) vs the fifth graph step"),
            &call,
            &fifth,
        );
        // The wide arm: one unit past the gemv arm's cut.
        fresh(&mut m)?;
        let plan_w = m.prefill_plan(wide.len(), PrefillPath::Gemm)?;
        let next = m.prefill_with(wide, PrefillPath::Gemm)?;
        let next_logits = m.logits()?;
        ok &= forced_calls(
            &format!(
                "placed unjoined wide arm, {} ids (plan={plan_w})",
                wide.len()
            ),
            (&[next], std::slice::from_ref(&next_logits)),
            (&[whole_wide.0], std::slice::from_ref(&whole_wide.1)),
        )?;
        ok &= proof;
        println!("placed unjoined: {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------- (o9) the common card leg on the placed load

    /// The layers of `file` whose three routed stacks the common card leg runs
    /// ([`LegSplit::Common`]: a type the dispatch's launches do not), with the
    /// stacks' file types, read from the header.
    fn common_layers(file: &Split) -> Result<Vec<(usize, [GgmlType; 3])>, GateError> {
        let mut out = Vec::new();
        for l in 0..n_layer() {
            let mut tys = [GgmlType::F32; 3];
            for (ty, name) in tys.iter_mut().zip(model::arch::names::routed_stacks(l)) {
                let (_, t) = file
                    .find(&name)
                    .ok_or_else(|| format!("no tensor {name}"))?;
                *ty = t.ty;
            }
            if accepts::qwen35_leg(tys) == Some(LegSplit::Common) {
                out.push((l, tys));
            }
        }
        Ok(out)
    }

    /// What one placed load of (o9) saw: its plan, its layers' card experts, the
    /// arena the planner counts against the one the load counts, five graph
    /// steps with each layer's output, the prompt call against the fifth step,
    /// a wide unit.
    struct CommonRun {
        /// The plan's card experts and host experts.
        experts: (u64, u64),
        /// The card experts each layer holds.
        counts: Vec<usize>,
        /// The plan's counted arena and placed side, and the load's.
        scratch: (u64, u64),
        /// Five graph steps over the batch tokens.
        steps: PathRun,
        /// The layers' output rows of each of the five steps, `[step][layer]`.
        taps: Vec<Vec<Vec<f32>>>,
        /// The prompt call (a unit within the gemv arm's cut) left the fifth
        /// step's bits.
        call_same: bool,
        /// The wide unit's greedy id and logits.
        wide: (u32, Vec<f32>),
        /// The wide unit's card route read back with the plan's slot map rows
        /// ([`common_route`]); the reference holds the empty tables of its
        /// no-card layers.
        route: Option<(WideRoute, Vec<Vec<u32>>)>,
        /// The host tier's slots on the step and the batch ports.
        host_slots: (u64, u64),
    }

    /// One placed load of (o9) at the card budget `pick` names for the file's
    /// planner, run over `toks` (the gemv arm) and a wide unit of an lcg's ids.
    fn common_chain(
        label: &str,
        pick: impl FnOnce(&PlaceQ3) -> Result<u64, GateError>,
        toks: &[u32],
        host: HostCfg,
    ) -> Result<CommonRun, GateError> {
        let file = open_model()?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?;
        let budget = pick(&q)?;
        let levers = PlanLevers {
            card_budget_bytes: Some(budget).into(),
        };
        let plan = q.plan(CTX, &levers)?;
        for line in q.records(&plan, None)? {
            line.print();
        }
        let experts = (
            plan.cards.first().ok_or("a plan of no card")?.experts,
            plan.host.experts,
        );
        let card = plan.machine.cards.first().ok_or("a plan of no card")?;
        let scratch = (
            model::placement::workstation::counted_arena_bytes(card),
            GpuModel::<Body35>::placed_arena_bytes(&file, o)?,
        );
        let t = Instant::now();
        let mut m = GpuModel::<Body35>::open_placed(file, &plan, o, host)?;
        let counts = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .card_counts();
        println!(
            "placed common {label}: card budget {budget} B card_experts={} host_experts={} card \
             experts per layer {counts:?} resident_bytes={} in {:.1} s (runtime value)",
            experts.0,
            experts.1,
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        // The gemv arm: five graph steps, then the same five eager with each
        // layer's output kept.
        m.set_mode(StepMode::Graph);
        println!(
            "placed common {label} capture graph_nodes={}",
            m.capture_step()?
        );
        let mut steps = decode_run(&mut m, toks, false)?;
        m.set_mode(StepMode::Eager);
        m.set_layer_taps(true)?;
        fresh(&mut m)?;
        let mut taps = Vec::new();
        for &tok in toks {
            m.step(&[tok])?;
            taps.push(m.layer_taps()?);
        }
        m.set_layer_taps(false)?;
        // The prompt call against the fifth step.
        m.set_mode(StepMode::Graph);
        fresh(&mut m)?;
        let plan_p = m.prefill_plan(toks.len(), PrefillPath::Auto)?;
        let tok = m.prefill_with(toks, PrefillPath::Auto)?;
        let call = PathRun {
            logits: vec![m.logits()?],
            tokens: vec![tok],
            stores: stores(&mut m)?,
        };
        let fifth = PathRun {
            logits: steps.logits.last().cloned().into_iter().collect(),
            tokens: steps.tokens.last().copied().into_iter().collect(),
            stores: std::mem::take(&mut steps.stores),
        };
        let call_same = same_run(
            &format!("placed common {label}: the prompt call ({plan_p}) vs the fifth graph step"),
            &call,
            &fifth,
        );
        // The wide arm: one unit past the gemv arm's cut, then its ids as steps.
        let vocab = m.body("gate_qwen35moe_e2e")?.vocab();
        let ids = lcg_ids(PLACED_UBATCH, vocab);
        fresh(&mut m)?;
        let plan_w = m.prefill_plan(ids.len(), PrefillPath::Gemm)?;
        let next = m.prefill_with(&ids, PrefillPath::Gemm)?;
        let wide = (next, m.logits()?);
        println!(
            "placed common {label}: wide unit of {} ids (plan={plan_w}) token {next}",
            ids.len()
        );
        let stats = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .hybrid()
            .stats();
        // (o9r)'s readback: the wide unit's card route, with the plan's slot
        // map rows (the map the wide arm reads, the shared expert's HOST
        // entry after each) as their oracle; both loads read back (the
        // host-only reference holds the empty tables of its no-card layers).
        let route = match m.body("gate_qwen35moe_e2e")?.wide_route(m.gpu().stream())? {
            Some(r) => {
                let n_expert = r
                    .layers
                    .first()
                    .ok_or("a map read back for every layer")?
                    .map
                    .len()
                    .checked_sub(1)
                    .ok_or("a map of no expert")?;
                let tiers: Vec<usize> = (0..plan.machine.expert_tiers().count())
                    .map(|i| plan.machine.cards.len() + i)
                    .collect();
                let rows = SlotMap::of_plan_tiers(&plan, 0, &tiers, 0..n_layer(), n_expert)
                    .map_err(|e| e.to_string())?
                    .stage_view();
                let rows: Vec<Vec<u32>> = rows
                    .chunks(n_expert)
                    .map(|row| row.iter().copied().chain([HOST]).collect())
                    .collect();
                Some((r, rows))
            }
            None => None,
        };
        Ok(CommonRun {
            experts,
            counts,
            scratch,
            steps,
            taps,
            call_same,
            wide,
            route,
            host_slots: (stats.host_slots, stats.batch_host_slots),
        })
    }

    /// The routed table the wide arm's remap builds from `ids` through `map`:
    /// the slots whose map value is a card slot below `n_card`, grouped by
    /// that slot (ascending slot order within an expert, the ids' order),
    /// cut into tiles of at most [`GEMM_BN`], and the refused count (an id
    /// past the map, or a map value at or past the card set that is not
    /// [`HOST`]). (o9r)'s oracle, the rule `gemm_route_remap` states.
    fn oracle_route(ids: &[u32], map: &[u32], n_card: usize) -> (Vec<u32>, Vec<GemmTile>, usize) {
        let mut per: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::default();
        let mut refused = 0usize;
        for (s, &id) in ids.iter().enumerate() {
            match map.get(id as usize).copied() {
                Some(HOST) => {}
                Some(v) if (v as usize) < n_card => per.entry(v).or_default().push(s as u32),
                _ => refused += 1,
            }
        }
        let mut cols = Vec::new();
        let mut tiles = Vec::new();
        for (&e, slots) in &per {
            for chunk in slots.chunks(GEMM_BN) {
                tiles.push(GemmTile {
                    expert: e,
                    start: cols.len() as u32,
                    len: chunk.len() as u32,
                });
                cols.extend_from_slice(chunk);
            }
        }
        (cols, tiles, refused)
    }

    /// (o9r) (module doc): the wide arm's card route after (o9)'s wide unit,
    /// read back against the plan's own: every layer's map is the plan's slot
    /// map row for it with [`HOST`] after it (the shared expert's id, which
    /// the remap skips) — the per-layer cut a plan of unequal layer counts
    /// stresses —, every layer's table refused no slot, every
    /// table's tiles tile its slot list in order (each window inside the
    /// list, its slots ascending, its expert below the layer's card count,
    /// its length 1..=[`GEMM_BN`]), and the last walked layer's table and
    /// places are the oracle's ([`oracle_route`] over the unit's ids and that
    /// layer's map; each slot's routed place the map at its id, its shared
    /// slot [`HOST`]). The oracle needs the unit's ids, which the arena holds
    /// only for the last layer that walked (each layer's router overwrites
    /// them), so the table oracle is the last layer's alone; the maps of
    /// every layer and the tables' shape are held for all. FAIL-first: the
    /// layer's card route reading another layer's map (mutant: `card_wide`
    /// passing `side.maps[0]` for every layer) reddens the map of every layer
    /// past the first whose count differs, and the remap's `n_experts` launch
    /// argument one lower (mutant: `enqueue_route_remap`'s `route.n_experts`
    /// read as `route.n_experts − 1`) reddens the last layer's table, its
    /// top expert's slots listed as refused.
    fn common_route(route: &Option<(WideRoute, Vec<Vec<u32>>)>) -> Result<bool, GateError> {
        tier::sc(
            "(o9r) the wide arm's card route against the plan's own (the maps, the last layer's \
             table)",
        )?;
        let Some((route, rows)) = route else {
            println!("placed common route: no placed side read back FAIL");
            return Ok(false);
        };
        let (mut maps, mut no_refused, mut shape) = (true, true, true);
        for (l, layer) in route.layers.iter().enumerate() {
            maps &= rows
                .get(l)
                .is_some_and(|row| row.as_slice() == layer.map.as_slice());
            no_refused &= layer.refused == 0;
            let mut at = 0usize;
            let mut last_e = None;
            for t in &layer.tiles {
                let (s, n) = (t.start as usize, t.len as usize);
                shape &= s == at
                    && (1..=GEMM_BN).contains(&n)
                    && (t.expert as usize) < layer.n_card
                    && last_e.is_none_or(|e| e <= t.expert)
                    && layer
                        .cols
                        .get(s..s + n)
                        .is_some_and(|w| w.windows(2).all(|p| p[0] < p[1]));
                at = s + n;
                last_e = Some(t.expert);
            }
            shape &= at <= layer.cols.len();
        }
        let last = route.layers.last().ok_or("a layer's route read back")?;
        let (cols, tiles, refused) = oracle_route(&route.ids, &last.map, last.n_card);
        // The table's slot list is the walk's buffer: its first `cols.len()`
        // entries are the routed slots, the rest capacity no tile reads.
        let table = last.cols.get(..cols.len()) == Some(cols.as_slice())
            && last.tiles == tiles
            && refused == last.refused;
        if !table {
            let col_at = last.cols.iter().zip(&cols).position(|(a, b)| a != b);
            let tile_at = last.tiles.iter().zip(&tiles).position(|(a, b)| a != b);
            println!(
                "placed common route: the last layer's table against the oracle's: slots {} vs {} \
                 (first apart {col_at:?}: {:?} vs {:?}), tiles {} vs {} (first apart {tile_at:?}: \
                 {:?} vs {:?}), refused {} vs {refused}",
                last.cols.len(),
                cols.len(),
                col_at.map(|i| &last.cols[i..(i + 4).min(last.cols.len())]),
                col_at.map(|i| &cols[i..(i + 4).min(cols.len())]),
                last.tiles.len(),
                tiles.len(),
                tile_at.map(|i| &last.tiles[i]),
                tile_at.map(|i| &tiles[i]),
                last.refused,
            );
        }
        let pitch = route.ids.len().div_ceil(PLACED_UBATCH);
        let used = pitch
            .checked_sub(1)
            .ok_or("a router pitch of no routed slot")?;
        let sized = route.ids.len() == PLACED_UBATCH * pitch && route.sel.len() == route.ids.len();
        let places = sized
            && route
                .ids
                .iter()
                .zip(&route.sel)
                .enumerate()
                .all(|(s, (&id, &p))| {
                    (s % pitch < used && last.map.get(id as usize) == Some(&p))
                        || (s % pitch == used && p == HOST)
                });
        let ok = maps && no_refused && shape && table && places;
        println!(
            "placed common route: {} layers' maps the plan's rows {}, no slot refused {}, the \
             tables' shape held {}, the last layer's table the oracle's ({} tiles of {} slots) \
             {}, its places the map's {} {}",
            route.layers.len(),
            verdict(maps),
            verdict(no_refused),
            verdict(shape),
            tiles.len(),
            cols.len(),
            verdict(table),
            verdict(places),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (o9) (module doc): the file's placed load with no card expert (the host
    /// tier's chain, ik's 8-bit rule) first, then its placed load at (o)'s
    /// budget, whose card runs every layer's experts by the leg its types name
    /// ([`accepts::qwen35_leg`]); each load opened and dropped alone.
    fn placed_common(host: HostCfg) -> Result<bool, GateError> {
        tier::sc(
            "(o9) the common card leg on the placed load: the host tier's chain, the card's, the \
             wide arm",
        )?;
        let file = open_model()?;
        let common = common_layers(&file)?;
        drop(file);
        let variant = refset::fixture::selected()?.is_some();
        let premise = !variant || !common.is_empty();
        println!(
            "placed common file: {} of {} layers hold stacks the common card leg runs {:?} {}",
            common.len(),
            n_layer(),
            common.iter().map(|(l, t)| (*l, *t)).collect::<Vec<_>>(),
            verdict(premise)
        );
        if !premise {
            return Ok(false);
        }
        let toks = TEXT_TOKENS;
        let want = common_chain(
            "host-only plan",
            |q| q.budget_for_card_experts(CTX, 0, placed_budget()),
            &toks,
            host,
        )?;
        let got = common_chain("plan", |_| Ok(placed_budget()), &toks, host)?;
        let mut ok = true;
        // The plans: the reference leaves (about) nothing on the card, the run
        // splits the experts, and both count the arena the load made.
        let split = got.experts.0 > 0 && got.experts.1 > 0;
        let lean = want.experts.0 * 8 <= got.experts.0;
        let counted = want.scratch.0 == want.scratch.1 && got.scratch.0 == got.scratch.1;
        println!(
            "placed common plans: reference card_experts={} host_experts={}, run card_experts={} \
             host_experts={}; the reference holds at most an eighth of the run's card experts {} \
             and the run splits {}; the planner's counted scratch is the load's: reference {} B \
             = {} B, run {} B = {} B {}",
            want.experts.0,
            want.experts.1,
            got.experts.0,
            got.experts.1,
            verdict(lean),
            verdict(split),
            want.scratch.0,
            want.scratch.1,
            got.scratch.0,
            got.scratch.1,
            verdict(counted)
        );
        ok &= lean && split && counted;
        // The legs ran: every common layer with card experts is one the run
        // carded, and a variant has one.
        let carded: Vec<usize> = common
            .iter()
            .map(|&(l, _)| l)
            .filter(|&l| got.counts.get(l).is_some_and(|&n| n > 0))
            .collect();
        let ran = !variant || !carded.is_empty();
        println!(
            "placed common legs: {} of the file's {} common layers hold card experts in the run \
             (layers {carded:?}) {}; host slots on the run's step and batch ports {:?} (both \
             above 0) {}",
            carded.len(),
            common.len(),
            verdict(ran),
            got.host_slots,
            verdict(got.host_slots.0 > 0 && got.host_slots.1 > 0)
        );
        ok &= ran && got.host_slots.0 > 0 && got.host_slots.1 > 0;
        // The gemv arm: the prompt call is the fifth step's bits on each load;
        // the steps and the layers' outputs against the host tier's.
        ok &= want.call_same && got.call_same;
        ok &= forced_calls(
            "placed common gemv arm, the batch tokens' steps against the host tier's chain",
            (&got.steps.tokens, &got.steps.logits),
            (&want.steps.tokens, &want.steps.logits),
        )?;
        // Each layer's output of each step against the host tier's: layer l
        // carries l + 1 layers of the host tier's arithmetic, so its bound is
        // (l + 1) times one layer's. A near-tie router flip moves one token's
        // expert set from its layer on, and a wiring fault moves every token's
        // rows, so the rows over their bound may belong to one token alone.
        let mut over = std::collections::BTreeSet::new();
        let mut first_clean = true;
        for l in 0..want.taps[0].len() {
            let per_token: Vec<f64> = got
                .taps
                .iter()
                .zip(&want.taps)
                .map(|(g, w)| rel_to(&g[l], &w[l]))
                .collect();
            let bound = (l + 1) as f64 * FREE_PROBE_HOST;
            let exceed: Vec<usize> = (0..per_token.len())
                .filter(|&t| per_token[t] > bound || per_token[t].is_nan())
                .collect();
            over.extend(exceed.iter().copied());
            first_clean &= l > 0 || exceed.is_empty();
            println!(
                "placed common layer={l} l_out_rel={:.3e} (bound {bound:.1e}, {} layers of the \
                 host tier's arithmetic) per step {:?}; steps over the bound {exceed:?}",
                per_token.iter().copied().fold(0.0f64, worse),
                l + 1,
                per_token
                    .iter()
                    .map(|e| format!("{e:.2e}"))
                    .collect::<Vec<_>>(),
            );
        }
        let layers_ok = over.len() <= 1 && first_clean;
        println!(
            "placed common layers: the steps over their bound are {over:?} (at most one step, and \
             none at layer 0, whose router sees the same input in both loads: a router flip moves \
             one token's experts, a fault every token's) {}",
            verdict(layers_ok)
        );
        ok &= layers_ok;
        // The wide arm: a unit past the gemv arm's cut against the host tier's
        // wide unit over the same ids.
        ok &= forced_calls(
            "placed common wide arm against the host tier's",
            (&[got.wide.0], std::slice::from_ref(&got.wide.1)),
            (&[want.wide.0], std::slice::from_ref(&want.wide.1)),
        )?;
        let (rel, _) = e_distance(&got.wide.1, &want.wide.1);
        let tight = rel <= COMMON_WIDE_REL;
        println!(
            "placed common wide arm: logits rel {rel:.3e} against the host tier's wide unit \
             (bound {COMMON_WIDE_REL:.0e}) {}",
            verdict(tight)
        );
        ok &= tight;
        // (o9r): the route the wide arm ran, against the plan's own.
        ok &= common_route(&got.route)?;
        println!("placed common: {}", verdict(ok));
        Ok(ok)
    }

    const NAME: &str = "gate_qwen35moe_e2e";

    // ------------------------------------- (e) residency on the placed load

    /// (e)'s history, from a clear: the first [`E_P1`] ids of the step set's
    /// prompt (one unit of the wide arm), [`E_STEPS1`] greedy steps — the
    /// rule plans every four kept rows, so the flips land here —, the next
    /// [`E_P2`] ids (a wide unit whose maps must follow the flips, the call
    /// the load-time maps were wrong for), and [`E_STEPS2`] steps. At the
    /// gate's [`CTX`].
    const E_P1: usize = PLACED_UBATCH;
    const E_STEPS1: usize = 64;
    const E_P2: usize = 256;
    const E_STEPS2: usize = 16;
    const _: () = assert!(E_P1 + E_STEPS1 + E_P2 + E_STEPS2 < CTX);
    const _: () = assert!(E_P2 >= WIDE_FROM && E_P2 <= PLACED_UBATCH);

    /// The `split` clause's rule costs: the family's, with a host column a
    /// thousand microseconds, so the stream's floor sits under one column and
    /// each card layer streams the host experts its units route, up to a
    /// half of the ring and the balance's cut.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1000.0,
        ..Body35::XSTREAM_COSTS
    };

    /// What one run of (e)'s history saw: every call's greedy id and logits
    /// row in order (the first prompt's, each step's, the second prompt's,
    /// each step's), the flips that landed at the boundaries before the
    /// second prompt, and after each wide prompt the layers whose map is not
    /// the card copy's row ([`Placed::stale_maps`]).
    #[derive(Default)]
    struct Seen {
        tokens: Vec<u32>,
        rows: Vec<Vec<f32>>,
        landed: usize,
        stale: [Vec<usize>; 2],
        /// Each prompt call's placement and streaming records.
        calls: [Call; 2],
    }

    /// One prompt call of (e)'s history: every layer's live ids before it
    /// and after it (none on a load with no machine), the flips its
    /// boundary landed (the second call's: the steps' plans land there, so
    /// its start set is `before` with that many experts swapped), and its
    /// streaming picks, each with its walk, and its end (none unless it
    /// streamed).
    #[derive(Default)]
    struct Call {
        before: Vec<Vec<u32>>,
        after: Vec<Vec<u32>>,
        landed: usize,
        picks: Vec<(usize, CallPick)>,
        end: Option<CallReport>,
        /// The layers the call streamed under `split`, each with its walk.
        xlayers: Vec<(usize, XLayer)>,
    }

    /// The step set's prompt ids (D1K): (e)'s two prompts' ids.
    fn e_ids() -> Result<Vec<u32>, GateError> {
        let man = ik_set("(e) the step-1,024 set's prompt ids", Set::D1k)?;
        let (_, _, prompt) = man.step()?;
        let ids: Vec<u32> = prompt.iter().copied().take(E_P1 + E_P2).collect();
        if ids.len() != E_P1 + E_P2 {
            return Err(format!(
                "the step set's prompt holds {} ids, (e)'s two prompts take {}",
                ids.len(),
                E_P1 + E_P2
            )
            .into());
        }
        Ok(ids)
    }

    /// (o)'s plan opened as a session, under the residency the plan's card
    /// experts give ([`residency_clauses::mid_of`]) when `on`, else off: the
    /// session and the residency it ran (`None` when the plan leaves a layer
    /// no room for the machine).
    fn e_open(host: HostCfg, on: bool) -> Result<(Session<Body35>, Option<Residency>), GateError> {
        let file = open_model()?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
            media: false,
        };
        // Both loads take the residency load's plan, which streams its prompt calls under (e)'s
        // `split` and so keeps the lane's staging on the host: the off load runs the same plan.
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?.streaming()?;
        let levers = PlanLevers {
            card_budget_bytes: Some(placed_budget()).into(),
        };
        let plan = q.plan(CTX, &levers)?;
        let mid = residency_clauses::mid_of(&plan.n_l);
        let residency = match (on, mid) {
            (false, _) => Residency::Off,
            (true, Some(r)) => r,
            (true, None) => {
                return Err(
                    "the plan's card experts leave no layer with room for the machine's \
                            pinned, spare and one that moves"
                        .into(),
                );
            }
        };
        let mut m = GpuModel::<Body35>::open_placed_residency(file, &plan, o, host, residency)?;
        m.body_parts(NAME)?
            .2
            .host()
            .ok_or("an open_placed load with no placed side")?
            .log_residency(0);
        Ok((Session::from_model(m, CTX as u32), mid))
    }

    /// The residency boundaries' reports since the last take.
    fn e_take(s: &mut Session<Body35>) -> Result<Vec<(PassKind, PassReport)>, GateError> {
        let (_, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.host()
            .ok_or("an open_placed load with no placed side")?
            .take_residency_passes())
    }

    /// Every layer's live ids as the residency machine's ledger holds them
    /// ([`residency_clauses::live_sets`]); none on a load with no machine.
    fn e_live(s: &Session<Body35>) -> Result<Vec<Vec<u32>>, GateError> {
        let placed = s
            .model()
            .body(NAME)?
            .placed()
            .ok_or("an open_placed load with no placed side")?;
        Ok(placed.hybrid().swap().map_or_else(Vec::new, |m| {
            residency_clauses::live_sets(m, placed.hybrid().slots().layers())
        }))
    }

    /// The prompt calls' host streaming from the next call on
    /// (`Body35::set_xstream`).
    fn e_stream(s: &mut Session<Body35>, mode: XMode) -> Result<(), GateError> {
        let (gpu, _, b) = s.model_mut().body_parts(NAME)?;
        Ok(b.set_xstream(gpu, mode)?)
    }

    /// The layers whose wide map is not the card copy's row now.
    fn e_stale(s: &Session<Body35>) -> Result<Vec<usize>, GateError> {
        let m = s.model();
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("an open_placed load with no placed side")?;
        Ok(placed.stale_maps(m.gpu().stream())?)
    }

    /// Note a call's greedy id and logits row in `seen`; the id.
    fn e_note(seen: &mut Seen, out: Out<'_>) -> Result<u32, GateError> {
        let id = out.argmax();
        let Out::Logits { row, .. } = out else {
            return Err("a call returned no logits row".into());
        };
        seen.tokens.push(id);
        seen.rows.push(row.to_vec());
        Ok(id)
    }

    /// One run of (e)'s history on `s` from a clear (the residency back to
    /// its seed): each step feeds the id `forced` holds for the call before
    /// it, or its own greedy id when `forced` is `None`.
    fn e_history(
        s: &mut Session<Body35>,
        ids: &[u32],
        forced: Option<&[u32]>,
    ) -> Result<Seen, GateError> {
        s.clear()?;
        e_take(s)?;
        let mut seen = Seen::default();
        let prompt = |s: &mut Session<Body35>, seen: &mut Seen, i: usize, ids: &[u32]| {
            let before = e_live(s)?;
            let out = s.prompt(ids, Want::Logits)?;
            e_note(seen, out)?;
            seen.stale[i] = e_stale(s)?;
            let (picks, end) = s.model_mut().body_parts(NAME)?.2.take_stream_records();
            let (xlayers, _) = s.model_mut().body_parts(NAME)?.2.take_xstream_records();
            // The first call's boundary is the clear's quiet one, and its
            // reports stay for the steps' count.
            let landed = match i {
                0 => 0,
                _ => e_take(s)?.iter().map(|(_, r)| r.landed).sum(),
            };
            seen.calls[i] = Call {
                before,
                after: e_live(s)?,
                landed,
                picks,
                end,
                xlayers,
            };
            Ok::<(), GateError>(())
        };
        prompt(s, &mut seen, 0, &ids[..E_P1])?;
        let steps = |s: &mut Session<Body35>, seen: &mut Seen, n: usize| {
            for _ in 0..n {
                let at = seen.tokens.len() - 1;
                let id = forced.map_or(seen.tokens[at], |f| f[at]);
                let out = s.step(id, Want::Logits)?;
                e_note(seen, out)?;
            }
            Ok::<(), GateError>(())
        };
        steps(s, &mut seen, E_STEPS1)?;
        seen.landed = e_take(s)?.iter().map(|(_, r)| r.landed).sum();
        prompt(s, &mut seen, 1, &ids[E_P1..])?;
        steps(s, &mut seen, E_STEPS2)?;
        e_take(s)?;
        Ok(seen)
    }

    /// Whether two runs of the history left the same ids and the same logits
    /// bit for bit.
    fn e_same(a: &Seen, b: &Seen) -> bool {
        a.tokens == b.tokens
            && a.landed == b.landed
            && a.rows.len() == b.rows.len()
            && a.rows.iter().zip(&b.rows).all(|(x, y)| {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
            })
    }

    /// How far a logits row `a` stands from `b`: ‖a − b‖ / ‖b‖ (f64 sums) and
    /// the largest single |a − b|.
    fn e_distance(a: &[f32], b: &[f32]) -> (f64, f32) {
        let (mut d2, mut n2, mut top) = (0.0f64, 0.0f64, 0.0f32);
        for (&x, &y) in a.iter().zip(b) {
            let d = f64::from(x) - f64::from(y);
            d2 += d * d;
            n2 += f64::from(y) * f64::from(y);
            top = top.max((x - y).abs());
        }
        ((d2 / n2).sqrt(), top)
    }

    /// (e): the placed load under the residency machine (`open_placed_residency`
    /// at the `mid` word [`residency_clauses::mid_of`] gives the plan) against
    /// the same plan with it off, through the session's calls — the prompt is
    /// one residency pass, a step its own boundary. Both runs take (e)'s
    /// history ([`e_history`]) on the step set's prompt, the residency run
    /// feeding the off run's ids, each model dropped before the next opens:
    /// - `flips` (the defect's catcher): the history lands at least one flip
    ///   before its second prompt, and after each wide prompt every layer's
    ///   map is the card copy's row followed by the shared expert's HOST
    ///   entry (`Placed::stale_maps`: none), on the residency run and the off
    ///   run alike; every call's logits row within [`FREE_BAND`] of the off
    ///   run's, relative (‖Δ‖/‖off‖) — the card and the host sum a token's
    ///   experts in another split with another 8-bit rule, the difference
    ///   (o2)'s band bounds on a layer's output, taken here through the
    ///   head —, and the ids equal or, where they first part, the residency
    ///   run's top-1 margin below [`GREEDY_MARGIN`] (a near tie). Mutant: the maps
    ///   kept as the load built them (`PlacedSide::refresh_maps` returning at
    ///   once): the first prompt's maps still name the spare experts the
    ///   machine emptied at its start and the second's the experts the flips
    ///   admitted, whose columns no launch then writes; the map line reads
    ///   it (layers named), the logits band may not.
    /// - `c1` (green-only): the history twice on the residency load gives the
    ///   same ids and logits bit for bit, and flips land.
    /// - `c7`: a residency reset after the history brings every layer's live
    ///   set back to its seed.
    /// - `admit`: the history once more on the residency load with the
    ///   prompt calls' streaming `admit` ([`e_admit`]), the lever off again
    ///   after it.
    /// - `split`: the history once more with the prompt calls' streaming
    ///   `split` at [`SPLIT_COSTS`] with the union check on
    ///   (`Placed::set_ring_check`) ([`e_split`]), the lever off again after
    ///   it.
    ///
    /// The clause opens the file `ref_model_path` names and is a
    /// self-consistency clause.
    fn residency(host: HostCfg) -> Result<bool, GateError> {
        tier::sc("(e) the residency machine on the placed load: flips, c1, c7, admit, split")?;
        let ids = e_ids()?;
        let (mut off, _) = e_open(host, false)?;
        let off_seen = e_history(&mut off, &ids, None)?;
        drop(off);
        let (mut on, mid) = e_open(host, true)?;
        let word = match mid {
            Some(Residency::Mid { pinned, spares }) => format!("mid-p{pinned}-s{spares}"),
            _ => return Err("no residency to run".into()),
        };
        println!("residency: {word} over the plan's card experts");
        let seeds = {
            let m = on.model();
            let b = m.body(NAME)?;
            let p = b
                .placed()
                .ok_or("an open_placed load with no placed side")?;
            let machine = p
                .hybrid()
                .swap()
                .ok_or("the load runs no residency machine")?;
            residency_clauses::seeds(machine, p.hybrid().slots().layers())?
        };
        let first = e_history(&mut on, &ids, Some(&off_seen.tokens))?;
        // `flips`
        let band = e_band(&first, &off_seen)?;
        let maps_ok = first.stale.iter().chain(&off_seen.stale).all(Vec::is_empty);
        let flips = first.landed > 0 && maps_ok && band.ok();
        println!(
            "flips: {} flips landed before the second prompt (above 0); wide maps off the card \
             copy's rows after the prompts: residency {:?} off {:?} (none); {}: {}",
            first.landed,
            first.stale,
            off_seen.stale,
            band.line(),
            verdict(flips)
        );
        // `c1`
        let second = e_history(&mut on, &ids, Some(&off_seen.tokens))?;
        let c1 =
            residency_clauses::c1_clause(first.tokens.len(), first.landed, e_same(&first, &second));
        // `c7`
        let r = on
            .residency_reset()?
            .ok_or("the load runs no residency machine")?;
        record::residency_reset(&r).print();
        let c7 = {
            let m = on.model();
            let b = m.body(NAME)?;
            let p = b
                .placed()
                .ok_or("an open_placed load with no placed side")?;
            let machine = p
                .hybrid()
                .swap()
                .ok_or("the load runs no residency machine")?;
            residency_clauses::c7_clause(&r, machine, &seeds)
        };
        // `admit`
        e_stream(&mut on, XMode::Admit)?;
        let streamed = e_history(&mut on, &ids, Some(&off_seen.tokens));
        e_stream(&mut on, XMode::Off)?;
        let admit = e_admit(&streamed?, &off_seen)?;
        // `split`
        e_stream(&mut on, XMode::Split)?;
        let b = on.model_mut().body_parts(NAME)?.2;
        b.set_xstream_costs(SPLIT_COSTS)?;
        println!("split: xstream={}", b.xstream_word());
        b.host()
            .ok_or("an open_placed load with no placed side")?
            .set_ring_check(true)?;
        let streamed = e_history(&mut on, &ids, Some(&off_seen.tokens));
        let p = on
            .model_mut()
            .body_parts(NAME)?
            .2
            .host()
            .ok_or("an open_placed load with no placed side")?;
        let checks: Vec<(usize, usize, usize)> = p
            .take_ring_checks()
            .iter()
            .map(|c| (c.layer, c.streamed, c.wrong))
            .collect();
        p.set_ring_check(false)?;
        e_stream(&mut on, XMode::Off)?;
        let split = e_split(&streamed?, &off_seen, &checks)?;
        Ok(flips && c1 && c7 && admit && split)
    }

    /// How a run's calls stand from the off run's: the call whose logits
    /// row stands farthest ([`e_distance`]: its index, ‖Δ‖/‖off‖ and the
    /// largest single |Δ|), whether every call has its row, where the greedy
    /// ids first part and whether the run's top-1 margin there is a near tie
    /// ([`GREEDY_MARGIN`]).
    struct Band {
        worst: (usize, f64, f32),
        rows: usize,
        same_len: bool,
        parted: Option<usize>,
        near: bool,
    }

    impl Band {
        /// Every row within [`FREE_BAND`] of the off run's (relative), the
        /// ids equal or parting at a near tie.
        fn ok(&self) -> bool {
            self.same_len && self.worst.1 <= FREE_BAND && self.near
        }

        fn line(&self) -> String {
            format!(
                "{} calls' logits within {FREE_BAND} of the off run's, relative: worst {:.3e} at \
                 call {} (largest single logit {:.3e}); greedy ids part at {:?} (equal or a near \
                 tie: {})",
                self.rows, self.worst.1, self.worst.0, self.worst.2, self.parted, self.near
            )
        }
    }

    /// `run`'s calls against the off run's ([`Band`]).
    fn e_band(run: &Seen, off: &Seen) -> Result<Band, GateError> {
        let worst = run
            .rows
            .iter()
            .zip(&off.rows)
            .map(|(a, b)| e_distance(a, b))
            .enumerate()
            .fold((0usize, 0.0f64, 0.0f32), |w, (i, (rel, abs))| {
                if rel > w.1 { (i, rel, abs) } else { w }
            });
        let parted = run.tokens.iter().zip(&off.tokens).position(|(a, b)| a != b);
        let near = match parted {
            None => true,
            Some(i) => flip::head_margin(&run.rows[i])? < GREEDY_MARGIN,
        };
        Ok(Band {
            worst,
            rows: run.rows.len(),
            same_len: run.rows.len() == off.rows.len(),
            parted,
            near,
        })
    }

    /// `admit`'s verdict on (e)'s history run with the prompt calls'
    /// streaming `admit` (`streamed`) against the off run's: the first call
    /// (from a clear: no decode history, its placement kept) admitted
    /// experts on the wide arm's layers — a call that admits none fails by
    /// name, not vacuously — and the run holds what a streaming run holds
    /// ([`e_held`]).
    fn e_admit(streamed: &Seen, off: &Seen) -> Result<bool, GateError> {
        const _: () = assert!(E_P1 == PLACED_UBATCH && E_P1 >= Body35::STREAM_FLOOR as usize);
        let first = &streamed.calls[0];
        let admitted: usize = first.picks.iter().map(|(_, p)| p.admitted).sum();
        if admitted == 0 {
            println!(
                "admit: the first call's {} picks admitted no expert, so the clause would hold \
                 vacuously: {}",
                first.picks.len(),
                verdict(false)
            );
            return Ok(false);
        }
        let (ok, line) = e_held(streamed, off)?;
        println!(
            "admit: the first call admitted {admitted} experts over {} picks, {line}: {}",
            first.picks.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `split`'s verdict on (e)'s history run with the prompt calls'
    /// streaming `split` (`streamed`) against the off run's: the first call
    /// streamed experts on the wide arm's layers (`take_xstream_records`) —
    /// a call that streams none fails by name, not vacuously; its walk may
    /// admit none, the stream serving those experts instead — and the run
    /// holds what a streaming run holds ([`e_held`]): the stream ended with
    /// the call (each layer's live set is the pick's alone) and the streamed
    /// experts, run from the ring beside the card's stacks, keep the logits
    /// in the off run's band; and every ring pass's union places read back
    /// (`checks`: the layer, its slots on a streamed expert, its wrong
    /// places) are the stack's row with the ring's experts at `n_card` plus
    /// their slot in the half, on a run whose passes put slots on the ring.
    fn e_split(
        streamed: &Seen,
        off: &Seen,
        checks: &[(usize, usize, usize)],
    ) -> Result<bool, GateError> {
        let first = &streamed.calls[0];
        let wide: Vec<&XLayer> = first
            .xlayers
            .iter()
            .map(|(_, x)| x)
            .filter(|x| x.cols >= WIDE_FROM)
            .collect();
        let n: usize = wide.iter().map(|x| x.streamed).sum();
        if n == 0 {
            println!(
                "split: the first call streamed no expert over its {} layer records of the wide \
                 arm, so the clause would hold vacuously: {}",
                wide.len(),
                verdict(false)
            );
            return Ok(false);
        }
        let layers = wide.iter().filter(|x| x.streamed > 0).count();
        let (held, line) = e_held(streamed, off)?;
        let on_ring: usize = checks.iter().map(|c| c.1).sum();
        let wrong: Vec<(usize, usize)> = checks
            .iter()
            .filter(|c| c.2 > 0)
            .map(|c| (c.0, c.2))
            .collect();
        let union = on_ring > 0 && wrong.is_empty();
        let ok = held && union;
        println!(
            "split: the first call streamed {n} experts over {layers} layers of the wide arm \
             ({} admitted by its picks), {line}; union places read back over {} ring passes, \
             {on_ring} slots on the ring (above 0), (layer, wrong places) {wrong:?} (none): {}",
            first.picks.iter().map(|(_, p)| p.admitted).sum::<usize>(),
            checks.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// What a streaming run of (e)'s history (`streamed`) holds against the
    /// off run's, and its line:
    /// - the first call's end kept, and each layer's live set after it its
    ///   start set with as many experts swapped as the layer's pick admitted
    ///   (one walk: one pick a layer), every wide map the card copy's row
    ///   after it;
    /// - the second call (after the steps' decode history) ended returned,
    ///   each layer's live set after it its start set: the one before it
    ///   with the flips its boundary landed swapped in, one expert each;
    /// - every call's logits row and greedy id as `flips` holds them
    ///   ([`e_band`]).
    fn e_held(streamed: &Seen, off: &Seen) -> Result<(bool, String), GateError> {
        let [first, second] = &streamed.calls;
        let swapped = |l: usize| -> usize {
            first
                .picks
                .iter()
                .filter(|(_, p)| p.layer == l)
                .map(|(_, p)| p.admitted)
                .sum()
        };
        let layers_ok = first.before.len() == first.after.len()
            && first.before.len() == second.before.len()
            && !first.before.is_empty();
        let kept_sets: Vec<usize> = first
            .before
            .iter()
            .zip(&first.after)
            .enumerate()
            .filter(|(l, (b, a))| {
                let out = b.iter().filter(|e| !a.contains(e)).count();
                a.len() != b.len() || out != swapped(*l)
            })
            .map(|(l, _)| l)
            .collect();
        let resized: Vec<usize> = second
            .before
            .iter()
            .zip(&second.after)
            .enumerate()
            .filter(|(_, (b, a))| b.len() != a.len())
            .map(|(l, _)| l)
            .collect();
        let swapped_back: usize = second
            .before
            .iter()
            .zip(&second.after)
            .map(|(b, a)| b.iter().filter(|e| !a.contains(e)).count())
            .sum();
        let kept = first.end.as_ref().is_some_and(|e| e.kept);
        let returned = second.end.as_ref().is_some_and(|e| !e.kept);
        let band = e_band(streamed, off)?;
        let ok = layers_ok
            && kept
            && returned
            && kept_sets.is_empty()
            && resized.is_empty()
            && swapped_back == second.landed
            && streamed.stale[0].is_empty()
            && band.ok();
        let line = format!(
            "its end kept {kept}, layers whose live set is not its start set with the pick's \
             admits swapped in {kept_sets:?} (none), wide maps off the card copy's rows after it \
             {:?} (none); the second call (a decode history before it) ended returned {returned} \
             ({} restored), experts out of the live sets across it {swapped_back} (the {} flips \
             its boundary landed), layers whose live count moved {resized:?} (none); {}",
            streamed.stale[0],
            second.end.as_ref().map_or(0, |e| e.restored),
            second.landed,
            band.line()
        );
        Ok((ok, line))
    }

    /// The batch set's five tokens.
    fn batch_ids() -> Result<Vec<u32>, GateError> {
        let man = ik_set("the batch set's tokens", Set::Batch)?;
        Ok(
            ref_ints(&man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
                .iter()
                .map(|&i| u32::try_from(i))
                .collect::<Result<_, _>>()?,
        )
    }

    // ------------------------------------------------ the text path

    /// "The capital of France is" under the vocabulary of every qwen35moe
    /// file this path has run on (`REF_TOKENS` of the profiles,
    /// `tools/ref/models/{qwen35moe,minicpmv47}.sh`: ik's `llama-tokenize`, no
    /// BOS); [`text_vocab`] holds the file's own tokenizer to them.
    const TEXT_TOKENS: [u32; 5] = [760, 6511, 314, 9338, 369];

    /// Greedy tokens each prompt of [`GREEDY_CASES`] generates.
    const GREEDY_N: usize = 32;

    /// One prompt of the greedy cross-check and what the reference generated
    /// from it.
    struct GreedyCase {
        name: &'static str,
        prompt: &'static str,
        /// The prompt's ids as the reference tokenized it (`tokenizer.ggml.add_bos_token`
        /// is false in the files this holds for, so no BOS).
        prompt_ids: &'static [u32],
        /// The reference's first [`GREEDY_N`] greedy ids.
        ids: &'static [u32],
    }

    /// The clauses of [`run`] that read ik's sets, which are dumped from
    /// [`MODEL`] alone: letter, name.
    const SET_CLAUSES: [(&str, &str); 9] = [
        ("c", "free-running"),
        ("f", "teacher-forced taps"),
        ("t", "the decode step after each step set's prefill"),
        ("u/d", "the wide arm against the gemv arm and ik's state"),
        ("w", "the ubatch size"),
        ("o", "the placed load"),
        ("j", "the unjoined shared expert"),
        ("o8", "the unjoined shared expert on the placed load"),
        ("e", "the residency machine"),
    ];

    /// The size of the file the reference below was cut from: MiniCPM-V 4.7
    /// 35B-A3B at Q4_K_M (`tools/ref/models/minicpmv47.sh`).
    const GREEDY_FILE_BYTES: u64 = 21_166_255_040;

    /// PIN(2026-10-13): the greedy cross-check's reference, mainline llama.cpp
    /// 358a420ec: `llama-completion -m FILE -f PROMPT --temp 0 --top-k 1 -n 32
    /// -ngl 99 -c 512 -fa on -no-cnv --no-display-prompt --special` on the A6000,
    /// on the file itself (mainline loads it). ik_llama.cpp (version 4906,
    /// db517b69) refuses the file (it reads `nextn_predict_layers` 1 and asks for
    /// the stripped `blk.40`); on a header-corrected copy it gives the same ids
    /// as mainline on all three prompts. The copy is `tools/ref/gguf-respell.py
    /// respell FILE OUT --from qwen35moe --to qwen35moe --set
    /// qwen35moe.block_count=41:40 --set qwen35moe.nextn_predict_layers=1:0`:
    /// source sha256 c729ae6c7bb16dd5b362cdb925c7fb09d744c95601f62b0f45ce51efb55bfe95
    /// (the file, [`GREEDY_FILE_BYTES`] bytes), copy sha256
    /// 840a959b75cc67f28774f04812f962bb69345f9fd4a437eadad75d42f7d7578e, the data
    /// region equal byte for byte. The driver prints text, not ids: `ids` are
    /// `llama-tokenize --ids` of that text (mainline's, `--no-escape`; the
    /// driver's trailing newline pair dropped), equal to ik's re-tokenized ids
    /// and to ours on the first 20 positions of "english", 24 of "korean" and all
    /// 32 of "code". The prompts' ids are `llama-tokenize`'s of the prompt text.
    const GREEDY_CASES: [GreedyCase; 3] = [
        GreedyCase {
            name: "english",
            prompt: "What is the capital of France, and what is it famous for?",
            prompt_ids: &[
                3710, 369, 279, 6511, 314, 9338, 11, 321, 1092, 369, 424, 10919, 364, 30,
            ],
            ids: &[
                271, 248068, 198, 77264, 11, 279, 1156, 369, 9859, 264, 29350, 57879, 3296, 883,
                279, 6511, 314, 9338, 321, 1141, 10919, 57902, 13, 1061, 369, 264, 4047, 3134, 11,
                748, 279, 1965,
            ],
        },
        GreedyCase {
            name: "korean",
            prompt: "한국의 수도는 어디이고, 그곳에서 가장 유명한 것은 무엇인가요?",
            prompt_ids: &[
                22836, 154735, 159951, 16072, 162634, 153572, 11, 239069, 54581, 170945, 204464,
                171362, 154029, 209495, 30,
            ],
            ids: &[
                271, 248068, 271, 248069, 271, 22836, 154735, 159951, 16072, 2972, 176490, 318,
                1460, 10644, 31230, 151396, 13, 271, 176490, 54581, 170945, 195492, 149789, 215926,
                178728, 16072, 2972, 63100, 94457, 169050, 318, 38,
            ],
        },
        GreedyCase {
            name: "code",
            prompt: "Write a Python function that returns the n-th Fibonacci number.",
            prompt_ids: &[
                7734, 264, 12654, 709, 421, 4523, 279, 307, 7324, 76938, 1324, 13,
            ],
            ids: &[
                271, 760, 76938, 8240, 369, 4364, 430, 25, 198, 12, 426, 7, 15, 8, 283, 220, 15,
                198, 12, 426, 7, 16, 8, 283, 220, 16, 198, 12, 426, 1393, 8, 283,
            ],
        },
    ];

    /// The profile's file holds the vocabulary [`TEXT_TOKENS`] was printed
    /// under: its own tokenizer encodes the prompt to those ids.
    fn text_vocab(path: &str) -> Result<bool, GateError> {
        let tok =
            Tokenizer::from_gguf(path).map_err(|e| format!("the vocabulary of {path}: {e}"))?;
        let got = tok.encode("The capital of France is", false, false);
        let pass = got == TEXT_TOKENS;
        println!(
            "text vocabulary: \"The capital of France is\" encodes to {got:?} (want {TEXT_TOKENS:?}) {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The greedy cross-check: each prompt of [`GREEDY_CASES`], tokenized by
    /// the file's own tokenizer and prefilled, then [`GREEDY_N`] steps fed the
    /// reference's own ids (teacher-forced, so the history is ik's after any
    /// difference): at each step our argmax must equal the reference's id,
    /// or the reference's token must lie within [`GREEDY_MARGIN`] below ours
    /// in our own logits row (the V4.1 free arm's rule: a near tie, where
    /// ik's 8-bit activation rule and ours round to either side). Until the
    /// first difference this is the free-running greedy run; past it a
    /// run that followed its own argmax would be another history, and a
    /// prompt fed one id late reads a gap past the margin at some step.
    fn greedy_cross(m: &mut Qwen35moeModel, path: &str) -> Result<bool, GateError> {
        let tok =
            Tokenizer::from_gguf(path).map_err(|e| format!("the vocabulary of {path}: {e}"))?;
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        for case in GREEDY_CASES {
            let prompt = tok.encode(case.prompt, true, false);
            let same_prompt = prompt == case.prompt_ids;
            let fed = &prompt[..];
            m.reset()?;
            let mut next = m.prefill_with(fed, PrefillPath::Auto)?;
            let mut flips = Vec::new();
            for (i, &want) in case.ids.iter().enumerate() {
                if next != want {
                    let row = m.logits()?;
                    let at = |t: u32| row.get(t as usize).copied().unwrap_or(f32::NAN);
                    let rank = row.iter().filter(|&&v| v > at(want)).count() + 1;
                    flips.push((i, next, want, at(next) - at(want), rank));
                }
                if i + 1 < case.ids.len() {
                    next = m.step(&[want])?;
                }
            }
            let worst = flips.iter().map(|f| f.3).fold(0.0f32, f32::max);
            let near = flips.iter().all(|f| f.3 < GREEDY_MARGIN);
            let pass = same_prompt && case.ids.len() == GREEDY_N && near;
            let shown: Vec<String> = flips
                .iter()
                .map(|(i, ours, want, gap, rank)| {
                    format!("step {i}: ours {ours}, reference {want}, our logit gap {gap:.4}, the reference's token ranks {rank}")
                })
                .collect();
            println!(
                "greedy {}: prompt ids {} ({}); {GREEDY_N} steps on the reference's ids, {} argmax equal, {} differ [{}], worst gap {worst:.4} (margin {GREEDY_MARGIN}) {}",
                case.name,
                fed.len(),
                if same_prompt {
                    "as the reference's"
                } else {
                    "NOT the reference's"
                },
                GREEDY_N - flips.len(),
                flips.len(),
                shown.join("; "),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// A qwen35moe file that is not [`MODEL`]: the clauses that read no ik
    /// set, on the file's own tokenizer's prompt, and the greedy
    /// cross-check against [`GREEDY_CASES`]; every clause that reads a set
    /// is named as skipped, never dropped silently.
    fn text_run() -> Result<(), GateError> {
        let path = cfg().path.to_string_lossy().into_owned();
        let name = std::path::Path::new(&path)
            .file_name()
            .map_or(path.clone(), |n| n.to_string_lossy().into_owned());
        println!("text path: {name} is not {MODEL}; no ik set was dumped from it");
        let toks = TEXT_TOKENS.to_vec();
        let mut m = open(CTX, KvQ8::F16)?;
        let mut ok = text_vocab(&path)?;
        ok &= file_shape()?;
        ok &= structure(&mut m)?;
        let (p_ok, decode) = paths(&mut m, &toks)?;
        ok &= p_ok;
        m.set_mode(StepMode::Graph);
        ok &= gemv_arm(&mut m, &toks, &decode)?;
        ok &= wide_plan(&mut m)?;
        ok &= refusals(&mut m)?;
        let mut skipped = SET_CLAUSES.len();
        let bytes = std::fs::metadata(&path)?.len();
        if bytes == GREEDY_FILE_BYTES {
            ok &= greedy_cross(&mut m, &path)?;
        } else {
            skipped += 1;
            println!(
                "SKIPPED (g) greedy cross-check: its reference was cut from a file of \
                 {GREEDY_FILE_BYTES} bytes, {name} has {bytes}"
            );
        }
        let vocab = m.body("gate_qwen35moe_e2e")?.vocab();
        drop(m);
        ok &= checkpoints()?;
        ok &= slots_two(vocab)?;
        ok &= q8_cache(&toks)?;
        for (letter, what) in SET_CLAUSES {
            println!("SKIPPED ({letter}) {what}: no ik reference set for {name}");
        }
        println!(
            "gate_qwen35moe_e2e (text path): {skipped} clauses skipped by name: {}",
            verdict(ok)
        );
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// The prompt ids (v) steps on the oracle's file. The tier's step-1,024
    /// set's prompt; a set that holds fewer than [`Q8_N`] is refused by name.
    fn d1k_ids() -> Result<Vec<u32>, GateError> {
        let man = ik_set("(v) the step-1,024 set's prompt ids", Set::D1k)?;
        let (_, _, prompt) = man.step()?;
        if prompt.len() < Q8_N {
            return Err(format!(
                "the step-1,024 set's prompt holds {} ids, (v) takes {Q8_N}",
                prompt.len()
            )
            .into());
        }
        Ok(prompt.to_vec())
    }

    /// The prompt ids the q8_0 cache arm (v) steps.
    const Q8_N: usize = 96;

    /// The q8_0 cache arm (`BLOOMERY_QWEN3_KV=q8_0`, the seat's
    /// `--cache-type-k q8_0`), on its own models dropped before the clause
    /// returns: the attention layers' planes hold the two-plane layout (the
    /// store bytes drop, and the resident bytes drop by exactly the stores'
    /// own delta; the delta layers' stores are f32 either way), and the
    /// oracle set's first ids stepped, the same ids prefilled on the pass
    /// path and the same steps through the captured graph all leave the
    /// same last logits bit for bit and the same argmax — the arm's own
    /// consistency, the (p) and (r) contracts on the q8 path. The distance
    /// to the f16 run's logits prints as the quantization diagnostic
    /// (`store`'s f16 readback refuses a q8_0 cache by name, so the bits
    /// clause holds the logits and tokens). Asking the q8 model for the
    /// scalar pass is refused by name at the call, and its next step still
    /// runs. The ids are `toks`' first 96: the step-1,024 set's prompt on
    /// the oracle's file ([`d1k_ids`]), [`TEXT_TOKENS`] on the text path.
    fn q8_cache(toks: &[u32]) -> Result<bool, GateError> {
        tier::sc("(v) the q8_0 cache arm: bytes, bits, replay and the scalar refusal")?;
        let ids: Vec<u32> = toks.iter().take(Q8_N).copied().collect();
        let mut f16 = open(CTX, KvQ8::F16)?;
        let (f16_resident, f16_store) = (f16.resident_bytes(), f16.body("q8")?.store_bytes());
        f16.reset()?;
        let mut f16_tok = 0;
        for &id in &ids {
            f16_tok = f16.step(&[id])?;
        }
        let f16_logits = f16.logits()?;
        drop(f16);
        let mut m = open(CTX, KvQ8::Q8)?;
        let (resident, store) = (m.resident_bytes(), m.body("q8")?.store_bytes());
        let bytes_ok = f16_store > store
            && f16_resident.saturating_sub(resident) == f16_store.saturating_sub(store);
        println!(
            "q8 load: resident_bytes={resident} (f16 {f16_resident}), store bytes {store} of \
             f16's {f16_store} — the residents differ by the stores' own delta {}",
            f16_store.saturating_sub(store)
        );
        let mut ok = bytes_ok;
        m.reset()?;
        let mut step_tok = 0;
        for &id in &ids {
            step_tok = m.step(&[id])?;
        }
        let logits = m.logits()?;
        // The quantization diagnostic against the f16 run: the argmax's
        // agreement and the relative distance, printed.
        let num: f64 = logits
            .iter()
            .zip(&f16_logits)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let den: f64 = f16_logits.iter().map(|b| f64::from(*b).powi(2)).sum();
        println!(
            "q8 steps: {Q8_N} ids; argmax ours={step_tok} f16={f16_tok} (printed); logits \
             rel dist {:.3e} (diagnostic)",
            if den > 0.0 { (num / den).sqrt() } else { 0.0 }
        );
        // The pass prefill of the same ids leaves the stepped run's bits.
        m.reset()?;
        let pass_tok = m.prefill_with(&ids, PrefillPath::Pass)?;
        let pass_logits = m.logits()?;
        let bits = pass_tok == step_tok && logits == pass_logits;
        println!(
            "q8 bits: pass prefill == {Q8_N} steps, last logits bit for bit {}",
            verdict(bits)
        );
        ok &= bits;
        // The graph replay: the same steps through the captured step graph.
        m.reset()?;
        m.set_mode(StepMode::Graph);
        println!("q8 capture graph_nodes={}", m.capture_step()?);
        m.reset()?;
        let mut last = 0;
        for &id in &ids {
            last = m.step(&[id])?;
        }
        let graph_logits = m.logits()?;
        let replay = last == step_tok && logits == graph_logits;
        println!(
            "q8 replay: {Q8_N} captured steps == the eager bits {}",
            verdict(replay)
        );
        ok &= replay;
        // The scalar pass on a q8_0 cache: refused by name at the call that
        // asks for it, in `q8_unserved`'s words, and the model's pass kept —
        // the next eager step runs.
        m.set_mode(StepMode::Eager);
        let named = match m.set_flash_mma(false) {
            Err(GpuError::Shape {
                what: "qwen35moe::set_flash_mma",
                detail,
            }) => detail.contains(
                "a q8_0 cache's flash runs the eight-head layout's tensor-core pass only; asked \
                 for the scalar pass",
            ),
            _ => false,
        };
        let kept = m.reset().is_ok() && m.step(&[ids[0]]).is_ok();
        println!(
            "q8 scalar refusal: set_flash_mma(false) refused by name at the call {named}, the \
             next step runs {kept} {}",
            verdict(named && kept)
        );
        ok &= named && kept;
        Ok(ok)
    }

    /// The fixture-oracle clauses a run declares, held by `tier::expect_fixture_oracle` at the
    /// gate's end in both tiers: `free` at (c), (o2) and (o7), `forced` on the batch set, one
    /// `step_set` a step set (the 4-token and the 1,024-token ones) and `wide_arm` at (u)(d) and
    /// (o5): 3 + 1 + 2 + 2 = 8. The real tier decides the same eight and leaves them to the
    /// fixture tier, so its count is 8 too.
    const FX_FREE: usize = 3;
    const FX_FORCED: usize = 1;
    const FX_STEP_SETS: usize = 2;
    const FX_WIDE: usize = 2;
    const FX_ORACLE: usize = FX_FREE + FX_FORCED + FX_STEP_SETS + FX_WIDE;

    /// The fixture-oracle clauses of `--placed-only`, which runs (o) and (o7): (o2)'s `free`,
    /// (o5)'s `wide_arm` and (o7)'s `free`.
    const FX_PLACED_ONLY: usize = 3;

    /// The closing lines of a run that declares `fixture_oracle` fixture-oracle clauses: the
    /// verdict under `label`, the tier's tally and the check of that count.
    fn close(label: &str, ok: bool, fixture_oracle: usize) -> Result<(), GateError> {
        println!("{label}: {}", verdict(ok));
        println!("{}", tier::tally_line());
        tier::expect_fixture_oracle(NAME, fixture_oracle)?;
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        // A saved `--model-file` or `--model-file=PATH` names a file `init` never
        // opens: the real tier reads `BLOOMERY_REF_MODEL`, so the flag would otherwise be ignored.
        if std::env::args()
            .skip(1)
            .any(|a| a == "--model-file" || a.starts_with("--model-file="))
        {
            return Err(
                "--model-file is no longer supported; select the model through BLOOMERY_REF_MODEL"
                    .into(),
            );
        }
        // The levers and the file's header are read, and each value the header
        // gives printed beside the literal it replaced, before anything loads.
        let proof = init()?;
        let host = cfg().host;
        if text_file()? {
            if let Some(flag) = std::env::args().skip(1).find(|a| a.starts_with("--")) {
                return Err(format!(
                    "{flag}: a clause that reads ik's sets, which are dumped from {MODEL}, not from {}",
                    cfg().path.display()
                )
                .into());
            }
            return text_run();
        }
        if std::env::args().any(|a| a == "--placed-only") {
            let toks = batch_ids()?;
            let mut ok = proof;
            ok &= placed(&toks, host)?;
            ok &= placed_no_host(&toks, host)?;
            return close("gate_qwen35moe_e2e --placed-only", ok, FX_PLACED_ONLY);
        }
        if std::env::args().any(|a| a == "--unjoined-placed-only") {
            let toks = batch_ids()?;
            let mut ok = proof;
            ok &= placed_unjoined(&toks, host)?;
            return close("gate_qwen35moe_e2e --unjoined-placed-only", ok, 0);
        }
        if std::env::args().any(|a| a == "--common-placed-only") {
            let mut ok = proof;
            ok &= placed_common(host)?;
            return close("gate_qwen35moe_e2e --common-placed-only", ok, 0);
        }
        if std::env::args().any(|a| a == "--residency-only") {
            let mut ok = proof;
            ok &= residency(host)?;
            return close("gate_qwen35moe_e2e --residency-only", ok, 0);
        }
        if std::env::args().any(|a| a == "--unjoined-only") {
            let toks = batch_ids()?;
            let mut ok = proof;
            ok &= unjoined(&toks)?;
            return close("gate_qwen35moe_e2e --unjoined-only", ok, 0);
        }
        if std::env::args().any(|a| a == "--memguard-only") {
            let mut ok = proof;
            ok &= memguard()?;
            return close("gate_qwen35moe_e2e --memguard-only", ok, 0);
        }
        let mut m = open(CTX, KvQ8::F16)?;
        let mut ok = proof;
        ok &= file_shape()?;
        ok &= structure(&mut m)?;
        let toks = batch_ids()?;
        let (p_ok, decode) = paths(&mut m, &toks)?;
        ok &= p_ok;
        ok &= free(&mut m, "(c) ", &toks, &decode, FREE_PROBE_CARD)?;
        let gains = Gains::read(m.layers().len())?;
        m.set_mode(StepMode::Eager);
        fresh(&mut m)?;
        let zero = |_: &mut Qwen35moeModel, _: usize| -> Result<(), GateError> { Ok(()) };
        let man = ik_oracle(
            "(f) each layer teacher-forced on ik's inputs, the batch set as one pass",
            Set::Batch,
        )?;
        ok &= forced(
            &mut m,
            &man,
            Set::Batch.name(),
            (0, toks.len()),
            &gains,
            &zero,
        )?;
        m.set_mode(StepMode::Graph);
        for set in [Set::Step4, Set::D1k] {
            ok &= step_set(&mut m, set, &gains)?;
        }
        ok &= gemv_arm(&mut m, &toks, &decode)?;
        ok &= wide_arm(&mut m, "", U_GATE)?;
        ok &= ubatch_bits(&mut m)?;
        ok &= wide_plan(&mut m)?;
        ok &= refusals(&mut m)?;
        let vocab = m.body("gate_qwen35moe_e2e")?.vocab();
        drop(m);
        ok &= unjoined(&toks)?;
        ok &= checkpoints()?;
        ok &= slots_two(vocab)?;
        ok &= placed(&toks, host)?;
        ok &= placed_no_host(&toks, host)?;
        ok &= placed_unjoined(&toks, host)?;
        ok &= residency(host)?;
        ok &= q8_cache(&d1k_ids()?)?;
        close("gate_qwen35moe_e2e", ok, FX_ORACLE)
    }
}
