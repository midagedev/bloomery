//! The Hy4-flash prompt-batch gate: a prompt walked as one batch (`bloomery_gpu_hy4::prompt`: one
//! pass per layer over every row, the projections and the routed experts as grouped int8 GEMMs, the
//! sparse layers' selector and the window layers' staged ring over the call's rows) against the
//! decode chain over the same ids (one id a step, the `Body` of the e2e gate) on the fixture file
//! (five layers, the cut [0, 1, 6, 11, 41]: sparse + dense, sparse + engram + routed, window,
//! window + engram, sparse; window 64, top-k 64).
//!
//! What the two walks differ by, and so how they are held. A Q8_0 projection reads its input in f32
//! in the decode chain's gemv and in int8 blocks of 32 in the batch's GEMM, a rounding of about
//! 0.7 % of each block's largest value: about 1e-2 relative L2 a sub-layer, the size of the gap
//! between the decode chain and the vendor's CPU path (`gate-gpu-hy4-e2e` pins its attention at
//! 1.41e-2 and its feed-forward at 2.12e-2), and the batch's is the vendor's rounding. The free
//! chains then part by the discrete choices that rounding turns (the router's 16 of 512, the
//! selector's kept pools: the fixture's random weights leave both near ties), so a free-chain
//! logits row differs by 0.19 to 0.33 relative L2 (the e2e gate's `σ̂` is 0.16 to 0.42) and stands as a
//! printed measure. The gate holds what a bug moves:
//!
//! - `forced_attention_is_decode`: on each of the five fixture sets (set `e`: 93 ids, past the
//!   64-row window and the selector's kept pools, so the ring's truncation and the indexer's
//!   selection run inside the batch), each layer's attention run alone by the decode chain over the
//!   batch's own normed rows of every position (`attn_norm`), reading the batch's selected rows on a
//!   sparse layer, gives the batch's output at the last row within [`ATTN_BAND`]; the decode chain's
//!   own selection of the same rows is the batch's as a set unless the cut is a tie at the two
//!   scorings' deviation, and the pool scores themselves are within [`SCORE_BAND`].
//! - `forced_ffn_is_decode`: each layer's feed-forward block run alone by the decode chain on the
//!   batch's own normed row (`ffn_norm`) of every position gives the batch's output within
//!   [`FFN_BAND`], and a routed layer's picks are the batch's for every row.
//! - `stores_are_read_by_decode`: after the batch walks a prompt, each layer's attention at the next
//!   position run by the decode chain over the stores the batch left (the window ring, the cache,
//!   the index ring, the pooled keys) against the same attention over stores the decode chain
//!   built from the batch's own normed rows, within [`STORE_BAND`], the selection the same or a tie
//!   and its scores within [`SCORE_BAND`]; on the five sets and on the long prompt, so the window
//!   ring and the index ring wrap.
//! - `batch_is_single`: the batch of a prompt against the same batch rule one id a call (`cap` 1:
//!   the same kernels over one row, so no rounding differs and no choice turns): every stage of
//!   every layer of every row, the head's fold and the logits row within [`SINGLE_BAND`], and the
//!   stores read through [`CONT_STEPS`] decode steps fed the same ids, each logits row within the
//!   band. This reads every store the batch writes (the cache rows, the index ring, the pooled
//!   keys, the window ring, the engram history, the position). It runs on the five sets and on a
//!   prompt of [`LONG`] ids (the index ring of 259 rows and the window ring wrap, a call of several
//!   walks).
//! - `chunking_is_invariant`: the same prompt through a batch whose walks, attention chunks and
//!   feed-forward units are a few rows each gives the large geometry's logits and continuation
//!   within [`CHUNK_BAND`]: on set `e`, on the long prompt and on a prompt of [`BIG`] ids at the
//!   released geometry (4,096 rows a walk, 2,560 a feed-forward unit).
//! - `refusals_are_named`: a geometry the kernels cannot hold, a prompt of no id and a prompt past
//!   the context end by name, and the model keeps decoding after the last one.
//!
//! The placed pass (`bloomery_gpu_hy4::prompt_placed`: a placed load's prompt as passes of up to 512
//! rows over the host tier's batch port) is held on the same file with the gate card's budget
//! planned to half the experts, so every routed layer has a host share (`placed_load::levers`):
//!
//! - `placed_attention_is_decode` and `placed_ffn_is_decode`: the five sets through the placed
//!   pass, each layer's attention and feed-forward block run alone by the whole-card decode chain on
//!   the pass's own rows, within [`ATTN_BAND`] and [`FFN_BAND`] with the router's picks the pass's:
//!   a hosted layer's block is the card's slots and the host's sum added, so a back that leaves the
//!   host's sum out moves every hosted layer's output far past the band.
//! - `placed_pass_is_census`: a prompt of [`PLACED_IDS`] ids through the pass serves each hosted
//!   layer once a walk (`batch_served` = hosted layers x ceil(ids / 512)) over every column
//!   (`batch_cols`), its picks the card's and the host file's (`picks` = `card` + `file`, none
//!   streamed), and the same ids stepped leave the batch counters where they were.
//! - `placed_pass_is_reserved`: the load's resident bytes stand inside the plan's terms with the
//!   pass's bytes as the prompt reserve, and a plan that reserves none refuses the load by name.
//! - `placed_residency_follows_the_map`: the same plan loaded under a `mid` word
//!   (`residency_clauses::mid_of` over the plan's card counts, the host set read in: the machine's
//!   source stages its parts from it) runs the common machine, each
//!   machine layer reseeded host experts first (`residency_clauses::host_first_orders`), so host
//!   experts go live in card slots the plan's experts held; the reseed copies experts onto the card
//!   and leaves no entry off its seed, and set `e` through the pass, then fed the `off` load's ids
//!   [`CONT_STEPS`] steps, reads the `off` load's logits rows within [`RESIDENCY_BAND`]. The card's
//!   slots are the live map's only when the side's map copy is the machine's view: FAIL-first
//!   (box), `PlacedHost::residency_parts` handing the machine a fresh upload of the map
//!   (`.mutants/hy4res-map-copy-fresh.patch`), so the handoff and the pass read the load's places
//!   while the slots hold the reseeded experts, moves every hosted layer's sum and the rows far
//!   past the band.
//! - `placed_pass_streams`: the plan with the stream's ring at its floor
//!   (`hy4_pass::streaming_for`) loaded under a `mid` word with walks of [`STREAM_SHAPE`] (several
//!   units in the fronts order), `BLOOMERY_XSTREAM=split` set and the rule driven to stream
//!   ([`SPLIT_COSTS`]); set `e` repeated to [`STREAM_IDS`] ids, one walk: the call streams experts
//!   (`streamed` > 0 in its records), the end counts the layers' sum, every streamed layer's ring
//!   pass joins its streamed experts once a unit and no other layer's joins, the host leaves out
//!   exactly the streamed columns (the census's `streamed`, `picks` = `card` + `file`), each unit
//!   is served once a hosted layer, and each layer's attention and feed-forward block run alone by
//!   the whole-card decode chain on the walk's own rows stand within [`ATTN_BAND`] and
//!   [`FFN_BAND`] (clauses `streamed_attention_is_decode`, `streamed_ffn_is_decode`): a streamed
//!   expert's rows the card's sum leaves out move every streamed layer far past the band.
//!   FAIL-first (box), the call's stream never opened (`.mutants/hy4r5s-stream-begin-skipped.patch`):
//!   `streamed` 0, red.
//!
//! The prompt sets only supply ids: the vendor's dumps are not read as an oracle here. The fixture
//! tier only: the released file (124 GB) does not fit one card and this gate opens the whole-card
//! body, so the real tier ends by name before anything loads.

#[cfg(feature = "hy4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_pp", gate::run())
}

#[cfg(feature = "hy4")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the placed clauses read its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "hy4")]
#[allow(
    dead_code,
    reason = "the shared placed-load body; this gate reads the plan's levers, the context, the host \
              configuration, the load line and the resident clause"
)]
#[path = "shared/placed_load.rs"]
mod placed_load;

#[cfg(feature = "hy4")]
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; this one reads the mid word and the \
              reseed orders"
)]
#[path = "shared/residency_clauses.rs"]
mod residency_clauses;

#[cfg(feature = "hy4")]
mod gate {
    use crate::gate_card;
    use crate::placed_load::{self, CTX as PLACED_CTX, HOST_CFG};
    use crate::residency_clauses;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::host::xstream::{Costs, XLayer, XMode, XReport, XStreamBody};
    use bloomery_gpu::latent::POOL;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::hy4_pass;
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, open_split, verdict};
    use bloomery_gpu_hy4::batch::{Cfg, Prompt, Q_MAX, WalkTaps, stage_width};
    use bloomery_gpu_hy4::{Body, PASS_ROWS, PassShape, TAP_STAGES, prompt, prompt_placed};
    use model::arch::Arch;
    use model::arch::hy4::place::PlanInputs;
    use model::placement::workstation;
    use refset::arch::hy4::fixture::{self, A, B, C, D, E};
    use refset::arch::prompt_tables;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_hy4_pp";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-hy4-pp";

    /// The sets whose ids the gate walks, in the order they run.
    const SETS: &[&str] = &[A, B, C, D, E];

    /// The cut's layers and their kinds, (sparse, engram site), in layer order.
    const KINDS: [(bool, bool); 5] = [
        (true, false),
        (true, true),
        (false, false),
        (false, true),
        (true, false),
    ];

    /// The pools the fixture's selector keeps beside the open one: top-k 64 rows over pools of 4.
    const KEPT: usize = 15;

    /// The ids of the long prompt: set `e` repeated, past the index ring (259 rows) and many
    /// windows of 64.
    const LONG: usize = 300;
    /// The ids of the big prompt: two walks of the released geometry (4,096 rows, then 104), and
    /// two feed-forward units in the first (2,560 rows, then 1,536).
    const BIG: usize = 4200;
    /// The ids of the placed pass's census prompt: three walks of the pass's rows (512, 512, 76).
    const PLACED_IDS: usize = 1_100;
    /// The streaming clause's walks: four units of a quarter of the pass's rows, so a walk of the
    /// fronts order holds four units in flight and its last is short.
    const STREAM_SHAPE: PassShape = PassShape::walk(PASS_ROWS / 4, 4);
    /// The streaming clause's ids: one walk of [`STREAM_SHAPE`] whose last unit is short.
    const STREAM_IDS: usize = STREAM_SHAPE.walk_rows() - 12;
    /// The streaming clause's fixed rule input: a host column of a tenth of a second, so the
    /// stream's floor sits under one column and every host expert a unit routes streams until the
    /// ring's half or the balance cuts it.
    const SPLIT_COSTS: Costs = Costs {
        host_us_per_col: 1.0e5,
        host_us_fixed: 240.0,
        card_us_fixed: 90.0,
        card_us_per_col: 2.0,
    };
    /// The decode steps after a prompt that read its stores.
    const CONT_STEPS: usize = 8;
    /// Positions of every store: the big prompt and the continuation.
    const CTX: usize = BIG + 64;

    /// The rows the tapped batch holds: the long prompt in one walk.
    const TAP_ROWS: usize = LONG;

    /// One id a call: the same kernels over one row.
    const ONE: Cfg = Cfg {
        cap: 1,
        q_rows: 1,
        ffn_rows: 1,
    };

    /// The small geometry of `chunking_is_invariant`: walks of 40 rows, attention chunks of 16 and
    /// feed-forward units of 24, so set `e` crosses three walks, six chunks and four units a walk.
    const SMALL: Cfg = Cfg {
        cap: 40,
        q_rows: 16,
        ffn_rows: 24,
    };

    /// A layer's attention run alone by the decode chain over the batch's rows, against the batch's
    /// output at the last row, relative L2: the e2e gate's own distance to the vendor (1.41e-2) is
    /// the same rounding.
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.18e-2 over five sets and five layers.
    const ATTN_BAND: f64 = 1.8e-2;

    /// A layer's attention at the position after the prompt, over the batch's stores, against the
    /// decode chain's over stores it built from the same rows, relative L2: the cache rows carry
    /// the batch's GEMM rounding of the key projection.
    /// PIN(2026-11-19): 1.5 x the worst measured, 5.97e-3 over six prompts and five layers.
    const STORE_BAND: f64 = 9.0e-3;

    /// A sparse layer's pool scores of the last row, the batch's against the decode chain's on the
    /// batch's own rows, relative L2: rounding of the key and query projections, which the pooled
    /// keys and the dot carry.
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.04e-2 over sets `e` and the long prompt.
    const SCORE_BAND: f64 = 1.6e-2;

    /// A layer's feed-forward block run alone by the decode chain on the batch's rows, against the
    /// batch's output over the rows, relative L2 (the e2e gate pins the vendor's at 2.12e-2).
    /// PIN(2026-11-19): 1.5 x the worst measured, 1.75e-2 over five sets and five layers.
    const FFN_BAND: f64 = 2.6e-2;

    /// A stage or a logits row of the batch against the same rule one id a call, relative L2: the
    /// same kernels over one row, bit-identical in every run (0 over six prompts, every stage of
    /// every row); the band leaves room for a reduction-order tie and nothing more.
    /// PIN(2026-11-19): measured 0.000e0.
    const SINGLE_BAND: f64 = 1.0e-6;

    /// A logits row of the small geometry against the large one, relative L2: the same kernels over
    /// other row sets, bit-identical in every run (0 on three prompts, 4,200 ids the largest).
    /// PIN(2026-11-19): measured 0.000e0.
    const CHUNK_BAND: f64 = 1.0e-6;

    /// A logits row of the placed load under a reseeded residency machine against the same plan's
    /// load with none, fed the same ids, relative L2: the two differ only in which of a hosted
    /// layer's experts the card's slots sum and which the host's, the card's rule against the
    /// host's, each layer's share inside [`FFN_BAND`] of the whole-card block; a slot the map
    /// misroutes is another expert's rows, of the order of the row itself.
    /// PIN(2026-11-19): 1.5 x the first box run's worst row, 1.654e-1 with the reseeded load's ids
    /// equal to the `off` load's; a map copy the machine does not write (the stale-map mutant) reads
    /// 5.339e-1 and its ids leave the `off` load's at the third id.
    const RESIDENCY_BAND: f64 = 2.5e-1;

    /// The host tier's configuration of the residency clause's load: [`HOST_CFG`] with the host set
    /// read in, which the machine's source stages its parts from (the load refuses a machine over a
    /// set it did not populate, by name). The other loads read no lever and leave the set to fault in.
    const HOST_RESIDENCY: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: true,
        ..HOST_CFG
    };

    /// The ids of set `name`.
    fn set_ids(name: &'static str) -> Result<Vec<u32>, GateError> {
        let path = tier::fixture_set("the prompt batch", &fixture::LCPP, name)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        Ok(prompt_tables::tokens(&man)?.to_vec())
    }

    /// `ids` repeated to `n` ids.
    fn repeated(ids: &[u32], n: usize) -> Vec<u32> {
        ids.iter().copied().cycle().take(n).collect()
    }

    /// The worst relative L2 of the logits rows `a` against `b`, and which row.
    fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, usize) {
        a.iter()
            .zip(b)
            .map(|(x, y)| rel(x, y))
            .enumerate()
            .fold(
                (0.0, 0),
                |(w, wi), (i, r)| {
                    if r > w { (r, i) } else { (w, wi) }
                },
            )
    }

    /// From the model's current position: [`CONT_STEPS`] decode steps, fed `feed` where given and
    /// its own argmax otherwise; the ids and logits rows after the prompt's own (`first`).
    fn continuation(
        m: &mut GpuModel<Body>,
        first: u32,
        feed: Option<&[u32]>,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        let mut ids = vec![first];
        let mut logits = vec![m.logits()?];
        for j in 0..CONT_STEPS {
            let input = feed.map_or(ids[ids.len() - 1], |f| f[j]);
            ids.push(m.step(&[input])?);
            logits.push(m.logits()?);
        }
        Ok((ids, logits))
    }

    /// `ids` through the batch `p` from a reset model: its argmax.
    fn batch_prompt(m: &mut GpuModel<Body>, p: &mut Prompt, ids: &[u32]) -> Result<u32, GateError> {
        m.reset()?;
        Ok(prompt(m, p, ids)?)
    }

    /// `ids` through the decode chain from a reset model, one id a step: its argmax.
    fn decode_prompt(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GateError> {
        m.reset()?;
        let mut token = 0;
        for &id in ids {
            token = m.step(&[id])?;
        }
        Ok(token)
    }

    /// Row `t` of `stage` of layer `l` in `taps`.
    fn row(taps: &WalkTaps, (l, stage): (usize, usize), t: usize, hidden: usize) -> &[f32] {
        let w = stage_width(stage, hidden);
        &taps.stages[l * TAP_STAGES.len() + stage][t * w..(t + 1) * w]
    }

    /// The tap slot of the stage named `name`.
    fn slot(name: &str) -> usize {
        TAP_STAGES
            .iter()
            .position(|s| *s == name)
            .unwrap_or(usize::MAX)
    }

    /// Two runs' selections of a sparse layer at the position whose live key count is `count`: they
    /// agree when the rows are the same as sets, or differ only where the cut is a tie at the two
    /// scorings' widest deviation, and the pool scores are within [`SCORE_BAND`]. The verdict and
    /// the line that shows it.
    fn selection_agrees(
        count: usize,
        (rows_a, scores_a): (&[u32], &[f32]),
        (rows_b, scores_b): (&[u32], &[f32]),
    ) -> (bool, String) {
        let (mut a, mut b) = (rows_a.to_vec(), rows_b.to_vec());
        a.sort_unstable();
        b.sort_unstable();
        // The pools a row scores: `(c - 1) / 4` of its count `c`, scored when more than the kept.
        let scored = (count - 1) / POOL;
        let (mut dev, mut margin, mut score_rel) = (0.0f32, f32::INFINITY, 0.0f64);
        if scored > KEPT {
            let (x, y) = (&scores_a[..scored], &scores_b[..scored]);
            score_rel = rel(x, y);
            dev = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f32::max);
            let mut sorted = y.to_vec();
            sorted.sort_unstable_by(|p, q| q.total_cmp(p));
            margin = sorted[KEPT - 1] - sorted[KEPT];
        }
        let same = a == b;
        let tie = !same && margin <= 2.0 * dev;
        let agree = (same || tie) && score_rel <= SCORE_BAND;
        (
            agree,
            format!(
                "rows {} vs {} same={same} tie={tie} scored_pools={scored} scores_rel={score_rel:.3e} \
                 widest={dev:.3e} cut_margin={margin:.3e}",
                a.len(),
                b.len()
            ),
        )
    }

    /// A walk of `ids` whose taps a clause reads: the whole-card batch's, or the placed pass's. The
    /// model it is given is the whole-card one, whose decode chain the clause runs each block on.
    type TapSource<'a> = dyn FnMut(&mut GpuModel<Body>, &[u32]) -> Result<WalkTaps, GateError> + 'a;

    /// Clauses `{kind}_attention_is_decode` and `{kind}_ffn_is_decode` of set `name`: a walk (`source`:
    /// the whole-card batch for `forced`, the placed pass for `placed`) runs `ids`, then the
    /// whole-card model's decode chain runs each block alone on the walk's inputs.
    fn forced_is_decode(
        m: &mut GpuModel<Body>,
        name: &str,
        ids: &[u32],
        (kind, source): (&str, &mut TapSource<'_>),
    ) -> Result<bool, GateError> {
        let attn = tier::run_clause(
            &format!("{kind}_attention_is_decode set={name}"),
            Tag::SelfConsistency,
        )?;
        let ffn = tier::run_clause(
            &format!("{kind}_ffn_is_decode set={name}"),
            Tag::SelfConsistency,
        )?;
        if !attn && !ffn {
            return Ok(true);
        }
        let t = ids.len();
        let taps = source(m, ids)?;
        let hidden = m.body(NAME)?.hidden();
        let (gpu, w, body) = m.body_parts(NAME)?;
        let mut ok = true;
        if attn {
            let (norm, out) = (slot("attn_norm"), slot("attn_out"));
            let (mut worst_out, mut cells) = (0.0f64, Vec::new());
            for l in 0..KINDS.len() {
                let rows: Vec<Vec<f32>> = (0..t)
                    .map(|i| row(&taps, (l, norm), i, hidden).to_vec())
                    .collect();
                let theirs = taps.sel[l].as_ref();
                let read = theirs.map(|s| s.0.as_slice());
                let (got, ours) = body.attention_alone(gpu, w, l, &rows, read)?;
                let r = rel(&got, row(&taps, (l, out), t - 1, hidden));
                worst_out = worst_out.max(r);
                cells.push(format!("{l}={r:.2e}"));
                let (Some((rows, scores)), Some((picked, our_scores))) = (theirs, ours) else {
                    continue;
                };
                let (agree, line) = selection_agrees(t, (rows, scores), (&picked, &our_scores));
                ok &= agree;
                println!("  selection set={name} layer={l} {line}");
            }
            let pass = ok && worst_out <= ATTN_BAND;
            println!(
                "{kind}_attention_is_decode set={name} attn_out_rel=[{}] worst={worst_out:.3e} \
                 band={ATTN_BAND:.1e} {}",
                cells.join(" "),
                verdict(pass)
            );
            ok = pass;
        }
        if ffn {
            let (norm, out) = (slot("ffn_norm"), slot("ffn_out"));
            let (mut worst_out, mut cells, mut flips) = (0.0f64, Vec::new(), 0usize);
            for l in 0..KINDS.len() {
                let (mut got, mut want) = (Vec::new(), Vec::new());
                for i in 0..t {
                    let (y, picks) = body.ffn_alone(gpu, w, l, row(&taps, (l, norm), i, hidden))?;
                    got.extend_from_slice(&y);
                    want.extend_from_slice(row(&taps, (l, out), i, hidden));
                    if let Some((mut ours, _)) = picks {
                        ours.sort_unstable();
                        let n = ours.len();
                        let mut theirs = taps.picks[l][i * n..(i + 1) * n].to_vec();
                        theirs.sort_unstable();
                        flips += usize::from(ours != theirs);
                    }
                }
                let r = rel(&got, &want);
                worst_out = worst_out.max(r);
                cells.push(format!("{l}={r:.2e}"));
            }
            let pass = flips == 0 && worst_out <= FFN_BAND;
            println!(
                "{kind}_ffn_is_decode set={name} ffn_out_rel=[{}] worst={worst_out:.3e} \
                 band={FFN_BAND:.1e} pick_mismatches={flips} {}",
                cells.join(" "),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `stores_are_read_by_decode` of the prompt `name`: after the batch walks `ids`, each
    /// layer's attention at the next position, run by the decode chain over the stores the batch
    /// left, against the same attention over stores the decode chain built from the batch's own
    /// normed rows (the position's input the last row's, any row standing in), reading the cache
    /// rows the first run selected.
    fn stores_are_read_by_decode(
        m: &mut GpuModel<Body>,
        p: &mut Prompt,
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("stores_are_read_by_decode set={name}"),
            Tag::SelfConsistency,
        )? {
            return Ok(true);
        }
        batch_prompt(m, p, ids)?;
        let t = ids.len();
        let taps = p.taps(m.gpu().stream(), t)?;
        let hidden = m.body(NAME)?.hidden();
        let (gpu, w, body) = m.body_parts(NAME)?;
        let norm = slot("attn_norm");
        let at = u32::try_from(t).map_err(|_| format!("{NAME}: a prompt past u32"))?;
        let (mut ok, mut worst_out, mut cells) = (true, 0.0f64, Vec::new());
        for l in 0..KINDS.len() {
            let mut rows: Vec<Vec<f32>> = (0..t)
                .map(|i| row(&taps, (l, norm), i, hidden).to_vec())
                .collect();
            rows.push(rows[t - 1].clone());
            let (read, sel_b) = body.attention_at(gpu, w, l, (&rows[t], at))?;
            let list = sel_b.as_ref().map(|s| s.0.as_slice());
            let (want, sel_d) = body.attention_alone(gpu, w, l, &rows, list)?;
            let r = rel(&read, &want);
            worst_out = worst_out.max(r);
            cells.push(format!("{l}={r:.2e}"));
            if let (Some(b), Some(d)) = (&sel_b, &sel_d) {
                let (agree, line) = selection_agrees(t + 1, (&b.0, &b.1), (&d.0, &d.1));
                ok &= agree;
                println!("  selection set={name} layer={l} at the next position {line}");
            }
        }
        let pass = ok && worst_out <= STORE_BAND;
        println!(
            "stores_are_read_by_decode set={name} ids={t} attn_rel=[{}] worst={worst_out:.3e} \
             band={STORE_BAND:.1e} {}",
            cells.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `batch_is_single` of the prompt `name`: the batch `p` against the single-id batch
    /// `one` (both tapped), stages, logits and the stores through the
    /// continuation; the free chains' distance from the decode chain prints beside it.
    fn batch_is_single(
        m: &mut GpuModel<Body>,
        (p, one): (&mut Prompt, &mut Prompt),
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(&format!("batch_is_single set={name}"), Tag::SelfConsistency)? {
            return Ok(true);
        }
        let layers = KINDS.len();
        let stages = layers * TAP_STAGES.len();
        // 1. The single-id arm: one call an id, its stages gathered row by row.
        m.reset()?;
        let mut first = 0;
        let mut single: Vec<Vec<f32>> = vec![Vec::new(); stages];
        let mut single_head = Vec::new();
        let mut single_last = None;
        for &id in ids {
            first = prompt(m, one, &[id])?;
            let t = one.taps(m.gpu().stream(), 1)?;
            for (acc, rows) in single.iter_mut().zip(&t.stages) {
                acc.extend_from_slice(rows);
            }
            single_head = t.head.clone();
            single_last = Some(t);
        }
        let (feed, single_logits) = continuation(m, first, None)?;
        // 2. The batch arm.
        let got = batch_prompt(m, p, ids)?;
        let batch_taps = Some(p.taps(m.gpu().stream(), ids.len())?);
        let (_, batch_logits) = continuation(m, got, Some(&feed))?;
        let (mut stage_worst, mut at) = (0.0f64, String::new());
        let mut head_rel = 0.0;
        if let Some(bt) = &batch_taps {
            for (i, (have, want)) in bt.stages.iter().zip(&single).enumerate() {
                let r = rel(have, want);
                if r > stage_worst {
                    let stage = TAP_STAGES[i % TAP_STAGES.len()];
                    (stage_worst, at) = (r, format!("layer {} {stage}", i / TAP_STAGES.len()));
                }
            }
            head_rel = rel(&bt.head, &single_head);
            // The first stage, in the chain's order, and the first row where the two differ at all.
            let hidden = m.body(NAME)?.hidden();
            let first_diff =
                bt.stages
                    .iter()
                    .zip(&single)
                    .enumerate()
                    .find_map(|(i, (have, want))| {
                        let w = stage_width(i % TAP_STAGES.len(), hidden);
                        let r = (0..ids.len())
                            .find(|&r| have[r * w..(r + 1) * w] != want[r * w..(r + 1) * w])?;
                        Some((i / TAP_STAGES.len(), TAP_STAGES[i % TAP_STAGES.len()], r))
                    });
            if let Some((l, stage, r)) = first_diff {
                println!("  first difference set={name}: layer {l} {stage} at row {r}");
            }
            if let Some(last) = &single_last {
                for (l, (a, b)) in bt.sel.iter().zip(&last.sel).enumerate() {
                    let (Some((ra, sa)), Some((rb, sb))) = (a, b) else {
                        continue;
                    };
                    let scored = ids.len() / POOL;
                    let dev = sa[..scored]
                        .iter()
                        .zip(&sb[..scored])
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "  selection at the last row set={name} layer={l}: lists_same={} rows={} widest_score={dev:.3e}",
                        ra == rb,
                        ra.len()
                    );
                }
            }
        }
        let (logits_worst, step) = worst(&batch_logits, &single_logits);
        let pass =
            stage_worst <= SINGLE_BAND && head_rel <= SINGLE_BAND && logits_worst <= SINGLE_BAND;
        // 3. The free chains' distance from the decode chain, a measure.
        let decode = decode_prompt(m, ids)?;
        let decode_logits = m.logits()?;
        println!(
            "batch_is_single set={name} ids={} stages_worst={stage_worst:.3e} ({at}) \
             hc_head={head_rel:.3e} logits_worst={logits_worst:.3e} (step {step}) \
             band={SINGLE_BAND:.1e} top1 batch={got} single={first} | free chain vs decode: \
             logits_rel={:.3e} top1 decode={decode} {}",
            ids.len(),
            rel(&batch_logits[0], &decode_logits),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `chunking_is_invariant` of the prompt `name`.
    fn chunking_is_invariant(
        m: &mut GpuModel<Body>,
        (large, small): (&mut Prompt, &mut Prompt),
        name: &str,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        if !tier::run_clause(
            &format!("chunking_is_invariant set={name}"),
            Tag::SelfConsistency,
        )? {
            return Ok(true);
        }
        let first = batch_prompt(m, large, ids)?;
        let (feed, a) = continuation(m, first, None)?;
        let got = batch_prompt(m, small, ids)?;
        let (_, b) = continuation(m, got, Some(&feed))?;
        let (w, at) = worst(&b, &a);
        let pass = w <= CHUNK_BAND;
        println!(
            "chunking_is_invariant set={name} ids={} worst_logits_rel={w:.3e} (step {at}) \
             band={CHUNK_BAND:.1e} top1 large={first} small={got} {}",
            ids.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `refusals_are_named`: the geometries and calls the batch refuses, each by its own
    /// name, and the model decoding after a refused call.
    fn refusals_are_named(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<bool, GateError> {
        if !tier::run_clause("refusals_are_named", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let mut ok = true;
        let mut check = |what: &str, got: Result<(), String>, want: &str| {
            let (hit, text) = match got {
                Ok(()) => (false, "accepted".to_string()),
                Err(e) => (e.contains(want), e),
            };
            println!("  refusal {what}: {text}");
            ok &= hit;
        };
        let (gpu, w, body) = m.body_parts(NAME)?;
        let try_cfg = |cfg: Cfg| {
            Prompt::new(gpu, w, body, cfg)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        check(
            "an attention chunk past the index ring",
            try_cfg(Cfg {
                q_rows: Q_MAX + 1,
                ..SMALL
            }),
            "index ring",
        );
        check(
            "a feed-forward unit past the GEMM's slots",
            try_cfg(Cfg {
                ffn_rows: 4096,
                ..SMALL
            }),
            "slots",
        );
        check(
            "a walk of no row",
            try_cfg(Cfg { cap: 0, ..SMALL }),
            "needs rows",
        );
        let mut p = Prompt::new(gpu, w, body, SMALL)?;
        m.reset()?;
        check(
            "a prompt of no id",
            prompt(m, &mut p, &[])
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "no id",
        );
        let past = vec![ids[0]; CTX + 1];
        check(
            "a prompt past the context end",
            prompt(m, &mut p, &past)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "exceeds the resident cache",
        );
        m.reset()?;
        let first = prompt(m, &mut p, &ids[..8])?;
        let next = m.step(&[first])?;
        println!("  the model decodes after the refusals: {first} then {next}");
        println!("refusals_are_named {}", verdict(ok));
        Ok(ok)
    }

    /// The placed pass's clauses on the fixture with a host share: `m` is the whole-card model whose
    /// decode chain the forced clauses run on, `sets` the prompt sets, `e` set `e`'s ids.
    fn placed_pass_clauses(
        m: &mut GpuModel<Body>,
        sets: &[(&str, Vec<u32>)],
        e: &[u32],
    ) -> Result<bool, GateError> {
        let split = open_split(Arch::Hy4, RECIPE)?;
        let inputs = PlanInputs::read(&split)?;
        let machine =
            workstation::placed_on(gate_card::card()?, inputs.hp.n_layer, workstation::SCRATCH);
        let shape = PassShape::of(PASS_ROWS);
        let ctx = usize::try_from(PLACED_CTX)?;
        let pass_inputs = hy4_pass::inputs_for(&inputs, ctx, shape)?;
        let levers = placed_load::levers(&pass_inputs, &machine, Tier::Fixture)?;
        let plan = pass_inputs.plan(&machine, PLACED_CTX, &levers)?;
        let t = std::time::Instant::now();
        let mut pm = Body::open_placed_pass(
            open_split(Arch::Hy4, RECIPE)?,
            &plan,
            &pass_inputs,
            0,
            HOST_CFG,
            shape.rows,
        )?;
        placed_load::load_line(pm.resident_bytes(), inputs.hp.n_layer, t.elapsed());
        let hosted = pm.body(NAME)?.hosted_layers();
        if hosted == 0 {
            return Err(format!(
                "{NAME}: the placed plan holds every expert on the card, so no layer has a host \
                 share to walk the pass over"
            )
            .into());
        }
        let mut ok = true;
        {
            let (gpu, _w, body) = pm.body_parts(NAME)?;
            body.pass_mut()
                .ok_or("the placed load carries its pass")?
                .arm_taps(gpu)?;
        }
        for (name, ids) in sets {
            ok &= forced_is_decode(
                m,
                name,
                ids,
                ("placed", &mut |_m, ids| {
                    pm.reset()?;
                    prompt_placed(&mut pm, ids)?;
                    let pass = pm
                        .body(NAME)?
                        .pass()
                        .ok_or("the placed load carries its pass")?;
                    Ok(pass.taps(pm.gpu().stream(), ids.len())?)
                }),
            )?;
        }
        ok &= placed_pass_is_census(&mut pm, e, hosted)?;
        ok &= placed_pass_is_reserved(&pm, (&plan, &pass_inputs), (&inputs, &machine, &levers))?;
        if tier::run_clause("placed_residency_follows_the_map", Tag::SelfConsistency)? {
            pm.reset()?;
            let first = prompt_placed(&mut pm, e)?;
            let off = continuation(&mut pm, first, None)?;
            drop(pm);
            ok &= placed_residency_follows_the_map((&plan, &pass_inputs), e, &off)?;
        }
        ok &= placed_pass_streams(m, (&inputs, &machine), e)?;
        Ok(ok)
    }

    /// Clause `placed_pass_streams` (the module header): `m` the whole-card model whose decode chain
    /// the forced halves run on, `inputs` and `machine` the fixture's, `e` set `e`'s ids.
    fn placed_pass_streams(
        m: &mut GpuModel<Body>,
        (inputs, machine): (&PlanInputs, &model::placement::Machine),
        e: &[u32],
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "placed_pass_streams";
        if !tier::run_clause(CLAUSE, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let ctx = usize::try_from(PLACED_CTX)?;
        let pass_inputs = hy4_pass::streaming_for(inputs, ctx, STREAM_SHAPE)?;
        let levers = placed_load::levers(&pass_inputs, machine, Tier::Fixture)?;
        let plan = pass_inputs.plan(machine, PLACED_CTX, &levers)?;
        let residency = residency_clauses::mid_of(&plan.n_l).ok_or_else(|| {
            format!("{CLAUSE}: no routed layer holds a pinned expert, a spare and a moving slot")
        })?;
        let mut sm = Body::open_placed_with(
            open_split(Arch::Hy4, RECIPE)?,
            (&plan, &pass_inputs),
            0,
            HOST_RESIDENCY,
            (Some(STREAM_SHAPE), residency),
        )?;
        let hosted = sm.body(NAME)?.hosted_layers();
        {
            let (gpu, _w, body) = sm.body_parts(NAME)?;
            body.set_xstream(gpu, XMode::Split)?;
            body.set_xstream_costs(SPLIT_COSTS)?;
            println!(
                "{CLAUSE} load shape={STREAM_SHAPE:?} hosted_layers={hosted} xstream={}",
                body.xstream_word()
            );
            body.pass_mut()
                .ok_or("the placed load carries its pass")?
                .arm_taps(gpu)?;
        }
        let ids = repeated(e, STREAM_IDS);
        sm.reset()?;
        let (s0, c0) = census_of(&sm)?;
        prompt_placed(&mut sm, &ids)?;
        let (s1, c1) = census_of(&sm)?;
        let (xlayers, xend, joined) = {
            let body = sm.body_parts(NAME)?.2;
            let (xlayers, xend) = body.take_xstream_records();
            (xlayers, xend, body.take_stream_joined())
        };
        // The walk's own rows, which the forced halves run the decode chain on.
        let decode_ok = forced_is_decode(
            m,
            "e",
            &ids,
            ("streamed", &mut |_m, ids| {
                let pass = sm
                    .body(NAME)?
                    .pass()
                    .ok_or("the placed load carries its pass")?;
                Ok(pass.taps(sm.gpu().stream(), ids.len())?)
            }),
        )?;
        let units = STREAM_IDS.div_ceil(STREAM_SHAPE.rows);
        let records_ok = stream_records_add_up(&xlayers, xend.as_ref(), &joined, units);
        let streamed: usize = xlayers.iter().map(|(_, x)| x.streamed).sum();
        let columns: u64 = xlayers.iter().map(|(_, x)| x.streamed_columns).sum();
        let served = s1.batch_served - s0.batch_served;
        let left_out = c1.streamed - c0.streamed;
        let (picks, card, file) = (c1.picks - c0.picks, c1.card - c0.card, c1.file - c0.file);
        let want_served = (hosted * units) as u64;
        let census_ok = streamed > 0
            && left_out == columns
            && left_out <= card
            && picks == card + file
            && served == want_served;
        println!(
            "{CLAUSE} ids={STREAM_IDS} units={units} streamed={streamed} streamed_columns={columns}              left_out={left_out} picks={picks} card={card} file={file} batch_served={served} (want              {want_served}) records={} {}",
            verdict(records_ok),
            verdict(census_ok)
        );
        let ok = decode_ok && records_ok && census_ok;
        println!("{CLAUSE} {}", verdict(ok));
        Ok(ok)
    }

    /// The host tier's service counters and census of `m` since its load.
    fn census_of(
        m: &GpuModel<Body>,
    ) -> Result<
        (
            bloomery_gpu::host::HybridStats,
            bloomery_gpu::host::census::TierCensus,
        ),
        GateError,
    > {
        let b = m.body(NAME)?;
        let stats = b
            .host_stats()
            .ok_or("a placed load carries the host tier")?;
        let census = b
            .host_census()
            .ok_or("the host tier counts no reads, so the stream's census has no picks to hold")?;
        Ok((stats, census))
    }

    /// One streaming call's records `xlayers` and `xend` and its ring passes `joined` (walk, layer,
    /// experts joined) over walks of `units` units: every streamed layer's ring pass joined its
    /// streamed experts once a unit, no other layer's did, and the end counts the layers' sum. Each
    /// disagreement is printed by name.
    fn stream_records_add_up(
        xlayers: &[(usize, XLayer)],
        xend: Option<&XReport>,
        joined: &[(usize, usize, usize)],
        units: usize,
    ) -> bool {
        let mut want: Vec<(usize, usize, usize)> = xlayers
            .iter()
            .filter(|(_, x)| x.streamed > 0)
            .flat_map(|(w, x)| std::iter::repeat_n((*w, x.layer, x.streamed), units))
            .collect();
        let mut got = joined.to_vec();
        want.sort_unstable();
        got.sort_unstable();
        let mut bad = Vec::new();
        if got != want {
            bad.push(format!(
                "joins {got:?}, the streamed layers' a unit {want:?}"
            ));
        }
        let streamed: usize = xlayers.iter().map(|(_, x)| x.streamed).sum();
        let layers = xlayers.iter().filter(|(_, x)| x.streamed > 0).count();
        match xend {
            Some(r) if r.streamed == streamed && r.layers == layers => {}
            Some(r) => bad.push(format!(
                "the end counts {} experts over {} layers, the layers {streamed} over {layers}",
                r.streamed, r.layers
            )),
            None => bad.push("no stream end recorded".to_string()),
        }
        for b in &bad {
            println!("  placed_pass_streams records: {b}");
        }
        bad.is_empty()
    }

    /// Clause `placed_residency_follows_the_map` (the module header): `off` the history of the
    /// plan's load with no machine, set `e` (`prompt`) through its pass and its own greedy ids.
    fn placed_residency_follows_the_map(
        (plan, pass_inputs): (&model::placement::Plan<'_>, &PlanInputs),
        prompt: &[u32],
        off: &(Vec<u32>, Vec<Vec<f32>>),
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "placed_residency_follows_the_map";
        let residency = residency_clauses::mid_of(&plan.n_l).ok_or_else(|| {
            format!("{CLAUSE}: no routed layer holds a pinned expert, a spare and a moving slot")
        })?;
        let mut mm = Body::open_placed_with(
            open_split(Arch::Hy4, RECIPE)?,
            (plan, pass_inputs),
            0,
            HOST_RESIDENCY,
            (Some(PassShape::of(PASS_ROWS)), residency),
        )?;
        let layers = {
            let b = mm.body(NAME)?;
            TierBody::hybrid(b).slots().layers()
        };
        let orders =
            residency_clauses::host_first_orders(plan, layers, pass_inputs.hp.experts.n_expert);
        let order: Vec<&[u32]> = orders.iter().map(Vec::as_slice).collect();
        let reseed = mm
            .residency_reseed(&order)?
            .ok_or_else(|| format!("{CLAUSE}: the {residency:?} load runs no machine"))?;
        mm.reset()?;
        let first = prompt_placed(&mut mm, prompt)?;
        let (ids, rows) = continuation(&mut mm, first, Some(&off.0))?;
        let (worst, at) = worst(&rows, &off.1);
        let flipped = reseed.copies > 0 && reseed.diff == 0;
        let ok = flipped && rows.len() == off.1.len() && worst <= RESIDENCY_BAND;
        println!(
            "{CLAUSE} {residency:?}: the reseed copied {} experts onto the card, {} entries off \
             its seed; fed the off ids {:?} (its own {ids:?}), worst logits row relative L2 \
             {worst:.3e} at row {at}, band {RESIDENCY_BAND:.1e} {}",
            reseed.copies,
            reseed.diff,
            off.0,
            verdict(ok)
        );
        Ok(ok)
    }

    /// Clause `placed_pass_is_census`: `PLACED_IDS` ids through the pass serve every hosted layer
    /// once a walk over every column, the picks being the card's and the file's, and the same ids
    /// stepped leave the batch counters alone.
    fn placed_pass_is_census(
        pm: &mut GpuModel<Body>,
        e: &[u32],
        hosted: usize,
    ) -> Result<bool, GateError> {
        if !tier::run_clause("placed_pass_is_census", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let ids = repeated(e, PLACED_IDS);
        let read = |pm: &GpuModel<Body>| -> Result<_, GateError> {
            let b = pm.body(NAME)?;
            let stats = b
                .host_stats()
                .ok_or("a placed load carries the host tier")?;
            let census = b.host_census().ok_or(
                "the host tier counts no reads, so the prompt census has no picks to hold",
            )?;
            Ok((stats, census))
        };
        pm.reset()?;
        let (s0, c0) = read(pm)?;
        prompt_placed(pm, &ids)?;
        let (s1, c1) = read(pm)?;
        let hosted = hosted as u64;
        let walks = PLACED_IDS.div_ceil(PASS_ROWS) as u64;
        let served = s1.batch_served - s0.batch_served;
        let cols = s1.batch_cols - s0.batch_cols;
        let (picks, card, file, streamed) = (
            c1.picks - c0.picks,
            c1.card - c0.card,
            c1.file - c0.file,
            c1.streamed - c0.streamed,
        );
        let pass_ok = served == hosted * walks
            && cols == hosted * PLACED_IDS as u64
            && picks > 0
            && picks == card + file
            && streamed == 0;
        println!(
            "placed_pass_is_census phase=prompt ids={PLACED_IDS} walks={walks} hosted_layers={hosted} \
             batch_served={served} (want {}) batch_cols={cols} (want {}) picks={picks} card={card} \
             file={file} streamed={streamed} {}",
            hosted * walks,
            hosted * PLACED_IDS as u64,
            verdict(pass_ok)
        );
        // The same ids one a step: the batch port stays idle and the step port serves.
        pm.reset()?;
        let (t0, _) = read(pm)?;
        for &id in &ids[..e.len()] {
            pm.step(&[id])?;
        }
        let (t1, _) = read(pm)?;
        let batch_idle = t1.batch_served == t0.batch_served && t1.batch_cols == t0.batch_cols;
        let step_served = t1.served - t0.served;
        let steps_ok = batch_idle && step_served > 0;
        println!(
            "placed_pass_is_census phase=steps ids={} batch_idle={batch_idle} step_served={step_served} {}",
            e.len(),
            verdict(steps_ok)
        );
        Ok(pass_ok && steps_ok)
    }

    /// Clause `placed_pass_is_reserved`: the load's resident bytes stand inside the plan's terms with
    /// the pass's bytes the prompt reserve, and a plan that reserves none refuses the load by name.
    fn placed_pass_is_reserved(
        pm: &GpuModel<Body>,
        (plan, pass_inputs): (&model::placement::Plan<'_>, &PlanInputs),
        (inputs, machine, levers): (
            &PlanInputs,
            &model::placement::Machine,
            &model::placement::PlanLevers,
        ),
    ) -> Result<bool, GateError> {
        if !tier::run_clause("placed_pass_is_reserved", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let held = placed_load::resident_inside_the_plan(
            pm.resident_bytes() as u64,
            plan,
            pass_inputs.prompt_reserve,
        )?;
        let pass = pm
            .body(NAME)?
            .pass()
            .ok_or("the placed load carries its pass")?
            .bytes();
        let within = pass as u64 <= pass_inputs.prompt_reserve;
        println!(
            "placed_pass_is_reserved pass_bytes={pass} reserve={} {}",
            pass_inputs.prompt_reserve,
            verdict(within)
        );
        // A plan with no reserve for the pass: the load refuses, naming the pass's buffers.
        let none = inputs.clone().with_prompt_reserve(0);
        let plan0 = none.plan(machine, PLACED_CTX, levers)?;
        let refused = Body::open_placed_pass(
            open_split(Arch::Hy4, RECIPE)?,
            &plan0,
            &none,
            0,
            HOST_CFG,
            PASS_ROWS,
        );
        let named = match refused {
            Ok(_) => false,
            Err(e) => {
                let text = e.to_string();
                println!("  a plan with no reserve: {text}");
                text.contains("placed pass buffers")
            }
        };
        println!(
            "placed_pass_is_reserved no_reserve_refused_by_name={named} {}",
            verdict(named)
        );
        Ok(held && within && named)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the released file (124 GB) does not fit one card and this gate opens the \
                 whole-card body, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<(&str, Vec<u32>)> = SETS
            .iter()
            .map(|&n| Ok((n, set_ids(n)?)))
            .collect::<Result<_, GateError>>()?;
        let e = &sets[SETS.len() - 1].1;
        if e.len() != 93 || e.len() > TAP_ROWS {
            return Err(format!("{NAME}: set e is {} ids, the gate holds 93", e.len()).into());
        }
        let file = open_split(Arch::Hy4, RECIPE)?;
        let t = std::time::Instant::now();
        let mut m = Body::open(Gpu::new()?, file, CTX)?;
        let kinds = m.body(NAME)?.kinds();
        if kinds != KINDS {
            return Err(
                format!("{NAME}: the fixture's layers are {kinds:?}, not {KINDS:?}").into(),
            );
        }
        println!(
            "opened the fixture: resident_bytes={} store_bytes={} ctx={CTX} in {:.1} s",
            m.resident_bytes(),
            m.body(NAME)?.store_bytes(),
            t.elapsed().as_secs_f64()
        );
        let tapped = Cfg {
            cap: TAP_ROWS,
            q_rows: Q_MAX,
            ffn_rows: Cfg::PROMPT.ffn_rows,
        };
        let (mut p, mut one) = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            (
                Prompt::new(gpu, w, body, tapped)?,
                Prompt::new(gpu, w, body, ONE)?,
            )
        };
        p.arm_taps(m.gpu())?;
        one.arm_taps(m.gpu())?;
        let mut ok = true;
        for (name, ids) in &sets {
            ok &= forced_is_decode(
                &mut m,
                name,
                ids,
                ("forced", &mut |m, ids| {
                    batch_prompt(m, &mut p, ids)?;
                    Ok(p.taps(m.gpu().stream(), ids.len())?)
                }),
            )?;
        }
        let long = repeated(e, LONG);
        for (name, ids) in sets.iter().map(|(n, i)| (*n, i)).chain([("long", &long)]) {
            ok &= stores_are_read_by_decode(&mut m, &mut p, name, ids)?;
        }
        for (name, ids) in sets.iter().map(|(n, i)| (*n, i)).chain([("long", &long)]) {
            ok &= batch_is_single(&mut m, (&mut p, &mut one), name, ids)?;
        }
        drop(one);
        p.disarm_taps();
        let mut small = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            Prompt::new(gpu, w, body, SMALL)?
        };
        ok &= chunking_is_invariant(&mut m, (&mut p, &mut small), "e", e)?;
        ok &= chunking_is_invariant(&mut m, (&mut p, &mut small), "long", &long)?;
        drop(p);
        let mut big_p = {
            let (gpu, w, body) = m.body_parts(NAME)?;
            Prompt::new(gpu, w, body, Cfg::PROMPT)?
        };
        println!("batch bytes at the released geometry: {}", big_p.bytes());
        let big = repeated(e, BIG);
        ok &= chunking_is_invariant(&mut m, (&mut big_p, &mut small), "big", &big)?;
        ok &= refusals_are_named(&mut m, e)?;
        ok &= placed_pass_clauses(&mut m, &sets, e)?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: every attention within {ATTN_BAND:.1e} and every feed-forward block \
             within {FFN_BAND:.1e} of the decode chain's on the batch's own inputs with its picks \
             and selection, the stores the batch left read by the decode chain within \
             {STORE_BAND:.1e}, the batch the single-id batch within {SINGLE_BAND:.1e} in every stage \
             and store on the sets and {LONG} ids, the geometry invariant within {CHUNK_BAND:.1e} \
             up to {BIG} ids, the refusals named, and the placed pass within the same bands with \
             its serves and picks counted and its bytes reserved"
        );
        Ok(())
    }
}
