//! The Solar-Open2 prefill gate: the prompt batch (`bloomery_gpu_solar2::prefill`: up to `T_MAX` rows a unit, every layer over
//! the whole unit, the KDA scan over the rows, the GQA flash over the unit, the routed block over its slots) is the one-token
//! step run per id to within the error model's band, in the logits row and in every layer's output, and the stores it leaves
//! are the ones the next step reads; the prompt group changes no bit.
//!
//! The batch keeps its own arithmetic (the GEMM sites and the f16 prefill flash), so it is the steps' within a band, not
//! bit for bit. The band is the error model's ([`SIGMA_LOGIT`](bloomery_gpu_gates::prompts::SIGMA_LOGIT), [`seat::tap_ceil`]): the two paths differ first by sum order
//! (`P = 1`: each GEMM site adds the same block dots as the gemv in another f32 order) or by the prefill flash's f16 query
//! and weights (`P ≥ 2`), and every site after quantizes its input to q8_1 blocks of 32 values. Two inputs `e` apart
//! (`e` well under the step `d = amax/127`) round to codes a whole step apart with probability `e/d` a value, so past one
//! site they are `√(e² + e·d)` apart: the map has no fixed point under `d`, and either seed grows toward two independent
//! q8 roundings within a few sites, and no further. The band is therefore two implementations' difference of one
//! quantized model, the one `gate_solar2_e2e` holds ours to PAI's by; the f16 rounding (2^-11 against q8's 1.29e-2 a
//! value) adds nothing to it. The fixture tier's four random-weight layers have no early stretch, so every layer takes the
//! wide ceiling there ([`seat::tap_ceil`]).
//!
//! The prompt lengths straddle the unit: `P = 1` (a unit of one row, the step's shape), `7` (rows short of any alignment),
//! `T_MAX` (one whole unit) and `T_MAX + 1` (a whole unit and a unit of one row: the KDA state, the conv ring and the K/V
//! planes cross the cut between two units). The ids are the fixed walk of `e2e::walk`, none of the oracle's: the clauses compare
//! the engine's two paths to each other and need no reference.
//!
//! The clauses (all self-consistency, so both tiers run them) are the common ones ([`prefill_clauses`]):
//! - `prefill_near_steps`: from a reset, `prompt(P ids)` and `P` steps give last logits rows within [`logprobs_near`] (the
//!   steps' top [`TOP`] ids) and layer outputs at the last position within [`seat::tap_ceil`], the same store list
//!   ([`Seat::stores`]: each KDA layer's state, stamp and conv ring, each GQA layer's K and V planes over the `P` written
//!   positions; the digests are printed, every one moves with the first rounding), and the position is `P`.
//! - `prefill_then_step_near_steps`: after `prompt(P ids)` one more step gives the logits row and layer outputs of `P + 1`
//!   steps within the same bands, so the stores a batch leaves are the ones the next step reads.
//! - `prefill_group_is_group_one`: at `P = T_MAX + 1` with the prompt group at 2 (the two units one walk, the second unit's
//!   front under the first's host serve) the prompt gives the group of 1's logits row, layer outputs and store digests bit
//!   for bit: the group orders the units' walks, not any sum. The group is put back at 1.
//! - `fault_word_clean`: the fault word is clean after every run.
//!
//! `--arm a|bp` picks the placement arm as `gate_solar2_e2e` does.
//!
//! FAIL-first (box): a batch that leaves each GQA layer's K rows in the V plane and its V rows in the K plane (its own flash
//! reading them the right way round) puts `prefill_then_step_near_steps` red at every `P` (the logits' worst z 5.2 at
//! `P = 1`, 26 to 35 at `P >= 7` with layer 0 past its ceiling) while `prefill_near_steps` and the group clause stay green.

#[cfg(feature = "solar2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_prefill", gate::run())
}

#[cfg(feature = "solar2")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the prefill clauses serve several gates; each takes the helpers they read"
)]
#[path = "shared/prefill_clauses.rs"]
mod prefill_clauses;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/solar2_seat.rs"]
mod seat;

#[cfg(feature = "solar2")]
mod gate {
    use crate::lcpp_e2e::arm_arg;
    use crate::prefill_clauses::Case;
    use crate::seat;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::prompts::k_of;
    use bloomery_gpu_gates::tier::Tier;
    use bloomery_gpu_solar2::body::T_MAX;

    const NAME: &str = "gate_solar2_prefill";

    /// The prompt lengths: one row, rows short of an alignment, one whole unit, a unit and a row.
    const LENGTHS: [usize; 4] = [1, 7, T_MAX, T_MAX + 1];

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// The ids of the steps' row the logits band reads.
    const TOP: usize = 40;

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let path = seat::model_path(tier_now, "gate-gpu-solar2-prefill")?;
        let ctx = T_MAX + 1 + CTX_SPARE;
        let t = std::time::Instant::now();
        let arm = arm_arg()?;
        let mut seat = seat::open(&path, ctx, arm)?;
        let dims = seat.dims();
        println!(
            "load arm={} layers={} embd={} vocab={} ctx={ctx} t_max={T_MAX} in {:.1} s (runtime value)",
            arm.word(),
            dims.layers,
            dims.embd,
            dims.vocab,
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            lengths: LENGTHS.to_vec(),
            tap_ceil: seat::tap_ceil,
            tap_ceils: seat::tap_ceils,
            group: 2,
            logit_bound: Some(k_of(TOP)),
            vocab: dims.vocab,
            embd: dims.embd,
            tier: tier_now,
        };
        crate::prefill_clauses::run(NAME, &case, seat.as_mut())
    }
}
