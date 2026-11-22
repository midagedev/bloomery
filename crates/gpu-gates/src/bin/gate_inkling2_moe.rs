//! GPU gate for Inkling 2's MoE leg on the card: the biased sigmoid router at
//! 128 experts and the top 8 (`bloomery_gpu::route_body::sigmoid_128_8` — the
//! selection bias steering the pick only, the kept unbiased scores renormalized
//! over their `1e-20`-guarded sum and scaled by the file's
//! `expert_weights_scale`), the fused `ffn_gate_up_exps` split at upload into
//! the gate and up stacks that share the per-expert scale
//! (`Weights::split_fused_w13`), and the NVFP4 and Q8_0 expert legs under SiTU
//! at the file's betas (`CardLeg`: the gate·up `_sel`, the SiTU rows, the down
//! `_sel` over 32-value blocks, the slots' weighted sum), against the oracle
//! node dumps of the patched llama.cpp PR #32695 tree — whichever tier this run
//! holds, the fixture's sets (`fx_ref_inkling2_*`, refset family
//! `fx-lcpp-inkling2`, thirteen stored layers of the Mini with the coda
//! layer's experts in Q8_0) or the Mini's own (`ref_inkling2_*`,
//! `lcpp-inkling2`). The router's input is each visit's ffn norm, read from the
//! set's `norm-<slot>` rows (the occurrence whose source the manifest names
//! `(reshaped)` or `loop_ffn_inp-<slot>` — the visit's last norm of each graph
//! the dump ran, one a ubatch), and the expert taps are set a's, the one set
//! that dumps the whole ladder.
//!
//! The clauses, one a contract:
//! - `router_ids_*`: every set, every visit, every token's kept ids EQUAL the
//!   vendor's `ffn_moe_topk` in rank order; the detail names how many tokens
//!   sit inside the tie margin (a biased cut gap of [`TIE_MARGIN`] relative,
//!   where the two engines' rounding could flip a pick — a flip there is a tie
//!   to adjudicate, not a hidden pass).
//! - `router_weights_*`: every set's `ffn_moe_weights_scaled` within
//!   [`WEIGHTS_BAND`] by relative L2, the worst visit's printed.
//! - `router_probs_a`: set a's `ffn_moe_probs` within [`PROBS_BAND`], the
//!   worst visit's.
//! - `router_logits_a`: set a's `ffn_moe_logits` within [`LOGITS_BAND`] at
//!   sampled tokens of every visit, through the one-token launch (the decode
//!   path — the batch's scores entry writes no logits).
//! - `router_one_token_is_batch`: the one-token launch's scores, ids and
//!   weights BIT-EQUAL the batch's slices at those tokens.
//! - `router_guard_rule`: a seeded row whose every logit sits far below zero,
//!   where the `+1e-20` divisor and the selected-sum floor differ: the weights
//!   BIT-EQUAL the host `+1e-20` rule on the kernel's own scores, and every
//!   one of them differs from the floor rule's.
//! - `split_gate_is_vendor` / `split_up_is_vendor`: the raw halves of the
//!   split — each half stack dotted with the vendor's own normed rows over
//!   every token and slot — within [`HALVES_BAND`] of `ffn_moe_gate` /
//!   `ffn_moe_up`, on the NVFP4 layer and the Q8_0 coda alike.
//! - `leg_situ_is_vendor` / `leg_down_is_vendor` / `leg_out_is_vendor`: the
//!   whole leg over the vendor's normed rows with the router's own ids and
//!   weights — the SiTU rows, the down rows and their weighted sum — within
//!   [`SITU_BAND`] / [`DOWN_BAND`] / [`OUT_BAND`] of `ffn_moe_situ`,
//!   `ffn_moe_down(_scaled)` and `ffn_moe_out`, on both quant legs.
//! - `no_fault`: nothing raised the fault word.
//! - `refused`: a split of a name that is not resident, of a non-stack weight,
//!   of rows that are not whole experts, a second split of the same stack, and
//!   a router weight of the wrong shape, each refused by name before anything
//!   is enqueued.
//!
//! The bands hold what the two engines' rounding gives, [derived] from where
//! their quantizations agree and differ:
//! - every quantized dot of the reference's CPU path (its NVFP4
//!   `vec_dot_nvfp4_q8_1`, its Q8_0 `vec_dot_q8_0_q8_1`) reads its input row
//!   as q8_1 blocks of 32 values (a step of the block's largest magnitude
//!   over 127), and so does our `Q8Blocks32` — on a site where both sides
//!   quantize the same blocks of the same row the activation share cancels,
//!   and what is left is the pair of f32 sum orders and the reference's bf16
//!   accumulation of the NVFP4 products (2^-9 relative a partial, over the
//!   dot's partials a share of 2^-9/√m at the 2^6..2^7 partials a row's walk
//!   rounds);
//! - our gate·up reads its activation as a `Q8Act` — q8_1 super-blocks of 256
//!   values — where the reference's dot quantizes blocks of 32: the two
//!   quantizations differ by the step their block sizes give (each under a
//!   percent of the row's RMS at the ratio a 256-value span's largest
//!   magnitude bears to a 32-value one), and their difference passes the dot
//!   as the root of the two shares' squares — about one percent;
//! - a site downstream (the down over the SiTU rows) quantizes blocks of 32
//!   on both sides again, so it carries its input's share through and adds
//!   only its own pair of f32 orders and the reference's bf16 accumulation.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_moe", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::kquant::Act;
    use bloomery_gpu::mxfp4_sel::{Nvfp4Down, Nvfp4SelKernels};
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::q5::Q8Blocks32;
    use bloomery_gpu::q8_0_sel32::{Q80SelDown, Q80SelKernels};
    use bloomery_gpu::route_body::sigmoid_128_8;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, ref_tensor_logical_in, split_f32,
        topk_ids_logical_within, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::inkling2::hparams::{Hparams, Loop};
    use model::arch::models::{LoopSpec, visits};
    use model::placement::CardFormat;
    use model::placement::accepts::LegArms;
    use model::placement::kernels::DownAct;
    use refset::arch::inkling2;
    use refset::family::Family;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_inkling2_moe";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-moe";
    /// What the refusals of this gate name.
    const WHAT: &str = "gate_inkling2_moe";

    /// The router's logits against the vendor's, relative L2 of a token's row.
    /// [derived] Each engine's f32 dot over the shared weight and row sits
    /// within ε·√k of the exact sum; the pair twice that; the band five times
    /// the pair.
    const LOGITS_BAND: f64 = 2e-5;
    /// The router's sigmoid scores against the vendor's, relative L2 of a
    /// visit's whole score set. [derived] The scores are the logits' sigmoid —
    /// a contraction — so the logits' share carries at the same band.
    const PROBS_BAND: f64 = 2e-5;
    /// The kept slots' scaled weights against the vendor's, relative L2 of a
    /// visit's weights. [derived] The weights are the scores' own bits through
    /// the same division and scale in both engines, so the scores' share
    /// carries; the band five times it.
    const WEIGHTS_BAND: f64 = 1e-5;
    /// The split's raw halves against the vendor's, relative L2 of a layer's
    /// whole half product. [derived] Both sides quantize the input row as q8_1
    /// blocks of the same 32 values, so the activation share cancels and what
    /// is left is the pair of f32 sum orders and the reference's bf16
    /// accumulation of the NVFP4 products: 2^-9 relative a partial, over the
    /// dot's 2^6..2^7 partials a share of 2^-9/√m ≈ 2-3·10^-4; the band at
    /// three times that bound.
    const HALVES_BAND: f64 = 1e-3;
    /// The leg's SiTU rows against the vendor's, relative L2 of the layer's
    /// rows. [derived] The gate·up reads a `Q8Act` of 256-value super-blocks
    /// where the reference's dot quantizes blocks of 32: the two
    /// quantizations' difference passes the dot as the root of the two
    /// shares' squares — each a step of its block's largest magnitude over
    /// 127, about 0.7-0.8 % of the row's RMS — so about 10^-2, with the
    /// halves' share beside it; the band at three times that bound.
    const SITU_BAND: f64 = 3e-2;
    /// The leg's down rows against the vendor's, relative L2 of the layer's
    /// rows. [derived] The down quantizes blocks of 32 on both sides again,
    /// so it carries the SiTU rows' share through its weights (about the
    /// quantizations' 10^-2) and adds only the f32 orders and the reference's
    /// bf16 accumulation; the band at three times that carried bound.
    const DOWN_BAND: f64 = 6e-2;
    /// The leg's weighted sum against the vendor's `ffn_moe_out`, relative L2.
    /// [derived] The down rows under weights both engines hold to the f32
    /// order — the down's share carried, nothing new added; the down's band,
    /// with headroom for the sum's own f32 order.
    const OUT_BAND: f64 = 6e-2;
    /// A biased cut gap below this relative distance is named in the ids
    /// clause's detail: the two engines' scores differ by their f32 dots'
    /// share, and a flip inside this margin is a tie to adjudicate, not a
    /// silent pass. [derived] Forty times the scores' share.
    const TIE_MARGIN: f64 = 1e-5;
    /// The seeded guard row's logit: every sigmoid score lands far enough below
    /// zero that the kept scores' sum sits under the selected-sum floor of the
    /// router rules, where the `+1e-20` addend and the floor differ.
    const GUARD_LOGIT: f32 = -40.0;

    /// One set's opened manifest and its token count.
    struct Set {
        man: RefManifest,
        tokens: usize,
    }

    /// The clauses' shared device state: the card, the router instance, the
    /// wide tile GEMM the batch's scores run on, the card leg and the two
    /// raw-half `_sel` modules.
    struct Dev {
        gpu: Gpu,
        router: sigmoid_128_8::RouterKernels,
        wide: Gemm32Kernels,
        leg: CardLeg,
        nvfp4: Nvfp4SelKernels,
        q80: Q80SelKernels,
    }

    impl Dev {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn up<T: cuda_core::DeviceCopy>(&self, v: &[T]) -> Result<DeviceBuffer<T>, GateError> {
            Ok(DeviceBuffer::from_host(self.stream(), v)?)
        }

        fn read<T: cuda_core::DeviceCopy>(&self, b: &DeviceBuffer<T>) -> Result<Vec<T>, GateError> {
            self.stream().synchronize()?;
            Ok(b.to_host_vec(self.stream())?)
        }
    }

    /// One clause's line and whether it passed.
    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{NAME} {name}: {detail} {}", verdict(pass));
        pass
    }

    /// `‖got − want‖₂ / ‖want‖₂` in f64, infinite on a length mismatch.
    fn rel64(got: &[f32], want: &[f32]) -> f64 {
        if got.len() != want.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(want) {
            num += (f64::from(g) - f64::from(w)).powi(2);
            den += f64::from(w).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// The description's loop of a file's, as [`visits`] takes it.
    fn loop_of(loop_: Option<Loop>) -> Option<LoopSpec> {
        loop_.map(|l| LoopSpec {
            core_start: l.core_start as u32,
            core_end: l.core_end as u32,
            passes: l.passes as u32,
            gate_rank: l.gate_rank as u32,
        })
    }

    /// The file's SiTU rule as the leg's launch argument.
    fn rule_of(hp: &Hparams) -> Act {
        Act::Situ {
            beta: hp.situ.gate_beta,
            linear_beta: hp.situ.value_beta,
        }
    }

    /// The leg's gate·up arm: the two forms this family's fused w13 splits
    /// into — NVFP4's planes and the file's Q8_0 blocks.
    fn gate_up_arm(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::NVFP4 | GgmlType::Q8_0)
    }

    /// The leg's down arm: the two forms the family's down stacks hold.
    fn down_arm(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::NVFP4 | GgmlType::Q8_0)
    }

    /// What a refusal of a gate or up stack names.
    const GATE_UP_ROLE: &str =
        "a resident fused-w13 half of NVFP4 planes or Q8_0 blocks (the card leg's gate·up arm)";
    /// What a refusal of a down stack names.
    const DOWN_ROLE: &str = "a resident down stack of NVFP4 planes or Q8_0 blocks";

    /// The stored layers the ladder runs on: the first whose fused w13 stack
    /// is NVFP4 and the first whose is Q8_0 — the two expert legs the family's
    /// files mix (the coda's is the Q8_0 one).
    fn ladder_layers(split: &Split, hp: &Hparams) -> Result<[usize; 2], GateError> {
        let (mut nvfp4, mut q8) = (None, None);
        for l in 0..hp.n_layer {
            let name = format!("blk.{l}.ffn_gate_up_exps.weight");
            let Some((_, t)) = split.find(&name) else {
                return Err(format!("{name} is not in the model file").into());
            };
            match t.ty {
                GgmlType::NVFP4 if nvfp4.is_none() => nvfp4 = Some(l),
                GgmlType::Q8_0 if q8.is_none() => q8 = Some(l),
                _ => {}
            }
        }
        match (nvfp4, q8) {
            (Some(a), Some(b)) => Ok([a, b]),
            _ => Err("the model file holds no NVFP4 and Q8_0 pair of expert sets".into()),
        }
    }

    /// The file's router weight and selection bias of every stored layer, once:
    /// a core layer's visits share them. The bias rides as a host copy too —
    /// the ids clause's tie-margin diagnostic reads it there.
    struct Routers {
        w: Vec<DeviceTensor<f32>>,
        bias: Vec<DeviceBuffer<f32>>,
        bias_host: Vec<Vec<f32>>,
    }

    impl Routers {
        fn of(split: &Split, dev: &Dev, hp: &Hparams) -> Result<Routers, GateError> {
            let stream = dev.stream();
            let (n_expert, embd) = (hp.n_expert, hp.n_embd);
            let mut w = Vec::with_capacity(hp.n_layer);
            let mut bias = Vec::with_capacity(hp.n_layer);
            let mut bias_host = Vec::with_capacity(hp.n_layer);
            for l in 0..hp.n_layer {
                w.push(DeviceTensor::upload(
                    stream,
                    &split_f32(
                        split,
                        &format!("blk.{l}.ffn_gate_inp.weight"),
                        n_expert * embd,
                    )?,
                    n_expert,
                    embd,
                )?);
                let host = split_f32(split, &format!("blk.{l}.exp_probs_b.bias"), n_expert)?;
                bias.push(dev.up(&host)?);
                bias_host.push(host);
            }
            Ok(Routers { w, bias, bias_host })
        }
    }

    /// One batch's router results on the device: the scores ([`Routers`]'s
    /// `n_expert` a token), the ids and the weights (`n_used` a token).
    struct Batch {
        probs: DeviceBuffer<f32>,
        ids: DeviceBuffer<u32>,
        weights: DeviceBuffer<f32>,
    }

    /// The batch router over `t` tokens of `x_rows` (the vendor's own ffn-norm
    /// rows) with layer `l`'s weight and bias: the scores, ids and weights that
    /// many one-token routers write.
    fn route_batch(
        dev: &Dev,
        routers: &Routers,
        hp: &Hparams,
        l: usize,
        x_rows: &[f32],
        t: usize,
    ) -> Result<Batch, GateError> {
        let stream = dev.stream();
        let (n_expert, n_used) = (hp.n_expert, hp.n_used[0]);
        let x = dev.up(x_rows)?;
        let mut probs = DeviceBuffer::zeroed(stream, t * n_expert)?;
        let mut ids = DeviceBuffer::zeroed(stream, t * n_used)?;
        let mut weights = DeviceBuffer::zeroed(stream, t * n_used)?;
        dev.router.enqueue_router_rows(
            stream,
            &dev.gpu,
            &dev.wide,
            &routers.w[l],
            &x,
            &routers.bias[l],
            hp.weights_scale,
            t,
            &mut probs,
            &mut ids,
            &mut weights,
            dev.gpu.unlabelled_sink(),
        )?;
        Ok(Batch {
            probs,
            ids,
            weights,
        })
    }

    /// The tier's model, the family of its sets and the sets themselves.
    fn sets_of(tier: Tier) -> (&'static Family, Vec<&'static str>) {
        match tier {
            Tier::Real => (&inkling2::LCPP, inkling2::SETS.to_vec()),
            Tier::Fixture => (
                &inkling2::fixture::LCPP,
                vec![
                    inkling2::fixture::A,
                    inkling2::fixture::B,
                    inkling2::fixture::C,
                    inkling2::fixture::D,
                    inkling2::fixture::E,
                ],
            ),
        }
    }

    /// Open the set `name` of `family` in this process's tier, checked by the
    /// family; a set absent, stale, foreign or unfinished ends the gate.
    fn open_set(tier: Tier, family: &Family, name: &str) -> Result<Set, GateError> {
        let path = match tier {
            Tier::Fixture => tier::fixture_set(NAME, family, name)?,
            Tier::Real => family.path(name),
        };
        let man = RefManifest::open(&path, family)?;
        let tokens = man
            .header
            .tokens
            .as_ref()
            .map(|t| t.len())
            .ok_or_else(|| format!("{} names no tokens line", path.display()))?;
        Ok(Set { man, tokens })
    }

    /// One set's tap `name`, occurrence `occ`, in its logical order.
    fn tap_at(man: &RefManifest, name: &str, occ: u32) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, occ)?)?)
    }

    /// The visit's ffn-norm rows — the router's input — every token of the
    /// prompt: the set's `norm-<slot>` occurrences whose source the manifest
    /// names `(reshaped)` or `loop_ffn_inp-<slot>` (the visit's last norm of
    /// each graph the dump ran, one a ubatch), concatenated in token order. A
    /// visit without such rows, or rows that do not sum to the prompt, errors
    /// naming the set.
    fn ffn_norm_rows(
        man: &RefManifest,
        slot: usize,
        (embd, tokens): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let name = format!("norm-{slot}");
        let ffn_src = format!("loop_ffn_inp-{slot}");
        let mut rows = Vec::with_capacity(tokens * embd);
        for row in man.tensors.iter().filter(|r| r.name == name) {
            let src = row.src0.as_deref().map(str::trim);
            if src != Some("(reshaped)") && src != Some(ffn_src.as_str()) {
                continue;
            }
            let vals = ref_tensor_logical_in(&man.dir, row)?;
            if vals.len() % embd != 0 {
                return Err(format!(
                    "{}: {name} occurrence {} holds {} values, not rows of {embd}",
                    man.dir.display(),
                    row.occurrence,
                    vals.len()
                )
                .into());
            }
            rows.extend(vals);
        }
        if rows.len() != tokens * embd {
            return Err(format!(
                "{}: {name} holds {} of {tokens} tokens' ffn-norm rows",
                man.dir.display(),
                rows.len() / embd
            )
            .into());
        }
        Ok(rows)
    }

    /// The vendor's kept ids of every token of the set's `ffn_moe_topk-<slot>`
    /// occurrences (one a graph the dump ran), `[tokens][n_used]` in rank
    /// order; their scaled weights of `ffn_moe_weights_scaled-<slot>` beside
    /// them.
    fn vendor_picks(
        man: &RefManifest,
        slot: usize,
        (tokens, n_used, n_expert): (usize, usize, u32),
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let (topk, scaled) = (
            format!("ffn_moe_topk-{slot}"),
            format!("ffn_moe_weights_scaled-{slot}"),
        );
        let mut ids = Vec::with_capacity(tokens * n_used);
        for row in man.tensors.iter().filter(|r| r.name == topk) {
            ids.extend(
                topk_ids_logical_within(man, row, n_expert)?
                    .into_iter()
                    .map(|id| u32::try_from(id).expect("topk_ids_logical_within holds a u32")),
            );
        }
        let mut weights = Vec::with_capacity(tokens * n_used);
        for row in man.tensors.iter().filter(|r| r.name == scaled) {
            weights.extend(ref_tensor_logical_in(&man.dir, row)?);
        }
        if ids.len() != tokens * n_used || weights.len() != tokens * n_used {
            return Err(format!(
                "{}: {topk} and {scaled} hold {} and {} of {tokens} tokens' {n_used} slots",
                man.dir.display(),
                ids.len() / n_used,
                weights.len() / n_used
            )
            .into());
        }
        Ok((ids, weights))
    }

    /// How many of `got`'s ids differ from `want`'s, and how many tokens' biased
    /// cut (their scores plus the selection bias) sits inside [`TIE_MARGIN`]:
    /// a flip there is a tie to adjudicate, not a silent pass.
    fn ids_report(
        (got, want): (&[u32], &[u32]),
        (probs, bias): (&[f32], &[f32]),
        (tokens, n_expert, n_used): (usize, usize, usize),
    ) -> (usize, usize) {
        let (mut differ, mut margin) = (0, 0);
        for t in 0..tokens {
            let (g, w) = (
                &got[t * n_used..(t + 1) * n_used],
                &want[t * n_used..(t + 1) * n_used],
            );
            differ += g.iter().zip(w).filter(|(a, b)| a != b).count();
            let col = &probs[t * n_expert..(t + 1) * n_expert];
            let (mut kept, mut left) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for (e, &p) in col.iter().enumerate() {
                let v = p + bias[e];
                if v > kept {
                    left = kept;
                    kept = v;
                } else if v > left {
                    left = v;
                }
            }
            if f64::from(kept - left) < TIE_MARGIN * f64::from(kept.abs().max(1e-9)) {
                margin += 1;
            }
        }
        (differ, margin)
    }

    /// One set's router clauses: per visit, the batch launches over the
    /// vendor's own ffn-norm rows, compared with the vendor's ids and scaled
    /// weights — and the set's scores where it dumps them.
    fn router_clauses(
        dev: &Dev,
        (hp, routers): (&Hparams, &Routers),
        set: &Set,
        label: &str,
        with_scores: bool,
    ) -> Result<Vec<bool>, GateError> {
        let (n_expert, n_used, embd) = (hp.n_expert, hp.n_used[0], hp.n_embd);
        let visit_list = visits(hp.n_layer, loop_of(hp.loop_));
        let (mut mismatches, mut margin, mut worst_w, mut worst_p) =
            (0usize, 0usize, 0.0f64, 0.0f64);
        for (slot, visit) in visit_list.iter().enumerate() {
            let l = visit.layer as usize;
            let t = set.tokens;
            let x_rows = ffn_norm_rows(&set.man, slot, (embd, t))?;
            let batch = route_batch(dev, routers, hp, l, &x_rows, t)?;
            let (got_ids, got_w, got_p) = (
                dev.read(&batch.ids)?,
                dev.read(&batch.weights)?,
                dev.read(&batch.probs)?,
            );
            let (want_ids, want_w) = vendor_picks(&set.man, slot, (t, n_used, n_expert as u32))?;
            let (differ, at_margin) = ids_report(
                (&got_ids, &want_ids),
                (&got_p, &routers.bias_host[l]),
                (t, n_expert, n_used),
            );
            mismatches += differ;
            margin += at_margin;
            worst_w = worst_w.max(rel64(&got_w, &want_w));
            if with_scores {
                let want_p = tap_at(&set.man, &format!("ffn_moe_probs-{slot}"), 0)?;
                worst_p = worst_p.max(rel64(&got_p, &want_p));
            }
        }
        let picks = set.tokens * visit_list.len() * n_used;
        Ok(vec![
            clause(
                &format!("router_ids_{label}"),
                mismatches == 0,
                &format!(
                    "{mismatches} of {picks} picks differ from the vendor's; {margin} tokens \
                     inside the tie margin {TIE_MARGIN:.0e}"
                ),
            ),
            clause(
                &format!("router_weights_{label}"),
                worst_w <= WEIGHTS_BAND,
                &format!("worst rel {worst_w:.3e} (band {WEIGHTS_BAND:.0e})"),
            ),
            clause(
                &format!("router_probs_{label}"),
                !with_scores || worst_p <= PROBS_BAND,
                &if with_scores {
                    format!("worst rel {worst_p:.3e} (band {PROBS_BAND:.0e})")
                } else {
                    "the set dumps no router scores".to_string()
                },
            ),
        ])
    }

    /// The one-token launch (the decode path) at sampled tokens of layer `l`'s
    /// first visit: its logits against the vendor's tap within
    /// [`LOGITS_BAND`], and its scores, ids and weights BIT-EQUAL the batch's
    /// slices.
    fn one_token_clauses(
        dev: &Dev,
        (hp, routers, set): (&Hparams, &Routers, &Set),
        l: usize,
    ) -> Result<Vec<bool>, GateError> {
        let (n_expert, n_used, embd) = (hp.n_expert, hp.n_used[0], hp.n_embd);
        let t = set.tokens;
        let slot = visit_of(hp, l)?;
        let x_rows = ffn_norm_rows(&set.man, slot, (embd, t))?;
        let logit_tap = tap_at(&set.man, &format!("ffn_moe_logits-{slot}"), 0)?;
        let batch = route_batch(dev, routers, hp, l, &x_rows, t)?;
        let (p_b, i_b, w_b) = (
            dev.read(&batch.probs)?,
            dev.read(&batch.ids)?,
            dev.read(&batch.weights)?,
        );
        let mut run = sigmoid_128_8::RouterOut::new(dev.stream())?;
        let mut worst = 0.0f64;
        let mut bits = true;
        for tok in [0, t / 2, t - 1] {
            let x = dev.up(&x_rows[tok * embd..(tok + 1) * embd])?;
            dev.router.enqueue_router(
                dev.stream(),
                &routers.w[l],
                &x,
                &routers.bias[l],
                hp.weights_scale,
                &mut run,
                dev.gpu.unlabelled_sink(),
            )?;
            let (lg, pr, id, wt) = (
                dev.read(&run.logits)?,
                dev.read(&run.probs)?,
                dev.read(&run.ids)?,
                dev.read(&run.weights)?,
            );
            worst = worst.max(rel64(&lg, &logit_tap[tok * n_expert..(tok + 1) * n_expert]));
            bits &= bits_equal(&pr, &p_b[tok * n_expert..(tok + 1) * n_expert])
                && id == i_b[tok * n_used..(tok + 1) * n_used]
                && bits_equal(&wt, &w_b[tok * n_used..(tok + 1) * n_used]);
        }
        Ok(vec![
            clause(
                &format!("router_logits_a l={l}"),
                worst <= LOGITS_BAND,
                &format!("worst rel {worst:.3e} over the sampled tokens (band {LOGITS_BAND:.0e})"),
            ),
            clause(
                &format!("router_one_token_is_batch l={l}"),
                bits,
                "the one-token launch's scores, ids and weights: the batch slices' bits",
            ),
        ])
    }

    /// The seeded `+1e-20` rule: a router weight whose every row dots the
    /// crafted row to [`GUARD_LOGIT`], so every sigmoid score sits where the
    /// kept scores' sum falls under the selected-sum floor and the two divisor
    /// rules differ. The kernel's weights BIT-EQUAL the host rule on the
    /// kernel's own scores (the f64 slot-order sum narrowed, the `+1e-20`
    /// addend, the division and the scale), and every one of them differs from
    /// the floor rule's.
    fn guard_clause(dev: &Dev, hp: &Hparams) -> Result<bool, GateError> {
        let stream = dev.stream();
        let (n_expert, n_used, k) = (hp.n_expert, hp.n_used[0], hp.n_embd);
        let mut wv = vec![0.0f32; n_expert * k];
        for row in wv.chunks_mut(k) {
            row[0] = 1.0;
        }
        let mut xv = vec![0.0f32; k];
        xv[0] = GUARD_LOGIT;
        let bias = vec![0.0f32; n_expert];
        let (w, x, b) = (
            DeviceTensor::upload(stream, &wv, n_expert, k)?,
            dev.up(&xv)?,
            dev.up(&bias)?,
        );
        let mut run = sigmoid_128_8::RouterOut::new(stream)?;
        dev.router.enqueue_router(
            stream,
            &w,
            &x,
            &b,
            hp.weights_scale,
            &mut run,
            dev.gpu.unlabelled_sink(),
        )?;
        let (probs, ids, weights) = (
            dev.read(&run.probs)?,
            dev.read(&run.ids)?,
            dev.read(&run.weights)?,
        );
        let slots: Vec<f32> = ids.iter().map(|&e| probs[e as usize]).collect();
        let mut sum = 0.0f64;
        for &p in &slots {
            sum += f64::from(p);
        }
        let div = sum as f32 + 1e-20;
        let floor = (sum as f32).max(1.0 / 16384.0);
        let (mut bit_ok, mut floor_differs) = (0usize, 0usize);
        for (&p, &got) in slots.iter().zip(&weights) {
            if (p / div * hp.weights_scale).to_bits() == got.to_bits() {
                bit_ok += 1;
            }
            if (p / floor * hp.weights_scale).to_bits() != got.to_bits() {
                floor_differs += 1;
            }
        }
        Ok(clause(
            "router_guard_rule",
            bit_ok == n_used && floor_differs == n_used,
            &format!(
                "{bit_ok} of {n_used} weights bit-equal the +1e-20 rule; {floor_differs} differ \
                 from the floor rule's"
            ),
        ))
    }

    /// The first visit that runs stored layer `l`, as the set's taps name it.
    fn visit_of(hp: &Hparams, l: usize) -> Result<usize, GateError> {
        visits(hp.n_layer, loop_of(hp.loop_))
            .iter()
            .position(|v| v.layer as usize == l)
            .ok_or_else(|| format!("stored layer {l} runs in no visit").into())
    }

    /// One test layer's ladder clauses: the split, its raw halves and the
    /// whole leg, against set a's expert taps at the layer's first visit. The
    /// layer's resident weights (its fused stack already split) come back for
    /// the refusal clauses.
    #[allow(
        clippy::too_many_lines,
        reason = "one clause group over the split, the halves and the leg"
    )]
    fn leg_clauses(
        dev: &Dev,
        (hp, routers, set): (&Hparams, &Routers, &Set),
        split: &Split,
        l: usize,
    ) -> Result<(Vec<bool>, Weights), GateError> {
        let stream = dev.stream();
        let (embd, ff) = (hp.n_embd, hp.expert_ff[l]);
        let (n_expert, n_used) = (hp.n_expert, hp.n_used[0]);
        let (fused, down_name) = (
            format!("blk.{l}.ffn_gate_up_exps.weight"),
            format!("blk.{l}.ffn_down_exps.weight"),
        );
        let scaled = matches!(split.find(&fused).map(|(_, t)| t.ty), Some(GgmlType::NVFP4));
        let mut w = Weights::load(stream, split, l..l + 1, false)?;
        if !scaled {
            // A Q8_0 stack's own upload is its two planes; the leg's Q8_0
            // entries read the file's blocks, so the load relayouts the
            // layer's stacks into them before the split (the engine's own
            // move for a Q8_0 routed stack under `Weights::load`,
            // `Weights::relayout`'s doc) — the fused one the split takes, the
            // down the leg reads.
            w.relayout(stream, split, &fused, CardFormat::KQuant)?;
            w.relayout(stream, split, &down_name, CardFormat::KQuant)?;
        }
        let (gate, up) = w.split_fused_w13(stream, &fused, ff)?;
        let stacks = LegStacks::of_arms(
            &w,
            [gate.as_str(), up.as_str(), down_name.as_str()],
            LegArms {
                gate_up: gate_up_arm,
                down: down_arm,
            },
            (GATE_UP_ROLE, DOWN_ROLE),
        )?;
        let slot = visit_of(hp, l)?;
        let t = set.tokens;
        let x_rows = ffn_norm_rows(&set.man, slot, (embd, t))?;
        let Batch { ids, weights, .. } = route_batch(dev, routers, hp, l, &x_rows, t)?;
        let slots = t * n_used;
        let x = dev.up(&x_rows)?;

        // The split's raw halves: each half stack dotted with the vendor's own
        // normed rows over every token and slot, the router's ids the places.
        let mut act = Q8Blocks32::with_slots(stream, embd, slots)?;
        dev.gpu.q5().enqueue_quantize_q8_rep(
            stream,
            &QuantSel {
                x: &x,
                cols: 0..slots,
                sel: &ids,
                n_card: n_expert,
            },
            n_used,
            &mut act,
            dev.gpu.layer_sink(l)?,
        )?;
        let half = |stack: &str, tap: &str| -> Result<f64, GateError> {
            let mut y = DeviceBuffer::zeroed(stream, slots * ff)?;
            match w.resident(WHAT, stack)? {
                DevWeight::Nvfp4Stack { .. } => dev.nvfp4.enqueue_sel32(
                    stream,
                    &Nvfp4Down {
                        w: w.nvfp4_stack(WHAT, stack)?,
                        act: &act,
                        sel: &ids,
                        n_slots: slots,
                        rows_per_expert: ff,
                        slots_per_col: n_used,
                    },
                    dev.gpu.layer_sink(l)?,
                    &mut y,
                )?,
                DevWeight::KQuant { w: words, .. } => dev.q80.enqueue_gemv_q8_0_sel32(
                    stream,
                    &Q80SelDown {
                        w: words,
                        act: &act,
                        sel: &ids,
                        n_slots: slots,
                        rows_per_expert: ff,
                    },
                    dev.gpu.layer_sink(l)?,
                    &mut y,
                )?,
                _ => {
                    return Err(
                        format!("{stack} is neither an NVFP4 stack nor a K-quant's words").into(),
                    );
                }
            }
            let want = tap_at(&set.man, tap, 0)?;
            Ok(rel64(&dev.read(&y)?, &want))
        };
        let halves = [
            half(&gate, &format!("ffn_moe_gate-{slot}"))?,
            half(&up, &format!("ffn_moe_up-{slot}"))?,
        ];

        // The whole leg over the same rows: the SiTU rows, the down rows and
        // their weighted sum.
        let shape = LegShape {
            cap: t,
            slots_per_col: n_used,
            n: embd,
            ff,
        };
        let mut bufs = LegBufs::for_down(stream, shape, DownAct::Q8Blocks32)?;
        let mut acc = DeviceBuffer::zeroed(stream, t * embd)?;
        dev.leg.enqueue(
            &dev.gpu,
            &mut bufs,
            &LegRun {
                layer: l,
                stacks: &stacks,
                normed: &x,
                m: t,
                sel: &ids,
                n_card: n_expert,
                rule: rule_of(hp),
            },
            Some(LegSum {
                w: &weights,
                acc: &mut acc,
            }),
        )?;
        let (situ, down_rows, sum) = (
            dev.read(bufs.act_rows())?,
            dev.read(bufs.down())?,
            dev.read(&acc)?,
        );
        let down_tap = format!("ffn_moe_down{}-{slot}", if scaled { "_scaled" } else { "" });
        let situ_rel = rel64(
            &situ,
            &tap_at(&set.man, &format!("ffn_moe_situ-{slot}"), 0)?,
        );
        let down_rel = rel64(&down_rows, &tap_at(&set.man, &down_tap, 0)?);
        let out_rel = rel64(&sum, &tap_at(&set.man, &format!("ffn_moe_out-{slot}"), 0)?);
        let fault = dev.gpu.fault()?;
        let quant = if scaled { "nvfp4" } else { "q8_0" };
        Ok((
            vec![
                clause(
                    &format!("split_gate_is_vendor l={l} ({quant})"),
                    halves[0] <= HALVES_BAND,
                    &format!("rel {:#.3e} (band {HALVES_BAND:.0e})", halves[0]),
                ),
                clause(
                    &format!("split_up_is_vendor l={l} ({quant})"),
                    halves[1] <= HALVES_BAND,
                    &format!("rel {:#.3e} (band {HALVES_BAND:.0e})", halves[1]),
                ),
                clause(
                    &format!("leg_situ_is_vendor l={l} ({quant})"),
                    situ_rel <= SITU_BAND,
                    &format!("rel {situ_rel:.3e} (band {SITU_BAND:.0e})"),
                ),
                clause(
                    &format!("leg_down_is_vendor l={l} ({quant})"),
                    down_rel <= DOWN_BAND,
                    &format!("rel {down_rel:.3e} (band {DOWN_BAND:.0e})"),
                ),
                clause(
                    &format!("leg_out_is_vendor l={l} ({quant})"),
                    out_rel <= OUT_BAND,
                    &format!("rel {out_rel:.3e} (band {OUT_BAND:.0e})"),
                ),
                clause(
                    &format!("no_fault l={l}"),
                    fault.is_none(),
                    &format!("{fault:?}"),
                ),
            ],
            w,
        ))
    }

    /// A refusal clause: the call refused by its shape or tensor check, before
    /// anything is enqueued.
    fn refused(name: &str, r: Result<(), GpuError>) -> bool {
        clause(
            &format!("refusal {name}"),
            matches!(r, Err(GpuError::Shape { .. } | GpuError::Tensor { .. })),
            &format!("{r:?}"),
        )
    }

    /// The refusal clauses over the last ladder layer's resident weights and
    /// the router's own shape check. The layer's stacks are the leg's own
    /// resident state — its fused stack already split, its Q8_0 stacks already
    /// relayouted into the file's blocks — so the whole-experts refusal runs
    /// on the down's words and the second-split refusal on the resident gate
    /// half (itself a derived stack): a vehicle with no file tensor to reload
    /// and no form to assume.
    fn refusal_clauses(
        dev: &Dev,
        hp: &Hparams,
        (w, l): (&mut Weights, usize),
    ) -> Result<Vec<bool>, GateError> {
        let stream = dev.stream();
        let embd = hp.n_embd;
        let ff = hp.expert_ff[l];
        let (absent, down_name, norm, fused) = (
            format!("blk.{l}.ffn_gate_up_exps"),
            format!("blk.{l}.ffn_down_exps.weight"),
            format!("blk.{l}.ffn_norm.weight"),
            format!("blk.{l}.ffn_gate_up_exps.weight"),
        );
        let split = |w: &mut Weights, name: &str, rows_half| {
            w.split_fused_w13(stream, name, rows_half).map(|_| ())
        };
        let mut ok = vec![
            refused(
                "split of a name that is not resident",
                split(w, &absent, ff),
            ),
            refused("split of a non-stack weight", split(w, &norm, 1)),
            refused(
                "split of rows that are not whole experts",
                split(w, &down_name, ff + 1),
            ),
        ];
        // A second split of the same stack: the first lands on the leg's own
        // gate half (whole experts of half its rows), the derived names it
        // files are the second's refusal.
        let gate = bloomery_gpu::w13_split::gate_of(&fused);
        split(w, &gate, ff / 2)?;
        ok.push(refused(
            "a second split of the same stack",
            split(w, &gate, ff / 2),
        ));
        let wrong = DeviceTensor::upload(
            stream,
            &vec![0.0f32; hp.n_expert / 2 * embd],
            hp.n_expert / 2,
            embd,
        )?;
        let x = dev.up(&vec![0.0f32; embd])?;
        let bias = dev.up(&vec![0.0f32; hp.n_expert])?;
        let mut run = sigmoid_128_8::RouterOut::new(stream)?;
        let routed = dev.router.enqueue_router(
            stream,
            &wrong,
            &x,
            &bias,
            hp.weights_scale,
            &mut run,
            dev.gpu.unlabelled_sink(),
        );
        ok.push(refused("router weight of the wrong rows", routed));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let tier = Tier::from_env()?;
        let (family, sets) = sets_of(tier);
        let path = bloomery_gpu_gates::ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        bloomery_gpu_gates::expect_arch(&split, model::arch::Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{}: {e}", path.display()))?;
        let dev = Dev {
            router: sigmoid_128_8::RouterKernels::load(gpu.context())?,
            wide: Gemm32Kernels::load(gpu.context())?,
            leg: CardLeg::load(&gpu)?,
            nvfp4: Nvfp4SelKernels::load(gpu.context(), gpu.fault_word())?,
            q80: Q80SelKernels::load(gpu.context(), gpu.fault_word())?,
            gpu,
        };
        println!(
            "{NAME}: device {} — tier {tier:?}, {} stored layers of {} visits, hidden {}, experts \
             {} wide {} (top {}), scale {}, situ {}/{}",
            dev.gpu.device_name()?,
            hp.n_layer,
            hp.n_blocks,
            hp.n_embd,
            hp.n_expert,
            hp.expert_ff[0],
            hp.n_used[0],
            hp.weights_scale,
            hp.situ.gate_beta,
            hp.situ.value_beta,
        );
        let routers = Routers::of(&split, &dev, &hp)?;
        let mut results = Vec::new();
        let mut set_a = None;
        for (i, &name) in sets.iter().enumerate() {
            let set = open_set(tier, family, name)?;
            let label = name
                .trim_start_matches("fx_ref_")
                .trim_start_matches("ref_");
            let with_scores = i == 0;
            results.extend(router_clauses(
                &dev,
                (&hp, &routers),
                &set,
                label,
                with_scores,
            )?);
            if i == 0 {
                set_a = Some(set);
            }
        }
        let set_a = set_a.ok_or("the tier's first set did not open")?;
        let ladder = ladder_layers(&split, &hp)?;
        let mut weights = None;
        for &l in &ladder {
            results.extend(one_token_clauses(&dev, (&hp, &routers, &set_a), l)?);
            let (clauses, w) = leg_clauses(&dev, (&hp, &routers, &set_a), &split, l)?;
            results.extend(clauses);
            weights = Some((w, l));
        }
        let (mut w, l) = weights.ok_or("no ladder layer loaded")?;
        results.extend(refusal_clauses(&dev, &hp, (&mut w, l))?);
        results.push(guard_clause(&dev, &hp)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "{NAME}: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
