//! GPU gate for Qwen3.6-35B-A3B's MoE block on the card: the gated router
//! (`arch::qwen3moe::router::gated`: softmax over 256 experts, the top 8
//! renormalized, and the shared expert's sigmoid gate as a ninth slot, its
//! logit the router's row 256), the joined stacks that carry the shared
//! expert as expert 256, the Q4_K and Q6_K down `_sel` at K = 512, and the
//! nine-slot FFN made of the qwen3moe kernels. Layers 0 (Q6_K down) and 5
//! (Q4_K down) of the Q4_K_M file, read-only.
//!
//! Cases (`--case <substring>[,<substring>…]` runs the ones whose name holds
//! one of them):
//! - `routing`: the routing alone (`qwen35moe_router_route` at one token)
//!   on every token of every layer of every set, its 257 logits
//!   `ffn_moe_logits-L` and `shared_expert_gate-L` in, against the host rule
//!   (`route_ref_within` at 256/8, the weights renormalized over the eight in
//!   f64, slot 8 `(256, route_core::sigmoid(logit_256))`): ids EXACT, the
//!   probabilities and the eight weights within [`BAND`], the ninth weight
//!   within [`BAND`] of the host sigmoid, a rerun bit-identical — three
//!   verdict lines, ids, weights and gate. Then constructed ties: all 256
//!   logits equal; an eighth place shared by experts 224, 32 (one lane) and 1
//!   (another); a first place shared by 255 and 31.
//! - `logits`: the fused launch at m = 1, 5, 8 against `f32_gemv` over the
//!   joined weight (all 257 rows) followed by the routing alone per token:
//!   logits, probabilities, ids and weights BIT-EQUAL, the ticket count zero;
//!   the ubatch pair (`qwen35moe_router_logits` + route) over 70 tokens (three
//!   token blocks, nine row blocks) BIT-EQUAL to the fused launch per token;
//!   the fused launch at m = 8 as a captured graph, one node, two replays
//!   bit-identical, the count zero after each.
//! - `norm`: the norm-fused launch at one token against `norm_quant` then the
//!   fused launch, on each token of `ssm_output-L` through layer L's
//!   `post_attention_norm`: the q8_1 planes and the router's outputs
//!   BIT-EQUAL, the count zero; as a captured graph, one node, the replay
//!   bit-identical.
//! - `fault`: each launch labelled with layer 5. The routing alone on a NaN
//!   expert logit and on a NaN and a +inf gate logit; the fused launch at
//!   five tokens with a NaN in router row 17 and in row 256, and the ubatch
//!   pair at nine tokens with a NaN in row 256 — every token refused; the
//!   ubatch pair with token 8's column overflowing row 256's logit to +inf —
//!   that token refused, the rest the clean run's; the norm-fused launch with
//!   a NaN in row 256 — its q8_1 bytes the clean run's, its routing refused.
//!   A refused token: `FaultSite::Router` with that layer, its probabilities
//!   and all nine weights NaN, its ids as they stood.
//! - `join`: the joined stacks (`Weights::join_rows` over
//!   `[ffn_*_exps, ffn_*_shexp]`) byte-equal to the file's stacks and, as
//!   expert 256, to its shexp tensors; the joined router
//!   (`[ffn_gate_inp, ffn_gate_inp_shexp]`, an F32 join) byte-equal to the
//!   file's rows, row 256 to `ffn_gate_inp_shexp`.
//! - `down512`: `q4k_gemv_sel` (layer 5) and `q6k_gemv_sel` (layer 0) over
//!   the joined down stacks at K = 512, thirteen slots with a repeated id and
//!   expert 256 twice, BIT-EQUAL to the host row dot of each row: the
//!   kernel's lane terms from the logical q8_1 codes read back from the
//!   activation planes, the product and fused-multiply-add order of its PTX,
//!   then the warp butterfly.
//! - `ffn`: the nine-slot FFN at m = 1, 5, 8 (router with the norm, gate·up
//!   over the joined stacks at 9m slots, one quantizer, the down `_sel`, the
//!   combine at nine slots) BIT-EQUAL to the unfused composition — the
//!   routed eight on the joined stacks, the shared expert on the file's
//!   shexp tensors as a one-expert stack, each quantized and projected
//!   apart, the downs put in slot order and combined — and to the host
//!   combine `((Σ_{s<8} w_s·d_s) + w_8·d_sh) + resid`, each term a fused
//!   multiply-add as the combine compiles. At m = 5 the distance to ik's
//!   `l_out-L` is printed.
//! - `ubatch`: 40 tokens through the ubatch path (the GEMM quantizer, the
//!   ubatch router, `gemm_route` over 257 experts, `gemm_q4k` gate and up
//!   with `Shared { top_k: 9 }`, `gemm_swiglu_quant`, the down GEMM at K =
//!   512, the combine) against the one-token path per token: the routing
//!   identical, and the outputs' relative distance within [`SPREAD_RATIO`]
//!   of the one-token path's own distance to the f64 reference.
//! - `graph`: the nine-slot FFN at one token as a captured graph: five nodes,
//!   two replays bit-identical to the eager run.
//! - `ik`: our fused router on ik's `ffn_inp_normed-L` of every layer of
//!   every set against ik: the logits and the gate logit within the two
//!   sums' rounding bound, the top 8 EQUAL `ffn_moe_topk-L` (a mismatch
//!   prints ik's eighth and ninth probabilities), the eight weights within
//!   twice the token's largest logit difference plus [`BAND`] of
//!   `ffn_moe_weights_norm-L`; the sigmoid's distance to
//!   `shared_expert_gate_sigmoid-L` printed.
//! - `wide512`: the second instance (512 experts, the top 10, the shared
//!   gate as row 512 and slot 10), synthetic: its widths against the gate's
//!   own 512 and 10; a router weight of 0, ±1/8 and activations of 0, ±1 at
//!   K = 2560, so every logit is exact in any sum order. The fused launch at
//!   m = 1, 5, 8, the routing alone over 70 tokens (rerun bit-identical) and
//!   the ubatch pair over 70 (three token blocks, all 17 row blocks) against
//!   the host: logits BIT-EQUAL to the host dot, the eleven ids EXACT to the
//!   host rule and to the selection over the kernel's own probabilities, the
//!   ten routed ids distinct and below 512, the ten weights BIT-EQUAL to the
//!   weight stage over the kernel's probabilities, the probabilities and the
//!   gate weight within [`BAND`]; the ubatch pair BIT-EQUAL to the fused
//!   launch per token; the fused ticket count zero. Constructed ties (all
//!   equal — bit-identical to the host outright; a tenth place shared across
//!   two lanes; a first place shared inside one lane; a first place shared
//!   in the upper half). Refusals at layer 5: the routing alone on a NaN,
//!   a +inf and a −inf expert logit in the upper half and a NaN gate logit,
//!   the fused launch with a NaN in row 400, the ubatch pair with a NaN in
//!   row 512 — every token refused, eleven NaN weights, ids kept. Then the
//!   sum orders on inputs that round (LCG weights and activations, K =
//!   2560: the one-column walk's 32-chunk trip with an 8-chunk tail): the
//!   fused launch at m = 1, 5, 8 against `f32_gemv` over 513 rows then the
//!   routing alone, and the ubatch pair over 70 against the fused launch
//!   per token, all BIT-EQUAL.
//! - `shape`: the seven entries compile with no local depot.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen35moe_moe: built without the `gpu` feature; see `just gate-gpu-qwen35moe-moe`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_moe", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::experts::{CombineArgs, ExpertKernels, GateUpArgs};
    use bloomery_gpu::arch::qwen3moe::router::gated::{ROW, ROW_WIDE, RouterKernels, RouterOut};
    use bloomery_gpu::arch::qwen3moe::router::{MAX_TOKENS, RouterDims};
    use bloomery_gpu::fused::{FusedKernels, Q8ActHost, readback_q8act};
    use bloomery_gpu::gemm::{GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmWeight};
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::weights::{DevWeight, Weights, upload_file_tensor};
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, Q8Act};
    use bloomery_gpu_gates::act_rule::q4k_scales;
    use bloomery_gpu_gates::rounding::{butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, bytes_to_words, checks_failed, data_dir,
        no_local_depot, ref_model_path, ref_tensor_logical_in, route_ref_within,
        topk_ids_logical_within, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use model::arch::models::shape::{MoeShape, rules};
    use refset::arch::qwen35moe as oracle;

    /// The narrow instance's experts, its row's; the routed slots the
    /// clauses run at, Qwen3.6's `expert_used_count` (`run` holds the file
    /// to both); the slots and logits a token (the shared expert's last),
    /// and the shared expert's id and gate row.
    const N_EXPERT: usize = ROW.experts() as usize;
    const N_USED: usize = 8;
    const N_SLOTS: usize = N_USED + 1;
    const ROWS: usize = N_EXPERT + 1;
    const SHARED: usize = N_EXPERT;

    /// The router's dims at `experts` and `top_k` routed slots.
    fn dims(experts: usize, top_k: usize) -> Result<RouterDims, GateError> {
        Ok(RouterDims::of(MoeShape {
            rule: rules::SOFTMAX_NORM_GATED,
            experts: u32::try_from(experts)?,
            top_k: u32::try_from(top_k)?,
        })?)
    }

    /// The narrow instance's dims at [`N_USED`].
    fn narrow() -> Result<RouterDims, GateError> {
        dims(N_EXPERT, N_USED)
    }

    /// The wide instance's dims at [`WIDE_USED`].
    fn wide() -> Result<RouterDims, GateError> {
        dims(WIDE_EXPERT, WIDE_USED)
    }

    /// Probabilities and weights against the host rule, absolute: all lie
    /// in [0, 1] and the only divergence is `exp`'s last ulps (and the
    /// f64 sums' order, which rounds to the same f32 unless the two straddle
    /// a rounding boundary).
    const BAND: f32 = 1e-6;

    /// The ubatch path's distance to the one-token path, over the one-token
    /// path's own distance to the f64 reference. Both are f32 roundings of
    /// one rule — the same q8_1 input, the same routing; their GEMM and gemv
    /// sum orders differ, and a SwiGLU value that lands on the other side of
    /// a q8_1 rounding boundary moves one code — so each sits about as far
    /// from the reference as the other and their distance is at most the sum
    /// of the two (about 2, or √2 when independent). A wiring fault (a slot
    /// on the wrong expert, a weight one slot off) moves a token's output by
    /// its own order, about 1/ (the reference distance, ~1e-2) ≈ 100 times
    /// the ruler. The value is the qwen3moe e2e gate's `GEMM_SPREAD_RATIO`.
    const SPREAD_RATIO: f64 = 3.0;

    /// The layers the real-weight cases run: layer 0's down is Q6_K, layer
    /// 5's Q4_K.
    const LAYERS: [usize; 2] = [0, 5];

    /// The layer the refusal cases label their launches with.
    const FAULT_LAYER: usize = 5;

    /// Values of the hidden state.
    const HIDDEN: usize = 2048;

    /// An expert's (and the shared expert's) FFN width, the down's K.
    const FF: usize = 512;

    /// What an output buffer holds before a launch.
    const SENT: f32 = 1.0e30;

    /// A finite activation value whose products stay finite while a row's
    /// dot does not (`Σ|w| · BIG` passes `f32::MAX`).
    const BIG: f32 = 1.0e38;

    /// Expert slots of the combine the fused FFN runs.
    const FUSED_SLOTS: usize = N_SLOTS;

    /// The ubatch case's token count: five tokens of each of eight layers'
    /// `ffn_inp_normed`.
    const UB_TOKENS: usize = 40;

    /// The logits case's ubatch: three 32-token blocks.
    const LOGITS_UB: usize = 70;

    /// The wide instance's experts and routed slots, Qwen3.8's
    /// `expert_count` and `expert_used_count`: the gate's own values, apart
    /// from the instance row, so a change to that is a red line rather than
    /// a rewrite of both sides.
    const WIDE_EXPERT: usize = 512;
    const WIDE_USED: usize = 10;
    /// The wide instance's slots per token (the shared expert last) and
    /// logits per token (its gate row last).
    const WIDE_SLOTS: usize = WIDE_USED + 1;
    const WIDE_ROWS: usize = WIDE_EXPERT + 1;
    /// The wide case's router width, Qwen3.8's hidden width.
    const WIDE_K: usize = 2560;
    /// The wide case's tokens: three 32-token blocks of the ubatch pair.
    const WIDE_UB: usize = 70;

    /// The routed slot counts the picks case runs at each instance besides
    /// the models' own: one, an odd count, and the lane picks' bound.
    const PICKS: [usize; 3] = [1, 3, 32];
    /// The picks case's tokens: two routing blocks and a part of a third.
    const PICK_TOKENS: usize = 19;

    // ------------------------------------------------------------ context

    struct Ctx {
        gpu: Gpu,
        split: Split,
        router: RouterKernels,
        experts: ExpertKernels,
        q6: Q6kSelKernels,
        gemm: GemmKernels,
        norm: FusedKernels,
        eps: f32,
        sets: Vec<(&'static str, RefManifest)>,
        filter: Option<String>,
    }

    impl Ctx {
        /// Whether `case` runs: no filter, or a comma-separated part of it
        /// that the case's name holds.
        fn want(&self, case: &str) -> bool {
            self.filter
                .as_deref()
                .is_none_or(|f| f.split(',').any(|p| case.contains(p)))
        }

        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        /// A tensor's file bytes.
        fn bytes(&self, name: &str) -> Result<(GgmlType, Vec<u64>, &[u8]), GateError> {
            let (s, t) = self
                .split
                .find(name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            let g = self.split.shard(s).ok_or("shard index out of range")?;
            Ok((t.ty, t.dims.clone(), g.data(t)?))
        }

        /// An F32 tensor as f32.
        fn f32s(&self, name: &str) -> Result<Vec<f32>, GateError> {
            let (ty, _, b) = self.bytes(name)?;
            if ty != GgmlType::F32 {
                return Err(format!("{name} is {ty:?}, want F32").into());
            }
            Ok(b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        }

        /// Layer `l`'s joined router weight on the host: `ffn_gate_inp`'s
        /// 256 rows, then `ffn_gate_inp_shexp` as row 256.
        fn router_host(&self, l: usize) -> Result<Vec<f32>, GateError> {
            let mut w = self.f32s(&format!("blk.{l}.ffn_gate_inp.weight"))?;
            let sh = self.f32s(&format!("blk.{l}.ffn_gate_inp_shexp.weight"))?;
            if w.len() != N_EXPERT * HIDDEN || sh.len() != HIDDEN {
                return Err(format!(
                    "layer {l}: router {} values, shared gate {}; want {N_EXPERT} x {HIDDEN} and \
                     {HIDDEN}",
                    w.len(),
                    sh.len()
                )
                .into());
            }
            w.extend_from_slice(&sh);
            Ok(w)
        }

        /// Tap `name` of set `set` as logical f32.
        fn tap(&self, set: usize, name: &str) -> Result<Vec<f32>, GateError> {
            let man = &self.sets[set].1;
            Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
        }
    }

    /// The names of layer `l`'s stacks, their shexp tensors and joints.
    fn stack_names(l: usize, part: &str) -> (String, String, String) {
        (
            format!("blk.{l}.ffn_{part}_exps.weight"),
            format!("blk.{l}.ffn_{part}_shexp.weight"),
            format!("derived.blk.{l}.ffn_{part}_exps_sh"),
        )
    }

    /// One real layer on the card.
    struct Layer {
        l: usize,
        down_ty: GgmlType,
        w: Weights,
        router: DeviceTensor<f32>,
        router_host: Vec<f32>,
        gain: DeviceBuffer<f32>,
        /// The file's shexp gate, up and down as one-expert stacks.
        sh: [DeviceTensor<u32>; 3],
    }

    impl Layer {
        fn joint(&self, part: &str) -> Result<&DeviceTensor<u32>, GateError> {
            let name = stack_names(self.l, part).2;
            match self.w.get(&name) {
                Some(DevWeight::KQuant { w, .. }) => Ok(w),
                _ => Err(format!("{name} is not resident as a K-quant").into()),
            }
        }
    }

    fn kq(dw: DevWeight, name: &str) -> Result<DeviceTensor<u32>, GateError> {
        match dw {
            DevWeight::KQuant { w, .. } => Ok(w),
            _ => Err(format!("{name} is not a K-quant").into()),
        }
    }

    fn load_layer(c: &Ctx, l: usize) -> Result<Layer, GateError> {
        let stream = c.stream();
        let mut keep: Vec<String> = Vec::new();
        for part in ["gate", "up", "down"] {
            let (e, s, _) = stack_names(l, part);
            keep.push(e);
            keep.push(s);
        }
        keep.push(format!("blk.{l}.ffn_gate_inp.weight"));
        keep.push(format!("blk.{l}.ffn_gate_inp_shexp.weight"));
        let mut w = Weights::load_where(stream, &c.split, |n| keep.iter().any(|k| k == n))?;
        let mut sh = Vec::new();
        for part in ["gate", "up", "down"] {
            let (e, s, j) = stack_names(l, part);
            let (shard, t) = c.split.find(&s).ok_or("shexp tensor missing")?;
            let g = c.split.shard(shard).ok_or("shard index out of range")?;
            sh.push(kq(upload_file_tensor(stream, g, t)?, &s)?);
            w.join_rows(stream, &[e.as_str(), s.as_str()], j)?;
        }
        let down_ty = c.bytes(&stack_names(l, "down").0)?.0;
        let router_host = c.router_host(l)?;
        let router = DeviceTensor::upload(stream, &router_host, ROWS, HIDDEN)?;
        let gain = DeviceBuffer::from_host(
            stream,
            &c.f32s(&format!("blk.{l}.post_attention_norm.weight"))?,
        )?;
        let [g, u, d]: [DeviceTensor<u32>; 3] =
            sh.try_into().map_err(|_| "three shexp tensors expected")?;
        Ok(Layer {
            l,
            down_ty,
            w,
            router,
            router_host,
            gain,
            sh: [g, u, d],
        })
    }

    // ------------------------------------------------------------ routing

    /// One launch's results for its first `m` tokens.
    #[derive(Clone)]
    struct Routed {
        logits: Vec<f32>,
        probs: Vec<f32>,
        ids: Vec<u32>,
        weights: Vec<f32>,
    }

    fn read_out(stream: &CudaStream, out: &RouterOut, m: usize) -> Result<Routed, GateError> {
        let mut logits = out.logits.to_host_vec(stream)?;
        let mut probs = out.probs.to_host_vec(stream)?;
        let mut ids = out.ids.to_host_vec(stream)?;
        let mut weights = out.weights.to_host_vec(stream)?;
        logits.truncate(m * ROWS);
        probs.truncate(m * N_EXPERT);
        ids.truncate(m * N_SLOTS);
        weights.truncate(m * N_SLOTS);
        Ok(Routed {
            logits,
            probs,
            ids,
            weights,
        })
    }

    fn routed_equal(a: &Routed, b: &Routed) -> bool {
        bits_equal(&a.logits, &b.logits)
            && bits_equal(&a.probs, &b.probs)
            && a.ids == b.ids
            && bits_equal(&a.weights, &b.weights)
    }

    /// Token `t` of two runs bit for bit.
    fn token_equal(a: &Routed, b: &Routed, t: usize) -> bool {
        let (r, e, s) = (
            t * ROWS..(t + 1) * ROWS,
            t * N_EXPERT..(t + 1) * N_EXPERT,
            t * N_SLOTS..(t + 1) * N_SLOTS,
        );
        bits_equal(&a.logits[r.clone()], &b.logits[r])
            && bits_equal(&a.probs[e.clone()], &b.probs[e])
            && a.ids[s.clone()] == b.ids[s.clone()]
            && bits_equal(&a.weights[s.clone()], &b.weights[s])
    }

    /// Whether token `t` of `got` is refused: NaN probabilities and nine NaN
    /// weights, and the ids `kept` held for it.
    fn token_refused(got: &Routed, t: usize, kept: &Routed) -> bool {
        let (e, s) = (
            t * N_EXPERT..(t + 1) * N_EXPERT,
            t * N_SLOTS..(t + 1) * N_SLOTS,
        );
        got.probs[e].iter().all(|v| v.is_nan())
            && got.weights[s.clone()].iter().all(|v| v.is_nan())
            && got.ids[s.clone()] == kept.ids[s]
    }

    /// The routing alone of one token's `ROWS` logits.
    fn route_one(
        c: &Ctx,
        logits: &[f32],
        sink: FaultSink,
        out: &mut RouterOut,
    ) -> Result<Routed, GateError> {
        let stream = c.stream();
        let x = DeviceBuffer::from_host(stream, logits)?;
        out.logits.copy_from_device_async(&x, stream)?;
        c.router.enqueue_route(stream, 1, sink, out)?;
        stream.synchronize()?;
        read_out(stream, out, 1)
    }

    /// The host rule for one token's `ROWS` logits: probabilities, the nine
    /// ids and the nine weights.
    fn host_route(logits: &[f32]) -> Result<Routed, GateError> {
        host_route_at(logits, N_EXPERT, N_USED)
    }

    /// The gated host rule at `n_expert` experts and `n_used` routed slots,
    /// for one token's `n_expert + 1` logits (the last the shared gate's):
    /// the probabilities, then `n_used + 1` ids and weights — the top
    /// `n_used` renormalized in f64, slot `n_used` `(n_expert,
    /// sigmoid(gate))`.
    fn host_route_at(logits: &[f32], n_expert: usize, n_used: usize) -> Result<Routed, GateError> {
        if logits.len() != n_expert + 1 {
            return Err(
                format!("host_route: {} logits, want {}", logits.len(), n_expert + 1).into(),
            );
        }
        let (probs, ids, w) = route_ref_within(&logits[..n_expert], 1, n_expert, n_used, 1.0)?;
        let g = logits[n_expert];
        if !g.is_finite() {
            return Err(format!("host_route: gate logit {g}").into());
        }
        let sum = w.iter().fold(0.0f64, |a, &v| a + f64::from(v)) as f32;
        let mut weights: Vec<f32> = w.iter().map(|&v| v / sum).collect();
        weights.push(sigmoid(g));
        let mut ids: Vec<u32> = ids.into_iter().map(|i| i as u32).collect();
        ids.push(n_expert as u32);
        Ok(Routed {
            logits: logits.to_vec(),
            probs,
            ids,
            weights,
        })
    }

    /// The kernels' selection and weight stage over one token's
    /// probabilities `p`, op for op: the top `n_used` by descending
    /// probability, ties toward the smaller id; their sum in f64 in slot
    /// order, rounded once to f32; each weight the f32 divide by it.
    fn host_select(p: &[f32], n_used: usize) -> (Vec<u32>, Vec<f32>) {
        let mut ranked: Vec<u32> = (0..p.len() as u32).collect();
        ranked.sort_by(|&a, &b| p[b as usize].total_cmp(&p[a as usize]).then(a.cmp(&b)));
        ranked.truncate(n_used);
        let sum = ranked
            .iter()
            .fold(0.0f64, |a, &e| a + f64::from(p[e as usize])) as f32;
        let w = ranked.iter().map(|&e| p[e as usize] / sum).collect();
        (ranked, w)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Set `s`, layer `l`'s 257 logits per token from ik: `ffn_moe_logits`
    /// and `shared_expert_gate`, and the token count.
    fn ik_logits(c: &Ctx, s: usize, l: usize) -> Result<(Vec<f32>, usize), GateError> {
        let lg = c.tap(s, &format!("ffn_moe_logits-{l}"))?;
        let gt = c.tap(s, &format!("shared_expert_gate-{l}"))?;
        let m = gt.len();
        if lg.len() != m * N_EXPERT {
            return Err(format!("layer {l}: {} logits for {m} gate logits", lg.len()).into());
        }
        let mut out = Vec::with_capacity(m * ROWS);
        for t in 0..m {
            out.extend_from_slice(&lg[t * N_EXPERT..(t + 1) * N_EXPERT]);
            out.push(gt[t]);
        }
        Ok((out, m))
    }

    fn routing_case(c: &Ctx, n_layer: usize) -> Result<bool, GateError> {
        let sink = c.gpu.unlabelled_sink();
        let mut out = RouterOut::with_tokens(c.stream(), narrow()?, 1)?;
        let (mut tokens, mut ids_ok, mut w_ok, mut gate_ok, mut rerun_ok) =
            (0, true, true, true, true);
        let (mut worst_p, mut worst_w, mut worst_g) = (0.0f32, 0.0f32, 0.0f32);
        for s in 0..c.sets.len() {
            for l in 0..n_layer {
                let (lg, m) = ik_logits(c, s, l)?;
                for t in 0..m {
                    let lt = &lg[t * ROWS..(t + 1) * ROWS];
                    let a = route_one(c, lt, sink, &mut out)?;
                    let b = route_one(c, lt, sink, &mut out)?;
                    let h = host_route(lt)?;
                    let pe = max_abs(&a.probs, &h.probs);
                    let we = max_abs(&a.weights[..N_USED], &h.weights[..N_USED]);
                    let ge = (a.weights[N_USED] - h.weights[N_USED]).abs();
                    let (i, w, g, r) = (
                        a.ids == h.ids,
                        pe <= BAND && we <= BAND,
                        ge <= BAND,
                        routed_equal(&a, &b),
                    );
                    if !(i && w && g && r) {
                        println!(
                            "routing set={} layer={l} t={t} ids={:?} host={:?} probs_err={pe:.3e} \
                             weights_err={we:.3e} gate_err={ge:.3e} rerun={r} FAIL",
                            c.sets[s].0, a.ids, h.ids
                        );
                    }
                    ids_ok &= i;
                    w_ok &= w;
                    gate_ok &= g;
                    rerun_ok &= r;
                    worst_p = worst_p.max(pe);
                    worst_w = worst_w.max(we);
                    worst_g = worst_g.max(ge);
                    tokens += 1;
                }
            }
        }
        println!(
            "routing ids: {tokens} tokens, the nine ids EXACT to the host rule (slot 8 = {SHARED}) \
             rerun_bits={rerun_ok} {}",
            verdict(ids_ok && rerun_ok)
        );
        println!(
            "routing weights: probs_err={worst_p:.3e} weights_err={worst_w:.3e} (slots 0..8, band \
             {BAND:.0e}) {}",
            verdict(w_ok)
        );
        println!(
            "routing gate: w[8] vs sigmoid(logit_256) err={worst_g:.3e} (band {BAND:.0e}) {}",
            verdict(gate_ok)
        );
        let mut ok = ids_ok && w_ok && gate_ok && rerun_ok;

        // Constructed ties.
        let mut cases: Vec<(&str, Vec<f32>, Vec<u32>)> = Vec::new();
        let mut l = vec![0.5f32; ROWS];
        l[SHARED] = 0.25;
        cases.push(("all-equal", l, vec![0, 1, 2, 3, 4, 5, 6, 7, 256]));
        let mut l = vec![-4.0f32; ROWS];
        for (i, &e) in [200usize, 90, 64, 33, 5, 255, 70].iter().enumerate() {
            l[e] = 3.0 - 0.25 * i as f32;
        }
        for e in [224usize, 32, 1] {
            l[e] = 1.0;
        }
        l[SHARED] = -1.5;
        cases.push((
            "eighth-shared-by-224-32-1",
            l,
            vec![200, 90, 64, 33, 5, 255, 70, 1, 256],
        ));
        let mut l = vec![-4.0f32; ROWS];
        for (i, e) in (40..47).enumerate() {
            l[e] = 1.0 - 0.1 * i as f32;
        }
        l[255] = 2.0;
        l[31] = 2.0;
        l[SHARED] = 3.0;
        cases.push((
            "first-shared-by-255-31",
            l,
            vec![31, 255, 40, 41, 42, 43, 44, 45, 256],
        ));
        for (name, logits, want) in &cases {
            let a = route_one(c, logits, sink, &mut out)?;
            let h = host_route(logits)?;
            let pass = a.ids == *want
                && h.ids == *want
                && max_abs(&a.weights[..N_USED], &h.weights[..N_USED]) <= BAND;
            println!(
                "tie case={name} ids={:?} host={:?} want={want:?} {}",
                a.ids,
                h.ids,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ wide512

    /// One step of a 64-bit LCG: its high 31 bits.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as u32
    }

    /// The wide case's joined router weight, [`WIDE_ROWS`] rows of
    /// [`WIDE_K`]: each value 1/8, −1/8 or 0 (half of them). With
    /// [`wide_cols`]' values every product is 0 or ±1/8 and every partial
    /// sum of a row a multiple of 1/8 at most `WIDE_K / 8` in size — at most
    /// `WIDE_K` < 2^24 eighths — so each logit is exact in f32 in any sum
    /// order, fused multiply-adds included: the kernels' logits equal the
    /// host's integer dot bit for bit, and two distinct logits differ by at
    /// least 1/8, far past the ulps where `exp` on the card and on the host
    /// part.
    fn wide_weight() -> Vec<f32> {
        let mut s = 0x5eed_0512_u64;
        (0..WIDE_ROWS * WIDE_K)
            .map(|_| match lcg(&mut s) % 4 {
                0 => 0.125,
                1 => -0.125,
                _ => 0.0,
            })
            .collect()
    }

    /// `m` synthetic activation columns of [`WIDE_K`]: each value 1, −1 or 0.
    fn wide_cols(m: usize) -> Vec<f32> {
        let mut s = 0xc01_0512_u64;
        (0..m * WIDE_K)
            .map(|_| match lcg(&mut s) % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect()
    }

    /// Token `t`'s [`WIDE_ROWS`] logits on the host: each row's dot with
    /// column `t`, exact (the sum of ±1 counts, over 8).
    fn wide_logits(w: &[f32], x: &[f32], t: usize) -> Vec<f32> {
        let col = &x[t * WIDE_K..(t + 1) * WIDE_K];
        (0..WIDE_ROWS)
            .map(|r| {
                let row = &w[r * WIDE_K..(r + 1) * WIDE_K];
                let eighths: i64 = row
                    .iter()
                    .zip(col)
                    .map(|(&a, &b)| (a * 8.0) as i64 * b as i64)
                    .sum();
                eighths as f32 / 8.0
            })
            .collect()
    }

    /// A wide run's buffers, read whole.
    fn read_wide(stream: &CudaStream, out: &RouterOut) -> Result<Routed, GateError> {
        Ok(Routed {
            logits: out.logits.to_host_vec(stream)?,
            probs: out.probs.to_host_vec(stream)?,
            ids: out.ids.to_host_vec(stream)?,
            weights: out.weights.to_host_vec(stream)?,
        })
    }

    /// Whether a wide run's buffers hold `tokens` tokens at the gate's own
    /// widths ([`WIDE_ROWS`], [`WIDE_EXPERT`], [`WIDE_SLOTS`] a token):
    /// every token view below indexes by those.
    fn wide_sized(r: &Routed, tokens: usize) -> bool {
        r.logits.len() >= tokens * WIDE_ROWS
            && r.probs.len() >= tokens * WIDE_EXPERT
            && r.ids.len() >= tokens * WIDE_SLOTS
            && r.weights.len() >= tokens * WIDE_SLOTS
    }

    /// Token `t` of a wide run as one token's [`Routed`].
    fn wide_token(r: &Routed, t: usize) -> Routed {
        Routed {
            logits: r.logits[t * WIDE_ROWS..(t + 1) * WIDE_ROWS].to_vec(),
            probs: r.probs[t * WIDE_EXPERT..(t + 1) * WIDE_EXPERT].to_vec(),
            ids: r.ids[t * WIDE_SLOTS..(t + 1) * WIDE_SLOTS].to_vec(),
            weights: r.weights[t * WIDE_SLOTS..(t + 1) * WIDE_SLOTS].to_vec(),
        }
    }

    /// One wide token against the host rule, its fields the clause's terms.
    #[derive(Default)]
    struct WideCheck {
        tokens: usize,
        logits_bits: bool,
        ids: bool,
        distinct: bool,
        weights_bits: bool,
        probs_err: f32,
        gate_err: f32,
    }

    impl WideCheck {
        fn new() -> WideCheck {
            WideCheck {
                logits_bits: true,
                ids: true,
                distinct: true,
                weights_bits: true,
                ..WideCheck::default()
            }
        }

        /// Add token `got` (one token's [`Routed`]) against `want`, its
        /// [`WIDE_ROWS`] logits: the logits bit for bit; the ids EXACT to the
        /// host rule over the host's probabilities and to the selection over
        /// the kernel's own; the ten routed ids distinct and below
        /// [`WIDE_EXPERT`]; the ten weights bit for bit the weight stage over
        /// the kernel's probabilities; the probabilities within [`BAND`] of
        /// the host's, the gate weight within [`BAND`] of the host sigmoid.
        fn add(&mut self, got: &Routed, want: &[f32]) -> Result<bool, GateError> {
            let h = host_route_at(want, WIDE_EXPERT, WIDE_USED)?;
            let (sel_ids, sel_w) = host_select(&got.probs, WIDE_USED);
            let routed = &got.ids[..WIDE_USED];
            let lb = bits_equal(&got.logits, want);
            let ids = got.ids == h.ids && routed == &sel_ids[..];
            let mut seen = routed.to_vec();
            seen.sort_unstable();
            seen.dedup();
            let distinct =
                seen.len() == WIDE_USED && routed.iter().all(|&e| (e as usize) < WIDE_EXPERT);
            let wb = bits_equal(&got.weights[..WIDE_USED], &sel_w);
            let pe = max_abs(&got.probs, &h.probs);
            let ge = (got.weights[WIDE_USED] - h.weights[WIDE_USED]).abs();
            self.tokens += 1;
            self.logits_bits &= lb;
            self.ids &= ids;
            self.distinct &= distinct;
            self.weights_bits &= wb;
            self.probs_err = self.probs_err.max(pe);
            self.gate_err = self.gate_err.max(ge);
            Ok(lb && ids && distinct && wb && pe <= BAND && ge <= BAND)
        }

        fn pass(&self) -> bool {
            self.logits_bits
                && self.ids
                && self.distinct
                && self.weights_bits
                && self.probs_err <= BAND
                && self.gate_err <= BAND
        }

        fn line(&self, what: &str) -> bool {
            let pass = self.pass();
            println!(
                "wide512 {what}: {} tokens logits_bits={} ids_exact={} distinct_below_512={} \
                 weights_bits={} probs_err={:.3e} gate_err={:.3e} (band {BAND:.0e}) {}",
                self.tokens,
                self.logits_bits,
                self.ids,
                self.distinct,
                self.weights_bits,
                self.probs_err,
                self.gate_err,
                verdict(pass)
            );
            pass
        }
    }

    /// Whether token `t` of a wide run is refused: NaN probabilities and
    /// eleven NaN weights, and the ids `kept` held for it.
    fn wide_refused(got: &Routed, t: usize, kept: &Routed) -> bool {
        let (e, s) = (
            t * WIDE_EXPERT..(t + 1) * WIDE_EXPERT,
            t * WIDE_SLOTS..(t + 1) * WIDE_SLOTS,
        );
        got.probs[e].iter().all(|v| v.is_nan())
            && got.weights[s.clone()].iter().all(|v| v.is_nan())
            && got.ids[s.clone()] == kept.ids[s]
    }

    /// The routing alone of `n` wide tokens' logits (`n ·` [`WIDE_ROWS`]).
    fn wide_route(
        c: &Ctx,
        logits: &[f32],
        n: usize,
        sink: FaultSink,
        out: &mut RouterOut,
    ) -> Result<Routed, GateError> {
        let stream = c.stream();
        let x = DeviceBuffer::from_host(stream, logits)?;
        out.logits.copy_from_device_async(&x, stream)?;
        c.router.enqueue_route(stream, n, sink, out)?;
        stream.synchronize()?;
        read_wide(stream, out)
    }

    /// The routed slot count is a launch argument: at each instance and each
    /// of [`PICKS`], the routing alone of [`PICK_TOKENS`] tokens of synthetic
    /// logits against the host rule at that count — the ids exact, the
    /// probabilities, the routed weights and the gate weight within [`BAND`]
    /// — and the wide fused launch at three slots over five tokens of the
    /// wide case's exact logits: its logits the host's bits, its routing the
    /// routing alone's over those logits, bit for bit.
    fn picks_case(c: &Ctx) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let mut ok = true;
        let mut s = 0x9_1c45_u64;
        for experts in [N_EXPERT, WIDE_EXPERT] {
            let rows = experts + 1;
            let logits: Vec<f32> = (0..PICK_TOKENS * rows)
                .map(|_| ((lcg(&mut s) % 2001) as f32 - 1000.0) * 4.0e-3)
                .collect();
            let x = DeviceBuffer::from_host(stream, &logits)?;
            for top_k in PICKS {
                let mut out = RouterOut::for_ubatch(stream, dims(experts, top_k)?, PICK_TOKENS)?;
                out.logits.copy_from_device_async(&x, stream)?;
                c.router
                    .enqueue_route(stream, PICK_TOKENS, sink, &mut out)?;
                stream.synchronize()?;
                let (probs, ids, weights) = (
                    out.probs.to_host_vec(stream)?,
                    out.ids.to_host_vec(stream)?,
                    out.weights.to_host_vec(stream)?,
                );
                let slots = top_k + 1;
                let (mut ids_ok, mut pe, mut we) = (true, 0.0f32, 0.0f32);
                for t in 0..PICK_TOKENS {
                    let h = host_route_at(&logits[t * rows..(t + 1) * rows], experts, top_k)?;
                    ids_ok &= ids[t * slots..(t + 1) * slots] == h.ids[..];
                    pe = pe.max(max_abs(&probs[t * experts..(t + 1) * experts], &h.probs));
                    we = we.max(max_abs(&weights[t * slots..(t + 1) * slots], &h.weights));
                }
                let pass = ids_ok && pe <= BAND && we <= BAND;
                println!(
                    "picks experts={experts} k={top_k} route n={PICK_TOKENS} vs host: \
                     ids_exact={ids_ok} probs_err={pe:.3e} weights_err={we:.3e} (band {BAND:.0e}) {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }

        const M: usize = 5;
        const K3: usize = 3;
        let wh = wide_weight();
        let w = DeviceTensor::upload(stream, &wh, WIDE_ROWS, WIDE_K)?;
        let cols = wide_cols(M);
        let want: Vec<f32> = (0..M).flat_map(|t| wide_logits(&wh, &cols, t)).collect();
        let d = dims(WIDE_EXPERT, K3)?;
        let mut fout = RouterOut::with_tokens(stream, d, MAX_TOKENS)?;
        let mut rout = RouterOut::for_ubatch(stream, d, M)?;
        c.router.enqueue_fused(
            stream,
            &w,
            &DeviceBuffer::from_host(stream, &cols)?,
            M,
            sink,
            &mut fout,
        )?;
        stream.synchronize()?;
        let got = read_wide(stream, &fout)?;
        rout.logits
            .copy_from_device_async(&DeviceBuffer::from_host(stream, &want)?, stream)?;
        c.router.enqueue_route(stream, M, sink, &mut rout)?;
        stream.synchronize()?;
        let alone = read_wide(stream, &rout)?;
        let slots = K3 + 1;
        let logits = bits_equal(&got.logits[..M * WIDE_ROWS], &want);
        let routing = bits_equal(
            &got.probs[..M * WIDE_EXPERT],
            &alone.probs[..M * WIDE_EXPERT],
        ) && got.ids[..M * slots] == alone.ids[..M * slots]
            && bits_equal(&got.weights[..M * slots], &alone.weights[..M * slots]);
        let pass = logits && routing;
        println!(
            "picks experts={WIDE_EXPERT} k={K3} fused m={M} (exact logits): logits_bits={logits} \
             routing = the routing alone's bits {routing} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    fn wide_case(c: &Ctx) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();

        // The instance against the gate's own widths.
        let d = wide()?;
        let consts = (d.experts(), d.used(), d.slots(), d.logits(), d.experts());
        let want = (WIDE_EXPERT, WIDE_USED, WIDE_SLOTS, WIDE_ROWS, WIDE_EXPERT);
        let row = d.inst() == ROW_WIDE;
        let pass = consts == want && row;
        println!(
            "wide512 instance: (experts, used, slots, rows, shared) = {consts:?} (want {want:?}), \
             the wide row {row} {}",
            verdict(pass)
        );
        if !pass {
            return Ok(false);
        }
        let mut ok = true;

        let wh = wide_weight();
        let w = DeviceTensor::upload(stream, &wh, WIDE_ROWS, WIDE_K)?;
        let cols = wide_cols(WIDE_UB);
        let want: Vec<Vec<f32>> = (0..WIDE_UB).map(|t| wide_logits(&wh, &cols, t)).collect();

        // The fused launch: m = 1 takes the one-column gemv, 5 and 8 the
        // m-column one.
        let mut fout = RouterOut::with_tokens(stream, wide()?, MAX_TOKENS)?;
        let mut fused = WideCheck::new();
        let mut tickets = Vec::new();
        for m in [1usize, 5, MAX_TOKENS] {
            let x = DeviceBuffer::from_host(stream, &cols[..m * WIDE_K])?;
            c.router.enqueue_fused(stream, &w, &x, m, sink, &mut fout)?;
            stream.synchronize()?;
            let got = read_wide(stream, &fout)?;
            tickets.push(fout.tickets(stream)?);
            if !wide_sized(&got, m) {
                println!(
                    "wide512 fused m={m}: buffers {} / {} / {} / {} values, want {m} tokens of \
                     {WIDE_ROWS} / {WIDE_EXPERT} / {WIDE_SLOTS} / {WIDE_SLOTS} FAIL",
                    got.logits.len(),
                    got.probs.len(),
                    got.ids.len(),
                    got.weights.len()
                );
                return Ok(false);
            }
            for (t, lt) in want.iter().enumerate().take(m) {
                fused.add(&wide_token(&got, t), lt)?;
            }
        }
        ok &= fused.line("fused m=1,5,8 vs host (exact logits)");
        let pass = tickets.iter().all(|&t| t == 0);
        println!(
            "wide512 fused tickets after each launch {tickets:?} {}",
            verdict(pass)
        );
        ok &= pass;

        // The routing alone over the host's exact logits of every token, one
        // launch (warps 0..8 of nine blocks), then again: bit-identical.
        let flat: Vec<f32> = want.concat();
        let mut rout = RouterOut::with_tokens(stream, wide()?, 1)?;
        let mut uout = RouterOut::for_ubatch(stream, wide()?, WIDE_UB)?;
        let a = wide_route(c, &flat, WIDE_UB, sink, &mut uout)?;
        let b = wide_route(c, &flat, WIDE_UB, sink, &mut uout)?;
        let mut route = WideCheck::new();
        for (t, lt) in want.iter().enumerate() {
            route.add(&wide_token(&a, t), lt)?;
        }
        ok &= route.line(&format!("route n={WIDE_UB} vs host"));
        let pass = routed_equal(&a, &b);
        println!("wide512 route rerun bit-identical={pass} {}", verdict(pass));
        ok &= pass;

        // The ubatch pair: its logits the host's exact ones, every token the
        // fused launch's bits at one token.
        let x = DeviceBuffer::from_host(stream, &cols)?;
        c.router
            .enqueue_ubatch(stream, &w, &x, WIDE_UB, sink, &mut uout)?;
        stream.synchronize()?;
        let ub = read_wide(stream, &uout)?;
        let mut pair = WideCheck::new();
        let mut same = 0usize;
        for (t, lt) in want.iter().enumerate() {
            let gt = wide_token(&ub, t);
            pair.add(&gt, lt)?;
            let xt = DeviceBuffer::from_host(stream, &cols[t * WIDE_K..(t + 1) * WIDE_K])?;
            c.router
                .enqueue_fused(stream, &w, &xt, 1, sink, &mut rout)?;
            stream.synchronize()?;
            same += usize::from(routed_equal(
                &gt,
                &wide_token(&read_wide(stream, &rout)?, 0),
            ));
        }
        ok &= pair.line(&format!(
            "ubatch n={WIDE_UB} (logits_512 + route_512, 3 token blocks x 17 row blocks) vs host"
        ));
        let pass = same == WIDE_UB;
        println!(
            "wide512 ubatch vs the fused launch per token: {same}/{WIDE_UB} bit-identical {}",
            verdict(pass)
        );
        ok &= pass;

        // Constructed ties: lanes own experts `lane + 32 j`, j < 16.
        let mut cases: Vec<(&str, Vec<f32>, Vec<u32>)> = Vec::new();
        let mut l = vec![0.5f32; WIDE_ROWS];
        l[WIDE_EXPERT] = 0.0;
        cases.push(("all-equal", l, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 512]));
        let mut l = vec![-4.0f32; WIDE_ROWS];
        for (i, &e) in [500usize, 300, 64, 33, 5, 511, 70, 290, 257]
            .iter()
            .enumerate()
        {
            l[e] = 3.0 - 0.2 * i as f32;
        }
        // Lanes 0, 0, 1, 1 at j = 15, 1, 0, 14.
        for e in [480usize, 32, 1, 449] {
            l[e] = 1.0;
        }
        l[WIDE_EXPERT] = -1.5;
        cases.push((
            "tenth-shared-by-480-32-1-449",
            l,
            vec![500, 300, 64, 33, 5, 511, 70, 290, 257, 1, 512],
        ));
        let mut l = vec![-4.0f32; WIDE_ROWS];
        for (i, e) in (40..48).enumerate() {
            l[e] = 1.0 - 0.1 * i as f32;
        }
        l[511] = 2.0;
        l[31] = 2.0;
        l[WIDE_EXPERT] = 3.0;
        cases.push((
            "first-shared-by-511-31",
            l,
            vec![31, 511, 40, 41, 42, 43, 44, 45, 46, 47, 512],
        ));
        let mut l = vec![-4.0f32; WIDE_ROWS];
        // Lanes 0, 1, 0, 0 at j = 11, 9, 9, 8: the upper half only this
        // instance has.
        for e in [352usize, 289, 288, 256] {
            l[e] = 2.5;
        }
        for (i, &e) in [100usize, 200, 300, 400, 500, 510].iter().enumerate() {
            l[e] = 2.0 - 0.1 * i as f32;
        }
        l[WIDE_EXPERT] = 0.0;
        cases.push((
            "first-four-shared-upper-half",
            l,
            vec![256, 288, 289, 352, 100, 200, 300, 400, 500, 510, 512],
        ));
        for (name, logits, want_ids) in &cases {
            let got = wide_token(&wide_route(c, logits, 1, sink, &mut rout)?, 0);
            let h = host_route_at(logits, WIDE_EXPERT, WIDE_USED)?;
            let mut one = WideCheck::new();
            let checked = one.add(&got, logits)?;
            let pass = checked && got.ids == *want_ids && h.ids == *want_ids;
            println!(
                "wide512 tie case={name} ids={:?} host={:?} want={want_ids:?} weights_bits={} {}",
                got.ids,
                h.ids,
                one.weights_bits,
                verdict(pass)
            );
            ok &= pass;
        }
        // All-equal is exact on both sides: probabilities 1/512, weights
        // 1/10 each, the gate sigmoid(0) = 1/2.
        let (_, l, _) = &cases[0];
        let got = wide_token(&wide_route(c, l, 1, sink, &mut rout)?, 0);
        let h = host_route_at(l, WIDE_EXPERT, WIDE_USED)?;
        let pass = bits_equal(&got.probs, &h.probs) && bits_equal(&got.weights, &h.weights);
        println!(
            "wide512 tie case=all-equal probabilities and all eleven weights bit-identical to the \
             host rule={pass} {}",
            verdict(pass)
        );
        ok &= pass;

        ok &= wide_fault(c, &wh, &w, &cols, &want)?;
        ok &= wide_order(c)?;
        Ok(ok)
    }

    /// The wide instance's sum orders, on inputs whose sums round: LCG
    /// weights and activations with full mantissas. The fused launch at m =
    /// 1, 5, 8 against `f32_gemv` over the same [`WIDE_ROWS`] rows followed
    /// by the routing alone per token, and the ubatch pair over
    /// [`WIDE_UB`] tokens against the fused launch per token: logits,
    /// probabilities, ids and weights BIT-EQUAL — the `logits` case's
    /// clauses at this width and at K = 2560, where the one-column walk's 80
    /// chunks run two 32-chunk trips and two 8-chunk ones. A line first
    /// shows the inputs round: logits of the m = 8 gemv that differ from
    /// the f64 dot rounded once, nonzero or the clauses below could not see
    /// an order.
    fn wide_order(c: &Ctx) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let mut s = 0x0de5_0512_u64;
        let mut val = |scale: f32| ((lcg(&mut s) % 4001) as f32 - 2000.0) * scale;
        let wh: Vec<f32> = (0..WIDE_ROWS * WIDE_K).map(|_| val(1.3e-5)).collect();
        let cols: Vec<f32> = (0..WIDE_UB * WIDE_K).map(|_| val(7.1e-4)).collect();
        let w = DeviceTensor::upload(stream, &wh, WIDE_ROWS, WIDE_K)?;
        let mut one = RouterOut::with_tokens(stream, wide()?, 1)?;
        let mut fout = RouterOut::with_tokens(stream, wide()?, MAX_TOKENS)?;
        let mut ok = true;
        for m in [1usize, 5, MAX_TOKENS] {
            let x = DeviceBuffer::from_host(stream, &cols[..m * WIDE_K])?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, WIDE_ROWS * m)?;
            c.gpu.q8f32().enqueue_f32_gemv(stream, &w, &x, m, &mut y)?;
            stream.synchronize()?;
            let y = y.to_host_vec(stream)?;
            if m == MAX_TOKENS {
                let mut rounded = 0usize;
                for t in 0..m {
                    let col = &cols[t * WIDE_K..(t + 1) * WIDE_K];
                    for r in 0..WIDE_ROWS {
                        let dot: f64 = wh[r * WIDE_K..(r + 1) * WIDE_K]
                            .iter()
                            .zip(col)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum();
                        rounded += usize::from((dot as f32).to_bits() != y[r * m + t].to_bits());
                    }
                }
                let pass = rounded > 0;
                println!(
                    "wide512 order inputs: {rounded}/{} logits of the m={m} gemv differ from the \
                     f64 dot rounded once (want > 0) {}",
                    WIDE_ROWS * m,
                    verdict(pass)
                );
                ok &= pass;
            }
            c.router.enqueue_fused(stream, &w, &x, m, sink, &mut fout)?;
            stream.synchronize()?;
            let got = read_wide(stream, &fout)?;
            let tickets = fout.tickets(stream)?;
            let mut same = 0usize;
            for t in 0..m {
                let col: Vec<f32> = (0..WIDE_ROWS).map(|r| y[r * m + t]).collect();
                let want = wide_token(&wide_route(c, &col, 1, sink, &mut one)?, 0);
                same += usize::from(routed_equal(&wide_token(&got, t), &want));
            }
            let pass = same == m && tickets == 0;
            println!(
                "wide512 order fused m={m} vs f32_gemv ({WIDE_ROWS} rows) + route_512: {same}/{m} \
                 tokens bit-identical tickets={tickets} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        let x = DeviceBuffer::from_host(stream, &cols)?;
        let mut ub = RouterOut::for_ubatch(stream, wide()?, WIDE_UB)?;
        c.router
            .enqueue_ubatch(stream, &w, &x, WIDE_UB, sink, &mut ub)?;
        stream.synchronize()?;
        let got = read_wide(stream, &ub)?;
        let mut same = 0usize;
        for t in 0..WIDE_UB {
            let xt = DeviceBuffer::from_host(stream, &cols[t * WIDE_K..(t + 1) * WIDE_K])?;
            c.router.enqueue_fused(stream, &w, &xt, 1, sink, &mut one)?;
            stream.synchronize()?;
            let want = wide_token(&read_wide(stream, &one)?, 0);
            same += usize::from(routed_equal(&wide_token(&got, t), &want));
        }
        let pass = same == WIDE_UB;
        println!(
            "wide512 order ubatch n={WIDE_UB} vs the fused launch per token: {same}/{WIDE_UB} \
             bit-identical {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// The wide instance's refusals, each launch labelled with layer 5: the
    /// routing alone on a non-finite expert logit in the upper half and on a
    /// NaN gate logit; the fused launch at five tokens with a NaN in router
    /// row 400, and the ubatch pair at nine tokens with a NaN in row 512 —
    /// every token refused. A refused token: `FaultSite::Router` with that
    /// layer, its probabilities and all eleven weights NaN, its ids as they
    /// stood; a clean run after it bit-identical to the clean run before.
    fn wide_fault(
        c: &Ctx,
        wh: &[f32],
        w: &DeviceTensor<f32>,
        cols: &[f32],
        want: &[Vec<f32>],
    ) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.layer_sink(FAULT_LAYER)?;
        let want_word = Some(Fault::at(u32::try_from(FAULT_LAYER)?, FaultSite::Router));
        let show = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let mut ok = true;

        let clean_l = want[0].clone();
        let mut out = RouterOut::with_tokens(stream, wide()?, 1)?;
        let before = c.gpu.fault()?;
        let clean = wide_route(c, &clean_l, 1, sink, &mut out)?;
        let after = c.gpu.fault()?;
        let pass = before.is_none() && after.is_none();
        println!(
            "wide512 fault clean routing: word before {before:?} after {after:?} {}",
            verdict(pass)
        );
        ok &= pass;
        // Expert 300 is lane 12's j = 9, 511 lane 31's j = 15.
        for (name, at, v) in [
            ("expert logit[300]=nan", 300usize, f32::NAN),
            ("expert logit[511]=+inf", 511, f32::INFINITY),
            ("expert logit[256]=-inf", 256, f32::NEG_INFINITY),
            ("gate logit[512]=nan", WIDE_EXPERT, f32::NAN),
        ] {
            let mut l = clean_l.clone();
            l[at] = v;
            let r = wide_route(c, &l, 1, sink, &mut out)?;
            let word = c.gpu.take_fault()?;
            let refused = wide_refused(&r, 0, &clean);
            let again = wide_route(c, &clean_l, 1, sink, &mut out)?;
            let clean_after = c.gpu.fault()?.is_none() && routed_equal(&again, &clean);
            let pass = word == want_word && refused && clean_after;
            println!(
                "wide512 fault op=qwen35moe_router_route_512 {name}: word \"{}\" (want \"{}\") \
                 refused={refused} clean_rerun={clean_after} {}",
                show(word),
                show(want_word),
                verdict(pass)
            );
            ok &= pass;
        }

        let m = 5usize;
        let xc = DeviceBuffer::from_host(stream, &cols[..m * WIDE_K])?;
        let mut fout = RouterOut::with_tokens(stream, wide()?, MAX_TOKENS)?;
        c.router.enqueue_fused(stream, w, &xc, m, sink, &mut fout)?;
        stream.synchronize()?;
        let clean_f = read_wide(stream, &fout)?;
        let clean_word = c.gpu.take_fault()?;
        let mut wn = wh.to_vec();
        wn[400 * WIDE_K + 3] = f32::NAN;
        let wnd = DeviceTensor::upload(stream, &wn, WIDE_ROWS, WIDE_K)?;
        c.router
            .enqueue_fused(stream, &wnd, &xc, m, sink, &mut fout)?;
        stream.synchronize()?;
        let got = read_wide(stream, &fout)?;
        let tickets = fout.tickets(stream)?;
        let word = c.gpu.take_fault()?;
        let refused = (0..m).all(|t| wide_refused(&got, t, &clean_f));
        let pass = clean_word.is_none() && word == want_word && refused && tickets == 0;
        println!(
            "wide512 fault op=qwen35moe_router_fused_512 m={m} router weight [400][3]=NaN: word \
             \"{}\" every token refused={refused} tickets={tickets} {}",
            show(word),
            verdict(pass)
        );
        ok &= pass;

        let n = 9usize;
        let xu = DeviceBuffer::from_host(stream, &cols[..n * WIDE_K])?;
        let mut ub = RouterOut::for_ubatch(stream, wide()?, n)?;
        c.router.enqueue_ubatch(stream, w, &xu, n, sink, &mut ub)?;
        stream.synchronize()?;
        let clean_u = read_wide(stream, &ub)?;
        let clean_word = c.gpu.take_fault()?;
        let mut wn = wh.to_vec();
        wn[WIDE_EXPERT * WIDE_K + 5] = f32::NAN;
        let wnd = DeviceTensor::upload(stream, &wn, WIDE_ROWS, WIDE_K)?;
        c.router
            .enqueue_ubatch(stream, &wnd, &xu, n, sink, &mut ub)?;
        stream.synchronize()?;
        let got = read_wide(stream, &ub)?;
        let word = c.gpu.take_fault()?;
        let refused = (0..n).all(|t| wide_refused(&got, t, &clean_u));
        let pass = clean_word.is_none() && word == want_word && refused;
        println!(
            "wide512 fault op=qwen35moe_router_logits_512+route_512 n={n} router weight [512][5]=NaN: \
             word \"{}\" every token refused={refused} {}",
            show(word),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ------------------------------------------------------------ logits

    /// `m` columns of `ffn_inp_normed` from layers `l0 ..`, five per layer.
    fn normed_cols(c: &Ctx, l0: usize, m: usize) -> Result<Vec<f32>, GateError> {
        let mut cols = Vec::with_capacity(m * HIDDEN);
        let mut l = l0;
        while cols.len() < m * HIDDEN {
            cols.extend(c.tap(0, &format!("ffn_inp_normed-{l}"))?);
            l += 1;
        }
        cols.truncate(m * HIDDEN);
        Ok(cols)
    }

    /// `f32_gemv` over the joined weight, then the routing alone per token.
    fn split_pair(
        c: &Ctx,
        ly: &Layer,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &mut RouterOut,
    ) -> Result<Routed, GateError> {
        let stream = c.stream();
        let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS * m)?;
        c.gpu
            .q8f32()
            .enqueue_f32_gemv(stream, &ly.router, x, m, &mut y)?;
        stream.synchronize()?;
        let y = y.to_host_vec(stream)?;
        let mut want = Routed {
            logits: Vec::new(),
            probs: Vec::new(),
            ids: Vec::new(),
            weights: Vec::new(),
        };
        for t in 0..m {
            let col: Vec<f32> = (0..ROWS).map(|r| y[r * m + t]).collect();
            let r = route_one(c, &col, c.gpu.unlabelled_sink(), out)?;
            want.logits.extend_from_slice(&col);
            want.probs.extend_from_slice(&r.probs);
            want.ids.extend_from_slice(&r.ids);
            want.weights.extend_from_slice(&r.weights);
        }
        Ok(want)
    }

    fn fused_run(
        c: &Ctx,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        sink: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(Routed, u32), GateError> {
        let stream = c.stream();
        c.router.enqueue_fused(stream, w, x, m, sink, out)?;
        stream.synchronize()?;
        Ok((read_out(stream, out, m)?, out.tickets(stream)?))
    }

    fn logits_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let mut ok = true;
        let mut one = RouterOut::with_tokens(stream, narrow()?, 1)?;
        let mut fout = RouterOut::with_tokens(stream, narrow()?, MAX_TOKENS)?;
        let cols = normed_cols(c, ly.l, LOGITS_UB)?;
        for m in [1usize, 5, MAX_TOKENS] {
            let x = DeviceBuffer::from_host(stream, &cols[..m * HIDDEN])?;
            let want = split_pair(c, ly, &x, m, &mut one)?;
            let (got, tickets) = fused_run(c, &ly.router, &x, m, sink, &mut fout)?;
            let pass = routed_equal(&got, &want) && tickets == 0;
            println!(
                "logits layer={} fused m={m} vs f32_gemv (257 rows) + route: logits_bits={} \
                 probs_bits={} ids={} weights_bits={} tickets={tickets} {}",
                ly.l,
                bits_equal(&got.logits, &want.logits),
                bits_equal(&got.probs, &want.probs),
                got.ids == want.ids,
                bits_equal(&got.weights, &want.weights),
                verdict(pass)
            );
            ok &= pass;
        }

        // The ubatch pair against the fused launch per token.
        let n = LOGITS_UB;
        let x = DeviceBuffer::from_host(stream, &cols[..n * HIDDEN])?;
        let mut ub = RouterOut::for_ubatch(stream, narrow()?, n)?;
        c.router
            .enqueue_ubatch(stream, &ly.router, &x, n, sink, &mut ub)?;
        stream.synchronize()?;
        let got = read_out(stream, &ub, n)?;
        let mut same = 0usize;
        for t in 0..n {
            let xt = DeviceBuffer::from_host(stream, &cols[t * HIDDEN..(t + 1) * HIDDEN])?;
            let (want, _) = fused_run(c, &ly.router, &xt, 1, sink, &mut one)?;
            let (r, e, s) = (
                t * ROWS..(t + 1) * ROWS,
                t * N_EXPERT..(t + 1) * N_EXPERT,
                t * N_SLOTS..(t + 1) * N_SLOTS,
            );
            let eq = bits_equal(&got.logits[r], &want.logits)
                && bits_equal(&got.probs[e], &want.probs)
                && got.ids[s.clone()] == want.ids[..]
                && bits_equal(&got.weights[s], &want.weights);
            same += usize::from(eq);
        }
        let pass = same == n;
        println!(
            "logits layer={} ubatch n={n} (logits + route) vs the fused launch per token: {same}/{n} \
             tokens bit-identical {}",
            ly.l,
            verdict(pass)
        );
        ok &= pass;

        // The fused launch as a captured graph.
        let x = DeviceBuffer::from_host(stream, &cols[..MAX_TOKENS * HIDDEN])?;
        let (eager, _) = fused_run(c, &ly.router, &x, MAX_TOKENS, sink, &mut fout)?;
        fout.logits.zero_async(stream)?;
        fout.probs.zero_async(stream)?;
        fout.ids.zero_async(stream)?;
        fout.weights.zero_async(stream)?;
        stream.synchronize()?;
        let graph = c.gpu.capture(|s| {
            c.router
                .enqueue_fused(s, &ly.router, &x, MAX_TOKENS, sink, &mut fout)
        })?;
        let (mut replays, mut tickets) = (true, Vec::new());
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            replays &= routed_equal(&eager, &read_out(stream, &fout, MAX_TOKENS)?);
            tickets.push(fout.tickets(stream)?);
        }
        let nodes = graph.node_count();
        let pass = replays && nodes == 1 && tickets.iter().all(|&t| t == 0);
        println!(
            "graph op=qwen35moe_router_fused m={MAX_TOKENS} two_replays_bit_identical={replays} \
             tickets_after={tickets:?} graph_nodes={nodes} {}",
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ------------------------------------------------------------ norm

    fn q8_equal(a: &Q8ActHost, b: &Q8ActHost) -> bool {
        a.q3 == b.q3 && a.q4 == b.q4 && a.q6 == b.q6 && a.s8 == b.s8 && bits_equal(&a.d8, &b.d8)
    }

    /// The FFN input residual of layer `l` (its `ssm_output`), `m` rows
    /// from layers `l ..`.
    fn resid_rows(c: &Ctx, l: usize, m: usize) -> Result<Vec<f32>, GateError> {
        let mut rows = Vec::with_capacity(m * HIDDEN);
        let mut at = l;
        while rows.len() < m * HIDDEN {
            rows.extend(c.tap(0, &format!("ssm_output-{at}"))?);
            at += 1;
        }
        rows.truncate(m * HIDDEN);
        Ok(rows)
    }

    /// The norm-fused launch on one token `x`: the q8_1 planes, the routing
    /// and the ticket count.
    fn norm_fused(
        c: &Ctx,
        ly: &Layer,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        sink: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(Q8ActHost, Routed, u32), GateError> {
        let stream = c.stream();
        let mut act = Q8Act::with_k(stream, 1, HIDDEN)?;
        c.router
            .enqueue_norm_fused(stream, w, x, &ly.gain, c.eps, &mut act, sink, out)?;
        stream.synchronize()?;
        Ok((
            readback_q8act(stream, &act)?,
            read_out(stream, out, 1)?,
            out.tickets(stream)?,
        ))
    }

    fn norm_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let rows = resid_rows(c, ly.l, 5)?;
        let mut out_s = RouterOut::with_tokens(stream, narrow()?, 1)?;
        let mut out_f = RouterOut::with_tokens(stream, narrow()?, 1)?;
        let (mut same, mut tickets_ok) = (0usize, true);
        for t in 0..5 {
            let x = DeviceBuffer::from_host(stream, &rows[t * HIDDEN..(t + 1) * HIDDEN])?;
            let mut act = Q8Act::with_k(stream, 1, HIDDEN)?;
            let mut normed = DeviceBuffer::<f32>::zeroed(stream, HIDDEN)?;
            c.norm
                .enqueue_norm_quant(stream, &x, &ly.gain, c.eps, &mut act, &mut normed, sink)?;
            let (want, _) = fused_run(c, &ly.router, &normed, 1, sink, &mut out_s)?;
            let want_q = readback_q8act(stream, &act)?;
            let (got_q, got, tickets) = norm_fused(c, ly, &ly.router, &x, sink, &mut out_f)?;
            let eq = q8_equal(&got_q, &want_q) && routed_equal(&got, &want);
            same += usize::from(eq);
            tickets_ok &= tickets == 0;
        }
        let pass = same == 5 && tickets_ok;
        println!(
            "norm layer={} src=ssm_output-{} vs norm_quant + fused: {same}/5 tokens bit-identical \
             (q8_1 planes, logits, probs, ids, weights) tickets_zero={tickets_ok} {}",
            ly.l,
            ly.l,
            verdict(pass)
        );

        let x = DeviceBuffer::from_host(stream, &rows[..HIDDEN])?;
        let (eq_q, eager, _) = norm_fused(c, ly, &ly.router, &x, sink, &mut out_f)?;
        let mut act = Q8Act::with_k(stream, 1, HIDDEN)?;
        let graph = c.gpu.capture(|s| {
            c.router.enqueue_norm_fused(
                s, &ly.router, &x, &ly.gain, c.eps, &mut act, sink, &mut out_f,
            )
        })?;
        let mut replays = true;
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            replays &= q8_equal(&readback_q8act(stream, &act)?, &eq_q)
                && routed_equal(&read_out(stream, &out_f, 1)?, &eager)
                && out_f.tickets(stream)? == 0;
        }
        let nodes = graph.node_count();
        let gpass = replays && nodes == 1;
        println!(
            "graph op=qwen35moe_router_norm two_replays_bit_identical={replays} graph_nodes={nodes} {}",
            verdict(gpass)
        );
        Ok(pass && gpass)
    }

    // ------------------------------------------------------------ fault

    fn fault_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.layer_sink(FAULT_LAYER)?;
        let want = Some(Fault::at(u32::try_from(FAULT_LAYER)?, FaultSite::Router));
        let show = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let mut ok = true;

        // The routing alone.
        let (lg, _) = ik_logits(c, 0, ly.l)?;
        let clean_l = lg[..ROWS].to_vec();
        let mut out = RouterOut::with_tokens(stream, narrow()?, 1)?;
        let before = c.gpu.fault()?;
        let clean = route_one(c, &clean_l, sink, &mut out)?;
        let after = c.gpu.fault()?;
        let pass = before.is_none() && after.is_none();
        println!(
            "fault clean routing: word before {before:?} after {after:?} {}",
            verdict(pass)
        );
        ok &= pass;
        for (name, at, v) in [
            ("expert logit[37]=nan", 37usize, f32::NAN),
            ("gate logit[256]=nan", SHARED, f32::NAN),
            ("gate logit[256]=+inf", SHARED, f32::INFINITY),
        ] {
            let mut l = clean_l.clone();
            l[at] = v;
            let r = route_one(c, &l, sink, &mut out)?;
            let word = c.gpu.take_fault()?;
            let refused = token_refused(&r, 0, &clean);
            let again = route_one(c, &clean_l, sink, &mut out)?;
            let clean_after = c.gpu.fault()?.is_none() && token_equal(&again, &clean, 0);
            let pass = word == want && refused && clean_after;
            println!(
                "fault op=qwen35moe_router_route {name}: word \"{}\" (want \"{}\") refused={refused} \
                 clean_rerun={clean_after} {}",
                show(word),
                show(want),
                verdict(pass)
            );
            ok &= pass;
        }

        // The fused launch: a NaN router weight in an expert row and in the
        // gate row refuses every token.
        let m = 5usize;
        let cols = normed_cols(c, ly.l, 9)?;
        let xc = DeviceBuffer::from_host(stream, &cols[..m * HIDDEN])?;
        let mut fout = RouterOut::with_tokens(stream, narrow()?, MAX_TOKENS)?;
        let (clean_f, _) = fused_run(c, &ly.router, &xc, m, sink, &mut fout)?;
        let clean_word = c.gpu.take_fault()?;
        let all: Vec<usize> = (0..m).collect();
        for row in [17usize, SHARED] {
            let mut wn = ly.router_host.clone();
            wn[row * HIDDEN + 3] = f32::NAN;
            let wnd = DeviceTensor::upload(stream, &wn, ROWS, HIDDEN)?;
            let (got, tickets) = fused_run(c, &wnd, &xc, m, sink, &mut fout)?;
            let word = c.gpu.take_fault()?;
            let refused = all.iter().all(|&t| token_refused(&got, t, &clean_f));
            let pass = clean_word.is_none() && word == want && refused && tickets == 0;
            println!(
                "fault op=qwen35moe_router_fused m={m} router weight [{row}][3]=NaN: word \"{}\" \
                 every token refused={refused} tickets={tickets} {}",
                show(word),
                verdict(pass)
            );
            ok &= pass;
        }

        // The ubatch pair at nine tokens: a NaN in the gate row refuses
        // every token; a column overflowing the gate row refuses its token.
        let ub_n = 9usize;
        let xu = DeviceBuffer::from_host(stream, &cols[..ub_n * HIDDEN])?;
        let mut ub = RouterOut::for_ubatch(stream, narrow()?, ub_n)?;
        c.router
            .enqueue_ubatch(stream, &ly.router, &xu, ub_n, sink, &mut ub)?;
        stream.synchronize()?;
        let clean_u = read_out(stream, &ub, ub_n)?;
        let clean_word = c.gpu.take_fault()?;
        let mut wn = ly.router_host.clone();
        wn[SHARED * HIDDEN + 5] = f32::NAN;
        let wnd = DeviceTensor::upload(stream, &wn, ROWS, HIDDEN)?;
        c.router
            .enqueue_ubatch(stream, &wnd, &xu, ub_n, sink, &mut ub)?;
        stream.synchronize()?;
        let got = read_out(stream, &ub, ub_n)?;
        let word = c.gpu.take_fault()?;
        let refused = (0..ub_n).all(|t| token_refused(&got, t, &clean_u));
        let pass = clean_word.is_none() && word == want && refused;
        println!(
            "fault op=qwen35moe_router_logits+route n={ub_n} router weight [256][5]=NaN: word \"{}\" \
             every token refused={refused} {}",
            show(word),
            verdict(pass)
        );
        ok &= pass;
        let hot: Vec<f32> = ly.router_host[SHARED * HIDDEN..(SHARED + 1) * HIDDEN]
            .iter()
            .map(|&w| if w < 0.0 { -BIG } else { BIG })
            .collect();
        let mut xb = cols[..ub_n * HIDDEN].to_vec();
        xb[(ub_n - 1) * HIDDEN..].copy_from_slice(&hot);
        c.router.enqueue_ubatch(
            stream,
            &ly.router,
            &DeviceBuffer::from_host(stream, &xb)?,
            ub_n,
            sink,
            &mut ub,
        )?;
        stream.synchronize()?;
        let got = read_out(stream, &ub, ub_n)?;
        let word = c.gpu.take_fault()?;
        let gate_logit = got.logits[(ub_n - 1) * ROWS + SHARED];
        let refused = token_refused(&got, ub_n - 1, &clean_u);
        let others = (0..ub_n - 1).all(|t| token_equal(&got, &clean_u, t));
        let pass = word == want && gate_logit == f32::INFINITY && refused && others;
        println!(
            "fault op=qwen35moe_router_logits+route n={ub_n} token {} overflows the gate row \
             (logit {gate_logit}): word \"{}\" refused={refused} other tokens bit-identical={others} {}",
            ub_n - 1,
            show(word),
            verdict(pass)
        );
        ok &= pass;

        // The norm-fused launch with a NaN in the gate row.
        let rows = resid_rows(c, ly.l, 1)?;
        let x0 = DeviceBuffer::from_host(stream, &rows)?;
        let mut o1 = RouterOut::with_tokens(stream, narrow()?, 1)?;
        let (cq, cr, _) = norm_fused(c, ly, &ly.router, &x0, sink, &mut o1)?;
        let clean_word = c.gpu.take_fault()?;
        let (gq, gr, _) = norm_fused(c, ly, &wnd, &x0, sink, &mut o1)?;
        let word = c.gpu.take_fault()?;
        let q8_same = q8_equal(&gq, &cq);
        let refused = token_refused(&gr, 0, &cr);
        let pass = clean_word.is_none() && word == want && refused && q8_same;
        println!(
            "fault op=qwen35moe_router_norm router weight [256][5]=NaN: word \"{}\" refused={refused} \
             q8_1 bytes = clean run's {q8_same} {}",
            show(word),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ------------------------------------------------------------ join

    fn join_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let mut ok = true;
        for part in ["gate", "up", "down"] {
            let (e, s, _) = stack_names(ly.l, part);
            let (_, _, eb) = c.bytes(&e)?;
            let (_, _, sb) = c.bytes(&s)?;
            let got = ly.joint(part)?.buf().to_host_vec(stream)?;
            let (we, ws) = (bytes_to_words(eb), bytes_to_words(sb));
            let pass = eb.len().is_multiple_of(4)
                && got.len() == we.len() + ws.len()
                && got[..we.len()] == we[..]
                && got[we.len()..] == ws[..];
            println!(
                "join layer={} {part}: experts 0..256 = {e} and expert 256 = {s} byte for byte ({} + {} \
                 words) {}",
                ly.l,
                we.len(),
                ws.len(),
                verdict(pass)
            );
            ok &= pass;
        }
        // The F32 join of the router weight.
        let mut w = Weights::load_where(stream, &c.split, |n| {
            n == format!("blk.{}.ffn_gate_inp.weight", ly.l)
                || n == format!("blk.{}.ffn_gate_inp_shexp.weight", ly.l)
        })?;
        let joint = format!("derived.blk.{}.ffn_gate_inp_sh", ly.l);
        w.join_rows(
            stream,
            &[
                format!("blk.{}.ffn_gate_inp.weight", ly.l).as_str(),
                format!("blk.{}.ffn_gate_inp_shexp.weight", ly.l).as_str(),
            ],
            joint.clone(),
        )?;
        let pass = match w.get(&joint) {
            Some(DevWeight::F32 { w: t, k }) => {
                let got = t.buf().to_host_vec(stream)?;
                *k == HIDDEN
                    && t.rows() == ROWS
                    && bits_equal(&got, &ly.router_host)
                    && bits_equal(&got[SHARED * HIDDEN..], &ly.router_host[SHARED * HIDDEN..])
            }
            _ => false,
        };
        println!(
            "join layer={} router: {joint} is F32 {ROWS} x {HIDDEN}, rows 0..256 = ffn_gate_inp and row \
             256 = ffn_gate_inp_shexp bit for bit {}",
            ly.l,
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ------------------------------------------------------------ down512

    /// The q4 slot of value-order word `v4` (`cores::q4_slot`).
    fn q4_slot(v4: usize) -> usize {
        256 * (v4 >> 8) + 32 * (v4 & 7) + 8 * ((v4 >> 6) & 3) + ((v4 >> 3) & 7)
    }

    /// Column `c`'s logical q8_1 codes, block scales and group sums from a
    /// readback of `cols` columns of `k`.
    fn act_col(a: &Q8ActHost, k: usize, c: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
        let n_sb = k / 256;
        let cw = 256 * n_sb.div_ceil(4);
        let codes = (0..k)
            .map(|v| {
                let w = a.q4[c * cw + q4_slot(v / 4)];
                i32::from((w >> (8 * (v % 4))) as u8 as i8)
            })
            .collect();
        (
            codes,
            a.d8[c * 2 * n_sb..(c + 1) * 2 * n_sb].to_vec(),
            a.s8[c * 8 * n_sb..(c + 1) * 8 * n_sb].to_vec(),
        )
    }

    /// `q4k_gemv_sel`'s row at K = 512 on the host: lane `(grp, s)` with grp
    /// < 2 holds super-block grp's sub-block `s` term `fma(e0, fma(cda, A,
    /// cdb·B), 0)` (`cda = d·sc`, `cdb = fma(8d, sc, −(dmin·m))`, A the dot of
    /// the nibbles less 8 with the codes, B the group's code sum), the
    /// others 0, then the warp butterfly.
    fn q4k_row_host(row: &[u8], codes: &[i32], d8: &[f32], s8: &[i32]) -> f32 {
        let mut f = [0.0f32; 32];
        for (lane, fl) in f.iter_mut().enumerate() {
            let (grp, s) = (lane >> 3, lane & 7);
            if grp >= 2 {
                continue;
            }
            let sb = &row[144 * grp..144 * (grp + 1)];
            let d = half_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
            let dmin = half_to_f32(u16::from_le_bytes([sb[2], sb[3]]));
            let scales: [u8; 12] = sb[4..16].try_into().expect("twelve scale bytes");
            let (sc, mn) = q4k_scales(&scales);
            let (sc, mn) = (f32::from(sc[s]), f32::from(mn[s]));
            let mut a = 0i32;
            for l in 0..32 {
                let nib = i32::from((sb[16 + 32 * (s >> 1) + l] >> (4 * (s & 1))) & 0x0f);
                a += (nib - 8) * codes[256 * grp + 32 * s + l];
            }
            let b = s8[8 * grp + s];
            let cda = d * sc;
            let cdb = (d * 8.0).mul_add(sc, -(dmin * mn));
            let sub = cda.mul_add(a as f32, cdb * b as f32);
            *fl = d8[2 * grp + (s >> 2)].mul_add(sub, 0.0);
        }
        butterfly(f)
    }

    /// `q6k_gemv_sel`'s row at K = 512 on the host: lane `(half, g)` holds
    /// super-block `half`'s scale group `g` term `fma((e0·d)·sc, A, 0)`, A the
    /// dot of the group's sixteen 6-bit values less 32 with the codes, then
    /// the warp butterfly.
    fn q6k_row_host(row: &[u8], codes: &[i32], d8: &[f32]) -> f32 {
        let mut f = [0.0f32; 32];
        for (lane, fl) in f.iter_mut().enumerate() {
            let (half, g) = (lane >> 4, lane & 15);
            let sb = &row[210 * half..210 * (half + 1)];
            let (ql, qh, sc) = (&sb[..128], &sb[128..192], &sb[192..208]);
            let d = half_to_f32(u16::from_le_bytes([sb[208], sb[209]]));
            let mut a = 0i32;
            for v in 16 * g..16 * g + 16 {
                let (n, k, l) = (v / 128, (v % 128) / 32, v % 32);
                let qlb = ql[64 * n + l + 32 * (k & 1)];
                let lo = if k < 2 { qlb & 0x0f } else { qlb >> 4 };
                let hi = (qh[32 * n + l] >> (2 * k)) & 3;
                a += (i32::from(lo | (hi << 4)) - 32) * codes[256 * half + v];
            }
            let e0 = d8[2 * half + (g >> 3)];
            let prod = (e0 * d) * f32::from(sc[g] as i8);
            *fl = prod.mul_add(a as f32, 0.0);
        }
        butterfly(f)
    }

    /// Row `r` of expert `e`'s part of layer `l`'s joined `part` stack, from
    /// the file: the exps tensor for e < 256, the shexp tensor for e = 256.
    fn file_row<'a>(
        c: &'a Ctx,
        l: usize,
        part: &str,
        e: usize,
        r: usize,
        rows: usize,
    ) -> Result<&'a [u8], GateError> {
        let (en, sn, _) = stack_names(l, part);
        let (name, e) = if e < N_EXPERT { (en, e) } else { (sn, 0) };
        let (_, dims, b) = c.bytes(&name)?;
        let all_rows: u64 = dims[1..].iter().product();
        let rb = b.len() / all_rows as usize;
        let at = (e * rows + r) * rb;
        Ok(&b[at..at + rb])
    }

    fn down512_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let sel: Vec<u32> = vec![3, 256, 77, 255, 3, 128, 0, 200, 256, 31, 64, 1, 250];
        let n = sel.len();
        let mut h = c.tap(0, &format!("ffn_moe_gate_par-{}", ly.l))?;
        h.extend(c.tap(0, &format!("ffn_up_gate-{}", ly.l))?);
        if h.len() < n * FF {
            return Err(
                format!("layer {}: {} SwiGLU values, want {n} x {FF}", ly.l, h.len()).into(),
            );
        }
        h.truncate(n * FF);
        let hd = DeviceBuffer::from_host(stream, &h)?;
        let mut act = Q8Act::with_slots(stream, n, FF)?;
        c.gpu.enqueue_quantize_q8_1(&hd, &mut act)?;
        let seld = DeviceBuffer::from_host(stream, &sel)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; n * HIDDEN])?;
        let wd = ly.joint("down")?;
        match ly.down_ty {
            GgmlType::Q4_K => c
                .gpu
                .q4k_sel()
                .enqueue_gemv_q4k_sel(stream, wd, &act, &seld, n, HIDDEN, &mut y)?,
            GgmlType::Q6_K => {
                c.q6.enqueue_gemv_q6k_sel(stream, wd, &act, &seld, n, HIDDEN, &mut y)?
            }
            ty => return Err(format!("layer {} down is {ty:?}", ly.l).into()),
        }
        stream.synchronize()?;
        let y = y.to_host_vec(stream)?;
        let a = readback_q8act(stream, &act)?;
        let (mut same, mut sums_ok) = (0usize, true);
        for (s, &e) in sel.iter().enumerate() {
            let (codes, d8, s8) = act_col(&a, FF, s);
            for (g, &want) in s8.iter().enumerate() {
                sums_ok &= codes[32 * g..32 * g + 32].iter().sum::<i32>() == want;
            }
            for r in 0..HIDDEN {
                let row = file_row(c, ly.l, "down", e as usize, r, HIDDEN)?;
                let want = match ly.down_ty {
                    GgmlType::Q4_K => q4k_row_host(row, &codes, &d8, &s8),
                    _ => q6k_row_host(row, &codes, &d8),
                };
                same += usize::from(y[s * HIDDEN + r].to_bits() == want.to_bits());
            }
        }
        let pass = same == n * HIDDEN && sums_ok;
        println!(
            "down512 layer={} {:?} _sel K={FF} n_sb=2 slots={n} (sel {sel:?}): {same}/{} rows \
             bit-identical to the host row dot, group sums = codes {sums_ok} {}",
            ly.l,
            ly.down_ty,
            n * HIDDEN,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ ffn

    /// The nine-slot FFN's buffers for `m` tokens.
    struct FfnBufs {
        m: usize,
        act_x: Q8Act,
        normed: DeviceBuffer<f32>,
        out: RouterOut,
        h: DeviceBuffer<f32>,
        act_h: Q8Act,
        down: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
    }

    impl FfnBufs {
        fn new(stream: &CudaStream, m: usize) -> Result<FfnBufs, GateError> {
            Ok(FfnBufs {
                m,
                act_x: Q8Act::with_k(stream, m, HIDDEN)?,
                normed: DeviceBuffer::zeroed(stream, m * HIDDEN)?,
                out: RouterOut::with_tokens(stream, narrow()?, MAX_TOKENS)?,
                h: DeviceBuffer::zeroed(stream, m * N_SLOTS * FF)?,
                act_h: Q8Act::with_slots(stream, m * N_SLOTS, FF)?,
                down: DeviceBuffer::zeroed(stream, m * N_SLOTS * HIDDEN)?,
                y: DeviceBuffer::zeroed(stream, m * HIDDEN)?,
            })
        }
    }

    /// The down `_sel` by the layer's type.
    #[allow(
        clippy::too_many_arguments,
        reason = "the down launch's operands, as the two kernels take them"
    )]
    fn enqueue_down(
        c: &Ctx,
        s: &CudaStream,
        ty: GgmlType,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        match ty {
            GgmlType::Q4_K => c
                .gpu
                .q4k_sel()
                .enqueue_gemv_q4k_sel(s, w, act, sel, n_slots, HIDDEN, y),
            _ => {
                c.q6.enqueue_gemv_q6k_sel(s, w, act, sel, n_slots, HIDDEN, y)
            }
        }
    }

    /// Enqueue the nine-slot FFN of `b.m` tokens with residual `x` into
    /// `b.y`: the router with the norm (one launch at one token), gate·up
    /// over the joined stacks at `9m` slots whose ids are the router's, one
    /// quantizer, the down, the combine at nine slots.
    fn enqueue_ffn(
        c: &Ctx,
        s: &CudaStream,
        ly: &Layer,
        st: Stacks<'_>,
        x: &DeviceBuffer<f32>,
        b: &mut FfnBufs,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let sink = c.gpu.unlabelled_sink();
        let m = b.m;
        if m == 1 {
            c.router.enqueue_norm_fused(
                s,
                &ly.router,
                x,
                &ly.gain,
                c.eps,
                &mut b.act_x,
                sink,
                &mut b.out,
            )?;
        } else {
            c.norm
                .enqueue_norm_quant(s, x, &ly.gain, c.eps, &mut b.act_x, &mut b.normed, sink)?;
            c.router
                .enqueue_fused(s, &ly.router, &b.normed, m, sink, &mut b.out)?;
        }
        let Stacks { wg, wu, wd } = st;
        c.experts.enqueue_gate_up(
            s,
            GateUpArgs {
                wg,
                wu,
                act: &b.act_x,
                sel: &b.out.ids,
                n_slots: m * N_SLOTS,
                rows_per_expert: FF,
                fault: sink,
                h: &mut b.h,
            },
        )?;
        c.gpu.enqueue_quantize_q8_1(&b.h, &mut b.act_h)?;
        enqueue_down(
            c,
            s,
            ly.down_ty,
            wd,
            &b.act_h,
            &b.out.ids,
            m * N_SLOTS,
            &mut b.down,
        )?;
        c.experts.enqueue_combine_tokens(
            s,
            CombineArgs {
                down: &b.down,
                w: &b.out.weights,
                resid: x,
                rows: HIDDEN,
                n_slots: FUSED_SLOTS,
                m,
                y: &mut b.y,
            },
        )
    }

    /// A layer's three joined stacks, resolved once.
    #[derive(Clone, Copy)]
    struct Stacks<'a> {
        wg: &'a DeviceTensor<u32>,
        wu: &'a DeviceTensor<u32>,
        wd: &'a DeviceTensor<u32>,
    }

    fn stacks(ly: &Layer) -> Result<Stacks<'_>, GateError> {
        Ok(Stacks {
            wg: ly.joint("gate")?,
            wu: ly.joint("up")?,
            wd: ly.joint("down")?,
        })
    }

    /// The unfused composition of `m` tokens given the fused run's
    /// activation, ids and weights: the routed eight on the joined stacks,
    /// the shared expert on the file's shexp tensors, each gate·up,
    /// quantized and projected apart, the downs put in slot order, the
    /// combine at nine slots; and the downs in slot order.
    fn unfused(
        c: &Ctx,
        ly: &Layer,
        x: &DeviceBuffer<f32>,
        b: &FfnBufs,
        ids: &[u32],
        w9: &DeviceBuffer<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let m = b.m;
        let sel8: Vec<u32> = (0..m)
            .flat_map(|t| ids[t * N_SLOTS..t * N_SLOTS + N_USED].to_vec())
            .collect();
        let sel8 = DeviceBuffer::from_host(stream, &sel8)?;
        let zeros = DeviceBuffer::<u32>::zeroed(stream, m)?;
        let mut h8 = DeviceBuffer::<f32>::zeroed(stream, m * N_USED * FF)?;
        let mut hs = DeviceBuffer::<f32>::zeroed(stream, m * FF)?;
        c.experts.enqueue_gate_up(
            stream,
            GateUpArgs {
                wg: ly.joint("gate")?,
                wu: ly.joint("up")?,
                act: &b.act_x,
                sel: &sel8,
                n_slots: m * N_USED,
                rows_per_expert: FF,
                fault: sink,
                h: &mut h8,
            },
        )?;
        c.experts.enqueue_gate_up(
            stream,
            GateUpArgs {
                wg: &ly.sh[0],
                wu: &ly.sh[1],
                act: &b.act_x,
                sel: &zeros,
                n_slots: m,
                rows_per_expert: FF,
                fault: sink,
                h: &mut hs,
            },
        )?;
        let mut a8 = Q8Act::with_slots(stream, m * N_USED, FF)?;
        let mut a_s = Q8Act::with_slots(stream, m, FF)?;
        c.gpu.enqueue_quantize_q8_1(&h8, &mut a8)?;
        c.gpu.enqueue_quantize_q8_1(&hs, &mut a_s)?;
        let mut d8 = DeviceBuffer::<f32>::zeroed(stream, m * N_USED * HIDDEN)?;
        let mut ds = DeviceBuffer::<f32>::zeroed(stream, m * HIDDEN)?;
        enqueue_down(
            c,
            stream,
            ly.down_ty,
            ly.joint("down")?,
            &a8,
            &sel8,
            m * N_USED,
            &mut d8,
        )?;
        enqueue_down(c, stream, ly.down_ty, &ly.sh[2], &a_s, &zeros, m, &mut ds)?;
        stream.synchronize()?;
        let (d8, ds) = (d8.to_host_vec(stream)?, ds.to_host_vec(stream)?);
        let mut down9 = Vec::with_capacity(m * N_SLOTS * HIDDEN);
        for t in 0..m {
            down9.extend_from_slice(&d8[t * N_USED * HIDDEN..(t + 1) * N_USED * HIDDEN]);
            down9.extend_from_slice(&ds[t * HIDDEN..(t + 1) * HIDDEN]);
        }
        let dd = DeviceBuffer::from_host(stream, &down9)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * HIDDEN)?;
        c.experts.enqueue_combine_tokens(
            stream,
            CombineArgs {
                down: &dd,
                w: w9,
                resid: x,
                rows: HIDDEN,
                n_slots: N_SLOTS,
                m,
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, down9))
    }

    /// `((Σ_{s<8} w_s·d_s) + w_8·d_sh) + resid` per value, each term a fused
    /// multiply-add into the running sum from 0, as the combine compiles.
    fn host_combine(down9: &[f32], w9: &[f32], resid: &[f32], m: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; m * HIDDEN];
        for t in 0..m {
            for d in 0..HIDDEN {
                let mut acc = 0.0f32;
                for s in 0..N_SLOTS {
                    acc = w9[t * N_SLOTS + s].mul_add(down9[(t * N_SLOTS + s) * HIDDEN + d], acc);
                }
                y[t * HIDDEN + d] = acc + resid[t * HIDDEN + d];
            }
        }
        y
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            num += (f64::from(*x) - f64::from(*y)).powi(2);
            den += f64::from(*y).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn ffn_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let mut ok = true;
        let rows = resid_rows(c, ly.l, MAX_TOKENS)?;
        for m in [1usize, 5, MAX_TOKENS] {
            let resid = &rows[..m * HIDDEN];
            let x = DeviceBuffer::from_host(stream, resid)?;
            let mut b = FfnBufs::new(stream, m)?;
            enqueue_ffn(c, stream, ly, stacks(ly)?, &x, &mut b)?;
            stream.synchronize()?;
            let y = b.y.to_host_vec(stream)?;
            let r = read_out(stream, &b.out, m)?;
            let (y_ref, down9) = unfused(c, ly, &x, &b, &r.ids, &b.out.weights)?;
            let y_host = host_combine(&down9, &r.weights, resid, m);
            let shared_ids = (0..m).all(|t| r.ids[t * N_SLOTS + N_USED] == SHARED as u32);
            let (vs_ref, vs_host) = (bits_equal(&y, &y_ref), bits_equal(&y, &y_host));
            let pass = vs_ref && vs_host && shared_ids;
            let ik = if m == 5 {
                let lo = c.tap(0, &format!("l_out-{}", ly.l))?;
                format!(" rel_to_ik_l_out={:.3e} (printed)", rel(&y, &lo))
            } else {
                String::new()
            };
            println!(
                "ffn layer={} {:?} m={m}: nine slots vs the unfused composition bits={vs_ref}, vs \
                 the host combine ((Σ8 w·d) + w8·d_sh) + resid bits={vs_host}, slot 8 = expert \
                 {SHARED} {shared_ids}{ik} {}",
                ly.l,
                ly.down_ty,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ graph

    fn graph_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let rows = resid_rows(c, ly.l, 1)?;
        let x = DeviceBuffer::from_host(stream, &rows)?;
        let mut b = FfnBufs::new(stream, 1)?;
        enqueue_ffn(c, stream, ly, stacks(ly)?, &x, &mut b)?;
        stream.synchronize()?;
        let eager = b.y.to_host_vec(stream)?;
        b.y.zero_async(stream)?;
        stream.synchronize()?;
        let st = stacks(ly)?;
        let graph = c.gpu.capture(|s| enqueue_ffn(c, s, ly, st, &x, &mut b))?;
        let mut replays = true;
        for _ in 0..2 {
            graph.launch(stream)?;
            stream.synchronize()?;
            replays &= bits_equal(&b.y.to_host_vec(stream)?, &eager);
        }
        let nodes = graph.node_count();
        let pass = replays && nodes == 5;
        println!(
            "graph layer={} nine-slot FFN m=1: graph_nodes={nodes} (want 5) two_replays_bit_identical\
             ={replays} {}",
            ly.l,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ ubatch

    /// Expert `e`'s rows of layer `l`'s `part` stack, dequantized.
    fn expert_rows(
        c: &Ctx,
        l: usize,
        part: &str,
        e: usize,
        rows: usize,
    ) -> Result<Vec<f32>, GateError> {
        let (en, sn, _) = stack_names(l, part);
        let (name, e) = if e < N_EXPERT { (en, e) } else { (sn, 0) };
        let (ty, dims, b) = c.bytes(&name)?;
        let all_rows: u64 = dims[1..].iter().product();
        let rb = b.len() / all_rows as usize;
        let k = dims[0] as usize;
        let mut out = vec![0.0f32; rows * k];
        dequant_row(ty, &b[e * rows * rb..(e + 1) * rows * rb], &mut out)?;
        Ok(out)
    }

    /// One expert's share of the f64 reference: the (token, slot) pairs
    /// routed to it and its dequantized gate, up and down rows.
    struct ExpertJob {
        slots: Vec<(usize, usize)>,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    }

    /// The f64 reference of `n` tokens' routed FFN (no residual) from the
    /// f32 normed columns `x` and the routing `ids`, `w` (nine slots each).
    fn f64_ref(
        c: &Ctx,
        l: usize,
        x: &[f32],
        ids: &[u32],
        w: &[f32],
        n: usize,
    ) -> Result<Vec<f64>, GateError> {
        let mut by_e: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
        for t in 0..n {
            for s in 0..N_SLOTS {
                by_e.entry(ids[t * N_SLOTS + s] as usize)
                    .or_default()
                    .push((t, s));
            }
        }
        let mut jobs = Vec::new();
        for (e, slots) in by_e {
            jobs.push(ExpertJob {
                slots,
                gate: expert_rows(c, l, "gate", e, FF)?,
                up: expert_rows(c, l, "up", e, FF)?,
                down: expert_rows(c, l, "down", e, HIDDEN)?,
            });
        }
        let threads = 16usize;
        let parts: Vec<Vec<f64>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|i| {
                    let jobs = &jobs;
                    sc.spawn(move || {
                        let mut y = vec![0.0f64; n * HIDDEN];
                        for ExpertJob {
                            slots,
                            gate: g,
                            up: u,
                            down: d,
                        } in jobs.iter().skip(i).step_by(threads)
                        {
                            for &(t, s) in slots {
                                let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
                                let mut hv = vec![0.0f64; FF];
                                for (r, hr) in hv.iter_mut().enumerate() {
                                    let (mut gs, mut us) = (0.0f64, 0.0f64);
                                    for k in 0..HIDDEN {
                                        gs += f64::from(g[r * HIDDEN + k]) * f64::from(xt[k]);
                                        us += f64::from(u[r * HIDDEN + k]) * f64::from(xt[k]);
                                    }
                                    *hr = gs / (1.0 + (-gs).exp()) * us;
                                }
                                let ws = f64::from(w[t * N_SLOTS + s]);
                                for j in 0..HIDDEN {
                                    let mut ds = 0.0f64;
                                    for (r, hr) in hv.iter().enumerate() {
                                        ds += f64::from(d[j * FF + r]) * hr;
                                    }
                                    y[t * HIDDEN + j] += ws * ds;
                                }
                            }
                        }
                        y
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("f64 reference thread"))
                .collect()
        });
        let mut y = vec![0.0f64; n * HIDDEN];
        for p in parts {
            for (a, b) in y.iter_mut().zip(p) {
                *a += b;
            }
        }
        Ok(y)
    }

    fn rel64(a: &[f32], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            num += (f64::from(*x) - y).powi(2);
            den += y.powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn ubatch_case(c: &Ctx, ly: &Layer) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let n = UB_TOKENS;
        let slots = n * N_SLOTS;
        let x = normed_cols(c, 0, n)?;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let zeros = DeviceBuffer::<f32>::zeroed(stream, n * HIDDEN)?;

        // The ubatch path.
        let mut act_hid = GemmAct::new(stream, n, HIDDEN)?;
        c.gpu.enqueue_quantize_gemm(&xd, n, &mut act_hid, sink)?;
        let mut ub = RouterOut::for_ubatch(stream, narrow()?, n)?;
        c.router
            .enqueue_ubatch(stream, &ly.router, &xd, n, sink, &mut ub)?;
        let mut route = GemmRoute::new(stream, slots, ROWS)?;
        c.gemm
            .enqueue_route(stream, &ub.ids, slots, &mut route, sink)?;
        let mut gate = DeviceBuffer::<f32>::zeroed(stream, slots * FF)?;
        let mut up = DeviceBuffer::<f32>::zeroed(stream, slots * FF)?;
        for (part, y) in [("gate", &mut gate), ("up", &mut up)] {
            c.gemm.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: GemmWeight::Q4K,
                    w: ly.joint(part)?,
                    rows_per_expert: FF,
                    act: &act_hid,
                    route: &route,
                    input: GemmInput::Shared { top_k: N_SLOTS },
                    y,
                },
            )?;
        }
        let mut act_h = GemmAct::new(stream, slots, FF)?;
        c.gemm
            .enqueue_swiglu_quant(stream, &gate, &up, slots, &mut act_h, sink)?;
        let mut down = DeviceBuffer::<f32>::zeroed(stream, slots * HIDDEN)?;
        c.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty: if ly.down_ty == GgmlType::Q4_K {
                    GemmWeight::Q4K
                } else {
                    GemmWeight::Q6K
                },
                w: ly.joint("down")?,
                rows_per_expert: HIDDEN,
                act: &act_h,
                route: &route,
                input: GemmInput::PerSlot,
                y: &mut down,
            },
        )?;
        let mut y_ub = DeviceBuffer::<f32>::zeroed(stream, n * HIDDEN)?;
        c.experts.enqueue_combine_tokens(
            stream,
            CombineArgs {
                down: &down,
                w: &ub.weights,
                resid: &zeros,
                rows: HIDDEN,
                n_slots: N_SLOTS,
                m: n,
                y: &mut y_ub,
            },
        )?;
        stream.synchronize()?;
        let y_ub = y_ub.to_host_vec(stream)?;
        let r_ub = read_out(stream, &ub, n)?;
        let word = c.gpu.take_fault()?;

        // The one-token path per token.
        let mut y_one = Vec::with_capacity(n * HIDDEN);
        let mut routing_same = true;
        let z1 = DeviceBuffer::<f32>::zeroed(stream, HIDDEN)?;
        let mut out = RouterOut::with_tokens(stream, narrow()?, 1)?;
        for t in 0..n {
            let xt = DeviceBuffer::from_host(stream, &x[t * HIDDEN..(t + 1) * HIDDEN])?;
            let mut act = Q8Act::with_k(stream, 1, HIDDEN)?;
            c.gpu.enqueue_quantize_q8_1(&xt, &mut act)?;
            c.router
                .enqueue_fused(stream, &ly.router, &xt, 1, sink, &mut out)?;
            let mut h = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * FF)?;
            c.experts.enqueue_gate_up(
                stream,
                GateUpArgs {
                    wg: ly.joint("gate")?,
                    wu: ly.joint("up")?,
                    act: &act,
                    sel: &out.ids,
                    n_slots: N_SLOTS,
                    rows_per_expert: FF,
                    fault: sink,
                    h: &mut h,
                },
            )?;
            let mut ah = Q8Act::with_slots(stream, N_SLOTS, FF)?;
            c.gpu.enqueue_quantize_q8_1(&h, &mut ah)?;
            let mut d = DeviceBuffer::<f32>::zeroed(stream, N_SLOTS * HIDDEN)?;
            enqueue_down(
                c,
                stream,
                ly.down_ty,
                ly.joint("down")?,
                &ah,
                &out.ids,
                N_SLOTS,
                &mut d,
            )?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, HIDDEN)?;
            c.experts.enqueue_combine_tokens(
                stream,
                CombineArgs {
                    down: &d,
                    w: &out.weights,
                    resid: &z1,
                    rows: HIDDEN,
                    n_slots: N_SLOTS,
                    m: 1,
                    y: &mut y,
                },
            )?;
            stream.synchronize()?;
            let r = read_out(stream, &out, 1)?;
            routing_same &= r.ids[..] == r_ub.ids[t * N_SLOTS..(t + 1) * N_SLOTS]
                && bits_equal(&r.weights, &r_ub.weights[t * N_SLOTS..(t + 1) * N_SLOTS]);
            y_one.extend(y.to_host_vec(stream)?);
        }
        let y_ref = f64_ref(c, ly.l, &x, &r_ub.ids, &r_ub.weights, n)?;
        let d_path = rel(&y_ub, &y_one);
        let d_one = rel64(&y_one, &y_ref);
        let d_ub = rel64(&y_ub, &y_ref);
        let ratio = d_path / d_one.max(f64::MIN_POSITIVE);
        let pass = word.is_none() && routing_same && ratio <= SPREAD_RATIO;
        println!(
            "ubatch layer={} {:?} n={n} ({slots} slots over {ROWS} experts, down GEMM K={FF}): \
             routing = one-token path's {routing_same}; rel(ubatch, one-token)={d_path:.3e}, \
             rel(one-token, f64)={d_one:.3e}, rel(ubatch, f64)={d_ub:.3e}; ratio {ratio:.3e} (band \
             {SPREAD_RATIO}) word={word:?} {}",
            ly.l,
            ly.down_ty,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ ik

    fn ik_case(c: &Ctx, n_layer: usize) -> Result<bool, GateError> {
        let stream = c.stream();
        let sink = c.gpu.unlabelled_sink();
        let mut fout = RouterOut::with_tokens(stream, narrow()?, MAX_TOKENS)?;
        let (mut tokens, mut logits_ok, mut ids_ok, mut w_ok, mut gate_ok) =
            (0usize, true, true, true, true);
        let (mut worst_l, mut worst_w, mut worst_sig) = (0.0f64, 0.0f32, 0.0f32);
        for l in 0..n_layer {
            let wh = c.router_host(l)?;
            let wd = DeviceTensor::upload(stream, &wh, ROWS, HIDDEN)?;
            for s in 0..c.sets.len() {
                let label = c.sets[s].0;
                let man = &c.sets[s].1;
                let x = c.tap(s, &format!("ffn_inp_normed-{l}"))?;
                let m = x.len() / HIDDEN;
                if m == 0 || m > MAX_TOKENS {
                    return Err(format!("{label} layer {l}: {m} tokens").into());
                }
                let (ik_l, _) = ik_logits(c, s, l)?;
                let ik_ids = topk_ids_logical_within(
                    man,
                    man.tensor(&format!("ffn_moe_topk-{l}"), 0)?,
                    N_EXPERT as u32,
                )?;
                let ik_w = c.tap(s, &format!("ffn_moe_weights_norm-{l}"))?;
                // A one-token graph fuses the sigmoid into its multiply, so the
                // step sets carry no sigmoid tap.
                let sig_name = format!("shared_expert_gate_sigmoid-{l}");
                let ik_sig = match man.tensor(&sig_name, 0) {
                    Ok(row) => Some(ref_tensor_logical_in(&man.dir, row)?),
                    Err(_) => None,
                };
                let ik_p = c.tap(s, &format!("ffn_moe_probs-{l}"))?;
                let xd = DeviceBuffer::from_host(stream, &x)?;
                let (got, _) = fused_run(c, &wd, &xd, m, sink, &mut fout)?;
                for t in 0..m {
                    let xt = &x[t * HIDDEN..(t + 1) * HIDDEN];
                    let mut dmax = 0.0f64;
                    let mut lok = true;
                    for r in 0..ROWS {
                        let mag: f64 = wh[r * HIDDEN..(r + 1) * HIDDEN]
                            .iter()
                            .zip(xt)
                            .map(|(w, v)| (f64::from(*w) * f64::from(*v)).abs())
                            .sum();
                        let band = (gamma(HIDDEN - 1) + gamma(69)) * mag;
                        let d = (f64::from(got.logits[t * ROWS + r])
                            - f64::from(ik_l[t * ROWS + r]))
                        .abs();
                        dmax = dmax.max(d);
                        worst_l = worst_l.max(d / band.max(f64::MIN_POSITIVE));
                        if d > band {
                            lok = false;
                            if r == SHARED {
                                gate_ok = false;
                            }
                        }
                    }
                    let ours: Vec<u32> = got.ids[t * N_SLOTS..t * N_SLOTS + N_USED].to_vec();
                    let want: Vec<u32> = ik_ids[t * N_USED..(t + 1) * N_USED]
                        .iter()
                        .map(|&i| i as u32)
                        .collect();
                    let same = ours == want;
                    if !same {
                        let mut p: Vec<f32> = ik_p[t * N_EXPERT..(t + 1) * N_EXPERT].to_vec();
                        p.sort_by(|a, b| b.total_cmp(a));
                        println!(
                            "ik set={label} layer={l} t={t} ids={ours:?} ik={want:?} ik p8={:.6e} \
                             p9={:.6e} FAIL",
                            p[7], p[8]
                        );
                    }
                    let wband = 2.0 * dmax as f32 + BAND;
                    let we = max_abs(
                        &got.weights[t * N_SLOTS..t * N_SLOTS + N_USED],
                        &ik_w[t * N_USED..(t + 1) * N_USED],
                    );
                    if we > wband {
                        println!(
                            "ik set={label} layer={l} t={t} weights_err={we:.3e} band={wband:.3e} FAIL"
                        );
                    }
                    worst_w = worst_w.max(we);
                    if let Some(sig) = &ik_sig {
                        worst_sig =
                            worst_sig.max((got.weights[t * N_SLOTS + N_USED] - sig[t]).abs());
                    }
                    logits_ok &= lok;
                    ids_ok &= same;
                    w_ok &= we <= wband;
                    tokens += 1;
                }
            }
        }
        println!(
            "ik logits: {tokens} tokens, 257 logits each within (γ(2047) + γ(69))·Σ|w·x| of \
             ffn_moe_logits / shared_expert_gate (worst at {worst_l:.3} of its band; gate row \
             {gate_ok}) {}",
            verdict(logits_ok)
        );
        println!(
            "ik topk: the top 8 EQUAL ffn_moe_topk on every token {}",
            verdict(ids_ok)
        );
        println!(
            "ik weights: within 2·max|Δlogit| + {BAND:.0e} of ffn_moe_weights_norm (worst \
             {worst_w:.3e}) {}; sigmoid vs shared_expert_gate_sigmoid worst {worst_sig:.3e} \
             (printed, not pinned)",
            verdict(w_ok)
        );
        Ok(logits_ok && ids_ok && w_ok)
    }

    // ------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let filter = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            [] => None,
            ["--case", f] => Some(f.to_string()),
            _ => return Err("usage: gate_qwen35moe_moe [--case <substring>]".into()),
        };
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if split.architecture() != Some(oracle::ARCH) {
            return Err(format!(
                "the model file is {:?}, want {} — run through `just gate-gpu-qwen35moe-moe`",
                split.architecture(),
                oracle::ARCH
            )
            .into());
        }
        let get = |k: &str| split.arch_get_u64(k).ok_or_else(|| format!("no {k}"));
        let (n_expert, n_used, ff, ff_sh, hidden, n_layer) = (
            get("expert_count")?,
            get("expert_used_count")?,
            get("expert_feed_forward_length")?,
            get("expert_shared_feed_forward_length")?,
            get("embedding_length")?,
            get("block_count")? as usize,
        );
        if (n_expert, n_used, ff, ff_sh, hidden)
            != (
                N_EXPERT as u64,
                N_USED as u64,
                FF as u64,
                FF as u64,
                HIDDEN as u64,
            )
        {
            return Err(format!(
                "the file routes {n_used} of {n_expert}, FFN {ff} / shared {ff_sh}, hidden \
                 {hidden}; the kernels are {N_USED} of {N_EXPERT}, {FF}, {HIDDEN}"
            )
            .into());
        }
        let eps = split
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .ok_or("no attention.layer_norm_rms_epsilon")?;
        let gpu = Gpu::new()?;
        let ctx = gpu.context().clone();
        let mut sets = Vec::new();
        for name in [oracle::BATCH, oracle::STEP4, oracle::D1K] {
            sets.push((
                name,
                RefManifest::open(&data_dir().join(name), &oracle::IK)?,
            ));
        }
        let c = Ctx {
            router: RouterKernels::load(&ctx)?,
            experts: ExpertKernels::load(&ctx)?,
            q6: Q6kSelKernels::load(&ctx, gpu.fault_word())?,
            gemm: GemmKernels::load(&ctx)?,
            norm: FusedKernels::load(&ctx)?,
            gpu,
            split,
            eps,
            sets,
            filter,
        };
        println!(
            "gate_qwen35moe_moe: device {} model {}",
            c.gpu.device_name()?,
            path.display()
        );
        let mut ok = true;
        if c.want("shape") {
            ok &= no_local_depot(&[
                "qwen35moe_router_fused",
                "qwen35moe_router_norm",
                "qwen35moe_router_logits",
                "qwen35moe_router_route",
                "qwen35moe_router_fused_512",
                "qwen35moe_router_logits_512",
                "qwen35moe_router_route_512",
            ])?;
        }
        if c.want("routing") {
            ok &= routing_case(&c, n_layer)?;
        }
        if c.want("wide512") {
            ok &= wide_case(&c)?;
        }
        if c.want("picks") {
            ok &= picks_case(&c)?;
        }
        if c.want("ik") {
            ok &= ik_case(&c, n_layer)?;
        }
        let per_layer = [
            "logits", "norm", "fault", "join", "down512", "ffn", "graph", "ubatch",
        ];
        if per_layer.iter().any(|k| c.want(k)) {
            for &l in &LAYERS {
                let ly = load_layer(&c, l)?;
                if c.want("logits") {
                    ok &= logits_case(&c, &ly)?;
                }
                if c.want("norm") {
                    ok &= norm_case(&c, &ly)?;
                }
                if c.want("fault") && l == FAULT_LAYER {
                    ok &= fault_case(&c, &ly)?;
                }
                if c.want("join") {
                    ok &= join_case(&c, &ly)?;
                }
                if c.want("down512") {
                    ok &= down512_case(&c, &ly)?;
                }
                if c.want("ffn") {
                    ok &= ffn_case(&c, &ly)?;
                }
                if c.want("graph") {
                    ok &= graph_case(&c, &ly)?;
                }
                if c.want("ubatch") {
                    ok &= ubatch_case(&c, &ly)?;
                }
            }
        }
        println!("gate_qwen35moe_moe: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
