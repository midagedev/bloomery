//! GPU gate for the head kernel of MiniMax-M3's attention,
//! `rope_neox::head_norm_neox_append_128_partial`, model-less at the model's
//! shapes: 64 query heads over 4 key heads of 128 values, a NEOX rope over the
//! first 64 values of each head (`rope.dimension_count`) at θ 5e6, RMS eps
//! 1e-6. Inputs are seeded ([`activations`]); nothing is read from a model
//! file.
//!
//! One launch normalizes each head by its own gain (the per-head `attn_q_norm`
//! and `attn_k_norm`), turns the NEOX pairs `(i, i + 32)` of its first 64
//! values by the table's row at the token's position, passes the other 64
//! through normalized, reads the query from plain query rows at head stride
//! 128, turns the key in place and appends it and the value to the f16 planes.
//! Against the shared host transcription of that rule (`rope_gate`, the norm's
//! two warp sums in warp order, `(sum / 128) as f32`, `(scale · gain) · x`, the
//! turn, the planes the f16 of the turned key and of the value): all bit for
//! bit, the planes' other rows untouched, and a rerun bit for bit. Runs use
//! 64/4 and 8/2 query/key heads and end at position `ctx - 1`. A position at
//! the cache's height raises `cache_pos` with the launch's layer, leaves that
//! token's heads NaN and appends nothing, every other token the clean run's; a
//! head whose squares overflow raises `f32_product` with its values NaN and
//! every other head the clean run's.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minimaxm3_rope", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::fault::FaultSink;
    use bloomery_gpu::rope_neox::{HEAD, ROT_PARTIAL, RopeNeoxKernels};
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu_gates::rope_gate::{self, RopeIn, RopeOut, RopeShape};
    use bloomery_gpu_gates::{GateError, activations, checks_failed, verdict};
    use cuda_core::CudaStream;

    /// The model's attention shape.
    const N_HEAD: usize = 64;
    const N_KV: usize = 4;
    const THETA: f32 = 5.0e6;
    const EPS: f32 = 1.0e-6;

    /// The tokens, first position and cache height of a launch.
    const ROPE_M: usize = 9;
    const ROPE_P0: usize = ROPE_CTX - ROPE_M;
    const ROPE_CTX: usize = 40;
    /// The layer the fault checks label their launches with.
    const LAYER: usize = 17;

    fn rope_run(
        kern: &RopeNeoxKernels,
        stream: &CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
        (n_head, n_kv): (usize, usize),
    ) -> Result<RopeOut, GateError> {
        rope_gate::run(
            stream,
            inp,
            pos,
            RopeShape {
                head: HEAD,
                n_head,
                n_kv,
                ctx: ROPE_CTX,
                eps: EPS,
            },
            fault,
            |s, args| {
                // SAFETY: rope_gate::run checks that the positions are distinct.
                unsafe { kern.enqueue_head_norm_neox_append_128_partial(s, args) }
            },
        )
    }

    fn rope_inputs((n_head, n_kv): (usize, usize)) -> Result<(RopeIn, Vec<u32>), GateError> {
        let table_rt = RopeTable::new(&RopeSpec::window(THETA, ROT_PARTIAL))?;
        let mut table = Vec::with_capacity(ROPE_CTX * ROT_PARTIAL);
        for p in 0..ROPE_CTX {
            table_rt.push(u32::try_from(p)?, Direction::Forward, &mut table);
        }
        let gain = |seed: u32, base: f32| -> Vec<f32> {
            activations(HEAD, 1, seed)
                .iter()
                .map(|v| base + 0.5 * v)
                .collect()
        };
        let inp = RopeIn {
            qg: activations(HEAD, ROPE_M * n_head, 3),
            k: activations(HEAD, ROPE_M * n_kv, 6)
                .iter()
                .map(|v| 4.0 * v)
                .collect(),
            v: activations(HEAD, ROPE_M * n_kv, 7),
            gq: gain(4, 1.25),
            gk: gain(5, 1.3),
            table,
        };
        let pos: Vec<u32> = (0..ROPE_M)
            .map(|t| u32::try_from(ROPE_P0 + t))
            .collect::<Result<_, _>>()?;
        Ok((inp, pos))
    }

    fn rope_check(
        gpu: &Gpu,
        kern: &RopeNeoxKernels,
        heads: (usize, usize),
    ) -> Result<bool, GateError> {
        let (inp, pos) = rope_inputs(heads)?;
        let shape = RopeShape {
            head: HEAD,
            n_head: heads.0,
            n_kv: heads.1,
            ctx: ROPE_CTX,
            eps: EPS,
        };
        let exact =
            rope_gate::check::<HEAD, ROT_PARTIAL, HEAD>(gpu, &inp, &pos, shape, LAYER, |p, f| {
                rope_run(kern, gpu.stream(), &inp, p, f, heads)
            })?;
        let overflow =
            rope_gate::check_overflow::<HEAD>(gpu, &inp, &pos, shape, LAYER, |i, p, f| {
                rope_run(kern, gpu.stream(), i, p, f, heads)
            })?;
        Ok(exact && overflow)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let rope = RopeNeoxKernels::load(gpu.context())?;
        println!(
            "gate_minimaxm3_rope: device {}: {N_HEAD}/{N_KV} heads of {HEAD}, rope over the first \
             {ROT_PARTIAL} at θ {THETA:e}",
            gpu.device_name()?
        );
        let mut ok = true;
        for heads in [(N_HEAD, N_KV), (8, 2)] {
            let r = rope_check(&gpu, &rope, heads)?;
            println!(
                "rope-128-partial heads {}/{} {}",
                heads.0,
                heads.1,
                verdict(r)
            );
            ok &= r;
        }
        println!("gate_minimaxm3_rope: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
