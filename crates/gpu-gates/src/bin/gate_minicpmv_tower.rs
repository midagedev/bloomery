//! GPU gate for the MiniCPM-V image tower on the card: today the two kernels its vit_merger adds to
//! the Qwen3-VL block set, each against its host rule on seeded synthetic inputs, with the inputs
//! on the card as the tower will hold them (rows in merge order, a 2×2 window being four
//! consecutive rows). The tower's own clauses join this bin as the tower is assembled.
//!
//! 1. The windowed attention (`vis_attn_group`): 16 heads of 72 over fused `qkv` rows of 3456,
//!    swept over `n ∈ {4, 64, 1024}` rows in windows of four (and windows of 2 and 3, which the
//!    kernel takes as its `group` argument): every output value inside the f32 bound of
//!    `attn_bound` around the exact attention of its own window alone, the count of values over
//!    it pinned at [`ATTN_OUTSIDE_PIN`]; the canary words past the output untouched.
//! 2. The group mean (`vis_mean_groups`): f32 rows of 1152 over the same sweep, equal to
//!    `mean_groups_ref` bit for bit, and a control that the inputs exercise the order — the count
//!    of values another association of the same sum rounds apart must be above zero, or a gate
//!    on the order would pass on any order.
//! 3. A row count that is not a whole number of windows, a window of 0 rows and an empty image
//!    are refused by name by both entries, and launch nothing.

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!(
        "gate_minicpmv_tower: built without the `vision` feature; see `just gate-gpu-minicpmv-tower`."
    );
    std::process::exit(2);
}

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minicpmv_tower", gate::run())
}

#[cfg(feature = "vision")]
mod gate {
    use bloomery_gpu::GpuError;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use bloomery_gpu_vision::attn::{HEAD_DIM_72, QkvLayout, attn_bound};
    use bloomery_gpu_vision::attn_group::{AttnGroupArgs, AttnGroupKernels, attn_group_ref};
    use bloomery_gpu_vision::f32_bf16;
    use bloomery_gpu_vision::gemm_bf16::within_order;
    use bloomery_gpu_vision::mean_groups::{MeanGroupsKernels, mean_groups_ref};
    use cuda_core::{CudaStream, DeviceBuffer};

    const NAME: &str = "gate_minicpmv_tower";
    /// Heads of the merger's attention, and the width of a row they make.
    const HEADS: usize = 16;
    const DIM: usize = HEADS * HEAD_DIM_72;
    /// Width of a fused `qkv` row.
    const QKV_WIDTH: usize = 3 * DIM;
    /// `(rows, window)` of every sweep point: the tower's windows of four at three sizes, then
    /// windows the kernels take as an argument.
    const CASES: [(usize, usize); 5] = [(4, 4), (64, 4), (1024, 4), (64, 2), (66, 3)];
    /// Words past the end of an output buffer that the kernels must leave as they were.
    const CANARY: usize = 1024;
    /// A bf16 NaN: what the output holds before the launch, so a value never written is outside
    /// every bound.
    const NAN_BF16: u16 = 0x7fc0;
    /// Output values of the attention sweep that may lie outside the f32 bound.
    /// PIN(2026-10-13): the bound is derived (`attn_bound`), the kernel has no tensor-core
    /// split, so no value may lie outside it.
    const ATTN_OUTSIDE_PIN: usize = 0;

    /// A fixed xorshift64 stream.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A value in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    // ------------------------------------------------------------ the windowed attention

    /// One sweep point of the attention: `n` rows in windows of `group`.
    fn attn_clause(
        stream: &CudaStream,
        k: &AttnGroupKernels,
        n: usize,
        group: usize,
    ) -> Result<bool, GateError> {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ((n as u64) << 8) ^ group as u64);
        // Uniform in ±2.5: the scores of a row spread over a few units, so a window's weights
        // are neither uniform nor one-hot.
        let qkv: Vec<u16> = (0..n * QKV_WIDTH)
            .map(|_| f32_bf16(rng.unit() * 5.0 - 2.5))
            .collect();
        let lay = QkvLayout {
            row_width: QKV_WIDTH,
            q0: 0,
            k0: DIM,
            v0: 2 * DIM,
            n_heads: HEADS,
        };
        let d_qkv = DeviceBuffer::from_host(stream, &qkv)?;
        let mut d_out = DeviceBuffer::from_host(stream, &vec![NAN_BF16; n * DIM + CANARY])?;
        k.enqueue(
            stream,
            AttnGroupArgs {
                qkv: &d_qkv,
                layout: lay,
                n,
                group,
                scale: 1.0 / (HEAD_DIM_72 as f32).sqrt(),
                out_width: DIM,
                out: &mut d_out,
            },
        )?;
        stream.synchronize()?;
        let got = d_out.to_host_vec(stream)?;
        let scale = 1.0 / (HEAD_DIM_72 as f64).sqrt();
        let (mut outside, mut differ) = (0usize, 0usize);
        for i in 0..n {
            for h in 0..HEADS {
                let (o, mag, logit) = attn_group_ref(&qkv, lay, group, scale, h, i);
                for d in 0..HEAD_DIM_72 {
                    let y = got[i * DIM + h * HEAD_DIM_72 + d];
                    let b = attn_bound(o[d], mag[d], logit, group);
                    outside += usize::from(!within_order(y, o[d], b));
                    differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
        }
        let canary = got[n * DIM..].iter().filter(|&&y| y != NAN_BF16).count();
        let pass = outside == ATTN_OUTSIDE_PIN && canary == 0;
        println!(
            "rule attn_group n {n:<4} window {group}: outside the f32 bound {outside} of {} (pin {ATTN_OUTSIDE_PIN}), differ from the exact rounding {differ} ({:.5}), canary words changed {canary}  {}",
            n * DIM,
            differ as f64 / (n * DIM) as f64,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ the group mean

    /// Another association of a window's sum, the control that the inputs exercise the order: the
    /// pairs `(p0 + p1) + (p2 + p3)` for a window of four, the right fold for a window of three.
    /// None for a window whose every order is the same sum.
    fn other_order(rows: &[f32], width: usize, group: usize, d: usize) -> Option<f32> {
        let p = |j: usize| rows[j * width + d];
        match group {
            4 => Some(((p(0) + p(1)) + (p(2) + p(3))) * 0.25),
            3 => Some((p(0) + (p(1) + p(2))) * (1.0f32 / 3.0)),
            _ => None,
        }
    }

    /// One sweep point of the mean: `n` rows in windows of `group`.
    fn mean_clause(
        stream: &CudaStream,
        k: &MeanGroupsKernels,
        n: usize,
        group: usize,
    ) -> Result<bool, GateError> {
        let mut rng = Rng(0xd1b5_4a32_d192_ed03 ^ ((n as u64) << 8) ^ group as u64);
        // Signed values over sixteen binades, so the partial sums round at different places
        // under different associations.
        let src: Vec<f32> = (0..n * DIM)
            .map(|_| {
                let e = (rng.unit() * 16.0) as i32 - 8;
                (rng.unit() - 0.5) * (e as f32).exp2()
            })
            .collect();
        let n_out = n / group;
        let d_src = DeviceBuffer::from_host(stream, &src)?;
        let mut d_out = DeviceBuffer::from_host(stream, &vec![f32::NAN; n_out * DIM + CANARY])?;
        k.enqueue(stream, n, group, DIM, &d_src, &mut d_out)?;
        stream.synchronize()?;
        let got = d_out.to_host_vec(stream)?;
        let want = mean_groups_ref(&src, DIM, group);
        let differ = got[..n_out * DIM]
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let canary = got[n_out * DIM..].iter().filter(|v| !v.is_nan()).count();
        let pass = differ == 0 && canary == 0;
        println!(
            "rule mean_groups n {n:<4} window {group}: bit-exact: {differ} of {} differ, canary words changed {canary}  {}",
            n_out * DIM,
            verdict(pass)
        );
        // The control: another association of the same sums must round apart somewhere.
        let apart = src
            .chunks_exact(group * DIM)
            .flat_map(|g| (0..DIM).map(move |d| (g, d)))
            .zip(&want)
            .filter(|((g, d), w)| {
                other_order(g, DIM, group, *d).is_some_and(|o| o.to_bits() != w.to_bits())
            })
            .count();
        let applies = other_order(&src, DIM, group, 0).is_some();
        let control = !applies || apart > 0;
        println!(
            "control mean_groups n {n:<4} window {group}: another association rounds apart in {apart} of {} values{}  {}",
            n_out * DIM,
            if applies {
                ""
            } else {
                " (every order is the same sum: not applicable)"
            },
            verdict(control)
        );
        Ok(pass && control)
    }

    // ------------------------------------------------------------ the refusals

    /// `r` is the named refusal of `what` for a window that does not divide the rows.
    fn refused(name: &str, what: &str, r: Result<(), GpuError>) -> bool {
        let pass = matches!(
            &r,
            Err(GpuError::Shape { what: w, detail }) if *w == what && detail.contains("n % group must be 0")
        );
        println!(
            "refusal {name:<28} {}  {}",
            match &r {
                Ok(()) => "launched".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(pass)
        );
        pass
    }

    /// Both entries refuse `n % group != 0`, a window of 0 and an empty image, before any launch;
    /// the stream then runs clean.
    fn refusals(
        stream: &CudaStream,
        a: &AttnGroupKernels,
        m: &MeanGroupsKernels,
    ) -> Result<bool, GateError> {
        let lay = QkvLayout {
            row_width: QKV_WIDTH,
            q0: 0,
            k0: DIM,
            v0: 2 * DIM,
            n_heads: HEADS,
        };
        // Buffers big enough for any of the shapes below, so only the window refuses.
        let rows = 8;
        let d_qkv = DeviceBuffer::<u16>::zeroed(stream, rows * QKV_WIDTH)?;
        let mut d_out = DeviceBuffer::<u16>::zeroed(stream, rows * DIM)?;
        let d_src = DeviceBuffer::<f32>::zeroed(stream, rows * DIM)?;
        let mut d_mean = DeviceBuffer::<f32>::zeroed(stream, rows * DIM)?;
        let mut ok = true;
        for (n, group) in [(6usize, 4usize), (4, 0), (0, 4)] {
            let r = a.enqueue(
                stream,
                AttnGroupArgs {
                    qkv: &d_qkv,
                    layout: lay,
                    n,
                    group,
                    scale: 1.0,
                    out_width: DIM,
                    out: &mut d_out,
                },
            );
            ok &= refused(
                &format!("attn_group n {n} window {group}"),
                "AttnGroupKernels::enqueue",
                r,
            );
            let r = m.enqueue(stream, n, group, DIM, &d_src, &mut d_mean);
            ok &= refused(
                &format!("mean_groups n {n} window {group}"),
                "MeanGroupsKernels::enqueue",
                r,
            );
        }
        stream.synchronize()?;
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(0)?;
        let attn = AttnGroupKernels::load(&ctx)?;
        let mean = MeanGroupsKernels::load(&ctx)?;
        let mut ok = refusals(&stream, &attn, &mean)?;
        for (n, group) in CASES {
            ok &= attn_clause(&stream, &attn, n, group)?;
        }
        for (n, group) in CASES {
            ok &= mean_clause(&stream, &mean, n, group)?;
        }
        if ok {
            println!("{NAME}: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
