//! Clause 10's Qwen3.8 combine launches beside `gate_kquant`'s card leg:
//! the one launch of a layer without a host leg (the common `card_sum_join`,
//! on F32 and on BF16 down rows, gated by the router's last weight from the
//! host list's zeros), the same launch ungated from the card sum itself
//! (GLM-5.3's shared expert), their checks, and the refusals of both combine
//! launchers.

use bloomery_gpu::card_slots::{CardJoin, DownRows, SharedJoin};
use bloomery_gpu::hybrid::HOST;
use bloomery_gpu::q38::{CardSharedAddArgs, SLOTS, W_PITCH};
use bloomery_gpu::{Fault, FaultSite, GpuError, LAYER_NONE};
use bloomery_gpu_gates::{GateError, bits_equal, verdict};
use cuda_core::DeviceBuffer;
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};

use crate::gate::{Ctx, E38, Leg38, M38, N38, SEL38, SEL38_STRAY, SENT, mismatch, shown};

/// Token 0's value of clause 10's `-0.0` case.
const NEG0: usize = 3;

/// One row of the combine refusals: `(launcher, case, result)`.
type CombineRefusal = (&'static str, &'static str, Result<(), GpuError>);

impl Leg38 {
    /// The common join on `down` and the leg's weights and shared rows, `m`
    /// tokens placed by `sel`, into `y`: Qwen3.8's (`shared` [`None`]: the
    /// router's last weight from the host list's zeros) or `shared`'s.
    fn join(
        &self,
        c: &Ctx,
        (down, sel, m): (DownRows<'_>, &DeviceBuffer<u32>, usize),
        shared: Option<SharedJoin<'_>>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let a = CardJoin {
            down,
            w: &self.w,
            sel,
            n: N38,
            m,
            n_card: E38,
            n_used: SLOTS,
            w_pitch: W_PITCH,
            shared: shared.unwrap_or(SharedJoin {
                sh: &self.sh,
                gate: Some(SLOTS),
            }),
            fault: c.gpu.unlabelled_sink(),
        };
        self.slots.enqueue_card_join(c.gpu.stream(), &a, y)
    }

    /// A layer without a host leg's one launch, Qwen3.8's join.
    fn sum_hostless(
        &self,
        c: &Ctx,
        down: DownRows<'_>,
        sel: &DeviceBuffer<u32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.join(c, (down, sel, m), None, y)
    }

    /// Clause 10's block output of a layer without a host leg over the leg's
    /// down rows, whose host slots `check_q38_card_places` left NaN, with
    /// token 0's value [`NEG0`] on every card slot the smallest negative
    /// subnormal (each product rounds to `-0.0`, so the card sum is `-0.0`)
    /// and its shared row `-0.0`: the runtime's combine over a zero host sum
    /// and the leg's two launches over zeros, bit for bit, that value `+0.0`,
    /// no fault; under [`SEL38_STRAY`] token 0 the same, token 1 NaN,
    /// `ExpertId` and the NaN sum's `F32Product` raised.
    pub(crate) fn check_hostless(&mut self, c: &Ctx) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let mut down = self.y_down.to_host_vec(stream)?;
        for j in (0..SLOTS).filter(|&j| SEL38[j] != HOST) {
            down[j * N38 + NEG0] = -f32::from_bits(1);
        }
        self.y_down.copy_from_host(stream, &down)?;
        let mut sh = self.sh.to_host_vec(stream)?;
        sh[NEG0] = -0.0;
        self.sh.copy_from_host(stream, &sh)?;
        self.hsum.copy_from_host(stream, &vec![0.0f32; M38 * N38])?;
        let (_, rule) = self.host_rule(c, &SEL38)?;
        self.sum(c, stream)?;
        stream.synchronize()?;
        let two = self.y.to_host_vec(stream)?;
        let fused = |leg: &Leg38, sel: &DeviceBuffer<u32>| -> Result<_, GateError> {
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?;
            leg.sum_hostless(c, DownRows::F32(&leg.y_down), sel, M38, &mut y)?;
            stream.synchronize()?;
            Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
        };
        let (y, fault) = fused(self, &DeviceBuffer::from_host(stream, &SEL38)?)?;
        let (is_rule, is_two) = (bits_equal(&y, &rule), bits_equal(&y, &two));
        let neg0 = (rule[NEG0].to_bits(), y[NEG0].to_bits());
        let pass = is_rule && is_two && neg0 == (0, 0) && fault.is_none();
        let mut ok = pass;
        println!(
            "q38_card_hostless[{M38} tokens x {SLOTS} slots, host slots NaN] \
             is_runtime_combine_over_zero_hsum={is_rule} is_two_launches_over_zeros={is_two} \
             neg_zero_sum_bits={:#010x} (want 0) fault=\"{}\" {}{}",
            neg0.1,
            shown(fault),
            verdict(pass),
            mismatch("first_mismatch", &y, &rule, N38),
        );
        let stray = DeviceBuffer::from_host(stream, &SEL38_STRAY)?;
        let (y, fault) = fused(self, &stray)?;
        let want = Some(Fault::of_sites(
            LAYER_NONE,
            &[FaultSite::ExpertId, FaultSite::F32Product],
        ));
        let token0 = bits_equal(&y[..N38], &rule[..N38]);
        let token1 = y[N38..].iter().all(|v| v.is_nan());
        let pass = token0 && token1 && fault == want;
        ok &= pass;
        println!(
            "q38_card_hostless[stray_place] token0_is_the_rule={token0} token1_all_nan={token1} \
             fault=\"{}\" want=\"{}\" {}",
            shown(fault),
            shown(want),
            verdict(pass)
        );
        // BF16 down rows: the F32 launch over the same rows widened, bit for
        // bit, its faults included.
        let bf16: Vec<u16> = down.iter().map(|&v| f32_to_bf16_rne(v)).collect();
        let widened: Vec<f32> = bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let bf16 = DeviceBuffer::from_host(stream, &bf16)?;
        let widened = DeviceBuffer::from_host(stream, &widened)?;
        for (name, places) in [("placed", &SEL38), ("stray_place", &SEL38_STRAY)] {
            let sel = DeviceBuffer::from_host(stream, places)?;
            let run = |rows: DownRows<'_>| -> Result<_, GateError> {
                let mut y = DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?;
                self.sum_hostless(c, rows, &sel, M38, &mut y)?;
                stream.synchronize()?;
                Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
            };
            let (got, got_fault) = run(DownRows::Bf16(&bf16))?;
            let (want, want_fault) = run(DownRows::F32(&widened))?;
            let same = bits_equal(&got, &want);
            let pass = same && got_fault == want_fault;
            ok &= pass;
            println!(
                "q38_card_hostless[bf16_rows, {name}] is_f32_launch_over_widened_rows={same} \
                 fault=\"{}\" want=\"{}\" {}{}",
                shown(got_fault),
                shown(want_fault),
                verdict(pass),
                mismatch("first_mismatch", &got, &want, N38),
            );
        }
        // Ungated from the card sum itself: the card sum launch, then `acc +
        // sh` (GLM-5.3's two launches), bit for bit, the `-0.0` sum kept.
        self.sum(c, stream)?;
        stream.synchronize()?;
        let acc = self.acc.to_host_vec(stream)?;
        let want: Vec<f32> = acc.iter().zip(&sh).map(|(a, s)| a + s).collect();
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?;
        let ungated = SharedJoin {
            sh: &self.sh,
            gate: None,
        };
        let sel = DeviceBuffer::from_host(stream, &SEL38)?;
        self.join(
            c,
            (DownRows::F32(&self.y_down), &sel, M38),
            Some(ungated),
            &mut y,
        )?;
        stream.synchronize()?;
        let (y, fault) = (y.to_host_vec(stream)?, c.gpu.take_fault()?);
        let same = bits_equal(&y, &want);
        let neg0 = y[NEG0].to_bits();
        let pass = same && neg0 == (-0.0f32).to_bits() && fault.is_none();
        ok &= pass;
        println!(
            "card_join[ungated, from the card sum] is_card_acc_plus_shared={same} \
             neg_zero_sum_bits={neg0:#010x} (want 0x80000000) fault=\"{}\" {}{}",
            shown(fault),
            verdict(pass),
            mismatch("first_mismatch", &y, &want, N38),
        );
        Ok(ok)
    }

    /// The combine launchers' rows of clause 10's refusals, each `(launcher,
    /// case, result)`: a combine slot past the weights, and a layer without
    /// a host leg with a shared gate among the routed slots or past the
    /// weights' pitch.
    pub(crate) fn combine_refusals(&mut self, c: &Ctx) -> Result<[CombineRefusal; 3], GateError> {
        // PIN(2026-11-02): no_token and short output are covered by
        // card_slots::tests::hw_the_card_sum_launchers_refuse_invalid_shapes.
        let stream = c.gpu.stream();
        let r_slot = self.q38.enqueue_card_shared_add(
            stream,
            CardSharedAddArgs {
                hsum: &self.hsum,
                acc: &self.acc,
                sh: &self.sh,
                w: &self.w,
                slot: W_PITCH,
                slots: W_PITCH,
                n: N38,
                m: M38,
                fault: c.gpu.unlabelled_sink(),
                y: &mut self.y,
            },
        );
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?;
        let rows = DownRows::F32(&self.y_down);
        let mut gated = |gate| {
            let shared = SharedJoin {
                sh: &self.sh,
                gate: Some(gate),
            };
            self.join(c, (rows, &self.sel, M38), Some(shared), &mut y)
        };
        let (r_routed_gate, r_gate_past) = (gated(SLOTS - 1), gated(W_PITCH));
        let what = "card_sum_join";
        Ok([
            ("q38::enqueue_card_shared_add", "slot_11_of_11", r_slot),
            (what, "a_gate_among_the_routed_slots", r_routed_gate),
            (what, "a_gate_past_the_pitch", r_gate_past),
        ])
    }
}
