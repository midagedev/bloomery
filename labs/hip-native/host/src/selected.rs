//! Selection and numerical contracts against the engine's existing scalar card-rule oracle.
use super::{Context, c_void};

const HOST: u32 = u32::MAX;
const FAULT_WORDS: usize = 65_537;
const SENTINEL: f32 = 1.0e30;

struct Plan {
    k: usize,
    rows: usize,
    experts: usize,
    slots: usize,
    cols: usize,
}

impl Plan {
    fn validate(
        &self,
        weights: usize,
        q: usize,
        sums: usize,
        scales: usize,
        ids: usize,
        out: usize,
    ) -> Result<(), String> {
        if !(256..=20_480).contains(&self.k)
            || !self.k.is_multiple_of(256)
            || self.rows == 0
            || self.experts == 0
            || self.slots == 0
            || self.cols < self.slots
        {
            return Err("q4_selected: invalid shape or fewer activation columns than slots".into());
        }
        for dim in [self.rows, self.experts, self.slots, self.cols] {
            u32::try_from(dim).map_err(|_| "q4_selected: dimension exceeds u32")?;
        }
        let n_sb = self.k / 256;
        let product = |a: usize, b: usize| a.checked_mul(b).ok_or("q4_selected: size overflow");
        let expected_weights = product(product(self.rows, self.experts)?, product(n_sb, 144)?)?;
        if weights != expected_weights
            || q < product(self.cols, 256 * n_sb.div_ceil(4))?
            || sums < product(self.cols, 8 * n_sb)?
            || scales < product(self.cols, 2 * n_sb)?
            || ids < self.slots
            || out < product(self.slots, self.rows)?
        {
            return Err("q4_selected: weight/plane/id/output span mismatch".into());
        }
        Ok(())
    }
}

struct Planes {
    q: Vec<u32>,
    sums: Vec<i32>,
    scales: Vec<f32>,
    oracle: Vec<qdot::CardQ81>,
}

fn planes(values: &[f32], k: usize) -> Planes {
    let n_sb = k / 256;
    let width = 256 * n_sb.div_ceil(4);
    let oracle: Vec<_> = values.chunks_exact(k).map(qdot::card_q8_1_scalar).collect();
    // Padding remains poisoned: the partial final super-block group must never read it.
    let mut q = vec![0xa5a5_a5a5; oracle.len() * width];
    let mut sums = Vec::new();
    let mut scales = Vec::new();
    for (col, act) in oracle.iter().enumerate() {
        for (v4, quad) in act.codes().as_chunks::<4>().0.iter().enumerate() {
            let slot = 256 * (v4 >> 8) + 32 * (v4 & 7) + 8 * ((v4 >> 6) & 3) + ((v4 >> 3) & 7);
            q[col * width + slot] = u32::from_le_bytes(quad.map(|v| v as u8));
        }
        sums.extend_from_slice(act.s8());
        scales.extend_from_slice(act.d8());
    }
    Planes {
        q,
        sums,
        scales,
        oracle,
    }
}

fn upload<T: Copy>(context: &mut Context, values: &[T]) -> Result<*mut c_void, String> {
    let ptr = context.alloc(std::mem::size_of_val(values))?;
    overwrite(context, ptr, values)?;
    Ok(ptr)
}

fn overwrite<T: Copy>(context: &Context, ptr: *mut c_void, values: &[T]) -> Result<(), String> {
    // SAFETY: each caller passes the allocation made for this exact-length typed vector; copy synchronizes.
    unsafe {
        context.copy(
            ptr,
            values.as_ptr().cast(),
            std::mem::size_of_val(values),
            1,
        )
    }
}

fn read<T: Copy>(context: &Context, ptr: *mut c_void, values: &mut [T]) -> Result<(), String> {
    // SAFETY: each caller passes the allocation made for this exact-length typed vector; copy synchronizes.
    unsafe {
        context.copy(
            values.as_mut_ptr().cast(),
            ptr,
            std::mem::size_of_val(values),
            2,
        )
    }
}

fn wave_size(context: &mut Context) -> Result<(), String> {
    let mut word = [0u32];
    let mut ptr = upload(context, &word)?;
    let f = context.function("bloomery_hip_wave_size")?;
    context.launch_shape(f, 1, 1, &mut [(&mut ptr as *mut *mut c_void).cast()])?;
    read(context, ptr, &mut word)?;
    if word != [32] {
        return Err(format!(
            "q4_selected: requires compiled wave32, got {}",
            word[0]
        ));
    }
    Ok(())
}

struct Device {
    w: *mut c_void,
    q: *mut c_void,
    sums: *mut c_void,
    scales: *mut c_void,
    ids: *mut c_void,
    fault: *mut c_void,
    out: *mut c_void,
    function: *mut c_void,
}

impl Device {
    fn launch(&mut self, context: &Context, plan: &Plan, layer: u32) -> Result<(), String> {
        if layer > 65_535 {
            return Err("q4_selected: layer exceeds fault allocation".into());
        }
        let (mut experts, mut rows, mut slots, mut n_sb, mut iters, mut layer) = (
            plan.experts as u32,
            plan.rows as u32,
            plan.slots as u32,
            (plan.k / 256) as u32,
            (plan.k / 256).div_ceil(4) as u32,
            layer,
        );
        let mut args = [
            (&mut self.w as *mut *mut c_void).cast(),
            (&mut self.q as *mut *mut c_void).cast(),
            (&mut self.sums as *mut *mut c_void).cast(),
            (&mut self.scales as *mut *mut c_void).cast(),
            (&mut self.ids as *mut *mut c_void).cast(),
            (&mut experts as *mut u32).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut slots as *mut u32).cast(),
            (&mut n_sb as *mut u32).cast(),
            (&mut iters as *mut u32).cast(),
            (&mut self.fault as *mut *mut c_void).cast(),
            (&mut layer as *mut u32).cast(),
            (&mut self.out as *mut *mut c_void).cast(),
        ];
        context.launch_shape(
            self.function,
            (plan.slots * plan.rows).div_ceil(8),
            256,
            &mut args,
        )
    }
}

fn check_output(
    plan: &Plan,
    weights: &[u8],
    acts: &Planes,
    ids: &[u32],
    got: &[f32],
) -> Result<(), String> {
    let row_bytes = plan.k / 256 * 144;
    for (slot, &id) in ids.iter().enumerate() {
        for row in 0..plan.rows {
            let got = got[slot * plan.rows + row];
            if id as usize >= plan.experts {
                if got.to_bits() != SENTINEL.to_bits() {
                    return Err(format!(
                        "q4_selected: skipped slot {slot} row {row} changed sentinel"
                    ));
                }
                continue;
            }
            let start = (id as usize * plan.rows + row) * row_bytes;
            let expected = qdot::card_q4k_dot_row_scalar(
                &weights[start..start + row_bytes],
                &acts.oracle[slot],
            )
            .map_err(|e| e.to_string())?;
            if acts.oracle[slot].refused_any() {
                if !expected.is_nan() || !got.is_nan() {
                    return Err(format!(
                        "q4_selected: refused activation slot {slot} row {row} did not propagate NaN"
                    ));
                }
            } else if !got.is_finite()
                || !expected.is_finite()
                || got.to_bits() != expected.to_bits()
            {
                return Err(format!(
                    "q4_selected: K={} slot={slot} row={row} got={got} ({:08x}) expected={expected} ({:08x})",
                    plan.k,
                    got.to_bits(),
                    expected.to_bits()
                ));
            }
        }
    }
    Ok(())
}

fn case(
    context: &mut Context,
    weights: &[u8],
    k: usize,
    rows: usize,
    experts: usize,
    real: bool,
) -> Result<(), String> {
    let plan = Plan {
        k,
        rows,
        experts,
        slots: 6,
        cols: 8,
    };
    let mut values: Vec<f32> = (0..k * plan.cols)
        .map(|i| ((i as f32 * 0.019).sin() - 0.31) * ((i / k + 1) as f32 * 0.13))
        .collect();
    // Extra columns are legal but must not be read by any selected slot.
    values[plan.slots * k..].fill(f32::NAN);
    let ordinary = planes(&values, k);
    let ids_a = [0, 2, 3, 2, 1, 0];
    let ids_b = [3, 0, 1, 1, 2, 3];
    let ids_bad = [2, HOST, experts as u32 + 7, 0, 1, 3];
    let mut out = vec![SENTINEL; plan.slots * rows];
    plan.validate(
        weights.len(),
        ordinary.q.len(),
        ordinary.sums.len(),
        ordinary.scales.len(),
        ids_a.len(),
        out.len(),
    )?;
    let mut fault = vec![0u32; FAULT_WORDS];
    fault[0] = u32::MAX;
    let mut device = Device {
        w: upload(context, weights)?,
        q: upload(context, &ordinary.q)?,
        sums: upload(context, &ordinary.sums)?,
        scales: upload(context, &ordinary.scales)?,
        ids: upload(context, &ids_a)?,
        fault: upload(context, &fault)?,
        out: upload(context, &out)?,
        function: context.function("bloomery_hip_q4_sel")?,
    };
    let allocated = context.buffers.len();
    let mut runs = 0;
    for ids in [&ids_a, &ids_a, &ids_a, &ids_b] {
        out.fill(SENTINEL);
        overwrite(context, device.ids, ids)?;
        overwrite(context, device.out, &out)?;
        device.launch(context, &plan, 9)?;
        read(context, device.out, &mut out)?;
        check_output(&plan, weights, &ordinary, ids, &out)?;
        runs += 1;
    }
    read(context, device.fault, &mut fault)?;
    if fault[0] != u32::MAX || fault[1..].iter().any(|&v| v != 0) {
        return Err("q4_selected: valid selection raised a fault".into());
    }
    out.fill(SENTINEL);
    overwrite(context, device.out, &out)?;
    overwrite(context, device.ids, &ids_bad)?;
    for layer in [9, 3, 9] {
        device.launch(context, &plan, layer)?;
        runs += 1;
    }
    read(context, device.out, &mut out)?;
    check_output(&plan, weights, &ordinary, &ids_bad, &out)?;
    read(context, device.fault, &mut fault)?;
    if fault[0] != (3 << 8) | 7
        || fault[4] != 1 << 7
        || fault[10] != 1 << 7
        || fault[1..]
            .iter()
            .enumerate()
            .any(|(i, &v)| i != 3 && i != 9 && v != 0)
    {
        return Err(
            "q4_selected: sticky earliest-layer fault word or per-layer site masks differ".into(),
        );
    }
    // Isolate HOST from invalid ids: it must not raise a fault of its own.
    let ids_host = [2, HOST, 3, 0, 1, 3];
    fault.fill(0);
    fault[0] = u32::MAX;
    overwrite(context, device.fault, &fault)?;
    overwrite(context, device.ids, &ids_host)?;
    out.fill(SENTINEL);
    overwrite(context, device.out, &out)?;
    device.launch(context, &plan, 9)?;
    runs += 1;
    read(context, device.out, &mut out)?;
    check_output(&plan, weights, &ordinary, &ids_host, &out)?;
    // A NaN input is refused by the HOST oracle/quantizer here. This entry propagates the
    // NaN scale; it does not pretend to have run the GPU quantizer or to raise its fault.
    values[0] = f32::NAN;
    values[k..2 * k].fill(0.0);
    for (i, value) in values[2 * k..3 * k].iter_mut().enumerate() {
        *value =
            f32::from_bits(100_000 + (i % 17) as u32 * 37) * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let refused = planes(&values, k);
    overwrite(context, device.q, &refused.q)?;
    overwrite(context, device.sums, &refused.sums)?;
    overwrite(context, device.scales, &refused.scales)?;
    overwrite(context, device.ids, &ids_a)?;
    out.fill(SENTINEL);
    overwrite(context, device.out, &out)?;
    device.launch(context, &plan, 9)?;
    runs += 1;
    read(context, device.out, &mut out)?;
    check_output(&plan, weights, &refused, &ids_a, &out)?;
    read(context, device.fault, &mut fault)?;
    if fault[0] != u32::MAX || fault[1..].iter().any(|&v| v != 0) {
        return Err("q4_selected: HOST or NaN scale raised an unexpected expert-id fault".into());
    }
    if context.buffers.len() != allocated {
        return Err("q4_selected: steady launch allocated".into());
    }
    println!(
        "{{\"operation\":\"q4_selected\",\"contract\":\"selected_bitwise\",\"k\":{k},\"rows_per_expert\":{rows},\"experts\":{experts},\"slots\":6,\"activation_cols\":8,\"launches\":{runs},\"real_weights\":{real},\"wave_size\":32,\"bitwise_equal\":true,\"finite_expected_outputs\":true,\"nan_propagation\":true,\"fault_and_sentinel\":true,\"steady_host_buffer_allocations\":0}}"
    );
    Ok(())
}

pub(super) fn synthetic(context: &mut Context) -> Result<(), String> {
    wave_size(context)?;
    for (i, n_sb) in [1, 2, 3, 4, 5, 7, 8, 9, 11, 24, 80].into_iter().enumerate() {
        let k = n_sb * 256;
        let rows = 7 + i % 3;
        case(context, &super::fixture(k, 4 * rows), k, rows, 4, false)?;
    }
    Ok(())
}

pub(super) fn real(
    context: &mut Context,
    weights: &[u8],
    k: usize,
    total_rows: usize,
) -> Result<(), String> {
    if total_rows < 4 || !total_rows.is_multiple_of(4) {
        return Err("q4_selected: real fixture needs a positive multiple of four rows".into());
    }
    wave_size(context)?;
    case(context, weights, k, total_rows / 4, 4, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_shape_refusals_precede_the_driver() {
        let mut plan = Plan {
            k: 256,
            rows: 7,
            experts: 4,
            slots: 6,
            cols: 8,
        };
        let weights = 144 * 7 * 4;
        assert!(plan.validate(weights, 8 * 256, 8 * 8, 8 * 2, 6, 42).is_ok());
        for lengths in [
            (weights - 1, 2048, 64, 16, 6, 42),
            (weights, 2047, 64, 16, 6, 42),
            (weights, 2048, 63, 16, 6, 42),
            (weights, 2048, 64, 15, 6, 42),
            (weights, 2048, 64, 16, 5, 42),
            (weights, 2048, 64, 16, 6, 41),
        ] {
            assert!(
                plan.validate(
                    lengths.0, lengths.1, lengths.2, lengths.3, lengths.4, lengths.5
                )
                .is_err()
            );
        }
        plan.cols = 5;
        assert!(plan.validate(weights, 2048, 64, 16, 6, 42).is_err());
        plan.cols = 8;
        plan.rows = 0;
        assert!(plan.validate(weights, 2048, 64, 16, 6, 42).is_err());
        plan.rows = usize::MAX;
        assert!(plan.validate(weights, 2048, 64, 16, 6, 42).is_err());
    }
}
