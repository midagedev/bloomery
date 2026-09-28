//! Int8 tensor-core GEMM probe: the four attention projections of one V4.1
//! layer (the joint qkv, q_b, the grouped wo_a and wo_b) as the grouped GEMM
//! (`bloomery_gpu::gemm`) at `m` tokens — the B4 question of a prompt
//! sub-block's projections run on the tensor cores instead of the m-column
//! gemv. The weights are the file's own bytes; the roles are joined as
//! `chain/attn.rs` joins its launches (q_a + kv [+ indexer.proj], q_b
//! [+ indexer.attn_q_b]: the parts after the first that share its K), as
//! `tools/ref/ds41pp.py role_shape` reads them. wo_a is block diagonal: one
//! expert per output group (`attention.output_group_count`), each slot
//! (token, group) reading its own column (`GemmInput::PerSlot`).
//!
//! `probe_imma --check [--layer L] [--m M]`: every role once at `M` tokens
//! over a fixed pattern of activations; six outputs a role (three rows, the
//! first and last columns) against an f64 dot of the same weight bytes
//! (`gguf::dequant_row`) with the f32 activations, within the int8
//! activation rounding's bound: half a code of each 128-value block's scale
//! times the block's summed |w|, plus 1 % of it. Red on any.
//!
//! `probe_imma --time [--layer L] [--m M] [--batch B]` (under
//! `tools/ref/time-gate.sh`: the lease, the A6000, the witness): the check,
//! then per role the launch's graph replay (a graph of 64 launches, seven
//! rounds, `shared/bench.rs`) and one `probe imma arm` record; then `probe imma
//! total`: the roles' minimums summed for one sub-block, and times `B / M`
//! for a `B`-token layer-batch. The activation quantization and the route
//! table are outside the timed launch. Defaults: layer 3, M 128, B 512.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("probe_imma: built without the `gpu` feature; see `just probe-gpu-imma`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("probe_imma", probe::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/bench.rs"]
#[allow(dead_code)] // print_arm and fill_pattern are the other bins'
mod bench_arm;

#[cfg(feature = "gpu")]
mod probe {
    use crate::bench_arm::{N, fill_pattern_f32, replay};
    use bloomery_gpu::gemm::{GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmWeight};
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, checks_failed};
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::GgmlType;
    use gguf::{Split, dequant_row};

    /// The roles, their file tensors (the first required, the rest joined
    /// when they share its K), and whether the role is block diagonal.
    const ROLES: [(&str, &[&str], bool); 4] = [
        ("qkv", &["attn_q_a", "attn_kv", "indexer.proj"], false),
        ("q_b", &["attn_q_b", "indexer.attn_q_b"], false),
        ("wo_a", &["attn_output_a"], true),
        ("wo_b", &["attn_output_b"], false),
    ];

    struct Args {
        time: bool,
        layer: usize,
        m: usize,
        batch: usize,
    }

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            time: false,
            layer: 3,
            m: 128,
            batch: 512,
        };
        let mut mode = None;
        let mut it = std::env::args().skip(1);
        while let Some(x) = it.next() {
            match x.as_str() {
                "--check" | "--time" => mode = Some(x == "--time"),
                "--layer" | "--m" | "--batch" => {
                    let v: usize = it
                        .next()
                        .ok_or_else(|| format!("{x} wants a value"))?
                        .parse()
                        .map_err(|e| format!("{x}: {e}"))?;
                    match x.as_str() {
                        "--layer" => a.layer = v,
                        "--m" => a.m = v,
                        _ => a.batch = v,
                    }
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }
        a.time = mode.ok_or("want one of --check, --time")?;
        if a.m == 0 || !a.batch.is_multiple_of(a.m) {
            return Err(format!(
                "--m {} must be positive and divide --batch {}",
                a.m, a.batch
            )
            .into());
        }
        Ok(a)
    }

    /// One role's stack: the joined weight bytes, its type, K and rows, and
    /// its groups (1 for a dense role).
    struct Stack {
        name: &'static str,
        ty: GgmlType,
        k: usize,
        rows: usize,
        groups: usize,
        bytes: Vec<u8>,
    }

    impl Stack {
        fn row(&self, r: usize) -> &[u8] {
            let rb = self.bytes.len() / self.rows;
            &self.bytes[r * rb..(r + 1) * rb]
        }
    }

    fn stack(
        split: &Split,
        layer: usize,
        name: &'static str,
        parts: &[&str],
        grouped: bool,
    ) -> Result<Stack, GateError> {
        let mut out: Option<Stack> = None;
        for (i, p) in parts.iter().enumerate() {
            let tname = format!("blk.{layer}.{p}.weight");
            let Some((shard, t)) = split.find(&tname) else {
                if i == 0 {
                    return Err(format!("{name}: the file has no {tname}").into());
                }
                continue;
            };
            let (k, rows) = match t.dims.as_slice() {
                [k, r] => (*k as usize, *r as usize),
                d => return Err(format!("{tname} has dims {d:?}, not two").into()),
            };
            let data = split
                .shard(shard)
                .ok_or_else(|| format!("{tname}: shard {shard} is not open"))?
                .data(t)?;
            match &mut out {
                None => {
                    out = Some(Stack {
                        name,
                        ty: t.ty,
                        k,
                        rows,
                        groups: 1,
                        bytes: data.to_vec(),
                    });
                }
                Some(s) if s.k == k => {
                    if s.ty != t.ty {
                        return Err(format!(
                            "{name}: {tname} is {:?}, the role's first part {:?}",
                            t.ty, s.ty
                        )
                        .into());
                    }
                    s.rows += rows;
                    s.bytes.extend_from_slice(data);
                }
                Some(_) => {}
            }
        }
        let mut s = out.ok_or_else(|| format!("{name}: no part"))?;
        if grouped {
            let g = split
                .arch_get_u64("attention.output_group_count")
                .ok_or("the file has no attention.output_group_count")?
                as usize;
            if g == 0 || !s.rows.is_multiple_of(g) {
                return Err(format!("{name}: {} rows are not {g} groups", s.rows).into());
            }
            s.groups = g;
        }
        Ok(s)
    }

    /// A role on the card: the stack, its route and activations at `m`
    /// tokens, and the output.
    struct OnCard {
        w: DeviceTensor<u32>,
        ty: GemmWeight,
        route: GemmRoute,
        act: GemmAct,
        input: GemmInput,
        y: DeviceBuffer<f32>,
        slots: usize,
        rows_per: usize,
        xs: Vec<f32>,
    }

    fn upload(gpu: &Gpu, gk: &GemmKernels, s: &Stack, m: usize) -> Result<OnCard, GateError> {
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let ty = GemmWeight::from_ggml(s.ty)?;
        let n_sb = s.k / 256;
        let cols = (s.rows * ty.block_bytes() * n_sb)
            .div_ceil(4)
            .div_ceil(s.rows);
        let mut words = vec![0u32; s.rows * cols];
        for (w, b) in words.iter_mut().zip(s.bytes.chunks(4)) {
            let mut q = [0u8; 4];
            q[..b.len()].copy_from_slice(b);
            *w = u32::from_le_bytes(q);
        }
        let w = DeviceTensor::upload(stream, &words, s.rows, cols)?;
        let slots = m * s.groups;
        let rows_per = s.rows / s.groups;
        let xs = fill_pattern_f32(slots * s.k);
        let xd = DeviceBuffer::<f32>::from_host(stream, &xs)?;
        let mut act = GemmAct::new(stream, slots, s.k)?;
        gpu.enqueue_quantize_gemm(&xd, slots, &mut act, sink)?;
        let mut route = GemmRoute::new(stream, slots, s.groups)?;
        if s.groups == 1 {
            gk.enqueue_route_dense(stream, slots, &mut route, sink)?;
        } else {
            let ids: Vec<u32> = (0..slots).map(|i| (i % s.groups) as u32).collect();
            let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
            gk.enqueue_route(stream, &ids_d, slots, &mut route, sink)?;
            stream.synchronize()?;
        }
        let y = DeviceBuffer::<f32>::zeroed(stream, slots * rows_per)?;
        stream.synchronize()?;
        Ok(OnCard {
            w,
            ty,
            route,
            act,
            input: GemmInput::PerSlot,
            y,
            slots,
            rows_per,
            xs,
        })
    }

    fn enqueue(
        gk: &GemmKernels,
        c: &mut OnCard,
        s: &CudaStream,
    ) -> Result<(), bloomery_gpu::GpuError> {
        gk.enqueue_gemm(
            s,
            GemmArgs {
                ty: c.ty,
                w: &c.w,
                rows_per_expert: c.rows_per,
                act: &c.act,
                route: &c.route,
                input: c.input,
                y: &mut c.y,
            },
        )
    }

    /// Six outputs of one launch against the f64 dot of the file's bytes;
    /// the worst ratio of |error| to the rounding bound.
    fn check(
        gpu: &Gpu,
        gk: &GemmKernels,
        s: &Stack,
        c: &mut OnCard,
    ) -> Result<(bool, String), GateError> {
        let stream = gpu.stream();
        enqueue(gk, c, stream)?;
        let y = c.y.to_host_vec(stream)?;
        if let Some(f) = gpu.fault()? {
            return Ok((false, format!("the launch raised {f:?}")));
        }
        let mut row = vec![0f32; s.k];
        let mut worst = 0f64;
        for slot in [0, c.slots - 1] {
            let g = slot % s.groups;
            let x = &c.xs[slot * s.k..(slot + 1) * s.k];
            for r in [0, c.rows_per / 2, c.rows_per - 1] {
                dequant_row(s.ty, s.row(g * c.rows_per + r), &mut row)?;
                let (mut dot, mut bound) = (0f64, 0f64);
                for (wb, xb) in row.chunks(128).zip(x.chunks(128)) {
                    let amax = xb.iter().fold(0f32, |a, v| a.max(v.abs())) as f64;
                    let wsum: f64 = wb.iter().map(|v| (*v as f64).abs()).sum();
                    bound += 0.5 * amax / 127.0 * wsum;
                    dot += wb
                        .iter()
                        .zip(xb)
                        .map(|(w, v)| *w as f64 * *v as f64)
                        .sum::<f64>();
                }
                let got = y[slot * c.rows_per + r] as f64;
                let ratio = (got - dot).abs() / (1.01 * bound + 1e-6);
                if !ratio.is_finite() {
                    return Ok((false, format!("slot {slot} row {r}: {got} against {dot}")));
                }
                worst = worst.max(ratio);
            }
        }
        Ok((worst <= 1.0, format!("worst_error_over_bound={worst:.3}")))
    }

    pub fn run() -> Result<(), GateError> {
        record::at_main("probe_imma", record::PROBE_IMMA_BIN);
        let a = args()?;
        let split = Split::open(gguf::v41::model())?;
        let stacks: Vec<Stack> = ROLES
            .iter()
            .map(|(n, p, g)| stack(&split, a.layer, n, p, *g))
            .collect::<Result<_, _>>()?;
        let gpu = Gpu::new()?;
        let gk = GemmKernels::load(gpu.context())?;
        let mut cards: Vec<OnCard> = Vec::with_capacity(stacks.len());
        let mut pass = true;
        for s in &stacks {
            let mut c = upload(&gpu, &gk, s, a.m)?;
            let (ok, saw) = check(&gpu, &gk, s, &mut c)?;
            Record::new(&record::PROBE_CHECK)
                .w("probe", "probe_imma")
                .w("case", s.name)
                .w(
                    "saw",
                    format!(
                        "layer={} {:?} groups={} rows={} k={} m={} {saw}",
                        a.layer,
                        s.ty,
                        s.groups,
                        s.rows / s.groups,
                        s.k,
                        a.m
                    ),
                )
                .w("pass", ok)
                .print();
            pass &= ok;
            cards.push(c);
        }
        Record::new(&record::PROBE_CHECK)
            .w("probe", "probe_imma")
            .w("case", "all")
            .w("saw", format!("{} roles", stacks.len()))
            .w("pass", pass)
            .print();
        if !pass {
            return Err(checks_failed());
        }
        if !a.time {
            return Ok(());
        }
        let mut sum = 0f64;
        for (s, c) in stacks.iter().zip(cards.iter_mut()) {
            let g = gpu.capture(|st| (0..N).try_for_each(|_| enqueue(&gk, c, st)))?;
            let (min, mean, _) = replay(gpu.stream(), &g)?;
            sum += min;
            let ops = 2.0 * (c.slots * c.rows_per * s.k) as f64;
            Record::new(&record::PROBE_IMMA_ARM)
                .u("layer", a.layer)
                .w("role", s.name)
                .w("ty", format!("{:?}", s.ty))
                .u("groups", s.groups)
                .u("rows", c.rows_per)
                .u("k", s.k)
                .u("m", a.m)
                .f("graph_us_min", min)
                .f("graph_us_mean", mean)
                .f("tops", ops / min / 1e6)
                .print();
        }
        Record::new(&record::PROBE_IMMA_TOTAL)
            .u("layer", a.layer)
            .u("m", a.m)
            .csv("roles", stacks.iter().map(|s| s.name))
            .f("us_per_sub_block", sum)
            .u("batch", a.batch)
            .f("ms_per_layer_batch", sum * (a.batch / a.m) as f64 / 1e3)
            .print();
        Ok(())
    }
}
