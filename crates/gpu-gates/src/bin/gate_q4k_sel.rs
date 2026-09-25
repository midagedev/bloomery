//! GPU kernel gate for `q4k_gemv_sel` (package B6): the Q4_K gemv whose
//! expert slots are read from a device buffer — the down-projection sibling
//! of the P9 kernels (`gate_p9`) for a Q4_K expert stack. Slot `s` reads the
//! rows of expert `sel[s]` and activation column `s`, so the launch can sit
//! inside a captured graph whose replay consumes whatever ids were written.
//!
//! The reference is the EXISTING `q4k_gemv` and the contract is bit
//! identity, not a band: slot `s` must equal `q4k_gemv` run on an upload of
//! expert `sel[s]`'s rows alone against a one-column activation quantized
//! from input column `s` alone. Three stacks of 16 experts: two real
//! V2-Lite Q4_K tensors viewed as 128-row experts (K = 2048, eight
//! super-blocks; K = 2816, eleven — three live in the last four-super-block
//! iteration) and a synthetic stack at V4.1's per-expert down geometry
//! (5120 rows, K = 2304, nine super-blocks — one live in the last
//! iteration). Checks: two sel vectors per stack, each carrying a duplicate
//! id whose two slots read different columns and so must differ; a
//! bit-identical rerun; the in-graph replay following ids overwritten
//! between replays; an id past the stack that raises the named fault
//! (`FaultSite::ExpertId`) and a host-served slot (`hybrid::HOST`) that
//! raises nothing, both leaving their slots untouched; and the host
//! contract's refusals. A
//! failing check names its first mismatch — slot, row, both bit patterns.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_q4k_sel: built without the `gpu` feature; see `just gate-gpu-q4k-sel`.");
    std::process::exit(2);
}

// Module-level because the stack and check helpers below `run` share them.
#[cfg(feature = "gpu")]
use bloomery_gpu::hybrid::HOST;
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    GateError, activations, bits_equal, bytes_to_words, open_model, tensor_bytes_as, verdict,
};
#[cfg(feature = "gpu")]
use cuda_core::DeviceBuffer;
#[cfg(feature = "gpu")]
use gguf::quant::GgmlType;
#[cfg(feature = "gpu")]
use std::collections::HashMap;

/// Slots per launch: V4.1's routed top-k.
#[cfg(feature = "gpu")]
const N_SLOTS: usize = 6;
/// Experts per stack; the real tensors are viewed as this many experts.
#[cfg(feature = "gpu")]
const N_EXPERTS: usize = 16;
/// Expert 9 in slots 3 and 4 — one expert, two different input columns.
#[cfg(feature = "gpu")]
const SEL_A: [u32; N_SLOTS] = [0, 5, 15, 9, 9, 2];
/// Expert 15 in slots 0 and 4.
#[cfg(feature = "gpu")]
const SEL_B: [u32; N_SLOTS] = [15, 0, 7, 1, 15, 5];
/// Slot 1 is three past the last expert; the rest are ordinary ids.
#[cfg(feature = "gpu")]
const SEL_OOR: [u32; N_SLOTS] = [3, N_EXPERTS as u32 + 3, 0, 12, 7, 1];
/// Slot 3 is [`HOST`], a slot the host tier serves; the rest are ordinary
/// ids.
#[cfg(feature = "gpu")]
const SEL_HOST: [u32; N_SLOTS] = [3, 5, 0, HOST, 7, 1];
/// What `y` holds before every launch here, so a slot the kernel leaves
/// alone — or forgets — reads back as these bits.
#[cfg(feature = "gpu")]
const SENT: f32 = 1.0e30;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_q4k_sel", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    // Fixed seeds: the gate is bit identity, so every input is fixed.
    const SEED_A: u32 = 6101;
    const SEED_B: u32 = 6203;
    const SEED_C: u32 = 6307;
    const SEED_W: u64 = 0x9e37_79b9_7f4a_7c15;

    let mut ok = true;
    let gguf = open_model()?;
    let gpu = Gpu::new()?;

    // dims [K, rows]: one 2-D tensor's 2048 rows viewed as 16 x 128.
    let (_, bytes) = tensor_bytes_as(
        &gguf,
        "blk.1.attn_output.weight",
        GgmlType::Q4_K,
        Some(&[2048, 2048]),
    )?;
    let mut a = Stack::new(
        &gpu,
        "A",
        "blk.1.attn_output.weight",
        2048,
        128,
        bytes_to_words(bytes),
        SEED_A,
    )?;
    let (_, bytes) = tensor_bytes_as(
        &gguf,
        "blk.1.ffn_down_shexp.weight",
        GgmlType::Q4_K,
        Some(&[2816, 2048]),
    )?;
    let mut b = Stack::new(
        &gpu,
        "B",
        "blk.1.ffn_down_shexp.weight",
        2816,
        128,
        bytes_to_words(bytes),
        SEED_B,
    )?;
    let mut c = Stack::new(
        &gpu,
        "C",
        "synthetic",
        2304,
        5120,
        synthetic_q4k(N_EXPERTS * 5120, 9, SEED_W),
        SEED_C,
    )?;

    // 1. Every stack under both sel vectors. Stack C's eager outputs are
    // what the graph replays must reproduce.
    for st in [&mut a, &mut b] {
        ok &= check_sel(&gpu, st, "a", &SEL_A)?.0;
        ok &= check_sel(&gpu, st, "b", &SEL_B)?.0;
    }
    let (pass_a, eager_a) = check_sel(&gpu, &mut c, "a", &SEL_A)?;
    let (pass_b, eager_b) = check_sel(&gpu, &mut c, "b", &SEL_B)?;
    ok &= pass_a && pass_b;

    // 2. In-graph, at V4.1's geometry.
    ok &= check_graph(&gpu, &c, &eager_a, &eager_b)?;

    // 3. Out-of-range ids, at V4.1's geometry.
    ok &= check_oor(&gpu, &mut c)?;

    // 4. The host contract, on a 2048-row stack.
    ok &= check_host_contract(&gpu, &a)?;

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_q4k_sel _sel slots bit-identical to q4k_gemv on each expert and column \
         alone (K = 2048, 2816, 2304); graph replay follows ids overwritten between replays; \
         an id past the stack raises expert_id, a host slot raises nothing, both leave their \
         slots untouched; host contract refuses \
         a shared-input act and a non-dividing rows_per_expert"
    );
    Ok(())
}

/// One expert stack under test: its rows on the host (a reference uploads
/// one expert's rows at a time) and on the device, and its `N_SLOTS`
/// activation columns, raw and quantized once for the `_sel` launch.
#[cfg(feature = "gpu")]
struct Stack {
    tag: &'static str,
    k: usize,
    rpe: usize,
    words: Vec<u32>,
    w: DeviceTensor<u32>,
    x: Vec<f32>,
    act: Q8Act,
    /// References by (expert, activation column).
    refs: HashMap<(usize, usize), Vec<f32>>,
}

#[cfg(feature = "gpu")]
impl Stack {
    /// Upload `words` — `N_EXPERTS * rpe` Q4_K rows of `k` values — and
    /// quantize `N_SLOTS` activation columns drawn from `seed`.
    fn new(
        gpu: &Gpu,
        tag: &'static str,
        source: &str,
        k: usize,
        rpe: usize,
        words: Vec<u32>,
        seed: u32,
    ) -> Result<Stack, GateError> {
        let n_sb = k / 256;
        let wpr = 36 * n_sb;
        let rows = N_EXPERTS * rpe;
        if words.len() != rows * wpr {
            return Err(format!(
                "gate_q4k_sel: stack {tag} ({source}) has {} words, want {rows} rows x {wpr}",
                words.len()
            )
            .into());
        }
        let iters = n_sb.div_ceil(4);
        println!(
            "stack {tag} source={source} K={k} n_sb={n_sb} iters={iters} \
             last_iter_live_sb={} experts={N_EXPERTS} rows_per_expert={rpe}",
            n_sb - 4 * (iters - 1)
        );
        let stream = gpu.stream();
        let w = DeviceTensor::upload(stream, &words, rows, wpr)?;
        let x = activations(k, N_SLOTS, seed);
        let x_dev = DeviceBuffer::from_host(stream, &x)?;
        let mut act = Q8Act::with_k(stream, N_SLOTS, k)?;
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        stream.synchronize()?;
        Ok(Stack {
            tag,
            k,
            rpe,
            words,
            w,
            x,
            act,
            refs: HashMap::new(),
        })
    }

    /// `y` filled with `SENT`, one row per output of a launch.
    fn sentinel_y(&self, gpu: &Gpu) -> Result<DeviceBuffer<f32>, GateError> {
        Ok(DeviceBuffer::from_host(
            gpu.stream(),
            &vec![SENT; N_SLOTS * self.rpe],
        )?)
    }

    /// One `_sel` launch reading ids from `sel_dev` into `y`, synchronized
    /// (a device fault surfaces here as an `Err`).
    fn launch(
        &self,
        gpu: &Gpu,
        sel_dev: &DeviceBuffer<u32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        gpu.q4k_sel()
            .enqueue_gemv_q4k_sel(stream, &self.w, &self.act, sel_dev, N_SLOTS, self.rpe, y)?;
        stream.synchronize()?;
        Ok(())
    }

    /// The reference for expert `id` on activation column `col`: the
    /// EXISTING `q4k_gemv` on an upload of that expert's rows alone,
    /// against a one-column activation quantized from column `col` alone.
    /// Cached, so a duplicate id maps to the same reference bytes.
    fn reference(&mut self, gpu: &Gpu, id: usize, col: usize) -> Result<&[f32], GateError> {
        let key = (id, col);
        if !self.refs.contains_key(&key) {
            let stream = gpu.stream();
            let wpr = self.w.cols();
            let lo = id * self.rpe * wpr;
            let w =
                DeviceTensor::upload(stream, &self.words[lo..lo + self.rpe * wpr], self.rpe, wpr)?;
            let x_col = DeviceBuffer::from_host(stream, &self.x[col * self.k..(col + 1) * self.k])?;
            let mut act = Q8Act::with_k(stream, 1, self.k)?;
            gpu.enqueue_quantize_q8_1(&x_col, &mut act)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, self.rpe)?;
            gpu.enqueue_gemv_q4k(&w, &act, &mut y)?;
            stream.synchronize()?;
            self.refs.insert(key, y.to_host_vec(stream)?);
        }
        self.refs
            .get(&key)
            .map(Vec::as_slice)
            .ok_or_else(|| "gate_q4k_sel: reference cache lost an entry".into())
    }

    /// What a launch of `sel` must leave in a `SENT`-filled `y`: slot s is
    /// the reference of (sel[s], s), or `SENT` untouched where sel[s] is out
    /// of range.
    fn expected(&mut self, gpu: &Gpu, sel: &[u32; N_SLOTS]) -> Result<Vec<f32>, GateError> {
        let mut out = Vec::with_capacity(N_SLOTS * self.rpe);
        for (s, &id) in sel.iter().enumerate() {
            if (id as usize) < N_EXPERTS {
                out.extend_from_slice(self.reference(gpu, id as usize, s)?);
            } else {
                out.extend(std::iter::repeat_n(SENT, self.rpe));
            }
        }
        Ok(out)
    }
}

/// Check 1 for one stack and one sel vector: every slot bit-identical to its
/// reference, a second launch bit-identical to the first, and the two slots
/// that share an expert different from each other — they read different
/// columns, which is what this sel vector is for. Returns the verdict and
/// the first launch's output.
#[cfg(feature = "gpu")]
fn check_sel(
    gpu: &Gpu,
    st: &mut Stack,
    tag: &str,
    sel: &[u32; N_SLOTS],
) -> Result<(bool, Vec<f32>), GateError> {
    let (da, db) = dup_slots(sel)?;
    let stream = gpu.stream();
    let sel_dev = DeviceBuffer::from_host(stream, sel)?;
    let mut y = st.sentinel_y(gpu)?;
    st.launch(gpu, &sel_dev, &mut y)?;
    let y1 = y.to_host_vec(stream)?;
    st.launch(gpu, &sel_dev, &mut y)?;
    let y2 = y.to_host_vec(stream)?;
    let want = st.expected(gpu, sel)?;
    let rpe = st.rpe;
    let slot_same = bits_equal(&y1, &want);
    let rerun_same = bits_equal(&y1, &y2);
    let dup_differ = !bits_equal(&y1[da * rpe..(da + 1) * rpe], &y1[db * rpe..(db + 1) * rpe]);
    let pass = slot_same && rerun_same && dup_differ;
    println!(
        "q4k_sel[{}:{tag}] sel={sel:?} slot_bit_identical={slot_same} \
         bit_identical_rerun={rerun_same} dup_slots={da},{db} dup_slots_differ={dup_differ} {}{}{}",
        st.tag,
        verdict(pass),
        mismatch("first_mismatch_vs_ref", &y1, &want, rpe),
        mismatch("first_mismatch_rerun", &y2, &y1, rpe),
    );
    Ok((pass, y1))
}

/// Check 2: capture one `_sel` launch with the sel buffer holding `SEL_A`,
/// replay; overwrite the buffer with `SEL_B` outside the graph, replay the
/// same graph. Each replay must equal the eager output of its ids, and the
/// graph is the one launch.
#[cfg(feature = "gpu")]
fn check_graph(gpu: &Gpu, st: &Stack, eager_a: &[f32], eager_b: &[f32]) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let mut sel_dev = DeviceBuffer::from_host(stream, &SEL_A)?;
    let mut y = st.sentinel_y(gpu)?;
    // Declared after the buffers it addresses, so it is dropped first.
    let graph = gpu.capture(|s| {
        gpu.q4k_sel()
            .enqueue_gemv_q4k_sel(s, &st.w, &st.act, &sel_dev, N_SLOTS, st.rpe, &mut y)
    })?;
    let nodes = graph.node_count();
    graph.launch(stream)?;
    stream.synchronize()?;
    let ya = y.to_host_vec(stream)?;
    sel_dev.copy_from_host(stream, &SEL_B)?; // host->device, OUTSIDE the graph
    graph.launch(stream)?;
    stream.synchronize()?;
    let yb = y.to_host_vec(stream)?;
    let a_same = bits_equal(&ya, eager_a);
    let b_same = bits_equal(&yb, eager_b);
    let pass = a_same && b_same && nodes == 1;
    println!(
        "q4k_graph[{}] replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} \
         graph_nodes={nodes} {}{}{}",
        st.tag,
        verdict(pass),
        mismatch("first_mismatch_a", &ya, eager_a, st.rpe),
        mismatch("first_mismatch_b", &yb, eager_b, st.rpe),
    );
    Ok(pass)
}

/// Check 3, twice into a `SENT`-filled `y`: [`SEL_OOR`] (an id past the
/// stack) must raise [`FaultSite::ExpertId`] on the fault word as an
/// unlabelled launch, and [`SEL_HOST`] (a slot the host tier serves) must
/// raise nothing. Either way the launch itself succeeds, the out-of-range
/// slot still holds `SENT`'s bits, and every other slot is bit-identical to
/// its reference.
#[cfg(feature = "gpu")]
fn check_oor(gpu: &Gpu, st: &mut Stack) -> Result<bool, GateError> {
    let clean = gpu.take_fault()?;
    if clean.is_some() {
        println!(
            "q4k_oor[{}] the fault word held {clean:?} before the case {}",
            st.tag,
            verdict(false)
        );
        return Ok(false);
    }
    let expert_id = Fault {
        layer: LAYER_NONE,
        code: FaultSite::ExpertId as u32,
    };
    let mut pass = oor_case(gpu, st, "past_stack", &SEL_OOR, Some(expert_id))?;
    pass &= oor_case(gpu, st, "host", &SEL_HOST, None)?;
    Ok(pass)
}

/// One of check 3's launches: `sel` into a `SENT`-filled `y`, then the fault
/// word read and cleared against `want_fault`.
#[cfg(feature = "gpu")]
fn oor_case(
    gpu: &Gpu,
    st: &mut Stack,
    case: &str,
    sel: &[u32; N_SLOTS],
    want_fault: Option<Fault>,
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let sel_dev = DeviceBuffer::from_host(stream, sel)?;
    let mut y = st.sentinel_y(gpu)?;
    if let Err(e) = st.launch(gpu, &sel_dev, &mut y) {
        println!(
            "q4k_oor[{}:{case}] sel={sel:?} launch failed: {e} {}",
            st.tag,
            verdict(false)
        );
        return Err(e);
    }
    let got = y.to_host_vec(stream)?;
    let fault = gpu.take_fault()?;
    let want = st.expected(gpu, sel)?;
    let rpe = st.rpe;
    let (mut good_same, mut bad_untouched) = (true, true);
    let mut bad = Vec::new();
    for (s, &id) in sel.iter().enumerate() {
        let same = bits_equal(&got[s * rpe..(s + 1) * rpe], &want[s * rpe..(s + 1) * rpe]);
        if (id as usize) < N_EXPERTS {
            good_same &= same;
        } else {
            bad_untouched &= same;
            bad.push(s);
        }
    }
    let fault_ok = fault == want_fault;
    let pass = good_same && bad_untouched && fault_ok;
    let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
    println!(
        "q4k_oor[{}:{case}] sel={sel:?} oor_slots={bad:?} fault=\"{}\" want=\"{}\" \
         good_slots_bit_identical={good_same} bad_slots_untouched={bad_untouched} {}{}",
        st.tag,
        shown(fault),
        shown(want_fault),
        verdict(pass),
        mismatch("first_mismatch", &got, &want, rpe),
    );
    Ok(pass)
}

/// Check 4: the launcher refuses, as a `Shape` error of its own, two
/// misuses the kernel's launch contract either reports as a different error
/// or misses: a one-column (shared-input) act for six slots, and a
/// `rows_per_expert` that does not divide the stack's rows (100 on 2048 —
/// twenty such experts fit in the buffer, so the contract alone would
/// launch it on a wrong split).
#[cfg(feature = "gpu")]
fn check_host_contract(gpu: &Gpu, st: &Stack) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let sel_dev = DeviceBuffer::from_host(stream, &SEL_A)?;
    let mut y = st.sentinel_y(gpu)?;
    let one_col = Q8Act::with_k(stream, 1, st.k)?;
    let cases = [
        (
            "act_m1_n_slots6",
            gpu.q4k_sel()
                .enqueue_gemv_q4k_sel(stream, &st.w, &one_col, &sel_dev, N_SLOTS, st.rpe, &mut y),
        ),
        (
            "rows_per_expert100_rows2048",
            gpu.q4k_sel()
                .enqueue_gemv_q4k_sel(stream, &st.w, &st.act, &sel_dev, N_SLOTS, 100, &mut y),
        ),
    ];
    let mut ok = true;
    for (case, r) in cases {
        let pass = matches!(
            r,
            Err(GpuError::Shape {
                what: "enqueue_gemv_q4k_sel",
                ..
            })
        );
        let seen = match &r {
            Ok(()) => "Ok (accepted)".to_string(),
            Err(e) => format!("Err: {e}"),
        };
        println!(
            "q4k_host[{}:{case}] want=Err(Shape enqueue_gemv_q4k_sel) got={seen} {}",
            st.tag,
            verdict(pass)
        );
        ok &= pass;
    }
    Ok(ok)
}

/// The two slots of `sel` that carry the same id — the fixture's point: an
/// error if it has none.
#[cfg(feature = "gpu")]
fn dup_slots(sel: &[u32; N_SLOTS]) -> Result<(usize, usize), GateError> {
    (0..N_SLOTS)
        .flat_map(|a| (a + 1..N_SLOTS).map(move |b| (a, b)))
        .find(|&(a, b)| sel[a] == sel[b])
        .ok_or_else(|| format!("gate_q4k_sel: sel {sel:?} carries no duplicate id").into())
}

/// ` <label>=slot S row R got 0x… want 0x…` at the first index where `got`
/// and `want` differ in bits, `rpe` rows per slot; empty when they agree.
#[cfg(feature = "gpu")]
fn mismatch(label: &str, got: &[f32], want: &[f32], rpe: usize) -> String {
    if got.len() != want.len() {
        return format!(" {label}=length got {} want {}", got.len(), want.len());
    }
    match got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits())
    {
        Some(i) => format!(
            " {label}=slot {} row {} got {:#010x} want {:#010x}",
            i / rpe,
            i % rpe,
            got[i].to_bits(),
            want[i].to_bits()
        ),
        None => String::new(),
    }
}

/// `rows` synthetic Q4_K rows of `n_sb` super-blocks as u32 words, from a
/// fixed-seed xorshift64: each super-block's 36 words are random except word
/// 0, whose low and high halves are `d` and `dmin` — positive normal f16
/// (sign 0, exponent field 1..=9, random mantissa), so no NaN or Inf
/// enters. Every pattern of the scale and quant words is a valid Q4_K
/// block. `seed` must be nonzero (xorshift's one fixed point).
#[cfg(feature = "gpu")]
fn synthetic_q4k(rows: usize, n_sb: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let half = |r: u64| -> u32 { ((1 + (r % 9) as u32) << 10) | ((r >> 32) as u32 & 0x3ff) };
    let mut out = Vec::with_capacity(rows * n_sb * 36);
    for _ in 0..rows * n_sb {
        let (d, dmin) = (half(next()), half(next()));
        out.push(d | (dmin << 16));
        for _ in 1..36 {
            out.push((next() >> 32) as u32);
        }
    }
    out
}
