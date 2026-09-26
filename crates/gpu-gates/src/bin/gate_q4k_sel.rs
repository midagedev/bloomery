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
//!
//! Two more on the grouped side. `q4k_gemv_grouped` over the table a bucket
//! pass writes for [`SEL_A`] gives each slot its `_sel` reference; a table
//! whose last run ends past the slots raises `ExpertId` and leaves that
//! run's slot untouched, and one whose last run starts after its end raises
//! `ExpertId` too (`q4k_sel::grouped_run`). And the card slots' quantizer
//! (`enqueue_quantize_sel`) over columns 6..12 of a twelve-column act: a
//! column whose place is on the card reads back — through the grouped down
//! — as the plain quantizer's column, every other column as what the act
//! held before; a NaN in a column the host serves raises nothing, one in a
//! card column raises `QuantColumn`.
//!
//! And the tile table (`enqueue_grouped_tiles`) and the tiled down
//! (`q4k_gemv_tiles`) at V4.1's geometry: runs of 0, 1, 7, 8, 9, 16, 17 and
//! 237 slots spread over the stack's experts in a shuffled slot order, with
//! slots the host serves between them. The table must be the host's cut of
//! the runs; the down's input is the order-position quantizer's
//! (`enqueue_quantize_ord`) on the slots' columns gathered on the host into
//! table order, NaN in every column past the table's count — so a quantized
//! column past it raises `QuantColumn`. Every card slot must equal
//! `q4k_gemv_grouped` over the same table bit for bit, every host slot stay
//! untouched, and nothing raise. Then a count past the slots and a run past
//! the count each leave no tiles and raise `ExpertId` from the table; a tile
//! word past its run and an entry that names no slot raise it from the down
//! and leave that tile's (that entry's) slots untouched.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_q4k_sel: built without the `gpu` feature; see `just gate-gpu-q4k-sel`.");
    std::process::exit(2);
}

// Module-level because the stack and check helpers below `run` share them.
#[cfg(feature = "gpu")]
use bloomery_gpu::hybrid::HOST;
#[cfg(feature = "gpu")]
use bloomery_gpu::q4k_sel::{QuantSel, TILE_COLS, tile_cap};
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

    // 5. The grouped down's run tables, on a 2048-row stack.
    ok &= check_grouped_runs(&gpu, &mut a)?;

    // 6. The card slots' quantizer, read back through the grouped down.
    ok &= check_quantize_sel(&gpu, &a)?;

    // 7. The tiled down and the order-position quantizer, at V4.1's geometry.
    ok &= check_tiles(&gpu, &c)?;

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    println!(
        "PASSED: gate_q4k_sel _sel slots bit-identical to q4k_gemv on each expert and column \
         alone (K = 2048, 2816, 2304); graph replay follows ids overwritten between replays; \
         an id past the stack raises expert_id, a host slot raises nothing, both leave their \
         slots untouched; host contract refuses \
         a shared-input act and a non-dividing rows_per_expert; the grouped down gives each \
         slot its reference over a bucket table and raises expert_id on a run that does not \
         fit; the card slots' quantizer writes the plain quantizer's bytes into card columns \
         alone; the tile table is the host's cut of runs of 0..237 slots, the tiled down over \
         it is the grouped down bit for bit, and both raise expert_id on a table they cannot read"
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
    let expert_id = Fault::at(LAYER_NONE, FaultSite::ExpertId);
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

/// The table a bucket pass writes for `sel`: every slot whose id is below
/// `experts`, grouped by id in increasing slot order (`order`), and each id's
/// run (`start`, `experts + 1` entries).
#[cfg(feature = "gpu")]
fn bucket_table(sel: &[u32], experts: usize) -> (Vec<u32>, Vec<u32>) {
    let mut order = Vec::with_capacity(sel.len());
    let mut start = Vec::with_capacity(experts + 1);
    for e in 0..experts {
        start.push(order.len() as u32);
        order.extend((0..sel.len() as u32).filter(|&s| sel[s as usize] as usize == e));
    }
    start.push(order.len() as u32);
    (order, start)
}

/// Check 5: `q4k_gemv_grouped` on stack `st`'s six columns over three run
/// tables for [`SEL_A`], each into a `SENT`-filled `y`: the bucket table
/// (every slot its reference, no fault); its last run ending past the slots
/// (`ExpertId`, that run's slot untouched); its last run starting after its
/// end (`ExpertId`).
#[cfg(feature = "gpu")]
fn check_grouped_runs(gpu: &Gpu, st: &mut Stack) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let (order, start) = bucket_table(&SEL_A, N_EXPERTS);
    let order_dev = DeviceBuffer::from_host(stream, &order)?;
    let want = st.expected(gpu, &SEL_A)?;
    let rpe = st.rpe;
    // The last expert's run: slot 2 of SEL_A, one entry from the end.
    let last = SEL_A
        .iter()
        .position(|&id| id as usize == N_EXPERTS - 1)
        .ok_or("gate_q4k_sel: SEL_A routes no slot to the last expert")?;
    let mut past = start.clone();
    past[N_EXPERTS] = N_SLOTS as u32 + 3;
    let mut after = start.clone();
    after[N_EXPERTS] = after[N_EXPERTS - 1] - 1;
    let expert_id = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
    // (case, the run table, the fault it raises, the slot it leaves untouched)
    type RunCase = (&'static str, Vec<u32>, Option<Fault>, Option<usize>);
    let cases: [RunCase; 3] = [
        ("bucket_table", start, None, None),
        ("last_run_past_the_slots", past, expert_id, Some(last)),
        (
            "last_run_starts_after_its_end",
            after,
            expert_id,
            Some(last),
        ),
    ];
    let mut ok = gpu.take_fault()?.is_none();
    for (case, start, want_fault, untouched) in cases {
        let start_dev = DeviceBuffer::from_host(stream, &start)?;
        let mut y = st.sentinel_y(gpu)?;
        gpu.q4k_sel().enqueue_gemv_q4k_grouped(
            stream,
            &st.w,
            &st.act,
            &order_dev,
            &start_dev,
            N_SLOTS,
            0,
            rpe,
            gpu.unlabelled_sink(),
            &mut y,
        )?;
        stream.synchronize()?;
        let got = y.to_host_vec(stream)?;
        let fault = gpu.take_fault()?;
        let mut expect = want.clone();
        if let Some(s) = untouched {
            expect[s * rpe..(s + 1) * rpe].fill(SENT);
        }
        let values_ok = bits_equal(&got, &expect);
        let fault_ok = fault == want_fault;
        let pass = values_ok && fault_ok;
        ok &= pass;
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        println!(
            "q4k_grouped[{}:{case}] start_last={:?} fault=\"{}\" want=\"{}\" \
             slots_as_want={values_ok} {}{}",
            st.tag,
            &start[N_EXPERTS - 1..],
            shown(fault),
            shown(want_fault),
            verdict(pass),
            mismatch("first_mismatch", &got, &expect, rpe),
        );
    }
    Ok(ok)
}

/// Columns of check 6's act, and the first one the card slots' quantizer
/// writes.
#[cfg(feature = "gpu")]
const QS_COLS: usize = 12;
#[cfg(feature = "gpu")]
const QS_FROM: usize = 6;
/// Check 6's places of columns `QS_FROM..QS_COLS`: two card experts, the
/// host's, a place past the card's experts, two more card experts.
#[cfg(feature = "gpu")]
const QS_SEL: [u32; QS_COLS - QS_FROM] = [3, HOST, 0, N_EXPERTS as u32 + 4, 7, 1];

/// Check 6: the card slots' quantizer on stack `st`'s rows, read back
/// through the grouped down with every slot on expert 0 (module doc).
#[cfg(feature = "gpu")]
fn check_quantize_sel(gpu: &Gpu, st: &Stack) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let k = st.k;
    let x = activations(k, QS_COLS, 6421);
    let x_old = activations(k, QS_COLS, 6529);
    let sel_dev = DeviceBuffer::from_host(stream, &QS_SEL)?;
    let quantized = |v: &[f32]| -> Result<Q8Act, GateError> {
        let v = DeviceBuffer::from_host(stream, v)?;
        let mut act = Q8Act::with_slots(stream, QS_COLS, k)?;
        gpu.enqueue_quantize_q8_1(&v, &mut act)?;
        Ok(act)
    };
    // Every slot on expert 0: slot s reads column s.
    let order: Vec<u32> = (0..QS_COLS as u32).collect();
    let mut start = vec![QS_COLS as u32; N_EXPERTS + 1];
    start[0] = 0;
    let order_dev = DeviceBuffer::from_host(stream, &order)?;
    let start_dev = DeviceBuffer::from_host(stream, &start)?;
    let read_back = |act: &Q8Act| -> Result<Vec<f32>, GateError> {
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; QS_COLS * st.rpe])?;
        gpu.q4k_sel().enqueue_gemv_q4k_grouped(
            stream,
            &st.w,
            act,
            &order_dev,
            &start_dev,
            QS_COLS,
            0,
            st.rpe,
            gpu.unlabelled_sink(),
            &mut y,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    };
    let y_new = read_back(&quantized(&x)?)?;
    let y_old = read_back(&quantized(&x_old)?)?;
    let mut ok = gpu.take_fault()?.is_none();
    let card = |c: usize| c >= QS_FROM && (QS_SEL[c - QS_FROM] as usize) < N_EXPERTS;
    let rpe = st.rpe;
    let mut expect = vec![0.0f32; QS_COLS * rpe];
    for c in 0..QS_COLS {
        let from = if card(c) { &y_new } else { &y_old };
        expect[c * rpe..(c + 1) * rpe].copy_from_slice(&from[c * rpe..(c + 1) * rpe]);
    }
    // (case, the NaN's column, the fault it raises)
    let quant_column = Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
    let cases: [(&str, Option<usize>, Option<Fault>); 3] = [
        ("finite", None, None),
        ("nan_in_a_host_column", Some(QS_FROM + 1), None),
        ("nan_in_a_card_column", Some(QS_FROM), quant_column),
    ];
    for (case, nan_at, want_fault) in cases {
        let mut xs = x.clone();
        if let Some(c) = nan_at {
            xs[c * k + 5] = f32::NAN;
        }
        let x_dev = DeviceBuffer::from_host(stream, &xs)?;
        let mut act = quantized(&x_old)?;
        let q = QuantSel {
            x: &x_dev,
            cols: QS_FROM..QS_COLS,
            sel: &sel_dev,
            n_card: N_EXPERTS,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, gpu.unlabelled_sink(), &mut act)?;
        stream.synchronize()?;
        let fault = gpu.take_fault()?;
        let fault_ok = fault == want_fault;
        let got = read_back(&act)?;
        let read_clean = gpu.take_fault()?.is_none();
        // A NaN in a card column: that column's bytes are not compared, only
        // its fault.
        let mut want = expect.clone();
        if let Some(c) = nan_at.filter(|&c| card(c)) {
            want[c * rpe..(c + 1) * rpe].copy_from_slice(&got[c * rpe..(c + 1) * rpe]);
        }
        let values_ok = bits_equal(&got, &want);
        let pass = fault_ok && read_clean && values_ok;
        ok &= pass;
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        println!(
            "q4k_quantize_sel[{}:{case}] cols={QS_FROM}..{QS_COLS} places={QS_SEL:?} \
             fault=\"{}\" want=\"{}\" card_columns_plain_bytes_others_untouched={values_ok} {}{}",
            st.tag,
            shown(fault),
            shown(want_fault),
            verdict(pass),
            mismatch("first_mismatch", &got, &want, rpe),
        );
    }
    Ok(ok)
}

/// Check 7's runs, one per expert of [`TILE_EXPERTS`]: the tile walk's
/// edges — an empty run, one column, a tile short of full, full, one past,
/// two full, two and one, and a long run with a short last tile.
#[cfg(feature = "gpu")]
const TILE_RUNS: [usize; 8] = [0, 1, 7, 8, 9, 16, 17, 237];
/// The experts that carry [`TILE_RUNS`], out of order; the stack's others
/// carry no slot.
#[cfg(feature = "gpu")]
const TILE_EXPERTS: [u32; 8] = [3, 0, 5, 15, 9, 2, 12, 7];
/// Check 7's slots the host serves, between the card slots.
#[cfg(feature = "gpu")]
const TILE_HOST_SLOTS: usize = 20;

/// Check 7: the tile table and the tiled down on stack `st` (module doc).
#[cfg(feature = "gpu")]
fn check_tiles(gpu: &Gpu, st: &Stack) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let (k, rpe) = (st.k, st.rpe);
    // Each slot's place: the runs' experts and the host's, shuffled by a
    // fixed xorshift so a run's slots are not consecutive.
    let mut sel: Vec<u32> = TILE_EXPERTS
        .iter()
        .zip(TILE_RUNS)
        .flat_map(|(&e, n)| std::iter::repeat_n(e, n))
        .chain(std::iter::repeat_n(HOST, TILE_HOST_SLOTS))
        .collect();
    let mut x32 = 0x2545_f491_u32;
    for i in (1..sel.len()).rev() {
        x32 ^= x32 << 13;
        x32 ^= x32 >> 17;
        x32 ^= x32 << 5;
        sel.swap(i, x32 as usize % (i + 1));
    }
    let n_slots = sel.len();
    let (mut order, start) = bucket_table(&sel, N_EXPERTS);
    let count = start[N_EXPERTS] as usize;
    // The table holds an entry a slot; those past the count are never read.
    order.resize(n_slots, u32::MAX);
    let x = activations(k, n_slots, 6607);
    // The reference: the grouped down, slot s reading column s.
    let x_dev = DeviceBuffer::from_host(stream, &x)?;
    let mut act = Q8Act::with_slots(stream, n_slots, k)?;
    gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
    let order_dev = DeviceBuffer::from_host(stream, &order)?;
    let start_dev = DeviceBuffer::from_host(stream, &start)?;
    let mut want = DeviceBuffer::from_host(stream, &vec![SENT; n_slots * rpe])?;
    gpu.q4k_sel().enqueue_gemv_q4k_grouped(
        stream,
        &st.w,
        &act,
        &order_dev,
        &start_dev,
        n_slots,
        0,
        rpe,
        gpu.unlabelled_sink(),
        &mut want,
    )?;
    stream.synchronize()?;
    let want = want.to_host_vec(stream)?;
    let mut ok = gpu.take_fault()?.is_none();
    // The tiled input: column j is slot order[j]'s, NaN past the count.
    let mut x_ord = vec![f32::NAN; n_slots * k];
    for (j, &s) in order[..count].iter().enumerate() {
        let s = s as usize;
        x_ord[j * k..(j + 1) * k].copy_from_slice(&x[s * k..(s + 1) * k]);
    }
    let x_ord_dev = DeviceBuffer::from_host(stream, &x_ord)?;
    let cap = tile_cap(n_slots, N_EXPERTS);
    let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
    // The table's tiles, as the host cuts them: count, then (e << 22) | j.
    let mut host_tiles = vec![0u32];
    for e in 0..N_EXPERTS {
        host_tiles.extend(
            (start[e]..start[e + 1])
                .step_by(TILE_COLS)
                .map(|j| (e as u32) << 22 | j),
        );
    }
    host_tiles[0] = host_tiles.len() as u32 - 1;
    // One pass: the tile table from `start` (or `tiles` as given), the
    // order-position quantizer, the tiled down; the builder's fault, the
    // table, the quantizer's fault, the down's fault and its output.
    type Pass = (
        Option<Fault>,
        Vec<u32>,
        Option<Fault>,
        Option<Fault>,
        Vec<f32>,
    );
    let run = |start: &[u32], order: &[u32], tiles: Option<&[u32]>| -> Result<Pass, GateError> {
        let start_dev = DeviceBuffer::from_host(stream, start)?;
        let order_dev = DeviceBuffer::from_host(stream, order)?;
        let mut tiles_dev = DeviceBuffer::from_host(stream, &vec![0u32; cap + 1])?;
        match tiles {
            Some(t) => {
                let mut v = t.to_vec();
                v.resize(cap + 1, 0);
                tiles_dev = DeviceBuffer::from_host(stream, &v)?;
            }
            None => gpu.q4k_sel().enqueue_grouped_tiles(
                stream,
                &start_dev,
                N_EXPERTS,
                n_slots,
                gpu.unlabelled_sink(),
                &mut tiles_dev,
            )?,
        }
        stream.synchronize()?;
        let table_fault = gpu.take_fault()?;
        let table = tiles_dev.to_host_vec(stream)?;
        let mut act = Q8Act::with_slots(stream, n_slots, k)?;
        gpu.q4k_sel().enqueue_quantize_ord(
            stream,
            &x_ord_dev,
            &start_dev,
            N_EXPERTS,
            n_slots,
            gpu.unlabelled_sink(),
            &mut act,
        )?;
        stream.synchronize()?;
        let quant_fault = gpu.take_fault()?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; n_slots * rpe])?;
        gpu.q4k_sel().enqueue_gemv_q4k_tiles(
            stream,
            &st.w,
            &act,
            &order_dev,
            &start_dev,
            &tiles_dev,
            n_slots,
            rpe,
            gpu.unlabelled_sink(),
            &mut y,
        )?;
        stream.synchronize()?;
        Ok((
            table_fault,
            table,
            quant_fault,
            gpu.take_fault()?,
            y.to_host_vec(stream)?,
        ))
    };
    let runs = start.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    {
        let (table_fault, table, quant_fault, fault, got) = run(&start, &order, None)?;
        let tiles_n = host_tiles.len() - 1;
        let table_ok = table[..=tiles_n] == host_tiles[..];
        let values_ok = bits_equal(&got, &want);
        let pass = table_fault.is_none()
            && table_ok
            && quant_fault.is_none()
            && fault.is_none()
            && values_ok;
        ok &= pass;
        println!(
            "q4k_tiles[{}:bucket_table] slots={n_slots} card={count} runs={runs:?} tiles={} \
             cap={cap} table_as_host={table_ok} faults=\"{}\",\"{}\",\"{}\" \
             slots_as_grouped={values_ok} {}{}",
            st.tag,
            table[0],
            shown(table_fault),
            shown(quant_fault),
            shown(fault),
            verdict(pass),
            mismatch("first_mismatch", &got, &want, rpe),
        );
    }
    // The table faults: a count past the slots and a run past the count
    // leave no tiles (nothing written); a tile word past its expert's run
    // skips that tile's slots; an entry that names no slot skips its slot.
    let bad_run = TILE_EXPERTS[4] as usize;
    let mut past_count = start.clone();
    past_count[N_EXPERTS] = n_slots as u32 + 3;
    let mut run_past = start.clone();
    run_past[bad_run + 1] = count as u32 + 1;
    let mut stray = order.clone();
    let stray_at = count / 2;
    stray[stray_at] = n_slots as u32 + 5;
    // The long run's second tile, pointed one past its run's end.
    let long = TILE_EXPERTS[7] as usize;
    let bad_tile = host_tiles
        .iter()
        .skip(1)
        .position(|&w| (w >> 22) as usize == long)
        .ok_or("gate_q4k_sel: the long run has no tile")?
        + 2;
    let mut word_past = host_tiles.clone();
    word_past[bad_tile] = (long as u32) << 22 | start[long + 1];
    let bad_tile_slots: Vec<usize> = (0..TILE_COLS)
        .map(|c| order[(host_tiles[bad_tile] & 0x3f_ffff) as usize + c] as usize)
        .collect();
    let none = |_: usize| true;
    let untouched_tile = |s: usize| bad_tile_slots.contains(&s);
    let untouched_stray = |s: usize| s == order[stray_at] as usize;
    // (case, start, order, a tile table to use instead of the builder's, the
    // slots left untouched)
    type TileCase<'a> = (
        &'static str,
        &'a [u32],
        &'a [u32],
        Option<&'a [u32]>,
        &'a dyn Fn(usize) -> bool,
    );
    let cases: [TileCase<'_>; 4] = [
        ("count_past_the_slots", &past_count, &order, None, &none),
        ("a_run_past_the_count", &run_past, &order, None, &none),
        (
            "a_tile_past_its_run",
            &start,
            &order,
            Some(&word_past),
            &untouched_tile,
        ),
        (
            "an_entry_naming_no_slot",
            &start,
            &stray,
            None,
            &untouched_stray,
        ),
    ];
    let expert_id = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
    for (case, start, order, tiles, untouched) in cases {
        let (table_fault, table, _, fault, got) = run(start, order, tiles)?;
        let mut expect = want.clone();
        for s in (0..n_slots).filter(|&s| sel[s] != HOST && untouched(s)) {
            expect[s * rpe..(s + 1) * rpe].fill(SENT);
        }
        let values_ok = bits_equal(&got, &expect);
        // The builder's refusals are the table's; the rest are the down's.
        let raised = if tiles.is_none() && table[0] == 0 {
            table_fault
        } else {
            fault
        };
        let pass = raised == expert_id && values_ok;
        ok &= pass;
        println!(
            "q4k_tiles[{}:{case}] tiles={} table_fault=\"{}\" down_fault=\"{}\" want=\"{}\" \
             written_as_want={values_ok} {}{}",
            st.tag,
            table[0],
            shown(table_fault),
            shown(fault),
            shown(expert_id),
            verdict(pass),
            mismatch("first_mismatch", &got, &expect, rpe),
        );
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
