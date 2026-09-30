//! The Qwen3.8-Flash-Next MTP draft's load gate: the target opened by its
//! placement on the gate card with the shared MTP draft file beside it
//! (`Body38::open_placed_mtp`), the draft's head reduced to a list of
//! [`LIST_ROWS`] rows over the target's tokenizer. No step runs; the draft's
//! program is `gate_qwen4exp_mtp`'s.
//!
//! What is asserted:
//! - (l) the draft is open: its weights, its store and the row map hold the
//!   bytes its plan (`PlanInputs::plan_mtp`) gives; the two matrices it
//!   borrows (`token_embd`, `output`) are the target's own buffers, at the
//!   target's addresses — nothing copied; the map holds the list's ids and
//!   each gathered row is the target's resident `output` row of its id.
//! - (r) `Mtp38::open` refuses by name, beside the loaded model: a draft
//!   read as carrying its own matrices, and a plan whose row map is not the
//!   one the load makes (after the draft's uploads, which it frees).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen4exp_mtp_load: built without the `gpu` feature; see `just \
         gate-gpu-qwen4exp-mtp-load`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_mtp_load", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::arch::qwen3moe::{Body38, Mtp38, Qwen38Model};
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use gguf::Split;
    use model::arch::models::{Borrows, HeadRows, MtpSource};
    use model::arch::qwen35moe::place::{
        MTP_HEAD_ROWS, MtpInputs, PlanInputs, machine, vocab_sha256,
    };
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::MODEL;

    /// The draft that uses the target's embedding and output matrix.
    const SHARED: &str = "/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
    /// Cache rows: the e2e gate's.
    const CTX: u64 = 3072;
    /// The head's rows: every sixth id, 0 to 245,754.
    const LIST_ROWS: u32 = 40_960;

    /// A Q8_0 weight's two planes' device addresses.
    fn planes(dw: Option<&DevWeight>) -> Option<[u64; 2]> {
        match dw {
            Some(DevWeight::Q8_0 { qs, d, .. }) => {
                Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()])
            }
            _ => None,
        }
    }

    /// The head's map holds `ids`, and its row `i` is the target's resident
    /// `output` row `ids[i]`, both planes, bit for bit: the gather against
    /// the placement loader's upload of the whole matrix.
    fn head_rows(m: &Qwen38Model, mtp: &Mtp38, ids: &[u32]) -> Result<bool, GateError> {
        let stream = m.gpu().stream();
        let Some((map, n)) = mtp.head_map() else {
            println!("(l) the draft has no head map FAIL");
            return Ok(false);
        };
        let map = map.to_host_vec(stream)?;
        let map_ok = n == ids.len() && map == ids;
        println!(
            "(l) head map: {n} rows, the list's ids {} (want {LIST_ROWS})",
            verdict(map_ok)
        );
        let (
            Some(DevWeight::Q8_0Derived { qs, d, k }),
            Some(DevWeight::Q8_0 { qs: tq, d: td, .. }),
        ) = (
            mtp.weights().get(MTP_HEAD_ROWS),
            m.weights().get("output.weight"),
        )
        else {
            println!("(l) the head's rows or the target's output are not Q8_0 planes FAIL");
            return Ok(false);
        };
        let (qs, d) = (qs.buf().to_host_vec(stream)?, d.buf().to_host_vec(stream)?);
        let (tq, td) = (tq.buf().to_host_vec(stream)?, td.buf().to_host_vec(stream)?);
        let (wq, wd) = (k / 4, k / 32);
        let differ: Vec<usize> = (0..n)
            .filter(|&i| {
                let r = ids[i] as usize;
                qs[i * wq..(i + 1) * wq] != tq[r * wq..(r + 1) * wq]
                    || d[i * wd..(i + 1) * wd] != td[r * wd..(r + 1) * wd]
            })
            .take(4)
            .collect();
        let rows_ok = differ.is_empty();
        println!(
            "(l) head row i = output row ids[i], {n} rows, both planes: first differing rows \
             {differ:?} {}",
            verdict(rows_ok)
        );
        Ok(map_ok && rows_ok)
    }

    pub(super) fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let open = |p: &str| Split::open(p).map_err(|e| format!("open {p}: {e}"));
        let (file, draft) = (open(MODEL)?, open(SHARED)?);
        let t = Instant::now();
        let inputs = PlanInputs::describe(&file)?;
        let ids: Vec<u32> = (0..LIST_ROWS).map(|i| i * 6).collect();
        let rows = HeadRows::List {
            ids: ids.clone().into(),
            digest: vocab_sha256(&file)?,
        };
        let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX)?)?;
        let machine = machine(RTX_3090, inputs.spec.layers.len(), u64::try_from(ub)?);
        let plan = inputs.plan_mtp(&machine, CTX, &PlanLevers::from_levers(&levers)?, &mtp)?;
        let d = &plan.draft.cards[0];
        println!(
            "plan card={} ctx_max={CTX} draft dense={} experts={} rounding={} kv={} map={} \
             arena={} headroom={}",
            RTX_3090.name,
            d.dense_bytes,
            d.expert_bytes,
            d.rounding_bytes,
            d.kv_bytes,
            plan.map_bytes,
            plan.arena_bytes,
            plan.headroom_bytes
        );
        let target = open(MODEL)?;
        let m = Body38::open_placed_mtp(file, &plan, &inputs, 0, levers.host(), ub, &draft, &mtp)?;
        println!(
            "load resident_bytes={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        let body = m.body("mtp load")?;
        let Some(mtp38) = body.mtp() else {
            return Err("the load opened no MTP draft".into());
        };
        let want = plan.draft_resident_bytes() + d.kv_bytes + plan.map_bytes;
        let got = mtp38.resident_bytes() as u64;
        let mut ok = got == want;
        println!(
            "(l) draft resident {got} (want {want}: weights {} + store {} + map {}) {}",
            plan.draft_resident_bytes(),
            d.kv_bytes,
            plan.map_bytes,
            verdict(got == want)
        );
        for b in mtp38.borrowed_planes() {
            let target_planes = planes(m.weights().get(&b.name));
            let same = target_planes == Some(b.planes);
            ok &= same;
            println!(
                "(l) {} read at {:x?}, the target's at {target_planes:x?} {}",
                b.name,
                b.planes,
                verdict(same)
            );
        }
        let arena = mtp38.arena_bytes() as u64;
        let arena_ok = arena == plan.arena_bytes;
        ok &= arena_ok;
        println!(
            "(l) the draft program's arena {arena} (the plan's {}) {}",
            plan.arena_bytes,
            verdict(arena_ok)
        );
        let borrowed = mtp38.borrowed(m.weights()).is_ok();
        ok &= borrowed;
        println!("(l) the walk's borrow resolves: {}", verdict(borrowed));
        let rows_ok = head_rows(&m, mtp38, &ids)?;
        ok &= rows_ok;
        let mut own = MtpInputs::read(&draft, &target, &inputs, HeadRows::Full)?;
        if let MtpSource::File { borrows, .. } = &mut own.draft.source {
            *borrows = Borrows {
                embedding: false,
                head: false,
            };
        }
        let r1 = Mtp38::open(m.gpu(), &target, m.weights(), &draft, &own, &plan, false)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r1_ok = r1.contains("the load borrows the target's token_embd and output");
        ok &= r1_ok;
        println!("(r) a draft with its own matrices: {r1} {}", verdict(r1_ok));
        let mut other = plan;
        other.map_bytes += 4;
        let r2 = Mtp38::open(m.gpu(), &target, m.weights(), &draft, &mtp, &other, false)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r2_ok = r2.contains("resident weights, store and row map hold");
        ok &= r2_ok;
        println!("(r) a plan of another row map: {r2} {}", verdict(r2_ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
