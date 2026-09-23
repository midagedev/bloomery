//! Gate `gate-ds41-plan`: the V4.1 host step plan (`model::arch::deepseek41::plan`) against
//! the graph inputs of the oracle sets, bit for bit. Host only: no card, no gate lock.
//!
//! The sets are the table's batch set (`ref_deepseek41`, five tokens from position 0) and its
//! decode-step sets (`step_sets`). A set's header names its sequence (`# tokens`) and, in a
//! decode-step set, the step's position (`# decode_pos`: the step runs the last token, the rest
//! is its history — `RefManifest::step`). The planner is laid out from the file the sets were
//! dumped from (`$BLOOMERY_REF_MODEL`, whose name must be each header's `# model_file`) at the
//! port's context (`-c` in `# flags`; the sequence's length on a set dumped before that line),
//! and plans the set's one step. Every input row of the set is then claimed by one check:
//!
//! * the integer inputs through their exact twins (`ref_ints`, which proves each twin's file
//!   against its row): tokens, positions, raw write cells, output ids, and per stream the
//!   state reads and writes, the write positions and the persist sources and destinations;
//! * the f16 masks (the raw window's and each stream's) through their f32 files, every value
//!   exactly `0.0` or `-inf` (`mask_bits_in`), mapped to its f16 bits and compared with the
//!   plan's rendering;
//! * each engram site's row ids: the plan's n-grams through the engram crate's hash;
//! * the engram gain ids, `0 .. hc`: the port's `get_rows` over its quantized gains, not a
//!   step input (ours decode to f32 at load).
//!
//! An input the plan has none of must be absent from the set: the port gives an empty input no
//! consumer. An input row no check claims fails the gate, so a new graph input shows up here
//! first; a step set's state inputs ([`is_state`]) are the caches the step reads, not the
//! plan's, and are only counted, by type and shape. The planner's streams map to the port's
//! names in order: the first `csa`, the second `hca`.
//!
//! One verdict line per set and input and a final PASS/FAIL; exit 1 on FAIL.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use bloomery_gpu_gates::oracle::{self, Set};
use bloomery_gpu_gates::{
    GateError, Layout, RefManifest, RefRow, RowKind, checks_failed, exit_with, mask_bits_in,
    ref_ints, ref_model_path, verdict,
};
use engram::Hash;
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::plan::{Planner, StepPlan};

const NAME: &str = "gate_deepseek41_plan";

/// The port's names of the compressed streams, in the planner's stream order.
const STREAMS: [&str; 2] = ["csa", "hca"];

fn main() -> std::process::ExitCode {
    exit_with(NAME, run())
}

/// A step set's state input: a cache or compressor state the step reads, which the dumper writes
/// as an input row at its first reader (`# state_inputs`). Its leaf number differs between sets,
/// so it is known by what it is: a layer's raw window (`cache_k_l<layer>`) or an unnamed leaf
/// (`leaf_<n>`: compressor states, compressed rows, index keys) that holds floats — an index or
/// position input is an integer, so an unnamed one still fails the gate. Not by its reader: a
/// ring state whose first reader writes into it (a `SET_ROWS` destination) is in neither of the
/// manifest's source columns.
fn is_state(row: &RefRow) -> bool {
    let named = row.name.starts_with("cache_k_l") || row.name.starts_with("leaf_");
    named && matches!(row.ty.as_str(), "f32" | "f16" | "bf16")
}

/// The verdict lines so far, and the inputs of the current set they claimed.
#[derive(Default)]
struct Checks {
    set: String,
    claimed: BTreeSet<String>,
    lines: usize,
    failed: usize,
}

impl Checks {
    fn line(&mut self, input: &str, pass: bool, what: &str) {
        self.claimed.insert(input.to_string());
        self.lines += 1;
        if !pass {
            self.failed += 1;
        }
        println!(
            "{NAME}: {:<36} {input:<26} {what} — {}",
            self.set,
            verdict(pass)
        );
    }
}

fn run() -> Result<(), GateError> {
    let path = ref_model_path()?;
    let split = Split::open(&path)?;
    let hp = Hparams::read(&split)?;
    let hash = Hash::from_gguf(&gguf::inventory_of(&path)?)?;
    let table = oracle::for_arch(Arch::Deepseek41)?;
    let mut sets = vec![table.open(Set::Cpu)?];
    for name in table.step_sets {
        sets.push(table.open_named(name)?);
    }
    let mut c = Checks::default();
    for man in &sets {
        gate_set(man, &path, &split, &hp, &hash, &mut c)?;
    }
    let pass = c.failed == 0;
    println!(
        "{NAME}: {} sets, {} inputs, {} failed — {}",
        sets.len(),
        c.lines,
        c.failed,
        verdict(pass)
    );
    if pass { Ok(()) } else { Err(checks_failed()) }
}

/// Plan the step set `man` holds and check every input it lists.
fn gate_set(
    man: &RefManifest,
    path: &Path,
    split: &Split,
    hp: &Hparams,
    hash: &Hash,
    c: &mut Checks,
) -> Result<(), GateError> {
    let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if let Some(want) = man.header.model_file.as_deref()
        && want != file
    {
        return Err(format!(
            "{} was dumped from {want}, $BLOOMERY_REF_MODEL is {}",
            man.dir.display(),
            path.display()
        )
        .into());
    }
    let (pos0, tokens, before) = man.step()?;
    let ctx = man
        .header
        .ctx
        .unwrap_or((before.len() + tokens.len()) as u64);
    let planner = Planner::from_file(split, hp, ctx)?;
    if planner.stream_ratios().len() != STREAMS.len() {
        return Err(format!(
            "the file has streams of ratios {:?}; the port names {}",
            planner.stream_ratios(),
            STREAMS.len()
        )
        .into());
    }
    let mut plan = StepPlan::default();
    planner.plan_into(tokens, pos0, before, &mut plan)?;
    c.set = man
        .dir
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("-")
        .to_string();
    c.claimed.clear();
    println!(
        "{NAME}: {} (build {}): a step of {} at position {pos0}, context {ctx}; window {}, streams {:?} ({}), n-gram {}",
        c.set,
        man.build.as_deref().unwrap_or("-"),
        plan.len(),
        planner.window(),
        planner.stream_ratios(),
        STREAMS.join(", "),
        planner.ngram()
    );

    ints(man, c, "inp_tokens", "i32", &wide(&plan.tokens));
    ints(man, c, "inp_pos", "i32", &wide(&plan.pos));
    ints(
        man,
        c,
        "dsv4_raw_k_write_idxs",
        "i32",
        &wide(&plan.raw_write),
    );
    out_ids(man, c, &plan);
    let mut bits = Vec::new();
    plan.raw_mask_into(&mut bits);
    // The port's one raw mask carries the name the last layer's callback gave
    // it (`dsv4_raw_mask_padded-<layer>`), so it is found by its prefix.
    let raw_mask = &man
        .only_with_prefix(RowKind::Input, "dsv4_raw_mask_padded")?
        .name;
    mask(man, c, raw_mask, plan.raw_n_kv, &bits);
    for (st, name) in plan.streams.iter().zip(STREAMS) {
        let input = |what: &str| format!("dsv4_{name}_{what}");
        let rows = st
            .state_write
            .iter()
            .map(|&r| i64::try_from(r))
            .collect::<Result<Vec<_>, _>>()?;
        ints(man, c, &input("state_read"), "i32", &wide(&st.state_read));
        ints(man, c, &input("state_write"), "i64", &rows);
        ints(man, c, &input("write_pos"), "i32", &wide(&st.write_pos));
        ints(man, c, &input("persist_src"), "i32", &wide(&st.persist_src));
        ints(man, c, &input("persist_dst"), "i32", &wide(&st.persist_dst));
        st.kq_mask_into(&mut bits);
        mask(man, c, &input("kq_mask"), st.n_kv, &bits);
    }
    engram(man, c, split, hash, &planner, &plan)?;

    let mut states = BTreeMap::<(&str, [u64; 4]), usize>::new();
    for row in &man.inputs {
        if c.claimed.contains(&row.name) {
            continue;
        }
        if man.header.decode_pos.is_some() && is_state(row) {
            *states.entry((row.ty.as_str(), row.ne)).or_default() += 1;
        } else {
            c.line(&row.name, false, "an input no check claims");
        }
    }
    let shapes: Vec<String> = states
        .iter()
        .map(|((ty, ne), n)| format!("{n} {ty} {}", shape(ne)))
        .collect();
    let by_shape = if shapes.is_empty() {
        String::new()
    } else {
        format!(" ({})", shapes.join(", "))
    };
    println!(
        "{NAME}: {:<36} {} state inputs{by_shape}: the caches the step reads, not the plan's",
        c.set,
        states.values().sum::<usize>()
    );
    Ok(())
}

/// `ne` as `ne0xne1…`, trailing unit dimensions after the second dropped.
fn shape(ne: &[u64; 4]) -> String {
    let n = ne
        .iter()
        .rposition(|&d| d != 1)
        .map_or(2, |i| (i + 1).max(2));
    ne[..n]
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn wide(v: &[u32]) -> Vec<i64> {
    v.iter().map(|&x| i64::from(x)).collect()
}

/// Where the set and the plan first differ, or that they do not.
fn compare<T: PartialEq + std::fmt::Debug>(set: &[T], plan: &[T]) -> (bool, String) {
    if set.len() != plan.len() {
        return (
            false,
            format!(
                "{} values in the set, {} in the plan",
                set.len(),
                plan.len()
            ),
        );
    }
    match set.iter().zip(plan).position(|(s, p)| s != p) {
        None => (true, format!("{} values bit-identical", set.len())),
        Some(i) => (
            false,
            format!(
                "differs at {i}: set {:?}, plan {:?} ({} values)",
                set[i],
                plan[i],
                set.len()
            ),
        ),
    }
}

/// Integer input `name` against `want` through its exact twin: a row of type `ty` holding
/// `want`, or no row when `want` is empty.
fn ints(man: &RefManifest, c: &mut Checks, name: &str, ty: &str, want: &[i64]) {
    let (pass, what) = match man.find(RowKind::Input, name, 0) {
        None if want.is_empty() => (true, "absent, and the plan has none".to_string()),
        None => (
            false,
            format!("absent from the set; the plan has {} values", want.len()),
        ),
        Some(row) if row.ty != ty => (false, format!("is {} in the set, want {ty}", row.ty)),
        Some(_) => match ref_ints(man, name, 0, RowKind::Input, Layout::Flat) {
            Err(e) => (false, format!("its twin: {e}")),
            Ok(set) => compare(&set, want),
        },
    };
    c.line(name, pass, &what);
}

/// `inp_out_ids`: the port builds it only when fewer tokens return logits than the step runs
/// (`llama.cpp:5836-5857`), so a set of a one-token step has none.
fn out_ids(man: &RefManifest, c: &mut Checks, plan: &StepPlan) {
    if plan.len() == 1 && man.find(RowKind::Input, "inp_out_ids", 0).is_none() {
        c.line(
            "inp_out_ids",
            true,
            "absent: a one-token step returns every token's logits",
        );
        return;
    }
    ints(man, c, "inp_out_ids", "i32", &wide(&plan.out_ids));
}

/// Mask input `name` against the plan's rendering `want`: f16 bits in lines of `width`, read
/// through the harness's mask reader.
fn mask(man: &RefManifest, c: &mut Checks, name: &str, width: usize, want: &[u16]) {
    let ne = [width as u64, (want.len() / width.max(1)) as u64, 1, 1];
    let (pass, what) = match man.find(RowKind::Input, name, 0) {
        None => (false, "absent from the set".to_string()),
        Some(row) if row.ty != "f16" || row.ne != ne => (
            false,
            format!("is {} {:?} in the set, want f16 {ne:?}", row.ty, row.ne),
        ),
        Some(row) => match mask_bits_in(&man.dir, row) {
            Err(e) => (false, e.to_string()),
            Ok(set) => compare(&set, want),
        },
    };
    c.line(name, pass, &what);
}

/// Each engram site's row ids, hashed from the plan's n-grams by the engram crate with the
/// file's constants and mapped as the port maps them (`llama_set_engram_rows`: the current
/// token, then each older one up to the first missing, the pad id from there on); and the
/// site's gain ids, `0 ..` its gains' second dimension.
fn engram(
    man: &RefManifest,
    c: &mut Checks,
    split: &Split,
    hash: &Hash,
    planner: &Planner,
    plan: &StepPlan,
) -> Result<(), GateError> {
    let n = planner.ngram();
    if hash.n_gram() != n {
        return Err(format!(
            "the engram hash folds {}-grams, the planner {n}-grams",
            hash.n_gram()
        )
        .into());
    }
    let mut ctx = vec![0u64; n];
    let mut rows = vec![0u32; hash.n_cols()];
    for (site, &layer) in hash.layer_ids().iter().enumerate() {
        let mut want = Vec::with_capacity(plan.len() * hash.n_cols());
        for window in plan.engram_window.chunks(n) {
            let mut blocked = false;
            for (slot, &token) in ctx.iter_mut().zip(window) {
                blocked |= token.is_none();
                *slot = match token {
                    Some(t) if !blocked => hash.map_token(t),
                    _ => hash.pad_id(),
                };
            }
            hash.rows_into(site, &ctx, &mut rows)?;
            want.extend(rows.iter().map(|&r| i64::from(r)));
        }
        ints(man, c, &format!("engram_rows-{layer}"), "i32", &want);
        for gain in ["engram_k", "engram_q"] {
            let tensor = format!("blk.{layer}.{gain}.weight");
            let (_, t) = split
                .find(&tensor)
                .ok_or_else(|| format!("{tensor} is not in the file"))?;
            let hc = t
                .dims
                .get(1)
                .copied()
                .ok_or_else(|| format!("{tensor} has dims {:?}, no second", t.dims))?;
            let ids = (0..hc).map(i64::try_from).collect::<Result<Vec<_>, _>>()?;
            ints(man, c, &format!("{gain}_ids-{layer}"), "i32", &ids);
        }
    }
    Ok(())
}
