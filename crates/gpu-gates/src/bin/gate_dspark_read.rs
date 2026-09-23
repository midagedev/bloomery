//! Gate `gate-dspark-read`: the DSpark draft file through the strict reader. Host only: no card,
//! no gate lock.
//!
//! - (i) open: the strict reader opens the draft (`$BLOOMERY_DSPARK_MODEL`, which the recipe
//!   exports from the V4.1 profile's `DSPARK_MODEL`), MXFP4 experts included.
//! - (ii) hparams: `DraftHparams::read` takes every key, and the values this port is built for
//!   hold (`EXPECT`).
//! - (iii) tensors: every tensor `dspark::tensors` names is in the file with its shape and type,
//!   and the file holds no other.
//! - (iv) bytes: the tensors tile the data section — each starts where the one before it ends,
//!   rounded up to the file's alignment, and the last one ends the file — so the byte table below
//!   is the file's size less its header and that padding, exactly.
//! - (v) dequant: our MXFP4 decode of the harness dump's rows (`mxfp4_ref`, ggml's own `to_float`)
//!   is bit-identical, and those rows use every one of the 16 codes.
//! - (vi) scales: every MXFP4 block's scale byte of the file, scanned (reads the experts whole):
//!   no block uses E = 255, whose code ±12 decodes to ±inf. The blocks at the two subnormal
//!   scales, E = 0 and 1, are counted and printed; the quant unit test pins their decode.
//!
//! Then the byte table: per group the file bytes and the draft card's bytes (the GPU loader's
//! `CardFormat` rule; MXFP4 at its native size), and the target's `token_embd` and `output`
//! (`$BLOOMERY_REF_MODEL`, the V4.1 profile's first shard).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bloomery_gpu_gates::{GateError, checks_failed, data_dir, exit_with, ref_model_path, verdict};
use gguf::quant::{GgmlType, dequant_row};
use gguf::{Gguf, Split};
use model::arch::dspark::{self, Borrow, DraftHparams, DraftInventory, Group};

const NAME: &str = "gate_dspark_read";

/// The harness dump (`tools/ref/mxfp4_ref.cpp`) under `$BLOOMERY_DATA/ref/`.
const DUMP: &str = "mxfp4-dspark-dequant";

/// The draft this port is built for; a file that reads otherwise fails (ii).
const EXPECT: [(&str, usize); 17] = [
    ("block_count", 3),
    ("embedding_length", 5120),
    ("attention.head_count", 64),
    ("attention.head_count_kv", 1),
    ("attention.key_length", 512),
    ("attention.q_lora_rank", 1280),
    ("attention.output_group_count", 8),
    ("attention.output_lora_rank", 1024),
    ("rope.dimension_count", 64),
    ("attention.sliding_window", 128),
    ("hyper_connection.count", 4),
    ("expert_count", 128),
    ("expert_used_count", 3),
    ("expert_shared_count", 1),
    ("expert_feed_forward_length", 2304),
    ("block_size", 5),
    ("vocab", 129_280),
];

/// `tokenizer.ggml.mask_token_id`, the Markov rank and `target_layers` as this port expects them.
const MASK_TOKEN: u32 = 128_799;
const MARKOV_RANK: usize = 256;
const TARGET_LAYERS: [usize; 3] = [37, 38, 39];

fn main() -> std::process::ExitCode {
    exit_with(NAME, run())
}

#[derive(Default)]
struct Checks {
    failed: usize,
    lines: usize,
}

impl Checks {
    fn line(&mut self, check: &str, pass: bool, what: &str) {
        self.lines += 1;
        if !pass {
            self.failed += 1;
        }
        println!("{NAME}: {check:<8} {what} — {}", verdict(pass));
    }
}

fn run() -> Result<(), GateError> {
    let path = std::env::var_os("BLOOMERY_DSPARK_MODEL")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .ok_or(
            "BLOOMERY_DSPARK_MODEL unset — run through `just gate-dspark-read`, which exports it \
             from the V4.1 profile's DSPARK_MODEL",
        )?;
    let mut c = Checks::default();

    let draft = Split::open(&path)?;
    let file = draft.shard(0).ok_or("the draft opened with no shard")?;
    c.line(
        "(i)",
        true,
        &format!(
            "{} opens strictly: {} tensors",
            path.display(),
            draft.tensor_count()
        ),
    );

    let hp = DraftHparams::read(&draft)?;
    hparams(&hp, &mut c);

    let target_path = ref_model_path()?;
    let target = Split::open(&target_path)?;
    let inv = dspark::inventory(&draft, &hp, &target)?;
    tensors(&inv, &mut c);
    tiling(file, &path, &mut c)?;
    dequant(file, &mut c)?;
    scales(file, &inv, &mut c)?;
    byte_table(&inv);
    println!(
        "{NAME}: target {}: rms_eps {:e} (the draft's {:e})",
        target_path.display(),
        target
            .arch_get_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(f32::NAN),
        hp.rms_eps
    );

    let pass = c.failed == 0;
    println!(
        "{NAME}: {} checks, {} failed — {}",
        c.lines,
        c.failed,
        verdict(pass)
    );
    if pass { Ok(()) } else { Err(checks_failed()) }
}

/// (ii): the values `EXPECT` pins, then the rest printed as read.
fn hparams(hp: &DraftHparams, c: &mut Checks) {
    let read = [
        hp.n_layer,
        hp.n_embd,
        hp.n_head,
        hp.n_head_kv,
        hp.head_dim,
        hp.q_lora_rank,
        hp.o_groups,
        hp.o_lora_rank,
        hp.rope_dims,
        hp.window,
        hp.hc.streams,
        hp.experts.n_expert,
        hp.experts.n_used,
        hp.experts.n_shared,
        hp.experts.ff,
        hp.block_size,
        hp.n_vocab,
    ];
    for ((key, want), got) in EXPECT.iter().zip(read) {
        c.line(
            "(ii)",
            got == *want,
            &format!("{key:<30} {got} (want {want})"),
        );
    }
    let mask = hp.mask_token == MASK_TOKEN;
    c.line(
        "(ii)",
        mask,
        &format!(
            "{:<30} {} (want {MASK_TOKEN})",
            "mask_token_id", hp.mask_token
        ),
    );
    let rank = hp.markov_rank == MARKOV_RANK;
    c.line(
        "(ii)",
        rank,
        &format!(
            "{:<30} {} (want {MARKOV_RANK})",
            "markov rank", hp.markov_rank
        ),
    );
    let layers = hp.target_layers == TARGET_LAYERS;
    c.line(
        "(ii)",
        layers,
        &format!(
            "{:<30} {:?} (want {TARGET_LAYERS:?})",
            "target_layers", hp.target_layers
        ),
    );
    println!(
        "{NAME}: read  rms_eps {:e}, rope_base {}, n_ctx_train {}, hc eps {:e}, sinkhorn {}, \
         routed_scale {}, weights_norm {}, swiglu {:?} / shared {:?}",
        hp.rms_eps,
        hp.rope_base,
        hp.n_ctx_train,
        hp.hc.eps,
        hp.hc.sinkhorn_iters,
        hp.experts.routed_scale,
        hp.experts.weights_norm,
        hp.swiglu_limit,
        hp.swiglu_limit_shared
    );
}

/// (iii): one line per tensor the draft reads, then the file's tensors it does not.
fn tensors(inv: &DraftInventory, c: &mut Checks) {
    for r in &inv.rows {
        let what = match &r.found {
            None => format!(
                "{:<34} absent (want {} {:?})",
                r.want.name, r.want.ty, r.want.dims
            ),
            Some((d, ty, bytes)) => {
                format!("{:<34} {ty:<5} {d:?} {bytes} B", r.want.name)
            }
        };
        c.line("(iii)", r.holds(), &what);
    }
    c.line(
        "(iii)",
        inv.unread.is_empty(),
        &format!("tensors no row names: {:?}", inv.unread),
    );
}

/// (iv): the tensors tile the data section with only alignment padding between them.
fn tiling(g: &Gguf, path: &Path, c: &mut Checks) -> Result<(), GateError> {
    let len = std::fs::metadata(path)?.len();
    let align = g.alignment();
    let mut spans: Vec<(u64, u64, &str)> = g
        .iter_tensors()
        .map(|t| (t.offset, t.nbytes, t.name.as_str()))
        .collect();
    spans.sort_unstable();
    let mut next = 0u64;
    let mut pad = 0u64;
    let mut gaps = Vec::new();
    for &(off, n, name) in &spans {
        if off != next {
            gaps.push(format!("{name} at {off}, expected {next}"));
        }
        let end = off + n;
        next = end.next_multiple_of(align);
        pad += next - end;
    }
    let data = len - g.data_base();
    let last_end = spans.last().map_or(0, |s| s.0 + s.1);
    // The writer pads the last tensor like every other, or ends the file at it.
    let ends = data == next || data == last_end;
    let tensor_bytes: u64 = spans.iter().map(|s| s.1).sum();
    let tail_pad = data - last_end;
    let inner_pad = pad - (next - last_end);
    c.line(
        "(iv)",
        gaps.is_empty() && ends,
        &format!(
            "file {len} B = header {} + tensors {tensor_bytes} + padding {} (alignment {align}; \
             between tensors {inner_pad}, after the last {tail_pad}); gaps {gaps:?}",
            g.data_base(),
            inner_pad + tail_pad
        ),
    );
    Ok(())
}

/// (v): our decode of the harness's rows against ggml's, bit for bit, and the codes they use.
fn dequant(g: &Gguf, c: &mut Checks) -> Result<(), GateError> {
    let dir = data_dir().join("ref");
    let meta_path = dir.join(format!("{DUMP}.meta"));
    let meta = std::fs::read_to_string(&meta_path).map_err(|e| {
        format!(
            "{}: {e} — `just build-ref` runs mxfp4_ref, which writes it",
            meta_path.display()
        )
    })?;
    let field = |key: &str| -> Result<&str, GateError> {
        meta.lines()
            .find_map(|l| l.strip_prefix(key))
            .ok_or_else(|| format!("{DUMP}.meta has no {key}").into())
    };
    let tname = field("tensor=")?;
    let rows: usize = field("rows=")?.parse()?;
    let k: usize = field("rowlen=")?.parse()?;
    let ggml = field("ggml=")?;
    let t = g
        .find(tname)
        .ok_or_else(|| format!("{tname} (the dump's tensor) is not in the draft"))?;
    if t.ty != GgmlType::MXFP4 || t.dims[0] as usize != k {
        return Err(format!(
            "{tname}: {} {:?}, the dump says mxfp4 rows of {k}",
            t.ty, t.dims
        )
        .into());
    }
    let raw = std::fs::read(dir.join(format!("{DUMP}.raw")))?;
    if raw.len() != rows * k * 4 {
        return Err(format!(
            "{DUMP}.raw has {} bytes, not {rows} rows of {k} f32",
            raw.len()
        )
        .into());
    }
    let row_bytes = k / 32 * 17;
    let src = &g.data(t)?[..rows * row_bytes];
    let mut ours = vec![0.0f32; rows * k];
    dequant_row(GgmlType::MXFP4, src, &mut ours)?;
    let differ = ours
        .iter()
        .zip(raw.as_chunks::<4>().0)
        .filter(|(o, r)| o.to_bits() != u32::from_le_bytes(**r))
        .count();
    let mut codes = [0u64; 16];
    for blk in src.as_chunks::<17>().0 {
        for &q in &blk[1..] {
            codes[usize::from(q & 0x0f)] += 1;
            codes[usize::from(q >> 4)] += 1;
        }
    }
    let unused: Vec<usize> = (0..16).filter(|&i| codes[i] == 0).collect();
    c.line(
        "(v)",
        differ == 0,
        &format!(
            "{tname} rows 0..{rows} x {k}: {differ} of {} values differ from ggml ({ggml})",
            rows * k
        ),
    );
    c.line(
        "(v)",
        unused.is_empty(),
        &format!("codes used by those rows {codes:?}, unused {unused:?}"),
    );
    Ok(())
}

/// (vi): every MXFP4 block's scale byte in the file.
fn scales(g: &Gguf, inv: &DraftInventory, c: &mut Checks) -> Result<(), GateError> {
    let mut hist: BTreeMap<u8, u64> = BTreeMap::new();
    let mut blocks = 0u64;
    for r in inv.rows.iter().filter(|r| r.want.ty == GgmlType::MXFP4) {
        let t = g.find(&r.want.name).ok_or("an MXFP4 row vanished")?;
        for blk in g.data(t)?.as_chunks::<17>().0 {
            *hist.entry(blk[0]).or_default() += 1;
            blocks += 1;
        }
    }
    let (lo, hi) = (hist.keys().next().copied(), hist.keys().last().copied());
    let at = |e: u8| hist.get(&e).copied().unwrap_or(0);
    c.line(
        "(vi)",
        at(255) == 0,
        &format!(
            "{blocks} blocks, E from {lo:?} to {hi:?} ({} distinct); blocks at E 255: {}, \
             at the subnormal scales E 0: {}, E 1: {}",
            hist.len(),
            at(255),
            at(0),
            at(1)
        ),
    );
    Ok(())
}

/// The byte table the lead pastes: per group the file bytes and the card bytes, per block
/// where the group repeats, then the borrowed target tensors and the totals.
fn byte_table(inv: &DraftInventory) {
    const GROUPS: [Group; 9] = [
        Group::Attention,
        Group::HyperConnection,
        Group::Router,
        Group::SharedExpert,
        Group::RoutedExperts,
        Group::Fc,
        Group::Markov,
        Group::Confidence,
        Group::OutputNorm,
    ];
    println!("{NAME}: bytes | group | file B | card B | per block file B");
    let (mut file, mut card) = (0u64, 0u64);
    for g in GROUPS {
        let f = inv.file_bytes(g);
        let d = inv.card_bytes(g).unwrap_or(0);
        let blocks: Vec<usize> = inv.rows.iter().filter_map(|r| r.want.block).collect();
        let n_blocks = blocks.iter().max().map_or(0, |m| m + 1);
        let per_block = inv
            .rows
            .iter()
            .any(|r| r.want.group == g && r.want.block.is_some());
        let per = if per_block {
            format!("{}", f / n_blocks as u64)
        } else {
            "-".to_string()
        };
        println!("{NAME}: bytes | {g:?} | {f} | {d} | {per}");
        file += f;
        card += d;
    }
    let experts = inv.file_bytes(Group::RoutedExperts);
    let experts_card = inv.card_bytes(Group::RoutedExperts).unwrap_or(0);
    println!("{NAME}: bytes | draft total | {file} | {card} | -");
    println!(
        "{NAME}: bytes | draft less experts | {} | {} | -",
        file - experts,
        card - experts_card
    );
    let mut head = 0u64;
    for b in &inv.borrowed {
        let on_card = match b.borrow {
            Borrow::Copied => b.bytes,
            Borrow::RowSource => 0,
        };
        head += on_card;
        println!(
            "{NAME}: bytes | target {} {} {:?} ({:?}) | {} | {on_card} | -",
            b.name, b.ty, b.dims, b.borrow, b.bytes
        );
    }
    println!(
        "{NAME}: bytes | resident on the draft card, option (iii) MXFP4 native | - | {} | -",
        card + head
    );
}
