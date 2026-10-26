//! The Q5_0 and Q5_1 clauses: GGUF's Q5_0 and Q5_1 decode (`quant_b32.metal`'s
//! owners), the types of Nemotron Nano's embedding, SSM input, attention
//! q, k and v and MLP up projection (Q5_0) and of Lightning's routed
//! experts (both). Every entry the owners instantiate runs on seeded synthetic
//! rows at the two models' real shapes against the f64 oracle: the mat-vec
//! modes, the segmented mat-vec, the embedding of a whole table and the
//! GEMM staging's decode. The Nano fixture's own tensors run the same bodies
//! where the file is present (`q5_0_fixture`).

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::encode::Binding;
use bloomery_metal::kernels::{EmbedParams, Quant};
use bloomery_metal::library::Library;

use super::oracle::{self, Kind};
use super::synthetic;
use super::{ClauseCtx, ClauseRun, Harness, Launch, ModeSpec, SENTINEL_BITS, SegLaunch, fixture};

/// The layer the clauses launch in: a model layer, so a raise is told apart
/// from a launch that names none.
const LAYER: u32 = 5;

/// Nano's RMS epsilon.
const EPS: f32 = 1e-5;

/// Nano's hidden width, 98 blocks of 32 values: not a multiple of an
/// embedding threadgroup's 256 threads.
const HIDDEN: usize = 3136;

/// Both models' vocabulary.
const VOCAB: usize = 131_072;

/// The gate-side kernel that stages a table through `dequant16`.
const PROBE: &str = include_str!("dequant16_probe.metal");

/// The mat-vec shapes the store cases run: a name, the columns and the rows.
/// Nano's SSM input, attention q and k, MLP up, and Lightning's routed up and
/// down for one expert.
const SHAPES: [(&str, usize, usize); 6] = [
    ("ssm_in", HIDDEN, 17_504),
    ("attn_q", HIDDEN, 5_120),
    ("attn_k", HIDDEN, 1_024),
    ("ffn_up", HIDDEN, 12_544),
    ("ltn_up_exp", 2_688, 1_856),
    ("ltn_down_exp", 1_856, 2_688),
];

/// The clause names of each type, in gate order.
const Q5_0_NAMES: [&str; 7] = [
    "q5_0_matvec",
    "q5_0_row_tail",
    "q5_0_norm",
    "q5_0_seg",
    "q5_0_embed",
    "q5_0_odd_blocks",
    "q5_0_dequant16",
];
const Q5_1_NAMES: [&str; 7] = [
    "q5_1_matvec",
    "q5_1_row_tail",
    "q5_1_norm",
    "q5_1_seg",
    "q5_1_embed",
    "q5_1_odd_blocks",
    "q5_1_dequant16",
];

/// The clauses of both owners.
pub fn table<'a>(ctx: &'a ClauseCtx<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    let h = ctx.harness;
    let source = ctx.source;
    let mut out: Vec<(&'static str, ClauseRun<'a>)> = Vec::new();
    for (quant, names) in [(Quant::Q5_0, Q5_0_NAMES), (Quant::Q5_1, Q5_1_NAMES)] {
        let [matvec, row_tail, norm, seg, embed, odd, dequant16] = names;
        out.push((matvec, Box::new(move || matvec_clause(h, quant))));
        out.push((row_tail, Box::new(move || row_tail_clause(h, quant))));
        out.push((norm, Box::new(move || norm_clause(h, quant))));
        out.push((seg, Box::new(move || seg_clause(h, quant))));
        out.push((embed, Box::new(move || embed_clause(h, quant))));
        out.push((odd, Box::new(move || odd_blocks_clause(h, quant))));
        out.push((
            dequant16,
            Box::new(move || dequant16_clause(h, source, quant)),
        ));
    }
    out.push(("q5_0_fixture", Box::new(move || ctx.with(fixture_clause))));
    out
}

/// A quant's seed base: two quants' rows differ.
fn salt(quant: Quant) -> u64 {
    (quant.block_bytes() as u64) << 24
}

/// A positive synthetic gain: a dropped gain shows.
fn gain(cols: usize, seed: u64) -> Vec<f32> {
    oracle::activations(cols, seed, 0.5)
        .iter()
        .map(|g| 1.0 + g.abs())
        .collect()
}

fn sentinel_rows(rows: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL_BITS); rows]
}

/// The note a mat-vec clause prints: each tensor's worst error over its band.
fn noted(parts: Vec<String>) -> Clause {
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}",
        parts.join(", ")
    )))
}

/// The first index where `got` and `want` differ, with both words.
fn first_diff(got: &[f32], want: &[f32]) -> Option<(usize, f32, f32)> {
    got.iter()
        .zip(want)
        .enumerate()
        .find_map(|(i, (g, w))| (g.to_bits() != w.to_bits()).then_some((i, *g, *w)))
}

/// The fault word names `site` in [`LAYER`], and only that site.
fn expect_fault(h: &Harness<'_>, what: &str, site: FaultSite) -> Result<(), String> {
    match h.fault()? {
        Some(f)
            if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            site.name()
        )),
        None => Err(format!("{what} raised nothing, not {}", site.name())),
    }
}

/// Run `launch` and hold the output to the oracle's band: the worst ratio of
/// the error to the band.
fn run_checked(h: &Harness<'_>, launch: &Launch<'_>) -> Result<f64, String> {
    let (effective, kind, base): (Vec<f64>, Kind, Option<&[f32]>) = match &launch.mode {
        ModeSpec::Store => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Store,
            None,
        ),
        ModeSpec::Norm { gain, eps, .. } => {
            (oracle::rms_norm(launch.x, gain, *eps), Kind::Norm, None)
        }
        ModeSpec::Residual => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Residual,
            Some(launch.y0),
        ),
    };
    let reference = oracle::reference(
        launch.quant,
        launch.weights,
        launch.rows,
        launch.cols,
        &effective,
    )?;
    h.clear_fault()?;
    let got = h.matvec(launch)?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let band = oracle::band(
        launch.quant,
        launch.cols / launch.quant.values_per_block(),
        kind,
    );
    oracle::check(&got, &reference, band, kind, base)
}

/// The store dot at the models' shapes.
fn matvec_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let mut parts = Vec::new();
    for (i, &(name, cols, rows)) in SHAPES.iter().enumerate() {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + i as u64);
        let x = oracle::activations(cols, 0x5A34 + i as u64, 1.0);
        let worst = run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("{name} {rows}x{cols} {worst:.3}"));
    }
    noted(parts)
}

/// Row counts no threadgroup divides: every row written, no word past the
/// last.
fn row_tail_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let counts = [1usize, 2, 3, 5, 6, 7, 4093, 4094, 4095];
    let all = synthetic::matrix(quant, 4095, HIDDEN, salt(quant) + 0x10);
    let row_bytes = HIDDEN / quant.values_per_block() * quant.block_bytes();
    let x = oracle::activations(HIDDEN, 0x5A60, 1.0);
    for rows in counts {
        run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols: HIDDEN,
                weights: &all[..rows * row_bytes],
                x: &x,
                y0: &sentinel_rows(rows),
                mode: ModeSpec::Store,
            },
        )
        .map_err(|e| format!("{rows} rows of {HIDDEN}: {e}"))?;
    }
    Ok(Verdict::Noted(format!(
        "{} row counts, every row written, no word past the last",
        counts.len()
    )))
}

/// The norm on load at the models' shapes, a zero input with a positive
/// epsilon, the fault inputs (a NaN input, an infinite one, a zero input with
/// a zero epsilon: the norm site and NaN rows), and the residual epilogue.
fn norm_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let mut parts = Vec::new();
    for (i, &(name, cols, rows)) in [SHAPES[0], SHAPES[1], SHAPES[3], SHAPES[4]]
        .iter()
        .enumerate()
    {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x20 + i as u64);
        let g = gain(cols, 0x5AA1 + i as u64);
        let x = oracle::activations(cols, 0x5A40 + i as u64, 3.7);
        let worst = run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &sentinel_rows(rows),
                mode: ModeSpec::Norm {
                    gain: &g,
                    eps: EPS,
                    layer: LAYER,
                },
            },
        )
        .map_err(|e| format!("{name} {rows}x{cols}: {e}"))?;
        parts.push(format!("{name} {rows}x{cols} {worst:.3}"));
    }

    let (cols, rows) = (HIDDEN, 2048);
    let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x21);
    let g = gain(cols, 0x5AB0);
    // A zero input norms to zero with a positive epsilon: not a fault.
    run_checked(
        h,
        &Launch {
            quant,
            rows,
            cols,
            weights: &weights,
            x: &vec![0f32; cols],
            y0: &sentinel_rows(rows),
            mode: ModeSpec::Norm {
                gain: &g,
                eps: EPS,
                layer: LAYER,
            },
        },
    )
    .map_err(|e| format!("a zero input with eps {EPS}: {e}"))?;
    for (what, mut x, eps) in [
        ("a NaN input", oracle::activations(cols, 1, 1.0), EPS),
        ("an infinite input", oracle::activations(cols, 2, 1.0), EPS),
        ("a zero input and a zero epsilon", vec![0f32; cols], 0.0),
    ] {
        match what {
            "a NaN input" => x[17] = f32::NAN,
            "an infinite input" => x[cols - 1] = f32::INFINITY,
            _ => {}
        }
        h.clear_fault()?;
        let got = h.matvec(&Launch {
            quant,
            rows,
            cols,
            weights: &weights,
            x: &x,
            y0: &sentinel_rows(rows),
            mode: ModeSpec::Norm {
                gain: &g,
                eps,
                layer: LAYER,
            },
        })?;
        expect_fault(h, what, FaultSite::NormQuant)?;
        if let Some(r) = got.iter().position(|v| !v.is_nan()) {
            return Err(format!("{what}: row {r} is {:?}, not NaN", got[r]));
        }
    }
    h.clear_fault()?;

    // The residual epilogue: the store's rows plus the output's own.
    for (i, &(name, cols, rows)) in [SHAPES[0], SHAPES[1], SHAPES[4]].iter().enumerate() {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x28 + i as u64);
        let x = oracle::activations(cols, 0x5A50 + i as u64, 1.0);
        let resid = oracle::activations(rows, 0x5BE5 + i as u64, 2.0);
        let worst = run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights: &weights,
                x: &x,
                y0: &resid,
                mode: ModeSpec::Residual,
            },
        )
        .map_err(|e| format!("resid {name} {rows}x{cols}: {e}"))?;
        parts.push(format!("resid {name} {rows}x{cols} {worst:.3}"));
    }
    noted(parts)
}

/// A segmented dispatch of `tensors` (a name, its rows and its weights) over
/// `cols` columns holds, segment by segment, the same bits as the same tensor
/// run alone under the norm on load.
fn seg_against_alone(
    h: &Harness<'_>,
    quant: Quant,
    cols: usize,
    g: &[f32],
    eps: f32,
    x: &[f32],
    tensors: &[(&str, usize, &[u8])],
) -> Result<(), String> {
    let rows: usize = tensors.iter().map(|(_, n, _)| n).sum();
    h.clear_fault()?;
    let got = h.seg(&SegLaunch {
        quant,
        cols,
        segments: tensors.iter().map(|&(_, n, w)| (n, w)).collect(),
        x,
        y0: &sentinel_rows(rows),
        gain: g,
        eps,
        layer: LAYER,
    })?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the segmented launch raised {fault}"));
    }
    let mut at = 0usize;
    for &(name, n, bytes) in tensors {
        let alone = h.matvec(&Launch {
            quant,
            rows: n,
            cols,
            weights: bytes,
            x,
            y0: &sentinel_rows(n),
            mode: ModeSpec::Norm {
                gain: g,
                eps,
                layer: LAYER,
            },
        })?;
        if let Some((k, a, b)) = first_diff(&got[at..at + n], &alone) {
            return Err(format!(
                "{name} row {k}: the segment read {a:?}, the tensor alone read {b:?}"
            ));
        }
        at += n;
    }
    Ok(())
}

/// Every segment of a segmented dispatch is the same bits as the same tensor
/// run alone, at Nano's attention q, k and v, two wider tensors and the
/// segment table's four.
fn seg_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let g = gain(HIDDEN, 0x5E80);
    let groups: [&[(&str, usize)]; 3] = [
        &[("attn_q", 5_120), ("attn_k", 1_024), ("attn_v", 1_024)],
        &[("ssm_in", 17_504), ("ffn_up", 12_544)],
        &[("a", 1_024), ("b", 513), ("c", 1_024), ("d", 7)],
    ];
    let mut parts = Vec::new();
    for (i, tensors) in groups.iter().enumerate() {
        let x = oracle::activations(HIDDEN, 0x5E86 + i as u64, 1.0);
        let owned: Vec<Vec<u8>> = tensors
            .iter()
            .enumerate()
            .map(|(j, (_, n))| {
                synthetic::matrix(quant, *n, HIDDEN, salt(quant) + 0x30 + (8 * i + j) as u64)
            })
            .collect();
        let views: Vec<(&str, usize, &[u8])> = tensors
            .iter()
            .zip(&owned)
            .map(|(&(name, n), w)| (name, n, w.as_slice()))
            .collect();
        seg_against_alone(h, quant, HIDDEN, &g, EPS, &x, &views)?;
        parts.push(format!("{} {} segments", tensors[0].0, tensors.len()));
    }
    Ok(Verdict::Noted(format!(
        "each segment the bits of its tensor alone: {}",
        parts.join(", ")
    )))
}

/// Embed `ids` from `table` (`vocab` rows of `cols` columns, of `quant`): the
/// output rows. The clause reads the fault word itself; the launch clears it
/// before it starts.
fn run_embed(
    h: &Harness<'_>,
    quant: Quant,
    table: &[u8],
    vocab: usize,
    cols: usize,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    let (device, stream) = (h.device, h.stream);
    let table = device.upload(stream, table).map_err(why)?;
    let ids_buffer = device.upload(stream, ids).map_err(why)?;
    let marker = 12345.0f32;
    let mut out = device
        .upload(stream, &vec![marker; ids.len() * cols])
        .map_err(why)?;
    h.clear_fault()?;
    let params = EmbedParams {
        quant,
        table: table
            .window(Elems(0), Elems(table.len().get()))
            .map_err(why)?,
        vocab,
        cols,
        ids: ids_buffer.window(Elems(0), Elems(ids.len())).map_err(why)?,
        out: out
            .window_mut(Elems(0), Elems(ids.len() * cols))
            .map_err(why)?,
        fault: device.fault_word(),
        layer: LAYER,
    };
    h.encode(|encoder| h.kernels.embed(encoder, params))?;
    let mut rows = vec![0f32; ids.len() * cols];
    device
        .read(stream, out.whole().map_err(why)?, &mut rows)
        .map_err(why)?;
    Ok(rows)
}

/// `table` (`vocab` rows of `cols` columns, of `quant`) embedded: ids in
/// range bit-equal to the reader's rows with a clean fault word, and ids at
/// and past the vocabulary raising the token-id site with NaN rows, the ids
/// beside them still the reader's. Five ids in range make a value count that
/// is not a whole number of threadgroups. The note says what was held.
fn embed_table(
    h: &Harness<'_>,
    quant: Quant,
    table: &[u8],
    vocab: usize,
    cols: usize,
) -> Result<String, String> {
    let good = [0u32, 1, 5, 4095, (vocab - 1) as u32];
    let rows = run_embed(h, quant, table, vocab, cols, &good)?;
    if let Some(fault) = h.fault()? {
        return Err(format!("ids in range raised {fault}"));
    }
    for (t, &id) in good.iter().enumerate() {
        let want = oracle::table_row(quant, table, cols, id as usize)?;
        if let Some((k, g, w)) = first_diff(&rows[t * cols..(t + 1) * cols], &want) {
            return Err(format!(
                "id {id} value {k}: got {g:?}, the reader has {w:?}"
            ));
        }
    }

    let mixed = [7u32, vocab as u32, 9, u32::MAX, (vocab - 1) as u32];
    let rows = run_embed(h, quant, table, vocab, cols, &mixed)?;
    expect_fault(h, "ids past the vocabulary", FaultSite::TokenId)?;
    h.clear_fault()?;
    for (t, &id) in mixed.iter().enumerate() {
        let row = &rows[t * cols..(t + 1) * cols];
        if (id as usize) < vocab {
            let want = oracle::table_row(quant, table, cols, id as usize)?;
            if let Some((k, g, _)) = first_diff(row, &want) {
                return Err(format!("id {id} beside a bad id, value {k}: got {g:?}"));
            }
        } else if let Some(k) = row.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "id {id} (vocab {vocab}) value {k} is {:?}, not NaN",
                row[k]
            ));
        }
    }
    Ok(format!(
        "{vocab}x{cols}: {} ids bit-equal to the reader, ids {vocab} and {} raised token_id with NaN rows",
        good.len(),
        u32::MAX
    ))
}

/// The embedding over a whole-vocabulary table of Nano's hidden width.
fn embed_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let table = synthetic::matrix(quant, VOCAB, HIDDEN, salt(quant) + 0x68);
    Ok(Verdict::Noted(embed_table(
        h, quant, &table, VOCAB, HIDDEN,
    )?))
}

/// Rows of an odd number of blocks, so row starts sit at a 2-byte alignment
/// and no further: the store, the norm on load and the residual epilogue
/// against the oracle, down to a one-block row.
fn odd_blocks_clause(h: &Harness<'_>, quant: Quant) -> Clause {
    let rows = 37;
    let mut parts = Vec::new();
    for (i, cols) in [32usize, 96, 544, 3_168].into_iter().enumerate() {
        let weights = synthetic::matrix(quant, rows, cols, salt(quant) + 0x40 + i as u64);
        let x = oracle::activations(cols, 0x5A70 + i as u64, 1.0);
        let g = gain(cols, 0x5AC0 + i as u64);
        let resid = oracle::activations(rows, 0x5BF0 + i as u64, 2.0);
        for (what, mode, y0) in [
            ("store", ModeSpec::Store, sentinel_rows(rows)),
            (
                "norm",
                ModeSpec::Norm {
                    gain: &g,
                    eps: EPS,
                    layer: LAYER,
                },
                sentinel_rows(rows),
            ),
            ("resid", ModeSpec::Residual, resid),
        ] {
            let worst = run_checked(
                h,
                &Launch {
                    quant,
                    rows,
                    cols,
                    weights: &weights,
                    x: &x,
                    y0: &y0,
                    mode,
                },
            )
            .map_err(|e| format!("{what} {rows}x{cols} ({} blocks): {e}", cols / 32))?;
            parts.push(format!("{what} {} blocks {worst:.3}", cols / 32));
        }
    }
    noted(parts)
}

/// The GEMM staging's decode, `dequant16`, holds the reader's values bit for
/// bit: every 16-value chunk of a table, staged by a gate-side kernel
/// compiled from this run's source.
fn dequant16_clause(h: &Harness<'_>, source: &str, quant: Quant) -> Clause {
    let blocks = 4096usize;
    let table = synthetic::matrix(quant, blocks, 32, salt(quant) + 0x50);
    let (device, stream) = (h.device, h.stream);
    let library = Library::from_source_text(device, &format!("{source}\n{PROBE}")).map_err(why)?;
    let entry = match quant {
        Quant::Q5_0 => "b32_dequant16_q5_0",
        Quant::Q5_1 => "b32_dequant16_q5_1",
        other => return Err(format!("{} has no staging probe", other.name())),
    };
    let pipeline = library.pipeline(entry).map_err(why)?;
    let chunks = 2 * blocks;
    let input = device.upload(stream, &table).map_err(why)?;
    let mut out = device
        .upload(stream, &sentinel_rows(16 * chunks))
        .map_err(why)?;
    let count = u32::try_from(chunks).map_err(|_| "the chunk count passes u32".to_string())?;
    let threads = 64usize;
    h.encode(|encoder| {
        encoder.set_pipeline(&pipeline);
        encoder.set_binding(
            0,
            Binding::of(&input.window(Elems(0), Elems(table.len()))?)?,
        )?;
        encoder.set_binding(
            1,
            Binding::of_mut(&out.window_mut(Elems(0), Elems(16 * chunks))?)?,
        )?;
        encoder.set_binding(2, Binding::bytes(&count.to_ne_bytes())?)?;
        encoder.dispatch_groups(chunks.div_ceil(threads), 1, threads)
    })?;
    let mut back = vec![0f32; 16 * chunks];
    device
        .read(stream, out.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    for block in 0..blocks {
        let want = oracle::table_row(quant, &table, 32, block)?;
        if let Some((k, g, w)) = first_diff(&back[32 * block..32 * (block + 1)], &want) {
            return Err(format!(
                "block {block} value {k}: staged {g:?}, the reader has {w:?}"
            ));
        }
    }
    Ok(Verdict::Noted(format!(
        "{blocks} blocks, both chunks of each, the reader's bits"
    )))
}

/// The Nano fixture's real tensors: the SSM input projection (store and the
/// norm on load with the layer's own gain), attention q, k and v in one
/// segmented dispatch, and the embedding table.
fn fixture_clause(h: &Harness<'_>, fx: &fixture::Fixture) -> Clause {
    let quant = Quant::Q5_0;
    let eps = fx.rms_eps()?;
    let mut parts = Vec::new();

    let (cols, rows) = (HIDDEN, 17_504);
    let weights = fx.rows("blk.0.ssm_in.weight", quant, cols, rows)?;
    let g = fx.f32s("blk.0.attn_norm.weight", cols)?;
    let x = oracle::activations(cols, 0xF1A0, 1.0);
    for (what, mode) in [
        ("store", ModeSpec::Store),
        (
            "norm",
            ModeSpec::Norm {
                gain: &g,
                eps,
                layer: LAYER,
            },
        ),
    ] {
        let worst = run_checked(
            h,
            &Launch {
                quant,
                rows,
                cols,
                weights,
                x: &x,
                y0: &sentinel_rows(rows),
                mode,
            },
        )
        .map_err(|e| format!("blk.0.ssm_in {what} {rows}x{cols}: {e}"))?;
        parts.push(format!("ssm_in {what} {worst:.3}"));
    }

    let g12 = fx.f32s("blk.12.attn_norm.weight", cols)?;
    let q = fx.rows("blk.12.attn_q.weight", quant, cols, 5_120)?;
    let k = fx.rows("blk.12.attn_k.weight", quant, cols, 1_024)?;
    let v = fx.rows("blk.12.attn_v.weight", quant, cols, 1_024)?;
    seg_against_alone(
        h,
        quant,
        cols,
        &g12,
        eps,
        &x,
        &[
            ("attn_q", 5_120, q),
            ("attn_k", 1_024, k),
            ("attn_v", 1_024, v),
        ],
    )?;
    parts.push("attn q/k/v 3 segments".to_string());

    let (table, vocab) = fx.table("token_embd.weight", quant, cols)?;
    parts.push(embed_table(h, quant, table, vocab, cols)?);
    Ok(Verdict::Noted(parts.join(", ")))
}
