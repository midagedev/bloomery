//! The embedding clauses: the Q4_K table's rows as the dequantizer's bits,
//! and the table window the launch refuses; and rows of a width that is not
//! whole groups of the launch's threads, every value written and none past
//! the rows.

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::kernels::{EmbedParams, Quant};

use super::Harness;
use super::common::{self, BadArg, LAYER, refuse_launches, still};
use super::fixture::Fixture;
use super::{oracle, synthetic};

/// What the output holds before the launch: no value of a table.
const MARKER: f32 = 12345.0;

/// The values past the output window, the marker before the launch and after
/// it: a group of the launch's threads, so a thread past the last value that
/// writes lands in them.
const SLACK: usize = 256;

/// Embed `ids` from `table` (`vocab` rows of `cols` columns, of `quant`): the
/// output rows. Every value of the output starts as the marker, and the
/// values past the rows must still hold it after the launch, or the call is
/// an error by name. The clause reads the fault word itself; the launch
/// clears it before it starts.
pub fn run_embed(
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
    let values = ids.len() * cols;
    let mut out = device
        .upload(stream, &vec![MARKER; values + SLACK])
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
        out: out.window_mut(Elems(0), Elems(values)).map_err(why)?,
        fault: device.fault_word(),
        layer: LAYER,
    };
    h.encode(|encoder| h.kernels.embed(encoder, params))?;
    let mut rows = vec![0f32; values + SLACK];
    device
        .read(stream, out.whole().map_err(why)?, &mut rows)
        .map_err(why)?;
    if let Some(k) = rows[values..]
        .iter()
        .position(|v| v.to_bits() != MARKER.to_bits())
    {
        return Err(format!(
            "{quant:?}, {} ids of {cols} columns: value {} past the rows is {:?}, written",
            ids.len(),
            values + k,
            rows[values + k]
        ));
    }
    rows.truncate(values);
    Ok(rows)
}

/// Widths of whole 32- and 64-value blocks that are not whole groups of the
/// launch's 256 threads: Qwen2.5-0.5B's hidden width (three groups and a
/// half), and a row under one group.
const PART_GROUP_COLS: [usize; 2] = [896, 64];

/// The rows of the synthetic tables of [`embed_part_groups`].
const PART_GROUP_VOCAB: usize = 16;

/// Every type with an embedding entry whose blocks divide a width of
/// [`PART_GROUP_COLS`] (A4, A8, A4H, A8H and Q8_0; the 256-value blocks of
/// Q4_K and Q5_K make every width whole groups) embeds rows of that width
/// from a synthetic table: one id, the step's launch, and three, a prompt
/// pass's rows, whose values end inside a group too. Every value is the
/// reader's, bit for bit, the word stays clean, and [`run_embed`] holds the
/// values past the rows untouched.
pub fn embed_part_groups(h: &Harness<'_>) -> Clause {
    let ids: [&[u32]; 2] = [&[5], &[0, 15, 7]];
    let mut types = Vec::new();
    for quant in Quant::ALL.into_iter().filter(|q| q.embeds()) {
        let widths: Vec<usize> = PART_GROUP_COLS
            .into_iter()
            .filter(|c| c.is_multiple_of(quant.values_per_block()))
            .collect();
        if widths.is_empty() {
            continue;
        }
        for cols in widths {
            let seed = ((quant.block_bytes() as u64) << 24) + cols as u64;
            let table = synthetic::matrix(quant, PART_GROUP_VOCAB, cols, seed);
            for ids in ids {
                let what = format!("{quant:?}, {} ids of {cols} columns", ids.len());
                let rows = run_embed(h, quant, &table, PART_GROUP_VOCAB, cols, ids)?;
                common::expect_clean(h, &what)?;
                for (t, &id) in ids.iter().enumerate() {
                    let want = oracle::table_row(quant, &table, cols, id as usize)?;
                    let row = &rows[t * cols..(t + 1) * cols];
                    if let Some(k) = row
                        .iter()
                        .zip(&want)
                        .position(|(g, w)| g.to_bits() != w.to_bits())
                    {
                        return Err(format!(
                            "{what}: id {id} value {k}: got {:?}, the reader has {:?}",
                            row[k], want[k]
                        ));
                    }
                }
            }
        }
        types.push(format!("{quant:?}"));
    }
    Ok(Verdict::Noted(format!(
        "{} at {PART_GROUP_COLS:?} columns, {} and {} ids: every value the reader's, none past the rows",
        types.join(", "),
        ids[0].len(),
        ids[1].len()
    )))
}

/// The buffers of the odd-table refusal: a table whose window starts at an
/// odd byte, and the output it must leave alone.
struct OddTable {
    table: bloomery_metal::MetalBuffer<u8>,
    ids: bloomery_metal::MetalBuffer<u32>,
    out: bloomery_metal::MetalBuffer<f32>,
    vocab: usize,
    cols: usize,
    bytes: usize,
}

/// A table window that starts at an odd byte offset is refused by name, and
/// nothing is encoded for it: the kernel reads the table as 16-bit words. The
/// well-formed launch is the clause's own ids-in-range launch, whose rows are
/// held bit-equal to the reader.
fn refuses_an_odd_table(h: &Harness<'_>) -> Result<(), String> {
    let (device, stream) = (h.device, h.stream);
    let (vocab, cols) = (2usize, 2048usize);
    let bytes = vocab * cols / Quant::Q4K.values_per_block() * Quant::Q4K.block_bytes();
    let mut bufs = OddTable {
        table: device.upload(stream, &vec![0u8; bytes + 1]).map_err(why)?,
        ids: device.upload(stream, &[0u32]).map_err(why)?,
        out: device.upload(stream, &vec![1f32; cols]).map_err(why)?,
        vocab,
        cols,
        bytes,
    };
    let table_at = 1usize;
    let case = BadArg::new(
        "a table at an odd byte offset",
        "16-bit words",
        Box::new(
            move |b: &mut OddTable, encoder: &mut bloomery_metal::encode::ComputeEncoder<'_>| {
                h.kernels.embed(
                    encoder,
                    EmbedParams {
                        quant: Quant::Q4K,
                        table: b.table.window(Elems(table_at), Elems(b.bytes))?,
                        vocab: b.vocab,
                        cols: b.cols,
                        ids: b.ids.window(Elems(0), Elems(1))?,
                        out: b.out.window_mut(Elems(0), Elems(b.cols))?,
                        fault: h.device.fault_word(),
                        layer: LAYER,
                    },
                )
            },
        ),
    );
    refuse_launches(h, "embed", &mut bufs, vec![case], |h, b| {
        still(h, &b.out, |v| *v == 1.0)
    })?;
    Ok(())
}

pub fn embed(h: &Harness<'_>, fixture: &Fixture) -> Clause {
    let (table, vocab, ty) = fixture.table("token_embd.weight", 2048)?;
    let quant = match ty {
        gguf::quant::GgmlType::Q4_K => Quant::Q4K,
        gguf::quant::GgmlType::MLX_A4 => Quant::A4,
        other => return Err(format!("token_embd.weight is {other:?}, not a table type")),
    };
    let checked = embed_table(h, quant, table, vocab)?;
    refuses_an_odd_table(h)?;
    Ok(Verdict::Noted(format!(
        "{checked}, a table at an odd byte refused"
    )))
}

/// `table` (`vocab` rows of 2048 columns, of `quant`) embedded: ids in range
/// bit-equal to the reader's rows with a clean fault word, and ids at and past
/// the vocabulary raising the token-id site with NaN rows, the ids beside them
/// still the reader's. The note says what was held.
pub fn embed_table(
    h: &Harness<'_>,
    quant: Quant,
    table: &[u8],
    vocab: usize,
) -> Result<String, String> {
    let cols = 2048;

    // Ids in range: the dequantizer's bits, row by row, and a clean word.
    let good = [0u32, 1, 5, 4095, (vocab - 1) as u32];
    let rows = run_embed(h, quant, table, vocab, cols, &good)?;
    common::expect_clean(h, "ids in range")?;
    for (t, &id) in good.iter().enumerate() {
        let want = oracle::table_row(quant, table, cols, id as usize)?;
        for (k, (g, w)) in rows[t * cols..(t + 1) * cols].iter().zip(&want).enumerate() {
            if g.to_bits() != w.to_bits() {
                return Err(format!(
                    "id {id} value {k}: got {g:?}, the reader has {w:?}"
                ));
            }
        }
    }

    // Ids at and past the vocabulary: the site is raised, the row is NaN, and
    // the ids around them are still right.
    let mixed = [7u32, vocab as u32, 9, u32::MAX, (vocab - 1) as u32];
    let rows = run_embed(h, quant, table, vocab, cols, &mixed)?;
    common::expect_fault(h, "ids past the vocabulary", FaultSite::TokenId)?;
    h.clear_fault()?;
    for (t, &id) in mixed.iter().enumerate() {
        let row = &rows[t * cols..(t + 1) * cols];
        if (id as usize) < vocab {
            let want = oracle::table_row(quant, table, cols, id as usize)?;
            if let Some(k) = row
                .iter()
                .zip(&want)
                .position(|(g, w)| g.to_bits() != w.to_bits())
            {
                return Err(format!(
                    "id {id} beside a bad id, value {k}: got {:?}",
                    row[k]
                ));
            }
        } else if let Some(k) = row.iter().position(|v| !v.is_nan()) {
            return Err(format!(
                "id {id} (vocab {vocab}) value {k} is {:?}, not NaN",
                row[k]
            ));
        }
    }
    Ok(format!(
        "{} ids bit-equal to the reader, ids {} and {} raised token_id with NaN rows",
        good.len(),
        vocab,
        u32::MAX
    ))
}
