//! The MTP draft's input join (`mtp_join`): the embedding row and the hidden
//! row of one draft step, each normed by its own gain and laid side by side,
//! the hidden row optionally normed by the target's output norm first.
//!
//! The oracle is the program the join fuses: [`Kernels::norm_rows`] launched
//! over one row for each norm and the halves laid by the host, a program
//! `norm_rows` holds to the decode router's normed input bit for bit. The join
//! is held to that program's bits, not to a band: its sums are `norm_rows'`
//! sums in the same order, and the value it writes is `norm_rows'` product
//! (`(x · gain) · scale`, each rounded once). A mutant that skips the output
//! norm, lays the halves the other way round, or swaps the two gains writes
//! other bits at every value of the row it touches, so none of them is
//! hidden by a band.

use bloomery_device::conformance::{Clause, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, Elems};
use bloomery_metal::kernels::{HiddenRow, MtpJoinParams};

use super::clauses_rows::{norm_rows_run_at, same_bits};
use super::common::{self, LAYER};
use super::oracle;
use super::{Harness, SENTINEL_BITS};

/// The epsilon every join of this clause takes: the Qwen3.8 family's.
const EPS: f32 = 1e-6;

/// The widths: the 27B's residual, and a width past the norm's tile so the
/// last pass of the threads is short.
fn widths(h: &Harness<'_>) -> [usize; 2] {
    let tile = h.kernels.router_threads() * 4;
    [5120, 5120usize.next_multiple_of(tile) + tile / 2]
}

/// One input of a join: its three gains, and the embedding and hidden rows.
struct Input {
    emb: Vec<f32>,
    hid: Vec<f32>,
    gain_emb: Vec<f32>,
    gain_hid: Vec<f32>,
    gain_out: Vec<f32>,
}

impl Input {
    /// Rows a mean square far from one, and three gains that differ in every
    /// value, so a swapped half or a swapped gain is not a no-op.
    fn drawn(cols: usize, seed: u64) -> Input {
        let gain = |s: u64| -> Vec<f32> {
            oracle::activations(cols, seed + s, 0.5)
                .into_iter()
                .map(|g| 1.0 + g)
                .collect()
        };
        Input {
            emb: oracle::activations(cols, seed + 1, 3.7),
            hid: oracle::activations(cols, seed + 2, 2.1),
            gain_emb: gain(3),
            gain_hid: gain(4),
            gain_out: gain(5),
        }
    }
}

/// The join's output over `input`: all `2 * cols` values, and the fault word
/// as the launch left it.
fn run(
    h: &Harness<'_>,
    input: &Input,
    hidden: HiddenRow,
) -> Result<(Vec<f32>, Option<bloomery_device::Fault>), String> {
    let (device, stream) = (h.device, h.stream);
    let cols = input.emb.len();
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let up = |v: &[f32]| device.upload(stream, v).map_err(why);
    let (emb, hid) = (up(&input.emb)?, up(&input.hid)?);
    let (ge, gh, go) = (
        up(&input.gain_emb)?,
        up(&input.gain_hid)?,
        up(&input.gain_out)?,
    );
    let mut y = up(&vec![sentinel; 2 * cols])?;
    h.clear_fault()?;
    h.encode(|encoder| {
        h.kernels.mtp_join(
            encoder,
            MtpJoinParams {
                cols,
                emb: emb.window(Elems(0), Elems(cols))?,
                hid: hid.window(Elems(0), Elems(cols))?,
                gain_emb: ge.window(Elems(0), Elems(cols))?,
                gain_hid: gh.window(Elems(0), Elems(cols))?,
                gain_out: go.window(Elems(0), Elems(cols))?,
                hidden,
                y: y.window_mut(Elems(0), Elems(2 * cols))?,
                eps: EPS,
                fault: device.fault_word(),
                layer: LAYER,
            },
        )
    })?;
    let mut out = vec![0f32; 2 * cols];
    device
        .read(stream, y.whole().map_err(why)?, &mut out)
        .map_err(why)?;
    Ok((out, h.fault()?))
}

/// What the program the join fuses writes: `norm_rows` over the embedding
/// row, and over the hidden row (after the output norm for the post row), the
/// halves laid in the join's order.
fn composed(h: &Harness<'_>, input: &Input, hidden: HiddenRow) -> Result<Vec<f32>, String> {
    let cols = input.emb.len();
    let norm = |x: &[f32], gain: &[f32]| -> Result<Vec<f32>, String> {
        let (y, fault) = norm_rows_run_at(h, x, gain, 1, cols, (cols, cols), EPS)?;
        match fault {
            None => Ok(y),
            Some(f) => Err(format!("the oracle's norm raised {f}")),
        }
    };
    let emb = norm(&input.emb, &input.gain_emb)?;
    let hid = match hidden {
        HiddenRow::Pre => norm(&input.hid, &input.gain_hid)?,
        HiddenRow::Post => norm(&norm(&input.hid, &input.gain_out)?, &input.gain_hid)?,
    };
    Ok([emb, hid].concat())
}

/// The join of a pre-norm and of a post-norm hidden row is the composed
/// norms' bits, at the 27B's width and past the norm's tile; the two hidden
/// rows differ (the clause tells them apart), and no fault is raised.
pub fn mtp_join(h: &Harness<'_>) -> Clause {
    let mut parts = Vec::new();
    for (w, cols) in widths(h).into_iter().enumerate() {
        let input = Input::drawn(cols, 0x4D70 + 16 * w as u64);
        let mut halves = Vec::new();
        for (name, hidden) in [("pre", HiddenRow::Pre), ("post", HiddenRow::Post)] {
            let (got, fault) = run(h, &input, hidden)?;
            if let Some(f) = fault {
                return Err(format!("{cols} columns, {name}: the join raised {f}"));
            }
            let want = composed(h, &input, hidden)?;
            same_bits(&format!("join {name}"), &got, &want)
                .map_err(|e| format!("{cols} columns: {e}"))?;
            halves.push(got[cols..].to_vec());
        }
        if halves[0]
            .iter()
            .zip(&halves[1])
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(format!(
                "{cols} columns: the pre and the post hidden half are the same bits, so this clause cannot tell the output norm from none"
            ));
        }
        parts.push(cols.to_string());
    }
    common::noted(vec![format!(
        "pre and post hidden rows at {} columns are the composed norms' bits",
        parts.join(" and ")
    )])
}

/// A half the norm cannot take is NaN and raises the norm site; the other
/// half is the composed norm's bits.
pub fn mtp_join_nan(h: &Harness<'_>) -> Clause {
    let cols = 5120;
    let clean = Input::drawn(cols, 0x4D90);
    for (name, hidden) in [("pre", HiddenRow::Pre), ("post", HiddenRow::Post)] {
        let want = composed(h, &clean, hidden)?;
        let mut bad_emb = Input::drawn(cols, 0x4D90);
        bad_emb.emb[17] = f32::NAN;
        let (got, _) = run(h, &bad_emb, hidden)?;
        common::expect_fault(h, "a NaN embedding value", FaultSite::NormQuant)?;
        if let Some(i) = got[..cols].iter().position(|v| !v.is_nan()) {
            return Err(format!("{name}: embedding half value {i} is not NaN"));
        }
        same_bits(
            &format!("{name} hidden half beside a NaN embedding"),
            &got[cols..],
            &want[cols..],
        )?;
        let mut bad_hid = Input::drawn(cols, 0x4D90);
        bad_hid.hid[cols - 1] = f32::INFINITY;
        let (got, _) = run(h, &bad_hid, hidden)?;
        common::expect_fault(h, "an infinite hidden value", FaultSite::NormQuant)?;
        if let Some(i) = got[cols..].iter().position(|v| !v.is_nan()) {
            return Err(format!("{name}: hidden half value {i} is not NaN"));
        }
        same_bits(
            &format!("{name} embedding half beside an infinite hidden row"),
            &got[..cols],
            &want[..cols],
        )?;
    }
    h.clear_fault()?;
    common::noted(vec![
        "a NaN embedding row and an infinite hidden row each raise the norm site and leave their own half NaN and the other half the composed norm's bits".to_string(),
    ])
}

/// A width that is not whole float4s, an output one float4 short and an
/// epsilon that is not finite are refused by name and encode nothing.
pub fn mtp_join_refused(h: &Harness<'_>) -> Clause {
    let cols = 5120;
    let input = Input::drawn(cols, 0x4DA0);
    let (device, stream) = (h.device, h.stream);
    let up = |v: &[f32]| device.upload(stream, v).map_err(why);
    let (emb, hid, ge, gh, go) = (
        up(&input.emb)?,
        up(&input.hid)?,
        up(&input.gain_emb)?,
        up(&input.gain_hid)?,
        up(&input.gain_out)?,
    );
    let mut y = up(&vec![0f32; 2 * cols])?;
    let mut refused = Vec::new();
    for (what, cols_arg, y_len, eps, expect) in [
        (
            "cols not a multiple of four",
            cols + 2,
            2 * cols,
            EPS,
            "multiple of four",
        ),
        ("a short output", cols, 2 * cols - 4, EPS, "y"),
        (
            "an infinite epsilon",
            cols,
            2 * cols,
            f32::INFINITY,
            "epsilon",
        ),
    ] {
        let got = h.encode(|encoder| {
            h.kernels.mtp_join(
                encoder,
                MtpJoinParams {
                    cols: cols_arg,
                    emb: emb.window(Elems(0), Elems(cols))?,
                    hid: hid.window(Elems(0), Elems(cols))?,
                    gain_emb: ge.window(Elems(0), Elems(cols))?,
                    gain_hid: gh.window(Elems(0), Elems(cols))?,
                    gain_out: go.window(Elems(0), Elems(cols))?,
                    hidden: HiddenRow::Post,
                    y: y.window_mut(Elems(0), Elems(y_len))?,
                    eps,
                    fault: device.fault_word(),
                    layer: LAYER,
                },
            )
        });
        match got {
            Err(e) if e.contains(expect) => refused.push(what),
            Err(e) => {
                return Err(format!(
                    "{what}: refused as {e:?}, which does not name {expect:?}"
                ));
            }
            Ok(()) => return Err(format!("{what}: the join was encoded")),
        }
    }
    common::noted(vec![format!("refused by name: {}", refused.join(", "))])
}

/// The clauses of the draft's join, in the order the gate runs them.
pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, super::ClauseRun<'a>)> {
    vec![
        (
            "mtp_join",
            Box::new(move || mtp_join(h)) as super::ClauseRun<'a>,
        ),
        ("mtp_join_nan", Box::new(move || mtp_join_nan(h))),
        ("mtp_join_refused", Box::new(move || mtp_join_refused(h))),
    ]
}
