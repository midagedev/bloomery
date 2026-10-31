//! The Qwen3.8-27B rows of the mat-vec timing probe (`--only d27`).
//!
//! The rows are the 27B's decode launches, each through the host call the
//! session makes, over A4 synthetic weights (group 64, 4 bits, the MLX layout
//! the loader produces), on a device opened at the 27B's own instance (a
//! dense FFN of 17408 values over a 5120-wide residual, 24/4 attention heads
//! of 256, a gated delta net of 48 value heads), so the pair and the fold run
//! at the one dense slot the session runs them at:
//!
//! - delta P1: segments of 10240 (q, k and v), 6144 (z), 48 (β) and 48 (α)
//!   rows, the norm on the load (`matvec_seg`);
//! - attention P1: segments of 12288 (the query with its gate), 1024 and 1024
//!   rows (`matvec_seg`);
//! - the attention out projection: 5120 by 6144 added into the stream
//!   (`matvec`, `Mode::Residual`);
//! - the dense FFN's gate-up: the experts' pair over one gate and one up of
//!   17408 by 5120 (`pair`);
//! - the dense FFN's down: 17408 columns by 5120 rows through the experts' fold
//!   at its one slot for comparison, and through `matvec` with
//!   `Mode::Residual`, the session's launch: the same sums with no per-block
//!   slot address;
//! - the head: the logits dispatch, a barrier and `logits_argmax`, over 248320
//!   rows of 5120.
//!
//! Each normed row (P1, the pair, the head) is also timed through its scaled
//! entry (`[scaled]`: the input already `x · gain`, the norm's scale one word,
//! what a hand-on feeds), so the norm on load's term is the row less its
//! scaled twin. A plain store mat-vec of the delta P1's rows is the plain line
//! at that shape. Two cuts split the norm's term on the delta P1 row:
//! `[nogain]` (the gain read as 1, no gain load) and `[nosq]` (no square in
//! the lane's sum of squares). The rows, the sweep (`sweep`) and the overlap
//! arm (`overlap`) are `lfm3.rs`'s machinery.

use std::process::ExitCode;

use bloomery_metal::instance::{FfnAxes, Key, QWEN35MOE_GDN48, QWEN35MOE_GQA6};
use bloomery_metal::kernels::Quant;

use super::lfm3::{Cut, Row, expect, names, run_rows};
use super::{HeadPart, Run, Shape};

/// The 27B's residual width, dense FFN width and vocabulary.
const HIDDEN: u32 = 5120;
const FF: u32 = 17_408;
const VOCAB: usize = 248_320;

/// The delta layer's P1 segments and the attention layer's, in the order the
/// session places them.
const DELTA_SEGMENTS: [usize; 4] = [10_240, 6144, 48, 48];
const ATTN_SEGMENTS: [usize; 3] = [12_288, 1024, 1024];

/// The out projections' columns: 24 heads of 256 and 48 value heads of 128.
const OUT_COLS: usize = 6144;

/// The A4 load's gain read and multiply.
const GAIN_FIND: &str = "                gq = act4(gain + base + 4 * l);\n";

/// The gain read as 1: the load keeps its square chain and loses the gain's
/// loads.
const GAIN_ONE: &str = "                gq = float4(1.f);\n";

/// The A4 load's square chain, with the line after it so the text is A4's alone.
const SQ_FIND: &str = concat!(
    "                *sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);\n",
    "                a0 *= gq.x;\n",
);

/// No squares: the lane's sum counts its loads.
const SQ_ONE: &str = concat!(
    "                *sum_sq += 1.f;\n",
    "                a0 *= gq.x;\n",
);

static NOGAIN: Cut = Cut {
    name: "nogain",
    edits: &[(GAIN_FIND, GAIN_ONE)],
};
static NOSQ: Cut = Cut {
    name: "nosq",
    edits: &[(SQ_FIND, SQ_ONE)],
};

fn a4_seg(segments: &[usize], scaled: bool) -> Shape {
    Shape::Seg {
        quant: Quant::A4,
        cols: HIDDEN as usize,
        segments: segments.to_vec(),
        scaled,
    }
}

fn a4_matvec(rows: usize, cols: usize, resid: bool) -> Shape {
    Shape::Matvec {
        quant: Quant::A4,
        rows,
        cols,
        resid,
    }
}

fn mark(scaled: bool) -> &'static str {
    if scaled { " [scaled]" } else { "" }
}

/// The band of a normed row and of its scaled twin.
const NORM_BASIS: &str = "the norm family of the 27B profile, 128.6 to 136.9 GB/s";
const SCALED_BASIS: &str =
    "derived: the plain line, attn_out 163.6 GB/s in the profile, LFM3's scaled family 178";

/// Every row, in the order the session meets them, then the diagnostics.
fn rows() -> Vec<Row> {
    let hidden = HIDDEN as usize;
    let ff = FF as usize;
    let delta_rows: usize = DELTA_SEGMENTS.iter().sum();
    let mut rows = Vec::new();
    for scaled in [false, true] {
        let band = if scaled {
            expect(158.0, 178.0, SCALED_BASIS)
        } else {
            expect(126.0, 140.0, NORM_BASIS)
        };
        rows.push(Row {
            label: format!(
                "d27 delta_p1 seg a4 (10240+6144+48+48)x{hidden}{}",
                mark(scaled)
            ),
            shape: a4_seg(&DELTA_SEGMENTS, scaled),
            expect: band,
            cut: None,
        });
    }
    for scaled in [false, true] {
        let band = if scaled {
            expect(158.0, 178.0, SCALED_BASIS)
        } else {
            expect(126.0, 140.0, NORM_BASIS)
        };
        rows.push(Row {
            label: format!(
                "d27 attn_p1 seg a4 (12288+1024+1024)x{hidden}{}",
                mark(scaled)
            ),
            shape: a4_seg(&ATTN_SEGMENTS, scaled),
            expect: band,
            cut: None,
        });
    }
    rows.push(Row {
        label: format!("d27 plain matvec a4 {delta_rows}x{hidden} store"),
        shape: a4_matvec(delta_rows, hidden, false),
        expect: expect(160.0, 175.0, SCALED_BASIS),
        cut: None,
    });
    rows.push(Row {
        label: format!("d27 attn_out matvec a4 {hidden}x{OUT_COLS} resid"),
        shape: a4_matvec(hidden, OUT_COLS, true),
        expect: expect(158.0, 172.0, "profile attn_out 163.6 GB/s"),
        cut: None,
    });
    for scaled in [false, true] {
        let band = if scaled {
            expect(158.0, 178.0, SCALED_BASIS)
        } else {
            expect(130.0, 142.0, "profile gate_up 133.7 GB/s")
        };
        rows.push(Row {
            label: format!("d27 gate_up pair a4 1x{ff}x{hidden}{}", mark(scaled)),
            shape: Shape::DensePair {
                quant: Quant::A4,
                rows: ff,
                cols: hidden,
                scaled,
            },
            expect: band,
            cut: None,
        });
    }
    rows.push(Row {
        label: format!("d27 down fold a4 {hidden}x{ff}x1"),
        shape: Shape::DenseFold {
            quant: Quant::A4,
            rows: hidden,
            k: ff,
        },
        expect: expect(132.0, 145.0, "profile down 136.9 GB/s"),
        cut: None,
    });
    rows.push(Row {
        label: format!("d27 down matvec a4 {hidden}x{ff} resid"),
        shape: a4_matvec(hidden, ff, true),
        expect: expect(
            155.0,
            172.0,
            "derived: the plain line with no per-block slot address",
        ),
        cut: None,
    });
    for scaled in [false, true] {
        let band = if scaled {
            expect(158.0, 178.0, SCALED_BASIS)
        } else {
            expect(130.0, 142.0, "profile head 136.5 GB/s")
        };
        rows.push(Row {
            label: format!(
                "d27 head a4_head_logits+logits_argmax {VOCAB}x{hidden}{}",
                mark(scaled)
            ),
            shape: Shape::Head {
                quant: Quant::A4,
                rows: VOCAB,
                cols: hidden,
                part: HeadPart::Pair,
                scaled,
            },
            expect: band,
            cut: None,
        });
    }
    for cut in [&NOGAIN, &NOSQ] {
        rows.push(Row {
            label: format!(
                "d27 delta_p1 seg a4 (10240+6144+48+48)x{hidden} [{}]",
                cut.name
            ),
            shape: a4_seg(&DELTA_SEGMENTS, false),
            expect: None,
            cut: Some(cut),
        });
    }
    rows
}

/// The selection names the 27B rows.
pub(super) fn selected(only: &[String]) -> bool {
    names(only, "d27")
}

pub(super) fn main(run: &Run) -> ExitCode {
    let key = Key {
        hidden: HIDDEN,
        ffn: FfnAxes::Dense { ff: FF },
        attn: QWEN35MOE_GQA6.key.attn,
        delta: QWEN35MOE_GDN48.key.delta,
    };
    run_rows(run, key, "d27", rows())
}
