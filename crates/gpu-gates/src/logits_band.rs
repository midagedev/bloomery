//! The logits band one f32 reassociation of a routed pass is held to
//! against another: the V4.1 prompt batch's card experts run the grouped
//! GEMM (`bloomery_gpu::gemm::enqueue_gemm_batch`) while the decode steps
//! (and the streamed passes' tile path) keep the steps' order, and both
//! fold the same exact integer block sums, so a batched call's logits sit
//! at a pinned distance from the other order's, not on its bits. The band
//! is `gate_deepseek41_prefill`'s rule (its module doc); the gates that
//! compare a batched call with another order of the same call defer to it
//! here rather than carrying a copy. A comparison of two whole runs, the
//! call's last position and the steps that follow it, is [`walk`]: the
//! one walk over their positions in order that the tier and two-card gates
//! share.

use crate::prompts::MARGIN_FLOOR;

/// Clear-position top-1 disagreements the band admits between a batched
/// call's logits and the steps' run — a compared position whose
/// steps' top1−top2 margin is at least `MARGIN_FLOOR` (the e2e clarity
/// rule) where the two top-1s differ. Clear-position top-1 equality is
/// the band's hard clause: both paths reassociate the same exact integer
/// block sums, and the derived band is the first routed seam clause's.
pub const LOGITS_CLEAR_PIN: usize = 0;
/// The largest |Δlogit| the band admits between a batched call's logits
/// and the steps' run, over every compared value. The last logits
/// compound the first routed seam's distance through the remaining
/// layers, past what the sum-order model derives (the first routed seam
/// clause carries the derived band), so this pin carries the measured
/// distance with twice its headroom; it catches a broken call, not a
/// value bug (the seam clause and the clear top-1 do).
// PIN(2026-10-30): the grouped GEMM's sum order; the largest of 16 cases a tier, real 7.852 (P=1), fixture
// 1.375, x2. Not derived: attention and requantization compound the first routed seam's sum-order distance.
pub const LOGITS_MAX_PIN: f64 = 16.0;
/// The RMS |Δlogit| the band admits, over the same values. The last
/// logits compound the first routed seam's distance through the
/// remaining layers, past what the sum-order model derives, so this pin
/// carries the measured distance with twice its headroom.
// PIN(2026-10-30): the grouped GEMM's sum order; the largest of 16 cases a tier, real 1.245 (P=1), fixture
// 0.184, x2. Not derived, as the max pin.
pub const LOGITS_RMS_PIN: f64 = 2.5;

/// The band one order's logits are held to against the other's (the
/// module doc): the largest and RMS |Δlogit| over every compared
/// value, and the top-1 disagreements over the positions the reference
/// is clear about. A compared pair of unequal width, or a position one
/// side holds and the other does not, is red whatever the values — the
/// two runs read the same head over the same ids.
pub struct LogitsBand {
    /// Whether every compared position was held by both sides at the
    /// same width.
    pub aligned: bool,
    /// Positions whose reference top1−top2 margin is at least
    /// `MARGIN_FLOOR`.
    pub clear: usize,
    /// Clear positions whose top-1 is not the reference's.
    pub disagree: usize,
    /// The compared values: the two sides' overlapping positions.
    values: usize,
    /// The largest |Δlogit| over the compared values.
    pub max: f64,
    sq: f64,
}

impl LogitsBand {
    /// An empty band: nothing compared yet.
    pub fn new() -> LogitsBand {
        LogitsBand {
            aligned: true,
            clear: 0,
            disagree: 0,
            values: 0,
            max: 0.0,
            sq: 0.0,
        }
    }

    /// Fold one position in: the reference's `want` decides the clarity
    /// margin and the top-1, the other order's `got` its distance from
    /// the reference's values. Both rows are f32 bits; the top two are
    /// [`crate::kld::top2`]'s, ties to the lower id.
    pub fn add(&mut self, got: &[u32], want: &[u32]) {
        if got.len() != want.len() || got.len() < 2 {
            self.aligned = false;
            return;
        }
        let got: Vec<f32> = got.iter().copied().map(f32::from_bits).collect();
        let want: Vec<f32> = want.iter().copied().map(f32::from_bits).collect();
        let (got_id, _, _, _) = crate::kld::top2(&got);
        let (want_id, top1, _, top2) = crate::kld::top2(&want);
        let clear = top1 - top2 >= MARGIN_FLOOR;
        self.clear += usize::from(clear);
        self.disagree += usize::from(clear && got_id != want_id);
        for (&g, &w) in got.iter().zip(&want) {
            let d = (f64::from(g) - f64::from(w)).abs();
            self.max = self.max.max(d);
            self.sq += d * d;
        }
        self.values += got.len();
    }

    /// The RMS |Δlogit| over the compared values.
    pub fn rms(&self) -> f64 {
        (self.sq / self.values.max(1) as f64).sqrt()
    }

    /// Whether the measured band sits within the pins.
    pub fn ok(&self) -> bool {
        self.aligned
            && self.disagree.saturating_sub(LOGITS_CLEAR_PIN) == 0
            && self.max <= LOGITS_MAX_PIN
            && self.rms() <= LOGITS_RMS_PIN
    }
}

impl Default for LogitsBand {
    fn default() -> Self {
        Self::new()
    }
}

/// One run's positions in order: the token each position took and the
/// logits row it took it from, one of each a position.
#[derive(Clone, Copy)]
pub struct Positions<'a> {
    pub tokens: &'a [u32],
    pub rows: &'a [Vec<f32>],
}

/// What [`walk`] compared.
pub struct Walk {
    /// The band over the compared positions' rows.
    pub band: LogitsBand,
    /// The positions whose rows are in the band: each one whose earlier
    /// tokens agree, through the first whose own token differs.
    pub compared: usize,
    /// The leading positions whose tokens agree: a span that ends inside
    /// them ran on the same context in both runs.
    pub agreed: usize,
    /// Whether the band sits within its pins and neither run ends before
    /// the other while their tokens agree.
    pub ok: bool,
}

/// Walk `got` against the reference `want`, two runs of the same ids read
/// in order, and print one `ok` or `FAIL` line.
///
/// A position's row joins the band while every earlier token agrees: both
/// runs fed it the same context, so the rows differ by the reassociation
/// alone. At the first position whose tokens differ the walk stops after
/// folding that row in. It is red when the reference was clear there (a
/// disagree, held to [`LOGITS_CLEAR_PIN`]); otherwise the two orders
/// resolved a near-tie differently, the positions after it ran on other
/// contexts, and the line says how many were left. Red by name: rows of
/// unequal width, a run that ends before the other while every token
/// agrees, runs with no position.
///
/// # Panics
/// A run whose tokens and rows are not as many as each other.
pub fn walk(what: &str, want: Positions<'_>, got: Positions<'_>) -> Walk {
    for (name, run) in [("reference", want), ("compared run", got)] {
        assert_eq!(
            run.tokens.len(),
            run.rows.len(),
            "{what}: the {name} holds {} tokens and {} logits rows, one of each a position",
            run.tokens.len(),
            run.rows.len()
        );
    }
    let shared = want.tokens.len().min(got.tokens.len());
    let mut band = LogitsBand::new();
    let (mut compared, mut agreed, mut apart_clear) = (0, shared, false);
    let runs = want
        .tokens
        .iter()
        .zip(want.rows)
        .zip(got.tokens.iter().zip(got.rows));
    for (i, ((wt, wr), (gt, gr))) in runs.enumerate() {
        let (clear, disagree) = (band.clear, band.disagree);
        band.add(&bits(gr), &bits(wr));
        compared += 1;
        if wt != gt {
            apart_clear = band.clear > clear;
            // A token apart at a clear position is a disagree whatever the
            // rows' argmax reads.
            if apart_clear && band.disagree == disagree {
                band.disagree += 1;
            }
            agreed = i;
            break;
        }
    }
    let stopped = agreed < shared;
    let ends = !stopped && want.tokens.len() != got.tokens.len();
    let ok = shared > 0 && !ends && band.ok();
    let mut red: Vec<String> = Vec::new();
    if shared == 0 {
        red.push("no position to compare".into());
    }
    if !band.aligned {
        red.push("rows of unequal width".into());
    }
    if apart_clear {
        red.push(format!(
            "tokens apart at position {agreed}, where the reference is clear"
        ));
    }
    if ends {
        red.push(format!(
            "a run ends at {shared} positions with every token agreeing, the other holds {}",
            want.tokens.len().max(got.tokens.len())
        ));
    }
    if band.disagree.saturating_sub(LOGITS_CLEAR_PIN) > 0 {
        red.push("clear top-1 disagreements over the pin".into());
    }
    if band.max > LOGITS_MAX_PIN {
        red.push("max |Δlogit| over the pin".into());
    }
    if band.rms().is_nan() || band.rms() > LOGITS_RMS_PIN {
        red.push("rms |Δlogit| over the pin".into());
    }
    let left = if stopped && !apart_clear {
        format!(
            ", the tokens part at position {agreed} where the reference is not clear, so the {} \
             after it, on other contexts, are not compared",
            want.tokens.len() - agreed - 1
        )
    } else {
        String::new()
    };
    println!(
        "{} {what}: {compared} of the reference's {} positions compared{left}: band clear top-1 \
         {} of {} (pin {LOGITS_CLEAR_PIN}), max |Δlogit| {:.3e} (pin {LOGITS_MAX_PIN}), rms \
         {:.3e} (pin {LOGITS_RMS_PIN}){}",
        if ok { "ok" } else { "FAIL" },
        want.tokens.len(),
        band.disagree,
        band.clear,
        band.max,
        band.rms(),
        if red.is_empty() {
            String::new()
        } else {
            format!("; red: {}", red.join("; "))
        }
    );
    Walk {
        band,
        compared,
        agreed,
        ok,
    }
}

/// An f32 row as the bits [`LogitsBand::add`] takes.
fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-wide row whose top-1 is `top`: over a margin the band reads
    /// as clear, or, unless `clear`, under it.
    fn row(top: usize, clear: bool) -> Vec<f32> {
        let mut r = vec![0.0; 4];
        r[top] = if clear { 5.0 } else { MARGIN_FLOOR / 2.0 };
        r
    }

    fn run<'a>(tokens: &'a [u32], rows: &'a [Vec<f32>]) -> Positions<'a> {
        Positions { tokens, rows }
    }

    /// Two runs that agree compare every position; a pin overrun is red
    /// whatever the tokens say.
    #[test]
    fn walk_compares_agreeing_runs_to_their_end_within_the_pins() {
        let rows = [row(1, true), row(2, true)];
        let w = walk("same", run(&[1, 2], &rows), run(&[1, 2], &rows));
        assert!(w.ok && w.compared == 2 && w.agreed == 2 && w.band.clear == 2);
        let mut far = rows.clone();
        far[1][2] += 2.0 * LOGITS_MAX_PIN as f32;
        assert!(!walk("far", run(&[1, 2], &rows), run(&[1, 2], &far)).ok);
    }

    /// A token apart where the reference is not clear ends the walk
    /// without a red: the row at that position is in the band, the ones
    /// after it are not.
    #[test]
    fn walk_stops_at_a_token_apart_where_the_reference_is_not_clear() {
        let want = [row(1, true), row(2, false), row(3, true)];
        let got = [row(1, true), row(0, false), row(0, true)];
        let w = walk("near-tie", run(&[1, 2, 3], &want), run(&[1, 0, 0], &got));
        assert!(w.ok && w.compared == 2 && w.agreed == 1);
        assert_eq!((w.band.clear, w.band.disagree), (1, 0));
    }

    /// A token apart where the reference is clear is a disagree.
    #[test]
    fn walk_is_red_at_a_token_apart_where_the_reference_is_clear() {
        let want = [row(1, true), row(2, true)];
        let got = [row(1, true), row(0, true)];
        let w = walk("clear", run(&[1, 2], &want), run(&[1, 0], &got));
        assert!(!w.ok && w.compared == 2 && w.agreed == 1 && w.band.disagree == 1);
    }

    /// A run that ends first while the tokens agree, and rows of another
    /// width, are red by name.
    #[test]
    fn walk_is_red_on_a_run_that_ends_first_and_on_unequal_widths() {
        let rows = [row(1, true), row(2, true), row(3, true)];
        let short = walk("short", run(&[1, 2, 3], &rows), run(&[1, 2], &rows[..2]));
        assert!(!short.ok && short.compared == 2 && short.agreed == 2);
        let narrow = [vec![5.0, 0.0, 0.0]];
        let width = walk("width", run(&[1], &rows[..1]), run(&[1], &narrow));
        assert!(!width.ok && !width.band.aligned);
    }
}
