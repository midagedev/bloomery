//! The logits band one f32 reassociation of a routed pass is held to
//! against another: the V4.1 prompt batch's card experts run the grouped
//! GEMM (`bloomery_gpu::gemm::enqueue_gemm_batch`) while the decode steps
//! (and the streamed passes' tile path) keep the steps' order, and both
//! fold the same exact integer block sums, so a batched call's logits sit
//! at a pinned distance from the other order's, not on its bits. The band
//! is `gate_deepseek41_prefill`'s rule (its module doc); the gates that
//! compare a batched call with another order of the same call defer to it
//! here rather than carrying a copy.

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
    aligned: bool,
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
    /// the reference's values. Both rows are f32 bits.
    pub fn add(&mut self, got: &[u32], want: &[u32]) {
        self.aligned &= got.len() == want.len();
        // The reference's top two decide whether the position is clear;
        // the first maximum wins, so ties go to the lower id.
        let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &b in want {
            let v = f32::from_bits(b);
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        let clear = top1 - top2 >= MARGIN_FLOOR;
        self.clear += usize::from(clear);
        if clear && argmax(got) != argmax(want) {
            self.disagree += 1;
        }
        for (&g, &w) in got.iter().zip(want) {
            let d = (f64::from(f32::from_bits(g)) - f64::from(f32::from_bits(w))).abs();
            self.max = self.max.max(d);
            self.sq += d * d;
        }
        self.values += got.len().min(want.len());
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

/// The row's argmax in the reference's order, logit descending and id
/// ascending: the first maximum. The row is f32 bits.
pub fn argmax(row: &[u32]) -> usize {
    let mut best = 0;
    for (i, &b) in row.iter().enumerate() {
        if f32::from_bits(b) > f32::from_bits(row[best]) {
            best = i;
        }
    }
    best
}
