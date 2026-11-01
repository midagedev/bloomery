//! Buffer geometry of the common routed/shared FFN and prompt walk.

use super::SiteAct;
use super::attention::Forms;
use crate::placement::workstation::TierLegScratch;

/// The first unit size whose shared and dense blocks use the wide path.
pub const GATED_FROM: u64 = 9;

/// Wide quantized forms and an owned f32 row, when a selected down reads one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WideForms {
    pub quant: Forms,
    pub f32: bool,
}

impl WideForms {
    /// Include a selected site's activation form.
    #[must_use]
    pub const fn with(self, reads: SiteAct) -> Self {
        Self {
            quant: self.quant.with(reads),
            f32: self.f32 || matches!(reads, SiteAct::F32),
        }
    }

    /// Input f32 rows are borrowed from the caller.
    #[must_use]
    pub const fn without_f32(self) -> Self {
        Self { f32: false, ..self }
    }

    /// Checked allocation bytes of these forms.
    #[must_use]
    pub fn bytes(self, rows: u64, width: u64) -> Option<u64> {
        let floats = if self.f32 {
            rows.checked_mul(width)?.checked_mul(4)?
        } else {
            0
        };
        self.quant.bytes(rows, width)?.checked_add(floats)
    }
}

/// One shared or dense block's wide scratch, with the actual selected read forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideBlock {
    pub hidden: u64,
    pub ff: u64,
    pub input: WideForms,
    pub down: WideForms,
}

impl WideBlock {
    /// Route table, borrowed-input forms, owned activation forms, and gate/up outputs.
    #[must_use]
    pub fn bytes(self, rows: u64) -> Option<u64> {
        dense_route_bytes(rows)?
            .checked_add(self.input.without_f32().bytes(rows, self.hidden)?)?
            .checked_add(self.down.bytes(rows, self.ff)?)?
            .checked_add(rows.checked_mul(self.ff)?.checked_mul(8)?)
    }
}

/// A single-expert wide route table: slot ids, tiles, counts and zero ids.
#[must_use]
pub const fn dense_route_bytes(rows: u64) -> Option<u64> {
    let bytes = (2 * rows as u128 + 2 * (rows as u128 / 64 + 1) + 2) * 4;
    if bytes > u64::MAX as u128 {
        None
    } else {
        Some(bytes as u64)
    }
}

/// The common card batch beside its leg: accumulation/base rows and selected places.
#[must_use]
pub fn card_batch_bytes(leg: TierLegScratch) -> Option<u64> {
    leg.bytes(leg.downs)
        .ok()?
        .checked_add(leg.chunk.checked_mul(leg.n_embd)?.checked_mul(8)?)?
        .checked_add(leg.chunk.checked_mul(leg.n_used)?.checked_mul(4)?)
}

/// Places, join places and join weights of a prompt unit on an expert-tier load.
#[must_use]
pub const fn tier_places_bytes(rows: u64, used: u64) -> Option<u64> {
    match rows.checked_mul(used) {
        Some(slots) => slots.checked_mul(12),
        None => None,
    }
}

/// Exact buffers shared by all units of the routed/shared prompt walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedBatch {
    pub hidden: u64,
    pub experts: u64,
    pub used: u64,
    pub shared: WideBlock,
    pub dense: Option<WideBlock>,
    pub card: Option<TierLegScratch>,
    pub tiered: bool,
}

impl RoutedBatch {
    /// Norm/router/shared/base rows, narrow activation rows, optional dense/card/wide/tier buffers.
    #[must_use]
    pub fn bytes(self, rows: u64) -> Option<u64> {
        let dense_ff = self.dense.map_or(0, |d| d.ff);
        let mut values = self
            .hidden
            .checked_mul(3)?
            .checked_add(self.experts)?
            .checked_add(self.used.checked_mul(2)?)?
            .checked_add(self.shared.ff.max(dense_ff))?;
        if self.dense.is_some() {
            values = values.checked_add(self.hidden.checked_mul(2)?)?;
        }
        let mut bytes = rows.checked_mul(values)?.checked_mul(4)?;
        if rows >= GATED_FROM {
            bytes = bytes.checked_add(self.shared.bytes(rows)?)?;
            if let Some(dense) = self.dense {
                bytes = bytes.checked_add(dense.bytes(rows)?)?;
            }
        }
        if let Some(card) = self.card {
            if card.chunk != rows {
                return None;
            }
            bytes = bytes.checked_add(card_batch_bytes(card)?)?;
        }
        if self.tiered {
            bytes = bytes.checked_add(tier_places_bytes(rows, self.used)?)?;
        }
        Some(bytes)
    }
}

/// How a load's card route holds a run's rows between the routing and the
/// down GEMM: a model reads its file's form and counts a run token's card
/// bytes by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteForm {
    /// The gate and up GEMMs read the run's rows in the q8_1 or the 32-value
    /// form and write f32 rows.
    Rows,
    /// One native FP4 launch runs the gate, the up and the SwiGLU over the
    /// run's rows in the native FP4 form alone.
    Native,
}

/// Prompt walk units and the shared host-sum upload buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptWalk {
    pub hidden: u64,
    pub sets: u64,
}

impl PromptWalk {
    /// Each unit owns x/x1 plus positions/key counts; all units share one host-sum row set.
    #[must_use]
    pub fn bytes(self, rows: u64) -> Option<u64> {
        let unit = self.hidden.checked_mul(8)?.checked_add(8)?;
        self.sets
            .checked_mul(unit)?
            .checked_add(self.hidden.checked_mul(4)?)?
            .checked_mul(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{DownAct, DownForms};

    #[test]
    fn wide_threshold_selected_forms_and_protection_are_independent() {
        let wide = WideBlock {
            hidden: 256,
            ff: 256,
            input: WideForms::default().with(SiteAct::F32),
            down: WideForms::default().with(SiteAct::F32),
        };
        let batch = RoutedBatch {
            hidden: 256,
            experts: 8,
            used: 2,
            shared: wide,
            dense: None,
            card: None,
            tiered: false,
        };
        assert_eq!(batch.bytes(8), Some(8 * 4 * (3 * 256 + 8 + 4 + 256)));
        assert_eq!(
            batch.bytes(9),
            Some(9 * 4 * (3 * 256 + 8 + 4 + 256) + dense_route_bytes(9).unwrap() + 9 * 256 * 12)
        );
        let leg = TierLegScratch {
            n_embd: 256,
            ff: 256,
            n_used: 2,
            chunk: 9,
            downs: DownForms::of(DownAct::Q8Act),
            input_scales: false,
        };
        let direct = card_batch_bytes(leg).unwrap();
        let scaled = card_batch_bytes(TierLegScratch {
            input_scales: true,
            ..leg
        })
        .unwrap();
        assert_eq!(scaled - direct, 4 * 9 * 2);
        let q32 = WideBlock {
            input: WideForms::default().with(SiteAct::Q8x32),
            ..wide
        };
        assert_eq!(q32.bytes(9).unwrap() - wide.bytes(9).unwrap(), 9 * 4 * 80);
        assert_eq!(batch.bytes(u64::MAX), None);
    }

    #[test]
    fn walk_groups_only_duplicate_unit_rows() {
        let one = PromptWalk {
            hidden: 256,
            sets: 1,
        }
        .bytes(512)
        .unwrap();
        let two = PromptWalk {
            hidden: 256,
            sets: 2,
        }
        .bytes(512)
        .unwrap();
        assert_eq!(one, 512 * (12 * 256 + 8));
        assert_eq!(two - one, 512 * (8 * 256 + 8));
        assert_eq!(
            PromptWalk {
                hidden: u64::MAX,
                sets: 1
            }
            .bytes(512),
            None
        );
    }
}
