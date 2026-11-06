//! The host rule on all cores: `gguf::iq_quant::quantize_row` over the resident pool (`crates/threads`), a
//! [`Backend`] that opens no card. Its bytes are the card's and ggml's (`gate-gpu-iq-quant`).

use crate::check::host_row;
use crate::run::{Backend, BackendError};
use crate::scan::row_bytes;
use gguf::iq_quant::{GridTables, IqKind};
use gguf::quant::GgmlType;
use std::sync::OnceLock;

/// The tensor being quantized: its kind, the kind's grids and the importance.
struct Bound {
    kind: IqKind,
    tables: GridTables,
    qw: Vec<f32>,
}

/// The host rule as a [`Backend`].
#[derive(Default)]
pub struct HostBackend {
    bound: Option<Bound>,
}

impl HostBackend {
    #[must_use]
    pub fn new() -> HostBackend {
        HostBackend::default()
    }
}

impl Backend for HostBackend {
    fn name(&self) -> String {
        format!("host rule, {} threads", threads::pool().threads())
    }

    fn bind(&mut self, kind: IqKind, qw: &[f32]) -> Result<(), BackendError> {
        let tables = match self.bound.take() {
            Some(b) if b.kind == kind => b.tables,
            _ => GridTables::new(kind),
        };
        self.bound = Some(Bound {
            kind,
            tables,
            qw: qw.to_vec(),
        });
        Ok(())
    }

    fn quantize(
        &self,
        ty: GgmlType,
        src: &[u8],
        ne0: usize,
        row0: usize,
        rows_per_expert: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let b = self.bound.as_ref().ok_or("quantize before bind")?;
        let rb = row_bytes(ty, ne0)
            .ok_or_else(|| format!("{ty:?} rows of {ne0} values are not read"))?;
        let nrows = src.len() / rb;
        let slots: Vec<OnceLock<Result<Vec<u8>, String>>> =
            (0..nrows).map(|_| OnceLock::new()).collect();
        threads::pool().for_each_chunk(nrows, |range| {
            for r in range {
                let g = row0 + r;
                let e = g / rows_per_expert;
                let out = host_row(
                    b.kind,
                    &b.tables,
                    ty,
                    &src[r * rb..(r + 1) * rb],
                    &b.qw[e * ne0..(e + 1) * ne0],
                )
                .map_err(|x| format!("expert {e} row {}: {x}", g % rows_per_expert));
                let _ = slots[r].set(out);
            }
        });
        let mut out = Vec::with_capacity(nrows * ne0 / 256 * b.kind.block_bytes());
        for s in slots {
            out.extend(s.into_inner().ok_or("a row was not quantized")??);
        }
        Ok(out)
    }
}
