//! The host's rules on all cores: the IQ search over the resident pool (`crates/threads`) as a [`Backend`] that
//! opens no card ([`HostBackend`]), and the K-quant rules and the widening to f32 over the same pool
//! ([`convert_rows`]). Their bytes are ggml's (`gate-quantize-oracle`, `gate-gpu-iq-quant`).

use crate::check::host_row;
use crate::plan::Target;
use crate::run::{Backend, BackendError};
use crate::scan::row_bytes;
use gguf::iq_quant::{GridTables, IqKind};
use gguf::kquant::quantize_row;
use gguf::quant::{GgmlType, dequant_row, half_to_f32};
use std::ops::Range;
use std::sync::Mutex;

/// The tensor being quantized: its kind, the kind's grids and the importance.
struct Bound {
    kind: IqKind,
    tables: GridTables,
    qw: Vec<f32>,
}

/// The IQ host rule as a [`Backend`].
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

/// Run `f` over `n` rows split across the pool; each chunk returns its rows' bytes (or a refusal), and the chunks
/// come back joined in row order.
fn pooled(
    n: usize,
    f: impl Fn(Range<usize>) -> Result<Vec<u8>, String> + Sync,
) -> Result<Vec<u8>, String> {
    type Part = (usize, Result<Vec<u8>, String>);
    let parts: Mutex<Vec<Part>> = Mutex::new(Vec::new());
    threads::pool().for_each_chunk(n, |range| {
        if range.is_empty() {
            return;
        }
        let start = range.start;
        let r = f(range);
        parts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((start, r));
    });
    let mut parts = parts.into_inner().unwrap_or_else(|e| e.into_inner());
    parts.sort_by_key(|p| p.0);
    let mut out = Vec::new();
    for (_, p) in parts {
        out.extend(p?);
    }
    Ok(out)
}

impl Backend for HostBackend {
    fn name(&self) -> String {
        format!("host rule, {} threads", threads::pool().threads())
    }

    fn uses_host_pool(&self) -> bool {
        true
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
        Ok(pooled(nrows, |range| {
            let mut out = Vec::new();
            for r in range {
                let g = row0 + r;
                let e = g / rows_per_expert;
                out.extend(
                    host_row(
                        b.kind,
                        &b.tables,
                        ty,
                        &src[r * rb..(r + 1) * rb],
                        &b.qw[e * ne0..(e + 1) * ne0],
                    )
                    .map_err(|x| format!("expert {e} row {}: {x}", g % rows_per_expert))?,
                );
            }
            Ok(out)
        })?)
    }
}

/// A row's `d` (and `dmin`) must be finite f16, as `ggml_validate_row_data` requires of a block it wrote: the
/// first block whose scale is not is named.
pub fn validate_blocks(dst: GgmlType, bytes: &[u8]) -> Result<(), String> {
    let (block, fields): (usize, &[usize]) = match dst {
        GgmlType::Q4_K => (144, &[0, 2]),
        GgmlType::Q5_K => (176, &[0, 2]),
        GgmlType::Q6_K => (210, &[208]),
        GgmlType::Q8_0 => (34, &[0]),
        GgmlType::IQ1_S => (50, &[0]),
        GgmlType::IQ2_XXS => (66, &[0]),
        _ => return Ok(()),
    };
    for (i, b) in bytes.chunks_exact(block).enumerate() {
        for &f in fields {
            let d = half_to_f32(u16::from_le_bytes([b[f], b[f + 1]]));
            if !d.is_finite() {
                return Err(format!("block {i}: scale {d} is not finite"));
            }
        }
    }
    Ok(())
}

/// Rows of `src` (whole rows of `ne0` values of `src_ty`, the first of them row `row0` of its tensor, `rows_per_expert`
/// rows an expert) converted to `dst` by the host: the K-quant rules, or the widening to f32. `qw` is the tensor's
/// importance, `ne0` per expert, when it has one.
pub fn convert_rows(
    target: Target,
    src_ty: GgmlType,
    src: &[u8],
    ne0: usize,
    row0: usize,
    rows_per_expert: usize,
    qw: Option<&[f32]>,
) -> Result<Vec<u8>, String> {
    let rb = row_bytes(src_ty, ne0)
        .ok_or_else(|| format!("{src_ty:?} rows of {ne0} values are not read"))?;
    let nrows = src.len() / rb;
    let out = pooled(nrows, |range| {
        let mut out = Vec::new();
        let mut x = vec![0.0f32; ne0];
        for r in range {
            let g = row0 + r;
            let e = g / rows_per_expert;
            let at = |what: &str| format!("expert {e} row {}: {what}", g % rows_per_expert);
            dequant_row(src_ty, &src[r * rb..(r + 1) * rb], &mut x)
                .map_err(|x| at(&x.to_string()))?;
            match target {
                Target::K(kt) => {
                    let w = qw.map(|q| &q[e * ne0..(e + 1) * ne0]);
                    out.extend(quantize_row(kt, &x, w).map_err(|x| at(&x.to_string()))?);
                }
                Target::F32 => {
                    if let Some((i, v)) = x.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                        return Err(at(&format!("value {i} is {v}, not finite")));
                    }
                    out.extend(x.iter().flat_map(|v| v.to_le_bytes()));
                }
                Target::Iq(_) => {
                    return Err(at("an IQ target is the backend's, not the host rules'"));
                }
            }
        }
        Ok(out)
    })?;
    if let Target::K(kt) = target {
        validate_blocks(kt.ggml(), &out)?;
    }
    Ok(out)
}

/// The threads of the resident pool the host rules run on.
#[must_use]
pub fn pool_threads() -> usize {
    threads::pool().threads()
}
