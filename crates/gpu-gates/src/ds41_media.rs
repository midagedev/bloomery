//! The V4.1 image input's span assembly — one owner for the media gate
//! (`gate_ds41_media`) and the serve seat (`serve_seats::ds41` under
//! `--mmproj`): the learned delimiter rows of an encoder file and the span's
//! rows and kinds. The engine side of a span
//! ([`bloomery_gpu_deepseek41::body::Span41`]) carries what these build;
//! nothing here touches a card. The encoder's card bytes are
//! `vision::arch::deepseek41v::card`'s.

use bloomery_gpu_deepseek41::body::MediaKind;
use gguf::Gguf;
use gguf::quant::GgmlType;
use vision::arch::deepseek41v::names;

use crate::GateError;

/// The learned delimiter rows of an encoder file, cast to the span's bf16 as
/// the reference's merge casts them (`to(h.dtype)`, RTNE): the file carries
/// them F32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delims {
    /// `v.token_embd.img_start` — the span's first position.
    pub start: Vec<u16>,
    /// `v.image_newline` — every token row's last position.
    pub newline: Vec<u16>,
    /// `v.token_embd.img_end` — the span's last position.
    pub end: Vec<u16>,
}

/// The delimiter rows of `file`, by name, each refused by name when it is not
/// F32 of the model's `n_embd` ([`Delims`]'s owner: vision's deepseek41v
/// names).
pub fn delims(file: &Gguf, n_embd: usize) -> Result<Delims, GateError> {
    let delimiter = |name: String| -> Result<Vec<u16>, GateError> {
        let t = file
            .find(&name)
            .ok_or_else(|| format!("{name}: not in the encoder file"))?;
        if t.dims != [n_embd as u64] || t.ty != GgmlType::F32 {
            return Err(format!("{name} is {:?} {:?}, want F32 [{n_embd}]", t.ty, t.dims).into());
        }
        Ok(file
            .data(t)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| cuda_device::convert::f32_to_bf16_rne(f32::from_le_bytes(*c)))
            .collect())
    };
    Ok(Delims {
        start: delimiter(names::img_start())?,
        newline: delimiter(names::image_newline())?,
        end: delimiter(names::img_end())?,
    })
}

/// One image's span, the reference's splice (`model.py:1235-1239`): a Start
/// row, each grid row's `w` aligner rows and a NewLine row, an End row — the
/// rows (`n_embd` bf16 values each) and the kinds of its `2 + h·(w + 1)`
/// positions.
pub fn span_rows(
    aligner: &[u16],
    (h, w): (usize, usize),
    n_embd: usize,
    d: &Delims,
) -> Result<(Vec<u16>, Vec<MediaKind>), GateError> {
    if aligner.len() != h * w * n_embd {
        return Err(format!(
            "{} aligner values for a {h}x{w} grid of {n_embd}, want {}",
            aligner.len(),
            h * w * n_embd
        )
        .into());
    }
    let mut rows = Vec::with_capacity((2 + h * (w + 1)) * n_embd);
    let mut kinds = Vec::with_capacity(2 + h * (w + 1));
    let mut push = |row: &[u16], kind: MediaKind| {
        rows.extend_from_slice(row);
        kinds.push(kind);
    };
    push(&d.start, MediaKind::Start);
    for r in 0..h {
        for c in 0..w {
            let at = (r * w + c) * n_embd;
            push(&aligner[at..at + n_embd], MediaKind::Image);
        }
        push(&d.newline, MediaKind::NewLine);
    }
    push(&d.end, MediaKind::End);
    Ok((rows, kinds))
}
