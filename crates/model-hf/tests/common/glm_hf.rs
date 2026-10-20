use std::collections::BTreeMap;
use std::io::Read;

use hf::quant_config::{Config, plan};
use hf::safetensors::Dtype;
use serde_json::Value;

use super::{Geometry, Layout};

pub(super) const GLM: Geometry = Geometry {
    layers: 45,
    heads: 64,
    nope: 256,
    v: 256,
    experts: 288,
};

pub(super) fn gunzip(path: &str) -> Value {
    let gz = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = Vec::new();
    flate2::read::GzDecoder::new(gz.as_slice())
        .read_to_end(&mut text)
        .unwrap();
    serde_json::from_slice(&text).unwrap()
}

/// The checkpoint's placed tensors with their logical shapes: an NVFP4 weight holds two values
/// a byte.
pub(super) fn logical(
    config: &Config,
    table: &[(String, Dtype, Vec<usize>)],
) -> Vec<(String, Vec<usize>)> {
    let p = plan(
        config,
        table.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
    );
    assert!(p.refused.is_empty(), "{:?}", p.refused);
    table
        .iter()
        .filter_map(|(n, _, s)| Some((n.clone(), p.placed.get(n)?.format.logical(s))))
        .collect()
}

pub(super) fn table(header: &Value) -> Vec<(String, Dtype, Vec<usize>)> {
    header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(n, _)| *n != "__metadata__")
        .map(|(n, e)| {
            let shape = e["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let d = e["dtype"].as_str().unwrap();
            let dtype = Dtype::of(d).unwrap_or_else(|| panic!("{n}: {d}"));
            (n.clone(), dtype, shape)
        })
        .collect()
}

/// The unsloth GGUF's names and ggml dims from a `tensor`, `dimensions` table.
pub(super) fn gguf_table(text: &str, keep: impl Fn(&str) -> bool) -> BTreeMap<String, Vec<usize>> {
    let cols: Vec<&str> = text.lines().next().unwrap().split('\t').collect();
    let (t, d) = (
        cols.iter().position(|c| *c == "tensor").unwrap(),
        cols.iter().position(|c| *c == "dimensions").unwrap(),
    );
    text.lines()
        .skip(1)
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|r| keep(r[t]))
        .map(|r| {
            (
                r[t].to_string(),
                r[d].split('x').map(|n| n.parse().unwrap()).collect(),
            )
        })
        .collect()
}

/// The glm5next tensor's elements from the source's `src`, `size` bytes an element, row-major
/// `shape`; for [`Layout::NegExp`], `src` is f32. A stack's experts are joined by the caller.
pub(super) fn apply(layout: &Layout, src: &[u8], shape: &[usize], size: usize) -> Vec<u8> {
    match *layout {
        Layout::Same | Layout::Stack(_) => src.to_vec(),
        Layout::KeyHalf { heads, nope, v } => {
            let rank = shape[1];
            let mut out = Vec::with_capacity(heads * rank * nope * size);
            for h in 0..heads {
                let head = &src[h * (nope + v) * rank * size..];
                for c in 0..rank {
                    for r in 0..nope {
                        out.extend_from_slice(&head[(r * rank + c) * size..][..size]);
                    }
                }
            }
            out
        }
        Layout::ValueHalf { heads, nope, v } => {
            let row = shape[1] * size;
            (0..heads)
                .flat_map(|h| &src[(h * (nope + v) + nope) * row..][..v * row])
                .copied()
                .collect()
        }
        Layout::NegExp { heads } => src
            .as_chunks::<4>()
            .0
            .iter()
            .take(heads)
            .flat_map(|&b| {
                let x = f32::from_le_bytes(b);
                ((-f64::from(x).exp()) as f32).to_le_bytes()
            })
            .collect(),
    }
}
