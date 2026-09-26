//! A synthetic routed layer and its r8 sidecar, for the gates that run the
//! host tier's row-lane path without a model file: one GGUF holding a layer's
//! `ffn_{gate,up,down}_exps` — the gate and the up Q3_K, the down Q4_K or
//! Q3_K, random codes under finite block scales — written with `gguf::write`
//! into its own directory, and `r8file::convert` of the gate and the up at
//! `r8file::sidecar_path` of it, where a load looks.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Split};
use model::moe::HostLayerSpec;
use model::r8file;

pub const GATE: &str = "blk.0.ffn_gate_exps.weight";
pub const UP: &str = "blk.0.ffn_up_exps.weight";
pub const DOWN: &str = "blk.0.ffn_down_exps.weight";

/// The layer's SwiGLU limit: the clamped combine, as V4.1's.
const LIMIT: f32 = 10.0;

/// The written layer: its directory, the source file and the sidecar.
pub struct Layer {
    pub dir: PathBuf,
    pub source: PathBuf,
    pub sidecar: PathBuf,
    pub embd: usize,
    pub ff: usize,
    pub n_expert: usize,
}

impl Layer {
    /// The layer of `n_expert` experts over `embd` and `ff`, its down of
    /// type `down`, under a fresh directory for `tag`, and its sidecar.
    /// `embd` and `ff` are multiples of 256 — the grids the tile and the
    /// down's blocks take.
    pub fn write(tag: &str, embd: usize, ff: usize, n_expert: usize, down: GgmlType) -> Layer {
        let dir = std::env::temp_dir().join(format!("r8layer-{}-{tag}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        let src_dir = dir.join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let source = src_dir.join("layer.gguf");
        let stacks = [
            (GATE, GgmlType::Q3_K, [embd, ff, n_expert], 0x6a7e),
            (UP, GgmlType::Q3_K, [embd, ff, n_expert], 0x0b),
            (DOWN, down, [ff, embd, n_expert], 0xd0),
        ];
        let tensors: Vec<(TensorDecl, Vec<u8>)> = stacks
            .iter()
            .map(|&(name, ty, dims, seed)| {
                let b = stack_bytes(ty, dims, seed);
                let decl = TensorDecl {
                    name: name.to_string(),
                    dims: dims.map(|d| d as u64).to_vec(),
                    type_id: ty.as_u32(),
                    nbytes: b.len() as u64,
                };
                (decl, b)
            })
            .collect();
        let layout = Layout::new(&[], tensors.iter().map(|(t, _)| t.clone()).collect()).unwrap();
        let mut w = Writer::new(BufWriter::new(File::create(&source).unwrap()), layout).unwrap();
        for (t, b) in &tensors {
            w.tensor(&t.name, b).unwrap();
        }
        w.finish().unwrap();
        let sidecar = r8file::sidecar_path(&source);
        let split = Split::open(&source).unwrap();
        r8file::convert(&split, &[GATE.into(), UP.into()], &sidecar, &mut |_| {})
            .unwrap_or_else(|e| panic!("convert {}: {e}", source.display()));
        Layer {
            dir,
            source,
            sidecar,
            embd,
            ff,
            n_expert,
        }
    }

    /// The layer as a host tier builds it.
    pub fn spec(&self) -> HostLayerSpec<'static> {
        HostLayerSpec {
            gate: GATE,
            up: UP,
            down: DOWN,
            n_expert: self.n_expert,
            embd: self.embd,
            ff: self.ff,
            swiglu_limit: LIMIT,
        }
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A stack `[k, n, n_expert]` of `ty` (Q3_K or Q4_K): xorshift codes and
/// scales, each block's f16 `d` (and Q4_K's `dmin`) fixed to a finite value.
fn stack_bytes(ty: GgmlType, [k, n, n_expert]: [usize; 3], seed: u64) -> Vec<u8> {
    let (bs, ts) = (
        ty.blck_size().unwrap() as usize,
        ty.type_size().unwrap() as usize,
    );
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let blocks = k / bs * n * n_expert;
    let mut b = Vec::with_capacity(blocks * ts);
    for _ in 0..blocks {
        let at = b.len();
        b.extend((0..ts).map(|_| next() as u8));
        let blk = &mut b[at..];
        match ty {
            // block_q3_K: d f16 at 108.
            GgmlType::Q3_K => blk[108..110].copy_from_slice(&0x2000u16.to_le_bytes()),
            // block_q4_K: d f16 at 0, dmin f16 at 2.
            GgmlType::Q4_K => {
                blk[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x1C00u16.to_le_bytes());
            }
            _ => panic!("no synthetic stack of {ty}"),
        }
    }
    b
}
