//! The synthetic `qwen3vl_merger` encoder file: a seeded fill of the real tensor table, for an
//! image-input seat whose real mmproj is not kept. Every hyperparameter and every tensor is the
//! loader's own ([`crate::arch::qwen3vl`]: the header the reader pins, the table
//! [`tensors::expected`] checks), so a file this module writes opens as the real one does; only the
//! values are drawn, not converted. The one free count is the merger's output width
//! ([`OUT_DIM`]): the embedding width of the text model the file pairs with.
//!
//! The values follow the text fixtures' rules (`model::fixture::fill`): each tensor's stream is
//! keyed by the seed and the tensor's name, cut into chunks so any thread count writes the same
//! bytes; a matrix's elements are uniform at standard deviation `1/√K` (`K` the values a row is
//! drawn over: `dims[0]` of a stored `[K, N]` matrix, the three leading axes of a conv kernel —
//! the input channel count they fold into), so a projection of a unit-RMS row has unit RMS; a
//! LayerNorm gain sits at `1 ± GAIN_SPREAD` and its bias at `±NORM_BIAS_SPREAD` (a gain of exactly
//! 1 or a bias of exactly 0 would hide a path that drops them); every other vector — the linear
//! biases — is drawn at standard deviation 1, the scale of the rows it is added to. The stream
//! generator and the name key are `model::fixture::fill`'s, copied because the vision crate does
//! not depend on the model crate (the shape tables' owner is here, and the model crate's generator
//! is not opened to other crates' fills).
//!
//! The file is pinned twice: the digest test below holds the written file to the loader's header
//! and table checks and to per-tensor digests (a fill change moves them and needs the pins
//! re-taken, which the sets dumped from an older file feel as the mmproj sha they state), and the
//! seat's families pin the file's own sha256.

use std::fs::File;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use gguf::write::{Layout, TensorDecl, WriteError, Writer};
use gguf::{GgmlType, Value, digest};

use crate::arch::qwen3vl::Hparams;
use crate::arch::qwen3vl::tensors::{self, Expected, Export};

/// The merger's output width: the embedding width of the text model the fixture pairs with (the
/// qwen35moe fixture's 2,048).
pub const OUT_DIM: usize = 2048;

/// The export the fixture writes: bf16 matrices, f32 patch kernels — the export of the real
/// projector files this seat's families pin.
const EXPORT: Export = Export::Bf16;

/// The seed the writer takes when none is given.
pub const DEFAULT_SEED: u64 = 1;

/// The counts the header carries, in file order: the values the loader's own table pins
/// (`arch::qwen3vl::hparams`), so the digest test's open of the written file is the check that they
/// have not drifted. The last entry is the converter's warm-up size, which no hyperparameter
/// reads.
const COUNTS: [(&str, u64); 7] = [
    ("clip.vision.block_count", 27),
    ("clip.vision.embedding_length", 1152),
    ("clip.vision.attention.head_count", 16),
    ("clip.vision.feed_forward_length", 4304),
    ("clip.vision.patch_size", 16),
    ("clip.vision.spatial_merge_size", 2),
    ("clip.vision.image_size", 768),
];

/// The vision tower's LayerNorm epsilon, as the f32 the file stores.
const EPS: f32 = 1e-6;

/// A LayerNorm gain is drawn in `1 ± GAIN_SPREAD`.
const GAIN_SPREAD: f32 = 0.1;
/// A LayerNorm bias in `±NORM_BIAS_SPREAD`.
const NORM_BIAS_SPREAD: f32 = 0.1;

/// Bytes a chunk of a tensor's stream holds at most, in whole values.
const CHUNK_TARGET: usize = 1 << 20;

/// What the writer did.
#[derive(Clone, Debug)]
pub struct Written {
    pub path: PathBuf,
    pub tensors: usize,
    /// The tensors' bytes, padding excluded.
    pub bytes: u64,
    /// The finished file's bytes.
    pub file_bytes: u64,
    /// Lowercase hex sha256 of the finished file: the figure the seat's families pin.
    pub sha256: String,
    pub gen_secs: f64,
    pub write_secs: f64,
    pub secs: f64,
}

/// Everything writing the fixture refuses, each naming what it refused.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("{path}: {op}: {source}")]
    Io {
        path: PathBuf,
        op: &'static str,
        source: io::Error,
    },
    #[error("{path}: writing: {source}")]
    Write { path: PathBuf, source: WriteError },
    #[error("{path} exists; the fixture is never written over")]
    Exists { path: PathBuf },
    #[error("{error}; and removing {tmp} failed: {cleanup}")]
    Abandoned {
        error: Box<FixtureError>,
        tmp: PathBuf,
        cleanup: io::Error,
    },
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
}

fn io_err(path: &Path, op: &'static str, source: io::Error) -> FixtureError {
    FixtureError::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
}

/// The hyperparameters of the fixture file, from the counts the header writes.
fn hparams() -> Hparams {
    let [n_layer, dim, n_head, ff, patch, merge, _] = COUNTS.map(|(_, v)| v as usize);
    Hparams {
        n_layer,
        dim,
        n_head,
        ff,
        patch,
        merge,
        out_dim: OUT_DIM,
        eps: EPS,
    }
}

/// The header's metadata pairs: the architecture, the projector, the booleans and counts, the
/// width, the epsilon, the preprocessing statistics and the no-DeepStack table.
fn kvs() -> Vec<(String, Value)> {
    let mut kvs = vec![
        (
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String("clip".into()),
        ),
        (
            "clip.projector_type".to_string(),
            Value::String(crate::arch::qwen3vl::PROJECTOR_TYPE.into()),
        ),
        ("clip.has_vision_encoder".to_string(), Value::Bool(true)),
        ("clip.use_gelu".to_string(), Value::Bool(true)),
    ];
    kvs.extend(COUNTS.iter().map(|(k, v)| {
        (
            (*k).to_string(),
            Value::U32(u32::try_from(*v).expect("a count")),
        )
    }));
    kvs.extend([
        (
            "clip.vision.projection_dim".to_string(),
            Value::U32(u32::try_from(OUT_DIM).expect("the width is a u32")),
        ),
        (
            "clip.vision.attention.layer_norm_epsilon".to_string(),
            Value::F32(EPS),
        ),
        (
            "clip.vision.image_mean".to_string(),
            Value::Array(vec![Value::F32(0.5); 3]),
        ),
        (
            "clip.vision.image_std".to_string(),
            Value::Array(vec![Value::F32(0.5); 3]),
        ),
        (
            "clip.vision.is_deepstack_layers".to_string(),
            Value::Array(vec![Value::Bool(false); hparams().n_layer]),
        ),
    ]);
    kvs
}

// ------------------------------------------------------------------ the fill

/// How one tensor's values are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Fill {
    /// Uniform at standard deviation `1/√K` in the tensor's own type: every stored matrix, the
    /// learned position table and the two conv kernels.
    Matrix,
    /// A LayerNorm gain, in `1 ± GAIN_SPREAD` (f32).
    Gain,
    /// A LayerNorm bias, in `±NORM_BIAS_SPREAD` (f32).
    NormBias,
    /// A linear bias, uniform at standard deviation 1 (f32): the scale of the rows it is added to.
    LinBias,
}

impl Fill {
    /// The fill of table row `e`: a tensor of more than one axis is a matrix; a vector is chosen by
    /// its leaf name, and a leaf the rules do not name is refused (a new tensor of the table gets
    /// its rule written, not a silent default).
    fn of(e: &Expected) -> Result<Fill, FixtureError> {
        if e.dims.len() > 1 {
            return Ok(Fill::Matrix);
        }
        if e.ty != GgmlType::F32 {
            return Err(FixtureError::Tensor {
                name: e.name.clone(),
                detail: format!("is {}, and the fill writes f32 vectors", e.ty),
            });
        }
        match leaf_of(&e.name) {
            "ln1.weight" | "ln2.weight" | "post_ln.weight" => Ok(Fill::Gain),
            "ln1.bias" | "ln2.bias" | "post_ln.bias" => Ok(Fill::NormBias),
            "patch_embd.bias" | "attn_qkv.bias" | "attn_out.bias" | "ffn_up.bias"
            | "ffn_down.bias" | "mm.0.bias" | "mm.2.bias" => Ok(Fill::LinBias),
            other => Err(FixtureError::Tensor {
                name: e.name.clone(),
                detail: format!("{other:?} is a vector the fill rules do not name"),
            }),
        }
    }
}

/// The leaf a vector's fill is chosen by: the last two dot-parts of a `v.` name
/// (`v.blk.3.ln1.weight` → `ln1.weight`), the whole name otherwise (`mm.0.bias`).
fn leaf_of(name: &str) -> &str {
    if !name.starts_with("v.") {
        return name;
    }
    match name
        .rsplit_once('.')
        .and_then(|(stem, _)| stem.rsplit_once('.'))
    {
        Some((head, _)) => &name[head.len() + 1..],
        None => name,
    }
}

/// The values a row of tensor `e` is drawn over: `dims[0]` of a stored `[K, N]` matrix, the three
/// leading axes of a conv kernel — the input channel count they fold into.
fn row_values(e: &Expected) -> u64 {
    e.dims[..e.dims.len() - 1].iter().product()
}

/// One value of `fill` from the random word `r`, its row `k` values long.
fn value(fill: Fill, k: u64, r: u64) -> f32 {
    let unit = ((r >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
    match fill {
        Fill::Matrix => unit * (3.0 / k as f32).sqrt(),
        Fill::Gain => 1.0 + unit * GAIN_SPREAD,
        Fill::NormBias => unit * NORM_BIAS_SPREAD,
        Fill::LinBias => unit * 3.0f32.sqrt(),
    }
}

/// f32 → bf16 bits, round to nearest even (the values are finite), the text fixture generator's
/// rule copied.
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The bytes one value of `ty` takes: the table's matrices are bf16 and every other tensor f32.
fn unit_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::F32 => 4,
        GgmlType::F16 | GgmlType::BF16 => 2,
        other => panic!("tensor type {other}: the table holds bf16 matrices and f32 vectors only"),
    }
}

/// SplitMix64: one `u64` per step, from the state it holds (the text fixture generator's, copied).
struct Rng(u64);

impl Rng {
    fn step(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn mix(x: u64) -> u64 {
    Rng(x).step()
}

/// FNV-1a of a tensor name (the text fixture generator's, copied).
fn name_key(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The generator of chunk `chunk` of the tensor named `name` under `seed`.
fn chunk_rng(seed: u64, name: &str, chunk: usize) -> Rng {
    Rng(mix(mix(mix(seed) ^ name_key(name)) ^ chunk as u64))
}

/// One chunk's whole values of `fill`, its rows `k` values long, from a fresh stream.
fn fill_chunk(fill: Fill, k: u64, ty: GgmlType, mut rng: Rng, chunk: &mut [u8]) {
    match ty {
        GgmlType::F32 => {
            for e in chunk.as_chunks_mut::<4>().0 {
                *e = value(fill, k, rng.step()).to_le_bytes();
            }
        }
        GgmlType::F16 => {
            for e in chunk.as_chunks_mut::<2>().0 {
                *e = gguf::quant::f32_to_f16_bits(value(fill, k, rng.step())).to_le_bytes();
            }
        }
        GgmlType::BF16 => {
            for e in chunk.as_chunks_mut::<2>().0 {
                *e = bf16_bits(value(fill, k, rng.step())).to_le_bytes();
            }
        }
        other => panic!("tensor type {other}: no fill of a whole value"),
    }
}

/// Fill `out` — whole values of `ty` — with the tensor named `name`'s `fill`, its rows `k` values
/// long, under `seed`. Each chunk draws from its own stream, keyed by the seed, the name and the
/// chunk index, and the chunks are spread over the machine's threads: the bytes are a function of
/// the seed and the name alone.
fn fill(name: &str, ty: GgmlType, fill: Fill, k: u64, seed: u64, out: &mut [u8]) {
    let unit = unit_bytes(ty);
    let len = (CHUNK_TARGET / unit).max(1) * unit;
    let n = out.len().div_ceil(len);
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(n.max(1));
    // Lane `i` holds the chunks `[(i·n)/threads, ((i+1)·n)/threads)`, so each split point is a
    // whole chunk's boundary and every lane is non-empty.
    let at = |i: usize| (i * n) / threads * len;
    let mut parts: Vec<&mut [u8]> = Vec::with_capacity(threads);
    let mut rest = out;
    let mut prev = 0;
    for i in 1..threads {
        let (head, tail) = rest.split_at_mut(at(i) - prev);
        parts.push(head);
        prev = at(i);
        rest = tail;
    }
    parts.push(rest);
    std::thread::scope(|s| {
        for (lane, part) in parts.into_iter().enumerate() {
            let first = (lane * n) / threads;
            s.spawn(move || {
                for (c, chunk) in part.chunks_mut(len).enumerate() {
                    fill_chunk(fill, k, ty, chunk_rng(seed, name, first + c), chunk);
                }
            });
        }
    });
}

// ------------------------------------------------------------------ the writer

/// The tensors of `table` under `seed`, written and synced: their count, their bytes and the
/// seconds spent generating and writing them.
type Filled = (usize, u64, f64, f64);

/// Write the fixture to `out` (which must not exist), under `seed`, one `progress` call a tensor
/// with its name and bytes. The bytes go to `<out>.tmp.<pid>` and the file is renamed into place
/// once complete and synced; a failure removes the temporary file.
pub fn write(
    out: &Path,
    seed: u64,
    progress: &mut dyn FnMut(&str, u64),
) -> Result<Written, FixtureError> {
    let t0 = Instant::now();
    if out.try_exists().map_err(|e| io_err(out, "stat", e))? {
        return Err(FixtureError::Exists {
            path: out.to_path_buf(),
        });
    }
    let hp = hparams();
    let table = tensors::expected(&hp, EXPORT);
    let decls: Vec<TensorDecl> = table
        .iter()
        .map(|e| TensorDecl {
            name: e.name.clone(),
            dims: e.dims.clone(),
            type_id: e.ty.file_id(),
            nbytes: e.bytes(),
        })
        .collect();
    let layout = Layout::new(&kvs(), decls).map_err(|e| FixtureError::Write {
        path: out.to_path_buf(),
        source: e,
    })?;
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    std::fs::create_dir_all(parent).map_err(|e| io_err(parent, "create the directory", e))?;
    let mut tmp = out.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let filled = match fill_all(&table, seed, &tmp, &layout, progress) {
        Ok(f) => f,
        Err(error) => {
            return Err(cleanup(tmp, error));
        }
    };
    // A link, not a rename: it fails on a path that appeared meanwhile instead of replacing it.
    if let Err(e) = std::fs::hard_link(&tmp, out) {
        let error = if e.kind() == io::ErrorKind::AlreadyExists {
            FixtureError::Exists {
                path: out.to_path_buf(),
            }
        } else {
            io_err(out, "link into place", e)
        };
        return Err(cleanup(tmp, error));
    }
    std::fs::remove_file(&tmp).map_err(|e| io_err(&tmp, "remove the temporary file", e))?;
    File::open(parent)
        .and_then(|d| d.sync_all())
        .map_err(|e| io_err(parent, "fsync", e))?;
    let sha256 =
        digest::sha256_file(out).map_err(|e| io_err(out, "read back", io::Error::other(e)))?;
    let (tensors, bytes, gen_secs, write_secs) = filled;
    Ok(Written {
        path: out.to_path_buf(),
        tensors,
        bytes,
        file_bytes: layout.file_len(),
        sha256,
        gen_secs,
        write_secs,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// A failed write's temporary file removed, the failure kept unless the removal fails too.
fn cleanup(tmp: PathBuf, error: FixtureError) -> FixtureError {
    match std::fs::remove_file(&tmp) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => FixtureError::Abandoned {
            error: Box::new(error),
            tmp,
            cleanup: e,
        },
        _ => error,
    }
}

/// Every tensor of `table` into the new file `tmp` as `layout` declares it, each synced with the
/// file at the end.
fn fill_all(
    table: &[Expected],
    seed: u64,
    tmp: &Path,
    layout: &Layout,
    progress: &mut dyn FnMut(&str, u64),
) -> Result<Filled, FixtureError> {
    let file = File::create_new(tmp).map_err(|e| io_err(tmp, "create", e))?;
    let werr = |source| FixtureError::Write {
        path: tmp.to_path_buf(),
        source,
    };
    let mut w =
        Writer::new(BufWriter::with_capacity(1 << 20, file), layout.clone()).map_err(werr)?;
    let largest = table.iter().map(|e| e.bytes() as usize).max().unwrap_or(0);
    let mut buf = vec![0u8; largest];
    let mut filled = (0usize, 0u64, 0.0f64, 0.0f64);
    for e in table {
        let rule = Fill::of(e)?;
        let k = row_values(e);
        let out = &mut buf[..e.bytes() as usize];
        let g = Instant::now();
        fill(&e.name, e.ty, rule, k, seed, out);
        filled.2 += g.elapsed().as_secs_f64();
        let wt = Instant::now();
        w.tensor(&e.name, out).map_err(werr)?;
        filled.3 += wt.elapsed().as_secs_f64();
        filled.0 += 1;
        filled.1 += e.bytes();
        progress(&e.name, e.bytes());
    }
    let file = w
        .finish()
        .map_err(werr)?
        .into_inner()
        .map_err(|e| io_err(tmp, "flush", e.into_error()))?;
    file.sync_all().map_err(|e| io_err(tmp, "fsync", e))?;
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::{DEFAULT_SEED, Fill, OUT_DIM, hparams, kvs, leaf_of, write};
    use crate::arch::qwen3vl::Hparams;
    use crate::arch::qwen3vl::tensors::{self, Export};

    /// The pinned digests: one tensor of every fill, read back from the written file, and the
    /// file's own sha256 below. The procedure when a digest moves is the text fixtures'
    /// (`model::fixture::digest`): this test prints `fill digest <name> <hex>` and
    /// `file sha256 <hex>` before it asserts, and the run's lines fill the row in — which the sets
    /// dumped from an older file feel as the mmproj sha they state.
    const PINS: &[(&str, &str)] = &[
        (
            "mm.0.weight",
            "b90f1386581dec349047349f03ee6cd053a2a292c2ace933129911c45c0f7075",
        ),
        (
            "v.blk.13.attn_qkv.weight",
            "4f93916ff8578b76559bf983973aff80629d0a77fcb984f36baffde45677f6bf",
        ),
        (
            "v.patch_embd.weight.1",
            "0b679b3eb6c644b077f7251e80f2584c5eb836f69a5d51d32805735844fd9337",
        ),
        (
            "v.position_embd.weight",
            "e40e9a2901107a7979a0d2905cd9ffd83f3909df9c023349ef9f45861c0964a5",
        ),
        (
            "v.blk.5.ln2.weight",
            "9b59570aa1e76b35211b88c0ce416c597584933671b0f3dd0ca6226211d7f140",
        ),
        (
            "v.blk.5.ln2.bias",
            "9eea1ddcb5e3b66616b73d70310a19182ac2d4ea4e6703085a1da85abb488809",
        ),
        (
            "v.blk.5.ffn_up.bias",
            "16db8c2e3c91c5d022ea91c2f0afc8f71b1d354e923df423faab4813433d9655",
        ),
        (
            "mm.2.bias",
            "78f19195272fdb2db4ddbe34f1291291238a6ca97a69606fb715ae671f82952d",
        ),
    ];

    /// The written file's sha256.
    const FILE_SHA256: &str = "8fb7ca9c074a62131187e6641a753b3debcda108397128a73fff0fe962d08ac7";

    /// The leaf a vector's fill is chosen by, and every tensor of the table having one: 113
    /// matrices (the two conv kernels, the position table, four stored matrices a block and the
    /// merger's two), 55 gains and 55 norm biases (two and one a block, the final norm), 111
    /// linear biases (four a block, the patch embedding's and the merger's two) — 334 tensors, of
    /// which 110 are bf16 and 224 f32.
    #[test]
    fn every_tensor_of_the_table_has_a_fill() {
        for (name, leaf) in [
            ("v.blk.3.ln1.weight", "ln1.weight"),
            ("v.post_ln.bias", "post_ln.bias"),
            ("mm.2.bias", "mm.2.bias"),
            ("v.patch_embd.bias", "patch_embd.bias"),
        ] {
            assert_eq!(leaf_of(name), leaf);
        }
        let table = tensors::expected(&hparams(), Export::Bf16);
        assert_eq!(table.len(), 334);
        let mut counts = std::collections::BTreeMap::new();
        for e in &table {
            *counts.entry(Fill::of(e).expect("a fill")).or_insert(0) += 1;
        }
        assert_eq!(
            counts,
            [
                (Fill::Matrix, 113),
                (Fill::Gain, 55),
                (Fill::NormBias, 55),
                (Fill::LinBias, 111),
            ]
            .into_iter()
            .collect()
        );
        let of = |ty| table.iter().filter(|e| e.ty == ty).count();
        assert_eq!((of(GgmlType::BF16), of(GgmlType::F32)), (110, 224));
    }

    /// The header is the loader's: sixteen pairs, every key the reader checks among them, and the
    /// hyperparameters at the fixture's width.
    #[test]
    fn the_header_is_the_loaders() {
        let kvs = kvs();
        let keys: Vec<&str> = kvs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys.len(), 16);
        for key in [
            gguf::GENERAL_ARCHITECTURE,
            "clip.projector_type",
            "clip.has_vision_encoder",
            "clip.use_gelu",
            "clip.vision.projection_dim",
            "clip.vision.attention.layer_norm_epsilon",
            "clip.vision.image_mean",
            "clip.vision.image_std",
            "clip.vision.is_deepstack_layers",
        ] {
            assert!(keys.contains(&key), "{key} missing");
        }
        let hp = hparams();
        assert_eq!(
            (
                hp.n_layer, hp.dim, hp.n_head, hp.ff, hp.patch, hp.merge, hp.out_dim
            ),
            (27, 1152, 16, 4304, 16, 2, OUT_DIM)
        );
    }

    /// The written file opens with the loader's header and table checks and is the pinned fill:
    /// every pinned tensor's bytes read back from the file hash to their pin, the file's own
    /// sha256 is the pinned one, and writing over the file is refused by name.
    #[test]
    fn the_written_file_opens_and_is_the_pinned_fill() {
        let dir = std::env::temp_dir().join(format!("vision-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the temp directory");
        let out = dir.join("mmproj-fixture-bf16.gguf");
        let _ = std::fs::remove_file(&out);
        let written = write(&out, DEFAULT_SEED, &mut |_, _| {}).expect("write");
        let g = gguf::Gguf::open(&out).expect("the written file opens");
        let hp = Hparams::read(&g).expect("the loader reads the header");
        hp.check_text_width(OUT_DIM).expect("the text width");
        let named = tensors::check(
            &hp,
            g.iter_tensors()
                .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ty)),
        )
        .expect("the loader takes the tensor table");
        assert_eq!(named, 334);
        assert_eq!(written.tensors, 334);
        let mut moved: Vec<String> = Vec::new();
        for (name, pin) in PINS {
            let t = g
                .find(name)
                .unwrap_or_else(|| panic!("the file holds {name}"));
            let bytes = g.data(t).unwrap_or_else(|e| panic!("{name}: {e}"));
            let got = gguf::digest::sha256_hex(bytes);
            println!("fill digest {name} {got}");
            if &got != pin {
                moved.push(format!("{name}: pinned {pin}, today {got}"));
            }
        }
        println!("file sha256 {}", written.sha256);
        if written.sha256 != FILE_SHA256 {
            moved.push(format!(
                "the file: pinned {FILE_SHA256}, today {}",
                written.sha256
            ));
        }
        assert!(
            moved.is_empty(),
            "the fill moved: {moved:#?}; take today's printed digests and the sha into PINS"
        );
        assert!(write(&out, DEFAULT_SEED, &mut |_, _| {}).is_err());
        std::fs::remove_dir_all(&dir).expect("the temp directory");
    }
}
