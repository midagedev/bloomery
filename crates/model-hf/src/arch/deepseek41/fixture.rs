//! The original-format deepseek41 fixture folder ([`write`]): a DeepSeek-V4.1 checkpoint folder
//! cut to a few layers ([`Shape`]), its routed and shared experts to a narrower inner width and
//! its Engram tables to the buckets of a small hash vocabulary, in the source's own formats (MXFP4
//! experts beside E8M0 scales, F8_E4M3 projections beside 32 by 32 E8M0 scales, the Engram
//! tables' two planes, BF16 and F32), with the config edited to match and the tokenizer files
//! copied. Only the source's headers are read: every value is drawn from the seed and the
//! tensor's name, a matrix's elements of standard deviation `1/√k`, a gain or a bias the GGUF
//! fixture's constant for its deepseek41 name ([`fixture::const_value`],
//! [`fixture::spread_value`]).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use model::arch::deepseek41::fixture;
use model::fixture::{Rng, chunk_rng};
use safetensors::Folder;
use serde_json::{Value as Json, json};

use super::{Ds41Map, skip_reason};
use crate::folder::{Error, FolderMap, refuse};

/// The fixture's routed and shared inner width: halves and quarters remain whole
/// 256-value MXFP4 kernel units.
pub const FF: usize = 1024;

/// The hash vocabulary the fixture's Engram buckets start past ([`hf::engram::primes`]).
pub const ENGRAM_VOCAB: u64 = fixture::ENGRAM_PRIME_FLOOR;

/// Which of the source's layers a fixture keeps, renumbered from 0, its inner width and, when
/// fewer than the source's, its routed experts (the first ones).
#[derive(Clone, Debug)]
pub struct Shape {
    pub layers: Vec<usize>,
    pub ff: usize,
    pub experts: Option<usize>,
}

impl Default for Shape {
    /// The GGUF fixture's layers ([`fixture::LAYER_MAP`]) at [`FF`].
    fn default() -> Shape {
        Shape {
            layers: fixture::LAYER_MAP.to_vec(),
            ff: FF,
            experts: None,
        }
    }
}

/// The files of the source the fixture copies.
const COPY: [&str; 2] = ["tokenizer.json", "tokenizer_config.json"];

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::Refused(format!("{}: {e}", path.display()))
}

/// `list` of layer ids as the fixture's: each kept id's index in `layers`, the others dropped.
fn remap(list: &Json, layers: &[usize]) -> Option<Json> {
    let ids = list
        .as_array()?
        .iter()
        .map(|v| v.as_u64().map(|l| l as usize))
        .collect::<Option<Vec<_>>>()?;
    Some(json!(
        ids.iter()
            .filter_map(|l| layers.iter().position(|k| k == l))
            .collect::<Vec<_>>()
    ))
}

/// The fixture's config of the source's `config` at `shape`, and each Engram site's rows.
fn edit_config(config: &Json, shape: &Shape) -> Result<(Json, Vec<u64>), Error> {
    let mut config = config.clone();
    let text = config
        .get_mut("text_config")
        .and_then(Json::as_object_mut)
        .ok_or_else(|| Error::Refused("config.json: no text_config".into()))?;
    let bad = |key: &str| Error::Refused(format!("config.json: text_config.{key} is not read"));
    let n_layer = text
        .get("num_hidden_layers")
        .and_then(Json::as_u64)
        .ok_or_else(|| bad("num_hidden_layers"))? as usize;
    if let Some(l) = shape.layers.iter().find(|&&l| l >= n_layer) {
        return Err(Error::Refused(format!(
            "layer {l} is past the source's {n_layer}"
        )));
    }
    let ratios = text
        .get("compress_ratios")
        .and_then(Json::as_array)
        .filter(|r| r.len() >= n_layer)
        .ok_or_else(|| bad("compress_ratios"))?;
    let kept: Vec<Json> = shape
        .layers
        .iter()
        .map(|&l| ratios[l].clone())
        .chain(ratios[n_layer..].iter().cloned())
        .collect();
    text.insert("compress_ratios".into(), json!(kept));
    for key in [
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "engram_layer_ids",
        "dspark_target_layer_ids",
    ] {
        let ids = text
            .get(key)
            .and_then(|list| remap(list, &shape.layers))
            .ok_or_else(|| bad(key))?;
        text.insert(key.into(), ids);
    }
    let source = text
        .get("candidate_source_layer_id")
        .and_then(Json::as_u64)
        .ok_or_else(|| bad("candidate_source_layer_id"))?;
    let at = shape
        .layers
        .iter()
        .position(|&l| l as u64 == source)
        .ok_or_else(|| {
            Error::Refused(format!("the candidate source layer {source} is not kept"))
        })?;
    text.insert("candidate_source_layer_id".into(), json!(at));
    text.insert("num_hidden_layers".into(), json!(shape.layers.len()));
    text.insert("moe_intermediate_size".into(), json!(shape.ff));
    if let Some(n) = shape.experts {
        let used = text
            .get("num_experts_per_tok")
            .and_then(Json::as_u64)
            .ok_or_else(|| bad("num_experts_per_tok"))?;
        text.insert("n_routed_experts".into(), json!(n));
        text.insert("num_experts_per_tok".into(), json!(used.min(n as u64)));
    }
    let count = |key: &str| {
        text.get(key)
            .and_then(Json::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| bad(key))
    };
    let sites = text
        .get("engram_layer_ids")
        .and_then(Json::as_array)
        .map_or(0, Vec::len);
    let orders = count("engram_max_ngram_size")?.saturating_sub(1);
    let sizes = hf::engram::primes(sites, orders, count("engram_n_heads")?, ENGRAM_VOCAB)
        .map_err(Error::Refused)?;
    let rows: Vec<u64> = sizes.iter().map(|s| s.iter().sum()).collect();
    text.insert("engram_vocab_size".into(), json!(ENGRAM_VOCAB));
    text.insert("engram_num_embeddings".into(), json!(rows));
    Ok((config, rows))
}

/// One fixture tensor: its name, dtype and row-major shape.
struct Planned {
    name: String,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}

impl Planned {
    fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size()
    }
}

/// The source's tensors the fixture keeps, each layer's in a shard of its own after the
/// unlayered ones' shard.
fn plan(
    folder: &Folder,
    shape: &Shape,
    (src_ff, src_experts): (usize, usize),
    rows: &[u64],
    sites: &[usize],
) -> Result<Vec<Vec<Planned>>, Error> {
    let experts = shape.experts.unwrap_or(src_experts);
    let mut shards: Vec<Vec<Planned>> = (0..=shape.layers.len()).map(|_| Vec::new()).collect();
    for name in folder.names() {
        let raw = folder.raw(name)?;
        let (at, fixture_name, rest) =
            match name.strip_prefix("layers.").and_then(|r| r.split_once('.')) {
                Some((l, rest)) => {
                    let l: usize = l
                        .parse()
                        .map_err(|_| refuse(name, "names no layer index"))?;
                    let Some(i) = shape.layers.iter().position(|&k| k == l) else {
                        continue;
                    };
                    (i + 1, format!("layers.{i}.{rest}"), rest)
                }
                None if skip_reason(name).is_some() => continue,
                None => (0, name.to_string(), name),
            };
        let mut dims = raw.shape.to_vec();
        if let Some((e, _)) = rest
            .strip_prefix("ffn.experts.")
            .and_then(|r| r.split_once('.'))
            && e.parse::<usize>()
                .map_err(|_| refuse(name, "names no expert"))?
                >= experts
        {
            continue;
        }
        if rest.starts_with("ffn.gate.") {
            dims.iter_mut()
                .filter(|d| **d == src_experts)
                .for_each(|d| *d = experts);
        }
        if rest.starts_with("ffn.experts.") || rest.starts_with("ffn.shared_experts.") {
            for d in &mut dims {
                *d = match *d {
                    d if d == src_ff => shape.ff,
                    d if d * 2 == src_ff => shape.ff / 2,
                    d if d * 32 == src_ff => shape.ff / 32,
                    d => d,
                };
            }
        }
        if rest.starts_with("engram.embed.") {
            let site = sites
                .iter()
                .position(|&s| s + 1 == at)
                .ok_or_else(|| refuse(name, "is an Engram table off the config's sites"))?;
            dims[0] = rows[site] as usize;
        }
        shards[at].push(Planned {
            name: fixture_name,
            dtype: raw.dtype,
            shape: dims,
        });
    }
    Ok(shards)
}

/// The next value of `rng`, uniform in `[-1, 1)`.
fn unit(rng: &mut Rng) -> f32 {
    (rng.step() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// The F8_E4M3 codes of magnitude at most 2: the values a block's codes draw from, of standard
/// deviation about [`E4M3_SD`].
fn e4m3_codes() -> Vec<u8> {
    (0u8..=255)
        .filter(|&b| {
            let (e, m) = ((b >> 3) & 0xf, b & 7);
            e < 8 || (e == 8 && m == 0)
        })
        .collect()
}

/// About the standard deviation of a code drawn uniformly from [`e4m3_codes`].
const E4M3_SD: f64 = 0.8;
/// The standard deviation of an FP4 (E2M1) code drawn uniformly from the 16.
const E2M1_SD: f64 = 2.93;

/// How a tensor's bytes are drawn.
enum Fill {
    /// Uniform from [`e4m3_codes`].
    E4m3,
    /// Uniform bytes: packed FP4 codes.
    Codes,
    /// One E8M0 exponent byte.
    Byte(u8),
    /// Uniform in `[-half, half)`, as BF16 or F32.
    Uniform { bf16: bool, half: f32 },
    /// Every value `value`, as BF16 or F32.
    Const { bf16: bool, value: f32 },
}

/// Every tensor's fill: a weight's codes; a block scale the E8M0 exponent nearest its weight's
/// `1/√k` over its codes' spread; a float vector (or gain) the GGUF fixture's constant or spread
/// for its deepseek41 name, any other float tensor uniform at `1/√k`.
fn fill(
    p: &Planned,
    all: &BTreeMap<&str, &Planned>,
    leaf: &BTreeMap<String, String>,
) -> Result<Fill, Error> {
    use safetensors::Dtype;
    let k = |q: &Planned| -> f64 {
        let last = *q.shape.last().unwrap_or(&1) as f64;
        if q.dtype == Dtype::I8 {
            last * 2.0
        } else {
            last
        }
    };
    Ok(match p.dtype {
        Dtype::F8E4M3 => Fill::E4m3,
        Dtype::I8 => Fill::Codes,
        Dtype::F8E8M0 => {
            let stem = p
                .name
                .strip_suffix(".scale")
                .ok_or_else(|| refuse(&p.name, "is an E8M0 tensor of no weight"))?;
            let w = all
                .get(format!("{stem}.weight").as_str())
                .ok_or_else(|| refuse(&p.name, "is a scale of no weight"))?;
            let sd = if w.dtype == Dtype::I8 {
                E2M1_SD
            } else {
                E4M3_SD
            };
            let e = (1.0 / k(w).sqrt() / sd).log2().round() as i32;
            Fill::Byte((127 + e).clamp(1, 254) as u8)
        }
        Dtype::Bf16 | Dtype::F32 => {
            let bf16 = p.dtype == Dtype::Bf16;
            let name = leaf.get(&p.name).map_or(p.name.as_str(), String::as_str);
            let short = name
                .split_once('.')
                .filter(|(b, _)| *b == "blk")
                .map_or(name, |(_, r)| r.split_once('.').map_or(r, |(_, r)| r));
            let vector = p.shape.len() < 2 || p.shape.last() == Some(&1);
            match (fixture::const_value(short), fixture::spread_value(short)) {
                (Some(value), _) => Fill::Const { bf16, value },
                (None, Some(half)) => Fill::Uniform { bf16, half },
                (None, None) if vector => {
                    return Err(refuse(
                        &p.name,
                        format!("is a vector of no fixture rule ({short})"),
                    ));
                }
                (None, None) => Fill::Uniform {
                    bf16,
                    half: (3.0 / k(p)).sqrt() as f32,
                },
            }
        }
        other => {
            return Err(refuse(
                &p.name,
                format!("is {}, which the fixture does not draw", other.name()),
            ));
        }
    })
}

/// Write `p`'s bytes by `fill` into `out`.
fn draw(
    out: &mut impl Write,
    p: &Planned,
    fill: &Fill,
    seed: u64,
    codes: &[u8],
) -> std::io::Result<()> {
    let mut rng = chunk_rng(seed, &p.name, 0);
    let mut left = p.bytes();
    let mut buf = Vec::with_capacity(1 << 20);
    while left > 0 {
        buf.clear();
        let n = left.min(1 << 20);
        match *fill {
            Fill::E4m3 => {
                buf.extend((0..n).map(|_| codes[(rng.step() % codes.len() as u64) as usize]))
            }
            Fill::Codes => buf.extend((0..n).map(|_| rng.step() as u8)),
            Fill::Byte(b) => buf.resize(n, b),
            Fill::Uniform { bf16: true, half } => {
                (0..n / 2).for_each(|_| {
                    buf.extend(gguf::quant::fp32_to_bf16_bits(unit(&mut rng) * half).to_le_bytes())
                });
            }
            Fill::Uniform { bf16: false, half } => {
                (0..n / 4).for_each(|_| buf.extend((unit(&mut rng) * half).to_le_bytes()));
            }
            Fill::Const { bf16: true, value } => (0..n / 2)
                .for_each(|_| buf.extend(gguf::quant::fp32_to_bf16_bits(value).to_le_bytes())),
            Fill::Const { bf16: false, value } => {
                (0..n / 4).for_each(|_| buf.extend(value.to_le_bytes()))
            }
        }
        out.write_all(&buf)?;
        left -= n;
    }
    Ok(())
}

/// Write the fixture of the checkpoint folder `source` at `shape` into the new folder `out`
/// (refused when it exists), values drawn from `seed`; the data bytes written. The folder is
/// written beside `out` and renamed into place once whole, so a failed write leaves nothing.
///
/// # Errors
///
/// A source whose config, headers or tokenizer files cannot be read, a layer past it, a
/// candidate source layer `shape` drops, a tensor of a format the fixture does not draw, a
/// fixture the map does not read, and any I/O error, each by name.
pub fn write(source: &Path, out: &Path, shape: &Shape, seed: u64) -> Result<u64, Error> {
    if out.exists() {
        return Err(Error::Refused(format!("{} exists", out.display())));
    }
    let config = crate::folder::read_json(&source.join("config.json"))?;
    let src_ff = config["text_config"]["moe_intermediate_size"]
        .as_u64()
        .ok_or_else(|| Error::Refused("config.json: no text_config.moe_intermediate_size".into()))?
        as usize;
    let src_experts = config["text_config"]["n_routed_experts"]
        .as_u64()
        .ok_or_else(|| Error::Refused("config.json: no text_config.n_routed_experts".into()))?
        as usize;
    let (config, rows) = edit_config(&config, shape)?;
    let sites: Vec<usize> = config["text_config"]["engram_layer_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_u64().map(|l| l as usize))
        .collect();
    let map = Ds41Map::parse(&config)?;
    let folder = Folder::open(source)?;
    let shards = plan(&folder, shape, (src_ff, src_experts), &rows, &sites)?;
    let all: BTreeMap<&str, &Planned> = shards
        .iter()
        .flatten()
        .map(|p| (p.name.as_str(), p))
        .collect();
    let table: Vec<(&str, &[usize])> = all
        .values()
        .filter(|p| !p.name.ends_with(".scale"))
        .map(|p| (p.name.as_str(), p.shape.as_slice()))
        .collect();
    let mut leaf = BTreeMap::new();
    for (name, m) in map.map(&table)?.tensors {
        for s in m.sources {
            leaf.insert(s, name.clone());
        }
    }
    let tmp = out.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::create_dir(&tmp).map_err(|e| io(&tmp, e))?;
    let done = write_into(&tmp, source, &shards, &all, &leaf, &config, seed).and_then(|total| {
        std::fs::rename(&tmp, out)
            .map_err(|e| io(out, e))
            .map(|()| total)
    });
    done.map_err(|e| match std::fs::remove_dir_all(&tmp) {
        Ok(()) => e,
        Err(c) => Error::Refused(format!("{e}; and removing {} failed: {c}", tmp.display())),
    })
}

/// The fixture's shards, index, config and copied tokenizer files written into `tmp`, and the
/// shards' data bytes.
fn write_into(
    tmp: &Path,
    source: &Path,
    shards: &[Vec<Planned>],
    all: &BTreeMap<&str, &Planned>,
    leaf: &BTreeMap<String, String>,
    config: &Json,
    seed: u64,
) -> Result<u64, Error> {
    let codes = e4m3_codes();
    let n = shards.iter().filter(|s| !s.is_empty()).count();
    let mut weight_map = serde_json::Map::new();
    let mut total = 0u64;
    for (i, shard) in shards.iter().filter(|s| !s.is_empty()).enumerate() {
        let file = format!("model-{:05}-of-{n:05}.safetensors", i + 1);
        let path = tmp.join(&file);
        let entries: Vec<(&str, &str, &[usize], usize)> = shard
            .iter()
            .map(|p| {
                (
                    p.name.as_str(),
                    p.dtype.name(),
                    p.shape.as_slice(),
                    p.bytes(),
                )
            })
            .collect();
        let mut w =
            BufWriter::with_capacity(8 << 20, File::create(&path).map_err(|e| io(&path, e))?);
        w.write_all(&safetensors::header(&entries))
            .map_err(|e| io(&path, e))?;
        for p in shard {
            let f = fill(p, all, leaf)?;
            draw(&mut w, p, &f, seed, &codes).map_err(|e| io(&path, e))?;
            weight_map.insert(p.name.clone(), json!(file));
            total += p.bytes() as u64;
        }
        w.flush().map_err(|e| io(&path, e))?;
    }
    let index = json!({"metadata": {"total_size": total}, "weight_map": weight_map});
    let put = |name: &str, bytes: &[u8]| {
        let path = tmp.join(name);
        std::fs::write(&path, bytes).map_err(|e| io(&path, e))
    };
    put(Folder::INDEX, index.to_string().as_bytes())?;
    put(
        "config.json",
        serde_json::to_string_pretty(config)
            .expect("config.json: a serde_json::Value always serializes as JSON")
            .as_bytes(),
    )?;
    for name in COPY {
        let from = source.join(name);
        put(name, &std::fs::read(&from).map_err(|e| io(&from, e))?)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value as Json};

    use super::{Ds41Map, FF, Shape, edit_config};
    use crate::folder::{Error, FolderMap};
    use crate::test_support::json;

    /// `config` edited by `edit` is refused by a message naming `key`.
    fn refused(config: &Json, key: &str, edit: impl FnOnce(&mut Map<String, Json>)) {
        let mut config = config.clone();
        edit(config["text_config"].as_object_mut().unwrap());
        match edit_config(&config, &Shape::default()) {
            Err(Error::Refused(m)) => assert!(m.contains(key), "{key}: {m}"),
            other => panic!("{key}: {other:?}"),
        }
    }

    /// A source config of fewer compress ratios than its layers, or without a key the fixture
    /// edits, is refused by name.
    #[test]
    fn a_short_or_incomplete_source_config_is_refused_by_name() {
        let config =
            json("deepseek-v4.1-flash-1017-tokenizer-files.json.gz")["config.json"].clone();
        edit_config(&config, &Shape::default()).unwrap();
        // Short of a layer the shape keeps, then short of the source's last layer alone.
        for len in [3, 39] {
            refused(&config, "compress_ratios", |t| {
                t["compress_ratios"].as_array_mut().unwrap().truncate(len);
            });
        }
        for key in [
            "num_hidden_layers",
            "kv_source_layer_ids",
            "candidate_source_layer_id",
            "engram_n_heads",
        ] {
            refused(&config, key, |t| {
                t.remove(key);
            });
        }
    }

    /// A fixture that keeps the candidate source layer (20) first renumbers it to layer 0,
    /// and the map reads that config.
    #[test]
    fn a_candidate_source_renumbered_to_layer_0_parses() {
        let config =
            json("deepseek-v4.1-flash-1017-tokenizer-files.json.gz")["config.json"].clone();
        let shape = Shape {
            layers: vec![20, 14],
            ff: FF,
            experts: None,
        };
        let (config, _) = edit_config(&config, &shape).unwrap();
        assert_eq!(config["text_config"]["candidate_source_layer_id"], 0);
        Ds41Map::parse(&config).unwrap();
    }
}
