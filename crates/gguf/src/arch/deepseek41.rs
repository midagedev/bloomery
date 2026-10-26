//! DeepSeek-V4.1's header facts under every publisher's spelling, and where
//! its Engram tables live.
//!
//! The engine reads each fact under one canonical key (`<arch>.<name>`, the
//! spelling of the `Q3_K_M` set). Other publishers write the same fact under
//! another key, as another type, or as a tensor:
//!
//! * ISTA-DASLab's sets: the router's scoring function as the string
//!   `expert_scoring_func` beside (or instead of) the number
//!   `expert_gating_func`; the Engram scalars as `engram_n_heads`,
//!   `engram_max_ngram_size`, `engram_head_dim`, `engram_pad_token_id`; the
//!   Engram hash constants as the four integer tensors of [`HASH_TENSORS`]
//!   (token map I32, primes, offsets and multipliers I64, one row a site)
//!   instead of KV arrays; the candidate mask under its own names.
//! * Unsloth's sets: the Engram as llama.cpp's `ple.*` keys
//!   (`ple.layers`, `ple.ngram_size`, `ple.heads_per_ngram`,
//!   `embedding_length_per_layer_input`, `ple.layer_multipliers`,
//!   `ple.head_vocab_sizes`, `ple.head_offsets`) and `engram.pad_token_id`.
//!
//! The spellings are Strata's packer tables (`ALIASES`, `ple_to_engram`,
//! `int_tensors_to_engram` of its v0.2.4 `pack_gguf.py`) checked against the
//! published headers. The mechanism that reads them is [`crate::facts`].
//!
//! The Engram tables of an ISTA set are not in its model split: a separate
//! GGUF (architecture [`ENGRAM_ARCH`]) holds them, and
//! `<arch>.engram.file` names it ([`crate::engram::attach_companion`]).

use std::path::PathBuf;

use crate::Value;
use crate::engram::{Companion, EngramFacts};
use crate::facts::{Conv, Fact, FactError, HashFact, Kind};

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`
/// (`llama-hparams.h:18`), which `llama-hparams.cpp:38` names
/// [`SQRT_SOFTPLUS`]. The DeepSeek reference's `score_func == "sqrtsoftplus"`
/// is `F.softplus(scores).sqrt()` (`inference/model.py:817`), the same
/// function as `sqrt(x > 20 ? x : ln(1 + e^x))`, so the string is this value.
pub const IK_SQRT_SOFTPLUS: u64 = 4;

/// The scoring function's name in the publishers' string spelling.
pub const SQRT_SOFTPLUS: &str = "sqrtsoftplus";

/// `general.architecture` of the Engram companion file.
pub const ENGRAM_ARCH: &str = "deepseek41-engram";

/// The architecture whose header holds the Engram facts.
pub const ARCH: &str = "deepseek41";

/// The environment variable that names the companion file: a path. Used only
/// when the model split lacks an Engram table.
pub const COMPANION_ENV: &str = "BLOOMERY_V41_ENGRAM";

/// The integer tensors an ISTA set stores the hash constants in, as
/// (token map, primes, offsets, multipliers).
pub const HASH_TENSORS: [&str; 4] = [
    "engram_token_map.weight",
    "engram_primes.weight",
    "engram_offsets.weight",
    "engram_multipliers.weight",
];

/// Whether `name` is one of the [`HASH_TENSORS`].
#[must_use]
pub fn is_hash_tensor(name: &str) -> bool {
    HASH_TENSORS.contains(&name)
}

/// The tensor of Engram site `layer`'s table.
#[must_use]
pub fn table_name(layer: u64) -> String {
    format!("blk.{layer}.engram_embd.weight")
}

/// The router's scoring function as a number: ik's type, or the publishers'
/// name for the one function the V4 and V4.1 routers apply.
fn scoring(value: &Value) -> Result<u64, String> {
    match value {
        Value::String(s) if s == SQRT_SOFTPLUS => Ok(IK_SQRT_SOFTPLUS),
        Value::String(s) => Err(format!(
            "is {s:?}; the V4 and V4.1 routers score with {SQRT_SOFTPLUS:?} (ik's type {IK_SQRT_SOFTPLUS})"
        )),
        other => other
            .as_unsigned()
            .ok_or_else(|| format!("is {other:?}, not a scoring function")),
    }
}

/// The router's scoring function, as the number the engine reads.
pub const GATING_FUNC: Fact = Fact::new(
    "expert_gating_func",
    &[("expert_scoring_func", Conv::Plain)],
    Kind::Converted(scoring),
);
/// The candidate mask's source layer.
pub const CANDIDATE_SOURCE: Fact = Fact::new(
    "attention.candidate_source_layer_id",
    &[("attention.indexer.candidate_source_layer", Conv::Plain)],
    Kind::Count,
);
/// The candidate mask's rows per block.
pub const CANDIDATE_BLOCK: Fact = Fact::new(
    "attention.candidate_block_size",
    &[("attention.indexer.candidate_block_size", Conv::Plain)],
    Kind::Count,
);
/// The candidate mask's kept blocks.
pub const CANDIDATE_TOPK: Fact = Fact::new(
    "attention.candidate_topk_blocks",
    &[("attention.indexer.candidate_top_k_blocks", Conv::Plain)],
    Kind::Count,
);

/// The path [`COMPANION_ENV`] holds, `None` when unset or empty.
fn companion_path() -> Option<PathBuf> {
    std::env::var_os(COMPANION_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// V4.1's Engram: the facts under each publisher's spelling, the tensor of a
/// site's table, and the companion file.
pub static ENGRAM: EngramFacts = EngramFacts {
    layer_ids: Fact::new(
        "engram.layer_ids",
        &[("ple.layers", Conv::Plain)],
        Kind::Counts,
    ),
    head_count: Fact::new(
        "engram.head_count",
        &[
            ("engram_n_heads", Conv::Plain),
            ("ple.heads_per_ngram", Conv::Plain),
        ],
        Kind::Count,
    ),
    max_ngram: Fact::new(
        "engram.max_ngram_size",
        &[
            ("engram_max_ngram_size", Conv::Plain),
            ("ple.ngram_size", Conv::Plain),
        ],
        Kind::Count,
    ),
    key_length: Fact::new(
        "engram.key_length",
        &[
            ("engram_head_dim", Conv::Plain),
            ("embedding_length_per_layer_input", Conv::Plain),
        ],
        Kind::Count,
    ),
    pad_id: Fact::new(
        "engram.pad_id",
        &[
            ("engram_pad_token_id", Conv::Plain),
            ("engram.pad_token_id", Conv::Plain),
        ],
        Kind::Count,
    ),
    rows: Fact::new("engram.num_embeddings", &[], Kind::Counts),
    file: Fact::new("engram.file", &[], Kind::Text),
    source: Fact::new("engram.source", &[], Kind::Text),
    multipliers: HashFact::new(
        Fact::new(
            "engram.multipliers",
            &[("ple.layer_multipliers", Conv::Plain)],
            Kind::Counts,
        ),
        HASH_TENSORS[3],
    ),
    primes: HashFact::new(
        Fact::new(
            "engram.primes",
            &[("ple.head_vocab_sizes", Conv::Plain)],
            Kind::Counts,
        ),
        HASH_TENSORS[1],
    ),
    // `ple.head_offsets` counts rows across the sites' tables, one after the other.
    offsets: HashFact::new(
        Fact::new(
            "engram.offsets",
            &[("ple.head_offsets", Conv::PerSite)],
            Kind::Counts,
        ),
        HASH_TENSORS[2],
    ),
    token_map: HashFact::new(
        Fact::new("engram.token_map", &[], Kind::Counts),
        HASH_TENSORS[0],
    ),
    table: table_name,
    companion: Companion {
        arch: ENGRAM_ARCH,
        env: COMPANION_ENV,
        explicit: companion_path,
    },
};

/// The router's scoring function as the number the engine reads
/// ([`IK_SQRT_SOFTPLUS`]).
pub fn gating_func(meta: &[(String, Value)], arch: &str) -> Result<u64, FactError> {
    crate::facts::count(meta, arch, &GATING_FUNC)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::engram::attach_companion_with;
    use crate::facts::{Norm, count, hash_constant, resolve};
    use crate::write::{TensorDecl, write_file};
    use crate::{GENERAL_ARCHITECTURE, GgmlType, LoadError, Split, inventory_of};

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("v41-facts-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn kv(key: &str, v: Value) -> (String, Value) {
        (key.to_string(), v)
    }

    fn text(v: &str) -> Value {
        Value::String(v.to_string())
    }

    fn u64s(v: &[u64]) -> Value {
        Value::Array(v.iter().copied().map(Value::U64).collect())
    }

    /// `<ARCH>.<suffix>` pairs on top of `general.architecture`.
    fn pairs(model: &[(&str, Value)]) -> Vec<(String, Value)> {
        let mut out = vec![kv(GENERAL_ARCHITECTURE, text(ARCH))];
        out.extend(
            model
                .iter()
                .map(|(k, v)| kv(&format!("{ARCH}.{k}"), v.clone())),
        );
        out
    }

    /// A tensor of `ty` holding `values` as little-endian integers of `width`.
    fn int_tensor_of(
        name: &str,
        ty: GgmlType,
        dims: &[u64],
        bytes: Vec<u8>,
    ) -> (TensorDecl, Vec<u8>) {
        let decl = TensorDecl {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id: ty.as_u32(),
            nbytes: bytes.len() as u64,
        };
        (decl, bytes)
    }

    fn i64_tensor(name: &str, dims: &[u64], values: &[i64]) -> (TensorDecl, Vec<u8>) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        int_tensor_of(name, GgmlType::I64, dims, bytes)
    }

    fn i32_tensor(name: &str, dims: &[u64], values: &[i32]) -> (TensorDecl, Vec<u8>) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        int_tensor_of(name, GgmlType::I32, dims, bytes)
    }

    /// The model file `<d>/<name>.gguf` of `kvs` and `tensors`.
    fn write(
        d: &Path,
        name: &str,
        kvs: &[(String, Value)],
        tensors: &[(TensorDecl, Vec<u8>)],
    ) -> PathBuf {
        let p = d.join(name);
        write_file(&p, kvs, tensors);
        p
    }

    /// The header facts of a model file carrying `model`'s keys, as
    /// [`Split::keys`] reads them.
    fn keys_of(name: &str, model: &[(&str, Value)]) -> Vec<(String, Value)> {
        let d = dir(name);
        let p = write(&d, "m.gguf", &pairs(model), &[]);
        let split = Split::open_model(&p).unwrap();
        let keys = split.keys().to_vec();
        std::fs::remove_dir_all(&d).unwrap();
        keys
    }

    #[test]
    fn a_fact_is_read_under_its_canonical_key_or_a_publishers_spelling() {
        let canonical = keys_of("canon", &[("expert_gating_func", Value::U64(4))]);
        assert_eq!(gating_func(&canonical, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let ista = keys_of(
            "ista-score",
            &[("expert_scoring_func", text("sqrtsoftplus"))],
        );
        assert_eq!(gating_func(&ista, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let ista = keys_of("ista-heads", &[("engram_n_heads", Value::U32(8))]);
        assert_eq!(count(&ista, ARCH, &ENGRAM.head_count), Ok(8));
        let unsloth = keys_of(
            "ple-heads",
            &[
                ("ple.heads_per_ngram", Value::U32(8)),
                ("engram.pad_token_id", Value::I32(2)),
            ],
        );
        assert_eq!(count(&unsloth, ARCH, &ENGRAM.head_count), Ok(8));
        assert_eq!(count(&unsloth, ARCH, &ENGRAM.pad_id), Ok(2));
        let none = keys_of("absent", &[]);
        assert_eq!(
            count(&none, ARCH, &ENGRAM.head_count)
                .unwrap_err()
                .to_string(),
            "deepseek41.engram.head_count is absent (also looked for deepseek41.engram_n_heads, deepseek41.ple.heads_per_ngram)"
        );
    }

    #[test]
    fn two_spellings_that_agree_read_once_and_that_differ_are_refused_by_name() {
        let both = keys_of(
            "agree",
            &[
                ("expert_gating_func", Value::U64(4)),
                ("expert_scoring_func", text("sqrtsoftplus")),
            ],
        );
        assert_eq!(gating_func(&both, ARCH), Ok(IK_SQRT_SOFTPLUS));
        let clash = keys_of(
            "clash",
            &[
                ("expert_gating_func", Value::U64(1)),
                ("expert_scoring_func", text("sqrtsoftplus")),
            ],
        );
        let e = gating_func(&clash, ARCH).unwrap_err();
        assert!(
            matches!(&e, FactError::Conflict { first, second, .. }
                if first == "deepseek41.expert_gating_func"
                    && second == "deepseek41.expert_scoring_func"),
            "{e}"
        );
        let heads = keys_of(
            "heads",
            &[
                ("engram.head_count", Value::U32(8)),
                ("engram_n_heads", Value::U32(4)),
            ],
        );
        let e = count(&heads, ARCH, &ENGRAM.head_count).unwrap_err();
        assert_eq!(
            e.to_string(),
            "deepseek41.engram.head_count is 8 but deepseek41.engram_n_heads is 4: the file states one fact two ways"
        );
    }

    #[test]
    fn a_scoring_function_that_is_not_sqrtsoftplus_is_refused_by_name() {
        for other in ["softmax", "sigmoid", ""] {
            let k = keys_of("score", &[("expert_scoring_func", text(other))]);
            let e = gating_func(&k, ARCH).unwrap_err();
            assert!(
                matches!(&e, FactError::Bad { key, detail }
                    if key == "deepseek41.expert_scoring_func" && detail.contains(&format!("{other:?}"))),
                "{other:?}: {e}"
            );
        }
        let k = keys_of("score-bool", &[("expert_scoring_func", Value::Bool(true))]);
        assert!(gating_func(&k, ARCH).is_err());
    }

    #[test]
    fn unsloths_head_offsets_count_from_each_sites_first_row() {
        let k = keys_of(
            "ple-offsets",
            &[("ple.head_offsets", u64s(&[0, 10, 20, 100, 110, 120]))],
        );
        let got = resolve(&k, ARCH, &ENGRAM.offsets.kv, 2).unwrap().unwrap().1;
        assert_eq!(got, Norm::Counts(vec![0, 10, 20, 0, 10, 20]));
        let e = resolve(&k, ARCH, &ENGRAM.offsets.kv, 4).unwrap_err();
        assert!(matches!(e, FactError::Bad { .. }), "{e}");
        // The test writer refuses an empty array (no element type); a file's reader yields one.
        let k = pairs(&[("ple.head_offsets", Value::Array(Vec::new()))]);
        let e = resolve(&k, ARCH, &ENGRAM.offsets.kv, 2).unwrap_err();
        assert!(matches!(e, FactError::Bad { .. }), "{e}");
    }

    /// A file with the four hash tensors of a one-site toy Engram (2 columns,
    /// 3 n-gram slots, 5 tokens) and `extra` keys.
    fn hash_file(d: &Path, name: &str, extra: &[(&str, Value)]) -> PathBuf {
        write(
            d,
            name,
            &pairs(extra),
            &[
                i32_tensor(HASH_TENSORS[0], &[5], &[0, 1, 2, 1, 0]),
                i64_tensor(HASH_TENSORS[1], &[2, 1], &[7, 11]),
                i64_tensor(HASH_TENSORS[2], &[2, 1], &[0, 7]),
                i64_tensor(HASH_TENSORS[3], &[3, 1], &[3, 5, -1]),
            ],
        )
    }

    #[test]
    fn hash_constants_are_read_from_their_tensors_or_their_arrays() {
        let d = dir("hash");
        let inv = inventory_of(hash_file(&d, "t.gguf", &[])).unwrap();
        let mult = hash_constant(&inv, ARCH, &ENGRAM.multipliers, 1).unwrap();
        // An I64 entry is its two's-complement bit pattern, as the hash's wrapping u64 reads it.
        assert_eq!(mult.values, vec![3, 5, u64::MAX]);
        assert_eq!(mult.dims, Some(vec![3, 1]));
        let map = hash_constant(&inv, ARCH, &ENGRAM.token_map, 1).unwrap();
        assert_eq!((map.values, map.dims), (vec![0, 1, 2, 1, 0], Some(vec![5])));

        let same = inventory_of(hash_file(
            &d,
            "same.gguf",
            &[("engram.primes", u64s(&[7, 11]))],
        ))
        .unwrap();
        assert_eq!(
            hash_constant(&same, ARCH, &ENGRAM.primes, 1)
                .unwrap()
                .values,
            vec![7, 11]
        );
        let apart = inventory_of(hash_file(
            &d,
            "apart.gguf",
            &[("engram.primes", u64s(&[7, 13]))],
        ))
        .unwrap();
        let e = hash_constant(&apart, ARCH, &ENGRAM.primes, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Conflict { first, second, .. }
                if first == "deepseek41.engram.primes" && second == HASH_TENSORS[1]),
            "{e}"
        );

        let kv_only = inventory_of(write(
            &d,
            "kv.gguf",
            &pairs(&[("engram.offsets", u64s(&[0, 7]))]),
            &[],
        ))
        .unwrap();
        let c = hash_constant(&kv_only, ARCH, &ENGRAM.offsets, 1).unwrap();
        assert_eq!((c.values, c.dims), (vec![0, 7], None));
        let e = hash_constant(&kv_only, ARCH, &ENGRAM.primes, 1).unwrap_err();
        assert!(matches!(e, FactError::Absent { .. }), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_hash_tensor_of_the_wrong_type_or_a_negative_id_is_refused_by_name() {
        let d = dir("hash-bad");
        let f32_primes = TensorDecl {
            name: HASH_TENSORS[1].to_string(),
            dims: vec![2, 1],
            type_id: GgmlType::F32.as_u32(),
            nbytes: 8,
        };
        let inv = inventory_of(write(
            &d,
            "f.gguf",
            &pairs(&[]),
            &[(f32_primes, vec![0; 8])],
        ))
        .unwrap();
        let e = hash_constant(&inv, ARCH, &ENGRAM.primes, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Bad { key, .. } if key == HASH_TENSORS[1]),
            "{e}"
        );

        let neg = write(
            &d,
            "n.gguf",
            &pairs(&[]),
            &[i32_tensor(HASH_TENSORS[0], &[2], &[1, -2])],
        );
        let e = hash_constant(&inventory_of(neg).unwrap(), ARCH, &ENGRAM.token_map, 1).unwrap_err();
        assert!(
            matches!(&e, FactError::Bad { key, detail } if key == HASH_TENSORS[0] && detail.contains("-2")),
            "{e}"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The model file of a one-site Engram (block 1, rows 3 of width 4) that
    /// names `file` as its companion, plus the companion `engram.gguf`
    /// holding the table.
    fn pair_of_files(d: &Path, rows_in_header: u64, rows_in_table: u64) -> PathBuf {
        let model = write(
            d,
            "model.gguf",
            &pairs(&[
                ("engram.layer_ids", Value::Array(vec![Value::I32(1)])),
                ("engram.key_length", Value::U32(4)),
                ("engram.num_embeddings", u64s(&[rows_in_header])),
                ("engram.file", text("engram.gguf")),
            ]),
            &[],
        );
        let table = TensorDecl {
            name: table_name(1),
            dims: vec![4, rows_in_table],
            type_id: GgmlType::F32.as_u32(),
            nbytes: 16 * rows_in_table,
        };
        let comp_kvs = vec![kv(GENERAL_ARCHITECTURE, text(ENGRAM_ARCH))];
        write(
            d,
            "engram.gguf",
            &comp_kvs,
            &[(table, vec![9; 16 * rows_in_table as usize])],
        );
        model
    }

    fn attached(model: &Path, explicit: Option<&Path>) -> Result<Split, LoadError> {
        let mut split = Split::open_model(model)?;
        attach_companion_with(&mut split, ARCH, &ENGRAM, explicit)?;
        Ok(split)
    }

    #[test]
    fn the_companion_joins_the_split_as_a_further_shard() {
        let d = dir("join");
        let model = pair_of_files(&d, 3, 3);
        let split = attached(&model, None).unwrap();
        assert_eq!(split.shard_count(), 2);
        assert_eq!(split.shard_path(1), Some(d.join("engram.gguf").as_path()));
        let (shard, info) = split.find(&table_name(1)).unwrap();
        assert_eq!((shard, info.dims.clone()), (1, vec![4, 3]));
        assert_eq!(
            split.data(&table_name(1)).unwrap(),
            vec![9u8; 48].as_slice()
        );
        // A table the header names and no file holds stays a lookup miss.
        assert!(split.find(&table_name(14)).is_none());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_companion_is_looked_for_beside_the_model_then_in_the_sibling_directory() {
        let root = dir("lookup");
        let model_dir = root.join("model");
        let sibling = root.join("engram");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        // The model sits in model/, its companion in the sibling directory named by the file's stem.
        let model = pair_of_files(&model_dir, 3, 3);
        std::fs::rename(model_dir.join("engram.gguf"), sibling.join("engram.gguf")).unwrap();
        let split = attached(&model, None).unwrap();
        assert_eq!(
            split.shard_path(1).map(|p| p.canonicalize().unwrap()),
            Some(sibling.join("engram.gguf").canonicalize().unwrap())
        );

        // Beside the model comes first.
        std::fs::copy(sibling.join("engram.gguf"), model_dir.join("engram.gguf")).unwrap();
        let split = attached(&model, None).unwrap();
        assert_eq!(
            split.shard_path(1),
            Some(model_dir.join("engram.gguf").as_path())
        );

        // Neither present: the error names both paths.
        std::fs::remove_file(sibling.join("engram.gguf")).unwrap();
        std::fs::remove_file(model_dir.join("engram.gguf")).unwrap();
        let e = attached(&model, None).err().unwrap().to_string();
        let beside = model_dir.join("engram.gguf");
        let there = model_dir.join("..").join("engram").join("engram.gguf");
        assert!(
            e.contains(&beside.display().to_string()) && e.contains(&there.display().to_string()),
            "{e}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_companion_that_does_not_match_the_header_is_refused_by_name() {
        let cases: [(&str, u64, u64, &str); 2] = [
            ("rows", 4, 3, "has 3 rows"),
            ("rows-more", 2, 3, "has 3 rows"),
        ];
        for (name, header_rows, table_rows, want) in cases {
            let d = dir(name);
            let model = pair_of_files(&d, header_rows, table_rows);
            let e = attached(&model, None).err().unwrap().to_string();
            assert!(
                e.contains(want) && e.contains("engram.num_embeddings"),
                "{name}: {e}"
            );
            std::fs::remove_dir_all(&d).unwrap();
        }

        // Another architecture, a table of no Engram layer, a table of another width.
        let d = dir("shape");
        let model = pair_of_files(&d, 3, 3);
        let comp = d.join("engram.gguf");
        let t = |name: String, dims: Vec<u64>| TensorDecl {
            name,
            dims,
            type_id: GgmlType::F32.as_u32(),
            nbytes: 4 * 12,
        };
        let keep = |arch: &str, tensor: TensorDecl| {
            write_file(
                &comp,
                &[kv(GENERAL_ARCHITECTURE, text(arch))],
                &[(tensor, vec![0; 48])],
            );
            attached(&model, None).err().unwrap().to_string()
        };
        assert!(
            keep("deepseek41", t(table_name(1), vec![4, 3])).contains("want \"deepseek41-engram\"")
        );
        assert!(
            keep(ENGRAM_ARCH, t(table_name(2), vec![4, 3]))
                .contains("not the table of an Engram layer")
        );
        assert!(keep(ENGRAM_ARCH, t(table_name(1), vec![6, 2])).contains("want [4, rows]"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn an_explicit_companion_is_taken_and_refused_for_a_model_that_needs_none() {
        let d = dir("explicit");
        let model = pair_of_files(&d, 3, 3);
        // The header names no file, so only the explicit path finds it.
        let mut split = Split::open_model(&model).unwrap();
        attach_companion_with(&mut split, ARCH, &ENGRAM, Some(&d.join("engram.gguf"))).unwrap();
        assert_eq!(split.shard_count(), 2);
        let mut split = Split::open_model(&model).unwrap();
        let e = attach_companion_with(&mut split, ARCH, &ENGRAM, Some(&d.join("none.gguf")))
            .unwrap_err();
        assert!(
            e.to_string().contains(COMPANION_ENV) && e.to_string().contains("none.gguf"),
            "{e}"
        );

        // A model whose split holds every table: the explicit file would be ignored.
        let full = write(
            &d,
            "full.gguf",
            &pairs(&[
                ("engram.layer_ids", Value::Array(vec![Value::I32(1)])),
                ("engram.key_length", Value::U32(4)),
            ]),
            &[(
                TensorDecl {
                    name: table_name(1),
                    dims: vec![4, 3],
                    type_id: GgmlType::F32.as_u32(),
                    nbytes: 48,
                },
                vec![0; 48],
            )],
        );
        let mut split = Split::open_model(&full).unwrap();
        assert!(attach_companion_with(&mut split, ARCH, &ENGRAM, None).is_ok());
        assert_eq!(split.shard_count(), 1);
        let e = attach_companion_with(&mut split, ARCH, &ENGRAM, Some(&d.join("engram.gguf")))
            .unwrap_err();
        assert!(e.to_string().contains("holds every Engram table"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
