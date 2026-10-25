//! V4.1 open gate: a published V4.1 set opens as it is, whichever publisher's
//! spellings its header uses and wherever it keeps its Engram tables. Three
//! contracts, one test each; each prints one line per fact it read, then fails
//! with the whole list of what differs.
//!
//! 1. `hw_ds41_open_gives_every_tensor_a_role` — `Split::open` with no flag
//!    finds the Engram companion file a header names (`deepseek41.engram.file`)
//!    and seats its tables as the model's own; `classify` accepts every tensor,
//!    the Engram hash constants a publisher stores as tensors included.
//! 2. `hw_ds41_open_reads_the_header_facts` — the router's scoring function, the
//!    Engram scalars and the candidate mask resolve to the values the
//!    `Q3_K_M` set's header states under the canonical keys.
//! 3. `hw_ds41_open_engram_matches_the_header` — the four hash constants (keys
//!    or tensors) are the `Q3_K_M` set's, the site tables are the rows the hash
//!    partitions and the header counts, and `Engram::open` over the split's
//!    shard paths (what the card engine passes) finds both tables.
//!
//! The pins are the 1017 `Q3_K_M` set's, so the gate passes on that set and on
//! every other publisher's file of the same model: ISTA-DASLab's 2.00 bpw
//! shards state the same facts under other keys and keep the tables in a file
//! of their own. `BLOOMERY_V41_MODEL` names the first shard to open;
//! `BLOOMERY_V41_ENGRAM` the companion file when it is not beside the model.
//!
//! `hw_`: needs the V4.1 shards on the box (`just gate-ds41-open`). Headers
//! only: seconds.

use std::collections::BTreeMap;
use std::fmt::{Debug, Write as _};

use engram::{Engram, hash::Hash};
use gguf::v41_facts::{self, ARCH, ENGRAM_FILE, ENGRAM_ROWS, HASH_TENSORS};
use gguf::{Split, Value};
use model::arch::deepseek41::hparams::{
    CANDIDATE_BLOCK_SIZE, CANDIDATE_SOURCE_LAYER, CANDIDATE_TOPK_BLOCKS, Hparams, Score,
};
use model::arch::deepseek41::roles::classify;
use model::placement::{Role, workstation};
use sha2::{Digest, Sha256};

/// The blocks that carry an Engram table, `engram.layer_ids` of the `Q3_K_M` set.
const LAYER_IDS: [usize; 2] = [1, 14];
/// `engram.head_count`, `engram.max_ngram_size`, `engram.key_length` and `engram.pad_id` of the same set.
const HEADS: usize = 8;
const MAX_NGRAM: usize = 4;
const KEY_LENGTH: usize = 256;
const PAD_ID: u64 = 2;

/// sha256 of each hash constant's little-endian entries (8 bytes for the
/// multipliers, primes and offsets, 4 for the token map), both sites in order,
/// as the 1017 `Q3_K_M` header's `engram.*` arrays state them: read by a
/// script that shares no code with the engine (a struct unpack of the array
/// bytes); ISTA-DASLab's four tensors give the same four digests.
const DIGESTS: [(&str, &str); 4] = [
    (
        "multipliers",
        "7345f44ec93e965df6581af59c78a5dd6efba7be2e855486450a6594d7ebd7c6",
    ),
    (
        "primes",
        "ed542f58c4b3593c4120e899620231785b695dfc61bcc3d04b61b113abd9cf9e",
    ),
    (
        "offsets",
        "edf229962df441c86f9c20cbc126579dd548421e231250b146ee6f5bea0b2f8b",
    ),
    (
        "token_map",
        "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481",
    ),
];

struct Open {
    path: String,
    split: Split,
    hp: Hparams,
}

fn open() -> Open {
    let path = workstation::model_v41();
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let hp = Hparams::read(&split).unwrap_or_else(|e| panic!("hyperparameters of {path}: {e}"));
    Open { path, split, hp }
}

/// One compared value: a line of the printed table, and an entry of `bad` when it differs.
fn row<T: PartialEq + Debug>(out: &mut String, bad: &mut Vec<String>, what: &str, got: T, want: T) {
    let ok = got == want;
    let _ = writeln!(
        out,
        "fact {what} = {got:?}{}",
        if ok { "" } else { "  <-- DIFFERS" }
    );
    if !ok {
        bad.push(format!("{what}: {got:?}, want {want:?}"));
    }
}

fn finish(out: &str, bad: &[String]) {
    print!("{out}");
    assert!(bad.is_empty(), "differences:\n  {}", bad.join("\n  "));
}

/// The model shards `split.count` names: the shards before any companion.
fn model_shards(split: &Split) -> usize {
    split
        .shard(0)
        .and_then(|g| g.value("split.count"))
        .and_then(Value::as_unsigned)
        .map_or(1, |c| usize::try_from(c).expect("a shard count"))
}

#[test]
#[ignore = "hw"]
fn hw_ds41_open_gives_every_tensor_a_role() {
    let o = open();
    let (mut out, mut bad) = (String::new(), Vec::new());
    let shards = model_shards(&o.split);
    let _ = writeln!(out, "fact file = {}", o.path);

    let model = classify(&o.split, &o.hp).unwrap_or_else(|e| panic!("classify {}: {e}", o.path));
    let mut roles: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &model.tensors {
        let r = roles.entry(t.role.to_string()).or_default();
        r.0 += 1;
        r.1 += t.file_bytes;
    }
    for (role, (n, bytes)) in &roles {
        let _ = writeln!(out, "fact role {role} = {n} tensors, {bytes} bytes");
    }
    row(
        &mut out,
        &mut bad,
        "tensors classified",
        model.tensors.len(),
        o.split.tensor_count(),
    );

    // The hash constants are tensors in some sets, header arrays in others: all four or none, each with its role.
    let stored: Vec<&str> = HASH_TENSORS
        .into_iter()
        .filter(|n| o.split.find(n).is_some())
        .collect();
    row(
        &mut out,
        &mut bad,
        "hash tensors stored (all four or none)",
        matches!(stored.len(), 0 | 4),
        true,
    );
    let hash_roles = model
        .tensors
        .iter()
        .filter(|t| t.role == Role::EngramHash)
        .count();
    row(
        &mut out,
        &mut bad,
        "tensors with the engram_hash role",
        hash_roles,
        stored.len(),
    );

    // The tables sit in the model's shards or, when the header names a file for them, in the companion.
    let tables: Vec<_> = model
        .tensors
        .iter()
        .filter(|t| t.role == Role::EngramTable)
        .collect();
    let layers: Vec<_> = tables.iter().filter_map(|t| t.layer).collect();
    row(
        &mut out,
        &mut bad,
        "engram table layers",
        layers,
        LAYER_IDS.to_vec(),
    );
    let in_companion = tables.iter().filter(|t| t.shard >= shards).count();
    let named = v41_facts::text_opt(o.split.keys(), ARCH, &ENGRAM_FILE).expect("engram.file");
    row(
        &mut out,
        &mut bad,
        "tables in the companion",
        in_companion,
        if named.is_some() { tables.len() } else { 0 },
    );
    row(
        &mut out,
        &mut bad,
        "shards",
        o.split.shard_count(),
        shards + usize::from(in_companion > 0),
    );
    if let Some(p) = o.split.shard_path(shards).filter(|_| in_companion > 0) {
        let _ = writeln!(out, "fact companion = {}", p.display());
    }
    finish(&out, &bad);
}

#[test]
#[ignore = "hw"]
fn hw_ds41_open_reads_the_header_facts() {
    let o = open();
    let (mut out, mut bad) = (String::new(), Vec::new());
    let hp = &o.hp;
    row(
        &mut out,
        &mut bad,
        "expert_gating_func",
        hp.experts.score,
        Score::SqrtSoftplus,
    );
    let en = hp.engram.as_ref().expect("engram sites");
    row(
        &mut out,
        &mut bad,
        "engram.layer_ids",
        en.layer_ids.clone(),
        LAYER_IDS.to_vec(),
    );
    row(&mut out, &mut bad, "engram.head_count", en.n_head, HEADS);
    row(
        &mut out,
        &mut bad,
        "engram.max_ngram_size",
        en.max_ngram,
        MAX_NGRAM,
    );
    row(
        &mut out,
        &mut bad,
        "engram.key_length",
        en.key_length,
        KEY_LENGTH,
    );
    let mask = hp.candidates.as_ref().expect("candidate mask");
    row(
        &mut out,
        &mut bad,
        "candidate source layer",
        mask.source_layer,
        CANDIDATE_SOURCE_LAYER,
    );
    row(
        &mut out,
        &mut bad,
        "candidate block size",
        mask.block_size,
        CANDIDATE_BLOCK_SIZE,
    );
    row(
        &mut out,
        &mut bad,
        "candidate top-k blocks",
        mask.topk_blocks,
        CANDIDATE_TOPK_BLOCKS,
    );
    // The set's own types, printed for the reader: they differ between publishers.
    let _ = writeln!(out, "fact token_embd type = {}", hp.rows.token_embd);
    let _ = writeln!(out, "fact engram table type = {:?}", hp.rows.engram);
    finish(&out, &bad);
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[test]
#[ignore = "hw"]
fn hw_ds41_open_engram_matches_the_header() {
    let o = open();
    let (mut out, mut bad) = (String::new(), Vec::new());
    let first = o.split.shard_path(0).expect("a first shard");
    let inv =
        gguf::inventory_of(first).unwrap_or_else(|e| panic!("header of {}: {e}", first.display()));
    let hash = Hash::from_gguf(&inv)
        .unwrap_or_else(|e| panic!("hash constants of {}: {e}", first.display()));
    row(
        &mut out,
        &mut bad,
        "hash layer ids",
        hash.layer_ids().to_vec(),
        LAYER_IDS.map(|l| l as u32).to_vec(),
    );
    row(&mut out, &mut bad, "hash n_heads", hash.n_heads(), HEADS);
    row(&mut out, &mut bad, "hash n_gram", hash.n_gram(), MAX_NGRAM);
    row(
        &mut out,
        &mut bad,
        "hash key_length",
        hash.key_length() as usize,
        KEY_LENGTH,
    );
    row(&mut out, &mut bad, "hash pad_id", hash.pad_id(), PAD_ID);
    if let Err(e) = hash.check_vocab(o.hp.n_vocab) {
        bad.push(format!("token map against the vocabulary: {e}"));
    }

    let le = |values: &[u64]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let per_site =
        |f: &dyn Fn(usize) -> Vec<u8>| (0..hash.sites()).flat_map(f).collect::<Vec<u8>>();
    let got = [
        per_site(&|s| le(hash.multipliers(s).expect("site"))),
        per_site(&|s| le(hash.primes(s).expect("site"))),
        per_site(&|s| le(hash.offsets(s).expect("site"))),
        hash.token_map()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    ];
    for ((name, want), bytes) in DIGESTS.iter().zip(&got) {
        row(
            &mut out,
            &mut bad,
            &format!("sha256 of {name}"),
            digest(bytes),
            (*want).to_string(),
        );
    }

    // The tables: the hash partitions exactly the rows a table holds, and a header that counts them agrees.
    let paths = (0..o.split.shard_count()).filter_map(|i| o.split.shard_path(i));
    let engram = Engram::open(paths).unwrap_or_else(|e| panic!("engram tables of {}: {e}", o.path));
    let counted =
        v41_facts::counts_opt(o.split.keys(), ARCH, &ENGRAM_ROWS).expect("engram.num_embeddings");
    for (e, site) in engram.sites().iter().enumerate() {
        let want = hash.partition_rows(e).expect("site");
        row(
            &mut out,
            &mut bad,
            &format!("{} rows", site.name()),
            site.rows(),
            want,
        );
        if let Some(counted) = &counted {
            row(
                &mut out,
                &mut bad,
                &format!("{} rows in the header", site.name()),
                Some(&site.rows()),
                counted.get(e),
            );
        }
        let _ = writeln!(
            out,
            "fact {} = type {}, {} bytes a row, in {}",
            site.name(),
            site.type_id(),
            site.row_bytes(),
            site.path().display()
        );
    }
    finish(&out, &bad);
}
