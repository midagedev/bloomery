//! The Hy4-flash fixture's plan (`model::arch::hy4::fixture`) against the real file's
//! header, with no tensor's bytes: the plan of the five-layer cut of the checked-in header
//! (`hy4_hdr.kv.json`, `hy4_tensors.tsv`), laid out as a header-only file and read through the
//! engine's own reader. What the cut keeps and what it minifies is derived here from the real
//! tables and the pins of the real file's structure, not from the fixture's own tables.

use gguf::{GgmlType, Split, Value};
use model::arch::hy4::fixture as hy4;
use model::arch::hy4::hparams::Hparams;
use model::fixture::{FilePlan, FixtureSpec, Rule};

#[path = "common/hy4_header.rs"]
mod hy4_header;
use hy4_header::{TENSORS, header};

/// Whether layer `l` of the real file runs sparse attention: 0, 1 and every fourth from 5.
fn dsa(l: usize) -> bool {
    l < 2 || (l >= 5 && (l - 5).is_multiple_of(4))
}

/// The real file's engram sites.
const SITES: [usize; 4] = [1, 11, 21, 31];

/// The fixture's header as a `Split` holding no tensor's bytes: the plan's one shard laid out.
fn fixture_header(plan: &FilePlan) -> Split {
    let layouts = plan.layouts().expect("the plan lays out");
    assert_eq!(layouts.len(), 1, "the fixture is one shard");
    let path =
        std::env::temp_dir().join(format!("bloomery-hy4-fixture-{}.gguf", std::process::id()));
    Split::header_only(path, &layouts[0].1).expect("the fixture's header-only split opens")
}

/// The smallest `n` primes above `floor`, by trial division: the engram buckets' sizes.
fn primes_above(floor: u64, n: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = floor;
    while out.len() < n {
        p += 1;
        if (2..)
            .take_while(|d| d * d <= p)
            .all(|d| !p.is_multiple_of(d))
        {
            out.push(p);
        }
    }
    out
}

/// The type id a header's tensor description carries for `name`: the name (a length and its
/// bytes), the dims (a count and that many `u64`), then the id.
fn wire_type_of(header: &[u8], name: &str) -> u32 {
    let at = header
        .windows(name.len())
        .position(|w| w == name.as_bytes())
        .unwrap_or_else(|| panic!("{name} is in the header"));
    let mut i = at + name.len();
    let n_dims = u32::from_le_bytes(header[i..i + 4].try_into().unwrap()) as usize;
    i += 4 + 8 * n_dims;
    u32::from_le_bytes(header[i..i + 4].try_into().unwrap())
}

/// The real tensor table as `(name, type name, dims)`.
fn real_tensors() -> Vec<(String, String, Vec<u64>)> {
    TENSORS
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            let (n, t, d) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
            (
                n.to_string(),
                t.to_string(),
                d.split('x').map(|x| x.parse().unwrap()).collect(),
            )
        })
        .collect()
}

/// The fixture clause. The plan of the five layers `[0, 1, 6, 11, 41]`, laid out as a header-only
/// file, reads through the engine's reader as the cut: the kinds are the real file's at the map's
/// layers (the sparse layer with the dense feed-forward, a sparse layer with a site, a window layer,
/// a window layer with a site, the last sparse layer), every tensor is the real layer's in its type
/// and dims but the minified axes, the engram table is the scale-less F8 table (id 43) of the kept
/// sites' primes, and the next-token block is dropped. A map of other kinds or stack types, a key
/// left at the source's, a removed, reshaped or retyped tensor and a missing key are each refused
/// by name.
#[test]
fn hy4_fixture_cut() {
    let source = header("hy4-fixture-source");
    let spec = hy4::spec();
    let plan = model::fixture::plan(&spec, &source, None, &spec.options())
        .unwrap_or_else(|e| panic!("plan: {e}"));
    let fx = fixture_header(&plan.target);
    let hp = Hparams::read(&fx).unwrap_or_else(|e| panic!("the fixture's hparams: {e}"));
    let read = model::arch::spec(&fx).unwrap_or_else(|e| panic!("the fixture's spec: {e}"));
    let s = &read.spec;
    let mut bad = Vec::new();

    // The kinds, from the pins of the real file.
    let map = [0usize, 1, 6, 11, 41];
    if hy4::LAYER_MAP != map {
        bad.push(format!("the map {:?}, the cut's {map:?}", hy4::LAYER_MAP));
    }
    let want = |f: fn(usize) -> bool| map.iter().map(|&l| f(l)).collect::<Vec<_>>();
    let got = |f: &dyn Fn(usize) -> bool| (0..5).map(f).collect::<Vec<_>>();
    if got(&|f| hp.dsa(f)) != want(dsa) {
        bad.push(format!("sparse {:?}", got(&|f| hp.dsa(f))));
    }
    if got(&|f| hp.dense(f)) != [true, false, false, false, false] {
        bad.push(format!("dense {:?}", got(&|f| hp.dense(f))));
    }
    if got(&|f| hp.site(f)) != want(|l| SITES.contains(&l)) {
        bad.push(format!("sites {:?}", got(&|f| hp.site(f))));
    }
    let heads: Vec<usize> = (0..5).map(|f| hp.heads(f)).collect();
    if heads != [32, 32, 64, 64, 32] {
        bad.push(format!("heads {heads:?}"));
    }
    if (hp.n_block, hp.n_layer) != (5, 5) || s.layers.len() != 5 || !s.mtp.is_empty() {
        bad.push(format!(
            "blocks {} layers {} spec layers {}",
            hp.n_block,
            hp.n_layer,
            s.layers.len()
        ));
    }

    // The kept widths and the minified ones.
    let model_widths = (
        hp.n_embd,
        hp.n_vocab,
        hp.n_ctx_train,
        hp.mla.q_lora,
        hp.mla.kv_lora,
        hp.mla.head_v,
        hp.mla.pe,
        hp.latent,
    );
    if model_widths != (3072, 120_832, 1_048_576, 1024, 512, 256, 64, 1536) {
        bad.push(format!("kept model widths {model_widths:?}"));
    }
    let router_widths = (
        hp.hc.streams,
        hp.indexer.heads,
        hp.indexer.d,
        hp.indexer.kpool,
        hp.experts.n_expert,
        hp.experts.n_used,
        hp.experts.n_shared,
        hp.dense_lead,
    );
    if router_widths != (4, 16, 128, 4, 512, 16, 1, 1) {
        bad.push(format!(
            "kept hyper-connection, indexer and router widths {router_widths:?}"
        ));
    }
    let minified = (
        hp.dense_ff,
        hp.experts.ff,
        hp.experts.shared_ff(),
        hp.window,
        hp.indexer.top_k,
    );
    if minified != (1024, 256, 256, 64, 64) {
        bad.push(format!("minified widths {minified:?}"));
    }
    if !hp.indexer.top_k.is_multiple_of(hp.indexer.kpool) {
        bad.push("top_k is not whole pools".into());
    }
    if hp.clamp != [10.0; 5] {
        bad.push(format!("clamps {:?}", hp.clamp));
    }

    // The engram: the first two sites at fixture layers 1 and 3, 8 buckets of each of the two orders
    // over 3-grams, the buckets the primes above 2^14 in one table.
    let eng = hp.engram.as_ref().expect("the fixture has an engram");
    let primes = primes_above(1 << 14, 32);
    let rows: u64 = primes.iter().sum();
    if (
        eng.layers.as_slice(),
        eng.ngram,
        eng.heads,
        eng.key_length,
        eng.rows,
    ) != (&[1usize, 3][..], 3, 8, 96, rows)
    {
        bad.push(format!("engram {eng:?}, rows wanted {rows}"));
    }
    let offsets = |key: &str| match plan.target.kvs.iter().find(|(k, _)| k == key) {
        Some((_, Value::Array(a))) => a.iter().map(|v| v.as_u64().unwrap()).collect::<Vec<u64>>(),
        other => panic!("{key} is {other:?}"),
    };
    if offsets("hy_v4_flash.ple.head_vocab_sizes") != primes {
        bad.push("head_vocab_sizes are not the 32 primes above 2^14".into());
    }
    let mut acc = 0;
    let want_offsets: Vec<u64> = primes
        .iter()
        .map(|p| {
            let at = acc;
            acc += p;
            at
        })
        .collect();
    if offsets("hy_v4_flash.ple.head_offsets") != want_offsets {
        bad.push("head_offsets are not the primes' prefix sums".into());
    }
    if offsets("hy_v4_flash.ple.layer_multipliers").len() != 6 {
        bad.push("layer_multipliers are not two sites of three orders".into());
    }

    // Every tensor of the cut is a real layer's, in its type and dims but the minified axes.
    let real = real_tensors();
    let mut expect: Vec<(String, String, Vec<u64>)> = Vec::new();
    for (name, ty, dims) in &real {
        let renamed = match name.strip_prefix("blk.") {
            Some(rest) => {
                let (l, leaf) = rest.split_once('.').expect("a block tensor");
                let l: usize = l.parse().unwrap();
                match map.iter().position(|&m| m == l) {
                    Some(f) => format!("blk.{f}.{leaf}"),
                    None => continue,
                }
            }
            None => name.clone(),
        };
        let mut dims = dims.clone();
        let tail = |t: &str| renamed.ends_with(t);
        if tail(".ffn_gate.weight") || tail(".ffn_up.weight") {
            dims[1] = 1024;
        } else if tail(".ffn_down.weight") {
            dims[0] = 1024;
        } else if tail(".ffn_gate_exps.weight")
            || tail(".ffn_up_exps.weight")
            || tail(".ffn_gate_shexp.weight")
            || tail(".ffn_up_shexp.weight")
        {
            dims[1] = 256;
        } else if tail(".ffn_down_exps.weight") || tail(".ffn_down_shexp.weight") {
            dims[0] = 256;
        } else if renamed == "per_layer_token_embd.weight" {
            dims[1] = rows;
        }
        expect.push((renamed, ty.clone(), dims));
    }
    let planned: Vec<(String, String, Vec<u64>)> = plan
        .target
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.ty.to_string(), t.dims.clone()))
        .collect();
    // The table's type names are the header's spelling, the plan's the ggml name: the same type.
    let norm = |t: &str| t.to_lowercase().replace('_', "");
    let mut a: Vec<_> = expect
        .iter()
        .map(|(n, t, d)| (n.clone(), norm(t), d.clone()))
        .collect();
    let mut b: Vec<_> = planned
        .iter()
        .map(|(n, t, d)| (n.clone(), norm(t), d.clone()))
        .collect();
    a.sort();
    b.sort();
    if a != b {
        let (only_real, only_plan): (Vec<_>, Vec<_>) = (
            a.iter().filter(|x| !b.contains(x)).collect(),
            b.iter().filter(|x| !a.contains(x)).collect(),
        );
        bad.push(format!(
            "tensors differ: {} derived from the real table, {} planned; derived only {:?}, planned only {:?}",
            a.len(),
            b.len(),
            &only_real[..only_real.len().min(6)],
            &only_plan[..only_plan.len().min(6)]
        ));
    }

    // The F8 table's wire id is ggml's 43 and its rule is the scale-less one; the written header
    // reads back as the scale-less type.
    let table = plan
        .target
        .tensors
        .iter()
        .find(|t| t.name == "per_layer_token_embd.weight")
        .expect("the table is planned");
    if table.ty != GgmlType::F8E4M3
        || table.nbytes != 96 * rows
        || !matches!(table.rule, Rule::E4m3)
        || table.ty.file_id() != 43
    {
        bad.push(format!(
            "table {:?} {} bytes rule {:?}, wanted F8E4M3 {} bytes E4m3",
            table.ty,
            table.nbytes,
            table.rule,
            96 * rows
        ));
    }
    let layouts = plan.target.layouts().expect("the plan lays out");
    let wire = wire_type_of(layouts[0].1.header_bytes(), "per_layer_token_embd.weight");
    if wire != 43 {
        bad.push(format!(
            "the table's type id in the header is {wire}, not ggml's 43"
        ));
    }
    let read_back = fx
        .find("per_layer_token_embd.weight")
        .map(|(_, t)| (t.ty, t.dims.clone()));
    if read_back != Some((GgmlType::F8E4M3, vec![96, rows])) {
        bad.push(format!("the table reads back as {read_back:?}"));
    }
    for gone in ["blk.5.attn_gate.weight", "blk.42.attn_gate.weight"] {
        if fx.find(gone).is_some() {
            bad.push(format!("{gone} is in the fixture"));
        }
    }
    if let Err(e) = hy4::check_kinds(&spec, &fx, &source) {
        bad.push(format!("check_kinds: {e}"));
    }

    // The refusals, each by name.
    let mutant = |what: &str, want: &str, edit: &dyn Fn(&mut FilePlan)| {
        let mut p = plan.target.clone();
        edit(&mut p);
        let split = fixture_header(&p);
        match hy4::check_kinds(&spec, &split, &source) {
            Err(e) if e.to_string().contains(want) => Ok(()),
            Err(e) => Err(format!("{what}: refused as {e}, wanted {want:?}")),
            Ok(_) => Err(format!("{what}: accepted")),
        }
    };
    let set = |p: &mut FilePlan, key: &str, v: Value| {
        p.kvs
            .iter_mut()
            .find(|(k, _)| k == &format!("hy_v4_flash.{key}"))
            .unwrap_or_else(|| panic!("{key} is in the plan"))
            .1 = v;
    };
    let retype = |p: &mut FilePlan, name: &str, ty: GgmlType, dims: Option<Vec<u64>>| {
        let t = p
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is in the plan"));
        t.ty = ty;
        if let Some(dims) = dims {
            t.dims = dims;
        }
        let (_, blck, tsz) = gguf::ggml_type_info(t.ty.as_u32()).expect("a type");
        t.nbytes = gguf::tensor_bytes(blck, tsz, &t.dims).expect("bytes");
    };
    // The one shard's range follows the tensors the plan holds.
    let drop = |p: &mut FilePlan, name: &str| {
        p.tensors.retain(|t| t.name != name);
        p.shards.clear();
        p.shards.push(0..p.tensors.len());
    };
    let results = [
        mutant(
            "the engram's kv projection removed",
            "blk.1.engram_kv.weight",
            &|p| drop(p, "blk.1.engram_kv.weight"),
        ),
        mutant(
            "the head's stream fold removed",
            "output_hc_fn.weight",
            &|p| drop(p, "output_hc_fn.weight"),
        ),
        mutant(
            "a window layer's output gate removed",
            "blk.2.attn_gate.weight",
            &|p| drop(p, "blk.2.attn_gate.weight"),
        ),
        mutant(
            "a routed stack of another expert count",
            "blk.2.ffn_gate_exps.weight",
            &|p| {
                retype(
                    p,
                    "blk.2.ffn_gate_exps.weight",
                    GgmlType::Q4_K,
                    Some(vec![1536, 256, 513]),
                );
            },
        ),
        mutant(
            "the dense layer's width left at the source's",
            "ffn_up.weight",
            &|p| {
                retype(
                    p,
                    "blk.0.ffn_up.weight",
                    GgmlType::Q8_0,
                    Some(vec![3072, 13_824]),
                );
            },
        ),
        mutant(
            "the routed experts' latent key missing",
            "hy_v4_flash.expert_latent_length",
            &|p| {
                p.kvs
                    .retain(|(k, _)| k != "hy_v4_flash.expert_latent_length")
            },
        ),
        mutant(
            "the indexer's selection left at the source's",
            "attention.indexer.top_k 2048, not 1024, 256, 64 and 64",
            &|p| set(p, "attention.indexer.top_k", Value::U32(2048)),
        ),
        mutant(
            "the window left at the source's",
            "attention.sliding_window 512 and attention.indexer.top_k 64",
            &|p| set(p, "attention.sliding_window", Value::U32(512)),
        ),
        mutant("the sites left at the source's", "ple.layers", &|p| {
            set(
                p,
                "ple.layers",
                Value::Array(SITES.iter().map(|&l| Value::U32(l as u32)).collect()),
            );
        }),
        mutant(
            "a retyped routed down stack",
            "ffn_down_exps.weight is",
            &|p| retype(p, "blk.2.ffn_down_exps.weight", GgmlType::Q6_K, None),
        ),
        mutant(
            "the engram table in another type",
            "per_layer_token_embd.weight is",
            &|p| {
                retype(
                    p,
                    "per_layer_token_embd.weight",
                    GgmlType::Q8_0,
                    Some(vec![96, rows]),
                );
            },
        ),
    ];
    for r in results {
        if let Err(why) = r {
            bad.push(why);
        }
    }
    // A map of other kinds or stack types is refused before any byte is planned.
    for (layers, want) in [
        (
            vec![0, 1, 7, 11, 41],
            "source layer 7 (fixture 2) holds ffn_down_exps.weight as Some(Q6_K)",
        ),
        (
            vec![0, 1, 5, 11, 41],
            "source layer 5 (fixture 2) is a sparse-attention layer, the kind keeps it a window layer",
        ),
        (
            vec![0, 2, 6, 11, 41],
            "source layer 2 (fixture 1) is a window layer, the kind keeps it a sparse-attention layer",
        ),
        (
            vec![1, 1, 6, 11, 41],
            "source layer 1 (fixture 0) is routed, the kind keeps it dense",
        ),
        (
            vec![0, 1, 6, 12, 41],
            "source layer 12 (fixture 3) is no engram site, the kind keeps it an engram site",
        ),
        (
            vec![0, 1, 6, 11, 42],
            "source layer 42 is past the source's 42 layers",
        ),
        (
            vec![0, 1, 6, 11],
            "holds 4 layers, the fixture is built for the 5",
        ),
    ] {
        let other = FixtureSpec {
            layers: layers.clone(),
            ..hy4::spec()
        };
        match model::fixture::plan(&other, &source, None, &other.options()) {
            Err(e) if e.to_string().contains(want) => {}
            Err(e) => bad.push(format!(
                "the map {layers:?}: refused as {e}, wanted {want:?}"
            )),
            Ok(_) => bad.push(format!("the map {layers:?}: planned")),
        }
    }

    println!(
        "hy4 fixture layers=5 tensors={} shards={} bytes={} pins_differ={}",
        plan.target.tensors.len(),
        plan.target.shards.len(),
        plan.target.tensors.iter().map(|t| t.nbytes).sum::<u64>(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} pin(s) differ:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
