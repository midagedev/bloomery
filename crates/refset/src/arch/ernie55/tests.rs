use super::{ARCH, BUILD, FAMILIES, fixture};
use crate::RefError;
use crate::arch::{families, named, node_dumps};
use crate::family::{Build, Identity, Provenance};
use crate::fixture::with_root;
use std::path::{Path, PathBuf};

/// The vendor's folder's first shard, the real file the profile names: a set of it is stale for the fixture family.
const VENDOR_SHARD: &str = "/models/ERNIE-5.5-Flash-Preview/model-00001-of-00128.safetensors";

/// The keys of the cut's first shard, as `tools/fixture-cuts/ernie55.json` and the cutter write them.
const CUT_META: [(&str, &str); 9] = [
    ("format", "pt"),
    ("bloomery.fixture.card_budget", "8589934592"),
    ("bloomery.fixture.dense_ff", "1024"),
    ("bloomery.fixture.expert_ff", "512"),
    ("bloomery.fixture.layers", "0,1,7"),
    (
        "bloomery.fixture.manifest_sha256",
        "99baab0354879952671fd041070a5b52e7475b6533de8043d2a166cb3942fa6f",
    ),
    (
        "bloomery.fixture.rev",
        "ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe",
    ),
    ("bloomery.fixture.source", "baidu/ERNIE-5.5-Flash-Preview"),
    ("bloomery.fixture.version", "1"),
];

/// A file of the tree, read whole; the crate's directory is two levels under its root.
fn repo(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A fresh directory for one test; tests of one process run in parallel, so each names its own.
fn temp(what: &str) -> Result<PathBuf, RefError> {
    let dir = std::env::temp_dir().join(format!(
        "bloomery-refset-fx-np-ernie55-{what}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
    Ok(dir)
}

fn remove(dir: &Path) -> Result<(), RefError> {
    std::fs::remove_dir_all(dir).map_err(|e| RefError::missing(dir, e.to_string()))
}

/// A safetensors shard at `path`: one 1-byte tensor and `meta` as its string `__metadata__`.
fn write_shard(path: &Path, meta: &[(&str, &str)]) {
    let pairs: Vec<String> = meta
        .iter()
        .map(|(k, v)| {
            format!(
                "{}:{}",
                serde_json::to_string(k).expect("a &str serializes"),
                serde_json::to_string(v).expect("a &str serializes")
            )
        })
        .collect();
    let mut header = format!(
        r#"{{"__metadata__":{{{}}},"t":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#,
        pairs.join(",")
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.push(0);
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// A set at `dir` of the writer's header shape: the title, `# model`, the `# fixture` line (`None` leaves it out), `# arch`
/// (`None` leaves it out) and `# build`, the writer's `# kind` column line, one tensor row (`result_norm`), and the trailer when
/// `complete`. The manifest is all a family's check reads.
fn write_set(
    dir: &Path,
    model: &str,
    stated: Option<&str>,
    build: &str,
    arch: Option<&str>,
    complete: bool,
) {
    let mut lines = vec![
        "# ernie55_ref -- layer-streaming CPU f32 reference (tools/ref/ernie55_ref.py), raw f32 little-endian".to_string(),
        format!("# model\t{model}"),
    ];
    lines.extend(stated.map(|s| format!("# fixture\t{s}")));
    lines.extend(arch.map(|a| format!("# arch\t{a}")));
    lines.extend([
        format!("# build\t{build}"),
        "# numpy\t2.5.4".to_string(),
        "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1"
            .to_string(),
        "tensor\tresult_norm\t0\tf32\t4096\t1\t1\t1\t16384\t0\tRMS_NORM\t1\t0\t-\t-".to_string(),
    ]);
    if complete {
        lines.push("# complete\t1\t0".to_string());
    }
    let path = dir.join("MANIFEST.tsv");
    std::fs::write(&path, format!("{}\n", lines.join("\n")))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// A fixture root holding the cut (`<root>/<its directory>/`: the first shard of [`CUT_META`], a second shard and the index) and
/// a directory for the set each check rewrites; with the first shard and its `# fixture` line.
struct Cut {
    root: PathBuf,
    set: PathBuf,
    first: String,
    line: String,
}

impl Cut {
    fn new(what: &str) -> Result<Self, RefError> {
        let root = temp(what)?;
        let dir = root.join(
            crate::fixture::dir_of(ARCH).expect("the fixture table gives ernie5_5 its directory"),
        );
        std::fs::create_dir_all(&dir).map_err(|e| RefError::missing(&dir, e.to_string()))?;
        write_shard(&dir.join("model-00001-of-00002.safetensors"), &CUT_META);
        write_shard(&dir.join("model-00002-of-00002.safetensors"), &[]);
        std::fs::write(dir.join("model.safetensors.index.json"), b"{}")
            .map_err(|e| RefError::missing(&dir, e.to_string()))?;
        let first = with_root(&root, || crate::fixture::first_shard(ARCH))?;
        let line = crate::fixture::line(Path::new(&first))?;
        let set = temp(&format!("{what}-set"))?;
        Ok(Self {
            root,
            set,
            first,
            line,
        })
    }

    /// The family's check of a set at the root's fixture that states `model`, the `# fixture` text `stated` (`None` leaves the
    /// line out), `build`, the `# arch` `arch` (`None` leaves it out) and the trailer when `complete`.
    fn check(
        &self,
        model: &str,
        stated: Option<&str>,
        build: &str,
        arch: Option<&str>,
        complete: bool,
    ) -> Result<Provenance, RefError> {
        write_set(&self.set, model, stated, build, arch, complete);
        with_root(&self.root, || fixture::NP.check_set(&self.set))
    }

    /// The text after the tab of the run file's `# fixture` line.
    fn body(&self) -> &str {
        self.line
            .strip_prefix(crate::fixture::LINE_PREFIX)
            .expect("fixture::line opens with its prefix")
    }

    fn remove(self) -> Result<(), RefError> {
        remove(&self.root)?;
        remove(&self.set)
    }
}

/// The pin is the writer in the tree: an edit of `tools/ref/ernie55_ref.py` without new sets and a new pin is red.
#[test]
fn the_writer_pin_is_the_source_in_the_tree() {
    let source = repo("tools/ref/ernie55_ref.py");
    let md5 = crate::md5::hex_of(source.as_bytes());
    assert_eq!(
        BUILD,
        format!("ernie55_ref-{}", &md5[..12]),
        "tools/ref/ernie55_ref.py changed: dump the sets again (BLOOMERY_TIER=fixture just dump-ref-ernie55) and pin the new md5"
    );
}

/// The fixture family stands without a real twin, as `arch::tests` lists it: the architecture's one family, of the fixture identity
/// at the architecture and the writer's build, its two sets behind `fx_ref_ernie55_`, written by its recipe and read by the
/// end-to-end gate.
#[test]
fn the_fixture_family_stands_without_a_real_twin() {
    let fx = &fixture::NP;
    assert_eq!(FAMILIES.len(), 1, "no real family yet");
    assert!(std::ptr::eq(FAMILIES[0], fx));
    assert!(named("fx-np-ernie55").is_some_and(|f| std::ptr::eq(f, fx)));
    assert_eq!(
        families(ARCH).iter().map(|f| f.name).collect::<Vec<_>>(),
        ["fx-np-ernie55"]
    );
    assert!(node_dumps(ARCH).is_none(), "a real node-dump family");
    assert_eq!(fx.identity, Identity::FixtureManifest);
    assert_eq!(fx.arch, Some(ARCH));
    assert_eq!(fx.build, Some(Build::Is(BUILD)));
    assert!(fx.resolve.is_none(), "one fixture file, no per-file suffix");
    assert!(
        fx.draft_runs.is_none(),
        "the model ships no draft file here"
    );
    assert_eq!(
        fx.consumers,
        &["gate-gpu-ernie55-e2e"],
        "the end-to-end gate reads it"
    );
    assert_eq!(
        fx.recipe, "BLOOMERY_TIER=fixture just dump-ref-ernie55",
        "the fixture tier's recipe"
    );
    assert_eq!(
        fx.sets,
        ["fx_ref_ernie55_short", "fx_ref_ernie55_long"],
        "the two prompts' sets, each the writer's set behind fx_ref_ernie55_"
    );
}

/// A set of the cut passes and states its first shard and the writer's build: `# model` is the first shard, `# fixture` the line
/// of the run file, the build and the architecture the family's, the trailer there.
#[test]
fn the_family_takes_a_set_of_the_cut() -> Result<(), RefError> {
    let cut = Cut::new("own")?;
    let p = cut.check(&cut.first, Some(cut.body()), BUILD, Some(ARCH), true)?;
    assert_eq!(p.dumped_from, cut.first, "the first shard it states");
    assert_eq!(p.build.as_deref(), Some(BUILD));
    assert_eq!(p.draft, None);
    cut.remove()
}

/// A set that states the folder (a path no generation tells apart), the vendor's first shard or the cut's second shard is stale,
/// the refusal naming the set, the path it states and the run file.
#[test]
fn the_family_refuses_a_set_of_another_file() -> Result<(), RefError> {
    let cut = Cut::new("file")?;
    let set = cut.set.display().to_string();
    let folder = format!("{}/ernie55", cut.root.display());
    let second = format!("{}/model-00002-of-00002.safetensors", folder);
    for other in [&folder, VENDOR_SHARD, second.as_str()] {
        match cut.check(other, Some(cut.body()), BUILD, Some(ARCH), true) {
            Err(e @ RefError::Stale { .. }) => {
                let e = e.to_string();
                assert!(
                    e.contains(&set)
                        && e.contains(&format!("dumped from {other}"))
                        && e.contains(&format!("the tree runs {}", cut.first)),
                    "{other}: {e}"
                );
            }
            r => panic!("a set of {other}: {r:?}"),
        }
    }
    cut.remove()
}

/// A set of the run file's name that states no `# fixture` line, or another generation's, is stale on that line: the refusal names
/// both lines.
#[test]
fn the_family_refuses_a_set_of_another_generation() -> Result<(), RefError> {
    let cut = Cut::new("generation")?;
    let theirs = format!(
        "{}bloomery.fixture.expert_ff=256",
        crate::fixture::LINE_PREFIX
    );
    for (stated, why) in [
        (None, "no line"),
        (Some(theirs.as_str()), "another generation"),
    ] {
        match cut.check(&cut.first, stated, BUILD, Some(ARCH), true) {
            Err(e @ RefError::Stale { .. }) => {
                let e = e.to_string();
                let said = stated.unwrap_or("an unstated fixture");
                assert!(e.contains(said) && e.contains(&cut.line), "{why}: {e}");
            }
            r => panic!("{why}: {r:?}"),
        }
    }
    cut.remove()
}

/// A set whose build names another writer revision (the hand-numbered build the writer once wrote) is foreign by its build field.
#[test]
fn the_family_refuses_a_set_of_another_build() -> Result<(), RefError> {
    let cut = Cut::new("build")?;
    for build in ["ernie55_ref-000000000000", "ernie55_ref.py-1"] {
        match cut.check(&cut.first, Some(cut.body()), build, Some(ARCH), true) {
            Err(RefError::Foreign {
                field: "build",
                family,
                got,
                want,
                ..
            }) if family == fixture::NP.name && got == build && want == BUILD => {}
            r => panic!("a set of build {build}: {r:?}"),
        }
    }
    cut.remove()
}

/// A set with no `# arch` line, or one of another architecture, is foreign by its arch field.
#[test]
fn the_family_refuses_a_set_of_another_arch() -> Result<(), RefError> {
    let cut = Cut::new("arch")?;
    for (arch, got) in [
        (None, "(no # arch line)"),
        (Some("minimax-m3"), "minimax-m3"),
    ] {
        match cut.check(&cut.first, Some(cut.body()), BUILD, arch, true) {
            Err(RefError::Foreign {
                field: "arch",
                family,
                got: g,
                want,
                ..
            }) if family == fixture::NP.name && g == got && want == ARCH => {}
            r => panic!("a set of arch {got}: {r:?}"),
        }
    }
    cut.remove()
}

/// A set without its completion trailer is unfinished, by the set's name.
#[test]
fn the_family_refuses_a_set_without_its_trailer() -> Result<(), RefError> {
    let cut = Cut::new("trailer")?;
    let set = cut.set.display().to_string();
    match cut.check(&cut.first, Some(cut.body()), BUILD, Some(ARCH), false) {
        Err(RefError::Unfinished { set: s }) if s == set => {}
        r => panic!("a set without its trailer: {r:?}"),
    }
    cut.remove()
}

/// The row is the writer's and the recipe's: the justfile's recipe runs the dump script under this profile, the script stages the
/// prompts the family's sets are named after and reads each prompt's ids from the data directory, the profile names the
/// architecture and the folder whose first shard is [`VENDOR_SHARD`], and the writer states the architecture the family pins.
#[test]
fn the_row_is_what_the_writer_and_the_recipe_use() {
    let fx = &fixture::NP;
    let recipe = fx
        .recipe
        .strip_prefix("BLOOMERY_TIER=fixture just ")
        .expect("a just recipe");
    let justfile = repo("justfile");
    let body = justfile
        .lines()
        .skip_while(|l| !l.starts_with(&format!("{recipe}:")))
        .skip(1)
        .take_while(|l| l.starts_with(char::is_whitespace))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!body.is_empty(), "the justfile has no recipe {recipe}");
    assert!(
        body.contains("bash tools/ref/ernie55-dump.sh")
            && body.contains("BLOOMERY_MODEL=ernie5_5 "),
        "{recipe} does not run the dump script under the ernie5_5 profile: {body}"
    );

    let script = repo("tools/ref/ernie55-dump.sh");
    let staged: Vec<&str> = script
        .lines()
        .find_map(|l| l.strip_prefix("SETS=("))
        .expect("the dump script lists its prompts in a SETS=(...) line")
        .trim_end_matches(')')
        .split_whitespace()
        .collect();
    let sets: Vec<String> = staged
        .iter()
        .map(|n| format!("fx_ref_ernie55_{n}"))
        .collect();
    assert_eq!(
        sets, fx.sets,
        "each staged prompt installs as the family's set"
    );
    assert!(
        script.contains("IDS=$HERE/tools/ref/data/ernie55")
            && script.contains("--prompt \"$n=$IDS/$n.ids\""),
        "the script does not feed each staged prompt the ids file the data directory holds for it"
    );
    for n in staged {
        let ids = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/ref/data/ernie55")
            .join(format!("{n}.ids"));
        let text =
            std::fs::read_to_string(&ids).unwrap_or_else(|e| panic!("{}: {e}", ids.display()));
        assert!(
            text.split(',').all(|id| id.trim().parse::<u32>().is_ok()),
            "{}: not comma-separated ids",
            ids.display()
        );
    }

    let profile = repo("tools/ref/models/ernie5_5.sh");
    assert!(
        profile.contains("MODEL_NAME=ernie5_5")
            && profile.contains(&format!(
                "MODEL=${{BLOOMERY_REF_MODEL:-{}}}",
                Path::new(VENDOR_SHARD)
                    .parent()
                    .expect("a first shard has a directory")
                    .display()
            )),
        "the profile does not name the ernie5_5 architecture and the vendor's folder"
    );
    assert!(
        repo("tools/ref/ernie55_ref.py").contains(&format!("# arch\\t{ARCH}")),
        "the writer does not write the family's # arch line"
    );
}

/// The two sets in place hold each prompt's rows, derived from the cut's own `config.json` as the writer derives them: every
/// layer's `l_out`, each routed layer's `ffn_moe_topk` (an i32 row with its twin) and `ffn_moe_weights`, `result_norm` and the
/// logits before and after MultiMax, and each DSA layer's `indexer_top_k` in `long` alone; every row's file is as long as its
/// bytes; and the prompt a set states is the data directory's (its `# tokens` the ids file's, `# tokens_file_sha256` the file's
/// digest).
#[test]
#[ignore = "hw: needs the reference sets in $BLOOMERY_DATA and the cut under the fixture root"]
fn hw_the_sets_in_place_hold_each_prompts_rows() -> Result<(), RefError> {
    use crate::ik::RowKind;
    use sha2::{Digest, Sha256};

    let fx = &fixture::NP;
    let first = fx.runs()?;
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(&first)
                .parent()
                .expect("a first shard has a directory")
                .join("config.json"),
        )
        .map_err(|e| RefError::missing(&first, format!("config.json: {e}")))?,
    )
    .expect("the cut's config.json parses");
    let layers: Vec<usize> = (0..config["num_hidden_layers"]
        .as_u64()
        .expect("num_hidden_layers") as usize)
        .collect();
    let dense_lead = config["first_k_dense_replace"]
        .as_u64()
        .expect("first_k_dense_replace") as usize;
    let hidden = config["hidden_size"].as_u64().expect("hidden_size");
    let streams = config["hc_mult"].as_u64().expect("hc_mult");
    let top = config["num_experts_per_tok"]
        .as_u64()
        .expect("num_experts_per_tok");
    let kept = config["index_topk"].as_u64().expect("index_topk");
    let vocab = config["vocab_size"].as_u64().expect("vocab_size");
    let dsa: Vec<usize> = config["compress_ratios"]
        .as_array()
        .expect("compress_ratios")
        .iter()
        .take(layers.len())
        .zip(&layers)
        .filter(|(r, _)| r.as_i64() == Some(-2))
        .map(|(_, il)| *il)
        .collect();
    assert!(
        !dsa.is_empty(),
        "the cut holds no DSA layer: long's indexer row is unchecked"
    );

    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/data/ernie55");
    for (set, dsa_rows) in [(fixture::SHORT, false), (fixture::LONG, true)] {
        let path = fx.path(set);
        let man = crate::ik::RefManifest::open(&path, fx)?;
        let name = set
            .strip_prefix("fx_ref_ernie55_")
            .expect("the family's set name");
        let ids_file = data.join(format!("{name}.ids"));
        let ids =
            std::fs::read(&ids_file).unwrap_or_else(|e| panic!("{}: {e}", ids_file.display()));
        let want: Vec<u32> = std::fs::read_to_string(&ids_file)
            .unwrap_or_else(|e| panic!("{}: {e}", ids_file.display()))
            .split(',')
            .map(|id| id.trim().parse().expect("an id of the ids file"))
            .collect();
        assert_eq!(
            man.header.tokens.as_deref(),
            Some(want.as_slice()),
            "{set}: # tokens is not the ids file's"
        );
        let digest = format!("{:x}", Sha256::digest(&ids));
        assert_eq!(
            man.header.tokens_file_sha256.as_deref(),
            Some(digest.as_str()),
            "{set}: # tokens_file_sha256"
        );

        let mut rows = 0usize;
        for il in &layers {
            let row = man
                .tensor(&format!("l_out-{il}"), 0)
                .unwrap_or_else(|e| panic!("{set}: {e}"));
            row.expect(
                &format!("{set}: l_out-{il}"),
                "f32",
                [hidden, streams, 4, 1],
                "HC_POST",
            )
            .unwrap_or_else(|e| panic!("{set}: {e}"));
            rows += 1;
            if *il >= dense_lead {
                let topk = man
                    .tensor(&format!("ffn_moe_topk-{il}"), 0)
                    .unwrap_or_else(|e| panic!("{set}: {e}"));
                topk.expect(
                    &format!("{set}: ffn_moe_topk-{il}"),
                    "i32",
                    [top, 4, 1, 1],
                    "ARGSORT",
                )
                .unwrap_or_else(|e| panic!("{set}: {e}"));
                let twins = man
                    .ints
                    .iter()
                    .filter(|r| r.name == topk.name && r.of == RowKind::Tensor)
                    .collect::<Vec<_>>();
                assert_eq!(
                    twins.len(),
                    1,
                    "{set}: ffn_moe_topk-{il} has {} int twins",
                    twins.len()
                );
                man.tensor(&format!("ffn_moe_weights-{il}"), 0)
                    .unwrap_or_else(|e| panic!("{set}: {e}"))
                    .expect(
                        &format!("{set}: ffn_moe_weights-{il}"),
                        "f32",
                        [top, 4, 1, 1],
                        "ROUTE",
                    )
                    .unwrap_or_else(|e| panic!("{set}: {e}"));
                rows += 2;
            }
            if dsa.contains(il) {
                let has = man
                    .find(RowKind::Tensor, &format!("indexer_top_k-{il}"), 0)
                    .is_some();
                assert_eq!(
                    has, dsa_rows,
                    "{set}: indexer_top_k-{il} is written in long alone"
                );
                if has {
                    man.tensor(&format!("indexer_top_k-{il}"), 0)
                        .unwrap_or_else(|e| panic!("{set}: {e}"))
                        .expect(
                            &format!("{set}: indexer_top_k-{il}"),
                            "i32",
                            [kept, 1, 1, 1],
                            "TOP_K",
                        )
                        .unwrap_or_else(|e| panic!("{set}: {e}"));
                    rows += 1;
                }
            }
        }
        for (rowname, ne, op) in [
            ("result_norm", [hidden, 1, 1, 1], "RMS_NORM"),
            ("result_output_raw", [vocab, 1, 1, 1], "MUL_MAT"),
            ("result_output", [vocab, 1, 1, 1], "MULTIMAX"),
        ] {
            man.tensor(rowname, 0)
                .unwrap_or_else(|e| panic!("{set}: {e}"))
                .expect(&format!("{set}: {rowname}"), "f32", ne, op)
                .unwrap_or_else(|e| panic!("{set}: {e}"));
            rows += 1;
        }
        assert_eq!(
            man.complete,
            Some((rows as u64, 0)),
            "{set}: the trailer's tensor rows"
        );
        assert_eq!(
            man.tensors.len(),
            rows,
            "{set}: a row the writer's grammar does not name"
        );
        for row in &man.tensors {
            let file = path.join(row.file_name());
            let size = std::fs::metadata(&file)
                .unwrap_or_else(|e| panic!("{}: {e}", file.display()))
                .len();
            assert_eq!(size, row.bytes, "{}", file.display());
        }
        for row in &man.ints {
            let file = path.join(&row.file);
            let size = std::fs::metadata(&file)
                .unwrap_or_else(|e| panic!("{}: {e}", file.display()))
                .len();
            assert_eq!(size, row.bytes, "{}", file.display());
        }
        let greedy = std::fs::read_to_string(path.join("greedy.tsv"))
            .unwrap_or_else(|e| panic!("{}: {e}", path.join("greedy.tsv").display()));
        let lines: Vec<&str> = greedy
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert_eq!(lines.len(), 1, "{set}: one prompt, one --gen row");
        assert_eq!(
            lines[0].split('\t').count(),
            7,
            "{set}: the 7-column --gen row"
        );
        assert_eq!(
            lines[0]
                .split('\t')
                .nth(5)
                .expect("gen_ids")
                .split(',')
                .count(),
            8,
            "{set}: 8 greedy steps"
        );
        println!(
            "hw_refset: {} {set}: {rows} tensor rows, {} int rows — PASS",
            fx.name,
            man.ints.len()
        );
    }
    Ok(())
}
