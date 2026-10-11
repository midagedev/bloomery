//! The shipped MTP head list against held-out text, per language and per kind of text.
//!
//! The draft scores only the rows of the list, so a target token outside it ends that window's accepted
//! prefix. The list beats the full head only while its miss rate stays under a break-even that falls as the list
//! grows ([`break_even`]); this gate holds the miss rate of every held-out set under it. A set is one file of
//! token ids under `tests/data/head-holdout/`; the texts it came from were not an input of the build that picked
//! the list (`build-sources.txt` names every input, and this gate refuses a held-out item the build read).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The directory the model crate keeps its built-in row lists in, and the one it ships (`head_list.rs`'s
/// `SHIPPED_PATH`). The paths are whole literals so that `just affected` selects this gate for a change to them.
const LIST_DIR: &str = "../../crates/model/data";
const SHIPPED_LIST: &str = "../../crates/model/data/mtp-head-rows-qwen38-cjk-76000.txt";

/// The held-out sets and the build sources.
const HOLDOUT_DIR: &str = "../../crates/models/tests/data/head-holdout";

/// The held-out sets, one file `<name>.ids` each, and nothing else but [`BUILD_SOURCES`] in the directory.
const SETS: &[&str] = &[
    "zh-wiki",
    "zh-chat",
    "ja-wiki",
    "ja-chat",
    "ko-wiki",
    "ko-chat",
    "en-prose",
    "en-math",
    "code-rust",
    "code-python",
    "code-c",
];

/// The inputs of the build that picked the shipped list, one `<source> <first row> <last row>` a line.
const BUILD_SOURCES: &str = "build-sources.txt";

/// A window walks the head three times, each walk reading every row of the list once.
const WALKS: f64 = 3.0;

/// A row of the Q8_0 `output`: 2,560 code bytes and 160 scale bytes.
const ROW_BYTES: f64 = 2_720.0;

/// PIN(2026-10-11): the head's read rate on the A6000, the top of its measured 579-701 GB/s, which saves the fewest
/// ms a window and so gives the lowest break-even.
const HEAD_GB_PER_S: f64 = 701.0;

/// PIN(2026-10-11): the full head's window at P 512 on the A6000, ms.
const WINDOW_MS: f64 = 37.43;

/// PIN(2026-10-11): the windows of docs/cards/q38head2-ab.card at P 512 by draft positions kept, 0 to 3 (26 windows).
const KEPT: [f64; 4] = [0.0, 3.0, 2.0, 21.0];

/// The miss rate q at which a list of `rows` of a `vocab`-token head stops beating the full head: the accepted
/// positions a window, AL(q) = 1 + sum_j A_j (1 - q)^j with A_j the share of windows keeping j or more draft
/// positions (KEPT), fall by the fraction of the window the list saves, AL(q) / AL(0) = 1 - S / T, with S the ms the
/// unread rows save (WALKS x the rows not in the list x ROW_BYTES at HEAD_GB_PER_S) and T the window [derived].
fn break_even(rows: usize, vocab: usize) -> f64 {
    let saved_ms = WALKS * (vocab - rows) as f64 * ROW_BYTES / (HEAD_GB_PER_S * 1e9) * 1e3;
    let windows: f64 = KEPT.iter().sum();
    let al = |q: f64| {
        1.0 + (1..KEPT.len())
            .map(|j| KEPT[j..].iter().sum::<f64>() / windows * (1.0 - q).powi(j as i32))
            .sum::<f64>()
    };
    let target = al(0.0) * (1.0 - saved_ms / WINDOW_MS);
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if al(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Fewer ids than this is a set cut down, not a measurement.
const MIN_IDS: usize = 20_000;

/// The line a list or a held-out file opens with names the tokenizer its ids index.
const SHA_FIELD: &str = "vocab_sha256=";

/// A span of rows of a text source: one article, one page of a dataset, one file.
#[derive(Debug, PartialEq, Eq)]
struct Item {
    source: String,
    first: u64,
    last: u64,
}

impl Item {
    /// `<source> <first> <last>`, the source holding no space.
    fn parse(path: &Path, text: &str) -> Item {
        let mut it = text.split_whitespace();
        let (Some(source), Some(first), Some(last), None) =
            (it.next(), it.next(), it.next(), it.next())
        else {
            panic!(
                "{}: `{text}` is not `<source> <first row> <last row>`",
                path.display()
            );
        };
        let row = |s: &str| {
            s.parse::<u64>()
                .unwrap_or_else(|e| panic!("{}: `{text}`: {e}", path.display()))
        };
        Item {
            source: source.to_owned(),
            first: row(first),
            last: row(last),
        }
    }

    fn overlaps(&self, other: &Item) -> bool {
        self.source == other.source && self.first <= other.last && other.first <= self.last
    }
}

/// One held-out file: the text it came from and its ids.
struct Holdout {
    name: &'static str,
    items: Vec<Item>,
    ids: Vec<u32>,
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn data_dir() -> PathBuf {
    crate_dir().join(HOLDOUT_DIR)
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The value of the `key=value` word of a header line that starts with `key`.
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    header.split_whitespace().find_map(|w| w.strip_prefix(key))
}

fn sha_of(path: &Path, header: &str) -> String {
    field(header, SHA_FIELD)
        .unwrap_or_else(|| panic!("{}: the first line names no {SHA_FIELD}", path.display()))
        .to_owned()
}

fn parse_ids(path: &Path, lines: &[&str]) -> Vec<u32> {
    lines
        .iter()
        .map(|l| {
            l.parse::<u32>()
                .unwrap_or_else(|e| panic!("{}: `{l}`: {e}", path.display()))
        })
        .collect()
}

/// The list the model crate ships.
struct List {
    /// The tokenizer digest its header names.
    sha: String,
    /// The vocabulary its header names.
    vocab: usize,
    ids: HashSet<u32>,
}

/// [`SHIPPED_LIST`], refused by name when [`LIST_DIR`] holds another list: the model crate ships one, so a second
/// file is a list this gate does not read.
fn shipped_list() -> List {
    let dir = crate_dir().join(LIST_DIR);
    let mut found: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            e.expect("a directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    found.sort();
    let want = Path::new(SHIPPED_LIST)
        .file_name()
        .expect("a file name")
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        found,
        [want],
        "{}: the one list this gate reads is {SHIPPED_LIST}",
        dir.display()
    );
    let path = crate_dir().join(SHIPPED_LIST);
    let text = read(&path);
    let lines: Vec<&str> = text.lines().collect();
    let first = lines
        .first()
        .unwrap_or_else(|| panic!("{}: empty", path.display()));
    let ids = parse_ids(&path, &lines[1..]);
    let set: HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(set.len(), ids.len(), "{}: a repeated id", path.display());
    let vocab = field(first, "vocab=")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("{}: the first line names no vocab=<n>", path.display()));
    List {
        sha: sha_of(&path, first),
        vocab,
        ids: set,
    }
}

/// Every set of [`SETS`], refused by name when the directory holds another file, a set is missing, its ids are
/// fewer than [`MIN_IDS`] or fewer than its header says, or it indexes another tokenizer than `sha`.
fn holdouts(sha: &str) -> Vec<Holdout> {
    let dir = data_dir();
    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            e.expect("a directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    files.sort();
    let mut want: Vec<String> = SETS.iter().map(|s| format!("{s}.ids")).collect();
    want.push(BUILD_SOURCES.to_owned());
    want.sort();
    assert_eq!(
        files,
        want,
        "{}: the held-out files are not the sets this gate names",
        dir.display()
    );
    SETS.iter()
        .map(|&name| {
            let path = dir.join(format!("{name}.ids"));
            let text = read(&path);
            let lines: Vec<&str> = text.lines().collect();
            let comments = lines.iter().take_while(|l| l.starts_with('#')).count();
            let header = lines[0];
            assert!(
                header.starts_with("# bloomery mtp-head-holdout 1 ")
                    && header.contains(&format!(" set={name} ")),
                "{}: first line `{header}` is not this set's header",
                path.display()
            );
            assert_eq!(
                sha_of(&path, header),
                sha,
                "{}: another tokenizer's ids",
                path.display()
            );
            let ids = parse_ids(&path, &lines[comments..]);
            let named: usize = field(header, "ids=")
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("{}: the first line names no ids=<n>", path.display()));
            assert_eq!(
                ids.len(),
                named,
                "{}: ids= and the ids it holds",
                path.display()
            );
            assert!(
                ids.len() >= MIN_IDS,
                "{}: {} ids is a set cut down",
                path.display(),
                ids.len()
            );
            let items: Vec<Item> = lines[..comments]
                .iter()
                .filter_map(|l| l.strip_prefix("# item "))
                .map(|t| Item::parse(&path, t))
                .collect();
            assert!(
                !items.is_empty(),
                "{}: names no source item",
                path.display()
            );
            Holdout { name, items, ids }
        })
        .collect()
}

/// The break-even the gate computes is the one derived for the shipped size and it falls as rows are added.
/// PIN(2026-10-11): 3.91 % for 76,000 rows of Qwen3.8's 248,320-token head: the 172,320 unread rows save
/// 3 x 172,320 x 2,720 B / 701 GB/s = 2.01 ms of the 37.43 ms window, and AL(q) / AL(0) = 1 - 2.01 / 37.43 gives
/// q = 0.0391 for the KEPT windows [derived].
#[test]
fn break_even_is_the_derived_one() {
    let at_76000 = break_even(76_000, 248_320);
    assert!((at_76000 - 0.0391).abs() < 5e-4, "{at_76000}");
    assert!(break_even(98_304, 248_320) < at_76000);
}

/// Every set's miss rate against the list is under the list's break-even: each language, and each kind of text in
/// it, still gains from the reduced head.
#[test]
fn every_set_misses_less_than_the_break_even() {
    let list = shipped_list();
    let q_star = break_even(list.ids.len(), list.vocab);
    println!("rows {}  break-even {q_star:.4}", list.ids.len());
    let mut over = Vec::new();
    for h in holdouts(&list.sha) {
        let miss = h.ids.iter().filter(|id| !list.ids.contains(id)).count();
        let q = miss as f64 / h.ids.len() as f64;
        println!("{:<12} q = {q:.4}  ({miss} of {} ids)", h.name, h.ids.len());
        if q >= q_star {
            over.push(format!(
                "{}: q = {q:.4} ({miss} of {} ids)",
                h.name,
                h.ids.len()
            ));
        }
    }
    assert!(
        over.is_empty(),
        "the list misses at least {q_star:.4} of: {}",
        over.join("; ")
    );
}

/// No text a held-out set came from was an input of the build that picked the list.
#[test]
fn held_out_text_is_not_a_build_input() {
    let list = shipped_list();
    let path = data_dir().join(BUILD_SOURCES);
    let text = read(&path);
    let build: Vec<Item> = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| Item::parse(&path, l))
        .collect();
    assert!(
        !build.is_empty(),
        "{}: names no build input",
        path.display()
    );
    let mut shared = Vec::new();
    for h in holdouts(&list.sha) {
        for item in &h.items {
            if let Some(b) = build.iter().find(|b| item.overlaps(b)) {
                shared.push(format!("{}: {item:?} is in the build input {b:?}", h.name));
            }
        }
    }
    assert!(shared.is_empty(), "{}", shared.join("; "));
}
