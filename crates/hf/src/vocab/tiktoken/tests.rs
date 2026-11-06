//! The tiktoken reader against a toy folder (`tests/fixtures/tiktoken-toy`) and the real Kimi K3
//! folder. The toy's ranks are the 256 bytes in byte order, then `he`, `ll`, `llo`, `hello`, `aa`,
//! `aaa`, two spaces, three spaces, `é`, the first two bytes of `한` and `한`; its special tokens
//! are Kimi's names at the ids Kimi's tokenizer config gives them, shifted by its 267 base tokens.
//! Its goldens come from the converter's own algorithm (llama.cpp `conversion/qwen.py` `bpe` and
//! `kimi_linear.py` `set_vocab`), run on the same files outside this crate.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

/// Mergeable ranks of the toy; its first special id.
const BASE: usize = 267;
/// The toy's `vocab_size`: its special ids and two fillers after them.
const SIZE: usize = 525;

fn toy() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiktoken-toy")
}

/// A copy of the toy folder in a directory of its own, each file's text passed through `edit`
/// with the file's name.
fn folder(edit: impl Fn(&str, String) -> String) -> PathBuf {
    static COPIES: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hf-tiktoken-{}-{}",
        std::process::id(),
        COPIES.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(toy()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let text = edit(&name, fs::read_to_string(&path).unwrap());
        fs::write(dir.join(&name), text).unwrap();
    }
    dir
}

/// The keys of the folder `dir` as the shared entry reads it (no `tokenizer.json`, so tiktoken's).
fn read(dir: &Path) -> Result<Keys, Error> {
    let config = crate::read_json(dir, "config.json", Error::Io, Error::Json)?
        .expect("the folder has a config.json");
    crate::vocab::keys(dir, &VocabSpec::parse(&config)?)
}

/// One edit of a toy file: the file, a text that occurs once in it, and what replaces it.
type Edit = (&'static str, &'static str, &'static str);

/// A stop token: its id and the token it spells.
type Stop = (u32, &'static str);

/// The toy folder with each edit made.
fn changed(edits: &[Edit]) -> PathBuf {
    let dir = folder(|_, text| text);
    for &(file, from, to) in edits {
        let text = fs::read_to_string(dir.join(file)).unwrap();
        assert_eq!(text.matches(from).count(), 1, "{file} holds {from:?} once");
        fs::write(dir.join(file), text.replace(from, to)).unwrap();
    }
    dir
}

fn get<'a>(keys: &'a Keys, key: &str) -> Option<&'a Value> {
    keys.keys.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// The ids of `ids` beside the token each spells, as a stop set reads.
fn spelled(keys: &Keys, ids: &[u32]) -> Vec<(u32, String)> {
    let Some(Value::Array(tokens)) = get(keys, "tokenizer.ggml.tokens") else {
        panic!("no tokens");
    };
    ids.iter()
        .map(|&id| match &tokens[usize::try_from(id).unwrap()] {
            Value::String(token) => (id, token.clone()),
            other => panic!("token {id} is {other:?}"),
        })
        .collect()
}

/// `(id, token)` pairs written as literals.
fn pairs(of: &[(u32, &str)]) -> Vec<(u32, String)> {
    of.iter().map(|&(id, t)| (id, t.to_string())).collect()
}

/// GPT-2's byte-level alphabet written out here, apart from the tokenizer crate's table the reader
/// spells by.
fn alphabet() -> [char; 256] {
    let kept: Vec<u32> = (0x21..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut next = 256;
    let mut glyph = ['\0'; 256];
    for byte in 0..256u32 {
        let cpt = if kept.contains(&byte) {
            byte
        } else {
            next += 1;
            next - 1
        };
        glyph[byte as usize] = char::from_u32(cpt).unwrap();
    }
    glyph
}

/// The toy's token spellings by id, from the alphabet above.
fn toy_tokens() -> Vec<String> {
    let glyph = alphabet();
    let spell = |bytes: &[u8]| -> String { bytes.iter().map(|&b| glyph[usize::from(b)]).collect() };
    let mut tokens: Vec<String> = (0..=255u8).map(|b| spell(&[b])).collect();
    for token in [
        &b"he"[..],
        b"ll",
        b"llo",
        b"hello",
        b"aa",
        b"aaa",
        b"  ",
        b"   ",
        "é".as_bytes(),
        &[0xed, 0x95],
        "한".as_bytes(),
    ] {
        tokens.push(spell(token));
    }
    assert_eq!(tokens.len(), BASE);
    let named = [
        (0, "[BOS]"),
        (1, "[EOS]"),
        (2, "<|end_of_msg|>"),
        (3, "<|open|>"),
        (9, "[EOT]"),
        (254, "[UNK]"),
        (255, "[PAD]"),
    ];
    for i in 0..RESERVED {
        tokens.push(named.iter().find(|&&(at, _)| at == i).map_or_else(
            || format!("<|reserved_token_{}|>", BASE + i),
            |&(_, n)| n.to_string(),
        ));
    }
    tokens.extend((BASE + RESERVED..SIZE).map(|id| format!("[PAD{id}]")));
    tokens
}

fn strings(v: &[String]) -> Value {
    Value::Array(v.iter().cloned().map(Value::String).collect())
}

/// Serving's tokenizer built from `keys`, with the checks the folder map makes of it (`model-hf`'s
/// `served_as_hf`): its split is the added ids, its stop set HF's, its control tokens the
/// header's, no automatic BOS or EOS, and the special ids the keys name.
fn served(keys: &Keys) -> tokenizer::Tokenizer {
    let served = tokenizer::Tokenizer::from_keys(&keys.keys).expect("serving reads the keys");
    let mut split = served.special_tokens().to_vec();
    split.sort_unstable();
    assert_eq!(split, keys.added, "serving splits text on the special ids");
    assert_eq!(
        served.eog(),
        keys.stop,
        "serving stops where HF's generate does"
    );
    let Some(Value::Array(types)) = get(keys, "tokenizer.ggml.token_type") else {
        panic!("no token types");
    };
    assert_eq!(types.len(), served.n_vocab());
    for (id, ty) in types.iter().enumerate() {
        let id = u32::try_from(id).unwrap();
        let control = served
            .attr(id)
            .is_some_and(|a| a & (tokenizer::attr::CONTROL | tokenizer::attr::UNKNOWN) != 0);
        assert_eq!(control, *ty == Value::I32(3), "control at token {id}");
    }
    assert!(!served.add_bos() && !served.add_eos());
    let specials = served.specials();
    for (key, actual) in [
        ("tokenizer.ggml.bos_token_id", specials.bos),
        ("tokenizer.ggml.eos_token_id", specials.eos),
        ("tokenizer.ggml.padding_token_id", specials.pad),
        ("tokenizer.ggml.unknown_token_id", specials.unk),
        ("tokenizer.ggml.seperator_token_id", specials.sep),
    ] {
        let want = get(keys, key).and_then(Value::as_u64);
        assert_eq!(actual.map(u64::from), want, "{key}");
    }
    served
}

/// RFC 4648's vectors, and the spellings that decode in Python's lenient `b64decode` but are
/// not the base64 of any token: whitespace, characters outside the alphabet, misplaced padding,
/// a length off a multiple of four, and bits the padding drops that are not zero.
#[test]
fn base64_is_rfc_4648_and_nothing_else() {
    for (text, bytes) in [
        ("Zg==", &b"f"[..]),
        ("Zm8=", b"fo"),
        ("Zm9v", b"foo"),
        ("Zm9vYg==", b"foob"),
        ("Zm9vYmE=", b"fooba"),
        ("Zm9vYmFy", b"foobar"),
        ("/+8=", &[0xff, 0xef]),
    ] {
        assert_eq!(base64_decode(text).as_deref(), Some(bytes), "{text}");
    }
    for text in [
        "", "Zg", "Zg=", "Zg===", "Z===", "====", "=Zg=", "Zg==Zg==", "Zm 9v", "Zm9v\n", "Zm9v\r",
        "Zm9*", "Zh==", "Zm9=", "Zm9vYmF=", "Zm-v", "Zm_v",
    ] {
        assert_eq!(base64_decode(text), None, "{text:?}");
    }
}

/// The file's lines are its ranks, so a repeat or a rank past the last line is named; a line that
/// is not `<base64> <rank>`, a blank one inside the file, a rank spelled any way but plainly, and
/// a missing file are each named. The last line's newline is optional.
#[test]
fn ranks_are_one_per_line_with_no_gap_or_repeat() {
    let ok = |text: &str| ranked(text).map_err(|e| e.to_string());
    assert_eq!(
        ok("QQ== 0\nQg== 1\n").unwrap(),
        [b"A".to_vec(), b"B".to_vec()]
    );
    assert_eq!(
        ok("Qg== 1\nQQ== 0").unwrap(),
        [b"A".to_vec(), b"B".to_vec()]
    );
    for (text, named) in [
        ("QQ== 0\nQg== 0\n", "lines 1 and 2 both have rank 0"),
        ("QQ== 0\nQg== 2\n", "line 2: rank 2 leaves a gap in 0..2"),
        ("QQ== 1\nQg== 2\n", "line 2: rank 2 leaves a gap in 0..2"),
        (
            "QQ== 0\n\nQg== 1\n",
            "line 2 is not `<base64 token> <rank>`",
        ),
        ("QQ== 0\nQg==1\n", "line 2 is not `<base64 token> <rank>`"),
        (
            "QQ== 0\nQg= 1\n",
            "line 2: \"Qg=\" is not the base64 of a token",
        ),
        ("QQ== 0\r\nQg== 1\r\n", "is not a rank"),
        ("QQ== 0\nQg== 01\n", "line 2: \"01\" is not a rank"),
        ("QQ== 0\nQg== +1\n", "line 2: \"+1\" is not a rank"),
        ("QQ== 0\nQg== -1\n", "line 2: \"-1\" is not a rank"),
        ("QQ== 0\nQg== 1 \n", "line 2: \"1 \" is not a rank"),
        ("QQ== 0\nQg== 99999999999999999999999\n", "is not a rank"),
        ("", "has no lines"),
        ("\n", "line 1 is not"),
    ] {
        let err = ok(text).expect_err(text);
        assert!(err.contains(named), "{text:?}: {err}");
    }
}

/// A token's two parts are what BPE leaves when it merges only pairs ranked below the token, the
/// leftmost of two equal pairs first; a token that does not fall into two is named.
#[test]
fn a_token_s_merge_is_its_two_parts_below_its_own_rank() {
    let spelling = Spelling::new();
    let bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    // The merges of the 256 bytes followed by `extra` in rank order, as spelled strings.
    let run = |extra: &[&[u8]]| {
        let ranked: Vec<Vec<u8>> = bytes
            .iter()
            .cloned()
            .chain(extra.iter().map(|t| t.to_vec()))
            .collect();
        let rank_of: HashMap<&[u8], usize> = ranked
            .iter()
            .enumerate()
            .map(|(rank, t)| (t.as_slice(), rank))
            .collect();
        merges(&ranked, &rank_of, &spelling)
            .map(|m| {
                m.into_iter()
                    .map(|v| match v {
                        Value::String(s) => s,
                        v => panic!("{v:?}"),
                    })
                    .collect::<Vec<_>>()
            })
            .map_err(|e| e.to_string())
    };
    // The lowest rank first: `he` (256) before `ll` (257) and `llo` (258) before `hello` (259).
    let hello = run(&[b"he", b"ll", b"llo", b"hello"]).unwrap();
    assert_eq!(hello, ["h e", "l l", "ll o", "he llo"]);
    // Two equal pairs: the leftmost merges, `aa a`, not `a aa`.
    let aaa = run(&[b"aa", b"aaa"]).unwrap();
    assert_eq!(aaa, ["a a", "aa a"]);
    // A pair ranked at the token's own rank or above is not merged: `ab` (256) is below `abab`
    // (258) and `ba` (257) is too, and the leftmost lowest pair is `ab`.
    let abab = run(&[b"ab", b"ba", b"abab"]).unwrap();
    assert_eq!(abab, ["a b", "b a", "ab ab"]);
    // Three bytes no pair of which is a token: three parts, refused.
    let err = run(&[b"xyz"]).unwrap_err();
    assert!(
        err.contains("falls into 3 tokens of lower rank, not two"),
        "{err}"
    );
}

/// The toy folder is the header llama.cpp's converter writes for it: model, pre-tokenizer, every
/// token and type, the merges, the special ids from `config.json` (the `[EOS]` of the tokenizer
/// config is not the EOS; no unknown id), the stop set and the split ids.
#[test]
fn the_toy_folder_becomes_the_converters_header() {
    let keys = read(&toy()).unwrap();
    let names: Vec<&str> = keys.keys.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        names,
        [
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.eog_token_ids",
        ]
    );
    assert_eq!(
        get(&keys, "tokenizer.ggml.model"),
        Some(&Value::String("gpt2".into()))
    );
    assert_eq!(
        get(&keys, "tokenizer.ggml.pre"),
        Some(&Value::String("kimi-k2".into()))
    );
    let tokens = toy_tokens();
    assert_eq!(tokens.len(), SIZE);
    assert_eq!(get(&keys, "tokenizer.ggml.tokens"), Some(&strings(&tokens)));
    let types: Vec<Value> = (0..SIZE)
        .map(|id| {
            Value::I32(if id < BASE {
                1
            } else if id < BASE + RESERVED {
                3
            } else {
                5
            })
        })
        .collect();
    assert_eq!(
        get(&keys, "tokenizer.ggml.token_type"),
        Some(&Value::Array(types))
    );
    let merges = [
        "h e",
        "l l",
        "ll o",
        "he llo",
        "a a",
        "aa a",
        "\u{120} \u{120}",
        "\u{120}\u{120} \u{120}",
        "\u{c3} \u{a9}",
        "\u{ed} \u{137}",
        "\u{ed}\u{137} \u{13e}",
    ];
    assert_eq!(
        get(&keys, "tokenizer.ggml.merges"),
        Some(&strings(&merges.map(String::from)))
    );
    for (key, id) in [("bos", 267), ("eos", 269), ("padding", 522)] {
        assert_eq!(
            get(&keys, &format!("tokenizer.ggml.{key}_token_id")),
            Some(&Value::U32(id)),
            "{key}"
        );
    }
    assert_eq!(
        get(&keys, "tokenizer.ggml.eog_token_ids"),
        Some(&Value::Array(vec![Value::U32(268), Value::U32(269)]))
    );
    assert_eq!(
        spelled(&keys, &keys.stop),
        pairs(&[(268, "[EOS]"), (269, "<|end_of_msg|>")])
    );
    assert_eq!(keys.added, (267..523).collect::<Vec<u32>>());
}

/// Serving's tokenizer reads the toy's keys: the merges rank as the file's ranks do (`hello` is one
/// token, ` aaa` a space and `aaa`), the special tokens split a text only when parsed, and every
/// check the folder map makes of serving's tokenizer holds.
#[test]
fn serving_encodes_the_toy_as_its_ranks_say() {
    let keys = read(&toy()).unwrap();
    let served = served(&keys);
    assert_eq!(served.encode("hello aaa", false, false), [259, 32, 261]);
    assert_eq!(served.encode("aaa", false, false), [261]);
    assert_eq!(served.encode("a   b", false, false), [97, 262, 32, 98]);
    assert_eq!(served.encode("<|end_of_msg|>", false, true), [269]);
    assert_eq!(served.encode("<|reserved_token_300|>", false, true), [300]);
    assert_ne!(served.encode("<|reserved_token_300|>", false, false), [300]);
    assert_eq!(served.pre(), "kimi-k2");
}

/// A folder the reader cannot read as the converter does is refused by name: each edit of the toy
/// below is red with the words that name what it broke.
#[test]
fn a_folder_that_is_not_the_converters_is_refused_by_name() {
    let tk = "tokenizer_config.json";
    let tm = "tiktoken.model";
    let kimi = "tokenization_kimi.py";
    let unpinned = "tokenization_kimi.py does not carry the kimi-k2 pre-tokenizer's source";
    // (what broke, its edits, the words the refusal carries)
    let cases: Vec<(&str, Vec<Edit>, &str)> = vec![
        (
            "a rank repeated",
            vec![(tm, "aGU= 256\n", "aGU= 255\n")],
            "both have rank 255",
        ),
        (
            "a rank gap",
            vec![(tm, "aGVsbG8= 259", "aGVsbG8= 300")],
            "line 260: rank 300 leaves a gap in 0..267",
        ),
        (
            "a malformed base64 line",
            vec![(tm, "aGU= 256", "aG*= 256")],
            "line 257: \"aG*=\" is not the base64 of a token",
        ),
        (
            "a base64 line with stray bits",
            vec![(tm, "aGU= 256", "aGV= 256")],
            "line 257: \"aGV=\" is not the base64 of a token",
        ),
        (
            "a blank line inside",
            vec![(tm, "aGU= 256\n", "\naGU= 256\n")],
            "line 257 is not `<base64 token> <rank>`",
        ),
        (
            "one token's bytes twice",
            vec![(tm, "bGw= 257", "aGU= 257")],
            "ranks 256 and 257 are the same token bytes",
        ),
        (
            "a byte with no token",
            vec![(tm, "AA== 0", "AAA= 0")],
            "no token for byte 0x00",
        ),
        (
            "a token of no two lower tokens",
            vec![(tm, "aGVsbG8= 259", "eHl6 259")],
            "token \"xyz\" of rank 259 falls into 3 tokens of lower rank, not two",
        ),
        (
            "a vocabulary smaller than its tokens",
            vec![
                ("config.json", "\"vocab_size\": 525", "\"vocab_size\": 522"),
                (
                    "config.json",
                    "\"pad_token_id\": 522",
                    "\"pad_token_id\": 521",
                ),
            ],
            "vocab_size 522 is below tiktoken.model's 267 ranks plus 256 special tokens",
        ),
        (
            "a special name past the ids",
            vec![(tk, "\"267\": {", "\"9267\": {")],
            "added_tokens_decoder[9267] is outside the special ids 267..523",
        ),
        (
            "a special name below the ids",
            vec![(tk, "\"267\": {", "\"100\": {")],
            "added_tokens_decoder[100] is outside the special ids 267..523",
        ),
        (
            "a special token the config names that no id spells",
            vec![(tk, "\"content\": \"[BOS]\"", "\"content\": \"[BOSX]\"")],
            "bos_token \"[BOS]\" is no special token",
        ),
        (
            "a special token at another id than the config's",
            vec![(
                "config.json",
                "\"bos_token_id\": 267",
                "\"bos_token_id\": 268",
            )],
            "bos_token \"[BOS]\" is id 267, where config.json's bos_token_id is 268",
        ),
        (
            "two ids of one name",
            vec![(tk, "\"content\": \"[EOS]\"", "\"content\": \"[BOS]\"")],
            "names another special id too",
        ),
        (
            "a special name that is a base token",
            vec![(tk, "\"content\": \"[EOS]\"", "\"content\": \"he\"")],
            "special token \"he\" is also a token of tiktoken.model",
        ),
        (
            "a name that is another id's reserved one",
            vec![(
                tk,
                "\"content\": \"[EOS]\"",
                "\"content\": \"<|reserved_token_300|>\"",
            )],
            "another special id's reserved name",
        ),
        (
            "an additional special token no id spells",
            vec![(tk, "\"[EOT]\"\n  ]", "\"<|nope|>\"\n  ]")],
            "additional_special_tokens \"<|nope|>\" is no special token",
        ),
        (
            "an automatic BOS the tokenizer never adds",
            vec![(
                tk,
                "\"extra_special_tokens\": {}",
                "\"extra_special_tokens\": {}, \"add_bos_token\": true",
            )],
            "add_bos_token true is unsupported",
        ),
        (
            "an extra special token",
            vec![(
                tk,
                "\"extra_special_tokens\": {}",
                "\"extra_special_tokens\": {\"x\": \"[BOS]\"}",
            )],
            "tokenizer_config.json.extra_special_tokens",
        ),
        (
            "another tokenizer class",
            vec![(tk, "\"TikTokenTokenizer\"", "\"Qwen2Tokenizer\"")],
            "tokenizer_class \"Qwen2Tokenizer\" is unsupported",
        ),
        (
            "another tokenizer module",
            vec![(
                tk,
                "tokenization_kimi.TikTokenTokenizer",
                "tokenization_other.TikTokenTokenizer",
            )],
            "tokenizer_config.json.auto_map is not",
        ),
        (
            "a key no reader honors",
            vec![(
                tk,
                "\"extra_special_tokens\": {}",
                "\"extra_special_tokens\": {}, \"vocab_file\": \"x\"",
            )],
            "tokenizer_config.json.vocab_file is not a key this converter reads",
        ),
        (
            "a regex the row does not spell",
            vec![(kimi, "r\"\"\"\\p{N}{1,3}\"\"\"", "r\"\"\"\\p{N}\"\"\"")],
            unpinned,
        ),
        (
            "another special-token count",
            vec![(
                kimi,
                "num_reserved_special_tokens = 256",
                "num_reserved_special_tokens = 128",
            )],
            unpinned,
        ),
        (
            "another reserved-name scheme",
            vec![(kimi, "<|reserved_token_{i}|>", "<|reserved_{i}|>")],
            unpinned,
        ),
        (
            "a generation setting no reader honors",
            vec![(
                "generation_config.json",
                "\"max_length\": 4096",
                "\"num_beams\": 2",
            )],
            "generation_config.json.num_beams is not a key this converter reads",
        ),
        (
            "no stop token anywhere",
            vec![
                ("generation_config.json", ",\n  \"eos_token_id\": 269", ""),
                ("config.json", "  \"eos_token_id\": 269,\n", ""),
                (tk, "\"eos_token\": \"[EOS]\"", "\"eos_token\": null"),
            ],
            "no generation stop tokens",
        ),
        (
            "a generation BOS that is not the config's",
            vec![(
                "generation_config.json",
                "\"eos_token_id\": 269",
                "\"eos_token_id\": 269, \"bos_token_id\": 268",
            )],
            "bos_token_id 268 disagrees",
        ),
    ];
    for (what, edits, named) in cases {
        let err = read(&changed(&edits))
            .err()
            .unwrap_or_else(|| panic!("{what}: read"))
            .to_string();
        assert!(err.contains(named), "{what}: {err}");
    }
}

/// The stop set is the folder's own word and nothing guessed: every id of `generation_config.json`'s
/// `eos_token_id` (a number or a list; `config.json`'s where it has none) and the token
/// `tokenizer_config.json`'s `eos_token` names. Serving stops at exactly those, so the toy's `[EOT]`
/// (a stop name for the engine's guesses) stays out.
#[test]
fn the_stop_set_is_the_folders_own() {
    let tk = "tokenizer_config.json";
    let generation = "generation_config.json";
    let (named, unnamed) = ("\"eos_token\": \"[EOS]\"", "\"eos_token\": null");
    let (turn_end, end) = ((269, "<|end_of_msg|>"), (268, "[EOS]"));
    // (what changed, its edits, the stop set that results)
    let cases: Vec<(&str, Vec<Edit>, Vec<Stop>)> = vec![
        (
            "the toy as it ships: the number and the named token",
            vec![],
            vec![end, turn_end],
        ),
        (
            "no eos_token: the number alone",
            vec![(tk, named, unnamed)],
            vec![turn_end],
        ),
        (
            "eos_token the number's own token",
            vec![(tk, named, "\"eos_token\": \"<|end_of_msg|>\"")],
            vec![turn_end],
        ),
        (
            "a list, and the named token",
            vec![(
                generation,
                "\"eos_token_id\": 269",
                "\"eos_token_id\": [270, 269]",
            )],
            vec![end, turn_end, (270, "<|open|>")],
        ),
        (
            "no generation id: config.json's, and the named token",
            vec![(generation, ",\n  \"eos_token_id\": 269", "")],
            vec![end, turn_end],
        ),
    ];
    for (what, edits, stop) in cases {
        let keys = read(&changed(&edits)).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(spelled(&keys, &keys.stop), pairs(&stop), "{what}");
        let ids = stop.iter().map(|&(id, _)| Value::U32(id)).collect();
        assert_eq!(
            get(&keys, "tokenizer.ggml.eog_token_ids"),
            Some(&Value::Array(ids)),
            "{what}"
        );
        served(&keys);
    }
}

/// Files that are missing or not text are named: a missing token file, module, tokenizer config or
/// config; bytes that are not UTF-8; an empty file.
#[test]
fn a_missing_or_unreadable_file_is_named() {
    let without = |file: &str| {
        let dir = folder(|_, text| text);
        fs::remove_file(dir.join(file)).unwrap();
        dir
    };
    let err = |dir: PathBuf| read(&dir).expect_err("refused").to_string();
    // No tiktoken.model and no tokenizer.json: the HF path's refusal, naming both.
    assert!(err(without("tiktoken.model")).contains("no tokenizer.json (and no tiktoken.model)"));
    assert!(err(without("tokenization_kimi.py")).contains("no tokenization_kimi.py"));
    assert!(
        err(without("tokenizer_config.json"))
            .contains("tokenizer_config.json.tokenizer_class is missing")
    );
    let dir = folder(|_, text| text);
    fs::write(dir.join("tiktoken.model"), b"QQ== 0\n\xff\n").unwrap();
    assert!(err(dir).contains("tiktoken.model: invalid utf-8"));
    let dir = folder(|_, text| text);
    fs::write(dir.join("tiktoken.model"), b"").unwrap();
    assert!(err(dir).contains("tiktoken.model has no lines"));
}

/// The model names the reader: a spec with `with_pre("kimi-k2")` reads `tiktoken.model` even
/// beside a `tokenizer.json`, a folder with no `tokenizer.json` reads it with no name given, a name
/// of another row is refused against this folder, and every row is one the tokenizer runs.
#[test]
fn a_model_s_pre_tokenizer_row_picks_the_reader() {
    let dir = folder(|_, text| text);
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    let config = crate::read_json(&dir, "config.json", Error::Io, Error::Json)
        .unwrap()
        .unwrap();
    let spec = VocabSpec::parse(&config).unwrap();
    assert!(crate::vocab::keys(&dir, &spec.with_pre("kimi-k2").unwrap()).is_ok());
    let spec = VocabSpec::parse(&config)
        .unwrap()
        .with_pre("qwen2")
        .unwrap();
    assert!(
        crate::vocab::keys(&dir, &spec)
            .unwrap_err()
            .to_string()
            .contains("tokenizer.json.decoder is missing")
    );
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let err = crate::vocab::keys(&dir, &spec).unwrap_err().to_string();
    assert!(
        err.contains("not one of the tiktoken rows: kimi-k2"),
        "{err}"
    );
    let err = VocabSpec::parse(&config)
        .unwrap()
        .with_pre("kimi-k3")
        .unwrap_err()
        .to_string();
    assert!(err.contains("kimi-k2"), "{err}");
    for row in &PRE {
        assert!(tokenizer::runs_pre_tokenizer(row.name), "{}", row.name);
    }
}

/// The Kimi K3 folder's vocabulary is the header llama.cpp's converter wrote for it, which the
/// Unsloth GGUF carries: every token, type and merge, the model and pre-tokenizer names, and the
/// BOS, EOS and padding ids; the header names no unknown or separator id. The keys also pass the
/// checks the folder map makes of serving's tokenizer. The folder (`tiktoken.model`,
/// `tokenization_kimi.py`, `tokenizer_config.json`, `config.json`, `generation_config.json` of
/// `moonshotai/Kimi-K3`) is at `$BLOOMERY_KIMI_K3_HF`; the header is the first shard of the K3
/// GGUF at `$BLOOMERY_KIMI_K3_GGUF`, by default the box's.
#[test]
#[ignore = "needs the Kimi-K3 folder's vocabulary files at $BLOOMERY_KIMI_K3_HF and the K3 GGUF's first shard"]
fn hw_the_kimi_k3_folder_derives_the_gguf_header_vocabulary() {
    let dir = PathBuf::from(
        std::env::var_os("BLOOMERY_KIMI_K3_HF").expect("$BLOOMERY_KIMI_K3_HF names the folder"),
    );
    let header = std::env::var_os("BLOOMERY_KIMI_K3_GGUF").map_or_else(
        || PathBuf::from("/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf"),
        PathBuf::from,
    );
    let keys = read(&dir).expect("the folder's vocabulary reads");
    let gguf = gguf::Gguf::open(&header).expect("the header opens");
    for key in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        let (ours, theirs) = (get(&keys, key), gguf.value(key));
        let at = match (ours, theirs) {
            (Some(Value::Array(a)), Some(Value::Array(b))) => {
                format!(
                    "{} and {} entries, first difference at {:?}",
                    a.len(),
                    b.len(),
                    a.iter().zip(b).position(|(x, y)| x != y)
                )
            }
            _ => format!("{ours:?} and {theirs:?}"),
        };
        assert!(ours.is_some() && ours == theirs, "{key}: {at}");
    }
    for key in [
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.seperator_token_id",
    ] {
        assert!(
            get(&keys, key).is_none() && gguf.value(key).is_none(),
            "{key}"
        );
    }
    // The stop set is the folder's: `generation_config.json`'s end of message and the `[EOS]` of
    // `tokenizer_config.json`; the header names no stop set of its own.
    assert_eq!(
        spelled(&keys, &keys.stop),
        pairs(&[(163_585, "[EOS]"), (163_586, "<|end_of_msg|>")])
    );
    let served = served(&keys);
    assert_eq!(served.eog(), [163_585, 163_586]);
    assert_eq!(served.pre(), "kimi-k2");
}
