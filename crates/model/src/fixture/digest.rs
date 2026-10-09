//! The generator-fill guard. A fixture's bytes are a function of the seed and
//! the source header alone, and the `bloomery.fixture.*` keys the reference
//! sets' `# fixture` line carries (`refset::fixture::line`) cannot see a change
//! to the fill code that leaves them as they were. This test pins what the
//! fill writes: per type, the sha256 of one tensor of that type filled under a
//! fixed name, seed, row length and window ([`PINS`]). A fill change that moves
//! a pinned digest turns the test red, and the way out is the version bump,
//! which reaches every fixture file's header, so every set dumped from an
//! older file is stale and is dumped again.

use gguf::GgmlType;
use sha2::{Digest, Sha256};

use super::FIXTURE_VERSION;
use super::fill::{CHUNK_TARGET, Rule, Window, rule_for};
use super::plan::PlannedTensor;

/// The pinned digests: `(version, [(type, sha256 hex)])`, versions strictly
/// increasing, the last row's version [`FIXTURE_VERSION`]. A type is the name
/// ggml gives it (`q4_K`); `const` is a constant [`Rule::Const`] tensor, which
/// no type's rule makes.
///
/// The procedure when a digest moves:
/// - A changed digest of a type the last row holds is a new row at
///   `FIXTURE_VERSION + 1` with the digests of today, and `FIXTURE_VERSION`
///   moves to it in the same change. The bump reaches every fixture set's
///   `# fixture` line, so each set dumped from an older file is stale and is
///   dumped again. A type whose rule did not change keeps its digest in the new
///   row.
/// - A type new to [`rule_for`] gets its digest added to the last row, with no
///   bump: no fixture file holds it, so no set goes stale.
///
/// This table sees a change to the fill only through the digests: an edit of
/// an existing digest in place, to make the test pass, is something no table
/// can see. The procedure above and the version bump in review are the guard.
///
/// `TBD` is a digest not yet taken: the test prints `fill digest <type> <hex>`
/// for every type before it asserts, and the box run's lines fill the row in.
const PINS: &[(u32, &[(&str, &str)])] = &[(
    1,
    &[
        (
            "q3_K",
            "963fa75e48fbe7cf4c62bb4ec19608ba7daf73b93cd4dcf4b2239ad3e0e08976",
        ),
        (
            "q4_K",
            "73571a040cba32ee0c57326a0eb8f95e940ff794d0cf9e115e1c121edef50087",
        ),
        (
            "q5_K",
            "ac37581a56d3910400bbd168372155760015cbd54b83415774f6f5de9bf3a0a3",
        ),
        (
            "q6_K",
            "ddae9e3f2ea90d2e3d241a460dc9fe587956554c6585fe0683a01ea220661b50",
        ),
        (
            "q8_0",
            "1a163e851e3452d59e9c88b55d65a0f30ba2570c546d235ddec44ff87e8c0948",
        ),
        (
            "q5_1",
            "318ad769954745bd74e463be51bc0958df90a86e012121e07fc4f869e0cbdfbc",
        ),
        (
            "iq4_nl",
            "d727851cb60a71be84a60c8c931e88699791f80b6f20791cde7897a7663b04e5",
        ),
        (
            "mxfp4",
            "e193a24fd5e8785529c6ad983f2abc083128d3c69fe1a20a6d43eab0718411d4",
        ),
        (
            "f32",
            "170e80ead00fcfc810f67f86d91c037b2c1e5c9c9575af44121401f10e1d5c81",
        ),
        (
            "f16",
            "05f2c18c1042dd31e5afd3ed9941f4552ec52bb03e693987592205e197a39678",
        ),
        (
            "bf16",
            "cad63e750127c0d3dd59e5719cbe413a542e6580a558dbf549e4b752431d315b",
        ),
        (
            "const",
            "9e1f48eb661d7032ee6c87ae4d79fe3801174a586820e965aa28bf09689a9c39",
        ),
    ],
)];

/// The types [`rule_for`] takes, in its order.
const TYPES: [GgmlType; 11] = [
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::Q5_1,
    GgmlType::IQ4_NL,
    GgmlType::MXFP4,
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
];

/// The name every tensor is generated under: a stream is keyed by it.
const NAME: &str = "digest.weight";
/// The row length: a multiple of every type's block.
const K: u64 = 256;
/// The seed.
const SEED: u64 = 1;
/// Each tensor is at least this many chunks, so the chunk keying is pinned too.
const CHUNKS: usize = 3;

/// A tensor of `ty` and `rule`, rows of [`K`] values, at least [`CHUNKS`]
/// chunks of bytes.
fn tensor(ty: GgmlType, rule: Rule) -> PlannedTensor {
    let (blck, size) = (
        ty.blck_size().expect("a type with a rule has a block size"),
        ty.type_size().expect("a type with a rule has a size"),
    );
    let row = size * (K / blck);
    let rows = ((CHUNKS * CHUNK_TARGET) as u64).div_ceil(row);
    PlannedTensor {
        name: NAME.to_string(),
        source: NAME.to_string(),
        layer: Some(0),
        dims: vec![K, rows],
        ty,
        nbytes: row * rows,
        rule,
    }
}

/// Lowercase hex sha256 of the bytes `t` fills under [`SEED`].
fn digest(t: &PlannedTensor) -> String {
    let mut out = vec![0u8; t.nbytes as usize];
    t.fill(SEED, 1, &mut out);
    Sha256::digest(&out)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Today's digest of each type [`rule_for`] takes and of a constant tensor,
/// as `(label, hex)`.
fn today() -> Vec<(String, String)> {
    let window = Window::new(1.0 / 16384.0, 1.0 / 64.0)
        .expect("[2^-14, 2^-6] is inside the normal f16 values");
    let mut out: Vec<(String, String)> = TYPES
        .iter()
        .map(|&ty| {
            let rule = rule_for(NAME, ty, K, window)
                .unwrap_or_else(|e| panic!("{ty} at K = {K} in {window}: {e}"));
            (ty.to_string(), digest(&tensor(ty, rule)))
        })
        .collect();
    let constant = tensor(GgmlType::F32, Rule::Const { value: 0.25 });
    out.push(("const".to_string(), digest(&constant)));
    out
}

/// The fill writes the pinned bytes: the table's versions increase and end at
/// [`FIXTURE_VERSION`], each row moves at least one digest of the row before,
/// and the last row holds a digest for every type [`rule_for`] takes (and
/// nothing else) that is today's.
#[test]
fn the_fill_of_each_type_is_the_pinned_digest() {
    let today = today();
    for (label, hex) in &today {
        println!("fill digest {label} {hex}");
    }

    let (last_version, last) = PINS.last().expect("the table has a row");
    assert_eq!(
        *last_version, FIXTURE_VERSION,
        "the last row is version {last_version}, FIXTURE_VERSION is {FIXTURE_VERSION}: a bump needs its row"
    );
    for pair in PINS.windows(2) {
        let ((before, rows_before), (after, rows_after)) = (pair[0], pair[1]);
        assert!(
            before < after,
            "versions {before} and {after} do not increase"
        );
        let moved = rows_after
            .iter()
            .any(|(label, hex)| rows_before.iter().any(|(l, h)| l == label && h != hex));
        assert!(
            moved,
            "row {after} moves no digest of row {before}: a bump with no fill change stales every set for nothing"
        );
    }

    let pinned: Vec<&str> = last.iter().map(|(l, _)| *l).collect();
    let taken: Vec<&str> = today.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(
        pinned, taken,
        "the last row's types are not the types rule_for takes (a new type is added to the last row, with no bump)"
    );
    let moved: Vec<String> = today
        .iter()
        .zip(last.iter())
        .filter(|((_, hex), (_, pin))| hex != pin)
        .map(|((label, hex), (_, pin))| format!("{label}: pinned {pin}, today {hex}"))
        .collect();
    assert!(
        moved.is_empty(),
        "the fill moved for {} type(s) of version {last_version}: {moved:#?}; a digest of a type the last row holds \
         needs a row at version {} and the bump (see PINS)",
        moved.len(),
        FIXTURE_VERSION + 1
    );
}
