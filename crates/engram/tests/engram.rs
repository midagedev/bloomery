//! Gates for the engram crate: what the mapping serves, and which rows the
//! hash asks it for. Not how fast — speed belongs to `engram-rate` under a
//! lease. The identity gates read only the rows they name, a few hundred page
//! faults.
//!
//! `hw_` prefix: these need the box and the V4.1 split. They do not need the
//! oracle — the reference here is the file itself, read a second way.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use engram::rows::{PageAdvice, SplitTable};
use engram::{Engram, EngramError, RowTable, Site};
use gguf::Split;

/// The split set the gates open: the directory of [`gguf::v41::model`].
fn model_dir() -> String {
    gguf::v41::dir().display().to_string()
}

fn open() -> Engram {
    let dir = model_dir();
    Engram::open_dir(&dir).unwrap_or_else(|e| {
        panic!(
            "no engram table under {dir} ({e}). The V4.1 split lives on the box; \
             set BLOOMERY_V41_DIR if it moved. Do not skip this test."
        )
    })
}

/// Ids uniform over a range from a fixed seed (splitmix64), so a gate reads
/// the same rows on every run.
struct Draw(u64);

impl Draw {
    /// The next id in `0..n`.
    fn below(&mut self, n: u64) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n) as u32
    }
}

/// Ids worth reading: the first row, the last row, one row that starts close
/// enough to the end of a page that it straddles into the next, and a seeded
/// spread over the table.
fn sample(site: &Site, n: usize) -> Vec<u32> {
    const PAGE: u64 = 4096;
    let mut ids = vec![0u32, (site.rows() - 1) as u32];

    // Row starts land on a lattice: with stride s the residues mod the page
    // repeat with period 4096/gcd(s, 4096) in the id (256 for Q8_0's 272 B,
    // 2,048 for Q3_K's 110 B). One of them straddles.
    let straddler = (0..4096u32)
        .find(|&r| site.file_offset(r) % PAGE + site.row_bytes() > PAGE)
        .unwrap_or_else(|| {
            panic!(
                "{}: no row in the first 4,096 straddles a page — the row stride {} B \
                 does not lattice the page as this gate assumes",
                site.name(),
                site.row_bytes()
            )
        });
    ids.push(straddler);

    let mut draw = Draw(0xB3E7_A001);
    ids.extend((0..n).map(|_| draw.below(site.rows())));
    ids
}

/// The mapping serves the file's bytes, prefetched or not.
///
/// This is the whole claim of the IO half: a row reached through the mmap is
/// the same bytes a plain `pread` returns **at the offset the GGUF header
/// gives** — `data_base + tensor.offset + id x stride`, recomputed here from the
/// header rather than asked of the crate. Asking the crate would compare the
/// mapping against itself: a stride or base that is wrong on both sides agrees
/// with itself, and a one-byte shift of `Site::file_offset` passes.
///
/// So this catches an offset off by a row or a byte, a stride taken from the
/// wrong type, a shard opened at the wrong tensor, and a prefetch that advises
/// a different range than the read touches.
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_rows_match_pread() {
    let engram = open();
    assert_eq!(
        engram.sites().len(),
        2,
        "V4.1-Flash has engram at blk.1 and blk.14"
    );

    for site in engram.sites() {
        let ids = sample(site, 128);
        let fd = File::open(site.path()).unwrap();
        // The reference offset, derived here: header data base + the tensor's
        // own offset, and a stride of ne[0]/block blocks of ggml's block size.
        let inv = gguf::inventory_of(site.path()).unwrap();
        let t = inv
            .tensors
            .iter()
            .find(|t| t.name == site.name())
            .expect("the site came from this shard's header");
        let base = inv.data_base + t.offset;
        let (_, block, block_bytes) =
            gguf::ggml_type_info(t.type_id).expect("an engram table type ggml sizes");
        let stride = t.dims[0] / block * block_bytes;

        let mut want = vec![0u8; stride as usize];
        let mut borrowed: Vec<&[u8]> = Vec::new();

        // Both arms: the advised read and the plain one must agree with the file.
        for prefetched in [false, true] {
            if prefetched {
                site.prefetch(&ids).unwrap();
            }
            site.rows_into(&ids, &mut borrowed).unwrap();
            assert_eq!(borrowed.len(), ids.len(), "{}: rows served", site.name());

            for (&id, got) in ids.iter().zip(&borrowed) {
                let at = base + u64::from(id) * stride;
                fd.read_exact_at(&mut want, at).unwrap();
                assert_eq!(
                    *got,
                    &want[..],
                    "{} row {id} (prefetched {prefetched}): the mapping disagrees with a \
                     pread at the header's offset {at}",
                    site.name(),
                );
            }
        }
    }
}

/// A copied row is the borrowed row.
///
/// The copied path is what a helper thread hands the step thread, so it is the
/// bytes the engine will actually consume — and it is the one path with an
/// arithmetic of its own: `ids.len() * row_bytes` of destination, sliced per
/// row. An off-by-one stride there, or a straddling row copied short, would be
/// invisible to the borrowing gates above. The sample carries the page
/// straddler for exactly that reason.
///
/// The same table read through its split's own mapping ([`SplitTable`], the
/// read Qwen3.8's PLE table takes from the NVMe tier) copies the same bytes,
/// and its advice takes rows that start mid-page: each is rounded out to the
/// whole pages that hold it, which `madvise` requires.
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_copy_rows_matches_row() {
    let engram = open();
    let split = Arc::new(Split::open(gguf::v41::model()).unwrap());

    for site in engram.sites() {
        let ids = sample(site, 64);
        let stride = site.row_bytes() as usize;
        let mut copied = vec![0u8; ids.len() * stride];
        site.copy_rows(&ids, &mut copied).unwrap();

        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(
                &copied[i * stride..(i + 1) * stride],
                site.row(id).unwrap(),
                "{} row {id}: the copy disagrees with the borrow",
                site.name(),
            );
        }

        // The length contract is the other half: a buffer that is not exactly
        // one row per id is an error, never a short copy.
        assert!(
            site.copy_rows(&ids, &mut copied[..ids.len() * stride - 1])
                .is_err(),
            "{}: a short buffer must be refused",
            site.name()
        );

        let table = SplitTable::open(Arc::clone(&split), site.name()).unwrap();
        assert_eq!(table.row_bytes(), site.row_bytes(), "{}", site.name());
        table.prefetch(&ids).unwrap();
        let mut via_split = vec![0u8; ids.len() * stride];
        table.copy_rows(&ids, &mut via_split).unwrap();
        assert!(
            via_split == copied,
            "{}: the split's mapping copies other bytes than the site's",
            site.name()
        );
        assert!(
            table
                .copy_rows(&ids, &mut via_split[..ids.len() * stride - 1])
                .is_err(),
            "{}: a short buffer must be refused through the split",
            site.name()
        );
    }
}

/// The row stride tiles the tensor exactly, and the last row ends on the
/// tensor's last byte.
///
/// What the identity gate above cannot catch: a wrong stride used by *both*
/// sides agrees with itself. This one re-derives the geometry from the header
/// inside the test and compares it against what the crate reports, so the
/// arithmetic has a second witness.
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_row_stride_tiles_the_tensor() {
    let engram = open();

    for site in engram.sites() {
        let inv = gguf::inventory_of(site.path()).unwrap();
        let t = inv
            .tensors
            .iter()
            .find(|t| t.name == site.name())
            .expect("the site came from this shard's header");
        let nbytes = t.nbytes.expect("the table's type is in ggml's size table");

        assert_eq!(t.dims.len(), 2, "{}: engram is a 2-D table", site.name());
        assert_eq!(site.rows(), t.dims[1], "{}: row count", site.name());
        assert_eq!(site.type_id(), t.type_id, "{}: row type", site.name());
        // 256 values a row. The two block types the V4.1 files carry, from
        // ggml-common.h's static_asserts, written here rather than read from
        // the size table the crate reads: Q8_0 = 32 values in 34 B, Q3_K = 256
        // values in 110 B.
        let (block, block_bytes) = match t.type_id {
            8 => (32, 34),
            11 => (256, 110),
            ty => panic!("{}: ggml type {ty} is not a V4.1 table type", site.name()),
        };
        assert_eq!(
            site.row_bytes(),
            t.dims[0] / block * block_bytes,
            "{}: row stride from the type's block",
            site.name()
        );
        assert_eq!(
            site.rows() * site.row_bytes(),
            nbytes,
            "{}: {} rows x {} B must tile the tensor's {nbytes} B exactly",
            site.name(),
            site.rows(),
            site.row_bytes(),
        );

        let last_end = site.file_offset((site.rows() - 1) as u32) + site.row_bytes();
        assert_eq!(
            last_end,
            inv.data_base + t.offset + nbytes,
            "{}: the last row must end on the tensor's last byte",
            site.name()
        );
        assert!(
            last_end <= inv.file_len,
            "{}: the table runs past the {} B shard",
            site.name(),
            inv.file_len
        );
    }
}

/// The port's formula, written out a second time from the same constants.
///
/// The gate below compares this against `Hash::rows_into`, the way
/// `hw_engram_rows_match_pread` compares the mapping against a `pread` at an
/// offset it recomputes: asking the crate for both sides would let a wrong
/// stride, a swapped context slot or an off-by-one agree with itself. Read it
/// beside `llama_set_engram_rows` in the port, which is where it comes from.
fn reference_rows(hash: &engram::Hash, site: usize, window: &[u64], out: &mut [u32]) {
    let mult = hash.multipliers(site).unwrap();
    let prime = hash.primes(site).unwrap();
    let offset = hash.offsets(site).unwrap();
    let heads = hash.n_heads();

    let mut rolling = window[0].wrapping_mul(mult[0]);
    for s in 1..hash.n_gram() {
        rolling ^= window[s].wrapping_mul(mult[s]);
        for h in 0..heads {
            let b = (s - 1) * heads + h;
            out[b] = (rolling % prime[b] + offset[b]) as u32;
        }
    }
}

/// The 24 buckets partition the site's table exactly, every id the hash
/// produces is a row of that table, and the ids are the port's own formula.
///
/// Three facts, one contract. The primes sum to the row count the *header*
/// states — a second witness, not a literal — and the offsets are their
/// exclusive prefix sums, so bucket `b` owns `[offset[b], offset[b] + prime[b])`
/// and the intervals tile the table with no gap and no overlap. Given those
/// two, every `offset[b] + rolling % prime[b]` is in range and in its own
/// bucket; the ids drawn here are the third fact, checked rather than argued,
/// because the arithmetic that produces them is the part that can be wrong.
///
/// The windows cover the two shapes that differ: every slot a token inside
/// the map, and a window fresh from a sequence start, where every older slot
/// is pad — the start the port spends one token in. A third of the ids drawn
/// are past the map's end; the map is the whole vocabulary, so each of those
/// is not a token of the model and must be refused by name, with its id and
/// the map's length, never mapped to a value a window could carry.
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_hash_buckets_partition_the_table() {
    let engram = open();
    let hash = engram.hash();
    assert_eq!(
        hash.sites(),
        engram.sites().len(),
        "one site per engram layer the metadata names"
    );

    let n_cols = hash.n_cols();
    assert_eq!(
        n_cols,
        (hash.n_gram() - 1) * hash.n_heads(),
        "buckets are (n_gram - 1) x heads"
    );

    let vocab = hash.token_map().len() as u32;
    let mut draw = Draw(0xB3D0_1CE5);
    let mut ids = vec![0u32; n_cols];

    for (e, site) in engram.sites().iter().enumerate() {
        let primes = hash.primes(e).unwrap();
        let offsets = hash.offsets(e).unwrap();

        assert_eq!(
            hash.partition_rows(e).unwrap(),
            site.rows(),
            "{}: the buckets must cover the table the header describes exactly",
            site.name()
        );
        let mut at = 0u64;
        for (b, (&p, &o)) in primes.iter().zip(offsets).enumerate() {
            assert_ne!(p, 0, "{}: bucket {b} divides by its prime", site.name());
            assert_eq!(
                o,
                at,
                "{}: bucket {b} must start where bucket {} ended",
                site.name(),
                b.wrapping_sub(1)
            );
            at += p;
        }

        // 10,000 draws: two thirds inside the vocabulary, one third past its
        // end, and every hundredth window starting fresh at a sequence start.
        let mut window = vec![hash.pad_id(); hash.n_gram()];
        let mut want = vec![0u32; n_cols];
        for n in 0..10_000u32 {
            if n % 100 == 0 {
                window.fill(hash.pad_id());
            }
            let token = draw.below(u64::from(vocab) * 3 / 2);
            let mapped = match hash.map_token(token) {
                Ok(mapped) if token < vocab => mapped,
                Err(EngramError::TokenPastMap { token: t, map })
                    if t == token && token >= vocab && map == vocab as usize =>
                {
                    continue;
                }
                other => panic!(
                    "context {n}: token {token} against a map of {vocab} entries gave \
                     {other:?}: an id inside the map maps, an id past its end is refused \
                     by name with its id and the map's length"
                ),
            };
            window.rotate_right(1);
            window[0] = mapped;
            hash.rows_into(e, &window, &mut ids).unwrap();
            reference_rows(hash, e, &window, &mut want);
            assert_eq!(
                ids,
                want,
                "{}: context {n} — the crate's ids differ from the formula \
                 re-derived here from the same constants",
                site.name()
            );

            for (b, &id) in ids.iter().enumerate() {
                let lo = offsets[b];
                let hi = lo + primes[b];
                assert!(
                    u64::from(id) >= lo && u64::from(id) < hi,
                    "{}: context {n} bucket {b} produced row {id}, outside its \
                     interval [{lo}, {hi})",
                    site.name(),
                );
                assert!(
                    u64::from(id) < site.rows(),
                    "{}: context {n} bucket {b} produced row {id}, past the \
                     {} the table holds",
                    site.name(),
                    site.rows(),
                );
            }
        }
    }
}

/// `token_map`'s codomain is the compressed vocabulary: it covers `0..=max`
/// with no hole, and its domain is the tokenizer's whole vocabulary.
///
/// A distinct contract from the partition above, and the one the hash's first
/// line rests on. Both halves have a witness inside the same file, so neither
/// is a literal: the domain is checked against `tokenizer.ggml.tokens`, and the
/// codomain's density is checked against itself — a map with a hole would be a
/// compression that lost a slot. The pad a window takes before the sequence
/// starts must itself be inside the codomain, or a sequence start would
/// address a row no trained embedding sits in.
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_token_map_is_the_compressed_vocabulary() {
    let engram = open();
    let hash = engram.hash();
    let map = hash.token_map();

    // The metadata is in shard 1 and the tables are not (blk.1's is in shard 2),
    // so the carrier is found here rather than asked of a site. Finding it the
    // way the crate does would compare the crate against itself.
    let dir = model_dir();
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let inv = shards
        .iter()
        .map(|p| gguf::inventory_of(p).unwrap())
        .find(|inv| inv.value("deepseek41.engram.token_map").is_some())
        .unwrap_or_else(|| panic!("no shard under {dir} carries the engram metadata"));
    let Some(gguf::Value::Array(tokens)) = inv.value("tokenizer.ggml.tokens") else {
        panic!("the shard that carries the engram metadata must also carry the vocabulary");
    };
    assert_eq!(
        map.len(),
        tokens.len(),
        "the map is indexed by token id, so it is as wide as the vocabulary"
    );

    let max = *map.iter().max().expect("the map is not empty");
    let mut hit = vec![false; max as usize + 1];
    for &v in map {
        hit[v as usize] = true;
    }
    let holes = hit.iter().filter(|h| !**h).count();
    assert_eq!(
        holes, 0,
        "the codomain 0..={max} must be dense; {holes} of its values are unused, \
         so the map is not a compression of the vocabulary"
    );
    assert!(
        hash.pad_id() <= u64::from(max),
        "pad {} is outside the codomain 0..={max}, so a sequence start would \
         address a row the model never trained",
        hash.pad_id()
    );
}

// ---------------------------------------------------------------------------
// A routed expert stack as a row table: a synthetic file, no model on the box.

/// The synthetic file's tensors, in file order. A filler first, so the stack's
/// bytes start 64 B into the data section and not on a page; the stack's
/// expert is 250 x 7 F32 values, 7,000 B, so no expert starts or ends on a
/// page either.
const PAD: &str = "pad.weight";
const STACK: &str = "blk.0.ffn_gate_exps.weight";
const TABLE: &str = "tab.weight";
const RANK4: &str = "rank4.weight";
const RANK1: &str = "rank1.weight";
const EXPERTS: u64 = 9;
const EXPERT_BYTES: u64 = 250 * 7 * 4;
const TABLE_ROWS: u64 = 20;
const TABLE_ROW_BYTES: u64 = 250 * 4;

/// Deterministic bytes (xorshift64*), so a shifted or short read differs.
fn random_bytes(seed: u64, n: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect()
}

/// A scratch directory under the build's own target directory (a disk, so the
/// page-cache advice has pages to act on), removed on drop.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("engram-{tag}-{}", std::process::id()));
        if d.exists() {
            std::fs::remove_dir_all(&d).unwrap();
        }
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The synthetic one-shard model in `dir`: returns its path and each tensor's
/// bytes, by name.
fn write_synthetic(dir: &std::path::Path) -> (std::path::PathBuf, Vec<(&'static str, Vec<u8>)>) {
    use gguf::write::{Layout, TensorDecl, Writer};
    let f32_tensor = |name: &'static str, dims: &[u64], seed: u64| {
        let nbytes = dims.iter().product::<u64>() * 4;
        (
            TensorDecl {
                name: name.to_string(),
                dims: dims.to_vec(),
                type_id: 0,
                nbytes,
            },
            random_bytes(seed, nbytes),
        )
    };
    let tensors = [
        f32_tensor(PAD, &[13], 1),
        f32_tensor(STACK, &[250, 7, EXPERTS], 2),
        f32_tensor(TABLE, &[250, TABLE_ROWS], 3),
        f32_tensor(RANK4, &[4, 3, 2, 2], 4),
        f32_tensor(RANK1, &[16], 5),
    ];
    let layout = Layout::new(&[], tensors.iter().map(|(d, _)| d.clone()).collect()).unwrap();
    let path = dir.join("synthetic.gguf");
    let mut w = Writer::new(
        std::io::BufWriter::new(File::create(&path).unwrap()),
        layout,
    )
    .unwrap();
    for (d, bytes) in &tensors {
        w.tensor(&d.name, bytes).unwrap();
    }
    w.finish()
        .unwrap()
        .into_inner()
        .unwrap()
        .sync_all()
        .unwrap();
    let named = tensors
        .into_iter()
        .map(|(d, b)| {
            let name: &'static str = [PAD, STACK, TABLE, RANK4, RANK1]
                .into_iter()
                .find(|n| *n == d.name)
                .unwrap();
            (name, b)
        })
        .collect();
    (path, named)
}

/// Where `name`'s first byte is in the file, from the header: the second
/// witness the row bytes are compared at, as `hw_engram_rows_match_pread` does.
fn header_offset(path: &std::path::Path, name: &str) -> u64 {
    let inv = gguf::inventory_of(path).unwrap();
    let t = inv.tensors.iter().find(|t| t.name == name).unwrap();
    inv.data_base + t.offset
}

/// Open `name` of `split` the way a routed stack is read.
fn open_table(split: &Arc<Split>, name: &str) -> Result<SplitTable, EngramError> {
    SplitTable::open_with(Arc::clone(split), name, PageAdvice::Normal)
}

/// Expert `e` of the stack, through the row table, is the bytes the file holds
/// at the header's offset for it.
///
/// What it holds the crate to: the row stride of a 3-D tensor is the bytes of
/// one `[ne0, ne1]` slab, and the rows are the experts, so a stride taken from
/// the first two dims only (a quarter of the bytes here), or from the wrong
/// type, or an id counted from the wrong end, reads other bytes than a `pread`
/// at `data base + offset + e x slab`.
#[test]
#[ignore = "hw: the box's disk-backed target directory"]
fn hw_engram_expert_stack_rows_are_the_files_bytes() {
    let scratch = Scratch::new("stack");
    let (path, tensors) = write_synthetic(&scratch.0);
    let base = header_offset(&path, STACK);
    assert_ne!(
        base % 4096,
        0,
        "the synthetic stack must start off a page boundary, or it does not exercise the rounding"
    );
    let split = Arc::new(Split::open(&path).unwrap());
    let table = open_table(&split, STACK).expect("a 3-D stack opens as a table of experts");
    assert_eq!(table.row_bytes(), EXPERT_BYTES, "one expert is one row");

    let file = File::open(&path).unwrap();
    let stack = &tensors.iter().find(|(n, _)| *n == STACK).unwrap().1;
    let ids: Vec<u32> = (0..EXPERTS as u32).rev().collect();
    let mut copied = vec![0u8; ids.len() * EXPERT_BYTES as usize];
    table.copy_rows(&ids, &mut copied).unwrap();
    for (i, &e) in ids.iter().enumerate() {
        let mut want = vec![0u8; EXPERT_BYTES as usize];
        file.read_exact_at(&mut want, base + u64::from(e) * EXPERT_BYTES)
            .unwrap();
        let got = &copied[i * EXPERT_BYTES as usize..(i + 1) * EXPERT_BYTES as usize];
        assert_eq!(
            got,
            &want[..],
            "expert {e}: the table disagrees with a pread"
        );
        assert_eq!(
            got,
            &stack[e as usize * EXPERT_BYTES as usize..(e as usize + 1) * EXPERT_BYTES as usize],
            "expert {e}: the table disagrees with the bytes the file was written from"
        );
    }
    assert!(
        matches!(
            table.copy_rows(&[EXPERTS as u32], &mut vec![0u8; EXPERT_BYTES as usize]),
            Err(EngramError::RowOutOfRange { id, rows, .. }) if id == EXPERTS as u32 && rows == EXPERTS
        ),
        "an expert id past the stack is refused by name"
    );
}

/// A 2-D table opens and reads as it did before the stack case.
#[test]
#[ignore = "hw: the box's disk-backed target directory"]
fn hw_engram_two_d_table_rows_are_unchanged() {
    let scratch = Scratch::new("table");
    let (path, tensors) = write_synthetic(&scratch.0);
    let base = header_offset(&path, TABLE);
    let split = Arc::new(Split::open(&path).unwrap());
    let table = open_table(&split, TABLE).expect("a 2-D table opens");
    assert_eq!(table.row_bytes(), TABLE_ROW_BYTES);

    let file = File::open(&path).unwrap();
    let source = &tensors.iter().find(|(n, _)| *n == TABLE).unwrap().1;
    let ids: Vec<u32> = vec![0, 7, TABLE_ROWS as u32 - 1];
    let mut copied = vec![0u8; ids.len() * TABLE_ROW_BYTES as usize];
    table.copy_rows(&ids, &mut copied).unwrap();
    for (i, &r) in ids.iter().enumerate() {
        let mut want = vec![0u8; TABLE_ROW_BYTES as usize];
        file.read_exact_at(&mut want, base + u64::from(r) * TABLE_ROW_BYTES)
            .unwrap();
        let got = &copied[i * TABLE_ROW_BYTES as usize..(i + 1) * TABLE_ROW_BYTES as usize];
        assert_eq!(got, &want[..], "row {r}: the table disagrees with a pread");
        assert_eq!(
            got,
            &source[r as usize * TABLE_ROW_BYTES as usize
                ..(r as usize + 1) * TABLE_ROW_BYTES as usize],
            "row {r}: the table disagrees with the bytes the file was written from"
        );
    }
}

/// A tensor that is neither a 2-D table nor a 3-D stack is refused, and the
/// refusal names the tensor and its dims.
#[test]
#[ignore = "hw: the box's disk-backed target directory"]
fn hw_engram_refuses_other_ranks_by_name() {
    let scratch = Scratch::new("rank");
    let (path, _) = write_synthetic(&scratch.0);
    let split = Arc::new(Split::open(&path).unwrap());
    for (name, dims) in [(RANK4, vec![4u64, 3, 2, 2]), (RANK1, vec![16u64])] {
        match open_table(&split, name) {
            Err(EngramError::NotATable {
                name: n, dims: d, ..
            }) => {
                assert_eq!(n, name);
                assert_eq!(d, dims, "{name}: the refusal carries the header's dims");
            }
            Err(other) => panic!("{name}: refused as {other:?}, not by name as NotATable"),
            Ok(_) => panic!("{name}: a tensor of rank {} opened as a table", dims.len()),
        }
    }
}

/// An eviction leaves none of the named experts' pages in the page cache, a
/// prefetch and populate leave all of them, and the file range a row names is
/// where the header puts it.
///
/// The drop is checked by `mincore`, which reports the page cache of a file
/// mapping to the file's owner (the box's gates run as the owner). The whole
/// stack is dropped first so a neighbour's page shared with a named expert
/// does not stay resident from the write.
#[test]
#[ignore = "hw: the box's disk-backed target directory"]
fn hw_engram_expert_stack_evicts_and_reads_back() {
    let scratch = Scratch::new("evict");
    let (path, _) = write_synthetic(&scratch.0);
    let base = header_offset(&path, STACK);
    let split = Arc::new(Split::open(&path).unwrap());
    let table = open_table(&split, STACK).unwrap();
    let all: Vec<u32> = (0..EXPERTS as u32).collect();
    let ids = [0u32, 3, 8];

    for &id in &all {
        assert_eq!(
            table.file_range(id).unwrap(),
            (base + u64::from(id) * EXPERT_BYTES, EXPERT_BYTES),
            "expert {id}: its range in the file"
        );
    }
    assert!(
        matches!(
            table.file_range(EXPERTS as u32),
            Err(EngramError::RowOutOfRange { .. })
        ),
        "a range past the stack is refused by name"
    );

    table.evict_rows(&all).unwrap();
    let cold = table.resident_pages(&ids).unwrap();
    assert!(cold.total > 0, "the named experts span pages");
    assert_eq!(
        cold.resident, 0,
        "after an eviction no page of the named experts is in the page cache ({} of {})",
        cold.resident, cold.total
    );
    assert_eq!(table.resident_rows(&ids).unwrap(), 0);

    table.prefetch(&ids).unwrap();
    table.populate(&ids).unwrap();
    let warm = table.resident_pages(&ids).unwrap();
    assert_eq!(warm.total, cold.total, "the same pages are counted");
    assert_eq!(
        warm.resident, warm.total,
        "after a populate every page of the named experts is resident"
    );
    assert_eq!(table.resident_rows(&ids).unwrap(), ids.len() as u64);

    table.evict_rows(&ids).unwrap();
    assert_eq!(
        table.resident_pages(&ids).unwrap().resident,
        0,
        "the experts just read are dropped again"
    );
    assert!(
        matches!(
            table.evict_rows(&[EXPERTS as u32]),
            Err(EngramError::RowOutOfRange { .. })
        ),
        "an eviction past the stack is refused by name"
    );
}

/// A direct handle reads the file's bytes at an aligned offset into an aligned
/// buffer, carries the `O_DIRECT` flag, and a path that is not there is
/// refused by name.
#[test]
#[ignore = "hw: the box's disk-backed target directory"]
fn hw_engram_open_direct_reads_the_files_bytes() {
    use std::os::fd::AsRawFd;
    let scratch = Scratch::new("direct");
    let (path, _) = write_synthetic(&scratch.0);
    let direct = engram::open_direct(&path).expect("a file on a disk opens for direct reads");

    let fdinfo =
        std::fs::read_to_string(format!("/proc/self/fdinfo/{}", direct.as_raw_fd())).unwrap();
    let flags = fdinfo
        .lines()
        .find_map(|l| l.strip_prefix("flags:"))
        .map(|v| u32::from_str_radix(v.trim(), 8).unwrap())
        .expect("fdinfo names the open flags");
    assert_ne!(
        flags & 0o40000,
        0,
        "the handle carries O_DIRECT: flags {flags:o}"
    );

    let align = engram::DIRECT_ALIGN;
    let mut backing = vec![0u8; 2 * align];
    let off = backing.as_ptr().align_offset(align);
    let buf = &mut backing[off..off + align];
    direct.read_exact_at(buf, 0).unwrap();
    let mut want = vec![0u8; align];
    File::open(&path)
        .unwrap()
        .read_exact_at(&mut want, 0)
        .unwrap();
    assert_eq!(
        &buf[..],
        &want[..],
        "an aligned direct read returns the file's first block"
    );

    let missing = scratch.0.join("absent.gguf");
    assert!(
        matches!(engram::open_direct(&missing), Err(EngramError::OpenDirect { path, .. }) if path == missing),
        "a path that is not there is refused by name, carrying the path"
    );
}
