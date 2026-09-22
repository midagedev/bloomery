//! Gates for the engram crate: what the mapping serves, and which rows the
//! hash asks it for. Not how fast — speed belongs to `engram-rate` under a
//! lease. Nothing here reads a row the identity gates did not name, so the
//! whole file costs a few hundred page faults and not a few hundred thousand.
//!
//! `hw_` prefix: these need the box and the V4.1 split. They do not need the
//! oracle — the reference here is the file itself, read a second way.

use std::fs::File;
use std::os::unix::fs::FileExt;

use engram::{Context, Engram, SeededRows, Site};

/// The split set the gates open: `$BLOOMERY_V41_DIR`, or the box's default.
fn model_dir() -> String {
    std::env::var("BLOOMERY_V41_DIR")
        .unwrap_or_else(|_| "/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8".into())
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

/// Ids worth reading: the first row, the last row, one row that starts close
/// enough to the end of a page that it straddles into the next, and a seeded
/// spread over the table.
fn sample(site: &Site, n: usize) -> Vec<u32> {
    const PAGE: u64 = 4096;
    let mut ids = vec![0u32, (site.rows() - 1) as u32];

    // Row starts land on a lattice: gcd(272, 4096) = 16, so the residues repeat
    // with period 4096/16 = 256 in the id. One of them straddles.
    let straddler = (0..256u32)
        .find(|&r| site.file_offset(r) % PAGE + site.row_bytes() > PAGE)
        .unwrap_or_else(|| {
            panic!(
                "{}: no row in the first 256 straddles a page — the row stride is \
                 not the 272 B lattice this gate assumes",
                site.name()
            )
        });
    ids.push(straddler);

    let mut rng = SeededRows::new(0xB3E7_A001);
    let mut drawn = Vec::new();
    rng.next_into(site.rows(), n, &mut drawn);
    ids.extend(drawn);
    ids
}

/// The mapping serves the file's bytes, prefetched or not.
///
/// This is the whole claim of the IO half: a row reached through the mmap is
/// the same bytes a plain `pread` returns **at the offset the GGUF header
/// gives** — `data_base + tensor.offset + id x 272`, recomputed here from the
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
        // own offset, and a stride of ne[0]/32 Q8_0 blocks of 34 B.
        let inv = gguf::inventory_of(site.path()).unwrap();
        let t = inv
            .tensors
            .iter()
            .find(|t| t.name == site.name())
            .expect("the site came from this shard's header");
        let base = inv.data_base + t.offset;
        let stride = t.dims[0] / 32 * 34;

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
#[test]
#[ignore = "hw: needs the box and the V4.1 split"]
fn hw_engram_copy_rows_matches_row() {
    let engram = open();

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
        let nbytes = t.nbytes.expect("Q8_0 is in ggml's size table");

        assert_eq!(t.dims.len(), 2, "{}: engram is a 2-D table", site.name());
        assert_eq!(site.rows(), t.dims[1], "{}: row count", site.name());
        // 256 values a row, Q8_0 = 32 values in 34 B.
        assert_eq!(
            site.row_bytes(),
            t.dims[0] / 32 * 34,
            "{}: row stride from the Q8_0 block",
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
/// The contexts cover the three shapes that differ: ids inside the map, ids
/// past its end (which must map to pad, not panic or index), and a window fresh
/// from `reset()`, where every older slot is pad — the sequence start the port
/// spends one token in.
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
    let mut rng = SeededRows::new(0xB3D0_1CE5);
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

        // 10,000 contexts: two thirds inside the vocabulary, one third past its
        // end, and every hundredth window starting fresh from a reset.
        let mut ctx = Context::new(hash);
        let mut draw = Vec::new();
        let mut want = vec![0u32; n_cols];
        for n in 0..10_000u32 {
            let fresh = n % 100 == 0;
            if fresh {
                ctx.reset();
            }
            let previous = ctx.window()[0];
            rng.next_into(u64::from(vocab) * 3 / 2, 1, &mut draw);
            ctx.push(draw[0]);

            // The window's slot order is what `reference_rows` cannot check:
            // it is handed the window and would agree with a shift the wrong
            // way. Slot 0 is the token just pushed, slot 1 the one before it,
            // and after a reset every older slot is pad.
            let window = ctx.window();
            assert_eq!(
                window[0],
                hash.map_token(draw[0]),
                "context {n}: slot 0 must be the token just pushed, mapped"
            );
            if fresh {
                assert!(
                    window[1..].iter().all(|&v| v == hash.pad_id()),
                    "context {n}: after a reset every older slot must be pad, \
                     the value the port substitutes before a sequence starts"
                );
            } else {
                assert_eq!(
                    window[1], previous,
                    "context {n}: slot 1 must be what slot 0 held before the push"
                );
            }
            hash.rows_into(e, ctx.window(), &mut ids).unwrap();
            reference_rows(hash, e, ctx.window(), &mut want);
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
/// compression that lost a slot, and `map_token` for an id past the domain
/// would then be indistinguishable from a real value. The pad the port
/// substitutes there must itself be inside the codomain, or a sequence start
/// would address a row no trained embedding sits in.
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
