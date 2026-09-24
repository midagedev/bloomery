//! Gates for the engram crate: what the mapping serves, which rows the hash
//! asks it for, and what the row cache in front of it serves. Not how fast —
//! speed belongs to `engram-rate` under a lease. The identity gates read only
//! the rows they name, a few hundred page faults; the cache's serving gate
//! reads what a few thousand tokens of a real stream ask for, and its
//! exactness gate reads no row at all.
//!
//! `hw_` prefix: these need the box and the V4.1 split, and the cache gates
//! also the token streams `just engram-corpus` writes. They do not need the
//! oracle — the reference here is the file itself, read a second way.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use engram::cache::{Access, LruIndex, RowCache, capacity_rows, key_of};
use engram::prefetch::{FillMode, Prefetcher};
use engram::reuse::{Lru, read_ids};
use engram::{Context, Engram, EngramError, Hash, SeededRows, Site};

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

/// `$BLOOMERY_DATA/engram/corpus-<name>.ids`: a real token stream.
fn corpus(name: &str) -> Vec<u32> {
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let path = format!("{data}/engram/corpus-{name}.ids");
    read_ids(&path).unwrap_or_else(|e| {
        panic!(
            "{e}. The token streams are written on the box by `just engram-corpus`; \
             run it. Do not skip this test."
        )
    })
}

/// Walk a token stream the way the cache is fed: per token, sites in order,
/// and within a site the buckets in `Hash::rows_into` order — the order
/// `engram-reuse` feeds its simulator. `f` gets `(site, bucket, row id)`.
fn walk(hash: &Hash, tokens: &[u32], mut f: impl FnMut(usize, usize, u32)) {
    let mut ctx = Context::new(hash);
    let mut ids = vec![0u32; hash.n_cols()];
    for &token in tokens {
        ctx.push(token);
        for site in 0..hash.sites() {
            hash.rows_into(site, ctx.window(), &mut ids).unwrap();
            for (bucket, &id) in ids.iter().enumerate() {
                f(site, bucket, id);
            }
        }
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

/// The row cache's index is exact LRU: over a real stream, through the real
/// hash, its misses are the stack-distance simulator's to the integer at every
/// capacity, and once full it holds exactly its capacity.
///
/// Two algorithms that share nothing but the definition — a hash table and a
/// linked list claiming and evicting slots one access at a time, and a Fenwick
/// tree counting the distinct rows between two touches of the same one — so an
/// eviction from the wrong end, or a probe that loses a key, shows up as a
/// count that differs. The index runs without a payload, which is what lets the
/// largest budget run here at all.
///
/// The capacities are chosen for what can hide. The budgets `engram-reuse`
/// reports are here, but at 4 and 16 GiB neither stream fills the cache and the
/// prose stream does not fill even 1 GiB, so two budgets below both streams'
/// distinct rows make eviction happen on both. A capacity one row off changes
/// the misses only through accesses at stack distance exactly `C - 1`, and deep
/// in a stream those are sparse — the heads of one order re-hit together, at
/// one shared distance — so the budgets alone can agree with a cache a row
/// short. The two small capacities sit where nearly every distance is taken:
/// one token's rows, where a row the next token asks again is at distance one
/// token minus one, and a few thousand. The count of keys held is the same
/// check without the stream's help.
#[test]
#[ignore = "hw: needs the box, the V4.1 split's headers and the token streams"]
fn hw_engram_cache_is_exact_lru() {
    const MIB: u64 = 1 << 20;
    let engram = open();
    let hash = engram.hash();
    let row_bytes = engram.sites()[0].row_bytes();
    let token = (hash.sites() * hash.n_cols()) as u64;
    let mut caps: Vec<(String, u64)> = [token, 4096]
        .into_iter()
        .map(|rows| (format!("{} B", rows * row_bytes), rows))
        .collect();
    for mib in [64, 256, 1024, 4096, 16384] {
        caps.push((format!("{mib} MiB"), capacity_rows(mib * MIB, row_bytes)));
    }
    let rows: Vec<u64> = caps.iter().map(|&(_, rows)| rows).collect();

    let mut disagree = Vec::new();
    for name in ["code", "prose"] {
        let tokens = corpus(name);
        let mut lru = Lru::new(tokens.len() * hash.sites() * hash.n_cols(), &rows);
        walk(hash, &tokens, |site, bucket, id| {
            lru.access(key_of(site, id), site, hash.order_of_bucket(bucket) - 2)
                .unwrap();
        });
        let total = lru.total().clone();
        let distinct = lru.distinct() as u64;
        drop(lru);

        for (i, (label, cap)) in caps.iter().enumerate() {
            let mut index = LruIndex::new(usize::try_from(*cap).unwrap()).unwrap();
            let allocated = index.allocated_bytes();
            let mut misses = 0u64;
            walk(hash, &tokens, |site, _, id| {
                if let Access::Miss(_) = index.access(key_of(site, id)) {
                    misses += 1;
                }
            });
            let want = total.requests - total.hits[i];
            let held = index.len() as u64;
            let want_held = distinct.min(*cap);
            let agree = misses == want && held == want_held;
            println!(
                "exact-lru {name} {cap} rows ({label}): stack distance {want} misses, \
                 linked list {misses}; holds {held} of {want_held} — {}",
                if agree { "agree" } else { "DISAGREE" }
            );
            if !agree {
                disagree.push(format!(
                    "{name} at {cap} rows: misses {want} vs {misses}, held {want_held} vs {held}"
                ));
            }
            assert_eq!(
                index.allocated_bytes(),
                allocated,
                "{name} at {cap} rows: the index allocated during the run"
            );
        }
    }
    assert!(
        disagree.is_empty(),
        "the linked-list LRU disagrees with the stack-distance simulator: {disagree:?}"
    );
}

/// Every row the cache serves — a hit out of its slab, or a miss the helper
/// filled — is the table's own bytes, and the cache hits exactly where the
/// stack-distance simulator says an LRU of its size would.
///
/// A small cache over a real stream evicts on nearly every miss, so its slots
/// are reused all the time: a hit copied from the wrong slot, a fill landed in
/// the wrong one, or an eviction that left a stale key behind serves some other
/// row's bytes, and the comparison against `Site::row` — the mapping read that
/// `hw_engram_rows_match_pread` ties to `pread` — sees it. The payloads go
/// through the real `Prefetcher`, misses only. And a row still pending is never
/// served: a lookup before the last token's misses were completed, and a token
/// that names one row twice, are both refused.
#[test]
#[ignore = "hw: needs the box, the V4.1 split and the token streams"]
fn hw_engram_cache_serves_table_bytes() {
    const TOKENS: usize = 3000;
    const CAPACITY: usize = 4096;
    let engram = Arc::new(open());
    let hash = engram.hash();
    let sites = engram.sites();
    let rb = sites[0].row_bytes() as usize;
    assert!(
        sites.iter().all(|s| s.row_bytes() as usize == rb),
        "the slab has one row stride for every site"
    );
    let n_cols = hash.n_cols();
    let rows_per_site = vec![n_cols; sites.len()];

    let stream = corpus("code");
    let tokens = stream.get(..TOKENS).unwrap_or_else(|| {
        panic!(
            "corpus-code has {} tokens and this gate reads the first {TOKENS}; \
             rerun `just engram-corpus`",
            stream.len()
        )
    });
    let mut cache = RowCache::new(CAPACITY, rb, &rows_per_site).unwrap();
    let mut pf = Prefetcher::new(Arc::clone(&engram), &rows_per_site, FillMode::Touch).unwrap();
    let mut lru = Lru::new(TOKENS * sites.len() * n_cols, &[CAPACITY as u64]);
    let allocated = cache.allocated_bytes();

    let mut out = vec![0u8; cache.token_bytes()];
    let mut ids = vec![vec![0u32; n_cols]; sites.len()];
    let mut ctx = Context::new(hash);
    let (mut hits, mut misses, mut no_trip) = (0u64, 0u64, 0u64);
    for (t, &token) in tokens.iter().enumerate() {
        ctx.push(token);
        for (site, site_ids) in ids.iter_mut().enumerate() {
            hash.rows_into(site, ctx.window(), site_ids).unwrap();
            for (bucket, &id) in site_ids.iter().enumerate() {
                lru.access(key_of(site, id), site, hash.order_of_bucket(bucket) - 2)
                    .unwrap();
            }
        }
        // A row the cache failed to write reads as zeros, not as the bytes the
        // same place held for the last token.
        out.fill(0);
        let got = cache.lookup(&ids, &mut out).unwrap();
        pf.submit(cache.miss_ids()).unwrap();
        pf.wait().unwrap();
        cache.complete(pf.filled(), &mut out).unwrap();
        hits += got.hits as u64;
        misses += got.misses as u64;
        no_trip += u64::from(got.misses == 0);

        for (site, site_ids) in ids.iter().enumerate() {
            for (bucket, &id) in site_ids.iter().enumerate() {
                let at = (site * n_cols + bucket) * rb;
                assert_eq!(
                    &out[at..at + rb],
                    sites[site].row(id).unwrap(),
                    "token {t}, site {site} bucket {bucket}, row {id}: the cache served \
                     bytes that are not the table's"
                );
            }
        }
    }
    let total = lru.total();
    let want = (total.hits[0], total.requests - total.hits[0]);
    println!(
        "serves-table-bytes: {TOKENS} tokens at {CAPACITY} rows, every row byte-identical \
         to Site::row; cache {hits} hits {misses} misses ({no_trip} tokens with no helper \
         round trip), stack distance {} hits {} misses",
        want.0, want.1
    );
    assert_eq!(
        (hits, misses),
        want,
        "the cache's hits and misses differ from an LRU of {CAPACITY} rows over the same accesses"
    );
    assert_eq!(
        cache.allocated_bytes(),
        allocated,
        "the cache allocated during the run"
    );

    // A fresh cache misses the whole token, so every row of it is pending.
    let mut early = RowCache::new(CAPACITY, rb, &rows_per_site).unwrap();
    early.lookup(&ids, &mut out).unwrap();
    let refused = early.lookup(&ids, &mut out);
    println!("serves-table-bytes: a lookup before complete -> {refused:?}");
    assert!(
        matches!(refused, Err(EngramError::Cache(_))),
        "a lookup before the last token's misses were completed must be refused"
    );
    // The second name of a row would find the first one's slot unfilled.
    let mut twice = ids.clone();
    twice[0][1] = twice[0][0];
    let mut doubled = RowCache::new(CAPACITY, rb, &rows_per_site).unwrap();
    let refused = doubled.lookup(&twice, &mut out);
    println!("serves-table-bytes: a token naming one row twice -> {refused:?}");
    assert!(
        matches!(refused, Err(EngramError::Cache(_))),
        "a hit on a pending slot must be refused, not served"
    );
}
