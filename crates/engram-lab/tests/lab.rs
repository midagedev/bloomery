//! Gates for the engram IO lab: the context window it walks a stream with,
//! and what the row cache in front of the engine's table serves. Not how fast
//! — speed belongs to `engram-rate` under a lease. The window gate reads
//! headers only; the cache's serving gate reads what a few thousand tokens of
//! a real stream ask for, and its exactness gate reads no row at all.
//!
//! `hw_` prefix: these need the box and the V4.1 split, and the cache gates
//! also the token streams `just engram-corpus` writes. They do not need the
//! oracle — the reference here is the file itself, read a second way.

use std::sync::Arc;

use engram::prefetch::{FillMode, Prefetcher};
use engram::{Engram, EngramError, Hash};
use engram_lab::cache::{Access, LruIndex, RowCache, capacity_rows, key_of};
use engram_lab::reuse::{Lru, read_ids};
use engram_lab::{Context, LabError, SeededRows};

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
        ctx.push(token)
            .expect("a corpus id is a token of the model");
        for site in 0..hash.sites() {
            hash.rows_into(site, ctx.window(), &mut ids).unwrap();
            for (bucket, &id) in ids.iter().enumerate() {
                f(site, bucket, id);
            }
        }
    }
}

/// A window holds the last `n_gram` mapped tokens newest first: slot 0 is the
/// token just pushed, mapped, and every older slot is what the slot before it
/// held before the push; after a reset every older slot is pad, the value the
/// port substitutes before a sequence starts. A token past the map is refused
/// by name and the window stays as it was.
///
/// The hash's own gate hashes windows it builds itself, so it cannot see a
/// shift the wrong way here; this is where the window's slot order is held.
/// A third of the ids drawn are past the map's end.
#[test]
#[ignore = "hw: needs the box and the V4.1 split's headers"]
fn hw_engram_context_window_slots() {
    let dir = model_dir();
    let hash = Hash::from_dir(&dir).unwrap_or_else(|e| {
        panic!(
            "no engram metadata under {dir} ({e}). The V4.1 split lives on the box; \
             set BLOOMERY_V41_DIR if it moved. Do not skip this test."
        )
    });
    let vocab = hash.token_map().len() as u32;
    let mut rng = SeededRows::new(0xB3D0_1CE5);
    let mut ctx = Context::new(&hash);
    let mut draw = Vec::new();
    let (mut pushed, mut refused) = (0u32, 0u32);
    for n in 0..10_000u32 {
        let fresh = n % 100 == 0;
        if fresh {
            ctx.reset();
        }
        let before = ctx.window().to_vec();
        rng.next_into(u64::from(vocab) * 3 / 2, 1, &mut draw);
        let token = draw[0];
        match ctx.push(token) {
            Ok(()) if token < vocab => pushed += 1,
            Err(EngramError::TokenPastMap { token: t, map })
                if t == token && token >= vocab && map == vocab as usize =>
            {
                assert_eq!(
                    ctx.window(),
                    &before[..],
                    "context {n}: the refused token {token} moved the window"
                );
                refused += 1;
                continue;
            }
            other => {
                panic!("context {n}: token {token} against a map of {vocab} entries gave {other:?}")
            }
        }
        let window = ctx.window();
        assert_eq!(
            window[0],
            hash.map_token(token).unwrap(),
            "context {n}: slot 0 must be the token just pushed, mapped"
        );
        assert_eq!(
            &window[1..],
            &before[..before.len() - 1],
            "context {n}: every older slot must be what the slot before it held before the push"
        );
        if fresh {
            assert!(
                window[1..].iter().all(|&v| v == hash.pad_id()),
                "context {n}: after a reset every older slot must be pad, \
                 the value the port substitutes before a sequence starts"
            );
        }
    }
    println!(
        "context-window: {pushed} tokens pushed, {refused} past the {vocab}-entry map refused"
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
        ctx.push(token)
            .expect("a corpus id is a token of the model");
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
        matches!(refused, Err(LabError::Cache(_))),
        "a lookup before the last token's misses were completed must be refused"
    );
    // The second name of a row would find the first one's slot unfilled.
    let mut twice = ids.clone();
    twice[0][1] = twice[0][0];
    let mut doubled = RowCache::new(CAPACITY, rb, &rows_per_site).unwrap();
    let refused = doubled.lookup(&twice, &mut out);
    println!("serves-table-bytes: a token naming one row twice -> {refused:?}");
    assert!(
        matches!(refused, Err(LabError::Cache(_))),
        "a hit on a pending slot must be refused, not served"
    );
}

/// The public V4.1 set in llama.cpp's layout: one table of every site's rows
/// (`per_layer_token_embd.weight`) and the hash under the `ple.*` keys.
const UPSTREAM_DIR: &str = "/models/DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS";

/// The model's two layouts hash alike and place each site's rows: the public
/// set in llama.cpp's layout (one table, its bucket offsets counted from the
/// table's row 0) and the gates' per-site set give every bucket of every site
/// the same row id for each of corpus-code's first 3000 tokens; their sites
/// hold as many rows; and in the one table each site's last row is readable,
/// the one past it is refused, and the next site's first row starts at the
/// byte after it. Headers and those rows only: no model is opened.
#[test]
#[ignore = "hw: needs the box, both V4.1 splits and the token streams"]
fn hw_engram_both_layouts_hash_alike() {
    const TOKENS: usize = 3000;
    let upstream = Engram::open_dir(UPSTREAM_DIR).unwrap_or_else(|e| {
        panic!("no engram table under {UPSTREAM_DIR} ({e}). Do not skip this test.")
    });
    let site_wise = open();
    let (one, each) = (upstream.sites(), site_wise.sites());
    assert_eq!(one.len(), each.len(), "the two sets' Engram sites");
    for (s, (site, twin)) in one.iter().zip(each).enumerate() {
        assert_eq!(site.rows(), twin.rows(), "site {s}: its rows in both sets");
        assert_eq!(
            site.name(),
            one[0].name(),
            "site {s}: the public set's sites share one table"
        );
        let last = u32::try_from(site.rows() - 1).expect("a site's row ids are u32");
        assert!(site.row(last).is_ok(), "site {s}: its last row");
        assert!(
            matches!(site.row(last + 1), Err(EngramError::RowOutOfRange { .. })),
            "site {s}: the row past its last is refused"
        );
        if let Some(next) = one.get(s + 1) {
            assert_eq!(
                site.file_offset(last) + site.row_bytes(),
                next.file_offset(0),
                "site {s} ends where site {} begins",
                s + 1
            );
        }
    }
    let stream = corpus("code");
    let tokens = stream.get(..TOKENS).unwrap_or_else(|| {
        panic!(
            "corpus-code has {} tokens and this gate reads the first {TOKENS}; \
             rerun `just engram-corpus`",
            stream.len()
        )
    });
    let mut ids = Vec::new();
    walk(upstream.hash(), tokens, |site, bucket, id| {
        ids.push((site, bucket, id));
    });
    let mut at = 0;
    walk(site_wise.hash(), tokens, |site, bucket, id| {
        assert_eq!(
            ids.get(at),
            Some(&(site, bucket, id)),
            "walk entry {at}: the two layouts hash this bucket to other rows"
        );
        at += 1;
    });
    assert_eq!(at, ids.len(), "both walks visit every bucket once");
}

/// The V4.1 fixture and its upstream twin (`just fixture-engram-upstream`;
/// `tools/ref/engram-relayout.py twin` names the twin's directory).
const FIXTURE_DIR: &str = "/models/fixtures/v41";
const TWIN_DIR: &str = "/models/fixture-variants/engram-upstream/v41";

/// The fixture and its upstream twin read the same rows: for each of
/// corpus-prose's first 3000 tokens, every bucket of every site hashes to
/// the same row id in both, and that row decodes (`gguf::quant::dequant_row`)
/// to the same f32 bits in both — the twin's Q6_K rows are the fixture's
/// Q3_K rows transcoded exactly, each site's run at its base row of the one
/// table. Headers and those rows only: no model is opened.
#[test]
#[ignore = "hw: needs the box, the V4.1 fixture, its upstream twin and the token streams"]
fn hw_engram_fixture_and_twin_read_the_same_rows() {
    const TOKENS: usize = 3000;
    let open_at = |dir: &str| {
        Engram::open_dir(dir).unwrap_or_else(|e| {
            panic!(
                "no engram table under {dir} ({e}); `just fixture-engram-upstream` writes the \
                 twin. Do not skip this test."
            )
        })
    };
    let (per_site, twin) = (open_at(FIXTURE_DIR), open_at(TWIN_DIR));
    let (each, one) = (per_site.sites(), twin.sites());
    assert_eq!(each.len(), one.len(), "the two sets' Engram sites");
    for (s, (a, b)) in each.iter().zip(one).enumerate() {
        assert_eq!(a.rows(), b.rows(), "site {s}: its rows in both sets");
        assert_eq!(
            b.name(),
            one[0].name(),
            "site {s}: the twin's sites share one table"
        );
    }
    let ty = |site: &engram::Site| gguf::GgmlType::from_u32(site.type_id());
    let width = |site: &engram::Site| {
        let t = ty(site);
        let (Some(block), Some(bytes)) = (t.blck_size(), t.type_size()) else {
            panic!("{}: {t} is not sized", site.name())
        };
        usize::try_from(site.row_bytes() / bytes * block).expect("a row's width is a usize")
    };
    let decode = |site: &engram::Site, id: u32, out: &mut [f32]| {
        let bytes = site
            .row(id)
            .unwrap_or_else(|e| panic!("{} row {id}: {e}", site.name()));
        gguf::quant::dequant_row(ty(site), bytes, out)
            .unwrap_or_else(|e| panic!("{} row {id}: {e}", site.name()));
    };
    let stream = corpus("prose");
    let tokens = stream.get(..TOKENS).unwrap_or_else(|| {
        panic!(
            "corpus-prose has {} tokens and this gate reads the first {TOKENS}; \
             rerun `just engram-corpus`",
            stream.len()
        )
    });
    let mut ids = Vec::new();
    walk(twin.hash(), tokens, |site, bucket, id| {
        ids.push((site, bucket, id));
    });
    let mut at = 0;
    walk(per_site.hash(), tokens, |site, bucket, id| {
        assert_eq!(
            ids.get(at),
            Some(&(site, bucket, id)),
            "walk entry {at}: the two layouts hash this bucket to other rows"
        );
        let n = width(&each[site]);
        assert_eq!(
            n,
            width(&one[site]),
            "site {site}: its row width in both sets"
        );
        let (mut x, mut y) = (vec![0f32; n], vec![0f32; n]);
        decode(&each[site], id, &mut x);
        decode(&one[site], id, &mut y);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert!(
            bits(&x) == bits(&y),
            "site {site} row {id}: the twin's {} row does not decode to the fixture's {} row",
            ty(&one[site]),
            ty(&each[site])
        );
        at += 1;
    });
    assert_eq!(at, ids.len(), "both walks visit every bucket once");
}
