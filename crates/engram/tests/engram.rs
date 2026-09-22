//! Gates for the engram crate. Both are about **what the mapping serves**, not
//! about how fast it serves it — speed belongs to `engram-rate` under a lease.
//!
//! `hw_` prefix: these need the box and the V4.1 split. They do not need the
//! oracle — the reference here is the file itself, read a second way.

use std::fs::File;
use std::os::unix::fs::FileExt;

use engram::{Engram, SeededRows, Site};

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
