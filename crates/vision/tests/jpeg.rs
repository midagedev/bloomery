//! The JPEG decoder against the reference's (Pillow, `Image.open(f).convert("RGB")`) on the fixtures
//! of `tools/ref/vision/dump-jpeg.py` (`tests/fixtures/jpeg/`, `just dump-ref-jpeg`), and the JPEG
//! files it refuses. `just gate-vision` runs it on the box; it needs nothing but the tree.
//!
//! * pixels: every fixture's RGB within its pinned frontier of the reference's, and the YCbCr before
//!   the colour conversion printed beside it, so a moved count names the term that moved;
//! * progressive: a progressive fixture decodes to the pixels of its baseline twin;
//! * refusals: CMYK and RGB-coded files by name, a file cut anywhere before its EOI, a scan with a
//!   code no Huffman table holds.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use vision::{Rgb8, VisionError};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// PIN(2026-10-05): zune-jpeg 0.5.15 against Pillow 10.2.0 (libjpeg-turbo 2.1.5) on the fixtures, measured on the box (AVX2) and the Mac (NEON).
///
/// Per fixture: the RGB pixels with any channel apart from the reference's, and the largest
/// channel distance. Three terms make the difference, each derived from the two decoders' code:
/// * the IDCT: zune-jpeg's is stb_image's, with 12-bit constants and its own descaling, where
///   libjpeg's ISLOW has 13-bit constants; the two put some samples one apart, and only the
///   measurement says how many (the greyscale fixture is this term alone);
/// * chroma upsampling (4:2:0 and 4:2:2): libjpeg-turbo's fancy upsampling keeps the vertical
///   sum unrounded and rounds once with biases 8 and 7 (h2v2), or 1 and 2 (h2v1); zune-jpeg
///   rounds after each pass with bias 2. On uniformly random chroma the two are one apart on
///   26 % of h2v2 and 13 % of h2v1 samples, and their rounding errors keep them within one;
/// * colour conversion: libjpeg-turbo rounds `1.402·Cr` and `1.772·Cb` to the nearest integer
///   with 16-bit tables, zune-jpeg sums 14-bit products with a bias of 0.49994; over every
///   (Y, Cb, Cr) 0.28 % of triples are one apart in G or B, never in R.
///
/// A chroma sample two apart (IDCT plus upsampling) moves B by about 3.5, so a 4:2:0 fixture's
/// largest distance can reach 4 or 5. A count or a distance above its pin is a decoder change.
const PINNED: &[(&str, usize, u8)] = &[
    ("noise-q75-420", 1131, 3),
    ("noise-q75-420-prog", 1131, 3),
    ("noise-q95-444", 64, 2),
    ("noise-q50-422", 531, 2),
    ("checker-q75-420", 461, 3),
    ("checker-q90-444-prog", 31, 2),
    ("grad-q90-420", 250, 2),
    ("mixed-q85-420", 838, 4),
    ("noise-q85-grey", 9, 1),
    ("tiny-1x1-420", 0, 0),
    ("tiny-3x2-420", 2, 1),
    ("tiny-17x9-420", 72, 2),
];

/// The fixtures, named from the tree's root so `tools/recipes.py` keys the gate on them.
fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/vision/tests/fixtures/jpeg")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fixture file, refused unless it is the one the manifest names.
fn read(name: &str, sha256: &str) -> Vec<u8> {
    let path = dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        hex(&Sha256::digest(&bytes)),
        sha256,
        "{}: not the file the manifest names",
        path.display()
    );
    bytes
}

/// One `jpeg` row of the fixtures' manifest.
struct Fixture {
    name: String,
    kind: String,
    w: usize,
    h: usize,
    jpg: Vec<u8>,
    rgb: Vec<u8>,
    /// libjpeg's output before the colour conversion; 3-component files only.
    ycc: Option<Vec<u8>>,
}

struct Manifest {
    fixtures: Vec<Fixture>,
    /// `(file, what it is, its bytes)`.
    refuse: Vec<(String, String, Vec<u8>)>,
    /// `(progressive, baseline)`.
    twins: Vec<(String, String)>,
}

/// `MANIFEST.tsv` of `tools/ref/vision/dump-jpeg.py`, each row read by its kind's column names.
fn manifest() -> Manifest {
    let path = dir().join("MANIFEST.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        text.lines().any(|l| l.starts_with("# complete")),
        "{}: no completion trailer",
        path.display()
    );
    let mut columns: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut m = Manifest {
        fixtures: Vec::new(),
        refuse: Vec::new(),
        twins: Vec::new(),
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# ") {
            if let Some((kind, names)) = rest.split_once(" columns\t") {
                columns.insert(kind, names.split(' ').collect());
            }
            continue;
        }
        let (kind, values) = line.split_once('\t').expect("a typed row");
        let names = columns
            .get(kind)
            .unwrap_or_else(|| panic!("{}: a {kind} row before its columns", path.display()));
        let values: Vec<&str> = values.split('\t').collect();
        assert_eq!(values.len(), names.len(), "{}: {line}", path.display());
        let row: HashMap<&str, &str> = names.iter().copied().zip(values).collect();
        match kind {
            "jpeg" => {
                let name = row["name"];
                let ycc = (row["ycc_sha256"] != "-")
                    .then(|| read(&format!("{name}.ycc"), row["ycc_sha256"]));
                m.fixtures.push(Fixture {
                    name: name.to_owned(),
                    kind: row["kind"].to_owned(),
                    w: row["w"].parse().expect("w"),
                    h: row["h"].parse().expect("h"),
                    jpg: read(&format!("{name}.jpg"), row["jpg_sha256"]),
                    rgb: read(&format!("{name}.rgb"), row["rgb_sha256"]),
                    ycc,
                });
            }
            "refuse" => m.refuse.push((
                row["file"].to_owned(),
                row["what"].to_owned(),
                read(row["file"], row["sha256"]),
            )),
            "twin" => m
                .twins
                .push((row["progressive"].to_owned(), row["baseline"].to_owned())),
            other => panic!("{}: a {other} row", path.display()),
        }
    }
    m
}

/// How two images of one size differ: the pixels with any channel apart, the largest channel
/// distance, and the channel values at distance 1, 2, 3 and 4 or more.
#[derive(Debug, Default)]
struct Diff {
    pixels: usize,
    max: u8,
    at: [usize; 4],
}

fn diff(ours: &[u8], theirs: &[u8]) -> Diff {
    assert_eq!(ours.len(), theirs.len(), "image sizes");
    let mut d = Diff::default();
    for (a, b) in ours
        .as_chunks::<3>()
        .0
        .iter()
        .zip(theirs.as_chunks::<3>().0)
    {
        let mut apart = false;
        for (x, y) in a.iter().zip(b) {
            let dist = x.abs_diff(*y);
            if dist > 0 {
                apart = true;
                d.max = d.max.max(dist);
                d.at[usize::from(dist.min(4)) - 1] += 1;
            }
        }
        d.pixels += usize::from(apart);
    }
    d
}

/// zune-jpeg's output before its colour conversion, the counterpart of the fixture's `.ycc`.
fn ycc(jpg: &[u8]) -> Vec<u8> {
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(ColorSpace::YCbCr);
    JpegDecoder::new_with_options(ZCursor::new(jpg), options)
        .decode()
        .expect("YCbCr decode")
}

#[test]
fn jpeg_matches_pillow_within_the_pinned_frontier() {
    let m = manifest();
    let pins: HashMap<&str, (usize, u8)> = PINNED.iter().map(|&(n, p, d)| (n, (p, d))).collect();
    let mut wrong = Vec::new();
    println!(
        "fixture                kind                  size    rgb: differ  max  |d|=1  2  3  4+   ycc: differ  max   pin"
    );
    for f in &m.fixtures {
        let ours = Rgb8::from_jpeg(&f.jpg).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert_eq!((ours.width, ours.height), (f.w, f.h), "{}", f.name);
        let d = diff(&ours.data, &f.rgb);
        let y = f.ycc.as_ref().map(|want| diff(&ycc(&f.jpg), want));
        let pin = pins.get(f.name.as_str()).copied();
        println!(
            "{:<22} {:<20} {:>3}x{:<3} {:>12} {:>4}  {:>5} {:>2} {:>2} {:>2}   {:>11} {:>4}   {}",
            f.name,
            f.kind,
            f.w,
            f.h,
            d.pixels,
            d.max,
            d.at[0],
            d.at[1],
            d.at[2],
            d.at[3],
            y.as_ref().map_or("-".into(), |y| y.pixels.to_string()),
            y.as_ref().map_or("-".into(), |y| y.max.to_string()),
            pin.map_or("none".into(), |(p, x)| format!("{p} px, max {x}"))
        );
        match pin {
            Some((p, x)) if d.pixels <= p && d.max <= x => {}
            Some((p, x)) => wrong.push(format!(
                "{}: {} pixels apart (max {}), pinned {p} (max {x})",
                f.name, d.pixels, d.max
            )),
            None => wrong.push(format!(
                "{}: no pin ({} pixels apart, max {})",
                f.name, d.pixels, d.max
            )),
        }
    }
    for name in pins.keys() {
        if !m.fixtures.iter().any(|f| f.name == *name) {
            wrong.push(format!("{name}: pinned, but not a fixture"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A progressive file carries the same coefficients as its baseline twin, so it decodes to the
/// same pixels; this holds the progressive path without the reference.
#[test]
fn progressive_twin_decodes_like_its_baseline() {
    let m = manifest();
    assert!(!m.twins.is_empty(), "the manifest names no twin");
    let decoded = |name: &str| {
        let f = m.fixtures.iter().find(|f| f.name == name).expect(name);
        Rgb8::from_jpeg(&f.jpg).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    for (prog, base) in &m.twins {
        assert_eq!(decoded(prog), decoded(base), "{prog} against {base}");
    }
}

#[test]
fn jpeg_refusals_are_named() {
    let m = manifest();
    assert!(!m.refuse.is_empty(), "the manifest names no refusal file");
    for (file, what, bytes) in &m.refuse {
        match Rgb8::from_bytes(bytes) {
            Err(VisionError::Format(msg)) if msg.contains("a JPEG coded as") => {
                println!("{file} ({what}): {msg}");
            }
            other => panic!("{file} ({what}): {other:?}"),
        }
    }
    // Pillow refuses a file cut anywhere before the end of its EOI marker ("image file is
    // truncated"): in a header, in the scan data, or with only the EOI missing.
    let mut wrong = Vec::new();
    for f in &m.fixtures {
        let whole = &f.jpg;
        for cut in [
            whole.len() / 4,
            whole.len() / 2,
            whole.len() * 7 / 8,
            whole.len() - 3,
            whole.len() - 1,
        ] {
            match Rgb8::from_jpeg(&whole[..cut]) {
                Err(VisionError::Decode(msg)) => println!("{} cut at {cut}: {msg}", f.name),
                Err(e) => wrong.push(format!("{} cut at {cut}: {e}", f.name)),
                Ok(_) => wrong.push(format!(
                    "{} cut at {cut} of {}: decoded",
                    f.name,
                    whole.len()
                )),
            }
        }
        // Scan bytes of all ones (stuffed `FF 00`) from the middle on: no Huffman table has
        // the all-ones code. Pillow returns an image whose blocks from there on are not the
        // file's (libjpeg only warns); here it is an error.
        let mut corrupt = whole.clone();
        let mid = whole.len() * 7 / 8;
        for (i, b) in corrupt[mid..whole.len() - 2].iter_mut().enumerate() {
            *b = if i % 2 == 0 { 0xFF } else { 0x00 };
        }
        match Rgb8::from_jpeg(&corrupt) {
            Err(VisionError::Decode(msg)) => println!("{} corrupt from {mid}: {msg}", f.name),
            Err(e) => wrong.push(format!("{} corrupt from {mid}: {e}", f.name)),
            Ok(_) => wrong.push(format!("{} corrupt from {mid}: decoded", f.name)),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
