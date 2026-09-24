//! The oracle set of `tools/ref/vision/dump-vision.sh`, read for the gates.

use std::path::{Path, PathBuf};

/// The revision every set must name; a set of another is stale.
pub const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

/// `$BLOOMERY_DATA/ref-vision/<set>`, the set `BLOOMERY_VISION_SET` names (default `deepseek41v`).
pub fn set_dir() -> PathBuf {
    let data = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
    let set = std::env::var("BLOOMERY_VISION_SET").unwrap_or_else(|_| "deepseek41v".into());
    Path::new(&data).join("ref-vision").join(set)
}

/// The committed test image `name`.
pub fn image_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/ref/vision/images")
        .join(name)
}

/// One `image` row of the manifest.
#[derive(Debug)]
pub struct ImageRow {
    pub name: String,
    pub sha256: String,
    pub w: usize,
    pub h: usize,
    pub best_w: usize,
    pub best_h: usize,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    pub n_tokens: usize,
}

impl ImageRow {
    pub fn stem(&self) -> &str {
        self.name.strip_suffix(".png").unwrap_or(&self.name)
    }
}

/// One row of `plans.tsv`: an input size, the reference's plan and its pad geometry.
#[derive(Debug)]
pub struct PlanRow {
    pub kind: String,
    pub w: usize,
    pub h: usize,
    /// n_llm_h, n_llm_w, best_h, best_w, n_tokens, resized_w, resized_h, off_x, off_y.
    pub want: [usize; 9],
}

pub struct Manifest {
    pub dir: PathBuf,
    pub mmproj: PathBuf,
    pub mmproj_sha256: String,
    pub image_token_id: u32,
    pub images: Vec<ImageRow>,
}

fn nums<const N: usize>(fields: &[&str], line: &str) -> [usize; N] {
    let v: Vec<usize> = fields
        .iter()
        .map(|f| {
            f.parse()
                .unwrap_or_else(|_| panic!("not a count: {f:?} in {line:?}"))
        })
        .collect();
    v.try_into()
        .unwrap_or_else(|_| panic!("{N} counts expected in {line:?}"))
}

impl Manifest {
    /// Read `MANIFEST.tsv`, refusing a set without its trailer or of another revision.
    pub fn load() -> Manifest {
        let dir = set_dir();
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv"))
            .unwrap_or_else(|e| panic!("{}: {e} — run: just dump-ref-vision", dir.display()));
        assert!(
            text.lines().any(|l| l.starts_with("# complete\t")),
            "{}: no # complete trailer",
            dir.display()
        );
        let checkpoint = text
            .lines()
            .find(|l| l.starts_with("# checkpoint\t"))
            .expect("# checkpoint line");
        assert!(
            checkpoint.contains(REVISION),
            "stale set, {checkpoint:?} is not revision {REVISION}"
        );
        let (mut mmproj, mut sha, mut id, mut images) = (None, None, None, Vec::new());
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# mmproj" => {
                    mmproj = Some(PathBuf::from(f[1]));
                    sha = Some(f[3].to_string());
                }
                "# image_token_id" => id = Some(f[1].parse().expect("image_token_id")),
                "image" => {
                    let n: [usize; 9] = nums(&f[3..12], line);
                    images.push(ImageRow {
                        name: f[1].to_string(),
                        sha256: f[2].to_string(),
                        w: n[0],
                        h: n[1],
                        best_w: n[2],
                        best_h: n[3],
                        n_vit_h: n[4],
                        n_vit_w: n[5],
                        n_llm_h: n[6],
                        n_llm_w: n[7],
                        n_tokens: n[8],
                    });
                }
                _ => {}
            }
        }
        assert!(!images.is_empty(), "{}: no image rows", dir.display());
        Manifest {
            dir,
            mmproj: mmproj.expect("# mmproj line"),
            mmproj_sha256: sha.expect("# mmproj sha256"),
            image_token_id: id.expect("# image_token_id line"),
            images,
        }
    }

    /// One file of the set.
    pub fn read(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.dir.join(name))
            .unwrap_or_else(|e| panic!("{}/{name}: {e}", self.dir.display()))
    }

    /// `plans.tsv`.
    pub fn plans(&self) -> Vec<PlanRow> {
        let text = String::from_utf8(self.read("plans.tsv")).expect("plans.tsv is text");
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .map(|line| {
                let f: Vec<&str> = line.split('\t').collect();
                let [w, h]: [usize; 2] = nums(&f[1..3], line);
                PlanRow {
                    kind: f[0].to_string(),
                    w,
                    h,
                    want: nums(&f[3..12], line),
                }
            })
            .collect()
    }
}

/// Little-endian i32s.
pub fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// Little-endian u16s.
pub fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

/// SHA-256 of a file, hex (FIPS 180-4), to tell a set from the images and the mmproj it names.
pub fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    h.finish()
}

struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    fill: usize,
    len: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            fill: 0,
            len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.fill).min(data.len());
            self.block[self.fill..self.fill + take].copy_from_slice(&data[..take]);
            self.fill += take;
            data = &data[take..];
            if self.fill == 64 {
                let b = self.block;
                self.compress(&b);
                self.fill = 0;
            }
        }
    }

    fn compress(&mut self, b: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, c) in b.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*c);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut bb, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (k, wi) in K.iter().zip(w) {
            let t1 = h
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ (!e & g))
                .wrapping_add(*k)
                .wrapping_add(wi);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & bb) ^ (a & c) ^ (bb & c));
            (h, g, f, e, d, c, bb, a) =
                (g, f, e, d.wrapping_add(t1), c, bb, a, t1.wrapping_add(t2));
        }
        for (s, v) in self.state.iter_mut().zip([a, bb, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    fn finish(mut self) -> String {
        let bits = self.len * 8;
        self.update(&[0x80]);
        while self.fill != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state.iter().map(|s| format!("{s:08x}")).collect()
    }
}

#[test]
fn sha256_known_answer() {
    let dir = std::env::temp_dir().join(format!("bloomery-vision-sha-{}", std::process::id()));
    std::fs::write(&dir, b"abc").expect("write");
    let got = sha256_file(&dir);
    std::fs::remove_file(&dir).ok();
    assert_eq!(
        got,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
