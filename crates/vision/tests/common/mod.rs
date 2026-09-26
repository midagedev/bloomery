//! The oracle set of `tools/ref/vision/dump-vision.sh`, read for the gates through the reference-set
//! reader (`refset::vision`), which refuses a set without its trailer or of another checkpoint
//! revision by name.

use refset::arch::deepseek41v::{VISION, VISION_SET};
use refset::vision::VisionSet;
use std::path::{Path, PathBuf};

/// One `image` row of the manifest.
pub use refset::vision::Image as ImageRow;

/// The vision family's set, or the one under `ref-vision` that `BLOOMERY_VISION_SET` names.
pub fn set_dir() -> PathBuf {
    match std::env::var("BLOOMERY_VISION_SET") {
        Ok(set) => refset::data_dir().join("ref-vision").join(set),
        Err(_) => VISION.path(VISION_SET),
    }
}

/// The committed test image `name`.
pub fn image_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/ref/vision/images")
        .join(name)
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
    set: VisionSet,
}

impl Manifest {
    /// Read `MANIFEST.tsv`, refusing a set without its trailer or of another revision.
    pub fn load() -> Manifest {
        let set = VisionSet::open(&set_dir(), &VISION).unwrap_or_else(|e| panic!("{e}"));
        Manifest {
            dir: set.dir.clone(),
            mmproj: set.mmproj.clone(),
            mmproj_sha256: set.mmproj_sha256.clone(),
            image_token_id: set.image_token_id,
            images: set.images.clone(),
            set,
        }
    }

    /// One file of the set.
    pub fn read(&self, name: &str) -> Vec<u8> {
        std::fs::read(self.dir.join(name))
            .unwrap_or_else(|e| panic!("{}/{name}: {e}", self.dir.display()))
    }

    /// `plans.tsv`.
    pub fn plans(&self) -> Vec<PlanRow> {
        self.set
            .plans()
            .unwrap_or_else(|e| panic!("{e}"))
            .into_iter()
            .map(|p| PlanRow {
                kind: p.kind,
                w: p.w,
                h: p.h,
                want: [
                    p.n_llm_h,
                    p.n_llm_w,
                    p.best_h,
                    p.best_w,
                    p.n_tokens,
                    p.resized_w,
                    p.resized_h,
                    p.off_x,
                    p.off_y,
                ],
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
