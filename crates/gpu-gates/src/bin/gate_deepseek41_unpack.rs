//! The V4.1 residency unpack's gate (`bloomery_gpu_deepseek41::swap::R8Kernels`):
//! synthetic parts, no model file. A flip stages a gate or an up in the r8
//! row-lane layout (`qdot::repack_q3k_r8`) and the copy stream turns it into
//! Q3_K rows in place with `ds41_r8_q3k_groups`.
//!
//! Clauses:
//! - `equal`: random words as the r8 layout of each shape below, through the
//!   group unpack in place inside a part with guard words on both sides,
//!   through the reference unpack `ds41_r8_q3k` (one thread an output word,
//!   a separate destination) and through `qdot::unpack_q3k_r8` on the host,
//!   give the same bytes, and the guard words are untouched. The shapes are
//!   V4.1's part (2304 rows of 20 super-blocks: more groups than the unpack's
//!   blocks, so each block walks several) and small ones with odd row widths
//!   (a Q3_K block at two bytes past a word) and fewer groups than blocks.
//! - `refusal`: a part off a 16-byte boundary, rows wider than the unpack's
//!   shared group and a part that is not a whole number of groups are each a
//!   named error, and the part is unchanged.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_unpack: built without the `deepseek41` feature; see `just \
         gate-gpu-ds41-unpack`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_unpack", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu_deepseek41::swap::R8Kernels;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use cuda_core::DeviceBuffer;

    /// `sizeof(block_q3_K)`, and the words of one 8-row group super-block.
    const Q3K_BLOCK: usize = 110;
    const GROUP_BLOCK_WORDS: usize = 8 * Q3K_BLOCK / 4;
    /// Guard words on each side of the part the group unpack turns in place.
    const GUARD: usize = 64;
    /// (rows, super-blocks a row): V4.1's gate and up part, then rows of odd
    /// widths and fewer groups than the unpack's blocks.
    const SHAPES: [(usize, usize); 5] = [(2304, 20), (8, 1), (24, 7), (40, 20), (16, 19)];

    /// `n` words of a xorshift stream from `seed`.
    fn words(n: usize, seed: u64) -> Vec<u32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 16) as u32
            })
            .collect()
    }

    fn bytes(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The first byte where `a` and `b` differ, as `row r byte o`.
    fn first_diff(a: &[u8], b: &[u8], nb: usize) -> Option<String> {
        let row_bytes = Q3K_BLOCK * nb;
        (0..a.len().min(b.len()))
            .find(|&o| a[o] != b[o])
            .map(|o| {
                format!(
                    "row {} byte {} ({:#04x} vs {:#04x})",
                    o / row_bytes,
                    o % row_bytes,
                    a[o],
                    b[o]
                )
            })
            .or_else(|| (a.len() != b.len()).then(|| format!("{} vs {} bytes", a.len(), b.len())))
    }

    /// `equal` for one shape.
    fn equal(
        gpu: &Gpu,
        k: &R8Kernels,
        rows: usize,
        nb: usize,
        seed: u64,
    ) -> Result<bool, GateError> {
        let n = rows * nb * Q3K_BLOCK / 4;
        let r8 = words(n, seed);
        let guard = words(2 * GUARD, !seed);
        let mut padded = guard[..GUARD].to_vec();
        padded.extend_from_slice(&r8);
        padded.extend_from_slice(&guard[GUARD..]);
        let s = gpu.stream();

        let part = DeviceBuffer::from_host(s, &padded)?;
        k.enqueue_in_place(s, part.cu_deviceptr() + 4 * GUARD as u64, n, nb)?;
        let src = DeviceBuffer::from_host(s, &r8)?;
        let mut dst = DeviceBuffer::<u32>::zeroed(s, n)?;
        k.enqueue_reference(s, &src, &mut dst, nb)?;
        let got = part.to_host_vec(s)?;
        let reference = bytes(&dst.to_host_vec(s)?);

        let mut host = vec![0u8; 4 * n];
        qdot::unpack_q3k_r8(&bytes(&r8), rows, 256 * nb, &mut host)
            .map_err(|e| format!("qdot::unpack_q3k_r8 refused {rows} rows of {nb}: {e:?}"))?;
        let group = bytes(&got[GUARD..GUARD + n]);
        let guards = got[..GUARD] == guard[..GUARD] && got[GUARD + n..] == guard[GUARD..];

        let vs_ref = first_diff(&group, &reference, nb);
        let vs_host = first_diff(&group, &host, nb);
        let ok = vs_ref.is_none() && vs_host.is_none() && guards;
        println!(
            "equal: {rows} rows x {nb} super-blocks ({} groups, {} B): group unpack vs \
             ds41_r8_q3k {}, vs qdot::unpack_q3k_r8 {}, guard words {} {}",
            rows / 8,
            4 * n,
            vs_ref.as_deref().unwrap_or("equal"),
            vs_host.as_deref().unwrap_or("equal"),
            if guards { "untouched" } else { "WRITTEN" },
            verdict(ok)
        );
        Ok(ok)
    }

    /// `refusal`: each bad call is a named error and the part keeps its words.
    fn refusal(gpu: &Gpu, k: &R8Kernels) -> Result<bool, GateError> {
        let s = gpu.stream();
        // Two groups at 20 super-blocks a row, and room for two at 21: a call
        // that is not refused stays inside the buffer.
        let n = 2 * GROUP_BLOCK_WORDS * 20;
        let before = words(2 * GROUP_BLOCK_WORDS * 21 + 4, 7);
        let part = DeviceBuffer::from_host(s, &before)?;
        let base = part.cu_deviceptr();
        let cases: [(&str, u64, usize, usize); 3] = [
            ("off a 16-byte boundary", base + 4, n, 20),
            (
                "rows of 21 super-blocks",
                base,
                2 * GROUP_BLOCK_WORDS * 21,
                21,
            ),
            ("half a group", base, n - GROUP_BLOCK_WORDS * 10, 20),
        ];
        let mut ok = true;
        for (what, at, words_, nb_) in cases {
            let named = match k.enqueue_in_place(s, at, words_, nb_) {
                Err(e) => {
                    let msg = e.to_string();
                    println!("refusal: {what}: {msg}");
                    msg.contains("R8Kernels::enqueue_in_place")
                }
                Ok(()) => {
                    println!("refusal: {what}: enqueued");
                    false
                }
            };
            ok &= named;
        }
        let unchanged = part.to_host_vec(s)? == before;
        ok &= unchanged;
        println!(
            "refusal: every bad call a named error, the part {} {}",
            if unchanged { "unchanged" } else { "WRITTEN" },
            verdict(ok)
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = R8Kernels::load(gpu.context())?;
        let mut ok = true;
        for (i, &(rows, nb)) in SHAPES.iter().enumerate() {
            ok &= equal(&gpu, &k, rows, nb, 0x9E37_79B9 + i as u64)?;
        }
        ok &= refusal(&gpu, &k)?;
        if ok {
            println!(
                "gate_deepseek41_unpack: PASS — the group unpack turns the r8 layout into the \
                 bytes of the reference kernel and of qdot::unpack_q3k_r8 in place, writes \
                 nothing outside its part, and refuses a bad part by name."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
