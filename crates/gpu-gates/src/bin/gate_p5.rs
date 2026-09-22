//! GPU kernel gate for package P5 (docs/gpu-design.md 작업 꾸러미): the KV
//! append (f32 -> f16) and the latent (MLA) flash attention, decode M=1
//! first, on real inputs from the ik CUDA oracle dump.
//!
//! Two layers per op, as the package's gate rule fixes them:
//! - asserted: device output vs a host f64 reference computed in this binary
//!   from the same inputs — plain softmax in f64 over the f16-roundtripped
//!   keys (the cache bits OUR `kv_append` wrote, read back), band 1e-5 for
//!   the flash arithmetic; exact bit equality for the conversion (it is
//!   exact by construction: the device calls the CPU oracle's own
//!   `f32_to_f16_bits`). Plus a bit-identical rerun per shape.
//! - printed, not asserted: distance to the oracle — `ik_rel` for the flash
//!   output vs `kqv_compressed-L`, and `ik_cache_bits_equal` for the cache
//!   bits vs `kv_cache-L`. The lead pins block-layer bands from these.
//!
//! Chains are verified from the manifest before use: dims, `op`, and the
//! CONCAT orders themselves (q-L occ 1 == [q_rope | q_nope2], kvr-L ==
//! [k_rope | kv_compressed], bit-exact against the dumped operands). The
//! depth cases cover the 32-key block edges (31/32/33, 1000, 4096) and the
//! segment edges the split launch adds (`seg_keys` either side, and a key
//! count far below one segment on a cache many segments tall), with the f16
//! NaN bit pattern in every padded row past `n_keys` — a padded row that
//! reached the result would fail the finiteness check inside `max_rel_err`,
//! which is what asserts that a segment past the causal limit writes its
//! neutral partial without reading a row.
//! Captured-graph checks: flash replay byte-identical to eager, and the
//! pos-buffer append re-reads its position on every replay while the scalar
//! variant stays frozen at its capture-time row. The depth cases add the one
//! a key-split launch needs — a single graph, captured with one live
//! segment, replayed at every key count up to a full cache and still
//! byte-identical to the eager run there. The grid comes from the cache
//! height; a grid derived from the key count passes every other arm and
//! fails that one.
//!
//! One shape check joins them, read off the device code this binary carries
//! rather than off a clock: every flash entry — the single-block
//! `flash_latent`, the split pair `flash_latent_seg`/`flash_merge`, the
//! tensor-core segment pass `flash_latent_mma`, and both appends — must
//! compile with no local depot.
//! A per-thread accumulator array the backend cannot hold in registers is
//! spilled to local memory, and every accumulate becomes a dependent local
//! round trip — a defect that changes no output and no band, so nothing else
//! here can see it.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_p5: built without the `gpu` feature; see `just gate-gpu-p5`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::flash::{
    FlashKernels, f32_to_f16_bits, mma_groups, partials_ms_len, partials_v_len, seg_keys,
    segments_for,
};
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Gpu};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::RefRow;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{
    activations, bits_equal, find_ref_row, max_rel_err, open_model, ref_dir, ref_manifest,
    ref_tensor_of, verdict, widened_f16_bits,
};
#[cfg(feature = "gpu")]
use cuda_core::{CudaStream, DeviceBuffer};
#[cfg(feature = "gpu")]
use gguf::Gguf;
#[cfg(feature = "gpu")]
use gguf::quant::half_to_f32;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The flash arithmetic is not exact by construction (exp, two reduction
    // trees): this package's asserted band. The conversion op asserts bits.
    const FLASH_BAND: f32 = 1e-5;
    // The tensor-core pass rounds the query rows to f16 and accumulates the
    // logits from f16 products, so it is a different arithmetic class from
    // the f32 scalar passes and cannot sit inside their band. Its own band,
    // twice the largest relative error measured over every depth case
    // against the f64 reference and against the one-row pass.
    // PIN(2026-09-22): 8.49e-4 was the largest over every depth case at both
    // segment sizes, against the f64 reference and against the one-row pass
    // alike; this is twice that. The class is f16 query rows and f16
    // products, which is ik's CUDA flash attention's class.
    const FLASH_MMA_BAND: f32 = 1.7e-3;
    const WIDTH: usize = 576; // rope 64 + latent 512 (checked against the dump)
    const ROPE: usize = 64;
    const LATENT: usize = 512;
    const N_HEADS: usize = 16;
    const TOKENS: usize = 6; // the dump's prompt length
    const CTX_MAX: usize = 64; // per-layer cache rows allocated for the real cases
    const DEPTH_ROWS: usize = 4160; // depth cache: 4096 keys + NaN pad

    let mut ok = true;
    let gguf = open_model()?;
    let scale = kq_scale_of(&gguf)?;
    let gpu = Gpu::new()?;
    let flash = FlashKernels::load(gpu.context())?;
    let stream = gpu.stream();
    let man = ref_manifest()?;
    println!(
        "gate_p5: dump {} ({} manifest rows), kq_scale={scale:.6}",
        ref_dir().display(),
        man.len()
    );

    for l in [0usize, 1, 13, 26] {
        real_layer(
            &man, &gpu, &flash, l, scale, FLASH_BAND, WIDTH, ROPE, LATENT, N_HEADS, TOKENS,
            CTX_MAX, &mut ok,
        )?;
    }

    depth_cases(
        &gpu,
        &flash,
        scale,
        WIDTH,
        ROPE,
        LATENT,
        N_HEADS,
        DEPTH_ROWS,
        FLASH_BAND,
        FLASH_MMA_BAND,
        &mut ok,
    )?;
    edge_values(&flash, stream, &mut ok)?;
    no_local_depot(&mut ok)?;

    if !ok {
        eprintln!("FAILED: gate_p5");
        std::process::exit(1);
    }
    println!(
        "PASSED: gate_p5 kv_append bits exact (incl. IEEE edges, ik cache bits equal); \
         the single-block flash within {FLASH_BAND} of the f64 reference on real layers, \
         both paths and their distance inside it on the depth and segment edges; reruns \
         bit-identical; padded NaN rows never read; one captured graph replays \
         bit-identically at every key count; the tensor-core segment pass inside its own \
         {FLASH_MMA_BAND} band against both the reference and the one-row pass; the flash \
         kernels compile with no local depot"
    );
    Ok(())
}

// ------------------------------------------------------------ kernel shape

/// This package's kernels, asserted to carry no local depot. The device
/// bundle is embedded in this executable as PTX text, so the check reads
/// `/proc/self/exe` and looks inside each entry's own body — the same thing
/// the backend would report, at no device cost and with no timing in it.
/// The scan itself is `bloomery_gpu_gates::ptx`; `tools/ptx-scan.sh` prints
/// the same counts for every entry without asserting any of them.
#[cfg(feature = "gpu")]
fn no_local_depot(ok: &mut bool) -> Result<(), Box<dyn std::error::Error>> {
    let blob = std::fs::read(std::env::current_exe()?)?;
    for name in [
        "flash_latent",
        "flash_latent_seg",
        "flash_latent_mma",
        "flash_merge",
        "kv_append",
        "kv_append_pos_buf",
    ] {
        let c = bloomery_gpu_gates::ptx::counts(&blob, name)
            .ok_or_else(|| format!("gate_p5: no PTX entry {name} in this executable"))?;
        let (depot, loads, stores) = (c.depot, c.ld_local, c.st_local);
        let pass = !depot && loads == 0 && stores == 0;
        println!(
            "shape kernel={name} local_depot={depot} ld_local={loads} st_local={stores} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            *ok = false;
        }
    }
    Ok(())
}

// ------------------------------------------------------------ real layers

/// One layer's real-input cases: the cache build, the five m=1 decode
/// positions, the m=TOKENS causal prefill shape, and (layer 0 only) the
/// captured-graph checks.
#[cfg(feature = "gpu")]
#[allow(
    clippy::too_many_arguments,
    reason = "gate harness: the case's buffers are passed flat; a params struct is the R8 round"
)]
fn real_layer(
    man: &[RefRow],
    gpu: &Gpu,
    flash: &FlashKernels,
    l: usize,
    scale: f32,
    band: f32,
    width: usize,
    rope: usize,
    latent: usize,
    n_heads: usize,
    tokens: usize,
    ctx_max: usize,
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = gpu.stream();
    // The split launch's partials, sized for the widest shape below (the
    // m=tokens prefill) over this cache height. At ctx_max = 64 the height
    // is one segment, so every call here takes the single-launch fast path
    // and leaves them untouched — that choice is itself asserted by the
    // graph node count.
    let mut part_v =
        DeviceBuffer::<f32>::zeroed(stream, partials_v_len(tokens * n_heads, ctx_max))?;
    let mut part_ms =
        DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(tokens * n_heads, ctx_max))?;

    // ---- chain: kvr-L [width, tokens] CONCAT of [k_rope | kv_compressed];
    // k_rope occ 1 is (d, t), kv_compressed occ 1 is (d, t).
    let kvr_row = find_ref_row(man, &format!("kvr-{l}"), 0)?;
    if kvr_row.op != "CONCAT"
        || kvr_row.ty != "f32"
        || kvr_row.ne != [width as u64, tokens as u64, 1, 1]
    {
        return Err(format!(
            "gate_p5: kvr-{l} is {} {} {:?}, want CONCAT f32 [{width}, {tokens}]",
            kvr_row.op, kvr_row.ty, kvr_row.ne
        )
        .into());
    }
    let kvr = ref_tensor_of(kvr_row)?;
    let k_rope_row = find_ref_row(man, &format!("k_rope-{l}"), 1)?;
    let kv_compressed_row = find_ref_row(man, &format!("kv_compressed-{l}"), 1)?;
    if k_rope_row.op != "ROPE"
        || k_rope_row.ty != "f32"
        || k_rope_row.ne != [rope as u64, 1, tokens as u64, 1]
    {
        return Err(format!(
            "gate_p5: k_rope-{l}/1 is {} {} {:?}, want ROPE f32 [{rope}, 1, {tokens}]",
            k_rope_row.op, k_rope_row.ty, k_rope_row.ne
        )
        .into());
    }
    if kv_compressed_row.op != "FUSED_RMS_NORM"
        || kv_compressed_row.ty != "f32"
        || kv_compressed_row.ne != [latent as u64, tokens as u64, 1, 1]
    {
        return Err(format!(
            "gate_p5: kv_compressed-{l}/1 is {} {} {:?}, want FUSED_RMS_NORM f32 \
             [{latent}, {tokens}]",
            kv_compressed_row.op, kv_compressed_row.ty, kv_compressed_row.ne
        )
        .into());
    }
    let k_rope = ref_tensor_of(k_rope_row)?;
    let kv_compressed = ref_tensor_of(kv_compressed_row)?;
    for t in 0..tokens {
        for d in 0..width {
            let want = if d < rope {
                k_rope[t * rope + d]
            } else {
                kv_compressed[t * latent + (d - rope)]
            };
            if kvr[t * width + d].to_bits() != want.to_bits() {
                return Err(format!(
                    "gate_p5: kvr-{l} is not [k_rope | kv_compressed]: token {t} dim {d}"
                )
                .into());
            }
        }
    }

    // ---- chain: q-L occ 1 [width, tokens, heads] CONCAT of [q_rope |
    // q_nope2]; rows (t + tokens*h); q_rope occ 1 is (d, h, t), q_nope2
    // occ 0 is (d, t, h).
    let q_row = find_ref_row(man, &format!("q-{l}"), 1)?;
    if q_row.op != "CONCAT"
        || q_row.ty != "f32"
        || q_row.ne != [width as u64, tokens as u64, n_heads as u64, 1]
    {
        return Err(format!(
            "gate_p5: q-{l}/1 is {} {} {:?}, want CONCAT f32 [{width}, {tokens}, {n_heads}]",
            q_row.op, q_row.ty, q_row.ne
        )
        .into());
    }
    let q1 = ref_tensor_of(q_row)?;
    let q_rope_row = find_ref_row(man, &format!("q_rope-{l}"), 1)?;
    let q_nope2_row = find_ref_row(man, &format!("q_nope2-{l}"), 0)?;
    if q_rope_row.op != "ROPE"
        || q_rope_row.ty != "f32"
        || q_rope_row.ne != [rope as u64, n_heads as u64, tokens as u64, 1]
    {
        return Err(format!(
            "gate_p5: q_rope-{l}/1 is {} {} {:?}, want ROPE f32 [{rope}, {n_heads}, {tokens}]",
            q_rope_row.op, q_rope_row.ty, q_rope_row.ne
        )
        .into());
    }
    if q_nope2_row.op != "MUL_MAT"
        || q_nope2_row.ty != "f32"
        || q_nope2_row.ne != [latent as u64, tokens as u64, n_heads as u64, 1]
    {
        return Err(format!(
            "gate_p5: q_nope2-{l}/0 is {} {} {:?}, want MUL_MAT f32 \
             [{latent}, {tokens}, {n_heads}]",
            q_nope2_row.op, q_nope2_row.ty, q_nope2_row.ne
        )
        .into());
    }
    let q_rope = ref_tensor_of(q_rope_row)?;
    let q_nope2 = ref_tensor_of(q_nope2_row)?;
    for h in 0..n_heads {
        for t in 0..tokens {
            for d in 0..width {
                let base = (t + tokens * h) * width;
                let want = if d < rope {
                    q_rope[t * n_heads * rope + h * rope + d]
                } else {
                    q_nope2[(t + tokens * h) * latent + (d - rope)]
                };
                if q1[base + d].to_bits() != want.to_bits() {
                    return Err(format!(
                        "gate_p5: q-{l}/1 is not [q_rope | q_nope2]: (t={t}, h={h}, d={d})"
                    )
                    .into());
                }
            }
        }
    }

    // ---- chain: kqv_compressed-L [latent, heads, tokens] FLASH_ATTN_EXT.
    let kqv_row = find_ref_row(man, &format!("kqv_compressed-{l}"), 0)?;
    if kqv_row.op != "FLASH_ATTN_EXT"
        || kqv_row.ty != "f32"
        || kqv_row.ne != [latent as u64, n_heads as u64, tokens as u64, 1]
    {
        return Err(format!(
            "gate_p5: kqv_compressed-{l} is {} {} {:?}, want FLASH_ATTN_EXT f32 \
             [{latent}, {n_heads}, {tokens}]",
            kqv_row.op, kqv_row.ty, kqv_row.ne
        )
        .into());
    }
    let kqv = ref_tensor_of(kqv_row)?;

    // ---- kv_append: build THIS layer's cache from the kvr rows.
    let src = DeviceBuffer::from_host(stream, &kvr[..tokens * width])?;
    let mut cache = DeviceTensor::<u16>::zeroed(stream, ctx_max, width)?;
    flash.enqueue_kv_append(stream, &src, &mut cache, tokens, 0)?;
    stream.synchronize()?;
    let bits1 = cache.buf().to_host_vec(stream)?;
    flash.enqueue_kv_append(stream, &src, &mut cache, tokens, 0)?;
    stream.synchronize()?;
    let bits2 = cache.buf().to_host_vec(stream)?;
    let rerun_same = bits1 == bits2;
    let bits_exact = kvr[..tokens * width]
        .iter()
        .zip(&bits1)
        .all(|(&v, &b)| b == f32_to_f16_bits(v));
    let neighbours = bits1[tokens * width..].iter().all(|&b| b == 0);
    // ik's own cache rows: an f16 VIEW whose file the dumper widened to f32
    // exactly, so rounding back recovers ik's bits (ref_tensor_of rejects
    // the f16 label, hence the widened_f16_bits helper in the lib).
    let ik_cache_row = find_ref_row(man, &format!("kv_cache-{l}"), 0)?;
    if ik_cache_row.ty != "f16"
        || ik_cache_row.op != "VIEW"
        || ik_cache_row.ne != [width as u64, 256, 1, 1]
    {
        return Err(format!(
            "gate_p5: kv_cache-{l} is {} {} {:?}, want VIEW f16 [{width}, 256]",
            ik_cache_row.op, ik_cache_row.ty, ik_cache_row.ne
        )
        .into());
    }
    let ik_bits = widened_f16_bits(ik_cache_row, tokens)?;
    let ik_cache_bits_equal = bits1[..tokens * width] == ik_bits[..];
    // The ik column is printed, not asserted (the package's gate rule): the
    // asserted layer is bit equality with the oracle's own conversion.
    let pass = bits_exact && neighbours && rerun_same;
    println!(
        "shape op=kv_append layer={l} rows={tokens} width={width} bits_exact={bits_exact} ik_cache_bits_equal={ik_cache_bits_equal} neighbours_untouched={neighbours} bit_identical_rerun={rerun_same} {}",
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }

    // ---- flash m=1: five decode positions (one query over a causal
    // prefix), the cache built above supplying the f16 keys.
    for t in 1..tokens {
        let n_keys = t + 1;
        let mut qbuf = vec![0.0f32; n_heads * width];
        for h in 0..n_heads {
            qbuf[h * width..(h + 1) * width]
                .copy_from_slice(&q1[(t + tokens * h) * width..(t + tokens * h + 1) * width]);
        }
        let q_dev = DeviceBuffer::from_host(stream, &qbuf)?;
        let n_keys_dev = DeviceBuffer::from_host(stream, &[n_keys as u32])?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
        let run = |y: &mut DeviceBuffer<f32>,
                   pv: &mut DeviceBuffer<f32>,
                   pm: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            flash.enqueue_flash_latent_split(
                stream,
                &q_dev,
                &cache,
                &n_keys_dev,
                scale,
                1,
                n_heads,
                rope,
                latent,
                pv,
                pm,
                y,
            )?;
            stream.synchronize()?;
            Ok(())
        };
        run(&mut y_dev, &mut part_v, &mut part_ms)?;
        let y1 = y_dev.to_host_vec(stream)?;
        run(&mut y_dev, &mut part_v, &mut part_ms)?;
        let y2 = y_dev.to_host_vec(stream)?;
        let rerun_same = bits_equal(&y1, &y2);
        let y_ref = flash_f64_ref(&qbuf, &bits1, n_keys, scale, 1, n_heads, rope, latent);
        let rel = max_rel_err(&y1, &y_ref)?;
        let ik = ik_kqv_rows(&kqv, t, n_heads, latent);
        let ik_rel = max_rel_err(&y1, &ik)?;
        let pass = rel <= band && rerun_same;
        println!(
            "shape op=flash_latent layer={l} t={t} n_keys={n_keys} m=1 max_rel_err={rel:.3e} bit_identical_rerun={rerun_same} ik_rel={ik_rel:.3e} {}",
            verdict(pass)
        );
        if !pass {
            *ok = false;
        }
        if l == 0 && t == tokens - 1 {
            graph_check_flash(
                gpu,
                flash,
                &q_dev,
                &cache,
                &n_keys_dev,
                scale,
                n_heads,
                rope,
                latent,
                &y1,
                ok,
            )?;
        }
    }

    // ---- flash m=tokens: the causal prefill shape, all heads and tokens in
    // one launch (the m<=8 variant; the loop above was the m=1 decode shape).
    let mut qbuf = vec![0.0f32; tokens * n_heads * width];
    for t in 0..tokens {
        for h in 0..n_heads {
            let src_base = (t + tokens * h) * width;
            let dst_base = (t * n_heads + h) * width;
            qbuf[dst_base..dst_base + width].copy_from_slice(&q1[src_base..src_base + width]);
        }
    }
    let q_dev = DeviceBuffer::from_host(stream, &qbuf)?;
    let n_keys_dev = DeviceBuffer::from_host(stream, &[tokens as u32])?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, tokens * n_heads * latent)?;
    let run = |y: &mut DeviceBuffer<f32>,
               pv: &mut DeviceBuffer<f32>,
               pm: &mut DeviceBuffer<f32>|
     -> Result<(), Box<dyn std::error::Error>> {
        flash.enqueue_flash_latent_split(
            stream,
            &q_dev,
            &cache,
            &n_keys_dev,
            scale,
            tokens,
            n_heads,
            rope,
            latent,
            pv,
            pm,
            y,
        )?;
        stream.synchronize()?;
        Ok(())
    };
    run(&mut y_dev, &mut part_v, &mut part_ms)?;
    let y1 = y_dev.to_host_vec(stream)?;
    run(&mut y_dev, &mut part_v, &mut part_ms)?;
    let y2 = y_dev.to_host_vec(stream)?;
    let rerun_same = bits_equal(&y1, &y2);
    let y_ref = flash_f64_ref(&qbuf, &bits1, tokens, scale, tokens, n_heads, rope, latent);
    let rel = max_rel_err(&y1, &y_ref)?;
    let mut ik = vec![0.0f32; tokens * n_heads * latent];
    for (t, chunk) in ik.chunks_mut(n_heads * latent).enumerate() {
        chunk.copy_from_slice(&ik_kqv_rows(&kqv, t, n_heads, latent));
    }
    let ik_rel = max_rel_err(&y1, &ik)?;
    let pass = rel <= band && rerun_same;
    println!(
        "shape op=flash_latent layer={l} m={tokens} n_keys={tokens} causal=prefill max_rel_err={rel:.3e} bit_identical_rerun={rerun_same} ik_rel={ik_rel:.3e} {}",
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }
    if l == 0 {
        graph_check_append(gpu, flash, &kvr[..width], ok)?;
    }
    Ok(())
}

// ----------------------------------------------------------- depth cases

/// Synthetic LCG keys/queries at the 32-key block edges, every row past
/// `n_keys` holding the f16 NaN bit pattern.
#[cfg(feature = "gpu")]
#[allow(
    clippy::too_many_arguments,
    reason = "gate harness: the case's buffers are passed flat; a params struct is the R8 round"
)]
fn depth_cases(
    gpu: &Gpu,
    flash: &FlashKernels,
    scale: f32,
    width: usize,
    rope: usize,
    latent: usize,
    n_heads: usize,
    depth_rows: usize,
    band: f32,
    mma_band: f32,
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = gpu.stream();
    let keys = activations(width, 4096, 50021);
    let queries = activations(width, n_heads, 60013);
    let mut cache_bits = vec![0x7e00u16; depth_rows * width]; // NaN pad everywhere
    for (b, &v) in cache_bits.iter_mut().zip(&keys) {
        *b = f32_to_f16_bits(v);
    }
    let cache = DeviceTensor::upload(stream, &cache_bits, depth_rows, width)?;
    let q_dev = DeviceBuffer::from_host(stream, &queries)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    // This cache is tall enough to be cut into segments, so every case below
    // runs the split launch. `segs` counts the whole cache, not the live
    // keys: at small `n_keys` most segments are past the causal limit and
    // must write their neutral partial without reading a row — every one of
    // those rows holds the f16 NaN pattern, so a segment that read one would
    // fail the finiteness check inside `max_rel_err`.
    let segs = segments_for(depth_rows);
    let seg = seg_keys();
    let mut part_v = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(n_heads, depth_rows))?;
    let mut part_ms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(n_heads, depth_rows))?;
    let mut y_one = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    // 1/31/32/33 are the key-tile edges; seg-1/seg/seg+1 the segment edges a
    // split launch adds; 1 and 31 are also `n_keys` far below one segment
    // with a tall cache. 4096 is a whole number of 256-key segments and
    // leaves the last segment of the 4160-row cache wholly empty.
    let mut cases = vec![1usize, 31, 32, 33, 1000, 4096];
    for extra in [seg - 1, seg, seg + 1, 2 * seg] {
        if extra < depth_rows && !cases.contains(&extra) {
            cases.push(extra);
        }
    }
    cases.sort_unstable();
    // One key-count buffer and ONE captured graph for every case below. The
    // grid was fixed from the cache height at capture time, so a replay must
    // follow `n_keys_buf` across segment boundaries — from a single live
    // segment at capture to every segment live at 4096 keys — and still land
    // bit for bit on the eager run. A grid derived from the key count would
    // pass every other arm here and fail this one.
    let mut n_keys_dev = DeviceBuffer::from_host(stream, &[1u32])?;
    // The tensor-core pass writes the same partials from its own buffers, so
    // its distance to the one-row pass is a comparison of two full results
    // and not of a shared scratch.
    let mut part_vm = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(n_heads, depth_rows))?;
    let mut part_msm = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(n_heads, depth_rows))?;
    let mut y_m = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    let graph = gpu.capture(|_s| {
        flash
            .enqueue_flash_latent_split(
                stream,
                &q_dev,
                &cache,
                &n_keys_dev,
                scale,
                1,
                n_heads,
                rope,
                latent,
                &mut part_v,
                &mut part_ms,
                &mut y_dev,
            )
            .map(|_| ())
    })?;
    let graph_nodes = graph.node_count();
    // The tensor-core pass's own graph, captured at the same one live
    // segment: its grid comes from the cache height and the head count, so a
    // replay must follow `n_keys_buf` across every segment boundary too.
    let graph_m = gpu.capture(|_s| {
        flash.enqueue_flash_latent_mma(
            stream,
            &q_dev,
            &cache,
            &n_keys_dev,
            scale,
            1,
            n_heads,
            rope,
            latent,
            &mut part_vm,
            &mut part_msm,
        )?;
        flash.enqueue_flash_merge(
            stream,
            &n_keys_dev,
            depth_rows,
            1,
            n_heads,
            latent,
            &part_vm,
            &part_msm,
            &mut y_m,
        )
    })?;
    let graph_m_nodes = graph_m.node_count();
    for n_keys in cases {
        n_keys_dev.copy_from_host(stream, &[n_keys as u32])?;
        let run = |y: &mut DeviceBuffer<f32>,
                   pv: &mut DeviceBuffer<f32>,
                   pm: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            flash.enqueue_flash_latent_split(
                stream,
                &q_dev,
                &cache,
                &n_keys_dev,
                scale,
                1,
                n_heads,
                rope,
                latent,
                pv,
                pm,
                y,
            )?;
            stream.synchronize()?;
            Ok(())
        };
        run(&mut y_dev, &mut part_v, &mut part_ms)?;
        let y1 = y_dev.to_host_vec(stream)?;
        run(&mut y_dev, &mut part_v, &mut part_ms)?;
        let y2 = y_dev.to_host_vec(stream)?;
        let rerun_same = bits_equal(&y1, &y2);
        let y_ref = flash_f64_ref(
            &queries,
            &cache_bits,
            n_keys,
            scale,
            1,
            n_heads,
            rope,
            latent,
        );
        let rel = max_rel_err(&y1, &y_ref)?;
        // The single-block entry on the same inputs: the path a cache short
        // enough to hold one segment takes. Both must land inside the band
        // against the f64 reference, and so must their distance to each
        // other — the split moves the summation order, nothing else.
        flash.enqueue_flash_latent(
            stream,
            &q_dev,
            &cache,
            &n_keys_dev,
            scale,
            1,
            n_heads,
            rope,
            latent,
            &mut y_one,
        )?;
        stream.synchronize()?;
        let y_single = y_one.to_host_vec(stream)?;
        let one_rel = max_rel_err(&y_single, &y_ref)?;
        let cross_rel = max_rel_err(&y1, &y_single)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let y_graph = y_dev.to_host_vec(stream)?;
        let replay_same = bits_equal(&y1, &y_graph);
        // The tensor-core pass on the same inputs, through the same merge.
        // Its query rows and products are f16, so it carries its own band
        // (`mma_band`) against the f64 reference and against the one-row
        // pass; reruns and graph replays of it are still bit-identical.
        let run_m = |y: &mut DeviceBuffer<f32>,
                     pv: &mut DeviceBuffer<f32>,
                     pm: &mut DeviceBuffer<f32>|
         -> Result<(), Box<dyn std::error::Error>> {
            flash.enqueue_flash_latent_mma(
                stream,
                &q_dev,
                &cache,
                &n_keys_dev,
                scale,
                1,
                n_heads,
                rope,
                latent,
                pv,
                pm,
            )?;
            flash.enqueue_flash_merge(
                stream,
                &n_keys_dev,
                depth_rows,
                1,
                n_heads,
                latent,
                pv,
                pm,
                y,
            )?;
            stream.synchronize()?;
            Ok(())
        };
        run_m(&mut y_m, &mut part_vm, &mut part_msm)?;
        let m1 = y_m.to_host_vec(stream)?;
        run_m(&mut y_m, &mut part_vm, &mut part_msm)?;
        let m2 = y_m.to_host_vec(stream)?;
        let rerun_same_m = bits_equal(&m1, &m2);
        let rel_m = max_rel_err(&m1, &y_ref)?;
        let cross_m = max_rel_err(&m1, &y1)?;
        graph_m.launch(stream)?;
        stream.synchronize()?;
        let m_graph = y_m.to_host_vec(stream)?;
        let replay_same_m = bits_equal(&m1, &m_graph);
        let pass_m = rel_m <= mma_band && cross_m <= mma_band && rerun_same_m && replay_same_m;
        println!(
            "shape op=flash_latent_mma_depth n_keys={n_keys} m=1 seg_keys={seg} segs={segs} groups={} band={mma_band:.3e} max_rel_err={rel_m:.3e} mma_vs_seg={cross_m:.3e} bit_identical_rerun={rerun_same_m} graph_nodes={graph_m_nodes} replay_bit_identical={replay_same_m} {}",
            mma_groups(n_heads),
            verdict(pass_m)
        );
        if !pass_m {
            *ok = false;
        }
        let pass = rel <= band && one_rel <= band && cross_rel <= band && rerun_same && replay_same;
        println!(
            "shape op=flash_latent_depth n_keys={n_keys} m=1 seg_keys={seg} segs={segs} live_segs={} nan_pad_rows={} max_rel_err={rel:.3e} single_launch_rel={one_rel:.3e} split_vs_single={cross_rel:.3e} bit_identical_rerun={rerun_same} graph_nodes={graph_nodes} replay_bit_identical={replay_same} {}",
            n_keys.div_ceil(seg),
            depth_rows - n_keys,
            verdict(pass)
        );
        if !pass {
            *ok = false;
        }
    }
    Ok(())
}

// ----------------------------------------------------------- IEEE edges

/// The conversion op's exactness on the edges the real rows cannot reach:
/// zeros, signed zero, subnormals on both sides, the f16 finite boundary,
/// overflow, and the NaN class (which collapses to inf, the oracle's rule).
#[cfg(feature = "gpu")]
fn edge_values(
    flash: &FlashKernels,
    stream: &CudaStream,
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let edges: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        5.5,
        f32::from_bits(1), // smallest f32 subnormal
        -f32::from_bits(1),
        2.0f32.powi(-24), // smallest f16 subnormal
        2.0f32.powi(-25), // ties to 0 (even)
        3.0f32.powi(-25), // ties to 1 subnormal (even)
        65504.0,          // f16 max
        65505.0,          // rounds down to f16 max
        65519.0,          // last below the overflow tie
        65520.0,          // overflow tie -> inf (even)
        -65520.0,
        1.0e30, // far past the finite range
        f32::NAN,
        -f32::NAN,
    ];
    let width = edges.len();
    let src = DeviceBuffer::from_host(stream, &edges)?;
    let mut cache = DeviceTensor::<u16>::zeroed(stream, 2, width)?;
    flash.enqueue_kv_append(stream, &src, &mut cache, 1, 0)?;
    stream.synchronize()?;
    let got = cache.buf().to_host_vec(stream)?;
    let want: Vec<u16> = edges.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let bits_exact = got[..width] == want[..];
    let row1_untouched = got[width..].iter().all(|&b| b == 0);
    let pass = bits_exact && row1_untouched;
    println!(
        "shape op=kv_append_edges width={width} bits_exact={bits_exact} row1_untouched={row1_untouched} {}",
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }
    Ok(())
}

// --------------------------------------------------------------- graphs

/// Flash inside a captured graph: replay must be byte-identical to eager.
#[cfg(feature = "gpu")]
#[allow(
    clippy::too_many_arguments,
    reason = "gate harness: the case's buffers are passed flat; a params struct is the R8 round"
)]
fn graph_check_flash(
    gpu: &Gpu,
    flash: &FlashKernels,
    q_dev: &DeviceBuffer<f32>,
    cache: &DeviceTensor<u16>,
    n_keys_dev: &DeviceBuffer<u32>,
    scale: f32,
    n_heads: usize,
    rope: usize,
    latent: usize,
    y_eager: &[f32],
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = gpu.stream();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    let segs = segments_for(cache.rows());
    let mut part_v = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(n_heads, cache.rows()))?;
    let mut part_ms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(n_heads, cache.rows()))?;
    let graph = gpu.capture(|_s| {
        flash
            .enqueue_flash_latent_split(
                stream,
                q_dev,
                cache,
                n_keys_dev,
                scale,
                1,
                n_heads,
                rope,
                latent,
                &mut part_v,
                &mut part_ms,
                &mut y_dev,
            )
            .map(|_| ())
    })?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let y_graph = y_dev.to_host_vec(stream)?;
    let identical = bits_equal(y_eager, &y_graph);
    let nodes = graph.node_count();
    // One node per launch, and the launch count is the cache height's
    // decision: a one-segment cache takes the single-block fast path.
    let want_nodes = if segs == 1 { 1 } else { 2 };
    let pass = identical && nodes == want_nodes;
    println!(
        "graph op=flash_latent segs={segs} graph_nodes={nodes} want_nodes={want_nodes} eager_vs_graph_bit_identical={identical} {}",
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }
    Ok(())
}

/// The pos-buffer append inside a captured graph: rewriting the buffer
/// between replays moves the landing rows — the form a captured decode step
/// needs. The scalar-pos append captured beside it stays frozen at its
/// capture-time row, which is why the step cannot use it.
#[cfg(feature = "gpu")]
fn graph_check_append(
    gpu: &Gpu,
    flash: &FlashKernels,
    one_row: &[f32],
    ok: &mut bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = gpu.stream();
    let src = DeviceBuffer::from_host(stream, one_row)?;
    let mut pos_buf = DeviceBuffer::from_host(stream, &[0u32])?;
    let width = one_row.len();
    let mut cache = DeviceTensor::<u16>::zeroed(stream, 8, width)?;
    let graph =
        gpu.capture(|_s| flash.enqueue_kv_append_pos_buf(stream, &src, &pos_buf, &mut cache, 1))?;

    let want: Vec<u16> = one_row.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let rows_equal =
        |bits: &[u16]| -> usize { bits.chunks(width).filter(|row| *row == &want[..]).count() };
    let row = |r: usize, bits: &[u16]| -> bool { bits[r * width..(r + 1) * width] == want[..] };

    // Replay at pos 3, then at pos 7: each replay writes exactly its row and
    // leaves the earlier one in place.
    pos_buf.copy_from_host(stream, &[3u32])?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let bits_a = cache.buf().to_host_vec(stream)?;
    pos_buf.copy_from_host(stream, &[7u32])?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let bits_b = cache.buf().to_host_vec(stream)?;
    let buf_ok = row(3, &bits_a)
        && rows_equal(&bits_a) == 1
        && row(3, &bits_b)
        && row(7, &bits_b)
        && rows_equal(&bits_b) == 2;

    // Scalar variant captured at pos 2: both replays write row 2 only.
    let mut cache2 = DeviceTensor::<u16>::zeroed(stream, 8, width)?;
    let graph2 = gpu.capture(|_s| flash.enqueue_kv_append(stream, &src, &mut cache2, 1, 2))?;
    graph2.launch(stream)?;
    stream.synchronize()?;
    let bits_c = cache2.buf().to_host_vec(stream)?;
    graph2.launch(stream)?;
    stream.synchronize()?;
    let bits_d = cache2.buf().to_host_vec(stream)?;
    let scalar_frozen = row(2, &bits_c) && row(2, &bits_d) && rows_equal(&bits_d) == 1;

    let pass = buf_ok && scalar_frozen;
    println!(
        "graph op=kv_append_pos_buf replays=2 pos=3,7 rows_follow_pos={} scalar_pos_frozen_at_capture_row={} {}",
        buf_ok,
        scalar_frozen,
        verdict(pass)
    );
    if !pass {
        *ok = false;
    }
    Ok(())
}

// -------------------------------------------------------------- helpers

/// `MlaParams::read`'s `kq_scale` (crates/model/src/attn.rs: mscale =
/// 1 + log_mul·ln(1/freq_scale), kq_scale = mscale²/√key_length — the YaRN
/// mscale is inside it, not 1/√d). Mirrored because gpu-gates has no edge to
/// bloomery-model; the `ik_rel` columns are what prove the value against the
/// oracle.
#[cfg(feature = "gpu")]
fn kq_scale_of(gguf: &Gguf) -> Result<f32, Box<dyn std::error::Error>> {
    let f32v = |key: &str| -> Result<f32, Box<dyn std::error::Error>> {
        gguf.value(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| format!("gate_p5: metadata {key} missing").into())
    };
    let scaling = f32v("deepseek2.rope.scaling.factor")?;
    let log_mul = f32v("deepseek2.rope.scaling.yarn_log_multiplier")?;
    let kq_head = gguf
        .arch_get_u64("attention.key_length")
        .ok_or("gate_p5: metadata attention.key_length missing")?;
    let freq_scale = 1.0 / scaling;
    let mscale = 1.0 + log_mul * (1.0 / freq_scale).ln();
    Ok(mscale * mscale / (kq_head as f32).sqrt())
}

/// Plain softmax in f64 over the f16-roundtripped keys: per query row
/// `t*n_heads + h` (the kernels' row order), `s_i = scale · Σ_d q[d]·k_i[d]`
/// accumulated in f64, weights `exp(s_i − max)` in f64, output
/// `Σ w_i·v_i[d] / Σ w_i` cast once to f32. Query `t` of `m` attends to
/// keys `0..n_keys − m + t + 1` (the causal prefix the kernel uses).
#[cfg(feature = "gpu")]
#[allow(
    clippy::too_many_arguments,
    reason = "gate harness: the case's buffers are passed flat; a params struct is the R8 round"
)]
fn flash_f64_ref(
    q: &[f32],
    keys16: &[u16],
    n_keys: usize,
    scale: f32,
    m: usize,
    n_heads: usize,
    rope: usize,
    latent: usize,
) -> Vec<f32> {
    let width = rope + latent;
    let mut out = vec![0.0f32; m * n_heads * latent];
    for row in 0..m * n_heads {
        let t = row / n_heads;
        let limit = n_keys + t + 1 - m;
        let qrow = &q[row * width..(row + 1) * width];
        let mut s = vec![0.0f64; limit];
        for (i, si) in s.iter_mut().enumerate() {
            let krow = &keys16[i * width..(i + 1) * width];
            let mut acc = 0.0f64;
            for d in 0..width {
                acc += f64::from(qrow[d]) * f64::from(half_to_f32(krow[d]));
            }
            *si = f64::from(scale) * acc;
        }
        let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        let mut num = vec![0.0f64; latent];
        for i in 0..limit {
            let w = (s[i] - mx).exp();
            denom += w;
            let vrow = &keys16[i * width + rope..(i + 1) * width];
            for (nd, &v) in num.iter_mut().zip(vrow) {
                *nd += w * f64::from(half_to_f32(v));
            }
        }
        let inv = 1.0 / denom;
        for d in 0..latent {
            out[row * latent + d] = (num[d] * inv) as f32;
        }
    }
    out
}

/// `kqv_compressed-L` rows for one token, in the kernels' row order (row
/// `h` at `h*latent`): dump value (d, h, t) sits at
/// `d + latent*h + latent*n_heads*t`.
#[cfg(feature = "gpu")]
fn ik_kqv_rows(kqv: &[f32], t: usize, n_heads: usize, latent: usize) -> Vec<f32> {
    (0..n_heads)
        .flat_map(|h| (0..latent).map(move |d| kqv[d + latent * h + latent * n_heads * t]))
        .collect()
}
