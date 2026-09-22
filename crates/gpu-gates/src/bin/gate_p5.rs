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
    FlashGeom, FlashInputs, FlashKernels, FlashLatentArgs, FlashMergeArgs, FlashSegArgs,
    FlashSplitArgs, f32_to_f16_bits, mma_groups, partials_ms_len, partials_v_len, seg_keys,
    segments_for,
};
#[cfg(feature = "gpu")]
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Graph};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, RefRow};
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
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_p5", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
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

    let dims = Dims {
        kq_scale: scale,
        width: WIDTH,
        rope_dims: ROPE,
        latent_dims: LATENT,
        heads: N_HEADS,
    };
    // Every helper runs all of its cases and returns whether they all
    // passed; `&=` (not `&&`) so a failure never skips a later helper's lines.
    let mut ok = true;
    for l in [0usize, 1, 13, 26] {
        let case = RealCase {
            man: &man,
            layer: l,
            dims,
            band: FLASH_BAND,
            tokens: TOKENS,
            ctx_max: CTX_MAX,
        };
        ok &= real_layer(&gpu, &flash, &case)?;
    }

    let case = DepthCase {
        dims,
        cache_rows: DEPTH_ROWS,
        band: FLASH_BAND,
        mma_band: FLASH_MMA_BAND,
    };
    ok &= depth_cases(&gpu, &flash, &case)?;
    ok &= edge_values(&flash, stream)?;
    ok &= no_local_depot()?;

    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
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
fn no_local_depot() -> Result<bool, GateError> {
    let blob = std::fs::read(std::env::current_exe()?)?;
    let mut ok = true;
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
        ok &= pass;
    }
    Ok(ok)
}

// ------------------------------------------------------------ cases

/// The attention shape every case here runs at, fixed by the dump:
/// `width = rope_dims + latent_dims` f32 per cache and query row.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
struct Dims {
    kq_scale: f32,
    width: usize,
    rope_dims: usize,
    latent_dims: usize,
    heads: usize,
}

#[cfg(feature = "gpu")]
impl Dims {
    /// The launch geometry for `tokens` query tokens at this shape.
    fn geom(self, tokens: usize) -> FlashGeom {
        FlashGeom {
            tokens,
            heads: self.heads,
            rope_dims: self.rope_dims,
            latent_dims: self.latent_dims,
        }
    }
}

/// One real layer's case: its dump rows, the asserted band, the prompt
/// length and the cache height allocated for it.
#[cfg(feature = "gpu")]
struct RealCase<'a> {
    man: &'a [RefRow],
    layer: usize,
    dims: Dims,
    band: f32,
    tokens: usize,
    ctx_max: usize,
}

/// The synthetic depth cases' shape: the cache height (keys plus NaN pad)
/// and the two bands, the scalar passes' and the tensor-core pass's.
#[cfg(feature = "gpu")]
struct DepthCase {
    dims: Dims,
    cache_rows: usize,
    band: f32,
    mma_band: f32,
}

/// The eager run [`graph_check_flash`] replays against: its inputs and the
/// output it produced.
#[cfg(feature = "gpu")]
struct GraphFlashCase<'a> {
    inputs: FlashInputs<'a>,
    y_eager: &'a [f32],
}

/// The split launch's partials, sized for `q_rows` query rows over a cache
/// of `cache_rows`.
#[cfg(feature = "gpu")]
struct Partials {
    v: DeviceBuffer<f32>,
    ms: DeviceBuffer<f32>,
}

#[cfg(feature = "gpu")]
impl Partials {
    fn zeroed(stream: &CudaStream, q_rows: usize, cache_rows: usize) -> Result<Self, GateError> {
        Ok(Self {
            v: DeviceBuffer::zeroed(stream, partials_v_len(q_rows, cache_rows))?,
            ms: DeviceBuffer::zeroed(stream, partials_ms_len(q_rows, cache_rows))?,
        })
    }
}

/// Two eager runs of the split launch into `y`, each synchronized and read
/// back: the result and its rerun.
#[cfg(feature = "gpu")]
fn split_twice(
    flash: &FlashKernels,
    stream: &CudaStream,
    inputs: FlashInputs<'_>,
    part: &mut Partials,
    y: &mut DeviceBuffer<f32>,
) -> Result<(Vec<f32>, Vec<f32>), GateError> {
    let mut run = || -> Result<Vec<f32>, GateError> {
        flash.enqueue_flash_latent_split(
            stream,
            FlashSplitArgs {
                inputs,
                part_v: &mut part.v,
                part_ms: &mut part.ms,
                y: &mut *y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    };
    let y1 = run()?;
    let y2 = run()?;
    Ok((y1, y2))
}

/// The split launch captured into a graph, not launched: its grid is fixed
/// from the cache height at capture time, the live key count read from
/// `n_keys_buf` on every replay.
#[cfg(feature = "gpu")]
fn split_graph(
    gpu: &Gpu,
    flash: &FlashKernels,
    inputs: FlashInputs<'_>,
    part: &mut Partials,
    y: &mut DeviceBuffer<f32>,
) -> Result<Graph, GpuError> {
    let stream = gpu.stream();
    gpu.capture(|_s| {
        flash
            .enqueue_flash_latent_split(
                stream,
                FlashSplitArgs {
                    inputs,
                    part_v: &mut part.v,
                    part_ms: &mut part.ms,
                    y,
                },
            )
            .map(|_| ())
    })
}

// ------------------------------------------------------------ real layers

/// One real layer's dump tensors, each proved against the manifest and the
/// CONCAT orders proved bit-exact against their operands.
#[cfg(feature = "gpu")]
struct RealChains {
    /// `kvr-L` `[width, tokens]`: `[k_rope | kv_compressed]` per token.
    kvr: Vec<f32>,
    /// `q-L` occ 1 `[width, tokens, heads]`, rows `t + tokens*h`.
    q1: Vec<f32>,
    /// `kqv_compressed-L` `[latent, heads, tokens]`, the oracle's output.
    kqv: Vec<f32>,
}

/// One layer's real-input cases: the cache build, the five m=1 decode
/// positions, the m=TOKENS causal prefill shape, and (layer 0 only) the
/// captured-graph checks.
#[cfg(feature = "gpu")]
fn real_layer(gpu: &Gpu, flash: &FlashKernels, case: &RealCase<'_>) -> Result<bool, GateError> {
    let stream = gpu.stream();
    // The split launch's partials, sized for the widest shape below (the
    // m=tokens prefill) over this cache height. At ctx_max = 64 the height
    // is one segment, so every call here takes the single-launch fast path
    // and leaves them untouched — that choice is itself asserted by the
    // graph node count.
    let mut part = Partials::zeroed(stream, case.tokens * case.dims.heads, case.ctx_max)?;
    let chains = RealChains {
        kvr: kvr_chain(case)?,
        q1: q_chain(case)?,
        kqv: kqv_chain(case)?,
    };
    let (cache, mut ok) = real_append(flash, stream, case, &chains.kvr)?;
    ok &= real_decode(gpu, flash, case, &chains, &cache, &mut part)?;
    ok &= real_prefill(flash, stream, case, &chains, &cache, &mut part)?;
    if case.layer == 0 {
        ok &= graph_check_append(gpu, flash, &chains.kvr[..case.dims.width])?;
    }
    Ok(ok)
}

/// kvr-L [width, tokens] CONCAT of [k_rope | kv_compressed]; k_rope occ 1
/// is (d, t), kv_compressed occ 1 is (d, t).
#[cfg(feature = "gpu")]
fn kvr_chain(case: &RealCase<'_>) -> Result<Vec<f32>, GateError> {
    let RealCase {
        man,
        layer: l,
        dims,
        tokens,
        ..
    } = *case;
    let Dims {
        width,
        rope_dims: rope,
        latent_dims: latent,
        ..
    } = dims;
    let kvr_row = find_ref_row(man, &format!("kvr-{l}"), 0)?;
    kvr_row.expect(
        "kvr_chain",
        "f32",
        [width as u64, tokens as u64, 1, 1],
        "CONCAT",
    )?;
    let kvr = ref_tensor_of(kvr_row)?;
    let k_rope_row = find_ref_row(man, &format!("k_rope-{l}"), 1)?;
    let kv_compressed_row = find_ref_row(man, &format!("kv_compressed-{l}"), 1)?;
    k_rope_row.expect(
        "kvr_chain",
        "f32",
        [rope as u64, 1, tokens as u64, 1],
        "ROPE",
    )?;
    kv_compressed_row.expect(
        "kvr_chain",
        "f32",
        [latent as u64, tokens as u64, 1, 1],
        "FUSED_RMS_NORM",
    )?;
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
    Ok(kvr)
}

/// q-L occ 1 [width, tokens, heads] CONCAT of [q_rope | q_nope2]; rows
/// (t + tokens*h); q_rope occ 1 is (d, h, t), q_nope2 occ 0 is (d, t, h).
#[cfg(feature = "gpu")]
fn q_chain(case: &RealCase<'_>) -> Result<Vec<f32>, GateError> {
    let RealCase {
        man,
        layer: l,
        dims,
        tokens,
        ..
    } = *case;
    let Dims {
        width,
        rope_dims: rope,
        latent_dims: latent,
        heads: n_heads,
        ..
    } = dims;
    let q_row = find_ref_row(man, &format!("q-{l}"), 1)?;
    q_row.expect(
        "q_chain",
        "f32",
        [width as u64, tokens as u64, n_heads as u64, 1],
        "CONCAT",
    )?;
    let q1 = ref_tensor_of(q_row)?;
    let q_rope_row = find_ref_row(man, &format!("q_rope-{l}"), 1)?;
    let q_nope2_row = find_ref_row(man, &format!("q_nope2-{l}"), 0)?;
    q_rope_row.expect(
        "q_chain",
        "f32",
        [rope as u64, n_heads as u64, tokens as u64, 1],
        "ROPE",
    )?;
    q_nope2_row.expect(
        "q_chain",
        "f32",
        [latent as u64, tokens as u64, n_heads as u64, 1],
        "MUL_MAT",
    )?;
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
    Ok(q1)
}

/// kqv_compressed-L [latent, heads, tokens] FLASH_ATTN_EXT.
#[cfg(feature = "gpu")]
fn kqv_chain(case: &RealCase<'_>) -> Result<Vec<f32>, GateError> {
    let RealCase {
        man,
        layer: l,
        dims,
        tokens,
        ..
    } = *case;
    let (latent, n_heads) = (dims.latent_dims, dims.heads);
    let kqv_row = find_ref_row(man, &format!("kqv_compressed-{l}"), 0)?;
    kqv_row.expect(
        "kqv_chain",
        "f32",
        [latent as u64, n_heads as u64, tokens as u64, 1],
        "FLASH_ATTN_EXT",
    )?;
    ref_tensor_of(kqv_row)
}

/// One real layer's f16 cache as [`real_append`] built it: on the device
/// for the kernels, and its bits as read back for the reference.
#[cfg(feature = "gpu")]
struct LayerCache {
    dev: DeviceTensor<u16>,
    bits: Vec<u16>,
}

/// kv_append: build THIS layer's cache from the kvr rows, and check it.
/// Returns the cache and the verdict.
#[cfg(feature = "gpu")]
fn real_append(
    flash: &FlashKernels,
    stream: &CudaStream,
    case: &RealCase<'_>,
    kvr: &[f32],
) -> Result<(LayerCache, bool), GateError> {
    let RealCase {
        man,
        layer: l,
        dims,
        tokens,
        ctx_max,
        ..
    } = *case;
    let width = dims.width;
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
    ik_cache_row.expect("real_append", "f16", [width as u64, 256, 1, 1], "VIEW")?;
    let ik_bits = widened_f16_bits(ik_cache_row, tokens)?;
    let ik_cache_bits_equal = bits1[..tokens * width] == ik_bits[..];
    // The ik column is printed, not asserted (the package's gate rule): the
    // asserted layer is bit equality with the oracle's own conversion.
    let pass = bits_exact && neighbours && rerun_same;
    println!(
        "shape op=kv_append layer={l} rows={tokens} width={width} bits_exact={bits_exact} ik_cache_bits_equal={ik_cache_bits_equal} neighbours_untouched={neighbours} bit_identical_rerun={rerun_same} {}",
        verdict(pass)
    );
    Ok((
        LayerCache {
            dev: cache,
            bits: bits1,
        },
        pass,
    ))
}

/// flash m=1: five decode positions (one query over a causal prefix), the
/// cache built by [`real_append`] supplying the f16 keys. On layer 0 the
/// last position is also the eager run [`graph_check_flash`] replays.
#[cfg(feature = "gpu")]
fn real_decode(
    gpu: &Gpu,
    flash: &FlashKernels,
    case: &RealCase<'_>,
    chains: &RealChains,
    cache: &LayerCache,
    part: &mut Partials,
) -> Result<bool, GateError> {
    let RealCase {
        layer: l,
        dims,
        band,
        tokens,
        ..
    } = *case;
    let Dims {
        kq_scale: scale,
        width,
        latent_dims: latent,
        heads: n_heads,
        ..
    } = dims;
    let stream = gpu.stream();
    let mut ok = true;
    for t in 1..tokens {
        let n_keys = t + 1;
        let mut qbuf = vec![0.0f32; n_heads * width];
        for h in 0..n_heads {
            qbuf[h * width..(h + 1) * width].copy_from_slice(
                &chains.q1[(t + tokens * h) * width..(t + tokens * h + 1) * width],
            );
        }
        let q_dev = DeviceBuffer::from_host(stream, &qbuf)?;
        let n_keys_dev = DeviceBuffer::from_host(stream, &[n_keys as u32])?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
        let inputs = FlashInputs {
            q: &q_dev,
            kv: &cache.dev,
            n_keys_buf: &n_keys_dev,
            kq_scale: scale,
            geom: dims.geom(1),
        };
        let (y1, y2) = split_twice(flash, stream, inputs, part, &mut y_dev)?;
        let rerun_same = bits_equal(&y1, &y2);
        let y_ref = flash_f64_ref(&qbuf, &cache.bits, n_keys, scale, dims.geom(1));
        let rel = max_rel_err(&y1, &y_ref)?;
        let ik = ik_kqv_rows(&chains.kqv, t, n_heads, latent);
        let ik_rel = max_rel_err(&y1, &ik)?;
        let pass = rel <= band && rerun_same;
        println!(
            "shape op=flash_latent layer={l} t={t} n_keys={n_keys} m=1 max_rel_err={rel:.3e} bit_identical_rerun={rerun_same} ik_rel={ik_rel:.3e} {}",
            verdict(pass)
        );
        ok &= pass;
        if l == 0 && t == tokens - 1 {
            let case = GraphFlashCase {
                inputs,
                y_eager: &y1,
            };
            ok &= graph_check_flash(gpu, flash, &case)?;
        }
    }
    Ok(ok)
}

/// flash m=tokens: the causal prefill shape, all heads and tokens in one
/// launch (the m<=8 variant; [`real_decode`] is the m=1 decode shape).
#[cfg(feature = "gpu")]
fn real_prefill(
    flash: &FlashKernels,
    stream: &CudaStream,
    case: &RealCase<'_>,
    chains: &RealChains,
    cache: &LayerCache,
    part: &mut Partials,
) -> Result<bool, GateError> {
    let RealCase {
        layer: l,
        dims,
        band,
        tokens,
        ..
    } = *case;
    let Dims {
        kq_scale: scale,
        width,
        latent_dims: latent,
        heads: n_heads,
        ..
    } = dims;
    let mut qbuf = vec![0.0f32; tokens * n_heads * width];
    for t in 0..tokens {
        for h in 0..n_heads {
            let src_base = (t + tokens * h) * width;
            let dst_base = (t * n_heads + h) * width;
            qbuf[dst_base..dst_base + width]
                .copy_from_slice(&chains.q1[src_base..src_base + width]);
        }
    }
    let q_dev = DeviceBuffer::from_host(stream, &qbuf)?;
    let n_keys_dev = DeviceBuffer::from_host(stream, &[tokens as u32])?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, tokens * n_heads * latent)?;
    let inputs = FlashInputs {
        q: &q_dev,
        kv: &cache.dev,
        n_keys_buf: &n_keys_dev,
        kq_scale: scale,
        geom: dims.geom(tokens),
    };
    let (y1, y2) = split_twice(flash, stream, inputs, part, &mut y_dev)?;
    let rerun_same = bits_equal(&y1, &y2);
    let y_ref = flash_f64_ref(&qbuf, &cache.bits, tokens, scale, dims.geom(tokens));
    let rel = max_rel_err(&y1, &y_ref)?;
    let mut ik = vec![0.0f32; tokens * n_heads * latent];
    for (t, chunk) in ik.chunks_mut(n_heads * latent).enumerate() {
        chunk.copy_from_slice(&ik_kqv_rows(&chains.kqv, t, n_heads, latent));
    }
    let ik_rel = max_rel_err(&y1, &ik)?;
    let pass = rel <= band && rerun_same;
    println!(
        "shape op=flash_latent layer={l} m={tokens} n_keys={tokens} causal=prefill max_rel_err={rel:.3e} bit_identical_rerun={rerun_same} ik_rel={ik_rel:.3e} {}",
        verdict(pass)
    );
    Ok(pass)
}

// ----------------------------------------------------------- depth cases

/// The depth cases' shared inputs: LCG queries, and LCG keys as f16 bits in
/// a cache whose every row past the keys holds the f16 NaN pattern — on the
/// host (for the reference) and on the device.
#[cfg(feature = "gpu")]
struct DepthInputs {
    queries: Vec<f32>,
    cache_bits: Vec<u16>,
    cache: DeviceTensor<u16>,
    q_dev: DeviceBuffer<f32>,
}

#[cfg(feature = "gpu")]
impl DepthInputs {
    fn new(stream: &CudaStream, case: &DepthCase) -> Result<Self, GateError> {
        let (width, n_heads) = (case.dims.width, case.dims.heads);
        let keys = activations(width, 4096, 50021);
        let queries = activations(width, n_heads, 60013);
        let mut cache_bits = vec![0x7e00u16; case.cache_rows * width]; // NaN pad everywhere
        for (b, &v) in cache_bits.iter_mut().zip(&keys) {
            *b = f32_to_f16_bits(v);
        }
        let cache = DeviceTensor::upload(stream, &cache_bits, case.cache_rows, width)?;
        let q_dev = DeviceBuffer::from_host(stream, &queries)?;
        Ok(Self {
            queries,
            cache_bits,
            cache,
            q_dev,
        })
    }

    /// The m=1 launch inputs over this cache, the live key count read from
    /// `n_keys_buf`.
    fn inputs<'a>(&'a self, n_keys_buf: &'a DeviceBuffer<u32>, dims: Dims) -> FlashInputs<'a> {
        FlashInputs {
            q: &self.q_dev,
            kv: &self.cache,
            n_keys_buf,
            kq_scale: dims.kq_scale,
            geom: dims.geom(1),
        }
    }
}

/// The key counts the depth cases run at. 1/31/32/33 are the key-tile
/// edges; seg-1/seg/seg+1 the segment edges a split launch adds; 1 and 31
/// are also `n_keys` far below one segment with a tall cache. 4096 is a
/// whole number of 256-key segments and leaves the last segment of the
/// 4160-row cache wholly empty.
#[cfg(feature = "gpu")]
fn depth_key_counts(cache_rows: usize) -> Vec<usize> {
    let seg = seg_keys();
    let mut cases = vec![1usize, 31, 32, 33, 1000, 4096];
    for extra in [seg - 1, seg, seg + 1, 2 * seg] {
        if extra < cache_rows && !cases.contains(&extra) {
            cases.push(extra);
        }
    }
    cases.sort_unstable();
    cases
}

/// What the split path measured at one key count.
#[cfg(feature = "gpu")]
struct SplitDepth {
    y1: Vec<f32>,
    rel: f32,
    one_rel: f32,
    cross_rel: f32,
    rerun_same: bool,
    replay_same: bool,
}

/// What the tensor-core path measured at one key count.
#[cfg(feature = "gpu")]
struct MmaDepth {
    rel: f32,
    cross: f32,
    rerun_same: bool,
    replay_same: bool,
}

/// Synthetic LCG keys/queries at the 32-key block edges, every row past
/// `n_keys` holding the f16 NaN bit pattern.
#[cfg(feature = "gpu")]
fn depth_cases(gpu: &Gpu, flash: &FlashKernels, case: &DepthCase) -> Result<bool, GateError> {
    let DepthCase {
        dims,
        cache_rows: depth_rows,
        band,
        mma_band,
    } = *case;
    let (n_heads, latent) = (dims.heads, dims.latent_dims);
    let stream = gpu.stream();
    let d = DepthInputs::new(stream, case)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    // This cache is tall enough to be cut into segments, so every case below
    // runs the split launch. `segs` counts the whole cache, not the live
    // keys: at small `n_keys` most segments are past the causal limit and
    // must write their neutral partial without reading a row — every one of
    // those rows holds the f16 NaN pattern, so a segment that read one would
    // fail the finiteness check inside `max_rel_err`.
    let segs = segments_for(depth_rows);
    let seg = seg_keys();
    let mut part = Partials::zeroed(stream, n_heads, depth_rows)?;
    let mut y_one = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
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
    let mut part_m = Partials::zeroed(stream, n_heads, depth_rows)?;
    let mut y_m = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    let graph = split_graph(
        gpu,
        flash,
        d.inputs(&n_keys_dev, dims),
        &mut part,
        &mut y_dev,
    )?;
    // The tensor-core pass's own graph, captured at the same one live
    // segment: its grid comes from the cache height and the head count, so a
    // replay must follow `n_keys_buf` across every segment boundary too.
    let graph_m = gpu.capture(|_s| {
        mma_enqueue(
            flash,
            stream,
            d.inputs(&n_keys_dev, dims),
            depth_rows,
            &mut part_m,
            &mut y_m,
        )
    })?;
    let lines = DepthLines {
        cache_rows: depth_rows,
        seg,
        segs,
        groups: mma_groups(n_heads),
        band,
        mma_band,
        graph_nodes: graph.node_count(),
        graph_m_nodes: graph_m.node_count(),
    };
    let mut ok = true;
    for n_keys in depth_key_counts(depth_rows) {
        n_keys_dev.copy_from_host(stream, &[n_keys as u32])?;
        let inputs = d.inputs(&n_keys_dev, dims);
        let y_ref = flash_f64_ref(
            &d.queries,
            &d.cache_bits,
            n_keys,
            dims.kq_scale,
            inputs.geom,
        );
        let s = depth_split(
            flash,
            stream,
            inputs,
            &graph,
            (&mut part, &mut y_dev, &mut y_one),
            &y_ref,
        )?;
        let m = depth_mma(
            flash,
            stream,
            inputs,
            &graph_m,
            (&mut part_m, &mut y_m),
            &y_ref,
            &s.y1,
        )?;
        ok &= depth_report(&lines, n_keys, &s, &m);
    }
    Ok(ok)
}

/// What the two depth verdict lines print besides one key count's results:
/// the cache and segment geometry, the bands, and the two graphs' node
/// counts — fixed for the whole run.
#[cfg(feature = "gpu")]
struct DepthLines {
    cache_rows: usize,
    seg: usize,
    segs: usize,
    groups: usize,
    band: f32,
    mma_band: f32,
    graph_nodes: usize,
    graph_m_nodes: usize,
}

/// The two verdict lines of one key count — the tensor-core pass first, the
/// split path second, the order the log has always had — and their joint
/// verdict. Both lines print whatever the first one says.
#[cfg(feature = "gpu")]
fn depth_report(p: &DepthLines, n_keys: usize, s: &SplitDepth, m: &MmaDepth) -> bool {
    let DepthLines {
        cache_rows: depth_rows,
        seg,
        segs,
        groups,
        band,
        mma_band,
        graph_nodes,
        graph_m_nodes,
    } = *p;
    let pass_m = m.rel <= mma_band && m.cross <= mma_band && m.rerun_same && m.replay_same;
    println!(
        "shape op=flash_latent_mma_depth n_keys={n_keys} m=1 seg_keys={seg} segs={segs} groups={groups} band={mma_band:.3e} max_rel_err={:.3e} mma_vs_seg={:.3e} bit_identical_rerun={} graph_nodes={graph_m_nodes} replay_bit_identical={} {}",
        m.rel,
        m.cross,
        m.rerun_same,
        m.replay_same,
        verdict(pass_m)
    );
    let pass =
        s.rel <= band && s.one_rel <= band && s.cross_rel <= band && s.rerun_same && s.replay_same;
    println!(
        "shape op=flash_latent_depth n_keys={n_keys} m=1 seg_keys={seg} segs={segs} live_segs={} nan_pad_rows={} max_rel_err={:.3e} single_launch_rel={:.3e} split_vs_single={:.3e} bit_identical_rerun={} graph_nodes={graph_nodes} replay_bit_identical={} {}",
        n_keys.div_ceil(seg),
        depth_rows - n_keys,
        s.rel,
        s.one_rel,
        s.cross_rel,
        s.rerun_same,
        s.replay_same,
        verdict(pass)
    );
    pass_m & pass
}

/// The split path at one key count: two eager runs, the single-block entry
/// on the same inputs, and a replay of the one captured graph.
#[cfg(feature = "gpu")]
fn depth_split(
    flash: &FlashKernels,
    stream: &CudaStream,
    inputs: FlashInputs<'_>,
    graph: &Graph,
    (part, y_dev, y_one): (
        &mut Partials,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    y_ref: &[f32],
) -> Result<SplitDepth, GateError> {
    let (y1, y2) = split_twice(flash, stream, inputs, part, y_dev)?;
    let rerun_same = bits_equal(&y1, &y2);
    let rel = max_rel_err(&y1, y_ref)?;
    // The single-block entry on the same inputs: the path a cache short
    // enough to hold one segment takes. Both must land inside the band
    // against the f64 reference, and so must their distance to each
    // other — the split moves the summation order, nothing else.
    flash.enqueue_flash_latent(
        stream,
        FlashLatentArgs {
            inputs,
            y: &mut *y_one,
        },
    )?;
    stream.synchronize()?;
    let y_single = y_one.to_host_vec(stream)?;
    let one_rel = max_rel_err(&y_single, y_ref)?;
    let cross_rel = max_rel_err(&y1, &y_single)?;
    graph.launch(stream)?;
    stream.synchronize()?;
    let y_graph = y_dev.to_host_vec(stream)?;
    let replay_same = bits_equal(&y1, &y_graph);
    Ok(SplitDepth {
        y1,
        rel,
        one_rel,
        cross_rel,
        rerun_same,
        replay_same,
    })
}

/// The tensor-core segment pass and the merge after it, enqueued (not
/// synchronized) — the body of both the eager runs and the captured graph.
#[cfg(feature = "gpu")]
fn mma_enqueue(
    flash: &FlashKernels,
    stream: &CudaStream,
    inputs: FlashInputs<'_>,
    cache_rows: usize,
    part: &mut Partials,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    flash.enqueue_flash_latent_mma(
        stream,
        FlashSegArgs {
            inputs,
            part_v: &mut part.v,
            part_ms: &mut part.ms,
        },
    )?;
    flash.enqueue_flash_merge(
        stream,
        FlashMergeArgs {
            n_keys_buf: inputs.n_keys_buf,
            cache_rows,
            tokens: inputs.geom.tokens,
            heads: inputs.geom.heads,
            latent_dims: inputs.geom.latent_dims,
            part_v: &part.v,
            part_ms: &part.ms,
            y,
        },
    )
}

/// The tensor-core pass at one key count, through the same merge. Its query
/// rows and products are f16, so it carries its own band (`mma_band`)
/// against the f64 reference and against the one-row pass (`y_split`);
/// reruns and graph replays of it are still bit-identical.
#[cfg(feature = "gpu")]
fn depth_mma(
    flash: &FlashKernels,
    stream: &CudaStream,
    inputs: FlashInputs<'_>,
    graph_m: &Graph,
    (part, y_m): (&mut Partials, &mut DeviceBuffer<f32>),
    y_ref: &[f32],
    y_split: &[f32],
) -> Result<MmaDepth, GateError> {
    let cache_rows = inputs.kv.rows();
    let mut run = || -> Result<Vec<f32>, GateError> {
        mma_enqueue(flash, stream, inputs, cache_rows, &mut *part, &mut *y_m)?;
        stream.synchronize()?;
        Ok(y_m.to_host_vec(stream)?)
    };
    let m1 = run()?;
    let m2 = run()?;
    let rerun_same = bits_equal(&m1, &m2);
    let rel = max_rel_err(&m1, y_ref)?;
    let cross = max_rel_err(&m1, y_split)?;
    graph_m.launch(stream)?;
    stream.synchronize()?;
    let m_graph = y_m.to_host_vec(stream)?;
    let replay_same = bits_equal(&m1, &m_graph);
    Ok(MmaDepth {
        rel,
        cross,
        rerun_same,
        replay_same,
    })
}

// ----------------------------------------------------------- IEEE edges

/// The conversion op's exactness on the edges the real rows cannot reach:
/// zeros, signed zero, subnormals on both sides, the f16 finite boundary,
/// overflow, and the NaN class (which collapses to inf, the oracle's rule).
#[cfg(feature = "gpu")]
fn edge_values(flash: &FlashKernels, stream: &CudaStream) -> Result<bool, GateError> {
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
    Ok(pass)
}

// --------------------------------------------------------------- graphs

/// Flash inside a captured graph: replay must be byte-identical to eager.
#[cfg(feature = "gpu")]
fn graph_check_flash(
    gpu: &Gpu,
    flash: &FlashKernels,
    case: &GraphFlashCase<'_>,
) -> Result<bool, GateError> {
    let GraphFlashCase { inputs, y_eager } = *case;
    let (cache, n_heads, latent) = (inputs.kv, inputs.geom.heads, inputs.geom.latent_dims);
    let stream = gpu.stream();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_heads * latent)?;
    let segs = segments_for(cache.rows());
    let mut part = Partials::zeroed(stream, n_heads, cache.rows())?;
    let graph = split_graph(gpu, flash, inputs, &mut part, &mut y_dev)?;
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
    Ok(pass)
}

/// The pos-buffer append inside a captured graph: rewriting the buffer
/// between replays moves the landing rows — the form a captured decode step
/// needs. The scalar-pos append captured beside it stays frozen at its
/// capture-time row, which is why the step cannot use it.
#[cfg(feature = "gpu")]
fn graph_check_append(gpu: &Gpu, flash: &FlashKernels, one_row: &[f32]) -> Result<bool, GateError> {
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
    Ok(pass)
}

// -------------------------------------------------------------- helpers

/// `MlaParams::read`'s `kq_scale` (crates/model/src/attn.rs: mscale =
/// 1 + log_mul·ln(1/freq_scale), kq_scale = mscale²/√key_length — the YaRN
/// mscale is inside it, not 1/√d). Mirrored because gpu-gates has no edge to
/// bloomery-model; the `ik_rel` columns are what prove the value against the
/// oracle.
#[cfg(feature = "gpu")]
fn kq_scale_of(gguf: &Gguf) -> Result<f32, GateError> {
    let f32v = |key: &str| -> Result<f32, GateError> {
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
/// `Σ w_i·v_i[d] / Σ w_i` cast once to f32. Query `t` of `geom.tokens`
/// attends to keys `0..n_keys − tokens + t + 1` (the causal prefix the kernel
/// uses).
#[cfg(feature = "gpu")]
fn flash_f64_ref(
    q: &[f32],
    keys16: &[u16],
    n_keys: usize,
    scale: f32,
    geom: FlashGeom,
) -> Vec<f32> {
    let FlashGeom {
        tokens: m,
        heads: n_heads,
        rope_dims: rope,
        latent_dims: latent,
    } = geom;
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
