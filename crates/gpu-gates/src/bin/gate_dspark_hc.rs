//! GPU gate for the DSpark draft's two kernels the target lacks
//! (`bloomery_gpu_deepseek41::hc_f32` and `::markov`) and the common head
//! chain over the Markov kernel, on one card.
//!
//! 1. **HC_PRE, F32 weights** (`ds41_hc_pre_f32`): block 0's `hc_attn_fn` and
//!    `hc_ffn_fn` against pseudo-random streams in ±2 for m = 1, 3, 5 tokens;
//!    the kernel's mixes and HC_PRE rows equal the host transcription of the
//!    module doc's rule bit for bit, and a second launch equals the first.
//! 2. **Markov head** (`ds41_markov`): for the first 64 ids of the code corpus
//!    and ids 0, 1 and `n_vocab - 1`, from a zero logits row, the kernel's row
//!    equals `0 + delta` by the host rule bit for bit, and the argmax the step
//!    writes equals the host's first maximum — `markov-accept`'s table entry
//!    for that id (its dot order is the rule's; its self-test pins that order
//!    to its SIMD kernel). Then the chain: three rows from the zero rows and
//!    `id_last` = the 64th id, tokens and rows equal the host chain.
//! 3. **Oracle diagnostics, no pin**: the dsref set `code64_n32_w3`, block 0
//!    — our HC_PRE on ik's layer-0 attention input against ik's mixes and
//!    HC_PRE rows, and our Markov step on ik's pre-Markov logits against ik's
//!    deltas, corrected rows and draft tokens. Printed as max|diff|; they do
//!    not fail the gate.
//! 4. **Reduced Markov head** (`ds41_markov` with the map on), no model file:
//!    seeded tables of 1000 target rows and 128 head rows, an injective map
//!    from head id to target id, base logits of m = 8 rows and a target id
//!    `first`. The reduced step's eight tokens and rows equal the host
//!    transcription of the layout doc's rule bit for bit; the guard words past
//!    the logits (a block's threads beyond `n_head`) are untouched. A previous
//!    head id at or past `n_head` raises `FaultSite::TokenId` and makes its
//!    row NaN, the other row unchanged.
//! 5. **Own head over a reduced vocabulary** (`dspark::head::Head`), no model
//!    file: the clause-4 tables behind a draft's own Q8_0 rows (hidden 64,
//!    128 head ids, rows of m = 8). The card's projection logits are read
//!    back, checked against the host's f64 norm and Q8_0 dot within the f32
//!    bound, and taken through the host's reduced Markov loop and argmax;
//!    `tokens` and `tokens_p` return the host's head ids, the corrected
//!    logits equal the host's bit for bit, and the probabilities equal the
//!    host transcription of `argmax_p_rows_fault`'s softmax.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_hc", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};

    use bloomery_gpu::dspark::head::{Head, HeadWeights, ReducedMarkovWeights};
    use bloomery_gpu::dspark::{HeadSource, Q8Rows};
    use bloomery_gpu::{DeviceTensor, FAULT_NONE, Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_deepseek41::hc::HC_MIX;
    use bloomery_gpu_deepseek41::hc_f32::{HcF32Args, HcF32Kernels, HcF32Params};
    use bloomery_gpu_deepseek41::markov::{
        MARKOV_RANK, MARKOV_ROW_WORDS, MarkovKernels, MarkovWeights,
    };
    use bloomery_gpu_gates::hc_host::{self, exp_ours, hc_pre_f32};
    use bloomery_gpu_gates::rounding::{butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, bits_equal, bytes_to_words, checks_failed, data_dir, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, f32_to_f16_bits, half_to_f32};
    use model::arch::dspark::{DraftHparams, names};
    use refset::arch::deepseek41::{DSREF, DSREF_SET};
    use refset::dsref::{DsRow, Dsref, Graph};

    const NAME: &str = "gate_dspark_hc";
    /// Corpus ids the Markov checks take, then the three edge ids.
    const CORPUS_IDS: usize = 64;
    /// Rows of the chained Markov check (the dsref set's block width).
    const CHAIN: usize = 3;
    /// The reduced head's synthetic shapes: target rows of `w1`, head rows of
    /// `w2` and the logits, and the rows of its chained step.
    const RED_PREV: usize = 1000;
    const RED_HEAD: usize = 128;
    const RED_M: usize = 8;
    /// Rows past `n_head` the reduced check's logits carry: the rest of the
    /// kernel's block of 256 threads, whose writes would land there.
    const RED_GUARD_ROWS: usize = 256 - RED_HEAD;
    // The host rule's layout is the kernel's.
    const _: () = assert!(hc_host::HC_MIX == HC_MIX);

    // ------------------------------------------------------------ the file

    struct Hc {
        k: usize,
        w: Vec<f32>,
        scale: Vec<f32>,
        base: Vec<f32>,
    }

    fn tensor_bytes<'a>(
        split: &'a Split,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
    ) -> Result<&'a [u8], GateError> {
        let (shard, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the draft"))?;
        if t.ty != ty || t.dims != dims {
            return Err(
                format!("{name}: {:?} {:?}, expected {ty:?} {dims:?}", t.ty, t.dims).into(),
            );
        }
        Ok(split
            .shard(shard)
            .ok_or("a tensor names a shard the split does not have")?
            .data(t)?)
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn read_hc(split: &Split, hp: &DraftHparams, sub: &str) -> Result<Hc, GateError> {
        let k = hp.hc.streams * hp.n_embd;
        let (fname, sname, bname) = match sub {
            "attn" => (
                names::hc_attn_fn(0),
                names::hc_attn_scale(0),
                names::hc_attn_base(0),
            ),
            _ => (
                names::hc_ffn_fn(0),
                names::hc_ffn_scale(0),
                names::hc_ffn_base(0),
            ),
        };
        let mix = HC_MIX as u64;
        Ok(Hc {
            k,
            w: f32s(tensor_bytes(
                split,
                &fname,
                GgmlType::F32,
                &[k as u64, mix],
            )?),
            scale: f32s(tensor_bytes(split, &sname, GgmlType::F32, &[3])?),
            base: f32s(tensor_bytes(split, &bname, GgmlType::F32, &[mix])?),
        })
    }

    // ---------------------------------------------- HC_PRE, the host rule

    /// One lane's `f32_gemv` partial: `x[32*it + lane] * w[32*it + lane]` by
    /// `mul_add` from 0, `it` ascending.
    fn gemv_lane(w: &[f32], x: &[f32], lane: usize) -> f32 {
        let mut f = 0.0f32;
        let mut i = lane;
        while i < x.len() {
            f = w[i].mul_add(x[i], f);
            i += 32;
        }
        f
    }

    /// The module doc's rule for `m` tokens: `(mixes, hc)`, [`HC_MIX`] each per token.
    fn hc_pre_f32_host(
        p: &Hc,
        x: &[f32],
        m: usize,
        rms_eps: f32,
        eps: f32,
        iters: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let k = p.k;
        let sc = [p.scale[0], p.scale[1], p.scale[2]];
        let (mut mixes, mut hc) = (Vec::new(), Vec::new());
        for t in 0..m {
            let xt = &x[t * k..(t + 1) * k];
            let sq = butterfly(std::array::from_fn(|lane| {
                let mut a = 0.0f32;
                let mut i = lane;
                while i < k {
                    a = xt[i].mul_add(xt[i], a);
                    i += 32;
                }
                a
            }));
            let scale = 1.0 / (sq / k as f32 + rms_eps).sqrt();
            let mix: Vec<f32> = (0..HC_MIX)
                .map(|r| {
                    let w = &p.w[r * k..(r + 1) * k];
                    butterfly(std::array::from_fn(|lane| gemv_lane(w, xt, lane))) * scale
                })
                .collect();
            hc.extend(hc_pre_f32(&mix, sc, &p.base, eps, iters, exp_ours));
            mixes.extend(mix);
        }
        (mixes, hc)
    }

    /// Streams in ±2 from a seeded LCG (Numerical Recipes' constants).
    fn streams(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0, f64::max)
    }

    fn max_abs(a: &[f32]) -> f64 {
        a.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max)
    }

    // ---------------------------------------------- Markov, the host rule

    /// Token `p`'s bf16 row of `w` widened (`bits << 16`), [`MARKOV_RANK`] values.
    fn row_f32(w: &[u32], p: usize) -> Vec<f32> {
        w[p * MARKOV_ROW_WORDS..(p + 1) * MARKOV_ROW_WORDS]
            .iter()
            .flat_map(|&x| [f32::from_bits(x << 16), f32::from_bits(x & 0xffff_0000)])
            .collect()
    }

    /// `markov-accept`'s `scalar_dot`: eight lanes by `mul_add` over the
    /// octets ascending, then `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))`.
    fn scalar_dot(w: &[f32], e: &[f32]) -> f32 {
        let mut a = [0.0f32; 8];
        for (wc, ec) in w.as_chunks::<8>().0.iter().zip(e.as_chunks::<8>().0) {
            for l in 0..8 {
                a[l] = wc[l].mul_add(ec[l], a[l]);
            }
        }
        ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
    }

    /// The delta row of previous token `p`, `n_vocab` values, `w2` widened once.
    fn delta_host(w1: &[u32], w2f: &[f32], p: usize) -> Vec<f32> {
        let e = row_f32(w1, p);
        w2f.as_chunks::<MARKOV_RANK>()
            .0
            .iter()
            .map(|w| scalar_dot(w, &e))
            .collect()
    }

    /// `x` rounded to the nearest bf16, ties to even (finite `x`).
    fn bf16_round(x: f32) -> f32 {
        let b = x.to_bits();
        f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
    }

    /// The first maximum (lowest index on a tie), `ggml_argmax`'s rule.
    fn argmax(v: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best as u32
    }

    fn read_ids(path: &Path, n: usize) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids: Vec<u32> = text
            .lines()
            .take(n)
            .map(|l| l.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if ids.len() < n {
            return Err(format!("{} holds fewer than {n} ids", path.display()).into());
        }
        Ok(ids)
    }

    // ------------------------------------------------------------- checks

    struct Ctx<'a> {
        gpu: &'a Gpu,
        hck: &'a HcF32Kernels,
        mk: &'a MarkovKernels,
        hp: &'a DraftHparams,
        ok: bool,
    }

    fn run_hc_kernel(
        cx: &Ctx<'_>,
        p: &Hc,
        x: &[f32],
        m: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let s = cx.gpu.stream();
        let w = DeviceTensor::upload(s, &p.w, HC_MIX, p.k)?;
        let scale = DeviceBuffer::from_host(s, &p.scale)?;
        let base = DeviceBuffer::from_host(s, &p.base)?;
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut mixes = DeviceBuffer::zeroed(s, HC_MIX * m)?;
        let mut hc = DeviceBuffer::zeroed(s, HC_MIX * m)?;
        let params = HcF32Params {
            w: &w,
            scale: &scale,
            base: &base,
            eps: cx.hp.hc.eps,
            iters: u32::try_from(cx.hp.hc.sinkhorn_iters)?,
        };
        let args = HcF32Args {
            params: &params,
            x: &xd,
            tokens: m,
            rms_eps: cx.hp.rms_eps,
        };
        cx.hck.enqueue_pre(s, &args, &mut mixes, &mut hc)?;
        s.synchronize()?;
        Ok((mixes.to_host_vec(s)?, hc.to_host_vec(s)?))
    }

    fn check_hc(cx: &mut Ctx<'_>, p: &Hc, sub: &str) -> Result<(), GateError> {
        let iters = u32::try_from(cx.hp.hc.sinkhorn_iters)?;
        for (m, seed) in [(1usize, 11u32), (3, 12), (5, 13)] {
            let x = streams(p.k * m, seed);
            let (km, kh) = run_hc_kernel(cx, p, &x, m)?;
            let (km2, kh2) = run_hc_kernel(cx, p, &x, m)?;
            let (hm, hh) = hc_pre_f32_host(p, &x, m, cx.hp.rms_eps, cx.hp.hc.eps, iters);
            let same_m = bits_equal(&km, &hm);
            let same_h = bits_equal(&kh, &hh);
            let rerun = bits_equal(&km, &km2) && bits_equal(&kh, &kh2);
            let pass = same_m && same_h && rerun;
            println!(
                "hc_pre_f32 site=blk0.{sub} m={m} K={} mixes_bit_identical={same_m} \
                 hc_bit_identical={same_h} (max|diff| mixes {:e} hc {:e}) rerun_bit_identical={rerun} {}",
                p.k,
                max_abs_diff(&km, &hm),
                max_abs_diff(&kh, &hh),
                verdict(pass)
            );
            cx.ok &= pass;
        }
        Ok(())
    }

    struct Markov {
        w1: Vec<u32>,
        w2f: Vec<f32>,
        w1d: DeviceTensor<u32>,
        w2d: DeviceTensor<u32>,
        n_vocab: usize,
    }

    impl Markov {
        fn weights(&self) -> MarkovWeights<'_> {
            MarkovWeights {
                w1: &self.w1d,
                w2: &self.w2d,
            }
        }
    }

    /// Our chain on the card from `logits0` (`m` rows, `logits[v*m + c]`) and
    /// `first`: the tokens and the corrected rows.
    fn run_chain(
        cx: &Ctx<'_>,
        mk: &Markov,
        logits0: &[f32],
        m: usize,
        first: u32,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        let s = cx.gpu.stream();
        let fd = DeviceBuffer::from_host(s, &[first])?;
        // The tokens, then the fault's word and site mask the step's argmax
        // copies after them.
        let mut tok = DeviceBuffer::<u32>::zeroed(s, m + 2)?;
        let mut logits = DeviceBuffer::from_host(s, logits0)?;
        for row in 0..m {
            cx.mk.enqueue_step(
                s,
                cx.gpu.elem(),
                &mk.weights(),
                &fd,
                &mut tok,
                m,
                row,
                cx.gpu.unlabelled_sink(),
                &mut logits,
            )?;
        }
        s.synchronize()?;
        let mut toks = tok.to_host_vec(s)?;
        let (Some(sites), Some(word)) = (toks.pop(), toks.pop()) else {
            return Err("no fault words after the tokens".into());
        };
        if word != FAULT_NONE || sites != 0 {
            return Err(format!(
                "the Markov chain's fault word is {word:#x} with site mask {sites:#x}, not clean"
            )
            .into());
        }
        Ok((toks, logits.to_host_vec(s)?))
    }

    /// The host chain in the same layout: for row c, `logit + delta(prev)`,
    /// then its first maximum is the next prev.
    fn host_chain(mk: &Markov, logits0: &[f32], m: usize, first: u32) -> (Vec<u32>, Vec<f32>) {
        let mut l = logits0.to_vec();
        let (mut prev, mut toks) = (first as usize, Vec::new());
        for c in 0..m {
            let d = delta_host(&mk.w1, &mk.w2f, prev);
            let mut row = vec![0.0f32; mk.n_vocab];
            for v in 0..mk.n_vocab {
                l[v * m + c] += d[v];
                row[v] = l[v * m + c];
            }
            let t = argmax(&row);
            toks.push(t);
            prev = t as usize;
        }
        (toks, l)
    }

    fn check_markov(cx: &mut Ctx<'_>, mk: &Markov, ids: &[u32]) -> Result<(), GateError> {
        let n = mk.n_vocab;
        let zero = vec![0.0f32; n];
        let mut worst = (0usize, 0usize);
        let (mut rows_ok, mut args_ok) = (0usize, 0usize);
        // Host deltas of every id, in parallel: they are independent.
        let (w1, w2f) = (&mk.w1[..], &mk.w2f[..]);
        let host: Vec<Vec<f32>> = std::thread::scope(|sc| {
            let hs: Vec<_> = ids
                .iter()
                .map(|&id| sc.spawn(move || delta_host(w1, w2f, id as usize)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_default())
                .collect()
        });
        for (&id, d) in ids.iter().zip(&host) {
            let (tok, got) = run_chain(cx, mk, &zero, 1, id)?;
            let want: Vec<f32> = zero.iter().zip(d).map(|(a, b)| a + b).collect();
            let same = bits_equal(&got, &want);
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if differ > worst.1 {
                worst = (id as usize, differ);
            }
            let arg = argmax(&want);
            rows_ok += usize::from(same);
            args_ok += usize::from(tok[0] == arg);
            if !same || tok[0] != arg {
                println!(
                    "markov id={id} row_bit_identical={same} ({differ} of {n} differ, max|diff| {:e}) \
                     argmax kernel {} host {arg} {}",
                    max_abs_diff(&got, &want),
                    tok[0],
                    verdict(false)
                );
            }
        }
        let pass = rows_ok == ids.len() && args_ok == ids.len();
        println!(
            "markov single ids={} (corpus-code first {CORPUS_IDS}, 0, 1, n_vocab-1={}) rows_bit_identical={rows_ok}/{} \
             argmax_equal_to_table_rule={args_ok}/{} worst id {} ({} values differ) {}",
            ids.len(),
            n - 1,
            ids.len(),
            ids.len(),
            worst.0,
            worst.1,
            verdict(pass)
        );
        cx.ok &= pass;

        let first = ids[CORPUS_IDS - 1];
        let zeros = vec![0.0f32; n * CHAIN];
        let (kt, kl) = run_chain(cx, mk, &zeros, CHAIN, first)?;
        let (ht, hl) = host_chain(mk, &zeros, CHAIN, first);
        let pass = kt == ht && bits_equal(&kl, &hl);
        println!(
            "markov chain m={CHAIN} id_last={first} tokens kernel {kt:?} host {ht:?} rows_bit_identical={} {}",
            bits_equal(&kl, &hl),
            verdict(pass)
        );
        cx.ok &= pass;
        Ok(())
    }

    // ------------------------------------- reduced Markov, synthetic tables

    /// The reduced head's tables: `w1` of [`RED_PREV`] rows and `w2` of
    /// [`RED_HEAD`] rows as bf16 words, `w2` widened, the head-to-target map,
    /// and the same on the card (the map with a spare word after it, a valid
    /// target id, so a bound missing in the kernel reads a row and raises
    /// nothing).
    struct Reduced {
        w1: Vec<u32>,
        w2: Vec<u32>,
        w2f: Vec<f32>,
        to_target: Vec<u32>,
        w1d: DeviceTensor<u32>,
        w2d: DeviceTensor<u32>,
        mapd: DeviceBuffer<u32>,
    }

    impl Reduced {
        fn weights(&self) -> ReducedMarkovWeights<'_> {
            ReducedMarkovWeights {
                w1: &self.w1d,
                w2: &self.w2d,
                to_target: &self.mapd,
            }
        }
    }

    /// `rows` rank-long bf16 rows from the seeded stream, as the file's words.
    fn bf16_words(rows: usize, seed: u32) -> Vec<u32> {
        let v: Vec<f32> = streams(rows * MARKOV_RANK, seed)
            .into_iter()
            .map(bf16_round)
            .collect();
        v.as_chunks::<2>()
            .0
            .iter()
            .map(|p| (p[0].to_bits() >> 16) | (p[1].to_bits() & 0xffff_0000))
            .collect()
    }

    /// An injective map of the head ids into the target ids from `RED_HEAD`
    /// up (no head id maps to itself), from the seeded stream: a partial
    /// shuffle of the candidates.
    fn head_to_target(seed: u32) -> Vec<u32> {
        let mut s = seed;
        let mut pool: Vec<u32> = (RED_HEAD as u32..RED_PREV as u32).collect();
        for i in 0..RED_HEAD {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let j = i + (s >> 8) as usize % (pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(RED_HEAD);
        pool
    }

    fn reduced_tables(gpu: &Gpu) -> Result<Reduced, GateError> {
        let s = gpu.stream();
        let w1 = bf16_words(RED_PREV, 41);
        let w2 = bf16_words(RED_HEAD, 42);
        let w2f: Vec<f32> = w2
            .iter()
            .flat_map(|&x| [f32::from_bits(x << 16), f32::from_bits(x & 0xffff_0000)])
            .collect();
        let to_target = head_to_target(43);
        let mut listed = to_target.clone();
        listed.push(0);
        Ok(Reduced {
            w1d: DeviceTensor::upload(s, &w1, RED_PREV, MARKOV_ROW_WORDS)?,
            w2d: DeviceTensor::upload(s, &w2, RED_HEAD, MARKOV_ROW_WORDS)?,
            mapd: DeviceBuffer::from_host(s, &listed)?,
            w1,
            w2,
            w2f,
            to_target,
        })
    }

    /// `-0.0`: any add moves its bits (`-0 + +0 = +0`), so a store the kernel
    /// makes into a guard word shows whatever it adds.
    const GUARD: f32 = -0.0;

    /// The logits `base` (`[v * m + c]`, `n_head` rows) with the guard rows after.
    fn guarded(base: &[f32], m: usize) -> Vec<f32> {
        let mut l = base.to_vec();
        l.resize(base.len() + RED_GUARD_ROWS * m, GUARD);
        l
    }

    fn guard_intact(l: &[f32], m: usize) -> bool {
        l[RED_HEAD * m..]
            .iter()
            .all(|x| x.to_bits() == GUARD.to_bits())
    }

    /// The reduced chain on the card from `base` and `first`: the tokens, the
    /// `n_head` rows and whether the guard rows stayed.
    fn run_reduced_chain(
        cx: &Ctx<'_>,
        rd: &Reduced,
        base: &[f32],
        m: usize,
        first: u32,
    ) -> Result<(Vec<u32>, Vec<f32>, bool), GateError> {
        let s = cx.gpu.stream();
        let fd = DeviceBuffer::from_host(s, &[first])?;
        let mut tok = DeviceBuffer::<u32>::zeroed(s, m + 2)?;
        let mut logits = DeviceBuffer::from_host(s, &guarded(base, m))?;
        for row in 0..m {
            cx.mk.enqueue_step_reduced(
                s,
                cx.gpu.elem(),
                &rd.weights(),
                &fd,
                &mut tok,
                m,
                row,
                cx.gpu.unlabelled_sink(),
                &mut logits,
            )?;
        }
        s.synchronize()?;
        let mut toks = tok.to_host_vec(s)?;
        let (Some(sites), Some(word)) = (toks.pop(), toks.pop()) else {
            return Err("no fault words after the tokens".into());
        };
        if word != FAULT_NONE || sites != 0 {
            return Err(format!(
                "the reduced Markov chain's fault word is {word:#x} with site mask {sites:#x}, not clean"
            )
            .into());
        }
        let mut l = logits.to_host_vec(s)?;
        let guard = guard_intact(&l, m);
        l.truncate(RED_HEAD * m);
        Ok((toks, l, guard))
    }

    /// The host chain of the layout doc's rule: row 0's previous token is the
    /// target id `first`, row `c`'s the target id of row `c - 1`'s argmax.
    fn host_reduced_chain(
        rd: &Reduced,
        base: &[f32],
        m: usize,
        first: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut l = base.to_vec();
        let (mut prev, mut toks) = (first as usize, Vec::new());
        for c in 0..m {
            let d = delta_host(&rd.w1, &rd.w2f, prev);
            let mut row = vec![0.0f32; RED_HEAD];
            for v in 0..RED_HEAD {
                l[v * m + c] += d[v];
                row[v] = l[v * m + c];
            }
            let t = argmax(&row);
            toks.push(t);
            prev = rd.to_target[t as usize] as usize;
        }
        (toks, l)
    }

    fn check_reduced(cx: &mut Ctx<'_>) -> Result<(), GateError> {
        let rd = reduced_tables(cx.gpu)?;
        let base = streams(RED_HEAD * RED_M, 44);
        let first = 777u32;
        let (kt, kl, guard) = run_reduced_chain(cx, &rd, &base, RED_M, first)?;
        let (ht, hl) = host_reduced_chain(&rd, &base, RED_M, first);
        let rows = bits_equal(&kl, &hl);
        let pass = kt == ht && rows && guard;
        println!(
            "markov reduced n_prev={RED_PREV} n_head={RED_HEAD} m={RED_M} first={first}: tokens kernel {kt:?} \
             host {ht:?} rows_bit_identical={rows} guard_rows_intact={guard} {}",
            verdict(pass)
        );
        cx.ok &= pass;

        // A previous head id one past the head's rows, row 1 of two. The
        // map's spare word is a valid target id: only the kernel's bound on
        // the head id raises.
        let s = cx.gpu.stream();
        let m = 2;
        let want = Fault::at(LAYER_NONE, FaultSite::TokenId);
        let base2 = streams(RED_HEAD * m, 45);
        let fd = DeviceBuffer::from_host(s, &[first])?;
        let mut tok = DeviceBuffer::from_host(s, &[RED_HEAD as u32, u32::MAX, u32::MAX, u32::MAX])?;
        let mut logits = DeviceBuffer::from_host(s, &guarded(&base2, m))?;
        let clean = cx.gpu.take_fault()?.is_none();
        cx.mk.enqueue_step_reduced(
            s,
            cx.gpu.elem(),
            &rd.weights(),
            &fd,
            &mut tok,
            m,
            1,
            cx.gpu.unlabelled_sink(),
            &mut logits,
        )?;
        let t = tok.to_host_vec(s)?;
        let word = Fault::from_words(t[m], t[m + 1]);
        let fault = cx.gpu.take_fault()?;
        let l = logits.to_host_vec(s)?;
        let nan = (0..RED_HEAD).filter(|v| l[v * m + 1].is_nan()).count();
        let col0: Vec<f32> = (0..RED_HEAD).map(|v| l[v * m]).collect();
        let want0: Vec<f32> = (0..RED_HEAD).map(|v| base2[v * m]).collect();
        let pass = clean
            && fault == Some(want)
            && word == Some(want)
            && nan == RED_HEAD
            && bits_equal(&col0, &want0)
            && guard_intact(&l, m);
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        println!(
            "markov reduced fault: previous head id {RED_HEAD} (n_head) at row 1 of {m}: raised \"{}\" \
             readback \"{}\" want \"{want}\"; row 1 NaN {nan} of {RED_HEAD}, row 0 unchanged={}, \
             guard_rows_intact={} {}",
            shown(fault),
            shown(word),
            bits_equal(&col0, &want0),
            guard_intact(&l, m),
            verdict(pass)
        );
        cx.ok &= pass;
        Ok(())
    }

    // --------------------------------- the own and reduced Head, synthetic

    /// The own head's width: its Q8_0 rows are this many values long.
    const HEAD_HIDDEN: usize = 64;
    /// The target id row 0 of the whole-head check follows.
    const HEAD_FIRST: u32 = 777;
    /// The own head's norm epsilon.
    const EPS: f32 = 1e-6;

    /// `vals` as Q8_0 rows in `bloomery_gpu::q8f32`'s device layout: per 32
    /// values the scale `d = f16(amax / 127)` and the codes `round_half_even(v
    /// / d)`, four codes to a word (the first in the low byte).
    fn q8_0_rows(vals: &[f32]) -> (Vec<u32>, Vec<u16>) {
        let (mut qs, mut ds) = (Vec::new(), Vec::new());
        for block in vals.as_chunks::<32>().0 {
            let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let d16 = f32_to_f16_bits(amax / 127.0);
            let d = half_to_f32(d16);
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            ds.push(d16);
            qs.extend(
                block.as_chunks::<4>().0.iter().map(|q| {
                    u32::from_le_bytes(q.map(|v| ((v * id).round_ties_even() as i8) as u8))
                }),
            );
        }
        (qs, ds)
    }

    /// The largest probability of a row of logits by `argmax_p_rows_fault`'s
    /// order: `fv` the row's maximum; thread `t` of 1024 sums `exp(x_i - fv)`
    /// over `i = t, t + 1024, ..` from 0; each warp's 32 lanes by the xor
    /// butterfly; the 32 warps' sums added from 0 in order; `1 / total`.
    fn host_p(row: &[f32]) -> f32 {
        const THREADS: usize = 1024;
        let fv = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let lane = |t: usize| {
            let mut z = 0.0f32;
            let mut i = t;
            while i < row.len() {
                z += exp_ours(row[i] - fv);
                i += THREADS;
            }
            z
        };
        let mut total = 0.0f32;
        for w in 0..THREADS / 32 {
            total += butterfly(std::array::from_fn(|l| lane(32 * w + l)));
        }
        1.0 / total
    }

    /// The card's projection logits (`[v * m + c]`) against the host's f64
    /// norm `x · gain / sqrt(mean(x²) + eps)` and dot with the dequantized
    /// codes, as the worst ratio of the difference to `γ(4 · HEAD_HIDDEN)`
    /// times the sum of the absolute terms: the norm's and the dot's f32
    /// roundings are each about `HEAD_HIDDEN` operations deep. At most 1 holds.
    #[allow(
        clippy::too_many_arguments,
        reason = "the head's seeded inputs and the card's output, each its own role"
    )]
    fn projection_ratio(
        x: &[f32],
        gain: &[f32],
        eps: f32,
        qs: &[u32],
        d: &[u16],
        card: &[f32],
        m: usize,
    ) -> f64 {
        let k = HEAD_HIDDEN;
        let normed: Vec<Vec<f64>> = (0..m)
            .map(|c| {
                let row = &x[c * k..(c + 1) * k];
                let ms = row
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
                    / k as f64;
                let r = 1.0 / (ms + f64::from(eps)).sqrt();
                row.iter()
                    .zip(gain)
                    .map(|(&v, &g)| f64::from(v) * r * f64::from(g))
                    .collect()
            })
            .collect();
        let bound = gamma(4 * k);
        let mut worst = 0.0f64;
        for v in 0..RED_HEAD {
            let w: Vec<f64> = (0..k)
                .map(|i| {
                    let scale = f64::from(half_to_f32(d[v * (k / 32) + i / 32]));
                    let code = (qs[v * (k / 4) + i / 4] >> (8 * (i % 4))) as u8 as i8;
                    scale * f64::from(code)
                })
                .collect();
            for (c, n) in normed.iter().enumerate() {
                let terms = w.iter().zip(n).map(|(a, b)| a * b);
                let (sum, abs) = terms.fold((0.0f64, 0.0f64), |(s, a), t| (s + t, a + t.abs()));
                let ratio = (f64::from(card[v * m + c]) - sum).abs() / (bound * abs);
                worst = if ratio.is_nan() {
                    f64::INFINITY
                } else {
                    worst.max(ratio)
                };
            }
        }
        worst
    }

    /// An own head over a reduced vocabulary on seeded weights: the card's
    /// projection against the host's f64, then the Markov loop, the argmax and
    /// the probabilities against the host rules, taking the projection's
    /// logits as the card made them. The head returns head ids.
    fn check_head_reduced(cx: &mut Ctx<'_>) -> Result<(), GateError> {
        let (rd, s, m) = (reduced_tables(cx.gpu)?, cx.gpu.stream(), RED_M);
        let (qs, d) = q8_0_rows(&streams(RED_HEAD * HEAD_HIDDEN, 51));
        let gain: Vec<f32> = streams(HEAD_HIDDEN, 52)
            .iter()
            .map(|g| 1.0 + g / 4.0)
            .collect();
        let w = HeadWeights {
            norm: DeviceTensor::upload(s, &gain, 1, HEAD_HIDDEN)?,
            eps: EPS,
            source: HeadSource::Own(Q8Rows {
                qs: DeviceTensor::upload(s, &qs, RED_HEAD, HEAD_HIDDEN / 4)?,
                d: DeviceTensor::upload(s, &d, RED_HEAD, HEAD_HIDDEN / 32)?,
            }),
            markov_w1: DeviceTensor::upload(s, &rd.w1, RED_PREV, MARKOV_ROW_WORDS)?,
            markov_w2: DeviceTensor::upload(s, &rd.w2, RED_HEAD, MARKOV_ROW_WORDS)?,
            to_target: Some(DeviceBuffer::from_host(s, &rd.to_target)?),
            conf: None,
        };
        let mut head = Head::new(cx.gpu, w, HEAD_HIDDEN, m)?;
        // One count, the same one `DsparkBody::own_held` adds: each weight
        // buffer, then the scratch. The shapes below are that count with no
        // q8_1 term. An own head's row kernel reads the f32 rows, so `new`
        // allocates no `Q8Act` and the scratch stays this. A Q6_K target
        // head allocates one `Q8Act` per row count, and `scratch_bytes`
        // counts those planes.
        let words = |n: usize| 4 * n;
        let want_weights = words(HEAD_HIDDEN)
            + words(RED_HEAD * HEAD_HIDDEN / 4)
            + 2 * RED_HEAD * HEAD_HIDDEN / 32
            + words(RED_PREV * MARKOV_ROW_WORDS)
            + words(RED_HEAD * MARKOV_ROW_WORDS)
            + words(RED_HEAD);
        let want_scratch = words(m * HEAD_HIDDEN)
            + words(m * RED_HEAD)
            + words(m + 2)
            + words(2 * m + 2)
            + words(1)
            + words(1);
        // 587008 + 6264 = 593272. Acts are 0, so the old total stays.
        let want_bytes = want_weights + want_scratch;
        let weights: u64 = head.weight_buffers().iter().sum();
        let scratch = head.scratch_bytes();
        let bytes = weights + scratch;
        let host_x = streams(m * HEAD_HIDDEN, 53);
        let x = DeviceBuffer::from_host(s, &host_x)?;
        let fd = DeviceBuffer::from_host(s, &[HEAD_FIRST])?;

        head.enqueue_logits(cx.gpu, &x, m)?;
        let base = head.logits(s, m)?;
        head.enqueue_markov(cx.gpu, m, &fd)?;
        let toks = head.tokens(s, m)?;
        let corrected = head.logits(s, m)?;
        head.enqueue_p(cx.gpu, m)?;
        let (ids_p, p) = head.tokens_p(s, m)?;

        let (heads, hl) = host_reduced_chain(&rd, &base, m, HEAD_FIRST);
        let want = heads;
        let want_p: Vec<f32> = (0..m)
            .map(|c| host_p(&(0..RED_HEAD).map(|v| hl[v * m + c]).collect::<Vec<_>>()))
            .collect();
        let ratio = projection_ratio(&host_x, &gain, EPS, &qs, &d, &base, m);
        let finite = base.len() == RED_HEAD * m && base.iter().all(|x| x.is_finite());
        let rows = bits_equal(&corrected, &hl);
        let p_bits = bits_equal(&p, &want_p);
        let pass = finite
            && ratio <= 1.0
            && head.n_vocab() == RED_HEAD
            && weights == want_weights as u64
            && scratch == want_scratch as u64
            && bytes == want_bytes as u64
            && toks == want
            && rows
            && ids_p == want
            && p_bits;
        println!(
            "head own reduced hidden={HEAD_HIDDEN} n_head={RED_HEAD} n_prev={RED_PREV} m={m} first={HEAD_FIRST}: \
             projection worst ratio to bound {ratio:.3e}; head ids {toks:?} want {want:?} tokens_equal={} rows_bit_identical={rows} tokens_p ids_equal={} \
             p_bit_identical={p_bits} (p {p:?}) weights {weights} want {want_weights} scratch {scratch} want {want_scratch} bytes {bytes} want {want_bytes} {}",
            toks == want,
            ids_p == want,
            verdict(pass)
        );
        cx.ok &= pass;
        Ok(())
    }

    // ------------------------------------------------- oracle diagnostics

    /// The draft set's block 0 (`refset::dsref`): its block-pass rows, the
    /// block's proposals and the token it starts from.
    struct DSet<'a> {
        rows: Vec<&'a DsRow>,
        set: &'a Dsref,
        drafts: Vec<u32>,
        id_last: u32,
    }

    impl<'a> DSet<'a> {
        fn block0(set: &'a Dsref) -> Result<DSet<'a>, GateError> {
            Ok(DSet {
                rows: set.in_block(0, Graph::Block),
                set,
                drafts: set.drafts_of(0),
                id_last: set.verify_of(0)?.id_last,
            })
        }

        fn find(&self, pred: impl Fn(&DsRow) -> bool) -> Vec<&'a DsRow> {
            self.rows.iter().copied().filter(|r| pred(r)).collect()
        }

        fn load(&self, r: &DsRow) -> Result<Vec<f32>, GateError> {
            Ok(self.set.logical_f32s(r)?)
        }
    }

    /// ik's HC_PRE node (`[pre S T][post S T][comb S S T]`) in our layout.
    fn node_to_hc(node: &[f32], t: usize) -> Vec<f32> {
        let mut hc = vec![0.0f32; HC_MIX * t];
        for (tt, h) in hc.as_chunks_mut::<HC_MIX>().0.iter_mut().enumerate() {
            h[..4].copy_from_slice(&node[4 * tt..4 * tt + 4]);
            h[4..8].copy_from_slice(&node[4 * t + 4 * tt..4 * t + 4 * tt + 4]);
            h[8..].copy_from_slice(&node[8 * t + 16 * tt..8 * t + 16 * tt + 16]);
        }
        hc
    }

    fn oracle(cx: &Ctx<'_>, attn: &Hc, mk: &Markov) -> Result<(), GateError> {
        let dir = DSREF.path(DSREF_SET);
        let full = Dsref::open(&dir, &DSREF)?;
        let set = DSet::block0(&full)?;
        println!(
            "oracle set {} block 0: {} block-graph rows, drafts {:?}, id_last {}",
            dir.display(),
            set.rows.len(),
            set.drafts,
            set.id_last
        );

        // HC_PRE: layer 0 attention.
        let input = set.find(|r| r.name == "dsv4_dflash_hc_init (reshaped)" && r.occurrence == 0);
        let mixes = set.find(|r| r.name == "hc_pre_mixes-0" && r.occurrence == 0);
        let node = set.find(|r| r.op == "HC_PRE" && r.src0.as_deref() == Some("hc_pre_mixes-0"));
        match (input.first(), mixes.first(), node.first()) {
            (Some(i), Some(mx), Some(nd)) => {
                let x = set.load(i)?;
                let t = usize::try_from(i.ne[1])?;
                let (om, oh) = run_hc_kernel(cx, attn, &x, t)?;
                let im = set.load(mx)?;
                let ih = node_to_hc(&set.load(nd)?, t);
                println!(
                    "oracle hc_pre_f32 blk0.attn T={t}: mixes max|diff| {:e} (ik max|mix| {:e}); \
                     hc max|diff| {:e} (pre/post {:e}, comb {:e}) — diagnostic, not pinned",
                    max_abs_diff(&om, &im),
                    max_abs(&im),
                    max_abs_diff(&oh, &ih),
                    (0..t)
                        .map(|tt| max_abs_diff(
                            &oh[tt * 24..tt * 24 + 8],
                            &ih[tt * 24..tt * 24 + 8]
                        ))
                        .fold(0.0, f64::max),
                    (0..t)
                        .map(|tt| max_abs_diff(
                            &oh[tt * 24 + 8..tt * 24 + 24],
                            &ih[tt * 24 + 8..tt * 24 + 24]
                        ))
                        .fold(0.0, f64::max),
                );
                // Our HC_PRE rule at ik's own mixes: the gemv's share of the gap removed.
                let iters = u32::try_from(cx.hp.hc.sinkhorn_iters)?;
                let own: Vec<f32> = (0..t)
                    .flat_map(|tt| {
                        hc_pre_f32(
                            &im[tt * 24..tt * 24 + 24],
                            [attn.scale[0], attn.scale[1], attn.scale[2]],
                            &attn.base,
                            cx.hp.hc.eps,
                            iters,
                            exp_ours,
                        )
                    })
                    .collect();
                println!(
                    "oracle hc_pre rule at ik's mixes: hc max|diff| {:e} — diagnostic, not pinned",
                    max_abs_diff(&own, &ih)
                );
            }
            _ => println!(
                "oracle hc_pre_f32: the set lacks a row it needs (input {}, mixes {}, HC_PRE node {})",
                input.len(),
                mixes.len(),
                node.len()
            ),
        }

        // Markov: from ik's pre-Markov logits.
        let base = set.find(|r| r.name == "dflash_base_result_output" && r.occurrence == 0);
        let deltas =
            set.find(|r| r.op == "MUL_MAT" && r.src0.as_deref() == Some("markov_w2.weight"));
        let adds = set.find(|r| {
            r.op == "ADD" && r.src0.as_deref() == Some("dflash_base_result_output (view)")
        });
        let result = set.find(|r| r.name == "result_output" && r.occurrence == 0);
        let (Some(b), Some(res)) = (base.first(), result.first()) else {
            println!(
                "oracle markov: the set lacks dflash_base_result_output ({}) or result_output ({})",
                base.len(),
                result.len()
            );
            return Ok(());
        };
        let n = mk.n_vocab;
        let tm = set.load(b)?; // token-major [t][v]
        let t = usize::try_from(b.ne[1])?;
        let mut vm = vec![0.0f32; n * t];
        for c in 0..t {
            for v in 0..n {
                vm[v * t + c] = tm[c * n + v];
            }
        }
        // ik's previous tokens: id_last, then its own draft tokens.
        let prevs: Vec<u32> = std::iter::once(set.id_last)
            .chain(set.drafts.iter().copied())
            .take(t)
            .collect();
        for (c, (&p, d)) in prevs.iter().zip(&deltas).enumerate() {
            let ik = set.load(d)?;
            let ours = delta_host(&mk.w1, &mk.w2f, p as usize);
            let not_bf16 = ik.iter().filter(|x| x.to_bits() & 0xffff != 0).count();
            let ours_bf16: Vec<f32> = ours.iter().map(|&x| bf16_round(x)).collect();
            println!(
                "oracle markov delta row {c}: ik values not bf16-representable {not_bf16} of {}; \
                 ours rounded to bf16 (nearest even) vs ik: max|diff| {:e}, {} values differ — diagnostic",
                ik.len(),
                max_abs_diff(&ours_bf16, &ik),
                ours_bf16
                    .iter()
                    .zip(&ik)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
            );
            println!(
                "oracle markov delta row {c} prev {p}: max|diff| {:e} (ik max|delta| {:e}), argmax ours {} ik {} — diagnostic",
                max_abs_diff(&ours, &ik),
                max_abs(&ik),
                argmax(&ours),
                argmax(&ik)
            );
        }
        for (c, a) in adds.iter().enumerate().take(t) {
            let ik = set.load(a)?;
            let ours: Vec<f32> = {
                let d = delta_host(&mk.w1, &mk.w2f, prevs[c] as usize);
                (0..n).map(|v| tm[c * n + v] + d[v]).collect()
            };
            println!(
                "oracle markov corrected row {c} (ik base + our delta at ik's prev) vs ik ADD: max|diff| {:e} — diagnostic",
                max_abs_diff(&ours, &ik)
            );
        }
        let (toks, rows) = run_chain(cx, mk, &vm, t, set.id_last)?;
        let ik_res = set.load(res)?;
        let mut ours_tm = vec![0.0f32; n * t];
        for c in 0..t {
            for v in 0..n {
                ours_tm[c * n + v] = rows[v * t + c];
            }
        }
        println!(
            "oracle markov chain on ik's base logits: tokens ours {toks:?} ik {:?}; result_output max|diff| {:e} \
             (ik max|logit| {:e}) — diagnostic, not pinned",
            set.drafts,
            max_abs_diff(&ours_tm, &ik_res),
            max_abs(&ik_res)
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let path = std::env::var_os("BLOOMERY_DSPARK_MODEL")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .ok_or(
                "BLOOMERY_DSPARK_MODEL unset — run through `just gate-gpu-dspark-hc`, which exports \
                 it from the V4.1 profile's DSPARK_MODEL",
            )?;
        let split = Split::open(&path)?;
        let hp = DraftHparams::read(&split)?;
        if hp.markov_rank != MARKOV_RANK || hp.hc.streams != 4 {
            return Err(format!(
                "markov rank {} / hc streams {}: the kernels take {MARKOV_RANK} / 4",
                hp.markov_rank, hp.hc.streams
            )
            .into());
        }
        println!(
            "draft {}: n_embd {} K {} n_vocab {} rank {} rms_eps {:e} hc eps {:e} sinkhorn {}",
            path.display(),
            hp.n_embd,
            hp.hc.streams * hp.n_embd,
            hp.n_vocab,
            hp.markov_rank,
            hp.rms_eps,
            hp.hc.eps,
            hp.hc.sinkhorn_iters
        );
        let attn = read_hc(&split, &hp, "attn")?;
        let ffn = read_hc(&split, &hp, "ffn")?;
        let n = hp.n_vocab;
        let dims = [MARKOV_RANK as u64, n as u64];
        let w1 = bytes_to_words(tensor_bytes(
            &split,
            &names::markov_w1(),
            GgmlType::BF16,
            &dims,
        )?);
        let w2 = bytes_to_words(tensor_bytes(
            &split,
            &names::markov_w2(),
            GgmlType::BF16,
            &dims,
        )?);
        let w2f: Vec<f32> = w2
            .iter()
            .flat_map(|&x| [f32::from_bits(x << 16), f32::from_bits(x & 0xffff_0000)])
            .collect();

        let gpu = Gpu::new()?;
        let hck = HcF32Kernels::load(gpu.context())?;
        let mkk = MarkovKernels::load(gpu.context())?;
        let s = gpu.stream();
        let mk = Markov {
            w1d: DeviceTensor::upload(s, &w1, n, MARKOV_ROW_WORDS)?,
            w2d: DeviceTensor::upload(s, &w2, n, MARKOV_ROW_WORDS)?,
            w1,
            w2f,
            n_vocab: n,
        };
        let mut cx = Ctx {
            gpu: &gpu,
            hck: &hck,
            mk: &mkk,
            hp: &hp,
            ok: true,
        };
        check_hc(&mut cx, &attn, "attn")?;
        check_hc(&mut cx, &ffn, "ffn")?;

        let mut ids = read_ids(&data_dir().join("engram/corpus-code.ids"), CORPUS_IDS)?;
        ids.extend([0, 1, (n - 1) as u32]);
        check_markov(&mut cx, &mk, &ids)?;
        check_reduced(&mut cx)?;
        check_head_reduced(&mut cx)?;

        oracle(&cx, &attn, &mk)?;

        if cx.ok {
            println!(
                "PASSED: {NAME} — ds41_hc_pre_f32 (m 1/3/5, attn and ffn) and ds41_markov (67 ids, \
                 3-row chain; reduced head, 8-row chain and its fault; own reduced Head chain) bit-identical \
                 to the host rules; oracle rows printed, not pinned"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
