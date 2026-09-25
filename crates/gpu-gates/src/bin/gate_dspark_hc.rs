//! GPU gate for the DSpark draft's two kernels the target lacks
//! (`bloomery_gpu_deepseek41::hc_f32` and `::markov`), on the 3090.
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

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_dspark_hc: built without the `deepseek41` feature; see `just gate-gpu-dspark-hc`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_hc", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};

    use bloomery_gpu::{DeviceTensor, FAULT_NONE, Gpu};
    use bloomery_gpu_deepseek41::hc::HC_MIX;
    use bloomery_gpu_deepseek41::hc_f32::{HcF32Args, HcF32Kernels, HcF32Params};
    use bloomery_gpu_deepseek41::markov::{
        MARKOV_RANK, MARKOV_ROW_WORDS, MarkovKernels, MarkovWeights,
    };
    use bloomery_gpu_gates::hc_host::{self, exp_ours, hc_pre_f32};
    use bloomery_gpu_gates::rounding::butterfly;
    use bloomery_gpu_gates::{
        GateError, bits_equal, bytes_to_words, checks_failed, data_dir, dump_stem, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::dspark::{DraftHparams, names};

    const NAME: &str = "gate_dspark_hc";
    /// The dsref set this gate reads its diagnostics from.
    const DSREF_SET: &str = "code64_n32_w3";
    /// The ik tree every V4.1 oracle set must name in its `# build` line.
    const IK_BUILD: &str = "db517b69";
    /// Corpus ids the Markov checks take, then the three edge ids.
    const CORPUS_IDS: usize = 64;
    /// Rows of the chained Markov check (the dsref set's block width).
    const CHAIN: usize = 3;
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

    // ------------------------------------------------- oracle diagnostics

    /// A block-0 row of the draft set's manifest (`tools/ref/dump_draft.cpp`).
    struct DRow {
        kind: String,
        name: String,
        occ: u32,
        ne: [usize; 4],
        op: String,
        logical: bool,
        src0: String,
    }

    struct DSet {
        dir: PathBuf,
        rows: Vec<DRow>,
        drafts: Vec<u32>,
        id_last: u32,
    }

    fn read_set(dir: &Path) -> Result<DSet, GateError> {
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv"))
            .map_err(|e| format!("{}/MANIFEST.tsv: {e}", dir.display()))?;
        let build = text
            .lines()
            .find_map(|l| l.strip_prefix("# build\t"))
            .unwrap_or("");
        if !build.contains(IK_BUILD) {
            return Err(format!(
                "{}: # build {build:?} does not name {IK_BUILD} — stale set",
                dir.display()
            )
            .into());
        }
        let (mut rows, mut drafts, mut id_last) = (Vec::new(), Vec::new(), None);
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = l.split('\t').collect();
            match f[0] {
                "tensor" | "input" if f.len() == 19 && f[15] == "0" && f[18] == "block" => {
                    let u = |i: usize| f[i].parse::<usize>().unwrap_or(0);
                    rows.push(DRow {
                        kind: f[0].to_string(),
                        name: f[1].to_string(),
                        occ: f[2].parse()?,
                        ne: [u(4), u(5), u(6), u(7)],
                        op: f[10].to_string(),
                        logical: f[12] == "1",
                        src0: f[13].to_string(),
                    });
                }
                "draft" if f.len() == 4 && f[1] == "0" => drafts.push(f[3].parse()?),
                "verify" if f.len() >= 4 && f[1] == "0" => id_last = Some(f[3].parse()?),
                _ => {}
            }
        }
        Ok(DSet {
            dir: dir.to_path_buf(),
            rows,
            drafts,
            id_last: id_last.ok_or("the set has no verify row for block 0")?,
        })
    }

    impl DSet {
        fn find(&self, pred: impl Fn(&DRow) -> bool) -> Vec<&DRow> {
            self.rows.iter().filter(|r| pred(r)).collect()
        }

        fn load(&self, r: &DRow) -> Result<Vec<f32>, GateError> {
            let input = if r.kind == "input" { ".input" } else { "" };
            let logical = if r.logical { ".logical" } else { "" };
            let path = self.dir.join(format!(
                "b0.{}.{}{input}{logical}.f32",
                dump_stem(&r.name),
                r.occ
            ));
            let v = f32s(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?);
            let want: usize = r.ne.iter().product();
            if v.len() != want {
                return Err(format!(
                    "{}: {} values, manifest ne {:?}",
                    path.display(),
                    v.len(),
                    r.ne
                )
                .into());
            }
            Ok(v)
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
        let dir = data_dir().join("ref-draft").join(DSREF_SET);
        let set = read_set(&dir)?;
        println!(
            "oracle set {} block 0: {} block-graph rows, drafts {:?}, id_last {}",
            dir.display(),
            set.rows.len(),
            set.drafts,
            set.id_last
        );

        // HC_PRE: layer 0 attention.
        let input = set.find(|r| r.name == "dsv4_dflash_hc_init (reshaped)" && r.occ == 0);
        let mixes = set.find(|r| r.name == "hc_pre_mixes-0" && r.occ == 0);
        let node = set.find(|r| r.op == "HC_PRE" && r.src0 == "hc_pre_mixes-0");
        match (input.first(), mixes.first(), node.first()) {
            (Some(i), Some(mx), Some(nd)) => {
                let x = set.load(i)?;
                let t = i.ne[1];
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
        let base = set.find(|r| r.name == "dflash_base_result_output" && r.occ == 0);
        let deltas = set.find(|r| r.op == "MUL_MAT" && r.src0 == "markov_w2.weight");
        let adds = set.find(|r| r.op == "ADD" && r.src0 == "dflash_base_result_output (view)");
        let result = set.find(|r| r.name == "result_output" && r.occ == 0);
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
        let t = b.ne[1];
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

        oracle(&cx, &attn, &mk)?;

        if cx.ok {
            println!(
                "PASSED: {NAME} — ds41_hc_pre_f32 (m 1/3/5, attn and ffn) and ds41_markov (67 ids, \
                 3-row chain) bit-identical to the host rules; oracle rows printed, not pinned"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
