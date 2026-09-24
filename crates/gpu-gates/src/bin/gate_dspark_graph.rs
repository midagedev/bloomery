//! GPU gate for the DSpark draft's block pass and head
//! (`bloomery_gpu_deepseek41::draft::{block, head}` and `DraftBody`), on the
//! 3090, against the dsref set `code64_n32_w3` (ik's draft, dumped at block
//! width 3).
//!
//! Inputs. For each block the set's own: each layer's ring as ik's block
//! graph read it (`dflash_k_ctx_cache_L`, the committed rows `0..n`, in
//! slot `row % window`), the block's first id and rope positions
//! (`inp_tokens`, `inp_pos`) and `n` visible ring rows. The set's mask is
//! decoded first and must name exactly the keys the arm below feeds — the
//! `n` ring rows and, for block row `t`, ik's block rows `0..=t`.
//!
//! Two arms of the same pass, differing only in data (`Rule`): the block
//! rows a row sees (the attention's `vis[2t + 1]`) and the SwiGLU limits.
//!
//! - **ik's rule** (`Rule::Ik`): ik's block-causal mask and its unclamped
//!   SwiGLUs, routed and shared — the two places ik's draft leaves the
//!   reference, both in the set. Like for like with the oracle, so every
//!   tap of every row is judged, blocks 0–2, and `draft_argmax` on every
//!   block. Which SwiGLU ik's shared expert computes is read off the set
//!   itself: each layer's `diag` line puts every candidate rule, from ik's
//!   own input through the file's weights in f64, against ik's
//!   `ffn_up_gate`.
//! - **the reference rule** (`Rule::Reference`, `model.py`): the engine's.
//!   Only the rows the oracle's mask leaves on ik's path are judged — every
//!   row before layer 0's attention, and row `w − 1` until layer 1's
//!   attention (the last row sees all of the block under both masks; from
//!   layer 1 on, the keys of every row carry the earlier rows' defect). The
//!   clamped SwiGLUs are in those rows too; where the shared expert's clamp
//!   fires, its share of the distance is the `diag` line's. Every other row
//!   is printed and named a defect row of the oracle.
//!
//! Taps (ours against the set's node): each layer's two HC_PRE (mixes and
//! result), the attention norm, `attn_kv` and the turned block row
//! (`dsv4_dflash_kv-L`), `q_a` and its norm, the turned query
//! (`dsv4_dflash_q-L`), the attention after its inverse rope, `wo_a`,
//! `wo_b`, both HC_POST and their folds, the ffn norm, the router's logits,
//! scores, ids (exact) and scaled weights, `ffn_moe_gate_par` (h, ours over
//! the concat plan), `ffn_moe_weighted`, the shared expert's h and output,
//! their sum; the head's norm, `dflash_base_result_output`,
//! `result_output`, `draft_argmax`. ik's per-slot `ffn_moe_down` has no
//! counterpart — our down pass combines as it goes.
//!
//! Band, per tap, per row: the relative RMS distance
//! `ρ = rms(ours − ik) / rms(ik)`. The prediction `ρ̂` is written before our
//! pass runs, from ik's own tensors: every rounding one side makes and the
//! other does not, taken as noise that enters in quadrature at unit gain
//! wherever it sits before the tap in graph order — ik's q8_1 activation
//! (32-value blocks, `d = amax/127`, variance `d²/12`) at every Q8_0, MXFP4
//! and Q6_K matmul, ours (128-value blocks) at the MXFP4 experts and the Q6_K
//! head, ik's bf16 activation at the router (`2^-9/√3`), ik's f16
//! attention accumulation ([`ATTN_SITE`], assumed), and ik's bf16 Markov
//! delta (from ik's own delta). The band is `BAND_GAIN · ρ̂`, never below
//! [`FLOOR`]; `BAND_GAIN` is the model's allowance for a gain above one
//! through a softmax, a Sinkhorn or a SwiGLU. A composition fault — a wrong
//! buffer, row, table, count or source — moves a tap by its own size.
//! The router's ids must equal ik's; where a row's differ, it is a tie when
//! that row's scores are within their band and ik's own gap between its third
//! and fourth biased scores is inside the band's share of the top score. A
//! tie takes rows off ik's path, each tracked on its own: the tied row from
//! its routed experts on, and every row whose attention reads it (the rows
//! after it under ik's rule, every row under the reference) from the next
//! layer's attention on. Their taps from there are printed, not judged; the
//! other rows stay judged. An argmax that differs from ik's is a tie when
//! ik's own margin between the two ids is inside the band carried to the
//! logits; the next row's Markov step then reads another id, so that row's
//! result and argmax are printed.
//!
//! Also: `w = 5` over block 0's inputs — five ids, a rerun bit-identical,
//! and under ik's mask rows 0–2 bit-identical to the `w = 3` pass (no kernel
//! reads a row it is not given; the first rows see the same keys); under the
//! reference rule row 0 sees every block row, so it changes with `w`
//! (printed). The accept count each arm's ids would give against the set's
//! target tokens, all twelve blocks, next to ik's — and `DraftBody`'s own,
//! which appends the set's features at the reference positions: printed,
//! not pinned.
//!
//! Graphs. The launches a pass of each width `1..=5` makes, counted in the
//! gate's own capture and in the graph `DraftBody` captured at load, equal
//! `BlockPass::launches` and the pinned `73 + 8w`; each append graph
//! (`1..=GROUP` rows) holds its eight launches; every node is a kernel. Then,
//! on `DraftBody`'s own pass and rings under the reference rule: for blocks
//! 0–2 and every width, the replay and then the eager twin leave every tap
//! bit-identical (the replay first, so one that did not re-read its inputs
//! would read the previous call's); blocks 0–2 in a row through the `w = 3`
//! graph equal the judged eager passes of the gate's own pass above; and
//! `DraftBody` over the whole set through its graphs gives the ids and the
//! rings of its eager twin after every block.
//!
//! FAIL-first: swapping the attention's two sources in `draft/block.rs` (the
//! ring as the second source, the block rows as the first) turns this gate
//! red from layer 0's attention on; a graph replay that does not restage its
//! inputs (`draft/mod.rs`'s `run_block` staging for the eager arm only)
//! turns the graph rows red.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_dspark_graph: built without the `deepseek41` feature; see `just gate-gpu-dspark-graph`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_graph", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bloomery_gpu::{Gpu, Graph};
    use bloomery_gpu_deepseek41::draft::block::{
        BlockInput, BlockPass, LayerBufs, MAX_WIDTH, Rule, embedding_row,
    };
    use bloomery_gpu_deepseek41::draft::kv::{DraftRings, GROUP};
    use bloomery_gpu_deepseek41::draft::load::DraftWeights;
    use bloomery_gpu_deepseek41::draft::{DraftBody, Submit};
    use bloomery_gpu_gates::{
        GateError, checks_failed, data_dir, dump_stem, ref_model_path, verdict,
    };
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits};
    use model::arch::dspark::{DraftHparams, names};

    const NAME: &str = "gate_dspark_graph";
    const DSREF_SET: &str = "code64_n32_w3";
    /// The ik tree every V4.1 oracle set must name in its `# build` line.
    const IK_BUILD: &str = "db517b69";
    /// Blocks whose every tap is judged.
    const TAP_BLOCKS: [i32; 3] = [0, 1, 2];
    /// The set's block width (its manifest's `n_max`).
    const SET_WIDTH: usize = 3;
    /// The synthetic block's width.
    const WIDE: usize = 5;
    /// Kernel nodes of the captured block pass of `w` rows, `w = 1..=5`:
    /// the widening, three layers of `23 + 2w`, the head's three and the
    /// Markov loop's `2w` — `73 + 8w`.
    const PASS_NODES: [usize; MAX_WIDTH] = [81, 89, 97, 105, 113];
    /// Kernel nodes of the captured append of `n <= GROUP` rows: `fc`, its
    /// norm, and per layer `attn_kv` and the append.
    const APPEND_NODES: usize = 8;
    /// The band's allowance over the unit-gain prediction.
    const BAND_GAIN: f64 = 8.0;
    /// The least band: f32 sums in another order, `expf` against `expf`.
    const FLOOR: f64 = 1e-4;
    /// ik's f16 flash attention against ours, relative, per attention —
    /// assumed, not derived: f16 accumulation over at most a window and a
    /// block of keys.
    const ATTN_SITE: f64 = 3e-3;
    /// A bf16 rounding's relative RMS: `2^-9 / √3`.
    const BF16_SITE: f64 = 1.0 / 512.0 / 1.732_050_807_568_877_2;

    // ------------------------------------------------------------- the set

    struct Row {
        kind: String,
        name: String,
        occ: u32,
        ne: [usize; 4],
        op: String,
        src0: String,
        src1: String,
        block: i32,
        graph: String,
    }

    /// One `verify` line: what the target made of a block.
    struct Verify {
        block: i32,
        id_last: u32,
        carry: u32,
        accepted: usize,
        target: Vec<u32>,
    }

    impl Verify {
        /// The target's token for each draft row, in order.
        fn targets(&self) -> &[u32] {
            let skip = usize::from(self.carry == 0).min(self.target.len());
            &self.target[skip..]
        }
    }

    struct Set {
        dir: PathBuf,
        rows: Vec<Row>,
        verify: Vec<Verify>,
        drafts: Vec<(i32, usize, u32)>,
    }

    fn read_set(dir: &Path) -> Result<Set, GateError> {
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
        let (mut rows, mut verify, mut drafts) = (Vec::new(), Vec::new(), Vec::new());
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = l.split('\t').collect();
            match f[0] {
                "tensor" | "input" if f.len() == 19 => {
                    let u = |i: usize| f[i].parse::<usize>().unwrap_or(0);
                    rows.push(Row {
                        kind: f[0].to_string(),
                        name: f[1].to_string(),
                        occ: f[2].parse()?,
                        ne: [u(4), u(5), u(6), u(7)],
                        op: f[10].to_string(),
                        src0: f[13].to_string(),
                        src1: f[14].to_string(),
                        block: f[15].parse()?,
                        graph: f[18].to_string(),
                    });
                }
                "verify" if f.len() == 8 => verify.push(Verify {
                    block: f[1].parse()?,
                    id_last: f[3].parse()?,
                    carry: f[4].parse()?,
                    accepted: f[6].parse()?,
                    target: f[7].split(',').map(str::parse).collect::<Result<_, _>>()?,
                }),
                "draft" if f.len() == 4 => {
                    drafts.push((f[1].parse()?, f[2].parse()?, f[3].parse()?))
                }
                _ => {}
            }
        }
        Ok(Set {
            dir: dir.to_path_buf(),
            rows,
            verify,
            drafts,
        })
    }

    impl Set {
        fn in_block(&self, b: i32, graph: &str) -> Vec<&Row> {
            self.rows
                .iter()
                .filter(|r| r.block == b && r.graph == graph)
                .collect()
        }

        fn find(&self, b: i32, graph: &str, name: &str, occ: u32) -> Result<&Row, GateError> {
            self.in_block(b, graph)
                .into_iter()
                .find(|r| r.name == name && r.occ == occ)
                .ok_or_else(|| {
                    format!("the set has no {graph} row {name:?}#{occ} in block {b}").into()
                })
        }

        /// The block graph's node of `op` whose sources match (`""` matches any).
        fn find_op(&self, b: i32, op: &str, src0: &str, src1: &str) -> Result<&Row, GateError> {
            self.in_block(b, "block")
                .into_iter()
                .find(|r| {
                    r.kind == "tensor"
                        && r.op == op
                        && (src0.is_empty() || r.src0 == src0)
                        && (src1.is_empty() || r.src1 == src1)
                })
                .ok_or_else(|| format!("block {b} has no {op} node over {src0:?}, {src1:?}").into())
        }

        /// The `i`-th block-graph node of `op`, in graph order.
        fn nth_op(&self, b: i32, op: &str, i: usize) -> Result<&Row, GateError> {
            self.in_block(b, "block")
                .into_iter()
                .filter(|r| r.kind == "tensor" && r.op == op)
                .nth(i)
                .ok_or_else(|| format!("block {b} has fewer than {} {op} nodes", i + 1).into())
        }

        fn path(&self, r: &Row, twin: &str, ext: &str) -> PathBuf {
            let kv = if r.graph == "kv" { "kv." } else { "" };
            let input = if r.kind == "input" { ".input" } else { "" };
            self.dir.join(format!(
                "b{}.{kv}{}.{}{input}{twin}.{ext}",
                r.block,
                dump_stem(&r.name),
                r.occ
            ))
        }

        fn f32s(&self, r: &Row) -> Result<Vec<f32>, GateError> {
            let p = self.path(r, "", "f32");
            let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let v: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let want: usize = r.ne.iter().product();
            if v.len() != want {
                return Err(format!(
                    "{}: {} values, manifest ne {:?}",
                    p.display(),
                    v.len(),
                    r.ne
                )
                .into());
            }
            Ok(v)
        }

        fn i32s(&self, r: &Row, twin: &str) -> Result<Vec<i32>, GateError> {
            let p = self.path(r, twin, "i32");
            let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            Ok(b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect())
        }

        fn verify_of(&self, b: i32) -> Result<&Verify, GateError> {
            self.verify
                .iter()
                .find(|v| v.block == b)
                .ok_or_else(|| format!("the set has no verify line for block {b}").into())
        }

        fn blocks(&self) -> Vec<i32> {
            self.verify.iter().map(|v| v.block).collect()
        }
    }

    // ------------------------------------------------------ one block's inputs

    struct BlockSet {
        b: i32,
        /// The block's width: the set's `n_max`, fewer where generation ended.
        width: usize,
        id_last: u32,
        pos: Vec<i32>,
        ring_rows: usize,
        /// Per layer, ik's committed ring rows `0..ring_rows`, f16 bits.
        rings: Vec<Vec<u16>>,
        /// The rows of the kv graph that ran before this block, and their
        /// feature rows (the `DraftBody` replay appends them).
        feats: Vec<f32>,
    }

    fn block_set(set: &Set, hp: &DraftHparams, b: i32) -> Result<BlockSet, GateError> {
        let tokens = set.i32s(set.find(b, "block", "inp_tokens", 0)?, "")?;
        let pos = set.i32s(set.find(b, "block", "CUDA0#inp_pos#0", 0)?, "")?;
        let tail = set.i32s(
            set.find(b, "block", "CUDA0#dflash_draft_tail_rows#0", 0)?,
            "",
        )?;
        let kv_rows = set.i32s(set.find(b, "kv", "CUDA0#dflash_kv_input_rows#0", 0)?, "")?;
        let feats = set.f32s(set.find(b, "kv", "CUDA0#dflash_kv_input_target_features#0", 0)?)?;
        let w = tokens.len();
        if w == 0 || w > SET_WIDTH || pos.len() != w || tail.len() != w {
            return Err(format!(
                "block {b}: {w} tokens, {} positions, {} tail rows",
                pos.len(),
                tail.len()
            )
            .into());
        }
        let ring_rows = kv_rows.iter().copied().max().ok_or("an empty kv graph")? as usize + 1;
        let v = set.verify_of(b)?;
        if tokens[0] as u32 != v.id_last || tokens[1..].iter().any(|&t| t as u32 != hp.mask_token) {
            return Err(format!(
                "block {b}: tokens {tokens:?}, id_last {}, mask {}",
                v.id_last, hp.mask_token
            )
            .into());
        }

        // The mask must name exactly the keys the ik arm feeds.
        let mrow = set.find(b, "block", "CUDA0#dsv4_dflash_kq_mask_swa#0", 0)?;
        let mask = set.f32s(mrow)?;
        let n_keys = mrow.ne[0];
        for t in 0..w {
            let seen: Vec<usize> = (0..n_keys)
                .filter(|&k| mask[t * n_keys + k] == 0.0)
                .collect();
            let mut want: Vec<usize> = (0..ring_rows).collect();
            want.extend(tail[..=t].iter().map(|&r| r as usize));
            if seen != want {
                return Err(format!(
                    "block {b} row {t}: the mask sees {} keys ({:?}…), the ik arm feeds ring 0..{ring_rows} and tail {:?}",
                    seen.len(),
                    &seen[..seen.len().min(4)],
                    &tail[..=t]
                )
                .into());
            }
        }

        let hd = hp.head_dim;
        let rings = (0..hp.n_layer)
            .map(|l| {
                let r = set.f32s(set.find(b, "block", &format!("dflash_k_ctx_cache_{l}"), 0)?)?;
                Ok(r[..ring_rows * hd]
                    .iter()
                    .map(|&x| f32_to_f16_bits(x))
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        Ok(BlockSet {
            b,
            width: w,
            id_last: v.id_last,
            pos,
            ring_rows,
            rings,
            feats,
        })
    }

    /// Seat ik's committed rows in our rings, every other slot zero.
    fn seat(
        gpu: &Gpu,
        hp: &DraftHparams,
        bs: &BlockSet,
        rings: &mut DraftRings,
    ) -> Result<(), GateError> {
        let hd = hp.head_dim;
        for l in 0..hp.n_layer {
            let mut ring = vec![0u16; hp.window * hd];
            for r in 0..bs.ring_rows {
                let slot = r % hp.window;
                ring[slot * hd..(slot + 1) * hd]
                    .copy_from_slice(&bs.rings[l][r * hd..(r + 1) * hd]);
            }
            rings
                .ring_mut(l)
                .ok_or("a ring")?
                .buf_mut()
                .copy_from_host(gpu.stream(), &ring)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------ our taps

    struct LayerTaps {
        mix_a: Vec<f32>,
        hc_a: Vec<f32>,
        normed: Vec<f32>,
        kv: Vec<f32>,
        kv_row: Vec<f32>,
        q_a: Vec<f32>,
        q_a_n: Vec<f32>,
        q: Vec<f32>,
        y: Vec<f32>,
        wo_a: Vec<f32>,
        out_a: Vec<f32>,
        streams_a: Vec<f32>,
        fold_a: Vec<f32>,
        mix_f: Vec<f32>,
        hc_f: Vec<f32>,
        normed_f: Vec<f32>,
        logits: Vec<f32>,
        probs: Vec<f32>,
        ids: Vec<u32>,
        weights: Vec<f32>,
        h: Vec<f32>,
        moe: Vec<f32>,
        sh_h: Vec<f32>,
        sh_y: Vec<f32>,
        ffn_out: Vec<f32>,
        streams_f: Vec<f32>,
        fold_f: Vec<f32>,
    }

    struct Ours {
        streams0: Vec<f32>,
        layers: Vec<LayerTaps>,
        head_normed: Vec<f32>,
        /// Logits before the Markov loop, then after it; row-major per row.
        base: Vec<Vec<f32>>,
        result: Vec<Vec<f32>>,
        tokens: Vec<u32>,
    }

    fn read_layer(gpu: &Gpu, b: &LayerBufs) -> Result<LayerTaps, GateError> {
        let s = gpu.stream();
        let f = |x: &cuda_core::DeviceBuffer<f32>| x.to_host_vec(s);
        Ok(LayerTaps {
            mix_a: f(&b.mix_a)?,
            hc_a: f(&b.hc_a)?,
            normed: f(&b.normed)?,
            kv: f(&b.kv)?,
            kv_row: f(&b.kv_row)?,
            q_a: f(&b.q_a)?,
            q_a_n: f(&b.q_a_n)?,
            q: f(&b.q)?,
            y: f(&b.y)?,
            wo_a: f(&b.wo_a)?,
            out_a: f(&b.out_a)?,
            streams_a: f(&b.streams_a)?,
            fold_a: f(&b.fold_a)?,
            mix_f: f(&b.mix_f)?,
            hc_f: f(&b.hc_f)?,
            normed_f: f(&b.normed_f)?,
            logits: f(&b.router.logits)?,
            probs: f(&b.router.probs)?,
            ids: b.router.ids.to_host_vec(s)?,
            weights: f(&b.router.weights)?,
            h: f(&b.h)?,
            moe: f(&b.moe)?,
            sh_h: f(&b.sh_h)?,
            sh_y: f(&b.sh_y)?,
            ffn_out: f(&b.ffn_out)?,
            streams_f: f(&b.streams_f)?,
            fold_f: f(&b.fold_f)?,
        })
    }

    /// The gemv layout `v · m + c` as `m` rows.
    fn rows_of(logits: &[f32], m: usize, n_vocab: usize) -> Vec<Vec<f32>> {
        (0..m)
            .map(|c| (0..n_vocab).map(|v| logits[v * m + c]).collect())
            .collect()
    }

    /// Stage, run the body, read every tap, run the Markov loop, read the rest.
    fn run_pass(
        gpu: &Gpu,
        w: &DraftWeights,
        pass: &mut BlockPass,
        rings: &DraftRings,
        input: &BlockInput<'_>,
    ) -> Result<Ours, GateError> {
        let s = gpu.stream();
        let m = input.width;
        pass.stage(s, input)?;
        pass.enqueue_body(gpu, w, rings)?;
        s.synchronize()?;
        let layers = (0..w.hp().n_layer)
            .map(|l| read_layer(gpu, pass.layer(l).ok_or("a layer")?))
            .collect::<Result<_, _>>()?;
        let n_vocab = pass.head().n_vocab();
        let base = rows_of(&pass.head().logits_to_host(s, m)?, m, n_vocab);
        let head_normed = pass.head().normed_to_host(s, m)?;
        let streams0 = pass.streams0().to_host_vec(s)?;
        pass.enqueue_markov(gpu, w)?;
        s.synchronize()?;
        let result = rows_of(&pass.head().logits_to_host(s, m)?, m, n_vocab);
        Ok(Ours {
            streams0,
            layers,
            head_normed,
            base,
            result,
            tokens: pass.tokens(s)?,
        })
    }

    // ------------------------------------------------------------ the bands

    /// A q8_1 activation's relative noise power at `x` (rows of `k`),
    /// `block`-value blocks: the worst row's `Σ block·d²/12 / Σ x²`.
    fn q8_site(x: &[f32], k: usize, block: usize) -> f64 {
        x.chunks(k)
            .map(|row| {
                let sq: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                let noise: f64 = row
                    .chunks(block)
                    .map(|b| {
                        let d = b.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max) / 127.0;
                        b.len() as f64 * d * d / 12.0
                    })
                    .sum();
                if sq > 0.0 { noise / sq } else { 0.0 }
            })
            .fold(0.0, f64::max)
    }

    /// Relative RMS distance of each row.
    fn rho_rows(ours: &[Vec<f32>], ik: &[Vec<f32>]) -> Vec<f64> {
        ours.iter()
            .zip(ik)
            .map(|(a, b)| {
                let (mut d2, mut r2) = (0.0f64, 0.0f64);
                for (x, y) in a.iter().zip(b) {
                    let d = f64::from(*x) - f64::from(*y);
                    d2 += d * d;
                    r2 += f64::from(*y) * f64::from(*y);
                }
                if r2 > 0.0 {
                    (d2 / r2).sqrt()
                } else if d2 > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// `m` rows of `k` from a flat token-major vector.
    fn split(v: &[f32], k: usize, m: usize) -> Vec<Vec<f32>> {
        (0..m).map(|t| v[t * k..(t + 1) * k].to_vec()).collect()
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        IkMask,
        Reference,
    }

    impl Arm {
        fn label(self) -> &'static str {
            match self {
                Arm::IkMask => "ik-rule",
                Arm::Reference => "reference",
            }
        }
        fn rule(self) -> Rule {
            match self {
                Arm::IkMask => Rule::Ik,
                Arm::Reference => Rule::Reference,
            }
        }
    }

    /// Where in a layer a tap sits.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Phase {
        /// Up to the attention.
        PreAttn,
        /// From the attention's output to the router, and the shared expert.
        PostAttn,
        /// After the routing: the routed experts and what they feed.
        PostRoute,
    }

    /// A tap's point in the walk, in graph order. A row leaves ik's path at
    /// one of these points; from there on its taps are printed, not judged.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Stage {
        Layer(usize, Phase),
        /// The head's norm and projection.
        Head,
        /// The Markov loop, where row `c` reads row `c - 1`'s id.
        Markov,
    }

    impl Stage {
        fn label(self) -> String {
            match self {
                Stage::Layer(l, _) => format!("L{l}"),
                Stage::Head | Stage::Markov => "head".to_string(),
            }
        }
    }

    /// One arm's walk through one block's taps.
    struct Judge {
        arm: Arm,
        b: i32,
        m: usize,
        n_layer: usize,
        /// The prediction's running noise power.
        acc: f64,
        /// Per row: where it left ik's path and why; `None` while on it.
        off: Vec<Option<(Stage, String)>>,
        ok: bool,
        judged: usize,
        printed_rows: usize,
    }

    impl Judge {
        /// Every row on ik's path, except under the reference rule, where
        /// ik's block-causal mask is an oracle defect: row `t < m - 1` sees
        /// rows ik hides from layer 0's attention on, and the last row, which
        /// sees the same rows under both, reads the others' keys from layer
        /// 1's attention on.
        fn new(arm: Arm, b: i32, m: usize, n_layer: usize) -> Judge {
            let mask = |at: Stage| Some((at, "oracle defect: ik's block-causal mask".to_string()));
            let off = (0..m)
                .map(|t| match arm {
                    Arm::IkMask => None,
                    Arm::Reference if t + 1 < m => mask(Stage::Layer(0, Phase::PostAttn)),
                    Arm::Reference if m > 1 && n_layer > 1 => {
                        mask(Stage::Layer(1, Phase::PostAttn))
                    }
                    Arm::Reference => None,
                })
                .collect();
            Judge {
                arm,
                b,
                m,
                n_layer,
                acc: 0.0,
                off,
                ok: true,
                judged: 0,
                printed_rows: 0,
            }
        }

        fn site(&mut self, power: f64) {
            self.acc += power;
        }

        fn band(&self) -> f64 {
            (BAND_GAIN * self.acc.sqrt()).max(FLOOR)
        }

        /// Is row `t` still on ik's path at `st`?
        fn on_path(&self, st: Stage, t: usize) -> bool {
            self.off[t].as_ref().is_none_or(|(at, _)| st < *at)
        }

        /// Row `t` leaves ik's path at `st`, unless it left earlier.
        fn leave(&mut self, t: usize, st: Stage, why: &str) {
            if self.on_path(st, t) {
                self.off[t] = Some((st, why.to_string()));
            }
        }

        /// A tie in layer `l`'s routing of row `c`: that row's routed experts
        /// differ from here on, and every row whose attention reads row `c`
        /// (the rows after it under ik's rule, all of them under the
        /// reference) differs from the next layer's attention on.
        fn route_tie(&mut self, l: usize, c: usize) {
            let why = format!("router tie at L{l} row {c}");
            self.leave(c, Stage::Layer(l, Phase::PostRoute), &why);
            if l + 1 < self.n_layer {
                let readers = match self.arm {
                    Arm::IkMask => c..self.m,
                    Arm::Reference => 0..self.m,
                };
                for r in readers {
                    self.leave(r, Stage::Layer(l + 1, Phase::PostAttn), &why);
                }
            }
        }

        /// Row `c`'s Markov step reads row `c - 1`'s id: where ours differs
        /// from ik's, row `c` leaves the path in the loop.
        fn ids_differ(&mut self, ours: &[u32], ik: &[i32]) {
            for (c, (&a, &t)) in ours.iter().zip(ik).enumerate().take(self.m - 1) {
                if a != t as u32 {
                    self.leave(
                        c + 1,
                        Stage::Markov,
                        &format!("row {c} proposed another id"),
                    );
                }
            }
        }

        /// Why the rows off ik's path at `st` left it.
        fn why_off(&self, st: Stage) -> String {
            let mut why: Vec<&str> = Vec::new();
            for (at, w) in self.off.iter().flatten() {
                if *at <= st && !why.contains(&w.as_str()) {
                    why.push(w);
                }
            }
            why.join("; ")
        }

        fn tap(&mut self, what: &str, st: Stage, ours: &[Vec<f32>], ik: &[Vec<f32>]) {
            let rho = rho_rows(ours, ik);
            let band = self.band();
            let mut pass = true;
            let mut judged = Vec::new();
            for (t, r) in rho.iter().enumerate() {
                if self.on_path(st, t) {
                    judged.push(t);
                    pass &= *r <= band;
                } else {
                    self.printed_rows += 1;
                }
            }
            let rows: Vec<String> = rho.iter().map(|r| format!("{r:.2e}")).collect();
            let note = if judged.len() == rho.len() {
                verdict(pass).to_string()
            } else if judged.is_empty() {
                format!("printed only ({})", self.why_off(st))
            } else {
                format!(
                    "rows {judged:?} judged {}, the others printed ({})",
                    verdict(pass),
                    self.why_off(st)
                )
            };
            println!(
                "{NAME}: {} b{} {} {what}: rho_hat {:.2e} band {band:.2e} | rho by row [{}] | {note}",
                self.arm.label(),
                self.b,
                st.label(),
                self.acc.sqrt(),
                rows.join(" ")
            );
            if !judged.is_empty() {
                self.judged += 1;
                self.ok &= pass;
            }
        }
    }

    /// ik's HC_PRE node (`[pre S T][post S T][comb S S T]`) in our layout.
    fn hc_ik(node: &[f32], m: usize) -> Vec<Vec<f32>> {
        (0..m)
            .map(|t| {
                let mut r = node[4 * t..4 * t + 4].to_vec();
                r.extend_from_slice(&node[4 * m + 4 * t..4 * m + 4 * t + 4]);
                r.extend_from_slice(&node[8 * m + 16 * t..8 * m + 16 * t + 16]);
                r
            })
            .collect()
    }

    struct Cx<'a> {
        set: &'a Set,
        hp: &'a DraftHparams,
        n_vocab: usize,
        /// Per layer, the shared expert's gate and up rows, dequantized.
        shexp: Vec<(Vec<f32>, Vec<f32>)>,
    }

    /// Q8_0 tensor `name` of the draft, dequantized.
    fn q8_rows(draft: &Split, name: &str) -> Result<Vec<f32>, GateError> {
        let (sh, t) = draft
            .find(name)
            .ok_or_else(|| format!("{name} is not in the draft"))?;
        if t.ty != GgmlType::Q8_0 {
            return Err(format!("{name} is {}, not Q8_0", t.ty).into());
        }
        let n: u64 = t.dims.iter().product();
        let mut w = vec![0.0f32; n as usize];
        dequant_row(
            GgmlType::Q8_0,
            draft.shard(sh).ok_or("no shard")?.data(t)?,
            &mut w,
        )?;
        Ok(w)
    }

    /// Which SwiGLU rule ik's shared expert follows: from ik's own ffn norm
    /// `x` and the file's weights in f64, each candidate rule's distance to
    /// ik's `ffn_up_gate`. Printed, not judged.
    fn shexp_rules(cx: &Cx<'_>, b: i32, l: usize, m: usize, x: &[f32], ik_h: &[f32]) {
        let (hp, (wg, wu)) = (cx.hp, &cx.shexp[l]);
        let (k, ff) = (hp.n_embd, hp.experts.ff * hp.experts.n_shared);
        let lim = f64::from(hp.swiglu_limit_shared[l]);
        let silu = |g: f64| g / (1.0 + (-g).exp());
        let clamp = |u: f64| u.clamp(-lim, lim);
        // A rule: `h` from the gate and up projections.
        type SwiGlu<'a> = &'a dyn Fn(f64, f64) -> f64;
        let rules: [(&str, SwiGlu<'_>); 5] = [
            ("min(silu(g),L)*clamp(u) (ours)", &|g, u| {
                silu(g).min(lim) * clamp(u)
            }),
            ("silu(min(g,L))*clamp(u) (model.py)", &|g, u| {
                silu(g.min(lim)) * clamp(u)
            }),
            ("silu(g)*u (no clamp)", &|g, u| silu(g) * u),
            ("min(silu(g),L)*u", &|g, u| silu(g).min(lim) * u),
            ("silu(g)*clamp(u)", &|g, u| silu(g) * clamp(u)),
        ];
        let (mut over_g, mut over_u) = (0usize, 0usize);
        let mut gu = Vec::with_capacity(m * ff);
        for t in 0..m {
            let xt = &x[t * k..(t + 1) * k];
            for r in 0..ff {
                let dot = |w: &[f32]| -> f64 {
                    w[r * k..(r + 1) * k]
                        .iter()
                        .zip(xt)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum()
                };
                let (g, u) = (dot(wg), dot(wu));
                over_g += usize::from(silu(g) > lim);
                over_u += usize::from(u.abs() > lim);
                gu.push((g, u));
            }
        }
        let line: Vec<String> = rules
            .iter()
            .map(|(name, f)| {
                let h: Vec<f32> = gu.iter().map(|&(g, u)| f(g, u) as f32).collect();
                let rho = rho_rows(&split(&h, ff, m), &split(ik_h, ff, m));
                let r: Vec<String> = rho.iter().map(|r| format!("{r:.2e}")).collect();
                format!("{name}: [{}]", r.join(" "))
            })
            .collect();
        println!(
            "{NAME}: diag b{b} L{l} shexp rule vs ik's ffn_up_gate, limit {lim}, silu(g) > L {over_g}, |u| > L {over_u} of {}: {}",
            m * ff,
            line.join("; ")
        );
    }

    /// Every tap of one layer.
    fn layer_taps(cx: &Cx<'_>, j: &mut Judge, l: usize, o: &LayerTaps) -> Result<(), GateError> {
        let (set, hp, b, m) = (cx.set, cx.hp, j.b, j.m);
        let n = hp.n_embd;
        let ik = |r: &Row| set.f32s(r);
        let pre = Stage::Layer(l, Phase::PreAttn);
        let post = Stage::Layer(l, Phase::PostAttn);
        let route = Stage::Layer(l, Phase::PostRoute);
        let hd = hp.head_dim;
        let blk = |s: &str| format!("blk.{l}.{s}.weight");

        let mix = ik(set.find(b, "block", &format!("hc_pre_mixes-{l}"), 0)?)?;
        j.tap(
            "hc_attn mixes",
            pre,
            &split(&o.mix_a, 24, m),
            &split(&mix, 24, m),
        );
        let node = ik(set.find_op(b, "HC_PRE", "", &blk("hc_attn_scale"))?)?;
        j.tap(
            "hc_attn pre/post/comb",
            pre,
            &split(&o.hc_a, 24, m),
            &hc_ik(&node, m),
        );
        let an = ik(set.find_op(b, "FUSED_RMS_NORM", "", &blk("attn_norm"))?)?;
        j.tap("attn_norm", pre, &split(&o.normed, n, m), &split(&an, n, m));

        j.site(q8_site(&an, n, 32));
        let kv = ik(set.find_op(b, "MUL_MAT", &blk("attn_kv"), "")?)?;
        j.tap("attn_kv", pre, &split(&o.kv, hd, m), &split(&kv, hd, m));
        let kvr = ik(set.find(b, "block", &format!("dsv4_dflash_kv-{l}"), 0)?)?;
        j.tap(
            "dsv4_dflash_kv (turned)",
            pre,
            &split(&o.kv_row, hd, m),
            &split(&kvr, hd, m),
        );
        let qa = ik(set.find_op(b, "MUL_MAT", &blk("attn_q_a"), "")?)?;
        j.tap(
            "q_a",
            pre,
            &split(&o.q_a, hp.q_lora_rank, m),
            &split(&qa, hp.q_lora_rank, m),
        );
        let qan = ik(set.find_op(b, "FUSED_RMS_NORM", "", &blk("attn_q_a_norm"))?)?;
        j.tap(
            "q_a_norm",
            pre,
            &split(&o.q_a_n, hp.q_lora_rank, m),
            &split(&qan, hp.q_lora_rank, m),
        );
        j.site(q8_site(&qan, hp.q_lora_rank, 32));
        let hl = hp.n_head * hd;
        let q = ik(set.find(b, "block", &format!("dsv4_dflash_q-{l}"), 0)?)?;
        j.tap(
            "dsv4_dflash_q (turned)",
            pre,
            &split(&o.q, hl, m),
            &split(&q, hl, m),
        );

        j.site(ATTN_SITE * ATTN_SITE);
        let y = ik(set.find(
            b,
            "block",
            &format!("dsv4_dflash_attn-{l} (reshaped) (view)"),
            0,
        )?)?;
        j.tap(
            "dsv4_dflash_attn (inverse rope)",
            post,
            &split(&o.y, hl, m),
            &split(&y, hl, m),
        );
        j.site(q8_site(&y, hl / hp.o_groups, 32));
        let go = hp.o_groups * hp.o_lora_rank;
        let wo_a = ik(set.nth_op(b, "CONT", l)?)?;
        j.tap("wo_a", post, &split(&o.wo_a, go, m), &split(&wo_a, go, m));
        j.site(q8_site(&wo_a, go, 32));
        let wob = set.find_op(b, "MUL_MAT", &blk("attn_output_b"), "")?;
        j.tap(
            "wo_b",
            post,
            &split(&o.out_a, n, m),
            &split(&ik(wob)?, n, m),
        );
        let hpa = set.find_op(b, "HC_POST", &wob.name, "")?;
        j.tap(
            "hc_post attn streams",
            post,
            &split(&o.streams_a, 4 * n, m),
            &split(&ik(hpa)?, 4 * n, m),
        );
        let fa = ik(set.find_op(b, "MUL_MULTI_ADD", &hpa.name, "")?)?;
        j.tap("ffn fold", post, &split(&o.fold_a, n, m), &split(&fa, n, m));

        let mixf = ik(set.find(b, "block", &format!("hc_pre_mixes-{l}"), 1)?)?;
        j.tap(
            "hc_ffn mixes",
            post,
            &split(&o.mix_f, 24, m),
            &split(&mixf, 24, m),
        );
        let nodef = ik(set.find_op(b, "HC_PRE", "", &blk("hc_ffn_scale"))?)?;
        j.tap(
            "hc_ffn pre/post/comb",
            post,
            &split(&o.hc_f, 24, m),
            &hc_ik(&nodef, m),
        );
        let fnrm = ik(set.find_op(b, "FUSED_RMS_NORM", "", &blk("ffn_norm"))?)?;
        j.tap(
            "ffn_norm",
            post,
            &split(&o.normed_f, n, m),
            &split(&fnrm, n, m),
        );

        router_taps(cx, j, l, o)?;

        j.site(q8_site(&fnrm, n, 32) + q8_site(&fnrm, n, 128));
        let ff = hp.experts.ff;
        let gp = ik(set.find(b, "block", &format!("ffn_moe_gate_par-{l}"), 0)?)?;
        let ours_h: Vec<Vec<f32>> = (0..m)
            .map(|c| {
                (0..3)
                    .flat_map(|u| {
                        let at = ((3 * c + u) * m + c) * ff;
                        o.h[at..at + ff].to_vec()
                    })
                    .collect()
            })
            .collect();
        j.tap(
            "ffn_moe_gate_par (h)",
            route,
            &ours_h,
            &split(&gp, 3 * ff, m),
        );
        j.site(q8_site(&gp, ff, 32) + q8_site(&gp, ff, 128));
        let mw = ik(set.find(b, "block", &format!("ffn_moe_weighted-{l}"), 0)?)?;
        j.tap(
            "ffn_moe_weighted",
            route,
            &split(&o.moe, n, m),
            &split(&mw, n, m),
        );
        let ffs = ff * hp.experts.n_shared;
        let shh = ik(set.find(b, "block", &format!("ffn_up_gate-{l}"), 0)?)?;
        j.tap(
            "shexp h",
            post,
            &split(&o.sh_h, ffs, m),
            &split(&shh, ffs, m),
        );
        if j.arm == Arm::IkMask {
            shexp_rules(cx, b, l, m, &fnrm, &shh);
        }
        j.site(q8_site(&shh, ffs, 32));
        let shy = ik(set.find(b, "block", &format!("ffn_down-{l}"), 0)?)?;
        j.tap(
            "shexp down",
            post,
            &split(&o.sh_y, n, m),
            &split(&shy, n, m),
        );
        let add = ik(set.find_op(b, "ADD", &format!("ffn_moe_weighted-{l}"), "")?)?;
        j.tap(
            "routed + shared",
            route,
            &split(&o.ffn_out, n, m),
            &split(&add, n, m),
        );
        let hpf = set.find_op(
            b,
            "HC_POST",
            &set.find_op(b, "ADD", &format!("ffn_moe_weighted-{l}"), "")?
                .name,
            "",
        )?;
        j.tap(
            "hc_post ffn streams",
            route,
            &split(&o.streams_f, 4 * n, m),
            &split(&ik(hpf)?, 4 * n, m),
        );
        let ff_fold = ik(set.find_op(b, "MUL_MULTI_ADD", &hpf.name, "")?)?;
        j.tap(
            "next fold",
            route,
            &split(&o.fold_f, n, m),
            &split(&ff_fold, n, m),
        );
        Ok(())
    }

    /// The router: logits and scores within the band, the ids exact but for
    /// a tie, the scaled weights within the band.
    fn router_taps(cx: &Cx<'_>, j: &mut Judge, l: usize, o: &LayerTaps) -> Result<(), GateError> {
        let (set, b, m) = (cx.set, j.b, j.m);
        let post = Stage::Layer(l, Phase::PostAttn);
        let route = Stage::Layer(l, Phase::PostRoute);
        let ne = 128;
        j.site(BF16_SITE * BF16_SITE);
        let lg = set.f32s(set.find(b, "block", &format!("ffn_moe_logits-{l}"), 0)?)?;
        j.tap(
            "router logits",
            post,
            &split(&o.logits, ne, m),
            &split(&lg, ne, m),
        );
        let pr = set.f32s(set.find(b, "block", &format!("ffn_moe_probs-{l}"), 0)?)?;
        let (ours_scores, ik_scores) = (split(&o.probs, ne, m), split(&pr, ne, m));
        j.tap("router scores", post, &ours_scores, &ik_scores);
        // A flip is a tie only on a row whose scores are within their band.
        let scores_rho = rho_rows(&ours_scores, &ik_scores);
        let biased = set.f32s(set.find(b, "block", &format!("ffn_moe_probs_biased-{l}"), 0)?)?;
        let topk = set.i32s(
            set.find(b, "block", &format!("ffn_moe_topk-{l}"), 0)?,
            ".logical",
        )?;
        let ws = set.f32s(set.find(b, "block", &format!("ffn_moe_weights_scaled-{l}"), 0)?)?;
        let mut w_ours = Vec::new();
        let mut w_ik = Vec::new();
        for c in 0..m {
            let mut ours: Vec<u32> = o.ids[3 * c..3 * c + 3].to_vec();
            let mut theirs: Vec<u32> = topk[3 * c..3 * c + 3].iter().map(|&i| i as u32).collect();
            let (so, st) = (ours.clone(), theirs.clone());
            ours.sort_unstable();
            theirs.sort_unstable();
            if ours == theirs {
                let pick = |ids: &[u32], w: &[f32], id: u32| {
                    w[ids.iter().position(|&x| x == id).unwrap_or(0)]
                };
                w_ours.push(
                    theirs
                        .iter()
                        .map(|&id| pick(&so, &o.weights[3 * c..3 * c + 3], id))
                        .collect::<Vec<f32>>(),
                );
                w_ik.push(
                    theirs
                        .iter()
                        .map(|&id| pick(&st, &ws[3 * c..3 * c + 3], id))
                        .collect::<Vec<f32>>(),
                );
                continue;
            }
            if !j.on_path(post, c) {
                println!(
                    "{NAME}: {} b{b} L{l} router row {c}: ids {so:?} vs ik {st:?} — off ik's path ({}), printed",
                    j.arm.label(),
                    j.off[c].as_ref().map_or("", |(_, why)| why.as_str())
                );
                w_ours.push(vec![0.0; 3]);
                w_ik.push(vec![0.0; 3]);
                continue;
            }
            let row = &biased[c * ne..(c + 1) * ne];
            let mut sorted: Vec<f32> = row.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let gap = f64::from(sorted[2] - sorted[3]);
            let tie_band = j.band() * f64::from(sorted[0].abs());
            let tie = gap <= tie_band && scores_rho[c] <= j.band();
            println!(
                "{NAME}: {} b{b} L{l} router row {c}: ids {so:?} vs ik {st:?}; ik's 3rd-4th biased gap {gap:.3e}, tie band {tie_band:.3e}, scores rho {:.2e} {}",
                j.arm.label(),
                scores_rho[c],
                if tie { "TIE" } else { "FAIL" }
            );
            if tie {
                j.route_tie(l, c);
            } else {
                j.ok = false;
            }
            w_ours.push(vec![0.0; 3]);
            w_ik.push(vec![0.0; 3]);
        }
        println!(
            "{NAME}: {} b{b} L{l} router ids equal ik's on {}/{m} rows (as sets)",
            j.arm.label(),
            w_ik.iter().filter(|r| r.iter().any(|v| *v != 0.0)).count()
        );
        j.tap("router scaled weights", route, &w_ours, &w_ik);
        Ok(())
    }

    fn head_taps(cx: &Cx<'_>, j: &mut Judge, o: &Ours) -> Result<(), GateError> {
        let (set, hp, b, m) = (cx.set, cx.hp, j.b, j.m);
        let n = hp.n_embd;
        let hn = set.f32s(set.find_op(b, "FUSED_RMS_NORM", "", "output_norm.weight")?)?;
        j.tap(
            "output_norm",
            Stage::Head,
            &split(&o.head_normed, n, m),
            &split(&hn, n, m),
        );
        j.site(q8_site(&hn, n, 32) + q8_site(&hn, n, 128));
        let base = set.f32s(set.find(b, "block", "dflash_base_result_output", 0)?)?;
        let base_rows = split(&base, cx.n_vocab, m);
        j.tap(
            "dflash_base_result_output",
            Stage::Head,
            &o.base,
            &base_rows,
        );
        let result = set.f32s(set.find(b, "block", "result_output", 0)?)?;
        let res_rows = split(&result, cx.n_vocab, m);
        j.site(markov_site(&base, &result, cx.n_vocab));
        let ik_tok = set.i32s(set.find(b, "block", "draft_argmax", 0)?, "")?;
        j.ids_differ(&o.tokens, &ik_tok);
        j.tap("result_output", Stage::Markov, &o.result, &res_rows);
        argmax_check(j, &o.tokens, &ik_tok, &res_rows);
        Ok(())
    }

    /// Our proposal against ik's on the rows still on its path: equal, or a
    /// tie by ik's own margin.
    fn argmax_check(j: &mut Judge, ours: &[u32], ik_tok: &[i32], ik_rows: &[Vec<f32>]) {
        let st = Stage::Markov;
        let mut line = Vec::new();
        let mut pass = true;
        let mut judged = 0;
        for (c, (&a, &t)) in ours.iter().zip(ik_tok).enumerate() {
            let on = j.on_path(st, c);
            judged += usize::from(on);
            if a == t as u32 {
                line.push(format!("{a}"));
                continue;
            }
            let row = &ik_rows[c];
            let rms = (row
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / row.len() as f64)
                .sqrt();
            let margin = f64::from(row[t as usize] - row[a as usize]);
            let tie = margin <= j.band() * rms;
            line.push(format!(
                "{a}≠{t}(margin {margin:.3e}, band {:.3e}, {})",
                j.band() * rms,
                match (on, tie) {
                    (false, _) => "off path",
                    (true, true) => "tie",
                    (true, false) => "FAIL",
                }
            ));
            pass &= !on || tie;
        }
        let note = if judged == ours.len() {
            verdict(pass).to_string()
        } else if judged == 0 {
            format!("printed only ({})", j.why_off(st))
        } else {
            format!(
                "{judged} rows judged {}, the others printed ({})",
                verdict(pass),
                j.why_off(st)
            )
        };
        println!(
            "{NAME}: {} b{} draft_argmax ours vs ik: [{}] {note}",
            j.arm.label(),
            j.b,
            line.join(", "),
        );
        if judged > 0 {
            j.ok &= pass;
        }
    }

    /// Every tap of one arm over one block.
    fn judge_block(
        cx: &Cx<'_>,
        arm: Arm,
        bs: &BlockSet,
        o: &Ours,
    ) -> Result<(bool, usize, usize), GateError> {
        let (set, hp) = (cx.set, cx.hp);
        let m = bs.width;
        let mut j = Judge::new(arm, bs.b, m, hp.n_layer);
        let init = set.f32s(set.find(bs.b, "block", "dsv4_dflash_hc_init", 0)?)?;
        let n4 = 4 * hp.n_embd;
        let same = o.streams0[..m * n4]
            .iter()
            .zip(&init)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{NAME}: {} b{} dsv4_dflash_hc_init (the embedding in four streams) bit-identical: {same} {}",
            arm.label(),
            bs.b,
            verdict(same)
        );
        j.ok &= same;
        for (l, taps) in o.layers.iter().enumerate() {
            layer_taps(cx, &mut j, l, taps)?;
        }
        head_taps(cx, &mut j, o)?;
        let why = j.why_off(Stage::Markov);
        println!(
            "{NAME}: {} b{}: {} taps judged, {} rows printed ({}) {}",
            arm.label(),
            bs.b,
            j.judged,
            j.printed_rows,
            if why.is_empty() {
                "every row on ik's path throughout"
            } else {
                &why
            },
            verdict(j.ok)
        );
        Ok((j.ok, j.judged, j.printed_rows))
    }

    /// Leading ids equal to the target's.
    fn accepted(ours: &[u32], targets: &[u32]) -> usize {
        ours.iter().zip(targets).take_while(|(a, b)| a == b).count()
    }

    fn input<'a>(bs: &BlockSet, row: &'a [u8], width: usize, rule: Rule) -> BlockInput<'a> {
        BlockInput {
            id_last: bs.id_last,
            row,
            width,
            first_pos: bs.pos[0] as u32,
            ring_rows: bs.ring_rows,
            rule,
        }
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|r| r.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    pub fn run() -> Result<(), GateError> {
        let path = std::env::var_os("BLOOMERY_DSPARK_MODEL")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .ok_or("BLOOMERY_DSPARK_MODEL unset — run through `just gate-gpu-dspark-graph`")?;
        let draft = Split::open(&path)?;
        let target = Arc::new(Split::open(ref_model_path()?)?);
        let gpu = Gpu::new()?;
        println!("{NAME}: card {}", gpu.device_name()?);
        let weights = DraftWeights::load(gpu.stream(), &draft, &target)?;
        let hp = weights.hp().clone();
        println!(
            "{NAME}: draft {} layers, block_size {} in the file, width {SET_WIDTH} (the set's) and {WIDE} (synthetic) gated",
            hp.n_layer, hp.block_size
        );
        let mut body = DraftBody::new(&gpu, weights, Arc::clone(&target))?;
        let set = read_set(&data_dir().join("ref-draft").join(DSREF_SET))?;
        let mut ok = true;
        let s = gpu.stream();

        let w = body.weights();
        let mut pass = BlockPass::new(&gpu, w)?;
        let mut rings = DraftRings::new(s, &hp)?;
        let cx = Cx {
            set: &set,
            hp: &hp,
            n_vocab: pass.head().n_vocab(),
            shexp: (0..hp.n_layer)
                .map(|l| {
                    Ok((
                        q8_rows(&draft, &names::ffn_gate_shexp(l))?,
                        q8_rows(&draft, &names::ffn_up_shexp(l))?,
                    ))
                })
                .collect::<Result<_, GateError>>()?,
        };

        // Launches: the gate's own capture of one pass per width, and the
        // graphs DraftBody captured at load — every node a kernel.
        let b0 = block_set(&set, &hp, 0)?;
        let row0 = embedding_row(&target, hp.n_embd, b0.id_last)?;
        seat(&gpu, &hp, &b0, &mut rings)?;
        for m in 1..=MAX_WIDTH {
            let nodes = pass.capture(&gpu, w, &rings, m)?.node_count();
            let g = body.pass_graph(m).ok_or("a pass graph per width")?;
            let kernels = all_kernels(g)?;
            let pass_ok = nodes == PASS_NODES[m - 1]
                && pass.launches(m) == PASS_NODES[m - 1]
                && g.node_count() == PASS_NODES[m - 1]
                && body.pass_launches(m) == PASS_NODES[m - 1]
                && kernels;
            println!(
                "{NAME}: launches w={m}: gate capture {nodes} nodes, DraftBody graph {} nodes (all kernel nodes {kernels}), \
                 BlockPass::launches {}, pinned {} {}",
                g.node_count(),
                pass.launches(m),
                PASS_NODES[m - 1],
                verdict(pass_ok)
            );
            ok &= pass_ok;
        }
        for n in 1..=GROUP {
            let g = body
                .append_graph(n)
                .ok_or("an append graph per row count")?;
            let kernels = all_kernels(g)?;
            let append_ok = g.node_count() == APPEND_NODES
                && body.append_launches(n) == APPEND_NODES
                && kernels;
            println!(
                "{NAME}: launches append n={n}: DraftBody graph {} nodes (all kernel nodes {kernels}), \
                 KvAppend::launches {}, pinned {APPEND_NODES} {}",
                g.node_count(),
                body.append_launches(n),
                verdict(append_ok)
            );
            ok &= append_ok;
        }

        // Every tap, both arms, blocks 0-2; argmax and accept counts, every block.
        let mut accept = Vec::new();
        let mut eager_ref = Vec::new();
        for b in set.blocks() {
            let bs = block_set(&set, &hp, b)?;
            let row = embedding_row(&target, hp.n_embd, bs.id_last)?;
            seat(&gpu, &hp, &bs, &mut rings)?;
            let v = set.verify_of(b)?;
            let mut line = (b, v.accepted, 0usize, 0usize);
            for arm in [Arm::IkMask, Arm::Reference] {
                let o = run_pass(
                    &gpu,
                    w,
                    &mut pass,
                    &rings,
                    &input(&bs, row, bs.width, arm.rule()),
                )?;
                let a = accepted(&o.tokens, v.targets());
                if arm == Arm::IkMask {
                    line.2 = a;
                } else {
                    line.3 = a;
                }
                if TAP_BLOCKS.contains(&b) {
                    let (pass_ok, _, _) = judge_block(&cx, arm, &bs, &o)?;
                    ok &= pass_ok;
                    if arm == Arm::Reference {
                        eager_ref.push((b, after_of(&o, hp.n_layer, bs.width)));
                    }
                } else if arm == Arm::IkMask {
                    let ik_tok = set.i32s(set.find(b, "block", "draft_argmax", 0)?, "")?;
                    let res = set.f32s(set.find(b, "block", "result_output", 0)?)?;
                    let mut j = Judge::new(arm, b, bs.width, hp.n_layer);
                    // The band carried to the logits, from this block's own
                    // tensors, as the tap blocks carry it.
                    judge_quiet(&cx, &mut j, &bs)?;
                    j.ids_differ(&o.tokens, &ik_tok);
                    argmax_check(
                        &mut j,
                        &o.tokens,
                        &ik_tok,
                        &split(&res, cx.n_vocab, bs.width),
                    );
                    ok &= j.ok;
                }
                let drafted: Vec<u32> = set
                    .drafts
                    .iter()
                    .filter(|d| d.0 == b)
                    .map(|d| d.2)
                    .collect();
                println!(
                    "{NAME}: {} b{b} ids {:?} (ik drafted {drafted:?}, target {:?}): accept {a}, ik {}",
                    arm.label(),
                    o.tokens,
                    v.targets(),
                    v.accepted
                );
            }
            accept.push(line);
        }

        // w = 5 over block 0's inputs.
        seat(&gpu, &hp, &b0, &mut rings)?;
        for arm in [Arm::IkMask, Arm::Reference] {
            let narrow = run_pass(
                &gpu,
                w,
                &mut pass,
                &rings,
                &input(&b0, row0, SET_WIDTH, arm.rule()),
            )?;
            let wide = run_pass(
                &gpu,
                w,
                &mut pass,
                &rings,
                &input(&b0, row0, WIDE, arm.rule()),
            )?;
            let again = run_pass(
                &gpu,
                w,
                &mut pass,
                &rings,
                &input(&b0, row0, WIDE, arm.rule()),
            )?;
            let shape =
                wide.tokens.len() == WIDE && wide.tokens.iter().all(|&t| (t as usize) < cx.n_vocab);
            let rerun = bits(&wide.result) == bits(&again.result) && wide.tokens == again.tokens;
            let first_rows = bits(&wide.result[..SET_WIDTH]) == bits(&narrow.result)
                && bits(&wide.base[..SET_WIDTH]) == bits(&narrow.base)
                && wide.tokens[..SET_WIDTH] == narrow.tokens[..];
            let row0_argmax = wide.tokens[0] == narrow.tokens[0];
            let row0_rho = rho_rows(&wide.base[..1], &narrow.base[..1])[0];
            let pass_ok = shape
                && rerun
                && match arm {
                    Arm::IkMask => first_rows,
                    Arm::Reference => true,
                };
            println!(
                "{NAME}: {} w={WIDE} on block 0: ids {:?} (w={SET_WIDTH}: {:?}); shape {shape}; rerun bit-identical {rerun}; \
                 rows 0-{} bit-identical to w={SET_WIDTH} {first_rows}; row 0 argmax equal {row0_argmax}, row 0 base rho {row0_rho:.2e} {}",
                arm.label(),
                wide.tokens,
                narrow.tokens,
                SET_WIDTH - 1,
                verdict(pass_ok)
            );
            ok &= pass_ok;
        }

        drop(rings);
        drop(pass);

        // Graph = eager on DraftBody's own pass and rings: blocks 0-2 of
        // the set at every width, the replay first — a replay that did not
        // re-read its inputs would read the previous call's — then the
        // eager twin, every tap the whole pass leaves, bit for bit.
        for b in TAP_BLOCKS {
            let bs = block_set(&set, &hp, b)?;
            let row = embedding_row(&target, hp.n_embd, bs.id_last)?;
            seat(&gpu, &hp, &bs, body.rings_mut())?;
            for m in 1..=MAX_WIDTH {
                let inp = input(&bs, row, m, Rule::Reference);
                let ids = body.block(&gpu, &inp, Submit::Graph)?;
                let replay = read_after(&gpu, body.pass(), hp.n_layer, m)?;
                body.block(&gpu, &inp, Submit::Eager)?;
                let eager = read_after(&gpu, body.pass(), hp.n_layer, m)?;
                let off = first_difference(&replay, &eager);
                println!(
                    "{NAME}: graph = eager b{b} w={m}: ids {ids:?} (eager {:?}); {} taps; {} {}",
                    eager.tokens,
                    replay.taps.len(),
                    off.as_deref()
                        .map_or("bit-identical".to_string(), |t| format!(
                            "first difference {t}"
                        )),
                    verdict(off.is_none())
                );
                ok &= off.is_none();
            }
        }

        // Three consecutive blocks through the w = 3 graph against phase B's
        // eager passes above (the gate's own pass, the reference rule).
        for (b, eager) in &eager_ref {
            let bs = block_set(&set, &hp, *b)?;
            let row = embedding_row(&target, hp.n_embd, bs.id_last)?;
            seat(&gpu, &hp, &bs, body.rings_mut())?;
            body.block(
                &gpu,
                &input(&bs, row, bs.width, Rule::Reference),
                Submit::Graph,
            )?;
            let replay = read_after(&gpu, body.pass(), hp.n_layer, bs.width)?;
            let off = first_difference(&replay, eager);
            println!(
                "{NAME}: graph b{b} w={} = the judged eager pass: ids {:?} (eager {:?}); {} taps; {} {}",
                bs.width,
                replay.tokens,
                eager.tokens,
                replay.taps.len(),
                off.as_deref()
                    .map_or("bit-identical".to_string(), |t| format!(
                        "first difference {t}"
                    )),
                verdict(off.is_none())
            );
            ok &= off.is_none();
        }

        // DraftBody over the set: the set's features appended at the
        // reference positions, every block's proposal — eager, then through
        // the graphs, the ids and the rings after each block bit for bit.
        let mut eager_run = Vec::new();
        body.reset(&gpu)?;
        for b in set.blocks() {
            let bs = block_set(&set, &hp, b)?;
            body.append_as(&gpu, &bs.feats, Submit::Eager)?;
            let ids = body.propose_as(&gpu, bs.id_last, bs.width, Submit::Eager)?;
            eager_run.push((ids, ring_bits(&gpu, body.rings(), hp.n_layer)?));
        }
        let mut body_accept = Vec::new();
        body.reset(&gpu)?;
        for (b, (e_ids, e_rings)) in set.blocks().into_iter().zip(&eager_run) {
            let bs = block_set(&set, &hp, b)?;
            let n = body.append(&gpu, &bs.feats)?;
            let v = set.verify_of(b)?;
            let ids = body.propose(&gpu, bs.id_last, bs.width)?;
            let same = ids == *e_ids && ring_bits(&gpu, body.rings(), hp.n_layer)? == *e_rings;
            let a = accepted(&ids, v.targets());
            println!(
                "{NAME}: DraftBody b{b} appended {n} ({}) committed {} ids {ids:?} target {:?}: accept {a} (ik {}); \
                 ids and rings = the eager run {same} {}",
                if n <= GROUP {
                    "graph"
                } else {
                    "eager: no graph"
                },
                body.committed(),
                v.targets(),
                v.accepted,
                verdict(same)
            );
            ok &= same;
            body_accept.push(a);
        }

        println!(
            "{NAME}: accept per block — block | ik | ik-rule arm | reference arm | DraftBody (printed, not pinned)"
        );
        let (mut t_ik, mut t_a, mut t_r, mut t_b, mut drafted) = (0, 0, 0, 0, 0);
        for (i, (b, ik, a, r)) in accept.iter().enumerate() {
            println!(
                "{NAME}: accept | {b} | {ik} | {a} | {r} | {}",
                body_accept[i]
            );
            t_ik += ik;
            t_a += a;
            t_r += r;
            t_b += body_accept[i];
            drafted += set.drafts.iter().filter(|d| d.0 == *b).count();
        }
        println!("{NAME}: accept | total of {drafted} drafted | {t_ik} | {t_a} | {t_r} | {t_b}");

        if ok {
            println!(
                "PASSED: {NAME} — the block pass and head against the dsref set: every tap of blocks 0-2 within its band under ik's rule, \
                 the rows on ik's path under the reference rule, draft_argmax on every block, launches = the kernel list, w=5 bit-identical where it must be; \
                 the captured passes and appends = their pinned node counts, graph = eager per block and width, blocks 0-2 through the graph = the judged eager passes, \
                 DraftBody through the graphs = its eager twin"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// Whether every node of `g` is a kernel launch.
    fn all_kernels(g: &Graph) -> Result<bool, GateError> {
        Ok(g.nodes()?
            .iter()
            .all(|n| n.kind == cuda_core::sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL))
    }

    /// Every tap a whole pass leaves readable after its Markov loop, as
    /// bits, each cut to the rows of the pass.
    struct After {
        taps: Vec<(String, Vec<u32>)>,
        tokens: Vec<u32>,
    }

    /// The first `m` of [`MAX_WIDTH`] token-major rows of `v`, as bits.
    fn cut(v: &[f32], m: usize) -> Vec<u32> {
        v[..v.len() / MAX_WIDTH * m]
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    fn layer_cut(l: usize, t: &LayerTaps, m: usize) -> Vec<(String, Vec<u32>)> {
        let rows = [
            ("mix_a", &t.mix_a),
            ("hc_a", &t.hc_a),
            ("normed", &t.normed),
            ("kv", &t.kv),
            ("kv_row", &t.kv_row),
            ("q_a", &t.q_a),
            ("q_a_n", &t.q_a_n),
            ("q", &t.q),
            ("y", &t.y),
            ("wo_a", &t.wo_a),
            ("out_a", &t.out_a),
            ("streams_a", &t.streams_a),
            ("fold_a", &t.fold_a),
            ("mix_f", &t.mix_f),
            ("hc_f", &t.hc_f),
            ("normed_f", &t.normed_f),
            ("router logits", &t.logits),
            ("router probs", &t.probs),
            ("router weights", &t.weights),
            ("moe", &t.moe),
            ("sh_h", &t.sh_h),
            ("sh_y", &t.sh_y),
            ("ffn_out", &t.ffn_out),
            ("streams_f", &t.streams_f),
            ("fold_f", &t.fold_f),
        ];
        let mut out: Vec<(String, Vec<u32>)> = rows
            .iter()
            .map(|(name, v)| (format!("L{l} {name}"), cut(v, m)))
            .collect();
        out.push((
            format!("L{l} router ids"),
            t.ids[..t.ids.len() / MAX_WIDTH * m].to_vec(),
        ));
        // Slot-major over `3m` slots, then token-major: the first `3m · m`
        // rows of `3 · MAX_WIDTH · MAX_WIDTH`.
        out.push((
            format!("L{l} h"),
            t.h[..t.h.len() / (MAX_WIDTH * MAX_WIDTH) * m * m]
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        ));
        out
    }

    fn after_parts(
        streams0: &[f32],
        layers: &[LayerTaps],
        head_normed: &[f32],
        result: &[Vec<f32>],
        tokens: &[u32],
        m: usize,
    ) -> After {
        let mut taps = vec![("streams0".to_string(), cut(streams0, m))];
        for (l, t) in layers.iter().enumerate() {
            taps.extend(layer_cut(l, t, m));
        }
        taps.push((
            "head norm".to_string(),
            head_normed.iter().map(|x| x.to_bits()).collect(),
        ));
        taps.push(("result_output".to_string(), bits(result).concat()));
        After {
            taps,
            tokens: tokens.to_vec(),
        }
    }

    /// [`After`] of an eager [`run_pass`]; its head norm and logits are
    /// already `m` rows.
    fn after_of(o: &Ours, n_layer: usize, m: usize) -> After {
        after_parts(
            &o.streams0,
            &o.layers[..n_layer],
            &o.head_normed,
            &o.result,
            &o.tokens,
            m,
        )
    }

    /// [`After`] of the pass `pass` last ran, `m` rows.
    fn read_after(
        gpu: &Gpu,
        pass: &BlockPass,
        n_layer: usize,
        m: usize,
    ) -> Result<After, GateError> {
        let s = gpu.stream();
        let layers = (0..n_layer)
            .map(|l| read_layer(gpu, pass.layer(l).ok_or("a layer")?))
            .collect::<Result<Vec<_>, _>>()?;
        let n_vocab = pass.head().n_vocab();
        Ok(after_parts(
            &pass.streams0().to_host_vec(s)?,
            &layers,
            &pass.head().normed_to_host(s, m)?,
            &rows_of(&pass.head().logits_to_host(s, m)?, m, n_vocab),
            &pass.tokens(s)?,
            m,
        ))
    }

    /// The first tap (or the ids) where `a` and `b` differ; `None` when
    /// every one is bit-identical.
    fn first_difference(a: &After, b: &After) -> Option<String> {
        if a.taps.len() != b.taps.len() {
            return Some(format!("{} taps against {}", a.taps.len(), b.taps.len()));
        }
        a.taps
            .iter()
            .zip(&b.taps)
            .find(|(x, y)| x.0 != y.0 || x.1 != y.1)
            .map(|(x, _)| x.0.clone())
            .or_else(|| (a.tokens != b.tokens).then(|| "the ids".to_string()))
    }

    /// Every ring, f16 bits.
    fn ring_bits(
        gpu: &Gpu,
        rings: &DraftRings,
        n_layer: usize,
    ) -> Result<Vec<Vec<u16>>, GateError> {
        (0..n_layer)
            .map(|l| {
                Ok(rings
                    .ring(l)
                    .ok_or("a ring")?
                    .buf()
                    .to_host_vec(gpu.stream())?)
            })
            .collect()
    }

    /// Walk a block's taps for their prediction only, printing nothing:
    /// the band the argmax check of a non-tap block needs.
    fn judge_quiet(cx: &Cx<'_>, j: &mut Judge, bs: &BlockSet) -> Result<(), GateError> {
        let (set, hp, b) = (cx.set, cx.hp, bs.b);
        let n = hp.n_embd;
        let blk = |l: usize, s: &str| format!("blk.{l}.{s}.weight");
        for l in 0..hp.n_layer {
            let an = set.f32s(set.find_op(b, "FUSED_RMS_NORM", "", &blk(l, "attn_norm"))?)?;
            j.site(q8_site(&an, n, 32));
            let qan =
                set.f32s(set.find_op(b, "FUSED_RMS_NORM", "", &blk(l, "attn_q_a_norm"))?)?;
            j.site(q8_site(&qan, hp.q_lora_rank, 32));
            j.site(ATTN_SITE * ATTN_SITE);
            let hl = hp.n_head * hp.head_dim;
            let y = set.f32s(set.find(
                b,
                "block",
                &format!("dsv4_dflash_attn-{l} (reshaped) (view)"),
                0,
            )?)?;
            j.site(q8_site(&y, hl / hp.o_groups, 32));
            let wo_a = set.f32s(set.nth_op(b, "CONT", l)?)?;
            j.site(q8_site(&wo_a, hp.o_groups * hp.o_lora_rank, 32));
            j.site(BF16_SITE * BF16_SITE);
            let fnrm = set.f32s(set.find_op(b, "FUSED_RMS_NORM", "", &blk(l, "ffn_norm"))?)?;
            j.site(q8_site(&fnrm, n, 32) + q8_site(&fnrm, n, 128));
            let ff = hp.experts.ff;
            let gp = set.f32s(set.find(b, "block", &format!("ffn_moe_gate_par-{l}"), 0)?)?;
            j.site(q8_site(&gp, ff, 32) + q8_site(&gp, ff, 128));
            let shh = set.f32s(set.find(b, "block", &format!("ffn_up_gate-{l}"), 0)?)?;
            j.site(q8_site(&shh, ff * hp.experts.n_shared, 32));
        }
        let hn = set.f32s(set.find_op(b, "FUSED_RMS_NORM", "", "output_norm.weight")?)?;
        j.site(q8_site(&hn, n, 32) + q8_site(&hn, n, 128));
        let base = set.f32s(set.find(b, "block", "dflash_base_result_output", 0)?)?;
        let result = set.f32s(set.find(b, "block", "result_output", 0)?)?;
        j.site(markov_site(&base, &result, cx.n_vocab));
        Ok(())
    }

    /// ik's bf16 Markov delta as noise on its result rows: the worst row's
    /// `Σ (delta · 2^-9)² / 3 / Σ result²`.
    fn markov_site(base: &[f32], result: &[f32], n_vocab: usize) -> f64 {
        result
            .chunks(n_vocab)
            .zip(base.chunks(n_vocab))
            .map(|(r, b)| {
                let s2: f64 = r.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                let n2: f64 = r
                    .iter()
                    .zip(b)
                    .map(|(r, b)| {
                        let d = f64::from(*r) - f64::from(*b);
                        d * d
                    })
                    .sum::<f64>()
                    * BF16_SITE
                    * BF16_SITE;
                if s2 > 0.0 { n2 / s2 } else { 0.0 }
            })
            .fold(0.0, f64::max)
    }
}
