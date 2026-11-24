//! GPU gate for the DSpark draft's block pass and head
//! (`bloomery_gpu_deepseek41::draft::block`, the owner's head chain
//! `bloomery_gpu::dspark::head::Head`, and `DraftBody`: the owner
//! `bloomery_gpu::dspark::DsparkBody` over V4.1's layers), on one card, over
//! seeded inputs. Every clause is the engine against itself: its launch
//! counts, its reruns, its graphs against its eager twin. No tap is judged
//! against another engine's.
//!
//! PIN(2026-11-18): the ik taps of every row and rule, `draft_argmax` and the router ids against ik's, the set's mask decode, the accept counts and the shared expert's rule diag retired with the dsref set `code64_n32_w3`, which was dumped from the Q3_K_M file; ik cannot read bq3km.
//!
//! Inputs. `N_BLOCKS` seeded blocks (`seeded_block`), shaped as a code
//! prompt's: block 0 sees `PROMPT_ROWS` committed ring rows, each later
//! block `BLOCK_WIDTH` more, and a block's rope positions run on from its
//! ring rows. Per block: the id before it (never the mask token, below the
//! head's vocabulary), each layer's ring rows in f16 and the feature rows
//! `DraftBody` appends before it (`PROMPT_ROWS` of them for block 0, one
//! block's width after), all from `hc_host::lcg` under one seed per block and
//! field.
//!
//! The gate's own pass runs the layers into rows of its own and the owner's
//! head over them, so the head's rows are read between its logits and its
//! Markov loop. Where a block runs through `DraftBody` (the graph clauses
//! below), the seeded rings are seated in the layers' rings and the block's
//! rope positions and visible rows are the layers' override (`set_oracle`),
//! so the body's pass and the gate's own read the same keys. The body
//! refuses a block with no committed position, so one context row is
//! appended first; the seat then rewrites every ring slot.
//!
//! The pass runs under either rule (`Rule`), which differ only in data: the
//! block rows a row sees (the attention's `vis[2t + 1]`) and the SwiGLU
//! limits. `Rule::Reference` (`model.py`) is the engine's. `Rule::Ik` is ik's
//! draft rule, a block-causal mask and unclamped SwiGLUs, routed and shared;
//! only this gate runs it, eager.
//!
//! Wide block. `w = 5` over block 0's inputs under each rule: five ids in the
//! head's vocabulary, a rerun bit-identical, and under `Rule::Ik` rows 0–2
//! bit-identical to the `w = 3` pass (no kernel reads a row it is not given;
//! the first rows see the same keys). Under the reference rule row 0 sees
//! every block row, so it changes with `w` (printed).
//!
//! Graphs. The launches a pass of each width `1..=5` makes, counted in the
//! gate's own capture (the layers, then the owner head's norm, projection
//! and Markov loop) and in the graph `DraftBody` captured at load, equal the
//! layers' launches plus the owner head's and the pinned `73 + 8w`; each
//! append graph (`1..=GROUP` rows) holds its eight launches, the layers' plus
//! the owner's fc and norm; every node is a kernel. Then, on `DraftBody`'s
//! own layers and rings under the reference rule: for blocks 0–2 and every
//! width, the replay and then the eager twin leave every tap bit-identical
//! (the replay first, so one that did not re-read its inputs would read the
//! previous call's); blocks 0–2 in a row through the `w = 3` graph equal the
//! eager passes of the gate's own pass under the reference rule; and
//! `DraftBody` over every seeded block through its graphs gives the ids and
//! the rings of its eager twin after every block. The last layer's fold is
//! the rows the layers leave in the owner's `x`: the gate's own pass reads
//! its own, the body's is read through `block_rows_to_host`.
//!
//! Replay refusal. The graphs are captured under the reference rule, whose
//! clamp limits are launch scalars; ik's rule runs eager. Under `Rule::Ik` a
//! block asked for through its graph and an append asked for through its
//! graph are each refused by name (`GpuError::State`, "the reference rule"),
//! nothing is replayed or appended, and the eager block of the same call
//! runs.
//!
//! Embedding rows. For the mask token, ids 0 and `n_vocab − 1` and two
//! seeded ids, row 0 of layer 0's four streams and of its fold after a pass
//! — the target's `token_embd` row of that id, decoded on the card by the
//! target's own broadcast entry — equals bit for bit `gguf::quant::dequant_row`
//! of the file's row and, for a Q3_K file, `elem::q3k_embed_value` on the host
//! at the glue kernel's call shape (super-block at byte `110·(i / 256)`, value
//! `i % 256`); the mask rows `1..` (decoded at load by the same entry) equal
//! the mask token's. This is the draft reading the same f32 the target reads,
//! whichever format the target file stores.
//!
//! Fault readback. A row of all-`0xFF` bytes (a bf16 NaN; a Q3_K super-block
//! whose f16 `d` is NaN) staged into `DraftBody`'s layers (`pass_mut`), run
//! eager over the owner's head and through the body's `w = 3` graph: the NaN
//! reaches the routed experts' q8_1 quantizer, which raises the card's fault
//! word, and the head's own readback must be `GpuError::Fault` rather than
//! ids. The word is cleared after each arm. Run last: the NaN stays in the
//! body's buffers.
//!
//! FAIL-first: a graph replay that does not restage its inputs
//! (`DsparkBody::propose_as_with` staging for the eager arm only) turns the
//! graph rows red; dropping the `replayable` check before a graph replay in
//! `propose_as_with`, or in `append_as_with`, turns the refusal row red.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_graph", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::PathBuf;
    use std::sync::Arc;

    use bloomery_gpu::dspark::{DraftLayers, Submit};
    use bloomery_gpu::elem::q3k_embed_value;
    use bloomery_gpu::hc::HC_STREAMS;
    use bloomery_gpu::{Gpu, GpuError, Graph};
    use bloomery_gpu_deepseek41::draft::block::{
        BlockInput, BlockPass, LayerBufs, MAX_WIDTH, Rule, embedding_row,
    };
    use bloomery_gpu_deepseek41::draft::kv::{DraftRings, GROUP};
    use bloomery_gpu_deepseek41::draft::layers::OracleBlock;
    use bloomery_gpu_deepseek41::draft::load::DraftWeights;
    use bloomery_gpu_deepseek41::draft::{DraftBody, open as open_draft};
    use bloomery_gpu_gates::hc_host::lcg;
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits};
    use model::arch::deepseek41::names as target_names;
    use model::arch::dspark::DraftHparams;
    use refset::arch::deepseek41::dspark_model;

    const NAME: &str = "gate_dspark_graph";
    /// Blocks the graph clauses compare tap by tap.
    const TAP_BLOCKS: [usize; 3] = [0, 1, 2];
    /// Seeded blocks the whole-body clause walks.
    const N_BLOCKS: usize = 12;
    /// Committed ring rows block 0 sees: a code prompt's positions.
    const PROMPT_ROWS: usize = 64;
    /// The seeded blocks' width: the draft's block size, and the rows each
    /// later block adds to the ring.
    const BLOCK_WIDTH: usize = 3;
    /// The synthetic wide block's width.
    const WIDE: usize = 5;
    /// Kernel nodes of the captured block pass of `w` rows, `w = 1..=5`:
    /// the widening, three layers of `23 + 2w`, the head's three and the
    /// Markov loop's `2w` — `73 + 8w`.
    const PASS_NODES: [usize; MAX_WIDTH] = [81, 89, 97, 105, 113];
    /// Kernel nodes of the captured append of `n <= GROUP` rows: `fc`, its
    /// norm, and per layer `attn_kv` and the append.
    const APPEND_NODES: usize = 8;

    // ------------------------------------------------------ one block's inputs

    struct BlockSet {
        b: usize,
        /// The block's width: [`BLOCK_WIDTH`].
        width: usize,
        id_last: u32,
        /// The rope positions of the block's rows, consecutive from the
        /// committed position.
        pos: Vec<u32>,
        ring_rows: usize,
        /// Per layer, the committed ring rows `0..ring_rows`, f16 bits.
        rings: Vec<Vec<u16>>,
        /// The feature rows the `DraftBody` replay appends before this block:
        /// [`PROMPT_ROWS`] for block 0, [`BLOCK_WIDTH`] after.
        feats: Vec<f32>,
    }

    /// Seed fields of one block: its id, its feature rows, and from
    /// [`SEED_RING`] its layers' rings.
    const SEED_ID: usize = 0;
    const SEED_FEATS: usize = 1;
    const SEED_RING: usize = 2;

    /// The seed of one field of one block, scrambled so neighbouring pairs do
    /// not walk neighbouring streams of the LCG.
    fn seed(block: usize, field: usize) -> u32 {
        ((block * 16 + field) as u32 + 1).wrapping_mul(2_654_435_761)
    }

    /// An id in `0..n_vocab` from one LCG draw, never `avoid`.
    fn seeded_id(seed: u32, n_vocab: usize, avoid: u32) -> u32 {
        let unit = (lcg(1, seed)[0] + 2.0) / 4.0;
        let id = ((unit * n_vocab as f32) as u32).min(n_vocab as u32 - 1);
        if id == avoid {
            (id + 1) % n_vocab as u32
        } else {
            id
        }
    }

    /// Block `b`: [`PROMPT_ROWS`] committed ring rows and [`BLOCK_WIDTH`] more
    /// for each block before it, positions from the committed position, an id
    /// below `n_vocab` and the feature rows the body appends before it.
    fn seeded_block(hp: &DraftHparams, n_vocab: usize, b: usize) -> Result<BlockSet, GateError> {
        let ring_rows = PROMPT_ROWS + b * BLOCK_WIDTH;
        if ring_rows > hp.window {
            return Err(format!(
                "block {b}: {ring_rows} ring rows past the draft's window of {}",
                hp.window
            )
            .into());
        }
        let hd = hp.head_dim;
        let rings: Vec<Vec<u16>> = (0..hp.n_layer)
            .map(|l| {
                lcg(ring_rows * hd, seed(b, SEED_RING + l))
                    .into_iter()
                    .map(f32_to_f16_bits)
                    .collect()
            })
            .collect();
        let append_rows = if b == 0 { PROMPT_ROWS } else { BLOCK_WIDTH };
        let feats = lcg(
            append_rows * hp.target_layers.len() * hp.n_embd,
            seed(b, SEED_FEATS),
        );
        let pos = (0..BLOCK_WIDTH)
            .map(|t| u32::try_from(ring_rows + t))
            .collect::<Result<Vec<u32>, _>>()?;
        Ok(BlockSet {
            b,
            width: BLOCK_WIDTH,
            id_last: seeded_id(seed(b, SEED_ID), n_vocab, hp.mask_token),
            pos,
            ring_rows,
            rings,
            feats,
        })
    }

    /// Seat the block's committed rows in our rings, every other slot zero.
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

    /// The gate's own pass: a `BlockPass` and rings beside the body's, and the
    /// rows the layers leave for the head. The head is the body's — its taps
    /// are read between the logits and the Markov loop, which rewrites them.
    struct Standalone {
        pass: BlockPass,
        rings: DraftRings,
        /// The layers' output rows, [`MAX_WIDTH`] of them.
        x: DeviceBuffer<f32>,
        /// The id before the block: the Markov loop's first previous token.
        first: DeviceBuffer<u32>,
    }

    impl Standalone {
        fn new(gpu: &Gpu, body: &DraftBody, hp: &DraftHparams) -> Result<Standalone, GateError> {
            let s = gpu.stream();
            Ok(Standalone {
                pass: BlockPass::new(gpu, body.layers().weights())?,
                rings: DraftRings::new(s, hp)?,
                x: DeviceBuffer::zeroed(s, MAX_WIDTH * hp.n_embd)?,
                first: DeviceBuffer::from_host(s, &[0u32])?,
            })
        }

        /// The staged pass's layers into `x`, then the head's norm,
        /// quantizer and projection over them. Capturable.
        fn enqueue_body(
            &mut self,
            gpu: &Gpu,
            body: &mut DraftBody,
            m: usize,
        ) -> Result<(), GpuError> {
            self.pass
                .enqueue_layers(gpu, body.layers().weights(), &self.rings, m, &mut self.x)?;
            body.head_mut().enqueue_logits(gpu, &self.x, m)
        }

        /// The head's Markov loop over the logits [`Standalone::enqueue_body`]
        /// left, after `first`. Capturable.
        fn enqueue_markov(
            &self,
            gpu: &Gpu,
            body: &mut DraftBody,
            m: usize,
        ) -> Result<(), GpuError> {
            body.head_mut().enqueue_markov(gpu, m, &self.first)
        }

        /// One captured pass of `input.width` rows under the reference rule:
        /// the layers, the head's logits and the Markov loop, as the body's
        /// own graphs hold them. Leaves the pass staged as `input` says.
        fn capture(
            &mut self,
            gpu: &Gpu,
            body: &mut DraftBody,
            input: &BlockInput<'_>,
        ) -> Result<Graph, GateError> {
            let m = input.width;
            self.pass.stage(gpu.stream(), input)?;
            Ok(gpu.capture(|_| {
                self.enqueue_body(gpu, body, m)?;
                self.enqueue_markov(gpu, body, m)
            })?)
        }
    }

    /// Stage, run the layers and the logits, read every tap, run the Markov
    /// loop, read the rest. `id_last` is the id the block runs after.
    fn run_pass(
        gpu: &Gpu,
        body: &mut DraftBody,
        sa: &mut Standalone,
        input: &BlockInput<'_>,
        id_last: u32,
    ) -> Result<Ours, GateError> {
        let s = gpu.stream();
        let m = input.width;
        sa.first.copy_from_host(s, &[id_last])?;
        sa.pass.stage(s, input)?;
        sa.enqueue_body(gpu, body, m)?;
        s.synchronize()?;
        let n_layer = body.layers().weights().hp().n_layer;
        let mut layers = (0..n_layer)
            .map(|l| read_layer(gpu, sa.pass.layer(l).ok_or("a layer")?))
            .collect::<Result<Vec<_>, GateError>>()?;
        // The last layer's fold is the rows the head reads, in `x`.
        layers.last_mut().ok_or("a layer")?.fold_f = sa.x.to_host_vec(s)?;
        let head = body.head();
        let n_vocab = head.n_vocab();
        let base = rows_of(&head.logits(s, m)?, m, n_vocab);
        let head_normed = head.normed_to_host(s, m)?;
        let streams0 = sa.pass.streams0().to_host_vec(s)?;
        sa.enqueue_markov(gpu, body, m)?;
        s.synchronize()?;
        let head = body.head();
        let result = rows_of(&head.logits(s, m)?, m, n_vocab);
        Ok(Ours {
            streams0,
            layers,
            head_normed,
            base,
            result,
            tokens: head.tokens(s, m)?,
        })
    }

    // ------------------------------------------------------------ the distance

    /// Relative RMS distance of each row of `ours` from the row of `base`.
    fn rho_rows(ours: &[Vec<f32>], base: &[Vec<f32>]) -> Vec<f64> {
        ours.iter()
            .zip(base)
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

    /// The rule's name in the gate's lines.
    fn label(rule: Rule) -> &'static str {
        match rule {
            Rule::Ik => "ik-rule",
            Rule::Reference => "reference",
        }
    }

    fn input<'a>(bs: &BlockSet, row: &'a [u8], width: usize, rule: Rule) -> BlockInput<'a> {
        BlockInput {
            row,
            width,
            first_pos: bs.pos[0],
            ring_rows: bs.ring_rows,
            rule,
        }
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|r| r.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    // ---------------------------------------------------- embedding rows

    /// Ids the embedding pin reads: the mask token, the first and last ids,
    /// and two seeded ones.
    fn embed_ids(hp: &DraftHparams, n_vocab: usize) -> Result<Vec<u32>, GateError> {
        let last = u32::try_from(n_vocab - 1)?;
        let seeded = |i: usize| seeded_id(seed(N_BLOCKS + i, SEED_ID), n_vocab, hp.mask_token);
        Ok(vec![hp.mask_token, 0, last, seeded(0), seeded(1)])
    }

    /// Values of `a` whose bits differ from `b`'s; a length mismatch counts
    /// every value.
    fn differ(a: &[f32], b: &[f32]) -> usize {
        if a.len() != b.len() {
            return a.len().max(b.len());
        }
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    }

    /// The target's embedding row `row` (file bytes of type `ty`, `n`
    /// values) by the host rules: `dequant_row`, and for Q3_K
    /// `q3k_embed_value` over the row's words at the glue kernel's call
    /// shape.
    fn host_row(
        ty: GgmlType,
        row: &[u8],
        n: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), GateError> {
        let mut deq = vec![0.0f32; n];
        dequant_row(ty, row, &mut deq)?;
        if ty != GgmlType::Q3_K {
            return Ok((deq, None));
        }
        let mut words = vec![0u32; row.len().div_ceil(4)];
        for (d, c) in words.iter_mut().zip(row.chunks(4)) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            *d = u32::from_le_bytes(b);
        }
        if !n.is_multiple_of(256) || row.len() != n / 256 * 110 {
            return Err(format!("a Q3_K row of {} bytes for {n} values", row.len()).into());
        }
        let q3k = (0..n)
            // The super-block starts 0 or 2 mod 4, the two alignments the core
            // funnels.
            // SAFETY: i < n and row.len() == n / 256 · 110 (checked above), so
            // super-block `i / 256` ends at byte 110·(i/256 + 1) <= row.len()
            // <= 4·words.len(); i % 256 < 256.
            .map(|i| unsafe { q3k_embed_value(&words, 110 * (i / 256), i % 256) })
            .collect();
        Ok((deq, Some(q3k)))
    }

    /// The embedding pin (module doc): row 0 of layer 0's streams and fold
    /// for each of [`embed_ids`], and the mask rows, against the host rules.
    fn check_embed_rows(
        gpu: &Gpu,
        body: &DraftBody,
        sa: &mut Standalone,
        target: &Split,
        hp: &DraftHparams,
    ) -> Result<bool, GateError> {
        let s = gpu.stream();
        let n = hp.n_embd;
        let w = body.layers().weights();
        let name = target_names::token_embd();
        let ty = target
            .find(&name)
            .ok_or_else(|| format!("{name} is not in the target"))?
            .1
            .ty;
        println!(
            "{NAME}: embed | target {name} is {ty}, the draft reads it as {:?}, the mask row as {:?}",
            sa.pass.embd_rows(),
            w.embd_rows()
        );
        let mut ok = true;
        for id in embed_ids(hp, body.head().n_vocab())? {
            let row = embedding_row(target, n, id)?;
            let (deq, q3k) = host_row(ty, row, n)?;
            sa.pass.stage(
                s,
                &BlockInput {
                    row,
                    width: 1,
                    first_pos: 0,
                    ring_rows: 1,
                    rule: Rule::Reference,
                },
            )?;
            sa.pass.enqueue_layers(gpu, w, &sa.rings, 1, &mut sa.x)?;
            s.synchronize()?;
            let streams = sa.pass.streams0().to_host_vec(s)?;
            let fold = sa.pass.fold0().to_host_vec(s)?;
            let row0: Vec<&[f32]> = (0..HC_STREAMS)
                .map(|k| &streams[k * n..(k + 1) * n])
                .chain([&fold[..n]])
                .collect();
            let vs_deq: usize = row0.iter().map(|c| differ(c, &deq)).sum();
            let vs_q3k: Option<usize> = q3k
                .as_ref()
                .map(|q| row0.iter().map(|c| differ(c, q)).sum());
            let host_pair = q3k.as_ref().map(|q| differ(q, &deq));
            let id_ok = vs_deq == 0 && vs_q3k.unwrap_or(0) == 0 && host_pair.unwrap_or(0) == 0;
            let fmt = |v: Option<usize>| v.map_or("n/a (bf16)".to_string(), |d| d.to_string());
            println!(
                "{NAME}: embed | id {id}: row 0 ({HC_STREAMS} streams + fold, {} values) differing \
                 from dequant_row {vs_deq}, from host q3k_embed_value {}; host q3k_embed_value vs \
                 dequant_row {} {}",
                row0.len() * n,
                fmt(vs_q3k),
                fmt(host_pair),
                verdict(id_ok)
            );
            ok &= id_ok;
            if id == hp.mask_token {
                let mut d = 0usize;
                for r in 1..MAX_WIDTH {
                    for k in 0..HC_STREAMS {
                        let at = (r * HC_STREAMS + k) * n;
                        d += differ(&streams[at..at + n], &deq);
                    }
                    d += differ(&fold[r * n..(r + 1) * n], &deq);
                }
                println!(
                    "{NAME}: embed | mask rows 1..{MAX_WIDTH} (decoded at load) differing from the \
                     mask token's dequant_row: {d} of {} values {}",
                    (MAX_WIDTH - 1) * (HC_STREAMS + 1) * n,
                    verdict(d == 0)
                );
                ok &= d == 0;
            }
        }
        Ok(ok)
    }

    /// The fault readback (module doc): a NaN row through `body`'s layers,
    /// eager over its head and through its graph, must read back as
    /// `GpuError::Fault`.
    fn check_fault(
        gpu: &Gpu,
        body: &mut DraftBody,
        target: &Split,
        hp: &DraftHparams,
    ) -> Result<bool, GateError> {
        let s = gpu.stream();
        let width = BLOCK_WIDTH;
        let len = embedding_row(target, hp.n_embd, 0)?.len();
        let nan_row = vec![0xFFu8; len];
        let input = BlockInput {
            row: &nan_row,
            width,
            first_pos: 0,
            ring_rows: 1,
            rule: Rule::Reference,
        };
        if let Some(f) = gpu.fault()? {
            return Err(format!("the fault word is raised before the fault case: {f}").into());
        }
        let mut x = DeviceBuffer::<f32>::zeroed(s, MAX_WIDTH * hp.n_embd)?;
        let first = DeviceBuffer::from_host(s, &[0u32])?;
        let mut ok = true;
        for submit in [Submit::Eager, Submit::Graph] {
            // The row goes in through the layers' own pass: a proposal stages
            // its first id's row from the target's file.
            body.layers_mut().pass_mut().stage(s, &input)?;
            let got = match submit {
                Submit::Eager => {
                    body.layers_mut().enqueue_block(gpu, &mut x, width)?;
                    body.head_mut().enqueue_logits(gpu, &x, width)?;
                    body.head_mut().enqueue_markov(gpu, width, &first)?;
                    body.head().tokens(s, width)
                }
                Submit::Graph => {
                    body.graph_block(width)
                        .ok_or("a block graph per width")?
                        .launch(s)?;
                    body.head().tokens(s, width)
                }
            };
            let (case_ok, what) = match &got {
                Err(GpuError::Fault { fault, .. }) => (true, format!("GpuError::Fault at {fault}")),
                Err(e) => (false, format!("another error: {e}")),
                Ok(ids) => (false, format!("ids {ids:?} (a silent proposal)")),
            };
            println!(
                "{NAME}: fault | {submit:?} pass over an all-0xFF row ({len} bytes, w={width}): the \
                 head's readback is {what} {}",
                verdict(case_ok)
            );
            ok &= case_ok;
            gpu.clear_fault()?;
        }
        let clean = gpu.fault()?.is_none();
        println!(
            "{NAME}: fault | word clean after the case: {clean} {}",
            verdict(clean)
        );
        Ok(ok && clean)
    }

    pub fn run() -> Result<(), GateError> {
        let path = PathBuf::from(dspark_model()?);
        let draft = Split::open(&path)?;
        let target = Arc::new(Split::open(ref_model_path()?)?);
        let gpu = Gpu::new()?;
        println!("{NAME}: card {}", gpu.device_name()?);
        let weights = DraftWeights::load(gpu.stream(), &draft, &target)?;
        let hp = weights.hp().clone();
        println!(
            "{NAME}: draft {} layers, block_size {} in the file, widths {BLOCK_WIDTH} (the seeded blocks') and {WIDE} (synthetic) gated, {N_BLOCKS} seeded blocks",
            hp.n_layer, hp.block_size
        );
        let mut body = open_draft(&gpu, weights, Arc::clone(&target))?;
        let mut ok = true;
        let n_vocab = body.head().n_vocab();

        let mut sa = Standalone::new(&gpu, &body, &hp)?;
        ok &= check_embed_rows(&gpu, &body, &mut sa, &target, &hp)?;
        let blocks = (0..N_BLOCKS)
            .map(|b| seeded_block(&hp, n_vocab, b))
            .collect::<Result<Vec<_>, _>>()?;
        let b0 = &blocks[0];

        // Launches: the gate's own capture of one pass per width (the layers
        // and the owner head's), and the graphs DraftBody captured at load —
        // every node a kernel.
        let row0 = embedding_row(&target, hp.n_embd, b0.id_last)?;
        seat(&gpu, &hp, b0, &mut sa.rings)?;
        for m in 1..=MAX_WIDTH {
            let nodes = sa
                .capture(&gpu, &mut body, &input(b0, row0, m, Rule::Reference))?
                .node_count();
            let g = body.graph_block(m).ok_or("a block graph per width")?;
            let kernels = all_kernels(g)?;
            let own = sa.pass.launches(m) + body.head_launches(m);
            let owned = body.layers().launches(m) + body.head_launches(m);
            let pass_ok = nodes == PASS_NODES[m - 1]
                && own == PASS_NODES[m - 1]
                && g.node_count() == PASS_NODES[m - 1]
                && owned == PASS_NODES[m - 1]
                && kernels;
            println!(
                "{NAME}: launches w={m}: gate capture {nodes} nodes, DraftBody graph {} nodes (all kernel nodes {kernels}), \
                 BlockPass::launches + head {own}, the body's layers + head {owned}, pinned {} {}",
                g.node_count(),
                PASS_NODES[m - 1],
                verdict(pass_ok)
            );
            ok &= pass_ok;
        }
        for n in 1..=GROUP {
            let g = body
                .graph_append(n)
                .ok_or("an append graph per row count")?;
            let kernels = all_kernels(g)?;
            let launches = body.layers().append_launches(n) + body.fc_launches(n);
            let append_ok = g.node_count() == APPEND_NODES && launches == APPEND_NODES && kernels;
            println!(
                "{NAME}: launches append n={n}: DraftBody graph {} nodes (all kernel nodes {kernels}), \
                 the layers' append + fc {launches}, pinned {APPEND_NODES} {}",
                g.node_count(),
                verdict(append_ok)
            );
            ok &= append_ok;
        }

        // The gate's own eager passes of blocks 0-2 under the reference rule:
        // what the graph clauses below compare the body's replays with.
        let mut eager_ref = Vec::new();
        for b in TAP_BLOCKS {
            let bs = &blocks[b];
            let row = embedding_row(&target, hp.n_embd, bs.id_last)?;
            seat(&gpu, &hp, bs, &mut sa.rings)?;
            let o = run_pass(
                &gpu,
                &mut body,
                &mut sa,
                &input(bs, row, bs.width, Rule::Reference),
                bs.id_last,
            )?;
            println!("{NAME}: {} b{b} ids {:?}", label(Rule::Reference), o.tokens);
            eager_ref.push((b, after_of(&o, hp.n_layer, bs.width)));
        }

        // w = 5 over block 0's inputs.
        seat(&gpu, &hp, b0, &mut sa.rings)?;
        for rule in [Rule::Ik, Rule::Reference] {
            let narrow = run_pass(
                &gpu,
                &mut body,
                &mut sa,
                &input(b0, row0, BLOCK_WIDTH, rule),
                b0.id_last,
            )?;
            let wide = run_pass(
                &gpu,
                &mut body,
                &mut sa,
                &input(b0, row0, WIDE, rule),
                b0.id_last,
            )?;
            let again = run_pass(
                &gpu,
                &mut body,
                &mut sa,
                &input(b0, row0, WIDE, rule),
                b0.id_last,
            )?;
            let shape =
                wide.tokens.len() == WIDE && wide.tokens.iter().all(|&t| (t as usize) < n_vocab);
            let rerun = bits(&wide.result) == bits(&again.result) && wide.tokens == again.tokens;
            let first_rows = bits(&wide.result[..BLOCK_WIDTH]) == bits(&narrow.result)
                && bits(&wide.base[..BLOCK_WIDTH]) == bits(&narrow.base)
                && wide.tokens[..BLOCK_WIDTH] == narrow.tokens[..];
            let row0_argmax = wide.tokens[0] == narrow.tokens[0];
            let row0_rho = rho_rows(&wide.base[..1], &narrow.base[..1])[0];
            let pass_ok = shape
                && rerun
                && match rule {
                    Rule::Ik => first_rows,
                    Rule::Reference => true,
                };
            println!(
                "{NAME}: {} w={WIDE} on block 0: ids {:?} (w={BLOCK_WIDTH}: {:?}); shape {shape}; rerun bit-identical {rerun}; \
                 rows 0-{} bit-identical to w={BLOCK_WIDTH} {first_rows}; row 0 argmax equal {row0_argmax}, row 0 base rho {row0_rho:.2e} {}",
                label(rule),
                wide.tokens,
                narrow.tokens,
                BLOCK_WIDTH - 1,
                verdict(pass_ok)
            );
            ok &= pass_ok;
        }

        drop(sa);

        // Graph = eager on DraftBody's own layers and rings: blocks 0-2 at
        // every width, the replay first — a replay that did not re-read its
        // inputs would read the previous call's — then the eager twin, every
        // tap the whole pass leaves, bit for bit.
        for b in TAP_BLOCKS {
            let bs = &blocks[b];
            seat_body(&gpu, &mut body, &hp, bs)?;
            for m in 1..=MAX_WIDTH {
                let ids = propose(&gpu, &mut body, bs.id_last, m, Submit::Graph)?;
                let replay = read_after(&gpu, &body, m)?;
                propose(&gpu, &mut body, bs.id_last, m, Submit::Eager)?;
                let eager = read_after(&gpu, &body, m)?;
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

        // Three consecutive blocks through the w = 3 graph against the gate's
        // own eager passes above (the reference rule).
        for (b, eager) in &eager_ref {
            let bs = &blocks[*b];
            seat_body(&gpu, &mut body, &hp, bs)?;
            propose(&gpu, &mut body, bs.id_last, bs.width, Submit::Graph)?;
            let replay = read_after(&gpu, &body, bs.width)?;
            let off = first_difference(&replay, eager);
            println!(
                "{NAME}: graph b{b} w={} = the gate's own eager pass: ids {:?} (eager {:?}); {} taps; {} {}",
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

        // The graphs are the reference rule's: ik's rule is refused by name
        // through a graph, and runs eager.
        ok &= check_ik_refusal(&gpu, &mut body, &hp, b0)?;

        // DraftBody over every seeded block: its features appended, every
        // block's proposal — eager, then through the graphs, the ids and the
        // rings after each block bit for bit. Its own positions and rows
        // again, not the override's.
        body.layers_mut().set_oracle(None);
        let mut eager_run = Vec::new();
        body.reset(&gpu)?;
        for bs in &blocks {
            append_block(&gpu, &mut body, bs, Submit::Eager)?;
            let ids = propose(&gpu, &mut body, bs.id_last, bs.width, Submit::Eager)?;
            eager_run.push((ids, ring_bits(&gpu, body.layers().rings(), hp.n_layer)?));
        }
        body.reset(&gpu)?;
        for (bs, (e_ids, e_rings)) in blocks.iter().zip(&eager_run) {
            let n = append_block(&gpu, &mut body, bs, Submit::Graph)?;
            let ids = propose(&gpu, &mut body, bs.id_last, bs.width, Submit::Graph)?;
            let same =
                ids == *e_ids && ring_bits(&gpu, body.layers().rings(), hp.n_layer)? == *e_rings;
            println!(
                "{NAME}: DraftBody b{} appended {n} ({}) committed {} ids {ids:?}: ids and rings = the eager run {same} {}",
                bs.b,
                if n <= GROUP {
                    "graph"
                } else {
                    "eager: no graph"
                },
                body.committed(),
                verdict(same)
            );
            ok &= same;
        }

        // Last: the NaN passes leave NaN in the body's buffers past the rows
        // a later pass rewrites.
        ok &= check_fault(&gpu, &mut body, &target, &hp)?;

        if ok {
            println!(
                "PASSED: {NAME} — the block pass and head over seeded blocks: the captured passes and appends = their pinned node counts, \
                 w=5 in range, rerun-identical and, under ik's rule, rows 0-2 bit-identical to w=3, graph = eager per block and width, \
                 blocks 0-2 through the graph = the gate's own eager passes, DraftBody through the graphs = its eager twin, \
                 ik's rule through a graph refused by name; the draft's embedding rows = the host rules bit for bit, a NaN row reads back as GpuError::Fault"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// Append the features that precede block `bs` at the committed position,
    /// returning the rows appended. The body then holds exactly the ring rows
    /// the block's own pass is seeded with.
    fn append_block(
        gpu: &Gpu,
        body: &mut DraftBody,
        bs: &BlockSet,
        submit: Submit,
    ) -> Result<usize, GateError> {
        let first = body.committed();
        let n = body.append_as_with(gpu, first, &bs.feats, submit)?;
        if body.committed() as usize != bs.ring_rows {
            return Err(format!(
                "block {}: {} positions committed after its features, {} ring rows seeded",
                bs.b,
                body.committed(),
                bs.ring_rows
            )
            .into());
        }
        Ok(n)
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

    /// [`After`] of the pass `body` last ran, `m` rows: its layers' taps and
    /// head, and the last layer's fold from the owner's rows.
    fn read_after(gpu: &Gpu, body: &DraftBody, m: usize) -> Result<After, GateError> {
        let s = gpu.stream();
        let pass = body.layers().pass();
        let hp = body.layers().weights().hp();
        let mut layers = (0..hp.n_layer)
            .map(|l| read_layer(gpu, pass.layer(l).ok_or("a layer")?))
            .collect::<Result<Vec<_>, GateError>>()?;
        // The tap holds `MAX_WIDTH` rows, as the layer's own buffer does.
        let mut fold = body.block_rows_to_host(s, m)?;
        fold.resize(MAX_WIDTH * hp.n_embd, 0.0);
        layers.last_mut().ok_or("a layer")?.fold_f = fold;
        let head = body.head();
        let n_vocab = head.n_vocab();
        Ok(after_parts(
            &pass.streams0().to_host_vec(s)?,
            &layers,
            &head.normed_to_host(s, m)?,
            &rows_of(&head.logits(s, m)?, m, n_vocab),
            &head.tokens(s, m)?,
            m,
        ))
    }

    /// A block of `width` ids after `id_last` through the body, at its
    /// committed position.
    fn propose(
        gpu: &Gpu,
        body: &mut DraftBody,
        id_last: u32,
        width: usize,
        submit: Submit,
    ) -> Result<Vec<u32>, GpuError> {
        let mut ids = vec![0u32; width];
        let pos = body.committed();
        body.propose_as_with(gpu, pos, id_last, width, &mut ids, &mut [], submit)?;
        Ok(ids)
    }

    /// The first feature row of the block's features: `target_layers` rows of
    /// `n_embd`.
    fn first_feature_row<'a>(hp: &DraftHparams, bs: &'a BlockSet) -> Result<&'a [f32], GateError> {
        Ok(bs
            .feats
            .get(..hp.target_layers.len() * hp.n_embd)
            .ok_or("a feature row in the block's features")?)
    }

    /// Put the body at the seeded block `bs`: its committed rows in the
    /// layers' rings, and the block's own rope positions and visible rows in
    /// force. The body refuses a block with no committed position, so one
    /// position is seated first (`DsparkBody::seat_positions`, no launch);
    /// the seat then rewrites every ring slot.
    fn seat_body(
        gpu: &Gpu,
        body: &mut DraftBody,
        hp: &DraftHparams,
        bs: &BlockSet,
    ) -> Result<(), GateError> {
        body.reset(gpu)?;
        body.seat_positions(1)?;
        seat(gpu, hp, bs, body.layers_mut().rings_mut())?;
        body.layers_mut().set_oracle(Some(OracleBlock {
            first_pos: bs.pos[0],
            ring_rows: bs.ring_rows,
        }));
        Ok(())
    }

    /// The replay refusal (module doc): under ik's rule a block and an
    /// append asked for through their graphs are each refused by name, the
    /// refused append commits nothing, and the eager block runs.
    fn check_ik_refusal(
        gpu: &Gpu,
        body: &mut DraftBody,
        hp: &DraftHparams,
        bs: &BlockSet,
    ) -> Result<bool, GateError> {
        seat_body(gpu, body, hp, bs)?;
        body.layers_mut().set_rule(Rule::Ik);
        let committed = body.committed();
        let row = first_feature_row(hp, bs)?;
        let block = propose(gpu, body, bs.id_last, bs.width, Submit::Graph);
        let append = body.append_as_with(gpu, committed, row, Submit::Graph);
        let eager = propose(gpu, body, bs.id_last, bs.width, Submit::Eager);
        body.layers_mut().set_rule(Rule::Reference);
        fn named<T>(r: &Result<T, GpuError>) -> bool {
            matches!(r, Err(GpuError::State { missing, .. }) if missing.contains("the reference rule"))
        }
        fn show<T>(r: &Result<T, GpuError>) -> String {
            match r {
                Err(e) => format!("refused: {e}"),
                Ok(_) => "replayed".to_string(),
            }
        }
        let (block_ok, append_ok) = (named(&block), named(&append));
        let kept = body.committed() == committed;
        let eager_ok = eager.as_ref().is_ok_and(|ids| ids.len() == bs.width);
        println!(
            "{NAME}: ik-rule replay | block through the graph {} {}; append through the graph {} {}; \
             committed kept at {committed}: {kept} {}; eager block {} {}",
            show(&block),
            verdict(block_ok),
            show(&append),
            verdict(append_ok),
            verdict(kept),
            if eager_ok { "ran" } else { "did not run" },
            verdict(eager_ok)
        );
        Ok(block_ok && append_ok && kept && eager_ok)
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
}
