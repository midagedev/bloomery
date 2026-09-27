//! GPU gate for the two-level candidate selection (`bloomery_gpu::cand`) —
//! the V4.1 candidate mask's kernels — bit for bit against this binary's
//! host transcription of the reference's `select_candidate_blocks`
//! (model.py:583-610) and of the family's own rules. Synthetic scores, no
//! model file. Shapes: V4.1's 2,048 blocks of 8 over 32,768 rows; the d1c
//! geometry, 16 blocks of 8 over 512 rows (top_k 64, and 200 so that the
//! candidates number fewer than top_k); 64 blocks of 1; 8 blocks of 32.
//! Every output is poisoned before a launch (a NaN payload the kernels never
//! make, `u32::MAX` list entries), and every score a launch must not read is
//! that NaN, so a read of it raises the site.
//!
//! 1. keys: a selecting row's block keys are the fold of its visible rows
//!    (the first largest in row order), bit for bit; nothing else of the
//!    buffer is written.
//! 2. kept: a selecting row's kept list is the reference's selection over the
//!    card's own keys — the block of row `n − 1` pinned at `+inf`, the best by
//!    (key descending, block ascending), `−inf` picks dropped — in ascending
//!    order, `blocks` entries, the rest of the row untouched; ties planted
//!    across the last kept place go to the lower blocks, `−0` and `+0` tie,
//!    and a row where only block 0 falls out is exact. Rows at or below the
//!    bound (`n ≤ blocks·block`) write nothing.
//! 3. compact, fed the host rule's kept lists: slot `s < n_c` holds the score
//!    of row `kept[s / block]·block + s % block`, the rows from `n_c` on are
//!    as they were, the histogram row holds the moved scores' top-ten-bit
//!    counts when the top-k will read them (`n_c > min(top_k, stride)`) and
//!    zero otherwise, and the counts view is `n_c` per row then `top_k`; a
//!    row that does not select keeps its scores and histogram and gets `n`.
//! 4. remap, fed the host kept lists and the list the row top-k writes over
//!    the compacted slots (exact top-k by key, ties to the lower slot,
//!    ascending): the list becomes the ascending top-`min(k, n_c)` rows of the
//!    masked scores by (key descending, row ascending) — the reference's
//!    consumer list whenever top_k is at most the candidate rows, as the
//!    loader requires. The slot-to-row map is strictly increasing, so ties
//!    broken to the lower slot are ties broken to the lower row: this is the
//!    composition with `ds41_indexer_topk`, whose exactness over its own
//!    scores and histogram `gate_deepseek41_index` pins.
//! 5. pipeline: the source launches, then compact on the card's own kept
//!    lists, the host top-k, then remap, equal the clauses' expectations.
//! 6. fault: a NaN, `+inf` or `−inf` source score, a count past the rows, a
//!    kept list out of order or not ending at its pin, a NaN consumer score
//!    and a list entry past `n_c` each raise [`FaultSite::CandMask`] at the
//!    launch's layer; a refused compaction writes `n_c = 0` and leaves the
//!    scores; a refused entry becomes `u32::MAX`. Clean launches raise
//!    nothing.
//! 7. replay: one graph of the three enqueues, replayed over rewritten
//!    counts (selecting, at the bound, below it), bit for bit an eager run.
//! 8. refuse: a zero block count, a block size that is not a power of two up
//!    to 32, a kept stride below the block count, a scratch of another shape
//!    and a source whose score pass would leave a selecting row unwritten
//!    (its `k` 0 or past `blocks·block`) are refused by name. The two
//!    consumer-only cases (top_k 0 and 200 at 16 blocks of 8) run clauses 3
//!    and 4 alone for that reason.
//! 9. shape: the four entries compile with no local depot.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_cand: built without the `gpu` feature; see `just gate-gpu-cand`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_cand", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::cand::{
        CandKernels, CandScratch, CandShape, CompactArgs, HIST_BINS, RemapArgs, SelectArgs,
    };
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError};
    use bloomery_gpu_gates::{GateError, no_local_depot, verdict};
    use cuda_core::DeviceBuffer;

    /// A NaN payload no kernel makes: a score that must not be read, and an
    /// output that must not be written.
    const POISON: u32 = 0x7fa5_a5a5;
    /// What the lists and kept lists are filled with.
    const LIST_POISON: u32 = u32::MAX;
    /// Kept-list and list entries past the written ones in every row.
    const SLACK: usize = 16;
    /// Where the words start in the gate's buffers: not at zero, so an
    /// offset a kernel drops shows.
    const TOP_K_AT: usize = 1;
    const N_AT: usize = 3;
    const COUNTS_AT: usize = 2;
    /// The layers the launches raise with: V4.1's source and first consumer.
    const SOURCE_LAYER: usize = 20;
    const CONSUMER_LAYER: usize = 24;

    fn poison() -> f32 {
        f32::from_bits(POISON)
    }

    /// A fixed xorshift64 stream.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        /// A value in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    // ------------------------------------------------------------ the rules
    //
    // Transcribed here, not taken from the kernel crate: a rule the gate
    // shared with the kernels would move with them.

    /// The order a score ranks in: an unsigned key that orders as the value
    /// does, `−0` and `+0` equal.
    fn key_of(v: f32) -> u32 {
        let b = if v == 0.0 { 0 } else { v.to_bits() };
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }

    /// Whether a row of `n` visible rows has more than `blocks` blocks of
    /// `block` rows.
    fn selecting(n: usize, blocks: usize, block: usize) -> bool {
        n.div_ceil(block) > blocks
    }

    /// The candidate rows of a selecting row: `blocks − 1` whole blocks and
    /// the newest block's visible rows.
    fn n_candidates(n: usize, blocks: usize, block: usize) -> usize {
        (blocks - 1) * block + n - (n.div_ceil(block) - 1) * block
    }

    /// A block's key as the kernel folds it: the first largest of its rows,
    /// by `>`, in row order.
    fn fold(rows: &[f32]) -> f32 {
        let mut m = rows[0];
        for &v in &rows[1..] {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// The reference's `select_candidate_blocks` for one row of `n` visible
    /// scores whose blocks' keys are `keys` (`⌈n / block⌉` of them): the last
    /// block set to `+inf`, the `min(blocks, nb)` best by (key descending,
    /// block ascending) — the order key makes `−0` and `+0` equal — the picks
    /// at `−inf` dropped, returned in ascending order.
    fn reference_kept(keys: &[f32], n: usize, blocks: usize, block: usize) -> Vec<u32> {
        let nb = keys.len();
        let mut k: Vec<f32> = keys.to_vec();
        k[(n - 1) / block] = f32::INFINITY;
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| key_of(k[b]).cmp(&key_of(k[a])).then(a.cmp(&b)));
        let mut kept: Vec<u32> = order[..blocks.min(nb)]
            .iter()
            .filter(|&&j| k[j] > f32::NEG_INFINITY)
            .map(|&j| j as u32)
            .collect();
        kept.sort_unstable();
        kept
    }

    /// Row of slot `s` under the kept list.
    fn slot_row(kept: &[u32], block: usize, s: usize) -> usize {
        kept[s / block] as usize * block + s % block
    }

    /// The top-ten-bit counts of `v`.
    fn counts_of(v: &[f32]) -> Vec<u32> {
        let mut h = vec![0u32; HIST_BINS];
        for &x in v {
            h[(key_of(x) >> 22) as usize] += 1;
        }
        h
    }

    /// The `k` best of `v` by (key descending, index ascending), ascending.
    fn top_ascending(v: &[f32], k: usize) -> Vec<u32> {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&a, &b| key_of(v[b]).cmp(&key_of(v[a])).then(a.cmp(&b)));
        let mut top: Vec<u32> = order[..k.min(v.len())].iter().map(|&i| i as u32).collect();
        top.sort_unstable();
        top
    }

    // ------------------------------------------------------------ the cases

    /// What a selecting row's keys carry besides distinct values.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Plant {
        /// Distinct keys.
        Plain,
        /// Six equal keys at ranks `blocks − 4 .. blocks + 2` of the ranked
        /// blocks: three kept, three not.
        Ties,
        /// The same group at `+0` and `−0` alternately, every key above it
        /// positive and every key below it negative.
        Zeros,
        /// Block 0 the lowest key: with `nb = blocks + 1`, the one block that
        /// falls out, so every slot moves.
        DropFirst,
    }

    struct Case {
        name: &'static str,
        blocks: usize,
        block: usize,
        rows: usize,
        top_k: usize,
        /// Per row: the visible count and the plant.
        rows_n: Vec<(usize, Plant)>,
        seed: u64,
    }

    impl Case {
        fn tokens(&self) -> usize {
            self.rows_n.len()
        }
        fn stride(&self) -> usize {
            self.top_k + SLACK
        }
        fn kstride(&self) -> usize {
            self.blocks + SLACK
        }
        fn shape(&self) -> Result<CandShape, GpuError> {
            CandShape::new(self.blocks, self.block)
        }
        fn selecting(&self, n: usize) -> bool {
            selecting(n, self.blocks, self.block)
        }
        fn k(&self) -> usize {
            self.top_k.min(self.stride())
        }
        /// Whether the source side can run: its score pass writes every
        /// selecting row's scores (`0 < k <= blocks·block`).
        fn source(&self) -> bool {
            self.k() > 0 && self.k() <= self.blocks * self.block
        }
    }

    fn cases() -> Vec<Case> {
        vec![
            Case {
                name: "v41",
                blocks: 2048,
                block: 8,
                rows: 32_768,
                top_k: 512,
                rows_n: vec![
                    (16_384, Plant::Plain),
                    (16_385, Plant::DropFirst),
                    (16_392, Plant::Plain),
                    (24_576, Plant::Ties),
                    (32_767, Plant::Zeros),
                    (32_768, Plant::Ties),
                ],
                seed: 0x0cad_0001,
            },
            Case {
                name: "v41-straddle",
                blocks: 2048,
                block: 8,
                rows: 32_768,
                top_k: 512,
                rows_n: (16_380..16_388).map(|n| (n, Plant::Plain)).collect(),
                seed: 0x0cad_0002,
            },
            Case {
                name: "d1c",
                blocks: 16,
                block: 8,
                rows: 512,
                top_k: 64,
                rows_n: vec![
                    (128, Plant::Plain),
                    (129, Plant::DropFirst),
                    (301, Plant::Ties),
                    (512, Plant::Zeros),
                ],
                seed: 0x0cad_0003,
            },
            Case {
                name: "d1c-k200",
                blocks: 16,
                block: 8,
                rows: 512,
                top_k: 200,
                rows_n: vec![(129, Plant::Plain), (301, Plant::Ties), (512, Plant::Plain)],
                seed: 0x0cad_0004,
            },
            Case {
                name: "d1c-k0",
                blocks: 16,
                block: 8,
                rows: 512,
                top_k: 0,
                rows_n: vec![(128, Plant::Plain), (301, Plant::Plain)],
                seed: 0x0cad_0007,
            },
            Case {
                name: "b1",
                blocks: 64,
                block: 1,
                rows: 256,
                top_k: 16,
                rows_n: vec![
                    (64, Plant::Plain),
                    (65, Plant::DropFirst),
                    (200, Plant::Ties),
                ],
                seed: 0x0cad_0005,
            },
            Case {
                name: "b32",
                blocks: 8,
                block: 32,
                rows: 1024,
                top_k: 64,
                rows_n: vec![
                    (256, Plant::Plain),
                    (257, Plant::Zeros),
                    (1000, Plant::Ties),
                ],
                seed: 0x0cad_0006,
            },
        ]
    }

    /// One case's host inputs and expectations.
    struct Host {
        /// `[tokens × rows]` source and consumer scores, poison where no
        /// launch may read.
        src: Vec<f32>,
        con: Vec<f32>,
        /// The words: `top_k` at [`TOP_K_AT`], row `t`'s count at `N_AT + t`.
        ints: Vec<u32>,
        /// Per selecting row: its block keys (the fold) and the reference's
        /// kept list.
        keys: Vec<Option<Vec<f32>>>,
        kept: Vec<Option<Vec<u32>>>,
        /// `[tokens × HIST_BINS]`: the histogram the consumer's score pass
        /// leaves (its scores' counts when `n > k`, zero otherwise).
        hist: Vec<u32>,
    }

    /// The source scores of a selecting row of `n` rows, planted as `plant`
    /// asks, and the consumer's.
    fn host(c: &Case) -> Host {
        let (t_n, r) = (c.tokens(), c.rows);
        let mut rng = Rng(c.seed);
        let mut src = vec![poison(); t_n * r];
        let mut con = vec![poison(); t_n * r];
        let mut ints = vec![0u32; N_AT + t_n];
        ints[TOP_K_AT] = c.top_k as u32;
        let (mut keys, mut kept) = (Vec::new(), Vec::new());
        let mut hist = vec![0u32; t_n * HIST_BINS];
        for (t, &(n, plant)) in c.rows_n.iter().enumerate() {
            ints[N_AT + t] = n as u32;
            let row = &mut con[t * r..t * r + n];
            for v in row.iter_mut() {
                *v = (rng.unit() - 0.5) * 8.0;
            }
            // Duplicates in the consumer's scores: every 97th row repeats
            // the one before it.
            for i in (97..n).step_by(97) {
                row[i] = row[i - 1];
            }
            if c.k() > 0 && n > c.k() {
                hist[t * HIST_BINS..(t + 1) * HIST_BINS].copy_from_slice(&counts_of(row));
            }
            if !c.selecting(n) {
                keys.push(None);
                kept.push(None);
                continue;
            }
            let b = c.block;
            let nb = n.div_ceil(b);
            let ranked = nb - 1;
            // Distinct keys: a shuffled ladder.
            let mut ladder: Vec<usize> = (0..ranked).collect();
            for i in (1..ranked).rev() {
                ladder.swap(i, rng.below(i + 1));
            }
            let mut key: Vec<f32> = ladder.iter().map(|&x| 1.0 + x as f32 * 0.25).collect();
            let mut by_rank: Vec<usize> = (0..ranked).collect();
            by_rank.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
            // The planted group: ranks blocks − 4 .. blocks + 2 of the ranked
            // blocks, the last kept place (rank blocks − 2) inside it.
            let g = by_rank[c.blocks.saturating_sub(4).min(ranked)..(c.blocks + 2).min(ranked)]
                .to_vec();
            match plant {
                Plant::Plain => {}
                Plant::Ties => {
                    let v = key[g[0]];
                    for &j in &g {
                        key[j] = v;
                    }
                }
                Plant::Zeros => {
                    let pivot = key[g[0]];
                    for k in &mut key {
                        *k -= pivot;
                    }
                    let mut sorted_g = g.clone();
                    sorted_g.sort_unstable();
                    for (i, &j) in sorted_g.iter().enumerate() {
                        key[j] = if i % 2 == 0 { -0.0 } else { 0.0 };
                    }
                }
                Plant::DropFirst => {
                    let low = key.iter().copied().fold(f32::INFINITY, f32::min);
                    key[0] = low - 1.0;
                }
            }
            let row = &mut src[t * r..t * r + n];
            for (j, rows_j) in row.chunks_mut(b).enumerate() {
                let Some(&kj) = key.get(j) else {
                    // The pin's block: any values.
                    for v in rows_j {
                        *v = (rng.unit() - 0.5) * 64.0;
                    }
                    continue;
                };
                let at = rng.below(rows_j.len());
                for (i, v) in rows_j.iter_mut().enumerate() {
                    // Below the key, or equal to it one time in eight.
                    let drop = if rng.below(8) == 0 {
                        0.0
                    } else {
                        0.125 + rng.unit()
                    };
                    *v = if i == at { kj } else { kj - drop };
                }
            }
            let k_fold: Vec<f32> = (0..nb)
                .map(|j| fold(&row[j * b..((j + 1) * b).min(n)]))
                .collect();
            kept.push(Some(reference_kept(&k_fold, n, c.blocks, b)));
            keys.push(Some(k_fold));
        }
        Host {
            src,
            con,
            ints,
            keys,
            kept,
            hist,
        }
    }

    /// The consumer expectations of row `t` under the kept list `kept`:
    /// `n_c`, the compacted slots, and the rows the remapped list holds.
    struct Consumer {
        n_c: usize,
        slots: Vec<f32>,
        /// The list the row top-k writes over the slots (slot ids).
        topk: Vec<u32>,
        /// What remap makes of it: the masked scores' best rows.
        rows: Vec<u32>,
    }

    fn consumer(c: &Case, h: &Host, t: usize, kept: &[u32]) -> Consumer {
        let (n, b, r) = (c.rows_n[t].0, c.block, c.rows);
        let n_c = n_candidates(n, c.blocks, b);
        let con = &h.con[t * r..t * r + n];
        let slots: Vec<f32> = (0..n_c).map(|s| con[slot_row(kept, b, s)]).collect();
        let k = c.k();
        let topk = if n_c <= k {
            (0..n_c as u32).collect()
        } else {
            top_ascending(&slots, k)
        };
        let mut masked = vec![f32::NEG_INFINITY; n];
        for &j in kept {
            let (r0, r1) = (j as usize * b, ((j as usize + 1) * b).min(n));
            masked[r0..r1].copy_from_slice(&con[r0..r1]);
        }
        let mut rows = top_ascending(&masked, k.min(n_c));
        rows.truncate(k.min(n_c));
        Consumer {
            n_c,
            slots,
            topk,
            rows,
        }
    }

    // ------------------------------------------------------------ the card

    struct Ctx {
        gpu: Gpu,
        cand: CandKernels,
    }

    /// One case's device buffers.
    struct Dev {
        ints: DeviceBuffer<u32>,
        src: DeviceBuffer<f32>,
        con: DeviceBuffer<f32>,
        hist: DeviceBuffer<u32>,
        kept: DeviceBuffer<u32>,
        counts: DeviceBuffer<u32>,
        list: DeviceBuffer<u32>,
        scratch: CandScratch,
    }

    impl Dev {
        fn new(cx: &Ctx, c: &Case, h: &Host) -> Result<Dev, GateError> {
            let st = cx.gpu.stream();
            let t = c.tokens();
            Ok(Dev {
                ints: DeviceBuffer::from_host(st, &h.ints)?,
                src: DeviceBuffer::from_host(st, &h.src)?,
                con: DeviceBuffer::from_host(st, &h.con)?,
                hist: DeviceBuffer::from_host(st, &h.hist)?,
                kept: DeviceBuffer::from_host(st, &vec![LIST_POISON; t * c.kstride()])?,
                counts: DeviceBuffer::from_host(st, &vec![LIST_POISON; COUNTS_AT + t + 1 + SLACK])?,
                list: DeviceBuffer::from_host(st, &vec![LIST_POISON; t * c.stride()])?,
                scratch: CandScratch::new(st, c.shape()?, t, c.rows)?,
            })
        }

        /// Poison every output and upload the inputs again.
        fn reset(&mut self, cx: &Ctx, c: &Case, h: &Host) -> Result<(), GateError> {
            let st = cx.gpu.stream();
            self.ints.copy_from_host(st, &h.ints)?;
            self.src.copy_from_host(st, &h.src)?;
            self.con.copy_from_host(st, &h.con)?;
            self.hist.copy_from_host(st, &h.hist)?;
            self.kept
                .copy_from_host(st, &vec![LIST_POISON; self.kept.len()])?;
            self.counts
                .copy_from_host(st, &vec![LIST_POISON; self.counts.len()])?;
            self.list
                .copy_from_host(st, &vec![LIST_POISON; self.list.len()])?;
            self.scratch
                .bmax
                .copy_from_host(st, &vec![poison(); self.scratch.bmax.len()])?;
            let _ = c;
            Ok(())
        }

        /// The kept lists the host rule makes, uploaded in place of the
        /// card's.
        fn host_kept(&mut self, cx: &Ctx, c: &Case, h: &Host) -> Result<(), GateError> {
            let mut v = vec![LIST_POISON; self.kept.len()];
            for (t, k) in h.kept.iter().enumerate() {
                if let Some(k) = k {
                    v[t * c.kstride()..t * c.kstride() + k.len()].copy_from_slice(k);
                }
            }
            self.kept.copy_from_host(cx.gpu.stream(), &v)?;
            Ok(())
        }

        /// The list the row top-k writes over each row's compacted slots,
        /// uploaded.
        fn host_topk(
            &mut self,
            cx: &Ctx,
            c: &Case,
            want: &[Option<Consumer>],
        ) -> Result<(), GateError> {
            let mut v = vec![LIST_POISON; self.list.len()];
            for (t, w) in want.iter().enumerate() {
                if let Some(w) = w {
                    v[t * c.stride()..t * c.stride() + w.topk.len()].copy_from_slice(&w.topk);
                }
            }
            self.list.copy_from_host(cx.gpu.stream(), &v)?;
            Ok(())
        }

        fn select(&mut self, cx: &Ctx, c: &Case) -> Result<(), GpuError> {
            cx.cand.enqueue_select(
                cx.gpu.stream(),
                SelectArgs {
                    ints: &self.ints,
                    n_at: N_AT,
                    scores: &self.src,
                    score_k: c.k(),
                    rows: c.rows,
                    tokens: c.tokens(),
                    shape: c.shape()?,
                    fault: cx.gpu.layer_sink(SOURCE_LAYER)?,
                    scratch: &mut self.scratch,
                    kept: &mut self.kept,
                    kstride: c.kstride(),
                },
            )?;
            Ok(())
        }

        fn compact(&mut self, cx: &Ctx, c: &Case) -> Result<(), GpuError> {
            cx.cand.enqueue_compact(
                cx.gpu.stream(),
                CompactArgs {
                    ints: &self.ints,
                    n_at: N_AT,
                    top_k_at: TOP_K_AT,
                    kept: &self.kept,
                    kstride: c.kstride(),
                    rows: c.rows,
                    tokens: c.tokens(),
                    stride: c.stride(),
                    shape: c.shape()?,
                    fault: cx.gpu.layer_sink(CONSUMER_LAYER)?,
                    scores: &mut self.con,
                    hist: &mut self.hist,
                    counts: &mut self.counts,
                    counts_at: COUNTS_AT,
                },
            )?;
            Ok(())
        }

        fn remap(&mut self, cx: &Ctx, c: &Case) -> Result<(), GpuError> {
            cx.cand.enqueue_remap(
                cx.gpu.stream(),
                RemapArgs {
                    ints: &self.ints,
                    n_at: N_AT,
                    kept: &self.kept,
                    kstride: c.kstride(),
                    counts: &self.counts,
                    counts_at: COUNTS_AT,
                    rows: c.rows,
                    tokens: c.tokens(),
                    shape: c.shape()?,
                    fault: cx.gpu.layer_sink(CONSUMER_LAYER)?,
                    list: &mut self.list,
                    stride: c.stride(),
                },
            )?;
            Ok(())
        }
    }

    /// Everything a run leaves, read back.
    #[derive(PartialEq)]
    struct Out {
        bmax: Vec<u32>,
        kept: Vec<u32>,
        con: Vec<u32>,
        hist: Vec<u32>,
        counts: Vec<u32>,
        list: Vec<u32>,
    }

    fn read(cx: &Ctx, d: &Dev) -> Result<Out, GateError> {
        let st = cx.gpu.stream();
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        Ok(Out {
            bmax: bits(d.scratch.bmax.to_host_vec(st)?),
            kept: d.kept.to_host_vec(st)?,
            con: bits(d.con.to_host_vec(st)?),
            hist: d.hist.to_host_vec(st)?,
            counts: d.counts.to_host_vec(st)?,
            list: d.list.to_host_vec(st)?,
        })
    }

    fn shown(f: Option<Fault>) -> String {
        f.map_or_else(|| "none".to_owned(), |f| f.to_string())
    }

    /// No fault after a clean run.
    fn clean(cx: &Ctx, what: &str) -> Result<bool, GateError> {
        let f = cx.gpu.take_fault()?;
        if f.is_some() {
            println!("clean {what}: fault {} FAIL", shown(f));
        }
        Ok(f.is_none())
    }

    // ------------------------------------------------------------ clauses

    /// Clauses 1 and 2 (module doc).
    fn check_source(cx: &Ctx, c: &Case, h: &Host, d: &mut Dev) -> Result<bool, GateError> {
        d.reset(cx, c, h)?;
        d.select(cx, c)?;
        let o = read(cx, d)?;
        let quiet = clean(cx, &format!("source {}", c.name))?;
        let nb_cap = c.rows.div_ceil(c.block);
        let (mut keys_ok, mut kept_ok) = (true, true);
        let mut notes = Vec::new();
        for (t, &(n, plant)) in c.rows_n.iter().enumerate() {
            let brow = &o.bmax[t * nb_cap..(t + 1) * nb_cap];
            let krow = &o.kept[t * c.kstride()..(t + 1) * c.kstride()];
            match (&h.keys[t], &h.kept[t]) {
                (Some(keys), Some(_)) => {
                    let nb = keys.len();
                    let k_ok = brow[..nb].iter().zip(keys).all(|(a, b)| *a == b.to_bits())
                        && brow[nb..].iter().all(|&v| v == POISON);
                    // The selection over the card's own keys: clause 2 holds
                    // the list to the rule whatever clause 1 finds.
                    let card: Vec<f32> = brow[..nb].iter().map(|&b| f32::from_bits(b)).collect();
                    let kept = &reference_kept(&card, n, c.blocks, c.block);
                    let l_ok = krow[..c.blocks] == kept[..]
                        && krow[c.blocks..].iter().all(|&v| v == LIST_POISON);
                    if !l_ok {
                        let first = krow
                            .iter()
                            .zip(kept)
                            .position(|(a, b)| a != b)
                            .unwrap_or(kept.len());
                        notes.push(format!(
                            "n={n} {plant:?}: first differing entry {first} (card {:?}, rule {:?})",
                            krow.get(first),
                            kept.get(first)
                        ));
                    }
                    keys_ok &= k_ok;
                    kept_ok &= l_ok;
                }
                _ => {
                    let untouched =
                        brow.iter().all(|&v| v == POISON) && krow.iter().all(|&v| v == LIST_POISON);
                    keys_ok &= untouched;
                    kept_ok &= untouched;
                }
            }
        }
        println!(
            "keys {} rows={} blocks={}x{} counts={} selecting={} {}",
            c.name,
            c.rows,
            c.blocks,
            c.block,
            ns(c),
            c.rows_n.iter().filter(|(n, _)| c.selecting(*n)).count(),
            verdict(keys_ok && quiet)
        );
        println!(
            "kept {} plants={} {}{}",
            c.name,
            c.rows_n
                .iter()
                .map(|(n, p)| format!("{n}:{p:?}"))
                .collect::<Vec<_>>()
                .join(","),
            verdict(kept_ok && quiet),
            if notes.is_empty() {
                String::new()
            } else {
                format!(" ({})", notes.join("; "))
            }
        );
        Ok(keys_ok && kept_ok && quiet)
    }

    fn ns(c: &Case) -> String {
        c.rows_n
            .iter()
            .map(|(n, _)| n.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The consumer expectations of every row of `c` under the host kept
    /// lists.
    fn wants(c: &Case, h: &Host) -> Vec<Option<Consumer>> {
        (0..c.tokens())
            .map(|t| h.kept[t].as_ref().map(|k| consumer(c, h, t, k)))
            .collect()
    }

    /// Clause 3's checks on a run's readback: scores, histogram and counts.
    fn compact_ok(c: &Case, h: &Host, want: &[Option<Consumer>], o: &Out) -> (bool, bool, bool) {
        let r = c.rows;
        let k = c.k();
        let (mut sc, mut hs, mut ct) = (true, true, true);
        for (t, w) in want.iter().enumerate() {
            let n = c.rows_n[t].0;
            let got = &o.con[t * r..(t + 1) * r];
            let was: Vec<u32> = h.con[t * r..(t + 1) * r]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let hrow = &o.hist[t * HIST_BINS..(t + 1) * HIST_BINS];
            let hwas = &h.hist[t * HIST_BINS..(t + 1) * HIST_BINS];
            match w {
                Some(w) if k > 0 && w.n_c > k => {
                    sc &= got[..w.n_c]
                        .iter()
                        .zip(&w.slots)
                        .all(|(a, b)| *a == b.to_bits())
                        && got[w.n_c..] == was[w.n_c..];
                    hs &= hrow == &counts_of(&w.slots)[..];
                    ct &= o.counts[COUNTS_AT + t] == w.n_c as u32;
                }
                Some(w) => {
                    sc &= got == &was[..];
                    hs &= hrow.iter().all(|&v| v == 0);
                    ct &= o.counts[COUNTS_AT + t] == w.n_c as u32;
                }
                None => {
                    sc &= got == &was[..];
                    hs &= hrow == hwas;
                    ct &= o.counts[COUNTS_AT + t] == n as u32;
                }
            }
        }
        let t_n = c.tokens();
        ct &= o.counts[COUNTS_AT + t_n] == c.top_k as u32
            && o.counts[..COUNTS_AT].iter().all(|&v| v == LIST_POISON)
            && o.counts[COUNTS_AT + t_n + 1..]
                .iter()
                .all(|&v| v == LIST_POISON);
        (sc, hs, ct)
    }

    /// Clause 4's check on a run's readback.
    fn remap_ok(c: &Case, want: &[Option<Consumer>], o: &Out) -> bool {
        let s = c.stride();
        want.iter().enumerate().all(|(t, w)| {
            let row = &o.list[t * s..(t + 1) * s];
            match w {
                Some(w) => {
                    row[..w.rows.len()] == w.rows[..]
                        && row[w.rows.len()..].iter().all(|&v| v == LIST_POISON)
                }
                None => row.iter().all(|&v| v == LIST_POISON),
            }
        })
    }

    /// Clauses 3 and 4 (module doc): the consumer entries on the host rule's
    /// kept lists.
    fn check_consumer(cx: &Ctx, c: &Case, h: &Host, d: &mut Dev) -> Result<bool, GateError> {
        let want = wants(c, h);
        d.reset(cx, c, h)?;
        d.host_kept(cx, c, h)?;
        d.compact(cx, c)?;
        let o = read(cx, d)?;
        let quiet_c = clean(cx, &format!("compact {}", c.name))?;
        let (sc, hs, ct) = compact_ok(c, h, &want, &o);
        println!(
            "compact {} counts={} top_k={} scores={} hist={} counts_view={} {}",
            c.name,
            ns(c),
            c.top_k,
            verdict(sc),
            verdict(hs),
            verdict(ct),
            verdict(sc && hs && ct && quiet_c)
        );
        d.host_topk(cx, c, &want)?;
        d.remap(cx, c)?;
        let o = read(cx, d)?;
        let quiet_r = clean(cx, &format!("remap {}", c.name))?;
        let rm = remap_ok(c, &want, &o);
        println!(
            "remap {} lists={} {}",
            c.name,
            want.iter()
                .map(|w| w
                    .as_ref()
                    .map_or("-".to_string(), |w| w.rows.len().to_string()))
                .collect::<Vec<_>>()
                .join(","),
            verdict(rm && quiet_r)
        );
        Ok(sc && hs && ct && quiet_c && rm && quiet_r)
    }

    /// Clause 5 (module doc): the card's own kept lists through the consumer.
    fn check_pipeline(cx: &Ctx, c: &Case, h: &Host, d: &mut Dev) -> Result<bool, GateError> {
        let want = wants(c, h);
        d.reset(cx, c, h)?;
        d.select(cx, c)?;
        d.compact(cx, c)?;
        d.host_topk(cx, c, &want)?;
        d.remap(cx, c)?;
        let o = read(cx, d)?;
        let quiet = clean(cx, &format!("pipeline {}", c.name))?;
        let (sc, hs, ct) = compact_ok(c, h, &want, &o);
        let rm = remap_ok(c, &want, &o);
        let pass = sc && hs && ct && rm && quiet;
        println!("pipeline {} {}", c.name, verdict(pass));
        Ok(pass)
    }

    /// Clause 6 (module doc).
    fn check_fault(cx: &Ctx) -> Result<bool, GateError> {
        let base = Case {
            name: "fault",
            blocks: 16,
            block: 8,
            rows: 512,
            top_k: 64,
            rows_n: vec![
                (301, Plant::Plain),
                (400, Plant::Plain),
                (100, Plant::Plain),
            ],
            seed: 0x0cad_00f0,
        };
        let h = host(&base);
        let mut d = Dev::new(cx, &base, &h)?;
        let src_at = Fault::at(SOURCE_LAYER as u32, FaultSite::CandMask);
        let con_at = Fault::at(CONSUMER_LAYER as u32, FaultSite::CandMask);
        let r = base.rows;
        let mut ok = true;

        // A source score that is not finite, one row of row 1.
        for (tag, v) in [
            ("nan", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let mut hh = host(&base);
            hh.src[r + 137] = v;
            d.reset(cx, &base, &hh)?;
            d.select(cx, &base)?;
            let o = read(cx, &d)?;
            let f = cx.gpu.take_fault()?;
            let row0 = &o.kept[..base.kstride()];
            let other = h.kept[0]
                .as_ref()
                .is_some_and(|k| row0[..base.blocks] == k[..]);
            let pass = f == Some(src_at) && other;
            println!(
                "fault source {tag}: {} (want {src_at}) other_row_exact={other} {}",
                shown(f),
                verdict(pass)
            );
            ok &= pass;
        }

        // A count past the rows, at the source and at the consumer.
        {
            let mut hh = host(&base);
            hh.ints[N_AT + 2] = (r + 1) as u32;
            d.reset(cx, &base, &hh)?;
            d.select(cx, &base)?;
            let fs = cx.gpu.take_fault()?;
            d.host_kept(cx, &base, &h)?;
            d.compact(cx, &base)?;
            let o = read(cx, &d)?;
            let fc = cx.gpu.take_fault()?;
            let pass = fs == Some(src_at) && fc == Some(con_at) && o.counts[COUNTS_AT + 2] == 0;
            println!(
                "fault count past rows: source {} consumer {} n_c={} {}",
                shown(fs),
                shown(fc),
                o.counts[COUNTS_AT + 2],
                verdict(pass)
            );
            ok &= pass;
        }

        // A kept list out of order (in range), ending past its pin, or with an
        // entry past every block.
        let kept0 = h.kept[0].clone().unwrap_or_default();
        let nb0 = 301usize.div_ceil(8) as u32;
        let mut corrupt: Vec<(&str, Vec<u32>)> = Vec::new();
        {
            let mut k = kept0.clone();
            k.swap(3, 4);
            corrupt.push(("swapped", k));
            let mut k = kept0.clone();
            if let Some(last) = k.last_mut() {
                *last = nb0;
            }
            corrupt.push(("past pin", k));
            let mut k = kept0.clone();
            k[5] = u32::MAX;
            corrupt.push(("huge entry", k));
        }
        // The entry past every block goes last and only after both in-range
        // corruptions were refused: a kernel that does not check would read
        // out of range there, and it has already failed on the others.
        let mut refused_in_range = true;
        for (tag, k) in corrupt {
            if tag == "huge entry" && !refused_in_range {
                println!("fault kept {tag}: skipped, an in-range corruption was not refused FAIL");
                ok = false;
                continue;
            }
            d.reset(cx, &base, &h)?;
            d.host_kept(cx, &base, &h)?;
            let mut v = d.kept.to_host_vec(cx.gpu.stream())?;
            v[..k.len()].copy_from_slice(&k);
            d.kept.copy_from_host(cx.gpu.stream(), &v)?;
            d.compact(cx, &base)?;
            let o = read(cx, &d)?;
            let f = cx.gpu.take_fault()?;
            let was: Vec<u32> = h.con[..r].iter().map(|v| v.to_bits()).collect();
            let pass = f == Some(con_at) && o.counts[COUNTS_AT] == 0 && o.con[..r] == was[..];
            println!(
                "fault kept {tag}: {} n_c={} scores_untouched={} {}",
                shown(f),
                o.counts[COUNTS_AT],
                o.con[..r] == was[..],
                verdict(pass)
            );
            ok &= pass;
            refused_in_range &= pass;
        }

        // A consumer score that is not finite, in a kept block of row 0.
        {
            let mut hh = host(&base);
            let row = kept0[2] as usize * 8 + 3;
            hh.con[row] = f32::NAN;
            d.reset(cx, &base, &hh)?;
            d.host_kept(cx, &base, &h)?;
            d.compact(cx, &base)?;
            let f = cx.gpu.take_fault()?;
            let pass = f == Some(con_at);
            println!("fault consumer nan: {} {}", shown(f), verdict(pass));
            ok &= pass;
        }

        // A list entry past n_c.
        {
            let want = wants(&base, &h);
            d.reset(cx, &base, &h)?;
            d.host_kept(cx, &base, &h)?;
            d.compact(cx, &base)?;
            let _ = cx.gpu.take_fault()?;
            d.host_topk(cx, &base, &want)?;
            let mut v = d.list.to_host_vec(cx.gpu.stream())?;
            let n_c = want[0].as_ref().map_or(0, |w| w.n_c) as u32;
            v[7] = n_c;
            d.list.copy_from_host(cx.gpu.stream(), &v)?;
            d.remap(cx, &base)?;
            let o = read(cx, &d)?;
            let f = cx.gpu.take_fault()?;
            let rows0 = want[0].as_ref().map(|w| w.rows.clone()).unwrap_or_default();
            let others = (0..rows0.len())
                .filter(|&i| i != 7)
                .all(|i| o.list[i] == rows0[i]);
            let pass = f == Some(con_at) && o.list[7] == u32::MAX && others;
            println!(
                "fault list entry at n_c: {} entry={:#x} others_mapped={others} {}",
                shown(f),
                o.list[7],
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause 7 (module doc).
    fn check_replay(cx: &Ctx) -> Result<bool, GateError> {
        let variants: [[usize; 2]; 4] = [
            [32_768, 20_001],
            [16_385, 16_384],
            [400, 32_767],
            [24_576, 16_392],
        ];
        let case_of = |ns: [usize; 2]| Case {
            name: "replay",
            blocks: 2048,
            block: 8,
            rows: 32_768,
            top_k: 512,
            rows_n: ns.iter().map(|&n| (n, Plant::Ties)).collect(),
            seed: 0x0cad_00e0,
        };
        let c0 = case_of(variants[0]);
        let h0 = host(&c0);
        let mut d = Dev::new(cx, &c0, &h0)?;
        d.reset(cx, &c0, &h0)?;
        let graph = cx.gpu.capture(|_s| {
            d.select(cx, &c0)?;
            d.compact(cx, &c0)?;
            d.remap(cx, &c0)
        })?;
        let (mut same, mut exact) = (0usize, 0usize);
        for ns in variants {
            let c = case_of(ns);
            let h = host(&c);
            let want = wants(&c, &h);
            // Eager, with the host top-k in place: select, compact, then the
            // list, then remap.
            d.reset(cx, &c, &h)?;
            d.select(cx, &c)?;
            d.compact(cx, &c)?;
            d.host_topk(cx, &c, &want)?;
            d.remap(cx, &c)?;
            let eager = read(cx, &d)?;
            let (sc, hs, ct) = compact_ok(&c, &h, &want, &eager);
            exact += usize::from(sc && hs && ct && remap_ok(&c, &want, &eager));
            // The graph over the same inputs: the list the eager run's remap
            // read, uploaded before the replay.
            d.reset(cx, &c, &h)?;
            d.host_topk(cx, &c, &want)?;
            graph.launch(cx.gpu.stream())?;
            let replay = read(cx, &d)?;
            same += usize::from(replay == eager);
        }
        let quiet = clean(cx, "replay")?;
        let m = variants.len();
        let pass = same == m && exact == m && quiet;
        println!(
            "replay rows=32768 nodes={} counts={} bit_identical_to_eager={same}/{m} eager_exact={exact}/{m} {}",
            graph.node_count(),
            variants
                .iter()
                .map(|v| format!("{}/{}", v[0], v[1]))
                .collect::<Vec<_>>()
                .join(","),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause 8 (module doc).
    fn check_refuse(cx: &Ctx) -> Result<bool, GateError> {
        let mut ok = true;
        for (blocks, block) in [(0usize, 8usize), (16, 6), (16, 64)] {
            let refused = CandShape::new(blocks, block).is_err();
            println!(
                "refuse shape blocks={blocks} block={block} {}",
                verdict(refused)
            );
            ok &= refused;
        }
        let c = Case {
            name: "refuse",
            blocks: 16,
            block: 8,
            rows: 512,
            top_k: 64,
            rows_n: vec![(301, Plant::Plain)],
            seed: 0x0cad_00d0,
        };
        let h = host(&c);
        let mut d = Dev::new(cx, &c, &h)?;
        let narrow = cx
            .cand
            .enqueue_select(
                cx.gpu.stream(),
                SelectArgs {
                    ints: &d.ints,
                    n_at: N_AT,
                    scores: &d.src,
                    score_k: c.k(),
                    rows: c.rows,
                    tokens: 1,
                    shape: c.shape()?,
                    fault: cx.gpu.layer_sink(SOURCE_LAYER)?,
                    scratch: &mut d.scratch,
                    kept: &mut d.kept,
                    kstride: c.blocks - 1,
                },
            )
            .is_err();
        let other = cx
            .cand
            .enqueue_select(
                cx.gpu.stream(),
                SelectArgs {
                    ints: &d.ints,
                    n_at: N_AT,
                    scores: &d.src,
                    score_k: c.k(),
                    rows: c.rows,
                    tokens: 1,
                    shape: CandShape::new(16, 4)?,
                    fault: cx.gpu.layer_sink(SOURCE_LAYER)?,
                    scratch: &mut d.scratch,
                    kept: &mut d.kept,
                    kstride: c.kstride(),
                },
            )
            .is_err();
        let mut unwritten = true;
        for score_k in [0usize, 129] {
            unwritten &= cx
                .cand
                .enqueue_select(
                    cx.gpu.stream(),
                    SelectArgs {
                        ints: &d.ints,
                        n_at: N_AT,
                        scores: &d.src,
                        score_k,
                        rows: c.rows,
                        tokens: 1,
                        shape: c.shape()?,
                        fault: cx.gpu.layer_sink(SOURCE_LAYER)?,
                        scratch: &mut d.scratch,
                        kept: &mut d.kept,
                        kstride: c.kstride(),
                    },
                )
                .is_err();
        }
        println!(
            "refuse kept stride below the blocks {} ; scratch of another block size {} ; score k 0 \
             or past the kept rows {}",
            verdict(narrow),
            verdict(other),
            verdict(unwritten)
        );
        ok &= narrow && other && unwritten && clean(cx, "refuse")?;
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let cand = CandKernels::load(gpu.context())?;
        let cx = Ctx { gpu, cand };
        if let Some(f) = cx.gpu.take_fault()? {
            return Err(format!("gate_cand: the fault word held {f} before any launch").into());
        }
        let mut ok = true;
        for c in cases() {
            let h = host(&c);
            let mut d = Dev::new(&cx, &c, &h)?;
            if c.source() {
                ok &= check_source(&cx, &c, &h, &mut d)?;
            }
            ok &= check_consumer(&cx, &c, &h, &mut d)?;
            if c.source() {
                ok &= check_pipeline(&cx, &c, &h, &mut d)?;
            }
        }
        ok &= check_fault(&cx)?;
        ok &= check_replay(&cx)?;
        ok &= check_refuse(&cx)?;
        ok &= no_local_depot(&[
            "cand_block_max",
            "cand_select",
            "cand_compact",
            "cand_remap",
        ])?;
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        println!(
            "PASSED: gate_cand block keys the fold of the visible rows and kept lists the \
             reference's selection over them (pin, ties to the lower block, ±0 tied, one block \
             dropped) at 2048x8, 16x8, 64x1 and 8x32; nothing written at or below the bound; the \
             compaction's slots, histogram and counts view and the remapped lists exact, alone \
             and after the card's own selection; every refusal raises cand_mask at its layer; a \
             captured replay equals eager; shapes refused by name; no local depot"
        );
        Ok(())
    }
}
