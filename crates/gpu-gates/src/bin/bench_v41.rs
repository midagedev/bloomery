//! Bench for the GPU's dense reads at DeepSeek-V4.1's shapes: every gemv a
//! V4.1 decode token issues on the GPU, launched the way the engine's
//! kernels launch today, over synthetic weights of the real device size.
//! It is not a gate; the verdicts it prints guard its own measurement.
//!
//! The shape table (`SITES`, in decode order) is derived from
//! `docs/v41-inventory.md` (tensor names, dims, types, bytes) and
//! `docs/research/v41-op-map.md` (how each tensor is applied). It is printed
//! at start with its per-token totals. Every row's per-tensor file bytes must
//! equal the inventory's, and the q8_0 rows must add up to the count and
//! bytes of `docs/v41-placement.md` §1, or the run stops before it touches
//! the card. The block-diagonal `attn_output_a` has three rows, each over
//! the same weight bytes: one launch per group (the row today's token
//! issues), one dense-equivalent launch, and one launch of
//! `q8_0_gemv_heads`, a head per group — each group's rows reading their own
//! window of the attention output, the batched form. The routed experts run at
//! `N_SLOTS` slots per launch over stacks of `N_SLOTS` experts, in the
//! blocks whose experts can live in VRAM (blocks `0..EXPERT_LO` keep theirs
//! on the host). The `hc_*_fn` projections are listed but not issued: the
//! V4.1 crate's `ds41_hc_pre` runs them split-K inside the fused HC_PRE
//! kernel, which this `gpu`-feature bench does not link; a plain q3_K gemv
//! at their shape would time a kernel the step never launches.
//!
//! Device data: one distinct synthetic copy per block, so the reads of a
//! token miss L2 the way the real step's do. A site whose copies together
//! would fit in a few L2s gets extra copies for its own timing only
//! (`CYCLE_FLOOR`); the token graph uses the real ones. f16 scale fields are
//! positive normal values, every other byte is random. Weights go up in the
//! device formats the engine's launchers take.
//!
//! `--check`: each site runs one launch on its first and last copy, and a
//! handful of output rows — for the heads launch, the first row of every
//! head among them — are compared with an f64 host reference computed
//! from the same bytes — `d × code` for q8_0, `gguf::quant::dequant_row` for
//! the K-quants against the host transcription of their q8_1 activation.
//! The band is [`KERNEL_BAND`]: accumulation rounding alone sits orders of
//! magnitude below it, and a wrong row, scale, activation or copy lands
//! orders of magnitude above it. A failing site is named and the run exits
//! non-zero. Each checked launch also prints an FNV-1a 64 digest of its
//! whole output (`digest` lines): the reference is sampled, the digest is
//! not, so two builds whose digest lines match wrote the same bits in every
//! row.
//!
//! `--time` (lead-only, under `tools/ref/time-gate.sh`, which owns the lease,
//! the witness blocks and the card pin): the check first, refusing to time
//! if it fails; then per site an eager burst and a graph replay cycling its
//! copies, in the pattern of `gate_p8 --bench-kernels`; then one captured
//! graph of a whole token in decode order (all blocks, then the head), a
//! second with the dense-equivalent `attn_output_a` in place of the grouped
//! launches, a third with the `q8_0_gemv_heads` launch there, and beside
//! each a graph of empty `touch` kernels with its node count as the per-node
//! floor.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("bench_v41: built without the `gpu` feature; see `just bench-gpu-v41-check`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("bench_v41", bench::run())
}

#[cfg(feature = "gpu")]
mod bench {
    use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
    use bloomery_gpu::probe::Probe;
    use bloomery_gpu::weights::resident_size;
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Graph, Q8Act};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, activations, bits_equal, bytes_to_words, checks_failed,
        max_rel_err, q8_1_dequant, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, DriverError, sys};
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use std::time::Instant;

    /// Backbone blocks of the model.
    const N_LAYERS: usize = 40;
    /// Experts in one routed stack of the file (the inventory's third dim).
    const N_EXPERTS_FILE: usize = 384;
    /// Routed top-k: slots per `_sel` launch, and experts per synthetic stack.
    const N_SLOTS: usize = 6;
    /// Slot s reads expert `SEL[s]` of its stack: every id distinct and no
    /// slot reading its own index, so a kernel that confused slot with id,
    /// or read one expert twice, fails the check.
    const SEL: [u32; N_SLOTS] = [3, 5, 0, 4, 1, 2];
    /// First block whose routed experts can live in VRAM; the blocks below
    /// keep their experts on the host (`docs/v41-placement.md` §5).
    const EXPERT_LO: usize = 2;
    /// Blocks that pool the compressed KV and own the index keys.
    const KV_SOURCE: &[usize] = &[2, 8, 14, 20];
    /// KV sources with a pooling gate (ratio above one).
    const POOL_GATE: &[usize] = &[2, 8, 14];
    /// Blocks that run the indexer's top-k.
    const INDEX_SOURCE: &[usize] = &[2, 8, 14, 20, 24, 28, 32, 36];
    /// Blocks with an engram lookup.
    const ENGRAM: &[usize] = &[1, 14];
    /// L2 of both cards.
    const L2_BYTES: u64 = 6 << 20;
    /// A site's timing cycles through at least this many bytes of distinct
    /// copies, so no copy is still in L2 when its turn comes again.
    const CYCLE_FLOOR: u64 = 4 * L2_BYTES;
    /// Free device memory the plan leaves untouched: the plan counts the
    /// bytes the bench asks for, not what the driver spends placing them,
    /// and the graphs need room of their own.
    const VRAM_MARGIN: u64 = 1 << 30;
    /// `docs/v41-placement.md` §1: q8_0 tensors outside `engram_embd`, their
    /// file bytes, and their bytes in the Q8_0 device planes — the file's
    /// bytes, since the scale plane keeps each block's f16 bits (the f16
    /// scale plane column of §1's q8_0 row).
    const DOC_Q8_TENSORS: usize = 330;
    const DOC_Q8_FILE_BYTES: u64 = 7_264_010_240;
    const DOC_Q8_DEVICE_BYTES: u64 = DOC_Q8_FILE_BYTES;
    /// The `hc_{attn,ffn}_fn` projections: q3_K, K = 4 streams × 5120, 24
    /// mix values per row, two per block.
    const HC_K: usize = 20_480;
    const HC_ROWS: usize = 24;
    const HC_TENSORS: usize = 2 * N_LAYERS;
    /// Launches per eager burst and nodes per site graph, at least; a site
    /// rounds this up to whole cycles of its copies.
    const MIN_BURST: usize = 64;
    /// Bursts (or graph replay rounds) per arm; their spread is printed.
    const ROUNDS: usize = 7;
    /// Graph launches per round of a site graph.
    const GREPS: usize = 4;
    /// Graph launches per round of a token graph.
    const TOKEN_REPS: usize = 2;

    /// The kernel a site launches.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kernel {
        Q8,
        /// `q8_0_gemv_heads`: the Q8_0 row body per head, each head's rows
        /// against their own window of the activation.
        Q8Heads,
        F32,
        Q3k,
        Q3kSel,
        Q4kSel,
        Q6k,
    }

    impl Kernel {
        fn name(self) -> &'static str {
            match self {
                Kernel::Q8 => "q8_0_gemv",
                Kernel::Q8Heads => "q8_0_gemv_heads",
                Kernel::F32 => "f32_gemv",
                Kernel::Q3k => "q3k_gemv",
                Kernel::Q3kSel => "q3k_gemv_sel",
                Kernel::Q4kSel => "q4k_gemv_sel",
                Kernel::Q6k => "q6k_gemv",
            }
        }

        /// The tensor's type in the file (the router's bf16 is decoded to
        /// f32 at load).
        fn file_type(self) -> &'static str {
            match self {
                Kernel::Q8 | Kernel::Q8Heads => "q8_0",
                Kernel::F32 => "bf16",
                Kernel::Q3k | Kernel::Q3kSel => "q3_K",
                Kernel::Q4kSel => "q4_K",
                Kernel::Q6k => "q6_K",
            }
        }

        fn ggml(self) -> Option<GgmlType> {
            match self {
                Kernel::Q3k | Kernel::Q3kSel => Some(GgmlType::Q3_K),
                Kernel::Q4kSel => Some(GgmlType::Q4_K),
                Kernel::Q6k => Some(GgmlType::Q6_K),
                Kernel::Q8 | Kernel::Q8Heads | Kernel::F32 => None,
            }
        }

        fn is_sel(self) -> bool {
            matches!(self, Kernel::Q3kSel | Kernel::Q4kSel)
        }
    }

    /// Which blocks carry a site.
    #[derive(Clone, Copy)]
    enum Layers {
        All,
        /// Blocks `lo..hi`.
        Span(usize, usize),
        Only(&'static [usize]),
        /// Once per token, after the last block.
        Head,
    }

    impl Layers {
        /// Positions in decode order; the head sits at `N_LAYERS`.
        fn list(self) -> Vec<usize> {
            match self {
                Layers::All => (0..N_LAYERS).collect(),
                Layers::Span(lo, hi) => (lo..hi).collect(),
                Layers::Only(l) => l.to_vec(),
                Layers::Head => vec![N_LAYERS],
            }
        }

        fn label(self) -> String {
            match self {
                Layers::All => format!("0..{N_LAYERS}"),
                Layers::Span(lo, hi) => format!("{lo}..{hi}"),
                Layers::Only(l) => l.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
                Layers::Head => "head".to_string(),
            }
        }
    }

    /// Whether a site belongs to a token graph.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        /// Issued by every token.
        Token,
        /// Issued by today's token: one launch per diagonal block.
        BlockDiag,
        /// Priced only: the same bytes as one dense launch, the byte floor of
        /// a batched kernel.
        DenseEquiv,
        /// Priced only: the same bytes as one `q8_0_gemv_heads` launch, a
        /// head per diagonal block — the batched kernel.
        Heads,
    }

    /// One gemv site of a decode token.
    #[derive(Clone, Copy)]
    struct Site {
        /// Tensor name under `blk.N.` (the file name for the head).
        name: &'static str,
        kernel: Kernel,
        /// Output rows of one launch — per expert for the `_sel` kernels.
        rows: usize,
        /// Values per row.
        k: usize,
        layers: Layers,
        /// Launches per block: one per diagonal block of `attn_output_a`.
        groups: usize,
        /// Heads of one `q8_0_gemv_heads` launch: `rows / heads` rows each,
        /// head h reading window h (`k` values) of the activation. One for
        /// every other kernel.
        heads: usize,
        arm: Arm,
        /// The inventory's bytes for one tensor of this site.
        inventory_bytes: u64,
    }

    const fn site(
        name: &'static str,
        kernel: Kernel,
        rows: usize,
        k: usize,
        layers: Layers,
        inventory_bytes: u64,
    ) -> Site {
        Site {
            name,
            kernel,
            rows,
            k,
            layers,
            groups: 1,
            heads: 1,
            arm: Arm::Token,
            inventory_bytes,
        }
    }

    /// The table, in decode order within a block (engram, attention, router,
    /// routed experts, shared expert), then the head.
    const SITES: &[Site] = &[
        site(
            "engram_wkv",
            Kernel::Q8,
            25_600,
            6144,
            Layers::Only(ENGRAM),
            167_116_800,
        ),
        site("attn_q_a", Kernel::Q8, 1280, 5120, Layers::All, 6_963_200),
        site(
            "attn_q_b",
            Kernel::Q8,
            32_768,
            1280,
            Layers::All,
            44_564_480,
        ),
        site("attn_kv", Kernel::Q8, 512, 5120, Layers::All, 2_785_280),
        site(
            "attn_compressor_kv",
            Kernel::Q3k,
            512,
            5120,
            Layers::Only(KV_SOURCE),
            1_126_400,
        ),
        site(
            "attn_compressor_gate",
            Kernel::Q3k,
            512,
            5120,
            Layers::Only(POOL_GATE),
            1_126_400,
        ),
        site(
            "indexer.attn_k",
            Kernel::Q3k,
            128,
            512,
            Layers::Only(KV_SOURCE),
            28_160,
        ),
        site(
            "indexer.attn_q_b",
            Kernel::Q8,
            4096,
            1280,
            Layers::Only(INDEX_SOURCE),
            5_570_560,
        ),
        site(
            "indexer.proj",
            Kernel::Q3k,
            32,
            5120,
            Layers::Only(INDEX_SOURCE),
            70_400,
        ),
        Site {
            groups: 8,
            arm: Arm::BlockDiag,
            ..site(
                "attn_output_a",
                Kernel::Q8,
                1024,
                4096,
                Layers::All,
                35_651_584,
            )
        },
        Site {
            arm: Arm::DenseEquiv,
            ..site(
                "attn_output_a",
                Kernel::Q8,
                8192,
                4096,
                Layers::All,
                35_651_584,
            )
        },
        Site {
            heads: 8,
            arm: Arm::Heads,
            ..site(
                "attn_output_a",
                Kernel::Q8Heads,
                8192,
                4096,
                Layers::All,
                35_651_584,
            )
        },
        site(
            "attn_output_b",
            Kernel::Q8,
            5120,
            8192,
            Layers::All,
            44_564_480,
        ),
        site(
            "ffn_gate_inp",
            Kernel::F32,
            384,
            5120,
            Layers::All,
            3_932_160,
        ),
        site(
            "ffn_gate_exps",
            Kernel::Q3kSel,
            2304,
            5120,
            Layers::Span(EXPERT_LO, N_LAYERS),
            1_946_419_200,
        ),
        site(
            "ffn_up_exps",
            Kernel::Q3kSel,
            2304,
            5120,
            Layers::Span(EXPERT_LO, N_LAYERS),
            1_946_419_200,
        ),
        site(
            "ffn_down_exps",
            Kernel::Q4kSel,
            5120,
            2304,
            Layers::Span(EXPERT_LO, N_LAYERS),
            2_548_039_680,
        ),
        site(
            "ffn_gate_shexp",
            Kernel::Q8,
            2304,
            5120,
            Layers::All,
            12_533_760,
        ),
        site(
            "ffn_up_shexp",
            Kernel::Q8,
            2304,
            5120,
            Layers::All,
            12_533_760,
        ),
        site(
            "ffn_down_shexp",
            Kernel::Q8,
            5120,
            2304,
            Layers::All,
            12_533_760,
        ),
        site(
            "output",
            Kernel::Q6k,
            129_280,
            5120,
            Layers::Head,
            542_976_000,
        ),
    ];

    impl Site {
        /// The name as printed: the pricing arm carries a suffix.
        fn label(&self) -> String {
            match self.arm {
                Arm::DenseEquiv => format!("{}:dense", self.name),
                Arm::Heads => format!("{}:heads", self.name),
                Arm::Token | Arm::BlockDiag => self.name.to_string(),
            }
        }

        /// Output values of one launch (all slots for the `_sel` kernels),
        /// and the rows of one device copy: a `_sel` stack holds `N_SLOTS`
        /// experts, which is what one launch reads.
        fn out_len(&self) -> usize {
            if self.kernel.is_sel() {
                N_SLOTS * self.rows
            } else {
                self.rows
            }
        }

        /// Activation columns a site holds: one per diagonal block, one per
        /// head window, one per slot for the per-slot down input, else the
        /// one shared column.
        fn x_cols(&self) -> usize {
            match self.kernel {
                Kernel::Q4kSel => N_SLOTS,
                Kernel::Q8Heads => self.heads,
                _ => self.groups,
            }
        }

        /// Activation columns one launch reads.
        fn launch_cols(&self) -> usize {
            match self.kernel {
                Kernel::Q4kSel => N_SLOTS,
                Kernel::Q8Heads => self.heads,
                _ => 1,
            }
        }

        /// Bytes of one weight row in the device format; an error naming the
        /// site when its k has no layout in the loader's q8_0 card format.
        fn device_row_bytes(&self) -> Result<usize, GateError> {
            match self.kernel {
                // The loader's q8_0 card format: k one-byte codes plus an f16
                // scale per 32 values.
                Kernel::Q8 | Kernel::Q8Heads => resident_size(GgmlType::Q8_0, self.k, 1)
                    .ok_or_else(|| {
                        format!(
                            "bench_v41: site {}: k = {} has no q8_0 device layout",
                            self.label(),
                            self.k
                        )
                        .into()
                    }),
                Kernel::F32 => Ok(4 * self.k),
                _ => Ok(self.file_row_bytes()),
            }
        }

        /// Bytes of one weight row in the file.
        fn file_row_bytes(&self) -> usize {
            match self.kernel {
                Kernel::Q8 | Kernel::Q8Heads => self.k / 32 * 34,
                Kernel::F32 => 2 * self.k,
                Kernel::Q3k | Kernel::Q3kSel => self.k / 256 * 110,
                Kernel::Q4kSel => self.k / 256 * 144,
                Kernel::Q6k => self.k / 256 * 210,
            }
        }

        /// Weight bytes one launch reads (the whole device copy).
        fn w_bytes(&self) -> Result<u64, GateError> {
            Ok((self.out_len() * self.device_row_bytes()?) as u64)
        }

        /// File bytes of the weight rows one launch reads.
        fn launch_file_bytes(&self) -> u64 {
            (self.out_len() * self.file_row_bytes()) as u64
        }

        /// Bytes of the activation one launch reads, in the device format.
        fn x_bytes(&self) -> u64 {
            let (k, m) = (self.k, self.launch_cols());
            let n_sb = k / 256;
            let per_col = match self.kernel {
                Kernel::Q8 | Kernel::Q8Heads | Kernel::F32 => 4 * k,
                Kernel::Q3k | Kernel::Q3kSel => 8 * 64 * n_sb.div_ceil(2) + 4 * 2 * n_sb,
                Kernel::Q4kSel => 4 * 256 * n_sb.div_ceil(4) + 4 * 8 * n_sb + 4 * 2 * n_sb,
                Kernel::Q6k => 4 * 128 * n_sb.div_ceil(2) + 4 * 2 * n_sb,
            };
            (m * per_col) as u64
        }

        /// What one launch addresses — weights, activation and outputs, the
        /// byte convention of `gate_p8 --bench-kernels`.
        fn launch_bytes(&self) -> Result<u64, GateError> {
            Ok(self.w_bytes()? + self.x_bytes() + 4 * self.out_len() as u64)
        }

        /// File bytes of one tensor of this site: a routed stack holds all
        /// the file's experts, a block-diagonal weight all its groups.
        fn tensor_file_bytes(&self) -> u64 {
            let rows = if self.kernel.is_sel() {
                N_EXPERTS_FILE * self.rows
            } else {
                self.groups * self.rows
            };
            (rows * self.file_row_bytes()) as u64
        }

        fn tensors(&self) -> usize {
            self.layers.list().len()
        }

        /// Launches per token, and the copies a token reads.
        fn launches(&self) -> usize {
            self.tensors() * self.groups
        }

        /// Copies on the device: the real ones, then padding up to
        /// `CYCLE_FLOOR` for the site's own timing.
        fn copies(&self) -> Result<usize, GateError> {
            let floor = CYCLE_FLOOR.div_ceil(self.w_bytes()?) as usize;
            Ok(self.launches().max(floor))
        }

        /// The row geometry the launcher and the upload need: K a multiple of
        /// the f32 kernels' 32-value chunk, or of the K-quant super-block with
        /// whole u32 words per row; the rows split evenly over the heads.
        fn layout_ok(&self) -> bool {
            let rows = self.rows.is_multiple_of(self.heads);
            match self.kernel {
                Kernel::Q8 | Kernel::Q8Heads | Kernel::F32 => rows && self.k.is_multiple_of(32),
                _ => rows && self.k.is_multiple_of(256) && self.file_row_bytes().is_multiple_of(4),
            }
        }

        /// Blocks of 256 threads, one warp per output row.
        fn grid(&self) -> usize {
            self.out_len().div_ceil(8)
        }

        /// Device bytes the site allocates.
        fn vram_bytes(&self) -> Result<u64, GateError> {
            let x = if matches!(self.kernel, Kernel::Q8 | Kernel::Q8Heads | Kernel::F32) {
                (4 * self.x_cols() * self.k) as u64
            } else {
                self.x_bytes() * (self.x_cols() / self.launch_cols()) as u64
            };
            Ok(self.copies()? as u64 * self.w_bytes()?
                + x
                + (4 * self.groups * self.out_len()) as u64
                + 4 * N_SLOTS as u64)
        }

        /// The weight row and activation column output `o` of a launch reads.
        fn source_of(&self, o: usize) -> (usize, usize) {
            match self.kernel {
                Kernel::Q3kSel => (SEL[o / self.rows] as usize * self.rows + o % self.rows, 0),
                Kernel::Q4kSel => (
                    SEL[o / self.rows] as usize * self.rows + o % self.rows,
                    o / self.rows,
                ),
                Kernel::Q8Heads => (o, o / (self.rows / self.heads)),
                _ => (o, 0),
            }
        }
    }

    /// A token graph's launch plan.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Plan {
        /// The grouped `attn_output_a`, as the engine issues it today.
        Today,
        /// The dense-equivalent `attn_output_a` in its place.
        Batched,
        /// The `q8_0_gemv_heads` launch in its place.
        Heads,
    }

    impl Plan {
        /// Every plan, in the order `--time` captures them.
        const ALL: [Plan; 3] = [Plan::Today, Plan::Batched, Plan::Heads];

        fn name(self) -> &'static str {
            match self {
                Plan::Today => "today",
                Plan::Batched => "wo_a_dense",
                Plan::Heads => "wo_a_heads",
            }
        }

        fn takes(self, site: &Site) -> bool {
            match site.arm {
                Arm::Token => true,
                Arm::BlockDiag => self == Plan::Today,
                Arm::DenseEquiv => self == Plan::Batched,
                Arm::Heads => self == Plan::Heads,
            }
        }

        /// (site, copy) pairs in decode order: block by block through the
        /// table, then the head.
        fn launches(self) -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            for pos in 0..=N_LAYERS {
                for (i, s) in SITES.iter().enumerate() {
                    if !self.takes(s) {
                        continue;
                    }
                    if let Some(p) = s.layers.list().iter().position(|&l| l == pos) {
                        out.extend((0..s.groups).map(|g| (i, p * s.groups + g)));
                    }
                }
            }
            out
        }
    }

    /// One site's weight copy on the device.
    enum Weight {
        Q8 {
            qs: DeviceTensor<u32>,
            d: DeviceTensor<u16>,
        },
        F32(DeviceTensor<f32>),
        Words(DeviceTensor<u32>),
    }

    /// The activation a site's launches read.
    enum Input {
        /// f32 columns, one buffer per diagonal block — for the heads kernel
        /// one buffer holding every head's window.
        F32(Vec<DeviceBuffer<f32>>),
        /// q8_1 scratch, quantized once at load.
        Q8(Q8Act),
    }

    /// One checked copy: the output rows compared and their f64 reference.
    struct Ref {
        copy: usize,
        rows: Vec<usize>,
        want: Vec<f32>,
    }

    /// A site on the device.
    struct Dev {
        w: Vec<Weight>,
        x: Input,
        sel: Option<DeviceBuffer<u32>>,
        /// One output buffer per diagonal block.
        y: Vec<DeviceBuffer<f32>>,
        refs: Vec<Ref>,
    }

    /// A weight copy on the host, as generated.
    enum Host {
        Q8 {
            qs: Vec<u32>,
            /// f16 bits, as the device plane holds them.
            d: Vec<u16>,
        },
        F32(Vec<f32>),
        /// ggml block bytes, rows back to back.
        Blocks(Vec<u8>),
    }

    /// xorshift64* over a splitmix64-scrambled seed: nearby seeds give
    /// unrelated streams and the state is never zero.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            Rng((z ^ (z >> 31)) | 1)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// A positive normal f16 with exponent field 1..=9 and a random
        /// mantissa: finite, never zero, never subnormal.
        fn scale_f16(&mut self) -> u16 {
            let r = self.next_u64();
            ((1 + (r % 9) as u16) << 10) | ((r >> 32) as u16 & 0x3ff)
        }

        fn fill_bytes(&mut self, out: &mut [u8]) {
            for c in out.chunks_mut(8) {
                let b = self.next_u64().to_le_bytes();
                c.copy_from_slice(&b[..c.len()]);
            }
        }

        fn fill_words(&mut self, out: &mut [u32]) {
            for c in out.chunks_mut(2) {
                let r = self.next_u64();
                c[0] = r as u32;
                if let Some(w) = c.get_mut(1) {
                    *w = (r >> 32) as u32;
                }
            }
        }

        /// A value in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }

    /// Byte offsets of the f16 scale fields in one ggml block, and the block size.
    fn f16_fields(ty: GgmlType) -> Result<(&'static [usize], usize), GateError> {
        match ty {
            GgmlType::Q3_K => Ok((&[108], 110)),
            GgmlType::Q4_K => Ok((&[0, 2], 144)),
            GgmlType::Q6_K => Ok((&[208], 210)),
            other => Err(format!("bench_v41: no synthetic layout for {other}").into()),
        }
    }

    /// The seed of copy `copy` of the site whose data stream is `idx`.
    fn seed(idx: usize, copy: usize) -> u64 {
        ((idx as u64) << 32) | copy as u64
    }

    /// The data stream of site `i` of the table: the heads arm draws the
    /// dense-equivalent arm's, so the two read the same bytes through their
    /// two kernels; every other site draws its own, numbered in table order
    /// with the heads arm left out.
    fn stream_of(i: usize) -> Result<usize, GateError> {
        match SITES[i].arm {
            Arm::Heads => SITES
                .iter()
                .position(|s| s.arm == Arm::DenseEquiv)
                .ok_or_else(|| "bench_v41: the heads arm has no dense-equivalent arm".into()),
            _ => Ok(SITES[..i].iter().filter(|s| s.arm != Arm::Heads).count()),
        }
    }

    /// Copy `copy` of `site` on the host.
    fn synth(site: &Site, idx: usize, copy: usize) -> Result<Host, GateError> {
        let mut rng = Rng::new(seed(idx, copy));
        let rows = site.out_len();
        Ok(match site.kernel {
            Kernel::Q8 | Kernel::Q8Heads => {
                let mut qs = vec![0u32; rows * site.k / 4];
                rng.fill_words(&mut qs);
                let d = (0..rows * site.k / 32).map(|_| rng.scale_f16()).collect();
                Host::Q8 { qs, d }
            }
            Kernel::F32 => Host::F32((0..rows * site.k).map(|_| rng.unit()).collect()),
            kernel => {
                let ty = kernel
                    .ggml()
                    .ok_or("bench_v41: K-quant site without a ggml type")?;
                let (fields, block) = f16_fields(ty)?;
                let mut b = vec![0u8; rows * site.file_row_bytes()];
                rng.fill_bytes(&mut b);
                for blk in b.chunks_exact_mut(block) {
                    for &off in fields {
                        blk[off..off + 2].copy_from_slice(&rng.scale_f16().to_le_bytes());
                    }
                }
                Host::Blocks(b)
            }
        })
    }

    /// Upload a host copy into the device types the site's launcher takes.
    fn upload(gpu: &Gpu, site: &Site, host: &Host) -> Result<Weight, GateError> {
        let s = gpu.stream();
        let rows = site.out_len();
        Ok(match host {
            Host::Q8 { qs, d } => Weight::Q8 {
                qs: DeviceTensor::upload(s, qs, rows, site.k / 4)?,
                d: DeviceTensor::upload(s, d, rows, site.k / 32)?,
            },
            Host::F32(w) => Weight::F32(DeviceTensor::upload(s, w, rows, site.k)?),
            Host::Blocks(b) => {
                let words = bytes_to_words(b);
                let per_row = site.file_row_bytes() / 4;
                Weight::Words(DeviceTensor::upload(s, &words, rows, per_row)?)
            }
        })
    }

    /// f64 reference for output `o` of a launch on `host`, whose activation
    /// columns (f32, or q8_1-reconstructed for the K-quants) are `x`; `g` is
    /// the launch's diagonal block.
    fn reference(
        site: &Site,
        host: &Host,
        x: &[f32],
        o: usize,
        g: usize,
    ) -> Result<f32, GateError> {
        let k = site.k;
        let (row, col) = site.source_of(o);
        let xs = &x[(g + col) * k..(g + col + 1) * k];
        let dot: f64 = match host {
            Host::Q8 { qs, d } => (0..k)
                .map(|j| {
                    let code = (qs[row * k / 4 + j / 4] >> (8 * (j % 4))) as u8 as i8;
                    let scale = half_to_f32(d[row * k / 32 + j / 32]);
                    f64::from(code) * f64::from(scale) * f64::from(xs[j])
                })
                .sum(),
            Host::F32(w) => w[row * k..(row + 1) * k]
                .iter()
                .zip(xs)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum(),
            Host::Blocks(b) => {
                let ty = site
                    .kernel
                    .ggml()
                    .ok_or("bench_v41: blocks without a ggml type")?;
                let rb = site.file_row_bytes();
                let mut vals = vec![0.0f32; k];
                dequant_row(ty, &b[row * rb..(row + 1) * rb], &mut vals)?;
                vals.iter()
                    .zip(xs)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum()
            }
        };
        Ok(dot as f32)
    }

    /// The output rows a check compares: both ends, and rows a third and
    /// two thirds in — which for a `_sel` launch fall in different slots —
    /// and the first row of each of `heads` heads, so a heads launch is
    /// checked in every activation window.
    fn check_rows(n: usize, heads: usize) -> Vec<usize> {
        let mut v = vec![0, 1, n / 3, n / 2, 2 * n / 3 + 1, n - 1];
        v.extend((0..heads).map(|h| h * (n / heads)));
        v.retain(|&o| o < n);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// FNV-1a 64 over the little-endian bytes of the f32 bits of `y` — the
    /// digest `gate_p1` pins, here over a launch's whole output.
    fn fnv1a64(y: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in y {
            for b in v.to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Generate, reference and upload every copy of `site` from data stream
    /// `idx`, and its activation, outputs and slot ids.
    fn load_site(gpu: &Gpu, idx: usize, site: &Site) -> Result<Dev, GateError> {
        let stream = gpu.stream();
        let k = site.k;
        let x_host = activations(k, site.x_cols(), 7001 + idx as u32);
        let (x, x_ref) = match site.kernel {
            Kernel::Q8 | Kernel::F32 => {
                let bufs = x_host
                    .chunks_exact(k)
                    .map(|c| DeviceBuffer::from_host(stream, c))
                    .collect::<Result<Vec<_>, _>>()?;
                (Input::F32(bufs), x_host)
            }
            // One buffer holding every head's window back to back, which the
            // launch reads at a stride of `k`.
            Kernel::Q8Heads => (
                Input::F32(vec![DeviceBuffer::from_host(stream, &x_host)?]),
                x_host,
            ),
            _ => {
                let x_dev = DeviceBuffer::from_host(stream, &x_host)?;
                let mut act = Q8Act::with_k(stream, site.x_cols(), k)?;
                gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
                stream.synchronize()?;
                let xq = q8_1_dequant(&x_host, k, site.x_cols());
                (Input::Q8(act), xq)
            }
        };
        let real = site.launches();
        let checked = [0, real - 1];
        let copies = site.copies()?;
        let mut w = Vec::with_capacity(copies);
        let mut refs: Vec<Ref> = Vec::new();
        for copy in 0..copies {
            let host = synth(site, idx, copy)?;
            if checked.contains(&copy) && !refs.iter().any(|r| r.copy == copy) {
                let rows = check_rows(site.out_len(), site.heads);
                let g = copy % site.groups;
                let want = rows
                    .iter()
                    .map(|&o| reference(site, &host, &x_ref, o, g))
                    .collect::<Result<Vec<_>, _>>()?;
                refs.push(Ref { copy, rows, want });
            }
            w.push(upload(gpu, site, &host)?);
        }
        let y = (0..site.groups)
            .map(|_| DeviceBuffer::<f32>::zeroed(stream, site.out_len()))
            .collect::<Result<Vec<_>, _>>()?;
        let sel = if site.kernel.is_sel() {
            Some(DeviceBuffer::from_host(stream, &SEL)?)
        } else {
            None
        };
        Ok(Dev { w, x, sel, y, refs })
    }

    /// Enqueue one launch of `site` on copy `copy`, through the engine's own
    /// launcher. Asynchronous and capturable.
    fn launch(
        gpu: &Gpu,
        step: &StepKernels,
        site: &Site,
        dev: &mut Dev,
        copy: usize,
    ) -> Result<(), GpuError> {
        let s = gpu.stream();
        let g = copy % site.groups;
        let Dev { w, x, sel, y, .. } = dev;
        let y = &mut y[g];
        match (site.kernel, &w[copy], &*x, sel.as_ref()) {
            (Kernel::Q8, Weight::Q8 { qs, d }, Input::F32(x), _) => {
                gpu.q8f32().enqueue_q8_0_gemv(s, qs, d, &x[g], 1, y)
            }
            (Kernel::Q8Heads, Weight::Q8 { qs, d }, Input::F32(x), _) => step
                .enqueue_q8_0_gemv_heads(
                    s,
                    Q8_0GemvHeadsArgs {
                        qs,
                        d,
                        x: &x[0],
                        rows_per_head: site.rows / site.heads,
                        x_head_stride: site.k,
                        y_head_stride: site.rows / site.heads,
                        y_off: 0,
                        y,
                    },
                ),
            (Kernel::F32, Weight::F32(w), Input::F32(x), _) => {
                gpu.q8f32().enqueue_f32_gemv(s, w, &x[g], 1, y)
            }
            (Kernel::Q3k, Weight::Words(w), Input::Q8(a), _) => gpu.enqueue_gemv_q3k(w, a, y),
            (Kernel::Q6k, Weight::Words(w), Input::Q8(a), _) => gpu.enqueue_gemv_q6k(w, a, y),
            (Kernel::Q3kSel, Weight::Words(w), Input::Q8(a), Some(sel)) => {
                gpu.enqueue_gemv_q3k_sel(w, a, sel, N_SLOTS, site.rows, y)
            }
            (Kernel::Q4kSel, Weight::Words(w), Input::Q8(a), Some(sel)) => gpu
                .q4k_sel()
                .enqueue_gemv_q4k_sel(s, w, a, sel, N_SLOTS, site.rows, y),
            _ => Err(GpuError::Shape {
                what: "bench_v41::launch",
                detail: format!(
                    "{} holds data its kernel {} does not take",
                    site.label(),
                    site.kernel.name()
                ),
            }),
        }
    }

    /// One launch per checked copy, the compared rows against their
    /// reference; prints the site's check line, then one digest line per
    /// checked copy over the launch's whole output — the rows the check
    /// does not compare included — so a round that re-shapes a kernel can
    /// show every output bit unchanged by diffing these lines.
    fn check_site(
        gpu: &Gpu,
        step: &StepKernels,
        site: &Site,
        dev: &mut Dev,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut rels = Vec::new();
        let mut outs = Vec::new();
        let mut errors = Vec::new();
        let mut digests = Vec::new();
        let checked: Vec<usize> = dev.refs.iter().map(|r| r.copy).collect();
        for (i, &copy) in checked.iter().enumerate() {
            launch(gpu, step, site, dev, copy)?;
            stream.synchronize()?;
            let y = dev.y[copy % site.groups].to_host_vec(stream)?;
            digests.push((copy, y.len(), fnv1a64(&y)));
            let r = &dev.refs[i];
            let got: Vec<f32> = r.rows.iter().map(|&o| y[o]).collect();
            match max_rel_err(&got, &r.want) {
                Ok(e) => rels.push(e),
                Err(e) => {
                    rels.push(f32::INFINITY);
                    errors.push(format!("copy {copy}: {e}"));
                }
            }
            outs.push(got);
        }
        let distinct = match (outs.first(), outs.last()) {
            (Some(a), Some(b)) if outs.len() > 1 => Some(!bits_equal(a, b)),
            _ => None,
        };
        let pass =
            errors.is_empty() && rels.iter().all(|&e| e <= KERNEL_BAND) && distinct != Some(false);
        let copies: Vec<String> = checked.iter().map(usize::to_string).collect();
        let rels: Vec<String> = rels.iter().map(|e| format!("{e:.3e}")).collect();
        println!(
            "check shape={} kernel={} out_rows={} k={} cols={} copies_checked={} rows_checked={} \
             max_rel_err={} band={KERNEL_BAND:e} copies_distinct={} {}{}",
            site.label(),
            site.kernel.name(),
            site.out_len(),
            site.k,
            site.launch_cols(),
            copies.join(","),
            dev.refs.first().map_or(0, |r| r.rows.len()),
            rels.join(","),
            distinct.map_or("n/a".to_string(), |d| d.to_string()),
            verdict(pass),
            if errors.is_empty() {
                String::new()
            } else {
                format!(" errors=[{}]", errors.join("; "))
            },
        );
        for (copy, rows, digest) in digests {
            println!(
                "digest shape={} kernel={} copy={copy} out_rows={rows} fnv1a64={digest:#018x}",
                site.label(),
                site.kernel.name()
            );
        }
        Ok(pass)
    }

    /// Microseconds per launch: min, mean and max over the rounds.
    #[derive(Clone, Copy)]
    struct Spread {
        min: f64,
        mean: f64,
        max: f64,
    }

    impl Spread {
        fn of(us: &[f64]) -> Spread {
            Spread {
                min: us.iter().copied().fold(f64::INFINITY, f64::min),
                mean: us.iter().sum::<f64>() / us.len() as f64,
                max: us.iter().copied().fold(0.0, f64::max),
            }
        }
    }

    /// `n` launches cycling the site's copies, one synchronize at the end,
    /// `ROUNDS` times after a warm burst.
    fn time_eager(
        gpu: &Gpu,
        step: &StepKernels,
        site: &Site,
        dev: &mut Dev,
        n: usize,
    ) -> Result<Spread, GateError> {
        let stream = gpu.stream();
        let copies = dev.w.len();
        for c in 0..n {
            launch(gpu, step, site, dev, c % copies)?;
        }
        stream.synchronize()?;
        let mut us = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for c in 0..n {
                launch(gpu, step, site, dev, c % copies)?;
            }
            stream.synchronize()?;
            us.push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        Ok(Spread::of(&us))
    }

    /// `reps` replays of `g` per round, one synchronize per round, `ROUNDS`
    /// rounds after two warm replays; microseconds per replay divided by
    /// `per_replay`.
    fn time_replay(
        stream: &CudaStream,
        g: &Graph,
        per_replay: usize,
        reps: usize,
    ) -> Result<Spread, GateError> {
        for _ in 0..2 {
            g.launch(stream)?;
        }
        stream.synchronize()?;
        let mut us = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..reps {
                g.launch(stream)?;
            }
            stream.synchronize()?;
            us.push(t0.elapsed().as_secs_f64() * 1e6 / (per_replay * reps) as f64);
        }
        Ok(Spread::of(&us))
    }

    /// Eager burst and graph replay of one site; prints its bench line and
    /// returns the replay spread.
    fn time_site(
        gpu: &Gpu,
        step: &StepKernels,
        site: &Site,
        dev: &mut Dev,
    ) -> Result<Spread, GateError> {
        let copies = dev.w.len();
        let n = copies * MIN_BURST.div_ceil(copies);
        let eager = time_eager(gpu, step, site, dev, n)?;
        let graph =
            gpu.capture(|_| (0..n).try_for_each(|c| launch(gpu, step, site, dev, c % copies)))?;
        let nodes = graph.node_count();
        if nodes != n {
            return Err(format!(
                "bench_v41: {} graph has {nodes} nodes for {n} launches",
                site.label()
            )
            .into());
        }
        let rep = time_replay(gpu.stream(), &graph, n, GREPS)?;
        let bytes = site.launch_bytes()?;
        println!(
            "bench shape={} kernel={} out_rows={} k={} cols={} grid={} copies={} real_copies={} \
             n_per_burst={n} nodes={nodes} w_bytes={} bytes={bytes} \
             eager_us_min={:.3} eager_us_mean={:.3} eager_us_max={:.3} \
             graph_us_min={:.3} graph_us_mean={:.3} graph_us_max={:.3} graph_gbps={:.1} \
             launches_per_token={} token_graph_us={:.3}",
            site.label(),
            site.kernel.name(),
            site.out_len(),
            site.k,
            site.launch_cols(),
            site.grid(),
            copies,
            site.launches(),
            site.w_bytes()?,
            eager.min,
            eager.mean,
            eager.max,
            rep.min,
            rep.mean,
            rep.max,
            bytes as f64 / rep.min / 1e3,
            site.launches(),
            rep.min * site.launches() as f64,
        );
        Ok(rep)
    }

    /// Capture and replay one token graph; prints its line next to the sum
    /// of its sites' own replay times and a `touch` graph of the same node
    /// count.
    fn time_token(
        gpu: &Gpu,
        step: &StepKernels,
        plan: Plan,
        devs: &mut [Dev],
        site_us: &[f64],
        touch: &Probe,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let launches = plan.launches();
        let graph = gpu.capture(|_| {
            launches
                .iter()
                .try_for_each(|&(i, c)| launch(gpu, step, &SITES[i], &mut devs[i], c))
        })?;
        let nodes = graph.node_count();
        if nodes != launches.len() {
            return Err(format!(
                "bench_v41: token graph {} has {nodes} nodes, the table issues {}",
                plan.name(),
                launches.len()
            )
            .into());
        }
        let rep = time_replay(stream, &graph, 1, TOKEN_REPS)?;
        let (mut w_bytes, mut bytes, mut sum_us) = (0u64, 0u64, 0.0f64);
        for &(i, _) in &launches {
            w_bytes += SITES[i].w_bytes()?;
            bytes += SITES[i].launch_bytes()?;
            sum_us += site_us[i];
        }
        let mut tbuf = DeviceBuffer::<f32>::zeroed(stream, 32)?;
        let tgraph =
            gpu.capture(|s| (0..nodes).try_for_each(|_| touch.enqueue_touch(s, &mut tbuf)))?;
        let t = time_replay(stream, &tgraph, 1, TOKEN_REPS)?;
        println!(
            "token plan={} nodes={nodes} w_bytes={w_bytes} bytes={bytes} \
             us_min={:.1} us_mean={:.1} us_max={:.1} gbps={:.1} sum_shape_graph_us={sum_us:.1} \
             token_minus_sum_us={:.1} touch_nodes={} touch_us_min={:.1} touch_us_mean={:.1} \
             touch_us_per_node={:.3}",
            plan.name(),
            rep.min,
            rep.mean,
            rep.max,
            bytes as f64 / rep.min / 1e3,
            rep.min - sum_us,
            tgraph.node_count(),
            t.min,
            t.mean,
            t.min / tgraph.node_count() as f64,
        );
        Ok(())
    }

    /// The lightning indexer's two launches (`ds41_indexer_score`, then
    /// `ds41_indexer_topk`) per emitting block, at three depths. Built only
    /// with the `deepseek41` feature, which links the V4.1 crate; a `gpu`
    /// build prints that they were skipped.
    ///
    /// Each site is one token over `n_vis` synthetic index keys (the rows the
    /// score pass reads), the file's top-k and rope width, a random query
    /// projection, weights and rope table. A site whose keys would stay in
    /// L2 across a burst gets distinct key copies up to [`CYCLE_FLOOR`]; the
    /// scratch and the list are shared, since the launches serialize.
    ///
    /// `--check`: the first and last copy each run once; the list must be
    /// the `top_k` rows of the launch's own scores with the largest
    /// order-preserving keys, ties to the lower row, in ascending order, the
    /// histogram zero again, and the two copies' scores distinct. It prints
    /// one digest line per copy for the scores and for the list. The scores'
    /// agreement with ik is `gate-gpu-ds41-index`'s, not this check's.
    ///
    /// `--time`: per site an eager burst and a graph replay cycling the key
    /// copies, microseconds per indexer step (two launches).
    #[cfg(feature = "deepseek41")]
    mod index {
        use super::{
            CYCLE_FLOOR, GREPS, INDEX_SOURCE, MIN_BURST, Rng, Spread, fnv1a64, time_replay,
        };
        use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
        use bloomery_gpu_deepseek41::indexer::{
            HEAD_DIM, HEADS, HIST_BINS, IndexerArgs, IndexerKernels, IndexerScratch,
        };
        use bloomery_gpu_gates::{GateError, verdict};
        use cuda_core::{CudaStream, DeviceBuffer};
        use gguf::quant::f32_to_f16_bits;
        use std::time::Instant;

        /// Visible rows of the three sites.
        const N_VIS: [usize; 3] = [1024, 4096, 32_768];
        /// The file's `indexer.top_k` and the query's roped values per head
        /// (`docs/v41-inventory.md`).
        const TOP_K: usize = 512;
        const ROPE_DIMS: usize = 64;
        /// Eager rounds per site.
        const ROUNDS: usize = super::ROUNDS;

        pub struct Site {
            n_vis: usize,
            keys: Vec<DeviceTensor<u16>>,
            q: DeviceBuffer<f32>,
            w: DeviceBuffer<f32>,
            ints: DeviceBuffer<u32>,
            tables: DeviceBuffer<u32>,
            scratch: IndexerScratch,
            list: DeviceBuffer<u32>,
        }

        impl Site {
            pub fn label(&self) -> String {
                format!("indexer_nvis{}", self.n_vis)
            }

            fn key_bytes(&self) -> u64 {
                (self.n_vis * HEAD_DIM * 2) as u64
            }

            fn enqueue(
                &mut self,
                kernels: &IndexerKernels,
                stream: &CudaStream,
                copy: usize,
            ) -> Result<(), GpuError> {
                kernels.enqueue(
                    stream,
                    IndexerArgs {
                        q: &self.q,
                        w: &self.w,
                        ints: &self.ints,
                        n_vis_at: 0,
                        top_k_at: 1,
                        tables: &self.tables,
                        rope_at: 0,
                        rope_stride: ROPE_DIMS,
                        keys: &self.keys[copy],
                        tokens: 1,
                        scratch: &mut self.scratch,
                        list: &mut self.list,
                        stride: TOP_K,
                    },
                )
            }
        }

        pub fn load(gpu: &Gpu) -> Result<(IndexerKernels, Vec<Site>), GateError> {
            let stream = gpu.stream();
            let kernels = IndexerKernels::with_shape(gpu.context(), HEADS, HEAD_DIM, ROPE_DIMS)?;
            let mut sites = Vec::with_capacity(N_VIS.len());
            for (i, &n_vis) in N_VIS.iter().enumerate() {
                let mut rng = Rng::new(0x1d0_0000 + i as u64);
                let key_bytes = (n_vis * HEAD_DIM * 2) as u64;
                let copies = usize::try_from(CYCLE_FLOOR.div_ceil(key_bytes).max(2))?;
                let keys = (0..copies)
                    .map(|_| {
                        let k: Vec<u16> = (0..n_vis * HEAD_DIM)
                            .map(|_| f32_to_f16_bits(rng.unit()))
                            .collect();
                        DeviceTensor::upload(stream, &k, n_vis, HEAD_DIM)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let q: Vec<f32> = (0..HEADS * HEAD_DIM).map(|_| 4.0 * rng.unit()).collect();
                let w: Vec<f32> = (0..HEADS).map(|_| 16.0 * rng.unit()).collect();
                let tables: Vec<u32> = (0..ROPE_DIMS / 2)
                    .flat_map(|_| {
                        let a = std::f32::consts::PI * rng.unit();
                        [a.cos().to_bits(), a.sin().to_bits()]
                    })
                    .collect();
                sites.push(Site {
                    n_vis,
                    keys,
                    q: DeviceBuffer::from_host(stream, &q)?,
                    w: DeviceBuffer::from_host(stream, &w)?,
                    ints: DeviceBuffer::from_host(stream, &[u32::try_from(n_vis)?, TOP_K as u32])?,
                    tables: DeviceBuffer::from_host(stream, &tables)?,
                    scratch: IndexerScratch::new(stream, 1, n_vis)?,
                    list: DeviceBuffer::zeroed(stream, TOP_K)?,
                });
            }
            Ok((kernels, sites))
        }

        /// The order-preserving key of a score: `-0` made `+0`, a
        /// negative's bits flipped whole, a non-negative's top bit set.
        fn key_of(v: f32) -> u32 {
            let b = v.to_bits();
            let b = if b == 0x8000_0000 { 0 } else { b };
            if b & 0x8000_0000 != 0 {
                !b
            } else {
                b | 0x8000_0000
            }
        }

        fn exact_top(scores: &[f32], k: usize) -> Vec<u32> {
            let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
            idx.sort_by(|&a, &b| {
                key_of(scores[b as usize])
                    .cmp(&key_of(scores[a as usize]))
                    .then(a.cmp(&b))
            });
            let mut top = idx[..k].to_vec();
            top.sort_unstable();
            top
        }

        pub fn check(
            gpu: &Gpu,
            kernels: &IndexerKernels,
            site: &mut Site,
        ) -> Result<bool, GateError> {
            let stream = gpu.stream();
            let copies = [0, site.keys.len() - 1];
            let mut exact = 0usize;
            let mut outs = Vec::with_capacity(2);
            for &c in &copies {
                site.enqueue(kernels, stream, c)?;
                stream.synchronize()?;
                let scores = site.scratch.scores.to_host_vec(stream)?;
                let list = site.list.to_host_vec(stream)?;
                let hist = site.scratch.hist.to_host_vec(stream)?;
                let ok = list == exact_top(&scores, TOP_K)
                    && hist.len() == HIST_BINS
                    && hist.iter().all(|&h| h == 0)
                    && scores.iter().all(|v| v.is_finite());
                exact += usize::from(ok);
                outs.push((c, scores, list));
            }
            let distinct = outs[0]
                .1
                .iter()
                .zip(&outs[1].1)
                .any(|(a, b)| a.to_bits() != b.to_bits());
            let pass = exact == copies.len() && distinct;
            println!(
                "check shape={} kernel=ds41_indexer_score+ds41_indexer_topk n_vis={} top_k={TOP_K} \
                 copies_checked={} lists_exact_top={exact}/{} copies_distinct={distinct} {}",
                site.label(),
                site.n_vis,
                copies.map(|c| c.to_string()).join(","),
                copies.len(),
                verdict(pass)
            );
            for (c, scores, list) in &outs {
                let list_f: Vec<f32> = list.iter().map(|&i| f32::from_bits(i)).collect();
                for (what, y) in [("scores", scores), ("list", &list_f)] {
                    println!(
                        "digest shape={} kernel=ds41_indexer_{what} copy={c} out_rows={} fnv1a64={:#018x}",
                        site.label(),
                        y.len(),
                        fnv1a64(y)
                    );
                }
            }
            Ok(pass)
        }

        /// Eager burst and graph replay of one site, microseconds per step.
        pub fn time(gpu: &Gpu, kernels: &IndexerKernels, site: &mut Site) -> Result<(), GateError> {
            let stream = gpu.stream();
            let copies = site.keys.len();
            let n = copies * MIN_BURST.div_ceil(copies);
            for c in 0..n {
                site.enqueue(kernels, stream, c % copies)?;
            }
            stream.synchronize()?;
            let mut us = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let t0 = Instant::now();
                for c in 0..n {
                    site.enqueue(kernels, stream, c % copies)?;
                }
                stream.synchronize()?;
                us.push(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            let eager = Spread::of(&us);
            let graph = gpu
                .capture(|_| (0..n).try_for_each(|c| site.enqueue(kernels, stream, c % copies)))?;
            let nodes = graph.node_count();
            if nodes != 2 * n {
                return Err(format!(
                    "bench_v41: {} graph has {nodes} nodes for {n} steps",
                    site.label()
                )
                .into());
            }
            let rep = time_replay(stream, &graph, n, GREPS)?;
            println!(
                "bench shape={} kernel=ds41_indexer_score+ds41_indexer_topk n_vis={} top_k={TOP_K} \
                 copies={copies} n_per_burst={n} nodes={nodes} key_bytes={} \
                 eager_us_min={:.3} eager_us_mean={:.3} eager_us_max={:.3} \
                 graph_us_min={:.3} graph_us_mean={:.3} graph_us_max={:.3} key_gbps={:.1} \
                 steps_per_token={} token_graph_us={:.3}",
                site.label(),
                site.n_vis,
                site.key_bytes(),
                eager.min,
                eager.mean,
                eager.max,
                rep.min,
                rep.mean,
                rep.max,
                site.key_bytes() as f64 / rep.min / 1e3,
                INDEX_SOURCE.len(),
                rep.min * INDEX_SOURCE.len() as f64,
            );
            Ok(())
        }
    }

    /// Validate the table against the inventory and the placement doc, and
    /// print it with its per-token totals.
    fn print_table() -> Result<(), GateError> {
        let mut bad = Vec::new();
        let (mut q8_tensors, mut q8_file, mut q8_dev) = (0usize, 0u64, 0u64);
        let (mut launches, mut w_bytes, mut file_bytes) = (0usize, 0u64, 0u64);
        let (mut exp_launches, mut exp_bytes) = (0usize, 0u64);
        let (mut tensors, mut tensor_bytes) = (0usize, 0u64);
        println!(
            "v41 table sites={} order=decode source=docs/v41-inventory.md,docs/research/v41-op-map.md",
            SITES.len()
        );
        for s in SITES {
            if s.tensor_file_bytes() != s.inventory_bytes || !s.layout_ok() {
                bad.push(s.label());
            }
            let site_w_bytes = s.w_bytes()?;
            let plans: Vec<&str> = Plan::ALL
                .iter()
                .filter(|p| p.takes(s))
                .map(|p| p.name())
                .collect();
            println!(
                "v41 site={} kernel={} type={} layers={} tensors={} launches_per_token={} \
                 out_rows={} k={} cols={} grid={} w_bytes={site_w_bytes} file_bytes={} \
                 tensor_file_bytes={} inventory_bytes={} copies={} in_token_graph={}",
                s.label(),
                s.kernel.name(),
                s.kernel.file_type(),
                s.layers.label(),
                s.tensors(),
                s.launches(),
                s.out_len(),
                s.k,
                s.launch_cols(),
                s.grid(),
                s.launch_file_bytes(),
                s.tensor_file_bytes(),
                s.inventory_bytes,
                s.copies()?,
                plans.join(","),
            );
            // The totals are today's token.
            if !Plan::Today.takes(s) {
                continue;
            }
            launches += s.launches();
            tensors += s.tensors();
            tensor_bytes += s.tensors() as u64 * s.tensor_file_bytes();
            w_bytes += s.launches() as u64 * site_w_bytes;
            file_bytes += s.launches() as u64 * s.launch_file_bytes();
            if s.kernel == Kernel::Q8 {
                q8_tensors += s.tensors();
                q8_file += s.launches() as u64 * s.launch_file_bytes();
                q8_dev += s.launches() as u64 * site_w_bytes;
            }
            if s.kernel.is_sel() {
                exp_launches += s.launches();
                exp_bytes += s.launches() as u64 * site_w_bytes;
            }
        }
        let gate_up = || SITES.iter().filter(|s| s.kernel == Kernel::Q3kSel);
        let gate_up_low_launches = gate_up().count() * EXPERT_LO;
        let gate_up_low = gate_up()
            .map(|s| Ok(EXPERT_LO as u64 * s.w_bytes()?))
            .sum::<Result<u64, GateError>>()?;
        println!(
            "v41 token launches={launches} tensors={tensors} tensor_file_bytes={tensor_bytes} \
             w_bytes={w_bytes} file_bytes={file_bytes} \
             dense_w_bytes={} expert_launches={exp_launches} expert_w_bytes={exp_bytes} \
             expert_layers={EXPERT_LO}..{N_LAYERS} n_slots={N_SLOTS} \
             gate_up_layers_0..{EXPERT_LO}_if_on_gpu_launches={gate_up_low_launches} \
             gate_up_layers_0..{EXPERT_LO}_if_on_gpu_w_bytes={gate_up_low}",
            w_bytes - exp_bytes,
        );
        let reconciled = q8_tensors == DOC_Q8_TENSORS
            && q8_file == DOC_Q8_FILE_BYTES
            && q8_dev == DOC_Q8_DEVICE_BYTES;
        println!(
            "v41 reconcile q8_0 tensors={q8_tensors} file_bytes={q8_file} device_bytes={q8_dev} \
             want={DOC_Q8_TENSORS},{DOC_Q8_FILE_BYTES},{DOC_Q8_DEVICE_BYTES} \
             source=docs/v41-placement.md§1 {}",
            verdict(reconciled)
        );
        println!(
            "v41 inventory rows_matching={}/{} source=docs/v41-inventory.md {}",
            SITES.len() - bad.len(),
            SITES.len(),
            verdict(bad.is_empty())
        );
        if !bad.is_empty() || !reconciled {
            return Err(format!(
                "bench_v41: the table disagrees with the docs (rows: [{}], q8_0 reconciled: {reconciled})",
                bad.join(", ")
            )
            .into());
        }
        Ok(())
    }

    /// Free and total device memory of the context's card.
    fn vram(gpu: &Gpu) -> Result<(u64, u64), GateError> {
        gpu.context().bind_to_thread()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: the context is current on this thread (bound above), and
        // both out-pointers are live locals the call only writes.
        let rc = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
        if rc != sys::cudaError_enum_CUDA_SUCCESS {
            return Err(DriverError(rc).into());
        }
        Ok((free as u64, total as u64))
    }

    enum Mode {
        Check,
        Time,
    }

    fn mode() -> Result<Mode, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["--check"] => Ok(Mode::Check),
            ["--time"] => Ok(Mode::Time),
            _ => {
                Err(format!("bench_v41: want exactly one of --check, --time; got {args:?}").into())
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let mode = mode()?;
        print_table()?;
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        println!(
            "v41 card={} sms={}",
            ctx.device_name()?,
            ctx.multiprocessor_count()?
        );
        println!(
            "v41 not_issued site=hc_attn_fn,hc_ffn_fn type=q3_K tensors={HC_TENSORS} \
             out_rows={HC_ROWS} k={HC_K} file_bytes={} served_by=ds41_hc_pre",
            HC_TENSORS * HC_ROWS * HC_K / 256 * 110
        );
        let plan = SITES
            .iter()
            .map(Site::vram_bytes)
            .sum::<Result<u64, GateError>>()?;
        let (free, total) = vram(&gpu)?;
        println!(
            "v41 vram plan_bytes={plan} free_bytes={free} total_bytes={total} margin_bytes={VRAM_MARGIN}"
        );
        if plan + VRAM_MARGIN > free {
            return Err(format!(
                "bench_v41: the card has {free} B free, the bench needs {plan} B plus a {VRAM_MARGIN} B margin"
            )
            .into());
        }
        let t0 = Instant::now();
        let mut devs = SITES
            .iter()
            .enumerate()
            .map(|(i, s)| load_site(&gpu, stream_of(i)?, s))
            .collect::<Result<Vec<_>, _>>()?;
        let (free_after, _) = vram(&gpu)?;
        println!(
            "v41 loaded sites={} copies={} used_bytes={} load_s={:.1}",
            SITES.len(),
            devs.iter().map(|d| d.w.len()).sum::<usize>(),
            free - free_after,
            t0.elapsed().as_secs_f64()
        );

        let step = StepKernels::load(ctx)?;
        let mut failed = Vec::new();
        for (s, d) in SITES.iter().zip(devs.iter_mut()) {
            if !check_site(&gpu, &step, s, d)? {
                failed.push(s.label());
            }
        }
        #[cfg(feature = "deepseek41")]
        let (ix_kernels, mut ix_sites) = index::load(&gpu)?;
        #[cfg(feature = "deepseek41")]
        for s in &mut ix_sites {
            if !index::check(&gpu, &ix_kernels, s)? {
                failed.push(s.label());
            }
        }
        #[cfg(not(feature = "deepseek41"))]
        println!("v41 not_built site=indexer reason=needs_feature_deepseek41");
        if !failed.is_empty() {
            eprintln!("FAIL: check failed for: {}", failed.join(", "));
            return Err(checks_failed());
        }
        println!(
            "PASSED: bench_v41 check — every site within {KERNEL_BAND:e} of its f64 reference on its first and last copy"
        );
        if let Mode::Check = mode {
            return Ok(());
        }

        println!(
            "bench min_burst={MIN_BURST} rounds={ROUNDS} graph_launches_per_round={GREPS} \
             token_replays_per_round={TOKEN_REPS} cycle_floor_bytes={CYCLE_FLOOR}"
        );
        let mut site_us = Vec::with_capacity(SITES.len());
        for (s, d) in SITES.iter().zip(devs.iter_mut()) {
            site_us.push(time_site(&gpu, &step, s, d)?.min);
        }
        #[cfg(feature = "deepseek41")]
        for s in &mut ix_sites {
            index::time(&gpu, &ix_kernels, s)?;
        }
        let touch = Probe::load(ctx)?;
        for plan in Plan::ALL {
            time_token(&gpu, &step, plan, &mut devs, &site_us, &touch)?;
        }
        Ok(())
    }
}
