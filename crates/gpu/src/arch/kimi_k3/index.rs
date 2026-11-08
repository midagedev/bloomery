//! The lightning indexer of a K3.1 file's latent layers on the card (llama.cpp
//! `src/models/kimi-k31.cpp`, `build_dsa_layer`): the layer attends the `top_k` cache rows its
//! indexer scores highest.
//!
//! A layer that runs the indexer keeps one f16 key of [`INDEX_DIM`] values for each position beside
//! its cached row ([`Index`]). The plan counts that row from the description's selector
//! (`model::arch::kimi_k3::place::KvLayout`), and the placed load holds its stores to that count
//! (`body::refuse_stores`), so [`width`] and the plan's reading are two readings of one fact.
//!
//! The step of such a layer, between the latent layer's front and its core
//! ([`crate::arch::gated_mla::front`], [`crate::arch::gated_mla::back`]), over common owners:
//! 1. `indexer.attn_k` of the attn-normed row, then the key's LayerNorm with `indexer.k_norm`'s
//!    gain and bias at `attention.layer_norm_epsilon`, no rope, f16 into the key row at the
//!    position (`index_key::enqueue_index_key_input`, `n_dims` 0);
//! 2. `indexer.proj` of the attn-normed row (the heads' weights) and `indexer.attn_q_b` of the
//!    normed query latent (the heads' queries);
//! 3. the score of every visible row, `Σ_h w_h · relu(q_h · k)` with the weights scaled by
//!    `1/√(heads · d)`, and the `min(pos + 1, top_k)` best rows in ascending row order, ties to the
//!    lower row (`indexer::IndexerKernels::enqueue`);
//! 4. the core over the list's rows ([`crate::latent_gated::attend_sel`]).
//!
//! The common kernels apply the 128-wide Hadamard transform to the key (step 1) and to the query
//! (step 3); the file's graph applies none. The transform is orthogonal, so `HT(q) · HT(k) = q · k`
//! in real arithmetic: it is a change of basis the common kernels share, rounding-neutral up to its
//! own roundings, not a term of the model.
//!
//! The counts are device words the step's refresh writes ([`Sel::words`]): the live rows
//! `pos + 1`, the list's width `top_k`, and the list's live length `min(pos + 1, top_k)`, which is
//! the gather's count and the attention's key count over the gathered rows. The current position is
//! in the list only when its score keeps it, as in the file's graph.
//!
//! A load whose context is at most `top_k` positions selects nothing ([`selects`]): every list it
//! could make is the identity, whose attention is the dense walk's, so its layers run the dense
//! walk and write no key row. A longer load runs the four steps at every step; a pass of rows
//! (`super::rows`, `super::pass`) runs the dense attention and writes its rows' keys, so a pass
//! whose last row would see more than `top_k` keys is refused by name ([`check_pass`]).

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::rows_kit::{RowsKit, project_rows};
use crate::index_key::{IndexKeyInputArgs, IndexKeyKernels};
use crate::indexer::{IndexerArgs, IndexerKernels, IndexerScratch};
use crate::latent_gated::Selection;
use crate::site::{self, SiteTy};
use crate::tensor::Window;
use crate::weights::Weights;
use crate::{DeviceTensor, FaultSink, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::kimi_k3::hparams::{Hparams, INDEX_DIM, INDEX_HEADS, Indexer};
use model::arch::kimi_k3::names;

/// What the refusals of this file name.
const WHAT: &str = "kimi_k3 indexer";

/// The types an indexer matrix runs: the file's BF16, which the load widens to f32, and Q8_0. Both
/// are the f32-row gemv of the common dense site.
const TYPES: &[SiteTy] = &[SiteTy::F32, SiteTy::Q8_0];

/// Where the step's count words sit ([`count_words`]): the live rows, the list's width, the list's
/// live length.
pub const N_VIS_AT: usize = 0;
pub const TOP_K_AT: usize = 1;
pub const N_SEL_AT: usize = 2;
pub const WORDS: usize = 3;

/// The count words of the step at `pos` over lists of `top_k` rows: `[pos + 1, top_k,
/// min(pos + 1, top_k)]`, at [`N_VIS_AT`], [`TOP_K_AT`] and [`N_SEL_AT`]. Refused by name at a
/// position of u32's last value.
///
/// # Errors
/// [`GpuError::Shape`] when `pos + 1` passes u32.
pub fn count_words(pos: u32, top_k: u32) -> Result<[u32; WORDS], GpuError> {
    let n_vis = pos
        .checked_add(1)
        .ok_or_else(|| GpuError::shape(WHAT, "a position at u32's last value"))?;
    let mut w = [0; WORDS];
    w[N_VIS_AT] = n_vis;
    w[TOP_K_AT] = top_k;
    w[N_SEL_AT] = n_vis.min(top_k);
    Ok(w)
}

/// The values of the index key row layer `l` keeps for each position: the indexer's key width on a
/// layer that runs the indexer ([`Hparams::dsa`]), `None` on a delta layer and on every layer of a
/// K3 file.
pub(super) fn width(hp: &Hparams, l: usize) -> Option<usize> {
    hp.indexer.as_ref().filter(|ix| ix.layers[l]).map(|ix| ix.d)
}

/// Whether a load of `hp`'s file at `ctx` positions selects: a K3.1 file at a context past its
/// `top_k` (module doc).
pub(super) fn selects(hp: &Hparams, ctx: usize) -> bool {
    hp.indexer.as_ref().is_some_and(|ix| ctx > ix.top_k)
}

/// A selection of lists `stride` rows wide, refused by name on a file with no indexer and at a
/// width other than the indexer's `top_k`.
///
/// # Errors
/// [`GpuError::Shape`] naming the refusal.
pub fn check_selection(ix: Option<&Indexer>, stride: usize) -> Result<usize, GpuError> {
    let ix = ix.ok_or_else(|| {
        GpuError::shape(
            WHAT,
            "a selection on a file with no indexer: only a K3.1 latent layer selects its rows",
        )
    })?;
    if stride != ix.top_k {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a list of {stride} rows; the indexer's top_k picks {}",
                ix.top_k
            ),
        ));
    }
    Ok(stride)
}

/// A pass of rows whose last row sees `keys` keys, refused by name on a file whose indexer picks
/// fewer: the pass attends every key, which past `top_k` is not the file's attention. Names the
/// first layer that runs the indexer, the count and `top_k`.
///
/// # Errors
/// [`GpuError::Shape`] naming the layer, the count and `top_k`.
pub fn check_pass(ix: Option<&Indexer>, keys: u64) -> Result<(), GpuError> {
    let Some(ix) = ix else {
        return Ok(());
    };
    if ix.pass_attends_as_the_file(keys) {
        return Ok(());
    }
    let layer = ix.first_layer();
    Err(GpuError::shape(
        WHAT,
        format!(
            "latent layer {layer:?}: a pass whose last row sees {keys} keys; the indexer picks \
             top_k {} of them and a pass attends every key: step the positions past {}",
            ix.top_k, ix.top_k
        ),
    ))
}

/// A K3.1 latent layer's indexer sites and the names of its key norm's gain and bias.
struct Sites {
    /// `indexer.attn_q_b`: `heads · d` rows over the query latent.
    q_b: Site,
    /// `indexer.attn_k`: `d` rows over the layer's input.
    k: Site,
    /// `indexer.proj`: `heads` rows over the layer's input.
    proj: Site,
    /// `indexer.k_norm.weight`, `indexer.k_norm.bias`.
    norm: [String; 2],
}

impl Sites {
    /// Layer `l`'s sites, each a resident weight of a type [`TYPES`] lists at the header's widths.
    /// Load-time only.
    fn load(w: &Weights, hp: &Hparams, ix: &Indexer, l: usize) -> Result<Sites, GpuError> {
        if (ix.heads, ix.d) != (INDEX_HEADS, INDEX_DIM) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} heads of {}; the score kernel runs {INDEX_HEADS} of {INDEX_DIM}",
                    ix.heads, ix.d
                ),
            ));
        }
        let site = |name: String, (rows, k): (usize, usize)| -> Result<Site, GpuError> {
            let ty = site::resident_site(w, WHAT, &name, (rows, k), TYPES)?;
            Ok(Site::new(name, ty, k))
        };
        let (embd, q_lora) = (hp.n_embd, hp.mla.q_lora);
        Ok(Sites {
            q_b: site(names::indexer_attn_q_b(l), (ix.heads * ix.d, q_lora))?,
            k: site(names::indexer_attn_k(l), (ix.d, embd))?,
            proj: site(names::indexer_proj(l), (ix.heads, embd))?,
            norm: [
                site::vector(w, WHAT, names::indexer_k_norm(l), ix.d)?,
                site::vector(w, WHAT, names::indexer_k_norm_bias(l), ix.d)?,
            ],
        })
    }
}

/// A K3.1 latent layer's indexer: its sites, its key norm's ε and its key rows, `[ctx][d]` f16.
pub(super) struct Index {
    sites: Sites,
    eps: f32,
    keys: DeviceTensor<u16>,
}

impl Index {
    /// Layer `l`'s indexer at `ctx` positions, `None` where the layer runs none ([`width`]).
    /// Load-time only.
    pub(super) fn load(
        stream: &CudaStream,
        w: &Weights,
        hp: &Hparams,
        (l, ctx): (usize, usize),
    ) -> Result<Option<Box<Index>>, GpuError> {
        let (Some(d), Some(ix)) = (width(hp, l), hp.indexer.as_ref()) else {
            return Ok(None);
        };
        Ok(Some(Box::new(Index {
            sites: Sites::load(w, hp, ix, l)?,
            eps: ix.norm_eps,
            keys: DeviceTensor::zeroed(stream, ctx, d)?,
        })))
    }

    /// Every site the indexer launches, for the load's activation forms.
    pub(super) fn sites(&self) -> [&Site; 3] {
        [&self.sites.q_b, &self.sites.k, &self.sites.proj]
    }

    /// Device bytes of the key rows: the plan's store.
    pub(super) fn bytes(&self) -> usize {
        self.keys.buf().num_bytes()
    }

    /// Step 1 of the module doc for `m` rows: `x` the projections `[m][d]` of the rows'
    /// attn-normed inputs, `pos` their positions.
    fn write_keys(
        &mut self,
        (gpu, w): (&Gpu, &Weights),
        k: &Kernels,
        (l, m): (usize, usize),
        x: &DeviceBuffer<f32>,
        pos: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let [gain, bias] = &self.sites.norm;
        enqueue_keys(
            gpu.stream(),
            k,
            (w.f32_buf(WHAT, gain)?, w.f32_buf(WHAT, bias)?, self.eps),
            (x, pos, m),
            gpu.layer_sink(l)?,
            &mut self.keys,
        )
    }

    /// Step 1 for a pass's `m` rows after the layer's dense rows
    /// ([`crate::arch::gated_mla::rows`]): `xn` the rows' attn-normed inputs, `pos` their positions,
    /// and `proj` the buffer their projections land in (the layer's query latent, dead once the
    /// attention ran).
    #[allow(
        clippy::too_many_arguments,
        reason = "the launch context, the kernels, the forms and stage, the layer and rows, and the \
                  three buffers, each named as the rows pass names it (rust-quality R8)"
    )]
    pub(super) fn keys_rows(
        &mut self,
        cx: &GqaCx<'_>,
        k: &Kernels,
        (acts, kit): (&mut Acts, RowsKit<'_>),
        (l, m): (usize, usize),
        xn: &DeviceBuffer<f32>,
        pos: &DeviceBuffer<u32>,
        proj: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = self.keys.cols();
        if proj.len() < m * d {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: {m} rows' key projections of {d} in a buffer of {}",
                    proj.len()
                ),
            ));
        }
        let mut kit = kit;
        project_rows(cx, &mut kit, acts, (l, m), (&self.sites.k, d), xn, proj)?;
        self.write_keys((cx.gpu, cx.w), k, (l, m), proj, pos)
    }

    /// Steps 1 to 3 of the module doc at one row: `xn` the layer's attn-normed input, `qr` its
    /// normed query latent, `pos` its position; the list into `sel` for [`Sel::selection`].
    pub(super) fn pick(
        &mut self,
        cx: &GqaCx<'_>,
        k: &Kernels,
        acts: &mut Acts,
        l: usize,
        (xn, qr, pos): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>),
        sel: &mut Sel,
    ) -> Result<(), GpuError> {
        let Sites {
            q_b, k: key, proj, ..
        } = &self.sites;
        key.project(cx, acts, l, xn, &mut sel.ik)?;
        proj.project(cx, acts, l, xn, &mut sel.iw)?;
        q_b.project(cx, acts, l, qr, &mut sel.iq)?;
        self.write_keys((cx.gpu, cx.w), k, (l, 1), &sel.ik, pos)?;
        let Sel {
            words,
            iq,
            iw,
            list,
            scratch,
            ..
        } = sel;
        enqueue_pick(
            cx.gpu.stream(),
            k,
            &self.keys,
            (iq, iw),
            words,
            scratch,
            list,
        )
    }
}

/// Step 1 of the module doc after the projection, for `m` rows: `x` the key projections `[m][d]`,
/// `pos` the rows' positions, the key norm's gain, bias and ε; the keys' LayerNorm, no rope, the
/// transform and the f16 row at each position of `keys`. The one owner of the launch's arguments,
/// which the step, the rows pass and the gate (`gate_kimi_k3_dsa`) run.
///
/// # Errors
/// [`GpuError`] from the launch's refusals.
pub fn enqueue_keys(
    stream: &CudaStream,
    k: &Kernels,
    (gain, bias, eps): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, f32),
    (x, pos, m): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, usize),
    fault: FaultSink,
    keys: &mut DeviceTensor<u16>,
) -> Result<(), GpuError> {
    k.key.enqueue_index_key_input(
        stream,
        IndexKeyInputArgs {
            x,
            gain,
            bias,
            pos,
            // Not read: the key turns no value (`n_dims` 0).
            cs: x,
            eps,
            n_dims: 0,
            m,
            cache: keys,
            fault,
        },
    )
}

/// Step 3 of the module doc at one row: `q` the heads' queries `[heads · d]`, `w` their weights
/// `[heads]` before the scale, `words` the step's count words ([`count_words`]); every visible row
/// of `keys` scored into `scratch` and the `min(n_vis, top_k)` best into `list`, whose length is the
/// list's width. The one owner of the launch's arguments, which the step and the gate run.
///
/// # Errors
/// [`GpuError`] from the launches' refusals.
pub fn enqueue_pick(
    stream: &CudaStream,
    k: &Kernels,
    keys: &DeviceTensor<u16>,
    (q, w): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
    words: &DeviceBuffer<u32>,
    scratch: &mut IndexerScratch,
    list: &mut DeviceBuffer<u32>,
) -> Result<(), GpuError> {
    let stride = list.len();
    k.score.enqueue(
        stream,
        IndexerArgs {
            q,
            w,
            ints: words,
            n_vis_at: N_VIS_AT,
            top_k_at: TOP_K_AT,
            // Not read: the query turns no value (the score's `n_dims` 0).
            tables: words,
            rope_at: 0,
            rope_stride: 0,
            keys,
            tokens: 1,
            scratch,
            list,
            stride,
        },
    )
}

/// The kernels the indexer launches beside the step's.
pub struct Kernels {
    key: IndexKeyKernels,
    score: IndexerKernels,
}

impl Kernels {
    /// The modules, the score's at the file's 32 heads of 128 with no rope. Load-time only.
    ///
    /// # Errors
    /// [`GpuError`] from the modules' loads.
    pub fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            key: IndexKeyKernels::load(ctx)?,
            score: IndexerKernels::with_shape(ctx, INDEX_HEADS, INDEX_DIM, 0)?,
        })
    }
}

/// The step's selection buffers, one set every selecting layer reuses in turn: the count words,
/// the indexer's projections, the list, the gathered rows and the indexer's scratch over the
/// layers' `ctx` key rows.
pub(super) struct Sel {
    top_k: usize,
    host: [u32; WORDS],
    /// `[pos + 1, top_k, min(pos + 1, top_k)]`.
    words: DeviceBuffer<u32>,
    /// `[d]`, `[heads · d]`, `[heads]`: the key's, the queries' and the weights' projections.
    ik: DeviceBuffer<f32>,
    iq: DeviceBuffer<f32>,
    iw: DeviceBuffer<f32>,
    /// `[top_k]`: the picked rows, ascending.
    list: DeviceBuffer<u32>,
    /// `[top_k][row]` f16.
    gathered: DeviceTensor<u16>,
    scratch: IndexerScratch,
}

impl Sel {
    /// The buffers for `hp`'s file at `ctx` positions with latent rows of `row` values. Load-time
    /// only. Refused by name as [`check_selection`] refuses.
    pub(super) fn new(
        stream: &CudaStream,
        hp: &Hparams,
        row: usize,
        ctx: usize,
    ) -> Result<Sel, GpuError> {
        let ix = hp.indexer.as_ref();
        let top_k = check_selection(ix, ix.map_or(0, |ix| ix.top_k))?;
        let (heads, d) = (INDEX_HEADS, INDEX_DIM);
        let top = u32::try_from(top_k)
            .map_err(|_| GpuError::shape(WHAT, format!("a top_k of {top_k} past u32")))?;
        Ok(Sel {
            top_k,
            host: [0, top, 0],
            words: DeviceBuffer::from_host(stream, &[0, top, 0])?,
            ik: DeviceBuffer::zeroed(stream, d)?,
            iq: DeviceBuffer::zeroed(stream, heads * d)?,
            iw: DeviceBuffer::zeroed(stream, heads)?,
            list: DeviceBuffer::zeroed(stream, top_k)?,
            gathered: DeviceTensor::zeroed(stream, top_k, row)?,
            scratch: IndexerScratch::new(stream, 1, ctx)?,
        })
    }

    /// The count words of the step at `pos`, in one copy before the step's launches.
    pub(super) fn words(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        self.host = count_words(pos, self.host[TOP_K_AT])?;
        Ok(self.words.copy_from_host(stream, &self.host)?)
    }

    /// The selection [`Index::pick`] left, for the latent core ([`crate::latent_gated::attend_sel`]),
    /// and the count word's view it reads.
    pub(super) fn parts(&mut self) -> Result<(SelParts<'_>, Window<'_, u32>), GpuError> {
        let n = Window::<u32>::of(&self.words, N_SEL_AT * 4, 1)?;
        Ok((
            SelParts {
                list: &self.list,
                stride: self.top_k,
                scratch: &mut self.gathered,
            },
            n,
        ))
    }

    /// Device bytes the buffers hold.
    pub(super) fn bytes(&self) -> usize {
        [&self.ik, &self.iq, &self.iw]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self.words.num_bytes()
            + self.list.num_bytes()
            + self.gathered.buf().num_bytes()
            + self.scratch.device_bytes()
    }
}

/// The parts of a [`Selection`] [`Sel`] holds; [`SelParts::with`] joins the count word.
pub(super) struct SelParts<'a> {
    list: &'a DeviceBuffer<u32>,
    stride: usize,
    scratch: &'a mut DeviceTensor<u16>,
}

impl<'a> SelParts<'a> {
    /// The selection over the count word `n`.
    pub(super) fn with(self, n: &'a DeviceBuffer<u32>) -> Selection<'a> {
        Selection {
            list: self.list,
            n,
            stride: self.stride,
            scratch: self.scratch,
        }
    }
}

#[cfg(test)]
mod tests {
    use model::arch::delta_mla::{Mla, Situ};
    use model::arch::kda::Keys;
    use model::arch::kimi_k3::hparams::{Hparams, INDEX_DIM, INDEX_HEADS, Indexer, Kind};
    use model::arch::kimi_k3::place::KvLayout;
    use model::arch::kimi_k3::spec::spec_of;
    use model::arch::models::{ChatSpec, KdaGate};
    use model::arch::noaux::Experts;
    use model::placement::KvBytes;

    use super::width;

    /// Positions of the layout the rows are counted at.
    const CTX: u64 = 64;

    /// The kinds of the small header: delta layers with a latent layer at 2 and 5.
    const KINDS: [Kind; 6] = [
        Kind::Kda,
        Kind::Kda,
        Kind::Mla,
        Kind::Kda,
        Kind::Kda,
        Kind::Mla,
    ];

    /// The small header's hyperparameters, a K3.1 file's when `indexer` is set: the indexer runs on
    /// the latent layers and no other.
    fn header(indexer: bool) -> Hparams {
        Hparams {
            arch: if indexer { "kimi-k31" } else { "kimi-k3" },
            n_layer: KINDS.len(),
            n_embd: 512,
            n_head: 2,
            kda: Keys {
                conv: 4,
                head_dim: 128,
            },
            gate: KdaGate::Sigmoid { lower_bound: -5.0 },
            mla: Mla {
                q_lora: 256,
                kv_lora: 512,
                head_k: 192,
                head_v: 128,
                pe: 64,
            },
            rms_eps: 1e-5,
            situ: Situ {
                gate_beta: 4.0,
                value_beta: 4.0,
            },
            latent: 256,
            attn_res_block: 2,
            n_vocab: 2,
            n_ctx_train: 4096,
            dense_lead: 1,
            dense_ff: 768,
            experts: Experts {
                n_expert: 8,
                n_used: 2,
                ff: 256,
                n_shared: 2,
                shared_width: 512,
                weights_scale: 1.0,
                weights_norm: true,
            },
            kinds: KINDS.to_vec(),
            indexer: indexer.then(|| Indexer {
                heads: INDEX_HEADS,
                d: INDEX_DIM,
                top_k: 16,
                norm_eps: 1e-5,
                layers: KINDS.iter().map(|&k| k == Kind::Mla).collect(),
            }),
            defaults: Vec::new(),
        }
    }

    /// The plan's layout of `hp`, from its description.
    fn layout(hp: &Hparams) -> KvLayout {
        let chat = ChatSpec {
            pre: "kimi-k2".to_string(),
            template: None,
            tools: None,
            reasoning: None,
        };
        let spec = spec_of(hp, chat).expect("the small header describes");
        KvLayout::of(spec.layers.iter().map(|l| &l.mixer)).expect("the description lays out")
    }

    /// The body's rows a position on a latent layer, the cached row and the index key row, are the
    /// plan's bytes for that layer at [`CTX`] positions, f16 values each [derived from the
    /// header's widths, apart from the reader]: a K3.1 latent layer holds 512 + 64 cached values
    /// and 128 key values a position, 90,112 B at 64 positions, where a K3 latent layer holds
    /// 73,728 B; a delta layer keeps no key row. The load's `refuse_stores` holds the stores to
    /// the same count. Mutants: the width `None` on a latent layer (the plan's key row is not
    /// held), the width kept on a delta layer, the width of the index heads, the width kept in a
    /// K3 file.
    #[test]
    fn a_latent_layers_rows_are_the_plans_bytes() {
        let f16 = size_of::<u16>() as u64;
        for indexer in [false, true] {
            let hp = header(indexer);
            let kv = layout(&hp);
            for (l, kind) in KINDS.into_iter().enumerate() {
                let keys = width(&hp, l);
                if kind == Kind::Kda {
                    assert_eq!(
                        keys, None,
                        "indexer {indexer}: delta layer {l} keeps no key"
                    );
                    continue;
                }
                let rows = (hp.mla.row() + keys.unwrap_or(0)) as u64;
                assert_eq!(
                    kv.layer_bytes(l, CTX),
                    CTX * f16 * rows,
                    "indexer {indexer}: latent layer {l}"
                );
            }
        }
    }
}
