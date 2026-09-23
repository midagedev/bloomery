//! The MoE sub-layer of the V4.1 step: the norm, the router, the go to the
//! host tier, the card's routed and shared experts, the wait, the combine
//! and HC_POST with the next fold. See [`super`] for the piece contract.
//!
//! One layer's launches, in stream order:
//!
//! ```text
//! norm+Q8 → router → handoff → go → HC_PRE → [gate·up → h q8_1 → down] → shared gate·up → shared down
//!         → wait → combine+HC_POST (+ fold)
//! ```
//!
//! - The norm writes the f32 activation into the boundary's handoff region
//!   and its q8_1 form into the piece's own scratch; the router and the
//!   shared expert read the f32 form there.
//! - `handoff` (`ds41_ffn_handoff`) writes the handoff straight into its
//!   image in the host-mapped page — the region's sequence word, the router's
//!   ids and weights, the activation — so no copy node carries it, and writes
//!   each slot's place in the card's routed stacks: the layer's row of the
//!   slot map's card copy at the id, [`HOST`] for an expert the card does not
//!   hold. The card's routed launches read those places, so which experts
//!   the card computes comes from the map alone.
//! - The go signals the host tier; the wait takes its answer back
//!   ([`bloomery_gpu::hybrid`]). With the overlap lever off the wait sits
//!   right after the go.
//! - HC_PRE sits after the go: its result feeds only this sub-layer's HC_POST
//!   and the next sub-layer's fold (the lag), so it runs while the host
//!   computes. So do the card's routed experts and the shared expert.
//! - A layer whose slot-map row holds no card expert runs no routed launch.
//! - One launch joins (`ds41_ffn_post`, `ds41_ffn_post_streams`): the combine
//!   ([`combine_elem`]) of the card's slots, the host's partial sum read in
//!   place through the mapping and the shared expert's output, then HC_POST
//!   of it by the HC_POST launch's own rule — the new streams and, except
//!   into an engram layer and after the last layer, the next sub-layer's
//!   fold. The combine's output is written too, for a gate to read.
//!
//! Launches per layer: ten kernels with card experts, seven without, and
//! each layer's two stream memory-operation batches (go, wait); no copy.
//!
//! The piece takes the resident weights at enqueue, not at [`FfnPiece::new`]:
//! the engine keeps its weights and its chain body apart, and the body that
//! holds this piece cannot borrow them. `new` resolves every name once; an
//! enqueue looks each up and checks its format.

use std::ops::Range;

use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::hybrid::{Boundary, HOST, HostExperts, Hybrid, SlotMap};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act, launch_u32};
use cuda_core::{DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::{GgmlType, Split};
use model::Tensor2;
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::{host, names};
use model::moe::{HostLayer, HostScratch};

use crate::experts::{ExpertGateUp, ExpertKernels};
use crate::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcParams, HcPreArgs, HcPreScratch,
    hc_fold_elem, hc_post_elem,
};
use crate::router::{N_EXPERT, N_USED, RouterKernels, RouterOut};

/// What the enqueue path's errors name.
const ENQUEUE: &str = "FfnPiece::enqueue";

/// Threads per block of the handoff and of the combine-and-HC_POST.
const HANDOFF_THREADS: u32 = 256;
const POST_THREADS: u32 = 256;

// The handoff kernel's contract spells the slot count as a literal.
const _: () = assert!(N_USED == 6);

/// The combine of one output value: the card's slots in slot order by
/// fused multiply-adds from zero, `acc = fma(down_j, w_j, acc)` for each slot
/// `j` whose place is on the card, then `(acc + hsum) + shexp`. The one rule
/// the device and a gate's host side both run.
#[inline(always)]
pub fn combine_elem(
    down: [f32; N_USED],
    w: [f32; N_USED],
    card: [bool; N_USED],
    hsum: f32,
    shexp: f32,
) -> f32 {
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < N_USED {
        if card[j] {
            acc = down[j].mul_add(w[j], acc);
        }
        j += 1;
    }
    (acc + hsum) + shexp
}

#[cuda_module]
mod ffn_kernels {
    use super::*;

    /// The handoff, written straight into its image in the host-mapped page,
    /// and the card's places — one thread per activation value `d < n`,
    /// which copies `x[d]` to the image's word `x_at + d`. Threads `s < 6`
    /// also copy slot `s`'s id and weight to words `ids_at + s` and `wts_at +
    /// s` and write `sel[s] = map[row_off + id]` — the id's slot in the card's
    /// routed stacks, or [`HOST`] — or [`HOST`] for an id not below
    /// `n_expert`; thread 0 copies the region's sequence word to `seq_at`.
    /// The go that follows orders every write here before its generation.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ids_in.len() >= 6,
            w_in.len() >= 6,
            map.len() >= row_off + n_expert,
            x.len() >= n,
            seq.len() >= 1,
            n >= 6,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + 6,
            x_at >= wts_at + 6,
            image.len() >= x_at + n,
            sel.len() >= 6
        )
    )]
    pub fn ds41_ffn_handoff(
        ids_in: &[u32],
        w_in: &[f32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        n: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        mut image: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        if d >= n as usize {
            return;
        }
        // SAFETY: d < n <= x.len(), and x_at + d < x_at + n <= image.len(), by
        // the launch contract; the image's words past x_at are the activation's
        // alone, and thread d is word x_at + d's only writer.
        unsafe {
            *image.get_unchecked_mut(x_at as usize + d) = (*x.get_unchecked(d)).to_bits();
        }
        if d < N_USED {
            // SAFETY: d < 6 <= ids_in.len() and w_in.len() by the launch
            // contract.
            let (id, w) = unsafe { (*ids_in.get_unchecked(d), *w_in.get_unchecked(d)) };
            let place = if id < n_expert {
                // SAFETY: id < n_expert, so row_off + id < map.len() by the
                // launch contract.
                unsafe { *map.get_unchecked(row_off as usize + id as usize) }
            } else {
                HOST
            };
            // SAFETY: d < 6 <= sel.len(); ids_at + d and wts_at + d lie in the
            // routing's two spans, which the contract keeps apart from each
            // other, from seq_at and from the activation, all inside the image;
            // thread d is each of those words' only writer.
            unsafe {
                *image.get_unchecked_mut(ids_at as usize + d) = id;
                *image.get_unchecked_mut(wts_at as usize + d) = w.to_bits();
                *sel.get_unchecked_mut(d) = place;
            }
        }
        if d == 0 {
            // SAFETY: seq.len() >= 1, and seq_at < ids_at is inside the image
            // and no other span's word, by the launch contract; thread 0 alone
            // writes it.
            unsafe { *image.get_unchecked_mut(seq_at as usize) = *seq.get_unchecked(0) };
        }
    }

    /// The combine and HC_POST with the next fold, one thread per value `d`
    /// ([`combine_post_at`]): writes the combine's `y[d]`, the four new
    /// streams at `d` to `out` and their fold by the HC_PRE result's `pre` to
    /// `fold`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows,
            fold.len() >= rows
        )
    )]
    pub fn ds41_ffn_post(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: d < rows, and the launch contract gives every length the
        // helper's contract asks for.
        let (yv, o, pre) = unsafe { combine_post_at(&a, rows, n_card, d) };
        // SAFETY: d < rows <= y.len() and fold.len(), and d + 3·rows < 4·rows
        // <= out.len(), by the launch contract; thread d is the only writer of
        // y[d], fold[d] and the four stream values at d.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
            *fold.get_unchecked_mut(d) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post`] without the fold: into an engram layer and after the
    /// last layer.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= 6 * rows,
            w.len() >= 6,
            sel.len() >= 6,
            hsum.len() >= rows,
            shexp.len() >= rows,
            res.len() >= 4 * rows,
            hc.len() >= 24,
            y.len() >= rows,
            out.len() >= 4 * rows
        )
    )]
    pub fn ds41_ffn_post_streams(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        rows: u32,
        n_card: u32,
        mut y: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
    ) {
        let d = thread::index_1d().get();
        let rows = rows as usize;
        if d >= rows {
            return;
        }
        let a = PostIn {
            down,
            w,
            sel,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: as in `ds41_ffn_post`.
        let (yv, o, _) = unsafe { combine_post_at(&a, rows, n_card, d) };
        // SAFETY: as in `ds41_ffn_post`, without the fold.
        unsafe {
            *y.get_unchecked_mut(d) = yv;
            store4(&mut out, rows, d, o);
        }
    }
}

/// What the combine-and-HC_POST launches read: `down` the six slots' down
/// outputs slot-major (`rows` each), `w` the six routing weights, `sel` each
/// slot's place (a slot is the card's when its place is below `n_card`, and
/// every other slot's rows are never read), `hsum` the host's partial sum,
/// `shexp` the shared expert's output, `res` the four streams the sub-layer
/// read (`rows` each) and `hc` its HC_PRE result.
struct PostIn<'a> {
    down: &'a [f32],
    w: &'a [f32],
    sel: &'a [u32],
    hsum: &'a [f32],
    shexp: &'a [f32],
    res: &'a [f32],
    hc: &'a [f32],
}

/// Value `d`'s combine ([`combine_elem`]), then HC_POST of it (`hc_post_elem`,
/// the HC_POST launch's own rule): the combine's `y`, the four new stream
/// values and `pre`.
///
/// SAFETY: `d < rows`, `down.len() >= 6 * rows`, `w.len()` and `sel.len() >=
/// 6`, `hsum.len()` and `shexp.len() >= rows`, `res.len() >= 4 * rows` and
/// `hc.len() >= 24`.
#[inline(always)]
unsafe fn combine_post_at(
    a: &PostIn<'_>,
    rows: usize,
    n_card: u32,
    d: usize,
) -> (f32, [f32; 4], [f32; 4]) {
    let mut dv = [0.0f32; N_USED];
    let mut wv = [0.0f32; N_USED];
    let mut card = [false; N_USED];
    let mut j = 0usize;
    while j < N_USED {
        // SAFETY: j < 6 <= sel.len() and w.len() by this fn's contract.
        let (place, wj) = unsafe { (*a.sel.get_unchecked(j), *a.w.get_unchecked(j)) };
        if place < n_card {
            card[j] = true;
            wv[j] = wj;
            // SAFETY: j < 6 and d < rows, so j·rows + d < 6·rows <=
            // down.len() by this fn's contract.
            dv[j] = unsafe { *a.down.get_unchecked(j * rows + d) };
        }
        j += 1;
    }
    // SAFETY: d < rows <= hsum.len() and shexp.len(); d + 3·rows < 4·rows <=
    // res.len(); 23 < 24 <= hc.len() — all by this fn's contract.
    let (hs, sh, r, pre, post, comb) = unsafe {
        (
            *a.hsum.get_unchecked(d),
            *a.shexp.get_unchecked(d),
            [
                *a.res.get_unchecked(d),
                *a.res.get_unchecked(d + rows),
                *a.res.get_unchecked(d + 2 * rows),
                *a.res.get_unchecked(d + 3 * rows),
            ],
            [
                *a.hc.get_unchecked(0),
                *a.hc.get_unchecked(1),
                *a.hc.get_unchecked(2),
                *a.hc.get_unchecked(3),
            ],
            [
                *a.hc.get_unchecked(4),
                *a.hc.get_unchecked(5),
                *a.hc.get_unchecked(6),
                *a.hc.get_unchecked(7),
            ],
            [
                *a.hc.get_unchecked(8),
                *a.hc.get_unchecked(9),
                *a.hc.get_unchecked(10),
                *a.hc.get_unchecked(11),
                *a.hc.get_unchecked(12),
                *a.hc.get_unchecked(13),
                *a.hc.get_unchecked(14),
                *a.hc.get_unchecked(15),
                *a.hc.get_unchecked(16),
                *a.hc.get_unchecked(17),
                *a.hc.get_unchecked(18),
                *a.hc.get_unchecked(19),
                *a.hc.get_unchecked(20),
                *a.hc.get_unchecked(21),
                *a.hc.get_unchecked(22),
                *a.hc.get_unchecked(23),
            ],
        )
    };
    let y = combine_elem(dv, wv, card, hs, sh);
    (y, hc_post_elem(y, r, post, &comb), pre)
}

/// The four stream values of value `d` into `out`, stream-major (`rows`
/// each).
///
/// SAFETY: `d < rows`, `out.len() >= 4 * rows`, and the four slots are the
/// calling thread's alone.
#[inline(always)]
unsafe fn store4(out: &mut DisjointSlice<f32>, rows: usize, d: usize, o: [f32; 4]) {
    // SAFETY: d + 3·rows < 4·rows <= out.len() by this fn's contract.
    unsafe {
        *out.get_unchecked_mut(d) = o[0];
        *out.get_unchecked_mut(d + rows) = o[1];
        *out.get_unchecked_mut(d + 2 * rows) = o[2];
        *out.get_unchecked_mut(d + 3 * rows) = o[3];
    }
}

/// A layer's routed stacks on the card: the experts its slot-map row puts
/// there, slot `s` at rows `s · rows .. (s + 1) · rows` of each — gate and up
/// q3_K (`ff` rows of `n_embd` per expert), down q4_K (`n_embd` rows of `ff`).
#[derive(Clone, Copy)]
pub struct CardStacks<'a> {
    pub gate: &'a DeviceTensor<u32>,
    pub up: &'a DeviceTensor<u32>,
    pub down: &'a DeviceTensor<u32>,
}

impl<'a> CardStacks<'a> {
    /// The stacks a placed load holds for layer `layer`: its routed gate, up
    /// and down file tensors in `w`, which a plan uploads with the experts of
    /// its card segment in slot order. `None` when `w` holds none of the
    /// three; an error when it holds some, or holds one in another format.
    pub fn of(w: &'a Weights, layer: usize) -> Result<Option<CardStacks<'a>>, GpuError> {
        const WHAT: &str = "CardStacks::of";
        let names = [
            names::ffn_gate_exps(layer),
            names::ffn_up_exps(layer),
            names::ffn_down_exps(layer),
        ];
        let found = names.each_ref().map(|n| w.get(n));
        if found.iter().all(Option::is_none) {
            return Ok(None);
        }
        let [gate, up, down] = [
            (0, GgmlType::Q3_K),
            (1, GgmlType::Q3_K),
            (2, GgmlType::Q4_K),
        ]
        .map(|(i, want)| match found[i] {
            Some(DevWeight::KQuant { ty, w: stack, .. }) if *ty == want => Ok(stack),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: names[i].clone(),
                need: "a resident K-quant routed stack beside the layer's other two",
            }),
        });
        Ok(Some(CardStacks {
            gate: gate?,
            up: up?,
            down: down?,
        }))
    }
}

/// The buffers the piece shares with the rest of the step, for one layer.
pub struct FfnIo<'a> {
    /// The streams the sub-layer reads — HC_PRE's input and HC_POST's
    /// residual: [`HC_STREAMS`] · `n_embd` f32.
    pub streams: &'a DeviceBuffer<f32>,
    /// The folded input the norm reads: `n_embd` f32.
    pub fold_in: &'a DeviceBuffer<f32>,
    /// The new streams: [`HC_STREAMS`] · `n_embd` f32.
    pub streams_out: &'a mut DeviceBuffer<f32>,
    /// The next sub-layer's fold, `n_embd` f32: `Some` exactly where the
    /// layer folds ([`FfnPiece::folds`]).
    pub fold_out: Option<&'a mut DeviceBuffer<f32>>,
    /// The slot map's card copy: a row of `n_expert` places per layer of the
    /// host tier's map, in its order.
    pub slots: &'a DeviceTensor<u32>,
}

/// One layer's tensor names and constants, resolved at load.
struct LayerCfg {
    norm: String,
    router: String,
    bias: String,
    sh_gate: String,
    sh_up: String,
    sh_down: String,
    hc_fn: String,
    hc_scale: String,
    hc_base: String,
    /// Experts the slot map puts on the card: 0 runs no routed launch.
    n_card: usize,
    /// The layer's row in the map's card copy, as a word offset.
    row_off: usize,
    limit: f32,
    limit_shared: f32,
    /// HC_POST with the next fold (`false`: HC_POST alone).
    fold: bool,
}

/// One layer's resident weights, looked up and format-checked at enqueue.
struct LayerWeights<'w> {
    gain: &'w DeviceBuffer<f32>,
    router: &'w DeviceTensor<f32>,
    bias: &'w DeviceBuffer<f32>,
    sh_gate: &'w DevWeight,
    sh_up: &'w DevWeight,
    sh_qs: &'w DeviceTensor<u32>,
    sh_d: &'w DeviceTensor<u16>,
    hc: HcParams<'w>,
}

impl<'w> LayerWeights<'w> {
    /// Layer `c`'s tensors in `w`, each in the format its launch reads, with
    /// the HC_PRE constants.
    fn resolve(
        c: &LayerCfg,
        w: &'w Weights,
        hc_eps: f32,
        hc_iters: u32,
    ) -> Result<LayerWeights<'w>, GpuError> {
        let gain = f32_weight(w, &c.norm)?;
        let router = f32_tensor(w, &c.router)?;
        let bias = f32_weight(w, &c.bias)?;
        let (sh_gate, sh_up) = (weight(w, &c.sh_gate)?, weight(w, &c.sh_up)?);
        let DevWeight::Q8_0 {
            qs: sh_qs, d: sh_d, ..
        } = weight(w, &c.sh_down)?
        else {
            return Err(tensor_err(&c.sh_down, "a q8_0 file tensor"));
        };
        let DevWeight::KQuant {
            ty: GgmlType::Q3_K,
            w: hc_w,
            ..
        } = weight(w, &c.hc_fn)?
        else {
            return Err(tensor_err(&c.hc_fn, "a q3_K file tensor"));
        };
        Ok(LayerWeights {
            gain,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_qs,
            sh_d,
            hc: HcParams {
                w: hc_w,
                scale: f32_weight(w, &c.hc_scale)?,
                base: f32_weight(w, &c.hc_base)?,
                eps: hc_eps,
                iters: hc_iters,
            },
        })
    }
}

/// The piece's own buffers, as the last enqueued layer left them — what a
/// gate reads back.
pub struct FfnTaps<'a> {
    /// The router's per-expert logits and scores and its routing.
    pub router: &'a RouterOut,
    /// Each slot's place in the card's routed stacks, [`HOST`] for the host's.
    pub sel: &'a DeviceBuffer<u32>,
    /// The routed SwiGLU outputs, slot-major (`ff` each): the card's slots.
    pub h: &'a DeviceBuffer<f32>,
    /// The routed down outputs, slot-major (`n_embd` each): the card's slots.
    pub down: &'a DeviceBuffer<f32>,
    /// The shared expert's SwiGLU output and its down output.
    pub shexp_h: &'a DeviceBuffer<f32>,
    pub shexp: &'a DeviceBuffer<f32>,
    /// The combine's output.
    pub y: &'a DeviceBuffer<f32>,
    /// HC_PRE's result, [`HC_MIX`] values.
    pub hc: &'a DeviceBuffer<f32>,
}

/// The MoE sub-layer of every layer of one slot map, built once at load.
pub struct FfnPiece {
    fused: FusedKernels,
    router: RouterKernels,
    experts: ExpertKernels,
    hc: HcKernels,
    module: ffn_kernels::LoadedModule,
    layers: Range<usize>,
    cfg: Vec<LayerCfg>,
    n_embd: usize,
    ff: usize,
    n_expert: usize,
    rms_eps: f32,
    hc_eps: f32,
    hc_iters: u32,
    scale: f32,
    act_x: Q8Act,
    rout: RouterOut,
    sel: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    down: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    sh_y: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    mixes: DeviceBuffer<f32>,
    hc_out: DeviceBuffer<f32>,
    hc_scratch: HcPreScratch,
}

impl FfnPiece {
    /// The piece for every layer of `map`, whose rows say which experts each
    /// layer's card stacks hold: names, constants, the kernels and the
    /// piece's scratch, all resolved here. Load-time only.
    pub fn new(gpu: &Gpu, hp: &Hparams, map: &SlotMap) -> Result<FfnPiece, GpuError> {
        const WHAT: &str = "FfnPiece::new";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let (n_embd, ff, ex) = (hp.n_embd, hp.experts.ff, &hp.experts);
        if ex.n_expert != N_EXPERT || ex.n_used != N_USED || map.n_expert() != ex.n_expert {
            return Err(refuse(format!(
                "{} experts, {} used, a map of {}: the kernels are built for {N_EXPERT} and {N_USED}",
                ex.n_expert,
                ex.n_used,
                map.n_expert()
            )));
        }
        if hp.hc.streams != HC_STREAMS || !ff.is_multiple_of(256) {
            return Err(refuse(format!(
                "{} streams and ff {ff}: want {HC_STREAMS} streams and whole super-blocks",
                hp.hc.streams
            )));
        }
        let layers = map.layers();
        if layers.end > hp.n_layer {
            return Err(refuse(format!(
                "the map's layers {layers:?} pass the model's {}",
                hp.n_layer
            )));
        }
        let mut cfg = Vec::with_capacity(layers.len());
        for (i, l) in layers.clone().enumerate() {
            let kind = &hp.layers[l];
            if !kind.routed {
                return Err(refuse(format!("layer {l} does not route")));
            }
            cfg.push(LayerCfg {
                norm: names::ffn_norm(l),
                router: names::ffn_gate_inp(l),
                bias: names::exp_probs_b(l),
                sh_gate: names::ffn_gate_shexp(l),
                sh_up: names::ffn_up_shexp(l),
                sh_down: names::ffn_down_shexp(l),
                hc_fn: names::hc_ffn_fn(l),
                hc_scale: names::hc_ffn_scale(l),
                hc_base: names::hc_ffn_base(l),
                n_card: map.on_card(l),
                row_off: i * map.n_expert(),
                limit: kind.swiglu_limit,
                limit_shared: kind.swiglu_limit_shared,
                fold: hp.layers.get(l + 1).is_some_and(|k| k.engram.is_none()),
            });
        }
        let ctx = gpu.context();
        let stream = gpu.stream();
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { ffn_kernels::load(ctx)? };
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(FfnPiece {
            fused: FusedKernels::load(ctx)?,
            router: RouterKernels::load(ctx)?,
            experts: ExpertKernels::load(ctx)?,
            hc: HcKernels::load(ctx)?,
            module,
            layers,
            cfg,
            n_embd,
            ff,
            n_expert: ex.n_expert,
            rms_eps: hp.rms_eps,
            hc_eps: hp.hc.eps,
            hc_iters: u32::try_from(hp.hc.sinkhorn_iters)
                .map_err(|_| refuse(format!("{} Sinkhorn rounds", hp.hc.sinkhorn_iters)))?,
            scale: ex.routed_scale,
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            rout: RouterOut::new(stream)?,
            sel: DeviceBuffer::zeroed(stream, N_USED)?,
            h: z(N_USED * ff)?,
            act_h: Q8Act::with_k(stream, N_USED, ff)?,
            down: z(N_USED * n_embd)?,
            sh_h: z(ff)?,
            sh_y: z(n_embd)?,
            y: z(n_embd)?,
            mixes: z(HC_MIX)?,
            hc_out: z(HC_MIX)?,
            hc_scratch: HcPreScratch::new(stream, HC_STREAMS * n_embd)?,
        })
    }

    /// The layers the piece was built for.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Whether layer `layer` ends with HC_POST and the next fold (`true`) or
    /// HC_POST alone — into an engram layer and after the last layer.
    #[must_use]
    pub fn folds(&self, layer: usize) -> Option<bool> {
        self.cfg_of(layer).map(|c| c.fold)
    }

    /// Kernel launches layer `layer` enqueues, besides its two
    /// memory-operation batches; it enqueues no copy.
    #[must_use]
    pub fn launches(&self, layer: usize) -> Option<usize> {
        self.cfg_of(layer)
            .map(|c| if c.n_card > 0 { 10 } else { 7 })
    }

    /// Device bytes of the piece's own scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let f32s = [
            &self.h,
            &self.down,
            &self.sh_h,
            &self.sh_y,
            &self.y,
            &self.mixes,
            &self.hc_out,
            &self.rout.logits,
            &self.rout.probs,
            &self.rout.weights,
        ];
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.sel.num_bytes()
            + self.rout.ids.num_bytes()
            + ROUTER_TICKET_BYTES
            + q8act_bytes(&self.act_x)
            + q8act_bytes(&self.act_h)
            + hc_pre_scratch_bytes(&self.hc_scratch)
    }

    /// The piece's buffers as the last enqueued layer left them.
    #[must_use]
    pub fn taps(&self) -> FfnTaps<'_> {
        FfnTaps {
            router: &self.rout,
            sel: &self.sel,
            h: &self.h,
            down: &self.down,
            shexp_h: &self.sh_h,
            shexp: &self.sh_y,
            y: &self.y,
            hc: &self.hc_out,
        }
    }

    fn cfg_of(&self, layer: usize) -> Option<&LayerCfg> {
        self.cfg.get(layer.checked_sub(self.layers.start)?)
    }

    /// Enqueue layer `layer`'s MoE sub-layer (the module comment's order):
    /// `io` the step's shared buffers, `w` the resident weights, `card` the
    /// layer's routed stacks on the card (`None` exactly when its map row
    /// puts no expert there), `hybrid` the host tier, which is told the
    /// layer is enqueued last — an eager chain has it served then, a capture
    /// notes it for its replays. Asynchronous apart from that service,
    /// allocation-free, capturable.
    pub fn enqueue<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        card: Option<CardStacks<'_>>,
        io: FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
    ) -> Result<(), GpuError> {
        let (i, card) = self.check(layer, hybrid.boundary().slots(), &io, card)?;
        let lw = LayerWeights::resolve(&self.cfg[i], w, self.hc_eps, self.hc_iters)?;
        self.enqueue_handoff(gpu, i, &lw, &io, hybrid, layer)?;
        self.enqueue_shadow(gpu, i, &lw, card, &io, hybrid.boundary())?;
        self.enqueue_join(gpu, i, io, hybrid.boundary())?;
        hybrid.layer_enqueued(layer)
    }

    /// Layer `layer`'s index in the piece, once the host tier's map, the
    /// shared buffers and the card's stacks are checked against it; the
    /// stacks back.
    fn check<'s>(
        &self,
        layer: usize,
        map: &SlotMap,
        io: &FfnIo<'_>,
        card: Option<CardStacks<'s>>,
    ) -> Result<(usize, Option<CardStacks<'s>>), GpuError> {
        let refuse = |detail: String| GpuError::Shape {
            what: ENQUEUE,
            detail,
        };
        let (n, ff) = (self.n_embd, self.ff);
        let i = layer
            .checked_sub(self.layers.start)
            .filter(|&i| i < self.cfg.len())
            .ok_or_else(|| refuse(format!("layer {layer} is outside {:?}", self.layers)))?;
        let c = &self.cfg[i];
        if map.layers() != self.layers || map.on_card(layer) != c.n_card {
            return Err(refuse(format!(
                "the host tier's map (layers {:?}, {} experts of layer {layer} on the card) is not the \
                 piece's ({:?}, {})",
                map.layers(),
                map.on_card(layer),
                self.layers,
                c.n_card
            )));
        }
        if io.slots.rows() != self.layers.len() || io.slots.cols() != self.n_expert {
            return Err(refuse(format!(
                "the map's card copy is {} x {}, want {} x {}",
                io.slots.rows(),
                io.slots.cols(),
                self.layers.len(),
                self.n_expert
            )));
        }
        if io.streams.len() < HC_STREAMS * n
            || io.streams_out.len() < HC_STREAMS * n
            || io.fold_in.len() < n
            || io.fold_out.as_ref().is_some_and(|f| f.len() < n)
            || io.fold_out.is_some() != c.fold
        {
            return Err(refuse(format!(
                "layer {layer}: streams {} and {} (want {}), fold in {} (want {n}), fold out {:?} \
                 (the layer folds: {})",
                io.streams.len(),
                io.streams_out.len(),
                HC_STREAMS * n,
                io.fold_in.len(),
                io.fold_out.as_ref().map(|f| f.len()),
                c.fold
            )));
        }
        let card = match (card, c.n_card) {
            (None, 0) => None,
            (Some(s), k) if k > 0 => {
                let (gu_words, d_words) = (110 * (n / 256) / 4, 36 * (ff / 256));
                if s.gate.rows() != k * ff
                    || s.up.rows() != k * ff
                    || s.down.rows() != k * n
                    || s.gate.cols() != gu_words
                    || s.up.cols() != gu_words
                    || s.down.cols() != d_words
                {
                    return Err(refuse(format!(
                        "layer {layer}: {k} card experts need gate/up {} x {gu_words} and down {} x \
                         {d_words}; got {} x {}, {} x {}, {} x {}",
                        k * ff,
                        k * n,
                        s.gate.rows(),
                        s.gate.cols(),
                        s.up.rows(),
                        s.up.cols(),
                        s.down.rows(),
                        s.down.cols()
                    )));
                }
                Some(s)
            }
            (s, k) => {
                return Err(refuse(format!(
                    "layer {layer}: the map puts {k} experts on the card and the stacks are {}",
                    if s.is_some() { "given" } else { "absent" }
                )));
            }
        };
        Ok((i, card))
    }

    /// The norm, the router, the handoff into the page and the go — and,
    /// with the overlap lever off, the wait right after it.
    fn enqueue_handoff<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        i: usize,
        lw: &LayerWeights<'_>,
        io: &FfnIo<'_>,
        hybrid: &mut Hybrid<H>,
        layer: usize,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let n = self.n_embd;
        let c = &self.cfg[i];
        self.fused.enqueue_norm_quant(
            stream,
            io.fold_in,
            lw.gain,
            self.rms_eps,
            &mut self.act_x,
            hybrid.boundary_mut().normed_mut(),
        )?;
        self.router.enqueue_router(
            stream,
            lw.router,
            hybrid.boundary().normed(),
            lw.bias,
            self.scale,
            &mut self.rout,
        )?;
        let what = "ds41_ffn_handoff";
        let target = hybrid.boundary_mut().handoff_target();
        let lay = target.layout;
        if lay.n_used != N_USED || lay.hidden != n {
            return Err(GpuError::Shape {
                what: ENQUEUE,
                detail: format!(
                    "the boundary carries {} slots of {} values; the piece hands over {N_USED} of {n}",
                    lay.n_used, lay.hidden
                ),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(HANDOFF_THREADS as usize))?;
        let prep =
            self.module
                .prepare_ds41_ffn_handoff(LaunchConfig1D::new(grid, HANDOFF_THREADS, 0))?;
        self.module.ds41_ffn_handoff(
            stream,
            &prep,
            &self.rout.ids,
            &self.rout.weights,
            io.slots.buf(),
            launch_u32(what, "row_off", c.row_off)?,
            launch_u32(what, "n_expert", self.n_expert)?,
            target.x,
            target.seq,
            launch_u32(what, "n", n)?,
            launch_u32(what, "seq_at", lay.seq)?,
            launch_u32(what, "ids_at", lay.ids)?,
            launch_u32(what, "wts_at", lay.weights)?,
            launch_u32(what, "x_at", lay.x)?,
            target.image,
            &mut self.sel,
        )?;
        let boundary = hybrid.boundary();
        boundary.enqueue_go(stream, layer)?;
        if !boundary.overlap() {
            boundary.enqueue_back(stream)?;
        }
        Ok(())
    }

    /// The shadow of the host's leg: HC_PRE, the card's routed experts, the
    /// shared expert — and, with the overlap lever on, the wait after them.
    fn enqueue_shadow(
        &mut self,
        gpu: &Gpu,
        i: usize,
        lw: &LayerWeights<'_>,
        card: Option<CardStacks<'_>>,
        io: &FfnIo<'_>,
        boundary: &Boundary,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let (n, ff) = (self.n_embd, self.ff);
        let c = &self.cfg[i];
        let pre = HcPreArgs {
            params: &lw.hc,
            x: io.streams,
            tokens: 1,
            rms_eps: self.rms_eps,
        };
        self.hc.enqueue_pre(
            stream,
            &pre,
            &mut self.hc_scratch,
            &mut self.mixes,
            &mut self.hc_out,
        )?;
        if let Some(s) = card {
            let args = ExpertGateUp {
                wg: s.gate,
                wu: s.up,
                act: &self.act_x,
                sel: &self.sel,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit: c.limit,
            };
            self.experts
                .enqueue_expert_gate_up(stream, &args, &mut self.h)?;
            gpu.enqueue_quantize_q8_1(&self.h, &mut self.act_h)?;
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                s.down,
                &self.act_h,
                &self.sel,
                N_USED,
                n,
                &mut self.down,
            )?;
        }
        self.experts.enqueue_shexp_gate_up(
            stream,
            lw.sh_gate,
            lw.sh_up,
            boundary.normed(),
            c.limit_shared,
            &mut self.sh_h,
        )?;
        gpu.q8f32()
            .enqueue_q8_0_gemv(stream, lw.sh_qs, lw.sh_d, &self.sh_h, 1, &mut self.sh_y)?;
        if boundary.overlap() {
            boundary.enqueue_back(stream)?;
        }
        Ok(())
    }

    /// The join: the combine and HC_POST in one launch, with the next fold
    /// where the layer folds.
    fn enqueue_join(
        &mut self,
        gpu: &Gpu,
        i: usize,
        io: FfnIo<'_>,
        boundary: &Boundary,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let n = self.n_embd;
        let grid = launch_u32("ds41_ffn_post", "grid", n.div_ceil(POST_THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, POST_THREADS, 0);
        let rows = launch_u32("ds41_ffn_post", "rows", n)?;
        let n_card = launch_u32("ds41_ffn_post", "n_card", self.cfg[i].n_card)?;
        match io.fold_out {
            Some(fold) => {
                let prep = self.module.prepare_ds41_ffn_post(cfg)?;
                self.module.ds41_ffn_post(
                    stream,
                    &prep,
                    &self.down,
                    &self.rout.weights,
                    &self.sel,
                    boundary.hsum(),
                    &self.sh_y,
                    io.streams,
                    &self.hc_out,
                    rows,
                    n_card,
                    &mut self.y,
                    io.streams_out,
                    fold,
                )?;
            }
            None => {
                let prep = self.module.prepare_ds41_ffn_post_streams(cfg)?;
                self.module.ds41_ffn_post_streams(
                    stream,
                    &prep,
                    &self.down,
                    &self.rout.weights,
                    &self.sel,
                    boundary.hsum(),
                    &self.sh_y,
                    io.streams,
                    &self.hc_out,
                    rows,
                    n_card,
                    &mut self.y,
                    io.streams_out,
                )?;
            }
        }
        Ok(())
    }
}

/// The resident weight `name`.
fn weight<'w>(w: &'w Weights, name: &str) -> Result<&'w DevWeight, GpuError> {
    w.get(name)
        .ok_or_else(|| tensor_err(name, "a resident weight"))
}

/// The resident f32 tensor `name` (an f32 file tensor, or bf16 decoded at
/// load).
fn f32_tensor<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<f32>, GpuError> {
    match weight(w, name)? {
        DevWeight::F32 { w, .. } => Ok(w),
        _ => Err(tensor_err(name, "an f32 tensor")),
    }
}

/// The resident f32 vector `name`, as the buffer a kernel reads.
fn f32_weight<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    f32_tensor(w, name).map(DeviceTensor::buf)
}

fn tensor_err(name: &str, need: &'static str) -> GpuError {
    GpuError::Tensor {
        what: ENQUEUE,
        name: name.to_string(),
        need,
    }
}

/// The router's block ticket count, which [`RouterOut`] keeps private: one
/// u32.
const ROUTER_TICKET_BYTES: usize = 4;

/// Device bytes an [`HcPreScratch`] of its K allocates (`HcPreScratch::new`'s
/// partial sums, squares and ticket count).
fn hc_pre_scratch_bytes(s: &HcPreScratch) -> usize {
    let pieces = s.k() / HC_PIECE;
    4 * HC_MIX * pieces * HC_MAX_TOKENS + 4 * pieces * HC_MAX_TOKENS + 4
}

/// Device bytes a [`Q8Act`] of its shape allocates (`Q8Act::with_k`'s five
/// buffers).
fn q8act_bytes(a: &Q8Act) -> usize {
    let (m, n_sb) = (a.m(), a.n_sb());
    8 * m * 64 * n_sb.div_ceil(2)
        + 4 * m * 256 * n_sb.div_ceil(4)
        + 4 * m * 128 * n_sb.div_ceil(2)
        + 4 * m * 8 * n_sb
        + 4 * m * 2 * n_sb
}

/// The V4.1 host tier's computation ([`HostExperts`]): each layer's routed
/// stacks as [`HostLayer`] reads them from the file, and the scratch every
/// call writes, made at load.
pub struct Ds41Host {
    file: Split,
    /// Per layer of `layers`, its host view; `None` for a layer that does
    /// not route.
    layers: Vec<Option<HostLayer>>,
    first: usize,
    scratch: HostScratch,
}

impl Ds41Host {
    /// The tier for layers `layers` of `file`, whose hyperparameters are
    /// `hp`. Load-time only: every stack is found and checked here.
    pub fn build(file: Split, hp: &Hparams, layers: Range<usize>) -> Result<Ds41Host, GpuError> {
        let first = layers.start;
        let views = layers
            .map(|l| host::layer(&file, hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Ds41Host {
            file,
            layers: views,
            first,
            scratch: HostScratch::new(hp.n_embd, hp.experts.ff),
        })
    }
}

impl HostExperts for Ds41Host {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let view = layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .and_then(Option::as_ref)
            .ok_or(GpuError::State {
                what: "Ds41Host::experts_into",
                missing: "the layer's routed stacks: it is outside the tier or does not route",
            })?;
        view.experts_into(&self.file, x, experts, out, &mut self.scratch)?;
        Ok(())
    }
}
