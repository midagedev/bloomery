//! A `nemotron_h_moe` file planned onto a machine: the family's half of the
//! one path from the file's headers to a plan, as the Nano's
//! (`nemotron_h::place`). The hyperparameters, every tensor's role, the typed
//! description and each layer's store bytes are read once ([`describe`];
//! [`read`] refuses a file whose features no program runs, every one listed);
//! the plan is the hybrid trunk's ([`PlanInputs::plan_with`]): the whole model
//! on one card, each routed stack in the format the body's `_sel` launches
//! read (IQ2_XXS, or a 32-value-block type or NVFP4 in its planes in the card
//! leg's relu² form: [`crate::placement::accepts::latent_leg_runs`]), and the routed experts where
//! [`Experts`] says. [`PlanInputs::plan`]
//! is the whole card's. A plan under [`Experts::Placed`] or [`Experts::Host`]
//! puts experts on the host, and the host experts the room cannot hold on the
//! NVMe expert tier.
//!
//! A sequence's stores are an attention layer's K and V planes and a Mamba-2
//! layer's scan state and conv ring; a latent MoE layer holds nothing
//! ([`Holds`]).

use gguf::{GgmlType, Split};

use super::hparams::{Hparams, Kind};
use super::spec;
pub use crate::arch::hybrid::{Experts, Holds, KvLayout, PlaceError, PlanInputs};
use crate::placement::PlacementError;
use crate::placement::kernels::{
    card_planes_runs, card_relu2_runs, gemm_act32_bytes, q8_act_bytes, q8_blocks32_bytes,
};

/// The rows the plan keeps a prompt pass's buffers for: the host union's columns, the most a
/// placed load's pass takes. A whole-card load's pass is the card's room, at least these.
pub const PASS_ROWS: usize = crate::moe::UNION_MAX_COLS;

/// The kernels' constants the pass's buffers are sized by (the columns of a gemv run, the leading
/// words of the word record, the flash's head width, key-head group, segments and rows of one pass
/// the attention takes at once, the GEMMs' column tile and 32-value step) and the widest router
/// instance's expert count, the `probs` rows; the body holds each equal to the kernels' by
/// assertion.
pub const COLS: usize = 8;
/// See [`COLS`].
pub const WORD_HEAD: usize = 1;
/// See [`COLS`].
pub const HEAD: usize = 128;
/// See [`COLS`].
pub const SPLIT_GROUP: usize = 8;
/// See [`COLS`].
pub const FLASH_SEGMENTS: usize = 80;
/// See [`COLS`].
pub const MAX_PASS_ROWS: usize = 8;
/// See [`COLS`].
pub const GEMM_BN: usize = 64;
/// See [`COLS`].
pub const GEMM32_STEP: usize = 64;
/// See [`COLS`].
pub const WIDEST_ROUTER: usize = 496;

/// The widths a prompt pass's buffers are cut from, every one a fact of the file or the kernels
/// ([`pass_bytes`]). The body collects them from its loaded layers; the plan from the file's
/// headers ([`pass_dims`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassDims {
    /// `embedding_length`.
    pub hidden: usize,
    /// The key heads of an attention layer.
    pub n_kv: usize,
    /// The query heads of an attention layer.
    pub n_head: usize,
    /// A Mamba-2 layer's in-projection row: `d_inner + conv_dim + ssm_heads`.
    pub in_proj: usize,
    /// A Mamba-2 layer's inner width.
    pub d_inner: usize,
    /// A Mamba-2 layer's scan heads.
    pub ssm_heads: usize,
    /// A Mamba-2 layer's conv channels.
    pub conv_dim: usize,
    /// The values the routed experts read and write.
    pub width: usize,
    /// The experts a token keeps.
    pub top_k: usize,
    /// The shared expert's width.
    pub shared_ff: usize,
    /// The widest routed expert's up width; 0 where no layer is a MoE layer.
    pub ff: usize,
    /// The widest router instance's expert count (the `probs` rows).
    pub n_expert: usize,
    /// The q8_1 slot forms the `_sel` fallback reads, `(k, columns)`.
    pub slots: Vec<(usize, usize)>,
    /// The routed tables' expert counts, one a table.
    pub route_n: Vec<usize>,
    /// The experts' GEMM forms, `(k, slots a row)`.
    pub gemm_acts: Vec<(usize, usize)>,
    /// The 32-value GEMM forms of 32-value routed stacks, `(k, slots a row)`.
    pub gemm32_acts: Vec<(usize, usize)>,
    /// The forms of `gemm32_acts` that also hold the native FP4 columns
    /// ([`native_act_col_bytes`]): those of NVFP4 routed stacks, at a width of whole 64-value
    /// steps, on a build with native FP4 products.
    pub native32_acts: Vec<(usize, usize)>,
    /// The NVFP4 gemv run forms' widths, at the widths the q8_1 form takes.
    pub run_k: Vec<usize>,
    /// The NVFP4 gemv run forms' widths at the widths it does not: [`COLS`] columns of q8 blocks
    /// of 32 for the `_sel` gemv.
    pub run32_k: Vec<usize>,
    /// The q8 blocks of 32 the wide GEMMs read, one a site width.
    pub wide_k: Vec<usize>,
    /// Whether the pass runs on a placed load: its rows' places and the host's sums are buffers of
    /// the pass.
    pub placed: bool,
}

/// Bytes of one q8_1 activation column of `k` values (a `Q8Act`'s or a `GemmAct`'s).
fn act_col_bytes(k: usize) -> usize {
    q8_act_bytes(1, k as u64) as usize
}

/// Bytes of one `GemmAct32` column of `k` values ([`gemm_act32_bytes`]).
fn act32_col_bytes(k: usize) -> usize {
    gemm_act32_bytes(1, k as u64) as usize
}

/// Bytes of one native FP4 column of `k` values (a multiple of 64) beside a `GemmAct32` column:
/// a code nibble a value (`k / 8` words), a scale word a 64-value step (`k / 64` words) and the
/// column's f32 scale.
fn native_act_col_bytes(k: usize) -> usize {
    k / 2 + k / 16 + size_of::<f32>()
}

/// Bytes of a `GemmRoute` table for `slots` slots over `experts` experts.
fn route_bytes(slots: usize, experts: usize) -> usize {
    let tiles = slots / GEMM_BN + experts.min(slots);
    let zeros = if experts == 1 { slots } else { 0 };
    (slots + 2 * tiles + 2 + zeros) * 4
}

/// The device bytes the pass's buffers hold at `rows` rows of `d`, every buffer the body's
/// `Prompt::new` allocates, by the same widths, four bytes a value. The one formula of those bytes:
/// the plan keeps them out of the expert budget ([`pass_reserve_bytes`]), the body's fit check
/// counts them before it allocates and asserts it allocated exactly them. Every row buffer is cut
/// at `rows` rounded up to [`COLS`], so a run of [`COLS`] columns past the pass's last row writes
/// inside its buffer, at the one-column floor where a width is 0. Terms: the residual streams twice,
/// the normed row and the sums (`4 · hidden`); the router's scores (`n_expert`) and weights
/// (`top_k`); the latent and the weighted sums (`2 · width`); the slots' up and down rows
/// (`top_k · ff`, `top_k · width`); the shared expert's up and relu² rows; the key and value rows
/// twice (`4 · n_kv · HEAD`); the trunk's row buffers (`2 · hidden`); a Mamba-2 layer's in-projection,
/// `d_inner` rows; the query and attention rows (`2 · n_head · HEAD`); the host sums on a placed
/// load (`width`); the flash's partials for at most [`MAX_PASS_ROWS`] rows; the word record, the
/// ids, the places, the shared expert's slots and the positions; the GEMM routes and activation
/// forms, the 32-value GEMMs' among them and the native FP4 columns beside those of NVFP4 stacks;
/// the gemv run forms, the q8_1 ones and the 32-value ones of an NVFP4 site at a width the q8_1
/// form does not take; the flash split's sub-group rows; and the Mamba-2 pass scratch.
#[must_use]
pub fn pass_bytes(d: &PassDims, rows: usize) -> usize {
    let cut = rows.next_multiple_of(COLS);
    let kv_width = d.n_kv * HEAD;
    let f = size_of::<f32>();
    let hsum_width = if d.placed { d.width } else { 0 };
    let attn_rows = cut.min(MAX_PASS_ROWS);
    let z = |n: usize| cut * n.max(1) * f;
    let f32s = 4 * z(d.hidden)
        + z(d.n_expert)
        + z(d.top_k)
        + z(d.width) * 2
        + z(d.top_k * d.ff)
        + z(d.top_k * d.width)
        + z(d.shared_ff) * 2
        + 2 * z(kv_width)
        + 2 * z(d.hidden)
        + z(d.in_proj)
        + z(d.d_inner)
        + 2 * z(d.n_head * HEAD)
        + 2 * z(kv_width)
        + 2 * z(0)
        + z(hsum_width)
        + attn_rows * d.n_head * FLASH_SEGMENTS * (HEAD + 2) * f;
    let u32s = (WORD_HEAD + d.n_kv + rows + cut * d.top_k + cut * d.top_k + cut + 2 * cut)
        * size_of::<u32>();
    let gemm = route_bytes(cut, 1)
        + d.route_n
            .iter()
            .map(|&n| route_bytes(d.top_k * cut, n))
            .sum::<usize>()
        + d.gemm_acts
            .iter()
            .map(|&(k, per_row)| per_row * cut * act_col_bytes(k))
            .sum::<usize>()
        + d.gemm32_acts
            .iter()
            .map(|&(k, per_row)| per_row * cut * act32_col_bytes(k))
            .sum::<usize>()
        + d.native32_acts
            .iter()
            .map(|&(k, per_row)| per_row * cut * native_act_col_bytes(k))
            .sum::<usize>();
    let forms = d
        .slots
        .iter()
        .map(|&(k, cols)| cols * act_col_bytes(k))
        .sum::<usize>()
        + d.run_k
            .iter()
            .map(|&k| COLS * act_col_bytes(k))
            .sum::<usize>()
        + d.run32_k
            .iter()
            .map(|&k| q8_blocks32_bytes(COLS as u64, k as u64) as usize)
            .sum::<usize>()
        + d.wide_k
            .iter()
            .map(|&k| cut * act32_col_bytes(k))
            .sum::<usize>();
    let split = 2 * rows * d.n_kv * SPLIT_GROUP * HEAD * f;
    let mamba2 = rows * d.conv_dim * f + 2 * rows * d.ssm_heads * f;
    f32s + u32s + gemm + forms + split + mamba2
}

/// `item` pushed to `list` unless it is there.
fn add<T: PartialEq>(list: &mut Vec<T>, item: T) {
    if !list.contains(&item) {
        list.push(item);
    }
}

/// Which of the body's two card forms a file's routed stacks take ([`forms_of`]): IQ2_XXS (the
/// `_sel` fallback's q8_1 forms and the grouped GEMM's), and a type the relu² form runs, a
/// 32-value-block type or NVFP4 in its planes (the 32-value GEMM's forms; the relu² form's step
/// buffers are scratch, not the pass's). `fp4` marks the NVFP4 planes among the second: their
/// 32-value GEMM forms may hold the native FP4 columns beside them ([`PassDims::native32_acts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Forms {
    pub iq2: bool,
    pub sel32: bool,
    pub fp4: bool,
}

impl Forms {
    /// These forms and the one a routed up stack of file type `ty` takes: IQ2_XXS counts `iq2`, a
    /// type the relu² form runs ([`card_relu2_runs`]) `sel32` (and `fp4` for NVFP4 and NVFP4_ROWS),
    /// any other type neither.
    #[must_use]
    pub fn with_up(mut self, ty: GgmlType) -> Forms {
        match ty {
            GgmlType::IQ2_XXS => self.iq2 = true,
            ty if card_relu2_runs(ty) => {
                self.sel32 = true;
                self.fp4 |= card_planes_runs(ty);
            }
            _ => {}
        }
        self
    }
}

/// The forms `split`'s routed up stacks take, over every MoE layer of `hp`: a layer whose up is
/// IQ2_XXS counts the first, one of a type the relu² form runs the second ([`Forms::with_up`]). A
/// layer of another type counts neither (its experts stay on the host: the coverage pin names it,
/// and the load refuses a pair of two forms by name).
#[must_use]
pub fn forms_of(split: &Split, hp: &Hparams) -> Forms {
    let mut forms = Forms::default();
    for (l, _) in hp.experts.iter().enumerate().filter(|(_, e)| e.is_some()) {
        let name = crate::arch::names::blk(l, super::names::FFN_UP_EXPS);
        if let Some((_, t)) = split.find(&name) {
            forms = forms.with_up(t.ty);
        }
    }
    forms
}

/// The widths of a placed pass for a file of `hp` whose routed stacks take `forms`, from its
/// headers alone: an upper bound of what the body collects from its loaded layers, since the plan
/// does not know how many experts of each layer the card holds. Every routed layer's table and
/// each form's activations are counted (the body's, only the layers with experts on the card, in
/// their own form), a router table at the layer's whole expert count, and every site width as a
/// gemv run form and as a wide GEMM form (the body's, the NVFP4 and the Q8_0 sites' only): the
/// Mamba-2 in-projection's `hidden` and the out-projection's `d_inner`, the attention's `hidden`
/// and `n_head · value_len`, the latent pair and the shared expert. The q8_1 run form of a width
/// holds at least the 32-value run form an NVFP4 site at a width the q8_1 form does not take
/// reads, so the body's `run32_k` is inside `run_k`; the native FP4 columns are counted for every
/// 32-value GEMM form of an NVFP4 file at a width of whole 64-value steps, as a build with native
/// FP4 products makes them (the plan does not know the build).
#[must_use]
pub fn pass_dims(hp: &Hparams, forms: Forms) -> PassDims {
    let width = hp.experts_width() as usize;
    let top_k = hp.top_k_max();
    let experts = || hp.experts.iter().flatten();
    let mut slots: Vec<(usize, usize)> = Vec::new();
    let mut gemm_acts: Vec<(usize, usize)> = Vec::new();
    let mut gemm32_acts: Vec<(usize, usize)> = Vec::new();
    for e in experts() {
        if forms.iq2 {
            add(&mut slots, (width, COLS));
            add(&mut slots, (e.ff, top_k * COLS));
            add(&mut gemm_acts, (width, 1));
            add(&mut gemm_acts, (e.ff, top_k));
        }
        if forms.sel32 {
            add(&mut gemm32_acts, (width, 1));
            add(&mut gemm32_acts, (e.ff, top_k));
        }
    }
    if hp.shared_ff > 0 {
        add(&mut slots, (hp.n_embd, COLS));
        add(&mut slots, (hp.shared_ff, COLS));
        add(&mut gemm_acts, (hp.n_embd, 1));
        add(&mut gemm_acts, (hp.shared_ff, 1));
    }
    let mut sites: Vec<usize> = Vec::new();
    for k in [
        hp.n_embd,
        hp.ssm.d_inner,
        hp.n_head * hp.value_len,
        width,
        hp.shared_ff,
    ] {
        if k > 0 {
            add(&mut sites, k);
        }
    }
    PassDims {
        hidden: hp.n_embd,
        n_kv: hp.kv.iter().copied().max().unwrap_or(0),
        n_head: hp.n_head,
        in_proj: hp.ssm.d_inner + hp.ssm.conv_dim + hp.ssm.n_heads,
        d_inner: hp.ssm.d_inner,
        ssm_heads: hp.ssm.n_heads,
        conv_dim: hp.ssm.conv_dim,
        width,
        top_k,
        shared_ff: hp.shared_ff,
        ff: experts().map(|e| e.ff).max().unwrap_or(0),
        n_expert: WIDEST_ROUTER,
        slots,
        route_n: experts().map(|e| e.n_expert).collect(),
        gemm_acts,
        native32_acts: if forms.fp4 {
            gemm32_acts
                .iter()
                .copied()
                .filter(|&(k, _)| k.is_multiple_of(GEMM32_STEP))
                .collect()
        } else {
            Vec::new()
        },
        gemm32_acts,
        run_k: sites.clone(),
        run32_k: Vec::new(),
        wide_k: sites,
        placed: true,
    }
}

/// The card bytes the plan keeps out of the expert budget for a load's prompt pass: the pass's
/// buffers at [`PASS_ROWS`] rows of a placed load of a file whose routed stacks take `forms`
/// ([`pass_dims`]). A placed load's pass is exactly these rows (`min(PASS_ROWS, ctx)` rounded down
/// to 64), whose bytes the body holds to this figure; a whole-card load's pass takes the card's
/// room past it, never fewer rows.
#[must_use]
pub fn pass_reserve_bytes(hp: &Hparams, forms: Forms) -> u64 {
    pass_bytes(&pass_dims(hp, forms), PASS_ROWS) as u64
}

/// `split`'s hyperparameters, then its tensors' roles, then its description,
/// the first that fails being the error; then a file with a feature the
/// engine does not run ([`PlacementError::Unimplemented`], every one listed).
pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
    describe(split)?.runnable()
}

/// [`read`] without the refusals: the file as it is.
pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
    let hp = Hparams::read(split)?;
    let read = spec::read(split)?;
    let reserve = pass_reserve_bytes(&hp, forms_of(split, &hp));
    PlanInputs::new(read.tensors, read.spec, kv_layout(&hp), reserve)
}

/// The store each layer of `hp` holds: a Mamba-2 layer its scan, an attention
/// layer its key heads' planes, a MoE layer nothing.
#[must_use]
pub fn kv_layout(hp: &Hparams) -> KvLayout {
    KvLayout::of(
        &hp.ssm,
        hp.key_len,
        hp.kinds
            .iter()
            .zip(&hp.kv)
            .map(|(kind, &kv_heads)| match kind {
                Kind::Mamba => Holds::Scan,
                Kind::Attention => Holds::Attention { kv_heads },
                Kind::Moe => Holds::Nothing,
            }),
    )
}

#[cfg(test)]
mod tests {
    use super::{COLS, Forms, PASS_ROWS, PassDims, act_col_bytes, pass_bytes};
    use crate::placement::kernels::q8_blocks32_bytes;
    use gguf::GgmlType;

    /// Bolt's widths as the plan collects them ([`super::pass_dims`]): 4,096 wide; 32 query heads
    /// over 2 key heads of 128; Mamba-2 of inner 8,192, 128 heads and 10,240 conv channels; routed
    /// experts read and write 1,024 (the latent), 9 a token, of the widths 1,536, 2,048, 2,304 and
    /// 2,560 over 40 layers of 240, 304, 368, 432 and 496 experts (12, 6, 9, 10 and 3 layers); the
    /// shared expert 5,376 wide; the widest router 496.
    fn bolt(placed: bool) -> PassDims {
        let ffs = [1536, 2048, 2304, 2560];
        let mut slots = vec![(1024, 8)];
        let mut gemm_acts = vec![(1024, 1)];
        for ff in ffs {
            slots.push((ff, 72));
            gemm_acts.push((ff, 9));
        }
        slots.extend([(4096, 8), (5376, 8)]);
        gemm_acts.extend([(4096, 1), (5376, 1)]);
        PassDims {
            hidden: 4096,
            n_kv: 2,
            n_head: 32,
            in_proj: 18_560,
            d_inner: 8192,
            ssm_heads: 128,
            conv_dim: 10_240,
            width: 1024,
            top_k: 9,
            shared_ff: 5376,
            ff: 2560,
            n_expert: 496,
            slots,
            route_n: [(240, 12), (304, 6), (368, 9), (432, 10), (496, 3)]
                .into_iter()
                .flat_map(|(n, layers)| std::iter::repeat_n(n, layers))
                .collect(),
            gemm_acts,
            gemm32_acts: Vec::new(),
            native32_acts: Vec::new(),
            run_k: vec![4096, 8192, 1024, 5376],
            run32_k: Vec::new(),
            wide_k: vec![4096, 8192, 1024, 5376],
            placed,
        }
    }

    /// The pass's reserve at the union's 512 rows, 426,763,612 B [derived] from Bolt's widths, four
    /// bytes a value, term by term: the row buffers 230,053,888 (a row 4 x 4,096 residual and sums,
    /// 496 scores, 9 weights, 2 x 1,024 latent and sums, 9 x 2,560 up, 9 x 1,024 down, 2 x 5,376
    /// shared, 4 x 256 key and value, 2 x 4,096 trunk, 18,560 in-projection, 8,192 inner, 2 x 4,096
    /// query and attention, 1,024 host sums; the flash's partials of 8 rows 8 x 32 x 80 x 130
    /// values) + the word record, ids, places and positions 45,068 + the GEMM routes and forms
    /// 151,976,848 (40 tables, and 9 x 512 columns of each expert width, 512 of 1,024, 4,096 and
    /// 5,376) + the slot, run and wide forms 14,803,392 + the flash split's rows 8,388,608 + the
    /// Mamba-2 scratch 21,495,808 (512 x 10,240 conv outputs, 512 x 128 x 2 words). A placed load's
    /// host sums are 2,095,104 of it (512 x 1,024 values less the one-column floor a load with none
    /// keeps). Mutant: a term left out, or the rows counted at 256.
    #[test]
    fn the_pass_reserve_is_the_terms_at_the_unions_columns() {
        assert_eq!(PASS_ROWS, 512);
        assert_eq!(pass_bytes(&bolt(true), PASS_ROWS), 426_763_612);
        assert_eq!(pass_bytes(&bolt(false), PASS_ROWS), 426_763_612 - 2_095_104);
    }

    /// The 32-value GEMM forms of a file whose routed stacks are 32-value types (Lightning's: no
    /// latent, so the experts read the 2,688-wide hidden; 6 slots a token of experts 1,856 wide)
    /// are 8,847,360 B of the pass at the union's 512 rows [derived]: 512 columns of 2,688 values
    /// (42 steps of 64, 80 B a step: 1,720,320) and 6 x 512 of 1,856 (29 steps: 7,127,040).
    /// Mutant: the term left out of `pass_bytes`, or counted at one column a row.
    #[test]
    fn the_32_value_forms_are_their_columns_at_the_unions_rows() {
        let mut d = bolt(true);
        let base = pass_bytes(&d, PASS_ROWS);
        d.gemm32_acts = vec![(2688, 1), (1856, 6)];
        assert_eq!(pass_bytes(&d, PASS_ROWS) - base, 8_847_360);
    }

    /// The native FP4 columns beside Lightning's 32-value GEMM forms are 3,995,648 B of the pass
    /// at the union's 512 rows [derived]: a column of `k` values holds `k / 2` bytes of codes,
    /// `k / 16` of block scales and one f32 scale, 1,516 B at 2,688 and 1,048 B at 1,856, over
    /// 512 columns and 6 x 512. Mutant: the codes counted a byte a value, or the term left out
    /// of `pass_bytes`.
    #[test]
    fn the_native_fp4_columns_are_their_columns_at_the_unions_rows() {
        let mut d = bolt(true);
        d.gemm32_acts = vec![(2688, 1), (1856, 6)];
        let base = pass_bytes(&d, PASS_ROWS);
        d.native32_acts = d.gemm32_acts.clone();
        assert_eq!(pass_bytes(&d, PASS_ROWS) - base, 3_995_648);
    }

    /// The 32-value run forms of Lightning's NVFP4 shared expert (a site at 2,688 and at 3,712
    /// values, widths the q8_1 form does not take) are 70,144 B of the pass [derived]: 8 columns
    /// of 3,744 B and of 5,024 B. The plan's q8_1 run forms at the same widths, which it counts
    /// for every site width, hold at least as much, so the plan's reserve covers them. Mutant:
    /// the term left out of `pass_bytes`.
    #[test]
    fn the_32_value_run_forms_are_inside_the_plans_q8_1_run_forms() {
        let mut d = bolt(true);
        let base = pass_bytes(&d, PASS_ROWS);
        d.run32_k = vec![2688, 3712];
        assert_eq!(pass_bytes(&d, PASS_ROWS) - base, 8 * 3_744 + 8 * 5_024);
        for k in [2688, 3712] {
            let blocks32 = q8_blocks32_bytes(COLS as u64, k as u64) as usize;
            assert!(COLS * act_col_bytes(k) >= blocks32, "k = {k}");
        }
    }

    /// A routed up stack counts the form its file type takes: IQ2_XXS the first, the four
    /// 32-value-block types the second, NVFP4 and NVFP4_ROWS the second and the native FP4
    /// columns, any other type none. Mutant: the rule over `card_sel32_runs`, which counts NVFP4
    /// and NVFP4_ROWS none.
    #[test]
    fn a_routed_up_counts_the_form_its_type_takes() {
        let of = |ty| Forms::default().with_up(ty);
        let iq2 = Forms {
            iq2: true,
            ..Forms::default()
        };
        let sel32 = Forms {
            sel32: true,
            ..Forms::default()
        };
        let fp4 = Forms {
            sel32: true,
            fp4: true,
            ..Forms::default()
        };
        assert_eq!(of(GgmlType::IQ2_XXS), iq2);
        for ty in [
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
        ] {
            assert_eq!(of(ty), sel32, "{ty}");
        }
        for ty in [GgmlType::NVFP4, GgmlType::NVFP4_ROWS] {
            assert_eq!(of(ty), fp4, "{ty}");
        }
        assert_eq!(of(GgmlType::Q4_K), Forms::default());
    }
}
