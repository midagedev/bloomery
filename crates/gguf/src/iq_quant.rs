//! IQ1_S and IQ2_XXS quantization with an importance matrix: ggml's
//! `quantize_row_iq1_s_impl` and `quantize_row_iq2_xxs_impl`
//! (`ggml/src/ggml-quants.c`), step for step, so the bytes are ggml's.
//!
//! The rules are written once, generic over the arithmetic ([`Fp`]): every
//! multiply, add, subtract, divide and square root is its own call, rounded
//! once to nearest even, in ggml's evaluation order (C's left to right). The
//! reference build compiles `ggml-quants.c` for baseline x86-64 (no FMA, no
//! F16C), so no product is fused into a sum there; the host instance
//! ([`Ieee`]) is Rust's plain operators, which never contract, and the card's
//! instance spells each operation as its `.rn` instruction. Comparisons, `MAX`
//! and `MIN` are ggml's (`a > b ? a : b`), and every search keeps ggml's
//! candidate order with its strict tie-breaks.
//!
//! The work splits where ggml's data flow does: a 32-value sub-block's search
//! ([`iq1_s_sub`], [`iq2_xxs_sub`]) reads only its own values, its weights
//! and its super-block's `sigma2` ([`sigma2_iq1_s`], [`sigma2_iq2_xxs`]); the
//! super-block's `d` and the sub-blocks' scale codes ([`iq1_s_finish`],
//! [`iq2_xxs_finish`]) read the eight sub-blocks' results. A sub-block that
//! ggml would abort on (a point the search leaves off the grid, a NaN scale)
//! returns a nonzero [`Sub::status`] instead of plausible bytes.
//!
//! The neighbour tables ([`GridTables`]) are ggml's `iq2xs_init_impl`
//! output, derived from the dequant grids this crate already owns
//! ([`crate::iq_tables`]): each grid point's 2-bit codes, the map from a code
//! word to its grid index (or to its neighbour list), and the lists, each the
//! points at the `nwant` smallest squared distances in `(distance, index)`
//! order.

use crate::iq_tables::{IQ1S_GRID, IQ2XXS_GRID};

/// Values per super-block (ggml's `QK_K`).
pub const QK: usize = 256;
/// Values per sub-block: one search, one scale.
pub const SUB: usize = 32;
/// Sub-blocks per super-block.
pub const SUBS: usize = QK / SUB;
/// Bytes of one IQ1_S super-block (`block_iq1_s`).
pub const IQ1_S_BYTES: usize = 50;
/// Bytes of one IQ2_XXS super-block (`block_iq2_xxs`).
pub const IQ2_XXS_BYTES: usize = 66;
/// The code-word space ggml's map covers (`kmap_size`).
pub const KMAP_SIZE: usize = 43692;

/// `GROUP_MAX_EPS`: a sub-block whose largest magnitude is below it is zero.
const EPS_IQ2_XXS: f32 = 1e-15;
/// `GROUP_MAX_EPS_IQ1_S`.
const EPS_IQ1_S: f32 = 1e-12;
/// `IQ1S_DELTA`.
const IQ1S_DELTA: f32 = 0.125;

/// [`Sub::status`]: the search left a group's code word off the grid (ggml
/// aborts with "found point not on grid").
pub const STATUS_OFF_GRID: u32 = 1;
/// [`Sub::status`]: the sub-block's scale is NaN (ggml's `scale >= 0` assert).
pub const STATUS_SCALE: u32 = 2;
/// [`Sub::status`]: no neighbour scored below `FLT_MAX` (ggml aborts, or
/// reads past the group in IQ1_S's whole-grid fallback).
pub const STATUS_NO_NEIGHBOUR: u32 = 4;
/// [`Sub::status`]: a value or weight of the sub-block is NaN.
pub const STATUS_NAN: u32 = 8;

/// The arithmetic the rules run on: each call one IEEE operation rounded to
/// nearest even, never fused with another.
pub trait Fp {
    fn mul(a: f32, b: f32) -> f32;
    fn add(a: f32, b: f32) -> f32;
    fn sub(a: f32, b: f32) -> f32;
    fn div(a: f32, b: f32) -> f32;
    fn sqrt(a: f32) -> f32;
}

/// The host's arithmetic: Rust's operators, which LLVM never contracts.
pub struct Ieee;

impl Fp for Ieee {
    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        a - b
    }
    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        a / b
    }
    #[inline(always)]
    fn sqrt(a: f32) -> f32 {
        a.sqrt()
    }
}

/// ggml's `nearest_int`: `x + 1.5·2^23` rounded once, its low mantissa bits
/// read as the integer (exact for |x| ≤ 2^22 − 1).
#[inline(always)]
pub fn nearest_int<F: Fp>(x: f32) -> i32 {
    let v = F::add(x, 12_582_912.0);
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// ggml's `MAX(a, b)`: `a > b ? a : b`.
#[inline(always)]
fn gmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Which grid a quantizer searches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IqKind {
    Iq2Xxs,
    Iq1S,
}

impl IqKind {
    /// Grid points.
    pub fn grid_size(self) -> usize {
        match self {
            IqKind::Iq2Xxs => 256,
            IqKind::Iq1S => 2048,
        }
    }

    /// Distinct distances each neighbour list spans (ggml's `nwant`).
    fn nwant(self) -> usize {
        match self {
            IqKind::Iq2Xxs => 2,
            IqKind::Iq1S => 3,
        }
    }

    /// Bytes of one super-block.
    pub fn block_bytes(self) -> usize {
        match self {
            IqKind::Iq2Xxs => IQ2_XXS_BYTES,
            IqKind::Iq1S => IQ1_S_BYTES,
        }
    }
}

/// ggml's `iq2xs_init_impl` output for one grid, as words a kernel reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridTables {
    /// Grid point `g`'s eight 2-bit codes `l`, value `k` at bits `2k..`; the
    /// point's coordinates are `2l + 1`.
    pub codes: Vec<u32>,
    /// Code word `u` → its grid index, or `-(offset + 1)` where its
    /// neighbour list starts at `neighbours[offset]`.
    pub map: Vec<i32>,
    /// Lists `[n, j_1 .. j_n]`.
    pub neighbours: Vec<u32>,
}

/// One grid point's eight 2-bit codes from a dequant grid entry: IQ2_XXS's
/// magnitudes `0x08, 0x19, 0x2b` and IQ1_S's values `-1, 0, 1` are codes
/// 0, 1, 2 in order.
fn point_codes(kind: IqKind, entry: u64) -> u32 {
    (0..8).fold(0u32, |acc, k| {
        let b = (entry >> (8 * k)) as u8;
        let l = match (kind, b) {
            (IqKind::Iq2Xxs, 0x08) | (IqKind::Iq1S, 0xff) => 0,
            (IqKind::Iq2Xxs, 0x19) | (IqKind::Iq1S, 0x00) => 1,
            (IqKind::Iq2Xxs, 0x2b) | (IqKind::Iq1S, 0x01) => 2,
            _ => panic!("{kind:?} grid byte {b:#04x} is no 2-bit code"),
        };
        acc | (l << (2 * k))
    })
}

impl GridTables {
    /// The tables of `kind`'s grid, as `iq2xs_init_impl` builds them.
    pub fn new(kind: IqKind) -> GridTables {
        let codes: Vec<u32> = match kind {
            IqKind::Iq2Xxs => IQ2XXS_GRID.iter().map(|&e| point_codes(kind, e)).collect(),
            IqKind::Iq1S => IQ1S_GRID.iter().map(|&e| point_codes(kind, e)).collect(),
        };
        let mut map = vec![-1i32; KMAP_SIZE];
        for (g, &c) in codes.iter().enumerate() {
            map[c as usize] = g as i32;
        }
        let mut neighbours = Vec::new();
        let mut by_d2: Vec<Vec<u32>> = Vec::new();
        for i in 0..KMAP_SIZE {
            if map.get(i).is_some_and(|&m| m >= 0) {
                continue;
            }
            // Every grid point's squared distance, bucketed by distance with
            // indices ascending: the buckets in order are ggml's qsort by
            // (distance, index).
            by_d2.iter_mut().for_each(Vec::clear);
            for (j, &c) in codes.iter().enumerate() {
                let d2: u32 = (0..8)
                    .map(|k| {
                        let a = ((i >> (2 * k)) & 3) as i32;
                        let b = ((c >> (2 * k)) & 3) as i32;
                        let d = 2 * (b - a);
                        (d * d) as u32
                    })
                    .sum();
                if by_d2.len() <= d2 as usize {
                    by_d2.resize(d2 as usize + 1, Vec::new());
                }
                by_d2[d2 as usize].push(j as u32);
            }
            map[i] = -(neighbours.len() as i32 + 1);
            let start = neighbours.len();
            neighbours.push(0);
            for bucket in by_d2.iter().filter(|b| !b.is_empty()).take(kind.nwant()) {
                neighbours.extend_from_slice(bucket);
            }
            neighbours[start] = (neighbours.len() - start - 1) as u32;
        }
        GridTables {
            codes,
            map,
            neighbours,
        }
    }

    /// The tables as the borrowed view the rules read.
    pub fn view(&self) -> Tables<'_> {
        Tables {
            codes: &self.codes,
            map: &self.map,
            neighbours: &self.neighbours,
        }
    }
}

/// [`GridTables`] borrowed: host vectors or a kernel's argument slices.
#[derive(Clone, Copy)]
pub struct Tables<'a> {
    pub codes: &'a [u32],
    pub map: &'a [i32],
    pub neighbours: &'a [u32],
}

impl Tables<'_> {
    /// Code `k` (0..8) of grid point `g`.
    #[inline(always)]
    fn code(&self, g: usize, k: usize) -> i32 {
        ((self.codes[g] >> (2 * k)) & 3) as i32
    }

    /// The map entry of code word `u`; `None` past the map.
    #[inline(always)]
    fn lookup(&self, u: u32) -> Option<i32> {
        self.map.get(u as usize).copied()
    }
}

/// One sub-block's search result. IQ2_XXS: `w0` the four grid indices (byte
/// `k` group `k`), `w1` the four 7-bit sign words at bits `7k..`. IQ1_S: `w0`
/// the four grid indices' low bytes, `w1` the `qh` half (index high bits at
/// `3k..`) with bit 16 set for the negative delta. `scale` is the sub-block's
/// scale before the super-block's code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sub {
    pub w0: u32,
    pub w1: u32,
    pub scale: f32,
    pub status: u32,
}

impl Sub {
    const ZERO: Sub = Sub {
        w0: 0,
        w1: 0,
        scale: 0.0,
        status: 0,
    };
}

/// The serial sum of squares over a super-block, ggml's `sumx2`.
#[inline(always)]
fn sumx2<F: Fp>(xbl: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in &xbl[..QK] {
        s = F::add(s, F::mul(x, x));
    }
    s
}

/// IQ2_XXS's `sigma2 = sumx2 / QK_K` of the super-block `xbl` (256 values).
#[inline(always)]
pub fn sigma2_iq2_xxs<F: Fp>(xbl: &[f32]) -> f32 {
    F::div(sumx2::<F>(xbl), QK as f32)
}

/// IQ1_S's `sigma2 = 2 · sumx2 / QK_K`.
#[inline(always)]
pub fn sigma2_iq1_s<F: Fp>(xbl: &[f32]) -> f32 {
    F::div(F::mul(2.0, sumx2::<F>(xbl)), QK as f32)
}

/// `weight[i] = qw[i] · sqrt(sigma2 + x[i]²)`.
#[inline(always)]
fn weights<F: Fp>(xb: &[f32; SUB], qw: &[f32; SUB], sigma2: f32) -> [f32; SUB] {
    let mut w = [0.0f32; SUB];
    for i in 0..SUB {
        w[i] = F::mul(qw[i], F::sqrt(F::add(sigma2, F::mul(xb[i], xb[i]))));
    }
    w
}

/// Whether any value or weight is NaN.
#[inline(always)]
fn any_nan(xb: &[f32; SUB], qw: &[f32; SUB]) -> bool {
    let mut bad = false;
    for i in 0..SUB {
        bad |= xb[i].is_nan() | qw[i].is_nan();
    }
    bad
}

/// ggml's `make_qp_quants(32, nmax, x, L, w)`: the scale only (IQ2_XXS reads
/// no `L` it leaves). `L` is ggml's `uint8_t` array: a negative level (the
/// sign-flipped value of an odd group) is stored wrapped, and every read of
/// `L` sees the wrapped value while the loops' own `l` stays signed.
#[inline(always)]
fn make_qp_quants<F: Fp>(nmax: i32, x: &[f32; SUB], qw: &[f32; SUB]) -> f32 {
    let mut max = 0.0f32;
    for &v in x {
        max = gmax(max, v);
    }
    if max < EPS_IQ2_XXS {
        return 0.0;
    }
    let nmaxf = nmax as f32;
    let mut l = [0u8; SUB];
    let mut iscale = F::div(nmaxf, max);
    for i in 0..SUB {
        l[i] = nearest_int::<F>(F::mul(iscale, x[i])) as u8;
    }
    let scale = F::div(1.0, iscale);
    let mut best_mse = 0.0f32;
    for i in 0..SUB {
        let diff = F::sub(x[i], F::mul(scale, f32::from(l[i])));
        best_mse = F::add(best_mse, F::mul(F::mul(qw[i], diff), diff));
    }
    for is in -4i32..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = F::div(F::add(F::mul(0.1, is as f32), nmaxf), max);
        let scale_is = F::div(1.0, iscale_is);
        let mut mse = 0.0f32;
        for i in 0..SUB {
            let li = nearest_int::<F>(F::mul(iscale_is, x[i])).min(nmax);
            let diff = F::sub(x[i], F::mul(scale_is, li as f32));
            mse = F::add(mse, F::mul(F::mul(qw[i], diff), diff));
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..SUB {
        let li = nearest_int::<F>(F::mul(iscale, x[i])).min(nmax);
        l[i] = li as u8;
        let lf = li as f32;
        sumlx = F::add(sumlx, F::mul(F::mul(qw[i], x[i]), lf));
        suml2 = F::add(suml2, F::mul(F::mul(qw[i], lf), lf));
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..SUB {
            let w = qw[i];
            let li = f32::from(l[i]);
            let mut slx = F::sub(sumlx, F::mul(F::mul(w, x[i]), li));
            let mut sl2 = F::sub(suml2, F::mul(F::mul(w, li), li));
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int::<F>(F::div(F::mul(x[i], sl2), slx)).min(nmax);
                if new_l != i32::from(l[i]) {
                    let nl = new_l as f32;
                    slx = F::add(slx, F::mul(F::mul(w, x[i]), nl));
                    sl2 = F::add(sl2, F::mul(F::mul(w, nl), nl));
                    if F::mul(F::mul(slx, slx), suml2) > F::mul(F::mul(sumlx, sumlx), sl2) {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        F::div(sumlx, suml2)
    } else {
        0.0
    }
}

/// ggml's `iq2_find_best_neighbour` over group `k` of a sub-block: the list's
/// first point of least weighted squared distance to `xval / scale`; its
/// codes go to `l`. `None` when no point scores below `FLT_MAX`.
#[inline(always)]
fn iq2_best_neighbour<F: Fp>(
    t: Tables<'_>,
    start: usize,
    xval: &[f32; SUB],
    waux: &[f32; SUB],
    k: usize,
    scale: f32,
    l: &mut [i32; SUB],
) -> Option<usize> {
    let n = t.neighbours[start] as usize;
    let mut best_d2 = f32::MAX;
    let mut best: Option<usize> = None;
    for j in 1..=n {
        let g = t.neighbours[start + j] as usize;
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = (2 * t.code(g, i) + 1) as f32;
            let diff = F::sub(F::mul(scale, q), xval[8 * k + i]);
            d2 = F::add(d2, F::mul(F::mul(waux[8 * k + i], diff), diff));
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(g);
        }
    }
    let g = best?;
    for i in 0..8 {
        l[8 * k + i] = t.code(g, i);
    }
    Some(g)
}

/// Group `k`'s code word at `1/scale`: `nearest_int(0.5·(id·x − 1))` clamped to
/// 0..=2, eight 2-bit fields.
#[inline(always)]
fn iq2_word<F: Fp>(xval: &[f32; SUB], k: usize, id: f32, l: &mut [i32; SUB]) -> u32 {
    let mut u = 0u32;
    for i in 0..8 {
        let v = nearest_int::<F>(F::mul(0.5, F::sub(F::mul(id, xval[8 * k + i]), 1.0)));
        let c = v.clamp(0, 2);
        l[8 * k + i] = c;
        u |= (c as u32) << (2 * i);
    }
    u
}

/// `Σ w·x·q` and `Σ w·q²` with `q = 2l + 1`, ggml's serial order.
#[inline(always)]
fn iq2_sums<F: Fp>(w: &[f32; SUB], xval: &[f32; SUB], l: &[i32; SUB]) -> (f32, f32) {
    let mut sumqx = 0.0f32;
    let mut sumq2 = 0.0f32;
    for i in 0..SUB {
        let q = (2 * l[i] + 1) as f32;
        sumqx = F::add(sumqx, F::mul(F::mul(w[i], xval[i]), q));
        sumq2 = F::add(sumq2, F::mul(F::mul(w[i], q), q));
    }
    (sumqx, sumq2)
}

/// One IQ2_XXS sub-block: values `xb`, importance `qw`, the super-block's
/// [`sigma2_iq2_xxs`].
#[inline(always)]
pub fn iq2_xxs_sub<F: Fp>(xb: &[f32; SUB], qw: &[f32; SUB], sigma2: f32, t: Tables<'_>) -> Sub {
    const KMAXQ: i32 = 3;
    if any_nan(xb, qw) {
        return Sub {
            status: STATUS_NAN,
            ..Sub::ZERO
        };
    }
    let weight = weights::<F>(xb, qw, sigma2);
    let mut waux = [0.0f32; SUB];
    for i in 0..SUB {
        waux[i] = F::sqrt(weight[i]);
    }
    let mut xval = [0.0f32; SUB];
    let mut block_signs = [0u32; 4];
    for k in 0..4 {
        let mut nflip = 0;
        let mut s = 0u32;
        for i in 0..8 {
            let v = xb[8 * k + i];
            if v >= 0.0 {
                xval[8 * k + i] = v;
            } else {
                xval[8 * k + i] = -v;
                nflip += 1;
                s |= 1 << i;
            }
        }
        if nflip % 2 == 1 {
            let sq = |i: usize| F::mul(F::mul(weight[8 * k + i], xb[8 * k + i]), xb[8 * k + i]);
            let mut imin = 0;
            let mut min = sq(0);
            for i in 1..8 {
                let ax = sq(i);
                if ax < min {
                    min = ax;
                    imin = i;
                }
            }
            xval[8 * k + imin] = -xval[8 * k + imin];
            s ^= 1 << imin;
        }
        block_signs[k] = s & 127;
    }
    let mut max = xval[0];
    for &v in &xval[1..] {
        max = gmax(max, v);
    }
    if max < EPS_IQ2_XXS {
        return Sub::ZERO;
    }
    let mut scale = make_qp_quants::<F>(KMAXQ + 1, &xval, &weight);
    let eff_max = F::mul(scale, KMAXQ as f32);
    if eff_max <= 0.0 {
        return Sub::ZERO;
    }
    let mut l = [0i32; SUB];
    let mut laux = [0i32; SUB];
    let mut best = 0.0f32;
    for is in -6i32..=6 {
        let id = F::div(
            F::add((2 * KMAXQ - 1) as f32, F::mul(is as f32, 0.1)),
            eff_max,
        );
        let this_scale = F::div(1.0, id);
        for k in 0..4 {
            let u = iq2_word::<F>(&xval, k, id, &mut laux);
            match t.lookup(u) {
                Some(g) if g >= 0 => {}
                Some(g) => {
                    let start = (-g - 1) as usize;
                    if iq2_best_neighbour::<F>(t, start, &xval, &waux, k, this_scale, &mut laux)
                        .is_none()
                    {
                        return Sub {
                            status: STATUS_NO_NEIGHBOUR,
                            ..Sub::ZERO
                        };
                    }
                }
                None => {
                    return Sub {
                        status: STATUS_OFF_GRID,
                        ..Sub::ZERO
                    };
                }
            }
        }
        let (sumqx, sumq2) = iq2_sums::<F>(&weight, &xval, &laux);
        if sumq2 > 0.0 && F::mul(sumqx, sumqx) > F::mul(best, sumq2) {
            scale = F::div(sumqx, sumq2);
            best = F::mul(scale, sumqx);
            l = laux;
        }
    }
    if scale > 0.0 {
        let id = F::div(1.0, scale);
        for k in 0..4 {
            let u = iq2_word::<F>(&xval, k, id, &mut l);
            let g = match t.lookup(u) {
                Some(g) if g >= 0 => g as usize,
                Some(g) => {
                    let start = (-g - 1) as usize;
                    match iq2_best_neighbour::<F>(t, start, &xval, &waux, k, scale, &mut l) {
                        Some(g) => g,
                        None => {
                            return Sub {
                                status: STATUS_NO_NEIGHBOUR,
                                ..Sub::ZERO
                            };
                        }
                    }
                }
                None => {
                    return Sub {
                        status: STATUS_OFF_GRID,
                        ..Sub::ZERO
                    };
                }
            };
            for i in 0..8 {
                l[8 * k + i] = t.code(g, i);
            }
        }
        let (sumqx, sumq2) = iq2_sums::<F>(&weight, &xval, &l);
        if sumq2 > 0.0 {
            scale = F::div(sumqx, sumq2);
        }
    }
    if scale < 0.0 {
        scale = -scale;
        for s in &mut block_signs {
            *s = !*s & 127;
        }
    }
    let mut w0 = 0u32;
    let mut w1 = 0u32;
    for k in 0..4 {
        let mut u = 0u32;
        for i in 0..8 {
            u |= ((l[8 * k + i] as u32) & 0xff) << (2 * i);
        }
        let g = match t.lookup(u) {
            Some(g) if g >= 0 => g as u32,
            _ => {
                return Sub {
                    status: STATUS_OFF_GRID,
                    ..Sub::ZERO
                };
            }
        };
        w0 |= g << (8 * k);
        w1 |= block_signs[k] << (7 * k);
    }
    if scale.is_nan() {
        return Sub {
            status: STATUS_SCALE,
            ..Sub::ZERO
        };
    }
    Sub {
        w0,
        w1,
        scale,
        status: 0,
    }
}

/// ggml's `iq1_find_best_neighbour2` over group `k`: the list's first point
/// of least weighted squared distance between `scale · xx[l]` and the signed
/// values; its codes go to `l`. `None` when no point scores below `FLT_MAX`
/// (ggml's whole-grid fallback there reads past the group).
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "ggml's argument list, one search over one group"
)]
fn iq1_best_neighbour<F: Fp>(
    t: Tables<'_>,
    start: usize,
    xb: &[f32; SUB],
    weight: &[f32; SUB],
    k: usize,
    scale: f32,
    xx: &[f32; 3],
    l: &mut [i32; SUB],
) -> Option<usize> {
    let n = t.neighbours[start] as usize;
    let mut best_d2 = f32::MAX;
    let mut best: Option<usize> = None;
    for j in 1..=n {
        let g = t.neighbours[start + j] as usize;
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = xx[t.code(g, i) as usize];
            let diff = F::sub(F::mul(scale, q), xb[8 * k + i]);
            d2 = F::add(d2, F::mul(F::mul(weight[8 * k + i], diff), diff));
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(g);
        }
    }
    let g = best?;
    for i in 0..8 {
        l[8 * k + i] = t.code(g, i);
    }
    Some(g)
}

/// The values' stable ascending order (ggml's `qsort` on glibc, a merge sort:
/// equal values keep their index order; `-0.0` equals `0.0`).
#[inline(always)]
fn stable_order(xb: &[f32; SUB]) -> [u8; SUB] {
    let mut idx = [0u8; SUB];
    for (j, v) in idx.iter_mut().enumerate() {
        *v = j as u8;
    }
    for j in 1..SUB {
        let key = idx[j];
        let mut p = j;
        while p > 0 && xb[idx[p - 1] as usize] > xb[key as usize] {
            idx[p] = idx[p - 1];
            p -= 1;
        }
        idx[p] = key;
    }
    idx
}

/// One IQ1_S sub-block: values `xb`, importance `qw`, the super-block's
/// [`sigma2_iq1_s`].
#[inline(always)]
pub fn iq1_s_sub<F: Fp>(xb: &[f32; SUB], qw: &[f32; SUB], sigma2: f32, t: Tables<'_>) -> Sub {
    const X_P: [f32; 3] = [-1.0 + IQ1S_DELTA, IQ1S_DELTA, 1.0 + IQ1S_DELTA];
    const X_M: [f32; 3] = [-1.0 - IQ1S_DELTA, -IQ1S_DELTA, 1.0 - IQ1S_DELTA];
    if any_nan(xb, qw) {
        return Sub {
            status: STATUS_NAN,
            ..Sub::ZERO
        };
    }
    let weight = weights::<F>(xb, qw, sigma2);
    let mut max = xb[0].abs();
    for &v in &xb[1..] {
        max = gmax(max, v.abs());
    }
    if max < EPS_IQ1_S {
        return Sub::ZERO;
    }
    let idx = stable_order(xb);
    let mut sumx = [0.0f32; SUB + 1];
    let mut sumw = [0.0f32; SUB + 1];
    for j in 0..SUB {
        let i = idx[j] as usize;
        sumx[j + 1] = F::add(sumx[j], F::mul(weight[i], xb[i]));
        sumw[j + 1] = F::add(sumw[j], weight[i]);
    }
    let mut best_score = -f32::MAX;
    let mut scale = max;
    let mut besti1 = -1i32;
    let mut besti2 = -1i32;
    let mut best_shift = 0i32;
    for i1 in 0..=SUB {
        for i2 in i1..=SUB {
            let a = F::sub(sumx[i1], sumx[0]);
            let b = F::sub(sumx[i2], sumx[i1]);
            let c = F::sub(sumx[SUB], sumx[i2]);
            let wa = F::sub(sumw[i1], sumw[0]);
            let wb = F::sub(sumw[i2], sumw[i1]);
            let wc = F::sub(sumw[SUB], sumw[i2]);
            for (xs, shift) in [(&X_P, 1i32), (&X_M, -1i32)] {
                let sumqx = F::add(F::add(F::mul(a, xs[0]), F::mul(b, xs[1])), F::mul(c, xs[2]));
                let sumq2 = F::add(
                    F::add(
                        F::mul(F::mul(wa, xs[0]), xs[0]),
                        F::mul(F::mul(wb, xs[1]), xs[1]),
                    ),
                    F::mul(F::mul(wc, xs[2]), xs[2]),
                );
                if sumq2 > 0.0 && F::mul(sumqx, sumqx) > F::mul(best_score, sumq2) {
                    scale = F::div(sumqx, sumq2);
                    best_score = F::mul(scale, sumqx);
                    besti1 = i1 as i32;
                    besti2 = i2 as i32;
                    best_shift = shift;
                }
            }
        }
    }
    if besti1 < 0 || besti2 < 0 || best_shift == 0 {
        return Sub::ZERO;
    }
    let mut l = [0i32; SUB];
    for (j, &i) in idx.iter().enumerate() {
        let j = j as i32;
        l[i as usize] = if j < besti1 {
            0
        } else if j < besti2 {
            1
        } else {
            2
        };
    }
    if scale < 0.0 {
        for v in &mut l {
            *v = 2 - *v;
        }
        scale = -scale;
        best_shift = -best_shift;
    }
    let xx = if best_shift == 1 { &X_P } else { &X_M };
    let mut index = [0u32; 4];
    let mut all_on_grid = true;
    for k in 0..4 {
        let mut u = 0u32;
        for j in 0..8 {
            u |= (l[8 * k + j] as u32) << (2 * j);
        }
        index[k] = match t.lookup(u) {
            Some(g) if g >= 0 => g as u32,
            Some(g) => {
                all_on_grid = false;
                let start = (-g - 1) as usize;
                match iq1_best_neighbour::<F>(t, start, xb, &weight, k, scale, xx, &mut l) {
                    Some(g) => g as u32,
                    None => {
                        return Sub {
                            status: STATUS_NO_NEIGHBOUR,
                            ..Sub::ZERO
                        };
                    }
                }
            }
            None => {
                return Sub {
                    status: STATUS_OFF_GRID,
                    ..Sub::ZERO
                };
            }
        };
    }
    if !all_on_grid {
        let mut sumqx = 0.0f32;
        let mut sumq2 = 0.0f32;
        for k in 0..4 {
            for j in 0..8 {
                let w = weight[8 * k + j];
                let q = xx[t.code(index[k] as usize, j) as usize];
                sumqx = F::add(sumqx, F::mul(F::mul(w, q), xb[8 * k + j]));
                sumq2 = F::add(sumq2, F::mul(F::mul(w, q), q));
            }
        }
        if sumqx > 0.0 && sumq2 > 0.0 {
            scale = F::div(sumqx, sumq2);
        }
    }
    let mut w0 = 0u32;
    let mut h = 0u32;
    for (k, &g) in index.iter().enumerate() {
        w0 |= (g & 255) << (8 * k);
        h |= (g >> 8) << (3 * k);
    }
    if scale.is_nan() {
        return Sub {
            status: STATUS_SCALE,
            ..Sub::ZERO
        };
    }
    Sub {
        w0,
        w1: h | (u32::from(best_shift == -1) << 16),
        scale,
        status: 0,
    }
}

/// The OR of the eight sub-blocks' statuses.
#[inline(always)]
fn status_of(subs: &[Sub; SUBS]) -> u32 {
    subs.iter().fold(0, |a, s| a | s.status)
}

/// ggml's `max_scale`: `MAX` over the sub-block scales in order from 0.
#[inline(always)]
fn max_scale(subs: &[Sub; SUBS]) -> f32 {
    let mut m = 0.0f32;
    for s in subs {
        m = gmax(m, s.scale);
    }
    m
}

/// An IQ2_XXS super-block's `d` (f16 bits) and its 16 code words
/// (`block_iq2_xxs::qs` as u32s), from its eight sub-blocks.
#[inline(always)]
pub fn iq2_xxs_finish<F: Fp>(subs: &[Sub; SUBS]) -> (u16, [u32; 2 * SUBS]) {
    let mut q2 = [0u32; 2 * SUBS];
    let m = max_scale(subs);
    if m == 0.0 {
        return (0, q2);
    }
    let d = F::div(m, 31.0);
    let id = F::div(1.0, d);
    for (ib, s) in subs.iter().enumerate() {
        let l = nearest_int::<F>(F::mul(0.5, F::sub(F::mul(id, s.scale), 1.0))).clamp(0, 15);
        q2[2 * ib] = s.w0;
        q2[2 * ib + 1] = s.w1 | ((l as u32) << 28);
    }
    (crate::quant::f32_to_f16_bits(d), q2)
}

/// An IQ1_S super-block's `d` (f16 bits), its 32 index bytes as eight words
/// and its eight `qh` halves, from its eight sub-blocks.
#[inline(always)]
pub fn iq1_s_finish<F: Fp>(subs: &[Sub; SUBS]) -> (u16, [u32; SUBS], [u16; SUBS]) {
    let mut qs = [0u32; SUBS];
    let mut qh = [0u16; SUBS];
    for (ib, s) in subs.iter().enumerate() {
        qs[ib] = s.w0;
        qh[ib] = (s.w1 & 0xffff) as u16;
    }
    let m = max_scale(subs);
    if m == 0.0 {
        return (0, qs, qh);
    }
    let d = F::div(m, 15.0);
    let id = F::div(1.0, d);
    for (ib, s) in subs.iter().enumerate() {
        let mut l = nearest_int::<F>(F::mul(0.5, F::sub(F::mul(id, s.scale), 1.0))).clamp(0, 7);
        if (s.w1 >> 16) & 1 == 1 {
            l |= 8;
        }
        qh[ib] |= (l as u16) << 12;
    }
    (crate::quant::f32_to_f16_bits(F::mul(d, 1.125)), qs, qh)
}

/// A quantizer's refusal of a row it cannot quantize as ggml would.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IqQuantError {
    #[error("row length {n} is not a multiple of {QK}")]
    RowLength { n: usize },
    #[error("importance row holds {got} values for a row of {n}")]
    Weights { got: usize, n: usize },
    #[error(
        "super-block {block}: status {status:#x} (1 off grid, 2 NaN scale, 4 no neighbour, 8 NaN input)"
    )]
    Status { block: usize, status: u32 },
}

/// One super-block's eight sub-blocks.
fn subs_of<F: Fp>(kind: IqKind, xbl: &[f32], qwbl: &[f32], t: Tables<'_>) -> [Sub; SUBS] {
    let sigma2 = match kind {
        IqKind::Iq2Xxs => sigma2_iq2_xxs::<F>(xbl),
        IqKind::Iq1S => sigma2_iq1_s::<F>(xbl),
    };
    let mut subs = [Sub::ZERO; SUBS];
    for (ib, s) in subs.iter_mut().enumerate() {
        let mut xb = [0.0f32; SUB];
        let mut qw = [0.0f32; SUB];
        xb.copy_from_slice(&xbl[SUB * ib..SUB * (ib + 1)]);
        qw.copy_from_slice(&qwbl[SUB * ib..SUB * (ib + 1)]);
        *s = match kind {
            IqKind::Iq2Xxs => iq2_xxs_sub::<F>(&xb, &qw, sigma2, t),
            IqKind::Iq1S => iq1_s_sub::<F>(&xb, &qw, sigma2, t),
        };
    }
    subs
}

/// One super-block's file bytes from its sub-blocks, appended to `out`.
pub fn block_bytes<F: Fp>(kind: IqKind, subs: &[Sub; SUBS], out: &mut Vec<u8>) {
    match kind {
        IqKind::Iq2Xxs => {
            let (d, q2) = iq2_xxs_finish::<F>(subs);
            out.extend_from_slice(&d.to_le_bytes());
            for w in q2 {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        IqKind::Iq1S => {
            let (d, qs, qh) = iq1_s_finish::<F>(subs);
            out.extend_from_slice(&d.to_le_bytes());
            for w in qs {
                out.extend_from_slice(&w.to_le_bytes());
            }
            for h in qh {
                out.extend_from_slice(&h.to_le_bytes());
            }
        }
    }
}

/// One row of `x` quantized with importance `qw` (one value per column) into
/// ggml's block bytes, on the host.
pub fn quantize_row(
    kind: IqKind,
    x: &[f32],
    qw: &[f32],
    t: Tables<'_>,
) -> Result<Vec<u8>, IqQuantError> {
    let n = x.len();
    if !n.is_multiple_of(QK) {
        return Err(IqQuantError::RowLength { n });
    }
    if qw.len() != n {
        return Err(IqQuantError::Weights { got: qw.len(), n });
    }
    let mut out = Vec::with_capacity(n / QK * kind.block_bytes());
    for (block, (xbl, qwbl)) in x
        .as_chunks::<QK>()
        .0
        .iter()
        .zip(qw.as_chunks::<QK>().0)
        .enumerate()
    {
        let subs = subs_of::<Ieee>(kind, xbl, qwbl, t);
        let status = status_of(&subs);
        if status != 0 {
            return Err(IqQuantError::Status { block, status });
        }
        block_bytes::<Ieee>(kind, &subs, &mut out);
    }
    Ok(out)
}

/// What an importance matrix holds for one tensor that a quantizer cannot use.
#[derive(Debug, thiserror::Error)]
pub enum ImatrixError {
    #[error("imatrix has no {name}.in_sum2 or no {name}.counts")]
    Missing { name: String },
    #[error("imatrix {tensor} is {ty:?}, not F32")]
    Type {
        tensor: String,
        ty: crate::quant::GgmlType,
    },
    #[error("imatrix {name}: {sums} sums do not split into {counts} counts")]
    Split {
        name: String,
        sums: usize,
        counts: usize,
    },
    #[error("imatrix {name}.{what}: entry {index} is {value}, not finite")]
    NotFiniteInput {
        name: String,
        what: &'static str,
        index: usize,
        value: f32,
    },
    #[error("imatrix {name}.{what}: entry {index} is {value}, negative")]
    Negative {
        name: String,
        what: &'static str,
        index: usize,
        value: f32,
    },
    #[error("imatrix {name}: value {index} is {value}, not finite")]
    NotFinite {
        name: String,
        index: usize,
        value: f32,
    },
    #[error("imatrix {name}: {source}")]
    Load {
        name: String,
        #[source]
        source: crate::LoadError,
    },
}

/// One F32 tensor of an imatrix file as values.
fn f32_tensor(m: &crate::Gguf, tensor: &str, name: &str) -> Result<Vec<f32>, ImatrixError> {
    let t = m.find(tensor).ok_or_else(|| ImatrixError::Missing {
        name: name.to_string(),
    })?;
    if t.ty != crate::quant::GgmlType::F32 {
        return Err(ImatrixError::Type {
            tensor: tensor.to_string(),
            ty: t.ty,
        });
    }
    let bytes = m.data(t).map_err(|e| ImatrixError::Load {
        name: name.to_string(),
        source: e,
    })?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// The importance values llama-quantize hands its quantizer for tensor
/// `name` from a GGUF imatrix (`load_imatrix`, `tools/quantize/quantize.cpp`):
/// `<name>.in_sum2` divided per expert by that expert's `<name>.counts`
/// entry (rounded with `lround`, then made a float), or all ones for an
/// expert counted zero times. `ne0 · n_expert` values, expert `e`'s at
/// `e · ne0`. A sum or count that is not finite or is negative is refused by
/// name and entry (a sum of squares and a call count are neither), and so is a
/// quotient that is not finite, as llama-quantize refuses the file.
pub fn imatrix_weights(m: &crate::Gguf, name: &str) -> Result<Vec<f32>, ImatrixError> {
    let sums = f32_tensor(m, &format!("{name}.in_sum2"), name)?;
    let counts = f32_tensor(m, &format!("{name}.counts"), name)?;
    for (what, v) in [("in_sum2", &sums), ("counts", &counts)] {
        if let Some((index, &value)) = v.iter().enumerate().find(|(_, x)| !x.is_finite()) {
            return Err(ImatrixError::NotFiniteInput {
                name: name.to_string(),
                what,
                index,
                value,
            });
        }
        if let Some((index, &value)) = v.iter().enumerate().find(|(_, x)| **x < 0.0) {
            return Err(ImatrixError::Negative {
                name: name.to_string(),
                what,
                index,
                value,
            });
        }
    }
    if counts.is_empty() || sums.len() % counts.len() != 0 {
        return Err(ImatrixError::Split {
            name: name.to_string(),
            sums: sums.len(),
            counts: counts.len(),
        });
    }
    let ne0 = sums.len() / counts.len();
    let mut e = vec![0.0f32; sums.len()];
    for (j, &c) in counts.iter().enumerate() {
        let count = c.round() as i64 as f32;
        for i in 0..ne0 {
            e[j * ne0 + i] = if count > 0.0 {
                sums[j * ne0 + i] / count
            } else {
                1.0
            };
        }
    }
    if let Some((index, &value)) = e.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(ImatrixError::NotFinite {
            name: name.to_string(),
            index,
            value,
        });
    }
    Ok(e)
}

#[cfg(test)]
mod tests;
