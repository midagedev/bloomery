//! Walk A: the row walk the K-quants with a 4-bit plane share (Q4_K, Q5_K).
//!
//! A row is `n_sb` super-blocks of `D::WORDS` u32 words. Lane `lane` of the
//! row's warp owns sub-block `s = lane & 7` of super-block `4·it + (lane >>
//! 3)` in iteration `it`, so one iteration covers four super-blocks. Per
//! super-block the format's decoder ([`SbDecode::decode`]) yields the lane's
//! eight code words (four signed bytes each, in the q8_1 q4 plane's
//! permutation: word `i` holds values `32·s + 4·i ..`) and the two chain
//! coefficients; everything after that is the format-free part: the A chain
//! `dp4a(codes, q8)` against the q4 plane, the B term (the sub-block's q8
//! sum from the s8 plane), and the block scale from the d8 plane, as
//! `(A·cda + B·cdb)·d8`.
//!
//! Both walks here are `cores::q4k_row_dot_1col` and `cores::q4k_row_dot`
//! with the decoder and the super-block stride as parameters: the same
//! loads, the same iteration pairs ([`Q4K_ITER_UNROLL`]) and the same
//! rounding ([`q4k_acc`]), so [`Q4k`] through this walk is `q4k_gemv_sel`'s
//! arithmetic bit for bit, and column `c` of [`row_dot`] is [`row_dot_1col`]
//! on that column bit for bit, for every format. Keep the pair-unrolled
//! loop and the term's expression as they are: the backend's contraction
//! of the term into its accumulator follows the basic blocks around it
//! (`cores::q4k_acc` says which terms fuse).

use crate::cores::{Q4K_ITER_UNROLL, q4k_a_chain, q4k_acc, q4k_sb_decode, q4k_sub_value};

/// One K-quant format's super-block decode for Walk A.
pub trait SbDecode {
    /// u32 words of one super-block (its bytes over 4).
    const WORDS: usize;

    /// The super-block at word `wk` of `w`, for sub-block `s` (0..8): the
    /// eight code words of that sub-block (signed bytes, word `i` holding
    /// values `32·s + 4·i .. + 4`) and the chain coefficients `(cda, cdb)`
    /// with which the sub-block's dot is `cda·Σ code·q + cdb·Σ q` before its
    /// q8_1 block scale.
    ///
    /// # Safety
    ///
    /// `wk + WORDS <= w.len()` and `s < 8`.
    unsafe fn decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32);
}

/// Q4_K (144 bytes: `d`, `dmin`, 12 scale bytes, 128 nibble bytes): the
/// decode is `cores::q4k_sb_decode`, the one every Q4_K gemv of this crate
/// runs. Its codes are the nibbles minus 8, so its `cdb` carries `8·cda`.
pub struct Q4k;

impl SbDecode for Q4k {
    const WORDS: usize = 36;

    #[inline(always)]
    unsafe fn decode(w: &[u32], wk: usize, s: usize) -> ([u32; 8], f32, f32) {
        // The core's contract is this fn's: wk + 35 is inside `w`.
        q4k_sb_decode(w, wk, s)
    }
}

/// One iteration's term of the single-column walk: super-block `4·it +
/// grp`'s decode, the A chain against the column whose planes start at
/// `qb0`/`s8b0`/`d8b0`, and the sub-block's value times its block scale —
/// the one value [`row_dot_1col`] adds. Every load the iteration makes is
/// here and none of the walk's accumulation is.
///
/// # Safety
///
/// `4·it + grp < n_sb`, `s = lane & 7`, `grp = lane >> 3`, `lane < 32`, and
/// the buffer bounds of [`row_dot_1col`] for the column at `qb0`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn iter_term<D: SbDecode>(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    it: u32,
    row_abs: usize,
    qb0: usize,
    s8b0: usize,
    d8b0: usize,
    s: usize,
    grp: usize,
    lane: usize,
) -> f32 {
    let sbp = 4 * it as usize + grp;
    // SAFETY: sbp < n_sb by this fn's contract, so the super-block's words
    // are inside row `row_abs` of `w`.
    let (vi, cda, cdb) = unsafe { D::decode(w, row_abs * D::WORDS * n_sb + D::WORDS * sbp, s) };
    let qb = 256 * it as usize + lane;
    let s8b = 32 * it as usize + lane;
    let d8b = 2 * sbp + (s >> 2);
    let a = q4k_a_chain(&vi, q, qb0 + qb);
    // SAFETY: sbp < n_sb keeps s8b < 8·n_sb inside the column's s8 words.
    let b = unsafe { *s8.get_unchecked(s8b0 + s8b) };
    // SAFETY: sbp < n_sb keeps d8b < 2·n_sb inside the column's d8 words.
    let e0 = unsafe { *d8.get_unchecked(d8b0 + d8b) };
    (a as f32 * cda + b as f32 * cdb) * e0
}

/// One row's dot with one q8_1 column, lane `lane`'s partial (the caller
/// reduces over the warp): [`row_dot`]'s column 0 with the column guards
/// gone. The iterations every lane takes run [`Q4K_ITER_UNROLL`] at a time,
/// their loads issued before the first add; the guarded walk is the tail.
///
/// # Safety
///
/// As [`row_dot`] with `m_cols` 1.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn row_dot_1col<D: SbDecode>(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    lane: usize,
) -> f32 {
    let q_col = 256 * iters as usize; // q8 words per column
    let s8_col = 8 * n_sb; // 32-value groups per column
    let d8_col = 2 * n_sb; // 128-value blocks per column

    let s = lane & 7; // sub-block within the super-block
    let grp = lane >> 3; // super-block within this iteration (0..4)

    let mut f0 = 0.0f32;

    let qb0 = col0 * q_col;
    let s8b0 = col0 * s8_col;
    let d8b0 = col0 * d8_col;

    // Iterations every lane takes: 4·it + grp <= 4·full − 1 <= n_sb − 1.
    let full = (n_sb / 4) as u32;
    let mut it: u32 = 0;
    while it + Q4K_ITER_UNROLL <= full {
        // SAFETY: it + Q4K_ITER_UNROLL <= full, so each term's super-block
        // is inside the row; the rest is this fn's contract.
        let t0 = unsafe {
            iter_term::<D>(
                w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
            )
        };
        // SAFETY: as t0, at it + 1.
        let t1 = unsafe {
            iter_term::<D>(
                w,
                q,
                s8,
                d8,
                n_sb,
                it + 1,
                row_abs,
                qb0,
                s8b0,
                d8b0,
                s,
                grp,
                lane,
            )
        };
        f0 += t0;
        f0 += t1;
        it += Q4K_ITER_UNROLL;
    }
    while it < iters {
        let sbp = 4 * it as usize + grp;
        // The guard makes a partial final iteration safe; it always holds
        // when n_sb is a multiple of 4.
        if sbp < n_sb {
            // SAFETY: the guard is this term's super-block bound.
            f0 += unsafe {
                iter_term::<D>(
                    w, q, s8, d8, n_sb, it, row_abs, qb0, s8b0, d8b0, s, grp, lane,
                )
            };
        }
        it += 1;
    }
    f0
}

/// Adds column `$c`'s share of one iteration to `$f[$c]` when `$c < $m`
/// (column 0 unguarded): its A chain at `$qb + $c·$q_col`, its B sum and
/// block scale, folded by [`q4k_acc`] with `$split`.
macro_rules! fold_cols {
    ($f:ident, $m:expr, ($vi:expr, $q:expr, $s8:expr, $d8:expr), ($qb:expr, $s8b:expr, $d8b:expr),
     ($q_col:expr, $s8_col:expr, $d8_col:expr), ($cda:expr, $cdb:expr), $split:expr) => {
        // SAFETY: column 0 < m_cols; the bounds are the caller's.
        let (x, e) = unsafe { col_term($vi, $q, $s8, $d8, $qb, $s8b, $d8b, $cda, $cdb) };
        $f[0] = q4k_acc($f[0], x, e, $split);
        fold_cols!(@each $f, $m, ($vi, $q, $s8, $d8), ($qb, $s8b, $d8b), ($q_col, $s8_col, $d8_col),
                   ($cda, $cdb), $split, 1 2 3 4 5 6 7);
    };
    (@each $f:ident, $m:expr, ($vi:expr, $q:expr, $s8:expr, $d8:expr), ($qb:expr, $s8b:expr, $d8b:expr),
     ($q_col:expr, $s8_col:expr, $d8_col:expr), ($cda:expr, $cdb:expr), $split:expr, $($c:literal)*) => {
        $(
            if $m > $c {
                // SAFETY: column $c < m_cols; the bounds are the caller's.
                let (x, e) = unsafe {
                    col_term(
                        $vi,
                        $q,
                        $s8,
                        $d8,
                        $qb + $c * $q_col,
                        $s8b + $c * $s8_col,
                        $d8b + $c * $d8_col,
                        $cda,
                        $cdb,
                    )
                };
                $f[$c] = q4k_acc($f[$c], x, e, $split);
            }
        )*
    };
}

/// One column's share of an m-column iteration: its A chain against the q8
/// window at `qb`, its B sum at `s8b` and block scale at `d8b`, as the
/// sub-block value ([`q4k_sub_value`]) and the scale [`q4k_acc`] folds.
///
/// # Safety
///
/// `qb + 224 < q.len()`, `s8b < s8.len()`, `d8b < d8.len()`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn col_term(
    vi: &[u32; 8],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    qb: usize,
    s8b: usize,
    d8b: usize,
    cda: f32,
    cdb: f32,
) -> (f32, f32) {
    let a = q4k_a_chain(vi, q, qb);
    // SAFETY: both indices are inside their buffers by this fn's contract.
    let (b, e) = unsafe { (*s8.get_unchecked(s8b), *d8.get_unchecked(d8b)) };
    (q4k_sub_value(a, b, cda, cdb), e)
}

/// One row's dots with `m_cols` (1..=8) q8_1 columns, lane `lane`'s
/// partials (the caller reduces each over the warp): per iteration the
/// super-block is decoded once and every live column's term folded into
/// its accumulator with [`row_dot_1col`]'s rounding, so column `c` is
/// [`row_dot_1col`] on column `col0 + c` bit for bit and each weight word
/// is read once for all `m_cols` columns. One column is the decode shape
/// and takes [`row_dot_1col`] itself.
///
/// Layouts: row `row_abs` of `w` is `D::WORDS · n_sb` words at `row_abs ·
/// D::WORDS · n_sb`; column `c` of the activation is `256 · iters` words of
/// `q` (the q4 plane), `8 · n_sb` of `s8` and `2 · n_sb` of `d8`.
///
/// # Safety
///
/// `w.len() >= (row_abs + 1) · D::WORDS · n_sb`, `q.len() >= (col0 +
/// m_cols) · 256 · iters`, `s8.len() >= (col0 + m_cols) · 8 · n_sb`,
/// `d8.len() >= (col0 + m_cols) · 2 · n_sb`, `iters = ceil(n_sb / 4)`,
/// `1 <= m_cols <= 8` warp-uniform, and all 32 lanes of one warp call it.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn row_dot<D: SbDecode>(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    n_sb: usize,
    iters: u32,
    row_abs: usize,
    col0: usize,
    m_cols: usize,
    lane: usize,
) -> [f32; 8] {
    if m_cols == 1 {
        // SAFETY: this fn's contract at m_cols 1.
        let f0 = unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb, iters, row_abs, col0, lane) };
        return [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let q_col = 256 * iters as usize;
    let s8_col = 8 * n_sb;
    let d8_col = 2 * n_sb;
    let s = lane & 7;
    let grp = lane >> 3;
    let full = (n_sb / 4) as u32;
    let mut f = [0.0f32; 8];
    let mut it: u32 = 0;
    while it < iters {
        let sbp = 4 * it as usize + grp;
        // Guarded lanes load nothing in a partial final iteration.
        if sbp < n_sb {
            // [`row_dot_1col`]'s rounding: the first iteration of each pair
            // of the guard-free prefix splits.
            let split = it & 1 == 0 && it + Q4K_ITER_UNROLL <= full;
            // SAFETY: sbp < n_sb, so the super-block is inside the row.
            let (vi, cda, cdb) =
                unsafe { D::decode(w, row_abs * D::WORDS * n_sb + D::WORDS * sbp, s) };
            let qb = col0 * q_col + 256 * it as usize + lane;
            let s8b = col0 * s8_col + 32 * it as usize + lane;
            let d8b = col0 * d8_col + 2 * sbp + (s >> 2);
            // Every column c < m_cols reads q words below (col0 + c + 1)·q_col
            // (the largest in quad group g is 256g + 255), and its s8 and d8
            // entries sit inside its column because sbp < n_sb.
            fold_cols!(
                f,
                m_cols,
                (&vi, q, s8, d8),
                (qb, s8b, d8b),
                (q_col, s8_col, d8_col),
                (cda, cdb),
                split
            );
        }
        it += 1;
    }
    f
}
