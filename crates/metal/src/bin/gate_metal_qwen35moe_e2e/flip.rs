//! The rules this gate holds a free-running chain to are `refset::flip`'s,
//! the one owner both backends' gates read: the relative distance a layer
//! output is read at, the router's flip test and the rows a near-tie flip
//! carries, and the head's tie band — re-exported here under the paths the
//! gate's modules read them at. The argmax over ik's row stays here: the
//! CUDA gates' owner of it (`bin/shared/e2e.rs`) serves every one of them.

pub use refset::flip::{
    RouteFlip, head_cap, head_gap, head_tie, held_rows, margin, picks_its_top, rel_base, rel_to,
    router_tie, worse,
};

/// The index of the largest value; the first on a tie.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    u32::try_from(best).expect("a row index fits u32")
}
