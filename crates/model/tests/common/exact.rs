//! The integer comparison the routing gates share.

/// Routing decisions are integers and get no tolerance. A wrong expert choice inside a 1e-3
/// numeric gate is invisible, and it is the failure that matters most in the MoE block.
pub fn assert_exact_i32(got: &[i32], want_f32: &[f32], what: &str) {
    assert_eq!(
        got.len(),
        want_f32.len(),
        "{what}: length {} vs reference {}",
        got.len(),
        want_f32.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want_f32).enumerate() {
        let w = w as i32;
        assert_eq!(g, w, "{what}: index {i} chose {g}, reference chose {w}");
    }
    eprintln!("{what:38} exact match on {} ids   ok", got.len());
}
