//! The float comparison the numeric gates share.

/// Elementwise comparison that reports WHERE it failed, not just that it did. A gate that
/// prints only a max is a gate you cannot act on.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs reference {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max |diff| = {worst:e} at index {at} (got {}, reference {}); gate is {tol:e}",
        got[at],
        want[at]
    );
    eprintln!("{what:38} max|diff| = {worst:e}   ok (gate {tol:e})");
}
