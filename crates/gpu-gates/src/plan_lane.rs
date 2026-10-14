//! A serve seat's lane probe. `--plan` returns before the load, so it does not
//! probe. A load probes through the closure the seat passes, which keeps the
//! probe where that seat placed the call.

/// `probe` runs only for a load. A plan-only run returns `None` and does not
/// call it.
pub fn probed_unless_plan<T, E>(
    plan_only: bool,
    probe: impl FnOnce() -> Result<T, E>,
) -> Result<Option<T>, E> {
    if plan_only {
        Ok(None)
    } else {
        Ok(Some(probe()?))
    }
}

#[cfg(test)]
mod tests {
    use super::probed_unless_plan;

    #[test]
    fn plan_only_does_not_probe() {
        let mut probed = false;
        let skipped = probed_unless_plan(true, || {
            probed = true;
            Ok::<(), ()>(())
        })
        .expect("plan-only");
        assert!(skipped.is_none(), "a plan-only run kept a lane");
        assert!(!probed, "a plan-only run probed");
        let loaded = probed_unless_plan(false, || {
            probed = true;
            Ok::<(), ()>(())
        })
        .expect("load");
        assert!(loaded.is_some(), "a load dropped its lane");
        assert!(probed, "a load did not probe");
    }
}
