//! A refusal that is a named panic, read back by the gates that hold it.

/// The text of the panic `call` raises, `None` when it returns. A panic on a
/// pool worker arrives with its own text: the pool replays the first one.
pub fn panic_text(call: &mut dyn FnMut()) -> Option<String> {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default(),
    )
}
