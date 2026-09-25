//! The model file every gate opens. One owner — the gates must all open the
//! same file the oracle came from.

/// `$BLOOMERY_MODEL`, or the box's default path.
pub fn model_path() -> String {
    std::env::var("BLOOMERY_MODEL")
        .unwrap_or_else(|_| "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf".into())
}
