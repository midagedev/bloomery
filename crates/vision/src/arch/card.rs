//! The encoder's bytes on its card, as every projector type states them: what a plan reserves
//! for the tower and what the loaded tower holds itself to.
//!
//! A projector module computes its own figure from its hyperparameters (`deepseek41v::card::of`,
//! `qwen3vl::card::of`); the figure's shape is the same for all of them.

/// The encoder's bytes on its card for a file of given hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardBytes {
    /// The weights the encoder uploads.
    pub weights: u64,
    /// The activations it allocates.
    pub scratch: u64,
}

impl CardBytes {
    /// The figure a plan reserves for the encoder.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.weights + self.scratch
    }
}
