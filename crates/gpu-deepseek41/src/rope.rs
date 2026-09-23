//! Rotary position embedding on the rope tail of each head, and its inverse
//! on the tail of the attention output (the same rotation at the negated
//! angle). The layer picks the base: plain rope on the window-only layers,
//! YaRN-scaled on the compressed ones.
