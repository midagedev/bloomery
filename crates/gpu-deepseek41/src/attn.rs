//! Attention over the key set window rows ⧺ compressed prefix, with K = V:
//! one latent row per key is both its key and its value. Each head's learned
//! sink joins the softmax denominator. The variant over the rows the indexer
//! selects belongs to this module as well.
