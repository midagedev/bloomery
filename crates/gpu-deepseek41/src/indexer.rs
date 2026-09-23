//! The indexer: per-head queries from the low-rank query, per-head weights
//! from the hidden state, scores Σ relu(q · k) · weight over the cached index
//! keys, and the top 512 compressed positions.
