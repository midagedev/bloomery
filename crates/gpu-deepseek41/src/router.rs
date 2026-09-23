//! Expert routing: each expert scores √softplus(logit); a per-expert bias is
//! added for the selection only; the top 6 of 384 experts are kept, and their
//! unbiased scores are renormalized to sum 1 and scaled by 1.5.
