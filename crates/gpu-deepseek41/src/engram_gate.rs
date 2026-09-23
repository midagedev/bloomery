//! The engram gate: the looked-up rows, projected into a key per stream copy
//! and one shared value, gate the residual update — gate = sigmoid of the
//! signed square root of the query–key dot, and h += gate ⊗ value.
