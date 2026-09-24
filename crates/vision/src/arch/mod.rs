//! Per projector type, what the encoder file holds: its tensor names and its hyperparameters.
//!
//! An encoder file is an mmproj GGUF (architecture `clip`); `clip.projector_type` names the tower
//! and projector it carries, and a module here is named by that string. Each module reads the
//! header once and refuses, by key or by tensor name, anything its encoder does not run.

pub mod deepseek41v;
