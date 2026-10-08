//! Per projector type, what the encoder file holds: its tensor names and its hyperparameters.
//!
//! An encoder file is an mmproj GGUF (architecture `clip`); `clip.projector_type` names the tower
//! and projector it carries, and a module here is named by that string. Each module reads the
//! header once and refuses, by key or by tensor name, anything its encoder does not run.
//!
//! What every projector type does the same way has one owner here, and a module holds its tables
//! as data: [`header`] (the key checks and their refusals), [`naming`] (the tensor-name scheme)
//! and [`table`] (the tensor table and the check of a file against it).

pub mod deepseek41v;
pub(crate) mod header;
pub(crate) mod naming;
pub mod qwen3vl;
pub mod table;
