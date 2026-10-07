//! What a server needs of each decision model to seat it: the route its requests are posted to, the
//! context its prompts fill, the backbone architectures it reads the hidden states of, and how a
//! model file says it holds the model's head. One module a model: [`clef`] is a joint head over a
//! Qwen3.5 backbone (a head file, or llama.cpp's Clef layout of the file), [`lev`] a label readout of
//! the file's own language-model head. The server's own table holds one row of these a model.

pub mod clef;
pub mod lev;
