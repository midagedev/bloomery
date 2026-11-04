//! [`HostTarget`]: the deepseek2 host forward as a `runtime::Target`, so the one generation loop
//! and the server's `TargetEngine` run the model on the CPU with no card. A prompt call is
//! [`forward::step`] over all its ids, a step the same over one id: the cache (`KvCache`) takes
//! the positions in order, and a batch of any split gives the same bits as one id a step.

use gguf::Gguf;
use runtime::{Out, Target, Want};

use super::derived::Derived;
use super::forward::{argmax, new_cache, step};
use crate::ModelError;
use crate::kv::KvCache;

/// Why a call of the host target failed.
#[derive(Debug, thiserror::Error)]
pub enum TargetError {
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A prompt of no ids: the forward needs a position to read back after.
    #[error("a prompt call of no ids")]
    NoIds,
    /// The call would pass the positions the target holds.
    #[error("{ids} ids at position {pos} pass the target's {ctx} positions")]
    Context { pos: u32, ids: usize, ctx: u32 },
    /// A `--ctx` past the context the file was trained for.
    #[error("a context of {asked} positions passes the file's trained {trained}")]
    CtxPastTrained { asked: u32, trained: u32 },
    /// The file states no (or an unusable) trained context.
    #[error("metadata {key}: missing or past u32")]
    NoContext { key: String },
    /// A cut past the positions held.
    #[error("a cut to position {n} past the target's position {pos}")]
    CutPast { n: u32, pos: u32 },
}

/// The deepseek2 forward on the host: the file, its load-time weight state and one sequence's
/// cache.
pub struct HostTarget {
    gguf: Gguf,
    derived: Derived,
    cache: KvCache,
    ctx: u32,
    /// The last call's logits row, which [`Want::Logits`] lends.
    row: Vec<f32>,
    head_rows: usize,
}

impl HostTarget {
    /// The target over `gguf` holding `ctx` positions, or the file's trained context when
    /// `None`; a `ctx` past the trained one is refused by name.
    pub fn new(gguf: Gguf, ctx: Option<u32>) -> Result<HostTarget, TargetError> {
        let key = "context_length";
        let trained = gguf
            .arch_get_u64(key)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| TargetError::NoContext {
                key: format!("{}.{key}", gguf.architecture().unwrap_or("<missing>")),
            })?;
        let ctx = match ctx {
            Some(asked) if asked > trained => {
                return Err(TargetError::CtxPastTrained { asked, trained });
            }
            Some(asked) => asked,
            None => trained,
        };
        let derived = Derived::new(&gguf)?;
        let cache = new_cache(&gguf)?;
        let head_rows = derived
            .plan()
            .head
            .out_w
            .dims
            .get(1)
            .and_then(|&n| usize::try_from(n).ok())
            .ok_or_else(|| ModelError::MissingTensor("output.weight".to_owned()))?;
        Ok(HostTarget {
            gguf,
            derived,
            cache,
            ctx,
            row: Vec::new(),
            head_rows,
        })
    }

    /// The logits row length: the head's rows, which the server holds to the vocabulary's count.
    #[must_use]
    pub fn head_rows(&self) -> usize {
        self.head_rows
    }

    fn run(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, TargetError> {
        if ids.is_empty() {
            return Err(TargetError::NoIds);
        }
        let pos = self.pos();
        if u64::from(pos) + ids.len() as u64 > u64::from(self.ctx) {
            return Err(TargetError::Context {
                pos,
                ids: ids.len(),
                ctx: self.ctx,
            });
        }
        let logits = step(&self.gguf, ids, &mut self.cache, &self.derived)?;
        let best = argmax(&logits.data);
        Ok(match want {
            Want::Argmax => Out::Argmax(best),
            Want::Logits => {
                self.row.clear();
                self.row.extend_from_slice(&logits.data);
                Out::Logits {
                    argmax: best,
                    row: &self.row,
                }
            }
        })
    }
}

impl Target for HostTarget {
    type Error = TargetError;

    fn pos(&self) -> u32 {
        self.cache.next_pos(0)
    }

    fn ctx(&self) -> u32 {
        self.ctx
    }

    fn prompt(&mut self, ids: &[u32], want: Want) -> Result<Out<'_>, TargetError> {
        self.run(ids, want)
    }

    fn step(&mut self, id: u32, want: Want) -> Result<Out<'_>, TargetError> {
        self.run(&[id], want)
    }

    fn keepable(&self, n: u32) -> u32 {
        n.min(self.pos())
    }

    fn cut(&mut self, n: u32) -> Result<(), TargetError> {
        let pos = self.pos();
        if n > pos {
            return Err(TargetError::CutPast { n, pos });
        }
        self.cache.truncate(n as usize);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), TargetError> {
        self.cache.truncate(0);
        Ok(())
    }
}
