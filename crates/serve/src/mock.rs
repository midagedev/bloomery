//! A model-free engine for the server's gates: a byte-level vocabulary plus the
//! special strings the V4.1 chat template emits, and a bigram echo for "inference".
//!
//! Vocabulary: ids `0..SPECIALS.len()` are the special strings, then one id per
//! byte. A byte id is `SPECIALS.len() + byte`, so any UTF-8 text round-trips and a
//! multi-byte character spans several tokens (the streaming decoder must hold it).
//!
//! Next token: look up the most recent earlier occurrence of the last token in the
//! whole context; the token that followed it is the prediction (logit 4), every
//! other token that ever followed the last token gets logit 2, the rest -8. A last
//! token never seen before predicts EOS. Deterministic, and seed-sensitive once a
//! sampler with temperature > 0 is in the loop.
//!
//! [`MockEngine::failing_at`] makes the `k`-th `next` of the engine's life an
//! error, for the crash-path gate.

use std::sync::Arc;

use crate::engine::{Decoder, Engine, EngineError, Tokenizer};

/// Special strings, in id order. Longest-match wins on encode.
pub const SPECIALS: [&str; 6] = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
];

const N_SPECIAL: u32 = SPECIALS.len() as u32;
const PREDICTED: f32 = 4.0;
const FOLLOWER: f32 = 2.0;
const FLOOR: f32 = -8.0;

/// The mock engine. `ctx_max` is chosen by the caller so the gate can hit the
/// context limit cheaply.
pub struct MockEngine {
    ctx: Vec<u32>,
    ctx_max: usize,
    tok: Arc<MockTokenizer>,
    nexts: usize,
    fail_at: Option<usize>,
}

impl MockEngine {
    /// A mock with room for `ctx_max` positions.
    #[must_use]
    pub fn new(ctx_max: usize) -> Self {
        MockEngine {
            ctx: Vec::new(),
            ctx_max,
            tok: Arc::new(MockTokenizer),
            nexts: 0,
            fail_at: None,
        }
    }

    /// A mock whose `k`-th call to `next` (counted from 1 over its whole life)
    /// fails with an `EngineError`.
    #[must_use]
    pub fn failing_at(ctx_max: usize, k: usize) -> Self {
        MockEngine {
            fail_at: Some(k),
            ..MockEngine::new(ctx_max)
        }
    }

    fn push(&mut self, id: u32) -> Result<(), EngineError> {
        if self.ctx.len() >= self.ctx_max {
            return Err(EngineError(format!(
                "mock: position {} is past ctx_max {}",
                self.ctx.len(),
                self.ctx_max
            )));
        }
        self.ctx.push(id);
        Ok(())
    }
}

fn token_bytes(id: u32) -> Vec<u8> {
    match usize::try_from(id).ok().and_then(|i| SPECIALS.get(i)) {
        Some(s) => s.as_bytes().to_vec(),
        None => id
            .checked_sub(N_SPECIAL)
            .and_then(|b| u8::try_from(b).ok())
            .map(|b| vec![b])
            .unwrap_or_default(),
    }
}

/// The mock's vocabulary (see the module header).
pub struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let special = SPECIALS
                .iter()
                .enumerate()
                .filter(|(_, s)| bytes[i..].starts_with(s.as_bytes()))
                .max_by_key(|(_, s)| s.len());
            match special {
                Some((id, s)) => {
                    out.push(u32::try_from(id).expect("SPECIALS has six entries"));
                    i += s.len();
                }
                None => {
                    out.push(N_SPECIAL + u32::from(bytes[i]));
                    i += 1;
                }
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(Utf8Decoder { held: Vec::new() })
    }

    fn bos(&self) -> u32 {
        0
    }

    fn eos(&self) -> u32 {
        1
    }

    fn add_bos(&self) -> bool {
        false
    }

    fn n_vocab(&self) -> usize {
        SPECIALS.len() + 256
    }
}

impl Engine for MockEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tok.clone()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        ids.iter().try_for_each(|&id| self.push(id))
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.nexts += 1;
        if self.fail_at == Some(self.nexts) {
            return Err(EngineError(format!(
                "mock: injected failure at next #{} (position {})",
                self.nexts,
                self.ctx.len()
            )));
        }
        self.push(last)?;
        let mut logits = logits_out;
        if let Some(l) = logits.as_deref_mut() {
            l.fill(FLOOR);
        }
        let mut raise = |id: u32, v: f32| {
            if let Some(slot) = logits
                .as_deref_mut()
                .and_then(|l| l.get_mut(usize::try_from(id).ok()?))
            {
                *slot = slot.max(v);
            }
        };
        let n = self.ctx.len();
        let mut predicted = MockTokenizer.eos();
        let mut found = false;
        for j in (0..n - 1).rev() {
            if self.ctx[j] != last {
                continue;
            }
            let follower = self.ctx[j + 1];
            raise(follower, FOLLOWER);
            if !found {
                predicted = follower;
                found = true;
            }
        }
        raise(predicted, PREDICTED);
        Ok(predicted)
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.ctx.clear();
        Ok(())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        format!("mock position={}", self.ctx.len())
    }
}

/// An engine that answers every prompt with the same text, then EOS: the model
/// output a parser gate scripts. Tokens are [`MockTokenizer`]'s, so the special
/// strings are single ids and everything else goes byte by byte.
pub struct ScriptedEngine {
    script: Vec<u32>,
    at: usize,
    pos: usize,
    ctx_max: usize,
}

impl ScriptedEngine {
    /// An engine whose every generation is `text` followed by EOS.
    #[must_use]
    pub fn new(ctx_max: usize, text: &str) -> Self {
        ScriptedEngine {
            script: MockTokenizer.encode(text),
            at: 0,
            pos: 0,
            ctx_max,
        }
    }
}

impl Engine for ScriptedEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::new(MockTokenizer)
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.pos += ids.len();
        Ok(())
    }

    fn next(&mut self, _last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        if self.pos >= self.ctx_max {
            return Err(EngineError(format!(
                "scripted: position {} is past ctx_max {}",
                self.pos, self.ctx_max
            )));
        }
        self.pos += 1;
        let id = self
            .script
            .get(self.at)
            .copied()
            .unwrap_or(MockTokenizer.eos());
        self.at += 1;
        if let Some(l) = logits_out {
            l.fill(FLOOR);
            if let Some(slot) = usize::try_from(id).ok().and_then(|i| l.get_mut(i)) {
                *slot = PREDICTED;
            }
        }
        Ok(id)
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.at = 0;
        self.pos = 0;
        Ok(())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        format!("scripted position={}", self.pos)
    }
}

/// Byte-accumulating decoder: emits the longest valid UTF-8 prefix it holds.
struct Utf8Decoder {
    held: Vec<u8>,
}

impl Decoder for Utf8Decoder {
    fn push(&mut self, id: u32) -> Option<String> {
        self.held.extend(token_bytes(id));
        let valid = match std::str::from_utf8(&self.held) {
            Ok(_) => self.held.len(),
            Err(e) if e.error_len().is_some() => {
                // An invalid (not merely incomplete) sequence: give up on it.
                let s = String::from_utf8_lossy(&self.held).into_owned();
                self.held.clear();
                return Some(s);
            }
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return None;
        }
        let rest = self.held.split_off(valid);
        let done = std::mem::replace(&mut self.held, rest);
        String::from_utf8(done).ok()
    }

    fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.held).into_owned();
        self.held.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip_with_specials_and_multibyte() {
        let m = MockTokenizer;
        let text = "<｜User｜>héllo</think>";
        let ids = m.encode(text);
        assert_eq!(ids[0], 2);
        assert_eq!(*ids.last().expect("non-empty"), 5);
        assert_eq!(m.decode(&ids), text);
        let mut d = m.decoder();
        let streamed: String = ids.iter().filter_map(|&id| d.push(id)).collect();
        assert_eq!(streamed, text);
    }

    #[test]
    fn bigram_echo_and_eos() {
        let mut m = MockEngine::new(64);
        let t = MockTokenizer;
        let ids = t.encode("abcab");
        let mut logits = vec![0.0; t.n_vocab()];
        m.prefill(&ids[..ids.len() - 1]).expect("fits");
        let next = m.next(ids[ids.len() - 1], Some(&mut logits)).expect("fits");
        assert_eq!(t.decode(&[next]), "c");
        assert_eq!(logits[next as usize], PREDICTED);
        m.reset().expect("mock reset");
        let ids = t.encode("xyz");
        m.prefill(&ids[..2]).expect("fits");
        assert_eq!(m.next(ids[2], None).expect("fits"), t.eos());
    }
}
