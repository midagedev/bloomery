//! The row derivation: which rows of an n-gram table one token wants.
//!
//! The constants are **file data, not code**. GGUF metadata keys carry the
//! multipliers, the bucket sizes, the offsets and — for V4.1's engram — the
//! compressed-vocabulary map, and a file missing any of them is an error here:
//! nothing in this module has a default, because a default would be a number
//! nobody measured. Two key sets exist, each read by its own constructor:
//! [`Hash::from_gguf`] (V4.1's engram, `deepseek41.engram.*`) and
//! [`Hash::ple_from_gguf`] (a PLE site, `<arch>.ple.*`).
//!
//! The formula, per site `e` and per token ([`ngram::rows`]):
//!
//! ```text
//! ctx[0]  = the current token's context value
//! ctx[s]  = the value of the token s positions back
//! rolling = ctx[0] * mult[0]
//! for s in 1..n_gram:
//!     rolling ^= ctx[s] * mult[s]
//!     for h in 0..n_heads:
//!         b      = (s-1)*n_heads + h
//!         idx[b] = rolling % prime[b] + offset[b]
//! ```
//!
//! What a context value is, is the window's ([`Window`]): V4.1 maps each id
//! through the token map and fills slots before the start with the pad id; a
//! PLE site hashes raw ids, and an EOS predecessor or the sequence start turns
//! that slot and every older one into the EOS ([`ngram::EosWindow`]).
//!
//! Two shapes follow from it and the caller has to know both:
//!
//! * **The buckets partition the table's first rows.** `offset` is the
//!   exclusive prefix sum of `prime`, so bucket `b` owns `[offset[b],
//!   offset[b] + prime[b])` and the intervals tile rows `0 .. Σ prime` with no
//!   gap and no overlap. V4.1's table is exactly that tall; a PLE table is
//!   padded past it (its row count rounded up), and no id reaches the pad.
//! * **All `n_heads` of one n-gram order share a `rolling`.** They differ only
//!   in which prime reduces it, so the distinct rows an order produces is the
//!   distinct n-grams it saw, not the distinct tokens.
//!
//! Arithmetic is `u64` throughout and wrapping where the port's `uint64_t` would
//! wrap. The reduction's result is a row id, which fits `u32` because the
//! partition total does — both constructors refuse a file where it does not.

use std::path::Path;

use gguf::{Inventory, Value};

use crate::EngramError;

pub mod ngram;

pub use ngram::{EosWindow, History, NgramError};

/// Every key [`Hash::from_gguf`] reads is `deepseek41.engram.<suffix>`.
const ENGRAM_PREFIX: &str = "deepseek41.engram.";

/// What a window slot holds: the part of the hash the two key sets differ in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Window {
    /// V4.1's engram: each id through `token_map` (indexed by token id,
    /// exactly as long as the model's vocabulary, [`Hash::check_vocab`]),
    /// `pad` in a slot before the sequence start. `pad` is used as a mapped id
    /// directly, not looked up in the map.
    Mapped { token_map: Vec<u32>, pad: u64 },
    /// A PLE site: raw ids with an EOS reset.
    Eos(EosWindow),
}

/// The hash constants of one model, one set per site.
///
/// Built by [`Hash::from_gguf`] or [`Hash::ple_from_gguf`] from a shard
/// header; nothing here is derived and nothing has a default.
pub struct Hash {
    /// Block index of each site, in the order the metadata lists them. This is
    /// the order every per-site slice below is indexed by.
    layer_ids: Vec<u32>,
    n_heads: usize,
    n_gram: usize,
    /// `(n_gram - 1) * n_heads`: rows one site reads for one token.
    n_cols: usize,
    key_length: u32,
    window: Window,
    /// `sites * n_gram` multipliers, site-major.
    mult: Vec<u64>,
    /// `sites * n_cols` primes, site-major. Each is a divisor, so none is zero.
    prime: Vec<u64>,
    /// `sites * n_cols` offsets, site-major.
    offset: Vec<u64>,
}

impl Hash {
    /// Read the constants out of one shard's header.
    ///
    /// Every key is mandatory, and so is its length: a file that carries eight
    /// of the nine, or a prime array of the wrong width, is an error naming the
    /// key rather than a set of constants that is silently short.
    pub fn from_gguf(meta: &Inventory) -> Result<Hash, EngramError> {
        let k = |suffix: &str| format!("{ENGRAM_PREFIX}{suffix}");
        let layer_ids: Vec<u32> = ints(meta, &k("layer_ids"))?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| value_err(k("layer_ids"), "a block index")))
            .collect::<Result<_, _>>()?;
        let sites = layer_ids.len();
        if sites == 0 {
            return Err(value_err(k("layer_ids"), "at least one engram site"));
        }

        let n_heads = usize::try_from(scalar(meta, &k("head_count"))?)
            .map_err(|_| value_err(k("head_count"), "a head count"))?;
        let n_gram = usize::try_from(scalar(meta, &k("max_ngram_size"))?)
            .map_err(|_| value_err(k("max_ngram_size"), "an n-gram size"))?;
        // The port's own floor: one head and a 2-gram, because `n_cols` is
        // `(n_gram - 1) * n_heads` and a site with no buckets reads no rows.
        if n_heads == 0 {
            return Err(value_err(k("head_count"), "at least one head"));
        }
        if n_gram < 2 {
            return Err(value_err(k("max_ngram_size"), "at least a 2-gram"));
        }
        let n_cols = (n_gram - 1) * n_heads;

        let key_length = u32::try_from(scalar(meta, &k("key_length"))?)
            .map_err(|_| value_err(k("key_length"), "a key length"))?;
        let pad = scalar(meta, &k("pad_id"))?;

        let mult = ints(meta, &k("multipliers"))?;
        need_len(k("multipliers"), mult.len(), sites * n_gram)?;
        let prime = ints(meta, &k("primes"))?;
        need_len(k("primes"), prime.len(), sites * n_cols)?;
        let offset = ints(meta, &k("offsets"))?;
        need_len(k("offsets"), offset.len(), sites * n_cols)?;
        check_buckets(
            &prime,
            &offset,
            [&k("primes"), &k("offsets")],
            u64::from(u32::MAX),
            "a bucket top inside u32",
        )?;

        let token_map: Vec<u32> = ints(meta, &k("token_map"))?
            .into_iter()
            .map(|v| {
                u32::try_from(v).map_err(|_| value_err(k("token_map"), "a compressed vocab id"))
            })
            .collect::<Result<_, _>>()?;
        if token_map.is_empty() {
            return Err(value_err(k("token_map"), "a non-empty map"));
        }

        Ok(Hash {
            layer_ids,
            n_heads,
            n_gram,
            n_cols,
            key_length,
            window: Window::Mapped { token_map, pad },
            mult,
            prime,
            offset,
        })
    }

    /// Read a PLE site's constants out of the header of the shard that
    /// carries `<arch>.ple.*` (the first shard of a split set).
    ///
    /// The keys, as the port's loader reads them (qwen4exp.cpp:66-121):
    /// `ple.layers` (exactly one site), `ple.ngram_size` (2 to 8),
    /// `ple.heads_per_ngram` (at most 64 rows a token),
    /// `embedding_length_per_layer_input` (values a row),
    /// `ple.layer_multipliers` (`ngram_size` of them), `ple.head_vocab_sizes`
    /// (the buckets' sizes, the formula's `prime`, none zero) and
    /// `ple.head_offsets`, one per row a token reads, every bucket's top
    /// inside an `i32` row index; `ple.eos_token_id` resets the window and
    /// `ple.image_token_id`, when present, is refused as an input. Both ids
    /// must be tokens of the `n_vocab`-token vocabulary the caller's reader
    /// states. The EOS is the file's `ple.eos_token_id`, never the
    /// tokenizer's end of text: the two differ (qwen4exp: 248,044 against
    /// 248,046).
    pub fn ple_from_gguf(meta: &Inventory, arch: &str, n_vocab: u32) -> Result<Hash, EngramError> {
        let k = |suffix: &str| format!("{arch}.ple.{suffix}");
        let layer_ids: Vec<u32> = ints(meta, &k("layers"))?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| value_err(k("layers"), "a block index")))
            .collect::<Result<_, _>>()?;
        if layer_ids.len() != 1 {
            return Err(value_err(k("layers"), "exactly one PLE site"));
        }
        let n_gram = usize::try_from(scalar(meta, &k("ngram_size"))?)
            .ok()
            .filter(|n| (2..=MAX_PLE_NGRAM).contains(n))
            .ok_or_else(|| value_err(k("ngram_size"), "an n-gram size of 2 to 8"))?;
        let n_heads = usize::try_from(scalar(meta, &k("heads_per_ngram"))?)
            .ok()
            .filter(|&h| h >= 1 && (n_gram - 1) * h <= MAX_PLE_ROWS)
            .ok_or_else(|| value_err(k("heads_per_ngram"), "1 to 64 rows a token"))?;
        let n_cols = (n_gram - 1) * n_heads;
        let row_key = format!("{arch}.embedding_length_per_layer_input");
        let key_length = u32::try_from(scalar(meta, &row_key)?)
            .ok()
            .filter(|&r| r > 0)
            .ok_or_else(|| value_err(row_key.clone(), "a positive row length"))?;

        let mult = ints(meta, &k("layer_multipliers"))?;
        need_len(k("layer_multipliers"), mult.len(), n_gram)?;
        let prime = ints(meta, &k("head_vocab_sizes"))?;
        need_len(k("head_vocab_sizes"), prime.len(), n_cols)?;
        let offset = ints(meta, &k("head_offsets"))?;
        need_len(k("head_offsets"), offset.len(), n_cols)?;
        check_buckets(
            &prime,
            &offset,
            [&k("head_vocab_sizes"), &k("head_offsets")],
            i32::MAX as u64,
            "a bucket top inside an i32 row index",
        )?;

        let token = |suffix: &str| -> Result<u32, EngramError> {
            u32::try_from(scalar(meta, &k(suffix))?)
                .ok()
                .filter(|&t| t < n_vocab)
                .ok_or_else(|| value_err(k(suffix), "a token of the vocabulary"))
        };
        let eos = token("eos_token_id")?;
        let image = match meta.value(&k("image_token_id")) {
            Some(_) => Some(token("image_token_id")?),
            None => None,
        };

        Ok(Hash {
            layer_ids,
            n_heads,
            n_gram,
            n_cols,
            key_length,
            window: Window::Eos(EosWindow {
                eos,
                image,
                n_vocab,
            }),
            mult,
            prime,
            offset,
        })
    }

    /// The constants of the split set in `dir`, read from headers only.
    ///
    /// The tables themselves are never mapped: this is the path for a caller
    /// that wants the derivation without the 194 GiB behind it.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Hash, EngramError> {
        let shards = crate::shards_in(dir.as_ref())?;
        let mut invs = Vec::with_capacity(shards.len());
        for shard in &shards {
            invs.push(gguf::inventory_of(shard)?);
        }
        Hash::from_shards(&invs)
    }

    /// The first shard whose header carries the constants owns them. A split
    /// set writes the metadata once, into shard 1.
    pub(crate) fn from_shards(invs: &[Inventory]) -> Result<Hash, EngramError> {
        let key = format!("{ENGRAM_PREFIX}layer_ids");
        let carrier = invs
            .iter()
            .find(|inv| inv.value(&key).is_some())
            .ok_or(EngramError::NoHashMetadata(key))?;
        Hash::from_gguf(carrier)
    }

    /// Engram sites the metadata names.
    pub fn sites(&self) -> usize {
        self.layer_ids.len()
    }

    /// Block index of each site, in metadata order.
    pub fn layer_ids(&self) -> &[u32] {
        &self.layer_ids
    }

    /// Longest n-gram the hash folds (4 in V4.1-Flash).
    pub fn n_gram(&self) -> usize {
        self.n_gram
    }

    /// Heads per n-gram order (8 in V4.1-Flash).
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Rows one site reads for one token: `(n_gram - 1) * n_heads`.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Values in one engram row, as the metadata states it.
    pub fn key_length(&self) -> u32 {
        self.key_length
    }

    /// What a window slot holds.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// The context value of a position before the sequence starts.
    ///
    /// # Panics
    ///
    /// On a raw-id hash ([`Window::Eos`]), which has no pad id: its slots
    /// before the start are its EOS ([`Hash::ple_rows_into`]).
    pub fn pad_id(&self) -> u64 {
        self.mapped("pad_id").1
    }

    /// The compressed-vocabulary map, indexed by token id.
    ///
    /// # Panics
    ///
    /// On a raw-id hash ([`Window::Eos`]), which has no map.
    pub fn token_map(&self) -> &[u32] {
        self.mapped("token_map").0
    }

    /// The map and the pad of a [`Window::Mapped`] hash; `what` names the
    /// accessor a raw-id hash was asked for.
    fn mapped(&self, what: &str) -> (&[u32], u64) {
        match &self.window {
            Window::Mapped { token_map, pad } => (token_map, *pad),
            Window::Eos(_) => panic!(
                "Hash::{what}: this hash reads raw ids with an EOS reset and has no token map or \
                 pad id; its rows come from Hash::ple_rows_into"
            ),
        }
    }

    /// `site`'s primes, one per bucket, in bucket order.
    pub fn primes(&self, site: usize) -> Result<&[u64], EngramError> {
        self.slice(&self.prime, site, self.n_cols)
    }

    /// `site`'s offsets, one per bucket, in bucket order.
    pub fn offsets(&self, site: usize) -> Result<&[u64], EngramError> {
        self.slice(&self.offset, site, self.n_cols)
    }

    /// `site`'s multipliers, one per n-gram position.
    pub fn multipliers(&self, site: usize) -> Result<&[u64], EngramError> {
        self.slice(&self.mult, site, self.n_gram)
    }

    /// Rows the buckets of `site` cover together: V4.1's table's row count
    /// (the partition is exact, which [`Hash::primes`] against the header is
    /// how a gate checks), and the rows below a PLE table's pad.
    pub fn partition_rows(&self, site: usize) -> Result<u64, EngramError> {
        Ok(self.primes(site)?.iter().sum())
    }

    /// The n-gram order a bucket belongs to: bucket `b` is order `b / n_heads + 2`.
    pub fn order_of_bucket(&self, bucket: usize) -> usize {
        bucket / self.n_heads + 2
    }

    /// `token` through the compressed vocabulary.
    ///
    /// The map is indexed by token id and is as long as the vocabulary
    /// ([`Hash::check_vocab`]), so an id past its end is not a token of this
    /// model: it is refused by name, never mapped to a plausible value. A
    /// position before the sequence starts is not a token either; its context
    /// value is [`Hash::pad_id`], which the caller puts there itself.
    ///
    /// # Panics
    ///
    /// On a raw-id hash ([`Window::Eos`]), which has no map.
    pub fn map_token(&self, token: u32) -> Result<u64, EngramError> {
        let map = self.mapped("map_token").0;
        map.get(token as usize)
            .map(|&v| u64::from(v))
            .ok_or(EngramError::TokenPastMap {
                token,
                map: map.len(),
            })
    }

    /// Refuse a model whose vocabulary is not the map's domain. The map is
    /// indexed by token id, so it must be exactly `n_vocab` long: shorter, and
    /// a token the model accepts would have no mapped value; longer, and the
    /// file pairs this table with another vocabulary.
    ///
    /// # Panics
    ///
    /// On a raw-id hash ([`Window::Eos`]), which has no map; its vocabulary
    /// is the one [`Hash::ple_from_gguf`] was given.
    pub fn check_vocab(&self, n_vocab: usize) -> Result<(), EngramError> {
        let map = self.mapped("check_vocab").0;
        if map.len() == n_vocab {
            return Ok(());
        }
        Err(EngramError::VocabMismatch {
            map: map.len(),
            vocab: n_vocab,
        })
    }

    /// The `n_cols` row ids `site` wants for the token at the head of `ctx`.
    ///
    /// `ctx` is the window of [`Hash::window`]'s kind — for V4.1 the mapped
    /// ids, `ctx[0]` the current token, `ctx[s]` the token `s` positions back,
    /// the pad id where that position is before the sequence start — and
    /// `out` is exactly `n_cols` long. Nothing is allocated: both buffers are
    /// the caller's and are reused across tokens.
    pub fn rows_into(&self, site: usize, ctx: &[u64], out: &mut [u32]) -> Result<(), EngramError> {
        if site >= self.sites() {
            return Err(EngramError::SiteOutOfRange {
                site,
                sites: self.sites(),
            });
        }
        if ctx.len() != self.n_gram {
            return Err(EngramError::WindowSize {
                want: self.n_gram,
                got: ctx.len(),
            });
        }
        if out.len() != self.n_cols {
            return Err(EngramError::RowBufferSize {
                want: self.n_cols,
                got: out.len(),
            });
        }
        let mult = &self.mult[site * self.n_gram..][..self.n_gram];
        let prime = &self.prime[site * self.n_cols..][..self.n_cols];
        let offset = &self.offset[site * self.n_cols..][..self.n_cols];
        ngram::rows(mult, prime, offset, self.n_heads, ctx, out);
        Ok(())
    }

    /// A raw-id hash's history of a new sequence: `n_gram − 1` EOS slots at
    /// position 0.
    pub fn new_history(&self) -> Result<History, NgramError> {
        match &self.window {
            Window::Eos(w) => Ok(History::new(self.n_gram, w.eos)),
            Window::Mapped { .. } => Err(NgramError::NotRawIds),
        }
    }

    /// The row ids of a raw-id hash's one site for `tokens`, the positions
    /// `pos, pos + 1, …` of the sequence whose history is `hist`, into `out`
    /// (`n_cols` a token, token-major), and `hist` advanced past them.
    ///
    /// Each token's window is [`EosWindow::window`] over the history and the
    /// tokens before it in the call. Every token is checked before any row is
    /// written or the history moves, so a refusal leaves both as they were:
    /// an id past the vocabulary or the image placeholder, a `pos` other than
    /// the history's next position (a new sequence starts from a fresh
    /// [`History::new`]; a rollback restores the copy from before the pass),
    /// or an `out` of another length. Allocation-free.
    pub fn ple_rows_into(
        &self,
        hist: &mut History,
        pos: u64,
        tokens: &[u32],
        out: &mut [u32],
    ) -> Result<(), NgramError> {
        let Window::Eos(w) = &self.window else {
            return Err(NgramError::NotRawIds);
        };
        assert_eq!(
            hist.tokens().len() + 1,
            self.n_gram,
            "Hash::ple_rows_into: a history of {} tokens for a {}-gram hash; History::new takes \
             the hash's n_gram",
            hist.tokens().len(),
            self.n_gram
        );
        if pos != hist.next_pos() {
            return Err(NgramError::PositionGap {
                want: hist.next_pos(),
                got: pos,
            });
        }
        let want = tokens.len() * self.n_cols;
        if out.len() != want {
            return Err(NgramError::RowBufferSize {
                tokens: tokens.len(),
                want,
                got: out.len(),
            });
        }
        for &t in tokens {
            w.check(t)?;
        }
        let mut ctx = [0u64; MAX_PLE_NGRAM];
        let ctx = &mut ctx[..self.n_gram];
        for (&t, rows) in tokens.iter().zip(out.chunks_exact_mut(self.n_cols)) {
            w.window(hist.tokens(), t, ctx);
            ngram::rows(
                &self.mult,
                &self.prime,
                &self.offset,
                self.n_heads,
                ctx,
                rows,
            );
            hist.push(t);
        }
        Ok(())
    }

    fn slice<'a>(
        &self,
        all: &'a [u64],
        site: usize,
        stride: usize,
    ) -> Result<&'a [u64], EngramError> {
        if site >= self.sites() {
            return Err(EngramError::SiteOutOfRange {
                site,
                sites: self.sites(),
            });
        }
        Ok(&all[site * stride..][..stride])
    }
}

/// The widest n-gram a PLE site folds (llama.cpp's `LLAMA_MAX_PLE_NGRAM`).
const MAX_PLE_NGRAM: usize = 8;
/// The most rows a PLE site reads a token (llama.cpp's `LLAMA_MAX_PLE_HEADS`).
const MAX_PLE_ROWS: usize = 64;

fn value_err(key: String, what: &'static str) -> EngramError {
    EngramError::KeyValue { key, at: 0, what }
}

fn need_len(key: String, got: usize, want: usize) -> Result<(), EngramError> {
    if got == want {
        return Ok(());
    }
    Err(EngramError::KeyLength { key, want, got })
}

/// A bucket size is a divisor in the hash and the width of a bucket: zero
/// would divide by zero, and every bucket's top must stay at or below `top`,
/// the widest row id the reader's port carries (`u32` for V4.1's engram, an
/// `i32` for a PLE site).
fn check_buckets(
    prime: &[u64],
    offset: &[u64],
    [prime_key, offset_key]: [&str; 2],
    top: u64,
    what: &'static str,
) -> Result<(), EngramError> {
    for (b, (&p, &o)) in prime.iter().zip(offset).enumerate() {
        if p == 0 {
            return Err(EngramError::KeyValue {
                key: prime_key.to_string(),
                at: b,
                what: "a non-zero divisor",
            });
        }
        if o.checked_add(p).is_none_or(|end| end > top) {
            return Err(EngramError::KeyValue {
                key: offset_key.to_string(),
                at: b,
                what,
            });
        }
    }
    Ok(())
}

/// A non-negative integer of any width or signedness the file may have used.
///
/// These nine keys are not written with one tag: the converter emits the three
/// `u64` arrays as `u64` and `layer_ids`/`token_map` as `i32`, because that is
/// what the tensor-index and token-id types are on the writing side. All of
/// them are counts and indices, so a negative entry is a corrupt file rather
/// than a value with a meaning, and it is refused here.
fn unsigned(v: &Value) -> Option<u64> {
    match v {
        Value::I8(n) => u64::try_from(*n).ok(),
        Value::I16(n) => u64::try_from(*n).ok(),
        Value::I32(n) => u64::try_from(*n).ok(),
        Value::I64(n) => u64::try_from(*n).ok(),
        other => other.as_u64(),
    }
}

/// One mandatory unsigned scalar.
fn scalar(meta: &Inventory, key: &str) -> Result<u64, EngramError> {
    let key = key.to_string();
    let v = meta
        .value(&key)
        .ok_or_else(|| EngramError::MissingKey { key: key.clone() })?;
    unsigned(v).ok_or(EngramError::KeyType {
        key,
        want: "a non-negative integer",
    })
}

/// One mandatory array of unsigned integers, widened to `u64`.
fn ints(meta: &Inventory, key: &str) -> Result<Vec<u64>, EngramError> {
    let key = key.to_string();
    let v = meta
        .value(&key)
        .ok_or_else(|| EngramError::MissingKey { key: key.clone() })?;
    let Value::Array(items) = v else {
        return Err(EngramError::KeyType {
            key,
            want: "an array",
        });
    };
    items
        .iter()
        .map(|item| {
            unsigned(item).ok_or(EngramError::KeyType {
                key: key.clone(),
                want: "an array of non-negative integers",
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Hash, History, NgramError, Window};
    use crate::EngramError;
    use gguf::{Inventory, Value};

    /// A hash whose map covers token ids `0..vocab`, with the smallest
    /// constants the reader accepts: these tests read the map only.
    fn with_map(vocab: u32) -> Hash {
        Hash {
            layer_ids: vec![1],
            n_heads: 1,
            n_gram: 2,
            n_cols: 1,
            key_length: 256,
            window: Window::Mapped {
                token_map: (0..vocab).map(|t| t / 2).collect(),
                pad: 2,
            },
            mult: vec![1, 1],
            prime: vec![7],
            offset: vec![0],
        }
    }

    /// An id past the map is refused by name, with the id and the map's
    /// length, and never mapped to the pad; the last id inside the map maps.
    #[test]
    fn map_token_refuses_an_id_past_the_map() {
        let hash = with_map(16);
        assert_eq!(
            hash.map_token(15).ok(),
            Some(7),
            "the last id inside the map"
        );
        for token in [16, u32::MAX] {
            match hash.map_token(token) {
                Err(EngramError::TokenPastMap { token: t, map: 16 }) if t == token => {}
                other => panic!(
                    "token {token} of a 16-entry map: want TokenPastMap naming it and the \
                     map's length, got {other:?}"
                ),
            }
        }
    }

    /// A map whose length is not the model's vocabulary is refused by name,
    /// with both lengths; equal lengths pass.
    #[test]
    fn check_vocab_refuses_a_map_of_another_length() {
        let hash = with_map(16);
        assert!(
            hash.check_vocab(16).is_ok(),
            "a map as long as the vocabulary"
        );
        for vocab in [15, 17] {
            match hash.check_vocab(vocab) {
                Err(EngramError::VocabMismatch { map: 16, vocab: v }) if v == vocab => {}
                other => panic!(
                    "a 16-entry map against a vocabulary of {vocab}: want VocabMismatch naming \
                     both, got {other:?}"
                ),
            }
        }
    }

    /// A V4.1-shaped constant set (two sites, 4-grams, eight heads, a
    /// 1,000-token map) as a header would carry it.
    fn v41_fixture() -> Inventory {
        let k = |s: &str| format!("deepseek41.engram.{s}");
        let u64s = |v: Vec<u64>| Value::Array(v.into_iter().map(Value::U64).collect());
        let i32s = |v: Vec<i32>| Value::Array(v.into_iter().map(Value::I32).collect());
        let mult = (0..8u64)
            .map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(2 * i + 1) | 1)
            .collect();
        let (mut primes, mut offsets) = (Vec::new(), Vec::new());
        for _site in 0..2 {
            let mut at = 0u64;
            for b in 0..24u64 {
                let p = 2_000_003 + 10 * b;
                primes.push(p);
                offsets.push(at);
                at += p;
            }
        }
        inventory(vec![
            (k("layer_ids"), i32s(vec![1, 14])),
            (k("head_count"), Value::U32(8)),
            (k("max_ngram_size"), Value::U32(4)),
            (k("key_length"), Value::U32(256)),
            (k("pad_id"), Value::U32(600)),
            (k("multipliers"), u64s(mult)),
            (k("primes"), u64s(primes)),
            (k("offsets"), u64s(offsets)),
            (
                k("token_map"),
                i32s((0..1000).map(|t| (t * 7919 + 13) % 600).collect()),
            ),
        ])
    }

    fn inventory(meta: Vec<(String, Value)>) -> Inventory {
        Inventory {
            version: 3,
            meta,
            tensors: Vec::new(),
            header_end: 0,
            data_base: 0,
            alignment: 32,
            file_len: 0,
        }
    }

    /// The rows the pre-generalisation `Hash::rows_into` gave for
    /// [`v41_fixture`] and the token run below: sites 0 and 1 at positions 0
    /// and 5, and an FNV-1a fold over every id of positions 0 to 5, sites in
    /// order. The window is built as `token_rows` builds it: mapped ids, the
    /// pad before the start.
    const V41_ROWS: [[u32; 24]; 4] = [
        [
            1_666_090, 2_930_876, 5_110_680, 6_198_149, 8_185_976, 11_066_841, 12_833_361,
            15_478_259, 17_215_677, 18_350_980, 21_512_944, 22_692_853, 25_882_403, 27_072_838,
            28_255_751, 31_422_775, 32_006_163, 34_311_601, 36_645_437, 39_005_751, 41_390_623,
            43_798_133, 44_226_138, 46_673_154,
        ],
        [
            1_195_479, 3_945_950, 4_888_691, 6_017_135, 9_324_755, 10_804_905, 12_451_048,
            14_256_614, 17_238_116, 19_045_294, 20_173_036, 22_609_995, 24_344_618, 27_365_578,
            29_661_302, 31_220_330, 32_806_025, 34_658_829, 36_431_811, 38_116_451, 41_704_432,
            43_186_838, 44_555_342, 47_801_657,
        ],
        [
            24_911, 3_549_910, 5_371_892, 7_490_020, 9_903_454, 10_611_301, 13_612_817, 14_907_046,
            17_584_567, 19_636_100, 21_356_653, 22_742_326, 25_789_342, 26_493_565, 28_851_351,
            30_858_677, 33_497_920, 35_594_376, 37_558_666, 39_384_430, 41_065_308, 42_594_940,
            45_967_189, 47_175_259,
        ],
        [
            1_678_064, 3_597_240, 4_506_724, 6_400_129, 9_271_065, 11_113_089, 13_919_844,
            15_684_867, 16_435_088, 18_600_960, 21_737_137, 23_838_856, 24_901_437, 26_920_333,
            29_890_894, 31_808_317, 32_602_837, 35_160_968, 36_400_072, 38_308_972, 40_876_328,
            42_090_587, 45_940_825, 46_415_033,
        ],
    ];
    const V41_FOLD: u64 = 0xaba991d0914c6e41;

    #[test]
    fn v41_rows_are_the_pinned_rows() {
        let hash = Hash::from_gguf(&v41_fixture()).unwrap();
        let tokens = [5u32, 999, 0, 17, 256, 42];
        let mut fold = 0xcbf2_9ce4_8422_2325u64;
        let mut pinned = V41_ROWS.iter();
        for p in 0..tokens.len() {
            let mut ctx = [hash.pad_id(); 4];
            for (s, slot) in ctx.iter_mut().enumerate().take(p + 1) {
                *slot = hash.map_token(tokens[p - s]).unwrap();
            }
            for site in 0..2 {
                let mut out = [0u32; 24];
                hash.rows_into(site, &ctx, &mut out).unwrap();
                for &id in &out {
                    fold = (fold ^ u64::from(id)).wrapping_mul(0x100_0000_01b3);
                }
                if p == 0 || p == 5 {
                    assert_eq!(&out, pinned.next().unwrap(), "position {p} site {site}");
                }
            }
        }
        assert_eq!(fold, V41_FOLD, "the fold over every id of the run");
    }

    /// Qwen3.8-Flash-Next's PLE constants as its UD-Q4_K_XL header carries
    /// them (`qwen4exp.ple.*`).
    fn ple_fixture() -> Inventory {
        let k = |s: &str| format!("qwen4exp.ple.{s}");
        let u64s = |v: &[u64]| Value::Array(v.iter().copied().map(Value::U64).collect());
        inventory(vec![
            (k("layers"), Value::Array(vec![Value::I32(1)])),
            (k("ngram_size"), Value::U32(3)),
            (k("heads_per_ngram"), Value::U32(8)),
            (k("conv_kernel"), Value::U32(4)),
            (k("eos_token_id"), Value::U32(248_044)),
            (k("image_token_id"), Value::U32(248_056)),
            (
                "qwen4exp.embedding_length_per_layer_input".to_string(),
                Value::U32(160),
            ),
            (
                k("layer_multipliers"),
                u64s(&[23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]),
            ),
            (k("head_offsets"), u64s(&PLE_OFFSETS)),
            (k("head_vocab_sizes"), u64s(&PLE_VOCAB)),
        ])
    }

    const PLE_VOCAB: [u64; 16] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    const PLE_OFFSETS: [u64; 16] = [
        0,
        20_000_003,
        40_000_026,
        60_000_059,
        80_000_106,
        100_000_165,
        120_000_228,
        140_000_297,
        160_000_374,
        180_000_455,
        200_000_548,
        220_000_655,
        240_000_802,
        260_000_955,
        280_001_114,
        300_001_275,
    ];

    /// The prompt below from the sequence start: ik_llama.cpp's PLE rows
    /// (`llama_set_inputs`, the `inp_ple_rows` block, c10fbbcc), computed by
    /// a line-for-line transcription of that block that agrees with
    /// transformers' `_shift_right_ignore_eos` form on every position. The
    /// ids exercise the rule, not a tokenized text: 248,046 (the tokenizer's
    /// end of text) is an ordinary id, 248,044 resets the window after it,
    /// and the last token is that EOS.
    const PLE_PROMPT: [u32; 9] = [248_045, 846, 198, 248_046, 198, 248_044, 9707, 11, 248_044];
    const PLE_IK_ROWS: [[u32; 16]; 9] = [
        [
            13_981_229,
            30_076_798,
            49_239_434,
            61_315_716,
            98_540_601,
            111_186_732,
            120_378_863,
            146_384_625,
            173_637_879,
            184_336_749,
            217_808_755,
            222_173_721,
            249_579_227,
            277_180_543,
            299_757_935,
            312_970_430,
        ],
        [
            4_908_448,
            36_872_991,
            46_611_251,
            64_451_521,
            83_649_409,
            110_850_044,
            122_402_102,
            159_207_155,
            170_314_019,
            195_445_770,
            214_314_282,
            239_389_039,
            254_862_662,
            271_565_595,
            284_073_139,
            308_659_387,
        ],
        [
            4_582_007,
            37_691_206,
            44_255_274,
            65_455_601,
            89_351_473,
            103_985_474,
            135_938_409,
            145_212_418,
            163_775_156,
            199_680_439,
            200_044_921,
            228_961_047,
            240_669_798,
            272_736_434,
            296_838_250,
            317_942_745,
        ],
        [
            800_878,
            26_890_228,
            44_344_143,
            73_717_970,
            83_481_085,
            114_342_838,
            131_517_223,
            156_062_515,
            178_597_259,
            183_407_612,
            204_367_527,
            227_415_283,
            250_927_472,
            275_497_467,
            290_589_253,
            307_810_700,
        ],
        [
            510_813,
            24_677_820,
            50_498_865,
            70_834_323,
            89_294_582,
            102_912_024,
            124_085_691,
            153_712_591,
            175_541_844,
            189_765_562,
            204_229_412,
            238_186_920,
            254_720_141,
            272_150_632,
            291_493_629,
            309_703_420,
        ],
        [
            16_849_591,
            37_537_304,
            42_290_399,
            73_883_125,
            88_405_327,
            114_186_760,
            123_740_665,
            158_125_394,
            162_126_929,
            190_169_925,
            201_084_305,
            223_990_392,
            248_114_762,
            273_201_310,
            295_110_750,
            306_260_765,
        ],
        [
            16_410_909,
            39_682_429,
            55_103_279,
            60_931_720,
            87_006_904,
            116_506_179,
            131_512_017,
            152_932_897,
            169_641_436,
            182_022_480,
            209_277_433,
            237_891_023,
            256_841_529,
            277_007_269,
            290_665_954,
            300_984_276,
        ],
        [
            18_158_303,
            36_390_029,
            45_652_312,
            70_783_524,
            98_191_154,
            114_024_957,
            127_804_916,
            159_566_246,
            175_132_467,
            192_583_600,
            217_919_476,
            237_199_054,
            259_336_807,
            261_799_367,
            289_359_313,
            307_699_687,
        ],
        [
            4_679_147,
            20_352_667,
            52_599_150,
            74_683_044,
            98_198_172,
            106_977_280,
            121_027_876,
            141_408_284,
            179_487_506,
            195_941_560,
            207_086_481,
            221_700_584,
            240_897_753,
            261_139_696,
            294_786_063,
            304_757_900,
        ],
    ];

    /// PLE rows equal ik's for the prompt in one call and in two calls
    /// through the history, and the history after either is the last two
    /// tokens at position 9.
    #[test]
    fn ple_rows_are_iks() {
        let hash = Hash::ple_from_gguf(&ple_fixture(), "qwen4exp", 248_320).unwrap();
        assert_eq!((hash.sites(), hash.n_gram(), hash.n_cols()), (1, 3, 16));
        let want: Vec<u32> = PLE_IK_ROWS.iter().flatten().copied().collect();

        let mut hist = History::new(3, 248_044);
        let mut out = vec![0u32; 9 * 16];
        hash.ple_rows_into(&mut hist, 0, &PLE_PROMPT, &mut out)
            .unwrap();
        assert_eq!(out, want, "one call");
        assert_eq!((hist.tokens(), hist.next_pos()), (&[11, 248_044][..], 9));

        let mut split = hash.new_history().unwrap();
        let mut out = vec![0u32; 9 * 16];
        let (a, b) = out.split_at_mut(4 * 16);
        hash.ple_rows_into(&mut split, 0, &PLE_PROMPT[..4], a)
            .unwrap();
        hash.ple_rows_into(&mut split, 4, &PLE_PROMPT[4..], b)
            .unwrap();
        assert_eq!(out, want, "two calls through the history");
        assert_eq!(split, hist);
    }

    /// Every refusal is by name and leaves the history and the rows as they
    /// were: the image placeholder, an id past the vocabulary (each after a
    /// valid token), a position the history does not end at, a row buffer of
    /// another length, and a V4.1 hash asked for raw-id rows.
    #[test]
    fn ple_refusals_leave_the_state() {
        let hash = Hash::ple_from_gguf(&ple_fixture(), "qwen4exp", 248_320).unwrap();
        let mut hist = History::new(3, 248_044);
        hash.ple_rows_into(&mut hist, 0, &[7, 8], &mut [0; 32])
            .unwrap();
        let before = hist.clone();
        let mut out = [u32::MAX; 32];
        let cases: [(u64, [u32; 2], usize, NgramError); 4] = [
            (
                2,
                [9, 248_056],
                32,
                NgramError::ImageToken { token: 248_056 },
            ),
            (
                2,
                [9, 248_320],
                32,
                NgramError::TokenPastVocab {
                    token: 248_320,
                    vocab: 248_320,
                },
            ),
            (0, [9, 10], 32, NgramError::PositionGap { want: 2, got: 0 }),
            (
                2,
                [9, 10],
                16,
                NgramError::RowBufferSize {
                    tokens: 2,
                    want: 32,
                    got: 16,
                },
            ),
        ];
        for (pos, tokens, len, err) in cases {
            assert_eq!(
                hash.ple_rows_into(&mut hist, pos, &tokens, &mut out[..len]),
                Err(err)
            );
            assert_eq!(hist, before);
            assert!(out.iter().all(|&r| r == u32::MAX), "no row written");
        }
        let v41 = Hash::from_gguf(&v41_fixture()).unwrap();
        assert_eq!(
            v41.ple_rows_into(&mut History::new(4, 0), 0, &[1], &mut [0; 24]),
            Err(NgramError::NotRawIds)
        );
    }

    /// A raw-id hash has no pad id: asking for one panics by name.
    #[test]
    #[should_panic(expected = "Hash::pad_id: this hash reads raw ids")]
    fn ple_hash_has_no_pad() {
        let hash = Hash::ple_from_gguf(&ple_fixture(), "qwen4exp", 248_320).unwrap();
        let _ = hash.pad_id();
    }

    /// The PLE reader refuses by key: a missing EOS, an EOS past the
    /// vocabulary, a bucket whose top leaves the i32 row index, a zero bucket
    /// size, and two sites.
    #[test]
    fn ple_from_gguf_refuses_by_key() {
        let edit = |key: &str, v: Option<Value>| {
            let mut inv = ple_fixture();
            inv.meta.retain(|(k, _)| k != key);
            if let Some(v) = v {
                inv.meta.push((key.to_string(), v));
            }
            Hash::ple_from_gguf(&inv, "qwen4exp", 248_320).err()
        };
        let key = |e: Option<EngramError>| match e {
            Some(EngramError::MissingKey { key } | EngramError::KeyValue { key, .. }) => key,
            other => panic!("want a key error, got {other:?}"),
        };
        assert_eq!(
            key(edit("qwen4exp.ple.eos_token_id", None)),
            "qwen4exp.ple.eos_token_id"
        );
        assert_eq!(
            key(edit("qwen4exp.ple.eos_token_id", Some(Value::U32(248_320)))),
            "qwen4exp.ple.eos_token_id"
        );
        let mut offsets = PLE_OFFSETS.to_vec();
        offsets[15] = i32::MAX as u64 - 20_000_170;
        let wide = Value::Array(offsets.into_iter().map(Value::U64).collect());
        assert_eq!(
            key(edit("qwen4exp.ple.head_offsets", Some(wide))),
            "qwen4exp.ple.head_offsets"
        );
        let mut vocab = PLE_VOCAB.to_vec();
        vocab[3] = 0;
        let zero = Value::Array(vocab.into_iter().map(Value::U64).collect());
        assert_eq!(
            key(edit("qwen4exp.ple.head_vocab_sizes", Some(zero))),
            "qwen4exp.ple.head_vocab_sizes"
        );
        let two = Value::Array(vec![Value::I32(1), Value::I32(5)]);
        assert_eq!(
            key(edit("qwen4exp.ple.layers", Some(two))),
            "qwen4exp.ple.layers"
        );
    }
}
