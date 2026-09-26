//! The row derivation: which rows of an engram table one token wants.
//!
//! The constants are **file data, not code**. Nine GGUF metadata keys carry the
//! multipliers, the primes, the offsets and the compressed-vocabulary map, and
//! a file missing any of them is an error here — nothing in this module has a
//! default, because a default would be a number nobody measured.
//!
//! The formula, per engram site `e` and per token:
//!
//! ```text
//! ctx[0]  = token_map[current token]
//! ctx[s]  = token_map[token s positions back]   (pad_id before the start)
//! rolling = ctx[0] * mult[0]
//! for s in 1..n_gram:
//!     rolling ^= ctx[s] * mult[s]
//!     for h in 0..n_heads:
//!         b      = (s-1)*n_heads + h
//!         idx[b] = rolling % prime[b] + offset[b]
//! ```
//!
//! Two shapes follow from it and the caller has to know both:
//!
//! * **The buckets partition the table.** `offset` is the exclusive prefix sum
//!   of `prime`, so bucket `b` owns `[offset[b], offset[b] + prime[b])` and the
//!   24 intervals tile the site's rows exactly. One token's rows are therefore
//!   spread over the whole table by construction, and an id can never collide
//!   across buckets.
//! * **All `n_heads` of one n-gram order share a `rolling`.** They differ only
//!   in which prime reduces it, so the distinct rows an order produces is the
//!   distinct n-grams it saw, not the distinct tokens.
//!
//! Arithmetic is `u64` throughout and wrapping where the port's `uint64_t` would
//! wrap. The reduction's result is a row id, which fits `u32` because the
//! partition total does — [`Hash::from_gguf`] refuses a file where it does not.

use std::path::Path;

use gguf::{Inventory, Value};

use crate::EngramError;

/// Every key this module reads is `deepseek41.engram.<suffix>`.
const KEY_PREFIX: &str = "deepseek41.engram.";

/// The hash constants of one model, one set per engram site.
///
/// Built by [`Hash::from_gguf`] from a shard header; nothing here is derived
/// and nothing has a default.
pub struct Hash {
    /// Block index of each site, in the order the metadata lists them. This is
    /// the order every per-site slice below is indexed by.
    layer_ids: Vec<u32>,
    n_heads: usize,
    n_gram: usize,
    /// `(n_gram - 1) * n_heads`: rows one site reads for one token.
    n_cols: usize,
    key_length: u32,
    /// The mapped value a context slot takes before the sequence starts. It is
    /// used as a mapped id directly, not looked up in `token_map`.
    pad: u64,
    /// `sites * n_gram` multipliers, site-major.
    mult: Vec<u64>,
    /// `sites * n_cols` primes, site-major. Each is a divisor, so none is zero.
    prime: Vec<u64>,
    /// `sites * n_cols` offsets, site-major.
    offset: Vec<u64>,
    /// Token id to compressed-vocabulary id, indexed by token id: exactly as
    /// long as the model's vocabulary ([`Hash::check_vocab`]).
    token_map: Vec<u32>,
}

impl Hash {
    /// Read the constants out of one shard's header.
    ///
    /// Every key is mandatory, and so is its length: a file that carries eight
    /// of the nine, or a prime array of the wrong width, is an error naming the
    /// key rather than a set of constants that is silently short.
    pub fn from_gguf(meta: &Inventory) -> Result<Hash, EngramError> {
        let layer_ids: Vec<u32> = ints(meta, "layer_ids")?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| value_err("layer_ids", "a block index")))
            .collect::<Result<_, _>>()?;
        let sites = layer_ids.len();
        if sites == 0 {
            return Err(value_err("layer_ids", "at least one engram site"));
        }

        let n_heads = usize::try_from(scalar(meta, "head_count")?)
            .map_err(|_| value_err("head_count", "a head count"))?;
        let n_gram = usize::try_from(scalar(meta, "max_ngram_size")?)
            .map_err(|_| value_err("max_ngram_size", "an n-gram size"))?;
        // The port's own floor: one head and a 2-gram, because `n_cols` is
        // `(n_gram - 1) * n_heads` and a site with no buckets reads no rows.
        if n_heads == 0 {
            return Err(value_err("head_count", "at least one head"));
        }
        if n_gram < 2 {
            return Err(value_err("max_ngram_size", "at least a 2-gram"));
        }
        let n_cols = (n_gram - 1) * n_heads;

        let key_length = u32::try_from(scalar(meta, "key_length")?)
            .map_err(|_| value_err("key_length", "a key length"))?;
        let pad = scalar(meta, "pad_id")?;

        let mult = ints(meta, "multipliers")?;
        need_len("multipliers", mult.len(), sites * n_gram)?;
        let prime = ints(meta, "primes")?;
        need_len("primes", prime.len(), sites * n_cols)?;
        let offset = ints(meta, "offsets")?;
        need_len("offsets", offset.len(), sites * n_cols)?;

        // A prime is a divisor in the hash and the width of a bucket; zero
        // would divide by zero, and the partition's top must stay inside the
        // `u32` a row id is carried in.
        for (b, (&p, &o)) in prime.iter().zip(&offset).enumerate() {
            if p == 0 {
                return Err(EngramError::KeyValue {
                    key: key_of("primes"),
                    at: b,
                    what: "a non-zero divisor",
                });
            }
            if u32::try_from(o.saturating_add(p)).is_err() {
                return Err(EngramError::KeyValue {
                    key: key_of("offsets"),
                    at: b,
                    what: "a bucket top inside u32",
                });
            }
        }

        let token_map: Vec<u32> = ints(meta, "token_map")?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| value_err("token_map", "a compressed vocab id")))
            .collect::<Result<_, _>>()?;
        if token_map.is_empty() {
            return Err(value_err("token_map", "a non-empty map"));
        }

        Ok(Hash {
            layer_ids,
            n_heads,
            n_gram,
            n_cols,
            key_length,
            pad,
            mult,
            prime,
            offset,
            token_map,
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
        let carrier = invs
            .iter()
            .find(|inv| inv.value(&key_of("layer_ids")).is_some())
            .ok_or_else(|| EngramError::NoHashMetadata(key_of("layer_ids")))?;
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

    /// The context value of a position before the sequence starts.
    pub fn pad_id(&self) -> u64 {
        self.pad
    }

    /// The compressed-vocabulary map, indexed by token id.
    pub fn token_map(&self) -> &[u32] {
        &self.token_map
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

    /// Rows the buckets of `site` cover together — the table's row count if the
    /// partition is exact, which [`Hash::primes`] against the header is how a
    /// gate checks.
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
    pub fn map_token(&self, token: u32) -> Result<u64, EngramError> {
        self.token_map
            .get(token as usize)
            .map(|&v| u64::from(v))
            .ok_or(EngramError::TokenPastMap {
                token,
                map: self.token_map.len(),
            })
    }

    /// Refuse a model whose vocabulary is not the map's domain. The map is
    /// indexed by token id, so it must be exactly `n_vocab` long: shorter, and
    /// a token the model accepts would have no mapped value; longer, and the
    /// file pairs this table with another vocabulary.
    pub fn check_vocab(&self, n_vocab: usize) -> Result<(), EngramError> {
        if self.token_map.len() == n_vocab {
            return Ok(());
        }
        Err(EngramError::VocabMismatch {
            map: self.token_map.len(),
            vocab: n_vocab,
        })
    }

    /// The `n_cols` row ids `site` wants for the token at the head of `ctx`.
    ///
    /// `ctx` is the mapped window — `ctx[0]` the current token, `ctx[s]` the
    /// token `s` positions back, the pad id where that position is before the
    /// sequence start — and `out` is exactly `n_cols` long. Nothing is
    /// allocated: both buffers are the caller's and are reused across tokens.
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

        let mut rolling = ctx[0].wrapping_mul(mult[0]);
        for s in 1..self.n_gram {
            rolling ^= ctx[s].wrapping_mul(mult[s]);
            let base = (s - 1) * self.n_heads;
            for h in 0..self.n_heads {
                let b = base + h;
                let id = rolling % prime[b] + offset[b];
                out[b] = u32::try_from(id)
                    .expect("every bucket top was checked to fit u32 when the constants were read");
            }
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

fn key_of(suffix: &str) -> String {
    format!("{KEY_PREFIX}{suffix}")
}

fn value_err(suffix: &str, what: &'static str) -> EngramError {
    EngramError::KeyValue {
        key: key_of(suffix),
        at: 0,
        what,
    }
}

fn need_len(suffix: &str, got: usize, want: usize) -> Result<(), EngramError> {
    if got == want {
        return Ok(());
    }
    Err(EngramError::KeyLength {
        key: key_of(suffix),
        want,
        got,
    })
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
fn scalar(meta: &Inventory, suffix: &str) -> Result<u64, EngramError> {
    let key = key_of(suffix);
    let v = meta
        .value(&key)
        .ok_or_else(|| EngramError::MissingKey { key: key.clone() })?;
    unsigned(v).ok_or(EngramError::KeyType {
        key,
        want: "a non-negative integer",
    })
}

/// One mandatory array of unsigned integers, widened to `u64`.
fn ints(meta: &Inventory, suffix: &str) -> Result<Vec<u64>, EngramError> {
    let key = key_of(suffix);
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
    use super::Hash;
    use crate::EngramError;

    /// A hash whose map covers token ids `0..vocab`, with the smallest
    /// constants the reader accepts: these tests read the map only.
    fn with_map(vocab: u32) -> Hash {
        Hash {
            layer_ids: vec![1],
            n_heads: 1,
            n_gram: 2,
            n_cols: 1,
            key_length: 256,
            pad: 2,
            mult: vec![1, 1],
            prime: vec![7],
            offset: vec![0],
            token_map: (0..vocab).map(|t| t / 2).collect(),
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
}
