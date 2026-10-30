//! One byte budget and LRU order for a side model's pooled vectors and image rows.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::media::MediaFeed;

/// The server and its side model share this cache for their lifetime.
pub type Shared = Arc<Mutex<Cache>>;

/// A cached result before response normalization, or the tower's BF16 rows.
#[derive(Clone)]
pub enum Value {
    Vector(Vec<f32>),
    Image(Vec<u16>),
}

/// Retained values and entry metadata are charged; container slack and allocator overhead are not.
/// `charge` prices one value; the server's prepared images share this order through their own.
pub struct Cache<T = Value> {
    limit: usize,
    bytes: usize,
    entries: VecDeque<([u8; 32], T)>,
    charge: fn(&T) -> usize,
}

impl Cache {
    /// Zero disables caching without allocating an owner.
    #[must_use]
    pub fn shared(limit: usize) -> Option<Shared> {
        (limit > 0).then(|| Arc::new(Mutex::new(Self::new(limit, Self::charge_value))))
    }

    fn charge_value(value: &Value) -> usize {
        size_of::<([u8; 32], Value)>()
            + match value {
                Value::Vector(v) => v.capacity() * size_of::<f32>(),
                Value::Image(v) => v.capacity() * size_of::<u16>(),
            }
    }
}

impl<T: Clone> Cache<T> {
    /// A cache of at most `limit` charged bytes; zero retains nothing.
    #[must_use]
    pub fn new(limit: usize, charge: fn(&T) -> usize) -> Self {
        Self {
            limit,
            bytes: 0,
            entries: VecDeque::new(),
            charge,
        }
    }

    /// A hit becomes the most recently used entry. The search starts at that end, where a repeated
    /// request's entry sits.
    pub fn get(&mut self, key: &[u8; 32]) -> Option<T> {
        let at = self.entries.iter().rposition(|(k, _)| k == key)?;
        let entry = self.entries.remove(at).expect("the entry was found");
        let value = entry.1.clone();
        self.entries.push_back(entry);
        Some(value)
    }

    /// Oversized results are not retained and do not evict useful entries.
    pub fn insert(&mut self, key: [u8; 32], value: T) {
        let bytes = (self.charge)(&value);
        if bytes > self.limit {
            return;
        }
        if let Some(at) = self.entries.iter().position(|(k, _)| *k == key) {
            let (_, old) = self.entries.remove(at).expect("the entry was found");
            self.bytes -= (self.charge)(&old);
        }
        while self.bytes > self.limit - bytes {
            let (_, old) = self.entries.pop_front().expect("charged entries exist");
            self.bytes -= (self.charge)(&old);
        }
        self.bytes += bytes;
        self.entries.push_back((key, value));
    }

    /// The charged bytes retained now.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The bound the charged bytes stay within.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The entries retained now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A file and its prepared grid identify a tower input independently of its text position.
/// Non-grid inputs are not retained by this cache.
#[must_use]
pub fn image_key(image: &MediaFeed) -> Option<[u8; 32]> {
    let vision::EncoderInput::Grid(grid) = &image.prepared.input else {
        return None;
    };
    let mut hash = Sha256::new();
    hash.update(b"image");
    hash.update(image.key.0);
    for n in [
        image.prepared.span_len,
        grid.n_vit_h,
        grid.n_vit_w,
        grid.patch_len,
        grid.plan.span_grid().0,
        grid.plan.span_grid().1,
        grid.plan.best_h,
        grid.plan.best_w,
    ] {
        hash.update(n.to_le_bytes());
    }
    Some(hash.finalize().into())
}

/// The pooled result depends on every token and each whole image at its position.
#[must_use]
pub fn vector_key(ids: &[u32], images: &[MediaFeed]) -> Option<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"vector");
    hash.update(ids.len().to_le_bytes());
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    for image in images {
        hash.update(image.at.to_le_bytes());
        hash.update(image_key(image)?);
    }
    Some(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_include_image_identity_shape_and_position_but_not_tower_text() {
        let image = |key, at, span_len| MediaFeed {
            key: crate::media::ImageKey([key; 32]),
            at,
            prepared: Arc::new(vision::Prepared::of_grid(
                span_len,
                vision::Patches {
                    plan: vision::GridPlan {
                        n_llm_h: 1,
                        n_llm_w: span_len,
                        best_h: 16,
                        best_w: span_len * 16,
                    },
                    n_vit_h: 1,
                    n_vit_w: span_len,
                    patch_len: 1,
                    bf16: vec![0; span_len],
                },
            )),
        };
        let a = image(1, 2, 4);
        let shifted = image(1, 3, 4);
        assert_eq!(image_key(&a), image_key(&shifted));
        assert_ne!(image_key(&a), image_key(&image(2, 2, 4)));
        assert_ne!(image_key(&a), image_key(&image(1, 2, 8)));
        assert_ne!(
            vector_key(&[1, 2], std::slice::from_ref(&a)),
            vector_key(&[1, 2], &[shifted])
        );
        assert_ne!(
            vector_key(&[1, 2], std::slice::from_ref(&a)),
            vector_key(&[2, 1], &[a])
        );
    }

    #[test]
    fn one_budget_evicts_the_least_recent_vector_or_image() {
        assert!(Cache::shared(0).is_none());
        let v = || Value::Vector(vec![1.0, -0.0]);
        let bytes = Cache::charge_value(&v());
        let shared = Cache::shared(2 * bytes).unwrap();
        let mut cache = shared.lock().unwrap();
        cache.insert([1; 32], v());
        cache.insert([2; 32], Value::Image(vec![1, 2, 3, 4]));
        let Some(Value::Vector(hit)) = cache.get(&[1; 32]) else {
            panic!("vector hit")
        };
        assert_eq!(
            hit.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![1.0f32.to_bits(), (-0.0f32).to_bits()]
        );
        cache.insert([3; 32], v());
        assert!(cache.get(&[2; 32]).is_none());
        cache.insert([4; 32], Value::Image(vec![0; 4 * bytes]));
        assert!(cache.get(&[4; 32]).is_none());
        assert!(cache.get(&[3; 32]).is_some());
        assert!(cache.bytes <= cache.limit);
        assert_ne!(vector_key(&[1, 2], &[]), vector_key(&[2, 1], &[]));
        assert_ne!(vector_key(&[1], &[]), vector_key(&[1, 0], &[]));
    }

    #[test]
    fn a_key_holds_one_entry_and_a_hit_is_found_wherever_it_sits() {
        let v = |x: f32| Value::Vector(vec![x]);
        let shared = Cache::shared(8 * Cache::charge_value(&v(0.0))).unwrap();
        let mut cache = shared.lock().unwrap();
        for k in 1..=4u8 {
            cache.insert([k; 32], v(f32::from(k)));
        }
        cache.insert([2; 32], v(20.0));
        assert_eq!(cache.entries.len(), 4);
        for (k, want) in [
            (1u8, 1.0f32),
            (4, 4.0),
            (2, 20.0),
            (3, 3.0),
            (2, 20.0),
            (1, 1.0),
        ] {
            let Some(Value::Vector(hit)) = cache.get(&[k; 32]) else {
                panic!("key {k} missing");
            };
            assert_eq!(hit, vec![want], "key {k}");
            assert_eq!(cache.entries.back().map(|(key, _)| key[0]), Some(k));
            assert_eq!(cache.entries.len(), 4);
        }
        assert!(cache.get(&[9; 32]).is_none());
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|(key, _)| key[0])
                .collect::<Vec<_>>(),
            [4, 3, 2, 1]
        );
    }
}
