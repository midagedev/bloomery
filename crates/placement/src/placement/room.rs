//! The host room a plan takes of a reading, and what holds the plan to the
//! load's own reading: the named reserves of what a load allocates between
//! the plan's reading of the host's room and its check's
//! ([`LoadReserves`]), the plan's own heap one of them is sized by
//! ([`plan_heap_bound`]), and the process's own pages each reading carries
//! beside the room ([`process_pages`]) — the witness the reserves are held
//! to, which no other process moves. The readings themselves are
//! `bloomery_gpu::room::HeldRoom`'s, taken once the plan's cards are held.

use std::mem::size_of;

use super::workstation::HostRead;
use super::{CardTotals, ModelTensors, Role, Row, Segment};

/// The reserve of the engine thread a seat loads on: its stack, the
/// thread's mapping of the size it is spawned with.
pub const ENGINE_STACK_RESERVE: &str = "engine_stack";

/// The reserve of what a placed load allocates on its own thread before its
/// check reads the room: the plan it makes again and the body's tables.
pub const LOAD_PREP_RESERVE: &str = "load_prep";

/// The most the allocator adds to a request: glibc's malloc keeps an
/// 8-byte size word before each chunk and rounds chunks to 16 bytes, at
/// least 32 (`MINSIZE` in malloc.c on a 64-bit target), so `n` bytes asked
/// take at most `n + 32`.
pub const CHUNK_BYTES: u64 = 32;

/// What a load allocates of the host between the plan's reading of the
/// host's room and its own check's reading, row by row, each sized from
/// what it is: the plan takes the room the reading leaves past them
/// ([`LoadReserves::room`]), so the load's check, which reads the room once
/// they are spent, finds the plan's host need inside it. Never a host
/// reserve of the machine: the check would count the spent bytes again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReserves {
    rows: Vec<(&'static str, u64)>,
}

impl LoadReserves {
    /// No reserve: a load whose reading plans nothing past its defaults
    /// (the prompt cache's budget, the residency word's room).
    #[must_use]
    pub fn none() -> LoadReserves {
        LoadReserves::default()
    }

    /// These reserves and one more, `name` of `bytes`. A name given twice is
    /// a panic by name: a load declares each reserve once.
    #[must_use]
    pub fn with(mut self, name: &'static str, bytes: u64) -> LoadReserves {
        assert!(
            self.get(name).is_none(),
            "the load reserves {name} already: a load declares each reserve once \
             (LoadReserves::with)"
        );
        self.rows.push((name, bytes));
        self
    }

    /// Every reserve, in the order they were declared.
    #[must_use]
    pub fn rows(&self) -> &[(&'static str, u64)] {
        &self.rows
    }

    /// The reserve named `name`, if declared.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<u64> {
        self.rows.iter().find(|(n, _)| *n == name).map(|&(_, b)| b)
    }

    /// The reserves' sum; `u64::MAX` past it, which no reading holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rows
            .iter()
            .fold(0u64, |sum, &(_, b)| sum.saturating_add(b))
    }

    /// The room a plan takes of `reading` bytes read as `read`: the reading
    /// less every reserve, 0 past it — a reading of this machine
    /// ([`HostRead::MemAvailable`], [`HostRead::Cgroup`]) loses what the
    /// load allocates before its own. A given room ([`HostRead::Given`]) is
    /// no reading: the load's check takes the same number, which nothing
    /// the process allocates moves, so it is the plan's room whole.
    #[must_use]
    pub fn room(&self, reading: u64, read: &HostRead) -> u64 {
        match read {
            HostRead::Given => reading,
            HostRead::MemAvailable | HostRead::Cgroup { .. } => {
                reading.saturating_sub(self.bytes())
            }
        }
    }
}

/// The process's own anonymous and shared resident pages, in bytes, of a
/// `/proc/<pid>/status` text: `RssAnon` plus `RssShmem` — the pages a
/// reading of the host's room loses to this process alone, which no other
/// process moves. A text missing either line, or one whose value is not a
/// whole number of kB, is refused by name.
pub fn process_pages(status: &str) -> Result<u64, String> {
    let kib = |key: &str| -> Result<u64, String> {
        let line = status
            .lines()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(':'))
            .ok_or_else(|| format!("the process status has no {key} line"))?;
        line.trim()
            .strip_suffix("kB")
            .map(str::trim_end)
            .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse::<u64>().ok())
            .and_then(|kb| kb.checked_mul(1024))
            .ok_or_else(|| {
                format!("the process status's {key} is {line:?}, not a whole number of kB")
            })
    };
    kib("RssAnon")?
        .checked_add(kib("RssShmem")?)
        .ok_or_else(|| "the process status's RssAnon and RssShmem pass u64 bytes".to_string())
}

/// The heap a plan of `model` over `cards` cards (stage cards and tiers)
/// holds at most, sized from the model's shape and the plan's own types:
/// the planner's table of optional rows and the plan's row table; every
/// row's segments, one a card, the host's and the NVMe tier's; every routed
/// stack's expert ids once more than its stacks — a stack's segments
/// partition its experts, and the planner clones each layer's lists from
/// one template; the per-card totals and every card's per-layer counts;
/// each allocation with its chunk ([`CHUNK_BYTES`]).
#[must_use]
pub fn plan_heap_bound(model: &ModelTensors, cards: usize) -> u64 {
    let size = |n: usize| n as u64;
    let tensors = size(model.tensors.len());
    let layers = size(model.layers);
    let cards = size(cards);
    let routed = model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts);
    let stacks = size(routed.clone().count());
    let mut routed_layers: Vec<usize> = routed.filter_map(|t| t.layer).collect();
    routed_layers.sort_unstable();
    routed_layers.dedup();
    let segments = cards + 2;
    let rows =
        tensors * (size(size_of::<Option<Row>>()) + size(size_of::<Row>())) + 2 * CHUNK_BYTES;
    let row_segments = tensors * (segments * size(size_of::<Segment>()) + CHUNK_BYTES);
    let lists = (stacks + size(routed_layers.len()))
        * (model.experts * size(size_of::<u32>()) + segments * CHUNK_BYTES);
    let totals = cards * size(size_of::<CardTotals>())
        + CHUNK_BYTES
        + (cards + 1) * (layers * size(size_of::<u64>()) + CHUNK_BYTES);
    rows + row_segments + lists + totals
}

#[cfg(test)]
mod tests {
    use super::{
        CHUNK_BYTES, ENGINE_STACK_RESERVE, LOAD_PREP_RESERVE, LoadReserves, plan_heap_bound,
        process_pages,
    };
    use crate::placement::workstation::HostRead;
    use crate::placement::{CardTotals, ModelTensor, ModelTensors, Role, Row, Segment};
    use gguf::GgmlType;
    use std::mem::size_of;

    /// A plan's room is the reading less every reserve, a given room is its
    /// own number whole, and the sum is each row once.
    #[test]
    fn a_reading_loses_the_reserves_and_a_given_room_none() {
        let r = LoadReserves::none()
            .with(ENGINE_STACK_RESERVE, 2 << 20)
            .with(LOAD_PREP_RESERVE, 600_000);
        assert_eq!(r.bytes(), (2 << 20) + 600_000);
        assert_eq!(r.get(LOAD_PREP_RESERVE), Some(600_000));
        assert_eq!(
            r.room(10 << 30, &HostRead::MemAvailable),
            (10 << 30) - (2 << 20) - 600_000
        );
        let cg = HostRead::Cgroup {
            path: "/a".to_owned(),
        };
        assert_eq!(r.room(1 << 20, &cg), 0, "a reading under the reserves");
        assert_eq!(r.room(10 << 30, &HostRead::Given), 10 << 30);
        assert_eq!(LoadReserves::none().room(7, &HostRead::MemAvailable), 7);
    }

    #[test]
    #[should_panic(expected = "the load reserves engine_stack already")]
    fn a_reserve_is_declared_once() {
        let _ = LoadReserves::none()
            .with(ENGINE_STACK_RESERVE, 1)
            .with(ENGINE_STACK_RESERVE, 2);
    }

    /// The pages are `RssAnon` and `RssShmem` in bytes; a missing line or a
    /// value in another unit is refused by name.
    #[test]
    fn the_process_pages_are_anon_and_shmem() {
        let status = "Name:\tx\nVmRSS:\t  900 kB\nRssAnon:\t  300 kB\nRssFile:\t  500 kB\n\
                      RssShmem:\t  100 kB\n";
        assert_eq!(process_pages(status), Ok(400 << 10));
        let e = process_pages("RssAnon:\t300 kB\n").unwrap_err();
        assert!(e.contains("no RssShmem line"), "{e}");
        let e = process_pages("RssAnon:\t3 MB\nRssShmem:\t1 kB\n").unwrap_err();
        assert!(
            e.contains("RssAnon") && e.contains("not a whole number of kB"),
            "{e}"
        );
    }

    fn tensor(name: &str, layer: Option<usize>, role: Role) -> ModelTensor {
        ModelTensor {
            name: name.to_owned(),
            shard: 0,
            layer,
            role,
            ty: GgmlType::Q4_K,
            dims: vec![256, 1],
            file_bytes: 144,
            gathered_rows: None,
        }
    }

    /// The bound counts each term from the shape: two layers of one dense
    /// tensor and three routed stacks, 8 experts, on one card.
    #[test]
    fn the_plan_heap_bound_is_its_shape() {
        let mut tensors = vec![tensor("token_embd", None, Role::TokenEmbedding)];
        for l in 0..2 {
            tensors.push(tensor(&format!("l{l}.attn"), Some(l), Role::Attention));
            for s in ["gate", "up", "down"] {
                tensors.push(tensor(&format!("l{l}.{s}"), Some(l), Role::RoutedExperts));
            }
        }
        let model = ModelTensors {
            tensors,
            layers: 2,
            experts: 8,
            experts_used: 2,
        };
        let n = 9u64;
        let segments = 3u64;
        let want = n * (size_of::<Option<Row>>() as u64 + size_of::<Row>() as u64)
            + 2 * CHUNK_BYTES
            + n * (segments * size_of::<Segment>() as u64 + CHUNK_BYTES)
            + (6 + 2) * (8 * 4 + segments * CHUNK_BYTES)
            + size_of::<CardTotals>() as u64
            + CHUNK_BYTES
            + 2 * (2 * 8 + CHUNK_BYTES);
        assert_eq!(plan_heap_bound(&model, 1), want);
        assert!(
            plan_heap_bound(&model, 2) > want,
            "a card more, a segment more a row"
        );
    }
}
