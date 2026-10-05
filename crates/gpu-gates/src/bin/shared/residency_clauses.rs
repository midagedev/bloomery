//! The residency clauses' model-free half, one owner for every gate that
//! runs the residency machine on a real model: V4.1's clauses
//! (`shared/ds41_residency.rs`, run by `gate_ds41_callstream`), GLM's
//! (`gate_glm5next_residency`) and Qwen3.8's (`gate_qwen38_residency`)
//! include this module with a `#[path]` line. It holds `refuse`'s halves
//! ([`refuse_head`] the short host, [`refuse_tail`] the open's timing, the
//! verdict on its error, the `refuse:` line), the seed lists ([`seeds`],
//! [`layer_seeds`]), `transform`'s check and its plain verdict
//! ([`transform_check`]: every
//! admitted expert's slot against its static bytes, [`transform_verdict`]
//! the line), the card table's verdict and its printed line
//! ([`table_clause`]), `static`'s two halves ([`StaticProbe::of`] the read,
//! [`static_row0`] the compare on the static load), the history verdicts
//! ([`c1_clause`], [`passes_clause`], [`c7_clause`]) and the rule walk
//! ([`rule_after`]); a gate supplies what its body forces — its body type
//! and accessors, its views of the residency (`generate::Residence`, the
//! card copy's accessor its body owns: V4.1 `Body::slots`, GLM
//! `Body::slot_copy`), its session open, its churn pool, its layer seeds,
//! its rule reader and pass taker, its constants (the refusal's bound, a
//! history's step count, a source's part count). The readback and the table
//! helpers stay in `generate` ([`Residence`], [`slot_table`], `place_table`).
//! The printed lines are keyed on by the records' readers, so the shared
//! half owns them whole; a gate hands the clause the history's own name for
//! its line.

use std::time::Instant;

use app::{Keep, Prompt, Session};
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::swap::{
    CardTable, PassReport, ResetReport, SlotState, SwapMachine, SwapSource,
};
use bloomery_gpu::host::swap_source::FileSwap;
use bloomery_gpu::{Gpu, window};
use bloomery_gpu_gates::generate::{Residence, ServeFeed, slot_table};
use bloomery_gpu_gates::{GateError, verdict};
use model::placement::Machine;
use runtime::swaprule::SwapRule;
use runtime::{Advance, Target as _, Want};

/// `refuse`'s head: `machine` with its host's usable bytes taken down to
/// one byte under the plan's `headroom_bytes` plus the churn pool's
/// `pool_bytes` — the pair it returns, `short` the byte count the clause's
/// line names and `small` the machine to open on, for [`refuse_tail`].
pub fn refuse_head(
    machine: &Machine,
    headroom_bytes: i128,
    pool_bytes: u64,
) -> Result<(i128, Machine), GateError> {
    let short = i128::from(machine.host.usable_bytes) - headroom_bytes + i128::from(pool_bytes) - 1;
    let mut small = machine.clone();
    small.host.usable_bytes = u64::try_from(short).map_err(|_| {
        format!(
            "refuse: the short host {short} B (usable {} - headroom {headroom_bytes} + pool \
             {pool_bytes} - 1) is below 0",
            machine.host.usable_bytes
        )
    })?;
    Ok((short, small))
}

/// `refuse`'s tail: the open of a host one byte short of the churn pool —
/// `short`, of `pool_bytes` — through `open`, timed: refused by name (its
/// error names the churn pool) within `bound_s` before anything loads, and
/// the `refuse:` line.
pub fn refuse_tail<T>(
    short: i128,
    pool_bytes: u64,
    bound_s: f64,
    open: impl FnOnce() -> Result<T, GateError>,
) -> Result<bool, GateError> {
    let t0 = Instant::now();
    let opened = open();
    let secs = t0.elapsed().as_secs_f64();
    let (ok, why) = match opened {
        Err(e) => {
            let text = e.to_string();
            (text.contains("churn pool") && secs < bound_s, text)
        }
        Ok(_) => (false, "the open loaded".to_string()),
    };
    println!(
        "refuse: a host of {short} B, one byte short of the {} B churn pool: {} in {secs:.1} \
         s — {why}",
        pool_bytes,
        verdict(ok)
    );
    Ok(ok)
}

/// The layers `layers` holds routed experts of, with their seeds
/// (`machine`'s), in their order, the empty seeds left out: [`c7_clause`]'s
/// seed list.
pub fn seeds(
    machine: &SwapMachine,
    layers: impl Iterator<Item = usize>,
) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
    let mut out = Vec::new();
    for l in layers {
        let seed = machine.seed(l)?;
        if !seed.is_empty() {
            out.push((l, seed));
        }
    }
    Ok(out)
}

/// Every layer of `layers` with its seed (`machine`'s), in their order, the
/// empty seeds included: [`transform_check`]'s layer list, which checks each
/// live expert of a layer that its seed does not hold.
pub fn layer_seeds(
    machine: &SwapMachine,
    layers: impl Iterator<Item = usize>,
) -> Result<Vec<(usize, Vec<u32>)>, GateError> {
    let mut out = Vec::new();
    for l in layers {
        out.push((l, machine.seed(l)?));
    }
    Ok(out)
}

/// `transform`'s check: every live non-seed expert of `layers` (each layer
/// with its seed) holds in its slot, part by part (`parts` of a layer), the
/// bytes a static load uploads for it (`source`), read back on `gpu`'s
/// stream — the count checked and each differing part's first byte.
pub fn transform_check(
    gpu: &Gpu,
    machine: &SwapMachine,
    source: &FileSwap,
    layers: &[(usize, Vec<u32>)],
    parts: impl Fn(usize) -> usize,
) -> Result<(usize, Vec<String>), GateError> {
    let stream = gpu.stream();
    stream.synchronize()?;
    let (mut checked, mut bad) = (0usize, Vec::new());
    for (l, seed) in layers {
        let l = *l;
        let parts = parts(l);
        let Some(row) = machine.ledger().row(l) else {
            continue;
        };
        for (slot, st) in row.iter().enumerate() {
            let SlotState::Live(e) = *st else { continue };
            if seed.contains(&e) {
                continue;
            }
            checked += 1;
            for part in 0..parts {
                let want = source.card_bytes(l, e, part)?;
                let at = source.dest(l, part, slot as u32)?;
                // SAFETY: `at` is slot `slot` of layer `l`'s stack `part`,
                // which holds `want.len()` bytes there and stays allocated
                // while the model lives.
                let view = unsafe { window::<u32>(at, want.len() / 4, gpu.context()) };
                let mut got = vec![0u32; want.len() / 4];
                view.copy_to_host(stream, &mut got)?;
                let got: Vec<u8> = got.iter().flat_map(|w| w.to_le_bytes()).collect();
                if got != want {
                    let first = got.iter().zip(want).position(|(a, b)| a != b);
                    bad.push(format!(
                        "layer {l} expert {e} slot {slot} part {part} at {first:?}"
                    ));
                }
            }
        }
    }
    Ok((checked, bad))
}

/// `transform`'s verdict, the plain line: [`transform_check`]'s answer with
/// at least one expert admitted and no part differing (a gate whose clause
/// reads more of the source than the check does prints its own line).
pub fn transform_verdict(checked: usize, bad: &[String]) -> bool {
    let ok = checked > 0 && bad.is_empty();
    println!(
        "transform: {checked} admitted experts, {} parts differ from a static load{}: {}",
        bad.len(),
        bad.first()
            .map(|f| format!(" (first {f})"))
            .unwrap_or_default(),
        verdict(ok)
    );
    ok
}

/// `table`: the stage card's copy of the map after the history — `start`
/// the map the history started from, `landed` the flips it landed, `what`
/// the history's name in the line — off neither the host map nor the
/// machine's ledger, no slot named twice in a layer, and off `start` by at
/// least one entry and at most two a landed flip (each moves its admitted
/// expert onto a slot and its victim off).
pub fn table_clause(
    r: &Residence<'_>,
    start: &[u32],
    landed: usize,
    what: &str,
) -> Result<bool, GateError> {
    let t = r.table()?;
    let c = r.check(&t)?;
    let vs_start = t.differ(start)?;
    slot_table(&t, &c, vs_start).print();
    let ok = c.vs_map == 0
        && c.vs_ledger == Some(0)
        && c.doubled == 0
        && vs_start > 0
        && vs_start <= 2 * landed;
    println!(
        "table: the card's copy after {what}: {} entries off the host map, {:?} off the \
         ledger, {} slots named twice, {vs_start} off the history's start for {} flips landed \
         (0 < {vs_start} <= {}): {}",
        c.vs_map,
        c.vs_ledger,
        c.doubled,
        landed,
        2 * landed,
        verdict(ok)
    );
    Ok(ok)
}

/// The residency's half of `static`: the stage card's copy of the map as
/// the history left it, the prompt `ids` the serve's way after the seat's
/// reset, the row of its last id (row 0), and the boundaries that feed made.
pub struct StaticProbe {
    pub ids: Vec<u32>,
    pub table: CardTable,
    pub row0: Vec<f32>,
    /// Each boundary since the seat's reset: the pass it ended, its number,
    /// the flips that landed there, whether it was made ahead.
    pub passes: Vec<(PassKind, u64, usize, bool)>,
}

impl StaticProbe {
    /// The probe of a history's end: the `ids` its prompt fed, the card's
    /// copy `table` as the history left it, the row of its last id `row0`,
    /// and the boundaries `reports` ended since the seat's reset — each the
    /// pass it ended, its number, the flips that landed there, whether it
    /// was made ahead.
    pub fn of(
        ids: &[u32],
        table: CardTable,
        row0: Vec<f32>,
        reports: &[(PassKind, PassReport)],
    ) -> StaticProbe {
        StaticProbe {
            ids: ids.to_vec(),
            table,
            row0,
            passes: reports
                .iter()
                .map(|(k, r)| (*k, r.boundary, r.landed, r.ahead))
                .collect(),
        }
    }

    /// Whether row 0 ran on the copy read: the feed's last boundary is the
    /// one row 0's step made ahead, after the row, and none before it
    /// landed a flip.
    pub fn quiet(&self) -> bool {
        match self.passes.split_last() {
            Some((&(PassKind::Step, _, _, true), before)) => {
                !before.is_empty() && before.iter().all(|&(_, _, landed, _)| landed == 0)
            }
            _ => false,
        }
    }
}

/// `static`'s row-0 compare, its second half, on `s`: the static load the
/// gate opened (`Loaded::open_edited`, `generate::place_table`, `load_s`
/// its seconds) with `residence` its body's views of the residency, fed
/// `p`'s ids the serve's way. The load holds `p`'s copy's sets on the card
/// and no machine (named), and its row 0 equals `p`'s bit for bit.
pub fn static_row0<B: Prompt + Keep>(
    s: &mut Session<B>,
    residence: impl Fn(&Session<B>) -> Result<Residence<'_>, GateError>,
    p: &StaticProbe,
    load_s: f64,
) -> Result<bool, GateError> {
    let (no_machine, sets) = {
        let r = residence(s)?;
        (r.machine.is_none(), r.table()?.sets_vs(&p.table)?)
    };
    ServeFeed {
        inner: &mut runtime::Plain,
    }
    .prompt(s, &p.ids)?;
    let row0 = s.model().logits()?;
    let (mut off, mut max) = (0usize, 0f32);
    if row0.len() == p.row0.len() {
        for (a, b) in row0.iter().zip(&p.row0) {
            if a.to_bits() != b.to_bits() {
                off += 1;
                max = max.max((a - b).abs());
            }
        }
    }
    let same = row0.len() == p.row0.len() && off == 0;
    let lens = if row0.len() == p.row0.len() {
        String::new()
    } else {
        format!(
            "; {} entries against the probe's {}",
            row0.len(),
            p.row0.len()
        )
    };
    let quiet = p.quiet();
    let ok = quiet && no_machine && sets.set == 0 && same;
    let boundaries: Vec<String> = p
        .passes
        .iter()
        .map(|&(kind, b, landed, ahead)| {
            format!(
                "{b}:{}{}+{landed}",
                kind.word(),
                if ahead { "(ahead)" } else { "" }
            )
        })
        .collect();
    println!(
        "static: the residency's row 0 of the serve's feed of {} ids on its card copy ({} on \
         the card), boundaries since the seat's reset [{}] (no landing before row 0: {quiet}); \
         a static load of that copy in {load_s:.1} s (no machine: {no_machine}; {} experts off \
         its sets, {} at another slot): row 0 bit for bit {same} ({off} of {} entries differ, \
         max {max}{lens}): {}",
        p.ids.len(),
        p.table.on_card(),
        boundaries.join(" "),
        sets.set,
        sets.slot,
        row0.len(),
        verdict(ok)
    );
    Ok(ok)
}

/// `c1`: the history twice gives the same tokens and logits (`same`) and
/// flips land (`landed` of them; `tokens` the history's token count for the
/// line).
pub fn c1_clause(tokens: usize, landed: usize, same: bool) -> bool {
    let c1 = same && landed > 0;
    println!(
        "c1: the history twice, {} tokens, {} flips landed: same {}: {}",
        tokens,
        landed,
        same,
        verdict(c1)
    );
    c1
}

/// `passes`: a history's boundaries end, in order, no pass, the prompt call
/// (one pass, 0 rows kept) and each of `steps` steps (1 kept), the last
/// step's own included: its boundary is made ahead of its readback.
pub fn passes_clause(passes: &[(PassKind, usize)], steps: usize) -> bool {
    // The prompt call is one pass that keeps 0 rows; each step keeps 1. A
    // step's own boundary is made ahead of its readback, so the history's
    // last step ends one too.
    // PIN(2026-10-01): STEPS steps, not STEPS - 1: the last one's boundary runs ahead.
    let mut want = vec![(PassKind::None, 0), (PassKind::Prompt, 0)];
    want.extend(std::iter::repeat_n((PassKind::Step, 1), steps));
    let passes_ok = passes == want.as_slice();
    println!(
        "passes: a history's boundaries end none, the prompt call (0 kept), then {steps} steps \
         (1 kept each): {} boundaries, same {passes_ok}: {}",
        passes.len(),
        verdict(passes_ok)
    );
    passes_ok
}

/// `c7`: the reset `r` brought every seed layer's live set back to its seed
/// (`machine`'s ledger — a layer's live experts its seed's, both ways), its
/// diff 0, and let go of no host byte (`dropped_bytes` 0: the churn pool
/// stays in the host set for the model's life).
pub fn c7_clause(r: &ResetReport, machine: &SwapMachine, seeds: &[(usize, Vec<u32>)]) -> bool {
    let live_is_seed = seeds.iter().all(|(l, seed)| {
        let live: Vec<u32> = machine
            .ledger()
            .row(*l)
            .unwrap_or(&[])
            .iter()
            .filter_map(|st| match st {
                SlotState::Live(e) => Some(*e),
                _ => None,
            })
            .collect();
        live.len() == seed.len() && seed.iter().all(|e| live.contains(e))
    });
    let c7 = r.diff == 0 && live_is_seed && r.dropped_bytes == 0;
    println!(
        "c7: the reset's diff {}, live sets the seed {live_is_seed}, dropped {} B (0: the \
         churn pool stays): {}",
        r.diff,
        r.dropped_bytes,
        verdict(c7)
    );
    c7
}

/// The rule after `pass` from a clear: the prompt, `pass`, then the next
/// boundary, which folds the pass's kept rows — the gate hands it its own
/// `rule_of` and `take_passes`, which reach its body.
pub fn rule_after<B: Prompt + Keep>(
    s: &mut Session<B>,
    ids: &[u32],
    rule_of: impl Fn(&Session<B>) -> Result<SwapRule, GateError>,
    take_passes: impl Fn(&mut Session<B>) -> Result<Vec<(PassKind, PassReport)>, GateError>,
    pass: impl FnOnce(&mut Session<B>, u32) -> Result<(), GateError>,
) -> Result<SwapRule, GateError> {
    s.clear()?;
    let next = s.prompt(ids, Want::Argmax)?.argmax();
    pass(s, next)?;
    s.model_mut().pass_boundary()?;
    let rule = rule_of(s)?;
    take_passes(s)?;
    Ok(rule)
}
