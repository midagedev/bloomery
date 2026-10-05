//! The residency `table` and `static` clauses' model-free half, one owner
//! for every gate that runs the residency machine on a real model: V4.1's
//! clauses (`shared/ds41_residency.rs`, run by `gate_ds41_callstream`) and
//! GLM's (`gate_glm5next_residency`) include this module with a `#[path]`
//! line. It holds the card table's verdict and its printed line
//! ([`table_clause`]), and `static`'s row-0 compare on the static load and
//! its printed line ([`StaticProbe`], [`static_row0`]); a gate supplies
//! what its body forces — its views of the residency (`generate::Residence`,
//! the card copy's accessor its body owns: V4.1 `Body::slots`, GLM
//! `Body::slot_copy`), its session open, its ids. The readback and the
//! table helpers stay in `generate` ([`Residence`], [`slot_table`],
//! `place_table`). The printed lines are keyed on by the records' readers,
//! so the shared half owns them whole; a gate hands the clause the history's
//! own name for its line.

use app::{Keep, Prompt, Session};
use bloomery_gpu::host::PassKind;
use bloomery_gpu::host::swap::CardTable;
use bloomery_gpu_gates::generate::{Residence, ServeFeed, slot_table};
use bloomery_gpu_gates::{GateError, verdict};
use runtime::Advance;

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
         max {max}): {}",
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
