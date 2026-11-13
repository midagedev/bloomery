//! The K-EXAONE-2.0 loss-tool gate: `ppl_exaone`'s walk (`bloomery_gpu_gates::kld::write_base`, the body of
//! `shared/exaone_ppl.rs`) on the fixture GGUF over the oracle prompts' ids (`tools/ref/data/exaone-moe`, 72 ids as two
//! chunks of 36: 17 scored positions each), held to the engine's own rows by an independent run and to a known loss by
//! a copy of the model whose logits are moved by a known amount. The recipe runs it on the Q4_K fixture and again on
//! the IQ2_XXS one, so both quantizations' paths answer the same clauses.
//!
//! - `base_is_ik_layout`: the walk's file opens as ik's base (the header, the ids as given, one record a scored
//!   position) and every record is one ik's quantizer can write (`Record::check`).
//! - `rows_are_the_steps`: every record is the log-softmax of the row an independent run of the same ids leaves
//!   (the serve seat's own load, a reset and a decode step a position, the logits read back by the seat's call):
//!   each entry above the floor within half its step plus the f32 slack, each entry at level 0 no higher than the
//!   floor says. A walk that feeds the wrong id, scores the wrong position, skips a reset or reads a stale row
//!   leaves a record no row of the independent run is.
//! - `walk_twice_is_kld_zero`: the walk run twice writes the same bytes, so the file against itself, and against its
//!   twin, reads KLD exactly 0, every top id the same, the same NLL.
//! - `perturbed_kld_is_known`: the walk over a copy whose logits entry `i` is moved by `+EPS` for an even `i` and
//!   `-EPS` for an odd one is another base; its KLD against the first, position by position, is the exact
//!   `Σ p (ln p − ln q)` of the two rows within the band the two records' storage derives (every entry above the
//!   floor off by at most half its step plus the f32 slack, the entries at level 0 by the whole floor), and every
//!   position's is positive.
//!
//! The family has no fixture tier: its file is the fixture GGUF itself, so the fixture tier is refused by name.
//!
//! FAIL-first (the mutants, applied on the box): `kld::write_base` resetting once instead of a chunk; `KldWriter::push`
//! taking the id of the position before as `next`; `feed_chunk` feeding `toks[pos + 1]`; `encode_record` rounding
//! down; the perturbation moving the walk's rows by `2 * EPS`.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with(gate::NAME, gate::run())
}

#[cfg(feature = "flint")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the seat serves several gates; each takes the calls it reads"
)]
#[path = "shared/exaone_seat.rs"]
mod seat;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the loss tool and its gate share the body; each takes the calls it reads"
)]
#[path = "shared/exaone_ppl.rs"]
mod ppl;

#[cfg(feature = "flint")]
mod gate {
    use crate::lcpp_e2e::{Arm, Seat as _};
    use bloomery_gpu_gates::kld::{
        F32_SLACK, FLOOR_NATS, KldBase, Pair, PplModel, compare, read_ids, write_base,
    };
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use refset::arch::exaone_moe;
    use std::path::{Path, PathBuf};

    /// The gate's name: the recipe's and the refusals'.
    pub const NAME: &str = "gate_exaone_kld";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-exaone-kld";

    /// Ids per chunk and chunks: the oracle prompts' a, b and c, 24 + 18 + 30 ids.
    const N_CTX: usize = 36;
    const N_CHUNK: usize = 2;
    const PROMPTS: [(&str, usize); 3] = [("a", 24), ("b", 18), ("c", 30)];

    /// The positions the load's caches hold.
    const CTX: usize = 64;

    /// The amount the perturbed copy moves an entry's logit by, up for an even entry and down for an odd one.
    const EPS: f32 = 1.0;

    /// The oracle prompts' ids, `tools/ref/data/exaone-moe/<name>.ids`, one chunk's worth in all.
    fn prompt_ids(vocab: usize) -> Result<Vec<u32>, GateError> {
        let mut ids = Vec::with_capacity(N_CTX * N_CHUNK);
        for (name, n) in PROMPTS {
            let file = format!("tools/ref/data/exaone-moe/{name}.ids");
            ids.extend(read_ids(Path::new(&file), n, vocab)?);
        }
        assert_eq!(ids.len(), N_CTX * N_CHUNK);
        Ok(ids)
    }

    /// The model with its logits moved: entry `i` by `+EPS` for an even `i`, `-EPS` for an odd one.
    struct Perturbed<'a, M: PplModel>(&'a mut M);

    fn delta(i: usize) -> f32 {
        if i.is_multiple_of(2) { EPS } else { -EPS }
    }

    impl<M: PplModel> PplModel for Perturbed<'_, M> {
        fn reset(&mut self) -> Result<(), GateError> {
            self.0.reset()
        }

        fn step(&mut self, token: u32) -> Result<(), GateError> {
            self.0.step(token)
        }

        fn logits(&mut self) -> Result<Vec<f32>, GateError> {
            let mut row = self.0.logits()?;
            for (i, v) in row.iter_mut().enumerate() {
                *v += delta(i);
            }
            Ok(row)
        }
    }

    /// A row's log-probabilities in f64, from the logits.
    fn log_softmax(logits: &[f32]) -> Vec<f64> {
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let sum: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
        let lse = max + sum.ln();
        logits.iter().map(|&l| f64::from(l) - lse).collect()
    }

    /// The independent run: the serve seat's load, every chunk from a reset, one decode step a position, the row at
    /// each scored position, in file order.
    fn reference_rows(path: &Path, ids: &[u32]) -> Result<Vec<Vec<f32>>, GateError> {
        let mut s = crate::seat::open_loaded(path, CTX, Arm::A)?;
        let mut rows = Vec::new();
        for toks in ids.chunks(N_CTX) {
            s.reset()?;
            for (pos, &tok) in toks.iter().enumerate().take(N_CTX - 1) {
                let row = s.step(tok)?;
                if pos >= N_CTX / 2 {
                    rows.push(row);
                }
            }
        }
        Ok(rows)
    }

    /// Clause `base_is_ik_layout` on `base`, the walk's file over `ids`.
    fn layout_clause(base: &KldBase, ids: &[u32], vocab: usize) -> Result<bool, GateError> {
        let name = "base_is_ik_layout";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let want = N_CHUNK * (N_CTX - 1 - N_CTX / 2);
        let mut failed = Vec::new();
        for r in base.records() {
            if let Err(e) = r.check() {
                failed.push(format!("chunk {} position {}: {e}", r.chunk, r.pos));
            }
        }
        for f in failed.iter().take(5) {
            println!("  record {f}");
        }
        let header = (base.n_ctx(), base.n_vocab(), base.n_chunk()) == (N_CTX, vocab, N_CHUNK);
        let pass =
            header && base.tokens() == ids && base.records().len() == want && failed.is_empty();
        println!(
            "{name} n_ctx={} n_vocab={} n_chunk={} records={} (want {want}) ids_as_given={} records_failed={} {}",
            base.n_ctx(),
            base.n_vocab(),
            base.n_chunk(),
            base.records().len(),
            base.tokens() == ids,
            failed.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `rows_are_the_steps`: every record against the independent run's row at its position.
    fn rows_clause(base: &KldBase, rows: &[Vec<f32>]) -> Result<bool, GateError> {
        let name = "rows_are_the_steps";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        if rows.len() != base.records().len() {
            println!(
                "{name}: {} rows, {} records FAIL",
                rows.len(),
                base.records().len()
            );
            return Ok(false);
        }
        let (mut bad, mut worst, mut floored, mut entries) = (0usize, 0.0f64, 0usize, 0usize);
        for (r, row) in base.records().zip(rows) {
            let exact = log_softmax(row);
            let half = f64::from(r.scale) / 2.0 + F32_SLACK;
            let mut here = false;
            for (i, (q, e)) in r.levels().zip(&exact).enumerate() {
                let stored = r.log_prob(i as u32);
                entries += 1;
                if q == 0 {
                    floored += 1;
                    here |= *e > stored + half;
                } else {
                    let d = (stored - e).abs();
                    worst = worst.max(d / half);
                    here |= d > half;
                }
            }
            if here && bad < 5 {
                println!(
                    "  chunk {} position {}: a row of another run",
                    r.chunk, r.pos
                );
            }
            bad += usize::from(here);
        }
        let pass = bad == 0;
        println!(
            "{name} positions={} entries={entries} at_floor={floored} positions_off={bad} \
             worst_entry_in_halfsteps={worst:.3} {}",
            rows.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `walk_twice_is_kld_zero`: the two files of one walk's twice.
    fn twice_clause(a: &Path, b: &Path) -> Result<bool, GateError> {
        let name = "walk_twice_is_kld_zero";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let same_bytes = std::fs::read(a)? == std::fs::read(b)?;
        let (p, q) = (KldBase::open_own_vocab(a)?, KldBase::open_own_vocab(b)?);
        let pair = Pair::new(&p, &q)?;
        let (mut n, mut zero, mut top) = (0usize, 0usize, 0usize);
        for (x, y) in pair.records() {
            let d = compare(&x, &y);
            n += 1;
            zero += usize::from(d.kld == 0.0 && d.nll_p == d.nll_q);
            top += usize::from(d.top_p == d.top_q);
        }
        let pass = same_bytes && n > 0 && zero == n && top == n;
        println!(
            "{name} same_bytes={same_bytes} positions={n} kld_exactly_0={zero} same_top={top} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `perturbed_kld_is_known`: the perturbed copy's KLD against the first, position by position, from
    /// the two exact rows.
    fn perturbed_clause(a: &Path, p: &Path, rows: &[Vec<f32>]) -> Result<bool, GateError> {
        let name = "perturbed_kld_is_known";
        if !tier::run_clause(name, Tag::SelfConsistency)? {
            return Ok(true);
        }
        let (pa, pp) = (KldBase::open_own_vocab(a)?, KldBase::open_own_vocab(p)?);
        let pair = Pair::new(&pa, &pp)?;
        if rows.len() != pair.records().len() {
            println!(
                "{name}: {} rows, {} records FAIL",
                rows.len(),
                pair.records().len()
            );
            return Ok(false);
        }
        let (mut n, mut off, mut nonpos) = (0usize, 0usize, 0usize);
        let (mut sum_got, mut sum_exact, mut worst) = (0.0f64, 0.0f64, 0.0f64);
        for ((x, y), row) in pair.records().zip(rows) {
            let q_row: Vec<f32> = row.iter().enumerate().map(|(i, v)| v + delta(i)).collect();
            let (lp, lq) = (log_softmax(row), log_softmax(&q_row));
            let (ep, eq) = (
                f64::from(x.scale) / 2.0 + F32_SLACK,
                f64::from(y.scale) / 2.0 + F32_SLACK,
            );
            let (mut exact, mut band) = (0.0f64, 0.0f64);
            for (a, b) in lp.iter().zip(&lq) {
                exact += a.exp() * (a - b);
                band += a.exp() * (ep.exp_m1() * (a - b).abs() + ep.exp() * (ep + eq));
            }
            // An entry stored at level 0 stands for any smaller probability: its stored mass is the most it adds.
            let at_floor = x.levels().filter(|&q| q == 0).count() as f64;
            band += at_floor
                * f64::from(x.min_log_prob).exp()
                * (f64::from(EPS) + f64::from(FLOOR_NATS) + ep + eq);
            let got = compare(&x, &y).kld;
            let miss = (got - exact).abs();
            n += 1;
            off += usize::from(miss > band);
            nonpos += usize::from(got <= 0.0);
            sum_got += got;
            sum_exact += exact;
            worst = worst.max(miss / band);
        }
        let pass = n > 0 && off == 0 && nonpos == 0;
        println!(
            "{name} eps={EPS} positions={n} mean_kld={:.6} exact_mean={:.6} positions_outside_band={off} \
             worst_miss_in_bands={worst:.3} not_positive={nonpos} {}",
            sum_got / n as f64,
            sum_exact / n as f64,
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? == Tier::Fixture {
            return Err(format!(
                "{NAME}: the exaone-moe family's file is the fixture GGUF itself, so the gate has \
                 no fixture tier: just {RECIPE}"
            )
            .into());
        }
        let path = crate::lcpp_e2e::model_path(Tier::Real, RECIPE, exaone_moe::ARCH)?;
        let dir = std::env::temp_dir().join(format!("bloomery-{NAME}-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let result = clauses(&path, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    fn clauses(path: &Path, dir: &Path) -> Result<(), GateError> {
        let mut m = crate::ppl::open(path, CTX, Arm::A)?;
        let vocab = m.vocab();
        println!(
            "load arm=a model={} layers={} vocab={vocab} ctx={CTX} chunks={N_CHUNK}x{N_CTX}",
            path.display(),
            m.layers()
        );
        let ids = prompt_ids(vocab)?;
        let file = |tag: &str| -> PathBuf { dir.join(format!("{tag}.kld")) };
        let (a, b, p) = (file("a"), file("b"), file("perturbed"));
        write_base(&mut m, &ids, N_CTX, vocab, &a)?;
        write_base(&mut m, &ids, N_CTX, vocab, &b)?;
        write_base(&mut Perturbed(&mut m), &ids, N_CTX, vocab, &p)?;
        drop(m);
        let rows = reference_rows(path, &ids)?;
        let base = KldBase::open(&a, vocab)?;
        let mut ok = layout_clause(&base, &ids, vocab)?;
        ok &= rows_clause(&base, &rows)?;
        ok &= twice_clause(&a, &b)?;
        ok &= perturbed_clause(&a, &p, &rows)?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the walk of the loss tool writes an ik base whose every record is the log-softmax of the \
             row an independent run leaves, the same bytes twice (KLD exactly 0), and over a copy with its logits moved \
             by ±{EPS} a KLD equal to the exact one of the two rows within the storage's band"
        );
        Ok(())
    }
}

#[cfg(not(feature = "flint"))]
fn main() {
    eprintln!("gate_exaone_kld: built without the `flint` feature");
    std::process::exit(2);
}
