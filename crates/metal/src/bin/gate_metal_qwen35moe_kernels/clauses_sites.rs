//! The site clause: the weight types the arena's sites admit, held to the
//! launches that run them.
//!
//! [`Quant::ALL`] is the one map from a file type to a kernel type, and
//! [`Quant::dense`], [`Quant::embeds`], [`Quant::heads`] and
//! [`Quant::headnorms`] name the types the plain mat-vec (and with it the
//! GEMMs and the experts), the embedding, the head and the head-norm
//! projection have an entry for: the arena's checks read them. A type a predicate admits must run in its
//! launch, one a predicate refuses must be refused by that launch by name, and
//! a predicate that moves without its launch is the finding. The numbers of
//! the admitted launches are the families' own clauses (`*_embed`, `*_head`,
//! `*_headnorm`).

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::kernels::Quant;

use super::clauses_embed::run_embed;
use super::clauses_head::{self, HeadRun};
use super::clauses_norm::head_partials;
use super::common::{self, LAYER};
use super::{ClauseRun, Harness, HeadNormLaunch, Launch, ModeSpec, oracle, synthetic};

/// The epsilon every launch of the clause takes.
const EPS: f32 = 1e-6;

/// The embedding table's rows of the clause: the launch embeds one of them.
const VOCAB: usize = 2;

/// The head's rows of the clause: a count no threadgroup divides.
const HEAD_ROWS: usize = 40;

/// The head-norm projection's rows of the clause.
const OUT_ROWS: usize = 64;

pub fn table<'a>(h: &'a Harness<'a>) -> Vec<(&'static str, ClauseRun<'a>)> {
    vec![("site_types", Box::new(move || site_types(h)))]
}

/// `launched` against `admitted`: an admitted type ran, a refused one was
/// refused by the launch with `entry` in its words.
fn agrees(
    site: &str,
    quant: Quant,
    admitted: bool,
    entry: &str,
    launched: Result<(), String>,
) -> Result<(), String> {
    let name = quant.name();
    match launched {
        Ok(()) if admitted => Ok(()),
        Ok(()) => Err(format!("{site} refuses {name} and its launch ran it")),
        Err(e) if !admitted && e.contains(entry) => Ok(()),
        Err(e) if admitted => Err(format!(
            "{site} admits {name} and its launch refused it: {e}"
        )),
        Err(e) => Err(format!(
            "{site} refuses {name} and its launch refused it for another reason: {e}"
        )),
    }
}

fn site_types(h: &Harness<'_>) -> Clause {
    for quant in Quant::ALL {
        if Quant::of_ggml(quant.ggml()) != Some(quant) {
            return Err(format!(
                "{}: the file type {:?} does not map back to it",
                quant.name(),
                quant.ggml()
            ));
        }
    }
    let delta = &h.geom().delta;
    let hidden = h.geom().hidden;
    let cols = delta.head_cols * delta.value_heads;
    let partials = head_partials(delta);
    let g = oracle::activations(cols, 0x5170, 1.0);
    let resid = oracle::activations(OUT_ROWS, 0x5171, 2.0);
    let x = oracle::activations(hidden, 0x5172, 1.0);
    let gain = vec![1f32; hidden];
    let (mut denses, mut embeds, mut heads, mut headnorms) = (0usize, 0usize, 0usize, 0usize);
    for quant in Quant::ALL {
        let seed = (quant.block_bytes() as u64) << 24;
        h.clear_fault()?;

        let weights = synthetic::matrix(quant, OUT_ROWS, hidden, seed + 0x63);
        let launched = h
            .matvec(&Launch {
                quant,
                rows: OUT_ROWS,
                cols: hidden,
                weights: &weights,
                x: &x,
                y0: &common::sentinel_rows(OUT_ROWS),
                mode: ModeSpec::Store,
            })
            .map(|_| ());
        agrees(
            "matvec",
            quant,
            quant.dense(),
            "has no plain mat-vec entry",
            launched,
        )?;
        denses += usize::from(quant.dense());

        let table = synthetic::matrix(quant, VOCAB, 2048, seed + 0x60);
        let launched = run_embed(h, quant, &table, VOCAB, &[1]).map(|_| ());
        agrees(
            "embed",
            quant,
            quant.embeds(),
            "has no embedding entry",
            launched,
        )?;
        embeds += usize::from(quant.embeds());

        let weights = synthetic::matrix(quant, HEAD_ROWS, hidden, seed + 0x61);
        let launched = clauses_head::run(
            h,
            HeadRun {
                quant,
                weights: &weights,
                x: &x,
                gain: &gain,
                rows: HEAD_ROWS,
                k: hidden,
                eps: EPS,
            },
        )
        .map(|_| ());
        agrees("head", quant, quant.heads(), "has no head entry", launched)?;
        heads += usize::from(quant.heads());

        let weights = synthetic::matrix(quant, OUT_ROWS, cols, seed + 0x62);
        let launched = h
            .headnorm(&HeadNormLaunch {
                quant,
                rows: OUT_ROWS,
                cols,
                weights: &weights,
                g: &g,
                partials: &partials,
                y0: &resid,
                eps: EPS,
                layer: LAYER,
            })
            .map(|_| ());
        agrees(
            "headnorm",
            quant,
            quant.headnorms(),
            "has no head-normed mat-vec",
            launched,
        )?;
        headnorms += usize::from(quant.headnorms());
        h.clear_fault()?;
    }
    Ok(Verdict::Noted(format!(
        "{} types map both ways; the plain mat-vec admits {denses}, the embedding {embeds}, the \
         head {heads} and the head-norm projection {headnorms}, each the launch's own answer",
        Quant::ALL.len()
    )))
}
