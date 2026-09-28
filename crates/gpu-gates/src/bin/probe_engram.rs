//! Engram page-cache residency probe: how many of the engram rows a prompt
//! asks for sit in the page cache right now — the prompt call's prologue
//! reads them serially, so a cold row is a device read on its critical path.
//! Host only; it maps the V4.1 split set's engram tables and reads no row
//! (`mincore`, `engram::Site::resident_rows`; root sees the whole page cache,
//! anyone else only this process's pages).
//!
//! The rows are the engine's: each site's `n_cols` row ids a position,
//! hashed by the engram crate with the file's constants over the window the
//! port builds (the current token, then each older one, the pad id before
//! the prompt's first position), as `gate_deepseek41_plan` derives them.
//!
//! `probe_engram --check`: the first 16 positions of the file's first ids,
//! every row id inside its table; then a row the page cache does not hold
//! (the first cold one of 4,096 spread over the first table) read once
//! (`copy_rows`) and counted resident. Red when it is not, or when no spread
//! row is cold: the count would not be shown to see the page cache.
//!
//! `probe_engram --ids FILE [--offset O] [--limit N]` (a read-only box
//! command, no lease: it times nothing): the check, then for the `N` (4096)
//! ids from line `O` (0) of `FILE` (one decimal id a line), one record a site
//! and the total — rows asked, distinct, resident. The check reads one row,
//! a spread row of the table, before the prompt's rows are counted (a hash
//! row of the prompt's only by chance).

use std::collections::HashSet;
use std::process::ExitCode;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::record::{self, Record};
use engram::Engram;

/// Positions the check hashes.
const CHECK_POSITIONS: usize = 16;

fn main() -> ExitCode {
    bloomery_gpu_gates::exit_with("probe_engram", run())
}

/// Each site's row ids for `ids`, position by position.
fn rows(eng: &Engram, ids: &[u32]) -> Result<Vec<Vec<u32>>, GateError> {
    let hash = eng.hash();
    let n = hash.n_gram();
    let mut ctx = vec![0u64; n];
    let mut row = vec![0u32; hash.n_cols()];
    let mapped: Vec<u64> = ids
        .iter()
        .map(|&t| hash.map_token(t))
        .collect::<Result<_, _>>()?;
    let mut out = vec![Vec::with_capacity(ids.len() * hash.n_cols()); hash.sites()];
    for p in 0..ids.len() {
        for (s, slot) in ctx.iter_mut().enumerate() {
            *slot = if s <= p { mapped[p - s] } else { hash.pad_id() };
        }
        for (site, o) in out.iter_mut().enumerate() {
            hash.rows_into(site, &ctx, &mut row)?;
            o.extend_from_slice(&row);
        }
    }
    Ok(out)
}

fn read_ids(path: &str) -> Result<Vec<u32>, GateError> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.parse::<u32>()
                .map_err(|e| format!("{path}: '{l}': {e}").into())
        })
        .collect()
}

/// Rows the check tries, spread over the first site's table, for one the
/// page cache does not hold.
const CHECK_SPREAD: u64 = 4096;

/// The check: the prompt's rows are inside their tables, and a row the page
/// cache did not hold counts as resident once it is read — so the count sees
/// the page cache. The row is the first of [`CHECK_SPREAD`] spread over the
/// first site's table that is not resident; none cold is red by name (the
/// count could not be told from "everything resident").
fn check(eng: &Engram, ids: &[u32]) -> Result<bool, GateError> {
    let head = &ids[..ids.len().min(CHECK_POSITIONS)];
    let per_site = rows(eng, head)?;
    let mut in_range = true;
    for (s, r) in eng.sites().iter().zip(&per_site) {
        in_range &= r.iter().all(|&id| u64::from(id) < s.rows());
    }
    let site = &eng.sites()[0];
    let step = (site.rows() / CHECK_SPREAD).max(1);
    let mut cold = None;
    for k in 0..CHECK_SPREAD.min(site.rows()) {
        let id = u32::try_from(k * step).map_err(|_| "a row id past u32")?;
        if site.resident_rows(&[id])? == 0 {
            cold = Some(id);
            break;
        }
    }
    let (saw, pass) = match cold {
        None => (
            format!("no cold row among {CHECK_SPREAD} spread over site 0; ids_in_range={in_range}"),
            false,
        ),
        Some(id) => {
            let mut buf = vec![0u8; site.row_bytes() as usize];
            site.copy_rows(&[id], &mut buf)?;
            let after = site.resident_rows(&[id])?;
            (
                format!(
                    "positions={} sites={} row={id} resident_before=0 resident_after_read={after} ids_in_range={in_range}",
                    head.len(),
                    per_site.len()
                ),
                after == 1 && in_range,
            )
        }
    };
    Record::new(&record::PROBE_CHECK)
        .w("probe", "probe_engram")
        .w("case", "read-row-resident")
        .w("saw", saw)
        .w("pass", pass)
        .print();
    Ok(pass)
}

fn run() -> Result<(), GateError> {
    record::at_main("probe_engram", record::PROBE_ENGRAM_BIN);
    let (mut file, mut offset, mut limit, mut check_only) = (None, 0usize, 4096usize, false);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--check" => check_only = true,
            "--ids" => file = Some(it.next().ok_or("--ids wants a file")?),
            "--offset" | "--limit" => {
                let v: usize = it
                    .next()
                    .ok_or_else(|| format!("{a} wants a value"))?
                    .parse()
                    .map_err(|e| format!("{a}: {e}"))?;
                if a == "--offset" {
                    offset = v;
                } else {
                    limit = v;
                }
            }
            other => return Err(format!("unknown argument '{other}'").into()),
        }
    }
    let eng = Engram::open_dir(gguf::v41::dir())?;
    let (path, ids) = match &file {
        Some(p) => (p.clone(), read_ids(p)?),
        None if check_only => {
            let p = format!(
                "{}/engram/corpus-prose.ids",
                std::env::var("BLOOMERY_DATA").map_err(|_| "BLOOMERY_DATA is unset")?
            );
            let ids = read_ids(&p)?;
            (p, ids)
        }
        None => return Err("want --check, or --ids FILE".into()),
    };
    if ids.is_empty() {
        return Err(format!("{path} holds no id").into());
    }
    let ok = check(&eng, &ids)?;
    if !ok {
        return Err(bloomery_gpu_gates::checks_failed());
    }
    if check_only {
        return Ok(());
    }
    let end = offset
        .checked_add(limit)
        .filter(|&e| e <= ids.len())
        .ok_or_else(|| {
            format!(
                "{path} holds {} ids; ids {offset}..{offset}+{limit} are past it",
                ids.len()
            )
        })?;
    let prompt = &ids[offset..end];
    let per_site = rows(&eng, prompt)?;
    let (mut all_distinct, mut all_resident) = (0u64, 0u64);
    for (e, (site, r)) in eng.sites().iter().zip(&per_site).enumerate() {
        let distinct: Vec<u32> = r
            .iter()
            .copied()
            .collect::<HashSet<u32>>()
            .into_iter()
            .collect();
        let resident = site.resident_rows(&distinct)?;
        all_distinct += distinct.len() as u64;
        all_resident += resident;
        Record::new(&record::PROBE_ENGRAM_SITE)
            .u("layer", eng.hash().layer_ids()[e])
            .u("positions", prompt.len())
            .u("rows", r.len())
            .u("distinct", distinct.len())
            .u("resident", resident)
            .f(
                "resident_share",
                resident as f64 / distinct.len().max(1) as f64,
            )
            .print();
    }
    let name = std::path::Path::new(&path)
        .file_name()
        .map_or_else(|| path.clone(), |n| n.to_string_lossy().into_owned());
    Record::new(&record::PROBE_ENGRAM_TOTAL)
        .w("ids", name)
        .u("offset", offset)
        .u("positions", prompt.len())
        .u("sites", per_site.len())
        .u("distinct", all_distinct)
        .u("resident", all_resident)
        .f(
            "resident_share",
            all_resident as f64 / all_distinct.max(1) as f64,
        )
        .print();
    Ok(())
}
