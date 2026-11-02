//! The MTP draft step against its reference dump (`tools/mlx-logits.py
//! mtp-dump`): the draft's logits row at each kept position, the hidden row
//! injected, held to a band derived from the step's rounding; and the draft's
//! agreement with the target's own next pick when the hidden row is the
//! session's.
//!
//! The dump is mlx-lm's modules and the sidecar's layer in f32 over the exact
//! dequant, so the gap is the step's own rounding. The band is the established
//! one (`gate_metal_qwen35moe_e2e/row.rs`, `QWEN_LOGIT_ABS`): were every
//! projection's input rounded once into half, a relative error uniform within
//! ±2^-11 (rms 2^-11/√3 = 2.8e-4), a dot carries Δ = 2.8e-4 of its rms, and a
//! logit moves at most `projections · Δ · max|logit|` of the row. The draft
//! step has nine projections (the fusion projection, q, k, v, o, the FFN's
//! gate, up and down, and the head) and the keys and values its cache stores
//! in half: ten roundings of that size, counted as additive. The band is
//! derived row by row from the reference row's own largest logit.
//!
//! The two readings of the hidden row (the row before the target's output
//! norm, `pre`, and after it, `post`) are each run and each held to its own
//! reference. The clause tells them apart: some kept position's two reference
//! rows differ by more than twice the band, so a draft joined with the wrong
//! reading is outside the band, whichever was run.
//!
//! The tapped run feeds the session's own hidden row (the residual a settled
//! host-fed step leaves) and counts the positions where the draft's pick is
//! the session's own next pick: the acceptance rate on this text, beside the
//! reference's count. That count is a measurement, not a band: it is printed
//! and decides nothing here.

use std::path::Path;

use crate::kernels::HiddenRow;

use super::session::{Session, StepMode};

/// The rounding of a projection's input into half: the relative error's rms,
/// 2^-11 / √3 (a value uniform within ±2^-11).
pub const DELTA: f64 = 2.8e-4;

/// The roundings the draft step carries: nine projections and the keys and
/// values the cache stores.
pub const ROUNDINGS: f64 = 10.0;

/// The band of a logit row whose largest logit is `max_logit`.
#[must_use]
pub fn band(max_logit: f32) -> f64 {
    ROUNDINGS * DELTA * f64::from(max_logit.abs())
}

/// FNV-1a 64 over `bytes`: the dump's checksum.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A reference dump, read and checked.
#[derive(Debug)]
pub struct Dump {
    pub ids: Vec<u32>,
    pub n: usize,
    pub hidden: usize,
    pub positions: Vec<usize>,
    /// The target's argmax after the id at each kept position's next step.
    pub target: Vec<u32>,
    /// The reference draft's argmax at each kept position, `pre` then `post`.
    pub picks: [Vec<u32>; 2],
    pub agree: [usize; 2],
    /// The hidden row (before the output norm) of every pair, `hidden` values each.
    pub rows: Vec<f32>,
    /// Each kept position's logits, `pre` then `post`, `n` values each.
    pub logits: Vec<[Vec<f32>; 2]>,
}

/// The ints of a dump text's list line.
fn list(v: &str) -> Result<Vec<u64>, String> {
    if v.is_empty() {
        return Ok(Vec::new());
    }
    v.split(',')
        .map(|x| x.parse::<u64>().map_err(|e| format!("{x:?}: {e}")))
        .collect()
}

/// The dump `<dir>/<family>.txt` and `.f32`: every field, the lengths and the
/// checksum checked, a refusal named.
pub fn read(dir: &Path, family: &str) -> Result<Dump, String> {
    let txt = dir.join(format!("{family}.txt"));
    let bin = dir.join(format!("{family}.f32"));
    let text = std::fs::read_to_string(&txt).map_err(|e| format!("{}: {e}", txt.display()))?;
    let raw = std::fs::read(&bin).map_err(|e| format!("{}: {e}", bin.display()))?;
    let field = |name: &str| -> Result<&str, String> {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' '))
            .ok_or_else(|| format!("{}: no `{name}` line", txt.display()))
    };
    let num = |name: &str| -> Result<usize, String> {
        field(name)?
            .parse::<usize>()
            .map_err(|e| format!("{}: {name}: {e}", txt.display()))
    };
    let ints = |name: &str| -> Result<Vec<u64>, String> {
        list(field(name)?).map_err(|e| format!("{}: {name}: {e}", txt.display()))
    };
    if field("kind")? != "mtp" || num("version")? != 1 {
        return Err(format!("{}: not an mtp dump of version 1", txt.display()));
    }
    let to_u32 = |v: Vec<u64>| -> Vec<u32> { v.into_iter().map(|x| x as u32).collect() };
    let (steps, n, hidden) = (num("steps")?, num("n")?, num("hidden")?);
    let ids = to_u32(ints("ids")?);
    let positions: Vec<usize> = ints("positions")?.into_iter().map(|x| x as usize).collect();
    if ids.len() != steps || steps < 2 || positions.is_empty() {
        return Err(format!(
            "{}: {} ids for {steps} steps and {} positions",
            txt.display(),
            ids.len(),
            positions.len()
        ));
    }
    if positions.windows(2).any(|w| w[1] <= w[0]) || positions[positions.len() - 1] >= steps - 1 {
        return Err(format!(
            "{}: positions {positions:?} are not increasing pairs below {}",
            txt.display(),
            steps - 1
        ));
    }
    let want = ((steps - 1) * hidden + positions.len() * 2 * n) * 4;
    if raw.len() != want {
        return Err(format!(
            "{}: {} bytes, the text names {want}",
            bin.display(),
            raw.len()
        ));
    }
    let sum = format!("{:016x}", fnv1a(&raw));
    if field("fnv1a")? != sum {
        return Err(format!(
            "{}: checksum {sum}, the text names {}",
            bin.display(),
            field("fnv1a")?
        ));
    }
    let floats: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let (rows, rest) = floats.split_at((steps - 1) * hidden);
    let logits = rest
        .chunks(2 * n)
        .map(|c| [c[..n].to_vec(), c[n..].to_vec()])
        .collect();
    let (target, pre, post) = (
        to_u32(ints("target")?),
        to_u32(ints("pre")?),
        to_u32(ints("post")?),
    );
    for (name, v) in [("target", &target), ("pre", &pre), ("post", &post)] {
        if v.len() != positions.len() {
            return Err(format!(
                "{}: {} {name} picks for {} positions",
                txt.display(),
                v.len(),
                positions.len()
            ));
        }
    }
    Ok(Dump {
        ids,
        n,
        hidden,
        positions,
        target,
        picks: [pre, post],
        agree: [num("agree_pre")?, num("agree_post")?],
        rows: rows.to_vec(),
        logits,
    })
}

/// The first index of the largest value, and the largest and second largest.
fn top2(row: &[f32]) -> (u32, f32, f32) {
    let (mut at, mut best, mut second) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            second = best;
            best = v;
            at = i;
        } else if v > second {
            second = v;
        }
    }
    (at as u32, best, second)
}

/// What a check found: its lines, and whether every clause held.
pub struct Checked {
    pub lines: Vec<String>,
    pub ok: bool,
}

fn reading(h: HiddenRow) -> (usize, &'static str) {
    match h {
        HiddenRow::Pre => (0, "pre"),
        HiddenRow::Post => (1, "post"),
    }
}

/// The draft step over `dump`: each reading's injected run against its
/// reference, the clause that tells the readings apart, and each reading's
/// tapped agreement with the session's own next pick. The session is reset
/// before each run. A step that refuses is the check's error.
pub fn check(s: &mut Session, dump: &Dump) -> Result<Checked, String> {
    let mut out = Checked {
        lines: Vec::new(),
        ok: true,
    };
    let say = |out: &mut Checked, ok: bool, line: String| {
        out.ok &= ok;
        out.lines
            .push(format!("{} {line}", if ok { "PASS" } else { "FAIL" }));
    };
    if s.vocab() != dump.n {
        return Err(format!(
            "the dump's vocabulary is {}, the session's {}",
            dump.n,
            s.vocab()
        ));
    }
    let pairs = dump.ids.len() - 1;
    for mode in [HiddenRow::Pre, HiddenRow::Post] {
        let (r, name) = reading(mode);
        s.reset().map_err(|e| e.to_string())?;
        let (mut worst, mut worst_at, mut decided, mut agree) = (0f64, 0usize, 0usize, 0usize);
        let mut over = Vec::new();
        for t in 0..pairs {
            let hid = &dump.rows[t * dump.hidden..(t + 1) * dump.hidden];
            s.set_mtp_hidden(hid).map_err(|e| e.to_string())?;
            let drafted = s
                .draft_step(dump.ids[t + 1], t as u32, mode)
                .map_err(|e| e.to_string())?;
            let Some(k) = dump.positions.iter().position(|&p| p == t) else {
                continue;
            };
            let got = s.logits_row().map_err(|e| e.to_string())?;
            let want = &dump.logits[k][r];
            let max_ref = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let b = band(max_ref);
            let d = got
                .iter()
                .zip(want)
                .map(|(g, w)| f64::from((g - w).abs()))
                .fold(
                    0f64,
                    |m, v| if v.is_nan() { f64::INFINITY } else { m.max(v) },
                );
            if d > worst {
                worst = d;
                worst_at = t;
            }
            if d > b {
                over.push(format!("position {t}: {d:.4} over band {b:.4}"));
            }
            let (ref_pick, top, second) = top2(want);
            if f64::from(top - second) > 2.0 * b {
                decided += 1;
                agree += usize::from(drafted.id == ref_pick);
            }
        }
        say(
            &mut out,
            over.is_empty(),
            format!(
                "mtp draft logits, hidden {name}, injected: {} positions, worst |metal − reference| \
                 {worst:.4} at position {worst_at}; band = {ROUNDINGS} · {DELTA:e} · the row's largest \
                 logit{}",
                dump.positions.len(),
                if over.is_empty() {
                    String::new()
                } else {
                    format!("; over: {}", over.join("; "))
                }
            ),
        );
        say(
            &mut out,
            agree == decided,
            format!(
                "mtp draft pick, hidden {name}, injected: {agree} of {decided} positions whose \
                 reference top-1 leads by more than twice the band are the reference's pick"
            ),
        );
    }
    let distinct = dump
        .logits
        .iter()
        .map(|l| {
            let d = l[0]
                .iter()
                .zip(&l[1])
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0f64, f64::max);
            let max_ref = l[0].iter().fold(0f32, |m, v| m.max(v.abs()));
            (d, band(max_ref))
        })
        .filter(|(d, b)| *d > 2.0 * b)
        .count();
    say(
        &mut out,
        distinct > 0,
        format!(
            "mtp readings are told apart: {distinct} of {} kept positions' pre and post reference \
             rows differ by more than twice the band",
            dump.positions.len()
        ),
    );
    for mode in [HiddenRow::Pre, HiddenRow::Post] {
        let (r, name) = reading(mode);
        s.reset().map_err(|e| e.to_string())?;
        let mut picks = Vec::with_capacity(dump.ids.len());
        let mut drafts = vec![0u32; pairs];
        for (t, &id) in dump.ids.iter().enumerate() {
            picks.push(s.step(id, StepMode::HostFed).map_err(|e| e.to_string())?.id);
            if t < pairs {
                s.tap_mtp_hidden().map_err(|e| e.to_string())?;
                drafts[t] = s
                    .draft_step(dump.ids[t + 1], t as u32, mode)
                    .map_err(|e| e.to_string())?
                    .id;
            }
        }
        let agree = dump
            .positions
            .iter()
            .filter(|&&t| drafts[t] == picks[t + 1])
            .count();
        out.lines.push(format!(
            "INFO mtp tapped, hidden {name}: the draft's pick is the session's next pick at {agree} \
             of {} kept positions (reference: {} of {})",
            dump.positions.len(),
            dump.agree[r],
            dump.positions.len()
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_is_ten_roundings_of_the_rows_largest_logit() {
        assert!((band(25.0) - 10.0 * 2.8e-4 * 25.0).abs() < 1e-12);
        assert_eq!(band(-25.0), band(25.0));
        assert_eq!(band(0.0), 0.0);
    }

    #[test]
    fn the_top_two_of_a_row_are_its_first_largest_and_the_runner_up() {
        assert_eq!(top2(&[1.0, 5.0, 5.0, 2.0]), (1, 5.0, 5.0));
        assert_eq!(top2(&[3.0, 1.0, 2.0]), (0, 3.0, 2.0));
    }

    /// The dump's text and bytes as `tools/mlx-logits.py mtp-dump` writes them
    /// (the script's own self-test holds the writer; this holds the reader): a
    /// flipped byte is a checksum refusal and a short file a length refusal.
    #[test]
    fn a_dump_reads_and_its_checksum_and_lengths_are_held() {
        let dir = std::env::temp_dir().join(format!("bloomery-mtp-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (n, hidden, steps) = (3usize, 2usize, 3usize);
        let floats: Vec<f32> = (0..(steps - 1) * hidden + 2 * n)
            .map(|i| i as f32)
            .collect();
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let text = format!(
            "version 1\nfamily t\nkind mtp\nids 5,6,7\nsteps {steps}\nn {n}\nhidden {hidden}\n\
             positions 1\ntarget 2\npre 2\npost 0\nagree_pre 1\nagree_post 0\nfnv1a {:016x}\n",
            fnv1a(&raw)
        );
        std::fs::write(dir.join("t.txt"), &text).unwrap();
        std::fs::write(dir.join("t.f32"), &raw).unwrap();
        let d = read(&dir, "t").unwrap();
        assert_eq!(d.ids, [5, 6, 7]);
        assert_eq!(d.rows, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(d.logits[0][0], [4.0, 5.0, 6.0]);
        assert_eq!(d.logits[0][1], [7.0, 8.0, 9.0]);
        assert_eq!(d.agree, [1, 0]);
        let mut bad = raw.clone();
        bad[0] ^= 1;
        std::fs::write(dir.join("t.f32"), &bad).unwrap();
        assert!(read(&dir, "t").unwrap_err().contains("checksum"));
        std::fs::write(dir.join("t.f32"), &raw[..raw.len() - 4]).unwrap();
        assert!(read(&dir, "t").unwrap_err().contains("bytes"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
