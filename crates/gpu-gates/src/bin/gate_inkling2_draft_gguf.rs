//! GPU gate for Inkling 2's DSpark draft opened from its GGUF file (llama.cpp's DSpark convention:
//! the one `dspark-*.gguf` beside the target, `refset::arch::draft::dspark_beside`), beside the same
//! draft opened from the official folder, both on one load of the Mini's GGUF target, in its shadow.
//!
//!     gate_inkling2_draft_gguf --draft <folder> --prompts <dir> [--gen <ids>]
//!
//! The target is the profile's file (`BLOOMERY_MODEL=inkling2`), the GGUF draft the one beside it,
//! the folder's `config.json` and `mtp.safetensors` the same draft as its publisher trained it;
//! `<dir>` holds each corpus's prompt as `gate_inkling2_draft` reads it. Both drafts load through
//! the common owner's one load (`bloomery_gpu::dspark::anchor_draft::AnchorDraft`), each from its
//! own weight source (`Bf16Source`): the GGUF's Q8_0 matrices go to the card with the file's blocks
//! unchanged, the folder's BF16 ones are quantized on the host by the engine's Q8_0 rule. Both open
//! through the one opener the inkling2 seat of `bloomery-serve` serves them by
//! (`shared/inkling2_draft.rs`).
//!
//! Clause `gguf_states_the_folder` (host, before any card work): over the target's header the GGUF's
//! keys state the folder config's draft (`hf::dspark::inkling2::read_gguf` equals `read`), its
//! header holds every tensor the draft reads and no other (`check_gguf`), every F32 and BF16 tensor
//! is the folder's bf16 bit for bit, and every Q8_0 value `d·q` lies within [`Q8_BOUND`]`·d` of the
//! folder's bf16 value. Same-shape tensors a wrong name map would swap (the two context projections,
//! gate and up, the two final gains) differ in value, so the clause holds the map too.
//!
//! The shadow: the target steps one position at a time, greedy, as in `gate_inkling2_draft`; each
//! position's taps are appended to both drafts, and before each generated position but the end both
//! propose [`WIDTH`] ids after the target's id.
//!
//! Clause `gguf_reserves_its_bytes` (host, before any card work): a plan sets the GGUF draft's bytes
//! aside as it does the folder's, from the opened draft: the per-position part is the folder's and
//! not zero, and the fixed part differs from the folder's by exactly the two weight files' lengths.
//!
//! Clause `gguf_proposes_as_the_folder` (self-consistency, real tier): over every proposal of the
//! three corpora, the GGUF draft's first id equals the folder draft's at a share of at least
//! [`AGREE`]. The first id alone: later ids chain through the recurrent head, so one flip there moves
//! every id after it.
//!
//! Clause `gguf_acceptance_in_band` (oracle, real tier): per corpus, the GGUF draft's τ over a
//! speculative schedule is within [`BAND`] of the reference's τ, as `gate_inkling2_draft` holds the
//! folder's; the folder draft's τ on the same walk is printed beside it.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_draft_gguf", gate::run())
}

#[cfg(not(feature = "gpu"))]
fn main() {}

// The draft's one opener: the inkling2 seat of `bloomery-serve` opens the same files through it.
#[cfg(feature = "gpu")]
#[path = "shared/inkling2_draft.rs"]
#[allow(
    dead_code,
    reason = "the gate and the inkling2 seat share the opener; each takes the parts it reads"
)]
mod inkling2_draft;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::inkling2_draft::{self as opener, Draft as Opened};
    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::dspark::DsparkBody;
    use bloomery_gpu::dspark::ctx_dense::CtxDenseLayers;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::record::DRAFT_ACCEPTANCE;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use bloomery_record::Record;
    use gguf::quant::Q8Block;
    use gguf::{GgmlType, Split};
    use hf::dspark::inkling2::{self as config, Draft};
    use model::arch::Arch;
    use model::arch::inkling2::hparams::Hparams;
    use refset::arch::draft::dspark_beside;
    use safetensors::Safetensors;

    const NAME: &str = "gate_inkling2_draft_gguf";
    const RECIPE: &str = "gate-gpu-inkling2-draft-gguf";
    const FILE_CLAUSE: &str = "gguf_states_the_folder";
    const RESERVE_CLAUSE: &str = "gguf_reserves_its_bytes";
    const AGREE_CLAUSE: &str = "gguf_proposes_as_the_folder";
    const TAU_CLAUSE: &str = "gguf_acceptance_in_band";

    /// Ids a block proposes: the reference's width.
    const WIDTH: usize = 7;
    /// Ids the target generates after each prompt at most: the reference run's.
    const GEN: usize = 1024;
    /// The positions the target and the drafts' global layers hold, rounded up to this.
    const CTX_STEP: usize = 512;
    /// The relative distance from the reference's τ the acceptance clause allows: the folder's.
    const BAND: f64 = 0.15;

    /// A Q8_0 value's distance from the bf16 it was made from, in its block's stored scale `d`: half
    /// a code, plus the f16 rounding of the scale (2^-11 of it) over at most 127 codes, plus 2^-10 of
    /// slack for the f32 products the reference rule takes.
    const Q8_BOUND: f64 = (0.5 + 127.0 / 2048.0) * (1.0 + 1.0 / 1024.0);

    /// PIN(2026-11-25): the GGUF draft's first id agrees with the folder draft's at this share at
    /// least. [derived] The two card images are both Q8_0 of the same bf16, by ggml's weight-file
    /// rule (the GGUF) and the engine's x86 rule (the folder): their blocks differ in 2.3 % of
    /// blocks [measured: 190,141 of 8,183,808], by one code. One code is `d ≈ amax/127 ≈ 0.02 σ_w`,
    /// at `f ≈ 0.023/32` of the weights, so a projection's output moves by `√f · 0.02 ≈ 5e-4` of its
    /// spread, against the 0.6 % both images already carry from the rounding itself; through the
    /// three layers' ~30 projections about `3e-3` of the final row, `≈ 0.01–0.015` of a logit at
    /// the head's cap. An argmax flips where the top two logits sit closer than that: ~1 % of rows at
    /// a margin density of ~0.2 per logit near zero, so 99 % agree, held at 98 %.
    const AGREE: f64 = 0.98;

    /// A corpus: its name, its ids file and the reference's τ on it.
    struct Corpus {
        name: &'static str,
        file: &'static str,
        reference: f64,
    }

    /// PIN(2026-11-23): `gate_inkling2_draft`'s, llama.cpp with the draft's upstream PR over a
    /// Q8_0-dense Mini, greedy, the draft proposing [`WIDTH`] ids a block (`#mean acc len`).
    const CORPORA: [Corpus; 3] = [
        Corpus {
            name: "chat",
            file: "chat.ids",
            reference: 2.15,
        },
        Corpus {
            name: "code",
            file: "code4096.ids",
            reference: 5.70,
        },
        Corpus {
            name: "prose",
            file: "prose4096.ids",
            reference: 4.27,
        },
    ];

    struct Args {
        folder: PathBuf,
        prompts: PathBuf,
        n_gen: usize,
    }

    fn args() -> Result<Args, GateError> {
        let (mut folder, mut prompts, mut n_gen) = (None, None, GEN);
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--draft" => folder = it.next().map(PathBuf::from),
                "--prompts" => prompts = it.next().map(PathBuf::from),
                "--gen" => {
                    let v = it.next().ok_or(format!("{NAME}: --gen takes a count"))?;
                    n_gen = v
                        .parse()
                        .ok()
                        .filter(|&n| n > WIDTH)
                        .ok_or(format!("{NAME}: --gen {v}: a count above {WIDTH}"))?;
                }
                other => return Err(format!("{NAME}: unknown argument {other}").into()),
            }
        }
        Ok(Args {
            folder: folder.ok_or(format!("{NAME}: --draft <folder> is required"))?,
            prompts: prompts.ok_or(format!("{NAME}: --prompts <dir> is required"))?,
            n_gen,
        })
    }

    /// One id a line; an empty file or a line that is not an id is refused by name.
    fn read_ids(path: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.trim()
                    .parse::<u32>()
                    .map_err(|e| format!("{}: {l:?}: {e}", path.display()))
            })
            .collect::<Result<Vec<u32>, String>>()?;
        if ids.is_empty() {
            return Err(format!("{}: no ids", path.display()).into());
        }
        Ok(ids)
    }

    /// The file-against-file clause: the GGUF's statement against the folder's, and each tensor's
    /// values against the folder's bf16. Returns whether it holds; the first difference is printed.
    fn file_clause(
        d: &Draft,
        gguf_draft: &Draft,
        split: &Split,
        st: &Safetensors,
    ) -> Result<bool, GateError> {
        if gguf_draft != d {
            println!("clause {FILE_CLAUSE}: the GGUF states {gguf_draft:?}, the folder {d:?}");
            return Ok(false);
        }
        if let Err(e) = config::check_gguf(d, split) {
            println!("clause {FILE_CLAUSE}: {e}");
            return Ok(false);
        }
        let (mut exact, mut q8, mut worst) = (0usize, 0usize, 0.0f64);
        for (name, _) in config::tensors(d) {
            let g = config::gguf_name(&name).ok_or(format!("{NAME}: {name} has no GGUF name"))?;
            let (bits, _) = st.bf16(&name).map_err(|e| format!("{NAME}: {e}"))?;
            let (_, info) = split
                .find(&g)
                .ok_or(format!("{NAME}: {g} is not in the file"))?;
            let bytes = split.data(&g).map_err(|e| format!("{NAME}: {g}: {e}"))?;
            let widen = |b: u16| f32::from_bits(u32::from(b) << 16);
            match info.ty {
                GgmlType::F32 | GgmlType::BF16 => {
                    let same =
                        if info.ty == GgmlType::F32 {
                            bytes.as_chunks::<4>().0.iter().zip(bits).all(|(v, &b)| {
                                f32::from_le_bytes(*v).to_bits() == widen(b).to_bits()
                            })
                        } else {
                            bytes
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .zip(bits)
                                .all(|(v, &b)| u16::from_le_bytes(*v) == b)
                        };
                    if !same {
                        println!("clause {FILE_CLAUSE}: {g} is not {name}'s values bit for bit");
                        return Ok(false);
                    }
                    exact += 1;
                }
                GgmlType::Q8_0 => {
                    for (blk, vals) in bytes
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .zip(bits.as_chunks::<32>().0)
                    {
                        let b = Q8Block::from_bytes(blk);
                        let d16 = f64::from(gguf::quant::half_to_f32(b.d));
                        for (&q, &v) in b.q.iter().zip(vals) {
                            let off = (d16 * f64::from(q) - f64::from(widen(v))).abs();
                            if off > Q8_BOUND * d16 {
                                println!(
                                    "clause {FILE_CLAUSE}: {g} holds {} where {name} holds {}, \
                                     past {Q8_BOUND:.4}·d (d {d16:e})",
                                    d16 * f64::from(q),
                                    widen(v)
                                );
                                return Ok(false);
                            }
                            if d16 > 0.0 {
                                worst = worst.max(off / d16);
                            }
                        }
                    }
                    q8 += 1;
                }
                ty => return Err(format!("{NAME}: {g} is {ty}, which check_gguf refuses").into()),
            }
        }
        println!(
            "clause {FILE_CLAUSE}: the GGUF states the folder's draft; {exact} F32/BF16 tensors bit \
             for bit, {q8} Q8_0 tensors within {worst:.4}·d of the bf16 (bound {Q8_BOUND:.4}) {}",
            verdict(true)
        );
        Ok(true)
    }

    /// A draft's proposals, one block of [`WIDTH`] ids a generated position.
    type Proposals = Vec<[u32; WIDTH]>;

    /// Both drafts in the target's shadow over `prompt`: the target's continuation, `n_gen` ids or
    /// up to the first of `eog`, and at each generated position before its end each draft's
    /// proposal.
    fn shadow(
        m: &mut GpuModel<Body>,
        drafts: [&mut DsparkBody<CtxDenseLayers>; 2],
        prompt: &[u32],
        n_gen: usize,
        eog: &[u32],
    ) -> Result<(Vec<u32>, [Proposals; 2]), GateError> {
        let [a, b] = drafts;
        m.reset()?;
        a.reset(m.gpu())?;
        b.reset(m.gpu())?;
        let mut next = 0;
        for (p, &t) in prompt.iter().enumerate() {
            next = m.step(&[t])?;
            let taps = m.draft_taps()?;
            a.append_as(m.gpu(), u32::try_from(p)?, &taps)?;
            b.append_as(m.gpu(), u32::try_from(p)?, &taps)?;
        }
        let mut cont = Vec::with_capacity(n_gen + 1);
        let mut proposals = [Vec::with_capacity(n_gen), Vec::with_capacity(n_gen)];
        for g in 0..n_gen {
            let pos = u32::try_from(prompt.len() + g)?;
            cont.push(next);
            if eog.contains(&next) {
                break;
            }
            for (d, out) in [&mut *a, &mut *b].into_iter().zip(&mut proposals) {
                let mut ids = [0u32; WIDTH];
                d.propose_as(m.gpu(), pos, next, WIDTH, &mut ids, &mut [])?;
                out.push(ids);
            }
            next = m.step(&[next])?;
            let taps = m.draft_taps()?;
            a.append_as(m.gpu(), pos, &taps)?;
            b.append_as(m.gpu(), pos, &taps)?;
        }
        if proposals[0].len() == n_gen {
            cont.push(next);
        }
        Ok((cont, proposals))
    }

    /// Per generated position, the ids `proposals` share with the continuation after it.
    fn accepted(cont: &[u32], proposals: &[[u32; WIDTH]]) -> Vec<usize> {
        proposals
            .iter()
            .enumerate()
            .map(|(g, ids)| {
                ids.iter()
                    .zip(&cont[g + 1..])
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .collect()
    }

    /// The speculative schedule over `accepted`: a block at the first position, the next at the
    /// first position after a block's accepted ids. Returns each block's accepted ids.
    fn schedule(accepted: &[usize]) -> Vec<usize> {
        let mut blocks = Vec::new();
        let mut q = 0;
        while let Some(&k) = accepted.get(q) {
            blocks.push(k);
            q += k + 1;
        }
        blocks
    }

    fn tau(blocks: &[usize]) -> f64 {
        1.0 + blocks.iter().sum::<usize>() as f64 / blocks.len().max(1) as f64
    }

    /// The reserve clause: a plan's bytes for the GGUF draft are the folder's terms with the GGUF's
    /// own file, so a draft opened from a file never reserves nothing. Returns whether it holds.
    fn reserve_clause(folder: &Opened, gguf: &Opened, hp: &Hparams) -> bool {
        let args = (8, hp.n_embd);
        let (pf, pg) = (folder.reserve_per_pos(), gguf.reserve_per_pos());
        let (ff, fg) = (folder.reserve_fixed(args), gguf.reserve_fixed(args));
        let len = |d: &Opened| std::fs::metadata(&d.file).map_or(0, |m| m.len());
        let (lf, lg) = (len(folder), len(gguf));
        let pass = pg > 0
            && pg == pf
            && lg > 0
            && lf > 0
            && i128::from(fg) - i128::from(ff) == i128::from(lg) - i128::from(lf)
            && fg > lg;
        println!(
            "clause {RESERVE_CLAUSE}: per position {pg} (the folder's {pf}), fixed {fg} over a file \
             of {lg} bytes (the folder's {ff} over {lf}) {}",
            verdict(pass)
        );
        pass
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        let target_path = bloomery_gpu_gates::ref_model_path()?;
        let gguf_path = dspark_beside(&target_path)
            .map_err(|e| format!("{NAME}: {e}"))?
            .ok_or(format!(
                "{NAME}: no dspark-*.gguf beside {}",
                target_path.display()
            ))?;
        let text = std::fs::read_to_string(a.folder.join("config.json"))
            .map_err(|e| format!("{}: {e}", a.folder.join("config.json").display()))?;
        let d = config::read(&text).map_err(|e| format!("{NAME}: config.json: {e}"))?;
        let st_path = a.folder.join("mtp.safetensors");
        let st = Safetensors::open(&st_path).map_err(|e| format!("{}: {e}", st_path.display()))?;
        config::check(&d, &st).map_err(|e| format!("{NAME}: {}: {e}", st_path.display()))?;
        let gsplit =
            Split::open(&gguf_path).map_err(|e| format!("{}: {e}", gguf_path.display()))?;

        let split = open_split(Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{NAME}: the file's header: {e}"))?;
        let gd = config::read_gguf(&gsplit, opener::target_of(&hp)?)
            .map_err(|e| format!("{NAME}: {}: {e}", gguf_path.display()))?;
        println!(
            "{NAME}: target {}, GGUF draft {} (beside it), folder draft {}",
            target_path.display(),
            gguf_path.display(),
            a.folder.display()
        );
        let mut ok = true;
        if tier::run_clause(FILE_CLAUSE, Tag::FileBound)? {
            ok &= file_clause(&d, &gd, &gsplit, &st)?;
        }
        let (folder_draft, gguf_draft) = (
            Opened::read(&a.folder)?,
            Opened::read_gguf(&gguf_path, &hp)?,
        );
        if tier::run_clause(RESERVE_CLAUSE, Tag::FileBound)? {
            ok &= reserve_clause(&folder_draft, &gguf_draft, &hp);
        }
        let agree_on = tier::run_clause(AGREE_CLAUSE, Tag::SelfConsistency)?;
        let tau_on = tier::run_clause(TAU_CLAUSE, Tag::Oracle)?;
        if !agree_on && !tau_on {
            return if ok { Ok(()) } else { Err(checks_failed()) };
        }

        let corpora = CORPORA
            .iter()
            .map(|c| Ok((c, read_ids(&a.prompts.join(c.file))?)))
            .collect::<Result<Vec<_>, GateError>>()?;
        let longest = corpora.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0);
        let ctx = (longest + a.n_gen + 1).next_multiple_of(CTX_STEP);
        let embed_split = Arc::new(open_split(Arch::Inkling2, RECIPE)?);
        let tok = tokenizer::Tokenizer::from_gguf(&target_path)
            .map_err(|e| format!("{NAME}: {}: {e}", target_path.display()))?;
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, split, ctx)?;
        m.set_draft_taps(&d.shape.taps)?;
        let mut folder = opener::load(
            &m,
            m.gpu(),
            opener::anchor_of_file(&embed_split, &hp)?,
            &folder_draft,
            ctx,
        )?;
        let mut gguf = opener::load(
            &m,
            m.gpu(),
            opener::anchor_of_file(&embed_split, &hp)?,
            &gguf_draft,
            ctx,
        )?;
        println!(
            "{NAME}: taps {:?}, {} generated ids a corpus, ctx {ctx}, draft card bytes {} each, \
             target and both drafts open in {:.1} s",
            d.shape.taps,
            a.n_gen,
            gguf.card_bytes(ctx),
            t.elapsed().as_secs_f64()
        );

        let (mut same, mut all) = (0usize, 0usize);
        for (c, prompt) in &corpora {
            let t = std::time::Instant::now();
            let (cont, [pf, pg]) =
                shadow(&mut m, [&mut folder, &mut gguf], prompt, a.n_gen, tok.eog())?;
            same += pf.iter().zip(&pg).filter(|(f, g)| f[0] == g[0]).count();
            all += pg.len();
            let (bf, bg) = (
                schedule(&accepted(&cont, &pf)),
                schedule(&accepted(&cont, &pg)),
            );
            let ag = accepted(&cont, &pg);
            let a_i: Vec<String> = (1..=WIDTH)
                .map(|i| {
                    let n = bg.iter().filter(|&&k| k >= i).count();
                    format!("{:.3}", n as f64 / bg.len().max(1) as f64)
                })
                .collect();
            let (tf, tg) = (tau(&bf), tau(&bg));
            Record::new(&DRAFT_ACCEPTANCE)
                .w("corpus", c.name)
                .u("prompt", prompt.len())
                .u("proposals", pg.len())
                .u("blocks", bg.len())
                .u("width", WIDTH)
                .f("tau", tg)
                .csv("a", a_i)
                .f("tau_every", tau(&ag))
                .f("ref", c.reference)
                .print();
            if tau_on {
                let pass = (tg / c.reference - 1.0).abs() <= BAND;
                println!(
                    "clause {TAU_CLAUSE} {}: tau {tg:.3} (the folder's {tf:.3}) vs {:.2} ±{:.0} % \
                     ({:.1} s) {}",
                    c.name,
                    c.reference,
                    BAND * 100.0,
                    t.elapsed().as_secs_f64(),
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        if agree_on {
            let share = same as f64 / all.max(1) as f64;
            let pass = all > 0 && share >= AGREE;
            println!(
                "clause {AGREE_CLAUSE}: the first id agrees at {same} of {all} proposals ({share:.4}, \
                 at least {AGREE}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        drop(gguf);
        drop(folder);
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
