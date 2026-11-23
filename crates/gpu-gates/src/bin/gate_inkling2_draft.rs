//! GPU gate for Inkling 2's DSpark draft on the card (`bloomery_gpu::dspark::DsparkBody` over
//! `ctx_dense::CtxDenseLayers`, the fused context and the anchor form of the recurrent head with
//! its correction folded through the target's output matrix), beside the Mini's GGUF target on one
//! card, in the target's shadow.
//!
//!     gate_inkling2_draft --draft <folder> --prompts <dir> [--gen <ids>]
//!
//! The target is the profile's file (`BLOOMERY_MODEL=inkling2`); the draft is the official
//! folder's `config.json` and `mtp.safetensors`; `<dir>` holds each corpus's prompt as the
//! reference tokenizer's ids, one a line: `chat.ids`, `code4096.ids`, `prose4096.ids`.
//!
//! The shadow: the target steps one position at a time, greedy, the prompt then up to `--gen` ids
//! of its own ([`GEN`] unless given), ending early at an end-of-generation id as the reference's run
//! does; after each position the body's taps (the rows entering the visits the draft's config names
//! and the final normed row) are appended to the draft's context, and before each generated
//! position but the end the draft proposes [`WIDTH`] ids after the target's id there. A proposal
//! accepts the prefix it shares with the target's continuation; the acceptance of a copy-heavy
//! continuation grows along it, so the window is the reference's.
//!
//! Clause `acceptance_in_band` (oracle, real tier): per corpus, a speculative schedule over the
//! continuation — a block at the prompt's end, the next at the first id after a block's accepted
//! ones — has τ (one plus the mean ids a block accepts) within [`BAND`] of the reference's τ on the
//! same prompts at the same width. The record also carries the share of blocks accepting at least
//! i ids (a_1..a_7) and τ over a proposal at every generated id, as readings.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_draft", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::dspark::ctx_dense::{AnchorEmbed, CtxDenseLayers, CtxSpec};
    use bloomery_gpu::dspark::dense::{Bf16Source, gguf_f32};
    use bloomery_gpu::dspark::fold::fold;
    use bloomery_gpu::dspark::fuse::mix_weights;
    use bloomery_gpu::dspark::head::{HeadTail, HeadWeights, RnnTail};
    use bloomery_gpu::dspark::{
        DsparkBody, EmbedSource, FuseWeights, HeadSource, InPlace, Q8Rows, TargetHead,
    };
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::record::DRAFT_ACCEPTANCE;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use bloomery_record::Record;
    use cuda_core::DeviceBuffer;
    use hf::dspark::inkling2::{self as config, Draft};
    use model::arch::Arch;
    use model::arch::inkling2::hparams::Hparams;
    use model::arch::names;
    use model::embed::EmbedTable;
    use safetensors::Safetensors;

    const NAME: &str = "gate_inkling2_draft";
    const RECIPE: &str = "gate-gpu-inkling2-draft";
    const CLAUSE: &str = "acceptance_in_band";

    /// Ids a block proposes: the reference's width.
    const WIDTH: usize = 7;
    /// Ids the target generates after each prompt at most: the reference run's.
    const GEN: usize = 1024;
    /// The positions the target and the draft's global layer hold are the longest prompt and the
    /// generated ids, rounded up to this.
    const CTX_STEP: usize = 512;
    /// The relative distance from the reference's τ the clause allows.
    const BAND: f64 = 0.15;

    /// A corpus: its name, its ids file and the reference's τ on it.
    struct Corpus {
        name: &'static str,
        file: &'static str,
        reference: f64,
    }

    /// PIN(2026-11-23): llama.cpp with the draft's upstream PR over a Q8_0-dense Mini, greedy, the
    /// draft proposing [`WIDTH`] ids a block, on these prompts, up to [`GEN`] ids or the end of
    /// generation (`#mean acc len`).
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
        draft: PathBuf,
        prompts: PathBuf,
        n_gen: usize,
    }

    fn args() -> Result<Args, GateError> {
        let (mut draft, mut prompts, mut n_gen) = (None, None, GEN);
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--draft" => draft = it.next().map(PathBuf::from),
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
            draft: draft.ok_or(format!("{NAME}: --draft <folder> is required"))?,
            prompts: prompts.ok_or(format!("{NAME}: --prompts <dir> is required"))?,
            n_gen,
        })
    }

    /// The draft folder's `mtp.safetensors` as the layers' weight source.
    struct Mtp {
        st: Safetensors,
    }

    impl Bf16Source for Mtp {
        fn bf16(&self, name: &str) -> Result<(&[u16], &[usize]), GpuError> {
            self.st.bf16(name).map_err(|e| GpuError::Shape {
                what: NAME,
                detail: e.to_string(),
            })
        }
    }

    impl Mtp {
        /// Tensor `name` as f32, refused by name unless its shape is `shape`.
        fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, GateError> {
            let t = self.st.tensor(name).map_err(|e| format!("{NAME}: {e}"))?;
            if t.shape != shape {
                return Err(
                    format!("{NAME}: {name} is {:?}, the draft reads {shape:?}", t.shape).into(),
                );
            }
            Ok(t.data)
        }

        /// Tensor `name`'s bf16 bits, refused by name unless its shape is `shape`.
        fn bits(&self, name: &str, shape: &[usize]) -> Result<&[u16], GateError> {
            let (bits, got) = self.bf16(name)?;
            if got != shape {
                return Err(format!("{NAME}: {name} is {got:?}, the draft reads {shape:?}").into());
            }
            Ok(bits)
        }
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

    /// The draft beside `m`: its layers, the fused context and the head, which reads the target's
    /// resident Q8_0 output matrix in place over the scored ids and folds the correction through it.
    fn load_draft(
        m: &GpuModel<Body>,
        split: Arc<gguf::Split>,
        hp: &Hparams,
        d: &Draft,
        mtp: &Mtp,
        ctx: usize,
    ) -> Result<DsparkBody<CtxDenseLayers>, GateError> {
        let gpu = m.gpu();
        let s = gpu.stream();
        let b = &d.backbone;
        let (hidden, rank) = (b.hidden, d.rank);
        let window = b
            .window
            .ok_or(format!("{NAME}: the draft states no window"))?;
        let anchor = AnchorEmbed {
            table: EmbedTable::new(Arc::clone(&split), &names::token_embd(), hp.n_embd)
                .map_err(|e| format!("{NAME}: token_embd: {e}"))?,
            gain: gguf_f32(&split, "token_embd_norm.weight", &[hp.n_embd])?,
            eps: hp.rms_eps,
        };
        let spec = CtxSpec {
            layers: b.layers,
            eps: b.eps,
            rope_theta: b.rope_theta as f32,
            window,
            windowed: b.windowed.clone(),
            ctx,
        };
        let layers = CtxDenseLayers::load(gpu, mtp, spec, anchor)?;

        let taps = d.shape.taps.len();
        let logits = mtp.f32s("fuse_weight", &[b.layers + 1, taps])?;
        let mut gains = Vec::with_capacity((b.layers + 1) * hidden);
        for l in 0..b.layers {
            gains.extend(mtp.f32s(&format!("layers.{l}.dflare_norm.weight"), &[hidden])?);
        }
        gains.extend(mtp.f32s("markov_head.dflare_norm.weight", &[hidden])?);
        let fuse = FuseWeights {
            mix: DeviceBuffer::from_host(s, &mix_weights(&logits, taps)?)?,
            gains: DeviceBuffer::from_host(s, &gains)?,
            eps: b.eps,
        };

        let Some(DevWeight::Q8_0 { qs, d: scales, .. }) = m.weights().get(&names::output()) else {
            return Err(format!(
                "{NAME}: the target's {} is not a resident Q8_0 matrix",
                names::output()
            )
            .into());
        };
        let n_head = hp.n_vocab_unpadded;
        // SAFETY: `m` owns the output matrix and outlives the draft (the caller drops the draft
        // first), and the draft runs on `m`'s own Gpu, so the windows' addresses stay allocated
        // and in place for every launch and captured graph.
        let output = unsafe {
            TargetHead::Q8_0InPlace {
                qs: InPlace::rows(qs, n_head, gpu.context())?,
                d: InPlace::rows(scales, n_head, gpu.context())?,
            }
        };
        let hidden_out = mtp.f32s("markov_head.hidden_out.weight", &[hidden, rank])?;
        let hidden_b = mtp.f32s("markov_head.hidden_out.bias", &[hidden])?;
        let folded = fold(gpu, &output, &hidden_out, &hidden_b, hidden, rank)?;
        let joint_in = 2 * rank + 2 * hidden;
        let w1 = mtp.bits(
            "markov_head.markov_w1.weight",
            &[d.target.padded_vocab, rank],
        )?;
        let joint = mtp.bits("markov_head.rnn_proj.weight", &[3 * rank, joint_in])?;
        let norm = mtp.f32s("norm.weight", &[hidden])?;
        let head = HeadWeights {
            norm: DeviceTensor::upload(s, &norm, 1, hidden)?,
            eps: b.eps,
            source: HeadSource::Target(output),
            tail: HeadTail::MarkovRnn(RnnTail {
                w1: Q8Rows::from_bf16(gpu, w1, d.target.padded_vocab, rank, &|r| r)?,
                w2: folded.w2,
                joint: Q8Rows::from_bf16(gpu, joint, 3 * rank, joint_in, &|r| r)?,
                joint_b: DeviceBuffer::from_host(
                    s,
                    &mtp.f32s("markov_head.rnn_proj.bias", &[3 * rank])?,
                )?,
                softcap: Some(d.target.softcap),
                row: RnnTail::ANCHOR,
                c: Some(folded.c),
            }),
        };
        Ok(DsparkBody::fused(
            gpu,
            &d.shape,
            layers,
            fuse,
            head,
            EmbedSource::Target,
        )?)
    }

    /// The target's continuation of `prompt`, `n_gen` ids or up to the first of `eog`, and for each
    /// generated position before its end the ids the draft accepted there, against the ids the
    /// continuation holds after it.
    fn shadow(
        m: &mut GpuModel<Body>,
        draft: &mut DsparkBody<CtxDenseLayers>,
        prompt: &[u32],
        n_gen: usize,
        eog: &[u32],
    ) -> Result<Vec<usize>, GateError> {
        m.reset()?;
        draft.reset(m.gpu())?;
        let mut next = 0;
        for (p, &t) in prompt.iter().enumerate() {
            next = m.step(&[t])?;
            let taps = m.draft_taps()?;
            draft.append_as(m.gpu(), u32::try_from(p)?, &taps)?;
        }
        let mut cont = Vec::with_capacity(n_gen + 1);
        let mut proposals = Vec::with_capacity(n_gen);
        for g in 0..n_gen {
            let pos = u32::try_from(prompt.len() + g)?;
            cont.push(next);
            if eog.contains(&next) {
                break;
            }
            let mut out = [0u32; WIDTH];
            draft.propose_as(m.gpu(), pos, next, WIDTH, &mut out, &mut [])?;
            proposals.push(out);
            next = m.step(&[next])?;
            let taps = m.draft_taps()?;
            draft.append_as(m.gpu(), pos, &taps)?;
        }
        if proposals.len() == n_gen {
            cont.push(next);
        }
        Ok(proposals
            .iter()
            .enumerate()
            .map(|(g, ids)| {
                ids.iter()
                    .zip(&cont[g + 1..])
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .collect())
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

    pub fn run() -> Result<(), GateError> {
        if !tier::run_clause(CLAUSE, Tag::Oracle)? {
            return Ok(());
        }
        let a = args()?;
        let text = std::fs::read_to_string(a.draft.join("config.json"))
            .map_err(|e| format!("{}: {e}", a.draft.join("config.json").display()))?;
        let d = config::read(&text).map_err(|e| format!("{NAME}: config.json: {e}"))?;
        let path = a.draft.join("mtp.safetensors");
        let mtp = Mtp {
            st: Safetensors::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        };
        let corpora = CORPORA
            .iter()
            .map(|c| Ok((c, read_ids(&a.prompts.join(c.file))?)))
            .collect::<Result<Vec<_>, GateError>>()?;
        let longest = corpora.iter().map(|(_, ids)| ids.len()).max().unwrap_or(0);
        let ctx = (longest + a.n_gen + 1).next_multiple_of(CTX_STEP);

        let split = open_split(Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{NAME}: the file's header: {e}"))?;
        let embed_split = Arc::new(open_split(Arch::Inkling2, RECIPE)?);
        let model = bloomery_gpu_gates::ref_model_path()?;
        let tok = tokenizer::Tokenizer::from_gguf(&model)
            .map_err(|e| format!("{NAME}: {}: {e}", model.display()))?;
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, split, ctx)?;
        m.set_draft_taps(&d.shape.taps)?;
        let mut draft = load_draft(&m, embed_split, &hp, &d, &mtp, ctx)?;
        println!(
            "{NAME}: taps {:?}, {} generated ids a corpus, ctx {ctx}, draft card bytes {}, target \
             and draft open in {:.1} s",
            d.shape.taps,
            a.n_gen,
            draft.card_bytes(ctx),
            t.elapsed().as_secs_f64()
        );
        for w in [3, WIDTH] {
            let m = d.shape.rows(w);
            println!(
                "{NAME}: width {w}: block graph {} nodes ({m} rows, the head's {}), append graph {} nodes",
                draft.graph_block(m).map_or(0, |g| g.node_count()),
                draft.head_launches(m),
                draft.graph_append(1).map_or(0, |g| g.node_count())
            );
        }

        let mut ok = true;
        for (c, prompt) in &corpora {
            let t = std::time::Instant::now();
            let accepted = shadow(&mut m, &mut draft, prompt, a.n_gen, tok.eog())?;
            let blocks = schedule(&accepted);
            let a_i: Vec<String> = (1..=WIDTH)
                .map(|i| {
                    let n = blocks.iter().filter(|&&k| k >= i).count();
                    format!("{:.3}", n as f64 / blocks.len() as f64)
                })
                .collect();
            let tau_blocks = tau(&blocks);
            Record::new(&DRAFT_ACCEPTANCE)
                .w("corpus", c.name)
                .u("prompt", prompt.len())
                .u("proposals", accepted.len())
                .u("blocks", blocks.len())
                .u("width", WIDTH)
                .f("tau", tau_blocks)
                .csv("a", a_i)
                .f("tau_every", tau(&accepted))
                .f("ref", c.reference)
                .print();
            let pass = (tau_blocks / c.reference - 1.0).abs() <= BAND;
            println!(
                "clause {CLAUSE} {}: tau {tau_blocks:.3} vs {:.2} ±{:.0} % ({:.1} s) {}",
                c.name,
                c.reference,
                BAND * 100.0,
                t.elapsed().as_secs_f64(),
                verdict(pass)
            );
            ok &= pass;
        }
        drop(draft);
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
