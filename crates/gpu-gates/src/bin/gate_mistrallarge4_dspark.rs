//! GPU gate for Mistral Large 4's DSpark draft
//! (`bloomery_gpu_mistrallarge4::draft`: the common owner
//! `bloomery_gpu::dspark::DsparkBody` over the file's MLA stages, the
//! target's Q6_K `output` and the recurrent Markov head), on one card,
//! against upstream llama.cpp's own draft run.
//!
//!     gate_mistrallarge4_dspark --draft <gguf>
//!
//! The target is the profile's file (`BLOOMERY_MODEL=mistral-large4`); the
//! draft reads only its `output` and `token_embd`. Both are the files the
//! oracle set's family names (`refset::arch::mistrallarge4::dspark`): the
//! oracle clauses refuse a run on another path by name.
//!
//! The set is the family's (`tools/ref/dump-ml4-dspark.sh`, read by
//! `refset::lcppdspark`): upstream llama.cpp's draft run on each prompt, the
//! rows it was fed and its calls. The family's check refuses by name a set
//! dumped from another target or draft file, by another llama.cpp tree or
//! patch, for another architecture, or without its trailer.
//!
//! Clauses:
//!
//! - `load_reads_every_tensor` (file-bound): the draft's resident weight
//!   buffers are the statement's tensors (`tensors`) in load order, each of
//!   its file bytes (a Q8_0 tensor as its quants and its scales), with the
//!   target's `output` between the head's norm and the Markov tables, and no
//!   other; every stage tensor's buffers hold the file's values; the block's
//!   mask rows are `mask_embd`'s rows. A tensor left unread, read twice or
//!   read under another name is red.
//! - `draft_matches_reference` (oracle): each prompt's first call — the
//!   draft fed the prompt's rows `0..pos0` and run after `id_last` at
//!   `pos0`, the state upstream's cache holds with nothing verified yet —
//!   proposes upstream's ids. The first id must agree on all but
//!   [`MAX_FIRST_MISSES`] prompts; the ids after a row's first disagreement
//!   are conditioned on it (the recurrent head reads the previous id) and
//!   are printed, not judged.
//! - `acceptance` (a reading, no band): for each first call followed by
//!   another, the target's tokens after `pos0` are upstream's accepted ids
//!   and the next call's `id_last`; ours and upstream's accepted lengths
//!   against them are printed with their sums.

#[cfg(feature = "mistrallarge4")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mistrallarge4_dspark", gate::run())
}

#[cfg(feature = "mistrallarge4")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bloomery_gpu::Gpu;
    use bloomery_gpu::dspark::dense::{gguf_f32, gguf_tensor};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, ref_model_path};
    use bloomery_gpu_mistrallarge4::draft::{self, Loaded, Ml4Draft};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block, dequant_row};
    use model::arch::Arch;
    use model::arch::mistrallarge4::dspark::{Hparams, Tensor, Ty, names, tensors};
    use model::arch::names::output;
    use refset::arch::mistrallarge4::MODEL;
    use refset::arch::mistrallarge4::dspark::{CALLS, DRAFT, SET};
    use refset::lcppdspark::{DsparkSet, Prompt};

    const NAME: &str = "gate_mistrallarge4_dspark";
    const RECIPE: &str = "gate-gpu-mistrallarge4-dspark";

    /// Ids one call proposes: the file's block, every row drafting.
    const WIDTH: usize = 3;

    /// PIN(2026-11-02): provisional, [derived] — upstream's matrix products
    /// round the activation to q8_1 and ours keep it f32, so a first id
    /// whose top two logits sit inside that rounding may flip; 2 of 32 is
    /// the allowance until the first run's table re-pins it.
    const MAX_FIRST_MISSES: usize = 2;

    struct Args {
        draft: PathBuf,
    }

    fn args() -> Result<Args, GateError> {
        let mut draft = None;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--draft" => draft = it.next().map(PathBuf::from),
                other => return Err(format!("{NAME}: unknown argument {other}").into()),
            }
        }
        Ok(Args {
            draft: draft.ok_or("--draft <gguf> is required")?,
        })
    }

    /// The oracle set through its family, once this run's target and draft are the files the family names: the set states
    /// the files it was dumped from, and a run on others would compare the draft against another model's calls.
    fn open_set(target: &Path, draft: &Path) -> Result<DsparkSet, GateError> {
        for (what, run, family) in [("target", target, MODEL), ("draft", draft, DRAFT)] {
            if run != Path::new(family) {
                return Err(format!(
                    "{NAME}: this run's {what} is {}, the oracle set's family names {family}",
                    run.display()
                )
                .into());
            }
        }
        let set = DsparkSet::open(&CALLS.path(SET), &CALLS)?;
        if set.n_max != Some(WIDTH) {
            return Err(format!(
                "{}: n_max {:?}, the file's block {WIDTH}",
                set.dir.display(),
                set.n_max
            )
            .into());
        }
        println!(
            "set {} build {}",
            set.dir.display(),
            set.build.as_deref().unwrap_or("-")
        );
        Ok(set)
    }

    /// Ours for `p`'s first call: the draft reset, fed the prompt's rows
    /// `0..pos0` in order, then run after `id_last` at `pos0`.
    fn first_call(
        gpu: &Gpu,
        body: &mut Ml4Draft,
        p: &Prompt,
        hp: &Hparams,
    ) -> Result<[u32; WIDTH], GateError> {
        let call = &p.calls[0];
        let n = call.pos0 as usize;
        if n == 0 || p.pos.len() < n || p.pos[..n].iter().zip(0u32..).any(|(&a, b)| a != b) {
            return Err(format!(
                "{}: the first call at {n} is not preceded by the rows 0..{n} in order",
                p.name
            )
            .into());
        }
        let feat = hp.fc_in();
        body.reset(gpu)?;
        let mut first = 0;
        while first < n {
            let k = (n - first).min(hp.window);
            body.append_as(
                gpu,
                u32::try_from(first)?,
                &p.taps[first * feat..(first + k) * feat],
            )?;
            first += k;
        }
        let mut out = [0u32; WIDTH];
        body.propose_as(gpu, call.pos0, call.last, WIDTH, &mut out, &mut [])?;
        Ok(out)
    }

    /// The length of the run `a` and `b` share from their first entries.
    fn prefix(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// The device bytes the statement says a tensor loads into: a Q8_0
    /// tensor's quants (a byte a value) and scales (two bytes a 32), an f32
    /// tensor's values.
    fn buffers(t: &Tensor) -> Vec<u64> {
        let n: u64 = t.dims.iter().product();
        match t.ty {
            Ty::F32 => vec![t.bytes()],
            Ty::Q8_0 => vec![n, n / 16],
        }
    }

    fn load_reads_every_tensor(
        gpu: &Gpu,
        body: &Ml4Draft,
        file: &Split,
        target: &Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        let stated = tensors(hp);
        let by = |name: &str| -> Result<&Tensor, GateError> {
            stated
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| format!("{name} is not in the statement").into())
        };
        // The stages, then fc and its norm, the head's norm, the target's
        // projection, the Markov tables and the joint projection.
        let mut want: Vec<u64> = Vec::new();
        for t in stated.iter().filter(|t| t.name.starts_with("blk.")) {
            want.extend(buffers(t));
        }
        for n in [names::fc(), names::enc_output_norm(), names::output_norm()] {
            want.extend(buffers(by(&n)?));
        }
        let (_, out) = target
            .find(&output())
            .ok_or_else(|| format!("the target has no {}", output()))?;
        want.push(out.nbytes);
        for n in [
            names::markov_w1(),
            names::markov_w2(),
            names::joint_proj(),
            names::joint_proj_bias(),
        ] {
            want.extend(buffers(by(&n)?));
        }
        let held = draft::held(body);
        let mut ok = held.weights == want;
        if !ok {
            println!(
                "load: {} weight buffers of {} bytes; the statement's {} of {}",
                held.weights.len(),
                held.weights.iter().sum::<u64>(),
                want.len(),
                want.iter().sum::<u64>()
            );
            for (i, (h, w)) in held.weights.iter().zip(&want).enumerate() {
                if h != w {
                    println!("load: buffer {i} holds {h} bytes, the statement's {w}");
                }
            }
        }
        let s = gpu.stream();
        for (name, loaded) in body.layers().loaded() {
            let t = by(&name)?;
            let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
            let same = match loaded {
                Loaded::F32(b) => b.to_host_vec(s)? == gguf_f32(file, &name, &dims)?,
                Loaded::Q8(q) => {
                    let blocks: Vec<Q8Block> = gguf_tensor(file, &name, GgmlType::Q8_0, &dims)?
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .map(Q8Block::from_bytes)
                        .collect();
                    let (qs, d) = q8_0_planes(&blocks);
                    q.qs.buf().to_host_vec(s)? == qs && q.d.buf().to_host_vec(s)? == d
                }
            };
            if !same {
                println!("load: {name} on the card is not the file's");
            }
            ok &= same;
        }
        let e = hp.n_embd;
        let mask = names::mask_embd();
        let bytes = gguf_tensor(file, &mask, GgmlType::Q8_0, &[e, hp.block_size - 1])?;
        let mut rows = vec![0.0f32; (hp.block_size - 1) * e];
        let row_bytes = e / 32 * 34;
        for (k, row) in rows.chunks_exact_mut(e).enumerate() {
            dequant_row(
                GgmlType::Q8_0,
                &bytes[k * row_bytes..(k + 1) * row_bytes],
                row,
            )
            .map_err(|e| format!("{mask} row {k}: {e}"))?;
        }
        let same = body.layers().mask_rows(gpu)? == rows;
        if !same {
            println!("load: the block's mask rows are not {mask}'s");
        }
        Ok(ok && same)
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        let target_path = ref_model_path()?;
        let target = Arc::new(open_split(Arch::MistralLarge4, RECIPE)?);
        let file = Split::open(&a.draft).map_err(|e| format!("open {}: {e}", a.draft.display()))?;
        let hp = Hparams::read(&file).map_err(|e| format!("the draft's header: {e}"))?;
        let gpu = Gpu::new()?;
        let t = std::time::Instant::now();
        let mut body = draft::open(&gpu, &file, Arc::clone(&target))?;
        println!(
            "load {} weight bytes, {} ring bytes in {:.1} s (runtime value)",
            draft::held(&body).weight_bytes(),
            draft::held(&body).rings.iter().sum::<u64>(),
            t.elapsed().as_secs_f64()
        );
        let mut ok = true;
        if tier::run_clause("load_reads_every_tensor", Tag::FileBound)? {
            let pass = load_reads_every_tensor(&gpu, &body, &file, &target, &hp)?;
            if !pass {
                println!("clause load_reads_every_tensor FAIL");
            }
            ok &= pass;
        }
        let judged = tier::run_clause("draft_matches_reference", Tag::Oracle)?;
        let read = tier::run_clause("acceptance", Tag::Oracle)?;
        if judged || read {
            let set = open_set(&target_path, &a.draft)?;
            let n = set.prompts.len();
            let feat = hp.fc_in();
            let (mut misses, mut ours_acc, mut up_acc, mut known) = (0, 0, 0, 0);
            for row in &set.prompts {
                let p = set.prompt(row, feat)?;
                let ours = first_call(&gpu, &mut body, &p, &hp)?;
                let call = &p.calls[0];
                let theirs = &call.ids[..call.ids.len().min(WIDTH)];
                let agree = prefix(&ours, theirs);
                if agree == 0 {
                    misses += 1;
                }
                let mut line = format!(
                    "{} pos0={} last={} ours={ours:?} upstream={theirs:?} agree={agree}",
                    p.name, call.pos0, call.last
                );
                if let Some(next) = p.calls.get(1) {
                    // The target's tokens after pos0: upstream's accepted ids,
                    // then the id the target chose where it stopped.
                    let k = next.pos0.checked_sub(call.pos0 + 1).ok_or_else(|| {
                        format!(
                            "{}: a second call at {} before the first's",
                            p.name, next.pos0
                        )
                    })? as usize;
                    let mut after: Vec<u32> = theirs[..k.min(theirs.len())].to_vec();
                    after.push(next.last);
                    let (o, u) = (prefix(&ours, &after), k.min(theirs.len()));
                    line += &format!(" accepted ours={o} upstream={u}");
                    ours_acc += o;
                    up_acc += u;
                    known += 1;
                }
                println!("{line}");
            }
            if read {
                println!(
                    "acceptance (reading, no band): ours {ours_acc} upstream {up_acc} ids over {known} \
                     calls of {WIDTH}"
                );
            }
            if judged {
                let pass = misses <= MAX_FIRST_MISSES;
                println!(
                    "draft_matches_reference: first ids differ on {misses} of {n} prompts (at most \
                     {MAX_FIRST_MISSES})"
                );
                if !pass {
                    println!("clause draft_matches_reference FAIL");
                }
                ok &= pass;
            }
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the draft holds every tensor of its file and the target's output once, \
             as the file's values; its first proposals are upstream's"
        );
        Ok(())
    }
}
