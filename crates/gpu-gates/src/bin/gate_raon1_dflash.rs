//! The raon1 DFlash2 draft gate: Raon-1-mini's draft file, found beside the target as the serve
//! seat finds it (`raon1_dflash::pick`, refset's `raon1::DFLASH_BESIDE` row), through the common DSpark owner over DFlash2's layers with
//! the selector head, on one card, against the vendor's taps (refset family `lcpp-raon1`, `just
//! dump-ref-raon1`) and the first block the vendor's speculative run drafts.
//!
//! - `draft_loads`: the row's file beside the target opens, and the owner runs the file's
//!   geometry ([`want_shape`]): a block of 8 rows whose row 0 is the anchor, the rows entering
//!   layers 2, 7, 12, 17 and 22 tapped (each the four streams of 2,560 values), the window of
//!   2,048, the mask id 259,839, a selector of rank 256 keeping 16 ids a row, the target's whole
//!   vocabulary. The draft holds a copy of the target's output head (no target is loaded here).
//!   The load holds the plan's formula term by term (`raon1_dflash::PlanBytes::hold`, a term off
//!   ends the gate by name), and its card bytes at the window are within the formula's total for
//!   a copied head (`raon1_dflash::plan_bytes`).
//! - `refuses_v41_dspark`: V4.1's DSpark draft (`--v41-draft`, a `dflash` file with a Markov head)
//!   through the same open is refused by name at its header, the error naming the selector's
//!   `selector_hidden.weight`.
//! - `first_block_is_oracles`: set a's prompt rows as the vendor tapped them
//!   (`l_out-{1,6,11,16,21}`: the streams leaving layer `L − 1` enter layer `L`), appended from
//!   position 0 a group at a time, then the vendor's first block (`BLOCK_1`): the prompt's last
//!   token as the anchor over the rows of the tokens before it. Its 7 ids are the vendor's, id for
//!   id, from the captured block and from its eager twin, and the two leave the same bits in the
//!   block's rows. A proposal of [`NARROW`] ids runs the same whole block (the selector's rows
//!   attend to one another) and is its first [`NARROW`] ids.
//! - `second_block_is_oracles`: the prompt's last row appended (a one-row append), then the vendor's
//!   second block (`BLOCK_2`): the target's first greedy id as the anchor over every prompt row,
//!   which is what the vendor's run drafts once block 1's first id is not the target's. Its 7 ids
//!   are the vendor's, captured and eager, the two the same bits.
//!
//!     gate_raon1_dflash --v41-draft <V4.1 DSpark file>
//!
//! A set absent from the box, dumped from another file or build, or a target file other than the
//! set's ends the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raon1_dflash", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the serve seat and two gates share the opener; each takes the parts it reads"
)]
#[path = "shared/raon1_dflash.rs"]
mod raon1_dflash;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::raon1::STREAMS;
    use bloomery_gpu::dspark::{DraftLayers, DsparkBody, Submit};
    use bloomery_gpu_gates::{
        GateError, RefManifest, bits_equal, checks_failed, data_dir, open_split, ref_model_path,
        ref_tensor_logical_in, verdict,
    };
    use model::arch::Arch;
    use model::arch::dflash2::{Hparams, names};
    use refset::arch::draft::DraftFrom;
    use refset::arch::prompt_tables;
    use refset::arch::raon1::{A, DFLASH_BESIDE, LCPP, MODEL};
    use runtime::dspark::{DraftVocab, DsparkShape, HeadKind};

    use crate::raon1_dflash::{self, HeadFrom, Layers, Pick};

    const RECIPE: &str = "gate-gpu-raon1-dflash";

    /// The target's hidden width: one stream's row.
    const N_EMBD: usize = 2560;

    /// A block of the vendor's speculative run on set a: its log's line `draft ids pos {pos} after
    /// {after}: {ids}` (`tools/ref/raon1-dflash-ref.sh`, prompt `a`). The host oracle (numpy f32
    /// over the same taps and the two files) reproduces each one id for id. The run evaluates the
    /// target on the prompt but its last token and drafts after that token; each block's context
    /// is the `pos` rows before its anchor.
    struct VendorBlock {
        /// The position of the anchor, which is also the rows of context before it.
        pos: u32,
        /// The token the block runs after.
        after: u32,
        ids: [u32; WIDTH],
    }

    /// The vendor's first block: after the prompt's last token, over the rows before it.
    const BLOCK_1: VendorBlock = VendorBlock {
        pos: 25,
        after: 44,
        ids: [166_387, 4832, 23_388, 166_436, 10_189, 46, 1794],
    };

    /// The vendor's second block: block 1's first id is not the target's, so the run goes on after
    /// the target's own token (the set's first greedy id) over every prompt row.
    const BLOCK_2: VendorBlock = VendorBlock {
        pos: 26,
        after: 883,
        ids: [15_073, 16_647, 166_387, 35_997, 46, 1794, 970],
    };

    /// The context the draft loads for: past the window, so the ring is the size a seat's is.
    const CTX: usize = 4096;

    /// The ids a block proposes: the block's rows after the anchor's.
    const WIDTH: usize = 7;

    /// A proposal narrower than the block, as the width chooser asks.
    const NARROW: usize = 3;

    /// The rows one append graph takes.
    const GROUP: usize = <Layers as DraftLayers>::APPEND_GROUP;

    /// The geometry the owner runs Raon-1-mini's draft by, for a target of `n_vocab` ids.
    fn want_shape(n_vocab: usize) -> DsparkShape {
        DsparkShape {
            block: 8,
            row0_drafts: false,
            taps: vec![2, 7, 12, 17, 22],
            hidden: STREAMS * N_EMBD,
            vocab: DraftVocab::Full { n: n_vocab },
            mask_id: 259_839,
            window: Some(2048),
            head: HeadKind::Selector {
                rank: 256,
                top_k: 16,
            },
        }
    }

    /// `--v41-draft`.
    fn parse_args() -> Result<PathBuf, GateError> {
        const USAGE: &str = "usage: gate_raon1_dflash --v41-draft <V4.1 DSpark file>";
        let mut v41 = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--v41-draft" => {
                    let p = it
                        .next()
                        .ok_or_else(|| format!("--v41-draft needs a file: {USAGE}"))?;
                    v41 = Some(PathBuf::from(p));
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        v41.ok_or_else(|| format!("no --v41-draft: {USAGE}").into())
    }

    /// Set a's prompt rows as the draft reads them, token-major: per position the taps in the
    /// shape's order, each the four streams entering that layer, stream-major. Every tap is held
    /// to `[N_EMBD, STREAMS, positions]`.
    fn set_a_rows(
        man: &RefManifest,
        taps: &[usize],
        positions: usize,
    ) -> Result<Vec<f32>, GateError> {
        let tap = STREAMS * N_EMBD;
        let mut layers = Vec::with_capacity(taps.len());
        for &l in taps {
            let name = format!(
                "l_out-{}",
                l.checked_sub(1).ok_or_else(|| format!(
                    "a tap of layer {l}: the rows entering layer 0 are the embedding's"
                ))?
            );
            let row = man.tensor(&name, 0)?;
            let want = [N_EMBD as u64, STREAMS as u64, positions as u64, 1];
            if row.ne != want {
                return Err(
                    format!("{A}: {name} is {:?}, the draft reads {want:?}", row.ne).into(),
                );
            }
            layers.push(ref_tensor_logical_in(&man.dir, row)?);
        }
        let mut rows = Vec::with_capacity(positions * taps.len() * tap);
        for t in 0..positions {
            for v in &layers {
                rows.extend_from_slice(&v[t * tap..(t + 1) * tap]);
            }
        }
        Ok(rows)
    }

    /// Clause `draft_loads`: the pick and the owner's geometry.
    fn draft_loads(target_path: &Path) -> Result<(bool, PathBuf), GateError> {
        let (path, from) = match raon1_dflash::pick(None, target_path)? {
            Pick::File(path, from) => (path, from),
            Pick::Absent(dirs) => {
                return Err(format!(
                    "{RECIPE}: no {} beside {} (looked at {dirs:?})",
                    DFLASH_BESIDE.candidates()[0].name(),
                    target_path.display()
                )
                .into());
            }
        };
        let beside = from == DraftFrom::Beside;
        println!(
            "draft_loads pick={} from={} {}",
            path.display(),
            from.describe(),
            verdict(beside)
        );
        Ok((beside, path))
    }

    /// Clause `refuses_v41_dspark`.
    fn refuses_v41(gpu: &Gpu, v41: &Path, target: &Arc<gguf::Split>) -> Result<bool, GateError> {
        let selector = names::selector_hidden();
        let from = (Arc::clone(target), HeadFrom::Copy);
        let (refused, why) = match raon1_dflash::open_body(gpu, v41, from, CTX) {
            Ok(_) => (false, "opened".to_owned()),
            Err(e) => {
                let e = e.to_string();
                (e.contains(&selector), e)
            }
        };
        println!(
            "refuses_v41_dspark file={} error={why:?} names {selector} {}",
            v41.display(),
            verdict(refused)
        );
        Ok(refused)
    }

    pub fn run() -> Result<(), GateError> {
        let v41 = parse_args()?;
        let target_path = ref_model_path()?;
        let same = std::fs::canonicalize(&target_path)
            .and_then(|t| std::fs::canonicalize(MODEL).map(|m| t == m))
            .map_err(|e| format!("resolve {} and {MODEL}: {e}", target_path.display()))?;
        if !same {
            return Err(format!(
                "{}: the sets' taps are {MODEL}'s; run through `just {RECIPE}`",
                target_path.display()
            )
            .into());
        }
        let man = RefManifest::open(&LCPP.path_in(&data_dir(), A), &LCPP)?;
        let prompt = prompt_tables::tokens(&man)?.to_vec();
        let greedy = prompt_tables::greedy(&man)?;
        let first_greedy = *greedy
            .ids
            .first()
            .ok_or_else(|| format!("{A}: an empty greedy table"))?;
        held_to_tables(&prompt, first_greedy)?;
        let target = Arc::new(open_split(Arch::Raon1, RECIPE)?);
        let n_vocab = n_vocab_of(&target)?;
        let want = want_shape(n_vocab);
        let rows = set_a_rows(&man, &want.taps, prompt.len())?;

        let (picked, path) = draft_loads(&target_path)?;
        let gpu = Gpu::new()?;
        let t = std::time::Instant::now();
        let from = (Arc::clone(&target), HeadFrom::Copy);
        let (shape, mut body) = raon1_dflash::open_body(&gpu, &path, from, CTX)?;
        let shaped = shape == want;
        let file = gguf::Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = Hparams::read(&file).map_err(|e| format!("{}: {e}", path.display()))?;
        let reserve = raon1_dflash::plan_bytes(&file, &hp, &target, (hp.window, true))?.total();
        let held = body.card_bytes(hp.window);
        let fits = held <= reserve;
        println!(
            "draft_loads shape={shape:?} card_bytes={held} at the window, formula_total={reserve}, in \
             {:.1} s (runtime value) {}",
            t.elapsed().as_secs_f64(),
            verdict(shaped && fits)
        );
        if !shaped {
            println!("draft_loads want={want:?}");
        }
        let mut ok = picked && shaped && fits;

        ok &= refuses_v41(&gpu, &v41, &target)?;

        let tap = want.taps.len() * want.hidden;
        let m = want.block;
        let (first_rows, last_row) = rows.split_at(usize::try_from(BLOCK_1.pos)? * tap);
        append_rows(&gpu, &mut body, first_rows, (tap, BLOCK_1.pos))?;
        let first = block_is_oracle(&gpu, &mut body, "first_block_is_oracles", (&BLOCK_1, m))?;
        // A narrower proposal runs the same whole block: its ids are the block's first ones.
        let mut short = [0u32; NARROW];
        body.propose_as(
            &gpu,
            BLOCK_1.pos,
            BLOCK_1.after,
            NARROW,
            &mut short,
            &mut [],
        )?;
        let narrow = short == BLOCK_1.ids[..NARROW];
        println!(
            "first_block_is_oracles width={NARROW} ids={short:?} = the block's first {NARROW} {}",
            verdict(narrow)
        );
        append_rows(&gpu, &mut body, last_row, (tap, BLOCK_2.pos))?;
        let second = block_is_oracle(&gpu, &mut body, "second_block_is_oracles", (&BLOCK_2, m))?;
        ok &= first && narrow && second;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_raon1_dflash the draft beside the target loads with its file's geometry, \
             V4.1's DSpark file is refused by name, and the vendor's first two blocks on set {A} are \
             the draft's, 7 of 7 each, captured and eager, and block 1's first {NARROW} at a width of \
             {NARROW}"
        );
        Ok(())
    }

    /// The pairs the set's own tables give the vendor's blocks: block 1 runs after the prompt's last
    /// token at the prompt's length less one, block 2 after the first greedy id at its length. Any
    /// other pairing would compare a run with the ids of another run, and ends the gate by name.
    fn held_to_tables(prompt: &[u32], first_greedy: u32) -> Result<(), GateError> {
        let last = *prompt
            .last()
            .ok_or_else(|| format!("{A}: an empty prompt table"))?;
        let len = u32::try_from(prompt.len())?;
        let tables = [(len.saturating_sub(1), last), (len, first_greedy)];
        for (block, (pos, after)) in [&BLOCK_1, &BLOCK_2].into_iter().zip(tables) {
            if (block.pos, block.after) != (pos, after) {
                return Err(format!(
                    "{A}: the vendor's block after token {} at position {} is not the set's: its \
                     tables give token {after} at position {pos}",
                    block.after, block.pos
                )
                .into());
            }
        }
        Ok(())
    }

    /// Append `rows` (whole tapped rows of `tap` values) after the committed positions, a group at a
    /// time, and hold the committed count to `to`.
    fn append_rows(
        gpu: &Gpu,
        body: &mut DsparkBody<Layers>,
        rows: &[f32],
        (tap, to): (usize, u32),
    ) -> Result<(), GateError> {
        for chunk in rows.chunks(GROUP * tap) {
            let at = body.committed();
            body.append_as(gpu, at, chunk)?;
        }
        let held = body.committed();
        if held != to {
            return Err(format!("{held} positions committed; the block runs after {to}").into());
        }
        Ok(())
    }

    /// One of the vendor's blocks through the draft, `clause`'s lines: its 7 ids from the captured
    /// block and from its eager twin are the vendor's, and the two leave the same bits in the
    /// block's `m` rows. Whether both hold.
    fn block_is_oracle(
        gpu: &Gpu,
        body: &mut DsparkBody<Layers>,
        clause: &str,
        (block, m): (&VendorBlock, usize),
    ) -> Result<bool, GateError> {
        let VendorBlock { pos, after, ids } = *block;
        let mut captured = [0u32; WIDTH];
        body.propose_as(gpu, pos, after, ids.len(), &mut captured, &mut [])?;
        let captured_rows = body.block_rows_to_host(gpu.stream(), m)?;
        let mut eager = [0u32; WIDTH];
        body.propose_as_with(
            gpu,
            pos,
            after,
            ids.len(),
            &mut eager,
            &mut [],
            Submit::Eager,
        )?;
        let eager_rows = body.block_rows_to_host(gpu.stream(), m)?;
        let hits = captured.iter().zip(&ids).filter(|(a, b)| a == b).count();
        let same = captured == ids && eager == ids;
        let twin = bits_equal(&captured_rows, &eager_rows);
        println!(
            "{clause} set={A} context={pos} anchor={after} captured={captured:?} eager={eager:?} \
             oracle={ids:?} {hits}/{} {}",
            ids.len(),
            verdict(same)
        );
        println!(
            "{clause} captured block rows = eager block rows, bit for bit ({} values) {}",
            captured_rows.len(),
            verdict(twin)
        );
        Ok(same && twin)
    }

    /// The target's vocabulary: the rows of its token embedding.
    fn n_vocab_of(target: &gguf::Split) -> Result<usize, GateError> {
        let (_, t) = target
            .find("token_embd.weight")
            .ok_or_else(|| format!("{MODEL}: no token_embd.weight"))?;
        Ok(usize::try_from(*t.dims.get(1).ok_or_else(|| {
            format!("token_embd.weight is {:?}", t.dims)
        })?)?)
    }
}
