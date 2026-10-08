//! The Qwen3.8 UD-Q3_K_XL i-quant host-leg gate: the file's routed experts — IQ3_XXS and IQ4_XS gate
//! and up, IQ4_NL down — through the threaded host leg (`crates/model/src/moe.rs`, the leg the card
//! engine's host tier calls), which no other gate runs on an i-quant stack: `gate-moe` has no IQ
//! type, the qwen4exp e2e gate loads the Q4 file, and qdot's tests pin the dots serially.
//!
//! What is asserted:
//! - (3) the leg computes what the dot says. For a fixed sample of layers — every layer whose gate
//!   or up is IQ4_XS and the first, middle and last IQ3_XXS gate layer and the first and last
//!   IQ4_NL down layer — and in each eight columns listing ten of twelve host-served experts (ids
//!   past the plan's card prefix `n_l`), the leg's `experts_into` per column — each expert's gate,
//!   up, combine and down, and the weighted sum — and its `experts_union_into` over all eight
//!   columns (every expert's down in runs of up to `qdot::TILE_COLS` columns, the tile path) equal
//!   an independent recompute bit for bit. The oracle reads the type, the shape and the row bytes
//!   from the GGUF file through `gguf::inventory_of` and a positioned read, quantizes the
//!   activation with `qdot::quantize_col`, dots every row with `qdot::dot_row` (the one-column
//!   entry the leg's lone column takes, `ops.rs`'s `compute_rows`), combines with
//!   `qdot::swiglu_clamp` at the layers' limit (0.0, the plain combine) and sums in list order from
//!   zero, as `moe.rs`'s `serve_resolved` does. The leg's tile is `dot_row` per column bit for bit
//!   (qdot's tile gates), so the equality is exact; a table, a row offset or a type the leg reads
//!   differently from the oracle is red. No tap: the leg's own public entries expose each
//!   expert's gate, up, combine and down (`HostScratch`).
//! - (1) the leg runs IQ. The engine child's stderr names `load host_tier type=iq3_xxs`, `iq4_xs`
//!   and `iq4_nl` with `path=fused`, and the 512-position prompt's ubatch walk (`Prompt38::Gemm`,
//!   which reaches the host through the union) listed host slots on every layer of the file that
//!   holds an i-quant stack (`PromptStats` rows, the `stat prompt lb` records' source), grouped by
//!   the layer's (gate, up, down) types.
//! - (2) the thread count does not move a bit. One routed output row's sum is computed by one
//!   participant in qdot's fixed order (`ops.rs` `compute_rows` hands a row range to one dot per
//!   column; the row pool only decides who runs which rows, `run_row_pool`'s "a row's value does
//!   not depend on who computes it"; a column's quantization is one `quantize_col`; the sum over a
//!   column's experts is `serve_resolved`'s serial loop in list order), so the same 512-position
//!   prompt and 8 decode steps at the pool's default width and at width 1 leave the same ids and
//!   the same logits bit for bit. The pool is process-wide and fixed at its first use, so the two
//!   widths are two processes (this bin re-runs itself with `--engine`), each loading the file;
//!   both run with `BLOOMERY_POISON=1`, so a cell the leg missed is a NaN, and every logit must be
//!   finite before any bit is compared (all-NaN logits are equal bit for bit). Each child prints
//!   the pool's width, and the parent holds the default to more than one thread and the other to
//!   one.
//!
//! Clause 3 runs first, then the engine children whatever it found (a red clause 3 does not hide
//! clauses 1 and 2); a red clause 1 skips the width-1 child, and a child that ends nonzero is red.
//! Loads the whole host set: alone in a batch, under the big-load lock.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen4exp_iqleg: built without the `gpu` feature; see `just gate-gpu-qwen4exp-iqleg`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_iqleg", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[path = "shared/qwen38_open.rs"]
#[allow(
    dead_code,
    reason = "the e2e gate prints and reads the plan's whole facts; this gate prints the counts and the context"
)]
mod qwen38_open;

#[cfg(feature = "gpu")]
mod gate {
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Instant;

    use crate::qwen38_open::Open38;

    use bloomery_gpu::arch::qwen3moe::{Prompt38, Qwen38Model};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use gguf::{GgmlType, Split};
    use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts};
    use model::moe::{HostLayer, HostLayerSpec, HostScratch, UnionScratch};
    use model::ops::Tensor2;
    use model::placement::PlanLevers;
    use model::r8file::R8Source;

    /// The UD-Q3_K_XL file, first of three shards (90.0 GB).
    const MODEL: &str =
        "/models/Qwen3.8-Flash-Next-UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf";

    /// Cache rows of the engine child: the prompt and the steps, with room.
    const CTX: usize = 1024;
    /// The prompt's positions: exactly the 512 under `xsplit::m_min`
    /// (`crates/runtime/src/xsplit.rs:75`), so nothing streams or is admitted and every
    /// non-resident expert runs on the leg.
    const PROMPT: usize = 512;
    /// Decode steps after the prompt's own token.
    const STEPS: usize = 8;

    /// Columns of a clause 3 union call: one `qdot::TILE_COLS` run.
    const COLS: usize = 8;
    /// Host-served experts a layer's sample draws its lists from, and each list's length.
    const POOL: usize = 12;
    const LIST: usize = 10;
    /// The pool's twelve experts as positions over 0..=410, scaled onto a layer's host-served ids.
    const POOL_AT: [u32; POOL] = [0, 5, 31, 60, 88, 120, 160, 200, 250, 300, 360, 410];

    /// The weight type, shape and bytes of one routed stack, read from the file's own header
    /// (`gguf::inventory_of`) and a positioned read: the oracle's reader, sharing nothing with the
    /// host tier's `ShardTensor`.
    struct Stack {
        name: String,
        path: PathBuf,
        base: u64,
        ty: GgmlType,
        k: usize,
        n: usize,
        n_expert: usize,
        row_bytes: usize,
    }

    impl Stack {
        /// The matrix of expert `e`: `n` rows of `row_bytes`, read from the file.
        fn expert(&self, e: usize) -> Result<Vec<u8>, GateError> {
            let per = self.n * self.row_bytes;
            let mut bytes = vec![0u8; per];
            File::open(&self.path)
                .and_then(|f| f.read_exact_at(&mut bytes, self.base + (e * per) as u64))
                .map_err(|err| format!("read {} expert {e}: {err}", self.name))?;
            Ok(bytes)
        }
    }

    /// The header inventories of the file's shards, by path.
    struct Inventories(Vec<(PathBuf, gguf::Inventory)>);

    impl Inventories {
        fn of(split: &Split) -> Result<Inventories, GateError> {
            let mut all = Vec::new();
            for i in 0..split.shard_count() {
                let path = split.shard_path(i).ok_or("a shard without a path")?;
                let inv =
                    gguf::inventory_of(path).map_err(|e| format!("{}: {e}", path.display()))?;
                all.push((path.to_path_buf(), inv));
            }
            Ok(Inventories(all))
        }

        fn stack(&self, name: &str) -> Result<Stack, GateError> {
            for (path, inv) in &self.0 {
                let Some(t) = inv.tensors.iter().find(|t| t.name == name) else {
                    continue;
                };
                let [k, n, n_expert] = t.dims[..] else {
                    return Err(format!("{name}: dims {:?} are not [k, n, experts]", t.dims).into());
                };
                let nbytes = t
                    .nbytes
                    .ok_or_else(|| format!("{name}: the header sizes no type"))?;
                let rows = (n * n_expert) as usize;
                if rows == 0 || !(nbytes as usize).is_multiple_of(rows) {
                    return Err(format!("{name}: {nbytes} bytes over {rows} rows").into());
                }
                return Ok(Stack {
                    name: name.to_owned(),
                    path: path.clone(),
                    base: inv.data_base + t.offset,
                    ty: GgmlType::from_u32(t.type_id),
                    k: k as usize,
                    n: n as usize,
                    n_expert: n_expert as usize,
                    row_bytes: nbytes as usize / rows,
                });
            }
            Err(format!("{name}: no shard of the file holds it").into())
        }
    }

    fn is_iq(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::IQ3_XXS | GgmlType::IQ4_XS | GgmlType::IQ4_NL)
    }

    fn names(l: usize) -> [String; 3] {
        ["gate", "up", "down"].map(|w| format!("blk.{l}.ffn_{w}_exps.weight"))
    }

    /// Every layer's `[gate, up, down]` types, from the file's header.
    fn layer_types(inv: &Inventories, n_layer: usize) -> Result<Vec<[GgmlType; 3]>, GateError> {
        (0..n_layer)
            .map(|l| {
                let [g, u, d] = names(l);
                Ok([inv.stack(&g)?.ty, inv.stack(&u)?.ty, inv.stack(&d)?.ty])
            })
            .collect()
    }

    /// The layer's kind as clause 1 groups it: `iq3_xxs/iq3_xxs/iq4_nl`.
    fn kind_of(t: &[GgmlType; 3]) -> String {
        format!("{}/{}/{}", t[0], t[1], t[2])
    }

    /// A splitmix64 stream: the sample's fixed activations.
    fn stream(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// `n` values in `(-scale, scale)`.
    fn seeded(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut next = stream(seed);
        (0..n)
            .map(|_| ((next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale)
            .collect()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The first differing index of two rows (a length mismatch is index `min(len)`).
    fn first_diff(a: &[f32], b: &[f32]) -> Option<usize> {
        a.iter()
            .zip(b)
            .position(|(x, y)| x.to_bits() != y.to_bits())
            .or((a.len() != b.len()).then(|| a.len().min(b.len())))
    }

    /// What the oracle computed for one (column, expert): the gate and up rows, the combine and
    /// the down rows.
    struct Expert {
        gate: Vec<f32>,
        up: Vec<f32>,
        act: Vec<f32>,
        down: Vec<f32>,
    }

    /// One routed layer's three stacks as bytes: gate, up and down, the experts the sample reads.
    struct Mats {
        stacks: [Stack; 3],
        bytes: BTreeMap<u32, [Vec<u8>; 3]>,
    }

    impl Mats {
        fn of(inv: &Inventories, l: usize, experts: &[u32]) -> Result<Mats, GateError> {
            let [g, u, d] = names(l);
            let stacks = [inv.stack(&g)?, inv.stack(&u)?, inv.stack(&d)?];
            let mut bytes = BTreeMap::new();
            for &e in experts {
                let e = e as usize;
                if e >= stacks[0].n_expert {
                    return Err(format!("layer {l}: expert {e} is past the stack's").into());
                }
                bytes.insert(
                    e as u32,
                    [
                        stacks[0].expert(e)?,
                        stacks[1].expert(e)?,
                        stacks[2].expert(e)?,
                    ],
                );
            }
            Ok(Mats { stacks, bytes })
        }

        /// Expert `e` over activation column `x`, by `qdot` only: `quantize_col`, `dot_row` a row,
        /// `swiglu_clamp` at limit 0.0, `quantize_col` and `dot_row` again for the down.
        fn expert(&self, e: u32, x: &[f32]) -> Result<Expert, GateError> {
            let [sg, su, sd] = &self.stacks;
            let [bg, bu, bd] = &self.bytes[&e];
            let rows = |s: &Stack, b: &[u8], xq: &[u8]| -> Result<Vec<f32>, GateError> {
                (0..s.n)
                    .map(|r| {
                        qdot::dot_row(s.ty, &b[r * s.row_bytes..(r + 1) * s.row_bytes], xq, s.k)
                            .map_err(|err| format!("{} row {r}: {err}", s.name).into())
                    })
                    .collect()
            };
            let quant = |ty: GgmlType, k: usize, v: &[f32]| {
                let mut q = vec![0u8; qdot::col_bytes(ty, k)];
                qdot::quantize_col(ty, v, &mut q);
                q
            };
            let gate = rows(sg, bg, &quant(sg.ty, sg.k, x))?;
            let up = rows(su, bu, &quant(su.ty, su.k, x))?;
            let mut act = vec![0.0f32; gate.len()];
            qdot::swiglu_clamp(&gate, &up, 0.0, &mut act);
            let down = rows(sd, bd, &quant(sd.ty, sd.k, &act))?;
            Ok(Expert {
                gate,
                up,
                act,
                down,
            })
        }
    }

    /// `Σ w · down` in list order from zero, as `serve_resolved`'s loop adds.
    fn weighted(list: &[(u32, f32)], downs: &[&[f32]], embd: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; embd];
        for (&(_, w), d) in list.iter().zip(downs) {
            for (o, &dv) in out.iter_mut().zip(*d) {
                *o += w * dv;
            }
        }
        out
    }

    /// The layers clause 3 samples, from the file's types: every layer whose gate or up is IQ4_XS,
    /// the first, middle and last IQ3_XXS gate layer, and the first and last IQ4_NL down layer.
    fn sample_layers(types: &[[GgmlType; 3]]) -> Vec<usize> {
        let mut at = std::collections::BTreeSet::new();
        let pick = |has: &dyn Fn(&[GgmlType; 3]) -> bool,
                    at: &mut std::collections::BTreeSet<usize>,
                    ends: bool| {
            let v: Vec<usize> = (0..types.len()).filter(|&l| has(&types[l])).collect();
            if ends {
                at.extend(v.first().copied());
                at.extend(v.get(v.len() / 2).copied());
                at.extend(v.last().copied());
            } else {
                at.extend(v);
            }
        };
        pick(
            &|t| t[0] == GgmlType::IQ4_XS || t[1] == GgmlType::IQ4_XS,
            &mut at,
            false,
        );
        pick(&|t| t[0] == GgmlType::IQ3_XXS, &mut at, true);
        pick(&|t| t[2] == GgmlType::IQ4_NL, &mut at, true);
        at.into_iter().collect()
    }

    /// The plan's per-layer card prefix `n_l` and the file's dims, for the model's own placement
    /// at the gate plan (`Experts::Card`): the host serves the ids past each layer's prefix.
    fn card_prefix(split: &Split) -> Result<(Vec<u64>, usize, usize, usize), GateError> {
        let inputs = PlanInputs::describe(split)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let machine = machine_for_experts(
            crate::gate_card::card()?,
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            Experts::Card,
        );
        let plan = inputs.plan_with_slots(
            &machine,
            CTX as u64,
            &PlanLevers::default(),
            Experts::Card,
            1,
        )?;
        let hp = &inputs.hp;
        Ok((plan.n_l.clone(), hp.n_embd, hp.expert_ff, hp.n_used))
    }

    // ----------------------------------------------------------------- clause 3

    /// A leg call's result, with a panic or an error told apart by name; both are red.
    fn leg<T>(what: &str, f: impl FnOnce() -> Result<T, model::ModelError>) -> Result<T, String> {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(format!("{what}: the leg refused: {e}")),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                Err(format!("{what}: the leg panicked: {msg}"))
            }
        }
    }

    /// Clause 3 on one layer; the number of cells compared, or the failure lines.
    fn leg_layer(
        split: &Split,
        inv: &Inventories,
        l: usize,
        card_n: u64,
        (embd, ff, n_used): (usize, usize, usize),
        n_expert: usize,
    ) -> Result<(usize, Vec<String>), GateError> {
        // The twelve ids spread over the host-served range `card_n..n_expert`.
        let span = n_expert as u64 - card_n;
        let pool: Vec<u32> = POOL_AT
            .iter()
            .map(|&o| (card_n + u64::from(o) * (span - 1) / u64::from(POOL_AT[POOL - 1])) as u32)
            .collect();
        if span < POOL as u64 || pool.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!(
                "clause 3: layer {l}: the card prefix {card_n} leaves no room for {POOL} host \
                 experts of {n_expert}"
            )
            .into());
        }
        let lists: Vec<Vec<(u32, f32)>> = (0..COLS)
            .map(|j| {
                (0..LIST)
                    .map(|s| {
                        (
                            pool[(2 * j + s) % POOL],
                            0.05 + 0.011 * s as f32 + 0.003 * j as f32,
                        )
                    })
                    .collect()
            })
            .collect();
        let xs: Vec<Vec<f32>> = (0..COLS)
            .map(|j| {
                seeded(
                    embd,
                    0x1e9_0000 + (l * COLS + j) as u64,
                    0.6 + 0.35 * j as f32,
                )
            })
            .collect();
        let mats = Mats::of(inv, l, &pool)?;
        let [gn, un, dn] = names(l);
        let spec = HostLayerSpec {
            gate: &gn,
            up: &un,
            down: &dn,
            n_expert,
            embd,
            ff,
            swiglu_limit: 0.0,
        };
        let src = R8Source::rows(split);
        let layer = HostLayer::build(src, &spec)?;
        let tag = format!(
            "layer {l} ({})",
            kind_of(&mats.stacks.each_ref().map(|s| s.ty))
        );
        let mut bad: Vec<String> = Vec::new();
        let mut cells = 0usize;
        let mut hs = HostScratch::new(embd, ff, n_used)?;
        let mut us = UnionScratch::new_routed(embd, ff, COLS, n_used)?;
        let mut want_out: Vec<Vec<f32>> = Vec::with_capacity(COLS);
        let mut single_out: Vec<Vec<f32>> = Vec::with_capacity(COLS);
        for j in 0..COLS {
            let oracle: Vec<Expert> = lists[j]
                .iter()
                .map(|&(e, _)| mats.expert(e, &xs[j]))
                .collect::<Result<_, _>>()?;
            let want = weighted(
                &lists[j],
                &oracle.iter().map(|o| &o.down[..]).collect::<Vec<_>>(),
                embd,
            );
            let xj = Tensor2::from_vec(embd, 1, xs[j].clone());
            let mut out = vec![f32::NAN; embd];
            let r = leg(&format!("{tag} column {j} experts_into"), || {
                layer.experts_into(src, &xj, &lists[j], &mut out, &mut hs)
            });
            if let Err(e) = r {
                bad.push(format!("clause 3: {e}"));
                want_out.push(want);
                single_out.push(out);
                continue;
            }
            for (i, o) in oracle.iter().enumerate() {
                for (stage, got, exp) in [
                    ("gate", hs.gate(i).col(0), &o.gate),
                    ("up", hs.up(i).col(0), &o.up),
                    ("combine", hs.par(i).col(0), &o.act),
                    ("down", hs.down(i).col(0), &o.down),
                ] {
                    cells += exp.len();
                    if let Some(at) = first_diff(got, exp) {
                        bad.push(format!(
                            "clause 3: {tag} column {j} expert {} {stage} row {at}: the leg {:?} \
                             != the dot {:?}",
                            lists[j][i].0,
                            got.get(at),
                            exp.get(at)
                        ));
                    }
                }
            }
            cells += embd;
            if let Some(at) = first_diff(&out, &want) {
                bad.push(format!(
                    "clause 3: {tag} column {j} weighted sum [{at}]: the leg {:?} != the dot {:?}",
                    out.get(at),
                    want.get(at)
                ));
            }
            want_out.push(want);
            single_out.push(out);
        }
        let x_all = Tensor2::from_vec(embd, COLS, xs.concat());
        let list_refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
        let mut out_all = vec![f32::NAN; embd * COLS];
        let r = leg(&format!("{tag} experts_union_into"), || {
            layer.experts_union_into(src, &x_all, &list_refs, &mut out_all, &mut us)
        });
        match r {
            Err(e) => bad.push(format!("clause 3: {e}")),
            Ok(()) => {
                for j in 0..COLS {
                    let got = &out_all[j * embd..(j + 1) * embd];
                    cells += embd;
                    if let Some(at) = first_diff(got, &want_out[j]) {
                        bad.push(format!(
                            "clause 3: {tag} union column {j} [{at}]: the leg's tile {:?} != the \
                             dot {:?}",
                            got.get(at),
                            want_out[j].get(at)
                        ));
                    }
                    if !same_bits(got, &single_out[j]) && bad.is_empty() {
                        bad.push(format!(
                            "clause 3: {tag} union column {j}: the union differs from \
                             experts_into"
                        ));
                    }
                }
            }
        }
        Ok((cells, bad))
    }

    /// Clause 3 over the sample.
    fn clause_leg(
        split: &Split,
        inv: &Inventories,
        types: &[[GgmlType; 3]],
    ) -> Result<bool, GateError> {
        let (n_l, embd, ff, n_used) = card_prefix(split)?;
        let n_expert = inv.stack(&names(0)[0])?.n_expert;
        let layers = sample_layers(types);
        let seen: Vec<GgmlType> = layers.iter().flat_map(|&l| types[l]).collect();
        let mut ok = layers.len() >= 3;
        for ty in [GgmlType::IQ3_XXS, GgmlType::IQ4_XS, GgmlType::IQ4_NL] {
            let has = seen.contains(&ty);
            println!(
                "clause 3 sample covers {ty}: {} {}",
                if has { "yes" } else { "no" },
                verdict(has)
            );
            ok &= has;
        }
        println!(
            "clause 3 sample: layers {layers:?} of {} (the card prefix n_l {:?}), {COLS} columns \
             of {LIST} experts over {POOL} host-served ids {}",
            types.len(),
            layers.iter().map(|&l| n_l[l]).collect::<Vec<_>>(),
            verdict(layers.len() >= 3)
        );
        for &l in &layers {
            let t = Instant::now();
            let (cells, bad) = leg_layer(split, inv, l, n_l[l], (embd, ff, n_used), n_expert)?;
            let pass = bad.is_empty();
            for b in bad.iter().take(6) {
                println!("{b}");
            }
            println!(
                "clause 3 layer {l} {} n_l={}: {cells} cells bit-equal to the dot in {:.1} s {}",
                kind_of(&types[l]),
                n_l[l],
                t.elapsed().as_secs_f64(),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------------------ engine child

    /// The 512-position prompt: fixed ids over the first 100,000 of the vocabulary (the image
    /// placeholder 248,056 and every id past the vocabulary are refused by the prompt call).
    fn prompt_ids() -> Vec<u32> {
        let mut next = stream(0x1415_9265);
        (0..PROMPT)
            .map(|_| 1000 + (next() % 99_000) as u32)
            .collect()
    }

    /// The model placed on the gate card by its plan, the routed experts on the card's id prefix
    /// and the host beyond it (`Experts::Card`), under the plan's own machine: the e2e gate's open.
    fn open() -> Result<Qwen38Model, GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let file = crate::qwen38_open::open_split(Path::new(MODEL))?;
        let t = Instant::now();
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let card = crate::gate_card::card()?;
        let plan_levers = PlanLevers::default();
        let planned = Open38 {
            card,
            ctx: CTX,
            ub,
            experts: Experts::Card,
            slots: 1,
            plan_levers: &plan_levers,
            host: levers.host(),
            room: None,
            residency: Residency::Off,
        }
        .plan(file)?;
        let p = planned.facts();
        println!(
            "plan card={} host_experts={} card_experts={} ctx_max={} ubatch={ub}",
            card.name, p.host_experts, p.card_experts, p.ctx_max
        );
        let o = planned.open()?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            o.model.resident_bytes(),
            o.model.layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok(o.model)
    }

    /// `ids` and `logits` as little-endian words, the dump the parent compares.
    fn write_dump(path: &str, ids: &[u32], logits: &[Vec<f32>]) -> Result<(), GateError> {
        let mut bytes = Vec::new();
        for &i in ids {
            bytes.extend(i.to_le_bytes());
        }
        for row in logits {
            for v in row {
                bytes.extend(v.to_bits().to_le_bytes());
            }
        }
        std::fs::write(path, bytes).map_err(|e| format!("write {path}: {e}").into())
    }

    /// The engine child: the pool's width, the load, the 512-position prompt through the ubatch
    /// walk, 8 decode steps, every logit held finite, the walk's per-layer host slots and the dump.
    fn child(dump: &str) -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        println!("iqleg width={}", threads::pool().threads());
        crate::gate_card::init()?;
        let mut m = open()?;
        m.set_prompt38_stats(true)?;
        let ids = prompt_ids();
        let before = m.body("iqleg")?.hybrid().stats();
        let t = Instant::now();
        let mut next = m.prompt38(&ids, Prompt38::Gemm)?;
        let after = m.body("iqleg")?.hybrid().stats();
        println!(
            "iqleg prompt positions={PROMPT} services={} cols={} host_slots={} in {:.1} s \
             (runtime value)",
            after.batch_served - before.batch_served,
            after.batch_cols - before.batch_cols,
            after.batch_host_slots - before.batch_host_slots,
            t.elapsed().as_secs_f64()
        );
        let stats = m
            .take_prompt38_stats()?
            .ok_or("clause 1: the prompt walked no ubatch (no stats row)")?;
        for r in &stats.rows {
            println!(
                "iqleg lb layer={} cols={} slots={}",
                r.layer, r.cols, r.slots
            );
        }
        let mut toks = vec![next];
        let mut logits = vec![m.logits()?];
        for s in 0..STEPS {
            next = m.step(&[next])?;
            toks.push(next);
            logits.push(m.logits()?);
            println!("iqleg step {s} token {next}");
        }
        for (i, row) in logits.iter().enumerate() {
            if let Some(at) = row.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "clause 2: logits row {i} holds a non-finite value at {at} (BLOOMERY_POISON=1 \
                     turns a cell the leg missed into NaN)"
                )
                .into());
            }
        }
        write_dump(dump, &toks, &logits)?;
        println!("iqleg dump {dump} ids={} rows={}", toks.len(), logits.len());
        Ok(())
    }

    // ------------------------------------------------------------------ parent

    /// What a child run left: its width, the layers' host slots, and its dump.
    struct Child {
        width: usize,
        slots: BTreeMap<usize, u64>,
        stderr: String,
        dump: Vec<u8>,
    }

    /// Runs this binary as `--engine` with `threads` (`None`: the pool's default), forwards its
    /// output, and reads what it left. A child that ends nonzero is the clause's red by name.
    fn run_child(
        tag: &str,
        threads: Option<usize>,
        clause: &str,
    ) -> Result<Option<Child>, GateError> {
        let dump = std::env::temp_dir().join(format!("iqleg-{}-{tag}.bin", std::process::id()));
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg("--engine").arg(&dump).env("BLOOMERY_POISON", "1");
        match threads {
            Some(n) => cmd.env(bloomery_levers::THREADS, n.to_string()),
            None => cmd.env_remove(bloomery_levers::THREADS),
        };
        let t = Instant::now();
        let out = cmd
            .output()
            .map_err(|e| format!("{clause}: run the {tag} child: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        for line in stdout.lines() {
            if !line.starts_with("iqleg lb ") {
                println!("[{tag}] {line}");
            }
        }
        for line in stderr.lines() {
            eprintln!("[{tag}] {line}");
        }
        println!(
            "child {tag}: {} in {:.1} s (runtime value)",
            out.status,
            t.elapsed().as_secs_f64()
        );
        if !out.status.success() {
            println!(
                "{clause}: the {tag} child ended {} (its output is above) {}",
                out.status,
                verdict(false)
            );
            return Ok(None);
        }
        let field = |line: &str, key: &str| -> Option<u64> {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(key)?.parse().ok())
        };
        let mut width = 0usize;
        let mut slots = BTreeMap::new();
        for line in stdout.lines() {
            if let Some(w) = line.strip_prefix("iqleg width=") {
                width = w.trim().parse()?;
            } else if line.starts_with("iqleg lb ") {
                let (Some(layer), Some(s)) = (field(line, "layer="), field(line, "slots=")) else {
                    return Err(format!("{clause}: a malformed record: {line}").into());
                };
                *slots.entry(layer as usize).or_insert(0) += s;
            }
        }
        let bytes =
            std::fs::read(&dump).map_err(|e| format!("{clause}: read {}: {e}", dump.display()))?;
        let _ = std::fs::remove_file(&dump);
        Ok(Some(Child {
            width,
            slots,
            stderr,
            dump: bytes,
        }))
    }

    /// Clause 1 on the default-width child: the three types' `host_tier` lines, and host slots on
    /// every layer that holds an i-quant stack, by kind.
    fn clause_runs(c: &Child, types: &[[GgmlType; 3]]) -> bool {
        let mut ok = true;
        for ty in ["iq3_xxs", "iq4_xs", "iq4_nl"] {
            let has = c.stderr.lines().any(|l| {
                l.contains(&format!("load host_tier type={ty} ")) && l.contains("path=fused")
            });
            println!(
                "clause 1 load host_tier type={ty} path=fused: {} {}",
                has,
                verdict(has)
            );
            ok &= has;
        }
        let mut kinds: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (l, t) in types.iter().enumerate() {
            if t.iter().copied().any(is_iq) {
                kinds.entry(kind_of(t)).or_default().push(l);
            }
        }
        for (kind, layers) in &kinds {
            let silent: Vec<usize> = layers
                .iter()
                .copied()
                .filter(|l| c.slots.get(l).copied().unwrap_or(0) == 0)
                .collect();
            let total: u64 = layers
                .iter()
                .map(|l| c.slots.get(l).copied().unwrap_or(0))
                .sum();
            let pass = silent.is_empty();
            println!(
                "clause 1 kind {kind}: {} layers, host slots {total}, layers with none {silent:?} {}",
                layers.len(),
                verdict(pass)
            );
            ok &= pass;
        }
        ok &= !kinds.is_empty();
        ok
    }

    /// Clause 2: the widths, then ids and logits bit for bit.
    fn clause_width(a: &Child, b: &Child) -> bool {
        let widths = a.width > 1 && b.width == 1;
        println!(
            "clause 2 widths: default {} and {} (want > 1 and 1) {}",
            a.width,
            b.width,
            verdict(widths)
        );
        let same = a.dump.len() == b.dump.len() && !a.dump.is_empty();
        let diff = a.dump.iter().zip(&b.dump).filter(|(x, y)| x != y).count();
        let first = a.dump.iter().zip(&b.dump).position(|(x, y)| x != y);
        let pass = same && diff == 0;
        println!(
            "clause 2 bits: {} bytes vs {}, {diff} differ{} {}",
            a.dump.len(),
            b.dump.len(),
            first.map_or(String::new(), |at| format!(
                ", first at byte {at} ({})",
                if at < 4 * (1 + STEPS) {
                    "an id"
                } else {
                    "a logit"
                }
            )),
            verdict(pass)
        );
        widths && pass
    }

    pub fn run() -> Result<(), GateError> {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() == Some("--engine") {
            let dump = args.next().ok_or("--engine takes the dump path")?;
            return child(&dump);
        }
        bloomery_levers::at_main(&[])?;
        crate::gate_card::init()?;
        let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inv = Inventories::of(&split)?;
        let n_layer = PlanInputs::describe(&split)?.spec.layers.len();
        let types = layer_types(&inv, n_layer)?;
        let mut ok = clause_leg(&split, &inv, &types)?;
        if !ok {
            println!(
                "clause 3 is red; clauses 1 and 2 still run {}",
                verdict(false)
            );
        }
        let Some(a) = run_child("default", None, "clause 1/2")? else {
            println!("clauses 1 and 2 not compared: the default-width child ended");
            return Err(checks_failed());
        };
        let runs = clause_runs(&a, &types);
        ok &= runs;
        if !runs {
            println!(
                "clause 2 not run: clause 1 is red, so the width-1 child is skipped {}",
                verdict(false)
            );
            return Err(checks_failed());
        }
        let Some(b) = run_child("width1", Some(1), "clause 2")? else {
            println!("clause 2 not compared: the width-1 child ended");
            return Err(checks_failed());
        };
        ok &= clause_width(&a, &b);
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
