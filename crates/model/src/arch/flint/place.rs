//! Flint's validated description attached to the common hybrid placement and buffer owners.

use gguf::Split;
use models::{Ffn, Mixer, ModelSpec};

use super::spec;
use crate::arch::{names, place_hybrid};
use crate::placement::accepts::{Program, leg_down_forms};
use crate::placement::kernels::attention::{Forms, Head128Scratch};
use crate::placement::kernels::ffn::{PromptWalk, RoutedBatch, WideBlock, WideForms};
use crate::placement::workstation::{TierBatchBytes, TierLegScratch};
use crate::placement::{ModelTensors, PlacementError, kernels, workstation};

pub use place_hybrid::{KvLayout, PlaceError};

/// No second hyperparameter reader: the validated description owns Flint's facts.
pub type PlanInputs = place_hybrid::PlanInputs<()>;

/// The common host union's prompt capacity, also the window ring's batch allowance.
pub const PROMPT_ROWS: usize = crate::moe::UNION_MAX_COLS;
/// Tokens served by one run of the common expert tier leg.
pub const TIER_CHUNK: u64 = 8;

fn overflow() -> PlacementError {
    PlacementError::Metadata {
        key: "flint prompt reserve".into(),
        detail: "buffer bytes overflow u64".into(),
    }
}

fn reads(model: &ModelTensors, name: &str) -> Result<kernels::SiteAct, PlacementError> {
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| PlacementError::Tensor {
            name: name.into(),
            detail: "is not in the validated description".into(),
        })?;
    kernels::dense(tensor.ty)
        .map(|site| site.reads())
        .ok_or_else(|| PlacementError::Tensor {
            name: name.into(),
            detail: format!("type {:?} has no dense projection site", tensor.ty),
        })
}

/// The actual Q/K/V and output forms, widest rows and maximum partial segments of the shared
/// head128 attention scratch. NoPE changes no buffer width; windowing changes stores, not the
/// maximum flash launch scratch a full layer needs.
pub fn attention_scratch(
    spec: &ModelSpec,
    model: &ModelTensors,
) -> Result<Head128Scratch, PlacementError> {
    let mut scratch = Head128Scratch {
        hidden: u64::from(spec.hidden),
        query: 0,
        key_value: 0,
        heads: 0,
        segments: 0,
        input: Forms::default(),
        output: Forms::default(),
    };
    for (l, layer) in spec.layers.iter().enumerate() {
        let Mixer::Gqa(g) = &layer.mixer else {
            return Err(PlacementError::Metadata {
                key: format!("layer {l} attention"),
                detail: "Flint requires GQA".into(),
            });
        };
        scratch.query = scratch
            .query
            .max(u64::from(g.heads) * u64::from(g.head_dim));
        scratch.key_value = scratch
            .key_value
            .max(u64::from(g.kv_heads) * u64::from(g.head_dim));
        scratch.heads = scratch.heads.max(u64::from(g.heads));
        scratch.segments = scratch
            .segments
            .max(g.window.map_or(80, |w| u64::from(w).div_ceil(64)));
        for name in [names::attn_q(l), names::attn_k(l), names::attn_v(l)] {
            scratch.input = scratch.input.with(reads(model, &name)?);
        }
        scratch.output = scratch.output.with(reads(model, &names::attn_output(l))?);
    }
    Ok(scratch)
}

fn block_forms(
    model: &ModelTensors,
    names: [String; 3],
    block: &mut WideBlock,
) -> Result<(), PlacementError> {
    block.input = block
        .input
        .with(reads(model, &names[0])?)
        .with(reads(model, &names[1])?);
    block.down = block.down.with(reads(model, &names[2])?);
    Ok(())
}

/// The common routed/shared batch with selected dense/shared forms. `card` prices the allocation
/// made when either resident card serves experts; a plan reserves that candidate before assigning
/// expert ids. `tiered` adds only the prompt join's three slot words.
pub fn routed_batch(
    spec: &ModelSpec,
    model: &ModelTensors,
    rows: u64,
    card: bool,
    tiered: bool,
) -> Result<RoutedBatch, PlacementError> {
    let empty = WideBlock {
        hidden: u64::from(spec.hidden),
        ff: 0,
        input: WideForms::default(),
        down: WideForms::default(),
    };
    let mut shared = empty;
    let mut dense = empty;
    let mut experts = 0;
    let mut used = 0;
    let mut expert_ff = 0;
    for (l, layer) in spec.layers.iter().enumerate() {
        match &layer.ffn {
            Ffn::Dense { ff, .. } => {
                dense.ff = dense.ff.max(u64::from(*ff));
                block_forms(
                    model,
                    [names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)],
                    &mut dense,
                )?;
            }
            Ffn::Moe(moe) => {
                experts = experts.max(u64::from(moe.experts));
                used = used.max(u64::from(moe.top_k));
                expert_ff = expert_ff.max(u64::from(moe.expert_ff));
                let sh = moe
                    .shared
                    .as_ref()
                    .ok_or_else(|| PlacementError::Metadata {
                        key: format!("layer {l} shared block"),
                        detail: "Flint requires one ungated shared MLP".into(),
                    })?;
                shared.ff = shared.ff.max(u64::from(sh.ff));
                block_forms(
                    model,
                    [
                        names::ffn_gate_shexp(l),
                        names::ffn_up_shexp(l),
                        names::ffn_down_shexp(l),
                    ],
                    &mut shared,
                )?;
            }
        }
    }
    Ok(RoutedBatch {
        hidden: u64::from(spec.hidden),
        experts,
        used,
        shared,
        dense: (dense.ff > 0).then_some(dense),
        card: card.then_some(TierLegScratch {
            n_embd: u64::from(spec.hidden),
            ff: expert_ff,
            n_used: used,
            chunk: rows,
            downs: leg_down_forms(Program::FlintBody, model),
            input_scales: true,
        }),
        tiered,
        router_logits: false,
    })
}

/// Prompt reserve from the same descriptors the attention, FFN and walk allocate by. It includes
/// the candidate card batch, not a quantization padding estimate or a per-layer copy of scratch.
pub fn prompt_reserve_bytes(
    spec: &ModelSpec,
    model: &ModelTensors,
    rows: u64,
) -> Result<u64, PlacementError> {
    let attention = attention_scratch(spec, model)?
        .bytes(rows)
        .ok_or_else(overflow)?;
    let walk = PromptWalk {
        hidden: u64::from(spec.hidden),
        sets: 1,
    }
    .bytes(rows)
    .ok_or_else(overflow)?;
    let ffn = routed_batch(spec, model, rows, true, false)?
        .bytes(rows)
        .ok_or_else(overflow)?;
    attention
        .checked_add(walk)
        .and_then(|bytes| bytes.checked_add(ffn))
        .ok_or_else(overflow)
}

/// Price the selected prompt group before borrowing the inputs for a plan. Reconfiguration is
/// idempotent: only unit rows multiply; attention, FFN and the host-sum upload stay shared.
pub fn configure_group(inputs: &mut PlanInputs, group: usize) -> Result<(), PlacementError> {
    runtime::prompt::check_group(group, bloomery_levers::PREFILL_GROUP_MAX as usize).map_err(
        |error| PlacementError::Metadata {
            key: runtime::prompt::GROUP_LEVER.into(),
            detail: error.to_string(),
        },
    )?;
    let rows = PROMPT_ROWS as u64;
    let hidden = u64::from(inputs.spec.hidden);
    let one = PromptWalk { hidden, sets: 1 }
        .bytes(rows)
        .ok_or_else(overflow)?;
    let selected = PromptWalk {
        hidden,
        sets: runtime::prompt::group_sets(group) as u64,
    }
    .bytes(rows)
    .ok_or_else(overflow)?;
    let base = prompt_reserve_bytes(&inputs.spec, &inputs.model, rows)?;
    inputs.prompt_reserve = base
        .checked_sub(one)
        .and_then(|bytes| bytes.checked_add(selected))
        .ok_or_else(overflow)?;
    Ok(())
}

/// The real expert tier's staging, protected leg scratch and host exchanges for one prompt unit.
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let batch = routed_batch(&inputs.spec, &inputs.model, TIER_CHUNK, true, false)?;
    let leg = batch.card.ok_or_else(overflow)?;
    workstation::try_tier_leg_batch_bytes_of(
        leg.n_embd,
        leg.ff,
        leg.n_used,
        (TIER_CHUNK, PROMPT_ROWS as u64),
        leg.downs,
        true,
    )
}

impl PlanInputs {
    /// Describe once, then require the genuine program and its numerical/site bindings.
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        Self::describe(split)?.runnable()
    }

    /// The validated file, common windowed stores, actual prompt reserve and tier loader contract.
    /// The common host-run loader serves NVMe through the same residency arena that placement split.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let read = spec::read(split)?;
        let reserve = prompt_reserve_bytes(&read.spec, &read.tensors, PROMPT_ROWS as u64)?;
        let batch = routed_batch(&read.spec, &read.tensors, PROMPT_ROWS as u64, false, true)?;
        let tier =
            kernels::ffn::tier_places_bytes(PROMPT_ROWS as u64, batch.used).ok_or_else(overflow)?;
        Ok(Self::new("Flint", (), read.tensors, read.spec)?
            .with_prompt_reserve(reserve)
            .with_tier_reserve(tier)
            .with_nvme_tier())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{check, keys, tensors};
    use super::*;
    use crate::arch::synthetic::{V, header_shaped};

    #[test]
    fn description_uses_real_prompt_and_tier_terms_without_admitting_unsupported_shapes() {
        let path = header_shaped(
            "flint-place-description",
            "flint",
            &keys(),
            &[("tokenizer.ggml.pre", V::Str("flint"))],
            &tensors(),
        );
        let split = Split::open(&path).unwrap();
        let mut inputs = PlanInputs::describe(&split).unwrap();
        assert_eq!(
            inputs.prompt_reserve,
            attention_scratch(&inputs.spec, &inputs.model)
                .unwrap()
                .bytes(PROMPT_ROWS as u64)
                .unwrap()
                + PromptWalk {
                    hidden: 16,
                    sets: 1
                }
                .bytes(PROMPT_ROWS as u64)
                .unwrap()
                + routed_batch(&inputs.spec, &inputs.model, PROMPT_ROWS as u64, true, false)
                    .unwrap()
                    .bytes(PROMPT_ROWS as u64)
                    .unwrap()
        );
        assert_eq!(inputs.tier_reserve, Some(12 * PROMPT_ROWS as u64 * 2));
        let tier = tier_batch(&inputs).unwrap();
        let leg = tier.leg.unwrap();
        assert!(leg.input_scales);
        assert_eq!(leg.chunk, TIER_CHUNK);
        assert_eq!(tier.scratch, leg.bytes(leg.downs).unwrap());
        assert!(matches!(
            PlanInputs::read(&split),
            Err(PlacementError::Unimplemented(_))
        ));
        let one = inputs.prompt_reserve;
        configure_group(&mut inputs, 2).unwrap();
        let grouped = inputs.prompt_reserve;
        assert_eq!(grouped - one, 2 * PROMPT_ROWS as u64 * (8 * 16 + 8));
        configure_group(&mut inputs, 2).unwrap();
        assert_eq!(inputs.prompt_reserve, grouped);
        configure_group(&mut inputs, 1).unwrap();
        assert_eq!(inputs.prompt_reserve, one);
        assert!(configure_group(&mut inputs, 0).is_err());
        assert_eq!(inputs.prompt_reserve, one);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn released_window_geometry_counts_each_resident_sequence() {
        let read = check("place-window-slots", &keys(), &tensors()).unwrap();
        let mut full = read.spec.layers[0].mixer.clone();
        let Mixer::Gqa(g) = &mut full else {
            panic!("GQA");
        };
        g.heads = 32;
        g.kv_heads = 4;
        g.head_dim = 128;
        g.value_dim = 128;
        g.window = None;
        let mut window = full.clone();
        let Mixer::Gqa(g) = &mut window else {
            panic!("GQA");
        };
        g.window = Some(513);
        let mixers: Vec<_> = (0..49)
            .map(|l| if l < 14 { full.clone() } else { window.clone() })
            .collect();
        let kv = KvLayout::of(&mixers).unwrap();
        for ctx in [1, 512, 513, 1023, 1024, 1025, 4096] {
            let expected = 2048 * (14 * ctx + 35 * ctx.min(1024));
            assert_eq!(kv.bytes(0..49, ctx), expected);
            assert_eq!(kv.rows_at(14, ctx), ctx.min(1024));
            assert_eq!(kv.rows_at(0, ctx), ctx);
            for slots in [1, 3] {
                assert_eq!(kv.seq_terms().bytes(ctx, slots), slots * expected);
            }
        }
    }
}

#[cfg(test)]
mod planning_tests {
    use super::super::tests::{check, keys, tensors};
    use super::*;
    use crate::arch::synthetic::{V, host_run_experts};
    use crate::placement::{PlanLevers, Role};

    #[test]
    fn selected_projection_forms_are_a_union_not_blanket_padding() {
        let mut read = check("place-selected-forms", &keys(), &tensors()).unwrap();
        let plain = attention_scratch(&read.spec, &read.tensors).unwrap();
        assert_eq!(plain.input, Forms::default());
        assert_eq!(plain.output, Forms::default());
        for (name, ty) in [
            (names::attn_q(0), gguf::GgmlType::Q4_K),
            (names::attn_k(1), gguf::GgmlType::Q8_0),
            (names::attn_output(2), gguf::GgmlType::Q6_K),
        ] {
            read.tensors
                .tensors
                .iter_mut()
                .find(|t| t.name == name)
                .unwrap()
                .ty = ty;
        }
        let mixed = attention_scratch(&read.spec, &read.tensors).unwrap();
        assert_eq!(
            mixed.input,
            Forms {
                q128: true,
                q32: true
            }
        );
        assert_eq!(
            mixed.output,
            Forms {
                q128: true,
                q32: false
            }
        );
        let rows = PROMPT_ROWS as u64;
        assert_eq!(
            mixed.bytes(rows).unwrap() - plain.bytes(rows).unwrap(),
            mixed.input.bytes(rows, mixed.hidden).unwrap()
                + mixed.output.bytes(rows, mixed.query).unwrap()
        );
    }

    #[test]
    fn nvme_loader_opt_in_preserves_unserved_loader_refusal_and_slot_kv() {
        let mut keys = keys();
        for (key, value) in &mut keys {
            *value = match *key {
                "block_count" => V::U32(8),
                "expert_count" => V::U32(256),
                "expert_used_count" => V::U32(6),
                "attention.sliding_window_pattern" => {
                    V::Bools((0..8).map(|l| l % 2 == 1).collect())
                }
                "attention.rope_pattern" => V::Bools((0..8).map(|l| l % 2 == 0).collect()),
                _ => continue,
            };
        }
        let mut ts = tensors();
        let template: Vec<_> = ts
            .iter()
            .filter(|(n, _)| n.starts_with("blk.1."))
            .cloned()
            .collect();
        for l in 4..8 {
            ts.extend(
                template
                    .iter()
                    .map(|(n, dims)| (n.replacen("blk.1.", &format!("blk.{l}."), 1), dims.clone())),
            );
        }
        for (name, dims) in &mut ts {
            if name.contains("_exps.weight") {
                dims[2] = 256;
            } else if name.ends_with("ffn_gate_inp.weight") {
                dims[1] = 256;
            }
        }
        let read = check("place-paging", &keys, &ts).unwrap();
        let mut inputs = PlanInputs::new("Flint", (), read.tensors, read.spec).unwrap();
        host_run_experts(&mut inputs.model);
        for tensor in &mut inputs.model.tensors {
            if tensor.role == Role::RoutedExperts {
                tensor.dims[1] *= 32_768;
                tensor.file_bytes *= 32_768;
            }
        }
        inputs.prompt_reserve =
            prompt_reserve_bytes(&inputs.spec, &inputs.model, PROMPT_ROWS as u64).unwrap();
        inputs.room = (u64::MAX, workstation::HostRead::Given);
        let machine = workstation::plan_gate(8);
        let levers = PlanLevers::default();
        let (need, held) = {
            let ample = inputs.plan(&machine, 2048, &levers).unwrap();
            (
                workstation::HostNeed::of(&ample, 0).bytes(),
                ample.host.expert_bytes,
            )
        };
        inputs.room.0 = need - held / 8;
        assert!(matches!(inputs.plan(&machine, 2048, &levers),
            Err(PlaceError::NoNvmeTier { paged, .. }) if paged > 0));
        let inputs = inputs.with_nvme_tier();
        for slots in [1, 3] {
            let plan = inputs
                .plan_with_slots(
                    &machine,
                    2048,
                    &levers,
                    std::num::NonZeroUsize::new(slots).unwrap(),
                )
                .unwrap();
            assert!(plan.host.nvme_expert_bytes > 0);
            assert_eq!(plan.host.expert_bytes + plan.host.nvme_expert_bytes, held);
            assert_eq!(
                plan.cards.iter().map(|c| c.kv_bytes).sum::<u64>(),
                slots as u64 * inputs.kv.bytes(0..8, 2048)
            );
        }
    }
}
