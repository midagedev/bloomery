//! `p_profile`: the profile step (`Session::step_profile`) held to the step
//! it instruments, on the fixture.
//!
//! The clause opens its own session — rows past `SEGMENT_KEYS`, the other
//! clauses' session holds that many — and walks it over the same ids
//! host-fed and profiled, on both sides of that row. Both sides run the
//! one-pass flash (one segment up to `ONE_PASS_KEYS`); the split path and its
//! merge are held by the kernel gate's forced-split cases:
//!
//! - every pick and the last logits equal bit for bit, and every profiled
//!   step's groups, dispatches and barriers are the plain step's own count,
//!   `count::token_count` at that step's live keys;
//! - one sample a group, and the samples' kinds cover every group the count
//!   counts, none twice: the per-kind counts the walk left against the
//!   kinds the file's layer plans derive (a sandwich layer's two joins and a
//!   short conv's three groups among them);
//! - the profiled groups' weight bytes sum to the file's own tensors walked
//!   under the plan's names (the embedding's one row, each routed pick's
//!   slice of its stacks, every other weight whole, a tied head the
//!   embedding's table) — two paths to one number;
//! - a steady profiled step allocates nothing on the host.

use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_metal::arch::qwen3moe::count::{
    GroupKind, KINDS, Layer, LiveKeys, Residual, Ticket, token_count,
};
use bloomery_metal::arch::qwen3moe::session::{Node, ProfileGroup, Session, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::EncodeCount;
use bloomery_metal::kernels::SEGMENT_KEYS;
use bloomery_metal::library::Library;
use chain::arch::qwen3moe::plan::{FfnRoute, LayerPlan, MixerPlan, ResidualPlan};
use gguf::Split;
use model_hf::ModelFile;

use super::says;
use super::{HOST_ALLOC_CALLS, seed};

const WHAT: &str = "gate_metal_qwen35moe_e2e p_profile";

/// The tokens the clause profiles within one segment of keys.
const WITHIN: usize = 3;

/// The session's rows past the segment: the token at the segment's last row
/// is profiled past one segment of keys.
const PAST_ROWS: usize = 4;

/// The session's rows the clause opens: one segment, then rows past it. The
/// token at `SEGMENT_KEYS` (the first past the segment) is the deep one
/// profiled.
const CTX: usize = SEGMENT_KEYS + PAST_ROWS;

/// Whether a token of `n_keys` live keys is past one segment.
fn live_of(n_keys: usize) -> LiveKeys {
    if n_keys <= SEGMENT_KEYS {
        LiveKeys::Within64
    } else {
        LiveKeys::Past64
    }
}

/// The kinds one token's groups are tagged with, as the file's own layer plans
/// derive them: a mixer's three groups (P1, core, out), a join after the mixer
/// and another after the FFN where the layer's residual is a sandwich, or a
/// hand-on after each where the session's layer hands its plain residual on
/// (`layers`, the session's own reading: the plans hold no reader types), the
/// router where the FFN routes, then the gate-up and the down, and at the end
/// the head's logits and its pick. The count model's own derivation is the
/// walk's, so the two agree only while both are right.
fn kinds_expected(plans: &[LayerPlan], layers: &[Layer]) -> Result<Vec<GroupKind>, String> {
    if plans.len() != layers.len() {
        return Err(format!(
            "{} plans against the session's {} layers",
            plans.len(),
            layers.len()
        ));
    }
    let mut out = vec![GroupKind::Embed];
    for (plan, layer) in plans.iter().zip(layers) {
        let joins = match (&plan.residual, layer.residual()) {
            (ResidualPlan::Sandwich(_), Residual::Sandwich) => Some(GroupKind::Join),
            (ResidualPlan::Plain, Residual::PlainHandOn) => Some(GroupKind::HandOn),
            (ResidualPlan::Plain, Residual::Plain) => None,
            (plan, layer) => {
                let plan = match plan {
                    ResidualPlan::Plain => "plain",
                    ResidualPlan::Sandwich(_) => "a sandwich",
                };
                return Err(format!(
                    "the plan's residual is {plan}, the session's layer {layer:?}"
                ));
            }
        };
        match &plan.mixer {
            MixerPlan::Gqa(_) => {
                out.push(GroupKind::AttnP1);
                out.push(GroupKind::AttnFlash);
                out.push(GroupKind::AttnOut);
            }
            MixerPlan::Delta(_) => {
                out.push(GroupKind::DeltaP1);
                out.push(GroupKind::DeltaCore);
                out.push(GroupKind::DeltaOut);
            }
            MixerPlan::ShortConv(_) => {
                out.push(GroupKind::ConvP1);
                out.push(GroupKind::ConvCore);
                out.push(GroupKind::ConvOut);
            }
        }
        out.extend(joins);
        if matches!(plan.ffn.route, FfnRoute::Router { .. }) {
            out.push(GroupKind::Router);
        }
        out.push(GroupKind::GateUp);
        out.push(GroupKind::Down);
        out.extend(joins);
    }
    out.push(GroupKind::Head);
    out.push(GroupKind::HeadPick);
    Ok(out)
}

/// The samples' kinds against the plans' derivation: every kind's count
/// equal, so no group is untagged and none counted twice.
fn tags_of(samples: &[ProfileGroup], expected: &[GroupKind]) -> Result<(), String> {
    for kind in KINDS {
        let saw = samples.iter().filter(|g| g.kind == kind).count();
        let want = expected.iter().filter(|k| **k == kind).count();
        if saw != want {
            return Err(format!(
                "{kind:?}: {saw} groups tagged, the plans derive {want}"
            ));
        }
    }
    Ok(())
}

/// One profiled step's own checks: its count the count model's, one sample
/// a group, the tags covering every group, and (past the first) no host
/// allocation call in the step. The pick is returned.
fn step_checked(
    s: &mut Session,
    id: u32,
    samples: &mut Vec<ProfileGroup>,
    steady: bool,
    expected: &[GroupKind],
) -> Result<u32, String> {
    let pos = s.pos();
    let live = live_of(pos as usize + 1);
    let want = token_count(s.token_layers(), live, Ticket::Folded);
    let calls = HOST_ALLOC_CALLS.load(Relaxed);
    let stepped = s.step_profile(id, samples).map_err(says)?;
    let calls = HOST_ALLOC_CALLS.load(Relaxed) - calls;
    let c = stepped.count;
    if c.groups != want.groups as usize
        || c.dispatches != want.dispatches as usize
        || c.barriers + 1 != c.groups
    {
        return Err(format!(
            "the profiled step at {pos}: {} groups, {} dispatches, {} barriers encoded; the \
             count is {} groups, {} dispatches, {} barriers",
            c.groups,
            c.dispatches,
            c.barriers,
            want.groups,
            want.dispatches,
            want.groups - 1
        ));
    }
    if samples.len() != c.groups {
        return Err(format!(
            "the profiled step at {pos}: {} samples for {} groups; one sample a group",
            samples.len(),
            c.groups
        ));
    }
    tags_of(samples, expected)?;
    if steady && calls != 0 {
        return Err(format!(
            "the profiled step at {pos}: {calls} host allocation calls; a steady step allocates \
             none"
        ));
    }
    Ok(stepped.id)
}

/// Every layer's router ids as the step left them, a layer an entry.
fn picks_of(s: &Session) -> Result<Vec<Vec<u32>>, String> {
    let probe = s.probe().map_err(says)?;
    (0..s.layer_count())
        .map(|l| probe.router_ids(l).map_err(says))
        .collect()
}

/// The profiled groups' weight bytes, the profile's own path: the arena's
/// byte rule over every sample of the token at `pos`, a layer an entry and
/// the embed's and the head's the last (so the entry count is the layer
/// count plus one).
fn weights_of_token(s: &Session, samples: &[ProfileGroup], pos: u32) -> Result<Vec<u64>, String> {
    let ids = picks_of(s)?;
    let mut out = vec![0u64; s.layer_count() + 1];
    for g in samples {
        let none: &[u32] = &[];
        let routed = match g.kind {
            GroupKind::GateUp | GroupKind::Down => &ids[g.layer.unwrap_or(0)],
            _ => none,
        };
        let at = g.layer.unwrap_or(s.layer_count());
        out[at] += s.group_bytes(g, pos, routed).map_err(says)?.weights;
    }
    Ok(out)
}

/// Two per-layer weight sums against one another: equal — their total comes
/// back — or every layer they differ at named, the last entry the embed's
/// and the head's.
fn weights_agree(ours: &[u64], file: &[u64]) -> Result<u64, String> {
    if ours.len() == file.len() && ours == file {
        return Ok(ours.iter().sum());
    }
    let mut line = String::new();
    for (l, (a, b)) in ours.iter().zip(file).enumerate() {
        if a != b {
            line.push_str(&format!(" layer {l}: {a} against {b},"));
        }
    }
    Err(format!(
        "the profiled groups' weight bytes differ from the file's own tensors:{} total {}          against {}",
        line,
        ours.iter().sum::<u64>(),
        file.iter().sum::<u64>()
    ))
}

/// `whole` cut into `parts` whole ones, or the named refusal: nothing the
/// file holds is silently rounded away.
fn whole_of(whole: u64, parts: u64, what: &str) -> Result<u64, String> {
    let one = whole / parts;
    if one * parts != whole {
        return Err(format!(
            "{what}: {whole} bytes cut into {parts} parts leaves a remainder"
        ));
    }
    Ok(one)
}

/// The file's layer plans, the second path's names: the plan the session
/// reads, rebuilt from the header alone, and the width of the residual.
fn plans_of(split: &Split) -> Result<(Vec<LayerPlan>, u64), String> {
    let read = model::arch::spec(split).map_err(|e| e.to_string())?;
    let spec = read.spec;
    let kinds = chain::arch::qwen3moe::plan::kinds_of(&spec.layers).map_err(|e| e.to_string())?;
    let pd = chain::arch::qwen3moe::read::PlanDims::of(&spec, WHAT).map_err(|e| e.to_string())?;
    let plans = chain::arch::qwen3moe::read::layer_plans(
        split,
        WHAT,
        &pd,
        &kinds,
        &spec.layers,
        &bloomery_metal::arch::qwen3moe::session::SITES,
        chain::arch::qwen3moe::read::Read35::Whole,
    )
    .map_err(|e| e.to_string())?;
    Ok((plans, u64::from(spec.hidden)))
}

/// The file's own weight bytes for one token, the second path: the plan's
/// names walked in the file's header — the embedding's one row, each routed
/// pick's slice of its stacks, every other weight whole, the sandwich norms
/// and a head gate among them. A tied head is the embedding's table read
/// whole. `picks` is every layer's router ids as the step left them.
fn file_weights(
    split: &Split,
    plans: &[LayerPlan],
    hidden: u64,
    vocab: usize,
    picks: &[Vec<u32>],
) -> Result<Vec<u64>, String> {
    let bytes = |name: &str| -> Result<u64, String> {
        let (_, info) = split
            .find(name)
            .ok_or_else(|| format!("the file has no tensor {name}"))?;
        Ok(info.nbytes)
    };
    // A value tensor the arena dequantizes at open: F32 as stored, MLX_A8 or
    // MLX_A8H to one f32 an element. The profile counts that buffer, so the
    // packed file length is the same number only for F32.
    let values_held = |name: &str| -> Result<u64, String> {
        let (_, info) = split
            .find(name)
            .ok_or_else(|| format!("the file has no tensor {name}"))?;
        let elems = info.dims.iter().product::<u64>();
        match info.ty {
            gguf::GgmlType::F32 | gguf::GgmlType::MLX_A8 | gguf::GgmlType::MLX_A8H => elems
                .checked_mul(4)
                .ok_or_else(|| format!("{name}: its f32 bytes do not fit u64")),
            other => Err(format!(
                "{name} is {other:?}; the profile holds F32, MLX_A8 or MLX_A8H"
            )),
        }
    };
    if plans.len() != picks.len() {
        return Err(format!(
            "{} plans for {} layers' picks",
            plans.len(),
            picks.len()
        ));
    }
    let vocab = u64::try_from(vocab).map_err(|_| "the vocabulary does not fit u64")?;
    // The embedding's row; the head whole. A layer an entry, the embed's
    // and the head's the last.
    let mut per = vec![0u64; plans.len() + 1];
    let head = per.len() - 1;
    let embed = model::arch::names::token_embd();
    per[head] = whole_of(bytes(&embed)?, vocab, "the embedding")?;
    let output = model::arch::names::output();
    per[head] += bytes(if split.find(&output).is_some() {
        &output
    } else {
        &embed
    })?;
    per[head] += bytes(&model::arch::names::output_norm())?;
    for (l, (plan, pick)) in plans.iter().zip(picks).enumerate() {
        let total = &mut per[l];
        match &plan.mixer {
            MixerPlan::Gqa(g) => {
                *total += bytes(&g.attn_norm)?;
                *total += bytes(&g.attn_q)?;
                *total += bytes(&g.attn_k)?;
                *total += bytes(&g.attn_v)?;
                *total += bytes(&g.attn_q_norm)?;
                *total += bytes(&g.attn_k_norm)?;
                *total += bytes(&g.attn_output)?;
                if let Some(gate) = &g.gate {
                    *total += bytes(&gate.name)?;
                }
            }
            MixerPlan::Delta(d) => {
                *total += bytes(&d.attn_norm)?;
                *total += bytes(&d.qkv)?;
                *total += bytes(&d.gate)?;
                *total += bytes(&d.beta)?;
                *total += bytes(&d.alpha)?;
                *total += bytes(&d.conv)?;
                *total += bytes(&d.dt_bias)?;
                *total += bytes(&d.ssm_a)?;
                *total += bytes(&d.ssm_norm)?;
                *total += bytes(&d.ssm_out)?;
            }
            MixerPlan::ShortConv(c) => {
                *total += bytes(&c.attn_norm)?;
                *total += bytes(&c.conv_in)?;
                *total += bytes(&c.conv)?;
                *total += bytes(&c.conv_out)?;
            }
        }
        if let ResidualPlan::Sandwich(sw) = &plan.residual {
            for name in &sw.norms {
                *total += bytes(name)?;
            }
        }
        match &plan.ffn.route {
            FfnRoute::Dense => {
                *total += bytes(&plan.ffn.ffn_norm)?
                    + bytes(&plan.ffn.gate)?
                    + bytes(&plan.ffn.up)?
                    + bytes(&plan.ffn.down)?;
            }
            FfnRoute::Router { shared, .. } => {
                let shared = shared.is_some();
                let f = chain::arch::qwen3moe::read::file_names(l);
                // The router's rows, from its shape. Its stored bytes are that
                // count only while every value is f32; an MLX 8-bit router is
                // not, and the profile counts the f32 buffer the open
                // dequantized.
                let (_, router_info) = split
                    .find(&f.router)
                    .ok_or_else(|| format!("the file has no tensor {}", f.router))?;
                let elems = router_info.dims.iter().product::<u64>();
                if hidden == 0 || elems % hidden != 0 {
                    return Err(format!(
                        "layer {l}: the router {} has shape {:?} which is not a whole number \
                         of hidden rows ({hidden})",
                        f.router, router_info.dims
                    ));
                }
                let experts = elems / hidden;
                let router = values_held(&f.router)?;
                // The layer's last slot reads the shared expert when the plan
                // has one.
                if experts == 0 || pick.len() <= usize::from(shared) {
                    return Err(format!(
                        "layer {l}: the router holds {} rows and the token picked {} ids",
                        experts,
                        pick.len()
                    ));
                }
                let picks = (pick.len() - usize::from(shared)) as u64;
                *total += bytes(&plan.ffn.ffn_norm)? + router;
                *total += picks
                    * (whole_of(bytes(&f.gate)?, experts, "a gate stack")?
                        + whole_of(bytes(&f.up)?, experts, "an up stack")?);
                *total += picks * whole_of(bytes(&f.down)?, experts, "a down stack")?;
                if shared {
                    *total += values_held(&f.router_gate)?
                        + bytes(&f.sh_gate)?
                        + bytes(&f.sh_up)?
                        + bytes(&f.sh_down)?;
                }
            }
        }
    }
    Ok(per)
}

/// Two logits rows equal bit for bit.
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// One step host-fed, its count checked against the count model (the
/// profiled step's count is held to this same one); the pick and the count
/// are returned.
fn plain_checked(s: &mut Session, id: u32) -> Result<(u32, EncodeCount), String> {
    let pos = s.pos();
    let want = token_count(s.token_layers(), live_of(pos as usize + 1), Ticket::Folded);
    let stepped = s.step(id, StepMode::HostFed).map_err(says)?;
    let c = stepped.count;
    if c.groups != want.groups as usize || c.dispatches != want.dispatches as usize {
        return Err(format!(
            "the host-fed step at {pos}: {} groups, {} dispatches; the count is {}, {}",
            c.groups, c.dispatches, want.groups, want.dispatches
        ));
    }
    Ok((stepped.id, c))
}

/// `p`: the profiled step against the step it instruments, within one
/// segment of keys and past it.
pub fn p_profile(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let file = ModelFile::open(path).map_err(|e| e.to_string())?;
    let split = file.header();
    let (plans, hidden) = plans_of(split)?;
    let mut s = Session::open(device, library, path, CTX).map_err(says)?;
    let expected = kinds_expected(&plans, s.token_layers())?;
    // The ids both arms feed, `ids[k]` the token at position `k`: the walk
    // is deterministic, so a replay of a prefix leaves the same state.
    let mut ids = vec![seed()];
    let mut plain_counts: Vec<EncodeCount> = Vec::with_capacity(CTX);
    s.reset().map_err(says)?;
    for _ in 0..WITHIN {
        let id = *ids.last().ok_or("no id to step")?;
        let (pick, count) = plain_checked(&mut s, id)?;
        ids.push(pick);
        plain_counts.push(count);
    }
    let logits_plain = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    s.reset().map_err(says)?;
    let mut samples: Vec<ProfileGroup> = Vec::new();
    for k in 0..WITHIN {
        let pick = step_checked(&mut s, ids[k], &mut samples, k > 0, &expected)?;
        if pick != ids[k + 1] {
            return Err(format!(
                "the profiled step {k} picked {pick}; the host-fed step picked {}",
                ids[k + 1]
            ));
        }
        if samples.len() != plain_counts[k].groups {
            return Err(format!(
                "the profiled step {k}: {} samples; the host-fed step encoded {} groups",
                samples.len(),
                plain_counts[k].groups
            ));
        }
    }
    let logits_profiled = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    if !bits_equal(&logits_plain, &logits_profiled) {
        return Err(format!(
            "the profiled run's last logits differ from the host-fed run's bit for bit ({} \
             values)",
            logits_plain.len()
        ));
    }
    // The profiled groups' weight bytes against the file's own tensors, for
    // the arm's last token.
    let pos_last = s.pos() - 1;
    let gpu_within = samples.iter().map(|g| g.gpu_s).sum::<f64>() * 1e3;
    let weights = weights_agree(
        &weights_of_token(&s, &samples, pos_last)?,
        &file_weights(split, &plans, hidden, s.vocab(), &picks_of(&s)?)?,
    )?;

    // Past one segment: the host-fed step at the segment's last row, then
    // the same token profiled over a replayed prefix.
    let deep = SEGMENT_KEYS;
    while (s.pos() as usize) < deep {
        let id = *ids.last().ok_or("no id to step")?;
        let (pick, _) = plain_checked(&mut s, id)?;
        ids.push(pick);
    }
    let t = Instant::now();
    let id = *ids.last().ok_or("no id to step")?;
    let (pick_deep, _) = plain_checked(&mut s, id)?;
    let wall_deep = t.elapsed();
    ids.push(pick_deep);
    let logits_deep = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    s.reset().map_err(says)?;
    for id in &ids[..deep] {
        plain_checked(&mut s, *id)?;
    }
    let pick_profiled = step_checked(&mut s, ids[deep], &mut samples, true, &expected)?;
    let gpu_deep = samples.iter().map(|g| g.gpu_s).sum::<f64>() * 1e3;
    if pick_profiled != pick_deep {
        return Err(format!(
            "the profiled step at {deep} picked {pick_profiled}; the host-fed step picked \
             {pick_deep}"
        ));
    }
    let logits_deep_profiled = s.probe().map_err(says)?.read(Node::Logits).map_err(says)?;
    if !bits_equal(&logits_deep, &logits_deep_profiled) {
        return Err("the deep profiled step's logits differ bit for bit".into());
    }
    let weights_deep = weights_agree(
        &weights_of_token(&s, &samples, deep as u32)?,
        &file_weights(split, &plans, hidden, s.vocab(), &picks_of(&s)?)?,
    )?;
    let flashes = samples
        .iter()
        .filter(|g| g.kind == GroupKind::AttnFlash)
        .count();
    let attns = s
        .token_layers()
        .iter()
        .filter(|l| matches!(l, Layer::Attention { .. }))
        .count();
    if flashes != attns {
        return Err(format!(
            "the deep profiled step tagged {flashes} flash groups; the session holds {attns} \
             attention layers"
        ));
    }
    println!(
        "p_profile: the step at {deep}: {} groups, their GPU sum {gpu_deep:.3} ms; the host-fed \
         step's wall {:.3} ms (development-loop numbers)",
        samples.len(),
        wall_deep.as_secs_f64() * 1e3,
    );
    Ok(Verdict::Noted(format!(
        "{WITHIN} tokens within a segment and the token at {deep} past it, host-fed and \
         profiled over the same ids: picks and last logits equal bit for bit; every step's \
         count the count model's, one sample a group, the tags covering every group, {flashes} \
         flashes among them; the groups' weight bytes {weights} (within) and {weights_deep} \
         (deep) equal the file's own walk; a steady profiled step allocated nothing; the \
         groups' GPU sum {:.3} ms a token within, {:.3} ms deep",
        gpu_within, gpu_deep,
    )))
}
