//! `f_deep`: the session past one segment of keys, on the d1k set
//! (`fx_ref_qwen35moe_d1k`: a prompt of about a thousand ids, then one
//! decode step, every node dumped, the step set's layout at its depth).
//!
//! The clause opens its own session at the set's depth — the other clauses'
//! session holds one segment of keys — and steps the prompt host-fed one id
//! at a time, then the step token through the traced step, as `--nodes`
//! does. It holds every step's count to `count::token_count` at that step's
//! live keys, both sides of the segment boundary on the real encode (one
//! wait a step, one barrier between groups), the device's free bytes
//! unmoved across the prompt and no host allocation call in a steady step,
//! every layer's step output to [`super::free::FREE_BAND`] over ik's
//! `l_out`, and the last position's argmax by the head's tie rule.

use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use bloomery_device::Device;
use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_metal::arch::qwen3moe::count::{LiveKeys, Ticket, token_count};
use bloomery_metal::arch::qwen3moe::session::{Node, Op, Session, StepMode};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::SEGMENT_KEYS;
use bloomery_metal::library::Library;

use super::flip;
use super::free::FREE_BAND;
use super::says;
use super::sets::{self, Set};
use super::{HOST_ALLOC_CALLS, spread};

/// Whether a token of `n_keys` live keys is past one segment.
fn live_of(n_keys: usize) -> LiveKeys {
    if n_keys <= SEGMENT_KEYS {
        LiveKeys::Within64
    } else {
        LiveKeys::Past64
    }
}

/// One prompt step's count and waits, `d_count`'s rules at the step's own
/// live keys; a steady step allocates nothing on the host.
fn step_counted(s: &mut Session, id: u32, steady: bool) -> Result<Duration, String> {
    let pos = s.pos();
    let want = token_count(s.token_layers(), live_of(pos as usize + 1), Ticket::Folded);
    let (groups, dispatches) = (want.groups as usize, want.dispatches as usize);
    let waits = s.stats().waits;
    let calls = HOST_ALLOC_CALLS.load(Relaxed);
    let t = Instant::now();
    let stepped = s.step(id, StepMode::HostFed).map_err(says)?;
    let wall = t.elapsed();
    let waits = s.stats().waits - waits;
    let calls = HOST_ALLOC_CALLS.load(Relaxed) - calls;
    let c = stepped.count;
    if c.groups != groups || c.dispatches != dispatches || c.barriers + 1 != c.groups {
        return Err(format!(
            "position {pos}: {} groups, {} dispatches, {} barriers encoded; the count is \
             {groups} groups, {dispatches} dispatches, {} barriers",
            c.groups,
            c.dispatches,
            c.barriers,
            groups - 1
        ));
    }
    if waits != 1 {
        return Err(format!(
            "position {pos}: the step waited {waits} times; a step waits once"
        ));
    }
    if steady && calls != 0 {
        return Err(format!(
            "position {pos}: {calls} host allocation calls in the step; a steady step \
             allocates none"
        ));
    }
    Ok(wall)
}

/// `f`: the d1k set's prompt and its step token through the session, every
/// layer's step output and the head's pick against ik's.
pub fn f_deep(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let man = sets::open(Set::D1k)?;
    let (pos, step, prompt) = man
        .step()
        .map_err(|e| format!("the d1k set's tokens: {e}"))?;
    let [tok] = step else {
        return Err(format!(
            "the d1k set holds {} step tokens, the clause runs one",
            step.len()
        ));
    };
    let ctx = pos as usize + step.len();
    let t = Instant::now();
    let mut s = Session::open(device, library, path, ctx).map_err(says)?;
    println!(
        "f_deep: session of {ctx} rows open in {:.1} s, {} prompt tokens then the step token",
        t.elapsed().as_secs_f64(),
        prompt.len()
    );
    let n = s.layer_count();
    let vocab = s.vocab();
    let ik_out: Vec<Vec<f32>> = (0..n)
        .map(|l| sets::tap(&man, &format!("l_out-{l}")))
        .collect::<Result<_, _>>()?;
    let ik_logits = sets::tap(&man, "result_output")?;
    let ik_last = ik_logits
        .len()
        .checked_sub(vocab)
        .map(|at| &ik_logits[at..])
        .ok_or("result_output holds no row")?;

    // The prompt, host-fed: every step's count by its live keys, one wait a
    // step, nothing allocated on either side of a steady step.
    let before = device.info().map_err(why)?.free_bytes;
    let mut walls = Vec::with_capacity(prompt.len());
    for (call, &id) in prompt.iter().enumerate() {
        walls.push(step_counted(&mut s, id, call > 0)?);
    }
    let after = device.info().map_err(why)?.free_bytes;
    if after != before {
        return Err(format!(
            "the device's free bytes moved from {before} to {after} across {} prompt steps: a \
             step allocated",
            prompt.len()
        ));
    }
    let (mid, p90) = spread(walls[1..].to_vec());
    println!(
        "f_deep: step wall per token, {} steady prompt steps: median {mid:.3} ms, p90 {p90:.3} ms \
         (development-loop number)",
        walls.len() - 1
    );

    // The step token, traced: every layer's output and the head's logits.
    let mut outs: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    let mut logits = Vec::new();
    let t = Instant::now();
    let want = token_count(s.token_layers(), LiveKeys::Past64, Ticket::Folded);
    let stepped = s
        .step_traced(*tok, |op, probe| {
            match op {
                Op::Experts(l) => outs[l] = Some(probe.read(Node::Residual)?),
                Op::Head => logits = probe.read(Node::Logits)?,
                _ => {}
            }
            Ok(())
        })
        .map_err(says)?;
    let traced = t.elapsed();
    let c = stepped.count;
    if c.groups != want.groups as usize || c.dispatches != want.dispatches as usize {
        return Err(format!(
            "the traced step at {pos}: {} groups, {} dispatches encoded; the count is {} \
             groups, {} dispatches",
            c.groups, c.dispatches, want.groups, want.dispatches
        ));
    }
    let top = stepped.id;
    println!(
        "f_deep: traced step at position {pos} in {:.3} ms: {} groups, {} dispatches (band \
         counts, the wall a development-loop number)",
        traced.as_secs_f64() * 1e3,
        c.groups,
        c.dispatches
    );

    // Every layer's step output over ik's row.
    let mut worst = 0.0f64;
    let mut head_in = 0.0f64;
    for (l, ik) in ik_out.iter().enumerate() {
        let ours = outs[l]
            .take()
            .ok_or_else(|| format!("layer {l}: the traced step left no output"))?;
        if ik.len() != ours.len() {
            return Err(format!(
                "l_out-{l}: {} values, the layer's output is {}",
                ik.len(),
                ours.len()
            ));
        }
        let e = flip::rel_to(&ours, ik);
        if l % 8 == 0 || l == n - 1 || e > FREE_BAND {
            println!("f_deep layer={l} l_out_rel={e:.3e}");
        }
        worst = flip::worse(worst, e);
        head_in = e;
    }
    let (gap, cap) = (
        flip::head_gap(ik_last, top),
        flip::head_cap(head_in, ik_last),
    );
    let argmax_ok = head_in.is_finite() && flip::head_tie(ik_last, top, head_in);
    let ik_top = flip::argmax(ik_last);
    let line = format!(
        "the d1k step at position {pos} after {} prompt tokens: worst l_out_rel={worst:.3e} \
         (band {FREE_BAND:.2}); last position argmax ours={top} ik={ik_top} gap {gap:.3} cap \
         {cap:.3} at head_in {head_in:.4} ({}), logits_rel={:.3e}; every step's count by its \
         live keys, one wait a step, device allocation unmoved, no host allocation call in a \
         steady step",
        prompt.len(),
        if argmax_ok {
            "tie band"
        } else {
            "past the cap"
        },
        flip::rel_to(&logits, ik_last),
    );
    if worst <= FREE_BAND && argmax_ok {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}
