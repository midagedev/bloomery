//! The prefill gates' cache fixtures, aligned-window equivalence and named refusals.

use crate::{GateError, bits_equal, verdict};
use bloomery_gpu::flash_gqa_prefill::GqaPrefillArgs;
use bloomery_gpu::model::GpuModel;
use bloomery_gpu::{Gpu, GpuError, Window};
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::prompt::CallEnd;

/// Host values of the f16 cache planes.
pub struct HostCache {
    pub kf: Vec<f32>,
    pub vf: Vec<f32>,
    pub ctx: usize,
}

/// Seeded planes, padded twins and host values.
pub struct Cache {
    pub kc: DeviceBuffer<u16>,
    pub vc: DeviceBuffer<u16>,
    pub kn: DeviceBuffer<u16>,
    pub vn: DeviceBuffer<u16>,
    pub host: HostCache,
    pub kb: Vec<u16>,
    pub vb: Vec<u16>,
}

/// Build the seeded f16 cache and its NaN-padded twin.
pub fn cache(
    stream: &CudaStream,
    head: usize,
    n_kv: usize,
    ctx: usize,
    live: usize,
    seed: u32,
) -> Result<Cache, GateError> {
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    let plane = |s| {
        crate::activations(head, n_kv * ctx, s)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect::<Vec<_>>()
    };
    let (kb, vb) = (plane(seed), plane(seed + 1));
    let pad = |b: &[u16]| {
        b.iter()
            .enumerate()
            .map(|(i, &v)| {
                if (i / head) % ctx >= live {
                    crate::NAN_F16
                } else {
                    v
                }
            })
            .collect::<Vec<_>>()
    };
    let widen = |b: &[u16]| b.iter().copied().map(half_to_f32).collect();
    Ok(Cache {
        kc: DeviceBuffer::from_host(stream, &kb)?,
        vc: DeviceBuffer::from_host(stream, &vb)?,
        kn: DeviceBuffer::from_host(stream, &pad(&kb))?,
        vn: DeviceBuffer::from_host(stream, &pad(&vb))?,
        host: HostCache {
            kf: widen(&kb),
            vf: widen(&vb),
            ctx,
        },
        kb,
        vb,
    })
}

/// The fault launch's host inputs and plain output, retained for its window check.
pub struct Windows {
    pub q: Vec<f32>,
    pub kb: Vec<u16>,
    pub vb: Vec<u16>,
    pub clean: Vec<u32>,
    pub y: Vec<f32>,
    pub ctx: usize,
}

/// Host inputs and the plain output of one prefill launch.
pub struct WindowCase<'a> {
    pub q: &'a [f32],
    pub planes: (&'a [u16], &'a [u16]),
    pub counts: &'a [u32],
    pub y: &'a [f32],
    pub heads: (usize, usize),
    pub ctx: usize,
    pub scale: f32,
}

/// Accept aligned windows with the plain bits and refuse misaligned ones by name.
/// Run last: a missing refusal can leave a sticky device error.
pub fn window_refusals(
    gpu: &Gpu,
    what: &str,
    label: &str,
    case: WindowCase<'_>,
    enqueue: impl Fn(&CudaStream, GqaPrefillArgs<'_>) -> Result<(), GpuError>,
) -> Result<bool, GateError> {
    let WindowCase {
        q,
        planes: (kb, vb),
        counts,
        y,
        heads: (n_head, n_kv),
        ctx,
        scale,
    } = case;
    let stream = gpu.stream();
    let t = counts.len();
    let qd = DeviceBuffer::from_host(stream, q)?;
    let kc = DeviceBuffer::from_host(stream, kb)?;
    let vc = DeviceBuffer::from_host(stream, vb)?;
    let nk = DeviceBuffer::from_host(stream, counts)?;
    let q_pad = DeviceBuffer::from_host(stream, &[&[0.0f32; 2][..], q].concat())?;
    let k_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], kb].concat())?;
    let v_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], vb].concat())?;
    let q_al = Window::<f32>::of(&q_pad, 8, q.len())?;
    let q_mis = Window::<f32>::of(&q_pad, 4, q.len())?;
    let k_al = Window::<u16>::of(&k_pad, 16, kb.len())?;
    let k_mis = Window::<u16>::of(&k_pad, 8, kb.len())?;
    let v_al = Window::<u16>::of(&v_pad, 16, vb.len())?;
    let v_mis = Window::<u16>::of(&v_pad, 8, vb.len())?;
    let mut ya = DeviceBuffer::from_host(stream, &vec![f32::NAN; y.len()])?;
    enqueue(
        stream,
        GqaPrefillArgs {
            q: &q_al,
            kc: &k_al,
            vc: &v_al,
            n_keys: &nk,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault: gpu.unlabelled_sink(),
            y: &mut ya,
        },
    )?;
    stream.synchronize()?;
    let aligned_ok = bits_equal(&ya.to_host_vec(stream)?, y);
    println!(
        "{label} windows at aligned offsets: accepted, bit-identical to the plain launch {aligned_ok} {}",
        verdict(aligned_ok)
    );
    let mut ok = aligned_ok;
    let mut ym = DeviceBuffer::<f32>::zeroed(stream, y.len())?;
    let cases = [
        ("q", "4 bytes past an 8-byte boundary", &*q_mis, &kc, &vc),
        ("kc", "8 bytes past a 16-byte boundary", &qd, &*k_mis, &vc),
        ("vc", "8 bytes past a 16-byte boundary", &qd, &kc, &*v_mis),
    ];
    for (name, off, q, kc, vc) in cases {
        let r = enqueue(
            stream,
            GqaPrefillArgs {
                q,
                kc,
                vc,
                n_keys: &nk,
                scale,
                n_head,
                n_kv,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut ym,
            },
        );
        let named = matches!(r, Err(GpuError::Shape { what: seen, ref detail })
            if seen == what && detail.starts_with(&format!("{name} at ")));
        let seen = match &r {
            _ if named => "refused by name".to_string(),
            Ok(()) => format!(
                "accepted; the stream then reads {:?}",
                stream.synchronize().err().map(|e| e.to_string())
            ),
            Err(e) => format!("refused without the name: {e}"),
        };
        println!(
            "{label} refusal {name} window {off}: {seen} {}",
            verdict(named)
        );
        ok &= named;
    }
    Ok(ok)
}

/// The argmax after a prompt call, the logits' bits, and three greedy
/// steps' ids and logits' bits.
pub type Closed = (u32, Vec<u32>, Vec<(u32, Vec<u32>)>);

/// [`Closed`] of `ids` fed from where `m` stands by its body's prompt
/// schedule ([`app::Prompt::prompt_call`]), ending as `end` asks.
pub fn closed_run<B: app::Prompt>(
    m: &mut GpuModel<B>,
    ids: &[u32],
    end: CallEnd,
) -> Result<Closed, GateError> {
    let bits = |m: &GpuModel<B>| -> Result<Vec<u32>, GateError> {
        Ok(m.logits()?.iter().map(|v| v.to_bits()).collect())
    };
    let first = B::prompt_call(m, ids, end)?.ok_or("the call read no token")?;
    let row = bits(m)?;
    let (mut steps, mut feed) = (Vec::new(), first);
    for _ in 0..3 {
        feed = m.step(&[feed])?;
        steps.push((feed, bits(m)?));
    }
    Ok((first, row, steps))
}
