//! The probe's routed mode (`--routed`): the experts' grouped GEMM over a
//! route table, the pair and the down timed apart, each one dispatch a
//! buffer as the dense arms are.
//!
//! A configuration is a pass of U rows (`--rows U`, repeatable; 512 and 2048
//! by default) whose picks are drawn on the host: each row picks eight
//! distinct experts of 256 with probability in proportion to
//! `1 / (1 + rank)^S` over a fixed random ranking (`--skew S`, repeatable;
//! 0, uniform, by default), then the shared expert. The arm's own
//! `route_table` sorts the slots on the device, so a variant that cuts the
//! table another way is timed over its own tiles. The weights are A4, the
//! model's routed shapes (gate and up 512 by 2048, down 2048 by 512, 256
//! experts and the shared tensors), one copy: a layer's experts are past
//! any cache, so a repeated dispatch reads its weights from memory as the
//! pass does.
//!
//! The arms (`--arm NAME`, repeatable; `ship` by default): `ship`, the
//! shipped source, and every routed variant (`variants::ROUTED`), each
//! launched with the engine's bindings and argument block. `--hot` sets the
//! stack's per-expert stride to zero, so every group reads expert 0's rows:
//! the same instructions over cache-resident weights.
//!
//! One line a launch:
//!
//! ```text
//! routed op=… u=… skew=… hot=… kind=pair|down tiles=T blocks=L ms=… tflops=… wall_ms=…
//! ```
//!
//! and a `kind=layer` line with the two summed. `tiles` is the table's tile
//! count and `blocks` its live 8-row blocks (the multiplies the tiles do).
//! `--hash` prints a hash of the pair's and the down's outputs in slot
//! order, so two arms that cut the table differently can be seen to write
//! the same bits.

use bloomery_device::conformance::why;
use bloomery_device::{Buffer, Device, Elems, LAYER_NONE, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::{Binding, Bindings, ComputeEncoder};
use bloomery_metal::fault::MetalFaultWord;
use bloomery_metal::kernels::{EXPERT_SLOTS, GEMM_THREADS, Quant, ROUTE_TILE};
use bloomery_metal::library::SOURCE;
use bloomery_metal::{Library, MetalBuffer, Pipeline};

use super::oracle::Rng;
use super::variants::{self, Variant};
use super::{Ctx, Timed, oracle, synthetic, time_dispatches};

/// The routed experts and the shared group after them.
const EXPERTS: usize = 256;
/// Routed picks a row keeps.
const TOP_K: usize = 8;
/// The hidden width.
const HIDDEN: usize = 2048;
/// An expert's outputs of the gate (and of the up).
const EXPERT_K: usize = 512;
/// The weight rows of a routed tile (`ROUTED_ROWS`).
const TILE_ROWS: usize = 64;
/// The route table's threads (`ROUTE_THREADS`).
const TABLE_THREADS: usize = 1024;
/// Distinct synthetic experts the stack cycles through.
const DISTINCT: usize = 4;

/// The routed mode's own flags.
pub struct Opts {
    pub rows: Vec<usize>,
    pub skews: Vec<f64>,
    pub arms: Vec<String>,
    pub hot: bool,
}

/// An arm: its name, its library, the table's tile height.
struct RArm<'l> {
    name: String,
    library: &'l Library,
    tile: usize,
}

/// The weights of a layer: the stacks and the shared tensors, uploaded once.
struct Weights {
    gate: MetalBuffer<u8>,
    up: MetalBuffer<u8>,
    down: MetalBuffer<u8>,
    sgate: MetalBuffer<u8>,
    sup: MetalBuffer<u8>,
    sdown: MetalBuffer<u8>,
}

fn stack(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let distinct: Vec<Vec<u8>> = (0..DISTINCT)
        .map(|d| synthetic::matrix(Quant::A4, rows, cols, seed ^ ((d as u64) << 32)))
        .collect();
    let mut out = Vec::with_capacity(EXPERTS * distinct[0].len());
    for e in 0..EXPERTS {
        out.extend_from_slice(&distinct[e % DISTINCT]);
    }
    out
}

fn weights(ctx: &Ctx<'_>) -> Result<Weights, String> {
    let up = |bytes: Vec<u8>| ctx.device.upload(ctx.stream, &bytes).map_err(why);
    Ok(Weights {
        gate: up(stack(EXPERT_K, HIDDEN, 0x9A7E))?,
        up: up(stack(EXPERT_K, HIDDEN, 0x0B0B))?,
        down: up(stack(HIDDEN, EXPERT_K, 0xD0D0))?,
        sgate: up(synthetic::matrix(Quant::A4, EXPERT_K, HIDDEN, 0x5A7E))?,
        sup: up(synthetic::matrix(Quant::A4, EXPERT_K, HIDDEN, 0x50B0))?,
        sdown: up(synthetic::matrix(Quant::A4, HIDDEN, EXPERT_K, 0x5D0D))?,
    })
}

/// The ids of `rows` rows: eight distinct picks a row by the skewed weights,
/// then the shared id.
fn picks(rows: usize, skew: f64, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    // A fixed random ranking of the experts, so the hot ones are not the
    // low ids.
    let mut rank: Vec<usize> = (0..EXPERTS).collect();
    for i in (1..EXPERTS).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        rank.swap(i, j);
    }
    let weight: Vec<f64> = rank
        .iter()
        .map(|&r| 1.0 / (1.0 + r as f64).powf(skew))
        .collect();
    let mut cdf = Vec::with_capacity(EXPERTS);
    let mut acc = 0.0;
    for w in &weight {
        acc += w;
        cdf.push(acc);
    }
    let mut ids = Vec::with_capacity(rows * EXPERT_SLOTS);
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * acc;
            let e = cdf.partition_point(|&c| c <= u).min(EXPERTS - 1) as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
        ids.push(EXPERTS as u32);
    }
    ids
}

/// The grouped GEMM's argument block (`RoutedArgs`).
fn routed_args(cols: usize, rows: usize, gather: bool, hot: bool) -> Vec<u8> {
    let nblocks = cols / Quant::A4.values_per_block();
    let stack_blocks = if hot { 0 } else { rows * nblocks };
    let x_stride = cols;
    let y_stride = rows;
    [
        cols as u32,
        x_stride as u32,
        y_stride as u32,
        LAYER_NONE,
        EXPERTS as u32,
        stack_blocks as u32,
        u32::from(gather),
        rows as u32,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect()
}

/// What one pass's launches read and write.
struct Pass {
    slots: usize,
    max_tiles: usize,
    tiles: MetalBuffer<u32>,
    sorted: MetalBuffer<u32>,
    inv: MetalBuffer<u32>,
    route_w: MetalBuffer<f32>,
    x: MetalBuffer<f32>,
    h: MetalBuffer<f32>,
    y: MetalBuffer<f32>,
    /// The table's tiles and live 8-row blocks, read back.
    n_tiles: usize,
    blocks: usize,
}

/// Encode one command buffer and wait for it.
fn once(
    ctx: &Ctx<'_>,
    encode: impl FnOnce(&mut ComputeEncoder<'_>) -> Result<(), String>,
) -> Result<(), String> {
    let mut recording = ctx.stream.begin().map_err(why)?;
    {
        let mut encoder = recording.compute_encoder().map_err(why)?;
        encode(&mut encoder)?;
        encoder.end();
    }
    ctx.stream.submit(recording).map_err(why)?;
    ctx.stream.sync().map_err(why)
}

fn read<T: bloomery_device::Elem + Default + Clone>(
    ctx: &Ctx<'_>,
    b: &MetalBuffer<T>,
    n: usize,
) -> Result<Vec<T>, String> {
    let mut out = vec![T::default(); n];
    ctx.device
        .read(
            ctx.stream,
            b.window(Elems(0), Elems(n)).map_err(why)?,
            &mut out,
        )
        .map_err(why)?;
    Ok(out)
}

/// The table of `ids` cut by the arm's own `route_table`, and the pass's
/// buffers.
fn pass(ctx: &Ctx<'_>, arm: &RArm<'_>, rows: usize, ids: &[u32]) -> Result<Pass, String> {
    let slots = rows * EXPERT_SLOTS;
    let max_tiles = slots.div_ceil(arm.tile) + EXPERTS + 1;
    let ids_b = ctx.device.upload(ctx.stream, ids).map_err(why)?;
    let zeroed_u32 = |n: usize| ctx.device.zeroed::<u32>(ctx.stream, Elems(n)).map_err(why);
    let mut tiles = zeroed_u32(4 * (1 + max_tiles))?;
    let mut sorted = zeroed_u32(slots)?;
    let mut inv = zeroed_u32(slots)?;
    let mut rng = Rng::new(0x5CE ^ rows as u64);
    let w: Vec<f32> = (0..slots)
        .map(|_| 0.1 + 0.9 * (rng.next() >> 40) as f32 / (1u64 << 24) as f32)
        .collect();
    let route_w = ctx.device.upload(ctx.stream, &w).map_err(why)?;
    let x = ctx
        .device
        .upload(
            ctx.stream,
            &oracle::activations(rows * HIDDEN, 0xA11CE, 1.0),
        )
        .map_err(why)?;
    let h = ctx
        .device
        .zeroed::<f32>(ctx.stream, Elems(slots * EXPERT_K))
        .map_err(why)?;
    let y = ctx
        .device
        .zeroed::<f32>(ctx.stream, Elems(slots * HIDDEN))
        .map_err(why)?;
    let pipeline = arm.library.pipeline("route_table").map_err(why)?;
    let args: Vec<u8> = [
        slots as u32,
        EXPERT_SLOTS as u32,
        EXPERTS as u32,
        max_tiles as u32,
        LAYER_NONE,
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    let fault = ctx.device.fault_word();
    once(ctx, |encoder| {
        encoder.set_pipeline(&pipeline);
        let ids_w = ids_b.window(Elems(0), Elems(slots)).map_err(why)?;
        let sorted_w = sorted.window_mut(Elems(0), Elems(slots)).map_err(why)?;
        let inv_w = inv.window_mut(Elems(0), Elems(slots)).map_err(why)?;
        let tiles_w = tiles
            .window_mut(Elems(0), Elems(4 * (1 + max_tiles)))
            .map_err(why)?;
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&ids_w).map_err(why)?).map_err(why)?;
        slot.set(Binding::of_mut(&sorted_w).map_err(why)?)
            .map_err(why)?;
        slot.set(Binding::of_mut(&inv_w).map_err(why)?)
            .map_err(why)?;
        slot.set(Binding::of_mut(&tiles_w).map_err(why)?)
            .map_err(why)?;
        slot.set(Binding::bytes(&args).map_err(why)?).map_err(why)?;
        slot.set(fault.binding()).map_err(why)?;
        slot.dispatch_groups(1, 1, TABLE_THREADS).map_err(why)
    })?;
    let table = read(ctx, &tiles, 4 * (1 + max_tiles))?;
    let n_tiles = table[0] as usize;
    if n_tiles > max_tiles {
        return Err(format!(
            "the table holds {n_tiles} tiles past its capacity {max_tiles}"
        ));
    }
    let blocks = (0..n_tiles)
        .map(|t| (table[4 * (1 + t) + 2] as usize).div_ceil(8))
        .sum();
    Ok(Pass {
        slots,
        max_tiles,
        tiles,
        sorted,
        inv,
        route_w,
        x,
        h,
        y,
        n_tiles,
        blocks,
    })
}

/// Encode one grouped launch: the pair (gate and up into `h`) or the down
/// (`h` into `y`), the engine's bindings.
fn whole(b: &MetalBuffer<u8>) -> Result<bloomery_device::Window<'_, u8, MetalDevice>, String> {
    b.whole().map_err(why)
}

fn launch(
    encoder: &mut ComputeEncoder<'_>,
    pipeline: &Pipeline,
    w: &Weights,
    p: &mut Pass,
    pair: bool,
    hot: bool,
    fault: &MetalFaultWord,
) -> Result<(), String> {
    encoder.set_pipeline(pipeline);
    let (cols, rows) = if pair {
        (HIDDEN, EXPERT_K)
    } else {
        (EXPERT_K, HIDDEN)
    };
    let args = routed_args(cols, rows, pair, hot);
    let (stack, stack_up, shared, shared_up) = if pair {
        (
            whole(&w.gate)?,
            whole(&w.up)?,
            whole(&w.sgate)?,
            whole(&w.sup)?,
        )
    } else {
        (
            whole(&w.down)?,
            whole(&w.down)?,
            whole(&w.sdown)?,
            whole(&w.sdown)?,
        )
    };
    let slots = p.slots;
    let (x, out) = if pair {
        (
            p.x.whole().map_err(why)?,
            p.h.window_mut(Elems(0), Elems(slots * EXPERT_K))
                .map_err(why)?,
        )
    } else {
        (
            p.h.whole().map_err(why)?,
            p.y.window_mut(Elems(0), Elems(slots * HIDDEN))
                .map_err(why)?,
        )
    };
    let tiles = p.tiles.whole().map_err(why)?;
    let sorted = p.sorted.whole().map_err(why)?;
    let route_w = p.route_w.whole().map_err(why)?;
    let mut slot = Bindings::new(encoder);
    for b in [&stack, &stack_up, &shared, &shared_up] {
        slot.set(Binding::of(b).map_err(why)?).map_err(why)?;
    }
    slot.set(Binding::of(&x).map_err(why)?).map_err(why)?;
    slot.set(Binding::of_mut(&out).map_err(why)?).map_err(why)?;
    slot.set(Binding::of(&tiles).map_err(why)?).map_err(why)?;
    slot.set(Binding::of(&sorted).map_err(why)?).map_err(why)?;
    slot.set(Binding::of(&route_w).map_err(why)?).map_err(why)?;
    slot.set(Binding::bytes(&args).map_err(why)?).map_err(why)?;
    slot.set(fault.binding()).map_err(why)?;
    let outputs = if pair { TILE_ROWS / 2 } else { TILE_ROWS };
    slot.dispatch_groups(p.max_tiles, rows / outputs, GEMM_THREADS)
        .map_err(why)
}

/// FNV-1a over the words of `v`.
fn fnv(h: u64, v: &[f32]) -> u64 {
    v.iter().fold(h, |h, w| {
        (h ^ u64::from(w.to_bits())).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The pair's and the down's outputs hashed in slot order: the place of a
/// slot inside its group is the table's arrival order, so a place-order hash
/// would differ between two runs of one arm.
fn hash(ctx: &Ctx<'_>, p: &Pass) -> Result<(u64, u64), String> {
    let inv = read(ctx, &p.inv, p.slots)?;
    let h = read(ctx, &p.h, p.slots * EXPERT_K)?;
    let y = read(ctx, &p.y, p.slots * HIDDEN)?;
    let (mut hh, mut hy) = (0xCBF2_9CE4_8422_2325u64, 0xCBF2_9CE4_8422_2325u64);
    for &place in &inv {
        let at = place as usize;
        hh = fnv(hh, &h[at * EXPERT_K..(at + 1) * EXPERT_K]);
        hy = fnv(hy, &y[at * HIDDEN..(at + 1) * HIDDEN]);
    }
    Ok((hh, hy))
}

/// Time every arm at every row count and skew.
pub fn run(ctx: &Ctx<'_>, opts: &Opts) -> Result<(), String> {
    let mut built: Vec<(&'static Variant, Library)> = Vec::new();
    for name in &opts.arms {
        if name != "ship" {
            let v = variants::find_routed(name).ok_or_else(|| {
                let known: Vec<&str> = variants::ROUTED.iter().map(|v| v.name).collect();
                format!(
                    "no routed arm named {name}: ship, or one of {}",
                    known.join(", ")
                )
            })?;
            let text = variants::apply(v, SOURCE)?;
            built.push((
                v,
                Library::from_source_text(ctx.device, &text).map_err(why)?,
            ));
        }
    }
    let mut arms = Vec::new();
    for name in &opts.arms {
        if name == "ship" {
            arms.push(RArm {
                name: name.clone(),
                library: ctx.device.library(),
                tile: ROUTE_TILE,
            });
        } else {
            let (v, library) = built
                .iter()
                .find(|(v, _)| v.name == name)
                .expect("every named variant was built");
            arms.push(RArm {
                name: name.clone(),
                library,
                tile: v.acts,
            });
        }
    }
    let w = weights(ctx)?;
    let fault = ctx.device.fault_word();
    for &rows in &opts.rows {
        for &skew in &opts.skews {
            let ids = picks(rows, skew, 0x7AB1E ^ rows as u64);
            for arm in &arms {
                let mut p = pass(ctx, arm, rows, &ids)?;
                let pair_p = arm
                    .library
                    .pipeline("gemm_routed_a4_pair_f16")
                    .map_err(why)?;
                let down_p = arm
                    .library
                    .pipeline("gemm_routed_a4_plain_f16")
                    .map_err(why)?;
                // One pair and one down first, so the down reads the pair's
                // outputs and the hash sees both.
                once(ctx, |e| {
                    launch(e, &pair_p, &w, &mut p, true, opts.hot, fault)?;
                    launch(e, &down_p, &w, &mut p, false, opts.hot, fault)
                })?;
                let mut layer = 0.0;
                for (kind, pipeline, pair) in [("pair", &pair_p, true), ("down", &down_p, false)] {
                    let Timed { seconds, wall } = time_dispatches(ctx, 1, 1, |e, _| {
                        launch(e, pipeline, &w, &mut p, pair, opts.hot, fault)
                    })?;
                    let (cols, n) = if pair {
                        (HIDDEN, 2 * EXPERT_K)
                    } else {
                        (EXPERT_K, HIDDEN)
                    };
                    let flops = 2.0 * (p.slots * n * cols) as f64;
                    println!(
                        "routed op={} u={rows} skew={skew} hot={} kind={kind} tiles={} blocks={} ms={:.3} tflops={:.2} wall_ms={:.3}",
                        arm.name,
                        u8::from(opts.hot),
                        p.n_tiles,
                        p.blocks,
                        seconds * 1e3,
                        flops / seconds / 1e12,
                        wall * 1e3
                    );
                    layer += seconds;
                }
                let flops = 2.0 * (p.slots * 3 * EXPERT_K * HIDDEN) as f64;
                println!(
                    "routed op={} u={rows} skew={skew} hot={} kind=layer tiles={} blocks={} ms={:.3} tflops={:.2}",
                    arm.name,
                    u8::from(opts.hot),
                    p.n_tiles,
                    p.blocks,
                    layer * 1e3,
                    flops / layer / 1e12
                );
                if ctx.method.hash {
                    let (hh, hy) = hash(ctx, &p)?;
                    println!(
                        "hash routed op={} u={rows} skew={skew} hot={} h_hash={hh:016x} y_hash={hy:016x}",
                        arm.name,
                        u8::from(opts.hot)
                    );
                }
            }
        }
    }
    Ok(())
}
