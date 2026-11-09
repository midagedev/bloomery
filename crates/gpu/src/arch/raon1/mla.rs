//! Raon-1's wide arm of the gated latent layer: [`rows_wide`] runs a prompt unit of more than
//! [`PASS_ROWS`] rows, up to [`WIDE_ROWS`], over the sites ([`MlaSites`]) and buffers
//! ([`MlaBufs`]) of the common layer walk ([`crate::arch::gated_mla`]): the projections as GEMMs,
//! the cache append, the gather, the absorb ([`absorb_heads`], one 32-value GEMM over every
//! (row, head)), the value projection and the output gate once over the unit, and the latent
//! attention in chunks of [`ATTN_CHUNK`] query tokens. A unit of up to [`PASS_ROWS`] rows is the
//! common walk's one row ([`crate::arch::gated_mla::step`]) or pass ([`crate::arch::gated_mla::rows`]).

use super::state::PASS_ROWS;
use super::wide::{
    AbsorbShape, Input, WIDE_ROWS, WideKit, absorb_heads, input, out_runs, qb_runs, quantize_hid,
    quantize_out, quantize_qb, route_heads, value_heads,
};
use crate::GpuError;
use crate::arch::gated_mla::{MlaBufs, MlaSites, value_w};
use crate::arch::hybrid::GqaCx;
use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs};
use crate::latent::{LATENT, LatentExtraAppendArgs, Rows};
use crate::latent_gated::{MlaDims, MlaKernels, ValueW};
use crate::linear::out_gate::OutGateArgs;
use crate::q8f32::Q8_0GemvHeadsMcolArgs;
use crate::site::WideActs;
use crate::tensor::{Window, WindowMut};
use model::arch::models::OutGate;

/// What the refusals of this file name.
const WHAT: &str = "raon1 mla";

/// The query tokens a wide unit's latent flash takes a launch: the partials
/// ([`partials_lens_rows`](crate::latent_gated::partials_lens_rows)) hold this many tokens' rows a
/// segment, and an even count keeps each launch's groups of sixteen rows in the same token
/// pairs whatever the chunk (`heads` is 40 rows a token).
pub(super) const ATTN_CHUNK: usize = 32;

const _: () = assert!(ATTN_CHUNK >= PASS_ROWS && ATTN_CHUNK.is_multiple_of(2));
const _: () = assert!(WIDE_ROWS.is_multiple_of(ATTN_CHUNK));

/// `Ok` when the wide arm runs the layer's projections ([`rows_wide`]), else the refusal it
/// raises: `attn_q_b` and the output projection read q8 blocks of 32.
pub(super) fn wide_runs(sites: &MlaSites) -> Result<(), GpuError> {
    qb_runs(sites.q_b.ty())?;
    out_runs(sites.out.ty())
}

/// Enqueue MLA layer `l` over a wide unit of `m` rows (above [`PASS_ROWS`]),
/// in position order: the projections as GEMMs over the unit's rows
/// ([`super::wide`]); the cache append, the gather of the extra part, the
/// key absorption ([`absorb_heads`]: one GEMM over every (row, head) slot of
/// the unit), a Q6_K per-head value projection ([`value_heads`]: one grouped
/// GEMM over the same slots) and the output gate once over the unit; and the
/// latent flash in chunks of [`ATTN_CHUNK`] query tokens on windows of the
/// unit's buffers, each chunk over the cache rows the unit's append wrote,
/// bounded by its own live counts (a Q8_0 value projection runs per chunk, in
/// windows of [`PASS_ROWS`] rows). `b`'s buffers hold at least `m` rows of
/// their widths, the partials [`ATTN_CHUNK`] rows.
pub(super) fn rows_wide(
    cx: &GqaCx<'_>,
    k: &MlaKernels,
    kit: &mut WideKit<'_>,
    (sites, d): (&MlaSites, &MlaDims),
    (l, m): (usize, usize),
    b: MlaBufs<'_>,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let [norm, q_a_norm, kv_a_norm] = &sites.vecs;
    gpu.elem().enqueue_rms_norm(
        stream,
        b.x,
        w.f32_buf(WHAT, norm)?,
        d.rms_eps,
        d.embd,
        m,
        b.xn,
    )?;
    let (wk, wb) = (&kit.k, &mut kit.b);
    let tys = [sites.q_a.ty(), sites.kv_a.ty(), sites.gate.ty()];
    quantize_hid(
        gpu,
        wk.g32,
        wb.forms,
        (&*b.xn, m),
        (fault, WideActs::wants_of(&tys)),
    )?;
    {
        let x_in = input(wb.forms, Input::Hid, &*b.xn);
        wk.project(cx, (&sites.q_a, d.q_lora), &x_in, m, b.qa)?;
        wk.project(cx, (&sites.kv_a, d.row()), &x_in, m, b.kv)?;
        wk.project(cx, (&sites.gate, d.heads * d.v_head), &x_in, m, b.gate)?;
    }
    gpu.elem().enqueue_rms_norm(
        stream,
        b.qa,
        w.f32_buf(WHAT, q_a_norm)?,
        d.latent_eps,
        d.q_lora,
        m,
        b.qr,
    )?;
    quantize_qb(gpu, wk.g32, wb.forms, (&*b.qr, m), (fault, sites.q_b.ty()))?;
    {
        let x_in = input(wb.forms, Input::Qb, &*b.qr);
        wk.project(cx, (&sites.q_b, d.heads * d.head_k()), &x_in, m, b.q)?;
    }
    let (row, hl, hv) = (d.heads * d.row(), d.heads * LATENT, d.heads * d.v_head);
    let (f, u) = (size_of::<f32>(), size_of::<u32>());
    let kv_norm = w.f32_buf(WHAT, kv_a_norm)?;
    k.latent.enqueue_latent_extra_append(
        stream,
        LatentExtraAppendArgs {
            rows: Rows {
                x: &*b.kv,
                stride: d.row(),
                m,
            },
            off: 0,
            e_off: LATENT,
            extra: d.extra,
            gain: kv_norm,
            pos: b.pos,
            eps: d.latent_eps,
            fault,
            cache: &mut *b.cache,
        },
    )?;
    k.step.enqueue_gather(
        stream,
        &*b.q,
        &k.src,
        &k.dst,
        m * d.heads * d.extra,
        &mut *b.qabs,
    )?;
    route_heads(gpu, (wk, wb), (m, fault))?;
    let shape = AbsorbShape {
        head_k: d.head_k(),
        nope: d.nope,
        row: d.row(),
    };
    absorb_heads(
        gpu,
        (wk, wb),
        (w, &sites.k_b),
        (&*b.q, m),
        (shape, fault, &mut *b.qabs),
    )?;
    let value = value_w(w, &sites.v_b)?;
    let q8 = gpu.q8f32();
    for c0 in (0..m).step_by(ATTN_CHUNK) {
        let mc = ATTN_CHUNK.min(m - c0);
        let n_keys = Window::<u32>::of(b.n_keys, (c0 + mc - 1) * u, 1)?;
        let qabs = Window::<f32>::of(&*b.qabs, c0 * row * f, mc * row)?;
        let mut att = WindowMut::<f32>::of_mut(&mut *b.att, c0 * hl * f, mc * hl)?;
        gpu.flash().enqueue_flash_latent_split(
            stream,
            FlashSplitArgs {
                inputs: FlashInputs {
                    q: &qabs,
                    kv: &*b.cache,
                    n_keys_buf: &n_keys,
                    kq_scale: 1.0 / (d.head_k() as f32).sqrt(),
                    geom: FlashGeom {
                        tokens: mc,
                        heads: d.heads,
                        rope_dims: d.extra,
                        latent_dims: LATENT,
                    },
                },
                part_v: &mut *b.part_v,
                part_ms: &mut *b.part_ms,
                y: &mut att,
            },
        )?;
        drop(att);
        if let ValueW::Q8_0(vqs, vd) = value {
            for c1 in (0..mc).step_by(PASS_ROWS) {
                let mm = PASS_ROWS.min(mc - c1);
                let at = c0 + c1;
                let x = Window::<f32>::of(&*b.att, at * hl * f, mm * hl)?;
                let mut av = WindowMut::<f32>::of_mut(&mut *b.av, at * hv * f, mm * hv)?;
                q8.enqueue_q8_0_gemv_heads_mcol(
                    stream,
                    Q8_0GemvHeadsMcolArgs {
                        qs: vqs,
                        d: vd,
                        x: &x,
                        rows_per_head: d.v_head,
                        x_head_stride: LATENT,
                        y_head_stride: d.v_head,
                        y_off: 0,
                        m: mm,
                        x_col_stride: hl,
                        y_col_stride: hv,
                        y: &mut av,
                    },
                )?;
            }
        }
    }
    if let ValueW::Q6K(wv) = value {
        value_heads(
            gpu,
            (wk, wb),
            wv,
            (&*b.att, m),
            (d.v_head, fault, &mut *b.av),
        )?;
    }
    k.out_gate.enqueue_out_gate(
        stream,
        OutGateArgs {
            kind: OutGate::Proj,
            attn: &*b.av,
            gate: &*b.gate,
            n_head: d.heads,
            m,
            fault,
            y: &mut *b.gated,
        },
    )?;
    quantize_out(
        gpu,
        wk.g32,
        wb.forms,
        (&*b.gated, m),
        (fault, sites.out.ty()),
    )?;
    let x_in = input(wb.forms, Input::Out, &*b.gated);
    wk.project(cx, (&sites.out, d.embd), &x_in, m, b.out)
}

#[cfg(test)]
pub(super) mod tests {
    use super::{WIDE_ROWS, wide_runs};
    use crate::Gpu;
    use crate::arch::gated_mla::{MlaBufs, MlaSites, ValueHeads, rows, stage_len};
    use crate::arch::hybrid::{Acts, GqaCx, Site};
    use crate::arch::raon1::rda::tests::{Lcg, f32s, q6k, q8_0, resident, same};
    use crate::arch::raon1::rda::{RowKernels, RowsKit, add_row_forms};
    use crate::arch::raon1::state::PASS_ROWS;
    use crate::flash::{FlashGeom, FlashInputs, FlashSplitArgs};
    use crate::flash_gqa::FlashGqaKernels;
    use crate::gemm::Gemm32Kernels;
    use crate::latent::LATENT;
    use crate::latent_gated::{MlaDims, MlaKernels, partials_lens_rows};
    use crate::linear::LinearKernels;
    use crate::site::{KGemvKernels, SiteTy};
    use crate::tensor::{DeviceTensor, Window, WindowMut};
    use crate::weights::Weights;
    use cuda_core::DeviceBuffer;
    use gguf::quant::f32_to_f16_bits;
    use model::arch::raon1::names;

    /// The layer the names carry.
    const L: usize = 3;
    /// Raon's per-head widths at four heads, the model width two Q6_K
    /// super-blocks.
    const D: MlaDims = MlaDims {
        embd: 512,
        heads: 4,
        q_lora: 576,
        nope: 128,
        extra: 64,
        v_head: 128,
        rms_eps: 1e-5,
        latent_eps: 1e-6,
    };
    /// The pass's rows.
    const M: usize = 7;
    /// (cache rows, the pass's first position): the single-launch attention
    /// and the split one.
    const RUNS: [(usize, usize); 2] = [(64, 40), (1024, 300)];

    /// One MLA layer's file tensors at [`D`]'s widths, resident, and its
    /// sites: Raon's types (Q6_K `q_a`, `kv_a` and `v_b`, Q8_0 the rest).
    fn layer(gpu: &Gpu) -> (Weights, MlaSites) {
        let mut r = Lcg(0x006d_6c61);
        let (e, h, hv) = (D.embd, D.heads, D.heads * D.v_head);
        let gains = (0.5, 1.5);
        let parts = [
            f32s(&mut r, names::attn_norm(L), vec![e], gains),
            q6k(&mut r, names::attn_q_a(L), vec![e, D.q_lora]),
            f32s(&mut r, names::attn_q_a_norm(L), vec![D.q_lora], gains),
            q6k(&mut r, names::attn_kv_a_mqa(L), vec![e, D.row()]),
            f32s(&mut r, names::attn_kv_a_norm(L), vec![LATENT], gains),
            q8_0(&mut r, names::attn_gate(L), vec![e, hv]),
            q8_0(&mut r, names::attn_q_b(L), vec![D.q_lora, h * D.head_k()]),
            q8_0(&mut r, names::attn_k_b(L), vec![D.nope, LATENT, h]),
            q6k(&mut r, names::attn_v_b(L), vec![LATENT, D.v_head, h]),
            q8_0(&mut r, names::attn_output(L), vec![hv, e]),
        ];
        let w = resident(gpu, &parts);
        let site = |name: String, ty: SiteTy, k: usize| Site::new(name, ty, k);
        let sites = MlaSites {
            q_a: site(names::attn_q_a(L), SiteTy::Q6K, e),
            kv_a: site(names::attn_kv_a_mqa(L), SiteTy::Q6K, e),
            gate: site(names::attn_gate(L), SiteTy::Q8_0, e),
            q_b: site(names::attn_q_b(L), SiteTy::Q8_0, D.q_lora),
            out: site(names::attn_output(L), SiteTy::Q8_0, hv),
            k_b: names::attn_k_b(L),
            v_b: ValueHeads::Q6K(names::attn_v_b(L)),
            vecs: [
                names::attn_norm(L),
                names::attn_q_a_norm(L),
                names::attn_kv_a_norm(L),
            ],
        };
        (w, sites)
    }

    /// A layer's sites with `q_b` and `out` of the given types, the others Raon's.
    pub(crate) fn typed_sites(q_b: SiteTy, out: SiteTy) -> MlaSites {
        let site = |name: String, ty: SiteTy, k: usize| Site::new(name, ty, k);
        let (e, hv) = (D.embd, D.heads * D.v_head);
        MlaSites {
            q_a: site(names::attn_q_a(L), SiteTy::Q6K, e),
            kv_a: site(names::attn_kv_a_mqa(L), SiteTy::Q6K, e),
            gate: site(names::attn_gate(L), SiteTy::Q8_0, e),
            q_b: site(names::attn_q_b(L), q_b, D.q_lora),
            out: site(names::attn_output(L), out, hv),
            k_b: names::attn_k_b(L),
            v_b: ValueHeads::Q6K(names::attn_v_b(L)),
            vecs: [
                names::attn_norm(L),
                names::attn_q_a_norm(L),
                names::attn_kv_a_norm(L),
            ],
        }
    }

    /// The wide arm runs a latent layer whose `attn_q_b` and output projection
    /// are Q8_0 (Raon's), and refuses by name one whose either is another type:
    /// a Q6_K output projection reads q8_1 blocks of 128, and the wide arm
    /// quantizes the gated rows in blocks of 32 alone.
    #[test]
    fn a_latent_layer_runs_wide_only_with_q8_0_query_up_and_output() {
        assert!(wide_runs(&typed_sites(SiteTy::Q8_0, SiteTy::Q8_0)).is_ok());
        for ty in [SiteTy::Q6K, SiteTy::Q4K, SiteTy::F32] {
            let e = wide_runs(&typed_sites(SiteTy::Q8_0, ty)).unwrap_err();
            assert!(e.to_string().contains("an output projection"), "{e}");
            let e = wide_runs(&typed_sites(ty, SiteTy::Q8_0)).unwrap_err();
            assert!(e.to_string().contains("a query up projection"), "{e}");
        }
    }

    /// The buffers of a pass of up to [`PASS_ROWS`] rows over a cache of
    /// `ctx` rows, each its width times [`PASS_ROWS`]: the input, the
    /// positions and live counts, [`MlaBufs`]' f32 buffers in its order from
    /// `xn` to `out`, and the stage.
    struct Bufs {
        x: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        n_keys: DeviceBuffer<u32>,
        f: [DeviceBuffer<f32>; 13],
        stage: DeviceBuffer<f32>,
    }

    impl Bufs {
        fn new(gpu: &Gpu, ctx: usize) -> Bufs {
            let s = gpu.stream();
            let z = |n: usize| DeviceBuffer::<f32>::zeroed(s, n).expect("a buffer");
            let u = || DeviceBuffer::<u32>::zeroed(s, PASS_ROWS).expect("a word row");
            let (h, hv) = (D.heads, D.heads * D.v_head);
            let (pv, pm) = partials_lens_rows(&D, ctx, PASS_ROWS);
            let rows = [
                D.embd,
                D.q_lora,
                D.q_lora,
                D.row(),
                hv,
                h * D.head_k(),
                h * D.row(),
                h * LATENT,
                hv,
                hv,
            ];
            let [xn, qa, qr, kv, gate, q, qabs, att, av, gated] = rows.map(|n| z(PASS_ROWS * n));
            Bufs {
                x: z(PASS_ROWS * D.embd),
                pos: u(),
                n_keys: u(),
                f: [
                    xn,
                    qa,
                    qr,
                    kv,
                    gate,
                    q,
                    qabs,
                    att,
                    av,
                    gated,
                    z(pv),
                    z(pm),
                    z(PASS_ROWS * D.embd),
                ],
                stage: z(stage_len(&D)),
            }
        }
    }

    /// One MLA layer and everything a pass of it reads.
    struct Rig<'g> {
        gpu: &'g Gpu,
        w: Weights,
        sites: MlaSites,
        mk: MlaKernels,
        lin: LinearKernels,
        rk: RowKernels,
        kgemv: KGemvKernels,
        g32: Gemm32Kernels,
        flash: FlashGqaKernels,
        acts: Acts,
    }

    impl Rig<'_> {
        /// The first `n` rows of `xs` from position `p` as one pass on
        /// `cache`: the rows' updates.
        fn pass(
            &mut self,
            (b, cache): (&mut Bufs, &mut DeviceTensor<u16>),
            (p, n): (usize, usize),
            xs: &[f32],
        ) -> Vec<f32> {
            let s = self.gpu.stream();
            let at = u32::try_from(p).expect("a u32 position");
            let mut x = vec![0.0f32; PASS_ROWS * D.embd];
            x[..n * D.embd].copy_from_slice(&xs[..n * D.embd]);
            let pos: Vec<u32> = (at..).take(PASS_ROWS).collect();
            let live: Vec<u32> = (at + 1..).take(PASS_ROWS).collect();
            b.x.copy_from_host(s, &x).expect("the rows up");
            b.pos.copy_from_host(s, &pos).expect("the positions up");
            b.n_keys
                .copy_from_host(s, &live)
                .expect("the live counts up");
            let cx = GqaCx {
                what: "raon1 mla test",
                gpu: self.gpu,
                w: &self.w,
                kgemv: &self.kgemv,
                g32: &self.g32,
                flash: &self.flash,
                mma: false,
                out_gate: None,
                hidden: D.embd,
                eps: D.rms_eps,
                n_kv: 1,
                ctx: cache.rows(),
            };
            let [
                xn,
                qa,
                qr,
                kv,
                gate,
                q,
                qabs,
                att,
                av,
                gated,
                part_v,
                part_ms,
                out,
            ] = &mut b.f;
            let bufs = MlaBufs {
                x: &b.x,
                pos: &b.pos,
                n_keys: &b.n_keys,
                xn,
                qa,
                qr,
                kv,
                gate,
                q,
                qabs,
                att,
                av,
                gated,
                part_v,
                part_ms,
                out: &mut *out,
                cache,
            };
            let kit = RowsKit {
                linear: &self.lin,
                rows: &self.rk,
                stage: &mut b.stage,
            };
            let sd = (&self.sites, &D);
            rows(&cx, &self.mk, &mut self.acts, sd, (L, n), bufs, kit).expect("the pass");
            assert!(
                self.gpu.fault().expect("the fault word").is_none(),
                "a clean pass raises no fault"
            );
            out.to_host_vec(s).expect("the updates back")[..n * D.embd].to_vec()
        }
    }

    /// A pass of rows is its rows run as steps, bit for bit — the updates
    /// and every cache row — from a position past a filled cache prefix, on
    /// a cache short enough for the single-launch attention and on one the
    /// split attention cuts.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_mla_rows_are_steps() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (w, sites) = layer(&gpu);
        let (s, ctx) = (gpu.stream(), gpu.context());
        let mut acts = Acts::new(s, sites.sites().into_iter()).expect("the one-row forms");
        add_row_forms(s, &mut acts, sites.sites().into_iter()).expect("the row forms");
        let mut rig = Rig {
            gpu: &gpu,
            mk: MlaKernels::load(&gpu, &D, WIDE_ROWS).expect("the MLA kernels"),
            lin: LinearKernels::load(ctx).expect("the linear kernels"),
            rk: RowKernels::load(ctx).expect("the row kernels"),
            kgemv: KGemvKernels::load(&gpu).expect("the K-quant gemvs"),
            g32: Gemm32Kernels::load(ctx).expect("the F32 tile"),
            flash: FlashGqaKernels::load(ctx).expect("the flash kernels"),
            acts,
            w,
            sites,
        };
        let e = D.embd;
        for (height, p0) in RUNS {
            let mut r = Lcg(0x0070_6173 + p0 as u64);
            let xs = r.fill(M * e, -1.0, 1.0);
            let mut prefix: Vec<u16> = r
                .fill(p0 * D.row(), -1.0, 1.0)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect();
            prefix.resize(height * D.row(), 0);
            let cache = || DeviceTensor::upload(s, &prefix, height, D.row()).expect("the cache");
            let (mut c_steps, mut c_pass) = (cache(), cache());
            let mut b = Bufs::new(&gpu, height);
            let mut steps = Vec::with_capacity(M * e);
            for t in 0..M {
                steps.extend(rig.pass((&mut b, &mut c_steps), (p0 + t, 1), &xs[t * e..]));
            }
            let pass = rig.pass((&mut b, &mut c_pass), (p0, M), &xs);
            assert!(
                same(&pass, &steps, f32::to_bits),
                "cache of {height} rows from {p0}: a pass of {M} rows is its rows as steps"
            );
            let held = |c: &DeviceTensor<u16>| c.buf().to_host_vec(s).expect("the cache back");
            assert!(
                same(&held(&c_pass), &held(&c_steps), u32::from),
                "cache of {height} rows from {p0}: the pass writes the steps' cache rows"
            );
        }
    }

    /// The latent flash over a unit's query tokens is the same bits however
    /// the tokens are cut into launches: a row's cut of the keys is its
    /// group's, and under the single-span regime (a live count within
    /// `SEGMENTS` segments of the floor) every group cuts alike. The split
    /// attention (a cache past one segment), chunks of the groups' token
    /// period and a cut that falls inside a group.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_flash_tokens_per_launch_are_bits() {
        const TOKENS: usize = 32;
        const HEIGHT: usize = 1024;
        // The last token's live key count.
        const LIVE: usize = 600;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let s = gpu.stream();
        let mut r = Lcg(0x0066_6c61);
        let mut cache_rows: Vec<u16> = r
            .fill(LIVE * D.row(), -1.0, 1.0)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        cache_rows.resize(HEIGHT * D.row(), 0);
        let cache = DeviceTensor::upload(s, &cache_rows, HEIGHT, D.row()).expect("the cache");
        let (row, hl) = (D.heads * D.row(), D.heads * LATENT);
        let q = r.fill(TOKENS * row, -1.0, 1.0);
        let q = DeviceBuffer::from_host(s, &q).expect("the queries up");
        // The tokens cut into launches of `chunks` tokens each, in order.
        let run = |chunks: &[usize]| -> Vec<f32> {
            assert_eq!(
                chunks.iter().sum::<usize>(),
                TOKENS,
                "the cut covers the tokens"
            );
            let widest = chunks.iter().copied().max().expect("a chunk");
            let (pv, pm) = partials_lens_rows(&D, HEIGHT, widest);
            let mut part_v = DeviceBuffer::<f32>::zeroed(s, pv).expect("the partials");
            let mut part_ms = DeviceBuffer::<f32>::zeroed(s, pm).expect("the partials");
            let mut y = DeviceBuffer::<f32>::zeroed(s, TOKENS * hl).expect("the output");
            let mut c0 = 0;
            for &chunk in chunks {
                let last = u32::try_from(LIVE - (TOKENS - c0 - chunk)).expect("a u32 count");
                let n_keys = DeviceBuffer::from_host(s, &[last]).expect("the live count up");
                let qw = Window::<f32>::of(&q, c0 * row * 4, chunk * row).expect("the queries");
                let mut yw = WindowMut::<f32>::of_mut(&mut y, c0 * hl * 4, chunk * hl)
                    .expect("the output window");
                gpu.flash()
                    .enqueue_flash_latent_split(
                        s,
                        FlashSplitArgs {
                            inputs: FlashInputs {
                                q: &qw,
                                kv: &cache,
                                n_keys_buf: &n_keys,
                                kq_scale: 1.0 / (D.head_k() as f32).sqrt(),
                                geom: FlashGeom {
                                    tokens: chunk,
                                    heads: D.heads,
                                    rope_dims: D.extra,
                                    latent_dims: LATENT,
                                },
                            },
                            part_v: &mut part_v,
                            part_ms: &mut part_ms,
                            y: &mut yw,
                        },
                    )
                    .expect("the latent flash");
                c0 += chunk;
            }
            y.to_host_vec(s).expect("the output back")
        };
        let whole = run(&[TOKENS]);
        for cut in [
            &[16, 16][..],
            &[8, 8, 8, 8],
            &[4; 8],
            &[6, 26],
            &[1, 31],
            &[9, 9, 14],
        ] {
            assert!(
                same(&whole, &run(cut), f32::to_bits),
                "{TOKENS} tokens in one launch are the same bits cut as {cut:?}"
            );
        }
    }
}
