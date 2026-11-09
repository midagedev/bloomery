//! Checked load, sequence ownership and common model seam for Flint.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::attention_h128::{Binding, Cache, Kernels, Scratch as AttnScratch};
use bloomery_gpu::card_experts::CardFamily;
use bloomery_gpu::card_leg::ExpertInput;
use bloomery_gpu::head::Head;
use bloomery_gpu::host::leg_tier::{Hang, LegOpen};
use bloomery_gpu::host::run::{HostRun, HostWidths};
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::shell::TierShell;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{ResidencyGlue, ResidencySpec, hosted_stacks};
use bloomery_gpu::host::tier::TierOpen;
use bloomery_gpu::hybrid::{Chain, HostResidency, Hybrid, SlotMap};
use bloomery_gpu::model::{
    CardBytes, ChainBody, HostServed, Rollback, Slots, StepMode, hosted_hybrid_tiered,
    refuse_reserve_past, refuse_scratch_past, stage_run,
};
use bloomery_gpu::rope_table::{RopeRows, RopeSpec, RopeTable};
use bloomery_gpu::route_body::sigmoid_floor_256_6::RouterKernels;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::prompt_walk::{self, BatchBufs};
use bloomery_gpu_deepseek41::shared_routed::{FfnConfig, ResidualOrder, Routed, RoutedDims};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::Arch;
use model::arch::flint::{host, place::PlanInputs};
use model::arch::names;
use model::embed::EmbedTable;
use model::placement::Plan;
use model::placement::accepts::Program;
use models::{Ffn, Mixer, RopeMode};
use runtime::layer::{Layer, MixerKind, ResidualKind, hosted};

pub(crate) const WHAT: &str = "flint Body";
pub type FlintModel = GpuModel<Body>;
pub(crate) type FfnBody = Routed<RouterKernels>;

/// The per-replay position. The embedding row is staged before refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInput {
    pub pos: u32,
}

pub(crate) struct StepRows {
    pub x: DeviceBuffer<f32>,
    pub x1: DeviceBuffer<f32>,
    pub xn: DeviceBuffer<f32>,
    pub delta: DeviceBuffer<f32>,
    pub pos: DeviceBuffer<u32>,
    pub n_keys: DeviceBuffer<u32>,
}

impl StepRows {
    fn new(stream: &CudaStream, hidden: usize) -> Result<Self, GpuError> {
        Ok(Self {
            x: DeviceBuffer::zeroed(stream, hidden)?,
            x1: DeviceBuffer::zeroed(stream, hidden)?,
            xn: DeviceBuffer::zeroed(stream, hidden)?,
            delta: DeviceBuffer::zeroed(stream, hidden)?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    fn bytes(&self) -> usize {
        [&self.x, &self.x1, &self.xn, &self.delta]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }
}

/// Weights, kernels and scratch belong to the load; only caches and their high-water mark belong to a sequence.
pub struct Body {
    pub(crate) shell: TierShell<HostRun>,
    pub(crate) cfg: Vec<Layer>,
    pub(crate) bindings: Vec<Binding>,
    pub(crate) stores: Vec<Cache>,
    pub(crate) k: Kernels,
    pub(crate) attention: AttnScratch,
    pub(crate) batch_attention: AttnScratch,
    pub(crate) s: StepRows,
    pub(crate) rope: RopeRows,
    pub(crate) ffn: FfnBody,
    pub(crate) units: BatchBufs,
    pub(crate) embed: EmbedTable,
    row: Vec<f32>,
    pub(crate) slots: Arc<DeviceTensor<u32>>,
    pub(crate) at: u32,
    pub(crate) ctx: usize,
    pub(crate) hidden: usize,
    eps: f32,
    slots_planned: usize,
    slots_made: usize,
    prompt_reserve: u64,
}

impl Body {
    /// A single-sequence placed load with fixed expert residency.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<FlintModel, GpuError> {
        Self::open_placed_slots(
            file,
            plan,
            inputs,
            card,
            host,
            Residency::Off,
            NonZeroUsize::MIN,
        )
    }

    /// Load exactly the number of resident sequences counted by the placement.
    #[allow(
        clippy::too_many_arguments,
        reason = "the placed load's file, plan, card and host policy, residency and sequence count"
    )]
    pub fn open_placed_slots(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        residency: Residency,
        slots: NonZeroUsize,
    ) -> Result<FlintModel, GpuError> {
        let tiers = TierOpen::one_of(plan.machine, WHAT)?;
        let (layers, _) = stage_run(WHAT, plan, card, inputs.spec.layers.len())?;
        let one = inputs.kv.bytes(layers, plan.ctx_max);
        refuse_store_count(plan, card, one, slots)?;
        let run = hosted(&inputs.spec.layers).map_err(|e| shape(e.to_string()))?;
        let residency = ResidencySpec::hosted(
            residency,
            6,
            1,
            Arc::new(hosted_stacks(&inputs.model, WHAT, (run.len(), true))?),
        );
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            residency,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set, glue| {
                Self::load(
                    gpu,
                    file,
                    w,
                    (plan, inputs, card),
                    host,
                    (set, glue),
                    (&tiers, slots),
                )
            },
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the checked placed load's disjoint common loader parts"
    )]
    fn load(
        gpu: &Gpu,
        file: &Arc<Split>,
        w: &Weights,
        (plan, inputs, card): (&Plan<'_>, &PlanInputs, usize),
        host_cfg: HostCfg,
        (set, glue): (HostResidency, ResidencyGlue),
        (tiers, seqs): (&[TierOpen], NonZeroUsize),
    ) -> Result<Self, GpuError> {
        gpu.context().bind_to_thread()?;
        let spec = &inputs.spec;
        let (layers, ctx) = stage_run(WHAT, plan, card, spec.layers.len())?;
        let hidden = spec.hidden as usize;
        let cfg: Vec<_> = spec.layers.iter().map(Layer::of).collect();
        for (l, kind) in cfg.iter().enumerate() {
            if kind.mixer != MixerKind::Gqa || kind.residual != ResidualKind::Plain {
                return Err(shape(format!(
                    "layer {l}: {kind:?}, not plain-residual GQA"
                )));
            }
            match &spec.layers[l].ffn {
                Ffn::Moe(m) if m.experts != 256 || m.top_k != 6 || m.expert_ff != 512 => {
                    return Err(shape(format!(
                        "layer {l}: expert geometry {m:?}, not the compiled floor256top6 block"
                    )));
                }
                _ => {}
            }
        }
        let run = hosted(&spec.layers).map_err(|e| shape(e.to_string()))?;
        let tier_cards: Vec<_> = tiers.iter().map(|t| t.card).collect();
        let map = SlotMap::of_plan_tiers(plan, card, &tier_cards, run.clone(), 256)?;
        let bindings = spec
            .layers
            .iter()
            .enumerate()
            .map(|(l, s)| Binding::load(file, w, l, s, hidden, spec.rms_eps, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let stores = bindings
            .iter()
            .map(|b| Cache::new(gpu, b))
            .collect::<Result<Vec<_>, _>>()?;
        let held: usize = stores.iter().map(Cache::bytes).sum();
        let planned = inputs.kv.bytes(layers, plan.ctx_max);
        if held as u64 != planned {
            return Err(shape(format!(
                "stores hold {held} bytes; the plan counts {planned}"
            )));
        }
        for (l, b) in bindings.iter().enumerate() {
            let counted = inputs.kv.rows_at(l, plan.ctx_max);
            if b.capacity() as u64 != counted {
                return Err(shape(format!(
                    "layer {l}: {} physical rows; the plan counts {counted}",
                    b.capacity()
                )));
            }
        }
        let rope_base = spec
            .layers
            .iter()
            .find_map(|s| match &s.mixer {
                Mixer::Gqa(g) if g.rope.mode == RopeMode::Neox => Some(g.rope.base),
                _ => None,
            })
            .ok_or_else(|| shape("no NeoX layer in the validated description"))?;
        let table = RopeTable::new(&RopeSpec::window(rope_base, 128))?;
        let rope = RopeRows::new(gpu.stream(), &table, 128, ctx)?;
        let slots = Arc::new(DeviceTensor::upload(
            gpu.stream(),
            &map.stage_view(),
            run.len(),
            256,
        )?);
        let ffn = FfnBody::new_configured(
            gpu,
            file,
            w,
            (&spec.layers, &map, &slots),
            RoutedDims {
                embd: hidden,
                rms_eps: spec.rms_eps,
                batch_rows: crate::prefill::T_MAX,
            },
            CardFamily {
                what: WHAT,
                n_expert: 256,
                n_used: 6,
                stacks: names::routed_stacks,
                program: Program::FlintBody,
            },
            FfnConfig {
                residual: ResidualOrder::RoutedSharedResidual,
                expert_input: ExpertInput::AmaxF16,
            },
        )?;
        let hang = Hang {
            tiers,
            family: &crate::tier::NODE_FAMILY,
            open: LegOpen {
                n_expert: 256,
                layers: cfg.len(),
                hidden,
                n_used: 6,
                rows: 1,
            },
            cols: crate::prefill::T_MAX,
            card_dontneed: host_cfg.card_dontneed,
            bytes: CardBytes::File,
        };
        let mut hybrid = hosted_hybrid_tiered(
            (gpu, file, plan),
            (run.clone(), map),
            (host_cfg, set, &glue),
            (
                HostWidths {
                    embd: hidden,
                    ff: 512,
                    n_used: 6,
                },
                1,
            ),
            Some(&hang),
            |src| {
                run.clone()
                    .map(|l| host::layer(src, spec, l))
                    .collect::<Result<Vec<_>, _>>()
            },
        )?;
        prompt_walk::prepare_host(gpu, &mut hybrid, crate::prefill::T_MAX)?;
        let embed = EmbedTable::new(Arc::clone(file), &names::token_embd(), hidden)?;
        if embed.n_vocab() != spec.vocab as usize {
            return Err(shape(format!(
                "embedding vocabulary {} differs from {}",
                embed.n_vocab(),
                spec.vocab
            )));
        }
        let prompt_reserve = inputs
            .prompt_batch_bytes(plan.machine)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let mut body = Self {
            shell: TierShell::new(hybrid, glue, hidden, WHAT),
            k: Kernels::load(gpu)?,
            attention: AttnScratch::new(gpu, &bindings, 1)?,
            batch_attention: AttnScratch::new(gpu, &bindings, crate::prefill::T_MAX)?,
            s: StepRows::new(gpu.stream(), hidden)?,
            units: BatchBufs::new(gpu, hidden, crate::prefill::T_MAX, 1)?,
            cfg,
            bindings,
            stores,
            rope,
            ffn,
            embed,
            row: vec![0.0; hidden],
            slots,
            at: 0,
            ctx,
            hidden,
            eps: spec.rms_eps,
            slots_planned: seqs.get(),
            slots_made: 1,
            prompt_reserve,
        };
        refuse_scratch_past(
            WHAT,
            plan,
            card,
            body.scratch_bytes(),
            "step rows, attention, routed blocks, rope, host boundary and slot map",
        )?;
        body.check_prompt_reserve()?;
        body.start_residency(gpu)?;
        Ok(body)
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }
    pub fn kinds(&self) -> Vec<Layer> {
        self.cfg.clone()
    }
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Cache::bytes).sum()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.s.bytes()
            + self.attention.bytes()
            + self.k.bytes()
            + self.ffn.bytes()
            + self.rope.table.num_bytes()
            + self.shell.hybrid.boundary().device_bytes()
            + self.slots.buf().num_bytes()
    }
    pub fn prompt_bytes(&self) -> usize {
        self.batch_attention.bytes() + self.units.bytes() + self.ffn.batch_bytes()
    }
    pub(crate) fn check_prompt_reserve(&self) -> Result<(), GpuError> {
        refuse_reserve_past(
            WHAT,
            "prompt batch buffers",
            self.prompt_reserve,
            self.prompt_bytes() as u64,
            "attention rows, common walk units and shared routed batch buffers",
        )
    }

    /// Rollback never restores a ring. Its depth is measured from the greatest written end, not the last cut.
    pub fn cut_keeps(&self, pos: u32, n: u32) -> bool {
        let end = self.at.max(pos);
        n <= end
            && self.bindings.iter().all(|b| {
                runtime::stores::cut_depth_keeps(
                    u64::from(end - n),
                    b.capacity() as u64,
                    self.ctx as u64,
                    b.window() as u64,
                )
            })
    }
    pub fn keepable_at(&self, pos: u32, n: u32) -> u32 {
        if self.cut_keeps(pos, n) { n } else { 0 }
    }
    pub fn stand_for_cut(&mut self, pos: u32) {
        self.at = self.at.max(pos);
    }
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }
    pub fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.shell.hybrid_mut()
    }
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        self.shell.taps(gpu)
    }
}

impl ChainBody for Body {
    type Input = StepInput;
    type Host = Body;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;
    fn arch() -> Arch {
        Arch::Flint
    }
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<StepInput, GpuError> {
        if pos as usize >= self.ctx {
            return Err(shape(format!("position {pos} past context {}", self.ctx)));
        }
        self.embed.row_into(token, &mut self.row)?;
        Ok(StepInput { pos })
    }
    fn refresh(&mut self, stream: &CudaStream, input: &StepInput) -> Result<(), GpuError> {
        self.s.x.copy_from_host(stream, &self.row)?;
        self.s.pos.copy_from_host(stream, &[input.pos])?;
        self.s.n_keys.copy_from_host(stream, &[input.pos + 1])?;
        self.at = self.at.max(input.pos + 1);
        Ok(())
    }
    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        crate::step::enqueue(gpu, w, self, head)
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.shell.hybrid.settle(gpu.stream())?;
        self.at = 0;
        Ok(())
    }
    fn head_eps(&self) -> f32 {
        self.eps
    }
    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.scratch_bytes()
            + self.prompt_bytes()
            + self
                .shell
                .taps
                .as_ref()
                .map_or(0, |t| t.iter().map(DeviceBuffer::num_bytes).sum())
    }
    fn layers(&self) -> Range<usize> {
        0..self.cfg.len()
    }
    fn host(&mut self) -> Option<&mut Body> {
        Some(self)
    }
}

impl TierBody for Body {
    type Experts = HostRun;
    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.shell.hybrid()
    }
    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.shell.hybrid_mut()
    }
    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.shell.residency_parts(&self.slots)
    }
    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.shell.hybrid.serve_captured_of(chain)
    }
}

impl Rollback for Body {
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if !self.cut_keeps(self.at, pos) {
            return Err(shape(format!(
                "rollback to {pos} from written end {} loses ring keys; reset and refeed",
                self.at
            )));
        }
        Ok(())
    }
}

/// One resident sequence's device stores and greatest written end.
pub struct Seq {
    stores: Vec<Cache>,
    at: u32,
}

impl Slots for Body {
    type Seq = Seq;
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Seq, GpuError> {
        if self.slots_made >= self.slots_planned {
            return Err(shape(format!(
                "sequence {} exceeds {} planned resident sequences",
                self.slots_made + 1,
                self.slots_planned
            )));
        }
        let need = self.store_bytes();
        let free = gpu.alloc_room()?;
        if need > free {
            return Err(shape(format!(
                "sequence needs {need} store bytes, card has {free}"
            )));
        }
        let stores = self
            .bindings
            .iter()
            .map(|b| Cache::new(gpu, b))
            .collect::<Result<Vec<_>, _>>()?;
        self.slots_made += 1;
        Ok(Seq { stores, at: 0 })
    }
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Seq) -> Result<(), GpuError> {
        if self.stores.len() != seq.stores.len()
            || self
                .stores
                .iter()
                .zip(&seq.stores)
                .any(|(a, b)| a.bytes() != b.bytes())
        {
            return Err(shape("parked sequence has another store layout"));
        }
        std::mem::swap(&mut self.stores, &mut seq.stores);
        std::mem::swap(&mut self.at, &mut seq.at);
        Ok(())
    }
    fn seq_bytes(&self) -> usize {
        self.store_bytes()
    }
}

impl Drop for Body {
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

pub fn set_taps(m: &mut FlintModel, on: bool) -> Result<(), GpuError> {
    m.set_mode(StepMode::Eager);
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.shell.set_taps(gpu, body.cfg.len(), on)
}

fn refuse_store_count(
    plan: &Plan<'_>,
    card: usize,
    one: u64,
    slots: NonZeroUsize,
) -> Result<(), GpuError> {
    let want = one.saturating_mul(slots.get() as u64);
    let actual = plan
        .cards
        .get(card)
        .ok_or_else(|| shape(format!("no card {card} in plan")))?
        .kv_bytes;
    if want != actual {
        return Err(shape(format!(
            "{slots} resident sequences need {want} store bytes; plan counts {actual}"
        )));
    }
    Ok(())
}

pub(crate) fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::shape(WHAT, detail)
}
