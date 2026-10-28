//! A host tier's computation over the layers that carry a host share
//! ([`HostRun`]): each such layer's stacks as [`HostLayer`] reads them from
//! the file, the scratch a step service writes, made at load, and the union
//! scratch a batch service writes, made once a caller asks for batches
//! ([`HostRun::prepare_union`]). The architecture supplies the layer set and
//! how its layers' stacks are found (a closure over its names and widths,
//! which refuses a layer by its own error); the protocol around a call is the
//! tier's ([`super::HostTier`]).

use std::sync::Arc;

use gguf::Split;
use model::moe::{HostLayer, HostScratch, UNION_MAX_COLS, UnionScratch};
use model::ops::{ExpertInput, PostNorm};
use model::r8file::{R8Pair, R8Source};
use model::{Tensor2, Tensor2View};

use super::HostExperts;
use super::census::HostReads;
use crate::GpuError;

use super::nvtier::NvTier;

/// The widths a host call serves: the model width, the routed experts'
/// feed-forward width and the routed slots a token. A model whose layers
/// differ in feed-forward width gives the widest: the step scratch and the
/// union's slabs are made per layer width, and the arena leg serves the
/// widest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostWidths {
    pub embd: usize,
    pub ff: usize,
    pub n_used: usize,
}

/// Every layer of the host run with its routed stacks, the one-column
/// scratch a step service writes, and a batch service's union scratch.
pub struct HostRun {
    /// The file and the r8 reading its layers were built from; every call
    /// reads through this pair.
    file: R8Pair,
    /// Per model layer, its host view; `None` for a layer that carries no
    /// host share, which no call may read: a call on it that lists no expert
    /// writes zeros ([`HostRun::zero_sum`]).
    layers: Vec<Option<HostLayer>>,
    /// One step scratch per feed-forward width the layers have, made for
    /// that width and [`HostWidths::embd`]; every layer of one width shares.
    scratch: Vec<HostScratch>,
    /// Per model layer, the index in `scratch` of its width's blocks; unread
    /// at a layer with no host share.
    scratch_of: Vec<usize>,
    widths: HostWidths,
    /// The union's slabs, once [`HostRun::prepare_union`] made them: one set
    /// per feed-forward width, in the order of `scratch`, so a layer's slabs
    /// are `union[scratch_of[layer]]`; empty until made.
    union: Vec<UnionScratch>,
    /// The NVMe expert tier's arena, when the plan built one and
    /// [`HostRun::attach_tier`] gave it: every paged layer's ids outside
    /// its host segment read from it.
    tier: Option<Arc<NvTier>>,
    /// The picks the calls read, by source, for the tier census.
    reads: HostReads,
}

impl HostRun {
    /// The layers of `run` a load's host run views ([`HostRun::build_at`]'s set): each that
    /// keeps fewer than `n_expert` experts on the card (`n_card`), or every one under a
    /// residency machine (`machine`), which moves experts to the host after the load. A layer
    /// that keeps every expert on the card has no view, so its stacks may be of a type no host
    /// kernel reads (NVFP4); a route trace's service of it lists no expert, whose sum the run
    /// writes without one ([`HostRun::zero_sum`]).
    pub fn viewed(
        run: std::ops::Range<usize>,
        machine: bool,
        (n_card, n_expert): (impl Fn(usize) -> usize + Clone, usize),
    ) -> impl Iterator<Item = usize> + Clone {
        run.filter(move |&l| machine || n_card(l) < n_expert)
    }

    /// The tier for the contiguous routed layers `first ..` of `file`, as
    /// many as `layers` builds ([`HostRun::build_at`] over
    /// `first .. first + n`).
    pub fn build<E>(
        file: Arc<Split>,
        first: usize,
        r8: bool,
        widths: HostWidths,
        layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
    ) -> Result<HostRun, GpuError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let (file, views) = Self::read(file, r8, layers)?;
        Self::assemble(file, first..first + views.len(), views, widths)
    }

    /// The tier for the layers of `set` of `file` — the model layers that
    /// carry a host share, strictly ascending, not necessarily contiguous —
    /// their host views made by `layers` from the file's reading, one per
    /// layer of `set` in its order: the architecture's names and widths,
    /// every refusal its own error. Load-time only: every stack is found and
    /// checked there, the gates and ups from `file`'s r8 sidecar when `r8`
    /// asks for it and there is one. A set out of order, or a count of views
    /// other than its layers', is refused by name.
    pub fn build_at<E>(
        file: Arc<Split>,
        set: impl IntoIterator<Item = usize>,
        r8: bool,
        widths: HostWidths,
        layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
    ) -> Result<HostRun, GpuError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let (file, views) = Self::read(file, r8, layers)?;
        Self::assemble(file, set, views, widths)
    }

    /// The file's r8 pair and the host views `layers` makes from its reading.
    fn read<E>(
        file: Arc<Split>,
        r8: bool,
        layers: impl FnOnce(R8Source<'_>) -> Result<Vec<HostLayer>, E>,
    ) -> Result<(R8Pair, Vec<HostLayer>), GpuError>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let file = R8Pair::at_load(file, r8)?;
        let views = layers(file.source()).map_err(|e| GpuError::Plan {
            what: "HostRun::build",
            source: Box::new(e),
        })?;
        Ok((file, views))
    }

    /// The run holding `views` at the layers of `set`, in order, and a `None`
    /// at every other layer below the set's last.
    fn assemble(
        file: R8Pair,
        set: impl IntoIterator<Item = usize>,
        views: Vec<HostLayer>,
        widths: HostWidths,
    ) -> Result<HostRun, GpuError> {
        const WHAT: &str = "HostRun::build_at";
        let set: Vec<usize> = set.into_iter().collect();
        if set.len() != views.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} host views for a set of {} layers",
                    views.len(),
                    set.len()
                ),
            ));
        }
        if let Some(w) = set.windows(2).find(|w| w[0] >= w[1]) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the layer set is not strictly ascending: {} then {}",
                    w[0], w[1]
                ),
            ));
        }
        let mut layers = Vec::new();
        layers.resize_with(set.last().map_or(0, |&last| last + 1), || None);
        let mut scratch: Vec<HostScratch> = Vec::new();
        let mut scratch_of = vec![0; layers.len()];
        for (l, view) in set.into_iter().zip(views) {
            scratch_of[l] = match scratch.iter().position(|s| s.ff() == view.ff()) {
                Some(i) => i,
                None => {
                    scratch.push(HostScratch::new(widths.embd, view.ff(), widths.n_used)?);
                    scratch.len() - 1
                }
            };
            if view.expert_input() == ExpertInput::AmaxF16 {
                scratch[scratch_of[l]].set_expert_input(ExpertInput::AmaxF16);
            }
            layers[l] = Some(view);
        }
        Ok(HostRun {
            file,
            layers,
            scratch,
            scratch_of,
            widths,
            union: Vec::new(),
            tier: None,
            reads: HostReads::default(),
        })
    }

    /// Give the run the NVMe expert tier's arena, and each layer the run
    /// covers its own handle: the plan's host segment and the arena, the
    /// parts resolved by the layer's stack names. Load-time only, once,
    /// after the build; a run with no tier keeps reading the file mapping
    /// alone. Refused by name: a tier over another open of the model file,
    /// whose drops would not reach the pages the run's union reads.
    pub fn attach_tier(&mut self, tier: Option<Arc<NvTier>>) -> Result<(), GpuError> {
        if tier
            .as_ref()
            .is_some_and(|t| !t.drops_reach(self.file.split()))
        {
            return Err(GpuError::shape(
                "HostRun::attach_tier",
                "the NVMe tier was built over another open of the model file than the run reads: \
                 its drops would leave the run's pages cached",
            ));
        }
        if let Some(tier) = tier.as_ref() {
            for (l, layer) in self.layers.iter_mut().enumerate() {
                let Some(layer) = layer else {
                    continue;
                };
                if !tier.covers(l) {
                    continue;
                }
                let names = layer.stacks().map(|s| s.info().name.clone());
                tier.note_parts(l, &names.each_ref().map(|n| n.as_str()))?;
                layer.attach_tier(l, tier.host_ids(l), tier.slots());
            }
        }
        self.tier = tier;
        Ok(())
    }

    /// The run's NVMe expert tier, when the plan built one; the counters a
    /// caller reads ([`NvTier::stats`]).
    #[must_use]
    pub fn tier(&self) -> Option<&Arc<NvTier>> {
        self.tier.as_ref()
    }

    /// The union's slabs for batch calls of up to `cols` columns, one set per
    /// feed-forward width the layers have (each over experts that map `embd
    /// -> ff -> embd` at its own `ff`), made once: a load that never serves a
    /// batch never holds them. Past [`UNION_MAX_COLS`] they are a group
    /// tail's ([`UnionScratch::new_tail`]) over whole batches of that many
    /// columns, with room for every slot, so a call of `cols` columns runs as
    /// one. A later call for no more columns than they hold keeps them; one
    /// for more is refused by name (the slabs are the load's size). Load-time
    /// only.
    pub fn prepare_union(&mut self, cols: usize) -> Result<(), GpuError> {
        if let Some(have) = self.union.first() {
            return if cols <= have.max_cols() {
                Ok(())
            } else {
                Err(GpuError::shape(
                    "HostRun::prepare_union",
                    format!(
                        "slabs for {cols} columns; the load made them for {}",
                        have.max_cols()
                    ),
                ))
            };
        }
        let (embd, n_used) = (self.widths.embd, self.widths.n_used);
        self.union = self
            .scratch
            .iter()
            .map(|s| {
                let mut union = if cols <= UNION_MAX_COLS {
                    UnionScratch::new_routed(embd, s.ff(), cols, n_used)
                } else {
                    let groups = cols.div_ceil(UNION_MAX_COLS);
                    let slots = groups * UNION_MAX_COLS * n_used;
                    UnionScratch::new_tail(embd, s.ff(), groups, slots, n_used)
                }?;
                union.set_expert_input(s.expert_input());
                Ok::<_, model::ModelError>(union)
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    /// Layers that carry a host share.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// Whether no layer carries a host share.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(Option::is_none)
    }

    /// Every layer's host view, indexed by model layer: `None` where the
    /// layer carries no host share, and nothing past the last layer that does.
    #[must_use]
    pub fn layers(&self) -> &[Option<HostLayer>] {
        &self.layers
    }

    /// The routed experts' feed-forward width the run serves.
    #[must_use]
    pub fn ff(&self) -> usize {
        self.widths.ff
    }

    /// Layer `layer`'s host view; a layer that carries no host share is
    /// refused by name.
    pub fn layer(&self, layer: usize) -> Result<&HostLayer, GpuError> {
        view_of(&self.layers, layer, "HostRun::layer")
    }
}

/// Layer `layer`'s host view in `layers`, for a call refused by name as
/// `what` when the layer carries no host share: the one lookup, so a gap in
/// the set never reads another layer's stacks.
fn view_of<'a>(
    layers: &'a [Option<HostLayer>],
    layer: usize,
    what: &'static str,
) -> Result<&'a HostLayer, GpuError> {
    layers.get(layer).and_then(Option::as_ref).ok_or_else(|| {
        GpuError::shape(
            what,
            format!("layer {layer} carries no host share: it has no routed stacks in the run"),
        )
    })
}

impl HostRun {
    /// A call on layer `layer` where the run holds no view of it and none of
    /// the call's `lists` names an expert, `x_cols` columns of `width` into
    /// `out`: a layer that keeps every expert on the card, whose step a route
    /// trace still serves (the service writes the trace), sums nothing, so
    /// `out` takes zeros, the empty lists' sums; `true` once written. `false`
    /// where the run holds a view or a list names an expert, which the view
    /// serves or [`view_of`] refuses by name. Refused by name as `what`:
    /// a width other than the model's, columns other than one a list and an
    /// `out` of another size.
    fn zero_sum(
        &self,
        layer: usize,
        lists: &[&[(u32, f32)]],
        (width, x_cols): (usize, usize),
        out: &mut [f32],
        what: &'static str,
    ) -> Result<bool, GpuError> {
        if lists.iter().any(|l| !l.is_empty())
            || self.layers.get(layer).is_some_and(Option::is_some)
        {
            return Ok(false);
        }
        let cols = lists.len();
        if width != self.widths.embd || x_cols != cols || width.checked_mul(cols) != Some(out.len())
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{x_cols} columns of {width} values beside {cols} lists, a sum of {} values",
                    out.len()
                ),
            ));
        }
        out.fill(0.0);
        Ok(true)
    }

    /// Layer `layer`'s host view, the run's byte source and the union's
    /// slabs of the layer's width, for a union call refused by name as
    /// `what`: a layer with no host share, and slabs never made
    /// ([`HostRun::prepare_union`]).
    fn union_of(
        &mut self,
        layer: usize,
        what: &'static str,
    ) -> Result<(&HostLayer, R8Source<'_>, &mut UnionScratch), GpuError> {
        let HostRun {
            file,
            layers,
            scratch_of,
            union,
            ..
        } = self;
        let view = view_of(layers, layer, what)?;
        let scratch = union.get_mut(scratch_of[layer]).ok_or(GpuError::State {
            what,
            missing: "the union's slabs (HostRun::prepare_union)",
        })?;
        Ok((view, file.source(), scratch))
    }
}

impl HostRun {
    /// Give layer `layer` the norm each of its routed experts' output rows
    /// takes before the row's weight ([`PostNorm`]): `gain` the layer's
    /// post-expert weight, `embd` values, and `eps` the norm's epsilon. The
    /// layer holds it ([`HostLayer::set_post_norm`]), so every call that
    /// serves the layer returns the sum of the normed, weighted rows, the norm
    /// taken in the leg's own sum; a layer without one keeps the plain
    /// weighted sum. Load-time only. Refused by name: a layer with no host
    /// share, a gain of another width and an epsilon that is not finite and
    /// positive.
    pub fn set_post_norm(
        &mut self,
        layer: usize,
        gain: Vec<f32>,
        eps: f32,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostRun::set_post_norm";
        view_of(&self.layers, layer, WHAT)?;
        if gain.len() != self.widths.embd {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer}: a gain of {} values for rows of {}",
                    gain.len(),
                    self.widths.embd
                ),
            ));
        }
        let Some(post) = PostNorm::new(gain, eps) else {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer}: an epsilon of {eps}, finite and positive wanted"),
            ));
        };
        self.layers[layer]
            .as_mut()
            .expect("view_of found the layer's host view")
            .set_post_norm(post)?;
        Ok(())
    }

    /// The union call of layer `layer` (`step`: the step port's form, whose
    /// NVMe ids read from the arena).
    fn union_call(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
        step: bool,
    ) -> Result<(), GpuError> {
        let what = if step {
            "HostRun::experts_step_union_into"
        } else {
            "HostRun::experts_union_into"
        };
        if self.zero_sum(layer, lists, (x.ne0(), x.ne1()), out, what)? {
            return Ok(());
        }
        let (view, src, scratch) = self.union_of(layer, what)?;
        if step {
            view.experts_step_union_into(src, x, lists, out, scratch)?;
        } else {
            view.experts_union_into(src, x, lists, out, scratch)?;
        }
        // A layer with an arena reads the ids it serves from the arena's
        // slots, batch by batch; the plan's host segment and a layer with
        // none read the mapping.
        self.reads.arena_or_file(
            self.tier.as_deref(),
            layer,
            lists.iter().flat_map(|l| l.iter().map(|&(id, _)| id)),
        );
        Ok(())
    }
}

impl HostExperts for HostRun {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
        columns: u64,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostRun::experts_into";
        if self.zero_sum(layer, &[experts], (x.ne0, x.ne1), out, WHAT)? {
            return Ok(());
        }
        // A tiered load prepares union slabs of at least one column. That
        // call is the resident path: hits compute while a miss fills. A load
        // with no slabs keeps today's fill-then-serve.
        if self.tier.is_some() && !self.union.is_empty() {
            let HostRun {
                file,
                layers,
                scratch,
                scratch_of,
                union,
                reads,
                tier,
                ..
            } = self;
            let view = view_of(layers, layer, WHAT)?;
            view.experts_into_on(
                model::ops::Lanes::host_tier(),
                file.source(),
                x,
                experts,
                out,
                &mut scratch[scratch_of[layer]],
                Some(&mut union[scratch_of[layer]]),
                columns,
            )?;
            reads.arena_or_file(tier.as_deref(), layer, experts.iter().map(|&(id, _)| id));
            return Ok(());
        }
        let view = view_of(&self.layers, layer, WHAT)?;
        view.experts_into_on(
            model::ops::Lanes::host_tier(),
            self.file.source(),
            x,
            experts,
            out,
            &mut self.scratch[self.scratch_of[layer]],
            None,
            columns,
        )?;
        self.reads.arena_or_file(
            self.tier.as_deref(),
            layer,
            experts.iter().map(|&(id, _)| id),
        );
        Ok(())
    }

    fn experts_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        self.union_call(layer, x, lists, out, false)
    }

    /// The union with the NVMe tier's ids read from the arena
    /// ([`HostLayer::experts_step_union_into`]): a decode step's reads leave
    /// no page of them in the page cache, so no drop follows.
    fn experts_step_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        self.union_call(layer, x, lists, out, true)
    }

    fn reads(&self) -> Option<HostReads> {
        Some(self.reads)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use gguf::write::{Layout, TensorDecl, Writer};
    use model::arch::names::routed_stacks;
    use model::moe::HostLayerSpec;

    use super::*;

    const WIDTH: usize = 256;
    const EXPERTS: usize = 2;
    const WIDTHS: HostWidths = HostWidths {
        embd: WIDTH,
        ff: WIDTH,
        n_used: 2,
    };

    /// A model file holding the three routed stacks of each `(layer, ff)` of
    /// `have` and nothing else, in a directory of its own that the test
    /// removes.
    fn file_with(have: &[(usize, usize)], dir: &std::path::Path) -> Arc<Split> {
        file_with_data(have, dir, &|_, nbytes| {
            let block = [[0x00, 0x24].as_slice(), &[1u8; 32]].concat();
            block.repeat(nbytes / block.len())
        })
    }

    /// [`file_with`] with the bytes of each stack `data` makes from its name and
    /// byte count (a whole number of q8_0 blocks of 34 bytes).
    fn file_with_data(
        have: &[(usize, usize)],
        dir: &std::path::Path,
        data: &dyn Fn(&str, usize) -> Vec<u8>,
    ) -> Arc<Split> {
        // The stacks' bytes: q8_0 rows of `k` columns, `rows` rows an expert.
        let bytes = |k: usize, rows: usize| 34 * (k / 32) * rows * EXPERTS;
        let decls: Vec<(String, Vec<u64>, usize)> = have
            .iter()
            .flat_map(|&(l, ff)| {
                let n = routed_stacks(l);
                let gate_up = vec![WIDTH as u64, ff as u64, EXPERTS as u64];
                let down = vec![ff as u64, WIDTH as u64, EXPERTS as u64];
                [
                    (n[0].clone(), gate_up.clone(), bytes(WIDTH, ff)),
                    (n[1].clone(), gate_up, bytes(WIDTH, ff)),
                    (n[2].clone(), down, bytes(ff, WIDTH)),
                ]
            })
            .collect();
        let layout = Layout::new(
            &[],
            decls
                .iter()
                .map(|(name, dims, nbytes)| TensorDecl {
                    name: name.clone(),
                    dims: dims.clone(),
                    type_id: 8,
                    nbytes: *nbytes as u64,
                })
                .collect(),
        )
        .unwrap();
        let path = dir.join("model.gguf");
        let mut writer = Writer::new(std::fs::File::create_new(&path).unwrap(), layout).unwrap();
        for (name, _, nbytes) in &decls {
            let bytes = data(name, *nbytes);
            assert_eq!(bytes.len(), *nbytes);
            writer.tensor(name, &bytes).unwrap();
        }
        writer.finish().unwrap();
        Arc::new(Split::open(&path).unwrap())
    }

    /// `body` over a file of the layers `have` at the widths `ff_of` gives,
    /// its directory removed after, a panic of `body` carried on.
    fn with_file_ff(
        have: &[usize],
        ff_of: fn(usize) -> usize,
        body: impl FnOnce(Arc<Split>) + std::panic::UnwindSafe,
    ) {
        with_file_data_ff(have, ff_of, None, body);
    }

    /// The bytes of a stack a test file holds, from its name and byte count.
    type StackData<'a> = &'a dyn Fn(&str, usize) -> Vec<u8>;

    /// [`with_file_ff`] over the bytes `data` makes, the repeated block's when `None`.
    fn with_file_data_ff(
        have: &[usize],
        ff_of: fn(usize) -> usize,
        data: Option<StackData<'_>>,
        body: impl FnOnce(Arc<Split>) + std::panic::UnwindSafe,
    ) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "bloomery-host-run-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let have: Vec<(usize, usize)> = have.iter().map(|&l| (l, ff_of(l))).collect();
        let file = match data {
            Some(data) => file_with_data(&have, &dir, data),
            None => file_with(&have, &dir),
        };
        let result = std::panic::catch_unwind(move || body(file));
        std::fs::remove_dir_all(&dir).unwrap();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    fn with_file(have: &[usize], body: impl FnOnce(Arc<Split>) + std::panic::UnwindSafe) {
        with_file_ff(have, |_| WIDTH, body);
    }

    fn views(src: R8Source<'_>, layers: &[usize]) -> Result<Vec<HostLayer>, model::ModelError> {
        views_ff(src, layers, |_| WIDTH)
    }

    fn views_ff(
        src: R8Source<'_>,
        layers: &[usize],
        ff_of: fn(usize) -> usize,
    ) -> Result<Vec<HostLayer>, model::ModelError> {
        layers
            .iter()
            .map(|&l| {
                let n = routed_stacks(l);
                HostLayer::build(
                    src,
                    &HostLayerSpec {
                        gate: &n[0],
                        up: &n[1],
                        down: &n[2],
                        n_expert: EXPERTS,
                        embd: WIDTH,
                        ff: ff_of(l),
                        swiglu_limit: 0.0,
                    },
                )
            })
            .collect()
    }

    fn stack_names(view: &HostLayer) -> [String; 3] {
        view.stacks().map(|s| s.info().name.clone())
    }

    /// A layer with no host share reading another layer's experts: a set with
    /// gaps maps each present layer to its own rows, and a call for a gap, or
    /// for a layer past the set's last, that lists an expert is a named error
    /// that writes nothing. A call there that lists none (a layer every
    /// expert of which is on the card, served for a route trace) writes the
    /// empty lists' zeros, step and union alike; one whose width, sum or
    /// columns differ is refused by name. Mutants: the zero branch dropped
    /// (the traced step refused), taken on a listed call, or the model width
    /// check dropped.
    #[test]
    fn a_sparse_set_serves_its_layers_and_refuses_the_gaps_by_name() {
        const SET: [usize; 3] = [1, 4, 5];
        with_file(&SET, |file| {
            let mut run =
                HostRun::build_at(file, SET, false, WIDTHS, |src| views(src, &SET)).unwrap();
            assert_eq!((run.len(), run.layers().len()), (3, 6));
            for l in SET {
                assert_eq!(stack_names(run.layer(l).unwrap()), routed_stacks(l));
            }
            let x = Tensor2 {
                ne0: WIDTH,
                ne1: 1,
                data: vec![0.5; WIDTH],
            };
            let mut out = vec![7.0f32; WIDTH];
            run.prepare_union(1).unwrap();
            for gap in [0, 2, 3, 6, 100] {
                let named = format!("layer {gap} carries no host share");
                let e = match run.layer(gap) {
                    Err(e) => e.to_string(),
                    Ok(view) => panic!("layer {gap} served the stacks {:?}", stack_names(view)),
                };
                assert!(e.contains(&named), "{e}");
                let e = run
                    .experts_into(gap, &x, &[(0, 1.0)], &mut out, 1)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains(&named), "{e}");
                let lists: [&[(u32, f32)]; 1] = [&[(0, 1.0)]];
                for union in [
                    HostExperts::experts_union_into,
                    HostExperts::experts_step_union_into,
                ] {
                    let e = union(&mut run, gap, x.view(), &lists, &mut out)
                        .unwrap_err()
                        .to_string();
                    assert!(e.contains(&named), "{e}");
                }
                assert!(out.iter().all(|&v| v == 7.0), "a refused call wrote");
                let e = run
                    .experts_into(gap, &x, &[], &mut out[1..], 1)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("a sum of 255 values"), "{e}");
                run.experts_into(gap, &x, &[], &mut out, 1).unwrap();
                assert!(out.iter().all(|&v| v == 0.0), "layer {gap} step");
                let empty: [&[(u32, f32)]; 1] = [&[]];
                let wrong = Tensor2::zeros(1, 1);
                let mut short = [7.0];
                let e = run
                    .experts_into(gap, &wrong, &[], &mut short, 1)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("HostRun::experts_into"), "{e}");
                assert_eq!(short, [7.0], "a wrong-width step wrote");
                let two: [&[(u32, f32)]; 2] = [&[], &[]];
                for union in [
                    HostExperts::experts_union_into,
                    HostExperts::experts_step_union_into,
                ] {
                    let e = union(&mut run, gap, wrong.view(), &empty, &mut short)
                        .unwrap_err()
                        .to_string();
                    assert!(e.contains("HostRun::experts_"), "{e}");
                    assert_eq!(short, [7.0], "a wrong-width union wrote");
                    out.fill(7.0);
                    let e = union(&mut run, gap, x.view(), &two, &mut out)
                        .unwrap_err()
                        .to_string();
                    assert!(e.contains("1 columns of 256 values beside 2 lists"), "{e}");
                    union(&mut run, gap, x.view(), &empty, &mut out).unwrap();
                    assert!(out.iter().all(|&v| v == 0.0), "layer {gap} union");
                }
                out.fill(7.0);
            }
            run.experts_into(4, &x, &[(0, 1.0)], &mut out, 1).unwrap();
        });
    }

    /// The contiguous entry is the set `first .. first + n`.
    #[test]
    fn the_contiguous_build_is_the_set_from_its_first_layer() {
        with_file(&[2, 3], |file| {
            let run = HostRun::build(file, 2, false, WIDTHS, |src| views(src, &[2, 3])).unwrap();
            assert_eq!((run.len(), run.layers().len()), (2, 4));
            assert!(run.layer(1).is_err() && run.layer(4).is_err());
            for l in [2, 3] {
                assert_eq!(stack_names(run.layer(l).unwrap()), routed_stacks(l));
            }
        });
    }

    /// A set that is not strictly ascending, or whose views are not one a
    /// layer, would place a stack at another layer's index.
    #[test]
    fn a_set_out_of_order_or_off_its_view_count_is_refused() {
        with_file(&[1, 4], |file| {
            let build = |set: &[usize], views_of: &[usize]| {
                HostRun::build_at(
                    Arc::clone(&file),
                    set.iter().copied(),
                    false,
                    WIDTHS,
                    |src| views(src, views_of),
                )
                .err()
                .map(|e| e.to_string())
            };
            let e = build(&[4, 1], &[1, 4]).unwrap();
            assert!(e.contains("not strictly ascending: 4 then 1"), "{e}");
            let e = build(&[1, 1], &[1, 4]).unwrap();
            assert!(e.contains("not strictly ascending: 1 then 1"), "{e}");
            let e = build(&[1, 4, 5], &[1, 4]).unwrap();
            assert!(e.contains("2 host views for a set of 3 layers"), "{e}");
            assert!(build(&[1, 4], &[1, 4]).is_none());
        });
    }

    /// A layer reading blocks made for another layer's width: a run whose
    /// layers differ in feed-forward width serves each through its own
    /// scratch, a scratch of another width is refused by name, and its union
    /// holds one set of slabs a width, each layer served through its own
    /// width's, column for column the step's bits.
    #[test]
    fn layers_of_different_ff_each_serve_through_their_own_scratch() {
        const SET: [usize; 3] = [1, 3, 4];
        fn ff_of(l: usize) -> usize {
            if l == 3 { WIDTH / 2 } else { WIDTH }
        }
        with_file_ff(&SET, ff_of, |file| {
            let mut run =
                HostRun::build_at(file, SET, false, WIDTHS, |src| views_ff(src, &SET, ff_of))
                    .unwrap();
            let x = Tensor2 {
                ne0: WIDTH,
                ne1: 1,
                data: vec![0.5; WIDTH],
            };
            let mut out = vec![7.0f32; WIDTH];
            for l in SET {
                run.experts_into(l, &x, &[(0, 1.0)], &mut out, 1).unwrap();
            }
            let mut wide = HostScratch::new(WIDTH, WIDTH, 2).unwrap();
            let e = run
                .layer(3)
                .unwrap()
                .experts_into(run.file.source(), &x, &[(0, 1.0)], &mut out, &mut wide)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("scratch must be made for the layer's widths"),
                "{e}"
            );
            let e = run.experts_union_into(1, x.view(), &[&[(0, 1.0)]], &mut out);
            assert!(
                e.unwrap_err()
                    .to_string()
                    .contains("HostRun::prepare_union"),
                "a union before its slabs is refused by name"
            );
            run.prepare_union(2).unwrap();
            let ffs: Vec<usize> = run.union.iter().map(UnionScratch::ff).collect();
            assert_eq!(ffs, [WIDTH, WIDTH / 2], "one set of slabs a width");
            let x2 = Tensor2 {
                ne0: WIDTH,
                ne1: 2,
                data: (0..2 * WIDTH)
                    .map(|i| 0.01 * (i % 13) as f32 - 0.05)
                    .collect(),
            };
            let lists: [&[(u32, f32)]; 2] = [&[(0, 1.0), (1, 0.5)], &[(1, 0.25)]];
            for l in SET {
                let mut union = vec![7.0f32; 2 * WIDTH];
                run.experts_union_into(l, x2.view(), &lists, &mut union)
                    .unwrap();
                for (j, list) in lists.iter().enumerate() {
                    let col = Tensor2 {
                        ne0: WIDTH,
                        ne1: 1,
                        data: x2.data[j * WIDTH..(j + 1) * WIDTH].to_vec(),
                    };
                    let mut one = vec![0.0f32; WIDTH];
                    run.experts_into(l, &col, list, &mut one, 1).unwrap();
                    assert_eq!(
                        &union[j * WIDTH..(j + 1) * WIDTH],
                        &one[..],
                        "layer {l} column {j}"
                    );
                }
            }
            let e = run.prepare_union(3).unwrap_err().to_string();
            assert!(e.contains("slabs for 3 columns"), "{e}");
        });
    }

    /// The bytes of a stack whose experts differ in direction, not only in
    /// scale: block `b` of expert `e` holds the quants `(13b + 5i + 7e) % 23 - 11`
    /// under one scale, so a norm of the experts' sum is not the sum of their
    /// norms.
    fn distinct(_name: &str, nbytes: usize) -> Vec<u8> {
        let blocks = nbytes / 34;
        let per_expert = blocks / EXPERTS;
        (0..blocks)
            .flat_map(|b| {
                let e = b / per_expert;
                let quants = (0..32).map(move |i| ((13 * b + 5 * i + 7 * e) % 23) as i8 - 11);
                [0x00u8, 0x24].into_iter().chain(quants.map(|q| q as u8))
            })
            .collect()
    }

    /// A layer with a post-norm serves, in all three entries, the sum of each
    /// listed expert's own down row normed and then weighted, in list order
    /// from zero — bit for bit the plain run's per-expert rows through the
    /// same norm, in a narrow union call and a wide one (its sums over the
    /// pool) — an empty list zero, and a layer without one the plain run's
    /// sum. FAIL-first: the norm of the weighted sum (or no norm) is
    /// far from the rows' (the experts differ in direction), so a post path
    /// that norms the aggregate or skips the norm fails here.
    #[test]
    fn a_post_norm_layer_norms_each_experts_row_before_its_weight() {
        const SET: [usize; 2] = [1, 2];
        // Columns of the wide call: past the narrow calls' columns.
        const WIDE: usize = model::ops::DEFER_MAX_COLS + 4;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let column = |c: usize| Tensor2 {
            ne0: WIDTH,
            ne1: 1,
            data: (0..WIDTH)
                .map(|i| (((i * 31 + c * 17) % 97) as f32 - 48.0) / 120.0)
                .collect(),
        };
        with_file_data_ff(
            &SET,
            |_| WIDTH,
            Some(&distinct),
            |file| {
                let build = |file: Arc<Split>| {
                    let mut run =
                        HostRun::build_at(file, SET, false, WIDTHS, |src| views(src, &SET))
                            .unwrap();
                    run.prepare_union(WIDE).unwrap();
                    run
                };
                let mut plain = build(Arc::clone(&file));
                let mut post = build(file);
                let gain: Vec<f32> = (0..WIDTH).map(|i| 0.5 + 0.01 * (i % 17) as f32).collect();
                let eps = 1e-8f32;
                post.set_post_norm(1, gain.clone(), eps).unwrap();
                let norm = PostNorm::new(gain, eps).unwrap();
                // The rule written out on the plain run's own rows.
                let want = |plain: &mut HostRun, layer: usize, x: &Tensor2, list: &[(u32, f32)]| {
                    let mut sum = vec![0.0f32; WIDTH];
                    let mut agg = vec![0.0f32; WIDTH];
                    for &(e, w) in list {
                        let mut row = vec![0.0f32; WIDTH];
                        plain
                            .experts_into(layer, x, &[(e, 1.0)], &mut row, 1)
                            .unwrap();
                        for (a, &d) in agg.iter_mut().zip(&row) {
                            *a += w * d;
                        }
                        norm.norm(&mut row);
                        for (o, &d) in sum.iter_mut().zip(&row) {
                            *o += w * d;
                        }
                    }
                    (sum, agg)
                };
                let (x0, x1) = (column(0), column(1));
                let l0: &[(u32, f32)] = &[(1, 0.75), (0, 0.25)];
                let l1: &[(u32, f32)] = &[(1, 0.5)];
                let (want0, agg0) = want(&mut plain, 1, &x0, l0);
                let (want1, _) = want(&mut plain, 1, &x1, l1);
                let far = |a: &[f32], b: &[f32]| {
                    let num: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(&a, &b)| f64::from(a - b).powi(2))
                        .sum();
                    let den: f64 = b.iter().map(|&b| f64::from(b).powi(2)).sum();
                    (num / den).sqrt()
                };
                let mut agg_normed = agg0;
                norm.norm(&mut agg_normed);
                assert!(
                    far(&agg_normed, &want0) > 1e-3,
                    "the experts' rows must differ in direction for the rule to tell its order from the aggregate's"
                );
                let mut plain_sum = vec![0.0f32; WIDTH];
                plain.experts_into(1, &x0, l0, &mut plain_sum, 1).unwrap();
                assert!(
                    far(&plain_sum, &want0) > 1e-3,
                    "the post-norm moves the plain sum"
                );

                let mut got = vec![7.0f32; WIDTH];
                post.experts_into(1, &x0, l0, &mut got, 1).unwrap();
                assert_eq!(bits(&got), bits(&want0), "the column call");
                let mut two = Tensor2 {
                    ne0: WIDTH,
                    ne1: 2,
                    data: x0.data.clone(),
                };
                two.data.extend_from_slice(&x1.data);
                let lists = [l0, l1];
                for union in [
                    HostExperts::experts_union_into,
                    HostExperts::experts_step_union_into,
                ] {
                    let mut out = vec![7.0f32; 2 * WIDTH];
                    union(&mut post, 1, two.view(), &lists, &mut out).unwrap();
                    assert_eq!(bits(&out[..WIDTH]), bits(&want0), "union column 0");
                    assert_eq!(bits(&out[WIDTH..]), bits(&want1), "union column 1");
                }
                // A call past the narrow ones' columns runs its sums over the
                // pool; every column is still the rule's.
                let wide = Tensor2 {
                    ne0: WIDTH,
                    ne1: WIDE,
                    data: (0..WIDE)
                        .flat_map(|j| if j % 2 == 0 { &x0.data } else { &x1.data })
                        .copied()
                        .collect(),
                };
                let wide_lists: Vec<&[(u32, f32)]> = (0..WIDE)
                    .map(|j| if j % 2 == 0 { l0 } else { l1 })
                    .collect();
                for union in [
                    HostExperts::experts_union_into,
                    HostExperts::experts_step_union_into,
                ] {
                    let mut out = vec![7.0f32; WIDE * WIDTH];
                    union(&mut post, 1, wide.view(), &wide_lists, &mut out).unwrap();
                    for (j, col) in out.as_chunks::<WIDTH>().0.iter().enumerate() {
                        let want = if j % 2 == 0 { &want0 } else { &want1 };
                        assert_eq!(bits(col), bits(want), "wide union column {j}");
                    }
                }
                let mut empty = vec![7.0f32; WIDTH];
                post.experts_into(1, &x0, &[], &mut empty, 1).unwrap();
                assert!(
                    empty.iter().all(|&v| v == 0.0),
                    "an empty list sums to zero"
                );
                let mut other = vec![7.0f32; WIDTH];
                let mut want_other = vec![7.0f32; WIDTH];
                post.experts_into(2, &x0, l0, &mut other, 1).unwrap();
                plain.experts_into(2, &x0, l0, &mut want_other, 1).unwrap();
                assert_eq!(
                    bits(&other),
                    bits(&want_other),
                    "a layer without a post-norm"
                );
                let e = post
                    .set_post_norm(0, vec![1.0; WIDTH], eps)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("layer 0 carries no host share"), "{e}");
                let e = post
                    .set_post_norm(1, vec![1.0; WIDTH - 1], eps)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("a gain of"), "{e}");
                let e = post
                    .set_post_norm(1, vec![1.0; WIDTH], 0.0)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("an epsilon of"), "{e}");
            },
        );
    }
}
