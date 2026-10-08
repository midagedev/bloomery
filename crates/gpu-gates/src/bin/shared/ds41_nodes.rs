//! The V4.1 step graph's node count, predicted from the file and the body: the one owner the step
//! gate's `--structure` clause and the DSpark loop gate's tapped counts read, so a count a gate
//! pinned for the real file is a value derived from the header and the file's tensor types.

use bloomery_gpu_deepseek41::body::Body;
use bloomery_gpu_gates::GateError;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::hparams::{CandidateRole, Hparams};
use model::arch::deepseek41::names;

/// Flag waits a step holds: the engram rows' arrival, one per step.
pub const ARRIVALS: usize = 1;

/// What [`predicted`] counted: the step's kernels by sub-layer and the memops, with the layer kinds
/// that make them.
pub struct Predicted {
    /// Per layer kind (its name), the attention's kernels, the MoE's and how many layers are of it.
    pub kinds: Vec<(String, usize, usize, usize)>,
    pub attn: usize,
    pub ffn: usize,
    pub glue: usize,
    pub kernels: usize,
    pub memops: usize,
}

impl Predicted {
    /// The table the step gate pins, one line a layer kind and the step's.
    pub fn print(&self) {
        for (name, a, f, n) in &self.kinds {
            println!(
                "predict kind={name} layers={n} attn_kernels={a} ffn_kernels={f} memops=2 \
                 nodes_per_layer={}",
                a + f + 2
            );
        }
        println!(
            "predict step: gather 1 + attn {} + ffn {} + glue {} = {} kernels, {} memops (go and \
             wait per layer, {ARRIVALS} rows wait), {ARRIVALS} memcpy (the rows' arrival), 0 other",
            self.attn, self.ffn, self.glue, self.kernels, self.memops
        );
    }

    /// The captured step's node count: kernels, memops and the rows' one copy.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.kernels + self.memops + ARRIVALS
    }
}

/// The step's predicted kernels: per layer the attention's (14, and on a
/// compressor's layer its kv projection, its gate and pool above ratio 1
/// or its row at ratio 1, and three for index keys; on an indexer layer
/// its two projections and the score and top-k passes, and without a
/// compressor the q8_1 of the normed input its weights read unless q_a
/// or kv, being K-quants, already had the norm leave it; two more on a
/// layer the candidate mask gives a role — the source's block keys and
/// selection, a consumer's compaction and remap; one more for the
/// q8_1 of the heads when wo_a is a K-quant and one for wo_a's when wo_b
/// is a q3_K/q4_K) and the MoE sub-layer's (its own count: ten with card
/// experts, seven without, one more for a q8_1 shared down projection);
/// the glue's (the broadcast, three and two per engram site and one more
/// for the looked-up rows' q8_1 when wkv is a K-quant, the collapse and
/// the head's four) and the gather — less one launch for each projection
/// a Q3_K row join folds into another's (`join_projections`). The types
/// come from `split`. Printed as the table G2 pins.
pub fn predicted(split: &Split, hp: &Hparams, body: &Body) -> Result<Predicted, GateError> {
    let ty = |name: String| {
        split
            .find(&name)
            .map(|(_, t)| t.ty)
            .ok_or_else(|| format!("{name} is not in the file"))
    };
    let kquant = |t: GgmlType| {
        matches!(
            t,
            GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
        )
    };
    let q8_1 = |t: GgmlType| matches!(t, GgmlType::Q3_K | GgmlType::Q4_K);
    let mut kinds: Vec<(String, usize, usize, usize)> = Vec::new();
    let mut attn_total = 0;
    let mut ffn_total = 0;
    for (l, k) in hp.layers.iter().enumerate() {
        let own = k
            .compressor
            .filter(|_| k.stream.is_some_and(|s| s.kv_source == l));
        let normed_q8_1 = q8_1(ty(names::attn_q_a(l))?) || q8_1(ty(names::attn_kv(l))?);
        // PIN(2026-09-24): `join_projections` folds each group of Q3_K projections of one
        // activation into one launch (ds41dense): kv and the indexer's weights into q_a's,
        // the indexer's query into q_b's, a gated compressor's gate into its kv.
        let q3k = |name: String| ty(name).map(|t| t == GgmlType::Q3_K);
        let proj_q3k = !k.indexer || q3k(names::indexer_proj(l))?;
        let join_qkv = q3k(names::attn_q_a(l))? && q3k(names::attn_kv(l))? && proj_q3k;
        let join_query = k.indexer && q3k(names::attn_q_b(l))? && q3k(names::indexer_attn_q_b(l))?;
        let join_kv_gate = own.is_some_and(|c| c.gated)
            && q3k(names::attn_compressor_kv(l))?
            && q3k(names::attn_compressor_gate(l))?;
        let folded = usize::from(join_qkv) * (1 + usize::from(k.indexer))
            + usize::from(join_query)
            + usize::from(join_kv_gate);
        let attn = 14 - folded
            + own.map_or(0, |c| 1 + if c.gated { 2 } else { 1 })
            + if k.index_keys { 3 } else { 0 }
            + match (k.indexer, own) {
                (false, _) => 0,
                (true, Some(_)) => 4,
                (true, None) if normed_q8_1 => 4,
                (true, None) => 5,
            }
            // PIN(2026-10-02): the candidate mask's launches (candwire): two on its
            // source and two on each consumer, 2 + 4 × 2 = 10 a step on the V4.1 file.
            + 2 * usize::from(hp.candidate_role(l).is_some())
            + usize::from(kquant(ty(names::attn_output_a(l))?))
            + usize::from(q8_1(ty(names::attn_output_b(l))?));
        let ffn = body
            .ffn_launches(l)
            .ok_or_else(|| format!("the ffn piece does not run layer {l}"))?;
        attn_total += attn;
        ffn_total += ffn;
        let name = format!(
            "{}{}{}{}{}",
            match (k.stream, own) {
                (None, _) => "window".to_string(),
                (Some(s), Some(_)) => format!("source-r{}", s.ratio),
                (Some(s), None) => format!("reader-r{}", s.ratio),
            },
            if k.indexer { "+indexer" } else { "" },
            match hp.candidate_role(l) {
                Some(CandidateRole::Source) => "+cand-source",
                Some(CandidateRole::Consumer) => "+cand-consumer",
                None => "",
            },
            if k.engram.is_some() { "+engram" } else { "" },
            if ffn > 7 { "+card" } else { "+host-only" }
        );
        match kinds.iter_mut().find(|e| e.0 == name) {
            Some(e) => e.3 += 1,
            None => kinds.push((name, attn, ffn, 1)),
        }
    }
    let sites = hp.engram()?.layer_ids.len();
    let mut wkv_q8_1 = 0;
    for &l in &hp.engram()?.layer_ids {
        wkv_q8_1 += usize::from(q8_1(ty(names::engram_wkv(l))?));
    }
    let glue = 1 + 3 * sites + wkv_q8_1 + 2 * sites + 1 + 4;
    let kernels = 1 + attn_total + ffn_total + glue;
    // PIN(2026-09-25): the engram rows reach the card after the launch — one flag wait and
    // one copy before the first site's token-only work (`RowsArrival`), where the step had
    // two batches per layer and no copy.
    let memops = 2 * hp.n_layer + ARRIVALS;
    Ok(Predicted {
        kinds,
        attn: attn_total,
        ffn: ffn_total,
        glue,
        kernels,
        memops,
    })
}
