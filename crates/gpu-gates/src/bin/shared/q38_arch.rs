//! The one piece of `q38_fixture` that reads the Qwen3.8 family's layer kinds off the file's
//! description. It sits beside the gates that read it; the library module keeps the family-free
//! formulas and the tier helpers.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::q38_fixture::{ByName, Head, Shape};
use gguf::{GgmlType, Split};
use model::arch::models::{Extra, Mixer, ModelSpec, NgramRule};
use model::placement::kernels::{self, HeadEntry, SiteAct};

/// The shape of a file's description: its layers, the attention layers among them, the PLE
/// site's layer and the image id the site hashes.
fn layers_of(spec: &ModelSpec) -> Result<Shape, GateError> {
    let attention: Vec<usize> = spec
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| matches!(l.mixer, Mixer::Gqa(_)))
        .map(|(layer, _)| layer)
        .collect();
    if attention.is_empty() {
        return Err("the description holds no attention layer".into());
    }
    let ple_layer = spec
        .layers
        .iter()
        .position(|l| l.extras.contains(&Extra::Ple));
    let image = spec.engram.and_then(|e| match e.rule {
        NgramRule::Ple { image, .. } => image,
        NgramRule::Engram => None,
    });
    Shape::from_layers(spec.layers.len(), attention, ple_layer, image)
}

/// The shape of `file`: the layers its description names (the attention layers among them, the
/// PLE site's layer and the image id the site hashes), with the head and the dense sites counted
/// by its tensor types, read from its header. A K-quant `output` plane the common head reads
/// ([`kernels::head`]) quantizes its input rows to q8_1 first, and a Q5_K one, whose gemv writes
/// token-major, copies its logits into the argmax's layout past one row; each dense site that
/// reads q8_1 rows adds launches too ([`dense_launches`]).
///
/// # Errors
/// A file with no layer or no attention layer, or a PLE site past the layers.
pub fn shape_of(spec: &ModelSpec, file: &Split) -> Result<Shape, GateError> {
    let entry = file
        .find("output.weight")
        .and_then(|(_, t)| kernels::head(t.ty));
    let head = Head {
        quantizes: matches!(
            entry,
            Some(HeadEntry::Q6k | HeadEntry::Q5k | HeadEntry::Q4k)
        ),
        copies_rows: matches!(entry, Some(HeadEntry::Q5k)),
    };
    let shape = layers_of(spec)?.with_head(head);
    let by_name = dense_launches(&shape, file);
    Ok(shape.with_by_name(by_name))
}

/// The extra launches of the file's dense sites over what a site of f32 rows makes, from its
/// header's types by the rule the engine's walks follow: a type whose dense entry reads q8_1
/// rows (the K-quants, NVFP4) quantizes its input first, and past one row Q3_K, Q4_K and Q6_K,
/// whose gemvs are row-major, copy the result token-major. The sites are the delta layer's
/// q·k·v, z and output, the selecting layer's q, k, v, output and the indexer's key and query,
/// every layer's shared expert and the PLE site's key and value; the joined β·α stack and the
/// router read f32 rows.
#[must_use]
fn dense_launches(shape: &Shape, file: &Split) -> ByName {
    use model::arch::qwen35moe::names as n;
    let extra = |name: &str, m: usize| -> usize {
        let Some((_, t)) = file.find(name) else {
            return 0;
        };
        match kernels::dense(t.ty) {
            Some(e) if e.reads() == SiteAct::Q8x128 => {
                let row_major = matches!(t.ty, GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q6_K);
                1 + usize::from(m > 1 && row_major)
            }
            _ => 0,
        }
    };
    let mut by = ByName::default();
    for l in 0..shape.n_layer {
        let mut names = if shape.qsa.contains(&l) {
            vec![
                n::attn_q(l),
                n::attn_k(l),
                n::attn_v(l),
                n::attn_output(l),
                n::indexer_k_proj(l),
                n::indexer_q_proj(l),
            ]
        } else {
            vec![n::attn_qkv(l), n::attn_gate(l), n::ssm_out(l)]
        };
        names.extend([
            n::ffn_gate_shexp(l),
            n::ffn_up_shexp(l),
            n::ffn_down_shexp(l),
        ]);
        if shape.ple_layer == Some(l) {
            names.extend([n::ple_key(l), n::ple_value(l)]);
        }
        for name in &names {
            by.one += extra(name, 1);
            by.rows += extra(name, 2);
        }
    }
    by
}
