//! The vision tensors a `mimovl` file must hold: every name of [`super::names`] once, at the shape
//! the hyperparameters give it and the type this crate reads, and nothing else of the vision half.
//!
//! The file is a bf16 export of llama.cpp's converter: matrices are bf16; the two patch kernels
//! are f32 (the converter keeps convolution weights out of the narrow type), narrowed to bf16 on
//! upload ([`OnCard::Bf16`]); biases, norm gains and the sinks are f32. A tensor of any other type
//! is refused by its name and type.
//!
//! The same file carries the audio tower's tensors (`a.*`, `mm.a.*`): [`check`] leaves those to the
//! audio tower and names none of them.

use gguf::GgmlType;

use super::{HEAD_DIM, Hparams, Mode, names};
use crate::VisionError;
use crate::arch::table;

/// How the card holds a tensor of the file, which its bytes depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnCard {
    /// As the file stores it.
    File,
    /// Narrowed to bf16 on upload (`arch::narrow`: refused unless every value round-trips).
    Bf16,
}

/// One tensor the file must hold, dims in ggml `ne[]` order (the contiguous axis first), and how
/// the card holds it.
pub type Expected = table::Expected<OnCard>;

/// The rows of the fused query/key/value projection: a head's values for every query head and for
/// each of the key and value heads.
#[must_use]
pub fn qkv_rows(hp: &Hparams) -> u64 {
    (HEAD_DIM * (hp.n_head + 2 * hp.n_kv_head)) as u64
}

/// The width of the heads together, the attention output's input width: not the model width.
#[must_use]
pub fn heads_width(hp: &Hparams) -> u64 {
    (HEAD_DIM * hp.n_head) as u64
}

/// Every vision tensor of the file with these hyperparameters, in the table's order.
#[must_use]
pub fn expected(hp: &Hparams) -> Vec<Expected> {
    let d = |n: usize| n as u64;
    let (dim, ff, p, out) = (d(hp.dim), d(hp.ff), d(hp.patch), d(hp.out_dim));
    let (qkv, heads, merged) = (qkv_rows(hp), heads_width(hp), dim * d(hp.merge * hp.merge));
    let mat = |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::BF16, OnCard::File);
    let vec = |name: String, n: u64| Expected::row(name, &[n], GgmlType::F32, OnCard::File);
    let narrowed =
        |name: String, dims: &[u64]| Expected::row(name, dims, GgmlType::F32, OnCard::Bf16);
    let mut all = vec![
        narrowed(names::patch_embd_weight(), &[p, p, 3, dim]),
        narrowed(names::patch_embd_weight_second_frame(), &[p, p, 3, dim]),
    ];
    for (b, mode) in hp.modes.iter().enumerate() {
        all.extend([
            vec(names::ln1_weight(b), dim),
            mat(names::attn_qkv_weight(b), &[dim, qkv]),
            vec(names::attn_qkv_bias(b), qkv),
            mat(names::attn_out_weight(b), &[heads, dim]),
            vec(names::attn_out_bias(b), dim),
        ]);
        if *mode != Mode::Full {
            all.push(vec(names::attn_sinks(b), d(hp.n_head)));
        }
        all.extend([
            vec(names::ln2_weight(b), dim),
            mat(names::ffn_gate_weight(b), &[dim, ff]),
            vec(names::ffn_gate_bias(b), ff),
            mat(names::ffn_up_weight(b), &[dim, ff]),
            vec(names::ffn_up_bias(b), ff),
            mat(names::ffn_down_weight(b), &[ff, dim]),
            vec(names::ffn_down_bias(b), dim),
        ]);
    }
    all.extend([
        vec(names::post_ln_weight(), dim),
        mat(names::mm0_weight(), &[merged, merged]),
        mat(names::mm2_weight(), &[merged, out]),
    ]);
    all
}

/// Whether a tensor of the file belongs to the audio tower.
fn is_audio(name: &str) -> bool {
    name.starts_with("a.") || name.starts_with("mm.a.")
}

/// Check a file's tensors (name, ggml dims, type) against [`expected`]: every vision one known,
/// none twice, each at its shape and type, and none of the table missing; the audio tower's are
/// skipped. Returns how many were named.
pub fn check<'a>(
    hp: &Hparams,
    tensors: impl IntoIterator<Item = (&'a str, &'a [u64], GgmlType)>,
) -> Result<usize, VisionError> {
    table::check(
        super::PROJECTOR_TYPE,
        &expected(hp),
        tensors.into_iter().filter(|(name, ..)| !is_audio(name)),
    )
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::{OnCard, check, expected, heads_width, qkv_rows};
    use crate::arch::mimovl::hparams::tests::hp;
    use crate::arch::table::testing::{changes_are_refused_by_name, list, text, triples};

    /// 12 tensors per block (5 bf16 matrices, 7 f32 vectors), 24 sinks on the windowed blocks, 2
    /// patch kernels and 3 after the blocks: the 365 of the vision half, 142 bf16 and 223 f32,
    /// 1,462,099,968 bytes of tensor data in the file and 1,458,167,808 on the card (the kernels
    /// narrowed to bf16).
    #[test]
    fn table_is_365() {
        let t = expected(&hp());
        assert_eq!(t.len(), 2 + 28 * 12 + 24 + 3);
        assert_eq!(t.len(), 365);
        let of = |ty| t.iter().filter(|e| e.ty == ty).count();
        assert_eq!(
            (of(GgmlType::BF16), of(GgmlType::F32)),
            (28 * 5 + 2, 28 * 7 + 24 + 1 + 2)
        );
        let file: u64 = t.iter().map(super::Expected::bytes).sum();
        let card: u64 = t
            .iter()
            .map(|e| match e.home {
                OnCard::File => e.bytes(),
                OnCard::Bf16 => e.dims.iter().product::<u64>() * 2,
            })
            .sum();
        assert_eq!((file, card), (1_462_099_968, 1_458_167_808));
        let narrowed: Vec<&str> = t
            .iter()
            .filter(|e| e.home == OnCard::Bf16)
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(narrowed, ["v.patch_embd.weight", "v.patch_embd.weight.1"]);
    }

    /// The heads are 64 wide and number 32 queries and 8 key/value pairs: the fused projection has
    /// 3,072 rows, the attention output reads 2,048 values, and the model is 1,280 wide.
    #[test]
    fn the_heads_are_not_the_model_width() {
        let hp = hp();
        assert_eq!((qkv_rows(&hp), heads_width(&hp)), (3072, 2048));
        let t = expected(&hp);
        let row = |n: &str| t.iter().find(|e| e.name == n).expect("row");
        assert_eq!(row("v.blk.0.attn_qkv.weight").dims, [1280, 3072]);
        assert_eq!(row("v.blk.0.attn_out.weight").dims, [2048, 1280]);
        assert_eq!(row("v.blk.3.attn_sinks").dims, [32]);
        assert_eq!(row("mm.0.weight").dims, [5120, 5120]);
        assert_eq!(row("mm.2.weight").dims, [5120, 4096]);
    }

    /// Sinks exist on the 24 windowed blocks and on none of the full ones (0, 9, 18, 27).
    #[test]
    fn sinks_are_on_the_windowed_blocks_only() {
        let t = expected(&hp());
        let sinks: Vec<usize> = (0..28)
            .filter(|b| t.iter().any(|e| e.name == format!("v.blk.{b}.attn_sinks")))
            .collect();
        let full = [0, 9, 18, 27];
        assert_eq!(sinks.len(), 24);
        assert!(sinks.iter().all(|b| !full.contains(b)));
        assert!(full.iter().all(|b| !sinks.contains(b)));
    }

    /// The table itself passes; one renamed tensor is refused by its new name, a dropped one is
    /// named as missing, a doubled one as twice, a reshaped one by its dims, a retyped one by its
    /// type.
    #[test]
    fn a_changed_tensor_list_is_refused_by_name() {
        let hp = hp();
        changes_are_refused_by_name(
            super::super::PROJECTOR_TYPE,
            &expected(&hp),
            365,
            "mm.2.weight",
            |l| text(check(&hp, triples(l))),
        );
    }

    /// The audio tower's tensors are not the vision check's: any number of `a.*` and `mm.a.*` rows
    /// pass and are not counted; a block's sink dropped is named, and a sink on a full block (the
    /// pattern of another network) is refused as no tensor of this projector.
    #[test]
    fn the_audio_half_is_skipped_and_a_wrong_sink_is_refused() {
        let hp = hp();
        let t = expected(&hp);
        let mut with_audio = list(&t);
        with_audio.push(("a.blk.0.ln1.weight".into(), vec![1024], GgmlType::F32));
        with_audio.push((
            "mm.a.mlp.1.weight".into(),
            vec![4096, 16384],
            GgmlType::BF16,
        ));
        assert_eq!(text(check(&hp, triples(&with_audio))), Ok(365));

        let mut no_sink = list(&t);
        let at = no_sink
            .iter()
            .position(|r| r.0 == "v.blk.5.attn_sinks")
            .expect("row");
        no_sink.remove(at);
        assert_eq!(
            text(check(&hp, triples(&no_sink))).unwrap_err(),
            "tensor v.blk.5.attn_sinks: is missing from the file"
        );
        let mut extra = list(&t);
        extra.push(("v.blk.9.attn_sinks".into(), vec![32], GgmlType::F32));
        assert_eq!(
            text(check(&hp, triples(&extra))).unwrap_err(),
            "tensor v.blk.9.attn_sinks: is not a mimovl tensor name"
        );
        let mut bias = list(&t);
        bias.push(("v.post_ln.bias".into(), vec![1280], GgmlType::F32));
        assert_eq!(
            text(check(&hp, triples(&bias))).unwrap_err(),
            "tensor v.post_ln.bias: is not a mimovl tensor name"
        );
    }
}
