//! The trunk's layers read from the file's header: every weight's name, and the type of each
//! projection site checked against the shape the plan's kernels take. Nothing is uploaded and no
//! tensor byte is read; a layer the kernels do not run, a type no launch reads and a shape that
//! disagrees are refused by name with the layer.
//!
//! Each reader takes `what`, the loader's entry-point name its refusals are attributed to, and
//! `allowed`, the backend's own list of the site types its launches read.

use super::plan::{MambaPlan, NopeGqaPlan};
use crate::ChainError;
use crate::linear::{CONV_TAPS, HEAD};
use crate::site::{SiteTy, file_site};
use gguf::Split;
use model::arch::hybrid::{
    ATTN_K, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_A, SSM_CONV, SSM_CONV_BIAS, SSM_D,
    SSM_DT_BIAS, SSM_IN, SSM_NORM, SSM_OUT,
};
use model::arch::models::{Ffn, Gqa, Mamba2, OutGate, RopeMode};
use model::arch::names::blk;
use model::mamba2::{self, Geom};

/// Whether `ffn` is the description's "no feed-forward block" of a Mamba-2 or attention layer.
#[must_use]
pub fn no_block(ffn: &Ffn) -> bool {
    matches!(ffn, Ffn::Dense { ff: 0, .. })
}

/// Layer `l`'s Mamba-2 plan over a `hidden`-value row: the description's widths checked against
/// the scan's (state 128, a conv of four taps, widths that divide, an input projection of the
/// widths' columns), and the two projections' types from `file`'s header at one of `allowed`.
pub fn mamba_plan(
    file: &Split,
    what: &'static str,
    l: usize,
    m: &Mamba2,
    hidden: usize,
    allowed: &[SiteTy],
) -> Result<MambaPlan, ChainError> {
    let refuse = |d: String| ChainError::shape(what, format!("layer {l}: {d}"));
    let d_conv = m.d_conv as usize;
    let checked = mamba2::check(Geom {
        d_inner: m.d_inner as usize,
        n_heads: m.n_heads as usize,
        n_groups: m.n_groups as usize,
        d_state: m.d_state as usize,
        d_conv,
    })
    .map_err(|e| refuse(e.to_string()))?;
    if d_conv != CONV_TAPS {
        return Err(refuse(format!(
            "a conv of {d_conv} taps; the scan kernel runs {CONV_TAPS}"
        )));
    }
    if m.d_in_proj as usize != checked.in_proj_width() {
        return Err(refuse(format!(
            "an input projection of {} columns; the widths give {}",
            m.d_in_proj,
            checked.in_proj_width()
        )));
    }
    let d_inner = checked.geom().d_inner;
    let (ssm_in, ssm_out) = (blk(l, SSM_IN), blk(l, SSM_OUT));
    let conv_b = blk(l, SSM_CONV_BIAS);
    Ok(MambaPlan {
        norm: blk(l, ATTN_NORM),
        a: blk(l, SSM_A),
        d: blk(l, SSM_D),
        dt_bias: blk(l, SSM_DT_BIAS),
        conv_w: blk(l, SSM_CONV),
        conv_b: file.find(&conv_b).map(|_| conv_b),
        ssm_norm: blk(l, SSM_NORM),
        shape: *m,
        conv_dim: checked.conv_dim(),
        in_ty: file_site(
            file,
            what,
            &ssm_in,
            (checked.in_proj_width(), hidden),
            allowed,
        )?,
        out_ty: file_site(file, what, &ssm_out, (hidden, d_inner), allowed)?,
        ssm_in,
        ssm_out,
    })
}

/// Whether `g` is the attention the NoPE kernels run: heads of [`HEAD`] values with no rope, q/k
/// norm, output norm, output gate, key selection, window, sinks or value scale, and query heads a
/// multiple of the key heads.
fn nope_fits(g: &Gqa) -> Result<(), String> {
    let plain = g.rope.mode == RopeMode::None
        && !g.qk_norm
        && !g.o_norm
        && g.out_gate == OutGate::None
        && g.select.is_none()
        && g.window.is_none()
        && !g.sinks
        && g.value_scale.is_none()
        && g.head_dim as usize == HEAD
        && g.value_dim == g.head_dim;
    if !plain {
        return Err(format!(
            "attention {g:?}; the kernels take heads of {HEAD} values with no rope, q/k norm, \
             output norm, output gate, key selection, window, sinks or value scale"
        ));
    }
    if g.kv_heads == 0 || !g.heads.is_multiple_of(g.kv_heads) {
        return Err(format!(
            "{} query heads over {} key heads; the query heads are a multiple of the key heads",
            g.heads, g.kv_heads
        ));
    }
    Ok(())
}

/// Layer `l`'s attention plan over a `hidden`-value row, its four sites' types from `file`'s
/// header at one of `allowed`.
pub fn nope_gqa_plan(
    file: &Split,
    what: &'static str,
    l: usize,
    g: &Gqa,
    hidden: usize,
    allowed: &[SiteTy],
) -> Result<NopeGqaPlan, ChainError> {
    nope_fits(g).map_err(|d| ChainError::shape(what, format!("layer {l}: {d}")))?;
    let (n_head, n_kv) = (g.heads as usize, g.kv_heads as usize);
    let (q_rows, kv_rows) = (n_head * HEAD, n_kv * HEAD);
    let (q, k, v, o) = (
        blk(l, ATTN_Q),
        blk(l, ATTN_K),
        blk(l, ATTN_V),
        blk(l, ATTN_OUTPUT),
    );
    Ok(NopeGqaPlan {
        norm: blk(l, ATTN_NORM),
        n_head,
        n_kv,
        q_ty: file_site(file, what, &q, (q_rows, hidden), allowed)?,
        k_ty: file_site(file, what, &k, (kv_rows, hidden), allowed)?,
        v_ty: file_site(file, what, &v, (kv_rows, hidden), allowed)?,
        o_ty: file_site(file, what, &o, (hidden, q_rows), allowed)?,
        q,
        k,
        v,
        o,
    })
}

#[cfg(test)]
mod tests {
    use super::super::plan::{MambaPlan, NopeGqaPlan};
    use super::super::test_file::{decl, header};
    use super::{mamba_plan, no_block, nope_gqa_plan};
    use crate::site::SiteTy;
    use gguf::Split;
    use gguf::quant::GgmlType::{F32, Q4_K, Q5_0, Q8_0};
    use model::arch::hybrid::{
        ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_CONV_BIAS, SSM_IN, SSM_OUT,
    };
    use model::arch::models::{Act, Ffn, Gqa, Mamba2, OutGate, PoolRule, Rope, RopeMode, Selector};
    use model::arch::names::blk;

    const WHAT: &str = "test plan";
    const HIDDEN: usize = 256;
    const ALLOWED: &[SiteTy] = &[SiteTy::Q5_0, SiteTy::Q8_0];

    fn mamba() -> Mamba2 {
        Mamba2 {
            d_inner: 512,
            d_state: 128,
            n_heads: 8,
            head_dim: 64,
            n_groups: 2,
            d_conv: 4,
            d_in_proj: 1544,
        }
    }

    /// Layer `l`'s two Mamba-2 sites for `mamba()` over `HIDDEN` values, and its conv bias when
    /// `bias`.
    fn mamba_file(l: usize, bias: bool) -> Split {
        let mut t = vec![
            decl(blk(l, SSM_IN), &[HIDDEN as u64, 1544], Q5_0),
            decl(blk(l, SSM_OUT), &[512, HIDDEN as u64], Q8_0),
        ];
        if bias {
            t.push(decl(blk(l, SSM_CONV_BIAS), &[1024], F32));
        }
        header(t)
    }

    fn mamba_err(file: &Split, m: &Mamba2) -> String {
        mamba_plan(file, WHAT, 3, m, HIDDEN, ALLOWED)
            .err()
            .map_or_else(|| "read".to_string(), |e| e.to_string())
    }

    /// A Mamba-2 layer's plan holds the file's names, the types of its two sites, the conv width,
    /// and the conv bias only where the file holds one.
    #[test]
    fn a_mamba_plan_reads_its_names_types_and_optional_bias() {
        let want = |conv_b: Option<&str>| MambaPlan {
            norm: "blk.3.attn_norm.weight".into(),
            ssm_in: "blk.3.ssm_in.weight".into(),
            ssm_out: "blk.3.ssm_out.weight".into(),
            a: "blk.3.ssm_a".into(),
            d: "blk.3.ssm_d".into(),
            dt_bias: "blk.3.ssm_dt.bias".into(),
            conv_w: "blk.3.ssm_conv1d.weight".into(),
            conv_b: conv_b.map(Into::into),
            ssm_norm: "blk.3.ssm_norm.weight".into(),
            shape: mamba(),
            conv_dim: 1024,
            in_ty: SiteTy::Q5_0,
            out_ty: SiteTy::Q8_0,
        };
        for (bias, conv_b) in [(true, Some("blk.3.ssm_conv1d.bias")), (false, None)] {
            let got = mamba_plan(&mamba_file(3, bias), WHAT, 3, &mamba(), HIDDEN, ALLOWED)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, want(conv_b));
        }
    }

    /// What the scan kernel does not run, an input projection its widths do not give, a site of a
    /// type outside the backend's list, another shape and a missing tensor are each refused by name
    /// with the layer.
    #[test]
    fn a_mamba_layer_the_scan_does_not_run_is_refused_by_name() {
        let file = mamba_file(3, true);
        let with = |f: fn(&mut Mamba2)| {
            let mut m = mamba();
            f(&mut m);
            m
        };
        let state = mamba_err(&file, &with(|m| m.d_state = 96));
        assert!(
            state.starts_with("test plan: layer 3: ") && state.contains("state 96 is refused"),
            "{state}"
        );
        let conv = mamba_err(&file, &with(|m| m.d_conv = 3));
        assert_eq!(
            conv,
            "test plan: layer 3: a conv of 3 taps; the scan kernel runs 4"
        );
        let proj = mamba_err(&file, &with(|m| m.d_in_proj += 1));
        assert_eq!(
            proj,
            "test plan: layer 3: an input projection of 1545 columns; the widths give 1544"
        );
        let groups = mamba_err(&file, &with(|m| m.n_groups = 3));
        assert!(
            groups.contains("n_head 8 is not a multiple of n_group 3"),
            "{groups}"
        );

        let other_ty = header(vec![
            decl(blk(3, SSM_IN), &[HIDDEN as u64, 1544], Q4_K),
            decl(blk(3, SSM_OUT), &[512, HIDDEN as u64], Q8_0),
        ]);
        let ty = mamba_err(&other_ty, &mamba());
        assert!(
            ty.contains("blk.3.ssm_in.weight") && ty.contains("this site launches Q5_0, Q8_0"),
            "{ty}"
        );
        let other_shape = header(vec![
            decl(blk(3, SSM_IN), &[HIDDEN as u64, 1543], Q5_0),
            decl(blk(3, SSM_OUT), &[512, HIDDEN as u64], Q8_0),
        ]);
        let shape = mamba_err(&other_shape, &mamba());
        assert!(
            shape.contains("blk.3.ssm_in.weight") && shape.contains("takes 1544 rows of 256"),
            "{shape}"
        );
        let missing = mamba_err(&header(vec![]), &mamba());
        assert_eq!(missing, "test plan: blk.3.ssm_in.weight is not in the file");
    }

    fn gqa() -> Gqa {
        Gqa {
            heads: 10,
            kv_heads: 2,
            head_dim: 128,
            value_dim: 128,
            rope: Rope {
                mode: RopeMode::None,
                dims: 0,
                base: 0.0,
                yarn: None,
            },
            qk_norm: false,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window: None,
            sinks: false,
            value_scale: None,
        }
    }

    /// Layer `l`'s four attention sites for 10 heads over 2 key heads at `HIDDEN` values.
    fn gqa_file(l: usize) -> Split {
        let h = HIDDEN as u64;
        header(vec![
            decl(blk(l, ATTN_Q), &[h, 1280], Q5_0),
            decl(blk(l, ATTN_K), &[h, 256], Q5_0),
            decl(blk(l, ATTN_V), &[h, 256], Q8_0),
            decl(blk(l, ATTN_OUTPUT), &[1280, h], Q5_0),
        ])
    }

    fn gqa_err(g: &Gqa) -> String {
        nope_gqa_plan(&gqa_file(2), WHAT, 2, g, HIDDEN, ALLOWED)
            .err()
            .map_or_else(|| "read".to_string(), |e| e.to_string())
    }

    /// An attention plan holds the file's names, the heads and the four sites' types.
    #[test]
    fn a_nope_attention_plan_reads_its_names_heads_and_types() {
        let got = nope_gqa_plan(&gqa_file(2), WHAT, 2, &gqa(), HIDDEN, ALLOWED)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            got,
            NopeGqaPlan {
                norm: "blk.2.attn_norm.weight".into(),
                q: "blk.2.attn_q.weight".into(),
                k: "blk.2.attn_k.weight".into(),
                v: "blk.2.attn_v.weight".into(),
                o: "blk.2.attn_output.weight".into(),
                n_head: 10,
                n_kv: 2,
                q_ty: SiteTy::Q5_0,
                k_ty: SiteTy::Q5_0,
                v_ty: SiteTy::Q8_0,
                o_ty: SiteTy::Q5_0,
            }
        );
    }

    /// Each thing the NoPE kernels do not take (a rope, a q/k or output norm, an output gate, key
    /// selection, a window, sinks, a value scale, a head of another width, and key heads that do
    /// not divide the query heads) is refused by name with the layer.
    #[test]
    fn an_attention_the_nope_kernels_do_not_take_is_refused_by_name() {
        let kernels = "the kernels take heads of 128 values";
        type Edit = fn(&mut Gqa);
        let cases: [(&str, Edit); 11] = [
            ("rope", |g| g.rope.mode = RopeMode::Neox),
            ("qk_norm", |g| g.qk_norm = true),
            ("o_norm", |g| g.o_norm = true),
            ("packed gate", |g| g.out_gate = OutGate::Packed),
            ("projected gate", |g| g.out_gate = OutGate::Proj),
            ("selector", |g| {
                g.select = Some(Selector::TokenPool {
                    heads: 1,
                    d: 1,
                    top_k: 1,
                    pool: 1,
                    rule: PoolRule::Learned { key_eps: 1e-5 },
                });
            }),
            ("window", |g| g.window = Some(64)),
            ("sinks", |g| g.sinks = true),
            ("value_scale", |g| g.value_scale = Some(2.0)),
            ("head", |g| {
                g.head_dim = 64;
                g.value_dim = 64;
            }),
            ("value head", |g| g.value_dim = 64),
        ];
        for (name, edit) in cases {
            let mut g = gqa();
            edit(&mut g);
            let err = gqa_err(&g);
            assert!(
                err.starts_with("test plan: layer 2: attention ") && err.contains(kernels),
                "{name}: {err}"
            );
        }
        let mut g = gqa();
        g.kv_heads = 3;
        assert_eq!(
            gqa_err(&g),
            "test plan: layer 2: 10 query heads over 3 key heads; the query heads are a multiple \
             of the key heads"
        );
        g.heads = 0;
        g.kv_heads = 0;
        assert!(gqa_err(&g).contains("0 query heads over 0 key heads"));
    }

    /// An attention site of a type outside the backend's list, or of another shape, is refused by
    /// name with the site.
    #[test]
    fn an_attention_site_outside_the_list_or_shape_is_refused_by_name() {
        let h = HIDDEN as u64;
        let mut t = vec![
            decl(blk(2, ATTN_Q), &[h, 1280], Q4_K),
            decl(blk(2, ATTN_K), &[h, 256], Q5_0),
            decl(blk(2, ATTN_V), &[h, 256], Q8_0),
            decl(blk(2, ATTN_OUTPUT), &[1280, h], Q5_0),
        ];
        let err = nope_gqa_plan(&header(t.clone()), WHAT, 2, &gqa(), HIDDEN, ALLOWED)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("blk.2.attn_q.weight") && err.contains("this site launches Q5_0, Q8_0"),
            "{err}"
        );
        t[0] = decl(blk(2, ATTN_Q), &[h, 1279], Q5_0);
        let err = nope_gqa_plan(&header(t), WHAT, 2, &gqa(), HIDDEN, ALLOWED)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("blk.2.attn_q.weight") && err.contains("takes 1280 rows of 256"),
            "{err}"
        );
    }

    /// The description's "no feed-forward block" is a dense block of no values, whatever its
    /// activation; a block with values is not it.
    #[test]
    fn no_block_is_a_dense_block_of_no_values() {
        assert!(no_block(&Ffn::Dense {
            ff: 0,
            act: Act::ReluSqr
        }));
        assert!(!no_block(&Ffn::Dense {
            ff: 64,
            act: Act::ReluSqr
        }));
    }
}
