//! The header test's shared half (`hw_*_spec` in the meta gates): a
//! [`ModelSpec`] rendered as literal lines — the model-wide fields, then one
//! line per run of layers that read alike — so a file's pins are a list of
//! strings, and the coverage check's items as lines.

use std::fmt::Write as _;

use model::arch::models::{
    Act, Compress, Ffn, IndexKeys, LayerSpec, Mixer, ModelSpec, OutGate, Residual, Rope, Selector,
    Source,
};
use model::placement::Unimplemented;

fn src<T>(s: &Source<T>, inner: impl Fn(&T) -> String) -> String {
    match s {
        Source::Own(t) => format!("Own{}", inner(t)),
        Source::From(l) => format!("From{l}"),
    }
}

fn rope(r: &Rope) -> String {
    // The net magnitude joins only where it leaves 1, so every attached
    // model's pins keep theirs.
    let magnitude = match r.yarn {
        Some(y) if y.magnitude != 1.0 => format!(" magnitude {}", y.magnitude),
        _ => String::new(),
    };
    let yarn = r.yarn.map_or("-".to_string(), |y| {
        format!(
            "{}/{}/{}/{}",
            y.factor, y.orig_ctx, y.beta_fast, y.beta_slow
        )
    });
    // The factors join the text only where the rope has them, so every attached model's pins keep
    // theirs.
    let factors = if r.factors { " factors" } else { "" };
    format!(
        "{:?} {} base {} yarn {yarn}{factors}{magnitude}",
        r.mode, r.dims, r.base
    )
}

fn act(a: &Act) -> String {
    match a {
        Act::SwiGlu { limit } => format!("swiglu {limit:?}"),
        Act::ReluSqr => "relusqr".to_string(),
        Act::GeGlu => "geglu".to_string(),
        Act::SwigluOai { alpha, limit } => format!("swigluoai {alpha}/{limit}"),
        Act::Situ {
            gate_beta,
            value_beta,
        } => format!("situ {gate_beta}/{value_beta}"),
    }
}

/// A layer's position selector, appended to its mixer's part of the line.
fn select(o: &mut String, select: Option<&Selector>) {
    match select {
        Some(Selector::StreamTopK {
            heads,
            d,
            k,
            keys,
            list,
            candidates,
        }) => {
            let _ = write!(
                o,
                " | sel {heads}x{d} k{k} keys {} list {} cand {candidates:?}",
                src(keys, |k| match k {
                    IndexKeys::FromRows => String::new(),
                    IndexKeys::Compressor(c) => format!("{{icmp {c:?}}}"),
                    IndexKeys::FromInput { .. } => " {input}".to_string(),
                }),
                src(list, |()| String::new())
            );
        }
        Some(t @ (Selector::TokenPool { .. } | Selector::BlockMax { .. })) => {
            let _ = write!(o, " | {t:?}");
        }
        None => {}
    }
}

/// One layer as one line.
pub fn layer_line(layer: &LayerSpec) -> String {
    let mut o = String::new();
    match &layer.mixer {
        Mixer::Gqa(g) => {
            // The value width joins the line only where it leaves the head's:
            // every attached model's pins keep their text.
            let v = if g.value_dim != g.head_dim {
                format!(" v {}", g.value_dim)
            } else {
                String::new()
            };
            let _ = write!(
                o,
                "gqa {}/{} x {}{v} rope {} qk_norm {} out_gate {:?}",
                g.heads,
                g.kv_heads,
                g.head_dim,
                rope(&g.rope),
                g.qk_norm,
                g.out_gate
            );
            // The window, sinks and value scale join the line only where set,
            // for the same reason.
            if let Some(w) = g.window {
                let _ = write!(o, " win {w}");
            }
            if g.sinks {
                o.push_str(" sinks");
            }
            if let Some(x) = g.value_scale {
                let _ = write!(o, " vscale {x}");
            }
            select(&mut o, g.select.as_ref());
        }
        Mixer::Latent(a) => {
            let _ = write!(
                o,
                "latent h {} q {} kv {} {:?} rope {} qhn {} out {:?} win {:?} sinks {}",
                a.heads,
                a.q_lora,
                a.latent,
                a.up,
                a.rope.as_ref().map_or("-".to_string(), rope),
                a.q_head_norm,
                a.out,
                a.window,
                a.sinks
            );
            if a.pe > 0 || a.out_gate != OutGate::None {
                let _ = write!(o, " pe {} gate {:?}", a.pe, a.out_gate);
            }
            // The SSMax flag and a shared cache join the line only where set,
            // for the same reason: every attached model's pins keep their text.
            if a.ssmax {
                o.push_str(" ssmax");
            }
            if let models::Source::From(l) = a.cache {
                let _ = write!(o, " cache<-{l}");
            }
            if let Some(Compress { ratio, rows }) = &a.compress {
                let _ = write!(
                    o,
                    " | cmp r{ratio} {}",
                    src(rows, |c| format!(
                        "{{g{} a{} o{}}}",
                        u8::from(c.gated),
                        u8::from(c.ape),
                        u8::from(c.overlap)
                    ))
                );
            }
            select(&mut o, a.select.as_ref());
        }
        Mixer::DeltaRule(r) => {
            let _ = write!(
                o,
                "delta {:?} k {} v {} d {} conv {}",
                r.kind, r.k_heads, r.v_heads, r.d, r.conv
            );
        }
        Mixer::Mamba2(m) => {
            let _ = write!(
                o,
                "mamba2 inner {} state {} heads {} dim {} groups {} conv {} proj {}",
                m.d_inner, m.d_state, m.n_heads, m.head_dim, m.n_groups, m.d_conv, m.d_in_proj
            );
        }
        Mixer::ShortConv(c) => {
            let _ = write!(o, "shortconv taps {}", c.taps);
        }
        Mixer::None => o.push_str("none"),
    }
    match &layer.ffn {
        Ffn::Moe(m) => {
            let r = m.router;
            let _ = write!(
                o,
                " || moe {}/{} ff {} {} {:?} bias {} norm {} x{} hash {}",
                m.experts,
                m.top_k,
                m.expert_ff,
                act(&m.act),
                r.score,
                r.bias,
                match r.norm {
                    models::RouterNorm::None => "false",
                    models::RouterNorm::SelectedSum => "true",
                    models::RouterNorm::SelectedSumFloor => "max(selected sum, 2^-14)",
                },
                r.scale,
                r.hash
            );
            if let Some(s) = m.shared {
                let _ = write!(
                    o,
                    " shared {} {} gate {}",
                    s.ff,
                    act(&s.act),
                    s.sigmoid_gate
                );
            }
        }
        Ffn::Dense { ff, act: a } => {
            if *ff == 0 && matches!(a, Act::ReluSqr) {
                o.push_str(" || none");
            } else {
                let _ = write!(o, " || dense {ff} {}", act(a));
            }
        }
    }
    let _ = write!(
        o,
        " || {}{}",
        match layer.residual {
            Residual::Plain => "plain",
            Residual::Hc => "hc",
            Residual::AlphaBeta { .. } => "alpha-beta",
            Residual::Sandwich { .. } => "sandwich",
        },
        if layer.extras.is_empty() {
            String::new()
        } else {
            format!(" {:?}", layer.extras)
        }
    );
    o
}

/// `layers` as the compact list `2-7,9`.
pub fn ranges_of(layers: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < layers.len() {
        let mut j = i;
        while j + 1 < layers.len() && layers[j + 1] == layers[j] + 1 {
            j += 1;
        }
        out.push(if i == j {
            layers[i].to_string()
        } else {
            format!("{}-{}", layers[i], layers[j])
        });
        i = j + 1;
    }
    out.join(",")
}

/// Distinct layer lines, each with the layers that read it, in first-seen order.
pub fn layer_lines(layers: &[LayerSpec]) -> Vec<String> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        let line = layer_line(layer);
        match groups.iter_mut().find(|(g, _)| *g == line) {
            Some((_, ls)) => ls.push(l),
            None => groups.push((line, vec![l])),
        }
    }
    groups
        .into_iter()
        .map(|(line, ls)| format!("[{}] {line}", ranges_of(&ls)))
        .collect()
}

/// The model as its pins read: the model-wide fields, then the layer lines.
pub fn view(s: &ModelSpec) -> Vec<String> {
    let mut v = vec![
        format!("arch {:?}", s.arch),
        format!(
            "hidden {} vocab {} ctx_train {}",
            s.hidden, s.vocab, s.ctx_train
        ),
        format!("rms_eps bits {:#010x}", s.rms_eps.to_bits()),
        format!("layers {} mtp {}", s.layers.len(), s.mtp.len()),
        format!("hc {:?}", s.hc),
        format!("engram {:?}", s.engram),
        format!(
            "chat pre {} template bytes {} tools {:?} reasoning {:?}",
            s.chat.pre,
            s.chat.template.as_ref().map_or(0, String::len),
            s.chat.tools,
            s.chat.reasoning
        ),
    ];
    v.extend(layer_lines(&s.layers));
    v
}

/// Each line of `got` against `want`, printed; a difference (a line missing,
/// a line added) is an entry of `bad`.
pub fn compare(out: &mut String, bad: &mut Vec<String>, what: &str, got: &[String], want: &[&str]) {
    let _ = writeln!(out, "{what}");
    for g in got {
        let mark = if want.contains(&g.as_str()) {
            "ok "
        } else {
            "BAD"
        };
        let _ = writeln!(out, "  {mark} {g}");
        if mark == "BAD" {
            bad.push(format!("{what}: unpinned line {g}"));
        }
    }
    for w in want {
        if !got.iter().any(|g| g == w) {
            bad.push(format!("{what}: pinned line missing {w}"));
        }
    }
}

/// The check's items as `feature: layers` lines.
pub fn items(list: &[Unimplemented]) -> Vec<String> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for f in list {
        match groups.iter_mut().find(|(g, _)| *g == f.feature) {
            Some((_, ls)) => ls.extend(f.layer),
            None => groups.push((f.feature.clone(), f.layer.into_iter().collect())),
        }
    }
    groups
        .into_iter()
        .map(|(f, ls)| {
            if ls.is_empty() {
                f
            } else {
                format!("{f}: {}", ranges_of(&ls))
            }
        })
        .collect()
}
