//! The entry of the common card tables each `mistral-large4` tensor is read through. The family
//! states which table a tensor goes to — an architecture fact: `attn_v_b` is a stack of the heads
//! read as experts, `attn_k_b` a per-head gemv — and the file's type picks the entry, tensor by
//! tensor and layer by layer, from the tables themselves (`placement::kernels`), so this file
//! holds no list of types. A tensor whose type has no entry is refused by name here, at load,
//! before any step runs; the body's launches and the header gate ask this one function.

use super::names;
use crate::placement::kernels::{
    self, DownAct, DownEntry, EmbedEntry, GateUpEntry, HeadEntry, SiteEntry,
};
use gguf::GgmlType;

/// How the body reads a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reader {
    /// Read as f32 values: a norm's gain, an SSMax pair, the router's rows, the rope's factors.
    F32,
    /// A dense gemv site: a projection, the indexer's, the dense lead's and the shared expert's.
    Dense,
    /// `attn_k_b`: the per-head gemv that absorbs the query into the latent width.
    KB,
    /// `attn_v_b`: the per-head up-projection of the latent to the head's values.
    VB,
    /// `token_embd`: the embedding rows.
    Embed,
    /// `output`: the lm_head.
    Head,
    /// A routed gate or up stack.
    GateUp,
    /// A routed down stack.
    Down,
}

/// How `attn_v_b` up-projects the heads' latent outputs, by the file's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpProj {
    /// Q8_0 planes through the per-head q8_0 gemv, the heads' rows of one weight.
    Planes,
    /// The heads as the experts of one stack, ids `0..heads`, through the down `_sel` entry the
    /// table names for the type, over one q8_1 column of the latent a head.
    Sel(SelUp),
}

/// The down `_sel` launch family a v_b stack runs, the table's [`DownEntry`] grouped by the
/// launcher that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelUp {
    /// `Q4kSelKernels::enqueue_gemv_q4k_sel`.
    Q4k,
    /// `QuantSelKernels::enqueue_gemv_q5k_sel`.
    Q5k,
    /// `Q6kSelKernels::enqueue_gemv_q6k_sel`.
    Q6k,
    /// `QuantSelKernels::enqueue_gemv_sel`, which takes the i-quant's file type.
    IQuant,
}

/// What launches a tensor of the file's type: the table's entry, or the family of launches a
/// tensor the tables read through more than one entry takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Launch {
    /// Read as f32 values.
    F32,
    /// A dense gemv site's entry ([`kernels::dense`]).
    Dense(SiteEntry),
    /// `attn_k_b`: the per-head gemv over Q8_0 planes.
    KUp,
    /// `attn_v_b`.
    VUp(UpProj),
    /// The embedding rows' entry ([`kernels::embed`]).
    Embed(EmbedEntry),
    /// The lm_head's entry ([`kernels::head`]).
    Head(HeadEntry),
    /// A routed stack's gate·up `_sel` entry ([`kernels::gate_up`]).
    GateUp(GateUpEntry),
    /// A routed stack's down `_sel` entry ([`kernels::down`]).
    Down(DownEntry),
}

/// The launch family of down entry `entry`, `None` for an entry no per-head launch runs: the
/// 32-value-form entries (Q5_0, Q5_1, Q8_0 in the file's blocks, IQ4_NL, NVFP4) and MXFP4's.
const fn sel_up(entry: DownEntry) -> Option<SelUp> {
    match entry {
        DownEntry::Q4k => Some(SelUp::Q4k),
        DownEntry::Q5k => Some(SelUp::Q5k),
        DownEntry::Q6k => Some(SelUp::Q6k),
        DownEntry::Iq2xxs
        | DownEntry::Iq2xs
        | DownEntry::Iq2s
        | DownEntry::Iq1s
        | DownEntry::Iq1m
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4xs => Some(SelUp::IQuant),
        DownEntry::Q5_0
        | DownEntry::Q5_1
        | DownEntry::Q8_0
        | DownEntry::Iq4nl
        | DownEntry::Mxfp4
        | DownEntry::Nvfp4 => None,
    }
}

/// How `attn_k_b` of file type `ty` is read: the per-head gemv over Q8_0 planes and nothing else,
/// for a head's K is 64 or 128 values, which no 256-value block type divides, and no per-head
/// launch reads a 32-value one. `name` is the tensor, named by the refusal.
pub fn k_b(name: &str, ty: GgmlType) -> Result<(), String> {
    if ty == GgmlType::Q8_0 {
        return Ok(());
    }
    Err(format!(
        "{name} is {ty}: the per-head k_b gemv reads Q8_0 planes only (a head's K is 64 or 128 \
         values, which no 256-value block type divides)"
    ))
}

/// How `attn_v_b` of file type `ty` is read: Q8_0 planes through the per-head gemv, or any type
/// whose table entry is a down `_sel` over q8_1 columns of 128 values (a head's K is the latent,
/// two whole super-blocks) that a per-head launch runs ([`SelUp`]), the heads as its experts.
/// Another type is refused by name.
pub fn v_b(name: &str, ty: GgmlType) -> Result<UpProj, String> {
    if ty == GgmlType::Q8_0 {
        return Ok(UpProj::Planes);
    }
    match kernels::down(ty) {
        Some((entry, DownAct::Q8Act)) => sel_up(entry).map(UpProj::Sel).ok_or_else(|| {
            format!("{name} is {ty}: no per-head launch runs its down entry {entry:?}")
        }),
        Some((entry, DownAct::Q8Blocks32)) => Err(format!(
            "{name} is {ty}: its down entry {entry:?} reads q8_1 columns of 32 values, which no \
             per-head launch runs"
        )),
        None => Err(format!(
            "{name} is {ty}: the per-head v_b reads Q8_0 planes or a down `_sel` entry, and the \
             table names none for {ty}"
        )),
    }
}

/// The reader of the tensor `name` of the file: its stem after the layer's `blk.{l}.`, or the
/// model-level tensor itself.
fn reader(name: &str) -> Option<Reader> {
    if name == names::token_embd() {
        return Some(Reader::Embed);
    }
    if name == names::output() {
        return Some(Reader::Head);
    }
    if name == names::output_norm() || name == crate::arch::names::rope_freqs() {
        return Some(Reader::F32);
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    layer.parse::<usize>().ok()?;
    Some(match stem {
        names::ATTN_NORM
        | names::ATTN_KV_A_NORM
        | names::ATTN_SSMAX_W
        | names::ATTN_SSMAX_B
        | names::INDEXER_Q_NORM_W
        | names::INDEXER_Q_NORM_B
        | names::INDEXER_K_NORM_W
        | names::INDEXER_K_NORM_B
        | names::INDEXER_SSMAX_W
        | names::INDEXER_SSMAX_B
        | names::FFN_NORM
        | names::FFN_GATE_INP => Reader::F32,
        names::ATTN_Q
        | names::ATTN_KV_A_MQA
        | names::ATTN_OUTPUT
        | names::INDEXER_ATTN_Q_B
        | names::INDEXER_ATTN_K
        | names::INDEXER_PROJ
        | names::FFN_GATE
        | names::FFN_UP
        | names::FFN_DOWN
        | names::FFN_GATE_SHEXP
        | names::FFN_UP_SHEXP
        | names::FFN_DOWN_SHEXP => Reader::Dense,
        names::ATTN_K_B => Reader::KB,
        names::ATTN_V_B => Reader::VB,
        names::FFN_GATE_EXPS | names::FFN_UP_EXPS => Reader::GateUp,
        names::FFN_DOWN_EXPS => Reader::Down,
        _ => return None,
    })
}

/// What launches the tensor `name` of file type `ty`: the entry of the table its reader asks, or
/// a refusal that names the tensor, its type and what the table lacks. A tensor no reader of the
/// family names is refused too.
pub fn launch(name: &str, ty: GgmlType) -> Result<Launch, String> {
    let Some(reader) = reader(name) else {
        return Err(format!(
            "{name}: no reader of mistral-large4 reads this tensor"
        ));
    };
    let lacks =
        |table: &str| format!("{name} is {ty}: the common {table} table has no entry for it");
    match reader {
        Reader::F32 if ty == GgmlType::F32 => Ok(Launch::F32),
        Reader::F32 => Err(format!("{name} is {ty}: the family reads it as f32 values")),
        Reader::Dense => kernels::dense(ty)
            .map(Launch::Dense)
            .ok_or_else(|| lacks("dense site")),
        Reader::KB => k_b(name, ty).map(|()| Launch::KUp),
        Reader::VB => v_b(name, ty).map(Launch::VUp),
        Reader::Embed => kernels::embed(ty)
            .map(Launch::Embed)
            .ok_or_else(|| lacks("embedding")),
        Reader::Head => kernels::head(ty)
            .map(Launch::Head)
            .ok_or_else(|| lacks("head")),
        Reader::GateUp => kernels::gate_up(ty)
            .map(Launch::GateUp)
            .ok_or_else(|| lacks("gate·up `_sel`")),
        Reader::Down => kernels::down(ty)
            .map(|(entry, _)| Launch::Down(entry))
            .ok_or_else(|| lacks("down `_sel`")),
    }
}

#[cfg(test)]
mod tests {
    use super::{Launch, SelUp, UpProj, k_b, launch, v_b};
    use crate::placement::kernels::{
        self, DownAct, DownEntry, EmbedEntry, GateUpEntry, HeadEntry, Q8_ACT_MAX_K, SiteAct,
        SiteEntry,
    };
    use gguf::GgmlType;

    /// The header tables of the UD-IQ1_S file and the Mix-256G file: every tensor of each, its
    /// name, type, dims and shard.
    const UD: &str = include_str!("../../../tests/mistrallarge4_tensors.tsv");
    const MIX: &str = include_str!("../../../tests/mistrallarge4_mix256_tensors.tsv");

    /// A ggml type by its table's name (the tables spell the ids in capitals, `ggml_type_name`
    /// in mixed case).
    fn ty_of(name: &str) -> GgmlType {
        (0..=45)
            .chain(2000..=2005)
            .map(GgmlType::from_u32)
            .find(|ty| ty.name().is_some_and(|n| n.eq_ignore_ascii_case(name)))
            .unwrap_or_else(|| panic!("{name}: no ggml type of this name"))
    }

    /// The `(name, type)` of every tensor of a table.
    fn rows(table: &str) -> Vec<(&str, GgmlType)> {
        table
            .lines()
            .map(|line| {
                let mut cells = line.split('\t');
                match (cells.next(), cells.next()) {
                    (Some(name), Some(ty)) => (name, ty_of(ty)),
                    _ => panic!("{line}: not a name and a type"),
                }
            })
            .collect()
    }

    /// The `(name, type, K)` of every tensor of a table, K the first dim of its shape.
    fn rows_k(table: &str) -> Vec<(&str, GgmlType, usize)> {
        table
            .lines()
            .map(|line| {
                let cells: Vec<&str> = line.split('\t').collect();
                let k = cells.get(2).and_then(|dims| dims.split('x').next());
                match (cells.first(), cells.get(1), k.map(str::parse::<usize>)) {
                    (Some(name), Some(ty), Some(Ok(k))) => (*name, ty_of(ty), k),
                    _ => panic!("{line}: not a name, a type and a shape"),
                }
            })
            .collect()
    }

    /// The launch of the tensor `name` of a table.
    fn of(table: &str, name: &str) -> Result<Launch, String> {
        let ty = rows(table)
            .into_iter()
            .find_map(|(n, ty)| (n == name).then_some(ty))
            .ok_or_else(|| format!("{name}: not in the table"))?;
        launch(name, ty)
    }

    /// Every dense site and down stack of both files' headers that reads a q8_1 column has a K
    /// the activation holds: the dense lead's FFN down of Mix-256G is a Q5_K site of 24,576.
    /// Mutant: the cap left at 22,016.
    #[test]
    fn every_q8_1_site_of_both_files_fits_the_activation_cap() {
        let mut bad = Vec::new();
        for (file, table) in [("UD-IQ1_S", UD), ("Mix-256G", MIX)] {
            for (name, ty, k) in rows_k(table) {
                let reads_q8_1 = match launch(name, ty) {
                    Ok(Launch::Dense(site)) => site.reads_at(k as u64) == SiteAct::Q8x128,
                    Ok(Launch::Down(_)) => {
                        matches!(kernels::down(ty), Some((_, DownAct::Q8Act)))
                    }
                    Ok(Launch::VUp(UpProj::Sel(_))) => true,
                    _ => false,
                };
                if reads_q8_1 && k > Q8_ACT_MAX_K {
                    bad.push(format!(
                        "{file}: {name} ({ty}) reads q8_1 columns of {k} values, past the \
                         activation's {Q8_ACT_MAX_K}"
                    ));
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// Every tensor of both files' headers resolves to an entry of the common tables, one
    /// refusal a tensor otherwise. Mutant: v_b's Q5_K down entry left out of `sel_up`.
    #[test]
    fn every_tensor_of_both_files_resolves_to_an_entry() {
        let mut bad = Vec::new();
        for (file, table) in [("UD-IQ1_S", UD), ("Mix-256G", MIX)] {
            let rows = rows(table);
            if rows.len() != 1602 {
                bad.push(format!(
                    "{file}: {} tensors, the header has 1602",
                    rows.len()
                ));
            }
            for (name, ty) in rows {
                if let Err(e) = launch(name, ty) {
                    bad.push(format!("{file}: {e}"));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} tensor(s) resolve to no entry:\n  {}",
            bad.len(),
            bad.join("\n  ")
        );
    }

    /// The entry each file's tensors take, at the sites the files differ in: UD-IQ1_S reads its
    /// Q6_K projections, Q8_0 up-projection planes and Q6_K head and embedding; Mix-256G its
    /// IQ4_XS query, Q4_K output, Q5_K v_b as a stack of the heads, Q5_K head, Q3_K embedding,
    /// mixed shared expert and Q5_K dense lead. Mutant: `Reader::Head` read through the embedding
    /// table.
    #[test]
    fn each_file_reads_through_its_own_entries() {
        let mut bad = Vec::new();
        for (file, table, want) in [
            (
                "UD-IQ1_S",
                UD,
                vec![
                    ("blk.1.attn_q.weight", Launch::Dense(SiteEntry::Q6k)),
                    ("blk.1.attn_output.weight", Launch::Dense(SiteEntry::Q6k)),
                    ("blk.1.attn_v_b.weight", Launch::VUp(UpProj::Planes)),
                    ("blk.1.attn_k_b.weight", Launch::KUp),
                    ("blk.1.ffn_up_shexp.weight", Launch::Dense(SiteEntry::Q8_0)),
                    ("blk.0.ffn_down.weight", Launch::Dense(SiteEntry::Q8_0)),
                    ("output.weight", Launch::Head(HeadEntry::Q6k)),
                    ("token_embd.weight", Launch::Embed(EmbedEntry::Q6k)),
                    (
                        "blk.1.ffn_gate_exps.weight",
                        Launch::GateUp(GateUpEntry::Iq2xxs),
                    ),
                    (
                        "blk.1.ffn_down_exps.weight",
                        Launch::Down(DownEntry::Iq2xxs),
                    ),
                    (
                        "blk.17.ffn_gate_exps.weight",
                        Launch::GateUp(GateUpEntry::Iq1s),
                    ),
                ],
            ),
            (
                "Mix-256G",
                MIX,
                vec![
                    ("blk.1.attn_q.weight", Launch::Dense(SiteEntry::Iq4xs)),
                    ("blk.1.attn_output.weight", Launch::Dense(SiteEntry::Q4k)),
                    (
                        "blk.1.attn_v_b.weight",
                        Launch::VUp(UpProj::Sel(SelUp::Q5k)),
                    ),
                    ("blk.1.attn_k_b.weight", Launch::KUp),
                    ("blk.1.ffn_gate_shexp.weight", Launch::Dense(SiteEntry::Q4k)),
                    ("blk.1.ffn_up_shexp.weight", Launch::Dense(SiteEntry::Iq4xs)),
                    (
                        "blk.1.indexer.attn_q_b.weight",
                        Launch::Dense(SiteEntry::Iq4xs),
                    ),
                    ("blk.0.ffn_down.weight", Launch::Dense(SiteEntry::Q5k)),
                    ("output.weight", Launch::Head(HeadEntry::Q5k)),
                    ("token_embd.weight", Launch::Embed(EmbedEntry::Q3k)),
                    (
                        "blk.1.ffn_gate_exps.weight",
                        Launch::GateUp(GateUpEntry::Iq1s),
                    ),
                    (
                        "blk.22.ffn_gate_exps.weight",
                        Launch::GateUp(GateUpEntry::Iq1m),
                    ),
                    (
                        "blk.22.ffn_up_exps.weight",
                        Launch::GateUp(GateUpEntry::Iq1s),
                    ),
                    ("blk.22.ffn_down_exps.weight", Launch::Down(DownEntry::Iq1m)),
                ],
            ),
        ] {
            for (name, entry) in want {
                let got = of(table, name);
                if got.as_ref() != Ok(&entry) {
                    bad.push(format!("{file}: {name} is {got:?}, want {entry:?}"));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} pin(s) differ:\n  {}",
            bad.len(),
            bad.join("\n  ")
        );
    }

    /// Every type but Q8_0 is refused for `attn_k_b` by name: the refusal names the tensor, the
    /// type and Q8_0. Mutant: `k_b` takes any type.
    #[test]
    fn a_k_b_of_another_type_is_refused_by_name() {
        let name = "blk.3.attn_k_b.weight";
        let mut bad = Vec::new();
        for id in (0..=45).chain(2000..=2005) {
            let ty = GgmlType::from_u32(id);
            if matches!(ty, GgmlType::Unknown(_)) || ty == GgmlType::Q8_0 {
                continue;
            }
            for got in [k_b(name, ty), launch(name, ty).map(|_| ())] {
                match got {
                    Err(e)
                        if e.contains(name)
                            && e.contains(&ty.to_string())
                            && e.contains("Q8_0") => {}
                    other => bad.push(format!("{ty}: {other:?}")),
                }
            }
        }
        assert_eq!(k_b(name, GgmlType::Q8_0), Ok(()));
        assert_eq!(launch(name, GgmlType::Q8_0), Ok(Launch::KUp));
        assert!(
            bad.is_empty(),
            "{} type(s) not refused by name:\n  {}",
            bad.len(),
            bad.join("\n  ")
        );
    }

    /// `attn_v_b` takes Q8_0 planes or a down entry over the 128-value form that a per-head launch
    /// runs, and refuses by name a type the table gives no down entry, a 32-value form and MXFP4.
    /// Mutant: the Q4_K arm of `sel_up` runs as Q5_K.
    #[test]
    fn a_v_b_takes_planes_or_a_q8_1_down_entry() {
        let name = "blk.1.attn_v_b.weight";
        for (ty, want) in [
            (GgmlType::Q8_0, UpProj::Planes),
            (GgmlType::Q4_K, UpProj::Sel(SelUp::Q4k)),
            (GgmlType::Q5_K, UpProj::Sel(SelUp::Q5k)),
            (GgmlType::Q6_K, UpProj::Sel(SelUp::Q6k)),
            (GgmlType::IQ4_XS, UpProj::Sel(SelUp::IQuant)),
            (GgmlType::IQ2_XXS, UpProj::Sel(SelUp::IQuant)),
        ] {
            assert_eq!(v_b(name, ty), Ok(want), "{ty}");
        }
        for ty in [
            GgmlType::Q3_K,
            GgmlType::F32,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::IQ4_NL,
            GgmlType::MXFP4,
            GgmlType::NVFP4,
        ] {
            match v_b(name, ty) {
                Err(e) if e.contains(name) && e.contains(&ty.to_string()) => {}
                other => panic!("{ty}: {other:?}"),
            }
        }
    }

    /// A routed stack of a type the table gives no gate·up or down entry is refused by name, and
    /// a type it names runs. Mutant: `Reader::GateUp` answers with the dense table.
    #[test]
    fn a_stack_without_its_entry_is_refused_by_name() {
        for (name, ty) in [
            ("blk.4.ffn_gate_exps.weight", GgmlType::Q3_K),
            ("blk.4.ffn_up_exps.weight", GgmlType::F32),
            ("blk.4.ffn_down_exps.weight", GgmlType::Q3_K),
        ] {
            match launch(name, ty) {
                Err(e) if e.contains(name) && e.contains(&ty.to_string()) => {}
                other => panic!("{name} as {ty}: {other:?}"),
            }
        }
        assert_eq!(
            launch("blk.4.ffn_up_exps.weight", GgmlType::IQ1_M),
            Ok(Launch::GateUp(GateUpEntry::Iq1m))
        );
        assert_eq!(
            launch("blk.4.ffn_down_exps.weight", GgmlType::IQ2_XS),
            Ok(Launch::Down(DownEntry::Iq2xs))
        );
    }

    /// A tensor no reader of the family names is refused by name.
    #[test]
    fn a_tensor_no_reader_names_is_refused() {
        match launch("blk.4.ffn_unknown.weight", GgmlType::F32) {
            Err(e) if e.contains("blk.4.ffn_unknown.weight") => {}
            other => panic!("{other:?}"),
        }
    }
}
