//! gguf-inventory — header-only tensor inventory for one GGUF file or a
//! split set of shards. Reads magic, version, KVs and tensor infos through
//! [`gguf::inventory_of`] and never touches tensor bytes, so a multi-
//! hundred-GiB split inventories in seconds with no GPU and no lease.
//!
//! Grouping rule set (first match wins):
//!
//! 1. name contains `engram` → engram
//! 2. name contains `mtp`/`nextn`/`spark` → mtp_nextn
//! 3. no `blk.` prefix → `token_embd*` token_embd, `output*` (incl
//!    output_norm) output, anything else other
//! 4. `blk.N.` suffix → contains `_exps` routed_exps, contains `_shexp`
//!    shexp, starts with `attn` attn (all MLA parts and attn_norm), starts
//!    with `ffn_gate` in a MoE block router, any other `ffn_*` ffn_dense
//!    (dense blocks' gate/up/down, and every ffn_norm), else other
//!
//! A block is MoE iff it owns a `*_exps` tensor. Dense-per-token is then
//! total − routed_exps − engram by construction.
//!
//! Per-token active bytes use the metadata's `<arch>.expert_count` and
//! `<arch>.expert_used_count` (top-k): routed/token = Σ over MoE blocks of
//! (exps bytes × top-k / expert count), exact integer division. The
//! roofline reference column quotes the engramQ8 split's hand-computed
//! numbers (docs/roofline.md) and is printed only for files that carry
//! engram tensors; other files get the raw numbers.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::process::ExitCode;

use gguf::{GgmlType, Inventory, ggml_type_info, inventory_of};

// docs/roofline.md's hand-computed figures for the engramQ8 split.
const REF_FILE_GIB: f64 = 444.23;
const REF_ROUTED_GIB: f64 = 3.7656;
const REF_DENSE_GIB: f64 = 7.1320;
const REF_ENGRAM_GIB: f64 = 194.867;

struct Shard {
    ordinal: usize,
    path: String,
    inv: Inventory,
}

/// One model tensor: the union row the report prints, borrowed from the
/// shard inventories.
struct Row<'a> {
    block: Option<u32>,
    name: &'a str,
    dims: &'a [u64],
    type_id: u32,
    bytes: Option<u64>,
    shard: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Group {
    TokenEmbd,
    Output,
    Attn,
    Router,
    FfnDense,
    Shexp,
    RoutedExps,
    Engram,
    MtpNextn,
    Other,
}

impl Group {
    fn name(self) -> &'static str {
        match self {
            Group::TokenEmbd => "token_embd",
            Group::Output => "output",
            Group::Attn => "attn",
            Group::Router => "router",
            Group::FfnDense => "ffn_dense",
            Group::Shexp => "ffn_shexp",
            Group::RoutedExps => "ffn_exps_routed",
            Group::Engram => "engram",
            Group::MtpNextn => "mtp_nextn",
            Group::Other => "other",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Group::TokenEmbd => "token_embd*",
            Group::Output => "output*/output_norm*",
            Group::Attn => "blk.*.attn*",
            Group::Router => "blk.MoE.ffn_gate*",
            Group::FfnDense => "ffn_* dense-path",
            Group::Shexp => "*_shexp*",
            Group::RoutedExps => "*_exps*",
            Group::Engram => "*engram*",
            Group::MtpNextn => "*mtp*/*nextn*/*spark*",
            Group::Other => "unmatched",
        }
    }
}

fn block_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let idx = rest.split('.').next()?;
    idx.parse().ok()
}

fn classify(name: &str, moe_blocks: &HashSet<u32>) -> Group {
    if name.contains("engram") {
        return Group::Engram;
    }
    if name.contains("mtp") || name.contains("nextn") || name.contains("spark") {
        return Group::MtpNextn;
    }
    let Some(block) = block_of(name) else {
        return if name.starts_with("token_embd") {
            Group::TokenEmbd
        } else if name.starts_with("output") {
            Group::Output
        } else {
            Group::Other
        };
    };
    let suffix = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map(|(_, s)| s)
        .unwrap_or("");
    if suffix.contains("_exps") {
        Group::RoutedExps
    } else if suffix.contains("_shexp") {
        Group::Shexp
    } else if suffix.starts_with("attn") {
        Group::Attn
    } else if suffix.starts_with("ffn_gate") && moe_blocks.contains(&block) {
        Group::Router
    } else if suffix.starts_with("ffn_") {
        Group::FfnDense
    } else {
        Group::Other
    }
}

fn arch_u64(inv: &Inventory, suffix: &str) -> Option<u64> {
    let arch = inv.value("general.architecture")?.as_str()?;
    inv.value(&format!("{arch}.{suffix}"))?.as_u64()
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn gib(n: u64) -> String {
    format!("{:.4}", n as f64 / (1u64 << 30) as f64)
}

fn dims_str(dims: &[u64]) -> String {
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("×")
}

fn type_str(id: u32) -> String {
    match ggml_type_info(id) {
        Some((name, _, _)) => name.to_string(),
        None => format!("type#{id}"),
    }
}

fn main() -> ExitCode {
    let mut md_path: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--markdown" => match args.next() {
                Some(p) => md_path = Some(p),
                None => {
                    eprintln!("gguf-inventory: --markdown needs a path");
                    return ExitCode::from(2);
                }
            },
            flag if flag.starts_with('-') => {
                eprintln!("gguf-inventory: unknown flag {flag}");
                eprintln!("usage: gguf-inventory [--markdown PATH] FILE...");
                return ExitCode::from(2);
            }
            p => paths.push(p.to_string()),
        }
    }
    if paths.is_empty() {
        eprintln!("usage: gguf-inventory [--markdown PATH] FILE...");
        return ExitCode::from(2);
    }

    let mut shards = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        match inventory_of(p) {
            Ok(inv) => shards.push(Shard {
                ordinal: i + 1,
                path: p.clone(),
                inv,
            }),
            Err(e) => {
                eprintln!("gguf-inventory: {p}: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // The union over shards is the model; a name in two shards would be a
    // malformed split, so it is reported, not merged.
    let mut names = BTreeSet::new();
    let mut duplicates = 0usize;
    let mut moe_blocks: HashSet<u32> = HashSet::new();
    let mut rows: Vec<Row> = Vec::new();
    for (si, s) in shards.iter().enumerate() {
        for t in &s.inv.tensors {
            if !names.insert(t.name.clone()) {
                duplicates += 1;
            }
            if t.name.contains("_exps")
                && let Some(b) = block_of(&t.name)
            {
                moe_blocks.insert(b);
            }
            rows.push(Row {
                block: block_of(&t.name),
                name: t.name.as_str(),
                dims: &t.dims,
                type_id: t.type_id,
                bytes: t.nbytes,
                shard: si,
            });
        }
    }
    rows.sort_by(|a, b| {
        let ka = (a.block.unwrap_or(u32::MAX), a.name);
        let kb = (b.block.unwrap_or(u32::MAX), b.name);
        ka.cmp(&kb)
    });

    let arch = shards
        .iter()
        .find_map(|s| s.inv.value("general.architecture"))
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    let meta_of =
        |suffix: &str| -> Option<u64> { shards.iter().find_map(|s| arch_u64(&s.inv, suffix)) };
    let expert_count = meta_of("expert_count");
    let top_k = meta_of("expert_used_count");
    let block_count = meta_of("block_count");
    let first_k_dense = meta_of("first_k_dense_replace");

    // Group sums, per-block exps sums, type census.
    let mut group_bytes: BTreeMap<Group, u64> = BTreeMap::new();
    let mut group_count: BTreeMap<Group, usize> = BTreeMap::new();
    let mut group_unknown: BTreeMap<Group, usize> = BTreeMap::new();
    let mut group_names: BTreeMap<Group, Vec<String>> = BTreeMap::new();
    let mut block_exps: BTreeMap<u32, u64> = BTreeMap::new();
    let mut types: BTreeMap<u32, (usize, u64, usize)> = BTreeMap::new(); // count, bytes, unknown
    let mut total_known = 0u64;
    let mut total_unknown = 0usize;
    let mut total_file = 0u64;
    for s in &shards {
        total_file += s.inv.file_len;
        for t in &s.inv.tensors {
            let g = classify(&t.name, &moe_blocks);
            *group_count.entry(g).or_default() += 1;
            let e = types.entry(t.type_id).or_default();
            e.0 += 1;
            match t.nbytes {
                Some(n) => {
                    *group_bytes.entry(g).or_default() += n;
                    e.1 += n;
                    total_known += n;
                    if g == Group::RoutedExps
                        && let Some(b) = block_of(&t.name)
                    {
                        *block_exps.entry(b).or_default() += n;
                    }
                }
                None => {
                    *group_unknown.entry(g).or_default() += 1;
                    e.2 += 1;
                    total_unknown += 1;
                }
            }
            if matches!(g, Group::Engram | Group::MtpNextn | Group::Other) {
                group_names
                    .entry(g)
                    .or_default()
                    .push(format!("{} {}", t.name, dims_str(&t.dims)));
            }
        }
    }

    let engram_total = group_bytes.get(&Group::Engram).copied().unwrap_or(0);
    let routed_total = group_bytes.get(&Group::RoutedExps).copied().unwrap_or(0);
    let dense_total = total_known
        .checked_sub(routed_total)
        .and_then(|v| v.checked_sub(engram_total))
        .unwrap_or(0);

    let routed_per_token: Option<u128> = match (expert_count, top_k) {
        (Some(ec), Some(k)) if ec > 0 => {
            let mut sum = 0u128;
            let mut exact = true;
            for bytes in block_exps.values() {
                let b = *bytes as u128 * k as u128;
                if b.is_multiple_of(ec as u128) {
                    sum += b / ec as u128;
                } else {
                    exact = false;
                }
            }
            if exact { Some(sum) } else { None }
        }
        _ => None,
    };

    // ---- stdout report -------------------------------------------------
    let mut out = String::new();
    for s in &shards {
        let tensor_bytes: u64 = s.inv.tensors.iter().filter_map(|t| t.nbytes).sum();
        let unknown = s.inv.tensors.iter().filter(|t| t.nbytes.is_none()).count();
        let split_no = s.inv.value("split.no").and_then(|v| v.as_u64());
        let split_count = s.inv.value("split.count").and_then(|v| v.as_u64());
        let split_tensors = s.inv.value("split.tensors.count").and_then(|v| v.as_u64());
        out.push_str(&format!(
            "shard {}: {}\n  version={} tensors={} kv={} header_end={} data_base={} alignment={}\n  file_bytes={} tensor_bytes={} unknown_size={} pad={}\n",
            s.ordinal,
            s.path,
            s.inv.version,
            s.inv.tensors.len(),
            s.inv.meta.len(),
            commas(s.inv.header_end),
            commas(s.inv.data_base),
            s.inv.alignment,
            commas(s.inv.file_len),
            commas(tensor_bytes),
            unknown,
            s.inv.file_len.saturating_sub(s.inv.data_base).saturating_sub(tensor_bytes),
        ));
        let mut tags = Vec::new();
        if let Some(v) = split_no {
            tags.push(format!("split.no={v}"));
        }
        if let Some(v) = split_count {
            tags.push(format!("split.count={v}"));
        }
        if let Some(v) = split_tensors {
            tags.push(format!("split.tensors.count={v}"));
        }
        if !tags.is_empty() {
            out.push_str(&format!("  {}\n", tags.join(" ")));
        }
    }
    out.push_str(&format!(
        "model: {} tensors across {} shard(s), {} duplicate name(s)\n",
        names.len(),
        shards.len(),
        duplicates
    ));

    out.push_str("types:\n");
    for (id, (count, bytes, unknown)) in &types {
        let lacks = matches!(GgmlType::from_u32(*id), GgmlType::Unknown(_));
        out.push_str(&format!(
            "  #{id:<3} {:<9} x{:<5} {:>15} bytes{}{}\n",
            type_str(*id),
            count,
            commas(*bytes),
            if *unknown > 0 {
                format!(" (+{unknown} unknown size)")
            } else {
                String::new()
            },
            if lacks {
                "  [absent from GgmlType]"
            } else {
                ""
            },
        ));
    }

    out.push_str("tensors (sorted by block index, then name; non-blk tensors last):\n");
    for r in &rows {
        let shard_label = shards
            .get(r.shard)
            .and_then(|s| s.inv.value("split.no"))
            .and_then(|v| v.as_u64())
            .map(|v| v.to_string())
            .unwrap_or_else(|| (r.shard + 1).to_string());
        out.push_str(&format!(
            "  {:>4}  {:<48} {:<24} {:<9} {:>15}  {}\n",
            r.block.map(|b| b.to_string()).unwrap_or_else(|| "-".into()),
            r.name,
            dims_str(r.dims),
            type_str(r.type_id),
            r.bytes.map(commas).unwrap_or_else(|| "unknown".into()),
            shard_label,
        ));
    }

    out.push_str("groups:\n");
    for g in [
        Group::TokenEmbd,
        Group::Output,
        Group::Attn,
        Group::Router,
        Group::FfnDense,
        Group::Shexp,
        Group::RoutedExps,
        Group::Engram,
        Group::MtpNextn,
        Group::Other,
    ] {
        let n = group_count.get(&g).copied().unwrap_or(0);
        let b = group_bytes.get(&g).copied().unwrap_or(0);
        let u = group_unknown.get(&g).copied().unwrap_or(0);
        out.push_str(&format!(
            "  {:<16} ({:<22}) x{:<5} {:>15} bytes = {} GiB{}\n",
            g.name(),
            g.label(),
            n,
            commas(b),
            gib(b),
            if u > 0 {
                format!(" (+{u} unknown size)")
            } else {
                String::new()
            },
        ));
        if matches!(g, Group::Engram | Group::MtpNextn | Group::Other)
            && let Some(list) = group_names.get(&g)
        {
            for n in list {
                out.push_str(&format!("    {n}\n"));
            }
        }
    }

    out.push_str(&format!(
        "metadata: arch={arch} expert_count={:?} expert_used_count={:?} block_count={:?} first_k_dense_replace={:?} moe_blocks={}\n",
        expert_count,
        top_k,
        block_count,
        first_k_dense,
        moe_blocks.len(),
    ));
    let has_engram = group_count.contains_key(&Group::Engram);
    out.push_str("per-token active bytes:\n");
    if let Some(r) = routed_per_token {
        let extra = if has_engram {
            format!(
                "  [roofline {} GiB, delta {:+.4}]",
                REF_ROUTED_GIB,
                r as f64 / (1u64 << 30) as f64 - REF_ROUTED_GIB
            )
        } else {
            String::new()
        };
        out.push_str(&format!(
            "  routed experts/token = {} bytes = {} GiB{}\n",
            commas(r as u64),
            gib(r as u64),
            extra
        ));
    } else {
        out.push_str(
            "  routed experts/token = n/a (expert_count/top-k missing or inexact division)\n",
        );
    }
    let extra = if has_engram {
        format!(
            "  [roofline {} GiB, delta {:+.4}]",
            REF_DENSE_GIB,
            dense_total as f64 / (1u64 << 30) as f64 - REF_DENSE_GIB
        )
    } else {
        String::new()
    };
    out.push_str(&format!(
        "  dense/token         = {} bytes = {} GiB{}\n",
        commas(dense_total),
        gib(dense_total),
        extra
    ));
    let embd = group_bytes.get(&Group::TokenEmbd).copied().unwrap_or(0);
    out.push_str(&format!(
        "  dense/token excl token_embd = {} bytes = {} GiB\n",
        commas(dense_total.saturating_sub(embd)),
        gib(dense_total.saturating_sub(embd)),
    ));
    out.push_str(&format!(
        "totals: file={} bytes = {} GiB{}, tensor bytes known={}{}, engram table={} GiB{}\n",
        commas(total_file),
        gib(total_file),
        if has_engram {
            format!("  [roofline {REF_FILE_GIB} GiB]")
        } else {
            String::new()
        },
        commas(total_known),
        if total_unknown > 0 {
            format!(", {total_unknown} tensors of unknown size (sums are lower bounds)")
        } else {
            String::new()
        },
        gib(engram_total),
        if has_engram {
            format!("  [roofline {REF_ENGRAM_GIB} GiB]")
        } else {
            String::new()
        },
    ));
    print!("{out}");

    // ---- markdown (tables only; the prose is the lead's) ----------------
    if let Some(md) = md_path {
        let mut m = String::new();
        m.push_str("## shards\n\n");
        m.push_str("| shard | path | version | tensors | kv | header_end | data_base | alignment | file_bytes | tensor_bytes | unknown_size | pad | split.no | split.count | split.tensors.count |\n");
        m.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for s in &shards {
            let tensor_bytes: u64 = s.inv.tensors.iter().filter_map(|t| t.nbytes).sum();
            let unknown = s.inv.tensors.iter().filter(|t| t.nbytes.is_none()).count();
            m.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                s.ordinal,
                s.path,
                s.inv.version,
                s.inv.tensors.len(),
                s.inv.meta.len(),
                commas(s.inv.header_end),
                commas(s.inv.data_base),
                s.inv.alignment,
                commas(s.inv.file_len),
                commas(tensor_bytes),
                unknown,
                s.inv
                    .file_len
                    .saturating_sub(s.inv.data_base)
                    .saturating_sub(tensor_bytes),
                s.inv
                    .value("split.no")
                    .and_then(|v| v.as_u64())
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                s.inv
                    .value("split.count")
                    .and_then(|v| v.as_u64())
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                s.inv
                    .value("split.tensors.count")
                    .and_then(|v| v.as_u64())
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            ));
        }
        m.push_str(&format!(
            "\nModel: {} tensors across {} shard(s), {} duplicate name(s).\n\n",
            names.len(),
            shards.len(),
            duplicates
        ));

        m.push_str("## types\n\n");
        m.push_str("| id | type | tensors | bytes | unknown_size | in engine GgmlType |\n");
        m.push_str("|---|---|---:|---:|---:|---|\n");
        for (id, (count, bytes, unknown)) in &types {
            let lacks = matches!(GgmlType::from_u32(*id), GgmlType::Unknown(_));
            m.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                id,
                type_str(*id),
                count,
                commas(*bytes),
                unknown,
                if lacks { "no" } else { "yes" },
            ));
        }

        m.push_str("\n## tensors\n\n");
        m.push_str("Sorted by block index, then name; non-blk tensors last.\n\n");
        m.push_str("| block | name | dims | type | bytes | shard |\n");
        m.push_str("|---|---|---|---|---:|---|\n");
        for r in &rows {
            let shard_label = shards
                .get(r.shard)
                .and_then(|s| s.inv.value("split.no"))
                .and_then(|v| v.as_u64())
                .map(|v| v.to_string())
                .unwrap_or_else(|| (r.shard + 1).to_string());
            m.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                r.block.map(|b| b.to_string()).unwrap_or_else(|| "-".into()),
                r.name,
                dims_str(r.dims),
                type_str(r.type_id),
                r.bytes.map(commas).unwrap_or_else(|| "unknown".into()),
                shard_label,
            ));
        }

        m.push_str("\n## groups\n\n");
        m.push_str("| group | rule | tensors | bytes | GiB | unknown_size |\n");
        m.push_str("|---|---|---:|---:|---:|---:|\n");
        for g in [
            Group::TokenEmbd,
            Group::Output,
            Group::Attn,
            Group::Router,
            Group::FfnDense,
            Group::Shexp,
            Group::RoutedExps,
            Group::Engram,
            Group::MtpNextn,
            Group::Other,
        ] {
            m.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                g.name(),
                g.label(),
                group_count.get(&g).copied().unwrap_or(0),
                commas(group_bytes.get(&g).copied().unwrap_or(0)),
                gib(group_bytes.get(&g).copied().unwrap_or(0)),
                group_unknown.get(&g).copied().unwrap_or(0),
            ));
        }
        for g in [Group::Engram, Group::MtpNextn, Group::Other] {
            if let Some(list) = group_names.get(&g) {
                m.push_str(&format!(
                    "\n### {} tensor names\n\n| name | dims |\n|---|---|\n",
                    g.name()
                ));
                for n in list {
                    if let Some((name, dims)) = n.split_once(' ') {
                        m.push_str(&format!("| {name} | {dims} |\n"));
                    }
                }
            }
        }

        m.push_str("\n## per-token active bytes\n\n");
        m.push_str(&format!(
            "| key | value |\n|---|---|\n| architecture | {arch} |\n| expert_count | {:?} |\n| expert_used_count (top-k) | {:?} |\n| block_count | {:?} |\n| first_k_dense_replace | {:?} |\n| moe blocks | {} |\n\n",
            expert_count,
            top_k,
            block_count,
            first_k_dense,
            moe_blocks.len(),
        ));
        m.push_str(
            "| quantity | bytes/token | GiB/token | roofline (engramQ8 split) | delta GiB |\n",
        );
        m.push_str("|---|---:|---:|---:|---:|\n");
        if let Some(r) = routed_per_token {
            m.push_str(&format!(
                "| routed experts | {} | {} | {} | {:+.4} |\n",
                commas(r as u64),
                gib(r as u64),
                REF_ROUTED_GIB,
                r as f64 / (1u64 << 30) as f64 - REF_ROUTED_GIB
            ));
        }
        m.push_str(&format!(
            "| dense | {} | {} | {} | {:+.4} |\n",
            commas(dense_total),
            gib(dense_total),
            REF_DENSE_GIB,
            dense_total as f64 / (1u64 << 30) as f64 - REF_DENSE_GIB
        ));
        m.push_str(&format!(
            "| dense excl token_embd | {} | {} | {} | {:+.4} |\n",
            commas(dense_total.saturating_sub(embd)),
            gib(dense_total.saturating_sub(embd)),
            REF_DENSE_GIB,
            (dense_total.saturating_sub(embd)) as f64 / (1u64 << 30) as f64 - REF_DENSE_GIB
        ));
        m.push_str(&format!(
            "| totals: file | {} | {} | {} | — |\n",
            commas(total_file),
            gib(total_file),
            REF_FILE_GIB,
        ));
        m.push_str(&format!(
            "| totals: engram table | {} | {} | {} | — |\n",
            commas(engram_total),
            gib(engram_total),
            REF_ENGRAM_GIB,
        ));
        if total_unknown > 0 {
            m.push_str(&format!(
                "\n{} tensor(s) have unknown size; every byte sum above is a lower bound.\n",
                total_unknown
            ));
        }
        fs::write(&md, m).unwrap_or_else(|e| panic!("write {md}: {e}"));
        eprintln!("markdown written to {md}");
    }
    ExitCode::SUCCESS
}
