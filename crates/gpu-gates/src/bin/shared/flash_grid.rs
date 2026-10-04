//! The decode flash's launch grids in a captured graph, the e2e gates' one
//! reading of them: the segment pass `rows · n_kv · SEGMENTS` blocks and the
//! merge `rows · n_head` blocks, one of each an attention layer, whatever the
//! cache height the load allocated.

use bloomery_gpu::NodeInfo;
use bloomery_gpu::flash_gqa::SEGMENTS;
use bloomery_gpu_gates::GateError;

/// Whether `nodes` hold exactly `layers` segment-pass and `layers` merge
/// launches, each at its grid for `rows` rows of `n_head` query heads over
/// `n_kv` key heads; and the line part naming each kind's distinct grids and
/// node count as the graph holds them. The entry names print on a line of
/// their own when they are not `names`, the ones the load picks.
pub fn flash_grids(
    nodes: &[NodeInfo],
    (rows, n_kv, n_head): (usize, usize, usize),
    layers: usize,
    names: [&str; 2],
) -> Result<(bool, String), GateError> {
    let want = [
        [u32::try_from(rows * n_kv * SEGMENTS)?, 1, 1],
        [u32::try_from(rows * n_head)?, 1, 1],
    ];
    let (mut grids, mut count): ([Vec<[u32; 3]>; 2], [usize; 2]) = Default::default();
    let mut seen: Vec<&str> = Vec::new();
    for kn in nodes.iter().filter_map(|n| n.kernel.as_ref()) {
        let Some(i) = ["gqa_flash_seg", "gqa_flash_merge"]
            .iter()
            .position(|p| kn.name.starts_with(p))
        else {
            continue;
        };
        count[i] += 1;
        if !grids[i].contains(&kn.grid) {
            grids[i].push(kn.grid);
        }
        if !seen.contains(&kn.name.as_str()) {
            seen.push(kn.name.as_str());
        }
    }
    if seen != names {
        println!("structure flash entries: {seen:?}");
    }
    let ok = (0..2).all(|i| grids[i] == [want[i]] && count[i] == layers);
    Ok((
        ok,
        format!(
            "seg {:?} x{} merge {:?} x{} (want [{:?}] and [{:?}] x{layers})",
            grids[0], count[0], grids[1], count[1], want[0], want[1]
        ),
    ))
}
