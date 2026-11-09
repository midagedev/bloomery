//! The shards a split of planned tensors fills: the one owner of where each tensor of a split lands, so `prepare`
//! writing it and `bloomery-quantize --hf` reading the folder as that split see the same shards.

use std::error::Error;

use gguf::Value;
use gguf::write::Layout;

use crate::folder::{Make, Planned};

/// One shard: its tensors and its file's layout.
pub type Shard<S> = (Vec<Planned<S>>, Layout);

/// The shards `planned` fills to `limit` bytes of file, in order, each with its layout (`kvs` in the first, every
/// shard's `split.*` keys last). A tensor and the `.scale` after it stay in one shard.
pub fn plan_shards<S>(
    planned: Vec<Planned<S>>,
    kvs: &[(String, Value)],
    limit: u64,
) -> Result<Vec<Shard<S>>, Box<dyn Error>> {
    let tensors32 =
        i32::try_from(planned.len()).map_err(|_| "too many tensors for a GGUF split")?;
    let shard_keys = |no: u16, count: u16| {
        let mut keys = if no == 0 { kvs.to_vec() } else { vec![] };
        keys.extend([
            ("split.no".to_string(), Value::U16(no)),
            ("split.count".to_string(), Value::U16(count)),
            ("split.tensors.count".to_string(), Value::I32(tensors32)),
        ]);
        keys
    };
    let mut shards: Vec<Vec<Planned<S>>> = vec![vec![]];
    let mut pending = planned.into_iter().peekable();
    while let Some(p) = pending.next() {
        let mut group = vec![p];
        if pending
            .peek()
            .is_some_and(|p| matches!(p.make, Make::Scale { .. }))
        {
            group.push(pending.next().expect("the scale was just inspected"));
        }
        let no = shards.len() - 1;
        let no16 = u16::try_from(no).map_err(|_| "too many shards for a GGUF split")?;
        let last = &mut shards[no];
        let declarations = last.iter().chain(&group).map(|p| p.decl.clone()).collect();
        // split.count has the same U16 width for every final count.
        let candidate = Layout::new(&shard_keys(no16, 1), declarations)?;
        if !last.is_empty() && candidate.file_len() > limit {
            shards.push(group);
        } else {
            last.extend(group);
        }
    }
    let count16 = u16::try_from(shards.len()).map_err(|_| "too many shards for a GGUF split")?;
    shards
        .into_iter()
        .enumerate()
        .map(|(no, shard)| {
            let keys = shard_keys(u16::try_from(no)?, count16);
            let layout = Layout::new(&keys, shard.iter().map(|p| p.decl.clone()).collect())?;
            Ok((shard, layout))
        })
        .collect()
}
