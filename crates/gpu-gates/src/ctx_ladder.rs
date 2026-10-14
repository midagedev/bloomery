//! The serve gates' side of the seats' context choice (`placement::ctx_ask`,
//! `generate::choose_ctx`): the ladder an unset `--ctx` walks, recomputed from
//! this module's own literals, and the one `ctx` line a seat prints, read and
//! held to its word. Every serve gate reads a seat's choice through here, so a
//! change of the rule is one edit on the gate side too.

/// The context an unset `--ctx` asks a slot first on a file whose limit is
/// past it: 256K positions (`ctx_ask::DEFAULT_TARGET`).
pub const TARGET: u64 = 262_144;

/// The least rung an unset `--ctx` asks, the base a slot
/// (`generate::SEAT_STEPS`' floor).
pub const FLOOR: u64 = 4096;

/// The rounding of every rung (`generate::CTX_GRAN`).
pub const STEP: u64 = 1024;

/// The context an unset `--ctx` asks a slot first under the model's `limit`:
/// [`TARGET`] or the limit, the lesser, down to a [`STEP`], never under
/// [`FLOOR`] a limit past it holds.
#[must_use]
pub fn target_of(limit: u64) -> u64 {
    let cap = TARGET.min(limit);
    (cap / STEP * STEP).max(FLOOR.min(cap))
}

/// The rungs an unset `--ctx` asks a slot, in order: the target
/// ([`target_of`]), then each rung's half down to a [`STEP`], [`FLOOR`] where
/// the half falls under it, which is the last.
#[must_use]
pub fn rungs(limit: u64) -> Vec<u64> {
    let target = target_of(limit);
    let mut rungs = vec![target];
    while let Some(&last) = rungs.last() {
        let next = (last / 2 / STEP * STEP).max(FLOOR.min(target));
        if next == last {
            break;
        }
        rungs.push(next);
    }
    rungs
}

/// The rung an unset `--ctx` asked a slot before `ctx`; `None` when `ctx` is
/// the target or no rung.
#[must_use]
pub fn rung_above(limit: u64, ctx: u64) -> Option<u64> {
    let rungs = rungs(limit);
    let at = rungs.iter().position(|&r| r == ctx)?;
    at.checked_sub(1).map(|i| rungs[i])
}

/// The one `ctx` line of a seat's stderr (`ctx rule=… ctx=… slots=… total=…
/// limit=… (…)[ from=… <memory>-short need=… room=…]`,
/// `placement::ctx_ask::Chosen::line`); `None` when there is none or more
/// than one.
#[must_use]
pub fn ctx_line(log: &str) -> Option<&str> {
    let mut lines = log.lines().filter(|l| l.starts_with("ctx rule="));
    let one = lines.next()?;
    lines.next().is_none().then_some(one)
}

/// The whole-number field `name` of a line of `name=value` words.
#[must_use]
pub fn field(line: &str, name: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(name)?.strip_prefix('='))?
        .parse()
        .ok()
}

/// The memory a `halved` line was short of (`card` or `host`).
#[must_use]
pub fn short_of(line: &str) -> Option<&str> {
    line.split_whitespace()
        .find_map(|w| w.strip_suffix("-short"))
}

/// Whether the `ctx` line of an unset `--ctx` on `slots` slots under the
/// model's `limit`, named `source`, holds its word: `target` asks the target
/// and took it; `halved` asked it (`from`), found the machine short of the
/// rung above its context (`need` past `room`, on the card or the host) and
/// took a rung below; the slots are `slots`, the total their sum, the limit
/// `limit (source)`. `None` when the line or one of its terms is missing.
#[must_use]
pub fn default_line_held(line: &str, slots: u64, (limit, source): (u64, &str)) -> Option<bool> {
    let rule = line
        .split_whitespace()
        .find_map(|w| w.strip_prefix("rule="))?;
    let num = |k: &str| field(line, k);
    let (ctx, n, total) = (num("ctx")?, num("slots")?, num("total")?);
    let by_word = match rule {
        "target" => ctx == target_of(limit) && !line.contains(" from="),
        "halved" => {
            let (from, need, room) = (num("from")?, num("need")?, num("room")?);
            from == target_of(limit)
                && rung_above(limit, ctx).is_some()
                && short_of(line).is_some_and(|m| m == "card" || m == "host")
                && need > room
        }
        _ => false,
    };
    Some(
        by_word
            && n == slots
            && total == slots * ctx
            && line.contains(&format!(" limit={limit} ({source})")),
    )
}

#[cfg(test)]
mod tests {
    use super::{default_line_held, rung_above, rungs, target_of};

    /// The ladder from a power of two reaches the floor by halves; one from
    /// another limit steps onto the floor; a limit under the floor is its own
    /// one rung.
    #[test]
    fn the_ladder_ends_on_the_floor() {
        assert_eq!(target_of(1_048_576), 262_144);
        assert_eq!(
            rungs(262_144),
            [262_144, 131_072, 65_536, 32_768, 16_384, 8192, 4096]
        );
        assert_eq!(rungs(24_576), [24_576, 12_288, 6144, 4096]);
        assert_eq!(rungs(2048), [2048]);
        assert_eq!(rung_above(262_144, 65_536), Some(131_072));
        assert_eq!(rung_above(262_144, 262_144), None);
        assert_eq!(rung_above(262_144, 5000), None);
    }

    /// A line holds its word, and a line off the ladder or of other slots
    /// does not.
    #[test]
    fn a_line_holds_its_word() {
        let limit = (262_144, "trained");
        let target = "ctx rule=target ctx=262144 slots=1 total=262144 limit=262144 (trained)";
        assert_eq!(default_line_held(target, 1, limit), Some(true));
        assert_eq!(default_line_held(target, 2, limit), Some(false));
        let halved = "ctx rule=halved ctx=65536 slots=1 total=65536 limit=262144 (trained) \
                      from=262144 host-short need=9 room=8";
        assert_eq!(default_line_held(halved, 1, limit), Some(true));
        let off = "ctx rule=halved ctx=65000 slots=1 total=65000 limit=262144 (trained) \
                   from=262144 card-short need=9 room=8";
        assert_eq!(default_line_held(off, 1, limit), Some(false));
        assert_eq!(default_line_held("ctx rule=target", 1, limit), None);
    }
}
