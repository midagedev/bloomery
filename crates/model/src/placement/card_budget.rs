//! A card byte budget: every card of a plan plans as if it had at most this
//! many usable bytes — `min(usable, budget)` — so one large card can stand in
//! for any smaller one. The expert rule then runs on the same usable − KV −
//! context − scratch − margin arithmetic and keeps fewer experts.
//!
//! The value is decimal bytes with optional `_` separators, or a whole
//! number of MiB or GiB with an `M` or `G` suffix (binary units: `38G` is
//! 40,802,189,312 B).

use std::sync::OnceLock;

use super::PlacementError;

/// The lever: a card byte budget, read once per process.
pub const LEVER: &str = "BLOOMERY_CARD_BUDGET";

/// The budget `BLOOMERY_CARD_BUDGET` sets, parsed on the first call of the
/// process; `None` when the variable is not set. Held in a `OnceLock`
/// because it is a lever: every plan of the process plans under it.
pub fn from_env() -> Result<Option<u64>, PlacementError> {
    static READ: OnceLock<Result<Option<u64>, (String, String)>> = OnceLock::new();
    let read = READ.get_or_init(|| match std::env::var(LEVER) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err((String::new(), e.to_string())),
        Ok(v) => parse(&v).map(Some).map_err(|detail| (v, detail)),
    });
    match read {
        Ok(budget) => Ok(*budget),
        Err((value, detail)) => Err(PlacementError::CardBudgetLever {
            value: value.clone(),
            detail: detail.clone(),
        }),
    }
}

/// Bytes `value` names: digits with optional `_` separators, then nothing
/// (bytes), `M` (MiB) or `G` (GiB). Anything else, and a value that passes
/// `u64`, is refused.
pub fn parse(value: &str) -> Result<u64, String> {
    let (digits, unit) = match value.strip_suffix('G') {
        Some(d) => (d, 1u64 << 30),
        None => match value.strip_suffix('M') {
            Some(d) => (d, 1u64 << 20),
            None => (value, 1),
        },
    };
    let digits: String = digits.chars().filter(|&c| c != '_').collect();
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "{value:?} is not bytes, or a count of MiB or GiB with an M or G suffix"
        ));
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(unit))
        .ok_or_else(|| format!("{value:?} passes u64 bytes"))
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parse_takes_bytes_mib_and_gib() {
        assert_eq!(parse("40802189312"), Ok(40_802_189_312));
        assert_eq!(parse("40_802_189_312"), Ok(40_802_189_312));
        assert_eq!(parse("38G"), Ok(40_802_189_312));
        assert_eq!(parse("24176M"), Ok(25_350_373_376));
        for bad in [
            "",
            "G",
            "38 G",
            "38g",
            "38GB",
            "-1",
            "1.5G",
            "18446744073709551616",
            "17179869184G",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} parsed");
        }
    }
}
