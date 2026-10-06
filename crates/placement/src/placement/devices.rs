//! The devices a plan runs on: the census of this process's visible CUDA
//! devices (the GPU side fills it without a context; this crate only reads
//! it), the card spec each device gives a plan ([`spec_of_device`]), the
//! `--place` picks a word names ([`word_picks`]) resolved against the census
//! ([`resolve`]), and the one device a plan's card opens on
//! ([`device_on_host`]).
//!
//! A device's identity is the driver's UUID. Its ordinal is this process's
//! enumeration (`CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` order it), so
//! an ordinal is trusted only beside the census of the process that took it,
//! and never from outside the process: the open checks the UUID against the
//! census again.

use std::fmt;
use std::sync::{Mutex, PoisonError};

use super::workstation::{A6000, CARDS, CardSpec, MIB, RTX_3090};

/// One visible CUDA device as this process's driver enumerates it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// This process's CUDA ordinal.
    pub ordinal: u32,
    /// The driver's name (`cuDeviceGetName`).
    pub name: String,
    /// `cuDeviceTotalMem`: what `nvidia-smi` calls the total less what it
    /// calls reserved, to under a MiB [measured on both of this
    /// workstation's cards].
    pub total_bytes: u64,
    /// `cuMemGetInfo`'s free bytes on the device's primary context, read at
    /// census time: what the device had left after every process done with
    /// it, our own future context included in the reading's cost. A plan
    /// sizes against this, not the total ([`spec_of_device`]).
    pub free_bytes: u64,
    /// `cuDeviceGetUuid`.
    pub uuid: [u8; 16],
    /// `cuDeviceGetPCIBusId`.
    pub pci_bus: String,
    /// The other processes holding the device's memory at census time, as
    /// one named list ("pid 4321 (1234 MiB), …"), when the tool that reads
    /// them answered; `None` when it did not or named none — a plan never
    /// fails for want of holders.
    pub held_by: Option<String>,
}

impl DeviceInfo {
    /// The device a plan's card names.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        DeviceId {
            ordinal: self.ordinal,
            uuid: self.uuid,
        }
    }
}

/// The device a plan's card was resolved to: its ordinal in the census it
/// was resolved against, and its UUID, the identity the open checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub ordinal: u32,
    pub uuid: [u8; 16],
}

/// A device as the records and the refusals name it: its ordinal. The UUID
/// stays in the process (the open's check); no output prints it.
impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda{}", self.ordinal)
    }
}

/// The driver names of the cards whose figures were measured, and those
/// figures.
const KNOWN: [(&str, CardSpec); 2] = [
    ("NVIDIA RTX A6000", A6000),
    ("NVIDIA GeForce RTX 3090", RTX_3090),
];

/// The usable bytes the census gives a device of `total_bytes`
/// (`cuDeviceTotalMem`): the total rounded down to a whole MiB. The driver's
/// reservation (`nvidia-smi`'s `reserved`) is already outside
/// `cuDeviceTotalMem`, so what is left is a tail under one MiB; on both
/// measured cards this rule gives exactly their usable bytes.
#[must_use]
pub const fn census_usable(total_bytes: u64) -> u64 {
    total_bytes / MIB * MIB
}

/// The card spec device `d` gives a plan, naming `d`: the device's free
/// reading and its holders ride along ([`CardSpec::free_bytes`],
/// [`CardSpec::held_by`]) for the plan's budget term and its refusals. A
/// device of a known card's driver name whose total gives that card's
/// usable bytes takes the card's measured figures; any other device takes
/// its census total with [`census_usable`]'s tail as the reserve, under a
/// label from its driver name ([`label`]).
#[must_use]
pub fn spec_of_device(d: &DeviceInfo) -> CardSpec {
    let device = Some(d.device());
    let free = Some(d.free_bytes);
    let held_by = held_word(&d.held_by);
    let known = KNOWN
        .iter()
        .find(|(name, k)| *name == d.name && k.usable_bytes() == census_usable(d.total_bytes));
    if let Some((_, k)) = known {
        return CardSpec {
            device,
            free_bytes: free,
            held_by,
            ..*k
        };
    }
    CardSpec {
        name: label(&d.name),
        total_bytes: d.total_bytes,
        driver_reserve_bytes: d.total_bytes - census_usable(d.total_bytes),
        device,
        free_bytes: free,
        held_by,
    }
}

/// A census holder list as a plan's card holds it: spelled once per list and
/// kept for the process, like a `--place` word.
fn held_word(held: &Option<String>) -> Option<&'static str> {
    static HELD: Mutex<Vec<(String, &'static str)>> = Mutex::new(Vec::new());
    let held = held.as_ref()?;
    let mut helds = HELD.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, w)) = helds.iter().find(|(h, _)| h == held) {
        return Some(w);
    }
    let w: &'static str = Box::leak(held.clone().into_boxed_str());
    helds.push((held.clone(), w));
    Some(w)
}

/// The name a plan and its records give a device of driver name `name`: a
/// known card's name ([`CARDS`]) when the driver name holds one, else the
/// driver name without its `NVIDIA ` prefix, each space written `_` (a
/// record word). Spelled once per driver name and kept for the process,
/// like a `--place` word.
#[must_use]
pub fn label(name: &str) -> &'static str {
    static LABELS: Mutex<Vec<(String, &'static str)>> = Mutex::new(Vec::new());
    if let Some(c) = CARDS.iter().find(|c| name.contains(c.name)) {
        return c.name;
    }
    let mut held = LABELS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, l)) = held.iter().find(|(n, _)| n == name) {
        return l;
    }
    let short = name.strip_prefix("NVIDIA ").unwrap_or(name).trim();
    let word: String = short
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect();
    let l: &'static str = Box::leak(
        if word.is_empty() {
            "device".into()
        } else {
            word
        }
        .into_boxed_str(),
    );
    held.push((name.to_string(), l));
    l
}

/// The visible devices, as a refusal lists them: each with its usable bytes
/// and what it had free at census time.
#[must_use]
pub fn visible(census: &[DeviceInfo]) -> String {
    if census.is_empty() {
        return "none".to_string();
    }
    census
        .iter()
        .map(|d| {
            format!(
                "cuda{} {} {} {} MiB usable, {} MiB free",
                d.ordinal,
                d.name,
                d.pci_bus,
                spec_of_device(d).usable_bytes() / MIB,
                d.free_bytes / MIB
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// How one card of a `--place` word is found among the visible devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The device of the `k`-th most usable bytes ([`spec_of_device`]),
    /// ties by the lower ordinal.
    Rank(u8),
    /// The one device whose driver name holds this card name, a [`CARDS`]
    /// name.
    Name(&'static str),
    /// The device of this process's ordinal.
    Ordinal(u32),
}

impl fmt::Display for Pick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pick::Rank(0) => write!(f, "the largest visible card"),
            Pick::Rank(k) => write!(f, "visible card {} by size", k + 1),
            Pick::Name(n) => write!(f, "{n}"),
            Pick::Ordinal(o) => write!(f, "cuda{o}"),
        }
    }
}

/// The placement aliases' picks: `a` the stage on the largest visible card,
/// `bp` that stage with the next-largest as its one expert tier, `gate` the
/// card named 3090 alone.
pub const ALIASES: [(&str, &[Pick]); 3] = [
    ("a", &[Pick::Rank(0)]),
    ("gate", &[Pick::Name(RTX_3090.name)]),
    ("bp", &[Pick::Rank(0), Pick::Rank(1)]),
];

/// What `pick` is on this workstation with no census, by name: a rank is
/// [`CARDS`] in its order (the most usable bytes first), a name that card;
/// an ordinal has no meaning without the census of the process that numbers
/// it.
#[must_use]
pub const fn workstation_spec(pick: Pick) -> Option<CardSpec> {
    match pick {
        Pick::Rank(k) => {
            let k = k as usize;
            if k < CARDS.len() {
                Some(CARDS[k])
            } else {
                None
            }
        }
        Pick::Name(n) => {
            let mut i = 0;
            while i < CARDS.len() {
                if same_str(CARDS[i].name, n) {
                    return Some(CARDS[i]);
                }
                i += 1;
            }
            None
        }
        Pick::Ordinal(_) => None,
    }
}

/// Whether two strings are one, at compile time.
const fn same_str(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// A bare number in a list word is an ordinal below this; a larger one is
/// read as a card name (`4090` names no card here, it is no ordinal).
pub const ORDINALS: u32 = 64;

/// The picks of a `--place` list word `<stage>[+<tier>…]`, the stage first:
/// each a [`CARDS`] name in any ASCII case, or an ordinal of this process —
/// `cuda<N>`, or a bare number under [`ORDINALS`] that is no card name (`0+1`
/// as `nvidia-smi` numbers the cards under `CUDA_DEVICE_ORDER=PCI_BUS_ID`). A
/// word of more than `max_tiers` tier cards, a part that is neither, and a
/// name or an ordinal given twice are refused by name, in that order.
pub fn word_picks(word: &str, max_tiers: usize) -> Result<Vec<Pick>, WordError> {
    let parts: Vec<&str> = word.split('+').collect();
    if parts.len() - 1 > max_tiers {
        return Err(WordError::TooManyTiers {
            word: word.to_string(),
            tiers: parts.len() - 1,
            max: max_tiers,
        });
    }
    let mut picks: Vec<Pick> = Vec::with_capacity(parts.len());
    for part in parts {
        let pick = if let Some(c) = CARDS.iter().find(|c| c.name.eq_ignore_ascii_case(part)) {
            Pick::Name(c.name)
        } else if let Some(o) = ordinal(part) {
            Pick::Ordinal(o)
        } else {
            return Err(WordError::UnknownCard {
                word: word.to_string(),
                name: part.to_string(),
            });
        };
        if picks.contains(&pick) {
            return Err(match pick {
                Pick::Ordinal(ordinal) => WordError::OrdinalTwice {
                    word: word.to_string(),
                    ordinal,
                },
                _ => WordError::CardTwice {
                    word: word.to_string(),
                    card: pick.to_string(),
                },
            });
        }
        picks.push(pick);
    }
    Ok(picks)
}

/// `part` as an ordinal: `cuda<N>` in any case, or a bare number under
/// [`ORDINALS`].
fn ordinal(part: &str) -> Option<u32> {
    let digits = |s: &str| {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .then(|| s.parse::<u32>().ok())
            .flatten()
    };
    let lower = part.to_ascii_lowercase();
    match lower.strip_prefix("cuda") {
        Some(n) => digits(n),
        None => digits(&lower).filter(|&o| o < ORDINALS),
    }
}

/// A `--place` list word [`word_picks`] refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WordError {
    TooManyTiers {
        word: String,
        tiers: usize,
        max: usize,
    },
    UnknownCard {
        word: String,
        name: String,
    },
    CardTwice {
        word: String,
        card: String,
    },
    OrdinalTwice {
        word: String,
        ordinal: u32,
    },
}

impl fmt::Display for WordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = CARDS.iter().map(|c| c.name).collect();
        match self {
            WordError::TooManyTiers { word, tiers, max } => write!(
                f,
                "{word} names {tiers} tier cards, past the {max} a plan's slot map can name"
            ),
            WordError::UnknownCard { word, name } => write!(
                f,
                "{word} names the card {name:?}, which is none of the card names {names:?} and \
                 no device ordinal (cuda<N>, or a number under {ORDINALS}, as this process's \
                 CUDA enumeration numbers the visible devices)"
            ),
            WordError::CardTwice { word, card } => write!(
                f,
                "{word} names the card {card} twice: one name is one device; two cards of a \
                 name are named by their ordinals"
            ),
            WordError::OrdinalTwice { word, ordinal } => {
                write!(f, "{word} names the device cuda{ordinal} twice")
            }
        }
    }
}

impl std::error::Error for WordError {}

/// The card specs `picks` resolve to on `census`, in pick order, each naming
/// its device ([`spec_of_device`]). A pick no visible device answers, a name
/// two devices carry, and a device two picks land on are refused by name,
/// with every visible device.
pub fn resolve(picks: &[Pick], census: &[DeviceInfo]) -> Result<Vec<CardSpec>, PickError> {
    let mut ranked: Vec<(u64, &DeviceInfo)> = census
        .iter()
        .map(|d| (spec_of_device(d).usable_bytes(), d))
        .collect();
    ranked.sort_by_key(|&(usable, d)| (std::cmp::Reverse(usable), d.ordinal));
    let refuse = |pick: Pick, why: PickWhy| PickError {
        pick,
        why,
        visible: visible(census),
    };
    let mut out: Vec<CardSpec> = Vec::with_capacity(picks.len());
    for &pick in picks {
        let d = match pick {
            Pick::Rank(k) => ranked
                .get(usize::from(k))
                .map(|&(_, d)| d)
                .ok_or_else(|| refuse(pick, PickWhy::Few(census.len())))?,
            Pick::Ordinal(o) => census
                .iter()
                .find(|d| d.ordinal == o)
                .ok_or_else(|| refuse(pick, PickWhy::Missing))?,
            Pick::Name(n) => {
                let named: Vec<&DeviceInfo> =
                    census.iter().filter(|d| d.name.contains(n)).collect();
                match named.as_slice() {
                    [d] => *d,
                    [] => return Err(refuse(pick, PickWhy::Missing)),
                    _ => {
                        let ordinals = named.iter().map(|d| d.ordinal).collect();
                        return Err(refuse(pick, PickWhy::Several(ordinals)));
                    }
                }
            }
        };
        if out.iter().any(|s| s.device == Some(d.device())) {
            return Err(refuse(pick, PickWhy::Twice(d.ordinal)));
        }
        out.push(spec_of_device(d));
    }
    Ok(out)
}

/// Why [`resolve`] refused a pick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PickWhy {
    /// A rank past the visible devices, of which there are this many.
    Few(usize),
    /// No visible device answers the pick.
    Missing,
    /// Several visible devices carry the pick's name: these ordinals.
    Several(Vec<u32>),
    /// The pick lands on this ordinal, which an earlier pick holds.
    Twice(u32),
}

/// A pick of a placement the visible devices do not answer ([`resolve`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PickError {
    pub pick: Pick,
    pub why: PickWhy,
    /// [`visible`] of the census.
    pub visible: String,
}

impl fmt::Display for PickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let PickError { pick, why, visible } = self;
        match why {
            PickWhy::Few(n) => {
                write!(f, "the placement takes {pick}, and {n} devices are visible")?
            }
            PickWhy::Missing => write!(f, "no visible device is {pick}")?,
            PickWhy::Several(o) => write!(
                f,
                "{} visible devices carry the name {pick} (cuda{}): name one by its ordinal",
                o.len(),
                o.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", cuda")
            )?,
            PickWhy::Twice(o) => write!(
                f,
                "{pick} is cuda{o}, which an earlier card of the placement already is"
            )?,
        }
        write!(f, " (visible: {visible})")
    }
}

impl std::error::Error for PickError {}

/// The ordinal of the one visible device a plan's card opens on: the card's
/// `device` when it was resolved ([`resolve`]) — the census device of its
/// UUID, at the ordinal the plan holds — else the one device whose driver
/// name holds the card's `name`, a census-free plan's card (a gate's). A
/// device out of view, at another ordinal, or a name none or several devices
/// carry is refused, naming what is in view.
pub fn device_on_host(
    name: &str,
    device: Option<DeviceId>,
    census: &[DeviceInfo],
) -> Result<u32, CardNotInView> {
    let refuse = |named: usize, at: Option<u32>| CardNotInView {
        card: name.to_string(),
        device,
        visible: visible(census),
        named,
        at,
    };
    match device {
        Some(dev) => match census.iter().find(|d| d.uuid == dev.uuid) {
            Some(d) if d.ordinal == dev.ordinal => Ok(d.ordinal),
            Some(d) => Err(refuse(1, Some(d.ordinal))),
            None => Err(refuse(0, None)),
        },
        None => {
            let named: Vec<u32> = census
                .iter()
                .filter(|d| d.name.contains(name))
                .map(|d| d.ordinal)
                .collect();
            match named.as_slice() {
                &[o] => Ok(o),
                _ => Err(refuse(named.len(), None)),
            }
        }
    }
}

/// The index `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`
/// lists (`listed`) for the device a plan's card opens on in `census`
/// ([`device_on_host`]): the one line whose bus id is that device's
/// (`cuDeviceGetPCIBusId`), compared as numbers — the driver writes a
/// four-digit domain (`0000:01:00.0`), nvidia-smi eight, in either case.
/// Two cards of one name sit on two buses, so each card names its own
/// index. A card not in view, and a bus nvidia-smi lists none or several
/// times, are refused naming what was listed.
pub fn listed_index(
    name: &str,
    device: Option<DeviceId>,
    census: &[DeviceInfo],
    listed: &str,
) -> Result<u32, String> {
    let ordinal = device_on_host(name, device, census).map_err(|e| e.to_string())?;
    let bus = census
        .iter()
        .find(|d| d.ordinal == ordinal)
        .map_or("", |d| d.pci_bus.as_str());
    let key =
        pci_key(bus).ok_or_else(|| format!("cuda{ordinal}'s PCI bus id {bus:?} reads as none"))?;
    let at: Vec<u32> = listed
        .lines()
        .filter_map(|line| {
            let (index, id) = line.split_once(',')?;
            (pci_key(id) == Some(key)).then(|| index.trim().parse().ok())?
        })
        .collect();
    match at.as_slice() {
        &[index] => Ok(index),
        _ => Err(format!(
            "{} nvidia-smi devices are on cuda{ordinal}'s bus {bus} ({name}), not one; it \
             listed:\n{listed}",
            at.len()
        )),
    }
}

/// A PCI bus id `domain:bus:device.function` as its four hex numbers.
fn pci_key(id: &str) -> Option<(u32, u32, u32, u32)> {
    let hex = |s: &str| u32::from_str_radix(s, 16).ok();
    let (domain, rest) = id.trim().split_once(':')?;
    let (bus, rest) = rest.split_once(':')?;
    let (dev, func) = rest.split_once('.')?;
    Some((hex(domain)?, hex(bus)?, hex(dev)?, hex(func)?))
}

/// A plan's card that is not one visible device ([`device_on_host`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardNotInView {
    pub card: String,
    /// The device the plan resolved the card to, when it did.
    pub device: Option<DeviceId>,
    /// [`visible`] of the census.
    pub visible: String,
    /// How many visible devices are the card: by UUID, or by name.
    pub named: usize,
    /// The ordinal the device's UUID has in this census, when it is not the
    /// plan's.
    pub at: Option<u32>,
}

impl fmt::Display for CardNotInView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let CardNotInView {
            card,
            device,
            visible,
            named,
            at,
        } = self;
        match (device, at) {
            (Some(dev), Some(at)) => {
                return write!(
                    f,
                    "the plan's device {dev} ({card}) is cuda{at} in this process, not \
                     cuda{}: the plan was resolved against another enumeration (visible: \
                     {visible})",
                    dev.ordinal
                );
            }
            (Some(dev), None) => {
                return write!(
                    f,
                    "the plan's device {dev} ({card}) is not in view (visible: {visible})"
                );
            }
            (None, _) => {}
        }
        if *named > 1 {
            return write!(
                f,
                "{named} visible devices carry the name of the plan's card {card} (visible: \
                 {visible}): a card planned by name needs exactly one; a placement by ordinals \
                 (--place cuda0+cuda1) plans each by its device"
            );
        }
        write!(
            f,
            "no visible device is the plan's card {card} (visible: {visible})"
        )?;
        if card == A6000.name {
            write!(
                f,
                ". --place a plans the stage on the largest visible card and --place gate on a \
                 3090; an A6000 out of view (CUDA_VISIBLE_DEVICES) must be put in view"
            )
        } else if card == RTX_3090.name {
            write!(
                f,
                ". --place gate runs the whole model on a 3090, --place a on the largest visible \
                 card; a 3090 out of view (CUDA_VISIBLE_DEVICES) must be put in view"
            )
        } else {
            Ok(())
        }
    }
}

impl std::error::Error for CardNotInView {}

#[cfg(test)]
mod tests {
    use super::{
        ALIASES, CardNotInView, DeviceInfo, Pick, PickError, PickWhy, WordError, census_usable,
        device_on_host, listed_index, resolve, spec_of_device, visible, word_picks,
        workstation_spec,
    };
    use crate::placement::PlacementError;
    use crate::placement::workstation::{
        A6000, CardSpec, MIB, RTX_3090, plan_a, plan_bp, plan_tiers, tier_batch_bytes,
    };

    /// `cuDeviceTotalMem` of this workstation's two cards [measured].
    const TOTAL_3090: u64 = 25_351_356_416;
    const TOTAL_A6000: u64 = 50_952_536_064;
    const NAME_3090: &str = "NVIDIA GeForce RTX 3090";
    const NAME_A6000: &str = "NVIDIA RTX A6000";
    const NAME_4090: &str = "NVIDIA GeForce RTX 4090";

    /// A fake census of `(name, total)` devices, by ordinal, each of its own
    /// UUID and its full usable bytes free (a quiet card) and no holder
    /// named.
    fn census(devices: &[(&str, u64)]) -> Vec<DeviceInfo> {
        devices
            .iter()
            .enumerate()
            .map(|(i, &(name, total_bytes))| DeviceInfo {
                ordinal: u32::try_from(i).expect("small"),
                name: name.to_string(),
                total_bytes,
                free_bytes: census_usable(total_bytes),
                uuid: [u8::try_from(i).expect("small") + 0x10; 16],
                pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
                held_by: None,
            })
            .collect()
    }

    fn alias(word: &str) -> &'static [Pick] {
        ALIASES
            .iter()
            .find(|(w, _)| *w == word)
            .map(|(_, p)| *p)
            .expect("an alias")
    }

    /// The word's picks resolved on `c`: the (name, ordinal) of each card,
    /// or the refusal.
    fn place(word: &str, c: &[DeviceInfo]) -> Result<Vec<(&'static str, u32)>, PickError> {
        let picks = match word {
            "a" | "bp" | "gate" => alias(word).to_vec(),
            _ => word_picks(word, 8).expect("a list word"),
        };
        Ok(resolve(&picks, c)?
            .into_iter()
            .map(|s| (s.name, s.device.expect("resolved").ordinal))
            .collect())
    }

    /// The census gives this workstation's two cards exactly their measured
    /// usable bytes: the reserve rule reproduces both known pairs.
    #[test]
    fn the_reserve_rule_reproduces_both_cards() {
        assert_eq!(census_usable(TOTAL_3090), RTX_3090.usable_bytes());
        assert_eq!(census_usable(TOTAL_A6000), A6000.usable_bytes());
        let c = census(&[(NAME_3090, TOTAL_3090), (NAME_A6000, TOTAL_A6000)]);
        let (t, a) = (spec_of_device(&c[0]), spec_of_device(&c[1]));
        assert_eq!(
            (t, a),
            (
                CardSpec {
                    device: Some(c[0].device()),
                    free_bytes: Some(census_usable(TOTAL_3090)),
                    ..RTX_3090
                },
                CardSpec {
                    device: Some(c[1].device()),
                    free_bytes: Some(census_usable(TOTAL_A6000)),
                    ..A6000
                }
            )
        );
        // Another device: its census total, the sub-MiB tail its reserve,
        // its label from the driver name.
        let other = census(&[(NAME_4090, 24_564 * MIB + 77)]);
        let s = spec_of_device(&other[0]);
        assert_eq!(
            (s.name, s.total_bytes, s.usable_bytes()),
            ("GeForce_RTX_4090", 24_564 * MIB + 77, 24_564 * MIB)
        );
        // A known name whose total gives other bytes (ECC on, say) takes the
        // census rule, not the measured constant.
        let short = census(&[(NAME_A6000, TOTAL_A6000 - 2048 * MIB)]);
        let s = spec_of_device(&short[0]);
        assert_eq!(
            (s.name, s.usable_bytes()),
            ("A6000", A6000.usable_bytes() - 2048 * MIB)
        );
    }

    /// The census's free reading and its holder list ride the device's card
    /// spec for the plan's budget term and its refusals, and a refusal
    /// listing the visible devices names what each had free; the spec's own
    /// identity figures stay the total's.
    #[test]
    fn the_free_reading_rides_the_spec() {
        let mut c = census(&[(NAME_3090, TOTAL_3090), (NAME_A6000, TOTAL_A6000)]);
        c[0].free_bytes = 8192 * MIB;
        c[0].held_by = Some("pid 7 (977 MiB)".to_string());
        let s = spec_of_device(&c[0]);
        assert_eq!(
            (s.free_bytes, s.held_by),
            (Some(8192 * MIB), Some("pid 7 (977 MiB)"))
        );
        assert_eq!(s.usable_bytes(), RTX_3090.usable_bytes());
        // A device the census read whole and named no holder of.
        assert_eq!(
            (
                spec_of_device(&c[1]).free_bytes,
                spec_of_device(&c[1]).held_by
            ),
            (Some(census_usable(TOTAL_A6000)), None)
        );
        let text = visible(&c);
        assert!(
            text.starts_with(&format!(
                "cuda0 {NAME_3090} 0000:41:00.0 {} MiB usable, 8192 MiB free; ",
                RTX_3090.usable_bytes() / MIB
            )),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "cuda1 {NAME_A6000} 0000:42:00.0 {} MiB usable, {} MiB free",
                A6000.usable_bytes() / MIB,
                census_usable(TOTAL_A6000) / MIB
            )),
            "{text}"
        );
    }

    /// Every fake host: what `a`, `bp`, `gate`, `0+1` and `1+0` resolve to,
    /// or why they are refused.
    #[test]
    fn the_words_on_every_fake_host() {
        type Want = [Result<Vec<(&'static str, u32)>, PickWhy>; 5];
        let ok = |v: &[(&'static str, u32)]| Ok(v.to_vec());
        let r4090 = ("GeForce_RTX_4090", 0);
        let hosts: Vec<(&str, Vec<DeviceInfo>, Want)> = vec![
            (
                "the box, BLOOMERY_CARD=both (3090 first)",
                census(&[(NAME_3090, TOTAL_3090), (NAME_A6000, TOTAL_A6000)]),
                [
                    ok(&[("A6000", 1)]),
                    ok(&[("A6000", 1), ("3090", 0)]),
                    ok(&[("3090", 0)]),
                    ok(&[("3090", 0), ("A6000", 1)]),
                    ok(&[("A6000", 1), ("3090", 0)]),
                ],
            ),
            (
                "the box, A6000 first",
                census(&[(NAME_A6000, TOTAL_A6000), (NAME_3090, TOTAL_3090)]),
                [
                    ok(&[("A6000", 0)]),
                    ok(&[("A6000", 0), ("3090", 1)]),
                    ok(&[("3090", 1)]),
                    ok(&[("A6000", 0), ("3090", 1)]),
                    ok(&[("3090", 1), ("A6000", 0)]),
                ],
            ),
            (
                "one 4090",
                census(&[(NAME_4090, 24_564 * MIB)]),
                [
                    ok(&[r4090]),
                    Err(PickWhy::Few(1)),
                    Err(PickWhy::Missing),
                    Err(PickWhy::Missing),
                    Err(PickWhy::Missing),
                ],
            ),
            (
                "two 3090s",
                census(&[(NAME_3090, TOTAL_3090), (NAME_3090, TOTAL_3090)]),
                [
                    ok(&[("3090", 0)]),
                    ok(&[("3090", 0), ("3090", 1)]),
                    Err(PickWhy::Several(vec![0, 1])),
                    ok(&[("3090", 0), ("3090", 1)]),
                    ok(&[("3090", 1), ("3090", 0)]),
                ],
            ),
            (
                "4090 + 3090",
                census(&[(NAME_4090, 24_564 * MIB), (NAME_3090, TOTAL_3090)]),
                [
                    ok(&[r4090]),
                    ok(&[r4090, ("3090", 1)]),
                    ok(&[("3090", 1)]),
                    ok(&[r4090, ("3090", 1)]),
                    ok(&[("3090", 1), r4090]),
                ],
            ),
            (
                "one 3090",
                census(&[(NAME_3090, TOTAL_3090)]),
                [
                    ok(&[("3090", 0)]),
                    Err(PickWhy::Few(1)),
                    ok(&[("3090", 0)]),
                    Err(PickWhy::Missing),
                    Err(PickWhy::Missing),
                ],
            ),
            (
                "two A6000s",
                census(&[(NAME_A6000, TOTAL_A6000), (NAME_A6000, TOTAL_A6000)]),
                [
                    ok(&[("A6000", 0)]),
                    ok(&[("A6000", 0), ("A6000", 1)]),
                    Err(PickWhy::Missing),
                    ok(&[("A6000", 0), ("A6000", 1)]),
                    ok(&[("A6000", 1), ("A6000", 0)]),
                ],
            ),
            (
                "no device",
                Vec::new(),
                [
                    Err(PickWhy::Few(0)),
                    Err(PickWhy::Few(0)),
                    Err(PickWhy::Missing),
                    Err(PickWhy::Missing),
                    Err(PickWhy::Missing),
                ],
            ),
        ];
        for (host, c, want) in hosts {
            for (word, want) in ["a", "bp", "gate", "0+1", "1+0"].into_iter().zip(want) {
                let got = place(word, &c).map_err(|e| {
                    let text = e.to_string();
                    assert!(text.contains("(visible: "), "{host} {word}: {text}");
                    if c.is_empty() {
                        assert!(text.ends_with("(visible: none)"), "{host} {word}: {text}");
                    }
                    e.why
                });
                assert_eq!(got, want, "{host}: --place {word}");
            }
        }
    }

    /// A card no measured figure names, as its driver names it, of a 96 GB
    /// class total [assumed: a 97,887 MiB total less the 548 MiB reserve
    /// measured on the A6000].
    const NAME_PRO6000: &str = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";
    const TOTAL_PRO6000: u64 = 97_339 * MIB;

    /// One, two and four such cards: each its census total, labelled from
    /// the driver name; `a` the lowest ordinal among equal usable bytes —
    /// whatever each card read free, so a card another process holds is
    /// still picked first — `bp` the two lowest, and `gate`, the 3090 by
    /// name, none. A card of fewer usable bytes (ECC on, say) ranks below
    /// the others.
    #[test]
    fn equal_large_cards_rank_by_ordinal() {
        let label = "RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition";
        for n in [1, 2, 4] {
            let mut c = census(&vec![(NAME_PRO6000, TOTAL_PRO6000); n]);
            let s = spec_of_device(&c[0]);
            assert_eq!(
                (s.name, s.usable_bytes(), s.driver_reserve_bytes),
                (label, TOTAL_PRO6000, 0)
            );
            assert_eq!(place("a", &c), Ok(vec![(label, 0)]), "{n} cards");
            let bp = if n == 1 {
                Err(PickWhy::Few(1))
            } else {
                Ok(vec![(label, 0), (label, 1)])
            };
            assert_eq!(place("bp", &c).map_err(|e| e.why), bp, "{n} cards");
            assert_eq!(
                place("gate", &c).map_err(|e| e.why),
                Err(PickWhy::Missing),
                "{n} cards"
            );
            c[0].free_bytes = 8192 * MIB;
            assert_eq!(
                place("a", &c),
                Ok(vec![(label, 0)]),
                "{n} cards, cuda0 held"
            );
        }
        let mut c = census(&[(NAME_PRO6000, TOTAL_PRO6000); 4]);
        c[0].total_bytes -= 2048 * MIB;
        assert_eq!(place("a", &c), Ok(vec![(label, 1)]));
        assert_eq!(place("bp", &c), Ok(vec![(label, 1), (label, 2)]));
    }

    /// Two 3090s plan as a stage and its tier ([`plan_tiers`]), where the
    /// same two cards by name (no census) are one device listed twice; the
    /// box's resolved `a` and `bp` are plan (a) and plan (b′) but for the
    /// devices they name.
    #[test]
    fn two_3090s_plan_as_stage_and_tier() {
        let batch = tier_batch_bytes(5120, 2304, 6, 512);
        let two = census(&[(NAME_3090, TOTAL_3090), (NAME_3090, TOTAL_3090)]);
        let specs = resolve(alias("bp"), &two).expect("two 3090s");
        let m = plan_tiers(43, specs[0], &specs[1..], None, batch).expect("stage + tier");
        assert_eq!(m.cards[0].device, Some(two[0].device()));
        assert_eq!(m.tiers[0].device, Some(two[1].device()));
        assert!(matches!(
            plan_tiers(43, RTX_3090, &[RTX_3090], None, batch),
            Err(PlacementError::DeviceTwice { cards: 2, .. })
        ));
        let strip = |mut m: crate::placement::Machine| {
            for c in m.cards.iter_mut().chain(m.tiers.iter_mut()) {
                c.device = None;
                c.free_bytes = None;
                c.held_by = None;
            }
            m
        };
        for order in [
            census(&[(NAME_3090, TOTAL_3090), (NAME_A6000, TOTAL_A6000)]),
            census(&[(NAME_A6000, TOTAL_A6000), (NAME_3090, TOTAL_3090)]),
        ] {
            let a = resolve(alias("a"), &order).expect("a");
            let m = plan_tiers(43, a[0], &[], None, batch).expect("a");
            assert_eq!(strip(m), plan_a(43));
            let bp = resolve(alias("bp"), &order).expect("bp");
            let m = plan_tiers(43, bp[0], &bp[1..], None, batch).expect("bp");
            assert_eq!(strip(m), plan_bp(43, None, batch));
        }
    }

    /// With no census, the aliases are this workstation's cards by name.
    #[test]
    fn aliases_without_a_census_are_this_workstation() {
        let specs = |w| {
            alias(w)
                .iter()
                .map(|&p| workstation_spec(p))
                .collect::<Vec<_>>()
        };
        assert_eq!(specs("a"), vec![Some(A6000)]);
        assert_eq!(specs("bp"), vec![Some(A6000), Some(RTX_3090)]);
        assert_eq!(specs("gate"), vec![Some(RTX_3090)]);
        assert_eq!(workstation_spec(Pick::Ordinal(0)), None);
    }

    /// The list word: names in any case, ordinals as `cuda<N>` or a bare
    /// number under the bound, the stage first; more tiers than the slot map
    /// names, a part that is neither, and a name or ordinal given twice are
    /// refused by name, the count first.
    #[test]
    fn word_picks_reads_the_list_word() {
        use Pick::{Name, Ordinal};
        assert_eq!(
            word_picks("a6000+3090", 8),
            Ok(vec![Name("A6000"), Name("3090")])
        );
        assert_eq!(
            word_picks("A6000+3090", 8),
            Ok(vec![Name("A6000"), Name("3090")])
        );
        assert_eq!(word_picks("3090", 8), Ok(vec![Name("3090")]));
        assert_eq!(word_picks("0+1", 8), Ok(vec![Ordinal(0), Ordinal(1)]));
        assert_eq!(
            word_picks("CUDA1+cuda0", 8),
            Ok(vec![Ordinal(1), Ordinal(0)])
        );
        assert_eq!(word_picks("cuda4090", 8), Ok(vec![Ordinal(4090)]));
        assert_eq!(
            word_picks("a6000+1", 8),
            Ok(vec![Name("A6000"), Ordinal(1)])
        );
        let many = ["a6000"; 10].join("+");
        assert_eq!(
            word_picks(&many, 8),
            Err(WordError::TooManyTiers {
                word: many.clone(),
                tiers: 9,
                max: 8
            })
        );
        assert!(matches!(
            word_picks("0+1", 0),
            Err(WordError::TooManyTiers {
                tiers: 1,
                max: 0,
                ..
            })
        ));
        for (word, name) in [
            ("a6000+4090", "4090"),
            ("b", "b"),
            ("a6000+", ""),
            ("64", "64"),
            ("cuda", "cuda"),
            ("cuda-1", "cuda-1"),
        ] {
            assert_eq!(
                word_picks(word, 8),
                Err(WordError::UnknownCard {
                    word: word.to_string(),
                    name: name.to_string()
                }),
                "{word}"
            );
        }
        assert_eq!(
            word_picks("3090+3090", 8),
            Err(WordError::CardTwice {
                word: "3090+3090".to_string(),
                card: "3090".to_string()
            })
        );
        assert_eq!(
            word_picks("0+cuda0", 8),
            Err(WordError::OrdinalTwice {
                word: "0+cuda0".to_string(),
                ordinal: 0
            })
        );
        let e = word_picks("a6000+A6000", 8).expect_err("twice");
        assert_eq!(
            e.to_string(),
            "a6000+A6000 names the card A6000 twice: one name is one device; two cards of a name \
             are named by their ordinals"
        );
        // A name word and an ordinal word that land on one device.
        let c = census(&[(NAME_A6000, TOTAL_A6000)]);
        let e = resolve(&word_picks("a6000+0", 8).expect("picks"), &c).expect_err("one device");
        assert_eq!(e.why, PickWhy::Twice(0));
    }

    /// The open's device: a resolved card by its UUID at its ordinal; a
    /// census-free card by the one device that carries its name. Out of
    /// view, at another ordinal, by a name none or two carry: refused, with
    /// what is visible.
    #[test]
    fn device_on_host_finds_the_plan_card() {
        let both = census(&[(NAME_3090, TOTAL_3090), (NAME_A6000, TOTAL_A6000)]);
        assert_eq!(device_on_host("A6000", None, &both), Ok(1));
        assert_eq!(device_on_host("3090", None, &both), Ok(0));
        assert_eq!(
            device_on_host("A6000", Some(both[1].device()), &both),
            Ok(1)
        );
        let no_a6000 = device_on_host("A6000", None, &both[..1]).expect_err("no A6000");
        assert_eq!(no_a6000.named, 0);
        assert!(no_a6000.to_string().contains("--place gate"), "{no_a6000}");
        let no_3090 = device_on_host("3090", None, &both[1..]).expect_err("no 3090");
        assert!(no_3090.to_string().contains("--place a"), "{no_3090}");
        let two = census(&[(NAME_3090, TOTAL_3090), (NAME_3090, TOTAL_3090)]);
        let e: CardNotInView = device_on_host("3090", None, &two).expect_err("two 3090s");
        assert_eq!(e.named, 2);
        assert!(e.to_string().contains("--place cuda0+cuda1"), "{e}");
        assert_eq!(device_on_host("3090", Some(two[1].device()), &two), Ok(1));
        // The A6000 resolved at ordinal 1 of the 3090-first enumeration, in
        // a process that numbers it 0, and in one that does not see it.
        let a6000_first = census(&[(NAME_A6000, TOTAL_A6000)]);
        let mut moved = a6000_first.clone();
        moved[0].uuid = both[1].uuid;
        let e = device_on_host("A6000", Some(both[1].device()), &moved).expect_err("moved");
        assert_eq!(e.at, Some(0));
        assert!(e.to_string().contains("another enumeration"), "{e}");
        let e = device_on_host("A6000", Some(both[1].device()), &both[..1]).expect_err("gone");
        assert!(
            e.to_string().starts_with(&format!(
                "the plan's device cuda1 (A6000) is not in view (visible: cuda0 {NAME_3090} \
                 0000:41:00.0"
            )),
            "{e}"
        );
    }

    /// Two cards of one name: the census's two devices with the driver's bus
    /// ids (four-digit domain, upper-case hex), and nvidia-smi's list of the
    /// same buses (eight-digit domain, lower case) in its own order.
    fn twins() -> (Vec<DeviceInfo>, &'static str) {
        let mut c = census(&[(NAME_3090, TOTAL_3090), (NAME_3090, TOTAL_3090)]);
        c[0].pci_bus = "0000:4B:00.0".to_string();
        c[1].pci_bus = "0000:01:00.0".to_string();
        (c, "0, 00000000:01:00.0\n1, 00000000:4b:00.0\n")
    }

    /// Each card of two of one name names its own nvidia-smi index, by its
    /// device's bus: cuda0 is nvidia-smi's 1 and cuda1 its 0.
    #[test]
    fn the_listed_index_follows_the_device_bus() {
        let (c, listed) = twins();
        let at = |o: usize| listed_index("3090", Some(c[o].device()), &c, listed);
        assert_eq!(at(0), Ok(1));
        assert_eq!(at(1), Ok(0));
    }

    /// A census-free card of a name two devices carry, and a bus nvidia-smi
    /// does not list, are refused by name.
    #[test]
    fn an_ambiguous_name_and_an_unlisted_bus_are_refused() {
        let (c, listed) = twins();
        let e = listed_index("3090", None, &c, listed).expect_err("two of one name");
        assert!(e.contains("3090"), "{e}");
        let e = listed_index("3090", Some(c[0].device()), &c, "0, 00000000:01:00.0\n")
            .expect_err("bus 4b unlisted");
        assert!(
            e.starts_with("0 nvidia-smi devices are on cuda0's bus 0000:4B:00.0"),
            "{e}"
        );
    }
}
