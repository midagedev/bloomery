//! The prompt ubatch of a dense whole load (a Body35 trunk with no routed FFN, whose only load is the
//! whole one): the user's size when one is asked, else the largest of [`LADDER`] whose whole need fits
//! the card at the asked context and slots. The serve seats decide it where they take the whole-fit
//! verdict (the serve seat's load, the decide seat's backbone open) through [`settle`],
//! which prints the one `ubatch` line [`text`] writes; the serve gates read that line back through
//! [`choices`] and hold it to its own terms ([`judge`]), so a rule change is one edit on each side.

/// The sizes an unasked dense whole load tries, largest first: the ubatch cap, then each half down to
/// the least that still runs the prompt through the GEMM walk in useful widths.
pub const LADDER: [usize; 4] = [4096, 2048, 1024, 512];

/// What asked for a ubatch size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `-ub` / `--ubatch-size` (`serve::flag::UBATCH`).
    Flag,
    /// The ubatch lever's environment variable.
    Env,
}

impl Source {
    /// The word the `ubatch` line names it by.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Source::Flag => "flag",
            Source::Env => "env",
        }
    }
}

/// A size the user asked for, which the load takes exactly or refuses by name with its terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ask {
    pub size: usize,
    pub source: Source,
    /// How a refusal spells what asked: the flag's, or the environment variable's name.
    pub name: &'static str,
}

/// How the ubatch of a dense whole load is settled ([`settle`]).
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// The load's own ubatch, counted as it stands: a run that names its size.
    Open,
    /// The user's size, taken exactly and refused by name with its terms when the whole load does not
    /// fit it.
    Asked(Ask),
    /// The largest of [`LADDER`] whose whole need fits.
    Fit,
}

impl Rule {
    /// The rule for a load whose size the user may name: `flag` (`-ub`, `--ubatch-size`) over `env`
    /// (the size and the name of the environment variable that gave it), else the card's room. A flag
    /// outside `1..=cap` is refused by name.
    ///
    /// # Errors
    /// The flag's size when it is not one the engine runs.
    pub fn of(
        flag: Option<usize>,
        env: Option<(usize, &'static str)>,
        cap: usize,
    ) -> Result<Rule, String> {
        let ask = match flag {
            Some(size) if (1..=cap).contains(&size) => Some(Ask {
                size,
                source: Source::Flag,
                name: "--ubatch-size",
            }),
            Some(size) => {
                return Err(format!(
                    "--ubatch-size {size} is not a ubatch size in 1..={cap}"
                ));
            }
            None => env.map(|(size, name)| Ask {
                size,
                source: Source::Env,
                name,
            }),
        };
        Ok(ask.map_or(Rule::Fit, Rule::Asked))
    }

    /// The user's ask, when the rule is one.
    #[must_use]
    pub fn asked(self) -> Option<Ask> {
        match self {
            Rule::Asked(a) => Some(a),
            _ => None,
        }
    }

    /// The sizes to take the verdict at, in order; `open` is the load's own.
    fn sizes(self, open: usize) -> Vec<usize> {
        match self {
            Rule::Open => vec![open],
            Rule::Asked(a) => vec![a.size],
            Rule::Fit => LADDER.to_vec(),
        }
    }
}

/// The largest of [`LADDER`] the card holds, walked from the top as an unasked load walks it: `fits(u)`
/// takes the whole-fit verdict at the size `u` and says whether the whole load fits it. `None`: none of
/// the ladder fits.
///
/// # Errors
/// The verdict's own error.
pub fn largest_fit<E>(mut fits: impl FnMut(usize) -> Result<bool, E>) -> Result<Option<usize>, E> {
    for u in LADDER {
        if fits(u)? {
            return Ok(Some(u));
        }
    }
    Ok(None)
}

/// The refusal of the user's `ask` at the least context a ladder tries, `(ctx, slots)` a slot's
/// positions and the slots, where the whole load takes `need` bytes and the card holds `room`: it names
/// the remedy that applies to a size the user chose, not the context's (fewer slots, a smaller file),
/// and the `largest` size of the ladder the card holds at that context and slots ([`largest_fit`]), or
/// that none does down to the least size.
#[must_use]
pub fn asked_short(
    ask: Ask,
    (ctx, slots): (usize, usize),
    need: u64,
    room: u64,
    largest: Option<usize>,
) -> String {
    let name = ask.name;
    let fit = largest.map_or_else(
        || format!("none at {}", LADDER[LADDER.len() - 1]),
        |u| u.to_string(),
    );
    format!(
        "{name} {}: the whole load at {ctx} positions a slot over {slots} slot(s) takes {need} B, and the \
         card holds {room} B: ask a smaller {name} (the largest that fits here: {fit}), or leave it unset \
         and the card picks the largest size that fits",
        ask.size
    )
}

/// One size's whole-fit verdict as [`settle`] reads it.
pub struct Step<T> {
    /// The whole need and what the card had, when the verdict has terms.
    pub terms: Option<(u64, u64)>,
    /// Whether the whole load fits at this size.
    pub fits: bool,
    /// The whole load refuses the file itself, which no size changes.
    pub refused: bool,
    /// What the caller keeps of this size's load.
    pub kept: T,
}

/// The ubatch a dense whole load settles on under `rule`, with the step of that size. `verdict(u)` takes
/// the whole-fit verdict at the size `u`; the sizes are tried in the rule's order and the walk ends at the
/// first that fits, at a refusal of the file or at the last size, whose step is then the one the caller
/// refuses with. Unless the rule is [`Rule::Open`], one `ubatch` line states the choice ([`text`]) on
/// stderr, before the caller's `whole fit` line of the size taken. `(open, ctx, slots)` are the load's
/// own size and the context and slots the verdicts are taken at.
///
/// # Errors
/// The verdict's own error.
pub fn settle<T, E: From<String>>(
    rule: Rule,
    (open, ctx, slots): (usize, usize, usize),
    mut verdict: impl FnMut(usize) -> Result<Step<T>, E>,
) -> Result<(usize, Step<T>), E> {
    let mut tried = Vec::new();
    let mut had = 0;
    let mut last = None;
    for u in rule.sizes(open) {
        let step = verdict(u)?;
        if let Some((need, budget)) = step.terms {
            tried.push((u, need));
            had = budget;
        }
        let done = step.fits || step.refused;
        last = Some((u, step));
        if done {
            break;
        }
    }
    let (u, step) =
        last.ok_or_else(|| E::from("a dense whole load with no ubatch size to try".to_owned()))?;
    if !matches!(rule, Rule::Open) && !tried.is_empty() {
        let asked = match rule {
            Rule::Asked(a) => Some(a.source),
            _ => None,
        };
        let pick = (asked.is_some() || step.fits).then_some(u);
        let choice = Choice {
            ctx,
            slots,
            asked,
            tried,
            had,
            pick,
        };
        eprintln!("{}", text(&choice));
    }
    Ok((u, step))
}

/// What one `ubatch` line states: the sizes whose whole need the load took, in the order it took them,
/// what the card had, and the size it settled on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    /// The context and the slots the verdicts were taken at.
    pub ctx: usize,
    pub slots: usize,
    /// `Some`: the user's ask; `None`: the card's room decided.
    pub asked: Option<Source>,
    /// Each size tried and its whole need in bytes.
    pub tried: Vec<(usize, u64)>,
    /// What the card had, the same for every size.
    pub had: u64,
    /// The size taken; `None` when the room fits none of the ladder.
    pub pick: Option<usize>,
}

/// The line a dense whole load prints once it has settled its ubatch, next to the `whole fit` line of
/// the size it took.
#[must_use]
pub fn text(c: &Choice) -> String {
    let (rule, from) = match c.asked {
        Some(s) => ("asked", s.word()),
        None => ("fit", "card"),
    };
    let tried: Vec<String> = c.tried.iter().map(|(u, n)| format!("{u}:{n}")).collect();
    let pick = c.pick.map_or_else(|| "none".to_owned(), |u| u.to_string());
    format!(
        "ubatch rule={rule} from={from} pick={pick} ctx={} slots={} tried={} had={}",
        c.ctx,
        c.slots,
        tried.join(","),
        c.had
    )
}

/// The field `name` of a line of `name=value` words.
fn word<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(name)?.strip_prefix('='))
}

/// The choice a `ubatch` line states; `None` for a line that is not one or whose fields do not read.
#[must_use]
pub fn parse(line: &str) -> Option<Choice> {
    let mut words = line.split_whitespace();
    if words.next()? != "ubatch" {
        return None;
    }
    let asked = match (word(line, "rule")?, word(line, "from")?) {
        ("fit", "card") => None,
        ("asked", "flag") => Some(Source::Flag),
        ("asked", "env") => Some(Source::Env),
        _ => return None,
    };
    let pick = match word(line, "pick")? {
        "none" => None,
        n => Some(n.parse().ok()?),
    };
    let tried = word(line, "tried")?
        .split(',')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let (u, n) = t.split_once(':')?;
            Some((u.parse().ok()?, n.parse().ok()?))
        })
        .collect::<Option<Vec<(usize, u64)>>>()?;
    Some(Choice {
        ctx: word(line, "ctx")?.parse().ok()?,
        slots: word(line, "slots")?.parse().ok()?,
        asked,
        tried,
        had: word(line, "had")?.parse().ok()?,
        pick,
    })
}

/// Every `ubatch` line of a server's stderr, in order.
#[must_use]
pub fn choices(log: &str) -> Vec<Choice> {
    log.lines().filter_map(parse).collect()
}

/// Holds a choice to its own terms: an asked size is the one size tried and the pick; an unasked one
/// tried [`LADDER`] from the top down to the first size whose need is within what the card had (every
/// size before it over), took that size, and took none when every size was over. The reason is the
/// first term that does not hold.
///
/// # Errors
/// The term of the choice that breaks the rule.
pub fn judge(c: &Choice) -> Result<(), String> {
    if c.asked.is_some() {
        return match (c.tried.as_slice(), c.pick) {
            ([(u, _)], Some(p)) if *u == p => Ok(()),
            _ => Err(format!(
                "an asked size is the one size tried and the pick, got {:?} and {:?}",
                c.tried, c.pick
            )),
        };
    }
    let mut want = None;
    for (i, &(u, need)) in c.tried.iter().enumerate() {
        if LADDER.get(i) != Some(&u) {
            return Err(format!(
                "the sizes tried are the ladder {LADDER:?} from the top, got {:?}",
                c.tried
            ));
        }
        if need <= c.had {
            want = Some(u);
            break;
        }
    }
    let walked = match want {
        Some(u) => LADDER.iter().position(|&l| l == u).map(|i| i + 1),
        None => Some(LADDER.len()),
    };
    if walked != Some(c.tried.len()) {
        return Err(format!(
            "the walk goes on past a size that fits, or stops short of the ladder: tried {:?} of {} had",
            c.tried, c.had
        ));
    }
    if want != c.pick {
        return Err(format!(
            "the largest size whose need is within {} B is {want:?}, the line picks {:?}",
            c.had, c.pick
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Ask, Choice, LADDER, Rule, Source, Step, asked_short, choices, judge, largest_fit, parse,
        settle, text,
    };

    fn fit(tried: &[(usize, u64)], had: u64, pick: Option<usize>) -> Choice {
        Choice {
            ctx: 4096,
            slots: 1,
            asked: None,
            tried: tried.to_vec(),
            had,
            pick,
        }
    }

    /// The `ubatch` line reads back as the choice it states, and the judge holds a choice to the rule:
    /// the largest of the ladder within what the card had, the walk stopping there, an asked size
    /// taken as it stands.
    #[test]
    fn the_ubatch_line_reads_back_and_is_judged_by_its_terms() {
        assert_eq!(LADDER, [4096, 2048, 1024, 512]);
        let ok = fit(&[(4096, 90), (2048, 60)], 70, Some(2048));
        assert_eq!(
            text(&ok),
            "ubatch rule=fit from=card pick=2048 ctx=4096 slots=1 tried=4096:90,2048:60 had=70"
        );
        assert_eq!(parse(&text(&ok)), Some(ok.clone()));
        assert_eq!(judge(&ok), Ok(()));
        let log = format!("whole fit at ctx 4096: x\n{}\nlistening\n", text(&ok));
        assert_eq!(choices(&log), vec![ok]);
        // Every size over the card: none is the pick, and the whole ladder was tried.
        let none = fit(&[(4096, 90), (2048, 80), (1024, 75), (512, 71)], 70, None);
        assert_eq!(parse(&text(&none)), Some(none.clone()));
        assert_eq!(judge(&none), Ok(()));
        // The first size fits: the walk is one size long.
        assert_eq!(judge(&fit(&[(4096, 50)], 70, Some(4096))), Ok(()));
        // A pick under the largest size that fits, a walk that skips a rung or goes on past a fit,
        // and a pick whose need is over what the card had are each refused.
        assert!(judge(&fit(&[(4096, 90), (2048, 60)], 70, Some(1024))).is_err());
        assert!(judge(&fit(&[(4096, 90), (1024, 60)], 70, Some(1024))).is_err());
        assert!(judge(&fit(&[(4096, 50), (2048, 40)], 70, Some(2048))).is_err());
        assert!(judge(&fit(&[(4096, 90), (2048, 80)], 70, Some(2048))).is_err());
        // An asked size is the one size tried and the pick, over the card or not.
        let asked = Choice {
            asked: Some(Source::Flag),
            ..fit(&[(4096, 90)], 70, Some(4096))
        };
        assert_eq!(parse(&text(&asked)), Some(asked.clone()));
        assert_eq!(judge(&asked), Ok(()));
        assert!(
            judge(&Choice {
                pick: Some(512),
                ..asked
            })
            .is_err()
        );
        assert_eq!(
            parse("ubatch rule=fit from=flag pick=1 ctx=1 slots=1 tried= had=1"),
            None
        );
        assert_eq!(parse("whole fit at ctx 4096: x"), None);
    }

    /// The refusal of an asked size names the largest ladder size the card holds, found from the top
    /// down, or that none does down to the least; a verdict that fails ends the walk with its error.
    #[test]
    fn an_asked_size_refusal_names_the_largest_size_that_fits() {
        let at = |room: u64| largest_fit(|u| Ok::<_, String>(u as u64 * 10 <= room));
        assert_eq!(at(25_000), Ok(Some(2048)));
        assert_eq!(at(50_000), Ok(Some(4096)));
        assert_eq!(at(5_120), Ok(Some(512)));
        assert_eq!(at(5_119), Ok(None));
        assert_eq!(
            largest_fit(|_| Err::<bool, _>("no verdict")),
            Err("no verdict")
        );
        let ask = Ask {
            size: 4096,
            source: Source::Flag,
            name: "--ubatch-size",
        };
        let short = |largest| asked_short(ask, (4096, 1), 90, 70, largest);
        assert!(short(Some(1024)).contains(
            "takes 90 B, and the card holds 70 B: ask a smaller --ubatch-size (the largest that fits here: 1024),"
        ));
        assert!(short(None).contains("(the largest that fits here: none at 512),"));
    }

    /// The walk the line states: the ladder from the top to the first size whose need is within the card,
    /// an asked size alone, the load's own size alone, and a refused file at the first size.
    #[test]
    fn settle_walks_the_ladder_to_the_first_size_that_fits() {
        let need = |u: usize| u64::try_from(u).expect("a size") * 10;
        let at = |rule: Rule, had: u64, refused: bool| {
            let mut tried = Vec::new();
            let (u, step) = settle::<usize, String>(rule, (777, 4096, 1), |u| {
                tried.push(u);
                Ok(Step {
                    terms: (!refused).then_some((need(u), had)),
                    fits: need(u) <= had,
                    refused,
                    kept: u,
                })
            })
            .expect("a verdict");
            assert_eq!(u, step.kept);
            (u, step.fits, tried)
        };
        assert_eq!(at(Rule::Fit, 25_000, false), (2048, true, vec![4096, 2048]));
        assert_eq!(at(Rule::Fit, 50_000, false), (4096, true, vec![4096]));
        let all = vec![4096, 2048, 1024, 512];
        assert_eq!(at(Rule::Fit, 100, false), (512, false, all));
        assert_eq!(at(Rule::Fit, 100, true), (4096, false, vec![4096]));
        let asked = Rule::of(Some(1024), Some((512, "ENV")), 4096).expect("in range");
        assert_eq!(at(asked, 100, false), (1024, false, vec![1024]));
        let env = Rule::of(None, Some((512, "ENV")), 4096).expect("no flag");
        assert_eq!(at(env, 100, false), (512, false, vec![512]));
        assert_eq!(at(Rule::Open, 100, false), (777, false, vec![777]));
        assert!(matches!(Rule::of(None, None, 4096), Ok(Rule::Fit)));
        assert!(Rule::of(Some(8192), None, 4096).is_err());
        assert!(Rule::of(Some(0), None, 4096).is_err());
    }
}
