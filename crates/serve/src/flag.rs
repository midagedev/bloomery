//! The seats' flags as llama-server spells them: the context flag's
//! spellings ([`CTX`]) and a flag's number, whose refusal names the flag and
//! the value ([`number`]).

use std::fmt::Display;
use std::str::FromStr;

/// The spellings of the context flag: `--ctx`, and llama-server's
/// `--ctx-size` and `-c`.
pub const CTX: [&str; 3] = ["--ctx", "--ctx-size", "-c"];

/// `v`, the value of `flag`, as a number; a value that is not one is refused
/// naming both, as `--ctx-size "12k": invalid digit found in string`.
pub fn number<T>(flag: &str, v: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    v.parse().map_err(|e| format!("{flag} {v:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::{CTX, number};

    #[test]
    fn the_context_flag_takes_llama_servers_spellings() {
        for f in ["--ctx", "--ctx-size", "-c"] {
            assert!(CTX.contains(&f), "{f}");
        }
        assert!(!CTX.contains(&"-C"));
    }

    #[test]
    fn a_number_that_is_none_names_its_flag_and_value() {
        assert_eq!(number::<usize>("-c", "32768"), Ok(32768));
        assert_eq!(
            number::<usize>("--ctx-size", "12k"),
            Err("--ctx-size \"12k\": invalid digit found in string".to_owned())
        );
        assert_eq!(
            number::<u16>("--port", "70000"),
            Err("--port \"70000\": number too large to fit in target type".to_owned())
        );
        assert_eq!(
            number::<u64>("--cache-ram", ""),
            Err("--cache-ram \"\": cannot parse integer from empty string".to_owned())
        );
    }
}
