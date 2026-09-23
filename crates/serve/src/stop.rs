//! Stop strings over streamed text. llama-server's rule: the stop word is not part
//! of the content, and text that could still grow into a stop word is held back
//! from the stream until it either completes one or stops matching.

/// Accumulates generated text and decides what may be sent.
pub(crate) struct StopScan {
    stops: Vec<String>,
    text: String,
    sent: usize,
}

/// What one push produced.
pub(crate) struct Pushed {
    /// Text now safe to send (may be empty).
    pub send: String,
    /// The stop word that matched, if the generation must end here.
    pub stopped: Option<String>,
}

impl StopScan {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        StopScan {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            text: String::new(),
            sent: 0,
        }
    }

    /// Appends `piece`. On a match the text is cut before the stop word.
    pub(crate) fn push(&mut self, piece: &str) -> Pushed {
        let from = self.sent;
        self.text.push_str(piece);
        let longest = self.stops.iter().map(String::len).max().unwrap_or(0);
        // A new match must end inside the new text, so it starts after this point.
        let mut search = from.saturating_sub(longest);
        while !self.text.is_char_boundary(search) {
            search -= 1;
        }
        let hit = self
            .stops
            .iter()
            .filter_map(|s| {
                self.text[search..]
                    .find(s.as_str())
                    .map(|at| (search + at, s))
            })
            .min_by_key(|(at, _)| *at);
        if let Some((at, word)) = hit {
            let word = word.clone();
            self.text.truncate(at);
            let send = self.text.get(self.sent..).unwrap_or("").to_owned();
            self.sent = self.text.len();
            return Pushed {
                send,
                stopped: Some(word),
            };
        }
        let hold = self.partial_suffix();
        let upto = self.text.len() - hold;
        let send = self.text[self.sent.min(upto)..upto].to_owned();
        self.sent = self.sent.max(upto);
        Pushed {
            send,
            stopped: None,
        }
    }

    /// Releases the held-back tail at the end of a generation.
    pub(crate) fn finish(&mut self) -> String {
        let send = self.text.get(self.sent..).unwrap_or("").to_owned();
        self.sent = self.text.len();
        send
    }

    /// The whole content so far (stop word excluded).
    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    /// Length of the longest suffix of the text that is a proper prefix of a stop word.
    fn partial_suffix(&self) -> usize {
        let t = self.text.as_bytes();
        let mut best = 0;
        for s in &self.stops {
            let s = s.as_bytes();
            for k in (best + 1..s.len()).rev() {
                if k <= t.len()
                    && t[t.len() - k..] == s[..k]
                    && self.text.is_char_boundary(t.len() - k)
                {
                    best = k;
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_partial_and_cuts_match() {
        let mut s = StopScan::new(vec!["END".into()]);
        assert_eq!(s.push("abc E").send, "abc ");
        assert_eq!(s.push("N").send, "");
        let p = s.push("Dxyz");
        assert_eq!(p.send, "");
        assert_eq!(p.stopped.as_deref(), Some("END"));
        assert_eq!(s.text(), "abc ");
    }

    #[test]
    fn releases_a_false_start() {
        let mut s = StopScan::new(vec!["END".into()]);
        assert_eq!(s.push("E").send, "");
        assert_eq!(s.push("x").send, "Ex");
        assert_eq!(s.push("EN").send, "");
        assert_eq!(s.finish(), "EN");
        assert_eq!(s.text(), "ExEN");
    }
}
