//! The ranks of a tensor-parallel group, one process a node: each rank's
//! control address (the TCP address its bootstrap reaches) and its data
//! addresses (the fabric addresses a peer's RoCE queue pair may reach it at).
//! The two differ on a ring: a relayed fabric address carries marked RDMA
//! packets and nothing else, so a control channel never assumes it.

use std::net::{Ipv4Addr, SocketAddr};

use crate::GpuError;

/// One rank's addresses: where its control channel listens, and every
/// fabric address it holds, in the order a peer tries them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankAddr {
    pub control: SocketAddr,
    pub data: Vec<Ipv4Addr>,
}

/// The ranks of one group in rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankList {
    ranks: Vec<RankAddr>,
}

impl RankList {
    /// The list a rank file holds: one line a rank, in rank order,
    /// `<control ip:port> <data ip>[,<data ip>...]`; blank lines and text
    /// after `#` are ignored. No rank, a line of another shape, a control
    /// address given twice or a data address held by two ranks is refused by
    /// line.
    pub fn parse(text: &str) -> Result<RankList, GpuError> {
        const WHAT: &str = "the rank list";
        let mut ranks: Vec<RankAddr> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let bad = |why: &str| GpuError::shape(WHAT, format!("line {}: {why}", n + 1));
            let mut words = line.split_whitespace();
            let (Some(control), Some(data), None) = (words.next(), words.next(), words.next())
            else {
                return Err(bad("want <control ip:port> <data ip>[,<data ip>...]"));
            };
            let control: SocketAddr = control
                .parse()
                .map_err(|_| bad(&format!("{control:?} is not an ip:port")))?;
            let data = data
                .split(',')
                .map(|a| {
                    a.parse()
                        .map_err(|_| bad(&format!("{a:?} is not an IPv4 address")))
                })
                .collect::<Result<Vec<Ipv4Addr>, _>>()?;
            if ranks.iter().any(|r| r.control == control) {
                return Err(bad(&format!("control address {control} is given twice")));
            }
            if let Some(a) = data
                .iter()
                .enumerate()
                .find(|&(i, a)| data[..i].contains(a) || ranks.iter().any(|r| r.data.contains(a)))
            {
                return Err(bad(&format!("data address {} is given twice", a.1)));
            }
            ranks.push(RankAddr { control, data });
        }
        if ranks.is_empty() {
            return Err(GpuError::shape(WHAT, "no rank"));
        }
        Ok(RankList { ranks })
    }

    /// The list in the rank file at `path` ([`RankList::parse`]).
    pub fn read(path: &std::path::Path) -> Result<RankList, GpuError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| GpuError::shape("the rank list", format!("{}: {e}", path.display())))?;
        RankList::parse(&text)
    }

    /// The ranks in the group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    /// Whether the list holds no rank; a parsed list never does.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    /// Rank `rank`'s addresses; a rank past the list panics.
    #[must_use]
    pub fn rank(&self, rank: usize) -> &RankAddr {
        &self.ranks[rank]
    }
}

#[cfg(test)]
mod tests {
    use super::RankList;

    #[test]
    fn a_rank_file_reads_in_rank_order_and_refuses_by_line() {
        let list = RankList::parse(
            "# rank 0 first\n192.168.1.10:7000 198.18.0.1,198.18.1.1\n\n\
             192.168.1.11:7000 198.18.0.2 # rank 1\n",
        )
        .expect("a rank list");
        assert_eq!(list.len(), 2);
        assert_eq!(list.rank(0).data.len(), 2);
        assert_eq!(list.rank(1).control.to_string(), "192.168.1.11:7000");
        for (text, says) in [
            ("", "no rank"),
            ("192.168.1.10:7000\n", "line 1: want"),
            ("192.168.1.10 198.18.0.1\n", "is not an ip:port"),
            ("192.168.1.10:7000 fe80::1\n", "is not an IPv4"),
            (
                "10.0.0.1:1 198.18.0.1\n10.0.0.1:1 198.18.0.2\n",
                "line 2: control",
            ),
            (
                "10.0.0.1:1 198.18.0.1\n10.0.0.2:1 198.18.0.1\n",
                "line 2: data",
            ),
            ("10.0.0.1:1 198.18.0.1,198.18.0.1\n", "line 1: data"),
        ] {
            let e = RankList::parse(text).unwrap_err().to_string();
            assert!(e.contains(says), "{text:?}: {e}");
        }
    }
}
