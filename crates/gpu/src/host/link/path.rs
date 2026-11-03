//! The path a rank's RDMA end takes to each other rank of a group
//! ([`PeerPath`]): the peer's control address, the fabric address this node
//! reaches it at, the local address the kernel sends from toward it, the RoCE
//! device and GID index of that local address, and the flow label the
//! fabric's relays need on its packets.
//!
//! A peer address is a RoCE path when this node reaches it over one link:
//! its route in the main table has no gateway, as a cabled neighbor's subnet
//! and a relay table's host route with a fixed neighbor entry do, so the
//! packet leaves with the peer's own MAC. A route through a gateway is a RoCE
//! path only on a fabric whose relays forward packets that carry a flow label
//! the fabric's manifest names (`flow_label_base`); such a peer's queue pair
//! sends with that label. Every other end sends with label 0. A peer none of
//! whose addresses is a path is refused by name: a link whose packets would
//! be dropped on the wire never opens.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::path::Path;

use crate::GpuError;
use crate::host::rank::RankList;

const WHAT: &str = "the peer path";

/// How this rank reaches rank `peer`: its control address, the fabric
/// address `remote` from the local address `local` on RoCE device `dev` at
/// GID index `gid`, every packet carrying `flow_label`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerPath {
    pub peer: usize,
    pub control: SocketAddr,
    pub local: IpAddr,
    pub remote: IpAddr,
    pub dev: String,
    pub gid: u8,
    pub flow_label: u32,
}

impl PeerPath {
    /// A path from `local` over RoCE device `dev` at GID index `gid` that
    /// needs no relay mark, to an end that is not a rank of a group (the
    /// other end of a one-machine link, a tier node): it names no peer
    /// address, the queue pair's endpoint exchange carries it.
    pub(crate) fn port(dev: String, gid: u8, local: IpAddr) -> PeerPath {
        let none = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        PeerPath {
            peer: 0,
            control: none,
            local,
            remote: none.ip(),
            dev,
            gid,
            flow_label: 0,
        }
    }
}

/// The path to every other rank of `ranks` from rank `me`, in rank order,
/// read from this node's main routing table and RoCE GID tables, and the
/// relay flow label from the fabric manifest at `fabric` when one is given.
/// A rank past the list, an unreadable manifest or table, or a peer with no
/// RoCE path is refused by name. Load-time only.
pub fn paths(
    ranks: &RankList,
    me: usize,
    fabric: Option<&Path>,
) -> Result<Vec<PeerPath>, GpuError> {
    if me >= ranks.len() {
        return Err(GpuError::shape(
            WHAT,
            format!("rank {me} of {}", ranks.len()),
        ));
    }
    let mark = fabric.map(relay_mark).transpose()?.flatten();
    let table = std::fs::read_to_string("/proc/net/route")
        .map_err(|e| GpuError::link(WHAT, format!("/proc/net/route: {e}")))?;
    let routes = Route::table(&table)?;
    (0..ranks.len())
        .filter(|&p| p != me)
        .map(|peer| {
            let addr = ranks.rank(peer);
            let mut tried = Vec::new();
            for &remote in &addr.data {
                let Some(flow_label) = lane(&routes, remote, mark) else {
                    tried.push(format!("{remote}: no route without a gateway"));
                    continue;
                };
                let local = match source(remote) {
                    Ok(local) => local,
                    Err(e) => {
                        tried.push(format!("{remote}: {e}"));
                        continue;
                    }
                };
                match bloomery_verbs::roce_v2_at(local) {
                    Ok((dev, gid)) => {
                        return Ok(PeerPath {
                            peer,
                            control: addr.control,
                            local: IpAddr::V4(local),
                            remote: IpAddr::V4(remote),
                            dev,
                            gid,
                            flow_label,
                        });
                    }
                    Err(e) => tried.push(format!("{remote} from {local}: {e}")),
                }
            }
            Err(GpuError::link(
                WHAT,
                format!("rank {peer}: no RoCE path ({})", tried.join("; ")),
            ))
        })
        .collect()
}

/// The flow label the manifest at `path` names for relayed packets, or
/// `None` when its relays need none (they tag by destination).
fn relay_mark(path: &Path) -> Result<Option<u32>, GpuError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| GpuError::shape(WHAT, format!("{}: {e}", path.display())))?;
    mark_in(&text).map_err(|why| GpuError::shape(WHAT, format!("{}: {why}", path.display())))
}

/// The value of the manifest's `"flow_label_base"` member in the JSON text
/// `text`, the one member a peer path reads, or `None` without one. A
/// member given twice or a value that is not a 20-bit whole number is
/// refused.
fn mark_in(text: &str) -> Result<Option<u32>, String> {
    const KEY: &str = "\"flow_label_base\"";
    let mut found = text.match_indices(KEY);
    let Some((at, _)) = found.next() else {
        return Ok(None);
    };
    if found.next().is_some() {
        return Err(format!("{KEY} is given twice"));
    }
    let value = text[at + KEY.len()..]
        .trim_start()
        .strip_prefix(':')
        .map(str::trim_start)
        .ok_or_else(|| format!("{KEY} is not a member"))?;
    let digits = &value[..value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len())];
    digits
        .parse::<u32>()
        .ok()
        .filter(|&l| l < 1 << 20)
        .map(Some)
        .ok_or_else(|| format!("{KEY} {digits:?}: want a 20-bit flow label"))
}

/// The local address the kernel sends from toward `remote`: a UDP socket's
/// connect picks it from the route and sends nothing.
fn source(remote: Ipv4Addr) -> Result<Ipv4Addr, String> {
    let s = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(|e| e.to_string())?;
    s.connect((remote, 9)).map_err(|e| e.to_string())?;
    match s.local_addr().map_err(|e| e.to_string())?.ip() {
        IpAddr::V4(a) => Ok(a),
        IpAddr::V6(a) => Err(format!("an IPv6 source {a}")),
    }
}

/// The flow label of the lane to `remote` when its best route is a RoCE path
/// (see the module comment), or `None`.
fn lane(routes: &[Route], remote: Ipv4Addr, mark: Option<u32>) -> Option<u32> {
    let a = u32::from(remote);
    let best = routes
        .iter()
        .filter(|r| a & r.mask == r.dest)
        .max_by_key(|r| (r.mask.count_ones(), std::cmp::Reverse(r.metric)))?;
    if best.gateway { mark } else { Some(0) }
}

/// One route of the main table, as `/proc/net/route` lists it.
#[derive(Clone, Copy, Debug)]
struct Route {
    dest: u32,
    mask: u32,
    gateway: bool,
    metric: u32,
}

impl Route {
    /// The routes that are up in `text`, the contents of `/proc/net/route`.
    fn table(text: &str) -> Result<Vec<Route>, GpuError> {
        const UP: u32 = 0x1;
        const GATEWAY: u32 = 0x2;
        let mut routes = Vec::new();
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            let hex = |i: usize| {
                f.get(i)
                    .and_then(|v| u32::from_str_radix(v, 16).ok())
                    .ok_or_else(|| GpuError::link(WHAT, format!("/proc/net/route line {line:?}")))
            };
            // Addresses are printed as the native word of their network-order bytes.
            let addr = |i: usize| hex(i).map(|v| u32::from(Ipv4Addr::from(v.to_ne_bytes())));
            let flags = hex(3)?;
            if flags & UP == 0 {
                continue;
            }
            let metric = f
                .get(6)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| GpuError::link(WHAT, format!("/proc/net/route line {line:?}")))?;
            routes.push(Route {
                dest: addr(1)?,
                mask: addr(7)?,
                gateway: flags & GATEWAY != 0,
                metric,
            });
        }
        Ok(routes)
    }
}

#[cfg(test)]
mod tests {
    use super::{Route, lane, mark_in};
    use std::net::Ipv4Addr;

    /// A `/proc/net/route` line for `dest/len` via `gw` (0.0.0.0 for none).
    fn line(dest: [u8; 4], len: u32, gw: [u8; 4], metric: u32) -> String {
        let word = |b: [u8; 4]| format!("{:08X}", u32::from_ne_bytes(b));
        let mask = u32::MAX.checked_shl(32 - len).unwrap_or(0).to_be_bytes();
        let flags = 1 | if gw == [0; 4] { 0 } else { 2 } | if len == 32 { 4 } else { 0 };
        format!(
            "dev0\t{}\t{}\t{flags:04X}\t0\t0\t{metric}\t{}\t0\t0\t0",
            word(dest),
            word(gw),
            word(mask)
        )
    }

    // A ring node's table: two cabled subnets, a relay table's host routes with fixed neighbor
    // entries, and the subnets behind the neighbors through a gateway. Only a route without a
    // gateway is a RoCE path, unless the fabric names the flow label its relays forward.
    #[test]
    fn a_peer_address_is_a_path_over_one_link_or_a_marked_relay() {
        let text = [
            "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT"
                .to_string(),
            line([0; 4], 0, [192, 168, 1, 1], 100),
            line([198, 18, 0, 0], 24, [198, 18, 6, 2], 0),
            line([198, 18, 0, 2], 32, [0; 4], 0),
            line([198, 18, 4, 0], 24, [0; 4], 103),
            line([198, 18, 6, 0], 24, [0; 4], 105),
        ]
        .join("\n");
        let routes = Route::table(&text).expect("a route table");
        let at = |a: [u8; 4], mark| lane(&routes, Ipv4Addr::from(a), mark);
        assert_eq!(at([198, 18, 0, 2], None), Some(0), "relay host route");
        assert_eq!(at([198, 18, 4, 1], None), Some(0), "cabled neighbor");
        assert_eq!(
            at([198, 18, 0, 1], None),
            None,
            "through a gateway, no mark"
        );
        assert_eq!(
            at([198, 18, 0, 1], Some(16383)),
            Some(16383),
            "marked relay"
        );
        assert_eq!(
            at([198, 18, 0, 2], Some(16383)),
            Some(0),
            "the host route wins"
        );
        assert_eq!(
            at([10, 0, 0, 1], None),
            None,
            "the default route has a gateway"
        );
    }

    // The deployed fabric document tags relayed packets by destination and names no label; a
    // manifest whose relays forward by flow label names it once.
    #[test]
    fn a_manifest_names_its_relay_label_once_or_not_at_all() {
        assert_eq!(
            mark_in(r#"{"schema": "sparkring-fabric/v1", "size": 4}"#),
            Ok(None)
        );
        assert_eq!(
            mark_in(r#"{"flow_label_base": 16383, "x": 1}"#),
            Ok(Some(16383))
        );
        for bad in [
            r#"{"flow_label_base": 1048576}"#,
            r#"{"flow_label_base": "16383"}"#,
            r#"{"flow_label_base": 1, "flow_label_base": 2}"#,
        ] {
            assert!(mark_in(bad).is_err(), "{bad}");
        }
    }
}
