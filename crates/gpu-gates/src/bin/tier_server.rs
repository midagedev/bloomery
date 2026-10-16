//! A tier node (`docs/contrib/dgx-spark.md` 9.5 items 5 and 6): the expert
//! tier of one load, served to node 0 over the control channel
//! (`bloomery_gpu::host::node`). Node 0 names this server's address in
//! `BLOOMERY_TIER_NODES`; each connection gets this process's census, and
//! the first load opened on one is served until node 0 closes it.
//!
//! ```text
//! tier_server --listen <ip:port> [-m <first shard>]
//! ```
//!
//! The model file is `-m`'s, else `$BLOOMERY_REF_MODEL`: this node's own
//! copy, which must hold the rows node 0's plan gives the tier with node 0's
//! bytes (each row's sha256). The explicit listen address is loopback or
//! private IPv4. Off-host nodes bind their own RDMA pages over this control
//! channel, which has no authentication and belongs on a trusted network.
//! A port of 0 takes a free
//! one. The server prints `listening on <ip:port>` once bound, serves one
//! load, and exits 0 once node 0 closes that load, whether node 0's side of
//! it came up or failed; a connection that opened no tier (a census, a
//! refused open) leaves it serving the next. A lost node 0 ends it with the
//! named loss.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("tier_server", server::run())
}

#[cfg(feature = "gpu")]
mod server {
    use std::io::Write;
    use std::net::{SocketAddr, TcpListener};
    use std::path::PathBuf;

    use bloomery_gpu::arch::qwen3moe::NODE_FAMILY;
    use bloomery_gpu::host::node;
    use bloomery_gpu_gates::{GateError, model_file, ref_model_path};
    use gguf::Split;

    const USAGE: &str = "usage: tier_server --listen <ip:port> [-m <first shard>]";

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let mut listen: Option<SocketAddr> = None;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let value = |args: &mut dyn Iterator<Item = String>| {
                args.next()
                    .ok_or_else(|| GateError::from(format!("{a} takes a value: {USAGE}")))
            };
            match a.as_str() {
                "--listen" => {
                    let v = value(&mut args)?;
                    listen = Some(v.parse().map_err(|e| format!("--listen {v:?}: {e}"))?);
                }
                "-m" => model_file::set_model_file(PathBuf::from(value(&mut args)?))?,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        let listen = listen.ok_or(USAGE)?;
        if !bloomery_levers::tier_ip(listen.ip()) {
            return Err(format!(
                "--listen {listen}: a tier server needs a loopback or private IPv4 address; \
                 the control channel has no authentication"
            )
            .into());
        }
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let listener = TcpListener::bind(listen).map_err(|e| format!("bind {listen}: {e}"))?;
        let at = listener.local_addr()?;
        println!("listening on {at}");
        std::io::stdout().flush()?;
        node::serve(&listener, &file, &[NODE_FAMILY])?;
        println!("node 0 closed the load");
        Ok(())
    }
}
