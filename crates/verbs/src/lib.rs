//! bloomery-verbs: the libibverbs calls of the RDMA link, loaded from the
//! system's `libibverbs.so.1` at run time: no crate and no link-time
//! library, so a build needs no RDMA headers, and a host without rdma-core
//! is refused by name when a link opens.
//!
//! The structs mirror rdma-core's `infiniband/verbs.h` on a 64-bit Linux
//! target, field for field where this crate builds them and as a prefix
//! where it only reads one. `const` asserts pin each struct's size and the
//! offset of every field set or read, as rdma-core 50.0's header lays them
//! out on aarch64 Linux. `ibv_post_send` and `ibv_poll_cq` are inline in
//! that header and call through the context's operation table, so this crate
//! takes those two entries from the table as the header does.
//!
//! The handles are owned: the library is loaded once for the process, and a
//! registered region or a queue pair holds its device, so a pump thread can
//! own them.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, OnceLock};

/// Why a verbs call failed, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerbsError(String);

impl fmt::Display for VerbsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VerbsError {}

impl From<String> for VerbsError {
    fn from(s: String) -> VerbsError {
        VerbsError(s)
    }
}

impl From<&str> for VerbsError {
    fn from(s: &str) -> VerbsError {
        VerbsError(s.to_string())
    }
}

/// `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE`; relaxed ordering
/// (`IBV_ACCESS_RELAXED_ORDERING`) is not requested.
const ACCESS_STRICT: c_int = 1 | 2;
const QPT_RC: u32 = 2;
const QPS_INIT: u32 = 1;
const QPS_RTR: u32 = 2;
const QPS_RTS: u32 = 3;
const PORT_ACTIVE: u32 = 4;
const QP_STATE: c_int = 1;
const QP_ACCESS_FLAGS: c_int = 8;
const QP_PKEY_INDEX: c_int = 16;
const QP_PORT: c_int = 32;
const QP_AV: c_int = 128;
const QP_PATH_MTU: c_int = 256;
const QP_TIMEOUT: c_int = 512;
const QP_RETRY_CNT: c_int = 1024;
const QP_RNR_RETRY: c_int = 2048;
const QP_RQ_PSN: c_int = 4096;
const QP_MAX_QP_RD_ATOMIC: c_int = 8192;
const QP_MIN_RNR_TIMER: c_int = 32768;
const QP_SQ_PSN: c_int = 65536;
const QP_MAX_DEST_RD_ATOMIC: c_int = 131_072;
const QP_DEST_QPN: c_int = 1_048_576;
const WR_RDMA_WRITE: u32 = 0;
const SEND_SIGNALED: u32 = 2;
/// `IBV_WC_RDMA_WRITE`: a send completion of an RDMA write.
pub const WC_RDMA_WRITE: u32 = 1;
/// `IBV_GID_TYPE_ROCE_V2`.
const GID_TYPE_ROCE_V2: u32 = 2;
/// The one port of a RoCE device.
const PORT: u8 = 1;
/// Completions the device's one queue holds.
const CQ_DEPTH: c_int = 1024;
/// Writes one post chains at most.
pub const MAX_CHAIN: usize = 8;

/// The rdma-core ABI mirrors.
mod abi {
    use std::ffi::{c_int, c_void};
    use std::mem::{offset_of, size_of};

    /// `union ibv_gid`.
    #[repr(C, align(8))]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Gid {
        pub raw: [u8; 16],
    }

    #[repr(C)]
    pub(super) struct GlobalRoute {
        pub(super) dgid: Gid,
        pub(super) flow_label: u32,
        pub(super) sgid_index: u8,
        pub(super) hop_limit: u8,
        pub(super) traffic_class: u8,
    }

    #[repr(C)]
    pub(super) struct AhAttr {
        pub(super) grh: GlobalRoute,
        pub(super) dlid: u16,
        pub(super) sl: u8,
        pub(super) src_path_bits: u8,
        pub(super) static_rate: u8,
        pub(super) is_global: u8,
        pub(super) port_num: u8,
    }

    #[repr(C)]
    pub(super) struct QpCap {
        pub(super) max_send_wr: u32,
        pub(super) max_recv_wr: u32,
        pub(super) max_send_sge: u32,
        pub(super) max_recv_sge: u32,
        pub(super) max_inline_data: u32,
    }

    #[repr(C)]
    pub(super) struct QpAttr {
        pub(super) qp_state: u32,
        pub(super) cur_qp_state: u32,
        pub(super) path_mtu: u32,
        pub(super) path_mig_state: u32,
        pub(super) qkey: u32,
        pub(super) rq_psn: u32,
        pub(super) sq_psn: u32,
        pub(super) dest_qp_num: u32,
        pub(super) qp_access_flags: u32,
        pub(super) cap: QpCap,
        pub(super) ah_attr: AhAttr,
        pub(super) alt_ah_attr: AhAttr,
        pub(super) pkey_index: u16,
        pub(super) alt_pkey_index: u16,
        pub(super) en_sqd_async_notify: u8,
        pub(super) sq_draining: u8,
        pub(super) max_rd_atomic: u8,
        pub(super) max_dest_rd_atomic: u8,
        pub(super) min_rnr_timer: u8,
        pub(super) port_num: u8,
        pub(super) timeout: u8,
        pub(super) retry_cnt: u8,
        pub(super) rnr_retry: u8,
        pub(super) alt_port_num: u8,
        pub(super) alt_timeout: u8,
        pub(super) rate_limit: u32,
    }

    #[repr(C)]
    pub(super) struct QpInitAttr {
        pub(super) qp_context: *mut c_void,
        pub(super) send_cq: *mut Cq,
        pub(super) recv_cq: *mut Cq,
        pub(super) srq: *mut c_void,
        pub(super) cap: QpCap,
        pub(super) qp_type: u32,
        pub(super) sq_sig_all: c_int,
    }

    /// `struct ibv_port_attr`: the four words read, then the rest.
    #[repr(C)]
    pub(super) struct PortAttr {
        pub(super) state: u32,
        pub(super) max_mtu: u32,
        pub(super) active_mtu: u32,
        pub(super) gid_tbl_len: c_int,
        pub(super) rest: [u32; 10],
    }

    /// `struct ibv_gid_entry`.
    #[repr(C)]
    pub(super) struct GidEntry {
        pub(super) gid: Gid,
        pub(super) gid_index: u32,
        pub(super) port_num: u32,
        pub(super) gid_type: u32,
        pub(super) ndev_ifindex: u32,
    }

    #[repr(C)]
    pub(super) struct Sge {
        pub(super) addr: u64,
        pub(super) length: u32,
        pub(super) lkey: u32,
    }

    /// `struct ibv_send_wr` with its `wr` union as the `rdma` member; the other
    /// members' bytes and the trailing unions are fields the probe zeroes and
    /// its opcodes do not read.
    #[repr(C)]
    pub(super) struct SendWr {
        pub(super) wr_id: u64,
        pub(super) next: *mut SendWr,
        pub(super) sg_list: *mut Sge,
        pub(super) num_sge: c_int,
        pub(super) opcode: u32,
        pub(super) send_flags: u32,
        pub(super) imm_data: u32,
        pub(super) remote_addr: u64,
        pub(super) rkey: u32,
        pub(super) wr_rest: [u32; 5],
        pub(super) qp_type: u32,
        pub(super) tail: [u64; 6],
    }

    /// `struct ibv_wc`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Wc {
        pub wr_id: u64,
        pub status: u32,
        pub opcode: u32,
        pub vendor_err: u32,
        pub byte_len: u32,
        pub imm_data: u32,
        pub qp_num: u32,
        pub src_qp: u32,
        pub wc_flags: u32,
        pub pkey_index: u16,
        pub slid: u16,
        pub sl: u8,
        pub dlid_path_bits: u8,
    }

    /// `struct ibv_mr`.
    #[repr(C)]
    pub(super) struct Mr {
        pub(super) context: *mut Context,
        pub(super) pd: *mut c_void,
        pub(super) addr: *mut c_void,
        pub(super) length: usize,
        pub(super) handle: u32,
        pub(super) lkey: u32,
        pub(super) rkey: u32,
    }

    /// The prefix of `struct ibv_qp` up to `qp_num`.
    #[repr(C)]
    pub(super) struct RawQp {
        pub(super) context: *mut Context,
        pub(super) qp_context: *mut c_void,
        pub(super) pd: *mut c_void,
        pub(super) send_cq: *mut Cq,
        pub(super) recv_cq: *mut Cq,
        pub(super) srq: *mut c_void,
        pub(super) handle: u32,
        pub(super) qp_num: u32,
    }

    /// The prefix of `struct ibv_cq`.
    #[repr(C)]
    pub(super) struct Cq {
        pub(super) context: *mut Context,
    }

    pub(super) type PollCqFn = unsafe extern "C" fn(*mut Cq, c_int, *mut Wc) -> c_int;
    pub(super) type PostSendFn =
        unsafe extern "C" fn(*mut RawQp, *mut SendWr, *mut *mut SendWr) -> c_int;

    /// The prefix of `struct ibv_context`: the device, then the operation table
    /// up to `post_send`, the slots before each entry kept as words.
    #[repr(C)]
    pub(super) struct Context {
        pub(super) device: *mut c_void,
        pub(super) ops_before_poll_cq: [usize; 11],
        pub(super) poll_cq: Option<PollCqFn>,
        pub(super) ops_before_post_send: [usize; 13],
        pub(super) post_send: Option<PostSendFn>,
    }

    const _: () = {
        assert!(size_of::<Gid>() == 16);
        assert!(size_of::<GlobalRoute>() == 24);
        assert!(offset_of!(GlobalRoute, dgid) == 0 && offset_of!(GlobalRoute, flow_label) == 16);
        assert!(offset_of!(GlobalRoute, sgid_index) == 20);
        assert!(offset_of!(GlobalRoute, hop_limit) == 21);
        assert!(offset_of!(GlobalRoute, traffic_class) == 22);
        assert!(size_of::<AhAttr>() == 32 && offset_of!(AhAttr, grh) == 0);
        assert!(offset_of!(AhAttr, dlid) == 24 && offset_of!(AhAttr, sl) == 26);
        assert!(offset_of!(AhAttr, src_path_bits) == 27 && offset_of!(AhAttr, static_rate) == 28);
        assert!(offset_of!(AhAttr, is_global) == 29 && offset_of!(AhAttr, port_num) == 30);
        assert!(size_of::<QpCap>() == 20);
        assert!(offset_of!(QpCap, max_send_wr) == 0 && offset_of!(QpCap, max_recv_wr) == 4);
        assert!(offset_of!(QpCap, max_send_sge) == 8 && offset_of!(QpCap, max_recv_sge) == 12);
        assert!(offset_of!(QpCap, max_inline_data) == 16);
        assert!(size_of::<QpAttr>() == 144);
        assert!(offset_of!(QpAttr, qp_state) == 0 && offset_of!(QpAttr, path_mtu) == 8);
        assert!(offset_of!(QpAttr, qkey) == 16 && offset_of!(QpAttr, rq_psn) == 20);
        assert!(offset_of!(QpAttr, sq_psn) == 24 && offset_of!(QpAttr, dest_qp_num) == 28);
        assert!(offset_of!(QpAttr, qp_access_flags) == 32 && offset_of!(QpAttr, cap) == 36);
        assert!(offset_of!(QpAttr, ah_attr) == 56 && offset_of!(QpAttr, alt_ah_attr) == 88);
        assert!(offset_of!(QpAttr, pkey_index) == 120);
        assert!(offset_of!(QpAttr, max_rd_atomic) == 126);
        assert!(offset_of!(QpAttr, max_dest_rd_atomic) == 127);
        assert!(offset_of!(QpAttr, min_rnr_timer) == 128 && offset_of!(QpAttr, port_num) == 129);
        assert!(offset_of!(QpAttr, timeout) == 130 && offset_of!(QpAttr, retry_cnt) == 131);
        assert!(offset_of!(QpAttr, rnr_retry) == 132 && offset_of!(QpAttr, rate_limit) == 136);
        assert!(size_of::<QpInitAttr>() == 64 && offset_of!(QpInitAttr, qp_context) == 0);
        assert!(offset_of!(QpInitAttr, send_cq) == 8 && offset_of!(QpInitAttr, recv_cq) == 16);
        assert!(offset_of!(QpInitAttr, srq) == 24 && offset_of!(QpInitAttr, cap) == 32);
        assert!(offset_of!(QpInitAttr, qp_type) == 52 && offset_of!(QpInitAttr, sq_sig_all) == 56);
        assert!(size_of::<PortAttr>() == 56 && offset_of!(PortAttr, state) == 0);
        assert!(offset_of!(PortAttr, max_mtu) == 4 && offset_of!(PortAttr, active_mtu) == 8);
        assert!(offset_of!(PortAttr, gid_tbl_len) == 12);
        assert!(size_of::<GidEntry>() == 32 && offset_of!(GidEntry, gid) == 0);
        assert!(offset_of!(GidEntry, gid_index) == 16 && offset_of!(GidEntry, port_num) == 20);
        assert!(offset_of!(GidEntry, gid_type) == 24 && offset_of!(GidEntry, ndev_ifindex) == 28);
        assert!(size_of::<Sge>() == 16 && offset_of!(Sge, addr) == 0);
        assert!(offset_of!(Sge, length) == 8 && offset_of!(Sge, lkey) == 12);
        assert!(size_of::<SendWr>() == 128);
        assert!(offset_of!(SendWr, wr_id) == 0 && offset_of!(SendWr, next) == 8);
        assert!(offset_of!(SendWr, sg_list) == 16 && offset_of!(SendWr, num_sge) == 24);
        assert!(offset_of!(SendWr, opcode) == 28 && offset_of!(SendWr, send_flags) == 32);
        assert!(offset_of!(SendWr, imm_data) == 36 && offset_of!(SendWr, remote_addr) == 40);
        assert!(offset_of!(SendWr, rkey) == 48);
        assert!(size_of::<Wc>() == 48);
        assert!(offset_of!(Wc, wr_id) == 0 && offset_of!(Wc, status) == 8);
        assert!(offset_of!(Wc, opcode) == 12 && offset_of!(Wc, vendor_err) == 16);
        assert!(offset_of!(Wc, byte_len) == 20 && offset_of!(Wc, imm_data) == 24);
        assert!(offset_of!(Wc, qp_num) == 28 && offset_of!(Wc, wc_flags) == 36);
        assert!(size_of::<Mr>() == 48 && offset_of!(Mr, addr) == 16);
        assert!(offset_of!(Mr, length) == 24 && offset_of!(Mr, lkey) == 36);
        assert!(offset_of!(Mr, rkey) == 40);
        assert!(offset_of!(RawQp, context) == 0 && offset_of!(RawQp, qp_num) == 52);
        assert!(offset_of!(Cq, context) == 0);
        assert!(offset_of!(Context, device) == 0 && offset_of!(Context, poll_cq) == 96);
        assert!(offset_of!(Context, post_send) == 208);
    };
}

use abi::{
    AhAttr, Context, Cq, GidEntry, GlobalRoute, Mr, PollCqFn, PortAttr, PostSendFn, QpAttr, QpCap,
    QpInitAttr, RawQp, SendWr, Sge,
};
pub use abi::{Gid, Wc};

/// The exported entries this crate calls, resolved once.
struct Fns {
    get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void,
    free_device_list: unsafe extern "C" fn(*mut *mut c_void),
    get_device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    open_device: unsafe extern "C" fn(*mut c_void) -> *mut Context,
    close_device: unsafe extern "C" fn(*mut Context) -> c_int,
    query_port: unsafe extern "C" fn(*mut Context, u8, *mut PortAttr) -> c_int,
    query_gid_ex: unsafe extern "C" fn(*mut Context, u32, u32, *mut GidEntry, u32, usize) -> c_int,
    alloc_pd: unsafe extern "C" fn(*mut Context) -> *mut c_void,
    dealloc_pd: unsafe extern "C" fn(*mut c_void) -> c_int,
    reg_mr: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut Mr,
    dereg_mr: unsafe extern "C" fn(*mut Mr) -> c_int,
    create_cq:
        unsafe extern "C" fn(*mut Context, c_int, *mut c_void, *mut c_void, c_int) -> *mut Cq,
    destroy_cq: unsafe extern "C" fn(*mut Cq) -> c_int,
    create_qp: unsafe extern "C" fn(*mut c_void, *mut QpInitAttr) -> *mut RawQp,
    destroy_qp: unsafe extern "C" fn(*mut RawQp) -> c_int,
    modify_qp: unsafe extern "C" fn(*mut RawQp, *mut QpAttr, c_int) -> c_int,
    wc_status_str: unsafe extern "C" fn(u32) -> *const c_char,
}

/// The library, loaded once for the process and never unloaded.
pub struct Verbs {
    f: Fns,
}

/// A raw rdma-core handle that moves between threads: rdma-core's verbs are
/// thread-safe, and each owner here frees its handle once.
struct Raw<T>(*mut T);

// SAFETY: the pointee is an rdma-core object whose calls the library makes
// thread-safe; the owner that holds the handle frees it once, in its Drop.
unsafe impl<T> Send for Raw<T> {}
// SAFETY: as for Send: concurrent calls on one handle are rdma-core's to order.
unsafe impl<T> Sync for Raw<T> {}

/// The library, once loaded; a failed load is tried again by the next call.
static LIB: OnceLock<Verbs> = OnceLock::new();

/// The exported entry `name` of `lib` as the function pointer type `F`, or
/// the refusal naming it.
///
/// # Safety
///
/// `lib` is a live handle from `dlopen`, and `F` is an `unsafe extern "C"
/// fn` type whose signature is `name`'s C prototype.
unsafe fn entry<F: Copy>(lib: *mut c_void, name: &CStr) -> Result<F, VerbsError> {
    const { assert!(size_of::<F>() == size_of::<*mut c_void>()) };
    // SAFETY: `lib` is a live handle from dlopen (the caller's contract) and
    // `name` a NUL-terminated string.
    let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
    if p.is_null() {
        return Err(format!("libibverbs.so.1 has no {name:?}").into());
    }
    // SAFETY: `p` is the exported symbol `name`, a function whose C
    // prototype `F` is (the caller's contract), and `F` is a pointer's size
    // (asserted above).
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

impl Verbs {
    /// The library: `libibverbs.so.1` loaded and every entry resolved, on
    /// the first call of the process.
    pub fn load() -> Result<&'static Verbs, VerbsError> {
        if let Some(v) = LIB.get() {
            return Ok(v);
        }
        let v = Verbs::open()?;
        Ok(LIB.get_or_init(|| v))
    }

    fn open() -> Result<Verbs, VerbsError> {
        // SAFETY: the name is a NUL-terminated string; the library's
        // initializers are rdma-core's own.
        let lib = unsafe { libc::dlopen(c"libibverbs.so.1".as_ptr(), libc::RTLD_NOW) };
        if lib.is_null() {
            return Err("cannot load libibverbs.so.1: is rdma-core installed?".into());
        }
        Ok(Verbs {
            f: Fns {
                // SAFETY: `lib` is open; the field's type is `ibv_get_device_list`'s prototype.
                get_device_list: unsafe { entry(lib, c"ibv_get_device_list")? },
                // SAFETY: `lib` is open; the field's type is `ibv_free_device_list`'s prototype.
                free_device_list: unsafe { entry(lib, c"ibv_free_device_list")? },
                // SAFETY: `lib` is open; the field's type is `ibv_get_device_name`'s prototype.
                get_device_name: unsafe { entry(lib, c"ibv_get_device_name")? },
                // SAFETY: `lib` is open; the field's type is `ibv_open_device`'s prototype.
                open_device: unsafe { entry(lib, c"ibv_open_device")? },
                // SAFETY: `lib` is open; the field's type is `ibv_close_device`'s prototype.
                close_device: unsafe { entry(lib, c"ibv_close_device")? },
                // SAFETY: `lib` is open; the field's type is `ibv_query_port`'s prototype.
                query_port: unsafe { entry(lib, c"ibv_query_port")? },
                // SAFETY: `lib` is open; the field's type is `_ibv_query_gid_ex`'s prototype.
                query_gid_ex: unsafe { entry(lib, c"_ibv_query_gid_ex")? },
                // SAFETY: `lib` is open; the field's type is `ibv_alloc_pd`'s prototype.
                alloc_pd: unsafe { entry(lib, c"ibv_alloc_pd")? },
                // SAFETY: `lib` is open; the field's type is `ibv_dealloc_pd`'s prototype.
                dealloc_pd: unsafe { entry(lib, c"ibv_dealloc_pd")? },
                // SAFETY: `lib` is open; the field's type is `ibv_reg_mr`'s prototype.
                reg_mr: unsafe { entry(lib, c"ibv_reg_mr")? },
                // SAFETY: `lib` is open; the field's type is `ibv_dereg_mr`'s prototype.
                dereg_mr: unsafe { entry(lib, c"ibv_dereg_mr")? },
                // SAFETY: `lib` is open; the field's type is `ibv_create_cq`'s prototype.
                create_cq: unsafe { entry(lib, c"ibv_create_cq")? },
                // SAFETY: `lib` is open; the field's type is `ibv_destroy_cq`'s prototype.
                destroy_cq: unsafe { entry(lib, c"ibv_destroy_cq")? },
                // SAFETY: `lib` is open; the field's type is `ibv_create_qp`'s prototype.
                create_qp: unsafe { entry(lib, c"ibv_create_qp")? },
                // SAFETY: `lib` is open; the field's type is `ibv_destroy_qp`'s prototype.
                destroy_qp: unsafe { entry(lib, c"ibv_destroy_qp")? },
                // SAFETY: `lib` is open; the field's type is `ibv_modify_qp`'s prototype.
                modify_qp: unsafe { entry(lib, c"ibv_modify_qp")? },
                // SAFETY: `lib` is open; the field's type is `ibv_wc_status_str`'s prototype.
                wc_status_str: unsafe { entry(lib, c"ibv_wc_status_str")? },
            },
        })
    }

    /// The name rdma-core gives a completion status.
    pub fn status_name(&self, status: u32) -> String {
        // SAFETY: the call takes any status and returns a static string.
        let raw = unsafe { (self.f.wc_status_str)(status) };
        if raw.is_null() {
            return format!("status {status}");
        }
        // SAFETY: a NUL-terminated string the library owns for its life.
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }
}

/// The first RoCE port the machine reports as active with a RoCE v2 GID
/// mapped from an IPv4 address: the device's name and that GID's index.
pub fn first_roce_v2() -> Result<(String, u8), VerbsError> {
    let v = Verbs::load()?;
    let mut seen = Vec::new();
    for name in device_names(v)? {
        let ctx = open_by_name(v, &name)?;
        let found = roce_v2_gid(v, ctx.0);
        // SAFETY: `ctx` was opened above and is closed once, here.
        let rc = unsafe { (v.f.close_device)(ctx.0) };
        teardown("ibv_close_device", rc);
        match found {
            Some(index) => return Ok((name, index)),
            None => seen.push(name),
        }
    }
    Err(format!(
        "no active RoCE port with an IPv4 RoCE v2 GID; devices: {}",
        seen.join(", ")
    )
    .into())
}

/// The first index of an IPv4 RoCE v2 GID on the active port of `ctx`.
fn roce_v2_gid(v: &Verbs, ctx: *mut Context) -> Option<u8> {
    let mut port = port_attr();
    // SAFETY: `ctx` is open and `port` a writable struct of the library's layout.
    if unsafe { (v.f.query_port)(ctx, PORT, &mut port) } != 0 || port.state != PORT_ACTIVE {
        return None;
    }
    let len = u8::try_from(port.gid_tbl_len.clamp(0, 255)).unwrap_or(u8::MAX);
    (0..len).find(|&i| {
        let mut e = GidEntry {
            gid: Gid::default(),
            gid_index: 0,
            port_num: 0,
            gid_type: 0,
            ndev_ifindex: 0,
        };
        // SAFETY: `ctx` is open and `e` a writable entry of the size passed.
        let rc = unsafe {
            (v.f.query_gid_ex)(
                ctx,
                u32::from(PORT),
                u32::from(i),
                &mut e,
                0,
                size_of::<GidEntry>(),
            )
        };
        let ipv4 = e.gid.raw[..10].iter().all(|&b| b == 0) && e.gid.raw[10..12] == [0xff, 0xff];
        rc == 0 && e.gid_type == GID_TYPE_ROCE_V2 && ipv4
    })
}

/// A zeroed `ibv_port_attr`.
fn port_attr() -> PortAttr {
    PortAttr {
        state: 0,
        max_mtu: 0,
        active_mtu: 0,
        gid_tbl_len: 0,
        rest: [0; 10],
    }
}

/// One RoCE device opened on its port: a protection domain and one
/// completion queue for every queue pair, both directions.
pub struct Nic {
    v: &'static Verbs,
    ctx: Raw<Context>,
    pd: Raw<c_void>,
    cq: Raw<Cq>,
    gid_index: u8,
    gid: Gid,
    mtu: u32,
    poll_cq: PollCqFn,
    post_send: PostSendFn,
}

impl Nic {
    /// Open device `name` (as `ibv_devices` lists it); its port must be
    /// active and GID `gid_index` set.
    pub fn open(name: &str, gid_index: u8) -> Result<Arc<Nic>, VerbsError> {
        let v = Verbs::load()?;
        let ctx = open_by_name(v, name)?;
        // SAFETY: `ctx` is the open context; its table holds the provider's
        // poll entry for the context's life.
        let poll_cq = unsafe { (*ctx.0).poll_cq };
        // SAFETY: `ctx` is the open context; its table holds the provider's
        // post entry for the context's life.
        let post_send = unsafe { (*ctx.0).post_send };
        let (Some(poll_cq), Some(post_send)) = (poll_cq, post_send) else {
            // SAFETY: `ctx` is open and closed once, here.
            let rc = unsafe { (v.f.close_device)(ctx.0) };
            teardown("ibv_close_device", rc);
            return Err(format!("{name}: the provider leaves a post or poll entry empty").into());
        };
        let mut nic = Nic {
            v,
            ctx,
            pd: Raw(ptr::null_mut()),
            cq: Raw(ptr::null_mut()),
            gid_index,
            gid: Gid::default(),
            mtu: 0,
            poll_cq,
            post_send,
        };
        let mut port = port_attr();
        // SAFETY: `ctx` is open and `port` a writable struct of the
        // library's layout.
        if unsafe { (v.f.query_port)(nic.ctx.0, PORT, &mut port) } != 0 || port.state != PORT_ACTIVE
        {
            return Err(format!("{name}: port {PORT} is not active (state {})", port.state).into());
        }
        nic.mtu = port.active_mtu;
        let mut e = GidEntry {
            gid: Gid::default(),
            gid_index: 0,
            port_num: 0,
            gid_type: 0,
            ndev_ifindex: 0,
        };
        // SAFETY: `ctx` is open and `e` a writable entry of the size passed.
        let rc = unsafe {
            (v.f.query_gid_ex)(
                nic.ctx.0,
                u32::from(PORT),
                u32::from(gid_index),
                &mut e,
                0,
                size_of::<GidEntry>(),
            )
        };
        if rc != 0 || e.gid == Gid::default() {
            return Err(format!("{name}: no GID at index {gid_index}").into());
        }
        nic.gid = e.gid;
        // SAFETY: `ctx` is open.
        nic.pd = Raw(unsafe { (v.f.alloc_pd)(nic.ctx.0) });
        if nic.pd.0.is_null() {
            return Err(format!("{name}: ibv_alloc_pd failed").into());
        }
        // SAFETY: `ctx` is open; no channel and no context pointer are asked.
        nic.cq = Raw(unsafe {
            (v.f.create_cq)(nic.ctx.0, CQ_DEPTH, ptr::null_mut(), ptr::null_mut(), 0)
        });
        if nic.cq.0.is_null() {
            return Err(format!("{name}: ibv_create_cq failed").into());
        }
        Ok(Arc::new(nic))
    }

    /// This port's GID at the index it was opened with.
    pub fn gid(&self) -> Gid {
        self.gid
    }

    /// The library this device was opened through.
    pub fn verbs(&self) -> &'static Verbs {
        self.v
    }

    /// Register `len` bytes at `addr` for local writes and remote writes,
    /// without relaxed ordering.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `addr` stay allocated until the region drops: the
    /// NIC reads and writes them until it is deregistered.
    pub unsafe fn register(
        self: &Arc<Self>,
        addr: *mut u8,
        len: usize,
    ) -> Result<Region, VerbsError> {
        // SAFETY: `pd` is live; the range outlives the region (the caller's
        // contract), which deregisters it on drop.
        let mr = unsafe { (self.v.f.reg_mr)(self.pd.0, addr.cast(), len, ACCESS_STRICT) };
        if mr.is_null() {
            return Err(format!(
                "ibv_reg_mr of {len} bytes failed: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        Ok(Region {
            nic: Arc::clone(self),
            mr: Raw(mr),
            addr: addr as u64,
        })
    }

    /// A reliable-connected queue pair on this device, in its reset state,
    /// that holds `depth` writes in flight.
    pub fn queue_pair(self: &Arc<Self>, depth: u32) -> Result<Qp, VerbsError> {
        let mut init = QpInitAttr {
            qp_context: ptr::null_mut(),
            send_cq: self.cq.0,
            recv_cq: self.cq.0,
            srq: ptr::null_mut(),
            cap: QpCap {
                max_send_wr: depth,
                max_recv_wr: 1,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 0,
            },
            qp_type: QPT_RC,
            sq_sig_all: 0,
        };
        // SAFETY: `pd` and `cq` are live; `init` is the library's layout.
        let qp = unsafe { (self.v.f.create_qp)(self.pd.0, &mut init) };
        if qp.is_null() {
            return Err(format!(
                "ibv_create_qp of {depth} writes failed: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        Ok(Qp {
            nic: Arc::clone(self),
            qp: Raw(qp),
        })
    }

    /// Take up to `wc.len()` completions off the device's queue; how many.
    pub fn poll(&self, wc: &mut [Wc]) -> Result<usize, VerbsError> {
        let n = c_int::try_from(wc.len()).map_err(|_| "poll: too many entries")?;
        // SAFETY: `cq` is live and `wc` holds `n` writable entries.
        let got = unsafe { (self.poll_cq)(self.cq.0, n, wc.as_mut_ptr()) };
        usize::try_from(got).map_err(|_| format!("ibv_poll_cq returned {got}").into())
    }
}

impl Drop for Nic {
    fn drop(&mut self) {
        // Every queue pair and region of this device held it and is gone.
        if !self.cq.0.is_null() {
            // SAFETY: `cq` came from ibv_create_cq and is destroyed once.
            let rc = unsafe { (self.v.f.destroy_cq)(self.cq.0) };
            teardown("ibv_destroy_cq", rc);
        }
        if !self.pd.0.is_null() {
            // SAFETY: `pd` came from ibv_alloc_pd and is freed once.
            let rc = unsafe { (self.v.f.dealloc_pd)(self.pd.0) };
            teardown("ibv_dealloc_pd", rc);
        }
        // SAFETY: `ctx` came from ibv_open_device and is closed once.
        let rc = unsafe { (self.v.f.close_device)(self.ctx.0) };
        teardown("ibv_close_device", rc);
    }
}

/// A teardown call's result, where nothing can return it: a failure is one
/// line on stderr.
fn teardown(call: &str, rc: c_int) {
    if rc != 0 {
        eprintln!("verbs: {call} failed: rc {rc}");
    }
}

/// The names of the devices the library lists.
fn device_names(v: &Verbs) -> Result<Vec<String>, VerbsError> {
    let mut n: c_int = 0;
    // SAFETY: `n` is a writable int; the list is freed below.
    let list = unsafe { (v.f.get_device_list)(&mut n) };
    if list.is_null() {
        return Err("ibv_get_device_list failed".into());
    }
    let count = usize::try_from(n).unwrap_or(0);
    let names = (0..count)
        .map(|i| {
            // SAFETY: the list holds `count` device pointers.
            let dev = unsafe { *list.add(i) };
            // SAFETY: `dev` is a listed device.
            let raw = unsafe { (v.f.get_device_name)(dev) };
            // SAFETY: the name is a NUL-terminated string the library owns
            // while the list lives.
            unsafe { CStr::from_ptr(raw) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    // SAFETY: `list` came from ibv_get_device_list and is freed once.
    unsafe { (v.f.free_device_list)(list) };
    Ok(names)
}

/// The open context of device `name`, the first listed by that name.
fn open_by_name(v: &Verbs, name: &str) -> Result<Raw<Context>, VerbsError> {
    let mut n: c_int = 0;
    // SAFETY: `n` is a writable int; the list is freed below.
    let list = unsafe { (v.f.get_device_list)(&mut n) };
    if list.is_null() {
        return Err("ibv_get_device_list failed".into());
    }
    let opened = match usize::try_from(n) {
        Ok(count) => open_listed(v, list, count, name),
        Err(_) => Err(format!("ibv_get_device_list returned a device count of {n}").into()),
    };
    // SAFETY: `list` came from ibv_get_device_list and is freed once.
    unsafe { (v.f.free_device_list)(list) };
    opened
}

/// The open context of the first of the `count` devices of `list` named
/// `name`, opened while the list lives.
fn open_listed(
    v: &Verbs,
    list: *mut *mut c_void,
    count: usize,
    name: &str,
) -> Result<Raw<Context>, VerbsError> {
    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: the list holds `count` device pointers.
        let dev = unsafe { *list.add(i) };
        // SAFETY: `dev` is a listed device.
        let raw = unsafe { (v.f.get_device_name)(dev) };
        // SAFETY: the name is a NUL-terminated string the library owns while
        // the list lives.
        let dev_name = unsafe { CStr::from_ptr(raw) }.to_string_lossy();
        if dev_name == name {
            // SAFETY: `dev` is a listed device; the context outlives the list.
            let ctx = unsafe { (v.f.open_device)(dev) };
            if ctx.is_null() {
                return Err(format!(
                    "ibv_open_device {name} failed: {}",
                    std::io::Error::last_os_error()
                )
                .into());
            }
            return Ok(Raw(ctx));
        }
        names.push(dev_name.into_owned());
    }
    Err(format!("no RDMA device {name}; devices: {}", names.join(", ")).into())
}

/// A registered range; deregistered on drop.
pub struct Region {
    nic: Arc<Nic>,
    mr: Raw<Mr>,
    addr: u64,
}

impl Region {
    /// The key a local work request names the range by.
    pub fn lkey(&self) -> u32 {
        // SAFETY: `mr` is live until drop.
        unsafe { (*self.mr.0).lkey }
    }

    /// The key a peer's write names the range by.
    pub fn rkey(&self) -> u32 {
        // SAFETY: `mr` is live until drop.
        unsafe { (*self.mr.0).rkey }
    }

    /// The range's first byte, as a work request names it.
    pub fn addr(&self) -> u64 {
        self.addr
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: `mr` came from ibv_reg_mr and is deregistered once.
        let rc = unsafe { (self.nic.v.f.dereg_mr)(self.mr.0) };
        teardown("ibv_dereg_mr", rc);
    }
}

/// What a peer needs to connect a queue pair to this one and write into
/// this node's page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub qpn: u32,
    pub psn: u32,
    pub gid: Gid,
    pub addr: u64,
    pub rkey: u32,
}

/// One RDMA write of a chain: `len` bytes at byte `from` of the local
/// region to the peer's address `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Write {
    pub from: u64,
    pub to: u64,
    pub len: u32,
}

/// One end of a reliable connection; destroyed on drop.
pub struct Qp {
    nic: Arc<Nic>,
    qp: Raw<RawQp>,
}

impl Qp {
    /// This queue pair's number.
    pub fn qpn(&self) -> u32 {
        // SAFETY: `qp` is live until drop.
        unsafe { (*self.qp.0).qp_num }
    }

    /// The device this queue pair is on.
    pub fn nic(&self) -> &Arc<Nic> {
        &self.nic
    }

    fn modify(&self, attr: &mut QpAttr, mask: c_int, step: &str) -> Result<(), VerbsError> {
        // SAFETY: `qp` is live and `attr` the library's layout.
        let rc = unsafe { (self.nic.v.f.modify_qp)(self.qp.0, attr, mask) };
        if rc != 0 {
            return Err(format!("ibv_modify_qp to {step} failed: rc {rc}").into());
        }
        Ok(())
    }

    /// Take the queue pair through init, ready to receive and ready to send
    /// against `remote`, starting this side's sends at `psn`.
    pub fn connect(&self, psn: u32, remote: &Endpoint) -> Result<(), VerbsError> {
        let nic = &self.nic;
        let mut a = qp_attr();
        a.qp_state = QPS_INIT;
        a.port_num = PORT;
        a.qp_access_flags = ACCESS_STRICT as u32;
        self.modify(
            &mut a,
            QP_STATE | QP_PKEY_INDEX | QP_PORT | QP_ACCESS_FLAGS,
            "INIT",
        )?;
        let mut a = qp_attr();
        a.qp_state = QPS_RTR;
        a.path_mtu = nic.mtu;
        a.dest_qp_num = remote.qpn;
        a.rq_psn = remote.psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;
        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = PORT;
        a.ah_attr.grh.dgid = remote.gid;
        a.ah_attr.grh.sgid_index = nic.gid_index;
        a.ah_attr.grh.hop_limit = 64;
        self.modify(
            &mut a,
            QP_STATE
                | QP_AV
                | QP_PATH_MTU
                | QP_DEST_QPN
                | QP_RQ_PSN
                | QP_MAX_DEST_RD_ATOMIC
                | QP_MIN_RNR_TIMER,
            "RTR",
        )?;
        let mut a = qp_attr();
        a.qp_state = QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = psn;
        a.max_rd_atomic = 1;
        self.modify(
            &mut a,
            QP_STATE | QP_TIMEOUT | QP_RETRY_CNT | QP_RNR_RETRY | QP_SQ_PSN | QP_MAX_QP_RD_ATOMIC,
            "RTS",
        )
    }

    /// Post `writes` (at most [`MAX_CHAIN`]) from `region` to the peer's
    /// region keyed `rkey` as one chain, in order; only the last is signaled,
    /// with `wr_id`.
    ///
    /// # Safety
    ///
    /// The NIC reads each source range after the call returns, until the
    /// chain's completion is polled or the queue pair is destroyed. Until
    /// then the caller keeps `region` registered and its memory allocated,
    /// and does not rewrite a posted range. On `Err` the writes before the
    /// one the post refused may have been taken and are still in flight
    /// under the same terms.
    pub unsafe fn post_writes(
        &self,
        region: &Region,
        writes: &[Write],
        rkey: u32,
        wr_id: u64,
    ) -> Result<(), VerbsError> {
        let n = writes.len();
        if n == 0 || n > MAX_CHAIN {
            return Err(format!("a chain of {n} writes: 1 to {MAX_CHAIN}").into());
        }
        let mut sge: [Sge; MAX_CHAIN] = std::array::from_fn(|_| Sge {
            addr: 0,
            length: 0,
            lkey: 0,
        });
        let mut wr: [SendWr; MAX_CHAIN] = std::array::from_fn(|_| SendWr {
            wr_id: 0,
            next: ptr::null_mut(),
            sg_list: ptr::null_mut(),
            num_sge: 1,
            opcode: WR_RDMA_WRITE,
            send_flags: 0,
            imm_data: 0,
            remote_addr: 0,
            rkey,
            wr_rest: [0; 5],
            qp_type: 0,
            tail: [0; 6],
        });
        for (i, w) in writes.iter().enumerate() {
            sge[i] = Sge {
                addr: region.addr + w.from,
                length: w.len,
                lkey: region.lkey(),
            };
            wr[i].remote_addr = w.to;
        }
        wr[n - 1].wr_id = wr_id;
        wr[n - 1].send_flags = SEND_SIGNALED;
        // From here on the chain is reached through these two pointers alone,
        // which the post reads.
        let sges = sge.as_mut_ptr();
        let base = wr.as_mut_ptr();
        for i in 0..n {
            // SAFETY: i < n <= MAX_CHAIN: inside the array, which nothing else
            // reaches while this entry is linked.
            let entry = unsafe { &mut *base.add(i) };
            entry.sg_list = sges.wrapping_add(i);
            entry.next = if i + 1 < n {
                base.wrapping_add(i + 1)
            } else {
                ptr::null_mut()
            };
        }
        let mut bad = ptr::null_mut();
        // SAFETY: `qp` is live; the chain and its SGEs live through the call,
        // which copies them into the send queue.
        let rc = unsafe { (self.nic.post_send)(self.qp.0, base, &mut bad) };
        if rc != 0 {
            return Err(format!("ibv_post_send of {n} writes failed: rc {rc}").into());
        }
        Ok(())
    }
}

impl Drop for Qp {
    fn drop(&mut self) {
        // SAFETY: `qp` came from ibv_create_qp and is destroyed once.
        let rc = unsafe { (self.nic.v.f.destroy_qp)(self.qp.0) };
        teardown("ibv_destroy_qp", rc);
    }
}

/// A zeroed `ibv_qp_attr`.
fn qp_attr() -> QpAttr {
    let ah = || AhAttr {
        grh: GlobalRoute {
            dgid: Gid::default(),
            flow_label: 0,
            sgid_index: 0,
            hop_limit: 0,
            traffic_class: 0,
        },
        dlid: 0,
        sl: 0,
        src_path_bits: 0,
        static_rate: 0,
        is_global: 0,
        port_num: 0,
    };
    QpAttr {
        qp_state: 0,
        cur_qp_state: 0,
        path_mtu: 0,
        path_mig_state: 0,
        qkey: 0,
        rq_psn: 0,
        sq_psn: 0,
        dest_qp_num: 0,
        qp_access_flags: 0,
        cap: QpCap {
            max_send_wr: 0,
            max_recv_wr: 0,
            max_send_sge: 0,
            max_recv_sge: 0,
            max_inline_data: 0,
        },
        ah_attr: ah(),
        alt_ah_attr: ah(),
        pkey_index: 0,
        alt_pkey_index: 0,
        en_sqd_async_notify: 0,
        sq_draining: 0,
        max_rd_atomic: 0,
        max_dest_rd_atomic: 0,
        min_rnr_timer: 0,
        port_num: 0,
        timeout: 0,
        retry_cnt: 0,
        rnr_retry: 0,
        alt_port_num: 0,
        alt_timeout: 0,
        rate_limit: 0,
    }
}
