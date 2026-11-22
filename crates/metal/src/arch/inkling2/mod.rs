//! The Inkling 2 family on this backend, as host arithmetic: the visits'
//! buffers ([`layout`]) and the dispatches and bytes a decoded token walks
//! ([`count`]). The session that runs a token is a later round's; these two
//! files are the byte model its arena and its plan are checked against.

pub mod count;
pub mod layout;
