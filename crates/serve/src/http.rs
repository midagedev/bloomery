//! A minimal HTTP/1.1 server over `std::net`: one thread per connection,
//! keep-alive, `Content-Length` and chunked request bodies, gzip and deflate
//! request bodies (`Content-Encoding`), `Expect: 100-continue`,
//! fixed-length responses, and streamed responses that are flushed per event
//! (chunked on HTTP/1.1, raw until close on HTTP/1.0).

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;

use flate2::bufread::{MultiGzDecoder, ZlibDecoder};

use crate::admit::Lease;
use crate::api::relock;

const MAX_LINE: usize = 16 * 1024;
const MAX_HEADERS: usize = 100;
pub(crate) const MAX_BODY: usize = 64 * 1024 * 1024;
/// The content codings a request body is read in besides `identity`, as `Accept-Encoding` lists
/// them on the refusal of another.
pub(crate) const BODY_CODINGS: &str = "gzip, deflate";

/// One parsed request.
pub(crate) struct Request {
    pub method: String,
    pub path: String,
    /// The query string's `key=value` pairs, percent-decoded (`+` is a
    /// space), in order; a key with no `=` has the empty value.
    pub query: Vec<(String, String)>,
    pub http10: bool,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub keep_alive: bool,
    /// The place the request holds among its key's in-flight requests
    /// ([`crate::admit`]), ended by [`Request::end_lease`]. A lock because the
    /// answer's writers hold the request shared: they release their share
    /// once the answer's last byte is written and flushed. A background
    /// worker may retain another share until its generation ends.
    pub lease: Mutex<Option<Lease>>,
}

impl Request {
    /// The writer's lease share, taken: the place returns when it and any
    /// background worker's share have dropped.
    pub(crate) fn end_lease(&self) -> Option<Lease> {
        relock(&self.lease).take()
    }

    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Whether the query string carries `key` (with or without a value).
    pub(crate) fn has_query(&self, key: &str) -> bool {
        self.query.iter().any(|(k, _)| k == key)
    }

    /// The decoded value of the first `key=value` in the query string; `key`
    /// with no `=` reads as the empty value.
    pub(crate) fn query_value(&self, key: &str) -> Option<&str> {
        self.query
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v.as_str()))
    }

    /// The 404's message for a request no route of the server answers: its method and path.
    pub(crate) fn not_found(&self) -> String {
        format!("File Not Found: {} {}", self.method, self.path)
    }
}

/// What a path names below `/v1/<collection>/` (or `/<collection>/`), as sent; `None` for
/// another path or an empty rest.
pub(crate) fn below<'a>(path: &'a str, collection: &str) -> Option<&'a str> {
    let path = path
        .strip_prefix("/v1")
        .filter(|p| p.starts_with('/'))
        .unwrap_or(path);
    path.strip_prefix('/')?
        .strip_prefix(collection)?
        .strip_prefix('/')
        .filter(|rest| !rest.is_empty())
}

/// The pairs of a query string, each side percent-decoded as a form field
/// (`+` is a space, `%XY` the byte `0xXY`). A `%` not followed by two hex
/// digits, or bytes that decode to no UTF-8, is refused by name.
fn parse_query(query: &str) -> io::Result<Vec<(String, String)>> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            Ok((percent_decode(k)?, percent_decode(v)?))
        })
        .collect()
}

/// One side of a query pair, decoded ([`parse_query`]); also a model name a path carries
/// ([`crate::models::named`], which escapes a `+` first: in a path it is a `+`).
pub(crate) fn percent_decode(s: &str) -> io::Result<String> {
    let hex = |b: u8| char::from(b).to_digit(16);
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let (hi, lo) = match (bytes.get(i + 1), bytes.get(i + 2)) {
                    (Some(&h), Some(&l)) => (hex(h), hex(l)),
                    _ => (None, None),
                };
                let (Some(hi), Some(lo)) = (hi, lo) else {
                    return Err(bad("malformed percent-encoding in the query string"));
                };
                out.push(u8::try_from(hi * 16 + lo).expect("two hex digits fit a byte"));
                i += 2;
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8(out).map_err(|_| bad("the query string decodes to no UTF-8"))
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_owned())
}

/// The refusal of a body the request declares past `MAX_BODY`: it names both sizes. The limit is
/// fixed, no flag raises it.
fn too_large(declared: usize) -> io::Error {
    bad(&format!(
        "body too large: the request declares {declared} bytes, the limit is {MAX_BODY} bytes ({} MiB)",
        MAX_BODY >> 20
    ))
}

fn read_raw_line(r: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let mut buf = Vec::new();
    let n = r
        .by_ref()
        .take(MAX_LINE as u64 + 1)
        .read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > MAX_LINE {
        return Err(bad("header line too long"));
    }
    Ok(Some(buf))
}

fn read_line(r: &mut impl BufRead) -> io::Result<Option<String>> {
    let Some(mut buf) = read_raw_line(r)? else {
        return Ok(None);
    };
    if buf.last() == Some(&b'\n') {
        buf.pop();
        if buf.last() == Some(&b'\r') {
            buf.pop();
        }
    }
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| bad("header is not UTF-8"))
}

/// Reads the next request's headers, leaving its body unread; `Ok(None)` when the peer closed.
pub(crate) fn read_head(r: &mut BufReader<TcpStream>) -> io::Result<Option<Request>> {
    let line = loop {
        match read_line(r)? {
            None => return Ok(None),
            // Tolerate stray CRLFs between keep-alive requests.
            Some(l) if l.is_empty() => {}
            Some(l) => break l,
        }
    };
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target), Some(version)) = (parts.next(), parts.next(), parts.next())
    else {
        return Err(bad("malformed request line"));
    };
    let http10 = version == "HTTP/1.0";
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers = Vec::new();
    loop {
        let Some(h) = read_line(r)? else {
            return Err(bad("connection closed inside headers"));
        };
        if h.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(bad("too many headers"));
        }
        // A header line needs a colon right after a field name that is an HTTP token.
        let Some((k, v)) = h.split_once(':').filter(|(k, _)| {
            !k.is_empty()
                && k.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
        }) else {
            return Err(bad("bad header line"));
        };
        headers.push((k.to_owned(), v.trim_matches([' ', '\t']).to_owned()));
    }
    let mut req = Request {
        method: method.to_owned(),
        path: path.to_owned(),
        query: parse_query(query)?,
        http10,
        headers,
        body: Vec::new(),
        keep_alive: false,
        lease: Mutex::default(),
    };
    let conn = req.header("connection").map(str::to_ascii_lowercase);
    req.keep_alive = if http10 {
        conn.as_deref() == Some("keep-alive")
    } else {
        conn.as_deref() != Some("close")
    };
    Ok(Some(req))
}

/// Reads an authorized request's body, sending `100 Continue` first when requested.
pub(crate) fn read_body(
    r: &mut BufReader<TcpStream>,
    w: &mut TcpStream,
    req: &mut Request,
) -> io::Result<()> {
    // The only transfer coding read is a single `chunked`; any other list is refused.
    let mut codings = req
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("transfer-encoding"))
        .map(|(_, t)| t.as_str());
    let chunked = match (codings.next(), codings.next()) {
        (None, _) => false,
        (Some(t), None) if t.eq_ignore_ascii_case("chunked") => true,
        _ => return Err(bad("unsupported transfer-encoding")),
    };
    let length = if chunked { None } else { content_length(req)? };
    if chunked && req.header("content-length").is_some() {
        // Both framings declared: read as chunked, then close.
        req.keep_alive = false;
    }
    if let Some(len) = length
        && len > MAX_BODY
    {
        return Err(too_large(len));
    }
    // A content coding describes a body; a request that frames none reads as sent.
    let coding = if chunked || length.is_some_and(|len| len > 0) {
        content_coding(req)?
    } else {
        None
    };
    if req
        .header("expect")
        .is_some_and(|e| e.eq_ignore_ascii_case("100-continue"))
    {
        w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        w.flush()?;
    }
    if chunked {
        req.body = read_chunked(r)?;
    } else if let Some(len) = length {
        let mut body = vec![0; len];
        read_framed(r, &mut body)?;
        req.body = body;
    }
    if let Some(coding) = coding {
        req.body = decode(coding, &req.body)?;
    }
    Ok(())
}

/// A content coding a request body is read in.
#[derive(Clone, Copy)]
enum Coding {
    /// `gzip` (or `x-gzip`): one or more gzip members.
    Gzip,
    /// `deflate`: a zlib stream.
    Deflate,
}

/// The coding `Content-Encoding` names over all its lines; `identity` and empty list elements are
/// none. Another coding, or a list of codings applied one after another, is refused as
/// `Unsupported` (a 415 that names [`BODY_CODINGS`]) before the body is read: one decode bounds a
/// request's work by the body limit.
fn content_coding(req: &Request) -> io::Result<Option<Coding>> {
    let named: Vec<&str> = req
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("content-encoding"))
        .flat_map(|(_, v)| v.split(','))
        .map(|c| c.trim_matches([' ', '\t']))
        .filter(|c| !c.is_empty() && !c.eq_ignore_ascii_case("identity"))
        .collect();
    match named[..] {
        [] => Ok(None),
        [c] if c.eq_ignore_ascii_case("gzip") || c.eq_ignore_ascii_case("x-gzip") => {
            Ok(Some(Coding::Gzip))
        }
        [c] if c.eq_ignore_ascii_case("deflate") => Ok(Some(Coding::Deflate)),
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "unsupported content-encoding {:?}: the server reads a body in one coding, {BODY_CODINGS} or identity",
                named.join(", ")
            ),
        )),
    }
}

/// `body` decoded from `coding`. A body that does not decode, decodes past `MAX_BODY` or carries
/// bytes past its stream is refused by name; decoding stops one byte past the limit.
fn decode(coding: Coding, body: &[u8]) -> io::Result<Vec<u8>> {
    let limit = MAX_BODY as u64 + 1;
    let mut rest = body;
    let mut decoded = Vec::new();
    let (name, read) = match coding {
        Coding::Gzip => (
            "gzip",
            MultiGzDecoder::new(&mut rest)
                .take(limit)
                .read_to_end(&mut decoded),
        ),
        Coding::Deflate => (
            "deflate",
            ZlibDecoder::new(&mut rest)
                .take(limit)
                .read_to_end(&mut decoded),
        ),
    };
    read.map_err(|e| bad(&format!("the body does not decode as {name}: {e}")))?;
    if decoded.len() > MAX_BODY {
        return Err(bad(&format!(
            "body too large: the {name} body decodes past the limit of {MAX_BODY} bytes ({} MiB)",
            MAX_BODY >> 20
        )));
    }
    if !rest.is_empty() {
        return Err(bad(&format!(
            "{} bytes follow the body's {name} stream",
            rest.len()
        )));
    }
    Ok(decoded)
}

/// The body length a request declares: `None` without `Content-Length`. A value that is not
/// all digits, or two values that differ, are refused.
pub(crate) fn content_length(req: &Request) -> io::Result<Option<usize>> {
    let mut length = None;
    for (_, value) in req
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
    {
        for value in value
            .split(',')
            .map(|value| value.trim_matches([' ', '\t']))
        {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad("bad content-length"));
            }
            let len: usize = value.parse().map_err(|_| bad("bad content-length"))?;
            if length.is_some_and(|previous| previous != len) {
                return Err(bad("conflicting content-length"));
            }
            length = Some(len);
        }
    }
    Ok(length)
}

/// Fills `buf` from the body; a connection that closes before the framing ends is refused.
fn read_framed(r: &mut impl Read, buf: &mut [u8]) -> io::Result<()> {
    r.read_exact(buf).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => bad("closed inside the body"),
        _ => e,
    })
}

fn read_chunk_line(r: &mut impl BufRead) -> io::Result<String> {
    let bytes = read_raw_line(r)?.ok_or_else(|| bad("closed inside a chunk"))?;
    let bytes = bytes
        .strip_suffix(b"\r\n")
        .ok_or_else(|| bad("bad chunk line ending"))?;
    String::from_utf8(bytes.to_vec()).map_err(|_| bad("chunk line is not UTF-8"))
}

fn read_chunked(r: &mut impl BufRead) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let line = read_chunk_line(r)?;
        let digits = match line.split_once(';') {
            Some((digits, _)) => digits.trim_end_matches([' ', '\t']),
            None => line.as_str(),
        };
        let size = digits
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
            .then(|| usize::from_str_radix(digits, 16).ok())
            .flatten()
            .ok_or_else(|| bad("bad chunk size"))?;
        if size == 0 {
            // Trailers until the blank line.
            while !read_chunk_line(r)?.is_empty() {}
            return Ok(body);
        }
        // The loop keeps `body.len() <= MAX_BODY`, so the subtraction cannot underflow.
        if size > MAX_BODY - body.len() {
            return Err(too_large(body.len().saturating_add(size)));
        }
        let at = body.len();
        body.resize(at + size, 0);
        read_framed(r, &mut body[at..])?;
        let mut end = [0; 2];
        read_framed(r, &mut end)?;
        if end != *b"\r\n" {
            return Err(bad("bad chunk data ending"));
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

fn head(req: &Request, status: u16, ctype: &str, extra: &[(&str, String)]) -> String {
    let version = if req.http10 { "HTTP/1.0" } else { "HTTP/1.1" };
    let mut h = format!(
        "{version} {status} {}\r\nContent-Type: {ctype}\r\nServer: bloomery-serve\r\nAccess-Control-Allow-Origin: *\r\n",
        reason(status)
    );
    for (k, v) in extra {
        h.push_str(k);
        h.push_str(": ");
        h.push_str(v);
        h.push_str("\r\n");
    }
    h
}

/// Writes a complete response.
pub(crate) fn respond(
    w: &mut impl Write,
    req: &Request,
    status: u16,
    ctype: &str,
    extra: &[(&str, String)],
    body: &[u8],
) -> io::Result<()> {
    let _lease = req.end_lease();
    let mut h = head(req, status, ctype, extra);
    h.push_str(&format!("Content-Length: {}\r\n", body.len()));
    h.push_str(if req.keep_alive {
        "Connection: keep-alive\r\n\r\n"
    } else {
        "Connection: close\r\n\r\n"
    });
    w.write_all(h.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// A response whose body is written as it is produced.
pub(crate) struct EventStream<'a, W: Write = TcpStream> {
    w: &'a mut W,
    chunked: bool,
    /// The request's lease, ended with the stream, after the body's end is
    /// written and flushed ([`Request::end_lease`]).
    lease: Option<Lease>,
}

impl<'a, W: Write> EventStream<'a, W> {
    /// Sends the head. HTTP/1.0 streams end by closing the connection.
    pub(crate) fn start(w: &'a mut W, req: &Request, status: u16, ctype: &str) -> io::Result<Self> {
        let chunked = !req.http10;
        let mut h = head(
            req,
            status,
            ctype,
            &[("Cache-Control", "no-cache".to_owned())],
        );
        if chunked {
            h.push_str("Transfer-Encoding: chunked\r\n");
        }
        h.push_str(if chunked && req.keep_alive {
            "Connection: keep-alive\r\n\r\n"
        } else {
            "Connection: close\r\n\r\n"
        });
        w.write_all(h.as_bytes())?;
        w.flush()?;
        Ok(EventStream {
            w,
            chunked,
            lease: req.end_lease(),
        })
    }

    /// Writes and flushes one piece of the body.
    pub(crate) fn send(&mut self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if self.chunked {
            write!(self.w, "{:x}\r\n", data.len())?;
            self.w.write_all(data)?;
            self.w.write_all(b"\r\n")?;
        } else {
            self.w.write_all(data)?;
        }
        self.w.flush()
    }

    /// Ends the body, then the lease.
    pub(crate) fn finish(self) -> io::Result<()> {
        if self.chunked {
            self.w.write_all(b"0\r\n\r\n")?;
        }
        let flushed = self.w.flush();
        drop(self.lease);
        flushed
    }

    /// Whether the connection can carry another request afterwards.
    pub(crate) fn reusable(&self) -> bool {
        self.chunked
    }
}

/// Reads one test response without consuming the next response on a persistent connection.
/// Heads and bodies stay below 64 KiB; `100 Continue` has no body, while final responses
/// must declare their exact body length. Truncated headers or bodies fail the test.
#[cfg(test)]
pub(crate) fn read_test_response(r: &mut impl BufRead) -> (u16, String, Vec<u8>) {
    const MAX_RESPONSE: usize = 64 * 1024;
    let mut head = String::new();
    loop {
        let mut line = String::new();
        let n = r
            .by_ref()
            .take((MAX_RESPONSE - head.len()) as u64)
            .read_line(&mut line)
            .expect("response head");
        assert!(
            n > 0 && line.ends_with("\r\n"),
            "incomplete response head: {head}{line}"
        );
        head.push_str(&line);
        assert!(head.len() < MAX_RESPONSE, "a bounded test response head");
        if line == "\r\n" {
            break;
        }
    }
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .expect("a status line");
    let len = if status == 100 {
        0
    } else {
        head.lines()
            .filter_map(|line| line.split_once(':'))
            .find_map(|(name, value)| {
                name.eq_ignore_ascii_case("Content-Length")
                    .then_some(value.trim())
            })
            .expect("a fixed-length response")
            .parse::<usize>()
            .expect("the response length")
    };
    assert!(len < MAX_RESPONSE, "a bounded test response body: {len}");
    let mut body = vec![0; len];
    r.read_exact(&mut body).expect("the complete response body");
    (status, head, body)
}

#[cfg(test)]
mod tests {
    use super::{EventStream, MAX_BODY, parse_query, read_body, read_chunked, read_head, respond};
    use crate::admit::{Limits, Meter};
    use std::io::{self, BufReader, ErrorKind, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;
    use std::time::Instant;

    /// A writer that notes, at each write and flush, whether its key's one
    /// place is free: a write that blocked there would block with the place
    /// already given to another request.
    struct PlaceProbe {
        meter: Arc<Meter>,
        free: Vec<bool>,
    }

    impl PlaceProbe {
        fn note(&mut self) {
            self.free
                .push(self.meter.admit(Instant::now(), true).is_ok());
        }
    }

    impl Write for PlaceProbe {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.note();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.note();
            Ok(())
        }
    }

    /// With `inflight=1`, a whole answer and a stream each hold the key's
    /// place through every write, the last write and flush included, and
    /// give it back once they end.
    #[test]
    fn an_answer_holds_its_place_through_its_last_write_and_flush() {
        let meter = Arc::new(Meter::new(Limits::parse([("inflight", "1")]).unwrap()));
        for stream in [false, true] {
            let req = crate::api::closing();
            *crate::api::relock(&req.lease) = meter.admit(Instant::now(), true).unwrap();
            let mut probe = PlaceProbe {
                meter: Arc::clone(&meter),
                free: Vec::new(),
            };
            if stream {
                let mut s = EventStream::start(&mut probe, &req, 200, "text/event-stream")
                    .expect("the head");
                s.send(b"data: x\n\n").expect("an event");
                s.finish().expect("the end");
            } else {
                respond(&mut probe, &req, 200, "text/plain", &[], b"ok").expect("the answer");
            }
            assert!(
                !probe.free.is_empty() && probe.free.iter().all(|free| !free),
                "stream={stream}: the place was free at a write: {:?}",
                probe.free
            );
            assert!(
                meter.admit(Instant::now(), true).is_ok(),
                "stream={stream}: the place returns once the answer ends"
            );
        }
    }

    /// Chunks join into the body, extensions and trailers aside; a chunk size
    /// that carries the body past the limit is refused by name, also when the
    /// client's size would wrap the running total.
    #[test]
    fn chunked_bodies_decode_and_stop_at_the_limit() {
        let body = read_chunked(&mut &b"3;x=y\r\nabc\r\n1\r\nd\r\n0\r\nT: v\r\n\r\n"[..])
            .expect("a valid chunked body");
        assert_eq!(body, b"abcd");
        for hostile in [
            "4000001\r\n",
            "1\r\na\r\nffffffffffffffff\r\n",
            "2\r\nab\r\nfffffffffffffffe\r\n",
        ] {
            let err = read_chunked(&mut hostile.as_bytes()).expect_err(hostile);
            assert_eq!(err.kind(), ErrorKind::InvalidData, "{hostile:?}");
            assert!(
                err.to_string()
                    .starts_with("body too large: the request declares "),
                "{hostile:?}: {err}"
            );
        }
    }

    /// A body past the limit is refused on either framing with a message that names the size the
    /// request declares and the limit, so a client can tell how far over it is.
    #[test]
    fn a_body_past_the_limit_is_refused_with_both_sizes() {
        let over = MAX_BODY + 1;
        let want = format!(
            "body too large: the request declares {over} bytes, the limit is {MAX_BODY} bytes ({} MiB)",
            MAX_BODY >> 20
        );
        let chunked = read_chunked(&mut format!("{:x}\r\n", over).as_bytes()).expect_err("chunked");
        assert_eq!(chunked.to_string(), want, "a chunk size past the limit");

        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let mut client = TcpStream::connect(listener.local_addr().expect("its address"))
            .expect("the client connects");
        let (server, _) = listener.accept().expect("the server accepts");
        client
            .write_all(
                format!("POST /v1/chat/completions HTTP/1.1\r\ncontent-length: {over}\r\n\r\n")
                    .as_bytes(),
            )
            .expect("the head is sent");
        let mut w = server.try_clone().expect("a writer half");
        let mut r = BufReader::new(server);
        let mut req = read_head(&mut r).expect("a valid head").expect("a request");
        let err = read_body(&mut r, &mut w, &mut req).expect_err("a body past the limit");
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert_eq!(err.to_string(), want, "a content-length past the limit");
    }

    /// `%2F` decodes to `/`, `+` to a space, in keys and values alike; a key
    /// with no `=` has the empty value; a broken escape is an error.
    #[test]
    fn query_pairs_are_percent_decoded() {
        let q = parse_query("action=a%2Fb+c&fail%5Fon%5Fno%5Fslot&x=%7e").expect("a valid query");
        assert_eq!(
            q,
            [
                ("action".to_owned(), "a/b c".to_owned()),
                ("fail_on_no_slot".to_owned(), String::new()),
                ("x".to_owned(), "~".to_owned()),
            ]
        );
        assert!(parse_query("").expect("the empty query").is_empty());
        for broken in ["a=%2", "a=%zz", "a=%", "a=%ff"] {
            assert!(parse_query(broken).is_err(), "{broken} was accepted");
        }
    }
}
