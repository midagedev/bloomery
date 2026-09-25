//! A minimal HTTP/1.1 server over `std::net`: one thread per connection,
//! keep-alive, `Content-Length` and chunked request bodies, `Expect: 100-continue`,
//! fixed-length responses, and streamed responses that are flushed per event
//! (chunked on HTTP/1.1, raw until close on HTTP/1.0).

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

const MAX_LINE: usize = 16 * 1024;
const MAX_HEADERS: usize = 100;
const MAX_BODY: usize = 64 * 1024 * 1024;

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
}

impl Request {
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

/// One side of a query pair, decoded ([`parse_query`]).
fn percent_decode(s: &str) -> io::Result<String> {
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

fn read_line(r: &mut BufReader<TcpStream>) -> io::Result<Option<String>> {
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
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| bad("header is not UTF-8"))
}

/// Reads the next request; `Ok(None)` when the peer closed between requests.
pub(crate) fn read_request(
    r: &mut BufReader<TcpStream>,
    w: &mut TcpStream,
) -> io::Result<Option<Request>> {
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
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_owned(), v.trim().to_owned()));
        }
    }
    let mut req = Request {
        method: method.to_owned(),
        path: path.to_owned(),
        query: parse_query(query)?,
        http10,
        headers,
        body: Vec::new(),
        keep_alive: false,
    };
    let conn = req.header("connection").map(str::to_ascii_lowercase);
    req.keep_alive = if http10 {
        conn.as_deref() == Some("keep-alive")
    } else {
        conn.as_deref() != Some("close")
    };
    if req
        .header("expect")
        .is_some_and(|e| e.eq_ignore_ascii_case("100-continue"))
    {
        w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        w.flush()?;
    }
    let chunked = req
        .header("transfer-encoding")
        .is_some_and(|t| t.to_ascii_lowercase().contains("chunked"));
    if chunked {
        req.body = read_chunked(r)?;
    } else if let Some(len) = req.header("content-length") {
        let len: usize = len.parse().map_err(|_| bad("bad content-length"))?;
        if len > MAX_BODY {
            return Err(bad("body too large"));
        }
        let mut body = vec![0; len];
        r.read_exact(&mut body)?;
        req.body = body;
    }
    Ok(Some(req))
}

fn read_chunked(r: &mut BufReader<TcpStream>) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let line = read_line(r)?.ok_or_else(|| bad("closed inside a chunk"))?;
        let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| bad("bad chunk size"))?;
        if size == 0 {
            // Trailers until the blank line.
            while read_line(r)?.is_some_and(|l| !l.is_empty()) {}
            return Ok(body);
        }
        if body.len() + size > MAX_BODY {
            return Err(bad("body too large"));
        }
        let at = body.len();
        body.resize(at + size, 0);
        r.read_exact(&mut body[at..])?;
        read_line(r)?;
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
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
    w: &mut TcpStream,
    req: &Request,
    status: u16,
    ctype: &str,
    extra: &[(&str, String)],
    body: &[u8],
) -> io::Result<()> {
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
pub(crate) struct EventStream<'a> {
    w: &'a mut TcpStream,
    chunked: bool,
}

impl<'a> EventStream<'a> {
    /// Sends the head. HTTP/1.0 streams end by closing the connection.
    pub(crate) fn start(
        w: &'a mut TcpStream,
        req: &Request,
        status: u16,
        ctype: &str,
    ) -> io::Result<Self> {
        let chunked = !req.http10;
        let mut h = head(
            req,
            status,
            ctype,
            &[("Cache-Control", "no-cache".to_owned())],
        );
        h.push_str(if chunked {
            "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"
        } else {
            "Connection: close\r\n\r\n"
        });
        w.write_all(h.as_bytes())?;
        w.flush()?;
        Ok(EventStream { w, chunked })
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

    /// Ends the body.
    pub(crate) fn finish(self) -> io::Result<()> {
        if self.chunked {
            self.w.write_all(b"0\r\n\r\n")?;
        }
        self.w.flush()
    }

    /// Whether the connection can carry another request afterwards.
    pub(crate) fn reusable(&self) -> bool {
        self.chunked
    }
}

#[cfg(test)]
mod tests {
    use super::parse_query;

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
