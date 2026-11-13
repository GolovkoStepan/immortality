//! The daemon↔client wire: line-delimited JSON over a unix socket.
//!
//! # Why line-delimited JSON and not JSON-RPC 2.0
//!
//! The design says "JSON-RPC + events", and events are the part that matters: a client attaches
//! and receives a stream of journal records, possibly for hours. JSON-RPC 2.0 has no framing at
//! all — it assumes a transport that already has message boundaries — so over a byte stream it
//! needs the same newline framing anyway, and its envelope adds `"jsonrpc": "2.0"` to every
//! line for a benefit this project has no use for (it has exactly one client implementation,
//! in the same repository).
//!
//! So the framing is chosen explicitly and the shape is a request/response with an optional
//! stream:
//!
//! ```text
//! client → {"id": 7, "method": "status", "params": {}}
//! daemon → {"id": 7, "result": {...}}
//!        | {"id": 7, "error": {"code": "no_daemon", "message": "..."}}
//! ```
//!
//! Exactly one response per request id, in order on one connection. A method that produces
//! events sends them as separate `event` lines before its response, which means a client that
//! only wants the answer can ignore everything without an `id` and still see the response.
//!
//! # Why the token is in the request and not in a header
//!
//! There are no headers. The socket is `0600` and owned by the user, so the token is a second
//! lock on a door that is already locked — but it is the lock that survives a mis-set umask on
//! a shared `$HOME` (a CI image, a remote dev box), and a file permission is not something a
//! later refactor can silently drop the way an `if` can. The daemon's first response tells the
//! client whether the token is accepted, and a rejected client is closed immediately.
//!
//! # Why the version is checked before anything else
//!
//! A client from a newer build sending a method the daemon has never heard of must get "your
//! client is newer" rather than "unknown method": the second message sends a human looking for
//! a bug in the wrong place.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of this wire protocol. Bumped on any incompatible change.
pub const VERSION: u32 = 1;

/// Longest line accepted, in bytes.
///
/// A `status` response is a few kilobytes; an `attach` event carrying a tool result can be
/// tens. A megabyte is far above anything legitimate and far below what it takes to exhaust
/// the daemon's memory from one connection — which is the point of a limit that exists.
pub const MAX_LINE_BYTES: usize = 1024 * 1024;

/// One request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Matches the response. The client mints it, so it can have several in flight.
    pub id: u64,
    pub method: String,
    #[serde(default)]
    pub params: Value,
    /// Omitted only by `hello`, which must be answerable before the token is known good.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

/// What the daemon sends back.
///
/// One struct with two optional halves rather than an enum with an internally tagged shape: a
/// response is exactly one of "result" and "error", and `deny_unknown_fields` on an enum's
/// variants does not survive serde's tag resolution in a way that stays readable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    /// A successful answer.
    #[must_use]
    pub const fn ok(id: u64, result: Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// A refusal.
    #[must_use]
    pub const fn failed(id: u64, error: RpcError) -> Self {
        Self {
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// One event pushed to an attached client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    /// The journal record, as it was written.
    pub record: Value,
}

/// A refusal with a code a client can branch on and a message a human can read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RpcError {
    pub code: ErrorCode,
    pub message: String,
}

impl RpcError {
    /// A refusal of `code` with `message`.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// The closed set of failure kinds.
///
/// Closed rather than a string: a client that branches on `code` and a daemon that invents a
/// new code must disagree loudly at compile time, not silently at 3 a.m.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The token did not match `daemon.token`.
    Unauthorized,
    /// The client speaks a different protocol version.
    VersionMismatch,
    /// The method does not exist in this build.
    UnknownMethod,
    /// The method exists but the arguments are wrong.
    BadParams,
    /// The thing asked about does not exist: a task, an escalation.
    NotFound,
    /// The daemon could not do it for an internal reason the message explains.
    Internal,
}

impl ErrorCode {
    /// Every code, for a CLI that renders them all.
    pub const ALL: &'static [Self] = &[
        Self::Unauthorized,
        Self::VersionMismatch,
        Self::UnknownMethod,
        Self::BadParams,
        Self::NotFound,
        Self::Internal,
    ];

    /// The wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::VersionMismatch => "version_mismatch",
            Self::UnknownMethod => "unknown_method",
            Self::BadParams => "bad_params",
            Self::NotFound => "not_found",
            Self::Internal => "internal",
        }
    }
}

/// The `hello` request's parameters.
///
/// `version` is the client's own protocol version. It is a field rather than a second method
/// because the whole point of `hello` is to be answerable *before* either side knows whether the
/// other can understand anything: a separate `version` method would itself be subject to the
/// version it is meant to establish.
///
/// Omitted means "do not check" — a hand-written `printf` during debugging must still get an
/// answer, and refusing that would make the one tool a human reaches for useless in exactly the
/// situation it is for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelloParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
}

/// The methods this build answers. Closed, for the same reason as [`ErrorCode`].
///
/// Every method needs the daemon's own state — the journal it exclusively writes, the config it
/// was started with, or the live set from the scheduler. A method that could be answered by a
/// client alone is a method that does not belong here: two readers of the same files are two
/// implementations that will disagree.
pub const METHODS: &[&str] = &[
    "hello",
    "status",
    "escalations",
    "approve",
    "attach",
    "plan",
    "log",
    "gc",
    "merge",
    "start",
];

/// Whether `method` is one this build knows.
#[must_use]
pub fn is_known_method(method: &str) -> bool {
    METHODS.contains(&method)
}

/// Encode one message as a line, rejecting anything that would not survive the read side.
///
/// # Errors
/// The value cannot be serialized, or it contains a newline (which would split one message into
/// two and desynchronise the connection for every message after it).
pub fn encode<T: Serialize>(message: &T) -> immortality_core::Result<String> {
    let text = serde_json::to_string(message)
        .map_err(|error| immortality_core::Error::Protocol(format!("cannot encode a message: {error}")))?;
    if text.contains('\n') {
        return Err(immortality_core::Error::Protocol(
            "an encoded message contains a newline".to_string(),
        ));
    }
    Ok(text)
}

/// Decode one line.
///
/// # Errors
/// Not JSON, or not this protocol's shape. The error names the line's first bytes: a client
/// reading a daemon's complaint about its own request needs to see what it sent.
pub fn decode<T: serde::de::DeserializeOwned>(line: &str) -> immortality_core::Result<T> {
    serde_json::from_str(line).map_err(|error| {
        let preview: String = line.chars().take(120).collect();
        immortality_core::Error::Protocol(format!("not a valid message ({error}): `{preview}`"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_request_roundtrips_with_and_without_a_token() {
        let with = Request {
            id: 3,
            method: "status".to_string(),
            params: json!({"task": "tsk_01"}),
            token: Some("abc".to_string()),
        };
        let text = encode(&with).unwrap();
        assert_eq!(decode::<Request>(&text).unwrap(), with);

        let without = Request {
            id: 1,
            method: "hello".to_string(),
            params: Value::Null,
            token: None,
        };
        let text = encode(&without).unwrap();
        assert!(!text.contains("token"), "{text}");
        assert_eq!(decode::<Request>(&text).unwrap(), without);
    }

    #[test]
    fn a_response_carries_exactly_one_half() {
        let ok = Response::ok(1, json!({"running": true}));
        let text = encode(&ok).unwrap();
        assert!(!text.contains("error"), "{text}");
        assert_eq!(decode::<Response>(&text).unwrap(), ok);

        let failed = Response::failed(2, RpcError::new(ErrorCode::NotFound, "no such task"));
        let text = encode(&failed).unwrap();
        assert!(!text.contains("result"), "{text}");
        assert_eq!(decode::<Response>(&text).unwrap(), failed);
    }

    #[test]
    fn an_unknown_field_is_refused_rather_than_ignored() {
        // A daemon that ignored `params` it did not understand would answer a question the
        // client did not ask.
        let error = decode::<Request>(r#"{"id":1,"method":"hello","surprise":true}"#).unwrap_err();
        assert!(error.to_string().contains("not a valid message"), "{error}");
    }

    #[test]
    fn the_error_message_shows_what_arrived() {
        let error = decode::<Request>("this is not JSON").unwrap_err();
        let text = error.to_string();
        assert!(text.contains("this is not JSON"), "{text}");
    }

    #[test]
    fn encoding_refuses_a_newline_that_would_split_the_stream() {
        // The failure this prevents is subtle and total: one embedded newline makes every
        // following message on the connection decode as garbage.
        let request = Request {
            id: 1,
            method: "hello".to_string(),
            params: json!({"text": "two\nlines"}),
            token: None,
        };
        // `serde_json` escapes the newline, so this specific value is safe; the guard is for a
        // serializer that stops doing that, which is why it asserts on the encoded text.
        let text = encode(&request).unwrap();
        assert!(!text.contains('\n'), "{text}");
    }

    #[test]
    fn every_error_code_roundtrips_through_its_name() {
        for code in ErrorCode::ALL {
            let text = serde_json::to_string(code).unwrap();
            assert_eq!(text, format!("\"{}\"", code.as_str()));
            let back: ErrorCode = serde_json::from_str(&text).unwrap();
            assert_eq!(back, *code);
        }
        assert!(serde_json::from_str::<ErrorCode>("\"made_up\"").is_err());
    }

    #[test]
    fn the_method_list_is_closed_and_has_no_duplicates() {
        let mut sorted = METHODS.to_vec();
        sorted.sort_unstable();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted, deduped, "a duplicate method name is a typo");
        assert!(is_known_method("status"));
        assert!(!is_known_method("push"));
    }

    #[test]
    fn an_event_roundtrips() {
        let event = Event {
            record: json!({"seq": 1, "kind": "task_state"}),
        };
        let text = encode(&event).unwrap();
        assert_eq!(decode::<Event>(&text).unwrap(), event);
    }
}
