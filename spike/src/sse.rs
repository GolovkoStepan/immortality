//! Incremental Server-Sent Events parser for OpenAI-compatible streaming endpoints.
//!
//! Written to be lifted into `immortality-llm` almost unchanged, so it is stricter than a
//! spike needs to be: it tolerates arbitrary chunk boundaries (including inside a UTF-8
//! sequence and inside a CRLF pair), ignores `:` comments (`: keep-alive`), and does not
//! assume `data:` payloads arrive in one piece.
//!
//! Reference: <https://html.spec.whatwg.org/multipage/server-sent-events.html>

/// One dispatched SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub id: Option<String>,
    /// All `data:` lines of this event, joined with `\n` as the spec requires.
    pub data: String,
}

/// Byte-level incremental parser. Feed it whatever the socket returns.
#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    data: Vec<String>,
    event: Option<String>,
    id: Option<String>,
    in_event: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a chunk of bytes and collect every event that completed inside it.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for &byte in chunk {
            if byte == b'\n' {
                let line = std::mem::take(&mut self.line);
                if let Some(event) = self.take_line(&line) {
                    out.push(event);
                }
            } else {
                self.line.push(byte);
            }
        }
        out
    }

    /// Flush a trailing event that was not terminated by a blank line.
    ///
    /// Real servers end with a blank line, but a truncated stream should still surface the
    /// last complete `data:` line rather than silently losing it.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            if let Some(event) = self.take_line(&line) {
                return Some(event);
            }
        }
        self.dispatch()
    }

    /// Returns true while an event has been started but not dispatched.
    pub fn in_event(&self) -> bool {
        self.in_event
    }

    fn take_line(&mut self, raw: &[u8]) -> Option<SseEvent> {
        // Strip a CR that belongs to a CRLF pair.
        let raw = match raw.split_last() {
            Some((b'\r', head)) => head,
            _ => raw,
        };
        let text = String::from_utf8_lossy(raw);

        if text.is_empty() {
            return self.dispatch();
        }
        // Comments are keep-alives. They must not dispatch or reset an in-progress event.
        if text.starts_with(':') {
            return None;
        }

        let (field, value) = match text.find(':') {
            Some(idx) => {
                let (f, rest) = text.split_at(idx);
                let rest = &rest[1..];
                (f, rest.strip_prefix(' ').unwrap_or(rest))
            }
            None => (text.as_ref(), ""),
        };

        match field {
            "data" => {
                self.data.push(value.to_string());
                self.in_event = true;
            }
            "event" => {
                self.event = Some(value.to_string());
                self.in_event = true;
            }
            "id" => {
                self.id = Some(value.to_string());
                self.in_event = true;
            }
            // `retry` and unknown fields are ignored per spec.
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if !self.in_event {
            return None;
        }
        let event = SseEvent {
            event: self.event.take(),
            id: self.id.take(),
            data: self.data.join("\n"),
        };
        self.data.clear();
        self.in_event = false;
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(parser: &mut SseParser, input: &str) -> Vec<SseEvent> {
        parser.push(input.as_bytes())
    }

    #[test]
    fn parses_two_events() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data: a\n\ndata: b\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
    }

    #[test]
    fn ignores_keep_alive_comments() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data: a\n\n: keep-alive\n\ndata: b\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
    }

    #[test]
    fn comment_inside_event_does_not_dispatch() {
        let mut p = SseParser::new();
        let mut events = feed(&mut p, "data: a\n: keep-alive\n");
        assert!(events.is_empty());
        events.extend(feed(&mut p, "\n"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a");
    }

    #[test]
    fn survives_one_byte_at_a_time() {
        let source = "data: {\"a\":1}\n\n: keep-alive\n\ndata: [DONE]\n\n";
        let mut p = SseParser::new();
        let mut events = Vec::new();
        for byte in source.as_bytes() {
            events.extend(p.push(std::slice::from_ref(byte)));
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\"a\":1}");
        assert_eq!(events[1].data, "[DONE]");
    }

    #[test]
    fn survives_split_utf8() {
        // "привет" split in the middle of a 2-byte sequence.
        let text = "data: привет\n\n".as_bytes();
        let split = text.iter().position(|&b| b >= 0xD0).unwrap() + 1;
        let mut p = SseParser::new();
        let mut events = p.push(&text[..split]);
        events.extend(p.push(&text[split..]));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "привет");
    }

    #[test]
    fn handles_crlf_terminators() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data: a\r\n\r\ndata: b\r\n\r\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
    }

    #[test]
    fn joins_multiline_data() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn strips_single_leading_space_only() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data:  two spaces\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, " two spaces");
    }

    #[test]
    fn names_event_and_id() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "event: message\nid: 7\ndata: x\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message"));
        assert_eq!(events[0].id.as_deref(), Some("7"));
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "data: tail");
        assert!(events.is_empty());
        // An unterminated line is not yet a field, so no event is in progress.
        assert!(!p.in_event());
        assert_eq!(p.finish().unwrap().data, "tail");
        assert!(!p.in_event());
        // Nothing left to flush.
        assert!(p.finish().is_none());
    }

    #[test]
    fn blank_lines_outside_events_are_ignored() {
        let mut p = SseParser::new();
        let events = feed(&mut p, "\n\n\n\ndata: x\n\n\n\n");
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn real_chunk_shape_from_strata() {
        // Captured from the live backend: a role delta, then a keep-alive, then content.
        let source = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            ": keep-alive\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"3\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let mut p = SseParser::new();
        let events = feed(&mut p, source);
        assert_eq!(events.len(), 3);
        assert_eq!(events[2].data, "[DONE]");
    }
}
