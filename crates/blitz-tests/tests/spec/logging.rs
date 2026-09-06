//! Structured logging per AGENTS.md compression-aware convention.
//!
//! `PORTAL_LOG_JSON=1` emits NDJSON events on stderr instead of ad-hoc
//! `eprintln!`. Zero cost when unset. `PORTAL_LOG_BAT=1` batches events per
//! test-function unit into single JSON arrays.

use std::io::Write;

#[derive(Clone)]
pub struct Logger {
    pub json: bool,
    pub batch: bool,
    pub batch_buf: std::cell::RefCell<Vec<String>>,
    pub unit: std::cell::RefCell<String>,
}

impl Logger {
    pub fn from_env() -> Self {
        Logger {
            json: std::env::var_os("PORTAL_LOG_JSON").as_deref() == Some(std::ffi::OsStr::new("1")),
            batch: std::env::var_os("PORTAL_LOG_BAT").as_deref() == Some(std::ffi::OsStr::new("1")),
            batch_buf: std::cell::RefCell::new(Vec::new()),
            unit: std::cell::RefCell::new(String::new()),
        }
    }

    pub fn begin_batch(&self, unit: &str) {
        if !self.batch {
            return;
        }
        self.batch_buf.borrow_mut().clear();
        *self.unit.borrow_mut() = unit.to_string();
    }

    pub fn end_batch(&self) {
        if !self.batch {
            return;
        }
        let buf = self.batch_buf.borrow();
        if buf.is_empty() {
            return;
        }
        let unit = self.unit.borrow();
        let mut line = String::from("[");
        for (i, ev) in buf.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(ev);
        }
        line.push(']');
        eprintln!(
            "{{\"l\":\"batch\",\"unit\":{},\"events\":{}}}",
            json_str(&unit),
            line
        );
    }

    pub fn event(&self, level: &str, phase: &str, msg: &str, fields: &[(&str, &str)]) {
        if !self.json {
            if level == "fail" && !self.batch {
                eprintln!("[{phase}] {msg}");
            }
            return;
        }
        let mut line = format!(
            "{{\"l\":{l},\"phase\":{p},\"msg\":{m}",
            l = json_str(level),
            p = json_str(phase),
            m = json_str(msg)
        );
        for (k, v) in fields {
            line.push_str(&format!(",{k}:{v}", k = json_str(k), v = json_str(v)));
        }
        line.push('}');
        if self.batch {
            self.batch_buf.borrow_mut().push(line);
        } else {
            let _ = writeln!(std::io::stderr(), "{line}");
        }
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
