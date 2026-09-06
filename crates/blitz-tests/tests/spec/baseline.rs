//! Tracked known-failure baseline (`baseline.toml`).
//!
//! Ratchet semantics (see `docs/spectests-plan.md`):
//! * failure on baseline  → known failure, CI green
//! * failure off baseline → new failure, CI red
//! * baseline entry that passes → stale entry, CI red (baseline only shrinks)
//!
//! Parsed by hand: the format is three fields per record, and keeping this
//! dependency-free avoids pulling serde+toml into the test build.

use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaselineEntry {
    pub file: String,
    pub idx: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Baseline {
    pub entries: Vec<BaselineEntry>,
}

impl Baseline {
    pub fn load(path: &Path) -> Baseline {
        let Ok(src) = std::fs::read_to_string(path) else {
            return Baseline::default();
        };
        Self::parse(&src).unwrap_or_else(|e| {
            panic!("baseline.toml is malformed ({e}); fix or delete it");
        })
    }

    pub fn parse(src: &str) -> Result<Baseline, String> {
        let mut baseline = Baseline::default();
        let mut current: Option<BaselineEntry> = None;
        let mut in_entries = false;
        for (lineno, raw) in src.lines().enumerate() {
            let line = raw.trim();
            let fail = |msg: &str| -> String { format!("line {}: {msg}", lineno + 1) };
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "[[entries]]" {
                if let Some(e) = current.take() {
                    baseline.entries.push(e);
                }
                in_entries = true;
                continue;
            }
            if !in_entries {
                // Top-level keys like `entries = []` from the empty template.
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(fail("expected `key = \"value\"`"));
            };
            let key = key.trim();
            let value = value.trim().trim_matches('"').to_string();
            match key {
                "file" => {
                    current
                        .get_or_insert_with(|| BaselineEntry {
                            file: String::new(),
                            idx: 0,
                            reason: String::new(),
                        })
                        .file = value;
                }
                "idx" => {
                    let idx = value
                        .parse()
                        .map_err(|_| fail("idx must be a non-negative integer"))?;
                    current
                        .get_or_insert_with(|| BaselineEntry {
                            file: String::new(),
                            idx: 0,
                            reason: String::new(),
                        })
                        .idx = idx;
                }
                "reason" => {
                    if let Some(e) = current.as_mut() {
                        e.reason = value;
                    }
                }
                other => return Err(fail(&format!("unknown key `{other}`"))),
            }
        }
        if let Some(e) = current.take() {
            baseline.entries.push(e);
        }
        for (i, e) in baseline.entries.iter().enumerate() {
            if e.file.is_empty() || e.reason.is_empty() {
                return Err(format!("entry #{i} is missing `file` or `reason`"));
            }
        }
        Ok(baseline)
    }

    pub fn contains(&self, file: &str, idx: usize) -> bool {
        self.entries.iter().any(|e| e.file == file && e.idx == idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_records() {
        let b = Baseline::parse(
            "entries = []\n[[entries]]\nfile = \"i32\"\nidx = 3\nreason = \"trap msg\"\n",
        )
        .unwrap();
        assert_eq!(b.entries.len(), 1);
        assert_eq!(b.entries[0].file, "i32");
        assert_eq!(b.entries[0].idx, 3);
        assert!(b.contains("i32", 3));
        assert!(!b.contains("i32", 4));
    }

    #[test]
    fn empty_template() {
        let b = Baseline::parse("# comment\nentries = []\n").unwrap();
        assert!(b.entries.is_empty());
    }

    #[test]
    fn rejects_missing_reason() {
        assert!(Baseline::parse("[[entries]]\nfile = \"x\"\nidx = 0\n").is_err());
    }
}
