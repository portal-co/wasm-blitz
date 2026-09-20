//! Conservative, backend-neutral accounting for assembly code sharing.
//!
//! This module decides whether an equivalence class is worth outlining.  It
//! intentionally does not inspect bytes or mutate a writer: backends provide
//! opaque blobs through `portal_solutions_asm_writer::RewindableCodeWriter` and
//! retain responsibility for safepoints, ABI lowering, and helper emission.

use alloc::collections::BTreeMap;
use portal_solutions_asm_writer::CodeHash;

/// Per-function sharing policy.  The default is deliberately off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeSharingConfig {
    /// Enable candidate accounting and transformation by a backend that has
    /// opted in.  Disabled means emitted code must remain byte-identical.
    pub enabled: bool,
    /// Do not consider small candidates: a call and helper return would cost
    /// more than they save in the common case.
    pub min_blob_len: usize,
    /// Hard cap on bytes that a function may add due to helper/call layout.
    pub max_added_bytes: usize,
}

impl Default for CodeSharingConfig {
    fn default() -> Self {
        Self { enabled: false, min_blob_len: 16, max_added_bytes: 0 }
    }
}

/// Facts that must be equal before two identically hashed blobs can share a
/// generated helper.  Backend adapters pack ISA-specific state into the keys;
/// the values are opaque to this module but never optional.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShareContract {
    /// Target ISA and enabled encoding features.
    pub target: u64,
    /// Generated-helper calling convention and clobber set.
    pub abi: u64,
    /// Caller frame/local layout.
    pub frame: u64,
    /// Canonical WASM operand-stack and allocator state at the safepoint.
    pub machine_state: u64,
    /// Memory, sandbox, instrumentation, and control-flow context.
    pub dynamic_context: u64,
}

impl ShareContract {
    /// Construct a contract from backend-defined, already-canonicalized keys.
    /// The coordinator never interprets these values; it only requires exact
    /// equality before opaque blobs may be considered for sharing.
    pub const fn new(
        target: u64,
        abi: u64,
        frame: u64,
        machine_state: u64,
        dynamic_context: u64,
    ) -> Self {
        Self { target, abi, frame, machine_state, dynamic_context }
    }
}

/// Key used only to form a candidate equivalence class.  A matching key is
/// not a proof of byte equality: callers must ask their rewindable writer's
/// `blob_eq` before committing to share a class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShareKey {
    pub contract: ShareContract,
    pub hash: CodeHash,
    pub len: usize,
}

/// One candidate recorded by a backend after it has proven a call safepoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub key: ShareKey,
    /// Actual inline bytes which would be removed.  Kept separate from the
    /// hash length so a backend can account for padding or non-code metadata.
    pub inline_bytes: usize,
}

/// Cost model for one generated helper class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutlineCost {
    pub helper_entry_bytes: usize,
    pub helper_return_bytes: usize,
    pub call_bytes: usize,
    pub layout_bytes: usize,
}

impl OutlineCost {
    /// Net change compared with retaining every candidate inline.  Negative
    /// means smaller output.
    pub fn net_bytes(self, occurrences: usize, inline_bytes: usize) -> isize {
        let outlined = self.helper_entry_bytes
            + self.helper_return_bytes
            + self.layout_bytes
            + occurrences.saturating_mul(self.call_bytes);
        outlined as isize - occurrences.saturating_mul(inline_bytes) as isize
    }
}

/// Aggregate candidate accounting for a single function.
#[derive(Default, Debug)]
pub struct SharingCoordinator {
    classes: BTreeMap<ShareKey, ClassStats>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ClassStats {
    occurrences: usize,
    inline_bytes: usize,
}

/// A profitable class selected for outlining.  Blob ownership and exact
/// collision checks stay with the backend/writer, so this result contains only
/// accounting facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutlinePlan {
    pub key: ShareKey,
    pub occurrences: usize,
    pub net_bytes: isize,
}

impl SharingCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a candidate.  A disabled policy records nothing, preserving the
    /// no-overhead default path.
    pub fn record(&mut self, config: CodeSharingConfig, candidate: Candidate) {
        if !config.enabled || candidate.key.len < config.min_blob_len {
            return;
        }
        let stats = self.classes.entry(candidate.key).or_default();
        stats.occurrences += 1;
        stats.inline_bytes += candidate.inline_bytes;
    }

    /// Select classes whose measured output size decreases.  The returned
    /// plans are deterministic because `BTreeMap` orders `ShareKey`.
    pub fn profitable(&self, config: CodeSharingConfig, cost: OutlineCost) -> alloc::vec::Vec<OutlinePlan> {
        if !config.enabled {
            return alloc::vec::Vec::new();
        }
        self.classes
            .iter()
            .filter_map(|(&key, stats)| {
                if stats.occurrences < 2 {
                    return None;
                }
                let net_bytes = cost.net_bytes(stats.occurrences, key.len);
                // A class whose final layout grows is never selected. The
                // per-function cap is enforced by the backend when it totals
                // selected helper-area overhead across classes.
                (net_bytes < 0)
                    .then_some(OutlinePlan { key, occurrences: stats.occurrences, net_bytes })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTRACT: ShareContract = ShareContract {
        target: 1,
        abi: 2,
        frame: 3,
        machine_state: 4,
        dynamic_context: 5,
    };

    fn candidate(hash: u8, len: usize) -> Candidate {
        Candidate {
            key: ShareKey { contract: CONTRACT, hash: CodeHash([hash; 32]), len },
            inline_bytes: len,
        }
    }

    #[test]
    fn disabled_policy_is_inert() {
        let mut sharing = SharingCoordinator::new();
        sharing.record(CodeSharingConfig::default(), candidate(1, 64));
        assert!(sharing.profitable(CodeSharingConfig::default(), OutlineCost {
            helper_entry_bytes: 0, helper_return_bytes: 1, call_bytes: 5, layout_bytes: 0,
        }).is_empty());
    }

    #[test]
    fn repeated_class_is_selected_only_when_smaller() {
        let config = CodeSharingConfig { enabled: true, min_blob_len: 8, max_added_bytes: 0 };
        let mut sharing = SharingCoordinator::new();
        sharing.record(config, candidate(1, 32));
        sharing.record(config, candidate(1, 32));
        sharing.record(config, candidate(2, 32));
        let plans = sharing.profitable(config, OutlineCost {
            helper_entry_bytes: 0, helper_return_bytes: 1, call_bytes: 5, layout_bytes: 0,
        });
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].occurrences, 2);
        assert!(plans[0].net_bytes < 0);
    }

    #[test]
    fn contract_partitions_equal_hashes() {
        let config = CodeSharingConfig { enabled: true, min_blob_len: 8, max_added_bytes: 0 };
        let mut sharing = SharingCoordinator::new();
        sharing.record(config, candidate(1, 32));
        let mut other = candidate(1, 32);
        other.key.contract.frame = 99;
        sharing.record(config, other);
        assert!(sharing.profitable(config, OutlineCost {
            helper_entry_bytes: 0, helper_return_bytes: 1, call_bytes: 5, layout_bytes: 0,
        }).is_empty());
    }
}
