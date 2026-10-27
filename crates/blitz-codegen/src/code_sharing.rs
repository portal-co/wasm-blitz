//! Conservative, backend-neutral accounting for assembly code sharing.
//!
//! This module decides whether an equivalence class is worth outlining.  It
//! intentionally does not inspect bytes or mutate a writer: backends provide
//! opaque blobs through `portal_solutions_asm_writer::RewindableCodeWriter` and
//! retain responsibility for safepoints, ABI lowering, and helper emission.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
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

/// Backend-local seam for a future x86-64 SysV outlining pass.
///
/// Implementations must flush/register-canonicalize before a helper call and
/// invalidate cached state afterward. The coordinator does not infer any of
/// these guarantees from a contract key.
pub trait SharingSafepoint {
    type Error;
    fn flush_for_sharing(&mut self) -> Result<(), Self::Error>;
    fn invalidate_after_sharing(&mut self);
    fn emit_helper_call(&mut self, helper_label: usize) -> Result<(), Self::Error>;
    fn emit_helper_return(&mut self) -> Result<(), Self::Error>;
    fn sharing_contract(&self) -> ShareContract;
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
    ///
    /// The outlined form is: one helper body (`inline_bytes`) plus its entry,
    /// return, and layout overhead, plus one call per occurrence.  The inline
    /// baseline is `inline_bytes` per occurrence.
    pub fn net_bytes(self, occurrences: usize, inline_bytes: usize) -> isize {
        let outlined = self.helper_entry_bytes
            + self.helper_return_bytes
            + self.layout_bytes
            + inline_bytes
            + occurrences.saturating_mul(self.call_bytes);
        outlined as isize - occurrences.saturating_mul(inline_bytes) as isize
    }
}

/// Emission hooks a backend provides so [`FunctionStaging`] can lay out
/// helper calls and helper bodies without knowing the concrete writer.
///
/// The implementing adapter wraps the same writer the staging collector
/// drains/appends through. Helper labels come from a separate namespace chosen
/// by the backend (see the plan: helper labels must not collide with `Func`,
/// `Indexed`, `External`, or `Ambient` identities in active use).
pub trait SharingLayout<L> {
    type Error;
    /// Allocate a fresh helper-entry label (monotonic, backend namespace).
    fn alloc_helper_label(&mut self) -> L;
    /// Place a helper-entry label at the current output position.
    fn place_helper_label(&mut self, label: &L) -> Result<(), Self::Error>;
    /// Emit a call to a helper entry at the current output position.
    fn call_helper(&mut self, label: &L) -> Result<(), Self::Error>;
    /// Emit the helper's ordinary return instruction.
    fn helper_return(&mut self) -> Result<(), Self::Error>;
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

/// One region staged by a backend: already drained from the writer at its
/// anchor, awaiting the end-of-function profitability decision.
struct StagedCandidate<L, B> {
    anchor: L,
    key: ShareKey,
    blob: Option<B>,
}

/// Per-function staging collector.
///
/// The backend emits candidate regions normally, then calls
/// [`stage_candidate`](FunctionStaging::stage_candidate) at each proven
/// safepoint; the region is drained into an opaque blob. At function end,
/// [`finalize`](FunctionStaging::finalize) replays the decisions: singleton or
/// unprofitable candidates are appended back inline exactly where they were
/// drained, profitable repeated classes are replaced by calls and their
/// representative blob is emitted once as a helper.
pub struct FunctionStaging<L, B> {
    coordinator: SharingCoordinator,
    config: CodeSharingConfig,
    cost: OutlineCost,
    candidates: Vec<StagedCandidate<L, B>>,
}

/// What `finalize` did, for tests and structured diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinalizeStats {
    pub outlined_classes: usize,
    pub outlined_occurrences: usize,
    pub restored_inline: usize,
    pub net_bytes: isize,
}

impl<L: Clone, B> FunctionStaging<L, B> {
    pub fn new(config: CodeSharingConfig, cost: OutlineCost) -> Self {
        Self { coordinator: SharingCoordinator::new(), config, cost, candidates: Vec::new() }
    }

    /// Number of candidates staged so far (diagnostics).
    pub fn staged_count(&self) -> usize { self.candidates.len() }

    /// Drain the region after `anchor` and stage it as a candidate.
    ///
    /// Returns `Ok(false)` (leaving the region inline) when sharing is off or
    /// the region is below the minimum size; the caller must have flushed and
    /// proven the safepoint before calling.
    pub fn stage_candidate<W, E>(
        &mut self,
        w: &mut W,
        anchor: L,
        contract: ShareContract,
    ) -> Result<bool, E>
    where
        W: portal_solutions_asm_writer::RewindableCodeWriter<L, Blob = B, Error = E>,
    {
        let blob = w.drain_after(&anchor)?;
        let len = w.blob_len(&blob);
        if !self.config.enabled || len < self.config.min_blob_len {
            w.append_blob(blob)?;
            return Ok(false);
        }
        let key = ShareKey { contract, hash: w.blob_hash(&blob), len };
        self.coordinator.record(self.config, Candidate { key, inline_bytes: len });
        self.candidates.push(StagedCandidate { anchor, key, blob: Some(blob) });
        Ok(true)
    }

    /// Apply the end-of-function layout decisions to `w`.
    ///
    /// Candidate processing walks the anchor list in reverse: the suffix after
    /// each anchor is drained, the candidate is replaced by a helper call (or
    /// re-appended inline), and the suffix is re-appended. Helpers are emitted
    /// at the current end of the writer, which the backend must have made
    /// unreachable by fallthrough (e.g. after the function epilogue).
    pub fn finalize<W, RErr, LErr, E>(mut self, w: &mut W) -> Result<FinalizeStats, E>
    where
        W: portal_solutions_asm_writer::RewindableCodeWriter<L, Blob = B, Error = RErr>
            + SharingLayout<L, Error = LErr>,
        E: From<RErr> + From<LErr>,
    {
        let mut stats = FinalizeStats::default();
        if self.candidates.is_empty() || !self.config.enabled {
            return Ok(stats);
        }
        let plans = self.coordinator.profitable(self.config, self.cost);
        let mut helper_labels: BTreeMap<ShareKey, L> = BTreeMap::new();
        for plan in &plans {
            helper_labels.insert(plan.key, w.alloc_helper_label());
            stats.net_bytes += plan.net_bytes;
        }

        // Collision-safe representative assignment: the first member of each
        // planned class is the helper body; later members must compare equal
        // through the writer's exact `blob_eq`, or fall back to inline.
        let mut outlined: Vec<bool> = self.candidates
            .iter()
            .map(|c| helper_labels.contains_key(&c.key))
            .collect();
        let mut representative: BTreeMap<ShareKey, usize> = BTreeMap::new();
        for i in 0..self.candidates.len() {
            if !outlined[i] {
                continue;
            }
            let key = self.candidates[i].key;
            match representative.get(&key) {
                None => {
                    representative.insert(key, i);
                }
                Some(&r) => {
                    let (rep, cand) = (self.candidates[r].blob.as_ref(), self.candidates[i].blob.as_ref());
                    let eq = match (rep, cand) {
                        (Some(rep), Some(cand)) => w.blob_eq(rep, cand),
                        _ => false,
                    };
                    if !eq {
                        outlined[i] = false;
                    }
                }
            }
        }
        // A class reduced to fewer than two outlined members by collisions is
        // no longer profitable; demote it entirely.
        let demoted: Vec<ShareKey> = representative
            .keys()
            .filter(|key| {
                self.candidates.iter().enumerate()
                    .filter(|(i, c)| outlined[*i] && c.key == **key)
                    .count()
                    < 2
            })
            .copied()
            .collect();
        for key in demoted {
            for (i, c) in self.candidates.iter().enumerate() {
                if outlined[i] && c.key == key {
                    outlined[i] = false;
                }
            }
            helper_labels.remove(&key);
            representative.remove(&key);
        }

        for i in (0..self.candidates.len()).rev() {
            let tail = w.drain_after(&self.candidates[i].anchor).map_err(E::from)?;
            if outlined[i] {
                let label = helper_labels[&self.candidates[i].key].clone();
                w.call_helper(&label).map_err(E::from)?;
                stats.outlined_occurrences += 1;
            } else {
                let blob = self.candidates[i].blob.take();
                if let Some(blob) = blob {
                    w.append_blob(blob).map_err(E::from)?;
                }
                stats.restored_inline += 1;
            }
            w.append_blob(tail).map_err(E::from)?;
        }

        for (key, label) in helper_labels.iter() {
            if let Some(&rep) = representative.get(key)
                && outlined[rep] {
                    w.place_helper_label(label).map_err(E::from)?;
                    if let Some(blob) = self.candidates[rep].blob.take() {
                        w.append_blob(blob).map_err(E::from)?;
                    }
                    w.helper_return().map_err(E::from)?;
                    stats.outlined_classes += 1;
                }
        }
        Ok(stats)
    }
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

    // ---- FunctionStaging: fake writer ----

    /// Byte-vector writer whose "hash" is a byte sum, letting tests force
    /// digest collisions at will. Anchors are `usize` labels recorded in a map.
    struct FakeWriter {
        buf: Vec<u8>,
        labels: BTreeMap<usize, usize>,
        next_helper: usize,
    }

    impl FakeWriter {
        fn new() -> Self {
            Self { buf: Vec::new(), labels: BTreeMap::new(), next_helper: 1_000_000 }
        }
        fn anchor(&mut self, label: usize) {
            self.labels.insert(label, self.buf.len());
        }
        fn emit(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }
    }

    impl portal_solutions_asm_writer::RewindableCodeWriter<usize> for FakeWriter {
        type Blob = Vec<u8>;
        type Error = ();
        fn drain_after(&mut self, anchor: &usize) -> Result<Vec<u8>, ()> {
            let off = *self.labels.get(anchor).ok_or(())?;
            Ok(self.buf.split_off(off))
        }
        fn rewind_after(&mut self, anchor: &usize) -> Result<(), ()> {
            drop(self.drain_after(anchor)?);
            Ok(())
        }
        fn append_blob(&mut self, blob: Vec<u8>) -> Result<(), ()> {
            self.buf.extend_from_slice(&blob);
            Ok(())
        }
        fn blob_hash(&self, blob: &Vec<u8>) -> CodeHash {
            let sum: u8 = blob.iter().fold(0u8, |a, b| a.wrapping_add(*b));
            CodeHash([sum; 32])
        }
        fn blob_len(&self, blob: &Vec<u8>) -> usize { blob.len() }
        fn blob_eq(&self, left: &Vec<u8>, right: &Vec<u8>) -> bool { left == right }
    }

    impl SharingLayout<usize> for FakeWriter {
        type Error = ();
        fn alloc_helper_label(&mut self) -> usize {
            let l = self.next_helper;
            self.next_helper += 1;
            l
        }
        fn place_helper_label(&mut self, label: &usize) -> Result<(), ()> {
            self.labels.insert(*label, self.buf.len());
            Ok(())
        }
        fn call_helper(&mut self, _label: &usize) -> Result<(), ()> {
            self.emit(&[0xE8, 0, 0, 0, 0]); // 5-byte call rel32 placeholder
            Ok(())
        }
        fn helper_return(&mut self) -> Result<(), ()> {
            self.emit(&[0xC3]);
            Ok(())
        }
    }

    const X86_COST: OutlineCost = OutlineCost {
        helper_entry_bytes: 0,
        helper_return_bytes: 1,
        call_bytes: 5,
        layout_bytes: 0,
    };

    fn staging_config() -> CodeSharingConfig {
        CodeSharingConfig { enabled: true, min_blob_len: 8, max_added_bytes: 0 }
    }

    #[test]
    fn duplicate_candidates_outline_once_and_restore_singletons() {
        let mut w = FakeWriter::new();
        let mut staging = FunctionStaging::new(staging_config(), X86_COST);
        // A: 16-byte region, twice; B: 16-byte singleton in between.
        w.emit(&[0x55; 8]); // prefix
        w.anchor(1);
        w.emit(&[0x10; 16]);
        assert!(staging.stage_candidate(&mut w, 1, CONTRACT).unwrap());
        w.emit(&[0x66; 4]);
        w.anchor(2);
        w.emit(&[0x99; 16]);
        assert!(staging.stage_candidate(&mut w, 2, CONTRACT).unwrap());
        w.emit(&[0x77; 4]);
        w.anchor(3);
        w.emit(&[0x10; 16]);
        assert!(staging.stage_candidate(&mut w, 3, CONTRACT).unwrap());
        w.emit(&[0xC3]); // function epilogue: helpers go after this point.

        let stats = staging.finalize::<_, (), (), ()>(&mut w).unwrap();
        assert_eq!(stats.outlined_classes, 1);
        assert_eq!(stats.outlined_occurrences, 2);
        assert_eq!(stats.restored_inline, 1);
        assert!(stats.net_bytes < 0);
        // prefix + call + mid + singleton B + mid + call + epilogue + helper+ret
        assert_eq!(w.buf.len(), 8 + 5 + 4 + 16 + 4 + 5 + 1 + 16 + 1);
        // The singleton B region was restored at its original position.
        assert_eq!(&w.buf[8 + 5 + 4..8 + 5 + 4 + 16], &[0x99; 16]);
        // Helper body is the A blob followed by a return.
        let helper_off = w.buf.len() - 17;
        assert_eq!(&w.buf[helper_off..helper_off + 16], &[0x10; 16]);
        assert_eq!(w.buf[helper_off + 16], 0xC3);
    }

    #[test]
    fn hash_collision_falls_back_to_inline() {
        let mut w = FakeWriter::new();
        let mut staging = FunctionStaging::new(staging_config(), X86_COST);
        // Two 16-byte regions with the same byte-sum hash but different bytes.
        w.anchor(1);
        w.emit(&[0x10; 16]);
        assert!(staging.stage_candidate(&mut w, 1, CONTRACT).unwrap());
        w.anchor(2);
        let mut other = [0x10u8; 16];
        other[0] = 0x11;
        other[1] = 0x0F;
        w.emit(&other);
        assert!(staging.stage_candidate(&mut w, 2, CONTRACT).unwrap());

        let stats = staging.finalize::<_, (), (), ()>(&mut w).unwrap();
        assert_eq!(stats.outlined_classes, 0);
        assert_eq!(stats.restored_inline, 2);
        assert_eq!(w.buf.len(), 32);
        assert_eq!(&w.buf[..16], &[0x10; 16]);
        assert_eq!(&w.buf[16..], &other);
    }

    #[test]
    fn singleton_is_restored_inline() {
        let mut w = FakeWriter::new();
        let mut staging = FunctionStaging::new(staging_config(), X86_COST);
        w.emit(&[0xAA; 3]);
        w.anchor(1);
        w.emit(&[0x10; 16]);
        assert!(staging.stage_candidate(&mut w, 1, CONTRACT).unwrap());
        w.emit(&[0xBB; 2]);

        let stats = staging.finalize::<_, (), (), ()>(&mut w).unwrap();
        assert_eq!(stats.outlined_classes, 0);
        assert_eq!(stats.restored_inline, 1);
        let mut expect = Vec::new();
        expect.extend_from_slice(&[0xAA; 3]);
        expect.extend_from_slice(&[0x10; 16]);
        expect.extend_from_slice(&[0xBB; 2]);
        assert_eq!(w.buf, expect);
    }
}
