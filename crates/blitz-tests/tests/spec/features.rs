//! Feature gating (phase 4, `docs/spectests-plan.md`).
//!
//! Maps spec-suite areas to the proposals wasm-blitz's backends claim to
//! support. Files requiring features beyond a backend's capability are
//! auto-skipped with a counted reason (not baseline entries).

use wasmparser::WasmFeatures;

/// Proposal features relevant to spectest file selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    Mvp,
    MutableGlobal,
    SignExtension,
    SaturatingFloatToInt,
    ReferenceTypes,
    MultiValue,
    BulkMemory,
    MultiMemory,
    TailCall,
}

impl Feature {
    pub fn wasmparser_flag(self) -> WasmFeatures {
        match self {
            Feature::Mvp => WasmFeatures::MVP,
            Feature::MutableGlobal => WasmFeatures::MUTABLE_GLOBAL,
            Feature::SignExtension => WasmFeatures::SIGN_EXTENSION,
            Feature::SaturatingFloatToInt => WasmFeatures::SATURATING_FLOAT_TO_INT,
            Feature::ReferenceTypes => WasmFeatures::REFERENCE_TYPES,
            Feature::MultiValue => WasmFeatures::MULTI_VALUE,
            Feature::BulkMemory => WasmFeatures::BULK_MEMORY,
            Feature::MultiMemory => WasmFeatures::MULTI_MEMORY,
            Feature::TailCall => WasmFeatures::TAIL_CALL,
        }
    }
}

/// The feature set validated by the spectest harness (matches
/// `validate_binary` in the driver and the backends' claimed coverage).
pub const HARNESS_FEATURES: &[Feature] = &[
    Feature::Mvp,
    Feature::MutableGlobal,
    Feature::SignExtension,
    Feature::SaturatingFloatToInt,
    Feature::ReferenceTypes,
    Feature::MultiValue,
    Feature::BulkMemory,
    Feature::MultiMemory,
    Feature::TailCall,
];

/// Build the wasmparser feature set for validation.
pub fn harness_features() -> WasmFeatures {
    let mut f = WasmFeatures::MVP;
    for feat in HARNESS_FEATURES.iter().skip(1) {
        f |= feat.wasmparser_flag();
    }
    f.remove(WasmFeatures::SIMD | WasmFeatures::THREADS);
    f
}

/// Proposal directories under `test/core/` in the spec repo, with the feature
/// each requires. Phase 4 opt-in: a backend enables a directory once its e2e
/// smoke exists for every instruction family in it.
pub const PROPOSAL_DIRS: &[(&str, Feature)] = &[];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_assemble() {
        let f = harness_features();
        assert!(f.contains(WasmFeatures::MULTI_MEMORY));
        assert!(!f.contains(WasmFeatures::SIMD));
    }
}
