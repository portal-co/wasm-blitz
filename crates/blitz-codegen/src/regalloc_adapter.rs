//! Generic [`portal_solutions_asm_regalloc::Length`]/`Index`/`IndexMut` adapter.
//!
//! Every regalloc-backed backend (`blitz-riscv64::naive`, `blitz-x86-64::fast`,
//! and eventually an AArch64 equivalent) previously hand-copied the same
//! `Frames(pub [[RegAllocFrame<K>; N]; 2])` wrapper plus its `Index`/`IndexMut`/
//! `Length` impls, differing only in the arch's `RegKind` type. This module
//! provides that wrapper once, generic over any `K` that `RegAlloc` itself
//! already requires (`Clone + Eq + TryFrom<usize>`) — no new per-arch trait
//! impl is needed, since `Frames` is the local type satisfying orphan rules.

use core::ops::{Index, IndexMut};
use portal_solutions_asm_regalloc::{Length, RegAllocFrame};

/// Register frames for a regalloc-backed backend with two kinds (int/float),
/// `N` physical registers each. Kind 0 and kind 1 are distinguished via
/// `K::try_from(0)`/`K::try_from(1)` (the same round-trip `RegAlloc` itself
/// relies on), so no extra trait needs implementing per arch.
pub struct Frames<K, const N: usize>(pub [[RegAllocFrame<K>; N]; 2]);

fn slot<K: Clone + Eq + TryFrom<usize>>(k: &K) -> usize {
    match K::try_from(0) {
        Ok(zero) if zero == *k => 0,
        _ => 1,
    }
}

impl<K: Clone + Eq + TryFrom<usize>, const N: usize> Index<K> for Frames<K, N> {
    type Output = [RegAllocFrame<K>; N];
    fn index(&self, k: K) -> &Self::Output {
        &self.0[slot(&k)]
    }
}

impl<K: Clone + Eq + TryFrom<usize>, const N: usize> IndexMut<K> for Frames<K, N> {
    fn index_mut(&mut self, k: K) -> &mut Self::Output {
        &mut self.0[slot(&k)]
    }
}

impl<K, const N: usize> Length for Frames<K, N> {
    fn len(&self) -> usize {
        2
    }
}
