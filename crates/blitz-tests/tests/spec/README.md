# Spec-test baseline

Known-failure list for the WASM spec-test harness (`tests/spectests.rs`).
Format: TOML with `[[entries]]` records.

```toml
[[entries]]
file = "i32"        # wast file stem (test/core/<file>.wast)
idx = 17            # 0-based directive index within the file
reason = "why this assertion fails"
```

## Ratchet rules (enforced by the harness)

1. A failure recorded here does **not** fail CI.
2. A failure **not** recorded here fails CI — no silent regressions.
3. An entry here whose assertion now **passes** fails CI ("stale entry") —
   the baseline can only shrink.

## Adding entries

Every entry requires a `reason`. Entries are keyed by directive index within
the file, valid for the pinned suite commit (see CI workflow). When the suite
pin is bumped, indices may shift: re-run and update entries accordingly.

## Re-baselining

Set `BLITZ_SPEC_REBASELINE=1` to print the new baseline diff instead of
failing (tooling lands with the execution bridge).
