// Shared scaffolding for the demos.
//
// This is not a crate and not a cargo target. Cargo examples cannot depend on one
// another, so each demo pulls this tree in with
//
//     #[path = "support/mod.rs"]
//     mod support;
//
// A directory under `examples/` with no `main.rs` is not auto-discovered, so this
// is not itself built as an example.
//
// Example targets default to `test = false`, so the `#[cfg(test)]` blocks below
// would never run. `tests/demos_support.rs` includes the same tree in test
// configuration, which is what gets them executed by `cargo test`.
//
// The flags, record schema and reporting conventions here are the cross-repo demo
// contract shared with `dcc-sph` and `dcc-htm`; see `doc/Demos.md`. The three
// repositories share no *code* — `dcc_sph` is CC BY-NC-SA 4.0 and this crate is
// AGPL-3.0, and those licences do not mix — so each implements the contract
// separately.

#![allow(dead_code)]

pub mod args;
pub mod checkpoint;
pub mod env;
pub mod metrics;
pub mod probe;
pub mod report;
pub mod rng;
pub mod sweep;
