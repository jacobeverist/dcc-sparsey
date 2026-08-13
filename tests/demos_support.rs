//! Runs the unit tests inside `examples/support/`.
//!
//! Example targets default to `test = false`, so the `#[cfg(test)]` blocks in that
//! tree are compiled by `cargo build --all-targets` but never executed. Including
//! the same tree here compiles it in test configuration, which is what gets them
//! run by `cargo test`.

#[path = "../examples/support/mod.rs"]
mod support;

/// The tree has to be referenced for the module to be compiled at all.
#[test]
fn the_support_tree_is_compiled_in_test_configuration() {
    let args = support::args::Args::from_iter(["--seed", "7"].iter().map(|s| s.to_string()));
    assert_eq!(args.seed(), 7);
}
