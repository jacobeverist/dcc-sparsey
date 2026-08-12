//! Local conformance checks for the dcc-core import contract.
//!
//! dcc-core imports this crate as a rev-pinned git dependency and wraps it as a `Node`.
//! That imposes requirements `R1`–`R16`, defined in dcc-core's
//! `docs/claude/third-party-import-pattern.md`; this crate's verdict on each, and the
//! rationale for its one exception, is in `doc/Conformance.md`.
//!
//! **Why the checks live here rather than in dcc-core.** A violation should fail in the
//! repository that can fix it, at the moment it is introduced — not months later in a
//! consumer's build, phrased as something else. The `getrandom` breakage this crate
//! shipped from extraction until 2026-08-12 is the worked example: `cargo check
//! --target wasm32-unknown-unknown` failed *here* the whole time, and nothing looked,
//! because dcc-core's own wasm build supplies a backend and masked it.
//!
//! These checks read `Cargo.toml` and `Cargo.lock` as text via `include_str!`. That is
//! deliberate — it keeps the test dependency-free (nothing to keep in step with
//! dcc-core, no toml parser) and it is why R3's "commit `Cargo.lock`" is load-bearing
//! for more than CI caching: without the lockfile there is nothing to assert against.

const CARGO_TOML: &str = include_str!("../Cargo.toml");
const CARGO_LOCK: &str = include_str!("../Cargo.lock");

/// Everything inside `[dependencies]`, stopping at the next top-level table.
fn dependencies_section() -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in CARGO_TOML.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t == "[dependencies]";
            continue;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(!out.is_empty(), "no [dependencies] section found in Cargo.toml");
    out
}

/// R4 — the long-lived algorithm object must be `Send + Sync`.
///
/// dcc-core's `Node` trait requires it, because the engine executes nodes across
/// dependency levels in parallel. This is a compile-time assertion: if `SparseyNet`
/// ever acquires an `Rc`, a `Cell`, or a raw pointer, this test stops compiling — which
/// is the correct moment to find out, rather than when the wrapper fails to build.
#[test]
fn r4_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<dcc_sparsey::SparseyNet>();
    assert_send_sync::<dcc_sparsey::NetworkConfig>();
    assert_send_sync::<dcc_sparsey::RegionConfig>();
    assert_send_sync::<dcc_sparsey::SparseyError>();
}

/// R1 — the package dcc-core depends on is library-only.
///
/// A binary target here would be built by nobody and shipped to everybody. Applications
/// belong in a separate crate in this repository (R16), which never enters a consumer's
/// dependency graph at all.
#[test]
fn r1_no_binary_targets() {
    assert!(
        !CARGO_TOML.contains("[[bin]]"),
        "R1: this package must be library-only. Put applications in a separate crate \
         under a workspace `members` list instead — see doc/Conformance.md."
    );
}

/// R2 — dcc-agnostic: this crate must not depend on dcc-core.
///
/// The whole point of the two-layer split. Prose references in doc comments are fine
/// and this crate has several; a *code* dependency would make the extraction pointless
/// and create a cycle, since dcc-core depends on this.
#[test]
fn r2_no_dcc_dependency() {
    for line in dependencies_section().lines() {
        let name = line.split(['=', ' ']).next().unwrap_or("").trim();
        assert!(
            !name.starts_with("dcc_") && !name.starts_with("dcc-"),
            "R2: this crate must not depend on dcc-core or any dcc_* crate, found: {line}"
        );
    }
}

/// R3 — its own workspace root, with a real SPDX license.
///
/// The `[workspace]` table stops cargo searching upward and absorbing this crate into
/// someone else's workspace — which is how a vendored crate silently acquires
/// workspace-inherited dependency versions. The license must be an SPDX expression
/// because it is a machine-readable claim consumers rely on; a copyright line there is
/// both invalid and unactionable.
#[test]
fn r3_standalone_workspace_root_with_spdx_license() {
    assert!(
        CARGO_TOML.contains("[workspace]"),
        "R3: a [workspace] table (empty, or with `members`) is required so cargo treats \
         this manifest as its own root."
    );

    let license = CARGO_TOML
        .lines()
        .find(|l| l.trim_start().starts_with("license"))
        .expect("R3: [package] must declare a license");
    assert!(
        license.contains("AGPL-3.0-or-later"),
        "R3: license must be the SPDX expression AGPL-3.0-or-later, never a copyright \
         line. Found: {license}"
    );
}

/// R10 — behavior-critical dependencies must match dcc-core's majors.
///
/// Two `serde` majors mean the derived types stop being the same types across the
/// boundary. Two `rand` majors mean a **second RNG stream**, which silently unseeds
/// whatever draws from it — cargo builds both without complaint and no test fails.
///
/// These are pinned here as an expectation rather than derived, because this crate
/// cannot see dcc-core's manifest. When dcc-core bumps one of these, this test is what
/// tells you this repo has to move first: the port repos bump, then dcc-core re-pins.
#[test]
fn r10_behavior_critical_majors_match_dcc_core() {
    const EXPECTED: &[(&str, &str)] = &[
        ("rand", "0.9"),
        ("rand_xoshiro", "0.7"),
        ("serde", "1.0"),
        ("thiserror", "2.0"),
        ("schemars", "0.8"),
    ];

    let deps = dependencies_section();
    for (crate_name, major) in EXPECTED {
        let line = deps
            .lines()
            .find(|l| l.trim_start().starts_with(crate_name))
            .unwrap_or_else(|| panic!("R10: expected a `{crate_name}` dependency"));
        assert!(
            line.contains(major),
            "R10: `{crate_name}` must stay on {major} to match dcc-core's \
             [workspace.dependencies]. Bump THIS repo first, then re-pin dcc-core. \
             Found: {line}"
        );
    }
}

/// R12 — `getrandom` must not be in the dependency graph at all.
///
/// Not merely "do not select a backend". `getrandom` cannot compile for
/// wasm32-unknown-unknown without being told which backend to use, and selecting one is
/// a *binary's* decision — a library that picks `wasm_js` imposes the browser on every
/// Node and WASI consumer. So the fix is not to reach for a backend but to not need
/// one: this crate is entirely seed-driven, so `rand`/`rand_xoshiro` take
/// `default-features = false` and `getrandom` never enters the graph.
///
/// This crate shipped the opposite from extraction until 2026-08-12, failing a
/// standalone wasm32 check the entire time with nobody looking. Asserting *absence*
/// rather than a build flag is what makes that unrepeatable.
#[test]
fn r12_getrandom_is_absent_from_the_graph() {
    assert!(
        !CARGO_LOCK.contains("name = \"getrandom\""),
        "R12: `getrandom` is in Cargo.lock, so this crate no longer builds for \
         wasm32-unknown-unknown standalone. It arrives through `rand`/`rand_xoshiro` \
         DEFAULT features (rand_core's `os_rng`). This crate is seed-driven and calls \
         no OS-entropy API, so the fix is `default-features = false` — never selecting \
         a getrandom backend here, which is a binary's decision."
    );
}

/// R12, second half — no `rand` default features, which is *how* getrandom stays out.
///
/// Separate from the assertion above so the failure names the cause rather than the
/// symptom: this one fails at the edit that would reintroduce the problem.
#[test]
fn r12_rand_crates_disable_default_features() {
    let deps = dependencies_section();
    for crate_name in ["rand", "rand_xoshiro"] {
        let line = deps
            .lines()
            .find(|l| l.trim_start().starts_with(crate_name))
            .unwrap_or_else(|| panic!("R12: expected a `{crate_name}` dependency"));
        assert!(
            line.contains("default-features = false"),
            "R12: `{crate_name}` must set default-features = false, or rand_core's \
             `os_rng` pulls in getrandom and the wasm32 build breaks. Found: {line}"
        );
    }
}
