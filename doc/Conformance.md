# Conformance with the dcc-core import contract

dcc-core imports this crate as a rev-pinned git dependency and wraps it as a `Node`. That imposes sixteen requirements, `R1`–`R16`.

**The requirements are defined in dcc-core**, in `docs/claude/third-party-import-pattern.md` under "What the crate itself must satisfy". This file is **this crate's answers**, not a second copy of the questions — copying sixteen definitions into three port repositories is how they would drift apart, which has already happened twice to this project's convention docs.

**What is mechanically checked lives in [`../tests/conformance.rs`](../tests/conformance.rs)** and runs on every `cargo test`. The rest is recorded here with rationale, because some requirements are judgements rather than assertions.

Checks belong here rather than in dcc-core so a violation fails in the repository that can fix it, at the moment it is introduced — see R12 below for what happens otherwise.

## Status

| | Requirement | Verdict | Checked by |
|---|---|---|---|
| R1 | Library-only package | ✅ | `r1_no_binary_targets` |
| R2 | dcc-agnostic — no dcc-core dependency | ✅ | `r2_no_dcc_dependency` |
| R3 | Own workspace root, SPDX license, committed lockfile | ✅ | `r3_standalone_workspace_root_with_spdx_license` |
| R4 | Algorithm object is `Send + Sync` | ✅ | `r4_public_types_are_send_and_sync` (compile-time) |
| R5 | State expressible as grouped CSDR or sparse index list | ✅ | wrapper-side; see note |
| R6 | Config types are serde `Serialize + Deserialize + Clone` | ✅ | `#[derive]` on `NetworkConfig` |
| R7 | Learned state serializes to bytes | ✅ | `serialize_state` / `load_state` |
| R8 | Forward/update separation declared | ✅ | monolithic; see note |
| R9 | RNG per-object and seed-parameterised, no globals | ✅ | by construction; see note |
| R10 | Behavior-critical majors match dcc-core | ✅ | `r10_behavior_critical_majors_match_dcc_core` |
| R11 | No `pyo3` | ✅ | no such dependency |
| R12 | `getrandom` absent; wasm32 clean | ✅ | `r12_*` ×2, plus a CI wasm32 build |
| R13 | `json-schema` feature for owned config types | ✅ | `json-schema = ["dep:schemars"]` |
| R14 | Builds in isolation under a single feature | ✅ | dcc-core's CI |
| R15 | Node type tag is prefix-identifiable | ✅ | `SparseyNetworkNode`, wrapper-side |
| R16 | Local apps kept out of a consumer's graph | n/a | no local crates yet; see note |

## Notes on the rows that are not a plain yes

**R5 — representation.** This crate is the hybrid of the three. Its *input* is a plain sparse pattern (active feature indices), while its *output* is a grouped code — one winning cell per CM, `num_macs · Q` columns of `K` cells. The wrapper bridges the output through dcc-core's shared grouped-SDR helper rather than a conversion of its own. Nothing is required of this crate beyond exposing both, which `set_input` and the `Recorder` callback do.

**R8 — monolithic, which is one of the two sanctioned answers.** R8 asks a crate to *declare* which of two shapes it has, not to have a particular one: "either a re-runnable deterministic forward pass, or accept the monolithic-step recipe." This crate is the second. `do_frame_learn` and `do_frame_recognize` each do forward *and* update in a single call, so there is no re-runnable forward pass to split.

What that obliges is on the wrapper side: no-op `compute()`, and override **both** `execute` and `execute_in_thread` to step exactly once per tick. Overriding only one compiles, passes `cargo test`, and is wrong in the browser — which is why the answer has to be written down rather than inferred.

*(This row read ❌ until 2026-08-12. That was a mis-marking, not a finding: it recorded a property as a failure. Nothing in this crate changed.)*

**R9 — no global RNG.** Satisfied by construction and worth stating explicitly, because the sibling port `dcc_sph` fails it and pays for it continuously. Every generator here is owned by the `SparseyNet` instance and seeded through `SparseyNet::build(config, seed)`; the Java reference's `globalFrameAcrossAllEpisodes` and `operationalMode` statics became instance fields during the port. Two networks in one process cannot contaminate each other, and no thread-isolation harness is needed to get reproducible numbers.

**R12 — the one this crate got wrong, and the reason these checks are local.** From extraction until 2026-08-12 this crate pulled in `getrandom` through `rand`'s default features, for an OS-entropy path it never calls, and therefore failed `cargo check --target wasm32-unknown-unknown` in its own repository the entire time. Nobody noticed, because dcc-core's wasm build supplies a `getrandom` backend and masked it. It was found only when someone sat down to write the requirement out.

The fix was not to select a backend — that is a *binary's* decision, and choosing `wasm_js` here would impose the browser on every Node and WASI consumer. It was to stop needing one: `default-features = false`, after which `getrandom` is absent from the graph rather than present-and-unsatisfied. `r12_getrandom_is_absent_from_the_graph` asserts the absence, so the failure cannot return quietly.

**R16 — no local crates yet.** This repository is a single package today. If it grows an app, viewer or integration, the mechanism is a workspace `members` list with the extra crate *not* a dependency of the library — never an optional feature, which would bind its constraints on every consumer that enables it. `examples/`, `tests/` and `benches/` are free: a consumer never resolves an external package's dev-dependencies, verified from dcc-core with `cargo tree -p dcc_sparsey --edges dev` printing nothing.

## Bumping a dependency

`R10` is the one requirement that cannot be satisfied locally in isolation. When dcc-core moves `rand`, `serde`, `schemars` or `thiserror` to a new major, **this repository moves first** and dcc-core then re-pins to the new SHA. `r10_behavior_critical_majors_match_dcc_core` holds the expected majors, so a mismatch fails here with a message saying which way the change has to flow.
