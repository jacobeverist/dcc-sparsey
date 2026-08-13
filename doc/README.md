# sparsey — documentation index

Docs for the `dcc_sparsey` crate (a dcc-core-owned Rust port of the **Sparsey** SDM architecture; upstream `SparseyCore`, Java). Standard doc set — see [`docs/claude/third-party-import-pattern.md`](https://github.com/jacobeverist/dcc-core/blob/main/docs/claude/third-party-import-pattern.md) → "Crate documentation standard":

| Doc | Purpose |
|-----|---------|
| [Architecture.md](Architecture.md) | The **Rust implementation** architecture (arena model, frame loop, learning mechanics). Upstream algorithm spec: [SparseyCore ARCHITECTURE.md](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md). |
| [NameReference.md](NameReference.md) | Terminology glossary — mapped to DCC canonical terms in the [nomenclature crosswalk](https://github.com/jacobeverist/dcc-core/blob/main/docs/canonical/vocabulary/nomenclature-crosswalk.md). |
| [PortNotes.md](PortNotes.md) | Upstream (Java `SparseyCore`) → Rust name/type mapping. |
| [Divergences.md](Divergences.md) | Architectural/structural differences vs upstream — the fidelity audit (what we changed / deferred / missed). |
| [MethodFidelity.md](MethodFidelity.md) | Method-by-method comparison vs upstream Java (faithful / divergent / missing / added). |
| [AlgorithmTriangulation.md](AlgorithmTriangulation.md) | Core algorithm cross-checked across 4 sources (this port, SparseyCore, the independent Sparsey_Alt Python, the V-to-mu jar) — "is our architecture correct?" Verdict + the pinned canonical CSA formula. |
| [Demos.md](Demos.md) | The instrumented demos, the cross-repo demo contract shared with `dcc-sph` and `dcc-htm`, and every load-bearing decision behind them. |
| [UserGuide.md](UserGuide.md) | Concepts, public API, and how to run the crate. |
| [Tuning.md](Tuning.md) | Parameters + tuning advice. |

See also the crate [`../README.md`](../README.md), [`../CLAUDE.md`](../CLAUDE.md), and the dcc-core node wrapper [`engine/src/nodes/ports/sparsey/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sparsey).
