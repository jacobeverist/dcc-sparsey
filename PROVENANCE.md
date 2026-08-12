# Provenance and attribution

## What this crate is

`dcc_sparsey` is a clean-room Rust **reimplementation** of the **Sparsey** sparse-distributed-memory architecture. It follows the algorithm as documented in the reference implementation's `ARCHITECTURE.md` rather than translating classes line by line — see [`doc/PortNotes.md`](doc/PortNotes.md) for the type-by-type mapping and [`doc/Divergences.md`](doc/Divergences.md) for every deliberate difference.

## Attribution

**Sparsey is the work of [Gerard Rinkus](http://sparsey.com/).** The architecture — macrocolumns of winner-take-all competitive modules, the sparse distributed codes they emit, and the critical-period learning that forms associations between them — is his, and this crate is an implementation of it, not an invention of it.

| | |
|---|---|
| **Architecture** | Sparsey, by [Gerard Rinkus](http://sparsey.com/) |
| **Reference implementation** | [SparseyCore](https://github.com/jacobeverist/SparseyCore) (Java), at commit `a0d4d34faf1e3b6cf532c3846446c941ff80eff5` |
| **This crate** | Rust reimplementation, © 2026 Jacob Everist |

## Permission

This port exists and is released with the original author's permission:

> The private SparseyCore repository was sent to Gerard Rinkus by email on **17 July 2026**.
> Gerard Rinkus granted **verbal permission on 20 July 2026** to port Sparsey to Rust and
> release it under an open source license.

`AGPL-3.0-or-later` is an OSI-approved open source license, so the license chosen here falls within the scope of that grant.

> **Note for whoever maintains this next.** That grant is *verbal*, which is the weakest kind of record. A one-line email to Gerard confirming the 20 July conversation — and pointing at this file — would durably back it, and costs nothing. Worth doing before this repository is ever made public.

## License

This crate is licensed under **AGPL-3.0-or-later**; the full text is in [`LICENSE`](LICENSE).

Copyright (c) 2026 Jacob Everist.

## Relationship to dcc-core

Extracted from the [dcc-core](https://github.com/jacobeverist/dcc-core) workspace, where it is wrapped as a `SparseyNetworkNode` so a Sparsey network can be compared head-to-head against HTM and SPH architectures in one `Network`. This crate itself knows nothing about dcc-core and must never depend on it — the wrapper is a boundary concern that lives on the dcc-core side.

Its terminology is mapped onto dcc-core's canonical names in the [nomenclature crosswalk](https://github.com/jacobeverist/dcc-core/blob/main/docs/canonical/vocabulary/nomenclature-crosswalk.md).

## Verifying fidelity against the Java

`fidelity/` holds the cross-language harness: a Java driver that emits golden vectors from SparseyCore, and a Python script that generates an independent reference from the papers. Neither is required to build or test this crate — the committed fixtures under `tests/fixtures/` are what the fidelity tests read, and a missing upstream checkout makes those tests skip rather than fail. See [`fidelity/README.md`](fidelity/README.md).
