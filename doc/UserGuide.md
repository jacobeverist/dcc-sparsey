# Sparsey — User Guide

Concepts, public API, and how to run the `sparsey` crate. Companion docs: [NameReference.md](NameReference.md) (terminology), [PortNotes.md](PortNotes.md) (upstream port map), [Tuning.md](Tuning.md) (parameters), and [Architecture.md](Architecture.md) (the Rust implementation architecture; upstream spec in [SparseyCore ARCHITECTURE.md](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md)).

> **Status:** scaffold. The authoritative usage material currently lives in [`../CLAUDE.md`](../CLAUDE.md) (public API `net.rs`, source layout, learning mechanics) and [`../README.md`](../README.md); this guide consolidates and expands it over time.

## Model in one paragraph

A Sparsey network is a **DAG of regions**. Leaf (input) regions present sparse features; internal regions hold a 2-D grid of **macrocolumns (MACs)**, each a set of `Q` **competitive modules (CMs)**, each a winner-take-all group of `K` **cells**. A MAC's **code** is one winning cell per CM (a one-hot-grouped `Q·K` SDC). Regions connect via `U`/`H`/`D` links; weights are stored implicitly via per-synapse timestamps + stiffness. See [NameReference.md](NameReference.md).

## Running (the `Network` loop)

1. Build a `NetworkConfig` (regions, `Q`/`K`, links, params) — see `config.rs` and [PortNotes.md](PortNotes.md).
2. `SparseyNet::from_config(...)`.
3. Per frame: set the input region's active features → `do_frame_learn` / `do_frame_recognize` / `do_frame_recall` (see [Modes in NameReference.md](NameReference.md)).
4. Read the top (max-DAG-height) region's code.

Worked examples: the integration tests under [`../tests/`](../tests/), and the dcc-core node wrapper [`engine/src/nodes/ports/sparsey/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sparsey) (how the crate plugs into a dcc `Network`).

## To expand

- [x] A minimal runnable example — four now exist under [`../examples/`](../examples/), tabulated in the README.
- [ ] Per-mode walkthrough (Learning / Recognition / Recall).
- [ ] Public-API reference table.
