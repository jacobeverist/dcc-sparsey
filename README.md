# dcc_sparsey — Sparsey in Rust

A clean-room Rust reimplementation of the **Sparsey** sparse-distributed-memory architecture ([Gerard Rinkus](http://sparsey.com/)), ported from the Java reference [`SparseyCore`](https://github.com/jacobeverist/SparseyCore) at commit `a0d4d34`.

Sparsey is a hierarchical **sparse-distributed-memory (SDM)** system modeled after neocortical macrocolumns. A network is a DAG of *regions*; each internal region holds a grid of *macrocolumns* (MACs); each MAC holds `Q` *competitive modules* (CMs); each CM holds `K` binary cells competing winner-take-all. A MAC's active **code** is one winning cell per CM (a one-hot-grouped sparse pattern of `Q·K` bits). Associations between codes are learned by temporal-Hebbian weight increases; recognition uses a data-driven *backoff* match and reports a scalar **familiarity `G` ∈ [0, 1]** (1.0 = perfectly familiar, ~0 = novel).

This is a standalone, self-contained crate. It was extracted from the `dcc-core` workspace and now builds and versions independently, with **no dependency on dcc-core**.

---

## Status

**Core mechanics implemented and tested.** The arena object model, the weight system, winner-take-all selection, data-driven *backoff* matching, DAG wiring, and learn/recognize frames with save/load.

**Partial:** `Recall` (sequence regeneration from a cue). The bottom-up retrieval pass is in place; full replay-to-input behavior is a documented gap — see the note in [`src/net/frame.rs`](src/net/frame.rs).

**Deferred:** SparseyCore's hand-rolled `DescriptorFile` NDF parser, replaced here by serde config structs.

---

## Features

- **Single-pass learning** — a code is committed per frame; no gradient descent, no epochs
- **Built-in novelty signal** — every frame reports familiarity `G ∈ [0, 1]`, no separate detector
- **Hierarchical** — regions form an arbitrary DAG, not just a chain
- **Fully reproducible** — one `u64` seed determines connectivity and every tie-break
- **Serialization** — save/load learned weights; rebuild structure from config + seed
- **No global mutable state** — the Java statics became fields on `SparseyNet`

---

## Quick start

Requires a Rust toolchain (stable).

```bash
git clone https://github.com/jacobeverist/dcc-sparsey
cd dcc-sparsey
cargo build            # build the library
cargo test             # unit + integration tests
cargo run --example learn_and_recognize
```

To use it from another crate:

```toml
[dependencies]
dcc_sparsey = { git = "https://github.com/jacobeverist/dcc-sparsey" }
```

---

## Concepts in 30 seconds

| Term | Meaning |
| --- | --- |
| **Region** | A node in the network DAG. *Input* regions present raw sparse features; *internal* regions hold a 2-D grid of MACs. |
| **MAC** (macrocolumn) | A group of `Q` competitive modules. Emits one **code** per frame. |
| **CM** (competitive module) | A winner-take-all group of `K` cells; exactly one wins. |
| **Code** | A MAC's output: the winning cell index (`0..K`) in each of its `Q` CMs — a sparse, one-hot-grouped pattern. |
| **`G`** | Familiarity of the current input, in `[0, 1]`. High = recognized, low = novel. Sparsey's built-in novelty signal. |
| **Modes** | `Learning` (updates weights), `Recognition` (read-only match), `Recall` (sequence regeneration — experimental, see Status). |

Full vocabulary: [`doc/NameReference.md`](doc/NameReference.md).

---

## Minimal example

```rust
use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::{NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

// A 3×3 input grid feeding one internal region with a single MAC (Q=4 CMs × K=6 cells).
let cfg = NetworkConfigBuilder::default()
    .region(RegionConfigBuilder::new("input", 0).grid(3, 3).build())
    .region(
        RegionConfigBuilder::new("l1", 1)
            .grid(1, 1)
            .qk(4, 6)
            .persistence(1)
            .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
            .build(),
    )
    .connect("input", "l1")
    .build();

let mut net = SparseyNet::build(cfg, 42).unwrap(); // 42 = RNG seed (fully reproducible)
let input = net.region_id("input").unwrap();

// Learning: present active input-cell indices, then run a learning frame.
net.set_input(input, &[0, 4, 8]).unwrap();
net.do_frame_learn();

// Lock in the weights and reset per-run state, then recognize (read-only).
net.finalize_learning();
net.prepare_for_new_run(false);
net.set_input(input, &[0, 4, 8]).unwrap();
net.do_frame_recognize();
```

A MAC clears its code at end-of-frame, so to read the code/`G` a frame produced, pass a
[`Recorder`](src/recorder.rs) to the `_rec` variants (`do_frame_learn_rec` / `do_frame_recognize_rec`)
— its `on_code_selected(region, mac, code, g, frame)` callback fires as each MAC commits. All four
demos use this pattern.

---

## Demos

Runnable, self-contained examples in [`examples/`](examples/). Each prints an annotated walkthrough:

| Demo | Command | Shows |
| --- | --- | --- |
| **Learn & recognize** | `cargo run --example learn_and_recognize` | Teach patterns, replay them → same code + `G≈1.0`; a novel input scores lower. The "hello world". |
| **Novelty detection** | `cargo run --example novelty_detection` | Stream known vs. unseen inputs; the single `G` scalar separates memorized from novel, and grades partial overlap. |
| **Hierarchy** | `cargo run --example hierarchy` | A 3-region DAG (input → l1 → l2); read the sparse code each region forms per frame. |
| **Save & load** | `cargo run --example save_and_load` | Serialize learned state to disk, reload into a fresh network, recognize identically. |

---

## Public API at a glance

- **Build:** `NetworkConfigBuilder` + `RegionConfigBuilder` → `NetworkConfig`; `SparseyNet::build(config, seed)`. Config also round-trips as JSON via `NetworkConfig::to_json` / `from_json`.
- **Drive a frame:** `set_input(region, &active_cells)` then `do_frame_learn` / `do_frame_recognize` / `do_frame_recall` (or the `_rec` variants taking a `&mut dyn Recorder`).
- **Read output:** `mac_code(mac)`, or a `Recorder`'s `on_code_selected` callback.
- **Lifecycle:** `finalize_learning()`, `prepare_for_new_run(erase_weights)`.
- **Persist:** `serialize_state()` / `load_state(bytes)` (weights only — rebuild structure from the same config + seed).

---

## Documentation

| Document | Description |
|---|---|
| [`doc/UserGuide.md`](doc/UserGuide.md) | Full user guide: concepts, API, worked examples |
| [`doc/Architecture.md`](doc/Architecture.md) | Object model, region/MAC/CM structure, frame lifecycle |
| [`doc/Tuning.md`](doc/Tuning.md) | Parameter descriptions and tuning advice |
| [`doc/NameReference.md`](doc/NameReference.md) | Abbreviation glossary for reading the source |
| [`doc/PortNotes.md`](doc/PortNotes.md) | Java class/type mapping and deliberate divergences |
| [`doc/MethodFidelity.md`](doc/MethodFidelity.md) | Method-by-method correspondence with SparseyCore |
| [`doc/Divergences.md`](doc/Divergences.md) | Where this port intentionally differs, and why |
| [`doc/AlgorithmTriangulation.md`](doc/AlgorithmTriangulation.md) | How ambiguous points in the reference were resolved |

---

## Source layout

```
src/
  lib.rs         — crate root and public re-exports
  net/           — SparseyNet: build.rs (wiring), frame.rs (learn/recognize/recall)
  region.rs      — regions (input and internal)
  mac.rs         — macrocolumns
  cm.rs          — competitive modules (winner-take-all groups)
  neuron.rs      — cells
  synapse.rs     — weights
  bundle.rs      — grouped afferent connections
  link.rs        — inter-region connectivity
  backoff.rs     — data-driven backoff match
  codetrie.rs    — code storage/lookup
  config.rs      — serde config structs and builders
  recorder.rs    — Recorder trait for observing per-frame code selection
  ids.rs         — typed arena indices (MacId, CmId, …)
examples/        — four runnable demos (see Demos)
benches/         — criterion benchmarks
fidelity/        — Java-parity harness (needs a SparseyCore checkout)
doc/             — documentation
```

**Arena / index object model.** Every region, MAC, CM, neuron, bundle and synapse lives in a flat `Vec` on `SparseyNet` and is referenced by a typed index (`MacId`, `CmId`, …) — no `Rc<RefCell<…>>`, no reference cycles. This mirrors the Java code's habit of "linear" arrays plus index fields.

---

## Fidelity

The port targets behavioral parity with the Java reference, not bit-exactness — SparseyCore's RNG is not reproduced, so identical seeds do not imply identical codes. What is pinned is the *behavior*: distinct inputs get distinct codes, learned inputs reactivate their code with `G ≈ 1.0`, and novel inputs score lower.

`cargo test` runs the unit, integration and behavioral suites everywhere. The Java comparison suites are marked `#[ignore]` and need an out-of-tree SparseyCore checkout, so a checkout with nothing beside it is still green. See [`fidelity/README.md`](fidelity/README.md) and [`doc/MethodFidelity.md`](doc/MethodFidelity.md).

---

## Optional features

- `json-schema` — derives [`schemars`](https://docs.rs/schemars) JSON Schema for the config types (`cargo build --features json-schema`).

---

## License and attribution

**AGPL-3.0-or-later.** Full text in [`LICENSE`](LICENSE).

Copyright (c) 2026 Jacob Everist.

The Sparsey architecture is the work of **[Gerard Rinkus](http://sparsey.com/)**. This port exists by his permission: the private SparseyCore repository was sent to him by email on 17 July 2026, and he granted verbal permission on 20 July 2026 to port Sparsey to Rust and release it under an open source license. AGPL-3.0-or-later is OSI-approved, so the chosen license is within the granted scope.

The permission record, upstream credit and the pinned reference commit are in [`PROVENANCE.md`](PROVENANCE.md).
