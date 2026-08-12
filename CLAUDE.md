# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dcc_sparsey** is a **clean-room Rust reimplementation** of the Sparsey hierarchical sparse-distributed-memory (SDM) network — a DAG of cortical-macrocolumn regions that learn associations between sparse one-hot-grouped codes via temporal-Hebbian weight increases. Ported (algorithm-faithful, **not** line-by-line) from the Java reference `SparseyCore` @ commit `a0d4d34`, following its `ARCHITECTURE.md`. See `doc/PortNotes.md` (class/type mapping + deliberate divergences) and `doc/NameReference.md` (abbreviation glossary).

**dcc-agnostic, and this is a hard rule:** this crate knows nothing about `dcc-core` (`Node`, `BitField`, `Network`), and must not learn. dcc-core wraps a whole network as one `SparseyNetworkNode` on *its* side of the boundary ([`engine/src/nodes/ports/sparsey/`](https://github.com/jacobeverist/dcc-core/tree/main/engine/src/nodes/ports/sparsey)); the pattern is described in [`docs/claude/third-party-import-pattern.md`](https://github.com/jacobeverist/dcc-core/blob/main/docs/claude/third-party-import-pattern.md). Don't add a dcc-core dependency here.

Attribution and the permission this port rests on: [`PROVENANCE.md`](PROVENANCE.md).

## Development Commands

```bash
cargo build  -p dcc_sparsey
cargo test   -p dcc_sparsey     # tests/learn_recognize.rs (4) + inline unit tests (synapse 7, backoff 3, config 2, ids 2)
cargo clippy -p dcc_sparsey
```

One cargo feature, `json-schema` (off by default), deriving `schemars::JsonSchema` on the config types. Deps: `rand`, `rand_xoshiro`, `thiserror`, `serde`, `serde_json`, `bincode = "1.3"`; dev-deps `approx`, `criterion`.

`examples/` holds four runnable demos — `learn_and_recognize`, `novelty_detection`, `save_and_load`, `hierarchy`. The cross-crate comparison against HTM stays in dcc-core (`engine/examples/sparsey_vs_htm_sequence.rs`) because it needs both crates.

**wasm32 note:** this crate does not select a `getrandom` backend. `rand 0.8` pulls `getrandom 0.2`, which has no `wasm32-unknown-unknown` backend unless something enables its `js` feature — and choosing a randomness backend is a *binary's* decision, not a library's. A wasm consumer must enable it themselves (dcc-core does, in `bindings/wasm/Cargo.toml`).

## Domain model & vocabulary

A network is a **DAG of regions**. `doc/NameReference.md` is the authoritative glossary; the essentials:

- **Region** (`RegionKind::{Input, Internal}`) — DAG node. Input = leaf (height 0, raw feature cells); Internal = a `width_in_blks × height_in_blks` grid of MACs.
- **MAC** (macrocolumn) — holds `Q` competitive modules; owns `learned_codes: CodeTrie`.
- **CM** (competitive module / minicolumn) — holds `K` cells competing **winner-take-all**.
- **Cell/neuron** — binary unit with `v: f32` evidence; per-signal afferent accumulators `aff: [_; 3]` (H/U/D).
- **Code** (`Vec<u32>`) — one winning cell index per CM = a one-hot-grouped `Q·K` sparse pattern. Read via `SparseyNet::mac_code(MacId)`.
- **Q / K** — CMs per MAC / cells per CM. Set via `RegionConfigBuilder::qk(q, k)`.
- **Signals** `SynapseType{H, U, D}` (`#[repr(u8)]` H=0, U=1, D=2) — U = up/feedforward, H = horizontal/lateral, D = down/feedback. Inferred at build from relative DAG height (U if src<tgt, H if ==, D if >).
- **V** = per-cell local evidence (product of adjusted normalized H/U/D sums); **G** = mean of per-CM max-V across a MAC.

## Source layout (`src/`)

Arena object model — every entity lives in a flat `Vec` on `SparseyNet`, referenced by **typed newtype IDs** (`ids.rs`); no `Rc<RefCell>`, no back-pointers, no cycles.

| File | Role |
|---|---|
| `net.rs` | **Core** — arena + per-frame drivers (the entry point) |
| `config.rs` | serde `NetworkConfig` / builders (replaces Java's NDF descriptor file) |
| `synapse.rs` | `Synapse` + implicit-decay `WeightTable` |
| `backoff.rs` | Data-driven recognition backoff strategy |
| `neuron.rs` / `mac.rs` / `cm.rs` / `region.rs` | Entity records |
| `bundle.rs` | Source-owned `EfferentBundle` weight storage |
| `codetrie.rs` | Per-MAC learned-code store (a `HashMap`, not a real trie) |
| `link.rs` | Inter-region link + signal-type inference |
| `ids.rs` | Typed arena index newtypes |
| `types.rs` | `SynapseType`, `OperationMode` enums |
| `error.rs` | `SparseyError` / `SparseyResult` (thiserror) |
| `recorder.rs` | `Recorder` stats-trait seam (`NullRecorder`) |
| `lib.rs` | Crate docs + public re-exports |

## Public API (`net.rs`, re-exported from `lib.rs`)

```rust
let mut net = SparseyNet::build(config, seed)?;   // config: NetworkConfig
let rid = net.region_id("input").unwrap();
net.set_input(rid, &active_indices)?;             // active: &[u32]
net.do_frame_learn();                             // or _recognize() / _recall()
let code = net.mac_code(mac_id);                  // Option<Vec<u32>>
net.finalize_learning();                          // promote transient → permanent synapses
```

- Build config via `NetworkConfig::builder()` → `.region(RegionConfigBuilder::new(name, height).grid().qk().persistence().backoff())` → `.connect(src, tgt)`. JSON round-trips (`from_json` / `to_json`).
- **Three modes** (`OperationMode::{Learning, Recognition, Recall}`): `do_frame_learn` (updates weights), `do_frame_recognize` (no weight change, runs backoff), `do_frame_recall` (M1: like recognition, D-replay deferred). Each has a `_rec(&mut dyn Recorder)` variant for stats capture.
- Lifecycle helpers: `prepare_for_new_run(erase_weights)`, `serialize_state()` / `load_state()` (bincode of per-synapse `(stiffness, timestamp)` + `global_frame`; structure is rebuilt from config).

## Learning mechanics

Weights are **never stored explicitly**: each `Synapse` keeps `stiffness: u8` + `timestamp_last_pre_post: i64`, and its effective weight is a `(stiffness, age)` step-function lookup in `WeightTable` (age = current_frame − timestamp), so decay is free. A learning frame: two-phase signal push (read-only accumulate → apply, to dodge aliasing on the single `neurons` Vec) → normalize (`raw / (count · MAX_WEIGHT=127)`, cutoffs, exponent) → per-cell V → per-MAC G → `select_winners` picks the max-V cell per CM (ties broken by seeded `Xoshiro256PlusPlus`). `update_weights` records pre-post coincidences on efferent synapses; fresh-window coincidences promote stiffness (repeated → permanent). Bundles freeze once their increased-synapse fraction exceeds the target region's `saturation_threshold` (critical period). **Recognition** instead runs the data-driven backoff: walk priority levels high→low, take the first version whose signal types are all available and that clears its threshold.

## Gotchas

- **Persistence timing:** with `persistence = 1`, a MAC's code is cleared at end-of-frame, so read it **during** the frame via a `Recorder` (see `tests/learn_recognize.rs`).
- **`ids.rs` over-declares:** `ApertureId` / `SubBundleId` / `SynapseId` exist with doc-comments referencing `SparseyNet` fields that **don't exist** yet — forward-looking scaffolding. `net.rs` uses only `RegionId/MacId/CmId/NeuronId/EfferentBundleId/LinkId`.
- **serde config, not NDF:** `NetworkConfig`/`RegionConfig` replace Java's `DescriptorFile`; most fields `#[serde(default)]`. Name mapping in `config.rs` header + `PortNotes.md`.
- **f64→f32 narrowing is deliberate and not bit-exact** with Java; weights are `u8` (≤ `MAX_WEIGHT=127`).
- **M1 divergences from Java** (documented in `PortNotes.md`): winner selection is max-likelihood (max V), **not** the probabilistic sigmoid CSA; connectivity is full-within-link (band-limited PF deferred); `CodeTrie` is a `HashMap`; one flat synapse list per bundle (no per-target sub-bundles); Recall D-replay-to-L0 deferred.
- **No global mutable state:** Java statics `globalFrameAcrossAllEpisodes` / `operationalMode` are now `SparseyNet` fields (`global_frame`, `op_mode`); weight lookup takes the frame as a parameter.
