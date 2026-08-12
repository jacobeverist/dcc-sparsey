# SparseyCore (Java) → `sparsey` (Rust) port map

This crate is a **clean-room reimplementation** of the Sparsey algorithm, ported from the Java reference [`SparseyCore`](https://github.com/jacobeverist/SparseyCore) @ `a0d4d34`, following that repo's `ARCHITECTURE.md`. It is **not** a line-by-line translation. This document records the class/type mapping and every deliberate divergence, so behavior can be cross-checked against the Java.

## Naming conventions

- Java classes → Rust structs in `snake_case.rs` modules, `PascalCase` type names kept where sensible (`Mac`, `CM` → `Cm`, `Synapse`).
- Java `camelCase` fields/methods → Rust `snake_case` (`owningMAC` → dropped, see below; `computeNormalizedInfluences` → `compute_normalized_influences`).
- Java constants (`SYNAPSE_TYPE_U`) → Rust enums (`SynapseType::U`) or associated consts.

## Object model — the big divergence

The Java graph is cyclic and full of parent back-pointers (`GenericNeuron.owningRegion`, `CM.owningMAC/owningRegion/owningNet`, `Block.owningNet/owningRegion`, `Bundle.owningNeuron`, `CodeTrie.owningMac`, `Link.sourceRegion/targetRegion`) plus one hard cross-region object pointer, `Synapse.targetNeuron`.

Rust model: a single owning **`SparseyNet`** holds flat `Vec`s (`regions`, `macs`, `cms`, `neurons`, `apertures`, `efferent_bundles`, `sub_bundles`, `synapses`, `links`). Every reference — including every `owning*` back-pointer and `Synapse.targetNeuron` — becomes a typed index newtype (`RegionId`, `MacId`, `CmId`, `NeuronId`, `SynapseId`, …; see `src/ids.rs`). No `Rc`/`RefCell`. The Java **push-signal** pattern is preserved: the efferent side writes into a target neuron's afferent accumulators *by index* (the afferent side never holds source pointers, so this is the only cross-region interaction).

| Java | Rust |
|------|------|
| `Network` (top-level DAG + drivers + static globals) | `net::SparseyNet` (arena + run context; globals become fields) |
| `GenericRegion` / `InputRegion` / `InternalRegion` | `region::Region` + `region::RegionKind::{Input, Internal}` |
| `Node` (base of Mac/Aperture) | fields folded into `mac::Mac` / `aperture` records (no separate base type) |
| `Mac` | `mac::Mac` |
| `CM` | `cm::Cm` |
| `GenericNeuron` / `InputRegionNeuron` / `InternalRegionNeuron` | `neuron::Neuron` (+ kind) |
| `Bundle` / `AfferentBundle` / `EfferentBundle` / `SubEfferentBundle` / `UpperBundle` | `bundle::{EfferentBundle, SubBundle, AfferentAccum, UpperBundle}` |
| `Link` | `link::Link` |
| `Synapse` (+ static `WT_TABLE`) | `synapse::Synapse` (+ `synapse::WeightTable`) |
| `BackoffStrategy` / `BackOffSpec` | `backoff::BackoffStrategy` |
| `CodeTrie` | `codetrie::CodeTrie` (index/arena nodes) |
| `TessellateRectangle` | `connectivity` (geometry helpers) |
| `DescriptorFile` (NDF parser) | **dropped** → `config` serde structs |
| `EpisodeContainer` | **deferred** (dataset driver, Phase 3) |
| `TraceStat`/`CodeStat`/`RecordMoments`/`*Stats`/`BufferedImageIO`/`MovieIO` | **deferred** → `recorder::Recorder` no-op trait seam |

## Type mappings & deliberate divergences

- **Numerics.** Parity is *not* bit-exact. Neuron `V` and bundle sums are `f32` (as in Java). Some Java `double` thresholds/cutoffs are narrowed to `f32` — an accepted divergence. `WT_TABLE` values (≤127) are `u8`. Synapse type / operation mode / stiffness become small enums or `u8` rather than signed Java `byte`.
- **Config replaces the NDF.** `DescriptorFile`'s hand-rolled parser is not ported. `NetworkConfig`/`RegionConfig` map the same parameters (`Region_Q`→`q`, `Region_K`→`k`, `RegionWidthInBlks`→`width_in_blks`, `RegionHeights`→`height_in_dag`, `RegionPersistence`→`persistence`, `U_NEI_Exp`/`U_EI_Exp`, `*_min/max_cutoff`, `*_saturation_threshold`, `*_MCH_*`, `*_Connectivity_Band_*`) and derive/serialize as JSON.
- **No global mutable state.** `Network.globalFrameAcrossAllEpisodes` and `Network.operationalMode` (Java `static`) → fields on `SparseyNet`; the weight lookup takes the current frame as a parameter.
- **Connection type inference.** Kept exactly: a connection `src → tgt` is `U` if `src.height < tgt.height`, `H` if equal, `D` if greater (`Network.createInterRegionLinks`).
- **Backoff.** Kept data-driven: `matchingRules[priority][case][synType]` + per-case thresholds (`BackoffStrategy`).

## Deferred / not yet ported

- Recall mode: a `do_frame_recall_rec` entry point + `OperationMode::Recall` exist and drive the retrieval pass, but faithful **D-signal regeneration of the input region's features** (the Java `doFrameRecall` "replay to L0" behavior, `numRecallCueFrames`) is a follow-on. For nets without downward (D) links, recall behaves like recognition.
- Learned-state serialization (`SparseyNet::serialize_state`/`load_state`) captures per-synapse stiffness + last-pre-post timestamp (weights), not the full transient-count bookkeeping — sufficient to reload a trained net for recognition/recall.
- Dataset/IDF reader and the per-episode driver loop (the driver owns the loop; the crate exposes per-frame primitives).
- All statistics, tracing, and image/movie output.
- Any binary `.dat` trained-network format compatibility.
