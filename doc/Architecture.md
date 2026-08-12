# Sparsey — Architecture (Rust implementation)

How the `sparsey` crate is built in Rust. This describes **our implementation**; the upstream algorithm spec it follows is [SparseyCore ARCHITECTURE.md](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md) (verbatim SparseyCore). Terms: [NameReference.md](NameReference.md); upstream → Rust mapping: [PortNotes.md](PortNotes.md); parameters: [Tuning.md](Tuning.md).

## The model in one paragraph

A network is a **DAG of regions**. Leaf (input) regions present raw feature cells; internal regions hold a `width_in_blks × height_in_blks` grid of **macrocolumns (MACs)**. Each MAC has `Q` **competitive modules (CMs)**; each CM has `K` binary **cells** in winner-take-all competition. A MAC's **code** is one winning cell per CM — a one-hot-grouped `Q·K` sparse pattern (SDC). Inter-region connections carry `U` (up/feedforward), `H` (horizontal/lateral), or `D` (down/feedback) signals, the type inferred from the regions' relative **DAG height**. Weights are never stored explicitly — each synapse keeps a `stiffness` + a last-coincidence `timestamp`, and its effective value is a table lookup on `(stiffness, age)`.

## Object model — flat arena + typed indices

The load-bearing design decision (`lib.rs:22-26`): to avoid the Java version's cyclic, back-pointer-laden object graph, **every entity lives in a flat arena on `SparseyNet` and is referenced by a typed-index newtype**. There is **no `Rc`/`Arc`/`RefCell`** and no back-pointer cycles.

`ids.rs` defines `#[repr(transparent)]` `usize` newtypes (`RegionId`, `MacId`, `CmId`, `NeuronId`, `EfferentBundleId`, `LinkId`, …) via a `define_id!` macro — not interchangeable, so index mix-ups are a compile error. An id is literally the index into its matching `Vec` on the arena (`self.neurons[id.index()]`):

| Arena `Vec` on `SparseyNet` (`net.rs`) | Element | Indexed by |
|---|---|---|
| `regions` | `Region` | `RegionId` |
| `macs` | `Mac` | `MacId` |
| `cms` | `Cm` | `CmId` |
| `neurons` | `Neuron` (input + internal cells, one flat pool) | `NeuronId` |
| `links` | `Link` | `LinkId` |
| `efferent_bundles` | `EfferentBundle` (source-owned weight storage) | `EfferentBundleId` |
| `backoffs` | `BackoffStrategy` (parallel to `regions`) | `RegionId` |

Non-arena run state on `SparseyNet`: `config`, `weight_table`, `global_frame: i64` (the weight-age clock), `op_mode`, `region_by_name: HashMap`, `rng: Xoshiro256PlusPlus`.

> `ApertureId` / `SubBundleId` / `SynapseId` newtypes exist in `ids.rs` but have **no backing arena vec** — forward-looking scaffolding. Synapses live inline in each bundle's `Vec<Synapse>`, so `SynapseId` is currently vestigial.

## Containment & connectivity

Parent→child containment and child→parent references are all ids:

```
SparseyNet (arena)
 └─ Region.macs: Vec<MacId>        Region.cells: Vec<NeuronId>
     └─ Mac.cms: Vec<CmId>         Mac.owning_region: RegionId
         └─ Cm.neurons: Vec<NeuronId>   Cm.owning_mac / owning_region
             └─ Neuron  (owning_cm: Option<CmId>)
```

Connectivity is **source-owned** (weights live on the sending side):

```
Link { source_region, target_region, syn_type }
Neuron.efferent_bundles: Vec<EfferentBundleId>              ← how a cell reaches its outgoing weights
 └─ EfferentBundle { source_neuron, link, syn_type, synapses: Vec<Synapse>, frozen }
     └─ Synapse.target_neuron: NeuronId                     ← the one cross-region reference
```

There is **no afferent synapse list**: the receiving side is value-only accumulators `Neuron.aff: [AfferentAccum; 3]` (one per `H`/`U`/`D`), written during the push. `wire_connectivity` (`net.rs`) creates, per (source cell, link), one bundle with a `Synapse` to **every** target cell (full within-link connectivity).

## Module map (`src/`)

| File | Role |
|---|---|
| `net.rs` | **The heart** — the arena, the builder, and all per-frame compute (~800 lines). |
| `config.rs` | serde config tree (`NetworkConfig`/`RegionConfig` + builders) — replaces Java's NDF descriptor file. |
| `synapse.rs` | `Synapse` (implicit-weight) + `WeightTable` decay step-function. |
| `bundle.rs` | `EfferentBundle` — source-owned synapse list + freeze/saturation. |
| `neuron.rs` / `mac.rs` / `cm.rs` / `region.rs` | Entity records (data only; processing lives in `net.rs`). |
| `link.rs` | Inter-region link + `syn_type_from_heights` (H/U/D inference). |
| `codetrie.rs` | Per-MAC learned-code store — a `HashMap<Vec<u32>, i64>`, not a real trie. |
| `backoff.rs` | Recognition-time version selection (the backoff strategy). |
| `ids.rs` | Typed arena-index newtypes. |
| `types.rs` | `SynapseType {H=0,U=1,D=2}`, `OperationMode {Learning,Recognition,Recall}`. |
| `recorder.rs` | `Recorder` stats-trait seam (`NullRecorder`). |
| `error.rs` | `SparseyError` / `SparseyResult` (crate-local, no dcc-core dep). |
| `lib.rs` | Crate docs + public re-exports. |

## Key structs (load-bearing fields)

- **`SparseyNet`** — the 7 arena vecs + `config`, `weight_table`, `global_frame`, `op_mode`, `region_by_name`, `rng`.
- **`Mac`** — `cms`, `active`, **`code_age`** (persistence countdown; 0 = eligible), **`g`** (global match), `learned_codes: CodeTrie`.
- **`Cm`** — `neurons`, **`winner: Option<NeuronId>`**, `prev_winner`, `v_max`, `v_ave`, `tied_max_count`.
- **`Neuron`** — `active`, `prev_active`, **`v: f32`** (evidence), **`aff: [AfferentAccum; 3]`** (`raw_sum`, `active_input_count`, `normalized`, `adjusted`), `efferent_bundles`.
- **`Synapse`** — `target_neuron`, **`stiffness: u8`** (0 = malleable, `max_stiffness` = permanent), **`timestamp_last_pre_post: i64`** (`VIRGIN_TIMESTAMP = i64::MAX`).
- **`WeightTable`** — `table[stiffness][age] → weight`, built from `WeightTableConfig` (default caps at `MAX_WEIGHT = 127`).

## The frame loop

Public drivers (`net.rs`) each delegate to a `_rec` variant taking a `&mut dyn Recorder` for stats:

- **`do_frame_learn`** → `op_mode = Learning` → `process_regions(recognition=false)` → `update_weights()` → `freeze_saturated_bundles()` → `end_frame()`.
- **`do_frame_recognize`** → `op_mode = Recognition` → `process_regions(true)` → `end_frame()` (no weight change).
- **`do_frame_recall`** → currently mirrors recognize; downward D-replay to the input region is deferred.

`process_regions`:
1. Reset internal-cell `v` and all `aff` accumulators.
2. Process regions **ascending by `height_in_dag`** (bottom-up); skip non-internal.
3. Per region: `push_into_region` then `process_region_macs`.

`push_into_region` — a **two-phase push** to avoid aliasing on the single `neurons` vec: phase 1 reads each afferent link's active source cells (`prev_active` for H/D per `use_previous_active`, else `active`) and accumulates `synapse.effective_value(global_frame, weight_table)` into a local map bucketed by `syn_type`; phase 2 applies the accumulated `raw_sum` + `active_input_count` onto the target neurons.

`process_region_macs` — per MAC:
1. Eligibility — has a `U` signal, or is `persisting` (`code_age > 0`); else deactivate. A persisting MAC holds its code and is skipped.
2. `normalize_mac` — per cell/type: `normalized = raw / (active_input_count · MAX_WEIGHT)`, clamp to `[min_cutoff, max_cutoff]`, `adjusted = normalized^exp`.
3. G — recognition **with** backoff rules → `recognize_version`; otherwise `set_v` (per-cell V = product of `adjusted` over active types) + `compute_g` (G = mean over CMs of each CM's `v_max`).
4. `select_winners` — **max-V per CM, ties broken uniformly via the seeded `rng`**; set winner active + `cm.winner`.
5. Store `g`, reset `code_age = persistence`, read `mac_code` (one-hot indices per CM), insert into `learned_codes`.

`end_frame` — decrement `code_age` + deactivate expired MACs; roll `prev_active ← active` (clear `active`) and `cm.prev_winner ← winner` (clear `winner`); `global_frame += 1`.

## Learning mechanics

Weights are implicit — decay is free via table lookup:

- **Effective weight**: `Synapse::effective_value(frame, wt)` = `MAX_WEIGHT` if permanent, else `wt.weight(stiffness, frame − timestamp)`. A virgin synapse (`timestamp = i64::MAX`) yields weight 0.
- **Pre-post coincidence** (`update_weights`, learn only): for each non-frozen bundle whose source cell is active, every synapse whose `target_neuron` is active calls `record_pre_post(global_frame)`.
- **Stiffness promotion** (`record_pre_post`): a coincidence arriving while the previous increase is still "fresh" (`1 ≤ age < fresh_index(stiffness)`) bumps `stiffness`; repeated close coincidences walk `0 → 1 → 2 (permanent)`.
- **Transient → permanent** (`promote_to_permanent`): triggered by bundle freezing and by the end-of-learning `finalize_learning()` sweep.
- **Bundle freezing / saturation**: a bundle whose `increased_fraction` exceeds the target region's `saturation_threshold` freezes — promotes all its synapses and stops learning (the critical-period mechanism).

## Config & build

serde tree (`config.rs`), JSON round-tripping (`from_json`/`to_json`), most fields `#[serde(default)]`: `NetworkConfig { regions, connections, weight_table }` → `RegionConfig { grid, q, k, persistence, signal_{h,u,d}: SignalParams, backoff, sigmoid }` → `ConnectionSpec { source, target }` (by name). Builders: `NetworkConfig::builder()` / `RegionConfigBuilder`; `BackoffConfig::canonical()` builds the `HUD → HU/UD → U` chain.

**Build path** (no `from_config`): `SparseyNet::build(config, seed)` → `build_regions` (regions/macs/cms/neurons + per-region backoff) → `build_links` (derives `syn_type` from DAG heights; `use_previous_active = syn_type != U`) → `wire_connectivity` (full connectivity). State persistence: `serialize_state` / `load_state` (bincode of per-synapse `(stiffness, timestamp)` + `global_frame`; structure is rebuilt from config).

## Public API

```rust
let mut net = SparseyNet::build(config, seed)?;   // config: NetworkConfig
let rid = net.region_id("input").unwrap();
net.set_input(rid, &active_indices)?;             // &[u32]
net.do_frame_learn();                             // or _recognize() / _recall()  (+ _rec variants)
let code = net.mac_code(mac_id);                  // Option<Vec<u32>>
net.finalize_learning();                          // promote transient → permanent
```

> **Persistence gotcha:** with `persistence = 1` a MAC's code is cleared at end-of-frame — read it **during** the frame via a `Recorder`.

## Divergences from the Java upstream (M1)

Documented in [PortNotes.md](PortNotes.md); visible in code:

- **`CodeTrie` is a `HashMap`** (a true trie can drop in later without changing callers).
- **Full connectivity** — band-limited projective fields (`band_thickness`/`band_rates`) are parsed but unused.
- **Winner selection is max-V** (max-likelihood + RNG tie-break) in *both* learn and recognize; the probabilistic sigmoid CSA (`SigmoidConfig`, `mu`/`rho`) and MCH discounting are deferred (fields exist, unimplemented).
- **Recall D-replay** to the input region is deferred (recall mirrors recognize).
- **One flat synapse list per bundle** — per-target sub-bundle granularity collapsed.
- **No global mutable state** — Java statics `globalFrameAcrossAllEpisodes` / `operationalMode` are `SparseyNet` fields (`global_frame`, `op_mode`); the frame is passed into weight lookups.
