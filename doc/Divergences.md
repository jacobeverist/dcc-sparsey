# Sparsey — Port Divergences

Architectural & structural differences between the upstream source and this dcc-core port — the **fidelity audit**: what we deliberately changed, and what we deferred or may have missed. Name/type mapping: [PortNotes.md](PortNotes.md); implementation: [Architecture.md](Architecture.md); method-by-method: [MethodFidelity.md](MethodFidelity.md); upstream spec: [SparseyCore ARCHITECTURE.md](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md).

**Upstream:** `SparseyCore` (Java) @ `a0d4d34`.

## Deliberate structural changes

- **Object model → flat arena + typed indices.** Java's back-pointer-laden object graph (owning-* references, `Synapse.targetNeuron` object refs) becomes flat `Vec` arenas on `SparseyNet` addressed by `#[repr(transparent)]` index newtypes — **no `Rc`/`RefCell`, no cycles**. The single biggest structural change (see [Architecture.md](Architecture.md)).
- **Config: serde tree, not NDF.** `NetworkConfig` / `RegionConfig` (serde, JSON round-trip) replace Java's `DescriptorFile` / NDF parser.
- **No global mutable state.** Java statics `globalFrameAcrossAllEpisodes` / `operationalMode` become `SparseyNet` fields (`global_frame`, `op_mode`); the frame is passed into weight lookups.
- **Stats seam.** Java's ~90 in-loop stat calls collapse to a `Recorder` trait (`NullRecorder` default).
- **Error handling.** Crate-local `SparseyError` / `SparseyResult` (thiserror); no dcc-core dependency (the adapter is a boundary concern for the block wrapper).
- **Numerics.** `f32` throughout (not bit-exact with Java `double`); some `double` thresholds narrowed to `f32`; weights are `u8` (≤ `MAX_WEIGHT = 127`); synapse-type / mode / stiffness are small enums / `u8`.

## Deliberate behavioral simplifications (M1)

- **Winner selection = max-V** (max-likelihood + seeded-RNG tie-break) by default in **both** learn and recognize. The probabilistic sigmoid CSA (`mu` / `rho`) is **now implemented as an opt-in mode** (`RegionConfigBuilder::enable_csa()`, `SigmoidConfig::enabled`): when on, *learning* samples winners from the V→μ→ρ distribution (recognition stays max-V). **CSA is now the default**; `RegionConfigBuilder::disable_csa()` forces max-V (M1). See [AlgorithmTriangulation.md](AlgorithmTriangulation.md).
- **MCH discounting** *(implemented 2026-07-05).* Per-CM `num_mch` (cells with `V ≥ v_thresh_hypothesis`) → per-MAC `num_mch` = round(mean), min 1; an internal source MAC's efferent signal is **ignored** when `num_mch ≥ MCH_ignore_thresh` or else **discounted** by `num_mch` / `1+num_mch^exp`. Input features contribute at full strength. Tests: `tests/mch.rs`.

## Deferred / not yet ported (gaps)

- **Band-limited projective-field connectivity.** *Implemented (2026-07-04).* `wire_connectivity` now honors `band_thickness` / `band_rates`: a source block connects to a target block only within a distance band (normalized grid distance vs cumulative outer radii), and each candidate synapse is created with that band's rate. Default (empty bands) stays full within-link connectivity. Divergence from Java: each candidate `(src cell, tgt cell)` is an independent `Bernoulli(rate)` draw rather than exactly `round(rate·K)` sources per target — same expected density/banding, and normalization divides by actual fan-in so coding is unaffected. Tests: `tests/connectivity.rs`.
- **Recall D-replay to L0.** `do_frame_recall` currently mirrors recognize; downward regeneration of input-region features is deferred.
- **Per-target sub-bundles.** One flat synapse list per bundle; per-target-block freezing granularity is collapsed (`SubEfferentBundle` / `UpperBundle` not built). Bundle-scale transient/permanent counts are available (`EfferentBundle::{transient,permanent}_count`, 2026-07-05); the multi-scale (neuron/CM/Mac/region) aggregation is Java stats-layer bookkeeping, handled here via the `Recorder` seam.
- **Dataset driver.** `EpisodeContainer` (IDF parser) deferred.

## Representation changes (behavior-equivalent, structure differs)

- **`CodeTrie` is a `HashMap<Vec<u32>, i64>`**, not a real trie — a trie can replace it later without changing callers.

## Scaffolding present but unused

- `ApertureId` / `SubBundleId` / `SynapseId` newtypes exist (`ids.rs`) with **no backing arena vec** — forward-looking; synapses live inline in each bundle.

## Fixed defects (from the method-level audit)

Surfaced by [MethodFidelity.md](MethodFidelity.md) and **fixed 2026-07-04** (with regression tests in `synapse.rs`):

- **Over-promotion of decayed synapses.** `Synapse::promote_to_permanent` now gates on `is_contributing_learning` (weight > 0), so `finalize_learning` / bundle `freeze` no longer consolidate decayed-to-zero synapses into permanent (matches Java's `getEffectiveValue > 0`). The previously-dead `reconcile_if_inert` is now wired into `update_weights`.
- **H-links no longer self-connect.** `wire_connectivity` now skips self and same-CM targets for horizontal links (matches Java `buildBlockMatrix`).
- **Promotion gate.** `record_pre_post` now promotes only when `age >= persistence` (was `age >= 1`), threading the source region's persistence through `update_weights` (matches Java; `persistence = 1` is unchanged).

## Open questions / to verify

- [ ] Numeric fidelity of the EI/NEI signal exponents, cutoffs, and `saturation_threshold` freezing vs Java (the `f32` narrowing is accepted — confirm no logic gaps).
- [ ] `WT_TABLE` default transition indexes / weights match the Java reference.
- [ ] Whether the deferred items (probabilistic CSA, MCH, band-limited PF, Recall D-replay) are needed for the intended head-to-head comparisons.

## Upstream spec for the deferred / simplified features

The detailed algorithm for the items above lives in the upstream **[SparseyCore `ARCHITECTURE.md`](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md)** (@ `a0d4d34`) — use it when implementing them:

| Deferred / simplified here | Upstream `ARCHITECTURE.md` section |
|---|---|
| Probabilistic CSA / sigmoid winner selection (we ship max-V) | §2 "V and G Computation" · §12 "Design Principles" |
| MCH discounting | §7 "Multiple Concurrent Hypotheses" |
| Projective-Field bands (we ship full connectivity) | §3 "Connectivity → Projective Field (PF) Bands" |
| Recall D-replay | §5 "Operation Modes → Recall" |
| `EpisodeContainer` dataset driver | §9 "Data Pipeline" |
| Backoff strategy · weight dynamics *(implemented — for cross-check)* | §6 · §4 |

