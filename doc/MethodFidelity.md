# Sparsey — Method-Level Fidelity

A method-by-method comparison of the Rust `sparsey` crate against upstream **SparseyCore** (Java) @ `a0d4d34`, so we know exactly how our algorithms are structured versus the original. Companion to [Divergences.md](Divergences.md) (the higher-level audit) and [Architecture.md](Architecture.md) (the Rust design). Produced by reading both sources method-by-method.

**Legend:** **FAITHFUL** (same algorithm) · **DIVERGENT** (ported but differs) · **MISSING** (no Rust equivalent) · **RUST-ONLY** (no Java counterpart).

> Java's per-frame compute lives mostly in `Mac.java` / `CM.java`; in Rust it collapses into `net.rs` (the arena owns the data, so `mac.rs`/`cm.rs`/`neuron.rs` are pure records). Trivial getters/setters/`toString`/debug IO are omitted below.

## Headline findings

- **The deterministic (max-likelihood) slice of Sparsey's CSA is a faithful port**: region ordering, learn/recognize frame drivers, V and G computation, ML winner selection with random tie-break, the normalize→cutoff→exponent *shape*, persistence carry-over, the implicit stiffness/timestamp weight lookup, and the weight table.
- **The defining *probabilistic* CSA is now implemented as an opt-in mode** (was M1-absent). `enable_csa()` turns on the sigmoid V→μ→ρ cumulative draw for learning, following the **upstream** formulas — expansivity (`determine_mu_Range`), the `max(eta/(1+exp(-nonlin(V-inflect))), lower_limit)` sigmoid (`recompute_mu_And_rho`), and the **dynamic inflection ratchet** (`determine_Inflection_Point`, gated on mean-`V_ave`) — in `net/frame.rs` `expansivity`/`cell_mu`/`sample_winner`/`select_winners`. Recognition stays max-V. **CSA is now the default** (`disable_csa()` forces max-V). Still simplified: `shouldUse_ML_CSA` post-freeze gating. See [AlgorithmTriangulation.md](AlgorithmTriangulation.md).
- **Band-limited projective-field connectivity is implemented** (2026-07-04); `band_thickness`/`band_rates` wire block-distance→band→rate. Empty bands (default) = full within-link connectivity. (Per-synapse `Bernoulli(rate)` vs Java's exact `round(rate·K)`-per-target.)
- **The MCH / hypothesis subsystem** — `num_mch` (per CM + per MAC) is now computed in `compute_g`, and the efferent signal ignore/discount is applied in `push_into_region` (internal sources only). Still MISSING: exclude-prev-winner and the efferent signal *boosting* variant.
- **Backoff now selects max-G per level** (was first-clearing). `canonical_full` adds every 2-way/1-way case (HUD → {HU,UD,HD} → {H,U,D}); the strategy is fully data-driven so any level structure is expressible. Still simplified: no chain recording; no All/AtLeastOne within-type policy; no distinct top-region strategy.
- **✓ Fixed 2026-07-04 (audit-surfaced defects, with regression tests):** over-promotion of decayed synapses (`promote_to_permanent` now gates on contribution; `reconcile_if_inert` now wired into `update_weights`), H-link self / same-CM connection (`wire_connectivity` now excludes both), and the promotion gate (`record_pre_post` now `age >= persistence`). See [Divergences.md](Divergences.md).
- **Still divergent (deliberate M1 scope, not a bug):** the non-"muddled" feature filtering (Java excludes ambiguous features from the active count) is not modeled; the multi-scale (neuron/CM/Mac/region) transient/permanent synapse-count aggregation is not modeled (Java feeds it to its stats layer; the port uses the `Recorder` seam instead); and sub-bundle-granular freezing needs the deferred `SubEfferentBundle`. **Bundle-scale** transient/permanent counts *are* now available (`EfferentBundle::transient_count`/`permanent_count`), and the freeze decision is count-equivalent (`increased_fraction`). *(The π⁻/π⁺ activation band is also implemented — see below.)*

---

## Compute path (`net.rs`)

### Frame orchestration
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Network::doFrameLearn/Recognize` | `do_frame_learn_rec` / `do_frame_recognize_rec` | FAITHFUL | region loop bottom→top; learn adds weight-update + end-frame. |
| `Network::doFrameRecall` | `do_frame_recall_rec` | DIVERGENT | runs the *recognition* path under a Recall flag; downward D-signal regen to L0 not implemented. |
| `InternalRegion::processLearning/processRecog` (two mac passes) | `process_regions` + `process_region_macs` (one pass) | DIVERGENT | eligibility + normalize + V/G + select folded into one pass; `numConsecActiveFrames`/`quiescenceAge` tracking MISSING. |
| region height ordering | height sort in `process_regions` | FAITHFUL | ascending DAG height preserved. |

### Signal push / accumulation
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Mac::pullSignals_*` (pull per afferent bundle) | `push_into_region` (push from active sources) | DIVERGENT | equivalent weighted-sum-per-type result; class-input-region skip and recall U-skip on post-cue frames MISSING. |
| — | two-phase read-buffer-then-apply push | RUST-ONLY | reads neuron arena immutably into a local buffer, then applies — sidesteps single-`Vec` aliasing. |
| `Mac::computeMaxRawInputSums` / `CM::computeMaxInputSum` | — | MISSING | per-CM max-raw-sum tally (feeds crosstalk stats). |
| `Mac::computeNumberOfMCHs` / `computeEffSignalMultipliers` | `compute_g` num_mch + `mch_signal_factor` in `push_into_region` | FAITHFUL *(discount, 2026-07-05)* | MCH count + ignore/discount ported; efferent *boosting* variant still MISSING. |

### Normalization / V / G
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Mac/CM::computeNormalizedInfluences` | `normalize_mac` | DIVERGENT | normalize→min/max-cutoff (>max ⇒ 1.0) faithful; **normalizer is dynamic `count·MAX_WEIGHT`**, ignores Java's `Q·MCH` weighting and static/dynamic policy. |
| `Mac::computeAdjustedSums` | folded into `normalize_mac` (`^exp`) | DIVERGENT | Java picks exponent by **episode-initial (EI) vs non-EI**; Rust uses one fixed `exp` per type — no EI/NEI switch. |
| `Mac::computeHighestOrder_V_Available` / `computeSpecific_V` | `set_v` | FAITHFUL | product over available/specified types. |
| `Mac::compute_G` | `compute_g` | FAITHFUL | G = mean over CMs of per-CM max-V. |
| `Mac/CM::compute_V_Maxes_And_Hypotheses_*` | `compute_g` | DIVERGENT | v_max/v_ave/tied/**num_mch** faithful; no `excludePrevWinner`. |
| `CM::compute_V_Max_And_Hypotheses_ExcludePrevWinners` | — | MISSING | can't enforce disjoint codes for consecutive-identical inputs during learning. |

### Winner / code selection
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Mac::select_Winners_Learning` | `select_winners` | DIVERGENT | Java default = **probabilistic CSA** (sigmoid μ/ρ draw), ML only post-freeze; Rust **always** max-V ML + seeded-RNG tie-break. |
| `CM::pick_Winner_ML` | inlined in `select_winners` | FAITHFUL | uniform random among max-V ties. |
| `CM::pick_Winner` (probabilistic ρ draw) | — | MISSING | cumulative-ρ draw. |
| `recompute_mu_And_rho` / `get_mu_From_V` / `determine_Inflection_Point` | `cell_mu` / `sample_winner` / `select_winners` inflection ratchet | FAITHFUL *(opt-in, 2026-07-04)* | V→μ→ρ sigmoid + dynamic inflection ported (upstream form). `shouldUse_ML_CSA` post-freeze gating still simplified. |
| — | `mac_code` + `learned_codes.insert` | RUST-ONLY | extract per-CM winner indices as the code + record in `CodeTrie`. |

### Backoff
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Mac::compute_G_Recog_With_Backoff` | `recognize_version` + `BackoffStrategy::evaluate` | FAITHFUL *(max-G, 2026-07-05)* | Within a level keeps the **max-G** clearing case (fixed); `canonical_full` supplies all combos. Still simplified: no-threshold fallback = highest-priority-available (vs Java highest-complexity/G); no chain recording; no All/AtLeastOne within-type policy. `>=` threshold + final re-set_v/compute_g faithful. |
| learning path (no backoff) | learning branch | FAITHFUL | highest-order-available V/G, no backoff. |

### Activation / persistence bookkeeping
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Mac::willBecomeActiveRetrieval` / `areFeatureBoundsSatisfied` | eligibility in `process_region_macs` (`mac_u_active_features` + region band) | FAITHFUL *(band, 2026-07-05)* | π⁻ ≤ active U feature count ≤ π⁺ now enforced (`active_{low,high}_frac` × U input size, resolved per region). Default `[0,1]` ⇒ `[1,∞]` (any input). Non-"muddled" filtering still MISSING. Persistence carry-over faithful. |
| `Mac::updateState`/`updateAgeVars` | `end_frame` | DIVERGENT | Java `codeAge` counts **up**; Rust counts **down**; Rust rolls prev uniformly vs Java's quiescence-gated; `quiescenceAge`/`numConsecActiveFrames` MISSING. |
| `Network/CM::prepareForNewRun` | `prepare_for_new_run` | FAITHFUL | resets activity/V/accumulators/winners/code-age; optional weight erase. |

---

## Learning & connectivity

### Synapse & weight table (`synapse.rs`)
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Synapse` ctor / `erase` / `isPermanent` / `wasFreshlyIncreased` / `isInert` / `getEffectiveAge` | `Synapse::new` / `is_permanent` / `fresh_index` / … | FAITHFUL | virgin ts = `i64::MAX`; implicit `(stiffness, age)` lookup faithful. |
| `getEffectiveValue` (returns weight **and** demotes decayed transient) | `effective_value` (read-only) + `reconcile_if_inert` | DIVERGENT | Rust splits out the demotion; `reconcile_if_inert` is now wired into `update_weights` *(fixed 2026-07-04 — was uncalled)*. Transient/permanent **count** subsystem still absent. |
| `promoteFromTransientToPermanent` | `promote_to_permanent` | DIVERGENT | now gated on `is_contributing_learning` so decayed synapses aren't consolidated *(fixed 2026-07-04)*; still no perm/trans count bookkeeping. |
| `Network.createWeightTable` | `WeightTable::build` | FAITHFUL | identical step-function; default matches Java (127/120/50; 2000/3000/4000 & 4000/6000/8000). |
| — | `Synapse::record_pre_post` | RUST-ONLY | consolidates Java `EfferentBundle.updateWeights`' per-syn body; promotion gate now `age >= persistence` *(fixed 2026-07-04 — was `age >= 1`)*. |

### Bundles & freezing (`bundle.rs`, `net.rs`)
| Java | Rust | Verdict | Note |
|---|---|---|---|
| per-target `SubEfferentBundle` map | flat `Vec<Synapse>` per (source, link) | DIVERGENT | no sub-bundle layer → loses per-target grouping, per-sub-bundle freezing, frozen fast-path. |
| `EfferentBundle.updateWeights` | `update_weights` + `record_pre_post` | DIVERGENT | whole-bundle iterate; **maintains no counts**; freezing is a separate pass. |
| freezing block | `freeze_saturated_bundles` + `EfferentBundle::freeze` | DIVERGENT | Java freezes at **sub-bundle** granularity on count `>=`; Rust freezes **whole bundle** on fraction `>`; no up-scale propagation. |
| `promoteAllTransToPerm` (only syns with weight > 0, + counts) | `EfferentBundle::freeze` (every transient-flagged syn) | DIVERGENT | no `>0` check, no counts. |
| perm/trans count family; `getSynapseByTargetNeuron`; frozen propagate fast-path | — | MISSING | count subsystem + frozen all-permanent fast-path absent. |
| `AfferentBundle` (counts) | `AfferentAccum` (`raw_sum`/`active_input_count`/`normalized`/`adjusted`) | DIVERGENT | afferent perm/trans counters dropped; the normalize/adjust compute is faithful. |

### CodeTrie (`codetrie.rs`)
| Java | Rust | Verdict | Note |
|---|---|---|---|
| K-ary trie | `HashMap<Vec<u32>, i64>` | DIVERGENT | full-code → first-seen frame; no prefix/overlap traversal. |
| `insert` (usage count + `(trial,frame)` history) / `search` / `GetNumUniqueCodes` | `insert`→bool / `contains` / `len` | DIVERGENT / FAITHFUL | Rust records only first-seen frame; `GetNumTimesUsed`, code listing, trie persistence MISSING. |

### Connectivity wiring & regions (`net.rs`, `region.rs`, `link.rs`)
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `Network.createInterRegionLinks` | `build_links` + `syn_type_from_heights` | DIVERGENT | H/U/D from DAG height + `use_previous_active` faithful; ~8 per-type NDF params moved from `Link` onto the target region's `SignalParams`. |
| **band-limited PF** (`readBandInfo` + distance→band→rate loop) | `wire_connectivity` + `cumulative_radii` | FAITHFUL *(2026-07-04)* | `band_thickness`/`band_rates` now wire block distance→band→rate; empty bands = full (default). Bands in normalized `[0,√2]` grid units. |
| `buildBlockMatrix` — rate<1 random draw + **H self/same-CM exclusion** | `wire_connectivity` | DIVERGENT (approx.) | H self/same-CM exclusion faithful; rate<1 is per-synapse `Bernoulli(rate)` vs Java's exact `round(rate·K)`-per-target draw. Same density; normalization uses actual fan-in. |
| `forceNearestNeighborConnectivity` | — | MISSING | guaranteed-coverage fixup. |
| `InternalRegion.build` (Macs→CMs→neurons) | `build_regions` Internal branch | FAITHFUL (structurally) | Q/K/persistence nesting reproduced; receptive-field-block setup dropped. |
| `InputRegion.build` (apertures, tessellation) | `build_regions` Input branch | DIVERGENT | one input cell per block position; no `Aperture`/tessellation objects. |
| `doFinalTransToPermSynapsePromotionPass` | `finalize_learning` | DIVERGENT | promotes every transient-flagged syn; Java only weight>0 + counts. |

### Config / NDF & dataset
| Java | Rust | Verdict | Note |
|---|---|---|---|
| `DescriptorFile.parseFile` (bespoke NDF regex grammar, ~2200 lines) | `NetworkConfig::from_json` (serde + `#[serde(default)]`) | DIVERGENT | entirely different format/mechanism; NDF→field mapping in `config.rs` doc comments. |
| `WT_TABLE_*` statics | `WeightTableConfig` (data-driven) | FAITHFUL | default matches Java statics exactly. |
| `EpisodeContainer` (dataset/episode loader, ~1660 lines) | — | MISSING | no dataset layer; input fed frame-by-frame via `set_input`. |

---

## Functional-fidelity harness (three layers)

Unlike `dcc_sph` (whose harness proves *bit-exact* integer parity — both use the same
PCG32 RNG), byte-exact parity with Java SparseyCore is **structurally impossible**:
different PRNG (`Xoshiro256++` vs `java.util.Random` LCG), different random-draw counts
and order (Java randomizes wiring + synapse targets; Rust is deterministic
full-connectivity), deterministic max-V selection vs Java's default probabilistic
sigmoid CSA, and `f32` vs `double`. So fidelity is checked at three graded strengths:

1. **Weight-dynamics cross-check — EXACT, RNG-free** (`tests/fidelity_weight_dynamics.rs`).
   The implicit-decay `WeightTable` + `Synapse` stiffness/promotion logic is integer-valued
   and its constants are *identical* to `Synapse.java` (`WT_TABLE_*`, `MAX_WEIGHT=127`,
   `MAX_SYNAPSE_STIFFNESS=2`). The full decay/promotion trajectory is pinned to
   hand-derived Java golden numbers. **Passing.** This is the highest-confidence surface.

2. **Rust golden snapshot** (`tests/fidelity_snapshot.rs`, `tests/fixtures/sparsey_snapshot_golden.json`).
   Locks in the current Rust coding behavior for a fixed input sequence (M1 subset,
   seed 42) so unintended drift is caught even where no exact upstream comparison exists.
   Regenerate with `UPDATE_SNAPSHOTS=1`.

3. **Behavioral invariants vs. an independent implementation — runnable**
   (`fidelity/python/`, `tests/fidelity_behavioral.rs`). The reference is **Sparsey_Alt**,
   a third-party *independent* Python reimplementation of Rinkus's algorithm (numpy-only).
   `generate_reference.py` learns an alphabet with the probabilistic CSA (seeded) and
   recognizes with a max-V readout; the committed fixture records the behavioral
   invariants. The Rust test asserts our port exhibits the **same** invariants — N inputs
   → N distinct codes, learned inputs reactivate, G_familiar≈1.0 ≫ G_novel — and passes.
   Runs in CI (fixture committed, no Python needed). A SparseyCore Java scaffold for the
   same purpose is parked (`fidelity/java/`, verified-wired but not completed — SparseyCore
   ships no sample NDF, and its selection is probabilistic anyway).

The deferred mechanisms in the tables above (probabilistic CSA, band-limited PF,
MCH, …) are exactly what keep layer 3 a *behavioral-invariant* check rather than exact.

## Algorithm triangulation

Beyond the port↔upstream tables, [AlgorithmTriangulation.md](AlgorithmTriangulation.md)
cross-checks the *core algorithm* (V, G, expansivity, sigmoid CSA, winner selection,
learning) across **four** sources — this port, SparseyCore (Java), the independent
Sparsey_Alt (Python), and the `V-to-mu` CSA demo (jar). Verdict: our V, G, and WT_TABLE
learning are faithful (they match the independent consensus); the single architectural
gap is the probabilistic CSA (a deliberate M1 simplification), whose canonical formula
is now pinned by the SparseyCore≡Sparsey_Alt agreement.

---

*See [Divergences.md](Divergences.md) for the prioritized audit and the deferred-feature → upstream-spec map. The upstream algorithm is the [SparseyCore `ARCHITECTURE.md`](https://github.com/jacobeverist/SparseyCore/blob/a0d4d34faf1e3b6cf532c3846446c941ff80eff5/ARCHITECTURE.md) @ `a0d4d34`.*
