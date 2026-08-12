# Sparsey abbreviation glossary

Shared vocabulary between SparseyCore (Java) and this crate. Kept verbatim where possible so the code reads the same as the reference `ARCHITECTURE.md`.

## Structural

| Term | Meaning |
|------|---------|
| **Region** | A node in the network DAG. Leaf = input region (height 0); internal region holds a grid of MACs. (Java: formerly "Level".) |
| **MAC** | Macrocolumn. A grid unit in an internal region; contains `Q` CMs. |
| **CM** | Competitive Module (a.k.a. minicolumn). Winner-take-all group of `K` cells. |
| **Cell / neuron** | Binary unit. Exactly one cell wins per CM per frame. |
| **Aperture** | Receptive-field grouping of input cells in an input region. |
| **Block** | Java base class of Mac + Aperture (spatial position). Folded into records here. |
| **DAG height** | Number of links on a region's longest path to a leaf. Determines connection type. |

## Signals & connectivity

| Term | Meaning |
|------|---------|
| **U** | Upward / feedforward signal (src height < target height). |
| **H** | Horizontal / lateral signal (same height). |
| **D** | Downward / feedback signal (src height > target height). |
| **Link** | A connection between two regions; carries one synapse type. |
| **Bundle** | A group of synapses. Efferent (source-owned, holds weights) vs afferent (target-owned, accumulators). Upper = block/region-scale aggregation. |
| **PF band** | Projective-field distance band: `*_Connectivity_Band_Thickness` + `*_Connectivity_Band_Rates` control connection distance/probability. |

## Codes, evidence & matching

| Term | Meaning |
|------|---------|
| **Code** *(SDC)* | The set of winning cells (one per CM) in an active MAC; a one-hot-grouped `Q·K` pattern. Upstream's canonical name is **sparse distributed code (SDC)** (a "cell assembly"); "SDR" appears only informally in one source comment. |
| **CSA** | **Code Selection Algorithm** — Sparsey's canonical name for how a MAC's code is chosen: compute per-cell V → global familiarity G → expansivity η → per-cell μ → cumulative ρ → sample one winner per CM. The algorithm's variant name (as "SPH" is OgmaNeo's). Ported as an opt-in mode (`enable_csa()`); default is deterministic max-V. See [AlgorithmTriangulation.md](AlgorithmTriangulation.md). |
| **V** | Local evidence/support per cell — normalized combination of H/U/D influences. |
| **G** | Global match for a MAC — mean V across its CMs, computed in "versions" (HUD, HU, HD, UD, H, U, D). |
| **η (expansivity / mu_Range)** | How sharply the CSA concentrates probability on high-V cells; grows with G (`1 + ((G−floor)/(1−floor))^exp·factor·K`). η→1 (novel) = uniform; η large (familiar) ≈ max-V. |
| **Backoff** | Falling back from higher-priority (more complex) match rules to simpler ones when a threshold isn't met. Data-driven `matchingRules`. |
| **MCH** | Multiple Concurrent Hypotheses — tied `V_max` cells in a CM; output signals scaled/discounted (`*_MCH_Ignore_Thresh`, `*_MCH_Discount_Exp`, `*_MCH_Discount_Thresh`). |
| **mu (μ)** | Per-cell unnormalized relative probability from the V→μ sigmoid (bounded `[1, η]`); the CSA's selection weight for a cell. |
| **rho (ρ)** | The cumulative distribution over a CM's cells (`cumsum(μ)`) that the CSA samples the winner from. |
| **Persistence** | Number of frames a code stays active once selected (`RegionPersistence`). |
| **CodeAge** | Frames since a MAC's current code was selected. |

## Weights

| Term | Meaning |
|------|---------|
| **Stiffness** | 0 = malleable (decays), max = permanent. Promotes with repeated pre-post coincidence. |
| **timestampLastPrePost** | Frame of a synapse's last pre-post coincidence; weight age = current frame − this. |
| **WT_TABLE** | `[stiffness][age] → weight` decay table (values ≤ 127). |
| **Freezing** | A bundle stops learning once its increased-synapse fraction exceeds a saturation threshold (critical period). |

## Frame timing

| Term | Meaning |
|------|---------|
| **EI / post-quiescent** | "Episode-initial" — the frame after quiescence. Uses `U_EI_Exp`. |
| **NEI / non-post-quiescent** | All other frames. Uses `U_NEI_Exp`. |
| **Q, K** | CMs per MAC (`Region_Q`), cells per CM (`Region_K`). |

## Modes

| Term | Meaning |
|------|---------|
| **Learning / Recognition / Recall** | Run modes (`OperationMode`). Learning updates weights (no backoff); Recognition runs backoff over max-V matching; Recall replays D-signals to reconstruct a learned sequence. |
