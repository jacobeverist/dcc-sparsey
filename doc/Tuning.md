# Sparsey — Tuning

Parameters and tuning advice for the `sparsey` crate. See [NameReference.md](NameReference.md) for term definitions and [UserGuide.md](UserGuide.md) for the run loop.

> **Status:** scaffold. Parameter definitions currently live inline in `config.rs` (`NetworkConfig` / `RegionConfig`, most fields `#[serde(default)]`) and in [NameReference.md](NameReference.md); this guide gathers the tuning-relevant ones with guidance over time.

## Key structural parameters

| Param (config) | NDF name | Effect |
|---|---|---|
| `q` | `Region_Q` | CMs per MAC — code capacity per MAC. |
| `k` | `Region_K` | cells per CM — winner-take-all width. |
| `persistence` | `RegionPersistence` | frames a code stays active once selected. |
| signal exponents | `U_NEI_Exp` / `U_EI_Exp` / `H_Exp` / `D_Exp` | relative weight of U/H/D evidence (EI vs NEI frames). |
| backoff thresholds | `matchingRules` | G thresholds for the HUD→…→U backoff chain. |
| stiffness / `WT_TABLE` | — | implicit-weight decay + consolidation. |
| saturation threshold | — | when a bundle freezes (critical period). |

## To expand

- [ ] Recommended starting points per parameter.
- [ ] Interaction notes (e.g. `Q`/`K` vs sparsity; persistence vs sequence length).
- [ ] PF connectivity-band tuning.
