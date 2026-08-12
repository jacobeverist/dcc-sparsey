# Sparsey algorithm triangulation — is our port faithful to the intended design?

This document cross-checks the **core Sparsey algorithm** across four independent
sources to answer: *does the dcc `sparsey` port capture Rinkus's intended algorithm,
and where does it diverge?* Triangulation is the method — where two independent
implementations, written from the papers by different authors, **agree**, that is
almost certainly the canonical algorithm; where our port differs from **both**, that is
a red flag; where it differs from only one, it inherits that one's choices.

## Sources

| # | Source | Kind | Author | Notes |
|---|---|---|---|---|
| A | **dcc `sparsey`** (this crate) | Rust port | — | ported from SparseyCore (B) |
| B | **SparseyCore** `a0d4d34` | Java | Rinkus (original) | the reference the port was written against |
| C | **Sparsey_Alt** `Python/sparsey` | Python (numpy) | *independent* 3rd party | reimplemented from the papers, not from B |
| D | **V-to-mu_Demo.jar** | Java Swing GUI | Rinkus | pedagogical demo of the CSA; class `CSApackage.Mac` |

C is the key addition: an **independent** reading of the same papers. B↔C agreement is
strong evidence of the canonical algorithm; D corroborates the CSA specifically.

Vocabulary across sources (all denote the same thing):

| Concept | A (Rust) | B (SparseyCore) | C (Sparsey_Alt) | D (jar) |
|---|---|---|---|---|
| cell familiarity | `neuron.v` | `neuron.V` | `neuron.familiarity` | `V` |
| global familiarity | `compute_g` | `V_ave`/G | `mac.globalFamiliarity` | `G` |
| expansivity | — (absent) | `mu_Range` | `mac.expansivity` | `eta` |
| per-cell prob (unnorm) | — | `neuron.mu` | likelihood | `mu` |
| cumulative dist | — | `rho` | probabilities | `rho`/`rhoCumulative` |
| CMs per MAC / cells per CM | Q / K | Q / K | `columns` / `neuronsPerColumn` | Q / K |

---

## Component-by-component

> **Note (state at analysis time).** §3–§5 below describe the port *as it was when this
> triangulation was written* — "A absent / max-V only". The CSA (expansivity + dynamic
> sigmoid + ρ-sampling) has **since been implemented and is now the default**; see the
> "canonical CSA — implemented" and summary sections at the end.

### 1. Familiarity V (per cell) — **A matches consensus** ✅

- **A**: `V = Π_type (clamp(rawΣ / (activeCount·127), min_cut, max_cut))^exp`, product over active synapse types (`net.rs:normalize_mac`/`set_v`). Clamp above `max_cutoff` saturates to 1.0.
- **B**: same — normalized weighted sum, per-type EI/NEI exponent, cutoffs, saturation. (A was ported from B.)
- **C**: `V = clamp(Σ signal·weight / normalizer, 0, 1)^10` (`neuron.getSignalSummation` + `familiarityModulator=10`); U-term only (single type).

**Verdict:** all three compute V as *(normalized weighted sum) raised to a power,
clamped to [0,1]*. A and B match exactly (per-type exponents + cutoffs); C uses a single
exponent (10) and only the U term, but the **shape is identical**. Our port is faithful.

### 2. Global familiarity G — **unanimous** ✅

- **A**: `G = mean over CMs of (max V in the CM)` (`compute_g`).
- **B**: `V_ave` accumulates per-CM maxima → mean.
- **C**: `getMaxFamiliarities` (argmax V per CM) → mean → `globalFamiliarity`.

**Verdict:** all three identical. Faithful.

### 3. Expansivity η / mu_Range — **B and C agree EXACTLY; A does not compute it** ⚠️

- **B** (`CM.determine_mu_Range`, `CM.java:738`):
  `mu_Range = 1 + ((G − lowerGCut)/(1 − lowerGCut))^MU_Range_Expansion_Exp · (MU_Range_Expansion_Factor · K)`
- **C** (`calculateExpansivity`):
  `η = 1 + max(0,(G − Gmod)/(1 − Gmod))^expansionExponent · expansionFactor · K`
- **D**: field `eta`, driven by `G`, `G_lowCutoff`/`G_highCutoff`, `V_to_mu_Multiplier`.
- **A**: **absent** — the port jumps straight from V to max-V winner selection.

**Verdict — the headline finding.** Two independent implementations produced the
*identical* expansivity formula `η = 1 + (rectified normalized G)^exp · factor · K`
(B's `lowerGCut` = C's `Gmod`; `Exp` = `expansionExponent`; `Factor` = `expansionFactor`;
both scale by K). This is canonical. **Our port does not implement it** — see §5.

### 4. V→μ sigmoid — **B and C agree; A absent** ⚠️

- **B** — the *uncommented* `CM.recompute_mu_And_rho` (`CM.java:812`):
  `μ = max(mu_Range / (1 + exp(−nonlin·(V − inflect))), lowerLimit)`, with `inflect`
  **ratcheting** from `min` to `max` via `Mac.determine_Inflection_Point` as the MAC
  saturates. (The `μ^denomExp` form at `CM.java:803` is commented out.)
- **C** (`neuron.getLikelihood`, `sigmoidMods=[0.1, 4, 1, 8]`), fixed inflection:
  `μ = (η − 1)/(1 + (0.1 · exp(−4·(V − 1)))^8) + 1`.
- **D**: `whole_sigmoid_V[] → whole_sigmoid_mu[]`, `muMin`/`muMax`, `horizInflectionLocation`.

**Verdict (corrected):** B and C share the **logistic shape** (a sigmoid in V, scaled by
expansivity), but *not* the exact parameterization — B is a plain logistic with a
**dynamic** inflection; C adds a `(rate·exp)^denomExp` term and a fixed inflection. Only
the **expansivity** (§3) is bit-identical between them. The port follows **B (upstream)**:
`mu = max(eta/(1+exp(−nonlin·(V−inflect))), lower_limit)` with the ratcheting inflection.
A previously absent; now implemented.

### 5. Winner selection — the Code Selection Algorithm (CSA) — **A diverges from BOTH references (deliberate)** ⚠️ red-flag-but-known

Sparsey's winner-selection procedure is the **Code Selection Algorithm (CSA)** — the
algorithm's canonical name (Sparsey's variant name the way "SPH" is OgmaNeo's; see
[NameReference.md](NameReference.md), which maps to DCC's nomenclature crosswalk). "CSA"
throughout this doc = this `V → μ → ρ → sample` procedure.

- **A** (`select_winners`): **max-V** (argmax) per CM, ties broken by seeded RNG. No μ, no ρ, no sampling.
- **B**: default `pick_Winner` **samples** from the cumulative ρ distribution; `pick_Winner_ML` (max-V) only post-freeze.
- **C**: `getWinnerNeuron` = `np.random.choice(p=probabilities)` — **samples**. No argmax path for selection.
- **D**: the entire demo exists to visualize this V→μ→ρ→sample pipeline.

**Verdict:** the probabilistic CSA is the **defining mechanism** of Sparsey — confirmed
by two independent implementations *and* a dedicated pedagogical demo. Our port ships
**max-V only** (a documented M1 simplification). This is the one place A differs from
both references. It is not a *bug* (it's deliberate), but the triangulation elevates its
significance: it is the algorithm's core, not a peripheral detail.

*Why max-V is a defensible approximation at the limits:* the CSA's η modulates how
sharply ρ concentrates on the max-V cell. As G→1 (familiar), η→large, ρ collapses onto
the max-V cell → CSA ≈ max-V. As G→lowerGCut (novel), η→1, ρ→uniform → the pick is
uniformly random, which A approximates via its uniform RNG tie-break (all V equal on a
fully novel input). **A captures both limits but not the graded middle** — for
partially-familiar inputs the CSA assigns codes probabilistically with overlap
proportional to similarity, which max-V cannot reproduce. (Empirically: with argmax +
zero-init, C degenerates to code `[0,0,…]` on novel inputs — the sampling is what breaks
the symmetry. A avoids that degeneracy only via its RNG tie-break.)

### 6. Learning / weights — **A matches B; C is simpler (diverges the other way)**

- **A**: implicit-decay weight table (`WeightTable`/`Synapse`): `(stiffness, age)→weight`,
  `MAX_WEIGHT=127`, promotion on re-coincidence, decay by table lookup. Matches B exactly
  (`WT_TABLE_*` constants identical — see `fidelity_weight_dynamics.rs`).
- **B**: the `Synapse` age+stiffness `WT_TABLE`.
- **C** (`neuron.learnSignal`): **set-once binary** — an active input on a zero synapse
  latches to 127 forever; no decay, no stiffness, no age (`weightMax=127`).

**Verdict:** A and B agree (decay/stiffness table). C omits decay entirely — a
*simplification in C*, not in A. So our port is **more** faithful to SparseyCore here;
the fact that an independent implementation dropped the decay table suggests it is not
essential to the *core* coding behavior (it governs forgetting/capacity over long runs),
but A correctly follows B. **Not a red flag for A.**

### 7. MAC activation gate — consistent

- **A**: activation = "any U input reached the MAC" (per `Divergences.md`).
- **B**: π⁻/π⁺ feature-count bounds + non-muddled filtering.
- **C**: `isMacActive` = bottom-up active count within `[minFeatures, maxFeatures]`
  (default `[0.03, 0.08]`); out-of-band ⇒ empty code `[]`.

**Verdict:** B and C both gate on an input-density band; A uses a looser "any input"
criterion (documented divergence from B). Worth noting for behavioral comparison —
inputs whose density is out of band produce empty codes in B/C but active codes in A.

---

## Summary verdict

| Component | A vs consensus | Status |
|---|---|---|
| V (familiarity) | matches B exactly; same shape as C | ✅ faithful |
| G (global familiarity) | unanimous | ✅ faithful |
| Expansivity η | B≡C canonical formula | ✅ implemented (opt-in `enable_csa()`) |
| V→μ sigmoid | B/C share shape; port follows B (dynamic inflection) | ✅ implemented (opt-in) |
| Winner selection | max-V default; **CSA ρ-sampling opt-in** | ✅ both paths (default max-V) |
| Learning/weights | matches B; C simpler | ✅ faithful to B |
| MAC activation gate | looser than B/C density band | ⚠️ minor divergence |

**No red flags** — there is no component where our port diverges from *both* references
in an unintended way. The port's V, G, and WT_TABLE learning are faithful. The one
architectural gap was the **probabilistic CSA** (expansivity + sigmoid + ρ-sampling) —
the *defining* mechanism (B, C, and the D demo all center it). Following this
triangulation it is **now implemented** (opt-in; default stays max-V), following the
**upstream (B)** formulas — see below.

## The canonical CSA — implemented (opt-in), following upstream

The **expansivity** is bit-identical between B and C, so it's unambiguous; the **sigmoid**
differs between them (B dynamic-inflection logistic; C fixed-inflection `^denomExp` form),
so the port follows **B (SparseyCore, the upstream)**:

```text
# per MAC, after V and G are computed:
η       = 1 + max(0, (G − G_floor)/(1 − G_floor))^η_exp · η_factor · K       # B≡C (identical)
inflect = ratchet toward max_inflect once mean(V_ave) > threshold           # B (determine_Inflection_Point)
# per cell in a CM:
μ(V)    = max( η / (1 + exp(−nonlin·(V − inflect))), lower_limit )           # B (recompute_mu_And_rho)
# per CM:
ρ       = cumsum(μ) / Σμ                                                     # cumulative distribution
winner  = first index where uniform_draw(0,1) · Σμ < cumsum(μ)              # sample (B: pick_Winner)
```

At η→∞ this reduces to max-V; the CSA adds the graded-familiarity regime.

**Status: implemented as an opt-in mode** (2026-07-04). `SigmoidConfig` (`config.rs`)
carries the coefficients (`g_floor=0.1`, `expansion_exp=2`, `expansion_factor=100`,
`nonlin=4`, `min_inflect=0.5`, `max_inflect=0.9`, `lower_limit=1`, `mean_v_ave_threshold=0.3`);
`expansivity()` / `cell_mu()` in `net/frame.rs` implement the formulas (unit-tested —
`η(G=1,K=8)=801`; the dynamic inflection ratchets, tested in `tests/csa.rs`);
`select_winners` samples from ρ when the region's sigmoid is
`enabled` **and** the frame is a learning frame (recognition stays max-V, mirroring
SparseyCore `Use_ML_Recog`). Enable per region with `RegionConfigBuilder::enable_csa()`.

The default is now **on** (probabilistic CSA) — the whole stack (engine `sparsey`
block, WASM, dashboard) runs the faithful CSA; `RegionConfigBuilder::disable_csa()`
forces deterministic max-V (the M1 subset) where determinism is wanted. Tests: `tests/csa.rs` (CSA path is
exercised, deterministic, preserves the behavioral invariants) + `net.rs` math unit
tests. See `fidelity/README.md` for the runnable Sparsey_Alt reference and the
interactive `V-to-mu_Demo.jar` (source D) for exploring the V→μ→ρ mapping by hand.

---

*Sources: A = this crate (`net.rs`, `synapse.rs`); B = SparseyCore `a0d4d34`
(`CM.java`, `Synapse.java`); C = Sparsey_Alt `Python/sparsey` (`neuron.py`,
`minicolumn.py`, `macrocolumn.py`); D = `V-to-mu_Demo.jar` (`CSApackage.Mac`). See also
[Divergences.md](Divergences.md), [MethodFidelity.md](MethodFidelity.md).*
