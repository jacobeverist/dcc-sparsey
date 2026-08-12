# sparsey functional-fidelity harness

Checks the Rust `sparsey` crate against upstream **SparseyCore** (Java, commit
`a0d4d34`) at three graded strengths, because — unlike aogmaneo — byte-exact parity is
**structurally impossible** here (see `../doc/Divergences.md`): different PRNG
(`Xoshiro256++` vs `java.util.Random`), different random draw order (Java randomizes
wiring + synapse targets; Rust is deterministic full-connectivity), `f32` vs Java
`double`, and SparseyCore's default probabilistic sigmoid CSA vs the port's
deterministic max-V.

| Layer | Strength | Where | Status |
|---|---|---|---|
| 1. Weight-dynamics | **exact, RNG-free** | `tests/fidelity_weight_dynamics.rs` | ✅ passing |
| 2. Rust golden snapshot | regression lock-in | `tests/fidelity_snapshot.rs` + `tests/fixtures/sparsey_snapshot_golden.json` | ✅ passing |
| 3. Behavioral invariants | cross-impl, runnable | `tests/fidelity_behavioral.rs` + `python/` | ✅ passing (Sparsey_Alt) |

All three run in CI (committed fixtures, no external toolchain). The reference for the
whole comparison is also cross-checked *algorithmically* in
[`../doc/AlgorithmTriangulation.md`](../doc/AlgorithmTriangulation.md), which triangulates
the core algorithm across our port, SparseyCore, the independent Sparsey_Alt, and the
`V-to-mu` CSA demo — verdict: our V/G/learning are faithful; the one gap is the
probabilistic CSA (deliberate M1 simplification).

## Layer 3: behavioral invariants (active reference = Sparsey_Alt Python)

Exact codes cannot match across implementations — Sparsey's winner selection is the
probabilistic sigmoid CSA, so novel-code assignment is stochastic and RNG-specific. What
every faithful Sparsey shares are the **behavioral invariants**, asserted for both the
reference (via a committed fixture) and our port (run in-process):

1. N well-separated inputs → N *distinct* codes.
2. A learned input, re-presented, *reactivates its own* code (max-V readout).
3. Global familiarity G is high (~1.0) for a learned input, low for a novel one.

The reference is **`Sparsey_Alt`** — a third-party, *independent* Python reimplementation
of Rinkus's algorithm (from the papers, not from SparseyCore). It is numpy-only and runs
on Python 3.12. `fidelity/python/generate_reference.py` builds a single MAC, learns an alphabet
with the probabilistic CSA (seeded), recognizes each input with a deterministic max-V
readout (mirroring SparseyCore `Use_ML_Recog`), and emits the invariants as JSON.

```bash
PYTHONPATH=/path/to/Sparsey_Alt/Python \
  python3 fidelity/python/generate_reference.py \
  > sparsey/tests/fixtures/sparsey_alt_reference.json
cargo test -p dcc_sparsey --test fidelity_behavioral
```

The fixture is committed, so the Rust test runs in CI with no Python toolchain (it SKIPs
if the fixture is absent). **Result: our port satisfies the identical invariant set —
4 distinct codes, full reactivation, G_familiar≈1.0 ≫ G_novel.**

## Parked: SparseyCore Java scaffold

The original plan drove layer 3 from **SparseyCore** (Java, `a0d4d34`) directly. That
path is *parked* (verified-wired but not completed — see below): SparseyCore ships no
sample NDF, so a full descriptor must be reconstructed, and its winner selection is
probabilistic anyway. Sparsey_Alt (above) is the go-forward reference because it runs
cleanly. The Java scaffold is retained here for the record.

## Files

**Active (Sparsey_Alt Python reference):**
- `fidelity/python/generate_reference.py` — builds a single MAC, learns the alphabet with the
  probabilistic CSA (seeded), recognizes with a max-V readout, emits the invariants JSON.
  numpy-only; run with `PYTHONPATH=/path/to/Sparsey_Alt/Python`.
- `../tests/fixtures/sparsey_alt_reference.json` — the committed reference fixture.
- `../tests/fidelity_behavioral.rs` — runs our port's scenario and asserts the same
  invariants.

**Parked (SparseyCore Java scaffold — retained, not on the active path):**
- `config/m1.ndf.json`, `config/m1.com.json` — descriptor files (comment-free; OS-context
  `ndfPath`; COM `supervisedMode`).
- `java/Driver.java` — `package SparseyCore;` driver (seeds RNG, builds network, learns,
  `afterLearningAllEpisodes()`, flips to `RECOGNITION_MODE`, prints `Mac.getCurrentCode()`).
- `build_and_run.sh` — headless `javac` over `SparseyCore/{SparseyCore,util,stats}` +
  the driver. **Requires JDK 9+.**

### Java scaffold status (verified-wired on JDK 26, parked)

Verified: SparseyCore compiles headless JDK-only (59 classes); `Driver.java` compiles
against it (all API signatures resolve); it runs through COM+NDF parsing →
`setSupervisedMode` → into region construction. Findings folded into the config/driver:
the `DescriptorFile` parser rejects `_comment` keys (comment syntax is a literal
`"comment"` with no internal commas); `ndfPath`/`Stats_Path` must be OS-context objects;
the COM needs `supervisedMode`; ML/max-V is forcible via `net.setUseMaxLikeWinSelMethod(true)`;
the `Network` ctor builds regions itself. **Remaining to a green run** (why it's parked):
(1) SparseyCore ships no sample NDF, so a full region-contexted keyset must be
reconstructed (`Region_Intrinsic_D`, `Sigmoid_*`, `Tiling_*`, `Backoff_*`, `U_*`, `MU_*`,
…), and (2) input injection via `EpisodeContainer.theEpisodes`. Because SparseyCore's
selection is probabilistic too, this would still only yield a *structural* comparison —
no better than the Sparsey_Alt path, which runs cleanly. Left for the record.
