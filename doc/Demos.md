# Demos

Two instrumented demos, plus the four minimal API walkthroughs that came before them. They all **run headless and text-only with no features enabled**, which is the default path and the one CI runs.

`noise_robustness` implements the cross-repo demo contract shared with [`dcc-sph`](https://github.com/jacobeverist/dcc-sph) and [`dcc-htm`](https://github.com/jacobeverist/dcc-htm). `capacity` shows something neither sibling models at all.

## Running them

```bash
cargo run --release --example noise_robustness
cargo run --release --example noise_robustness -- --sweep noise=0,0.2,0.4,0.6,0.8 --repeat 3
cargo run --release --example capacity
```

| Flag | Meaning |
|---|---|
| `--seed N` | default 12345; any `u64`, including zero |
| `--every N` | periodic sample interval |
| `--quiet` / `--silent` | suppress periodic lines / also the final report |
| `--metrics <path>`, `--metrics-format {jsonl,csv}` | machine-readable output |
| `--repeat N`, `--sweep key=v1,v2,...` | run matrix |
| `--save <path>`, `--load <path>` | checkpointing |

## The cross-repo demo contract

Three clean-room ports share this shape. Their architectures are too different to benchmark against one another, but an experiment run against any of them should *look* the same, so results are legible side by side.

**The three repositories share no code, and cannot.** `dcc_sph` is CC BY-NC-SA 4.0 (NonCommercial) while this crate and `dcc_htm` are AGPL-3.0, and those licences are mutually incompatible — no crate may ever link more than one of them. What they share is this contract, which is the arrangement the dcc-core import contract already uses: the rules are common and each repository answers them in its own [`Conformance.md`](Conformance.md).

**Records.** Four kinds in order — `run`, `sample`, `summary`, `verdict` — one JSON object per line, each carrying `demo`, `seed` and `run`. Config keys are sorted, so **the same seed produces byte-identical metrics**. Non-finite values serialise as `null`, so a file always parses.

**Baselines.** Every headline metric ships with a `baseline_*` twin and a `*_vs_*` ratio. A bare number cannot distinguish learning from noise.

**Verdicts.** `"Learned: …"` or `"Not converged: …"`. A **correct negative result is `learned: true`** with a note explaining why more training cannot change it — `capacity` relies on this directly.

**Shape.** `fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary`, with `main` doing nothing but parse, build a `Recorder`, and call `sweep::drive`.

## The demos

| Demo | Capability | What it exercises |
|---|---|---|
| `noise_robustness` | **contract** — graceful degradation | one-pass learning, recognition, familiarity `G`, code similarity |
| `capacity` | plasticity as a finite resource | stiffness/timestamp dynamics, bundle freezing, crosstalk |

And the four pre-existing walkthroughs, unchanged: `hierarchy`, `learn_and_recognize`, `novelty_detection`, `save_and_load`. They are the shortest way to see the API and are deliberately not instrumented.

## Layout

`examples/support/` holds everything shared. Cargo examples cannot depend on each other, so each demo pulls it in with `#[path = "support/mod.rs"] mod support;`. A directory under `examples/` with no `main.rs` is not auto-discovered, so `support/` is not itself a target.

Example targets default to `test = false`, so `#[cfg(test)]` code inside them never runs. `tests/demos_support.rs` includes the same tree in test configuration and runs its unit tests as part of `cargo test` — **63 of them**.

## Cross-cutting decisions

**Use the `_rec` frame drivers.** With `persistence = 1` a MAC's code is cleared in `end_frame`, so `net.mac_code(mac)` after the frame returns `None` and a demo reading it that way silently sees nothing. `support::probe::Capture` is the recorder; it replaces the four private copies the older examples each defined.

**Code similarity is per-CM agreement, not bit overlap.** A code is one winning cell per CM, so two codes agree in some number of their `Q` modules. Comparing them as if they were plain SDRs would count the losing cells as agreement.

**Corruption preserves the active-cell count**, and here that is load-bearing in a way it is not in the sibling ports. A MAC only activates when its active-input count falls inside its activation band (`active_low_frac`…`active_high_frac`), so a corruption that thinned the input would push it out of the band and the MAC would not activate at all. The result would look like total failure under noise when the input was simply rejected as malformed — a completely different finding.

**Checkpoints carry the seed and a config hash.** `serialize_state` writes only the synapse state, never the structure, which is rebuilt from config + seed — and band-limited wiring draws from the RNG, so the seed is part of the structure too. `load_state` validates the bundle *count* and nothing else, so a changed config that happened to preserve that count would load without complaint and attach every synapse to the wrong target. `support::checkpoint` refuses both mismatches, with a test for each.

**Seeds are per-object.** This crate satisfies R9 — `SparseyNet::build(config, seed)` owns its stream — so the demo's randomness is a separate object with a separate derived seed. Giving both the same seed would let the environment replay the draws the network made while wiring its connectivity.

## Deviations and findings, demo by demo

### `noise_robustness`

The demo shared by all three ports. Patterns are learned in one frame each — Sparsey commits a code per frame, single-pass, no epochs — then recognised under controlled corruption.

**Two things are measured from the same frame, and this port is the only one of the three that can report both:** whether the right code comes back (the content-addressable answer), and how familiar the network says the input is (the scalar `G`, which needs no classifier, no labels and no threshold training).

**Recognition is scored by nearest stored code**, done in the demo rather than in `src/` because this crate has no classification head — Java's `supervisedMode` and its class-input region are not ported. That is a genuine gap, but doing the readout in the demo *showcases* the SDM rather than working around it: the network answers with a code, and the class is whichever stored code it most resembles.

**A frame where the MAC did not activate is skipped, not counted wrong.** Falling outside the activation band is a different event from recalling the wrong thing, and merging them would hide it.

**The control is exact-match lookup, and it is not there to be beaten.** Perfect at zero noise and blind one cell off it, its curve is the shape memorisation produces. There is deliberately no `accuracy_vs_lookup` ratio: exact match is zero above zero noise *by construction*.

**The verdict keys on the clean pass.** Failing to recall at heavy corruption is the correct answer, not a failed run.

Typical result, 3 seeds:

| `--noise` | accuracy | `G` | code similarity | exact match |
|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.2 | 1.000 | 0.802 | 1.000 | 0.000 |
| 0.4 | 1.000 | 0.604 | 1.000 | 0.000 |
| 0.5 | 1.000 | 0.525 | 0.999 | 0.000 |
| 0.6 | 1.000 | 0.446 | 0.994 | 0.000 |
| 0.8 | 0.728 | 0.270 | 0.631 | 0.000 |

Chance is 0.125.

**This is the third distinct shape the three ports produce on the same task, and the comparison is the most interesting thing the suite yields.** Recall here is *flat* to 60% corruption and the stored code comes back all but exactly, while `G` declines almost linearly the whole way. `dcc-sph` decays smoothly from 0.98 to 0.50 across the same range; `dcc-htm` is a cliff, perfect through 0.40 and at chance by 0.50. Neither sibling has a confidence signal at all. So this network keeps answering correctly *while reporting, accurately, how corrupted the input was* — which is the capability worth showing, and it is not visible in an accuracy number.

### `capacity`

**Neither sibling port models this.** Sparsey stores no weights: a synapse carries a `(stiffness, timestamp)` pair and its effective value is a table lookup on how long ago the last pre-post coincidence was. Repeated coincidences walk stiffness up to permanent, and a bundle whose increased fraction passes the target region's saturation threshold **freezes for good**. Plasticity is a finite resource with an observable level.

The demo stores patterns one at a time and watches three things move together: synapses touched, bundles frozen, and whether the *earliest* patterns can still be recalled.

**The baseline is the network's own initial recall**, measured before capacity is a factor. Without it, "recall fell to 12%" cannot be told apart from "recall was never better than 12%".

**A negative result here is a correct result, and the verdict is `learned: true` either way.** This demo does not claim the network learns something; it claims capacity is finite and observable. If nothing saturated within `--patterns`, that is a true measurement of the configuration and more training cannot change it — so the note says so rather than the run reporting failure.

Typical result at defaults (800 patterns, Q=12, K=16):

```
  synapses touched over the run:  ▁▄▆▇▇███
  recall of the first 16:         █▇▅▃▁▁▁▁

  bundles                     256
  frozen                      247  (96.5%)
  synapses                  49152
  touched at least once     44312  (90.2%)
  permanent                 43948  (89.4%)

  first bundle froze after 387 patterns
  recall of the first 16 fell below 90% after 300 patterns
```

**The ordering of those last two numbers is the finding.** Recall of the earliest cohort degrades at **300** patterns, but the first bundle does not freeze until **387** — so crosstalk bites *before* plasticity is exhausted. The network is still perfectly able to learn when it has already begun to forget, which means "it stopped learning" and "it stopped remembering" are separate events with separate causes and should not be diagnosed as one.

## What CI does

`cargo build --all-targets`, `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`, and then **runs every demo** at small budgets. Building is not enough — a demo that panics on step one still compiles.

**CI checks that a demo runs, never that it learns.** The gate is the exit code and nothing else. No metric value is asserted anywhere; the metrics step checks the record *format* and same-seed determinism rather than the numbers in it. Whether a configuration learns is an experiment, and experiments belong in an offline run with `--repeat` and `--sweep`, with someone reading the spread.

## Not yet ported

Four demos from the plan this suite was built to are not written yet: `classify_stream` (the contract's classification cell, which would use the nearest-stored-code readout `noise_robustness` already implements), `sequence_recognition`, `partial_cue` and `backoff_modality`. The scaffolding each needs is in place — `env/sequences.rs` and `env/patterns::occlude` were written for the first three.

**`sequence_recognition` is the one worth reading about before attempting.** The other two ports answer the temporal cell of the demo matrix with next-symbol *prediction*. This crate cannot: it has temporal context — H and D links read the previous frame (`use_previous_active = syn_type != SynapseType::U`, `src/net/build.rs`) — but no path from a code back down to input features. `src/net/frame.rs` records that faithful D-signal regeneration of the input region is a documented follow-on, and that "for nets without downward (D) links this behaves like recognition". So the demo would be sequence *recognition*: learn episodes, then report familiarity for seen and unseen orderings. Same family, honestly different task.

It would also be **the first code in this repository to exercise recurrent H links at all** — no test, example or bench currently connects a region to itself, so that path is untested. Budget for finding real bugs there rather than assuming it works.
