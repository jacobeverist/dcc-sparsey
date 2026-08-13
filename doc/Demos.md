# Demos

Six instrumented demos, plus the four minimal API walkthroughs that came before them. They all **run headless and text-only with no features enabled**, which is the default path and the one CI runs.

Three implement the cross-repo demo contract shared with [`dcc-sph`](https://github.com/jacobeverist/dcc-sph) and [`dcc-htm`](https://github.com/jacobeverist/dcc-htm); three show capabilities specific to this architecture.

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
| `sequence_recognition` | **contract** — temporal structure | recurrent H links, backoff chains, episode familiarity |
| `classify_stream` | **contract** — online classification | nearest-stored-code readout over a codebook |
| `noise_robustness` | **contract** — graceful degradation | one-pass learning, recognition, familiarity `G`, code similarity |
| `partial_cue` | content-addressable completion | occlusion, code completion vs identification |
| `capacity` | plasticity as a finite resource | stiffness/timestamp dynamics, bundle freezing, crosstalk |
| `backoff_modality` | degradation under a missing signal | two input regions, the backoff priority chain, the activation band |

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

### `sequence_recognition`

**This is the temporal cell of the demo matrix, and it is a different task here than in the sibling ports.** They answer it with next-symbol *prediction*. This crate cannot: it has temporal context — H and D links read the previous frame (`use_previous_active = syn_type != SynapseType::U`, `src/net/build.rs`) — but no path from a code back down to input features, and `src/net/frame.rs` records D-signal regeneration of the input region as a documented follow-on. So the question is recognition: having learned some episodes, does familiarity distinguish a learned *ordering* from a novel one?

**The recurrent H path was untested before this demo, and it works.** Nothing in the test suite, examples or benches connected a region to itself. `connect("l1", "l1")` gives both ends the same DAG height, so `syn_type_from_heights` infers H and `use_previous_active` is set. A probe confirmed the links are wired and learned: 96 H bundles, 8064 synapses, **20.7% of them touched** after training. It is a real path, not a stub.

**The canonical backoff chain masks temporal novelty completely, and this is the finding to take away.** `BackoffConfig::canonical` is `HUD → {HU,UD} → U`. When the `{H,U}` case misses its threshold the chain falls back to `U` alone — and `U` alone is a *perfect* match, because every frame's content is familiar by construction; only the order is novel. Measured over 8 seeds:

| backoff chain | G, learned ordering | G, novel ordering | separation |
|---|---|---|---|
| canonical (has a U fallback) | 1.0000 | 1.0000 | **0.0000** |
| `{H,U}` only, no fallback | 0.1272 | 0.0545 | 0.0727 |

The fallback is doing exactly what backoff is for. It just happens to erase the only thing this demo measures. `--backoff canonical` reproduces the masked result, and the demo reports it as a correct negative rather than a failure.

**The control is the same network with the recurrent link removed** — same frames, same training, same scoring, and the only difference is whether the region can see its own previous code. That isolates the claim exactly, and it works: without the link the separation is `0.0000` and discriminability is `0.500`, precisely no signal.

**Episodes share their frames.** Each is a different ordering over a small alphabet, and a novel episode is an unseen *order* of seen frames. Novelty of content is spatial and any content-addressable memory detects it; only novelty of order is the temporal question.

**Frame 0 is excluded from scoring.** It has no predecessor, so its familiarity says nothing about order, and averaging it in would dilute the measurement with a constant.

**The headline is a rank statistic, not a difference of means.** The absolute separation is small (~0.017) because `V` is a product over signal types and the H term is only ever partially matched. Discriminability — the fraction of (learned, novel) pairs ranked correctly — survives that, and it is what the verdict gates on.

Typical result at defaults, 3 episodes of 6 frames over an alphabet of 10:

```
                        with H link   without H link
  G, learned ordering      0.4481          1.0000
  G, novel ordering        0.4309          1.0000
  separation               0.0172          0.0000
  discriminability          0.927           0.500
  H synapses touched        72917               0
```

**More competitive modules is not monotonically better.** Sweeping `--q` over 3 seeds gave discriminability 0.689 ± 0.061 at Q=12, **0.915 ± 0.016** at Q=24, and 0.821 ± 0.114 at Q=36 — the last both worse and much noisier. Q=24 is the default for that reason rather than by preference.

### `classify_stream`

**This crate has no classification head.** Java's `supervisedMode` and its class-input region are not ported (`doc/PortNotes.md`), so the readout is done in the demo: learn labelled exemplars, keep the code each produced, and answer with the label of whichever stored code the response most resembles.

That is not really a workaround. Sparsey's output *is* a content-addressable key, so classification is a lookup over codes rather than a trained head — no gradient, no epochs, no held-out tuning. The cost is that the codebook grows with the number of stored items.

**Several exemplars per class**, each a corrupted variant of the class prototype, so a class is a neighbourhood in input space rather than one memorised point. Testing on freshly drawn variants is what makes this classification rather than storage.

Typical result: **100%** on held-out exemplars against 16.7% chance. The task has real headroom — sweeping `--spread` over 2 seeds:

| `--spread` | accuracy | mean `G` |
|---|---|---|
| 0.2 | 0.997 | 0.798 |
| 0.4 | 0.981 | 0.598 |
| 0.6 | 0.678 | 0.489 |
| 0.8 | 0.197 | 0.437 |

Chance is 0.167, so 0.8 is the point where the class neighbourhoods have merged and the demo correctly reports "not converged".

### `partial_cue`

**Occlusion, not corruption, and the distinction is the demo.** `noise_robustness` moves active cells elsewhere, so the surviving evidence is wrong; here cells are dropped, so all surviving evidence is correct and there is merely less of it. A memory can be good at one and bad at the other.

**Two things are measured because they can disagree:** *completion* (how much of the stored code comes back) and *identification* (whether the recovered code is nearest the right memory). A cue can identify a memory while completing it poorly, which is the honest description of a degraded recall and is invisible if only one is reported.

**`G` is invariant to occlusion.** Sweeping `--keep` from 1.0 down to 0.2 over 3 seeds leaves completion, identification and `G` all at 1.000 — a fifth of the active cells is enough to recover the code exactly. The mechanism is that `V` is normalised by the *active input count*, so `G` measures the fraction of surviving evidence that matches, not how much evidence there is. Contrast `noise_robustness`, where `G` falls steadily with corruption: **familiarity here is sensitive to wrong evidence and blind to missing evidence**, which is a precise and useful statement about what the scalar means.

The activation band matters and is left at its default `[0.0, 1.0]` — "any U input at all". A narrower band would find the MAC refusing to activate on a thinned cue, which looks like total recall failure but is the input being rejected as malformed.

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

### `backoff_modality`

Two input regions ("shape" and "colour") feed one MAC, which also has a lateral self-link, so both U and H signals exist. Items are learned with everything present, then recognised with one modality, the other, or neither.

**Reference codes come from a recognition pass, not from learning, and getting this wrong scored below chance.** Code selection on a *learning* frame is probabilistic — `SigmoidConfig::enabled` defaults true, so the CSA samples rather than taking max-V — while recognition is deterministic max-V, so the two need not agree. The item's code also depends on its lateral context, which is the lead-in frame's code, which is itself still moving while training continues. The first version stored the final training epoch's codes and reported 0.05 identification against 0.125 chance. Taking the reference from a full-input recognition pass also states the question properly: this is what the network answers when it can see everything, and the conditions ask how close it stays with less.

**Graceful degradation is real:** losing half the bottom-up input costs about five points.

| condition | identification | completion | mean `G` | silent frames |
|---|---|---|---|---|
| both | 0.812 | 0.454 | 0.096 | 0 |
| shape only | 0.738 | 0.449 | 0.096 | 0 |
| colour only | 0.767 | 0.461 | 0.096 | 0 |
| neither (H only) | 0.000 | 0.000 | 0.000 | **240** |

Chance is 0.125.

**The fourth row is the most useful thing this demo reports, and it is not what it was built to show.** The intent was that with both inputs silent every U-bearing case becomes unavailable and the chain falls to a case built on H alone. It does not. The MAC never activates: 240 silent frames, no code, no familiarity.

The cause is upstream of backoff. `resolve_activation_bands` in `src/net/build.rs` computes a region's activation band from its **U afferent input size only**, and floors the lower bound at `max(1, …)`, so a MAC is eligible only if at least one *U* feature is active. `activate_region` checks that eligibility before `recognize_version` is ever called. **An H-only backoff case is therefore unreachable by construction, however the chain is written.** Lateral context can modulate a recognition that bottom-up input has already licensed; it cannot license one by itself.

The H-only level is kept in the demo's chain deliberately, because its being unreachable is the finding — a chain without it could not distinguish "the case was never selected" from "the case was never offered". Worth knowing before designing anything that expects a region to run on context alone.

## What CI does

`cargo build --all-targets`, `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`, and then **runs every demo** at small budgets. Building is not enough — a demo that panics on step one still compiles.

**CI checks that a demo runs, never that it learns.** The gate is the exit code and nothing else. No metric value is asserted anywhere; the metrics step checks the record *format* and same-seed determinism rather than the numbers in it. Whether a configuration learns is an experiment, and experiments belong in an offline run with `--repeat` and `--sweep`, with someone reading the spread.

## Three findings about the library, collected

The demos were the first code to drive several paths, and three results are worth reading even if the demos themselves are not.

**Recurrent H links work.** `connect("region", "region")` gives both ends the same DAG height, so the link is typed H and reads the previous frame. Nothing in the test suite, examples or benches had ever connected a region to itself; the path wires correctly and learns (20.7% of H synapses touched after training in `sequence_recognition`).

**A backoff chain with a lower level can hide the thing you are measuring.** The canonical `HUD → {HU,UD} → U` chain reports G ≈ 1.0 for a novel *ordering* of familiar frames, because when the `{H,U}` case misses its threshold the chain falls back to `U` alone, and `U` alone is a perfect match on content. Separation drops from 0.073 to exactly 0.000. This is backoff working as designed; it is only a problem when the lower level can answer a question the higher level was asked.

**A MAC cannot activate on lateral context alone.** The activation band is computed from **U afferent input size only** and floored at `max(1, …)`, and eligibility is checked before the backoff chain is consulted. So an H-only case is unreachable however the chain is written — `backoff_modality` names one and it never fires in 240 attempts. Lateral context modulates a recognition that bottom-up input has licensed; it cannot license one.

None of the three is a defect, and none is written down anywhere else. All three would cost a day to rediscover.
