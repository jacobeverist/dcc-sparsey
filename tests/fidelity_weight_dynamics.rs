//! Functional-fidelity cross-check: synapse weight dynamics vs. upstream Java.
//!
//! This is the **highest-confidence** fidelity surface for the sparsey port. The
//! implicit-decay weight table (`WeightTable`) and the per-synapse stiffness /
//! pre-post promotion logic (`Synapse`) are integer-valued and **RNG-free**, and the
//! governing constants are *identical* to SparseyCore's `Synapse.java` (verified
//! against the upstream checkout at commit `a0d4d34`):
//!
//! ```text
//! Synapse.MAX_WEIGHT                 = 127
//! Synapse.MAX_SYNAPSE_AGE            = 30000
//! Synapse.MAX_SYNAPSE_STIFFNESS      = 2
//! Synapse.WT_TABLE_TRANSITION_INDEXES = {{2000,3000,4000},{4000,6000,8000}}
//! Synapse.WT_TABLE_WEIGHTS            = {{127,120,50},   {127,120,50}}
//! ```
//!
//! So unlike the *behavioral* Sparsey code comparison (which cannot be bit-exact —
//! different PRNG, deterministic max-V vs. the probabilistic sigmoid CSA, `f32` vs
//! Java `double`; see `fidelity_behavioral.rs` and `doc/Divergences.md`), the weight
//! trajectory below can be pinned to exact upstream numbers. The golden values are
//! hand-derived from `Network.createWeightTable()` (the step-function interpolation)
//! and `Synapse.getEffectiveValue` / `wasFreshlyIncreased` in the Java source.
//!
//! If these ever fail, either a constant drifted from upstream or the decay/promotion
//! algorithm changed — both are true fidelity regressions.

use dcc_sparsey::config::WeightTableConfig;
use dcc_sparsey::ids::NeuronId;
use dcc_sparsey::synapse::{Synapse, WeightTable, MAX_WEIGHT};

/// The upstream constants, restated here as the source of truth for this test so a
/// silent change to `WeightTableConfig::default()` is caught by the assertions below
/// rather than tracking it.
const JAVA_MAX_WEIGHT: u8 = 127;
const JAVA_MAX_SYNAPSE_AGE: u32 = 30000;
const JAVA_MAX_STIFFNESS: u8 = 2;
const JAVA_TRANSITIONS: [[u32; 3]; 2] = [[2000, 3000, 4000], [4000, 6000, 8000]];
const JAVA_WEIGHTS: [[u8; 3]; 2] = [[127, 120, 50], [127, 120, 50]];

fn table() -> WeightTable {
    WeightTable::build(&WeightTableConfig::default())
}

/// The crate defaults must equal the upstream `Synapse.java` constants verbatim.
#[test]
fn defaults_match_upstream_constants() {
    let cfg = WeightTableConfig::default();
    assert_eq!(MAX_WEIGHT, JAVA_MAX_WEIGHT, "MAX_WEIGHT drifted from Synapse.java");
    assert_eq!(cfg.max_synapse_age, JAVA_MAX_SYNAPSE_AGE, "MAX_SYNAPSE_AGE drifted");
    assert_eq!(cfg.max_stiffness, JAVA_MAX_STIFFNESS, "MAX_SYNAPSE_STIFFNESS drifted");
    assert_eq!(
        cfg.transition_indexes,
        JAVA_TRANSITIONS.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "WT_TABLE_TRANSITION_INDEXES drifted",
    );
    assert_eq!(
        cfg.weights,
        JAVA_WEIGHTS.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "WT_TABLE_WEIGHTS drifted",
    );
}

/// The full `(stiffness, age) -> weight` step function, sampled at every breakpoint
/// and its neighbours, must match the Java `createWeightTable()` interpolation
/// exactly. This is the golden decay curve.
#[test]
fn decay_step_function_matches_java_golden() {
    let wt = table();

    // Derive the expected weight for `(stiffness, age)` straight from the Java tables:
    // hold WT_TABLE_WEIGHTS[s][t] for every age strictly below TRANSITION_INDEXES[s][t];
    // ages at/after the last breakpoint decay to 0; virgin (age < 0) is 0.
    let expected = |s: usize, age: i64| -> u8 {
        if age < 0 {
            return 0;
        }
        let transitions = &JAVA_TRANSITIONS[s];
        let weights = &JAVA_WEIGHTS[s];
        for t in 0..transitions.len() {
            if (age as u32) < transitions[t] {
                return weights[t];
            }
        }
        0
    };

    // `.enumerate().take()` rather than a range: `s` is the stiffness VALUE (passed to
    // `weight`/`expected`, not merely an index), and this way the indexing is in bounds
    // by construction.
    for (s, transitions) in JAVA_TRANSITIONS.iter().enumerate().take(JAVA_MAX_STIFFNESS as usize) {
        // Sample around every breakpoint (one before, at, one after) plus the origin.
        let mut ages: Vec<i64> = vec![-1, 0];
        for &b in transitions {
            ages.push(b as i64 - 1);
            ages.push(b as i64);
            ages.push(b as i64 + 1);
        }
        for age in ages {
            assert_eq!(
                wt.weight(s as u8, age),
                expected(s, age),
                "weight(stiffness={s}, age={age}) diverged from Java golden",
            );
        }
    }
}

/// A single fresh coincidence then pure decay, sampled across stiffness-0's whole
/// life. Mirrors driving one Java `Synapse` with one pre-post then reading
/// `getEffectiveValue` at increasing frames.
#[test]
fn single_coincidence_decay_trajectory() {
    let wt = table();
    let mut s = Synapse::new(NeuronId(0));
    let t0 = 100i64; // arbitrary coincidence frame

    // Golden (frame_offset -> weight) pairs for stiffness 0.
    let golden: &[(i64, u8)] = &[
        (0, 127),
        (1999, 127),
        (2000, 120),
        (2999, 120),
        (3000, 50),
        (3999, 50),
        (4000, 0),
        (10_000, 0),
    ];

    s.record_pre_post(t0, &wt, 1);
    assert_eq!(s.stiffness, 0, "one coincidence must not change stiffness");
    for &(off, w) in golden {
        assert_eq!(
            s.effective_value(t0 + off, &wt),
            w,
            "decay weight at age {off} diverged from Java golden",
        );
    }
}

/// Two close coincidences promote stiffness (0->1), lengthening the decay schedule
/// to stiffness-1's slower curve. Mirrors `wasFreshlyIncreased` + the stiffness-1
/// row of the Java weight table.
#[test]
fn promotion_extends_decay_to_stiffer_row() {
    let wt = table();
    let mut s = Synapse::new(NeuronId(0));

    s.record_pre_post(0, &wt, 1);
    s.record_pre_post(500, &wt, 1); // age 500 < fresh_index(0)=2000 -> promote to 1
    assert_eq!(s.stiffness, 1, "close re-coincidence must promote to stiffness 1");

    // Now the synapse decays on stiffness-1's schedule, measured from frame 500.
    let golden: &[(i64, u8)] = &[
        (0, 127),
        (3999, 127),
        (4000, 120),
        (5999, 120),
        (6000, 50),
        (7999, 50),
        (8000, 0),
    ];
    for &(off, w) in golden {
        assert_eq!(
            s.effective_value(500 + off, &wt),
            w,
            "stiffness-1 decay weight at age {off} diverged from Java golden",
        );
    }
}

/// Third close coincidence promotes to `max_stiffness` = permanent; permanent
/// synapses never decay (`getEffectiveValue` returns MAX_WEIGHT for a permanent
/// synapse regardless of age).
#[test]
fn permanence_never_decays() {
    let wt = table();
    let mut s = Synapse::new(NeuronId(0));
    s.record_pre_post(0, &wt, 1);
    s.record_pre_post(500, &wt, 1); // -> stiffness 1
    s.record_pre_post(1000, &wt, 1); // age 500 < fresh_index(1)=4000 -> stiffness 2 = permanent
    assert_eq!(s.stiffness, JAVA_MAX_STIFFNESS);
    assert!(s.is_permanent(&wt));
    assert_eq!(s.effective_value(0, &wt), MAX_WEIGHT);
    assert_eq!(s.effective_value(1_000_000, &wt), MAX_WEIGHT, "permanent must not decay");
}

/// A coincidence past the fresh window does NOT promote — it merely refreshes the
/// timestamp (Java `wasFreshlyIncreased` is false once age >= transition[0]).
#[test]
fn stale_recoincidence_refreshes_without_promoting() {
    let wt = table();
    let mut s = Synapse::new(NeuronId(0));
    s.record_pre_post(0, &wt, 1);
    s.record_pre_post(2500, &wt, 1); // age 2500 >= fresh_index(0)=2000 -> no promotion
    assert_eq!(s.stiffness, 0);
    // But the timestamp advanced: weight is full again at the new coincidence frame.
    assert_eq!(s.effective_value(2500, &wt), 127);
    assert_eq!(s.effective_value(2500 + 4000, &wt), 0);
}
