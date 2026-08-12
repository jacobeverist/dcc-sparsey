//! Rust-side golden snapshot of sparsey coding behavior.
//!
//! Unlike `fidelity_weight_dynamics.rs` (which pins integer weight dynamics to exact
//! upstream Java constants) and `fidelity_behavioral.rs` (best-effort *structural*
//! comparison against a live Java run), this test locks in the **current Rust
//! behavior** of the implemented M1 subset — the per-MAC codes selected for a fixed
//! input sequence under a fixed seed. It is a regression guard: it catches
//! unintended drift in the Rust coding path even where a byte-exact upstream
//! comparison is impossible (different PRNG / selection algorithm; see
//! `doc/Divergences.md`).
//!
//! The scenario is the M1 acceptance topology (input 2×2 -> one MAC, Q=3, K=4,
//! persistence=1, full connectivity, deterministic max-V selection, seed 42), driven
//! by a fixed sequence of inputs through a learn phase then a recognition phase.
//!
//! To (re)generate the committed fixture after an *intended* behavior change:
//!
//! ```text
//! UPDATE_SNAPSHOTS=1 cargo test -p dcc_sparsey --test fidelity_snapshot
//! ```
//!
//! then review the diff to `tests/fixtures/sparsey_snapshot_golden.json` and commit it.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

const Q: u32 = 3;
const K: u32 = 4;

/// One captured frame: which input was presented and the code the MAC selected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FrameRecord {
    phase: String,
    features: Vec<u32>,
    code: Vec<u32>,
    /// Familiarity G, rounded to 3 dp so the snapshot is stable across trivial FP
    /// noise while still flagging real changes.
    g: f32,
}

#[derive(Default)]
struct CaptureRecorder {
    last: Option<(MacId, Vec<u32>, f32)>,
}

impl Recorder for CaptureRecorder {
    fn on_code_selected(&mut self, _r: RegionId, mac: MacId, code: &[u32], g: f32, _f: i64) {
        self.last = Some((mac, code.to_vec(), g));
    }
}

fn build_net() -> SparseyNet {
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(2, 2).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(Q, K)
                .persistence(1)
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build();
    SparseyNet::build(cfg, 42).expect("build snapshot net")
}

fn round3(x: f32) -> f32 {
    (x * 1000.0).round() / 1000.0
}

/// The fixed input alphabet presented to the network (2×2 input = cell indices 0..4).
const INPUTS: &[&[u32]] = &[&[0, 1], &[2, 3], &[0, 3], &[1, 2]];

/// Run the deterministic scenario and return the captured per-frame codes.
fn run_scenario() -> Vec<FrameRecord> {
    let mut net = build_net();
    let input = net.region_id("input").unwrap();
    let mut records = Vec::new();

    // Learn phase: present each input once, in order.
    for features in INPUTS {
        net.set_input(input, features).unwrap();
        let mut rec = CaptureRecorder::default();
        net.do_frame_learn_rec(&mut rec);
        let (_m, code, g) = rec.last.clone().expect("a code selected during learning");
        records.push(FrameRecord {
            phase: "learn".into(),
            features: features.to_vec(),
            code,
            g: round3(g),
        });
    }

    net.finalize_learning();
    net.prepare_for_new_run(false);

    // Recognition phase: present each input again, in order.
    for features in INPUTS {
        net.set_input(input, features).unwrap();
        let mut rec = CaptureRecorder::default();
        net.do_frame_recognize_rec(&mut rec);
        let (_m, code, g) = rec.last.clone().expect("a code selected during recognition");
        records.push(FrameRecord {
            phase: "recognize".into(),
            features: features.to_vec(),
            code,
            g: round3(g),
        });
    }

    records
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sparsey_snapshot_golden.json")
}

#[test]
fn coding_matches_committed_snapshot() {
    let actual = run_scenario();

    // Sanity: the scenario itself must stay well-formed regardless of the fixture.
    for r in &actual {
        assert_eq!(r.code.len(), Q as usize, "each code has one winner per CM");
        assert!(r.code.iter().all(|&w| w < K), "winner indices in range");
    }

    let path = fixture_path();
    let json = serde_json::to_string_pretty(&actual).unwrap();

    if std::env::var("UPDATE_SNAPSHOTS").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, format!("{json}\n")).unwrap();
        eprintln!("UPDATE_SNAPSHOTS: wrote {}", path.display());
        return;
    }

    let expected_str = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing snapshot fixture {}; regenerate with UPDATE_SNAPSHOTS=1",
            path.display()
        )
    });
    let expected: Vec<FrameRecord> = serde_json::from_str(&expected_str).unwrap();

    assert_eq!(
        actual, expected,
        "sparsey coding drifted from the committed snapshot; if intended, regenerate with UPDATE_SNAPSHOTS=1",
    );
}

/// Determinism guard: the scenario is fully reproducible from the fixed seed.
#[test]
fn scenario_is_deterministic() {
    assert_eq!(run_scenario(), run_scenario(), "same seed must give identical codes");
}
