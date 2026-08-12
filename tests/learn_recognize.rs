//! M1 acceptance: a hand-built tiny 2-region network that learns codes and
//! re-activates the learned code when the same input is recognized.
//!
//! Structure: an input region (2×2 = 4 feature cells) → one internal region with a
//! single MAC of Q=3 CMs × K=4 cells, connected by a U link. Because the MAC's code
//! is cleared at end-of-frame (persistence = 1), we capture it *during* the frame
//! via a `Recorder`, which also exercises the recorder seam.

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{
    NetworkConfigBuilder, OperationMode, Recorder, RegionConfigBuilder, SparseyNet,
};

/// A recorder that captures the last code each MAC selected.
#[derive(Default)]
struct CaptureRecorder {
    last: Option<(MacId, Vec<u32>, f32)>,
    codes_selected: usize,
}

impl Recorder for CaptureRecorder {
    fn on_code_selected(&mut self, _region: RegionId, mac: MacId, code: &[u32], g: f32, _frame: i64) {
        self.last = Some((mac, code.to_vec(), g));
        self.codes_selected += 1;
    }
}

const Q: u32 = 3;
const K: u32 = 4;

fn tiny_net() -> SparseyNet {
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
    SparseyNet::build(cfg, 42).expect("build tiny net")
}

/// Present `features` and return the code the single MAC selected this frame.
fn learn_frame(net: &mut SparseyNet, input: RegionId, features: &[u32]) -> Vec<u32> {
    net.set_input(input, features).unwrap();
    let mut rec = CaptureRecorder::default();
    net.do_frame_learn_rec(&mut rec);
    let (_mac, code, _g) = rec.last.expect("a code was selected during learning");
    code
}

fn recognize_frame(net: &mut SparseyNet, input: RegionId, features: &[u32]) -> (Vec<u32>, f32) {
    net.set_input(input, features).unwrap();
    let mut rec = CaptureRecorder::default();
    net.do_frame_recognize_rec(&mut rec);
    let (_mac, code, g) = rec.last.expect("a code was selected during recognition");
    (code, g)
}

#[test]
fn code_is_one_winner_per_cm() {
    let mut net = tiny_net();
    let input = net.region_id("input").unwrap();
    let code = learn_frame(&mut net, input, &[0, 3]);

    // Exactly one winner per CM, each a valid cell index.
    assert_eq!(code.len(), Q as usize, "code must have one entry per CM");
    for &w in &code {
        assert!(w < K, "winner index {w} out of range 0..{K}");
    }
}

#[test]
fn learned_input_reactivates_its_code() {
    let mut net = tiny_net();
    let input = net.region_id("input").unwrap();

    // Learn a code for input A.
    let code_a = learn_frame(&mut net, input, &[0, 1]);

    // End of learning: make learned weights permanent, then reset dynamic state.
    net.finalize_learning();
    net.prepare_for_new_run(false);

    // Recognizing the same input must re-activate exactly the learned code, and G
    // should be maximal (perfect familiarity).
    let (recognized, g) = recognize_frame(&mut net, input, &[0, 1]);
    assert_eq!(recognized, code_a, "recognition must reproduce the learned code");
    assert!(g >= 0.99, "G for a perfectly familiar input should be ~1.0, got {g}");
    assert_eq!(net.op_mode, OperationMode::Recognition);
}

#[test]
fn distinct_inputs_get_reproducible_codes() {
    let mut net = tiny_net();
    let input = net.region_id("input").unwrap();

    // Learn two disjoint inputs.
    let code_a = learn_frame(&mut net, input, &[0, 1]);
    let code_b = learn_frame(&mut net, input, &[2, 3]);

    net.finalize_learning();
    net.prepare_for_new_run(false);

    // Each input recognizes back to the code it was learned with.
    let (rec_a, _) = recognize_frame(&mut net, input, &[0, 1]);
    let (rec_b, _) = recognize_frame(&mut net, input, &[2, 3]);
    assert_eq!(rec_a, code_a, "input A must recognize to code A");
    assert_eq!(rec_b, code_b, "input B must recognize to code B");
}

#[test]
fn novel_input_activates_mac_and_forms_a_code() {
    let mut net = tiny_net();
    let input = net.region_id("input").unwrap();
    let mut rec = CaptureRecorder::default();
    net.set_input(input, &[1, 2]).unwrap();
    net.do_frame_learn_rec(&mut rec);
    // The single MAC should have activated and selected exactly one code.
    assert_eq!(rec.codes_selected, 1, "the one MAC should select exactly one code");
}
