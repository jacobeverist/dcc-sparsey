//! Activation criterion: a MAC is eligible only when its active U input-feature count
//! falls within the region's activation band `[low, high]` — fractions of the U
//! afferent input size (SparseyCore `ActiveInputFeatures{Low,High}BoundAsFrac`). The
//! default band `[0.0, 1.0]` reproduces the prior "any nonempty U input" behavior.

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

/// Counts codes selected — i.e. how many MACs activated this frame.
#[derive(Default)]
struct CountRec {
    codes: usize,
}
impl Recorder for CountRec {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, _c: &[u32], _g: f32, _f: i64) {
        self.codes += 1;
    }
}

/// 4×4 input (16 cells) → one internal MAC (Q=2, K=4) with the given activation band.
fn net_with_band(low: f32, high: f32) -> SparseyNet {
    let internal = RegionConfigBuilder::new("internal", 1)
        .grid(1, 1)
        .qk(2, 4)
        .persistence(1)
        .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
        .activation_band(low, high)
        .build();
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(internal)
        .connect("input", "internal")
        .build();
    SparseyNet::build(cfg, 42).expect("build net")
}

/// Present `feats` as a single learning frame on a fresh net; report whether the MAC
/// activated (selected a code). Fresh net per call ⇒ no persistence carryover.
fn active(low: f32, high: f32, feats: &[u32]) -> bool {
    let mut net = net_with_band(low, high);
    let input = net.region_id("input").unwrap();
    net.set_input(input, feats).unwrap();
    let mut rec = CountRec::default();
    net.do_frame_learn_rec(&mut rec);
    rec.codes > 0
}

#[test]
fn default_band_admits_any_nonempty_input() {
    // [0.0, 1.0] → resolved [1, MAX]: active iff ≥ 1 feature.
    assert!(active(0.0, 1.0, &[0]), "one feature is enough by default");
    assert!(active(0.0, 1.0, &[0, 1, 2, 3, 4, 5, 6, 7]), "dense input still admitted");
    assert!(!active(0.0, 1.0, &[]), "no input ⇒ inactive");
}

#[test]
fn band_rejects_too_sparse_input() {
    // u_input=16, band [0.25, 0.5] → [4, 8]. 2 active features < 4 ⇒ inactive.
    assert!(!active(0.25, 0.5, &[0, 1]));
}

#[test]
fn band_admits_in_range_input() {
    // 5 active features ∈ [4, 8] ⇒ active.
    assert!(active(0.25, 0.5, &[0, 1, 2, 3, 4]));
}

#[test]
fn band_rejects_too_dense_input() {
    // 12 active features > 8 ⇒ inactive.
    assert!(!active(0.25, 0.5, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]));
}
