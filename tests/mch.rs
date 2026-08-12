//! MCH (Multiple Concurrent Hypotheses) signal discounting. A source MAC representing
//! many concurrent hypotheses (high `num_mch`) has its efferent signal ignored (when
//! `num_mch ≥ MCH_ignore_threshold`) or discounted. Only applies to internal→internal
//! links (input features have no `num_mch`), so single-region nets are unaffected.

use dcc_sparsey::config::{BackoffConfig, SignalParams};
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

#[derive(Default)]
struct CountRec {
    codes: usize,
}
impl Recorder for CountRec {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, _c: &[u32], _g: f32, _f: i64) {
        self.codes += 1;
    }
}

/// input(2×2) → L1(1 MAC) → L2(1 MAC). L2's U-link MCH ignore threshold is `l2_ignore`.
/// Returns how many MACs activated (selected a code) on one learning frame.
fn active_macs(l2_ignore: f32) -> usize {
    let l2 = RegionConfigBuilder::new("l2", 2)
        .grid(1, 1)
        .qk(2, 4)
        .persistence(1)
        .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
        .signal(
            dcc_sparsey::types::SynapseType::U,
            SignalParams {
                mch_ignore_thresh: l2_ignore,
                ..Default::default()
            },
        )
        .build();
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(2, 2).build())
        .region(RegionConfigBuilder::new("l1", 1).grid(1, 1).qk(2, 4).persistence(1).build())
        .region(l2)
        .connect("input", "l1")
        .connect("l1", "l2")
        .build();
    let mut net = SparseyNet::build(cfg, 42).expect("build 3-region net");
    let input = net.region_id("input").unwrap();
    net.set_input(input, &[0, 3]).unwrap();
    let mut rec = CountRec::default();
    net.do_frame_learn_rec(&mut rec);
    rec.codes
}

#[test]
fn default_threshold_passes_signal() {
    // Default ignore threshold (1e6) never ignores → both L1 and L2 activate.
    assert_eq!(active_macs(1.0e6), 2, "L1 and L2 both activate by default");
}

#[test]
fn low_ignore_threshold_suppresses_source() {
    // L1's MAC has num_mch ≥ 1; an ignore threshold of 1.0 ignores its signal, so L2
    // receives no U input and stays inactive — only L1 activates.
    assert_eq!(active_macs(1.0), 1, "L2's ambiguous source is ignored → L2 inactive");
}
