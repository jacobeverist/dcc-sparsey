//! Demo 2 — Novelty / anomaly detection with the familiarity signal `G`.
//!
//! After learning a small "known" set of patterns, we stream a mix of known and
//! unseen patterns through the network in recognition mode and print the
//! familiarity `G` for each. High `G` ⇒ recognized; low `G` ⇒ novel/anomalous.
//! This is Sparsey's SDM behaving as a content-addressable memory.
//!
//! Run with:  cargo run --example novelty_detection

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

#[derive(Default)]
struct Capture {
    g: f32,
}
impl Recorder for Capture {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, _c: &[u32], g: f32, _f: i64) {
        self.g = g;
    }
}

fn build_net() -> SparseyNet {
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(5, 8)
                .persistence(1)
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build();
    SparseyNet::build(cfg, 7).expect("build network")
}

fn learn(net: &mut SparseyNet, input: RegionId, feats: &[u32]) {
    net.set_input(input, feats).unwrap();
    net.do_frame_learn();
}

fn familiarity(net: &mut SparseyNet, input: RegionId, feats: &[u32]) -> f32 {
    net.set_input(input, feats).unwrap();
    let mut cap = Capture::default();
    net.do_frame_recognize_rec(&mut cap);
    cap.g
}

fn main() {
    let mut net = build_net();
    let input = net.region_id("input").unwrap();

    // Four "known" patterns over the 16-cell (4×4) input grid.
    let known: [(&str, &[u32]); 4] = [
        ("row-0", &[0, 1, 2, 3]),
        ("row-3", &[12, 13, 14, 15]),
        ("col-0", &[0, 4, 8, 12]),
        ("diag", &[0, 5, 10, 15]),
    ];
    for (_, feats) in known {
        learn(&mut net, input, feats);
    }
    net.finalize_learning();
    net.prepare_for_new_run(false);

    // A test stream: some are exactly the known patterns, some are unseen. The
    // novel ones range from "shares no cells with any memory" to "partially
    // overlaps a memory" — watch G grade accordingly.
    let stream: [(&str, &[u32], bool); 6] = [
        ("row-0 (known)", &[0, 1, 2, 3], true),
        ("diag (known)", &[0, 5, 10, 15], true),
        ("col-0 (known)", &[0, 4, 8, 12], true),
        ("disjoint (novel)", &[6, 7, 9, 11], false), // shares no cell with any memory
        ("row-1 (novel)", &[4, 5, 6, 7], false),     // partial overlap
        ("mixed (novel)", &[5, 6, 9, 10], false),    // partial overlap
    ];

    let threshold = 0.75;
    println!("Familiarity G per input (threshold {threshold:.2} → 'known'):\n");
    println!("  {:<20} {:>6}   {:>8}   expected", "input", "G", "verdict");
    println!("  {}", "-".repeat(52));
    for (label, feats, is_known) in stream {
        let g = familiarity(&mut net, input, feats);
        let verdict = if g >= threshold { "KNOWN" } else { "novel" };
        let expected = if is_known { "known" } else { "novel" };
        let ok = (g >= threshold) == is_known;
        println!(
            "  {:<20} {:>6.3}   {:>8}   {} {}",
            label, g, verdict, expected, if ok { "✓" } else { "✗" }
        );
    }
    println!(
        "\nThe single G scalar separates memorized inputs from unseen ones — no \
         labels, no separate classifier. Note G is *graded*: a fully disjoint \
         input scores ~0, while an input that partially overlaps a memory scores \
         in between — Sparsey generalizes rather than memorizing verbatim."
    );
}
