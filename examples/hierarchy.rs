//! Demo 3 — A multi-region hierarchy.
//!
//! Sparsey networks are a DAG of regions. Here we stack three: a 4×4 input, a
//! middle region `l1` of 2×2 macrocolumns, and a top region `l2` of a single
//! macrocolumn. Each internal region forms its own sparse code every frame; the
//! codes higher up are progressively more abstract summaries of the input.
//!
//! Run with:  cargo run --example hierarchy

use std::collections::BTreeMap;

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

/// Collects, per region, the code each MAC selected this frame.
#[derive(Default)]
struct PerRegion {
    // region index -> list of (mac index, code)
    by_region: BTreeMap<usize, Vec<(usize, Vec<u32>)>>,
}
impl Recorder for PerRegion {
    fn on_code_selected(&mut self, r: RegionId, m: MacId, code: &[u32], _g: f32, _f: i64) {
        self.by_region
            .entry(r.index())
            .or_default()
            .push((m.index(), code.to_vec()));
    }
}

fn internal(name: &str, height: u32, w: u32, h: u32) -> dcc_sparsey::RegionConfig {
    RegionConfigBuilder::new(name, height)
        .grid(w, h)
        .qk(4, 6)
        .persistence(1)
        .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
        .build()
}

fn main() {
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(internal("l1", 1, 2, 2)) // 2×2 = 4 macrocolumns
        .region(internal("l2", 2, 1, 1)) // 1 macrocolumn at the top
        .connect("input", "l1")
        .connect("l1", "l2")
        .build();
    let mut net = SparseyNet::build(cfg, 1).expect("build hierarchy");

    let input = net.region_id("input").unwrap();
    // Map region index -> name for pretty printing.
    let names: BTreeMap<usize, &str> = ["input", "l1", "l2"]
        .iter()
        .map(|n| (net.region_id(n).unwrap().index(), *n))
        .collect();

    // Present two different inputs and learn a code at every level for each.
    let inputs: [(&str, &[u32]); 2] = [
        ("shape-P", &[0, 1, 5, 10, 15]),
        ("shape-Q", &[3, 6, 9, 12, 14]),
    ];

    for (label, feats) in inputs {
        net.set_input(input, feats).unwrap();
        let mut rec = PerRegion::default();
        net.do_frame_learn_rec(&mut rec);

        println!("input '{label}' = {feats:?}");
        for (ridx, macs) in &rec.by_region {
            let rname = names.get(ridx).copied().unwrap_or("?");
            let codes: Vec<String> = macs
                .iter()
                .map(|(mi, code)| format!("mac{mi}={code:?}"))
                .collect();
            println!("    {:<4} ({} MAC): {}", rname, macs.len(), codes.join("  "));
        }
        println!();
    }

    println!(
        "Every region emits one winner-per-CM code each frame. `l1` holds 4 \
         macrocolumns (one code each); `l2` fuses them into a single top-level code."
    );
}
