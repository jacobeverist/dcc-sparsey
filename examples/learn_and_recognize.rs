//! Demo 1 — Learn & Recognize (the "hello world" of Sparsey).
//!
//! Build a tiny network, teach it a few sparse input patterns, then show that
//! presenting a learned pattern again re-activates the *same* code with a high
//! familiarity signal `G ≈ 1.0` — while a never-seen pattern produces a fresh
//! code and a lower `G`.
//!
//! Run with:  cargo run --example learn_and_recognize

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

/// Captures the code (and familiarity G) the network's single MAC selected in a
/// frame. Sparsey clears a MAC's code at end-of-frame (persistence = 1), so we
/// read it *during* the frame through the recorder seam rather than after.
#[derive(Default)]
struct Capture {
    code: Option<Vec<u32>>,
    g: f32,
}

impl Recorder for Capture {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, code: &[u32], g: f32, _frame: i64) {
        self.code = Some(code.to_vec());
        self.g = g;
    }
}

fn build_net() -> SparseyNet {
    // 3×3 input grid (9 feature cells) → one internal region with a single MAC of
    // Q=4 competitive modules × K=6 cells. The MAC's code is one winner per CM.
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(3, 3).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(4, 6)
                .persistence(1)
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build();
    // The `42` is the RNG seed — wiring and tie-breaks are fully reproducible.
    SparseyNet::build(cfg, 42).expect("build network")
}

/// Present `features` (active input-cell indices) for one LEARNING frame and
/// return the code the MAC formed.
fn learn(net: &mut SparseyNet, input: RegionId, features: &[u32]) -> Vec<u32> {
    net.set_input(input, features).unwrap();
    let mut cap = Capture::default();
    net.do_frame_learn_rec(&mut cap);
    cap.code.expect("a code is selected while learning")
}

/// Present `features` for one RECOGNITION frame (no weight change) and return
/// the code plus the familiarity signal G (1.0 = perfectly familiar).
fn recognize(net: &mut SparseyNet, input: RegionId, features: &[u32]) -> (Vec<u32>, f32) {
    net.set_input(input, features).unwrap();
    let mut cap = Capture::default();
    net.do_frame_recognize_rec(&mut cap);
    (cap.code.expect("a code is selected while recognizing"), cap.g)
}

fn main() {
    let mut net = build_net();
    let input = net.region_id("input").unwrap();

    // Three distinct sparse patterns over the 9 input cells.
    let patterns: [(&str, &[u32]); 3] = [
        ("A", &[0, 4, 8]), // main diagonal
        ("B", &[2, 4, 6]), // anti-diagonal
        ("C", &[1, 3, 5]), // scattered
    ];

    println!("== Learning {} patterns ==", patterns.len());
    let mut learned = Vec::new();
    for (name, feats) in patterns {
        let code = learn(&mut net, input, feats);
        println!("  learn {name}: features {feats:?}  ->  code {code:?}");
        learned.push((name, feats, code));
    }

    // Lock in the learned weights and reset dynamic per-run state so we can
    // replay inputs in recognition mode.
    net.finalize_learning();
    net.prepare_for_new_run(false);

    println!("\n== Recognizing the learned patterns ==");
    for (name, feats, want) in &learned {
        let (got, g) = recognize(&mut net, input, feats);
        let mark = if got == *want { "✓ same code" } else { "✗ DIFFERENT" };
        println!("  see {name}: code {got:?}  G={g:.3}  {mark}");
    }

    println!("\n== Recognizing a NOVEL pattern ==");
    let novel: &[u32] = &[0, 1, 2]; // top row — never taught
    let (code, g) = recognize(&mut net, input, novel);
    println!("  see NEW: features {novel:?}  ->  code {code:?}  G={g:.3}");
    println!(
        "\nFamiliar inputs score G≈1.0; the novel input scores lower ({g:.3}) — \
         that G value is Sparsey's built-in novelty/familiarity detector."
    );
}
