//! Demo 4 — Persist and restore learned memory.
//!
//! Train a network, serialize its learned state to a file, then build a fresh
//! network from the *same config + seed* and load the state back in. The restored
//! network recognizes the trained patterns identically — memory survives a
//! round-trip to disk.
//!
//! `serialize_state` stores only the learned weights (per-synapse stiffness +
//! timestamps), not the structure; the structure is rebuilt from the config, so
//! loader and saver must share the same `NetworkConfig` and RNG seed.
//!
//! Run with:  cargo run --example save_and_load

use std::env;
use std::fs;

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfig, NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

const SEED: u64 = 99;

#[derive(Default)]
struct Capture {
    code: Option<Vec<u32>>,
    g: f32,
}
impl Recorder for Capture {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, code: &[u32], g: f32, _f: i64) {
        self.code = Some(code.to_vec());
        self.g = g;
    }
}

fn config() -> NetworkConfig {
    NetworkConfigBuilder::default()
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
        .build()
}

fn recognize(net: &mut SparseyNet, input: RegionId, feats: &[u32]) -> (Vec<u32>, f32) {
    net.set_input(input, feats).unwrap();
    let mut cap = Capture::default();
    net.do_frame_recognize_rec(&mut cap);
    (cap.code.unwrap(), cap.g)
}

fn main() {
    let patterns: [&[u32]; 3] = [&[0, 4, 8], &[2, 4, 6], &[1, 3, 5]];

    // --- Train the original network and capture its recognized codes. ---
    let mut original = SparseyNet::build(config(), SEED).expect("build");
    let input = original.region_id("input").unwrap();
    for feats in patterns {
        original.set_input(input, feats).unwrap();
        original.do_frame_learn();
    }
    original.finalize_learning();
    original.prepare_for_new_run(false);

    let before: Vec<(Vec<u32>, f32)> = patterns
        .iter()
        .map(|f| recognize(&mut original, input, f))
        .collect();

    // --- Serialize learned state to a file. ---
    let bytes = original.serialize_state().expect("serialize");
    let path = env::temp_dir().join("sparsey_demo_state.bin");
    fs::write(&path, &bytes).expect("write state");
    println!(
        "Trained, then wrote {} bytes of learned state to {}\n",
        bytes.len(),
        path.display()
    );

    // --- Build a FRESH network and load the state back. ---
    let mut restored = SparseyNet::build(config(), SEED).expect("rebuild");
    let loaded = fs::read(&path).expect("read state");
    restored.load_state(&loaded).expect("load state");
    restored.prepare_for_new_run(false);
    let input2 = restored.region_id("input").unwrap();

    println!("Recognizing the trained patterns in the restored network:\n");
    println!("  {:<14} {:<18} {:<18} match", "pattern", "original code", "restored code");
    println!("  {}", "-".repeat(60));
    let mut all_match = true;
    for (i, feats) in patterns.iter().enumerate() {
        let (orig_code, _) = &before[i];
        let (code, g) = recognize(&mut restored, input2, feats);
        let ok = &code == orig_code;
        all_match &= ok;
        println!(
            "  {:<14} {:<18} {:<18} {} (G={:.3})",
            format!("{feats:?}"),
            format!("{orig_code:?}"),
            format!("{code:?}"),
            if ok { "✓" } else { "✗" },
            g
        );
    }

    // Clean up the demo artifact.
    let _ = fs::remove_file(&path);

    println!(
        "\n{}",
        if all_match {
            "All patterns recognized identically — learned memory round-tripped through disk."
        } else {
            "Mismatch! (unexpected)"
        }
    );
}
