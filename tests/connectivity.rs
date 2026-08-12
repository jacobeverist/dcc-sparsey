//! Band-limited projective-field connectivity (SparseyCore `buildBlockMatrix` /
//! `readBandInfo`). A source block connects to a target block only if their normalized
//! grid distance falls within a band; within a band, each candidate synapse is created
//! with that band's rate. Empty bands ⇒ full within-link connectivity (the default).

use dcc_sparsey::config::SignalParams;
use dcc_sparsey::types::SynapseType;
use dcc_sparsey::{NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

/// 4×4 input grid → one internal region of 4×4 MACs (Q=2, K=4), single U link.
/// `bands` sets the U-link band_thickness/band_rates on the internal region.
fn build(bands: Option<(Vec<f32>, Vec<f32>)>, seed: u64) -> SparseyNet {
    let mut internal = RegionConfigBuilder::new("internal", 1).grid(4, 4).qk(2, 4).persistence(1);
    if let Some((thickness, rates)) = bands {
        internal = internal.signal(
            SynapseType::U,
            SignalParams {
                band_thickness: thickness,
                band_rates: rates,
                ..Default::default()
            },
        );
    }
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(internal.build())
        .connect("input", "internal")
        .build();
    SparseyNet::build(cfg, seed).expect("build net")
}

fn total_synapses(net: &SparseyNet) -> usize {
    net.efferent_bundles.iter().map(|e| e.synapses.len()).sum()
}

/// FULL = 16 input cells × (16 MACs × 2 CMs × 4 cells) = 2048.
const FULL: usize = 16 * (16 * 2 * 4);

#[test]
fn empty_bands_is_full_connectivity() {
    assert_eq!(total_synapses(&build(None, 42)), FULL);
}

#[test]
fn wide_band_rate_one_equals_full() {
    // A single band whose outer radius (√2) covers the whole normalized grid, rate 1.0.
    let net = build(Some((vec![2.0], vec![1.0])), 42);
    assert_eq!(total_synapses(&net), FULL, "wide band @ rate 1.0 == full");
}

#[test]
fn narrow_band_limits_connectivity() {
    // Radius 0.2 < nearest-neighbour distance (0.25 on a 4×4 grid), so each input cell
    // connects only to the co-located MAC: 16 cells × (1 MAC × 8 cells) = 128.
    let net = build(Some((vec![0.2], vec![1.0])), 42);
    let n = total_synapses(&net);
    assert!(n > 0 && n < FULL, "narrow band must limit connectivity (got {n} of {FULL})");
    assert_eq!(n, 16 * 8, "each input cell wires only to the co-located MAC");
}

#[test]
fn rate_below_one_thins_and_is_deterministic() {
    // Wide band, rate 0.5 → ~half of FULL, reproducible from the seed.
    let a = build(Some((vec![2.0], vec![0.5])), 42);
    let b = build(Some((vec![2.0], vec![0.5])), 42);
    let na = total_synapses(&a);
    assert_eq!(na, total_synapses(&b), "same seed → identical band wiring");
    let frac = na as f32 / FULL as f32;
    assert!((0.35..0.65).contains(&frac), "rate 0.5 should thin to ~half (got {frac})");
}

#[test]
fn narrow_band_is_spatially_local() {
    // With a co-located-only band, every synapse connects blocks at the same grid
    // position, so the normalized block distance of every synapse is ~0.
    let net = build(Some((vec![0.2], vec![1.0])), 42);
    let w = 4u32;
    // normalized block center of an input cell (index within region) or internal cell.
    let center = |net: &SparseyNet, nid: usize| -> (f32, f32) {
        let n = &net.neurons[nid];
        let (col, row) = match n.owning_cm {
            Some(cm) => {
                let m = &net.macs[net.cms[cm.index()].owning_mac.index()];
                (m.col, m.row)
            }
            None => (n.index_within_region % w, n.index_within_region / w),
        };
        ((col as f32 + 0.5) / w as f32, (row as f32 + 0.5) / w as f32)
    };
    for eb in &net.efferent_bundles {
        let s = center(&net, eb.source_neuron.index());
        for syn in &eb.synapses {
            let t = center(&net, syn.target_neuron.index());
            let d = ((s.0 - t.0).powi(2) + (s.1 - t.1).powi(2)).sqrt();
            assert!(d <= 0.2, "synapse spans distance {d} > band radius");
        }
    }
}
