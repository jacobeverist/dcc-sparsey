//! The probabilistic Code Selection Algorithm (CSA), opt-in via `enable_csa()`.
//!
//! Verifies that the CSA path (learning samples winners from the `V→mu→rho`
//! distribution, spread by expansivity `eta(G)`) runs, is deterministic under a fixed
//! seed, is actually exercised (differs from max-V), and preserves the behavioral
//! invariants a faithful Sparsey exhibits. See `doc/AlgorithmTriangulation.md`.

use dcc_sparsey::config::{BackoffConfig, SigmoidConfig};
use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{NetworkConfigBuilder, Recorder, RegionConfigBuilder, SparseyNet};

const Q: u32 = 5;
const K: u32 = 8;
const ALPHABET: &[&[u32]] = &[&[0, 1, 2], &[6, 7, 8], &[11, 12, 13], &[3, 4, 14]];
const NOVEL: &[u32] = &[5, 9, 10, 15];

#[derive(Default)]
struct Capture {
    last: Option<(Vec<u32>, f32)>,
}
impl Recorder for Capture {
    fn on_code_selected(&mut self, _r: RegionId, _m: MacId, code: &[u32], g: f32, _f: i64) {
        self.last = Some((code.to_vec(), g));
    }
}

fn net(csa: bool) -> SparseyNet {
    net_seeded(csa, 42)
}

fn net_seeded(csa: bool, seed: u64) -> SparseyNet {
    let internal = {
        let b = RegionConfigBuilder::new("internal", 1)
            .grid(1, 1)
            .qk(Q, K)
            .persistence(1)
            .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96));
        if csa {
            b.enable_csa()
        } else {
            b.disable_csa()
        }
        .build()
    };
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(internal)
        .connect("input", "internal")
        .build();
    SparseyNet::build(cfg, seed).expect("build net")
}

/// Learn the alphabet; return the per-input learned codes.
fn learn_codes(net: &mut SparseyNet) -> Vec<Vec<u32>> {
    let input = net.region_id("input").unwrap();
    ALPHABET
        .iter()
        .map(|feats| {
            net.set_input(input, feats).unwrap();
            let mut rec = Capture::default();
            net.do_frame_learn_rec(&mut rec);
            rec.last.take().unwrap().0
        })
        .collect()
}

fn distinct(codes: &[Vec<u32>]) -> usize {
    let mut s: Vec<&Vec<u32>> = Vec::new();
    for c in codes {
        if !s.contains(&c) {
            s.push(c);
        }
    }
    s.len()
}

#[test]
fn csa_produces_valid_codes() {
    let codes = learn_codes(&mut net(true));
    assert_eq!(codes.len(), ALPHABET.len());
    for c in &codes {
        assert_eq!(c.len(), Q as usize, "one winner per CM");
        assert!(c.iter().all(|&w| w < K), "winner indices in range");
    }
}

#[test]
fn csa_is_deterministic() {
    assert_eq!(
        learn_codes(&mut net(true)),
        learn_codes(&mut net(true)),
        "same seed must give identical CSA codes",
    );
}

#[test]
fn csa_path_differs_from_max_v() {
    // With CSA on, learning SAMPLES winners from the V→mu→rho distribution; with it
    // off, it takes the argmax. The property worth pinning is that the two are not
    // silently equivalent — that the CSA branch is really taken.
    //
    // Checked across several seeds, deliberately. This asserted `csa(42) != max_v(42)`
    // for a single seed, which is a COINCIDENCE rather than a property: under rand 0.8's
    // sampling the two disagreed, and under 0.9's the sampled winner happened to equal
    // the argmax for every letter of this alphabet. The branch was still being taken;
    // the proxy for "was it taken" was just fragile enough that an RNG upgrade broke it.
    //
    // Not expressible as "max-V is seed-invariant", which would be the cleaner test:
    // the seed also drives initial connectivity (`build.rs`), so max-V codes move with
    // it too, for a reason that has nothing to do with winner selection.
    //
    // It must also run on a network whose V values have DIFFERENTIATED. On a fresh net
    // every cell in a CM is tied, and `max_v_winner` breaks ties "uniformly with the
    // seeded RNG" — so both paths collapse to the same thing: one uniform pick from one
    // stream. Under rand 0.8 they happened to land differently; under 0.9,
    // `random::<f32>()` and `random_range` derive from the same high bits, so a tied CM
    // makes them agree exactly, for every seed. Hence the warm-up below: once V differs
    // across cells, max-V is pinned to the argmax while CSA can sample away from it,
    // which is the distinction this test exists to prove.
    let seeds = [42u64, 7, 1234, 99];
    let differs = seeds.iter().any(|&s| {
        warm_then_learn(&mut net_seeded(true, s)) != warm_then_learn(&mut net_seeded(false, s))
    });
    assert!(
        differs,
        "CSA learning matched max-V learning for every seed in {seeds:?} — \
         the sampling branch is probably not being taken at all",
    );
}

/// Learn the alphabet several times so cell V values differentiate, then return the
/// codes from a final pass. A single pass over a fresh net leaves every CM tied, where
/// max-V's tie-break and CSA's sampling are indistinguishable — see the caller.
fn warm_then_learn(net: &mut SparseyNet) -> Vec<Vec<u32>> {
    let input = net.region_id("input").unwrap();
    for _ in 0..4 {
        for feats in ALPHABET {
            net.set_input(input, feats).unwrap();
            net.do_frame_learn();
        }
    }
    learn_codes(net)
}

#[test]
fn inflection_ratchets_with_saturation() {
    // The CSA sigmoid's inflection point ratchets rightward as a MAC saturates
    // (SparseyCore Mac.determine_Inflection_Point). With the mean-V threshold set to 0,
    // learning drives the ratchet; the inflection must move above its min and stay
    // capped at max.
    let internal = RegionConfigBuilder::new("internal", 1)
        .grid(1, 1)
        .qk(Q, K)
        .persistence(1)
        .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
        .sigmoid(SigmoidConfig {
            enabled: true,
            mean_v_ave_threshold: 0.0,
            ..Default::default()
        })
        .build();
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
        .region(internal)
        .connect("input", "internal")
        .build();
    let mut net = SparseyNet::build(cfg, 42).unwrap();
    let input = net.region_id("input").unwrap();

    let start = net.macs[0].inflect_point;
    assert!((start - 0.5).abs() < 1e-6, "inflection starts at min_inflect");
    for i in 0..24 {
        net.set_input(input, ALPHABET[i % ALPHABET.len()]).unwrap();
        net.do_frame_learn();
    }
    let end = net.macs[0].inflect_point;
    assert!(end > start, "inflection should ratchet right as the MAC saturates ({start} → {end})");
    assert!(end <= 0.9 + 1e-6, "inflection capped at max_inflect");
}

#[test]
fn csa_preserves_behavioral_invariants() {
    // Same invariants as fidelity_behavioral.rs, now with the CSA enabled.
    let mut n = net(true);
    let input = n.region_id("input").unwrap();

    let learned = learn_codes(&mut n);
    n.finalize_learning();
    n.prepare_for_new_run(false);

    // Recognition (always max-V) reactivates each learned code; G high for familiar.
    let mut g_fam_min = f32::INFINITY;
    for (feats, learned_code) in ALPHABET.iter().zip(&learned) {
        n.set_input(input, feats).unwrap();
        let mut rec = Capture::default();
        n.do_frame_recognize_rec(&mut rec);
        let (code, g) = rec.last.take().unwrap();
        assert_eq!(&code, learned_code, "learned input must reactivate its code");
        g_fam_min = g_fam_min.min(g);
    }
    assert!(g_fam_min >= 0.99, "G for familiar inputs ~1.0, got {g_fam_min}");

    // Distinct inputs → distinct codes.
    assert_eq!(distinct(&learned), ALPHABET.len(), "codes {learned:?}");

    // G low for a novel input.
    n.set_input(input, NOVEL).unwrap();
    let mut rec = Capture::default();
    n.do_frame_recognize_rec(&mut rec);
    let g_novel = rec.last.take().unwrap().1;
    assert!(g_novel < g_fam_min, "G_novel {g_novel} < G_familiar_min {g_fam_min}");
}
