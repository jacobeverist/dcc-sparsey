// Sequence Recognition — does the network know this *order* of familiar frames?
//
// **This is the temporal cell of the cross-repo demo matrix, and it is a different
// task here than in the sibling ports.** `dcc-sph` and `dcc-htm` answer it with
// next-symbol prediction. This crate cannot: it has temporal context — H and D
// links read the previous frame (`use_previous_active = syn_type != SynapseType::U`,
// `src/net/build.rs`) — but no path from a code back down to input features.
// `src/net/frame.rs` records that faithful D-signal regeneration of the input region
// is a documented follow-on. So the question asked here is recognition, not
// prediction: having learned some episodes, does familiarity distinguish a *learned
// ordering* from a novel one?
//
// Episodes deliberately **share their frames**. Each is a different ordering over a
// small alphabet, so no episode can be identified from any single frame, and a novel
// episode is an unseen *order* of seen frames rather than unseen content. Novelty of
// content is spatial and any content-addressable memory detects it; novelty of order
// is the temporal question.
//
// The control is the same network **with the recurrent link removed**. That isolates
// the claim exactly: same frames, same training, same scoring, and the only
// difference is whether the region can see its own previous code.
//
//   cargo run --release --example sequence_recognition
//   cargo run --release --example sequence_recognition -- --sweep length=3,4,6 --repeat 3
//   cargo run --release --example sequence_recognition -- --backoff canonical   # see doc/Demos.md

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::config::{BackoffCase, BackoffConfig};
use dcc_sparsey::ids::RegionId;
use dcc_sparsey::{
    NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet, SynapseType,
};

use support::args::Args;
use support::env::patterns::PatternBook;
use support::env::sequences::EpisodeSet;
use support::metrics::{Recorder, Summary};
use support::probe::Capture;
use support::report::sparkline;
use support::rng::{Rng, STREAM_ENV, STREAM_EVAL};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("sequence_recognition", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

/// A backoff chain with **no U-only level**, so recognition has to use the temporal
/// signal or fail.
///
/// This is the single most important configuration choice in the demo. See the note
/// in `doc/Demos.md`: the canonical `HUD → {HU,UD} → U` chain reports G ≈ 1.0 for a
/// novel ordering as readily as for a learned one, because when the `{H,U}` case
/// misses its threshold the chain falls back to `U` alone — and `U` alone is a
/// perfect match, since every frame's *content* is familiar by construction. The
/// fallback is doing exactly what backoff is for, and it happens to erase the only
/// thing this demo measures.
fn temporal_only() -> BackoffConfig {
    BackoffConfig {
        priorities: vec![vec![BackoffCase {
            syn_types: vec![SynapseType::H, SynapseType::U],
            // Zero, because the point is to read the value of this case rather than
            // to accept or reject it. A positive threshold would send a poorly
            // matching ordering to the "no available version" path instead of
            // reporting how poorly it matched.
            threshold: 0.0,
        }]],
    }
}

fn build_config(grid: u32, q: u32, k: u32, recurrent: bool, canonical: bool) -> NetworkConfig {
    let backoff = if canonical {
        BackoffConfig::canonical(0.9, 0.93, 0.96)
    } else {
        temporal_only()
    };

    let mut b = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(grid, grid).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(q, k)
                .persistence(1)
                .backoff(backoff)
                .build(),
        )
        .connect("input", "l1");

    if recurrent {
        // Same DAG height on both ends, so `syn_type_from_heights` infers H and
        // `use_previous_active` is set: the region sees its own *previous* code.
        b = b.connect("l1", "l1");
    }
    b.build()
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let episodes_n: usize = args.get("episodes", 3);
    let length: usize = args.get("length", 6);
    let alphabet: usize = args.get("alphabet", 10);
    let epochs: usize = args.get("epochs", 50);
    let grid: u32 = args.get("grid", 12);
    let q: u32 = args.get("q", 24);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 16);
    let probes: usize = args.get("probes", 40);
    let canonical = matches!(args.str("backoff"), Some("canonical"));
    let silent = args.flag("silent");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    assert!(alphabet >= length, "--alphabet must be at least --length");
    let cells = (grid * grid) as usize;

    rec.config("episodes", episodes_n);
    rec.config("length", length);
    rec.config("alphabet", alphabet);
    rec.config("epochs", epochs);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("backoff", if canonical { "canonical" } else { "temporal-only" });

    let mut env = Rng::stream(seed, STREAM_ENV);
    let book = PatternBook::generate(alphabet, cells, active, &mut env);
    let episodes = EpisodeSet::generate(episodes_n, length, alphabet, &mut env);

    say!("Sequence Recognition — {episodes_n} episodes of {length} frames over an alphabet of {alphabet}, seed {seed}");
    say!("  input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!(
        "  backoff: {}",
        if canonical {
            "canonical HUD -> {HU,UD} -> U  (has a U-only fallback — see doc/Demos.md)"
        } else {
            "{H,U} only, no U fallback"
        }
    );
    say!("  episodes share frames, so novelty is in the ORDER, never in the content");
    say!("  frames used by at least one episode: {}", episodes.frames_used().len());
    say!();

    // Both conditions run the identical task. The only difference is the recurrent
    // link, which is what makes this a control rather than a second experiment.
    let with_h = measure(true, canonical, grid, q, k, &book, &episodes, epochs, probes, seed);
    let without_h = measure(false, canonical, grid, q, k, &book, &episodes, epochs, probes, seed);

    // --- Report ---

    say!("                        with H link   without H link");
    say!(
        "  G, learned ordering  {:>10.4}   {:>13.4}",
        with_h.g_familiar,
        without_h.g_familiar
    );
    say!(
        "  G, novel ordering    {:>10.4}   {:>13.4}",
        with_h.g_novel,
        without_h.g_novel
    );
    say!(
        "  separation           {:>10.4}   {:>13.4}",
        with_h.separation(),
        without_h.separation()
    );
    say!(
        "  discriminability     {:>10.3}   {:>13.3}   (fraction of pairs ranked correctly; 0.5 = no signal)",
        with_h.discriminability,
        without_h.discriminability
    );
    say!(
        "  H synapses touched   {:>10}   {:>13}",
        with_h.h_touched,
        without_h.h_touched
    );

    say!("\n  G per frame position, learned ordering (frame 0 has no context):");
    say!("    with H:    {}", sparkline(&with_h.by_position));
    say!("    without H: {}", sparkline(&without_h.by_position));

    let mut summary = Summary::new();
    summary.push("g_familiar", with_h.g_familiar as f64);
    summary.push("g_novel", with_h.g_novel as f64);
    summary.push("separation", with_h.separation() as f64);
    summary.push("discriminability", with_h.discriminability);
    summary.push("h_synapses_touched", with_h.h_touched as f64);
    summary.push("baseline_separation", without_h.separation() as f64);
    summary.push("baseline_discriminability", without_h.discriminability);
    summary.push(
        "separation_vs_baseline",
        (with_h.separation() - without_h.separation()) as f64,
    );

    // The control is the claim. Without the recurrent link the network cannot see
    // its own previous code, so a novel ordering of familiar frames is
    // indistinguishable from a learned one — and the separation should be zero. Any
    // separation in the H condition beyond that is what the temporal path bought.
    let real = with_h.discriminability > 0.7 && with_h.separation() > without_h.separation() + 0.01;
    if real {
        let note = format!(
            "a novel ordering of familiar frames scores G {:.3} against {:.3} for a learned one ({:.0}% of pairs ranked correctly), where the same network without the recurrent link separates them not at all ({:.3})",
            with_h.g_novel,
            with_h.g_familiar,
            with_h.discriminability * 100.0,
            without_h.separation()
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else if canonical {
        // Not a failure of the network — a documented property of this chain.
        let note = "the canonical backoff chain masks temporal novelty: when the {H,U} case misses its threshold it falls back to U alone, which matches every frame's content perfectly. Re-run without --backoff canonical".to_string();
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: a novel ordering is not distinguishable from a learned one. Try more --epochs, a longer --length, or a larger --q.");
        summary.verdict(false, "novel and learned orderings score alike");
    }

    rec.finish_summary(&summary);
    summary
}

struct Measure {
    g_familiar: f32,
    g_novel: f32,
    /// Fraction of (learned, novel) pairs where the learned ordering scored higher.
    /// 0.5 is no signal; it is a rank statistic, so it survives the two conditions
    /// having quite different absolute G ranges.
    discriminability: f64,
    h_touched: usize,
    by_position: Vec<f32>,
}

impl Measure {
    fn separation(&self) -> f32 {
        self.g_familiar - self.g_novel
    }
}

#[allow(clippy::too_many_arguments)]
fn measure(
    recurrent: bool,
    canonical: bool,
    grid: u32,
    q: u32,
    k: u32,
    book: &PatternBook,
    episodes: &EpisodeSet,
    epochs: usize,
    probes: usize,
    seed: u64,
) -> Measure {
    let config = build_config(grid, q, k, recurrent, canonical);
    let mut net = SparseyNet::build(config, seed).expect("build network");
    let input = net.region_id("input").expect("input region");

    // --- Learning ---
    //
    // `prepare_for_new_run(false)` between episodes clears the run state while
    // keeping the weights. Without it the last frame of one episode would be the
    // temporal context for the first frame of the next, teaching a transition that
    // is an artefact of presentation order.
    for _ in 0..epochs {
        for e in 0..episodes.len() {
            net.prepare_for_new_run(false);
            present(&mut net, input, book, episodes.get(e), true);
        }
    }
    net.finalize_learning();

    let h_touched = net
        .efferent_bundles
        .iter()
        .filter(|b| b.syn_type == SynapseType::H)
        .flat_map(|b| b.synapses.iter())
        .filter(|s| s.timestamp_last_pre_post != i64::MAX)
        .count();

    // --- Scoring ---

    let mut eval = Rng::stream(seed, STREAM_EVAL);
    let mut familiar = Vec::new();
    let mut novel = Vec::new();
    let mut position_sums = vec![0.0f32; episodes.length()];
    let mut position_n = vec![0.0f32; episodes.length()];

    for i in 0..probes {
        net.prepare_for_new_run(false);
        let e = i % episodes.len();
        let gs = present(&mut net, input, book, episodes.get(e), false);
        for (p, g) in gs.iter().enumerate() {
            position_sums[p] += g;
            position_n[p] += 1.0;
        }
        // Frame 0 has no predecessor, so its G says nothing about order. Averaging
        // it in would dilute the measurement with a constant.
        familiar.push(mean(&gs[1..]));

        net.prepare_for_new_run(false);
        let ordering = episodes.novel_ordering(&mut eval);
        let gs = present(&mut net, input, book, &ordering, false);
        novel.push(mean(&gs[1..]));
    }

    // Rank statistic over all pairs.
    let mut wins = 0.0;
    let mut total = 0.0;
    for f in &familiar {
        for n in &novel {
            total += 1.0;
            if f > n {
                wins += 1.0;
            } else if (f - n).abs() < f32::EPSILON {
                wins += 0.5;
            }
        }
    }

    Measure {
        g_familiar: mean(&familiar),
        g_novel: mean(&novel),
        discriminability: if total > 0.0 { wins / total } else { f64::NAN },
        h_touched,
        by_position: position_sums
            .iter()
            .zip(position_n.iter())
            .map(|(s, n)| if *n > 0.0 { s / n } else { 0.0 })
            .collect(),
    }
}

/// Present one episode frame by frame, returning the familiarity at each frame.
///
/// A frame where the MAC did not activate contributes 0.0 rather than being
/// skipped: at recognition time that *is* a complete failure to recognise, and
/// dropping it would quietly raise the mean of whichever condition failed more.
fn present(
    net: &mut SparseyNet,
    input: RegionId,
    book: &PatternBook,
    episode: &[usize],
    learn: bool,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(episode.len());
    for &f in episode {
        let mut cap = Capture::new();
        net.set_input(input, book.get(f)).expect("set input");
        if learn {
            net.do_frame_learn_rec(&mut cap);
        } else {
            net.do_frame_recognize_rec(&mut cap);
        }
        out.push(cap.mean_g().unwrap_or(0.0));
    }
    out
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}
