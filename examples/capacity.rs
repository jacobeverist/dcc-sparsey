// Capacity — plasticity as a finite resource, and what running out looks like.
//
// **Neither sibling port models this.** Sparsey stores no weights: a synapse carries
// a `(stiffness, timestamp)` pair and its effective value is a table lookup on how
// long ago the last pre-post coincidence was. Repeated coincidences walk stiffness
// up to permanent; and a bundle whose increased fraction passes the target region's
// saturation threshold **freezes for good** and stops learning altogether.
//
// So a network fills up. This demo stores patterns one at a time and watches three
// things move together: how many synapses have been touched, how many bundles have
// frozen, and whether the patterns stored *earliest* can still be recalled. That
// last one is the point — a memory that quietly stops learning while still reporting
// high familiarity for what it already knows is a specific and recognisable failure,
// and it is a developmental phenomenon rather than a bug.
//
//   cargo run --release --example capacity
//   cargo run --release --example capacity -- --sweep q=6,12,24 --repeat 3

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::{NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

use support::args::Args;
use support::env::patterns::PatternBook;
use support::metrics::{Recorder, Summary};
use support::probe::{capacity_stats, code_similarity, Capture};
use support::report::sparkline;
use support::rng::{Rng, STREAM_ENV};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("capacity", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn build_config(grid: u32, q: u32, k: u32, saturation: f32) -> NetworkConfig {
    use dcc_sparsey::config::SignalParams;
    let signal = SignalParams {
        saturation_threshold: saturation,
        ..Default::default()
    };

    NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(grid, grid).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(q, k)
                .persistence(1)
                .signal(dcc_sparsey::SynapseType::U, signal)
                .build(),
        )
        .connect("input", "l1")
        .build()
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let patterns: usize = args.get("patterns", 800);
    let grid: u32 = args.get("grid", 16);
    let q: u32 = args.get("q", 12);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 24);
    let saturation: f32 = args.get("saturation", 0.9);
    // How many of the earliest patterns are re-tested as the network fills.
    let cohort: usize = args.get("cohort", 16);
    let every: usize = args.get("every", 100);
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    let cells = (grid * grid) as usize;

    rec.config("patterns", patterns);
    rec.config("grid", grid);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("active", active);
    rec.config("saturation", saturation);
    rec.config("cohort", cohort);

    let mut env = Rng::stream(seed, STREAM_ENV);
    let book = PatternBook::generate(patterns, cells, active, &mut env);

    let config = build_config(grid, q, k, saturation);
    let mut net = SparseyNet::build(config.clone(), seed).expect("build network");
    let input = net.region_id("input").expect("input region");

    say!("Capacity — storing {patterns} patterns one at a time, seed {seed}");
    say!("  input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!("  a bundle freezes permanently once its increased fraction passes {saturation}");
    say!("  the first {cohort} patterns are re-tested as the network fills");
    say!();

    // The first `cohort` patterns are the ones re-tested, so their codes are kept.
    let mut cohort_codes: Vec<Vec<u32>> = Vec::with_capacity(cohort);

    let mut touched_trace: Vec<f32> = Vec::new();
    let mut recall_trace: Vec<f32> = Vec::new();
    let mut frozen_at: Option<usize> = None;
    let mut first_recall_loss: Option<usize> = None;

    let mut early_recall = f64::NAN;
    let mut final_g = f64::NAN;

    for i in 0..patterns {
        let mut cap = Capture::new();
        net.set_input(input, book.get(i)).expect("set input");
        net.do_frame_learn_rec(&mut cap);
        if i < cohort {
            cohort_codes.push(cap.first_code().unwrap_or(&[]).to_vec());
        }

        let stats = capacity_stats(&net);
        if frozen_at.is_none() && stats.frozen_bundles > 0 {
            frozen_at = Some(i + 1);
        }

        if (i + 1) % every == 0 || i + 1 == patterns {
            // Re-test the earliest cohort without learning, then resume.
            let (recall, g) = retest(&mut net, input, &book, &cohort_codes);
            early_recall = recall;
            final_g = g;

            touched_trace.push(stats.touched_fraction() as f32);
            recall_trace.push(recall as f32);

            if first_recall_loss.is_none() && recall < 0.9 {
                first_recall_loss = Some(i + 1);
            }

            rec.sample(
                i as u64 + 1,
                &[
                    ("touched_synapse_fraction", stats.touched_fraction()),
                    ("permanent_synapse_fraction", stats.permanent_fraction()),
                    ("frozen_bundle_fraction", stats.frozen_fraction()),
                    ("early_cohort_recall", recall),
                    ("early_cohort_g", g),
                ],
            );
            if !quiet {
                say!(
                    "  after {:>5} patterns | synapses touched {:>5.1}% | permanent {:>5.1}% | frozen bundles {:>5.1}% | first-{cohort} recall {:>5.1}% (G {:.3})",
                    i + 1,
                    stats.touched_fraction() * 100.0,
                    stats.permanent_fraction() * 100.0,
                    stats.frozen_fraction() * 100.0,
                    recall * 100.0,
                    g
                );
            }
        }
    }

    // --- Report ---

    let stats = capacity_stats(&net);

    say!("\n  synapses touched over the run:  {}", sparkline(&touched_trace));
    say!("  recall of the first {cohort}:        {}", sparkline(&recall_trace));

    say!("\n  bundles                {:>8}", stats.bundles);
    say!("  frozen                 {:>8}  ({:.1}%)", stats.frozen_bundles, stats.frozen_fraction() * 100.0);
    say!("  synapses               {:>8}", stats.synapses);
    say!("  touched at least once  {:>8}  ({:.1}%)", stats.touched_synapses, stats.touched_fraction() * 100.0);
    say!("  permanent              {:>8}  ({:.1}%)", stats.permanent_synapses, stats.permanent_fraction() * 100.0);

    match frozen_at {
        Some(n) => say!("\n  first bundle froze after {n} patterns"),
        None => say!("\n  no bundle froze — the network did not reach saturation in {patterns} patterns"),
    }
    match first_recall_loss {
        Some(n) => say!("  recall of the first {cohort} fell below 90% after {n} patterns"),
        None => say!("  recall of the first {cohort} never fell below 90%"),
    }

    let mut summary = Summary::new();
    summary.push("patterns", patterns as f64);
    summary.push("early_cohort_recall", early_recall);
    summary.push("early_cohort_g", final_g);
    summary.push("touched_synapse_fraction", stats.touched_fraction());
    summary.push("permanent_synapse_fraction", stats.permanent_fraction());
    summary.push("frozen_bundle_fraction", stats.frozen_fraction());
    summary.push("frozen_after", frozen_at.map(|n| n as f64).unwrap_or(f64::NAN));
    summary.push(
        "recall_lost_after",
        first_recall_loss.map(|n| n as f64).unwrap_or(f64::NAN),
    );
    // The baseline is what an unsaturated network manages: recall right after
    // storing, before capacity is a factor. Without it "recall fell to 60%" cannot
    // be told apart from "recall was never better than 60%".
    summary.push("baseline_initial_recall", recall_trace.first().copied().unwrap_or(f32::NAN) as f64);
    summary.push(
        "recall_vs_initial",
        early_recall / recall_trace.first().copied().unwrap_or(f32::NAN) as f64,
    );

    // **A negative result here is a correct result.** This demo does not claim the
    // network learns something; it claims capacity is finite and observable. If
    // nothing saturated within `--patterns`, that is a true measurement of this
    // configuration and more training cannot change it — so the verdict is `true`
    // with a note saying so, per the contract in doc/Demos.md.
    let note = if stats.frozen_fraction() > 0.0 {
        format!(
            "capacity is finite and observable: {:.0}% of bundles frozen after {patterns} patterns, {:.0}% of synapses touched, and recall of the first {cohort} at {:.0}%",
            stats.frozen_fraction() * 100.0,
            stats.touched_fraction() * 100.0,
            early_recall * 100.0
        )
    } else {
        format!(
            "no saturation in {patterns} patterns — {:.0}% of synapses touched and {:.0}% permanent, with recall of the first {cohort} at {:.0}%. Try more --patterns or a lower --saturation",
            stats.touched_fraction() * 100.0,
            stats.permanent_fraction() * 100.0,
            early_recall * 100.0
        )
    };
    say!("\nLearned: {note}.");
    summary.verdict(true, note);

    rec.finish_summary(&summary);
    summary
}

/// Re-present the earliest patterns without learning and see whether their codes
/// still come back.
///
/// `prepare_for_new_run(false)` keeps the weights and resets the run state, which is
/// what makes this a probe rather than another training pass.
fn retest(
    net: &mut SparseyNet,
    input: dcc_sparsey::ids::RegionId,
    book: &PatternBook,
    codes: &[Vec<u32>],
) -> (f64, f64) {
    if codes.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    net.prepare_for_new_run(false);

    let mut sim_sum = 0.0;
    let mut g_sum = 0.0;
    let mut n = 0.0;

    for (i, stored) in codes.iter().enumerate() {
        let mut cap = Capture::new();
        net.set_input(input, book.get(i)).expect("set input");
        net.do_frame_recognize_rec(&mut cap);
        if let Some(code) = cap.first_code() {
            sim_sum += code_similarity(code, stored);
            g_sum += cap.mean_g().unwrap_or(0.0) as f64;
            n += 1.0;
        }
    }

    if n == 0.0 {
        return (0.0, 0.0);
    }
    (sim_sum / n, g_sum / n)
}
