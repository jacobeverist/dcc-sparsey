// Classify Stream — supervised readout from a memory that has no classifier.
//
// One of the three demos the cross-repo demo contract asks every port to implement;
// see `doc/Demos.md`.
//
// **This crate has no classification head.** Java's `supervisedMode` and its
// class-input region are not ported (`doc/PortNotes.md`), so there is nothing here
// that takes a label. The readout is done in the demo instead: learn labelled
// patterns, keep the code each one produced, and at test time answer with the label
// of whichever stored code the network's response most resembles.
//
// That is not a workaround so much as the thing itself. Sparsey's output *is* a
// content-addressable key, so classification is a lookup over codes rather than a
// trained head — which is why it needs no gradient, no epochs and no held-out
// tuning, and why it generalises to a corrupted input at all. The cost is that the
// codebook grows with the number of stored items, and a class is only as separable
// as its code.
//
// Several patterns per class, so the class has to be recovered from a *family* of
// inputs rather than from one memorised exemplar — otherwise this measures storage
// and not classification.
//
//   cargo run --release --example classify_stream
//   cargo run --release --example classify_stream -- --sweep per-class=1,3,6 --repeat 3

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::{NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

use support::args::Args;
use support::checkpoint;
use support::env::patterns::{corrupt, PatternBook};
use support::metrics::{Recorder, Summary};
use support::probe::{code_similarity, Capture};
use support::report::confusion_table;
use support::rng::{Rng, STREAM_ENV, STREAM_EVAL};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("classify_stream", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn build_config(grid: u32, q: u32, k: u32) -> NetworkConfig {
    NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(grid, grid).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(q, k)
                .persistence(1)
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build()
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let classes: usize = args.get("classes", 6);
    // Exemplars per class. Each is a corrupted variant of the class prototype, so a
    // class is a neighbourhood in input space rather than a single point.
    let per_class: usize = args.get("per-class", 4);
    let spread: f32 = args.get("spread", 0.2);
    let grid: u32 = args.get("grid", 16);
    let q: u32 = args.get("q", 12);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 24);
    let test_reps: usize = args.get("test-reps", 30);
    let silent = args.flag("silent");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    assert!(classes >= 2, "--classes must be at least 2");
    let cells = (grid * grid) as usize;

    rec.config("classes", classes);
    rec.config("per_class", per_class);
    rec.config("spread", spread);
    rec.config("grid", grid);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("test_reps", test_reps);

    let mut env = Rng::stream(seed, STREAM_ENV);
    let prototypes = PatternBook::generate(classes, cells, active, &mut env);

    let config = build_config(grid, q, k);
    let mut net = SparseyNet::build(config.clone(), seed).expect("build network");
    checkpoint::maybe_load(&mut net, &config, seed, args);
    let input = net.region_id("input").expect("input region");

    say!("Classify Stream — {classes} classes x {per_class} exemplars, seed {seed}");
    say!("  input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!("  exemplars are the class prototype corrupted by {:.0}%, so a class is a neighbourhood", spread * 100.0);
    say!("  closest two prototypes share {} of {active} active cells", prototypes.max_overlap());
    say!("  there is no classifier: the readout is a lookup over stored codes");
    say!();

    // --- Learning ---
    //
    // Every exemplar is stored with its class label. The codebook is the classifier.

    let mut codebook: Vec<(Vec<u32>, usize)> = Vec::with_capacity(classes * per_class);
    for c in 0..classes {
        for _ in 0..per_class {
            let exemplar = corrupt(prototypes.get(c), spread, cells, &mut env);
            let mut cap = Capture::new();
            net.set_input(input, &exemplar).expect("set input");
            net.do_frame_learn_rec(&mut cap);
            if let Some(code) = cap.first_code() {
                codebook.push((code.to_vec(), c));
            }
        }
    }
    net.finalize_learning();

    say!("Stored {} exemplars in one pass each.", codebook.len());

    // --- Classification ---
    //
    // Held-out exemplars: freshly drawn variants the network has never seen, so this
    // measures the neighbourhood rather than the stored points.

    let mut eval = Rng::stream(seed, STREAM_EVAL);
    net.prepare_for_new_run(false);

    let mut confusion = vec![vec![0u64; classes]; classes];
    let mut scored = 0u64;
    let mut silent_frames = 0u64;
    let mut g_sum = 0.0f64;

    for _ in 0..test_reps {
        for (c, row) in confusion.iter_mut().enumerate() {
            let probe = corrupt(prototypes.get(c), spread, cells, &mut eval);
            let mut cap = Capture::new();
            net.set_input(input, &probe).expect("set input");
            net.do_frame_recognize_rec(&mut cap);

            let Some(code) = cap.first_code() else {
                // Outside the activation band — a different event from a wrong
                // answer, and counted separately so it cannot hide in the accuracy.
                silent_frames += 1;
                continue;
            };

            let mut best = 0usize;
            let mut best_sim = f64::NEG_INFINITY;
            for (stored, label) in &codebook {
                let sim = code_similarity(code, stored);
                if sim > best_sim {
                    best_sim = sim;
                    best = *label;
                }
            }
            row[best] += 1;
            g_sum += cap.mean_g().unwrap_or(0.0) as f64;
            scored += 1;
        }
    }

    // --- Report ---

    let labels: Vec<String> = (0..classes).map(|c| format!("c{c}")).collect();
    let chance = 1.0 / classes as f64;
    let accuracy = if scored == 0 {
        f64::NAN
    } else {
        (0..classes).map(|i| confusion[i][i]).sum::<u64>() as f64 / scored as f64
    };

    say!("Held-out exemplars (rows true, columns recalled):");
    say!("{}", confusion_table(&confusion, &labels));
    say!("  accuracy          {:>6.1}%", accuracy * 100.0);
    say!("  chance            {:>6.1}%", chance * 100.0);
    say!("  mean G            {:>6.3}", g_sum / scored.max(1) as f64);
    if silent_frames > 0 {
        say!("  frames where the MAC did not activate: {silent_frames}");
    }

    let mut summary = Summary::new();
    summary.push("accuracy", accuracy);
    summary.push("g", g_sum / scored.max(1) as f64);
    summary.push("codebook_size", codebook.len() as f64);
    summary.push("silent_frames", silent_frames as f64);
    // Classes are presented in equal numbers, so the majority-class baseline is
    // chance. Both are recorded because they only coincide by construction here,
    // and a future unbalanced variant would separate them.
    summary.push("baseline_chance", chance);
    summary.push("baseline_majority", chance);
    summary.push("accuracy_vs_chance", accuracy / chance);
    summary.push("max_prototype_overlap", prototypes.max_overlap() as f64);
    for (c, row) in confusion.iter().enumerate() {
        let total: u64 = row.iter().sum();
        summary.push(
            &format!("recall_c{c}"),
            if total == 0 { f64::NAN } else { row[c] as f64 / total as f64 },
        );
    }

    if accuracy > chance * 2.0 {
        let note = format!(
            "held-out exemplars classified at {:.0}% against {:.0}% chance, by nearest stored code and nothing else",
            accuracy * 100.0,
            chance * 100.0
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: accuracy is near chance — try a smaller --spread, more --per-class, or a larger --q.");
        summary.verdict(false, "accuracy is near chance");
    }

    checkpoint::maybe_save(&net, &config, seed, args);
    rec.finish_summary(&summary);
    summary
}
