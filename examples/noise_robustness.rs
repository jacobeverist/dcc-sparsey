// Noise Robustness — how much input corruption a learned memory survives.
//
// **This is the shared demo of the cross-repo contract**: `dcc-sph` and `dcc-htm`
// implement the same task, so the three can be put beside each other honestly — not
// *which scores higher*, but *what shape each one's degradation curve has*. See
// `doc/Demos.md`.
//
// A book of patterns is learned, one per class. At test time a fraction of each
// pattern's active cells is moved elsewhere and the network is asked again, in
// recognition mode. Two things are measured, and Sparsey is the only one of the
// three that can report both from the same frame:
//
//   * whether the *right code* comes back — the content-addressable answer, scored
//     as classification accuracy against the code stored at learning time;
//   * how familiar the network says the input *is* — the scalar `G`, which needs no
//     classifier, no labels and no threshold training.
//
// The control is exact-match lookup: perfect at zero noise, blind one cell off it.
// It is not a competitor to beat — it is the null hypothesis for the *shape* of the
// curve.
//
//   cargo run --release --example noise_robustness
//   cargo run --release --example noise_robustness -- --sweep noise=0,0.1,0.2,0.3,0.4,0.5 --repeat 5

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::{NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

use support::args::Args;
use support::checkpoint;
use support::env::patterns::{corrupt, LookupTable, PatternBook};
use support::metrics::{Recorder, Summary};
use support::probe::{capacity_stats, code_similarity, Capture};
use support::report::confusion_table;
use support::rng::{Rng, STREAM_ENV, STREAM_EVAL};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("noise_robustness", &args);
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
                // The backoff chain is what lets recognition fall back to a weaker
                // signal set when the strongest is not available. With one input
                // region there is only the U level to fall back to, but the
                // thresholds still govern how readily a partial match is accepted.
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build()
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let classes: usize = args.get("classes", 8);
    let grid: u32 = args.get("grid", 16);
    let q: u32 = args.get("q", 12);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 24);
    let test_reps: usize = args.get("test-reps", 40);
    let noise: f32 = args.get("noise", 0.25);
    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    assert!(classes >= 2, "--classes must be at least 2");
    assert!((0.0..=1.0).contains(&noise), "--noise must be in [0, 1]");

    let cells = (grid * grid) as usize;

    rec.config("classes", classes);
    rec.config("grid", grid);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("active", active);
    rec.config("test_reps", test_reps);
    rec.config("noise", noise);

    let mut env = Rng::stream(seed, STREAM_ENV);
    let book = PatternBook::generate(classes, cells, active, &mut env);

    let mut lookup = LookupTable::new();
    for c in 0..book.len() {
        lookup.learn(book.get(c), c);
    }

    let config = build_config(grid, q, k);
    let mut net = SparseyNet::build(config.clone(), seed).expect("build network");
    checkpoint::maybe_load(&mut net, &config, seed, args);
    let input = net.region_id("input").expect("input region");

    say!("Noise Robustness — {classes} patterns, seed {seed}");
    say!("  input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!("  closest two patterns share {} of {active} active cells", book.max_overlap());
    say!("  testing at noise {noise:.2} — {:.0}% of active cells moved elsewhere", noise * 100.0);
    say!();

    // --- Learning: one frame per pattern ---
    //
    // Sparsey commits a code per frame — single-pass, no epochs — so this really is
    // one presentation each. The stored codes are what recognition is scored
    // against, so they are captured here rather than recomputed later.

    let mut stored: Vec<Vec<u32>> = Vec::with_capacity(classes);
    for c in 0..classes {
        let mut cap = Capture::new();
        net.set_input(input, book.get(c)).expect("set input");
        net.do_frame_learn_rec(&mut cap);
        stored.push(cap.first_code().unwrap_or(&[]).to_vec());
    }
    net.finalize_learning();

    let distinct = {
        let mut s = stored.clone();
        s.sort();
        s.dedup();
        s.len()
    };
    say!("Learned {classes} patterns in one pass each; {distinct} of them got distinct codes.");
    if distinct < classes {
        say!("  (Codes colliding means two patterns cannot be told apart however the rest goes.)");
    }

    // --- Recognition ---

    let mut eval = Rng::stream(seed, STREAM_EVAL);
    let clean = evaluate(&mut net, input, &book, &stored, &lookup, 0.0, cells, test_reps, &mut eval);
    let noisy = evaluate(&mut net, input, &book, &stored, &lookup, noise, cells, test_reps, &mut eval);

    // --- Report ---

    let labels: Vec<String> = (0..classes).map(|c| format!("p{c}")).collect();
    let chance = 1.0 / classes as f64;

    say!("\nClean (noise 0.00), rows true and columns recalled:");
    say!("{}", confusion_table(&clean.confusion, &labels));
    say!("  accuracy          {:>6.1}%", clean.accuracy() * 100.0);
    say!("  mean G            {:>6.3}", clean.mean_g);
    say!("  code similarity   {:>6.3}   (fraction of CMs matching the stored code)", clean.similarity);
    say!("  exact-match       {:>6.1}%", clean.lookup_accuracy() * 100.0);

    say!("\nCorrupted (noise {noise:.2}):");
    say!("{}", confusion_table(&noisy.confusion, &labels));
    say!("  accuracy          {:>6.1}%", noisy.accuracy() * 100.0);
    say!("  mean G            {:>6.3}", noisy.mean_g);
    say!("  code similarity   {:>6.3}", noisy.similarity);
    say!("  exact-match       {:>6.1}%", noisy.lookup_accuracy() * 100.0);
    say!("  chance            {:>6.1}%", chance * 100.0);

    let retention = if clean.accuracy() > 0.0 {
        noisy.accuracy() / clean.accuracy()
    } else {
        f64::NAN
    };
    say!("\n  retention         {:>6.1}% of clean accuracy survives noise {noise:.2}", retention * 100.0);

    let cap = capacity_stats(&net);

    let mut summary = Summary::new();
    summary.push("noise", noise as f64);
    summary.push("accuracy", noisy.accuracy());
    summary.push("clean_accuracy", clean.accuracy());
    summary.push("retention", retention);
    summary.push("g", noisy.mean_g);
    summary.push("clean_g", clean.mean_g);
    summary.push("code_similarity", noisy.similarity);
    summary.push("clean_code_similarity", clean.similarity);
    summary.push("baseline_lookup_accuracy", noisy.lookup_accuracy());
    summary.push("baseline_chance", chance);
    summary.push("accuracy_vs_chance", if chance > 0.0 { noisy.accuracy() / chance } else { f64::NAN });
    // No `accuracy_vs_lookup`: exact match is zero at every noise level above zero
    // *by construction*, so the ratio is either 0/0 or x/0.
    summary.push("distinct_codes", distinct as f64);
    summary.push("max_pattern_overlap", book.max_overlap() as f64);
    summary.push("touched_synapse_fraction", cap.touched_fraction());

    // The verdict keys on the *clean* pass. Failing to recall at heavy corruption is
    // the correct answer, not a failed run, and a verdict keyed on the noisy pass
    // would report "not converged" for most rows of the sweep this demo produces.
    if clean.accuracy() > chance * 2.0 {
        let note = format!(
            "patterns learned in one pass each (clean {:.0}% vs chance {:.0}%); {:.0}% of that survives noise {noise:.2}, against {:.0}% for exact match, and G falls {:.2} to {:.2} rather than collapsing",
            clean.accuracy() * 100.0,
            chance * 100.0,
            retention * 100.0,
            noisy.lookup_accuracy() * 100.0,
            clean.mean_g,
            noisy.mean_g
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: clean recall is near chance — the patterns were never separated, so the noise result says nothing. Try a larger --q or --k.");
        summary.verdict(false, "clean recall is near chance, so the noise result is uninformative");
    }

    checkpoint::maybe_save(&net, &config, seed, args);
    rec.finish_summary(&summary);
    summary
}

struct Eval {
    confusion: Vec<Vec<u64>>,
    lookup_hits: u64,
    scored: u64,
    mean_g: f64,
    similarity: f64,
}

impl Eval {
    fn accuracy(&self) -> f64 {
        if self.scored == 0 {
            return f64::NAN;
        }
        let correct: u64 = (0..self.confusion.len()).map(|i| self.confusion[i][i]).sum();
        correct as f64 / self.scored as f64
    }

    fn lookup_accuracy(&self) -> f64 {
        if self.scored == 0 {
            f64::NAN
        } else {
            self.lookup_hits as f64 / self.scored as f64
        }
    }
}

/// Present each pattern `reps` times at a given corruption level and score what
/// comes back.
///
/// Recognition is scored by nearest *stored code*: the network answers with a code,
/// and the class is whichever stored code it most resembles. That is the
/// content-addressable reading, and it is done here rather than in `src/` because
/// this crate has no classification head — Java's supervised mode is not ported.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    net: &mut SparseyNet,
    input: dcc_sparsey::ids::RegionId,
    book: &PatternBook,
    stored: &[Vec<u32>],
    lookup: &LookupTable,
    fraction: f32,
    cells: usize,
    reps: usize,
    rng: &mut Rng,
) -> Eval {
    let classes = book.len();
    let mut confusion = vec![vec![0u64; classes]; classes];
    let mut lookup_hits = 0u64;
    let mut scored = 0u64;
    let mut g_sum = 0.0f64;
    let mut sim_sum = 0.0f64;

    // Recognition must not learn, or the "test" quietly trains on its own probes.
    net.prepare_for_new_run(false);

    for _ in 0..reps {
        for class in 0..classes {
            let probe = corrupt(book.get(class), fraction, cells, rng);

            let mut cap = Capture::new();
            net.set_input(input, &probe).expect("set input");
            net.do_frame_recognize_rec(&mut cap);

            let Some(code) = cap.first_code() else {
                // The MAC did not activate at all — outside its activation band.
                // Not scored as a wrong answer, because it is a different event and
                // merging the two would hide it.
                continue;
            };

            let mut best = 0usize;
            let mut best_sim = f64::NEG_INFINITY;
            for (c, s) in stored.iter().enumerate() {
                let sim = code_similarity(code, s);
                if sim > best_sim {
                    best_sim = sim;
                    best = c;
                }
            }

            confusion[class][best] += 1;
            scored += 1;
            g_sum += cap.mean_g().unwrap_or(0.0) as f64;
            sim_sum += code_similarity(code, &stored[class]);

            if lookup.classify(&probe) == Some(class) {
                lookup_hits += 1;
            }
        }
    }

    let n = scored.max(1) as f64;
    Eval {
        confusion,
        lookup_hits,
        scored,
        mean_g: g_sum / n,
        similarity: sim_sum / n,
    }
}
