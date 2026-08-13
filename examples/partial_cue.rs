// Partial Cue — how little of a pattern brings back the whole code.
//
// The classical sparse-distributed-memory claim: a memory addressed by content
// should complete a partial address. Show the network a fraction of a stored
// pattern's active cells and ask whether the code it produces is the code that
// pattern produced when it was learned.
//
// This is **occlusion, not corruption**, and the distinction is the point.
// `noise_robustness` moves active cells elsewhere, so the evidence is wrong; here
// cells are simply dropped, so all the surviving evidence is correct and there is
// merely less of it. Those are different failure modes and a memory can be good at
// one and bad at the other — which is why they are separate demos rather than two
// settings of one.
//
// Two things are measured, and they can disagree:
//
//   * **completion** — how much of the stored code comes back, per CM;
//   * **identification** — whether the recovered code is nearer the right pattern's
//     stored code than any other's.
//
// A cue can identify the right memory while completing it poorly, which is the
// honest description of a degraded recall and is invisible if only one is reported.
//
//   cargo run --release --example partial_cue
//   cargo run --release --example partial_cue -- --sweep keep=1.0,0.8,0.6,0.4,0.2 --repeat 5

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::{NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

use support::args::Args;
use support::env::patterns::{occlude, LookupTable, PatternBook};
use support::metrics::{Recorder, Summary};
use support::probe::{code_similarity, Capture};
use support::report::ascii_bar;
use support::rng::{Rng, STREAM_ENV, STREAM_EVAL};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("partial_cue", &args);
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
                // The activation band defaults to [0.0, 1.0], i.e. "any U input at
                // all". That is what makes occlusion testable without further
                // configuration: a thinned cue still activates the MAC. A demo that
                // narrowed the band would find the MAC simply refusing to activate,
                // which looks like total recall failure but is the input being
                // rejected as malformed — a different finding entirely.
                .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                .build(),
        )
        .connect("input", "l1")
        .build()
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let patterns: usize = args.get("patterns", 8);
    let grid: u32 = args.get("grid", 16);
    let q: u32 = args.get("q", 12);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 24);
    let keep: f32 = args.get("keep", 0.5);
    let reps: usize = args.get("reps", 40);
    let silent = args.flag("silent");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    assert!((0.0..=1.0).contains(&keep), "--keep must be in [0, 1]");
    let cells = (grid * grid) as usize;

    rec.config("patterns", patterns);
    rec.config("grid", grid);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("active", active);
    rec.config("keep", keep);

    let mut env = Rng::stream(seed, STREAM_ENV);
    let book = PatternBook::generate(patterns, cells, active, &mut env);

    let mut lookup = LookupTable::new();
    for c in 0..book.len() {
        lookup.learn(book.get(c), c);
    }

    let config = build_config(grid, q, k);
    let mut net = SparseyNet::build(config.clone(), seed).expect("build network");
    let input = net.region_id("input").expect("input region");

    // --- Learning ---

    let mut stored: Vec<Vec<u32>> = Vec::with_capacity(patterns);
    for c in 0..patterns {
        let mut cap = Capture::new();
        net.set_input(input, book.get(c)).expect("set input");
        net.do_frame_learn_rec(&mut cap);
        stored.push(cap.first_code().unwrap_or(&[]).to_vec());
    }
    net.finalize_learning();

    let kept_cells = ((keep * active as f32) + 0.5) as usize;
    say!("Partial Cue — {patterns} patterns, seed {seed}");
    say!("  input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!("  cue keeps {:.0}% of the active cells — {kept_cells} of {active}, all of them correct", keep * 100.0);
    say!("  closest two patterns share {} active cells", book.max_overlap());
    say!();

    // --- Recall from partial cues ---

    let mut eval = Rng::stream(seed, STREAM_EVAL);
    net.prepare_for_new_run(false);

    let mut completion = 0.0f64;
    let mut identified = 0u64;
    let mut lookup_hits = 0u64;
    let mut g_sum = 0.0f64;
    let mut scored = 0u64;
    let mut silent_frames = 0u64;

    for _ in 0..reps {
        for c in 0..patterns {
            let cue = occlude(book.get(c), keep, &mut eval);

            let mut cap = Capture::new();
            net.set_input(input, &cue).expect("set input");
            net.do_frame_recognize_rec(&mut cap);

            let Some(code) = cap.first_code() else {
                silent_frames += 1;
                continue;
            };

            completion += code_similarity(code, &stored[c]);
            g_sum += cap.mean_g().unwrap_or(0.0) as f64;

            // Identification: is the recovered code nearest the right memory?
            let mut best = 0usize;
            let mut best_sim = f64::NEG_INFINITY;
            for (i, s) in stored.iter().enumerate() {
                let sim = code_similarity(code, s);
                if sim > best_sim {
                    best_sim = sim;
                    best = i;
                }
            }
            if best == c {
                identified += 1;
            }

            // The control, on exactly the same cue: exact match is blind to any cue
            // that is not the whole pattern, which is every cue with keep < 1.
            if lookup.classify(&cue) == Some(c) {
                lookup_hits += 1;
            }
            scored += 1;
        }
    }

    // --- Report ---

    let n = scored.max(1) as f64;
    let completion = completion / n;
    let identification = identified as f64 / n;
    let lookup_acc = lookup_hits as f64 / n;
    let g = g_sum / n;
    let chance = 1.0 / patterns as f64;

    say!("  completion     {:>6.3}  {}  (fraction of the stored code's CMs recovered)", completion, ascii_bar(completion as f32));
    say!("  identification {:>6.3}  {}  (nearest stored code is the right one)", identification, ascii_bar(identification as f32));
    say!("  mean G         {:>6.3}  {}", g, ascii_bar(g as f32));
    say!("  exact match    {:>6.3}", lookup_acc);
    say!("  chance         {:>6.3}", chance);
    if silent_frames > 0 {
        say!("  frames where the MAC did not activate: {silent_frames}");
    }

    let mut summary = Summary::new();
    summary.push("keep", keep as f64);
    summary.push("completion", completion);
    summary.push("identification", identification);
    summary.push("g", g);
    summary.push("baseline_lookup_accuracy", lookup_acc);
    summary.push("baseline_chance", chance);
    summary.push("identification_vs_chance", identification / chance);
    summary.push("silent_frames", silent_frames as f64);

    if identification > chance * 2.0 {
        let note = format!(
            "{:.0}% of the active cells is enough to identify the right memory {:.0}% of the time and recover {:.0}% of its code, where exact match recovers {:.0}%",
            keep * 100.0,
            identification * 100.0,
            completion * 100.0,
            lookup_acc * 100.0
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: the cue is too small to identify a memory — raise --keep, or lower --patterns so the codebook is less crowded.");
        summary.verdict(false, "the cue does not identify a memory better than chance");
    }

    rec.finish_summary(&summary);
    summary
}
