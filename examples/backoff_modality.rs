// Backoff Modality — recognising with a signal missing.
//
// Sparsey's recognition is not one match but a *priority chain* of them. A region
// carries a list of cases, each naming a set of signal types and a threshold; at
// recognition time `src/backoff.rs` walks them high to low, considers only the cases
// whose types are actually available this frame, and keeps the best one that clears
// its threshold. So a network that normally recognises from bottom-up input plus
// lateral context can still recognise from either alone — it falls to a lower case
// rather than failing.
//
// The demo drives that directly. Two input regions ("shape" and "colour") feed one
// MAC, and the MAC also has a lateral self-link, so both U and H signals exist.
// Items are learned with everything present, then recognised under four conditions:
//
//   both modalities · shape only · colour only · neither (lateral context alone)
//
// The first three vary how *much* bottom-up signal there is. The fourth was meant to
// vary which signal *types* exist — with both inputs silent, every U-bearing case
// becomes unavailable and the chain should fall to a case built on H alone.
//
// **It does not, and that is the most useful thing this demo reports.** The MAC never
// activates at all: 240 silent frames, no code, no familiarity. The cause is upstream
// of backoff. `resolve_activation_bands` in `src/net/build.rs` computes a region's
// activation band from its **U afferent input size only**, and floors the lower bound
// at `max(1, …)` — so a MAC is eligible only if at least one *U* feature is active.
// `activate_region` checks that eligibility before `recognize_version` is ever
// called. An H-only case is therefore unreachable by construction, however the chain
// is written: lateral context can modulate a recognition that bottom-up input has
// already licensed, but it cannot license one by itself.
//
// That is worth knowing before designing anything that expects a region to run on
// context alone, and it is why the verdict below gates on the single-modality
// conditions and merely reports the fourth.
//
//   cargo run --release --example backoff_modality
//   cargo run --release --example backoff_modality -- --sweep items=4,8,16 --repeat 3

#[path = "support/mod.rs"]
mod support;

use dcc_sparsey::config::{BackoffCase, BackoffConfig};
use dcc_sparsey::ids::RegionId;
use dcc_sparsey::{
    NetworkConfig, NetworkConfigBuilder, RegionConfigBuilder, SparseyNet, SynapseType,
};

use support::args::Args;
use support::env::patterns::PatternBook;
use support::metrics::{Recorder, Summary};
use support::probe::{code_similarity, Capture};
use support::report::ascii_bar;
use support::rng::{Rng, STREAM_ENV, STREAM_EVAL};
use support::sweep;

fn main() {
    let args = Args::parse();
    let mut rec = Recorder::from_args("backoff_modality", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

/// A chain with an explicit `{H,U}` level above a `{U}` level above an `{H}` level.
///
/// The H-only level is named deliberately and turns out to be **dead code in
/// practice** — see the header. It is kept because its being unreachable is the
/// finding, and a chain without it could not distinguish "the case was never
/// selected" from "the case was never offered".
fn chain() -> BackoffConfig {
    use SynapseType::{H, U};
    BackoffConfig {
        priorities: vec![
            vec![BackoffCase {
                syn_types: vec![H, U],
                threshold: 0.0,
            }],
            vec![BackoffCase {
                syn_types: vec![U],
                threshold: 0.0,
            }],
            vec![BackoffCase {
                syn_types: vec![H],
                threshold: 0.0,
            }],
        ],
    }
}

fn build_config(grid: u32, q: u32, k: u32) -> NetworkConfig {
    NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("shape", 0).grid(grid, grid).build())
        .region(RegionConfigBuilder::new("colour", 0).grid(grid, grid).build())
        .region(
            RegionConfigBuilder::new("l1", 1)
                .grid(1, 1)
                .qk(q, k)
                .persistence(1)
                .backoff(chain())
                .build(),
        )
        .connect("shape", "l1")
        .connect("colour", "l1")
        // Same DAG height at both ends, so this is an H link and carries the
        // previous frame's code.
        .connect("l1", "l1")
        .build()
}

/// Which modalities are presented.
#[derive(Clone, Copy)]
struct Condition {
    name: &'static str,
    shape: bool,
    colour: bool,
}

const CONDITIONS: [Condition; 4] = [
    Condition { name: "both", shape: true, colour: true },
    Condition { name: "shape only", shape: true, colour: false },
    Condition { name: "colour only", shape: false, colour: true },
    Condition { name: "neither (H only)", shape: false, colour: false },
];

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let items: usize = args.get("items", 8);
    let grid: u32 = args.get("grid", 12);
    let q: u32 = args.get("q", 12);
    let k: u32 = args.get("k", 16);
    let active: usize = args.get("active", 16);
    let epochs: usize = args.get("epochs", 20);
    let reps: usize = args.get("reps", 30);
    let silent = args.flag("silent");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    let cells = (grid * grid) as usize;

    rec.config("items", items);
    rec.config("grid", grid);
    rec.config("q", q);
    rec.config("k", k);
    rec.config("epochs", epochs);

    let mut env = Rng::stream(seed, STREAM_ENV);
    let shapes = PatternBook::generate(items, cells, active, &mut env);
    let colours = PatternBook::generate(items, cells, active, &mut env);
    // A fixed lead-in frame before every item, so the H signal has something to
    // carry. Without a predecessor there is no lateral context and the fourth
    // condition would have no signal of any kind.
    let lead_shape = PatternBook::generate(1, cells, active, &mut env);
    let lead_colour = PatternBook::generate(1, cells, active, &mut env);

    let config = build_config(grid, q, k);
    let mut net = SparseyNet::build(config.clone(), seed).expect("build network");
    let shape_r = net.region_id("shape").expect("shape region");
    let colour_r = net.region_id("colour").expect("colour region");

    say!("Backoff Modality — {items} items, two input modalities plus lateral context, seed {seed}");
    say!("  each input {grid}x{grid} = {cells} cells, {active} active; one MAC with Q={q} CMs of K={k} cells");
    say!("  chain: {{H,U}} -> {{U}} -> {{H}}, so losing all bottom-up input falls to a real level");
    say!();

    // --- Learning: a lead-in frame, then the item, with everything present ---

    for _ in 0..epochs {
        for i in 0..items {
            net.prepare_for_new_run(false);
            present(&mut net, shape_r, colour_r, Some(lead_shape.get(0)), Some(lead_colour.get(0)), true);
            present(&mut net, shape_r, colour_r, Some(shapes.get(i)), Some(colours.get(i)), true);
        }
    }
    net.finalize_learning();

    // **The reference codes come from a recognition pass, not from learning.**
    //
    // Capturing them during learning gives the wrong answer twice over. Code
    // selection on a *learning* frame is probabilistic — `SigmoidConfig::enabled`
    // defaults true, so the CSA samples rather than taking max-V — while
    // recognition is deterministic max-V, so the two need not agree. And the item's
    // code depends on its lateral context, which is the lead-in's code, which is
    // itself still moving while learning continues. The first version of this demo
    // stored the last training epoch's codes and scored *below chance* as a result.
    //
    // Taking the reference from a full-input recognition pass also states the
    // question properly: this is what the network answers when it can see
    // everything, and the conditions below ask how close it stays with less.
    net.prepare_for_new_run(false);
    let mut stored: Vec<Vec<u32>> = vec![Vec::new(); items];
    for (i, slot) in stored.iter_mut().enumerate() {
        net.prepare_for_new_run(false);
        present(&mut net, shape_r, colour_r, Some(lead_shape.get(0)), Some(lead_colour.get(0)), false);
        let cap = present(&mut net, shape_r, colour_r, Some(shapes.get(i)), Some(colours.get(i)), false);
        *slot = cap.first_code().unwrap_or(&[]).to_vec();
    }

    // --- Recognition under each condition ---

    let mut eval = Rng::stream(seed, STREAM_EVAL);
    let _ = &mut eval;
    net.prepare_for_new_run(false);

    say!("  condition            identification   completion    mean G   silent");
    let mut results = Vec::new();

    for cond in CONDITIONS {
        let mut identified = 0u64;
        let mut completion = 0.0f64;
        let mut g_sum = 0.0f64;
        let mut scored = 0u64;
        let mut silent_frames = 0u64;

        for _ in 0..reps {
            for i in 0..items {
                net.prepare_for_new_run(false);
                // The lead-in is always fully present: it is context, not the probe.
                present(&mut net, shape_r, colour_r, Some(lead_shape.get(0)), Some(lead_colour.get(0)), false);

                let cap = present(
                    &mut net,
                    shape_r,
                    colour_r,
                    cond.shape.then(|| shapes.get(i)),
                    cond.colour.then(|| colours.get(i)),
                    false,
                );

                let Some(code) = cap.first_code() else {
                    silent_frames += 1;
                    continue;
                };

                completion += code_similarity(code, &stored[i]);
                g_sum += cap.mean_g().unwrap_or(0.0) as f64;

                let mut best = 0usize;
                let mut best_sim = f64::NEG_INFINITY;
                for (j, s) in stored.iter().enumerate() {
                    let sim = code_similarity(code, s);
                    if sim > best_sim {
                        best_sim = sim;
                        best = j;
                    }
                }
                if best == i {
                    identified += 1;
                }
                scored += 1;
            }
        }

        let n = scored.max(1) as f64;
        let id = identified as f64 / n;
        let comp = completion / n;
        let g = g_sum / n;
        say!(
            "  {:<20} {:>10.3}    {:>9.3}   {:>7.3}   {:>6}  {}",
            cond.name,
            id,
            comp,
            g,
            silent_frames,
            ascii_bar(id as f32)
        );
        results.push((cond.name, id, comp, g, silent_frames));
    }

    // --- Report ---

    let chance = 1.0 / items as f64;
    say!("\n  chance identification {:.3}", chance);

    let mut summary = Summary::new();
    summary.push("baseline_chance", chance);
    for (name, id, comp, g, silent_frames) in &results {
        let key = name.replace([' ', '(', ')'], "_");
        summary.push(&format!("identification_{key}"), *id);
        summary.push(&format!("completion_{key}"), *comp);
        summary.push(&format!("g_{key}"), *g);
        summary.push(&format!("silent_{key}"), *silent_frames as f64);
    }

    let both = results[0].1;
    let one = results[1].1.max(results[2].1);
    let none = results[3].1;
    summary.push("identification_both", both);
    summary.push("identification_best_single", one);
    summary.push("identification_none", none);
    summary.push("single_vs_chance", one / chance);
    summary.push("retention_single", if both > 0.0 { one / both } else { f64::NAN });

    // The claim is graceful degradation: one modality alone still identifies the
    // item well above chance. The fourth condition is reported but deliberately not
    // gated on — it does not measure a weaker recognition, it measures a MAC that
    // never became eligible to recognise anything.
    let silent_none = results[3].4;
    if one > chance * 2.0 {
        let note = format!(
            "a single modality identifies the item {:.0}% of the time against {:.0}% with both and {:.0}% chance — losing half the input costs {:.0} points; with no bottom-up input the MAC does not activate at all ({silent_none} silent frames), because eligibility is gated on U-feature count before backoff is consulted",
            one * 100.0,
            both * 100.0,
            chance * 100.0,
            (both - one) * 100.0
        );
        say!("\nLearned: {note}.");
        summary.verdict(true, note);
    } else {
        say!("\nNot converged: one modality alone does not identify the item above chance — try fewer --items or a larger --q.");
        summary.verdict(false, "a single modality does not identify the item above chance");
    }

    rec.finish_summary(&summary);
    summary
}

/// Present one frame. A modality passed as `None` is set to no active cells, which
/// is what "this signal is missing" means here — the region is silent rather than
/// carrying wrong information.
fn present(
    net: &mut SparseyNet,
    shape_r: RegionId,
    colour_r: RegionId,
    shape: Option<&[u32]>,
    colour: Option<&[u32]>,
    learn: bool,
) -> Capture {
    net.set_input(shape_r, shape.unwrap_or(&[])).expect("set shape");
    net.set_input(colour_r, colour.unwrap_or(&[])).expect("set colour");
    let mut cap = Capture::new();
    if learn {
        net.do_frame_learn_rec(&mut cap);
    } else {
        net.do_frame_recognize_rec(&mut cap);
    }
    cap
}
