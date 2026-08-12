//! Performance baseline for the sparsey frame pipeline.
//!
//! Establishes the reference numbers for the "performance-oriented coder" work
//! (rayon MAC-parallelism, clone/allocation elimination, bitfield accumulation).
//! Every optimization is measured against these. The workload is a single internal
//! region of several MACs fed a sparse binary input — the hot path is
//! `process_region_macs` → `normalize_mac` / `set_v` / `compute_g` / `select_winners`
//! / `update_weights`.
//!
//! Run: `cargo bench -p dcc_sparsey`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dcc_sparsey::config::BackoffConfig;
use dcc_sparsey::{NetworkConfigBuilder, RegionConfigBuilder, SparseyNet};

/// Build an `input_side × input_side` input region → one internal region of
/// `mac_side × mac_side` MACs (Q CMs × K cells), full connectivity, fixed seed.
fn build(input_side: u32, mac_side: u32, q: u32, k: u32, csa: bool) -> SparseyNet {
    let internal = {
        let b = RegionConfigBuilder::new("internal", 1)
            .grid(mac_side, mac_side)
            .qk(q, k)
            .persistence(1)
            .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96));
        if csa { b.enable_csa() } else { b }.build()
    };
    let cfg = NetworkConfigBuilder::default()
        .region(RegionConfigBuilder::new("input", 0).grid(input_side, input_side).build())
        .region(internal)
        .connect("input", "internal")
        .build();
    SparseyNet::build(cfg, 42).expect("build bench net")
}

/// A deterministic set of sparse input vectors (~`density` fraction of cells on).
fn inputs(n: usize, cells: u32, density: f32) -> Vec<Vec<u32>> {
    // simple LCG so the bench has no rand dep and is reproducible
    let mut s: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let on = ((cells as f32 * density).round() as u32).max(1);
    (0..n)
        .map(|_| {
            let mut v: Vec<u32> = (0..on).map(|_| next() % cells).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect()
}

fn bench_frames(c: &mut Criterion) {
    let input_side = 12u32; // 144 input cells
    let cells = input_side * input_side;
    let alphabet = inputs(16, cells, 0.08);

    let mut group = c.benchmark_group("frame");
    for (label, mac_side, q, k) in [("small_3x3_q8k12", 3u32, 8u32, 12u32), ("mid_5x5_q12k16", 5, 12, 16)]
    {
        // Learning frame (max-V).
        group.bench_with_input(BenchmarkId::new("learn_maxv", label), &(), |b, _| {
            let mut net = build(input_side, mac_side, q, k, false);
            let input = net.region_id("input").unwrap();
            let mut i = 0usize;
            b.iter(|| {
                net.set_input(input, &alphabet[i % alphabet.len()]).unwrap();
                net.do_frame_learn();
                i += 1;
                black_box(&net);
            });
        });

        // Learning frame (probabilistic CSA).
        group.bench_with_input(BenchmarkId::new("learn_csa", label), &(), |b, _| {
            let mut net = build(input_side, mac_side, q, k, true);
            let input = net.region_id("input").unwrap();
            let mut i = 0usize;
            b.iter(|| {
                net.set_input(input, &alphabet[i % alphabet.len()]).unwrap();
                net.do_frame_learn();
                i += 1;
                black_box(&net);
            });
        });

        // Recognition frame (max-V, stable — no weight growth).
        group.bench_with_input(BenchmarkId::new("recognize", label), &(), |b, _| {
            let mut net = build(input_side, mac_side, q, k, false);
            let input = net.region_id("input").unwrap();
            for v in &alphabet {
                net.set_input(input, v).unwrap();
                net.do_frame_learn();
            }
            net.finalize_learning();
            net.prepare_for_new_run(false);
            let mut i = 0usize;
            b.iter(|| {
                net.set_input(input, &alphabet[i % alphabet.len()]).unwrap();
                net.do_frame_recognize();
                i += 1;
                black_box(&net);
            });
        });
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    c.bench_function("build/mid_5x5_q12k16", |b| {
        b.iter(|| black_box(build(12, 5, 12, 16, false)));
    });
}

criterion_group!(benches, bench_frames, bench_build);
criterion_main!(benches);
