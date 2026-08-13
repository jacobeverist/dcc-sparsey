// Randomness for the demos.
//
// `rand` and `rand_xoshiro` are already real dependencies of this crate at
// `default-features = false`, so the demos use the same generator the network does
// and nothing new enters the lockfile. That matters: `getrandom` must stay absent
// (R12), and the test that enforces it reads `Cargo.lock`, so a dev-dependency
// would count.
//
// **There is no global to seed.** This crate satisfies R9 of the dcc-core import
// contract — `SparseyNet::build(config, seed)` owns its stream — so the demo's own
// randomness is a separate object with a separate seed. `stream_seed` derives those
// so the environment cannot accidentally replay the draws the network made while
// wiring its connectivity.

use rand::{Rng as _, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Sub-streams of one run seed. Distinct constants, so no two objects in a demo
/// ever share a stream.
pub const STREAM_NET: u64 = 1;
pub const STREAM_ENV: u64 = 2;
pub const STREAM_EVAL: u64 = 3;

/// Derive an independent seed for one sub-stream of a run.
///
/// The multiplier is the 64-bit golden-ratio constant, which spreads consecutive
/// run seeds (`--repeat` uses `base`, `base + 1`, …) across the space rather than
/// leaving them adjacent — adjacent seeds would make "independent" repeats
/// correlated, which is the one property repeats exist to provide.
pub fn stream_seed(seed: u64, stream: u64) -> u64 {
    let mixed = seed
        .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed ^ (mixed >> 31)
}

pub struct Rng {
    inner: Xoshiro256PlusPlus,
    spare_normal: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            inner: Xoshiro256PlusPlus::seed_from_u64(seed),
            spare_normal: None,
        }
    }

    /// One sub-stream of a run seed.
    pub fn stream(seed: u64, stream: u64) -> Self {
        Self::new(stream_seed(seed, stream))
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        self.inner.random::<f32>()
    }

    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// Standard normal, Box–Muller. The second variate of each pair is kept rather
    /// than discarded, so this costs one pair of uniforms per two calls.
    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare_normal.take() {
            return v;
        }
        let mut u1 = self.unit();
        while u1 <= f32::EPSILON {
            u1 = self.unit();
        }
        let u2 = self.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Uniform in `0..n`, or 0 when `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        self.inner.random_range(0..n)
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }

    /// Pick an index other than `current`, so a "switch" is always a real change.
    pub fn other_than(&mut self, current: usize, n: usize) -> usize {
        if n <= 1 {
            return current;
        }
        (current + 1 + self.below(n - 1)) % n
    }

    /// `k` distinct values from `0..n`, sorted ascending.
    ///
    /// Sorted because `SparseyNet::set_input` takes active *cell indices* and every
    /// consumer of these is easier to reason about in order; ascending also makes
    /// two patterns comparable by plain slice equality.
    pub fn sample_sorted(&mut self, n: usize, k: usize) -> Vec<u32> {
        let k = k.min(n);
        let mut pool: Vec<u32> = (0..n as u32).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            pool.swap(i, j);
        }
        let mut out = pool[..k].to_vec();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_seeds_are_distinct() {
        for seed in [0u64, 1, 12345, u64::MAX] {
            let seeds: Vec<u64> = (1..=3).map(|s| stream_seed(seed, s)).collect();
            for i in 0..seeds.len() {
                for j in (i + 1)..seeds.len() {
                    assert_ne!(seeds[i], seeds[j], "streams {i} and {j} collide");
                }
            }
        }
    }

    #[test]
    fn adjacent_run_seeds_do_not_produce_adjacent_streams() {
        let a = stream_seed(12345, STREAM_ENV);
        let b = stream_seed(12346, STREAM_ENV);
        assert!(a.abs_diff(b) > 1_000_000, "a={a} b={b}");
    }

    #[test]
    fn the_same_seed_replays_exactly() {
        let draw = || {
            let mut r = Rng::new(99);
            (0..50).map(|_| r.below(1000)).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn different_streams_do_not_replay_each_other() {
        let take = |s| {
            let mut r = Rng::stream(7, s);
            (0..50).map(|_| r.below(1000)).collect::<Vec<_>>()
        };
        assert_ne!(take(STREAM_ENV), take(STREAM_NET));
    }

    #[test]
    fn below_zero_is_zero_rather_than_an_empty_range_panic() {
        let mut r = Rng::new(1);
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn sample_sorted_is_distinct_sorted_and_the_right_length() {
        let mut r = Rng::new(3);
        for _ in 0..200 {
            let s = r.sample_sorted(64, 8);
            assert_eq!(s.len(), 8);
            assert!(s.windows(2).all(|w| w[0] < w[1]), "not sorted or not distinct: {s:?}");
            assert!(s.iter().all(|&v| v < 64));
        }
    }

    #[test]
    fn sampling_more_than_exists_returns_everything_once() {
        let mut r = Rng::new(5);
        let s = r.sample_sorted(4, 10);
        assert_eq!(s, vec![0, 1, 2, 3]);
    }

    #[test]
    fn other_than_never_returns_the_current_value() {
        let mut r = Rng::new(13);
        for _ in 0..500 {
            assert_ne!(r.other_than(2, 5), 2);
        }
        assert_eq!(r.other_than(0, 1), 0);
    }

    #[test]
    fn normal_has_roughly_unit_variance() {
        let mut r = Rng::new(11);
        let n = 20_000;
        let xs: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }
}
