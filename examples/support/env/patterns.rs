// Labelled sparse patterns, corruption of them, and partial cues.
//
// Used by `classify_stream`, `noise_robustness`, `partial_cue` and `capacity`. The
// patterns carry no structure on purpose: what those demos ask about is the
// *learned representation*, and structure in the stimulus would let a model exploit
// that instead.

use std::collections::HashMap;

use crate::support::rng::Rng;

/// A fixed set of patterns over an input region's cells, one per class.
pub struct PatternBook {
    patterns: Vec<Vec<u32>>,
    cells: usize,
    active: usize,
}

impl PatternBook {
    /// `count` patterns, each `active` distinct active cells drawn from `cells`.
    pub fn generate(count: usize, cells: usize, active: usize, rng: &mut Rng) -> Self {
        let patterns = (0..count).map(|_| rng.sample_sorted(cells, active)).collect();
        PatternBook {
            patterns,
            cells,
            active,
        }
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn get(&self, i: usize) -> &[u32] {
        &self.patterns[i]
    }

    pub fn cells(&self) -> usize {
        self.cells
    }

    pub fn active(&self) -> usize {
        self.active
    }

    /// Largest number of active cells shared by any two patterns.
    ///
    /// Reported rather than assumed: two patterns drawn independently *can* land
    /// close together, and if they do the task is harder for reasons that have
    /// nothing to do with the architecture — which reads as a worse result unless
    /// this number is on the page beside it.
    pub fn max_overlap(&self) -> usize {
        let mut worst = 0;
        for i in 0..self.patterns.len() {
            for j in (i + 1)..self.patterns.len() {
                worst = worst.max(overlap(&self.patterns[i], &self.patterns[j]));
            }
        }
        worst
    }
}

/// Active cells common to both patterns. Both must be sorted ascending.
pub fn overlap(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// Move a fraction of the active cells elsewhere, leaving overlap `1 - fraction`.
///
/// The active-cell *count* is preserved. That is load-bearing: a MAC only activates
/// when its active-input count falls inside its activation band (`active_low_frac`
/// to `active_high_frac`), so a corruption that thinned the input would push it out
/// of the band and the MAC would not activate at all. The result would look like
/// total failure under noise when in fact the input was simply rejected as
/// malformed — a completely different finding.
pub fn corrupt(clean: &[u32], fraction: f32, cells: usize, rng: &mut Rng) -> Vec<u32> {
    let move_count = ((fraction.clamp(0.0, 1.0) * clean.len() as f32) + 0.5) as usize;
    if move_count == 0 {
        return clean.to_vec();
    }

    // Keep a random subset of the originals.
    let keep_idx = rng.sample_sorted(clean.len(), clean.len() - move_count);
    let mut out: Vec<u32> = keep_idx.iter().map(|&i| clean[i as usize]).collect();

    // Draw replacements from cells the pattern does not already use, so the
    // requested fraction is the fraction actually moved.
    let mut added = 0;
    let mut guard = 0;
    while added < move_count && guard < cells * 10 {
        guard += 1;
        let candidate = rng.below(cells) as u32;
        if clean.binary_search(&candidate).is_err() && !out.contains(&candidate) {
            out.push(candidate);
            added += 1;
        }
    }

    out.sort_unstable();
    out
}

/// Keep only a fraction of the active cells, dropping the rest.
///
/// This is occlusion rather than corruption — the surviving evidence is all
/// correct, there is simply less of it — and it is what `partial_cue` asks about:
/// how much of a stored pattern is needed to bring back the whole code.
///
/// Note that this *does* change the active count, so a demo using it has to widen
/// the activation band or the MAC will refuse to activate. That is a real
/// constraint of the architecture rather than an inconvenience, and the demo says so.
pub fn occlude(clean: &[u32], keep_fraction: f32, rng: &mut Rng) -> Vec<u32> {
    let keep = ((keep_fraction.clamp(0.0, 1.0) * clean.len() as f32) + 0.5) as usize;
    let idx = rng.sample_sorted(clean.len(), keep);
    let mut out: Vec<u32> = idx.iter().map(|&i| clean[i as usize]).collect();
    out.sort_unstable();
    out
}

/// Exact-match memorisation — the control the noise demo is compared against.
///
/// Perfect on a clean pattern and blind one cell off it, so its curve is the shape
/// memorisation rather than generalisation produces.
#[derive(Default)]
pub struct LookupTable {
    table: HashMap<Vec<u32>, usize>,
}

impl LookupTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn learn(&mut self, pattern: &[u32], label: usize) {
        self.table.insert(pattern.to_vec(), label);
    }

    pub fn classify(&self, pattern: &[u32]) -> Option<usize> {
        self.table.get(pattern).copied()
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::STREAM_ENV;

    #[test]
    fn corruption_moves_exactly_the_requested_fraction_and_keeps_the_count() {
        let mut rng = Rng::stream(1, STREAM_ENV);
        let book = PatternBook::generate(1, 256, 20, &mut rng);
        let clean = book.get(0).to_vec();

        for f in [0.0f32, 0.25, 0.5, 1.0] {
            let noisy = corrupt(&clean, f, 256, &mut rng);
            let expected_kept = clean.len() - ((f * clean.len() as f32) + 0.5) as usize;
            assert_eq!(
                overlap(&clean, &noisy),
                expected_kept,
                "at f={f} the overlap was not 1-f"
            );
            // The activation band depends on this: thinning the input would push
            // the MAC out of its band and it would not activate at all.
            assert_eq!(noisy.len(), clean.len(), "active count changed at f={f}");
            assert!(noisy.windows(2).all(|w| w[0] < w[1]), "not sorted at f={f}");
        }
    }

    #[test]
    fn occlusion_keeps_a_correct_subset() {
        let mut rng = Rng::stream(2, STREAM_ENV);
        let book = PatternBook::generate(1, 256, 20, &mut rng);
        let clean = book.get(0);

        let cue = occlude(clean, 0.5, &mut rng);
        assert_eq!(cue.len(), 10);
        // Everything that survives is genuinely part of the original.
        assert_eq!(overlap(clean, &cue), cue.len());
    }

    #[test]
    fn overlap_counts_shared_cells() {
        assert_eq!(overlap(&[1, 2, 3], &[2, 3, 4]), 2);
        assert_eq!(overlap(&[1, 2], &[3, 4]), 0);
        assert_eq!(overlap(&[], &[1]), 0);
    }

    #[test]
    fn lookup_is_perfect_on_clean_input_and_blind_off_it() {
        let mut rng = Rng::stream(3, STREAM_ENV);
        let book = PatternBook::generate(6, 256, 16, &mut rng);
        let mut table = LookupTable::new();
        for c in 0..book.len() {
            table.learn(book.get(c), c);
        }
        for c in 0..book.len() {
            assert_eq!(table.classify(book.get(c)), Some(c));
        }
        let noisy = corrupt(book.get(0), 0.25, 256, &mut rng);
        assert_eq!(table.classify(&noisy), None);
    }

    #[test]
    fn independent_patterns_barely_overlap() {
        let mut rng = Rng::stream(4, STREAM_ENV);
        let book = PatternBook::generate(8, 256, 16, &mut rng);
        assert!(book.max_overlap() < 8, "max overlap {}", book.max_overlap());
    }
}
