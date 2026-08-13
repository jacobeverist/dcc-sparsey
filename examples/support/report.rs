// Text reporting primitives.
//
// Everything here is std-only and knows nothing about HTM. Demos are headless and
// text-only by design: a windowed viewer is not instrumentation, and real
// visualisation is dcc-dashboard's job, fed by the `--metrics` records rather than
// by anything drawn here.

use std::collections::VecDeque;
use std::fmt::Write as _;

/// A fixed-width window *and* an exponential moving average over the same series.
///
/// Both are needed rather than one or the other. A windowed mean over a sparse
/// signal — an anomaly flag that fires on 2% of steps, say — is mostly zeros and
/// jumps around; an EMA smooths it but lags a real change. Reporting both makes it
/// obvious which is happening.
pub struct Rolling {
    window: VecDeque<f32>,
    capacity: usize,
    sum: f64,
    ema: f32,
    alpha: f32,
    seeded: bool,
    count: u64,
}

impl Rolling {
    pub fn new(capacity: usize, alpha: f32) -> Self {
        Rolling {
            window: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            sum: 0.0,
            ema: 0.0,
            alpha,
            seeded: false,
            count: 0,
        }
    }

    pub fn push(&mut self, v: f32) {
        self.count += 1;

        if self.window.len() == self.capacity {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old as f64;
            }
        }
        self.window.push_back(v);
        self.sum += v as f64;

        // Seeding with the first sample rather than with zero: an EMA started at
        // zero spends its first 1/alpha samples climbing out of a value that was
        // never observed, which reads as "still learning" when nothing is.
        if self.seeded {
            self.ema += self.alpha * (v - self.ema);
        } else {
            self.ema = v;
            self.seeded = true;
        }
    }

    pub fn mean(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        (self.sum / self.window.len() as f64) as f32
    }

    pub fn ema(&self) -> f32 {
        self.ema
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Total samples pushed, which is not `len()` once the window has filled.
    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn as_slice(&self) -> Vec<f32> {
        self.window.iter().copied().collect()
    }
}

const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// A one-line plot, auto-scaled to the data's own range.
///
/// A flat series renders as the lowest block rather than dividing by zero.
pub fn sparkline(values: &[f32]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = hi - lo;

    values
        .iter()
        .map(|&v| {
            if span <= 0.0 || !span.is_finite() {
                BLOCKS[0]
            } else {
                let t = ((v - lo) / span * (BLOCKS.len() - 1) as f32).round();
                BLOCKS[(t as usize).min(BLOCKS.len() - 1)]
            }
        })
        .collect()
}

/// A 20-cell bar for a value in `[0, 1]`. Out-of-range input is clamped.
pub fn ascii_bar(x: f32) -> String {
    let filled = (x.clamp(0.0, 1.0) * 20.0).round() as usize;
    format!("[{}{}]", "#".repeat(filled), ".".repeat(20 - filled))
}

/// A rows-are-true, columns-are-predicted table with per-class recall.
///
/// Recall per row rather than only overall accuracy, because "70% accurate" hides
/// a model that has learned four classes perfectly and one not at all — which is
/// the failure mode that actually occurs.
pub fn confusion_table(counts: &[Vec<u64>], labels: &[String]) -> String {
    let n = counts.len();
    if n == 0 || labels.len() < n {
        return String::new();
    }

    let mut out = String::new();
    let w = labels.iter().map(|l| l.len()).max().unwrap_or(4).max(7);

    let _ = write!(out, "{:>w$} |", "true\\pred");
    for l in labels.iter().take(n) {
        let _ = write!(out, " {l:>w$}");
    }
    let _ = writeln!(out, " |  recall");
    let _ = writeln!(out, "{}", "-".repeat(w + 2 + n * (w + 1) + 10));

    for (i, row) in counts.iter().enumerate() {
        let _ = write!(out, "{:>w$} |", labels[i]);
        for &c in row.iter().take(n) {
            let _ = write!(out, " {c:>w$}");
        }
        let total: u64 = row.iter().take(n).sum();
        if total == 0 {
            let _ = writeln!(out, " |       —");
        } else {
            let _ = writeln!(out, " | {:>6.1}%", row[i] as f64 / total as f64 * 100.0);
        }
    }

    out
}

/// Render two multi-line blocks next to each other, for before/after comparisons.
pub fn side_by_side(left: &str, right: &str, gap: usize) -> String {
    let l: Vec<&str> = left.lines().collect();
    let r: Vec<&str> = right.lines().collect();
    let width = l.iter().map(|s| s.chars().count()).max().unwrap_or(0);

    let mut out = String::new();
    for i in 0..l.len().max(r.len()) {
        let lhs = l.get(i).copied().unwrap_or("");
        let pad = width - lhs.chars().count();
        let _ = writeln!(
            out,
            "{lhs}{}{}",
            " ".repeat(pad + gap),
            r.get(i).copied().unwrap_or("")
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_mean_forgets_beyond_the_window() {
        let mut r = Rolling::new(3, 0.5);
        for v in [1.0, 1.0, 1.0, 7.0, 7.0, 7.0] {
            r.push(v);
        }
        assert_eq!(r.len(), 3);
        assert_eq!(r.count(), 6);
        assert!((r.mean() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn ema_seeds_from_the_first_sample_rather_than_from_zero() {
        let mut r = Rolling::new(10, 0.1);
        r.push(5.0);
        // Seeded at zero this would read 0.5 and look like a signal climbing.
        assert!((r.ema() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn a_flat_series_does_not_divide_by_zero() {
        let s = sparkline(&[2.0, 2.0, 2.0]);
        assert_eq!(s.chars().count(), 3);
        assert!(s.chars().all(|c| c == BLOCKS[0]));
    }

    #[test]
    fn sparkline_spans_the_block_ramp() {
        let s = sparkline(&[0.0, 1.0]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], BLOCKS[0]);
        assert_eq!(chars[1], BLOCKS[BLOCKS.len() - 1]);
    }

    #[test]
    fn ascii_bar_clamps_rather_than_panicking() {
        assert!(ascii_bar(-1.0).contains(".................."));
        assert_eq!(ascii_bar(2.0), format!("[{}]", "#".repeat(20)));
    }

    #[test]
    fn confusion_table_reports_per_class_recall() {
        let counts = vec![vec![8, 2], vec![0, 10]];
        let labels = vec!["a".to_string(), "b".to_string()];
        let t = confusion_table(&counts, &labels);
        assert!(t.contains("80.0%"), "{t}");
        assert!(t.contains("100.0%"), "{t}");
    }

    #[test]
    fn an_empty_confusion_row_reports_no_recall_rather_than_nan() {
        let counts = vec![vec![0, 0], vec![0, 4]];
        let labels = vec!["a".to_string(), "b".to_string()];
        assert!(confusion_table(&counts, &labels).contains("—"));
    }
}
