// Reading what the network is actually doing.
//
// Sparsey reports a scalar familiarity `G ∈ [0, 1]` for every MAC on every frame,
// which is unusual and is most of what a demo needs — no separate detector, no
// classifier, no threshold to train. `Capture` is the `Recorder` that collects it.
//
// The four existing examples each defined their own private `Capture`, four times
// over. This is that recorder written once, plus the diagnostics that explain *why*
// a run behaves as it does rather than only that it did.

use std::collections::BTreeMap;

use dcc_sparsey::ids::{MacId, RegionId};
use dcc_sparsey::{Recorder, SparseyNet};

/// One MAC's result for a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MacResult {
    pub region: RegionId,
    pub mac: MacId,
    /// One winning cell index per CM — the code, of length `Q`.
    pub code: Vec<u32>,
    /// Familiarity in `[0, 1]`.
    pub g: f32,
}

/// Collects every code selected during a frame, in MAC order.
///
/// **Use the `_rec` frame drivers with this.** With `persistence = 1` a MAC's code
/// is cleared in `end_frame`, so calling `net.mac_code(mac)` after the frame
/// returns `None` and a demo reading it that way silently sees nothing. The
/// recorder callback fires while the code is still live, which is the only reliable
/// way to observe it.
#[derive(Default)]
pub struct Capture {
    results: BTreeMap<(usize, usize), MacResult>,
    frames: u64,
}

impl Capture {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the previous frame. Call before each frame, or codes from an earlier
    /// frame linger and are read as if they were this frame's.
    pub fn clear(&mut self) {
        self.results.clear();
    }

    pub fn results(&self) -> impl Iterator<Item = &MacResult> {
        self.results.values()
    }

    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The result for one MAC, if it activated this frame. A MAC outside its
    /// activation band does not activate at all, so this genuinely can be `None`.
    pub fn get(&self, region: RegionId, mac: MacId) -> Option<&MacResult> {
        self.results.get(&(region.0, mac.0))
    }

    /// The code of the first MAC that activated — the common case, where a demo has
    /// one internal region holding one MAC.
    pub fn first_code(&self) -> Option<&[u32]> {
        self.results.values().next().map(|r| r.code.as_slice())
    }

    /// Mean familiarity across every MAC that activated this frame.
    ///
    /// `None` rather than 0.0 when nothing activated: "no MAC met its activation
    /// band" and "every MAC found the input completely unfamiliar" are different
    /// facts, and averaging the empty set into a zero would merge them.
    pub fn mean_g(&self) -> Option<f32> {
        if self.results.is_empty() {
            return None;
        }
        Some(self.results.values().map(|r| r.g).sum::<f32>() / self.results.len() as f32)
    }

    pub fn max_g(&self) -> Option<f32> {
        self.results
            .values()
            .map(|r| r.g)
            .fold(None, |acc, g| Some(acc.map_or(g, |b: f32| b.max(g))))
    }
}

impl Recorder for Capture {
    fn on_code_selected(&mut self, region: RegionId, mac: MacId, code: &[u32], g: f32, _frame: i64) {
        self.results.insert(
            (region.0, mac.0),
            MacResult {
                region,
                mac,
                code: code.to_vec(),
                g,
            },
        );
    }

    fn on_frame_end(&mut self, _frame: i64) {
        self.frames += 1;
    }
}

/// How much of the network's capacity has been consumed.
///
/// Sparsey's synapses carry a `(stiffness, timestamp)` pair rather than a weight,
/// and a bundle whose increased fraction passes the target region's saturation
/// threshold **freezes permanently** and stops learning. That makes plasticity a
/// finite resource with an observable level, which is a phenomenon neither sibling
/// port models — and it is what `capacity` reports against.
#[derive(Clone, Copy, Debug, Default)]
pub struct CapacityStats {
    pub bundles: usize,
    pub frozen_bundles: usize,
    pub synapses: usize,
    /// Synapses that have been increased at least once.
    pub touched_synapses: usize,
    /// Synapses at maximum stiffness — permanent, no longer decaying.
    pub permanent_synapses: usize,
}

impl CapacityStats {
    pub fn frozen_fraction(&self) -> f64 {
        ratio(self.frozen_bundles, self.bundles)
    }

    pub fn touched_fraction(&self) -> f64 {
        ratio(self.touched_synapses, self.synapses)
    }

    pub fn permanent_fraction(&self) -> f64 {
        ratio(self.permanent_synapses, self.synapses)
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Read the arena directly. `SparseyNet`'s arena vectors are public, which is what
/// makes this possible without adding accessors to `src/`.
pub fn capacity_stats(net: &SparseyNet) -> CapacityStats {
    let mut stats = CapacityStats {
        bundles: net.efferent_bundles.len(),
        ..Default::default()
    };

    for bundle in &net.efferent_bundles {
        if bundle.frozen {
            stats.frozen_bundles += 1;
        }
        for syn in &bundle.synapses {
            stats.synapses += 1;
            if syn.timestamp_last_pre_post != i64::MAX {
                stats.touched_synapses += 1;
            }
            if syn.stiffness >= net.config.weight_table.max_stiffness {
                stats.permanent_synapses += 1;
            }
        }
    }

    stats
}

/// Fraction of matching positions between two codes.
///
/// Codes are one winning cell per CM, so this is the natural similarity: two codes
/// of length `Q` agree in some number of their `Q` competitive modules. It is *not*
/// a bit overlap — comparing Sparsey codes as if they were plain SDRs would count
/// the losing cells as agreement.
pub fn code_similarity(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NAN;
    }
    a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_similarity_is_per_cm_agreement() {
        assert_eq!(code_similarity(&[1, 2, 3, 4], &[1, 2, 3, 4]), 1.0);
        assert_eq!(code_similarity(&[1, 2, 3, 4], &[1, 2, 9, 9]), 0.5);
        assert_eq!(code_similarity(&[1, 2], &[9, 9]), 0.0);
    }

    #[test]
    fn mismatched_or_empty_codes_are_not_a_number_rather_than_zero() {
        assert!(code_similarity(&[], &[]).is_nan());
        assert!(code_similarity(&[1, 2], &[1]).is_nan());
    }

    #[test]
    fn an_empty_capture_reports_no_familiarity_rather_than_zero() {
        let c = Capture::new();
        // "nothing activated" and "everything was completely unfamiliar" are
        // different facts and must not both read as 0.0.
        assert!(c.mean_g().is_none());
        assert!(c.max_g().is_none());
        assert!(c.is_empty());
    }

    #[test]
    fn capture_collects_per_mac_and_clears() {
        let mut c = Capture::new();
        c.on_code_selected(RegionId(1), MacId(0), &[1, 2], 0.5, 0);
        c.on_code_selected(RegionId(1), MacId(1), &[3, 4], 0.9, 0);
        assert_eq!(c.len(), 2);
        assert_eq!(c.first_code(), Some(&[1u32, 2][..]));
        assert!((c.mean_g().unwrap() - 0.7).abs() < 1e-6);
        assert!((c.max_g().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(c.get(RegionId(1), MacId(1)).unwrap().code, vec![3, 4]);

        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn a_later_frame_replaces_the_same_macs_result() {
        let mut c = Capture::new();
        c.on_code_selected(RegionId(1), MacId(0), &[1, 1], 0.1, 0);
        c.on_code_selected(RegionId(1), MacId(0), &[2, 2], 0.8, 1);
        assert_eq!(c.len(), 1);
        assert_eq!(c.first_code(), Some(&[2u32, 2][..]));
    }
}
