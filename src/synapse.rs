//! Synapses and the implicit-decay weight table.
//!
//! Ported from `Synapse.java` + `Network.createWeightTable()`. Weights are never
//! stored explicitly: a synapse keeps only the global frame of its last pre-post
//! coincidence (`timestamp_last_pre_post`) and a `stiffness`. Its weight is a table
//! lookup on `(stiffness, age)` where `age = current_frame - timestamp`. This models
//! passive decay "for free" — no per-frame decay work.
//!
//! Differences from the Java (see `doc/PortNotes.md`):
//! - No global `Network.globalFrameAcrossAllEpisodes`; the current frame is passed in.
//! - [`Synapse::effective_value`] is read-only. The Java `getEffectiveValue` also
//!   *demotes* a synapse that has decayed to zero (resetting stiffness and adjusting
//!   transient counts); we do that reconciliation explicitly via
//!   [`Synapse::reconcile_if_inert`] on the weight-update path instead, so signal
//!   pulls need only `&self`. The returned weight value is identical either way.

use crate::config::WeightTableConfig;
use crate::ids::NeuronId;

/// Maximum synaptic weight (`Synapse.MAX_WEIGHT`).
pub const MAX_WEIGHT: u8 = 127;

/// Sentinel timestamp for a "virgin" synapse (never had a pre-post) — Java uses
/// `Integer.MAX_VALUE`; the resulting negative age flags weight 0.
pub const VIRGIN_TIMESTAMP: i64 = i64::MAX;

/// The `(stiffness, age) → weight` decay table plus the derived breakpoints.
///
/// Built exactly like `Network.createWeightTable()`: a step function that holds the
/// weight for `WT_TABLE_WEIGHTS[stiffness][t]` across every age up to
/// `WT_TABLE_TRANSITION_INDEXES[stiffness][t]`. Rows `0..max_stiffness` exist;
/// `stiffness == max_stiffness` means *permanent* (weight = [`MAX_WEIGHT`], no row).
#[derive(Clone, Debug)]
pub struct WeightTable {
    /// `table[stiffness][age] = weight`, for `stiffness in 0..max_stiffness`.
    table: Vec<Vec<u8>>,
    /// The transition (age breakpoint) rows, one per stiffness row.
    transition_indexes: Vec<Vec<u32>>,
    /// Stiffness value meaning "permanent".
    max_stiffness: u8,
    /// Largest age with a table entry (max last-transition across rows).
    max_age: u32,
}

impl WeightTable {
    /// Build the table from config, reproducing `Network.createWeightTable()`.
    pub fn build(cfg: &WeightTableConfig) -> Self {
        let max_stiffness = cfg.max_stiffness;
        let num_rows = max_stiffness as usize;
        assert!(
            cfg.transition_indexes.len() >= num_rows && cfg.weights.len() >= num_rows,
            "weight-table config needs at least max_stiffness ({num_rows}) rows"
        );

        // max_age = highest last-transition index across the stiffness rows.
        let mut max_age = 0u32;
        for row in cfg.transition_indexes.iter().take(num_rows) {
            if let Some(&last) = row.last() {
                max_age = max_age.max(last);
            }
        }

        let mut table = vec![vec![0u8; max_age as usize]; num_rows];
        for (stiffness, row) in table.iter_mut().enumerate() {
            let transitions = &cfg.transition_indexes[stiffness];
            let weights = &cfg.weights[stiffness];
            assert_eq!(
                transitions.len(),
                weights.len(),
                "transition/weight row length mismatch at stiffness {stiffness}"
            );
            let mut age = 0usize;
            for (t, &breakpoint) in transitions.iter().enumerate() {
                while (age as u32) < breakpoint {
                    row[age] = weights[t];
                    age += 1;
                }
            }
            // ages >= last breakpoint remain 0 (already initialized).
        }

        WeightTable {
            table,
            transition_indexes: cfg.transition_indexes[..num_rows].to_vec(),
            max_stiffness,
            max_age,
        }
    }

    /// Stiffness value that denotes a permanent synapse.
    #[inline]
    pub fn max_stiffness(&self) -> u8 {
        self.max_stiffness
    }

    /// Highest age (for `stiffness`) that still yields a positive weight
    /// (`getHighestPosWtAge` = last transition − 1).
    #[inline]
    pub fn highest_pos_wt_age(&self, stiffness: u8) -> i64 {
        let row = &self.transition_indexes[stiffness as usize];
        *row.last().unwrap() as i64 - 1
    }

    /// The "freshly increased" breakpoint for `stiffness` (transition index 0). A
    /// pre-post arriving before this age promotes stiffness (`wasFreshlyIncreased`).
    #[inline]
    pub fn fresh_index(&self, stiffness: u8) -> i64 {
        self.transition_indexes[stiffness as usize][0] as i64
    }

    /// Weight for a non-permanent synapse at `(stiffness, age)`. `age < 0` (virgin)
    /// and ages past the last breakpoint yield 0.
    #[inline]
    pub fn weight(&self, stiffness: u8, age: i64) -> u8 {
        if age < 0 || age > self.highest_pos_wt_age(stiffness) {
            0
        } else {
            self.table[stiffness as usize][age as usize]
        }
    }

    /// Largest age with a table entry.
    #[inline]
    pub fn max_age(&self) -> u32 {
        self.max_age
    }
}

/// One synapse. Owned (in the arena) by a source neuron's efferent sub-bundle; its
/// [`Synapse::target_neuron`] is an index into another (or the same) region — the
/// single hard cross-region reference in the Java graph, here just a `NeuronId`.
#[derive(Clone, Debug)]
pub struct Synapse {
    /// The post-synaptic (target) neuron.
    pub target_neuron: NeuronId,
    /// 0 = malleable (decays); `max_stiffness` = permanent.
    pub stiffness: u8,
    /// Global frame of the last pre-post coincidence, or [`VIRGIN_TIMESTAMP`].
    pub timestamp_last_pre_post: i64,
    /// Whether this synapse is currently counted in transiently-increased tallies.
    pub included_in_transient_counts: bool,
}

impl Synapse {
    /// A fresh virgin synapse onto `target` (weight 0 until its first pre-post).
    pub fn new(target: NeuronId) -> Self {
        Synapse {
            target_neuron: target,
            stiffness: 0,
            timestamp_last_pre_post: VIRGIN_TIMESTAMP,
            included_in_transient_counts: false,
        }
    }

    /// Is this synapse permanent (max stiffness)?
    #[inline]
    pub fn is_permanent(&self, wt: &WeightTable) -> bool {
        self.stiffness >= wt.max_stiffness
    }

    /// Effective age at `frame` (`frame - timestamp`), or a negative value for a
    /// virgin synapse.
    #[inline]
    pub fn effective_age(&self, frame: i64) -> i64 {
        frame - self.timestamp_last_pre_post
    }

    /// The synapse's current weight (read-only). Permanent ⇒ [`MAX_WEIGHT`];
    /// virgin/decayed ⇒ 0; otherwise the table lookup. Mirrors `getEffectiveValue`
    /// minus the demotion side effect (see [`Synapse::reconcile_if_inert`]).
    #[inline]
    pub fn effective_value(&self, frame: i64, wt: &WeightTable) -> u8 {
        if self.is_permanent(wt) {
            return MAX_WEIGHT;
        }
        wt.weight(self.stiffness, self.effective_age(frame))
    }

    /// Is the synapse contributing a non-zero weight during learning? (Java
    /// `isContributing` for LEARNING_MODE: age below the last breakpoint.)
    #[inline]
    pub fn is_contributing_learning(&self, frame: i64, wt: &WeightTable) -> bool {
        if self.is_permanent(wt) {
            return true;
        }
        let age = self.effective_age(frame);
        age >= 0 && age <= wt.highest_pos_wt_age(self.stiffness)
    }

    /// Record a pre-post coincidence at `frame`, promoting stiffness if this
    /// pre-post arrived while the previous increase was still "fresh"
    /// (`wasFreshlyIncreased`). Sets the transient flag. Does not exceed permanence.
    pub fn record_pre_post(&mut self, frame: i64, wt: &WeightTable, persistence: u32) {
        if self.timestamp_last_pre_post != VIRGIN_TIMESTAMP && self.stiffness < wt.max_stiffness {
            let age = self.effective_age(frame);
            // Promote only when the pre-post recurs at least `persistence` frames
            // later — a genuine re-coincidence across codes, not the same code merely
            // held active over its persistence window — and still within the fresh
            // window. Java gates on `age >= region.getPersistence()`; with the default
            // `persistence == 1` this is the original `age >= 1`.
            if age >= persistence.max(1) as i64 && age < wt.fresh_index(self.stiffness) {
                self.stiffness += 1;
            }
        }
        self.timestamp_last_pre_post = frame;
        self.included_in_transient_counts = true;
    }

    /// Promote a transiently-increased synapse to permanent (bundle freezing / end
    /// of learning). Mirrors `promoteFromTransientToPermanent` (counts handled by
    /// the caller). Returns `true` if it was transient and got promoted.
    pub fn promote_to_permanent(&mut self, frame: i64, wt: &WeightTable) -> bool {
        // Only consolidate a transiently-increased synapse that is *still contributing*
        // a non-zero weight. One that has decayed to zero must not become permanent
        // (Java's `promoteAllTransToPerm` promotes only synapses with
        // `getEffectiveValue > 0`); otherwise every past coincidence, however stale,
        // would freeze in at full strength.
        if self.included_in_transient_counts && self.is_contributing_learning(frame, wt) {
            self.included_in_transient_counts = false;
            self.stiffness = wt.max_stiffness;
            self.timestamp_last_pre_post = frame;
            true
        } else {
            false
        }
    }

    /// If a non-permanent synapse has decayed to zero and is still flagged as
    /// transient, reset it to virgin and clear the flag. Returns `true` if it was
    /// reconciled (so the caller can decrement transient counts). This is the
    /// bookkeeping side effect the Java `getEffectiveValue` performs inline.
    pub fn reconcile_if_inert(&mut self, frame: i64, wt: &WeightTable) -> bool {
        if self.is_permanent(wt) || !self.included_in_transient_counts {
            return false;
        }
        let age = self.effective_age(frame);
        if age >= 0 && age > wt.highest_pos_wt_age(self.stiffness) {
            self.included_in_transient_counts = false;
            self.stiffness = 0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> WeightTable {
        WeightTable::build(&WeightTableConfig::default())
    }

    #[test]
    fn table_is_a_step_function() {
        let wt = table();
        // stiffness 0: 127 up to 2000, 120 up to 3000, 50 up to 4000, then 0.
        assert_eq!(wt.weight(0, 0), 127);
        assert_eq!(wt.weight(0, 1999), 127);
        assert_eq!(wt.weight(0, 2000), 120);
        assert_eq!(wt.weight(0, 2999), 120);
        assert_eq!(wt.weight(0, 3000), 50);
        assert_eq!(wt.weight(0, 3999), 50);
        assert_eq!(wt.weight(0, 4000), 0);
        // stiffness 1 has a slower schedule: 127 all the way to 4000.
        assert_eq!(wt.weight(1, 3999), 127);
        assert_eq!(wt.weight(1, 4000), 120);
        assert_eq!(wt.weight(1, 7999), 50);
        assert_eq!(wt.weight(1, 8000), 0);
        // max_age = max last-transition = 8000.
        assert_eq!(wt.max_age(), 8000);
    }

    #[test]
    fn virgin_has_zero_weight() {
        let wt = table();
        let s = Synapse::new(NeuronId(0));
        assert_eq!(s.effective_value(0, &wt), 0);
        assert_eq!(s.effective_value(1_000_000, &wt), 0);
        assert!(s.effective_age(5) < 0);
    }

    #[test]
    fn fresh_pre_post_gives_max_and_then_decays() {
        let wt = table();
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(100, &wt, 1);
        assert_eq!(s.effective_value(100, &wt), 127); // age 0
        assert_eq!(s.effective_value(100 + 1999, &wt), 127);
        assert_eq!(s.effective_value(100 + 2000, &wt), 120);
        assert_eq!(s.effective_value(100 + 3000, &wt), 50);
        assert_eq!(s.effective_value(100 + 4000, &wt), 0); // decayed away
    }

    #[test]
    fn close_pre_post_promotes_stiffness() {
        let wt = table();
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(0, &wt, 1);
        assert_eq!(s.stiffness, 0);
        // Second pre-post within the fresh window (< 2000) → promote to stiffness 1.
        s.record_pre_post(500, &wt, 1);
        assert_eq!(s.stiffness, 1);
        // Third close pre-post (< fresh_index(1) = 4000) → promote to 2 = permanent.
        s.record_pre_post(1000, &wt, 1);
        assert_eq!(s.stiffness, 2);
        assert!(s.is_permanent(&wt));
        assert_eq!(s.effective_value(9_999_999, &wt), 127); // permanent never decays
    }

    #[test]
    fn far_pre_post_does_not_promote() {
        let wt = table();
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(0, &wt, 1);
        // Second pre-post past the fresh window (>= 2000) → no promotion.
        s.record_pre_post(2500, &wt, 1);
        assert_eq!(s.stiffness, 0);
    }

    #[test]
    fn freezing_promotes_transient_to_permanent() {
        let wt = table();
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(10, &wt, 1);
        assert!(s.included_in_transient_counts);
        assert!(s.promote_to_permanent(50, &wt));
        assert!(s.is_permanent(&wt));
        assert!(!s.included_in_transient_counts);
        // Already permanent → nothing to promote.
        assert!(!s.promote_to_permanent(60, &wt));
    }

    #[test]
    fn inert_synapse_reconciles_to_virgin() {
        let wt = table();
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(0, &wt, 1);
        // Well past the last breakpoint for stiffness 0 (4000).
        assert!(s.reconcile_if_inert(5000, &wt));
        assert_eq!(s.stiffness, 0);
        assert!(!s.included_in_transient_counts);
    }

    #[test]
    fn decayed_transient_is_not_promoted() {
        // Regression (over-promotion bug): a synapse that coincided once then decayed
        // to zero must NOT be consolidated to permanent by freezing / finalize.
        let wt = table();
        let mut stale = Synapse::new(NeuronId(0));
        stale.record_pre_post(0, &wt, 1);
        assert_eq!(stale.effective_value(5000, &wt), 0); // decayed past 4000 → weight 0
        assert!(!stale.promote_to_permanent(5000, &wt)); // not contributing → not promoted
        assert!(!stale.is_permanent(&wt));
        // A still-contributing transient synapse IS promoted.
        let mut fresh = Synapse::new(NeuronId(0));
        fresh.record_pre_post(0, &wt, 1);
        assert!(fresh.promote_to_permanent(100, &wt)); // age 100, weight > 0 → promoted
        assert!(fresh.is_permanent(&wt));
    }

    #[test]
    fn persistence_gate_suppresses_within_window_promotion() {
        // Regression (promotion-gate bug): with persistence P, a re-coincidence fewer
        // than P frames later must NOT promote stiffness.
        let wt = table();
        let p = 10u32;
        let mut s = Synapse::new(NeuronId(0));
        s.record_pre_post(0, &wt, p);
        s.record_pre_post(5, &wt, p); // age 5 < persistence 10 → no promotion
        assert_eq!(s.stiffness, 0);
        s.record_pre_post(20, &wt, p); // age 15 >= 10 (and < fresh window) → promote
        assert_eq!(s.stiffness, 1);
    }
}
