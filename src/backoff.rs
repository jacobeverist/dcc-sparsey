//! Data-driven backoff strategy for recognition.
//!
//! Ported from `BackoffStrategy.java`. Recognition computes G in progressively
//! simpler "versions" (subsets of `{U,H,D}`) and takes the first that clears its
//! threshold. The rules are data: a list of priority levels (highest first), each
//! holding one or more competing cases; each case names the synapse types to
//! combine and the G threshold to beat.
//!
//! During *learning* backoff is never used — the highest-order available version is
//! taken directly (see [`crate::net`]).

use crate::config::BackoffConfig;
use crate::types::SynapseType;

/// One backoff case: the synapse types combined for its G, and the threshold.
#[derive(Clone, Debug)]
pub struct MatchCase {
    /// Synapse types to combine for this case's V/G.
    pub syn_types: Vec<SynapseType>,
    /// G threshold this case must reach.
    pub threshold: f32,
}

/// The full strategy: priority levels (index 0 = highest priority), each a set of
/// competing cases.
#[derive(Clone, Debug, Default)]
pub struct BackoffStrategy {
    /// Priority levels, outer→inner = priority→cases.
    pub priorities: Vec<Vec<MatchCase>>,
}

/// The outcome of running backoff: which case won and its computed G.
#[derive(Clone, Debug)]
pub struct BackoffResult {
    /// Synapse types of the winning version.
    pub syn_types: Vec<SynapseType>,
    /// The G value computed for the winning version.
    pub g: f32,
    /// Whether the winning version actually met its threshold (vs. fell through to
    /// the simplest available version without clearing any threshold).
    pub met_threshold: bool,
}

impl BackoffStrategy {
    /// Build from config.
    pub fn from_config(cfg: &BackoffConfig) -> Self {
        BackoffStrategy {
            priorities: cfg
                .priorities
                .iter()
                .map(|level| {
                    level
                        .iter()
                        .map(|c| MatchCase {
                            syn_types: c.syn_types.clone(),
                            threshold: c.threshold,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// Is the strategy empty (no rules configured)?
    pub fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    /// Walk priority levels high→low. For each case whose synapse types are *all*
    /// available (active bundles), compute its G via `g_for`. Return the first case
    /// that clears its threshold. If none clears, return the highest-priority
    /// *available* case's G with `met_threshold = false` (graceful fallback).
    ///
    /// `available` reports whether a given synapse type has active input this frame;
    /// `g_for` computes G for a set of synapse types.
    pub fn evaluate<A, G>(&self, mut available: A, mut g_for: G) -> Option<BackoffResult>
    where
        A: FnMut(SynapseType) -> bool,
        G: FnMut(&[SynapseType]) -> f32,
    {
        let mut fallback: Option<BackoffResult> = None;

        for level in &self.priorities {
            // Within a priority level, among the *available* cases that clear their
            // threshold, keep the one with the highest G (Java keeps the max-G case per
            // level, not the first to clear). Only descend to the next level if no case
            // in this level clears.
            let mut best: Option<BackoffResult> = None;
            for case in level {
                if !case.syn_types.iter().all(|&t| available(t)) {
                    continue;
                }
                let g = g_for(&case.syn_types);
                // Highest-priority available case, as a graceful fallback if nothing
                // clears any threshold at all.
                if fallback.is_none() {
                    fallback = Some(BackoffResult {
                        syn_types: case.syn_types.clone(),
                        g,
                        met_threshold: false,
                    });
                }
                if g >= case.threshold && best.as_ref().is_none_or(|b| g > b.g) {
                    best = Some(BackoffResult {
                        syn_types: case.syn_types.clone(),
                        g,
                        met_threshold: true,
                    });
                }
            }
            if best.is_some() {
                return best;
            }
        }

        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BackoffConfig;
    use SynapseType::{D, H, U};

    fn strat() -> BackoffStrategy {
        BackoffStrategy::from_config(&BackoffConfig::canonical(0.9, 0.93, 0.96))
    }

    #[test]
    fn picks_highest_order_when_it_clears() {
        let s = strat();
        // All types available; HUD scores 0.95 ≥ 0.9 → HUD wins.
        let r = s
            .evaluate(|_| true, |types| if types == [H, U, D] { 0.95 } else { 0.0 })
            .unwrap();
        assert_eq!(r.syn_types, vec![H, U, D]);
        assert!(r.met_threshold);
    }

    #[test]
    fn backs_off_when_high_order_misses() {
        let s = strat();
        // HUD only 0.5 (< 0.9); HU scores 0.94 (≥ 0.93) → backs off to HU.
        let r = s
            .evaluate(
                |_| true,
                |types| match types {
                    [H, U, D] => 0.5,
                    [H, U] => 0.94,
                    _ => 0.0,
                },
            )
            .unwrap();
        assert_eq!(r.syn_types, vec![H, U]);
        assert!(r.met_threshold);
    }

    #[test]
    fn picks_max_g_case_within_a_level() {
        // Level 1 of the canonical strategy holds both HU and UD at threshold 0.93.
        // HUD misses; HU=0.94 and UD=0.96 both clear → the max-G case (UD) wins, not
        // the first-listed (HU).
        let s = strat();
        let r = s
            .evaluate(
                |_| true,
                |types| match types {
                    [H, U, D] => 0.5,
                    [H, U] => 0.94,
                    [U, D] => 0.96,
                    _ => 0.0,
                },
            )
            .unwrap();
        assert_eq!(r.syn_types, vec![U, D], "max-G case within a level wins");
        assert!(r.met_threshold);
    }

    #[test]
    fn full_backoff_has_all_combinations() {
        use crate::config::BackoffConfig;
        let s = BackoffStrategy::from_config(&BackoffConfig::canonical_full(0.9, 0.93, 0.96));
        // 3 levels: {HUD} / {HU,UD,HD} / {H,U,D} = 7 cases total.
        let total: usize = s.priorities.iter().map(|l| l.len()).sum();
        assert_eq!(total, 7);
        // HD backs off correctly when only H and D are available.
        let r = s
            .evaluate(
                |t| t == H || t == D,
                |types| if types == [H, D] { 0.95 } else { 0.0 },
            )
            .unwrap();
        assert_eq!(r.syn_types, vec![H, D]);
        assert!(r.met_threshold);
    }

    #[test]
    fn skips_unavailable_types_and_falls_through() {
        let s = strat();
        // Only U available (no H, no D). HUD and HU/UD cases are skipped; U scores
        // 0.5 (< 0.96) so nothing clears → fallback is U, met_threshold=false.
        let r = s
            .evaluate(|t| t == U, |types| if types == [U] { 0.5 } else { 0.0 })
            .unwrap();
        assert_eq!(r.syn_types, vec![U]);
        assert!(!r.met_threshold);
    }
}
