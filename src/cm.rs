//! Competitive module (minicolumn) records.
//!
//! Ported from `CM.java` (data + winner bookkeeping). The winner-selection and
//! V-max logic lives in [`crate::net`] where it can see the neuron arena.

use crate::ids::{MacId, NeuronId, RegionId};

/// A competitive module: `K` cells competing winner-take-all.
#[derive(Clone, Debug)]
pub struct Cm {
    /// Owning MAC.
    pub owning_mac: MacId,
    /// Owning region.
    pub owning_region: RegionId,
    /// Index of this CM within its MAC (`0..Q`).
    pub index_within_mac: u32,
    /// The `K` cells of this CM.
    pub neurons: Vec<NeuronId>,
    /// Current winner (set during winner selection).
    pub winner: Option<NeuronId>,
    /// Winner from the previous frame (for exclude-prev-winner + H timing).
    pub prev_winner: Option<NeuronId>,
    /// Max V across the CM's cells (last computed).
    pub v_max: f32,
    /// Mean V across the CM's cells (last computed).
    pub v_ave: f32,
    /// Number of cells tied at `v_max`.
    pub tied_max_count: u32,
    /// Number of cells qualifying as hypotheses (V ≥ region V_thresh) — MCH count.
    pub num_mch: u32,
    /// Number of cells at/above the hypothesis threshold (MCH count contribution).
    pub num_hypotheses: u32,
}

impl Cm {
    /// Create an empty CM (cells filled in by the builder).
    pub fn new(owning_mac: MacId, owning_region: RegionId, index_within_mac: u32) -> Self {
        Cm {
            owning_mac,
            owning_region,
            index_within_mac,
            neurons: Vec::new(),
            winner: None,
            prev_winner: None,
            v_max: 0.0,
            v_ave: 0.0,
            tied_max_count: 0,
            num_mch: 0,
            num_hypotheses: 0,
        }
    }
}
