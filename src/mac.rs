//! Macrocolumn records.
//!
//! Ported from `Mac.java` (data + code/activation bookkeeping). The signal/V/G/
//! winner algorithms live in [`crate::net`].

use crate::codetrie::CodeTrie;
use crate::ids::{CmId, RegionId};

/// A macrocolumn: `Q` competitive modules whose winners form its code.
#[derive(Clone, Debug)]
pub struct Mac {
    /// Owning region.
    pub owning_region: RegionId,
    /// Index of this MAC within its region.
    pub index: u32,
    /// Column position in the region's block grid.
    pub col: u32,
    /// Row position in the region's block grid.
    pub row: u32,
    /// The `Q` CMs of this MAC.
    pub cms: Vec<CmId>,
    /// Whether the MAC is active this frame.
    pub active: bool,
    /// Frames since the current code was selected (0 = eligible / inactive).
    pub code_age: u32,
    /// Global match G of the current frame's winning version.
    pub g: f32,
    /// Number of concurrent hypotheses represented by the current code.
    pub num_mch: u32,
    /// Codes this MAC has learned.
    pub learned_codes: CodeTrie,

    /// CSA sigmoid inflection point, ratcheting from `min_inflect` toward `max_inflect`
    /// as this MAC saturates (`Mac.determine_Inflection_Point`). Unused under max-V.
    pub inflect_point: f32,
    /// Highest mean-`V_ave` (across this MAC's CMs) seen so far — gates the ratchet.
    pub max_mean_v_ave: f32,
    /// `num_mch` rolled from the previous frame (for prev-active H/D signal discount).
    pub prev_num_mch: u32,
}

impl Mac {
    /// Create an empty MAC (CMs filled in by the builder).
    pub fn new(owning_region: RegionId, index: u32, col: u32, row: u32) -> Self {
        Mac {
            owning_region,
            index,
            col,
            row,
            cms: Vec::new(),
            active: false,
            code_age: 0,
            g: 0.0,
            num_mch: 0,
            learned_codes: CodeTrie::new(),
            inflect_point: 0.0,
            max_mean_v_ave: 0.0,
            prev_num_mch: 1,
        }
    }
}
