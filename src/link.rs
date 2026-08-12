//! Inter-region links.
//!
//! Ported from `Link.java`. A link connects a source region to a target region and
//! carries one [`SynapseType`], derived from the endpoints' DAG heights in
//! [`crate::net::SparseyNet::build`] exactly as `Network.createInterRegionLinks`
//! does. Per-type numeric parameters (exps, cutoffs, saturation, MCH) are read from
//! the target region's [`crate::config::SignalParams`] rather than duplicated here.

use crate::ids::RegionId;
use crate::types::SynapseType;

/// A connection between two regions.
#[derive(Clone, Debug)]
pub struct Link {
    /// Source (presynaptic) region.
    pub source_region: RegionId,
    /// Target (postsynaptic) region.
    pub target_region: RegionId,
    /// Synapse type inferred from relative DAG height.
    pub syn_type: SynapseType,
    /// Whether signals use the source's *previous*-frame activity (`H`/`D`) rather
    /// than the current frame (`U`). Mirrors `Link.usePreviousActive`.
    pub use_previous_active: bool,
}

impl Link {
    /// Derive the synapse type for a source→target pair from their DAG heights
    /// (`src < tgt ⇒ U`, `==` ⇒ `H`, `>` ⇒ `D`).
    pub fn syn_type_from_heights(source_height: u32, target_height: u32) -> SynapseType {
        use std::cmp::Ordering::*;
        match source_height.cmp(&target_height) {
            Less => SynapseType::U,
            Equal => SynapseType::H,
            Greater => SynapseType::D,
        }
    }
}
