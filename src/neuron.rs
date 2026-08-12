//! Neuron records (input cells and internal cells).
//!
//! Ported from `GenericNeuron`/`InputRegionNeuron`/`InternalRegionNeuron`. All the
//! Java `owning*` back-pointers become arena indices; efferent bundles live in
//! [`crate::net::SparseyNet::efferent_bundles`] and are referenced by id here.

use crate::ids::{CmId, EfferentBundleId, RegionId};
use crate::types::SynapseType;

/// Whether a neuron is an input (leaf) cell or an internal (MAC/CM) cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NeuronKind {
    /// A raw feature cell in an input region.
    Input,
    /// A principal cell inside a CM of an internal region.
    Internal,
}

/// Per-synapse-type afferent accumulators for a neuron. Mirrors the value-only
/// `AfferentBundle` (`inputSum`, `activeInputCount`, `normalizedInputSum`,
/// `adjustedInputSum`) — the target side of the push-signal model.
#[derive(Clone, Copy, Debug, Default)]
pub struct AfferentAccum {
    /// Raw weighted input sum (`inputSum`).
    pub raw_sum: f32,
    /// Number of active presynaptic inputs via this bundle (`activeInputCount`).
    pub active_input_count: u32,
    /// Normalized sum in `[0,1]` (`normalizedInputSum`).
    pub normalized: f32,
    /// Normalized sum raised to the signal exponent (`adjustedInputSum`).
    pub adjusted: f32,
}

impl AfferentAccum {
    /// Reset for a new frame.
    pub fn reset(&mut self) {
        *self = AfferentAccum::default();
    }
}

/// A single neuron in the arena.
#[derive(Clone, Debug)]
pub struct Neuron {
    /// Input or internal.
    pub kind: NeuronKind,
    /// The region this neuron belongs to.
    pub owning_region: RegionId,
    /// For internal cells: the owning CM. `None` for input cells.
    pub owning_cm: Option<CmId>,
    /// Index of this cell within its CM (`0..K`). Unused for input cells.
    pub index_within_cm: u32,
    /// Index of this neuron within its region.
    pub index_within_region: u32,
    /// Active this frame.
    pub active: bool,
    /// Active on the previous frame (for H/D temporal signaling).
    pub prev_active: bool,
    /// Evidence value V (internal cells).
    pub v: f32,
    /// Afferent accumulators, indexed by [`SynapseType::index`].
    pub aff: [AfferentAccum; 3],
    /// Efferent bundles owned by this neuron (this cell as the presynaptic source).
    pub efferent_bundles: Vec<EfferentBundleId>,
}

impl Neuron {
    /// Create a neuron of the given kind in a region.
    pub fn new(kind: NeuronKind, region: RegionId, index_within_region: u32) -> Self {
        Neuron {
            kind,
            owning_region: region,
            owning_cm: None,
            index_within_cm: 0,
            index_within_region,
            active: false,
            prev_active: false,
            v: 0.0,
            aff: [AfferentAccum::default(); 3],
            efferent_bundles: Vec::new(),
        }
    }

    /// Mutable accessor for the afferent accumulator of a given type.
    #[inline]
    pub fn aff_mut(&mut self, ty: SynapseType) -> &mut AfferentAccum {
        &mut self.aff[ty.index()]
    }

    /// Shared accessor for the afferent accumulator of a given type.
    #[inline]
    pub fn aff(&self, ty: SynapseType) -> &AfferentAccum {
        &self.aff[ty.index()]
    }

    /// Reset per-frame state (activation carried to `prev_active` by the caller).
    pub fn reset_for_new_frame(&mut self) {
        self.v = 0.0;
        for a in &mut self.aff {
            a.reset();
        }
    }
}
