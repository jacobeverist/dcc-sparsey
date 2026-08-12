//! Region records (DAG nodes).
//!
//! Ported from `GenericRegion`/`InputRegion`/`InternalRegion` (data). The
//! per-frame processing lives in [`crate::net`].

use crate::ids::{LinkId, MacId, NeuronId};

/// Whether a region is an input (leaf) region or an internal region.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegionKind {
    /// Leaf region (DAG height 0): raw feature cells.
    Input,
    /// Internal region (height ≥ 1): a grid of MACs.
    Internal,
}

/// One region in the network DAG.
#[derive(Clone, Debug)]
pub struct Region {
    /// Region name.
    pub name: String,
    /// Input or internal.
    pub kind: RegionKind,
    /// Index of the source [`crate::config::RegionConfig`] in the network config.
    pub config_index: usize,
    /// Height in the DAG (0 = input).
    pub height_in_dag: u32,
    /// Block-grid width.
    pub width_in_blks: u32,
    /// Block-grid height.
    pub height_in_blks: u32,
    /// CMs per MAC (`Q`); 0 for input regions.
    pub q: u32,
    /// Cells per CM (`K`); for input regions, cells per feature block.
    pub k: u32,
    /// Persistence (frames a code stays active).
    pub persistence: u32,
    /// MACs (internal regions).
    pub macs: Vec<MacId>,
    /// All cells belonging to this region (input cells, or every internal cell).
    pub cells: Vec<NeuronId>,
    /// Links into this region (afferent).
    pub afferent_links: Vec<LinkId>,
    /// Links out of this region (efferent).
    pub efferent_links: Vec<LinkId>,

    /// Resolved activation band (absolute active-U-feature counts) for MACs of this
    /// region — `active_{low,high}_frac × U afferent input size`, computed at build.
    /// A MAC is eligible only if its active U feature count is within `[low, high]`.
    /// Defaults `[1, u32::MAX]` (any U input) until resolved.
    pub active_low: u32,
    /// Resolved activation-band upper bound (`u32::MAX` ⇒ no cap).
    pub active_high: u32,
}

impl Region {
    /// Is this an input (leaf) region?
    #[inline]
    pub fn is_input(&self) -> bool {
        self.kind == RegionKind::Input
    }
}
