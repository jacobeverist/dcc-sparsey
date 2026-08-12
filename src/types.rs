//! Small shared enums used across config, links, bundles and synapses.

use serde::{Deserialize, Serialize};

/// The three synapse/signal types. A connection's type is inferred from the
/// relative DAG heights of its endpoint regions (see [`crate::link`]).
///
/// Mirrors SparseyCore's `Bundle.SYNAPSE_TYPE_{H,U,D}` (which are `0,1,2`); we keep
/// the same discriminants so any serialized indices line up.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
#[repr(u8)]
pub enum SynapseType {
    /// Horizontal — same-height (lateral / recurrent) connection.
    H = 0,
    /// Upward — feedforward, source lower than target in the DAG.
    U = 1,
    /// Downward — feedback, source higher than target in the DAG.
    D = 2,
}

impl SynapseType {
    /// All three types in a stable order (`H, U, D`).
    pub const ALL: [SynapseType; 3] = [SynapseType::H, SynapseType::U, SynapseType::D];

    /// Index into a per-type array (`H=0, U=1, D=2`).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Operation mode of a run. Mirrors `Network.{LEARNING,RECOGNITION,RECALL}_MODE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum OperationMode {
    /// Weights increase; persistence not trumpable; no backoff.
    Learning,
    /// Persistence trumpable; data-driven backoff; winners by max V.
    Recognition,
    /// D-signals propagate down to regenerate a sequence (deferred in M1).
    Recall,
}
