//! Error type for the `sparsey` crate.
//!
//! Modeled on dcc-core's `DCC`/`DCCResult` convention (`engine/src/error.rs`) but
//! crate-local so `sparsey` stays a dependency-light, standalone library. The
//! dcc-core adapter Node (Phase 2) converts these into `DCCResult` at the boundary.

use thiserror::Error;

/// Errors produced while building or running a Sparsey network.
#[derive(Error, Debug)]
pub enum SparseyError {
    /// A configuration value was missing, malformed, or internally inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A parameter was outside its permitted range.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Network construction failed (e.g. a connection referenced an unknown region,
    /// or the region graph was not a DAG).
    #[error("network build error: {0}")]
    Build(String),

    /// An index (region/mac/cm/neuron/synapse) was out of range.
    #[error("index out of bounds: {index} (len {length})")]
    IndexOutOfBounds {
        /// The offending index.
        index: usize,
        /// The length of the collection being indexed.
        length: usize,
    },

    /// Underlying I/O error (config load/save).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error for configs.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Catch-all for anything not worth its own variant.
    #[error("{0}")]
    Other(String),
}

/// Convenient result alias used throughout the crate.
pub type SparseyResult<T> = Result<T, SparseyError>;
