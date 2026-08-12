//! # sparsey
//!
//! A clean-room Rust reimplementation of the **Sparsey** neural network — a
//! hierarchical sparse-distributed-memory (SDM) architecture modeled after
//! neocortical macrocolumns. Ported from the Java reference implementation
//! `SparseyCore` (see `doc/PortNotes.md`), following the algorithm as documented
//! in that project's `ARCHITECTURE.md` rather than translating the Java classes
//! verbatim.
//!
//! ## Model in one paragraph
//!
//! A network is a **DAG of regions**. Leaf regions ([`region::RegionKind::Input`])
//! present raw features; internal regions hold a 2-D grid of **macrocolumns (MACs)**.
//! Each MAC contains `Q` **competitive modules (CMs)**, and each CM contains `K`
//! binary **cells** in winner-take-all competition. A MAC's active **code** is one
//! winning cell per CM — i.e. a one-hot-grouped sparse pattern of `Q·K` bits.
//! Connections between regions carry `U` (up), `H` (horizontal) or `D` (down)
//! signals, the type inferred from the regions' relative DAG height. Weights are
//! stored implicitly via per-synapse timestamps + stiffness and decay by table
//! lookup. See the module docs for details.
//!
//! ## Object model
//!
//! To avoid the cyclic, back-pointer-laden object graph of the Java version, every
//! entity lives in a flat arena on [`net::SparseyNet`] and is referenced by a typed
//! index newtype ([`ids`]). There is no `Rc<RefCell<…>>`.

pub mod backoff;
pub mod bundle;
pub mod cm;
pub mod codetrie;
pub mod config;
pub mod error;
pub mod ids;
pub mod link;
pub mod mac;
pub mod net;
pub mod neuron;
pub mod recorder;
pub mod region;
pub mod synapse;
pub mod types;

pub use config::{NetworkConfig, NetworkConfigBuilder, RegionConfig, RegionConfigBuilder};
pub use error::{SparseyError, SparseyResult};
pub use net::SparseyNet;
pub use recorder::{NullRecorder, Recorder};
pub use types::{OperationMode, SynapseType};
