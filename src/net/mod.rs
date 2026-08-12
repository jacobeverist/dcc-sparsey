//! The network arena and its public per-frame API.
//!
//! `SparseyNet` owns every entity in flat `Vec`s and is the run context (global
//! frame counter + operation mode, which were Java statics). Construction lives in
//! [`build`](mod@self), the per-frame pipeline in [`frame`](mod@self); this file
//! holds the arena struct and the small public surface (`set_input`, `mac_code`,
//! serialization, `prepare_for_new_run`, `finalize_learning`).

use std::collections::HashMap;

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::backoff::BackoffStrategy;
use crate::bundle::EfferentBundle;
use crate::cm::Cm;
use crate::config::NetworkConfig;
use crate::error::{SparseyError, SparseyResult};
use crate::ids::{MacId, RegionId};
use crate::link::Link;
use crate::mac::Mac;
use crate::neuron::{AfferentAccum, Neuron};
use crate::region::{Region, RegionKind};
use crate::synapse::{Synapse, WeightTable};
use crate::types::OperationMode;

mod build;
mod frame;

/// Serializable learned-state snapshot (structure excluded — rebuilt from config).
/// Per efferent bundle (in arena order), a `(stiffness, timestamp)` per synapse.
#[derive(serde::Serialize, serde::Deserialize)]
struct StateSnapshot {
    global_frame: i64,
    bundles: Vec<Vec<(u8, i64)>>,
}

/// A fully-built Sparsey network plus its run state.
pub struct SparseyNet {
    /// The configuration this net was built from.
    pub config: NetworkConfig,
    /// The weight decay table.
    pub weight_table: WeightTable,

    /// Regions (DAG nodes).
    pub regions: Vec<Region>,
    /// Macrocolumns.
    pub macs: Vec<Mac>,
    /// Competitive modules.
    pub cms: Vec<Cm>,
    /// All neurons (input + internal).
    pub neurons: Vec<Neuron>,
    /// Inter-region links.
    pub links: Vec<Link>,
    /// Efferent bundles (source-owned weight storage).
    pub efferent_bundles: Vec<EfferentBundle>,
    /// Per-region backoff strategy (empty for input regions / no rules).
    pub backoffs: Vec<BackoffStrategy>,

    /// Global frame counter across all episodes (weight-age clock).
    pub global_frame: i64,
    /// Current operation mode.
    pub op_mode: OperationMode,

    region_by_name: HashMap<String, RegionId>,
    rng: Xoshiro256PlusPlus,

    /// Reused per-frame signal-push scratch, dense and indexed by target neuron —
    /// avoids a per-region `HashMap` allocation and per-synapse hashing in the hot
    /// loop. Grown to `neurons.len()`; only the pushed-into region's cells are written,
    /// and they are applied + reset in one pass, so it stays clean between pushes.
    push_scratch: Vec<[AfferentAccum; 3]>,
}

impl SparseyNet {
    /// Look up a region id by name.
    pub fn region_id(&self, name: &str) -> Option<RegionId> {
        self.region_by_name.get(name).copied()
    }

    /// Set the active input features of an input region (by cell index within the
    /// region). Clears the region's previous activity first.
    pub fn set_input(&mut self, region: RegionId, active: &[u32]) -> SparseyResult<()> {
        if self.regions[region.index()].kind != RegionKind::Input {
            return Err(SparseyError::InvalidParameter(format!(
                "set_input called on non-input region '{}'",
                self.regions[region.index()].name
            )));
        }
        let cells = self.regions[region.index()].cells.clone();
        for &c in &cells {
            self.neurons[c.index()].active = false;
        }
        for &idx in active {
            let cell = *cells.get(idx as usize).ok_or(SparseyError::IndexOutOfBounds {
                index: idx as usize,
                length: cells.len(),
            })?;
            self.neurons[cell.index()].active = true;
        }
        Ok(())
    }

    /// The current code of a MAC: the winning cell index (`0..K`) for each of its
    /// CMs, or `None` if the MAC is inactive / unselected.
    pub fn mac_code(&self, mac: MacId) -> Option<Vec<u32>> {
        let m = &self.macs[mac.index()];
        if !m.active {
            return None;
        }
        let mut code = Vec::with_capacity(m.cms.len());
        for &cid in &m.cms {
            let w = self.cms[cid.index()].winner?;
            code.push(self.neurons[w.index()].index_within_cm);
        }
        Some(code)
    }

    /// Serialize the network's learned state (per-synapse stiffness + last-pre-post
    /// timestamp, and the global frame) to bytes. Structure is *not* included — it is
    /// rebuilt from the [`NetworkConfig`]; only the learned weights are captured.
    /// Synapses are serialized in arena order, which is deterministic for a given
    /// config, so [`Self::load_state`] can zip them straight back.
    pub fn serialize_state(&self) -> SparseyResult<Vec<u8>> {
        let snap = StateSnapshot {
            global_frame: self.global_frame,
            bundles: self
                .efferent_bundles
                .iter()
                .map(|eb| {
                    eb.synapses
                        .iter()
                        .map(|s| (s.stiffness, s.timestamp_last_pre_post))
                        .collect()
                })
                .collect(),
        };
        bincode::serialize(&snap)
            .map_err(|e| SparseyError::Other(format!("serialize_state: {e}")))
    }

    /// Restore learned state produced by [`Self::serialize_state`] into a network
    /// built from the same config.
    pub fn load_state(&mut self, bytes: &[u8]) -> SparseyResult<()> {
        let snap: StateSnapshot = bincode::deserialize(bytes)
            .map_err(|e| SparseyError::Other(format!("load_state: {e}")))?;
        if snap.bundles.len() != self.efferent_bundles.len() {
            return Err(SparseyError::Other(format!(
                "load_state: bundle count mismatch ({} vs {}) — config differs",
                snap.bundles.len(),
                self.efferent_bundles.len()
            )));
        }
        self.global_frame = snap.global_frame;
        for (eb, syns) in self.efferent_bundles.iter_mut().zip(snap.bundles) {
            for (syn, (stiffness, ts)) in eb.synapses.iter_mut().zip(syns) {
                syn.stiffness = stiffness;
                syn.timestamp_last_pre_post = ts;
            }
        }
        Ok(())
    }

    /// Reset per-run dynamic state (activity, code ages, winners). Optionally erase
    /// all learned weights. Does not change the frame clock's monotonicity.
    pub fn prepare_for_new_run(&mut self, erase_weights: bool) {
        for n in &mut self.neurons {
            n.active = false;
            n.prev_active = false;
            n.v = 0.0;
            for a in &mut n.aff {
                a.reset();
            }
        }
        for m in &mut self.macs {
            m.active = false;
            m.code_age = 0;
        }
        for cm in &mut self.cms {
            cm.winner = None;
            cm.prev_winner = None;
        }
        if erase_weights {
            for eb in &mut self.efferent_bundles {
                eb.frozen = false;
                for syn in &mut eb.synapses {
                    *syn = Synapse::new(syn.target_neuron);
                }
            }
        }
    }

    /// End-of-learning pass: promote every transiently-increased synapse to
    /// permanent, so recognition sees learned weights at full strength regardless of
    /// age. Mirrors `Network.doFinalTransToPermSynapsePromotionPass`.
    pub fn finalize_learning(&mut self) {
        let frame = self.global_frame;
        let wt = &self.weight_table;
        for eb in &mut self.efferent_bundles {
            for syn in &mut eb.synapses {
                syn.promote_to_permanent(frame, wt);
            }
        }
    }
}
