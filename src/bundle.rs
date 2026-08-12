//! Efferent bundles — the source-owned weight storage.
//!
//! Ported from `EfferentBundle`/`SubEfferentBundle`. In the Java, synapses are
//! grouped into per-target-block sub-bundles; for the M1 core we keep a single flat
//! synapse list per (source neuron, link), which is enough to drive learning and
//! freezing. Sub-bundle granularity can be reintroduced later if per-target-block
//! freezing is needed (see `doc/PortNotes.md`).

use crate::ids::{LinkId, NeuronId};
use crate::synapse::{Synapse, WeightTable};
use crate::types::SynapseType;

/// All synapses from one source neuron along one link.
#[derive(Clone, Debug)]
pub struct EfferentBundle {
    /// The presynaptic (source) neuron that owns this bundle.
    pub source_neuron: NeuronId,
    /// The link this bundle belongs to.
    pub link: LinkId,
    /// Synapse type of the link (cached).
    pub syn_type: SynapseType,
    /// The synapses (each onto a target neuron).
    pub synapses: Vec<Synapse>,
    /// Whether this bundle is frozen (no further learning).
    pub frozen: bool,
}

impl EfferentBundle {
    /// Create an empty bundle.
    pub fn new(source_neuron: NeuronId, link: LinkId, syn_type: SynapseType) -> Self {
        EfferentBundle {
            source_neuron,
            link,
            syn_type,
            synapses: Vec::new(),
            frozen: false,
        }
    }

    /// Fraction of synapses currently carrying a non-zero weight at `frame`.
    /// Used to decide freezing against a saturation threshold.
    pub fn increased_fraction(&self, frame: i64, wt: &WeightTable) -> f32 {
        if self.synapses.is_empty() {
            return 0.0;
        }
        let increased = self
            .synapses
            .iter()
            .filter(|s| s.is_contributing_learning(frame, wt))
            .count();
        increased as f32 / self.synapses.len() as f32
    }

    /// Number of **permanently**-increased synapses in this bundle (Java
    /// `permanentlyIncreasedSynapseCount`).
    pub fn permanent_count(&self, wt: &WeightTable) -> u32 {
        self.synapses.iter().filter(|s| s.is_permanent(wt)).count() as u32
    }

    /// Number of **transiently**-increased (still-contributing, not-yet-permanent)
    /// synapses in this bundle (Java `transientlyIncreasedSynapseCount`).
    pub fn transient_count(&self, frame: i64, wt: &WeightTable) -> u32 {
        self.synapses
            .iter()
            .filter(|s| {
                s.included_in_transient_counts
                    && !s.is_permanent(wt)
                    && s.is_contributing_learning(frame, wt)
            })
            .count() as u32
    }

    /// Freeze the bundle: promote all transiently-increased synapses to permanent.
    /// Mirrors `SubEfferentBundle.promoteAllTransToPerm`.
    pub fn freeze(&mut self, frame: i64, wt: &WeightTable) {
        for syn in &mut self.synapses {
            syn.promote_to_permanent(frame, wt);
        }
        self.frozen = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WeightTableConfig;
    use crate::ids::{LinkId, NeuronId};
    use crate::synapse::Synapse;
    use crate::types::SynapseType;

    #[test]
    fn transient_and_permanent_counts() {
        let wt = WeightTable::build(&WeightTableConfig::default());
        let mut eb = EfferentBundle::new(NeuronId(0), LinkId(0), SynapseType::U);
        for i in 0..4 {
            eb.synapses.push(Synapse::new(NeuronId(i)));
        }
        // All virgin: no transient, no permanent.
        assert_eq!(eb.transient_count(0, &wt), 0);
        assert_eq!(eb.permanent_count(&wt), 0);

        // Three coincide → transient (contributing).
        for i in 0..3 {
            eb.synapses[i].record_pre_post(10, &wt, 1);
        }
        assert_eq!(eb.transient_count(10, &wt), 3);
        assert_eq!(eb.permanent_count(&wt), 0);

        // Promote one to permanent: 1 permanent, 2 still transient.
        eb.synapses[0].promote_to_permanent(10, &wt);
        assert_eq!(eb.permanent_count(&wt), 1);
        assert_eq!(eb.transient_count(10, &wt), 2);
    }
}
