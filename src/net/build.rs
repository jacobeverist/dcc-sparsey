//! Network construction: region / MAC / CM / neuron arena build + connectivity wiring.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::backoff::BackoffStrategy;
use crate::bundle::EfferentBundle;
use crate::cm::Cm;
use crate::config::NetworkConfig;
use crate::error::{SparseyError, SparseyResult};
use crate::ids::{CmId, EfferentBundleId, LinkId, MacId, NeuronId, RegionId};
use crate::link::Link;
use crate::mac::Mac;
use crate::neuron::{Neuron, NeuronKind};
use crate::region::{Region, RegionKind};
use crate::synapse::{Synapse, WeightTable};
use crate::types::{OperationMode, SynapseType};

use super::SparseyNet;

impl SparseyNet {
    /// Build a network from `config`, seeding winner-selection RNG with `seed`.
    pub fn build(config: NetworkConfig, seed: u64) -> SparseyResult<Self> {
        let weight_table = WeightTable::build(&config.weight_table);
        let mut net = SparseyNet {
            weight_table,
            regions: Vec::new(),
            macs: Vec::new(),
            cms: Vec::new(),
            neurons: Vec::new(),
            links: Vec::new(),
            efferent_bundles: Vec::new(),
            backoffs: Vec::new(),
            global_frame: 0,
            op_mode: OperationMode::Learning,
            region_by_name: HashMap::new(),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            push_scratch: Vec::new(),
            config,
        };
        net.build_regions()?;
        net.build_links()?;
        net.resolve_activation_bands();
        net.wire_connectivity();
        Ok(net)
    }

    // ---- construction ------------------------------------------------------

    fn build_regions(&mut self) -> SparseyResult<()> {
        // Clone the region configs out so we can mutate `self` while iterating.
        let region_cfgs = self.config.regions.clone();
        for (cfg_index, rc) in region_cfgs.iter().enumerate() {
            let rid = RegionId(self.regions.len());
            if self.region_by_name.insert(rc.name.clone(), rid).is_some() {
                return Err(SparseyError::InvalidConfig(format!(
                    "duplicate region name '{}'",
                    rc.name
                )));
            }
            let kind = if rc.is_input() {
                RegionKind::Input
            } else {
                RegionKind::Internal
            };
            let mut region = Region {
                name: rc.name.clone(),
                kind,
                config_index: cfg_index,
                height_in_dag: rc.height_in_dag,
                width_in_blks: rc.width_in_blks,
                height_in_blks: rc.height_in_blks,
                q: rc.q,
                k: rc.k,
                persistence: rc.persistence.max(1),
                macs: Vec::new(),
                cells: Vec::new(),
                afferent_links: Vec::new(),
                efferent_links: Vec::new(),
                active_low: 1,
                active_high: u32::MAX,
            };

            match kind {
                RegionKind::Input => {
                    // One input cell per block position.
                    let n = (rc.width_in_blks * rc.height_in_blks) as usize;
                    for i in 0..n {
                        let nid = NeuronId(self.neurons.len());
                        self.neurons
                            .push(Neuron::new(NeuronKind::Input, rid, i as u32));
                        region.cells.push(nid);
                    }
                }
                RegionKind::Internal => {
                    let num_macs = rc.width_in_blks * rc.height_in_blks;
                    for m in 0..num_macs {
                        let col = m % rc.width_in_blks;
                        let row = m / rc.width_in_blks;
                        let mac_id = MacId(self.macs.len());
                        let mut mac = Mac::new(rid, m, col, row);
                        mac.inflect_point = rc.sigmoid.min_inflect;
                        for cm_ix in 0..rc.q {
                            let cm_id = CmId(self.cms.len());
                            let mut cm = Cm::new(mac_id, rid, cm_ix);
                            for cell_ix in 0..rc.k {
                                let nid = NeuronId(self.neurons.len());
                                let mut cell =
                                    Neuron::new(NeuronKind::Internal, rid, self_region_cell_index(&region));
                                cell.owning_cm = Some(cm_id);
                                cell.index_within_cm = cell_ix;
                                self.neurons.push(cell);
                                cm.neurons.push(nid);
                                region.cells.push(nid);
                            }
                            self.cms.push(cm);
                            mac.cms.push(cm_id);
                        }
                        self.macs.push(mac);
                        region.macs.push(mac_id);
                    }
                }
            }

            // Backoff strategy for this region.
            self.backoffs.push(BackoffStrategy::from_config(&rc.backoff));
            self.regions.push(region);
        }
        Ok(())
    }

    fn build_links(&mut self) -> SparseyResult<()> {
        let conns = self.config.connections.clone();
        for c in &conns {
            let src = *self.region_by_name.get(&c.source).ok_or_else(|| {
                SparseyError::Build(format!("connection references unknown region '{}'", c.source))
            })?;
            let tgt = *self.region_by_name.get(&c.target).ok_or_else(|| {
                SparseyError::Build(format!("connection references unknown region '{}'", c.target))
            })?;
            let sh = self.regions[src.index()].height_in_dag;
            let th = self.regions[tgt.index()].height_in_dag;
            let syn_type = Link::syn_type_from_heights(sh, th);
            let use_previous_active = syn_type != SynapseType::U;
            let lid = LinkId(self.links.len());
            self.links.push(Link {
                source_region: src,
                target_region: tgt,
                syn_type,
                use_previous_active,
            });
            self.regions[src.index()].efferent_links.push(lid);
            self.regions[tgt.index()].afferent_links.push(lid);
        }
        Ok(())
    }

    /// Resolve each region's activation band (`Region::active_low`/`active_high`) from
    /// its config fractions and its **U afferent input size** — the total source-cell
    /// count over its U afferent links. A MAC is then eligible only if its active
    /// U-feature count is within `[low, high]` (SparseyCore
    /// `ActiveInputFeatures{Low,High}BoundAsFrac`). Defaults `[0.0, 1.0]` ⇒ `[1, MAX]`
    /// (any U input), preserving prior behavior.
    fn resolve_activation_bands(&mut self) {
        for rid_ix in 0..self.regions.len() {
            let cfg_ix = self.regions[rid_ix].config_index;
            let low_frac = self.config.regions[cfg_ix].active_low_frac;
            let high_frac = self.config.regions[cfg_ix].active_high_frac;
            let aff = self.regions[rid_ix].afferent_links.clone();
            let mut u_input: u32 = 0;
            for lid in aff {
                let link = &self.links[lid.index()];
                if link.syn_type == SynapseType::U {
                    u_input += self.regions[link.source_region.index()].cells.len() as u32;
                }
            }
            let low = ((low_frac * u_input as f32).round() as u32).max(1);
            let high = if high_frac >= 1.0 {
                u32::MAX
            } else {
                ((high_frac * u_input as f32).round() as u32).max(low)
            };
            self.regions[rid_ix].active_low = low;
            self.regions[rid_ix].active_high = high;
        }
    }

    /// Wire synapses for every link.
    ///
    /// Connectivity is **band-limited projective-field** when the target region's
    /// signal params for the link type carry `band_thickness` / `band_rates`
    /// (SparseyCore `buildBlockMatrix` + `readBandInfo`): a source block connects to a
    /// target block only if their normalized grid distance falls within a band, and
    /// then each candidate synapse is created with that band's `rate` (via the seeded
    /// RNG). With empty bands — the default — it is **full within-link connectivity**
    /// (rate 1, no RNG draws), so unbanded configs stay deterministic and unchanged.
    ///
    /// Bands are cumulative outer radii in normalized `[0, √2]` grid-distance units
    /// (both regions' blocks are placed on the unit square by grid position), so the
    /// same band config is dimension-independent across region sizes.
    ///
    /// Fidelity note: SparseyCore draws exactly `round(rate·K)` sources per target
    /// without replacement; here each candidate `(source cell, target cell)` synapse is
    /// an independent `Bernoulli(rate)` draw — the same expected density and spatial
    /// banding. Normalization divides by the *actual* fan-in (`active_input_count`), so
    /// the variable per-target fan-in does not affect coding. See `doc/MethodFidelity.md`.
    fn wire_connectivity(&mut self) {
        for lid_ix in 0..self.links.len() {
            let lid = LinkId(lid_ix);
            let link = self.links[lid_ix].clone();

            // Band config from the target region's signal params for this link type.
            let tgt_cfg = self.regions[link.target_region.index()].config_index;
            let params = self.config.regions[tgt_cfg].signal(link.syn_type).clone();
            let outer_radii = cumulative_radii(&params.band_thickness);
            let band_limited = !outer_radii.is_empty() && !params.band_rates.is_empty();

            let src_cells = self.regions[link.source_region.index()].cells.clone();
            let tgt_cells = self.regions[link.target_region.index()].cells.clone();
            for sc in src_cells {
                let ebid = EfferentBundleId(self.efferent_bundles.len());
                let mut eb = EfferentBundle::new(sc, lid, link.syn_type);
                let sc_cm = self.neurons[sc.index()].owning_cm;
                let sc_pos = self.block_center_norm(sc, link.source_region);
                for &tc in &tgt_cells {
                    // A horizontal (lateral) link must not connect a cell to itself or
                    // to a same-CM winner-take-all competitor (Java `buildBlockMatrix`
                    // excludes both) — either would feed a cell's own / its
                    // competitors' activity back into its horizontal evidence.
                    if link.syn_type == SynapseType::H
                        && (tc == sc
                            || (sc_cm.is_some() && self.neurons[tc.index()].owning_cm == sc_cm))
                    {
                        continue;
                    }
                    if band_limited {
                        let tc_pos = self.block_center_norm(tc, link.target_region);
                        let d = ((sc_pos.0 - tc_pos.0).powi(2) + (sc_pos.1 - tc_pos.1).powi(2))
                            .sqrt();
                        // First band whose outer radius covers the distance.
                        let Some(b) = outer_radii.iter().position(|&r| d <= r) else {
                            continue; // beyond all bands → no connection
                        };
                        let rate = params.band_rates.get(b).copied().unwrap_or(0.0);
                        if rate <= 0.0 {
                            continue;
                        }
                        if rate < 1.0 && self.rng.random::<f32>() >= rate {
                            continue;
                        }
                    }
                    eb.synapses.push(Synapse::new(tc));
                }
                self.efferent_bundles.push(eb);
                self.neurons[sc.index()].efferent_bundles.push(ebid);
            }
        }
    }

    /// Normalized `[0,1]²` center of the block owning `cell`, within `region`: the MAC
    /// position for internal regions, the aperture/cell grid position for input
    /// regions. Used for band-limited projective-field distances.
    fn block_center_norm(&self, cell: NeuronId, region: RegionId) -> (f32, f32) {
        let r = &self.regions[region.index()];
        let w = r.width_in_blks.max(1);
        let h = r.height_in_blks.max(1);
        let (col, row) = match self.neurons[cell.index()].owning_cm {
            Some(cm) => {
                let m = &self.macs[self.cms[cm.index()].owning_mac.index()];
                (m.col, m.row)
            }
            None => {
                let idx = self.neurons[cell.index()].index_within_region;
                (idx % w, idx / w)
            }
        };
        ((col as f32 + 0.5) / w as f32, (row as f32 + 0.5) / h as f32)
    }
}

/// Cumulative outer radii from per-band thicknesses (`[t0,t1,…] → [t0, t0+t1, …]`).
fn cumulative_radii(thickness: &[f32]) -> Vec<f32> {
    let mut acc = 0.0f32;
    thickness
        .iter()
        .map(|&t| {
            acc += t;
            acc
        })
        .collect()
}


/// Helper: the next per-region cell index equals the current region cell count.
#[inline]
fn self_region_cell_index(region: &Region) -> u32 {
    region.cells.len() as u32
}
