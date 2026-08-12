//! The per-frame drivers and the winner-selection / learning pipeline.

use rand::Rng;

use crate::config::{SigmoidConfig, SignalParams};
use crate::ids::{MacId, NeuronId, RegionId};
use crate::neuron::NeuronKind;
use crate::recorder::{NullRecorder, Recorder};
use crate::region::RegionKind;
use crate::synapse::MAX_WEIGHT;
use crate::types::{OperationMode, SynapseType};

use super::SparseyNet;

impl SparseyNet {
    pub fn do_frame_learn(&mut self) {
        let mut rec = NullRecorder;
        self.do_frame_learn_rec(&mut rec);
    }

    /// Process one learning frame, reporting to `rec`.
    pub fn do_frame_learn_rec(&mut self, rec: &mut dyn Recorder) {
        self.op_mode = OperationMode::Learning;
        self.process_regions(false, rec);
        self.update_weights();
        self.freeze_saturated_bundles();
        self.end_frame(rec);
    }

    /// Process one recognition frame (no weight change), default recorder.
    pub fn do_frame_recognize(&mut self) {
        let mut rec = NullRecorder;
        self.do_frame_recognize_rec(&mut rec);
    }

    /// Process one recognition frame, reporting to `rec`.
    pub fn do_frame_recognize_rec(&mut self, rec: &mut dyn Recorder) {
        self.op_mode = OperationMode::Recognition;
        self.process_regions(true, rec);
        self.end_frame(rec);
    }

    /// Process one recall frame (default recorder).
    pub fn do_frame_recall(&mut self) {
        let mut rec = NullRecorder;
        self.do_frame_recall_rec(&mut rec);
    }

    /// Process one recall frame, reporting to `rec`.
    ///
    /// Recall regenerates a learned sequence from a cue. Internal regions are driven
    /// by the available signals (retrieval-style winner selection) under the Recall
    /// op-mode, then D-signals propagate downward so lower regions regenerate from
    /// higher-region context.
    ///
    /// NOTE (M1 fidelity): the bottom-up retrieval pass and the Recall op-mode are in
    /// place; faithful D-signal regeneration of the *input region's* features (the
    /// full "replay to L0" behavior of the Java `doFrameRecall`) is a documented
    /// follow-on — see `doc/PortNotes.md`. For nets without downward (D) links this
    /// behaves like recognition.
    pub fn do_frame_recall_rec(&mut self, rec: &mut dyn Recorder) {
        self.op_mode = OperationMode::Recall;
        self.process_regions(true, rec);
        self.end_frame(rec);
    }

    // ---- per-frame internals ----------------------------------------------

    /// Process every internal region in ascending DAG height.
    fn process_regions(&mut self, recognition: bool, rec: &mut dyn Recorder) {
        // Reset per-frame neuron state for internal cells (input activity is set
        // externally). Carry nothing here; prev_active is updated in end_frame.
        for n in &mut self.neurons {
            if n.kind == NeuronKind::Internal {
                n.v = 0.0;
            }
            for a in &mut n.aff {
                a.reset();
            }
        }

        // Region ids ordered by height (stable).
        let mut order: Vec<RegionId> = (0..self.regions.len()).map(RegionId).collect();
        order.sort_by_key(|r| self.regions[r.index()].height_in_dag);

        for rid in order {
            if self.regions[rid.index()].kind != RegionKind::Internal {
                continue;
            }
            self.push_into_region(rid);
            self.process_region_macs(rid, recognition, rec);
        }
    }

    /// Two-phase signal push into region `rid` (see module docs).
    fn push_into_region(&mut self, rid: RegionId) {
        // Dense scratch indexed by target neuron (reused; avoids a per-region HashMap
        // and per-synapse hashing). Only this region's cells are written below.
        if self.push_scratch.len() < self.neurons.len() {
            self.push_scratch.resize(self.neurons.len(), Default::default());
        }

        let cfg_ix = self.regions[rid.index()].config_index;
        let aff_links = self.regions[rid.index()].afferent_links.clone();
        for lid in aff_links {
            let link = self.links[lid.index()].clone();
            let params = self.config.regions[cfg_ix].signal(link.syn_type).clone();
            let src_cells = self.regions[link.source_region.index()].cells.clone();
            let ti = link.syn_type.index();
            for sc in src_cells {
                let active = {
                    let src = &self.neurons[sc.index()];
                    if link.use_previous_active {
                        src.prev_active
                    } else {
                        src.active
                    }
                };
                if !active {
                    continue;
                }
                // MCH: an ambiguous source MAC (high num_mch) has its signal ignored or
                // discounted. Input features (no owning MAC) contribute at full strength.
                let factor = match self.mch_signal_factor(sc, link.use_previous_active, &params) {
                    Some(f) => f,
                    None => continue, // ignored (num_mch >= ignore threshold)
                };
                let ebs = self.neurons[sc.index()].efferent_bundles.clone();
                for ebid in ebs {
                    if self.efferent_bundles[ebid.index()].link != lid {
                        continue;
                    }
                    let eb = &self.efferent_bundles[ebid.index()];
                    for syn in &eb.synapses {
                        let raw = syn.effective_value(self.global_frame, &self.weight_table) as f32;
                        let w = if factor == 1.0 { raw } else { raw / factor };
                        let slot = &mut self.push_scratch[syn.target_neuron.index()][ti];
                        slot.raw_sum += w;
                        slot.active_input_count += 1;
                    }
                }
            }
        }

        // Apply this region's scratch to its cells' afferent accumulators, resetting
        // each slot so the dense buffer stays clean for the next push.
        let n_cells = self.regions[rid.index()].cells.len();
        for i in 0..n_cells {
            let tid = self.regions[rid.index()].cells[i].index();
            let scratch = std::mem::take(&mut self.push_scratch[tid]);
            let n = &mut self.neurons[tid];
            for (acc, slot) in n.aff.iter_mut().zip(scratch.iter()) {
                if slot.active_input_count > 0 {
                    acc.raw_sum += slot.raw_sum;
                    acc.active_input_count += slot.active_input_count;
                }
            }
        }
    }

    /// Activate eligible MACs of a region and select their codes.
    fn process_region_macs(&mut self, rid: RegionId, recognition: bool, rec: &mut dyn Recorder) {
        let mac_ids = self.regions[rid.index()].macs.clone();
        let active_low = self.regions[rid.index()].active_low;
        let active_high = self.regions[rid.index()].active_high;
        for mac_id in mac_ids {
            // Eligibility: the MAC's active U input-feature count must fall within the
            // region's activation band (SparseyCore ActiveInputFeatures{Low,High}Bound).
            let n_active = self.mac_u_active_features(mac_id);
            let eligible = n_active >= active_low && n_active <= active_high;
            let persisting = self.macs[mac_id.index()].code_age > 0;

            if !eligible && !persisting {
                self.macs[mac_id.index()].active = false;
                continue;
            }
            self.macs[mac_id.index()].active = true;
            if persisting {
                // Code held over from a prior frame (persistence). Keep it.
                continue;
            }

            self.normalize_mac(mac_id);

            let active_types = self.mac_active_types(mac_id);
            let g = if recognition && !self.backoffs[rid.index()].is_empty() {
                self.recognize_version(rid, mac_id, &active_types)
            } else {
                // Learning (or no backoff rules): highest-order available version.
                self.set_v(mac_id, &active_types);
                self.compute_g(mac_id)
            };

            self.select_winners(mac_id, g);
            self.macs[mac_id.index()].g = g;
            self.macs[mac_id.index()].code_age = self.regions[rid.index()].persistence;

            if let Some(code) = self.mac_code(mac_id) {
                self.macs[mac_id.index()]
                    .learned_codes
                    .insert(code.clone(), self.global_frame);
                rec.on_code_selected(rid, mac_id, &code, g, self.global_frame);
            }
            rec.on_mac_active(rid, mac_id, self.global_frame);
        }
    }

    /// Which synapse types delivered active input to a MAC this frame.
    /// MCH signal factor for a source cell: `None` = ignore the source's contribution
    /// (its MAC is too ambiguous, `num_mch ≥ ignore threshold`); `Some(f)` = divide the
    /// contribution by `f`. Input features (no owning MAC) return `Some(1.0)`. Uses the
    /// source MAC's `num_mch`, or `prev_num_mch` for previous-frame (H/D) signals.
    fn mch_signal_factor(
        &self,
        src_cell: NeuronId,
        use_prev: bool,
        params: &SignalParams,
    ) -> Option<f32> {
        let cm = match self.neurons[src_cell.index()].owning_cm {
            Some(cm) => cm,
            None => return Some(1.0), // input feature — no MCH, full strength
        };
        let mac = self.cms[cm.index()].owning_mac;
        let n_mch = if use_prev {
            self.macs[mac.index()].prev_num_mch
        } else {
            self.macs[mac.index()].num_mch
        } as f32;
        if n_mch >= params.mch_ignore_thresh {
            return None;
        }
        let factor = if n_mch >= params.mch_discount_thresh {
            1.0 + n_mch.powf(params.mch_discount_exp)
        } else {
            n_mch.max(1.0)
        };
        Some(factor)
    }

    /// The MAC's active U input-feature count = the max over its cells of the U
    /// afferent `active_input_count` (equal across cells under full connectivity).
    /// Drives the activation band.
    fn mac_u_active_features(&self, mac_id: MacId) -> u32 {
        let mac = &self.macs[mac_id.index()];
        let mut max_n = 0u32;
        for &cid in &mac.cms {
            for &nid in &self.cms[cid.index()].neurons {
                let n = self.neurons[nid.index()].aff(SynapseType::U).active_input_count;
                if n > max_n {
                    max_n = n;
                }
            }
        }
        max_n
    }

    fn mac_active_types(&self, mac_id: MacId) -> Vec<SynapseType> {
        let mut types = Vec::new();
        let mac = &self.macs[mac_id.index()];
        'ty: for &ty in &SynapseType::ALL {
            for &cid in &mac.cms {
                for &nid in &self.cms[cid.index()].neurons {
                    if self.neurons[nid.index()].aff(ty).active_input_count > 0 {
                        types.push(ty);
                        continue 'ty;
                    }
                }
            }
        }
        types
    }

    /// Compute normalized + adjusted afferent sums for every cell of a MAC, for each
    /// active type. Normalizer = `active_input_count * MAX_WEIGHT` (per cell; equal
    /// across the MAC under full connectivity).
    fn normalize_mac(&mut self, mac_id: MacId) {
        let region_cfg_ix = {
            let rid = self.macs[mac_id.index()].owning_region;
            self.regions[rid.index()].config_index
        };
        let n_cms = self.macs[mac_id.index()].cms.len();
        for ci in 0..n_cms {
            let cid = self.macs[mac_id.index()].cms[ci];
            let n_cells = self.cms[cid.index()].neurons.len();
            for ni in 0..n_cells {
                let nid = self.cms[cid.index()].neurons[ni];
                for &ty in &SynapseType::ALL {
                    let (raw, count) = {
                        let acc = self.neurons[nid.index()].aff(ty);
                        (acc.raw_sum, acc.active_input_count)
                    };
                    if count == 0 {
                        continue;
                    }
                    let params = self.config.regions[region_cfg_ix].signal(ty);
                    let normalizer = count as f32 * MAX_WEIGHT as f32;
                    let mut normalized = if normalizer > 0.0 { raw / normalizer } else { 0.0 };
                    if normalized < params.min_cutoff {
                        normalized = params.min_cutoff;
                    } else if normalized > params.max_cutoff {
                        // Matches Java: values above the max cutoff saturate to 1.0.
                        normalized = 1.0;
                    }
                    let exp = params.exp.max(1);
                    let adjusted = if exp > 1 {
                        normalized.powi(exp)
                    } else {
                        normalized
                    };
                    let acc = self.neurons[nid.index()].aff_mut(ty);
                    acc.normalized = normalized;
                    acc.adjusted = adjusted;
                }
            }
        }
    }

    /// Set each cell's V to the product of its adjusted sums over `types` (only those
    /// active for the cell). Cells with no contributing type get V = 1.0 (neutral).
    fn set_v(&mut self, mac_id: MacId, types: &[SynapseType]) {
        let n_cms = self.macs[mac_id.index()].cms.len();
        for ci in 0..n_cms {
            let cid = self.macs[mac_id.index()].cms[ci];
            let n_cells = self.cms[cid.index()].neurons.len();
            for ni in 0..n_cells {
                let nid = self.cms[cid.index()].neurons[ni];
                let mut v = 1.0f32;
                let n = &self.neurons[nid.index()];
                for &ty in types {
                    if n.aff(ty).active_input_count > 0 {
                        v *= n.aff(ty).adjusted;
                    }
                }
                self.neurons[nid.index()].v = v;
            }
        }
    }

    /// Compute G = mean over CMs of the CM's max V, filling per-CM `v_max`, `v_ave`,
    /// `tied_max_count`, and `num_mch` (cells with `V ≥ v_thresh_hypothesis`). Also sets
    /// the MAC's `num_mch` = round(mean per-CM MCH count), min 1 (SparseyCore
    /// `Mac.computeNumMCHs`).
    fn compute_g(&mut self, mac_id: MacId) -> f32 {
        let cfg_ix = {
            let rid = self.macs[mac_id.index()].owning_region;
            self.regions[rid.index()].config_index
        };
        let v_thresh = self.config.regions[cfg_ix].v_thresh_hypothesis;
        let n_cms = self.macs[mac_id.index()].cms.len();
        let mut sum_max = 0.0f32;
        let mut sum_mch = 0u32;
        for ci in 0..n_cms {
            let cid = self.macs[mac_id.index()].cms[ci];
            let n_cells = self.cms[cid.index()].neurons.len();
            let mut v_max = 0.0f32;
            let mut v_sum = 0.0f32;
            let mut tied = 0u32;
            let mut n_mch = 0u32;
            for ni in 0..n_cells {
                let nid = self.cms[cid.index()].neurons[ni];
                let v = self.neurons[nid.index()].v;
                v_sum += v;
                if v > v_max {
                    v_max = v;
                    tied = 1;
                } else if v == v_max {
                    tied += 1;
                }
                if v >= v_thresh {
                    n_mch += 1;
                }
            }
            let cm = &mut self.cms[cid.index()];
            cm.v_max = v_max;
            cm.v_ave = v_sum / n_cells.max(1) as f32;
            cm.tied_max_count = tied;
            cm.num_mch = n_mch;
            sum_max += v_max;
            sum_mch += n_mch;
        }
        let q = n_cms.max(1);
        self.macs[mac_id.index()].num_mch =
            ((sum_mch as f32 / q as f32).round() as u32).max(1);
        sum_max / q as f32
    }

    /// Recognition: run the region's backoff strategy to choose the G version, then
    /// leave the winning version's V values in place. Returns the winning G.
    fn recognize_version(
        &mut self,
        rid: RegionId,
        mac_id: MacId,
        active_types: &[SynapseType],
    ) -> f32 {
        let strat = self.backoffs[rid.index()].clone();
        let is_available = |t: SynapseType| active_types.contains(&t);
        let result = strat.evaluate(is_available, |types| {
            self.set_v(mac_id, types);
            self.compute_g(mac_id)
        });
        match result {
            Some(r) => {
                // Re-establish the winning version's V (evaluate may have left a
                // different, later version's V in place).
                self.set_v(mac_id, &r.syn_types);
                self.compute_g(mac_id)
            }
            None => {
                // No available version at all — fall back to highest-order available.
                self.set_v(mac_id, active_types);
                self.compute_g(mac_id)
            }
        }
    }

    /// Winner selection: one winner per CM.
    ///
    /// - Deterministic **max-V** (the M1 default): the max-V cell, ties broken uniformly
    ///   with the seeded RNG.
    /// - Probabilistic **CSA** (when the region's sigmoid is `enabled` *and* this is a
    ///   learning frame): sample the winner from the `V → mu → rho` distribution, its
    ///   spread set by expansivity `eta(G)` and the MAC's ratcheting inflection point.
    ///   Recognition always uses max-V (mirrors SparseyCore `Use_ML_Recog`).
    ///
    /// See `doc/AlgorithmTriangulation.md` for the expansivity/sigmoid formulas.
    fn select_winners(&mut self, mac_id: MacId, g: f32) {
        let rid = self.macs[mac_id.index()].owning_region;
        let cfg_ix = self.regions[rid.index()].config_index;
        let sigmoid = self.config.regions[cfg_ix].sigmoid.clone();
        let probabilistic = sigmoid.enabled && self.op_mode == OperationMode::Learning;

        let cm_ids = self.macs[mac_id.index()].cms.clone();

        // Dynamic inflection: ratchet this MAC's sigmoid inflection point rightward as
        // it saturates — once the mean of its CMs' `V_ave` passes the threshold, and
        // only when a new max is reached (SparseyCore `Mac.determine_Inflection_Point`).
        // Higher inflection ⇒ the CSA demands higher V before it favors a cell.
        if probabilistic {
            let mean_v_ave = cm_ids
                .iter()
                .map(|c| self.cms[c.index()].v_ave)
                .sum::<f32>()
                / cm_ids.len().max(1) as f32;
            let mac = &mut self.macs[mac_id.index()];
            if mean_v_ave > sigmoid.mean_v_ave_threshold
                && mean_v_ave > mac.max_mean_v_ave
                && mac.inflect_point < sigmoid.max_inflect
            {
                mac.max_mean_v_ave = mean_v_ave;
                let delta = (sigmoid.max_inflect - sigmoid.min_inflect) / 100.0;
                mac.inflect_point = (mac.inflect_point + delta).min(sigmoid.max_inflect);
            }
        }
        let inflect = self.macs[mac_id.index()].inflect_point;

        for cid in cm_ids {
            let cell_ids = self.cms[cid.index()].neurons.clone();
            let winner = if probabilistic {
                self.sample_winner(&cell_ids, g, inflect, &sigmoid)
            } else {
                self.max_v_winner(&cell_ids)
            };
            self.neurons[winner.index()].active = true;
            self.cms[cid.index()].winner = Some(winner);
        }
    }

    /// Deterministic max-V winner for one CM (ties broken uniformly with the seeded RNG).
    fn max_v_winner(&mut self, cell_ids: &[NeuronId]) -> NeuronId {
        let mut v_max = f32::NEG_INFINITY;
        for &nid in cell_ids {
            let v = self.neurons[nid.index()].v;
            if v > v_max {
                v_max = v;
            }
        }
        let tied: Vec<NeuronId> = cell_ids
            .iter()
            .copied()
            .filter(|nid| self.neurons[nid.index()].v == v_max)
            .collect();
        if tied.len() == 1 {
            tied[0]
        } else {
            let pick = self.rng.random_range(0..tied.len());
            tied[pick]
        }
    }

    /// Probabilistic CSA winner for one CM: sample from the cumulative `mu` distribution
    /// over the CM's cells. `eta` (expansivity) sets the distribution's range and grows
    /// with global familiarity `G`; at `eta = 1` (novel input) every `mu` floors to
    /// `lower_limit`, so the pick is uniform — breaking the max-V zero-init symmetry that
    /// would otherwise collapse novel codes onto cell 0. `inflect` is the MAC's current
    /// sigmoid inflection point.
    fn sample_winner(
        &mut self,
        cell_ids: &[NeuronId],
        g: f32,
        inflect: f32,
        s: &SigmoidConfig,
    ) -> NeuronId {
        let eta = expansivity(g, cell_ids.len(), s);
        let mus: Vec<f32> = cell_ids
            .iter()
            .map(|&nid| cell_mu(self.neurons[nid.index()].v, eta, inflect, s))
            .collect();
        let total: f32 = mus.iter().sum();
        // total >= K * lower_limit > 0, so this is safe.
        let draw = self.rng.random::<f32>() * total;
        let mut acc = 0.0f32;
        for (i, &m) in mus.iter().enumerate() {
            acc += m;
            if draw < acc {
                return cell_ids[i];
            }
        }
        cell_ids[cell_ids.len() - 1] // fp guard: fall through to the last cell
    }

    /// Hebbian weight update: for every efferent synapse whose presynaptic cell is
    /// active (previous-frame for H/D) and whose target is an active winner, record a
    /// pre-post coincidence.
    fn update_weights(&mut self) {
        let SparseyNet {
            efferent_bundles,
            neurons,
            links,
            regions,
            weight_table,
            global_frame,
            ..
        } = self;
        for eb in efferent_bundles.iter_mut() {
            if eb.frozen {
                continue;
            }
            let link = &links[eb.link.index()];
            let src = &neurons[eb.source_neuron.index()];
            let pre_active = if link.use_previous_active {
                src.prev_active
            } else {
                src.active
            };
            if !pre_active {
                continue;
            }
            // Promotion is gated on the source region's persistence — a pre-post may
            // reinforce only once the code has persisted (Java `region.getPersistence()`).
            let persistence = regions[src.owning_region.index()].persistence;
            for syn in eb.synapses.iter_mut() {
                if neurons[syn.target_neuron.index()].active {
                    syn.record_pre_post(*global_frame, weight_table, persistence);
                } else {
                    // Reconcile a decayed transient synapse back to virgin so it is not
                    // later frozen/finalized in as permanent (was previously dead code).
                    syn.reconcile_if_inert(*global_frame, weight_table);
                }
            }
        }
    }

    /// Freeze any bundle whose increased-synapse fraction exceeds its link's target
    /// saturation threshold.
    fn freeze_saturated_bundles(&mut self) {
        let frame = self.global_frame;
        for i in 0..self.efferent_bundles.len() {
            if self.efferent_bundles[i].frozen {
                continue;
            }
            // A bundle can only *newly* saturate when its source fired this frame (that
            // is the only way synapses are increased); decay only lowers the increased
            // fraction and never triggers freezing. So skip inactive-source bundles —
            // this avoids the per-synapse `increased_fraction` scan for the vast
            // majority of bundles each frame.
            let (target_cfg_ix, ty) = {
                let eb = &self.efferent_bundles[i];
                let link = &self.links[eb.link.index()];
                let src = &self.neurons[eb.source_neuron.index()];
                let src_active = if link.use_previous_active {
                    src.prev_active
                } else {
                    src.active
                };
                if !src_active {
                    continue;
                }
                (
                    self.regions[link.target_region.index()].config_index,
                    eb.syn_type,
                )
            };
            let threshold = self.config.regions[target_cfg_ix]
                .signal(ty)
                .saturation_threshold;
            let frac = self.efferent_bundles[i].increased_fraction(frame, &self.weight_table);
            if frac > threshold {
                let wt = self.weight_table.clone();
                self.efferent_bundles[i].freeze(frame, &wt);
            }
        }
    }

    /// End-of-frame housekeeping: carry activity to `prev_active`, age codes, and
    /// deactivate MACs whose persistence has elapsed. Advances the global frame.
    fn end_frame(&mut self, rec: &mut dyn Recorder) {
        // Age codes; deactivate expired MACs' winners.
        for mac_id in 0..self.macs.len() {
            let age = self.macs[mac_id].code_age;
            if age > 0 {
                let new_age = age - 1;
                self.macs[mac_id].code_age = new_age;
                if new_age == 0 {
                    self.macs[mac_id].active = false;
                }
            }
        }

        // prev_active <- active for all neurons; clear internal current activity for
        // the next frame's recomputation (input activity is set externally).
        for n in &mut self.neurons {
            n.prev_active = n.active;
            if n.kind == NeuronKind::Internal {
                n.active = false;
            }
        }
        // Roll CM winners into prev_winner.
        for cm in &mut self.cms {
            cm.prev_winner = cm.winner;
            cm.winner = None;
        }

        // Roll each MAC's MCH count for next-frame (prev-active H/D) signal discount.
        for mac in &mut self.macs {
            mac.prev_num_mch = mac.num_mch;
        }

        rec.on_frame_end(self.global_frame);
        self.global_frame += 1;
    }
}


/// Expansivity `eta` from global familiarity `G`. SparseyCore `determine_mu_Range` ≡
/// Sparsey_Alt `calculateExpansivity` (see `doc/AlgorithmTriangulation.md`):
/// `eta = 1 + max(0, (G - g_floor)/(1 - g_floor))^expansion_exp · expansion_factor · K`.
/// At `G <= g_floor`, `eta = 1` (uniform selection).
fn expansivity(g: f32, k: usize, s: &SigmoidConfig) -> f32 {
    let denom = (1.0 - s.g_floor).max(f32::EPSILON);
    let rect = ((g - s.g_floor) / denom).max(0.0);
    1.0 + rect.powi(s.expansion_exp) * s.expansion_factor * k as f32
}

/// `V → mu` (unnormalized relative selection weight of one cell), the uncommented
/// SparseyCore `recompute_mu_And_rho` form:
/// `mu = max(mu_range / (1 + exp(-nonlin·(V - inflect))), lower_limit)`, with cutoffs —
/// `V < lower_v_cutoff` ⇒ `lower_limit`; `V >= upper_v_cutoff` ⇒ `mu_range`. `mu_range`
/// is the expansivity `eta`; `inflect` the MAC's current inflection point. A cell near
/// `V=1` → `mu ≈ eta`; a low-V cell → `lower_limit`.
fn cell_mu(v: f32, mu_range: f32, inflect: f32, s: &SigmoidConfig) -> f32 {
    if v < s.lower_v_cutoff {
        return s.lower_limit;
    }
    if v >= s.upper_v_cutoff {
        return mu_range.max(s.lower_limit);
    }
    let denom = 1.0 + (-s.nonlin * (v - inflect)).exp();
    (mu_range / denom).max(s.lower_limit)
}

#[cfg(test)]
mod csa_math_tests {
    use super::{cell_mu, expansivity};
    use crate::config::SigmoidConfig;

    /// Expansivity matches Sparsey_Alt `calculateExpansivity` exactly (defaults:
    /// g_floor=0.1, exp=2, factor=100). At G=1, K=8: 1 + 1^2 · 100 · 8 = 801.
    #[test]
    fn expansivity_matches_reference() {
        let s = SigmoidConfig::default();
        assert!((expansivity(1.0, 8, &s) - 801.0).abs() < 1e-3, "G=1,K=8 → 801");
        // At/below the G floor, eta collapses to 1 (uniform selection).
        assert!((expansivity(0.1, 8, &s) - 1.0).abs() < 1e-6);
        assert!((expansivity(0.0, 8, &s) - 1.0).abs() < 1e-6);
        // Monotonic in G and in K.
        assert!(expansivity(0.5, 8, &s) > expansivity(0.3, 8, &s));
        assert!(expansivity(1.0, 16, &s) > expansivity(1.0, 8, &s));
    }

    /// V→mu follows SparseyCore's `mu = max(eta/(1+exp(-nonlin·(V-inflect))), lower_limit)`.
    #[test]
    fn cell_mu_matches_upstream_form() {
        let s = SigmoidConfig::default();
        let eta = 801.0;
        let inflect = s.min_inflect; // 0.5

        // V above the upper cutoff → mu = eta (the max).
        assert!((cell_mu(1.0, eta, inflect, &s) - eta).abs() < 1e-3, "V≥cutoff → mu=eta");
        // Monotonic increasing in V, bounded in [lower_limit, eta].
        assert!(cell_mu(0.8, eta, inflect, &s) > cell_mu(0.4, eta, inflect, &s));
        assert!(cell_mu(0.2, eta, inflect, &s) >= s.lower_limit);
        assert!(cell_mu(0.7, eta, inflect, &s) <= eta);
        // eta = 1 (novel MAC): every cell floors to lower_limit → uniform distribution.
        assert!((cell_mu(0.7, 1.0, inflect, &s) - s.lower_limit).abs() < 1e-6);
        // Ratcheting the inflection point right lowers a mid-V cell's mu (more
        // selective — it now takes higher V to earn a high mu).
        assert!(cell_mu(0.5, eta, 0.5, &s) > cell_mu(0.5, eta, 0.9, &s));
    }
}
