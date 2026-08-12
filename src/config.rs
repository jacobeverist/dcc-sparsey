//! Network configuration — the serde replacement for SparseyCore's `DescriptorFile`
//! (NDF) parser.
//!
//! A [`NetworkConfig`] fully describes a network's structure and parameters and
//! round-trips as JSON. Parameter names map to the Java NDF params (see
//! `doc/PortNotes.md`): `Region_Q`→[`RegionConfig::q`], `Region_K`→`k`,
//! `RegionWidthInBlks`→`width_in_blks`, `RegionHeights`→`height_in_dag`,
//! `RegionPersistence`→`persistence`, `U_NEI_Exp`/`U_EI_Exp`→[`SignalParams::exp`]/
//! `exp_post_quiescent`, `*_min/max_cutoff`, `*_saturation_threshold`, `*_MCH_*`,
//! `*_Connectivity_Band_*`.
//!
//! Most fields have `#[serde(default)]` so JSON configs stay terse and the
//! [`NetworkConfigBuilder`] / [`RegionConfigBuilder`] can fill in the rest.

use serde::{Deserialize, Serialize};

use crate::types::SynapseType;

/// Per-synapse-type signal-processing + connectivity parameters for a region's
/// afferent connections of that type.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct SignalParams {
    /// Signal exponent on non-post-quiescent frames (`{U,H,D}_NEI_Exp` / `*_Exp`).
    pub exp: i32,
    /// Signal exponent on post-quiescent (episode-initial) frames (`U_EI_Exp`).
    /// For H/D this equals [`SignalParams::exp`].
    pub exp_post_quiescent: i32,
    /// Lower influence cutoff (`*_min_cutoff`).
    pub min_cutoff: f32,
    /// Upper influence cutoff (`*_max_cutoff`).
    pub max_cutoff: f32,
    /// Freezing saturation threshold for bundles of this type (`*_saturation_threshold`).
    pub saturation_threshold: f32,
    /// MCH ignore threshold (`*_MCH_Ignore_Thresh`).
    pub mch_ignore_thresh: f32,
    /// MCH discount exponent (`*_MCH_Discount_Exp`).
    pub mch_discount_exp: f32,
    /// MCH discount threshold (`*_MCH_Discount_Thresh`).
    pub mch_discount_thresh: f32,
    /// Projective-field band distance thresholds (`*_Connectivity_Band_Thickness`).
    pub band_thickness: Vec<f32>,
    /// Per-band connection probabilities (`*_Connectivity_Band_Rates`).
    pub band_rates: Vec<f32>,
}

impl Default for SignalParams {
    fn default() -> Self {
        SignalParams {
            exp: 1,
            exp_post_quiescent: 1,
            min_cutoff: 0.0,
            max_cutoff: 1.0,
            saturation_threshold: 0.5,
            mch_ignore_thresh: 1.0e6,
            mch_discount_exp: 1.0,
            mch_discount_thresh: 1.0e6,
            band_thickness: Vec::new(),
            band_rates: Vec::new(),
        }
    }
}

/// A single backoff match case: the set of synapse types combined to compute G, and
/// the G threshold this case must reach to win.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct BackoffCase {
    /// Synapse types combined for this case's G computation (e.g. `[U,H,D]`).
    pub syn_types: Vec<SynapseType>,
    /// The G threshold this case must meet.
    pub threshold: f32,
}

/// Backoff strategy for a region: priority levels, each holding one or more cases
/// that compete on equal footing. Mirrors `BackoffStrategy.matchingRules` +
/// `thresholds`, but stores the threshold alongside each case.
///
/// Priority order is the outer `Vec` order (index 0 = highest priority, usually the
/// highest-complexity rule).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct BackoffConfig {
    /// Priority levels, outer→inner = priority→cases.
    pub priorities: Vec<Vec<BackoffCase>>,
}

impl BackoffConfig {
    /// The canonical HUD → HU/UD → U chain used by most NDFs, with the given
    /// thresholds for the 3-way, 2-way and 1-way priority levels respectively.
    pub fn canonical(t_3way: f32, t_2way: f32, t_1way: f32) -> Self {
        use SynapseType::{D, H, U};
        BackoffConfig {
            priorities: vec![
                vec![BackoffCase {
                    syn_types: vec![H, U, D],
                    threshold: t_3way,
                }],
                vec![
                    BackoffCase {
                        syn_types: vec![H, U],
                        threshold: t_2way,
                    },
                    BackoffCase {
                        syn_types: vec![U, D],
                        threshold: t_2way,
                    },
                ],
                vec![BackoffCase {
                    syn_types: vec![U],
                    threshold: t_1way,
                }],
            ],
        }
    }

    /// The fuller backoff chain with every 2-way and 1-way combination: HUD →
    /// {HU, UD, HD} → {H, U, D}. Within each priority level the recognizer keeps the
    /// max-G case that clears its threshold (see `BackoffStrategy::evaluate`).
    pub fn canonical_full(t_3way: f32, t_2way: f32, t_1way: f32) -> Self {
        use SynapseType::{D, H, U};
        let case = |syn_types: Vec<SynapseType>, threshold: f32| BackoffCase { syn_types, threshold };
        BackoffConfig {
            priorities: vec![
                vec![case(vec![H, U, D], t_3way)],
                vec![
                    case(vec![H, U], t_2way),
                    case(vec![U, D], t_2way),
                    case(vec![H, D], t_2way),
                ],
                vec![case(vec![H], t_1way), case(vec![U], t_1way), case(vec![D], t_1way)],
            ],
        }
    }
}

/// Parameters for the probabilistic Code Selection Algorithm (CSA): the sigmoid that
/// maps a CM's per-cell V into selection probabilities via expansivity `eta` (=
/// `mu_Range`) and a **dynamic inflection point**.
///
/// [`enabled`](SigmoidConfig::enabled) defaults to `true` — *learning* frames sample
/// winners from the `V -> mu -> rho` distribution (the full Sparsey CSA). Set it to
/// `false` (`RegionConfigBuilder::disable_csa`) for deterministic max-V (the M1 subset).
/// Recognition always uses max-V regardless (mirrors SparseyCore `Use_ML_Recog`).
///
/// This follows the **upstream SparseyCore** formulas (`CM.determine_mu_Range`,
/// `CM.recompute_mu_And_rho`, `Mac.determine_Inflection_Point`):
/// - `eta = 1 + max(0,(G - g_floor)/(1 - g_floor))^expansion_exp * expansion_factor * K`
///   (this expansivity form is confirmed identical in the independent Sparsey_Alt);
/// - `mu(V) = max(eta / (1 + exp(-nonlin*(V - inflect))), lower_limit)`, with the
///   `inflect` **ratcheting** from `min_inflect` toward `max_inflect` (by
///   `(max_inflect-min_inflect)/100` per learning frame) once a MAC's mean-`V_ave`
///   exceeds `mean_v_ave_threshold` — an adaptive selectivity that tracks saturation.
///
/// (Sparsey_Alt uses the same logistic *shape* with a fixed inflection and a different
/// coefficient parameterization; the port matches upstream. See
/// `doc/AlgorithmTriangulation.md`.)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct SigmoidConfig {
    /// Enable the probabilistic CSA for learning (default). `false` ⇒ deterministic
    /// max-V (the M1 subset).
    #[serde(default = "default_csa_enabled")]
    pub enabled: bool,

    // --- expansivity: eta = 1 + max(0,(G - g_floor)/(1 - g_floor))^exp * factor * K ---
    /// G floor below which eta = 1 (uniform-random selection). SparseyCore
    /// `lower_G_Cutoff` / Sparsey_Alt `globalFamiliarityModifier`.
    #[serde(default = "default_g_floor")]
    pub g_floor: f32,
    /// Expansivity exponent (`MU_Range_Expansion_Exp` / `expansionExponent`).
    #[serde(default = "default_expansion_exp")]
    pub expansion_exp: i32,
    /// Expansivity multiplier (`MU_Range_Expansion_Factor` / `expansionFactor`).
    #[serde(default = "default_expansion_factor")]
    pub expansion_factor: f32,

    // --- V->mu sigmoid: mu = max(eta / (1 + exp(-nonlin*(V - inflect))), lower_limit) ---
    /// Sigmoid nonlinearity strength (`V_to_mu_nonlinStrength`).
    #[serde(default = "default_sig_nonlin")]
    pub nonlin: f32,
    /// Minimum (initial) sigmoid inflection point (`Sigmoid_Min_Inflection_Point`).
    #[serde(default = "default_min_inflect")]
    pub min_inflect: f32,
    /// Maximum sigmoid inflection point the ratchet can reach (`Sigmoid_Max_Inflection_Point`).
    #[serde(default = "default_max_inflect")]
    pub max_inflect: f32,
    /// Lower clamp on `mu` (`sigmoidLowerLimit`) — the floor selection weight.
    #[serde(default = "default_lower_limit")]
    pub lower_limit: f32,

    /// Lower V cutoff: cells with `V < lower_v_cutoff` get `mu = lower_limit`.
    #[serde(default)]
    pub lower_v_cutoff: f32,
    /// Upper V cutoff: cells with `V >= upper_v_cutoff` get `mu = eta` (`Sigmoid_Upper_V_Cutoff`).
    #[serde(default = "default_upper_v_cutoff")]
    pub upper_v_cutoff: f32,
    /// Mean-`V_ave` threshold above which the inflection point starts ratcheting right.
    #[serde(default = "default_mean_v_ave_threshold")]
    pub mean_v_ave_threshold: f32,
}

fn default_csa_enabled() -> bool {
    true
}
fn default_g_floor() -> f32 {
    0.1
}
fn default_expansion_exp() -> i32 {
    2
}
fn default_expansion_factor() -> f32 {
    100.0
}
fn default_sig_nonlin() -> f32 {
    4.0
}
fn default_min_inflect() -> f32 {
    0.5
}
fn default_max_inflect() -> f32 {
    0.9
}
fn default_lower_limit() -> f32 {
    1.0
}
fn default_upper_v_cutoff() -> f32 {
    0.999
}
fn default_mean_v_ave_threshold() -> f32 {
    0.3
}

impl Default for SigmoidConfig {
    fn default() -> Self {
        SigmoidConfig {
            enabled: true,
            g_floor: default_g_floor(),
            expansion_exp: default_expansion_exp(),
            expansion_factor: default_expansion_factor(),
            nonlin: default_sig_nonlin(),
            min_inflect: default_min_inflect(),
            max_inflect: default_max_inflect(),
            lower_limit: default_lower_limit(),
            lower_v_cutoff: 0.0,
            upper_v_cutoff: default_upper_v_cutoff(),
            mean_v_ave_threshold: default_mean_v_ave_threshold(),
        }
    }
}

/// Configuration for one region (DAG node).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct RegionConfig {
    /// Region name; connections reference regions by name.
    pub name: String,
    /// Height in the DAG. `0` ⇒ input region; `> 0` ⇒ internal region.
    pub height_in_dag: u32,
    /// Region width in blocks (MACs for internal regions; apertures/feature columns
    /// for input regions). `RegionWidthInBlks`.
    pub width_in_blks: u32,
    /// Region height in blocks. `RegionHeightInBlks`.
    pub height_in_blks: u32,
    /// CMs per MAC (`Region_Q`). Ignored for input regions.
    #[serde(default = "default_q")]
    pub q: u32,
    /// Cells per CM (`Region_K`). Ignored for input regions.
    #[serde(default = "default_k")]
    pub k: u32,
    /// Frames a code persists once selected (`RegionPersistence`).
    #[serde(default = "default_persistence")]
    pub persistence: u32,
    /// Per-type signal params (`H`, `U`, `D`).
    #[serde(default)]
    pub signal_h: SignalParams,
    /// Upward signal params.
    #[serde(default)]
    pub signal_u: SignalParams,
    /// Downward signal params.
    #[serde(default)]
    pub signal_d: SignalParams,
    /// Backoff strategy (recognition).
    #[serde(default)]
    pub backoff: BackoffConfig,
    /// Sigmoid / probabilistic-selection params.
    #[serde(default)]
    pub sigmoid: SigmoidConfig,

    /// Activation-band lower bound as a fraction of the U afferent input size
    /// (`ActiveInputFeaturesLowBoundAsFrac`). A MAC is eligible only if its active U
    /// input-feature count is within `[low, high] × input_size`.
    #[serde(default)]
    pub active_low_frac: f32,
    /// Activation-band upper bound as a fraction of the U afferent input size
    /// (`ActiveInputFeaturesHighBoundAsFrac`). `>= 1.0` ⇒ no upper limit. Defaults
    /// `[0.0, 1.0]` reproduce the prior "any U input" behavior (count ≥ 1, no cap).
    #[serde(default = "default_active_high_frac")]
    pub active_high_frac: f32,
    /// V threshold above which a cell counts as an MCH hypothesis
    /// (`V_ThreshToBeHypothesis`). Drives per-CM/MAC `num_mch`.
    #[serde(default = "default_v_thresh_hypothesis")]
    pub v_thresh_hypothesis: f32,
}

fn default_active_high_frac() -> f32 {
    1.0
}
fn default_v_thresh_hypothesis() -> f32 {
    0.5
}

fn default_q() -> u32 {
    1
}
fn default_k() -> u32 {
    1
}
fn default_persistence() -> u32 {
    1
}

impl RegionConfig {
    /// Per-type signal params accessor.
    pub fn signal(&self, ty: SynapseType) -> &SignalParams {
        match ty {
            SynapseType::H => &self.signal_h,
            SynapseType::U => &self.signal_u,
            SynapseType::D => &self.signal_d,
        }
    }

    /// Is this an input (leaf) region?
    pub fn is_input(&self) -> bool {
        self.height_in_dag == 0
    }
}

/// A connection between two regions, referenced by name. The synapse type is
/// derived from the endpoints' relative `height_in_dag` at build time.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct ConnectionSpec {
    /// Source region name.
    pub source: String,
    /// Target region name.
    pub target: String,
}

/// Weight-table parameters. Mirrors the static tables in `Synapse.java`:
/// `WT_TABLE_TRANSITION_INDEXES` (ages) and `WT_TABLE_WEIGHTS` (values at those
/// ages), one row per stiffness level. `WT_TABLE[stiffness][age]` is built by
/// interpolation (see [`crate::synapse::WeightTable`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct WeightTableConfig {
    /// Maximum representable synapse age (`MAX_SYNAPSE_AGE`).
    pub max_synapse_age: u32,
    /// Maximum stiffness = permanent (`MAX_SYNAPSE_STIFFNESS`).
    pub max_stiffness: u8,
    /// Age breakpoints per stiffness row (`WT_TABLE_TRANSITION_INDEXES`).
    pub transition_indexes: Vec<Vec<u32>>,
    /// Weight values at those breakpoints per stiffness row (`WT_TABLE_WEIGHTS`).
    pub weights: Vec<Vec<u8>>,
}

impl Default for WeightTableConfig {
    /// The default table from `Synapse.java`.
    fn default() -> Self {
        WeightTableConfig {
            max_synapse_age: 30000,
            max_stiffness: 2,
            transition_indexes: vec![vec![2000, 3000, 4000], vec![4000, 6000, 8000]],
            weights: vec![vec![127, 120, 50], vec![127, 120, 50]],
        }
    }
}

/// A complete network configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct NetworkConfig {
    /// Regions (DAG nodes).
    pub regions: Vec<RegionConfig>,
    /// Inter-region connections.
    pub connections: Vec<ConnectionSpec>,
    /// Weight-table configuration.
    #[serde(default)]
    pub weight_table: WeightTableConfig,
}

impl NetworkConfig {
    /// Parse a config from a JSON string.
    pub fn from_json(s: &str) -> crate::SparseyResult<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Serialize this config to pretty JSON.
    pub fn to_json(&self) -> crate::SparseyResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Look up a region config by name.
    pub fn region(&self, name: &str) -> Option<&RegionConfig> {
        self.regions.iter().find(|r| r.name == name)
    }

    /// Start building a config.
    pub fn builder() -> NetworkConfigBuilder {
        NetworkConfigBuilder::default()
    }
}

/// Builder for [`NetworkConfig`].
#[derive(Default)]
pub struct NetworkConfigBuilder {
    regions: Vec<RegionConfig>,
    connections: Vec<ConnectionSpec>,
    weight_table: WeightTableConfig,
}

impl NetworkConfigBuilder {
    /// Add a region.
    pub fn region(mut self, region: RegionConfig) -> Self {
        self.regions.push(region);
        self
    }

    /// Add a connection `source → target` (both by region name).
    pub fn connect(mut self, source: &str, target: &str) -> Self {
        self.connections.push(ConnectionSpec {
            source: source.to_string(),
            target: target.to_string(),
        });
        self
    }

    /// Override the weight table.
    pub fn weight_table(mut self, wt: WeightTableConfig) -> Self {
        self.weight_table = wt;
        self
    }

    /// Finish building.
    pub fn build(self) -> NetworkConfig {
        NetworkConfig {
            regions: self.regions,
            connections: self.connections,
            weight_table: self.weight_table,
        }
    }
}

/// Builder for [`RegionConfig`] with ergonomic defaults.
pub struct RegionConfigBuilder {
    cfg: RegionConfig,
}

impl RegionConfigBuilder {
    /// Start a region builder with a name and DAG height.
    pub fn new(name: &str, height_in_dag: u32) -> Self {
        RegionConfigBuilder {
            cfg: RegionConfig {
                name: name.to_string(),
                height_in_dag,
                width_in_blks: 1,
                height_in_blks: 1,
                q: default_q(),
                k: default_k(),
                persistence: default_persistence(),
                signal_h: SignalParams::default(),
                signal_u: SignalParams::default(),
                signal_d: SignalParams::default(),
                backoff: BackoffConfig::default(),
                sigmoid: SigmoidConfig::default(),
                active_low_frac: 0.0,
                active_high_frac: default_active_high_frac(),
                v_thresh_hypothesis: default_v_thresh_hypothesis(),
            },
        }
    }

    /// Set the activation band (fractions of the U afferent input size). A MAC is
    /// eligible only if its active U input-feature count is within
    /// `[low_frac, high_frac] × input_size`. `high_frac >= 1.0` ⇒ no upper limit.
    pub fn activation_band(mut self, low_frac: f32, high_frac: f32) -> Self {
        self.cfg.active_low_frac = low_frac;
        self.cfg.active_high_frac = high_frac;
        self
    }

    /// Set the block grid dimensions (MACs for internal, feature grid for input).
    pub fn grid(mut self, width_in_blks: u32, height_in_blks: u32) -> Self {
        self.cfg.width_in_blks = width_in_blks;
        self.cfg.height_in_blks = height_in_blks;
        self
    }

    /// Set `Q` (CMs per MAC) and `K` (cells per CM).
    pub fn qk(mut self, q: u32, k: u32) -> Self {
        self.cfg.q = q;
        self.cfg.k = k;
        self
    }

    /// Set persistence.
    pub fn persistence(mut self, p: u32) -> Self {
        self.cfg.persistence = p;
        self
    }

    /// Set the per-type signal params.
    pub fn signal(mut self, ty: SynapseType, params: SignalParams) -> Self {
        match ty {
            SynapseType::H => self.cfg.signal_h = params,
            SynapseType::U => self.cfg.signal_u = params,
            SynapseType::D => self.cfg.signal_d = params,
        }
        self
    }

    /// Set the backoff strategy.
    pub fn backoff(mut self, backoff: BackoffConfig) -> Self {
        self.cfg.backoff = backoff;
        self
    }

    /// Set the CSA / sigmoid selection params.
    pub fn sigmoid(mut self, sigmoid: SigmoidConfig) -> Self {
        self.cfg.sigmoid = sigmoid;
        self
    }

    /// Enable the probabilistic CSA (learning samples winners from the `V→mu→rho`
    /// distribution) with default (Sparsey_Alt) coefficients. Leaves other sigmoid
    /// params at their defaults.
    pub fn enable_csa(mut self) -> Self {
        self.cfg.sigmoid.enabled = true;
        self
    }

    /// Force deterministic max-V winner selection (disable the probabilistic CSA).
    pub fn disable_csa(mut self) -> Self {
        self.cfg.sigmoid.enabled = false;
        self
    }

    /// Finish building.
    pub fn build(self) -> RegionConfig {
        self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip() {
        let cfg = NetworkConfig::builder()
            .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
            .region(
                RegionConfigBuilder::new("l1", 1)
                    .grid(2, 2)
                    .qk(3, 5)
                    .persistence(2)
                    .backoff(BackoffConfig::canonical(0.9, 0.93, 0.96))
                    .build(),
            )
            .connect("input", "l1")
            .build();

        let json = cfg.to_json().unwrap();
        let back = NetworkConfig::from_json(&json).unwrap();

        assert_eq!(back.regions.len(), 2);
        assert_eq!(back.connections.len(), 1);
        assert_eq!(back.region("l1").unwrap().q, 3);
        assert_eq!(back.region("l1").unwrap().k, 5);
        assert_eq!(back.region("l1").unwrap().backoff.priorities.len(), 3);
        assert_eq!(back.weight_table.max_stiffness, 2);
    }

    #[test]
    fn terse_json_uses_defaults() {
        let json = r#"{
            "regions": [
                {"name": "in", "height_in_dag": 0, "width_in_blks": 2, "height_in_blks": 2},
                {"name": "a", "height_in_dag": 1, "width_in_blks": 1, "height_in_blks": 1, "q": 4, "k": 8}
            ],
            "connections": [{"source": "in", "target": "a"}]
        }"#;
        let cfg = NetworkConfig::from_json(json).unwrap();
        assert_eq!(cfg.region("a").unwrap().q, 4);
        // Defaults filled in.
        assert_eq!(cfg.region("in").unwrap().persistence, 1);
        assert_eq!(cfg.weight_table.max_synapse_age, 30000);
        assert_eq!(cfg.region("a").unwrap().signal_u.max_cutoff, 1.0);
    }
}
