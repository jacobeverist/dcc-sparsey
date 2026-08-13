// Saving and resuming what a demo learned.
//
// `SparseyNet::serialize_state` writes only the learned synapse state — the
// `(stiffness, timestamp)` pairs — and **not the structure**. Restoring means
// rebuilding the network from the same `NetworkConfig` with the same seed and then
// zipping the synapses back in arena order.
//
// That is a sharp trap, and guarding it is most of why this module exists.
// `load_state` validates the bundle *count* and nothing else, so a configuration
// that changed but happened to preserve the bundle count would load without
// complaint and silently associate every synapse with the wrong target. And because
// band-limited wiring draws from the RNG, the *seed* is part of the structure too:
// the same config at a different seed produces a different connectivity and the
// same silent corruption.
//
// So a checkpoint here carries a header — the seed and a hash of the config JSON —
// and refuses to load into anything that does not match.

use std::fs;
use std::path::Path;

use dcc_sparsey::{NetworkConfig, SparseyNet};

use crate::support::args::Args;

const MAGIC: &[u8; 8] = b"SPARSEY1";

/// FNV-1a over the canonical JSON of the config.
///
/// Hashing the serialised config rather than comparing structs: `NetworkConfig` is
/// a deep tree and this needs to detect *any* change to it, including one in a
/// field a future version adds.
fn config_hash(config: &NetworkConfig) -> u64 {
    let json = config.to_json().expect("config serialises");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in json.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Write the learned state, tagged with the seed and config it belongs to.
pub fn save(net: &SparseyNet, config: &NetworkConfig, seed: u64, path: &Path) -> Result<(), String> {
    let state = net
        .serialize_state()
        .map_err(|e| format!("serialising state: {e:?}"))?;

    let mut out = Vec::with_capacity(state.len() + 24);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&config_hash(config).to_le_bytes());
    out.extend_from_slice(&state);

    fs::write(path, &out).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read a checkpoint into a network built from the *same* config and seed.
///
/// Refuses rather than corrupts on any mismatch.
pub fn load(
    net: &mut SparseyNet,
    config: &NetworkConfig,
    seed: u64,
    path: &Path,
) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 24 || &bytes[..8] != MAGIC {
        return Err(format!("{}: not a Sparsey checkpoint", path.display()));
    }

    let saved_seed = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let saved_hash = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

    if saved_seed != seed {
        return Err(format!(
            "{}: checkpoint was written at seed {saved_seed}, this run is seed {seed}. \
             The structure is rebuilt from config + seed, so loading across seeds would \
             attach every synapse to the wrong target.",
            path.display()
        ));
    }
    if saved_hash != config_hash(config) {
        return Err(format!(
            "{}: the configuration has changed since this checkpoint was written. \
             Only the synapse state is stored, so it can only be loaded into the \
             structure it was learned in.",
            path.display()
        ));
    }

    net.load_state(&bytes[24..])
        .map_err(|e| format!("{}: {e:?}", path.display()))
}

/// Honour `--load` if it was given. Returns whether anything was loaded.
///
/// A missing or mismatched checkpoint is **fatal**. A run told to resume that
/// silently started from scratch would waste exactly the time the checkpoint
/// existed to save, and would report its results as though nothing had happened.
pub fn maybe_load(net: &mut SparseyNet, config: &NetworkConfig, seed: u64, args: &Args) -> bool {
    match args.str("load") {
        None => false,
        Some(path) => {
            if let Err(e) = load(net, config, seed, Path::new(path)) {
                panic!("--load {e}");
            }
            true
        }
    }
}

/// Honour `--save` if it was given.
pub fn maybe_save(net: &SparseyNet, config: &NetworkConfig, seed: u64, args: &Args) {
    if let Some(path) = args.str("save") {
        if let Err(e) = save(net, config, seed, Path::new(path)) {
            panic!("--save {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcc_sparsey::{NetworkConfigBuilder, RegionConfigBuilder};

    use crate::support::probe::Capture;

    fn config(q: u32) -> NetworkConfig {
        NetworkConfigBuilder::default()
            .region(RegionConfigBuilder::new("input", 0).grid(4, 4).build())
            .region(
                RegionConfigBuilder::new("l1", 1)
                    .grid(1, 1)
                    .qk(q, 8)
                    .persistence(1)
                    .build(),
            )
            .connect("input", "l1")
            .build()
    }

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("dcc_sparsey_ckpt_{name}.bin"))
    }

    fn learn_and_read(net: &mut SparseyNet, feats: &[u32], learn: bool) -> Vec<u32> {
        let input = net.region_id("input").unwrap();
        net.set_input(input, feats).unwrap();
        let mut cap = Capture::new();
        if learn {
            net.do_frame_learn_rec(&mut cap);
        } else {
            net.do_frame_recognize_rec(&mut cap);
        }
        cap.first_code().unwrap_or(&[]).to_vec()
    }

    #[test]
    fn a_reloaded_network_reproduces_the_saved_ones_codes() {
        let cfg = config(5);
        let mut trained = SparseyNet::build(cfg.clone(), 7).unwrap();
        for feats in [&[0u32, 1, 2, 3][..], &[12, 13, 14, 15][..], &[0, 4, 8, 12][..]] {
            learn_and_read(&mut trained, feats, true);
        }
        trained.finalize_learning();

        let path = temp("resume");
        save(&trained, &cfg, 7, &path).unwrap();

        let mut restored = SparseyNet::build(cfg.clone(), 7).unwrap();
        load(&mut restored, &cfg, 7, &path).unwrap();

        trained.prepare_for_new_run(false);
        restored.prepare_for_new_run(false);
        for feats in [&[0u32, 1, 2, 3][..], &[12, 13, 14, 15][..]] {
            assert_eq!(
                learn_and_read(&mut trained, feats, false),
                learn_and_read(&mut restored, feats, false),
                "codes diverged after reload for {feats:?}"
            );
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn the_checkpoint_actually_contains_the_learned_state() {
        let cfg = config(5);
        let mut net = SparseyNet::build(cfg.clone(), 7).unwrap();
        learn_and_read(&mut net, &[0, 1, 2, 3], true);

        let path = temp("nonempty");
        save(&net, &cfg, 7, &path).unwrap();
        // Guards the round trip that "passes" because nothing was written: two
        // empty files also compare equal.
        let len = fs::metadata(&path).unwrap().len();
        assert!(len > 24, "checkpoint is only {len} bytes — header and nothing else");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn loading_across_seeds_is_refused_rather_than_silently_wrong() {
        let cfg = config(5);
        let net = SparseyNet::build(cfg.clone(), 7).unwrap();
        let path = temp("seed");
        save(&net, &cfg, 7, &path).unwrap();

        let mut other = SparseyNet::build(cfg.clone(), 8).unwrap();
        let err = load(&mut other, &cfg, 8, &path).unwrap_err();
        // The structure is rebuilt from config + seed, so this would attach every
        // synapse to the wrong target — and load_state alone would not notice.
        assert!(err.contains("seed 7"), "{err}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn loading_across_configs_is_refused_even_when_the_shape_survives() {
        let cfg_a = config(5);
        let net = SparseyNet::build(cfg_a.clone(), 7).unwrap();
        let path = temp("config");
        save(&net, &cfg_a, 7, &path).unwrap();

        let cfg_b = config(6);
        let mut other = SparseyNet::build(cfg_b.clone(), 7).unwrap();
        let err = load(&mut other, &cfg_b, 7, &path).unwrap_err();
        assert!(err.contains("configuration has changed"), "{err}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_that_is_not_a_checkpoint_is_rejected() {
        let path = temp("garbage");
        fs::write(&path, b"not a checkpoint at all").unwrap();
        let cfg = config(5);
        let mut net = SparseyNet::build(cfg.clone(), 7).unwrap();
        assert!(load(&mut net, &cfg, 7, &path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[should_panic(expected = "--load")]
    fn a_missing_checkpoint_is_fatal_rather_than_a_silent_fresh_start() {
        let cfg = config(5);
        let mut net = SparseyNet::build(cfg.clone(), 7).unwrap();
        let args = Args::from_iter(
            ["--load", "/nonexistent/dcc_sparsey_no_such.bin"]
                .iter()
                .map(|s| s.to_string()),
        );
        maybe_load(&mut net, &cfg, 7, &args);
    }
}
