// Minimal command-line parsing for the demos.
//
// Deliberately dependency-free, and that is a hard constraint here rather than a
// preference: `tests/conformance.rs::r12_getrandom_is_absent_from_the_graph` reads
// `Cargo.lock`, so a dev-dependency counts. `clap` pulls `getrandom` on several
// resolutions and would fail that test and the wasm32 CI step with it. This
// repository shipped exactly that regression once already — see the R12 note in
// `doc/Conformance.md` — so the rule is not theoretical.
//
// Accepted forms are `--key value`, `--key=value` and `--flag`. A `--key` followed
// by another `--`-prefixed token (or by nothing) is treated as a flag, so
// `--quiet --steps 10` parses the way you would expect.

use std::collections::{HashMap, HashSet};

pub struct Args {
    values: HashMap<String, String>,
    flags: HashSet<String>,
}

impl Args {
    /// Parse `std::env::args()`, skipping the binary name.
    pub fn parse() -> Self {
        Self::from_iter(std::env::args().skip(1))
    }

    pub fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let argv: Vec<String> = iter.into_iter().collect();
        let mut values = HashMap::new();
        let mut flags = HashSet::new();

        let mut i = 0;
        while i < argv.len() {
            let tok = &argv[i];
            if let Some(key) = tok.strip_prefix("--") {
                if let Some((k, v)) = key.split_once('=') {
                    values.insert(k.to_string(), v.to_string());
                    i += 1;
                    continue;
                }
                match argv.get(i + 1) {
                    Some(next) if !next.starts_with("--") => {
                        values.insert(key.to_string(), next.clone());
                        i += 2;
                    }
                    _ => {
                        flags.insert(key.to_string());
                        i += 1;
                    }
                }
            } else {
                i += 1;
            }
        }

        Args { values, flags }
    }

    pub fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    /// A plain copy.
    ///
    /// Named rather than deriving `Clone` so a caller has to mean it: `Args` is the
    /// whole configuration of a run, and copying one is normally a sign of a sweep.
    pub fn clone_args(&self) -> Args {
        Args {
            values: self.values.clone(),
            flags: self.flags.clone(),
        }
    }

    /// A copy with one value replaced. This is how `--sweep` varies a parameter:
    /// the demo reads its knobs off `Args`, so overriding one and re-running is
    /// enough — no demo needs to know it is being swept.
    pub fn with_override(&self, key: &str, value: &str) -> Args {
        let mut values = self.values.clone();
        values.insert(key.to_string(), value.to_string());
        Args {
            values,
            flags: self.flags.clone(),
        }
    }

    /// A copy with a flag set.
    pub fn with_flag(&self, key: &str) -> Args {
        let mut flags = self.flags.clone();
        flags.insert(key.to_string());
        Args {
            values: self.values.clone(),
            flags,
        }
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Look up `key`, falling back to `default`. A value that fails to parse is a
    /// hard error rather than a silent fallback — a typo'd `--steps 10O` that
    /// quietly ran the default would be worse than a panic.
    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T
    where
        T::Err: std::fmt::Display,
    {
        match self.values.get(key) {
            None => default,
            Some(raw) => match raw.parse::<T>() {
                Ok(v) => v,
                Err(e) => panic!("--{key}: cannot parse {raw:?}: {e}"),
            },
        }
    }

    /// The run seed.
    ///
    /// `SparseyNet::build(config, seed)` accepts any `u64`, including zero, so
    /// unlike the sibling HTM port there is nothing to guard against here. It is
    /// still worth going through one accessor: the seed is the whole of a run's
    /// reproducibility, and `--repeat` derives its sequence from this value.
    pub fn seed(&self) -> u64 {
        self.get("seed", 12345u64)
    }
}

#[cfg(test)]
mod tests {
    use super::Args;

    fn args(s: &[&str]) -> Args {
        Args::from_iter(s.iter().map(|x| x.to_string()))
    }

    #[test]
    fn parses_values_flags_and_equals_form() {
        let a = args(&["--steps", "500", "--quiet", "--seed=7"]);
        assert_eq!(a.get::<usize>("steps", 1), 500);
        assert_eq!(a.seed(), 7);
        assert!(a.flag("quiet"));
        assert!(!a.flag("steps"));
        assert_eq!(a.get::<usize>("missing", 42), 42);
    }

    #[test]
    fn overrides_replace_a_value_without_touching_the_original() {
        let base = args(&["--layers", "2", "--steps", "10"]);
        let derived = base.with_override("layers", "5");
        assert_eq!(derived.get::<usize>("layers", 0), 5);
        assert_eq!(derived.get::<usize>("steps", 0), 10);
        assert_eq!(base.get::<usize>("layers", 0), 2);
    }

    #[test]
    fn with_flag_adds_without_disturbing_values() {
        let base = args(&["--steps", "10"]);
        let derived = base.with_flag("silent");
        assert!(derived.flag("silent"));
        assert!(!base.flag("silent"));
        assert_eq!(derived.get::<usize>("steps", 0), 10);
    }

    #[test]
    fn flag_before_value_does_not_swallow_the_next_key() {
        let a = args(&["--quiet", "--steps", "10"]);
        assert!(a.flag("quiet"));
        assert_eq!(a.get::<usize>("steps", 1), 10);
    }

    #[test]
    fn the_default_seed_is_stable() {
        // Shared across the three ports so a reader comparing their output is
        // comparing runs that were seeded the same way.
        assert_eq!(args(&[]).seed(), 12345);
    }

    #[test]
    fn a_zero_seed_is_accepted_here() {
        // Unlike dcc-htm, where Random::new(0) panics. Xoshiro256PlusPlus takes it.
        assert_eq!(args(&["--seed", "0"]).seed(), 0);
    }
}
