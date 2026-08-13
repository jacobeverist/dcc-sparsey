// Machine-readable output for the demos.
//
// Stdout stays exactly as it is: `--metrics` is strictly additive, and `--quiet`
// still governs printing only. Without `--metrics` the recorder is inert — no file,
// no buffer, and `sample()` returns before touching its arguments.
//
// Four record kinds are emitted in order, one JSON object per line:
//
//   {"kind":"run","demo":"anomaly_stream","seed":12345,"run":0,"config":{...}}
//   {"kind":"sample","demo":...,"seed":...,"run":...,"step":5000,"metrics":{...}}
//   {"kind":"summary","demo":...,"seed":...,"run":...,"metrics":{...}}
//   {"kind":"verdict","demo":...,"seed":...,"run":...,"learned":true,"note":"..."}
//
// Every record carries `demo`, `seed` and `run`, so files from different demos or
// different seeds can be concatenated and still make sense. This is the cross-repo
// demo contract; `dcc-sph` and `dcc-sparsey` emit the same shape from their own
// implementations. See `doc/Demos.md`.
//
// Unlike `dcc-sph`, which hand-rolls its JSON because `serde_json` is not available
// to it, this uses the `serde_json` dev-dependency the fidelity harness already
// pulls in. That buys correct string escaping and, importantly, non-finite floats
// serialising as `null` rather than as bare `NaN` — so a metrics file always parses
// even when a metric is undefined.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use crate::support::args::Args;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetricValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

macro_rules! metric_from_int {
    ($($t:ty),*) => { $(
        impl From<$t> for MetricValue {
            fn from(v: $t) -> Self { MetricValue::Int(v as i64) }
        }
    )* };
}
metric_from_int!(i64, usize, u64, i32, u32, u16, u8);

impl From<f32> for MetricValue {
    fn from(v: f32) -> Self {
        MetricValue::Float(v as f64)
    }
}
impl From<f64> for MetricValue {
    fn from(v: f64) -> Self {
        MetricValue::Float(v)
    }
}
impl From<bool> for MetricValue {
    fn from(v: bool) -> Self {
        MetricValue::Bool(v)
    }
}
impl From<&str> for MetricValue {
    fn from(v: &str) -> Self {
        MetricValue::Text(v.to_string())
    }
}
impl From<String> for MetricValue {
    fn from(v: String) -> Self {
        MetricValue::Text(v)
    }
}

/// A JSON object built from an ordered slice rather than from a map.
///
/// `serde_json::Map` is a `BTreeMap` unless the `preserve_order` feature is on, and
/// turning that on would add `indexmap` to the lockfile for cosmetic reasons. The
/// order matters: metrics should read in the order the demo reports them, not
/// alphabetically, because that order is how the author grouped them.
struct Ordered<'a, K, V>(&'a [(K, V)]);

impl<K: Serialize, V: Serialize> Serialize for Ordered<'_, K, V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

#[derive(Serialize)]
struct Head<'a> {
    kind: &'static str,
    demo: &'a str,
    seed: u64,
    run: usize,
}

#[derive(Serialize)]
struct RunRec<'a> {
    #[serde(flatten)]
    head: Head<'a>,
    config: &'a BTreeMap<String, MetricValue>,
}

#[derive(Serialize)]
struct MetricsRec<'a, M: Serialize> {
    #[serde(flatten)]
    head: Head<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    step: Option<u64>,
    metrics: M,
}

#[derive(Serialize)]
struct VerdictRec<'a> {
    #[serde(flatten)]
    head: Head<'a>,
    learned: bool,
    note: &'a str,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Jsonl,
    Csv,
}

/// What one run produced: named metrics plus the verdict.
///
/// Deliberately untyped beyond `f64` so `sweep` can aggregate across runs without
/// knowing what any particular demo measures.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub metrics: Vec<(String, f64)>,
    pub learned: bool,
    pub note: String,
}

impl Summary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: &str, value: f64) {
        self.metrics.push((name.to_string(), value));
    }

    /// Record the outcome. Note that a *correct negative result* is `learned =
    /// true` with a note explaining why more training cannot change it — see the
    /// contract in `doc/Demos.md`.
    pub fn verdict(&mut self, learned: bool, note: impl Into<String>) {
        self.learned = learned;
        self.note = note.into();
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }
}

enum Sink {
    None,
    File(BufWriter<File>),
    Buffer(Vec<u8>),
}

pub struct Recorder {
    demo: &'static str,
    seed: u64,
    run_index: usize,
    config: BTreeMap<String, MetricValue>,
    sink: Sink,
    format: Format,
    header_written: bool,
}

impl Recorder {
    /// Read `--metrics <path>` and `--metrics-format {jsonl,csv}` off `Args`.
    pub fn from_args(demo: &'static str, args: &Args) -> Self {
        let format = match args.str("metrics-format") {
            None | Some("jsonl") => Format::Jsonl,
            Some("csv") => Format::Csv,
            Some(other) => panic!("--metrics-format: expected jsonl or csv, got {other:?}"),
        };

        let sink = match args.str("metrics") {
            None => Sink::None,
            Some(path) => {
                let path = PathBuf::from(path);
                match File::create(&path) {
                    Ok(f) => Sink::File(BufWriter::new(f)),
                    // Fatal, unlike a *write* failure below. A run told to record
                    // that silently recorded nothing would waste the whole run.
                    Err(e) => panic!("--metrics {}: {e}", path.display()),
                }
            }
        };

        Recorder {
            demo,
            seed: args.seed(),
            run_index: 0,
            config: BTreeMap::new(),
            sink,
            format,
            header_written: false,
        }
    }

    pub fn disabled(demo: &'static str) -> Self {
        Recorder {
            demo,
            seed: 0,
            run_index: 0,
            config: BTreeMap::new(),
            sink: Sink::None,
            format: Format::Jsonl,
            header_written: false,
        }
    }

    /// Record into memory, for tests that want to inspect the output.
    pub fn to_buffer(demo: &'static str, seed: u64, format: Format) -> Self {
        Recorder {
            demo,
            seed,
            run_index: 0,
            config: BTreeMap::new(),
            sink: Sink::Buffer(Vec::new()),
            format,
            header_written: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self.sink, Sink::None)
    }

    /// Start a new run within the same file. Called by `sweep::drive`.
    pub fn begin_run(&mut self, run_index: usize, seed: u64) {
        self.run_index = run_index;
        self.seed = seed;
        self.config.clear();
        self.header_written = false;
    }

    /// Record a configuration value. Must be called before the first `sample`,
    /// which is what writes the run header.
    pub fn config(&mut self, key: &str, value: impl Into<MetricValue>) {
        if self.is_enabled() {
            self.config.insert(key.to_string(), value.into());
        }
    }

    fn head(&self, kind: &'static str) -> Head<'_> {
        Head {
            kind,
            demo: self.demo,
            seed: self.seed,
            run: self.run_index,
        }
    }

    fn write_line(&mut self, line: &str) {
        let bytes = line.as_bytes();
        let result = match &mut self.sink {
            Sink::None => return,
            Sink::File(f) => f.write_all(bytes).and_then(|_| f.write_all(b"\n")),
            Sink::Buffer(b) => {
                b.extend_from_slice(bytes);
                b.push(b'\n');
                Ok(())
            }
        };
        // A failed write warns and continues. Killing a training run an hour in
        // because a disk filled would destroy more than it protects.
        if let Err(e) = result {
            eprintln!("warning: could not write metrics: {e}");
        }
    }

    fn ensure_header(&mut self) {
        if self.header_written || !self.is_enabled() {
            return;
        }
        self.header_written = true;

        match self.format {
            Format::Jsonl => {
                let line = serde_json::to_string(&RunRec {
                    head: self.head("run"),
                    config: &self.config,
                })
                .expect("run record serialises");
                self.write_line(&line);
            }
            Format::Csv => {
                if self.run_index == 0 {
                    self.write_line("demo,seed,run,kind,step,metric,value");
                }
                let rows: Vec<String> = self
                    .config
                    .iter()
                    .map(|(k, v)| {
                        let value = match v {
                            MetricValue::Int(i) => i.to_string(),
                            MetricValue::Float(f) => f.to_string(),
                            MetricValue::Bool(b) => b.to_string(),
                            MetricValue::Text(t) => t.clone(),
                        };
                        format!(
                            "{},{},{},config,,{},{}",
                            csv_field(self.demo),
                            self.seed,
                            self.run_index,
                            csv_field(k),
                            csv_field(&value)
                        )
                    })
                    .collect();
                for row in rows {
                    self.write_line(&row);
                }
            }
        }
    }

    pub fn sample(&mut self, step: u64, metrics: &[(&str, f64)]) {
        if !self.is_enabled() {
            return;
        }
        self.ensure_header();

        match self.format {
            Format::Jsonl => {
                let line = serde_json::to_string(&MetricsRec {
                    head: self.head("sample"),
                    step: Some(step),
                    metrics: Ordered(metrics),
                })
                .expect("sample record serialises");
                self.write_line(&line);
            }
            Format::Csv => self.write_rows("sample", Some(step), metrics),
        }
    }

    pub fn summary(&mut self, metrics: &[(&str, f64)]) {
        if !self.is_enabled() {
            return;
        }
        self.ensure_header();

        match self.format {
            Format::Jsonl => {
                let line = serde_json::to_string(&MetricsRec {
                    head: self.head("summary"),
                    step: None,
                    metrics: Ordered(metrics),
                })
                .expect("summary record serialises");
                self.write_line(&line);
            }
            Format::Csv => self.write_rows("summary", None, metrics),
        }
    }

    /// Emit a `Summary`'s metrics and its verdict together.
    pub fn finish_summary(&mut self, summary: &Summary) {
        if !self.is_enabled() {
            return;
        }
        self.ensure_header();

        match self.format {
            Format::Jsonl => {
                let line = serde_json::to_string(&MetricsRec {
                    head: self.head("summary"),
                    step: None,
                    metrics: Ordered(&summary.metrics),
                })
                .expect("summary record serialises");
                self.write_line(&line);
            }
            Format::Csv => {
                let pairs: Vec<(&str, f64)> = summary
                    .metrics
                    .iter()
                    .map(|(k, v)| (k.as_str(), *v))
                    .collect();
                self.write_rows("summary", None, &pairs);
            }
        }

        self.verdict(summary.learned, &summary.note);
    }

    pub fn verdict(&mut self, learned: bool, note: &str) {
        if !self.is_enabled() {
            return;
        }
        self.ensure_header();

        match self.format {
            Format::Jsonl => {
                let line = serde_json::to_string(&VerdictRec {
                    head: self.head("verdict"),
                    learned,
                    note,
                })
                .expect("verdict record serialises");
                self.write_line(&line);
            }
            Format::Csv => {
                let row = format!(
                    "{},{},{},verdict,,learned,{}",
                    csv_field(self.demo),
                    self.seed,
                    self.run_index,
                    if learned { 1 } else { 0 }
                );
                self.write_line(&row);
            }
        }
    }

    fn write_rows(&mut self, kind: &str, step: Option<u64>, metrics: &[(&str, f64)]) {
        let step = step.map(|s| s.to_string()).unwrap_or_default();
        let rows: Vec<String> = metrics
            .iter()
            .map(|(name, value)| {
                format!(
                    "{},{},{},{kind},{step},{},{}",
                    csv_field(self.demo),
                    self.seed,
                    self.run_index,
                    csv_field(name),
                    if value.is_finite() {
                        value.to_string()
                    } else {
                        String::new()
                    }
                )
            })
            .collect();
        for row in rows {
            self.write_line(&row);
        }
    }

    pub fn finish(mut self) {
        if let Sink::File(f) = &mut self.sink {
            if let Err(e) = f.flush() {
                eprintln!("warning: could not flush metrics: {e}");
            }
        }
    }

    pub fn buffer(&self) -> Option<&[u8]> {
        match &self.sink {
            Sink::Buffer(b) => Some(b),
            _ => None,
        }
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(rec: &Recorder) -> Vec<String> {
        String::from_utf8(rec.buffer().unwrap().to_vec())
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn a_disabled_recorder_writes_nothing_at_all() {
        let mut rec = Recorder::disabled("d");
        rec.config("k", 1);
        rec.sample(1, &[("a", 1.0)]);
        rec.verdict(true, "x");
        assert!(rec.buffer().is_none());
        assert!(!rec.is_enabled());
    }

    #[test]
    fn records_appear_in_contract_order_with_the_shared_fields() {
        let mut rec = Recorder::to_buffer("d", 7, Format::Jsonl);
        rec.config("steps", 10usize);
        rec.sample(5, &[("a", 1.0)]);
        let mut s = Summary::new();
        s.push("a", 2.0);
        s.verdict(true, "note");
        rec.finish_summary(&s);

        let ls = lines(&rec);
        let kinds: Vec<String> = ls
            .iter()
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l).unwrap()["kind"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(kinds, vec!["run", "sample", "summary", "verdict"]);

        for l in &ls {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            assert_eq!(v["demo"], "d");
            assert_eq!(v["seed"], 7);
            assert_eq!(v["run"], 0);
        }
    }

    #[test]
    fn metric_order_follows_the_demo_not_the_alphabet() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.sample(1, &[("zebra", 1.0), ("alpha", 2.0)]);
        let sample = &lines(&rec)[1];
        let z = sample.find("zebra").unwrap();
        let a = sample.find("alpha").unwrap();
        assert!(z < a, "serde_json's default Map would have sorted these: {sample}");
    }

    #[test]
    fn non_finite_values_serialise_as_null_so_the_file_still_parses() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.sample(1, &[("nan", f64::NAN), ("inf", f64::INFINITY)]);
        let sample = &lines(&rec)[1];
        let v: serde_json::Value = serde_json::from_str(sample).expect("still parses");
        assert!(v["metrics"]["nan"].is_null());
        assert!(v["metrics"]["inf"].is_null());
    }

    #[test]
    fn config_is_sorted_so_two_runs_are_byte_identical() {
        let build = || {
            let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
            rec.config("zeta", 1);
            rec.config("alpha", 2);
            rec.sample(1, &[("a", 1.0)]);
            String::from_utf8(rec.buffer().unwrap().to_vec()).unwrap()
        };
        assert_eq!(build(), build());
        assert!(build().contains(r#"{"alpha":2,"zeta":1}"#));
    }

    #[test]
    fn the_run_header_is_written_lazily_so_config_can_come_first() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.config("a", 1);
        assert!(rec.buffer().unwrap().is_empty(), "nothing written yet");
        rec.sample(1, &[("m", 1.0)]);
        assert!(lines(&rec)[0].contains(r#""kind":"run""#));
    }

    #[test]
    fn begin_run_resets_config_so_runs_do_not_leak_into_each_other() {
        let mut rec = Recorder::to_buffer("d", 1, Format::Jsonl);
        rec.config("only_first", 1);
        rec.sample(1, &[("m", 1.0)]);
        rec.begin_run(1, 2);
        rec.sample(1, &[("m", 1.0)]);
        let header = &lines(&rec)[2];
        assert!(!header.contains("only_first"), "{header}");
        assert!(header.contains(r#""run":1"#));
    }

    #[test]
    fn csv_is_long_format_with_one_row_per_number() {
        let mut rec = Recorder::to_buffer("d", 3, Format::Csv);
        rec.config("steps", 10usize);
        rec.sample(5, &[("a", 1.5), ("b", 2.5)]);
        let ls = lines(&rec);
        assert_eq!(ls[0], "demo,seed,run,kind,step,metric,value");
        assert_eq!(ls[1], "d,3,0,config,,steps,10");
        assert_eq!(ls[2], "d,3,0,sample,5,a,1.5");
        assert_eq!(ls[3], "d,3,0,sample,5,b,2.5");
    }

    #[test]
    fn csv_quotes_fields_containing_separators() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_field("plain"), "plain");
    }

    #[test]
    fn summary_get_finds_what_push_recorded() {
        let mut s = Summary::new();
        s.push("accuracy", 0.5);
        assert_eq!(s.get("accuracy"), Some(0.5));
        assert_eq!(s.get("missing"), None);
    }
}
