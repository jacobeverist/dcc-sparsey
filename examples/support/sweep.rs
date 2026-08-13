// Running a demo more than once, and comparing the results.
//
// A single run tells you very little. Whether a configuration learns is a question
// about a distribution, and the spread between seeds is routinely larger than the
// difference between the configurations being compared.
//
//     --repeat 5                      run five seeds, report mean +/- stddev
//     --sweep noise=0,0.25,0.5        run each value of --noise
//     --sweep noise=0,0.5 --repeat 5  both: five seeds at each of two settings
//
// `--sweep` works by overriding one argument and re-running, so no demo needs to
// know it is being swept — it reads its knobs off `Args` as usual.

use std::fmt::Write as _;

use crate::support::args::Args;
use crate::support::metrics::{Recorder, Summary};

/// One metric's distribution across the runs of a single sweep point.
#[derive(Clone, Debug)]
pub struct Aggregate {
    pub metric: String,
    pub n: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
}

/// The results at one setting of the swept parameter.
#[derive(Clone, Debug)]
pub struct SweepPoint {
    /// Empty when nothing is being swept.
    pub label: String,
    pub aggregates: Vec<Aggregate>,
    /// Fraction of runs whose verdict was `learned`.
    pub learned_fraction: f64,
}

impl SweepPoint {
    pub fn get(&self, metric: &str) -> Option<&Aggregate> {
        self.aggregates.iter().find(|a| a.metric == metric)
    }
}

/// Run a demo once, or many times if `--repeat` or `--sweep` asks for it.
///
/// This is every demo's `main`. With neither flag it is exactly one call to `run`
/// and nothing is printed beyond what the demo prints itself.
pub fn drive<F>(args: &Args, rec: &mut Recorder, mut run: F) -> Vec<SweepPoint>
where
    F: FnMut(&Args, u64, &mut Recorder) -> Summary,
{
    let base_seed = args.seed();
    let repeat: usize = args.get("repeat", 1).max(1);
    let sweep = parse_sweep(args.str("sweep"));

    let points: Vec<(String, Args)> = match &sweep {
        None => vec![(String::new(), args.clone_args())],
        Some((key, values)) => values
            .iter()
            .map(|v| (format!("{key}={v}"), args.with_override(key, v)))
            .collect(),
    };

    let matrix = points.len() > 1 || repeat > 1;

    // In matrix mode each individual run is silenced: a six-point sweep at five
    // seeds is thirty runs, and thirty full reports would bury the comparison table
    // the sweep exists to produce.
    let prepare = |a: &Args| {
        if matrix {
            a.with_flag("quiet").with_flag("silent")
        } else {
            a.clone_args()
        }
    };

    let mut results = Vec::new();
    let mut run_index = 0usize;

    for (label, point_args) in &points {
        let point_args = prepare(point_args);
        let mut summaries = Vec::with_capacity(repeat);

        for r in 0..repeat {
            // Seeds must stay nonzero for `Random`, which `Args::seed` guarantees
            // for the base; adding to it keeps that true.
            let seed = base_seed + r as u64;
            rec.begin_run(run_index, seed);
            if !label.is_empty() {
                // Record which sweep point this run belongs to, so a metrics file
                // covering a whole sweep can be split back apart.
                let (k, v) = label.split_once('=').unwrap_or((label.as_str(), ""));
                rec.config(k, v);
            }

            if matrix {
                println!(
                    "  run {:>3}/{:<3}{}  seed {seed}",
                    run_index + 1,
                    points.len() * repeat,
                    if label.is_empty() {
                        String::new()
                    } else {
                        format!("  {label}")
                    }
                );
            }

            summaries.push(run(&point_args, seed, rec));
            run_index += 1;
        }

        results.push(aggregate(label.clone(), &summaries));
    }

    if matrix {
        println!("\n{}", render(&results, repeat));
    }

    results
}

/// `key=v1,v2,v3` — the argument to `--sweep`.
fn parse_sweep(spec: Option<&str>) -> Option<(String, Vec<String>)> {
    let spec = spec?;
    let (key, values) = spec.split_once('=').unwrap_or_else(|| {
        panic!("--sweep expects key=v1,v2,...; got {spec:?}");
    });
    let values: Vec<String> = values
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(!values.is_empty(), "--sweep {spec:?} lists no values");
    Some((key.to_string(), values))
}

fn aggregate(label: String, summaries: &[Summary]) -> SweepPoint {
    let mut aggregates: Vec<Aggregate> = Vec::new();

    // Metric order follows the first summary, so the table reads the way the demo
    // reports rather than alphabetically.
    if let Some(first) = summaries.first() {
        for (name, _) in &first.metrics {
            let values: Vec<f64> = summaries
                .iter()
                .filter_map(|s| s.get(name))
                .filter(|v| v.is_finite())
                .collect();

            if values.is_empty() {
                continue;
            }

            let n = values.len();
            let mean = values.iter().sum::<f64>() / n as f64;
            // Sample standard deviation; zero for a single run rather than NaN.
            let stddev = if n > 1 {
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
            } else {
                0.0
            };

            aggregates.push(Aggregate {
                metric: name.clone(),
                n,
                mean,
                stddev,
                min: values.iter().copied().fold(f64::INFINITY, f64::min),
                max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            });
        }
    }

    let learned = summaries.iter().filter(|s| s.learned).count();
    let learned_fraction = if summaries.is_empty() {
        0.0
    } else {
        learned as f64 / summaries.len() as f64
    };

    SweepPoint {
        label,
        aggregates,
        learned_fraction,
    }
}

/// A table of mean ± stddev per metric, one column per sweep point.
fn render(points: &[SweepPoint], repeat: usize) -> String {
    let mut out = String::new();

    let swept = points.len() > 1;
    let _ = writeln!(
        out,
        "{} over {repeat} seed{}:",
        if swept { "Sweep" } else { "Repeats" },
        if repeat == 1 { "" } else { "s" }
    );

    // Union of metric names, in first-seen order.
    let mut names: Vec<String> = Vec::new();
    for p in points {
        for a in &p.aggregates {
            if !names.contains(&a.metric) {
                names.push(a.metric.clone());
            }
        }
    }

    let name_w = names.iter().map(|n| n.len()).max().unwrap_or(6).max(6);
    let col_w = 22usize;

    if swept {
        let _ = write!(out, "{:<name_w$} |", "metric");
        for p in points {
            let _ = write!(out, " {:^col_w$} |", p.label);
        }
        out.push('\n');
        let _ = writeln!(
            out,
            "{}",
            "-".repeat(name_w + 2 + points.len() * (col_w + 3))
        );
    }

    for name in &names {
        let _ = write!(out, "{name:<name_w$} |");
        for p in points {
            match p.get(name) {
                Some(a) if repeat > 1 => {
                    let _ = write!(
                        out,
                        " {:^col_w$} |",
                        format!("{:.4} ± {:.4}", a.mean, a.stddev)
                    );
                }
                Some(a) => {
                    let _ = write!(out, " {:^col_w$} |", format!("{:.4}", a.mean));
                }
                None => {
                    let _ = write!(out, " {:^col_w$} |", "—");
                }
            }
        }
        out.push('\n');
    }

    let _ = write!(out, "{:<name_w$} |", "learned");
    for p in points {
        let _ = write!(
            out,
            " {:^col_w$} |",
            format!("{:.0}% of runs", p.learned_fraction * 100.0)
        );
    }
    out.push('\n');

    if repeat > 1 {
        let _ = writeln!(
            out,
            "\n  Values are mean ± sample stddev. A difference smaller than the spread is not a\n  difference — that is the whole reason to repeat."
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_from(v: &[&str]) -> Args {
        Args::from_iter(v.iter().map(|s| s.to_string()))
    }

    fn summary(score: f64, learned: bool) -> Summary {
        let mut s = Summary::new();
        s.push("score", score);
        s.verdict(learned, "");
        s
    }

    #[test]
    fn a_plain_run_calls_the_demo_exactly_once() {
        let args = args_from(&["--seed", "5"]);
        let mut rec = Recorder::disabled("d");
        let mut seeds = Vec::new();
        drive(&args, &mut rec, |_, seed, _| {
            seeds.push(seed);
            summary(1.0, true)
        });
        assert_eq!(seeds, vec![5]);
    }

    #[test]
    fn repeat_uses_consecutive_seeds() {
        let args = args_from(&["--seed", "10", "--repeat", "4"]);
        let mut rec = Recorder::disabled("d");
        let mut seeds = Vec::new();
        drive(&args, &mut rec, |_, seed, _| {
            seeds.push(seed);
            summary(1.0, true)
        });
        assert_eq!(seeds, vec![10, 11, 12, 13]);
    }

    #[test]
    fn sweep_overrides_the_named_argument_at_each_point() {
        let args = args_from(&["--layers", "1", "--sweep", "layers=2,3,4"]);
        let mut rec = Recorder::disabled("d");
        let mut seen = Vec::new();
        drive(&args, &mut rec, |a, _, _| {
            seen.push(a.get::<usize>("layers", 0));
            summary(1.0, true)
        });
        assert_eq!(seen, vec![2, 3, 4]);
    }

    #[test]
    fn sweep_crosses_with_repeat() {
        let args = args_from(&["--seed", "1", "--repeat", "2", "--sweep", "k=a,b"]);
        let mut rec = Recorder::disabled("d");
        let mut seen = Vec::new();
        drive(&args, &mut rec, |a, seed, _| {
            seen.push((a.str("k").unwrap().to_string(), seed));
            summary(1.0, true)
        });
        // Each sweep point gets the same seed sequence, so points are comparable.
        assert_eq!(
            seen,
            vec![
                ("a".to_string(), 1),
                ("a".to_string(), 2),
                ("b".to_string(), 1),
                ("b".to_string(), 2)
            ]
        );
    }

    #[test]
    fn aggregate_reports_mean_and_sample_stddev() {
        let args = args_from(&["--repeat", "4"]);
        let mut rec = Recorder::disabled("d");
        let mut n = 0.0;
        let points = drive(&args, &mut rec, |_, _, _| {
            n += 1.0;
            summary(n, n > 2.0)
        });

        assert_eq!(points.len(), 1);
        let a = points[0].get("score").unwrap();
        assert_eq!(a.n, 4);
        assert!((a.mean - 2.5).abs() < 1e-9);
        // 1,2,3,4 -> sample stddev sqrt(5/3)
        assert!((a.stddev - (5.0f64 / 3.0).sqrt()).abs() < 1e-9);
        assert_eq!(a.min, 1.0);
        assert_eq!(a.max, 4.0);
        assert!((points[0].learned_fraction - 0.5).abs() < 1e-9);
    }

    #[test]
    fn non_finite_metrics_are_excluded_rather_than_poisoning_the_mean() {
        let args = args_from(&["--repeat", "3"]);
        let mut rec = Recorder::disabled("d");
        let mut i = 0;
        let points = drive(&args, &mut rec, |_, _, _| {
            i += 1;
            summary(if i == 2 { f64::NAN } else { 4.0 }, true)
        });
        let a = points[0].get("score").unwrap();
        assert_eq!(a.n, 2);
        assert!((a.mean - 4.0).abs() < 1e-9);
    }

    #[test]
    fn matrix_mode_silences_the_individual_runs() {
        let args = args_from(&["--repeat", "2"]);
        let mut rec = Recorder::disabled("d");
        let mut flags = Vec::new();
        drive(&args, &mut rec, |a, _, _| {
            flags.push((a.flag("quiet"), a.flag("silent")));
            summary(1.0, true)
        });
        assert_eq!(flags, vec![(true, true), (true, true)]);
    }

    #[test]
    fn a_single_run_is_left_as_verbose_as_the_user_asked_for() {
        let args = args_from(&["--seed", "1"]);
        let mut rec = Recorder::disabled("d");
        let mut flags = Vec::new();
        drive(&args, &mut rec, |a, _, _| {
            flags.push((a.flag("quiet"), a.flag("silent")));
            summary(1.0, true)
        });
        assert_eq!(flags, vec![(false, false)]);
    }

    #[test]
    #[should_panic(expected = "--sweep expects key=v1,v2")]
    fn a_malformed_sweep_spec_is_an_error_not_a_silent_single_run() {
        parse_sweep(Some("nonsense"));
    }
}
