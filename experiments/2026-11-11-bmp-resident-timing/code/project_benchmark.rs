//! Resident corrected-BMP latency adapter; retrieval semantics live in search.rs.
use anyhow::{ensure, Result};
use bmp::index::{forward_index::BlockForwardIndex, inverted_index::Index};
use bmp::query::{cursors_from_query_text, cursor::{RangeMaxScore, RangeMaxScoreCursor}};
use bmp::search::{canonical_prepared, CanonicalContext};
use bmp::util::to_trec;
use std::{collections::HashSet, fs::{self, OpenOptions}, io::Write, path::PathBuf,
          process::Command, thread, time::{Duration, Instant}};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Args {
    #[structopt(long)] raw_index: PathBuf,
    #[structopt(long)] compressed_index: PathBuf,
    #[structopt(long)] queries: PathBuf,
    #[structopt(long)] samples: PathBuf,
    #[structopt(long)] results: PathBuf,
    #[structopt(long)] monitoring: PathBuf,
    #[structopt(long)] expected_docs: usize,
    #[structopt(long)] expected_queries: usize,
    #[structopt(long)] cpu: usize,
    #[structopt(long)] k: usize,
    #[structopt(long)] block_size: usize,
    #[structopt(long)] warmups: usize,
    #[structopt(long)] repeats: usize,
}

// Runtime paths/config are restricted to ASCII JSON-safe strings rather than
// introducing a new dependency and changing the validated Cargo.lock.
fn quoted(s: &str) -> String {
    assert!(s.bytes().all(|c| c >= 32 && c < 127 && c != b'"' && c != b'\\'));
    format!("\"{}\"", s)
}

fn gate_timeout(value: &str) -> Result<Duration> {
    // The Python runner serializes its floating-point CLI value as e.g. 185.0.
    let seconds = value.parse::<f64>()?;
    ensure!(seconds.is_finite() && seconds > 0.0 && seconds <= 86400.0,
            "Runtime gate timeout must be finite and between zero and one day");
    Ok(Duration::from_secs_f64(seconds))
}

fn gate(a: &Args, docs: usize, queries: usize, raw_block: usize, compressed_block: usize) -> Result<()> {
    let directory = match std::env::var("RESEARCH_RUNTIME_DIR") { Ok(s) => PathBuf::from(s), Err(_) => return Ok(()) };
    let timeout = gate_timeout(&std::env::var("RESEARCH_GATE_SECONDS").unwrap_or_else(|_| "65".into()))?;
    let config = format!(concat!("{{\"mode\":\"bmp-resident-paired\",\"docs\":{},\"query_count\":{},",
        "\"k\":{},\"raw_block_size\":{},\"compressed_block_size\":{},\"cpu\":{},",
        "\"warmups\":{},\"repeats\":{},\"rayon_threads\":{},\"raw_index\":{},",
        "\"compressed_index\":{},\"queries_path\":{},\"alpha\":1.0,\"beta\":1.0,",
        "\"order\":\"alternating-round-plus-query\",\"boundary\":\"resident-query-through-trec\"}}"),
        docs, queries, a.k, raw_block, compressed_block, a.cpu, a.warmups, a.repeats,
        rayon::current_num_threads(), quoted(a.raw_index.to_str().unwrap()),
        quoted(a.compressed_index.to_str().unwrap()), quoted(a.queries.to_str().unwrap()));
    let behavior = concat!("{\"contract\":\"GSP-CAP32-U64-NUMERIC-TIES-v1\",",
        "\"arithmetic\":\"uint64\",\"document_impacts\":\"uint8-exact\",",
        "\"query_policy\":\"BMP-f32-ceil-cap32\",\"tie_order\":\"score_desc_original_numeric_id_asc\",",
        "\"eligibility\":\"all_documents\",\"resident_context\":\"numeric-ids-and-zero-order\",",
        "\"bounds\":\"raw-and-compressed\",\"selection\":\"sorted-wide-block-candidates\"}");
    let mut pending = OpenOptions::new().write(true).create_new(true).open(directory.join("runtime.pending"))?;
    writeln!(pending, "{{\"config\":{},\"behavior\":{}}}", config, behavior)?;
    pending.sync_all()?;
    ensure!(!directory.join("runtime.json").exists(), "Duplicate runtime report");
    fs::rename(directory.join("runtime.pending"), directory.join("runtime.json"))?;
    let start = Instant::now();
    while start.elapsed() < timeout {
        ensure!(!directory.join("reject").exists(), "Runtime gate rejected");
        if directory.join("continue").exists() { return Ok(()); }
        thread::sleep(Duration::from_millis(25));
    }
    anyhow::bail!("Runtime gate timed out")
}

fn validate_ranges(index: &Index, queries: &[(String, String)], compressed: bool) {
    let mut count = 0;
    for (_, text) in queries {
        for cursor in cursors_from_query_text(text, index) {
            assert_eq!(matches!(cursor.range_max_scores(), RangeMaxScore::Compressed(_)), compressed);
            count += 1;
        }
    }
    assert!(count > 0, "No live terms to verify representation");
}

fn query(index: &Index, forward: &BlockForwardIndex, context: &CanonicalContext,
         id: &str, text: &str, k: usize) -> (u128, String) {
    let start = Instant::now();
    let cursors = cursors_from_query_text(text, index);
    let heaps = canonical_prepared(vec![cursors], forward, k, context, false);
    let output = to_trec(&[id.to_string()], heaps, index.documents());
    let ns = start.elapsed().as_nanos();
    (ns, output)
}

fn monitor(cpu: usize) -> String {
    let stat = fs::read_to_string("/proc/stat").unwrap_or_default();
    let line = stat.lines().find(|s| s.starts_with(&format!("cpu{} ", cpu))).unwrap_or("unavailable");
    let load = fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unavailable".into());
    let frequency = fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", cpu))
        .unwrap_or_else(|_| "unavailable".into());
    format!("{}\t{}\t{}", line, load.split_whitespace().take(3).collect::<Vec<_>>().join(" "), frequency.trim())
}

fn main() -> Result<()> {
    let a = Args::from_args();
    ensure!(a.k > 0 && a.repeats > 0, "Positive k and repeats required");
    ensure!(rayon::current_num_threads() == 1, "Require RAYON_NUM_THREADS=1");
    let pinned = Command::new("taskset").args(["--pid", "--cpu-list", &a.cpu.to_string(), &std::process::id().to_string()]).output()?;
    ensure!(pinned.status.success(), "CPU affinity failed");
    let status = fs::read_to_string("/proc/self/status")?;
    let affinity = status.lines().find_map(|s| s.strip_prefix("Cpus_allowed_list:")).unwrap_or("").trim();
    ensure!(affinity == a.cpu.to_string(), "Affinity not a single requested CPU");
    let (raw, raw_fwd) = bmp::index::from_file(&a.raw_index)?;
    let (compressed, compressed_fwd) = bmp::index::from_file(&a.compressed_index)?;
    ensure!(raw.documents() == compressed.documents(), "Passage mapping differs");
    ensure!(raw.documents().len() == a.expected_docs, "Document count mismatch");
    ensure!(raw_fwd.block_size == a.block_size && compressed_fwd.block_size == a.block_size, "Block size mismatch");
    let queries = fs::read_to_string(&a.queries)?.lines().map(|line| {
        let (id, text) = line.split_once(':').expect("PISA query needs colon");
        assert!(!id.trim().is_empty() && id.trim().bytes().all(|c| c.is_ascii_alphanumeric() || b"_-.".contains(&c)));
        (id.trim().to_owned(), text.to_owned())
    }).collect::<Vec<_>>();
    ensure!(queries.len() == a.expected_queries && !queries.is_empty(), "Query count mismatch");
    ensure!(queries.iter().map(|(id,_)|id).collect::<HashSet<_>>().len() == queries.len(), "Duplicate query ID");
    validate_ranges(&raw, &queries, false);
    validate_ranges(&compressed, &queries, true);
    let context = CanonicalContext::new(raw.documents());
    let mut samples = OpenOptions::new().write(true).create_new(true).open(&a.samples)?;
    let mut results = OpenOptions::new().write(true).create_new(true).open(&a.results)?;
    let mut monitoring = OpenOptions::new().write(true).create_new(true).open(&a.monitoring)?;
    writeln!(samples, "phase\tround\tquery_index\torder\tlayout\tquery_id\telapsed_ns")?;
    writeln!(results, "phase\tround\tlayout\tquery_id\tdocument_id\trank\tscore")?;
    writeln!(monitoring, "phase\tround\tposition\tcpu_jiffies\tload_1_5_15\tfrequency_khz")?;
    gate(&a, raw.documents().len(), queries.len(), raw_fwd.block_size, compressed_fwd.block_size)?;
    for (phase, rounds) in [("warmup", a.warmups), ("measured", a.repeats)] {
        for round in 0..rounds {
            writeln!(monitoring, "{}\t{}\tbefore\t{}", phase, round, monitor(a.cpu))?;
            // A round's tiny outputs are retained and flushed outside measured regions.
            let mut saved = Vec::new();
            for (qi, (id, text)) in queries.iter().enumerate() {
                for order in 0..2 {
                    let is_compressed = (round + qi + order) % 2 == 1;
                    let (index, forward, layout) = if is_compressed { (&compressed, &compressed_fwd, "compressed") }
                        else { (&raw, &raw_fwd, "raw") };
                    let (ns, output) = query(index, forward, &context, id, text, a.k);
                    saved.push((qi, order, layout, id, ns, output));
                }
            }
            writeln!(monitoring, "{}\t{}\tafter\t{}", phase, round, monitor(a.cpu))?;
            for (qi, order, layout, id, ns, output) in saved {
                writeln!(samples, "{}\t{}\t{}\t{}\t{}\t{}\t{}", phase, round, qi, order, layout, id, ns)?;
                for line in output.lines() {
                    let fields = line.split_whitespace().collect::<Vec<_>>();
                    assert_eq!(fields.len(), 6);
                    writeln!(results, "{}\t{}\t{}\t{}\t{}\t{}\t{}", phase, round, layout, fields[0], fields[2], fields[3], fields[4])?;
                }
            }
            samples.flush()?; results.flush()?; monitoring.flush()?;
            eprintln!("ROUND_COMPLETE {} {}", phase, round);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::gate_timeout;
    use std::time::Duration;

    #[test]
    fn accepts_runner_decimal_timeout() {
        assert_eq!(gate_timeout("185.0").unwrap(), Duration::from_secs(185));
        assert_eq!(gate_timeout("65").unwrap(), Duration::from_secs(65));
        assert_eq!(gate_timeout("0.5").unwrap(), Duration::from_millis(500));
    }

    #[test]
    fn rejects_invalid_or_unbounded_timeout() {
        for value in ["NaN", "inf", "-1", "0", "86401", "invalid"] {
            assert!(gate_timeout(value).is_err(), "{}", value);
        }
    }
}
