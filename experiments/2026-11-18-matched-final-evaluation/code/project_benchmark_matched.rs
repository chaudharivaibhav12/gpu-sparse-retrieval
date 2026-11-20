//! Matched parsed-query to ordered numeric top-k BMP adapter. Search semantics unchanged.
use anyhow::{ensure, Result};
use bmp::index::{forward_index::BlockForwardIndex, inverted_index::Index};
use bmp::query::{cursors_from_query_text, cursor::{RangeMaxScore, RangeMaxScoreCursor}};
use bmp::search::{canonical_prepared, CanonicalContext};

use std::{collections::{HashSet, BTreeMap}, fs::{self, OpenOptions}, io::Write, path::PathBuf,
          process::Command, thread, time::{Duration, Instant}};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Args {
    #[structopt(long)] index: PathBuf,
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

fn quoted(s: &str) -> String {
    assert!(s.bytes().all(|c| c >= 32 && c < 127 && c != b'"' && c != b'\\'));
    format!("\"{}\"", s)
}

fn gate_timeout(value: &str) -> Result<Duration> {
    let seconds = value.parse::<f64>()?;
    ensure!(seconds.is_finite() && seconds > 0.0 && seconds <= 86400.0,
            "Runtime gate timeout must be finite and between zero and one day");
    Ok(Duration::from_secs_f64(seconds))
}

fn gate(a: &Args, docs: usize, queries: usize, block: usize) -> Result<()> {
    let directory = match std::env::var("RESEARCH_RUNTIME_DIR") { Ok(s) => PathBuf::from(s), Err(_) => return Ok(()) };
    let timeout = gate_timeout(&std::env::var("RESEARCH_GATE_SECONDS").unwrap_or_else(|_| "65".into()))?;
    let config = format!(concat!("{{\"mode\":\"bmp-resident-parsed\",\"docs\":{},\"query_count\":{},",
        "\"k\":{},\"block_size\":{},\"cpu\":{},\"warmups\":{},\"repeats\":{},\"rayon_threads\":{},",
        "\"index\":{},\"queries_path\":{},\"alpha\":1.0,\"beta\":1.0,",
        "\"order\":\"file-order\",\"boundary\":\"parsed-query-through-host-numeric-topk\"}}"),
        docs, queries, a.k, block, a.cpu, a.warmups, a.repeats, rayon::current_num_threads(),
        quoted(a.index.to_str().unwrap()), quoted(a.queries.to_str().unwrap()));
    let behavior = concat!("{\"contract\":\"GSP-CAP32-U64-NUMERIC-TIES-v1\",",
        "\"arithmetic\":\"uint64\",\"document_impacts\":\"uint8-exact\",",
        "\"query_policy\":\"BMP-f32-ceil-cap32\",\"tie_order\":\"score_desc_original_numeric_id_asc\",",
        "\"eligibility\":\"all_documents\",\"resident_context\":\"numeric-ids-and-zero-order\",",
        "\"bounds\":\"raw\",\"selection\":\"sorted-wide-block-candidates\"}");
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

fn validate_raw(index: &Index, queries: &[(String, String)]) {
    let mut count = 0;
    for (_, text) in queries {
        for cursor in cursors_from_query_text(text, index) {
            assert!(!matches!(cursor.range_max_scores(), RangeMaxScore::Compressed(_)), "Index has compressed ranges");
            count += 1;
        }
    }
    assert!(count > 0, "No live terms to verify representation");
}

// Query text is decoded and validated once, outside all query timing. The exported
// PISA representation already contains cap32 integer weights as term repetitions.
// A deterministic lexical order replaces the parent's randomized HashMap order;
// this changes traversal order only, and is disclosed and checked against the oracle.
fn prepared_terms(text: &str) -> Result<Vec<(String, u32)>> {
    let mut terms = BTreeMap::<String, u32>::new();
    for token in text.split_whitespace() {
        let n = terms.entry(token.to_owned()).or_insert(0);
        *n = n.checked_add(1).ok_or_else(|| anyhow::anyhow!("Query count overflow"))?;
        ensure!(*n <= 32, "Export must already contain cap32 weights");
    }
    ensure!(terms.len() <= 128, "Query exceeds shared supported term domain");
    Ok(terms.into_iter().collect())
}

fn query(index: &Index, forward: &BlockForwardIndex, context: &CanonicalContext,
         numeric_ids: &[u64], terms: &[(String, u32)], k: usize) -> (u128, Vec<(u64, u64)>) {
    let start = Instant::now();
    let cursors = terms.iter().flat_map(|(token, weight)| index.get_cursor(token, *weight)).collect();
    let mut heaps = canonical_prepared(vec![cursors], forward, k, context, false);
    let output = heaps[0].to_sorted_vec().into_iter()
        .map(|r| (numeric_ids[r.doc_id.0 as usize], r.score)).collect();
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
    let (index, forward) = bmp::index::from_file(&a.index)?;
    ensure!(index.documents().len() == a.expected_docs, "Document count mismatch");
    ensure!(forward.block_size == a.block_size, "Block size mismatch");
    let queries = fs::read_to_string(&a.queries)?.lines().map(|line| {
        let (id, text) = line.split_once(':').expect("PISA query needs colon");
        assert!(!id.trim().is_empty() && id.trim().bytes().all(|c| c.is_ascii_alphanumeric() || b"_-.".contains(&c)));
        (id.trim().to_owned(), text.to_owned())
    }).collect::<Vec<_>>();
    ensure!(queries.len() == a.expected_queries && !queries.is_empty(), "Query count mismatch");
    ensure!(queries.iter().map(|(id,_)|id).collect::<HashSet<_>>().len() == queries.len(), "Duplicate query ID");
    validate_raw(&index, &queries);
    let context = CanonicalContext::new(index.documents());
    let numeric_ids = index.documents().iter().map(|s| s.parse::<u64>()).collect::<std::result::Result<Vec<_>, _>>()?;
    let prepared = queries.iter().map(|(_, text)| prepared_terms(text)).collect::<Result<Vec<_>>>()?;
    let mut samples = OpenOptions::new().write(true).create_new(true).open(&a.samples)?;
    let mut results = OpenOptions::new().write(true).create_new(true).open(&a.results)?;
    let mut monitoring = OpenOptions::new().write(true).create_new(true).open(&a.monitoring)?;
    writeln!(samples, "phase\tround\tquery_index\torder\tlayout\tquery_id\telapsed_ns")?;
    writeln!(results, "phase\tround\tlayout\tquery_id\tdocument_id\trank\tscore")?;
    writeln!(monitoring, "phase\tround\tposition\tcpu_jiffies\tload_1_5_15\tfrequency_khz")?;
    gate(&a, index.documents().len(), queries.len(), forward.block_size)?;
    for (phase, rounds) in [("warmup", a.warmups), ("measured", a.repeats)] {
        for round in 0..rounds {
            writeln!(monitoring, "{}\t{}\tbefore\t{}", phase, round, monitor(a.cpu))?;
            let mut saved = Vec::new();
            for (qi, (id, _text)) in queries.iter().enumerate() {
                let (ns, output) = query(&index, &forward, &context, &numeric_ids, &prepared[qi], a.k);
                saved.push((qi, id, ns, output));
            }
            writeln!(monitoring, "{}\t{}\tafter\t{}", phase, round, monitor(a.cpu))?;
            for (qi, id, ns, output) in saved {
                writeln!(samples, "{}\t{}\t{}\t0\traw\t{}\t{}", phase, round, qi, id, ns)?;
                for (rank, (doc, score)) in output.into_iter().enumerate() {
                    writeln!(results, "{}\t{}\traw\t{}\t{}\t{}\t{}", phase, round, id, doc, rank + 1, score)?;
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
    use super::{gate_timeout, prepared_terms};
    use std::time::Duration;

    #[test]
    fn prepared_query_domain_and_order() {
        assert_eq!(prepared_terms("z a z").unwrap(), vec![("a".to_string(), 1), ("z".to_string(), 2)]);
        assert!(prepared_terms("").unwrap().is_empty());
        assert_eq!(prepared_terms(&vec!["a"; 32].join(" ")).unwrap()[0].1, 32);
        assert!(prepared_terms(&vec!["a"; 33].join(" ")).is_err());
        assert!(prepared_terms(&(0..129).map(|i| i.to_string()).collect::<Vec<_>>().join(" ")).is_err());
    }

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
