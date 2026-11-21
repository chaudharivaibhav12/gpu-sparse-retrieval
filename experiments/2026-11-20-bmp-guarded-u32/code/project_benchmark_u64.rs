//! Matched CPU score-width ablation; 64-bit temporary arithmetic, U64 host output.
type Score = u64;
const CONTRACT: &str = "GSP-CAP32-U64-NUMERIC-TIES-v1";
const ARITHMETIC: &str = "uint64";
use anyhow::{ensure, Result};
use bmp::index::{forward_index::BlockForwardIndex, inverted_index::Index};
use bmp::project_u32::{GuardedQuery, VerifiedCollection};
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
    #[structopt(long)] expected_vocab: usize,
    #[structopt(long)] expected_nnz: u64,
    #[structopt(long)] validation: PathBuf,
    #[structopt(long)] memory_report: PathBuf,
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

fn gate(a: &Args, checked: &VerifiedCollection, prepared: &[GuardedQuery]) -> Result<()> {
    let directory = match std::env::var("RESEARCH_RUNTIME_DIR") { Ok(s) => PathBuf::from(s), Err(_) => return Ok(()) };
    let timeout = gate_timeout(&std::env::var("RESEARCH_GATE_SECONDS").unwrap_or_else(|_| "65".into()))?;
    let stats = checked.stats();
    let max_terms = prepared.iter().map(|q|q.terms().len()).max().unwrap_or(0);
    let max_bound = prepared.iter().map(|q|q.bound()).max().unwrap_or(0);
    let config = format!(concat!(r#"{{"mode":"bmp-resident-score-width","docs":{},"query_count":{},"k":{},"block_size":{},"cpu":{},"warmups":{},"repeats":{},"rayon_threads":{},"index":{},"queries_path":{},"alpha":1.0,"beta":1.0,"order":"file-order","boundary":"parsed-query-through-host-numeric-topk","score_bits":{},"bound_score_bits":{},"heap_score_bits":{},"original_id_bits":64,"vocab":{},"nnz":{},"max_query_terms":{},"max_conservative_score_bound":{},"index_domain_verified":true}}"#),
        stats.docs, prepared.len(), a.k, stats.block_size, a.cpu, a.warmups, a.repeats,
        rayon::current_num_threads(),quoted(a.index.to_str().unwrap()),quoted(a.queries.to_str().unwrap()),
        std::mem::size_of::<Score>()*8,std::mem::size_of::<Score>()*8,std::mem::size_of::<Score>()*8,
        stats.vocab,stats.nnz,max_terms,max_bound);
    let behavior = format!(r#"{{"contract":"{}","arithmetic":"{}","document_impacts":"uint8-exact","query_policy":"BMP-f32-ceil-cap32","tie_order":"score_desc_original_numeric_id_asc","eligibility":"all_documents","resident_context":"numeric-ids-and-zero-order","bounds":"raw","selection":"sorted-score-width-block-candidates","index_validation":"full-forward-unique-terms-and-conservative-raw-bounds-v1","query_validation":"whole-file-cap32-max128-unique-prepared-terms-u64-bound-before-retrieval","output_arithmetic":"uint64-widened-inside-query-timer"}}"#, CONTRACT, ARITHMETIC);
    let mut pending = OpenOptions::new().write(true).create_new(true).open(directory.join("runtime.pending"))?;
    writeln!(pending,"{{\"config\":{},\"behavior\":{}}}",config,behavior)?;pending.sync_all()?;
    ensure!(!directory.join("runtime.json").exists(),"Duplicate runtime report");
    fs::rename(directory.join("runtime.pending"),directory.join("runtime.json"))?;
    let start = Instant::now();
    while start.elapsed() < timeout {
        ensure!(!directory.join("reject").exists(),"Runtime gate rejected");
        if directory.join("continue").exists() { return Ok(()); }
        thread::sleep(Duration::from_millis(25));
    }
    anyhow::bail!("Runtime gate timed out")
}

fn validation_report(a: &Args, checked: &VerifiedCollection, queries: &[GuardedQuery]) -> Result<()> {
    let mut f=OpenOptions::new().write(true).create_new(true).open(&a.validation)?;
    writeln!(f,r#"{{"docs":{},"vocab":{},"nnz":{},"block_size":{},"score_bits":{},"full_forward_unique":true,"raw_bounds_conservative":true,"query_count":{},"max_query_terms":{},"max_conservative_score_bound":{}}}"#,
        checked.stats().docs,checked.stats().vocab,checked.stats().nnz,checked.stats().block_size,
        std::mem::size_of::<Score>()*8,queries.len(),queries.iter().map(|q|q.terms().len()).max().unwrap_or(0),queries.iter().map(|q|q.bound()).max().unwrap_or(0))?;
    Ok(())
}
fn memory_report(a: &Args, checked: &VerifiedCollection) -> Result<()> {
    let status=fs::read_to_string("/proc/self/status")?;
    let number=|name:&str| -> Result<u64> { Ok(status.lines().find(|s|s.starts_with(name)).ok_or_else(||anyhow::anyhow!("Missing process memory field"))?.split_whitespace().nth(1).unwrap().parse()?) };
    let mut f=OpenOptions::new().write(true).create_new(true).open(&a.memory_report)?;
    writeln!(f,r#"{{"score_bytes":{},"candidate_tuple_bytes":{},"heap_entry_bytes":{},"bound_buffer_elements":{},"document_score_buffer_elements":{},"process_vm_hwm_kib":{},"process_vm_rss_kib":{},"process_vm_swap_kib":{},"qualification":"Rust-layout sizes and post-run whole-process memory;not a complete allocation census or measured-region peak"}}"#,
        std::mem::size_of::<Score>(),std::mem::size_of::<(usize,Score)>(),std::mem::size_of::<bmp::query::topk_heap::Entry<Score>>(),
        checked.stats().docs/checked.stats().block_size+usize::from(checked.stats().docs%checked.stats().block_size!=0),checked.stats().block_size,
        number("VmHWM:")?,number("VmRSS:")?,number("VmSwap:")?)?;
    Ok(())
}

// Query text is decoded and validated once, outside all query timing. The exported
// PISA representation already contains cap32 integer weights as term repetitions.
// A deterministic lexical order replaces the parent's randomized HashMap order;
// this changes traversal order only, and is disclosed and checked against the oracle.
fn prepared_terms(text: &str) -> Result<GuardedQuery> {
    let mut terms = BTreeMap::<String, u32>::new();
    for token in text.split_whitespace() {
        let n = terms.entry(token.to_owned()).or_insert(0);
        *n = n.checked_add(1).ok_or_else(|| anyhow::anyhow!("Query count overflow"))?;
        ensure!(*n <= 32, "Export must already contain cap32 weights");
    }
    ensure!(terms.len() <= 128, "Query exceeds shared supported term domain");
    GuardedQuery::new(terms.into_iter().collect())
}

fn query(index: &Index, forward: &BlockForwardIndex, context: &CanonicalContext,
         numeric_ids: &[u64], terms: &GuardedQuery, _checked: &VerifiedCollection, k: usize) -> (u128, Vec<(u64, u64)>) {
    let start = Instant::now();
    let cursors = terms.terms().iter().flat_map(|(token, weight)| index.get_cursor(token, *weight)).collect();
    let mut heaps = canonical_prepared(vec![cursors], forward, k, context, false);
    let output = heaps[0].to_sorted_vec().into_iter()
        .map(|r| (numeric_ids[r.doc_id.0 as usize], u64::from(r.score))).collect();
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
    let context = CanonicalContext::new(index.documents());
    let numeric_ids = index.documents().iter().map(|s| s.parse::<u64>()).collect::<std::result::Result<Vec<_>, _>>()?;
    let prepared = queries.iter().map(|(_, text)| prepared_terms(text)).collect::<Result<Vec<_>>>()?;
    let checked = VerifiedCollection::new(&index, &forward, &context)?;
    ensure!(checked.stats().vocab == a.expected_vocab && checked.stats().nnz == a.expected_nnz,"Index domain count mismatch");
    validation_report(&a, &checked, &prepared)?;
    let mut samples = OpenOptions::new().write(true).create_new(true).open(&a.samples)?;
    let mut results = OpenOptions::new().write(true).create_new(true).open(&a.results)?;
    let mut monitoring = OpenOptions::new().write(true).create_new(true).open(&a.monitoring)?;
    writeln!(samples, "phase\tround\tquery_index\torder\tlayout\tquery_id\telapsed_ns")?;
    writeln!(results, "phase\tround\tlayout\tquery_id\tdocument_id\trank\tscore")?;
    writeln!(monitoring, "phase\tround\tposition\tcpu_jiffies\tload_1_5_15\tfrequency_khz")?;
    gate(&a, &checked, &prepared)?;
    for (phase, rounds) in [("warmup", a.warmups), ("measured", a.repeats)] {
        for round in 0..rounds {
            writeln!(monitoring, "{}\t{}\tbefore\t{}", phase, round, monitor(a.cpu))?;
            let mut saved = Vec::new();
            for (qi, (id, _text)) in queries.iter().enumerate() {
                let (ns, output) = query(&index, &forward, &context, &numeric_ids, &prepared[qi], &checked, a.k);
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
    memory_report(&a, &checked)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{gate_timeout,prepared_terms};
    use std::time::Duration;
    #[test] fn prepared_query_preserves_weight_tokens_and_lexical_order() {
        assert_eq!(prepared_terms("z a z").unwrap().terms(),&[("a".to_string(),1),("z".to_string(),2)]);
        assert!(prepared_terms("").unwrap().terms().is_empty());
        assert_eq!(prepared_terms(&vec!["a";32].join(" ")).unwrap().terms()[0].1,32);
        assert!(prepared_terms(&vec!["a";33].join(" ")).is_err());
        assert!(prepared_terms(&(0..129).map(|i|i.to_string()).collect::<Vec<_>>().join(" ")).is_err());
    }
    #[test] fn accepts_runner_decimal_timeout() {
        assert_eq!(gate_timeout("185.0").unwrap(),Duration::from_secs(185));
        assert_eq!(gate_timeout("65").unwrap(),Duration::from_secs(65));
        assert_eq!(gate_timeout("0.5").unwrap(),Duration::from_millis(500));
    }
    #[test] fn rejects_invalid_or_unbounded_timeout() {
        for v in ["NaN","inf","-1","0","86401","invalid"] { assert!(gate_timeout(v).is_err()); }
    }
}
