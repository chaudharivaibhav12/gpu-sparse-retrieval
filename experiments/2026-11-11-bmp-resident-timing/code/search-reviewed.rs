use crate::index::forward_index::{block_score, BlockForwardIndex};
use crate::index::posting_list::PostingListIterator;
use crate::query::cursor::{DocId, RangeMaxScore, RangeMaxScoreCursor};
use crate::query::live_block;
use crate::query::topk_heap::TopKHeap;
use crate::util::progress_bar;
use std::collections::HashSet;
use std::time::Instant;

pub fn b_search(
    queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    k: usize, alpha: f32, terms_r: f32,
) -> Vec<TopKHeap<u64>> {
    b_search_verbose(queries, forward_index, k, alpha, terms_r, true)
}

pub fn b_search_verbose(
    queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    k: usize, alpha: f32, terms_r: f32, verbose: bool,
) -> Vec<TopKHeap<u64>> {
    search_with_ids(queries, forward_index, k, alpha, terms_r, verbose, None, &[])
}

/// Project-contract CLI entry point. Native API entry points remain callable.
pub fn b_search_canonical(
    queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    document_names: &[String], k: usize, alpha: f32, terms_r: f32,
) -> Vec<TopKHeap<u64>> {
    assert!(alpha == 1.0 && terms_r == 1.0, "Canonical exact mode requires alpha=beta=1");
    let context = CanonicalContext::new(document_names);
    canonical_prepared(queries, forward_index, k, &context, true)
}

/// Resident corpus metadata; identical eligibility and numeric tie policy.
pub struct CanonicalContext {
    ids: Vec<u64>,
    zero_order: Vec<usize>,
}

impl CanonicalContext {
    pub fn new(document_names: &[String]) -> Self {
        let ids: Vec<u64> = document_names.iter().map(|s|
            s.parse::<u64>().expect("Canonical exact mode requires numeric passage IDs")).collect();
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len(), "Duplicate passage ID");
        let mut zero_order: Vec<usize> = (0..ids.len()).collect();
        zero_order.sort_unstable_by_key(|&d| ids[d]);
        Self { ids, zero_order }
    }
}

/// Exact alpha=beta=1 only; the caller owns resident corpus metadata.
pub fn canonical_prepared(
    queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    k: usize, context: &CanonicalContext, verbose: bool,
) -> Vec<TopKHeap<u64>> {
    search_with_ids(queries, forward_index, k, 1.0, 1.0, verbose,
                    Some(&context.ids), &context.zero_order)
}

fn search_with_ids(
    mut queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    k: usize, alpha: f32, terms_r: f32, verbose: bool, numeric_ids: Option<&[u64]>, zero_order: &[usize],
) -> Vec<TopKHeap<u64>> {
    let mut results = Vec::new();
    let progress = if verbose { Some(progress_bar("Forward index-based search", queries.len())) } else { None };
    let limit = numeric_ids.map_or(k, |ids| k.min(ids.len()));
    let mut search_elapsed = 0;
    for query in queries.iter_mut() {
        query.sort_by(|a, b| b.term_weight().cmp(&a.term_weight()));
        query.truncate((query.len() as f32 * terms_r).ceil() as usize);
        let mut query_vec: Vec<_> = query.iter().map(|p| (p.term_id() as u16, p.term_weight())).collect();
        query_vec.sort_by_key(|p| p.0);
        let seed_threshold = query.iter().map(|p| p.kth(limit) as u64 * p.term_weight() as u64).max().unwrap_or(0);
        let mut raw = Vec::new(); let mut raw_weights = Vec::new();
        let mut compressed = Vec::new(); let mut compressed_weights = Vec::new();
        for posting in query.iter() {
            match posting.range_max_scores() {
                RangeMaxScore::Raw(ranges) => { raw.push(ranges); raw_weights.push(posting.term_weight()); }
                RangeMaxScore::Compressed(ranges) => { compressed.push(ranges); compressed_weights.push(posting.term_weight()); }
            }
        }
        let start = Instant::now();
        let mut bounds = live_block::compute_upper_bounds_raw(&raw, &raw_weights, forward_index.data.len());
        if !compressed.is_empty() {
            let extra = live_block::compute_upper_bounds(&compressed, &compressed_weights, forward_index.data.len());
            for (bound, addition) in bounds.iter_mut().zip(extra) { *bound += addition; }
        }
        let mut candidates: Vec<_> = bounds.into_iter().enumerate()
            .filter(|&(_, bound)| bound > 0 && bound >= seed_threshold).collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut topk = TopKHeap::new(limit);
        if limit > 0 {
            for (block, bound) in candidates {
                let can_stop = if alpha == 1.0 { topk.threshold() > bound }
                    else { topk.threshold() as f64 > bound as f64 * alpha as f64 };
                if topk.len() == limit && can_stop { break; }
                let scores = block_score(&query_vec, &forward_index.data[block], forward_index.block_size);
                for (offset, score) in scores.into_iter().enumerate() {
                    let doc = block * forward_index.block_size + offset;
                    if score == 0 { continue; }
                    let tie_id = numeric_ids.map_or(doc as u64, |ids| ids[doc]);
                    topk.insert_with_tie(DocId(doc as u32), score, tie_id);
                }
            }
            if let Some(ids) = numeric_ids {
                if topk.len() < limit {
                    let present: HashSet<_> = topk.clone().to_sorted_vec().iter().map(|e| e.doc_id.0 as usize).collect();
                    for &doc in zero_order {
                        if !present.contains(&doc) { topk.insert_with_tie(DocId(doc as u32), 0, ids[doc]); }
                        if topk.len() == limit { break; }
                    }
                }
            }
        }
        search_elapsed += start.elapsed().as_micros();
        results.push(topk);
        if let Some(ref progress) = progress { progress.inc(1); }
    }
    if let Some(progress) = progress { progress.finish(); }
    if verbose && !results.is_empty() { eprintln!("search_elapsed = {}", search_elapsed / results.len() as u128); }
    results
}

#[cfg(test)]
mod resident_context_tests {
    use super::CanonicalContext;

    #[test]
    fn resident_zero_order_uses_numeric_ids() {
        let context = CanonicalContext::new(&["10".into(), "2".into(), "1".into()]);
        assert_eq!(context.ids, vec![10, 2, 1]);
        assert_eq!(context.zero_order, vec![2, 1, 0]);
    }

    #[test]
    fn resident_empty_context_is_valid() {
        let context = CanonicalContext::new(&[]);
        assert!(context.ids.is_empty() && context.zero_order.is_empty());
    }

    #[test]
    #[should_panic(expected = "Duplicate passage ID")]
    fn resident_duplicate_numeric_ids_rejected() {
        CanonicalContext::new(&["1".into(), "01".into()]);
    }
}
