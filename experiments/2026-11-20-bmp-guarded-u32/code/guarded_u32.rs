//! Guarded U32 query-time arithmetic. Serialized index and U64 IDs are unchanged.
use anyhow::{ensure, Result};
use crate::index::inverted_index::Index;
use crate::index::forward_index::BlockForwardIndex;
use crate::index::posting_list::PostingListIterator;
use crate::query::cursor::{DocId, RangeMaxScore, RangeMaxScoreCursor};
use crate::query::topk_heap::TopKHeap;
use crate::search::CanonicalContext;
use crate::util::progress_bar;
use std::collections::HashSet;
use std::time::Instant;

pub fn checked_score_bound(sum_weights: u64, multiplicity: u64) -> Result<u64> {
    ensure!(multiplicity == 1, "Require unique document terms");
    let bound = sum_weights.checked_mul(255).and_then(|v| v.checked_mul(multiplicity))
        .ok_or_else(|| anyhow::anyhow!("Score bound arithmetic overflow"))?;
    ensure!(bound < u32::MAX as u64, "Score bound outside guarded U32 domain");
    Ok(bound)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardedQuery { terms: Vec<(String, u32)>, bound: u64 }
impl GuardedQuery {
    pub fn new(terms: Vec<(String, u32)>) -> Result<Self> {
        ensure!(terms.len() <= 128, "More than 128 prepared terms");
        let mut seen = HashSet::new();
        let mut sum = 0u64;
        for (term, weight) in &terms {
            ensure!(!term.is_empty() && !term.chars().any(char::is_whitespace), "Invalid prepared token");
            ensure!(seen.insert(term), "Duplicate prepared term");
            ensure!(*weight <= 32, "Prepared weight above cap32");
            sum = sum.checked_add(*weight as u64).ok_or_else(|| anyhow::anyhow!("Weight sum overflow"))?;
        }
        let bound = checked_score_bound(sum, 1)?;
        Ok(Self { terms, bound })
    }
    pub fn terms(&self) -> &[(String, u32)] { &self.terms }
    pub fn bound(&self) -> u64 { self.bound }
}

#[derive(Debug)]
pub struct DomainStats { pub docs: usize, pub vocab: usize, pub nnz: u64, pub block_size: usize }

/// All fields are private: only a completed validation can create this token.
/// The full scan and query guards run before the runtime gate and all timing.
pub struct VerifiedCollection<'a> {
    index: &'a Index,
    forward: &'a BlockForwardIndex,
    context: &'a CanonicalContext,
    stats: DomainStats,
}
impl<'a> VerifiedCollection<'a> {
    pub fn new(index: &'a Index, forward: &'a BlockForwardIndex, context: &'a CanonicalContext) -> Result<Self> {
        let docs = index.documents().len();
        let block_size = forward.block_size;
        ensure!((1..=256).contains(&block_size), "Within-block offsets require block size 1..256");
        ensure!(docs <= u32::MAX as usize && index.num_documents() == docs, "Invalid document count");
        ensure!(forward.data.len() == docs / block_size + usize::from(docs % block_size != 0), "Invalid block count");
        ensure!(context.project_ids().len() == docs && context.project_zero_order().len() == docs, "Context shape mismatch");
        for (id, name) in context.project_ids().iter().zip(index.documents()) {
            ensure!(*id == name.parse::<u64>()?, "Context original-ID mismatch");
        }
        let vocab = index.posting_lists().len();
        ensure!(vocab <= 65536, "Term IDs do not fit the existing serialized format");
        let cursors: Vec<_> = index.posting_lists().iter().enumerate().map(|(t,p)| p.iter(t as u32, 1)).collect();
        let mut raw_rows = Vec::new();
        for cursor in &cursors {
            match cursor.range_max_scores() {
                RangeMaxScore::Raw(row) => {
                    ensure!(row.len() == forward.data.len(), "Invalid raw bound row length");
                    raw_rows.push(row);
                },
                RangeMaxScore::Compressed(_) => anyhow::bail!("This study requires raw bounds"),
            }
        }
        let mut nnz = 0u64;
        for (block, terms) in forward.data.iter().enumerate() {
            let remaining = (docs - block * block_size).min(block_size);
            let mut previous = None;
            for (term, postings) in terms {
                ensure!((*term as usize) < vocab, "Unknown forward term ID");
                ensure!(previous.map_or(true, |p| p < *term), "Duplicate/unsorted forward term");
                previous = Some(*term);
                let mut offsets = [0u64; 4];
                let mut max_impact = 0u8;
                for (offset, impact) in postings {
                    let d = *offset as usize;
                    ensure!(d < remaining, "Forward posting outside real document/tail block");
                    let bit = 1u64 << (d % 64);
                    ensure!(offsets[d / 64] & bit == 0, "Duplicate term/document contribution");
                    offsets[d / 64] |= bit;
                    max_impact = max_impact.max(*impact);
                    nnz = nnz.checked_add(1).ok_or_else(|| anyhow::anyhow!("Posting count overflow"))?;
                }
                ensure!(raw_rows[*term as usize][block] >= max_impact, "Nonconservative raw bound");
            }
        }
        Ok(Self { index, forward, context, stats: DomainStats { docs, vocab, nnz, block_size } })
    }
    pub fn stats(&self) -> &DomainStats { &self.stats }
    /// Fresh cursor construction remains inside the caller's query timer.
    pub fn search(&self, query: &GuardedQuery, k: usize, verbose: bool) -> Vec<TopKHeap<u32>> {
        let cursors = query.terms().iter().flat_map(|(token, weight)| self.index.get_cursor(token, *weight)).collect();
        search_with_ids(vec![cursors], self.forward, k, 1.0, 1.0, verbose,
                        Some(self.context.project_ids()), self.context.project_zero_order())
    }
}

// The following arithmetic/search bodies are mechanically copied from the pinned
// parent. Only score types/products/callees change. See materialize.py assertions.

#[inline]
fn block_score(
    query: &Vec<(u16, u8)>,
    document: &[(u16, Vec<(u8, u8)>)],
    bsize: usize,
) -> Vec<u32> {
    let mut doc_score = vec![0u32; bsize];

    unsafe {
        let mut term_ptr = document.as_ptr();
        let end = term_ptr.wrapping_offset(document.len() as isize);
        for &(coordinate, value) in query {
            while term_ptr != end && (*term_ptr).0 < coordinate {
                term_ptr = term_ptr.add(1);
            }
            if term_ptr == end {
                break;
            }
            if (*term_ptr).0 == coordinate {
                let mut inner_ptr = (*term_ptr).1.as_ptr();
                let end_inner_ptr = inner_ptr.wrapping_offset((*term_ptr).1.len() as isize);
                while inner_ptr != end_inner_ptr {
                    doc_score[(*inner_ptr).0 as usize] += (value as u32) * ((*inner_ptr).1 as u32);
                    inner_ptr = inner_ptr.add(1);
                }
            }
        }
    }

    doc_score
}

#[inline]
fn compute_upper_bounds_raw(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
) -> Vec<u32> {
    // let vector_len: usize = query_ranges[0].len();
    let mut upper_bounds: Vec<u32> = vec![0; vector_len];

    // Iterate over each vector in scores and add its elements to the result
    for (&vec, &weight) in query_ranges.iter().zip(query_weights.iter()) {
        for (i, &score) in vec.iter().enumerate() {
            let multiplied = score as u32 * weight as u32;
            upper_bounds[i] = upper_bounds[i].saturating_add(multiplied);
        }
    }
    upper_bounds
}

#[inline]
fn compute_upper_bounds(
    query_ranges: &[&[crate::index::posting_list::CompressedBlock]],
    query_weights: &[u8],
    vector_len: usize,
) -> Vec<u32> {
    // let vector_len: usize = query_ranges[0].len();
    let mut upper_bounds: Vec<u32> = vec![0; vector_len];

    // Iterate over each vector in scores and add its elements to the result
    for (&vec, &weight) in query_ranges.iter().zip(query_weights.iter()) {
        for (bid, block) in vec.iter().enumerate() {
            for &(offset, score) in &block.max_scores {
                let multiplied = score as u32 * weight as u32;
                upper_bounds[bid * 256 + offset] =
                    upper_bounds[bid * 256 + offset].saturating_add(multiplied);
            }
        }
    }
    upper_bounds
}

fn search_with_ids(
    mut queries: Vec<Vec<PostingListIterator>>, forward_index: &BlockForwardIndex,
    k: usize, alpha: f32, terms_r: f32, verbose: bool, numeric_ids: Option<&[u64]>, zero_order: &[usize],
) -> Vec<TopKHeap<u32>> {
    let mut results = Vec::new();
    let progress = if verbose { Some(progress_bar("Forward index-based search", queries.len())) } else { None };
    let limit = numeric_ids.map_or(k, |ids| k.min(ids.len()));
    let mut search_elapsed = 0;
    for query in queries.iter_mut() {
        query.sort_by(|a, b| b.term_weight().cmp(&a.term_weight()));
        query.truncate((query.len() as f32 * terms_r).ceil() as usize);
        let mut query_vec: Vec<_> = query.iter().map(|p| (p.term_id() as u16, p.term_weight())).collect();
        query_vec.sort_by_key(|p| p.0);
        let seed_threshold = query.iter().map(|p| p.kth(limit) as u32 * p.term_weight() as u32).max().unwrap_or(0);
        let mut raw = Vec::new(); let mut raw_weights = Vec::new();
        let mut compressed = Vec::new(); let mut compressed_weights = Vec::new();
        for posting in query.iter() {
            match posting.range_max_scores() {
                RangeMaxScore::Raw(ranges) => { raw.push(ranges); raw_weights.push(posting.term_weight()); }
                RangeMaxScore::Compressed(ranges) => { compressed.push(ranges); compressed_weights.push(posting.term_weight()); }
            }
        }
        let start = Instant::now();
        let mut bounds = compute_upper_bounds_raw(&raw, &raw_weights, forward_index.data.len());
        if !compressed.is_empty() {
            let extra = compute_upper_bounds(&compressed, &compressed_weights, forward_index.data.len());
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
mod tests {
    use super::*;
    use crate::index::inverted_index::IndexBuilder;
    use crate::search::canonical_prepared;
    fn fixture(values: &[Vec<u8>], ids: &[u64], b: usize, compressed: bool) -> (Index, BlockForwardIndex) {
        let mut builder = IndexBuilder::new(ids.len(), b);
        for id in ids { builder.insert_document(&id.to_string()); }
        let mut forward = BlockForwardIndex { data: vec![Vec::new(); ids.len()/b+usize::from(ids.len()%b!=0)], block_size: b };
        for (t, row) in values.iter().enumerate() {
            assert_eq!(row.len(), ids.len());
            builder.insert_term(&format!("t{:03}", t), row.iter().enumerate().filter(|(_,v)| **v>0).map(|(d,v)| (d as u32,*v as u32)).collect());
            for (block, slice) in row.chunks(b).enumerate() {
                let postings: Vec<_> = slice.iter().enumerate().filter(|(_,v)| **v>0).map(|(d,v)| (d as u8,*v)).collect();
                if !postings.is_empty() { forward.data[block].push((t as u16, postings)); }
            }
        }
        (builder.build(compressed), forward)
    }
    fn compare(values: &[Vec<u8>], ids: &[u64], weights: &[u32], b: usize, k: usize) {
        let (index,forward) = fixture(values, ids, b, false);
        let ctx = CanonicalContext::new(index.documents());
        let checked = VerifiedCollection::new(&index,&forward,&ctx).unwrap();
        let query = GuardedQuery::new(weights.iter().enumerate().map(|(t,w)| (format!("t{:03}",t),*w)).collect()).unwrap();
        let mut actual = checked.search(&query,k,false);
        let got: Vec<_> = actual[0].to_sorted_vec().iter().map(|e| (ids[e.doc_id.0 as usize], e.score as u64)).collect();
        let cursors = query.terms().iter().flat_map(|(t,w)| index.get_cursor(t,*w)).collect();
        let mut wide = canonical_prepared(vec![cursors],&forward,k,&ctx,false);
        let wide: Vec<_> = wide[0].to_sorted_vec().iter().map(|e| (ids[e.doc_id.0 as usize],e.score)).collect();
        let mut gold: Vec<_> = ids.iter().enumerate().map(|(d,id)| (*id,values.iter().zip(weights).map(|(row,w)| row[d] as u64 * *w as u64).sum::<u64>())).collect();
        gold.sort_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));gold.truncate(k.min(ids.len()));
        assert_eq!(got,gold);assert_eq!(wide,gold);
    }
    #[test] fn query_guard_rejects_unsafe_or_ambiguous_inputs() {
        assert!(GuardedQuery::new(vec![("x".into(),33)]).is_err());
        assert!(GuardedQuery::new((0..129).map(|t|(format!("unknown{}",t),1)).collect()).is_err());
        assert!(GuardedQuery::new(vec![("x".into(),1),("x".into(),1)]).is_err());
        assert!(GuardedQuery::new(vec![("two tokens".into(),1)]).is_err());
        assert!(checked_score_bound(u64::MAX,1).is_err());
        assert!(checked_score_bound(1,2).is_err());
        assert!(checked_score_bound((u32::MAX as u64)/255+1,1).is_err());
        assert_eq!(GuardedQuery::new(vec![]).unwrap().bound(),0);
    }
    #[test] fn maximum_supported_score_and_above_u16() {
        compare(&vec![vec![255];128],&[u64::MAX],&vec![32;128],1,10);
        let q = GuardedQuery::new((0..128).map(|t|(format!("t{:03}",t),32)).collect()).unwrap();
        assert_eq!(q.bound(),1_044_480);compare(&vec![vec![255];10],&[9],&vec![32;10],1,10);
    }
    #[test] fn ties_zero_completion_tail_block_and_depths() {
        let ids = [u64::MAX,4294967297,8,3,1];
        let values = vec![vec![1,0,1,0,1],vec![0,0,0,0,0]];
        for k in [0,1,2,5,10] { compare(&values,&ids,&[7,0],2,k); }
        compare(&[vec![0;5]],&ids,&[32],2,10);
        compare(&[],&[],&[],2,10);
    }
    #[test] fn unknown_query_is_all_zero_not_dropped() {
        let (index,forward) = fixture(&[vec![1,2,3]],&[u64::MAX,9,1],2,false);
        let ctx=CanonicalContext::new(index.documents());let collection=VerifiedCollection::new(&index,&forward,&ctx).unwrap();
        let query=GuardedQuery::new(vec![("missing".into(),32)]).unwrap();let mut r=collection.search(&query,10,false);
        assert_eq!(r[0].to_sorted_vec().iter().map(|e|(e.tie_id,e.score)).collect::<Vec<_>>(),vec![(1,0),(9,0),(u64::MAX,0)]);
    }
    #[test] fn deterministic_generated_cases_match_independent_integer_oracle() {
        for n in [1,7,9,33] {
            let ids: Vec<_> = (0..n).map(|i| 4294967300+(n-i) as u64).collect();
            let values: Vec<_> = (0..8).map(|t| (0..n).map(|d| ((t*17+d*29+t*d)%256) as u8).collect()).collect();
            for k in [1,10,100] { compare(&values,&ids,&[0,1,2,3,4,8,16,32],8,k); }
        }
    }
    #[test] fn bad_forward_uniqueness_tail_and_bounds_rejected_before_search() {
        let (i,mut f)=fixture(&[vec![5,0,2]],&[3,2,1],2,false);let c=CanonicalContext::new(i.documents());
        let original=f.clone();f.data[0][0].1.push((0,5));assert!(VerifiedCollection::new(&i,&f,&c).is_err());
        f=original.clone();f.data[1][0].1.push((1,1));assert!(VerifiedCollection::new(&i,&f,&c).is_err());
        f=original.clone();let t=f.data[0][0].clone();f.data[0].push(t);assert!(VerifiedCollection::new(&i,&f,&c).is_err());
        f=original;f.data[0][0].1[0].1=6;assert!(VerifiedCollection::new(&i,&f,&c).is_err());
    }
    #[test] fn compressed_index_is_explicitly_unsupported_by_this_raw_study() {
        let (i,f)=fixture(&[vec![1,2,3]],&[1,2,3],2,true);let c=CanonicalContext::new(i.documents());assert!(VerifiedCollection::new(&i,&f,&c).is_err());
    }
    #[test] fn raw_and_compressed_private_bound_arithmetic_is_wide_enough() {
        let rows=vec![vec![255u8];128];let refs: Vec<&[u8]>=rows.iter().map(|v|v.as_slice()).collect();
        assert_eq!(compute_upper_bounds_raw(&refs,&vec![32;128],1),vec![1_044_480]);
        let rows: Vec<_>=(0..128).map(|_|vec![crate::index::posting_list::CompressedBlock{max_scores:vec![(0,255)]}]).collect();let refs: Vec<&[crate::index::posting_list::CompressedBlock]>=rows.iter().map(|v|v.as_slice()).collect();
        assert_eq!(compute_upper_bounds(&refs,&vec![32;128],1),vec![1_044_480]);
        assert_eq!(compute_upper_bounds_raw(&[],&[],1)[0]+compute_upper_bounds(&refs,&vec![32;128],1)[0],1_044_480);
    }
}
