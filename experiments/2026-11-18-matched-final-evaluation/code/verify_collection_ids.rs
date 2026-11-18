//! Preparation-only full original-ID sequence check; no query scoring or timing.
use anyhow::{ensure, Result};
use std::{convert::TryInto, fs::File, io::{BufReader, Read}, path::PathBuf};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(args.len() == 3, "Usage: verify_collection_ids INDEX.bmp SOURCE.gsp");
    let (index, _) = bmp::index::from_file(&PathBuf::from(&args[1]))?;
    let mut input = BufReader::with_capacity(1 << 20, File::open(&args[2])?);
    let mut header = [0u8; 24]; input.read_exact(&mut header)?;
    ensure!(&header[..8] == b"GSPIDX01", "Unexpected GSP magic");
    let documents = u32::from_le_bytes(header[8..12].try_into()?) as usize;
    let vocab = u32::from_le_bytes(header[12..16].try_into()?);
    let nonzeros = u64::from_le_bytes(header[16..24].try_into()?);
    ensure!(documents == index.documents().len(), "Collection size differs");
    for (position, document) in index.documents().iter().enumerate() {
        let mut bytes = [0u8; 8]; input.read_exact(&mut bytes)?;
        ensure!(document.parse::<u64>()? == u64::from_le_bytes(bytes),
                "Original passage ID differs at internal position {}", position);
    }
    println!("{{\"documents\":{},\"ordered_original_ids_equal\":true,\"gsp_vocab\":{},\"gsp_nnz\":{}}}", documents, vocab, nonzeros);
    Ok(())
}
