// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! The file that keeps the offline RWKV state between app starts
//! (`collection.rwkv-offline`, next to the collection), so that a start
//! replays only the reviews that came after it instead of the whole history.
//!
//! It holds the model state (the engine's warm-up snapshot and cache state)
//! and the identity of the review history it has absorbed. The header ties it
//! to one backend build and one model file: anything else is ignored, and the
//! history is replayed in full as before. `offline.rs` decides when to write
//! and read it; the history check there catches a file that no longer fits
//! the collection.

use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;

use crate::rwkv::RwkvWarmUpSnapshot;

const MAGIC: &[u8] = b"KUMA3RWKVOFFLINE1\n";

pub(super) struct SavedState {
    pub identity: Vec<u8>,
    pub snapshot: RwkvWarmUpSnapshot,
    pub cache_state: Vec<u8>,
}

/// What the file must have been written for: this backend build and this model
/// file (its size and an FNV-1a hash of its bytes).
pub(super) fn header(model_path: &Path) -> io::Result<String> {
    let model = fs::read(model_path)?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in &model {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(format!(
        "{} {} {:016x}",
        crate::version::buildhash(),
        model.len(),
        hash
    ))
}

/// Writes the file whole (to a temporary file, then renamed), so a crash
/// never leaves half a file behind.
pub(super) fn write(
    path: &Path,
    header: &str,
    identity: &[u8],
    snapshot: &RwkvWarmUpSnapshot,
    cache_state: &[u8],
) -> io::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_bytes(&mut out, header.as_bytes());
    put_bytes(&mut out, identity);
    for map in [
        &snapshot.card_states,
        &snapshot.note_states,
        &snapshot.deck_states,
        &snapshot.preset_states,
    ] {
        out.extend_from_slice(&(map.len() as u64).to_le_bytes());
        for (id, state) in map {
            out.extend_from_slice(&id.to_le_bytes());
            put_bytes(&mut out, state);
        }
    }
    match &snapshot.global_state {
        Some(state) => {
            out.push(1);
            put_bytes(&mut out, state);
        }
        None => out.push(0),
    }
    put_bytes(&mut out, cache_state);

    let temporary = path.with_extension("rwkv-offline.tmp");
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&out)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)
}

/// Reads the file written by [write]; an error when it is missing, damaged or
/// was written for another header.
pub(super) fn read(path: &Path, header: &str) -> io::Result<SavedState> {
    let data = fs::read(path)?;
    let mut input = Reader { data: &data, at: 0 };
    if input.take(MAGIC.len())? != MAGIC {
        return Err(invalid("not an offline RWKV state file"));
    }
    if input.bytes()? != header.as_bytes() {
        return Err(invalid("written by another build or for another model"));
    }
    let identity = input.bytes()?.to_vec();
    let mut maps = Vec::with_capacity(4);
    for _ in 0..4 {
        let count = input.u64()? as usize;
        let mut map = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            let id = input.i64()?;
            map.push((id, input.bytes()?.to_vec()));
        }
        maps.push(map);
    }
    let global_state = match input.take(1)?[0] {
        1 => Some(input.bytes()?.to_vec()),
        _ => None,
    };
    let cache_state = input.bytes()?.to_vec();
    if input.at != data.len() {
        return Err(invalid("trailing data"));
    }
    let preset_states = maps.pop().unwrap_or_default();
    let deck_states = maps.pop().unwrap_or_default();
    let note_states = maps.pop().unwrap_or_default();
    let card_states = maps.pop().unwrap_or_default();
    Ok(SavedState {
        identity,
        snapshot: RwkvWarmUpSnapshot {
            card_states,
            note_states,
            deck_states,
            preset_states,
            global_state,
        },
        cache_state,
    })
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| invalid("truncated"))?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let len = usize::try_from(self.u64()?).map_err(|_| invalid("length"))?;
        self.take(len)
    }
}
