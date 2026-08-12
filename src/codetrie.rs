//! Storage of the codes a MAC has learned.
//!
//! Ported from `CodeTrie.java`. The Java uses a trie keyed by the per-CM winner
//! indices for fast prefix lookup and overlap analysis. For M1 we keep a simpler
//! map from the full code (the `Q` winner indices) to the frame it first appeared;
//! this supports `insert`/`contains`/`first_frame` which is all the core needs. A
//! true trie can replace this later without changing callers (see
//! `doc/PortNotes.md`).

use std::collections::HashMap;

/// A learned code = one winning cell index per CM (`Q` entries).
pub type Code = Vec<u32>;

/// The set of codes a MAC has learned, each with its first-seen global frame.
#[derive(Clone, Debug, Default)]
pub struct CodeTrie {
    codes: HashMap<Code, i64>,
}

impl CodeTrie {
    /// Empty store.
    pub fn new() -> Self {
        CodeTrie {
            codes: HashMap::new(),
        }
    }

    /// Record `code` as seen at `frame`. Returns `true` if it was new.
    pub fn insert(&mut self, code: Code, frame: i64) -> bool {
        use std::collections::hash_map::Entry;
        match self.codes.entry(code) {
            Entry::Occupied(_) => false,
            Entry::Vacant(v) => {
                v.insert(frame);
                true
            }
        }
    }

    /// Has this code been learned before?
    pub fn contains(&self, code: &[u32]) -> bool {
        self.codes.contains_key(code)
    }

    /// The frame a code first appeared, if learned.
    pub fn first_frame(&self, code: &[u32]) -> Option<i64> {
        self.codes.get(code).copied()
    }

    /// Number of distinct learned codes.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether no codes have been learned.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}
