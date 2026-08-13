// Episodes — multi-frame sequences over the pattern alphabet.
//
// **This is the one place the cross-repo contract does not line up, and the reason
// is worth stating rather than hiding.** The other two ports answer the temporal
// cell of the demo matrix with next-symbol *prediction*. Sparsey cannot: it has
// temporal context (H and D links read the previous frame — `use_previous_active =
// syn_type != SynapseType::U` in `src/net/build.rs`) but no path from a code back
// down to input features. `src/net/frame.rs` records that faithful D-signal
// regeneration of the input region is a documented follow-on, and that "for nets
// without downward (D) links this behaves like recognition".
//
// So the temporal demo here is sequence *recognition*: learn episodes, then present
// sequences and report familiarity. Same family, honestly different task. Making it
// prediction would mean implementing D-replay-to-L0 in `src/`, which is algorithm
// work rather than demo work.

use crate::support::env::patterns::PatternBook;
use crate::support::rng::Rng;

/// A set of fixed-length episodes over a shared alphabet of frame patterns.
pub struct EpisodeSet {
    episodes: Vec<Vec<usize>>,
    length: usize,
}

impl EpisodeSet {
    /// `count` episodes of `length` frames each, drawn from `alphabet` patterns.
    ///
    /// Episodes are built to **share their frames** — each is a different ordering
    /// over a small alphabet — so no episode can be recognised from any single
    /// frame. That is what makes this a test of temporal context rather than of
    /// spatial memory: if every episode used its own private frames, recognising
    /// one frame would identify the episode and the sequence would be irrelevant.
    pub fn generate(count: usize, length: usize, alphabet: usize, rng: &mut Rng) -> Self {
        assert!(alphabet >= length, "an episode cannot use more frames than exist");
        let mut episodes = Vec::with_capacity(count);
        for _ in 0..count {
            let mut frames: Vec<usize> = rng
                .sample_sorted(alphabet, length)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            // Shuffle, so two episodes drawing the same frame set differ only in
            // their order — the hardest case, and the one that isolates the
            // temporal claim completely.
            for i in (1..frames.len()).rev() {
                frames.swap(i, rng.below(i + 1));
            }
            episodes.push(frames);
        }
        EpisodeSet { episodes, length }
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn get(&self, i: usize) -> &[usize] {
        &self.episodes[i]
    }

    /// An episode that is *not* in the set: the same frames in an order none of the
    /// stored episodes uses.
    ///
    /// This is the control a novelty claim needs. Comparing a stored episode against
    /// one built from unseen *frames* would measure spatial novelty, which any
    /// content-addressable memory detects; comparing it against unseen *order* over
    /// seen frames is the temporal question.
    pub fn novel_ordering(&self, rng: &mut Rng) -> Vec<usize> {
        for _ in 0..64 {
            let base = rng.below(self.episodes.len());
            let mut candidate = self.episodes[base].clone();
            for i in (1..candidate.len()).rev() {
                candidate.swap(i, rng.below(i + 1));
            }
            if !self.episodes.contains(&candidate) {
                return candidate;
            }
        }
        // Fall back rather than loop forever: with a short episode over a small
        // alphabet every ordering can already be taken.
        self.episodes[0].clone()
    }

    /// Frames used by at least one episode.
    pub fn frames_used(&self) -> Vec<usize> {
        let mut all: Vec<usize> = self.episodes.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        all
    }
}

/// Present one episode frame by frame, returning the input for each.
pub fn episode_inputs<'a>(book: &'a PatternBook, episode: &[usize]) -> Vec<&'a [u32]> {
    episode.iter().map(|&f| book.get(f)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::rng::STREAM_ENV;

    #[test]
    fn episodes_are_the_requested_shape() {
        let mut rng = Rng::stream(1, STREAM_ENV);
        let set = EpisodeSet::generate(6, 4, 8, &mut rng);
        assert_eq!(set.len(), 6);
        for i in 0..set.len() {
            assert_eq!(set.get(i).len(), 4);
            let mut sorted = set.get(i).to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 4, "an episode repeated a frame");
            assert!(sorted.iter().all(|&f| f < 8));
        }
    }

    #[test]
    fn episodes_share_frames_so_no_single_frame_identifies_one() {
        let mut rng = Rng::stream(2, STREAM_ENV);
        let set = EpisodeSet::generate(8, 4, 6, &mut rng);
        // With 8 episodes of 4 frames over an alphabet of 6, frames must be reused.
        assert!(set.frames_used().len() <= 6);

        let mut shared = 0;
        for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                if set.get(i).iter().any(|f| set.get(j).contains(f)) {
                    shared += 1;
                }
            }
        }
        assert!(shared > 0, "no two episodes share a frame — the task is spatial, not temporal");
    }

    #[test]
    fn a_novel_ordering_is_not_one_of_the_stored_episodes() {
        let mut rng = Rng::stream(3, STREAM_ENV);
        let set = EpisodeSet::generate(4, 5, 10, &mut rng);
        for _ in 0..50 {
            let novel = set.novel_ordering(&mut rng);
            assert_eq!(novel.len(), 5);
            assert!(
                !(0..set.len()).any(|i| set.get(i) == novel.as_slice()),
                "novel_ordering returned a stored episode: {novel:?}"
            );
        }
    }

    #[test]
    fn a_novel_ordering_reuses_only_seen_frames() {
        let mut rng = Rng::stream(4, STREAM_ENV);
        let set = EpisodeSet::generate(4, 4, 8, &mut rng);
        let used = set.frames_used();
        // The whole point: novelty must be in the order, not in the frames, or the
        // measurement is about spatial novelty instead.
        for _ in 0..50 {
            assert!(set.novel_ordering(&mut rng).iter().all(|f| used.contains(f)));
        }
    }

    #[test]
    #[should_panic(expected = "cannot use more frames than exist")]
    fn an_episode_longer_than_the_alphabet_is_rejected() {
        let mut rng = Rng::stream(5, STREAM_ENV);
        EpisodeSet::generate(2, 8, 4, &mut rng);
    }
}
