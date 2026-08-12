//! Statistics-recording seam.
//!
//! In SparseyCore the stats layer (`TraceStat`, `CodeStat`, `RecordMoments`, …) is
//! called ~90× from inside the core hot loop, so it cannot simply be deleted. Here
//! it is abstracted behind a [`Recorder`] trait with a zero-overhead no-op default
//! ([`NullRecorder`]). The core threads a `&mut dyn Recorder` through its per-frame
//! methods; real recording (Excel traces, code stats, images) can be added later by
//! implementing this trait without touching the core (see `doc/PortNotes.md`).

use crate::ids::{MacId, RegionId};

/// Hooks the core calls during a frame. All methods default to no-ops.
#[allow(unused_variables)]
pub trait Recorder {
    /// A MAC selected a code (its winners' per-CM indices) at `frame`.
    fn on_code_selected(&mut self, region: RegionId, mac: MacId, code: &[u32], g: f32, frame: i64) {}

    /// A MAC activated (or remained active) this frame.
    fn on_mac_active(&mut self, region: RegionId, mac: MacId, frame: i64) {}

    /// A frame finished.
    fn on_frame_end(&mut self, frame: i64) {}
}

/// The default recorder: records nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullRecorder;

impl Recorder for NullRecorder {}
