//! Typed arena indices.
//!
//! Every entity in a [`crate::net::SparseyNet`] is stored in a flat `Vec` and
//! referenced by one of these newtype indices instead of a pointer. This replaces
//! the Java version's `owning*` back-pointers and `Synapse.targetNeuron` object
//! references, breaking the reference cycles that would otherwise fight Rust's
//! ownership model.
//!
//! The indices are deliberately *not* interchangeable (a `MacId` cannot be used
//! where a `CmId` is expected), which catches a whole class of mix-ups at compile
//! time. Each is a transparent wrapper over a `usize`.

/// Generates a `#[repr(transparent)]` newtype index over `usize` with the common
/// helper methods and trait impls.
macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(transparent)]
        pub struct $name(pub usize);

        impl $name {
            /// The raw index value.
            #[inline]
            pub const fn index(self) -> usize {
                self.0
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(v: usize) -> Self {
                $name(v)
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(v: $name) -> usize {
                v.0
            }
        }
    };
}

define_id!(
    /// Index of a region within [`crate::net::SparseyNet::regions`].
    RegionId
);
define_id!(
    /// Index of a macrocolumn within [`crate::net::SparseyNet::macs`].
    MacId
);
define_id!(
    /// Index of a competitive module (minicolumn) within [`crate::net::SparseyNet::cms`].
    CmId
);
define_id!(
    /// Index of a neuron (input or internal cell) within [`crate::net::SparseyNet::neurons`].
    NeuronId
);
define_id!(
    /// Aperture index — **reserved, and currently unused**.
    ///
    /// The Java's `Aperture` (a leaf region's window onto the raw input) has no arena
    /// of its own here: an input region's feature cells are ordinary [`NeuronId`]s in
    /// [`crate::net::SparseyNet::neurons`]. Kept because `doc/Architecture.md` maps
    /// the Java object model onto these ids, and a gap reads as an oversight.
    ApertureId
);
define_id!(
    /// Index of an efferent bundle within [`crate::net::SparseyNet::efferent_bundles`].
    EfferentBundleId
);
define_id!(
    /// Sub-efferent bundle index — **reserved, and currently unused**.
    ///
    /// The Java splits a bundle per target MAC; this port keeps one flat
    /// [`crate::bundle::EfferentBundle`] per (source neuron, link) and does not
    /// sub-divide it. See `doc/Divergences.md`.
    SubBundleId
);
define_id!(
    /// Index of a synapse within its owning bundle's
    /// [`crate::bundle::EfferentBundle::synapses`].
    ///
    /// Note this is bundle-local, not a global arena index: unlike neurons or MACs,
    /// synapses are nested inside the bundle that owns them rather than held in one
    /// flat `Vec` on [`crate::net::SparseyNet`].
    SynapseId
);
define_id!(
    /// Index of an inter-region link within [`crate::net::SparseyNet::links`].
    LinkId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_and_is_transparent() {
        let m = MacId::from(7usize);
        assert_eq!(m.index(), 7);
        assert_eq!(usize::from(m), 7);
        assert_eq!(std::mem::size_of::<MacId>(), std::mem::size_of::<usize>());
    }

    #[test]
    fn distinct_id_types_are_comparable_within_type() {
        assert_eq!(CmId(3), CmId(3));
        assert_ne!(CmId(3), CmId(4));
    }
}
