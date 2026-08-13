// Stimulus sources for the demos.
//
// Each module holds only the data and its encoding — no network setup, no
// reporting. Everything is generated procedurally from `--seed`, so no asset is
// committed and CI needs no network.
//
// Sparsey takes input as a plain sparse list of **active cell indices** within an
// input region (`&[u32]`), which is what these produce. Note the asymmetry recorded
// in `doc/Conformance.md` R5: the input is a flat sparse set, but a MAC's *output*
// is a grouped code — one winning cell per CM — so the two are not interchangeable
// and chaining one into another would need flattening.

pub mod patterns;
pub mod sequences;
