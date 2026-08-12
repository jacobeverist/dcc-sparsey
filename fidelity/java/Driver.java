// SparseyCore fidelity golden-vector driver — reference output for the dcc_sparsey Rust port.
//
// Declared in `package SparseyCore;` (Java package membership is by declaration, not
// file location) so it can reach the protected frame-loop methods and read winners
// directly. Compile it ALONGSIDE the SparseyCore sources; the class lands in the
// SparseyCore package and needs no imports of core classes. See ../README.md.
//
// REQUIRES JDK 9+ (SparseyCore uses java.util.List.of()).
//
// Status (verified on JDK 26 / Temurin): SparseyCore + this driver COMPILE headless,
// and the driver RUNS through COM+NDF parsing → setSupervisedMode → into region
// construction. Two pieces remain to reach a green run (see ../README.md):
//   1. Complete the NDF — region/Mac/CM construction reads a large keyset that
//      SparseyCore ships no sample for (Region_Intrinsic_D, Sigmoid_*, Tiling_*,
//      Backoff_* policy strings, U_* link params, MU_*, …). Fill until it constructs.
//   2. Implement input injection (the empty injectInputs/presentFrame stubs) via an
//      EpisodeContainer populated with the fixed INPUTS.
// Note: the Network constructor builds the regions itself (no separate build-order
// calls needed), and ML/max-V can be forced with net.setUseMaxLikeWinSelMethod(true).
//
// It prints a JSON array of {"phase","features","code"} to stdout; the recognition
// frames are the ones the Rust behavioral test consumes (learning is probabilistic in
// SparseyCore — only recognition with Use_ML_Recog:true is deterministic max-V).
//
// Usage: java -cp <out> SparseyCore.Driver <com.json> <ndf-dir>

package SparseyCore;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Driver {
    // The input alphabet — MUST match INPUTS in sparsey/tests/fidelity_behavioral.rs
    // (2x2 grid => feature-cell indices 0..4, row-major).
    static final int[][] INPUTS = {{0, 1}, {2, 3}, {0, 3}, {1, 2}};
    static final long SEED = 42;

    public static void main(String[] args) throws Exception {
        String comPath = args.length > 0 ? args[0] : "config/m1.com.json";

        // Deterministic RNG (Utility.SetRandomSeed no-ops on negative seeds).  [verified]
        Utility.SetRandomSeed(SEED);

        // Build the network from the COM → NDF descriptors.  [verified: Network.java:188]
        DescriptorFile comDF = new DescriptorFile(new File(comPath));
        comDF.parseFile();                                  // UNVERIFIED: NDF/COM JSON grammar
        Network net = new Network(comDF, "train");

        // NOTE (verified): the `Network` constructor above already builds the regions,
        // so no separate build-order calls are needed. `buildNetwork()` is retained
        // below only as a reference to the individual public entry points.

        // Feed the fixed input.  UNVERIFIED: attach an EpisodeContainer whose frames are
        // INPUTS. The lowest-risk route (agent-recommended) is to construct an
        // EpisodeContainer for the input region and populate its protected
        // `theEpisodes : List<ArrayList<ArrayList<Integer>>>` = [episode][frame][idx]
        // directly, one single-frame episode per input, then attach it to the input
        // region (getCurrEpisodeContainer / setter). Implement against the actual
        // InputRegion/EpisodeContainer API when compiling on JDK 9+.
        injectInputs(net);

        StringBuilder json = new StringBuilder();
        json.append("[\n");
        boolean first = true;

        // --- Learn phase (probabilistic in SparseyCore; recorded for completeness) ---
        first = runPhase(net, "learn", json, first, /*learn=*/true);

        // Promote transient synapses to permanent.  [verified: afterLearningAllEpisodes
        // → doFinalTransToPermSynapsePromotionPass, Network.java:4063/4073]
        net.afterLearningAllEpisodes();

        // Flip to recognition (operationalMode is a static byte).  [verified: Network.java:90]
        Network.operationalMode = Network.RECOGNITION_MODE;

        // --- Recognize phase (deterministic max-V via Use_ML_Recog:true) ---
        first = runPhase(net, "recognize", json, first, /*learn=*/false);

        json.append("\n]\n");
        System.out.print(json);
    }

    // UNVERIFIED helper: the exact build-order calls.
    static void buildNetwork(Network net) throws Exception {
        net.buildNetworkObjects();
        net.buildNetworkMatrices();
        net.setupRFsAndOtherStuff();
        net.establishBackoffStrategies();
    }

    // UNVERIFIED helper: attach a fixed-input EpisodeContainer to the input region.
    static void injectInputs(Network net) throws Exception {
        // Placeholder: implement EpisodeContainer population here (see comment above).
        // Left intentionally minimal — this is the main integration point to complete
        // on a JDK 9+ machine with the compiler available to iterate against.
    }

    // Run one phase over the alphabet, appending {phase,features,code} per input.
    static boolean runPhase(Network net, String phase, StringBuilder json, boolean first, boolean learn)
            throws Exception {
        net.prepareForNewRun(false);                         // [verified: Network.java:3019]

        InternalRegion region = (InternalRegion) net.getRegion("internal"); // [verified: 1534]
        Mac mac = region.getMac(0);                          // [verified: InternalRegion.java:613]

        for (int[] input : INPUTS) {
            presentFrame(net, input);                        // UNVERIFIED: per-frame input set
            if (learn) {
                net.prepareEventLearn();                     // [verified: 4604]
                net.doFrameLearn();                          // [verified: 1822]
            } else {
                net.prepareEventRecognize();                 // [verified: 4646]
                net.doFrameRecognize();                      // [verified: 2243]
            }
            byte[] code = mac.getCurrentCode();              // [verified: Mac.java:4398]
            if (!first) json.append(",\n");
            first = false;
            json.append("  {\"phase\": \"").append(phase)
                .append("\", \"features\": ").append(intArray(input))
                .append(", \"code\": ").append(byteArray(code)).append("}");
        }
        return first;
    }

    // UNVERIFIED: set the active input cells for the next frame (drives the input
    // region's current-frame active list). Depends on how injectInputs models frames.
    static void presentFrame(Network net, int[] activeCells) throws Exception {
        // If using an injected EpisodeContainer, advance to the frame for `activeCells`
        // (loadNextFrame). If setting cells directly, mark InputRegionNeuron.setActive.
    }

    static String intArray(int[] a) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) { if (i > 0) b.append(","); b.append(a[i]); }
        return b.append("]").toString();
    }

    static String byteArray(byte[] a) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) { if (i > 0) b.append(","); b.append(a[i] & 0xff); }
        return b.append("]").toString();
    }
}
