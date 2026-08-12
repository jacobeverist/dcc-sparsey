#!/usr/bin/env python3
"""Generate a behavioral-invariant reference fixture from the Sparsey_Alt Python
implementation — an *independent* reimplementation of Rinkus's Sparsey algorithm
(from the papers, by a third party). Used to cross-check that the dcc `sparsey`
Rust port exhibits the same coding behavior.

Why behavioral invariants, not exact codes: Sparsey's winner selection is the
probabilistic sigmoid CSA (see ../../doc/AlgorithmTriangulation.md), so novel-code
assignment is stochastic and cannot be bit-matched across implementations / RNGs.
What IS comparable is the qualitative behavior every faithful Sparsey exhibits:

  1. N well-separated inputs learn into N *distinct* codes.
  2. A learned input, re-presented, *reactivates its own* code (max-V readout).
  3. Global familiarity G is high (~1.0) for a learned input, ~0 for a novel one.

The scenario: one MAC (Q CMs x K cells) fed a flat U-bit input; learn the alphabet
with the probabilistic CSA (seeded), then recognize each input with a deterministic
max-V readout (mirrors SparseyCore `Use_ML_Recog`).

Requires: Python 3, numpy, and the Sparsey_Alt Python package on PYTHONPATH:
    PYTHONPATH=/path/to/Sparsey_Alt/Python python3 generate_reference.py > ../../tests/fixtures/sparsey_alt_reference.json
"""
import json
import sys

import numpy as np

from sparsey import macrocolumn

K, Q, U = 8, 5, 16
SEED = 42

# Alphabet: 4 well-separated sparse U-bit inputs, plus one novel held-out input.
ALPHABET = {"A": [0, 1, 2], "B": [6, 7, 8], "C": [11, 12, 13], "D": [3, 4, 14]}
NOVEL = [5, 9, 10, 15]


def vec(active_idx):
    return [1 if i in active_idx else 0 for i in range(U)]


def build():
    mac = macrocolumn.unit(K, Q, [U, 0, 0])
    macrocolumn.createMinicolumns(mac)
    return mac


def familiarity(mac, v):
    macrocolumn.calculateGlobalFamiliarity(mac, [v, [], []])
    macrocolumn.calculateExpansivity(mac)
    return float(mac.globalFamiliarity)


def learn(mac, v):
    """Probabilistic CSA winner selection + Hebbian learn (the real Sparsey path)."""
    familiarity(mac, v)
    probs = macrocolumn.getColumnProbabilities(mac)
    code = macrocolumn.getWinners(probs)  # np.random.choice sample
    macrocolumn.learnSignal(mac, code, [v, [], []])
    return [int(c) for c in code]


def recognize(mac, v):
    """Deterministic max-V readout (like SparseyCore Use_ML_Recog) + G."""
    g = familiarity(mac, v)
    probs = macrocolumn.getColumnProbabilities(mac)
    return [int(np.argmax(p)) for p in probs], g


def main():
    np.random.seed(SEED)
    mac = build()

    learned = {name: learn(mac, vec(idx)) for name, idx in ALPHABET.items()}
    recog = {name: recognize(mac, vec(idx)) for name, idx in ALPHABET.items()}
    _, g_novel = recognize(mac, vec(NOVEL))

    distinct = len({tuple(c) for c in learned.values()})
    reactivates = {name: recog[name][0] == learned[name] for name in ALPHABET}
    g_familiar = [recog[name][1] for name in ALPHABET]

    fixture = {
        "reference": "Sparsey_Alt/Python (independent Rinkus reimplementation)",
        "scenario": {"K": K, "Q": Q, "U": U, "seed": SEED, "n_inputs": len(ALPHABET)},
        "learned_codes": learned,
        "recognition_codes": {n: recog[n][0] for n in ALPHABET},
        "invariants": {
            "distinct_codes": distinct,
            "all_reactivate": all(reactivates.values()),
            "g_familiar_min": round(min(g_familiar), 6),
            "g_novel": round(g_novel, 6),
        },
    }
    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
