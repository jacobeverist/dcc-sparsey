#!/usr/bin/env bash
# Compile SparseyCore + the fidelity Driver headless and emit the Java golden vector
# consumed by `cargo test -p dcc_sparsey --test fidelity_behavioral -- --ignored`.
#
# Usage:
#   fidelity/build_and_run.sh [path-to-SparseyCore] > tests/fixtures/sparsey_java_golden.json
#
# The SparseyCore checkout (commit a0d4d34) is passed as $1 or via $SPARSEYCORE. There is
# deliberately NO default: a path that resolves on one machine is a trap dressed as a
# convenience — it makes the script look like it worked for everyone who happens to have
# something at that location.
#
# REQUIRES JDK 9+ — SparseyCore uses java.util.List.of(). With only JDK 8 the compile
# fails at Bundle.java (List.of). Set JAVA_HOME to a 9+ JDK if `javac -version` < 9.
#
# The Driver is `package SparseyCore;` so it compiles into that package and reaches the
# protected frame-loop methods — no SparseyCore repo modification needed.
set -euo pipefail

SC="${1:-${SPARSEYCORE:?pass the SparseyCore checkout as \$1 or set SPARSEYCORE}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
DRIVER="$HERE/java/Driver.java"
COM="$HERE/config/m1.com.json"

if [[ ! -d "$SC/src/SparseyCore" ]]; then
    echo "error: SparseyCore source not found at $SC/src/SparseyCore" >&2
    exit 1
fi

# Enforce JDK 9+.
JV="$(javac -version 2>&1 | sed -E 's/javac ([0-9]+).*/\1/; s/1\.([0-9]+).*/\1/')"
if [[ "$JV" -lt 9 ]]; then
    echo "error: need JDK 9+ (SparseyCore uses List.of); found javac major=$JV." >&2
    echo "       set JAVA_HOME to a 9+ JDK and re-run." >&2
    exit 1
fi

echo "Compiling SparseyCore + Driver (JDK $JV)..." >&2
rm -rf "$OUT"; mkdir -p "$OUT"
javac -d "$OUT" \
    "$SC/src/SparseyCore/"*.java \
    "$SC/src/util/"*.java \
    "$SC/src/stats/"*.java \
    "$DRIVER"

echo "Running Driver (config in $HERE/config)..." >&2
# Run from the config dir so the COM's relative ndfPath/ndf resolve.
( cd "$HERE/config" && java -cp "$OUT" SparseyCore.Driver "$(basename "$COM")" )
