#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*


leccalc -r -Kgul_S1_summaryleccalc -f output/gul_S1_leccalc_full_uncertainty_oep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
