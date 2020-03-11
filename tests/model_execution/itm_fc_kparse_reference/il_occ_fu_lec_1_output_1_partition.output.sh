#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


leccalc -r -Kil_S1_summaryleccalc -f output/il_S1_leccalc_full_uncertainty_oep.csv & lpid1=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -f output/full_correlation/il_S1_leccalc_full_uncertainty_oep.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
