#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


leccalc -r -Kil_S1_summaryleccalc -s output/il_S1_leccalc_sample_mean_oep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
