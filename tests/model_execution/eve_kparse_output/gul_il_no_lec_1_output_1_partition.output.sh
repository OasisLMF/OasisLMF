#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
