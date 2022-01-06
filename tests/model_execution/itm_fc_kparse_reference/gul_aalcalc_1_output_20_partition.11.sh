#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

mkfifo fifo/gul_P12

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_summary_P12.idx

mkfifo fifo/full_correlation/gul_P12

mkfifo fifo/full_correlation/gul_S1_summary_P12
mkfifo fifo/full_correlation/gul_S1_summary_P12.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P12 work/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P12.idx work/gul_S1_summaryaalcalc/P12.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 &

# --- Do ground up loss computes ---
tee < fifo/full_correlation/gul_S1_summary_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid3=$!
tee < fifo/full_correlation/gul_S1_summary_P12.idx work/full_correlation/gul_S1_summaryaalcalc/P12.idx > /dev/null & pid4=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P12 < fifo/full_correlation/gul_P12 &

eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P12 -a1 -i - > fifo/gul_P12  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

