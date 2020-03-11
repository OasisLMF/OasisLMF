#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryleccalc

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9

mkfifo il_S1_summary_P9



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P9 work/il_S1_summaryleccalc/P9.bin > /dev/null & pid1=$!
summarycalc -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 &

eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P9 -a1 -i - | fmcalc -a2 > fifo/il_P9  &

wait $pid1

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P9 > il_P9 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
tee < il_S1_summary_P9 work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid1=$!
summarycalc -f  -1 il_S1_summary_P9 < il_P9 &

wait $pid1


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

