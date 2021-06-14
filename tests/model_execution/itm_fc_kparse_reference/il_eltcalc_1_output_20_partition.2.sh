#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
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


mkfifo fifo/full_correlation/gul_fc_P3

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_eltcalc_P3

mkfifo fifo/full_correlation/il_P3

mkfifo fifo/full_correlation/il_S1_summary_P3
mkfifo fifo/full_correlation/il_S1_eltcalc_P3



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid1=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &

# --- Do insured loss computes ---
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P3 fifo/full_correlation/il_S1_eltcalc_P3 > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P3 < fifo/full_correlation/il_P3 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P3 > fifo/full_correlation/il_P3 &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P3 -a1 -i - | fmcalc -a2 > fifo/il_P3  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P3 > output/il_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P3 > output/full_correlation/il_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

