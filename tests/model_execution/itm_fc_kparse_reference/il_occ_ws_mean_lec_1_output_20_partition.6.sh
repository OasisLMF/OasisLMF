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

mkdir work/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryleccalc

mkfifo fifo/full_correlation/gul_fc_P7

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx

mkfifo fifo/full_correlation/il_P7

mkfifo fifo/full_correlation/il_S1_summary_P7
mkfifo fifo/full_correlation/il_S1_summary_P7.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &

# --- Do insured loss computes ---
tee < fifo/full_correlation/il_S1_summary_P7 work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P7.idx work/full_correlation/il_S1_summaryleccalc/P7.idx > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P7 < fifo/full_correlation/il_P7 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P7 > fifo/full_correlation/il_P7 &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P7 -a1 -i - | fmcalc -a2 > fifo/il_P7  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

