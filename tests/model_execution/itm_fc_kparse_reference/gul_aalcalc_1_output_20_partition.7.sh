#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

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

mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P8

mkfifo fifo/full_correlation/gul_P8

mkfifo fifo/full_correlation/gul_S1_summary_P8



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P8 work/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid1=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 &

# --- Do ground up loss computes ---
tee < fifo/full_correlation/gul_S1_summary_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P8 < fifo/full_correlation/gul_P8 &

eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P8 -a1 -i - > fifo/gul_P8  &

wait $pid1 $pid2


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

