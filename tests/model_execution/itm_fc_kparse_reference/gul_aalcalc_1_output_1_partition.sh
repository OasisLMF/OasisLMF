#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1

mkfifo fifo/full_correlation/gul_P1

mkfifo fifo/full_correlation/gul_S1_summary_P1



# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

# --- Do ground up loss computes ---


tee < fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid2=$!

summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  &

wait $pid1 $pid2


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
