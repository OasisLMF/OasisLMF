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

mkdir work/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryleccalc

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx

mkfifo fifo/full_correlation/gul_P1

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_summary_P1.idx



# --- Do ground up loss computes ---



tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

# --- Do ground up loss computes ---



tee < fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid3=$!
tee < fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid4=$!

summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


leccalc -r -Kgul_S1_summaryleccalc -W output/gul_S1_leccalc_wheatsheaf_aep.csv & lpid1=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
