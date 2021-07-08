#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summarycalc_P1



# --- Do ground up loss computes ---

summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid1=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summarycalc_P1 > /dev/null & pid2=$!

summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P1  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*
