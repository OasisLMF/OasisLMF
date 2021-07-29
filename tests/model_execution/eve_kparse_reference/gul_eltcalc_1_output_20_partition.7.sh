#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_eltcalc_P8



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid1=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 &

eve -R 8 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P8  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P8 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

