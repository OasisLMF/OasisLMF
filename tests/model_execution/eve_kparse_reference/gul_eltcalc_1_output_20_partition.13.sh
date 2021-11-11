#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo fifo/gul_P14

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_eltcalc_P14



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 & pid1=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_eltcalc_P14 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 &

eve -R 14 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P14  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P14 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

