#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/


mkfifo fifo/il_P12

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summarypltcalc_P12
mkfifo fifo/il_S1_pltcalc_P12



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_summarypltcalc_P12 > work/kat/il_S1_pltcalc_P12 & pid1=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_summarypltcalc_P12 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 &

eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P12  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P12 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1

