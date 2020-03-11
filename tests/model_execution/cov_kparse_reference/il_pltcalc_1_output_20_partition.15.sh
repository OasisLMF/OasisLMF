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


mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summarypltcalc_P16
mkfifo fifo/il_S1_pltcalc_P16



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_summarypltcalc_P16 > work/kat/il_S1_pltcalc_P16 & pid1=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_summarypltcalc_P16 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &

eve 16 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P16  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P16 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1

