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


mkfifo fifo/il_P17

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_summarycalc_P17



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 & pid1=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_summarycalc_P17 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 &

eve -R 17 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P17  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat -u work/kat/il_S1_summarycalc_P17 > output/il_S1_summarycalc.csv & kpid1=$!
wait $kpid1

