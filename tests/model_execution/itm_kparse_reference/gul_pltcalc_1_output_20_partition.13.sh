#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

find fifo/ \( -name '*P14[^0-9]*' -o -name '*P14' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/gul_P14

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_pltcalc_P14



# --- Do ground up loss computes ---
pltcalc -H < fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 & pid1=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_pltcalc_P14 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 &

eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P14  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P14 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1

