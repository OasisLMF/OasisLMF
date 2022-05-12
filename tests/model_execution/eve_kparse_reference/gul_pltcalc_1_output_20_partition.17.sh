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


mkfifo fifo/gul_P18

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_pltcalc_P18



# --- Do ground up loss computes ---
pltcalc -H < fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 & pid1=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_pltcalc_P18 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 &

eve -R 18 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P18  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat -u work/kat/gul_S1_pltcalc_P18 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1

