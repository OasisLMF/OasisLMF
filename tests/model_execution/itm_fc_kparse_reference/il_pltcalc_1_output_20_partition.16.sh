#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

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


mkfifo fifo/full_correlation/gul_fc_P17

mkfifo fifo/il_P17

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/full_correlation/il_P17

mkfifo fifo/full_correlation/il_S1_summary_P17
mkfifo fifo/full_correlation/il_S1_pltcalc_P17



# --- Do insured loss computes ---
pltcalc -H < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 & pid1=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_pltcalc_P17 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 &

# --- Do insured loss computes ---
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P17 > work/full_correlation/kat/il_S1_pltcalc_P17 & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P17 fifo/full_correlation/il_S1_pltcalc_P17 > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P17 < fifo/full_correlation/il_P17 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P17 > fifo/full_correlation/il_P17 &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P17 -a1 -i - | fmcalc -a2 > fifo/il_P17  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P17 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P17 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2
