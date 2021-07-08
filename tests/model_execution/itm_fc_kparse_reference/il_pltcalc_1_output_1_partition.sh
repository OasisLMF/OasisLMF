#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/full_correlation/gul_fc_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/full_correlation/il_P1

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1



# --- Do insured loss computes ---

pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_pltcalc_P1 > /dev/null & pid2=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

# --- Do insured loss computes ---

pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid3=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_pltcalc_P1 > /dev/null & pid4=$!

summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P1 > fifo/full_correlation/il_P1 &
eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P1 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2


rm -R -f work/*
rm -R -f fifo/*
