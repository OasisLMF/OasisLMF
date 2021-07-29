#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

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


mkfifo fifo/full_correlation/gul_fc_P16

mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/full_correlation/il_P16

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_pltcalc_P16



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 & pid1=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_pltcalc_P16 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &

# --- Do insured loss computes ---
pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P16 > work/full_correlation/kat/il_S1_pltcalc_P16 & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P16 fifo/full_correlation/il_S1_pltcalc_P16 > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P16 > fifo/full_correlation/il_P16 &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P16 -a1 -i - | fmcalc -a2 > fifo/il_P16  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P16 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P16 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

