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


mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_pltcalc_P10



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid1=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_pltcalc_P10 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 &

eve -R 10 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P10  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P10 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1

