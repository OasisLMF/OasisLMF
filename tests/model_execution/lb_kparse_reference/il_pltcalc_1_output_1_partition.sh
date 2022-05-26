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

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2



# --- Do insured loss computes ---

pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid1=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid2=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_pltcalc_P1 > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_pltcalc_P2 > /dev/null & pid4=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P1  &
eve 2 2 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P2  &
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 &
fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*
