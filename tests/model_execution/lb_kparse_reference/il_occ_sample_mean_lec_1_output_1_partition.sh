#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

fmpy -a2 --create-financial-structure-files
mkdir work/il_S1_summaryleccalc

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1

mkfifo fifo/il_S1_summary_P2

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2



# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid2=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P1  &
eve 2 2 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P2  &
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 &
fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 &

wait $pid1 $pid2


# --- Do insured loss kats ---


leccalc -r -Kil_S1_summaryleccalc -s output/il_S1_leccalc_sample_mean_oep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
