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
mkfifo fifo/il_P3
mkfifo fifo/il_P4
mkfifo fifo/il_P5
mkfifo fifo/il_P6
mkfifo fifo/il_P7
mkfifo fifo/il_P8
mkfifo fifo/il_P9
mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2
mkfifo fifo/gul_lb_P3
mkfifo fifo/gul_lb_P4
mkfifo fifo/gul_lb_P5
mkfifo fifo/gul_lb_P6
mkfifo fifo/gul_lb_P7
mkfifo fifo/gul_lb_P8
mkfifo fifo/gul_lb_P9
mkfifo fifo/gul_lb_P10

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2
mkfifo fifo/lb_il_P3
mkfifo fifo/lb_il_P4
mkfifo fifo/lb_il_P5
mkfifo fifo/lb_il_P6
mkfifo fifo/lb_il_P7
mkfifo fifo/lb_il_P8
mkfifo fifo/lb_il_P9
mkfifo fifo/lb_il_P10



# --- Do insured loss computes ---

pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid1=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid3=$!
pltcalc -H < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid4=$!
pltcalc -H < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid5=$!
pltcalc -H < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid6=$!
pltcalc -H < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid7=$!
pltcalc -H < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid8=$!
pltcalc -H < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid9=$!
pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid10=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_pltcalc_P1 > /dev/null & pid11=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_pltcalc_P2 > /dev/null & pid12=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_pltcalc_P3 > /dev/null & pid13=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_pltcalc_P4 > /dev/null & pid14=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_pltcalc_P5 > /dev/null & pid15=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_pltcalc_P6 > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_pltcalc_P7 > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_pltcalc_P8 > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_pltcalc_P9 > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_pltcalc_P10 > /dev/null & pid20=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarycalc -m -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarycalc -m -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarycalc -m -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarycalc -m -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 &
summarycalc -m -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 &
summarycalc -m -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 &

( eve 1 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P1  ) & 
( eve 2 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P2  ) & 
( eve 3 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P3  ) & 
( eve 4 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P4  ) & 
( eve 5 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P5  ) & 
( eve 6 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P6  ) & 
( eve 7 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P7  ) & 
( eve 8 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P8  ) & 
( eve 9 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P9  ) & 
( eve 10 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P10  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
load_balancer -i fifo/gul_lb_P3 fifo/gul_lb_P4 -o fifo/lb_il_P3 fifo/lb_il_P4 &
load_balancer -i fifo/gul_lb_P5 fifo/gul_lb_P6 -o fifo/lb_il_P5 fifo/lb_il_P6 &
load_balancer -i fifo/gul_lb_P7 fifo/gul_lb_P8 -o fifo/lb_il_P7 fifo/lb_il_P8 &
load_balancer -i fifo/gul_lb_P9 fifo/gul_lb_P10 -o fifo/lb_il_P9 fifo/lb_il_P10 &
( fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 ) & pid21=$!
( fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 ) & pid22=$!
( fmpy -a2 < fifo/lb_il_P3 > fifo/il_P3 ) & pid23=$!
( fmpy -a2 < fifo/lb_il_P4 > fifo/il_P4 ) & pid24=$!
( fmpy -a2 < fifo/lb_il_P5 > fifo/il_P5 ) & pid25=$!
( fmpy -a2 < fifo/lb_il_P6 > fifo/il_P6 ) & pid26=$!
( fmpy -a2 < fifo/lb_il_P7 > fifo/il_P7 ) & pid27=$!
( fmpy -a2 < fifo/lb_il_P8 > fifo/il_P8 ) & pid28=$!
( fmpy -a2 < fifo/lb_il_P9 > fifo/il_P9 ) & pid29=$!
( fmpy -a2 < fifo/lb_il_P10 > fifo/il_P10 ) & pid30=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*
