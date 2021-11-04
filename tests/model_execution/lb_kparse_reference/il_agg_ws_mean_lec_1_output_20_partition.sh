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

fmpy -a2 --create-financial-structure-files
mkdir work/il_S1_summaryleccalc

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
mkfifo fifo/il_S1_summary_P1.idx

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summary_P4.idx

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summary_P8.idx

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summary_P9.idx

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summary_P10.idx

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



tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!
tee < fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid5=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid6=$!
tee < fifo/il_S1_summary_P4 work/il_S1_summaryleccalc/P4.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid8=$!
tee < fifo/il_S1_summary_P5 work/il_S1_summaryleccalc/P5.bin > /dev/null & pid9=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid10=$!
tee < fifo/il_S1_summary_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid11=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid12=$!
tee < fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid13=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid14=$!
tee < fifo/il_S1_summary_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid15=$!
tee < fifo/il_S1_summary_P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P9 work/il_S1_summaryleccalc/P9.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P10 work/il_S1_summaryleccalc/P10.bin > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid20=$!

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

eve 1 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P1  &
eve 2 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P2  &
eve 3 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P3  &
eve 4 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P4  &
eve 5 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P5  &
eve 6 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P6  &
eve 7 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P7  &
eve 8 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P8  &
eve 9 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P9  &
eve 10 10 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_lb_P10  &
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
load_balancer -i fifo/gul_lb_P3 fifo/gul_lb_P4 -o fifo/lb_il_P3 fifo/lb_il_P4 &
load_balancer -i fifo/gul_lb_P5 fifo/gul_lb_P6 -o fifo/lb_il_P5 fifo/lb_il_P6 &
load_balancer -i fifo/gul_lb_P7 fifo/gul_lb_P8 -o fifo/lb_il_P7 fifo/lb_il_P8 &
load_balancer -i fifo/gul_lb_P9 fifo/gul_lb_P10 -o fifo/lb_il_P9 fifo/lb_il_P10 &
fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 &
fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 &
fmpy -a2 < fifo/lb_il_P3 > fifo/il_P3 &
fmpy -a2 < fifo/lb_il_P4 > fifo/il_P4 &
fmpy -a2 < fifo/lb_il_P5 > fifo/il_P5 &
fmpy -a2 < fifo/lb_il_P6 > fifo/il_P6 &
fmpy -a2 < fifo/lb_il_P7 > fifo/il_P7 &
fmpy -a2 < fifo/lb_il_P8 > fifo/il_P8 &
fmpy -a2 < fifo/lb_il_P9 > fifo/il_P9 &
fmpy -a2 < fifo/lb_il_P10 > fifo/il_P10 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---


leccalc -r -Kil_S1_summaryleccalc -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
