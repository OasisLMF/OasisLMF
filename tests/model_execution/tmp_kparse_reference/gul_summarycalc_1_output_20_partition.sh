#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +

rm -R -f work/*
mkdir work/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P20

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20



# --- Do ground up loss computes ---

summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid2=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid3=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid4=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid5=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid6=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid8=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid9=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 & pid11=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 & pid12=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 & pid13=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 & pid14=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid15=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 & pid17=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 & pid18=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 & pid19=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 & pid20=$!

tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > /dev/null & pid21=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 > /dev/null & pid22=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 > /dev/null & pid23=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11 > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12 > /dev/null & pid32=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13 > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14 > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15 > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16 > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17 > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18 > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19 > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20 > /dev/null & pid40=$!

summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/gul_P3 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/gul_P5 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/gul_P6 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/gul_P9 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/gul_P10 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/gul_P11 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/gul_P12 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/gul_P13 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/gul_P14 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/gul_P15 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/gul_P16 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/gul_P17 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/gul_P18 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/gul_P19 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/gul_P20 &

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
