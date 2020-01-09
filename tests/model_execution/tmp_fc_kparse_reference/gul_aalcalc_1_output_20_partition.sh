#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20



# --- Do ground up loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 work/gul_S1_summaryaalcalc/P3.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 work/gul_S1_summaryaalcalc/P4.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 work/gul_S1_summaryaalcalc/P5.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 work/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 work/gul_S1_summaryaalcalc/P7.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 work/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 work/gul_S1_summaryaalcalc/P9.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 work/gul_S1_summaryaalcalc/P10.bin > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 work/gul_S1_summaryaalcalc/P11.bin > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 work/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 work/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 work/gul_S1_summaryaalcalc/P14.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 work/gul_S1_summaryaalcalc/P15.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 work/gul_S1_summaryaalcalc/P16.bin > /dev/null & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 work/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 work/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 work/gul_S1_summaryaalcalc/P19.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 work/gul_S1_summaryaalcalc/P20.bin > /dev/null & pid20=$!

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

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P11 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P12 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P14 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P15 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P16 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P18 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P19 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P20 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11 work/full_correlation/gul_S1_summaryaalcalc/P11.bin > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14 work/full_correlation/gul_S1_summaryaalcalc/P14.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16 work/full_correlation/gul_S1_summaryaalcalc/P16.bin > /dev/null & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19 work/full_correlation/gul_S1_summaryaalcalc/P19.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20 work/full_correlation/gul_S1_summaryaalcalc/P20.bin > /dev/null & pid20=$!

summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P11 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P12 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P14 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P15 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P16 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P18 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P19 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P20 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
