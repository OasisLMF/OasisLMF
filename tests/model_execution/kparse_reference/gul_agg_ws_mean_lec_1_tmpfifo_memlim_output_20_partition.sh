#!/bin/bash

rm -R -f output/*
rm -R -f work/*

mkdir work/kat
mkdir -p /tmp/6nGisiWgNZ/fifo
mkfifo /tmp/6nGisiWgNZ/fifo/gul_P1

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P1

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P2

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P2

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P3

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P3

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P4

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P4

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P5

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P5

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P6

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P6

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P7

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P7

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P8

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P8

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P9

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P9

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P10

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P10

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P11

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P11

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P12

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P12

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P13

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P13

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P14

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P14

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P15

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P15

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P16

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P16

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P17

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P17

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P18

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P18

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P19

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P19

mkfifo /tmp/6nGisiWgNZ/fifo/gul_P20

mkfifo /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P20

mkdir work/gul_S1_summaryleccalc

# --- Use Ktools per process memory limit ---

ulimit -v $(ktgetmem 20)

# --- Do ground up loss computes ---





















tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid2=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid3=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid4=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid5=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid6=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid7=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid8=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P9 work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid9=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid10=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P11 work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid11=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P12 work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid12=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P13 work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid13=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P14 work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid14=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P15 work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid15=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P16 work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid16=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P17 work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid17=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P18 work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid18=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P19 work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid19=$!
tee < /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P20 work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid20=$!
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P1 < /tmp/6nGisiWgNZ/fifo/gul_P1 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P2 < /tmp/6nGisiWgNZ/fifo/gul_P2 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P3 < /tmp/6nGisiWgNZ/fifo/gul_P3 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P4 < /tmp/6nGisiWgNZ/fifo/gul_P4 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P5 < /tmp/6nGisiWgNZ/fifo/gul_P5 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P6 < /tmp/6nGisiWgNZ/fifo/gul_P6 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P7 < /tmp/6nGisiWgNZ/fifo/gul_P7 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P8 < /tmp/6nGisiWgNZ/fifo/gul_P8 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P9 < /tmp/6nGisiWgNZ/fifo/gul_P9 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P10 < /tmp/6nGisiWgNZ/fifo/gul_P10 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P11 < /tmp/6nGisiWgNZ/fifo/gul_P11 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P12 < /tmp/6nGisiWgNZ/fifo/gul_P12 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P13 < /tmp/6nGisiWgNZ/fifo/gul_P13 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P14 < /tmp/6nGisiWgNZ/fifo/gul_P14 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P15 < /tmp/6nGisiWgNZ/fifo/gul_P15 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P16 < /tmp/6nGisiWgNZ/fifo/gul_P16 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P17 < /tmp/6nGisiWgNZ/fifo/gul_P17 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P18 < /tmp/6nGisiWgNZ/fifo/gul_P18 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P19 < /tmp/6nGisiWgNZ/fifo/gul_P19 &
summarycalc -g  -1 /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P20 < /tmp/6nGisiWgNZ/fifo/gul_P20 &

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -c - > /tmp/6nGisiWgNZ/fifo/gul_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do ground up loss kats ---


# --- Remove per process memory limit ---

ulimit -v $(ktgetmem 1)

leccalc -r -Kgul_S1_summaryleccalc -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv & lpid1=$!
wait $lpid1

rm /tmp/6nGisiWgNZ/fifo/gul_P1

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P1

rm /tmp/6nGisiWgNZ/fifo/gul_P2

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P2

rm /tmp/6nGisiWgNZ/fifo/gul_P3

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P3

rm /tmp/6nGisiWgNZ/fifo/gul_P4

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P4

rm /tmp/6nGisiWgNZ/fifo/gul_P5

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P5

rm /tmp/6nGisiWgNZ/fifo/gul_P6

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P6

rm /tmp/6nGisiWgNZ/fifo/gul_P7

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P7

rm /tmp/6nGisiWgNZ/fifo/gul_P8

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P8

rm /tmp/6nGisiWgNZ/fifo/gul_P9

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P9

rm /tmp/6nGisiWgNZ/fifo/gul_P10

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P10

rm /tmp/6nGisiWgNZ/fifo/gul_P11

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P11

rm /tmp/6nGisiWgNZ/fifo/gul_P12

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P12

rm /tmp/6nGisiWgNZ/fifo/gul_P13

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P13

rm /tmp/6nGisiWgNZ/fifo/gul_P14

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P14

rm /tmp/6nGisiWgNZ/fifo/gul_P15

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P15

rm /tmp/6nGisiWgNZ/fifo/gul_P16

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P16

rm /tmp/6nGisiWgNZ/fifo/gul_P17

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P17

rm /tmp/6nGisiWgNZ/fifo/gul_P18

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P18

rm /tmp/6nGisiWgNZ/fifo/gul_P19

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P19

rm /tmp/6nGisiWgNZ/fifo/gul_P20

rm /tmp/6nGisiWgNZ/fifo/gul_S1_summary_P20

rm -rf work/kat
rm work/gul_S1_summaryleccalc/*
rmdir work/gul_S1_summaryleccalc

rm /tmp/6nGisiWgNZ/fifo/*
rmdir /tmp/6nGisiWgNZ/fifo
rmdir /tmp/6nGisiWgNZ/
