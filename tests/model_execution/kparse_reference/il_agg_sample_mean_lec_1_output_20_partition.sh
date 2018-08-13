#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4

mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5

mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7

mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P8

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9

mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10

mkfifo fifo/il_P11

mkfifo fifo/il_S1_summary_P11

mkfifo fifo/il_P12

mkfifo fifo/il_S1_summary_P12

mkfifo fifo/il_P13

mkfifo fifo/il_S1_summary_P13

mkfifo fifo/il_P14

mkfifo fifo/il_S1_summary_P14

mkfifo fifo/il_P15

mkfifo fifo/il_S1_summary_P15

mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16

mkfifo fifo/il_P17

mkfifo fifo/il_S1_summary_P17

mkfifo fifo/il_P18

mkfifo fifo/il_S1_summary_P18

mkfifo fifo/il_P19

mkfifo fifo/il_S1_summary_P19

mkfifo fifo/il_P20

mkfifo fifo/il_S1_summary_P20

mkdir work/il_S1_summaryleccalc

# --- Do insured loss computes ---





















tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid2=$!
tee < fifo/il_S1_summary_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P4 work/il_S1_summaryleccalc/P4.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P5 work/il_S1_summaryleccalc/P5.bin > /dev/null & pid5=$!
tee < fifo/il_S1_summary_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid6=$!
tee < fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid8=$!
tee < fifo/il_S1_summary_P9 work/il_S1_summaryleccalc/P9.bin > /dev/null & pid9=$!
tee < fifo/il_S1_summary_P10 work/il_S1_summaryleccalc/P10.bin > /dev/null & pid10=$!
tee < fifo/il_S1_summary_P11 work/il_S1_summaryleccalc/P11.bin > /dev/null & pid11=$!
tee < fifo/il_S1_summary_P12 work/il_S1_summaryleccalc/P12.bin > /dev/null & pid12=$!
tee < fifo/il_S1_summary_P13 work/il_S1_summaryleccalc/P13.bin > /dev/null & pid13=$!
tee < fifo/il_S1_summary_P14 work/il_S1_summaryleccalc/P14.bin > /dev/null & pid14=$!
tee < fifo/il_S1_summary_P15 work/il_S1_summaryleccalc/P15.bin > /dev/null & pid15=$!
tee < fifo/il_S1_summary_P16 work/il_S1_summaryleccalc/P16.bin > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P17 work/il_S1_summaryleccalc/P17.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P18 work/il_S1_summaryleccalc/P18.bin > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P19 work/il_S1_summaryleccalc/P19.bin > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P20 work/il_S1_summaryleccalc/P20.bin > /dev/null & pid20=$!
summarycalc -f -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -f -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarycalc -f -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarycalc -f -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarycalc -f -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarycalc -f -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarycalc -f -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarycalc -f -1 fifo/il_S1_summary_P8 < fifo/il_P8 &
summarycalc -f -1 fifo/il_S1_summary_P9 < fifo/il_P9 &
summarycalc -f -1 fifo/il_S1_summary_P10 < fifo/il_P10 &
summarycalc -f -1 fifo/il_S1_summary_P11 < fifo/il_P11 &
summarycalc -f -1 fifo/il_S1_summary_P12 < fifo/il_P12 &
summarycalc -f -1 fifo/il_S1_summary_P13 < fifo/il_P13 &
summarycalc -f -1 fifo/il_S1_summary_P14 < fifo/il_P14 &
summarycalc -f -1 fifo/il_S1_summary_P15 < fifo/il_P15 &
summarycalc -f -1 fifo/il_S1_summary_P16 < fifo/il_P16 &
summarycalc -f -1 fifo/il_S1_summary_P17 < fifo/il_P17 &
summarycalc -f -1 fifo/il_S1_summary_P18 < fifo/il_P18 &
summarycalc -f -1 fifo/il_S1_summary_P19 < fifo/il_P19 &
summarycalc -f -1 fifo/il_S1_summary_P20 < fifo/il_P20 &

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---


leccalc -r -Kil_S1_summaryleccalc -S output/il_S1_leccalc_sample_mean_aep.csv & lpid1=$!
wait $lpid1


rm fifo/il_P1

rm fifo/il_S1_summary_P1

rm fifo/il_P2

rm fifo/il_S1_summary_P2

rm fifo/il_P3

rm fifo/il_S1_summary_P3

rm fifo/il_P4

rm fifo/il_S1_summary_P4

rm fifo/il_P5

rm fifo/il_S1_summary_P5

rm fifo/il_P6

rm fifo/il_S1_summary_P6

rm fifo/il_P7

rm fifo/il_S1_summary_P7

rm fifo/il_P8

rm fifo/il_S1_summary_P8

rm fifo/il_P9

rm fifo/il_S1_summary_P9

rm fifo/il_P10

rm fifo/il_S1_summary_P10

rm fifo/il_P11

rm fifo/il_S1_summary_P11

rm fifo/il_P12

rm fifo/il_S1_summary_P12

rm fifo/il_P13

rm fifo/il_S1_summary_P13

rm fifo/il_P14

rm fifo/il_S1_summary_P14

rm fifo/il_P15

rm fifo/il_S1_summary_P15

rm fifo/il_P16

rm fifo/il_S1_summary_P16

rm fifo/il_P17

rm fifo/il_S1_summary_P17

rm fifo/il_P18

rm fifo/il_S1_summary_P18

rm fifo/il_P19

rm fifo/il_S1_summary_P19

rm fifo/il_P20

rm fifo/il_S1_summary_P20

rm -rf work/kat
rm work/il_S1_summaryleccalc/*
rmdir work/il_S1_summaryleccalc
