#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryaalcalc_P1

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summaryaalcalc_P2

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summaryaalcalc_P3

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summaryaalcalc_P4

mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summaryaalcalc_P5

mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summaryaalcalc_P6

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summaryaalcalc_P7

mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summaryaalcalc_P8

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summaryaalcalc_P9

mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summaryaalcalc_P10

mkfifo fifo/il_P11

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summaryaalcalc_P11

mkfifo fifo/il_P12

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summaryaalcalc_P12

mkfifo fifo/il_P13

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_summaryaalcalc_P13

mkfifo fifo/il_P14

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_summaryaalcalc_P14

mkfifo fifo/il_P15

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summaryaalcalc_P15

mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summaryaalcalc_P16

mkfifo fifo/il_P17

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_summaryaalcalc_P17

mkfifo fifo/il_P18

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_summaryaalcalc_P18

mkfifo fifo/il_P19

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_summaryaalcalc_P19

mkfifo fifo/il_P20

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_summaryaalcalc_P20

mkdir work/il_S1_aalcalc

# --- Do insured loss computes ---

aalcalc < fifo/il_S1_summaryaalcalc_P1 > work/il_S1_aalcalc/P1.bin & pid1=$!

aalcalc < fifo/il_S1_summaryaalcalc_P2 > work/il_S1_aalcalc/P2.bin & pid2=$!

aalcalc < fifo/il_S1_summaryaalcalc_P3 > work/il_S1_aalcalc/P3.bin & pid3=$!

aalcalc < fifo/il_S1_summaryaalcalc_P4 > work/il_S1_aalcalc/P4.bin & pid4=$!

aalcalc < fifo/il_S1_summaryaalcalc_P5 > work/il_S1_aalcalc/P5.bin & pid5=$!

aalcalc < fifo/il_S1_summaryaalcalc_P6 > work/il_S1_aalcalc/P6.bin & pid6=$!

aalcalc < fifo/il_S1_summaryaalcalc_P7 > work/il_S1_aalcalc/P7.bin & pid7=$!

aalcalc < fifo/il_S1_summaryaalcalc_P8 > work/il_S1_aalcalc/P8.bin & pid8=$!

aalcalc < fifo/il_S1_summaryaalcalc_P9 > work/il_S1_aalcalc/P9.bin & pid9=$!

aalcalc < fifo/il_S1_summaryaalcalc_P10 > work/il_S1_aalcalc/P10.bin & pid10=$!

aalcalc < fifo/il_S1_summaryaalcalc_P11 > work/il_S1_aalcalc/P11.bin & pid11=$!

aalcalc < fifo/il_S1_summaryaalcalc_P12 > work/il_S1_aalcalc/P12.bin & pid12=$!

aalcalc < fifo/il_S1_summaryaalcalc_P13 > work/il_S1_aalcalc/P13.bin & pid13=$!

aalcalc < fifo/il_S1_summaryaalcalc_P14 > work/il_S1_aalcalc/P14.bin & pid14=$!

aalcalc < fifo/il_S1_summaryaalcalc_P15 > work/il_S1_aalcalc/P15.bin & pid15=$!

aalcalc < fifo/il_S1_summaryaalcalc_P16 > work/il_S1_aalcalc/P16.bin & pid16=$!

aalcalc < fifo/il_S1_summaryaalcalc_P17 > work/il_S1_aalcalc/P17.bin & pid17=$!

aalcalc < fifo/il_S1_summaryaalcalc_P18 > work/il_S1_aalcalc/P18.bin & pid18=$!

aalcalc < fifo/il_S1_summaryaalcalc_P19 > work/il_S1_aalcalc/P19.bin & pid19=$!

aalcalc < fifo/il_S1_summaryaalcalc_P20 > work/il_S1_aalcalc/P20.bin & pid20=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryaalcalc_P1 > /dev/null & pid21=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_summaryaalcalc_P2 > /dev/null & pid22=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summaryaalcalc_P3 > /dev/null & pid23=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summaryaalcalc_P4 > /dev/null & pid24=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summaryaalcalc_P5 > /dev/null & pid25=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_summaryaalcalc_P6 > /dev/null & pid26=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_summaryaalcalc_P7 > /dev/null & pid27=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_summaryaalcalc_P8 > /dev/null & pid28=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_summaryaalcalc_P9 > /dev/null & pid29=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_summaryaalcalc_P10 > /dev/null & pid30=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_summaryaalcalc_P11 > /dev/null & pid31=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_summaryaalcalc_P12 > /dev/null & pid32=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_summaryaalcalc_P13 > /dev/null & pid33=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_summaryaalcalc_P14 > /dev/null & pid34=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_summaryaalcalc_P15 > /dev/null & pid35=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_summaryaalcalc_P16 > /dev/null & pid36=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_summaryaalcalc_P17 > /dev/null & pid37=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_summaryaalcalc_P18 > /dev/null & pid38=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_summaryaalcalc_P19 > /dev/null & pid39=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_summaryaalcalc_P20 > /dev/null & pid40=$!
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

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40


# --- Do insured loss kats ---


aalsummary -Kil_S1_aalcalc > output/il_S1_aalcalc.csv & apid1=$!
wait $apid1


rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryaalcalc_P1

rm fifo/il_P2

rm fifo/il_S1_summary_P2
rm fifo/il_S1_summaryaalcalc_P2

rm fifo/il_P3

rm fifo/il_S1_summary_P3
rm fifo/il_S1_summaryaalcalc_P3

rm fifo/il_P4

rm fifo/il_S1_summary_P4
rm fifo/il_S1_summaryaalcalc_P4

rm fifo/il_P5

rm fifo/il_S1_summary_P5
rm fifo/il_S1_summaryaalcalc_P5

rm fifo/il_P6

rm fifo/il_S1_summary_P6
rm fifo/il_S1_summaryaalcalc_P6

rm fifo/il_P7

rm fifo/il_S1_summary_P7
rm fifo/il_S1_summaryaalcalc_P7

rm fifo/il_P8

rm fifo/il_S1_summary_P8
rm fifo/il_S1_summaryaalcalc_P8

rm fifo/il_P9

rm fifo/il_S1_summary_P9
rm fifo/il_S1_summaryaalcalc_P9

rm fifo/il_P10

rm fifo/il_S1_summary_P10
rm fifo/il_S1_summaryaalcalc_P10

rm fifo/il_P11

rm fifo/il_S1_summary_P11
rm fifo/il_S1_summaryaalcalc_P11

rm fifo/il_P12

rm fifo/il_S1_summary_P12
rm fifo/il_S1_summaryaalcalc_P12

rm fifo/il_P13

rm fifo/il_S1_summary_P13
rm fifo/il_S1_summaryaalcalc_P13

rm fifo/il_P14

rm fifo/il_S1_summary_P14
rm fifo/il_S1_summaryaalcalc_P14

rm fifo/il_P15

rm fifo/il_S1_summary_P15
rm fifo/il_S1_summaryaalcalc_P15

rm fifo/il_P16

rm fifo/il_S1_summary_P16
rm fifo/il_S1_summaryaalcalc_P16

rm fifo/il_P17

rm fifo/il_S1_summary_P17
rm fifo/il_S1_summaryaalcalc_P17

rm fifo/il_P18

rm fifo/il_S1_summary_P18
rm fifo/il_S1_summaryaalcalc_P18

rm fifo/il_P19

rm fifo/il_S1_summary_P19
rm fifo/il_S1_summaryaalcalc_P19

rm fifo/il_P20

rm fifo/il_S1_summary_P20
rm fifo/il_S1_summaryaalcalc_P20

rm -rf work/kat
rm work/il_S1_aalcalc/*
rmdir work/il_S1_aalcalc
