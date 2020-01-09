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

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


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
mkfifo fifo/il_P11
mkfifo fifo/il_P12
mkfifo fifo/il_P13
mkfifo fifo/il_P14
mkfifo fifo/il_P15
mkfifo fifo/il_P16
mkfifo fifo/il_P17
mkfifo fifo/il_P18
mkfifo fifo/il_P19
mkfifo fifo/il_P20

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summaryeltcalc_P2
mkfifo fifo/il_S1_eltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summaryeltcalc_P3
mkfifo fifo/il_S1_eltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summaryeltcalc_P4
mkfifo fifo/il_S1_eltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summaryeltcalc_P5
mkfifo fifo/il_S1_eltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summaryeltcalc_P6
mkfifo fifo/il_S1_eltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summaryeltcalc_P7
mkfifo fifo/il_S1_eltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summaryeltcalc_P8
mkfifo fifo/il_S1_eltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summaryeltcalc_P9
mkfifo fifo/il_S1_eltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summaryeltcalc_P10
mkfifo fifo/il_S1_eltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summaryeltcalc_P11
mkfifo fifo/il_S1_eltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summaryeltcalc_P12
mkfifo fifo/il_S1_eltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_summaryeltcalc_P13
mkfifo fifo/il_S1_eltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_summaryeltcalc_P14
mkfifo fifo/il_S1_eltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summaryeltcalc_P15
mkfifo fifo/il_S1_eltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summaryeltcalc_P16
mkfifo fifo/il_S1_eltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_summaryeltcalc_P17
mkfifo fifo/il_S1_eltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_summaryeltcalc_P18
mkfifo fifo/il_S1_eltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_summaryeltcalc_P19
mkfifo fifo/il_S1_eltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_summaryeltcalc_P20
mkfifo fifo/il_S1_eltcalc_P20

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P1
mkfifo fifo/full_correlation/il_S1_eltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P2
mkfifo fifo/full_correlation/il_S1_eltcalc_P2

mkfifo fifo/full_correlation/il_S1_summary_P3
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P3
mkfifo fifo/full_correlation/il_S1_eltcalc_P3

mkfifo fifo/full_correlation/il_S1_summary_P4
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P4
mkfifo fifo/full_correlation/il_S1_eltcalc_P4

mkfifo fifo/full_correlation/il_S1_summary_P5
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P5
mkfifo fifo/full_correlation/il_S1_eltcalc_P5

mkfifo fifo/full_correlation/il_S1_summary_P6
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P6
mkfifo fifo/full_correlation/il_S1_eltcalc_P6

mkfifo fifo/full_correlation/il_S1_summary_P7
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P7
mkfifo fifo/full_correlation/il_S1_eltcalc_P7

mkfifo fifo/full_correlation/il_S1_summary_P8
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P8
mkfifo fifo/full_correlation/il_S1_eltcalc_P8

mkfifo fifo/full_correlation/il_S1_summary_P9
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P9
mkfifo fifo/full_correlation/il_S1_eltcalc_P9

mkfifo fifo/full_correlation/il_S1_summary_P10
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P10
mkfifo fifo/full_correlation/il_S1_eltcalc_P10

mkfifo fifo/full_correlation/il_S1_summary_P11
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P11
mkfifo fifo/full_correlation/il_S1_eltcalc_P11

mkfifo fifo/full_correlation/il_S1_summary_P12
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P12
mkfifo fifo/full_correlation/il_S1_eltcalc_P12

mkfifo fifo/full_correlation/il_S1_summary_P13
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P13
mkfifo fifo/full_correlation/il_S1_eltcalc_P13

mkfifo fifo/full_correlation/il_S1_summary_P14
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P14
mkfifo fifo/full_correlation/il_S1_eltcalc_P14

mkfifo fifo/full_correlation/il_S1_summary_P15
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P15
mkfifo fifo/full_correlation/il_S1_eltcalc_P15

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P16
mkfifo fifo/full_correlation/il_S1_eltcalc_P16

mkfifo fifo/full_correlation/il_S1_summary_P17
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P17
mkfifo fifo/full_correlation/il_S1_eltcalc_P17

mkfifo fifo/full_correlation/il_S1_summary_P18
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P18
mkfifo fifo/full_correlation/il_S1_eltcalc_P18

mkfifo fifo/full_correlation/il_S1_summary_P19
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P19
mkfifo fifo/full_correlation/il_S1_eltcalc_P19

mkfifo fifo/full_correlation/il_S1_summary_P20
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P20
mkfifo fifo/full_correlation/il_S1_eltcalc_P20



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid2=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid3=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid4=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid5=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid6=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid7=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid8=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid9=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid10=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P11 > work/kat/il_S1_eltcalc_P11 & pid11=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P12 > work/kat/il_S1_eltcalc_P12 & pid12=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P13 > work/kat/il_S1_eltcalc_P13 & pid13=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid14=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid15=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid16=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P17 > work/kat/il_S1_eltcalc_P17 & pid17=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid18=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P19 > work/kat/il_S1_eltcalc_P19 & pid19=$!
eltcalc -s < fifo/il_S1_summaryeltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid20=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 > /dev/null & pid21=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_summaryeltcalc_P2 > /dev/null & pid22=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summaryeltcalc_P3 > /dev/null & pid23=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summaryeltcalc_P4 > /dev/null & pid24=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summaryeltcalc_P5 > /dev/null & pid25=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_summaryeltcalc_P6 > /dev/null & pid26=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_summaryeltcalc_P7 > /dev/null & pid27=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_summaryeltcalc_P8 > /dev/null & pid28=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_summaryeltcalc_P9 > /dev/null & pid29=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_summaryeltcalc_P10 > /dev/null & pid30=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_summaryeltcalc_P11 > /dev/null & pid31=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_summaryeltcalc_P12 > /dev/null & pid32=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_summaryeltcalc_P13 > /dev/null & pid33=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_summaryeltcalc_P14 > /dev/null & pid34=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_summaryeltcalc_P15 > /dev/null & pid35=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_summaryeltcalc_P16 > /dev/null & pid36=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_summaryeltcalc_P17 > /dev/null & pid37=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_summaryeltcalc_P18 > /dev/null & pid38=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_summaryeltcalc_P19 > /dev/null & pid39=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_summaryeltcalc_P20 > /dev/null & pid40=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarycalc -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarycalc -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarycalc -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarycalc -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarycalc -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarycalc -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 &
summarycalc -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 &
summarycalc -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 &
summarycalc -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 &
summarycalc -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 &
summarycalc -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 &
summarycalc -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 &
summarycalc -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 &
summarycalc -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &
summarycalc -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 &
summarycalc -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 &
summarycalc -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 &
summarycalc -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 &

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - | fmcalc -a2 > fifo/il_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P2 -a1 -i - | fmcalc -a2 > fifo/il_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P3 -a1 -i - | fmcalc -a2 > fifo/il_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P4 -a1 -i - | fmcalc -a2 > fifo/il_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P5 -a1 -i - | fmcalc -a2 > fifo/il_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P6 -a1 -i - | fmcalc -a2 > fifo/il_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P7 -a1 -i - | fmcalc -a2 > fifo/il_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P8 -a1 -i - | fmcalc -a2 > fifo/il_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P9 -a1 -i - | fmcalc -a2 > fifo/il_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P10 -a1 -i - | fmcalc -a2 > fifo/il_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P11 -a1 -i - | fmcalc -a2 > fifo/il_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P12 -a1 -i - | fmcalc -a2 > fifo/il_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P13 -a1 -i - | fmcalc -a2 > fifo/il_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P14 -a1 -i - | fmcalc -a2 > fifo/il_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P15 -a1 -i - | fmcalc -a2 > fifo/il_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P16 -a1 -i - | fmcalc -a2 > fifo/il_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P17 -a1 -i - | fmcalc -a2 > fifo/il_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P18 -a1 -i - | fmcalc -a2 > fifo/il_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P19 -a1 -i - | fmcalc -a2 > fifo/il_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P20 -a1 -i - | fmcalc -a2 > fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40

# --- Do computes for fully correlated output ---

fmcalc-a2 < fifo/full_correlation/gul_P1 > fifo/full_correlation/il_P1 & fcpid1=$!
fmcalc-a2 < fifo/full_correlation/gul_P2 > fifo/full_correlation/il_P2 & fcpid2=$!
fmcalc-a2 < fifo/full_correlation/gul_P3 > fifo/full_correlation/il_P3 & fcpid3=$!
fmcalc-a2 < fifo/full_correlation/gul_P4 > fifo/full_correlation/il_P4 & fcpid4=$!
fmcalc-a2 < fifo/full_correlation/gul_P5 > fifo/full_correlation/il_P5 & fcpid5=$!
fmcalc-a2 < fifo/full_correlation/gul_P6 > fifo/full_correlation/il_P6 & fcpid6=$!
fmcalc-a2 < fifo/full_correlation/gul_P7 > fifo/full_correlation/il_P7 & fcpid7=$!
fmcalc-a2 < fifo/full_correlation/gul_P8 > fifo/full_correlation/il_P8 & fcpid8=$!
fmcalc-a2 < fifo/full_correlation/gul_P9 > fifo/full_correlation/il_P9 & fcpid9=$!
fmcalc-a2 < fifo/full_correlation/gul_P10 > fifo/full_correlation/il_P10 & fcpid10=$!
fmcalc-a2 < fifo/full_correlation/gul_P11 > fifo/full_correlation/il_P11 & fcpid11=$!
fmcalc-a2 < fifo/full_correlation/gul_P12 > fifo/full_correlation/il_P12 & fcpid12=$!
fmcalc-a2 < fifo/full_correlation/gul_P13 > fifo/full_correlation/il_P13 & fcpid13=$!
fmcalc-a2 < fifo/full_correlation/gul_P14 > fifo/full_correlation/il_P14 & fcpid14=$!
fmcalc-a2 < fifo/full_correlation/gul_P15 > fifo/full_correlation/il_P15 & fcpid15=$!
fmcalc-a2 < fifo/full_correlation/gul_P16 > fifo/full_correlation/il_P16 & fcpid16=$!
fmcalc-a2 < fifo/full_correlation/gul_P17 > fifo/full_correlation/il_P17 & fcpid17=$!
fmcalc-a2 < fifo/full_correlation/gul_P18 > fifo/full_correlation/il_P18 & fcpid18=$!
fmcalc-a2 < fifo/full_correlation/gul_P19 > fifo/full_correlation/il_P19 & fcpid19=$!
fmcalc-a2 < fifo/full_correlation/gul_P20 > fifo/full_correlation/il_P20 & fcpid20=$!

wait $fcpid1 $fcpid2 $fcpid3 $fcpid4 $fcpid5 $fcpid6 $fcpid7 $fcpid8 $fcpid9 $fcpid10 $fcpid11 $fcpid12 $fcpid13 $fcpid14 $fcpid15 $fcpid16 $fcpid17 $fcpid18 $fcpid19 $fcpid20


# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_summaryeltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid1=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid2=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid3=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid4=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 & pid5=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 & pid6=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid7=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 & pid8=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 & pid9=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 & pid10=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P11 > work/full_correlation/kat/il_S1_eltcalc_P11 & pid11=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P12 > work/full_correlation/kat/il_S1_eltcalc_P12 & pid12=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P13 > work/full_correlation/kat/il_S1_eltcalc_P13 & pid13=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P14 > work/full_correlation/kat/il_S1_eltcalc_P14 & pid14=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 & pid15=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P16 > work/full_correlation/kat/il_S1_eltcalc_P16 & pid16=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P17 > work/full_correlation/kat/il_S1_eltcalc_P17 & pid17=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P18 > work/full_correlation/kat/il_S1_eltcalc_P18 & pid18=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P19 > work/full_correlation/kat/il_S1_eltcalc_P19 & pid19=$!
eltcalc -s < fifo/full_correlation/il_S1_summaryeltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 & pid20=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_summaryeltcalc_P1 > /dev/null & pid21=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_summaryeltcalc_P2 > /dev/null & pid22=$!
tee < fifo/full_correlation/il_S1_summary_P3 fifo/full_correlation/il_S1_summaryeltcalc_P3 > /dev/null & pid23=$!
tee < fifo/full_correlation/il_S1_summary_P4 fifo/full_correlation/il_S1_summaryeltcalc_P4 > /dev/null & pid24=$!
tee < fifo/full_correlation/il_S1_summary_P5 fifo/full_correlation/il_S1_summaryeltcalc_P5 > /dev/null & pid25=$!
tee < fifo/full_correlation/il_S1_summary_P6 fifo/full_correlation/il_S1_summaryeltcalc_P6 > /dev/null & pid26=$!
tee < fifo/full_correlation/il_S1_summary_P7 fifo/full_correlation/il_S1_summaryeltcalc_P7 > /dev/null & pid27=$!
tee < fifo/full_correlation/il_S1_summary_P8 fifo/full_correlation/il_S1_summaryeltcalc_P8 > /dev/null & pid28=$!
tee < fifo/full_correlation/il_S1_summary_P9 fifo/full_correlation/il_S1_summaryeltcalc_P9 > /dev/null & pid29=$!
tee < fifo/full_correlation/il_S1_summary_P10 fifo/full_correlation/il_S1_summaryeltcalc_P10 > /dev/null & pid30=$!
tee < fifo/full_correlation/il_S1_summary_P11 fifo/full_correlation/il_S1_summaryeltcalc_P11 > /dev/null & pid31=$!
tee < fifo/full_correlation/il_S1_summary_P12 fifo/full_correlation/il_S1_summaryeltcalc_P12 > /dev/null & pid32=$!
tee < fifo/full_correlation/il_S1_summary_P13 fifo/full_correlation/il_S1_summaryeltcalc_P13 > /dev/null & pid33=$!
tee < fifo/full_correlation/il_S1_summary_P14 fifo/full_correlation/il_S1_summaryeltcalc_P14 > /dev/null & pid34=$!
tee < fifo/full_correlation/il_S1_summary_P15 fifo/full_correlation/il_S1_summaryeltcalc_P15 > /dev/null & pid35=$!
tee < fifo/full_correlation/il_S1_summary_P16 fifo/full_correlation/il_S1_summaryeltcalc_P16 > /dev/null & pid36=$!
tee < fifo/full_correlation/il_S1_summary_P17 fifo/full_correlation/il_S1_summaryeltcalc_P17 > /dev/null & pid37=$!
tee < fifo/full_correlation/il_S1_summary_P18 fifo/full_correlation/il_S1_summaryeltcalc_P18 > /dev/null & pid38=$!
tee < fifo/full_correlation/il_S1_summary_P19 fifo/full_correlation/il_S1_summaryeltcalc_P19 > /dev/null & pid39=$!
tee < fifo/full_correlation/il_S1_summary_P20 fifo/full_correlation/il_S1_summaryeltcalc_P20 > /dev/null & pid40=$!

summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P2 < fifo/full_correlation/il_P2 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P3 < fifo/full_correlation/il_P3 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P4 < fifo/full_correlation/il_P4 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P5 < fifo/full_correlation/il_P5 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P6 < fifo/full_correlation/il_P6 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P7 < fifo/full_correlation/il_P7 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P8 < fifo/full_correlation/il_P8 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P9 < fifo/full_correlation/il_P9 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P10 < fifo/full_correlation/il_P10 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P11 < fifo/full_correlation/il_P11 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P12 < fifo/full_correlation/il_P12 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P13 < fifo/full_correlation/il_P13 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P14 < fifo/full_correlation/il_P14 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P15 < fifo/full_correlation/il_P15 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P17 < fifo/full_correlation/il_P17 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P18 < fifo/full_correlation/il_P18 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P19 < fifo/full_correlation/il_P19 &
summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P20 < fifo/full_correlation/il_P20 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 work/full_correlation/kat/il_S1_eltcalc_P11 work/full_correlation/kat/il_S1_eltcalc_P12 work/full_correlation/kat/il_S1_eltcalc_P13 work/full_correlation/kat/il_S1_eltcalc_P14 work/full_correlation/kat/il_S1_eltcalc_P15 work/full_correlation/kat/il_S1_eltcalc_P16 work/full_correlation/kat/il_S1_eltcalc_P17 work/full_correlation/kat/il_S1_eltcalc_P18 work/full_correlation/kat/il_S1_eltcalc_P19 work/full_correlation/kat/il_S1_eltcalc_P20 > output/full_correlation/il_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2


