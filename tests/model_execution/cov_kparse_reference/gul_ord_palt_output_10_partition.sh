#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summary_palt
mkdir work/gul_S2_summary_palt

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2
mkfifo fifo/gul_P3
mkfifo fifo/gul_P4
mkfifo fifo/gul_P5
mkfifo fifo/gul_P6
mkfifo fifo/gul_P7
mkfifo fifo/gul_P8
mkfifo fifo/gul_P9
mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S2_summary_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S2_summary_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S2_summary_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S2_summary_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S2_summary_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S2_summary_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S2_summary_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S2_summary_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S2_summary_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S2_summary_P10



# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summary_palt/P1.bin > /dev/null & pid1=$!
tee < fifo/gul_S2_summary_P1 work/gul_S2_summary_palt/P1.bin > /dev/null & pid2=$!
tee < fifo/gul_S1_summary_P2 work/gul_S1_summary_palt/P2.bin > /dev/null & pid3=$!
tee < fifo/gul_S2_summary_P2 work/gul_S2_summary_palt/P2.bin > /dev/null & pid4=$!
tee < fifo/gul_S1_summary_P3 work/gul_S1_summary_palt/P3.bin > /dev/null & pid5=$!
tee < fifo/gul_S2_summary_P3 work/gul_S2_summary_palt/P3.bin > /dev/null & pid6=$!
tee < fifo/gul_S1_summary_P4 work/gul_S1_summary_palt/P4.bin > /dev/null & pid7=$!
tee < fifo/gul_S2_summary_P4 work/gul_S2_summary_palt/P4.bin > /dev/null & pid8=$!
tee < fifo/gul_S1_summary_P5 work/gul_S1_summary_palt/P5.bin > /dev/null & pid9=$!
tee < fifo/gul_S2_summary_P5 work/gul_S2_summary_palt/P5.bin > /dev/null & pid10=$!
tee < fifo/gul_S1_summary_P6 work/gul_S1_summary_palt/P6.bin > /dev/null & pid11=$!
tee < fifo/gul_S2_summary_P6 work/gul_S2_summary_palt/P6.bin > /dev/null & pid12=$!
tee < fifo/gul_S1_summary_P7 work/gul_S1_summary_palt/P7.bin > /dev/null & pid13=$!
tee < fifo/gul_S2_summary_P7 work/gul_S2_summary_palt/P7.bin > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P8 work/gul_S1_summary_palt/P8.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P8 work/gul_S2_summary_palt/P8.bin > /dev/null & pid16=$!
tee < fifo/gul_S1_summary_P9 work/gul_S1_summary_palt/P9.bin > /dev/null & pid17=$!
tee < fifo/gul_S2_summary_P9 work/gul_S2_summary_palt/P9.bin > /dev/null & pid18=$!
tee < fifo/gul_S1_summary_P10 work/gul_S1_summary_palt/P10.bin > /dev/null & pid19=$!
tee < fifo/gul_S2_summary_P10 work/gul_S2_summary_palt/P10.bin > /dev/null & pid20=$!

summarycalc -m -g  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 &
summarycalc -m -g  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 &

eve 1 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -c - > fifo/gul_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do ground up loss kats ---


aalcalc -Kgul_S1_summary_palt -o > output/gul_S1_palt.csv & lpid1=$!
aalcalc -Kgul_S2_summary_palt -o > output/gul_S2_palt.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
