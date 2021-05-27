#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S2_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S2_summaryleccalc

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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9.idx

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10.idx



# --- Do ground up loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3 work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx work/gul_S2_summaryleccalc/P3.idx > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4 work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx work/gul_S2_summaryleccalc/P4.idx > /dev/null & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5 work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx work/gul_S2_summaryleccalc/P5.idx > /dev/null & pid20=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid21=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid22=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6 work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid23=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx work/gul_S2_summaryleccalc/P6.idx > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid32=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9 work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9.idx work/gul_S2_summaryleccalc/P9.idx > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10 work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10.idx work/gul_S2_summaryleccalc/P10.idx > /dev/null & pid40=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/gul_P3 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/gul_P5 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/gul_P6 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/gul_P9 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10 < /tmp/%FIFO_DIR%/fifo/gul_P10 &

# --- Do ground up loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid41=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid42=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1 work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid43=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1.idx work/full_correlation/gul_S2_summaryleccalc/P1.idx > /dev/null & pid44=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid45=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryleccalc/P2.idx > /dev/null & pid46=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2 work/full_correlation/gul_S2_summaryleccalc/P2.bin > /dev/null & pid47=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2.idx work/full_correlation/gul_S2_summaryleccalc/P2.idx > /dev/null & pid48=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid49=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3.idx work/full_correlation/gul_S1_summaryleccalc/P3.idx > /dev/null & pid50=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3 work/full_correlation/gul_S2_summaryleccalc/P3.bin > /dev/null & pid51=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3.idx work/full_correlation/gul_S2_summaryleccalc/P3.idx > /dev/null & pid52=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid53=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4.idx work/full_correlation/gul_S1_summaryleccalc/P4.idx > /dev/null & pid54=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4 work/full_correlation/gul_S2_summaryleccalc/P4.bin > /dev/null & pid55=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4.idx work/full_correlation/gul_S2_summaryleccalc/P4.idx > /dev/null & pid56=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid57=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5.idx work/full_correlation/gul_S1_summaryleccalc/P5.idx > /dev/null & pid58=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5 work/full_correlation/gul_S2_summaryleccalc/P5.bin > /dev/null & pid59=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5.idx work/full_correlation/gul_S2_summaryleccalc/P5.idx > /dev/null & pid60=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6.idx work/full_correlation/gul_S1_summaryleccalc/P6.idx > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6 work/full_correlation/gul_S2_summaryleccalc/P6.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6.idx work/full_correlation/gul_S2_summaryleccalc/P6.idx > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryleccalc/P7.idx > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7 work/full_correlation/gul_S2_summaryleccalc/P7.bin > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7.idx work/full_correlation/gul_S2_summaryleccalc/P7.idx > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8.idx work/full_correlation/gul_S1_summaryleccalc/P8.idx > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8 work/full_correlation/gul_S2_summaryleccalc/P8.bin > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8.idx work/full_correlation/gul_S2_summaryleccalc/P8.idx > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9.idx work/full_correlation/gul_S1_summaryleccalc/P9.idx > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9 work/full_correlation/gul_S2_summaryleccalc/P9.bin > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9.idx work/full_correlation/gul_S2_summaryleccalc/P9.idx > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10.idx work/full_correlation/gul_S1_summaryleccalc/P10.idx > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10 work/full_correlation/gul_S2_summaryleccalc/P10.bin > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10.idx work/full_correlation/gul_S2_summaryleccalc/P10.idx > /dev/null & pid80=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 &

eve 1 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


ordleccalc  -Kgul_S1_summaryleccalc -W -w -o output/gul_S1_psept.csv & lpid1=$!
ordleccalc -r -Kgul_S2_summaryleccalc -W -w -o output/gul_S2_psept.csv & lpid2=$!
ordleccalc  -Kfull_correlation/gul_S1_summaryleccalc -W -w -o output/full_correlation/gul_S1_psept.csv & lpid3=$!
ordleccalc -r -Kfull_correlation/gul_S2_summaryleccalc -W -w -o output/full_correlation/gul_S2_psept.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
