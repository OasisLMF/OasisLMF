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
mkdir -p output/full_correlation/

rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P20



# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid2=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid3=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid4=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid5=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid7=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid8=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid10=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 & pid11=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 & pid12=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 & pid13=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid14=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid15=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid16=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 & pid17=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid18=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 & pid19=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid20=$!


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 > /dev/null & pid21=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > /dev/null & pid22=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 > /dev/null & pid23=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11 > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12 > /dev/null & pid32=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13 > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14 > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15 > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16 > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17 > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18 > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19 > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20 > /dev/null & pid40=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/il_P4 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/il_P9 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/il_P10 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/il_P11 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/il_P12 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/il_P13 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/il_P14 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/il_P15 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/il_P16 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/il_P17 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/il_P18 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/il_P19 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/il_P20 &

# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid41=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid42=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid43=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid44=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 & pid45=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 & pid46=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid47=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 & pid48=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 & pid49=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 & pid50=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P11 > work/full_correlation/kat/il_S1_eltcalc_P11 & pid51=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P12 > work/full_correlation/kat/il_S1_eltcalc_P12 & pid52=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P13 > work/full_correlation/kat/il_S1_eltcalc_P13 & pid53=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P14 > work/full_correlation/kat/il_S1_eltcalc_P14 & pid54=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 & pid55=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P16 > work/full_correlation/kat/il_S1_eltcalc_P16 & pid56=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P17 > work/full_correlation/kat/il_S1_eltcalc_P17 & pid57=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P18 > work/full_correlation/kat/il_S1_eltcalc_P18 & pid58=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P19 > work/full_correlation/kat/il_S1_eltcalc_P19 & pid59=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 & pid60=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3 > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4 > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5 > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6 > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8 > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9 > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10 > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P11 > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P12 > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P13 > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P14 > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P15 > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P16 > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P17 > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P18 > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P19 > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P20 > /dev/null & pid80=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20 &

( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 ) & pid81=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 ) & pid82=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 ) & pid83=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 ) & pid84=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 ) & pid85=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 ) & pid86=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 ) & pid87=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 ) & pid88=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 ) & pid89=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 ) & pid90=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P11 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11 ) & pid91=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 ) & pid92=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P13 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13 ) & pid93=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P14 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14 ) & pid94=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P15 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15 ) & pid95=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P16 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16 ) & pid96=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P17 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17 ) & pid97=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P18 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18 ) & pid98=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P19 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19 ) & pid99=$!
( fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P20 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20 ) & pid100=$!
( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  ) & pid101=$!
( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  ) & pid102=$!
( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  ) & pid103=$!
( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  ) & pid104=$!
( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  ) & pid105=$!
( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  ) & pid106=$!
( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  ) & pid107=$!
( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  ) & pid108=$!
( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  ) & pid109=$!
( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  ) & pid110=$!
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P11 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P11  ) & pid111=$!
( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  ) & pid112=$!
( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P13 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P13  ) & pid113=$!
( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P14 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P14  ) & pid114=$!
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P15 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  ) & pid115=$!
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P16 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P16  ) & pid116=$!
( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P17 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P17  ) & pid117=$!
( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P18 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P18  ) & pid118=$!
( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P19 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P19  ) & pid119=$!
( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P20 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  ) & pid120=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 work/full_correlation/kat/il_S1_eltcalc_P11 work/full_correlation/kat/il_S1_eltcalc_P12 work/full_correlation/kat/il_S1_eltcalc_P13 work/full_correlation/kat/il_S1_eltcalc_P14 work/full_correlation/kat/il_S1_eltcalc_P15 work/full_correlation/kat/il_S1_eltcalc_P16 work/full_correlation/kat/il_S1_eltcalc_P17 work/full_correlation/kat/il_S1_eltcalc_P18 work/full_correlation/kat/il_S1_eltcalc_P19 work/full_correlation/kat/il_S1_eltcalc_P20 > output/full_correlation/il_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2


rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
