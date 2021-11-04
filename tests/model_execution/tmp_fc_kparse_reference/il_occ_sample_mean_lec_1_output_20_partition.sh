#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

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
mkdir work/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryleccalc

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
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19.idx

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20.idx

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
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20.idx



# --- Do insured loss computes ---



tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid5=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 work/il_S1_summaryleccalc/P4.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 work/il_S1_summaryleccalc/P5.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid10=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid11=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid12=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 work/il_S1_summaryleccalc/P9.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 work/il_S1_summaryleccalc/P10.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid20=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 work/il_S1_summaryleccalc/P11.bin > /dev/null & pid21=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11.idx work/il_S1_summaryleccalc/P11.idx > /dev/null & pid22=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 work/il_S1_summaryleccalc/P12.bin > /dev/null & pid23=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12.idx work/il_S1_summaryleccalc/P12.idx > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 work/il_S1_summaryleccalc/P13.bin > /dev/null & pid25=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13.idx work/il_S1_summaryleccalc/P13.idx > /dev/null & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 work/il_S1_summaryleccalc/P14.bin > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14.idx work/il_S1_summaryleccalc/P14.idx > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 work/il_S1_summaryleccalc/P15.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15.idx work/il_S1_summaryleccalc/P15.idx > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 work/il_S1_summaryleccalc/P16.bin > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16.idx work/il_S1_summaryleccalc/P16.idx > /dev/null & pid32=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 work/il_S1_summaryleccalc/P17.bin > /dev/null & pid33=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17.idx work/il_S1_summaryleccalc/P17.idx > /dev/null & pid34=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 work/il_S1_summaryleccalc/P18.bin > /dev/null & pid35=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18.idx work/il_S1_summaryleccalc/P18.idx > /dev/null & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 work/il_S1_summaryleccalc/P19.bin > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19.idx work/il_S1_summaryleccalc/P19.idx > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 work/il_S1_summaryleccalc/P20.bin > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20.idx work/il_S1_summaryleccalc/P20.idx > /dev/null & pid40=$!

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



tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid41=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx work/full_correlation/il_S1_summaryleccalc/P1.idx > /dev/null & pid42=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid43=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2.idx work/full_correlation/il_S1_summaryleccalc/P2.idx > /dev/null & pid44=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 work/full_correlation/il_S1_summaryleccalc/P3.bin > /dev/null & pid45=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3.idx work/full_correlation/il_S1_summaryleccalc/P3.idx > /dev/null & pid46=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid47=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4.idx work/full_correlation/il_S1_summaryleccalc/P4.idx > /dev/null & pid48=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 work/full_correlation/il_S1_summaryleccalc/P5.bin > /dev/null & pid49=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5.idx work/full_correlation/il_S1_summaryleccalc/P5.idx > /dev/null & pid50=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 work/full_correlation/il_S1_summaryleccalc/P6.bin > /dev/null & pid51=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx work/full_correlation/il_S1_summaryleccalc/P6.idx > /dev/null & pid52=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid53=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7.idx work/full_correlation/il_S1_summaryleccalc/P7.idx > /dev/null & pid54=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 work/full_correlation/il_S1_summaryleccalc/P8.bin > /dev/null & pid55=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8.idx work/full_correlation/il_S1_summaryleccalc/P8.idx > /dev/null & pid56=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid57=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9.idx work/full_correlation/il_S1_summaryleccalc/P9.idx > /dev/null & pid58=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 work/full_correlation/il_S1_summaryleccalc/P10.bin > /dev/null & pid59=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10.idx work/full_correlation/il_S1_summaryleccalc/P10.idx > /dev/null & pid60=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11 work/full_correlation/il_S1_summaryleccalc/P11.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11.idx work/full_correlation/il_S1_summaryleccalc/P11.idx > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 work/full_correlation/il_S1_summaryleccalc/P12.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12.idx work/full_correlation/il_S1_summaryleccalc/P12.idx > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13 work/full_correlation/il_S1_summaryleccalc/P13.bin > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13.idx work/full_correlation/il_S1_summaryleccalc/P13.idx > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14 work/full_correlation/il_S1_summaryleccalc/P14.bin > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14.idx work/full_correlation/il_S1_summaryleccalc/P14.idx > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15 work/full_correlation/il_S1_summaryleccalc/P15.bin > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15.idx work/full_correlation/il_S1_summaryleccalc/P15.idx > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16 work/full_correlation/il_S1_summaryleccalc/P16.bin > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16.idx work/full_correlation/il_S1_summaryleccalc/P16.idx > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17 work/full_correlation/il_S1_summaryleccalc/P17.bin > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17.idx work/full_correlation/il_S1_summaryleccalc/P17.idx > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18 work/full_correlation/il_S1_summaryleccalc/P18.bin > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18.idx work/full_correlation/il_S1_summaryleccalc/P18.idx > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19 work/full_correlation/il_S1_summaryleccalc/P19.bin > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19.idx work/full_correlation/il_S1_summaryleccalc/P19.idx > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20 work/full_correlation/il_S1_summaryleccalc/P20.bin > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20.idx work/full_correlation/il_S1_summaryleccalc/P20.idx > /dev/null & pid80=$!

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

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P11 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P13 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P14 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P15 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P16 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P17 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P18 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P19 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P20 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20 &
eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P11 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P13 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P14 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P15 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P16 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P17 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P18 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P19 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P20 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---


leccalc -r -Kil_S1_summaryleccalc -s output/il_S1_leccalc_sample_mean_oep.csv & lpid1=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -s output/full_correlation/il_S1_leccalc_sample_mean_oep.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
