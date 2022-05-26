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

rm -R -f work/*
mkdir -p work/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

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
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P21
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P22
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P23
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P24
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P25
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P26
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P27
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P28
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P29
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P31
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P32
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P33
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P34
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P35
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P37
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P40

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P21
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P21.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P21
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P21
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P21

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P22
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P22.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P22
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P22
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P22

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P23
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P23.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P23
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P23
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P23

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P24
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P24.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P24
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P24
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P24

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P25
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P25.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P25
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P25
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P25

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P26
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P26.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P26
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P26
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P26

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P27
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P27.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P27
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P27
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P27

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P28
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P28.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P28
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P28
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P28

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P29
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P29.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P29
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P29
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P29

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P31
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P31.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P31
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P31
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P31

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P32
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P32.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P32
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P32
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P32

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P33
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P33.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P33
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P33
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P33

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P34
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P34.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P34
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P34
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P34

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P35
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P35.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P35
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P35
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P35

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P37
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P37.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P37
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P37
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P37

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P40
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P40.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P40
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P40
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P40

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
mkfifo /tmp/%FIFO_DIR%/fifo/il_P21
mkfifo /tmp/%FIFO_DIR%/fifo/il_P22
mkfifo /tmp/%FIFO_DIR%/fifo/il_P23
mkfifo /tmp/%FIFO_DIR%/fifo/il_P24
mkfifo /tmp/%FIFO_DIR%/fifo/il_P25
mkfifo /tmp/%FIFO_DIR%/fifo/il_P26
mkfifo /tmp/%FIFO_DIR%/fifo/il_P27
mkfifo /tmp/%FIFO_DIR%/fifo/il_P28
mkfifo /tmp/%FIFO_DIR%/fifo/il_P29
mkfifo /tmp/%FIFO_DIR%/fifo/il_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_P31
mkfifo /tmp/%FIFO_DIR%/fifo/il_P32
mkfifo /tmp/%FIFO_DIR%/fifo/il_P33
mkfifo /tmp/%FIFO_DIR%/fifo/il_P34
mkfifo /tmp/%FIFO_DIR%/fifo/il_P35
mkfifo /tmp/%FIFO_DIR%/fifo/il_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_P37
mkfifo /tmp/%FIFO_DIR%/fifo/il_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_P40

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P21
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P21.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P21
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P21
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P21

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P22
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P22.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P22
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P22
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P22

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P23
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P23.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P23
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P23
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P23

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P24
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P24.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P24
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P24
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P24

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P25
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P25.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P25
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P25
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P25

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P26
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P26.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P26
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P26
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P26

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P27
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P27.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P27
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P27
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P27

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P28
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P28.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P28
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P28
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P28

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P29
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P29.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P29
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P29
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P29

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P31
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P31.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P31
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P31
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P31

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P32
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P32.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P32
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P32
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P32

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P33
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P33.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P33
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P33
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P33

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P34
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P34.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P34
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P34
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P34

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P35
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P35.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P35
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P35
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P35

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P37
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P37.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P37
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P37
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P37

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P40
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P40.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P40
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P40
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P40



# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid4=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid5=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid8=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid11=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid12=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid13=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid14=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid15=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid17=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid18=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid19=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid20=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid21=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid22=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid23=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid24=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid25=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid26=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid27=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid28=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid29=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid30=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 & pid31=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 & pid32=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 & pid33=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 & pid34=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P12 > work/kat/il_S1_summarycalc_P12 & pid35=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 & pid36=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 & pid37=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P13 > work/kat/il_S1_summarycalc_P13 & pid38=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 & pid39=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid40=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P14 > work/kat/il_S1_summarycalc_P14 & pid41=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 & pid42=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid43=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P15 > work/kat/il_S1_summarycalc_P15 & pid44=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid45=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid46=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P16 > work/kat/il_S1_summarycalc_P16 & pid47=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 & pid48=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 & pid49=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 & pid50=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 & pid51=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid52=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 & pid53=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 & pid54=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 & pid55=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P19 > work/kat/il_S1_summarycalc_P19 & pid56=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 & pid57=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid58=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P20 > work/kat/il_S1_summarycalc_P20 & pid59=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 & pid60=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P21 > work/kat/il_S1_eltcalc_P21 & pid61=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P21 > work/kat/il_S1_summarycalc_P21 & pid62=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P21 > work/kat/il_S1_pltcalc_P21 & pid63=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P22 > work/kat/il_S1_eltcalc_P22 & pid64=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P22 > work/kat/il_S1_summarycalc_P22 & pid65=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P22 > work/kat/il_S1_pltcalc_P22 & pid66=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P23 > work/kat/il_S1_eltcalc_P23 & pid67=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P23 > work/kat/il_S1_summarycalc_P23 & pid68=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P23 > work/kat/il_S1_pltcalc_P23 & pid69=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P24 > work/kat/il_S1_eltcalc_P24 & pid70=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P24 > work/kat/il_S1_summarycalc_P24 & pid71=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P24 > work/kat/il_S1_pltcalc_P24 & pid72=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P25 > work/kat/il_S1_eltcalc_P25 & pid73=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P25 > work/kat/il_S1_summarycalc_P25 & pid74=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P25 > work/kat/il_S1_pltcalc_P25 & pid75=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P26 > work/kat/il_S1_eltcalc_P26 & pid76=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P26 > work/kat/il_S1_summarycalc_P26 & pid77=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P26 > work/kat/il_S1_pltcalc_P26 & pid78=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P27 > work/kat/il_S1_eltcalc_P27 & pid79=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P27 > work/kat/il_S1_summarycalc_P27 & pid80=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P27 > work/kat/il_S1_pltcalc_P27 & pid81=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P28 > work/kat/il_S1_eltcalc_P28 & pid82=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P28 > work/kat/il_S1_summarycalc_P28 & pid83=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P28 > work/kat/il_S1_pltcalc_P28 & pid84=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P29 > work/kat/il_S1_eltcalc_P29 & pid85=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P29 > work/kat/il_S1_summarycalc_P29 & pid86=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P29 > work/kat/il_S1_pltcalc_P29 & pid87=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30 > work/kat/il_S1_eltcalc_P30 & pid88=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30 > work/kat/il_S1_summarycalc_P30 & pid89=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30 > work/kat/il_S1_pltcalc_P30 & pid90=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P31 > work/kat/il_S1_eltcalc_P31 & pid91=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P31 > work/kat/il_S1_summarycalc_P31 & pid92=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P31 > work/kat/il_S1_pltcalc_P31 & pid93=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P32 > work/kat/il_S1_eltcalc_P32 & pid94=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P32 > work/kat/il_S1_summarycalc_P32 & pid95=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P32 > work/kat/il_S1_pltcalc_P32 & pid96=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P33 > work/kat/il_S1_eltcalc_P33 & pid97=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P33 > work/kat/il_S1_summarycalc_P33 & pid98=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P33 > work/kat/il_S1_pltcalc_P33 & pid99=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P34 > work/kat/il_S1_eltcalc_P34 & pid100=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P34 > work/kat/il_S1_summarycalc_P34 & pid101=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P34 > work/kat/il_S1_pltcalc_P34 & pid102=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P35 > work/kat/il_S1_eltcalc_P35 & pid103=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P35 > work/kat/il_S1_summarycalc_P35 & pid104=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P35 > work/kat/il_S1_pltcalc_P35 & pid105=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36 > work/kat/il_S1_eltcalc_P36 & pid106=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36 > work/kat/il_S1_summarycalc_P36 & pid107=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36 > work/kat/il_S1_pltcalc_P36 & pid108=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P37 > work/kat/il_S1_eltcalc_P37 & pid109=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P37 > work/kat/il_S1_summarycalc_P37 & pid110=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P37 > work/kat/il_S1_pltcalc_P37 & pid111=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38 > work/kat/il_S1_eltcalc_P38 & pid112=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38 > work/kat/il_S1_summarycalc_P38 & pid113=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38 > work/kat/il_S1_pltcalc_P38 & pid114=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39 > work/kat/il_S1_eltcalc_P39 & pid115=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39 > work/kat/il_S1_summarycalc_P39 & pid116=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39 > work/kat/il_S1_pltcalc_P39 & pid117=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P40 > work/kat/il_S1_eltcalc_P40 & pid118=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P40 > work/kat/il_S1_summarycalc_P40 & pid119=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P40 > work/kat/il_S1_pltcalc_P40 & pid120=$!


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid121=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid122=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid123=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid124=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid125=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx work/il_S1_summaryaalcalc/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid126=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid127=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx work/il_S1_summaryaalcalc/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid128=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid129=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx work/il_S1_summaryaalcalc/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid130=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid131=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx work/il_S1_summaryaalcalc/P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid132=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid133=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryaalcalc/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid134=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid135=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx work/il_S1_summaryaalcalc/P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid136=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid137=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx work/il_S1_summaryaalcalc/P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid138=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid139=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx work/il_S1_summaryaalcalc/P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid140=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P11 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid141=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11.idx work/il_S1_summaryaalcalc/P11.idx work/il_S1_summaryleccalc/P11.idx > /dev/null & pid142=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P12 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid143=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12.idx work/il_S1_summaryaalcalc/P12.idx work/il_S1_summaryleccalc/P12.idx > /dev/null & pid144=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P13 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid145=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13.idx work/il_S1_summaryaalcalc/P13.idx work/il_S1_summaryleccalc/P13.idx > /dev/null & pid146=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P14 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid147=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14.idx work/il_S1_summaryaalcalc/P14.idx work/il_S1_summaryleccalc/P14.idx > /dev/null & pid148=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid149=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15.idx work/il_S1_summaryaalcalc/P15.idx work/il_S1_summaryleccalc/P15.idx > /dev/null & pid150=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P16 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid151=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16.idx work/il_S1_summaryaalcalc/P16.idx work/il_S1_summaryleccalc/P16.idx > /dev/null & pid152=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P17 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid153=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17.idx work/il_S1_summaryaalcalc/P17.idx work/il_S1_summaryleccalc/P17.idx > /dev/null & pid154=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid155=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18.idx work/il_S1_summaryaalcalc/P18.idx work/il_S1_summaryleccalc/P18.idx > /dev/null & pid156=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P19 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid157=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19.idx work/il_S1_summaryaalcalc/P19.idx work/il_S1_summaryleccalc/P19.idx > /dev/null & pid158=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid159=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20.idx work/il_S1_summaryaalcalc/P20.idx work/il_S1_summaryleccalc/P20.idx > /dev/null & pid160=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P21 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P21 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P21 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P21 work/il_S1_summaryaalcalc/P21.bin work/il_S1_summaryleccalc/P21.bin > /dev/null & pid161=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P21.idx work/il_S1_summaryaalcalc/P21.idx work/il_S1_summaryleccalc/P21.idx > /dev/null & pid162=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P22 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P22 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P22 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P22 work/il_S1_summaryaalcalc/P22.bin work/il_S1_summaryleccalc/P22.bin > /dev/null & pid163=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P22.idx work/il_S1_summaryaalcalc/P22.idx work/il_S1_summaryleccalc/P22.idx > /dev/null & pid164=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P23 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P23 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P23 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P23 work/il_S1_summaryaalcalc/P23.bin work/il_S1_summaryleccalc/P23.bin > /dev/null & pid165=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P23.idx work/il_S1_summaryaalcalc/P23.idx work/il_S1_summaryleccalc/P23.idx > /dev/null & pid166=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P24 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P24 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P24 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P24 work/il_S1_summaryaalcalc/P24.bin work/il_S1_summaryleccalc/P24.bin > /dev/null & pid167=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P24.idx work/il_S1_summaryaalcalc/P24.idx work/il_S1_summaryleccalc/P24.idx > /dev/null & pid168=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P25 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P25 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P25 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P25 work/il_S1_summaryaalcalc/P25.bin work/il_S1_summaryleccalc/P25.bin > /dev/null & pid169=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P25.idx work/il_S1_summaryaalcalc/P25.idx work/il_S1_summaryleccalc/P25.idx > /dev/null & pid170=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P26 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P26 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P26 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P26 work/il_S1_summaryaalcalc/P26.bin work/il_S1_summaryleccalc/P26.bin > /dev/null & pid171=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P26.idx work/il_S1_summaryaalcalc/P26.idx work/il_S1_summaryleccalc/P26.idx > /dev/null & pid172=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P27 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P27 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P27 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P27 work/il_S1_summaryaalcalc/P27.bin work/il_S1_summaryleccalc/P27.bin > /dev/null & pid173=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P27.idx work/il_S1_summaryaalcalc/P27.idx work/il_S1_summaryleccalc/P27.idx > /dev/null & pid174=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P28 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P28 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P28 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P28 work/il_S1_summaryaalcalc/P28.bin work/il_S1_summaryleccalc/P28.bin > /dev/null & pid175=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P28.idx work/il_S1_summaryaalcalc/P28.idx work/il_S1_summaryleccalc/P28.idx > /dev/null & pid176=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P29 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P29 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P29 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P29 work/il_S1_summaryaalcalc/P29.bin work/il_S1_summaryleccalc/P29.bin > /dev/null & pid177=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P29.idx work/il_S1_summaryaalcalc/P29.idx work/il_S1_summaryleccalc/P29.idx > /dev/null & pid178=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30 work/il_S1_summaryaalcalc/P30.bin work/il_S1_summaryleccalc/P30.bin > /dev/null & pid179=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30.idx work/il_S1_summaryaalcalc/P30.idx work/il_S1_summaryleccalc/P30.idx > /dev/null & pid180=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P31 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P31 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P31 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P31 work/il_S1_summaryaalcalc/P31.bin work/il_S1_summaryleccalc/P31.bin > /dev/null & pid181=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P31.idx work/il_S1_summaryaalcalc/P31.idx work/il_S1_summaryleccalc/P31.idx > /dev/null & pid182=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P32 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P32 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P32 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P32 work/il_S1_summaryaalcalc/P32.bin work/il_S1_summaryleccalc/P32.bin > /dev/null & pid183=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P32.idx work/il_S1_summaryaalcalc/P32.idx work/il_S1_summaryleccalc/P32.idx > /dev/null & pid184=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P33 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P33 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P33 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P33 work/il_S1_summaryaalcalc/P33.bin work/il_S1_summaryleccalc/P33.bin > /dev/null & pid185=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P33.idx work/il_S1_summaryaalcalc/P33.idx work/il_S1_summaryleccalc/P33.idx > /dev/null & pid186=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P34 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P34 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P34 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P34 work/il_S1_summaryaalcalc/P34.bin work/il_S1_summaryleccalc/P34.bin > /dev/null & pid187=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P34.idx work/il_S1_summaryaalcalc/P34.idx work/il_S1_summaryleccalc/P34.idx > /dev/null & pid188=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P35 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P35 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P35 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P35 work/il_S1_summaryaalcalc/P35.bin work/il_S1_summaryleccalc/P35.bin > /dev/null & pid189=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P35.idx work/il_S1_summaryaalcalc/P35.idx work/il_S1_summaryleccalc/P35.idx > /dev/null & pid190=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36 work/il_S1_summaryaalcalc/P36.bin work/il_S1_summaryleccalc/P36.bin > /dev/null & pid191=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36.idx work/il_S1_summaryaalcalc/P36.idx work/il_S1_summaryleccalc/P36.idx > /dev/null & pid192=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P37 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P37 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P37 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P37 work/il_S1_summaryaalcalc/P37.bin work/il_S1_summaryleccalc/P37.bin > /dev/null & pid193=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P37.idx work/il_S1_summaryaalcalc/P37.idx work/il_S1_summaryleccalc/P37.idx > /dev/null & pid194=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38 work/il_S1_summaryaalcalc/P38.bin work/il_S1_summaryleccalc/P38.bin > /dev/null & pid195=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38.idx work/il_S1_summaryaalcalc/P38.idx work/il_S1_summaryleccalc/P38.idx > /dev/null & pid196=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39 work/il_S1_summaryaalcalc/P39.bin work/il_S1_summaryleccalc/P39.bin > /dev/null & pid197=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39.idx work/il_S1_summaryaalcalc/P39.idx work/il_S1_summaryleccalc/P39.idx > /dev/null & pid198=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P40 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P40 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P40 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P40 work/il_S1_summaryaalcalc/P40.bin work/il_S1_summaryleccalc/P40.bin > /dev/null & pid199=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P40.idx work/il_S1_summaryaalcalc/P40.idx work/il_S1_summaryleccalc/P40.idx > /dev/null & pid200=$!

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
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P21 < /tmp/%FIFO_DIR%/fifo/il_P21 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P22 < /tmp/%FIFO_DIR%/fifo/il_P22 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P23 < /tmp/%FIFO_DIR%/fifo/il_P23 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P24 < /tmp/%FIFO_DIR%/fifo/il_P24 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P25 < /tmp/%FIFO_DIR%/fifo/il_P25 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P26 < /tmp/%FIFO_DIR%/fifo/il_P26 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P27 < /tmp/%FIFO_DIR%/fifo/il_P27 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P28 < /tmp/%FIFO_DIR%/fifo/il_P28 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P29 < /tmp/%FIFO_DIR%/fifo/il_P29 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/il_P30 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P31 < /tmp/%FIFO_DIR%/fifo/il_P31 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P32 < /tmp/%FIFO_DIR%/fifo/il_P32 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P33 < /tmp/%FIFO_DIR%/fifo/il_P33 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P34 < /tmp/%FIFO_DIR%/fifo/il_P34 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P35 < /tmp/%FIFO_DIR%/fifo/il_P35 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36 < /tmp/%FIFO_DIR%/fifo/il_P36 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P37 < /tmp/%FIFO_DIR%/fifo/il_P37 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38 < /tmp/%FIFO_DIR%/fifo/il_P38 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/il_P39 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P40 < /tmp/%FIFO_DIR%/fifo/il_P40 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid201=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid202=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid203=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid204=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid205=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid206=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid207=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid208=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid209=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid210=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid211=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid212=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid213=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid214=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid215=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid216=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid217=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid218=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid219=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid220=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid221=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid222=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid223=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid224=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid225=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid226=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid227=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid228=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid229=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid230=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 & pid231=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 & pid232=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 & pid233=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 & pid234=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 & pid235=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P12 > work/kat/gul_S1_pltcalc_P12 & pid236=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 & pid237=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 & pid238=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P13 > work/kat/gul_S1_pltcalc_P13 & pid239=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 & pid240=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 & pid241=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 & pid242=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 & pid243=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid244=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P15 > work/kat/gul_S1_pltcalc_P15 & pid245=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 & pid246=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 & pid247=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P16 > work/kat/gul_S1_pltcalc_P16 & pid248=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 & pid249=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 & pid250=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 & pid251=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P18 > work/kat/gul_S1_eltcalc_P18 & pid252=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 & pid253=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 & pid254=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P19 > work/kat/gul_S1_eltcalc_P19 & pid255=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 & pid256=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P19 > work/kat/gul_S1_pltcalc_P19 & pid257=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 & pid258=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 & pid259=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P20 > work/kat/gul_S1_pltcalc_P20 & pid260=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P21 > work/kat/gul_S1_eltcalc_P21 & pid261=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P21 > work/kat/gul_S1_summarycalc_P21 & pid262=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P21 > work/kat/gul_S1_pltcalc_P21 & pid263=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P22 > work/kat/gul_S1_eltcalc_P22 & pid264=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P22 > work/kat/gul_S1_summarycalc_P22 & pid265=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P22 > work/kat/gul_S1_pltcalc_P22 & pid266=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P23 > work/kat/gul_S1_eltcalc_P23 & pid267=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P23 > work/kat/gul_S1_summarycalc_P23 & pid268=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P23 > work/kat/gul_S1_pltcalc_P23 & pid269=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P24 > work/kat/gul_S1_eltcalc_P24 & pid270=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P24 > work/kat/gul_S1_summarycalc_P24 & pid271=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P24 > work/kat/gul_S1_pltcalc_P24 & pid272=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P25 > work/kat/gul_S1_eltcalc_P25 & pid273=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P25 > work/kat/gul_S1_summarycalc_P25 & pid274=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P25 > work/kat/gul_S1_pltcalc_P25 & pid275=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P26 > work/kat/gul_S1_eltcalc_P26 & pid276=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P26 > work/kat/gul_S1_summarycalc_P26 & pid277=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P26 > work/kat/gul_S1_pltcalc_P26 & pid278=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P27 > work/kat/gul_S1_eltcalc_P27 & pid279=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P27 > work/kat/gul_S1_summarycalc_P27 & pid280=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P27 > work/kat/gul_S1_pltcalc_P27 & pid281=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P28 > work/kat/gul_S1_eltcalc_P28 & pid282=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P28 > work/kat/gul_S1_summarycalc_P28 & pid283=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P28 > work/kat/gul_S1_pltcalc_P28 & pid284=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P29 > work/kat/gul_S1_eltcalc_P29 & pid285=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P29 > work/kat/gul_S1_summarycalc_P29 & pid286=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P29 > work/kat/gul_S1_pltcalc_P29 & pid287=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30 > work/kat/gul_S1_eltcalc_P30 & pid288=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30 > work/kat/gul_S1_summarycalc_P30 & pid289=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30 > work/kat/gul_S1_pltcalc_P30 & pid290=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P31 > work/kat/gul_S1_eltcalc_P31 & pid291=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P31 > work/kat/gul_S1_summarycalc_P31 & pid292=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P31 > work/kat/gul_S1_pltcalc_P31 & pid293=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P32 > work/kat/gul_S1_eltcalc_P32 & pid294=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P32 > work/kat/gul_S1_summarycalc_P32 & pid295=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P32 > work/kat/gul_S1_pltcalc_P32 & pid296=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P33 > work/kat/gul_S1_eltcalc_P33 & pid297=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P33 > work/kat/gul_S1_summarycalc_P33 & pid298=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P33 > work/kat/gul_S1_pltcalc_P33 & pid299=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P34 > work/kat/gul_S1_eltcalc_P34 & pid300=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P34 > work/kat/gul_S1_summarycalc_P34 & pid301=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P34 > work/kat/gul_S1_pltcalc_P34 & pid302=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P35 > work/kat/gul_S1_eltcalc_P35 & pid303=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P35 > work/kat/gul_S1_summarycalc_P35 & pid304=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P35 > work/kat/gul_S1_pltcalc_P35 & pid305=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36 > work/kat/gul_S1_eltcalc_P36 & pid306=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36 > work/kat/gul_S1_summarycalc_P36 & pid307=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36 > work/kat/gul_S1_pltcalc_P36 & pid308=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P37 > work/kat/gul_S1_eltcalc_P37 & pid309=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P37 > work/kat/gul_S1_summarycalc_P37 & pid310=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P37 > work/kat/gul_S1_pltcalc_P37 & pid311=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38 > work/kat/gul_S1_eltcalc_P38 & pid312=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38 > work/kat/gul_S1_summarycalc_P38 & pid313=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38 > work/kat/gul_S1_pltcalc_P38 & pid314=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39 > work/kat/gul_S1_eltcalc_P39 & pid315=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39 > work/kat/gul_S1_summarycalc_P39 & pid316=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39 > work/kat/gul_S1_pltcalc_P39 & pid317=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P40 > work/kat/gul_S1_eltcalc_P40 & pid318=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P40 > work/kat/gul_S1_summarycalc_P40 & pid319=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P40 > work/kat/gul_S1_pltcalc_P40 & pid320=$!


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid321=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid322=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid323=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid324=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid325=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx work/gul_S1_summaryaalcalc/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid326=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid327=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx work/gul_S1_summaryaalcalc/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid328=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid329=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx work/gul_S1_summaryaalcalc/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid330=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid331=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx work/gul_S1_summaryaalcalc/P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid332=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid333=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid334=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid335=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx work/gul_S1_summaryaalcalc/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid336=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid337=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx work/gul_S1_summaryaalcalc/P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid338=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid339=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid340=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid341=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11.idx work/gul_S1_summaryaalcalc/P11.idx work/gul_S1_summaryleccalc/P11.idx > /dev/null & pid342=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid343=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12.idx work/gul_S1_summaryaalcalc/P12.idx work/gul_S1_summaryleccalc/P12.idx > /dev/null & pid344=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid345=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13.idx work/gul_S1_summaryaalcalc/P13.idx work/gul_S1_summaryleccalc/P13.idx > /dev/null & pid346=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid347=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14.idx work/gul_S1_summaryaalcalc/P14.idx work/gul_S1_summaryleccalc/P14.idx > /dev/null & pid348=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid349=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15.idx work/gul_S1_summaryaalcalc/P15.idx work/gul_S1_summaryleccalc/P15.idx > /dev/null & pid350=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid351=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16.idx work/gul_S1_summaryaalcalc/P16.idx work/gul_S1_summaryleccalc/P16.idx > /dev/null & pid352=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid353=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17.idx work/gul_S1_summaryaalcalc/P17.idx work/gul_S1_summaryleccalc/P17.idx > /dev/null & pid354=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid355=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18.idx work/gul_S1_summaryaalcalc/P18.idx work/gul_S1_summaryleccalc/P18.idx > /dev/null & pid356=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid357=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19.idx work/gul_S1_summaryaalcalc/P19.idx work/gul_S1_summaryleccalc/P19.idx > /dev/null & pid358=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid359=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20.idx work/gul_S1_summaryaalcalc/P20.idx work/gul_S1_summaryleccalc/P20.idx > /dev/null & pid360=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P21 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P21 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P21 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P21 work/gul_S1_summaryaalcalc/P21.bin work/gul_S1_summaryleccalc/P21.bin > /dev/null & pid361=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P21.idx work/gul_S1_summaryaalcalc/P21.idx work/gul_S1_summaryleccalc/P21.idx > /dev/null & pid362=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P22 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P22 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P22 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P22 work/gul_S1_summaryaalcalc/P22.bin work/gul_S1_summaryleccalc/P22.bin > /dev/null & pid363=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P22.idx work/gul_S1_summaryaalcalc/P22.idx work/gul_S1_summaryleccalc/P22.idx > /dev/null & pid364=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P23 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P23 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P23 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P23 work/gul_S1_summaryaalcalc/P23.bin work/gul_S1_summaryleccalc/P23.bin > /dev/null & pid365=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P23.idx work/gul_S1_summaryaalcalc/P23.idx work/gul_S1_summaryleccalc/P23.idx > /dev/null & pid366=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P24 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P24 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P24 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P24 work/gul_S1_summaryaalcalc/P24.bin work/gul_S1_summaryleccalc/P24.bin > /dev/null & pid367=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P24.idx work/gul_S1_summaryaalcalc/P24.idx work/gul_S1_summaryleccalc/P24.idx > /dev/null & pid368=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P25 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P25 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P25 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P25 work/gul_S1_summaryaalcalc/P25.bin work/gul_S1_summaryleccalc/P25.bin > /dev/null & pid369=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P25.idx work/gul_S1_summaryaalcalc/P25.idx work/gul_S1_summaryleccalc/P25.idx > /dev/null & pid370=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P26 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P26 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P26 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P26 work/gul_S1_summaryaalcalc/P26.bin work/gul_S1_summaryleccalc/P26.bin > /dev/null & pid371=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P26.idx work/gul_S1_summaryaalcalc/P26.idx work/gul_S1_summaryleccalc/P26.idx > /dev/null & pid372=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P27 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P27 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P27 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P27 work/gul_S1_summaryaalcalc/P27.bin work/gul_S1_summaryleccalc/P27.bin > /dev/null & pid373=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P27.idx work/gul_S1_summaryaalcalc/P27.idx work/gul_S1_summaryleccalc/P27.idx > /dev/null & pid374=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P28 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P28 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P28 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P28 work/gul_S1_summaryaalcalc/P28.bin work/gul_S1_summaryleccalc/P28.bin > /dev/null & pid375=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P28.idx work/gul_S1_summaryaalcalc/P28.idx work/gul_S1_summaryleccalc/P28.idx > /dev/null & pid376=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P29 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P29 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P29 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P29 work/gul_S1_summaryaalcalc/P29.bin work/gul_S1_summaryleccalc/P29.bin > /dev/null & pid377=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P29.idx work/gul_S1_summaryaalcalc/P29.idx work/gul_S1_summaryleccalc/P29.idx > /dev/null & pid378=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30 work/gul_S1_summaryaalcalc/P30.bin work/gul_S1_summaryleccalc/P30.bin > /dev/null & pid379=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30.idx work/gul_S1_summaryaalcalc/P30.idx work/gul_S1_summaryleccalc/P30.idx > /dev/null & pid380=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P31 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P31 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P31 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P31 work/gul_S1_summaryaalcalc/P31.bin work/gul_S1_summaryleccalc/P31.bin > /dev/null & pid381=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P31.idx work/gul_S1_summaryaalcalc/P31.idx work/gul_S1_summaryleccalc/P31.idx > /dev/null & pid382=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P32 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P32 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P32 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P32 work/gul_S1_summaryaalcalc/P32.bin work/gul_S1_summaryleccalc/P32.bin > /dev/null & pid383=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P32.idx work/gul_S1_summaryaalcalc/P32.idx work/gul_S1_summaryleccalc/P32.idx > /dev/null & pid384=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P33 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P33 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P33 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P33 work/gul_S1_summaryaalcalc/P33.bin work/gul_S1_summaryleccalc/P33.bin > /dev/null & pid385=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P33.idx work/gul_S1_summaryaalcalc/P33.idx work/gul_S1_summaryleccalc/P33.idx > /dev/null & pid386=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P34 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P34 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P34 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P34 work/gul_S1_summaryaalcalc/P34.bin work/gul_S1_summaryleccalc/P34.bin > /dev/null & pid387=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P34.idx work/gul_S1_summaryaalcalc/P34.idx work/gul_S1_summaryleccalc/P34.idx > /dev/null & pid388=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P35 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P35 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P35 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P35 work/gul_S1_summaryaalcalc/P35.bin work/gul_S1_summaryleccalc/P35.bin > /dev/null & pid389=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P35.idx work/gul_S1_summaryaalcalc/P35.idx work/gul_S1_summaryleccalc/P35.idx > /dev/null & pid390=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36 work/gul_S1_summaryaalcalc/P36.bin work/gul_S1_summaryleccalc/P36.bin > /dev/null & pid391=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36.idx work/gul_S1_summaryaalcalc/P36.idx work/gul_S1_summaryleccalc/P36.idx > /dev/null & pid392=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P37 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P37 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P37 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P37 work/gul_S1_summaryaalcalc/P37.bin work/gul_S1_summaryleccalc/P37.bin > /dev/null & pid393=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P37.idx work/gul_S1_summaryaalcalc/P37.idx work/gul_S1_summaryleccalc/P37.idx > /dev/null & pid394=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38 work/gul_S1_summaryaalcalc/P38.bin work/gul_S1_summaryleccalc/P38.bin > /dev/null & pid395=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38.idx work/gul_S1_summaryaalcalc/P38.idx work/gul_S1_summaryleccalc/P38.idx > /dev/null & pid396=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39 work/gul_S1_summaryaalcalc/P39.bin work/gul_S1_summaryleccalc/P39.bin > /dev/null & pid397=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39.idx work/gul_S1_summaryaalcalc/P39.idx work/gul_S1_summaryleccalc/P39.idx > /dev/null & pid398=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P40 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P40 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P40 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P40 work/gul_S1_summaryaalcalc/P40.bin work/gul_S1_summaryleccalc/P40.bin > /dev/null & pid399=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P40.idx work/gul_S1_summaryaalcalc/P40.idx work/gul_S1_summaryleccalc/P40.idx > /dev/null & pid400=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/gul_P3 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/gul_P5 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/gul_P6 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/gul_P8 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/gul_P9 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/gul_P10 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/gul_P11 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/gul_P12 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/gul_P13 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/gul_P14 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/gul_P15 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/gul_P16 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/gul_P17 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/gul_P18 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/gul_P19 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/gul_P20 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P21 < /tmp/%FIFO_DIR%/fifo/gul_P21 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P22 < /tmp/%FIFO_DIR%/fifo/gul_P22 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P23 < /tmp/%FIFO_DIR%/fifo/gul_P23 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P24 < /tmp/%FIFO_DIR%/fifo/gul_P24 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P25 < /tmp/%FIFO_DIR%/fifo/gul_P25 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P26 < /tmp/%FIFO_DIR%/fifo/gul_P26 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P27 < /tmp/%FIFO_DIR%/fifo/gul_P27 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P28 < /tmp/%FIFO_DIR%/fifo/gul_P28 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P29 < /tmp/%FIFO_DIR%/fifo/gul_P29 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/gul_P30 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P31 < /tmp/%FIFO_DIR%/fifo/gul_P31 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P32 < /tmp/%FIFO_DIR%/fifo/gul_P32 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P33 < /tmp/%FIFO_DIR%/fifo/gul_P33 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P34 < /tmp/%FIFO_DIR%/fifo/gul_P34 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P35 < /tmp/%FIFO_DIR%/fifo/gul_P35 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36 < /tmp/%FIFO_DIR%/fifo/gul_P36 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P37 < /tmp/%FIFO_DIR%/fifo/gul_P37 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38 < /tmp/%FIFO_DIR%/fifo/gul_P38 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/gul_P39 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P40 < /tmp/%FIFO_DIR%/fifo/gul_P40 &

eve 1 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  &
eve 2 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P2 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  &
eve 3 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P3 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  &
eve 4 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P4 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  &
eve 5 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P5 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  &
eve 6 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P6 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  &
eve 7 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  &
eve 8 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P8 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  &
eve 9 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P9 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  &
eve 10 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P10 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  &
eve 11 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P11 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P11  &
eve 12 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P12 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  &
eve 13 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P13 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P13  &
eve 14 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P14 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P14  &
eve 15 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P15 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  &
eve 16 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P16 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P16  &
eve 17 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P17 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P17  &
eve 18 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P18 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P18  &
eve 19 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P19 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P19  &
eve 20 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P20 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  &
eve 21 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P21 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P21  &
eve 22 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P22 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P22  &
eve 23 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P23 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P23  &
eve 24 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P24 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P24  &
eve 25 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P25 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P25  &
eve 26 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P26 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P26  &
eve 27 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P27 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P27  &
eve 28 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P28 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P28  &
eve 29 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P29 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P29  &
eve 30 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P30 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P30  &
eve 31 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P31 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P31  &
eve 32 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P32 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P32  &
eve 33 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P33 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P33  &
eve 34 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P34 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P34  &
eve 35 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P35 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P35  &
eve 36 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P36 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P36  &
eve 37 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P37 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P37  &
eve 38 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P38 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P38  &
eve 39 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P39 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P39  &
eve 40 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P40 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P40  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210 $pid211 $pid212 $pid213 $pid214 $pid215 $pid216 $pid217 $pid218 $pid219 $pid220 $pid221 $pid222 $pid223 $pid224 $pid225 $pid226 $pid227 $pid228 $pid229 $pid230 $pid231 $pid232 $pid233 $pid234 $pid235 $pid236 $pid237 $pid238 $pid239 $pid240 $pid241 $pid242 $pid243 $pid244 $pid245 $pid246 $pid247 $pid248 $pid249 $pid250 $pid251 $pid252 $pid253 $pid254 $pid255 $pid256 $pid257 $pid258 $pid259 $pid260 $pid261 $pid262 $pid263 $pid264 $pid265 $pid266 $pid267 $pid268 $pid269 $pid270 $pid271 $pid272 $pid273 $pid274 $pid275 $pid276 $pid277 $pid278 $pid279 $pid280 $pid281 $pid282 $pid283 $pid284 $pid285 $pid286 $pid287 $pid288 $pid289 $pid290 $pid291 $pid292 $pid293 $pid294 $pid295 $pid296 $pid297 $pid298 $pid299 $pid300 $pid301 $pid302 $pid303 $pid304 $pid305 $pid306 $pid307 $pid308 $pid309 $pid310 $pid311 $pid312 $pid313 $pid314 $pid315 $pid316 $pid317 $pid318 $pid319 $pid320 $pid321 $pid322 $pid323 $pid324 $pid325 $pid326 $pid327 $pid328 $pid329 $pid330 $pid331 $pid332 $pid333 $pid334 $pid335 $pid336 $pid337 $pid338 $pid339 $pid340 $pid341 $pid342 $pid343 $pid344 $pid345 $pid346 $pid347 $pid348 $pid349 $pid350 $pid351 $pid352 $pid353 $pid354 $pid355 $pid356 $pid357 $pid358 $pid359 $pid360 $pid361 $pid362 $pid363 $pid364 $pid365 $pid366 $pid367 $pid368 $pid369 $pid370 $pid371 $pid372 $pid373 $pid374 $pid375 $pid376 $pid377 $pid378 $pid379 $pid380 $pid381 $pid382 $pid383 $pid384 $pid385 $pid386 $pid387 $pid388 $pid389 $pid390 $pid391 $pid392 $pid393 $pid394 $pid395 $pid396 $pid397 $pid398 $pid399 $pid400


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 work/kat/il_S1_eltcalc_P21 work/kat/il_S1_eltcalc_P22 work/kat/il_S1_eltcalc_P23 work/kat/il_S1_eltcalc_P24 work/kat/il_S1_eltcalc_P25 work/kat/il_S1_eltcalc_P26 work/kat/il_S1_eltcalc_P27 work/kat/il_S1_eltcalc_P28 work/kat/il_S1_eltcalc_P29 work/kat/il_S1_eltcalc_P30 work/kat/il_S1_eltcalc_P31 work/kat/il_S1_eltcalc_P32 work/kat/il_S1_eltcalc_P33 work/kat/il_S1_eltcalc_P34 work/kat/il_S1_eltcalc_P35 work/kat/il_S1_eltcalc_P36 work/kat/il_S1_eltcalc_P37 work/kat/il_S1_eltcalc_P38 work/kat/il_S1_eltcalc_P39 work/kat/il_S1_eltcalc_P40 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 work/kat/il_S1_pltcalc_P21 work/kat/il_S1_pltcalc_P22 work/kat/il_S1_pltcalc_P23 work/kat/il_S1_pltcalc_P24 work/kat/il_S1_pltcalc_P25 work/kat/il_S1_pltcalc_P26 work/kat/il_S1_pltcalc_P27 work/kat/il_S1_pltcalc_P28 work/kat/il_S1_pltcalc_P29 work/kat/il_S1_pltcalc_P30 work/kat/il_S1_pltcalc_P31 work/kat/il_S1_pltcalc_P32 work/kat/il_S1_pltcalc_P33 work/kat/il_S1_pltcalc_P34 work/kat/il_S1_pltcalc_P35 work/kat/il_S1_pltcalc_P36 work/kat/il_S1_pltcalc_P37 work/kat/il_S1_pltcalc_P38 work/kat/il_S1_pltcalc_P39 work/kat/il_S1_pltcalc_P40 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 work/kat/il_S1_summarycalc_P21 work/kat/il_S1_summarycalc_P22 work/kat/il_S1_summarycalc_P23 work/kat/il_S1_summarycalc_P24 work/kat/il_S1_summarycalc_P25 work/kat/il_S1_summarycalc_P26 work/kat/il_S1_summarycalc_P27 work/kat/il_S1_summarycalc_P28 work/kat/il_S1_summarycalc_P29 work/kat/il_S1_summarycalc_P30 work/kat/il_S1_summarycalc_P31 work/kat/il_S1_summarycalc_P32 work/kat/il_S1_summarycalc_P33 work/kat/il_S1_summarycalc_P34 work/kat/il_S1_summarycalc_P35 work/kat/il_S1_summarycalc_P36 work/kat/il_S1_summarycalc_P37 work/kat/il_S1_summarycalc_P38 work/kat/il_S1_summarycalc_P39 work/kat/il_S1_summarycalc_P40 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 work/kat/gul_S1_eltcalc_P21 work/kat/gul_S1_eltcalc_P22 work/kat/gul_S1_eltcalc_P23 work/kat/gul_S1_eltcalc_P24 work/kat/gul_S1_eltcalc_P25 work/kat/gul_S1_eltcalc_P26 work/kat/gul_S1_eltcalc_P27 work/kat/gul_S1_eltcalc_P28 work/kat/gul_S1_eltcalc_P29 work/kat/gul_S1_eltcalc_P30 work/kat/gul_S1_eltcalc_P31 work/kat/gul_S1_eltcalc_P32 work/kat/gul_S1_eltcalc_P33 work/kat/gul_S1_eltcalc_P34 work/kat/gul_S1_eltcalc_P35 work/kat/gul_S1_eltcalc_P36 work/kat/gul_S1_eltcalc_P37 work/kat/gul_S1_eltcalc_P38 work/kat/gul_S1_eltcalc_P39 work/kat/gul_S1_eltcalc_P40 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 work/kat/gul_S1_pltcalc_P21 work/kat/gul_S1_pltcalc_P22 work/kat/gul_S1_pltcalc_P23 work/kat/gul_S1_pltcalc_P24 work/kat/gul_S1_pltcalc_P25 work/kat/gul_S1_pltcalc_P26 work/kat/gul_S1_pltcalc_P27 work/kat/gul_S1_pltcalc_P28 work/kat/gul_S1_pltcalc_P29 work/kat/gul_S1_pltcalc_P30 work/kat/gul_S1_pltcalc_P31 work/kat/gul_S1_pltcalc_P32 work/kat/gul_S1_pltcalc_P33 work/kat/gul_S1_pltcalc_P34 work/kat/gul_S1_pltcalc_P35 work/kat/gul_S1_pltcalc_P36 work/kat/gul_S1_pltcalc_P37 work/kat/gul_S1_pltcalc_P38 work/kat/gul_S1_pltcalc_P39 work/kat/gul_S1_pltcalc_P40 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 work/kat/gul_S1_summarycalc_P21 work/kat/gul_S1_summarycalc_P22 work/kat/gul_S1_summarycalc_P23 work/kat/gul_S1_summarycalc_P24 work/kat/gul_S1_summarycalc_P25 work/kat/gul_S1_summarycalc_P26 work/kat/gul_S1_summarycalc_P27 work/kat/gul_S1_summarycalc_P28 work/kat/gul_S1_summarycalc_P29 work/kat/gul_S1_summarycalc_P30 work/kat/gul_S1_summarycalc_P31 work/kat/gul_S1_summarycalc_P32 work/kat/gul_S1_summarycalc_P33 work/kat/gul_S1_summarycalc_P34 work/kat/gul_S1_summarycalc_P35 work/kat/gul_S1_summarycalc_P36 work/kat/gul_S1_summarycalc_P37 work/kat/gul_S1_summarycalc_P38 work/kat/gul_S1_summarycalc_P39 work/kat/gul_S1_summarycalc_P40 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
