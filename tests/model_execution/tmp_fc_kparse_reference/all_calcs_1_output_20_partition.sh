#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc

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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P20

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
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P20

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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P10

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P11
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P11

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P12
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P13

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P14
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P14

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P16
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P16

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P17
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P19

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P20



# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid4=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid5=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid8=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid11=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid12=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid13=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid14=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid15=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid17=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid18=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid19=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid20=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid21=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid22=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid23=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid24=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid25=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid26=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid27=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid28=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid29=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid30=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P11 > work/kat/il_S1_eltcalc_P11 & pid31=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P11 > work/kat/il_S1_summarycalc_P11 & pid32=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P11 > work/kat/il_S1_pltcalc_P11 & pid33=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P12 > work/kat/il_S1_eltcalc_P12 & pid34=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P12 > work/kat/il_S1_summarycalc_P12 & pid35=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P12 > work/kat/il_S1_pltcalc_P12 & pid36=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P13 > work/kat/il_S1_eltcalc_P13 & pid37=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P13 > work/kat/il_S1_summarycalc_P13 & pid38=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P13 > work/kat/il_S1_pltcalc_P13 & pid39=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid40=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P14 > work/kat/il_S1_summarycalc_P14 & pid41=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P14 > work/kat/il_S1_pltcalc_P14 & pid42=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid43=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15 > work/kat/il_S1_summarycalc_P15 & pid44=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid45=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid46=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P16 > work/kat/il_S1_summarycalc_P16 & pid47=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P16 > work/kat/il_S1_pltcalc_P16 & pid48=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P17 > work/kat/il_S1_eltcalc_P17 & pid49=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P17 > work/kat/il_S1_summarycalc_P17 & pid50=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P17 > work/kat/il_S1_pltcalc_P17 & pid51=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid52=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18 > work/kat/il_S1_summarycalc_P18 & pid53=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18 > work/kat/il_S1_pltcalc_P18 & pid54=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P19 > work/kat/il_S1_eltcalc_P19 & pid55=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P19 > work/kat/il_S1_summarycalc_P19 & pid56=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P19 > work/kat/il_S1_pltcalc_P19 & pid57=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid58=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20 > work/kat/il_S1_summarycalc_P20 & pid59=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20 > work/kat/il_S1_pltcalc_P20 & pid60=$!

tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P11 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P11 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P12 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P12 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P13 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P13 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P14 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P14 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P16 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P16 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P17 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P17 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P19 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P19 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid80=$!

summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/il_P4 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/il_P9 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/il_P10 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/il_P11 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/il_P12 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/il_P13 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/il_P14 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/il_P15 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/il_P16 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/il_P17 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/il_P18 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/il_P19 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/il_P20 &

# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid4=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid5=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P3 > work/full_correlation/kat/il_S1_summarycalc_P3 & pid8=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P3 > work/full_correlation/kat/il_S1_pltcalc_P3 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P4 > work/full_correlation/kat/il_S1_summarycalc_P4 & pid11=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 & pid12=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 & pid13=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P5 > work/full_correlation/kat/il_S1_summarycalc_P5 & pid14=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P5 > work/full_correlation/kat/il_S1_pltcalc_P5 & pid15=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P6 > work/full_correlation/kat/il_S1_summarycalc_P6 & pid17=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 & pid18=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid19=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P7 > work/full_correlation/kat/il_S1_summarycalc_P7 & pid20=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 & pid21=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 & pid22=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P8 > work/full_correlation/kat/il_S1_summarycalc_P8 & pid23=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P8 > work/full_correlation/kat/il_S1_pltcalc_P8 & pid24=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 & pid25=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P9 > work/full_correlation/kat/il_S1_summarycalc_P9 & pid26=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P9 > work/full_correlation/kat/il_S1_pltcalc_P9 & pid27=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 & pid28=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P10 > work/full_correlation/kat/il_S1_summarycalc_P10 & pid29=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P10 > work/full_correlation/kat/il_S1_pltcalc_P10 & pid30=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P11 > work/full_correlation/kat/il_S1_eltcalc_P11 & pid31=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P11 > work/full_correlation/kat/il_S1_summarycalc_P11 & pid32=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P11 > work/full_correlation/kat/il_S1_pltcalc_P11 & pid33=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P12 > work/full_correlation/kat/il_S1_eltcalc_P12 & pid34=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P12 > work/full_correlation/kat/il_S1_summarycalc_P12 & pid35=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P12 > work/full_correlation/kat/il_S1_pltcalc_P12 & pid36=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P13 > work/full_correlation/kat/il_S1_eltcalc_P13 & pid37=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P13 > work/full_correlation/kat/il_S1_summarycalc_P13 & pid38=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P13 > work/full_correlation/kat/il_S1_pltcalc_P13 & pid39=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P14 > work/full_correlation/kat/il_S1_eltcalc_P14 & pid40=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P14 > work/full_correlation/kat/il_S1_summarycalc_P14 & pid41=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P14 > work/full_correlation/kat/il_S1_pltcalc_P14 & pid42=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 & pid43=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P15 > work/full_correlation/kat/il_S1_summarycalc_P15 & pid44=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 & pid45=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P16 > work/full_correlation/kat/il_S1_eltcalc_P16 & pid46=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P16 > work/full_correlation/kat/il_S1_summarycalc_P16 & pid47=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P16 > work/full_correlation/kat/il_S1_pltcalc_P16 & pid48=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P17 > work/full_correlation/kat/il_S1_eltcalc_P17 & pid49=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P17 > work/full_correlation/kat/il_S1_summarycalc_P17 & pid50=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P17 > work/full_correlation/kat/il_S1_pltcalc_P17 & pid51=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P18 > work/full_correlation/kat/il_S1_eltcalc_P18 & pid52=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P18 > work/full_correlation/kat/il_S1_summarycalc_P18 & pid53=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P18 > work/full_correlation/kat/il_S1_pltcalc_P18 & pid54=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P19 > work/full_correlation/kat/il_S1_eltcalc_P19 & pid55=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P19 > work/full_correlation/kat/il_S1_summarycalc_P19 & pid56=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P19 > work/full_correlation/kat/il_S1_pltcalc_P19 & pid57=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 & pid58=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P20 > work/full_correlation/kat/il_S1_summarycalc_P20 & pid59=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P20 > work/full_correlation/kat/il_S1_pltcalc_P20 & pid60=$!

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P3 work/full_correlation/il_S1_summaryaalcalc/P3.bin work/full_correlation/il_S1_summaryleccalc/P3.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P4 work/full_correlation/il_S1_summaryaalcalc/P4.bin work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P5 work/full_correlation/il_S1_summaryaalcalc/P5.bin work/full_correlation/il_S1_summaryleccalc/P5.bin > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P6 work/full_correlation/il_S1_summaryaalcalc/P6.bin work/full_correlation/il_S1_summaryleccalc/P6.bin > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P7 work/full_correlation/il_S1_summaryaalcalc/P7.bin work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P8 work/full_correlation/il_S1_summaryaalcalc/P8.bin work/full_correlation/il_S1_summaryleccalc/P8.bin > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P9 work/full_correlation/il_S1_summaryaalcalc/P9.bin work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P10 work/full_correlation/il_S1_summaryaalcalc/P10.bin work/full_correlation/il_S1_summaryleccalc/P10.bin > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P11 work/full_correlation/il_S1_summaryaalcalc/P11.bin work/full_correlation/il_S1_summaryleccalc/P11.bin > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P12 work/full_correlation/il_S1_summaryaalcalc/P12.bin work/full_correlation/il_S1_summaryleccalc/P12.bin > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P13 work/full_correlation/il_S1_summaryaalcalc/P13.bin work/full_correlation/il_S1_summaryleccalc/P13.bin > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P14 work/full_correlation/il_S1_summaryaalcalc/P14.bin work/full_correlation/il_S1_summaryleccalc/P14.bin > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P15 work/full_correlation/il_S1_summaryaalcalc/P15.bin work/full_correlation/il_S1_summaryleccalc/P15.bin > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P16 work/full_correlation/il_S1_summaryaalcalc/P16.bin work/full_correlation/il_S1_summaryleccalc/P16.bin > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P17 work/full_correlation/il_S1_summaryaalcalc/P17.bin work/full_correlation/il_S1_summaryleccalc/P17.bin > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P18 work/full_correlation/il_S1_summaryaalcalc/P18.bin work/full_correlation/il_S1_summaryleccalc/P18.bin > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P19 work/full_correlation/il_S1_summaryaalcalc/P19.bin work/full_correlation/il_S1_summaryleccalc/P19.bin > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarysummarycalc_P20 work/full_correlation/il_S1_summaryaalcalc/P20.bin work/full_correlation/il_S1_summaryleccalc/P20.bin > /dev/null & pid80=$!

summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19 &
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid81=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid82=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid83=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid84=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid85=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid86=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid87=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid88=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid89=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid90=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid91=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid92=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid93=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid94=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid95=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid96=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid97=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid98=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid99=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid100=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid101=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid102=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid103=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid104=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid105=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid106=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid107=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid108=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid109=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid110=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P11 > work/kat/gul_S1_eltcalc_P11 & pid111=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P11 > work/kat/gul_S1_summarycalc_P11 & pid112=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P11 > work/kat/gul_S1_pltcalc_P11 & pid113=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P12 > work/kat/gul_S1_eltcalc_P12 & pid114=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P12 > work/kat/gul_S1_summarycalc_P12 & pid115=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P12 > work/kat/gul_S1_pltcalc_P12 & pid116=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P13 > work/kat/gul_S1_eltcalc_P13 & pid117=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P13 > work/kat/gul_S1_summarycalc_P13 & pid118=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P13 > work/kat/gul_S1_pltcalc_P13 & pid119=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P14 > work/kat/gul_S1_eltcalc_P14 & pid120=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P14 > work/kat/gul_S1_summarycalc_P14 & pid121=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P14 > work/kat/gul_S1_pltcalc_P14 & pid122=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15 > work/kat/gul_S1_eltcalc_P15 & pid123=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid124=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15 > work/kat/gul_S1_pltcalc_P15 & pid125=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P16 > work/kat/gul_S1_eltcalc_P16 & pid126=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P16 > work/kat/gul_S1_summarycalc_P16 & pid127=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P16 > work/kat/gul_S1_pltcalc_P16 & pid128=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P17 > work/kat/gul_S1_eltcalc_P17 & pid129=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P17 > work/kat/gul_S1_summarycalc_P17 & pid130=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P17 > work/kat/gul_S1_pltcalc_P17 & pid131=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18 > work/kat/gul_S1_eltcalc_P18 & pid132=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18 > work/kat/gul_S1_summarycalc_P18 & pid133=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18 > work/kat/gul_S1_pltcalc_P18 & pid134=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P19 > work/kat/gul_S1_eltcalc_P19 & pid135=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P19 > work/kat/gul_S1_summarycalc_P19 & pid136=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19 > work/kat/gul_S1_pltcalc_P19 & pid137=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20 > work/kat/gul_S1_eltcalc_P20 & pid138=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20 > work/kat/gul_S1_summarycalc_P20 & pid139=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20 > work/kat/gul_S1_pltcalc_P20 & pid140=$!

tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid141=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid142=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid143=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid144=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid145=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid146=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid147=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid148=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid149=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid150=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P11 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid151=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P12 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid152=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P13 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid153=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P14 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid154=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid155=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P16 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid156=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid157=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid158=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid159=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid160=$!

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


eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid161=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid162=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid163=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid164=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid165=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid166=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P3 > work/full_correlation/kat/gul_S1_eltcalc_P3 & pid167=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P3 > work/full_correlation/kat/gul_S1_summarycalc_P3 & pid168=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P3 > work/full_correlation/kat/gul_S1_pltcalc_P3 & pid169=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid170=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 & pid171=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 & pid172=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P5 > work/full_correlation/kat/gul_S1_eltcalc_P5 & pid173=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P5 > work/full_correlation/kat/gul_S1_summarycalc_P5 & pid174=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P5 > work/full_correlation/kat/gul_S1_pltcalc_P5 & pid175=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P6 > work/full_correlation/kat/gul_S1_eltcalc_P6 & pid176=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P6 > work/full_correlation/kat/gul_S1_summarycalc_P6 & pid177=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P6 > work/full_correlation/kat/gul_S1_pltcalc_P6 & pid178=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 & pid179=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 & pid180=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 & pid181=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P8 > work/full_correlation/kat/gul_S1_eltcalc_P8 & pid182=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P8 > work/full_correlation/kat/gul_S1_summarycalc_P8 & pid183=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P8 > work/full_correlation/kat/gul_S1_pltcalc_P8 & pid184=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 & pid185=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P9 > work/full_correlation/kat/gul_S1_summarycalc_P9 & pid186=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P9 > work/full_correlation/kat/gul_S1_pltcalc_P9 & pid187=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P10 > work/full_correlation/kat/gul_S1_eltcalc_P10 & pid188=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 & pid189=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P10 > work/full_correlation/kat/gul_S1_pltcalc_P10 & pid190=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P11 > work/full_correlation/kat/gul_S1_eltcalc_P11 & pid191=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P11 > work/full_correlation/kat/gul_S1_summarycalc_P11 & pid192=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P11 > work/full_correlation/kat/gul_S1_pltcalc_P11 & pid193=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P12 > work/full_correlation/kat/gul_S1_eltcalc_P12 & pid194=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P12 > work/full_correlation/kat/gul_S1_summarycalc_P12 & pid195=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P12 > work/full_correlation/kat/gul_S1_pltcalc_P12 & pid196=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P13 > work/full_correlation/kat/gul_S1_eltcalc_P13 & pid197=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P13 > work/full_correlation/kat/gul_S1_summarycalc_P13 & pid198=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P13 > work/full_correlation/kat/gul_S1_pltcalc_P13 & pid199=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P14 > work/full_correlation/kat/gul_S1_eltcalc_P14 & pid200=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P14 > work/full_correlation/kat/gul_S1_summarycalc_P14 & pid201=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P14 > work/full_correlation/kat/gul_S1_pltcalc_P14 & pid202=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P15 > work/full_correlation/kat/gul_S1_eltcalc_P15 & pid203=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P15 > work/full_correlation/kat/gul_S1_summarycalc_P15 & pid204=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P15 > work/full_correlation/kat/gul_S1_pltcalc_P15 & pid205=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P16 > work/full_correlation/kat/gul_S1_eltcalc_P16 & pid206=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P16 > work/full_correlation/kat/gul_S1_summarycalc_P16 & pid207=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P16 > work/full_correlation/kat/gul_S1_pltcalc_P16 & pid208=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P17 > work/full_correlation/kat/gul_S1_eltcalc_P17 & pid209=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P17 > work/full_correlation/kat/gul_S1_summarycalc_P17 & pid210=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P17 > work/full_correlation/kat/gul_S1_pltcalc_P17 & pid211=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P18 > work/full_correlation/kat/gul_S1_eltcalc_P18 & pid212=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P18 > work/full_correlation/kat/gul_S1_summarycalc_P18 & pid213=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P18 > work/full_correlation/kat/gul_S1_pltcalc_P18 & pid214=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P19 > work/full_correlation/kat/gul_S1_eltcalc_P19 & pid215=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P19 > work/full_correlation/kat/gul_S1_summarycalc_P19 & pid216=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P19 > work/full_correlation/kat/gul_S1_pltcalc_P19 & pid217=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P20 > work/full_correlation/kat/gul_S1_eltcalc_P20 & pid218=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P20 > work/full_correlation/kat/gul_S1_summarycalc_P20 & pid219=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P20 > work/full_correlation/kat/gul_S1_pltcalc_P20 & pid220=$!

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid221=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid222=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid223=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid224=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid225=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid226=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid227=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid228=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid229=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid230=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P11 work/full_correlation/gul_S1_summaryaalcalc/P11.bin work/full_correlation/gul_S1_summaryleccalc/P11.bin > /dev/null & pid231=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin work/full_correlation/gul_S1_summaryleccalc/P12.bin > /dev/null & pid232=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin work/full_correlation/gul_S1_summaryleccalc/P13.bin > /dev/null & pid233=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P14 work/full_correlation/gul_S1_summaryaalcalc/P14.bin work/full_correlation/gul_S1_summaryleccalc/P14.bin > /dev/null & pid234=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin work/full_correlation/gul_S1_summaryleccalc/P15.bin > /dev/null & pid235=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P16 work/full_correlation/gul_S1_summaryaalcalc/P16.bin work/full_correlation/gul_S1_summaryleccalc/P16.bin > /dev/null & pid236=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin work/full_correlation/gul_S1_summaryleccalc/P17.bin > /dev/null & pid237=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin work/full_correlation/gul_S1_summaryleccalc/P18.bin > /dev/null & pid238=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P19 work/full_correlation/gul_S1_summaryaalcalc/P19.bin work/full_correlation/gul_S1_summaryleccalc/P19.bin > /dev/null & pid239=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarysummarycalc_P20 work/full_correlation/gul_S1_summaryaalcalc/P20.bin work/full_correlation/gul_S1_summaryleccalc/P20.bin > /dev/null & pid240=$!

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P1 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P3 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P4 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P5 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P6 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P7 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P8 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P9 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P10 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P11 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P11 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P12 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P13 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P13 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P14 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P14 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P15 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P15 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P16 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P16 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P17 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P17 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P18 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P18 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P19 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P19 &
fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P20 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P20 &

summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P1 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P2 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P3 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P4 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P5 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P6 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P7 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P8 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P9 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P10 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P11 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P12 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P13 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P14 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P15 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P16 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P16 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P17 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P18 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P19 &
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P20 &

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P1 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P2 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P3 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P4 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P5 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P6 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P7 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P8 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P9 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P10 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P11 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P11 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P12 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P12 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P13 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P13 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P14 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P14 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P15 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P15 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P16 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P16 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P17 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P17 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P18 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P18 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P19 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P19 > /dev/null &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_sumcalc_P20 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fmcalc_P20 > /dev/null &

eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P2 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P3 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P4 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P5 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P6 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P8 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P9 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P10 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P11 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P11 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P12 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P12 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P13 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P14 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P14 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P15 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P15 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P16 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P16 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P17 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P18 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P18 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P19 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P19 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P20 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P20 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 work/full_correlation/kat/il_S1_eltcalc_P11 work/full_correlation/kat/il_S1_eltcalc_P12 work/full_correlation/kat/il_S1_eltcalc_P13 work/full_correlation/kat/il_S1_eltcalc_P14 work/full_correlation/kat/il_S1_eltcalc_P15 work/full_correlation/kat/il_S1_eltcalc_P16 work/full_correlation/kat/il_S1_eltcalc_P17 work/full_correlation/kat/il_S1_eltcalc_P18 work/full_correlation/kat/il_S1_eltcalc_P19 work/full_correlation/kat/il_S1_eltcalc_P20 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 work/full_correlation/kat/il_S1_pltcalc_P3 work/full_correlation/kat/il_S1_pltcalc_P4 work/full_correlation/kat/il_S1_pltcalc_P5 work/full_correlation/kat/il_S1_pltcalc_P6 work/full_correlation/kat/il_S1_pltcalc_P7 work/full_correlation/kat/il_S1_pltcalc_P8 work/full_correlation/kat/il_S1_pltcalc_P9 work/full_correlation/kat/il_S1_pltcalc_P10 work/full_correlation/kat/il_S1_pltcalc_P11 work/full_correlation/kat/il_S1_pltcalc_P12 work/full_correlation/kat/il_S1_pltcalc_P13 work/full_correlation/kat/il_S1_pltcalc_P14 work/full_correlation/kat/il_S1_pltcalc_P15 work/full_correlation/kat/il_S1_pltcalc_P16 work/full_correlation/kat/il_S1_pltcalc_P17 work/full_correlation/kat/il_S1_pltcalc_P18 work/full_correlation/kat/il_S1_pltcalc_P19 work/full_correlation/kat/il_S1_pltcalc_P20 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 work/full_correlation/kat/il_S1_summarycalc_P2 work/full_correlation/kat/il_S1_summarycalc_P3 work/full_correlation/kat/il_S1_summarycalc_P4 work/full_correlation/kat/il_S1_summarycalc_P5 work/full_correlation/kat/il_S1_summarycalc_P6 work/full_correlation/kat/il_S1_summarycalc_P7 work/full_correlation/kat/il_S1_summarycalc_P8 work/full_correlation/kat/il_S1_summarycalc_P9 work/full_correlation/kat/il_S1_summarycalc_P10 work/full_correlation/kat/il_S1_summarycalc_P11 work/full_correlation/kat/il_S1_summarycalc_P12 work/full_correlation/kat/il_S1_summarycalc_P13 work/full_correlation/kat/il_S1_summarycalc_P14 work/full_correlation/kat/il_S1_summarycalc_P15 work/full_correlation/kat/il_S1_summarycalc_P16 work/full_correlation/kat/il_S1_summarycalc_P17 work/full_correlation/kat/il_S1_summarycalc_P18 work/full_correlation/kat/il_S1_summarycalc_P19 work/full_correlation/kat/il_S1_summarycalc_P20 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 work/full_correlation/kat/gul_S1_eltcalc_P3 work/full_correlation/kat/gul_S1_eltcalc_P4 work/full_correlation/kat/gul_S1_eltcalc_P5 work/full_correlation/kat/gul_S1_eltcalc_P6 work/full_correlation/kat/gul_S1_eltcalc_P7 work/full_correlation/kat/gul_S1_eltcalc_P8 work/full_correlation/kat/gul_S1_eltcalc_P9 work/full_correlation/kat/gul_S1_eltcalc_P10 work/full_correlation/kat/gul_S1_eltcalc_P11 work/full_correlation/kat/gul_S1_eltcalc_P12 work/full_correlation/kat/gul_S1_eltcalc_P13 work/full_correlation/kat/gul_S1_eltcalc_P14 work/full_correlation/kat/gul_S1_eltcalc_P15 work/full_correlation/kat/gul_S1_eltcalc_P16 work/full_correlation/kat/gul_S1_eltcalc_P17 work/full_correlation/kat/gul_S1_eltcalc_P18 work/full_correlation/kat/gul_S1_eltcalc_P19 work/full_correlation/kat/gul_S1_eltcalc_P20 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 work/full_correlation/kat/gul_S1_pltcalc_P3 work/full_correlation/kat/gul_S1_pltcalc_P4 work/full_correlation/kat/gul_S1_pltcalc_P5 work/full_correlation/kat/gul_S1_pltcalc_P6 work/full_correlation/kat/gul_S1_pltcalc_P7 work/full_correlation/kat/gul_S1_pltcalc_P8 work/full_correlation/kat/gul_S1_pltcalc_P9 work/full_correlation/kat/gul_S1_pltcalc_P10 work/full_correlation/kat/gul_S1_pltcalc_P11 work/full_correlation/kat/gul_S1_pltcalc_P12 work/full_correlation/kat/gul_S1_pltcalc_P13 work/full_correlation/kat/gul_S1_pltcalc_P14 work/full_correlation/kat/gul_S1_pltcalc_P15 work/full_correlation/kat/gul_S1_pltcalc_P16 work/full_correlation/kat/gul_S1_pltcalc_P17 work/full_correlation/kat/gul_S1_pltcalc_P18 work/full_correlation/kat/gul_S1_pltcalc_P19 work/full_correlation/kat/gul_S1_pltcalc_P20 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 work/full_correlation/kat/gul_S1_summarycalc_P3 work/full_correlation/kat/gul_S1_summarycalc_P4 work/full_correlation/kat/gul_S1_summarycalc_P5 work/full_correlation/kat/gul_S1_summarycalc_P6 work/full_correlation/kat/gul_S1_summarycalc_P7 work/full_correlation/kat/gul_S1_summarycalc_P8 work/full_correlation/kat/gul_S1_summarycalc_P9 work/full_correlation/kat/gul_S1_summarycalc_P10 work/full_correlation/kat/gul_S1_summarycalc_P11 work/full_correlation/kat/gul_S1_summarycalc_P12 work/full_correlation/kat/gul_S1_summarycalc_P13 work/full_correlation/kat/gul_S1_summarycalc_P14 work/full_correlation/kat/gul_S1_summarycalc_P15 work/full_correlation/kat/gul_S1_summarycalc_P16 work/full_correlation/kat/gul_S1_summarycalc_P17 work/full_correlation/kat/gul_S1_summarycalc_P18 work/full_correlation/kat/gul_S1_summarycalc_P19 work/full_correlation/kat/gul_S1_summarycalc_P20 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid4=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -F output/full_correlation/il_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/il_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/il_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/il_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/il_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/il_S1_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid7=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
