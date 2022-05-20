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

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P1
mkfifo fifo/full_correlation/gul_fc_P2
mkfifo fifo/full_correlation/gul_fc_P3
mkfifo fifo/full_correlation/gul_fc_P4
mkfifo fifo/full_correlation/gul_fc_P5
mkfifo fifo/full_correlation/gul_fc_P6
mkfifo fifo/full_correlation/gul_fc_P7
mkfifo fifo/full_correlation/gul_fc_P8
mkfifo fifo/full_correlation/gul_fc_P9
mkfifo fifo/full_correlation/gul_fc_P10
mkfifo fifo/full_correlation/gul_fc_P11
mkfifo fifo/full_correlation/gul_fc_P12
mkfifo fifo/full_correlation/gul_fc_P13
mkfifo fifo/full_correlation/gul_fc_P14
mkfifo fifo/full_correlation/gul_fc_P15
mkfifo fifo/full_correlation/gul_fc_P16
mkfifo fifo/full_correlation/gul_fc_P17
mkfifo fifo/full_correlation/gul_fc_P18
mkfifo fifo/full_correlation/gul_fc_P19
mkfifo fifo/full_correlation/gul_fc_P20

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
mkfifo fifo/gul_P11
mkfifo fifo/gul_P12
mkfifo fifo/gul_P13
mkfifo fifo/gul_P14
mkfifo fifo/gul_P15
mkfifo fifo/gul_P16
mkfifo fifo/gul_P17
mkfifo fifo/gul_P18
mkfifo fifo/gul_P19
mkfifo fifo/gul_P20

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_summary_P11.idx
mkfifo fifo/gul_S1_eltcalc_P11
mkfifo fifo/gul_S1_summarycalc_P11
mkfifo fifo/gul_S1_pltcalc_P11

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_summary_P12.idx
mkfifo fifo/gul_S1_eltcalc_P12
mkfifo fifo/gul_S1_summarycalc_P12
mkfifo fifo/gul_S1_pltcalc_P12

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summary_P13.idx
mkfifo fifo/gul_S1_eltcalc_P13
mkfifo fifo/gul_S1_summarycalc_P13
mkfifo fifo/gul_S1_pltcalc_P13

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summary_P14.idx
mkfifo fifo/gul_S1_eltcalc_P14
mkfifo fifo/gul_S1_summarycalc_P14
mkfifo fifo/gul_S1_pltcalc_P14

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_summary_P15.idx
mkfifo fifo/gul_S1_eltcalc_P15
mkfifo fifo/gul_S1_summarycalc_P15
mkfifo fifo/gul_S1_pltcalc_P15

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_summary_P16.idx
mkfifo fifo/gul_S1_eltcalc_P16
mkfifo fifo/gul_S1_summarycalc_P16
mkfifo fifo/gul_S1_pltcalc_P16

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_summary_P17.idx
mkfifo fifo/gul_S1_eltcalc_P17
mkfifo fifo/gul_S1_summarycalc_P17
mkfifo fifo/gul_S1_pltcalc_P17

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_summary_P18.idx
mkfifo fifo/gul_S1_eltcalc_P18
mkfifo fifo/gul_S1_summarycalc_P18
mkfifo fifo/gul_S1_pltcalc_P18

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summary_P19.idx
mkfifo fifo/gul_S1_eltcalc_P19
mkfifo fifo/gul_S1_summarycalc_P19
mkfifo fifo/gul_S1_pltcalc_P19

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_summary_P20.idx
mkfifo fifo/gul_S1_eltcalc_P20
mkfifo fifo/gul_S1_summarycalc_P20
mkfifo fifo/gul_S1_pltcalc_P20

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
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summary_P4.idx
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summary_P8.idx
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summary_P9.idx
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summary_P10.idx
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summary_P11.idx
mkfifo fifo/il_S1_eltcalc_P11
mkfifo fifo/il_S1_summarycalc_P11
mkfifo fifo/il_S1_pltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summary_P12.idx
mkfifo fifo/il_S1_eltcalc_P12
mkfifo fifo/il_S1_summarycalc_P12
mkfifo fifo/il_S1_pltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_summary_P13.idx
mkfifo fifo/il_S1_eltcalc_P13
mkfifo fifo/il_S1_summarycalc_P13
mkfifo fifo/il_S1_pltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_summary_P14.idx
mkfifo fifo/il_S1_eltcalc_P14
mkfifo fifo/il_S1_summarycalc_P14
mkfifo fifo/il_S1_pltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summary_P15.idx
mkfifo fifo/il_S1_eltcalc_P15
mkfifo fifo/il_S1_summarycalc_P15
mkfifo fifo/il_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summary_P16.idx
mkfifo fifo/il_S1_eltcalc_P16
mkfifo fifo/il_S1_summarycalc_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_summary_P17.idx
mkfifo fifo/il_S1_eltcalc_P17
mkfifo fifo/il_S1_summarycalc_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_summary_P18.idx
mkfifo fifo/il_S1_eltcalc_P18
mkfifo fifo/il_S1_summarycalc_P18
mkfifo fifo/il_S1_pltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_summary_P19.idx
mkfifo fifo/il_S1_eltcalc_P19
mkfifo fifo/il_S1_summarycalc_P19
mkfifo fifo/il_S1_pltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_summary_P20.idx
mkfifo fifo/il_S1_eltcalc_P20
mkfifo fifo/il_S1_summarycalc_P20
mkfifo fifo/il_S1_pltcalc_P20

mkfifo fifo/full_correlation/gul_P1
mkfifo fifo/full_correlation/gul_P2
mkfifo fifo/full_correlation/gul_P3
mkfifo fifo/full_correlation/gul_P4
mkfifo fifo/full_correlation/gul_P5
mkfifo fifo/full_correlation/gul_P6
mkfifo fifo/full_correlation/gul_P7
mkfifo fifo/full_correlation/gul_P8
mkfifo fifo/full_correlation/gul_P9
mkfifo fifo/full_correlation/gul_P10
mkfifo fifo/full_correlation/gul_P11
mkfifo fifo/full_correlation/gul_P12
mkfifo fifo/full_correlation/gul_P13
mkfifo fifo/full_correlation/gul_P14
mkfifo fifo/full_correlation/gul_P15
mkfifo fifo/full_correlation/gul_P16
mkfifo fifo/full_correlation/gul_P17
mkfifo fifo/full_correlation/gul_P18
mkfifo fifo/full_correlation/gul_P19
mkfifo fifo/full_correlation/gul_P20

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_summary_P1.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo fifo/full_correlation/gul_S1_pltcalc_P1

mkfifo fifo/full_correlation/gul_S1_summary_P2
mkfifo fifo/full_correlation/gul_S1_summary_P2.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo fifo/full_correlation/gul_S1_pltcalc_P2

mkfifo fifo/full_correlation/gul_S1_summary_P3
mkfifo fifo/full_correlation/gul_S1_summary_P3.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P3
mkfifo fifo/full_correlation/gul_S1_summarycalc_P3
mkfifo fifo/full_correlation/gul_S1_pltcalc_P3

mkfifo fifo/full_correlation/gul_S1_summary_P4
mkfifo fifo/full_correlation/gul_S1_summary_P4.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P4
mkfifo fifo/full_correlation/gul_S1_summarycalc_P4
mkfifo fifo/full_correlation/gul_S1_pltcalc_P4

mkfifo fifo/full_correlation/gul_S1_summary_P5
mkfifo fifo/full_correlation/gul_S1_summary_P5.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P5
mkfifo fifo/full_correlation/gul_S1_summarycalc_P5
mkfifo fifo/full_correlation/gul_S1_pltcalc_P5

mkfifo fifo/full_correlation/gul_S1_summary_P6
mkfifo fifo/full_correlation/gul_S1_summary_P6.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P6
mkfifo fifo/full_correlation/gul_S1_summarycalc_P6
mkfifo fifo/full_correlation/gul_S1_pltcalc_P6

mkfifo fifo/full_correlation/gul_S1_summary_P7
mkfifo fifo/full_correlation/gul_S1_summary_P7.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P7
mkfifo fifo/full_correlation/gul_S1_summarycalc_P7
mkfifo fifo/full_correlation/gul_S1_pltcalc_P7

mkfifo fifo/full_correlation/gul_S1_summary_P8
mkfifo fifo/full_correlation/gul_S1_summary_P8.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P8
mkfifo fifo/full_correlation/gul_S1_summarycalc_P8
mkfifo fifo/full_correlation/gul_S1_pltcalc_P8

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S1_summary_P9.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P9
mkfifo fifo/full_correlation/gul_S1_summarycalc_P9
mkfifo fifo/full_correlation/gul_S1_pltcalc_P9

mkfifo fifo/full_correlation/gul_S1_summary_P10
mkfifo fifo/full_correlation/gul_S1_summary_P10.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P10
mkfifo fifo/full_correlation/gul_S1_summarycalc_P10
mkfifo fifo/full_correlation/gul_S1_pltcalc_P10

mkfifo fifo/full_correlation/gul_S1_summary_P11
mkfifo fifo/full_correlation/gul_S1_summary_P11.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P11
mkfifo fifo/full_correlation/gul_S1_summarycalc_P11
mkfifo fifo/full_correlation/gul_S1_pltcalc_P11

mkfifo fifo/full_correlation/gul_S1_summary_P12
mkfifo fifo/full_correlation/gul_S1_summary_P12.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P12
mkfifo fifo/full_correlation/gul_S1_summarycalc_P12
mkfifo fifo/full_correlation/gul_S1_pltcalc_P12

mkfifo fifo/full_correlation/gul_S1_summary_P13
mkfifo fifo/full_correlation/gul_S1_summary_P13.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P13
mkfifo fifo/full_correlation/gul_S1_summarycalc_P13
mkfifo fifo/full_correlation/gul_S1_pltcalc_P13

mkfifo fifo/full_correlation/gul_S1_summary_P14
mkfifo fifo/full_correlation/gul_S1_summary_P14.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P14
mkfifo fifo/full_correlation/gul_S1_summarycalc_P14
mkfifo fifo/full_correlation/gul_S1_pltcalc_P14

mkfifo fifo/full_correlation/gul_S1_summary_P15
mkfifo fifo/full_correlation/gul_S1_summary_P15.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P15
mkfifo fifo/full_correlation/gul_S1_summarycalc_P15
mkfifo fifo/full_correlation/gul_S1_pltcalc_P15

mkfifo fifo/full_correlation/gul_S1_summary_P16
mkfifo fifo/full_correlation/gul_S1_summary_P16.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P16
mkfifo fifo/full_correlation/gul_S1_summarycalc_P16
mkfifo fifo/full_correlation/gul_S1_pltcalc_P16

mkfifo fifo/full_correlation/gul_S1_summary_P17
mkfifo fifo/full_correlation/gul_S1_summary_P17.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P17
mkfifo fifo/full_correlation/gul_S1_summarycalc_P17
mkfifo fifo/full_correlation/gul_S1_pltcalc_P17

mkfifo fifo/full_correlation/gul_S1_summary_P18
mkfifo fifo/full_correlation/gul_S1_summary_P18.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P18
mkfifo fifo/full_correlation/gul_S1_summarycalc_P18
mkfifo fifo/full_correlation/gul_S1_pltcalc_P18

mkfifo fifo/full_correlation/gul_S1_summary_P19
mkfifo fifo/full_correlation/gul_S1_summary_P19.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P19
mkfifo fifo/full_correlation/gul_S1_summarycalc_P19
mkfifo fifo/full_correlation/gul_S1_pltcalc_P19

mkfifo fifo/full_correlation/gul_S1_summary_P20
mkfifo fifo/full_correlation/gul_S1_summary_P20.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P20
mkfifo fifo/full_correlation/gul_S1_summarycalc_P20
mkfifo fifo/full_correlation/gul_S1_pltcalc_P20

mkfifo fifo/full_correlation/il_P1
mkfifo fifo/full_correlation/il_P2
mkfifo fifo/full_correlation/il_P3
mkfifo fifo/full_correlation/il_P4
mkfifo fifo/full_correlation/il_P5
mkfifo fifo/full_correlation/il_P6
mkfifo fifo/full_correlation/il_P7
mkfifo fifo/full_correlation/il_P8
mkfifo fifo/full_correlation/il_P9
mkfifo fifo/full_correlation/il_P10
mkfifo fifo/full_correlation/il_P11
mkfifo fifo/full_correlation/il_P12
mkfifo fifo/full_correlation/il_P13
mkfifo fifo/full_correlation/il_P14
mkfifo fifo/full_correlation/il_P15
mkfifo fifo/full_correlation/il_P16
mkfifo fifo/full_correlation/il_P17
mkfifo fifo/full_correlation/il_P18
mkfifo fifo/full_correlation/il_P19
mkfifo fifo/full_correlation/il_P20

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summary_P1.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_summary_P2.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P2
mkfifo fifo/full_correlation/il_S1_summarycalc_P2
mkfifo fifo/full_correlation/il_S1_pltcalc_P2

mkfifo fifo/full_correlation/il_S1_summary_P3
mkfifo fifo/full_correlation/il_S1_summary_P3.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P3
mkfifo fifo/full_correlation/il_S1_summarycalc_P3
mkfifo fifo/full_correlation/il_S1_pltcalc_P3

mkfifo fifo/full_correlation/il_S1_summary_P4
mkfifo fifo/full_correlation/il_S1_summary_P4.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P4
mkfifo fifo/full_correlation/il_S1_summarycalc_P4
mkfifo fifo/full_correlation/il_S1_pltcalc_P4

mkfifo fifo/full_correlation/il_S1_summary_P5
mkfifo fifo/full_correlation/il_S1_summary_P5.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P5
mkfifo fifo/full_correlation/il_S1_summarycalc_P5
mkfifo fifo/full_correlation/il_S1_pltcalc_P5

mkfifo fifo/full_correlation/il_S1_summary_P6
mkfifo fifo/full_correlation/il_S1_summary_P6.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P6
mkfifo fifo/full_correlation/il_S1_summarycalc_P6
mkfifo fifo/full_correlation/il_S1_pltcalc_P6

mkfifo fifo/full_correlation/il_S1_summary_P7
mkfifo fifo/full_correlation/il_S1_summary_P7.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P7
mkfifo fifo/full_correlation/il_S1_summarycalc_P7
mkfifo fifo/full_correlation/il_S1_pltcalc_P7

mkfifo fifo/full_correlation/il_S1_summary_P8
mkfifo fifo/full_correlation/il_S1_summary_P8.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P8
mkfifo fifo/full_correlation/il_S1_summarycalc_P8
mkfifo fifo/full_correlation/il_S1_pltcalc_P8

mkfifo fifo/full_correlation/il_S1_summary_P9
mkfifo fifo/full_correlation/il_S1_summary_P9.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P9
mkfifo fifo/full_correlation/il_S1_summarycalc_P9
mkfifo fifo/full_correlation/il_S1_pltcalc_P9

mkfifo fifo/full_correlation/il_S1_summary_P10
mkfifo fifo/full_correlation/il_S1_summary_P10.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P10
mkfifo fifo/full_correlation/il_S1_summarycalc_P10
mkfifo fifo/full_correlation/il_S1_pltcalc_P10

mkfifo fifo/full_correlation/il_S1_summary_P11
mkfifo fifo/full_correlation/il_S1_summary_P11.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P11
mkfifo fifo/full_correlation/il_S1_summarycalc_P11
mkfifo fifo/full_correlation/il_S1_pltcalc_P11

mkfifo fifo/full_correlation/il_S1_summary_P12
mkfifo fifo/full_correlation/il_S1_summary_P12.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P12
mkfifo fifo/full_correlation/il_S1_summarycalc_P12
mkfifo fifo/full_correlation/il_S1_pltcalc_P12

mkfifo fifo/full_correlation/il_S1_summary_P13
mkfifo fifo/full_correlation/il_S1_summary_P13.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P13
mkfifo fifo/full_correlation/il_S1_summarycalc_P13
mkfifo fifo/full_correlation/il_S1_pltcalc_P13

mkfifo fifo/full_correlation/il_S1_summary_P14
mkfifo fifo/full_correlation/il_S1_summary_P14.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P14
mkfifo fifo/full_correlation/il_S1_summarycalc_P14
mkfifo fifo/full_correlation/il_S1_pltcalc_P14

mkfifo fifo/full_correlation/il_S1_summary_P15
mkfifo fifo/full_correlation/il_S1_summary_P15.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P15
mkfifo fifo/full_correlation/il_S1_summarycalc_P15
mkfifo fifo/full_correlation/il_S1_pltcalc_P15

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_summary_P16.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P16
mkfifo fifo/full_correlation/il_S1_summarycalc_P16
mkfifo fifo/full_correlation/il_S1_pltcalc_P16

mkfifo fifo/full_correlation/il_S1_summary_P17
mkfifo fifo/full_correlation/il_S1_summary_P17.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P17
mkfifo fifo/full_correlation/il_S1_summarycalc_P17
mkfifo fifo/full_correlation/il_S1_pltcalc_P17

mkfifo fifo/full_correlation/il_S1_summary_P18
mkfifo fifo/full_correlation/il_S1_summary_P18.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P18
mkfifo fifo/full_correlation/il_S1_summarycalc_P18
mkfifo fifo/full_correlation/il_S1_pltcalc_P18

mkfifo fifo/full_correlation/il_S1_summary_P19
mkfifo fifo/full_correlation/il_S1_summary_P19.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P19
mkfifo fifo/full_correlation/il_S1_summarycalc_P19
mkfifo fifo/full_correlation/il_S1_pltcalc_P19

mkfifo fifo/full_correlation/il_S1_summary_P20
mkfifo fifo/full_correlation/il_S1_summary_P20.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P20
mkfifo fifo/full_correlation/il_S1_summarycalc_P20
mkfifo fifo/full_correlation/il_S1_pltcalc_P20



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid4=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid5=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid6=$!
eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid7=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid8=$!
pltcalc -H < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid9=$!
eltcalc -s < fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid10=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid11=$!
pltcalc -H < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid12=$!
eltcalc -s < fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid13=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid14=$!
pltcalc -H < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid15=$!
eltcalc -s < fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid16=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid17=$!
pltcalc -H < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid18=$!
eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid19=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid20=$!
pltcalc -H < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid21=$!
eltcalc -s < fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid22=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid23=$!
pltcalc -H < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid24=$!
eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid25=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid26=$!
pltcalc -H < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid27=$!
eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid28=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid29=$!
pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid30=$!
eltcalc -s < fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 & pid31=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 & pid32=$!
pltcalc -H < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 & pid33=$!
eltcalc -s < fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 & pid34=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P12 > work/kat/il_S1_summarycalc_P12 & pid35=$!
pltcalc -H < fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 & pid36=$!
eltcalc -s < fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 & pid37=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P13 > work/kat/il_S1_summarycalc_P13 & pid38=$!
pltcalc -H < fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 & pid39=$!
eltcalc -s < fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid40=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P14 > work/kat/il_S1_summarycalc_P14 & pid41=$!
pltcalc -H < fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 & pid42=$!
eltcalc -s < fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid43=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P15 > work/kat/il_S1_summarycalc_P15 & pid44=$!
pltcalc -H < fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid45=$!
eltcalc -s < fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid46=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P16 > work/kat/il_S1_summarycalc_P16 & pid47=$!
pltcalc -H < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 & pid48=$!
eltcalc -s < fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 & pid49=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 & pid50=$!
pltcalc -H < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 & pid51=$!
eltcalc -s < fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid52=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 & pid53=$!
pltcalc -H < fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 & pid54=$!
eltcalc -s < fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 & pid55=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P19 > work/kat/il_S1_summarycalc_P19 & pid56=$!
pltcalc -H < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 & pid57=$!
eltcalc -s < fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid58=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P20 > work/kat/il_S1_summarycalc_P20 & pid59=$!
pltcalc -H < fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 & pid60=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid62=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid63=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid64=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 fifo/il_S1_summarycalc_P3 fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid65=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid66=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_eltcalc_P4 fifo/il_S1_summarycalc_P4 fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid67=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid68=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_eltcalc_P5 fifo/il_S1_summarycalc_P5 fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid69=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid70=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_eltcalc_P6 fifo/il_S1_summarycalc_P6 fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid71=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid72=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 fifo/il_S1_summarycalc_P7 fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid73=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid74=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_eltcalc_P8 fifo/il_S1_summarycalc_P8 fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid75=$!
tee < fifo/il_S1_summary_P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid76=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid77=$!
tee < fifo/il_S1_summary_P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid78=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid79=$!
tee < fifo/il_S1_summary_P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid80=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_eltcalc_P11 fifo/il_S1_summarycalc_P11 fifo/il_S1_pltcalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid81=$!
tee < fifo/il_S1_summary_P11.idx work/il_S1_summaryleccalc/P11.idx > /dev/null & pid82=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_eltcalc_P12 fifo/il_S1_summarycalc_P12 fifo/il_S1_pltcalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid83=$!
tee < fifo/il_S1_summary_P12.idx work/il_S1_summaryleccalc/P12.idx > /dev/null & pid84=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_eltcalc_P13 fifo/il_S1_summarycalc_P13 fifo/il_S1_pltcalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid85=$!
tee < fifo/il_S1_summary_P13.idx work/il_S1_summaryleccalc/P13.idx > /dev/null & pid86=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_eltcalc_P14 fifo/il_S1_summarycalc_P14 fifo/il_S1_pltcalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid87=$!
tee < fifo/il_S1_summary_P14.idx work/il_S1_summaryleccalc/P14.idx > /dev/null & pid88=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_eltcalc_P15 fifo/il_S1_summarycalc_P15 fifo/il_S1_pltcalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid89=$!
tee < fifo/il_S1_summary_P15.idx work/il_S1_summaryleccalc/P15.idx > /dev/null & pid90=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_eltcalc_P16 fifo/il_S1_summarycalc_P16 fifo/il_S1_pltcalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid91=$!
tee < fifo/il_S1_summary_P16.idx work/il_S1_summaryleccalc/P16.idx > /dev/null & pid92=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_eltcalc_P17 fifo/il_S1_summarycalc_P17 fifo/il_S1_pltcalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid93=$!
tee < fifo/il_S1_summary_P17.idx work/il_S1_summaryleccalc/P17.idx > /dev/null & pid94=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_eltcalc_P18 fifo/il_S1_summarycalc_P18 fifo/il_S1_pltcalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid95=$!
tee < fifo/il_S1_summary_P18.idx work/il_S1_summaryleccalc/P18.idx > /dev/null & pid96=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_eltcalc_P19 fifo/il_S1_summarycalc_P19 fifo/il_S1_pltcalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid97=$!
tee < fifo/il_S1_summary_P19.idx work/il_S1_summaryleccalc/P19.idx > /dev/null & pid98=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_eltcalc_P20 fifo/il_S1_summarycalc_P20 fifo/il_S1_pltcalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid99=$!
tee < fifo/il_S1_summary_P20.idx work/il_S1_summaryleccalc/P20.idx > /dev/null & pid100=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarycalc -m -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarycalc -m -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarycalc -m -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarycalc -m -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 &
summarycalc -m -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 &
summarycalc -m -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 &
summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 &
summarycalc -m -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 &
summarycalc -m -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 &
summarycalc -m -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 &
summarycalc -m -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 &
summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &
summarycalc -m -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 &
summarycalc -m -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 &
summarycalc -m -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 &
summarycalc -m -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid101=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid102=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid103=$!
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid104=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid105=$!
pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid106=$!
eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid107=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid108=$!
pltcalc -H < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid109=$!
eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid110=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid111=$!
pltcalc -H < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid112=$!
eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid113=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid114=$!
pltcalc -H < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid115=$!
eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid116=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid117=$!
pltcalc -H < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid118=$!
eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid119=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid120=$!
pltcalc -H < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid121=$!
eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid122=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid123=$!
pltcalc -H < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid124=$!
eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid125=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid126=$!
pltcalc -H < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid127=$!
eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid128=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid129=$!
pltcalc -H < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid130=$!
eltcalc -s < fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 & pid131=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 & pid132=$!
pltcalc -H < fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 & pid133=$!
eltcalc -s < fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 & pid134=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 & pid135=$!
pltcalc -H < fifo/gul_S1_pltcalc_P12 > work/kat/gul_S1_pltcalc_P12 & pid136=$!
eltcalc -s < fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 & pid137=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 & pid138=$!
pltcalc -H < fifo/gul_S1_pltcalc_P13 > work/kat/gul_S1_pltcalc_P13 & pid139=$!
eltcalc -s < fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 & pid140=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 & pid141=$!
pltcalc -H < fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 & pid142=$!
eltcalc -s < fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 & pid143=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid144=$!
pltcalc -H < fifo/gul_S1_pltcalc_P15 > work/kat/gul_S1_pltcalc_P15 & pid145=$!
eltcalc -s < fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 & pid146=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 & pid147=$!
pltcalc -H < fifo/gul_S1_pltcalc_P16 > work/kat/gul_S1_pltcalc_P16 & pid148=$!
eltcalc -s < fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 & pid149=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 & pid150=$!
pltcalc -H < fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 & pid151=$!
eltcalc -s < fifo/gul_S1_eltcalc_P18 > work/kat/gul_S1_eltcalc_P18 & pid152=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 & pid153=$!
pltcalc -H < fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 & pid154=$!
eltcalc -s < fifo/gul_S1_eltcalc_P19 > work/kat/gul_S1_eltcalc_P19 & pid155=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 & pid156=$!
pltcalc -H < fifo/gul_S1_pltcalc_P19 > work/kat/gul_S1_pltcalc_P19 & pid157=$!
eltcalc -s < fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 & pid158=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 & pid159=$!
pltcalc -H < fifo/gul_S1_pltcalc_P20 > work/kat/gul_S1_pltcalc_P20 & pid160=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid161=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid162=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid163=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid164=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid165=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid166=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid167=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid168=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid169=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid170=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid171=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid172=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid173=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid174=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid175=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid176=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid177=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid178=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid179=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid180=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_eltcalc_P11 fifo/gul_S1_summarycalc_P11 fifo/gul_S1_pltcalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid181=$!
tee < fifo/gul_S1_summary_P11.idx work/gul_S1_summaryleccalc/P11.idx > /dev/null & pid182=$!
tee < fifo/gul_S1_summary_P12 fifo/gul_S1_eltcalc_P12 fifo/gul_S1_summarycalc_P12 fifo/gul_S1_pltcalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid183=$!
tee < fifo/gul_S1_summary_P12.idx work/gul_S1_summaryleccalc/P12.idx > /dev/null & pid184=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_eltcalc_P13 fifo/gul_S1_summarycalc_P13 fifo/gul_S1_pltcalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid185=$!
tee < fifo/gul_S1_summary_P13.idx work/gul_S1_summaryleccalc/P13.idx > /dev/null & pid186=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_eltcalc_P14 fifo/gul_S1_summarycalc_P14 fifo/gul_S1_pltcalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid187=$!
tee < fifo/gul_S1_summary_P14.idx work/gul_S1_summaryleccalc/P14.idx > /dev/null & pid188=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_eltcalc_P15 fifo/gul_S1_summarycalc_P15 fifo/gul_S1_pltcalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid189=$!
tee < fifo/gul_S1_summary_P15.idx work/gul_S1_summaryleccalc/P15.idx > /dev/null & pid190=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_eltcalc_P16 fifo/gul_S1_summarycalc_P16 fifo/gul_S1_pltcalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid191=$!
tee < fifo/gul_S1_summary_P16.idx work/gul_S1_summaryleccalc/P16.idx > /dev/null & pid192=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_eltcalc_P17 fifo/gul_S1_summarycalc_P17 fifo/gul_S1_pltcalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid193=$!
tee < fifo/gul_S1_summary_P17.idx work/gul_S1_summaryleccalc/P17.idx > /dev/null & pid194=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_eltcalc_P18 fifo/gul_S1_summarycalc_P18 fifo/gul_S1_pltcalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid195=$!
tee < fifo/gul_S1_summary_P18.idx work/gul_S1_summaryleccalc/P18.idx > /dev/null & pid196=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_eltcalc_P19 fifo/gul_S1_summarycalc_P19 fifo/gul_S1_pltcalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid197=$!
tee < fifo/gul_S1_summary_P19.idx work/gul_S1_summaryleccalc/P19.idx > /dev/null & pid198=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_eltcalc_P20 fifo/gul_S1_summarycalc_P20 fifo/gul_S1_pltcalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid199=$!
tee < fifo/gul_S1_summary_P20.idx work/gul_S1_summaryleccalc/P20.idx > /dev/null & pid200=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 &

# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid201=$!
summarycalctocsv < fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid202=$!
pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid203=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid204=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid205=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid206=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid207=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P3 > work/full_correlation/kat/il_S1_summarycalc_P3 & pid208=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P3 > work/full_correlation/kat/il_S1_pltcalc_P3 & pid209=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid210=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P4 > work/full_correlation/kat/il_S1_summarycalc_P4 & pid211=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 & pid212=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 & pid213=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P5 > work/full_correlation/kat/il_S1_summarycalc_P5 & pid214=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P5 > work/full_correlation/kat/il_S1_pltcalc_P5 & pid215=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 & pid216=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P6 > work/full_correlation/kat/il_S1_summarycalc_P6 & pid217=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 & pid218=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid219=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P7 > work/full_correlation/kat/il_S1_summarycalc_P7 & pid220=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 & pid221=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 & pid222=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P8 > work/full_correlation/kat/il_S1_summarycalc_P8 & pid223=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P8 > work/full_correlation/kat/il_S1_pltcalc_P8 & pid224=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 & pid225=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P9 > work/full_correlation/kat/il_S1_summarycalc_P9 & pid226=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P9 > work/full_correlation/kat/il_S1_pltcalc_P9 & pid227=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 & pid228=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P10 > work/full_correlation/kat/il_S1_summarycalc_P10 & pid229=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P10 > work/full_correlation/kat/il_S1_pltcalc_P10 & pid230=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P11 > work/full_correlation/kat/il_S1_eltcalc_P11 & pid231=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P11 > work/full_correlation/kat/il_S1_summarycalc_P11 & pid232=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P11 > work/full_correlation/kat/il_S1_pltcalc_P11 & pid233=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P12 > work/full_correlation/kat/il_S1_eltcalc_P12 & pid234=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P12 > work/full_correlation/kat/il_S1_summarycalc_P12 & pid235=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P12 > work/full_correlation/kat/il_S1_pltcalc_P12 & pid236=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P13 > work/full_correlation/kat/il_S1_eltcalc_P13 & pid237=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P13 > work/full_correlation/kat/il_S1_summarycalc_P13 & pid238=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P13 > work/full_correlation/kat/il_S1_pltcalc_P13 & pid239=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P14 > work/full_correlation/kat/il_S1_eltcalc_P14 & pid240=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P14 > work/full_correlation/kat/il_S1_summarycalc_P14 & pid241=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P14 > work/full_correlation/kat/il_S1_pltcalc_P14 & pid242=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 & pid243=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P15 > work/full_correlation/kat/il_S1_summarycalc_P15 & pid244=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 & pid245=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P16 > work/full_correlation/kat/il_S1_eltcalc_P16 & pid246=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P16 > work/full_correlation/kat/il_S1_summarycalc_P16 & pid247=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P16 > work/full_correlation/kat/il_S1_pltcalc_P16 & pid248=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P17 > work/full_correlation/kat/il_S1_eltcalc_P17 & pid249=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P17 > work/full_correlation/kat/il_S1_summarycalc_P17 & pid250=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P17 > work/full_correlation/kat/il_S1_pltcalc_P17 & pid251=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P18 > work/full_correlation/kat/il_S1_eltcalc_P18 & pid252=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P18 > work/full_correlation/kat/il_S1_summarycalc_P18 & pid253=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P18 > work/full_correlation/kat/il_S1_pltcalc_P18 & pid254=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P19 > work/full_correlation/kat/il_S1_eltcalc_P19 & pid255=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P19 > work/full_correlation/kat/il_S1_summarycalc_P19 & pid256=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P19 > work/full_correlation/kat/il_S1_pltcalc_P19 & pid257=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 & pid258=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P20 > work/full_correlation/kat/il_S1_summarycalc_P20 & pid259=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P20 > work/full_correlation/kat/il_S1_pltcalc_P20 & pid260=$!


tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_eltcalc_P1 fifo/full_correlation/il_S1_summarycalc_P1 fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid261=$!
tee < fifo/full_correlation/il_S1_summary_P1.idx work/full_correlation/il_S1_summaryleccalc/P1.idx > /dev/null & pid262=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_eltcalc_P2 fifo/full_correlation/il_S1_summarycalc_P2 fifo/full_correlation/il_S1_pltcalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid263=$!
tee < fifo/full_correlation/il_S1_summary_P2.idx work/full_correlation/il_S1_summaryleccalc/P2.idx > /dev/null & pid264=$!
tee < fifo/full_correlation/il_S1_summary_P3 fifo/full_correlation/il_S1_eltcalc_P3 fifo/full_correlation/il_S1_summarycalc_P3 fifo/full_correlation/il_S1_pltcalc_P3 work/full_correlation/il_S1_summaryaalcalc/P3.bin work/full_correlation/il_S1_summaryleccalc/P3.bin > /dev/null & pid265=$!
tee < fifo/full_correlation/il_S1_summary_P3.idx work/full_correlation/il_S1_summaryleccalc/P3.idx > /dev/null & pid266=$!
tee < fifo/full_correlation/il_S1_summary_P4 fifo/full_correlation/il_S1_eltcalc_P4 fifo/full_correlation/il_S1_summarycalc_P4 fifo/full_correlation/il_S1_pltcalc_P4 work/full_correlation/il_S1_summaryaalcalc/P4.bin work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid267=$!
tee < fifo/full_correlation/il_S1_summary_P4.idx work/full_correlation/il_S1_summaryleccalc/P4.idx > /dev/null & pid268=$!
tee < fifo/full_correlation/il_S1_summary_P5 fifo/full_correlation/il_S1_eltcalc_P5 fifo/full_correlation/il_S1_summarycalc_P5 fifo/full_correlation/il_S1_pltcalc_P5 work/full_correlation/il_S1_summaryaalcalc/P5.bin work/full_correlation/il_S1_summaryleccalc/P5.bin > /dev/null & pid269=$!
tee < fifo/full_correlation/il_S1_summary_P5.idx work/full_correlation/il_S1_summaryleccalc/P5.idx > /dev/null & pid270=$!
tee < fifo/full_correlation/il_S1_summary_P6 fifo/full_correlation/il_S1_eltcalc_P6 fifo/full_correlation/il_S1_summarycalc_P6 fifo/full_correlation/il_S1_pltcalc_P6 work/full_correlation/il_S1_summaryaalcalc/P6.bin work/full_correlation/il_S1_summaryleccalc/P6.bin > /dev/null & pid271=$!
tee < fifo/full_correlation/il_S1_summary_P6.idx work/full_correlation/il_S1_summaryleccalc/P6.idx > /dev/null & pid272=$!
tee < fifo/full_correlation/il_S1_summary_P7 fifo/full_correlation/il_S1_eltcalc_P7 fifo/full_correlation/il_S1_summarycalc_P7 fifo/full_correlation/il_S1_pltcalc_P7 work/full_correlation/il_S1_summaryaalcalc/P7.bin work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid273=$!
tee < fifo/full_correlation/il_S1_summary_P7.idx work/full_correlation/il_S1_summaryleccalc/P7.idx > /dev/null & pid274=$!
tee < fifo/full_correlation/il_S1_summary_P8 fifo/full_correlation/il_S1_eltcalc_P8 fifo/full_correlation/il_S1_summarycalc_P8 fifo/full_correlation/il_S1_pltcalc_P8 work/full_correlation/il_S1_summaryaalcalc/P8.bin work/full_correlation/il_S1_summaryleccalc/P8.bin > /dev/null & pid275=$!
tee < fifo/full_correlation/il_S1_summary_P8.idx work/full_correlation/il_S1_summaryleccalc/P8.idx > /dev/null & pid276=$!
tee < fifo/full_correlation/il_S1_summary_P9 fifo/full_correlation/il_S1_eltcalc_P9 fifo/full_correlation/il_S1_summarycalc_P9 fifo/full_correlation/il_S1_pltcalc_P9 work/full_correlation/il_S1_summaryaalcalc/P9.bin work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid277=$!
tee < fifo/full_correlation/il_S1_summary_P9.idx work/full_correlation/il_S1_summaryleccalc/P9.idx > /dev/null & pid278=$!
tee < fifo/full_correlation/il_S1_summary_P10 fifo/full_correlation/il_S1_eltcalc_P10 fifo/full_correlation/il_S1_summarycalc_P10 fifo/full_correlation/il_S1_pltcalc_P10 work/full_correlation/il_S1_summaryaalcalc/P10.bin work/full_correlation/il_S1_summaryleccalc/P10.bin > /dev/null & pid279=$!
tee < fifo/full_correlation/il_S1_summary_P10.idx work/full_correlation/il_S1_summaryleccalc/P10.idx > /dev/null & pid280=$!
tee < fifo/full_correlation/il_S1_summary_P11 fifo/full_correlation/il_S1_eltcalc_P11 fifo/full_correlation/il_S1_summarycalc_P11 fifo/full_correlation/il_S1_pltcalc_P11 work/full_correlation/il_S1_summaryaalcalc/P11.bin work/full_correlation/il_S1_summaryleccalc/P11.bin > /dev/null & pid281=$!
tee < fifo/full_correlation/il_S1_summary_P11.idx work/full_correlation/il_S1_summaryleccalc/P11.idx > /dev/null & pid282=$!
tee < fifo/full_correlation/il_S1_summary_P12 fifo/full_correlation/il_S1_eltcalc_P12 fifo/full_correlation/il_S1_summarycalc_P12 fifo/full_correlation/il_S1_pltcalc_P12 work/full_correlation/il_S1_summaryaalcalc/P12.bin work/full_correlation/il_S1_summaryleccalc/P12.bin > /dev/null & pid283=$!
tee < fifo/full_correlation/il_S1_summary_P12.idx work/full_correlation/il_S1_summaryleccalc/P12.idx > /dev/null & pid284=$!
tee < fifo/full_correlation/il_S1_summary_P13 fifo/full_correlation/il_S1_eltcalc_P13 fifo/full_correlation/il_S1_summarycalc_P13 fifo/full_correlation/il_S1_pltcalc_P13 work/full_correlation/il_S1_summaryaalcalc/P13.bin work/full_correlation/il_S1_summaryleccalc/P13.bin > /dev/null & pid285=$!
tee < fifo/full_correlation/il_S1_summary_P13.idx work/full_correlation/il_S1_summaryleccalc/P13.idx > /dev/null & pid286=$!
tee < fifo/full_correlation/il_S1_summary_P14 fifo/full_correlation/il_S1_eltcalc_P14 fifo/full_correlation/il_S1_summarycalc_P14 fifo/full_correlation/il_S1_pltcalc_P14 work/full_correlation/il_S1_summaryaalcalc/P14.bin work/full_correlation/il_S1_summaryleccalc/P14.bin > /dev/null & pid287=$!
tee < fifo/full_correlation/il_S1_summary_P14.idx work/full_correlation/il_S1_summaryleccalc/P14.idx > /dev/null & pid288=$!
tee < fifo/full_correlation/il_S1_summary_P15 fifo/full_correlation/il_S1_eltcalc_P15 fifo/full_correlation/il_S1_summarycalc_P15 fifo/full_correlation/il_S1_pltcalc_P15 work/full_correlation/il_S1_summaryaalcalc/P15.bin work/full_correlation/il_S1_summaryleccalc/P15.bin > /dev/null & pid289=$!
tee < fifo/full_correlation/il_S1_summary_P15.idx work/full_correlation/il_S1_summaryleccalc/P15.idx > /dev/null & pid290=$!
tee < fifo/full_correlation/il_S1_summary_P16 fifo/full_correlation/il_S1_eltcalc_P16 fifo/full_correlation/il_S1_summarycalc_P16 fifo/full_correlation/il_S1_pltcalc_P16 work/full_correlation/il_S1_summaryaalcalc/P16.bin work/full_correlation/il_S1_summaryleccalc/P16.bin > /dev/null & pid291=$!
tee < fifo/full_correlation/il_S1_summary_P16.idx work/full_correlation/il_S1_summaryleccalc/P16.idx > /dev/null & pid292=$!
tee < fifo/full_correlation/il_S1_summary_P17 fifo/full_correlation/il_S1_eltcalc_P17 fifo/full_correlation/il_S1_summarycalc_P17 fifo/full_correlation/il_S1_pltcalc_P17 work/full_correlation/il_S1_summaryaalcalc/P17.bin work/full_correlation/il_S1_summaryleccalc/P17.bin > /dev/null & pid293=$!
tee < fifo/full_correlation/il_S1_summary_P17.idx work/full_correlation/il_S1_summaryleccalc/P17.idx > /dev/null & pid294=$!
tee < fifo/full_correlation/il_S1_summary_P18 fifo/full_correlation/il_S1_eltcalc_P18 fifo/full_correlation/il_S1_summarycalc_P18 fifo/full_correlation/il_S1_pltcalc_P18 work/full_correlation/il_S1_summaryaalcalc/P18.bin work/full_correlation/il_S1_summaryleccalc/P18.bin > /dev/null & pid295=$!
tee < fifo/full_correlation/il_S1_summary_P18.idx work/full_correlation/il_S1_summaryleccalc/P18.idx > /dev/null & pid296=$!
tee < fifo/full_correlation/il_S1_summary_P19 fifo/full_correlation/il_S1_eltcalc_P19 fifo/full_correlation/il_S1_summarycalc_P19 fifo/full_correlation/il_S1_pltcalc_P19 work/full_correlation/il_S1_summaryaalcalc/P19.bin work/full_correlation/il_S1_summaryleccalc/P19.bin > /dev/null & pid297=$!
tee < fifo/full_correlation/il_S1_summary_P19.idx work/full_correlation/il_S1_summaryleccalc/P19.idx > /dev/null & pid298=$!
tee < fifo/full_correlation/il_S1_summary_P20 fifo/full_correlation/il_S1_eltcalc_P20 fifo/full_correlation/il_S1_summarycalc_P20 fifo/full_correlation/il_S1_pltcalc_P20 work/full_correlation/il_S1_summaryaalcalc/P20.bin work/full_correlation/il_S1_summaryleccalc/P20.bin > /dev/null & pid299=$!
tee < fifo/full_correlation/il_S1_summary_P20.idx work/full_correlation/il_S1_summaryleccalc/P20.idx > /dev/null & pid300=$!

summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P2 < fifo/full_correlation/il_P2 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P3 < fifo/full_correlation/il_P3 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P4 < fifo/full_correlation/il_P4 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P5 < fifo/full_correlation/il_P5 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P6 < fifo/full_correlation/il_P6 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P7 < fifo/full_correlation/il_P7 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P8 < fifo/full_correlation/il_P8 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P9 < fifo/full_correlation/il_P9 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P10 < fifo/full_correlation/il_P10 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P11 < fifo/full_correlation/il_P11 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P12 < fifo/full_correlation/il_P12 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P13 < fifo/full_correlation/il_P13 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P14 < fifo/full_correlation/il_P14 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P15 < fifo/full_correlation/il_P15 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P17 < fifo/full_correlation/il_P17 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P18 < fifo/full_correlation/il_P18 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P19 < fifo/full_correlation/il_P19 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P20 < fifo/full_correlation/il_P20 &

# --- Do ground up loss computes ---

eltcalc < fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid301=$!
summarycalctocsv < fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid302=$!
pltcalc < fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid303=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid304=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid305=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid306=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P3 > work/full_correlation/kat/gul_S1_eltcalc_P3 & pid307=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P3 > work/full_correlation/kat/gul_S1_summarycalc_P3 & pid308=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P3 > work/full_correlation/kat/gul_S1_pltcalc_P3 & pid309=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid310=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 & pid311=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 & pid312=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P5 > work/full_correlation/kat/gul_S1_eltcalc_P5 & pid313=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P5 > work/full_correlation/kat/gul_S1_summarycalc_P5 & pid314=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P5 > work/full_correlation/kat/gul_S1_pltcalc_P5 & pid315=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P6 > work/full_correlation/kat/gul_S1_eltcalc_P6 & pid316=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P6 > work/full_correlation/kat/gul_S1_summarycalc_P6 & pid317=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P6 > work/full_correlation/kat/gul_S1_pltcalc_P6 & pid318=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 & pid319=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 & pid320=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 & pid321=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P8 > work/full_correlation/kat/gul_S1_eltcalc_P8 & pid322=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P8 > work/full_correlation/kat/gul_S1_summarycalc_P8 & pid323=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P8 > work/full_correlation/kat/gul_S1_pltcalc_P8 & pid324=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 & pid325=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P9 > work/full_correlation/kat/gul_S1_summarycalc_P9 & pid326=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P9 > work/full_correlation/kat/gul_S1_pltcalc_P9 & pid327=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P10 > work/full_correlation/kat/gul_S1_eltcalc_P10 & pid328=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 & pid329=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P10 > work/full_correlation/kat/gul_S1_pltcalc_P10 & pid330=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P11 > work/full_correlation/kat/gul_S1_eltcalc_P11 & pid331=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P11 > work/full_correlation/kat/gul_S1_summarycalc_P11 & pid332=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P11 > work/full_correlation/kat/gul_S1_pltcalc_P11 & pid333=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P12 > work/full_correlation/kat/gul_S1_eltcalc_P12 & pid334=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P12 > work/full_correlation/kat/gul_S1_summarycalc_P12 & pid335=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P12 > work/full_correlation/kat/gul_S1_pltcalc_P12 & pid336=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P13 > work/full_correlation/kat/gul_S1_eltcalc_P13 & pid337=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P13 > work/full_correlation/kat/gul_S1_summarycalc_P13 & pid338=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P13 > work/full_correlation/kat/gul_S1_pltcalc_P13 & pid339=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P14 > work/full_correlation/kat/gul_S1_eltcalc_P14 & pid340=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P14 > work/full_correlation/kat/gul_S1_summarycalc_P14 & pid341=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P14 > work/full_correlation/kat/gul_S1_pltcalc_P14 & pid342=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P15 > work/full_correlation/kat/gul_S1_eltcalc_P15 & pid343=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P15 > work/full_correlation/kat/gul_S1_summarycalc_P15 & pid344=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P15 > work/full_correlation/kat/gul_S1_pltcalc_P15 & pid345=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P16 > work/full_correlation/kat/gul_S1_eltcalc_P16 & pid346=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P16 > work/full_correlation/kat/gul_S1_summarycalc_P16 & pid347=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P16 > work/full_correlation/kat/gul_S1_pltcalc_P16 & pid348=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P17 > work/full_correlation/kat/gul_S1_eltcalc_P17 & pid349=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P17 > work/full_correlation/kat/gul_S1_summarycalc_P17 & pid350=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P17 > work/full_correlation/kat/gul_S1_pltcalc_P17 & pid351=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P18 > work/full_correlation/kat/gul_S1_eltcalc_P18 & pid352=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P18 > work/full_correlation/kat/gul_S1_summarycalc_P18 & pid353=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P18 > work/full_correlation/kat/gul_S1_pltcalc_P18 & pid354=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P19 > work/full_correlation/kat/gul_S1_eltcalc_P19 & pid355=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P19 > work/full_correlation/kat/gul_S1_summarycalc_P19 & pid356=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P19 > work/full_correlation/kat/gul_S1_pltcalc_P19 & pid357=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P20 > work/full_correlation/kat/gul_S1_eltcalc_P20 & pid358=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P20 > work/full_correlation/kat/gul_S1_summarycalc_P20 & pid359=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P20 > work/full_correlation/kat/gul_S1_pltcalc_P20 & pid360=$!


tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_eltcalc_P1 fifo/full_correlation/gul_S1_summarycalc_P1 fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid361=$!
tee < fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid362=$!
tee < fifo/full_correlation/gul_S1_summary_P2 fifo/full_correlation/gul_S1_eltcalc_P2 fifo/full_correlation/gul_S1_summarycalc_P2 fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid363=$!
tee < fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryleccalc/P2.idx > /dev/null & pid364=$!
tee < fifo/full_correlation/gul_S1_summary_P3 fifo/full_correlation/gul_S1_eltcalc_P3 fifo/full_correlation/gul_S1_summarycalc_P3 fifo/full_correlation/gul_S1_pltcalc_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid365=$!
tee < fifo/full_correlation/gul_S1_summary_P3.idx work/full_correlation/gul_S1_summaryleccalc/P3.idx > /dev/null & pid366=$!
tee < fifo/full_correlation/gul_S1_summary_P4 fifo/full_correlation/gul_S1_eltcalc_P4 fifo/full_correlation/gul_S1_summarycalc_P4 fifo/full_correlation/gul_S1_pltcalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid367=$!
tee < fifo/full_correlation/gul_S1_summary_P4.idx work/full_correlation/gul_S1_summaryleccalc/P4.idx > /dev/null & pid368=$!
tee < fifo/full_correlation/gul_S1_summary_P5 fifo/full_correlation/gul_S1_eltcalc_P5 fifo/full_correlation/gul_S1_summarycalc_P5 fifo/full_correlation/gul_S1_pltcalc_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid369=$!
tee < fifo/full_correlation/gul_S1_summary_P5.idx work/full_correlation/gul_S1_summaryleccalc/P5.idx > /dev/null & pid370=$!
tee < fifo/full_correlation/gul_S1_summary_P6 fifo/full_correlation/gul_S1_eltcalc_P6 fifo/full_correlation/gul_S1_summarycalc_P6 fifo/full_correlation/gul_S1_pltcalc_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid371=$!
tee < fifo/full_correlation/gul_S1_summary_P6.idx work/full_correlation/gul_S1_summaryleccalc/P6.idx > /dev/null & pid372=$!
tee < fifo/full_correlation/gul_S1_summary_P7 fifo/full_correlation/gul_S1_eltcalc_P7 fifo/full_correlation/gul_S1_summarycalc_P7 fifo/full_correlation/gul_S1_pltcalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid373=$!
tee < fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryleccalc/P7.idx > /dev/null & pid374=$!
tee < fifo/full_correlation/gul_S1_summary_P8 fifo/full_correlation/gul_S1_eltcalc_P8 fifo/full_correlation/gul_S1_summarycalc_P8 fifo/full_correlation/gul_S1_pltcalc_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid375=$!
tee < fifo/full_correlation/gul_S1_summary_P8.idx work/full_correlation/gul_S1_summaryleccalc/P8.idx > /dev/null & pid376=$!
tee < fifo/full_correlation/gul_S1_summary_P9 fifo/full_correlation/gul_S1_eltcalc_P9 fifo/full_correlation/gul_S1_summarycalc_P9 fifo/full_correlation/gul_S1_pltcalc_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid377=$!
tee < fifo/full_correlation/gul_S1_summary_P9.idx work/full_correlation/gul_S1_summaryleccalc/P9.idx > /dev/null & pid378=$!
tee < fifo/full_correlation/gul_S1_summary_P10 fifo/full_correlation/gul_S1_eltcalc_P10 fifo/full_correlation/gul_S1_summarycalc_P10 fifo/full_correlation/gul_S1_pltcalc_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid379=$!
tee < fifo/full_correlation/gul_S1_summary_P10.idx work/full_correlation/gul_S1_summaryleccalc/P10.idx > /dev/null & pid380=$!
tee < fifo/full_correlation/gul_S1_summary_P11 fifo/full_correlation/gul_S1_eltcalc_P11 fifo/full_correlation/gul_S1_summarycalc_P11 fifo/full_correlation/gul_S1_pltcalc_P11 work/full_correlation/gul_S1_summaryaalcalc/P11.bin work/full_correlation/gul_S1_summaryleccalc/P11.bin > /dev/null & pid381=$!
tee < fifo/full_correlation/gul_S1_summary_P11.idx work/full_correlation/gul_S1_summaryleccalc/P11.idx > /dev/null & pid382=$!
tee < fifo/full_correlation/gul_S1_summary_P12 fifo/full_correlation/gul_S1_eltcalc_P12 fifo/full_correlation/gul_S1_summarycalc_P12 fifo/full_correlation/gul_S1_pltcalc_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin work/full_correlation/gul_S1_summaryleccalc/P12.bin > /dev/null & pid383=$!
tee < fifo/full_correlation/gul_S1_summary_P12.idx work/full_correlation/gul_S1_summaryleccalc/P12.idx > /dev/null & pid384=$!
tee < fifo/full_correlation/gul_S1_summary_P13 fifo/full_correlation/gul_S1_eltcalc_P13 fifo/full_correlation/gul_S1_summarycalc_P13 fifo/full_correlation/gul_S1_pltcalc_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin work/full_correlation/gul_S1_summaryleccalc/P13.bin > /dev/null & pid385=$!
tee < fifo/full_correlation/gul_S1_summary_P13.idx work/full_correlation/gul_S1_summaryleccalc/P13.idx > /dev/null & pid386=$!
tee < fifo/full_correlation/gul_S1_summary_P14 fifo/full_correlation/gul_S1_eltcalc_P14 fifo/full_correlation/gul_S1_summarycalc_P14 fifo/full_correlation/gul_S1_pltcalc_P14 work/full_correlation/gul_S1_summaryaalcalc/P14.bin work/full_correlation/gul_S1_summaryleccalc/P14.bin > /dev/null & pid387=$!
tee < fifo/full_correlation/gul_S1_summary_P14.idx work/full_correlation/gul_S1_summaryleccalc/P14.idx > /dev/null & pid388=$!
tee < fifo/full_correlation/gul_S1_summary_P15 fifo/full_correlation/gul_S1_eltcalc_P15 fifo/full_correlation/gul_S1_summarycalc_P15 fifo/full_correlation/gul_S1_pltcalc_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin work/full_correlation/gul_S1_summaryleccalc/P15.bin > /dev/null & pid389=$!
tee < fifo/full_correlation/gul_S1_summary_P15.idx work/full_correlation/gul_S1_summaryleccalc/P15.idx > /dev/null & pid390=$!
tee < fifo/full_correlation/gul_S1_summary_P16 fifo/full_correlation/gul_S1_eltcalc_P16 fifo/full_correlation/gul_S1_summarycalc_P16 fifo/full_correlation/gul_S1_pltcalc_P16 work/full_correlation/gul_S1_summaryaalcalc/P16.bin work/full_correlation/gul_S1_summaryleccalc/P16.bin > /dev/null & pid391=$!
tee < fifo/full_correlation/gul_S1_summary_P16.idx work/full_correlation/gul_S1_summaryleccalc/P16.idx > /dev/null & pid392=$!
tee < fifo/full_correlation/gul_S1_summary_P17 fifo/full_correlation/gul_S1_eltcalc_P17 fifo/full_correlation/gul_S1_summarycalc_P17 fifo/full_correlation/gul_S1_pltcalc_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin work/full_correlation/gul_S1_summaryleccalc/P17.bin > /dev/null & pid393=$!
tee < fifo/full_correlation/gul_S1_summary_P17.idx work/full_correlation/gul_S1_summaryleccalc/P17.idx > /dev/null & pid394=$!
tee < fifo/full_correlation/gul_S1_summary_P18 fifo/full_correlation/gul_S1_eltcalc_P18 fifo/full_correlation/gul_S1_summarycalc_P18 fifo/full_correlation/gul_S1_pltcalc_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin work/full_correlation/gul_S1_summaryleccalc/P18.bin > /dev/null & pid395=$!
tee < fifo/full_correlation/gul_S1_summary_P18.idx work/full_correlation/gul_S1_summaryleccalc/P18.idx > /dev/null & pid396=$!
tee < fifo/full_correlation/gul_S1_summary_P19 fifo/full_correlation/gul_S1_eltcalc_P19 fifo/full_correlation/gul_S1_summarycalc_P19 fifo/full_correlation/gul_S1_pltcalc_P19 work/full_correlation/gul_S1_summaryaalcalc/P19.bin work/full_correlation/gul_S1_summaryleccalc/P19.bin > /dev/null & pid397=$!
tee < fifo/full_correlation/gul_S1_summary_P19.idx work/full_correlation/gul_S1_summaryleccalc/P19.idx > /dev/null & pid398=$!
tee < fifo/full_correlation/gul_S1_summary_P20 fifo/full_correlation/gul_S1_eltcalc_P20 fifo/full_correlation/gul_S1_summarycalc_P20 fifo/full_correlation/gul_S1_pltcalc_P20 work/full_correlation/gul_S1_summaryaalcalc/P20.bin work/full_correlation/gul_S1_summaryleccalc/P20.bin > /dev/null & pid399=$!
tee < fifo/full_correlation/gul_S1_summary_P20.idx work/full_correlation/gul_S1_summaryleccalc/P20.idx > /dev/null & pid400=$!

summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P2 < fifo/full_correlation/gul_P2 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P3 < fifo/full_correlation/gul_P3 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P4 < fifo/full_correlation/gul_P4 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P5 < fifo/full_correlation/gul_P5 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P6 < fifo/full_correlation/gul_P6 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P7 < fifo/full_correlation/gul_P7 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P8 < fifo/full_correlation/gul_P8 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P9 < fifo/full_correlation/gul_P9 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P10 < fifo/full_correlation/gul_P10 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P11 < fifo/full_correlation/gul_P11 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P12 < fifo/full_correlation/gul_P12 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P13 < fifo/full_correlation/gul_P13 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P14 < fifo/full_correlation/gul_P14 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P15 < fifo/full_correlation/gul_P15 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P16 < fifo/full_correlation/gul_P16 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P17 < fifo/full_correlation/gul_P17 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P18 < fifo/full_correlation/gul_P18 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P19 < fifo/full_correlation/gul_P19 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P20 < fifo/full_correlation/gul_P20 &

tee < fifo/full_correlation/gul_fc_P1 fifo/full_correlation/gul_P1  | fmcalc -a2 > fifo/full_correlation/il_P1  &
tee < fifo/full_correlation/gul_fc_P2 fifo/full_correlation/gul_P2  | fmcalc -a2 > fifo/full_correlation/il_P2  &
tee < fifo/full_correlation/gul_fc_P3 fifo/full_correlation/gul_P3  | fmcalc -a2 > fifo/full_correlation/il_P3  &
tee < fifo/full_correlation/gul_fc_P4 fifo/full_correlation/gul_P4  | fmcalc -a2 > fifo/full_correlation/il_P4  &
tee < fifo/full_correlation/gul_fc_P5 fifo/full_correlation/gul_P5  | fmcalc -a2 > fifo/full_correlation/il_P5  &
tee < fifo/full_correlation/gul_fc_P6 fifo/full_correlation/gul_P6  | fmcalc -a2 > fifo/full_correlation/il_P6  &
tee < fifo/full_correlation/gul_fc_P7 fifo/full_correlation/gul_P7  | fmcalc -a2 > fifo/full_correlation/il_P7  &
tee < fifo/full_correlation/gul_fc_P8 fifo/full_correlation/gul_P8  | fmcalc -a2 > fifo/full_correlation/il_P8  &
tee < fifo/full_correlation/gul_fc_P9 fifo/full_correlation/gul_P9  | fmcalc -a2 > fifo/full_correlation/il_P9  &
tee < fifo/full_correlation/gul_fc_P10 fifo/full_correlation/gul_P10  | fmcalc -a2 > fifo/full_correlation/il_P10  &
tee < fifo/full_correlation/gul_fc_P11 fifo/full_correlation/gul_P11  | fmcalc -a2 > fifo/full_correlation/il_P11  &
tee < fifo/full_correlation/gul_fc_P12 fifo/full_correlation/gul_P12  | fmcalc -a2 > fifo/full_correlation/il_P12  &
tee < fifo/full_correlation/gul_fc_P13 fifo/full_correlation/gul_P13  | fmcalc -a2 > fifo/full_correlation/il_P13  &
tee < fifo/full_correlation/gul_fc_P14 fifo/full_correlation/gul_P14  | fmcalc -a2 > fifo/full_correlation/il_P14  &
tee < fifo/full_correlation/gul_fc_P15 fifo/full_correlation/gul_P15  | fmcalc -a2 > fifo/full_correlation/il_P15  &
tee < fifo/full_correlation/gul_fc_P16 fifo/full_correlation/gul_P16  | fmcalc -a2 > fifo/full_correlation/il_P16  &
tee < fifo/full_correlation/gul_fc_P17 fifo/full_correlation/gul_P17  | fmcalc -a2 > fifo/full_correlation/il_P17  &
tee < fifo/full_correlation/gul_fc_P18 fifo/full_correlation/gul_P18  | fmcalc -a2 > fifo/full_correlation/il_P18  &
tee < fifo/full_correlation/gul_fc_P19 fifo/full_correlation/gul_P19  | fmcalc -a2 > fifo/full_correlation/il_P19  &
tee < fifo/full_correlation/gul_fc_P20 fifo/full_correlation/gul_P20  | fmcalc -a2 > fifo/full_correlation/il_P20  &
eve 1 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P2 -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  &
eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P3 -a1 -i - | tee fifo/gul_P3 | fmcalc -a2 > fifo/il_P3  &
eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P4 -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P5 -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P6 -a1 -i - | tee fifo/gul_P6 | fmcalc -a2 > fifo/il_P6  &
eve 7 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P7 -a1 -i - | tee fifo/gul_P7 | fmcalc -a2 > fifo/il_P7  &
eve 8 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P8 -a1 -i - | tee fifo/gul_P8 | fmcalc -a2 > fifo/il_P8  &
eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P9 -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  &
eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P10 -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  &
eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P11 -a1 -i - | tee fifo/gul_P11 | fmcalc -a2 > fifo/il_P11  &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P12 -a1 -i - | tee fifo/gul_P12 | fmcalc -a2 > fifo/il_P12  &
eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P13 -a1 -i - | tee fifo/gul_P13 | fmcalc -a2 > fifo/il_P13  &
eve 14 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P14 -a1 -i - | tee fifo/gul_P14 | fmcalc -a2 > fifo/il_P14  &
eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P15 -a1 -i - | tee fifo/gul_P15 | fmcalc -a2 > fifo/il_P15  &
eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P16 -a1 -i - | tee fifo/gul_P16 | fmcalc -a2 > fifo/il_P16  &
eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P17 -a1 -i - | tee fifo/gul_P17 | fmcalc -a2 > fifo/il_P17  &
eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P18 -a1 -i - | tee fifo/gul_P18 | fmcalc -a2 > fifo/il_P18  &
eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P19 -a1 -i - | tee fifo/gul_P19 | fmcalc -a2 > fifo/il_P19  &
eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P20 -a1 -i - | tee fifo/gul_P20 | fmcalc -a2 > fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210 $pid211 $pid212 $pid213 $pid214 $pid215 $pid216 $pid217 $pid218 $pid219 $pid220 $pid221 $pid222 $pid223 $pid224 $pid225 $pid226 $pid227 $pid228 $pid229 $pid230 $pid231 $pid232 $pid233 $pid234 $pid235 $pid236 $pid237 $pid238 $pid239 $pid240 $pid241 $pid242 $pid243 $pid244 $pid245 $pid246 $pid247 $pid248 $pid249 $pid250 $pid251 $pid252 $pid253 $pid254 $pid255 $pid256 $pid257 $pid258 $pid259 $pid260 $pid261 $pid262 $pid263 $pid264 $pid265 $pid266 $pid267 $pid268 $pid269 $pid270 $pid271 $pid272 $pid273 $pid274 $pid275 $pid276 $pid277 $pid278 $pid279 $pid280 $pid281 $pid282 $pid283 $pid284 $pid285 $pid286 $pid287 $pid288 $pid289 $pid290 $pid291 $pid292 $pid293 $pid294 $pid295 $pid296 $pid297 $pid298 $pid299 $pid300 $pid301 $pid302 $pid303 $pid304 $pid305 $pid306 $pid307 $pid308 $pid309 $pid310 $pid311 $pid312 $pid313 $pid314 $pid315 $pid316 $pid317 $pid318 $pid319 $pid320 $pid321 $pid322 $pid323 $pid324 $pid325 $pid326 $pid327 $pid328 $pid329 $pid330 $pid331 $pid332 $pid333 $pid334 $pid335 $pid336 $pid337 $pid338 $pid339 $pid340 $pid341 $pid342 $pid343 $pid344 $pid345 $pid346 $pid347 $pid348 $pid349 $pid350 $pid351 $pid352 $pid353 $pid354 $pid355 $pid356 $pid357 $pid358 $pid359 $pid360 $pid361 $pid362 $pid363 $pid364 $pid365 $pid366 $pid367 $pid368 $pid369 $pid370 $pid371 $pid372 $pid373 $pid374 $pid375 $pid376 $pid377 $pid378 $pid379 $pid380 $pid381 $pid382 $pid383 $pid384 $pid385 $pid386 $pid387 $pid388 $pid389 $pid390 $pid391 $pid392 $pid393 $pid394 $pid395 $pid396 $pid397 $pid398 $pid399 $pid400


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 work/full_correlation/kat/il_S1_eltcalc_P11 work/full_correlation/kat/il_S1_eltcalc_P12 work/full_correlation/kat/il_S1_eltcalc_P13 work/full_correlation/kat/il_S1_eltcalc_P14 work/full_correlation/kat/il_S1_eltcalc_P15 work/full_correlation/kat/il_S1_eltcalc_P16 work/full_correlation/kat/il_S1_eltcalc_P17 work/full_correlation/kat/il_S1_eltcalc_P18 work/full_correlation/kat/il_S1_eltcalc_P19 work/full_correlation/kat/il_S1_eltcalc_P20 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 work/full_correlation/kat/il_S1_pltcalc_P3 work/full_correlation/kat/il_S1_pltcalc_P4 work/full_correlation/kat/il_S1_pltcalc_P5 work/full_correlation/kat/il_S1_pltcalc_P6 work/full_correlation/kat/il_S1_pltcalc_P7 work/full_correlation/kat/il_S1_pltcalc_P8 work/full_correlation/kat/il_S1_pltcalc_P9 work/full_correlation/kat/il_S1_pltcalc_P10 work/full_correlation/kat/il_S1_pltcalc_P11 work/full_correlation/kat/il_S1_pltcalc_P12 work/full_correlation/kat/il_S1_pltcalc_P13 work/full_correlation/kat/il_S1_pltcalc_P14 work/full_correlation/kat/il_S1_pltcalc_P15 work/full_correlation/kat/il_S1_pltcalc_P16 work/full_correlation/kat/il_S1_pltcalc_P17 work/full_correlation/kat/il_S1_pltcalc_P18 work/full_correlation/kat/il_S1_pltcalc_P19 work/full_correlation/kat/il_S1_pltcalc_P20 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 work/full_correlation/kat/il_S1_summarycalc_P2 work/full_correlation/kat/il_S1_summarycalc_P3 work/full_correlation/kat/il_S1_summarycalc_P4 work/full_correlation/kat/il_S1_summarycalc_P5 work/full_correlation/kat/il_S1_summarycalc_P6 work/full_correlation/kat/il_S1_summarycalc_P7 work/full_correlation/kat/il_S1_summarycalc_P8 work/full_correlation/kat/il_S1_summarycalc_P9 work/full_correlation/kat/il_S1_summarycalc_P10 work/full_correlation/kat/il_S1_summarycalc_P11 work/full_correlation/kat/il_S1_summarycalc_P12 work/full_correlation/kat/il_S1_summarycalc_P13 work/full_correlation/kat/il_S1_summarycalc_P14 work/full_correlation/kat/il_S1_summarycalc_P15 work/full_correlation/kat/il_S1_summarycalc_P16 work/full_correlation/kat/il_S1_summarycalc_P17 work/full_correlation/kat/il_S1_summarycalc_P18 work/full_correlation/kat/il_S1_summarycalc_P19 work/full_correlation/kat/il_S1_summarycalc_P20 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 work/full_correlation/kat/gul_S1_eltcalc_P3 work/full_correlation/kat/gul_S1_eltcalc_P4 work/full_correlation/kat/gul_S1_eltcalc_P5 work/full_correlation/kat/gul_S1_eltcalc_P6 work/full_correlation/kat/gul_S1_eltcalc_P7 work/full_correlation/kat/gul_S1_eltcalc_P8 work/full_correlation/kat/gul_S1_eltcalc_P9 work/full_correlation/kat/gul_S1_eltcalc_P10 work/full_correlation/kat/gul_S1_eltcalc_P11 work/full_correlation/kat/gul_S1_eltcalc_P12 work/full_correlation/kat/gul_S1_eltcalc_P13 work/full_correlation/kat/gul_S1_eltcalc_P14 work/full_correlation/kat/gul_S1_eltcalc_P15 work/full_correlation/kat/gul_S1_eltcalc_P16 work/full_correlation/kat/gul_S1_eltcalc_P17 work/full_correlation/kat/gul_S1_eltcalc_P18 work/full_correlation/kat/gul_S1_eltcalc_P19 work/full_correlation/kat/gul_S1_eltcalc_P20 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
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
rm -R -f fifo/*
