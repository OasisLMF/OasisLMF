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
mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S2_summaryleccalc
mkdir work/full_correlation/gul_S2_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S2_summaryleccalc
mkdir work/full_correlation/il_S2_summaryaalcalc

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
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P10

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

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P10

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
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P10

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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P2.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P3.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P3

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P4.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P5.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P6.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P7
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P8.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P8
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P8

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P9.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P9
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P9

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P10.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P10



# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P1 > work/kat/il_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P1 > work/kat/il_S2_summarycalc_P1 & pid5=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P1 > work/kat/il_S2_pltcalc_P1 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid8=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P2 > work/kat/il_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P2 > work/kat/il_S2_summarycalc_P2 & pid11=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P2 > work/kat/il_S2_pltcalc_P2 & pid12=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid13=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid14=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid15=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P3 > work/kat/il_S2_eltcalc_P3 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P3 > work/kat/il_S2_summarycalc_P3 & pid17=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P3 > work/kat/il_S2_pltcalc_P3 & pid18=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid19=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid20=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid21=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P4 > work/kat/il_S2_eltcalc_P4 & pid22=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P4 > work/kat/il_S2_summarycalc_P4 & pid23=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P4 > work/kat/il_S2_pltcalc_P4 & pid24=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid25=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid26=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid27=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P5 > work/kat/il_S2_eltcalc_P5 & pid28=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P5 > work/kat/il_S2_summarycalc_P5 & pid29=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P5 > work/kat/il_S2_pltcalc_P5 & pid30=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid31=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid32=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid33=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P6 > work/kat/il_S2_eltcalc_P6 & pid34=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P6 > work/kat/il_S2_summarycalc_P6 & pid35=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P6 > work/kat/il_S2_pltcalc_P6 & pid36=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid37=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid38=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid39=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P7 > work/kat/il_S2_eltcalc_P7 & pid40=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P7 > work/kat/il_S2_summarycalc_P7 & pid41=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P7 > work/kat/il_S2_pltcalc_P7 & pid42=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid43=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid44=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid45=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P8 > work/kat/il_S2_eltcalc_P8 & pid46=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P8 > work/kat/il_S2_summarycalc_P8 & pid47=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P8 > work/kat/il_S2_pltcalc_P8 & pid48=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid49=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid50=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid51=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P9 > work/kat/il_S2_eltcalc_P9 & pid52=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P9 > work/kat/il_S2_summarycalc_P9 & pid53=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P9 > work/kat/il_S2_pltcalc_P9 & pid54=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid55=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid56=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid57=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P10 > work/kat/il_S2_eltcalc_P10 & pid58=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P10 > work/kat/il_S2_summarycalc_P10 & pid59=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P10 > work/kat/il_S2_pltcalc_P10 & pid60=$!


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid62=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P1 work/il_S2_summaryaalcalc/P1.bin work/il_S2_summaryleccalc/P1.bin > /dev/null & pid63=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1.idx work/il_S2_summaryaalcalc/P1.idx work/il_S2_summaryleccalc/P1.idx > /dev/null & pid64=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid65=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid66=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P2 work/il_S2_summaryaalcalc/P2.bin work/il_S2_summaryleccalc/P2.bin > /dev/null & pid67=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2.idx work/il_S2_summaryaalcalc/P2.idx work/il_S2_summaryleccalc/P2.idx > /dev/null & pid68=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid69=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3.idx work/il_S1_summaryaalcalc/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid70=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P3 work/il_S2_summaryaalcalc/P3.bin work/il_S2_summaryleccalc/P3.bin > /dev/null & pid71=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3.idx work/il_S2_summaryaalcalc/P3.idx work/il_S2_summaryleccalc/P3.idx > /dev/null & pid72=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid73=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4.idx work/il_S1_summaryaalcalc/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid74=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P4 work/il_S2_summaryaalcalc/P4.bin work/il_S2_summaryleccalc/P4.bin > /dev/null & pid75=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4.idx work/il_S2_summaryaalcalc/P4.idx work/il_S2_summaryleccalc/P4.idx > /dev/null & pid76=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid77=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5.idx work/il_S1_summaryaalcalc/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid78=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P5 work/il_S2_summaryaalcalc/P5.bin work/il_S2_summaryleccalc/P5.bin > /dev/null & pid79=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5.idx work/il_S2_summaryaalcalc/P5.idx work/il_S2_summaryleccalc/P5.idx > /dev/null & pid80=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid81=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx work/il_S1_summaryaalcalc/P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid82=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P6 work/il_S2_summaryaalcalc/P6.bin work/il_S2_summaryleccalc/P6.bin > /dev/null & pid83=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6.idx work/il_S2_summaryaalcalc/P6.idx work/il_S2_summaryleccalc/P6.idx > /dev/null & pid84=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid85=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryaalcalc/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid86=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P7 work/il_S2_summaryaalcalc/P7.bin work/il_S2_summaryleccalc/P7.bin > /dev/null & pid87=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7.idx work/il_S2_summaryaalcalc/P7.idx work/il_S2_summaryleccalc/P7.idx > /dev/null & pid88=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid89=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8.idx work/il_S1_summaryaalcalc/P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid90=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P8 work/il_S2_summaryaalcalc/P8.bin work/il_S2_summaryleccalc/P8.bin > /dev/null & pid91=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8.idx work/il_S2_summaryaalcalc/P8.idx work/il_S2_summaryleccalc/P8.idx > /dev/null & pid92=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid93=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9.idx work/il_S1_summaryaalcalc/P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid94=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P9 work/il_S2_summaryaalcalc/P9.bin work/il_S2_summaryleccalc/P9.bin > /dev/null & pid95=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9.idx work/il_S2_summaryaalcalc/P9.idx work/il_S2_summaryleccalc/P9.idx > /dev/null & pid96=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid97=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10.idx work/il_S1_summaryaalcalc/P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid98=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P10 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P10 work/il_S2_summaryaalcalc/P10.bin work/il_S2_summaryleccalc/P10.bin > /dev/null & pid99=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P10.idx work/il_S2_summaryaalcalc/P10.idx work/il_S2_summaryleccalc/P10.idx > /dev/null & pid100=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/il_P4 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/il_P9 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P10 < /tmp/%FIFO_DIR%/fifo/il_P10 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid101=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid102=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid103=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid104=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid105=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid106=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid107=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid108=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid109=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid110=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid111=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid112=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid113=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid114=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid115=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P3 > work/kat/gul_S2_eltcalc_P3 & pid116=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P3 > work/kat/gul_S2_summarycalc_P3 & pid117=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P3 > work/kat/gul_S2_pltcalc_P3 & pid118=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid119=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid120=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid121=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P4 > work/kat/gul_S2_eltcalc_P4 & pid122=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P4 > work/kat/gul_S2_summarycalc_P4 & pid123=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P4 > work/kat/gul_S2_pltcalc_P4 & pid124=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid125=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid126=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid127=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P5 > work/kat/gul_S2_eltcalc_P5 & pid128=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P5 > work/kat/gul_S2_summarycalc_P5 & pid129=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P5 > work/kat/gul_S2_pltcalc_P5 & pid130=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid131=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid132=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid133=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P6 > work/kat/gul_S2_eltcalc_P6 & pid134=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P6 > work/kat/gul_S2_summarycalc_P6 & pid135=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P6 > work/kat/gul_S2_pltcalc_P6 & pid136=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid137=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid138=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid139=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P7 > work/kat/gul_S2_eltcalc_P7 & pid140=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P7 > work/kat/gul_S2_summarycalc_P7 & pid141=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P7 > work/kat/gul_S2_pltcalc_P7 & pid142=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid143=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid144=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid145=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P8 > work/kat/gul_S2_eltcalc_P8 & pid146=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P8 > work/kat/gul_S2_summarycalc_P8 & pid147=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P8 > work/kat/gul_S2_pltcalc_P8 & pid148=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid149=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid150=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid151=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P9 > work/kat/gul_S2_eltcalc_P9 & pid152=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P9 > work/kat/gul_S2_summarycalc_P9 & pid153=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P9 > work/kat/gul_S2_pltcalc_P9 & pid154=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid155=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid156=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid157=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P10 > work/kat/gul_S2_eltcalc_P10 & pid158=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P10 > work/kat/gul_S2_summarycalc_P10 & pid159=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P10 > work/kat/gul_S2_pltcalc_P10 & pid160=$!


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid161=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid162=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid163=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1.idx work/gul_S2_summaryaalcalc/P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid164=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid165=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid166=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid167=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2.idx work/gul_S2_summaryaalcalc/P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid168=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid169=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P3.idx work/gul_S1_summaryaalcalc/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid170=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P3 work/gul_S2_summaryaalcalc/P3.bin work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid171=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P3.idx work/gul_S2_summaryaalcalc/P3.idx work/gul_S2_summaryleccalc/P3.idx > /dev/null & pid172=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid173=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4.idx work/gul_S1_summaryaalcalc/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid174=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P4 work/gul_S2_summaryaalcalc/P4.bin work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid175=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P4.idx work/gul_S2_summaryaalcalc/P4.idx work/gul_S2_summaryleccalc/P4.idx > /dev/null & pid176=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid177=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P5.idx work/gul_S1_summaryaalcalc/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid178=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P5 work/gul_S2_summaryaalcalc/P5.bin work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid179=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P5.idx work/gul_S2_summaryaalcalc/P5.idx work/gul_S2_summaryleccalc/P5.idx > /dev/null & pid180=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid181=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P6.idx work/gul_S1_summaryaalcalc/P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid182=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P6 work/gul_S2_summaryaalcalc/P6.bin work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid183=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P6.idx work/gul_S2_summaryaalcalc/P6.idx work/gul_S2_summaryleccalc/P6.idx > /dev/null & pid184=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid185=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid186=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P7 work/gul_S2_summaryaalcalc/P7.bin work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid187=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx work/gul_S2_summaryaalcalc/P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid188=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid189=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P8.idx work/gul_S1_summaryaalcalc/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid190=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P8 work/gul_S2_summaryaalcalc/P8.bin work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid191=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P8.idx work/gul_S2_summaryaalcalc/P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid192=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid193=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9.idx work/gul_S1_summaryaalcalc/P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid194=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P9 work/gul_S2_summaryaalcalc/P9.bin work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid195=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9.idx work/gul_S2_summaryaalcalc/P9.idx work/gul_S2_summaryleccalc/P9.idx > /dev/null & pid196=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid197=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid198=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P10 work/gul_S2_summaryaalcalc/P10.bin work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid199=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P10.idx work/gul_S2_summaryaalcalc/P10.idx work/gul_S2_summaryleccalc/P10.idx > /dev/null & pid200=$!

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

# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid201=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid202=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid203=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P1 > work/full_correlation/kat/il_S2_eltcalc_P1 & pid204=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P1 > work/full_correlation/kat/il_S2_summarycalc_P1 & pid205=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P1 > work/full_correlation/kat/il_S2_pltcalc_P1 & pid206=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid207=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid208=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid209=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P2 > work/full_correlation/kat/il_S2_eltcalc_P2 & pid210=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P2 > work/full_correlation/kat/il_S2_summarycalc_P2 & pid211=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P2 > work/full_correlation/kat/il_S2_pltcalc_P2 & pid212=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid213=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P3 > work/full_correlation/kat/il_S1_summarycalc_P3 & pid214=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P3 > work/full_correlation/kat/il_S1_pltcalc_P3 & pid215=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P3 > work/full_correlation/kat/il_S2_eltcalc_P3 & pid216=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P3 > work/full_correlation/kat/il_S2_summarycalc_P3 & pid217=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P3 > work/full_correlation/kat/il_S2_pltcalc_P3 & pid218=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid219=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P4 > work/full_correlation/kat/il_S1_summarycalc_P4 & pid220=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 & pid221=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P4 > work/full_correlation/kat/il_S2_eltcalc_P4 & pid222=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P4 > work/full_correlation/kat/il_S2_summarycalc_P4 & pid223=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P4 > work/full_correlation/kat/il_S2_pltcalc_P4 & pid224=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 & pid225=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5 > work/full_correlation/kat/il_S1_summarycalc_P5 & pid226=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P5 > work/full_correlation/kat/il_S1_pltcalc_P5 & pid227=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P5 > work/full_correlation/kat/il_S2_eltcalc_P5 & pid228=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P5 > work/full_correlation/kat/il_S2_summarycalc_P5 & pid229=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P5 > work/full_correlation/kat/il_S2_pltcalc_P5 & pid230=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 & pid231=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P6 > work/full_correlation/kat/il_S1_summarycalc_P6 & pid232=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 & pid233=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P6 > work/full_correlation/kat/il_S2_eltcalc_P6 & pid234=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P6 > work/full_correlation/kat/il_S2_summarycalc_P6 & pid235=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P6 > work/full_correlation/kat/il_S2_pltcalc_P6 & pid236=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid237=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7 > work/full_correlation/kat/il_S1_summarycalc_P7 & pid238=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 & pid239=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P7 > work/full_correlation/kat/il_S2_eltcalc_P7 & pid240=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P7 > work/full_correlation/kat/il_S2_summarycalc_P7 & pid241=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P7 > work/full_correlation/kat/il_S2_pltcalc_P7 & pid242=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 & pid243=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P8 > work/full_correlation/kat/il_S1_summarycalc_P8 & pid244=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P8 > work/full_correlation/kat/il_S1_pltcalc_P8 & pid245=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P8 > work/full_correlation/kat/il_S2_eltcalc_P8 & pid246=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P8 > work/full_correlation/kat/il_S2_summarycalc_P8 & pid247=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P8 > work/full_correlation/kat/il_S2_pltcalc_P8 & pid248=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 & pid249=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P9 > work/full_correlation/kat/il_S1_summarycalc_P9 & pid250=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P9 > work/full_correlation/kat/il_S1_pltcalc_P9 & pid251=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P9 > work/full_correlation/kat/il_S2_eltcalc_P9 & pid252=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P9 > work/full_correlation/kat/il_S2_summarycalc_P9 & pid253=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P9 > work/full_correlation/kat/il_S2_pltcalc_P9 & pid254=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 & pid255=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P10 > work/full_correlation/kat/il_S1_summarycalc_P10 & pid256=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P10 > work/full_correlation/kat/il_S1_pltcalc_P10 & pid257=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P10 > work/full_correlation/kat/il_S2_eltcalc_P10 & pid258=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P10 > work/full_correlation/kat/il_S2_summarycalc_P10 & pid259=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P10 > work/full_correlation/kat/il_S2_pltcalc_P10 & pid260=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid261=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx work/full_correlation/il_S1_summaryaalcalc/P1.idx work/full_correlation/il_S1_summaryleccalc/P1.idx > /dev/null & pid262=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P1 work/full_correlation/il_S2_summaryaalcalc/P1.bin work/full_correlation/il_S2_summaryleccalc/P1.bin > /dev/null & pid263=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P1.idx work/full_correlation/il_S2_summaryaalcalc/P1.idx work/full_correlation/il_S2_summaryleccalc/P1.idx > /dev/null & pid264=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid265=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2.idx work/full_correlation/il_S1_summaryaalcalc/P2.idx work/full_correlation/il_S1_summaryleccalc/P2.idx > /dev/null & pid266=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P2 work/full_correlation/il_S2_summaryaalcalc/P2.bin work/full_correlation/il_S2_summaryleccalc/P2.bin > /dev/null & pid267=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P2.idx work/full_correlation/il_S2_summaryaalcalc/P2.idx work/full_correlation/il_S2_summaryleccalc/P2.idx > /dev/null & pid268=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P3 work/full_correlation/il_S1_summaryaalcalc/P3.bin work/full_correlation/il_S1_summaryleccalc/P3.bin > /dev/null & pid269=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3.idx work/full_correlation/il_S1_summaryaalcalc/P3.idx work/full_correlation/il_S1_summaryleccalc/P3.idx > /dev/null & pid270=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P3 work/full_correlation/il_S2_summaryaalcalc/P3.bin work/full_correlation/il_S2_summaryleccalc/P3.bin > /dev/null & pid271=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P3.idx work/full_correlation/il_S2_summaryaalcalc/P3.idx work/full_correlation/il_S2_summaryleccalc/P3.idx > /dev/null & pid272=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P4 work/full_correlation/il_S1_summaryaalcalc/P4.bin work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid273=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4.idx work/full_correlation/il_S1_summaryaalcalc/P4.idx work/full_correlation/il_S1_summaryleccalc/P4.idx > /dev/null & pid274=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P4 work/full_correlation/il_S2_summaryaalcalc/P4.bin work/full_correlation/il_S2_summaryleccalc/P4.bin > /dev/null & pid275=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P4.idx work/full_correlation/il_S2_summaryaalcalc/P4.idx work/full_correlation/il_S2_summaryleccalc/P4.idx > /dev/null & pid276=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P5 work/full_correlation/il_S1_summaryaalcalc/P5.bin work/full_correlation/il_S1_summaryleccalc/P5.bin > /dev/null & pid277=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5.idx work/full_correlation/il_S1_summaryaalcalc/P5.idx work/full_correlation/il_S1_summaryleccalc/P5.idx > /dev/null & pid278=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P5 work/full_correlation/il_S2_summaryaalcalc/P5.bin work/full_correlation/il_S2_summaryleccalc/P5.bin > /dev/null & pid279=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P5.idx work/full_correlation/il_S2_summaryaalcalc/P5.idx work/full_correlation/il_S2_summaryleccalc/P5.idx > /dev/null & pid280=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P6 work/full_correlation/il_S1_summaryaalcalc/P6.bin work/full_correlation/il_S1_summaryleccalc/P6.bin > /dev/null & pid281=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx work/full_correlation/il_S1_summaryaalcalc/P6.idx work/full_correlation/il_S1_summaryleccalc/P6.idx > /dev/null & pid282=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P6 work/full_correlation/il_S2_summaryaalcalc/P6.bin work/full_correlation/il_S2_summaryleccalc/P6.bin > /dev/null & pid283=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P6.idx work/full_correlation/il_S2_summaryaalcalc/P6.idx work/full_correlation/il_S2_summaryleccalc/P6.idx > /dev/null & pid284=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7 work/full_correlation/il_S1_summaryaalcalc/P7.bin work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid285=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7.idx work/full_correlation/il_S1_summaryaalcalc/P7.idx work/full_correlation/il_S1_summaryleccalc/P7.idx > /dev/null & pid286=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P7 work/full_correlation/il_S2_summaryaalcalc/P7.bin work/full_correlation/il_S2_summaryleccalc/P7.bin > /dev/null & pid287=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7.idx work/full_correlation/il_S2_summaryaalcalc/P7.idx work/full_correlation/il_S2_summaryleccalc/P7.idx > /dev/null & pid288=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P8 work/full_correlation/il_S1_summaryaalcalc/P8.bin work/full_correlation/il_S1_summaryleccalc/P8.bin > /dev/null & pid289=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8.idx work/full_correlation/il_S1_summaryaalcalc/P8.idx work/full_correlation/il_S1_summaryleccalc/P8.idx > /dev/null & pid290=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P8 work/full_correlation/il_S2_summaryaalcalc/P8.bin work/full_correlation/il_S2_summaryleccalc/P8.bin > /dev/null & pid291=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P8.idx work/full_correlation/il_S2_summaryaalcalc/P8.idx work/full_correlation/il_S2_summaryleccalc/P8.idx > /dev/null & pid292=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P9 work/full_correlation/il_S1_summaryaalcalc/P9.bin work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid293=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9.idx work/full_correlation/il_S1_summaryaalcalc/P9.idx work/full_correlation/il_S1_summaryleccalc/P9.idx > /dev/null & pid294=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P9 work/full_correlation/il_S2_summaryaalcalc/P9.bin work/full_correlation/il_S2_summaryleccalc/P9.bin > /dev/null & pid295=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P9.idx work/full_correlation/il_S2_summaryaalcalc/P9.idx work/full_correlation/il_S2_summaryleccalc/P9.idx > /dev/null & pid296=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P10 work/full_correlation/il_S1_summaryaalcalc/P10.bin work/full_correlation/il_S1_summaryleccalc/P10.bin > /dev/null & pid297=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10.idx work/full_correlation/il_S1_summaryaalcalc/P10.idx work/full_correlation/il_S1_summaryleccalc/P10.idx > /dev/null & pid298=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P10 work/full_correlation/il_S2_summaryaalcalc/P10.bin work/full_correlation/il_S2_summaryleccalc/P10.bin > /dev/null & pid299=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P10.idx work/full_correlation/il_S2_summaryaalcalc/P10.idx work/full_correlation/il_S2_summaryleccalc/P10.idx > /dev/null & pid300=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P3 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P3 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P4 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P8 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P8 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9 &
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P10 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P10 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid301=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid302=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid303=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1 > work/full_correlation/kat/gul_S2_eltcalc_P1 & pid304=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1 > work/full_correlation/kat/gul_S2_summarycalc_P1 & pid305=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1 > work/full_correlation/kat/gul_S2_pltcalc_P1 & pid306=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid307=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid308=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid309=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2 > work/full_correlation/kat/gul_S2_eltcalc_P2 & pid310=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2 > work/full_correlation/kat/gul_S2_summarycalc_P2 & pid311=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2 > work/full_correlation/kat/gul_S2_pltcalc_P2 & pid312=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P3 > work/full_correlation/kat/gul_S1_eltcalc_P3 & pid313=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P3 > work/full_correlation/kat/gul_S1_summarycalc_P3 & pid314=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P3 > work/full_correlation/kat/gul_S1_pltcalc_P3 & pid315=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P3 > work/full_correlation/kat/gul_S2_eltcalc_P3 & pid316=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P3 > work/full_correlation/kat/gul_S2_summarycalc_P3 & pid317=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P3 > work/full_correlation/kat/gul_S2_pltcalc_P3 & pid318=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid319=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 & pid320=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 & pid321=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P4 > work/full_correlation/kat/gul_S2_eltcalc_P4 & pid322=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P4 > work/full_correlation/kat/gul_S2_summarycalc_P4 & pid323=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P4 > work/full_correlation/kat/gul_S2_pltcalc_P4 & pid324=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P5 > work/full_correlation/kat/gul_S1_eltcalc_P5 & pid325=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P5 > work/full_correlation/kat/gul_S1_summarycalc_P5 & pid326=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P5 > work/full_correlation/kat/gul_S1_pltcalc_P5 & pid327=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P5 > work/full_correlation/kat/gul_S2_eltcalc_P5 & pid328=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P5 > work/full_correlation/kat/gul_S2_summarycalc_P5 & pid329=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P5 > work/full_correlation/kat/gul_S2_pltcalc_P5 & pid330=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P6 > work/full_correlation/kat/gul_S1_eltcalc_P6 & pid331=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P6 > work/full_correlation/kat/gul_S1_summarycalc_P6 & pid332=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P6 > work/full_correlation/kat/gul_S1_pltcalc_P6 & pid333=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P6 > work/full_correlation/kat/gul_S2_eltcalc_P6 & pid334=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P6 > work/full_correlation/kat/gul_S2_summarycalc_P6 & pid335=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P6 > work/full_correlation/kat/gul_S2_pltcalc_P6 & pid336=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 & pid337=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 & pid338=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 & pid339=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P7 > work/full_correlation/kat/gul_S2_eltcalc_P7 & pid340=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P7 > work/full_correlation/kat/gul_S2_summarycalc_P7 & pid341=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P7 > work/full_correlation/kat/gul_S2_pltcalc_P7 & pid342=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P8 > work/full_correlation/kat/gul_S1_eltcalc_P8 & pid343=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P8 > work/full_correlation/kat/gul_S1_summarycalc_P8 & pid344=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P8 > work/full_correlation/kat/gul_S1_pltcalc_P8 & pid345=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P8 > work/full_correlation/kat/gul_S2_eltcalc_P8 & pid346=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P8 > work/full_correlation/kat/gul_S2_summarycalc_P8 & pid347=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P8 > work/full_correlation/kat/gul_S2_pltcalc_P8 & pid348=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 & pid349=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P9 > work/full_correlation/kat/gul_S1_summarycalc_P9 & pid350=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P9 > work/full_correlation/kat/gul_S1_pltcalc_P9 & pid351=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P9 > work/full_correlation/kat/gul_S2_eltcalc_P9 & pid352=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P9 > work/full_correlation/kat/gul_S2_summarycalc_P9 & pid353=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P9 > work/full_correlation/kat/gul_S2_pltcalc_P9 & pid354=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P10 > work/full_correlation/kat/gul_S1_eltcalc_P10 & pid355=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 & pid356=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P10 > work/full_correlation/kat/gul_S1_pltcalc_P10 & pid357=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P10 > work/full_correlation/kat/gul_S2_eltcalc_P10 & pid358=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P10 > work/full_correlation/kat/gul_S2_summarycalc_P10 & pid359=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P10 > work/full_correlation/kat/gul_S2_pltcalc_P10 & pid360=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid361=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryaalcalc/P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid362=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1 work/full_correlation/gul_S2_summaryaalcalc/P1.bin work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid363=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1.idx work/full_correlation/gul_S2_summaryaalcalc/P1.idx work/full_correlation/gul_S2_summaryleccalc/P1.idx > /dev/null & pid364=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid365=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryaalcalc/P2.idx work/full_correlation/gul_S1_summaryleccalc/P2.idx > /dev/null & pid366=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2 work/full_correlation/gul_S2_summaryaalcalc/P2.bin work/full_correlation/gul_S2_summaryleccalc/P2.bin > /dev/null & pid367=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2.idx work/full_correlation/gul_S2_summaryaalcalc/P2.idx work/full_correlation/gul_S2_summaryleccalc/P2.idx > /dev/null & pid368=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid369=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P3.idx work/full_correlation/gul_S1_summaryaalcalc/P3.idx work/full_correlation/gul_S1_summaryleccalc/P3.idx > /dev/null & pid370=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P3 work/full_correlation/gul_S2_summaryaalcalc/P3.bin work/full_correlation/gul_S2_summaryleccalc/P3.bin > /dev/null & pid371=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P3.idx work/full_correlation/gul_S2_summaryaalcalc/P3.idx work/full_correlation/gul_S2_summaryleccalc/P3.idx > /dev/null & pid372=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid373=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4.idx work/full_correlation/gul_S1_summaryaalcalc/P4.idx work/full_correlation/gul_S1_summaryleccalc/P4.idx > /dev/null & pid374=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P4 work/full_correlation/gul_S2_summaryaalcalc/P4.bin work/full_correlation/gul_S2_summaryleccalc/P4.bin > /dev/null & pid375=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P4.idx work/full_correlation/gul_S2_summaryaalcalc/P4.idx work/full_correlation/gul_S2_summaryleccalc/P4.idx > /dev/null & pid376=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid377=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P5.idx work/full_correlation/gul_S1_summaryaalcalc/P5.idx work/full_correlation/gul_S1_summaryleccalc/P5.idx > /dev/null & pid378=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P5 work/full_correlation/gul_S2_summaryaalcalc/P5.bin work/full_correlation/gul_S2_summaryleccalc/P5.bin > /dev/null & pid379=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P5.idx work/full_correlation/gul_S2_summaryaalcalc/P5.idx work/full_correlation/gul_S2_summaryleccalc/P5.idx > /dev/null & pid380=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid381=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P6.idx work/full_correlation/gul_S1_summaryaalcalc/P6.idx work/full_correlation/gul_S1_summaryleccalc/P6.idx > /dev/null & pid382=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P6 work/full_correlation/gul_S2_summaryaalcalc/P6.bin work/full_correlation/gul_S2_summaryleccalc/P6.bin > /dev/null & pid383=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P6.idx work/full_correlation/gul_S2_summaryaalcalc/P6.idx work/full_correlation/gul_S2_summaryleccalc/P6.idx > /dev/null & pid384=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid385=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryaalcalc/P7.idx work/full_correlation/gul_S1_summaryleccalc/P7.idx > /dev/null & pid386=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P7 work/full_correlation/gul_S2_summaryaalcalc/P7.bin work/full_correlation/gul_S2_summaryleccalc/P7.bin > /dev/null & pid387=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7.idx work/full_correlation/gul_S2_summaryaalcalc/P7.idx work/full_correlation/gul_S2_summaryleccalc/P7.idx > /dev/null & pid388=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid389=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P8.idx work/full_correlation/gul_S1_summaryaalcalc/P8.idx work/full_correlation/gul_S1_summaryleccalc/P8.idx > /dev/null & pid390=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P8 work/full_correlation/gul_S2_summaryaalcalc/P8.bin work/full_correlation/gul_S2_summaryleccalc/P8.bin > /dev/null & pid391=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P8.idx work/full_correlation/gul_S2_summaryaalcalc/P8.idx work/full_correlation/gul_S2_summaryleccalc/P8.idx > /dev/null & pid392=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid393=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P9.idx work/full_correlation/gul_S1_summaryaalcalc/P9.idx work/full_correlation/gul_S1_summaryleccalc/P9.idx > /dev/null & pid394=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P9 work/full_correlation/gul_S2_summaryaalcalc/P9.bin work/full_correlation/gul_S2_summaryleccalc/P9.bin > /dev/null & pid395=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P9.idx work/full_correlation/gul_S2_summaryaalcalc/P9.idx work/full_correlation/gul_S2_summaryleccalc/P9.idx > /dev/null & pid396=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid397=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P10.idx work/full_correlation/gul_S1_summaryaalcalc/P10.idx work/full_correlation/gul_S1_summaryleccalc/P10.idx > /dev/null & pid398=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P10 work/full_correlation/gul_S2_summaryaalcalc/P10.bin work/full_correlation/gul_S2_summaryleccalc/P10.bin > /dev/null & pid399=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P10.idx work/full_correlation/gul_S2_summaryaalcalc/P10.idx work/full_correlation/gul_S2_summaryleccalc/P10.idx > /dev/null & pid400=$!

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

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P3  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P3  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P4  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P5  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P6  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P8  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P8  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P9  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P9  &
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P10  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P10  &
eve 1 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P2 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P3 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P3 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P4 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P4 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P5 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P6 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P8 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P8 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P9 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P9 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P10 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P10 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210 $pid211 $pid212 $pid213 $pid214 $pid215 $pid216 $pid217 $pid218 $pid219 $pid220 $pid221 $pid222 $pid223 $pid224 $pid225 $pid226 $pid227 $pid228 $pid229 $pid230 $pid231 $pid232 $pid233 $pid234 $pid235 $pid236 $pid237 $pid238 $pid239 $pid240 $pid241 $pid242 $pid243 $pid244 $pid245 $pid246 $pid247 $pid248 $pid249 $pid250 $pid251 $pid252 $pid253 $pid254 $pid255 $pid256 $pid257 $pid258 $pid259 $pid260 $pid261 $pid262 $pid263 $pid264 $pid265 $pid266 $pid267 $pid268 $pid269 $pid270 $pid271 $pid272 $pid273 $pid274 $pid275 $pid276 $pid277 $pid278 $pid279 $pid280 $pid281 $pid282 $pid283 $pid284 $pid285 $pid286 $pid287 $pid288 $pid289 $pid290 $pid291 $pid292 $pid293 $pid294 $pid295 $pid296 $pid297 $pid298 $pid299 $pid300 $pid301 $pid302 $pid303 $pid304 $pid305 $pid306 $pid307 $pid308 $pid309 $pid310 $pid311 $pid312 $pid313 $pid314 $pid315 $pid316 $pid317 $pid318 $pid319 $pid320 $pid321 $pid322 $pid323 $pid324 $pid325 $pid326 $pid327 $pid328 $pid329 $pid330 $pid331 $pid332 $pid333 $pid334 $pid335 $pid336 $pid337 $pid338 $pid339 $pid340 $pid341 $pid342 $pid343 $pid344 $pid345 $pid346 $pid347 $pid348 $pid349 $pid350 $pid351 $pid352 $pid353 $pid354 $pid355 $pid356 $pid357 $pid358 $pid359 $pid360 $pid361 $pid362 $pid363 $pid364 $pid365 $pid366 $pid367 $pid368 $pid369 $pid370 $pid371 $pid372 $pid373 $pid374 $pid375 $pid376 $pid377 $pid378 $pid379 $pid380 $pid381 $pid382 $pid383 $pid384 $pid385 $pid386 $pid387 $pid388 $pid389 $pid390 $pid391 $pid392 $pid393 $pid394 $pid395 $pid396 $pid397 $pid398 $pid399 $pid400


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 > output/il_S1_summarycalc.csv & kpid3=$!
kat -s work/kat/il_S2_eltcalc_P1 work/kat/il_S2_eltcalc_P2 work/kat/il_S2_eltcalc_P3 work/kat/il_S2_eltcalc_P4 work/kat/il_S2_eltcalc_P5 work/kat/il_S2_eltcalc_P6 work/kat/il_S2_eltcalc_P7 work/kat/il_S2_eltcalc_P8 work/kat/il_S2_eltcalc_P9 work/kat/il_S2_eltcalc_P10 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P1 work/kat/il_S2_pltcalc_P2 work/kat/il_S2_pltcalc_P3 work/kat/il_S2_pltcalc_P4 work/kat/il_S2_pltcalc_P5 work/kat/il_S2_pltcalc_P6 work/kat/il_S2_pltcalc_P7 work/kat/il_S2_pltcalc_P8 work/kat/il_S2_pltcalc_P9 work/kat/il_S2_pltcalc_P10 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P1 work/kat/il_S2_summarycalc_P2 work/kat/il_S2_summarycalc_P3 work/kat/il_S2_summarycalc_P4 work/kat/il_S2_summarycalc_P5 work/kat/il_S2_summarycalc_P6 work/kat/il_S2_summarycalc_P7 work/kat/il_S2_summarycalc_P8 work/kat/il_S2_summarycalc_P9 work/kat/il_S2_summarycalc_P10 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 > output/full_correlation/il_S1_eltcalc.csv & kpid7=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 work/full_correlation/kat/il_S1_pltcalc_P3 work/full_correlation/kat/il_S1_pltcalc_P4 work/full_correlation/kat/il_S1_pltcalc_P5 work/full_correlation/kat/il_S1_pltcalc_P6 work/full_correlation/kat/il_S1_pltcalc_P7 work/full_correlation/kat/il_S1_pltcalc_P8 work/full_correlation/kat/il_S1_pltcalc_P9 work/full_correlation/kat/il_S1_pltcalc_P10 > output/full_correlation/il_S1_pltcalc.csv & kpid8=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 work/full_correlation/kat/il_S1_summarycalc_P2 work/full_correlation/kat/il_S1_summarycalc_P3 work/full_correlation/kat/il_S1_summarycalc_P4 work/full_correlation/kat/il_S1_summarycalc_P5 work/full_correlation/kat/il_S1_summarycalc_P6 work/full_correlation/kat/il_S1_summarycalc_P7 work/full_correlation/kat/il_S1_summarycalc_P8 work/full_correlation/kat/il_S1_summarycalc_P9 work/full_correlation/kat/il_S1_summarycalc_P10 > output/full_correlation/il_S1_summarycalc.csv & kpid9=$!
kat -s work/full_correlation/kat/il_S2_eltcalc_P1 work/full_correlation/kat/il_S2_eltcalc_P2 work/full_correlation/kat/il_S2_eltcalc_P3 work/full_correlation/kat/il_S2_eltcalc_P4 work/full_correlation/kat/il_S2_eltcalc_P5 work/full_correlation/kat/il_S2_eltcalc_P6 work/full_correlation/kat/il_S2_eltcalc_P7 work/full_correlation/kat/il_S2_eltcalc_P8 work/full_correlation/kat/il_S2_eltcalc_P9 work/full_correlation/kat/il_S2_eltcalc_P10 > output/full_correlation/il_S2_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/il_S2_pltcalc_P1 work/full_correlation/kat/il_S2_pltcalc_P2 work/full_correlation/kat/il_S2_pltcalc_P3 work/full_correlation/kat/il_S2_pltcalc_P4 work/full_correlation/kat/il_S2_pltcalc_P5 work/full_correlation/kat/il_S2_pltcalc_P6 work/full_correlation/kat/il_S2_pltcalc_P7 work/full_correlation/kat/il_S2_pltcalc_P8 work/full_correlation/kat/il_S2_pltcalc_P9 work/full_correlation/kat/il_S2_pltcalc_P10 > output/full_correlation/il_S2_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/il_S2_summarycalc_P1 work/full_correlation/kat/il_S2_summarycalc_P2 work/full_correlation/kat/il_S2_summarycalc_P3 work/full_correlation/kat/il_S2_summarycalc_P4 work/full_correlation/kat/il_S2_summarycalc_P5 work/full_correlation/kat/il_S2_summarycalc_P6 work/full_correlation/kat/il_S2_summarycalc_P7 work/full_correlation/kat/il_S2_summarycalc_P8 work/full_correlation/kat/il_S2_summarycalc_P9 work/full_correlation/kat/il_S2_summarycalc_P10 > output/full_correlation/il_S2_summarycalc.csv & kpid12=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 > output/gul_S1_eltcalc.csv & kpid13=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 > output/gul_S1_pltcalc.csv & kpid14=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 > output/gul_S1_summarycalc.csv & kpid15=$!
kat -s work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 work/kat/gul_S2_eltcalc_P3 work/kat/gul_S2_eltcalc_P4 work/kat/gul_S2_eltcalc_P5 work/kat/gul_S2_eltcalc_P6 work/kat/gul_S2_eltcalc_P7 work/kat/gul_S2_eltcalc_P8 work/kat/gul_S2_eltcalc_P9 work/kat/gul_S2_eltcalc_P10 > output/gul_S2_eltcalc.csv & kpid16=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 work/kat/gul_S2_pltcalc_P3 work/kat/gul_S2_pltcalc_P4 work/kat/gul_S2_pltcalc_P5 work/kat/gul_S2_pltcalc_P6 work/kat/gul_S2_pltcalc_P7 work/kat/gul_S2_pltcalc_P8 work/kat/gul_S2_pltcalc_P9 work/kat/gul_S2_pltcalc_P10 > output/gul_S2_pltcalc.csv & kpid17=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 work/kat/gul_S2_summarycalc_P3 work/kat/gul_S2_summarycalc_P4 work/kat/gul_S2_summarycalc_P5 work/kat/gul_S2_summarycalc_P6 work/kat/gul_S2_summarycalc_P7 work/kat/gul_S2_summarycalc_P8 work/kat/gul_S2_summarycalc_P9 work/kat/gul_S2_summarycalc_P10 > output/gul_S2_summarycalc.csv & kpid18=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 work/full_correlation/kat/gul_S1_eltcalc_P3 work/full_correlation/kat/gul_S1_eltcalc_P4 work/full_correlation/kat/gul_S1_eltcalc_P5 work/full_correlation/kat/gul_S1_eltcalc_P6 work/full_correlation/kat/gul_S1_eltcalc_P7 work/full_correlation/kat/gul_S1_eltcalc_P8 work/full_correlation/kat/gul_S1_eltcalc_P9 work/full_correlation/kat/gul_S1_eltcalc_P10 > output/full_correlation/gul_S1_eltcalc.csv & kpid19=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 work/full_correlation/kat/gul_S1_pltcalc_P3 work/full_correlation/kat/gul_S1_pltcalc_P4 work/full_correlation/kat/gul_S1_pltcalc_P5 work/full_correlation/kat/gul_S1_pltcalc_P6 work/full_correlation/kat/gul_S1_pltcalc_P7 work/full_correlation/kat/gul_S1_pltcalc_P8 work/full_correlation/kat/gul_S1_pltcalc_P9 work/full_correlation/kat/gul_S1_pltcalc_P10 > output/full_correlation/gul_S1_pltcalc.csv & kpid20=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 work/full_correlation/kat/gul_S1_summarycalc_P3 work/full_correlation/kat/gul_S1_summarycalc_P4 work/full_correlation/kat/gul_S1_summarycalc_P5 work/full_correlation/kat/gul_S1_summarycalc_P6 work/full_correlation/kat/gul_S1_summarycalc_P7 work/full_correlation/kat/gul_S1_summarycalc_P8 work/full_correlation/kat/gul_S1_summarycalc_P9 work/full_correlation/kat/gul_S1_summarycalc_P10 > output/full_correlation/gul_S1_summarycalc.csv & kpid21=$!
kat -s work/full_correlation/kat/gul_S2_eltcalc_P1 work/full_correlation/kat/gul_S2_eltcalc_P2 work/full_correlation/kat/gul_S2_eltcalc_P3 work/full_correlation/kat/gul_S2_eltcalc_P4 work/full_correlation/kat/gul_S2_eltcalc_P5 work/full_correlation/kat/gul_S2_eltcalc_P6 work/full_correlation/kat/gul_S2_eltcalc_P7 work/full_correlation/kat/gul_S2_eltcalc_P8 work/full_correlation/kat/gul_S2_eltcalc_P9 work/full_correlation/kat/gul_S2_eltcalc_P10 > output/full_correlation/gul_S2_eltcalc.csv & kpid22=$!
kat work/full_correlation/kat/gul_S2_pltcalc_P1 work/full_correlation/kat/gul_S2_pltcalc_P2 work/full_correlation/kat/gul_S2_pltcalc_P3 work/full_correlation/kat/gul_S2_pltcalc_P4 work/full_correlation/kat/gul_S2_pltcalc_P5 work/full_correlation/kat/gul_S2_pltcalc_P6 work/full_correlation/kat/gul_S2_pltcalc_P7 work/full_correlation/kat/gul_S2_pltcalc_P8 work/full_correlation/kat/gul_S2_pltcalc_P9 work/full_correlation/kat/gul_S2_pltcalc_P10 > output/full_correlation/gul_S2_pltcalc.csv & kpid23=$!
kat work/full_correlation/kat/gul_S2_summarycalc_P1 work/full_correlation/kat/gul_S2_summarycalc_P2 work/full_correlation/kat/gul_S2_summarycalc_P3 work/full_correlation/kat/gul_S2_summarycalc_P4 work/full_correlation/kat/gul_S2_summarycalc_P5 work/full_correlation/kat/gul_S2_summarycalc_P6 work/full_correlation/kat/gul_S2_summarycalc_P7 work/full_correlation/kat/gul_S2_summarycalc_P8 work/full_correlation/kat/gul_S2_summarycalc_P9 work/full_correlation/kat/gul_S2_summarycalc_P10 > output/full_correlation/gul_S2_summarycalc.csv & kpid24=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18 $kpid19 $kpid20 $kpid21 $kpid22 $kpid23 $kpid24


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
ordleccalc -r -Kil_S1_summaryleccalc -F -S -s -M -m -W -w -O output/il_S1_ept.csv -o output/il_S1_psept.csv & lpid2=$!
aalcalc -Kil_S2_summaryaalcalc > output/il_S2_aalcalc.csv & lpid3=$!
ordleccalc -r -Kil_S2_summaryleccalc -F -S -s -M -m -W -w -O output/il_S2_ept.csv -o output/il_S2_psept.csv & lpid4=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid5=$!
ordleccalc -r -Kgul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S1_ept.csv -o output/gul_S1_psept.csv & lpid6=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid7=$!
ordleccalc -r -Kgul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S2_ept.csv -o output/gul_S2_psept.csv & lpid8=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid9=$!
ordleccalc -r -Kfull_correlation/il_S1_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/il_S1_ept.csv -o output/full_correlation/il_S1_psept.csv & lpid10=$!
aalcalc -Kfull_correlation/il_S2_summaryaalcalc > output/full_correlation/il_S2_aalcalc.csv & lpid11=$!
ordleccalc -r -Kfull_correlation/il_S2_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/il_S2_ept.csv -o output/full_correlation/il_S2_psept.csv & lpid12=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid13=$!
ordleccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/gul_S1_ept.csv -o output/full_correlation/gul_S1_psept.csv & lpid14=$!
aalcalc -Kfull_correlation/gul_S2_summaryaalcalc > output/full_correlation/gul_S2_aalcalc.csv & lpid15=$!
ordleccalc -r -Kfull_correlation/gul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/gul_S2_ept.csv -o output/full_correlation/gul_S2_psept.csv & lpid16=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8 $lpid9 $lpid10 $lpid11 $lpid12 $lpid13 $lpid14 $lpid15 $lpid16

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
