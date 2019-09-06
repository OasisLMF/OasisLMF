#!/bin/bash

set -e
set -o pipefail

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summaryeltcalc_P1
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarysummarycalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_summarypltcalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_summaryeltcalc_P1
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarysummarycalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_summarypltcalc_P1
mkfifo fifo/gul_S2_pltcalc_P1

mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summaryeltcalc_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarysummarycalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_summarypltcalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summaryeltcalc_P2
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarysummarycalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_summarypltcalc_P2
mkfifo fifo/gul_S2_pltcalc_P2

mkfifo fifo/gul_P3

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summaryeltcalc_P3
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarysummarycalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_summarypltcalc_P3
mkfifo fifo/gul_S1_pltcalc_P3
mkfifo fifo/gul_S2_summary_P3
mkfifo fifo/gul_S2_summaryeltcalc_P3
mkfifo fifo/gul_S2_eltcalc_P3
mkfifo fifo/gul_S2_summarysummarycalc_P3
mkfifo fifo/gul_S2_summarycalc_P3
mkfifo fifo/gul_S2_summarypltcalc_P3
mkfifo fifo/gul_S2_pltcalc_P3

mkfifo fifo/gul_P4

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summaryeltcalc_P4
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarysummarycalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_summarypltcalc_P4
mkfifo fifo/gul_S1_pltcalc_P4
mkfifo fifo/gul_S2_summary_P4
mkfifo fifo/gul_S2_summaryeltcalc_P4
mkfifo fifo/gul_S2_eltcalc_P4
mkfifo fifo/gul_S2_summarysummarycalc_P4
mkfifo fifo/gul_S2_summarycalc_P4
mkfifo fifo/gul_S2_summarypltcalc_P4
mkfifo fifo/gul_S2_pltcalc_P4

mkfifo fifo/gul_P5

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summaryeltcalc_P5
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarysummarycalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_summarypltcalc_P5
mkfifo fifo/gul_S1_pltcalc_P5
mkfifo fifo/gul_S2_summary_P5
mkfifo fifo/gul_S2_summaryeltcalc_P5
mkfifo fifo/gul_S2_eltcalc_P5
mkfifo fifo/gul_S2_summarysummarycalc_P5
mkfifo fifo/gul_S2_summarycalc_P5
mkfifo fifo/gul_S2_summarypltcalc_P5
mkfifo fifo/gul_S2_pltcalc_P5

mkfifo fifo/gul_P6

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summaryeltcalc_P6
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarysummarycalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_summarypltcalc_P6
mkfifo fifo/gul_S1_pltcalc_P6
mkfifo fifo/gul_S2_summary_P6
mkfifo fifo/gul_S2_summaryeltcalc_P6
mkfifo fifo/gul_S2_eltcalc_P6
mkfifo fifo/gul_S2_summarysummarycalc_P6
mkfifo fifo/gul_S2_summarycalc_P6
mkfifo fifo/gul_S2_summarypltcalc_P6
mkfifo fifo/gul_S2_pltcalc_P6

mkfifo fifo/gul_P7

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summaryeltcalc_P7
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarysummarycalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_summarypltcalc_P7
mkfifo fifo/gul_S1_pltcalc_P7
mkfifo fifo/gul_S2_summary_P7
mkfifo fifo/gul_S2_summaryeltcalc_P7
mkfifo fifo/gul_S2_eltcalc_P7
mkfifo fifo/gul_S2_summarysummarycalc_P7
mkfifo fifo/gul_S2_summarycalc_P7
mkfifo fifo/gul_S2_summarypltcalc_P7
mkfifo fifo/gul_S2_pltcalc_P7

mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summaryeltcalc_P8
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarysummarycalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_summarypltcalc_P8
mkfifo fifo/gul_S1_pltcalc_P8
mkfifo fifo/gul_S2_summary_P8
mkfifo fifo/gul_S2_summaryeltcalc_P8
mkfifo fifo/gul_S2_eltcalc_P8
mkfifo fifo/gul_S2_summarysummarycalc_P8
mkfifo fifo/gul_S2_summarycalc_P8
mkfifo fifo/gul_S2_summarypltcalc_P8
mkfifo fifo/gul_S2_pltcalc_P8

mkfifo fifo/gul_P9

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summaryeltcalc_P9
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarysummarycalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_summarypltcalc_P9
mkfifo fifo/gul_S1_pltcalc_P9
mkfifo fifo/gul_S2_summary_P9
mkfifo fifo/gul_S2_summaryeltcalc_P9
mkfifo fifo/gul_S2_eltcalc_P9
mkfifo fifo/gul_S2_summarysummarycalc_P9
mkfifo fifo/gul_S2_summarycalc_P9
mkfifo fifo/gul_S2_summarypltcalc_P9
mkfifo fifo/gul_S2_pltcalc_P9

mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summaryeltcalc_P10
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarysummarycalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_summarypltcalc_P10
mkfifo fifo/gul_S1_pltcalc_P10
mkfifo fifo/gul_S2_summary_P10
mkfifo fifo/gul_S2_summaryeltcalc_P10
mkfifo fifo/gul_S2_eltcalc_P10
mkfifo fifo/gul_S2_summarysummarycalc_P10
mkfifo fifo/gul_S2_summarycalc_P10
mkfifo fifo/gul_S2_summarypltcalc_P10
mkfifo fifo/gul_S2_pltcalc_P10

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_summaryeltcalc_P1
mkfifo fifo/il_S2_eltcalc_P1
mkfifo fifo/il_S2_summarysummarycalc_P1
mkfifo fifo/il_S2_summarycalc_P1
mkfifo fifo/il_S2_summarypltcalc_P1
mkfifo fifo/il_S2_pltcalc_P1

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summaryeltcalc_P2
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarysummarycalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_summarypltcalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_summaryeltcalc_P2
mkfifo fifo/il_S2_eltcalc_P2
mkfifo fifo/il_S2_summarysummarycalc_P2
mkfifo fifo/il_S2_summarycalc_P2
mkfifo fifo/il_S2_summarypltcalc_P2
mkfifo fifo/il_S2_pltcalc_P2

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summaryeltcalc_P3
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarysummarycalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_summarypltcalc_P3
mkfifo fifo/il_S1_pltcalc_P3
mkfifo fifo/il_S2_summary_P3
mkfifo fifo/il_S2_summaryeltcalc_P3
mkfifo fifo/il_S2_eltcalc_P3
mkfifo fifo/il_S2_summarysummarycalc_P3
mkfifo fifo/il_S2_summarycalc_P3
mkfifo fifo/il_S2_summarypltcalc_P3
mkfifo fifo/il_S2_pltcalc_P3

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summaryeltcalc_P4
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarysummarycalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_summarypltcalc_P4
mkfifo fifo/il_S1_pltcalc_P4
mkfifo fifo/il_S2_summary_P4
mkfifo fifo/il_S2_summaryeltcalc_P4
mkfifo fifo/il_S2_eltcalc_P4
mkfifo fifo/il_S2_summarysummarycalc_P4
mkfifo fifo/il_S2_summarycalc_P4
mkfifo fifo/il_S2_summarypltcalc_P4
mkfifo fifo/il_S2_pltcalc_P4

mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summaryeltcalc_P5
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarysummarycalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_summarypltcalc_P5
mkfifo fifo/il_S1_pltcalc_P5
mkfifo fifo/il_S2_summary_P5
mkfifo fifo/il_S2_summaryeltcalc_P5
mkfifo fifo/il_S2_eltcalc_P5
mkfifo fifo/il_S2_summarysummarycalc_P5
mkfifo fifo/il_S2_summarycalc_P5
mkfifo fifo/il_S2_summarypltcalc_P5
mkfifo fifo/il_S2_pltcalc_P5

mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summaryeltcalc_P6
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarysummarycalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_summarypltcalc_P6
mkfifo fifo/il_S1_pltcalc_P6
mkfifo fifo/il_S2_summary_P6
mkfifo fifo/il_S2_summaryeltcalc_P6
mkfifo fifo/il_S2_eltcalc_P6
mkfifo fifo/il_S2_summarysummarycalc_P6
mkfifo fifo/il_S2_summarycalc_P6
mkfifo fifo/il_S2_summarypltcalc_P6
mkfifo fifo/il_S2_pltcalc_P6

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summaryeltcalc_P7
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarysummarycalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_summarypltcalc_P7
mkfifo fifo/il_S1_pltcalc_P7
mkfifo fifo/il_S2_summary_P7
mkfifo fifo/il_S2_summaryeltcalc_P7
mkfifo fifo/il_S2_eltcalc_P7
mkfifo fifo/il_S2_summarysummarycalc_P7
mkfifo fifo/il_S2_summarycalc_P7
mkfifo fifo/il_S2_summarypltcalc_P7
mkfifo fifo/il_S2_pltcalc_P7

mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summaryeltcalc_P8
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarysummarycalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_summarypltcalc_P8
mkfifo fifo/il_S1_pltcalc_P8
mkfifo fifo/il_S2_summary_P8
mkfifo fifo/il_S2_summaryeltcalc_P8
mkfifo fifo/il_S2_eltcalc_P8
mkfifo fifo/il_S2_summarysummarycalc_P8
mkfifo fifo/il_S2_summarycalc_P8
mkfifo fifo/il_S2_summarypltcalc_P8
mkfifo fifo/il_S2_pltcalc_P8

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summaryeltcalc_P9
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarysummarycalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_summarypltcalc_P9
mkfifo fifo/il_S1_pltcalc_P9
mkfifo fifo/il_S2_summary_P9
mkfifo fifo/il_S2_summaryeltcalc_P9
mkfifo fifo/il_S2_eltcalc_P9
mkfifo fifo/il_S2_summarysummarycalc_P9
mkfifo fifo/il_S2_summarycalc_P9
mkfifo fifo/il_S2_summarypltcalc_P9
mkfifo fifo/il_S2_pltcalc_P9

mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summaryeltcalc_P10
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarysummarycalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_summarypltcalc_P10
mkfifo fifo/il_S1_pltcalc_P10
mkfifo fifo/il_S2_summary_P10
mkfifo fifo/il_S2_summaryeltcalc_P10
mkfifo fifo/il_S2_eltcalc_P10
mkfifo fifo/il_S2_summarysummarycalc_P10
mkfifo fifo/il_S2_summarycalc_P10
mkfifo fifo/il_S2_summarypltcalc_P10
mkfifo fifo/il_S2_pltcalc_P10

mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_summaryaalcalc

# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!

eltcalc < fifo/il_S2_summaryeltcalc_P1 > work/kat/il_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < fifo/il_S2_summarysummarycalc_P1 > work/kat/il_S2_summarycalc_P1 & pid5=$!
pltcalc < fifo/il_S2_summarypltcalc_P1 > work/kat/il_S2_pltcalc_P1 & pid6=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid8=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid9=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P2 > work/kat/il_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P2 > work/kat/il_S2_summarycalc_P2 & pid11=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P2 > work/kat/il_S2_pltcalc_P2 & pid12=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid13=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid14=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid15=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P3 > work/kat/il_S2_eltcalc_P3 & pid16=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P3 > work/kat/il_S2_summarycalc_P3 & pid17=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P3 > work/kat/il_S2_pltcalc_P3 & pid18=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid19=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid20=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid21=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P4 > work/kat/il_S2_eltcalc_P4 & pid22=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P4 > work/kat/il_S2_summarycalc_P4 & pid23=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P4 > work/kat/il_S2_pltcalc_P4 & pid24=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid25=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid26=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid27=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P5 > work/kat/il_S2_eltcalc_P5 & pid28=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P5 > work/kat/il_S2_summarycalc_P5 & pid29=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P5 > work/kat/il_S2_pltcalc_P5 & pid30=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid31=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid32=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid33=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P6 > work/kat/il_S2_eltcalc_P6 & pid34=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P6 > work/kat/il_S2_summarycalc_P6 & pid35=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P6 > work/kat/il_S2_pltcalc_P6 & pid36=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid37=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid38=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid39=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P7 > work/kat/il_S2_eltcalc_P7 & pid40=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P7 > work/kat/il_S2_summarycalc_P7 & pid41=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P7 > work/kat/il_S2_pltcalc_P7 & pid42=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid43=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid44=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid45=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P8 > work/kat/il_S2_eltcalc_P8 & pid46=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P8 > work/kat/il_S2_summarycalc_P8 & pid47=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P8 > work/kat/il_S2_pltcalc_P8 & pid48=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid49=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid50=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid51=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P9 > work/kat/il_S2_eltcalc_P9 & pid52=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P9 > work/kat/il_S2_summarycalc_P9 & pid53=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P9 > work/kat/il_S2_pltcalc_P9 & pid54=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid55=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid56=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid57=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P10 > work/kat/il_S2_eltcalc_P10 & pid58=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P10 > work/kat/il_S2_summarycalc_P10 & pid59=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P10 > work/kat/il_S2_pltcalc_P10 & pid60=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 fifo/il_S1_summarypltcalc_P1 fifo/il_S1_summarysummarycalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_summaryeltcalc_P1 fifo/il_S2_summarypltcalc_P1 fifo/il_S2_summarysummarycalc_P1 work/il_S2_summaryaalcalc/P1.bin work/il_S2_summaryleccalc/P1.bin > /dev/null & pid62=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_summaryeltcalc_P2 fifo/il_S1_summarypltcalc_P2 fifo/il_S1_summarysummarycalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid63=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_summaryeltcalc_P2 fifo/il_S2_summarypltcalc_P2 fifo/il_S2_summarysummarycalc_P2 work/il_S2_summaryaalcalc/P2.bin work/il_S2_summaryleccalc/P2.bin > /dev/null & pid64=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summaryeltcalc_P3 fifo/il_S1_summarypltcalc_P3 fifo/il_S1_summarysummarycalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid65=$!
tee < fifo/il_S2_summary_P3 fifo/il_S2_summaryeltcalc_P3 fifo/il_S2_summarypltcalc_P3 fifo/il_S2_summarysummarycalc_P3 work/il_S2_summaryaalcalc/P3.bin work/il_S2_summaryleccalc/P3.bin > /dev/null & pid66=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summaryeltcalc_P4 fifo/il_S1_summarypltcalc_P4 fifo/il_S1_summarysummarycalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid67=$!
tee < fifo/il_S2_summary_P4 fifo/il_S2_summaryeltcalc_P4 fifo/il_S2_summarypltcalc_P4 fifo/il_S2_summarysummarycalc_P4 work/il_S2_summaryaalcalc/P4.bin work/il_S2_summaryleccalc/P4.bin > /dev/null & pid68=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summaryeltcalc_P5 fifo/il_S1_summarypltcalc_P5 fifo/il_S1_summarysummarycalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid69=$!
tee < fifo/il_S2_summary_P5 fifo/il_S2_summaryeltcalc_P5 fifo/il_S2_summarypltcalc_P5 fifo/il_S2_summarysummarycalc_P5 work/il_S2_summaryaalcalc/P5.bin work/il_S2_summaryleccalc/P5.bin > /dev/null & pid70=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_summaryeltcalc_P6 fifo/il_S1_summarypltcalc_P6 fifo/il_S1_summarysummarycalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid71=$!
tee < fifo/il_S2_summary_P6 fifo/il_S2_summaryeltcalc_P6 fifo/il_S2_summarypltcalc_P6 fifo/il_S2_summarysummarycalc_P6 work/il_S2_summaryaalcalc/P6.bin work/il_S2_summaryleccalc/P6.bin > /dev/null & pid72=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_summaryeltcalc_P7 fifo/il_S1_summarypltcalc_P7 fifo/il_S1_summarysummarycalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid73=$!
tee < fifo/il_S2_summary_P7 fifo/il_S2_summaryeltcalc_P7 fifo/il_S2_summarypltcalc_P7 fifo/il_S2_summarysummarycalc_P7 work/il_S2_summaryaalcalc/P7.bin work/il_S2_summaryleccalc/P7.bin > /dev/null & pid74=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_summaryeltcalc_P8 fifo/il_S1_summarypltcalc_P8 fifo/il_S1_summarysummarycalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid75=$!
tee < fifo/il_S2_summary_P8 fifo/il_S2_summaryeltcalc_P8 fifo/il_S2_summarypltcalc_P8 fifo/il_S2_summarysummarycalc_P8 work/il_S2_summaryaalcalc/P8.bin work/il_S2_summaryleccalc/P8.bin > /dev/null & pid76=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_summaryeltcalc_P9 fifo/il_S1_summarypltcalc_P9 fifo/il_S1_summarysummarycalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid77=$!
tee < fifo/il_S2_summary_P9 fifo/il_S2_summaryeltcalc_P9 fifo/il_S2_summarypltcalc_P9 fifo/il_S2_summarysummarycalc_P9 work/il_S2_summaryaalcalc/P9.bin work/il_S2_summaryleccalc/P9.bin > /dev/null & pid78=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_summaryeltcalc_P10 fifo/il_S1_summarypltcalc_P10 fifo/il_S1_summarysummarycalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid79=$!
tee < fifo/il_S2_summary_P10 fifo/il_S2_summaryeltcalc_P10 fifo/il_S2_summarypltcalc_P10 fifo/il_S2_summarysummarycalc_P10 work/il_S2_summaryaalcalc/P10.bin work/il_S2_summaryleccalc/P10.bin > /dev/null & pid80=$!
summarycalc -f  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 < fifo/il_P1 &
summarycalc -f  -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 < fifo/il_P2 &
summarycalc -f  -1 fifo/il_S1_summary_P3 -2 fifo/il_S2_summary_P3 < fifo/il_P3 &
summarycalc -f  -1 fifo/il_S1_summary_P4 -2 fifo/il_S2_summary_P4 < fifo/il_P4 &
summarycalc -f  -1 fifo/il_S1_summary_P5 -2 fifo/il_S2_summary_P5 < fifo/il_P5 &
summarycalc -f  -1 fifo/il_S1_summary_P6 -2 fifo/il_S2_summary_P6 < fifo/il_P6 &
summarycalc -f  -1 fifo/il_S1_summary_P7 -2 fifo/il_S2_summary_P7 < fifo/il_P7 &
summarycalc -f  -1 fifo/il_S1_summary_P8 -2 fifo/il_S2_summary_P8 < fifo/il_P8 &
summarycalc -f  -1 fifo/il_S1_summary_P9 -2 fifo/il_S2_summary_P9 < fifo/il_P9 &
summarycalc -f  -1 fifo/il_S1_summary_P10 -2 fifo/il_S2_summary_P10 < fifo/il_P10 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid81=$!
summarycalctocsv < fifo/gul_S1_summarysummarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid82=$!
pltcalc < fifo/gul_S1_summarypltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid83=$!

eltcalc < fifo/gul_S2_summaryeltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid84=$!
summarycalctocsv < fifo/gul_S2_summarysummarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid85=$!
pltcalc < fifo/gul_S2_summarypltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid86=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid87=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid88=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid89=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid90=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid91=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid92=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid93=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid94=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid95=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P3 > work/kat/gul_S2_eltcalc_P3 & pid96=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P3 > work/kat/gul_S2_summarycalc_P3 & pid97=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P3 > work/kat/gul_S2_pltcalc_P3 & pid98=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid99=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid100=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid101=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P4 > work/kat/gul_S2_eltcalc_P4 & pid102=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P4 > work/kat/gul_S2_summarycalc_P4 & pid103=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P4 > work/kat/gul_S2_pltcalc_P4 & pid104=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid105=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid106=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid107=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P5 > work/kat/gul_S2_eltcalc_P5 & pid108=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P5 > work/kat/gul_S2_summarycalc_P5 & pid109=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P5 > work/kat/gul_S2_pltcalc_P5 & pid110=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid111=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid112=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid113=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P6 > work/kat/gul_S2_eltcalc_P6 & pid114=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P6 > work/kat/gul_S2_summarycalc_P6 & pid115=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P6 > work/kat/gul_S2_pltcalc_P6 & pid116=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid117=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid118=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid119=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P7 > work/kat/gul_S2_eltcalc_P7 & pid120=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P7 > work/kat/gul_S2_summarycalc_P7 & pid121=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P7 > work/kat/gul_S2_pltcalc_P7 & pid122=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid123=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid124=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid125=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P8 > work/kat/gul_S2_eltcalc_P8 & pid126=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P8 > work/kat/gul_S2_summarycalc_P8 & pid127=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P8 > work/kat/gul_S2_pltcalc_P8 & pid128=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid129=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid130=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid131=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P9 > work/kat/gul_S2_eltcalc_P9 & pid132=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P9 > work/kat/gul_S2_summarycalc_P9 & pid133=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P9 > work/kat/gul_S2_pltcalc_P9 & pid134=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid135=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid136=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid137=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P10 > work/kat/gul_S2_eltcalc_P10 & pid138=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P10 > work/kat/gul_S2_summarycalc_P10 & pid139=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P10 > work/kat/gul_S2_pltcalc_P10 & pid140=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryeltcalc_P1 fifo/gul_S1_summarypltcalc_P1 fifo/gul_S1_summarysummarycalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid141=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_summaryeltcalc_P1 fifo/gul_S2_summarypltcalc_P1 fifo/gul_S2_summarysummarycalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid142=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_summaryeltcalc_P2 fifo/gul_S1_summarypltcalc_P2 fifo/gul_S1_summarysummarycalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid143=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_summaryeltcalc_P2 fifo/gul_S2_summarypltcalc_P2 fifo/gul_S2_summarysummarycalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid144=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_summaryeltcalc_P3 fifo/gul_S1_summarypltcalc_P3 fifo/gul_S1_summarysummarycalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid145=$!
tee < fifo/gul_S2_summary_P3 fifo/gul_S2_summaryeltcalc_P3 fifo/gul_S2_summarypltcalc_P3 fifo/gul_S2_summarysummarycalc_P3 work/gul_S2_summaryaalcalc/P3.bin work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid146=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_summaryeltcalc_P4 fifo/gul_S1_summarypltcalc_P4 fifo/gul_S1_summarysummarycalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid147=$!
tee < fifo/gul_S2_summary_P4 fifo/gul_S2_summaryeltcalc_P4 fifo/gul_S2_summarypltcalc_P4 fifo/gul_S2_summarysummarycalc_P4 work/gul_S2_summaryaalcalc/P4.bin work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid148=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_summaryeltcalc_P5 fifo/gul_S1_summarypltcalc_P5 fifo/gul_S1_summarysummarycalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid149=$!
tee < fifo/gul_S2_summary_P5 fifo/gul_S2_summaryeltcalc_P5 fifo/gul_S2_summarypltcalc_P5 fifo/gul_S2_summarysummarycalc_P5 work/gul_S2_summaryaalcalc/P5.bin work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid150=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_summaryeltcalc_P6 fifo/gul_S1_summarypltcalc_P6 fifo/gul_S1_summarysummarycalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid151=$!
tee < fifo/gul_S2_summary_P6 fifo/gul_S2_summaryeltcalc_P6 fifo/gul_S2_summarypltcalc_P6 fifo/gul_S2_summarysummarycalc_P6 work/gul_S2_summaryaalcalc/P6.bin work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid152=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_summaryeltcalc_P7 fifo/gul_S1_summarypltcalc_P7 fifo/gul_S1_summarysummarycalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid153=$!
tee < fifo/gul_S2_summary_P7 fifo/gul_S2_summaryeltcalc_P7 fifo/gul_S2_summarypltcalc_P7 fifo/gul_S2_summarysummarycalc_P7 work/gul_S2_summaryaalcalc/P7.bin work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid154=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_summaryeltcalc_P8 fifo/gul_S1_summarypltcalc_P8 fifo/gul_S1_summarysummarycalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid155=$!
tee < fifo/gul_S2_summary_P8 fifo/gul_S2_summaryeltcalc_P8 fifo/gul_S2_summarypltcalc_P8 fifo/gul_S2_summarysummarycalc_P8 work/gul_S2_summaryaalcalc/P8.bin work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid156=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_summaryeltcalc_P9 fifo/gul_S1_summarypltcalc_P9 fifo/gul_S1_summarysummarycalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid157=$!
tee < fifo/gul_S2_summary_P9 fifo/gul_S2_summaryeltcalc_P9 fifo/gul_S2_summarypltcalc_P9 fifo/gul_S2_summarysummarycalc_P9 work/gul_S2_summaryaalcalc/P9.bin work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid158=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_summaryeltcalc_P10 fifo/gul_S1_summarypltcalc_P10 fifo/gul_S1_summarysummarycalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid159=$!
tee < fifo/gul_S2_summary_P10 fifo/gul_S2_summaryeltcalc_P10 fifo/gul_S2_summarypltcalc_P10 fifo/gul_S2_summarysummarycalc_P10 work/gul_S2_summaryaalcalc/P10.bin work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid160=$!
summarycalc -g  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 &
summarycalc -g  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &
summarycalc -g  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 &
summarycalc -g  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 &
summarycalc -g  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 &
summarycalc -g  -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 &
summarycalc -g  -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 &
summarycalc -g  -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 &
summarycalc -g  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 &
summarycalc -g  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 &

eve 1 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P1 -i - | fmcalc -a2 > fifo/il_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P2 -i - | fmcalc -a2 > fifo/il_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P3 -i - | fmcalc -a2 > fifo/il_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P4 -i - | fmcalc -a2 > fifo/il_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P5 -i - | fmcalc -a2 > fifo/il_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P6 -i - | fmcalc -a2 > fifo/il_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P7 -i - | fmcalc -a2 > fifo/il_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P8 -i - | fmcalc -a2 > fifo/il_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P9 -i - | fmcalc -a2 > fifo/il_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P10 -i - | fmcalc -a2 > fifo/il_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P1 work/kat/il_S2_eltcalc_P2 work/kat/il_S2_eltcalc_P3 work/kat/il_S2_eltcalc_P4 work/kat/il_S2_eltcalc_P5 work/kat/il_S2_eltcalc_P6 work/kat/il_S2_eltcalc_P7 work/kat/il_S2_eltcalc_P8 work/kat/il_S2_eltcalc_P9 work/kat/il_S2_eltcalc_P10 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P1 work/kat/il_S2_pltcalc_P2 work/kat/il_S2_pltcalc_P3 work/kat/il_S2_pltcalc_P4 work/kat/il_S2_pltcalc_P5 work/kat/il_S2_pltcalc_P6 work/kat/il_S2_pltcalc_P7 work/kat/il_S2_pltcalc_P8 work/kat/il_S2_pltcalc_P9 work/kat/il_S2_pltcalc_P10 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P1 work/kat/il_S2_summarycalc_P2 work/kat/il_S2_summarycalc_P3 work/kat/il_S2_summarycalc_P4 work/kat/il_S2_summarycalc_P5 work/kat/il_S2_summarycalc_P6 work/kat/il_S2_summarycalc_P7 work/kat/il_S2_summarycalc_P8 work/kat/il_S2_summarycalc_P9 work/kat/il_S2_summarycalc_P10 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 > output/gul_S1_summarycalc.csv & kpid9=$!
kat work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 work/kat/gul_S2_eltcalc_P3 work/kat/gul_S2_eltcalc_P4 work/kat/gul_S2_eltcalc_P5 work/kat/gul_S2_eltcalc_P6 work/kat/gul_S2_eltcalc_P7 work/kat/gul_S2_eltcalc_P8 work/kat/gul_S2_eltcalc_P9 work/kat/gul_S2_eltcalc_P10 > output/gul_S2_eltcalc.csv & kpid10=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 work/kat/gul_S2_pltcalc_P3 work/kat/gul_S2_pltcalc_P4 work/kat/gul_S2_pltcalc_P5 work/kat/gul_S2_pltcalc_P6 work/kat/gul_S2_pltcalc_P7 work/kat/gul_S2_pltcalc_P8 work/kat/gul_S2_pltcalc_P9 work/kat/gul_S2_pltcalc_P10 > output/gul_S2_pltcalc.csv & kpid11=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 work/kat/gul_S2_summarycalc_P3 work/kat/gul_S2_summarycalc_P4 work/kat/gul_S2_summarycalc_P5 work/kat/gul_S2_summarycalc_P6 work/kat/gul_S2_summarycalc_P7 work/kat/gul_S2_summarycalc_P8 work/kat/gul_S2_summarycalc_P9 work/kat/gul_S2_summarycalc_P10 > output/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kil_S2_summaryaalcalc > output/il_S2_aalcalc.csv & lpid3=$!
leccalc -r -Kil_S2_summaryleccalc -F output/il_S2_leccalc_full_uncertainty_aep.csv -f output/il_S2_leccalc_full_uncertainty_oep.csv -S output/il_S2_leccalc_sample_mean_aep.csv -s output/il_S2_leccalc_sample_mean_oep.csv -W output/il_S2_leccalc_wheatsheaf_aep.csv -M output/il_S2_leccalc_wheatsheaf_mean_aep.csv -m output/il_S2_leccalc_wheatsheaf_mean_oep.csv -w output/il_S2_leccalc_wheatsheaf_oep.csv & lpid4=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid7=$!
leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8


set +e

rm fifo/gul_P1

rm fifo/gul_S1_summary_P1
rm fifo/gul_S1_summaryeltcalc_P1
rm fifo/gul_S1_eltcalc_P1
rm fifo/gul_S1_summarysummarycalc_P1
rm fifo/gul_S1_summarycalc_P1
rm fifo/gul_S1_summarypltcalc_P1
rm fifo/gul_S1_pltcalc_P1
rm fifo/gul_S2_summary_P1
rm fifo/gul_S2_summaryeltcalc_P1
rm fifo/gul_S2_eltcalc_P1
rm fifo/gul_S2_summarysummarycalc_P1
rm fifo/gul_S2_summarycalc_P1
rm fifo/gul_S2_summarypltcalc_P1
rm fifo/gul_S2_pltcalc_P1

rm fifo/gul_P2

rm fifo/gul_S1_summary_P2
rm fifo/gul_S1_summaryeltcalc_P2
rm fifo/gul_S1_eltcalc_P2
rm fifo/gul_S1_summarysummarycalc_P2
rm fifo/gul_S1_summarycalc_P2
rm fifo/gul_S1_summarypltcalc_P2
rm fifo/gul_S1_pltcalc_P2
rm fifo/gul_S2_summary_P2
rm fifo/gul_S2_summaryeltcalc_P2
rm fifo/gul_S2_eltcalc_P2
rm fifo/gul_S2_summarysummarycalc_P2
rm fifo/gul_S2_summarycalc_P2
rm fifo/gul_S2_summarypltcalc_P2
rm fifo/gul_S2_pltcalc_P2

rm fifo/gul_P3

rm fifo/gul_S1_summary_P3
rm fifo/gul_S1_summaryeltcalc_P3
rm fifo/gul_S1_eltcalc_P3
rm fifo/gul_S1_summarysummarycalc_P3
rm fifo/gul_S1_summarycalc_P3
rm fifo/gul_S1_summarypltcalc_P3
rm fifo/gul_S1_pltcalc_P3
rm fifo/gul_S2_summary_P3
rm fifo/gul_S2_summaryeltcalc_P3
rm fifo/gul_S2_eltcalc_P3
rm fifo/gul_S2_summarysummarycalc_P3
rm fifo/gul_S2_summarycalc_P3
rm fifo/gul_S2_summarypltcalc_P3
rm fifo/gul_S2_pltcalc_P3

rm fifo/gul_P4

rm fifo/gul_S1_summary_P4
rm fifo/gul_S1_summaryeltcalc_P4
rm fifo/gul_S1_eltcalc_P4
rm fifo/gul_S1_summarysummarycalc_P4
rm fifo/gul_S1_summarycalc_P4
rm fifo/gul_S1_summarypltcalc_P4
rm fifo/gul_S1_pltcalc_P4
rm fifo/gul_S2_summary_P4
rm fifo/gul_S2_summaryeltcalc_P4
rm fifo/gul_S2_eltcalc_P4
rm fifo/gul_S2_summarysummarycalc_P4
rm fifo/gul_S2_summarycalc_P4
rm fifo/gul_S2_summarypltcalc_P4
rm fifo/gul_S2_pltcalc_P4

rm fifo/gul_P5

rm fifo/gul_S1_summary_P5
rm fifo/gul_S1_summaryeltcalc_P5
rm fifo/gul_S1_eltcalc_P5
rm fifo/gul_S1_summarysummarycalc_P5
rm fifo/gul_S1_summarycalc_P5
rm fifo/gul_S1_summarypltcalc_P5
rm fifo/gul_S1_pltcalc_P5
rm fifo/gul_S2_summary_P5
rm fifo/gul_S2_summaryeltcalc_P5
rm fifo/gul_S2_eltcalc_P5
rm fifo/gul_S2_summarysummarycalc_P5
rm fifo/gul_S2_summarycalc_P5
rm fifo/gul_S2_summarypltcalc_P5
rm fifo/gul_S2_pltcalc_P5

rm fifo/gul_P6

rm fifo/gul_S1_summary_P6
rm fifo/gul_S1_summaryeltcalc_P6
rm fifo/gul_S1_eltcalc_P6
rm fifo/gul_S1_summarysummarycalc_P6
rm fifo/gul_S1_summarycalc_P6
rm fifo/gul_S1_summarypltcalc_P6
rm fifo/gul_S1_pltcalc_P6
rm fifo/gul_S2_summary_P6
rm fifo/gul_S2_summaryeltcalc_P6
rm fifo/gul_S2_eltcalc_P6
rm fifo/gul_S2_summarysummarycalc_P6
rm fifo/gul_S2_summarycalc_P6
rm fifo/gul_S2_summarypltcalc_P6
rm fifo/gul_S2_pltcalc_P6

rm fifo/gul_P7

rm fifo/gul_S1_summary_P7
rm fifo/gul_S1_summaryeltcalc_P7
rm fifo/gul_S1_eltcalc_P7
rm fifo/gul_S1_summarysummarycalc_P7
rm fifo/gul_S1_summarycalc_P7
rm fifo/gul_S1_summarypltcalc_P7
rm fifo/gul_S1_pltcalc_P7
rm fifo/gul_S2_summary_P7
rm fifo/gul_S2_summaryeltcalc_P7
rm fifo/gul_S2_eltcalc_P7
rm fifo/gul_S2_summarysummarycalc_P7
rm fifo/gul_S2_summarycalc_P7
rm fifo/gul_S2_summarypltcalc_P7
rm fifo/gul_S2_pltcalc_P7

rm fifo/gul_P8

rm fifo/gul_S1_summary_P8
rm fifo/gul_S1_summaryeltcalc_P8
rm fifo/gul_S1_eltcalc_P8
rm fifo/gul_S1_summarysummarycalc_P8
rm fifo/gul_S1_summarycalc_P8
rm fifo/gul_S1_summarypltcalc_P8
rm fifo/gul_S1_pltcalc_P8
rm fifo/gul_S2_summary_P8
rm fifo/gul_S2_summaryeltcalc_P8
rm fifo/gul_S2_eltcalc_P8
rm fifo/gul_S2_summarysummarycalc_P8
rm fifo/gul_S2_summarycalc_P8
rm fifo/gul_S2_summarypltcalc_P8
rm fifo/gul_S2_pltcalc_P8

rm fifo/gul_P9

rm fifo/gul_S1_summary_P9
rm fifo/gul_S1_summaryeltcalc_P9
rm fifo/gul_S1_eltcalc_P9
rm fifo/gul_S1_summarysummarycalc_P9
rm fifo/gul_S1_summarycalc_P9
rm fifo/gul_S1_summarypltcalc_P9
rm fifo/gul_S1_pltcalc_P9
rm fifo/gul_S2_summary_P9
rm fifo/gul_S2_summaryeltcalc_P9
rm fifo/gul_S2_eltcalc_P9
rm fifo/gul_S2_summarysummarycalc_P9
rm fifo/gul_S2_summarycalc_P9
rm fifo/gul_S2_summarypltcalc_P9
rm fifo/gul_S2_pltcalc_P9

rm fifo/gul_P10

rm fifo/gul_S1_summary_P10
rm fifo/gul_S1_summaryeltcalc_P10
rm fifo/gul_S1_eltcalc_P10
rm fifo/gul_S1_summarysummarycalc_P10
rm fifo/gul_S1_summarycalc_P10
rm fifo/gul_S1_summarypltcalc_P10
rm fifo/gul_S1_pltcalc_P10
rm fifo/gul_S2_summary_P10
rm fifo/gul_S2_summaryeltcalc_P10
rm fifo/gul_S2_eltcalc_P10
rm fifo/gul_S2_summarysummarycalc_P10
rm fifo/gul_S2_summarycalc_P10
rm fifo/gul_S2_summarypltcalc_P10
rm fifo/gul_S2_pltcalc_P10

rm -rf work/kat
rm work/gul_S1_summaryleccalc/*
rmdir work/gul_S1_summaryleccalc
rm -rf work/gul_S1_summaryaalcalc/*
rmdir work/gul_S1_summaryaalcalc
rm work/gul_S2_summaryleccalc/*
rmdir work/gul_S2_summaryleccalc
rm -rf work/gul_S2_summaryaalcalc/*
rmdir work/gul_S2_summaryaalcalc

rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryeltcalc_P1
rm fifo/il_S1_eltcalc_P1
rm fifo/il_S1_summarysummarycalc_P1
rm fifo/il_S1_summarycalc_P1
rm fifo/il_S1_summarypltcalc_P1
rm fifo/il_S1_pltcalc_P1
rm fifo/il_S2_summary_P1
rm fifo/il_S2_summaryeltcalc_P1
rm fifo/il_S2_eltcalc_P1
rm fifo/il_S2_summarysummarycalc_P1
rm fifo/il_S2_summarycalc_P1
rm fifo/il_S2_summarypltcalc_P1
rm fifo/il_S2_pltcalc_P1

rm fifo/il_P2

rm fifo/il_S1_summary_P2
rm fifo/il_S1_summaryeltcalc_P2
rm fifo/il_S1_eltcalc_P2
rm fifo/il_S1_summarysummarycalc_P2
rm fifo/il_S1_summarycalc_P2
rm fifo/il_S1_summarypltcalc_P2
rm fifo/il_S1_pltcalc_P2
rm fifo/il_S2_summary_P2
rm fifo/il_S2_summaryeltcalc_P2
rm fifo/il_S2_eltcalc_P2
rm fifo/il_S2_summarysummarycalc_P2
rm fifo/il_S2_summarycalc_P2
rm fifo/il_S2_summarypltcalc_P2
rm fifo/il_S2_pltcalc_P2

rm fifo/il_P3

rm fifo/il_S1_summary_P3
rm fifo/il_S1_summaryeltcalc_P3
rm fifo/il_S1_eltcalc_P3
rm fifo/il_S1_summarysummarycalc_P3
rm fifo/il_S1_summarycalc_P3
rm fifo/il_S1_summarypltcalc_P3
rm fifo/il_S1_pltcalc_P3
rm fifo/il_S2_summary_P3
rm fifo/il_S2_summaryeltcalc_P3
rm fifo/il_S2_eltcalc_P3
rm fifo/il_S2_summarysummarycalc_P3
rm fifo/il_S2_summarycalc_P3
rm fifo/il_S2_summarypltcalc_P3
rm fifo/il_S2_pltcalc_P3

rm fifo/il_P4

rm fifo/il_S1_summary_P4
rm fifo/il_S1_summaryeltcalc_P4
rm fifo/il_S1_eltcalc_P4
rm fifo/il_S1_summarysummarycalc_P4
rm fifo/il_S1_summarycalc_P4
rm fifo/il_S1_summarypltcalc_P4
rm fifo/il_S1_pltcalc_P4
rm fifo/il_S2_summary_P4
rm fifo/il_S2_summaryeltcalc_P4
rm fifo/il_S2_eltcalc_P4
rm fifo/il_S2_summarysummarycalc_P4
rm fifo/il_S2_summarycalc_P4
rm fifo/il_S2_summarypltcalc_P4
rm fifo/il_S2_pltcalc_P4

rm fifo/il_P5

rm fifo/il_S1_summary_P5
rm fifo/il_S1_summaryeltcalc_P5
rm fifo/il_S1_eltcalc_P5
rm fifo/il_S1_summarysummarycalc_P5
rm fifo/il_S1_summarycalc_P5
rm fifo/il_S1_summarypltcalc_P5
rm fifo/il_S1_pltcalc_P5
rm fifo/il_S2_summary_P5
rm fifo/il_S2_summaryeltcalc_P5
rm fifo/il_S2_eltcalc_P5
rm fifo/il_S2_summarysummarycalc_P5
rm fifo/il_S2_summarycalc_P5
rm fifo/il_S2_summarypltcalc_P5
rm fifo/il_S2_pltcalc_P5

rm fifo/il_P6

rm fifo/il_S1_summary_P6
rm fifo/il_S1_summaryeltcalc_P6
rm fifo/il_S1_eltcalc_P6
rm fifo/il_S1_summarysummarycalc_P6
rm fifo/il_S1_summarycalc_P6
rm fifo/il_S1_summarypltcalc_P6
rm fifo/il_S1_pltcalc_P6
rm fifo/il_S2_summary_P6
rm fifo/il_S2_summaryeltcalc_P6
rm fifo/il_S2_eltcalc_P6
rm fifo/il_S2_summarysummarycalc_P6
rm fifo/il_S2_summarycalc_P6
rm fifo/il_S2_summarypltcalc_P6
rm fifo/il_S2_pltcalc_P6

rm fifo/il_P7

rm fifo/il_S1_summary_P7
rm fifo/il_S1_summaryeltcalc_P7
rm fifo/il_S1_eltcalc_P7
rm fifo/il_S1_summarysummarycalc_P7
rm fifo/il_S1_summarycalc_P7
rm fifo/il_S1_summarypltcalc_P7
rm fifo/il_S1_pltcalc_P7
rm fifo/il_S2_summary_P7
rm fifo/il_S2_summaryeltcalc_P7
rm fifo/il_S2_eltcalc_P7
rm fifo/il_S2_summarysummarycalc_P7
rm fifo/il_S2_summarycalc_P7
rm fifo/il_S2_summarypltcalc_P7
rm fifo/il_S2_pltcalc_P7

rm fifo/il_P8

rm fifo/il_S1_summary_P8
rm fifo/il_S1_summaryeltcalc_P8
rm fifo/il_S1_eltcalc_P8
rm fifo/il_S1_summarysummarycalc_P8
rm fifo/il_S1_summarycalc_P8
rm fifo/il_S1_summarypltcalc_P8
rm fifo/il_S1_pltcalc_P8
rm fifo/il_S2_summary_P8
rm fifo/il_S2_summaryeltcalc_P8
rm fifo/il_S2_eltcalc_P8
rm fifo/il_S2_summarysummarycalc_P8
rm fifo/il_S2_summarycalc_P8
rm fifo/il_S2_summarypltcalc_P8
rm fifo/il_S2_pltcalc_P8

rm fifo/il_P9

rm fifo/il_S1_summary_P9
rm fifo/il_S1_summaryeltcalc_P9
rm fifo/il_S1_eltcalc_P9
rm fifo/il_S1_summarysummarycalc_P9
rm fifo/il_S1_summarycalc_P9
rm fifo/il_S1_summarypltcalc_P9
rm fifo/il_S1_pltcalc_P9
rm fifo/il_S2_summary_P9
rm fifo/il_S2_summaryeltcalc_P9
rm fifo/il_S2_eltcalc_P9
rm fifo/il_S2_summarysummarycalc_P9
rm fifo/il_S2_summarycalc_P9
rm fifo/il_S2_summarypltcalc_P9
rm fifo/il_S2_pltcalc_P9

rm fifo/il_P10

rm fifo/il_S1_summary_P10
rm fifo/il_S1_summaryeltcalc_P10
rm fifo/il_S1_eltcalc_P10
rm fifo/il_S1_summarysummarycalc_P10
rm fifo/il_S1_summarycalc_P10
rm fifo/il_S1_summarypltcalc_P10
rm fifo/il_S1_pltcalc_P10
rm fifo/il_S2_summary_P10
rm fifo/il_S2_summaryeltcalc_P10
rm fifo/il_S2_eltcalc_P10
rm fifo/il_S2_summarysummarycalc_P10
rm fifo/il_S2_summarycalc_P10
rm fifo/il_S2_summarypltcalc_P10
rm fifo/il_S2_pltcalc_P10

rm -rf work/kat
rm work/il_S1_summaryleccalc/*
rmdir work/il_S1_summaryleccalc
rm -rf work/il_S1_summaryaalcalc/*
rmdir work/il_S1_summaryaalcalc
rm work/il_S2_summaryleccalc/*
rmdir work/il_S2_summaryleccalc
rm -rf work/il_S2_summaryaalcalc/*
rmdir work/il_S2_summaryaalcalc
