#!/bin/bash

rm -R -f output/*
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
mkfifo fifo/gul_S1_summaryaalcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_summaryeltcalc_P1
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarysummarycalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_summarypltcalc_P1
mkfifo fifo/gul_S2_pltcalc_P1
mkfifo fifo/gul_S2_summaryaalcalc_P1

mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summaryeltcalc_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarysummarycalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_summarypltcalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S1_summaryaalcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summaryeltcalc_P2
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarysummarycalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_summarypltcalc_P2
mkfifo fifo/gul_S2_pltcalc_P2
mkfifo fifo/gul_S2_summaryaalcalc_P2

mkfifo fifo/gul_P3

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summaryeltcalc_P3
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarysummarycalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_summarypltcalc_P3
mkfifo fifo/gul_S1_pltcalc_P3
mkfifo fifo/gul_S1_summaryaalcalc_P3
mkfifo fifo/gul_S2_summary_P3
mkfifo fifo/gul_S2_summaryeltcalc_P3
mkfifo fifo/gul_S2_eltcalc_P3
mkfifo fifo/gul_S2_summarysummarycalc_P3
mkfifo fifo/gul_S2_summarycalc_P3
mkfifo fifo/gul_S2_summarypltcalc_P3
mkfifo fifo/gul_S2_pltcalc_P3
mkfifo fifo/gul_S2_summaryaalcalc_P3

mkfifo fifo/gul_P4

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summaryeltcalc_P4
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarysummarycalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_summarypltcalc_P4
mkfifo fifo/gul_S1_pltcalc_P4
mkfifo fifo/gul_S1_summaryaalcalc_P4
mkfifo fifo/gul_S2_summary_P4
mkfifo fifo/gul_S2_summaryeltcalc_P4
mkfifo fifo/gul_S2_eltcalc_P4
mkfifo fifo/gul_S2_summarysummarycalc_P4
mkfifo fifo/gul_S2_summarycalc_P4
mkfifo fifo/gul_S2_summarypltcalc_P4
mkfifo fifo/gul_S2_pltcalc_P4
mkfifo fifo/gul_S2_summaryaalcalc_P4

mkfifo fifo/gul_P5

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summaryeltcalc_P5
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarysummarycalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_summarypltcalc_P5
mkfifo fifo/gul_S1_pltcalc_P5
mkfifo fifo/gul_S1_summaryaalcalc_P5
mkfifo fifo/gul_S2_summary_P5
mkfifo fifo/gul_S2_summaryeltcalc_P5
mkfifo fifo/gul_S2_eltcalc_P5
mkfifo fifo/gul_S2_summarysummarycalc_P5
mkfifo fifo/gul_S2_summarycalc_P5
mkfifo fifo/gul_S2_summarypltcalc_P5
mkfifo fifo/gul_S2_pltcalc_P5
mkfifo fifo/gul_S2_summaryaalcalc_P5

mkfifo fifo/gul_P6

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summaryeltcalc_P6
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarysummarycalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_summarypltcalc_P6
mkfifo fifo/gul_S1_pltcalc_P6
mkfifo fifo/gul_S1_summaryaalcalc_P6
mkfifo fifo/gul_S2_summary_P6
mkfifo fifo/gul_S2_summaryeltcalc_P6
mkfifo fifo/gul_S2_eltcalc_P6
mkfifo fifo/gul_S2_summarysummarycalc_P6
mkfifo fifo/gul_S2_summarycalc_P6
mkfifo fifo/gul_S2_summarypltcalc_P6
mkfifo fifo/gul_S2_pltcalc_P6
mkfifo fifo/gul_S2_summaryaalcalc_P6

mkfifo fifo/gul_P7

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summaryeltcalc_P7
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarysummarycalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_summarypltcalc_P7
mkfifo fifo/gul_S1_pltcalc_P7
mkfifo fifo/gul_S1_summaryaalcalc_P7
mkfifo fifo/gul_S2_summary_P7
mkfifo fifo/gul_S2_summaryeltcalc_P7
mkfifo fifo/gul_S2_eltcalc_P7
mkfifo fifo/gul_S2_summarysummarycalc_P7
mkfifo fifo/gul_S2_summarycalc_P7
mkfifo fifo/gul_S2_summarypltcalc_P7
mkfifo fifo/gul_S2_pltcalc_P7
mkfifo fifo/gul_S2_summaryaalcalc_P7

mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summaryeltcalc_P8
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarysummarycalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_summarypltcalc_P8
mkfifo fifo/gul_S1_pltcalc_P8
mkfifo fifo/gul_S1_summaryaalcalc_P8
mkfifo fifo/gul_S2_summary_P8
mkfifo fifo/gul_S2_summaryeltcalc_P8
mkfifo fifo/gul_S2_eltcalc_P8
mkfifo fifo/gul_S2_summarysummarycalc_P8
mkfifo fifo/gul_S2_summarycalc_P8
mkfifo fifo/gul_S2_summarypltcalc_P8
mkfifo fifo/gul_S2_pltcalc_P8
mkfifo fifo/gul_S2_summaryaalcalc_P8

mkfifo fifo/gul_P9

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summaryeltcalc_P9
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarysummarycalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_summarypltcalc_P9
mkfifo fifo/gul_S1_pltcalc_P9
mkfifo fifo/gul_S1_summaryaalcalc_P9
mkfifo fifo/gul_S2_summary_P9
mkfifo fifo/gul_S2_summaryeltcalc_P9
mkfifo fifo/gul_S2_eltcalc_P9
mkfifo fifo/gul_S2_summarysummarycalc_P9
mkfifo fifo/gul_S2_summarycalc_P9
mkfifo fifo/gul_S2_summarypltcalc_P9
mkfifo fifo/gul_S2_pltcalc_P9
mkfifo fifo/gul_S2_summaryaalcalc_P9

mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summaryeltcalc_P10
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarysummarycalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_summarypltcalc_P10
mkfifo fifo/gul_S1_pltcalc_P10
mkfifo fifo/gul_S1_summaryaalcalc_P10
mkfifo fifo/gul_S2_summary_P10
mkfifo fifo/gul_S2_summaryeltcalc_P10
mkfifo fifo/gul_S2_eltcalc_P10
mkfifo fifo/gul_S2_summarysummarycalc_P10
mkfifo fifo/gul_S2_summarycalc_P10
mkfifo fifo/gul_S2_summarypltcalc_P10
mkfifo fifo/gul_S2_pltcalc_P10
mkfifo fifo/gul_S2_summaryaalcalc_P10

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_aalcalc
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_aalcalc
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S1_summaryaalcalc_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_summaryeltcalc_P1
mkfifo fifo/il_S2_eltcalc_P1
mkfifo fifo/il_S2_summarysummarycalc_P1
mkfifo fifo/il_S2_summarycalc_P1
mkfifo fifo/il_S2_summarypltcalc_P1
mkfifo fifo/il_S2_pltcalc_P1
mkfifo fifo/il_S2_summaryaalcalc_P1

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summaryeltcalc_P2
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarysummarycalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_summarypltcalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S1_summaryaalcalc_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_summaryeltcalc_P2
mkfifo fifo/il_S2_eltcalc_P2
mkfifo fifo/il_S2_summarysummarycalc_P2
mkfifo fifo/il_S2_summarycalc_P2
mkfifo fifo/il_S2_summarypltcalc_P2
mkfifo fifo/il_S2_pltcalc_P2
mkfifo fifo/il_S2_summaryaalcalc_P2

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summaryeltcalc_P3
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarysummarycalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_summarypltcalc_P3
mkfifo fifo/il_S1_pltcalc_P3
mkfifo fifo/il_S1_summaryaalcalc_P3
mkfifo fifo/il_S2_summary_P3
mkfifo fifo/il_S2_summaryeltcalc_P3
mkfifo fifo/il_S2_eltcalc_P3
mkfifo fifo/il_S2_summarysummarycalc_P3
mkfifo fifo/il_S2_summarycalc_P3
mkfifo fifo/il_S2_summarypltcalc_P3
mkfifo fifo/il_S2_pltcalc_P3
mkfifo fifo/il_S2_summaryaalcalc_P3

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summaryeltcalc_P4
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarysummarycalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_summarypltcalc_P4
mkfifo fifo/il_S1_pltcalc_P4
mkfifo fifo/il_S1_summaryaalcalc_P4
mkfifo fifo/il_S2_summary_P4
mkfifo fifo/il_S2_summaryeltcalc_P4
mkfifo fifo/il_S2_eltcalc_P4
mkfifo fifo/il_S2_summarysummarycalc_P4
mkfifo fifo/il_S2_summarycalc_P4
mkfifo fifo/il_S2_summarypltcalc_P4
mkfifo fifo/il_S2_pltcalc_P4
mkfifo fifo/il_S2_summaryaalcalc_P4

mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summaryeltcalc_P5
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarysummarycalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_summarypltcalc_P5
mkfifo fifo/il_S1_pltcalc_P5
mkfifo fifo/il_S1_summaryaalcalc_P5
mkfifo fifo/il_S2_summary_P5
mkfifo fifo/il_S2_summaryeltcalc_P5
mkfifo fifo/il_S2_eltcalc_P5
mkfifo fifo/il_S2_summarysummarycalc_P5
mkfifo fifo/il_S2_summarycalc_P5
mkfifo fifo/il_S2_summarypltcalc_P5
mkfifo fifo/il_S2_pltcalc_P5
mkfifo fifo/il_S2_summaryaalcalc_P5

mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summaryeltcalc_P6
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarysummarycalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_summarypltcalc_P6
mkfifo fifo/il_S1_pltcalc_P6
mkfifo fifo/il_S1_summaryaalcalc_P6
mkfifo fifo/il_S2_summary_P6
mkfifo fifo/il_S2_summaryeltcalc_P6
mkfifo fifo/il_S2_eltcalc_P6
mkfifo fifo/il_S2_summarysummarycalc_P6
mkfifo fifo/il_S2_summarycalc_P6
mkfifo fifo/il_S2_summarypltcalc_P6
mkfifo fifo/il_S2_pltcalc_P6
mkfifo fifo/il_S2_summaryaalcalc_P6

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summaryeltcalc_P7
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarysummarycalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_summarypltcalc_P7
mkfifo fifo/il_S1_pltcalc_P7
mkfifo fifo/il_S1_summaryaalcalc_P7
mkfifo fifo/il_S2_summary_P7
mkfifo fifo/il_S2_summaryeltcalc_P7
mkfifo fifo/il_S2_eltcalc_P7
mkfifo fifo/il_S2_summarysummarycalc_P7
mkfifo fifo/il_S2_summarycalc_P7
mkfifo fifo/il_S2_summarypltcalc_P7
mkfifo fifo/il_S2_pltcalc_P7
mkfifo fifo/il_S2_summaryaalcalc_P7

mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summaryeltcalc_P8
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarysummarycalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_summarypltcalc_P8
mkfifo fifo/il_S1_pltcalc_P8
mkfifo fifo/il_S1_summaryaalcalc_P8
mkfifo fifo/il_S2_summary_P8
mkfifo fifo/il_S2_summaryeltcalc_P8
mkfifo fifo/il_S2_eltcalc_P8
mkfifo fifo/il_S2_summarysummarycalc_P8
mkfifo fifo/il_S2_summarycalc_P8
mkfifo fifo/il_S2_summarypltcalc_P8
mkfifo fifo/il_S2_pltcalc_P8
mkfifo fifo/il_S2_summaryaalcalc_P8

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summaryeltcalc_P9
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarysummarycalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_summarypltcalc_P9
mkfifo fifo/il_S1_pltcalc_P9
mkfifo fifo/il_S1_summaryaalcalc_P9
mkfifo fifo/il_S2_summary_P9
mkfifo fifo/il_S2_summaryeltcalc_P9
mkfifo fifo/il_S2_eltcalc_P9
mkfifo fifo/il_S2_summarysummarycalc_P9
mkfifo fifo/il_S2_summarycalc_P9
mkfifo fifo/il_S2_summarypltcalc_P9
mkfifo fifo/il_S2_pltcalc_P9
mkfifo fifo/il_S2_summaryaalcalc_P9

mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summaryeltcalc_P10
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarysummarycalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_summarypltcalc_P10
mkfifo fifo/il_S1_pltcalc_P10
mkfifo fifo/il_S1_summaryaalcalc_P10
mkfifo fifo/il_S2_summary_P10
mkfifo fifo/il_S2_summaryeltcalc_P10
mkfifo fifo/il_S2_eltcalc_P10
mkfifo fifo/il_S2_summarysummarycalc_P10
mkfifo fifo/il_S2_summarycalc_P10
mkfifo fifo/il_S2_summarypltcalc_P10
mkfifo fifo/il_S2_pltcalc_P10
mkfifo fifo/il_S2_summaryaalcalc_P10

mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_aalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_aalcalc

# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
aalcalc < fifo/il_S1_summaryaalcalc_P1 > work/il_S1_aalcalc/P1.bin & pid4=$!

eltcalc < fifo/il_S2_summaryeltcalc_P1 > work/kat/il_S2_eltcalc_P1 & pid5=$!
summarycalctocsv < fifo/il_S2_summarysummarycalc_P1 > work/kat/il_S2_summarycalc_P1 & pid6=$!
pltcalc < fifo/il_S2_summarypltcalc_P1 > work/kat/il_S2_pltcalc_P1 & pid7=$!
aalcalc < fifo/il_S2_summaryaalcalc_P1 > work/il_S2_aalcalc/P1.bin & pid8=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid9=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid10=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid11=$!
aalcalc < fifo/il_S1_summaryaalcalc_P2 > work/il_S1_aalcalc/P2.bin & pid12=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P2 > work/kat/il_S2_eltcalc_P2 & pid13=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P2 > work/kat/il_S2_summarycalc_P2 & pid14=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P2 > work/kat/il_S2_pltcalc_P2 & pid15=$!
aalcalc < fifo/il_S2_summaryaalcalc_P2 > work/il_S2_aalcalc/P2.bin & pid16=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid17=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid18=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid19=$!
aalcalc < fifo/il_S1_summaryaalcalc_P3 > work/il_S1_aalcalc/P3.bin & pid20=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P3 > work/kat/il_S2_eltcalc_P3 & pid21=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P3 > work/kat/il_S2_summarycalc_P3 & pid22=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P3 > work/kat/il_S2_pltcalc_P3 & pid23=$!
aalcalc < fifo/il_S2_summaryaalcalc_P3 > work/il_S2_aalcalc/P3.bin & pid24=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid25=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid26=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid27=$!
aalcalc < fifo/il_S1_summaryaalcalc_P4 > work/il_S1_aalcalc/P4.bin & pid28=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P4 > work/kat/il_S2_eltcalc_P4 & pid29=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P4 > work/kat/il_S2_summarycalc_P4 & pid30=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P4 > work/kat/il_S2_pltcalc_P4 & pid31=$!
aalcalc < fifo/il_S2_summaryaalcalc_P4 > work/il_S2_aalcalc/P4.bin & pid32=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid33=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid34=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid35=$!
aalcalc < fifo/il_S1_summaryaalcalc_P5 > work/il_S1_aalcalc/P5.bin & pid36=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P5 > work/kat/il_S2_eltcalc_P5 & pid37=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P5 > work/kat/il_S2_summarycalc_P5 & pid38=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P5 > work/kat/il_S2_pltcalc_P5 & pid39=$!
aalcalc < fifo/il_S2_summaryaalcalc_P5 > work/il_S2_aalcalc/P5.bin & pid40=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P6 > work/kat/il_S1_eltcalc_P6 & pid41=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P6 > work/kat/il_S1_summarycalc_P6 & pid42=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid43=$!
aalcalc < fifo/il_S1_summaryaalcalc_P6 > work/il_S1_aalcalc/P6.bin & pid44=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P6 > work/kat/il_S2_eltcalc_P6 & pid45=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P6 > work/kat/il_S2_summarycalc_P6 & pid46=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P6 > work/kat/il_S2_pltcalc_P6 & pid47=$!
aalcalc < fifo/il_S2_summaryaalcalc_P6 > work/il_S2_aalcalc/P6.bin & pid48=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid49=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid50=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid51=$!
aalcalc < fifo/il_S1_summaryaalcalc_P7 > work/il_S1_aalcalc/P7.bin & pid52=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P7 > work/kat/il_S2_eltcalc_P7 & pid53=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P7 > work/kat/il_S2_summarycalc_P7 & pid54=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P7 > work/kat/il_S2_pltcalc_P7 & pid55=$!
aalcalc < fifo/il_S2_summaryaalcalc_P7 > work/il_S2_aalcalc/P7.bin & pid56=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P8 > work/kat/il_S1_eltcalc_P8 & pid57=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P8 > work/kat/il_S1_summarycalc_P8 & pid58=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P8 > work/kat/il_S1_pltcalc_P8 & pid59=$!
aalcalc < fifo/il_S1_summaryaalcalc_P8 > work/il_S1_aalcalc/P8.bin & pid60=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P8 > work/kat/il_S2_eltcalc_P8 & pid61=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P8 > work/kat/il_S2_summarycalc_P8 & pid62=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P8 > work/kat/il_S2_pltcalc_P8 & pid63=$!
aalcalc < fifo/il_S2_summaryaalcalc_P8 > work/il_S2_aalcalc/P8.bin & pid64=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P9 > work/kat/il_S1_eltcalc_P9 & pid65=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P9 > work/kat/il_S1_summarycalc_P9 & pid66=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P9 > work/kat/il_S1_pltcalc_P9 & pid67=$!
aalcalc < fifo/il_S1_summaryaalcalc_P9 > work/il_S1_aalcalc/P9.bin & pid68=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P9 > work/kat/il_S2_eltcalc_P9 & pid69=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P9 > work/kat/il_S2_summarycalc_P9 & pid70=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P9 > work/kat/il_S2_pltcalc_P9 & pid71=$!
aalcalc < fifo/il_S2_summaryaalcalc_P9 > work/il_S2_aalcalc/P9.bin & pid72=$!

eltcalc -s < fifo/il_S1_summaryeltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid73=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid74=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid75=$!
aalcalc < fifo/il_S1_summaryaalcalc_P10 > work/il_S1_aalcalc/P10.bin & pid76=$!

eltcalc -s < fifo/il_S2_summaryeltcalc_P10 > work/kat/il_S2_eltcalc_P10 & pid77=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P10 > work/kat/il_S2_summarycalc_P10 & pid78=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P10 > work/kat/il_S2_pltcalc_P10 & pid79=$!
aalcalc < fifo/il_S2_summaryaalcalc_P10 > work/il_S2_aalcalc/P10.bin & pid80=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 fifo/il_S1_summarypltcalc_P1 fifo/il_S1_summarysummarycalc_P1 fifo/il_S1_summaryaalcalc_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid81=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_summaryeltcalc_P1 fifo/il_S2_summarypltcalc_P1 fifo/il_S2_summarysummarycalc_P1 fifo/il_S2_summaryaalcalc_P1 work/il_S2_summaryleccalc/P1.bin > /dev/null & pid82=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_summaryeltcalc_P2 fifo/il_S1_summarypltcalc_P2 fifo/il_S1_summarysummarycalc_P2 fifo/il_S1_summaryaalcalc_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid83=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_summaryeltcalc_P2 fifo/il_S2_summarypltcalc_P2 fifo/il_S2_summarysummarycalc_P2 fifo/il_S2_summaryaalcalc_P2 work/il_S2_summaryleccalc/P2.bin > /dev/null & pid84=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summaryeltcalc_P3 fifo/il_S1_summarypltcalc_P3 fifo/il_S1_summarysummarycalc_P3 fifo/il_S1_summaryaalcalc_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid85=$!
tee < fifo/il_S2_summary_P3 fifo/il_S2_summaryeltcalc_P3 fifo/il_S2_summarypltcalc_P3 fifo/il_S2_summarysummarycalc_P3 fifo/il_S2_summaryaalcalc_P3 work/il_S2_summaryleccalc/P3.bin > /dev/null & pid86=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summaryeltcalc_P4 fifo/il_S1_summarypltcalc_P4 fifo/il_S1_summarysummarycalc_P4 fifo/il_S1_summaryaalcalc_P4 work/il_S1_summaryleccalc/P4.bin > /dev/null & pid87=$!
tee < fifo/il_S2_summary_P4 fifo/il_S2_summaryeltcalc_P4 fifo/il_S2_summarypltcalc_P4 fifo/il_S2_summarysummarycalc_P4 fifo/il_S2_summaryaalcalc_P4 work/il_S2_summaryleccalc/P4.bin > /dev/null & pid88=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summaryeltcalc_P5 fifo/il_S1_summarypltcalc_P5 fifo/il_S1_summarysummarycalc_P5 fifo/il_S1_summaryaalcalc_P5 work/il_S1_summaryleccalc/P5.bin > /dev/null & pid89=$!
tee < fifo/il_S2_summary_P5 fifo/il_S2_summaryeltcalc_P5 fifo/il_S2_summarypltcalc_P5 fifo/il_S2_summarysummarycalc_P5 fifo/il_S2_summaryaalcalc_P5 work/il_S2_summaryleccalc/P5.bin > /dev/null & pid90=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_summaryeltcalc_P6 fifo/il_S1_summarypltcalc_P6 fifo/il_S1_summarysummarycalc_P6 fifo/il_S1_summaryaalcalc_P6 work/il_S1_summaryleccalc/P6.bin > /dev/null & pid91=$!
tee < fifo/il_S2_summary_P6 fifo/il_S2_summaryeltcalc_P6 fifo/il_S2_summarypltcalc_P6 fifo/il_S2_summarysummarycalc_P6 fifo/il_S2_summaryaalcalc_P6 work/il_S2_summaryleccalc/P6.bin > /dev/null & pid92=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_summaryeltcalc_P7 fifo/il_S1_summarypltcalc_P7 fifo/il_S1_summarysummarycalc_P7 fifo/il_S1_summaryaalcalc_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid93=$!
tee < fifo/il_S2_summary_P7 fifo/il_S2_summaryeltcalc_P7 fifo/il_S2_summarypltcalc_P7 fifo/il_S2_summarysummarycalc_P7 fifo/il_S2_summaryaalcalc_P7 work/il_S2_summaryleccalc/P7.bin > /dev/null & pid94=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_summaryeltcalc_P8 fifo/il_S1_summarypltcalc_P8 fifo/il_S1_summarysummarycalc_P8 fifo/il_S1_summaryaalcalc_P8 work/il_S1_summaryleccalc/P8.bin > /dev/null & pid95=$!
tee < fifo/il_S2_summary_P8 fifo/il_S2_summaryeltcalc_P8 fifo/il_S2_summarypltcalc_P8 fifo/il_S2_summarysummarycalc_P8 fifo/il_S2_summaryaalcalc_P8 work/il_S2_summaryleccalc/P8.bin > /dev/null & pid96=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_summaryeltcalc_P9 fifo/il_S1_summarypltcalc_P9 fifo/il_S1_summarysummarycalc_P9 fifo/il_S1_summaryaalcalc_P9 work/il_S1_summaryleccalc/P9.bin > /dev/null & pid97=$!
tee < fifo/il_S2_summary_P9 fifo/il_S2_summaryeltcalc_P9 fifo/il_S2_summarypltcalc_P9 fifo/il_S2_summarysummarycalc_P9 fifo/il_S2_summaryaalcalc_P9 work/il_S2_summaryleccalc/P9.bin > /dev/null & pid98=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_summaryeltcalc_P10 fifo/il_S1_summarypltcalc_P10 fifo/il_S1_summarysummarycalc_P10 fifo/il_S1_summaryaalcalc_P10 work/il_S1_summaryleccalc/P10.bin > /dev/null & pid99=$!
tee < fifo/il_S2_summary_P10 fifo/il_S2_summaryeltcalc_P10 fifo/il_S2_summarypltcalc_P10 fifo/il_S2_summarysummarycalc_P10 fifo/il_S2_summaryaalcalc_P10 work/il_S2_summaryleccalc/P10.bin > /dev/null & pid100=$!
summarycalc -f -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 < fifo/il_P1 &
summarycalc -f -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 < fifo/il_P2 &
summarycalc -f -1 fifo/il_S1_summary_P3 -2 fifo/il_S2_summary_P3 < fifo/il_P3 &
summarycalc -f -1 fifo/il_S1_summary_P4 -2 fifo/il_S2_summary_P4 < fifo/il_P4 &
summarycalc -f -1 fifo/il_S1_summary_P5 -2 fifo/il_S2_summary_P5 < fifo/il_P5 &
summarycalc -f -1 fifo/il_S1_summary_P6 -2 fifo/il_S2_summary_P6 < fifo/il_P6 &
summarycalc -f -1 fifo/il_S1_summary_P7 -2 fifo/il_S2_summary_P7 < fifo/il_P7 &
summarycalc -f -1 fifo/il_S1_summary_P8 -2 fifo/il_S2_summary_P8 < fifo/il_P8 &
summarycalc -f -1 fifo/il_S1_summary_P9 -2 fifo/il_S2_summary_P9 < fifo/il_P9 &
summarycalc -f -1 fifo/il_S1_summary_P10 -2 fifo/il_S2_summary_P10 < fifo/il_P10 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid101=$!
summarycalctocsv < fifo/gul_S1_summarysummarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid102=$!
pltcalc < fifo/gul_S1_summarypltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid103=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P1 > work/gul_S1_aalcalc/P1.bin & pid104=$!

eltcalc < fifo/gul_S2_summaryeltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid105=$!
summarycalctocsv < fifo/gul_S2_summarysummarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid106=$!
pltcalc < fifo/gul_S2_summarypltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid107=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P1 > work/gul_S2_aalcalc/P1.bin & pid108=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid109=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid110=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid111=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P2 > work/gul_S1_aalcalc/P2.bin & pid112=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid113=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid114=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid115=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P2 > work/gul_S2_aalcalc/P2.bin & pid116=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid117=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid118=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid119=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P3 > work/gul_S1_aalcalc/P3.bin & pid120=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P3 > work/kat/gul_S2_eltcalc_P3 & pid121=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P3 > work/kat/gul_S2_summarycalc_P3 & pid122=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P3 > work/kat/gul_S2_pltcalc_P3 & pid123=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P3 > work/gul_S2_aalcalc/P3.bin & pid124=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid125=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid126=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid127=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P4 > work/gul_S1_aalcalc/P4.bin & pid128=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P4 > work/kat/gul_S2_eltcalc_P4 & pid129=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P4 > work/kat/gul_S2_summarycalc_P4 & pid130=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P4 > work/kat/gul_S2_pltcalc_P4 & pid131=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P4 > work/gul_S2_aalcalc/P4.bin & pid132=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid133=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid134=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid135=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P5 > work/gul_S1_aalcalc/P5.bin & pid136=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P5 > work/kat/gul_S2_eltcalc_P5 & pid137=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P5 > work/kat/gul_S2_summarycalc_P5 & pid138=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P5 > work/kat/gul_S2_pltcalc_P5 & pid139=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P5 > work/gul_S2_aalcalc/P5.bin & pid140=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid141=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid142=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid143=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P6 > work/gul_S1_aalcalc/P6.bin & pid144=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P6 > work/kat/gul_S2_eltcalc_P6 & pid145=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P6 > work/kat/gul_S2_summarycalc_P6 & pid146=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P6 > work/kat/gul_S2_pltcalc_P6 & pid147=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P6 > work/gul_S2_aalcalc/P6.bin & pid148=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid149=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid150=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid151=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P7 > work/gul_S1_aalcalc/P7.bin & pid152=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P7 > work/kat/gul_S2_eltcalc_P7 & pid153=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P7 > work/kat/gul_S2_summarycalc_P7 & pid154=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P7 > work/kat/gul_S2_pltcalc_P7 & pid155=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P7 > work/gul_S2_aalcalc/P7.bin & pid156=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid157=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid158=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid159=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P8 > work/gul_S1_aalcalc/P8.bin & pid160=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P8 > work/kat/gul_S2_eltcalc_P8 & pid161=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P8 > work/kat/gul_S2_summarycalc_P8 & pid162=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P8 > work/kat/gul_S2_pltcalc_P8 & pid163=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P8 > work/gul_S2_aalcalc/P8.bin & pid164=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid165=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid166=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid167=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P9 > work/gul_S1_aalcalc/P9.bin & pid168=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P9 > work/kat/gul_S2_eltcalc_P9 & pid169=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P9 > work/kat/gul_S2_summarycalc_P9 & pid170=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P9 > work/kat/gul_S2_pltcalc_P9 & pid171=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P9 > work/gul_S2_aalcalc/P9.bin & pid172=$!

eltcalc -s < fifo/gul_S1_summaryeltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid173=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid174=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid175=$!
aalcalc < fifo/gul_S1_summaryaalcalc_P10 > work/gul_S1_aalcalc/P10.bin & pid176=$!

eltcalc -s < fifo/gul_S2_summaryeltcalc_P10 > work/kat/gul_S2_eltcalc_P10 & pid177=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P10 > work/kat/gul_S2_summarycalc_P10 & pid178=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P10 > work/kat/gul_S2_pltcalc_P10 & pid179=$!
aalcalc < fifo/gul_S2_summaryaalcalc_P10 > work/gul_S2_aalcalc/P10.bin & pid180=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryeltcalc_P1 fifo/gul_S1_summarypltcalc_P1 fifo/gul_S1_summarysummarycalc_P1 fifo/gul_S1_summaryaalcalc_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid181=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_summaryeltcalc_P1 fifo/gul_S2_summarypltcalc_P1 fifo/gul_S2_summarysummarycalc_P1 fifo/gul_S2_summaryaalcalc_P1 work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid182=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_summaryeltcalc_P2 fifo/gul_S1_summarypltcalc_P2 fifo/gul_S1_summarysummarycalc_P2 fifo/gul_S1_summaryaalcalc_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid183=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_summaryeltcalc_P2 fifo/gul_S2_summarypltcalc_P2 fifo/gul_S2_summarysummarycalc_P2 fifo/gul_S2_summaryaalcalc_P2 work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid184=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_summaryeltcalc_P3 fifo/gul_S1_summarypltcalc_P3 fifo/gul_S1_summarysummarycalc_P3 fifo/gul_S1_summaryaalcalc_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid185=$!
tee < fifo/gul_S2_summary_P3 fifo/gul_S2_summaryeltcalc_P3 fifo/gul_S2_summarypltcalc_P3 fifo/gul_S2_summarysummarycalc_P3 fifo/gul_S2_summaryaalcalc_P3 work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid186=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_summaryeltcalc_P4 fifo/gul_S1_summarypltcalc_P4 fifo/gul_S1_summarysummarycalc_P4 fifo/gul_S1_summaryaalcalc_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid187=$!
tee < fifo/gul_S2_summary_P4 fifo/gul_S2_summaryeltcalc_P4 fifo/gul_S2_summarypltcalc_P4 fifo/gul_S2_summarysummarycalc_P4 fifo/gul_S2_summaryaalcalc_P4 work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid188=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_summaryeltcalc_P5 fifo/gul_S1_summarypltcalc_P5 fifo/gul_S1_summarysummarycalc_P5 fifo/gul_S1_summaryaalcalc_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid189=$!
tee < fifo/gul_S2_summary_P5 fifo/gul_S2_summaryeltcalc_P5 fifo/gul_S2_summarypltcalc_P5 fifo/gul_S2_summarysummarycalc_P5 fifo/gul_S2_summaryaalcalc_P5 work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid190=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_summaryeltcalc_P6 fifo/gul_S1_summarypltcalc_P6 fifo/gul_S1_summarysummarycalc_P6 fifo/gul_S1_summaryaalcalc_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid191=$!
tee < fifo/gul_S2_summary_P6 fifo/gul_S2_summaryeltcalc_P6 fifo/gul_S2_summarypltcalc_P6 fifo/gul_S2_summarysummarycalc_P6 fifo/gul_S2_summaryaalcalc_P6 work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid192=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_summaryeltcalc_P7 fifo/gul_S1_summarypltcalc_P7 fifo/gul_S1_summarysummarycalc_P7 fifo/gul_S1_summaryaalcalc_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid193=$!
tee < fifo/gul_S2_summary_P7 fifo/gul_S2_summaryeltcalc_P7 fifo/gul_S2_summarypltcalc_P7 fifo/gul_S2_summarysummarycalc_P7 fifo/gul_S2_summaryaalcalc_P7 work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid194=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_summaryeltcalc_P8 fifo/gul_S1_summarypltcalc_P8 fifo/gul_S1_summarysummarycalc_P8 fifo/gul_S1_summaryaalcalc_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid195=$!
tee < fifo/gul_S2_summary_P8 fifo/gul_S2_summaryeltcalc_P8 fifo/gul_S2_summarypltcalc_P8 fifo/gul_S2_summarysummarycalc_P8 fifo/gul_S2_summaryaalcalc_P8 work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid196=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_summaryeltcalc_P9 fifo/gul_S1_summarypltcalc_P9 fifo/gul_S1_summarysummarycalc_P9 fifo/gul_S1_summaryaalcalc_P9 work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid197=$!
tee < fifo/gul_S2_summary_P9 fifo/gul_S2_summaryeltcalc_P9 fifo/gul_S2_summarypltcalc_P9 fifo/gul_S2_summarysummarycalc_P9 fifo/gul_S2_summaryaalcalc_P9 work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid198=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_summaryeltcalc_P10 fifo/gul_S1_summarypltcalc_P10 fifo/gul_S1_summarysummarycalc_P10 fifo/gul_S1_summaryaalcalc_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid199=$!
tee < fifo/gul_S2_summary_P10 fifo/gul_S2_summaryeltcalc_P10 fifo/gul_S2_summarypltcalc_P10 fifo/gul_S2_summarysummarycalc_P10 fifo/gul_S2_summaryaalcalc_P10 work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid200=$!
summarycalc -g -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 &
summarycalc -g -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &
summarycalc -g -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 &
summarycalc -g -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 &
summarycalc -g -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 &
summarycalc -g -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 &
summarycalc -g -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 &
summarycalc -g -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 &
summarycalc -g -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 &
summarycalc -g -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 &

eve 1 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P1 -i - | fmcalc -a 2 > fifo/il_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P2 -i - | fmcalc -a 2 > fifo/il_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P3 -i - | fmcalc -a 2 > fifo/il_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P4 -i - | fmcalc -a 2 > fifo/il_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P5 -i - | fmcalc -a 2 > fifo/il_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P6 -i - | fmcalc -a 2 > fifo/il_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P7 -i - | fmcalc -a 2 > fifo/il_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P8 -i - | fmcalc -a 2 > fifo/il_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P9 -i - | fmcalc -a 2 > fifo/il_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P10 -i - | fmcalc -a 2 > fifo/il_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200


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


aalsummary -Kil_S1_aalcalc > output/il_S1_aalcalc.csv & apid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid1=$!
aalsummary -Kil_S2_aalcalc > output/il_S2_aalcalc.csv & apid2=$!
leccalc -r -Kil_S2_summaryleccalc -F output/il_S2_leccalc_full_uncertainty_aep.csv -f output/il_S2_leccalc_full_uncertainty_oep.csv -S output/il_S2_leccalc_sample_mean_aep.csv -s output/il_S2_leccalc_sample_mean_oep.csv -W output/il_S2_leccalc_wheatsheaf_aep.csv -M output/il_S2_leccalc_wheatsheaf_mean_aep.csv -m output/il_S2_leccalc_wheatsheaf_mean_oep.csv -w output/il_S2_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalsummary -Kgul_S1_aalcalc > output/gul_S1_aalcalc.csv & apid3=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid3=$!
aalsummary -Kgul_S2_aalcalc > output/gul_S2_aalcalc.csv & apid4=$!
leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv & lpid4=$!
wait $apid1 $apid2 $apid3 $apid4

wait $lpid1 $lpid2 $lpid3 $lpid4

rm fifo/gul_P1

rm fifo/gul_S1_summary_P1
rm fifo/gul_S1_summaryeltcalc_P1
rm fifo/gul_S1_eltcalc_P1
rm fifo/gul_S1_summarysummarycalc_P1
rm fifo/gul_S1_summarycalc_P1
rm fifo/gul_S1_summarypltcalc_P1
rm fifo/gul_S1_pltcalc_P1
rm fifo/gul_S1_summaryaalcalc_P1
rm fifo/gul_S2_summary_P1
rm fifo/gul_S2_summaryeltcalc_P1
rm fifo/gul_S2_eltcalc_P1
rm fifo/gul_S2_summarysummarycalc_P1
rm fifo/gul_S2_summarycalc_P1
rm fifo/gul_S2_summarypltcalc_P1
rm fifo/gul_S2_pltcalc_P1
rm fifo/gul_S2_summaryaalcalc_P1

rm fifo/gul_P2

rm fifo/gul_S1_summary_P2
rm fifo/gul_S1_summaryeltcalc_P2
rm fifo/gul_S1_eltcalc_P2
rm fifo/gul_S1_summarysummarycalc_P2
rm fifo/gul_S1_summarycalc_P2
rm fifo/gul_S1_summarypltcalc_P2
rm fifo/gul_S1_pltcalc_P2
rm fifo/gul_S1_summaryaalcalc_P2
rm fifo/gul_S2_summary_P2
rm fifo/gul_S2_summaryeltcalc_P2
rm fifo/gul_S2_eltcalc_P2
rm fifo/gul_S2_summarysummarycalc_P2
rm fifo/gul_S2_summarycalc_P2
rm fifo/gul_S2_summarypltcalc_P2
rm fifo/gul_S2_pltcalc_P2
rm fifo/gul_S2_summaryaalcalc_P2

rm fifo/gul_P3

rm fifo/gul_S1_summary_P3
rm fifo/gul_S1_summaryeltcalc_P3
rm fifo/gul_S1_eltcalc_P3
rm fifo/gul_S1_summarysummarycalc_P3
rm fifo/gul_S1_summarycalc_P3
rm fifo/gul_S1_summarypltcalc_P3
rm fifo/gul_S1_pltcalc_P3
rm fifo/gul_S1_summaryaalcalc_P3
rm fifo/gul_S2_summary_P3
rm fifo/gul_S2_summaryeltcalc_P3
rm fifo/gul_S2_eltcalc_P3
rm fifo/gul_S2_summarysummarycalc_P3
rm fifo/gul_S2_summarycalc_P3
rm fifo/gul_S2_summarypltcalc_P3
rm fifo/gul_S2_pltcalc_P3
rm fifo/gul_S2_summaryaalcalc_P3

rm fifo/gul_P4

rm fifo/gul_S1_summary_P4
rm fifo/gul_S1_summaryeltcalc_P4
rm fifo/gul_S1_eltcalc_P4
rm fifo/gul_S1_summarysummarycalc_P4
rm fifo/gul_S1_summarycalc_P4
rm fifo/gul_S1_summarypltcalc_P4
rm fifo/gul_S1_pltcalc_P4
rm fifo/gul_S1_summaryaalcalc_P4
rm fifo/gul_S2_summary_P4
rm fifo/gul_S2_summaryeltcalc_P4
rm fifo/gul_S2_eltcalc_P4
rm fifo/gul_S2_summarysummarycalc_P4
rm fifo/gul_S2_summarycalc_P4
rm fifo/gul_S2_summarypltcalc_P4
rm fifo/gul_S2_pltcalc_P4
rm fifo/gul_S2_summaryaalcalc_P4

rm fifo/gul_P5

rm fifo/gul_S1_summary_P5
rm fifo/gul_S1_summaryeltcalc_P5
rm fifo/gul_S1_eltcalc_P5
rm fifo/gul_S1_summarysummarycalc_P5
rm fifo/gul_S1_summarycalc_P5
rm fifo/gul_S1_summarypltcalc_P5
rm fifo/gul_S1_pltcalc_P5
rm fifo/gul_S1_summaryaalcalc_P5
rm fifo/gul_S2_summary_P5
rm fifo/gul_S2_summaryeltcalc_P5
rm fifo/gul_S2_eltcalc_P5
rm fifo/gul_S2_summarysummarycalc_P5
rm fifo/gul_S2_summarycalc_P5
rm fifo/gul_S2_summarypltcalc_P5
rm fifo/gul_S2_pltcalc_P5
rm fifo/gul_S2_summaryaalcalc_P5

rm fifo/gul_P6

rm fifo/gul_S1_summary_P6
rm fifo/gul_S1_summaryeltcalc_P6
rm fifo/gul_S1_eltcalc_P6
rm fifo/gul_S1_summarysummarycalc_P6
rm fifo/gul_S1_summarycalc_P6
rm fifo/gul_S1_summarypltcalc_P6
rm fifo/gul_S1_pltcalc_P6
rm fifo/gul_S1_summaryaalcalc_P6
rm fifo/gul_S2_summary_P6
rm fifo/gul_S2_summaryeltcalc_P6
rm fifo/gul_S2_eltcalc_P6
rm fifo/gul_S2_summarysummarycalc_P6
rm fifo/gul_S2_summarycalc_P6
rm fifo/gul_S2_summarypltcalc_P6
rm fifo/gul_S2_pltcalc_P6
rm fifo/gul_S2_summaryaalcalc_P6

rm fifo/gul_P7

rm fifo/gul_S1_summary_P7
rm fifo/gul_S1_summaryeltcalc_P7
rm fifo/gul_S1_eltcalc_P7
rm fifo/gul_S1_summarysummarycalc_P7
rm fifo/gul_S1_summarycalc_P7
rm fifo/gul_S1_summarypltcalc_P7
rm fifo/gul_S1_pltcalc_P7
rm fifo/gul_S1_summaryaalcalc_P7
rm fifo/gul_S2_summary_P7
rm fifo/gul_S2_summaryeltcalc_P7
rm fifo/gul_S2_eltcalc_P7
rm fifo/gul_S2_summarysummarycalc_P7
rm fifo/gul_S2_summarycalc_P7
rm fifo/gul_S2_summarypltcalc_P7
rm fifo/gul_S2_pltcalc_P7
rm fifo/gul_S2_summaryaalcalc_P7

rm fifo/gul_P8

rm fifo/gul_S1_summary_P8
rm fifo/gul_S1_summaryeltcalc_P8
rm fifo/gul_S1_eltcalc_P8
rm fifo/gul_S1_summarysummarycalc_P8
rm fifo/gul_S1_summarycalc_P8
rm fifo/gul_S1_summarypltcalc_P8
rm fifo/gul_S1_pltcalc_P8
rm fifo/gul_S1_summaryaalcalc_P8
rm fifo/gul_S2_summary_P8
rm fifo/gul_S2_summaryeltcalc_P8
rm fifo/gul_S2_eltcalc_P8
rm fifo/gul_S2_summarysummarycalc_P8
rm fifo/gul_S2_summarycalc_P8
rm fifo/gul_S2_summarypltcalc_P8
rm fifo/gul_S2_pltcalc_P8
rm fifo/gul_S2_summaryaalcalc_P8

rm fifo/gul_P9

rm fifo/gul_S1_summary_P9
rm fifo/gul_S1_summaryeltcalc_P9
rm fifo/gul_S1_eltcalc_P9
rm fifo/gul_S1_summarysummarycalc_P9
rm fifo/gul_S1_summarycalc_P9
rm fifo/gul_S1_summarypltcalc_P9
rm fifo/gul_S1_pltcalc_P9
rm fifo/gul_S1_summaryaalcalc_P9
rm fifo/gul_S2_summary_P9
rm fifo/gul_S2_summaryeltcalc_P9
rm fifo/gul_S2_eltcalc_P9
rm fifo/gul_S2_summarysummarycalc_P9
rm fifo/gul_S2_summarycalc_P9
rm fifo/gul_S2_summarypltcalc_P9
rm fifo/gul_S2_pltcalc_P9
rm fifo/gul_S2_summaryaalcalc_P9

rm fifo/gul_P10

rm fifo/gul_S1_summary_P10
rm fifo/gul_S1_summaryeltcalc_P10
rm fifo/gul_S1_eltcalc_P10
rm fifo/gul_S1_summarysummarycalc_P10
rm fifo/gul_S1_summarycalc_P10
rm fifo/gul_S1_summarypltcalc_P10
rm fifo/gul_S1_pltcalc_P10
rm fifo/gul_S1_summaryaalcalc_P10
rm fifo/gul_S2_summary_P10
rm fifo/gul_S2_summaryeltcalc_P10
rm fifo/gul_S2_eltcalc_P10
rm fifo/gul_S2_summarysummarycalc_P10
rm fifo/gul_S2_summarycalc_P10
rm fifo/gul_S2_summarypltcalc_P10
rm fifo/gul_S2_pltcalc_P10
rm fifo/gul_S2_summaryaalcalc_P10

rm -rf work/kat
rm work/gul_S1_summaryleccalc/*
rmdir work/gul_S1_summaryleccalc
rm work/gul_S1_aalcalc/*
rmdir work/gul_S1_aalcalc
rm work/gul_S2_summaryleccalc/*
rmdir work/gul_S2_summaryleccalc
rm work/gul_S2_aalcalc/*
rmdir work/gul_S2_aalcalc

rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryeltcalc_P1
rm fifo/il_S1_eltcalc_P1
rm fifo/il_S1_summarysummarycalc_P1
rm fifo/il_S1_summarycalc_P1
rm fifo/il_S1_summarypltcalc_P1
rm fifo/il_S1_pltcalc_P1
rm fifo/il_S1_summaryaalcalc_P1
rm fifo/il_S2_summary_P1
rm fifo/il_S2_summaryeltcalc_P1
rm fifo/il_S2_eltcalc_P1
rm fifo/il_S2_summarysummarycalc_P1
rm fifo/il_S2_summarycalc_P1
rm fifo/il_S2_summarypltcalc_P1
rm fifo/il_S2_pltcalc_P1
rm fifo/il_S2_summaryaalcalc_P1

rm fifo/il_P2

rm fifo/il_S1_summary_P2
rm fifo/il_S1_summaryeltcalc_P2
rm fifo/il_S1_eltcalc_P2
rm fifo/il_S1_summarysummarycalc_P2
rm fifo/il_S1_summarycalc_P2
rm fifo/il_S1_summarypltcalc_P2
rm fifo/il_S1_pltcalc_P2
rm fifo/il_S1_summaryaalcalc_P2
rm fifo/il_S2_summary_P2
rm fifo/il_S2_summaryeltcalc_P2
rm fifo/il_S2_eltcalc_P2
rm fifo/il_S2_summarysummarycalc_P2
rm fifo/il_S2_summarycalc_P2
rm fifo/il_S2_summarypltcalc_P2
rm fifo/il_S2_pltcalc_P2
rm fifo/il_S2_summaryaalcalc_P2

rm fifo/il_P3

rm fifo/il_S1_summary_P3
rm fifo/il_S1_summaryeltcalc_P3
rm fifo/il_S1_eltcalc_P3
rm fifo/il_S1_summarysummarycalc_P3
rm fifo/il_S1_summarycalc_P3
rm fifo/il_S1_summarypltcalc_P3
rm fifo/il_S1_pltcalc_P3
rm fifo/il_S1_summaryaalcalc_P3
rm fifo/il_S2_summary_P3
rm fifo/il_S2_summaryeltcalc_P3
rm fifo/il_S2_eltcalc_P3
rm fifo/il_S2_summarysummarycalc_P3
rm fifo/il_S2_summarycalc_P3
rm fifo/il_S2_summarypltcalc_P3
rm fifo/il_S2_pltcalc_P3
rm fifo/il_S2_summaryaalcalc_P3

rm fifo/il_P4

rm fifo/il_S1_summary_P4
rm fifo/il_S1_summaryeltcalc_P4
rm fifo/il_S1_eltcalc_P4
rm fifo/il_S1_summarysummarycalc_P4
rm fifo/il_S1_summarycalc_P4
rm fifo/il_S1_summarypltcalc_P4
rm fifo/il_S1_pltcalc_P4
rm fifo/il_S1_summaryaalcalc_P4
rm fifo/il_S2_summary_P4
rm fifo/il_S2_summaryeltcalc_P4
rm fifo/il_S2_eltcalc_P4
rm fifo/il_S2_summarysummarycalc_P4
rm fifo/il_S2_summarycalc_P4
rm fifo/il_S2_summarypltcalc_P4
rm fifo/il_S2_pltcalc_P4
rm fifo/il_S2_summaryaalcalc_P4

rm fifo/il_P5

rm fifo/il_S1_summary_P5
rm fifo/il_S1_summaryeltcalc_P5
rm fifo/il_S1_eltcalc_P5
rm fifo/il_S1_summarysummarycalc_P5
rm fifo/il_S1_summarycalc_P5
rm fifo/il_S1_summarypltcalc_P5
rm fifo/il_S1_pltcalc_P5
rm fifo/il_S1_summaryaalcalc_P5
rm fifo/il_S2_summary_P5
rm fifo/il_S2_summaryeltcalc_P5
rm fifo/il_S2_eltcalc_P5
rm fifo/il_S2_summarysummarycalc_P5
rm fifo/il_S2_summarycalc_P5
rm fifo/il_S2_summarypltcalc_P5
rm fifo/il_S2_pltcalc_P5
rm fifo/il_S2_summaryaalcalc_P5

rm fifo/il_P6

rm fifo/il_S1_summary_P6
rm fifo/il_S1_summaryeltcalc_P6
rm fifo/il_S1_eltcalc_P6
rm fifo/il_S1_summarysummarycalc_P6
rm fifo/il_S1_summarycalc_P6
rm fifo/il_S1_summarypltcalc_P6
rm fifo/il_S1_pltcalc_P6
rm fifo/il_S1_summaryaalcalc_P6
rm fifo/il_S2_summary_P6
rm fifo/il_S2_summaryeltcalc_P6
rm fifo/il_S2_eltcalc_P6
rm fifo/il_S2_summarysummarycalc_P6
rm fifo/il_S2_summarycalc_P6
rm fifo/il_S2_summarypltcalc_P6
rm fifo/il_S2_pltcalc_P6
rm fifo/il_S2_summaryaalcalc_P6

rm fifo/il_P7

rm fifo/il_S1_summary_P7
rm fifo/il_S1_summaryeltcalc_P7
rm fifo/il_S1_eltcalc_P7
rm fifo/il_S1_summarysummarycalc_P7
rm fifo/il_S1_summarycalc_P7
rm fifo/il_S1_summarypltcalc_P7
rm fifo/il_S1_pltcalc_P7
rm fifo/il_S1_summaryaalcalc_P7
rm fifo/il_S2_summary_P7
rm fifo/il_S2_summaryeltcalc_P7
rm fifo/il_S2_eltcalc_P7
rm fifo/il_S2_summarysummarycalc_P7
rm fifo/il_S2_summarycalc_P7
rm fifo/il_S2_summarypltcalc_P7
rm fifo/il_S2_pltcalc_P7
rm fifo/il_S2_summaryaalcalc_P7

rm fifo/il_P8

rm fifo/il_S1_summary_P8
rm fifo/il_S1_summaryeltcalc_P8
rm fifo/il_S1_eltcalc_P8
rm fifo/il_S1_summarysummarycalc_P8
rm fifo/il_S1_summarycalc_P8
rm fifo/il_S1_summarypltcalc_P8
rm fifo/il_S1_pltcalc_P8
rm fifo/il_S1_summaryaalcalc_P8
rm fifo/il_S2_summary_P8
rm fifo/il_S2_summaryeltcalc_P8
rm fifo/il_S2_eltcalc_P8
rm fifo/il_S2_summarysummarycalc_P8
rm fifo/il_S2_summarycalc_P8
rm fifo/il_S2_summarypltcalc_P8
rm fifo/il_S2_pltcalc_P8
rm fifo/il_S2_summaryaalcalc_P8

rm fifo/il_P9

rm fifo/il_S1_summary_P9
rm fifo/il_S1_summaryeltcalc_P9
rm fifo/il_S1_eltcalc_P9
rm fifo/il_S1_summarysummarycalc_P9
rm fifo/il_S1_summarycalc_P9
rm fifo/il_S1_summarypltcalc_P9
rm fifo/il_S1_pltcalc_P9
rm fifo/il_S1_summaryaalcalc_P9
rm fifo/il_S2_summary_P9
rm fifo/il_S2_summaryeltcalc_P9
rm fifo/il_S2_eltcalc_P9
rm fifo/il_S2_summarysummarycalc_P9
rm fifo/il_S2_summarycalc_P9
rm fifo/il_S2_summarypltcalc_P9
rm fifo/il_S2_pltcalc_P9
rm fifo/il_S2_summaryaalcalc_P9

rm fifo/il_P10

rm fifo/il_S1_summary_P10
rm fifo/il_S1_summaryeltcalc_P10
rm fifo/il_S1_eltcalc_P10
rm fifo/il_S1_summarysummarycalc_P10
rm fifo/il_S1_summarycalc_P10
rm fifo/il_S1_summarypltcalc_P10
rm fifo/il_S1_pltcalc_P10
rm fifo/il_S1_summaryaalcalc_P10
rm fifo/il_S2_summary_P10
rm fifo/il_S2_summaryeltcalc_P10
rm fifo/il_S2_eltcalc_P10
rm fifo/il_S2_summarysummarycalc_P10
rm fifo/il_S2_summarycalc_P10
rm fifo/il_S2_summarypltcalc_P10
rm fifo/il_S2_pltcalc_P10
rm fifo/il_S2_summaryaalcalc_P10

rm -rf work/kat
rm work/il_S1_summaryleccalc/*
rmdir work/il_S1_summaryleccalc
rm work/il_S1_aalcalc/*
rmdir work/il_S1_aalcalc
rm work/il_S2_summaryleccalc/*
rmdir work/il_S2_summaryleccalc
rm work/il_S2_aalcalc/*
rmdir work/il_S2_aalcalc
