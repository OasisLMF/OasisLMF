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
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S2_summaryleccalc
mkdir work/full_correlation/gul_S2_summaryaalcalc

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
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_summary_P1.idx
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summary_P2.idx
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3
mkfifo fifo/gul_S2_summary_P3
mkfifo fifo/gul_S2_summary_P3.idx
mkfifo fifo/gul_S2_eltcalc_P3
mkfifo fifo/gul_S2_summarycalc_P3
mkfifo fifo/gul_S2_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4
mkfifo fifo/gul_S2_summary_P4
mkfifo fifo/gul_S2_summary_P4.idx
mkfifo fifo/gul_S2_eltcalc_P4
mkfifo fifo/gul_S2_summarycalc_P4
mkfifo fifo/gul_S2_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5
mkfifo fifo/gul_S2_summary_P5
mkfifo fifo/gul_S2_summary_P5.idx
mkfifo fifo/gul_S2_eltcalc_P5
mkfifo fifo/gul_S2_summarycalc_P5
mkfifo fifo/gul_S2_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6
mkfifo fifo/gul_S2_summary_P6
mkfifo fifo/gul_S2_summary_P6.idx
mkfifo fifo/gul_S2_eltcalc_P6
mkfifo fifo/gul_S2_summarycalc_P6
mkfifo fifo/gul_S2_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7
mkfifo fifo/gul_S2_summary_P7
mkfifo fifo/gul_S2_summary_P7.idx
mkfifo fifo/gul_S2_eltcalc_P7
mkfifo fifo/gul_S2_summarycalc_P7
mkfifo fifo/gul_S2_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8
mkfifo fifo/gul_S2_summary_P8
mkfifo fifo/gul_S2_summary_P8.idx
mkfifo fifo/gul_S2_eltcalc_P8
mkfifo fifo/gul_S2_summarycalc_P8
mkfifo fifo/gul_S2_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9
mkfifo fifo/gul_S2_summary_P9
mkfifo fifo/gul_S2_summary_P9.idx
mkfifo fifo/gul_S2_eltcalc_P9
mkfifo fifo/gul_S2_summarycalc_P9
mkfifo fifo/gul_S2_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10
mkfifo fifo/gul_S2_summary_P10
mkfifo fifo/gul_S2_summary_P10.idx
mkfifo fifo/gul_S2_eltcalc_P10
mkfifo fifo/gul_S2_summarycalc_P10
mkfifo fifo/gul_S2_pltcalc_P10

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

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_summary_P1.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo fifo/full_correlation/gul_S1_pltcalc_P1
mkfifo fifo/full_correlation/gul_S2_summary_P1
mkfifo fifo/full_correlation/gul_S2_summary_P1.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P1
mkfifo fifo/full_correlation/gul_S2_summarycalc_P1
mkfifo fifo/full_correlation/gul_S2_pltcalc_P1

mkfifo fifo/full_correlation/gul_S1_summary_P2
mkfifo fifo/full_correlation/gul_S1_summary_P2.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo fifo/full_correlation/gul_S1_pltcalc_P2
mkfifo fifo/full_correlation/gul_S2_summary_P2
mkfifo fifo/full_correlation/gul_S2_summary_P2.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P2
mkfifo fifo/full_correlation/gul_S2_summarycalc_P2
mkfifo fifo/full_correlation/gul_S2_pltcalc_P2

mkfifo fifo/full_correlation/gul_S1_summary_P3
mkfifo fifo/full_correlation/gul_S1_summary_P3.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P3
mkfifo fifo/full_correlation/gul_S1_summarycalc_P3
mkfifo fifo/full_correlation/gul_S1_pltcalc_P3
mkfifo fifo/full_correlation/gul_S2_summary_P3
mkfifo fifo/full_correlation/gul_S2_summary_P3.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P3
mkfifo fifo/full_correlation/gul_S2_summarycalc_P3
mkfifo fifo/full_correlation/gul_S2_pltcalc_P3

mkfifo fifo/full_correlation/gul_S1_summary_P4
mkfifo fifo/full_correlation/gul_S1_summary_P4.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P4
mkfifo fifo/full_correlation/gul_S1_summarycalc_P4
mkfifo fifo/full_correlation/gul_S1_pltcalc_P4
mkfifo fifo/full_correlation/gul_S2_summary_P4
mkfifo fifo/full_correlation/gul_S2_summary_P4.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P4
mkfifo fifo/full_correlation/gul_S2_summarycalc_P4
mkfifo fifo/full_correlation/gul_S2_pltcalc_P4

mkfifo fifo/full_correlation/gul_S1_summary_P5
mkfifo fifo/full_correlation/gul_S1_summary_P5.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P5
mkfifo fifo/full_correlation/gul_S1_summarycalc_P5
mkfifo fifo/full_correlation/gul_S1_pltcalc_P5
mkfifo fifo/full_correlation/gul_S2_summary_P5
mkfifo fifo/full_correlation/gul_S2_summary_P5.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P5
mkfifo fifo/full_correlation/gul_S2_summarycalc_P5
mkfifo fifo/full_correlation/gul_S2_pltcalc_P5

mkfifo fifo/full_correlation/gul_S1_summary_P6
mkfifo fifo/full_correlation/gul_S1_summary_P6.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P6
mkfifo fifo/full_correlation/gul_S1_summarycalc_P6
mkfifo fifo/full_correlation/gul_S1_pltcalc_P6
mkfifo fifo/full_correlation/gul_S2_summary_P6
mkfifo fifo/full_correlation/gul_S2_summary_P6.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P6
mkfifo fifo/full_correlation/gul_S2_summarycalc_P6
mkfifo fifo/full_correlation/gul_S2_pltcalc_P6

mkfifo fifo/full_correlation/gul_S1_summary_P7
mkfifo fifo/full_correlation/gul_S1_summary_P7.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P7
mkfifo fifo/full_correlation/gul_S1_summarycalc_P7
mkfifo fifo/full_correlation/gul_S1_pltcalc_P7
mkfifo fifo/full_correlation/gul_S2_summary_P7
mkfifo fifo/full_correlation/gul_S2_summary_P7.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P7
mkfifo fifo/full_correlation/gul_S2_summarycalc_P7
mkfifo fifo/full_correlation/gul_S2_pltcalc_P7

mkfifo fifo/full_correlation/gul_S1_summary_P8
mkfifo fifo/full_correlation/gul_S1_summary_P8.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P8
mkfifo fifo/full_correlation/gul_S1_summarycalc_P8
mkfifo fifo/full_correlation/gul_S1_pltcalc_P8
mkfifo fifo/full_correlation/gul_S2_summary_P8
mkfifo fifo/full_correlation/gul_S2_summary_P8.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P8
mkfifo fifo/full_correlation/gul_S2_summarycalc_P8
mkfifo fifo/full_correlation/gul_S2_pltcalc_P8

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S1_summary_P9.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P9
mkfifo fifo/full_correlation/gul_S1_summarycalc_P9
mkfifo fifo/full_correlation/gul_S1_pltcalc_P9
mkfifo fifo/full_correlation/gul_S2_summary_P9
mkfifo fifo/full_correlation/gul_S2_summary_P9.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P9
mkfifo fifo/full_correlation/gul_S2_summarycalc_P9
mkfifo fifo/full_correlation/gul_S2_pltcalc_P9

mkfifo fifo/full_correlation/gul_S1_summary_P10
mkfifo fifo/full_correlation/gul_S1_summary_P10.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P10
mkfifo fifo/full_correlation/gul_S1_summarycalc_P10
mkfifo fifo/full_correlation/gul_S1_pltcalc_P10
mkfifo fifo/full_correlation/gul_S2_summary_P10
mkfifo fifo/full_correlation/gul_S2_summary_P10.idx
mkfifo fifo/full_correlation/gul_S2_eltcalc_P10
mkfifo fifo/full_correlation/gul_S2_summarycalc_P10
mkfifo fifo/full_correlation/gul_S2_pltcalc_P10



# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid3=$!
eltcalc < fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid5=$!
pltcalc < fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid6=$!
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid8=$!
pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid9=$!
eltcalc -s < fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid11=$!
pltcalc -H < fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid12=$!
eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid13=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid14=$!
pltcalc -H < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid15=$!
eltcalc -s < fifo/gul_S2_eltcalc_P3 > work/kat/gul_S2_eltcalc_P3 & pid16=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P3 > work/kat/gul_S2_summarycalc_P3 & pid17=$!
pltcalc -H < fifo/gul_S2_pltcalc_P3 > work/kat/gul_S2_pltcalc_P3 & pid18=$!
eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid19=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid20=$!
pltcalc -H < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid21=$!
eltcalc -s < fifo/gul_S2_eltcalc_P4 > work/kat/gul_S2_eltcalc_P4 & pid22=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P4 > work/kat/gul_S2_summarycalc_P4 & pid23=$!
pltcalc -H < fifo/gul_S2_pltcalc_P4 > work/kat/gul_S2_pltcalc_P4 & pid24=$!
eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid25=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid26=$!
pltcalc -H < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid27=$!
eltcalc -s < fifo/gul_S2_eltcalc_P5 > work/kat/gul_S2_eltcalc_P5 & pid28=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P5 > work/kat/gul_S2_summarycalc_P5 & pid29=$!
pltcalc -H < fifo/gul_S2_pltcalc_P5 > work/kat/gul_S2_pltcalc_P5 & pid30=$!
eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 & pid31=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid32=$!
pltcalc -H < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid33=$!
eltcalc -s < fifo/gul_S2_eltcalc_P6 > work/kat/gul_S2_eltcalc_P6 & pid34=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P6 > work/kat/gul_S2_summarycalc_P6 & pid35=$!
pltcalc -H < fifo/gul_S2_pltcalc_P6 > work/kat/gul_S2_pltcalc_P6 & pid36=$!
eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid37=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid38=$!
pltcalc -H < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid39=$!
eltcalc -s < fifo/gul_S2_eltcalc_P7 > work/kat/gul_S2_eltcalc_P7 & pid40=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P7 > work/kat/gul_S2_summarycalc_P7 & pid41=$!
pltcalc -H < fifo/gul_S2_pltcalc_P7 > work/kat/gul_S2_pltcalc_P7 & pid42=$!
eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 & pid43=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 & pid44=$!
pltcalc -H < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 & pid45=$!
eltcalc -s < fifo/gul_S2_eltcalc_P8 > work/kat/gul_S2_eltcalc_P8 & pid46=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P8 > work/kat/gul_S2_summarycalc_P8 & pid47=$!
pltcalc -H < fifo/gul_S2_pltcalc_P8 > work/kat/gul_S2_pltcalc_P8 & pid48=$!
eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid49=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid50=$!
pltcalc -H < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 & pid51=$!
eltcalc -s < fifo/gul_S2_eltcalc_P9 > work/kat/gul_S2_eltcalc_P9 & pid52=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P9 > work/kat/gul_S2_summarycalc_P9 & pid53=$!
pltcalc -H < fifo/gul_S2_pltcalc_P9 > work/kat/gul_S2_pltcalc_P9 & pid54=$!
eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid55=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid56=$!
pltcalc -H < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid57=$!
eltcalc -s < fifo/gul_S2_eltcalc_P10 > work/kat/gul_S2_eltcalc_P10 & pid58=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P10 > work/kat/gul_S2_summarycalc_P10 & pid59=$!
pltcalc -H < fifo/gul_S2_pltcalc_P10 > work/kat/gul_S2_pltcalc_P10 & pid60=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid61=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid62=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_eltcalc_P1 fifo/gul_S2_summarycalc_P1 fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid63=$!
tee < fifo/gul_S2_summary_P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid64=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid65=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid66=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_eltcalc_P2 fifo/gul_S2_summarycalc_P2 fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid67=$!
tee < fifo/gul_S2_summary_P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid68=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid69=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid70=$!
tee < fifo/gul_S2_summary_P3 fifo/gul_S2_eltcalc_P3 fifo/gul_S2_summarycalc_P3 fifo/gul_S2_pltcalc_P3 work/gul_S2_summaryaalcalc/P3.bin work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid71=$!
tee < fifo/gul_S2_summary_P3.idx work/gul_S2_summaryleccalc/P3.idx > /dev/null & pid72=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid73=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid74=$!
tee < fifo/gul_S2_summary_P4 fifo/gul_S2_eltcalc_P4 fifo/gul_S2_summarycalc_P4 fifo/gul_S2_pltcalc_P4 work/gul_S2_summaryaalcalc/P4.bin work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid75=$!
tee < fifo/gul_S2_summary_P4.idx work/gul_S2_summaryleccalc/P4.idx > /dev/null & pid76=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid77=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid78=$!
tee < fifo/gul_S2_summary_P5 fifo/gul_S2_eltcalc_P5 fifo/gul_S2_summarycalc_P5 fifo/gul_S2_pltcalc_P5 work/gul_S2_summaryaalcalc/P5.bin work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid79=$!
tee < fifo/gul_S2_summary_P5.idx work/gul_S2_summaryleccalc/P5.idx > /dev/null & pid80=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid81=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid82=$!
tee < fifo/gul_S2_summary_P6 fifo/gul_S2_eltcalc_P6 fifo/gul_S2_summarycalc_P6 fifo/gul_S2_pltcalc_P6 work/gul_S2_summaryaalcalc/P6.bin work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid83=$!
tee < fifo/gul_S2_summary_P6.idx work/gul_S2_summaryleccalc/P6.idx > /dev/null & pid84=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid85=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid86=$!
tee < fifo/gul_S2_summary_P7 fifo/gul_S2_eltcalc_P7 fifo/gul_S2_summarycalc_P7 fifo/gul_S2_pltcalc_P7 work/gul_S2_summaryaalcalc/P7.bin work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid87=$!
tee < fifo/gul_S2_summary_P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid88=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid89=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid90=$!
tee < fifo/gul_S2_summary_P8 fifo/gul_S2_eltcalc_P8 fifo/gul_S2_summarycalc_P8 fifo/gul_S2_pltcalc_P8 work/gul_S2_summaryaalcalc/P8.bin work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid91=$!
tee < fifo/gul_S2_summary_P8.idx work/gul_S2_summaryleccalc/P8.idx > /dev/null & pid92=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid93=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid94=$!
tee < fifo/gul_S2_summary_P9 fifo/gul_S2_eltcalc_P9 fifo/gul_S2_summarycalc_P9 fifo/gul_S2_pltcalc_P9 work/gul_S2_summaryaalcalc/P9.bin work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid95=$!
tee < fifo/gul_S2_summary_P9.idx work/gul_S2_summaryleccalc/P9.idx > /dev/null & pid96=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid97=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid98=$!
tee < fifo/gul_S2_summary_P10 fifo/gul_S2_eltcalc_P10 fifo/gul_S2_summarycalc_P10 fifo/gul_S2_pltcalc_P10 work/gul_S2_summaryaalcalc/P10.bin work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid99=$!
tee < fifo/gul_S2_summary_P10.idx work/gul_S2_summaryleccalc/P10.idx > /dev/null & pid100=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 &

# --- Do ground up loss computes ---

eltcalc < fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid101=$!
summarycalctocsv < fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid102=$!
pltcalc < fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid103=$!
eltcalc < fifo/full_correlation/gul_S2_eltcalc_P1 > work/full_correlation/kat/gul_S2_eltcalc_P1 & pid104=$!
summarycalctocsv < fifo/full_correlation/gul_S2_summarycalc_P1 > work/full_correlation/kat/gul_S2_summarycalc_P1 & pid105=$!
pltcalc < fifo/full_correlation/gul_S2_pltcalc_P1 > work/full_correlation/kat/gul_S2_pltcalc_P1 & pid106=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid107=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid108=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid109=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P2 > work/full_correlation/kat/gul_S2_eltcalc_P2 & pid110=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P2 > work/full_correlation/kat/gul_S2_summarycalc_P2 & pid111=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P2 > work/full_correlation/kat/gul_S2_pltcalc_P2 & pid112=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P3 > work/full_correlation/kat/gul_S1_eltcalc_P3 & pid113=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P3 > work/full_correlation/kat/gul_S1_summarycalc_P3 & pid114=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P3 > work/full_correlation/kat/gul_S1_pltcalc_P3 & pid115=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P3 > work/full_correlation/kat/gul_S2_eltcalc_P3 & pid116=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P3 > work/full_correlation/kat/gul_S2_summarycalc_P3 & pid117=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P3 > work/full_correlation/kat/gul_S2_pltcalc_P3 & pid118=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid119=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 & pid120=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 & pid121=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P4 > work/full_correlation/kat/gul_S2_eltcalc_P4 & pid122=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P4 > work/full_correlation/kat/gul_S2_summarycalc_P4 & pid123=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P4 > work/full_correlation/kat/gul_S2_pltcalc_P4 & pid124=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P5 > work/full_correlation/kat/gul_S1_eltcalc_P5 & pid125=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P5 > work/full_correlation/kat/gul_S1_summarycalc_P5 & pid126=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P5 > work/full_correlation/kat/gul_S1_pltcalc_P5 & pid127=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P5 > work/full_correlation/kat/gul_S2_eltcalc_P5 & pid128=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P5 > work/full_correlation/kat/gul_S2_summarycalc_P5 & pid129=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P5 > work/full_correlation/kat/gul_S2_pltcalc_P5 & pid130=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P6 > work/full_correlation/kat/gul_S1_eltcalc_P6 & pid131=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P6 > work/full_correlation/kat/gul_S1_summarycalc_P6 & pid132=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P6 > work/full_correlation/kat/gul_S1_pltcalc_P6 & pid133=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P6 > work/full_correlation/kat/gul_S2_eltcalc_P6 & pid134=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P6 > work/full_correlation/kat/gul_S2_summarycalc_P6 & pid135=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P6 > work/full_correlation/kat/gul_S2_pltcalc_P6 & pid136=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 & pid137=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 & pid138=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 & pid139=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P7 > work/full_correlation/kat/gul_S2_eltcalc_P7 & pid140=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P7 > work/full_correlation/kat/gul_S2_summarycalc_P7 & pid141=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P7 > work/full_correlation/kat/gul_S2_pltcalc_P7 & pid142=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P8 > work/full_correlation/kat/gul_S1_eltcalc_P8 & pid143=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P8 > work/full_correlation/kat/gul_S1_summarycalc_P8 & pid144=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P8 > work/full_correlation/kat/gul_S1_pltcalc_P8 & pid145=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P8 > work/full_correlation/kat/gul_S2_eltcalc_P8 & pid146=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P8 > work/full_correlation/kat/gul_S2_summarycalc_P8 & pid147=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P8 > work/full_correlation/kat/gul_S2_pltcalc_P8 & pid148=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 & pid149=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P9 > work/full_correlation/kat/gul_S1_summarycalc_P9 & pid150=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P9 > work/full_correlation/kat/gul_S1_pltcalc_P9 & pid151=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P9 > work/full_correlation/kat/gul_S2_eltcalc_P9 & pid152=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P9 > work/full_correlation/kat/gul_S2_summarycalc_P9 & pid153=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P9 > work/full_correlation/kat/gul_S2_pltcalc_P9 & pid154=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P10 > work/full_correlation/kat/gul_S1_eltcalc_P10 & pid155=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 & pid156=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P10 > work/full_correlation/kat/gul_S1_pltcalc_P10 & pid157=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P10 > work/full_correlation/kat/gul_S2_eltcalc_P10 & pid158=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P10 > work/full_correlation/kat/gul_S2_summarycalc_P10 & pid159=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P10 > work/full_correlation/kat/gul_S2_pltcalc_P10 & pid160=$!


tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_eltcalc_P1 fifo/full_correlation/gul_S1_summarycalc_P1 fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid161=$!
tee < fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid162=$!
tee < fifo/full_correlation/gul_S2_summary_P1 fifo/full_correlation/gul_S2_eltcalc_P1 fifo/full_correlation/gul_S2_summarycalc_P1 fifo/full_correlation/gul_S2_pltcalc_P1 work/full_correlation/gul_S2_summaryaalcalc/P1.bin work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid163=$!
tee < fifo/full_correlation/gul_S2_summary_P1.idx work/full_correlation/gul_S2_summaryleccalc/P1.idx > /dev/null & pid164=$!
tee < fifo/full_correlation/gul_S1_summary_P2 fifo/full_correlation/gul_S1_eltcalc_P2 fifo/full_correlation/gul_S1_summarycalc_P2 fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid165=$!
tee < fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryleccalc/P2.idx > /dev/null & pid166=$!
tee < fifo/full_correlation/gul_S2_summary_P2 fifo/full_correlation/gul_S2_eltcalc_P2 fifo/full_correlation/gul_S2_summarycalc_P2 fifo/full_correlation/gul_S2_pltcalc_P2 work/full_correlation/gul_S2_summaryaalcalc/P2.bin work/full_correlation/gul_S2_summaryleccalc/P2.bin > /dev/null & pid167=$!
tee < fifo/full_correlation/gul_S2_summary_P2.idx work/full_correlation/gul_S2_summaryleccalc/P2.idx > /dev/null & pid168=$!
tee < fifo/full_correlation/gul_S1_summary_P3 fifo/full_correlation/gul_S1_eltcalc_P3 fifo/full_correlation/gul_S1_summarycalc_P3 fifo/full_correlation/gul_S1_pltcalc_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid169=$!
tee < fifo/full_correlation/gul_S1_summary_P3.idx work/full_correlation/gul_S1_summaryleccalc/P3.idx > /dev/null & pid170=$!
tee < fifo/full_correlation/gul_S2_summary_P3 fifo/full_correlation/gul_S2_eltcalc_P3 fifo/full_correlation/gul_S2_summarycalc_P3 fifo/full_correlation/gul_S2_pltcalc_P3 work/full_correlation/gul_S2_summaryaalcalc/P3.bin work/full_correlation/gul_S2_summaryleccalc/P3.bin > /dev/null & pid171=$!
tee < fifo/full_correlation/gul_S2_summary_P3.idx work/full_correlation/gul_S2_summaryleccalc/P3.idx > /dev/null & pid172=$!
tee < fifo/full_correlation/gul_S1_summary_P4 fifo/full_correlation/gul_S1_eltcalc_P4 fifo/full_correlation/gul_S1_summarycalc_P4 fifo/full_correlation/gul_S1_pltcalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid173=$!
tee < fifo/full_correlation/gul_S1_summary_P4.idx work/full_correlation/gul_S1_summaryleccalc/P4.idx > /dev/null & pid174=$!
tee < fifo/full_correlation/gul_S2_summary_P4 fifo/full_correlation/gul_S2_eltcalc_P4 fifo/full_correlation/gul_S2_summarycalc_P4 fifo/full_correlation/gul_S2_pltcalc_P4 work/full_correlation/gul_S2_summaryaalcalc/P4.bin work/full_correlation/gul_S2_summaryleccalc/P4.bin > /dev/null & pid175=$!
tee < fifo/full_correlation/gul_S2_summary_P4.idx work/full_correlation/gul_S2_summaryleccalc/P4.idx > /dev/null & pid176=$!
tee < fifo/full_correlation/gul_S1_summary_P5 fifo/full_correlation/gul_S1_eltcalc_P5 fifo/full_correlation/gul_S1_summarycalc_P5 fifo/full_correlation/gul_S1_pltcalc_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid177=$!
tee < fifo/full_correlation/gul_S1_summary_P5.idx work/full_correlation/gul_S1_summaryleccalc/P5.idx > /dev/null & pid178=$!
tee < fifo/full_correlation/gul_S2_summary_P5 fifo/full_correlation/gul_S2_eltcalc_P5 fifo/full_correlation/gul_S2_summarycalc_P5 fifo/full_correlation/gul_S2_pltcalc_P5 work/full_correlation/gul_S2_summaryaalcalc/P5.bin work/full_correlation/gul_S2_summaryleccalc/P5.bin > /dev/null & pid179=$!
tee < fifo/full_correlation/gul_S2_summary_P5.idx work/full_correlation/gul_S2_summaryleccalc/P5.idx > /dev/null & pid180=$!
tee < fifo/full_correlation/gul_S1_summary_P6 fifo/full_correlation/gul_S1_eltcalc_P6 fifo/full_correlation/gul_S1_summarycalc_P6 fifo/full_correlation/gul_S1_pltcalc_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid181=$!
tee < fifo/full_correlation/gul_S1_summary_P6.idx work/full_correlation/gul_S1_summaryleccalc/P6.idx > /dev/null & pid182=$!
tee < fifo/full_correlation/gul_S2_summary_P6 fifo/full_correlation/gul_S2_eltcalc_P6 fifo/full_correlation/gul_S2_summarycalc_P6 fifo/full_correlation/gul_S2_pltcalc_P6 work/full_correlation/gul_S2_summaryaalcalc/P6.bin work/full_correlation/gul_S2_summaryleccalc/P6.bin > /dev/null & pid183=$!
tee < fifo/full_correlation/gul_S2_summary_P6.idx work/full_correlation/gul_S2_summaryleccalc/P6.idx > /dev/null & pid184=$!
tee < fifo/full_correlation/gul_S1_summary_P7 fifo/full_correlation/gul_S1_eltcalc_P7 fifo/full_correlation/gul_S1_summarycalc_P7 fifo/full_correlation/gul_S1_pltcalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid185=$!
tee < fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryleccalc/P7.idx > /dev/null & pid186=$!
tee < fifo/full_correlation/gul_S2_summary_P7 fifo/full_correlation/gul_S2_eltcalc_P7 fifo/full_correlation/gul_S2_summarycalc_P7 fifo/full_correlation/gul_S2_pltcalc_P7 work/full_correlation/gul_S2_summaryaalcalc/P7.bin work/full_correlation/gul_S2_summaryleccalc/P7.bin > /dev/null & pid187=$!
tee < fifo/full_correlation/gul_S2_summary_P7.idx work/full_correlation/gul_S2_summaryleccalc/P7.idx > /dev/null & pid188=$!
tee < fifo/full_correlation/gul_S1_summary_P8 fifo/full_correlation/gul_S1_eltcalc_P8 fifo/full_correlation/gul_S1_summarycalc_P8 fifo/full_correlation/gul_S1_pltcalc_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid189=$!
tee < fifo/full_correlation/gul_S1_summary_P8.idx work/full_correlation/gul_S1_summaryleccalc/P8.idx > /dev/null & pid190=$!
tee < fifo/full_correlation/gul_S2_summary_P8 fifo/full_correlation/gul_S2_eltcalc_P8 fifo/full_correlation/gul_S2_summarycalc_P8 fifo/full_correlation/gul_S2_pltcalc_P8 work/full_correlation/gul_S2_summaryaalcalc/P8.bin work/full_correlation/gul_S2_summaryleccalc/P8.bin > /dev/null & pid191=$!
tee < fifo/full_correlation/gul_S2_summary_P8.idx work/full_correlation/gul_S2_summaryleccalc/P8.idx > /dev/null & pid192=$!
tee < fifo/full_correlation/gul_S1_summary_P9 fifo/full_correlation/gul_S1_eltcalc_P9 fifo/full_correlation/gul_S1_summarycalc_P9 fifo/full_correlation/gul_S1_pltcalc_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid193=$!
tee < fifo/full_correlation/gul_S1_summary_P9.idx work/full_correlation/gul_S1_summaryleccalc/P9.idx > /dev/null & pid194=$!
tee < fifo/full_correlation/gul_S2_summary_P9 fifo/full_correlation/gul_S2_eltcalc_P9 fifo/full_correlation/gul_S2_summarycalc_P9 fifo/full_correlation/gul_S2_pltcalc_P9 work/full_correlation/gul_S2_summaryaalcalc/P9.bin work/full_correlation/gul_S2_summaryleccalc/P9.bin > /dev/null & pid195=$!
tee < fifo/full_correlation/gul_S2_summary_P9.idx work/full_correlation/gul_S2_summaryleccalc/P9.idx > /dev/null & pid196=$!
tee < fifo/full_correlation/gul_S1_summary_P10 fifo/full_correlation/gul_S1_eltcalc_P10 fifo/full_correlation/gul_S1_summarycalc_P10 fifo/full_correlation/gul_S1_pltcalc_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid197=$!
tee < fifo/full_correlation/gul_S1_summary_P10.idx work/full_correlation/gul_S1_summaryleccalc/P10.idx > /dev/null & pid198=$!
tee < fifo/full_correlation/gul_S2_summary_P10 fifo/full_correlation/gul_S2_eltcalc_P10 fifo/full_correlation/gul_S2_summarycalc_P10 fifo/full_correlation/gul_S2_pltcalc_P10 work/full_correlation/gul_S2_summaryaalcalc/P10.bin work/full_correlation/gul_S2_summaryleccalc/P10.bin > /dev/null & pid199=$!
tee < fifo/full_correlation/gul_S2_summary_P10.idx work/full_correlation/gul_S2_summaryleccalc/P10.idx > /dev/null & pid200=$!

summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P1 -2 fifo/full_correlation/gul_S2_summary_P1 < fifo/full_correlation/gul_P1 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P2 -2 fifo/full_correlation/gul_S2_summary_P2 < fifo/full_correlation/gul_P2 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P3 -2 fifo/full_correlation/gul_S2_summary_P3 < fifo/full_correlation/gul_P3 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P4 -2 fifo/full_correlation/gul_S2_summary_P4 < fifo/full_correlation/gul_P4 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P5 -2 fifo/full_correlation/gul_S2_summary_P5 < fifo/full_correlation/gul_P5 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P6 -2 fifo/full_correlation/gul_S2_summary_P6 < fifo/full_correlation/gul_P6 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P7 -2 fifo/full_correlation/gul_S2_summary_P7 < fifo/full_correlation/gul_P7 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P8 -2 fifo/full_correlation/gul_S2_summary_P8 < fifo/full_correlation/gul_P8 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P9 -2 fifo/full_correlation/gul_S2_summary_P9 < fifo/full_correlation/gul_P9 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P10 -2 fifo/full_correlation/gul_S2_summary_P10 < fifo/full_correlation/gul_P10 &

eve 1 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  &
eve 2 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P2 -a1 -i - > fifo/gul_P2  &
eve 3 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P3 -a1 -i - > fifo/gul_P3  &
eve 4 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P4 -a1 -i - > fifo/gul_P4  &
eve 5 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P5 -a1 -i - > fifo/gul_P5  &
eve 6 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P6 -a1 -i - > fifo/gul_P6  &
eve 7 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P7 -a1 -i - > fifo/gul_P7  &
eve 8 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P8 -a1 -i - > fifo/gul_P8  &
eve 9 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P9 -a1 -i - > fifo/gul_P9  &
eve 10 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P10 -a1 -i - > fifo/gul_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 > output/gul_S1_summarycalc.csv & kpid3=$!
kat -s work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 work/kat/gul_S2_eltcalc_P3 work/kat/gul_S2_eltcalc_P4 work/kat/gul_S2_eltcalc_P5 work/kat/gul_S2_eltcalc_P6 work/kat/gul_S2_eltcalc_P7 work/kat/gul_S2_eltcalc_P8 work/kat/gul_S2_eltcalc_P9 work/kat/gul_S2_eltcalc_P10 > output/gul_S2_eltcalc.csv & kpid4=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 work/kat/gul_S2_pltcalc_P3 work/kat/gul_S2_pltcalc_P4 work/kat/gul_S2_pltcalc_P5 work/kat/gul_S2_pltcalc_P6 work/kat/gul_S2_pltcalc_P7 work/kat/gul_S2_pltcalc_P8 work/kat/gul_S2_pltcalc_P9 work/kat/gul_S2_pltcalc_P10 > output/gul_S2_pltcalc.csv & kpid5=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 work/kat/gul_S2_summarycalc_P3 work/kat/gul_S2_summarycalc_P4 work/kat/gul_S2_summarycalc_P5 work/kat/gul_S2_summarycalc_P6 work/kat/gul_S2_summarycalc_P7 work/kat/gul_S2_summarycalc_P8 work/kat/gul_S2_summarycalc_P9 work/kat/gul_S2_summarycalc_P10 > output/gul_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 work/full_correlation/kat/gul_S1_eltcalc_P3 work/full_correlation/kat/gul_S1_eltcalc_P4 work/full_correlation/kat/gul_S1_eltcalc_P5 work/full_correlation/kat/gul_S1_eltcalc_P6 work/full_correlation/kat/gul_S1_eltcalc_P7 work/full_correlation/kat/gul_S1_eltcalc_P8 work/full_correlation/kat/gul_S1_eltcalc_P9 work/full_correlation/kat/gul_S1_eltcalc_P10 > output/full_correlation/gul_S1_eltcalc.csv & kpid7=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 work/full_correlation/kat/gul_S1_pltcalc_P3 work/full_correlation/kat/gul_S1_pltcalc_P4 work/full_correlation/kat/gul_S1_pltcalc_P5 work/full_correlation/kat/gul_S1_pltcalc_P6 work/full_correlation/kat/gul_S1_pltcalc_P7 work/full_correlation/kat/gul_S1_pltcalc_P8 work/full_correlation/kat/gul_S1_pltcalc_P9 work/full_correlation/kat/gul_S1_pltcalc_P10 > output/full_correlation/gul_S1_pltcalc.csv & kpid8=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 work/full_correlation/kat/gul_S1_summarycalc_P3 work/full_correlation/kat/gul_S1_summarycalc_P4 work/full_correlation/kat/gul_S1_summarycalc_P5 work/full_correlation/kat/gul_S1_summarycalc_P6 work/full_correlation/kat/gul_S1_summarycalc_P7 work/full_correlation/kat/gul_S1_summarycalc_P8 work/full_correlation/kat/gul_S1_summarycalc_P9 work/full_correlation/kat/gul_S1_summarycalc_P10 > output/full_correlation/gul_S1_summarycalc.csv & kpid9=$!
kat -s work/full_correlation/kat/gul_S2_eltcalc_P1 work/full_correlation/kat/gul_S2_eltcalc_P2 work/full_correlation/kat/gul_S2_eltcalc_P3 work/full_correlation/kat/gul_S2_eltcalc_P4 work/full_correlation/kat/gul_S2_eltcalc_P5 work/full_correlation/kat/gul_S2_eltcalc_P6 work/full_correlation/kat/gul_S2_eltcalc_P7 work/full_correlation/kat/gul_S2_eltcalc_P8 work/full_correlation/kat/gul_S2_eltcalc_P9 work/full_correlation/kat/gul_S2_eltcalc_P10 > output/full_correlation/gul_S2_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S2_pltcalc_P1 work/full_correlation/kat/gul_S2_pltcalc_P2 work/full_correlation/kat/gul_S2_pltcalc_P3 work/full_correlation/kat/gul_S2_pltcalc_P4 work/full_correlation/kat/gul_S2_pltcalc_P5 work/full_correlation/kat/gul_S2_pltcalc_P6 work/full_correlation/kat/gul_S2_pltcalc_P7 work/full_correlation/kat/gul_S2_pltcalc_P8 work/full_correlation/kat/gul_S2_pltcalc_P9 work/full_correlation/kat/gul_S2_pltcalc_P10 > output/full_correlation/gul_S2_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S2_summarycalc_P1 work/full_correlation/kat/gul_S2_summarycalc_P2 work/full_correlation/kat/gul_S2_summarycalc_P3 work/full_correlation/kat/gul_S2_summarycalc_P4 work/full_correlation/kat/gul_S2_summarycalc_P5 work/full_correlation/kat/gul_S2_summarycalc_P6 work/full_correlation/kat/gul_S2_summarycalc_P7 work/full_correlation/kat/gul_S2_summarycalc_P8 work/full_correlation/kat/gul_S2_summarycalc_P9 work/full_correlation/kat/gul_S2_summarycalc_P10 > output/full_correlation/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
ordleccalc -r -Kgul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S1_ept.csv -o output/gul_S1_psept.csv & lpid2=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid3=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid4=$!
ordleccalc -r -Kgul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S2_ept.csv -o output/gul_S2_psept.csv & lpid5=$!
leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid7=$!
ordleccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/gul_S1_ept.csv -o output/full_correlation/gul_S1_psept.csv & lpid8=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv & lpid9=$!
aalcalc -Kfull_correlation/gul_S2_summaryaalcalc > output/full_correlation/gul_S2_aalcalc.csv & lpid10=$!
ordleccalc -r -Kfull_correlation/gul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/full_correlation/gul_S2_ept.csv -o output/full_correlation/gul_S2_psept.csv & lpid11=$!
leccalc -r -Kfull_correlation/gul_S2_summaryleccalc -F output/full_correlation/gul_S2_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S2_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S2_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S2_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S2_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S2_leccalc_wheatsheaf_oep.csv & lpid12=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8 $lpid9 $lpid10 $lpid11 $lpid12

rm -R -f work/*
rm -R -f fifo/*
