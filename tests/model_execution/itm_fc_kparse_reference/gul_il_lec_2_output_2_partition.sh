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

rm -R -f fifo/*
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/gul_S2_summaryleccalc
mkdir -p work/gul_S2_summaryaalcalc
mkdir -p work/full_correlation/gul_S1_summaryleccalc
mkdir -p work/full_correlation/gul_S1_summaryaalcalc
mkdir -p work/full_correlation/gul_S2_summaryleccalc
mkdir -p work/full_correlation/gul_S2_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/il_S2_summaryleccalc
mkdir -p work/il_S2_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryleccalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S2_summaryleccalc
mkdir -p work/full_correlation/il_S2_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P1
mkfifo fifo/full_correlation/gul_fc_P2

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2

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

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_summary_P1.idx
mkfifo fifo/il_S2_eltcalc_P1
mkfifo fifo/il_S2_summarycalc_P1
mkfifo fifo/il_S2_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_summary_P2.idx
mkfifo fifo/il_S2_eltcalc_P2
mkfifo fifo/il_S2_summarycalc_P2
mkfifo fifo/il_S2_pltcalc_P2

mkfifo fifo/full_correlation/gul_P1
mkfifo fifo/full_correlation/gul_P2

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

mkfifo fifo/full_correlation/il_P1
mkfifo fifo/full_correlation/il_P2

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summary_P1.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1
mkfifo fifo/full_correlation/il_S2_summary_P1
mkfifo fifo/full_correlation/il_S2_summary_P1.idx
mkfifo fifo/full_correlation/il_S2_eltcalc_P1
mkfifo fifo/full_correlation/il_S2_summarycalc_P1
mkfifo fifo/full_correlation/il_S2_pltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_summary_P2.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P2
mkfifo fifo/full_correlation/il_S1_summarycalc_P2
mkfifo fifo/full_correlation/il_S1_pltcalc_P2
mkfifo fifo/full_correlation/il_S2_summary_P2
mkfifo fifo/full_correlation/il_S2_summary_P2.idx
mkfifo fifo/full_correlation/il_S2_eltcalc_P2
mkfifo fifo/full_correlation/il_S2_summarycalc_P2
mkfifo fifo/full_correlation/il_S2_pltcalc_P2



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc < fifo/il_S2_eltcalc_P1 > work/kat/il_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < fifo/il_S2_summarycalc_P1 > work/kat/il_S2_summarycalc_P1 & pid5=$!
pltcalc < fifo/il_S2_pltcalc_P1 > work/kat/il_S2_pltcalc_P1 & pid6=$!
eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid8=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid9=$!
eltcalc -s < fifo/il_S2_eltcalc_P2 > work/kat/il_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < fifo/il_S2_summarycalc_P2 > work/kat/il_S2_summarycalc_P2 & pid11=$!
pltcalc -H < fifo/il_S2_pltcalc_P2 > work/kat/il_S2_pltcalc_P2 & pid12=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid13=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid14=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_eltcalc_P1 fifo/il_S2_summarycalc_P1 fifo/il_S2_pltcalc_P1 work/il_S2_summaryaalcalc/P1.bin work/il_S2_summaryleccalc/P1.bin > /dev/null & pid15=$!
tee < fifo/il_S2_summary_P1.idx work/il_S2_summaryaalcalc/P1.idx work/il_S2_summaryleccalc/P1.idx > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid18=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_eltcalc_P2 fifo/il_S2_summarycalc_P2 fifo/il_S2_pltcalc_P2 work/il_S2_summaryaalcalc/P2.bin work/il_S2_summaryleccalc/P2.bin > /dev/null & pid19=$!
tee < fifo/il_S2_summary_P2.idx work/il_S2_summaryaalcalc/P2.idx work/il_S2_summaryleccalc/P2.idx > /dev/null & pid20=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 < fifo/il_P2 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid21=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid22=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid23=$!
eltcalc < fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid24=$!
summarycalctocsv < fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid25=$!
pltcalc < fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid26=$!
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid27=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid28=$!
pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid29=$!
eltcalc -s < fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid30=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid31=$!
pltcalc -H < fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid32=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid33=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid34=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_eltcalc_P1 fifo/gul_S2_summarycalc_P1 fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid35=$!
tee < fifo/gul_S2_summary_P1.idx work/gul_S2_summaryaalcalc/P1.idx work/gul_S2_summaryleccalc/P1.idx > /dev/null & pid36=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid37=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid38=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_eltcalc_P2 fifo/gul_S2_summarycalc_P2 fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid39=$!
tee < fifo/gul_S2_summary_P2.idx work/gul_S2_summaryaalcalc/P2.idx work/gul_S2_summaryleccalc/P2.idx > /dev/null & pid40=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &

# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid41=$!
summarycalctocsv < fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid42=$!
pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid43=$!
eltcalc < fifo/full_correlation/il_S2_eltcalc_P1 > work/full_correlation/kat/il_S2_eltcalc_P1 & pid44=$!
summarycalctocsv < fifo/full_correlation/il_S2_summarycalc_P1 > work/full_correlation/kat/il_S2_summarycalc_P1 & pid45=$!
pltcalc < fifo/full_correlation/il_S2_pltcalc_P1 > work/full_correlation/kat/il_S2_pltcalc_P1 & pid46=$!
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid47=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid48=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid49=$!
eltcalc -s < fifo/full_correlation/il_S2_eltcalc_P2 > work/full_correlation/kat/il_S2_eltcalc_P2 & pid50=$!
summarycalctocsv -s < fifo/full_correlation/il_S2_summarycalc_P2 > work/full_correlation/kat/il_S2_summarycalc_P2 & pid51=$!
pltcalc -H < fifo/full_correlation/il_S2_pltcalc_P2 > work/full_correlation/kat/il_S2_pltcalc_P2 & pid52=$!


tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_eltcalc_P1 fifo/full_correlation/il_S1_summarycalc_P1 fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid53=$!
tee < fifo/full_correlation/il_S1_summary_P1.idx work/full_correlation/il_S1_summaryaalcalc/P1.idx work/full_correlation/il_S1_summaryleccalc/P1.idx > /dev/null & pid54=$!
tee < fifo/full_correlation/il_S2_summary_P1 fifo/full_correlation/il_S2_eltcalc_P1 fifo/full_correlation/il_S2_summarycalc_P1 fifo/full_correlation/il_S2_pltcalc_P1 work/full_correlation/il_S2_summaryaalcalc/P1.bin work/full_correlation/il_S2_summaryleccalc/P1.bin > /dev/null & pid55=$!
tee < fifo/full_correlation/il_S2_summary_P1.idx work/full_correlation/il_S2_summaryaalcalc/P1.idx work/full_correlation/il_S2_summaryleccalc/P1.idx > /dev/null & pid56=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_eltcalc_P2 fifo/full_correlation/il_S1_summarycalc_P2 fifo/full_correlation/il_S1_pltcalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid57=$!
tee < fifo/full_correlation/il_S1_summary_P2.idx work/full_correlation/il_S1_summaryaalcalc/P2.idx work/full_correlation/il_S1_summaryleccalc/P2.idx > /dev/null & pid58=$!
tee < fifo/full_correlation/il_S2_summary_P2 fifo/full_correlation/il_S2_eltcalc_P2 fifo/full_correlation/il_S2_summarycalc_P2 fifo/full_correlation/il_S2_pltcalc_P2 work/full_correlation/il_S2_summaryaalcalc/P2.bin work/full_correlation/il_S2_summaryleccalc/P2.bin > /dev/null & pid59=$!
tee < fifo/full_correlation/il_S2_summary_P2.idx work/full_correlation/il_S2_summaryaalcalc/P2.idx work/full_correlation/il_S2_summaryleccalc/P2.idx > /dev/null & pid60=$!

summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P1 -2 fifo/full_correlation/il_S2_summary_P1 < fifo/full_correlation/il_P1 &
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P2 -2 fifo/full_correlation/il_S2_summary_P2 < fifo/full_correlation/il_P2 &

# --- Do ground up loss computes ---

eltcalc < fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid61=$!
summarycalctocsv < fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid62=$!
pltcalc < fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid63=$!
eltcalc < fifo/full_correlation/gul_S2_eltcalc_P1 > work/full_correlation/kat/gul_S2_eltcalc_P1 & pid64=$!
summarycalctocsv < fifo/full_correlation/gul_S2_summarycalc_P1 > work/full_correlation/kat/gul_S2_summarycalc_P1 & pid65=$!
pltcalc < fifo/full_correlation/gul_S2_pltcalc_P1 > work/full_correlation/kat/gul_S2_pltcalc_P1 & pid66=$!
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid67=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid68=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid69=$!
eltcalc -s < fifo/full_correlation/gul_S2_eltcalc_P2 > work/full_correlation/kat/gul_S2_eltcalc_P2 & pid70=$!
summarycalctocsv -s < fifo/full_correlation/gul_S2_summarycalc_P2 > work/full_correlation/kat/gul_S2_summarycalc_P2 & pid71=$!
pltcalc -H < fifo/full_correlation/gul_S2_pltcalc_P2 > work/full_correlation/kat/gul_S2_pltcalc_P2 & pid72=$!


tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_eltcalc_P1 fifo/full_correlation/gul_S1_summarycalc_P1 fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid73=$!
tee < fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryaalcalc/P1.idx work/full_correlation/gul_S1_summaryleccalc/P1.idx > /dev/null & pid74=$!
tee < fifo/full_correlation/gul_S2_summary_P1 fifo/full_correlation/gul_S2_eltcalc_P1 fifo/full_correlation/gul_S2_summarycalc_P1 fifo/full_correlation/gul_S2_pltcalc_P1 work/full_correlation/gul_S2_summaryaalcalc/P1.bin work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid75=$!
tee < fifo/full_correlation/gul_S2_summary_P1.idx work/full_correlation/gul_S2_summaryaalcalc/P1.idx work/full_correlation/gul_S2_summaryleccalc/P1.idx > /dev/null & pid76=$!
tee < fifo/full_correlation/gul_S1_summary_P2 fifo/full_correlation/gul_S1_eltcalc_P2 fifo/full_correlation/gul_S1_summarycalc_P2 fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid77=$!
tee < fifo/full_correlation/gul_S1_summary_P2.idx work/full_correlation/gul_S1_summaryaalcalc/P2.idx work/full_correlation/gul_S1_summaryleccalc/P2.idx > /dev/null & pid78=$!
tee < fifo/full_correlation/gul_S2_summary_P2 fifo/full_correlation/gul_S2_eltcalc_P2 fifo/full_correlation/gul_S2_summarycalc_P2 fifo/full_correlation/gul_S2_pltcalc_P2 work/full_correlation/gul_S2_summaryaalcalc/P2.bin work/full_correlation/gul_S2_summaryleccalc/P2.bin > /dev/null & pid79=$!
tee < fifo/full_correlation/gul_S2_summary_P2.idx work/full_correlation/gul_S2_summaryaalcalc/P2.idx work/full_correlation/gul_S2_summaryleccalc/P2.idx > /dev/null & pid80=$!

summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P1 -2 fifo/full_correlation/gul_S2_summary_P1 < fifo/full_correlation/gul_P1 &
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P2 -2 fifo/full_correlation/gul_S2_summary_P2 < fifo/full_correlation/gul_P2 &

tee < fifo/full_correlation/gul_fc_P1 fifo/full_correlation/gul_P1  | fmcalc -a2 > fifo/full_correlation/il_P1  &
tee < fifo/full_correlation/gul_fc_P2 fifo/full_correlation/gul_P2  | fmcalc -a2 > fifo/full_correlation/il_P2  &
eve 1 2 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  &
eve 2 2 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_fc_P2 -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P1 work/kat/il_S2_eltcalc_P2 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P1 work/kat/il_S2_pltcalc_P2 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P1 work/kat/il_S2_summarycalc_P2 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do insured loss kats for fully correlated output ---

