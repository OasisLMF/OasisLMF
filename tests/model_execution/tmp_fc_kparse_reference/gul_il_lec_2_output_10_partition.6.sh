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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P7[^0-9]*' -o -name '*P7' \) -exec rm -R -f {} +
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/
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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P7

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

mkfifo /tmp/%FIFO_DIR%/fifo/il_P7

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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7

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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7

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



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 & pid2=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 & pid3=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P7 > work/kat/il_S2_eltcalc_P7 & pid4=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P7 > work/kat/il_S2_summarycalc_P7 & pid5=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P7 > work/kat/il_S2_pltcalc_P7 & pid6=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryaalcalc/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/il_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/il_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/il_S2_pltcalc_P7 work/il_S2_summaryaalcalc/P7.bin work/il_S2_summaryleccalc/P7.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7.idx work/il_S2_summaryaalcalc/P7.idx work/il_S2_summaryleccalc/P7.idx > /dev/null & pid10=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid11=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 & pid12=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 & pid13=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P7 > work/kat/gul_S2_eltcalc_P7 & pid14=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P7 > work/kat/gul_S2_summarycalc_P7 & pid15=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P7 > work/kat/gul_S2_pltcalc_P7 & pid16=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid17=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P7 work/gul_S2_summaryaalcalc/P7.bin work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7.idx work/gul_S2_summaryaalcalc/P7.idx work/gul_S2_summaryleccalc/P7.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/gul_P7 &

# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 & pid21=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7 > work/full_correlation/kat/il_S1_summarycalc_P7 & pid22=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 & pid23=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P7 > work/full_correlation/kat/il_S2_eltcalc_P7 & pid24=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P7 > work/full_correlation/kat/il_S2_summarycalc_P7 & pid25=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P7 > work/full_correlation/kat/il_S2_pltcalc_P7 & pid26=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P7 work/full_correlation/il_S1_summaryaalcalc/P7.bin work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid27=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7.idx work/full_correlation/il_S1_summaryaalcalc/P7.idx work/full_correlation/il_S1_summaryleccalc/P7.idx > /dev/null & pid28=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_pltcalc_P7 work/full_correlation/il_S2_summaryaalcalc/P7.bin work/full_correlation/il_S2_summaryleccalc/P7.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7.idx work/full_correlation/il_S2_summaryaalcalc/P7.idx work/full_correlation/il_S2_summaryleccalc/P7.idx > /dev/null & pid30=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 & pid31=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 & pid32=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 & pid33=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P7 > work/full_correlation/kat/gul_S2_eltcalc_P7 & pid34=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P7 > work/full_correlation/kat/gul_S2_summarycalc_P7 & pid35=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P7 > work/full_correlation/kat/gul_S2_pltcalc_P7 & pid36=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid37=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7.idx work/full_correlation/gul_S1_summaryaalcalc/P7.idx work/full_correlation/gul_S1_summaryleccalc/P7.idx > /dev/null & pid38=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P7 work/full_correlation/gul_S2_summaryaalcalc/P7.bin work/full_correlation/gul_S2_summaryleccalc/P7.bin > /dev/null & pid39=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7.idx work/full_correlation/gul_S2_summaryaalcalc/P7.idx work/full_correlation/gul_S2_summaryleccalc/P7.idx > /dev/null & pid40=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P7 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P7 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7 &

( tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P7  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P7  ) & pid41=$!
( eve 7 10 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P7 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  ) & pid42=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42

