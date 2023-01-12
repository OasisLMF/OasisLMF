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

rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/full_correlation/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc
mkdir -p work/ri_S1_summaryleccalc
mkdir -p work/ri_S1_summaryaalcalc
mkdir -p work/full_correlation/ri_S1_summaryleccalc
mkdir -p work/full_correlation/ri_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/ri_P1

mkfifo /tmp/%FIFO_DIR%/fifo/ri_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/ri_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/ri_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/ri_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/ri_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_pltcalc_P1



# --- Do reinsurance loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/ri_S1_eltcalc_P1 > work/kat/ri_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/ri_S1_summarycalc_P1 > work/kat/ri_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/ri_S1_pltcalc_P1 > work/kat/ri_S1_pltcalc_P1 & pid3=$!


tee < /tmp/%FIFO_DIR%/fifo/ri_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/ri_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/ri_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/ri_S1_pltcalc_P1 work/ri_S1_summaryaalcalc/P1.bin work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/ri_S1_summary_P1.idx work/ri_S1_summaryaalcalc/P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid5=$!

summarycalc -m -f -p RI_1 -1 /tmp/%FIFO_DIR%/fifo/ri_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/ri_P1 &

# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid6=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid7=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid8=$!


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx > /dev/null & pid10=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid11=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid12=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid13=$!


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid15=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &

# --- Do reinsurance loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_eltcalc_P1 > work/full_correlation/kat/ri_S1_eltcalc_P1 & pid16=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summarycalc_P1 > work/full_correlation/kat/ri_S1_summarycalc_P1 & pid17=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_pltcalc_P1 > work/full_correlation/kat/ri_S1_pltcalc_P1 & pid18=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_pltcalc_P1 work/full_correlation/ri_S1_summaryaalcalc/P1.bin work/full_correlation/ri_S1_summaryleccalc/P1.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summary_P1.idx work/full_correlation/ri_S1_summaryaalcalc/P1.idx work/full_correlation/ri_S1_summaryleccalc/P1.idx > /dev/null & pid20=$!

summarycalc -m -f -p RI_1 -1 /tmp/%FIFO_DIR%/fifo/full_correlation/ri_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/ri_P1 &

# --- Do insured loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid21=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid22=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid23=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin > /dev/null & pid24=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1.idx work/full_correlation/il_S1_summaryaalcalc/P1.idx > /dev/null & pid25=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid26=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid27=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid28=$!


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1.idx work/full_correlation/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid30=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 &

( tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1  | fmcalc -a2 | tee /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 | fmcalc -a3 -n -p RI_1 > /tmp/%FIFO_DIR%/fifo/full_correlation/ri_P1 ) & pid31=$!
( eve 1 1 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P1 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmcalc -a2 | tee /tmp/%FIFO_DIR%/fifo/il_P1 | fmcalc -a3 -n -p RI_1 > /tmp/%FIFO_DIR%/fifo/ri_P1 ) & pid32=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32

