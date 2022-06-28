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

mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/ri_S1_summaryleccalc
mkdir -p work/ri_S1_summaryaalcalc

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

eve 1 1 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmcalc -a2 | tee /tmp/%FIFO_DIR%/fifo/il_P1 | fmcalc -a3 -n -p RI_1 > /tmp/%FIFO_DIR%/fifo/ri_P1 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15

