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

find fifo/ \( -name '*P34[^0-9]*' -o -name '*P34' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

mkfifo fifo/gul_P34

mkfifo fifo/gul_S1_summary_P34
mkfifo fifo/gul_S1_summary_P34.idx
mkfifo fifo/gul_S1_eltcalc_P34
mkfifo fifo/gul_S1_summarycalc_P34
mkfifo fifo/gul_S1_pltcalc_P34

mkfifo fifo/il_P34

mkfifo fifo/il_S1_summary_P34
mkfifo fifo/il_S1_summary_P34.idx
mkfifo fifo/il_S1_eltcalc_P34
mkfifo fifo/il_S1_summarycalc_P34
mkfifo fifo/il_S1_pltcalc_P34



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P34 > work/kat/il_S1_eltcalc_P34 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P34 > work/kat/il_S1_summarycalc_P34 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P34 > work/kat/il_S1_pltcalc_P34 & pid3=$!
tee < fifo/il_S1_summary_P34 fifo/il_S1_eltcalc_P34 fifo/il_S1_summarycalc_P34 fifo/il_S1_pltcalc_P34 work/il_S1_summaryaalcalc/P34.bin work/il_S1_summaryleccalc/P34.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P34.idx work/il_S1_summaryaalcalc/P34.idx work/il_S1_summaryleccalc/P34.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P34 < fifo/il_P34 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P34 > work/kat/gul_S1_eltcalc_P34 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P34 > work/kat/gul_S1_summarycalc_P34 & pid7=$!
pltcalc -H < fifo/gul_S1_pltcalc_P34 > work/kat/gul_S1_pltcalc_P34 & pid8=$!
tee < fifo/gul_S1_summary_P34 fifo/gul_S1_eltcalc_P34 fifo/gul_S1_summarycalc_P34 fifo/gul_S1_pltcalc_P34 work/gul_S1_summaryaalcalc/P34.bin work/gul_S1_summaryleccalc/P34.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P34.idx work/gul_S1_summaryaalcalc/P34.idx work/gul_S1_summaryleccalc/P34.idx > /dev/null & pid10=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P34 < fifo/gul_P34 &

eve 34 40 | getmodel | gulcalc -S100 -L100 -r -c fifo/gul_P34 -i - | fmcalc -a2 > fifo/il_P34  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10

