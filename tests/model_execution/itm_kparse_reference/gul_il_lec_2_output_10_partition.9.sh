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

find fifo/ \( -name '*P10[^0-9]*' -o -name '*P10' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/gul_S2_summaryleccalc
mkdir -p work/gul_S2_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/il_S2_summaryleccalc
mkdir -p work/il_S2_summaryaalcalc

mkfifo fifo/gul_P10

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

mkfifo fifo/il_P10

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summary_P10.idx
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10
mkfifo fifo/il_S2_summary_P10
mkfifo fifo/il_S2_summary_P10.idx
mkfifo fifo/il_S2_eltcalc_P10
mkfifo fifo/il_S2_summarycalc_P10
mkfifo fifo/il_S2_pltcalc_P10



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 & pid3=$!
eltcalc -s < fifo/il_S2_eltcalc_P10 > work/kat/il_S2_eltcalc_P10 & pid4=$!
summarycalctocsv -s < fifo/il_S2_summarycalc_P10 > work/kat/il_S2_summarycalc_P10 & pid5=$!
pltcalc -H < fifo/il_S2_pltcalc_P10 > work/kat/il_S2_pltcalc_P10 & pid6=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P10.idx work/il_S1_summaryaalcalc/P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid8=$!
tee < fifo/il_S2_summary_P10 fifo/il_S2_eltcalc_P10 fifo/il_S2_summarycalc_P10 fifo/il_S2_pltcalc_P10 work/il_S2_summaryaalcalc/P10.bin work/il_S2_summaryleccalc/P10.bin > /dev/null & pid9=$!
tee < fifo/il_S2_summary_P10.idx work/il_S2_summaryaalcalc/P10.idx work/il_S2_summaryleccalc/P10.idx > /dev/null & pid10=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P10 -2 fifo/il_S2_summary_P10 < fifo/il_P10 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid11=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid12=$!
pltcalc -H < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 & pid13=$!
eltcalc -s < fifo/gul_S2_eltcalc_P10 > work/kat/gul_S2_eltcalc_P10 & pid14=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P10 > work/kat/gul_S2_summarycalc_P10 & pid15=$!
pltcalc -H < fifo/gul_S2_pltcalc_P10 > work/kat/gul_S2_pltcalc_P10 & pid16=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid17=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid18=$!
tee < fifo/gul_S2_summary_P10 fifo/gul_S2_eltcalc_P10 fifo/gul_S2_summarycalc_P10 fifo/gul_S2_pltcalc_P10 work/gul_S2_summaryaalcalc/P10.bin work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid19=$!
tee < fifo/gul_S2_summary_P10.idx work/gul_S2_summaryaalcalc/P10.idx work/gul_S2_summaryleccalc/P10.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 &

eve 10 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20

