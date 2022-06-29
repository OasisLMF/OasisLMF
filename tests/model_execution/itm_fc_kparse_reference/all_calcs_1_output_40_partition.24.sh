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

find fifo/ \( -name '*P25[^0-9]*' -o -name '*P25' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/full_correlation/gul_S1_summaryleccalc
mkdir -p work/full_correlation/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryleccalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P25

mkfifo fifo/gul_P25

mkfifo fifo/gul_S1_summary_P25
mkfifo fifo/gul_S1_summary_P25.idx
mkfifo fifo/gul_S1_eltcalc_P25
mkfifo fifo/gul_S1_summarycalc_P25
mkfifo fifo/gul_S1_pltcalc_P25

mkfifo fifo/il_P25

mkfifo fifo/il_S1_summary_P25
mkfifo fifo/il_S1_summary_P25.idx
mkfifo fifo/il_S1_eltcalc_P25
mkfifo fifo/il_S1_summarycalc_P25
mkfifo fifo/il_S1_pltcalc_P25

mkfifo fifo/full_correlation/gul_P25

mkfifo fifo/full_correlation/gul_S1_summary_P25
mkfifo fifo/full_correlation/gul_S1_summary_P25.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P25
mkfifo fifo/full_correlation/gul_S1_summarycalc_P25
mkfifo fifo/full_correlation/gul_S1_pltcalc_P25

mkfifo fifo/full_correlation/il_P25

mkfifo fifo/full_correlation/il_S1_summary_P25
mkfifo fifo/full_correlation/il_S1_summary_P25.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P25
mkfifo fifo/full_correlation/il_S1_summarycalc_P25
mkfifo fifo/full_correlation/il_S1_pltcalc_P25



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P25 > work/kat/il_S1_eltcalc_P25 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P25 > work/kat/il_S1_summarycalc_P25 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P25 > work/kat/il_S1_pltcalc_P25 & pid3=$!
tee < fifo/il_S1_summary_P25 fifo/il_S1_eltcalc_P25 fifo/il_S1_summarycalc_P25 fifo/il_S1_pltcalc_P25 work/il_S1_summaryaalcalc/P25.bin work/il_S1_summaryleccalc/P25.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P25.idx work/il_S1_summaryaalcalc/P25.idx work/il_S1_summaryleccalc/P25.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P25 < fifo/il_P25 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P25 > work/kat/gul_S1_eltcalc_P25 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P25 > work/kat/gul_S1_summarycalc_P25 & pid7=$!
pltcalc -H < fifo/gul_S1_pltcalc_P25 > work/kat/gul_S1_pltcalc_P25 & pid8=$!
tee < fifo/gul_S1_summary_P25 fifo/gul_S1_eltcalc_P25 fifo/gul_S1_summarycalc_P25 fifo/gul_S1_pltcalc_P25 work/gul_S1_summaryaalcalc/P25.bin work/gul_S1_summaryleccalc/P25.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P25.idx work/gul_S1_summaryaalcalc/P25.idx work/gul_S1_summaryleccalc/P25.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P25 < fifo/gul_P25 &

# --- Do insured loss computes ---
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P25 > work/full_correlation/kat/il_S1_eltcalc_P25 & pid11=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P25 > work/full_correlation/kat/il_S1_summarycalc_P25 & pid12=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P25 > work/full_correlation/kat/il_S1_pltcalc_P25 & pid13=$!
tee < fifo/full_correlation/il_S1_summary_P25 fifo/full_correlation/il_S1_eltcalc_P25 fifo/full_correlation/il_S1_summarycalc_P25 fifo/full_correlation/il_S1_pltcalc_P25 work/full_correlation/il_S1_summaryaalcalc/P25.bin work/full_correlation/il_S1_summaryleccalc/P25.bin > /dev/null & pid14=$!
tee < fifo/full_correlation/il_S1_summary_P25.idx work/full_correlation/il_S1_summaryaalcalc/P25.idx work/full_correlation/il_S1_summaryleccalc/P25.idx > /dev/null & pid15=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P25 < fifo/full_correlation/il_P25 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P25 > work/full_correlation/kat/gul_S1_eltcalc_P25 & pid16=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P25 > work/full_correlation/kat/gul_S1_summarycalc_P25 & pid17=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P25 > work/full_correlation/kat/gul_S1_pltcalc_P25 & pid18=$!
tee < fifo/full_correlation/gul_S1_summary_P25 fifo/full_correlation/gul_S1_eltcalc_P25 fifo/full_correlation/gul_S1_summarycalc_P25 fifo/full_correlation/gul_S1_pltcalc_P25 work/full_correlation/gul_S1_summaryaalcalc/P25.bin work/full_correlation/gul_S1_summaryleccalc/P25.bin > /dev/null & pid19=$!
tee < fifo/full_correlation/gul_S1_summary_P25.idx work/full_correlation/gul_S1_summaryaalcalc/P25.idx work/full_correlation/gul_S1_summaryleccalc/P25.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P25 < fifo/full_correlation/gul_P25 &

tee < fifo/full_correlation/gul_fc_P25 fifo/full_correlation/gul_P25  | fmcalc -a2 > fifo/full_correlation/il_P25  &
eve 25 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P25 -a1 -i - | tee fifo/gul_P25 | fmcalc -a2 > fifo/il_P25  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20

