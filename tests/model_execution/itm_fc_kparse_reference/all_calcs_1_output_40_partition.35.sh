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

find fifo/ \( -name '*P36[^0-9]*' -o -name '*P36' \) -exec rm -R -f {} +
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

mkfifo fifo/full_correlation/gul_fc_P36

mkfifo fifo/gul_P36

mkfifo fifo/gul_S1_summary_P36
mkfifo fifo/gul_S1_summary_P36.idx
mkfifo fifo/gul_S1_eltcalc_P36
mkfifo fifo/gul_S1_summarycalc_P36
mkfifo fifo/gul_S1_pltcalc_P36

mkfifo fifo/il_P36

mkfifo fifo/il_S1_summary_P36
mkfifo fifo/il_S1_summary_P36.idx
mkfifo fifo/il_S1_eltcalc_P36
mkfifo fifo/il_S1_summarycalc_P36
mkfifo fifo/il_S1_pltcalc_P36

mkfifo fifo/full_correlation/gul_P36

mkfifo fifo/full_correlation/gul_S1_summary_P36
mkfifo fifo/full_correlation/gul_S1_summary_P36.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P36
mkfifo fifo/full_correlation/gul_S1_summarycalc_P36
mkfifo fifo/full_correlation/gul_S1_pltcalc_P36

mkfifo fifo/full_correlation/il_P36

mkfifo fifo/full_correlation/il_S1_summary_P36
mkfifo fifo/full_correlation/il_S1_summary_P36.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P36
mkfifo fifo/full_correlation/il_S1_summarycalc_P36
mkfifo fifo/full_correlation/il_S1_pltcalc_P36



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P36 > work/kat/il_S1_eltcalc_P36 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P36 > work/kat/il_S1_summarycalc_P36 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P36 > work/kat/il_S1_pltcalc_P36 & pid3=$!
tee < fifo/il_S1_summary_P36 fifo/il_S1_eltcalc_P36 fifo/il_S1_summarycalc_P36 fifo/il_S1_pltcalc_P36 work/il_S1_summaryaalcalc/P36.bin work/il_S1_summaryleccalc/P36.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P36.idx work/il_S1_summaryaalcalc/P36.idx work/il_S1_summaryleccalc/P36.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P36 < fifo/il_P36 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P36 > work/kat/gul_S1_eltcalc_P36 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P36 > work/kat/gul_S1_summarycalc_P36 & pid7=$!
pltcalc -H < fifo/gul_S1_pltcalc_P36 > work/kat/gul_S1_pltcalc_P36 & pid8=$!
tee < fifo/gul_S1_summary_P36 fifo/gul_S1_eltcalc_P36 fifo/gul_S1_summarycalc_P36 fifo/gul_S1_pltcalc_P36 work/gul_S1_summaryaalcalc/P36.bin work/gul_S1_summaryleccalc/P36.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P36.idx work/gul_S1_summaryaalcalc/P36.idx work/gul_S1_summaryleccalc/P36.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P36 < fifo/gul_P36 &

# --- Do insured loss computes ---
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P36 > work/full_correlation/kat/il_S1_eltcalc_P36 & pid11=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P36 > work/full_correlation/kat/il_S1_summarycalc_P36 & pid12=$!
pltcalc -H < fifo/full_correlation/il_S1_pltcalc_P36 > work/full_correlation/kat/il_S1_pltcalc_P36 & pid13=$!
tee < fifo/full_correlation/il_S1_summary_P36 fifo/full_correlation/il_S1_eltcalc_P36 fifo/full_correlation/il_S1_summarycalc_P36 fifo/full_correlation/il_S1_pltcalc_P36 work/full_correlation/il_S1_summaryaalcalc/P36.bin work/full_correlation/il_S1_summaryleccalc/P36.bin > /dev/null & pid14=$!
tee < fifo/full_correlation/il_S1_summary_P36.idx work/full_correlation/il_S1_summaryaalcalc/P36.idx work/full_correlation/il_S1_summaryleccalc/P36.idx > /dev/null & pid15=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P36 < fifo/full_correlation/il_P36 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P36 > work/full_correlation/kat/gul_S1_eltcalc_P36 & pid16=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P36 > work/full_correlation/kat/gul_S1_summarycalc_P36 & pid17=$!
pltcalc -H < fifo/full_correlation/gul_S1_pltcalc_P36 > work/full_correlation/kat/gul_S1_pltcalc_P36 & pid18=$!
tee < fifo/full_correlation/gul_S1_summary_P36 fifo/full_correlation/gul_S1_eltcalc_P36 fifo/full_correlation/gul_S1_summarycalc_P36 fifo/full_correlation/gul_S1_pltcalc_P36 work/full_correlation/gul_S1_summaryaalcalc/P36.bin work/full_correlation/gul_S1_summaryleccalc/P36.bin > /dev/null & pid19=$!
tee < fifo/full_correlation/gul_S1_summary_P36.idx work/full_correlation/gul_S1_summaryaalcalc/P36.idx work/full_correlation/gul_S1_summaryleccalc/P36.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P36 < fifo/full_correlation/gul_P36 &

tee < fifo/full_correlation/gul_fc_P36 fifo/full_correlation/gul_P36  | fmcalc -a2 > fifo/full_correlation/il_P36  &
eve 36 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P36 -a1 -i - | tee fifo/gul_P36 | fmcalc -a2 > fifo/il_P36  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P36 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P36 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P36 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P36 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P36 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P36 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P36 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P36 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P36 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P36 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P36 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P36 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

