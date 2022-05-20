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

find fifo/ \( -name '*P21[^0-9]*' -o -name '*P21' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

mkfifo fifo/gul_P21

mkfifo fifo/gul_S1_summary_P21
mkfifo fifo/gul_S1_summary_P21.idx
mkfifo fifo/gul_S1_eltcalc_P21
mkfifo fifo/gul_S1_summarycalc_P21
mkfifo fifo/gul_S1_pltcalc_P21

mkfifo fifo/il_P21

mkfifo fifo/il_S1_summary_P21
mkfifo fifo/il_S1_summary_P21.idx
mkfifo fifo/il_S1_eltcalc_P21
mkfifo fifo/il_S1_summarycalc_P21
mkfifo fifo/il_S1_pltcalc_P21



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P21 > work/kat/il_S1_eltcalc_P21 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P21 > work/kat/il_S1_summarycalc_P21 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P21 > work/kat/il_S1_pltcalc_P21 & pid3=$!
tee < fifo/il_S1_summary_P21 fifo/il_S1_eltcalc_P21 fifo/il_S1_summarycalc_P21 fifo/il_S1_pltcalc_P21 work/il_S1_summaryaalcalc/P21.bin work/il_S1_summaryleccalc/P21.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P21.idx work/il_S1_summaryaalcalc/P21.idx work/il_S1_summaryleccalc/P21.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P21 < fifo/il_P21 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P21 > work/kat/gul_S1_eltcalc_P21 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P21 > work/kat/gul_S1_summarycalc_P21 & pid7=$!
pltcalc -H < fifo/gul_S1_pltcalc_P21 > work/kat/gul_S1_pltcalc_P21 & pid8=$!
tee < fifo/gul_S1_summary_P21 fifo/gul_S1_eltcalc_P21 fifo/gul_S1_summarycalc_P21 fifo/gul_S1_pltcalc_P21 work/gul_S1_summaryaalcalc/P21.bin work/gul_S1_summaryleccalc/P21.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P21.idx work/gul_S1_summaryaalcalc/P21.idx work/gul_S1_summaryleccalc/P21.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P21 < fifo/gul_P21 &

eve 21 40 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | tee fifo/gul_P21 | fmpy -a2 > fifo/il_P21  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P21 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P21 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P21 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P21 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P21 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P21 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

