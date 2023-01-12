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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P7[^0-9]*' -o -name '*P7' \) -exec rm -R -f {} +
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

( eve 7 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P7 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  ) & pid21=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21

