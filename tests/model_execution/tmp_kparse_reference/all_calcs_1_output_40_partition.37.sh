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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P38[^0-9]*' -o -name '*P38' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P38

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38

mkfifo /tmp/%FIFO_DIR%/fifo/il_P38

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38 > work/kat/il_S1_eltcalc_P38 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38 > work/kat/il_S1_summarycalc_P38 & pid2=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38 > work/kat/il_S1_pltcalc_P38 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P38 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P38 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P38 work/il_S1_summaryaalcalc/P38.bin work/il_S1_summaryleccalc/P38.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38.idx work/il_S1_summaryaalcalc/P38.idx work/il_S1_summaryleccalc/P38.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P38 < /tmp/%FIFO_DIR%/fifo/il_P38 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38 > work/kat/gul_S1_eltcalc_P38 & pid6=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38 > work/kat/gul_S1_summarycalc_P38 & pid7=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38 > work/kat/gul_S1_pltcalc_P38 & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P38 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P38 work/gul_S1_summaryaalcalc/P38.bin work/gul_S1_summaryleccalc/P38.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38.idx work/gul_S1_summaryaalcalc/P38.idx work/gul_S1_summaryleccalc/P38.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P38 < /tmp/%FIFO_DIR%/fifo/gul_P38 &

eve 38 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P38 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P38  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10

