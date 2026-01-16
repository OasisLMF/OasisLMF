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

find fifo/ \( -name '*P29[^0-9]*' -o -name '*P29' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/gul_P29

mkfifo fifo/gul_S1_summary_P29

mkfifo fifo/il_P29

mkfifo fifo/il_S1_summary_P29



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P29 > /dev/null & pid1=$!
summarypy -m -t il  -1 fifo/il_S1_summary_P29 < fifo/il_P29 &

# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P29 > /dev/null & pid2=$!
summarypy -m -t gul  -1 fifo/gul_S1_summary_P29 < fifo/gul_P29 &

( evepy 29 40 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | tee fifo/gul_P29 | fmpy -a2 > fifo/il_P29  ) & pid3=$!

wait $pid1 $pid2 $pid3

