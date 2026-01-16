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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P9[^0-9]*' -o -name '*P9' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_P9

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9 > /dev/null & pid2=$!
summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/il_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/il_P9 &

# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9 > /dev/null & pid4=$!
summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P9 < /tmp/%FIFO_DIR%/fifo/gul_P9 &

( evepy 9 10 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P9 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P9  ) & pid5=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5

