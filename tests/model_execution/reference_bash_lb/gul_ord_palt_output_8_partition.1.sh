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

find fifo/ \( -name '*P2[^0-9]*' -o -name '*P2' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S2_summary_palt

mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summary_P2.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P2 work/gul_S1_summary_palt/P2.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summary_palt/P2.idx > /dev/null & pid2=$!
tee < fifo/gul_S2_summary_P2 work/gul_S2_summary_palt/P2.bin > /dev/null & pid3=$!
tee < fifo/gul_S2_summary_P2.idx work/gul_S2_summary_palt/P2.idx > /dev/null & pid4=$!
summarypy -m -t gul  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &

( evepy 2 8 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a0  > fifo/gul_P2  ) &  pid5=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5

