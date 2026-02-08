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

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1

mkfifo fifo/il_S1_summary_P2

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2



# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P2 > /dev/null & pid2=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

( evepy 1 2 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P1  ) & 
( evepy 2 2 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P2  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
( fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 ) & pid3=$!
( fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 ) & pid4=$!

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---


rm -R -f work/*
rm -R -f fifo/*
