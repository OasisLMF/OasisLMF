#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


check_fifos() {
    local has_error=0
    for f in "$@"; do
        [ -e "$f" ] || { echo "[ERROR] Expected FIFO not found: $f"; has_error=1; continue; }
        [ -p "$f" ] || { echo "[ERROR] Not a FIFO: $f"; has_error=1; }
    done
    [ "$has_error" -eq 0 ] || false
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

find fifo/ -regextype posix-extended -regex '.*/[^/]*_P3([^0-9].*)?$' -exec rm -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/il_S1_summaryleccalc

mkfifo fifo/gul_P3

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx


# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P3 work/il_S1_summaryleccalc/P3.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid2=$!
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &

# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid3=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid4=$!
summarypy -m -t gul  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &


# --- Verify FIFO pipes ---
check_fifos \
    fifo/gul_P3 \
    fifo/gul_S1_summary_P3 \
    fifo/gul_S1_summary_P3.idx \
    fifo/il_P3 \
    fifo/il_S1_summary_P3 \
    fifo/il_S1_summary_P3.idx

( evepy 3 8 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S0 -L0 -a0  | tee fifo/gul_P3 | fmpy -a2 > fifo/il_P3  ) & pid5=$!

wait -p pid_exitcode $pid1 $pid2 $pid3 $pid4 $pid5

