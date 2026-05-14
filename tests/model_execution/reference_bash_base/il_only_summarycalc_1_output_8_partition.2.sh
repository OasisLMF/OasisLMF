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

exec_wait(){
    local BASH_VER_MAJOR=${BASH_VERSION:0:1}
    local BASH_VER_MINOR=${BASH_VERSION:2:1}
    if [[ "$BASH_VER_MAJOR" -gt 5 ]] || { [[ "$BASH_VER_MAJOR" -eq 5 ]] && [[ "$BASH_VER_MINOR" -ge 1 ]]; }; then
        local pid_exitcode
        wait -p pid_exitcode "$@"
    else
        wait "$@"
    fi
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

find fifo/ -regextype posix-extended -regex '.*/[^/]*_P3([^0-9].*)?$' -exec rm -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3


# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P3 > /dev/null & pid1=$!
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &


# --- Verify FIFO pipes ---
check_fifos \
    fifo/il_P3 \
    fifo/il_S1_summary_P3

( evepy 3 8 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P3  ) & pid2=$!

exec_wait $pid1 $pid2

