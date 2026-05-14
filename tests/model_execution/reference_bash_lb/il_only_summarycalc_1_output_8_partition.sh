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

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P4
mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2
mkfifo fifo/gul_lb_P3
mkfifo fifo/gul_lb_P4

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2
mkfifo fifo/lb_il_P3
mkfifo fifo/lb_il_P4



# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P2 > /dev/null & pid2=$!
tee < fifo/il_S1_summary_P3 > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P4 > /dev/null & pid4=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarypy -m -t il  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &

( evepy 1 4 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P1  ) & 
( evepy 2 4 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P2  ) & 
( evepy 3 4 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P3  ) & 
( evepy 4 4 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P4  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
load_balancer -i fifo/gul_lb_P3 fifo/gul_lb_P4 -o fifo/lb_il_P3 fifo/lb_il_P4 &

# --- Verify FIFO pipes ---
check_fifos \
    fifo/il_P1 \
    fifo/il_P2 \
    fifo/il_P3 \
    fifo/il_P4 \
    fifo/il_S1_summary_P1 \
    fifo/il_S1_summary_P2 \
    fifo/il_S1_summary_P3 \
    fifo/il_S1_summary_P4 \
    fifo/gul_lb_P1 \
    fifo/gul_lb_P2 \
    fifo/gul_lb_P3 \
    fifo/gul_lb_P4 \
    fifo/lb_il_P1 \
    fifo/lb_il_P2 \
    fifo/lb_il_P3 \
    fifo/lb_il_P4

( fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 ) & pid5=$!
( fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 ) & pid6=$!
( fmpy -a2 < fifo/lb_il_P3 > fifo/il_P3 ) & pid7=$!
( fmpy -a2 < fifo/lb_il_P4 > fifo/il_P4 ) & pid8=$!

exec_wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---


rm -R -f work/*
rm -R -f fifo/*
