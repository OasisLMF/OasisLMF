#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*
export OASIS_PYTOOLS_LOG_DIR=$LOG_DIR


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

find fifo/ -regextype posix-extended -regex '.*/[^/]*_P2([^0-9].*)?$' -exec rm -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
#fmpy -a2 --create-financial-structure-files -p input/RI_1
#fmpy -a2 --create-financial-structure-files -p input/RI_2
#fmpy -a2 --create-financial-structure-files -p input/RI_3
mkdir -p work/rl_S1_summaryleccalc
mkdir -p work/rl_S1_summary_palt

mkfifo fifo/rl_IP1_P2

mkfifo fifo/rl_IP1_S1_rlcombine_P2

mkfifo fifo/rl_IP2_P2

mkfifo fifo/rl_IP2_S1_rlcombine_P2

mkfifo fifo/rl_IP3_P2

mkfifo fifo/rl_IP3_S1_rlcombine_P2

mkfifo fifo/rl_S1_summary_P2
mkfifo fifo/rl_S1_summary_P2.idx
mkfifo fifo/rl_S1_plt_ord_P2
mkfifo fifo/rl_S1_selt_ord_P2


# --- Do reinsurance gross loss computes ---
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P2 < fifo/rl_S1_plt_ord_P2 & pid1=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P2 < fifo/rl_S1_selt_ord_P2 & pid2=$!
tee < fifo/rl_S1_summary_P2 fifo/rl_S1_plt_ord_P2 fifo/rl_S1_selt_ord_P2 work/rl_S1_summary_palt/P2.bin work/rl_S1_summaryleccalc/P2.bin > /dev/null & pid3=$!
tee < fifo/rl_S1_summary_P2.idx work/rl_S1_summary_palt/P2.idx work/rl_S1_summaryleccalc/P2.idx > /dev/null & pid4=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P2 fifo/rl_IP2_S1_rlcombine_P2 fifo/rl_IP3_S1_rlcombine_P2 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P2 -m & pid5=$!
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P2 < fifo/rl_IP1_P2 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P2 < fifo/rl_IP2_P2 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P2 < fifo/rl_IP3_P2 &


# --- Verify FIFO pipes ---
check_fifos \
    fifo/rl_IP1_P2 \
    fifo/rl_IP1_S1_rlcombine_P2 \
    fifo/rl_IP2_P2 \
    fifo/rl_IP2_S1_rlcombine_P2 \
    fifo/rl_IP3_P2 \
    fifo/rl_IP3_S1_rlcombine_P2 \
    fifo/rl_S1_summary_P2 \
    fifo/rl_S1_summary_P2.idx \
    fifo/rl_S1_plt_ord_P2 \
    fifo/rl_S1_selt_ord_P2

( evepy 2 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P2 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P2 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P2 ) & pid6=$!

exec_wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6

