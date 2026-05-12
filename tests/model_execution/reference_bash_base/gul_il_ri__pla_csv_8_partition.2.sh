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
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summary_altmeanonly

mkfifo fifo/gul_P3

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_plt_ord_P3
mkfifo fifo/gul_S1_elt_ord_P3
mkfifo fifo/gul_S1_selt_ord_P3
mkfifo fifo/il_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx
mkfifo fifo/il_S1_plt_ord_P3
mkfifo fifo/il_S1_elt_ord_P3
mkfifo fifo/il_S1_selt_ord_P3


# --- Do insured loss computes ---
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P3 -q work/kat/il_S1_plt_quantile_P3 -m work/kat/il_S1_plt_moment_P3 < fifo/il_S1_plt_ord_P3 & pid1=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P3 -m work/kat/il_S1_elt_moment_P3 < fifo/il_S1_elt_ord_P3 & pid2=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P3 < fifo/il_S1_selt_ord_P3 & pid3=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_plt_ord_P3 fifo/il_S1_elt_ord_P3 fifo/il_S1_selt_ord_P3 work/il_S1_summary_palt/P3.bin work/il_S1_summary_altmeanonly/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summary_palt/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid5=$!
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &

# --- Do ground up loss computes ---
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P3 -q work/kat/gul_S1_plt_quantile_P3 -m work/kat/gul_S1_plt_moment_P3 < fifo/gul_S1_plt_ord_P3 & pid6=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P3 -m work/kat/gul_S1_elt_moment_P3 < fifo/gul_S1_elt_ord_P3 & pid7=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P3 < fifo/gul_S1_selt_ord_P3 & pid8=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_plt_ord_P3 fifo/gul_S1_elt_ord_P3 fifo/gul_S1_selt_ord_P3 work/gul_S1_summary_palt/P3.bin work/gul_S1_summary_altmeanonly/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summary_palt/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid10=$!
summarypy -m -t gul  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &


# --- Verify FIFO pipes ---
check_fifos \
    fifo/gul_P3 \
    fifo/gul_S1_summary_P3 \
    fifo/gul_S1_summary_P3.idx \
    fifo/gul_S1_plt_ord_P3 \
    fifo/gul_S1_elt_ord_P3 \
    fifo/gul_S1_selt_ord_P3 \
    fifo/il_P3 \
    fifo/il_S1_summary_P3 \
    fifo/il_S1_summary_P3.idx \
    fifo/il_S1_plt_ord_P3 \
    fifo/il_S1_elt_ord_P3 \
    fifo/il_S1_selt_ord_P3

( evepy 3 8 | gulmc --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | tee fifo/gul_P3 | fmpy -a2 > fifo/il_P3  ) & pid11=$!

exec_wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11

