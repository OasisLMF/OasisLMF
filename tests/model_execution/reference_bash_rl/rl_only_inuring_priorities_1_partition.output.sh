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

# --- Do reinsurance gross loss kats ---

katpy -S -f bin -i work/kat/rl_IP1_S1_plt_sample_P1 -o output/rl_IP1_S1_splt.csv & kpid1=$!
katpy -s -f bin -i work/kat/rl_IP1_S1_elt_sample_P1 -o output/rl_IP1_S1_selt.csv & kpid2=$!
katpy -S -f bin -i work/kat/rl_S1_plt_sample_P1 -o output/rl_S1_splt.csv & kpid3=$!
katpy -s -f bin -i work/kat/rl_S1_elt_sample_P1 -o output/rl_S1_selt.csv & kpid4=$!
exec_wait $kpid1 $kpid2 $kpid3 $kpid4


aalpy -Krl_IP1_S1_summary_palt -a output/rl_IP1_S1_palt.csv & lpid1=$!
lecpy -r -Krl_IP1_S1_summaryleccalc -F -f -O output/rl_IP1_S1_ept.csv & lpid2=$!
aalpy -Krl_S1_summary_palt -a output/rl_S1_palt.csv & lpid3=$!
lecpy -r -Krl_S1_summaryleccalc -F -f -O output/rl_S1_ept.csv & lpid4=$!
exec_wait $lpid1 $lpid2 $lpid3 $lpid4


# --- removing working files
rm -R -f work/*
rm -R -f fifo/*
