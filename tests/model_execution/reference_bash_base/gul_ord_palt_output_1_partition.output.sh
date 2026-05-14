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

# --- Do ground up loss kats ---


aalpy -Kgul_S1_summary_palt -a output/gul_S1_palt.csv & lpid1=$!
aalpy -Kgul_S2_summary_palt -a output/gul_S2_palt.csv & lpid2=$!
exec_wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
