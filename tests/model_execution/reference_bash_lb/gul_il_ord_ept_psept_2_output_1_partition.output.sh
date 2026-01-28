#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---


# --- Do ground up loss kats ---


lecpy -r -Kil_S1_summaryleccalc -F -S -s -M -m -W -w -O output/il_S1_ept.csv -o output/il_S1_psept.csv & lpid1=$!
lecpy -r -Kil_S2_summaryleccalc -F -S -s -M -m -W -w -O output/il_S2_ept.csv -o output/il_S2_psept.csv & lpid2=$!
lecpy -r -Kgul_S1_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S1_ept.csv -o output/gul_S1_psept.csv & lpid3=$!
lecpy -r -Kgul_S2_summaryleccalc -F -S -s -M -m -W -w -O output/gul_S2_ept.csv -o output/gul_S2_psept.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*
