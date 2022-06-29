#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do reinsurance loss kats ---

kat work/kat/ri_S1_eltcalc_P1 > output/ri_S1_eltcalc.csv & kpid1=$!
kat work/kat/ri_S1_pltcalc_P1 > output/ri_S1_pltcalc.csv & kpid2=$!
kat work/kat/ri_S1_summarycalc_P1 > output/ri_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid4=$!
kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid5=$!
kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid9=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
wait $lpid1 $lpid2 $lpid3

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
