#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---

katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 -o output/il_S1_qelt.csv & kpid1=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 -o output/il_S1_melt.csv & kpid2=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 -o output/il_S1_selt.csv & kpid3=$!
katpy -S -f bin -i work/kat/il_S2_plt_sample_P1 -o output/il_S2_splt.csv & kpid4=$!
katpy -Q -f bin -i work/kat/il_S2_plt_quantile_P1 -o output/il_S2_qplt.csv & kpid5=$!
katpy -M -f bin -i work/kat/il_S2_plt_moment_P1 -o output/il_S2_mplt.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


aalpy -Kil_S3_summary_palt -c output/il_S3_alct.csv -l 0.95 -a output/il_S3_palt.csv & lpid1=$!
aalpy -Kil_S3_summary_altmeanonly -a output/il_S3_altmeanonly.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
