#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---

katpy -S -f bin -i work/kat/il_S1_plt_sample_P1 work/kat/il_S1_plt_sample_P2 work/kat/il_S1_plt_sample_P3 work/kat/il_S1_plt_sample_P4 work/kat/il_S1_plt_sample_P5 work/kat/il_S1_plt_sample_P6 work/kat/il_S1_plt_sample_P7 work/kat/il_S1_plt_sample_P8 -o output/il_S1_splt.csv & kpid1=$!
katpy -Q -f bin -i work/kat/il_S1_plt_quantile_P1 work/kat/il_S1_plt_quantile_P2 work/kat/il_S1_plt_quantile_P3 work/kat/il_S1_plt_quantile_P4 work/kat/il_S1_plt_quantile_P5 work/kat/il_S1_plt_quantile_P6 work/kat/il_S1_plt_quantile_P7 work/kat/il_S1_plt_quantile_P8 -o output/il_S1_qplt.csv & kpid2=$!
katpy -M -f bin -i work/kat/il_S1_plt_moment_P1 work/kat/il_S1_plt_moment_P2 work/kat/il_S1_plt_moment_P3 work/kat/il_S1_plt_moment_P4 work/kat/il_S1_plt_moment_P5 work/kat/il_S1_plt_moment_P6 work/kat/il_S1_plt_moment_P7 work/kat/il_S1_plt_moment_P8 -o output/il_S1_mplt.csv & kpid3=$!
katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 work/kat/il_S1_elt_quantile_P2 work/kat/il_S1_elt_quantile_P3 work/kat/il_S1_elt_quantile_P4 work/kat/il_S1_elt_quantile_P5 work/kat/il_S1_elt_quantile_P6 work/kat/il_S1_elt_quantile_P7 work/kat/il_S1_elt_quantile_P8 -o output/il_S1_qelt.csv & kpid4=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 work/kat/il_S1_elt_moment_P2 work/kat/il_S1_elt_moment_P3 work/kat/il_S1_elt_moment_P4 work/kat/il_S1_elt_moment_P5 work/kat/il_S1_elt_moment_P6 work/kat/il_S1_elt_moment_P7 work/kat/il_S1_elt_moment_P8 -o output/il_S1_melt.csv & kpid5=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 work/kat/il_S1_elt_sample_P2 work/kat/il_S1_elt_sample_P3 work/kat/il_S1_elt_sample_P4 work/kat/il_S1_elt_sample_P5 work/kat/il_S1_elt_sample_P6 work/kat/il_S1_elt_sample_P7 work/kat/il_S1_elt_sample_P8 -o output/il_S1_selt.csv & kpid6=$!

# --- Do ground up loss kats ---

katpy -S -f bin -i work/kat/gul_S1_plt_sample_P1 work/kat/gul_S1_plt_sample_P2 work/kat/gul_S1_plt_sample_P3 work/kat/gul_S1_plt_sample_P4 work/kat/gul_S1_plt_sample_P5 work/kat/gul_S1_plt_sample_P6 work/kat/gul_S1_plt_sample_P7 work/kat/gul_S1_plt_sample_P8 -o output/gul_S1_splt.csv & kpid7=$!
katpy -Q -f bin -i work/kat/gul_S1_plt_quantile_P1 work/kat/gul_S1_plt_quantile_P2 work/kat/gul_S1_plt_quantile_P3 work/kat/gul_S1_plt_quantile_P4 work/kat/gul_S1_plt_quantile_P5 work/kat/gul_S1_plt_quantile_P6 work/kat/gul_S1_plt_quantile_P7 work/kat/gul_S1_plt_quantile_P8 -o output/gul_S1_qplt.csv & kpid8=$!
katpy -M -f bin -i work/kat/gul_S1_plt_moment_P1 work/kat/gul_S1_plt_moment_P2 work/kat/gul_S1_plt_moment_P3 work/kat/gul_S1_plt_moment_P4 work/kat/gul_S1_plt_moment_P5 work/kat/gul_S1_plt_moment_P6 work/kat/gul_S1_plt_moment_P7 work/kat/gul_S1_plt_moment_P8 -o output/gul_S1_mplt.csv & kpid9=$!
katpy -q -f bin -i work/kat/gul_S1_elt_quantile_P1 work/kat/gul_S1_elt_quantile_P2 work/kat/gul_S1_elt_quantile_P3 work/kat/gul_S1_elt_quantile_P4 work/kat/gul_S1_elt_quantile_P5 work/kat/gul_S1_elt_quantile_P6 work/kat/gul_S1_elt_quantile_P7 work/kat/gul_S1_elt_quantile_P8 -o output/gul_S1_qelt.csv & kpid10=$!
katpy -m -f bin -i work/kat/gul_S1_elt_moment_P1 work/kat/gul_S1_elt_moment_P2 work/kat/gul_S1_elt_moment_P3 work/kat/gul_S1_elt_moment_P4 work/kat/gul_S1_elt_moment_P5 work/kat/gul_S1_elt_moment_P6 work/kat/gul_S1_elt_moment_P7 work/kat/gul_S1_elt_moment_P8 -o output/gul_S1_melt.csv & kpid11=$!
katpy -s -f bin -i work/kat/gul_S1_elt_sample_P1 work/kat/gul_S1_elt_sample_P2 work/kat/gul_S1_elt_sample_P3 work/kat/gul_S1_elt_sample_P4 work/kat/gul_S1_elt_sample_P5 work/kat/gul_S1_elt_sample_P6 work/kat/gul_S1_elt_sample_P7 work/kat/gul_S1_elt_sample_P8 -o output/gul_S1_selt.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalpy -Kil_S1_summary_palt -c output/il_S1_alct.csv -l 0.95 -a output/il_S1_palt.csv & lpid1=$!
aalpy -Kil_S1_summary_altmeanonly -a output/il_S1_altmeanonly.csv & lpid2=$!
lecpy -r -Kil_S1_summaryleccalc -F -f -S -s -M -m -W -w -O output/il_S1_ept.csv -o output/il_S1_psept.csv & lpid3=$!
aalpy -Kgul_S1_summary_palt -c output/gul_S1_alct.csv -l 0.95 -a output/gul_S1_palt.csv & lpid4=$!
aalpy -Kgul_S1_summary_altmeanonly -a output/gul_S1_altmeanonly.csv & lpid5=$!
lecpy -r -Kgul_S1_summaryleccalc -F -f -S -s -M -m -W -w -O output/gul_S1_ept.csv -o output/gul_S1_psept.csv & lpid6=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6

rm -R -f work/*
rm -R -f fifo/*
