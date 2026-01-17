#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---

katpy -S -f bin -i work/kat/il_S1_plt_sample_P1 -o output/il_S1_splt.parquet & kpid1=$!
katpy -Q -f bin -i work/kat/il_S1_plt_quantile_P1 -o output/il_S1_qplt.parquet & kpid2=$!
katpy -M -f bin -i work/kat/il_S1_plt_moment_P1 -o output/il_S1_mplt.parquet & kpid3=$!
katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 -o output/il_S1_qelt.parquet & kpid4=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 -o output/il_S1_melt.parquet & kpid5=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 -o output/il_S1_selt.parquet & kpid6=$!

# --- Do ground up loss kats ---

katpy -S -f bin -i work/kat/gul_S1_plt_sample_P1 -o output/gul_S1_splt.parquet & kpid7=$!
katpy -Q -f bin -i work/kat/gul_S1_plt_quantile_P1 -o output/gul_S1_qplt.parquet & kpid8=$!
katpy -M -f bin -i work/kat/gul_S1_plt_moment_P1 -o output/gul_S1_mplt.parquet & kpid9=$!
katpy -q -f bin -i work/kat/gul_S1_elt_quantile_P1 -o output/gul_S1_qelt.parquet & kpid10=$!
katpy -m -f bin -i work/kat/gul_S1_elt_moment_P1 -o output/gul_S1_melt.parquet & kpid11=$!
katpy -s -f bin -i work/kat/gul_S1_elt_sample_P1 -o output/gul_S1_selt.parquet & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalpy -Kil_S1_summary_palt -c output/il_S1_alct.parquet -l 0.95 -E parquet -a output/il_S1_palt.parquet & lpid1=$!
aalpy -Kil_S1_summary_altmeanonly -E parquet -a output/il_S1_altmeanonly.cparquetsv & lpid2=$!
lecpy -r -Kil_S1_summaryleccalc -F -f -S -s -M -m -W -w -E parquet -O output/il_S1_ept.parquet -o output/il_S1_psept.parquet & lpid3=$!
aalpy -Kgul_S1_summary_palt -c output/gul_S1_alct.parquet -l 0.95 -E parquet -a output/gul_S1_palt.parquet & lpid4=$!
aalpy -Kgul_S1_summary_altmeanonly -E parquet -a output/gul_S1_altmeanonly.cparquetsv & lpid5=$!
lecpy -r -Kgul_S1_summaryleccalc -F -f -S -s -M -m -W -w -E parquet -O output/gul_S1_ept.parquet -o output/gul_S1_psept.parquet & lpid6=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6

rm -R -f work/*
rm -R -f fifo/*
