#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 work/full_correlation/kat/il_S1_summarycalc_P2 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid4=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -F output/full_correlation/il_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/il_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/il_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/il_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/il_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/il_S1_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid7=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8

rm -R -f work/*
rm -R -f fifo/*
