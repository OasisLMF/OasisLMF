#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summary_altmeanonly

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_plt_ord_P1
mkfifo fifo/gul_S1_elt_ord_P1
mkfifo fifo/gul_S1_selt_ord_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_plt_ord_P1
mkfifo fifo/il_S1_elt_ord_P1
mkfifo fifo/il_S1_selt_ord_P1



# --- Do insured loss computes ---


pltpy -E bin  -s work/kat/il_S1_plt_sample_P1 -q work/kat/il_S1_plt_quantile_P1 -m work/kat/il_S1_plt_moment_P1 < fifo/il_S1_plt_ord_P1 & pid1=$!
eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 & pid2=$!
eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 & pid3=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_plt_ord_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 work/il_S1_summary_palt/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid5=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

# --- Do ground up loss computes ---


pltpy -E bin  -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < fifo/gul_S1_plt_ord_P1 & pid6=$!
eltpy -E bin  -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < fifo/gul_S1_elt_ord_P1 & pid7=$!
eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 & pid8=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_plt_ord_P1 fifo/gul_S1_elt_ord_P1 fifo/gul_S1_selt_ord_P1 work/gul_S1_summary_palt/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid10=$!

summarypy -m -t gul  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

( evepy 1 1 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | tee fifo/gul_P1 | fmpy -a2 > fifo/il_P1  ) & pid11=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11


# --- Do insured loss kats ---

katpy -S -f bin -i work/kat/il_S1_plt_sample_P1 -o output/il_S1_splt.csv & kpid1=$!
katpy -Q -f bin -i work/kat/il_S1_plt_quantile_P1 -o output/il_S1_qplt.csv & kpid2=$!
katpy -M -f bin -i work/kat/il_S1_plt_moment_P1 -o output/il_S1_mplt.csv & kpid3=$!
katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 -o output/il_S1_qelt.csv & kpid4=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 -o output/il_S1_melt.csv & kpid5=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 -o output/il_S1_selt.csv & kpid6=$!

# --- Do ground up loss kats ---

katpy -S -f bin -i work/kat/gul_S1_plt_sample_P1 -o output/gul_S1_splt.csv & kpid7=$!
katpy -Q -f bin -i work/kat/gul_S1_plt_quantile_P1 -o output/gul_S1_qplt.csv & kpid8=$!
katpy -M -f bin -i work/kat/gul_S1_plt_moment_P1 -o output/gul_S1_mplt.csv & kpid9=$!
katpy -q -f bin -i work/kat/gul_S1_elt_quantile_P1 -o output/gul_S1_qelt.csv & kpid10=$!
katpy -m -f bin -i work/kat/gul_S1_elt_moment_P1 -o output/gul_S1_melt.csv & kpid11=$!
katpy -s -f bin -i work/kat/gul_S1_elt_sample_P1 -o output/gul_S1_selt.csv & kpid12=$!
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
