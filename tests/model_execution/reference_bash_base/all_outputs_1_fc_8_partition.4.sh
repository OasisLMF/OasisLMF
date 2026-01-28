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

find fifo/ \( -name '*P5[^0-9]*' -o -name '*P5' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summary_altmeanonly

mkfifo fifo/gul_P5

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_plt_ord_P5
mkfifo fifo/gul_S1_elt_ord_P5
mkfifo fifo/gul_S1_selt_ord_P5

mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx
mkfifo fifo/il_S1_plt_ord_P5
mkfifo fifo/il_S1_elt_ord_P5
mkfifo fifo/il_S1_selt_ord_P5



# --- Do insured loss computes ---
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P5 -q work/kat/il_S1_plt_quantile_P5 -m work/kat/il_S1_plt_moment_P5 < fifo/il_S1_plt_ord_P5 & pid1=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P5 -m work/kat/il_S1_elt_moment_P5 < fifo/il_S1_elt_ord_P5 & pid2=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P5 < fifo/il_S1_selt_ord_P5 & pid3=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_plt_ord_P5 fifo/il_S1_elt_ord_P5 fifo/il_S1_selt_ord_P5 work/il_S1_summary_palt/P5.bin work/il_S1_summary_altmeanonly/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summary_palt/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid5=$!
summarypy -m -t il  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &

# --- Do ground up loss computes ---
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P5 -q work/kat/gul_S1_plt_quantile_P5 -m work/kat/gul_S1_plt_moment_P5 < fifo/gul_S1_plt_ord_P5 & pid6=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P5 -m work/kat/gul_S1_elt_moment_P5 < fifo/gul_S1_elt_ord_P5 & pid7=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P5 < fifo/gul_S1_selt_ord_P5 & pid8=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_plt_ord_P5 fifo/gul_S1_elt_ord_P5 fifo/gul_S1_selt_ord_P5 work/gul_S1_summary_palt/P5.bin work/gul_S1_summary_altmeanonly/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summary_palt/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid10=$!
summarypy -m -t gul  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 &

( evepy 5 8 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P5 | fmpy -a2 > fifo/il_P5  ) & pid11=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11

