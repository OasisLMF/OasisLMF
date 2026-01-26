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

find fifo/ \( -name '*P1[^0-9]*' -o -name '*P1' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
mkdir -p work/il_S3_summary_palt
mkdir -p work/il_S3_summary_altmeanonly

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_elt_ord_P1
mkfifo fifo/il_S1_selt_ord_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_plt_ord_P1
mkfifo fifo/il_S3_summary_P1
mkfifo fifo/il_S3_summary_P1.idx



# --- Do insured loss computes ---

eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 & pid1=$!
eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 & pid2=$!
pltpy -E bin  -s work/kat/il_S2_plt_sample_P1 -q work/kat/il_S2_plt_quantile_P1 -m work/kat/il_S2_plt_moment_P1 < fifo/il_S2_plt_ord_P1 & pid3=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 > /dev/null & pid4=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_plt_ord_P1 > /dev/null & pid5=$!
tee < fifo/il_S3_summary_P1 work/il_S3_summary_palt/P1.bin work/il_S3_summary_altmeanonly/P1.bin > /dev/null & pid6=$!
tee < fifo/il_S3_summary_P1.idx work/il_S3_summary_palt/P1.idx > /dev/null & pid7=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 -3 fifo/il_S3_summary_P1 < fifo/il_P1 &

( evepy 1 1 | gulmc --socket-server='False' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P1  ) & pid8=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

