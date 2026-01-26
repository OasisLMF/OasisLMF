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
mkfifo fifo/gul_P2
mkfifo fifo/gul_P3
mkfifo fifo/gul_P4
mkfifo fifo/gul_P5
mkfifo fifo/gul_P6
mkfifo fifo/gul_P7
mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_plt_ord_P1
mkfifo fifo/gul_S1_elt_ord_P1
mkfifo fifo/gul_S1_selt_ord_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_plt_ord_P2
mkfifo fifo/gul_S1_elt_ord_P2
mkfifo fifo/gul_S1_selt_ord_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_plt_ord_P3
mkfifo fifo/gul_S1_elt_ord_P3
mkfifo fifo/gul_S1_selt_ord_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx
mkfifo fifo/gul_S1_plt_ord_P4
mkfifo fifo/gul_S1_elt_ord_P4
mkfifo fifo/gul_S1_selt_ord_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_plt_ord_P5
mkfifo fifo/gul_S1_elt_ord_P5
mkfifo fifo/gul_S1_selt_ord_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx
mkfifo fifo/gul_S1_plt_ord_P6
mkfifo fifo/gul_S1_elt_ord_P6
mkfifo fifo/gul_S1_selt_ord_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx
mkfifo fifo/gul_S1_plt_ord_P7
mkfifo fifo/gul_S1_elt_ord_P7
mkfifo fifo/gul_S1_selt_ord_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx
mkfifo fifo/gul_S1_plt_ord_P8
mkfifo fifo/gul_S1_elt_ord_P8
mkfifo fifo/gul_S1_selt_ord_P8

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4
mkfifo fifo/il_P5
mkfifo fifo/il_P6
mkfifo fifo/il_P7
mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_plt_ord_P1
mkfifo fifo/il_S1_elt_ord_P1
mkfifo fifo/il_S1_selt_ord_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_plt_ord_P2
mkfifo fifo/il_S1_elt_ord_P2
mkfifo fifo/il_S1_selt_ord_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx
mkfifo fifo/il_S1_plt_ord_P3
mkfifo fifo/il_S1_elt_ord_P3
mkfifo fifo/il_S1_selt_ord_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summary_P4.idx
mkfifo fifo/il_S1_plt_ord_P4
mkfifo fifo/il_S1_elt_ord_P4
mkfifo fifo/il_S1_selt_ord_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx
mkfifo fifo/il_S1_plt_ord_P5
mkfifo fifo/il_S1_elt_ord_P5
mkfifo fifo/il_S1_selt_ord_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx
mkfifo fifo/il_S1_plt_ord_P6
mkfifo fifo/il_S1_elt_ord_P6
mkfifo fifo/il_S1_selt_ord_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx
mkfifo fifo/il_S1_plt_ord_P7
mkfifo fifo/il_S1_elt_ord_P7
mkfifo fifo/il_S1_selt_ord_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summary_P8.idx
mkfifo fifo/il_S1_plt_ord_P8
mkfifo fifo/il_S1_elt_ord_P8
mkfifo fifo/il_S1_selt_ord_P8



# --- Do insured loss computes ---

pltpy -E bin  -s work/kat/il_S1_plt_sample_P1 -q work/kat/il_S1_plt_quantile_P1 -m work/kat/il_S1_plt_moment_P1 < fifo/il_S1_plt_ord_P1 & pid1=$!
eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 & pid2=$!
eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 & pid3=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P2 -q work/kat/il_S1_plt_quantile_P2 -m work/kat/il_S1_plt_moment_P2 < fifo/il_S1_plt_ord_P2 & pid4=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P2 -m work/kat/il_S1_elt_moment_P2 < fifo/il_S1_elt_ord_P2 & pid5=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P2 < fifo/il_S1_selt_ord_P2 & pid6=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P3 -q work/kat/il_S1_plt_quantile_P3 -m work/kat/il_S1_plt_moment_P3 < fifo/il_S1_plt_ord_P3 & pid7=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P3 -m work/kat/il_S1_elt_moment_P3 < fifo/il_S1_elt_ord_P3 & pid8=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P3 < fifo/il_S1_selt_ord_P3 & pid9=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P4 -q work/kat/il_S1_plt_quantile_P4 -m work/kat/il_S1_plt_moment_P4 < fifo/il_S1_plt_ord_P4 & pid10=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P4 -m work/kat/il_S1_elt_moment_P4 < fifo/il_S1_elt_ord_P4 & pid11=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P4 < fifo/il_S1_selt_ord_P4 & pid12=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P5 -q work/kat/il_S1_plt_quantile_P5 -m work/kat/il_S1_plt_moment_P5 < fifo/il_S1_plt_ord_P5 & pid13=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P5 -m work/kat/il_S1_elt_moment_P5 < fifo/il_S1_elt_ord_P5 & pid14=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P5 < fifo/il_S1_selt_ord_P5 & pid15=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P6 -q work/kat/il_S1_plt_quantile_P6 -m work/kat/il_S1_plt_moment_P6 < fifo/il_S1_plt_ord_P6 & pid16=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P6 -m work/kat/il_S1_elt_moment_P6 < fifo/il_S1_elt_ord_P6 & pid17=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P6 < fifo/il_S1_selt_ord_P6 & pid18=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P7 -q work/kat/il_S1_plt_quantile_P7 -m work/kat/il_S1_plt_moment_P7 < fifo/il_S1_plt_ord_P7 & pid19=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P7 -m work/kat/il_S1_elt_moment_P7 < fifo/il_S1_elt_ord_P7 & pid20=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P7 < fifo/il_S1_selt_ord_P7 & pid21=$!
pltpy -E bin  -H -s work/kat/il_S1_plt_sample_P8 -q work/kat/il_S1_plt_quantile_P8 -m work/kat/il_S1_plt_moment_P8 < fifo/il_S1_plt_ord_P8 & pid22=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P8 -m work/kat/il_S1_elt_moment_P8 < fifo/il_S1_elt_ord_P8 & pid23=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P8 < fifo/il_S1_selt_ord_P8 & pid24=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_plt_ord_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 work/il_S1_summary_palt/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid25=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid26=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_plt_ord_P2 fifo/il_S1_elt_ord_P2 fifo/il_S1_selt_ord_P2 work/il_S1_summary_palt/P2.bin work/il_S1_summary_altmeanonly/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid27=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summary_palt/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid28=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_plt_ord_P3 fifo/il_S1_elt_ord_P3 fifo/il_S1_selt_ord_P3 work/il_S1_summary_palt/P3.bin work/il_S1_summary_altmeanonly/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid29=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summary_palt/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid30=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_plt_ord_P4 fifo/il_S1_elt_ord_P4 fifo/il_S1_selt_ord_P4 work/il_S1_summary_palt/P4.bin work/il_S1_summary_altmeanonly/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid31=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summary_palt/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid32=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_plt_ord_P5 fifo/il_S1_elt_ord_P5 fifo/il_S1_selt_ord_P5 work/il_S1_summary_palt/P5.bin work/il_S1_summary_altmeanonly/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid33=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summary_palt/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid34=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_plt_ord_P6 fifo/il_S1_elt_ord_P6 fifo/il_S1_selt_ord_P6 work/il_S1_summary_palt/P6.bin work/il_S1_summary_altmeanonly/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid35=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summary_palt/P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid36=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_plt_ord_P7 fifo/il_S1_elt_ord_P7 fifo/il_S1_selt_ord_P7 work/il_S1_summary_palt/P7.bin work/il_S1_summary_altmeanonly/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid37=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summary_palt/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid38=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_plt_ord_P8 fifo/il_S1_elt_ord_P8 fifo/il_S1_selt_ord_P8 work/il_S1_summary_palt/P8.bin work/il_S1_summary_altmeanonly/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid39=$!
tee < fifo/il_S1_summary_P8.idx work/il_S1_summary_palt/P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid40=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarypy -m -t il  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarypy -m -t il  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarypy -m -t il  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarypy -m -t il  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarypy -m -t il  -1 fifo/il_S1_summary_P8 < fifo/il_P8 &

# --- Do ground up loss computes ---

pltpy -E bin  -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < fifo/gul_S1_plt_ord_P1 & pid41=$!
eltpy -E bin  -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < fifo/gul_S1_elt_ord_P1 & pid42=$!
eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 & pid43=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P2 -q work/kat/gul_S1_plt_quantile_P2 -m work/kat/gul_S1_plt_moment_P2 < fifo/gul_S1_plt_ord_P2 & pid44=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P2 -m work/kat/gul_S1_elt_moment_P2 < fifo/gul_S1_elt_ord_P2 & pid45=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P2 < fifo/gul_S1_selt_ord_P2 & pid46=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P3 -q work/kat/gul_S1_plt_quantile_P3 -m work/kat/gul_S1_plt_moment_P3 < fifo/gul_S1_plt_ord_P3 & pid47=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P3 -m work/kat/gul_S1_elt_moment_P3 < fifo/gul_S1_elt_ord_P3 & pid48=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P3 < fifo/gul_S1_selt_ord_P3 & pid49=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P4 -q work/kat/gul_S1_plt_quantile_P4 -m work/kat/gul_S1_plt_moment_P4 < fifo/gul_S1_plt_ord_P4 & pid50=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P4 -m work/kat/gul_S1_elt_moment_P4 < fifo/gul_S1_elt_ord_P4 & pid51=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P4 < fifo/gul_S1_selt_ord_P4 & pid52=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P5 -q work/kat/gul_S1_plt_quantile_P5 -m work/kat/gul_S1_plt_moment_P5 < fifo/gul_S1_plt_ord_P5 & pid53=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P5 -m work/kat/gul_S1_elt_moment_P5 < fifo/gul_S1_elt_ord_P5 & pid54=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P5 < fifo/gul_S1_selt_ord_P5 & pid55=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P6 -q work/kat/gul_S1_plt_quantile_P6 -m work/kat/gul_S1_plt_moment_P6 < fifo/gul_S1_plt_ord_P6 & pid56=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P6 -m work/kat/gul_S1_elt_moment_P6 < fifo/gul_S1_elt_ord_P6 & pid57=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P6 < fifo/gul_S1_selt_ord_P6 & pid58=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P7 -q work/kat/gul_S1_plt_quantile_P7 -m work/kat/gul_S1_plt_moment_P7 < fifo/gul_S1_plt_ord_P7 & pid59=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P7 -m work/kat/gul_S1_elt_moment_P7 < fifo/gul_S1_elt_ord_P7 & pid60=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P7 < fifo/gul_S1_selt_ord_P7 & pid61=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P8 -q work/kat/gul_S1_plt_quantile_P8 -m work/kat/gul_S1_plt_moment_P8 < fifo/gul_S1_plt_ord_P8 & pid62=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P8 -m work/kat/gul_S1_elt_moment_P8 < fifo/gul_S1_elt_ord_P8 & pid63=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P8 < fifo/gul_S1_selt_ord_P8 & pid64=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_plt_ord_P1 fifo/gul_S1_elt_ord_P1 fifo/gul_S1_selt_ord_P1 work/gul_S1_summary_palt/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid65=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid66=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_plt_ord_P2 fifo/gul_S1_elt_ord_P2 fifo/gul_S1_selt_ord_P2 work/gul_S1_summary_palt/P2.bin work/gul_S1_summary_altmeanonly/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid67=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summary_palt/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid68=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_plt_ord_P3 fifo/gul_S1_elt_ord_P3 fifo/gul_S1_selt_ord_P3 work/gul_S1_summary_palt/P3.bin work/gul_S1_summary_altmeanonly/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid69=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summary_palt/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid70=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_plt_ord_P4 fifo/gul_S1_elt_ord_P4 fifo/gul_S1_selt_ord_P4 work/gul_S1_summary_palt/P4.bin work/gul_S1_summary_altmeanonly/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid71=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summary_palt/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid72=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_plt_ord_P5 fifo/gul_S1_elt_ord_P5 fifo/gul_S1_selt_ord_P5 work/gul_S1_summary_palt/P5.bin work/gul_S1_summary_altmeanonly/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid73=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summary_palt/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid74=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_plt_ord_P6 fifo/gul_S1_elt_ord_P6 fifo/gul_S1_selt_ord_P6 work/gul_S1_summary_palt/P6.bin work/gul_S1_summary_altmeanonly/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid75=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summary_palt/P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid76=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_plt_ord_P7 fifo/gul_S1_elt_ord_P7 fifo/gul_S1_selt_ord_P7 work/gul_S1_summary_palt/P7.bin work/gul_S1_summary_altmeanonly/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid77=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summary_palt/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid78=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_plt_ord_P8 fifo/gul_S1_elt_ord_P8 fifo/gul_S1_selt_ord_P8 work/gul_S1_summary_palt/P8.bin work/gul_S1_summary_altmeanonly/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid79=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summary_palt/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid80=$!

summarypy -m -t gul  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 &

( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P1 | fmpy -a2 > fifo/il_P1  ) & pid81=$!
( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P2 | fmpy -a2 > fifo/il_P2  ) & pid82=$!
( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P3 | fmpy -a2 > fifo/il_P3  ) & pid83=$!
( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P4 | fmpy -a2 > fifo/il_P4  ) & pid84=$!
( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P5 | fmpy -a2 > fifo/il_P5  ) & pid85=$!
( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P6 | fmpy -a2 > fifo/il_P6  ) & pid86=$!
( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P7 | fmpy -a2 > fifo/il_P7  ) & pid87=$!
( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P8 | fmpy -a2 > fifo/il_P8  ) & pid88=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88


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
