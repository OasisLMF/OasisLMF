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
mkdir -p work/il_S3_summary_palt
mkdir -p work/il_S3_summary_altmeanonly

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_elt_ord_P1
mkfifo fifo/il_S1_selt_ord_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_plt_ord_P1
mkfifo fifo/il_S3_summary_P1
mkfifo fifo/il_S3_summary_P1.idx

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_elt_ord_P2
mkfifo fifo/il_S1_selt_ord_P2
mkfifo fifo/il_S2_summary_P2
mkfifo fifo/il_S2_plt_ord_P2
mkfifo fifo/il_S3_summary_P2
mkfifo fifo/il_S3_summary_P2.idx

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_elt_ord_P3
mkfifo fifo/il_S1_selt_ord_P3
mkfifo fifo/il_S2_summary_P3
mkfifo fifo/il_S2_plt_ord_P3
mkfifo fifo/il_S3_summary_P3
mkfifo fifo/il_S3_summary_P3.idx

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_elt_ord_P4
mkfifo fifo/il_S1_selt_ord_P4
mkfifo fifo/il_S2_summary_P4
mkfifo fifo/il_S2_plt_ord_P4
mkfifo fifo/il_S3_summary_P4
mkfifo fifo/il_S3_summary_P4.idx

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2
mkfifo fifo/gul_lb_P3
mkfifo fifo/gul_lb_P4

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2
mkfifo fifo/lb_il_P3
mkfifo fifo/lb_il_P4



# --- Do insured loss computes ---

eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 & pid1=$!
eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 & pid2=$!
pltpy -E bin  -s work/kat/il_S2_plt_sample_P1 -q work/kat/il_S2_plt_quantile_P1 -m work/kat/il_S2_plt_moment_P1 < fifo/il_S2_plt_ord_P1 & pid3=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P2 -m work/kat/il_S1_elt_moment_P2 < fifo/il_S1_elt_ord_P2 & pid4=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P2 < fifo/il_S1_selt_ord_P2 & pid5=$!
pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P2 -q work/kat/il_S2_plt_quantile_P2 -m work/kat/il_S2_plt_moment_P2 < fifo/il_S2_plt_ord_P2 & pid6=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P3 -m work/kat/il_S1_elt_moment_P3 < fifo/il_S1_elt_ord_P3 & pid7=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P3 < fifo/il_S1_selt_ord_P3 & pid8=$!
pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P3 -q work/kat/il_S2_plt_quantile_P3 -m work/kat/il_S2_plt_moment_P3 < fifo/il_S2_plt_ord_P3 & pid9=$!
eltpy -E bin  -H -q work/kat/il_S1_elt_quantile_P4 -m work/kat/il_S1_elt_moment_P4 < fifo/il_S1_elt_ord_P4 & pid10=$!
eltpy -E bin  -H -s work/kat/il_S1_elt_sample_P4 < fifo/il_S1_selt_ord_P4 & pid11=$!
pltpy -E bin  -H -s work/kat/il_S2_plt_sample_P4 -q work/kat/il_S2_plt_quantile_P4 -m work/kat/il_S2_plt_moment_P4 < fifo/il_S2_plt_ord_P4 & pid12=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 > /dev/null & pid13=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_plt_ord_P1 > /dev/null & pid14=$!
tee < fifo/il_S3_summary_P1 work/il_S3_summary_palt/P1.bin work/il_S3_summary_altmeanonly/P1.bin > /dev/null & pid15=$!
tee < fifo/il_S3_summary_P1.idx work/il_S3_summary_palt/P1.idx > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_elt_ord_P2 fifo/il_S1_selt_ord_P2 > /dev/null & pid17=$!
tee < fifo/il_S2_summary_P2 fifo/il_S2_plt_ord_P2 > /dev/null & pid18=$!
tee < fifo/il_S3_summary_P2 work/il_S3_summary_palt/P2.bin work/il_S3_summary_altmeanonly/P2.bin > /dev/null & pid19=$!
tee < fifo/il_S3_summary_P2.idx work/il_S3_summary_palt/P2.idx > /dev/null & pid20=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_elt_ord_P3 fifo/il_S1_selt_ord_P3 > /dev/null & pid21=$!
tee < fifo/il_S2_summary_P3 fifo/il_S2_plt_ord_P3 > /dev/null & pid22=$!
tee < fifo/il_S3_summary_P3 work/il_S3_summary_palt/P3.bin work/il_S3_summary_altmeanonly/P3.bin > /dev/null & pid23=$!
tee < fifo/il_S3_summary_P3.idx work/il_S3_summary_palt/P3.idx > /dev/null & pid24=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_elt_ord_P4 fifo/il_S1_selt_ord_P4 > /dev/null & pid25=$!
tee < fifo/il_S2_summary_P4 fifo/il_S2_plt_ord_P4 > /dev/null & pid26=$!
tee < fifo/il_S3_summary_P4 work/il_S3_summary_palt/P4.bin work/il_S3_summary_altmeanonly/P4.bin > /dev/null & pid27=$!
tee < fifo/il_S3_summary_P4.idx work/il_S3_summary_palt/P4.idx > /dev/null & pid28=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 -3 fifo/il_S3_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 -2 fifo/il_S2_summary_P2 -3 fifo/il_S3_summary_P2 < fifo/il_P2 &
summarypy -m -t il  -1 fifo/il_S1_summary_P3 -2 fifo/il_S2_summary_P3 -3 fifo/il_S3_summary_P3 < fifo/il_P3 &
summarypy -m -t il  -1 fifo/il_S1_summary_P4 -2 fifo/il_S2_summary_P4 -3 fifo/il_S3_summary_P4 < fifo/il_P4 &

( evepy 1 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P1  ) & 
( evepy 2 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P2  ) & 
( evepy 3 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P3  ) & 
( evepy 4 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_lb_P4  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
load_balancer -i fifo/gul_lb_P3 fifo/gul_lb_P4 -o fifo/lb_il_P3 fifo/lb_il_P4 &
( fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 ) & pid29=$!
( fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 ) & pid30=$!
( fmpy -a2 < fifo/lb_il_P3 > fifo/il_P3 ) & pid31=$!
( fmpy -a2 < fifo/lb_il_P4 > fifo/il_P4 ) & pid32=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32


# --- Do insured loss kats ---

katpy -q -f bin -i work/kat/il_S1_elt_quantile_P1 work/kat/il_S1_elt_quantile_P2 work/kat/il_S1_elt_quantile_P3 work/kat/il_S1_elt_quantile_P4 work/kat/il_S1_elt_quantile_P5 work/kat/il_S1_elt_quantile_P6 work/kat/il_S1_elt_quantile_P7 work/kat/il_S1_elt_quantile_P8 -o output/il_S1_qelt.csv & kpid1=$!
katpy -m -f bin -i work/kat/il_S1_elt_moment_P1 work/kat/il_S1_elt_moment_P2 work/kat/il_S1_elt_moment_P3 work/kat/il_S1_elt_moment_P4 work/kat/il_S1_elt_moment_P5 work/kat/il_S1_elt_moment_P6 work/kat/il_S1_elt_moment_P7 work/kat/il_S1_elt_moment_P8 -o output/il_S1_melt.csv & kpid2=$!
katpy -s -f bin -i work/kat/il_S1_elt_sample_P1 work/kat/il_S1_elt_sample_P2 work/kat/il_S1_elt_sample_P3 work/kat/il_S1_elt_sample_P4 work/kat/il_S1_elt_sample_P5 work/kat/il_S1_elt_sample_P6 work/kat/il_S1_elt_sample_P7 work/kat/il_S1_elt_sample_P8 -o output/il_S1_selt.csv & kpid3=$!
katpy -S -f bin -i work/kat/il_S2_plt_sample_P1 work/kat/il_S2_plt_sample_P2 work/kat/il_S2_plt_sample_P3 work/kat/il_S2_plt_sample_P4 work/kat/il_S2_plt_sample_P5 work/kat/il_S2_plt_sample_P6 work/kat/il_S2_plt_sample_P7 work/kat/il_S2_plt_sample_P8 -o output/il_S2_splt.csv & kpid4=$!
katpy -Q -f bin -i work/kat/il_S2_plt_quantile_P1 work/kat/il_S2_plt_quantile_P2 work/kat/il_S2_plt_quantile_P3 work/kat/il_S2_plt_quantile_P4 work/kat/il_S2_plt_quantile_P5 work/kat/il_S2_plt_quantile_P6 work/kat/il_S2_plt_quantile_P7 work/kat/il_S2_plt_quantile_P8 -o output/il_S2_qplt.csv & kpid5=$!
katpy -M -f bin -i work/kat/il_S2_plt_moment_P1 work/kat/il_S2_plt_moment_P2 work/kat/il_S2_plt_moment_P3 work/kat/il_S2_plt_moment_P4 work/kat/il_S2_plt_moment_P5 work/kat/il_S2_plt_moment_P6 work/kat/il_S2_plt_moment_P7 work/kat/il_S2_plt_moment_P8 -o output/il_S2_mplt.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


aalpy -Kil_S3_summary_palt -c output/il_S3_alct.csv -l 0.95 -a output/il_S3_palt.csv & lpid1=$!
aalpy -Kil_S3_summary_altmeanonly -a output/il_S3_altmeanonly.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
