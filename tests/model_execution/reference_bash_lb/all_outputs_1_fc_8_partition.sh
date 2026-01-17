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

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4

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

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2
mkfifo fifo/gul_lb_P3
mkfifo fifo/gul_lb_P4

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2
mkfifo fifo/lb_il_P3
mkfifo fifo/lb_il_P4



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

tee < fifo/il_S1_summary_P1 fifo/il_S1_plt_ord_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 work/il_S1_summary_palt/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid13=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid14=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_plt_ord_P2 fifo/il_S1_elt_ord_P2 fifo/il_S1_selt_ord_P2 work/il_S1_summary_palt/P2.bin work/il_S1_summary_altmeanonly/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid15=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summary_palt/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid16=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_plt_ord_P3 fifo/il_S1_elt_ord_P3 fifo/il_S1_selt_ord_P3 work/il_S1_summary_palt/P3.bin work/il_S1_summary_altmeanonly/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summary_palt/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_plt_ord_P4 fifo/il_S1_elt_ord_P4 fifo/il_S1_selt_ord_P4 work/il_S1_summary_palt/P4.bin work/il_S1_summary_altmeanonly/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summary_palt/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid20=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarypy -m -t il  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &

# --- Do ground up loss computes ---


pltpy -E bin  -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < fifo/gul_S1_plt_ord_P1 & pid21=$!
eltpy -E bin  -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < fifo/gul_S1_elt_ord_P1 & pid22=$!
eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 & pid23=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P2 -q work/kat/gul_S1_plt_quantile_P2 -m work/kat/gul_S1_plt_moment_P2 < fifo/gul_S1_plt_ord_P2 & pid24=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P2 -m work/kat/gul_S1_elt_moment_P2 < fifo/gul_S1_elt_ord_P2 & pid25=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P2 < fifo/gul_S1_selt_ord_P2 & pid26=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P3 -q work/kat/gul_S1_plt_quantile_P3 -m work/kat/gul_S1_plt_moment_P3 < fifo/gul_S1_plt_ord_P3 & pid27=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P3 -m work/kat/gul_S1_elt_moment_P3 < fifo/gul_S1_elt_ord_P3 & pid28=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P3 < fifo/gul_S1_selt_ord_P3 & pid29=$!
pltpy -E bin  -H -s work/kat/gul_S1_plt_sample_P4 -q work/kat/gul_S1_plt_quantile_P4 -m work/kat/gul_S1_plt_moment_P4 < fifo/gul_S1_plt_ord_P4 & pid30=$!
eltpy -E bin  -H -q work/kat/gul_S1_elt_quantile_P4 -m work/kat/gul_S1_elt_moment_P4 < fifo/gul_S1_elt_ord_P4 & pid31=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P4 < fifo/gul_S1_selt_ord_P4 & pid32=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_plt_ord_P1 fifo/gul_S1_elt_ord_P1 fifo/gul_S1_selt_ord_P1 work/gul_S1_summary_palt/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid33=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid34=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_plt_ord_P2 fifo/gul_S1_elt_ord_P2 fifo/gul_S1_selt_ord_P2 work/gul_S1_summary_palt/P2.bin work/gul_S1_summary_altmeanonly/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid35=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summary_palt/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid36=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_plt_ord_P3 fifo/gul_S1_elt_ord_P3 fifo/gul_S1_selt_ord_P3 work/gul_S1_summary_palt/P3.bin work/gul_S1_summary_altmeanonly/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid37=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summary_palt/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid38=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_plt_ord_P4 fifo/gul_S1_elt_ord_P4 fifo/gul_S1_selt_ord_P4 work/gul_S1_summary_palt/P4.bin work/gul_S1_summary_altmeanonly/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid39=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summary_palt/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid40=$!

summarypy -m -t gul  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 &

( evepy 1 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P1 > fifo/gul_lb_P1  ) & 
( evepy 2 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P2 > fifo/gul_lb_P2  ) & 
( evepy 3 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P3 > fifo/gul_lb_P3  ) & 
( evepy 4 4 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S50 -L10 -a0  | tee fifo/gul_P4 > fifo/gul_lb_P4  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
load_balancer -i fifo/gul_lb_P3 fifo/gul_lb_P4 -o fifo/lb_il_P3 fifo/lb_il_P4 &
( fmpy -a2 < fifo/lb_il_P1 > fifo/il_P1 ) & pid41=$!
( fmpy -a2 < fifo/lb_il_P2 > fifo/il_P2 ) & pid42=$!
( fmpy -a2 < fifo/lb_il_P3 > fifo/il_P3 ) & pid43=$!
( fmpy -a2 < fifo/lb_il_P4 > fifo/il_P4 ) & pid44=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44


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
