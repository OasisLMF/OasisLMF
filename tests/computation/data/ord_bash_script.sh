#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


touch $LOG_DIR/stderror.err
ktools_monitor.sh $$ $LOG_DIR & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Ktools Run Error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> $LOG_DIR/killout.txt

       ps -jf f -g $sess_pid > $LOG_DIR/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *\\.sh$ | sort -n -r)
       echo "$PIDS_KILL" >> $LOG_DIR/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
       exit $exit_code
   else
       # script successful
       exit 0
   fi
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT

check_complete(){
    set +e
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc aalcalcmeanonly leccalc pltcalc ordleccalc modelpy gulpy fmpy gulmc"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "${p}_[0-9]*.log" | wc -l)
        finished=$(find log -name "${p}_[0-9]*.log" -exec grep -l "finish" {} + | wc -l)
        if [ "$finished" -lt "$started" ]; then
            echo "[ERROR] $p - $((started-finished)) processes lost"
            has_error=1
        elif [ "$started" -gt 0 ]; then
            echo "[OK] $p"
        fi
    done
    if [ "$has_error" -ne 0 ]; then
        false # raise non-zero exit code
    else
        echo 'Run Completed'
    fi
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
#fmpy -a3 --create-financial-structure-files -p RI_1
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summaryaalcalcmeanonly
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summaryaalcalcmeanonly
mkdir -p work/il_S1_summary_altmeanonly
mkdir -p work/ri_S1_summaryleccalc
mkdir -p work/ri_S1_summaryaalcalc
mkdir -p work/ri_S1_summary_palt
mkdir -p work/ri_S1_summaryaalcalcmeanonly
mkdir -p work/ri_S1_summary_altmeanonly

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S1_plt_ord_P1
mkfifo fifo/gul_S1_elt_ord_P1
mkfifo fifo/gul_S1_selt_ord_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S1_plt_ord_P2
mkfifo fifo/gul_S1_elt_ord_P2
mkfifo fifo/gul_S1_selt_ord_P2

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S1_plt_ord_P1
mkfifo fifo/il_S1_elt_ord_P1
mkfifo fifo/il_S1_selt_ord_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2
mkfifo fifo/il_S1_plt_ord_P2
mkfifo fifo/il_S1_elt_ord_P2
mkfifo fifo/il_S1_selt_ord_P2

mkfifo fifo/ri_P1
mkfifo fifo/ri_P2

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summary_P1.idx
mkfifo fifo/ri_S1_eltcalc_P1
mkfifo fifo/ri_S1_summarycalc_P1
mkfifo fifo/ri_S1_pltcalc_P1
mkfifo fifo/ri_S1_plt_ord_P1
mkfifo fifo/ri_S1_elt_ord_P1
mkfifo fifo/ri_S1_selt_ord_P1

mkfifo fifo/ri_S1_summary_P2
mkfifo fifo/ri_S1_summary_P2.idx
mkfifo fifo/ri_S1_eltcalc_P2
mkfifo fifo/ri_S1_summarycalc_P2
mkfifo fifo/ri_S1_pltcalc_P2
mkfifo fifo/ri_S1_plt_ord_P2
mkfifo fifo/ri_S1_elt_ord_P2
mkfifo fifo/ri_S1_selt_ord_P2



# --- Do reinsurance loss computes ---

( eltcalc < fifo/ri_S1_eltcalc_P1 > work/kat/ri_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( summarycalctocsv < fifo/ri_S1_summarycalc_P1 > work/kat/ri_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltcalc < fifo/ri_S1_pltcalc_P1 > work/kat/ri_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltcalc -s < fifo/ri_S1_eltcalc_P2 > work/kat/ri_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/ri_S1_summarycalc_P2 > work/kat/ri_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( pltcalc -H < fifo/ri_S1_pltcalc_P2 > work/kat/ri_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid6=$!

( pltcalc -s work/kat/ri_S1_plt_sample_P1 -q work/kat/ri_S1_plt_quantile_P1 -m work/kat/ri_S1_plt_moment_P1 < fifo/ri_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( eltcalc -q work/kat/ri_S1_elt_quantile_P1 -m work/kat/ri_S1_elt_moment_P1 < fifo/ri_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( summarycalctocsv -p work/kat/ri_S1_elt_sample_P1 < fifo/ri_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( pltcalc -H -s work/kat/ri_S1_plt_sample_P2 -q work/kat/ri_S1_plt_quantile_P2 -m work/kat/ri_S1_plt_moment_P2 < fifo/ri_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( eltcalc -s -q work/kat/ri_S1_elt_quantile_P2 -m work/kat/ri_S1_elt_moment_P2 < fifo/ri_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( summarycalctocsv -s -p work/kat/ri_S1_elt_sample_P2 < fifo/ri_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid12=$!

tee < fifo/ri_S1_summary_P1 fifo/ri_S1_eltcalc_P1 fifo/ri_S1_summarycalc_P1 fifo/ri_S1_pltcalc_P1 fifo/ri_S1_plt_ord_P1 fifo/ri_S1_elt_ord_P1 fifo/ri_S1_selt_ord_P1 work/ri_S1_summaryaalcalc/P1.bin work/ri_S1_summary_palt/P1.bin work/ri_S1_summaryaalcalcmeanonly/P1.bin work/ri_S1_summary_altmeanonly/P1.bin work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid13=$!
tee < fifo/ri_S1_summary_P1.idx work/ri_S1_summaryaalcalc/P1.idx work/ri_S1_summary_palt/P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid14=$!
tee < fifo/ri_S1_summary_P2 fifo/ri_S1_eltcalc_P2 fifo/ri_S1_summarycalc_P2 fifo/ri_S1_pltcalc_P2 fifo/ri_S1_plt_ord_P2 fifo/ri_S1_elt_ord_P2 fifo/ri_S1_selt_ord_P2 work/ri_S1_summaryaalcalc/P2.bin work/ri_S1_summary_palt/P2.bin work/ri_S1_summaryaalcalcmeanonly/P2.bin work/ri_S1_summary_altmeanonly/P2.bin work/ri_S1_summaryleccalc/P2.bin > /dev/null & pid15=$!
tee < fifo/ri_S1_summary_P2.idx work/ri_S1_summaryaalcalc/P2.idx work/ri_S1_summary_palt/P2.idx work/ri_S1_summaryleccalc/P2.idx > /dev/null & pid16=$!

( summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P2 < fifo/ri_P2 ) 2>> $LOG_DIR/stderror.err  &

# --- Do insured loss computes ---

( eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid17=$!
( summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid18=$!
( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid19=$!
( eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid20=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid21=$!
( pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid22=$!

( pltcalc -s work/kat/il_S1_plt_sample_P1 -q work/kat/il_S1_plt_quantile_P1 -m work/kat/il_S1_plt_moment_P1 < fifo/il_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid23=$!
( eltcalc -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid24=$!
( summarycalctocsv -p work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid25=$!
( pltcalc -H -s work/kat/il_S1_plt_sample_P2 -q work/kat/il_S1_plt_quantile_P2 -m work/kat/il_S1_plt_moment_P2 < fifo/il_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid26=$!
( eltcalc -s -q work/kat/il_S1_elt_quantile_P2 -m work/kat/il_S1_elt_moment_P2 < fifo/il_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid27=$!
( summarycalctocsv -s -p work/kat/il_S1_elt_sample_P2 < fifo/il_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid28=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 fifo/il_S1_plt_ord_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summary_palt/P1.bin work/il_S1_summaryaalcalcmeanonly/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid29=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid30=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 fifo/il_S1_plt_ord_P2 fifo/il_S1_elt_ord_P2 fifo/il_S1_selt_ord_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summary_palt/P2.bin work/il_S1_summaryaalcalcmeanonly/P2.bin work/il_S1_summary_altmeanonly/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid31=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summary_palt/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid32=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid33=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid34=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid35=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid36=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid37=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid38=$!

( pltcalc -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < fifo/gul_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid39=$!
( eltcalc -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < fifo/gul_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid40=$!
( summarycalctocsv -p work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid41=$!
( pltcalc -H -s work/kat/gul_S1_plt_sample_P2 -q work/kat/gul_S1_plt_quantile_P2 -m work/kat/gul_S1_plt_moment_P2 < fifo/gul_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid42=$!
( eltcalc -s -q work/kat/gul_S1_elt_quantile_P2 -m work/kat/gul_S1_elt_moment_P2 < fifo/gul_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid43=$!
( summarycalctocsv -s -p work/kat/gul_S1_elt_sample_P2 < fifo/gul_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid44=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 fifo/gul_S1_plt_ord_P1 fifo/gul_S1_elt_ord_P1 fifo/gul_S1_selt_ord_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summary_palt/P1.bin work/gul_S1_summaryaalcalcmeanonly/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid45=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid46=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 fifo/gul_S1_plt_ord_P2 fifo/gul_S1_elt_ord_P2 fifo/gul_S1_selt_ord_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summary_palt/P2.bin work/gul_S1_summaryaalcalcmeanonly/P2.bin work/gul_S1_summary_altmeanonly/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid47=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summary_palt/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid48=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &

( ( eve 1 2 | modelpy | gulpy --random-generator=1 -S1 -L0 -a0  | tee fifo/gul_P1 | fmpy -a2 | tee fifo/il_P1 | fmpy -a3 -n -p RI_1 > fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err ) & pid49=$!
( ( eve 2 2 | modelpy | gulpy --random-generator=1 -S1 -L0 -a0  | tee fifo/gul_P2 | fmpy -a2 | tee fifo/il_P2 | fmpy -a3 -n -p RI_1 > fifo/ri_P2 ) 2>> $LOG_DIR/stderror.err ) & pid50=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50


# --- Do reinsurance loss kats ---

kat work/kat/ri_S1_eltcalc_P1 work/kat/ri_S1_eltcalc_P2 > output/ri_S1_eltcalc.csv & kpid1=$!
kat work/kat/ri_S1_pltcalc_P1 work/kat/ri_S1_pltcalc_P2 > output/ri_S1_pltcalc.csv & kpid2=$!
kat work/kat/ri_S1_summarycalc_P1 work/kat/ri_S1_summarycalc_P2 > output/ri_S1_summarycalc.csv & kpid3=$!
katparquet -S work/kat/ri_S1_plt_sample_P1 work/kat/ri_S1_plt_sample_P2 -o output/ri_S1_splt.parquet & kpid4=$!
katparquet -Q work/kat/ri_S1_plt_quantile_P1 work/kat/ri_S1_plt_quantile_P2 -o output/ri_S1_qplt.parquet & kpid5=$!
katparquet -M work/kat/ri_S1_plt_moment_P1 work/kat/ri_S1_plt_moment_P2 -o output/ri_S1_mplt.parquet & kpid6=$!
katparquet -q work/kat/ri_S1_elt_quantile_P1 work/kat/ri_S1_elt_quantile_P2 -o output/ri_S1_qelt.parquet & kpid7=$!
katparquet -m work/kat/ri_S1_elt_moment_P1 work/kat/ri_S1_elt_moment_P2 -o output/ri_S1_melt.parquet & kpid8=$!
katparquet -s work/kat/ri_S1_elt_sample_P1 work/kat/ri_S1_elt_sample_P2 -o output/ri_S1_selt.parquet & kpid9=$!

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 > output/il_S1_eltcalc.csv & kpid10=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 > output/il_S1_pltcalc.csv & kpid11=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 > output/il_S1_summarycalc.csv & kpid12=$!
katparquet -S work/kat/il_S1_plt_sample_P1 work/kat/il_S1_plt_sample_P2 -o output/il_S1_splt.parquet & kpid13=$!
katparquet -Q work/kat/il_S1_plt_quantile_P1 work/kat/il_S1_plt_quantile_P2 -o output/il_S1_qplt.parquet & kpid14=$!
katparquet -M work/kat/il_S1_plt_moment_P1 work/kat/il_S1_plt_moment_P2 -o output/il_S1_mplt.parquet & kpid15=$!
katparquet -q work/kat/il_S1_elt_quantile_P1 work/kat/il_S1_elt_quantile_P2 -o output/il_S1_qelt.parquet & kpid16=$!
katparquet -m work/kat/il_S1_elt_moment_P1 work/kat/il_S1_elt_moment_P2 -o output/il_S1_melt.parquet & kpid17=$!
katparquet -s work/kat/il_S1_elt_sample_P1 work/kat/il_S1_elt_sample_P2 -o output/il_S1_selt.parquet & kpid18=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid19=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid20=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid21=$!
katparquet -S work/kat/gul_S1_plt_sample_P1 work/kat/gul_S1_plt_sample_P2 -o output/gul_S1_splt.parquet & kpid22=$!
katparquet -Q work/kat/gul_S1_plt_quantile_P1 work/kat/gul_S1_plt_quantile_P2 -o output/gul_S1_qplt.parquet & kpid23=$!
katparquet -M work/kat/gul_S1_plt_moment_P1 work/kat/gul_S1_plt_moment_P2 -o output/gul_S1_mplt.parquet & kpid24=$!
katparquet -q work/kat/gul_S1_elt_quantile_P1 work/kat/gul_S1_elt_quantile_P2 -o output/gul_S1_qelt.parquet & kpid25=$!
katparquet -m work/kat/gul_S1_elt_moment_P1 work/kat/gul_S1_elt_moment_P2 -o output/gul_S1_melt.parquet & kpid26=$!
katparquet -s work/kat/gul_S1_elt_sample_P1 work/kat/gul_S1_elt_sample_P2 -o output/gul_S1_selt.parquet & kpid27=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18 $kpid19 $kpid20 $kpid21 $kpid22 $kpid23 $kpid24 $kpid25 $kpid26 $kpid27


( aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( aalcalc -Kri_S1_summary_palt -p output/ri_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( aalcalcmeanonly -Kri_S1_summaryaalcalcmeanonly > output/ri_S1_aalcalcmeanonly.csv ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( aalcalcmeanonly -Kri_S1_summary_altmeanonly -p output/ri_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid4=$!
( ordleccalc  -Kri_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/ri_S1_ept.parquet -p output/ri_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid5=$!
( leccalc -r -Kri_S1_summaryleccalc -F output/ri_S1_leccalc_full_uncertainty_aep.csv -f output/ri_S1_leccalc_full_uncertainty_oep.csv -S output/ri_S1_leccalc_sample_mean_aep.csv -s output/ri_S1_leccalc_sample_mean_oep.csv -W output/ri_S1_leccalc_wheatsheaf_aep.csv -M output/ri_S1_leccalc_wheatsheaf_mean_aep.csv -m output/ri_S1_leccalc_wheatsheaf_mean_oep.csv -w output/ri_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid6=$!
( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid7=$!
( aalcalc -Kil_S1_summary_palt -p output/il_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid8=$!
( aalcalcmeanonly -Kil_S1_summaryaalcalcmeanonly > output/il_S1_aalcalcmeanonly.csv ) 2>> $LOG_DIR/stderror.err & lpid9=$!
( aalcalcmeanonly -Kil_S1_summary_altmeanonly -p output/il_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid10=$!
( ordleccalc  -Kil_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/il_S1_ept.parquet -p output/il_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid11=$!
( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid12=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid13=$!
( aalcalc -Kgul_S1_summary_palt -p output/gul_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid14=$!
( aalcalcmeanonly -Kgul_S1_summaryaalcalcmeanonly > output/gul_S1_aalcalcmeanonly.csv ) 2>> $LOG_DIR/stderror.err & lpid15=$!
( aalcalcmeanonly -Kgul_S1_summary_altmeanonly -p output/gul_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid16=$!
( ordleccalc  -Kgul_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/gul_S1_ept.parquet -p output/gul_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid17=$!
( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid18=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8 $lpid9 $lpid10 $lpid11 $lpid12 $lpid13 $lpid14 $lpid15 $lpid16 $lpid17 $lpid18

rm -R -f work/*
rm -R -f fifo/*

check_complete
