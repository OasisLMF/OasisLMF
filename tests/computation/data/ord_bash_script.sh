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
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summary_altmeanonly
mkdir -p work/ri_S1_summaryleccalc
mkdir -p work/ri_S1_summary_palt
mkdir -p work/ri_S1_summary_altmeanonly

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2

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

mkfifo fifo/il_P1
mkfifo fifo/il_P2

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

mkfifo fifo/ri_P1
mkfifo fifo/ri_P2

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summary_P1.idx
mkfifo fifo/ri_S1_plt_ord_P1
mkfifo fifo/ri_S1_elt_ord_P1
mkfifo fifo/ri_S1_selt_ord_P1

mkfifo fifo/ri_S1_summary_P2
mkfifo fifo/ri_S1_summary_P2.idx
mkfifo fifo/ri_S1_plt_ord_P2
mkfifo fifo/ri_S1_elt_ord_P2
mkfifo fifo/ri_S1_selt_ord_P2



# --- Do reinsurance loss computes ---


( pltcalc -s work/kat/ri_S1_plt_sample_P1 -q work/kat/ri_S1_plt_quantile_P1 -m work/kat/ri_S1_plt_moment_P1 < fifo/ri_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( eltcalc -q work/kat/ri_S1_elt_quantile_P1 -m work/kat/ri_S1_elt_moment_P1 < fifo/ri_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( summarycalctocsv -p work/kat/ri_S1_elt_sample_P1 < fifo/ri_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( pltcalc -H -s work/kat/ri_S1_plt_sample_P2 -q work/kat/ri_S1_plt_quantile_P2 -m work/kat/ri_S1_plt_moment_P2 < fifo/ri_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( eltcalc -s -q work/kat/ri_S1_elt_quantile_P2 -m work/kat/ri_S1_elt_moment_P2 < fifo/ri_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( summarycalctocsv -s -p work/kat/ri_S1_elt_sample_P2 < fifo/ri_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid6=$!

tee < fifo/ri_S1_summary_P1 fifo/ri_S1_plt_ord_P1 fifo/ri_S1_elt_ord_P1 fifo/ri_S1_selt_ord_P1 work/ri_S1_summary_palt/P1.bin work/ri_S1_summary_altmeanonly/P1.bin work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid7=$!
tee < fifo/ri_S1_summary_P1.idx work/ri_S1_summary_palt/P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid8=$!
tee < fifo/ri_S1_summary_P2 fifo/ri_S1_plt_ord_P2 fifo/ri_S1_elt_ord_P2 fifo/ri_S1_selt_ord_P2 work/ri_S1_summary_palt/P2.bin work/ri_S1_summary_altmeanonly/P2.bin work/ri_S1_summaryleccalc/P2.bin > /dev/null & pid9=$!
tee < fifo/ri_S1_summary_P2.idx work/ri_S1_summary_palt/P2.idx work/ri_S1_summaryleccalc/P2.idx > /dev/null & pid10=$!

( summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P2 < fifo/ri_P2 ) 2>> $LOG_DIR/stderror.err  &

# --- Do insured loss computes ---


( pltcalc -s work/kat/il_S1_plt_sample_P1 -q work/kat/il_S1_plt_quantile_P1 -m work/kat/il_S1_plt_moment_P1 < fifo/il_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( eltcalc -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < fifo/il_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( summarycalctocsv -p work/kat/il_S1_elt_sample_P1 < fifo/il_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( pltcalc -H -s work/kat/il_S1_plt_sample_P2 -q work/kat/il_S1_plt_quantile_P2 -m work/kat/il_S1_plt_moment_P2 < fifo/il_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( eltcalc -s -q work/kat/il_S1_elt_quantile_P2 -m work/kat/il_S1_elt_moment_P2 < fifo/il_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( summarycalctocsv -s -p work/kat/il_S1_elt_sample_P2 < fifo/il_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid16=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_plt_ord_P1 fifo/il_S1_elt_ord_P1 fifo/il_S1_selt_ord_P1 work/il_S1_summary_palt/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_plt_ord_P2 fifo/il_S1_elt_ord_P2 fifo/il_S1_selt_ord_P2 work/il_S1_summary_palt/P2.bin work/il_S1_summary_altmeanonly/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summary_palt/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid20=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---


( pltcalc -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < fifo/gul_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid21=$!
( eltcalc -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < fifo/gul_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid22=$!
( summarycalctocsv -p work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid23=$!
( pltcalc -H -s work/kat/gul_S1_plt_sample_P2 -q work/kat/gul_S1_plt_quantile_P2 -m work/kat/gul_S1_plt_moment_P2 < fifo/gul_S1_plt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid24=$!
( eltcalc -s -q work/kat/gul_S1_elt_quantile_P2 -m work/kat/gul_S1_elt_moment_P2 < fifo/gul_S1_elt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid25=$!
( summarycalctocsv -s -p work/kat/gul_S1_elt_sample_P2 < fifo/gul_S1_selt_ord_P2 ) 2>> $LOG_DIR/stderror.err & pid26=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_plt_ord_P1 fifo/gul_S1_elt_ord_P1 fifo/gul_S1_selt_ord_P1 work/gul_S1_summary_palt/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid27=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid28=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_plt_ord_P2 fifo/gul_S1_elt_ord_P2 fifo/gul_S1_selt_ord_P2 work/gul_S1_summary_palt/P2.bin work/gul_S1_summary_altmeanonly/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid29=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summary_palt/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid30=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &

( ( eve 1 2 | modelpy | gulpy --random-generator=1 -S1 -L0 -a0  | tee fifo/gul_P1 | fmpy -a2 | tee fifo/il_P1 | fmpy -a3 -n -p RI_1 > fifo/ri_P1 ) 2>> $LOG_DIR/stderror.err ) & pid31=$!
( ( eve 2 2 | modelpy | gulpy --random-generator=1 -S1 -L0 -a0  | tee fifo/gul_P2 | fmpy -a2 | tee fifo/il_P2 | fmpy -a3 -n -p RI_1 > fifo/ri_P2 ) 2>> $LOG_DIR/stderror.err ) & pid32=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32


# --- Do reinsurance loss kats ---

katparquet -S work/kat/ri_S1_plt_sample_P1 work/kat/ri_S1_plt_sample_P2 -o output/ri_S1_splt.parquet & kpid1=$!
katparquet -Q work/kat/ri_S1_plt_quantile_P1 work/kat/ri_S1_plt_quantile_P2 -o output/ri_S1_qplt.parquet & kpid2=$!
katparquet -M work/kat/ri_S1_plt_moment_P1 work/kat/ri_S1_plt_moment_P2 -o output/ri_S1_mplt.parquet & kpid3=$!
katparquet -q work/kat/ri_S1_elt_quantile_P1 work/kat/ri_S1_elt_quantile_P2 -o output/ri_S1_qelt.parquet & kpid4=$!
katparquet -m work/kat/ri_S1_elt_moment_P1 work/kat/ri_S1_elt_moment_P2 -o output/ri_S1_melt.parquet & kpid5=$!
katparquet -s work/kat/ri_S1_elt_sample_P1 work/kat/ri_S1_elt_sample_P2 -o output/ri_S1_selt.parquet & kpid6=$!

# --- Do insured loss kats ---

katparquet -S work/kat/il_S1_plt_sample_P1 work/kat/il_S1_plt_sample_P2 -o output/il_S1_splt.parquet & kpid7=$!
katparquet -Q work/kat/il_S1_plt_quantile_P1 work/kat/il_S1_plt_quantile_P2 -o output/il_S1_qplt.parquet & kpid8=$!
katparquet -M work/kat/il_S1_plt_moment_P1 work/kat/il_S1_plt_moment_P2 -o output/il_S1_mplt.parquet & kpid9=$!
katparquet -q work/kat/il_S1_elt_quantile_P1 work/kat/il_S1_elt_quantile_P2 -o output/il_S1_qelt.parquet & kpid10=$!
katparquet -m work/kat/il_S1_elt_moment_P1 work/kat/il_S1_elt_moment_P2 -o output/il_S1_melt.parquet & kpid11=$!
katparquet -s work/kat/il_S1_elt_sample_P1 work/kat/il_S1_elt_sample_P2 -o output/il_S1_selt.parquet & kpid12=$!

# --- Do ground up loss kats ---

katparquet -S work/kat/gul_S1_plt_sample_P1 work/kat/gul_S1_plt_sample_P2 -o output/gul_S1_splt.parquet & kpid13=$!
katparquet -Q work/kat/gul_S1_plt_quantile_P1 work/kat/gul_S1_plt_quantile_P2 -o output/gul_S1_qplt.parquet & kpid14=$!
katparquet -M work/kat/gul_S1_plt_moment_P1 work/kat/gul_S1_plt_moment_P2 -o output/gul_S1_mplt.parquet & kpid15=$!
katparquet -q work/kat/gul_S1_elt_quantile_P1 work/kat/gul_S1_elt_quantile_P2 -o output/gul_S1_qelt.parquet & kpid16=$!
katparquet -m work/kat/gul_S1_elt_moment_P1 work/kat/gul_S1_elt_moment_P2 -o output/gul_S1_melt.parquet & kpid17=$!
katparquet -s work/kat/gul_S1_elt_sample_P1 work/kat/gul_S1_elt_sample_P2 -o output/gul_S1_selt.parquet & kpid18=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18


( aalcalc -Kri_S1_summary_palt -p output/ri_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( aalcalcmeanonly -Kri_S1_summary_altmeanonly -p output/ri_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( ordleccalc  -Kri_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/ri_S1_ept.parquet -p output/ri_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( aalcalc -Kil_S1_summary_palt -p output/il_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid4=$!
( aalcalcmeanonly -Kil_S1_summary_altmeanonly -p output/il_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid5=$!
( ordleccalc  -Kil_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/il_S1_ept.parquet -p output/il_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid6=$!
( aalcalc -Kgul_S1_summary_palt -p output/gul_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid7=$!
( aalcalcmeanonly -Kgul_S1_summary_altmeanonly -p output/gul_S1_altmeanonly.parquet ) 2>> $LOG_DIR/stderror.err & lpid8=$!
( ordleccalc  -Kgul_S1_summaryleccalc -F -f -S -s -M -m -W -w -P output/gul_S1_ept.parquet -p output/gul_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid9=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8 $lpid9

rm -R -f work/*
rm -R -f fifo/*

check_complete
