#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


touch $LOG_DIR/stderror.err
oasis_exec_monitor.sh $$ $LOG_DIR & pid0=$!

exit_handler(){
   exit_code=$?

   # disable handler
   trap - QUIT HUP INT KILL TERM ERR EXIT

   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       # Error - run process clean up
       echo 'Kernel execution error - exitcode='$exit_code

       set +x
       group_pid=$(ps -p $$ -o pgid --no-headers)
       sess_pid=$(ps -p $$ -o sess --no-headers)
       script_pid=$$
       printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> $LOG_DIR/killout.txt

       ps -jf f -g $sess_pid > $LOG_DIR/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *startup.sh$ | sort -n -r)
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
    proc_list="evepy modelpy gulpy fmpy gulmc summarypy plapy katpy eltpy pltpy aalpy lecpy"
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

rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summary_palt
mkdir -p work/gul_S1_summary_altmeanonly
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summary_palt
mkdir -p work/il_S1_summary_altmeanonly

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_plt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_elt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_plt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1



# --- Do insured loss computes ---

( pltpy -E bin  -s work/kat/il_S1_plt_sample_P1 -q work/kat/il_S1_plt_quantile_P1 -m work/kat/il_S1_plt_moment_P1 < /tmp/%FIFO_DIR%/fifo/il_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( eltpy -E bin  -q work/kat/il_S1_elt_quantile_P1 -m work/kat/il_S1_elt_moment_P1 < /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( eltpy -E bin  -s work/kat/il_S1_elt_sample_P1 < /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!

tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/il_S1_plt_ord_P1 /tmp/%FIFO_DIR%/fifo/il_S1_elt_ord_P1 /tmp/%FIFO_DIR%/fifo/il_S1_selt_ord_P1 work/il_S1_summary_palt/P1.bin work/il_S1_summary_altmeanonly/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summary_palt/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid5=$!

( summarypy -m -t il  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---

( pltpy -E bin  -s work/kat/gul_S1_plt_sample_P1 -q work/kat/gul_S1_plt_quantile_P1 -m work/kat/gul_S1_plt_moment_P1 < /tmp/%FIFO_DIR%/fifo/gul_S1_plt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltpy -E bin  -q work/kat/gul_S1_elt_quantile_P1 -m work/kat/gul_S1_elt_moment_P1 < /tmp/%FIFO_DIR%/fifo/gul_S1_elt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1 ) 2>> $LOG_DIR/stderror.err & pid8=$!

tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_plt_ord_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_elt_ord_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_selt_ord_P1 work/gul_S1_summary_palt/P1.bin work/gul_S1_summary_altmeanonly/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summary_palt/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid10=$!

( summarypy -m -t gul  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &

( ( evepy 1 1 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a1  | tee /tmp/%FIFO_DIR%/fifo/gul_P1 | fmpy -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid11=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11


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


( aalpy -Kil_S1_summary_palt -c output/il_S1_alct.parquet -l 0.95 -E parquet -a output/il_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( aalpy -Kil_S1_summary_altmeanonly -E parquet -a output/il_S1_altmeanonly.cparquetsv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( lecpy -r -Kil_S1_summaryleccalc -F -f -S -s -M -m -W -w -E parquet -O output/il_S1_ept.parquet -o output/il_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( aalpy -Kgul_S1_summary_palt -c output/gul_S1_alct.parquet -l 0.95 -E parquet -a output/gul_S1_palt.parquet ) 2>> $LOG_DIR/stderror.err & lpid4=$!
( aalpy -Kgul_S1_summary_altmeanonly -E parquet -a output/gul_S1_altmeanonly.cparquetsv ) 2>> $LOG_DIR/stderror.err & lpid5=$!
( lecpy -r -Kgul_S1_summaryleccalc -F -f -S -s -M -m -W -w -E parquet -O output/gul_S1_ept.parquet -o output/gul_S1_psept.parquet ) 2>> $LOG_DIR/stderror.err & lpid6=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/

check_complete
