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
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc ordleccalc"
    has_error=0
    for p in $proc_list; do
        started=$(find log -name "$p*.log" | wc -l)
        finished=$(find log -name "$p*.log" -exec grep -l "finish" {} + | wc -l)
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

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 work/kat/il_S1_eltcalc_P21 work/kat/il_S1_eltcalc_P22 work/kat/il_S1_eltcalc_P23 work/kat/il_S1_eltcalc_P24 work/kat/il_S1_eltcalc_P25 work/kat/il_S1_eltcalc_P26 work/kat/il_S1_eltcalc_P27 work/kat/il_S1_eltcalc_P28 work/kat/il_S1_eltcalc_P29 work/kat/il_S1_eltcalc_P30 work/kat/il_S1_eltcalc_P31 work/kat/il_S1_eltcalc_P32 work/kat/il_S1_eltcalc_P33 work/kat/il_S1_eltcalc_P34 work/kat/il_S1_eltcalc_P35 work/kat/il_S1_eltcalc_P36 work/kat/il_S1_eltcalc_P37 work/kat/il_S1_eltcalc_P38 work/kat/il_S1_eltcalc_P39 work/kat/il_S1_eltcalc_P40 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 work/kat/il_S1_pltcalc_P21 work/kat/il_S1_pltcalc_P22 work/kat/il_S1_pltcalc_P23 work/kat/il_S1_pltcalc_P24 work/kat/il_S1_pltcalc_P25 work/kat/il_S1_pltcalc_P26 work/kat/il_S1_pltcalc_P27 work/kat/il_S1_pltcalc_P28 work/kat/il_S1_pltcalc_P29 work/kat/il_S1_pltcalc_P30 work/kat/il_S1_pltcalc_P31 work/kat/il_S1_pltcalc_P32 work/kat/il_S1_pltcalc_P33 work/kat/il_S1_pltcalc_P34 work/kat/il_S1_pltcalc_P35 work/kat/il_S1_pltcalc_P36 work/kat/il_S1_pltcalc_P37 work/kat/il_S1_pltcalc_P38 work/kat/il_S1_pltcalc_P39 work/kat/il_S1_pltcalc_P40 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 work/kat/il_S1_summarycalc_P21 work/kat/il_S1_summarycalc_P22 work/kat/il_S1_summarycalc_P23 work/kat/il_S1_summarycalc_P24 work/kat/il_S1_summarycalc_P25 work/kat/il_S1_summarycalc_P26 work/kat/il_S1_summarycalc_P27 work/kat/il_S1_summarycalc_P28 work/kat/il_S1_summarycalc_P29 work/kat/il_S1_summarycalc_P30 work/kat/il_S1_summarycalc_P31 work/kat/il_S1_summarycalc_P32 work/kat/il_S1_summarycalc_P33 work/kat/il_S1_summarycalc_P34 work/kat/il_S1_summarycalc_P35 work/kat/il_S1_summarycalc_P36 work/kat/il_S1_summarycalc_P37 work/kat/il_S1_summarycalc_P38 work/kat/il_S1_summarycalc_P39 work/kat/il_S1_summarycalc_P40 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

