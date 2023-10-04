#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Redirect Bash trace to file ---
bash_logging_supported(){
    local BASH_VER_MAJOR=${BASH_VERSION:0:1}
    local BASH_VER_MINOR=${BASH_VERSION:2:1}

    if [[ "$BASH_VER_MAJOR" -gt 4 ]]; then
        echo 1; exit
    fi
    if [[ $BASH_VER_MAJOR -eq 4 ]] && [[ $BASH_VER_MINOR -gt 3 ]]; then
        echo 1; exit
    fi
    echo 0
}
if [ $(bash_logging_supported) == 1 ]; then
    exec   > >(tee -ia $LOG_DIR/bash.log)
    exec  2> >(tee -ia $LOG_DIR/bash.log >& 2)
    exec 19> $LOG_DIR/bash.log
    export BASH_XTRACEFD="19"
    set -x
else
    echo "WARNING: logging disabled, bash version '$BASH_VERSION' is not supported, minimum requirement is bash v4.4"
fi 

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
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc ordleccalc modelpy gulpy fmpy gulmc"
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
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f work/*
mkdir -p work/kat/

rm -R -f /tmp/jdhlfRbtW9/
mkdir -p /tmp/jdhlfRbtW9/fifo/

mkfifo /tmp/jdhlfRbtW9/fifo/gul_P1
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P2
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P3
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P4
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P5
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P6
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P7
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P8
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P9
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P10
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P11
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P12
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P13
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P14
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P15
mkfifo /tmp/jdhlfRbtW9/fifo/gul_P16

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P1
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P1

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P2
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P2

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P3
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P3

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P4
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P4

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P5
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P5

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P6
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P6

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P7
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P7

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P8
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P8

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P9
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P9

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P10
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P10

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P11
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P11

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P12
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P12

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P13
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P13

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P14
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P14

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P15
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P15

mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P16
mkfifo /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P16



# --- Do ground up loss computes ---

( eltcalc < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( eltcalc -s < /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid16=$!


tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P1 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P1 > /dev/null & pid17=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P2 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P2 > /dev/null & pid18=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P3 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P3 > /dev/null & pid19=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P4 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P4 > /dev/null & pid20=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P5 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P5 > /dev/null & pid21=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P6 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P6 > /dev/null & pid22=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P7 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P7 > /dev/null & pid23=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P8 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P8 > /dev/null & pid24=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P9 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P9 > /dev/null & pid25=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P10 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P10 > /dev/null & pid26=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P11 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P11 > /dev/null & pid27=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P12 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P12 > /dev/null & pid28=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P13 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P13 > /dev/null & pid29=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P14 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P14 > /dev/null & pid30=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P15 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P15 > /dev/null & pid31=$!
tee < /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P16 /tmp/jdhlfRbtW9/fifo/gul_S1_eltcalc_P16 > /dev/null & pid32=$!

( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P1 < /tmp/jdhlfRbtW9/fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P2 < /tmp/jdhlfRbtW9/fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P3 < /tmp/jdhlfRbtW9/fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P4 < /tmp/jdhlfRbtW9/fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P5 < /tmp/jdhlfRbtW9/fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P6 < /tmp/jdhlfRbtW9/fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P7 < /tmp/jdhlfRbtW9/fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P8 < /tmp/jdhlfRbtW9/fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P9 < /tmp/jdhlfRbtW9/fifo/gul_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P10 < /tmp/jdhlfRbtW9/fifo/gul_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P11 < /tmp/jdhlfRbtW9/fifo/gul_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P12 < /tmp/jdhlfRbtW9/fifo/gul_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P13 < /tmp/jdhlfRbtW9/fifo/gul_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P14 < /tmp/jdhlfRbtW9/fifo/gul_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P15 < /tmp/jdhlfRbtW9/fifo/gul_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 /tmp/jdhlfRbtW9/fifo/gul_S1_summary_P16 < /tmp/jdhlfRbtW9/fifo/gul_P16 ) 2>> $LOG_DIR/stderror.err  &

( ( (gulmc -e 1 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P1  ) 2>> $LOG_DIR/stderror.err ) &  pid33=$!
( ( (gulmc -e 2 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P2  ) 2>> $LOG_DIR/stderror.err ) &  pid34=$!
( ( (gulmc -e 3 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P3  ) 2>> $LOG_DIR/stderror.err ) &  pid35=$!
( ( (gulmc -e 4 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P4  ) 2>> $LOG_DIR/stderror.err ) &  pid36=$!
( ( (gulmc -e 5 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P5  ) 2>> $LOG_DIR/stderror.err ) &  pid37=$!
( ( (gulmc -e 6 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P6  ) 2>> $LOG_DIR/stderror.err ) &  pid38=$!
( ( (gulmc -e 7 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P7  ) 2>> $LOG_DIR/stderror.err ) &  pid39=$!
( ( (gulmc -e 8 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P8  ) 2>> $LOG_DIR/stderror.err ) &  pid40=$!
( ( (gulmc -e 9 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P9  ) 2>> $LOG_DIR/stderror.err ) &  pid41=$!
( ( (gulmc -e 10 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P10  ) 2>> $LOG_DIR/stderror.err ) &  pid42=$!
( ( (gulmc -e 11 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P11  ) 2>> $LOG_DIR/stderror.err ) &  pid43=$!
( ( (gulmc -e 12 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P12  ) 2>> $LOG_DIR/stderror.err ) &  pid44=$!
( ( (gulmc -e 13 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P13  ) 2>> $LOG_DIR/stderror.err ) &  pid45=$!
( ( (gulmc -e 14 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P14  ) 2>> $LOG_DIR/stderror.err ) &  pid46=$!
( ( (gulmc -e 15 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P15  ) 2>> $LOG_DIR/stderror.err ) &  pid47=$!
( ( (gulmc -e 16 16 -a /home/jbanorthwest.co.uk/carlfischer/repos/OasisLMF/runs/losses-20231003160658/analysis_settings.json -p input -i -) 2>> log/gul_stderror.err > /tmp/jdhlfRbtW9/fifo/gul_P16  ) 2>> $LOG_DIR/stderror.err ) &  pid48=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f /tmp/jdhlfRbtW9/

check_complete
