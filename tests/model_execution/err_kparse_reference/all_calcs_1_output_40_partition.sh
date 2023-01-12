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
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc
mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryleccalc
mkdir -p work/il_S1_summaryaalcalc

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2
mkfifo fifo/gul_P3
mkfifo fifo/gul_P4
mkfifo fifo/gul_P5
mkfifo fifo/gul_P6
mkfifo fifo/gul_P7
mkfifo fifo/gul_P8
mkfifo fifo/gul_P9
mkfifo fifo/gul_P10
mkfifo fifo/gul_P11
mkfifo fifo/gul_P12
mkfifo fifo/gul_P13
mkfifo fifo/gul_P14
mkfifo fifo/gul_P15
mkfifo fifo/gul_P16
mkfifo fifo/gul_P17
mkfifo fifo/gul_P18
mkfifo fifo/gul_P19
mkfifo fifo/gul_P20
mkfifo fifo/gul_P21
mkfifo fifo/gul_P22
mkfifo fifo/gul_P23
mkfifo fifo/gul_P24
mkfifo fifo/gul_P25
mkfifo fifo/gul_P26
mkfifo fifo/gul_P27
mkfifo fifo/gul_P28
mkfifo fifo/gul_P29
mkfifo fifo/gul_P30
mkfifo fifo/gul_P31
mkfifo fifo/gul_P32
mkfifo fifo/gul_P33
mkfifo fifo/gul_P34
mkfifo fifo/gul_P35
mkfifo fifo/gul_P36
mkfifo fifo/gul_P37
mkfifo fifo/gul_P38
mkfifo fifo/gul_P39
mkfifo fifo/gul_P40

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summary_P3.idx
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summary_P4.idx
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summary_P5.idx
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summary_P6.idx
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_summary_P8.idx
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_summary_P11.idx
mkfifo fifo/gul_S1_eltcalc_P11
mkfifo fifo/gul_S1_summarycalc_P11
mkfifo fifo/gul_S1_pltcalc_P11

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_summary_P12.idx
mkfifo fifo/gul_S1_eltcalc_P12
mkfifo fifo/gul_S1_summarycalc_P12
mkfifo fifo/gul_S1_pltcalc_P12

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summary_P13.idx
mkfifo fifo/gul_S1_eltcalc_P13
mkfifo fifo/gul_S1_summarycalc_P13
mkfifo fifo/gul_S1_pltcalc_P13

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summary_P14.idx
mkfifo fifo/gul_S1_eltcalc_P14
mkfifo fifo/gul_S1_summarycalc_P14
mkfifo fifo/gul_S1_pltcalc_P14

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_summary_P15.idx
mkfifo fifo/gul_S1_eltcalc_P15
mkfifo fifo/gul_S1_summarycalc_P15
mkfifo fifo/gul_S1_pltcalc_P15

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_summary_P16.idx
mkfifo fifo/gul_S1_eltcalc_P16
mkfifo fifo/gul_S1_summarycalc_P16
mkfifo fifo/gul_S1_pltcalc_P16

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_summary_P17.idx
mkfifo fifo/gul_S1_eltcalc_P17
mkfifo fifo/gul_S1_summarycalc_P17
mkfifo fifo/gul_S1_pltcalc_P17

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_summary_P18.idx
mkfifo fifo/gul_S1_eltcalc_P18
mkfifo fifo/gul_S1_summarycalc_P18
mkfifo fifo/gul_S1_pltcalc_P18

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summary_P19.idx
mkfifo fifo/gul_S1_eltcalc_P19
mkfifo fifo/gul_S1_summarycalc_P19
mkfifo fifo/gul_S1_pltcalc_P19

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_summary_P20.idx
mkfifo fifo/gul_S1_eltcalc_P20
mkfifo fifo/gul_S1_summarycalc_P20
mkfifo fifo/gul_S1_pltcalc_P20

mkfifo fifo/gul_S1_summary_P21
mkfifo fifo/gul_S1_summary_P21.idx
mkfifo fifo/gul_S1_eltcalc_P21
mkfifo fifo/gul_S1_summarycalc_P21
mkfifo fifo/gul_S1_pltcalc_P21

mkfifo fifo/gul_S1_summary_P22
mkfifo fifo/gul_S1_summary_P22.idx
mkfifo fifo/gul_S1_eltcalc_P22
mkfifo fifo/gul_S1_summarycalc_P22
mkfifo fifo/gul_S1_pltcalc_P22

mkfifo fifo/gul_S1_summary_P23
mkfifo fifo/gul_S1_summary_P23.idx
mkfifo fifo/gul_S1_eltcalc_P23
mkfifo fifo/gul_S1_summarycalc_P23
mkfifo fifo/gul_S1_pltcalc_P23

mkfifo fifo/gul_S1_summary_P24
mkfifo fifo/gul_S1_summary_P24.idx
mkfifo fifo/gul_S1_eltcalc_P24
mkfifo fifo/gul_S1_summarycalc_P24
mkfifo fifo/gul_S1_pltcalc_P24

mkfifo fifo/gul_S1_summary_P25
mkfifo fifo/gul_S1_summary_P25.idx
mkfifo fifo/gul_S1_eltcalc_P25
mkfifo fifo/gul_S1_summarycalc_P25
mkfifo fifo/gul_S1_pltcalc_P25

mkfifo fifo/gul_S1_summary_P26
mkfifo fifo/gul_S1_summary_P26.idx
mkfifo fifo/gul_S1_eltcalc_P26
mkfifo fifo/gul_S1_summarycalc_P26
mkfifo fifo/gul_S1_pltcalc_P26

mkfifo fifo/gul_S1_summary_P27
mkfifo fifo/gul_S1_summary_P27.idx
mkfifo fifo/gul_S1_eltcalc_P27
mkfifo fifo/gul_S1_summarycalc_P27
mkfifo fifo/gul_S1_pltcalc_P27

mkfifo fifo/gul_S1_summary_P28
mkfifo fifo/gul_S1_summary_P28.idx
mkfifo fifo/gul_S1_eltcalc_P28
mkfifo fifo/gul_S1_summarycalc_P28
mkfifo fifo/gul_S1_pltcalc_P28

mkfifo fifo/gul_S1_summary_P29
mkfifo fifo/gul_S1_summary_P29.idx
mkfifo fifo/gul_S1_eltcalc_P29
mkfifo fifo/gul_S1_summarycalc_P29
mkfifo fifo/gul_S1_pltcalc_P29

mkfifo fifo/gul_S1_summary_P30
mkfifo fifo/gul_S1_summary_P30.idx
mkfifo fifo/gul_S1_eltcalc_P30
mkfifo fifo/gul_S1_summarycalc_P30
mkfifo fifo/gul_S1_pltcalc_P30

mkfifo fifo/gul_S1_summary_P31
mkfifo fifo/gul_S1_summary_P31.idx
mkfifo fifo/gul_S1_eltcalc_P31
mkfifo fifo/gul_S1_summarycalc_P31
mkfifo fifo/gul_S1_pltcalc_P31

mkfifo fifo/gul_S1_summary_P32
mkfifo fifo/gul_S1_summary_P32.idx
mkfifo fifo/gul_S1_eltcalc_P32
mkfifo fifo/gul_S1_summarycalc_P32
mkfifo fifo/gul_S1_pltcalc_P32

mkfifo fifo/gul_S1_summary_P33
mkfifo fifo/gul_S1_summary_P33.idx
mkfifo fifo/gul_S1_eltcalc_P33
mkfifo fifo/gul_S1_summarycalc_P33
mkfifo fifo/gul_S1_pltcalc_P33

mkfifo fifo/gul_S1_summary_P34
mkfifo fifo/gul_S1_summary_P34.idx
mkfifo fifo/gul_S1_eltcalc_P34
mkfifo fifo/gul_S1_summarycalc_P34
mkfifo fifo/gul_S1_pltcalc_P34

mkfifo fifo/gul_S1_summary_P35
mkfifo fifo/gul_S1_summary_P35.idx
mkfifo fifo/gul_S1_eltcalc_P35
mkfifo fifo/gul_S1_summarycalc_P35
mkfifo fifo/gul_S1_pltcalc_P35

mkfifo fifo/gul_S1_summary_P36
mkfifo fifo/gul_S1_summary_P36.idx
mkfifo fifo/gul_S1_eltcalc_P36
mkfifo fifo/gul_S1_summarycalc_P36
mkfifo fifo/gul_S1_pltcalc_P36

mkfifo fifo/gul_S1_summary_P37
mkfifo fifo/gul_S1_summary_P37.idx
mkfifo fifo/gul_S1_eltcalc_P37
mkfifo fifo/gul_S1_summarycalc_P37
mkfifo fifo/gul_S1_pltcalc_P37

mkfifo fifo/gul_S1_summary_P38
mkfifo fifo/gul_S1_summary_P38.idx
mkfifo fifo/gul_S1_eltcalc_P38
mkfifo fifo/gul_S1_summarycalc_P38
mkfifo fifo/gul_S1_pltcalc_P38

mkfifo fifo/gul_S1_summary_P39
mkfifo fifo/gul_S1_summary_P39.idx
mkfifo fifo/gul_S1_eltcalc_P39
mkfifo fifo/gul_S1_summarycalc_P39
mkfifo fifo/gul_S1_pltcalc_P39

mkfifo fifo/gul_S1_summary_P40
mkfifo fifo/gul_S1_summary_P40.idx
mkfifo fifo/gul_S1_eltcalc_P40
mkfifo fifo/gul_S1_summarycalc_P40
mkfifo fifo/gul_S1_pltcalc_P40

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4
mkfifo fifo/il_P5
mkfifo fifo/il_P6
mkfifo fifo/il_P7
mkfifo fifo/il_P8
mkfifo fifo/il_P9
mkfifo fifo/il_P10
mkfifo fifo/il_P11
mkfifo fifo/il_P12
mkfifo fifo/il_P13
mkfifo fifo/il_P14
mkfifo fifo/il_P15
mkfifo fifo/il_P16
mkfifo fifo/il_P17
mkfifo fifo/il_P18
mkfifo fifo/il_P19
mkfifo fifo/il_P20
mkfifo fifo/il_P21
mkfifo fifo/il_P22
mkfifo fifo/il_P23
mkfifo fifo/il_P24
mkfifo fifo/il_P25
mkfifo fifo/il_P26
mkfifo fifo/il_P27
mkfifo fifo/il_P28
mkfifo fifo/il_P29
mkfifo fifo/il_P30
mkfifo fifo/il_P31
mkfifo fifo/il_P32
mkfifo fifo/il_P33
mkfifo fifo/il_P34
mkfifo fifo/il_P35
mkfifo fifo/il_P36
mkfifo fifo/il_P37
mkfifo fifo/il_P38
mkfifo fifo/il_P39
mkfifo fifo/il_P40

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summary_P3.idx
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summary_P4.idx
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summary_P5.idx
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summary_P6.idx
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_summary_P8.idx
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summary_P9.idx
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_summary_P10.idx
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summary_P11.idx
mkfifo fifo/il_S1_eltcalc_P11
mkfifo fifo/il_S1_summarycalc_P11
mkfifo fifo/il_S1_pltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_summary_P12.idx
mkfifo fifo/il_S1_eltcalc_P12
mkfifo fifo/il_S1_summarycalc_P12
mkfifo fifo/il_S1_pltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_summary_P13.idx
mkfifo fifo/il_S1_eltcalc_P13
mkfifo fifo/il_S1_summarycalc_P13
mkfifo fifo/il_S1_pltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_summary_P14.idx
mkfifo fifo/il_S1_eltcalc_P14
mkfifo fifo/il_S1_summarycalc_P14
mkfifo fifo/il_S1_pltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summary_P15.idx
mkfifo fifo/il_S1_eltcalc_P15
mkfifo fifo/il_S1_summarycalc_P15
mkfifo fifo/il_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summary_P16.idx
mkfifo fifo/il_S1_eltcalc_P16
mkfifo fifo/il_S1_summarycalc_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_summary_P17.idx
mkfifo fifo/il_S1_eltcalc_P17
mkfifo fifo/il_S1_summarycalc_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_summary_P18.idx
mkfifo fifo/il_S1_eltcalc_P18
mkfifo fifo/il_S1_summarycalc_P18
mkfifo fifo/il_S1_pltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_summary_P19.idx
mkfifo fifo/il_S1_eltcalc_P19
mkfifo fifo/il_S1_summarycalc_P19
mkfifo fifo/il_S1_pltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_summary_P20.idx
mkfifo fifo/il_S1_eltcalc_P20
mkfifo fifo/il_S1_summarycalc_P20
mkfifo fifo/il_S1_pltcalc_P20

mkfifo fifo/il_S1_summary_P21
mkfifo fifo/il_S1_summary_P21.idx
mkfifo fifo/il_S1_eltcalc_P21
mkfifo fifo/il_S1_summarycalc_P21
mkfifo fifo/il_S1_pltcalc_P21

mkfifo fifo/il_S1_summary_P22
mkfifo fifo/il_S1_summary_P22.idx
mkfifo fifo/il_S1_eltcalc_P22
mkfifo fifo/il_S1_summarycalc_P22
mkfifo fifo/il_S1_pltcalc_P22

mkfifo fifo/il_S1_summary_P23
mkfifo fifo/il_S1_summary_P23.idx
mkfifo fifo/il_S1_eltcalc_P23
mkfifo fifo/il_S1_summarycalc_P23
mkfifo fifo/il_S1_pltcalc_P23

mkfifo fifo/il_S1_summary_P24
mkfifo fifo/il_S1_summary_P24.idx
mkfifo fifo/il_S1_eltcalc_P24
mkfifo fifo/il_S1_summarycalc_P24
mkfifo fifo/il_S1_pltcalc_P24

mkfifo fifo/il_S1_summary_P25
mkfifo fifo/il_S1_summary_P25.idx
mkfifo fifo/il_S1_eltcalc_P25
mkfifo fifo/il_S1_summarycalc_P25
mkfifo fifo/il_S1_pltcalc_P25

mkfifo fifo/il_S1_summary_P26
mkfifo fifo/il_S1_summary_P26.idx
mkfifo fifo/il_S1_eltcalc_P26
mkfifo fifo/il_S1_summarycalc_P26
mkfifo fifo/il_S1_pltcalc_P26

mkfifo fifo/il_S1_summary_P27
mkfifo fifo/il_S1_summary_P27.idx
mkfifo fifo/il_S1_eltcalc_P27
mkfifo fifo/il_S1_summarycalc_P27
mkfifo fifo/il_S1_pltcalc_P27

mkfifo fifo/il_S1_summary_P28
mkfifo fifo/il_S1_summary_P28.idx
mkfifo fifo/il_S1_eltcalc_P28
mkfifo fifo/il_S1_summarycalc_P28
mkfifo fifo/il_S1_pltcalc_P28

mkfifo fifo/il_S1_summary_P29
mkfifo fifo/il_S1_summary_P29.idx
mkfifo fifo/il_S1_eltcalc_P29
mkfifo fifo/il_S1_summarycalc_P29
mkfifo fifo/il_S1_pltcalc_P29

mkfifo fifo/il_S1_summary_P30
mkfifo fifo/il_S1_summary_P30.idx
mkfifo fifo/il_S1_eltcalc_P30
mkfifo fifo/il_S1_summarycalc_P30
mkfifo fifo/il_S1_pltcalc_P30

mkfifo fifo/il_S1_summary_P31
mkfifo fifo/il_S1_summary_P31.idx
mkfifo fifo/il_S1_eltcalc_P31
mkfifo fifo/il_S1_summarycalc_P31
mkfifo fifo/il_S1_pltcalc_P31

mkfifo fifo/il_S1_summary_P32
mkfifo fifo/il_S1_summary_P32.idx
mkfifo fifo/il_S1_eltcalc_P32
mkfifo fifo/il_S1_summarycalc_P32
mkfifo fifo/il_S1_pltcalc_P32

mkfifo fifo/il_S1_summary_P33
mkfifo fifo/il_S1_summary_P33.idx
mkfifo fifo/il_S1_eltcalc_P33
mkfifo fifo/il_S1_summarycalc_P33
mkfifo fifo/il_S1_pltcalc_P33

mkfifo fifo/il_S1_summary_P34
mkfifo fifo/il_S1_summary_P34.idx
mkfifo fifo/il_S1_eltcalc_P34
mkfifo fifo/il_S1_summarycalc_P34
mkfifo fifo/il_S1_pltcalc_P34

mkfifo fifo/il_S1_summary_P35
mkfifo fifo/il_S1_summary_P35.idx
mkfifo fifo/il_S1_eltcalc_P35
mkfifo fifo/il_S1_summarycalc_P35
mkfifo fifo/il_S1_pltcalc_P35

mkfifo fifo/il_S1_summary_P36
mkfifo fifo/il_S1_summary_P36.idx
mkfifo fifo/il_S1_eltcalc_P36
mkfifo fifo/il_S1_summarycalc_P36
mkfifo fifo/il_S1_pltcalc_P36

mkfifo fifo/il_S1_summary_P37
mkfifo fifo/il_S1_summary_P37.idx
mkfifo fifo/il_S1_eltcalc_P37
mkfifo fifo/il_S1_summarycalc_P37
mkfifo fifo/il_S1_pltcalc_P37

mkfifo fifo/il_S1_summary_P38
mkfifo fifo/il_S1_summary_P38.idx
mkfifo fifo/il_S1_eltcalc_P38
mkfifo fifo/il_S1_summarycalc_P38
mkfifo fifo/il_S1_pltcalc_P38

mkfifo fifo/il_S1_summary_P39
mkfifo fifo/il_S1_summary_P39.idx
mkfifo fifo/il_S1_eltcalc_P39
mkfifo fifo/il_S1_summarycalc_P39
mkfifo fifo/il_S1_pltcalc_P39

mkfifo fifo/il_S1_summary_P40
mkfifo fifo/il_S1_summary_P40.idx
mkfifo fifo/il_S1_eltcalc_P40
mkfifo fifo/il_S1_summarycalc_P40
mkfifo fifo/il_S1_pltcalc_P40



# --- Do insured loss computes ---

( eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid1=$!
( summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid2=$!
( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid3=$!
( eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid5=$!
( pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid6=$!
( eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid7=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid8=$!
( pltcalc -H < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid9=$!
( eltcalc -s < fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid10=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid11=$!
( pltcalc -H < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid12=$!
( eltcalc -s < fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid13=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid14=$!
( pltcalc -H < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid15=$!
( eltcalc -s < fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid16=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid17=$!
( pltcalc -H < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid18=$!
( eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid19=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid20=$!
( pltcalc -H < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid21=$!
( eltcalc -s < fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid22=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid23=$!
( pltcalc -H < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid24=$!
( eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid25=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid26=$!
( pltcalc -H < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid27=$!
( eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid28=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid29=$!
( pltcalc -H < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid30=$!
( eltcalc -s < fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid31=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 ) 2>> $LOG_DIR/stderror.err & pid32=$!
( pltcalc -H < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid33=$!
( eltcalc -s < fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid34=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P12 > work/kat/il_S1_summarycalc_P12 ) 2>> $LOG_DIR/stderror.err & pid35=$!
( pltcalc -H < fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid36=$!
( eltcalc -s < fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid37=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P13 > work/kat/il_S1_summarycalc_P13 ) 2>> $LOG_DIR/stderror.err & pid38=$!
( pltcalc -H < fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid39=$!
( eltcalc -s < fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid40=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P14 > work/kat/il_S1_summarycalc_P14 ) 2>> $LOG_DIR/stderror.err & pid41=$!
( pltcalc -H < fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid42=$!
( eltcalc -s < fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid43=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P15 > work/kat/il_S1_summarycalc_P15 ) 2>> $LOG_DIR/stderror.err & pid44=$!
( pltcalc -H < fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid45=$!
( eltcalc -s < fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid46=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P16 > work/kat/il_S1_summarycalc_P16 ) 2>> $LOG_DIR/stderror.err & pid47=$!
( pltcalc -H < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid48=$!
( eltcalc -s < fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid49=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 ) 2>> $LOG_DIR/stderror.err & pid50=$!
( pltcalc -H < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid51=$!
( eltcalc -s < fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid52=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 ) 2>> $LOG_DIR/stderror.err & pid53=$!
( pltcalc -H < fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid54=$!
( eltcalc -s < fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid55=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P19 > work/kat/il_S1_summarycalc_P19 ) 2>> $LOG_DIR/stderror.err & pid56=$!
( pltcalc -H < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid57=$!
( eltcalc -s < fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid58=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P20 > work/kat/il_S1_summarycalc_P20 ) 2>> $LOG_DIR/stderror.err & pid59=$!
( pltcalc -H < fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid60=$!
( eltcalc -s < fifo/il_S1_eltcalc_P21 > work/kat/il_S1_eltcalc_P21 ) 2>> $LOG_DIR/stderror.err & pid61=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P21 > work/kat/il_S1_summarycalc_P21 ) 2>> $LOG_DIR/stderror.err & pid62=$!
( pltcalc -H < fifo/il_S1_pltcalc_P21 > work/kat/il_S1_pltcalc_P21 ) 2>> $LOG_DIR/stderror.err & pid63=$!
( eltcalc -s < fifo/il_S1_eltcalc_P22 > work/kat/il_S1_eltcalc_P22 ) 2>> $LOG_DIR/stderror.err & pid64=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P22 > work/kat/il_S1_summarycalc_P22 ) 2>> $LOG_DIR/stderror.err & pid65=$!
( pltcalc -H < fifo/il_S1_pltcalc_P22 > work/kat/il_S1_pltcalc_P22 ) 2>> $LOG_DIR/stderror.err & pid66=$!
( eltcalc -s < fifo/il_S1_eltcalc_P23 > work/kat/il_S1_eltcalc_P23 ) 2>> $LOG_DIR/stderror.err & pid67=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P23 > work/kat/il_S1_summarycalc_P23 ) 2>> $LOG_DIR/stderror.err & pid68=$!
( pltcalc -H < fifo/il_S1_pltcalc_P23 > work/kat/il_S1_pltcalc_P23 ) 2>> $LOG_DIR/stderror.err & pid69=$!
( eltcalc -s < fifo/il_S1_eltcalc_P24 > work/kat/il_S1_eltcalc_P24 ) 2>> $LOG_DIR/stderror.err & pid70=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P24 > work/kat/il_S1_summarycalc_P24 ) 2>> $LOG_DIR/stderror.err & pid71=$!
( pltcalc -H < fifo/il_S1_pltcalc_P24 > work/kat/il_S1_pltcalc_P24 ) 2>> $LOG_DIR/stderror.err & pid72=$!
( eltcalc -s < fifo/il_S1_eltcalc_P25 > work/kat/il_S1_eltcalc_P25 ) 2>> $LOG_DIR/stderror.err & pid73=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P25 > work/kat/il_S1_summarycalc_P25 ) 2>> $LOG_DIR/stderror.err & pid74=$!
( pltcalc -H < fifo/il_S1_pltcalc_P25 > work/kat/il_S1_pltcalc_P25 ) 2>> $LOG_DIR/stderror.err & pid75=$!
( eltcalc -s < fifo/il_S1_eltcalc_P26 > work/kat/il_S1_eltcalc_P26 ) 2>> $LOG_DIR/stderror.err & pid76=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P26 > work/kat/il_S1_summarycalc_P26 ) 2>> $LOG_DIR/stderror.err & pid77=$!
( pltcalc -H < fifo/il_S1_pltcalc_P26 > work/kat/il_S1_pltcalc_P26 ) 2>> $LOG_DIR/stderror.err & pid78=$!
( eltcalc -s < fifo/il_S1_eltcalc_P27 > work/kat/il_S1_eltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid79=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P27 > work/kat/il_S1_summarycalc_P27 ) 2>> $LOG_DIR/stderror.err & pid80=$!
( pltcalc -H < fifo/il_S1_pltcalc_P27 > work/kat/il_S1_pltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid81=$!
( eltcalc -s < fifo/il_S1_eltcalc_P28 > work/kat/il_S1_eltcalc_P28 ) 2>> $LOG_DIR/stderror.err & pid82=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P28 > work/kat/il_S1_summarycalc_P28 ) 2>> $LOG_DIR/stderror.err & pid83=$!
( pltcalc -H < fifo/il_S1_pltcalc_P28 > work/kat/il_S1_pltcalc_P28 ) 2>> $LOG_DIR/stderror.err & pid84=$!
( eltcalc -s < fifo/il_S1_eltcalc_P29 > work/kat/il_S1_eltcalc_P29 ) 2>> $LOG_DIR/stderror.err & pid85=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P29 > work/kat/il_S1_summarycalc_P29 ) 2>> $LOG_DIR/stderror.err & pid86=$!
( pltcalc -H < fifo/il_S1_pltcalc_P29 > work/kat/il_S1_pltcalc_P29 ) 2>> $LOG_DIR/stderror.err & pid87=$!
( eltcalc -s < fifo/il_S1_eltcalc_P30 > work/kat/il_S1_eltcalc_P30 ) 2>> $LOG_DIR/stderror.err & pid88=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P30 > work/kat/il_S1_summarycalc_P30 ) 2>> $LOG_DIR/stderror.err & pid89=$!
( pltcalc -H < fifo/il_S1_pltcalc_P30 > work/kat/il_S1_pltcalc_P30 ) 2>> $LOG_DIR/stderror.err & pid90=$!
( eltcalc -s < fifo/il_S1_eltcalc_P31 > work/kat/il_S1_eltcalc_P31 ) 2>> $LOG_DIR/stderror.err & pid91=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P31 > work/kat/il_S1_summarycalc_P31 ) 2>> $LOG_DIR/stderror.err & pid92=$!
( pltcalc -H < fifo/il_S1_pltcalc_P31 > work/kat/il_S1_pltcalc_P31 ) 2>> $LOG_DIR/stderror.err & pid93=$!
( eltcalc -s < fifo/il_S1_eltcalc_P32 > work/kat/il_S1_eltcalc_P32 ) 2>> $LOG_DIR/stderror.err & pid94=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P32 > work/kat/il_S1_summarycalc_P32 ) 2>> $LOG_DIR/stderror.err & pid95=$!
( pltcalc -H < fifo/il_S1_pltcalc_P32 > work/kat/il_S1_pltcalc_P32 ) 2>> $LOG_DIR/stderror.err & pid96=$!
( eltcalc -s < fifo/il_S1_eltcalc_P33 > work/kat/il_S1_eltcalc_P33 ) 2>> $LOG_DIR/stderror.err & pid97=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P33 > work/kat/il_S1_summarycalc_P33 ) 2>> $LOG_DIR/stderror.err & pid98=$!
( pltcalc -H < fifo/il_S1_pltcalc_P33 > work/kat/il_S1_pltcalc_P33 ) 2>> $LOG_DIR/stderror.err & pid99=$!
( eltcalc -s < fifo/il_S1_eltcalc_P34 > work/kat/il_S1_eltcalc_P34 ) 2>> $LOG_DIR/stderror.err & pid100=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P34 > work/kat/il_S1_summarycalc_P34 ) 2>> $LOG_DIR/stderror.err & pid101=$!
( pltcalc -H < fifo/il_S1_pltcalc_P34 > work/kat/il_S1_pltcalc_P34 ) 2>> $LOG_DIR/stderror.err & pid102=$!
( eltcalc -s < fifo/il_S1_eltcalc_P35 > work/kat/il_S1_eltcalc_P35 ) 2>> $LOG_DIR/stderror.err & pid103=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P35 > work/kat/il_S1_summarycalc_P35 ) 2>> $LOG_DIR/stderror.err & pid104=$!
( pltcalc -H < fifo/il_S1_pltcalc_P35 > work/kat/il_S1_pltcalc_P35 ) 2>> $LOG_DIR/stderror.err & pid105=$!
( eltcalc -s < fifo/il_S1_eltcalc_P36 > work/kat/il_S1_eltcalc_P36 ) 2>> $LOG_DIR/stderror.err & pid106=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P36 > work/kat/il_S1_summarycalc_P36 ) 2>> $LOG_DIR/stderror.err & pid107=$!
( pltcalc -H < fifo/il_S1_pltcalc_P36 > work/kat/il_S1_pltcalc_P36 ) 2>> $LOG_DIR/stderror.err & pid108=$!
( eltcalc -s < fifo/il_S1_eltcalc_P37 > work/kat/il_S1_eltcalc_P37 ) 2>> $LOG_DIR/stderror.err & pid109=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P37 > work/kat/il_S1_summarycalc_P37 ) 2>> $LOG_DIR/stderror.err & pid110=$!
( pltcalc -H < fifo/il_S1_pltcalc_P37 > work/kat/il_S1_pltcalc_P37 ) 2>> $LOG_DIR/stderror.err & pid111=$!
( eltcalc -s < fifo/il_S1_eltcalc_P38 > work/kat/il_S1_eltcalc_P38 ) 2>> $LOG_DIR/stderror.err & pid112=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P38 > work/kat/il_S1_summarycalc_P38 ) 2>> $LOG_DIR/stderror.err & pid113=$!
( pltcalc -H < fifo/il_S1_pltcalc_P38 > work/kat/il_S1_pltcalc_P38 ) 2>> $LOG_DIR/stderror.err & pid114=$!
( eltcalc -s < fifo/il_S1_eltcalc_P39 > work/kat/il_S1_eltcalc_P39 ) 2>> $LOG_DIR/stderror.err & pid115=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P39 > work/kat/il_S1_summarycalc_P39 ) 2>> $LOG_DIR/stderror.err & pid116=$!
( pltcalc -H < fifo/il_S1_pltcalc_P39 > work/kat/il_S1_pltcalc_P39 ) 2>> $LOG_DIR/stderror.err & pid117=$!
( eltcalc -s < fifo/il_S1_eltcalc_P40 > work/kat/il_S1_eltcalc_P40 ) 2>> $LOG_DIR/stderror.err & pid118=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P40 > work/kat/il_S1_summarycalc_P40 ) 2>> $LOG_DIR/stderror.err & pid119=$!
( pltcalc -H < fifo/il_S1_pltcalc_P40 > work/kat/il_S1_pltcalc_P40 ) 2>> $LOG_DIR/stderror.err & pid120=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid121=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid122=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid123=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid124=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 fifo/il_S1_summarycalc_P3 fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid125=$!
tee < fifo/il_S1_summary_P3.idx work/il_S1_summaryaalcalc/P3.idx work/il_S1_summaryleccalc/P3.idx > /dev/null & pid126=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_eltcalc_P4 fifo/il_S1_summarycalc_P4 fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid127=$!
tee < fifo/il_S1_summary_P4.idx work/il_S1_summaryaalcalc/P4.idx work/il_S1_summaryleccalc/P4.idx > /dev/null & pid128=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_eltcalc_P5 fifo/il_S1_summarycalc_P5 fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid129=$!
tee < fifo/il_S1_summary_P5.idx work/il_S1_summaryaalcalc/P5.idx work/il_S1_summaryleccalc/P5.idx > /dev/null & pid130=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_eltcalc_P6 fifo/il_S1_summarycalc_P6 fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid131=$!
tee < fifo/il_S1_summary_P6.idx work/il_S1_summaryaalcalc/P6.idx work/il_S1_summaryleccalc/P6.idx > /dev/null & pid132=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 fifo/il_S1_summarycalc_P7 fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid133=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryaalcalc/P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid134=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_eltcalc_P8 fifo/il_S1_summarycalc_P8 fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid135=$!
tee < fifo/il_S1_summary_P8.idx work/il_S1_summaryaalcalc/P8.idx work/il_S1_summaryleccalc/P8.idx > /dev/null & pid136=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid137=$!
tee < fifo/il_S1_summary_P9.idx work/il_S1_summaryaalcalc/P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid138=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid139=$!
tee < fifo/il_S1_summary_P10.idx work/il_S1_summaryaalcalc/P10.idx work/il_S1_summaryleccalc/P10.idx > /dev/null & pid140=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_eltcalc_P11 fifo/il_S1_summarycalc_P11 fifo/il_S1_pltcalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid141=$!
tee < fifo/il_S1_summary_P11.idx work/il_S1_summaryaalcalc/P11.idx work/il_S1_summaryleccalc/P11.idx > /dev/null & pid142=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_eltcalc_P12 fifo/il_S1_summarycalc_P12 fifo/il_S1_pltcalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid143=$!
tee < fifo/il_S1_summary_P12.idx work/il_S1_summaryaalcalc/P12.idx work/il_S1_summaryleccalc/P12.idx > /dev/null & pid144=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_eltcalc_P13 fifo/il_S1_summarycalc_P13 fifo/il_S1_pltcalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid145=$!
tee < fifo/il_S1_summary_P13.idx work/il_S1_summaryaalcalc/P13.idx work/il_S1_summaryleccalc/P13.idx > /dev/null & pid146=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_eltcalc_P14 fifo/il_S1_summarycalc_P14 fifo/il_S1_pltcalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid147=$!
tee < fifo/il_S1_summary_P14.idx work/il_S1_summaryaalcalc/P14.idx work/il_S1_summaryleccalc/P14.idx > /dev/null & pid148=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_eltcalc_P15 fifo/il_S1_summarycalc_P15 fifo/il_S1_pltcalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid149=$!
tee < fifo/il_S1_summary_P15.idx work/il_S1_summaryaalcalc/P15.idx work/il_S1_summaryleccalc/P15.idx > /dev/null & pid150=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_eltcalc_P16 fifo/il_S1_summarycalc_P16 fifo/il_S1_pltcalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid151=$!
tee < fifo/il_S1_summary_P16.idx work/il_S1_summaryaalcalc/P16.idx work/il_S1_summaryleccalc/P16.idx > /dev/null & pid152=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_eltcalc_P17 fifo/il_S1_summarycalc_P17 fifo/il_S1_pltcalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid153=$!
tee < fifo/il_S1_summary_P17.idx work/il_S1_summaryaalcalc/P17.idx work/il_S1_summaryleccalc/P17.idx > /dev/null & pid154=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_eltcalc_P18 fifo/il_S1_summarycalc_P18 fifo/il_S1_pltcalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid155=$!
tee < fifo/il_S1_summary_P18.idx work/il_S1_summaryaalcalc/P18.idx work/il_S1_summaryleccalc/P18.idx > /dev/null & pid156=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_eltcalc_P19 fifo/il_S1_summarycalc_P19 fifo/il_S1_pltcalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid157=$!
tee < fifo/il_S1_summary_P19.idx work/il_S1_summaryaalcalc/P19.idx work/il_S1_summaryleccalc/P19.idx > /dev/null & pid158=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_eltcalc_P20 fifo/il_S1_summarycalc_P20 fifo/il_S1_pltcalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid159=$!
tee < fifo/il_S1_summary_P20.idx work/il_S1_summaryaalcalc/P20.idx work/il_S1_summaryleccalc/P20.idx > /dev/null & pid160=$!
tee < fifo/il_S1_summary_P21 fifo/il_S1_eltcalc_P21 fifo/il_S1_summarycalc_P21 fifo/il_S1_pltcalc_P21 work/il_S1_summaryaalcalc/P21.bin work/il_S1_summaryleccalc/P21.bin > /dev/null & pid161=$!
tee < fifo/il_S1_summary_P21.idx work/il_S1_summaryaalcalc/P21.idx work/il_S1_summaryleccalc/P21.idx > /dev/null & pid162=$!
tee < fifo/il_S1_summary_P22 fifo/il_S1_eltcalc_P22 fifo/il_S1_summarycalc_P22 fifo/il_S1_pltcalc_P22 work/il_S1_summaryaalcalc/P22.bin work/il_S1_summaryleccalc/P22.bin > /dev/null & pid163=$!
tee < fifo/il_S1_summary_P22.idx work/il_S1_summaryaalcalc/P22.idx work/il_S1_summaryleccalc/P22.idx > /dev/null & pid164=$!
tee < fifo/il_S1_summary_P23 fifo/il_S1_eltcalc_P23 fifo/il_S1_summarycalc_P23 fifo/il_S1_pltcalc_P23 work/il_S1_summaryaalcalc/P23.bin work/il_S1_summaryleccalc/P23.bin > /dev/null & pid165=$!
tee < fifo/il_S1_summary_P23.idx work/il_S1_summaryaalcalc/P23.idx work/il_S1_summaryleccalc/P23.idx > /dev/null & pid166=$!
tee < fifo/il_S1_summary_P24 fifo/il_S1_eltcalc_P24 fifo/il_S1_summarycalc_P24 fifo/il_S1_pltcalc_P24 work/il_S1_summaryaalcalc/P24.bin work/il_S1_summaryleccalc/P24.bin > /dev/null & pid167=$!
tee < fifo/il_S1_summary_P24.idx work/il_S1_summaryaalcalc/P24.idx work/il_S1_summaryleccalc/P24.idx > /dev/null & pid168=$!
tee < fifo/il_S1_summary_P25 fifo/il_S1_eltcalc_P25 fifo/il_S1_summarycalc_P25 fifo/il_S1_pltcalc_P25 work/il_S1_summaryaalcalc/P25.bin work/il_S1_summaryleccalc/P25.bin > /dev/null & pid169=$!
tee < fifo/il_S1_summary_P25.idx work/il_S1_summaryaalcalc/P25.idx work/il_S1_summaryleccalc/P25.idx > /dev/null & pid170=$!
tee < fifo/il_S1_summary_P26 fifo/il_S1_eltcalc_P26 fifo/il_S1_summarycalc_P26 fifo/il_S1_pltcalc_P26 work/il_S1_summaryaalcalc/P26.bin work/il_S1_summaryleccalc/P26.bin > /dev/null & pid171=$!
tee < fifo/il_S1_summary_P26.idx work/il_S1_summaryaalcalc/P26.idx work/il_S1_summaryleccalc/P26.idx > /dev/null & pid172=$!
tee < fifo/il_S1_summary_P27 fifo/il_S1_eltcalc_P27 fifo/il_S1_summarycalc_P27 fifo/il_S1_pltcalc_P27 work/il_S1_summaryaalcalc/P27.bin work/il_S1_summaryleccalc/P27.bin > /dev/null & pid173=$!
tee < fifo/il_S1_summary_P27.idx work/il_S1_summaryaalcalc/P27.idx work/il_S1_summaryleccalc/P27.idx > /dev/null & pid174=$!
tee < fifo/il_S1_summary_P28 fifo/il_S1_eltcalc_P28 fifo/il_S1_summarycalc_P28 fifo/il_S1_pltcalc_P28 work/il_S1_summaryaalcalc/P28.bin work/il_S1_summaryleccalc/P28.bin > /dev/null & pid175=$!
tee < fifo/il_S1_summary_P28.idx work/il_S1_summaryaalcalc/P28.idx work/il_S1_summaryleccalc/P28.idx > /dev/null & pid176=$!
tee < fifo/il_S1_summary_P29 fifo/il_S1_eltcalc_P29 fifo/il_S1_summarycalc_P29 fifo/il_S1_pltcalc_P29 work/il_S1_summaryaalcalc/P29.bin work/il_S1_summaryleccalc/P29.bin > /dev/null & pid177=$!
tee < fifo/il_S1_summary_P29.idx work/il_S1_summaryaalcalc/P29.idx work/il_S1_summaryleccalc/P29.idx > /dev/null & pid178=$!
tee < fifo/il_S1_summary_P30 fifo/il_S1_eltcalc_P30 fifo/il_S1_summarycalc_P30 fifo/il_S1_pltcalc_P30 work/il_S1_summaryaalcalc/P30.bin work/il_S1_summaryleccalc/P30.bin > /dev/null & pid179=$!
tee < fifo/il_S1_summary_P30.idx work/il_S1_summaryaalcalc/P30.idx work/il_S1_summaryleccalc/P30.idx > /dev/null & pid180=$!
tee < fifo/il_S1_summary_P31 fifo/il_S1_eltcalc_P31 fifo/il_S1_summarycalc_P31 fifo/il_S1_pltcalc_P31 work/il_S1_summaryaalcalc/P31.bin work/il_S1_summaryleccalc/P31.bin > /dev/null & pid181=$!
tee < fifo/il_S1_summary_P31.idx work/il_S1_summaryaalcalc/P31.idx work/il_S1_summaryleccalc/P31.idx > /dev/null & pid182=$!
tee < fifo/il_S1_summary_P32 fifo/il_S1_eltcalc_P32 fifo/il_S1_summarycalc_P32 fifo/il_S1_pltcalc_P32 work/il_S1_summaryaalcalc/P32.bin work/il_S1_summaryleccalc/P32.bin > /dev/null & pid183=$!
tee < fifo/il_S1_summary_P32.idx work/il_S1_summaryaalcalc/P32.idx work/il_S1_summaryleccalc/P32.idx > /dev/null & pid184=$!
tee < fifo/il_S1_summary_P33 fifo/il_S1_eltcalc_P33 fifo/il_S1_summarycalc_P33 fifo/il_S1_pltcalc_P33 work/il_S1_summaryaalcalc/P33.bin work/il_S1_summaryleccalc/P33.bin > /dev/null & pid185=$!
tee < fifo/il_S1_summary_P33.idx work/il_S1_summaryaalcalc/P33.idx work/il_S1_summaryleccalc/P33.idx > /dev/null & pid186=$!
tee < fifo/il_S1_summary_P34 fifo/il_S1_eltcalc_P34 fifo/il_S1_summarycalc_P34 fifo/il_S1_pltcalc_P34 work/il_S1_summaryaalcalc/P34.bin work/il_S1_summaryleccalc/P34.bin > /dev/null & pid187=$!
tee < fifo/il_S1_summary_P34.idx work/il_S1_summaryaalcalc/P34.idx work/il_S1_summaryleccalc/P34.idx > /dev/null & pid188=$!
tee < fifo/il_S1_summary_P35 fifo/il_S1_eltcalc_P35 fifo/il_S1_summarycalc_P35 fifo/il_S1_pltcalc_P35 work/il_S1_summaryaalcalc/P35.bin work/il_S1_summaryleccalc/P35.bin > /dev/null & pid189=$!
tee < fifo/il_S1_summary_P35.idx work/il_S1_summaryaalcalc/P35.idx work/il_S1_summaryleccalc/P35.idx > /dev/null & pid190=$!
tee < fifo/il_S1_summary_P36 fifo/il_S1_eltcalc_P36 fifo/il_S1_summarycalc_P36 fifo/il_S1_pltcalc_P36 work/il_S1_summaryaalcalc/P36.bin work/il_S1_summaryleccalc/P36.bin > /dev/null & pid191=$!
tee < fifo/il_S1_summary_P36.idx work/il_S1_summaryaalcalc/P36.idx work/il_S1_summaryleccalc/P36.idx > /dev/null & pid192=$!
tee < fifo/il_S1_summary_P37 fifo/il_S1_eltcalc_P37 fifo/il_S1_summarycalc_P37 fifo/il_S1_pltcalc_P37 work/il_S1_summaryaalcalc/P37.bin work/il_S1_summaryleccalc/P37.bin > /dev/null & pid193=$!
tee < fifo/il_S1_summary_P37.idx work/il_S1_summaryaalcalc/P37.idx work/il_S1_summaryleccalc/P37.idx > /dev/null & pid194=$!
tee < fifo/il_S1_summary_P38 fifo/il_S1_eltcalc_P38 fifo/il_S1_summarycalc_P38 fifo/il_S1_pltcalc_P38 work/il_S1_summaryaalcalc/P38.bin work/il_S1_summaryleccalc/P38.bin > /dev/null & pid195=$!
tee < fifo/il_S1_summary_P38.idx work/il_S1_summaryaalcalc/P38.idx work/il_S1_summaryleccalc/P38.idx > /dev/null & pid196=$!
tee < fifo/il_S1_summary_P39 fifo/il_S1_eltcalc_P39 fifo/il_S1_summarycalc_P39 fifo/il_S1_pltcalc_P39 work/il_S1_summaryaalcalc/P39.bin work/il_S1_summaryleccalc/P39.bin > /dev/null & pid197=$!
tee < fifo/il_S1_summary_P39.idx work/il_S1_summaryaalcalc/P39.idx work/il_S1_summaryleccalc/P39.idx > /dev/null & pid198=$!
tee < fifo/il_S1_summary_P40 fifo/il_S1_eltcalc_P40 fifo/il_S1_summarycalc_P40 fifo/il_S1_pltcalc_P40 work/il_S1_summaryaalcalc/P40.bin work/il_S1_summaryleccalc/P40.bin > /dev/null & pid199=$!
tee < fifo/il_S1_summary_P40.idx work/il_S1_summaryaalcalc/P40.idx work/il_S1_summaryleccalc/P40.idx > /dev/null & pid200=$!

( summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P21 < fifo/il_P21 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P22 < fifo/il_P22 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P23 < fifo/il_P23 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P24 < fifo/il_P24 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P25 < fifo/il_P25 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P26 < fifo/il_P26 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P27 < fifo/il_P27 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P28 < fifo/il_P28 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P29 < fifo/il_P29 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P30 < fifo/il_P30 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P31 < fifo/il_P31 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P32 < fifo/il_P32 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P33 < fifo/il_P33 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P34 < fifo/il_P34 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P35 < fifo/il_P35 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P36 < fifo/il_P36 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P37 < fifo/il_P37 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P38 < fifo/il_P38 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P39 < fifo/il_P39 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -f  -1 fifo/il_S1_summary_P40 < fifo/il_P40 ) 2>> $LOG_DIR/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid201=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> $LOG_DIR/stderror.err & pid202=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid203=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid204=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> $LOG_DIR/stderror.err & pid205=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 ) 2>> $LOG_DIR/stderror.err & pid206=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid207=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 ) 2>> $LOG_DIR/stderror.err & pid208=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 ) 2>> $LOG_DIR/stderror.err & pid209=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid210=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 ) 2>> $LOG_DIR/stderror.err & pid211=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 ) 2>> $LOG_DIR/stderror.err & pid212=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid213=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 ) 2>> $LOG_DIR/stderror.err & pid214=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 ) 2>> $LOG_DIR/stderror.err & pid215=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid216=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 ) 2>> $LOG_DIR/stderror.err & pid217=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 ) 2>> $LOG_DIR/stderror.err & pid218=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid219=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 ) 2>> $LOG_DIR/stderror.err & pid220=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 ) 2>> $LOG_DIR/stderror.err & pid221=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid222=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 ) 2>> $LOG_DIR/stderror.err & pid223=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 ) 2>> $LOG_DIR/stderror.err & pid224=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid225=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> $LOG_DIR/stderror.err & pid226=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 ) 2>> $LOG_DIR/stderror.err & pid227=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid228=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 ) 2>> $LOG_DIR/stderror.err & pid229=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 ) 2>> $LOG_DIR/stderror.err & pid230=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid231=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 ) 2>> $LOG_DIR/stderror.err & pid232=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 ) 2>> $LOG_DIR/stderror.err & pid233=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid234=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 ) 2>> $LOG_DIR/stderror.err & pid235=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P12 > work/kat/gul_S1_pltcalc_P12 ) 2>> $LOG_DIR/stderror.err & pid236=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid237=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 ) 2>> $LOG_DIR/stderror.err & pid238=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P13 > work/kat/gul_S1_pltcalc_P13 ) 2>> $LOG_DIR/stderror.err & pid239=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid240=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 ) 2>> $LOG_DIR/stderror.err & pid241=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 ) 2>> $LOG_DIR/stderror.err & pid242=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid243=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 ) 2>> $LOG_DIR/stderror.err & pid244=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P15 > work/kat/gul_S1_pltcalc_P15 ) 2>> $LOG_DIR/stderror.err & pid245=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid246=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 ) 2>> $LOG_DIR/stderror.err & pid247=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P16 > work/kat/gul_S1_pltcalc_P16 ) 2>> $LOG_DIR/stderror.err & pid248=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid249=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 ) 2>> $LOG_DIR/stderror.err & pid250=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 ) 2>> $LOG_DIR/stderror.err & pid251=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P18 > work/kat/gul_S1_eltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid252=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 ) 2>> $LOG_DIR/stderror.err & pid253=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 ) 2>> $LOG_DIR/stderror.err & pid254=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P19 > work/kat/gul_S1_eltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid255=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 ) 2>> $LOG_DIR/stderror.err & pid256=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P19 > work/kat/gul_S1_pltcalc_P19 ) 2>> $LOG_DIR/stderror.err & pid257=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid258=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 ) 2>> $LOG_DIR/stderror.err & pid259=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P20 > work/kat/gul_S1_pltcalc_P20 ) 2>> $LOG_DIR/stderror.err & pid260=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P21 > work/kat/gul_S1_eltcalc_P21 ) 2>> $LOG_DIR/stderror.err & pid261=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P21 > work/kat/gul_S1_summarycalc_P21 ) 2>> $LOG_DIR/stderror.err & pid262=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P21 > work/kat/gul_S1_pltcalc_P21 ) 2>> $LOG_DIR/stderror.err & pid263=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P22 > work/kat/gul_S1_eltcalc_P22 ) 2>> $LOG_DIR/stderror.err & pid264=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P22 > work/kat/gul_S1_summarycalc_P22 ) 2>> $LOG_DIR/stderror.err & pid265=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P22 > work/kat/gul_S1_pltcalc_P22 ) 2>> $LOG_DIR/stderror.err & pid266=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P23 > work/kat/gul_S1_eltcalc_P23 ) 2>> $LOG_DIR/stderror.err & pid267=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P23 > work/kat/gul_S1_summarycalc_P23 ) 2>> $LOG_DIR/stderror.err & pid268=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P23 > work/kat/gul_S1_pltcalc_P23 ) 2>> $LOG_DIR/stderror.err & pid269=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P24 > work/kat/gul_S1_eltcalc_P24 ) 2>> $LOG_DIR/stderror.err & pid270=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P24 > work/kat/gul_S1_summarycalc_P24 ) 2>> $LOG_DIR/stderror.err & pid271=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P24 > work/kat/gul_S1_pltcalc_P24 ) 2>> $LOG_DIR/stderror.err & pid272=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P25 > work/kat/gul_S1_eltcalc_P25 ) 2>> $LOG_DIR/stderror.err & pid273=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P25 > work/kat/gul_S1_summarycalc_P25 ) 2>> $LOG_DIR/stderror.err & pid274=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P25 > work/kat/gul_S1_pltcalc_P25 ) 2>> $LOG_DIR/stderror.err & pid275=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P26 > work/kat/gul_S1_eltcalc_P26 ) 2>> $LOG_DIR/stderror.err & pid276=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P26 > work/kat/gul_S1_summarycalc_P26 ) 2>> $LOG_DIR/stderror.err & pid277=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P26 > work/kat/gul_S1_pltcalc_P26 ) 2>> $LOG_DIR/stderror.err & pid278=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P27 > work/kat/gul_S1_eltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid279=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P27 > work/kat/gul_S1_summarycalc_P27 ) 2>> $LOG_DIR/stderror.err & pid280=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P27 > work/kat/gul_S1_pltcalc_P27 ) 2>> $LOG_DIR/stderror.err & pid281=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P28 > work/kat/gul_S1_eltcalc_P28 ) 2>> $LOG_DIR/stderror.err & pid282=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P28 > work/kat/gul_S1_summarycalc_P28 ) 2>> $LOG_DIR/stderror.err & pid283=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P28 > work/kat/gul_S1_pltcalc_P28 ) 2>> $LOG_DIR/stderror.err & pid284=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P29 > work/kat/gul_S1_eltcalc_P29 ) 2>> $LOG_DIR/stderror.err & pid285=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P29 > work/kat/gul_S1_summarycalc_P29 ) 2>> $LOG_DIR/stderror.err & pid286=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P29 > work/kat/gul_S1_pltcalc_P29 ) 2>> $LOG_DIR/stderror.err & pid287=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P30 > work/kat/gul_S1_eltcalc_P30 ) 2>> $LOG_DIR/stderror.err & pid288=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P30 > work/kat/gul_S1_summarycalc_P30 ) 2>> $LOG_DIR/stderror.err & pid289=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P30 > work/kat/gul_S1_pltcalc_P30 ) 2>> $LOG_DIR/stderror.err & pid290=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P31 > work/kat/gul_S1_eltcalc_P31 ) 2>> $LOG_DIR/stderror.err & pid291=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P31 > work/kat/gul_S1_summarycalc_P31 ) 2>> $LOG_DIR/stderror.err & pid292=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P31 > work/kat/gul_S1_pltcalc_P31 ) 2>> $LOG_DIR/stderror.err & pid293=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P32 > work/kat/gul_S1_eltcalc_P32 ) 2>> $LOG_DIR/stderror.err & pid294=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P32 > work/kat/gul_S1_summarycalc_P32 ) 2>> $LOG_DIR/stderror.err & pid295=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P32 > work/kat/gul_S1_pltcalc_P32 ) 2>> $LOG_DIR/stderror.err & pid296=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P33 > work/kat/gul_S1_eltcalc_P33 ) 2>> $LOG_DIR/stderror.err & pid297=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P33 > work/kat/gul_S1_summarycalc_P33 ) 2>> $LOG_DIR/stderror.err & pid298=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P33 > work/kat/gul_S1_pltcalc_P33 ) 2>> $LOG_DIR/stderror.err & pid299=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P34 > work/kat/gul_S1_eltcalc_P34 ) 2>> $LOG_DIR/stderror.err & pid300=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P34 > work/kat/gul_S1_summarycalc_P34 ) 2>> $LOG_DIR/stderror.err & pid301=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P34 > work/kat/gul_S1_pltcalc_P34 ) 2>> $LOG_DIR/stderror.err & pid302=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P35 > work/kat/gul_S1_eltcalc_P35 ) 2>> $LOG_DIR/stderror.err & pid303=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P35 > work/kat/gul_S1_summarycalc_P35 ) 2>> $LOG_DIR/stderror.err & pid304=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P35 > work/kat/gul_S1_pltcalc_P35 ) 2>> $LOG_DIR/stderror.err & pid305=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P36 > work/kat/gul_S1_eltcalc_P36 ) 2>> $LOG_DIR/stderror.err & pid306=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P36 > work/kat/gul_S1_summarycalc_P36 ) 2>> $LOG_DIR/stderror.err & pid307=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P36 > work/kat/gul_S1_pltcalc_P36 ) 2>> $LOG_DIR/stderror.err & pid308=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P37 > work/kat/gul_S1_eltcalc_P37 ) 2>> $LOG_DIR/stderror.err & pid309=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P37 > work/kat/gul_S1_summarycalc_P37 ) 2>> $LOG_DIR/stderror.err & pid310=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P37 > work/kat/gul_S1_pltcalc_P37 ) 2>> $LOG_DIR/stderror.err & pid311=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P38 > work/kat/gul_S1_eltcalc_P38 ) 2>> $LOG_DIR/stderror.err & pid312=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P38 > work/kat/gul_S1_summarycalc_P38 ) 2>> $LOG_DIR/stderror.err & pid313=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P38 > work/kat/gul_S1_pltcalc_P38 ) 2>> $LOG_DIR/stderror.err & pid314=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P39 > work/kat/gul_S1_eltcalc_P39 ) 2>> $LOG_DIR/stderror.err & pid315=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P39 > work/kat/gul_S1_summarycalc_P39 ) 2>> $LOG_DIR/stderror.err & pid316=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P39 > work/kat/gul_S1_pltcalc_P39 ) 2>> $LOG_DIR/stderror.err & pid317=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P40 > work/kat/gul_S1_eltcalc_P40 ) 2>> $LOG_DIR/stderror.err & pid318=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P40 > work/kat/gul_S1_summarycalc_P40 ) 2>> $LOG_DIR/stderror.err & pid319=$!
( pltcalc -H < fifo/gul_S1_pltcalc_P40 > work/kat/gul_S1_pltcalc_P40 ) 2>> $LOG_DIR/stderror.err & pid320=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid321=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid322=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid323=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx work/gul_S1_summaryleccalc/P2.idx > /dev/null & pid324=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid325=$!
tee < fifo/gul_S1_summary_P3.idx work/gul_S1_summaryaalcalc/P3.idx work/gul_S1_summaryleccalc/P3.idx > /dev/null & pid326=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid327=$!
tee < fifo/gul_S1_summary_P4.idx work/gul_S1_summaryaalcalc/P4.idx work/gul_S1_summaryleccalc/P4.idx > /dev/null & pid328=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid329=$!
tee < fifo/gul_S1_summary_P5.idx work/gul_S1_summaryaalcalc/P5.idx work/gul_S1_summaryleccalc/P5.idx > /dev/null & pid330=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid331=$!
tee < fifo/gul_S1_summary_P6.idx work/gul_S1_summaryaalcalc/P6.idx work/gul_S1_summaryleccalc/P6.idx > /dev/null & pid332=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid333=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryaalcalc/P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid334=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid335=$!
tee < fifo/gul_S1_summary_P8.idx work/gul_S1_summaryaalcalc/P8.idx work/gul_S1_summaryleccalc/P8.idx > /dev/null & pid336=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid337=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryaalcalc/P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid338=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid339=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryaalcalc/P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid340=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_eltcalc_P11 fifo/gul_S1_summarycalc_P11 fifo/gul_S1_pltcalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid341=$!
tee < fifo/gul_S1_summary_P11.idx work/gul_S1_summaryaalcalc/P11.idx work/gul_S1_summaryleccalc/P11.idx > /dev/null & pid342=$!
tee < fifo/gul_S1_summary_P12 fifo/gul_S1_eltcalc_P12 fifo/gul_S1_summarycalc_P12 fifo/gul_S1_pltcalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid343=$!
tee < fifo/gul_S1_summary_P12.idx work/gul_S1_summaryaalcalc/P12.idx work/gul_S1_summaryleccalc/P12.idx > /dev/null & pid344=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_eltcalc_P13 fifo/gul_S1_summarycalc_P13 fifo/gul_S1_pltcalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid345=$!
tee < fifo/gul_S1_summary_P13.idx work/gul_S1_summaryaalcalc/P13.idx work/gul_S1_summaryleccalc/P13.idx > /dev/null & pid346=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_eltcalc_P14 fifo/gul_S1_summarycalc_P14 fifo/gul_S1_pltcalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid347=$!
tee < fifo/gul_S1_summary_P14.idx work/gul_S1_summaryaalcalc/P14.idx work/gul_S1_summaryleccalc/P14.idx > /dev/null & pid348=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_eltcalc_P15 fifo/gul_S1_summarycalc_P15 fifo/gul_S1_pltcalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid349=$!
tee < fifo/gul_S1_summary_P15.idx work/gul_S1_summaryaalcalc/P15.idx work/gul_S1_summaryleccalc/P15.idx > /dev/null & pid350=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_eltcalc_P16 fifo/gul_S1_summarycalc_P16 fifo/gul_S1_pltcalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid351=$!
tee < fifo/gul_S1_summary_P16.idx work/gul_S1_summaryaalcalc/P16.idx work/gul_S1_summaryleccalc/P16.idx > /dev/null & pid352=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_eltcalc_P17 fifo/gul_S1_summarycalc_P17 fifo/gul_S1_pltcalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid353=$!
tee < fifo/gul_S1_summary_P17.idx work/gul_S1_summaryaalcalc/P17.idx work/gul_S1_summaryleccalc/P17.idx > /dev/null & pid354=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_eltcalc_P18 fifo/gul_S1_summarycalc_P18 fifo/gul_S1_pltcalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid355=$!
tee < fifo/gul_S1_summary_P18.idx work/gul_S1_summaryaalcalc/P18.idx work/gul_S1_summaryleccalc/P18.idx > /dev/null & pid356=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_eltcalc_P19 fifo/gul_S1_summarycalc_P19 fifo/gul_S1_pltcalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid357=$!
tee < fifo/gul_S1_summary_P19.idx work/gul_S1_summaryaalcalc/P19.idx work/gul_S1_summaryleccalc/P19.idx > /dev/null & pid358=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_eltcalc_P20 fifo/gul_S1_summarycalc_P20 fifo/gul_S1_pltcalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid359=$!
tee < fifo/gul_S1_summary_P20.idx work/gul_S1_summaryaalcalc/P20.idx work/gul_S1_summaryleccalc/P20.idx > /dev/null & pid360=$!
tee < fifo/gul_S1_summary_P21 fifo/gul_S1_eltcalc_P21 fifo/gul_S1_summarycalc_P21 fifo/gul_S1_pltcalc_P21 work/gul_S1_summaryaalcalc/P21.bin work/gul_S1_summaryleccalc/P21.bin > /dev/null & pid361=$!
tee < fifo/gul_S1_summary_P21.idx work/gul_S1_summaryaalcalc/P21.idx work/gul_S1_summaryleccalc/P21.idx > /dev/null & pid362=$!
tee < fifo/gul_S1_summary_P22 fifo/gul_S1_eltcalc_P22 fifo/gul_S1_summarycalc_P22 fifo/gul_S1_pltcalc_P22 work/gul_S1_summaryaalcalc/P22.bin work/gul_S1_summaryleccalc/P22.bin > /dev/null & pid363=$!
tee < fifo/gul_S1_summary_P22.idx work/gul_S1_summaryaalcalc/P22.idx work/gul_S1_summaryleccalc/P22.idx > /dev/null & pid364=$!
tee < fifo/gul_S1_summary_P23 fifo/gul_S1_eltcalc_P23 fifo/gul_S1_summarycalc_P23 fifo/gul_S1_pltcalc_P23 work/gul_S1_summaryaalcalc/P23.bin work/gul_S1_summaryleccalc/P23.bin > /dev/null & pid365=$!
tee < fifo/gul_S1_summary_P23.idx work/gul_S1_summaryaalcalc/P23.idx work/gul_S1_summaryleccalc/P23.idx > /dev/null & pid366=$!
tee < fifo/gul_S1_summary_P24 fifo/gul_S1_eltcalc_P24 fifo/gul_S1_summarycalc_P24 fifo/gul_S1_pltcalc_P24 work/gul_S1_summaryaalcalc/P24.bin work/gul_S1_summaryleccalc/P24.bin > /dev/null & pid367=$!
tee < fifo/gul_S1_summary_P24.idx work/gul_S1_summaryaalcalc/P24.idx work/gul_S1_summaryleccalc/P24.idx > /dev/null & pid368=$!
tee < fifo/gul_S1_summary_P25 fifo/gul_S1_eltcalc_P25 fifo/gul_S1_summarycalc_P25 fifo/gul_S1_pltcalc_P25 work/gul_S1_summaryaalcalc/P25.bin work/gul_S1_summaryleccalc/P25.bin > /dev/null & pid369=$!
tee < fifo/gul_S1_summary_P25.idx work/gul_S1_summaryaalcalc/P25.idx work/gul_S1_summaryleccalc/P25.idx > /dev/null & pid370=$!
tee < fifo/gul_S1_summary_P26 fifo/gul_S1_eltcalc_P26 fifo/gul_S1_summarycalc_P26 fifo/gul_S1_pltcalc_P26 work/gul_S1_summaryaalcalc/P26.bin work/gul_S1_summaryleccalc/P26.bin > /dev/null & pid371=$!
tee < fifo/gul_S1_summary_P26.idx work/gul_S1_summaryaalcalc/P26.idx work/gul_S1_summaryleccalc/P26.idx > /dev/null & pid372=$!
tee < fifo/gul_S1_summary_P27 fifo/gul_S1_eltcalc_P27 fifo/gul_S1_summarycalc_P27 fifo/gul_S1_pltcalc_P27 work/gul_S1_summaryaalcalc/P27.bin work/gul_S1_summaryleccalc/P27.bin > /dev/null & pid373=$!
tee < fifo/gul_S1_summary_P27.idx work/gul_S1_summaryaalcalc/P27.idx work/gul_S1_summaryleccalc/P27.idx > /dev/null & pid374=$!
tee < fifo/gul_S1_summary_P28 fifo/gul_S1_eltcalc_P28 fifo/gul_S1_summarycalc_P28 fifo/gul_S1_pltcalc_P28 work/gul_S1_summaryaalcalc/P28.bin work/gul_S1_summaryleccalc/P28.bin > /dev/null & pid375=$!
tee < fifo/gul_S1_summary_P28.idx work/gul_S1_summaryaalcalc/P28.idx work/gul_S1_summaryleccalc/P28.idx > /dev/null & pid376=$!
tee < fifo/gul_S1_summary_P29 fifo/gul_S1_eltcalc_P29 fifo/gul_S1_summarycalc_P29 fifo/gul_S1_pltcalc_P29 work/gul_S1_summaryaalcalc/P29.bin work/gul_S1_summaryleccalc/P29.bin > /dev/null & pid377=$!
tee < fifo/gul_S1_summary_P29.idx work/gul_S1_summaryaalcalc/P29.idx work/gul_S1_summaryleccalc/P29.idx > /dev/null & pid378=$!
tee < fifo/gul_S1_summary_P30 fifo/gul_S1_eltcalc_P30 fifo/gul_S1_summarycalc_P30 fifo/gul_S1_pltcalc_P30 work/gul_S1_summaryaalcalc/P30.bin work/gul_S1_summaryleccalc/P30.bin > /dev/null & pid379=$!
tee < fifo/gul_S1_summary_P30.idx work/gul_S1_summaryaalcalc/P30.idx work/gul_S1_summaryleccalc/P30.idx > /dev/null & pid380=$!
tee < fifo/gul_S1_summary_P31 fifo/gul_S1_eltcalc_P31 fifo/gul_S1_summarycalc_P31 fifo/gul_S1_pltcalc_P31 work/gul_S1_summaryaalcalc/P31.bin work/gul_S1_summaryleccalc/P31.bin > /dev/null & pid381=$!
tee < fifo/gul_S1_summary_P31.idx work/gul_S1_summaryaalcalc/P31.idx work/gul_S1_summaryleccalc/P31.idx > /dev/null & pid382=$!
tee < fifo/gul_S1_summary_P32 fifo/gul_S1_eltcalc_P32 fifo/gul_S1_summarycalc_P32 fifo/gul_S1_pltcalc_P32 work/gul_S1_summaryaalcalc/P32.bin work/gul_S1_summaryleccalc/P32.bin > /dev/null & pid383=$!
tee < fifo/gul_S1_summary_P32.idx work/gul_S1_summaryaalcalc/P32.idx work/gul_S1_summaryleccalc/P32.idx > /dev/null & pid384=$!
tee < fifo/gul_S1_summary_P33 fifo/gul_S1_eltcalc_P33 fifo/gul_S1_summarycalc_P33 fifo/gul_S1_pltcalc_P33 work/gul_S1_summaryaalcalc/P33.bin work/gul_S1_summaryleccalc/P33.bin > /dev/null & pid385=$!
tee < fifo/gul_S1_summary_P33.idx work/gul_S1_summaryaalcalc/P33.idx work/gul_S1_summaryleccalc/P33.idx > /dev/null & pid386=$!
tee < fifo/gul_S1_summary_P34 fifo/gul_S1_eltcalc_P34 fifo/gul_S1_summarycalc_P34 fifo/gul_S1_pltcalc_P34 work/gul_S1_summaryaalcalc/P34.bin work/gul_S1_summaryleccalc/P34.bin > /dev/null & pid387=$!
tee < fifo/gul_S1_summary_P34.idx work/gul_S1_summaryaalcalc/P34.idx work/gul_S1_summaryleccalc/P34.idx > /dev/null & pid388=$!
tee < fifo/gul_S1_summary_P35 fifo/gul_S1_eltcalc_P35 fifo/gul_S1_summarycalc_P35 fifo/gul_S1_pltcalc_P35 work/gul_S1_summaryaalcalc/P35.bin work/gul_S1_summaryleccalc/P35.bin > /dev/null & pid389=$!
tee < fifo/gul_S1_summary_P35.idx work/gul_S1_summaryaalcalc/P35.idx work/gul_S1_summaryleccalc/P35.idx > /dev/null & pid390=$!
tee < fifo/gul_S1_summary_P36 fifo/gul_S1_eltcalc_P36 fifo/gul_S1_summarycalc_P36 fifo/gul_S1_pltcalc_P36 work/gul_S1_summaryaalcalc/P36.bin work/gul_S1_summaryleccalc/P36.bin > /dev/null & pid391=$!
tee < fifo/gul_S1_summary_P36.idx work/gul_S1_summaryaalcalc/P36.idx work/gul_S1_summaryleccalc/P36.idx > /dev/null & pid392=$!
tee < fifo/gul_S1_summary_P37 fifo/gul_S1_eltcalc_P37 fifo/gul_S1_summarycalc_P37 fifo/gul_S1_pltcalc_P37 work/gul_S1_summaryaalcalc/P37.bin work/gul_S1_summaryleccalc/P37.bin > /dev/null & pid393=$!
tee < fifo/gul_S1_summary_P37.idx work/gul_S1_summaryaalcalc/P37.idx work/gul_S1_summaryleccalc/P37.idx > /dev/null & pid394=$!
tee < fifo/gul_S1_summary_P38 fifo/gul_S1_eltcalc_P38 fifo/gul_S1_summarycalc_P38 fifo/gul_S1_pltcalc_P38 work/gul_S1_summaryaalcalc/P38.bin work/gul_S1_summaryleccalc/P38.bin > /dev/null & pid395=$!
tee < fifo/gul_S1_summary_P38.idx work/gul_S1_summaryaalcalc/P38.idx work/gul_S1_summaryleccalc/P38.idx > /dev/null & pid396=$!
tee < fifo/gul_S1_summary_P39 fifo/gul_S1_eltcalc_P39 fifo/gul_S1_summarycalc_P39 fifo/gul_S1_pltcalc_P39 work/gul_S1_summaryaalcalc/P39.bin work/gul_S1_summaryleccalc/P39.bin > /dev/null & pid397=$!
tee < fifo/gul_S1_summary_P39.idx work/gul_S1_summaryaalcalc/P39.idx work/gul_S1_summaryleccalc/P39.idx > /dev/null & pid398=$!
tee < fifo/gul_S1_summary_P40 fifo/gul_S1_eltcalc_P40 fifo/gul_S1_summarycalc_P40 fifo/gul_S1_pltcalc_P40 work/gul_S1_summaryaalcalc/P40.bin work/gul_S1_summaryleccalc/P40.bin > /dev/null & pid399=$!
tee < fifo/gul_S1_summary_P40.idx work/gul_S1_summaryaalcalc/P40.idx work/gul_S1_summaryleccalc/P40.idx > /dev/null & pid400=$!

( summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P21 < fifo/gul_P21 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P22 < fifo/gul_P22 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P23 < fifo/gul_P23 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P24 < fifo/gul_P24 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P25 < fifo/gul_P25 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P26 < fifo/gul_P26 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P27 < fifo/gul_P27 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P28 < fifo/gul_P28 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P29 < fifo/gul_P29 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P30 < fifo/gul_P30 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P31 < fifo/gul_P31 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P32 < fifo/gul_P32 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P33 < fifo/gul_P33 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P34 < fifo/gul_P34 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P35 < fifo/gul_P35 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P36 < fifo/gul_P36 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P37 < fifo/gul_P37 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P38 < fifo/gul_P38 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P39 < fifo/gul_P39 ) 2>> $LOG_DIR/stderror.err  &
( summarycalc -m -i  -1 fifo/gul_S1_summary_P40 < fifo/gul_P40 ) 2>> $LOG_DIR/stderror.err  &

( ( eve 1 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  ) 2>> $LOG_DIR/stderror.err ) & pid401=$!
( ( eve 2 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  ) 2>> $LOG_DIR/stderror.err ) & pid402=$!
( ( eve 3 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P3 | fmcalc -a2 > fifo/il_P3  ) 2>> $LOG_DIR/stderror.err ) & pid403=$!
( ( eve 4 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  ) 2>> $LOG_DIR/stderror.err ) & pid404=$!
( ( eve 5 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  ) 2>> $LOG_DIR/stderror.err ) & pid405=$!
( ( eve 6 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P6 | fmcalc -a2 > fifo/il_P6  ) 2>> $LOG_DIR/stderror.err ) & pid406=$!
( ( eve 7 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P7 | fmcalc -a2 > fifo/il_P7  ) 2>> $LOG_DIR/stderror.err ) & pid407=$!
( ( eve 8 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P8 | fmcalc -a2 > fifo/il_P8  ) 2>> $LOG_DIR/stderror.err ) & pid408=$!
( ( eve 9 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  ) 2>> $LOG_DIR/stderror.err ) & pid409=$!
( ( eve 10 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  ) 2>> $LOG_DIR/stderror.err ) & pid410=$!
( ( eve 11 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P11 | fmcalc -a2 > fifo/il_P11  ) 2>> $LOG_DIR/stderror.err ) & pid411=$!
( ( eve 12 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P12 | fmcalc -a2 > fifo/il_P12  ) 2>> $LOG_DIR/stderror.err ) & pid412=$!
( ( eve 13 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P13 | fmcalc -a2 > fifo/il_P13  ) 2>> $LOG_DIR/stderror.err ) & pid413=$!
( ( eve 14 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P14 | fmcalc -a2 > fifo/il_P14  ) 2>> $LOG_DIR/stderror.err ) & pid414=$!
( ( eve 15 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P15 | fmcalc -a2 > fifo/il_P15  ) 2>> $LOG_DIR/stderror.err ) & pid415=$!
( ( eve 16 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P16 | fmcalc -a2 > fifo/il_P16  ) 2>> $LOG_DIR/stderror.err ) & pid416=$!
( ( eve 17 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P17 | fmcalc -a2 > fifo/il_P17  ) 2>> $LOG_DIR/stderror.err ) & pid417=$!
( ( eve 18 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P18 | fmcalc -a2 > fifo/il_P18  ) 2>> $LOG_DIR/stderror.err ) & pid418=$!
( ( eve 19 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P19 | fmcalc -a2 > fifo/il_P19  ) 2>> $LOG_DIR/stderror.err ) & pid419=$!
( ( eve 20 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P20 | fmcalc -a2 > fifo/il_P20  ) 2>> $LOG_DIR/stderror.err ) & pid420=$!
( ( eve 21 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P21 | fmcalc -a2 > fifo/il_P21  ) 2>> $LOG_DIR/stderror.err ) & pid421=$!
( ( eve 22 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P22 | fmcalc -a2 > fifo/il_P22  ) 2>> $LOG_DIR/stderror.err ) & pid422=$!
( ( eve 23 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P23 | fmcalc -a2 > fifo/il_P23  ) 2>> $LOG_DIR/stderror.err ) & pid423=$!
( ( eve 24 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P24 | fmcalc -a2 > fifo/il_P24  ) 2>> $LOG_DIR/stderror.err ) & pid424=$!
( ( eve 25 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P25 | fmcalc -a2 > fifo/il_P25  ) 2>> $LOG_DIR/stderror.err ) & pid425=$!
( ( eve 26 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P26 | fmcalc -a2 > fifo/il_P26  ) 2>> $LOG_DIR/stderror.err ) & pid426=$!
( ( eve 27 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P27 | fmcalc -a2 > fifo/il_P27  ) 2>> $LOG_DIR/stderror.err ) & pid427=$!
( ( eve 28 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P28 | fmcalc -a2 > fifo/il_P28  ) 2>> $LOG_DIR/stderror.err ) & pid428=$!
( ( eve 29 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P29 | fmcalc -a2 > fifo/il_P29  ) 2>> $LOG_DIR/stderror.err ) & pid429=$!
( ( eve 30 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P30 | fmcalc -a2 > fifo/il_P30  ) 2>> $LOG_DIR/stderror.err ) & pid430=$!
( ( eve 31 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P31 | fmcalc -a2 > fifo/il_P31  ) 2>> $LOG_DIR/stderror.err ) & pid431=$!
( ( eve 32 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P32 | fmcalc -a2 > fifo/il_P32  ) 2>> $LOG_DIR/stderror.err ) & pid432=$!
( ( eve 33 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P33 | fmcalc -a2 > fifo/il_P33  ) 2>> $LOG_DIR/stderror.err ) & pid433=$!
( ( eve 34 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P34 | fmcalc -a2 > fifo/il_P34  ) 2>> $LOG_DIR/stderror.err ) & pid434=$!
( ( eve 35 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P35 | fmcalc -a2 > fifo/il_P35  ) 2>> $LOG_DIR/stderror.err ) & pid435=$!
( ( eve 36 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P36 | fmcalc -a2 > fifo/il_P36  ) 2>> $LOG_DIR/stderror.err ) & pid436=$!
( ( eve 37 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P37 | fmcalc -a2 > fifo/il_P37  ) 2>> $LOG_DIR/stderror.err ) & pid437=$!
( ( eve 38 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P38 | fmcalc -a2 > fifo/il_P38  ) 2>> $LOG_DIR/stderror.err ) & pid438=$!
( ( eve 39 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P39 | fmcalc -a2 > fifo/il_P39  ) 2>> $LOG_DIR/stderror.err ) & pid439=$!
( ( eve 40 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P40 | fmcalc -a2 > fifo/il_P40  ) 2>> $LOG_DIR/stderror.err ) & pid440=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210 $pid211 $pid212 $pid213 $pid214 $pid215 $pid216 $pid217 $pid218 $pid219 $pid220 $pid221 $pid222 $pid223 $pid224 $pid225 $pid226 $pid227 $pid228 $pid229 $pid230 $pid231 $pid232 $pid233 $pid234 $pid235 $pid236 $pid237 $pid238 $pid239 $pid240 $pid241 $pid242 $pid243 $pid244 $pid245 $pid246 $pid247 $pid248 $pid249 $pid250 $pid251 $pid252 $pid253 $pid254 $pid255 $pid256 $pid257 $pid258 $pid259 $pid260 $pid261 $pid262 $pid263 $pid264 $pid265 $pid266 $pid267 $pid268 $pid269 $pid270 $pid271 $pid272 $pid273 $pid274 $pid275 $pid276 $pid277 $pid278 $pid279 $pid280 $pid281 $pid282 $pid283 $pid284 $pid285 $pid286 $pid287 $pid288 $pid289 $pid290 $pid291 $pid292 $pid293 $pid294 $pid295 $pid296 $pid297 $pid298 $pid299 $pid300 $pid301 $pid302 $pid303 $pid304 $pid305 $pid306 $pid307 $pid308 $pid309 $pid310 $pid311 $pid312 $pid313 $pid314 $pid315 $pid316 $pid317 $pid318 $pid319 $pid320 $pid321 $pid322 $pid323 $pid324 $pid325 $pid326 $pid327 $pid328 $pid329 $pid330 $pid331 $pid332 $pid333 $pid334 $pid335 $pid336 $pid337 $pid338 $pid339 $pid340 $pid341 $pid342 $pid343 $pid344 $pid345 $pid346 $pid347 $pid348 $pid349 $pid350 $pid351 $pid352 $pid353 $pid354 $pid355 $pid356 $pid357 $pid358 $pid359 $pid360 $pid361 $pid362 $pid363 $pid364 $pid365 $pid366 $pid367 $pid368 $pid369 $pid370 $pid371 $pid372 $pid373 $pid374 $pid375 $pid376 $pid377 $pid378 $pid379 $pid380 $pid381 $pid382 $pid383 $pid384 $pid385 $pid386 $pid387 $pid388 $pid389 $pid390 $pid391 $pid392 $pid393 $pid394 $pid395 $pid396 $pid397 $pid398 $pid399 $pid400 $pid401 $pid402 $pid403 $pid404 $pid405 $pid406 $pid407 $pid408 $pid409 $pid410 $pid411 $pid412 $pid413 $pid414 $pid415 $pid416 $pid417 $pid418 $pid419 $pid420 $pid421 $pid422 $pid423 $pid424 $pid425 $pid426 $pid427 $pid428 $pid429 $pid430 $pid431 $pid432 $pid433 $pid434 $pid435 $pid436 $pid437 $pid438 $pid439 $pid440


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 work/kat/il_S1_eltcalc_P21 work/kat/il_S1_eltcalc_P22 work/kat/il_S1_eltcalc_P23 work/kat/il_S1_eltcalc_P24 work/kat/il_S1_eltcalc_P25 work/kat/il_S1_eltcalc_P26 work/kat/il_S1_eltcalc_P27 work/kat/il_S1_eltcalc_P28 work/kat/il_S1_eltcalc_P29 work/kat/il_S1_eltcalc_P30 work/kat/il_S1_eltcalc_P31 work/kat/il_S1_eltcalc_P32 work/kat/il_S1_eltcalc_P33 work/kat/il_S1_eltcalc_P34 work/kat/il_S1_eltcalc_P35 work/kat/il_S1_eltcalc_P36 work/kat/il_S1_eltcalc_P37 work/kat/il_S1_eltcalc_P38 work/kat/il_S1_eltcalc_P39 work/kat/il_S1_eltcalc_P40 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 work/kat/il_S1_pltcalc_P21 work/kat/il_S1_pltcalc_P22 work/kat/il_S1_pltcalc_P23 work/kat/il_S1_pltcalc_P24 work/kat/il_S1_pltcalc_P25 work/kat/il_S1_pltcalc_P26 work/kat/il_S1_pltcalc_P27 work/kat/il_S1_pltcalc_P28 work/kat/il_S1_pltcalc_P29 work/kat/il_S1_pltcalc_P30 work/kat/il_S1_pltcalc_P31 work/kat/il_S1_pltcalc_P32 work/kat/il_S1_pltcalc_P33 work/kat/il_S1_pltcalc_P34 work/kat/il_S1_pltcalc_P35 work/kat/il_S1_pltcalc_P36 work/kat/il_S1_pltcalc_P37 work/kat/il_S1_pltcalc_P38 work/kat/il_S1_pltcalc_P39 work/kat/il_S1_pltcalc_P40 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 work/kat/il_S1_summarycalc_P21 work/kat/il_S1_summarycalc_P22 work/kat/il_S1_summarycalc_P23 work/kat/il_S1_summarycalc_P24 work/kat/il_S1_summarycalc_P25 work/kat/il_S1_summarycalc_P26 work/kat/il_S1_summarycalc_P27 work/kat/il_S1_summarycalc_P28 work/kat/il_S1_summarycalc_P29 work/kat/il_S1_summarycalc_P30 work/kat/il_S1_summarycalc_P31 work/kat/il_S1_summarycalc_P32 work/kat/il_S1_summarycalc_P33 work/kat/il_S1_summarycalc_P34 work/kat/il_S1_summarycalc_P35 work/kat/il_S1_summarycalc_P36 work/kat/il_S1_summarycalc_P37 work/kat/il_S1_summarycalc_P38 work/kat/il_S1_summarycalc_P39 work/kat/il_S1_summarycalc_P40 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 work/kat/gul_S1_eltcalc_P21 work/kat/gul_S1_eltcalc_P22 work/kat/gul_S1_eltcalc_P23 work/kat/gul_S1_eltcalc_P24 work/kat/gul_S1_eltcalc_P25 work/kat/gul_S1_eltcalc_P26 work/kat/gul_S1_eltcalc_P27 work/kat/gul_S1_eltcalc_P28 work/kat/gul_S1_eltcalc_P29 work/kat/gul_S1_eltcalc_P30 work/kat/gul_S1_eltcalc_P31 work/kat/gul_S1_eltcalc_P32 work/kat/gul_S1_eltcalc_P33 work/kat/gul_S1_eltcalc_P34 work/kat/gul_S1_eltcalc_P35 work/kat/gul_S1_eltcalc_P36 work/kat/gul_S1_eltcalc_P37 work/kat/gul_S1_eltcalc_P38 work/kat/gul_S1_eltcalc_P39 work/kat/gul_S1_eltcalc_P40 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 work/kat/gul_S1_pltcalc_P21 work/kat/gul_S1_pltcalc_P22 work/kat/gul_S1_pltcalc_P23 work/kat/gul_S1_pltcalc_P24 work/kat/gul_S1_pltcalc_P25 work/kat/gul_S1_pltcalc_P26 work/kat/gul_S1_pltcalc_P27 work/kat/gul_S1_pltcalc_P28 work/kat/gul_S1_pltcalc_P29 work/kat/gul_S1_pltcalc_P30 work/kat/gul_S1_pltcalc_P31 work/kat/gul_S1_pltcalc_P32 work/kat/gul_S1_pltcalc_P33 work/kat/gul_S1_pltcalc_P34 work/kat/gul_S1_pltcalc_P35 work/kat/gul_S1_pltcalc_P36 work/kat/gul_S1_pltcalc_P37 work/kat/gul_S1_pltcalc_P38 work/kat/gul_S1_pltcalc_P39 work/kat/gul_S1_pltcalc_P40 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 work/kat/gul_S1_summarycalc_P21 work/kat/gul_S1_summarycalc_P22 work/kat/gul_S1_summarycalc_P23 work/kat/gul_S1_summarycalc_P24 work/kat/gul_S1_summarycalc_P25 work/kat/gul_S1_summarycalc_P26 work/kat/gul_S1_summarycalc_P27 work/kat/gul_S1_summarycalc_P28 work/kat/gul_S1_summarycalc_P29 work/kat/gul_S1_summarycalc_P30 work/kat/gul_S1_summarycalc_P31 work/kat/gul_S1_summarycalc_P32 work/kat/gul_S1_summarycalc_P33 work/kat/gul_S1_summarycalc_P34 work/kat/gul_S1_summarycalc_P35 work/kat/gul_S1_summarycalc_P36 work/kat/gul_S1_summarycalc_P37 work/kat/gul_S1_summarycalc_P38 work/kat/gul_S1_summarycalc_P39 work/kat/gul_S1_summarycalc_P40 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid1=$!
( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid2=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid3=$!
( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*

check_complete
