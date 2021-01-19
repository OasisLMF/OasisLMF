#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

exit_handler(){
   exit_code=$?
   kill -9 $pid0 2> /dev/null
   if [ "$exit_code" -gt 0 ]; then
       echo 'Ktools Run Error - exitcode='$exit_code
   else
       echo 'Run Completed'
   fi

   set +x
   group_pid=$(ps -p $$ -o pgid --no-headers)
   sess_pid=$(ps -p $$ -o sess --no-headers)
   script_pid=$$
   printf "Script PID:%d, GPID:%s, SPID:%d
" $script_pid $group_pid $sess_pid >> log/killout.txt

   ps f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk -F: '$1>$script_pid' | grep -v celery | grep -v python | grep -v $group_pid | grep -v run_ktools)
   echo "$PIDS_KILL" >> log/killout.txt
   kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   exit $exit_code
}
trap exit_handler QUIT HUP INT KILL TERM ERR

check_complete(){
    set +e
    proc_list="eve getmodel gulcalc fmcalc summarycalc eltcalc aalcalc leccalc pltcalc"
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
    fi
}
# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P1
mkfifo fifo/full_correlation/gul_fc_P2
mkfifo fifo/full_correlation/gul_fc_P3
mkfifo fifo/full_correlation/gul_fc_P4
mkfifo fifo/full_correlation/gul_fc_P5
mkfifo fifo/full_correlation/gul_fc_P6
mkfifo fifo/full_correlation/gul_fc_P7
mkfifo fifo/full_correlation/gul_fc_P8
mkfifo fifo/full_correlation/gul_fc_P9
mkfifo fifo/full_correlation/gul_fc_P10
mkfifo fifo/full_correlation/gul_fc_P11
mkfifo fifo/full_correlation/gul_fc_P12
mkfifo fifo/full_correlation/gul_fc_P13
mkfifo fifo/full_correlation/gul_fc_P14
mkfifo fifo/full_correlation/gul_fc_P15
mkfifo fifo/full_correlation/gul_fc_P16
mkfifo fifo/full_correlation/gul_fc_P17
mkfifo fifo/full_correlation/gul_fc_P18
mkfifo fifo/full_correlation/gul_fc_P19
mkfifo fifo/full_correlation/gul_fc_P20
mkfifo fifo/full_correlation/gul_fc_P21
mkfifo fifo/full_correlation/gul_fc_P22
mkfifo fifo/full_correlation/gul_fc_P23
mkfifo fifo/full_correlation/gul_fc_P24
mkfifo fifo/full_correlation/gul_fc_P25
mkfifo fifo/full_correlation/gul_fc_P26
mkfifo fifo/full_correlation/gul_fc_P27
mkfifo fifo/full_correlation/gul_fc_P28
mkfifo fifo/full_correlation/gul_fc_P29
mkfifo fifo/full_correlation/gul_fc_P30
mkfifo fifo/full_correlation/gul_fc_P31
mkfifo fifo/full_correlation/gul_fc_P32
mkfifo fifo/full_correlation/gul_fc_P33
mkfifo fifo/full_correlation/gul_fc_P34
mkfifo fifo/full_correlation/gul_fc_P35
mkfifo fifo/full_correlation/gul_fc_P36
mkfifo fifo/full_correlation/gul_fc_P37
mkfifo fifo/full_correlation/gul_fc_P38
mkfifo fifo/full_correlation/gul_fc_P39
mkfifo fifo/full_correlation/gul_fc_P40

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
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_pltcalc_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_pltcalc_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_pltcalc_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_eltcalc_P6
mkfifo fifo/gul_S1_summarycalc_P6
mkfifo fifo/gul_S1_pltcalc_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_eltcalc_P7
mkfifo fifo/gul_S1_summarycalc_P7
mkfifo fifo/gul_S1_pltcalc_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_eltcalc_P8
mkfifo fifo/gul_S1_summarycalc_P8
mkfifo fifo/gul_S1_pltcalc_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_eltcalc_P10
mkfifo fifo/gul_S1_summarycalc_P10
mkfifo fifo/gul_S1_pltcalc_P10

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_eltcalc_P11
mkfifo fifo/gul_S1_summarycalc_P11
mkfifo fifo/gul_S1_pltcalc_P11

mkfifo fifo/gul_S1_summary_P12
mkfifo fifo/gul_S1_eltcalc_P12
mkfifo fifo/gul_S1_summarycalc_P12
mkfifo fifo/gul_S1_pltcalc_P12

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_eltcalc_P13
mkfifo fifo/gul_S1_summarycalc_P13
mkfifo fifo/gul_S1_pltcalc_P13

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_eltcalc_P14
mkfifo fifo/gul_S1_summarycalc_P14
mkfifo fifo/gul_S1_pltcalc_P14

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_eltcalc_P15
mkfifo fifo/gul_S1_summarycalc_P15
mkfifo fifo/gul_S1_pltcalc_P15

mkfifo fifo/gul_S1_summary_P16
mkfifo fifo/gul_S1_eltcalc_P16
mkfifo fifo/gul_S1_summarycalc_P16
mkfifo fifo/gul_S1_pltcalc_P16

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_eltcalc_P17
mkfifo fifo/gul_S1_summarycalc_P17
mkfifo fifo/gul_S1_pltcalc_P17

mkfifo fifo/gul_S1_summary_P18
mkfifo fifo/gul_S1_eltcalc_P18
mkfifo fifo/gul_S1_summarycalc_P18
mkfifo fifo/gul_S1_pltcalc_P18

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_eltcalc_P19
mkfifo fifo/gul_S1_summarycalc_P19
mkfifo fifo/gul_S1_pltcalc_P19

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_eltcalc_P20
mkfifo fifo/gul_S1_summarycalc_P20
mkfifo fifo/gul_S1_pltcalc_P20

mkfifo fifo/gul_S1_summary_P21
mkfifo fifo/gul_S1_eltcalc_P21
mkfifo fifo/gul_S1_summarycalc_P21
mkfifo fifo/gul_S1_pltcalc_P21

mkfifo fifo/gul_S1_summary_P22
mkfifo fifo/gul_S1_eltcalc_P22
mkfifo fifo/gul_S1_summarycalc_P22
mkfifo fifo/gul_S1_pltcalc_P22

mkfifo fifo/gul_S1_summary_P23
mkfifo fifo/gul_S1_eltcalc_P23
mkfifo fifo/gul_S1_summarycalc_P23
mkfifo fifo/gul_S1_pltcalc_P23

mkfifo fifo/gul_S1_summary_P24
mkfifo fifo/gul_S1_eltcalc_P24
mkfifo fifo/gul_S1_summarycalc_P24
mkfifo fifo/gul_S1_pltcalc_P24

mkfifo fifo/gul_S1_summary_P25
mkfifo fifo/gul_S1_eltcalc_P25
mkfifo fifo/gul_S1_summarycalc_P25
mkfifo fifo/gul_S1_pltcalc_P25

mkfifo fifo/gul_S1_summary_P26
mkfifo fifo/gul_S1_eltcalc_P26
mkfifo fifo/gul_S1_summarycalc_P26
mkfifo fifo/gul_S1_pltcalc_P26

mkfifo fifo/gul_S1_summary_P27
mkfifo fifo/gul_S1_eltcalc_P27
mkfifo fifo/gul_S1_summarycalc_P27
mkfifo fifo/gul_S1_pltcalc_P27

mkfifo fifo/gul_S1_summary_P28
mkfifo fifo/gul_S1_eltcalc_P28
mkfifo fifo/gul_S1_summarycalc_P28
mkfifo fifo/gul_S1_pltcalc_P28

mkfifo fifo/gul_S1_summary_P29
mkfifo fifo/gul_S1_eltcalc_P29
mkfifo fifo/gul_S1_summarycalc_P29
mkfifo fifo/gul_S1_pltcalc_P29

mkfifo fifo/gul_S1_summary_P30
mkfifo fifo/gul_S1_eltcalc_P30
mkfifo fifo/gul_S1_summarycalc_P30
mkfifo fifo/gul_S1_pltcalc_P30

mkfifo fifo/gul_S1_summary_P31
mkfifo fifo/gul_S1_eltcalc_P31
mkfifo fifo/gul_S1_summarycalc_P31
mkfifo fifo/gul_S1_pltcalc_P31

mkfifo fifo/gul_S1_summary_P32
mkfifo fifo/gul_S1_eltcalc_P32
mkfifo fifo/gul_S1_summarycalc_P32
mkfifo fifo/gul_S1_pltcalc_P32

mkfifo fifo/gul_S1_summary_P33
mkfifo fifo/gul_S1_eltcalc_P33
mkfifo fifo/gul_S1_summarycalc_P33
mkfifo fifo/gul_S1_pltcalc_P33

mkfifo fifo/gul_S1_summary_P34
mkfifo fifo/gul_S1_eltcalc_P34
mkfifo fifo/gul_S1_summarycalc_P34
mkfifo fifo/gul_S1_pltcalc_P34

mkfifo fifo/gul_S1_summary_P35
mkfifo fifo/gul_S1_eltcalc_P35
mkfifo fifo/gul_S1_summarycalc_P35
mkfifo fifo/gul_S1_pltcalc_P35

mkfifo fifo/gul_S1_summary_P36
mkfifo fifo/gul_S1_eltcalc_P36
mkfifo fifo/gul_S1_summarycalc_P36
mkfifo fifo/gul_S1_pltcalc_P36

mkfifo fifo/gul_S1_summary_P37
mkfifo fifo/gul_S1_eltcalc_P37
mkfifo fifo/gul_S1_summarycalc_P37
mkfifo fifo/gul_S1_pltcalc_P37

mkfifo fifo/gul_S1_summary_P38
mkfifo fifo/gul_S1_eltcalc_P38
mkfifo fifo/gul_S1_summarycalc_P38
mkfifo fifo/gul_S1_pltcalc_P38

mkfifo fifo/gul_S1_summary_P39
mkfifo fifo/gul_S1_eltcalc_P39
mkfifo fifo/gul_S1_summarycalc_P39
mkfifo fifo/gul_S1_pltcalc_P39

mkfifo fifo/gul_S1_summary_P40
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
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_pltcalc_P3

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_pltcalc_P4

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_pltcalc_P5

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_eltcalc_P6
mkfifo fifo/il_S1_summarycalc_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_eltcalc_P7
mkfifo fifo/il_S1_summarycalc_P7
mkfifo fifo/il_S1_pltcalc_P7

mkfifo fifo/il_S1_summary_P8
mkfifo fifo/il_S1_eltcalc_P8
mkfifo fifo/il_S1_summarycalc_P8
mkfifo fifo/il_S1_pltcalc_P8

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9

mkfifo fifo/il_S1_summary_P10
mkfifo fifo/il_S1_eltcalc_P10
mkfifo fifo/il_S1_summarycalc_P10
mkfifo fifo/il_S1_pltcalc_P10

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_eltcalc_P11
mkfifo fifo/il_S1_summarycalc_P11
mkfifo fifo/il_S1_pltcalc_P11

mkfifo fifo/il_S1_summary_P12
mkfifo fifo/il_S1_eltcalc_P12
mkfifo fifo/il_S1_summarycalc_P12
mkfifo fifo/il_S1_pltcalc_P12

mkfifo fifo/il_S1_summary_P13
mkfifo fifo/il_S1_eltcalc_P13
mkfifo fifo/il_S1_summarycalc_P13
mkfifo fifo/il_S1_pltcalc_P13

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_eltcalc_P14
mkfifo fifo/il_S1_summarycalc_P14
mkfifo fifo/il_S1_pltcalc_P14

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_eltcalc_P15
mkfifo fifo/il_S1_summarycalc_P15
mkfifo fifo/il_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_eltcalc_P16
mkfifo fifo/il_S1_summarycalc_P16
mkfifo fifo/il_S1_pltcalc_P16

mkfifo fifo/il_S1_summary_P17
mkfifo fifo/il_S1_eltcalc_P17
mkfifo fifo/il_S1_summarycalc_P17
mkfifo fifo/il_S1_pltcalc_P17

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_eltcalc_P18
mkfifo fifo/il_S1_summarycalc_P18
mkfifo fifo/il_S1_pltcalc_P18

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_eltcalc_P19
mkfifo fifo/il_S1_summarycalc_P19
mkfifo fifo/il_S1_pltcalc_P19

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_eltcalc_P20
mkfifo fifo/il_S1_summarycalc_P20
mkfifo fifo/il_S1_pltcalc_P20

mkfifo fifo/il_S1_summary_P21
mkfifo fifo/il_S1_eltcalc_P21
mkfifo fifo/il_S1_summarycalc_P21
mkfifo fifo/il_S1_pltcalc_P21

mkfifo fifo/il_S1_summary_P22
mkfifo fifo/il_S1_eltcalc_P22
mkfifo fifo/il_S1_summarycalc_P22
mkfifo fifo/il_S1_pltcalc_P22

mkfifo fifo/il_S1_summary_P23
mkfifo fifo/il_S1_eltcalc_P23
mkfifo fifo/il_S1_summarycalc_P23
mkfifo fifo/il_S1_pltcalc_P23

mkfifo fifo/il_S1_summary_P24
mkfifo fifo/il_S1_eltcalc_P24
mkfifo fifo/il_S1_summarycalc_P24
mkfifo fifo/il_S1_pltcalc_P24

mkfifo fifo/il_S1_summary_P25
mkfifo fifo/il_S1_eltcalc_P25
mkfifo fifo/il_S1_summarycalc_P25
mkfifo fifo/il_S1_pltcalc_P25

mkfifo fifo/il_S1_summary_P26
mkfifo fifo/il_S1_eltcalc_P26
mkfifo fifo/il_S1_summarycalc_P26
mkfifo fifo/il_S1_pltcalc_P26

mkfifo fifo/il_S1_summary_P27
mkfifo fifo/il_S1_eltcalc_P27
mkfifo fifo/il_S1_summarycalc_P27
mkfifo fifo/il_S1_pltcalc_P27

mkfifo fifo/il_S1_summary_P28
mkfifo fifo/il_S1_eltcalc_P28
mkfifo fifo/il_S1_summarycalc_P28
mkfifo fifo/il_S1_pltcalc_P28

mkfifo fifo/il_S1_summary_P29
mkfifo fifo/il_S1_eltcalc_P29
mkfifo fifo/il_S1_summarycalc_P29
mkfifo fifo/il_S1_pltcalc_P29

mkfifo fifo/il_S1_summary_P30
mkfifo fifo/il_S1_eltcalc_P30
mkfifo fifo/il_S1_summarycalc_P30
mkfifo fifo/il_S1_pltcalc_P30

mkfifo fifo/il_S1_summary_P31
mkfifo fifo/il_S1_eltcalc_P31
mkfifo fifo/il_S1_summarycalc_P31
mkfifo fifo/il_S1_pltcalc_P31

mkfifo fifo/il_S1_summary_P32
mkfifo fifo/il_S1_eltcalc_P32
mkfifo fifo/il_S1_summarycalc_P32
mkfifo fifo/il_S1_pltcalc_P32

mkfifo fifo/il_S1_summary_P33
mkfifo fifo/il_S1_eltcalc_P33
mkfifo fifo/il_S1_summarycalc_P33
mkfifo fifo/il_S1_pltcalc_P33

mkfifo fifo/il_S1_summary_P34
mkfifo fifo/il_S1_eltcalc_P34
mkfifo fifo/il_S1_summarycalc_P34
mkfifo fifo/il_S1_pltcalc_P34

mkfifo fifo/il_S1_summary_P35
mkfifo fifo/il_S1_eltcalc_P35
mkfifo fifo/il_S1_summarycalc_P35
mkfifo fifo/il_S1_pltcalc_P35

mkfifo fifo/il_S1_summary_P36
mkfifo fifo/il_S1_eltcalc_P36
mkfifo fifo/il_S1_summarycalc_P36
mkfifo fifo/il_S1_pltcalc_P36

mkfifo fifo/il_S1_summary_P37
mkfifo fifo/il_S1_eltcalc_P37
mkfifo fifo/il_S1_summarycalc_P37
mkfifo fifo/il_S1_pltcalc_P37

mkfifo fifo/il_S1_summary_P38
mkfifo fifo/il_S1_eltcalc_P38
mkfifo fifo/il_S1_summarycalc_P38
mkfifo fifo/il_S1_pltcalc_P38

mkfifo fifo/il_S1_summary_P39
mkfifo fifo/il_S1_eltcalc_P39
mkfifo fifo/il_S1_summarycalc_P39
mkfifo fifo/il_S1_pltcalc_P39

mkfifo fifo/il_S1_summary_P40
mkfifo fifo/il_S1_eltcalc_P40
mkfifo fifo/il_S1_summarycalc_P40
mkfifo fifo/il_S1_pltcalc_P40

mkfifo fifo/full_correlation/gul_P1
mkfifo fifo/full_correlation/gul_P2
mkfifo fifo/full_correlation/gul_P3
mkfifo fifo/full_correlation/gul_P4
mkfifo fifo/full_correlation/gul_P5
mkfifo fifo/full_correlation/gul_P6
mkfifo fifo/full_correlation/gul_P7
mkfifo fifo/full_correlation/gul_P8
mkfifo fifo/full_correlation/gul_P9
mkfifo fifo/full_correlation/gul_P10
mkfifo fifo/full_correlation/gul_P11
mkfifo fifo/full_correlation/gul_P12
mkfifo fifo/full_correlation/gul_P13
mkfifo fifo/full_correlation/gul_P14
mkfifo fifo/full_correlation/gul_P15
mkfifo fifo/full_correlation/gul_P16
mkfifo fifo/full_correlation/gul_P17
mkfifo fifo/full_correlation/gul_P18
mkfifo fifo/full_correlation/gul_P19
mkfifo fifo/full_correlation/gul_P20
mkfifo fifo/full_correlation/gul_P21
mkfifo fifo/full_correlation/gul_P22
mkfifo fifo/full_correlation/gul_P23
mkfifo fifo/full_correlation/gul_P24
mkfifo fifo/full_correlation/gul_P25
mkfifo fifo/full_correlation/gul_P26
mkfifo fifo/full_correlation/gul_P27
mkfifo fifo/full_correlation/gul_P28
mkfifo fifo/full_correlation/gul_P29
mkfifo fifo/full_correlation/gul_P30
mkfifo fifo/full_correlation/gul_P31
mkfifo fifo/full_correlation/gul_P32
mkfifo fifo/full_correlation/gul_P33
mkfifo fifo/full_correlation/gul_P34
mkfifo fifo/full_correlation/gul_P35
mkfifo fifo/full_correlation/gul_P36
mkfifo fifo/full_correlation/gul_P37
mkfifo fifo/full_correlation/gul_P38
mkfifo fifo/full_correlation/gul_P39
mkfifo fifo/full_correlation/gul_P40

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo fifo/full_correlation/gul_S1_pltcalc_P1

mkfifo fifo/full_correlation/gul_S1_summary_P2
mkfifo fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo fifo/full_correlation/gul_S1_pltcalc_P2

mkfifo fifo/full_correlation/gul_S1_summary_P3
mkfifo fifo/full_correlation/gul_S1_eltcalc_P3
mkfifo fifo/full_correlation/gul_S1_summarycalc_P3
mkfifo fifo/full_correlation/gul_S1_pltcalc_P3

mkfifo fifo/full_correlation/gul_S1_summary_P4
mkfifo fifo/full_correlation/gul_S1_eltcalc_P4
mkfifo fifo/full_correlation/gul_S1_summarycalc_P4
mkfifo fifo/full_correlation/gul_S1_pltcalc_P4

mkfifo fifo/full_correlation/gul_S1_summary_P5
mkfifo fifo/full_correlation/gul_S1_eltcalc_P5
mkfifo fifo/full_correlation/gul_S1_summarycalc_P5
mkfifo fifo/full_correlation/gul_S1_pltcalc_P5

mkfifo fifo/full_correlation/gul_S1_summary_P6
mkfifo fifo/full_correlation/gul_S1_eltcalc_P6
mkfifo fifo/full_correlation/gul_S1_summarycalc_P6
mkfifo fifo/full_correlation/gul_S1_pltcalc_P6

mkfifo fifo/full_correlation/gul_S1_summary_P7
mkfifo fifo/full_correlation/gul_S1_eltcalc_P7
mkfifo fifo/full_correlation/gul_S1_summarycalc_P7
mkfifo fifo/full_correlation/gul_S1_pltcalc_P7

mkfifo fifo/full_correlation/gul_S1_summary_P8
mkfifo fifo/full_correlation/gul_S1_eltcalc_P8
mkfifo fifo/full_correlation/gul_S1_summarycalc_P8
mkfifo fifo/full_correlation/gul_S1_pltcalc_P8

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S1_eltcalc_P9
mkfifo fifo/full_correlation/gul_S1_summarycalc_P9
mkfifo fifo/full_correlation/gul_S1_pltcalc_P9

mkfifo fifo/full_correlation/gul_S1_summary_P10
mkfifo fifo/full_correlation/gul_S1_eltcalc_P10
mkfifo fifo/full_correlation/gul_S1_summarycalc_P10
mkfifo fifo/full_correlation/gul_S1_pltcalc_P10

mkfifo fifo/full_correlation/gul_S1_summary_P11
mkfifo fifo/full_correlation/gul_S1_eltcalc_P11
mkfifo fifo/full_correlation/gul_S1_summarycalc_P11
mkfifo fifo/full_correlation/gul_S1_pltcalc_P11

mkfifo fifo/full_correlation/gul_S1_summary_P12
mkfifo fifo/full_correlation/gul_S1_eltcalc_P12
mkfifo fifo/full_correlation/gul_S1_summarycalc_P12
mkfifo fifo/full_correlation/gul_S1_pltcalc_P12

mkfifo fifo/full_correlation/gul_S1_summary_P13
mkfifo fifo/full_correlation/gul_S1_eltcalc_P13
mkfifo fifo/full_correlation/gul_S1_summarycalc_P13
mkfifo fifo/full_correlation/gul_S1_pltcalc_P13

mkfifo fifo/full_correlation/gul_S1_summary_P14
mkfifo fifo/full_correlation/gul_S1_eltcalc_P14
mkfifo fifo/full_correlation/gul_S1_summarycalc_P14
mkfifo fifo/full_correlation/gul_S1_pltcalc_P14

mkfifo fifo/full_correlation/gul_S1_summary_P15
mkfifo fifo/full_correlation/gul_S1_eltcalc_P15
mkfifo fifo/full_correlation/gul_S1_summarycalc_P15
mkfifo fifo/full_correlation/gul_S1_pltcalc_P15

mkfifo fifo/full_correlation/gul_S1_summary_P16
mkfifo fifo/full_correlation/gul_S1_eltcalc_P16
mkfifo fifo/full_correlation/gul_S1_summarycalc_P16
mkfifo fifo/full_correlation/gul_S1_pltcalc_P16

mkfifo fifo/full_correlation/gul_S1_summary_P17
mkfifo fifo/full_correlation/gul_S1_eltcalc_P17
mkfifo fifo/full_correlation/gul_S1_summarycalc_P17
mkfifo fifo/full_correlation/gul_S1_pltcalc_P17

mkfifo fifo/full_correlation/gul_S1_summary_P18
mkfifo fifo/full_correlation/gul_S1_eltcalc_P18
mkfifo fifo/full_correlation/gul_S1_summarycalc_P18
mkfifo fifo/full_correlation/gul_S1_pltcalc_P18

mkfifo fifo/full_correlation/gul_S1_summary_P19
mkfifo fifo/full_correlation/gul_S1_eltcalc_P19
mkfifo fifo/full_correlation/gul_S1_summarycalc_P19
mkfifo fifo/full_correlation/gul_S1_pltcalc_P19

mkfifo fifo/full_correlation/gul_S1_summary_P20
mkfifo fifo/full_correlation/gul_S1_eltcalc_P20
mkfifo fifo/full_correlation/gul_S1_summarycalc_P20
mkfifo fifo/full_correlation/gul_S1_pltcalc_P20

mkfifo fifo/full_correlation/gul_S1_summary_P21
mkfifo fifo/full_correlation/gul_S1_eltcalc_P21
mkfifo fifo/full_correlation/gul_S1_summarycalc_P21
mkfifo fifo/full_correlation/gul_S1_pltcalc_P21

mkfifo fifo/full_correlation/gul_S1_summary_P22
mkfifo fifo/full_correlation/gul_S1_eltcalc_P22
mkfifo fifo/full_correlation/gul_S1_summarycalc_P22
mkfifo fifo/full_correlation/gul_S1_pltcalc_P22

mkfifo fifo/full_correlation/gul_S1_summary_P23
mkfifo fifo/full_correlation/gul_S1_eltcalc_P23
mkfifo fifo/full_correlation/gul_S1_summarycalc_P23
mkfifo fifo/full_correlation/gul_S1_pltcalc_P23

mkfifo fifo/full_correlation/gul_S1_summary_P24
mkfifo fifo/full_correlation/gul_S1_eltcalc_P24
mkfifo fifo/full_correlation/gul_S1_summarycalc_P24
mkfifo fifo/full_correlation/gul_S1_pltcalc_P24

mkfifo fifo/full_correlation/gul_S1_summary_P25
mkfifo fifo/full_correlation/gul_S1_eltcalc_P25
mkfifo fifo/full_correlation/gul_S1_summarycalc_P25
mkfifo fifo/full_correlation/gul_S1_pltcalc_P25

mkfifo fifo/full_correlation/gul_S1_summary_P26
mkfifo fifo/full_correlation/gul_S1_eltcalc_P26
mkfifo fifo/full_correlation/gul_S1_summarycalc_P26
mkfifo fifo/full_correlation/gul_S1_pltcalc_P26

mkfifo fifo/full_correlation/gul_S1_summary_P27
mkfifo fifo/full_correlation/gul_S1_eltcalc_P27
mkfifo fifo/full_correlation/gul_S1_summarycalc_P27
mkfifo fifo/full_correlation/gul_S1_pltcalc_P27

mkfifo fifo/full_correlation/gul_S1_summary_P28
mkfifo fifo/full_correlation/gul_S1_eltcalc_P28
mkfifo fifo/full_correlation/gul_S1_summarycalc_P28
mkfifo fifo/full_correlation/gul_S1_pltcalc_P28

mkfifo fifo/full_correlation/gul_S1_summary_P29
mkfifo fifo/full_correlation/gul_S1_eltcalc_P29
mkfifo fifo/full_correlation/gul_S1_summarycalc_P29
mkfifo fifo/full_correlation/gul_S1_pltcalc_P29

mkfifo fifo/full_correlation/gul_S1_summary_P30
mkfifo fifo/full_correlation/gul_S1_eltcalc_P30
mkfifo fifo/full_correlation/gul_S1_summarycalc_P30
mkfifo fifo/full_correlation/gul_S1_pltcalc_P30

mkfifo fifo/full_correlation/gul_S1_summary_P31
mkfifo fifo/full_correlation/gul_S1_eltcalc_P31
mkfifo fifo/full_correlation/gul_S1_summarycalc_P31
mkfifo fifo/full_correlation/gul_S1_pltcalc_P31

mkfifo fifo/full_correlation/gul_S1_summary_P32
mkfifo fifo/full_correlation/gul_S1_eltcalc_P32
mkfifo fifo/full_correlation/gul_S1_summarycalc_P32
mkfifo fifo/full_correlation/gul_S1_pltcalc_P32

mkfifo fifo/full_correlation/gul_S1_summary_P33
mkfifo fifo/full_correlation/gul_S1_eltcalc_P33
mkfifo fifo/full_correlation/gul_S1_summarycalc_P33
mkfifo fifo/full_correlation/gul_S1_pltcalc_P33

mkfifo fifo/full_correlation/gul_S1_summary_P34
mkfifo fifo/full_correlation/gul_S1_eltcalc_P34
mkfifo fifo/full_correlation/gul_S1_summarycalc_P34
mkfifo fifo/full_correlation/gul_S1_pltcalc_P34

mkfifo fifo/full_correlation/gul_S1_summary_P35
mkfifo fifo/full_correlation/gul_S1_eltcalc_P35
mkfifo fifo/full_correlation/gul_S1_summarycalc_P35
mkfifo fifo/full_correlation/gul_S1_pltcalc_P35

mkfifo fifo/full_correlation/gul_S1_summary_P36
mkfifo fifo/full_correlation/gul_S1_eltcalc_P36
mkfifo fifo/full_correlation/gul_S1_summarycalc_P36
mkfifo fifo/full_correlation/gul_S1_pltcalc_P36

mkfifo fifo/full_correlation/gul_S1_summary_P37
mkfifo fifo/full_correlation/gul_S1_eltcalc_P37
mkfifo fifo/full_correlation/gul_S1_summarycalc_P37
mkfifo fifo/full_correlation/gul_S1_pltcalc_P37

mkfifo fifo/full_correlation/gul_S1_summary_P38
mkfifo fifo/full_correlation/gul_S1_eltcalc_P38
mkfifo fifo/full_correlation/gul_S1_summarycalc_P38
mkfifo fifo/full_correlation/gul_S1_pltcalc_P38

mkfifo fifo/full_correlation/gul_S1_summary_P39
mkfifo fifo/full_correlation/gul_S1_eltcalc_P39
mkfifo fifo/full_correlation/gul_S1_summarycalc_P39
mkfifo fifo/full_correlation/gul_S1_pltcalc_P39

mkfifo fifo/full_correlation/gul_S1_summary_P40
mkfifo fifo/full_correlation/gul_S1_eltcalc_P40
mkfifo fifo/full_correlation/gul_S1_summarycalc_P40
mkfifo fifo/full_correlation/gul_S1_pltcalc_P40

mkfifo fifo/full_correlation/il_P1
mkfifo fifo/full_correlation/il_P2
mkfifo fifo/full_correlation/il_P3
mkfifo fifo/full_correlation/il_P4
mkfifo fifo/full_correlation/il_P5
mkfifo fifo/full_correlation/il_P6
mkfifo fifo/full_correlation/il_P7
mkfifo fifo/full_correlation/il_P8
mkfifo fifo/full_correlation/il_P9
mkfifo fifo/full_correlation/il_P10
mkfifo fifo/full_correlation/il_P11
mkfifo fifo/full_correlation/il_P12
mkfifo fifo/full_correlation/il_P13
mkfifo fifo/full_correlation/il_P14
mkfifo fifo/full_correlation/il_P15
mkfifo fifo/full_correlation/il_P16
mkfifo fifo/full_correlation/il_P17
mkfifo fifo/full_correlation/il_P18
mkfifo fifo/full_correlation/il_P19
mkfifo fifo/full_correlation/il_P20
mkfifo fifo/full_correlation/il_P21
mkfifo fifo/full_correlation/il_P22
mkfifo fifo/full_correlation/il_P23
mkfifo fifo/full_correlation/il_P24
mkfifo fifo/full_correlation/il_P25
mkfifo fifo/full_correlation/il_P26
mkfifo fifo/full_correlation/il_P27
mkfifo fifo/full_correlation/il_P28
mkfifo fifo/full_correlation/il_P29
mkfifo fifo/full_correlation/il_P30
mkfifo fifo/full_correlation/il_P31
mkfifo fifo/full_correlation/il_P32
mkfifo fifo/full_correlation/il_P33
mkfifo fifo/full_correlation/il_P34
mkfifo fifo/full_correlation/il_P35
mkfifo fifo/full_correlation/il_P36
mkfifo fifo/full_correlation/il_P37
mkfifo fifo/full_correlation/il_P38
mkfifo fifo/full_correlation/il_P39
mkfifo fifo/full_correlation/il_P40

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P2
mkfifo fifo/full_correlation/il_S1_eltcalc_P2
mkfifo fifo/full_correlation/il_S1_summarycalc_P2
mkfifo fifo/full_correlation/il_S1_pltcalc_P2

mkfifo fifo/full_correlation/il_S1_summary_P3
mkfifo fifo/full_correlation/il_S1_eltcalc_P3
mkfifo fifo/full_correlation/il_S1_summarycalc_P3
mkfifo fifo/full_correlation/il_S1_pltcalc_P3

mkfifo fifo/full_correlation/il_S1_summary_P4
mkfifo fifo/full_correlation/il_S1_eltcalc_P4
mkfifo fifo/full_correlation/il_S1_summarycalc_P4
mkfifo fifo/full_correlation/il_S1_pltcalc_P4

mkfifo fifo/full_correlation/il_S1_summary_P5
mkfifo fifo/full_correlation/il_S1_eltcalc_P5
mkfifo fifo/full_correlation/il_S1_summarycalc_P5
mkfifo fifo/full_correlation/il_S1_pltcalc_P5

mkfifo fifo/full_correlation/il_S1_summary_P6
mkfifo fifo/full_correlation/il_S1_eltcalc_P6
mkfifo fifo/full_correlation/il_S1_summarycalc_P6
mkfifo fifo/full_correlation/il_S1_pltcalc_P6

mkfifo fifo/full_correlation/il_S1_summary_P7
mkfifo fifo/full_correlation/il_S1_eltcalc_P7
mkfifo fifo/full_correlation/il_S1_summarycalc_P7
mkfifo fifo/full_correlation/il_S1_pltcalc_P7

mkfifo fifo/full_correlation/il_S1_summary_P8
mkfifo fifo/full_correlation/il_S1_eltcalc_P8
mkfifo fifo/full_correlation/il_S1_summarycalc_P8
mkfifo fifo/full_correlation/il_S1_pltcalc_P8

mkfifo fifo/full_correlation/il_S1_summary_P9
mkfifo fifo/full_correlation/il_S1_eltcalc_P9
mkfifo fifo/full_correlation/il_S1_summarycalc_P9
mkfifo fifo/full_correlation/il_S1_pltcalc_P9

mkfifo fifo/full_correlation/il_S1_summary_P10
mkfifo fifo/full_correlation/il_S1_eltcalc_P10
mkfifo fifo/full_correlation/il_S1_summarycalc_P10
mkfifo fifo/full_correlation/il_S1_pltcalc_P10

mkfifo fifo/full_correlation/il_S1_summary_P11
mkfifo fifo/full_correlation/il_S1_eltcalc_P11
mkfifo fifo/full_correlation/il_S1_summarycalc_P11
mkfifo fifo/full_correlation/il_S1_pltcalc_P11

mkfifo fifo/full_correlation/il_S1_summary_P12
mkfifo fifo/full_correlation/il_S1_eltcalc_P12
mkfifo fifo/full_correlation/il_S1_summarycalc_P12
mkfifo fifo/full_correlation/il_S1_pltcalc_P12

mkfifo fifo/full_correlation/il_S1_summary_P13
mkfifo fifo/full_correlation/il_S1_eltcalc_P13
mkfifo fifo/full_correlation/il_S1_summarycalc_P13
mkfifo fifo/full_correlation/il_S1_pltcalc_P13

mkfifo fifo/full_correlation/il_S1_summary_P14
mkfifo fifo/full_correlation/il_S1_eltcalc_P14
mkfifo fifo/full_correlation/il_S1_summarycalc_P14
mkfifo fifo/full_correlation/il_S1_pltcalc_P14

mkfifo fifo/full_correlation/il_S1_summary_P15
mkfifo fifo/full_correlation/il_S1_eltcalc_P15
mkfifo fifo/full_correlation/il_S1_summarycalc_P15
mkfifo fifo/full_correlation/il_S1_pltcalc_P15

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_eltcalc_P16
mkfifo fifo/full_correlation/il_S1_summarycalc_P16
mkfifo fifo/full_correlation/il_S1_pltcalc_P16

mkfifo fifo/full_correlation/il_S1_summary_P17
mkfifo fifo/full_correlation/il_S1_eltcalc_P17
mkfifo fifo/full_correlation/il_S1_summarycalc_P17
mkfifo fifo/full_correlation/il_S1_pltcalc_P17

mkfifo fifo/full_correlation/il_S1_summary_P18
mkfifo fifo/full_correlation/il_S1_eltcalc_P18
mkfifo fifo/full_correlation/il_S1_summarycalc_P18
mkfifo fifo/full_correlation/il_S1_pltcalc_P18

mkfifo fifo/full_correlation/il_S1_summary_P19
mkfifo fifo/full_correlation/il_S1_eltcalc_P19
mkfifo fifo/full_correlation/il_S1_summarycalc_P19
mkfifo fifo/full_correlation/il_S1_pltcalc_P19

mkfifo fifo/full_correlation/il_S1_summary_P20
mkfifo fifo/full_correlation/il_S1_eltcalc_P20
mkfifo fifo/full_correlation/il_S1_summarycalc_P20
mkfifo fifo/full_correlation/il_S1_pltcalc_P20

mkfifo fifo/full_correlation/il_S1_summary_P21
mkfifo fifo/full_correlation/il_S1_eltcalc_P21
mkfifo fifo/full_correlation/il_S1_summarycalc_P21
mkfifo fifo/full_correlation/il_S1_pltcalc_P21

mkfifo fifo/full_correlation/il_S1_summary_P22
mkfifo fifo/full_correlation/il_S1_eltcalc_P22
mkfifo fifo/full_correlation/il_S1_summarycalc_P22
mkfifo fifo/full_correlation/il_S1_pltcalc_P22

mkfifo fifo/full_correlation/il_S1_summary_P23
mkfifo fifo/full_correlation/il_S1_eltcalc_P23
mkfifo fifo/full_correlation/il_S1_summarycalc_P23
mkfifo fifo/full_correlation/il_S1_pltcalc_P23

mkfifo fifo/full_correlation/il_S1_summary_P24
mkfifo fifo/full_correlation/il_S1_eltcalc_P24
mkfifo fifo/full_correlation/il_S1_summarycalc_P24
mkfifo fifo/full_correlation/il_S1_pltcalc_P24

mkfifo fifo/full_correlation/il_S1_summary_P25
mkfifo fifo/full_correlation/il_S1_eltcalc_P25
mkfifo fifo/full_correlation/il_S1_summarycalc_P25
mkfifo fifo/full_correlation/il_S1_pltcalc_P25

mkfifo fifo/full_correlation/il_S1_summary_P26
mkfifo fifo/full_correlation/il_S1_eltcalc_P26
mkfifo fifo/full_correlation/il_S1_summarycalc_P26
mkfifo fifo/full_correlation/il_S1_pltcalc_P26

mkfifo fifo/full_correlation/il_S1_summary_P27
mkfifo fifo/full_correlation/il_S1_eltcalc_P27
mkfifo fifo/full_correlation/il_S1_summarycalc_P27
mkfifo fifo/full_correlation/il_S1_pltcalc_P27

mkfifo fifo/full_correlation/il_S1_summary_P28
mkfifo fifo/full_correlation/il_S1_eltcalc_P28
mkfifo fifo/full_correlation/il_S1_summarycalc_P28
mkfifo fifo/full_correlation/il_S1_pltcalc_P28

mkfifo fifo/full_correlation/il_S1_summary_P29
mkfifo fifo/full_correlation/il_S1_eltcalc_P29
mkfifo fifo/full_correlation/il_S1_summarycalc_P29
mkfifo fifo/full_correlation/il_S1_pltcalc_P29

mkfifo fifo/full_correlation/il_S1_summary_P30
mkfifo fifo/full_correlation/il_S1_eltcalc_P30
mkfifo fifo/full_correlation/il_S1_summarycalc_P30
mkfifo fifo/full_correlation/il_S1_pltcalc_P30

mkfifo fifo/full_correlation/il_S1_summary_P31
mkfifo fifo/full_correlation/il_S1_eltcalc_P31
mkfifo fifo/full_correlation/il_S1_summarycalc_P31
mkfifo fifo/full_correlation/il_S1_pltcalc_P31

mkfifo fifo/full_correlation/il_S1_summary_P32
mkfifo fifo/full_correlation/il_S1_eltcalc_P32
mkfifo fifo/full_correlation/il_S1_summarycalc_P32
mkfifo fifo/full_correlation/il_S1_pltcalc_P32

mkfifo fifo/full_correlation/il_S1_summary_P33
mkfifo fifo/full_correlation/il_S1_eltcalc_P33
mkfifo fifo/full_correlation/il_S1_summarycalc_P33
mkfifo fifo/full_correlation/il_S1_pltcalc_P33

mkfifo fifo/full_correlation/il_S1_summary_P34
mkfifo fifo/full_correlation/il_S1_eltcalc_P34
mkfifo fifo/full_correlation/il_S1_summarycalc_P34
mkfifo fifo/full_correlation/il_S1_pltcalc_P34

mkfifo fifo/full_correlation/il_S1_summary_P35
mkfifo fifo/full_correlation/il_S1_eltcalc_P35
mkfifo fifo/full_correlation/il_S1_summarycalc_P35
mkfifo fifo/full_correlation/il_S1_pltcalc_P35

mkfifo fifo/full_correlation/il_S1_summary_P36
mkfifo fifo/full_correlation/il_S1_eltcalc_P36
mkfifo fifo/full_correlation/il_S1_summarycalc_P36
mkfifo fifo/full_correlation/il_S1_pltcalc_P36

mkfifo fifo/full_correlation/il_S1_summary_P37
mkfifo fifo/full_correlation/il_S1_eltcalc_P37
mkfifo fifo/full_correlation/il_S1_summarycalc_P37
mkfifo fifo/full_correlation/il_S1_pltcalc_P37

mkfifo fifo/full_correlation/il_S1_summary_P38
mkfifo fifo/full_correlation/il_S1_eltcalc_P38
mkfifo fifo/full_correlation/il_S1_summarycalc_P38
mkfifo fifo/full_correlation/il_S1_pltcalc_P38

mkfifo fifo/full_correlation/il_S1_summary_P39
mkfifo fifo/full_correlation/il_S1_eltcalc_P39
mkfifo fifo/full_correlation/il_S1_summarycalc_P39
mkfifo fifo/full_correlation/il_S1_pltcalc_P39

mkfifo fifo/full_correlation/il_S1_summary_P40
mkfifo fifo/full_correlation/il_S1_eltcalc_P40
mkfifo fifo/full_correlation/il_S1_summarycalc_P40
mkfifo fifo/full_correlation/il_S1_pltcalc_P40



# --- Do insured loss computes ---

( eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> log/stderror.err & pid1=$!
( summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 ) 2>> log/stderror.err & pid2=$!
( pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 ) 2>> log/stderror.err & pid3=$!
( eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 ) 2>> log/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 ) 2>> log/stderror.err & pid5=$!
( pltcalc -s < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 ) 2>> log/stderror.err & pid6=$!
( eltcalc -s < fifo/il_S1_eltcalc_P3 > work/kat/il_S1_eltcalc_P3 ) 2>> log/stderror.err & pid7=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P3 > work/kat/il_S1_summarycalc_P3 ) 2>> log/stderror.err & pid8=$!
( pltcalc -s < fifo/il_S1_pltcalc_P3 > work/kat/il_S1_pltcalc_P3 ) 2>> log/stderror.err & pid9=$!
( eltcalc -s < fifo/il_S1_eltcalc_P4 > work/kat/il_S1_eltcalc_P4 ) 2>> log/stderror.err & pid10=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 ) 2>> log/stderror.err & pid11=$!
( pltcalc -s < fifo/il_S1_pltcalc_P4 > work/kat/il_S1_pltcalc_P4 ) 2>> log/stderror.err & pid12=$!
( eltcalc -s < fifo/il_S1_eltcalc_P5 > work/kat/il_S1_eltcalc_P5 ) 2>> log/stderror.err & pid13=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 ) 2>> log/stderror.err & pid14=$!
( pltcalc -s < fifo/il_S1_pltcalc_P5 > work/kat/il_S1_pltcalc_P5 ) 2>> log/stderror.err & pid15=$!
( eltcalc -s < fifo/il_S1_eltcalc_P6 > work/kat/il_S1_eltcalc_P6 ) 2>> log/stderror.err & pid16=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P6 > work/kat/il_S1_summarycalc_P6 ) 2>> log/stderror.err & pid17=$!
( pltcalc -s < fifo/il_S1_pltcalc_P6 > work/kat/il_S1_pltcalc_P6 ) 2>> log/stderror.err & pid18=$!
( eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 ) 2>> log/stderror.err & pid19=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P7 > work/kat/il_S1_summarycalc_P7 ) 2>> log/stderror.err & pid20=$!
( pltcalc -s < fifo/il_S1_pltcalc_P7 > work/kat/il_S1_pltcalc_P7 ) 2>> log/stderror.err & pid21=$!
( eltcalc -s < fifo/il_S1_eltcalc_P8 > work/kat/il_S1_eltcalc_P8 ) 2>> log/stderror.err & pid22=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P8 > work/kat/il_S1_summarycalc_P8 ) 2>> log/stderror.err & pid23=$!
( pltcalc -s < fifo/il_S1_pltcalc_P8 > work/kat/il_S1_pltcalc_P8 ) 2>> log/stderror.err & pid24=$!
( eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 ) 2>> log/stderror.err & pid25=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 ) 2>> log/stderror.err & pid26=$!
( pltcalc -s < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> log/stderror.err & pid27=$!
( eltcalc -s < fifo/il_S1_eltcalc_P10 > work/kat/il_S1_eltcalc_P10 ) 2>> log/stderror.err & pid28=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P10 > work/kat/il_S1_summarycalc_P10 ) 2>> log/stderror.err & pid29=$!
( pltcalc -s < fifo/il_S1_pltcalc_P10 > work/kat/il_S1_pltcalc_P10 ) 2>> log/stderror.err & pid30=$!
( eltcalc -s < fifo/il_S1_eltcalc_P11 > work/kat/il_S1_eltcalc_P11 ) 2>> log/stderror.err & pid31=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 ) 2>> log/stderror.err & pid32=$!
( pltcalc -s < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 ) 2>> log/stderror.err & pid33=$!
( eltcalc -s < fifo/il_S1_eltcalc_P12 > work/kat/il_S1_eltcalc_P12 ) 2>> log/stderror.err & pid34=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P12 > work/kat/il_S1_summarycalc_P12 ) 2>> log/stderror.err & pid35=$!
( pltcalc -s < fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 ) 2>> log/stderror.err & pid36=$!
( eltcalc -s < fifo/il_S1_eltcalc_P13 > work/kat/il_S1_eltcalc_P13 ) 2>> log/stderror.err & pid37=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P13 > work/kat/il_S1_summarycalc_P13 ) 2>> log/stderror.err & pid38=$!
( pltcalc -s < fifo/il_S1_pltcalc_P13 > work/kat/il_S1_pltcalc_P13 ) 2>> log/stderror.err & pid39=$!
( eltcalc -s < fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 ) 2>> log/stderror.err & pid40=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P14 > work/kat/il_S1_summarycalc_P14 ) 2>> log/stderror.err & pid41=$!
( pltcalc -s < fifo/il_S1_pltcalc_P14 > work/kat/il_S1_pltcalc_P14 ) 2>> log/stderror.err & pid42=$!
( eltcalc -s < fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 ) 2>> log/stderror.err & pid43=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P15 > work/kat/il_S1_summarycalc_P15 ) 2>> log/stderror.err & pid44=$!
( pltcalc -s < fifo/il_S1_pltcalc_P15 > work/kat/il_S1_pltcalc_P15 ) 2>> log/stderror.err & pid45=$!
( eltcalc -s < fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 ) 2>> log/stderror.err & pid46=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P16 > work/kat/il_S1_summarycalc_P16 ) 2>> log/stderror.err & pid47=$!
( pltcalc -s < fifo/il_S1_pltcalc_P16 > work/kat/il_S1_pltcalc_P16 ) 2>> log/stderror.err & pid48=$!
( eltcalc -s < fifo/il_S1_eltcalc_P17 > work/kat/il_S1_eltcalc_P17 ) 2>> log/stderror.err & pid49=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P17 > work/kat/il_S1_summarycalc_P17 ) 2>> log/stderror.err & pid50=$!
( pltcalc -s < fifo/il_S1_pltcalc_P17 > work/kat/il_S1_pltcalc_P17 ) 2>> log/stderror.err & pid51=$!
( eltcalc -s < fifo/il_S1_eltcalc_P18 > work/kat/il_S1_eltcalc_P18 ) 2>> log/stderror.err & pid52=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 ) 2>> log/stderror.err & pid53=$!
( pltcalc -s < fifo/il_S1_pltcalc_P18 > work/kat/il_S1_pltcalc_P18 ) 2>> log/stderror.err & pid54=$!
( eltcalc -s < fifo/il_S1_eltcalc_P19 > work/kat/il_S1_eltcalc_P19 ) 2>> log/stderror.err & pid55=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P19 > work/kat/il_S1_summarycalc_P19 ) 2>> log/stderror.err & pid56=$!
( pltcalc -s < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 ) 2>> log/stderror.err & pid57=$!
( eltcalc -s < fifo/il_S1_eltcalc_P20 > work/kat/il_S1_eltcalc_P20 ) 2>> log/stderror.err & pid58=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P20 > work/kat/il_S1_summarycalc_P20 ) 2>> log/stderror.err & pid59=$!
( pltcalc -s < fifo/il_S1_pltcalc_P20 > work/kat/il_S1_pltcalc_P20 ) 2>> log/stderror.err & pid60=$!
( eltcalc -s < fifo/il_S1_eltcalc_P21 > work/kat/il_S1_eltcalc_P21 ) 2>> log/stderror.err & pid61=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P21 > work/kat/il_S1_summarycalc_P21 ) 2>> log/stderror.err & pid62=$!
( pltcalc -s < fifo/il_S1_pltcalc_P21 > work/kat/il_S1_pltcalc_P21 ) 2>> log/stderror.err & pid63=$!
( eltcalc -s < fifo/il_S1_eltcalc_P22 > work/kat/il_S1_eltcalc_P22 ) 2>> log/stderror.err & pid64=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P22 > work/kat/il_S1_summarycalc_P22 ) 2>> log/stderror.err & pid65=$!
( pltcalc -s < fifo/il_S1_pltcalc_P22 > work/kat/il_S1_pltcalc_P22 ) 2>> log/stderror.err & pid66=$!
( eltcalc -s < fifo/il_S1_eltcalc_P23 > work/kat/il_S1_eltcalc_P23 ) 2>> log/stderror.err & pid67=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P23 > work/kat/il_S1_summarycalc_P23 ) 2>> log/stderror.err & pid68=$!
( pltcalc -s < fifo/il_S1_pltcalc_P23 > work/kat/il_S1_pltcalc_P23 ) 2>> log/stderror.err & pid69=$!
( eltcalc -s < fifo/il_S1_eltcalc_P24 > work/kat/il_S1_eltcalc_P24 ) 2>> log/stderror.err & pid70=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P24 > work/kat/il_S1_summarycalc_P24 ) 2>> log/stderror.err & pid71=$!
( pltcalc -s < fifo/il_S1_pltcalc_P24 > work/kat/il_S1_pltcalc_P24 ) 2>> log/stderror.err & pid72=$!
( eltcalc -s < fifo/il_S1_eltcalc_P25 > work/kat/il_S1_eltcalc_P25 ) 2>> log/stderror.err & pid73=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P25 > work/kat/il_S1_summarycalc_P25 ) 2>> log/stderror.err & pid74=$!
( pltcalc -s < fifo/il_S1_pltcalc_P25 > work/kat/il_S1_pltcalc_P25 ) 2>> log/stderror.err & pid75=$!
( eltcalc -s < fifo/il_S1_eltcalc_P26 > work/kat/il_S1_eltcalc_P26 ) 2>> log/stderror.err & pid76=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P26 > work/kat/il_S1_summarycalc_P26 ) 2>> log/stderror.err & pid77=$!
( pltcalc -s < fifo/il_S1_pltcalc_P26 > work/kat/il_S1_pltcalc_P26 ) 2>> log/stderror.err & pid78=$!
( eltcalc -s < fifo/il_S1_eltcalc_P27 > work/kat/il_S1_eltcalc_P27 ) 2>> log/stderror.err & pid79=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P27 > work/kat/il_S1_summarycalc_P27 ) 2>> log/stderror.err & pid80=$!
( pltcalc -s < fifo/il_S1_pltcalc_P27 > work/kat/il_S1_pltcalc_P27 ) 2>> log/stderror.err & pid81=$!
( eltcalc -s < fifo/il_S1_eltcalc_P28 > work/kat/il_S1_eltcalc_P28 ) 2>> log/stderror.err & pid82=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P28 > work/kat/il_S1_summarycalc_P28 ) 2>> log/stderror.err & pid83=$!
( pltcalc -s < fifo/il_S1_pltcalc_P28 > work/kat/il_S1_pltcalc_P28 ) 2>> log/stderror.err & pid84=$!
( eltcalc -s < fifo/il_S1_eltcalc_P29 > work/kat/il_S1_eltcalc_P29 ) 2>> log/stderror.err & pid85=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P29 > work/kat/il_S1_summarycalc_P29 ) 2>> log/stderror.err & pid86=$!
( pltcalc -s < fifo/il_S1_pltcalc_P29 > work/kat/il_S1_pltcalc_P29 ) 2>> log/stderror.err & pid87=$!
( eltcalc -s < fifo/il_S1_eltcalc_P30 > work/kat/il_S1_eltcalc_P30 ) 2>> log/stderror.err & pid88=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P30 > work/kat/il_S1_summarycalc_P30 ) 2>> log/stderror.err & pid89=$!
( pltcalc -s < fifo/il_S1_pltcalc_P30 > work/kat/il_S1_pltcalc_P30 ) 2>> log/stderror.err & pid90=$!
( eltcalc -s < fifo/il_S1_eltcalc_P31 > work/kat/il_S1_eltcalc_P31 ) 2>> log/stderror.err & pid91=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P31 > work/kat/il_S1_summarycalc_P31 ) 2>> log/stderror.err & pid92=$!
( pltcalc -s < fifo/il_S1_pltcalc_P31 > work/kat/il_S1_pltcalc_P31 ) 2>> log/stderror.err & pid93=$!
( eltcalc -s < fifo/il_S1_eltcalc_P32 > work/kat/il_S1_eltcalc_P32 ) 2>> log/stderror.err & pid94=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P32 > work/kat/il_S1_summarycalc_P32 ) 2>> log/stderror.err & pid95=$!
( pltcalc -s < fifo/il_S1_pltcalc_P32 > work/kat/il_S1_pltcalc_P32 ) 2>> log/stderror.err & pid96=$!
( eltcalc -s < fifo/il_S1_eltcalc_P33 > work/kat/il_S1_eltcalc_P33 ) 2>> log/stderror.err & pid97=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P33 > work/kat/il_S1_summarycalc_P33 ) 2>> log/stderror.err & pid98=$!
( pltcalc -s < fifo/il_S1_pltcalc_P33 > work/kat/il_S1_pltcalc_P33 ) 2>> log/stderror.err & pid99=$!
( eltcalc -s < fifo/il_S1_eltcalc_P34 > work/kat/il_S1_eltcalc_P34 ) 2>> log/stderror.err & pid100=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P34 > work/kat/il_S1_summarycalc_P34 ) 2>> log/stderror.err & pid101=$!
( pltcalc -s < fifo/il_S1_pltcalc_P34 > work/kat/il_S1_pltcalc_P34 ) 2>> log/stderror.err & pid102=$!
( eltcalc -s < fifo/il_S1_eltcalc_P35 > work/kat/il_S1_eltcalc_P35 ) 2>> log/stderror.err & pid103=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P35 > work/kat/il_S1_summarycalc_P35 ) 2>> log/stderror.err & pid104=$!
( pltcalc -s < fifo/il_S1_pltcalc_P35 > work/kat/il_S1_pltcalc_P35 ) 2>> log/stderror.err & pid105=$!
( eltcalc -s < fifo/il_S1_eltcalc_P36 > work/kat/il_S1_eltcalc_P36 ) 2>> log/stderror.err & pid106=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P36 > work/kat/il_S1_summarycalc_P36 ) 2>> log/stderror.err & pid107=$!
( pltcalc -s < fifo/il_S1_pltcalc_P36 > work/kat/il_S1_pltcalc_P36 ) 2>> log/stderror.err & pid108=$!
( eltcalc -s < fifo/il_S1_eltcalc_P37 > work/kat/il_S1_eltcalc_P37 ) 2>> log/stderror.err & pid109=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P37 > work/kat/il_S1_summarycalc_P37 ) 2>> log/stderror.err & pid110=$!
( pltcalc -s < fifo/il_S1_pltcalc_P37 > work/kat/il_S1_pltcalc_P37 ) 2>> log/stderror.err & pid111=$!
( eltcalc -s < fifo/il_S1_eltcalc_P38 > work/kat/il_S1_eltcalc_P38 ) 2>> log/stderror.err & pid112=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P38 > work/kat/il_S1_summarycalc_P38 ) 2>> log/stderror.err & pid113=$!
( pltcalc -s < fifo/il_S1_pltcalc_P38 > work/kat/il_S1_pltcalc_P38 ) 2>> log/stderror.err & pid114=$!
( eltcalc -s < fifo/il_S1_eltcalc_P39 > work/kat/il_S1_eltcalc_P39 ) 2>> log/stderror.err & pid115=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P39 > work/kat/il_S1_summarycalc_P39 ) 2>> log/stderror.err & pid116=$!
( pltcalc -s < fifo/il_S1_pltcalc_P39 > work/kat/il_S1_pltcalc_P39 ) 2>> log/stderror.err & pid117=$!
( eltcalc -s < fifo/il_S1_eltcalc_P40 > work/kat/il_S1_eltcalc_P40 ) 2>> log/stderror.err & pid118=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P40 > work/kat/il_S1_summarycalc_P40 ) 2>> log/stderror.err & pid119=$!
( pltcalc -s < fifo/il_S1_pltcalc_P40 > work/kat/il_S1_pltcalc_P40 ) 2>> log/stderror.err & pid120=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid121=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid122=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_eltcalc_P3 fifo/il_S1_summarycalc_P3 fifo/il_S1_pltcalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid123=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_eltcalc_P4 fifo/il_S1_summarycalc_P4 fifo/il_S1_pltcalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid124=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_eltcalc_P5 fifo/il_S1_summarycalc_P5 fifo/il_S1_pltcalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid125=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_eltcalc_P6 fifo/il_S1_summarycalc_P6 fifo/il_S1_pltcalc_P6 work/il_S1_summaryaalcalc/P6.bin work/il_S1_summaryleccalc/P6.bin > /dev/null & pid126=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 fifo/il_S1_summarycalc_P7 fifo/il_S1_pltcalc_P7 work/il_S1_summaryaalcalc/P7.bin work/il_S1_summaryleccalc/P7.bin > /dev/null & pid127=$!
tee < fifo/il_S1_summary_P8 fifo/il_S1_eltcalc_P8 fifo/il_S1_summarycalc_P8 fifo/il_S1_pltcalc_P8 work/il_S1_summaryaalcalc/P8.bin work/il_S1_summaryleccalc/P8.bin > /dev/null & pid128=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid129=$!
tee < fifo/il_S1_summary_P10 fifo/il_S1_eltcalc_P10 fifo/il_S1_summarycalc_P10 fifo/il_S1_pltcalc_P10 work/il_S1_summaryaalcalc/P10.bin work/il_S1_summaryleccalc/P10.bin > /dev/null & pid130=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_eltcalc_P11 fifo/il_S1_summarycalc_P11 fifo/il_S1_pltcalc_P11 work/il_S1_summaryaalcalc/P11.bin work/il_S1_summaryleccalc/P11.bin > /dev/null & pid131=$!
tee < fifo/il_S1_summary_P12 fifo/il_S1_eltcalc_P12 fifo/il_S1_summarycalc_P12 fifo/il_S1_pltcalc_P12 work/il_S1_summaryaalcalc/P12.bin work/il_S1_summaryleccalc/P12.bin > /dev/null & pid132=$!
tee < fifo/il_S1_summary_P13 fifo/il_S1_eltcalc_P13 fifo/il_S1_summarycalc_P13 fifo/il_S1_pltcalc_P13 work/il_S1_summaryaalcalc/P13.bin work/il_S1_summaryleccalc/P13.bin > /dev/null & pid133=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_eltcalc_P14 fifo/il_S1_summarycalc_P14 fifo/il_S1_pltcalc_P14 work/il_S1_summaryaalcalc/P14.bin work/il_S1_summaryleccalc/P14.bin > /dev/null & pid134=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_eltcalc_P15 fifo/il_S1_summarycalc_P15 fifo/il_S1_pltcalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid135=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_eltcalc_P16 fifo/il_S1_summarycalc_P16 fifo/il_S1_pltcalc_P16 work/il_S1_summaryaalcalc/P16.bin work/il_S1_summaryleccalc/P16.bin > /dev/null & pid136=$!
tee < fifo/il_S1_summary_P17 fifo/il_S1_eltcalc_P17 fifo/il_S1_summarycalc_P17 fifo/il_S1_pltcalc_P17 work/il_S1_summaryaalcalc/P17.bin work/il_S1_summaryleccalc/P17.bin > /dev/null & pid137=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_eltcalc_P18 fifo/il_S1_summarycalc_P18 fifo/il_S1_pltcalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid138=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_eltcalc_P19 fifo/il_S1_summarycalc_P19 fifo/il_S1_pltcalc_P19 work/il_S1_summaryaalcalc/P19.bin work/il_S1_summaryleccalc/P19.bin > /dev/null & pid139=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_eltcalc_P20 fifo/il_S1_summarycalc_P20 fifo/il_S1_pltcalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid140=$!
tee < fifo/il_S1_summary_P21 fifo/il_S1_eltcalc_P21 fifo/il_S1_summarycalc_P21 fifo/il_S1_pltcalc_P21 work/il_S1_summaryaalcalc/P21.bin work/il_S1_summaryleccalc/P21.bin > /dev/null & pid141=$!
tee < fifo/il_S1_summary_P22 fifo/il_S1_eltcalc_P22 fifo/il_S1_summarycalc_P22 fifo/il_S1_pltcalc_P22 work/il_S1_summaryaalcalc/P22.bin work/il_S1_summaryleccalc/P22.bin > /dev/null & pid142=$!
tee < fifo/il_S1_summary_P23 fifo/il_S1_eltcalc_P23 fifo/il_S1_summarycalc_P23 fifo/il_S1_pltcalc_P23 work/il_S1_summaryaalcalc/P23.bin work/il_S1_summaryleccalc/P23.bin > /dev/null & pid143=$!
tee < fifo/il_S1_summary_P24 fifo/il_S1_eltcalc_P24 fifo/il_S1_summarycalc_P24 fifo/il_S1_pltcalc_P24 work/il_S1_summaryaalcalc/P24.bin work/il_S1_summaryleccalc/P24.bin > /dev/null & pid144=$!
tee < fifo/il_S1_summary_P25 fifo/il_S1_eltcalc_P25 fifo/il_S1_summarycalc_P25 fifo/il_S1_pltcalc_P25 work/il_S1_summaryaalcalc/P25.bin work/il_S1_summaryleccalc/P25.bin > /dev/null & pid145=$!
tee < fifo/il_S1_summary_P26 fifo/il_S1_eltcalc_P26 fifo/il_S1_summarycalc_P26 fifo/il_S1_pltcalc_P26 work/il_S1_summaryaalcalc/P26.bin work/il_S1_summaryleccalc/P26.bin > /dev/null & pid146=$!
tee < fifo/il_S1_summary_P27 fifo/il_S1_eltcalc_P27 fifo/il_S1_summarycalc_P27 fifo/il_S1_pltcalc_P27 work/il_S1_summaryaalcalc/P27.bin work/il_S1_summaryleccalc/P27.bin > /dev/null & pid147=$!
tee < fifo/il_S1_summary_P28 fifo/il_S1_eltcalc_P28 fifo/il_S1_summarycalc_P28 fifo/il_S1_pltcalc_P28 work/il_S1_summaryaalcalc/P28.bin work/il_S1_summaryleccalc/P28.bin > /dev/null & pid148=$!
tee < fifo/il_S1_summary_P29 fifo/il_S1_eltcalc_P29 fifo/il_S1_summarycalc_P29 fifo/il_S1_pltcalc_P29 work/il_S1_summaryaalcalc/P29.bin work/il_S1_summaryleccalc/P29.bin > /dev/null & pid149=$!
tee < fifo/il_S1_summary_P30 fifo/il_S1_eltcalc_P30 fifo/il_S1_summarycalc_P30 fifo/il_S1_pltcalc_P30 work/il_S1_summaryaalcalc/P30.bin work/il_S1_summaryleccalc/P30.bin > /dev/null & pid150=$!
tee < fifo/il_S1_summary_P31 fifo/il_S1_eltcalc_P31 fifo/il_S1_summarycalc_P31 fifo/il_S1_pltcalc_P31 work/il_S1_summaryaalcalc/P31.bin work/il_S1_summaryleccalc/P31.bin > /dev/null & pid151=$!
tee < fifo/il_S1_summary_P32 fifo/il_S1_eltcalc_P32 fifo/il_S1_summarycalc_P32 fifo/il_S1_pltcalc_P32 work/il_S1_summaryaalcalc/P32.bin work/il_S1_summaryleccalc/P32.bin > /dev/null & pid152=$!
tee < fifo/il_S1_summary_P33 fifo/il_S1_eltcalc_P33 fifo/il_S1_summarycalc_P33 fifo/il_S1_pltcalc_P33 work/il_S1_summaryaalcalc/P33.bin work/il_S1_summaryleccalc/P33.bin > /dev/null & pid153=$!
tee < fifo/il_S1_summary_P34 fifo/il_S1_eltcalc_P34 fifo/il_S1_summarycalc_P34 fifo/il_S1_pltcalc_P34 work/il_S1_summaryaalcalc/P34.bin work/il_S1_summaryleccalc/P34.bin > /dev/null & pid154=$!
tee < fifo/il_S1_summary_P35 fifo/il_S1_eltcalc_P35 fifo/il_S1_summarycalc_P35 fifo/il_S1_pltcalc_P35 work/il_S1_summaryaalcalc/P35.bin work/il_S1_summaryleccalc/P35.bin > /dev/null & pid155=$!
tee < fifo/il_S1_summary_P36 fifo/il_S1_eltcalc_P36 fifo/il_S1_summarycalc_P36 fifo/il_S1_pltcalc_P36 work/il_S1_summaryaalcalc/P36.bin work/il_S1_summaryleccalc/P36.bin > /dev/null & pid156=$!
tee < fifo/il_S1_summary_P37 fifo/il_S1_eltcalc_P37 fifo/il_S1_summarycalc_P37 fifo/il_S1_pltcalc_P37 work/il_S1_summaryaalcalc/P37.bin work/il_S1_summaryleccalc/P37.bin > /dev/null & pid157=$!
tee < fifo/il_S1_summary_P38 fifo/il_S1_eltcalc_P38 fifo/il_S1_summarycalc_P38 fifo/il_S1_pltcalc_P38 work/il_S1_summaryaalcalc/P38.bin work/il_S1_summaryleccalc/P38.bin > /dev/null & pid158=$!
tee < fifo/il_S1_summary_P39 fifo/il_S1_eltcalc_P39 fifo/il_S1_summarycalc_P39 fifo/il_S1_pltcalc_P39 work/il_S1_summaryaalcalc/P39.bin work/il_S1_summaryleccalc/P39.bin > /dev/null & pid159=$!
tee < fifo/il_S1_summary_P40 fifo/il_S1_eltcalc_P40 fifo/il_S1_summarycalc_P40 fifo/il_S1_pltcalc_P40 work/il_S1_summaryaalcalc/P40.bin work/il_S1_summaryleccalc/P40.bin > /dev/null & pid160=$!

( summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P3 < fifo/il_P3 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P8 < fifo/il_P8 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P9 < fifo/il_P9 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P10 < fifo/il_P10 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P12 < fifo/il_P12 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P13 < fifo/il_P13 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P17 < fifo/il_P17 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P21 < fifo/il_P21 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P22 < fifo/il_P22 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P23 < fifo/il_P23 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P24 < fifo/il_P24 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P25 < fifo/il_P25 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P26 < fifo/il_P26 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P27 < fifo/il_P27 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P28 < fifo/il_P28 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P29 < fifo/il_P29 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P30 < fifo/il_P30 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P31 < fifo/il_P31 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P32 < fifo/il_P32 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P33 < fifo/il_P33 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P34 < fifo/il_P34 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P35 < fifo/il_P35 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P36 < fifo/il_P36 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P37 < fifo/il_P37 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P38 < fifo/il_P38 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P39 < fifo/il_P39 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/il_S1_summary_P40 < fifo/il_P40 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> log/stderror.err & pid161=$!
( summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 ) 2>> log/stderror.err & pid162=$!
( pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 ) 2>> log/stderror.err & pid163=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 ) 2>> log/stderror.err & pid164=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 ) 2>> log/stderror.err & pid165=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 ) 2>> log/stderror.err & pid166=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P3 > work/kat/gul_S1_eltcalc_P3 ) 2>> log/stderror.err & pid167=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P3 > work/kat/gul_S1_summarycalc_P3 ) 2>> log/stderror.err & pid168=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P3 > work/kat/gul_S1_pltcalc_P3 ) 2>> log/stderror.err & pid169=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 ) 2>> log/stderror.err & pid170=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P4 > work/kat/gul_S1_summarycalc_P4 ) 2>> log/stderror.err & pid171=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P4 > work/kat/gul_S1_pltcalc_P4 ) 2>> log/stderror.err & pid172=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P5 > work/kat/gul_S1_eltcalc_P5 ) 2>> log/stderror.err & pid173=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P5 > work/kat/gul_S1_summarycalc_P5 ) 2>> log/stderror.err & pid174=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P5 > work/kat/gul_S1_pltcalc_P5 ) 2>> log/stderror.err & pid175=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P6 > work/kat/gul_S1_eltcalc_P6 ) 2>> log/stderror.err & pid176=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P6 > work/kat/gul_S1_summarycalc_P6 ) 2>> log/stderror.err & pid177=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 ) 2>> log/stderror.err & pid178=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P7 > work/kat/gul_S1_eltcalc_P7 ) 2>> log/stderror.err & pid179=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P7 > work/kat/gul_S1_summarycalc_P7 ) 2>> log/stderror.err & pid180=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P7 > work/kat/gul_S1_pltcalc_P7 ) 2>> log/stderror.err & pid181=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P8 > work/kat/gul_S1_eltcalc_P8 ) 2>> log/stderror.err & pid182=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P8 > work/kat/gul_S1_summarycalc_P8 ) 2>> log/stderror.err & pid183=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P8 > work/kat/gul_S1_pltcalc_P8 ) 2>> log/stderror.err & pid184=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> log/stderror.err & pid185=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> log/stderror.err & pid186=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 ) 2>> log/stderror.err & pid187=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 ) 2>> log/stderror.err & pid188=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P10 > work/kat/gul_S1_summarycalc_P10 ) 2>> log/stderror.err & pid189=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P10 > work/kat/gul_S1_pltcalc_P10 ) 2>> log/stderror.err & pid190=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P11 > work/kat/gul_S1_eltcalc_P11 ) 2>> log/stderror.err & pid191=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P11 > work/kat/gul_S1_summarycalc_P11 ) 2>> log/stderror.err & pid192=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 ) 2>> log/stderror.err & pid193=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P12 > work/kat/gul_S1_eltcalc_P12 ) 2>> log/stderror.err & pid194=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P12 > work/kat/gul_S1_summarycalc_P12 ) 2>> log/stderror.err & pid195=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P12 > work/kat/gul_S1_pltcalc_P12 ) 2>> log/stderror.err & pid196=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P13 > work/kat/gul_S1_eltcalc_P13 ) 2>> log/stderror.err & pid197=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P13 > work/kat/gul_S1_summarycalc_P13 ) 2>> log/stderror.err & pid198=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P13 > work/kat/gul_S1_pltcalc_P13 ) 2>> log/stderror.err & pid199=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P14 > work/kat/gul_S1_eltcalc_P14 ) 2>> log/stderror.err & pid200=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P14 > work/kat/gul_S1_summarycalc_P14 ) 2>> log/stderror.err & pid201=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P14 > work/kat/gul_S1_pltcalc_P14 ) 2>> log/stderror.err & pid202=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P15 > work/kat/gul_S1_eltcalc_P15 ) 2>> log/stderror.err & pid203=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P15 > work/kat/gul_S1_summarycalc_P15 ) 2>> log/stderror.err & pid204=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P15 > work/kat/gul_S1_pltcalc_P15 ) 2>> log/stderror.err & pid205=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P16 > work/kat/gul_S1_eltcalc_P16 ) 2>> log/stderror.err & pid206=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P16 > work/kat/gul_S1_summarycalc_P16 ) 2>> log/stderror.err & pid207=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P16 > work/kat/gul_S1_pltcalc_P16 ) 2>> log/stderror.err & pid208=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 ) 2>> log/stderror.err & pid209=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P17 > work/kat/gul_S1_summarycalc_P17 ) 2>> log/stderror.err & pid210=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 ) 2>> log/stderror.err & pid211=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P18 > work/kat/gul_S1_eltcalc_P18 ) 2>> log/stderror.err & pid212=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P18 > work/kat/gul_S1_summarycalc_P18 ) 2>> log/stderror.err & pid213=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P18 > work/kat/gul_S1_pltcalc_P18 ) 2>> log/stderror.err & pid214=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P19 > work/kat/gul_S1_eltcalc_P19 ) 2>> log/stderror.err & pid215=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 ) 2>> log/stderror.err & pid216=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P19 > work/kat/gul_S1_pltcalc_P19 ) 2>> log/stderror.err & pid217=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 ) 2>> log/stderror.err & pid218=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P20 > work/kat/gul_S1_summarycalc_P20 ) 2>> log/stderror.err & pid219=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P20 > work/kat/gul_S1_pltcalc_P20 ) 2>> log/stderror.err & pid220=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P21 > work/kat/gul_S1_eltcalc_P21 ) 2>> log/stderror.err & pid221=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P21 > work/kat/gul_S1_summarycalc_P21 ) 2>> log/stderror.err & pid222=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P21 > work/kat/gul_S1_pltcalc_P21 ) 2>> log/stderror.err & pid223=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P22 > work/kat/gul_S1_eltcalc_P22 ) 2>> log/stderror.err & pid224=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P22 > work/kat/gul_S1_summarycalc_P22 ) 2>> log/stderror.err & pid225=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P22 > work/kat/gul_S1_pltcalc_P22 ) 2>> log/stderror.err & pid226=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P23 > work/kat/gul_S1_eltcalc_P23 ) 2>> log/stderror.err & pid227=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P23 > work/kat/gul_S1_summarycalc_P23 ) 2>> log/stderror.err & pid228=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P23 > work/kat/gul_S1_pltcalc_P23 ) 2>> log/stderror.err & pid229=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P24 > work/kat/gul_S1_eltcalc_P24 ) 2>> log/stderror.err & pid230=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P24 > work/kat/gul_S1_summarycalc_P24 ) 2>> log/stderror.err & pid231=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P24 > work/kat/gul_S1_pltcalc_P24 ) 2>> log/stderror.err & pid232=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P25 > work/kat/gul_S1_eltcalc_P25 ) 2>> log/stderror.err & pid233=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P25 > work/kat/gul_S1_summarycalc_P25 ) 2>> log/stderror.err & pid234=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P25 > work/kat/gul_S1_pltcalc_P25 ) 2>> log/stderror.err & pid235=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P26 > work/kat/gul_S1_eltcalc_P26 ) 2>> log/stderror.err & pid236=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P26 > work/kat/gul_S1_summarycalc_P26 ) 2>> log/stderror.err & pid237=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P26 > work/kat/gul_S1_pltcalc_P26 ) 2>> log/stderror.err & pid238=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P27 > work/kat/gul_S1_eltcalc_P27 ) 2>> log/stderror.err & pid239=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P27 > work/kat/gul_S1_summarycalc_P27 ) 2>> log/stderror.err & pid240=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P27 > work/kat/gul_S1_pltcalc_P27 ) 2>> log/stderror.err & pid241=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P28 > work/kat/gul_S1_eltcalc_P28 ) 2>> log/stderror.err & pid242=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P28 > work/kat/gul_S1_summarycalc_P28 ) 2>> log/stderror.err & pid243=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P28 > work/kat/gul_S1_pltcalc_P28 ) 2>> log/stderror.err & pid244=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P29 > work/kat/gul_S1_eltcalc_P29 ) 2>> log/stderror.err & pid245=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P29 > work/kat/gul_S1_summarycalc_P29 ) 2>> log/stderror.err & pid246=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P29 > work/kat/gul_S1_pltcalc_P29 ) 2>> log/stderror.err & pid247=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P30 > work/kat/gul_S1_eltcalc_P30 ) 2>> log/stderror.err & pid248=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P30 > work/kat/gul_S1_summarycalc_P30 ) 2>> log/stderror.err & pid249=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P30 > work/kat/gul_S1_pltcalc_P30 ) 2>> log/stderror.err & pid250=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P31 > work/kat/gul_S1_eltcalc_P31 ) 2>> log/stderror.err & pid251=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P31 > work/kat/gul_S1_summarycalc_P31 ) 2>> log/stderror.err & pid252=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P31 > work/kat/gul_S1_pltcalc_P31 ) 2>> log/stderror.err & pid253=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P32 > work/kat/gul_S1_eltcalc_P32 ) 2>> log/stderror.err & pid254=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P32 > work/kat/gul_S1_summarycalc_P32 ) 2>> log/stderror.err & pid255=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P32 > work/kat/gul_S1_pltcalc_P32 ) 2>> log/stderror.err & pid256=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P33 > work/kat/gul_S1_eltcalc_P33 ) 2>> log/stderror.err & pid257=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P33 > work/kat/gul_S1_summarycalc_P33 ) 2>> log/stderror.err & pid258=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P33 > work/kat/gul_S1_pltcalc_P33 ) 2>> log/stderror.err & pid259=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P34 > work/kat/gul_S1_eltcalc_P34 ) 2>> log/stderror.err & pid260=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P34 > work/kat/gul_S1_summarycalc_P34 ) 2>> log/stderror.err & pid261=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P34 > work/kat/gul_S1_pltcalc_P34 ) 2>> log/stderror.err & pid262=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P35 > work/kat/gul_S1_eltcalc_P35 ) 2>> log/stderror.err & pid263=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P35 > work/kat/gul_S1_summarycalc_P35 ) 2>> log/stderror.err & pid264=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P35 > work/kat/gul_S1_pltcalc_P35 ) 2>> log/stderror.err & pid265=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P36 > work/kat/gul_S1_eltcalc_P36 ) 2>> log/stderror.err & pid266=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P36 > work/kat/gul_S1_summarycalc_P36 ) 2>> log/stderror.err & pid267=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P36 > work/kat/gul_S1_pltcalc_P36 ) 2>> log/stderror.err & pid268=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P37 > work/kat/gul_S1_eltcalc_P37 ) 2>> log/stderror.err & pid269=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P37 > work/kat/gul_S1_summarycalc_P37 ) 2>> log/stderror.err & pid270=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P37 > work/kat/gul_S1_pltcalc_P37 ) 2>> log/stderror.err & pid271=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P38 > work/kat/gul_S1_eltcalc_P38 ) 2>> log/stderror.err & pid272=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P38 > work/kat/gul_S1_summarycalc_P38 ) 2>> log/stderror.err & pid273=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P38 > work/kat/gul_S1_pltcalc_P38 ) 2>> log/stderror.err & pid274=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P39 > work/kat/gul_S1_eltcalc_P39 ) 2>> log/stderror.err & pid275=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P39 > work/kat/gul_S1_summarycalc_P39 ) 2>> log/stderror.err & pid276=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P39 > work/kat/gul_S1_pltcalc_P39 ) 2>> log/stderror.err & pid277=$!
( eltcalc -s < fifo/gul_S1_eltcalc_P40 > work/kat/gul_S1_eltcalc_P40 ) 2>> log/stderror.err & pid278=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P40 > work/kat/gul_S1_summarycalc_P40 ) 2>> log/stderror.err & pid279=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P40 > work/kat/gul_S1_pltcalc_P40 ) 2>> log/stderror.err & pid280=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid281=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid282=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_eltcalc_P3 fifo/gul_S1_summarycalc_P3 fifo/gul_S1_pltcalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid283=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_eltcalc_P4 fifo/gul_S1_summarycalc_P4 fifo/gul_S1_pltcalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid284=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_eltcalc_P5 fifo/gul_S1_summarycalc_P5 fifo/gul_S1_pltcalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid285=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_eltcalc_P6 fifo/gul_S1_summarycalc_P6 fifo/gul_S1_pltcalc_P6 work/gul_S1_summaryaalcalc/P6.bin work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid286=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_eltcalc_P7 fifo/gul_S1_summarycalc_P7 fifo/gul_S1_pltcalc_P7 work/gul_S1_summaryaalcalc/P7.bin work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid287=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_eltcalc_P8 fifo/gul_S1_summarycalc_P8 fifo/gul_S1_pltcalc_P8 work/gul_S1_summaryaalcalc/P8.bin work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid288=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid289=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 fifo/gul_S1_summarycalc_P10 fifo/gul_S1_pltcalc_P10 work/gul_S1_summaryaalcalc/P10.bin work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid290=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_eltcalc_P11 fifo/gul_S1_summarycalc_P11 fifo/gul_S1_pltcalc_P11 work/gul_S1_summaryaalcalc/P11.bin work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid291=$!
tee < fifo/gul_S1_summary_P12 fifo/gul_S1_eltcalc_P12 fifo/gul_S1_summarycalc_P12 fifo/gul_S1_pltcalc_P12 work/gul_S1_summaryaalcalc/P12.bin work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid292=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_eltcalc_P13 fifo/gul_S1_summarycalc_P13 fifo/gul_S1_pltcalc_P13 work/gul_S1_summaryaalcalc/P13.bin work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid293=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_eltcalc_P14 fifo/gul_S1_summarycalc_P14 fifo/gul_S1_pltcalc_P14 work/gul_S1_summaryaalcalc/P14.bin work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid294=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_eltcalc_P15 fifo/gul_S1_summarycalc_P15 fifo/gul_S1_pltcalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid295=$!
tee < fifo/gul_S1_summary_P16 fifo/gul_S1_eltcalc_P16 fifo/gul_S1_summarycalc_P16 fifo/gul_S1_pltcalc_P16 work/gul_S1_summaryaalcalc/P16.bin work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid296=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_eltcalc_P17 fifo/gul_S1_summarycalc_P17 fifo/gul_S1_pltcalc_P17 work/gul_S1_summaryaalcalc/P17.bin work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid297=$!
tee < fifo/gul_S1_summary_P18 fifo/gul_S1_eltcalc_P18 fifo/gul_S1_summarycalc_P18 fifo/gul_S1_pltcalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid298=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_eltcalc_P19 fifo/gul_S1_summarycalc_P19 fifo/gul_S1_pltcalc_P19 work/gul_S1_summaryaalcalc/P19.bin work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid299=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_eltcalc_P20 fifo/gul_S1_summarycalc_P20 fifo/gul_S1_pltcalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid300=$!
tee < fifo/gul_S1_summary_P21 fifo/gul_S1_eltcalc_P21 fifo/gul_S1_summarycalc_P21 fifo/gul_S1_pltcalc_P21 work/gul_S1_summaryaalcalc/P21.bin work/gul_S1_summaryleccalc/P21.bin > /dev/null & pid301=$!
tee < fifo/gul_S1_summary_P22 fifo/gul_S1_eltcalc_P22 fifo/gul_S1_summarycalc_P22 fifo/gul_S1_pltcalc_P22 work/gul_S1_summaryaalcalc/P22.bin work/gul_S1_summaryleccalc/P22.bin > /dev/null & pid302=$!
tee < fifo/gul_S1_summary_P23 fifo/gul_S1_eltcalc_P23 fifo/gul_S1_summarycalc_P23 fifo/gul_S1_pltcalc_P23 work/gul_S1_summaryaalcalc/P23.bin work/gul_S1_summaryleccalc/P23.bin > /dev/null & pid303=$!
tee < fifo/gul_S1_summary_P24 fifo/gul_S1_eltcalc_P24 fifo/gul_S1_summarycalc_P24 fifo/gul_S1_pltcalc_P24 work/gul_S1_summaryaalcalc/P24.bin work/gul_S1_summaryleccalc/P24.bin > /dev/null & pid304=$!
tee < fifo/gul_S1_summary_P25 fifo/gul_S1_eltcalc_P25 fifo/gul_S1_summarycalc_P25 fifo/gul_S1_pltcalc_P25 work/gul_S1_summaryaalcalc/P25.bin work/gul_S1_summaryleccalc/P25.bin > /dev/null & pid305=$!
tee < fifo/gul_S1_summary_P26 fifo/gul_S1_eltcalc_P26 fifo/gul_S1_summarycalc_P26 fifo/gul_S1_pltcalc_P26 work/gul_S1_summaryaalcalc/P26.bin work/gul_S1_summaryleccalc/P26.bin > /dev/null & pid306=$!
tee < fifo/gul_S1_summary_P27 fifo/gul_S1_eltcalc_P27 fifo/gul_S1_summarycalc_P27 fifo/gul_S1_pltcalc_P27 work/gul_S1_summaryaalcalc/P27.bin work/gul_S1_summaryleccalc/P27.bin > /dev/null & pid307=$!
tee < fifo/gul_S1_summary_P28 fifo/gul_S1_eltcalc_P28 fifo/gul_S1_summarycalc_P28 fifo/gul_S1_pltcalc_P28 work/gul_S1_summaryaalcalc/P28.bin work/gul_S1_summaryleccalc/P28.bin > /dev/null & pid308=$!
tee < fifo/gul_S1_summary_P29 fifo/gul_S1_eltcalc_P29 fifo/gul_S1_summarycalc_P29 fifo/gul_S1_pltcalc_P29 work/gul_S1_summaryaalcalc/P29.bin work/gul_S1_summaryleccalc/P29.bin > /dev/null & pid309=$!
tee < fifo/gul_S1_summary_P30 fifo/gul_S1_eltcalc_P30 fifo/gul_S1_summarycalc_P30 fifo/gul_S1_pltcalc_P30 work/gul_S1_summaryaalcalc/P30.bin work/gul_S1_summaryleccalc/P30.bin > /dev/null & pid310=$!
tee < fifo/gul_S1_summary_P31 fifo/gul_S1_eltcalc_P31 fifo/gul_S1_summarycalc_P31 fifo/gul_S1_pltcalc_P31 work/gul_S1_summaryaalcalc/P31.bin work/gul_S1_summaryleccalc/P31.bin > /dev/null & pid311=$!
tee < fifo/gul_S1_summary_P32 fifo/gul_S1_eltcalc_P32 fifo/gul_S1_summarycalc_P32 fifo/gul_S1_pltcalc_P32 work/gul_S1_summaryaalcalc/P32.bin work/gul_S1_summaryleccalc/P32.bin > /dev/null & pid312=$!
tee < fifo/gul_S1_summary_P33 fifo/gul_S1_eltcalc_P33 fifo/gul_S1_summarycalc_P33 fifo/gul_S1_pltcalc_P33 work/gul_S1_summaryaalcalc/P33.bin work/gul_S1_summaryleccalc/P33.bin > /dev/null & pid313=$!
tee < fifo/gul_S1_summary_P34 fifo/gul_S1_eltcalc_P34 fifo/gul_S1_summarycalc_P34 fifo/gul_S1_pltcalc_P34 work/gul_S1_summaryaalcalc/P34.bin work/gul_S1_summaryleccalc/P34.bin > /dev/null & pid314=$!
tee < fifo/gul_S1_summary_P35 fifo/gul_S1_eltcalc_P35 fifo/gul_S1_summarycalc_P35 fifo/gul_S1_pltcalc_P35 work/gul_S1_summaryaalcalc/P35.bin work/gul_S1_summaryleccalc/P35.bin > /dev/null & pid315=$!
tee < fifo/gul_S1_summary_P36 fifo/gul_S1_eltcalc_P36 fifo/gul_S1_summarycalc_P36 fifo/gul_S1_pltcalc_P36 work/gul_S1_summaryaalcalc/P36.bin work/gul_S1_summaryleccalc/P36.bin > /dev/null & pid316=$!
tee < fifo/gul_S1_summary_P37 fifo/gul_S1_eltcalc_P37 fifo/gul_S1_summarycalc_P37 fifo/gul_S1_pltcalc_P37 work/gul_S1_summaryaalcalc/P37.bin work/gul_S1_summaryleccalc/P37.bin > /dev/null & pid317=$!
tee < fifo/gul_S1_summary_P38 fifo/gul_S1_eltcalc_P38 fifo/gul_S1_summarycalc_P38 fifo/gul_S1_pltcalc_P38 work/gul_S1_summaryaalcalc/P38.bin work/gul_S1_summaryleccalc/P38.bin > /dev/null & pid318=$!
tee < fifo/gul_S1_summary_P39 fifo/gul_S1_eltcalc_P39 fifo/gul_S1_summarycalc_P39 fifo/gul_S1_pltcalc_P39 work/gul_S1_summaryaalcalc/P39.bin work/gul_S1_summaryleccalc/P39.bin > /dev/null & pid319=$!
tee < fifo/gul_S1_summary_P40 fifo/gul_S1_eltcalc_P40 fifo/gul_S1_summarycalc_P40 fifo/gul_S1_pltcalc_P40 work/gul_S1_summaryaalcalc/P40.bin work/gul_S1_summaryleccalc/P40.bin > /dev/null & pid320=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P12 < fifo/gul_P12 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P16 < fifo/gul_P16 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P18 < fifo/gul_P18 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P21 < fifo/gul_P21 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P22 < fifo/gul_P22 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P23 < fifo/gul_P23 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P24 < fifo/gul_P24 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P25 < fifo/gul_P25 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P26 < fifo/gul_P26 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P27 < fifo/gul_P27 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P28 < fifo/gul_P28 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P29 < fifo/gul_P29 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P30 < fifo/gul_P30 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P31 < fifo/gul_P31 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P32 < fifo/gul_P32 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P33 < fifo/gul_P33 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P34 < fifo/gul_P34 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P35 < fifo/gul_P35 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P36 < fifo/gul_P36 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P37 < fifo/gul_P37 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P38 < fifo/gul_P38 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P39 < fifo/gul_P39 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P40 < fifo/gul_P40 ) 2>> log/stderror.err  &

# --- Do insured loss computes ---

( eltcalc < fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 ) 2>> log/stderror.err & pid321=$!
( summarycalctocsv < fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 ) 2>> log/stderror.err & pid322=$!
( pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 ) 2>> log/stderror.err & pid323=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 ) 2>> log/stderror.err & pid324=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 ) 2>> log/stderror.err & pid325=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 ) 2>> log/stderror.err & pid326=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 ) 2>> log/stderror.err & pid327=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P3 > work/full_correlation/kat/il_S1_summarycalc_P3 ) 2>> log/stderror.err & pid328=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P3 > work/full_correlation/kat/il_S1_pltcalc_P3 ) 2>> log/stderror.err & pid329=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 ) 2>> log/stderror.err & pid330=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P4 > work/full_correlation/kat/il_S1_summarycalc_P4 ) 2>> log/stderror.err & pid331=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 ) 2>> log/stderror.err & pid332=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P5 > work/full_correlation/kat/il_S1_eltcalc_P5 ) 2>> log/stderror.err & pid333=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P5 > work/full_correlation/kat/il_S1_summarycalc_P5 ) 2>> log/stderror.err & pid334=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P5 > work/full_correlation/kat/il_S1_pltcalc_P5 ) 2>> log/stderror.err & pid335=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P6 > work/full_correlation/kat/il_S1_eltcalc_P6 ) 2>> log/stderror.err & pid336=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P6 > work/full_correlation/kat/il_S1_summarycalc_P6 ) 2>> log/stderror.err & pid337=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 ) 2>> log/stderror.err & pid338=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P7 > work/full_correlation/kat/il_S1_eltcalc_P7 ) 2>> log/stderror.err & pid339=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P7 > work/full_correlation/kat/il_S1_summarycalc_P7 ) 2>> log/stderror.err & pid340=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P7 > work/full_correlation/kat/il_S1_pltcalc_P7 ) 2>> log/stderror.err & pid341=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P8 > work/full_correlation/kat/il_S1_eltcalc_P8 ) 2>> log/stderror.err & pid342=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P8 > work/full_correlation/kat/il_S1_summarycalc_P8 ) 2>> log/stderror.err & pid343=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P8 > work/full_correlation/kat/il_S1_pltcalc_P8 ) 2>> log/stderror.err & pid344=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P9 > work/full_correlation/kat/il_S1_eltcalc_P9 ) 2>> log/stderror.err & pid345=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P9 > work/full_correlation/kat/il_S1_summarycalc_P9 ) 2>> log/stderror.err & pid346=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P9 > work/full_correlation/kat/il_S1_pltcalc_P9 ) 2>> log/stderror.err & pid347=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P10 > work/full_correlation/kat/il_S1_eltcalc_P10 ) 2>> log/stderror.err & pid348=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P10 > work/full_correlation/kat/il_S1_summarycalc_P10 ) 2>> log/stderror.err & pid349=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P10 > work/full_correlation/kat/il_S1_pltcalc_P10 ) 2>> log/stderror.err & pid350=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P11 > work/full_correlation/kat/il_S1_eltcalc_P11 ) 2>> log/stderror.err & pid351=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P11 > work/full_correlation/kat/il_S1_summarycalc_P11 ) 2>> log/stderror.err & pid352=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P11 > work/full_correlation/kat/il_S1_pltcalc_P11 ) 2>> log/stderror.err & pid353=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P12 > work/full_correlation/kat/il_S1_eltcalc_P12 ) 2>> log/stderror.err & pid354=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P12 > work/full_correlation/kat/il_S1_summarycalc_P12 ) 2>> log/stderror.err & pid355=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P12 > work/full_correlation/kat/il_S1_pltcalc_P12 ) 2>> log/stderror.err & pid356=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P13 > work/full_correlation/kat/il_S1_eltcalc_P13 ) 2>> log/stderror.err & pid357=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P13 > work/full_correlation/kat/il_S1_summarycalc_P13 ) 2>> log/stderror.err & pid358=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P13 > work/full_correlation/kat/il_S1_pltcalc_P13 ) 2>> log/stderror.err & pid359=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P14 > work/full_correlation/kat/il_S1_eltcalc_P14 ) 2>> log/stderror.err & pid360=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P14 > work/full_correlation/kat/il_S1_summarycalc_P14 ) 2>> log/stderror.err & pid361=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P14 > work/full_correlation/kat/il_S1_pltcalc_P14 ) 2>> log/stderror.err & pid362=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 ) 2>> log/stderror.err & pid363=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P15 > work/full_correlation/kat/il_S1_summarycalc_P15 ) 2>> log/stderror.err & pid364=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 ) 2>> log/stderror.err & pid365=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P16 > work/full_correlation/kat/il_S1_eltcalc_P16 ) 2>> log/stderror.err & pid366=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P16 > work/full_correlation/kat/il_S1_summarycalc_P16 ) 2>> log/stderror.err & pid367=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P16 > work/full_correlation/kat/il_S1_pltcalc_P16 ) 2>> log/stderror.err & pid368=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P17 > work/full_correlation/kat/il_S1_eltcalc_P17 ) 2>> log/stderror.err & pid369=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P17 > work/full_correlation/kat/il_S1_summarycalc_P17 ) 2>> log/stderror.err & pid370=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P17 > work/full_correlation/kat/il_S1_pltcalc_P17 ) 2>> log/stderror.err & pid371=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P18 > work/full_correlation/kat/il_S1_eltcalc_P18 ) 2>> log/stderror.err & pid372=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P18 > work/full_correlation/kat/il_S1_summarycalc_P18 ) 2>> log/stderror.err & pid373=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P18 > work/full_correlation/kat/il_S1_pltcalc_P18 ) 2>> log/stderror.err & pid374=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P19 > work/full_correlation/kat/il_S1_eltcalc_P19 ) 2>> log/stderror.err & pid375=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P19 > work/full_correlation/kat/il_S1_summarycalc_P19 ) 2>> log/stderror.err & pid376=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P19 > work/full_correlation/kat/il_S1_pltcalc_P19 ) 2>> log/stderror.err & pid377=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 ) 2>> log/stderror.err & pid378=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P20 > work/full_correlation/kat/il_S1_summarycalc_P20 ) 2>> log/stderror.err & pid379=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P20 > work/full_correlation/kat/il_S1_pltcalc_P20 ) 2>> log/stderror.err & pid380=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P21 > work/full_correlation/kat/il_S1_eltcalc_P21 ) 2>> log/stderror.err & pid381=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P21 > work/full_correlation/kat/il_S1_summarycalc_P21 ) 2>> log/stderror.err & pid382=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P21 > work/full_correlation/kat/il_S1_pltcalc_P21 ) 2>> log/stderror.err & pid383=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P22 > work/full_correlation/kat/il_S1_eltcalc_P22 ) 2>> log/stderror.err & pid384=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P22 > work/full_correlation/kat/il_S1_summarycalc_P22 ) 2>> log/stderror.err & pid385=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P22 > work/full_correlation/kat/il_S1_pltcalc_P22 ) 2>> log/stderror.err & pid386=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P23 > work/full_correlation/kat/il_S1_eltcalc_P23 ) 2>> log/stderror.err & pid387=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P23 > work/full_correlation/kat/il_S1_summarycalc_P23 ) 2>> log/stderror.err & pid388=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P23 > work/full_correlation/kat/il_S1_pltcalc_P23 ) 2>> log/stderror.err & pid389=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P24 > work/full_correlation/kat/il_S1_eltcalc_P24 ) 2>> log/stderror.err & pid390=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P24 > work/full_correlation/kat/il_S1_summarycalc_P24 ) 2>> log/stderror.err & pid391=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P24 > work/full_correlation/kat/il_S1_pltcalc_P24 ) 2>> log/stderror.err & pid392=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P25 > work/full_correlation/kat/il_S1_eltcalc_P25 ) 2>> log/stderror.err & pid393=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P25 > work/full_correlation/kat/il_S1_summarycalc_P25 ) 2>> log/stderror.err & pid394=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P25 > work/full_correlation/kat/il_S1_pltcalc_P25 ) 2>> log/stderror.err & pid395=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P26 > work/full_correlation/kat/il_S1_eltcalc_P26 ) 2>> log/stderror.err & pid396=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P26 > work/full_correlation/kat/il_S1_summarycalc_P26 ) 2>> log/stderror.err & pid397=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P26 > work/full_correlation/kat/il_S1_pltcalc_P26 ) 2>> log/stderror.err & pid398=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P27 > work/full_correlation/kat/il_S1_eltcalc_P27 ) 2>> log/stderror.err & pid399=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P27 > work/full_correlation/kat/il_S1_summarycalc_P27 ) 2>> log/stderror.err & pid400=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P27 > work/full_correlation/kat/il_S1_pltcalc_P27 ) 2>> log/stderror.err & pid401=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P28 > work/full_correlation/kat/il_S1_eltcalc_P28 ) 2>> log/stderror.err & pid402=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P28 > work/full_correlation/kat/il_S1_summarycalc_P28 ) 2>> log/stderror.err & pid403=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P28 > work/full_correlation/kat/il_S1_pltcalc_P28 ) 2>> log/stderror.err & pid404=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P29 > work/full_correlation/kat/il_S1_eltcalc_P29 ) 2>> log/stderror.err & pid405=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P29 > work/full_correlation/kat/il_S1_summarycalc_P29 ) 2>> log/stderror.err & pid406=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P29 > work/full_correlation/kat/il_S1_pltcalc_P29 ) 2>> log/stderror.err & pid407=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P30 > work/full_correlation/kat/il_S1_eltcalc_P30 ) 2>> log/stderror.err & pid408=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P30 > work/full_correlation/kat/il_S1_summarycalc_P30 ) 2>> log/stderror.err & pid409=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P30 > work/full_correlation/kat/il_S1_pltcalc_P30 ) 2>> log/stderror.err & pid410=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P31 > work/full_correlation/kat/il_S1_eltcalc_P31 ) 2>> log/stderror.err & pid411=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P31 > work/full_correlation/kat/il_S1_summarycalc_P31 ) 2>> log/stderror.err & pid412=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P31 > work/full_correlation/kat/il_S1_pltcalc_P31 ) 2>> log/stderror.err & pid413=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P32 > work/full_correlation/kat/il_S1_eltcalc_P32 ) 2>> log/stderror.err & pid414=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P32 > work/full_correlation/kat/il_S1_summarycalc_P32 ) 2>> log/stderror.err & pid415=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P32 > work/full_correlation/kat/il_S1_pltcalc_P32 ) 2>> log/stderror.err & pid416=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P33 > work/full_correlation/kat/il_S1_eltcalc_P33 ) 2>> log/stderror.err & pid417=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P33 > work/full_correlation/kat/il_S1_summarycalc_P33 ) 2>> log/stderror.err & pid418=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P33 > work/full_correlation/kat/il_S1_pltcalc_P33 ) 2>> log/stderror.err & pid419=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P34 > work/full_correlation/kat/il_S1_eltcalc_P34 ) 2>> log/stderror.err & pid420=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P34 > work/full_correlation/kat/il_S1_summarycalc_P34 ) 2>> log/stderror.err & pid421=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P34 > work/full_correlation/kat/il_S1_pltcalc_P34 ) 2>> log/stderror.err & pid422=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P35 > work/full_correlation/kat/il_S1_eltcalc_P35 ) 2>> log/stderror.err & pid423=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P35 > work/full_correlation/kat/il_S1_summarycalc_P35 ) 2>> log/stderror.err & pid424=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P35 > work/full_correlation/kat/il_S1_pltcalc_P35 ) 2>> log/stderror.err & pid425=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P36 > work/full_correlation/kat/il_S1_eltcalc_P36 ) 2>> log/stderror.err & pid426=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P36 > work/full_correlation/kat/il_S1_summarycalc_P36 ) 2>> log/stderror.err & pid427=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P36 > work/full_correlation/kat/il_S1_pltcalc_P36 ) 2>> log/stderror.err & pid428=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P37 > work/full_correlation/kat/il_S1_eltcalc_P37 ) 2>> log/stderror.err & pid429=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P37 > work/full_correlation/kat/il_S1_summarycalc_P37 ) 2>> log/stderror.err & pid430=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P37 > work/full_correlation/kat/il_S1_pltcalc_P37 ) 2>> log/stderror.err & pid431=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P38 > work/full_correlation/kat/il_S1_eltcalc_P38 ) 2>> log/stderror.err & pid432=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P38 > work/full_correlation/kat/il_S1_summarycalc_P38 ) 2>> log/stderror.err & pid433=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P38 > work/full_correlation/kat/il_S1_pltcalc_P38 ) 2>> log/stderror.err & pid434=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P39 > work/full_correlation/kat/il_S1_eltcalc_P39 ) 2>> log/stderror.err & pid435=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P39 > work/full_correlation/kat/il_S1_summarycalc_P39 ) 2>> log/stderror.err & pid436=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P39 > work/full_correlation/kat/il_S1_pltcalc_P39 ) 2>> log/stderror.err & pid437=$!
( eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P40 > work/full_correlation/kat/il_S1_eltcalc_P40 ) 2>> log/stderror.err & pid438=$!
( summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P40 > work/full_correlation/kat/il_S1_summarycalc_P40 ) 2>> log/stderror.err & pid439=$!
( pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P40 > work/full_correlation/kat/il_S1_pltcalc_P40 ) 2>> log/stderror.err & pid440=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_eltcalc_P1 fifo/full_correlation/il_S1_summarycalc_P1 fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid441=$!
tee < fifo/full_correlation/il_S1_summary_P2 fifo/full_correlation/il_S1_eltcalc_P2 fifo/full_correlation/il_S1_summarycalc_P2 fifo/full_correlation/il_S1_pltcalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin work/full_correlation/il_S1_summaryleccalc/P2.bin > /dev/null & pid442=$!
tee < fifo/full_correlation/il_S1_summary_P3 fifo/full_correlation/il_S1_eltcalc_P3 fifo/full_correlation/il_S1_summarycalc_P3 fifo/full_correlation/il_S1_pltcalc_P3 work/full_correlation/il_S1_summaryaalcalc/P3.bin work/full_correlation/il_S1_summaryleccalc/P3.bin > /dev/null & pid443=$!
tee < fifo/full_correlation/il_S1_summary_P4 fifo/full_correlation/il_S1_eltcalc_P4 fifo/full_correlation/il_S1_summarycalc_P4 fifo/full_correlation/il_S1_pltcalc_P4 work/full_correlation/il_S1_summaryaalcalc/P4.bin work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid444=$!
tee < fifo/full_correlation/il_S1_summary_P5 fifo/full_correlation/il_S1_eltcalc_P5 fifo/full_correlation/il_S1_summarycalc_P5 fifo/full_correlation/il_S1_pltcalc_P5 work/full_correlation/il_S1_summaryaalcalc/P5.bin work/full_correlation/il_S1_summaryleccalc/P5.bin > /dev/null & pid445=$!
tee < fifo/full_correlation/il_S1_summary_P6 fifo/full_correlation/il_S1_eltcalc_P6 fifo/full_correlation/il_S1_summarycalc_P6 fifo/full_correlation/il_S1_pltcalc_P6 work/full_correlation/il_S1_summaryaalcalc/P6.bin work/full_correlation/il_S1_summaryleccalc/P6.bin > /dev/null & pid446=$!
tee < fifo/full_correlation/il_S1_summary_P7 fifo/full_correlation/il_S1_eltcalc_P7 fifo/full_correlation/il_S1_summarycalc_P7 fifo/full_correlation/il_S1_pltcalc_P7 work/full_correlation/il_S1_summaryaalcalc/P7.bin work/full_correlation/il_S1_summaryleccalc/P7.bin > /dev/null & pid447=$!
tee < fifo/full_correlation/il_S1_summary_P8 fifo/full_correlation/il_S1_eltcalc_P8 fifo/full_correlation/il_S1_summarycalc_P8 fifo/full_correlation/il_S1_pltcalc_P8 work/full_correlation/il_S1_summaryaalcalc/P8.bin work/full_correlation/il_S1_summaryleccalc/P8.bin > /dev/null & pid448=$!
tee < fifo/full_correlation/il_S1_summary_P9 fifo/full_correlation/il_S1_eltcalc_P9 fifo/full_correlation/il_S1_summarycalc_P9 fifo/full_correlation/il_S1_pltcalc_P9 work/full_correlation/il_S1_summaryaalcalc/P9.bin work/full_correlation/il_S1_summaryleccalc/P9.bin > /dev/null & pid449=$!
tee < fifo/full_correlation/il_S1_summary_P10 fifo/full_correlation/il_S1_eltcalc_P10 fifo/full_correlation/il_S1_summarycalc_P10 fifo/full_correlation/il_S1_pltcalc_P10 work/full_correlation/il_S1_summaryaalcalc/P10.bin work/full_correlation/il_S1_summaryleccalc/P10.bin > /dev/null & pid450=$!
tee < fifo/full_correlation/il_S1_summary_P11 fifo/full_correlation/il_S1_eltcalc_P11 fifo/full_correlation/il_S1_summarycalc_P11 fifo/full_correlation/il_S1_pltcalc_P11 work/full_correlation/il_S1_summaryaalcalc/P11.bin work/full_correlation/il_S1_summaryleccalc/P11.bin > /dev/null & pid451=$!
tee < fifo/full_correlation/il_S1_summary_P12 fifo/full_correlation/il_S1_eltcalc_P12 fifo/full_correlation/il_S1_summarycalc_P12 fifo/full_correlation/il_S1_pltcalc_P12 work/full_correlation/il_S1_summaryaalcalc/P12.bin work/full_correlation/il_S1_summaryleccalc/P12.bin > /dev/null & pid452=$!
tee < fifo/full_correlation/il_S1_summary_P13 fifo/full_correlation/il_S1_eltcalc_P13 fifo/full_correlation/il_S1_summarycalc_P13 fifo/full_correlation/il_S1_pltcalc_P13 work/full_correlation/il_S1_summaryaalcalc/P13.bin work/full_correlation/il_S1_summaryleccalc/P13.bin > /dev/null & pid453=$!
tee < fifo/full_correlation/il_S1_summary_P14 fifo/full_correlation/il_S1_eltcalc_P14 fifo/full_correlation/il_S1_summarycalc_P14 fifo/full_correlation/il_S1_pltcalc_P14 work/full_correlation/il_S1_summaryaalcalc/P14.bin work/full_correlation/il_S1_summaryleccalc/P14.bin > /dev/null & pid454=$!
tee < fifo/full_correlation/il_S1_summary_P15 fifo/full_correlation/il_S1_eltcalc_P15 fifo/full_correlation/il_S1_summarycalc_P15 fifo/full_correlation/il_S1_pltcalc_P15 work/full_correlation/il_S1_summaryaalcalc/P15.bin work/full_correlation/il_S1_summaryleccalc/P15.bin > /dev/null & pid455=$!
tee < fifo/full_correlation/il_S1_summary_P16 fifo/full_correlation/il_S1_eltcalc_P16 fifo/full_correlation/il_S1_summarycalc_P16 fifo/full_correlation/il_S1_pltcalc_P16 work/full_correlation/il_S1_summaryaalcalc/P16.bin work/full_correlation/il_S1_summaryleccalc/P16.bin > /dev/null & pid456=$!
tee < fifo/full_correlation/il_S1_summary_P17 fifo/full_correlation/il_S1_eltcalc_P17 fifo/full_correlation/il_S1_summarycalc_P17 fifo/full_correlation/il_S1_pltcalc_P17 work/full_correlation/il_S1_summaryaalcalc/P17.bin work/full_correlation/il_S1_summaryleccalc/P17.bin > /dev/null & pid457=$!
tee < fifo/full_correlation/il_S1_summary_P18 fifo/full_correlation/il_S1_eltcalc_P18 fifo/full_correlation/il_S1_summarycalc_P18 fifo/full_correlation/il_S1_pltcalc_P18 work/full_correlation/il_S1_summaryaalcalc/P18.bin work/full_correlation/il_S1_summaryleccalc/P18.bin > /dev/null & pid458=$!
tee < fifo/full_correlation/il_S1_summary_P19 fifo/full_correlation/il_S1_eltcalc_P19 fifo/full_correlation/il_S1_summarycalc_P19 fifo/full_correlation/il_S1_pltcalc_P19 work/full_correlation/il_S1_summaryaalcalc/P19.bin work/full_correlation/il_S1_summaryleccalc/P19.bin > /dev/null & pid459=$!
tee < fifo/full_correlation/il_S1_summary_P20 fifo/full_correlation/il_S1_eltcalc_P20 fifo/full_correlation/il_S1_summarycalc_P20 fifo/full_correlation/il_S1_pltcalc_P20 work/full_correlation/il_S1_summaryaalcalc/P20.bin work/full_correlation/il_S1_summaryleccalc/P20.bin > /dev/null & pid460=$!
tee < fifo/full_correlation/il_S1_summary_P21 fifo/full_correlation/il_S1_eltcalc_P21 fifo/full_correlation/il_S1_summarycalc_P21 fifo/full_correlation/il_S1_pltcalc_P21 work/full_correlation/il_S1_summaryaalcalc/P21.bin work/full_correlation/il_S1_summaryleccalc/P21.bin > /dev/null & pid461=$!
tee < fifo/full_correlation/il_S1_summary_P22 fifo/full_correlation/il_S1_eltcalc_P22 fifo/full_correlation/il_S1_summarycalc_P22 fifo/full_correlation/il_S1_pltcalc_P22 work/full_correlation/il_S1_summaryaalcalc/P22.bin work/full_correlation/il_S1_summaryleccalc/P22.bin > /dev/null & pid462=$!
tee < fifo/full_correlation/il_S1_summary_P23 fifo/full_correlation/il_S1_eltcalc_P23 fifo/full_correlation/il_S1_summarycalc_P23 fifo/full_correlation/il_S1_pltcalc_P23 work/full_correlation/il_S1_summaryaalcalc/P23.bin work/full_correlation/il_S1_summaryleccalc/P23.bin > /dev/null & pid463=$!
tee < fifo/full_correlation/il_S1_summary_P24 fifo/full_correlation/il_S1_eltcalc_P24 fifo/full_correlation/il_S1_summarycalc_P24 fifo/full_correlation/il_S1_pltcalc_P24 work/full_correlation/il_S1_summaryaalcalc/P24.bin work/full_correlation/il_S1_summaryleccalc/P24.bin > /dev/null & pid464=$!
tee < fifo/full_correlation/il_S1_summary_P25 fifo/full_correlation/il_S1_eltcalc_P25 fifo/full_correlation/il_S1_summarycalc_P25 fifo/full_correlation/il_S1_pltcalc_P25 work/full_correlation/il_S1_summaryaalcalc/P25.bin work/full_correlation/il_S1_summaryleccalc/P25.bin > /dev/null & pid465=$!
tee < fifo/full_correlation/il_S1_summary_P26 fifo/full_correlation/il_S1_eltcalc_P26 fifo/full_correlation/il_S1_summarycalc_P26 fifo/full_correlation/il_S1_pltcalc_P26 work/full_correlation/il_S1_summaryaalcalc/P26.bin work/full_correlation/il_S1_summaryleccalc/P26.bin > /dev/null & pid466=$!
tee < fifo/full_correlation/il_S1_summary_P27 fifo/full_correlation/il_S1_eltcalc_P27 fifo/full_correlation/il_S1_summarycalc_P27 fifo/full_correlation/il_S1_pltcalc_P27 work/full_correlation/il_S1_summaryaalcalc/P27.bin work/full_correlation/il_S1_summaryleccalc/P27.bin > /dev/null & pid467=$!
tee < fifo/full_correlation/il_S1_summary_P28 fifo/full_correlation/il_S1_eltcalc_P28 fifo/full_correlation/il_S1_summarycalc_P28 fifo/full_correlation/il_S1_pltcalc_P28 work/full_correlation/il_S1_summaryaalcalc/P28.bin work/full_correlation/il_S1_summaryleccalc/P28.bin > /dev/null & pid468=$!
tee < fifo/full_correlation/il_S1_summary_P29 fifo/full_correlation/il_S1_eltcalc_P29 fifo/full_correlation/il_S1_summarycalc_P29 fifo/full_correlation/il_S1_pltcalc_P29 work/full_correlation/il_S1_summaryaalcalc/P29.bin work/full_correlation/il_S1_summaryleccalc/P29.bin > /dev/null & pid469=$!
tee < fifo/full_correlation/il_S1_summary_P30 fifo/full_correlation/il_S1_eltcalc_P30 fifo/full_correlation/il_S1_summarycalc_P30 fifo/full_correlation/il_S1_pltcalc_P30 work/full_correlation/il_S1_summaryaalcalc/P30.bin work/full_correlation/il_S1_summaryleccalc/P30.bin > /dev/null & pid470=$!
tee < fifo/full_correlation/il_S1_summary_P31 fifo/full_correlation/il_S1_eltcalc_P31 fifo/full_correlation/il_S1_summarycalc_P31 fifo/full_correlation/il_S1_pltcalc_P31 work/full_correlation/il_S1_summaryaalcalc/P31.bin work/full_correlation/il_S1_summaryleccalc/P31.bin > /dev/null & pid471=$!
tee < fifo/full_correlation/il_S1_summary_P32 fifo/full_correlation/il_S1_eltcalc_P32 fifo/full_correlation/il_S1_summarycalc_P32 fifo/full_correlation/il_S1_pltcalc_P32 work/full_correlation/il_S1_summaryaalcalc/P32.bin work/full_correlation/il_S1_summaryleccalc/P32.bin > /dev/null & pid472=$!
tee < fifo/full_correlation/il_S1_summary_P33 fifo/full_correlation/il_S1_eltcalc_P33 fifo/full_correlation/il_S1_summarycalc_P33 fifo/full_correlation/il_S1_pltcalc_P33 work/full_correlation/il_S1_summaryaalcalc/P33.bin work/full_correlation/il_S1_summaryleccalc/P33.bin > /dev/null & pid473=$!
tee < fifo/full_correlation/il_S1_summary_P34 fifo/full_correlation/il_S1_eltcalc_P34 fifo/full_correlation/il_S1_summarycalc_P34 fifo/full_correlation/il_S1_pltcalc_P34 work/full_correlation/il_S1_summaryaalcalc/P34.bin work/full_correlation/il_S1_summaryleccalc/P34.bin > /dev/null & pid474=$!
tee < fifo/full_correlation/il_S1_summary_P35 fifo/full_correlation/il_S1_eltcalc_P35 fifo/full_correlation/il_S1_summarycalc_P35 fifo/full_correlation/il_S1_pltcalc_P35 work/full_correlation/il_S1_summaryaalcalc/P35.bin work/full_correlation/il_S1_summaryleccalc/P35.bin > /dev/null & pid475=$!
tee < fifo/full_correlation/il_S1_summary_P36 fifo/full_correlation/il_S1_eltcalc_P36 fifo/full_correlation/il_S1_summarycalc_P36 fifo/full_correlation/il_S1_pltcalc_P36 work/full_correlation/il_S1_summaryaalcalc/P36.bin work/full_correlation/il_S1_summaryleccalc/P36.bin > /dev/null & pid476=$!
tee < fifo/full_correlation/il_S1_summary_P37 fifo/full_correlation/il_S1_eltcalc_P37 fifo/full_correlation/il_S1_summarycalc_P37 fifo/full_correlation/il_S1_pltcalc_P37 work/full_correlation/il_S1_summaryaalcalc/P37.bin work/full_correlation/il_S1_summaryleccalc/P37.bin > /dev/null & pid477=$!
tee < fifo/full_correlation/il_S1_summary_P38 fifo/full_correlation/il_S1_eltcalc_P38 fifo/full_correlation/il_S1_summarycalc_P38 fifo/full_correlation/il_S1_pltcalc_P38 work/full_correlation/il_S1_summaryaalcalc/P38.bin work/full_correlation/il_S1_summaryleccalc/P38.bin > /dev/null & pid478=$!
tee < fifo/full_correlation/il_S1_summary_P39 fifo/full_correlation/il_S1_eltcalc_P39 fifo/full_correlation/il_S1_summarycalc_P39 fifo/full_correlation/il_S1_pltcalc_P39 work/full_correlation/il_S1_summaryaalcalc/P39.bin work/full_correlation/il_S1_summaryleccalc/P39.bin > /dev/null & pid479=$!
tee < fifo/full_correlation/il_S1_summary_P40 fifo/full_correlation/il_S1_eltcalc_P40 fifo/full_correlation/il_S1_summarycalc_P40 fifo/full_correlation/il_S1_pltcalc_P40 work/full_correlation/il_S1_summaryaalcalc/P40.bin work/full_correlation/il_S1_summaryleccalc/P40.bin > /dev/null & pid480=$!

( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P2 < fifo/full_correlation/il_P2 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P3 < fifo/full_correlation/il_P3 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P4 < fifo/full_correlation/il_P4 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P5 < fifo/full_correlation/il_P5 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P6 < fifo/full_correlation/il_P6 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P7 < fifo/full_correlation/il_P7 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P8 < fifo/full_correlation/il_P8 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P9 < fifo/full_correlation/il_P9 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P10 < fifo/full_correlation/il_P10 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P11 < fifo/full_correlation/il_P11 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P12 < fifo/full_correlation/il_P12 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P13 < fifo/full_correlation/il_P13 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P14 < fifo/full_correlation/il_P14 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P15 < fifo/full_correlation/il_P15 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P17 < fifo/full_correlation/il_P17 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P18 < fifo/full_correlation/il_P18 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P19 < fifo/full_correlation/il_P19 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P20 < fifo/full_correlation/il_P20 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P21 < fifo/full_correlation/il_P21 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P22 < fifo/full_correlation/il_P22 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P23 < fifo/full_correlation/il_P23 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P24 < fifo/full_correlation/il_P24 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P25 < fifo/full_correlation/il_P25 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P26 < fifo/full_correlation/il_P26 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P27 < fifo/full_correlation/il_P27 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P28 < fifo/full_correlation/il_P28 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P29 < fifo/full_correlation/il_P29 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P30 < fifo/full_correlation/il_P30 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P31 < fifo/full_correlation/il_P31 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P32 < fifo/full_correlation/il_P32 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P33 < fifo/full_correlation/il_P33 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P34 < fifo/full_correlation/il_P34 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P35 < fifo/full_correlation/il_P35 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P36 < fifo/full_correlation/il_P36 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P37 < fifo/full_correlation/il_P37 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P38 < fifo/full_correlation/il_P38 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P39 < fifo/full_correlation/il_P39 ) 2>> log/stderror.err  &
( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P40 < fifo/full_correlation/il_P40 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---

( eltcalc < fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 ) 2>> log/stderror.err & pid481=$!
( summarycalctocsv < fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 ) 2>> log/stderror.err & pid482=$!
( pltcalc < fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 ) 2>> log/stderror.err & pid483=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 ) 2>> log/stderror.err & pid484=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 ) 2>> log/stderror.err & pid485=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 ) 2>> log/stderror.err & pid486=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P3 > work/full_correlation/kat/gul_S1_eltcalc_P3 ) 2>> log/stderror.err & pid487=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P3 > work/full_correlation/kat/gul_S1_summarycalc_P3 ) 2>> log/stderror.err & pid488=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P3 > work/full_correlation/kat/gul_S1_pltcalc_P3 ) 2>> log/stderror.err & pid489=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 ) 2>> log/stderror.err & pid490=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 ) 2>> log/stderror.err & pid491=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 ) 2>> log/stderror.err & pid492=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P5 > work/full_correlation/kat/gul_S1_eltcalc_P5 ) 2>> log/stderror.err & pid493=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P5 > work/full_correlation/kat/gul_S1_summarycalc_P5 ) 2>> log/stderror.err & pid494=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P5 > work/full_correlation/kat/gul_S1_pltcalc_P5 ) 2>> log/stderror.err & pid495=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P6 > work/full_correlation/kat/gul_S1_eltcalc_P6 ) 2>> log/stderror.err & pid496=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P6 > work/full_correlation/kat/gul_S1_summarycalc_P6 ) 2>> log/stderror.err & pid497=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P6 > work/full_correlation/kat/gul_S1_pltcalc_P6 ) 2>> log/stderror.err & pid498=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P7 > work/full_correlation/kat/gul_S1_eltcalc_P7 ) 2>> log/stderror.err & pid499=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P7 > work/full_correlation/kat/gul_S1_summarycalc_P7 ) 2>> log/stderror.err & pid500=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P7 > work/full_correlation/kat/gul_S1_pltcalc_P7 ) 2>> log/stderror.err & pid501=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P8 > work/full_correlation/kat/gul_S1_eltcalc_P8 ) 2>> log/stderror.err & pid502=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P8 > work/full_correlation/kat/gul_S1_summarycalc_P8 ) 2>> log/stderror.err & pid503=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P8 > work/full_correlation/kat/gul_S1_pltcalc_P8 ) 2>> log/stderror.err & pid504=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 ) 2>> log/stderror.err & pid505=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P9 > work/full_correlation/kat/gul_S1_summarycalc_P9 ) 2>> log/stderror.err & pid506=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P9 > work/full_correlation/kat/gul_S1_pltcalc_P9 ) 2>> log/stderror.err & pid507=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P10 > work/full_correlation/kat/gul_S1_eltcalc_P10 ) 2>> log/stderror.err & pid508=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 ) 2>> log/stderror.err & pid509=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P10 > work/full_correlation/kat/gul_S1_pltcalc_P10 ) 2>> log/stderror.err & pid510=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P11 > work/full_correlation/kat/gul_S1_eltcalc_P11 ) 2>> log/stderror.err & pid511=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P11 > work/full_correlation/kat/gul_S1_summarycalc_P11 ) 2>> log/stderror.err & pid512=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P11 > work/full_correlation/kat/gul_S1_pltcalc_P11 ) 2>> log/stderror.err & pid513=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P12 > work/full_correlation/kat/gul_S1_eltcalc_P12 ) 2>> log/stderror.err & pid514=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P12 > work/full_correlation/kat/gul_S1_summarycalc_P12 ) 2>> log/stderror.err & pid515=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P12 > work/full_correlation/kat/gul_S1_pltcalc_P12 ) 2>> log/stderror.err & pid516=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P13 > work/full_correlation/kat/gul_S1_eltcalc_P13 ) 2>> log/stderror.err & pid517=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P13 > work/full_correlation/kat/gul_S1_summarycalc_P13 ) 2>> log/stderror.err & pid518=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P13 > work/full_correlation/kat/gul_S1_pltcalc_P13 ) 2>> log/stderror.err & pid519=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P14 > work/full_correlation/kat/gul_S1_eltcalc_P14 ) 2>> log/stderror.err & pid520=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P14 > work/full_correlation/kat/gul_S1_summarycalc_P14 ) 2>> log/stderror.err & pid521=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P14 > work/full_correlation/kat/gul_S1_pltcalc_P14 ) 2>> log/stderror.err & pid522=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P15 > work/full_correlation/kat/gul_S1_eltcalc_P15 ) 2>> log/stderror.err & pid523=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P15 > work/full_correlation/kat/gul_S1_summarycalc_P15 ) 2>> log/stderror.err & pid524=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P15 > work/full_correlation/kat/gul_S1_pltcalc_P15 ) 2>> log/stderror.err & pid525=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P16 > work/full_correlation/kat/gul_S1_eltcalc_P16 ) 2>> log/stderror.err & pid526=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P16 > work/full_correlation/kat/gul_S1_summarycalc_P16 ) 2>> log/stderror.err & pid527=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P16 > work/full_correlation/kat/gul_S1_pltcalc_P16 ) 2>> log/stderror.err & pid528=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P17 > work/full_correlation/kat/gul_S1_eltcalc_P17 ) 2>> log/stderror.err & pid529=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P17 > work/full_correlation/kat/gul_S1_summarycalc_P17 ) 2>> log/stderror.err & pid530=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P17 > work/full_correlation/kat/gul_S1_pltcalc_P17 ) 2>> log/stderror.err & pid531=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P18 > work/full_correlation/kat/gul_S1_eltcalc_P18 ) 2>> log/stderror.err & pid532=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P18 > work/full_correlation/kat/gul_S1_summarycalc_P18 ) 2>> log/stderror.err & pid533=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P18 > work/full_correlation/kat/gul_S1_pltcalc_P18 ) 2>> log/stderror.err & pid534=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P19 > work/full_correlation/kat/gul_S1_eltcalc_P19 ) 2>> log/stderror.err & pid535=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P19 > work/full_correlation/kat/gul_S1_summarycalc_P19 ) 2>> log/stderror.err & pid536=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P19 > work/full_correlation/kat/gul_S1_pltcalc_P19 ) 2>> log/stderror.err & pid537=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P20 > work/full_correlation/kat/gul_S1_eltcalc_P20 ) 2>> log/stderror.err & pid538=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P20 > work/full_correlation/kat/gul_S1_summarycalc_P20 ) 2>> log/stderror.err & pid539=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P20 > work/full_correlation/kat/gul_S1_pltcalc_P20 ) 2>> log/stderror.err & pid540=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P21 > work/full_correlation/kat/gul_S1_eltcalc_P21 ) 2>> log/stderror.err & pid541=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P21 > work/full_correlation/kat/gul_S1_summarycalc_P21 ) 2>> log/stderror.err & pid542=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P21 > work/full_correlation/kat/gul_S1_pltcalc_P21 ) 2>> log/stderror.err & pid543=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P22 > work/full_correlation/kat/gul_S1_eltcalc_P22 ) 2>> log/stderror.err & pid544=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P22 > work/full_correlation/kat/gul_S1_summarycalc_P22 ) 2>> log/stderror.err & pid545=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P22 > work/full_correlation/kat/gul_S1_pltcalc_P22 ) 2>> log/stderror.err & pid546=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P23 > work/full_correlation/kat/gul_S1_eltcalc_P23 ) 2>> log/stderror.err & pid547=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P23 > work/full_correlation/kat/gul_S1_summarycalc_P23 ) 2>> log/stderror.err & pid548=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P23 > work/full_correlation/kat/gul_S1_pltcalc_P23 ) 2>> log/stderror.err & pid549=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P24 > work/full_correlation/kat/gul_S1_eltcalc_P24 ) 2>> log/stderror.err & pid550=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P24 > work/full_correlation/kat/gul_S1_summarycalc_P24 ) 2>> log/stderror.err & pid551=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P24 > work/full_correlation/kat/gul_S1_pltcalc_P24 ) 2>> log/stderror.err & pid552=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P25 > work/full_correlation/kat/gul_S1_eltcalc_P25 ) 2>> log/stderror.err & pid553=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P25 > work/full_correlation/kat/gul_S1_summarycalc_P25 ) 2>> log/stderror.err & pid554=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P25 > work/full_correlation/kat/gul_S1_pltcalc_P25 ) 2>> log/stderror.err & pid555=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P26 > work/full_correlation/kat/gul_S1_eltcalc_P26 ) 2>> log/stderror.err & pid556=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P26 > work/full_correlation/kat/gul_S1_summarycalc_P26 ) 2>> log/stderror.err & pid557=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P26 > work/full_correlation/kat/gul_S1_pltcalc_P26 ) 2>> log/stderror.err & pid558=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P27 > work/full_correlation/kat/gul_S1_eltcalc_P27 ) 2>> log/stderror.err & pid559=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P27 > work/full_correlation/kat/gul_S1_summarycalc_P27 ) 2>> log/stderror.err & pid560=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P27 > work/full_correlation/kat/gul_S1_pltcalc_P27 ) 2>> log/stderror.err & pid561=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P28 > work/full_correlation/kat/gul_S1_eltcalc_P28 ) 2>> log/stderror.err & pid562=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P28 > work/full_correlation/kat/gul_S1_summarycalc_P28 ) 2>> log/stderror.err & pid563=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P28 > work/full_correlation/kat/gul_S1_pltcalc_P28 ) 2>> log/stderror.err & pid564=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P29 > work/full_correlation/kat/gul_S1_eltcalc_P29 ) 2>> log/stderror.err & pid565=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P29 > work/full_correlation/kat/gul_S1_summarycalc_P29 ) 2>> log/stderror.err & pid566=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P29 > work/full_correlation/kat/gul_S1_pltcalc_P29 ) 2>> log/stderror.err & pid567=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P30 > work/full_correlation/kat/gul_S1_eltcalc_P30 ) 2>> log/stderror.err & pid568=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P30 > work/full_correlation/kat/gul_S1_summarycalc_P30 ) 2>> log/stderror.err & pid569=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P30 > work/full_correlation/kat/gul_S1_pltcalc_P30 ) 2>> log/stderror.err & pid570=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P31 > work/full_correlation/kat/gul_S1_eltcalc_P31 ) 2>> log/stderror.err & pid571=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P31 > work/full_correlation/kat/gul_S1_summarycalc_P31 ) 2>> log/stderror.err & pid572=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P31 > work/full_correlation/kat/gul_S1_pltcalc_P31 ) 2>> log/stderror.err & pid573=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P32 > work/full_correlation/kat/gul_S1_eltcalc_P32 ) 2>> log/stderror.err & pid574=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P32 > work/full_correlation/kat/gul_S1_summarycalc_P32 ) 2>> log/stderror.err & pid575=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P32 > work/full_correlation/kat/gul_S1_pltcalc_P32 ) 2>> log/stderror.err & pid576=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P33 > work/full_correlation/kat/gul_S1_eltcalc_P33 ) 2>> log/stderror.err & pid577=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P33 > work/full_correlation/kat/gul_S1_summarycalc_P33 ) 2>> log/stderror.err & pid578=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P33 > work/full_correlation/kat/gul_S1_pltcalc_P33 ) 2>> log/stderror.err & pid579=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P34 > work/full_correlation/kat/gul_S1_eltcalc_P34 ) 2>> log/stderror.err & pid580=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P34 > work/full_correlation/kat/gul_S1_summarycalc_P34 ) 2>> log/stderror.err & pid581=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P34 > work/full_correlation/kat/gul_S1_pltcalc_P34 ) 2>> log/stderror.err & pid582=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P35 > work/full_correlation/kat/gul_S1_eltcalc_P35 ) 2>> log/stderror.err & pid583=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P35 > work/full_correlation/kat/gul_S1_summarycalc_P35 ) 2>> log/stderror.err & pid584=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P35 > work/full_correlation/kat/gul_S1_pltcalc_P35 ) 2>> log/stderror.err & pid585=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P36 > work/full_correlation/kat/gul_S1_eltcalc_P36 ) 2>> log/stderror.err & pid586=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P36 > work/full_correlation/kat/gul_S1_summarycalc_P36 ) 2>> log/stderror.err & pid587=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P36 > work/full_correlation/kat/gul_S1_pltcalc_P36 ) 2>> log/stderror.err & pid588=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P37 > work/full_correlation/kat/gul_S1_eltcalc_P37 ) 2>> log/stderror.err & pid589=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P37 > work/full_correlation/kat/gul_S1_summarycalc_P37 ) 2>> log/stderror.err & pid590=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P37 > work/full_correlation/kat/gul_S1_pltcalc_P37 ) 2>> log/stderror.err & pid591=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P38 > work/full_correlation/kat/gul_S1_eltcalc_P38 ) 2>> log/stderror.err & pid592=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P38 > work/full_correlation/kat/gul_S1_summarycalc_P38 ) 2>> log/stderror.err & pid593=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P38 > work/full_correlation/kat/gul_S1_pltcalc_P38 ) 2>> log/stderror.err & pid594=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P39 > work/full_correlation/kat/gul_S1_eltcalc_P39 ) 2>> log/stderror.err & pid595=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P39 > work/full_correlation/kat/gul_S1_summarycalc_P39 ) 2>> log/stderror.err & pid596=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P39 > work/full_correlation/kat/gul_S1_pltcalc_P39 ) 2>> log/stderror.err & pid597=$!
( eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P40 > work/full_correlation/kat/gul_S1_eltcalc_P40 ) 2>> log/stderror.err & pid598=$!
( summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P40 > work/full_correlation/kat/gul_S1_summarycalc_P40 ) 2>> log/stderror.err & pid599=$!
( pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P40 > work/full_correlation/kat/gul_S1_pltcalc_P40 ) 2>> log/stderror.err & pid600=$!

tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_eltcalc_P1 fifo/full_correlation/gul_S1_summarycalc_P1 fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid601=$!
tee < fifo/full_correlation/gul_S1_summary_P2 fifo/full_correlation/gul_S1_eltcalc_P2 fifo/full_correlation/gul_S1_summarycalc_P2 fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid602=$!
tee < fifo/full_correlation/gul_S1_summary_P3 fifo/full_correlation/gul_S1_eltcalc_P3 fifo/full_correlation/gul_S1_summarycalc_P3 fifo/full_correlation/gul_S1_pltcalc_P3 work/full_correlation/gul_S1_summaryaalcalc/P3.bin work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid603=$!
tee < fifo/full_correlation/gul_S1_summary_P4 fifo/full_correlation/gul_S1_eltcalc_P4 fifo/full_correlation/gul_S1_summarycalc_P4 fifo/full_correlation/gul_S1_pltcalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid604=$!
tee < fifo/full_correlation/gul_S1_summary_P5 fifo/full_correlation/gul_S1_eltcalc_P5 fifo/full_correlation/gul_S1_summarycalc_P5 fifo/full_correlation/gul_S1_pltcalc_P5 work/full_correlation/gul_S1_summaryaalcalc/P5.bin work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid605=$!
tee < fifo/full_correlation/gul_S1_summary_P6 fifo/full_correlation/gul_S1_eltcalc_P6 fifo/full_correlation/gul_S1_summarycalc_P6 fifo/full_correlation/gul_S1_pltcalc_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid606=$!
tee < fifo/full_correlation/gul_S1_summary_P7 fifo/full_correlation/gul_S1_eltcalc_P7 fifo/full_correlation/gul_S1_summarycalc_P7 fifo/full_correlation/gul_S1_pltcalc_P7 work/full_correlation/gul_S1_summaryaalcalc/P7.bin work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid607=$!
tee < fifo/full_correlation/gul_S1_summary_P8 fifo/full_correlation/gul_S1_eltcalc_P8 fifo/full_correlation/gul_S1_summarycalc_P8 fifo/full_correlation/gul_S1_pltcalc_P8 work/full_correlation/gul_S1_summaryaalcalc/P8.bin work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid608=$!
tee < fifo/full_correlation/gul_S1_summary_P9 fifo/full_correlation/gul_S1_eltcalc_P9 fifo/full_correlation/gul_S1_summarycalc_P9 fifo/full_correlation/gul_S1_pltcalc_P9 work/full_correlation/gul_S1_summaryaalcalc/P9.bin work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid609=$!
tee < fifo/full_correlation/gul_S1_summary_P10 fifo/full_correlation/gul_S1_eltcalc_P10 fifo/full_correlation/gul_S1_summarycalc_P10 fifo/full_correlation/gul_S1_pltcalc_P10 work/full_correlation/gul_S1_summaryaalcalc/P10.bin work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid610=$!
tee < fifo/full_correlation/gul_S1_summary_P11 fifo/full_correlation/gul_S1_eltcalc_P11 fifo/full_correlation/gul_S1_summarycalc_P11 fifo/full_correlation/gul_S1_pltcalc_P11 work/full_correlation/gul_S1_summaryaalcalc/P11.bin work/full_correlation/gul_S1_summaryleccalc/P11.bin > /dev/null & pid611=$!
tee < fifo/full_correlation/gul_S1_summary_P12 fifo/full_correlation/gul_S1_eltcalc_P12 fifo/full_correlation/gul_S1_summarycalc_P12 fifo/full_correlation/gul_S1_pltcalc_P12 work/full_correlation/gul_S1_summaryaalcalc/P12.bin work/full_correlation/gul_S1_summaryleccalc/P12.bin > /dev/null & pid612=$!
tee < fifo/full_correlation/gul_S1_summary_P13 fifo/full_correlation/gul_S1_eltcalc_P13 fifo/full_correlation/gul_S1_summarycalc_P13 fifo/full_correlation/gul_S1_pltcalc_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin work/full_correlation/gul_S1_summaryleccalc/P13.bin > /dev/null & pid613=$!
tee < fifo/full_correlation/gul_S1_summary_P14 fifo/full_correlation/gul_S1_eltcalc_P14 fifo/full_correlation/gul_S1_summarycalc_P14 fifo/full_correlation/gul_S1_pltcalc_P14 work/full_correlation/gul_S1_summaryaalcalc/P14.bin work/full_correlation/gul_S1_summaryleccalc/P14.bin > /dev/null & pid614=$!
tee < fifo/full_correlation/gul_S1_summary_P15 fifo/full_correlation/gul_S1_eltcalc_P15 fifo/full_correlation/gul_S1_summarycalc_P15 fifo/full_correlation/gul_S1_pltcalc_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin work/full_correlation/gul_S1_summaryleccalc/P15.bin > /dev/null & pid615=$!
tee < fifo/full_correlation/gul_S1_summary_P16 fifo/full_correlation/gul_S1_eltcalc_P16 fifo/full_correlation/gul_S1_summarycalc_P16 fifo/full_correlation/gul_S1_pltcalc_P16 work/full_correlation/gul_S1_summaryaalcalc/P16.bin work/full_correlation/gul_S1_summaryleccalc/P16.bin > /dev/null & pid616=$!
tee < fifo/full_correlation/gul_S1_summary_P17 fifo/full_correlation/gul_S1_eltcalc_P17 fifo/full_correlation/gul_S1_summarycalc_P17 fifo/full_correlation/gul_S1_pltcalc_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin work/full_correlation/gul_S1_summaryleccalc/P17.bin > /dev/null & pid617=$!
tee < fifo/full_correlation/gul_S1_summary_P18 fifo/full_correlation/gul_S1_eltcalc_P18 fifo/full_correlation/gul_S1_summarycalc_P18 fifo/full_correlation/gul_S1_pltcalc_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin work/full_correlation/gul_S1_summaryleccalc/P18.bin > /dev/null & pid618=$!
tee < fifo/full_correlation/gul_S1_summary_P19 fifo/full_correlation/gul_S1_eltcalc_P19 fifo/full_correlation/gul_S1_summarycalc_P19 fifo/full_correlation/gul_S1_pltcalc_P19 work/full_correlation/gul_S1_summaryaalcalc/P19.bin work/full_correlation/gul_S1_summaryleccalc/P19.bin > /dev/null & pid619=$!
tee < fifo/full_correlation/gul_S1_summary_P20 fifo/full_correlation/gul_S1_eltcalc_P20 fifo/full_correlation/gul_S1_summarycalc_P20 fifo/full_correlation/gul_S1_pltcalc_P20 work/full_correlation/gul_S1_summaryaalcalc/P20.bin work/full_correlation/gul_S1_summaryleccalc/P20.bin > /dev/null & pid620=$!
tee < fifo/full_correlation/gul_S1_summary_P21 fifo/full_correlation/gul_S1_eltcalc_P21 fifo/full_correlation/gul_S1_summarycalc_P21 fifo/full_correlation/gul_S1_pltcalc_P21 work/full_correlation/gul_S1_summaryaalcalc/P21.bin work/full_correlation/gul_S1_summaryleccalc/P21.bin > /dev/null & pid621=$!
tee < fifo/full_correlation/gul_S1_summary_P22 fifo/full_correlation/gul_S1_eltcalc_P22 fifo/full_correlation/gul_S1_summarycalc_P22 fifo/full_correlation/gul_S1_pltcalc_P22 work/full_correlation/gul_S1_summaryaalcalc/P22.bin work/full_correlation/gul_S1_summaryleccalc/P22.bin > /dev/null & pid622=$!
tee < fifo/full_correlation/gul_S1_summary_P23 fifo/full_correlation/gul_S1_eltcalc_P23 fifo/full_correlation/gul_S1_summarycalc_P23 fifo/full_correlation/gul_S1_pltcalc_P23 work/full_correlation/gul_S1_summaryaalcalc/P23.bin work/full_correlation/gul_S1_summaryleccalc/P23.bin > /dev/null & pid623=$!
tee < fifo/full_correlation/gul_S1_summary_P24 fifo/full_correlation/gul_S1_eltcalc_P24 fifo/full_correlation/gul_S1_summarycalc_P24 fifo/full_correlation/gul_S1_pltcalc_P24 work/full_correlation/gul_S1_summaryaalcalc/P24.bin work/full_correlation/gul_S1_summaryleccalc/P24.bin > /dev/null & pid624=$!
tee < fifo/full_correlation/gul_S1_summary_P25 fifo/full_correlation/gul_S1_eltcalc_P25 fifo/full_correlation/gul_S1_summarycalc_P25 fifo/full_correlation/gul_S1_pltcalc_P25 work/full_correlation/gul_S1_summaryaalcalc/P25.bin work/full_correlation/gul_S1_summaryleccalc/P25.bin > /dev/null & pid625=$!
tee < fifo/full_correlation/gul_S1_summary_P26 fifo/full_correlation/gul_S1_eltcalc_P26 fifo/full_correlation/gul_S1_summarycalc_P26 fifo/full_correlation/gul_S1_pltcalc_P26 work/full_correlation/gul_S1_summaryaalcalc/P26.bin work/full_correlation/gul_S1_summaryleccalc/P26.bin > /dev/null & pid626=$!
tee < fifo/full_correlation/gul_S1_summary_P27 fifo/full_correlation/gul_S1_eltcalc_P27 fifo/full_correlation/gul_S1_summarycalc_P27 fifo/full_correlation/gul_S1_pltcalc_P27 work/full_correlation/gul_S1_summaryaalcalc/P27.bin work/full_correlation/gul_S1_summaryleccalc/P27.bin > /dev/null & pid627=$!
tee < fifo/full_correlation/gul_S1_summary_P28 fifo/full_correlation/gul_S1_eltcalc_P28 fifo/full_correlation/gul_S1_summarycalc_P28 fifo/full_correlation/gul_S1_pltcalc_P28 work/full_correlation/gul_S1_summaryaalcalc/P28.bin work/full_correlation/gul_S1_summaryleccalc/P28.bin > /dev/null & pid628=$!
tee < fifo/full_correlation/gul_S1_summary_P29 fifo/full_correlation/gul_S1_eltcalc_P29 fifo/full_correlation/gul_S1_summarycalc_P29 fifo/full_correlation/gul_S1_pltcalc_P29 work/full_correlation/gul_S1_summaryaalcalc/P29.bin work/full_correlation/gul_S1_summaryleccalc/P29.bin > /dev/null & pid629=$!
tee < fifo/full_correlation/gul_S1_summary_P30 fifo/full_correlation/gul_S1_eltcalc_P30 fifo/full_correlation/gul_S1_summarycalc_P30 fifo/full_correlation/gul_S1_pltcalc_P30 work/full_correlation/gul_S1_summaryaalcalc/P30.bin work/full_correlation/gul_S1_summaryleccalc/P30.bin > /dev/null & pid630=$!
tee < fifo/full_correlation/gul_S1_summary_P31 fifo/full_correlation/gul_S1_eltcalc_P31 fifo/full_correlation/gul_S1_summarycalc_P31 fifo/full_correlation/gul_S1_pltcalc_P31 work/full_correlation/gul_S1_summaryaalcalc/P31.bin work/full_correlation/gul_S1_summaryleccalc/P31.bin > /dev/null & pid631=$!
tee < fifo/full_correlation/gul_S1_summary_P32 fifo/full_correlation/gul_S1_eltcalc_P32 fifo/full_correlation/gul_S1_summarycalc_P32 fifo/full_correlation/gul_S1_pltcalc_P32 work/full_correlation/gul_S1_summaryaalcalc/P32.bin work/full_correlation/gul_S1_summaryleccalc/P32.bin > /dev/null & pid632=$!
tee < fifo/full_correlation/gul_S1_summary_P33 fifo/full_correlation/gul_S1_eltcalc_P33 fifo/full_correlation/gul_S1_summarycalc_P33 fifo/full_correlation/gul_S1_pltcalc_P33 work/full_correlation/gul_S1_summaryaalcalc/P33.bin work/full_correlation/gul_S1_summaryleccalc/P33.bin > /dev/null & pid633=$!
tee < fifo/full_correlation/gul_S1_summary_P34 fifo/full_correlation/gul_S1_eltcalc_P34 fifo/full_correlation/gul_S1_summarycalc_P34 fifo/full_correlation/gul_S1_pltcalc_P34 work/full_correlation/gul_S1_summaryaalcalc/P34.bin work/full_correlation/gul_S1_summaryleccalc/P34.bin > /dev/null & pid634=$!
tee < fifo/full_correlation/gul_S1_summary_P35 fifo/full_correlation/gul_S1_eltcalc_P35 fifo/full_correlation/gul_S1_summarycalc_P35 fifo/full_correlation/gul_S1_pltcalc_P35 work/full_correlation/gul_S1_summaryaalcalc/P35.bin work/full_correlation/gul_S1_summaryleccalc/P35.bin > /dev/null & pid635=$!
tee < fifo/full_correlation/gul_S1_summary_P36 fifo/full_correlation/gul_S1_eltcalc_P36 fifo/full_correlation/gul_S1_summarycalc_P36 fifo/full_correlation/gul_S1_pltcalc_P36 work/full_correlation/gul_S1_summaryaalcalc/P36.bin work/full_correlation/gul_S1_summaryleccalc/P36.bin > /dev/null & pid636=$!
tee < fifo/full_correlation/gul_S1_summary_P37 fifo/full_correlation/gul_S1_eltcalc_P37 fifo/full_correlation/gul_S1_summarycalc_P37 fifo/full_correlation/gul_S1_pltcalc_P37 work/full_correlation/gul_S1_summaryaalcalc/P37.bin work/full_correlation/gul_S1_summaryleccalc/P37.bin > /dev/null & pid637=$!
tee < fifo/full_correlation/gul_S1_summary_P38 fifo/full_correlation/gul_S1_eltcalc_P38 fifo/full_correlation/gul_S1_summarycalc_P38 fifo/full_correlation/gul_S1_pltcalc_P38 work/full_correlation/gul_S1_summaryaalcalc/P38.bin work/full_correlation/gul_S1_summaryleccalc/P38.bin > /dev/null & pid638=$!
tee < fifo/full_correlation/gul_S1_summary_P39 fifo/full_correlation/gul_S1_eltcalc_P39 fifo/full_correlation/gul_S1_summarycalc_P39 fifo/full_correlation/gul_S1_pltcalc_P39 work/full_correlation/gul_S1_summaryaalcalc/P39.bin work/full_correlation/gul_S1_summaryleccalc/P39.bin > /dev/null & pid639=$!
tee < fifo/full_correlation/gul_S1_summary_P40 fifo/full_correlation/gul_S1_eltcalc_P40 fifo/full_correlation/gul_S1_summarycalc_P40 fifo/full_correlation/gul_S1_pltcalc_P40 work/full_correlation/gul_S1_summaryaalcalc/P40.bin work/full_correlation/gul_S1_summaryleccalc/P40.bin > /dev/null & pid640=$!

( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P2 < fifo/full_correlation/gul_P2 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P3 < fifo/full_correlation/gul_P3 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P4 < fifo/full_correlation/gul_P4 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P5 < fifo/full_correlation/gul_P5 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P6 < fifo/full_correlation/gul_P6 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P7 < fifo/full_correlation/gul_P7 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P8 < fifo/full_correlation/gul_P8 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P9 < fifo/full_correlation/gul_P9 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P10 < fifo/full_correlation/gul_P10 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P11 < fifo/full_correlation/gul_P11 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P12 < fifo/full_correlation/gul_P12 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P13 < fifo/full_correlation/gul_P13 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P14 < fifo/full_correlation/gul_P14 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P15 < fifo/full_correlation/gul_P15 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P16 < fifo/full_correlation/gul_P16 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P17 < fifo/full_correlation/gul_P17 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P18 < fifo/full_correlation/gul_P18 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P19 < fifo/full_correlation/gul_P19 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P20 < fifo/full_correlation/gul_P20 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P21 < fifo/full_correlation/gul_P21 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P22 < fifo/full_correlation/gul_P22 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P23 < fifo/full_correlation/gul_P23 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P24 < fifo/full_correlation/gul_P24 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P25 < fifo/full_correlation/gul_P25 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P26 < fifo/full_correlation/gul_P26 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P27 < fifo/full_correlation/gul_P27 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P28 < fifo/full_correlation/gul_P28 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P29 < fifo/full_correlation/gul_P29 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P30 < fifo/full_correlation/gul_P30 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P31 < fifo/full_correlation/gul_P31 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P32 < fifo/full_correlation/gul_P32 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P33 < fifo/full_correlation/gul_P33 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P34 < fifo/full_correlation/gul_P34 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P35 < fifo/full_correlation/gul_P35 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P36 < fifo/full_correlation/gul_P36 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P37 < fifo/full_correlation/gul_P37 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P38 < fifo/full_correlation/gul_P38 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P39 < fifo/full_correlation/gul_P39 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P40 < fifo/full_correlation/gul_P40 ) 2>> log/stderror.err  &

( tee < fifo/full_correlation/gul_fc_P1 fifo/full_correlation/gul_P1  | fmcalc -a2 > fifo/full_correlation/il_P1  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P2 fifo/full_correlation/gul_P2  | fmcalc -a2 > fifo/full_correlation/il_P2  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P3 fifo/full_correlation/gul_P3  | fmcalc -a2 > fifo/full_correlation/il_P3  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P4 fifo/full_correlation/gul_P4  | fmcalc -a2 > fifo/full_correlation/il_P4  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P5 fifo/full_correlation/gul_P5  | fmcalc -a2 > fifo/full_correlation/il_P5  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P6 fifo/full_correlation/gul_P6  | fmcalc -a2 > fifo/full_correlation/il_P6  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P7 fifo/full_correlation/gul_P7  | fmcalc -a2 > fifo/full_correlation/il_P7  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P8 fifo/full_correlation/gul_P8  | fmcalc -a2 > fifo/full_correlation/il_P8  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P9 fifo/full_correlation/gul_P9  | fmcalc -a2 > fifo/full_correlation/il_P9  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P10 fifo/full_correlation/gul_P10  | fmcalc -a2 > fifo/full_correlation/il_P10  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P11 fifo/full_correlation/gul_P11  | fmcalc -a2 > fifo/full_correlation/il_P11  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P12 fifo/full_correlation/gul_P12  | fmcalc -a2 > fifo/full_correlation/il_P12  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P13 fifo/full_correlation/gul_P13  | fmcalc -a2 > fifo/full_correlation/il_P13  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P14 fifo/full_correlation/gul_P14  | fmcalc -a2 > fifo/full_correlation/il_P14  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P15 fifo/full_correlation/gul_P15  | fmcalc -a2 > fifo/full_correlation/il_P15  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P16 fifo/full_correlation/gul_P16  | fmcalc -a2 > fifo/full_correlation/il_P16  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P17 fifo/full_correlation/gul_P17  | fmcalc -a2 > fifo/full_correlation/il_P17  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P18 fifo/full_correlation/gul_P18  | fmcalc -a2 > fifo/full_correlation/il_P18  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P19 fifo/full_correlation/gul_P19  | fmcalc -a2 > fifo/full_correlation/il_P19  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P20 fifo/full_correlation/gul_P20  | fmcalc -a2 > fifo/full_correlation/il_P20  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P21 fifo/full_correlation/gul_P21  | fmcalc -a2 > fifo/full_correlation/il_P21  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P22 fifo/full_correlation/gul_P22  | fmcalc -a2 > fifo/full_correlation/il_P22  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P23 fifo/full_correlation/gul_P23  | fmcalc -a2 > fifo/full_correlation/il_P23  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P24 fifo/full_correlation/gul_P24  | fmcalc -a2 > fifo/full_correlation/il_P24  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P25 fifo/full_correlation/gul_P25  | fmcalc -a2 > fifo/full_correlation/il_P25  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P26 fifo/full_correlation/gul_P26  | fmcalc -a2 > fifo/full_correlation/il_P26  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P27 fifo/full_correlation/gul_P27  | fmcalc -a2 > fifo/full_correlation/il_P27  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P28 fifo/full_correlation/gul_P28  | fmcalc -a2 > fifo/full_correlation/il_P28  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P29 fifo/full_correlation/gul_P29  | fmcalc -a2 > fifo/full_correlation/il_P29  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P30 fifo/full_correlation/gul_P30  | fmcalc -a2 > fifo/full_correlation/il_P30  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P31 fifo/full_correlation/gul_P31  | fmcalc -a2 > fifo/full_correlation/il_P31  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P32 fifo/full_correlation/gul_P32  | fmcalc -a2 > fifo/full_correlation/il_P32  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P33 fifo/full_correlation/gul_P33  | fmcalc -a2 > fifo/full_correlation/il_P33  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P34 fifo/full_correlation/gul_P34  | fmcalc -a2 > fifo/full_correlation/il_P34  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P35 fifo/full_correlation/gul_P35  | fmcalc -a2 > fifo/full_correlation/il_P35  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P36 fifo/full_correlation/gul_P36  | fmcalc -a2 > fifo/full_correlation/il_P36  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P37 fifo/full_correlation/gul_P37  | fmcalc -a2 > fifo/full_correlation/il_P37  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P38 fifo/full_correlation/gul_P38  | fmcalc -a2 > fifo/full_correlation/il_P38  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P39 fifo/full_correlation/gul_P39  | fmcalc -a2 > fifo/full_correlation/il_P39  ) 2>> log/stderror.err &
( tee < fifo/full_correlation/gul_fc_P40 fifo/full_correlation/gul_P40  | fmcalc -a2 > fifo/full_correlation/il_P40  ) 2>> log/stderror.err &
( eve 1 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  ) 2>> log/stderror.err &
( eve 2 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P2 -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  ) 2>> log/stderror.err &
( eve 3 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P3 -a1 -i - | tee fifo/gul_P3 | fmcalc -a2 > fifo/il_P3  ) 2>> log/stderror.err &
( eve 4 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P4 -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  ) 2>> log/stderror.err &
( eve 5 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P5 -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  ) 2>> log/stderror.err &
( eve 6 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P6 -a1 -i - | tee fifo/gul_P6 | fmcalc -a2 > fifo/il_P6  ) 2>> log/stderror.err &
( eve 7 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P7 -a1 -i - | tee fifo/gul_P7 | fmcalc -a2 > fifo/il_P7  ) 2>> log/stderror.err &
( eve 8 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P8 -a1 -i - | tee fifo/gul_P8 | fmcalc -a2 > fifo/il_P8  ) 2>> log/stderror.err &
( eve 9 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P9 -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  ) 2>> log/stderror.err &
( eve 10 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P10 -a1 -i - | tee fifo/gul_P10 | fmcalc -a2 > fifo/il_P10  ) 2>> log/stderror.err &
( eve 11 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P11 -a1 -i - | tee fifo/gul_P11 | fmcalc -a2 > fifo/il_P11  ) 2>> log/stderror.err &
( eve 12 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P12 -a1 -i - | tee fifo/gul_P12 | fmcalc -a2 > fifo/il_P12  ) 2>> log/stderror.err &
( eve 13 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P13 -a1 -i - | tee fifo/gul_P13 | fmcalc -a2 > fifo/il_P13  ) 2>> log/stderror.err &
( eve 14 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P14 -a1 -i - | tee fifo/gul_P14 | fmcalc -a2 > fifo/il_P14  ) 2>> log/stderror.err &
( eve 15 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P15 -a1 -i - | tee fifo/gul_P15 | fmcalc -a2 > fifo/il_P15  ) 2>> log/stderror.err &
( eve 16 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P16 -a1 -i - | tee fifo/gul_P16 | fmcalc -a2 > fifo/il_P16  ) 2>> log/stderror.err &
( eve 17 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P17 -a1 -i - | tee fifo/gul_P17 | fmcalc -a2 > fifo/il_P17  ) 2>> log/stderror.err &
( eve 18 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P18 -a1 -i - | tee fifo/gul_P18 | fmcalc -a2 > fifo/il_P18  ) 2>> log/stderror.err &
( eve 19 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P19 -a1 -i - | tee fifo/gul_P19 | fmcalc -a2 > fifo/il_P19  ) 2>> log/stderror.err &
( eve 20 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P20 -a1 -i - | tee fifo/gul_P20 | fmcalc -a2 > fifo/il_P20  ) 2>> log/stderror.err &
( eve 21 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P21 -a1 -i - | tee fifo/gul_P21 | fmcalc -a2 > fifo/il_P21  ) 2>> log/stderror.err &
( eve 22 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P22 -a1 -i - | tee fifo/gul_P22 | fmcalc -a2 > fifo/il_P22  ) 2>> log/stderror.err &
( eve 23 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P23 -a1 -i - | tee fifo/gul_P23 | fmcalc -a2 > fifo/il_P23  ) 2>> log/stderror.err &
( eve 24 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P24 -a1 -i - | tee fifo/gul_P24 | fmcalc -a2 > fifo/il_P24  ) 2>> log/stderror.err &
( eve 25 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P25 -a1 -i - | tee fifo/gul_P25 | fmcalc -a2 > fifo/il_P25  ) 2>> log/stderror.err &
( eve 26 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P26 -a1 -i - | tee fifo/gul_P26 | fmcalc -a2 > fifo/il_P26  ) 2>> log/stderror.err &
( eve 27 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P27 -a1 -i - | tee fifo/gul_P27 | fmcalc -a2 > fifo/il_P27  ) 2>> log/stderror.err &
( eve 28 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P28 -a1 -i - | tee fifo/gul_P28 | fmcalc -a2 > fifo/il_P28  ) 2>> log/stderror.err &
( eve 29 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P29 -a1 -i - | tee fifo/gul_P29 | fmcalc -a2 > fifo/il_P29  ) 2>> log/stderror.err &
( eve 30 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P30 -a1 -i - | tee fifo/gul_P30 | fmcalc -a2 > fifo/il_P30  ) 2>> log/stderror.err &
( eve 31 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P31 -a1 -i - | tee fifo/gul_P31 | fmcalc -a2 > fifo/il_P31  ) 2>> log/stderror.err &
( eve 32 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P32 -a1 -i - | tee fifo/gul_P32 | fmcalc -a2 > fifo/il_P32  ) 2>> log/stderror.err &
( eve 33 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P33 -a1 -i - | tee fifo/gul_P33 | fmcalc -a2 > fifo/il_P33  ) 2>> log/stderror.err &
( eve 34 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P34 -a1 -i - | tee fifo/gul_P34 | fmcalc -a2 > fifo/il_P34  ) 2>> log/stderror.err &
( eve 35 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P35 -a1 -i - | tee fifo/gul_P35 | fmcalc -a2 > fifo/il_P35  ) 2>> log/stderror.err &
( eve 36 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P36 -a1 -i - | tee fifo/gul_P36 | fmcalc -a2 > fifo/il_P36  ) 2>> log/stderror.err &
( eve 37 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P37 -a1 -i - | tee fifo/gul_P37 | fmcalc -a2 > fifo/il_P37  ) 2>> log/stderror.err &
( eve 38 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P38 -a1 -i - | tee fifo/gul_P38 | fmcalc -a2 > fifo/il_P38  ) 2>> log/stderror.err &
( eve 39 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P39 -a1 -i - | tee fifo/gul_P39 | fmcalc -a2 > fifo/il_P39  ) 2>> log/stderror.err &
( eve 40 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P40 -a1 -i - | tee fifo/gul_P40 | fmcalc -a2 > fifo/il_P40  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80 $pid81 $pid82 $pid83 $pid84 $pid85 $pid86 $pid87 $pid88 $pid89 $pid90 $pid91 $pid92 $pid93 $pid94 $pid95 $pid96 $pid97 $pid98 $pid99 $pid100 $pid101 $pid102 $pid103 $pid104 $pid105 $pid106 $pid107 $pid108 $pid109 $pid110 $pid111 $pid112 $pid113 $pid114 $pid115 $pid116 $pid117 $pid118 $pid119 $pid120 $pid121 $pid122 $pid123 $pid124 $pid125 $pid126 $pid127 $pid128 $pid129 $pid130 $pid131 $pid132 $pid133 $pid134 $pid135 $pid136 $pid137 $pid138 $pid139 $pid140 $pid141 $pid142 $pid143 $pid144 $pid145 $pid146 $pid147 $pid148 $pid149 $pid150 $pid151 $pid152 $pid153 $pid154 $pid155 $pid156 $pid157 $pid158 $pid159 $pid160 $pid161 $pid162 $pid163 $pid164 $pid165 $pid166 $pid167 $pid168 $pid169 $pid170 $pid171 $pid172 $pid173 $pid174 $pid175 $pid176 $pid177 $pid178 $pid179 $pid180 $pid181 $pid182 $pid183 $pid184 $pid185 $pid186 $pid187 $pid188 $pid189 $pid190 $pid191 $pid192 $pid193 $pid194 $pid195 $pid196 $pid197 $pid198 $pid199 $pid200 $pid201 $pid202 $pid203 $pid204 $pid205 $pid206 $pid207 $pid208 $pid209 $pid210 $pid211 $pid212 $pid213 $pid214 $pid215 $pid216 $pid217 $pid218 $pid219 $pid220 $pid221 $pid222 $pid223 $pid224 $pid225 $pid226 $pid227 $pid228 $pid229 $pid230 $pid231 $pid232 $pid233 $pid234 $pid235 $pid236 $pid237 $pid238 $pid239 $pid240 $pid241 $pid242 $pid243 $pid244 $pid245 $pid246 $pid247 $pid248 $pid249 $pid250 $pid251 $pid252 $pid253 $pid254 $pid255 $pid256 $pid257 $pid258 $pid259 $pid260 $pid261 $pid262 $pid263 $pid264 $pid265 $pid266 $pid267 $pid268 $pid269 $pid270 $pid271 $pid272 $pid273 $pid274 $pid275 $pid276 $pid277 $pid278 $pid279 $pid280 $pid281 $pid282 $pid283 $pid284 $pid285 $pid286 $pid287 $pid288 $pid289 $pid290 $pid291 $pid292 $pid293 $pid294 $pid295 $pid296 $pid297 $pid298 $pid299 $pid300 $pid301 $pid302 $pid303 $pid304 $pid305 $pid306 $pid307 $pid308 $pid309 $pid310 $pid311 $pid312 $pid313 $pid314 $pid315 $pid316 $pid317 $pid318 $pid319 $pid320 $pid321 $pid322 $pid323 $pid324 $pid325 $pid326 $pid327 $pid328 $pid329 $pid330 $pid331 $pid332 $pid333 $pid334 $pid335 $pid336 $pid337 $pid338 $pid339 $pid340 $pid341 $pid342 $pid343 $pid344 $pid345 $pid346 $pid347 $pid348 $pid349 $pid350 $pid351 $pid352 $pid353 $pid354 $pid355 $pid356 $pid357 $pid358 $pid359 $pid360 $pid361 $pid362 $pid363 $pid364 $pid365 $pid366 $pid367 $pid368 $pid369 $pid370 $pid371 $pid372 $pid373 $pid374 $pid375 $pid376 $pid377 $pid378 $pid379 $pid380 $pid381 $pid382 $pid383 $pid384 $pid385 $pid386 $pid387 $pid388 $pid389 $pid390 $pid391 $pid392 $pid393 $pid394 $pid395 $pid396 $pid397 $pid398 $pid399 $pid400 $pid401 $pid402 $pid403 $pid404 $pid405 $pid406 $pid407 $pid408 $pid409 $pid410 $pid411 $pid412 $pid413 $pid414 $pid415 $pid416 $pid417 $pid418 $pid419 $pid420 $pid421 $pid422 $pid423 $pid424 $pid425 $pid426 $pid427 $pid428 $pid429 $pid430 $pid431 $pid432 $pid433 $pid434 $pid435 $pid436 $pid437 $pid438 $pid439 $pid440 $pid441 $pid442 $pid443 $pid444 $pid445 $pid446 $pid447 $pid448 $pid449 $pid450 $pid451 $pid452 $pid453 $pid454 $pid455 $pid456 $pid457 $pid458 $pid459 $pid460 $pid461 $pid462 $pid463 $pid464 $pid465 $pid466 $pid467 $pid468 $pid469 $pid470 $pid471 $pid472 $pid473 $pid474 $pid475 $pid476 $pid477 $pid478 $pid479 $pid480 $pid481 $pid482 $pid483 $pid484 $pid485 $pid486 $pid487 $pid488 $pid489 $pid490 $pid491 $pid492 $pid493 $pid494 $pid495 $pid496 $pid497 $pid498 $pid499 $pid500 $pid501 $pid502 $pid503 $pid504 $pid505 $pid506 $pid507 $pid508 $pid509 $pid510 $pid511 $pid512 $pid513 $pid514 $pid515 $pid516 $pid517 $pid518 $pid519 $pid520 $pid521 $pid522 $pid523 $pid524 $pid525 $pid526 $pid527 $pid528 $pid529 $pid530 $pid531 $pid532 $pid533 $pid534 $pid535 $pid536 $pid537 $pid538 $pid539 $pid540 $pid541 $pid542 $pid543 $pid544 $pid545 $pid546 $pid547 $pid548 $pid549 $pid550 $pid551 $pid552 $pid553 $pid554 $pid555 $pid556 $pid557 $pid558 $pid559 $pid560 $pid561 $pid562 $pid563 $pid564 $pid565 $pid566 $pid567 $pid568 $pid569 $pid570 $pid571 $pid572 $pid573 $pid574 $pid575 $pid576 $pid577 $pid578 $pid579 $pid580 $pid581 $pid582 $pid583 $pid584 $pid585 $pid586 $pid587 $pid588 $pid589 $pid590 $pid591 $pid592 $pid593 $pid594 $pid595 $pid596 $pid597 $pid598 $pid599 $pid600 $pid601 $pid602 $pid603 $pid604 $pid605 $pid606 $pid607 $pid608 $pid609 $pid610 $pid611 $pid612 $pid613 $pid614 $pid615 $pid616 $pid617 $pid618 $pid619 $pid620 $pid621 $pid622 $pid623 $pid624 $pid625 $pid626 $pid627 $pid628 $pid629 $pid630 $pid631 $pid632 $pid633 $pid634 $pid635 $pid636 $pid637 $pid638 $pid639 $pid640


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 work/kat/il_S1_eltcalc_P3 work/kat/il_S1_eltcalc_P4 work/kat/il_S1_eltcalc_P5 work/kat/il_S1_eltcalc_P6 work/kat/il_S1_eltcalc_P7 work/kat/il_S1_eltcalc_P8 work/kat/il_S1_eltcalc_P9 work/kat/il_S1_eltcalc_P10 work/kat/il_S1_eltcalc_P11 work/kat/il_S1_eltcalc_P12 work/kat/il_S1_eltcalc_P13 work/kat/il_S1_eltcalc_P14 work/kat/il_S1_eltcalc_P15 work/kat/il_S1_eltcalc_P16 work/kat/il_S1_eltcalc_P17 work/kat/il_S1_eltcalc_P18 work/kat/il_S1_eltcalc_P19 work/kat/il_S1_eltcalc_P20 work/kat/il_S1_eltcalc_P21 work/kat/il_S1_eltcalc_P22 work/kat/il_S1_eltcalc_P23 work/kat/il_S1_eltcalc_P24 work/kat/il_S1_eltcalc_P25 work/kat/il_S1_eltcalc_P26 work/kat/il_S1_eltcalc_P27 work/kat/il_S1_eltcalc_P28 work/kat/il_S1_eltcalc_P29 work/kat/il_S1_eltcalc_P30 work/kat/il_S1_eltcalc_P31 work/kat/il_S1_eltcalc_P32 work/kat/il_S1_eltcalc_P33 work/kat/il_S1_eltcalc_P34 work/kat/il_S1_eltcalc_P35 work/kat/il_S1_eltcalc_P36 work/kat/il_S1_eltcalc_P37 work/kat/il_S1_eltcalc_P38 work/kat/il_S1_eltcalc_P39 work/kat/il_S1_eltcalc_P40 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 work/kat/il_S1_pltcalc_P2 work/kat/il_S1_pltcalc_P3 work/kat/il_S1_pltcalc_P4 work/kat/il_S1_pltcalc_P5 work/kat/il_S1_pltcalc_P6 work/kat/il_S1_pltcalc_P7 work/kat/il_S1_pltcalc_P8 work/kat/il_S1_pltcalc_P9 work/kat/il_S1_pltcalc_P10 work/kat/il_S1_pltcalc_P11 work/kat/il_S1_pltcalc_P12 work/kat/il_S1_pltcalc_P13 work/kat/il_S1_pltcalc_P14 work/kat/il_S1_pltcalc_P15 work/kat/il_S1_pltcalc_P16 work/kat/il_S1_pltcalc_P17 work/kat/il_S1_pltcalc_P18 work/kat/il_S1_pltcalc_P19 work/kat/il_S1_pltcalc_P20 work/kat/il_S1_pltcalc_P21 work/kat/il_S1_pltcalc_P22 work/kat/il_S1_pltcalc_P23 work/kat/il_S1_pltcalc_P24 work/kat/il_S1_pltcalc_P25 work/kat/il_S1_pltcalc_P26 work/kat/il_S1_pltcalc_P27 work/kat/il_S1_pltcalc_P28 work/kat/il_S1_pltcalc_P29 work/kat/il_S1_pltcalc_P30 work/kat/il_S1_pltcalc_P31 work/kat/il_S1_pltcalc_P32 work/kat/il_S1_pltcalc_P33 work/kat/il_S1_pltcalc_P34 work/kat/il_S1_pltcalc_P35 work/kat/il_S1_pltcalc_P36 work/kat/il_S1_pltcalc_P37 work/kat/il_S1_pltcalc_P38 work/kat/il_S1_pltcalc_P39 work/kat/il_S1_pltcalc_P40 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 work/kat/il_S1_summarycalc_P2 work/kat/il_S1_summarycalc_P3 work/kat/il_S1_summarycalc_P4 work/kat/il_S1_summarycalc_P5 work/kat/il_S1_summarycalc_P6 work/kat/il_S1_summarycalc_P7 work/kat/il_S1_summarycalc_P8 work/kat/il_S1_summarycalc_P9 work/kat/il_S1_summarycalc_P10 work/kat/il_S1_summarycalc_P11 work/kat/il_S1_summarycalc_P12 work/kat/il_S1_summarycalc_P13 work/kat/il_S1_summarycalc_P14 work/kat/il_S1_summarycalc_P15 work/kat/il_S1_summarycalc_P16 work/kat/il_S1_summarycalc_P17 work/kat/il_S1_summarycalc_P18 work/kat/il_S1_summarycalc_P19 work/kat/il_S1_summarycalc_P20 work/kat/il_S1_summarycalc_P21 work/kat/il_S1_summarycalc_P22 work/kat/il_S1_summarycalc_P23 work/kat/il_S1_summarycalc_P24 work/kat/il_S1_summarycalc_P25 work/kat/il_S1_summarycalc_P26 work/kat/il_S1_summarycalc_P27 work/kat/il_S1_summarycalc_P28 work/kat/il_S1_summarycalc_P29 work/kat/il_S1_summarycalc_P30 work/kat/il_S1_summarycalc_P31 work/kat/il_S1_summarycalc_P32 work/kat/il_S1_summarycalc_P33 work/kat/il_S1_summarycalc_P34 work/kat/il_S1_summarycalc_P35 work/kat/il_S1_summarycalc_P36 work/kat/il_S1_summarycalc_P37 work/kat/il_S1_summarycalc_P38 work/kat/il_S1_summarycalc_P39 work/kat/il_S1_summarycalc_P40 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 work/full_correlation/kat/il_S1_eltcalc_P2 work/full_correlation/kat/il_S1_eltcalc_P3 work/full_correlation/kat/il_S1_eltcalc_P4 work/full_correlation/kat/il_S1_eltcalc_P5 work/full_correlation/kat/il_S1_eltcalc_P6 work/full_correlation/kat/il_S1_eltcalc_P7 work/full_correlation/kat/il_S1_eltcalc_P8 work/full_correlation/kat/il_S1_eltcalc_P9 work/full_correlation/kat/il_S1_eltcalc_P10 work/full_correlation/kat/il_S1_eltcalc_P11 work/full_correlation/kat/il_S1_eltcalc_P12 work/full_correlation/kat/il_S1_eltcalc_P13 work/full_correlation/kat/il_S1_eltcalc_P14 work/full_correlation/kat/il_S1_eltcalc_P15 work/full_correlation/kat/il_S1_eltcalc_P16 work/full_correlation/kat/il_S1_eltcalc_P17 work/full_correlation/kat/il_S1_eltcalc_P18 work/full_correlation/kat/il_S1_eltcalc_P19 work/full_correlation/kat/il_S1_eltcalc_P20 work/full_correlation/kat/il_S1_eltcalc_P21 work/full_correlation/kat/il_S1_eltcalc_P22 work/full_correlation/kat/il_S1_eltcalc_P23 work/full_correlation/kat/il_S1_eltcalc_P24 work/full_correlation/kat/il_S1_eltcalc_P25 work/full_correlation/kat/il_S1_eltcalc_P26 work/full_correlation/kat/il_S1_eltcalc_P27 work/full_correlation/kat/il_S1_eltcalc_P28 work/full_correlation/kat/il_S1_eltcalc_P29 work/full_correlation/kat/il_S1_eltcalc_P30 work/full_correlation/kat/il_S1_eltcalc_P31 work/full_correlation/kat/il_S1_eltcalc_P32 work/full_correlation/kat/il_S1_eltcalc_P33 work/full_correlation/kat/il_S1_eltcalc_P34 work/full_correlation/kat/il_S1_eltcalc_P35 work/full_correlation/kat/il_S1_eltcalc_P36 work/full_correlation/kat/il_S1_eltcalc_P37 work/full_correlation/kat/il_S1_eltcalc_P38 work/full_correlation/kat/il_S1_eltcalc_P39 work/full_correlation/kat/il_S1_eltcalc_P40 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 work/full_correlation/kat/il_S1_pltcalc_P2 work/full_correlation/kat/il_S1_pltcalc_P3 work/full_correlation/kat/il_S1_pltcalc_P4 work/full_correlation/kat/il_S1_pltcalc_P5 work/full_correlation/kat/il_S1_pltcalc_P6 work/full_correlation/kat/il_S1_pltcalc_P7 work/full_correlation/kat/il_S1_pltcalc_P8 work/full_correlation/kat/il_S1_pltcalc_P9 work/full_correlation/kat/il_S1_pltcalc_P10 work/full_correlation/kat/il_S1_pltcalc_P11 work/full_correlation/kat/il_S1_pltcalc_P12 work/full_correlation/kat/il_S1_pltcalc_P13 work/full_correlation/kat/il_S1_pltcalc_P14 work/full_correlation/kat/il_S1_pltcalc_P15 work/full_correlation/kat/il_S1_pltcalc_P16 work/full_correlation/kat/il_S1_pltcalc_P17 work/full_correlation/kat/il_S1_pltcalc_P18 work/full_correlation/kat/il_S1_pltcalc_P19 work/full_correlation/kat/il_S1_pltcalc_P20 work/full_correlation/kat/il_S1_pltcalc_P21 work/full_correlation/kat/il_S1_pltcalc_P22 work/full_correlation/kat/il_S1_pltcalc_P23 work/full_correlation/kat/il_S1_pltcalc_P24 work/full_correlation/kat/il_S1_pltcalc_P25 work/full_correlation/kat/il_S1_pltcalc_P26 work/full_correlation/kat/il_S1_pltcalc_P27 work/full_correlation/kat/il_S1_pltcalc_P28 work/full_correlation/kat/il_S1_pltcalc_P29 work/full_correlation/kat/il_S1_pltcalc_P30 work/full_correlation/kat/il_S1_pltcalc_P31 work/full_correlation/kat/il_S1_pltcalc_P32 work/full_correlation/kat/il_S1_pltcalc_P33 work/full_correlation/kat/il_S1_pltcalc_P34 work/full_correlation/kat/il_S1_pltcalc_P35 work/full_correlation/kat/il_S1_pltcalc_P36 work/full_correlation/kat/il_S1_pltcalc_P37 work/full_correlation/kat/il_S1_pltcalc_P38 work/full_correlation/kat/il_S1_pltcalc_P39 work/full_correlation/kat/il_S1_pltcalc_P40 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 work/full_correlation/kat/il_S1_summarycalc_P2 work/full_correlation/kat/il_S1_summarycalc_P3 work/full_correlation/kat/il_S1_summarycalc_P4 work/full_correlation/kat/il_S1_summarycalc_P5 work/full_correlation/kat/il_S1_summarycalc_P6 work/full_correlation/kat/il_S1_summarycalc_P7 work/full_correlation/kat/il_S1_summarycalc_P8 work/full_correlation/kat/il_S1_summarycalc_P9 work/full_correlation/kat/il_S1_summarycalc_P10 work/full_correlation/kat/il_S1_summarycalc_P11 work/full_correlation/kat/il_S1_summarycalc_P12 work/full_correlation/kat/il_S1_summarycalc_P13 work/full_correlation/kat/il_S1_summarycalc_P14 work/full_correlation/kat/il_S1_summarycalc_P15 work/full_correlation/kat/il_S1_summarycalc_P16 work/full_correlation/kat/il_S1_summarycalc_P17 work/full_correlation/kat/il_S1_summarycalc_P18 work/full_correlation/kat/il_S1_summarycalc_P19 work/full_correlation/kat/il_S1_summarycalc_P20 work/full_correlation/kat/il_S1_summarycalc_P21 work/full_correlation/kat/il_S1_summarycalc_P22 work/full_correlation/kat/il_S1_summarycalc_P23 work/full_correlation/kat/il_S1_summarycalc_P24 work/full_correlation/kat/il_S1_summarycalc_P25 work/full_correlation/kat/il_S1_summarycalc_P26 work/full_correlation/kat/il_S1_summarycalc_P27 work/full_correlation/kat/il_S1_summarycalc_P28 work/full_correlation/kat/il_S1_summarycalc_P29 work/full_correlation/kat/il_S1_summarycalc_P30 work/full_correlation/kat/il_S1_summarycalc_P31 work/full_correlation/kat/il_S1_summarycalc_P32 work/full_correlation/kat/il_S1_summarycalc_P33 work/full_correlation/kat/il_S1_summarycalc_P34 work/full_correlation/kat/il_S1_summarycalc_P35 work/full_correlation/kat/il_S1_summarycalc_P36 work/full_correlation/kat/il_S1_summarycalc_P37 work/full_correlation/kat/il_S1_summarycalc_P38 work/full_correlation/kat/il_S1_summarycalc_P39 work/full_correlation/kat/il_S1_summarycalc_P40 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 work/kat/gul_S1_eltcalc_P3 work/kat/gul_S1_eltcalc_P4 work/kat/gul_S1_eltcalc_P5 work/kat/gul_S1_eltcalc_P6 work/kat/gul_S1_eltcalc_P7 work/kat/gul_S1_eltcalc_P8 work/kat/gul_S1_eltcalc_P9 work/kat/gul_S1_eltcalc_P10 work/kat/gul_S1_eltcalc_P11 work/kat/gul_S1_eltcalc_P12 work/kat/gul_S1_eltcalc_P13 work/kat/gul_S1_eltcalc_P14 work/kat/gul_S1_eltcalc_P15 work/kat/gul_S1_eltcalc_P16 work/kat/gul_S1_eltcalc_P17 work/kat/gul_S1_eltcalc_P18 work/kat/gul_S1_eltcalc_P19 work/kat/gul_S1_eltcalc_P20 work/kat/gul_S1_eltcalc_P21 work/kat/gul_S1_eltcalc_P22 work/kat/gul_S1_eltcalc_P23 work/kat/gul_S1_eltcalc_P24 work/kat/gul_S1_eltcalc_P25 work/kat/gul_S1_eltcalc_P26 work/kat/gul_S1_eltcalc_P27 work/kat/gul_S1_eltcalc_P28 work/kat/gul_S1_eltcalc_P29 work/kat/gul_S1_eltcalc_P30 work/kat/gul_S1_eltcalc_P31 work/kat/gul_S1_eltcalc_P32 work/kat/gul_S1_eltcalc_P33 work/kat/gul_S1_eltcalc_P34 work/kat/gul_S1_eltcalc_P35 work/kat/gul_S1_eltcalc_P36 work/kat/gul_S1_eltcalc_P37 work/kat/gul_S1_eltcalc_P38 work/kat/gul_S1_eltcalc_P39 work/kat/gul_S1_eltcalc_P40 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 work/kat/gul_S1_pltcalc_P3 work/kat/gul_S1_pltcalc_P4 work/kat/gul_S1_pltcalc_P5 work/kat/gul_S1_pltcalc_P6 work/kat/gul_S1_pltcalc_P7 work/kat/gul_S1_pltcalc_P8 work/kat/gul_S1_pltcalc_P9 work/kat/gul_S1_pltcalc_P10 work/kat/gul_S1_pltcalc_P11 work/kat/gul_S1_pltcalc_P12 work/kat/gul_S1_pltcalc_P13 work/kat/gul_S1_pltcalc_P14 work/kat/gul_S1_pltcalc_P15 work/kat/gul_S1_pltcalc_P16 work/kat/gul_S1_pltcalc_P17 work/kat/gul_S1_pltcalc_P18 work/kat/gul_S1_pltcalc_P19 work/kat/gul_S1_pltcalc_P20 work/kat/gul_S1_pltcalc_P21 work/kat/gul_S1_pltcalc_P22 work/kat/gul_S1_pltcalc_P23 work/kat/gul_S1_pltcalc_P24 work/kat/gul_S1_pltcalc_P25 work/kat/gul_S1_pltcalc_P26 work/kat/gul_S1_pltcalc_P27 work/kat/gul_S1_pltcalc_P28 work/kat/gul_S1_pltcalc_P29 work/kat/gul_S1_pltcalc_P30 work/kat/gul_S1_pltcalc_P31 work/kat/gul_S1_pltcalc_P32 work/kat/gul_S1_pltcalc_P33 work/kat/gul_S1_pltcalc_P34 work/kat/gul_S1_pltcalc_P35 work/kat/gul_S1_pltcalc_P36 work/kat/gul_S1_pltcalc_P37 work/kat/gul_S1_pltcalc_P38 work/kat/gul_S1_pltcalc_P39 work/kat/gul_S1_pltcalc_P40 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 work/kat/gul_S1_summarycalc_P3 work/kat/gul_S1_summarycalc_P4 work/kat/gul_S1_summarycalc_P5 work/kat/gul_S1_summarycalc_P6 work/kat/gul_S1_summarycalc_P7 work/kat/gul_S1_summarycalc_P8 work/kat/gul_S1_summarycalc_P9 work/kat/gul_S1_summarycalc_P10 work/kat/gul_S1_summarycalc_P11 work/kat/gul_S1_summarycalc_P12 work/kat/gul_S1_summarycalc_P13 work/kat/gul_S1_summarycalc_P14 work/kat/gul_S1_summarycalc_P15 work/kat/gul_S1_summarycalc_P16 work/kat/gul_S1_summarycalc_P17 work/kat/gul_S1_summarycalc_P18 work/kat/gul_S1_summarycalc_P19 work/kat/gul_S1_summarycalc_P20 work/kat/gul_S1_summarycalc_P21 work/kat/gul_S1_summarycalc_P22 work/kat/gul_S1_summarycalc_P23 work/kat/gul_S1_summarycalc_P24 work/kat/gul_S1_summarycalc_P25 work/kat/gul_S1_summarycalc_P26 work/kat/gul_S1_summarycalc_P27 work/kat/gul_S1_summarycalc_P28 work/kat/gul_S1_summarycalc_P29 work/kat/gul_S1_summarycalc_P30 work/kat/gul_S1_summarycalc_P31 work/kat/gul_S1_summarycalc_P32 work/kat/gul_S1_summarycalc_P33 work/kat/gul_S1_summarycalc_P34 work/kat/gul_S1_summarycalc_P35 work/kat/gul_S1_summarycalc_P36 work/kat/gul_S1_summarycalc_P37 work/kat/gul_S1_summarycalc_P38 work/kat/gul_S1_summarycalc_P39 work/kat/gul_S1_summarycalc_P40 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 work/full_correlation/kat/gul_S1_eltcalc_P3 work/full_correlation/kat/gul_S1_eltcalc_P4 work/full_correlation/kat/gul_S1_eltcalc_P5 work/full_correlation/kat/gul_S1_eltcalc_P6 work/full_correlation/kat/gul_S1_eltcalc_P7 work/full_correlation/kat/gul_S1_eltcalc_P8 work/full_correlation/kat/gul_S1_eltcalc_P9 work/full_correlation/kat/gul_S1_eltcalc_P10 work/full_correlation/kat/gul_S1_eltcalc_P11 work/full_correlation/kat/gul_S1_eltcalc_P12 work/full_correlation/kat/gul_S1_eltcalc_P13 work/full_correlation/kat/gul_S1_eltcalc_P14 work/full_correlation/kat/gul_S1_eltcalc_P15 work/full_correlation/kat/gul_S1_eltcalc_P16 work/full_correlation/kat/gul_S1_eltcalc_P17 work/full_correlation/kat/gul_S1_eltcalc_P18 work/full_correlation/kat/gul_S1_eltcalc_P19 work/full_correlation/kat/gul_S1_eltcalc_P20 work/full_correlation/kat/gul_S1_eltcalc_P21 work/full_correlation/kat/gul_S1_eltcalc_P22 work/full_correlation/kat/gul_S1_eltcalc_P23 work/full_correlation/kat/gul_S1_eltcalc_P24 work/full_correlation/kat/gul_S1_eltcalc_P25 work/full_correlation/kat/gul_S1_eltcalc_P26 work/full_correlation/kat/gul_S1_eltcalc_P27 work/full_correlation/kat/gul_S1_eltcalc_P28 work/full_correlation/kat/gul_S1_eltcalc_P29 work/full_correlation/kat/gul_S1_eltcalc_P30 work/full_correlation/kat/gul_S1_eltcalc_P31 work/full_correlation/kat/gul_S1_eltcalc_P32 work/full_correlation/kat/gul_S1_eltcalc_P33 work/full_correlation/kat/gul_S1_eltcalc_P34 work/full_correlation/kat/gul_S1_eltcalc_P35 work/full_correlation/kat/gul_S1_eltcalc_P36 work/full_correlation/kat/gul_S1_eltcalc_P37 work/full_correlation/kat/gul_S1_eltcalc_P38 work/full_correlation/kat/gul_S1_eltcalc_P39 work/full_correlation/kat/gul_S1_eltcalc_P40 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 work/full_correlation/kat/gul_S1_pltcalc_P3 work/full_correlation/kat/gul_S1_pltcalc_P4 work/full_correlation/kat/gul_S1_pltcalc_P5 work/full_correlation/kat/gul_S1_pltcalc_P6 work/full_correlation/kat/gul_S1_pltcalc_P7 work/full_correlation/kat/gul_S1_pltcalc_P8 work/full_correlation/kat/gul_S1_pltcalc_P9 work/full_correlation/kat/gul_S1_pltcalc_P10 work/full_correlation/kat/gul_S1_pltcalc_P11 work/full_correlation/kat/gul_S1_pltcalc_P12 work/full_correlation/kat/gul_S1_pltcalc_P13 work/full_correlation/kat/gul_S1_pltcalc_P14 work/full_correlation/kat/gul_S1_pltcalc_P15 work/full_correlation/kat/gul_S1_pltcalc_P16 work/full_correlation/kat/gul_S1_pltcalc_P17 work/full_correlation/kat/gul_S1_pltcalc_P18 work/full_correlation/kat/gul_S1_pltcalc_P19 work/full_correlation/kat/gul_S1_pltcalc_P20 work/full_correlation/kat/gul_S1_pltcalc_P21 work/full_correlation/kat/gul_S1_pltcalc_P22 work/full_correlation/kat/gul_S1_pltcalc_P23 work/full_correlation/kat/gul_S1_pltcalc_P24 work/full_correlation/kat/gul_S1_pltcalc_P25 work/full_correlation/kat/gul_S1_pltcalc_P26 work/full_correlation/kat/gul_S1_pltcalc_P27 work/full_correlation/kat/gul_S1_pltcalc_P28 work/full_correlation/kat/gul_S1_pltcalc_P29 work/full_correlation/kat/gul_S1_pltcalc_P30 work/full_correlation/kat/gul_S1_pltcalc_P31 work/full_correlation/kat/gul_S1_pltcalc_P32 work/full_correlation/kat/gul_S1_pltcalc_P33 work/full_correlation/kat/gul_S1_pltcalc_P34 work/full_correlation/kat/gul_S1_pltcalc_P35 work/full_correlation/kat/gul_S1_pltcalc_P36 work/full_correlation/kat/gul_S1_pltcalc_P37 work/full_correlation/kat/gul_S1_pltcalc_P38 work/full_correlation/kat/gul_S1_pltcalc_P39 work/full_correlation/kat/gul_S1_pltcalc_P40 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 work/full_correlation/kat/gul_S1_summarycalc_P3 work/full_correlation/kat/gul_S1_summarycalc_P4 work/full_correlation/kat/gul_S1_summarycalc_P5 work/full_correlation/kat/gul_S1_summarycalc_P6 work/full_correlation/kat/gul_S1_summarycalc_P7 work/full_correlation/kat/gul_S1_summarycalc_P8 work/full_correlation/kat/gul_S1_summarycalc_P9 work/full_correlation/kat/gul_S1_summarycalc_P10 work/full_correlation/kat/gul_S1_summarycalc_P11 work/full_correlation/kat/gul_S1_summarycalc_P12 work/full_correlation/kat/gul_S1_summarycalc_P13 work/full_correlation/kat/gul_S1_summarycalc_P14 work/full_correlation/kat/gul_S1_summarycalc_P15 work/full_correlation/kat/gul_S1_summarycalc_P16 work/full_correlation/kat/gul_S1_summarycalc_P17 work/full_correlation/kat/gul_S1_summarycalc_P18 work/full_correlation/kat/gul_S1_summarycalc_P19 work/full_correlation/kat/gul_S1_summarycalc_P20 work/full_correlation/kat/gul_S1_summarycalc_P21 work/full_correlation/kat/gul_S1_summarycalc_P22 work/full_correlation/kat/gul_S1_summarycalc_P23 work/full_correlation/kat/gul_S1_summarycalc_P24 work/full_correlation/kat/gul_S1_summarycalc_P25 work/full_correlation/kat/gul_S1_summarycalc_P26 work/full_correlation/kat/gul_S1_summarycalc_P27 work/full_correlation/kat/gul_S1_summarycalc_P28 work/full_correlation/kat/gul_S1_summarycalc_P29 work/full_correlation/kat/gul_S1_summarycalc_P30 work/full_correlation/kat/gul_S1_summarycalc_P31 work/full_correlation/kat/gul_S1_summarycalc_P32 work/full_correlation/kat/gul_S1_summarycalc_P33 work/full_correlation/kat/gul_S1_summarycalc_P34 work/full_correlation/kat/gul_S1_summarycalc_P35 work/full_correlation/kat/gul_S1_summarycalc_P36 work/full_correlation/kat/gul_S1_summarycalc_P37 work/full_correlation/kat/gul_S1_summarycalc_P38 work/full_correlation/kat/gul_S1_summarycalc_P39 work/full_correlation/kat/gul_S1_summarycalc_P40 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> log/stderror.err & lpid1=$!
( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid2=$!
( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> log/stderror.err & lpid3=$!
( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid4=$!
( aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv ) 2>> log/stderror.err & lpid5=$!
( leccalc -r -Kfull_correlation/il_S1_summaryleccalc -F output/full_correlation/il_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/il_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/il_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/il_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/il_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/il_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid6=$!
( aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv ) 2>> log/stderror.err & lpid7=$!
( leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
