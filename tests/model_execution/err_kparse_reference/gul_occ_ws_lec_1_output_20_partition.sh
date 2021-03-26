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
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | grep -v *.sh)
   echo "$PIDS_KILL" >> log/killout.txt
   kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   exit $exit_code
}
trap exit_handler QUIT HUP INT KILL TERM ERR EXIT

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

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc

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

mkfifo fifo/gul_S1_summary_P1

mkfifo fifo/gul_S1_summary_P2

mkfifo fifo/gul_S1_summary_P3

mkfifo fifo/gul_S1_summary_P4

mkfifo fifo/gul_S1_summary_P5

mkfifo fifo/gul_S1_summary_P6

mkfifo fifo/gul_S1_summary_P7

mkfifo fifo/gul_S1_summary_P8

mkfifo fifo/gul_S1_summary_P9

mkfifo fifo/gul_S1_summary_P10

mkfifo fifo/gul_S1_summary_P11

mkfifo fifo/gul_S1_summary_P12

mkfifo fifo/gul_S1_summary_P13

mkfifo fifo/gul_S1_summary_P14

mkfifo fifo/gul_S1_summary_P15

mkfifo fifo/gul_S1_summary_P16

mkfifo fifo/gul_S1_summary_P17

mkfifo fifo/gul_S1_summary_P18

mkfifo fifo/gul_S1_summary_P19

mkfifo fifo/gul_S1_summary_P20



# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid2=$!
tee < fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid3=$!
tee < fifo/gul_S1_summary_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid4=$!
tee < fifo/gul_S1_summary_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid5=$!
tee < fifo/gul_S1_summary_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid6=$!
tee < fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid7=$!
tee < fifo/gul_S1_summary_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid8=$!
tee < fifo/gul_S1_summary_P9 work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid10=$!
tee < fifo/gul_S1_summary_P11 work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid11=$!
tee < fifo/gul_S1_summary_P12 work/gul_S1_summaryleccalc/P12.bin > /dev/null & pid12=$!
tee < fifo/gul_S1_summary_P13 work/gul_S1_summaryleccalc/P13.bin > /dev/null & pid13=$!
tee < fifo/gul_S1_summary_P14 work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P15 work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid15=$!
tee < fifo/gul_S1_summary_P16 work/gul_S1_summaryleccalc/P16.bin > /dev/null & pid16=$!
tee < fifo/gul_S1_summary_P17 work/gul_S1_summaryleccalc/P17.bin > /dev/null & pid17=$!
tee < fifo/gul_S1_summary_P18 work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid18=$!
tee < fifo/gul_S1_summary_P19 work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid19=$!
tee < fifo/gul_S1_summary_P20 work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid20=$!

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

( eve 1 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &
( eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P2  ) 2>> log/stderror.err &
( eve 3 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P3  ) 2>> log/stderror.err &
( eve 4 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P4  ) 2>> log/stderror.err &
( eve 5 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P5  ) 2>> log/stderror.err &
( eve 6 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P6  ) 2>> log/stderror.err &
( eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P7  ) 2>> log/stderror.err &
( eve 8 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P8  ) 2>> log/stderror.err &
( eve 9 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P9  ) 2>> log/stderror.err &
( eve 10 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P10  ) 2>> log/stderror.err &
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P11  ) 2>> log/stderror.err &
( eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P12  ) 2>> log/stderror.err &
( eve 13 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P13  ) 2>> log/stderror.err &
( eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P14  ) 2>> log/stderror.err &
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P15  ) 2>> log/stderror.err &
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P16  ) 2>> log/stderror.err &
( eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P17  ) 2>> log/stderror.err &
( eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P18  ) 2>> log/stderror.err &
( eve 19 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P19  ) 2>> log/stderror.err &
( eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P20  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do ground up loss kats ---


( leccalc -r -Kgul_S1_summaryleccalc -w output/gul_S1_leccalc_wheatsheaf_oep.csv ) 2>> log/stderror.err & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
