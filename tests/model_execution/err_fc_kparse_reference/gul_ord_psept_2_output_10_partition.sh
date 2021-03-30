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
    fi
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S2_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S2_summaryleccalc

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

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S2_summary_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S2_summary_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S2_summary_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S2_summary_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S2_summary_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S2_summary_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S2_summary_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S2_summary_P8

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S2_summary_P9

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S2_summary_P10

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

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S2_summary_P1

mkfifo fifo/full_correlation/gul_S1_summary_P2
mkfifo fifo/full_correlation/gul_S2_summary_P2

mkfifo fifo/full_correlation/gul_S1_summary_P3
mkfifo fifo/full_correlation/gul_S2_summary_P3

mkfifo fifo/full_correlation/gul_S1_summary_P4
mkfifo fifo/full_correlation/gul_S2_summary_P4

mkfifo fifo/full_correlation/gul_S1_summary_P5
mkfifo fifo/full_correlation/gul_S2_summary_P5

mkfifo fifo/full_correlation/gul_S1_summary_P6
mkfifo fifo/full_correlation/gul_S2_summary_P6

mkfifo fifo/full_correlation/gul_S1_summary_P7
mkfifo fifo/full_correlation/gul_S2_summary_P7

mkfifo fifo/full_correlation/gul_S1_summary_P8
mkfifo fifo/full_correlation/gul_S2_summary_P8

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S2_summary_P9

mkfifo fifo/full_correlation/gul_S1_summary_P10
mkfifo fifo/full_correlation/gul_S2_summary_P10



# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/gul_S2_summary_P1 work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid2=$!
tee < fifo/gul_S1_summary_P2 work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid3=$!
tee < fifo/gul_S2_summary_P2 work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid4=$!
tee < fifo/gul_S1_summary_P3 work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid5=$!
tee < fifo/gul_S2_summary_P3 work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid6=$!
tee < fifo/gul_S1_summary_P4 work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid7=$!
tee < fifo/gul_S2_summary_P4 work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid8=$!
tee < fifo/gul_S1_summary_P5 work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid9=$!
tee < fifo/gul_S2_summary_P5 work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid10=$!
tee < fifo/gul_S1_summary_P6 work/gul_S1_summaryleccalc/P6.bin > /dev/null & pid11=$!
tee < fifo/gul_S2_summary_P6 work/gul_S2_summaryleccalc/P6.bin > /dev/null & pid12=$!
tee < fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid13=$!
tee < fifo/gul_S2_summary_P7 work/gul_S2_summaryleccalc/P7.bin > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P8 work/gul_S1_summaryleccalc/P8.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P8 work/gul_S2_summaryleccalc/P8.bin > /dev/null & pid16=$!
tee < fifo/gul_S1_summary_P9 work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid17=$!
tee < fifo/gul_S2_summary_P9 work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid18=$!
tee < fifo/gul_S1_summary_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid19=$!
tee < fifo/gul_S2_summary_P10 work/gul_S2_summaryleccalc/P10.bin > /dev/null & pid20=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P6 -2 fifo/gul_S2_summary_P6 < fifo/gul_P6 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P7 -2 fifo/gul_S2_summary_P7 < fifo/gul_P7 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P8 -2 fifo/gul_S2_summary_P8 < fifo/gul_P8 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P10 -2 fifo/gul_S2_summary_P10 < fifo/gul_P10 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---


tee < fifo/full_correlation/gul_S1_summary_P1 work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid21=$!
tee < fifo/full_correlation/gul_S2_summary_P1 work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid22=$!
tee < fifo/full_correlation/gul_S1_summary_P2 work/full_correlation/gul_S1_summaryleccalc/P2.bin > /dev/null & pid23=$!
tee < fifo/full_correlation/gul_S2_summary_P2 work/full_correlation/gul_S2_summaryleccalc/P2.bin > /dev/null & pid24=$!
tee < fifo/full_correlation/gul_S1_summary_P3 work/full_correlation/gul_S1_summaryleccalc/P3.bin > /dev/null & pid25=$!
tee < fifo/full_correlation/gul_S2_summary_P3 work/full_correlation/gul_S2_summaryleccalc/P3.bin > /dev/null & pid26=$!
tee < fifo/full_correlation/gul_S1_summary_P4 work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid27=$!
tee < fifo/full_correlation/gul_S2_summary_P4 work/full_correlation/gul_S2_summaryleccalc/P4.bin > /dev/null & pid28=$!
tee < fifo/full_correlation/gul_S1_summary_P5 work/full_correlation/gul_S1_summaryleccalc/P5.bin > /dev/null & pid29=$!
tee < fifo/full_correlation/gul_S2_summary_P5 work/full_correlation/gul_S2_summaryleccalc/P5.bin > /dev/null & pid30=$!
tee < fifo/full_correlation/gul_S1_summary_P6 work/full_correlation/gul_S1_summaryleccalc/P6.bin > /dev/null & pid31=$!
tee < fifo/full_correlation/gul_S2_summary_P6 work/full_correlation/gul_S2_summaryleccalc/P6.bin > /dev/null & pid32=$!
tee < fifo/full_correlation/gul_S1_summary_P7 work/full_correlation/gul_S1_summaryleccalc/P7.bin > /dev/null & pid33=$!
tee < fifo/full_correlation/gul_S2_summary_P7 work/full_correlation/gul_S2_summaryleccalc/P7.bin > /dev/null & pid34=$!
tee < fifo/full_correlation/gul_S1_summary_P8 work/full_correlation/gul_S1_summaryleccalc/P8.bin > /dev/null & pid35=$!
tee < fifo/full_correlation/gul_S2_summary_P8 work/full_correlation/gul_S2_summaryleccalc/P8.bin > /dev/null & pid36=$!
tee < fifo/full_correlation/gul_S1_summary_P9 work/full_correlation/gul_S1_summaryleccalc/P9.bin > /dev/null & pid37=$!
tee < fifo/full_correlation/gul_S2_summary_P9 work/full_correlation/gul_S2_summaryleccalc/P9.bin > /dev/null & pid38=$!
tee < fifo/full_correlation/gul_S1_summary_P10 work/full_correlation/gul_S1_summaryleccalc/P10.bin > /dev/null & pid39=$!
tee < fifo/full_correlation/gul_S2_summary_P10 work/full_correlation/gul_S2_summaryleccalc/P10.bin > /dev/null & pid40=$!

( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 -2 fifo/full_correlation/gul_S2_summary_P1 < fifo/full_correlation/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P2 -2 fifo/full_correlation/gul_S2_summary_P2 < fifo/full_correlation/gul_P2 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P3 -2 fifo/full_correlation/gul_S2_summary_P3 < fifo/full_correlation/gul_P3 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P4 -2 fifo/full_correlation/gul_S2_summary_P4 < fifo/full_correlation/gul_P4 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P5 -2 fifo/full_correlation/gul_S2_summary_P5 < fifo/full_correlation/gul_P5 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P6 -2 fifo/full_correlation/gul_S2_summary_P6 < fifo/full_correlation/gul_P6 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P7 -2 fifo/full_correlation/gul_S2_summary_P7 < fifo/full_correlation/gul_P7 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P8 -2 fifo/full_correlation/gul_S2_summary_P8 < fifo/full_correlation/gul_P8 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P9 -2 fifo/full_correlation/gul_S2_summary_P9 < fifo/full_correlation/gul_P9 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P10 -2 fifo/full_correlation/gul_S2_summary_P10 < fifo/full_correlation/gul_P10 ) 2>> log/stderror.err  &

( eve 1 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P1 -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &
( eve 2 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P2 -a1 -i - > fifo/gul_P2  ) 2>> log/stderror.err &
( eve 3 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P3 -a1 -i - > fifo/gul_P3  ) 2>> log/stderror.err &
( eve 4 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P4 -a1 -i - > fifo/gul_P4  ) 2>> log/stderror.err &
( eve 5 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P5 -a1 -i - > fifo/gul_P5  ) 2>> log/stderror.err &
( eve 6 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P6 -a1 -i - > fifo/gul_P6  ) 2>> log/stderror.err &
( eve 7 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P7 -a1 -i - > fifo/gul_P7  ) 2>> log/stderror.err &
( eve 8 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P8 -a1 -i - > fifo/gul_P8  ) 2>> log/stderror.err &
( eve 9 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P9 -a1 -i - > fifo/gul_P9  ) 2>> log/stderror.err &
( eve 10 10 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P10 -a1 -i - > fifo/gul_P10  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


( ordleccalc  -Kgul_S1_summaryleccalc -W -w -o output/gul_S1_psept.csv ) 2>> log/stderror.err & lpid1=$!
( ordleccalc -r -Kgul_S2_summaryleccalc -W -w -o output/gul_S2_psept.csv ) 2>> log/stderror.err & lpid2=$!
( ordleccalc  -Kfull_correlation/gul_S1_summaryleccalc -W -w -o output/full_correlation/gul_S1_psept.csv ) 2>> log/stderror.err & lpid3=$!
( ordleccalc -r -Kfull_correlation/gul_S2_summaryleccalc -W -w -o output/full_correlation/gul_S2_psept.csv ) 2>> log/stderror.err & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
