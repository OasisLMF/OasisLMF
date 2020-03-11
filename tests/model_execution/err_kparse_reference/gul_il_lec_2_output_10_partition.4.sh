#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

error_handler(){
   echo 'Run Error - terminating'
   exit_code=$?
   set +x
   group_pid=$(ps -p $$ -o pgid --no-headers)
   sess_pid=$(ps -p $$ -o sess --no-headers)
   printf "Script PID:%d, GPID:%s, SPID:%d" $$ $group_pid $sess_pid >> log/killout.txt

   if hash pstree 2>/dev/null; then
       pstree -pn $$ >> log/killout.txt
       PIDS_KILL=$(pstree -pn $$ | grep -o "([[:digit:]]*)" | grep -o "[[:digit:]]*")
       kill -9 $(echo "$PIDS_KILL" | grep -v $group_pid | grep -v $$) 2>/dev/null
   else
       ps f -g $sess_pid > log/subprocess_list
       PIDS_KILL=$(pgrep -a --pgroup $group_pid | grep -v celery | grep -v $group_pid | grep -v $$)
       echo "$PIDS_KILL" >> log/killout.txt
       kill -9 $(echo "$PIDS_KILL" | awk 'BEGIN { FS = "[ \t\n]+" }{ print $1 }') 2>/dev/null
   fi
   exit $(( 1 > $exit_code ? 1 : $exit_code ))
}
trap error_handler QUIT HUP INT KILL TERM ERR

touch log/stderror.err
ktools_monitor.sh $$ & pid0=$!

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_summaryaalcalc

mkfifo fifo/gul_P5

mkfifo fifo/il_P5

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summaryeltcalc_P5
mkfifo fifo/gul_S1_eltcalc_P5
mkfifo fifo/gul_S1_summarysummarycalc_P5
mkfifo fifo/gul_S1_summarycalc_P5
mkfifo fifo/gul_S1_summarypltcalc_P5
mkfifo fifo/gul_S1_pltcalc_P5
mkfifo fifo/gul_S2_summary_P5
mkfifo fifo/gul_S2_summaryeltcalc_P5
mkfifo fifo/gul_S2_eltcalc_P5
mkfifo fifo/gul_S2_summarysummarycalc_P5
mkfifo fifo/gul_S2_summarycalc_P5
mkfifo fifo/gul_S2_summarypltcalc_P5
mkfifo fifo/gul_S2_pltcalc_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summaryeltcalc_P5
mkfifo fifo/il_S1_eltcalc_P5
mkfifo fifo/il_S1_summarysummarycalc_P5
mkfifo fifo/il_S1_summarycalc_P5
mkfifo fifo/il_S1_summarypltcalc_P5
mkfifo fifo/il_S1_pltcalc_P5
mkfifo fifo/il_S2_summary_P5
mkfifo fifo/il_S2_summaryeltcalc_P5
mkfifo fifo/il_S2_eltcalc_P5
mkfifo fifo/il_S2_summarysummarycalc_P5
mkfifo fifo/il_S2_summarycalc_P5
mkfifo fifo/il_S2_summarypltcalc_P5
mkfifo fifo/il_S2_pltcalc_P5



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P5 > work/kat/il_S1_eltcalc_P5 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P5 > work/kat/il_S1_pltcalc_P5 & pid3=$!
eltcalc -s < fifo/il_S2_summaryeltcalc_P5 > work/kat/il_S2_eltcalc_P5 & pid4=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P5 > work/kat/il_S2_summarycalc_P5 & pid5=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P5 > work/kat/il_S2_pltcalc_P5 & pid6=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summaryeltcalc_P5 fifo/il_S1_summarypltcalc_P5 fifo/il_S1_summarysummarycalc_P5 work/il_S1_summaryaalcalc/P5.bin work/il_S1_summaryleccalc/P5.bin > /dev/null & pid7=$!
tee < fifo/il_S2_summary_P5 fifo/il_S2_summaryeltcalc_P5 fifo/il_S2_summarypltcalc_P5 fifo/il_S2_summarysummarycalc_P5 work/il_S2_summaryaalcalc/P5.bin work/il_S2_summaryleccalc/P5.bin > /dev/null & pid8=$!
( summarycalc -f  -1 fifo/il_S1_summary_P5 -2 fifo/il_S2_summary_P5 < fifo/il_P5 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P5 > work/kat/gul_S1_eltcalc_P5 & pid9=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P5 > work/kat/gul_S1_summarycalc_P5 & pid10=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid11=$!
eltcalc -s < fifo/gul_S2_summaryeltcalc_P5 > work/kat/gul_S2_eltcalc_P5 & pid12=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P5 > work/kat/gul_S2_summarycalc_P5 & pid13=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P5 > work/kat/gul_S2_pltcalc_P5 & pid14=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_summaryeltcalc_P5 fifo/gul_S1_summarypltcalc_P5 fifo/gul_S1_summarysummarycalc_P5 work/gul_S1_summaryaalcalc/P5.bin work/gul_S1_summaryleccalc/P5.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P5 fifo/gul_S2_summaryeltcalc_P5 fifo/gul_S2_summarypltcalc_P5 fifo/gul_S2_summarysummarycalc_P5 work/gul_S2_summaryaalcalc/P5.bin work/gul_S2_summaryleccalc/P5.bin > /dev/null & pid16=$!
( summarycalc -i  -1 fifo/gul_S1_summary_P5 -2 fifo/gul_S2_summary_P5 < fifo/gul_P5 ) 2>> log/stderror.err  &

( eve 5 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P5 | fmcalc -a2 > fifo/il_P5  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P5 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P5 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P5 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P5 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P5 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P5 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P5 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P5 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P5 > output/gul_S1_summarycalc.csv & kpid9=$!
kat work/kat/gul_S2_eltcalc_P5 > output/gul_S2_eltcalc.csv & kpid10=$!
kat work/kat/gul_S2_pltcalc_P5 > output/gul_S2_pltcalc.csv & kpid11=$!
kat work/kat/gul_S2_summarycalc_P5 > output/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


# Stop ktools watcher
kill -9 $pid0
