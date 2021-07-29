#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

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

   ps -jf f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | awk 'BEGIN { FS = "[ \t\n]+" }{ if ($1 >= '$script_pid') print}' | grep -v celery | egrep -v *\\.log$  | egrep -v *\\.sh$ | sort -n -r)
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

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
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

mkfifo fifo/gul_P9

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_summary_P9.idx
mkfifo fifo/gul_S1_eltcalc_P9
mkfifo fifo/gul_S1_summarycalc_P9
mkfifo fifo/gul_S1_pltcalc_P9
mkfifo fifo/gul_S2_summary_P9
mkfifo fifo/gul_S2_summary_P9.idx
mkfifo fifo/gul_S2_eltcalc_P9
mkfifo fifo/gul_S2_summarycalc_P9
mkfifo fifo/gul_S2_pltcalc_P9

mkfifo fifo/il_P9

mkfifo fifo/il_S1_summary_P9
mkfifo fifo/il_S1_summary_P9.idx
mkfifo fifo/il_S1_eltcalc_P9
mkfifo fifo/il_S1_summarycalc_P9
mkfifo fifo/il_S1_pltcalc_P9
mkfifo fifo/il_S2_summary_P9
mkfifo fifo/il_S2_summary_P9.idx
mkfifo fifo/il_S2_eltcalc_P9
mkfifo fifo/il_S2_summarycalc_P9
mkfifo fifo/il_S2_pltcalc_P9



# --- Do insured loss computes ---
( eltcalc -s < fifo/il_S1_eltcalc_P9 > work/kat/il_S1_eltcalc_P9 ) 2>> log/stderror.err & pid1=$!
( summarycalctocsv -s < fifo/il_S1_summarycalc_P9 > work/kat/il_S1_summarycalc_P9 ) 2>> log/stderror.err & pid2=$!
( pltcalc -s < fifo/il_S1_pltcalc_P9 > work/kat/il_S1_pltcalc_P9 ) 2>> log/stderror.err & pid3=$!
( eltcalc -s < fifo/il_S2_eltcalc_P9 > work/kat/il_S2_eltcalc_P9 ) 2>> log/stderror.err & pid4=$!
( summarycalctocsv -s < fifo/il_S2_summarycalc_P9 > work/kat/il_S2_summarycalc_P9 ) 2>> log/stderror.err & pid5=$!
( pltcalc -s < fifo/il_S2_pltcalc_P9 > work/kat/il_S2_pltcalc_P9 ) 2>> log/stderror.err & pid6=$!
tee < fifo/il_S1_summary_P9 fifo/il_S1_eltcalc_P9 fifo/il_S1_summarycalc_P9 fifo/il_S1_pltcalc_P9 work/il_S1_summaryaalcalc/P9.bin work/il_S1_summaryleccalc/P9.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P9.idx work/il_S1_summaryleccalc/P9.idx > /dev/null & pid8=$!
tee < fifo/il_S2_summary_P9 fifo/il_S2_eltcalc_P9 fifo/il_S2_summarycalc_P9 fifo/il_S2_pltcalc_P9 work/il_S2_summaryaalcalc/P9.bin work/il_S2_summaryleccalc/P9.bin > /dev/null & pid9=$!
tee < fifo/il_S2_summary_P9.idx work/il_S2_summaryleccalc/P9.idx > /dev/null & pid10=$!
( summarycalc -m -f  -1 fifo/il_S1_summary_P9 -2 fifo/il_S2_summary_P9 < fifo/il_P9 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---
( eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 ) 2>> log/stderror.err & pid11=$!
( summarycalctocsv -s < fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 ) 2>> log/stderror.err & pid12=$!
( pltcalc -s < fifo/gul_S1_pltcalc_P9 > work/kat/gul_S1_pltcalc_P9 ) 2>> log/stderror.err & pid13=$!
( eltcalc -s < fifo/gul_S2_eltcalc_P9 > work/kat/gul_S2_eltcalc_P9 ) 2>> log/stderror.err & pid14=$!
( summarycalctocsv -s < fifo/gul_S2_summarycalc_P9 > work/kat/gul_S2_summarycalc_P9 ) 2>> log/stderror.err & pid15=$!
( pltcalc -s < fifo/gul_S2_pltcalc_P9 > work/kat/gul_S2_pltcalc_P9 ) 2>> log/stderror.err & pid16=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 fifo/gul_S1_summarycalc_P9 fifo/gul_S1_pltcalc_P9 work/gul_S1_summaryaalcalc/P9.bin work/gul_S1_summaryleccalc/P9.bin > /dev/null & pid17=$!
tee < fifo/gul_S1_summary_P9.idx work/gul_S1_summaryleccalc/P9.idx > /dev/null & pid18=$!
tee < fifo/gul_S2_summary_P9 fifo/gul_S2_eltcalc_P9 fifo/gul_S2_summarycalc_P9 fifo/gul_S2_pltcalc_P9 work/gul_S2_summaryaalcalc/P9.bin work/gul_S2_summaryleccalc/P9.bin > /dev/null & pid19=$!
tee < fifo/gul_S2_summary_P9.idx work/gul_S2_summaryleccalc/P9.idx > /dev/null & pid20=$!
( summarycalc -m -i  -1 fifo/gul_S1_summary_P9 -2 fifo/gul_S2_summary_P9 < fifo/gul_P9 ) 2>> log/stderror.err  &

( eve 9 10 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P9 | fmcalc -a2 > fifo/il_P9  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P9 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P9 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P9 > output/il_S1_summarycalc.csv & kpid3=$!
kat -s work/kat/il_S2_eltcalc_P9 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P9 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P9 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P9 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P9 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P9 > output/gul_S1_summarycalc.csv & kpid9=$!
kat -s work/kat/gul_S2_eltcalc_P9 > output/gul_S2_eltcalc.csv & kpid10=$!
kat work/kat/gul_S2_pltcalc_P9 > output/gul_S2_pltcalc.csv & kpid11=$!
kat work/kat/gul_S2_summarycalc_P9 > output/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


check_complete
exit_handler
