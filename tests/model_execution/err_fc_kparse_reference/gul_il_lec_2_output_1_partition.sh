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
   printf "Script PID:%d, GPID:%s, SPID:%d" $$ $group_pid $sess_pid >> log/killout.txt

   ps f -g $sess_pid > log/subprocess_list
   PIDS_KILL=$(pgrep -a --pgroup $group_pid | grep -v celery | grep -v python | grep -v $group_pid | grep -v run_ktools)
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
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S2_summaryleccalc
mkdir work/full_correlation/gul_S2_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S2_summaryleccalc
mkdir work/full_correlation/il_S2_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P1

mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_pltcalc_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1
mkfifo fifo/il_S2_summary_P1
mkfifo fifo/il_S2_eltcalc_P1
mkfifo fifo/il_S2_summarycalc_P1
mkfifo fifo/il_S2_pltcalc_P1

mkfifo fifo/full_correlation/gul_P1

mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo fifo/full_correlation/gul_S1_pltcalc_P1
mkfifo fifo/full_correlation/gul_S2_summary_P1
mkfifo fifo/full_correlation/gul_S2_eltcalc_P1
mkfifo fifo/full_correlation/gul_S2_summarycalc_P1
mkfifo fifo/full_correlation/gul_S2_pltcalc_P1

mkfifo fifo/full_correlation/il_P1

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1
mkfifo fifo/full_correlation/il_S2_summary_P1
mkfifo fifo/full_correlation/il_S2_eltcalc_P1
mkfifo fifo/full_correlation/il_S2_summarycalc_P1
mkfifo fifo/full_correlation/il_S2_pltcalc_P1



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!
eltcalc < fifo/il_S2_eltcalc_P1 > work/kat/il_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < fifo/il_S2_summarycalc_P1 > work/kat/il_S2_summarycalc_P1 & pid5=$!
pltcalc < fifo/il_S2_pltcalc_P1 > work/kat/il_S2_pltcalc_P1 & pid6=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid7=$!
tee < fifo/il_S2_summary_P1 fifo/il_S2_eltcalc_P1 fifo/il_S2_summarycalc_P1 fifo/il_S2_pltcalc_P1 work/il_S2_summaryaalcalc/P1.bin work/il_S2_summaryleccalc/P1.bin > /dev/null & pid8=$!

( summarycalc -f  -1 fifo/il_S1_summary_P1 -2 fifo/il_S2_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid9=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid10=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid11=$!
eltcalc < fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid12=$!
summarycalctocsv < fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid13=$!
pltcalc < fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid14=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_eltcalc_P1 fifo/gul_S2_summarycalc_P1 fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid16=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &

# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_eltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid17=$!
summarycalctocsv < fifo/full_correlation/il_S1_summarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid18=$!
pltcalc < fifo/full_correlation/il_S1_pltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid19=$!
eltcalc < fifo/full_correlation/il_S2_eltcalc_P1 > work/full_correlation/kat/il_S2_eltcalc_P1 & pid20=$!
summarycalctocsv < fifo/full_correlation/il_S2_summarycalc_P1 > work/full_correlation/kat/il_S2_summarycalc_P1 & pid21=$!
pltcalc < fifo/full_correlation/il_S2_pltcalc_P1 > work/full_correlation/kat/il_S2_pltcalc_P1 & pid22=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_eltcalc_P1 fifo/full_correlation/il_S1_summarycalc_P1 fifo/full_correlation/il_S1_pltcalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin work/full_correlation/il_S1_summaryleccalc/P1.bin > /dev/null & pid23=$!
tee < fifo/full_correlation/il_S2_summary_P1 fifo/full_correlation/il_S2_eltcalc_P1 fifo/full_correlation/il_S2_summarycalc_P1 fifo/full_correlation/il_S2_pltcalc_P1 work/full_correlation/il_S2_summaryaalcalc/P1.bin work/full_correlation/il_S2_summaryleccalc/P1.bin > /dev/null & pid24=$!

( summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 -2 fifo/full_correlation/il_S2_summary_P1 < fifo/full_correlation/il_P1 ) 2>> log/stderror.err  &

# --- Do ground up loss computes ---

eltcalc < fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid25=$!
summarycalctocsv < fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid26=$!
pltcalc < fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid27=$!
eltcalc < fifo/full_correlation/gul_S2_eltcalc_P1 > work/full_correlation/kat/gul_S2_eltcalc_P1 & pid28=$!
summarycalctocsv < fifo/full_correlation/gul_S2_summarycalc_P1 > work/full_correlation/kat/gul_S2_summarycalc_P1 & pid29=$!
pltcalc < fifo/full_correlation/gul_S2_pltcalc_P1 > work/full_correlation/kat/gul_S2_pltcalc_P1 & pid30=$!

tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_eltcalc_P1 fifo/full_correlation/gul_S1_summarycalc_P1 fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin work/full_correlation/gul_S1_summaryleccalc/P1.bin > /dev/null & pid31=$!
tee < fifo/full_correlation/gul_S2_summary_P1 fifo/full_correlation/gul_S2_eltcalc_P1 fifo/full_correlation/gul_S2_summarycalc_P1 fifo/full_correlation/gul_S2_pltcalc_P1 work/full_correlation/gul_S2_summaryaalcalc/P1.bin work/full_correlation/gul_S2_summaryleccalc/P1.bin > /dev/null & pid32=$!

( summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 -2 fifo/full_correlation/gul_S2_summary_P1 < fifo/full_correlation/gul_P1 ) 2>> log/stderror.err  &

( tee < fifo/full_correlation/gul_fc_P1 fifo/full_correlation/gul_P1  | fmcalc -a2 > fifo/full_correlation/il_P1  ) 2>> log/stderror.err &
( eve 1 1 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P1 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P1 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P1 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 > output/full_correlation/il_S1_eltcalc.csv & kpid7=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 > output/full_correlation/il_S1_pltcalc.csv & kpid8=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 > output/full_correlation/il_S1_summarycalc.csv & kpid9=$!
kat work/full_correlation/kat/il_S2_eltcalc_P1 > output/full_correlation/il_S2_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/il_S2_pltcalc_P1 > output/full_correlation/il_S2_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/il_S2_summarycalc_P1 > output/full_correlation/il_S2_summarycalc.csv & kpid12=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid13=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid14=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid15=$!
kat work/kat/gul_S2_eltcalc_P1 > output/gul_S2_eltcalc.csv & kpid16=$!
kat work/kat/gul_S2_pltcalc_P1 > output/gul_S2_pltcalc.csv & kpid17=$!
kat work/kat/gul_S2_summarycalc_P1 > output/gul_S2_summarycalc.csv & kpid18=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P1 > output/full_correlation/gul_S1_eltcalc.csv & kpid19=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 > output/full_correlation/gul_S1_pltcalc.csv & kpid20=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 > output/full_correlation/gul_S1_summarycalc.csv & kpid21=$!
kat work/full_correlation/kat/gul_S2_eltcalc_P1 > output/full_correlation/gul_S2_eltcalc.csv & kpid22=$!
kat work/full_correlation/kat/gul_S2_pltcalc_P1 > output/full_correlation/gul_S2_pltcalc.csv & kpid23=$!
kat work/full_correlation/kat/gul_S2_summarycalc_P1 > output/full_correlation/gul_S2_summarycalc.csv & kpid24=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18 $kpid19 $kpid20 $kpid21 $kpid22 $kpid23 $kpid24


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv -S output/il_S1_leccalc_sample_mean_aep.csv -s output/il_S1_leccalc_sample_mean_oep.csv -W output/il_S1_leccalc_wheatsheaf_aep.csv -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kil_S2_summaryaalcalc > output/il_S2_aalcalc.csv & lpid3=$!
leccalc -r -Kil_S2_summaryleccalc -F output/il_S2_leccalc_full_uncertainty_aep.csv -f output/il_S2_leccalc_full_uncertainty_oep.csv -S output/il_S2_leccalc_sample_mean_aep.csv -s output/il_S2_leccalc_sample_mean_oep.csv -W output/il_S2_leccalc_wheatsheaf_aep.csv -M output/il_S2_leccalc_wheatsheaf_mean_aep.csv -m output/il_S2_leccalc_wheatsheaf_mean_oep.csv -w output/il_S2_leccalc_wheatsheaf_oep.csv & lpid4=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid7=$!
leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv & lpid8=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid9=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -F output/full_correlation/il_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/il_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/il_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/il_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/il_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/il_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/il_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/il_S1_leccalc_wheatsheaf_oep.csv & lpid10=$!
aalcalc -Kfull_correlation/il_S2_summaryaalcalc > output/full_correlation/il_S2_aalcalc.csv & lpid11=$!
leccalc -r -Kfull_correlation/il_S2_summaryleccalc -F output/full_correlation/il_S2_leccalc_full_uncertainty_aep.csv -f output/full_correlation/il_S2_leccalc_full_uncertainty_oep.csv -S output/full_correlation/il_S2_leccalc_sample_mean_aep.csv -s output/full_correlation/il_S2_leccalc_sample_mean_oep.csv -W output/full_correlation/il_S2_leccalc_wheatsheaf_aep.csv -M output/full_correlation/il_S2_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/il_S2_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/il_S2_leccalc_wheatsheaf_oep.csv & lpid12=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid13=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv & lpid14=$!
aalcalc -Kfull_correlation/gul_S2_summaryaalcalc > output/full_correlation/gul_S2_aalcalc.csv & lpid15=$!
leccalc -r -Kfull_correlation/gul_S2_summaryleccalc -F output/full_correlation/gul_S2_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S2_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S2_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S2_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S2_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S2_leccalc_wheatsheaf_oep.csv & lpid16=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8 $lpid9 $lpid10 $lpid11 $lpid12 $lpid13 $lpid14 $lpid15 $lpid16

rm -R -f work/*
rm -R -f fifo/*

check_complete
exit_handler
