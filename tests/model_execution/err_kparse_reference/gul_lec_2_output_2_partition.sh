#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail

error_handler(){
    echo 'Run Error - terminating, see the log dir for details'
    proc_group_id=$(ps -p $$ -o pgid --no-headers)
    pgrep -a --pgroup $proc_group_id >> log/killout.txt
    pkill -9 --pgroup $proc_group_id
}
trap error_handler QUIT HUP INT KILL TERM ERR

mkdir -p log
rm -R -f log/*
touch log/stderror.err
ktools_monitor $$ & pid0=$!

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/gul_P1
mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summaryeltcalc_P1
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarysummarycalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_summarypltcalc_P1
mkfifo fifo/gul_S1_pltcalc_P1
mkfifo fifo/gul_S2_summary_P1
mkfifo fifo/gul_S2_summaryeltcalc_P1
mkfifo fifo/gul_S2_eltcalc_P1
mkfifo fifo/gul_S2_summarysummarycalc_P1
mkfifo fifo/gul_S2_summarycalc_P1
mkfifo fifo/gul_S2_summarypltcalc_P1
mkfifo fifo/gul_S2_pltcalc_P1

mkfifo fifo/gul_P2
mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summaryeltcalc_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarysummarycalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_summarypltcalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summaryeltcalc_P2
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarysummarycalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_summarypltcalc_P2
mkfifo fifo/gul_S2_pltcalc_P2

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/gul_S1_summarysummarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/gul_S1_summarypltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid3=$!
eltcalc < fifo/gul_S2_summaryeltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < fifo/gul_S2_summarysummarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid5=$!
pltcalc < fifo/gul_S2_summarypltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid6=$!
eltcalc -s < fifo/gul_S1_summaryeltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid8=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid9=$!
eltcalc -s < fifo/gul_S2_summaryeltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid11=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid12=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryeltcalc_P1 fifo/gul_S1_summarypltcalc_P1 fifo/gul_S1_summarysummarycalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid13=$!
tee < fifo/gul_S2_summary_P1 fifo/gul_S2_summaryeltcalc_P1 fifo/gul_S2_summarypltcalc_P1 fifo/gul_S2_summarysummarycalc_P1 work/gul_S2_summaryaalcalc/P1.bin work/gul_S2_summaryleccalc/P1.bin > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_summaryeltcalc_P2 fifo/gul_S1_summarypltcalc_P2 fifo/gul_S1_summarysummarycalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_summaryeltcalc_P2 fifo/gul_S2_summarypltcalc_P2 fifo/gul_S2_summarysummarycalc_P2 work/gul_S2_summaryaalcalc/P2.bin work/gul_S2_summaryleccalc/P2.bin > /dev/null & pid16=$!

( summarycalc -i  -1 fifo/gul_S1_summary_P1 -2 fifo/gul_S2_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err  &
( summarycalc -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 ) 2>> log/stderror.err  &

( eve 1 2 | getmodel | gulcalc -S0 -L0 -r -a1 -i - > fifo/gul_P1  ) 2>> log/stderror.err &
( eve 2 2 | getmodel | gulcalc -S0 -L0 -r -a1 -i - > fifo/gul_P2  ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid3=$!
kat work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 > output/gul_S2_eltcalc.csv & kpid4=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 > output/gul_S2_pltcalc.csv & kpid5=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 > output/gul_S2_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid3=$!
leccalc -r -Kgul_S2_summaryleccalc -F output/gul_S2_leccalc_full_uncertainty_aep.csv -f output/gul_S2_leccalc_full_uncertainty_oep.csv -S output/gul_S2_leccalc_sample_mean_aep.csv -s output/gul_S2_leccalc_sample_mean_oep.csv -W output/gul_S2_leccalc_wheatsheaf_aep.csv -M output/gul_S2_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S2_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S2_leccalc_wheatsheaf_oep.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*

# Stop ktools watcher
kill -9 $pid0
