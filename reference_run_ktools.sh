#!/bin/bash

set -e
set -o pipefail


handler_exit(){
    proc_group_id=$(ps -p $$ -o pgid --no-headers)
    proc_ktools=($(pgrep --pgroup $proc_group_id))

    pgrep -a --pgroup $proc_group_id
    pkill -9 --pgroup $proc_group_id

}    
trap handler_exit QUIT HUP INT KILL TERM ERR


rm -f stderror.err
rm -f killout.txt
touch stderror.err
ktools_monitor $$ & pid0=$!

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
rm -R -f work/*

mkdir work/kat
rm -R -f /tmp/ryaj5HHwcc/
mkdir -p /tmp/ryaj5HHwcc/fifo
mkfifo /tmp/ryaj5HHwcc/fifo/gul_P1

mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P1
mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P1
mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_eltcalc_P1

mkfifo /tmp/ryaj5HHwcc/fifo/gul_P2

mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P2
mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P2
mkfifo /tmp/ryaj5HHwcc/fifo/gul_S1_eltcalc_P2

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkfifo /tmp/ryaj5HHwcc/fifo/il_P1

mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_summary_P1
mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P1
mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_eltcalc_P1

mkfifo /tmp/ryaj5HHwcc/fifo/il_P2

mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_summary_P2
mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P2
mkfifo /tmp/ryaj5HHwcc/fifo/il_S1_eltcalc_P2

mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkfifo /tmp/ryaj5HHwcc/fifo/ri_P1

mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P1
mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P1
mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_eltcalc_P1

mkfifo /tmp/ryaj5HHwcc/fifo/ri_P2

mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P2
mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P2
mkfifo /tmp/ryaj5HHwcc/fifo/ri_S1_eltcalc_P2

mkdir work/ri_S1_summaryleccalc
mkdir work/ri_S1_summaryaalcalc


# --- Do reinsurance loss computes ---

eltcalc < /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P1 > work/kat/ri_S1_eltcalc_P1 & pid1=$!

eltcalc -s < /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P2 > work/kat/ri_S1_eltcalc_P2 & pid2=$!

tee < /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P1 /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P1 work/ri_S1_summaryaalcalc/P1.bin work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid3=$!
tee < /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P2 /tmp/ryaj5HHwcc/fifo/ri_S1_summaryeltcalc_P2 work/ri_S1_summaryaalcalc/P2.bin work/ri_S1_summaryleccalc/P2.bin > /dev/null & pid4=$!
( summarycalc -f -p RI_1 -1 /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P1 < /tmp/ryaj5HHwcc/fifo/ri_P1 ) 2>> stderror.err  &
( summarycalc -f -p RI_1 -1 /tmp/ryaj5HHwcc/fifo/ri_S1_summary_P2 < /tmp/ryaj5HHwcc/fifo/ri_P2 ) 2>> stderror.err  &

# --- Do insured loss computes ---

eltcalc < /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid5=$!

eltcalc -s < /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid6=$!

tee < /tmp/ryaj5HHwcc/fifo/il_S1_summary_P1 /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid7=$!
tee < /tmp/ryaj5HHwcc/fifo/il_S1_summary_P2 /tmp/ryaj5HHwcc/fifo/il_S1_summaryeltcalc_P2 work/il_S1_summaryaalcalc/P2.bin work/il_S1_summaryleccalc/P2.bin > /dev/null & pid8=$!
( summarycalc -f  -1 /tmp/ryaj5HHwcc/fifo/il_S1_summary_P1 < /tmp/ryaj5HHwcc/fifo/il_P1 ) 2>> stderror.err  &
( summarycalc -f  -1 /tmp/ryaj5HHwcc/fifo/il_S1_summary_P2 < /tmp/ryaj5HHwcc/fifo/il_P2 ) 2>> stderror.err  &

# --- Do ground up loss computes ---

eltcalc < /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid9=$!

eltcalc -s < /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid10=$!

tee < /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P1 /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid11=$!
tee < /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P2 /tmp/ryaj5HHwcc/fifo/gul_S1_summaryeltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin work/gul_S1_summaryleccalc/P2.bin > /dev/null & pid12=$!
( summarycalc -i  -1 /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P1 < /tmp/ryaj5HHwcc/fifo/gul_P1 ) 2>> stderror.err  &
( summarycalc -i  -1 /tmp/ryaj5HHwcc/fifo/gul_S1_summary_P2 < /tmp/ryaj5HHwcc/fifo/gul_P2 ) 2>> stderror.err  &

( eve 1 2 | getmodel | gulcalc -S10 -L0 -a1 -i - | tee /tmp/ryaj5HHwcc/fifo/gul_P1 | fmcalc -a2 | tee /tmp/ryaj5HHwcc/fifo/il_P1 | fmcalc -a2 -n -p RI_1 > /tmp/ryaj5HHwcc/fifo/ri_P1 ) 2>> stderror.err  &
( eve 2 2 | getmodel | gulcalc -S10 -L0 -a1 -i - | tee /tmp/ryaj5HHwcc/fifo/gul_P2 | fmcalc -a2 | tee /tmp/ryaj5HHwcc/fifo/il_P2 | fmcalc -a2 -n -p RI_1 > /tmp/ryaj5HHwcc/fifo/ri_P2 ) 2>> stderror.err  &
wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12

# --- Do reinsurance loss kats ---
false
kat work/kat/ri_S1_eltcalc_P1 work/kat/ri_S1_eltcalc_P2 > output/ri_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 work/kat/il_S1_eltcalc_P2 > output/il_S1_eltcalc.csv & kpid2=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid3=$!
wait $kpid1 $kpid2 $kpid3


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kri_S1_summaryleccalc -F output/ri_S1_leccalc_full_uncertainty_aep.csv -f output/ri_S1_leccalc_full_uncertainty_oep.csv & lpid2=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid3=$!
leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv & lpid4=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv & lpid6=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6

kill $pid0
rm -rf  work/* 
rm -rf /tmp/Jqvb6PZ4m8
