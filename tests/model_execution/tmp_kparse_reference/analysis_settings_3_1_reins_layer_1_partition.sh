#!/bin/bash

set -e
set -o pipefail

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

mkdir work/gul_S1_summaryaalcalc
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkdir work/il_S1_summaryaalcalc
mkfifo fifo/ri_P1

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summaryeltcalc_P1
mkfifo fifo/ri_S1_eltcalc_P1
mkfifo fifo/ri_S1_summarysummarycalc_P1
mkfifo fifo/ri_S1_summarycalc_P1
mkfifo fifo/ri_S1_summarypltcalc_P1
mkfifo fifo/ri_S1_pltcalc_P1

mkdir work/ri_S1_summaryaalcalc


# --- Do reinsurance loss computes ---

eltcalc < fifo/ri_S1_summaryeltcalc_P1 > work/kat/ri_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/ri_S1_summarysummarycalc_P1 > work/kat/ri_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/ri_S1_summarypltcalc_P1 > work/kat/ri_S1_pltcalc_P1 & pid3=$!

tee < fifo/ri_S1_summary_P1 fifo/ri_S1_summaryeltcalc_P1 fifo/ri_S1_summarypltcalc_P1 fifo/ri_S1_summarysummarycalc_P1 work/ri_S1_summaryaalcalc/P1.bin > /dev/null & pid4=$!
summarycalc -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 &

# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid5=$!
summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid6=$!
pltcalc < fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid7=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 fifo/il_S1_summarypltcalc_P1 fifo/il_S1_summarysummarycalc_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid8=$!
summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid9=$!
summarycalctocsv < fifo/gul_S1_summarysummarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid10=$!
pltcalc < fifo/gul_S1_summarypltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid11=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryeltcalc_P1 fifo/gul_S1_summarypltcalc_P1 fifo/gul_S1_summarysummarycalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid12=$!
summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S0 -L0 -r -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 | tee fifo/il_P1 | fmcalc -a2 -n -p RI_1 > fifo/ri_P1 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12


# --- Do reinsurance loss kats ---

kat work/kat/ri_S1_eltcalc_P1 > output/ri_S1_eltcalc.csv & kpid1=$!
kat work/kat/ri_S1_pltcalc_P1 > output/ri_S1_pltcalc.csv & kpid2=$!
kat work/kat/ri_S1_summarycalc_P1 > output/ri_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid4=$!
kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid5=$!
kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid9=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
wait $lpid1 $lpid2 $lpid3


set +e

rm fifo/gul_P1

rm fifo/gul_S1_summary_P1
rm fifo/gul_S1_summaryeltcalc_P1
rm fifo/gul_S1_eltcalc_P1
rm fifo/gul_S1_summarysummarycalc_P1
rm fifo/gul_S1_summarycalc_P1
rm fifo/gul_S1_summarypltcalc_P1
rm fifo/gul_S1_pltcalc_P1

rm -rf work/kat
rm -rf work/gul_S1_summaryaalcalc/*
rmdir work/gul_S1_summaryaalcalc

rm fifo/ri_P1

rm fifo/ri_S1_summary_P1
rm fifo/ri_S1_summaryeltcalc_P1
rm fifo/ri_S1_eltcalc_P1
rm fifo/ri_S1_summarysummarycalc_P1
rm fifo/ri_S1_summarycalc_P1
rm fifo/ri_S1_summarypltcalc_P1
rm fifo/ri_S1_pltcalc_P1

rm -rf work/kat
rm -rf work/ri_S1_summaryaalcalc/*
rmdir work/ri_S1_summaryaalcalc
rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryeltcalc_P1
rm fifo/il_S1_eltcalc_P1
rm fifo/il_S1_summarysummarycalc_P1
rm fifo/il_S1_summarycalc_P1
rm fifo/il_S1_summarypltcalc_P1
rm fifo/il_S1_pltcalc_P1

rm -rf work/kat
rm -rf work/il_S1_summaryaalcalc/*
rmdir work/il_S1_summaryaalcalc
