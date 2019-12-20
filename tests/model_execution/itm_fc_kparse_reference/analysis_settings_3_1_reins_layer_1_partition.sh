#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/
mkfifo fifo/gul_P1
mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summaryeltcalc_P1
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarysummarycalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_summarypltcalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkdir work/gul_S1_summaryaalcalc
mkfifo fifo/full_correlation/gul_S1_summary_P1
mkfifo fifo/full_correlation/gul_S1_summaryeltcalc_P1
mkfifo fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo fifo/full_correlation/gul_S1_summarysummarycalc_P1
mkfifo fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo fifo/full_correlation/gul_S1_summarypltcalc_P1
mkfifo fifo/full_correlation/gul_S1_pltcalc_P1

mkdir work/full_correlation/gul_S1_summaryaalcalc
mkfifo fifo/il_P1
mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkdir work/il_S1_summaryaalcalc
mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P1
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarysummarycalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_summarypltcalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1

mkdir work/full_correlation/il_S1_summaryaalcalc
mkfifo fifo/ri_P1
mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summaryeltcalc_P1
mkfifo fifo/ri_S1_eltcalc_P1
mkfifo fifo/ri_S1_summarysummarycalc_P1
mkfifo fifo/ri_S1_summarycalc_P1
mkfifo fifo/ri_S1_summarypltcalc_P1
mkfifo fifo/ri_S1_pltcalc_P1

mkdir work/ri_S1_summaryaalcalc
mkfifo fifo/full_correlation/ri_S1_summary_P1
mkfifo fifo/full_correlation/ri_S1_summaryeltcalc_P1
mkfifo fifo/full_correlation/ri_S1_eltcalc_P1
mkfifo fifo/full_correlation/ri_S1_summarysummarycalc_P1
mkfifo fifo/full_correlation/ri_S1_summarycalc_P1
mkfifo fifo/full_correlation/ri_S1_summarypltcalc_P1
mkfifo fifo/full_correlation/ri_S1_pltcalc_P1

mkdir work/full_correlation/ri_S1_summaryaalcalc


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

eve 1 1 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 | tee fifo/il_P1 | fmcalc -a3 -n -p RI_1 > fifo/ri_P1 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12

# --- Do computes for fully correlated output ---

fmcalc -a2 < fifo/full_correlation/gul_P1 | tee fifo/full_correlation/il_P1 | fmcalc -a3 -n -p RI_1 > fifo/full_correlation/ri_P1 & fcpid1=$!

wait $fcpid1


# --- Do reinsurance loss computes ---

eltcalc < fifo/full_correlation/ri_S1_summaryeltcalc_P1 > work/full_correlation/kat/ri_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/full_correlation/ri_S1_summarysummarycalc_P1 > work/full_correlation/kat/ri_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/full_correlation/ri_S1_summarypltcalc_P1 > work/full_correlation/kat/ri_S1_pltcalc_P1 & pid3=$!

tee < fifo/full_correlation/ri_S1_summary_P1 fifo/full_correlation/ri_S1_summaryeltcalc_P1 fifo/full_correlation/ri_S1_summarypltcalc_P1 fifo/full_correlation/ri_S1_summarysummarycalc_P1 work/full_correlation/ri_S1_summaryaalcalc/P1.bin > /dev/null & pid4=$!

summarycalc -f -p RI_1 -1 fifo/full_correlation/ri_S1_summary_P1 < fifo/full_correlation/ri_P1 &

# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_summaryeltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid5=$!
summarycalctocsv < fifo/full_correlation/il_S1_summarysummarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid6=$!
pltcalc < fifo/full_correlation/il_S1_summarypltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid7=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_summaryeltcalc_P1 fifo/full_correlation/il_S1_summarypltcalc_P1 fifo/full_correlation/il_S1_summarysummarycalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin > /dev/null & pid8=$!

summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &

# --- Do ground up loss computes ---

eltcalc < fifo/full_correlation/gul_S1_summaryeltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid9=$!
summarycalctocsv < fifo/full_correlation/gul_S1_summarysummarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid10=$!
pltcalc < fifo/full_correlation/gul_S1_summarypltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid11=$!

tee < fifo/full_correlation/gul_S1_summary_P1 fifo/full_correlation/gul_S1_summaryeltcalc_P1 fifo/full_correlation/gul_S1_summarypltcalc_P1 fifo/full_correlation/gul_S1_summarysummarycalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid12=$!

summarycalc -i  -1 fifo/full_correlation/gul_S1_summary_P1 < fifo/full_correlation/gul_P1 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12


# --- Do reinsurance loss kats ---

kat work/kat/ri_S1_eltcalc_P1 > output/ri_S1_eltcalc.csv & kpid1=$!
kat work/kat/ri_S1_pltcalc_P1 > output/ri_S1_pltcalc.csv & kpid2=$!
kat work/kat/ri_S1_summarycalc_P1 > output/ri_S1_summarycalc.csv & kpid3=$!

# --- Do reinsurance loss kats for fully correlated output ---

kat work/full_correlation/kat/ri_S1_eltcalc_P1 > output/full_correlation/ri_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/ri_S1_pltcalc_P1 > output/full_correlation/ri_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/ri_S1_summarycalc_P1 > output/full_correlation/ri_S1_summarycalc.csv & kpid6=$!

# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid7=$!
kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid8=$!
kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid9=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 > output/full_correlation/il_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 > output/full_correlation/il_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 > output/full_correlation/il_S1_summarycalc.csv & kpid12=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid13=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid14=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid15=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P1 > output/full_correlation/gul_S1_eltcalc.csv & kpid16=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 > output/full_correlation/gul_S1_pltcalc.csv & kpid17=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 > output/full_correlation/gul_S1_summarycalc.csv & kpid18=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
aalcalc -Kfull_correlation/ri_S1_summaryaalcalc > output/full_correlation/ri_S1_aalcalc.csv & lpid4=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid5=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid6=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6

rm -R -f work/*
rm -R -f fifo/*
