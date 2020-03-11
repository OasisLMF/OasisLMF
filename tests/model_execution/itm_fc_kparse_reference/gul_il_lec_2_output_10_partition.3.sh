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

mkfifo fifo/gul_P4

mkfifo fifo/il_P4

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_summaryeltcalc_P4
mkfifo fifo/gul_S1_eltcalc_P4
mkfifo fifo/gul_S1_summarysummarycalc_P4
mkfifo fifo/gul_S1_summarycalc_P4
mkfifo fifo/gul_S1_summarypltcalc_P4
mkfifo fifo/gul_S1_pltcalc_P4
mkfifo fifo/gul_S2_summary_P4
mkfifo fifo/gul_S2_summaryeltcalc_P4
mkfifo fifo/gul_S2_eltcalc_P4
mkfifo fifo/gul_S2_summarysummarycalc_P4
mkfifo fifo/gul_S2_summarycalc_P4
mkfifo fifo/gul_S2_summarypltcalc_P4
mkfifo fifo/gul_S2_pltcalc_P4

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summaryeltcalc_P4
mkfifo fifo/il_S1_eltcalc_P4
mkfifo fifo/il_S1_summarysummarycalc_P4
mkfifo fifo/il_S1_summarycalc_P4
mkfifo fifo/il_S1_summarypltcalc_P4
mkfifo fifo/il_S1_pltcalc_P4
mkfifo fifo/il_S2_summary_P4
mkfifo fifo/il_S2_summaryeltcalc_P4
mkfifo fifo/il_S2_eltcalc_P4
mkfifo fifo/il_S2_summarysummarycalc_P4
mkfifo fifo/il_S2_summarycalc_P4
mkfifo fifo/il_S2_summarypltcalc_P4
mkfifo fifo/il_S2_pltcalc_P4

mkfifo gul_S1_summary_P4
mkfifo gul_S1_summaryeltcalc_P4
mkfifo gul_S1_eltcalc_P4
mkfifo gul_S1_summarysummarycalc_P4
mkfifo gul_S1_summarycalc_P4
mkfifo gul_S1_summarypltcalc_P4
mkfifo gul_S1_pltcalc_P4
mkfifo gul_S2_summary_P4
mkfifo gul_S2_summaryeltcalc_P4
mkfifo gul_S2_eltcalc_P4
mkfifo gul_S2_summarysummarycalc_P4
mkfifo gul_S2_summarycalc_P4
mkfifo gul_S2_summarypltcalc_P4
mkfifo gul_S2_pltcalc_P4

mkfifo il_S1_summary_P4
mkfifo il_S1_summaryeltcalc_P4
mkfifo il_S1_eltcalc_P4
mkfifo il_S1_summarysummarycalc_P4
mkfifo il_S1_summarycalc_P4
mkfifo il_S1_summarypltcalc_P4
mkfifo il_S1_pltcalc_P4
mkfifo il_S2_summary_P4
mkfifo il_S2_summaryeltcalc_P4
mkfifo il_S2_eltcalc_P4
mkfifo il_S2_summarysummarycalc_P4
mkfifo il_S2_summarycalc_P4
mkfifo il_S2_summarypltcalc_P4
mkfifo il_S2_pltcalc_P4



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P4 > work/kat/il_S1_eltcalc_P4 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P4 > work/kat/il_S1_pltcalc_P4 & pid3=$!
eltcalc -s < fifo/il_S2_summaryeltcalc_P4 > work/kat/il_S2_eltcalc_P4 & pid4=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P4 > work/kat/il_S2_summarycalc_P4 & pid5=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P4 > work/kat/il_S2_pltcalc_P4 & pid6=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summaryeltcalc_P4 fifo/il_S1_summarypltcalc_P4 fifo/il_S1_summarysummarycalc_P4 work/il_S1_summaryaalcalc/P4.bin work/il_S1_summaryleccalc/P4.bin > /dev/null & pid7=$!
tee < fifo/il_S2_summary_P4 fifo/il_S2_summaryeltcalc_P4 fifo/il_S2_summarypltcalc_P4 fifo/il_S2_summarysummarycalc_P4 work/il_S2_summaryaalcalc/P4.bin work/il_S2_summaryleccalc/P4.bin > /dev/null & pid8=$!
summarycalc -f  -1 fifo/il_S1_summary_P4 -2 fifo/il_S2_summary_P4 < fifo/il_P4 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid9=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P4 > work/kat/gul_S1_summarycalc_P4 & pid10=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P4 > work/kat/gul_S1_pltcalc_P4 & pid11=$!
eltcalc -s < fifo/gul_S2_summaryeltcalc_P4 > work/kat/gul_S2_eltcalc_P4 & pid12=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P4 > work/kat/gul_S2_summarycalc_P4 & pid13=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P4 > work/kat/gul_S2_pltcalc_P4 & pid14=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_summaryeltcalc_P4 fifo/gul_S1_summarypltcalc_P4 fifo/gul_S1_summarysummarycalc_P4 work/gul_S1_summaryaalcalc/P4.bin work/gul_S1_summaryleccalc/P4.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P4 fifo/gul_S2_summaryeltcalc_P4 fifo/gul_S2_summarypltcalc_P4 fifo/gul_S2_summarysummarycalc_P4 work/gul_S2_summaryaalcalc/P4.bin work/gul_S2_summaryleccalc/P4.bin > /dev/null & pid16=$!
summarycalc -i  -1 fifo/gul_S1_summary_P4 -2 fifo/gul_S2_summary_P4 < fifo/gul_P4 &

eve 4 10 | getmodel | gulcalc -S0 -L0 -r -j gul_P4 -a1 -i - | tee fifo/gul_P4 | fmcalc -a2 > fifo/il_P4  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P4 > il_P4 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
eltcalc -s < il_S1_summaryeltcalc_P4 > work/full_correlation/kat/il_S1_eltcalc_P4 & pid1=$!
summarycalctocsv -s < il_S1_summarysummarycalc_P4 > work/full_correlation/kat/il_S1_summarycalc_P4 & pid2=$!
pltcalc -s < il_S1_summarypltcalc_P4 > work/full_correlation/kat/il_S1_pltcalc_P4 & pid3=$!
eltcalc -s < il_S2_summaryeltcalc_P4 > work/full_correlation/kat/il_S2_eltcalc_P4 & pid4=$!
summarycalctocsv -s < il_S2_summarysummarycalc_P4 > work/full_correlation/kat/il_S2_summarycalc_P4 & pid5=$!
pltcalc -s < il_S2_summarypltcalc_P4 > work/full_correlation/kat/il_S2_pltcalc_P4 & pid6=$!
tee < il_S1_summary_P4 il_S1_summaryeltcalc_P4 il_S1_summarypltcalc_P4 il_S1_summarysummarycalc_P4 work/full_correlation/il_S1_summaryaalcalc/P4.bin work/full_correlation/il_S1_summaryleccalc/P4.bin > /dev/null & pid7=$!
tee < il_S2_summary_P4 il_S2_summaryeltcalc_P4 il_S2_summarypltcalc_P4 il_S2_summarysummarycalc_P4 work/full_correlation/il_S2_summaryaalcalc/P4.bin work/full_correlation/il_S2_summaryleccalc/P4.bin > /dev/null & pid8=$!
summarycalc -f  -1 il_S1_summary_P4 -2 il_S2_summary_P4 < il_P4 &

# --- Do ground up loss computes ---
eltcalc -s < gul_S1_summaryeltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid9=$!
summarycalctocsv -s < gul_S1_summarysummarycalc_P4 > work/full_correlation/kat/gul_S1_summarycalc_P4 & pid10=$!
pltcalc -s < gul_S1_summarypltcalc_P4 > work/full_correlation/kat/gul_S1_pltcalc_P4 & pid11=$!
eltcalc -s < gul_S2_summaryeltcalc_P4 > work/full_correlation/kat/gul_S2_eltcalc_P4 & pid12=$!
summarycalctocsv -s < gul_S2_summarysummarycalc_P4 > work/full_correlation/kat/gul_S2_summarycalc_P4 & pid13=$!
pltcalc -s < gul_S2_summarypltcalc_P4 > work/full_correlation/kat/gul_S2_pltcalc_P4 & pid14=$!
tee < gul_S1_summary_P4 gul_S1_summaryeltcalc_P4 gul_S1_summarypltcalc_P4 gul_S1_summarysummarycalc_P4 work/full_correlation/gul_S1_summaryaalcalc/P4.bin work/full_correlation/gul_S1_summaryleccalc/P4.bin > /dev/null & pid15=$!
tee < gul_S2_summary_P4 gul_S2_summaryeltcalc_P4 gul_S2_summarypltcalc_P4 gul_S2_summarysummarycalc_P4 work/full_correlation/gul_S2_summaryaalcalc/P4.bin work/full_correlation/gul_S2_summaryleccalc/P4.bin > /dev/null & pid16=$!
summarycalc -i  -1 gul_S1_summary_P4 -2 gul_S2_summary_P4 < gul_P4 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P4 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P4 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P4 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P4 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P4 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P4 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P4 > output/full_correlation/il_S1_eltcalc.csv & kpid7=$!
kat work/full_correlation/kat/il_S1_pltcalc_P4 > output/full_correlation/il_S1_pltcalc.csv & kpid8=$!
kat work/full_correlation/kat/il_S1_summarycalc_P4 > output/full_correlation/il_S1_summarycalc.csv & kpid9=$!
kat work/full_correlation/kat/il_S2_eltcalc_P4 > output/full_correlation/il_S2_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/il_S2_pltcalc_P4 > output/full_correlation/il_S2_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/il_S2_summarycalc_P4 > output/full_correlation/il_S2_summarycalc.csv & kpid12=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P4 > output/gul_S1_eltcalc.csv & kpid13=$!
kat work/kat/gul_S1_pltcalc_P4 > output/gul_S1_pltcalc.csv & kpid14=$!
kat work/kat/gul_S1_summarycalc_P4 > output/gul_S1_summarycalc.csv & kpid15=$!
kat work/kat/gul_S2_eltcalc_P4 > output/gul_S2_eltcalc.csv & kpid16=$!
kat work/kat/gul_S2_pltcalc_P4 > output/gul_S2_pltcalc.csv & kpid17=$!
kat work/kat/gul_S2_summarycalc_P4 > output/gul_S2_summarycalc.csv & kpid18=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P4 > output/full_correlation/gul_S1_eltcalc.csv & kpid19=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P4 > output/full_correlation/gul_S1_pltcalc.csv & kpid20=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P4 > output/full_correlation/gul_S1_summarycalc.csv & kpid21=$!
kat work/full_correlation/kat/gul_S2_eltcalc_P4 > output/full_correlation/gul_S2_eltcalc.csv & kpid22=$!
kat work/full_correlation/kat/gul_S2_pltcalc_P4 > output/full_correlation/gul_S2_pltcalc.csv & kpid23=$!
kat work/full_correlation/kat/gul_S2_summarycalc_P4 > output/full_correlation/gul_S2_summarycalc.csv & kpid24=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12 $kpid13 $kpid14 $kpid15 $kpid16 $kpid17 $kpid18 $kpid19 $kpid20 $kpid21 $kpid22 $kpid23 $kpid24

