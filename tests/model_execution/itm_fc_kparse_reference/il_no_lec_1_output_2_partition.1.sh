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

mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/gul_P2

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summaryeltcalc_P2
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarysummarycalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_summarypltcalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo il_S1_summary_P2
mkfifo il_S1_summaryeltcalc_P2
mkfifo il_S1_eltcalc_P2
mkfifo il_S1_summarysummarycalc_P2
mkfifo il_S1_summarycalc_P2
mkfifo il_S1_summarypltcalc_P2
mkfifo il_S1_pltcalc_P2



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid3=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_summaryeltcalc_P2 fifo/il_S1_summarypltcalc_P2 fifo/il_S1_summarysummarycalc_P2 work/il_S1_summaryaalcalc/P2.bin > /dev/null & pid4=$!
summarycalc -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

# --- Do ground up loss computes ---

eve 2 2 | getmodel | gulcalc -S0 -L0 -r -j gul_P2 -a1 -i - | tee fifo/gul_P2 | fmcalc -a2 > fifo/il_P2  &

wait $pid1 $pid2 $pid3 $pid4

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P2 > il_P2 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
eltcalc -s < il_S1_summaryeltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid1=$!
summarycalctocsv -s < il_S1_summarysummarycalc_P2 > work/full_correlation/kat/il_S1_summarycalc_P2 & pid2=$!
pltcalc -s < il_S1_summarypltcalc_P2 > work/full_correlation/kat/il_S1_pltcalc_P2 & pid3=$!
tee < il_S1_summary_P2 il_S1_summaryeltcalc_P2 il_S1_summarypltcalc_P2 il_S1_summarysummarycalc_P2 work/full_correlation/il_S1_summaryaalcalc/P2.bin > /dev/null & pid4=$!
summarycalc -f  -1 il_S1_summary_P2 < il_P2 &

# --- Do ground up loss computes ---

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P2 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P2 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P2 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P2 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P2 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P2 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

