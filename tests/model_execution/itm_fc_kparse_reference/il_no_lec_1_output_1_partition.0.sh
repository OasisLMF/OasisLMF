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

mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/gul_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_summarypltcalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summaryeltcalc_P1
mkfifo fifo/full_correlation/il_S1_eltcalc_P1
mkfifo fifo/full_correlation/il_S1_summarysummarycalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1
mkfifo fifo/full_correlation/il_S1_summarypltcalc_P1
mkfifo fifo/full_correlation/il_S1_pltcalc_P1



# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/il_S1_summarypltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid3=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 fifo/il_S1_summarypltcalc_P1 fifo/il_S1_summarysummarycalc_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid4=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

# --- Do ground up loss computes ---

eve 1 1 | getmodel | gulcalc -S0 -L0 -r -j fifo/full_correlation/gul_P1 -a1 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2 $pid3 $pid4

# --- Do computes for fully correlated output ---

fmcalc-a2 < fifo/full_correlation/gul_P1 > fifo/full_correlation/il_P1 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---

eltcalc < fifo/full_correlation/il_S1_summaryeltcalc_P1 > work/full_correlation/kat/il_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/full_correlation/il_S1_summarysummarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/full_correlation/il_S1_summarypltcalc_P1 > work/full_correlation/kat/il_S1_pltcalc_P1 & pid3=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_summaryeltcalc_P1 fifo/full_correlation/il_S1_summarypltcalc_P1 fifo/full_correlation/il_S1_summarysummarycalc_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin > /dev/null & pid4=$!

summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &

# --- Do ground up loss computes ---

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P1 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P1 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P1 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

