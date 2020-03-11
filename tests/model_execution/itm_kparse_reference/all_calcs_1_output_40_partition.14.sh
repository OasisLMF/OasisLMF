#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

mkfifo fifo/gul_P15

mkfifo fifo/il_P15

mkfifo fifo/gul_S1_summary_P15
mkfifo fifo/gul_S1_summaryeltcalc_P15
mkfifo fifo/gul_S1_eltcalc_P15
mkfifo fifo/gul_S1_summarysummarycalc_P15
mkfifo fifo/gul_S1_summarycalc_P15
mkfifo fifo/gul_S1_summarypltcalc_P15
mkfifo fifo/gul_S1_pltcalc_P15

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summaryeltcalc_P15
mkfifo fifo/il_S1_eltcalc_P15
mkfifo fifo/il_S1_summarysummarycalc_P15
mkfifo fifo/il_S1_summarycalc_P15
mkfifo fifo/il_S1_summarypltcalc_P15
mkfifo fifo/il_S1_pltcalc_P15



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P15 > work/kat/il_S1_summarycalc_P15 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid3=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_summaryeltcalc_P15 fifo/il_S1_summarypltcalc_P15 fifo/il_S1_summarysummarycalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid4=$!
summarycalc -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P15 > work/kat/gul_S1_eltcalc_P15 & pid5=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid6=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P15 > work/kat/gul_S1_pltcalc_P15 & pid7=$!
tee < fifo/gul_S1_summary_P15 fifo/gul_S1_summaryeltcalc_P15 fifo/gul_S1_summarypltcalc_P15 fifo/gul_S1_summarysummarycalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid8=$!
summarycalc -i  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 &

eve 15 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P15 | fmcalc -a2 > fifo/il_P15  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P15 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P15 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P15 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P15 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P15 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P15 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

