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
mkdir work/gul_S2_summaryleccalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/il_S2_summaryleccalc
mkdir work/il_S2_summaryaalcalc

mkfifo fifo/gul_P3

mkfifo fifo/il_P3

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_summaryeltcalc_P3
mkfifo fifo/gul_S1_eltcalc_P3
mkfifo fifo/gul_S1_summarysummarycalc_P3
mkfifo fifo/gul_S1_summarycalc_P3
mkfifo fifo/gul_S1_summarypltcalc_P3
mkfifo fifo/gul_S1_pltcalc_P3
mkfifo fifo/gul_S2_summary_P3
mkfifo fifo/gul_S2_summaryeltcalc_P3
mkfifo fifo/gul_S2_eltcalc_P3
mkfifo fifo/gul_S2_summarysummarycalc_P3
mkfifo fifo/gul_S2_summarycalc_P3
mkfifo fifo/gul_S2_summarypltcalc_P3
mkfifo fifo/gul_S2_pltcalc_P3

mkfifo fifo/il_S1_summary_P3
mkfifo fifo/il_S1_summaryeltcalc_P3
mkfifo fifo/il_S1_eltcalc_P3
mkfifo fifo/il_S1_summarysummarycalc_P3
mkfifo fifo/il_S1_summarycalc_P3
mkfifo fifo/il_S1_summarypltcalc_P3
mkfifo fifo/il_S1_pltcalc_P3
mkfifo fifo/il_S2_summary_P3
mkfifo fifo/il_S2_summaryeltcalc_P3
mkfifo fifo/il_S2_eltcalc_P3
mkfifo fifo/il_S2_summarysummarycalc_P3
mkfifo fifo/il_S2_summarycalc_P3
mkfifo fifo/il_S2_summarypltcalc_P3
mkfifo fifo/il_S2_pltcalc_P3



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P3 > work/kat/il_S1_summarycalc_P3 & pid2=$!
pltcalc -s < fifo/il_S1_summarypltcalc_P3 > work/kat/il_S1_pltcalc_P3 & pid3=$!
eltcalc -s < fifo/il_S2_summaryeltcalc_P3 > work/kat/il_S2_eltcalc_P3 & pid4=$!
summarycalctocsv -s < fifo/il_S2_summarysummarycalc_P3 > work/kat/il_S2_summarycalc_P3 & pid5=$!
pltcalc -s < fifo/il_S2_summarypltcalc_P3 > work/kat/il_S2_pltcalc_P3 & pid6=$!
tee < fifo/il_S1_summary_P3 fifo/il_S1_summaryeltcalc_P3 fifo/il_S1_summarypltcalc_P3 fifo/il_S1_summarysummarycalc_P3 work/il_S1_summaryaalcalc/P3.bin work/il_S1_summaryleccalc/P3.bin > /dev/null & pid7=$!
tee < fifo/il_S2_summary_P3 fifo/il_S2_summaryeltcalc_P3 fifo/il_S2_summarypltcalc_P3 fifo/il_S2_summarysummarycalc_P3 work/il_S2_summaryaalcalc/P3.bin work/il_S2_summaryleccalc/P3.bin > /dev/null & pid8=$!
summarycalc -f  -1 fifo/il_S1_summary_P3 -2 fifo/il_S2_summary_P3 < fifo/il_P3 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P3 > work/kat/gul_S1_eltcalc_P3 & pid9=$!
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P3 > work/kat/gul_S1_summarycalc_P3 & pid10=$!
pltcalc -s < fifo/gul_S1_summarypltcalc_P3 > work/kat/gul_S1_pltcalc_P3 & pid11=$!
eltcalc -s < fifo/gul_S2_summaryeltcalc_P3 > work/kat/gul_S2_eltcalc_P3 & pid12=$!
summarycalctocsv -s < fifo/gul_S2_summarysummarycalc_P3 > work/kat/gul_S2_summarycalc_P3 & pid13=$!
pltcalc -s < fifo/gul_S2_summarypltcalc_P3 > work/kat/gul_S2_pltcalc_P3 & pid14=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_summaryeltcalc_P3 fifo/gul_S1_summarypltcalc_P3 fifo/gul_S1_summarysummarycalc_P3 work/gul_S1_summaryaalcalc/P3.bin work/gul_S1_summaryleccalc/P3.bin > /dev/null & pid15=$!
tee < fifo/gul_S2_summary_P3 fifo/gul_S2_summaryeltcalc_P3 fifo/gul_S2_summarypltcalc_P3 fifo/gul_S2_summarysummarycalc_P3 work/gul_S2_summaryaalcalc/P3.bin work/gul_S2_summaryleccalc/P3.bin > /dev/null & pid16=$!
summarycalc -g  -1 fifo/gul_S1_summary_P3 -2 fifo/gul_S2_summary_P3 < fifo/gul_P3 &

eve 3 10 | getmodel | gulcalc -S0 -L0 -r -c fifo/gul_P3 -i - | fmcalc -a2 > fifo/il_P3  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P3 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P3 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P3 > output/il_S1_summarycalc.csv & kpid3=$!
kat work/kat/il_S2_eltcalc_P3 > output/il_S2_eltcalc.csv & kpid4=$!
kat work/kat/il_S2_pltcalc_P3 > output/il_S2_pltcalc.csv & kpid5=$!
kat work/kat/il_S2_summarycalc_P3 > output/il_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P3 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P3 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P3 > output/gul_S1_summarycalc.csv & kpid9=$!
kat work/kat/gul_S2_eltcalc_P3 > output/gul_S2_eltcalc.csv & kpid10=$!
kat work/kat/gul_S2_pltcalc_P3 > output/gul_S2_pltcalc.csv & kpid11=$!
kat work/kat/gul_S2_summarycalc_P3 > output/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

