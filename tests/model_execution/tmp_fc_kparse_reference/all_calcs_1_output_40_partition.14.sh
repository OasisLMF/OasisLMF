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
mkdir work/full_correlation/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryleccalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_P15

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15

mkfifo gul_S1_summary_P15
mkfifo gul_S1_summaryeltcalc_P15
mkfifo gul_S1_eltcalc_P15
mkfifo gul_S1_summarysummarycalc_P15
mkfifo gul_S1_summarycalc_P15
mkfifo gul_S1_summarypltcalc_P15
mkfifo gul_S1_pltcalc_P15

mkfifo il_S1_summary_P15
mkfifo il_S1_summaryeltcalc_P15
mkfifo il_S1_eltcalc_P15
mkfifo il_S1_summarysummarycalc_P15
mkfifo il_S1_summarycalc_P15
mkfifo il_S1_summarypltcalc_P15
mkfifo il_S1_pltcalc_P15



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15 > work/kat/il_S1_summarycalc_P15 & pid2=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P15 work/il_S1_summaryaalcalc/P15.bin work/il_S1_summaryleccalc/P15.bin > /dev/null & pid4=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/il_P15 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15 > work/kat/gul_S1_eltcalc_P15 & pid5=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15 > work/kat/gul_S1_summarycalc_P15 & pid6=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15 > work/kat/gul_S1_pltcalc_P15 & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P15 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P15 work/gul_S1_summaryaalcalc/P15.bin work/gul_S1_summaryleccalc/P15.bin > /dev/null & pid8=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/gul_P15 &

eve 15 40 | getmodel | gulcalc -S100 -L100 -r -j gul_P15 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P15 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P15 > il_P15 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
eltcalc -s < il_S1_summaryeltcalc_P15 > work/full_correlation/kat/il_S1_eltcalc_P15 & pid1=$!
summarycalctocsv -s < il_S1_summarysummarycalc_P15 > work/full_correlation/kat/il_S1_summarycalc_P15 & pid2=$!
pltcalc -s < il_S1_summarypltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 & pid3=$!
tee < il_S1_summary_P15 il_S1_summaryeltcalc_P15 il_S1_summarypltcalc_P15 il_S1_summarysummarycalc_P15 work/full_correlation/il_S1_summaryaalcalc/P15.bin work/full_correlation/il_S1_summaryleccalc/P15.bin > /dev/null & pid4=$!
summarycalc -f  -1 il_S1_summary_P15 < il_P15 &

# --- Do ground up loss computes ---
eltcalc -s < gul_S1_summaryeltcalc_P15 > work/full_correlation/kat/gul_S1_eltcalc_P15 & pid5=$!
summarycalctocsv -s < gul_S1_summarysummarycalc_P15 > work/full_correlation/kat/gul_S1_summarycalc_P15 & pid6=$!
pltcalc -s < gul_S1_summarypltcalc_P15 > work/full_correlation/kat/gul_S1_pltcalc_P15 & pid7=$!
tee < gul_S1_summary_P15 gul_S1_summaryeltcalc_P15 gul_S1_summarypltcalc_P15 gul_S1_summarysummarycalc_P15 work/full_correlation/gul_S1_summaryaalcalc/P15.bin work/full_correlation/gul_S1_summaryleccalc/P15.bin > /dev/null & pid8=$!
summarycalc -i  -1 gul_S1_summary_P15 < gul_P15 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P15 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P15 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P15 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P15 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P15 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P15 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P15 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P15 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P15 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P15 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P15 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P15 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

