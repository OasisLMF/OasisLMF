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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_P20

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P20



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20 > work/kat/il_S1_summarycalc_P20 & pid2=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20 > work/kat/il_S1_pltcalc_P20 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P20 work/il_S1_summaryaalcalc/P20.bin work/il_S1_summaryleccalc/P20.bin > /dev/null & pid4=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/il_P20 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20 > work/kat/gul_S1_eltcalc_P20 & pid5=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20 > work/kat/gul_S1_summarycalc_P20 & pid6=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20 > work/kat/gul_S1_pltcalc_P20 & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P20 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P20 work/gul_S1_summaryaalcalc/P20.bin work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid8=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/gul_P20 &

eve 20 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P20 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P20 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P20 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P20 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P20 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P20 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

