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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P18

mkfifo /tmp/%FIFO_DIR%/fifo/il_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P18

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P18



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18 > work/kat/il_S1_eltcalc_P18 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18 > work/kat/il_S1_summarycalc_P18 & pid2=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18 > work/kat/il_S1_pltcalc_P18 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/il_S1_summarysummarycalc_P18 work/il_S1_summaryaalcalc/P18.bin work/il_S1_summaryleccalc/P18.bin > /dev/null & pid4=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/il_P18 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18 > work/kat/gul_S1_eltcalc_P18 & pid5=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18 > work/kat/gul_S1_summarycalc_P18 & pid6=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18 > work/kat/gul_S1_pltcalc_P18 & pid7=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summaryeltcalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P18 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P18 work/gul_S1_summaryaalcalc/P18.bin work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid8=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/gul_P18 &

eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P18 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P18  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P18 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P18 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P18 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P18 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P18 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P18 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

