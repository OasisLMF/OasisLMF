#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f work/*
mkdir work/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir work/gul_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1



# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid3=$!


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid4=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S0 -L0 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid3=$!
wait $kpid1 $kpid2 $kpid3


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
