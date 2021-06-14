#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f /tmp/%FIFO_DIR%/fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P36

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36

mkfifo /tmp/%FIFO_DIR%/fifo/il_P36

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36 > work/kat/il_S1_eltcalc_P36 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36 > work/kat/il_S1_summarycalc_P36 & pid2=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36 > work/kat/il_S1_pltcalc_P36 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P36 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P36 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P36 work/il_S1_summaryaalcalc/P36.bin work/il_S1_summaryleccalc/P36.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36.idx work/il_S1_summaryleccalc/P36.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P36 < /tmp/%FIFO_DIR%/fifo/il_P36 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36 > work/kat/gul_S1_eltcalc_P36 & pid6=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36 > work/kat/gul_S1_summarycalc_P36 & pid7=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36 > work/kat/gul_S1_pltcalc_P36 & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P36 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P36 work/gul_S1_summaryaalcalc/P36.bin work/gul_S1_summaryleccalc/P36.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36.idx work/gul_S1_summaryleccalc/P36.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P36 < /tmp/%FIFO_DIR%/fifo/gul_P36 &

eve 36 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P36 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P36  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P36 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P36 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P36 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P36 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P36 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P36 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

