#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

mkfifo fifo/gul_P30

mkfifo fifo/gul_S1_summary_P30
mkfifo fifo/gul_S1_summary_P30.idx
mkfifo fifo/gul_S1_eltcalc_P30
mkfifo fifo/gul_S1_summarycalc_P30
mkfifo fifo/gul_S1_pltcalc_P30

mkfifo fifo/il_P30

mkfifo fifo/il_S1_summary_P30
mkfifo fifo/il_S1_summary_P30.idx
mkfifo fifo/il_S1_eltcalc_P30
mkfifo fifo/il_S1_summarycalc_P30
mkfifo fifo/il_S1_pltcalc_P30



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P30 > work/kat/il_S1_eltcalc_P30 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P30 > work/kat/il_S1_summarycalc_P30 & pid2=$!
pltcalc -s < fifo/il_S1_pltcalc_P30 > work/kat/il_S1_pltcalc_P30 & pid3=$!
tee < fifo/il_S1_summary_P30 fifo/il_S1_eltcalc_P30 fifo/il_S1_summarycalc_P30 fifo/il_S1_pltcalc_P30 work/il_S1_summaryaalcalc/P30.bin work/il_S1_summaryleccalc/P30.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P30.idx work/il_S1_summaryaalcalc/P30.idx work/il_S1_summaryleccalc/P30.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P30 < fifo/il_P30 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P30 > work/kat/gul_S1_eltcalc_P30 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P30 > work/kat/gul_S1_summarycalc_P30 & pid7=$!
pltcalc -s < fifo/gul_S1_pltcalc_P30 > work/kat/gul_S1_pltcalc_P30 & pid8=$!
tee < fifo/gul_S1_summary_P30 fifo/gul_S1_eltcalc_P30 fifo/gul_S1_summarycalc_P30 fifo/gul_S1_pltcalc_P30 work/gul_S1_summaryaalcalc/P30.bin work/gul_S1_summaryleccalc/P30.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P30.idx work/gul_S1_summaryaalcalc/P30.idx work/gul_S1_summaryleccalc/P30.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P30 < fifo/gul_P30 &

eve -R 30 40 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | tee fifo/gul_P30 | fmcalc -a2 > fifo/il_P30  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P30 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P30 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P30 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P30 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P30 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P30 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

