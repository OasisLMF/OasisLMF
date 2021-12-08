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

mkfifo fifo/gul_P27

mkfifo fifo/gul_S1_summary_P27
mkfifo fifo/gul_S1_summary_P27.idx
mkfifo fifo/gul_S1_eltcalc_P27
mkfifo fifo/gul_S1_summarycalc_P27
mkfifo fifo/gul_S1_pltcalc_P27

mkfifo fifo/il_P27

mkfifo fifo/il_S1_summary_P27
mkfifo fifo/il_S1_summary_P27.idx
mkfifo fifo/il_S1_eltcalc_P27
mkfifo fifo/il_S1_summarycalc_P27
mkfifo fifo/il_S1_pltcalc_P27



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P27 > work/kat/il_S1_eltcalc_P27 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P27 > work/kat/il_S1_summarycalc_P27 & pid2=$!
pltcalc -s < fifo/il_S1_pltcalc_P27 > work/kat/il_S1_pltcalc_P27 & pid3=$!
tee < fifo/il_S1_summary_P27 fifo/il_S1_eltcalc_P27 fifo/il_S1_summarycalc_P27 fifo/il_S1_pltcalc_P27 work/il_S1_summaryaalcalc/P27.bin work/il_S1_summaryleccalc/P27.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P27.idx work/il_S1_summaryleccalc/P27.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P27 < fifo/il_P27 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P27 > work/kat/gul_S1_eltcalc_P27 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P27 > work/kat/gul_S1_summarycalc_P27 & pid7=$!
pltcalc -s < fifo/gul_S1_pltcalc_P27 > work/kat/gul_S1_pltcalc_P27 & pid8=$!
tee < fifo/gul_S1_summary_P27 fifo/gul_S1_eltcalc_P27 fifo/gul_S1_summarycalc_P27 fifo/gul_S1_pltcalc_P27 work/gul_S1_summaryaalcalc/P27.bin work/gul_S1_summaryleccalc/P27.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P27.idx work/gul_S1_summaryleccalc/P27.idx > /dev/null & pid10=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P27 < fifo/gul_P27 &

eve 27 40 | getmodel | gulcalc -S100 -L100 -r -c fifo/gul_P27 -i - | fmcalc -a2 > fifo/il_P27  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P27 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P27 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P27 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P27 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P27 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P27 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

