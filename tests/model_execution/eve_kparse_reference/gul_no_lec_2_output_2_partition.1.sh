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

mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryaalcalc

mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2
mkfifo fifo/gul_S2_summary_P2
mkfifo fifo/gul_S2_summary_P2.idx
mkfifo fifo/gul_S2_eltcalc_P2
mkfifo fifo/gul_S2_summarycalc_P2
mkfifo fifo/gul_S2_pltcalc_P2



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid1=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid2=$!
pltcalc -s < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid3=$!
eltcalc -s < fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid4=$!
summarycalctocsv -s < fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid5=$!
pltcalc -s < fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid6=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid7=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx > /dev/null & pid8=$!
tee < fifo/gul_S2_summary_P2 fifo/gul_S2_eltcalc_P2 fifo/gul_S2_summarycalc_P2 fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin > /dev/null & pid9=$!
tee < fifo/gul_S2_summary_P2.idx work/gul_S2_summaryaalcalc/P2.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 -2 fifo/gul_S2_summary_P2 < fifo/gul_P2 &

eve -R 2 2 | getmodel | gulcalc -S0 -L0 -r -a0 -i - > fifo/gul_P2  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid3=$!
kat work/kat/gul_S2_eltcalc_P2 > output/gul_S2_eltcalc.csv & kpid4=$!
kat work/kat/gul_S2_pltcalc_P2 > output/gul_S2_pltcalc.csv & kpid5=$!
kat work/kat/gul_S2_summarycalc_P2 > output/gul_S2_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6

