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

mkfifo fifo/gul_P40

mkfifo fifo/gul_S1_summary_P40
mkfifo fifo/gul_S1_summary_P40.idx
mkfifo fifo/gul_S1_eltcalc_P40
mkfifo fifo/gul_S1_summarycalc_P40
mkfifo fifo/gul_S1_pltcalc_P40

mkfifo fifo/il_P40

mkfifo fifo/il_S1_summary_P40
mkfifo fifo/il_S1_summary_P40.idx
mkfifo fifo/il_S1_eltcalc_P40
mkfifo fifo/il_S1_summarycalc_P40
mkfifo fifo/il_S1_pltcalc_P40



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P40 > work/kat/il_S1_eltcalc_P40 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P40 > work/kat/il_S1_summarycalc_P40 & pid2=$!
pltcalc -H < fifo/il_S1_pltcalc_P40 > work/kat/il_S1_pltcalc_P40 & pid3=$!
tee < fifo/il_S1_summary_P40 fifo/il_S1_eltcalc_P40 fifo/il_S1_summarycalc_P40 fifo/il_S1_pltcalc_P40 work/il_S1_summaryaalcalc/P40.bin work/il_S1_summaryleccalc/P40.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P40.idx work/il_S1_summaryaalcalc/P40.idx work/il_S1_summaryleccalc/P40.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P40 < fifo/il_P40 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P40 > work/kat/gul_S1_eltcalc_P40 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P40 > work/kat/gul_S1_summarycalc_P40 & pid7=$!
pltcalc -H < fifo/gul_S1_pltcalc_P40 > work/kat/gul_S1_pltcalc_P40 & pid8=$!
tee < fifo/gul_S1_summary_P40 fifo/gul_S1_eltcalc_P40 fifo/gul_S1_summarycalc_P40 fifo/gul_S1_pltcalc_P40 work/gul_S1_summaryaalcalc/P40.bin work/gul_S1_summaryleccalc/P40.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P40.idx work/gul_S1_summaryaalcalc/P40.idx work/gul_S1_summaryleccalc/P40.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P40 < fifo/gul_P40 &

eve 40 40 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | tee fifo/gul_P40 | fmcalc -a2 > fifo/il_P40  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P40 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P40 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P40 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P40 > output/gul_S1_eltcalc.csv & kpid4=$!
kat work/kat/gul_S1_pltcalc_P40 > output/gul_S1_pltcalc.csv & kpid5=$!
kat work/kat/gul_S1_summarycalc_P40 > output/gul_S1_summarycalc.csv & kpid6=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6
