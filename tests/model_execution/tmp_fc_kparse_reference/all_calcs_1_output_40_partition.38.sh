#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f /tmp/%FIFO_DIR%/fifo/*
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P39

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/il_P39

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P39

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P39

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P39

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P39.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P39
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P39



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39 > work/kat/il_S1_eltcalc_P39 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39 > work/kat/il_S1_summarycalc_P39 & pid2=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39 > work/kat/il_S1_pltcalc_P39 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P39 work/il_S1_summaryaalcalc/P39.bin work/il_S1_summaryleccalc/P39.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39.idx work/il_S1_summaryaalcalc/P39.idx work/il_S1_summaryleccalc/P39.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/il_P39 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39 > work/kat/gul_S1_eltcalc_P39 & pid6=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39 > work/kat/gul_S1_summarycalc_P39 & pid7=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39 > work/kat/gul_S1_pltcalc_P39 & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P39 work/gul_S1_summaryaalcalc/P39.bin work/gul_S1_summaryleccalc/P39.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39.idx work/gul_S1_summaryaalcalc/P39.idx work/gul_S1_summaryleccalc/P39.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/gul_P39 &

# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P39 > work/full_correlation/kat/il_S1_eltcalc_P39 & pid11=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P39 > work/full_correlation/kat/il_S1_summarycalc_P39 & pid12=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P39 > work/full_correlation/kat/il_S1_pltcalc_P39 & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P39 work/full_correlation/il_S1_summaryaalcalc/P39.bin work/full_correlation/il_S1_summaryleccalc/P39.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P39.idx work/full_correlation/il_S1_summaryaalcalc/P39.idx work/full_correlation/il_S1_summaryleccalc/P39.idx > /dev/null & pid15=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P39 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P39 > work/full_correlation/kat/gul_S1_eltcalc_P39 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P39 > work/full_correlation/kat/gul_S1_summarycalc_P39 & pid17=$!
pltcalc -H < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P39 > work/full_correlation/kat/gul_S1_pltcalc_P39 & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P39 work/full_correlation/gul_S1_summaryaalcalc/P39.bin work/full_correlation/gul_S1_summaryleccalc/P39.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P39.idx work/full_correlation/gul_S1_summaryaalcalc/P39.idx work/full_correlation/gul_S1_summaryleccalc/P39.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P39 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P39 &

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P39 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P39  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P39  &
eve 39 40 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P39 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P39 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P39  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P39 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P39 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P39 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P39 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P39 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P39 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P39 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P39 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P39 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P39 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P39 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P39 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

