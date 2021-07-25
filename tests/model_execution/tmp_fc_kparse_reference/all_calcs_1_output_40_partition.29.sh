#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

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

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P30

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/il_P30

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P30

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P30

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P30

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P30.idx
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P30
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P30



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30 > work/kat/il_S1_eltcalc_P30 & pid1=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30 > work/kat/il_S1_summarycalc_P30 & pid2=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30 > work/kat/il_S1_pltcalc_P30 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P30 work/il_S1_summaryaalcalc/P30.bin work/il_S1_summaryleccalc/P30.bin > /dev/null & pid4=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30.idx work/il_S1_summaryleccalc/P30.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/il_P30 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30 > work/kat/gul_S1_eltcalc_P30 & pid6=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30 > work/kat/gul_S1_summarycalc_P30 & pid7=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30 > work/kat/gul_S1_pltcalc_P30 & pid8=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P30 work/gul_S1_summaryaalcalc/P30.bin work/gul_S1_summaryleccalc/P30.bin > /dev/null & pid9=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30.idx work/gul_S1_summaryleccalc/P30.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/gul_P30 &

# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P30 > work/full_correlation/kat/il_S1_eltcalc_P30 & pid11=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P30 > work/full_correlation/kat/il_S1_summarycalc_P30 & pid12=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P30 > work/full_correlation/kat/il_S1_pltcalc_P30 & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_pltcalc_P30 work/full_correlation/il_S1_summaryaalcalc/P30.bin work/full_correlation/il_S1_summaryleccalc/P30.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P30.idx work/full_correlation/il_S1_summaryleccalc/P30.idx > /dev/null & pid15=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P30 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P30 > work/full_correlation/kat/gul_S1_eltcalc_P30 & pid16=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P30 > work/full_correlation/kat/gul_S1_summarycalc_P30 & pid17=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P30 > work/full_correlation/kat/gul_S1_pltcalc_P30 & pid18=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P30 work/full_correlation/gul_S1_summaryaalcalc/P30.bin work/full_correlation/gul_S1_summaryleccalc/P30.bin > /dev/null & pid19=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P30.idx work/full_correlation/gul_S1_summaryleccalc/P30.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P30 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P30 &

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P30 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P30  | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P30  &
eve 30 40 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P30 -a1 -i - | tee /tmp/%FIFO_DIR%/fifo/gul_P30 | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P30  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P30 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P30 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P30 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P30 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P30 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P30 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P30 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P30 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P30 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P30 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P30 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P30 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

