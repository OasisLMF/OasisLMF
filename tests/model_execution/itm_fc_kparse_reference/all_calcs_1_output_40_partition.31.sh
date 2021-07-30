#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
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

mkfifo fifo/full_correlation/gul_fc_P32

mkfifo fifo/gul_P32

mkfifo fifo/gul_S1_summary_P32
mkfifo fifo/gul_S1_summary_P32.idx
mkfifo fifo/gul_S1_eltcalc_P32
mkfifo fifo/gul_S1_summarycalc_P32
mkfifo fifo/gul_S1_pltcalc_P32

mkfifo fifo/il_P32

mkfifo fifo/il_S1_summary_P32
mkfifo fifo/il_S1_summary_P32.idx
mkfifo fifo/il_S1_eltcalc_P32
mkfifo fifo/il_S1_summarycalc_P32
mkfifo fifo/il_S1_pltcalc_P32

mkfifo fifo/full_correlation/gul_P32

mkfifo fifo/full_correlation/gul_S1_summary_P32
mkfifo fifo/full_correlation/gul_S1_summary_P32.idx
mkfifo fifo/full_correlation/gul_S1_eltcalc_P32
mkfifo fifo/full_correlation/gul_S1_summarycalc_P32
mkfifo fifo/full_correlation/gul_S1_pltcalc_P32

mkfifo fifo/full_correlation/il_P32

mkfifo fifo/full_correlation/il_S1_summary_P32
mkfifo fifo/full_correlation/il_S1_summary_P32.idx
mkfifo fifo/full_correlation/il_S1_eltcalc_P32
mkfifo fifo/full_correlation/il_S1_summarycalc_P32
mkfifo fifo/full_correlation/il_S1_pltcalc_P32



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P32 > work/kat/il_S1_eltcalc_P32 & pid1=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P32 > work/kat/il_S1_summarycalc_P32 & pid2=$!
pltcalc -s < fifo/il_S1_pltcalc_P32 > work/kat/il_S1_pltcalc_P32 & pid3=$!
tee < fifo/il_S1_summary_P32 fifo/il_S1_eltcalc_P32 fifo/il_S1_summarycalc_P32 fifo/il_S1_pltcalc_P32 work/il_S1_summaryaalcalc/P32.bin work/il_S1_summaryleccalc/P32.bin > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P32.idx work/il_S1_summaryleccalc/P32.idx > /dev/null & pid5=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P32 < fifo/il_P32 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P32 > work/kat/gul_S1_eltcalc_P32 & pid6=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P32 > work/kat/gul_S1_summarycalc_P32 & pid7=$!
pltcalc -s < fifo/gul_S1_pltcalc_P32 > work/kat/gul_S1_pltcalc_P32 & pid8=$!
tee < fifo/gul_S1_summary_P32 fifo/gul_S1_eltcalc_P32 fifo/gul_S1_summarycalc_P32 fifo/gul_S1_pltcalc_P32 work/gul_S1_summaryaalcalc/P32.bin work/gul_S1_summaryleccalc/P32.bin > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P32.idx work/gul_S1_summaryleccalc/P32.idx > /dev/null & pid10=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P32 < fifo/gul_P32 &

# --- Do insured loss computes ---
eltcalc -s < fifo/full_correlation/il_S1_eltcalc_P32 > work/full_correlation/kat/il_S1_eltcalc_P32 & pid11=$!
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P32 > work/full_correlation/kat/il_S1_summarycalc_P32 & pid12=$!
pltcalc -s < fifo/full_correlation/il_S1_pltcalc_P32 > work/full_correlation/kat/il_S1_pltcalc_P32 & pid13=$!
tee < fifo/full_correlation/il_S1_summary_P32 fifo/full_correlation/il_S1_eltcalc_P32 fifo/full_correlation/il_S1_summarycalc_P32 fifo/full_correlation/il_S1_pltcalc_P32 work/full_correlation/il_S1_summaryaalcalc/P32.bin work/full_correlation/il_S1_summaryleccalc/P32.bin > /dev/null & pid14=$!
tee < fifo/full_correlation/il_S1_summary_P32.idx work/full_correlation/il_S1_summaryleccalc/P32.idx > /dev/null & pid15=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P32 < fifo/full_correlation/il_P32 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P32 > work/full_correlation/kat/gul_S1_eltcalc_P32 & pid16=$!
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P32 > work/full_correlation/kat/gul_S1_summarycalc_P32 & pid17=$!
pltcalc -s < fifo/full_correlation/gul_S1_pltcalc_P32 > work/full_correlation/kat/gul_S1_pltcalc_P32 & pid18=$!
tee < fifo/full_correlation/gul_S1_summary_P32 fifo/full_correlation/gul_S1_eltcalc_P32 fifo/full_correlation/gul_S1_summarycalc_P32 fifo/full_correlation/gul_S1_pltcalc_P32 work/full_correlation/gul_S1_summaryaalcalc/P32.bin work/full_correlation/gul_S1_summaryleccalc/P32.bin > /dev/null & pid19=$!
tee < fifo/full_correlation/gul_S1_summary_P32.idx work/full_correlation/gul_S1_summaryleccalc/P32.idx > /dev/null & pid20=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P32 < fifo/full_correlation/gul_P32 &

tee < fifo/full_correlation/gul_fc_P32 fifo/full_correlation/gul_P32  | fmcalc -a2 > fifo/full_correlation/il_P32  &
eve 32 40 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P32 -a1 -i - | tee fifo/gul_P32 | fmcalc -a2 > fifo/il_P32  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P32 > output/il_S1_eltcalc.csv & kpid1=$!
kat work/kat/il_S1_pltcalc_P32 > output/il_S1_pltcalc.csv & kpid2=$!
kat work/kat/il_S1_summarycalc_P32 > output/il_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats for fully correlated output ---

kat -s work/full_correlation/kat/il_S1_eltcalc_P32 > output/full_correlation/il_S1_eltcalc.csv & kpid4=$!
kat work/full_correlation/kat/il_S1_pltcalc_P32 > output/full_correlation/il_S1_pltcalc.csv & kpid5=$!
kat work/full_correlation/kat/il_S1_summarycalc_P32 > output/full_correlation/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P32 > output/gul_S1_eltcalc.csv & kpid7=$!
kat work/kat/gul_S1_pltcalc_P32 > output/gul_S1_pltcalc.csv & kpid8=$!
kat work/kat/gul_S1_summarycalc_P32 > output/gul_S1_summarycalc.csv & kpid9=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P32 > output/full_correlation/gul_S1_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P32 > output/full_correlation/gul_S1_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P32 > output/full_correlation/gul_S1_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12

