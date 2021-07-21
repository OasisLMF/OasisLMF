#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir work/gul_S1_summaryaalcalc
mkdir work/gul_S2_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S2_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_P2

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2



# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid2=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid3=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1 > work/kat/gul_S2_eltcalc_P1 & pid4=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1 > work/kat/gul_S2_summarycalc_P1 & pid5=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1 > work/kat/gul_S2_pltcalc_P1 & pid6=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid7=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid8=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid9=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2 > work/kat/gul_S2_eltcalc_P2 & pid10=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2 > work/kat/gul_S2_summarycalc_P2 & pid11=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2 > work/kat/gul_S2_pltcalc_P2 & pid12=$!

tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid13=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P1 work/gul_S2_summaryaalcalc/P1.bin > /dev/null & pid14=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid15=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/gul_S2_pltcalc_P2 work/gul_S2_summaryaalcalc/P2.bin > /dev/null & pid16=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/gul_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/gul_P2 &

# --- Do ground up loss computes ---

eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 > work/full_correlation/kat/gul_S1_eltcalc_P1 & pid17=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 > work/full_correlation/kat/gul_S1_summarycalc_P1 & pid18=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 > work/full_correlation/kat/gul_S1_pltcalc_P1 & pid19=$!
eltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1 > work/full_correlation/kat/gul_S2_eltcalc_P1 & pid20=$!
summarycalctocsv < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1 > work/full_correlation/kat/gul_S2_summarycalc_P1 & pid21=$!
pltcalc < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1 > work/full_correlation/kat/gul_S2_pltcalc_P1 & pid22=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2 > work/full_correlation/kat/gul_S1_eltcalc_P2 & pid23=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2 > work/full_correlation/kat/gul_S1_summarycalc_P2 & pid24=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2 > work/full_correlation/kat/gul_S1_pltcalc_P2 & pid25=$!
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2 > work/full_correlation/kat/gul_S2_eltcalc_P2 & pid26=$!
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2 > work/full_correlation/kat/gul_S2_summarycalc_P2 & pid27=$!
pltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2 > work/full_correlation/kat/gul_S2_pltcalc_P2 & pid28=$!

tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P1 work/full_correlation/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid29=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P1 work/full_correlation/gul_S2_summaryaalcalc/P1.bin > /dev/null & pid30=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_pltcalc_P2 work/full_correlation/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid31=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_eltcalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summarycalc_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_pltcalc_P2 work/full_correlation/gul_S2_summaryaalcalc/P2.bin > /dev/null & pid32=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P1 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 &
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P2 -2 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S2_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 &

eve 1 2 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &
eve 2 2 | getmodel | gulcalc -S0 -L0 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P2 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P2  &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid3=$!
kat -s work/kat/gul_S2_eltcalc_P1 work/kat/gul_S2_eltcalc_P2 > output/gul_S2_eltcalc.csv & kpid4=$!
kat work/kat/gul_S2_pltcalc_P1 work/kat/gul_S2_pltcalc_P2 > output/gul_S2_pltcalc.csv & kpid5=$!
kat work/kat/gul_S2_summarycalc_P1 work/kat/gul_S2_summarycalc_P2 > output/gul_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats for fully correlated output ---

kat -s work/full_correlation/kat/gul_S1_eltcalc_P1 work/full_correlation/kat/gul_S1_eltcalc_P2 > output/full_correlation/gul_S1_eltcalc.csv & kpid7=$!
kat work/full_correlation/kat/gul_S1_pltcalc_P1 work/full_correlation/kat/gul_S1_pltcalc_P2 > output/full_correlation/gul_S1_pltcalc.csv & kpid8=$!
kat work/full_correlation/kat/gul_S1_summarycalc_P1 work/full_correlation/kat/gul_S1_summarycalc_P2 > output/full_correlation/gul_S1_summarycalc.csv & kpid9=$!
kat -s work/full_correlation/kat/gul_S2_eltcalc_P1 work/full_correlation/kat/gul_S2_eltcalc_P2 > output/full_correlation/gul_S2_eltcalc.csv & kpid10=$!
kat work/full_correlation/kat/gul_S2_pltcalc_P1 work/full_correlation/kat/gul_S2_pltcalc_P2 > output/full_correlation/gul_S2_pltcalc.csv & kpid11=$!
kat work/full_correlation/kat/gul_S2_summarycalc_P1 work/full_correlation/kat/gul_S2_summarycalc_P2 > output/full_correlation/gul_S2_summarycalc.csv & kpid12=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9 $kpid10 $kpid11 $kpid12


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
aalcalc -Kgul_S2_summaryaalcalc > output/gul_S2_aalcalc.csv & lpid2=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid3=$!
aalcalc -Kfull_correlation/gul_S2_summaryaalcalc > output/full_correlation/gul_S2_aalcalc.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
