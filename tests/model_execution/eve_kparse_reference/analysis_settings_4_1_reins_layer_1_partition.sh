#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/ri_S1_summaryleccalc
mkdir -p work/ri_S1_summaryaalcalc

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summary_P1.idx
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_summary_P2.idx
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx
mkfifo fifo/il_S1_eltcalc_P1
mkfifo fifo/il_S1_summarycalc_P1
mkfifo fifo/il_S1_pltcalc_P1

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx
mkfifo fifo/il_S1_eltcalc_P2
mkfifo fifo/il_S1_summarycalc_P2
mkfifo fifo/il_S1_pltcalc_P2

mkfifo fifo/ri_P1
mkfifo fifo/ri_P2

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summary_P1.idx
mkfifo fifo/ri_S1_eltcalc_P1
mkfifo fifo/ri_S1_summarycalc_P1
mkfifo fifo/ri_S1_pltcalc_P1

mkfifo fifo/ri_S1_summary_P2
mkfifo fifo/ri_S1_summary_P2.idx
mkfifo fifo/ri_S1_eltcalc_P2
mkfifo fifo/ri_S1_summarycalc_P2
mkfifo fifo/ri_S1_pltcalc_P2

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2



# --- Do reinsurance loss computes ---

eltcalc < fifo/ri_S1_eltcalc_P1 > work/kat/ri_S1_eltcalc_P1 & pid1=$!
summarycalctocsv < fifo/ri_S1_summarycalc_P1 > work/kat/ri_S1_summarycalc_P1 & pid2=$!
pltcalc < fifo/ri_S1_pltcalc_P1 > work/kat/ri_S1_pltcalc_P1 & pid3=$!
eltcalc -s < fifo/ri_S1_eltcalc_P2 > work/kat/ri_S1_eltcalc_P2 & pid4=$!
summarycalctocsv -s < fifo/ri_S1_summarycalc_P2 > work/kat/ri_S1_summarycalc_P2 & pid5=$!
pltcalc -H < fifo/ri_S1_pltcalc_P2 > work/kat/ri_S1_pltcalc_P2 & pid6=$!


tee < fifo/ri_S1_summary_P1 fifo/ri_S1_eltcalc_P1 fifo/ri_S1_summarycalc_P1 fifo/ri_S1_pltcalc_P1 work/ri_S1_summaryaalcalc/P1.bin work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid7=$!
tee < fifo/ri_S1_summary_P1.idx work/ri_S1_summaryaalcalc/P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid8=$!
tee < fifo/ri_S1_summary_P2 fifo/ri_S1_eltcalc_P2 fifo/ri_S1_summarycalc_P2 fifo/ri_S1_pltcalc_P2 work/ri_S1_summaryaalcalc/P2.bin work/ri_S1_summaryleccalc/P2.bin > /dev/null & pid9=$!
tee < fifo/ri_S1_summary_P2.idx work/ri_S1_summaryaalcalc/P2.idx work/ri_S1_summaryleccalc/P2.idx > /dev/null & pid10=$!

summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 &
summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P2 < fifo/ri_P2 &

# --- Do insured loss computes ---

eltcalc < fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid11=$!
summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid12=$!
pltcalc < fifo/il_S1_pltcalc_P1 > work/kat/il_S1_pltcalc_P1 & pid13=$!
eltcalc -s < fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid14=$!
summarycalctocsv -s < fifo/il_S1_summarycalc_P2 > work/kat/il_S1_summarycalc_P2 & pid15=$!
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid16=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_eltcalc_P1 fifo/il_S1_summarycalc_P1 fifo/il_S1_pltcalc_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid17=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryaalcalc/P1.idx > /dev/null & pid18=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_eltcalc_P2 fifo/il_S1_summarycalc_P2 fifo/il_S1_pltcalc_P2 work/il_S1_summaryaalcalc/P2.bin > /dev/null & pid19=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryaalcalc/P2.idx > /dev/null & pid20=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid21=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid22=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid23=$!
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid24=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid25=$!
pltcalc -H < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid26=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid27=$!
tee < fifo/gul_S1_summary_P1.idx work/gul_S1_summaryaalcalc/P1.idx > /dev/null & pid28=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid29=$!
tee < fifo/gul_S1_summary_P2.idx work/gul_S1_summaryaalcalc/P2.idx > /dev/null & pid30=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &

( eve -R 1 2 | getmodel | gulcalc -S0 -L0 -r -a0 -i - | tee fifo/gul_P1 > fifo/gul_lb_P1  ) & 
( eve -R 2 2 | getmodel | gulcalc -S0 -L0 -r -a0 -i - | tee fifo/gul_P2 > fifo/gul_lb_P2  ) & 
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
( fmcalc -a2 < fifo/lb_il_P1 | tee fifo/il_P1 | fmcalc -a2 -n -p RI_1 > fifo/ri_P1 ) & pid31=$!
( fmcalc -a2 < fifo/lb_il_P2 | tee fifo/il_P2 | fmcalc -a2 -n -p RI_1 > fifo/ri_P2 ) & pid32=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32


# --- Do reinsurance loss kats ---

kat -u work/kat/ri_S1_eltcalc_P1 > output/ri_S1_eltcalc.csv & kpid1=$!
kat -u work/kat/ri_S1_pltcalc_P1 > output/ri_S1_pltcalc.csv & kpid2=$!
kat -u work/kat/ri_S1_summarycalc_P1 > output/ri_S1_summarycalc.csv & kpid3=$!

# --- Do insured loss kats ---

kat -u work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid4=$!
kat -u work/kat/il_S1_pltcalc_P1 > output/il_S1_pltcalc.csv & kpid5=$!
kat -u work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats ---

kat -u work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid7=$!
kat -u work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid8=$!
kat -u work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid9=$!
wait $kpid1 $kpid2 $kpid3 $kpid4 $kpid5 $kpid6 $kpid7 $kpid8 $kpid9


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kri_S1_summaryleccalc -F output/ri_S1_leccalc_full_uncertainty_aep.csv -f output/ri_S1_leccalc_full_uncertainty_oep.csv -S output/ri_S1_leccalc_sample_mean_aep.csv -s output/ri_S1_leccalc_sample_mean_oep.csv -W output/ri_S1_leccalc_wheatsheaf_aep.csv -M output/ri_S1_leccalc_wheatsheaf_mean_aep.csv -m output/ri_S1_leccalc_wheatsheaf_mean_oep.csv -w output/ri_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid3=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f fifo/*
