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

fmpy -a2 --create-financial-structure-files
fmpy -a2 --create-financial-structure-files -p RI_1
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/ri_S1_summaryleccalc

mkfifo fifo/gul_P1
mkfifo fifo/gul_P2

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_eltcalc_P1
mkfifo fifo/gul_S1_summarycalc_P1
mkfifo fifo/gul_S1_pltcalc_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_eltcalc_P2
mkfifo fifo/gul_S1_summarycalc_P2
mkfifo fifo/gul_S1_pltcalc_P2

mkfifo fifo/il_P1
mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summary_P1.idx

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_summary_P2.idx

mkfifo fifo/ri_P1
mkfifo fifo/ri_P2

mkfifo fifo/ri_S1_summary_P1
mkfifo fifo/ri_S1_summary_P1.idx

mkfifo fifo/ri_S1_summary_P2
mkfifo fifo/ri_S1_summary_P2.idx

mkfifo fifo/gul_lb_P1
mkfifo fifo/gul_lb_P2

mkfifo fifo/lb_il_P1
mkfifo fifo/lb_il_P2



# --- Do reinsurance loss computes ---



tee < fifo/ri_S1_summary_P1 work/ri_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < fifo/ri_S1_summary_P1.idx work/ri_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!
tee < fifo/ri_S1_summary_P2 work/ri_S1_summaryleccalc/P2.bin > /dev/null & pid3=$!
tee < fifo/ri_S1_summary_P2.idx work/ri_S1_summaryleccalc/P2.idx > /dev/null & pid4=$!

summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P1 < fifo/ri_P1 &
summarycalc -m -f -p RI_1 -1 fifo/ri_S1_summary_P2 < fifo/ri_P2 &

# --- Do insured loss computes ---



tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid5=$!
tee < fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid6=$!
tee < fifo/il_S1_summary_P2 work/il_S1_summaryleccalc/P2.bin > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P2.idx work/il_S1_summaryleccalc/P2.idx > /dev/null & pid8=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

# --- Do ground up loss computes ---

eltcalc < fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid9=$!
summarycalctocsv < fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid10=$!
pltcalc < fifo/gul_S1_pltcalc_P1 > work/kat/gul_S1_pltcalc_P1 & pid11=$!
eltcalc -s < fifo/gul_S1_eltcalc_P2 > work/kat/gul_S1_eltcalc_P2 & pid12=$!
summarycalctocsv -s < fifo/gul_S1_summarycalc_P2 > work/kat/gul_S1_summarycalc_P2 & pid13=$!
pltcalc -s < fifo/gul_S1_pltcalc_P2 > work/kat/gul_S1_pltcalc_P2 & pid14=$!


tee < fifo/gul_S1_summary_P1 fifo/gul_S1_eltcalc_P1 fifo/gul_S1_summarycalc_P1 fifo/gul_S1_pltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid15=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_eltcalc_P2 fifo/gul_S1_summarycalc_P2 fifo/gul_S1_pltcalc_P2 work/gul_S1_summaryaalcalc/P2.bin > /dev/null & pid16=$!

summarycalc -m -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarycalc -m -i  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &

eve 1 2 | getmodel | gulcalc -S0 -L0 -r -a0 -i - | tee fifo/gul_P1 > fifo/gul_lb_P1  &
eve 2 2 | getmodel | gulcalc -S0 -L0 -r -a0 -i - | tee fifo/gul_P2 > fifo/gul_lb_P2  &
load_balancer -i fifo/gul_lb_P1 fifo/gul_lb_P2 -o fifo/lb_il_P1 fifo/lb_il_P2 &
fmpy -a2 < fifo/lb_il_P1 | tee fifo/il_P1 | fmpy -a2 -n -p RI_1 > fifo/ri_P1 &
fmpy -a2 < fifo/lb_il_P2 | tee fifo/il_P2 | fmpy -a2 -n -p RI_1 > fifo/ri_P2 &

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do reinsurance loss kats ---


# --- Do insured loss kats ---


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P1 work/kat/gul_S1_eltcalc_P2 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 work/kat/gul_S1_pltcalc_P2 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 work/kat/gul_S1_summarycalc_P2 > output/gul_S1_summarycalc.csv & kpid3=$!
wait $kpid1 $kpid2 $kpid3


ordleccalc -r -Kri_S1_summaryleccalc -F -f -S -s -M -m -W -w -O output/ri_S1_ept.csv -o output/ri_S1_psept.csv & lpid1=$!
ordleccalc  -Kil_S1_summaryleccalc -F -S -s -M -m -O output/il_S1_ept.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
wait $lpid1 $lpid2 $lpid3

rm -R -f work/*
rm -R -f fifo/*
