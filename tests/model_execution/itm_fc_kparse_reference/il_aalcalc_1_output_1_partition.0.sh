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

mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P1

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1

mkfifo fifo/full_correlation/il_P1

mkfifo fifo/full_correlation/il_S1_summary_P1



# --- Do insured loss computes ---



tee < fifo/il_S1_summary_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

# --- Do insured loss computes ---



tee < fifo/full_correlation/il_S1_summary_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin > /dev/null & pid2=$!

summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &

fmcalc -a2 < fifo/full_correlation/gul_fc_P1 > fifo/full_correlation/il_P1 &
eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P1 -a1 -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

