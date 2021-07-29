#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

fmpy -a2 --create-financial-structure-files
mkdir work/il_S1_summaryaalcalc

mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P2 work/il_S1_summaryaalcalc/P2.bin > /dev/null & pid1=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmpy -a2 > fifo/il_P2  &

wait $pid1


# --- Do insured loss kats ---

