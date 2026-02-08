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

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P1
mkfifo fifo/il_P2
mkfifo fifo/il_P3
mkfifo fifo/il_P4
mkfifo fifo/il_P5
mkfifo fifo/il_P6
mkfifo fifo/il_P7
mkfifo fifo/il_P8

mkfifo fifo/il_S1_summary_P1

mkfifo fifo/il_S1_summary_P2

mkfifo fifo/il_S1_summary_P3

mkfifo fifo/il_S1_summary_P4

mkfifo fifo/il_S1_summary_P5

mkfifo fifo/il_S1_summary_P6

mkfifo fifo/il_S1_summary_P7

mkfifo fifo/il_S1_summary_P8



# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P2 > /dev/null & pid2=$!
tee < fifo/il_S1_summary_P3 > /dev/null & pid3=$!
tee < fifo/il_S1_summary_P4 > /dev/null & pid4=$!
tee < fifo/il_S1_summary_P5 > /dev/null & pid5=$!
tee < fifo/il_S1_summary_P6 > /dev/null & pid6=$!
tee < fifo/il_S1_summary_P7 > /dev/null & pid7=$!
tee < fifo/il_S1_summary_P8 > /dev/null & pid8=$!

summarypy -m -t il  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &
summarypy -m -t il  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &
summarypy -m -t il  -1 fifo/il_S1_summary_P3 < fifo/il_P3 &
summarypy -m -t il  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &
summarypy -m -t il  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &
summarypy -m -t il  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &
summarypy -m -t il  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &
summarypy -m -t il  -1 fifo/il_S1_summary_P8 < fifo/il_P8 &

( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P1  ) & pid9=$!
( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P2  ) & pid10=$!
( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P3  ) & pid11=$!
( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P4  ) & pid12=$!
( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P5  ) & pid13=$!
( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P6  ) & pid14=$!
( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P7  ) & pid15=$!
( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 > fifo/il_P8  ) & pid16=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16


# --- Do insured loss kats ---


rm -R -f work/*
rm -R -f fifo/*
