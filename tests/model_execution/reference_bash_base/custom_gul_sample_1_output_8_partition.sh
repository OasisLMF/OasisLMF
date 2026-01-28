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


mkfifo fifo/gul_P1
mkfifo fifo/gul_P2
mkfifo fifo/gul_P3
mkfifo fifo/gul_P4
mkfifo fifo/gul_P5
mkfifo fifo/gul_P6
mkfifo fifo/gul_P7
mkfifo fifo/gul_P8

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_selt_ord_P1

mkfifo fifo/gul_S1_summary_P2
mkfifo fifo/gul_S1_selt_ord_P2

mkfifo fifo/gul_S1_summary_P3
mkfifo fifo/gul_S1_selt_ord_P3

mkfifo fifo/gul_S1_summary_P4
mkfifo fifo/gul_S1_selt_ord_P4

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_selt_ord_P5

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_selt_ord_P6

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_selt_ord_P7

mkfifo fifo/gul_S1_summary_P8
mkfifo fifo/gul_S1_selt_ord_P8



# --- Do ground up loss computes ---

eltpy -E bin  -s work/kat/gul_S1_elt_sample_P1 < fifo/gul_S1_selt_ord_P1 & pid1=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P2 < fifo/gul_S1_selt_ord_P2 & pid2=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P3 < fifo/gul_S1_selt_ord_P3 & pid3=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P4 < fifo/gul_S1_selt_ord_P4 & pid4=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P5 < fifo/gul_S1_selt_ord_P5 & pid5=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P6 < fifo/gul_S1_selt_ord_P6 & pid6=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P7 < fifo/gul_S1_selt_ord_P7 & pid7=$!
eltpy -E bin  -H -s work/kat/gul_S1_elt_sample_P8 < fifo/gul_S1_selt_ord_P8 & pid8=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_selt_ord_P1 > /dev/null & pid9=$!
tee < fifo/gul_S1_summary_P2 fifo/gul_S1_selt_ord_P2 > /dev/null & pid10=$!
tee < fifo/gul_S1_summary_P3 fifo/gul_S1_selt_ord_P3 > /dev/null & pid11=$!
tee < fifo/gul_S1_summary_P4 fifo/gul_S1_selt_ord_P4 > /dev/null & pid12=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_selt_ord_P5 > /dev/null & pid13=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_selt_ord_P6 > /dev/null & pid14=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_selt_ord_P7 > /dev/null & pid15=$!
tee < fifo/gul_S1_summary_P8 fifo/gul_S1_selt_ord_P8 > /dev/null & pid16=$!

summarypy -m -t gul  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P2 < fifo/gul_P2 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P3 < fifo/gul_P3 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P4 < fifo/gul_P4 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 &
summarypy -m -t gul  -1 fifo/gul_S1_summary_P8 < fifo/gul_P8 &

( evepy 1 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P1  ) &  pid17=$!
( evepy 2 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P2  ) &  pid18=$!
( evepy 3 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P3  ) &  pid19=$!
( evepy 4 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P4  ) &  pid20=$!
( evepy 5 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P5  ) &  pid21=$!
( evepy 6 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P6  ) &  pid22=$!
( evepy 7 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P7  ) &  pid23=$!
( evepy 8 8 | gulmc --socket-server='None' --random-generator=1  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  > fifo/gul_P8  ) &  pid24=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24


# --- Do ground up loss kats ---

katpy -s -f bin -i work/kat/gul_S1_elt_sample_P1 work/kat/gul_S1_elt_sample_P2 work/kat/gul_S1_elt_sample_P3 work/kat/gul_S1_elt_sample_P4 work/kat/gul_S1_elt_sample_P5 work/kat/gul_S1_elt_sample_P6 work/kat/gul_S1_elt_sample_P7 work/kat/gul_S1_elt_sample_P8 -o output/gul_S1_selt.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*
