#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*
export OASIS_PYTOOLS_LOG_DIR=$LOG_DIR


check_fifos() {
    local has_error=0
    for f in "$@"; do
        [ -e "$f" ] || { echo "[ERROR] Expected FIFO not found: $f"; has_error=1; continue; }
        [ -p "$f" ] || { echo "[ERROR] Not a FIFO: $f"; has_error=1; }
    done
    [ "$has_error" -eq 0 ] || false
}

exec_wait(){
    local BASH_VER_MAJOR=${BASH_VERSION:0:1}
    local BASH_VER_MINOR=${BASH_VERSION:2:1}
    if [[ "$BASH_VER_MAJOR" -gt 5 ]] || { [[ "$BASH_VER_MAJOR" -eq 5 ]] && [[ "$BASH_VER_MINOR" -ge 1 ]]; }; then
        local pid_exitcode
        wait -p pid_exitcode "$@"
    else
        wait "$@"
    fi
}
# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files
#fmpy -a2 --create-financial-structure-files -p input/RI_1
#fmpy -a2 --create-financial-structure-files -p input/RI_2
#fmpy -a2 --create-financial-structure-files -p input/RI_3
mkdir -p work/rl_IP1_S1_summaryleccalc
mkdir -p work/rl_IP1_S1_summary_palt
mkdir -p work/rl_S1_summaryleccalc
mkdir -p work/rl_S1_summary_palt

mkfifo fifo/rl_IP1_P1
mkfifo fifo/rl_IP1_P2
mkfifo fifo/rl_IP1_P3
mkfifo fifo/rl_IP1_P4
mkfifo fifo/rl_IP1_P5
mkfifo fifo/rl_IP1_P6
mkfifo fifo/rl_IP1_P7
mkfifo fifo/rl_IP1_P8

mkfifo fifo/rl_IP1_S1_rlcombine_P1
mkfifo fifo/rl_IP1_S1_rlcombine_P2
mkfifo fifo/rl_IP1_S1_rlcombine_P3
mkfifo fifo/rl_IP1_S1_rlcombine_P4
mkfifo fifo/rl_IP1_S1_rlcombine_P5
mkfifo fifo/rl_IP1_S1_rlcombine_P6
mkfifo fifo/rl_IP1_S1_rlcombine_P7
mkfifo fifo/rl_IP1_S1_rlcombine_P8

mkfifo fifo/rl_IP2_P1
mkfifo fifo/rl_IP2_P2
mkfifo fifo/rl_IP2_P3
mkfifo fifo/rl_IP2_P4
mkfifo fifo/rl_IP2_P5
mkfifo fifo/rl_IP2_P6
mkfifo fifo/rl_IP2_P7
mkfifo fifo/rl_IP2_P8

mkfifo fifo/rl_IP2_S1_rlcombine_P1
mkfifo fifo/rl_IP2_S1_rlcombine_P2
mkfifo fifo/rl_IP2_S1_rlcombine_P3
mkfifo fifo/rl_IP2_S1_rlcombine_P4
mkfifo fifo/rl_IP2_S1_rlcombine_P5
mkfifo fifo/rl_IP2_S1_rlcombine_P6
mkfifo fifo/rl_IP2_S1_rlcombine_P7
mkfifo fifo/rl_IP2_S1_rlcombine_P8

mkfifo fifo/rl_IP3_P1
mkfifo fifo/rl_IP3_P2
mkfifo fifo/rl_IP3_P3
mkfifo fifo/rl_IP3_P4
mkfifo fifo/rl_IP3_P5
mkfifo fifo/rl_IP3_P6
mkfifo fifo/rl_IP3_P7
mkfifo fifo/rl_IP3_P8

mkfifo fifo/rl_IP3_S1_rlcombine_P1
mkfifo fifo/rl_IP3_S1_rlcombine_P2
mkfifo fifo/rl_IP3_S1_rlcombine_P3
mkfifo fifo/rl_IP3_S1_rlcombine_P4
mkfifo fifo/rl_IP3_S1_rlcombine_P5
mkfifo fifo/rl_IP3_S1_rlcombine_P6
mkfifo fifo/rl_IP3_S1_rlcombine_P7
mkfifo fifo/rl_IP3_S1_rlcombine_P8

mkfifo fifo/rl_IP1_S1_summary_P1
mkfifo fifo/rl_IP1_S1_summary_P1.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P1
mkfifo fifo/rl_IP1_S1_selt_ord_P1
mkfifo fifo/rl_IP1_S1_summary_P2
mkfifo fifo/rl_IP1_S1_summary_P2.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P2
mkfifo fifo/rl_IP1_S1_selt_ord_P2
mkfifo fifo/rl_IP1_S1_summary_P3
mkfifo fifo/rl_IP1_S1_summary_P3.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P3
mkfifo fifo/rl_IP1_S1_selt_ord_P3
mkfifo fifo/rl_IP1_S1_summary_P4
mkfifo fifo/rl_IP1_S1_summary_P4.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P4
mkfifo fifo/rl_IP1_S1_selt_ord_P4
mkfifo fifo/rl_IP1_S1_summary_P5
mkfifo fifo/rl_IP1_S1_summary_P5.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P5
mkfifo fifo/rl_IP1_S1_selt_ord_P5
mkfifo fifo/rl_IP1_S1_summary_P6
mkfifo fifo/rl_IP1_S1_summary_P6.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P6
mkfifo fifo/rl_IP1_S1_selt_ord_P6
mkfifo fifo/rl_IP1_S1_summary_P7
mkfifo fifo/rl_IP1_S1_summary_P7.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P7
mkfifo fifo/rl_IP1_S1_selt_ord_P7
mkfifo fifo/rl_IP1_S1_summary_P8
mkfifo fifo/rl_IP1_S1_summary_P8.idx
mkfifo fifo/rl_IP1_S1_plt_ord_P8
mkfifo fifo/rl_IP1_S1_selt_ord_P8
mkfifo fifo/rl_S1_summary_P1
mkfifo fifo/rl_S1_summary_P1.idx
mkfifo fifo/rl_S1_plt_ord_P1
mkfifo fifo/rl_S1_selt_ord_P1
mkfifo fifo/rl_S1_summary_P2
mkfifo fifo/rl_S1_summary_P2.idx
mkfifo fifo/rl_S1_plt_ord_P2
mkfifo fifo/rl_S1_selt_ord_P2
mkfifo fifo/rl_S1_summary_P3
mkfifo fifo/rl_S1_summary_P3.idx
mkfifo fifo/rl_S1_plt_ord_P3
mkfifo fifo/rl_S1_selt_ord_P3
mkfifo fifo/rl_S1_summary_P4
mkfifo fifo/rl_S1_summary_P4.idx
mkfifo fifo/rl_S1_plt_ord_P4
mkfifo fifo/rl_S1_selt_ord_P4
mkfifo fifo/rl_S1_summary_P5
mkfifo fifo/rl_S1_summary_P5.idx
mkfifo fifo/rl_S1_plt_ord_P5
mkfifo fifo/rl_S1_selt_ord_P5
mkfifo fifo/rl_S1_summary_P6
mkfifo fifo/rl_S1_summary_P6.idx
mkfifo fifo/rl_S1_plt_ord_P6
mkfifo fifo/rl_S1_selt_ord_P6
mkfifo fifo/rl_S1_summary_P7
mkfifo fifo/rl_S1_summary_P7.idx
mkfifo fifo/rl_S1_plt_ord_P7
mkfifo fifo/rl_S1_selt_ord_P7
mkfifo fifo/rl_S1_summary_P8
mkfifo fifo/rl_S1_summary_P8.idx
mkfifo fifo/rl_S1_plt_ord_P8
mkfifo fifo/rl_S1_selt_ord_P8


# --- Do reinsurance gross loss computes ---

pltpy -E bin  -s work/kat/rl_IP1_S1_plt_sample_P1 < fifo/rl_IP1_S1_plt_ord_P1 & pid1=$!
eltpy -E bin  -s work/kat/rl_IP1_S1_elt_sample_P1 < fifo/rl_IP1_S1_selt_ord_P1 & pid2=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P2 < fifo/rl_IP1_S1_plt_ord_P2 & pid3=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P2 < fifo/rl_IP1_S1_selt_ord_P2 & pid4=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P3 < fifo/rl_IP1_S1_plt_ord_P3 & pid5=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P3 < fifo/rl_IP1_S1_selt_ord_P3 & pid6=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P4 < fifo/rl_IP1_S1_plt_ord_P4 & pid7=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P4 < fifo/rl_IP1_S1_selt_ord_P4 & pid8=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P5 < fifo/rl_IP1_S1_plt_ord_P5 & pid9=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P5 < fifo/rl_IP1_S1_selt_ord_P5 & pid10=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P6 < fifo/rl_IP1_S1_plt_ord_P6 & pid11=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P6 < fifo/rl_IP1_S1_selt_ord_P6 & pid12=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P7 < fifo/rl_IP1_S1_plt_ord_P7 & pid13=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P7 < fifo/rl_IP1_S1_selt_ord_P7 & pid14=$!
pltpy -E bin  -H -s work/kat/rl_IP1_S1_plt_sample_P8 < fifo/rl_IP1_S1_plt_ord_P8 & pid15=$!
eltpy -E bin  -H -s work/kat/rl_IP1_S1_elt_sample_P8 < fifo/rl_IP1_S1_selt_ord_P8 & pid16=$!

tee < fifo/rl_IP1_S1_summary_P1 fifo/rl_IP1_S1_plt_ord_P1 fifo/rl_IP1_S1_selt_ord_P1 work/rl_IP1_S1_summary_palt/P1.bin work/rl_IP1_S1_summaryleccalc/P1.bin > /dev/null & pid17=$!
tee < fifo/rl_IP1_S1_summary_P1.idx work/rl_IP1_S1_summary_palt/P1.idx work/rl_IP1_S1_summaryleccalc/P1.idx > /dev/null & pid18=$!
tee < fifo/rl_IP1_S1_summary_P2 fifo/rl_IP1_S1_plt_ord_P2 fifo/rl_IP1_S1_selt_ord_P2 work/rl_IP1_S1_summary_palt/P2.bin work/rl_IP1_S1_summaryleccalc/P2.bin > /dev/null & pid19=$!
tee < fifo/rl_IP1_S1_summary_P2.idx work/rl_IP1_S1_summary_palt/P2.idx work/rl_IP1_S1_summaryleccalc/P2.idx > /dev/null & pid20=$!
tee < fifo/rl_IP1_S1_summary_P3 fifo/rl_IP1_S1_plt_ord_P3 fifo/rl_IP1_S1_selt_ord_P3 work/rl_IP1_S1_summary_palt/P3.bin work/rl_IP1_S1_summaryleccalc/P3.bin > /dev/null & pid21=$!
tee < fifo/rl_IP1_S1_summary_P3.idx work/rl_IP1_S1_summary_palt/P3.idx work/rl_IP1_S1_summaryleccalc/P3.idx > /dev/null & pid22=$!
tee < fifo/rl_IP1_S1_summary_P4 fifo/rl_IP1_S1_plt_ord_P4 fifo/rl_IP1_S1_selt_ord_P4 work/rl_IP1_S1_summary_palt/P4.bin work/rl_IP1_S1_summaryleccalc/P4.bin > /dev/null & pid23=$!
tee < fifo/rl_IP1_S1_summary_P4.idx work/rl_IP1_S1_summary_palt/P4.idx work/rl_IP1_S1_summaryleccalc/P4.idx > /dev/null & pid24=$!
tee < fifo/rl_IP1_S1_summary_P5 fifo/rl_IP1_S1_plt_ord_P5 fifo/rl_IP1_S1_selt_ord_P5 work/rl_IP1_S1_summary_palt/P5.bin work/rl_IP1_S1_summaryleccalc/P5.bin > /dev/null & pid25=$!
tee < fifo/rl_IP1_S1_summary_P5.idx work/rl_IP1_S1_summary_palt/P5.idx work/rl_IP1_S1_summaryleccalc/P5.idx > /dev/null & pid26=$!
tee < fifo/rl_IP1_S1_summary_P6 fifo/rl_IP1_S1_plt_ord_P6 fifo/rl_IP1_S1_selt_ord_P6 work/rl_IP1_S1_summary_palt/P6.bin work/rl_IP1_S1_summaryleccalc/P6.bin > /dev/null & pid27=$!
tee < fifo/rl_IP1_S1_summary_P6.idx work/rl_IP1_S1_summary_palt/P6.idx work/rl_IP1_S1_summaryleccalc/P6.idx > /dev/null & pid28=$!
tee < fifo/rl_IP1_S1_summary_P7 fifo/rl_IP1_S1_plt_ord_P7 fifo/rl_IP1_S1_selt_ord_P7 work/rl_IP1_S1_summary_palt/P7.bin work/rl_IP1_S1_summaryleccalc/P7.bin > /dev/null & pid29=$!
tee < fifo/rl_IP1_S1_summary_P7.idx work/rl_IP1_S1_summary_palt/P7.idx work/rl_IP1_S1_summaryleccalc/P7.idx > /dev/null & pid30=$!
tee < fifo/rl_IP1_S1_summary_P8 fifo/rl_IP1_S1_plt_ord_P8 fifo/rl_IP1_S1_selt_ord_P8 work/rl_IP1_S1_summary_palt/P8.bin work/rl_IP1_S1_summaryleccalc/P8.bin > /dev/null & pid31=$!
tee < fifo/rl_IP1_S1_summary_P8.idx work/rl_IP1_S1_summary_palt/P8.idx work/rl_IP1_S1_summaryleccalc/P8.idx > /dev/null & pid32=$!

pltpy -E bin  -s work/kat/rl_S1_plt_sample_P1 < fifo/rl_S1_plt_ord_P1 & pid33=$!
eltpy -E bin  -s work/kat/rl_S1_elt_sample_P1 < fifo/rl_S1_selt_ord_P1 & pid34=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P2 < fifo/rl_S1_plt_ord_P2 & pid35=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P2 < fifo/rl_S1_selt_ord_P2 & pid36=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P3 < fifo/rl_S1_plt_ord_P3 & pid37=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P3 < fifo/rl_S1_selt_ord_P3 & pid38=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P4 < fifo/rl_S1_plt_ord_P4 & pid39=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P4 < fifo/rl_S1_selt_ord_P4 & pid40=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P5 < fifo/rl_S1_plt_ord_P5 & pid41=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P5 < fifo/rl_S1_selt_ord_P5 & pid42=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P6 < fifo/rl_S1_plt_ord_P6 & pid43=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P6 < fifo/rl_S1_selt_ord_P6 & pid44=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P7 < fifo/rl_S1_plt_ord_P7 & pid45=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P7 < fifo/rl_S1_selt_ord_P7 & pid46=$!
pltpy -E bin  -H -s work/kat/rl_S1_plt_sample_P8 < fifo/rl_S1_plt_ord_P8 & pid47=$!
eltpy -E bin  -H -s work/kat/rl_S1_elt_sample_P8 < fifo/rl_S1_selt_ord_P8 & pid48=$!

tee < fifo/rl_S1_summary_P1 fifo/rl_S1_plt_ord_P1 fifo/rl_S1_selt_ord_P1 work/rl_S1_summary_palt/P1.bin work/rl_S1_summaryleccalc/P1.bin > /dev/null & pid49=$!
tee < fifo/rl_S1_summary_P1.idx work/rl_S1_summary_palt/P1.idx work/rl_S1_summaryleccalc/P1.idx > /dev/null & pid50=$!
tee < fifo/rl_S1_summary_P2 fifo/rl_S1_plt_ord_P2 fifo/rl_S1_selt_ord_P2 work/rl_S1_summary_palt/P2.bin work/rl_S1_summaryleccalc/P2.bin > /dev/null & pid51=$!
tee < fifo/rl_S1_summary_P2.idx work/rl_S1_summary_palt/P2.idx work/rl_S1_summaryleccalc/P2.idx > /dev/null & pid52=$!
tee < fifo/rl_S1_summary_P3 fifo/rl_S1_plt_ord_P3 fifo/rl_S1_selt_ord_P3 work/rl_S1_summary_palt/P3.bin work/rl_S1_summaryleccalc/P3.bin > /dev/null & pid53=$!
tee < fifo/rl_S1_summary_P3.idx work/rl_S1_summary_palt/P3.idx work/rl_S1_summaryleccalc/P3.idx > /dev/null & pid54=$!
tee < fifo/rl_S1_summary_P4 fifo/rl_S1_plt_ord_P4 fifo/rl_S1_selt_ord_P4 work/rl_S1_summary_palt/P4.bin work/rl_S1_summaryleccalc/P4.bin > /dev/null & pid55=$!
tee < fifo/rl_S1_summary_P4.idx work/rl_S1_summary_palt/P4.idx work/rl_S1_summaryleccalc/P4.idx > /dev/null & pid56=$!
tee < fifo/rl_S1_summary_P5 fifo/rl_S1_plt_ord_P5 fifo/rl_S1_selt_ord_P5 work/rl_S1_summary_palt/P5.bin work/rl_S1_summaryleccalc/P5.bin > /dev/null & pid57=$!
tee < fifo/rl_S1_summary_P5.idx work/rl_S1_summary_palt/P5.idx work/rl_S1_summaryleccalc/P5.idx > /dev/null & pid58=$!
tee < fifo/rl_S1_summary_P6 fifo/rl_S1_plt_ord_P6 fifo/rl_S1_selt_ord_P6 work/rl_S1_summary_palt/P6.bin work/rl_S1_summaryleccalc/P6.bin > /dev/null & pid59=$!
tee < fifo/rl_S1_summary_P6.idx work/rl_S1_summary_palt/P6.idx work/rl_S1_summaryleccalc/P6.idx > /dev/null & pid60=$!
tee < fifo/rl_S1_summary_P7 fifo/rl_S1_plt_ord_P7 fifo/rl_S1_selt_ord_P7 work/rl_S1_summary_palt/P7.bin work/rl_S1_summaryleccalc/P7.bin > /dev/null & pid61=$!
tee < fifo/rl_S1_summary_P7.idx work/rl_S1_summary_palt/P7.idx work/rl_S1_summaryleccalc/P7.idx > /dev/null & pid62=$!
tee < fifo/rl_S1_summary_P8 fifo/rl_S1_plt_ord_P8 fifo/rl_S1_selt_ord_P8 work/rl_S1_summary_palt/P8.bin work/rl_S1_summaryleccalc/P8.bin > /dev/null & pid63=$!
tee < fifo/rl_S1_summary_P8.idx work/rl_S1_summary_palt/P8.idx work/rl_S1_summaryleccalc/P8.idx > /dev/null & pid64=$!

rlcombine -i fifo/rl_IP1_S1_rlcombine_P1 fifo/rl_IP2_S1_rlcombine_P1 fifo/rl_IP3_S1_rlcombine_P1 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P1 -I 0=fifo/rl_IP1_S1_summary_P1 -m & pid65=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P2 fifo/rl_IP2_S1_rlcombine_P2 fifo/rl_IP3_S1_rlcombine_P2 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P2 -I 0=fifo/rl_IP1_S1_summary_P2 -m & pid66=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P3 fifo/rl_IP2_S1_rlcombine_P3 fifo/rl_IP3_S1_rlcombine_P3 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P3 -I 0=fifo/rl_IP1_S1_summary_P3 -m & pid67=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P4 fifo/rl_IP2_S1_rlcombine_P4 fifo/rl_IP3_S1_rlcombine_P4 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P4 -I 0=fifo/rl_IP1_S1_summary_P4 -m & pid68=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P5 fifo/rl_IP2_S1_rlcombine_P5 fifo/rl_IP3_S1_rlcombine_P5 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P5 -I 0=fifo/rl_IP1_S1_summary_P5 -m & pid69=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P6 fifo/rl_IP2_S1_rlcombine_P6 fifo/rl_IP3_S1_rlcombine_P6 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P6 -I 0=fifo/rl_IP1_S1_summary_P6 -m & pid70=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P7 fifo/rl_IP2_S1_rlcombine_P7 fifo/rl_IP3_S1_rlcombine_P7 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P7 -I 0=fifo/rl_IP1_S1_summary_P7 -m & pid71=$!
rlcombine -i fifo/rl_IP1_S1_rlcombine_P8 fifo/rl_IP2_S1_rlcombine_P8 fifo/rl_IP3_S1_rlcombine_P8 -p input/RI_1 input/RI_2 input/RI_3 -o fifo/rl_S1_summary_P8 -I 0=fifo/rl_IP1_S1_summary_P8 -m & pid72=$!

summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P1 < fifo/rl_IP1_P1 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P2 < fifo/rl_IP1_P2 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P3 < fifo/rl_IP1_P3 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P4 < fifo/rl_IP1_P4 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P5 < fifo/rl_IP1_P5 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P6 < fifo/rl_IP1_P6 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P7 < fifo/rl_IP1_P7 &
summarypy -s -t rl -p input/RI_1 -1 fifo/rl_IP1_S1_rlcombine_P8 < fifo/rl_IP1_P8 &

summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P1 < fifo/rl_IP2_P1 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P2 < fifo/rl_IP2_P2 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P3 < fifo/rl_IP2_P3 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P4 < fifo/rl_IP2_P4 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P5 < fifo/rl_IP2_P5 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P6 < fifo/rl_IP2_P6 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P7 < fifo/rl_IP2_P7 &
summarypy -s -t rl -p input/RI_2 -1 fifo/rl_IP2_S1_rlcombine_P8 < fifo/rl_IP2_P8 &

summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P1 < fifo/rl_IP3_P1 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P2 < fifo/rl_IP3_P2 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P3 < fifo/rl_IP3_P3 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P4 < fifo/rl_IP3_P4 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P5 < fifo/rl_IP3_P5 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P6 < fifo/rl_IP3_P6 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P7 < fifo/rl_IP3_P7 &
summarypy -s -t rl -p input/RI_3 -1 fifo/rl_IP3_S1_rlcombine_P8 < fifo/rl_IP3_P8 &


# --- Verify FIFO pipes ---
check_fifos \
    fifo/rl_IP1_P1 \
    fifo/rl_IP1_P2 \
    fifo/rl_IP1_P3 \
    fifo/rl_IP1_P4 \
    fifo/rl_IP1_P5 \
    fifo/rl_IP1_P6 \
    fifo/rl_IP1_P7 \
    fifo/rl_IP1_P8 \
    fifo/rl_IP1_S1_rlcombine_P1 \
    fifo/rl_IP1_S1_rlcombine_P2 \
    fifo/rl_IP1_S1_rlcombine_P3 \
    fifo/rl_IP1_S1_rlcombine_P4 \
    fifo/rl_IP1_S1_rlcombine_P5 \
    fifo/rl_IP1_S1_rlcombine_P6 \
    fifo/rl_IP1_S1_rlcombine_P7 \
    fifo/rl_IP1_S1_rlcombine_P8 \
    fifo/rl_IP2_P1 \
    fifo/rl_IP2_P2 \
    fifo/rl_IP2_P3 \
    fifo/rl_IP2_P4 \
    fifo/rl_IP2_P5 \
    fifo/rl_IP2_P6 \
    fifo/rl_IP2_P7 \
    fifo/rl_IP2_P8 \
    fifo/rl_IP2_S1_rlcombine_P1 \
    fifo/rl_IP2_S1_rlcombine_P2 \
    fifo/rl_IP2_S1_rlcombine_P3 \
    fifo/rl_IP2_S1_rlcombine_P4 \
    fifo/rl_IP2_S1_rlcombine_P5 \
    fifo/rl_IP2_S1_rlcombine_P6 \
    fifo/rl_IP2_S1_rlcombine_P7 \
    fifo/rl_IP2_S1_rlcombine_P8 \
    fifo/rl_IP3_P1 \
    fifo/rl_IP3_P2 \
    fifo/rl_IP3_P3 \
    fifo/rl_IP3_P4 \
    fifo/rl_IP3_P5 \
    fifo/rl_IP3_P6 \
    fifo/rl_IP3_P7 \
    fifo/rl_IP3_P8 \
    fifo/rl_IP3_S1_rlcombine_P1 \
    fifo/rl_IP3_S1_rlcombine_P2 \
    fifo/rl_IP3_S1_rlcombine_P3 \
    fifo/rl_IP3_S1_rlcombine_P4 \
    fifo/rl_IP3_S1_rlcombine_P5 \
    fifo/rl_IP3_S1_rlcombine_P6 \
    fifo/rl_IP3_S1_rlcombine_P7 \
    fifo/rl_IP3_S1_rlcombine_P8 \
    fifo/rl_IP1_S1_summary_P1 \
    fifo/rl_IP1_S1_summary_P1.idx \
    fifo/rl_IP1_S1_plt_ord_P1 \
    fifo/rl_IP1_S1_selt_ord_P1 \
    fifo/rl_IP1_S1_summary_P2 \
    fifo/rl_IP1_S1_summary_P2.idx \
    fifo/rl_IP1_S1_plt_ord_P2 \
    fifo/rl_IP1_S1_selt_ord_P2 \
    fifo/rl_IP1_S1_summary_P3 \
    fifo/rl_IP1_S1_summary_P3.idx \
    fifo/rl_IP1_S1_plt_ord_P3 \
    fifo/rl_IP1_S1_selt_ord_P3 \
    fifo/rl_IP1_S1_summary_P4 \
    fifo/rl_IP1_S1_summary_P4.idx \
    fifo/rl_IP1_S1_plt_ord_P4 \
    fifo/rl_IP1_S1_selt_ord_P4 \
    fifo/rl_IP1_S1_summary_P5 \
    fifo/rl_IP1_S1_summary_P5.idx \
    fifo/rl_IP1_S1_plt_ord_P5 \
    fifo/rl_IP1_S1_selt_ord_P5 \
    fifo/rl_IP1_S1_summary_P6 \
    fifo/rl_IP1_S1_summary_P6.idx \
    fifo/rl_IP1_S1_plt_ord_P6 \
    fifo/rl_IP1_S1_selt_ord_P6 \
    fifo/rl_IP1_S1_summary_P7 \
    fifo/rl_IP1_S1_summary_P7.idx \
    fifo/rl_IP1_S1_plt_ord_P7 \
    fifo/rl_IP1_S1_selt_ord_P7 \
    fifo/rl_IP1_S1_summary_P8 \
    fifo/rl_IP1_S1_summary_P8.idx \
    fifo/rl_IP1_S1_plt_ord_P8 \
    fifo/rl_IP1_S1_selt_ord_P8 \
    fifo/rl_S1_summary_P1 \
    fifo/rl_S1_summary_P1.idx \
    fifo/rl_S1_plt_ord_P1 \
    fifo/rl_S1_selt_ord_P1 \
    fifo/rl_S1_summary_P2 \
    fifo/rl_S1_summary_P2.idx \
    fifo/rl_S1_plt_ord_P2 \
    fifo/rl_S1_selt_ord_P2 \
    fifo/rl_S1_summary_P3 \
    fifo/rl_S1_summary_P3.idx \
    fifo/rl_S1_plt_ord_P3 \
    fifo/rl_S1_selt_ord_P3 \
    fifo/rl_S1_summary_P4 \
    fifo/rl_S1_summary_P4.idx \
    fifo/rl_S1_plt_ord_P4 \
    fifo/rl_S1_selt_ord_P4 \
    fifo/rl_S1_summary_P5 \
    fifo/rl_S1_summary_P5.idx \
    fifo/rl_S1_plt_ord_P5 \
    fifo/rl_S1_selt_ord_P5 \
    fifo/rl_S1_summary_P6 \
    fifo/rl_S1_summary_P6.idx \
    fifo/rl_S1_plt_ord_P6 \
    fifo/rl_S1_selt_ord_P6 \
    fifo/rl_S1_summary_P7 \
    fifo/rl_S1_summary_P7.idx \
    fifo/rl_S1_plt_ord_P7 \
    fifo/rl_S1_selt_ord_P7 \
    fifo/rl_S1_summary_P8 \
    fifo/rl_S1_summary_P8.idx \
    fifo/rl_S1_plt_ord_P8 \
    fifo/rl_S1_selt_ord_P8

( evepy 1 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P1 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P1 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P1 ) & pid73=$!
( evepy 2 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P2 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P2 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P2 ) & pid74=$!
( evepy 3 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P3 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P3 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P3 ) & pid75=$!
( evepy 4 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P4 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P4 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P4 ) & pid76=$!
( evepy 5 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P5 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P5 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P5 ) & pid77=$!
( evepy 6 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P6 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P6 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P6 ) & pid78=$!
( evepy 7 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P7 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P7 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P7 ) & pid79=$!
( evepy 8 8 | gulmc --random-generator=2  --model-df-engine='oasis_data_manager.df_reader.reader.OasisPandasReader' --vuln-cache-size 200 -S100 -L100 -a0  | fmpy -a2 | fmpy -a2 -p input/RI_1 -o fifo/rl_IP1_P8 -n - | fmpy -a2 -p input/RI_2 -o fifo/rl_IP2_P8 -n - | fmpy -a2 -p input/RI_3 -o fifo/rl_IP3_P8 ) & pid80=$!

exec_wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8 $pid9 $pid10 $pid11 $pid12 $pid13 $pid14 $pid15 $pid16 $pid17 $pid18 $pid19 $pid20 $pid21 $pid22 $pid23 $pid24 $pid25 $pid26 $pid27 $pid28 $pid29 $pid30 $pid31 $pid32 $pid33 $pid34 $pid35 $pid36 $pid37 $pid38 $pid39 $pid40 $pid41 $pid42 $pid43 $pid44 $pid45 $pid46 $pid47 $pid48 $pid49 $pid50 $pid51 $pid52 $pid53 $pid54 $pid55 $pid56 $pid57 $pid58 $pid59 $pid60 $pid61 $pid62 $pid63 $pid64 $pid65 $pid66 $pid67 $pid68 $pid69 $pid70 $pid71 $pid72 $pid73 $pid74 $pid75 $pid76 $pid77 $pid78 $pid79 $pid80


# --- Do reinsurance gross loss kats ---

katpy -S -f bin -i work/kat/rl_IP1_S1_plt_sample_P1 work/kat/rl_IP1_S1_plt_sample_P2 work/kat/rl_IP1_S1_plt_sample_P3 work/kat/rl_IP1_S1_plt_sample_P4 work/kat/rl_IP1_S1_plt_sample_P5 work/kat/rl_IP1_S1_plt_sample_P6 work/kat/rl_IP1_S1_plt_sample_P7 work/kat/rl_IP1_S1_plt_sample_P8 -o output/rl_IP1_S1_splt.csv & kpid1=$!
katpy -s -f bin -i work/kat/rl_IP1_S1_elt_sample_P1 work/kat/rl_IP1_S1_elt_sample_P2 work/kat/rl_IP1_S1_elt_sample_P3 work/kat/rl_IP1_S1_elt_sample_P4 work/kat/rl_IP1_S1_elt_sample_P5 work/kat/rl_IP1_S1_elt_sample_P6 work/kat/rl_IP1_S1_elt_sample_P7 work/kat/rl_IP1_S1_elt_sample_P8 -o output/rl_IP1_S1_selt.csv & kpid2=$!
katpy -S -f bin -i work/kat/rl_S1_plt_sample_P1 work/kat/rl_S1_plt_sample_P2 work/kat/rl_S1_plt_sample_P3 work/kat/rl_S1_plt_sample_P4 work/kat/rl_S1_plt_sample_P5 work/kat/rl_S1_plt_sample_P6 work/kat/rl_S1_plt_sample_P7 work/kat/rl_S1_plt_sample_P8 -o output/rl_S1_splt.csv & kpid3=$!
katpy -s -f bin -i work/kat/rl_S1_elt_sample_P1 work/kat/rl_S1_elt_sample_P2 work/kat/rl_S1_elt_sample_P3 work/kat/rl_S1_elt_sample_P4 work/kat/rl_S1_elt_sample_P5 work/kat/rl_S1_elt_sample_P6 work/kat/rl_S1_elt_sample_P7 work/kat/rl_S1_elt_sample_P8 -o output/rl_S1_selt.csv & kpid4=$!
exec_wait $kpid1 $kpid2 $kpid3 $kpid4


aalpy -Krl_IP1_S1_summary_palt -a output/rl_IP1_S1_palt.csv & lpid1=$!
lecpy -r -Krl_IP1_S1_summaryleccalc -F -f -O output/rl_IP1_S1_ept.csv & lpid2=$!
aalpy -Krl_S1_summary_palt -a output/rl_S1_palt.csv & lpid3=$!
lecpy -r -Krl_S1_summaryleccalc -F -f -O output/rl_S1_ept.csv & lpid4=$!
exec_wait $lpid1 $lpid2 $lpid3 $lpid4


# --- removing working files
rm -R -f work/*
rm -R -f fifo/*
