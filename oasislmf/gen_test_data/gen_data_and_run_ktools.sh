#!/bin/bash

# -------- PART 1: GENERATE DATA --------
# Set up input variables
numVulnerabilities=20
numIntensityBins=40
numDamageBins=12
vulnerabilitySparseness=1.0		# default value 1.0
numEvents=100
numAreaperils=100
areaperilsPerEvent=$numAreaperils	# default value numAreaperils
intensitySparseness=1.0			# default value 1.0
noIntensityUncertainty=false		# default value False
numPeriods=10
numRandoms=0				# default value 0 (no random.bin file)
randomSeed=-1				# default value -1 (random seed 1234)
numLocations=100
coveragesPerLocation=3
numLayers=1				# default value 1

if [ "$noIntensityUncertainty" = true ] ; then
	noIntensityUncertainty="--no-intensity-uncertainty"
else
	noIntensityUncertainty=""
fi

# Generate data
python generate_test_data.py --num-vulnerabilities $numVulnerabilities --num-intensity-bins $numIntensityBins --num-damage-bins $numDamageBins --vulnerability-sparseness $vulnerabilitySparseness --num-events $numEvents --num-areaperils $numAreaperils --areaperils-per-event $areaperilsPerEvent --intensity-sparseness $intensitySparseness --num-periods $numPeriods --num-randoms $numRandoms --random-seed $randomSeed --num-locations $numLocations --coverages-per-location $coveragesPerLocation --num-layers $numLayers $noIntensityUncertainty

# -------- PART 2: RUN KTOOLS --------
# Set up run directories

rm -rf log
mkdir log/

rm -rf work/
mkdir work/
mkdir work/kat
mkdir work/gul_S1_summaryleccalc
mkdir work/gul_S1_summaryaalcalc
mkdir work/il_S1_summaryleccalc
mkdir work/il_S1_summaryaalcalc

rm -rf fifo/
mkdir fifo/

rm -rf output
mkdir output

mkfifo fifo/gul_P1

mkfifo fifo/il_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summaryeltcalc_P1
mkfifo fifo/gul_S1_eltcalc_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1

# Do insured loss computes

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 work/il_S1_summaryaalcalc/P1.bin work/il_S1_summaryleccalc/P1.bin > /dev/null & pid2=$!

( summarycalc -f -1 fifo/il_S1_summary_P1 < fifo/il_P1 ) 2>> log/stderror.err &

# Do ground up loss computes

eltcalc < fifo/gul_S1_summaryeltcalc_P1 > work/kat/gul_S1_eltcalc_P1 & pid3=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryeltcalc_P1 work/gul_S1_summaryaalcalc/P1.bin work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid4=$!

( summarycalc -f -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 ) 2>> log/stderror.err &

( eve 1 1 | getmodel | gulcalc -S10 -L0 -a0 -i - | tee fifo/gul_P1 | fmcalc -a2 > fifo/il_P1 ) 2>> log/stderror.err &

wait $pid1 $pid2 $pid3 $pid4

# Do insured loss kats

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid1=$!

# Do ground up loss kats

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid2=$!

wait $kpid1 $kpid2

aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
leccalc -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv & lpid2=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid3=$!
leccalc -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4
