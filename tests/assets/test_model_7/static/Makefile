#
# Generate binary files from .csv files
# See https://github.com/OasisLMF/ktools/blob/master/docs/md/DataConversionComponents.md
#

#
# Model specific parameters
#

MAX_PERIODS=1500

# extract number of intensity bins from intensity_bin_dict.csv
INTENSITY_BINS := $(shell awk -F',' '{print $$1}' intensity_bin_dict.csv | sort -n | tail -n 1)

# extract number of damage bins from damage_bin_dict.csv
DAMAGE_BINS := $(shell awk -F',' '{print $$1}' damage_bin_dict.csv | sort -n | tail -n 1)

#
# Binary file we want to generate
#
BIN_FILES= \
damage_bin_dict.bin \
footprint.bin \
footprint.idx \
vulnerability.bin
# events.bin
# events_p.bin \
# occurrence.bin \
returnperiods.bin


#
# Standard targets
#

all: $(BIN_FILES)

clean:
	rm -f $(BIN_FILES)

#
# OASIS Specifc targets
#

damage_bin_dict.bin: damage_bin_dict.csv
	damagebintobin < damage_bin_dict.csv > damage_bin_dict.bin

footprint.bin: footprint.csv
	footprinttobin -i $(INTENSITY_BINS) < footprint.csv > footprint.bin

footprint.csv.gz:
	wget -N -O footprint.csv.gz http://sunstonerisk.com/files/domrep/footprint.csv.gz

# Need to suppress tests as first damage_bin is not 1
vulnerability.bin: vulnerability.csv
	vulnerabilitytobin -d $(DAMAGE_BINS) -N \
		< vulnerability.csv > vulnerability.bin

events.bin: events.csv
	evetobin < events.csv > events.bin

events_p.bin: events.bin
	cp events.bin events_p.bin

items.bin: items.csv
	itemtobin < items.csv > items.bin

occurrence.bin: occurrence.csv intensity_bin_dict.csv
	occurrencetobin -P$(MAX_PERIODS) < occurrence.csv > occurrence.bin

returnperiods.bin: returnperiods.csv
	returnperiodtobin < returnperiods.csv > returnperiods.bin

periods.bin: periods.csv
	periodstobin < periods.csv > periods.bin
