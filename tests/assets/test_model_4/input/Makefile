#
# Generate binary files from .csv files
# See https://github.com/OasisLMF/ktools/blob/master/docs/md/DataConversionComponents.md
#

# Binary file we want to generate
#
BIN_FILES= \
events.bin \
items.bin \
coverages.bin

#
# Standard targets
#

all: $(BIN_FILES)

clean:
	rm -f $(BIN_FILES)

#
# OASIS Specifc targets
#

# uncomment this when correlationstobin is implemented
# correlations.bin: correlations.csv
# 	correlationstobin < correlations.csv > correlations.bin
	
coverages.bin: coverages.csv
	coveragetobin < coverages.csv > coverages.bin

events.bin: events.csv
	evetobin < events.csv > events.bin

items.bin: items.csv
	itemtobin < items.csv > items.bin
