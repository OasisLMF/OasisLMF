# -*- coding: utf-8 -*-
import json
import logging

__all__ = [
    'Translator'
]

import csv
import io
import os
import pandas as pd

from itertools import islice

from lxml import etree


class Translator(object):
    def __init__(self, input_path, output_path, xsd_path, xslt_path, append_row_nums=False, chunk_size=5000, logger=None):
        self.logger = logger or logging.getLogger()
        self.threshold = 100000000
        # Max file size for file_ReadAll() method, in bytes [100MB]

        self.xsd = etree.parse(xsd_path)   # validation file
        self.xslt = etree.parse(xslt_path)  # transform file
        self.fpath_input = input_path        # file_in.csv
        self.fpath_output = output_path       # file_out.csv
        self.ext = os.path.splitext(self.fpath_output)[1]

        # Row Control Vars
        self.row_nums = append_row_nums  # Is 's' or 'S' as well?
        self.row_limit = chunk_size    # MAX number of CSV input rows
        self.row_start = 1             # Start of input file segment to process
        self.row_end = self.row_limit  # end of input file segment to process
        self.row_header_in = None      # CSV col header
        self.row_header_out = None     # CSV col header Post Transform

        # Input Validation
        if (self.ext == ''):
            raise TypeError("Missing file extention for output file.")


# --- Main loop --------------------------------------------------------------#

    def __call__(self):

        # read input CSV header
        with io.open(self.fpath_input, 'r') as fd_input:
            self.row_header_in = self.read_file_slice(fd_input, 0, 0)[0]

            for data, first_row_number, last_row_number in self.next_file_slice(fd_input):
                self.logger.debug('--- lines[%d .. %d] -------------------------------', first_row_number, last_row_number)

                # Convert CSV -> XML
                xml_input_slice = self.csv_to_xml(
                    self.row_header_in,
                    data
                )
                self.print_xml(xml_input_slice)

                # Transform
                xml_output = self.xml_transform(xml_input_slice, self.xslt)
                self.print_xml(xml_output)

                # Validate Output
                self.logger.info(self.xml_validate(xml_output, self.xsd))

                # Convert transform XML back to CSV
                self.xml_to_csv(
                    xml_output,          # XML etree
                    first_row_number,  # First Row in this slice
                    last_row_number    # Last Row in this slice
                )

# --- Transform Functions ----------------------------------------------------#
# https://pymotw.com/2/xml/etree/ElementTree/create.html
# http://lxml.de/api/lxml.etree._Element-class.html

    # --- CSV Funcs --- #
    def csv_to_xml(self, csv_header, csv_data):
        root = etree.Element('root')
        # fetch each row
        for row in csv_data:
            # Create new 'empty' record
            rec = etree.SubElement(root, 'rec')
            # Iter over columns and set attributs
            for i in range(0, len(row)):
                rec.set(self.row_header_in[i], row[i])
        return root

    def csv_insert_row_nums(self, csv_data, r_start, r_end):
        row_index = range(r_start, r_end)
        row_total = r_end - r_start
        if(row_total != len(csv_data)):
            raise TypeError('Size mismatch between row numbering and dataset')
        return [[row_index[i]] + csv_data[i] for i in range(0, row_total)]

    # --- XML Funcs --- #
    def xml_to_csv(self, xml_elementTree, row_first, row_last):
        root = xml_elementTree.getroot()

        # Check if this is the first file chunk processed
        # and Extract the New CSV header
        if not (self.row_header_out):
            self.row_header_out = root[0].keys()

            if self.row_nums:
                self.row_header_out.insert(0, 'ROW_ID')
            # Append first line to output csv file
            self.write_file_header(self.row_header_out)

        # Convert each chunk into Python Dict then pass to:
        #     class csv.DictWriter(csvfile ... )
        #     see: https://docs.python.org/2/library/csv.html

        line_counter = row_first

        for rec in root:
            # print([rec.get(key) for key in self.row_header_out])

            # Convert Row record to python dict() Object
            rec_d = rec.attrib

            # append ROW_ID
            if self.row_nums:
                rec_d['ROW_ID'] = str(line_counter)
                line_counter += 1

            # Append to output file
            self.print_dict(rec_d)
            self.append_file_row(self.row_header_out, rec_d)

        # guard for correct Row numbering
        if ((self.row_nums) and (line_counter != row_last)):
            self.logger.error('Line_count: %d, Row_last: %d', line_counter, row_last)

            raise KeyError('Row_ID missmatch')

    # http://lxml.de/2.0/validation.html
    # If valid   -> Return True
    #    invalid -> error_log
    def xml_validate(self, xml_etree, xsd_etree):
        xmlSchema = etree.XMLSchema(xsd_etree)
        self.print_xml(xml_etree)
        self.print_xml(xsd_etree)
        # Calling 'assertValid' will raise execptions,
        # --> should be handled above this level
        # if (xmlSchema.assertValid(xml_etree)):
        if (xmlSchema.validate(xml_etree)):
            return True
        else:
            self.logger.error('Input failed to Validate')
            log = xmlSchema.error_log
            self.logger.error(log.last_error)
            return False

    # http://lxml.de/xpathxslt.html#xslt
    def xml_transform(self, xml_doc, xslt):
        lxml_transform = etree.XSLT(self.xslt)
        return lxml_transform(xml_doc)


# --- File I/O Functions -----------------------------------------------------#

    # Generator Function which processes and returns batches of the input CSV
    def next_file_slice(self, file_object):
        while True:
            csv_chunk = self.read_file_slice(
                file_object,
                self.row_start,
                self.row_end
            )

            # Exit check for EOF
            rowsInChunk = len(csv_chunk)

            if (rowsInChunk == 0):
                break
            else:
                slice_start = self.row_start
                slice_end = self.row_start + rowsInChunk

                self.row_start += self.row_limit
                self.row_end += self.row_limit
                yield csv_chunk, slice_start, slice_end

    # Return Line numbers of a file between [l_start .. l_end]
    def read_file_slice(self, file_obj, l_start, l_end):
        # return pointer to start of file
        file_obj.seek(0)
        input_reader = csv.reader(file_obj, delimiter=',')
        # create iterator for the file slice
        file_slice = islice(input_reader, l_start, l_end + 1)
        return [line for line in file_slice]

    # Function to append output as its processed in batches
    #
    # file_object: OutputFile
    # row_names: list of row names  -->   ['ROW_ID', 'ACCNTNUM', .. ]
    # row_data: dict mapping for XML Atribuites:
    # {'ROW_ID': '1', 'ACCNTNUM': '0.02',   ....  }
    # https://docs.python.org/2/library/csv.html
    def append_file_row(self, row_names, row_data):
        pd.DataFrame(columns=row_names, data={k: [v] for k, v in row_data.items()}).to_csv(
            self.fpath_output,
            mode='a',
            encoding='utf-8',
            header=False,
            index=False,
        )

    # WARNING: This will overwrite the file path
    def write_file_header(self, row_names):
        pd.DataFrame(columns=row_names).to_csv(
            self.fpath_output,
            encoding='utf-8',
            header=True,
            index=False,
        )

# --- Verbose Print Funcs ----------------------------------------------------#

    def print_dict(self, d):
        self.logger.debug(json.dumps(dict(d), indent=4))

    def print_xml(self, etree_obj):
        self.logger.debug('___________________________________________')
        self.logger.debug(etree.tostring(etree_obj, pretty_print=True))
