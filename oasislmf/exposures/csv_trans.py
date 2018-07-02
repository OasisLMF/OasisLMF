# -*- coding: utf-8 -*-

__all__ = [
    'Translator'
]

import json
import logging
import multiprocessing
import os

import pandas as pd

from lxml import etree

from oasislmf.utils.concurrency import (
    multiprocess,
    multithread,
    Task,
)

class Translator(object):
    """
    Transforms exposures/locations in CSV format
    by converting a source file to XML and applying an XSLT transform
    to apply rules for selecting, merging or updating columns

    An optional step is to passing an XSD file for output validation

    :param input_path: Source exposures file path, which should be in CSV comma delimited format 
    :type input_path: str

    :param output_path: File to write transform results
    :type output_path: str

    :param xslt_path: Source exposures Transformation rules file
    :type xslt_path: str

    :param xsd_path: Source exposures output validation file
    :type xsd_path: str

    :param append_row_nums: Append line numbers to first column of output called `ROW_ID` [1 .. n] when n is the number of rows processed. 
    :type append_row_nums: boolean

    :param chunk_size: Number of rows to process per multiprocess Task
    :type chunk_size: int
    """

    def __init__(self, input_path, output_path, xslt_path, xsd_path=None, append_row_nums=False, chunk_size=5000, logger=None):
        self.logger = logger or logging.getLogger()
        self.fpath_input = input_path
        self.fpath_output = output_path
        self.xslt = etree.parse(xslt_path)
        self.xsd = (etree.parse(xsd_path) if xsd_path else None)

        self.row_nums = append_row_nums
        self.row_limit = chunk_size
        self.row_header_in = None
        self.row_header_out = None

    def __call__(self):
        csv_reader = pd.read_csv(self.fpath_input, iterator=True, dtype=object, encoding='utf-8')

        task_list = []
        for data, first_row, last_row in self.next_file_slice(csv_reader):
            task_list.append(Task(self.process_chunk, (data,first_row,last_row)))
        # run is process file chunk
        for task in multithread(task_list):
            pass

        # write output
        for i in range(0, len(task_list)):
            if (i == 0):
                self.write_file_header(task_list[i].result.columns.tolist())

            task_list[i].result.to_csv(
                self.fpath_output,
                mode='a',
                encoding='utf-8',
                header=False,
                index=False,
            )


    def process_chunk(self, data, first_row_number, last_row_number): 
        print('Run chunk (%s,%s)' % (first_row_number,last_row_number))
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
        if self.xsd:
            self.logger.debug(self.xml_validate(xml_output, self.xsd))

        # Convert transform XML back to CSV
        print('lines %s to %s' % (first_row_number, last_row_number))
        return self.xml_to_csv(
            xml_output,        # XML etree
            first_row_number,  # First Row in this slice
            last_row_number    # Last Row in this slice
        )

# --- Transform Functions ----------------------------------------------------#

    def csv_to_xml(self, csv_header, csv_data):
        """
        Coverts a list of lists structure into an lxml XML object
        [
            [row 0 ,,,],
            [row 1 ,,,],
            [row n ,,,]
        ]

        Create root of new XML file outside of loop
        For each row in the file chunk to process 
            -> Create new 'rec' or record sub element
            -> Iterate over each column and append as an attribut if its valid

        https://pymotw.com/2/xml/etree/ElementTree/create.html
        http://lxml.de/api/lxml.etree._Element-class.html
        """
        root = etree.Element('root')
        for row in csv_data:
            rec = etree.SubElement(root, 'rec')
            for i in range(0, len(row)):
                if(row[i] not in [None, "", 'NaN']):
                    rec.set(self.row_header_in[i], row[i])
        return root

    def xml_to_csv(self, xml_elementTree, row_first, row_last):
        root = xml_elementTree.getroot()
        # Set output col headers
        if not (self.row_header_out):
            self.row_header_out = root[0].keys()

        # create Dataframe from xml and append each row
        rows = []
        for rec in root:
            rows.append(dict(rec.attrib))

        df_out = pd.DataFrame(rows, columns=self.row_header_out)

        # Add column for row_nums if set
        if self.row_nums:
            start = row_first + 1
            end = start + len(df_out)
            df_out.insert(0, 'ROW_ID', pd.Series(range(start,end)))

        return df_out


    # http://lxml.de/2.0/validation.html
    def xml_validate(self, xml_etree, xsd_etree):
        xmlSchema = etree.XMLSchema(xsd_etree)
        self.print_xml(xml_etree)
        self.print_xml(xsd_etree)
        if (xmlSchema.validate(xml_etree)):
            return True
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.error('Input failed to Validate')
                log = xmlSchema.error_log
                self.logger.error(log.last_error)
            return False

    # http://lxml.de/xpathxslt.html#xslt
    def xml_transform(self, xml_doc, xslt):
        lxml_transform = etree.XSLT(self.xslt)
        return lxml_transform(xml_doc)

    # Generator Function which processes and returns batches of the input CSV
    # http://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    #
    # file_reader = pandas.read_csv( ... ,iterator=True)
    def next_file_slice(self, file_reader):
        while True:
            try:
                df_slice = file_reader.get_chunk(self.row_limit)
                if(not self.row_header_in):
                    self.row_header_in = df_slice.columns.values.tolist()
                yield (
                    df_slice.fillna("").values.astype("unicode").tolist(),
                    df_slice.first_valid_index(),
                    df_slice.last_valid_index()
                )
            except StopIteration:
                self.logger.debug('End of input file')
                break

    def write_file_header(self, row_names):
        pd.DataFrame(columns=row_names).to_csv(
            self.fpath_output,
            encoding='utf-8',
            header=True,
            index=False,
        )

    def print_xml(self, etree_obj):
        self.logger.debug('___________________________________________')
        self.logger.debug(etree.tostring(etree_obj, pretty_print=True))
