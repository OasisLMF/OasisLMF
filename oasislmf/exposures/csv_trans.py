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
from oasislmf.utils.exceptions import OasisException


class Translator(object):
    def __init__(self, input_path, output_path, xslt_path, xsd_path=None, append_row_nums=False, chunk_size=5000, logger=None):
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

        self.logger = logger or logging.getLogger()
        self.xsd = (etree.parse(xsd_path) if xsd_path else None)
        self.xslt = etree.parse(xslt_path)
        self.fpath_input = input_path
        self.fpath_output = output_path

        self.row_nums = append_row_nums
        self.row_limit = chunk_size
        self.row_header_in = None
        self.row_header_out = None

    def __call__(self):
        self.test_run()
        csv_reader = pd.read_csv(self.fpath_input, iterator=True, dtype=object, encoding='utf-8')

        task_list = []
        for chunk_id, (data, first_row, last_row) in enumerate(self.next_file_slice(csv_reader)):
            task_list.append(Task(self.process_chunk, args=(data,first_row,last_row, chunk_id), key=chunk_id))

        results = {}
        num_ps = multiprocessing.cpu_count()
        for key, data in multithread(task_list, pool_size=num_ps):
            results[key] = data

        ## write output to disk
        for i in range(0, len(results)):
            if (i == 0):
                self.write_file_header(results[i].columns.tolist())
            results[i].to_csv(
                self.fpath_output,
                mode='a',
                encoding='utf-8',
                header=False,
                index=False,
            )

    def test_run(self, row_sample_size=5):
        """
            Test transformation run using the first 5 rows of input,
            Guard for invalid input files before starting multiprocessing
        """
        sample_reader = pd.read_csv(self.fpath_input, 
                                    iterator=True, 
                                    dtype=object, 
                                    encoding='utf-8', 
                                    nrows=row_sample_size)

        df_generator = self.next_file_slice(sample_reader)
        (data, first_row, last_row) = next(df_generator)
        sample_out_df = self.process_chunk(data, first_row, last_row, 1) 
        if sample_out_df.empty:
            raise OasisException('Input Test Failed: Output DataFrame is empty')

    def process_chunk(self, data, first_row_number, last_row_number, seq_id):
        xml_input_slice = self.csv_to_xml(
            self.row_header_in,
            data
        )
        self.print_xml(xml_input_slice)

        # Transform
        xml_output = self.xml_transform(xml_input_slice, self.xslt)

        # Validate Output
        if self.xsd:
            self.logger.debug(self.xml_validate(xml_output, self.xsd))

        # Convert transform XML back to CSV
        return self.xml_to_csv(
                 xml_output,        # XML etree
                 first_row_number,  # First Row in this slice
                 last_row_number    # Last Row in this slice
        )


    def csv_to_xml(self, csv_header, csv_data):
        root = etree.Element('root')
        for row in csv_data:
            rec = etree.SubElement(root, 'rec')
            for i in range(0, len(row)):
                if(row[i] not in [None, "", 'NaN']):
                    rec.set(csv_header[i], row[i])
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

    def xml_transform(self, xml_doc, xslt):
        lxml_transform = etree.XSLT(xslt)
        return lxml_transform(xml_doc)

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
