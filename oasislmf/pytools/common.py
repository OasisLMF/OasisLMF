"""
This file defines quantities reused across the pytools stack.

"""
import os
import selectors

import pandas as pd
import numba as nb
import numpy as np

# streams
PIPE_CAPACITY = 65536  # bytes

# data types

oasis_int = np.dtype(os.environ.get('OASIS_FLOAT', 'i4'))
nb_oasis_int = nb.from_dtype(oasis_int)
oasis_int_size = oasis_int.itemsize

oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))
nb_oasis_float = nb.from_dtype(oasis_float)
oasis_float_size = oasis_float.itemsize

areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
areaperil_int_size = areaperil_int.itemsize

#
CDF_STREAM_ID = 0
GUL_STREAM_ID = 1
FM_STREAM_ID = 2
SUMMARY_STREAM_ID = 3

ITEM_STREAM = 1
COVERAGE_STREAM = 2

null_index = oasis_int.type(-1)

def bytes_to_stream_info(stream_header):
    return np.frombuffer(stream_header[:3] + b'\x00', '<i4')[0], np.frombuffer(stream_header[3:], 'i1')[0]
def stream_info_to_bytes(stream_type, stream_agg_type):
    return np.int32(stream_type).tobytes()[:3] + np.int8(stream_agg_type).tobytes()

def read_stream_info(stream_obj):
    stream_type, stream_agg_type = bytes_to_stream_info(stream_obj.read(4))
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]
    return stream_type, stream_agg_type, len_sample


def load_as_ndarray(dir_path, name, _dtype, must_exist=True, col_map=None):
    """
    load a file as a numpy ndarray
    useful for multi-columns files
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: np.dtype
        must_exist: raise FileNotFoundError if no file is present
        col_map: name re-mapping to change name of csv columns
    Returns:
        numpy ndarray
    """

    if os.path.isfile(os.path.join(dir_path, name + '.bin')):
        return np.fromfile(os.path.join(dir_path, name + '.bin'), dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        # in csv column cam be out of order and have different name,
        # we load with pandas and write each column to the ndarray
        if col_map is None:
            col_map = {}
        with open(os.path.join(dir_path, name + '.csv')) as file_in:
            cvs_dtype = {col_map.get(key, key): col_dtype for key, (col_dtype, _) in _dtype.fields.items()}
            df = pd.read_csv(file_in, delimiter=',', dtype=cvs_dtype, usecols=list(cvs_dtype.keys()))
            res = np.empty(df.shape[0], dtype=_dtype)
            for name in _dtype.names:
                res[name] = df[col_map.get(name, name)]
            return res
    else:
        return np.empty(0, dtype=_dtype)

def load_as_array(dir_path, name, _dtype, must_exist=True):
    """
    load file as a single numpy array,
     useful for files with a binary version with only one type of value where their index correspond to an id.
     For example coverage.bin only contains tiv value for each coverage id
     coverage_id n correspond to index n-1
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: numpy dtype of the required array
        must_exist: raise FileNotFoundError if no file is present
    Returns:
        numpy array of dtype type
    """
    fp = os.path.join(dir_path, name + '.bin')
    if os.path.isfile(fp):
        return np.fromfile(fp, dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        fp = os.path.join(dir_path, name + '.csv')
        with open(fp) as file_in:
            return np.loadtxt(file_in, dtype=_dtype, delimiter=',', skiprows=1, usecols=1)
    else:
        return np.empty(0, dtype=_dtype)


class LossReader:

    @staticmethod
    def register_streams_in(selector_class, streams_in):
        """
        Data from input process is generally sent by event block, meaning once a stream receive data, the complete event is
        going to be sent in a short amount of time.
        Therefore, we can focus on each stream one by one using their specific selector 'stream_selector'.

        """
        main_selector = selector_class()
        stream_data = []
        for stream_in in streams_in:
            mv = memoryview(bytearray(PIPE_CAPACITY))
            byte_mv = np.frombuffer(buffer=mv, dtype='b')

            stream_selector = selector_class()
            stream_selector.register(stream_in, selectors.EVENT_READ)
            data = {'mv': mv,
                    'byte_mv': byte_mv,
                    'cursor': 0,
                    'valid_buff': 0,
                    'stream_selector': stream_selector
                    }
            stream_data.append(data)
            main_selector.register(stream_in, selectors.EVENT_READ, data)
        return main_selector, stream_data

    def read_streams(self, streams_in):
        try:
            main_selector, stream_data = self.register_streams_in(selectors.DefaultSelector, streams_in)
            self.logger.debug("Streams read with DefaultSelector")
        except PermissionError:  # Fall back option if stream_in contain regular files
            main_selector, stream_data = self.register_streams_in(selectors.SelectSelector, streams_in)
            self.logger.debug("Streams read with SelectSelector")
        try:
            while main_selector.get_map():
                for sKey, _ in main_selector.select():
                    event = self.read_event(sKey.fileobj, main_selector, **sKey.data)

                    if event:
                        event_id, cursor, valid_buff = event
                        sKey.data['cursor'] = cursor
                        sKey.data['valid_buff'] = valid_buff
                        self.event_read_log()
                        yield event_id

            # Stream is read, we need to check if there is remaining event to be parsed
            for data in stream_data:
                if data['cursor'] < data['valid_buff']:
                    byte_mv = data['byte_mv']
                    cursor = data['cursor']
                    valid_buff = data['valid_buff']
                    yield_event = True
                    while yield_event:
                        cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, 0, 0)

                        if event_id:
                            self.item_exit()
                            self.event_read_log()
                            yield event_id

        finally:
            main_selector.close()

    def read_event(self, stream_in, main_selector, stream_selector, mv, byte_mv, cursor, valid_buff):
        event_id = 0
        item_id = 0

        while True:
            if valid_buff < PIPE_CAPACITY:
                len_read = stream_in.readinto1(mv[valid_buff:])
                valid_buff += len_read

                if len_read == 0:
                    stream_selector.close()
                    main_selector.unregister(stream_in)
                    if event_id:
                        self.item_exit()
                        return event_id, cursor, valid_buff

                    break

            cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, event_id, item_id)

            if yield_event:
                if cursor == valid_buff:
                    valid_buff = 0
                return event_id, cursor, valid_buff
            else:
                mv[:valid_buff - cursor] = mv[cursor: valid_buff]
                valid_buff -= cursor
                cursor = 0
                stream_selector.select()

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        raise NotImplementedError

    def item_exit(self):
        raise NotImplementedError