from enum import Enum
import numpy as np
import numba as nb
import math
from pathlib import Path
import pyarrow as pa
import zlib

TARGET_ROW_GROUP_BYTES = 1_048_576
BUFFER_READ_BYTES = 1_048_576

OFPT_file_state = Enum('OFPT_file_state', [('init', 1), ('footer_read', 2)])


magic_file = b"OFPT"
magic_footer = b"OFPF"
magic_event = b"OFPE"

trailing_locator_dtype = np.dtype([
    ('file_total_size', '<i8'),
    ('footer_size', '<i8'),
    ('magic', 'S4')
])

file_footer_dtype = np.dtype([
    ('magic', 'S4'),
    ('version', '<i4'),
    ('total_events', '<i4'),
    ('max_intensity_bin_id', '<i4'),
    ('has_intensity_uncertainty', '<i4'),
    ('areaperil_width', '<i4'),
    ('compression_codec', '<i4'),
    ('event_footer_block_size', '<i4'),
    ('footer_block_crc32', '<u4'),
    ('slot_table_crc32', '<u4'),
    ('footer_size', '<i8'),
])

slot_table_len = 256
slot_table_dtype = np.dtype([
    ('event_footer_offset', '<i8'),
    ('event_footer_size', '<i8'),
])


event_footer_bloc_header_dtype_u4 = np.dtype([
    ('magic', 'S4'),
    ('event_id', '<i4'),
    ('total_areaperils', '<u4'),
    ('num_chunks', '<i4'),
])


event_footer_bloc_chunk_dtype_ap_u4 = np.dtype([
    ('offset', '<i8'),
    ('compressed_size', '<i4'),
    ('decompressed_size', '<i4'),
    ('min_areaperil_id', '<u4'),
    ('max_areaperil_id', '<u4'),
    ('num_intensity_types', '<i4'),
    ('crc32', '<u4'),
])


@nb.njit(cache=True)
def has_number_in_range(areaperil_ids, min_areaperil_id, max_areaperil_id):
    for apid in areaperil_ids:
        if min_areaperil_id <= apid <= max_areaperil_id:
            return True
    return False


@nb.jit(nopython=True, cache=True)
def reverse_mv_read(byte_mv, cursor, _dtype, itemsize):
    """
    read a certain dtype from numpy byte view starting at cursor, return the value and the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object end
        _dtype: data type of the object
        itemsize: size of the data type

    Returns:
        (object value, end of object index)
    """
    return byte_mv[cursor-itemsize:cursor].view(_dtype)[0], cursor - itemsize

def complex_reverse_mv_read(byte_mv, cursor, _dtype, _rows=1):
    if _rows == 1:
        return np.frombuffer(byte_mv[cursor-_dtype.itemsize:cursor], dtype=_dtype)[0], cursor - _dtype.itemsize
    else:
        return np.frombuffer(byte_mv[cursor - _dtype.itemsize * _rows:cursor], dtype=_dtype), cursor - _dtype.itemsize * _rows


class Reader:
    def read_bytes_all(self):
        raise NotImplementedError

    def read_bytes_range(self, begin, end):
        raise NotImplementedError


class NpMemMap(Reader):
    def __init__(self, fileobj, dtype='b', mode='r', **kwargs):
        if dtype != 'b':
            raise ValueError(f"{Reader} only works with 'b' dtype")
        self.mmap = np.memmap(fileobj, dtype=dtype, mode=mode, **kwargs)


    def read_bytes_all(self):
        return self.mmap

    def read_bytes_range(self, begin, end):
        return self.mmap[begin:end]

    def close(self):
        self.mmap.flush()


class BinScanner:
    footprint_filename = 'footprint.bin'
    footprint_index_filename = 'footprint.idx'

    footprint_bin_header = np.dtype([
        ('num_intensity_bins', np.int32),
        ('has_intensity_uncertainty', np.int32)
    ])
    EventIndexBin_dtype = np.dtype([
        ('event_id', np.int32),
        ('offset', np.int64),
        ('size', np.int64)
    ])

    Event_dtype = np.dtype([
        ('areaperil_id', '<u4'),
        ('intensity_bin_id', '<i4'),
        ('probability', '<f4')
    ])

    def __init__(self, bin_root_dir, reader:Reader):
        import os
        self.footprint_path = os.path.join(bin_root_dir, self.footprint_filename)
        self.footprint_index_path = os.path.join(bin_root_dir, self.footprint_index_filename)

        # I need an interface for the reader but not clear yet
        self.footprint_reader = reader(self.footprint_path, mode='r')
        self.footprint_index_reader = reader(self.footprint_index_path, mode='r')

        self.footprint_index = np.frombuffer(self.footprint_index_reader.read_bytes_all(), dtype=self.EventIndexBin_dtype)

    def sorted_iter(self):
        sorted_event_id_indices = np.argsort(self.footprint_index['event_id'])
        for event_i in sorted_event_id_indices:
            event_info = self.footprint_index[event_i]
            yield event_info, np.frombuffer(self.footprint_reader.read_bytes_range(event_info['offset'], event_info['offset'] + event_info['size']), dtype=self.Event_dtype)

class OFPTScanner:
    def __init__(self, root_dir, reader_interface):
        self.root_dir = root_dir
        self.event_path = None
        self.reader = None
        self.footer_buffer = None
        self.buffer_offset = 0
        self.buffer = None
        self.reader_interface = reader_interface

    def initialize(self):
        self.reader = self.reader_interface(self.event_path)
        self.buffer = self.footer_buffer = self.reader.read_bytes_range(-BUFFER_READ_BYTES, None)
        backward_cursor = len(self.footer_buffer)

        self.trailing_locator, backward_cursor = complex_reverse_mv_read(self.footer_buffer, backward_cursor, trailing_locator_dtype)
        assert self.trailing_locator['magic'] == magic_file

        self.buffer_offset = self.trailing_locator['file_total_size']-len(self.footer_buffer)

        self.file_footer, backward_cursor = complex_reverse_mv_read(self.footer_buffer, backward_cursor, file_footer_dtype)
        assert self.file_footer['magic'] == magic_footer
        assert self.file_footer['footer_size'] == self.trailing_locator['footer_size']

        self.slot_table, backward_cursor = complex_reverse_mv_read(self.footer_buffer, backward_cursor, slot_table_dtype, slot_table_len)
        assert (zlib.crc32(self.slot_table.tobytes()) & 0xFFFFFFFF) == self.file_footer['slot_table_crc32']

        if backward_cursor < self.file_footer["event_footer_block_size"]:
            raise NotImplementedError("read more")

        self.event_footer_bloc = self.footer_buffer[backward_cursor - self.file_footer["event_footer_block_size"]: backward_cursor]
        assert (zlib.crc32(self.event_footer_bloc.tobytes()) & 0xFFFFFFFF) == self.file_footer['footer_block_crc32']


    def get_event_info(self, event_id):
        hex_event_id = f"{event_id:08X}"
        event_path = Path(self.root_dir, "footprint", hex_event_id[:2], hex_event_id[2:4], f"{hex_event_id[4:6]}.ofpt")
        if self.event_path != event_path:
            self.event_path = event_path
            self.initialize()
        slot_i = event_id & 0xFF

        if self.slot_table[slot_i]['event_footer_size']==0:
            return None, None
        event_start = self.slot_table[slot_i]['event_footer_offset']
        event_end = event_start + self.slot_table[slot_i]['event_footer_size']
        event_footer = self.event_footer_bloc[event_start: event_end]

        event_footer_bloc_header = np.frombuffer(event_footer[:event_footer_bloc_header_dtype_u4.itemsize], dtype=event_footer_bloc_header_dtype_u4)[0]
        assert event_footer_bloc_header['magic'] == magic_event
        assert event_footer_bloc_header['event_id'] == event_id

        event_footer_bloc_chunks = np.frombuffer(
            event_footer[event_footer_bloc_header_dtype_u4.itemsize:
                         event_footer_bloc_header_dtype_u4.itemsize + event_footer_bloc_chunk_dtype_ap_u4.itemsize*event_footer_bloc_header['num_chunks']],
            dtype=event_footer_bloc_chunk_dtype_ap_u4)

        return event_footer_bloc_header, event_footer_bloc_chunks


    def get_event_chunk(self, event_footer_bloc_chunk):
        chunk_start = event_footer_bloc_chunk['offset']
        chunk_end = chunk_start + event_footer_bloc_chunk['compressed_size']

        if chunk_start < self.buffer_offset or chunk_end > self.buffer_offset + len(self.buffer):
            range_end = chunk_start + max(BUFFER_READ_BYTES, event_footer_bloc_chunk['compressed_size'])
            self.buffer = self.reader.read_bytes_range(chunk_start, range_end)
            self.buffer_offset = chunk_start

        compressed_chunk = self.buffer[chunk_start - self.buffer_offset: chunk_start - self.buffer_offset + event_footer_bloc_chunk['compressed_size']]
        uncompressed_chunk = pa.decompress(compressed_chunk, event_footer_bloc_chunk['decompressed_size'], codec='zstd')
        return uncompressed_chunk


    def get_event(self, event_id, areaperil_ids):
        event_footer_bloc_header, event_footer_bloc_chunks = self.get_event_info(event_id)
        if event_footer_bloc_header is None:
            return None
        for event_footer_bloc_chunk in event_footer_bloc_chunks:
            if has_number_in_range(areaperil_ids, event_footer_bloc_chunk["min_areaperil_id"], event_footer_bloc_chunk["max_areaperil_id"]):
                areaperil_ids, cum_offsets, probabilities, intensity_bin_ids = decode_chunk(self.get_event_chunk(event_footer_bloc_chunk), areaperil_width=self.file_footer['areaperil_width'])
                return areaperil_ids, cum_offsets, probabilities, intensity_bin_ids


def decode_chunk(raw, areaperil_width=4):
    ap_dtype = f'<u{areaperil_width}'
    ap_size = areaperil_width
    cursor = 0

    num_areaperils = np.frombuffer(raw[cursor:cursor + ap_size], dtype=ap_dtype)[0]
    cursor += ap_size

    areaperil_ids = np.frombuffer(raw[cursor:cursor + num_areaperils * ap_size], dtype=ap_dtype)
    cursor += num_areaperils * ap_size

    num_intensity_types = np.frombuffer(raw[cursor:cursor + 4], dtype=np.int32)[0]
    cursor += 4

    cum_offsets = np.frombuffer(raw[cursor:cursor + (num_areaperils + 1) * 4], dtype=np.int32)
    cursor += (num_areaperils + 1) * 4

    num_probabilities = cum_offsets[-1]
    probabilities = np.frombuffer(raw[cursor:cursor + num_probabilities * 4], dtype=np.float32)
    cursor += num_probabilities * 4

    intensity_bin_ids = np.frombuffer(raw[cursor:], dtype=np.int32)
    assert len(intensity_bin_ids) == num_intensity_types * num_probabilities

    return areaperil_ids, cum_offsets, probabilities, intensity_bin_ids

def event_to_columnar(event_data):
    # Sort by areaperil_id then intensity_bin_id
    sorted_idx = np.lexsort((event_data['intensity_bin_id'], event_data['areaperil_id']))
    sorted_data = event_data[sorted_idx]

    # Unique sorted areaperils
    areaperil_ids, counts = np.unique(sorted_data['areaperil_id'], return_counts=True)
    num_areaperils = len(areaperil_ids)

    # Cumulative offsets (CSR-style row pointers)
    cum_offsets = np.zeros(num_areaperils + 1, dtype=np.int32)
    np.cumsum(counts, out=cum_offsets[1:])

    # Unique intensity bin ids
    intensity_bin_id_name = [name for name in event_data.dtype.names if name.startswith('intensity_bin_id')]
    num_intensity_types = len(intensity_bin_id_name)
    intensity_bin_ids = np.empty((len(event_data), num_intensity_types), dtype=np.int32)
    for i, name in enumerate(intensity_bin_id_name):
        intensity_bin_ids[:, i] = sorted_data[name]

    return areaperil_ids, cum_offsets, sorted_data['probability'], intensity_bin_ids


class OFPTFileWriter:
    def __init__(self, fileobj, writer_interface):
        self.fileobj = fileobj
        self.writer_interface = writer_interface


    def initialize(self):
        self.trailing_locator = np.zeros(1, trailing_locator_dtype)[0]
        self.trailing_locator['magic'] = magic_file
        # self.trailing_locator['footer_size'] = trailing_locator_dtype.itemsize + file_footer_dtype.itemsize
        # self.trailing_locator['file_total_size'] = 0

        self.file_footer = np.zeros(1, file_footer_dtype)[0]
        self.file_footer['magic'] = magic_footer
        self.file_footer['version'] = 1
        self.file_footer['total_events'] = 0
        self.file_footer['max_intensity_bin_id'] = 0
        self.file_footer['has_intensity_uncertainty'] = 0
        self.file_footer['areaperil_width'] = 4
        self.file_footer['compression_codec'] = 1 # zstd
        # self.file_footer['event_footer_block_size'] = 0
        # self.file_footer['footer_block_crc32'] = 0
        # self.file_footer['slot_table_crc32'] = 0
        # self.file_footer['footer_size'] = 0

        self.slot_table = np.zeros(slot_table_len, slot_table_dtype)
        self.event_footer_blocs = []

        self._writer = self.writer_interface(self.fileobj, mode='wb')
        self.offset = 0

        self.write(magic_file)

    def close(self):
        self._writer.close()

    def write(self, bytes):
        self._writer.write(bytes)
        self.offset += len(bytes)

    def write_sorted_event(self, event_info, areaperil_ids, cum_offsets, probabilities, intensity_bin_ids):
        # for now we will only do 1 chunk
        rough_size = areaperil_ids.nbytes + cum_offsets.nbytes + intensity_bin_ids.nbytes + probabilities.nbytes
        # num_chunks = math.ceil(rough_size / TARGET_ROW_GROUP_BYTES)
        num_chunks = 1
        num_areaperil_ids_per_chunk = math.ceil(len(areaperil_ids) / num_chunks)
        num_intensity_types = np.int32(len(intensity_bin_ids)//len(probabilities))
        intensity_bin_ids = intensity_bin_ids.reshape((len(probabilities), num_intensity_types))

        # slot_i = event_info['event_id'] & 0xFF
        # sorted_areaperil_id_indices = np.argsort(event_data['areaperil_id'])

        event_footer_bloc_header = np.zeros(1, dtype=event_footer_bloc_header_dtype_u4)[0]
        event_footer_bloc_header['magic'] = magic_event
        event_footer_bloc_header['event_id'] = event_info['event_id']
        event_footer_bloc_header['total_areaperils'] = len(areaperil_ids)
        event_footer_bloc_header['num_chunks'] = num_chunks
        event_footer_blocs_chunk = np.zeros(num_chunks, dtype=event_footer_bloc_chunk_dtype_ap_u4)
        self.event_footer_blocs.append((event_footer_bloc_header, event_footer_blocs_chunk))

        self.file_footer['has_intensity_uncertainty'] = (self.file_footer['has_intensity_uncertainty']
                                                         or (event_footer_bloc_header['total_areaperils'] != len(probabilities)))
        self.file_footer['max_intensity_bin_id'] = max(self.file_footer['max_intensity_bin_id'], np.max(intensity_bin_ids))

        areaperil_index_start = 0
        for i in range(event_footer_bloc_header['num_chunks']):
            areaperil_index_end = areaperil_index_start + num_areaperil_ids_per_chunk
            areaperil_ids_chunk = areaperil_ids[areaperil_index_start:areaperil_index_end]
            cum_offsets_chunk = cum_offsets[areaperil_index_start:areaperil_index_end + 1]
            probabilities_chunk = probabilities[cum_offsets_chunk[0]: cum_offsets_chunk[-1]]
            intensity_bin_ids_chunk = intensity_bin_ids[cum_offsets_chunk[0]: cum_offsets_chunk[-1]]

            # decompressed_size = (
            #         areaperil_ids.itemsize
            #         + areaperil_ids[areaperil_index_start:areaperil_index_end].nbytes
            #         + np.int32.itemsize # num_intensity_types
            #         + cum_offsets[areaperil_index_start:areaperil_index_end].nbytes
            #         + probabilities[cum_offsets[areaperil_index_start]: cum_offsets[areaperil_index_end]].nbytes
            #         + intensity_bin_ids[cum_offsets[areaperil_index_start]: cum_offsets[areaperil_index_end]].nbytes
            # )

            raw = b''.join([
                np.int32(len(areaperil_ids_chunk)).tobytes(),
                areaperil_ids_chunk.tobytes(),
                num_intensity_types.tobytes(),
                cum_offsets_chunk.tobytes(),
                probabilities_chunk.tobytes(),
                intensity_bin_ids_chunk.tobytes(),
            ])
            decompressed_size = len(raw)
            compressed = pa.compress(raw, codec='zstd')

            event_footer_bloc_chunk = event_footer_blocs_chunk[i]
            event_footer_bloc_chunk['offset'] = self.offset
            event_footer_bloc_chunk['compressed_size'] = len(compressed)
            event_footer_bloc_chunk['decompressed_size'] = decompressed_size
            event_footer_bloc_chunk['min_areaperil_id'] = areaperil_ids_chunk[areaperil_index_start]
            event_footer_bloc_chunk['max_areaperil_id'] = areaperil_ids_chunk[-1]
            event_footer_bloc_chunk['num_intensity_types'] = num_intensity_types
            event_footer_bloc_chunk['crc32'] = zlib.crc32(compressed) & 0xFFFFFFFF

            self.write(compressed)

    def do_events_end(self):
        event_footer_offset = 0
        raw_event_footer = b""
        for event_info, event_chunks in self.event_footer_blocs:
            slot_i = event_info['event_id'] & 0xFF
            raw = event_info.tobytes() + event_chunks.tobytes()
            self.slot_table[slot_i]["event_footer_offset"] = event_footer_offset
            self.slot_table[slot_i]["event_footer_size"] = len(raw)
            raw_event_footer += raw
            event_footer_offset += len(raw)

        self.file_footer['event_footer_block_size'] = len(raw_event_footer)
        self.file_footer['footer_block_crc32'] = zlib.crc32(raw_event_footer) & 0xFFFFFFFF
        self.write(raw_event_footer)

        raw_slot_table = self.slot_table.tobytes()
        self.file_footer['slot_table_crc32'] = zlib.crc32(raw_slot_table) & 0xFFFFFFFF
        self.write(raw_slot_table)

        footer_size = len(raw_event_footer) + len(raw_slot_table) + self.file_footer.nbytes
        self.trailing_locator['footer_size'] = self.file_footer['footer_size'] = footer_size
        self.write(self.file_footer.tobytes())

        self.trailing_locator['file_total_size'] = self.offset + self.trailing_locator.nbytes
        self.write(self.trailing_locator.tobytes())


def bin_to_OFPF(bin_root_dir, OFPF_root_dir):
    bin_scanner = BinScanner(bin_root_dir, NpMemMap)
    OFPT_file_writer = None
    cur_file_path = None
    for event_info, event_data in bin_scanner.sorted_iter():
        hex_event_id = f"{event_info['event_id']:08X}"
        dirpath = Path(OFPF_root_dir, "footprint", hex_event_id[:2], hex_event_id[2:4])
        dirpath.mkdir(parents=True, exist_ok=True)
        file_path = Path(dirpath, f"{hex_event_id[4:6]}.ofpt")
        if cur_file_path != file_path:
            if OFPT_file_writer is not None:
                OFPT_file_writer.do_events_end()
                OFPT_file_writer.close()
            OFPT_file_writer = OFPTFileWriter(file_path, open)
            OFPT_file_writer.initialize()
            cur_file_path = file_path

        OFPT_file_writer.write_sorted_event(event_info, *event_to_columnar(event_data))
    OFPT_file_writer.do_events_end()


if __name__ == "__main__":
    # bs = BinScanner("/home/sstruzik/OasisPiWind/model_data/PiWind/", NpMemMap)
    # for event_info, event_data in bs.sorter_iter():
    #     print(event_info)
    #     print(event_data)
    #     break
    import shutil

    root_dir = "/home/sstruzik/OasisPiWind/model_data/PiWind/"
    OFPT_DIR = Path(root_dir, "footprint_OFPT")
    shutil.rmtree(OFPT_DIR, ignore_errors=True)
    bin_to_OFPF(root_dir, OFPT_DIR)

    OFPT_scanner = OFPTScanner(OFPT_DIR, NpMemMap)
    event_header, event_chunks = OFPT_scanner.get_event_info(1)
    print(event_header)
    print(event_chunks)