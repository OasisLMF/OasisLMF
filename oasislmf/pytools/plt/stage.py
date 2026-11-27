"""Period loss table stage: set up once, then append the PLT rows of one block of events at a time.

``PltStage`` holds everything ``pltpy`` builds before its event loop (occurrences, period
weights, quantile intervals, the ``PLTReader`` and its row buffers) and writes the rows of the
events handed to it in memory into per-chunk output files. The ``pltpy`` command is built on it
(see ``manager.run``). Header validation, the per-chunk files and the event loop are shared with
``EltStage`` through ``oasislmf.pytools.elt.stage.TableStage``.

The occurrence index is set up once per stage: ``read_occurrence`` memory-maps the index saved by
``write_occurrence_index`` when there is one for the current occurrence file, and builds it in
memory otherwise.

The reader, its Numba kernels and ``read_input_files`` stay in ``manager``, where tests patch
them; this module looks them up through the ``manager`` module at call time.
"""
from oasislmf.pytools.elt.stage import TableStage
from oasislmf.pytools.plt import manager as plt_manager
from oasislmf.pytools.plt.data import MPLT_fmt, MPLT_headers, QPLT_fmt, QPLT_headers, SPLT_fmt, SPLT_headers


class PltStage(TableStage):
    """``pltpy`` set up once and fed summary blocks in memory, writing SPLT, MPLT and QPLT files per chunk.

    Paths resolve as in ``pltpy``: ``run_dir`` and the output paths are relative to the working
    directory.

    Args:
        run_dir (str | os.PathLike | None): run directory; occurrences, periods and quantiles are
            read from its ``input`` folder. None means './work'.
        input_header (bytes): the 12-byte summary stream header summarypy writes, or its first 8
            bytes. It is validated as pltpy validates its input and gives the sample count.
        splt (bool): produce the sample PLT.
        mplt (bool): produce the moment PLT.
        qplt (bool): produce the quantile PLT.
        output_format (str): 'bin', 'csv' or 'parquet'.
        noheader (bool): do not write the csv header line.
    """

    out_types = ("splt", "mplt", "qplt")
    headers = {"splt": SPLT_headers, "mplt": MPLT_headers, "qplt": QPLT_headers}
    fmts = {"splt": SPLT_fmt, "mplt": MPLT_fmt, "qplt": QPLT_fmt}

    def __init__(self, run_dir, input_header, splt=False, mplt=False, qplt=False, output_format='bin', noheader=False):
        super().__init__(run_dir, input_header, {"splt": splt, "mplt": mplt, "qplt": qplt}, output_format, noheader)
        self.file_data = plt_manager.read_input_files(self.run_dir, self.computes["qplt"], self.len_sample)
        self._init_reader()

    def _make_reader(self, n_files):
        return plt_manager.PLTReader(
            self.len_sample,
            self.computes["splt"],
            self.computes["mplt"],
            self.computes["qplt"],
            self.file_data["occ_csr"],
            self.file_data["period_weights"],
            self.file_data["granular_date"],
            self.file_data["intervals"],
            n_files=n_files,
        )
