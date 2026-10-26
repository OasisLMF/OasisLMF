"""The loss step refuses a packed input set the configured engine cannot read.

Generation tries to avoid this by downgrading to ``items``, but it cannot always see the
problem: it infers the custom binary from analysis settings that ``generate-oasis-files`` need
not be given, and it never sees the random generator at all, which is a loss-step parameter.
Both failures are otherwise silent -- a third-party gulcalc understates every loss by the
building count, and generators 0/1 raise mid-run after generation has already succeeded.
"""
import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from oasislmf.computation.generate.losses import GenerateLossesDir
from oasislmf.pytools.common.data import coverage_buildings_dtype
from oasislmf.utils.exceptions import OasisException


def _files_dir(tmp, n_building):
    """An input dir whose coverage_buildings declares one coverage carrying n_building.

    Flat, as the loss step sees it: ``oasis_files_dir`` is already ``<run>/input`` (RunModel sets
    it so), not the run dir above it. A fixture nesting the file one level down once let a guard
    that read ``<run>/input/input`` pass every test here while never firing in a real run.
    """
    os.makedirs(tmp, exist_ok=True)
    if n_building is not None:
        np.array([(1, n_building)], dtype=coverage_buildings_dtype).tofile(
            os.path.join(tmp, 'coverage_buildings.bin'))
    return tmp


class _Step:
    """The three attributes the guard reads, without building a whole computation step."""

    def __init__(self, files_dir, generator=2, gulcalc=None, supports=False):
        self.oasis_files_dir = files_dir
        self.gul_random_generator = generator
        self.model_custom_gulcalc = gulcalc
        self.custom_gulcalc_supports_packing = supports

    def check(self, settings=None):
        return GenerateLossesDir._check_packed_inputs_can_be_run(self, settings or {})


class TestPackedInputsGuard(TestCase):

    def test_unpacked_inputs_are_never_refused(self):
        """Every engine can read these, so no configuration should be rejected."""
        with TemporaryDirectory() as tmp:
            for n in (None, 1):   # no file at all, and one building per coverage
                with self.subTest(n_building=n):
                    d = _files_dir(os.path.join(tmp, f'u{n}'), n)
                    _Step(d, generator=0, gulcalc='anything').check()

    def test_a_positive_count_above_one_is_still_packed(self):
        """Summed-at-source is not the same as unpacked, and the sign does not say otherwise.

        A positive count means the buildings are summed before the stream, but generation has
        still divided the TIV and the ground-up tool still draws per building -- it only adds
        them up afterwards. So the magnitude is what both engines key on: check_packing_supported
        tests ``abs(packed_buildings).max()``, and a third-party gulcalc reading the divided TIV
        understates by the count either way.
        """
        with TemporaryDirectory() as tmp:
            d = _files_dir(tmp, 3)
            with self.assertRaises(OasisException):
                _Step(d, generator=0).check()

    def test_packed_inputs_refuse_generator_0_and_1(self):
        """check_packing_supported would raise mid-run; this says so before anything runs."""
        with TemporaryDirectory() as tmp:
            d = _files_dir(tmp, -3)
            for gen in (0, 1):
                with self.subTest(generator=gen):
                    with self.assertRaises(OasisException) as e:
                        _Step(d, generator=gen).check()
                    self.assertIn('generator 2', str(e.exception))
            _Step(d, generator=2).check()   # generator 2 is fine

    @patch('oasislmf.computation.generate.losses.resolve_custom_gulcalc_cmd')
    def test_packed_inputs_refuse_an_undeclared_custom_gulcalc(self, resolve):
        """The case generation misses when it is run without analysis settings.

        generate-oasis-files with no -a infers ``None_None_gulcalc``, finds nothing and packs;
        the loss step then resolves the real binary. Without this guard every loss silently
        comes out a factor of NumberOfBuildings too small.
        """
        resolve.return_value = 'supplier_model_gulcalc'
        with TemporaryDirectory() as tmp:
            d = _files_dir(tmp, -4)
            with self.assertRaises(OasisException) as e:
                _Step(d).check({'model_supplier_id': 'supplier', 'model_name_id': 'model'})
            self.assertIn('supplier_model_gulcalc', str(e.exception))
            self.assertIn('custom_gulcalc_supports_packing', str(e.exception))

    @patch('oasislmf.computation.generate.losses.resolve_custom_gulcalc_cmd')
    def test_a_gulcalc_that_declares_support_is_allowed(self, resolve):
        resolve.return_value = 'supplier_model_gulcalc'
        with TemporaryDirectory() as tmp:
            _Step(_files_dir(tmp, -4), supports=True).check()

    @patch('oasislmf.computation.generate.losses.resolve_custom_gulcalc_cmd')
    def test_the_builtin_pipeline_is_allowed(self, resolve):
        resolve.return_value = None
        with TemporaryDirectory() as tmp:
            _Step(_files_dir(tmp, -4)).check()
