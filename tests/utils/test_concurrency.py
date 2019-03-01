# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import json

from unittest import TestCase

from hypothesis import (
    given,
    HealthCheck,
    reproduce_failure,
    settings,
)

from hypothesis.strategies import (
    dictionaries,
    integers,
    floats,
    just,
    lists,
    sampled_from,
    text,
    tuples,
)

from mock import patch, Mock

from tempfile import NamedTemporaryFile

from oasislmf.utils.concurrency import (
    multiprocess as mp,
    multithread as mt,
    SignalHandler as Sh,
    Task as Tk,
)
from oasislmf.utils.exceptions import OasisException


class SignalHandler(TestCase):
    pass

class Task(TestCase):
    pass

class Multiprocess(TestCase):
    pass

class Multithread(TestCase):
    pass
