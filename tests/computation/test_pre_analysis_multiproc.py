from queue import Empty, Full

import pandas as pd

from oasislmf.computation.hooks.pre_analysis_multiproc import (
    exposure_producer,
    pre_analysis_multiproc_worker,
    result_producer,
    with_error_queue,
)


class FakeQueue:
    """A queue.Queue-like stub with no real blocking, so unit tests run synchronously."""

    def __init__(self, items=None, full_times=0, empty_times=0):
        self.items = list(items) if items is not None else []
        self.put_items = []
        self.full_times = full_times
        self.empty_times = empty_times

    def get(self, timeout=None):
        if self.empty_times > 0:
            self.empty_times -= 1
            raise Empty
        if not self.items:
            raise Empty
        return self.items.pop(0)

    def put(self, item, timeout=None):
        if self.full_times > 0:
            self.full_times -= 1
            raise Full
        self.put_items.append(item)

    def empty(self):
        return not self.items


class FakeErrorQueue:
    def __init__(self, has_error=False):
        self.put_items = []
        self._has_error = has_error

    def empty(self):
        return not self._has_error

    def put(self, item):
        self.put_items.append(item)
        self._has_error = True

    def get(self, timeout=None):
        return self.put_items.pop(0)


class TogglingErrorQueue:
    """Reports empty() as True for the first `true_calls` calls, then False."""

    def __init__(self, true_calls):
        self.calls = 0
        self.true_calls = true_calls

    def empty(self):
        self.calls += 1
        return self.calls <= self.true_calls


class DummyLocation:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class DummyAccount:
    def __init__(self, dataframe):
        self.dataframe = dataframe


class DummyExposureData:
    def __init__(self, location_df, account_df=None):
        self.location = DummyLocation(location_df)
        self.account = DummyAccount(account_df) if account_df is not None else None


class DummyHook:
    def __init__(self, exposure_data, multiplier=1, **kwargs):
        self.exposure_data = exposure_data
        self.multiplier = multiplier

    def run(self):
        self.exposure_data.location.dataframe['BuildingTIV'] *= self.multiplier
        if self.exposure_data.account is not None:
            self.exposure_data.account.dataframe['LayerLimit'] *= self.multiplier
        return 'ok'


def _make_loc_acc_df():
    loc_df = pd.DataFrame({
        'PortNumber': [1, 1, 1, 1],
        'AccNumber': ['A1', 'A1', 'A2', 'A2'],
        'loc_id': [1, 2, 3, 4],
        'BuildingTIV': [1, 2, 3, 4],
    })
    acc_df = pd.DataFrame({
        'PortNumber': [1, 1],
        'AccNumber': ['A1', 'A2'],
        'LayerLimit': [10, 20],
    })
    return loc_df, acc_df


def test_with_error_queue_passes_through_return_value():
    @with_error_queue
    def fct(error_queue, x, y):
        return x + y

    error_queue = FakeErrorQueue()
    assert fct(error_queue, 2, 3) == 5
    assert error_queue.empty()


def test_with_error_queue_captures_exception():
    @with_error_queue
    def fct(error_queue):
        raise ValueError('boom')

    error_queue = FakeErrorQueue()
    result = fct(error_queue)

    assert result is None
    assert not error_queue.empty()
    exc_type, exc_value, _ = error_queue.get()
    assert exc_type is ValueError
    assert str(exc_value) == 'boom'


def test_exposure_producer_splits_by_group_cols():
    loc_df, acc_df = _make_loc_acc_df()
    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue()

    exposure_producer(error_queue, loc_df, acc_df, 2, ['PortNumber', 'AccNumber'], exposure_queue)

    assert error_queue.empty()
    *parts, sentinel = exposure_queue.put_items
    assert sentinel == (None, None)
    assert len(parts) == 2
    seen_accounts = set()
    for loc_part, acc_part in parts:
        assert set(loc_part['AccNumber'].unique()) == set(acc_part['AccNumber'].unique())
        seen_accounts.update(acc_part['AccNumber'].tolist())
    assert seen_accounts == {'A1', 'A2'}


def test_exposure_producer_splits_by_loc_id_when_no_group_cols():
    loc_df = pd.DataFrame({'loc_id': [1, 2, 3, 4], 'BuildingTIV': [1, 2, 3, 4]})
    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue()

    exposure_producer(error_queue, loc_df, None, 2, None, exposure_queue)

    assert error_queue.empty()
    *parts, sentinel = exposure_queue.put_items
    assert sentinel == (None, None)
    assert len(parts) == 2
    all_loc_ids = sorted(sum((p[0]['loc_id'].tolist() for p in parts), []))
    assert all_loc_ids == [1, 2, 3, 4]
    assert all(acc_part is None for _, acc_part in parts)


def test_exposure_producer_returns_immediately_if_error_already_present():
    loc_df, acc_df = _make_loc_acc_df()
    error_queue = FakeErrorQueue(has_error=True)
    exposure_queue = FakeQueue()

    exposure_producer(error_queue, loc_df, acc_df, 2, ['PortNumber', 'AccNumber'], exposure_queue)

    assert exposure_queue.put_items == []


def test_exposure_producer_retries_when_queue_full():
    loc_df = pd.DataFrame({'loc_id': [1], 'BuildingTIV': [1]})
    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue(full_times=2)

    exposure_producer(error_queue, loc_df, None, 1, None, exposure_queue)

    assert len(exposure_queue.put_items) == 2
    assert exposure_queue.put_items[-1] == (None, None)


def test_worker_processes_chunk_and_reports_result_with_account():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0, 2.0]})
    acc_df = pd.DataFrame({'LayerLimit': [10]})
    exposure_data = DummyExposureData(loc_df, acc_df)

    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue(items=[(loc_df.copy(), acc_df.copy()), (None, None)])
    result_queue = FakeQueue()

    pre_analysis_multiproc_worker(error_queue, exposure_data, DummyHook, {'multiplier': 2}, exposure_queue, result_queue)

    assert error_queue.empty()
    assert exposure_queue.put_items == [(None, None)]
    assert result_queue.put_items[-1] is None
    loc_result, acc_result, class_return = result_queue.put_items[0]
    assert loc_result['BuildingTIV'].tolist() == [2.0, 4.0]
    assert acc_result['LayerLimit'].tolist() == [20]
    assert class_return == 'ok'


def test_worker_processes_chunk_without_account():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    exposure_data = DummyExposureData(loc_df, account_df=None)

    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue(items=[(loc_df.copy(), None), (None, None)])
    result_queue = FakeQueue()

    pre_analysis_multiproc_worker(error_queue, exposure_data, DummyHook, {'multiplier': 3}, exposure_queue, result_queue)

    loc_result, acc_result, class_return = result_queue.put_items[0]
    assert loc_result['BuildingTIV'].tolist() == [3.0]
    assert acc_result is None


def test_worker_returns_immediately_if_error_already_present():
    exposure_data = DummyExposureData(pd.DataFrame({'BuildingTIV': [1.0]}))
    error_queue = FakeErrorQueue(has_error=True)
    exposure_queue = FakeQueue()
    result_queue = FakeQueue()

    pre_analysis_multiproc_worker(error_queue, exposure_data, DummyHook, {}, exposure_queue, result_queue)

    assert exposure_queue.put_items == []
    assert result_queue.put_items == []


def test_worker_retries_get_on_empty_queue():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    exposure_data = DummyExposureData(loc_df)
    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue(items=[(loc_df.copy(), None), (None, None)], empty_times=1)
    result_queue = FakeQueue()

    pre_analysis_multiproc_worker(error_queue, exposure_data, DummyHook, {'multiplier': 1}, exposure_queue, result_queue)

    assert result_queue.put_items[-1] is None


def test_worker_retries_put_when_result_queue_full():
    loc_df = pd.DataFrame({'BuildingTIV': [1.0]})
    exposure_data = DummyExposureData(loc_df)
    error_queue = FakeErrorQueue()
    exposure_queue = FakeQueue(items=[(loc_df.copy(), None), (None, None)])
    result_queue = FakeQueue(full_times=2)

    pre_analysis_multiproc_worker(error_queue, exposure_data, DummyHook, {'multiplier': 1}, exposure_queue, result_queue)

    assert len(result_queue.put_items) == 2
    assert result_queue.put_items[-1] is None


def test_result_producer_yields_until_all_workers_finish():
    result_queue = FakeQueue(items=['r1', None, 'r2', None])
    error_queue = FakeErrorQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=2))

    assert results == ['r1', 'r2']


def test_result_producer_stops_early_if_error_present():
    result_queue = FakeQueue(items=['r1'])
    error_queue = FakeErrorQueue(has_error=True)

    results = list(result_producer(result_queue, error_queue, worker_count=2))

    assert results == []


def test_result_producer_retries_get_on_empty_queue():
    result_queue = FakeQueue(items=[None], empty_times=1)
    error_queue = FakeErrorQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=1))

    assert results == []


def test_result_producer_stops_if_error_appears_mid_loop():
    error_queue = TogglingErrorQueue(true_calls=1)
    result_queue = FakeQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=5))

    assert results == []
