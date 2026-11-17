"""The pool worker's protocol (``orchestrator/worker.py``), served in the test process over a pipe.

The pool tests run the worker as a subprocess; these drive ``serve`` directly, so that each answer to a message, and
each way it can fail, is checked on its own.
"""
from multiprocessing import Pipe

import numpy as np
import pytest

from oasislmf.execution.orchestrator import worker
from oasislmf.execution.orchestrator.outputs import JobError
from oasislmf.execution.orchestrator.plan import OutputJob

from .test_chain import expected_streams, make_spec
from .test_pool import FAKE_FACTORIES, TAPS, fake_headers


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / 'run'
    (path / 'input').mkdir(parents=True)
    return path


def serve(run_dir, out_dir, messages, factories=FAKE_FACTORIES, jobs_only=False):
    """Queue ``messages`` from the orchestrator, serve them, and return the exit status and what the worker sent."""
    ours, theirs = Pipe()
    for message in messages:
        ours.send(message)
    config = {'worker': 3, 'spec': make_spec(str(run_dir)).to_dict(), 'factories': factories, 'taps': TAPS,
              'out_dir': str(out_dir), 'jobs_only': jobs_only}
    try:
        status = worker.serve(theirs, config)
        sent = []
        while ours.poll():
            sent.append(ours.recv())
    finally:
        ours.close()
        theirs.close()
    return status, sent


def test_serve_runs_chunks_until_stop(tmp_path, run_dir):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    status, sent = serve(run_dir, out_dir, [{'type': 'chunk', 'chunk': 2, 'events': [1, 2, 3]}, {'type': 'stop'}])
    assert status == 0
    assert [m['type'] for m in sent] == ['hello', 'ready', 'done', 'ready']
    assert sent[0]['worker'] == 3
    assert sent[2]['chunk'] == 2 and sent[2]['events'] == 3
    expected = expected_streams([1, 2, 3])
    for tap in TAPS:
        assert (out_dir / f'{tap}_P2.bin').read_bytes() == fake_headers()[tap] + expected[tap]


def test_serve_reports_a_chain_that_cannot_be_built(tmp_path, run_dir):
    factories = {**FAKE_FACTORIES, 'gul': 'tests.orchestrator.test_chain:NoSuchGul'}
    status, sent = serve(run_dir, tmp_path, [{'type': 'stop'}], factories=factories)
    assert status == worker.EXIT_ERROR
    assert [m['type'] for m in sent] == ['hello', 'error']
    assert sent[1]['stage'] == 'setup' and 'NoSuchGul' in sent[1]['error']


def test_serve_jobs_only_worker_refuses_a_chunk(tmp_path, run_dir):
    status, sent = serve(run_dir, tmp_path, [{'type': 'chunk', 'chunk': 1, 'events': [1]}], jobs_only=True)
    assert status == worker.EXIT_ERROR
    assert [m['type'] for m in sent] == ['hello', 'ready', 'error']
    assert sent[2]['stage'] == 'chain' and sent[2]['chunk'] == 1 and 'without a chain' in sent[2]['error']


def test_serve_reports_a_failed_output_job_with_its_tool_and_index(tmp_path, run_dir, monkeypatch):
    def failing_job(message, run_dir):
        job = OutputJob(tool=message['tool'], kwargs=message['kwargs'])
        try:
            raise OSError('disk full')
        except OSError as exc:
            raise JobError('output job failed', job=job, index=message['job']) from exc

    monkeypatch.setattr(worker, 'run_output_job', failing_job)
    status, sent = serve(run_dir, tmp_path, [{'type': 'job', 'job': 4, 'tool': 'aalpy', 'kwargs': {}}])
    assert status == worker.EXIT_ERROR
    assert [m['type'] for m in sent] == ['hello', 'ready', 'error']
    assert sent[2]['stage'] == 'aalpy' and sent[2]['job'] == 4 and sent[2]['error'] == 'OSError: disk full'


def test_serve_closes_the_chain_before_the_first_job(tmp_path, run_dir, monkeypatch):
    closed = []
    monkeypatch.setattr(worker.Chain, 'close', lambda self: closed.append(True))
    monkeypatch.setattr(worker, 'run_output_job', lambda message, run_dir: {'type': 'done', 'job': message['job'], 'seconds': 0.0})
    status, sent = serve(run_dir, tmp_path, [{'type': 'job', 'job': 0, 'tool': 'aalpy', 'kwargs': {}},
                                             {'type': 'job', 'job': 1, 'tool': 'aalpy', 'kwargs': {}}, {'type': 'stop'}])
    assert status == 0
    assert [m['type'] for m in sent] == ['hello', 'ready', 'done', 'ready', 'done', 'ready']
    assert [m['job'] for m in sent if m['type'] == 'done'] == [0, 1]
    assert closed == [True]


def test_run_chunk_reports_progress_and_the_failing_event(tmp_path, run_dir, monkeypatch):
    monkeypatch.chdir(run_dir)
    chain = worker.Chain(make_spec(str(run_dir)), factories=FAKE_FACTORIES)
    reported = []
    done = worker.run_chunk(chain, str(tmp_path), TAPS, 1, np.array([1, 2, 3], dtype=np.int32), report=reported.append,
                            report_interval=0)
    assert done['events'] == 3 and sum(reported) == 3

    def failing_event(event_id, sinks):
        if event_id == 2:
            raise ValueError('bad event')

    monkeypatch.setattr(chain, 'run_event', failing_event)
    with pytest.raises(worker.ChunkError) as info:
        worker.run_chunk(chain, str(tmp_path), TAPS, 2, [1, 2, 3])
    assert info.value.event_id == 2 and isinstance(info.value.__cause__, ValueError)
    chain.close()
