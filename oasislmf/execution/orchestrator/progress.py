"""Run progress of an orchestrated kernel run, reported as the bash run reports it.

In a bash run ``gulmc --socket-server`` sends the number of events it has completed since its previous message
(``{"events_complete": n, "analysis_pk": pk}``, ``oasislmf.utils.ping.oasis_ping``), at most every
``SERVER_UPDATE_TIME`` seconds and once more when its input ends: to the platform (HTTP or websocket) when the run
has an ``analysis_pk``, else to the local ``socket-server`` the script starts, which draws a progress bar. The
orchestrator knows how many events are done (the pool's ``progress`` callback), so it reports them itself:
``ProgressReporter`` sends the same messages to the platform, or draws the same bar in-process when the run is local.
"""
import logging
import threading

from oasislmf.utils.defaults import SERVER_UPDATE_TIME
from oasislmf.utils.ping import oasis_ping
from oasislmf.utils.rounded_tqdm import rounded_tqdm

logger = logging.getLogger(__name__)

BAR_DESCRIPTION = 'Gul events completed'


class ProgressReporter:
    """Report the events done during a run: pings to the platform, or a local progress bar.

    ``update(events_done)`` may be called as often as wanted, from one thread; with an ``analysis_pk`` a background
    thread sends what is new every ``interval`` seconds (a ping that fails is retried with the next one, so nothing
    is lost or counted twice), and ``close`` sends the rest. Without one a progress bar of ``bar_total`` events is
    drawn on stderr, as the ``socket-server`` of a local bash run does; with neither, nothing is reported.

    Usage::

        with ProgressReporter(analysis_pk=pk) as reporter:
            run_chunks(..., progress=lambda done, total: reporter.update(done))
    """

    def __init__(self, analysis_pk=None, bar_total=None, interval=SERVER_UPDATE_TIME, ping=None):
        """Start reporting.

        Args:
            analysis_pk (int, optional): the platform's analysis id; pings are sent when it is set.
            bar_total (int, optional): without ``analysis_pk``, the number of events of the progress bar (bash's
                ``socket_server_size``); no bar when None or 0.
            interval (float): seconds between two pings.
            ping (callable, optional): sends one message, returns True when it got through; ``oasis_ping`` by default.
        """
        self.analysis_pk = analysis_pk
        self.interval = interval
        self._ping = ping or oasis_ping
        self._done = 0
        self._sent = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._bar = None
        if analysis_pk is not None:
            self._thread = threading.Thread(target=self._loop, name='orchestrator-progress', daemon=True)
            self._thread.start()
        elif bar_total:
            self._bar = rounded_tqdm(total=bar_total, unit='events', desc=BAR_DESCRIPTION, leave=True)

    @property
    def sent(self):
        """int: events reported to the platform so far."""
        return self._sent

    def update(self, events_done):
        """Record that ``events_done`` events of the run are done in all (a running total, never decreasing)."""
        with self._lock:
            new = events_done - self._done
            if new <= 0:
                return
            self._done = events_done
        if self._bar is not None:
            self._bar.update(new)

    def _send(self):
        """Ping the events done since the last ping that got through; returns False if this one did not."""
        with self._lock:
            new = self._done - self._sent
        if new <= 0:
            return True
        try:
            delivered = self._ping({'events_complete': new, 'analysis_pk': self.analysis_pk})
        except Exception as exc:
            logger.warning('progress ping failed: %s', exc)
            delivered = False
        if delivered:
            with self._lock:
                self._sent += new
        return delivered

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._send()

    def close(self):
        """Stop the background thread and report what is left, once."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._send()
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
