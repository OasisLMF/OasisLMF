"""The linear stage chain ``gul -> fm(il) -> fm(ri_1) -> ...`` run one event at a time in one process.

The real stage classes are resolved at run time from ``oasislmf.execution.orchestrator.stages`` so that this
module can be imported, and tested with fake stages, without them.
"""
import importlib
import json
from dataclasses import asdict, dataclass, field

from oasislmf.pytools.common.stage import EventBlock

STAGES_MODULE = 'oasislmf.execution.orchestrator.stages'
TAPS = ('gul', 'il', 'ri')
FEEDS = ('gross', 'net')
STEP_KEYS = ('name', 'feeds_next')


@dataclass
class ChainSpec:
    """Description of a chain, JSON-serialisable so that it can be sent to worker processes and Ray actors.

    Attributes:
        run_dir (str): run directory. The stages resolve their paths relative to it, so they are used with it
            as the working directory.
        gul (dict): ``GulmcStage`` keyword arguments. ``run_dir`` defaults to ``ChainSpec.run_dir``.
        fm (list[dict]): one dict per fm step, in order: ``FmStage`` keyword arguments (without
            ``input_header``) plus ``name`` (``'il'`` or ``'ri'``) and ``feeds_next`` (``'gross'`` or ``'net'``,
            the output the next step reads; ignored on the last step). ``gross`` and ``net`` default to the
            outputs the chain can ask the step for.
    """
    run_dir: str
    gul: dict = field(default_factory=dict)
    fm: list = field(default_factory=list)

    def to_dict(self):
        """Return the spec as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a spec from the output of ``to_dict``."""
        return cls(run_dir=data['run_dir'], gul=dict(data.get('gul', {})), fm=[dict(step) for step in data.get('fm', [])])

    def to_json(self):
        """Return the spec as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Build a spec from the output of ``to_json``."""
        return cls.from_dict(json.loads(text))


def build_chain_spec(run_dir, gul, il=None, ri=()):
    """Build the spec of the bash pipeline ``gulmc | fmpy (il) | fmpy -n - (ri_1) | fmpy -n - (ri_2) ...``.

    Args:
        run_dir (str): run directory.
        gul (dict): ``GulmcStage`` keyword arguments.
        il (dict, optional): ``FmStage`` keyword arguments of the IL step, or None for no IL step.
        ri (Sequence[dict]): ``FmStage`` keyword arguments of each RI step, in order. RI steps read the gross
            loss of the IL step and pass their net loss on to the next RI step.

    Returns:
        ChainSpec: the spec.
    """
    steps = []
    if il is not None:
        steps.append({**il, 'name': 'il'})
    steps.extend({**r, 'name': 'ri'} for r in ri)
    for i, step in enumerate(steps):
        if i + 1 < len(steps):
            step['feeds_next'] = 'gross' if step['name'] == 'il' else 'net'
        else:
            step['feeds_next'] = None
    return ChainSpec(run_dir=run_dir, gul=dict(gul), fm=steps)


def _resolve(factory):
    """Return ``factory``, importing it first when given as ``'package.module:attribute'``."""
    if isinstance(factory, str):
        module_name, _, attr = factory.partition(':')
        return getattr(importlib.import_module(module_name), attr)
    return factory


def default_factories():
    """Return the real stage classes, ``{'gul': GulmcStage, 'fm': FmStage}``."""
    return dict(importlib.import_module(STAGES_MODULE).FACTORIES)


class Chain:
    """A gul stage followed by fm steps, running one event at a time through all of them in-process.

    Blocks are passed between stages as ``EventBlock`` buffers owned by the chain and reused for every event.

    Attributes:
        spec (ChainSpec): the spec the chain was built from.
        gul: the gul stage.
        fm_steps (list): the fm stages, in order.
        headers (dict[str, bytes]): stream header of each available tap.
    """

    def __init__(self, spec, factories=None):
        """Build the stages.

        Args:
            spec (ChainSpec): what to build.
            factories (dict, optional): ``{'gul': callable, 'fm': callable}`` replacing the real stage classes,
                each called with the stage's keyword arguments (fm also gets ``input_header``). A value may be a
                ``'package.module:attribute'`` string. Missing keys use the real stages.
        """
        self.spec = spec
        factories = dict(factories or {})
        if 'gul' not in factories or 'fm' not in factories:
            factories = {**default_factories(), **factories}
        gul_factory = _resolve(factories['gul'])
        fm_factory = _resolve(factories['fm'])

        self._names = [step.get('name') for step in spec.fm]
        self._feeds = []
        for i, step in enumerate(spec.fm):
            feeds_next = step.get('feeds_next') if i + 1 < len(spec.fm) else None
            if i + 1 < len(spec.fm) and feeds_next not in FEEDS:
                raise ValueError(f"fm step {i} ({step.get('name')}) must have feeds_next in {FEEDS}, got {feeds_next!r}")
            self._feeds.append(feeds_next)
        self._il_index = self._names.index('il') if 'il' in self._names else None
        ri_indices = [i for i, name in enumerate(self._names) if name == 'ri']
        self._ri_index = ri_indices[-1] if ri_indices else None

        self.gul = None
        self.fm_steps = []
        try:
            self.gul = gul_factory(**{'run_dir': spec.run_dir, **spec.gul})
            header = self.gul.header
            for i, step in enumerate(spec.fm):
                kwargs = {k: v for k, v in step.items() if k not in STEP_KEYS}
                kwargs.setdefault('gross', self._feeds[i] == 'gross' or i == self._il_index)
                kwargs.setdefault('net', self._feeds[i] == 'net' or i == self._ri_index)
                stage = fm_factory(input_header=header, **kwargs)
                self.fm_steps.append(stage)
                header = stage.header
        except BaseException:
            self.close()
            raise

        self.headers = {'gul': self.gul.header}
        if self._il_index is not None:
            self.headers['il'] = self.fm_steps[self._il_index].header
        if self._ri_index is not None:
            self.headers['ri'] = self.fm_steps[self._ri_index].header

        self._gul_block = EventBlock()
        self._out_blocks = [{'gross': EventBlock(), 'net': EventBlock()} for _ in self.fm_steps]
        self._plans = {}

    def _tap_source(self, tap):
        """Return ``(step index, output)`` of the block a tap reads, or ``(None, None)`` for the gul tap."""
        if tap == 'gul':
            return None, None
        if tap == 'il':
            if self._il_index is None:
                raise ValueError("tap 'il' requested but the chain has no fm step named 'il'")
            return self._il_index, 'gross'
        if tap == 'ri':
            if self._ri_index is None:
                raise ValueError("tap 'ri' requested but the chain has no fm step named 'ri'")
            return self._ri_index, 'net'
        raise ValueError(f"unknown tap {tap!r}, expected one of {TAPS}")

    def plan(self, taps):
        """Work out which outputs each fm step must produce for a set of taps.

        Args:
            taps (Iterable[str]): requested tap names.

        Returns:
            tuple: ``(outputs, tap_sources)``. ``outputs[i]`` is the tuple of outputs (``'gross'``, ``'net'``)
            fm step ``i`` is asked for; empty when the step is not run. ``tap_sources`` maps each tap to its
            ``(step index, output)``.
        """
        key = frozenset(taps)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        tap_sources = {tap: self._tap_source(tap) for tap in key}
        needed = [set() for _ in self.fm_steps]
        for i, output in tap_sources.values():
            if i is not None:
                needed[i].add(output)
        for i in range(len(self.fm_steps) - 2, -1, -1):
            if needed[i + 1]:
                needed[i].add(self._feeds[i])
        outputs = [tuple(o for o in FEEDS if o in n) for n in needed]
        self._plans[key] = (outputs, tap_sources)
        return outputs, tap_sources

    def run_event(self, event_id, taps):
        """Run one event through the chain.

        The gul stage writes the event into a block; each fm step that is needed reads the block that feeds it
        and writes the outputs that are needed, either by the next step or by a tap. Nothing downstream runs
        once a stage writes nothing for the event.

        Args:
            event_id (int): event id.
            taps (dict[str, sink]): tap name -> sink receiving the bytes of that tap for this event.
        """
        outputs, tap_sources = self.plan(taps)
        for out in self._out_blocks:
            out['gross'].clear()
            out['net'].clear()
        block = self._gul_block
        block.clear()
        self.gul.process_event(event_id, block)
        if not block.size:
            return
        if 'gul' in taps:
            taps['gul'].write(block.view())

        for i, step in enumerate(self.fm_steps):
            wanted = outputs[i]
            if not wanted:
                break
            out = self._out_blocks[i]
            gross_out = out['gross'] if 'gross' in wanted else None
            net_out = out['net'] if 'net' in wanted else None
            step.process_block(block.view(), gross_out=gross_out, net_out=net_out)
            block = out[self._feeds[i]] if self._feeds[i] is not None else None
            if block is None or not block.size:
                break

        for tap, (i, output) in tap_sources.items():
            if i is not None and output in outputs[i]:
                produced = self._out_blocks[i][output]
                if produced.size:
                    taps[tap].write(produced.view())

    def reset(self):
        """Make every stage behave like a freshly built one for the events that follow.

        Calls ``reset()`` on the gul stage and each fm step that has one. Executors call it at the start of every
        chunk, so that a chunk's output does not depend on the chunks run before it in the same chain.
        """
        for stage in [self.gul, *self.fm_steps]:
            reset = getattr(stage, 'reset', None)
            if reset is not None:
                reset()

    def close(self):
        """Close every stage that has a ``close`` method."""
        for stage in [self.gul, *self.fm_steps]:
            close = getattr(stage, 'close', None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
