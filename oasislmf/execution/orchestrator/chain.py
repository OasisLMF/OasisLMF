"""The stage tree ``gul -> fm(il) -> fm(ri_1) -> ...`` with its summaries, run one event at a time in one process.

The trunk is the linear chain of loss stages: gulmc, then one fmpy per IL/RI step. Any loss stream of the trunk
(the gul stream, or the gross or net output of an fm step) can feed a summary node, the in-process ``summarypy``,
which writes one summary block per summary set. Each summary block feeds that set's outputs: ``PltStage`` and
``EltStage`` (``pltpy``/``eltpy``, rows appended to per-chunk files) and plain files written exactly as ``tee``
writes them (the summary header, then the blocks)::

    gul -> fm(il) -> fm(ri)
     |       |         |
     S(gul)  S(il)     S(ri)        one summary block per summary set, each feeding
                                    pltpy, eltpy, and the palt / altmeanonly / leccalc work files

The real stage classes are resolved at run time from ``oasislmf.execution.orchestrator.stages`` so that this
module can be imported, and tested with fake stages, without them.
"""
import importlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from oasislmf.pytools.common.stage import EventBlock

STAGES_MODULE = 'oasislmf.execution.orchestrator.stages'
TAPS = ('gul', 'il', 'ri')
FEEDS = ('gross', 'net')
STEP_KEYS = ('name', 'feeds_next')
CHUNK_KEY = '{k}'
TABLE_KINDS = ('plt', 'elt')


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
        summaries (list[dict]): the summary nodes, one per ``summarypy``. Each is a dict with

            * ``source``: ``{'step': None | int, 'output': None | 'gross' | 'net'}``, the stream it reads: the gul
              stream (``step`` None) or an output of fm step ``step`` (0-based);
            * ``stage``: ``SummaryStage`` keyword arguments (without ``input_header``), with ``summary_set_ids``;
            * ``sets``: one ``{'id': n, 'outputs': [...]}`` per summary set, each output either
              ``{'kind': 'plt' | 'elt', 'kwargs': {...}, 'paths': {table: template}}`` (``PltStage`` / ``EltStage``
              keyword arguments without ``run_dir`` and ``input_header``; ``paths`` as ``start_chunk`` takes them) or
              ``{'kind': 'file', 'path': template}`` (a file receiving the summary header and the set's blocks),
              and optionally ``idx_paths``: templates of files receiving the set's ``.idx`` records (no header),
              for a stage built with ``low_memory``;
            * any other keys (``runtype``, ``inuring_priority``... as the planner writes them) are labels.

            Path templates contain ``{k}``, replaced by the 1-based chunk number, and are relative to ``run_dir``
            unless absolute.
    """
    run_dir: str
    gul: dict = field(default_factory=dict)
    fm: list = field(default_factory=list)
    summaries: list = field(default_factory=list)

    def to_dict(self):
        """Return the spec as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a spec from the output of ``to_dict``."""
        return cls(run_dir=data['run_dir'], gul=dict(data.get('gul', {})), fm=[dict(step) for step in data.get('fm', [])],
                   summaries=json.loads(json.dumps(data.get('summaries', []))))

    def to_json(self):
        """Return the spec as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Build a spec from the output of ``to_json``."""
        return cls.from_dict(json.loads(text))


def build_chain_spec(run_dir, gul, il=None, ri=(), summaries=()):
    """Build the spec of the bash pipeline ``gulmc | fmpy (il) | fmpy -n - (ri_1) | fmpy -n - (ri_2) ...``.

    Args:
        run_dir (str): run directory.
        gul (dict): ``GulmcStage`` keyword arguments.
        il (dict, optional): ``FmStage`` keyword arguments of the IL step, or None for no IL step.
        ri (Sequence[dict]): ``FmStage`` keyword arguments of each RI step, in order. RI steps read the gross
            loss of the IL step and pass their net loss on to the next RI step.
        summaries (Sequence[dict]): summary nodes, see ``ChainSpec.summaries``.

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
    return ChainSpec(run_dir=run_dir, gul=dict(gul), fm=steps, summaries=[dict(node) for node in summaries])


def spec_for_taps(spec, taps):
    """Return a copy of a spec whose fm steps are built to produce the outputs the taps read.

    A step's ``gross`` and ``net`` default to what the chain can ask of it, taps included, but a spec may set
    them explicitly (the planner does, from the summaries bash runs). A tap reading an output its step was not
    built for would then fail on the first event, so the outputs the taps read are switched on here: the gross
    of the ``il`` step for ``il`` and the net of the last ``ri`` step for ``ri``. Taps whose step does not exist
    are left for ``Chain`` to reject.

    Args:
        spec (ChainSpec): the spec.
        taps (Iterable[str]): requested tap names.

    Returns:
        ChainSpec: a new spec; ``spec`` is not modified.
    """
    spec = ChainSpec.from_dict(spec.to_dict())
    names = [step.get('name') for step in spec.fm]
    taps = set(taps)
    if 'il' in taps and 'il' in names:
        step = spec.fm[names.index('il')]
        if 'gross' in step:
            step['gross'] = True
    if 'ri' in taps and 'ri' in names:
        step = spec.fm[len(names) - 1 - names[::-1].index('ri')]
        if 'net' in step:
            step['net'] = True
    return spec


def _resolve(factory):
    """Return ``factory``, importing it first when given as ``'package.module:attribute'``."""
    if isinstance(factory, str):
        module_name, _, attr = factory.partition(':')
        return getattr(importlib.import_module(module_name), attr)
    return factory


def default_factories():
    """Return the real stages, ``{'gul': GulmcStage, 'fm': FmStage, 'summary': ..., 'elt': ..., 'plt': ...}``."""
    return dict(importlib.import_module(STAGES_MODULE).FACTORIES)


def needed_factories(spec):
    """The factory keys a chain of ``spec`` uses: ``gul`` and ``fm``, ``summary`` with a summary tree, and its table kinds."""
    needed = {'gul', 'fm'}
    if spec.summaries:
        needed.add('summary')
    for node in spec.summaries:
        for summary_set in node.get('sets', []):
            needed.update(out['kind'] for out in summary_set.get('outputs', []) if out['kind'] in TABLE_KINDS)
    return needed


def import_factories(spec, factories=None):
    """Import the stage factories a ``Chain(spec, factories)`` uses: those given, the real stages for the rest.

    Args:
        spec (ChainSpec): the chain.
        factories (dict, optional): as for ``Chain``.

    Returns:
        dict: the factories of ``needed_factories(spec)``, imported.
    """
    needed = needed_factories(spec)
    factories = dict(factories or {})
    if not needed <= set(factories):
        factories = {**default_factories(), **factories}
    return {key: _resolve(factories[key]) for key in needed}


def fill_chunk(template, k):
    """Fill the ``{k}`` of a per-chunk path template with the 1-based chunk number ``k``."""
    return template.replace(CHUNK_KEY, str(k))


def chunk_path_templates(summaries):
    """Every per-chunk path template of the summary nodes of a spec (``ChainSpec.summaries``).

    Args:
        summaries (list[dict]): the summary nodes.

    Returns:
        list[str]: the templates of the plt/elt tables, plain files and ``.idx`` files, node by node and set by set.
    """
    templates = []
    for node in summaries:
        for summary_set in node.get('sets', []):
            for out in summary_set.get('outputs', []):
                if out['kind'] == 'file':
                    templates.append(out['path'])
                else:
                    templates.extend(out['paths'].values())
            templates.extend(summary_set.get('idx_paths', []))
    return templates


def summary_set_header(stage, summary_set_id):
    """The stream header a summary stage writes before the first event of one summary set.

    ``summarypy`` writes the set id in each set's header, so a ``SummaryStage`` exposes ``headers`` (set id ->
    bytes). A stage with only ``header`` (8 bytes) gets the set id appended as summarypy does.
    """
    headers = getattr(stage, 'headers', None)
    if isinstance(headers, dict):
        return bytes(headers[summary_set_id])
    return bytes(stage.header) + np.int32(summary_set_id).tobytes()


class SummaryNode:
    """Run-time state of one summary node: its stage, one block per set, and the consumers of each set.

    Attributes:
        index (int): position in ``ChainSpec.summaries``.
        spec (dict): the node's spec.
        source (tuple): ``(step, output)``, see ``ChainSpec.summaries``.
        stage: the summary stage.
        set_ids (list[int]): summary set ids, in order.
        headers (dict[int, bytes]): summary set id -> stream header.
        blocks (dict[int, EventBlock]): summary set id -> the block the stage writes the set's bytes into.
        tables (dict[int, list]): summary set id -> ``[(stage, paths template), ...]`` of its plt/elt stages.
        files (dict[int, list[str]]): summary set id -> path templates of its plain file outputs.
        open_files (dict[int, list]): summary set id -> its plain files open for the current chunk.
        idx_blocks (dict[int, EventBlock] | None): summary set id -> the block the stage writes the set's ``.idx``
            records into, when the stage is built with ``low_memory``; None otherwise.
        idx_files (dict[int, list[str]]): summary set id -> path templates of the files receiving its ``.idx`` records.
        open_idx_files (dict[int, list]): summary set id -> those files, open for the current chunk.
    """

    def __init__(self, index, spec):
        self.index = index
        self.spec = spec
        source = spec.get('source') or {}
        self.source = (source.get('step'), source.get('output'))
        self.stage = None
        self.set_ids = [int(s['id']) for s in spec.get('sets', [])]
        self.headers = {}
        self.blocks = {n: EventBlock() for n in self.set_ids}
        self.tables = {n: [] for n in self.set_ids}
        self.files = {n: [] for n in self.set_ids}
        self.open_files = {n: [] for n in self.set_ids}
        low_memory = bool(spec.get('stage', {}).get('low_memory'))
        self.idx_blocks = {n: EventBlock() for n in self.set_ids} if low_memory else None
        self.idx_files = {int(s['id']): list(s.get('idx_paths', [])) for s in spec.get('sets', [])}
        self.open_idx_files = {n: [] for n in self.set_ids}

    @property
    def name(self):
        """str: label of the node for messages, e.g. ``'summary ri_IP1'``."""
        runtype = self.spec.get('runtype') or self.spec.get('stage', {}).get('run_type', '?')
        ip = (self.spec.get('inuring_priority') or '').rstrip('_')
        return f"summary {runtype}{'_' + ip if ip else ''}"


class Chain:
    """A gul stage followed by fm steps and the summary tree, running one event at a time through all of them.

    Blocks are passed between stages as ``EventBlock`` buffers owned by the chain and reused for every event.

    Executors run a chunk as ``start_chunk(k)``, ``run_event`` for each of its events, ``finish_chunk()``. The
    per-chunk files of the summary tree are opened by ``start_chunk`` and closed by ``finish_chunk``; the gul/il/ri
    taps are sinks the executor passes to ``run_event``.

    Attributes:
        spec (ChainSpec): the spec the chain was built from.
        gul: the gul stage.
        fm_steps (list): the fm stages, in order.
        summary_nodes (list[SummaryNode]): the summary nodes, in spec order.
        headers (dict[str, bytes]): stream header of each available tap.
        chunk (int | None): the chunk opened by ``start_chunk``, None outside a chunk.
    """

    def __init__(self, spec, factories=None):
        """Build the stages.

        Args:
            spec (ChainSpec): what to build.
            factories (dict, optional): ``{'gul'|'fm'|'summary'|'elt'|'plt': callable}`` replacing the real stage
                classes, each called with the stage's keyword arguments (plus ``input_header`` for all but gul, and
                ``run_dir`` for gul, elt and plt). A value may be a ``'package.module:attribute'`` string. Keys
                missing use the real stages.
        """
        self.spec = spec
        self.chunk = None
        self.gul = None
        self.fm_steps = []
        self.summary_nodes = [SummaryNode(i, node) for i, node in enumerate(spec.summaries)]

        resolved = import_factories(spec, factories)

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

        consumed = [set() for _ in spec.fm]
        for node in self.summary_nodes:
            step, output = node.source
            if step is None:
                continue
            if not 0 <= step < len(spec.fm) or output not in FEEDS:
                raise ValueError(f"{node.name} reads {node.source}, not an output of the {len(spec.fm)} fm steps")
            consumed[step].add(output)
        self._summary_outputs = consumed

        try:
            self.gul = resolved['gul'](**{'run_dir': spec.run_dir, **spec.gul})
            header = self.gul.header
            for i, step in enumerate(spec.fm):
                kwargs = {k: v for k, v in step.items() if k not in STEP_KEYS}
                kwargs.setdefault('gross', self._feeds[i] == 'gross' or i == self._il_index or 'gross' in consumed[i])
                kwargs.setdefault('net', self._feeds[i] == 'net' or i == self._ri_index or 'net' in consumed[i])
                stage = resolved['fm'](input_header=header, **kwargs)
                self.fm_steps.append(stage)
                header = stage.header
            for node in self.summary_nodes:
                self._build_node(node, resolved)
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

    def _source_header(self, source):
        step, _output = source
        return self.gul.header if step is None else self.fm_steps[step].header

    def _build_node(self, node, resolved):
        """Build the summary stage of a node and the plt/elt stages of its sets."""
        node.stage = resolved['summary'](input_header=self._source_header(node.source), **node.spec['stage'])
        for summary_set in node.spec['sets']:
            n = int(summary_set['id'])
            node.headers[n] = summary_set_header(node.stage, n)
            for out in summary_set.get('outputs', []):
                if out['kind'] == 'file':
                    node.files[n].append(out['path'])
                elif out['kind'] in TABLE_KINDS:
                    kwargs = {'run_dir': self.spec.run_dir, **out.get('kwargs', {})}
                    table = resolved[out['kind']](input_header=node.headers[n], **kwargs)
                    node.tables[n].append((table, dict(out['paths'])))
                else:
                    raise ValueError(f"{node.name} set {n}: unknown output kind {out['kind']!r}")

    def stages(self):
        """Every stage of the chain with its name, trunk first: ``[(name, stage), ...]``."""
        named = [] if self.gul is None else [('gul', self.gul)]
        named += [(f"{self.spec.fm[i].get('name')} (fm step {i + 1})", stage) for i, stage in enumerate(self.fm_steps)]
        for node in self.summary_nodes:
            if node.stage is not None:
                named.append((node.name, node.stage))
            for n in node.set_ids:
                named += [(f'{node.name} S{n} {type(table).__name__}', table) for table, _paths in node.tables[n]]
        return named

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
        """Work out which outputs each fm step must produce for a set of taps and the summary tree.

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
        needed = [set(outputs) for outputs in self._summary_outputs]
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
        and writes the outputs that are needed, by the next step, a tap or a summary node. Nothing downstream runs
        once a stage writes nothing for the event. Then each summary node whose source block is not empty writes
        one block per summary set, and each non-empty set block goes to the set's plt/elt stages and files.

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

        for node in self.summary_nodes:
            self._run_node(node)

    def _run_node(self, node):
        """Run one summary node on the event held in its source block."""
        step, output = node.source
        source = self._gul_block if step is None else self._out_blocks[step][output]
        if not source.size:
            return
        for summary_block in node.blocks.values():
            summary_block.clear()
        if node.idx_blocks is None:
            node.stage.process_block(source.view(), node.blocks)
        else:
            for idx_block in node.idx_blocks.values():
                idx_block.clear()
            node.stage.process_block(source.view(), node.blocks, idx_outs=node.idx_blocks)
            for n, idx_block in node.idx_blocks.items():
                if idx_block.size:
                    for f in node.open_idx_files[n]:
                        f.write(idx_block.view())
        for n, summary_block in node.blocks.items():
            if not summary_block.size:
                continue
            view = summary_block.view()
            for table, _paths in node.tables[n]:
                table.process_block(view)
            for f in node.open_files[n]:
                f.write(view)

    def reset(self):
        """Make every stage behave like a freshly built one for the events that follow.

        Calls ``reset()`` on every stage that has one. ``start_chunk`` calls it; it is kept for callers that manage
        no per-chunk files.
        """
        for _name, stage in self.stages():
            reset = getattr(stage, 'reset', None)
            if reset is not None:
                reset()

    def chunk_paths(self, k):
        """Every file the summary tree writes for chunk k, as paths relative to the run directory."""
        return [fill_chunk(template, k) for template in chunk_path_templates(self.spec.summaries)]

    def start_chunk(self, k):
        """Start chunk k: reset every stage, then open the chunk's files of the summary tree.

        Directories are created as needed. Plain files get their summary header at once, so a chunk without any
        event still leaves header-only files, as ``tee`` does.

        Args:
            k (int): 1-based chunk number, filled into the path templates.
        """
        if self.chunk is not None:
            raise ValueError(f'chunk {self.chunk} is still open, call finish_chunk first')
        self.reset()
        for path in self.chunk_paths(k):
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        self.chunk = k
        try:
            for node in self.summary_nodes:
                for n in node.set_ids:
                    for table, templates in node.tables[n]:
                        table.start_chunk({name: fill_chunk(t, k) for name, t in templates.items()})
                    for template in node.files[n]:
                        f = open(fill_chunk(template, k), 'wb')
                        node.open_files[n].append(f)
                        f.write(node.headers[n])
                    for template in node.idx_files[n]:
                        node.open_idx_files[n].append(open(fill_chunk(template, k), 'wb'))
        except BaseException:
            self._abort_chunk()
            raise

    def finish_chunk(self):
        """Finish the open chunk: flush and close every file it wrote."""
        if self.chunk is None:
            raise ValueError('no chunk is open, call start_chunk first')
        try:
            for node in self.summary_nodes:
                for n in node.set_ids:
                    for table, _templates in node.tables[n]:
                        table.finish_chunk()
        finally:
            self._close_files()
            self.chunk = None

    def _close_files(self):
        """Close the plain files of the open chunk."""
        for node in self.summary_nodes:
            for n in node.set_ids:
                files, node.open_files[n] = node.open_files[n], []
                idx_files, node.open_idx_files[n] = node.open_idx_files[n], []
                for f in files + idx_files:
                    f.close()

    def _abort_chunk(self):
        """Close whatever the open chunk opened, ignoring the stages that had not opened their files."""
        for node in self.summary_nodes:
            for n in node.set_ids:
                for table, _templates in node.tables[n]:
                    try:
                        table.finish_chunk()
                    except Exception:
                        pass
        self._close_files()
        self.chunk = None

    def close(self):
        """Close the files of an unfinished chunk and every stage that has a ``close`` method."""
        if self.chunk is not None:
            self._close_files()
            self.chunk = None
        for _name, stage in self.stages():
            close = getattr(stage, 'close', None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

