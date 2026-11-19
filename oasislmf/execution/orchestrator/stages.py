"""The real stages used by ``Chain`` when no factories are given.

The gulpy (``modelpy | gulpy``), PLA, summary, ELT and PLT stages are given as ``'module:attribute'`` strings, imported
only by a chain that uses them.
"""
from oasislmf.pytools.fm.stage import FmStage
from oasislmf.pytools.gulmc.stage import GulmcStage

FACTORIES = {
    'gul': GulmcStage,
    'gulpy': 'oasislmf.pytools.gul.stage:ModelpyGulpyStage',
    'fm': FmStage,
    'pla': 'oasislmf.pytools.pla.stage:PlaStage',
    'summary': 'oasislmf.pytools.summary.stage:SummaryStage',
    'elt': 'oasislmf.pytools.elt.stage:EltStage',
    'plt': 'oasislmf.pytools.plt.stage:PltStage',
}
