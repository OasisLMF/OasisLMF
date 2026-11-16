"""The real stages used by ``Chain`` when no factories are given.

The summary, ELT and PLT stages are given as ``'module:attribute'`` strings, imported only by a chain that has a
summary tree.
"""
from oasislmf.pytools.fm.stage import FmStage
from oasislmf.pytools.gulmc.stage import GulmcStage

FACTORIES = {
    'gul': GulmcStage,
    'fm': FmStage,
    'summary': 'oasislmf.pytools.summary.stage:SummaryStage',
    'elt': 'oasislmf.pytools.elt.stage:EltStage',
    'plt': 'oasislmf.pytools.plt.stage:PltStage',
}
