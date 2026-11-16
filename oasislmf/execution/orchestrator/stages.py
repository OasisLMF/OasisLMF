"""The real stage classes used by ``Chain`` when no factories are given."""
from oasislmf.pytools.fm.stage import FmStage
from oasislmf.pytools.gulmc.stage import GulmcStage

FACTORIES = {'gul': GulmcStage, 'fm': FmStage}
