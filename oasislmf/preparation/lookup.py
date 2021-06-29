__all__ = [
    'OasisBuiltinBaseLookup',
    'OasisBaseKeysLookup',
    'OasisLookup',
    'OasisPerilLookup',
    'OasisVulnerabilityLookup',
    'OasisLookupFactory'
]


## This is a backwards compatibility file for lookup code based on version 1.9.0 and older
## Add deprecated warning, Load equiv classes from ..lookup. * 

from ..lookup.factory import KeyServerFactory as OasisLookupFactory
from ..lookup.base import OasisBaseLookup as OasisBuiltinBaseLookup
from ..lookup.interface import OasisLookupInterface as OasisBaseKeysLookup 
