__all__ = [
    'PERILS',
    'PERIL_GROUPS'
]

from collections import OrderedDict

PRL_BBF = 'BBF'
PRL_BFR = 'BFR'
PRL_BSK = 'BSK'
PRL_MNT = 'MNT'
PRL_MTR = 'MTR'
PRL_ORF = 'ORF'
PRL_OSF = 'OSF'
PRL_QEQ = 'QEQ'
PRL_QFF = 'QFF'
PRL_QLF = 'QLF'
PRL_QLS = 'QLS'
PRL_QSL = 'QSL'
PRL_QTS = 'QTS'
PRL_WEC = 'WEC'
PRL_WSS = 'WSS'
PRL_WTC = 'WTC'
PRL_XHL = 'XHL'
PRL_XLT = 'XLT'
PRL_XSL = 'XSL'
PRL_XTD = 'XTD'
PRL_ZFZ = 'ZFZ'
PRL_ZIC = 'ZIC'
PRL_ZSN = 'ZSN'
PRL_ZST = 'ZST'

PRL_GRP_AA1 = 'AA1'
PRL_GRP_BB1 = 'BB1'
PRL_GRP_MM1 = 'MM1'
PRL_GRP_OO1 = 'OO1'
PRL_GRP_QQ1 = 'QQ1'
PRL_GRP_WW1 = 'WW1'
PRL_GRP_WW2 = 'WW2'
PRL_GRP_XX1 = 'XX1'
PRL_GRP_XZ1 = 'XZ1'
PRL_GRP_ZZ1 = 'ZZ1'

PERIL_GROUPS = OrderedDict({
    'all': {'id': PRL_GRP_AA1, 'desc': 'All perils', 'peril_ids': [PRL_QEQ, PRL_QFF, PRL_QTS, PRL_QSL, PRL_QLS, PRL_QLF, PRL_WTC, PRL_WEC, PRL_WSS, PRL_ORF, PRL_OSF, PRL_XSL, PRL_XTD, PRL_XHL, PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_BFR, PRL_BBF, PRL_MNT, PRL_MTR, PRL_XLT, PRL_ZST, PRL_BSK]},
    'wildfire w/ smoke': {'id': PRL_GRP_BB1, 'desc': 'Wildfire with smoke', 'peril_ids': [PRL_BBF, PRL_BSK]},
    'terrorism': {'id': PRL_GRP_MM1, 'desc': 'Terrorism', 'peril_ids': [PRL_MNT, PRL_MTR]},
    'flood w/o storm surge': {'id': PRL_GRP_OO1, 'desc': 'Flood w/o storm surge', 'peril_ids': [PRL_ORF, PRL_OSF]},
    'earthquake': {'id': PRL_GRP_QQ1, 'desc': 'All EQ perils', 'peril_ids': [PRL_QEQ, PRL_QFF, PRL_QTS, PRL_QSL, PRL_QLS, PRL_QLF]},
    'windstorm w/ storm surge': {'id': PRL_GRP_WW1, 'desc': 'Windstorm with storm surge', 'peril_ids': [PRL_WTC, PRL_WEC, PRL_WSS]},
    'windstorm w/o storm surge': {'id': PRL_GRP_WW2, 'desc': 'Windstorm w/o storm surge', 'peril_ids': [PRL_WTC, PRL_WEC]},
    'convective storm': {'id': PRL_GRP_XX1, 'desc': 'Convective Storm', 'peril_ids': [PRL_XSL, PRL_XTD, PRL_XHL, PRL_XLT]},
    'convective storm rms': {'id': PRL_GRP_XZ1, 'desc': 'Convective storm (incl winter storm) - for RMS users', 'peril_ids': [PRL_XSL, PRL_XTD, PRL_XHL, PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_XLT, PRL_ZST]},
    'winter storm': {'id': PRL_GRP_ZZ1, 'desc': 'Winter storm', 'peril_ids': [PRL_ZSN, PRL_ZIC, PRL_ZFZ, PRL_ZST]}
})

PERILS = OrderedDict({
    'wildfire': {'id': PRL_BBF, 'desc': 'Wildfire / Bushfire', 'group_peril': 'wildfire w/ smoke'},
    'noncat': {'id': PRL_BFR, 'desc': 'NonCat', 'group_peril': 'wildfire w/ smoke'},
    'smoke': {'id': PRL_BSK, 'desc': 'Smoke', 'group_peril': 'wildfire w/ smoke'},
    'nbcr terrorism': {'id': PRL_MNT, 'desc': 'NBCR Terrorism', 'group_peril': 'terrorism'},
    'terrorism': {'id': PRL_MTR, 'desc': 'Conventional Terrorism', 'group_peril': 'terrorism'},
    'river flood': {'id': PRL_ORF, 'desc': 'River / Fluvial Flood', 'group_peril': 'flood w/o storm surge'},
    'flash flood': {'id': PRL_OSF, 'desc': 'Flash / Surface / Pluvial Flood', 'group_peril': 'flood w/o storm surge'},
    'earthquake': {'id': PRL_QEQ, 'desc': 'Earthquake - Shake only', 'group_peril': 'earthquake'},
    'fire following': {'id': PRL_QFF, 'desc': 'Fire Following', 'group_peril': 'earthquake'},
    'liquefaction': {'id': PRL_QLF, 'desc': 'Liquefaction', 'group_peril': 'earthquake'},
    'landslide': {'id': PRL_QLS, 'desc': 'Landslide', 'group_peril': 'earthquake'},
    'sprinkler leakage': {'id': PRL_QSL, 'desc': 'Sprinkler Leakage', 'group_peril': 'earthquake'},
    'tsunami': {'id': PRL_QTS, 'desc': 'Tsunami', 'group_peril': 'earthquake'},
    'extra tropical cyclone': {'id': PRL_WEC, 'desc': 'Extra Tropical Cyclone', 'group_peril': 'windstorm w/o storm surge'},
    'storm surge': {'id': PRL_WSS, 'desc': 'Storm Surge', 'group_peril': 'windstorm with storm surge'},
    'tropical cyclone': {'id': PRL_WTC, 'desc': 'Tropical Cyclone', 'group_peril': 'windstorm w/o storm surge'},
    'hail': {'id': PRL_XHL, 'desc': 'Hail', 'group_peril': 'convective storm rms'},
    'lightning': {'id': PRL_XLT, 'desc': 'Lightning', 'group_peril': 'convective storm'},
    'convective wind': {'id': PRL_XSL, 'desc': 'Straight-line / other convective wind', 'group_peril': 'convective storm'},
    'tornado': {'id': PRL_XTD, 'desc': 'Tornado', 'group_peril': 'convective storm'},
    'freeze': {'id': PRL_ZFZ, 'desc': 'Freeze', 'group_peril': 'winter storm'},
    'ice': {'id': PRL_ZIC, 'desc': 'Ice', 'group_peril': 'winter storm'},
    'snow': {'id': PRL_ZSN, 'desc': 'Snow', 'eqv_oasis_peril': None, 'group_peril': 'winter storm'},
    'winterstorm wind': {'id': PRL_ZST, 'desc': 'Winterstorm Wind', 'group_peril': 'winter storm'}
})
