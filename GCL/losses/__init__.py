from .jsd import JSD, DebiasedJSD, HardnessJSD
from .vicreg import VICReg
from .infonce import InfoNCE, DebiasedInfoNCE, HardnessInfoNCE, ReweightedInfoNCE
from .triplet import TripletMargin, TripletMarginSP
from .bootstrap import BootstrapLatent
from .barlow_twins import BarlowTwins
from .loss import Loss, ContrastInstance

__all__ = [
    'Loss',
    'ContrastInstance',
    'InfoNCE',
    'DebiasedInfoNCE',
    'HardnessInfoNCE',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'TripletMarginSP',
    'VICReg',
    'BarlowTwins'
]

classes = __all__
