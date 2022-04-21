from .jsd import JSD, DebiasedJSD, HardnessJSD
from .vicreg import VICReg
from .infonce import InfoNCE, DebiasedInfoNCE, HardnessInfoNCE, ReweightedInfoNCE, RobustInfoNCE
from .triplet import TripletMargin
from .bootstrap import BootstrapLatent
from .barlow_twins import BarlowTwins
from .loss import Loss, ContrastInstance

__all__ = [
    'Loss',
    'ContrastInstance',
    'InfoNCE',
    'DebiasedInfoNCE',
    'HardnessInfoNCE',
    'RobustInfoNCE',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'VICReg',
    'BarlowTwins'
]

classes = __all__
