from .jsd import JSDLoss, DebiasedJSDLoss, HardnessJSDLoss
from .vicreg import VICRegLoss
from .infonce import InfoNCELoss, DebiasedInfoNCELoss, HardnessInfoNCELoss
from .triplet import TripletLoss
from .bootstrap import BootstrapLoss
from .barlow_twins import BarlowTwinsLoss
from .losses import Loss

__all__ = [
    'Loss',
    'InfoNCELoss',
    'DebiasedInfoNCELoss',
    'HardnessInfoNCELoss',
    'JSDLoss',
    'DebiasedJSDLoss',
    'HardnessJSDLoss',
    'TripletLoss',
    'VICRegLoss',
    'BarlowTwinsLoss'
]

classes = __all__
