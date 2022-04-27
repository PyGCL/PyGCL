from .augmentor import Augmentor, Compose, RandomChoice, PyGAugmentor, DGLAugmentor
from .identity import Identity
from .rw_sampling import RWSampling
from .ppr_diffusion import PPRDiffusion
from .markov_diffusion import MarkovDiffusion
from .edge_adding import EdgeAdding
from .edge_removing import EdgeRemoving
from .node_dropping import NodeDropping
from .node_shuffling import NodeShuffling
from .feature_masking import FeatureMasking
from .feature_dropout import FeatureDropout
from .edge_attr_masking import EdgeAttrMasking
from .adaptive_edge_removing import AdaptiveEdgeRemoving, DegreeWeights, PageRankWeights, EigenVectorWeights
from .adaptive_feature_masking import AdaptiveFeatureMasking

__all__ = [
    'Augmentor',
    'Compose',
    'RandomChoice',
    'EdgeAdding',
    'EdgeRemoving',
    'EdgeAttrMasking',
    'FeatureMasking',
    'FeatureDropout',
    'Identity',
    'PPRDiffusion',
    'MarkovDiffusion',
    'NodeDropping',
    'NodeShuffling',
    'RWSampling',
    'PyGAugmentor',
    'DGLAugmentor',
    "AdaptiveEdgeRemoving",
    "AdaptiveFeatureMasking",
    "DegreeWeights",
    "PageRankWeights",
    "EigenVectorWeights"
]

classes = __all__
