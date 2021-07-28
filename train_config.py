from enum import Enum
from dataclasses import dataclass
from happy_config.param_tuning import with_search_space


@dataclass
class OptConfig:
    learning_rate: float = with_search_space(0.001,
                                             space_type='choice', space_value=[1 / (10 ** i) for i in range(1, 5)])
    weight_decay: float = with_search_space(1e-5,
                                            space_type='choice', space_value=[1 / (10 ** i) for i in range(3, 9)])
    num_epochs: int = with_search_space(1000,
                                        space_type='choice', space_value=[50, 100, 200, 500, 1000, 2000, 5000])
    patience: int = 200


class ConvType(Enum):
    GCNConv = 'GCNConv'
    GINConv = 'GINConv'
    GATConv = 'GATConv'


class ActivationType(Enum):
    ReLU = 'relu'
    PReLU = 'prelu'
    RReLU = 'rrelu'
    ELU = 'elu'
    LeakyReLU = 'leakyrelu'


@dataclass
class EncoderConfig:
    hidden_dim: int = with_search_space(256, space_type='choice', space_value=[2 ** i for i in range(6, 10)])
    proj_dim: int = with_search_space(256, space_type='choice', space_value=[2 ** i for i in range(6, 10)])
    conv: ConvType = ConvType.GCNConv
    activation: ActivationType = ActivationType.ReLU
    num_layers: int = 2


class Objective(Enum):
    InfoNCE = 'infonce'
    JSD = 'jsd'
    Triplet = 'triplet'


@dataclass
class InfoNCE:
    tau: float = 0.4


@dataclass
class JSD:
    pass


@dataclass
class Triplet:
    margin: float = 10.0


@dataclass
class ObjConfig:
    loss: Objective = Objective.InfoNCE
    infonce: InfoNCE = InfoNCE()
    jsd: JSD = JSD()
    triplet: Triplet = Triplet()


@dataclass
class AugmentorConfig:
    scheme: str = 'FM+ER'
    drop_edge_prob: float = 0.2
    drop_feat_prob: float = 0.2


@dataclass
class ExpConfig:
    device: str = 'cuda:0'
    dataset: str = 'Amazon-Computers'

    seed: int = 39788
    opt: OptConfig = OptConfig()
    encoder: EncoderConfig = EncoderConfig()
    obj: ObjConfig = ObjConfig()

    augmentor1: AugmentorConfig = AugmentorConfig()
    augmentor2: AugmentorConfig = AugmentorConfig()
