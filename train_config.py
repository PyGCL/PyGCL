from enum import Enum
from dataclasses import dataclass


@dataclass
class OptConfig:
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    num_epochs: int = 1000
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
    hidden_dim: int = 256
    proj_dim: int = 256
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
    drop_edge_prob: float = 0.1
    drop_feat_prob: float = 0.1


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
