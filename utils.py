import torch
import os.path as osp
import GCL.augmentors as A
import GCL.losses as L
import torch_geometric.transforms as T

from torch_scatter import scatter_add
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Coauthor, WikiCS, Amazon, CitationFull, Planetoid, TUDataset


def load_dataset(path: str, name: str, to_sparse_tensor: bool = True, to_dense: bool = False):
    node_datasets = {'Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                     'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code'}
    if name in node_datasets:
        return load_node_dataset(path, name, to_sparse_tensor, to_dense)
    else:
        return load_graph_dataset(path, name, to_sparse_tensor, to_dense)


def load_node_dataset(path, name, to_sparse_tensor=True, to_dense=False):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~')
    path = osp.join(root_path, path)
    transform = [T.NormalizeFeatures()]
    if to_sparse_tensor:
        transform.append(T.ToSparseTensor())
    if to_dense:
        transform.append(T.ToDense())
    transform = T.Compose(transform)

    if name == 'Coauthor-CS':
        return Coauthor(root=osp.join(path, 'Coauthor-CS'), name='cs', transform=transform)

    if name == 'Coauthor-Phy':
        return Coauthor(root=osp.join(path, 'Coauthor-Phy'), name='physics', transform=transform)

    if name == 'WikiCS':
        return WikiCS(root=osp.join(path, 'WikiCS'), transform=transform)

    if name == 'Amazon-Computers':
        return Amazon(root=osp.join(path, 'Amazon-Computers'), name='computers', transform=transform)

    if name == 'Amazon-Photo':
        return Amazon(root=osp.join(path, 'Amazon-Photo'), name='photo', transform=transform)

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=transform)

    if name == 'dblp':
        return CitationFull(path, name=name, transform=transform)

    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(path, name=name, transform=transform)


def load_graph_dataset(path, name, to_sparse_tensor=True, to_dense=False):
    root_path = osp.expanduser('~')
    path = osp.join(root_path, path)
    if to_sparse_tensor:
        transform = T.ToSparseTensor(remove_edge_index=False)
    elif to_dense:
        transform = T.ToDense()
    else:
        transform = None
    return TUDataset(path, name=name, transform=transform)


def get_activation(name: str):
    activations = {
        'relu': torch.nn.ReLU,
        'hardtanh': torch.nn.Hardtanh,
        'elu': torch.nn.ELU,
        'leakyrelu': torch.nn.LeakyReLU,
        'prelu': torch.nn.PReLU,
        'rrelu': torch.nn.RReLU
    }

    return activations[name]


def get_augmentor(aug_name: str, param: dict):
    if aug_name == 'ER':
        return A.EdgeRemoving(pe=param['drop_edge_prob'])
    if aug_name == 'EA':
        return A.EdgeAdding(pe=param['add_edge_prob'])
    if aug_name == 'ND':
        return A.NodeDropping(pn=param['drop_node_prob'])
    if aug_name == 'RWS':
        return A.RWSampling(num_seeds=param['num_seeds'], walk_length=param['walk_length'])
    if aug_name == 'PPR':
        return A.PPRDiffusion(eps=param['sp_eps'], use_cache=False)
    if aug_name == 'MKD':
        return A.MarkovDiffusion(sp_eps=param['sp_eps'], use_cache=False)
    if aug_name == 'ORI':
        return A.Identity()
    if aug_name == 'FM':
        return A.FeatureMasking(pf=param['drop_feat_prob'])
    if aug_name == 'EAM':
        return A.EdgeAttrMasking(pf=param['drop_feat_prob'])
    if aug_name == 'FD':
        return A.FeatureDropout(pf=param['drop_feat_prob'])

    raise NotImplementedError(f'unsupported augmentation name: {aug_name}')


def get_compositional_augmentor(param: dict) -> A.Augmentor:
    augs = param['scheme'].split('+')
    augs = [get_augmentor(x, param) for x in augs]

    aug = augs[0]
    for a in augs[1:]:
        aug = aug >> a
    return aug


def get_loss(loss, mode):
    if mode == 'G2LEN':
        if loss == 'infonce':
            return L.InfoNCELossG2LEN()
        elif loss == 'jsd':
            return L.JSDLossG2LEN(discriminator=lambda x, y: x @ y.t())
        elif loss == 'triplet':
            return L.TripletLossG2LEN()
    elif mode == 'G2L':
        if loss == 'infonce':
            return L.InfoNCELossG2L()
        elif loss == 'jsd':
            return L.JSDLossG2L(discriminator=lambda x, y: x @ y.t())
        elif loss == 'triplet':
            return L.TripletLossG2L()
    else:  # mode is L2L or G2G
        if loss == 'infonce':
            return L.InfoNCELoss(loss_fn=L.nt_xent_loss)
        elif loss == 'debiased_infonce':
            return L.InfoNCELoss(loss_fn=L.debiased_nt_xent_loss)
        elif loss == 'hardness_infonce':
            return L.InfoNCELoss(loss_fn=L.hardness_nt_xent_loss)
        elif loss == 'subsampling_infonce':
            return L.InfoNCELoss(loss_fn=L.subsampling_nt_xent_loss)
        elif loss == 'ring_loss':
            return L.RingLoss()
        elif loss == 'jsd':
            return L.JSDLoss(discriminator=lambda x, y: x @ y.t())
        elif loss == 'triplet':
            return L.TripletLoss()
        elif loss == 'barlow_twins':
            return L.BTLoss()
        elif loss == 'vicreg':
            return L.VICRegLoss()
    raise NotImplementedError(f'Unsupported loss {loss} or contrasting mode {mode}')


def set_differ(s1, s2):
    combined = torch.cat([s1, s2])
    uniques, counts = combined.unique(return_counts=True, dim=0)
    difference = uniques[counts == 1]
    return difference


def set_intersect(s1, s2):
    combined = torch.cat([s1, s2])
    uniques, counts = combined.unique(return_counts=True, dim=0)
    intersection = uniques[counts > 1]
    return intersection


def indices_to_mask(indices: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    return scatter_add(torch.ones_like(indices, dtype=torch.float32), indices, dim=1, dim_size=num_nodes)
