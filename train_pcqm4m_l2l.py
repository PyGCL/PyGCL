import os
# import tensorflow
import nni
import math
import torch
import torch.nn as nn
import argparse
import pretty_errors
from time import time_ns

from ogb.lsc import PCQM4MEvaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.utils.data import Subset
from tqdm import tqdm
from time import perf_counter

# import GCL.augmentation as A
import GCL.augmentations as A
import GCL.augmentations.functional as AF
import GCL.utils.simple_param as SP
from GCL.eval import MLP_regression

from torch import nn
from torch.optim import Adam
from GCL.eval import LR_classification
from GCL.utils import seed_everything
from torch_geometric.nn import global_add_pool, GINEConv
from torch_geometric.data import DataLoader

from utils import load_graph_dataset, get_activation
from models.GRACE import GRACE


def make_gin_conv(input_dim: int, out_dim: int) -> GINEConv:
    return GINEConv(nn.Sequential(
        nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.ReLU(), nn.Linear(input_dim, out_dim)))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation, num_layers: int, batch_norm: bool = False):
        super(Encoder, self).__init__()
        self.activation = activation()

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if batch_norm else None
        self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        for _ in range(num_layers - 1):
            # add batch norm layer if batch norm is used
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight):
        z = x
        edge_attr = edge_weight
        num_layers = len(self.layers)
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_attr)
            z = self.activation(z)
            if self.batch_norms is not None and i != num_layers - 1:
                z = self.batch_norms[i](z)
        return z


def train(model: GRACE, optimizer: Adam, loader: DataLoader, device, args, epoch: int, num_graphs: int):
    model.train()
    tot_loss = 0.0
    pbar = tqdm(total=num_graphs)
    pbar.set_description(f'epoch {epoch}')
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        optimizer.zero_grad()
        dense_x = model.encoder.atom_encoder(data.x)
        edge_attr = model.encoder.bond_encoder(data.edge_attr)
        _, z1, z2 = model(dense_x, data.edge_index, edge_attr)

        if args.loss == 'nt_xent':
            loss = model.loss(z1, z2)
        elif args.loss == 'jsd':
            loss = model.jsd_loss(z1, z2)
        elif args.loss == 'triplet':
            loss = model.triplet_loss(z1, z2)
        else:
            raise NotImplementedError(f'Unknown loss type: {args.loss}')

        # loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

        pbar.update(data.batch.max().item() + 1)
    pbar.close()
    return tot_loss


def test(model, loader, device, seed, num_graphs, split):
    model.eval()
    x = []
    y = []
    pbar = tqdm(total=num_graphs)
    pbar.set_description('(E) embedding')
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        dense_x = model.encoder.atom_encoder(data.x)
        edge_attr = model.encoder.bond_encoder(data.edge_attr)
        z, _, _ = model(dense_x, data.edge_index, edge_attr)

        g = global_add_pool(z, data.batch)

        x.append(g.detach())
        y.append(data.y)

        pbar.update(data.batch.max().item() + 1)
    pbar.close()

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    results = dict()

    evaluator = PCQM4MEvaluator()
    res = MLP_regression(x, y, target=(0, 'default'), evaluator=evaluator, split=split, num_epochs=5000)

    return res


def main():
    default_param = {
        'seed': 39788,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'proj_dim': 256,
        'weight_decay': 1e-5,
        'activation': 'prelu',
        'base_model': 'GINConv',
        'num_layers': 2,
        'drop_edge_prob1': 0.2,
        'drop_edge_prob2': 0.1,
        'add_edge_prob1': 0.1,
        'add_edge_prob2': 0.1,
        'drop_node_prob1': 0.1,
        'drop_node_prob2': 0.1,
        'drop_feat_prob1': 0.3,
        'drop_feat_prob2': 0.2,
        'patience': 10000,
        'num_epochs': 2,
        'batch_size': 256,
        'tau': 0.8,
        'sp_eps': 0.001,
        'num_seeds': 1000,
        'walk_length': 10
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--param_path', type=str, default='params/GlobalGRACE/pcqm4m.json')
    parser.add_argument('--aug1', type=str, default='FM+ER')
    parser.add_argument('--aug2', type=str, default='FM+ER')
    parser.add_argument('--loss', type=str, default='nt_xent', choices=['nt_xent', 'jsd', 'triplet', 'mixup'])
    parser.add_argument('--subset_size', type=int, default=10000)
    for k, v in default_param.items():
        parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    param = sp(args.param_path, preprocess_nni=False)
    # param = sp()

    nni_mode = args.param_path == 'nni'

    param = SP.SimpleParam.merge_args(list(default_param.keys()), args, param)

    seed_everything(param['seed'])
    device = torch.device(args.device if args.param_path != 'nni' else 'cuda')

    dataset = load_graph_dataset('datasets', 'PCQM4M')
    num_graphs = len(dataset)
    input_dim = dataset.num_features if dataset.num_features > 0 else 1

    # subsampling dataset for efficient training
    subset_path = f'pcqm4m_subset_{args.subset_size}.pt'
    if os.path.exists(subset_path):
        print(f'loading subset indices from {subset_path} ...')
        subset_indices = torch.load(subset_path)
    else:
        def idx2mask(indices: torch.LongTensor, num_graphs: int = num_graphs) -> torch.BoolTensor:
            num_graphs = indices.max().item() + 1 if num_graphs is None else num_graphs
            mask = torch.zeros((num_graphs,), dtype=torch.bool)
            mask[indices] = True
            return mask

        # compute masks
        train_idx = dataset.get_idx_split()['train']
        val_idx = dataset.get_idx_split()['valid']
        test_idx = dataset.get_idx_split()['test']
        train_mask, val_mask, test_mask = [idx2mask(idx) for idx in [train_idx, val_idx, test_idx]]

        # do the sampling
        labeled_mask = train_mask.logical_or(val_mask)
        labeled_indices = torch.nonzero(labeled_mask, as_tuple=False).view(-1)

        indices = torch.randperm(labeled_indices.shape[0])[:args.subset_size]
        subset_indices = labeled_indices[indices]
        torch.save(subset_indices, f'pcqm4m_subset_{args.subset_size}.pt')

    dataset = Subset(dataset, subset_indices.tolist())

    train_loader = DataLoader(dataset, batch_size=param['batch_size'])
    test_loader = DataLoader(dataset, batch_size=param['batch_size'])

    print(param)
    print(args.__dict__)

    def get_aug(aug_name: str, view_id: int):
        if aug_name == 'ER':
            return A.EdgeRemoving(pe=param[f'drop_edge_prob{view_id}'])
        if aug_name == 'EA':
            return A.EdgeAdding(pe=param[f'add_edge_prob{view_id}'])
        if aug_name == 'ND':
            return A.NodeDropping(pn=param[f'drop_node_prob{view_id}'])
        if aug_name == 'RWS':
            return A.RWSampling(num_seeds=param['num_seeds'], walk_length=param['walk_length'])
        if aug_name == 'PPR':
            return A.PPRDiffusion(eps=param['sp_eps'], use_cache=False)
        if aug_name == 'MKD':
            return A.MarkovDiffusion(sp_eps=param['sp_eps'], use_cache=False)
        if aug_name == 'ORI':
            return A.Identity()
        if aug_name == 'FM':
            return A.FeatureMasking(pf=param[f'drop_feat_prob{view_id}'])
        if aug_name == 'FD':
            return A.FeatureDropout(pf=param[f'drop_feat_prob{view_id}'])

        raise NotImplementedError(f'unsupported augmentation name: {aug_name}')

    def compile_aug_schema(schema: str, view_id: int) -> A.GraphAug:
        augs = schema.split('+')
        augs = [get_aug(x, view_id) for x in augs]

        ret = augs[0]
        for a in augs[1:]:
            ret = ret >> a
        return ret

    aug1 = compile_aug_schema(args.aug1, view_id=1)
    aug2 = compile_aug_schema(args.aug2, view_id=2)

    model = GRACE(encoder=Encoder(
                      input_dim, param['hidden_dim'],
                      activation=get_activation(param['activation']),
                      num_layers=param['num_layers']),
                  augmentation=(
                      aug1, aug2
                      # A.FeatureMasking(pf=param['drop_feat_prob1']) >> A.EdgeRemoving(pe=param['drop_edge_prob1']),
                      # A.FeatureMasking(pf=param['drop_feat_prob2']) >> A.EdgeRemoving(pe=param['drop_edge_prob2']),
                      # A.FeatureMasking(pf=param['drop_feat_prob2']) >> A.PPRDiffusion(eps=0.1),
                  ),
                  hidden_dim=param['hidden_dim'],
                  proj_dim=param['proj_dim'],
                  tau=param['tau']).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])

    best_loss = 1e10
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{args.aug1}-{args.aug2}.pkl'
    for epoch in range(param['num_epochs']):
        tic = perf_counter()
        loss = train(model, optimizer, train_loader, device=device, args=args, epoch=epoch, num_graphs=len(dataset))
        toc = perf_counter()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, time={toc - tic:.4f}')

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            wait_window = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            wait_window += 1

        if wait_window >= param['patience']:
            break

    print("=== Final ===")
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, test_loader, device, param['seed'], num_graphs=len(dataset), split=None)
    test_mae = test_result['mae']
    print(f'(E) | test mae={test_mae:.4f}')

    if nni_mode:
        nni.report_final_result(test_mae)


if __name__ == '__main__':
    main()