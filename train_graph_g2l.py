import numpy
import argparse
from time import perf_counter

import nni
import torch
import pretty_errors
import GCL.augmentors as A
import GCL.utils.simple_param as SP

from GCL.eval import SVM_classification
from GCL.utils import seed_everything
from utils import load_graph_dataset
from models.MVGRL import MVGRL
from torch_geometric.data import DataLoader
from torch_geometric import nn


def make_gin_conv(input_dim: int, out_dim: int) -> nn.GINConv:
    return nn.GINConv(torch.nn.Sequential(torch.nn.Linear(input_dim, out_dim), torch.nn.ReLU(), torch.nn.Linear(out_dim, out_dim)))


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, readout):
        super(GCN, self).__init__()
        self.activation = activation()
        self.readout = readout
        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weights, batch):
        z = x
        g = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            if self.readout == 'mean':
                g.append(nn.global_mean_pool(z, batch))
            elif self.readout == 'max':
                g.append(nn.global_max_pool(z, batch))
            elif self.readout == 'sum':
                g.append(nn.global_add_pool(z, batch))
        g = torch.cat(g, dim=1)
        return z, g


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


def train(model: MVGRL, optimizer, dataloader, device, batch_size: int):
    model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()
        z1, g1, z2, g2, _, _ = model(data.batch, data.x, data.edge_index, sample_size=None)

        if args.loss == 'nt_xent':
            loss = model.nt_xent_loss(z1, g1, z2, g2, data.batch, temperature=param['tau'])
        elif args.loss == 'triplet':
            loss = model.triplet_loss(z1, g1, z2, g2, data.batch, eps=1)
        else:
            loss = model.loss(z1, g1, z2, g2, data.batch)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss


def test(model: MVGRL, dataloader, device, seed):
    model.eval()
    x = []
    y = []

    for data in dataloader:
        data = data.to(device)
        z1, g1, z2, g2, _, _ = model(data.batch, data.x, data.edge_index)
        z = g1 + g2

        x.append(z.detach().cpu())
        y.append(data.y.cpu())

    x = torch.cat(x).numpy()
    y = torch.cat(y).numpy()

    res = SVM_classification(x, y, seed)
    return res


def main():
    hidden_dim = param['hidden_dim']
    num_GCN_layers = param['num_layers']
    num_MLP_layers = 3
    batch_size = param['batch_size']
    learning_rate = param['learning_rate']

    dataset = load_graph_dataset('datasets', args.dataset, to_sparse_tensor=False)

    if dataset.num_features != 0:
        input_dim = dataset.num_features
    else:
        input_dim = 1

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
            return A.PPRDiffusion(eps=param['sp_eps'], use_cache=True)
        if aug_name == 'MKD':
            return A.MarkovDiffusion(sp_eps=param['sp_eps'])
        if aug_name == 'ORI':
            return A.Identity()
        if aug_name == 'FM':
            return A.FeatureMasking(pf=param[f'drop_feat_prob{view_id}'])
        if aug_name == 'FD':
            return A.FeatureDropout(pf=param[f'drop_feat_prob{view_id}'])

        raise NotImplementedError(f'unsupported augmentation name: {aug_name}')

    def compile_aug_schema(schema: str, view_id: int) -> A.Augmentor:
        augs = schema.split('+')
        augs = [get_aug(x, view_id) for x in augs]

        ret = augs[0]
        for a in augs[1:]:
            ret = ret >> a
        return ret

    aug1 = compile_aug_schema(args.aug1, view_id=1)
    aug2 = compile_aug_schema(args.aug2, view_id=2)

    model = MVGRL(
        gnn1=GCN(input_dim, hidden_dim, num_GCN_layers, torch.nn.PReLU, 'mean'),
        gnn2=GCN(input_dim, hidden_dim, num_GCN_layers, torch.nn.PReLU, 'mean'),
        mlp1=MLP(hidden_dim, hidden_dim, num_MLP_layers, torch.nn.PReLU),
        mlp2=MLP(num_GCN_layers * hidden_dim, hidden_dim, num_MLP_layers, torch.nn.PReLU),
        sample_size=0,
        augmentations=(
            aug1, aug2
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset, batch_size=len(dataset))

    for i in range(param['num_epochs']):
        tic = perf_counter()
        epoch_loss = train(model, optimizer, train_loader, device, batch_size=2000)
        toc = perf_counter()
        print(f'(T) | Epoch={i:03d}, loss={epoch_loss:.4f}, time={toc - tic:.6f}')

        # if (i + 1) % 100 == 0:
        #     test_result = test(model, test_loader, device, seed=1234)
        #     print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    print("=== Final ===")
    test_result = test(model, test_loader, device, seed=1234)
    # test_result = test(model, data)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"][0]:.4f}, F1Ma={test_result["F1Ma"][0]:.4f}')

    if args.param_path == 'nni':
        nni.report_final_result(test_result['F1Mi'][0])


if __name__ == '__main__':
    default_param = {
        'seed': 39788,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'proj_dim': 256,
        'weight_decay': 1e-5,
        'activation': 'prelu',
        'batch_norm': False,
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_prob1': 0.2,
        'drop_edge_prob2': 0.1,
        'add_edge_prob1': 0.1,
        'add_edge_prob2': 0.1,
        'drop_node_prob1': 0.1,
        'drop_node_prob2': 0.1,
        'drop_feat_prob1': 0.3,
        'drop_feat_prob2': 0.2,
        'patience': 50,
        'num_epochs': 10,
        'batch_size': 10,
        'tau': 0.8,
        'sp_eps': 0.01,
        'num_seeds': 1000,
        'walk_length': 10,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='IMDB-MULTI')
    parser.add_argument('--param_path', type=str, default='None')
    parser.add_argument('--aug1', type=str, default='FM+ER')
    parser.add_argument('--aug2', type=str, default='FM+ER')
    parser.add_argument('--loss', type=str, default='triplet', choices=['nt_xent', 'jsd', 'triplet'])
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    for k, v in default_param.items():
        parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    param = sp(args.param_path, preprocess_nni=False)
    # param = sp()

    param = SP.SimpleParam.merge_args(list(default_param.keys()), args, param)

    seed_everything(param['seed'])
    device = torch.device(args.device)

    main()