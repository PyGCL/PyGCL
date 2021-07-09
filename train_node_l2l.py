import nni
import numpy
import math
import torch
import argparse
import pretty_errors
from time import time_ns
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter

import GCL.augmentations as A
import GCL.utils.simple_param as SP

from torch import nn
from torch.optim import Adam
from GCL.eval import LR_classification
from GCL.utils import seed_everything
from torch_geometric.nn import GCNConv

from utils import load_node_dataset, get_activation
from models.GRACE import GRACE


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation, num_layers: int, batch_norm: bool = False):
        super(Encoder, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if batch_norm else None
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))

        for _ in range(num_layers - 1):
            # add batch norm layer if batch norm is used
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        num_layers = len(self.layers)
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            if self.batch_norms is not None and i != num_layers - 1:
                z = self.batch_norms[i](z)
        return z


def train(model, optimizer, data, epoch, args, param, dump_embed_path=None, loss='nt_xent'):
    model.train()
    device = data.edge_index.device
    if loss == 'batch':
        sample_size = args.sample_size
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes).to(device)
        num_batches = math.ceil(num_nodes / sample_size)
        total_loss = 0.0
        for i in range(num_batches):
            batch_indices = indices[i * sample_size:(i + 1) * sample_size]
            optimizer.zero_grad()
            _, z1, z2 = model(data.x, data.edge_index)
            z1 = z1[batch_indices]
            z2 = z2[batch_indices]
            loss = model.loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_indices.size(0)
        return total_loss / data.num_nodes
    else:
        optimizer.zero_grad()
        _, z1, z2 = model(data.x, data.edge_index)
        # if dump_embed_path is not None:
        #     torch.save((z1, z2), dump_embed_path)
        if loss == 'nt_xent':
            loss = model.loss(z1, z2, batch_size=512)
        elif loss == 'subsampling':
            loss = model.subsampling_loss(z1, z2, sample_size=args.sample_size)
        elif loss == 'jsd':
            loss = model.jsd_loss(z1, z2)
        elif loss == 'triplet':
            loss = model.triplet_loss(z1, z2)
        elif loss == 'barlow_twins':
            loss = model.bt_loss(z1, z2)
        elif loss == 'vicreg':
            loss = model.vicreg_loss(z1, z2)
        elif loss == 'mixup':
            loss = model.hard_mixing_loss(z1, z2, threshold=param['mixup_threshold'], s=param['mixup_s'])
        else:
            raise NotImplementedError(f'Unknown loss type: {loss}')
        loss.backward()
        optimizer.step()
        return loss.item()


def test(model, data, args, verbose=True):
    model.eval()
    z, _, _ = model(data.x, data.edge_index)

    test_result = LR_classification(
        z, data, split_mode='ogb' if args.dataset.startswith('ogb') else 'rand', train_ratio=0.1, test_ratio=0.8, verbose=verbose)
    return test_result


def main():
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
        'num_epochs': 200,
        'tau': 0.8,
        'sp_eps': 0.001,
        'ppr_alpha': 0.2,
        'mkd_order': 16,
        'mkd_alpha': 0.05,
        'num_seeds': 1000,
        'walk_length': 10,
        'mixup_threshold': 0.1,
        'mixup_s': 200
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param_path', type=str, default='params/GRACE/wikics.json')
    parser.add_argument('--aug1', type=str, default='ORI')
    parser.add_argument('--aug2', type=str, default='ORI')
    parser.add_argument('--tensorboard', nargs='?')
    parser.add_argument('--loss', type=str,
                        choices=['nt_xent', 'jsd', 'triplet', 'mixup', 'subsampling', 'batch', 'barlow_twins', 'vicreg'],
                        default='vicreg')
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    for k, v in default_param.items():
        parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    param = sp(args.param_path, preprocess_nni=False)
    # param = sp()

    param = SP.SimpleParam.merge_args(list(default_param.keys()), args, param)

    use_nni = args.param_path == 'nni'

    seed_everything(param['seed'])
    device = torch.device(args.device if not use_nni else 'cuda')
    dataset = load_node_dataset('datasets', args.dataset, to_sparse_tensor=False)
    data = dataset[0].to(device)
    data.get_idx_split = lambda: dataset.get_idx_split()

    writer = SummaryWriter(comment=args.tensorboard) if args.tensorboard is not None else None

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
            return A.PPRDiffusion(eps=param['sp_eps'], alpha=param['ppr_alpha'])
        if aug_name == 'MKD':
            return A.MarkovDiffusion(sp_eps=param['sp_eps'], order=param['mkd_order'], alpha=param['mkd_alpha'])
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
                      data.num_features, param['hidden_dim'],
                      activation=get_activation(param['activation']),
                      batch_norm=param['batch_norm'],
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

    best_loss = 1e3
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{args.aug1}-{args.aug2}.pkl'
    for epoch in range(param['num_epochs']):
        # if epoch % 20 == 0:
        tic = perf_counter()
        loss = train(model, optimizer, data, epoch, args, param, f'intermediate/{epoch}_{args.dataset}_ring.pkl', loss=args.loss)
        toc = perf_counter()
        # else:
            # loss = train(model, optimizer, data)
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.8f}, time={toc - tic:.4f}')

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            wait_window = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            wait_window += 1

        if writer is not None:
            # log training loss
            writer.add_scalar('Train/loss', loss, epoch)

            # log evaluation metrics
            if epoch % 10 == 0:
                test_result = test(model, data, args, verbose=False)
                writer.add_scalar('Eval/MicroF1', test_result['F1Mi'], epoch)
                writer.add_scalar('Eval/MacroF1', test_result['F1Ma'], epoch)
                print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

            writer.flush()

        if wait_window >= param['patience']:
            break

    print("=== Final ===")
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, data, args)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'])

    return data, model


if __name__ == '__main__':
    data, model = main()
