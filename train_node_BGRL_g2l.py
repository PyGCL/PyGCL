import nni
import numpy
import math
import torch
import argparse
import pretty_errors
from time import time_ns
from time import perf_counter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# import GCL.augmentation as A
import GCL.augmentations as A
import GCL.augmentations.functional as AF
import GCL.utils.simple_param as SP

from torch import nn
from torch.optim import Adam
from GCL.eval import LR_classification
from GCL.utils import seed_everything
from torch_geometric.nn import GCNConv

from utils import load_node_dataset, get_activation
from models.GRACE_mod import GRACE
from models.BGRL import BGRL


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation,
                 num_layers: int, dropout: float = 0.2):
        super(Encoder, self).__init__()
        self.activation = activation()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.projection_head = torch.nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


def bgrl_loss(g: torch.FloatTensor, h: torch.FloatTensor):
    num_nodes = h.size()[0]  # M := num_nodes
    device = h.device

    # pos_mask = []
    # for i in range(num_graphs):
    #     mask = batch == i
    #     pos_mask.append(mask.to(torch.long))
    # pos_mask = torch.stack(pos_mask, dim=0).to(torch.float32)

    g = F.normalize(g, dim=-1, p=2)
    h = F.normalize(h, dim=-1, p=2)

    similarity = torch.unsqueeze(g, dim=0) @ h.t()
    return similarity.sum(dim=-1)


def train(model, optimizer, data, epoch, args):
    model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = model(data.x, data.edge_index)

    g1_target, g2_target = h1_target.sum(dim=0), h2_target.sum(dim=0)

    loss = bgrl_loss(g2_target.detach(), h1_pred)
    loss += bgrl_loss(g1_target.detach(), h2_pred)
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    model.update_target_encoder(momentum=0.99)

    return loss.item()


def test(model, data, args, verbose=True):
    model.eval()
    h1, h2, _, _, _, _ = model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)

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
        'patience': 5000,
        'num_epochs': 200,
        'tau': 0.8,
        'sp_eps': 0.001,
        'num_seeds': 1000,
        'walk_length': 10,
        'dropout': 0.2
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Coauthor-CS')
    parser.add_argument('--param_path', type=str, default='params/GRACE/coauthor_cs.json')
    parser.add_argument('--aug1', type=str, default='FM+ER')
    parser.add_argument('--aug2', type=str, default='FM+ER')
    parser.add_argument('--tensorboard', nargs='?')
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
            return A.PPRDiffusion(eps=param['sp_eps'])
        if aug_name == 'MKD':
            return A.MarkovDiffusion(sp_eps=param['sp_eps'])
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

    model = BGRL(encoder=Encoder(data.num_features, param['hidden_dim'],
                                 activation=get_activation(param['activation']),
                                 num_layers=param['num_layers'],
                                 dropout=param['dropout']),
                 augmentation=(aug1, aug2),
                 hidden_dim=param['hidden_dim'],
                 dropout=param['dropout']).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])

    best_loss = 1e3
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{args.aug1}-{args.aug2}.pkl'
    for epoch in range(param['num_epochs']):
        tic = perf_counter()
        loss = train(model, optimizer, data, epoch, args)
        toc = perf_counter()
        print(f'\r(T) | Epoch={epoch:03d}, loss={loss:.8f}, time={toc - tic:.4f}', end='')

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

    print('\n=== Final ===')
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, data, args)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'])

    return data, model


if __name__ == '__main__':
    data, model = main()
