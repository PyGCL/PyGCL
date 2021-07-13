import nni
import torch
import argparse
import pretty_errors
import GCL.utils.simple_param as SP

from tqdm import tqdm
from time import time_ns
from torch import nn
from torch.optim import Adam
from GCL.eval import LR_classification
from GCL.utils import seed_everything
from torch_geometric.nn import GCNConv
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


from utils import load_node_dataset, get_activation, get_compositional_augmentor, get_loss
from models.L2L import L2L


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


def train(model, optimizer, data, param):
    model.train()
    optimizer.zero_grad()
    _, z1, z2 = model(data.x, data.edge_index)
    h1 = model.projection(z1)
    h2 = model.projection(z2)
    loss = model.loss(h1, h2, **param)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, args, verbose=True):
    model.eval()
    z, _, _ = model(data.x, data.edge_index)

    test_result = LR_classification(
        z, data, split_mode='ogb' if args.dataset.startswith('ogb') else 'rand',
        train_ratio=0.1, test_ratio=0.8, verbose=verbose)
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
        'patience': 50,
        'num_epochs': 200
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param_path', type=str, default='params/GRACE/wikics@current.json')
    parser.add_argument('--tensorboard', nargs='?')
    for k, v in default_param.items():
        if type(v) is dict:
            for subk, subv in v.items():
                parser.add_argument(f'--{k}:{subk}', type=type(subv), nargs='?')
        else:
            parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    param = sp(args.param_path, preprocess_nni=False)

    use_nni = args.param_path == 'nni'

    seed_everything(param['seed'])
    device = torch.device(args.device if not use_nni else 'cuda')
    dataset = load_node_dataset('datasets', args.dataset, to_sparse_tensor=False)
    data = dataset[0].to(device)
    data.get_idx_split = lambda: dataset.get_idx_split()

    writer = SummaryWriter(comment=args.tensorboard) if args.tensorboard is not None else None

    print(param)
    print(args.__dict__)

    aug1 = get_compositional_augmentor(param['augmentor1'])
    aug2 = get_compositional_augmentor(param['augmentor2'])
    loss = get_loss(param['loss'], param[param['loss']])

    model = L2L(encoder=Encoder(data.num_features, param['hidden_dim'],
                                activation=get_activation(param['activation']),
                                batch_norm=param['batch_norm'],
                                num_layers=param['num_layers']),
                augmentor=(aug1, aug2),
                hidden_dim=param['hidden_dim'],
                proj_dim=param['proj_dim'],
                loss=loss).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=param['warmup_epochs'],
        max_epochs=param['num_epochs'])

    best_loss = 1e3
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{param["augmentor1"]["scheme"]}-{param["augmentor2"]["scheme"]}.pkl'

    with tqdm(total=param['num_epochs'], desc='(T)') as pbar:
        for epoch in range(param['num_epochs']):
            loss = train(model, optimizer, data, param[param['loss']])
            if param['loss'] == 'barlow_twins':
                scheduler.step()
            # else:
                # loss = train(model, optimizer, data)
            pbar.set_postfix({'loss': loss})
            pbar.update()

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
                print()
                break

    print('=== Final ===')
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, data, args)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'])

    return data, model


if __name__ == '__main__':
    data, model = main()
