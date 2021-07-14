import nni
import math
import torch
import argparse
import pretty_errors

import GCL.utils.simple_param as SP
from GCL.eval import SVM_classification

from tqdm import tqdm
from time import time_ns
from torch.optim import Adam
from GCL.utils import seed_everything
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

from models.GConv import Encoder
from utils import load_graph_dataset, get_activation, get_compositional_augmentor, get_loss
from models.G2G import G2G


def train(model, optimizer, loader: DataLoader, device, param):
    model.train()
    tot_loss = 0.0
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        optimizer.zero_grad()
        _, _, _, g1, g2 = model(data.x, edge_index=data.edge_index, batch=data.batch)
        loss = model.loss(g1, g2, **param)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    return tot_loss


def test(model, loader, device, seed):
    model.eval()
    x = []
    y = []
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        z, _, _, g1, g2 = model(data.x, edge_index=data.edge_index, batch=data.batch)
        g = global_add_pool(z, data.batch)

        x.append(g.detach().cpu())
        y.append(data.y.cpu())

    x = torch.cat(x, dim=0).numpy()
    y = torch.cat(y, dim=0).numpy()

    res = SVM_classification(x, y, seed)
    # print(f'(E) | Accuracy: {accuracy[0]:.4f} +- {accuracy[1]:.4f}')

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
        'batch_norm': False,
        'batch_size': 10,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--param_path', type=str, default='params/GlobalGRACE/proteins@current.json')
    for k, v in default_param.items():
        if type(v) is dict:
            for subk, subv in v.items():
                parser.add_argument(f'--{k}:{subk}', type=type(subv), nargs='?')
        else:
            parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    sp.update(args.param_path, preprocess_nni=False)
    param = sp()

    nni_mode = args.param_path == 'nni'

    seed_everything(param['seed'])
    device = torch.device(args.device if args.param_path != 'nni' else 'cuda')

    dataset = load_graph_dataset('datasets', args.dataset)
    input_dim = dataset.num_features if dataset.num_features > 0 else 1
    train_loader = DataLoader(dataset, batch_size=param['batch_size'])
    test_loader = DataLoader(dataset, batch_size=math.ceil(len(dataset) / 2))

    print(param)
    print(args.__dict__)

    aug1 = get_compositional_augmentor(param['augmentor1'])
    aug2 = get_compositional_augmentor(param['augmentor2'])
    loss = get_loss(param['loss'], 'L2L')

    model = G2G(encoder=Encoder(input_dim, param['hidden_dim'],
                                activation=get_activation(param['activation']),
                                num_layers=param['num_layers'],
                                batch_norm=param['batch_norm'],
                                base_conv='GINConv'),
                augmentor=(aug1, aug2),
                hidden_dim=param['hidden_dim'],
                proj_dim=param['proj_dim'],
                loss=loss).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])
    if param['loss'] == 'barlow_twins':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=param['warmup_epochs'],
            max_epochs=param['num_epochs'])

    best_loss = 1e10
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{param["augmentor1"]["scheme"]}-{param["augmentor2"]["scheme"]}.pkl'
    with tqdm(total=param['num_epochs'], desc='(T)') as pbar:
        for epoch in range(param['num_epochs']):
            loss = train(model, optimizer, train_loader, device=device, param=param[param['loss']])
            if param['loss'] == 'barlow_twins':
                scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

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
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss:.4f}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, test_loader, device, param['seed'])
    print(f'(E) | Best test F1Mi={test_result["F1Mi"][0]:.4f}, F1Ma={test_result["F1Ma"][0]:.4f}')

    if nni_mode:
        nni.report_final_result(test_result["F1Mi"][0])


if __name__ == '__main__':
    main()
