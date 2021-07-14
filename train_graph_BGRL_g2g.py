import warnings

import nni
import math
import torch
import argparse
import pretty_errors
from tqdm import tqdm

import GCL.utils.simple_param as SP

from time import time_ns
from torch.optim import Adam
from GCL.eval import SVM_classification
from GCL.utils import seed_everything
from sklearn.exceptions import ConvergenceWarning
from torch_geometric.data import DataLoader

from utils import get_activation, load_graph_dataset, get_compositional_augmentor
from models.BGRL import BGRL, GraphEncoder


def train(model, optimizer, loader, device, param):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        _, _, g1_pred, g2_pred, g1_target, g2_target = model(data.x, data.edge_index, batch=data.batch)

        loss = model.loss(g1_pred, g2_pred, g1_target.detach(), g2_target.detach())
        loss.backward()
        optimizer.step()
        model.update_target_encoder(momentum=param['momentum'])

        total_loss += loss.item()

    return total_loss


def test(model, loader, device, seed):
    model.eval()
    x = []
    y = []
    for data in loader:
        data = data.to(device)
        if data.x is None:
            data.x = torch.ones((data.batch.size(0), 1), dtype=torch.float32).to(device)
        g1, g2, _, _, _, _ = model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)

        x.append(z.detach().cpu())
        y.append(data.y.cpu())

    x = torch.cat(x, dim=0).numpy()
    y = torch.cat(y, dim=0).numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        res = SVM_classification(x, y, seed)

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
        'patience': 100,
        'num_epochs': 1000,
        'batch_size': 32,
        'dropout': 0.2
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='IMDB-MULTI')
    parser.add_argument('--param_path', type=str, default='params/BGRL/imdb_multi.json')
    for k, v in default_param.items():
        if type(v) is dict:
            for subk, subv in v.items():
                parser.add_argument(f'--{k}:{subk}', type=type(subv), nargs='?')
        else:
            parser.add_argument(f'--{k}', type=type(v), nargs='?')
    args = parser.parse_args()
    sp = SP.SimpleParam(default=default_param)
    sp.update(args.param_path, preprocess_nni=False)
    overwrite_params = {k: v for k, v in args.__dict__.items() if v is not None}
    sp.load(overwrite_params)
    param = sp()

    use_nni = args.param_path == 'nni'

    seed_everything(param['seed'])
    device = torch.device(args.device if not use_nni else 'cuda')
    dataset = load_graph_dataset('datasets', args.dataset)
    input_dim = dataset.num_features if dataset.num_features > 0 else 1
    train_loader = DataLoader(dataset, batch_size=param['batch_size'])
    test_loader = DataLoader(dataset, batch_size=math.ceil(len(dataset) / 2))

    print(param)
    print(args.__dict__)

    aug1 = get_compositional_augmentor(param['augmentor1'])
    aug2 = get_compositional_augmentor(param['augmentor2'])

    model = BGRL(encoder=GraphEncoder(input_dim, param['hidden_dim'],
                                      activation=get_activation(param['activation']),
                                      num_layers=param['num_layers'],
                                      dropout=param['dropout'],
                                      encoder_norm=param['bootstrap']['encoder_norm'],
                                      projector_norm=param['bootstrap']['projector_norm']),
                 augmentor=(aug1, aug2),
                 hidden_dim=param['hidden_dim'],
                 dropout=param['dropout'],
                 predictor_norm=param['bootstrap']['predictor_norm'],
                 mode='G2G').to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])

    best_loss = 1e3
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{param["augmentor1"]["scheme"]}-{param["augmentor2"]["scheme"]}.pkl'

    with tqdm(total=param['num_epochs'], desc='(T)') as pbar:
        for epoch in range(param['num_epochs']):
            loss = train(model, optimizer, train_loader, device=device, param=param['bootstrap'])
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

    print('\n=== Final ===')
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss:.4f}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, test_loader, device, param['seed'])
    print(f'(E) | Best test F1Mi={test_result["F1Mi"][0]:.4f}, F1Ma={test_result["F1Ma"][0]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'][0])


if __name__ == '__main__':
    main()
