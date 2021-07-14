import nni
import torch
import argparse
import pretty_errors
import GCL.utils.simple_param as SP

from tqdm import tqdm
from time import time_ns
from torch_geometric.data import DataLoader
from GCL.eval import LR_classification
from GCL.utils import seed_everything

from models.G2L import G2L, GCN, MLP
from utils import load_node_dataset, get_loss, get_compositional_augmentor


def train(model, optimizer, dataloader, device, param):
    model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()
        z1, g1, z2, g2, z3, z4 = model(data.batch, data.x, data.edge_index)
        loss = model.loss(z1, g1, z2, g2, z3, z4, **param)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss


def test(model, dataloader, device):
    model.eval()
    z = []

    for data in dataloader:
        data = data.to(device)
        z1, g1, z2, g2, _, _ = model(data.batch, data.x, data.edge_index)
        z.append(z1 + z2)
    z = torch.cat(z, dim=-1).to(z1.device)

    test_result = LR_classification(z, data, split_mode='rand', train_ratio=0.1, test_ratio=0.8, verbose=False)
    return test_result


def main():
    default_param = {
        'seed': 39788,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'proj_dim': 256,
        'weight_decay': 1e-5,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'patience': 50,
        'num_epochs': 200,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    parser.add_argument('--param_path', type=str, default='params/MVGRL/amazon_computers@current.json')
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

    seed_everything(param['seed'])
    device = torch.device(args.device)

    hidden_dim = param['hidden_dim']
    num_GCN_layers = param['num_layers']
    num_MLP_layers = 3
    batch_size = 32
    learning_rate = param['learning_rate']

    dataset = load_node_dataset('datasets', args.dataset, to_sparse_tensor=False)

    if dataset.num_features != 0:
        input_dim = dataset.num_features
    else:
        input_dim = 1

    aug1 = get_compositional_augmentor(param['augmentor1'])
    aug2 = get_compositional_augmentor(param['augmentor2'])
    loss = get_loss(param['loss'], 'G2LEN')

    model = G2L(
        gnn1=GCN(input_dim, hidden_dim, num_GCN_layers, torch.nn.PReLU, 'mean', base_conv='GCNConv'),
        gnn2=GCN(input_dim, hidden_dim, num_GCN_layers, torch.nn.PReLU, 'mean', base_conv='GCNConv'),
        mlp1=MLP(hidden_dim, hidden_dim, num_MLP_layers, torch.nn.PReLU),
        mlp2=MLP(num_GCN_layers * hidden_dim, hidden_dim, num_MLP_layers, torch.nn.PReLU),
        augmentor=(aug1, aug2),
        loss=loss
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset, batch_size=len(dataset))

    best_loss = 1e3
    wait_window = 0
    model_save_path = f'intermediate/{time_ns()}-{param["augmentor1"]["scheme"]}-{param["augmentor2"]["scheme"]}.pkl'

    with tqdm(total=param['num_epochs'], desc='(T)') as pbar:
        for epoch in range(param['num_epochs']):
            loss = train(model, optimizer, train_loader, device, param[param['loss']])
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
                print()
                break

    print("=== Final ===")
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss:.4f}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, test_loader, device)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if args.param_path == 'nni':
        nni.report_final_result(test_result['F1Mi'])


if __name__ == '__main__':
    main()
