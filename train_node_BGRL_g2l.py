import nni
import torch
import argparse
import pretty_errors
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import GCL.utils.simple_param as SP

from time import time_ns
from torch.optim import Adam
from GCL.eval import LR_classification
from GCL.losses import BootstrapLossG2L
from GCL.utils import seed_everything

from utils import load_node_dataset, get_activation, get_compositional_augmentor
from models.BGRL import BGRL, Encoder


def train(model, optimizer, data, param):
    model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = model(data.x, data.edge_index)
    g1_target, g2_target = h1_target.sum(dim=0), h2_target.sum(dim=0)
    loss = model.loss(h1_pred, h2_pred, g1_target.detach(), g2_target.detach())
    loss.backward()
    optimizer.step()
    model.update_target_encoder(momentum=param['momentum'])

    return loss.item()


def test(model, data, args):
    model.eval()
    h1, h2, _, _, _, _ = model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)

    test_result = LR_classification(
        z, data, split_mode='ogb' if args.dataset.startswith('ogb') else 'rand',
        train_ratio=0.1, test_ratio=0.8)
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
        'patience': 5000,
        'num_epochs': 200,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param_path', type=str, default='params/BGRL/wikics.json')
    parser.add_argument('--tensorboard', nargs='?')
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

    model = BGRL(encoder=Encoder(data.num_features, param['hidden_dim'],
                                 activation=get_activation(param['activation']),
                                 num_layers=param['num_layers'],
                                 dropout=param['dropout'],
                                 encoder_norm=param['bootstrap']['encoder_norm'],
                                 projector_norm=param['bootstrap']['projector_norm'],
                                 base_conv='GCNConv'),
                 augmentor=(aug1, aug2),
                 hidden_dim=param['hidden_dim'],
                 dropout=param['dropout'],
                 loss=BootstrapLossG2L()).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'])

    best_loss = 1e3
    wait_window = 0

    model_save_path = f'intermediate/{time_ns()}-{param["augmentor1"]["scheme"]}-{param["augmentor2"]["scheme"]}.pkl'
    with tqdm(total=param['num_epochs'], desc='(T)') as pbar:
        for epoch in range(param['num_epochs']):
            loss = train(model, optimizer, data, param['bootstrap'])
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
                    test_result = test(model, data, args)
                    writer.add_scalar('Eval/MicroF1', test_result['F1Mi'], epoch)
                    writer.add_scalar('Eval/MacroF1', test_result['F1Ma'], epoch)
                    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

                writer.flush()

            if wait_window >= param['patience']:
                break

    print('\n=== Final ===')
    print(f'(T) | Best epoch={best_epoch}, best loss={best_loss:.4f}')
    model.load_state_dict(torch.load(model_save_path))

    test_result = test(model, data, args)
    print(f'(E) | Best test F1Mi={test_result["F1Mi"]:.4f}, F1Ma={test_result["F1Ma"]:.4f}')

    if use_nni:
        nni.report_final_result(test_result['F1Mi'])

    return data, model


if __name__ == '__main__':
    data, model = main()
