from dataclasses import asdict
from time import time_ns
import torch

from tqdm import tqdm
from time import time_ns
from GCL.eval import LREvaluator, get_split
from GCL.utils import seed_everything
from GCL.models import EncoderModel, ContrastModel
from HC.config_loader import ConfigLoader
from torch_geometric.data import DataLoader

from utils import load_dataset, get_compositional_augmentor, get_activation, get_loss, is_node_dataset
from models.GConv import Encoder

from train_config import *


def train(encoder_model: EncoderModel, contrast_model: ContrastModel,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer, config: ExpConfig):
    encoder_model.train()
    epoch_loss = 0.0
    device = torch.device(config.device)

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size()[0]
            x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        else:
            x = data.x

        z, g, z1, z2, g1, g2, z3, z4 = encoder_model(x, data.batch, data.edge_index, data.edge_attr)
        h1, h2, h3, h4 = [encoder_model.projection(x) for x in [z1, z2, z3, z4]]

        loss = contrast_model(h1, h2, g1, g2, data.batch, h3, h4)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss


def evaluate(encoder_model: EncoderModel, test_loader: DataLoader, dataset, config: ExpConfig):
    encoder_model.eval()

    x = []
    y = []
    for data in test_loader:
        data = data.to(config.device)

        if data.x is None:
            num_nodes = data.batch.size()[0]
            input_x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        else:
            input_x = data.x

        z, g, z1, z2, g1, g2, z3, z4 = encoder_model(input_x, data.batch, data.edge_index, data.edge_attr)
        x.append(z if is_node_dataset(config.dataset) else g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(name=config.dataset, num_samples=x.size()[0], dataset=dataset)
    evaluator = LREvaluator()
    result = evaluator.evaluate(x, y, split)

    return result


def main(config: ExpConfig):
    seed_everything(config.seed)
    device = torch.device(config.device)

    dataset = load_dataset('datasets', config.dataset, to_sparse_tensor=False)
    train_loader = DataLoader(dataset, batch_size=config.opt.batch_size)
    test_loader = DataLoader(dataset, batch_size=config.opt.batch_size, shuffle=False)

    input_dim = 1 if dataset.num_features == 0 else dataset.num_features

    aug1 = get_compositional_augmentor(asdict(config.augmentor1))
    aug2 = get_compositional_augmentor(asdict(config.augmentor2))

    encoder = Encoder(
        input_dim=input_dim,
        hidden_dim=config.encoder.hidden_dim,
        activation=get_activation(config.encoder.activation.value),
        num_layers=config.encoder.num_layers,
        base_conv=config.encoder.conv.value
    ).to(device)

    loss_name = config.obj.loss.value
    loss_params = asdict(config.obj)[loss_name]
    encoder_model = EncoderModel(
        encoder=encoder,
        augmentor=(aug1, aug2),
        hidden_dim=config.encoder.hidden_dim,
        proj_dim=config.encoder.proj_dim
    ).to(device)
    contrast_model = ContrastModel(
        loss=get_loss(loss_name, **loss_params),
        mode=config.mode.value
    )

    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=config.opt.learning_rate)

    model_path = f'intermediate/{time_ns()}.pkl'
    best_loss = 1e20
    best_epoch = 0
    wait_window = 0
    with tqdm(total=config.opt.num_epochs, desc='(T)') as pbar:
        for epoch in range(1, config.opt.num_epochs + 1):
            loss = train(encoder_model, contrast_model, train_loader, optimizer, config)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                wait_window = 0
                torch.save(encoder_model.state_dict(), model_path)
            else:
                wait_window += 1

            if wait_window > config.opt.patience:
                break

    print("=== Final ===")
    print(f'(T): Best epoch={best_epoch}, best loss={best_loss:.4f}')
    saved_model = torch.load(model_path)
    encoder_model.load_state_dict(saved_model)
    test_result = evaluate(encoder_model, test_loader, dataset, config)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    return {
        'evaluation': test_result
    }


if __name__ == '__main__':
    loader = ConfigLoader(model=ExpConfig, config='params/GRACE/imdb_multi@ng.json')
    config = loader()

    main(config)