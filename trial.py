from typing import *
import os
from pprint import PrettyPrinter
from dataclasses import asdict
from time import time_ns
import torch
from visualdl import LogWriter

from tqdm import tqdm
from time import time_ns
from GCL.eval import LREvaluator, get_split
from GCL.utils import seed_everything, batchify_dict
from GCL.models import EncoderModel, ContrastModel
from HC.config_loader import ConfigLoader
from torch_geometric.data import DataLoader

from utils import load_dataset, get_compositional_augmentor, get_activation, get_loss, is_node_dataset, get_augmentor
from models.GConv import Encoder

from train_config import *


class GCLTrial(object):
    def __init__(self, config: ExpConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.writer = LogWriter(logdir=f'./log/{config.visualdl}/train')
        self.dataset = load_dataset('datasets', config.dataset, to_sparse_tensor=False)
        self.train_loader = DataLoader(self.dataset, batch_size=config.opt.batch_size)
        self.test_loader = DataLoader(self.dataset, batch_size=config.opt.batch_size, shuffle=False)

        input_dim = 1 if self.dataset.num_features == 0 else self.dataset.num_features

        def augmentor_from_conf(conf: AugmentorConfig):
            scheme = conf.scheme.split('+')

            augs = [get_augmentor(aug_name, asdict(conf)[aug_name]) for aug_name in scheme]

            aug = augs[0]
            for a in augs[1:]:
                aug = aug >> a
            return aug

        aug1 = augmentor_from_conf(config.augmentor1)
        aug2 = augmentor_from_conf(config.augmentor2)

        encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=config.encoder.hidden_dim,
            activation=get_activation(config.encoder.activation.value),
            num_layers=config.encoder.num_layers,
            base_conv=config.encoder.conv.value
        ).to(self.device)
        self.encoder = encoder

        loss_name = config.obj.loss.value
        loss_params = asdict(config.obj)[loss_name]
        encoder_model = EncoderModel(
            encoder=encoder,
            augmentor=(aug1, aug2),
            hidden_dim=config.encoder.hidden_dim,
            proj_dim=config.encoder.proj_dim
        ).to(self.device)
        self.encoder_model = encoder_model
        contrast_model = ContrastModel(
            loss=get_loss(loss_name, **loss_params),
            mode=config.mode.value
        )
        self.contrast_model = contrast_model

        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=config.opt.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.opt.reduce_lr_patience)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.model_save_path = f'intermediate/{time_ns()}.pkl'
        self.best_loss = 1e20
        self.best_epoch = -1
        self.wait_window = 0
        self.trained = False
        self.train_step_cbs = []

    def register_train_step_callback(self, cb: Callable[[dict], None]):
        self.train_step_cbs.append(cb)

    def train_step(self):
        self.encoder_model.train()
        epoch_losses = []

        for data in self.train_loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()

            if data.x is None:
                num_nodes = data.batch.size()[0]
                x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            else:
                x = data.x

            z, g, z1, z2, g1, g2, z3, z4 = self.encoder_model(x, data.batch, data.edge_index, data.edge_attr)
            h1, h2, h3, h4 = [self.encoder_model.projection(x) for x in [z1, z2, z3, z4]]

            loss = self.contrast_model(h1, h2, g1, g2, data.batch, h3, h4)

            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        return sum(epoch_losses) / len(epoch_losses)

    def evaluate(self):
        self.encoder_model.eval()

        x = []
        y = []
        for data in self.test_loader:
            data = data.to(self.config.device)

            if data.x is None:
                num_nodes = data.batch.size()[0]
                input_x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            else:
                input_x = data.x

            z, g, z1, z2, g1, g2, z3, z4 = self.encoder_model(input_x, data.batch, data.edge_index, data.edge_attr)
            x.append(z if is_node_dataset(self.config.dataset) else g)
            y.append(data.y)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)

        split = get_split(name=self.config.dataset, num_samples=x.size()[0], dataset=self.dataset)
        if isinstance(split, list):
            results = []
            for sp in split:
                evaluator = LREvaluator()
                result = evaluator.evaluate(x, y, sp)
                results.append(result)
            result = batchify_dict(results, aggr_func=lambda xs: sum(xs) / len(xs))
        else:
            evaluator = LREvaluator()
            result = evaluator.evaluate(x, y, split)

        return result

    def run_train_loop(self):
        if self.trained:
            return

        with tqdm(total=self.config.opt.num_epochs, desc='(T)') as pbar:
            for epoch in range(1, self.config.opt.num_epochs + 1):
                loss = self.train_step()
                self.lr_scheduler.step(loss)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'wait': self.wait_window, 'lr': self.optimizer_lr})
                pbar.update()

                for cb in self.train_step_cbs:
                    cb({'loss': loss})

                if self.writer is not None:
                    self.writer.add_scalar('loss', step=epoch, value=loss)
                    self.writer.add_scalar('lr', step=epoch, value=self.optimizer_lr)

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.wait_window = 0
                    self.save_checkpoint()
                else:
                    self.wait_window += 1

                if self.wait_window > self.config.opt.patience:
                    break

        self.trained = True
        if self.writer is not None:
            self.writer.close()

    def save_checkpoint(self):
        torch.save(self.encoder_model.state_dict(), self.model_save_path)

    def load_checkpoint(self):
        saved_state = torch.load(self.model_save_path)
        self.encoder_model.load_state_dict(saved_state)

    @property
    def optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _prepare_env(self):
        seed_everything(self.config.seed)
        os.makedirs('intermediate', exist_ok=True)

    def execute(self):
        self._prepare_env()
        self.run_train_loop()
        self.load_checkpoint()
        result = self.evaluate()
        return result


if __name__ == '__main__':
    import pretty_errors  # noqa
    loader = ConfigLoader(model=ExpConfig, config='params/GRACE/general.json')
    config = loader()

    printer = PrettyPrinter(indent=2)
    printer.pprint(asdict(config))

    trial = GCLTrial(config)
    result = trial.execute()

    print("=== Final ===")
    print(f'(T): Best epoch={trial.best_epoch}, best loss={trial.best_loss:.4f}')
    print(f'(E): {result}')
