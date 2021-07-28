import torch

from GCL.utils import seed_everything
from utils import load_dataset, get_compositional_augmentor
from happy_config.config_loader import ConfigLoader

from train_config import *


def main(config: ExpConfig):
    seed_everything(config.seed)
    dataset = load_dataset('datasets', config.dataset, to_sparse_tensor=False)
    input_dim = 1 if dataset.num_features == 0 else dataset.num_features

    aug1 = get_compositional_augmentor(config.augmentor1.asdict())
    aug2 = get_compositional_augmentor(config.augmentor2.asdict())


if __name__ == '__main__':
    loader = ConfigLoader(model=ExpConfig, default_param_path='params/GRACE/wikics@bad.json')
    config = loader()

    # main(config)
