import torch
import numpy as np

from typing import List, Dict, Union, Iterator
from torch_geometric.data import Data
from sklearn.model_selection import BaseCrossValidator


def random_split(
        num_samples: int, num_splits: int = 1,
        train_ratio: float = 0.1, test_ratio: float = 0.8) -> List[Dict]:
    """
    Generate split indices for training, test, and validation sets.

    Args:
        num_samples (int): The size of the dataset.
        num_splits (int, optional): The number of splits to generate. (default: :obj:`1`)
        train_ratio (float, optional): The ratio of the training set. (default: :obj:`0.1`)
        test_ratio (float, optional): The ratio of the test set. (default: :obj:`0.8`)

    Returns:
        List[Dict]: A list of dictionaries of split indices.

    Examples:
        >>> random_split(10, num_splits=1, train_ratio=0.5, test_ratio=0.4)
        [{'train': [3, 4, 0, 1, 2], 'test': [5, 7, 6, 8], 'valid': [9]}]
    """
    assert train_ratio + test_ratio < 1

    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    out = []
    for i in range(num_splits):
        indices = torch.randperm(num_samples)
        out.append({
            'train': indices[:train_size],
            'valid': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        })
    return out


def from_PyG_split(data: Data) -> Union[Dict, List[Dict]]:
    """
    Convert from PyG split indices of training, test, and validation sets.

    Args:
        data (Data): A PyG data object.

    Returns:
        Union[Dict, List[Dict]]: A dictionary of split indices or a list of dictionaries of split indices.

    Raises:
        ValueError: If the :obj:`data` object does not have the split indices.
    """
    if any([mask is None for mask in [data.train_mask, data.test_mask, data.val_mask]]):
        raise ValueError('The data object does not have the split indices.')
    num_samples = data.num_nodes
    indices = torch.arange(num_samples)

    if data.train_mask.dim() == 1:
        return [{
            'train': indices[data.train_mask],
            'valid': indices[data.val_mask],
            'test': indices[data.test_mask]
        }]
    else:
        out = []
        for i in range(data.train_mask.size(1)):
            out_dict = {}
            for mask in ['train_mask', 'val_mask', 'test_mask']:
                if data[mask].dim() == 1:
                    # Datasets like WikiCS have only one split for the test set.
                    out_dict[mask[:-5]] = indices[data[mask]]
                else:
                    out_dict[mask[:-5]] = indices[data[mask][:, i]]
            out.append(out_dict)
        return out


def iter_split(
        split: Union[List[Dict], BaseCrossValidator],
        x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> Iterator[Dict]:
    """
    Iterate through multiple folds of the splits given the dataset.

    Args:
        split (Union[List[Dict], BaseCrossValidator]): A list of dictionaries of split indices
            or a sklearn cross-validator.
        x (Union[torch.Tensor, np.ndarray]): The data.
        y (Union[torch.Tensor, np.ndarray]): The targets (labels).

    Returns:
        Iterator[Dict]: An iterator of a dictionary with training, test, and validation sets.
    """
    if isinstance(split, list):
        for i in range(len(split)):
            yield {
                'x_train': x[split[i]['train']],
                'x_test': x[split[i]['test']],
                'x_valid': x[split[i]['valid']],
                'y_train': y[split[i]['train']],
                'y_test': y[split[i]['test']],
                'y_valid': y[split[i]['valid']]
            }
    elif isinstance(split, BaseCrossValidator):
        for train_idx, test_idx in split.split(x, y):
            yield {
                'x_train': x[train_idx],
                'x_test': x[test_idx],
                'x_valid': None,
                'y_train': y[train_idx],
                'y_test': y[test_idx],
                'y_valid': None
            }
    else:
        raise ValueError('The split object is not supported.')
