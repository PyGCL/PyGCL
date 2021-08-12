import torch
import numpy as np

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


def nparray(shape=(100, 200), dtype=np.float32, min=0.0, max=2.0):
    return arrays(dtype=dtype, shape=shape, elements=st.floats(min, max, width=32))


def tensor(shape=(100, 200), dtype=np.float32, min=0.019999999552965164, max=2.0):
    return nparray(shape, dtype, min, max).map(lambda x: torch.from_numpy(x))


def edge_index(num_nodes, num_edges):
    return arrays(dtype=np.long, shape=(2, num_edges), elements=st.integers(0, num_nodes - 1)).map(lambda x: torch.from_numpy(x))
