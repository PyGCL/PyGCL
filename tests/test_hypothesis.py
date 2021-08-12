import torch
import numpy as np

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


@given(a=arrays(dtype=np.float32, shape=(10, 10), elements=st.floats(0, 1, width=32)),
       b=arrays(dtype=np.float32, shape=(10, 10), elements=st.floats(0, 1, width=32)))
def test_add_congruence(a, b):
    o1 = torch.from_numpy(a + b)
    o2 = torch.from_numpy(a) + torch.from_numpy(b)

    assert (o1 == o2).sum().item() == o1.numel()
