import dgl
import torch
from dgl import DGLGraph
from torch_geometric.data import Data as PyGGraph


def from_dglgraph_to_pyggraph(pyggraph: PyGGraph) -> DGLGraph:
    raise NotImplementedError


def from_pyggraph_to_dglgraph(dglgraph: DGLGraph) -> PyGGraph:
    raise NotImplementedError
