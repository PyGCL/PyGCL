# PyGCL: Graph Contrastive Learning for PyTorch

PyGCL is an open-source library for graph contrastive learning (GCL), which features modularized GCL components from published papers, standardized evaluation, and experiment management. 

[![Made with Python](https://img.shields.io/github/languages/top/GraphCL/PyGCL?color=lightgrey)](https://github.com/GraphCL/PyGCL)
[![GitHub stars](https://img.shields.io/github/stars/GraphCL/PyGCL?color=yellow)](https://github.com/GraphCL/PyGCL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GraphCL/PyGCL?color=brightgreen&label=fork)](https://github.com/GraphCL/PyGCL/network/members)
[![Total lines](https://img.shields.io/tokei/lines/github/GraphCL/PyGCL?color=red)](https://github.com/GraphCL/PyGCL)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=GraphCL.PyGCL)](https://github.com/GraphCL/PyGCL)

---

# What is Graph Contrastive Learning?

Graph Contrastive Learning (GCL) establishes a new paradigm for learning graph representations without human annotations. A typical GCL algorithm firstly constructs multiple graph views via stochastic augmentation of the input and then learns representations by contrasting positive samples against negative ones.

👉 For a general introduction of GCL, please refer to our [paper] and [blog](https://sxkdz.github.io/research/GraphCL/). Also, this [repo](https://github.com/SXKDZ/awesome-self-supervised-learning-for-graphs) tracks newly published GCL papers.

# Prerequisites

PyGCL needs the following packages to be installed beforehand:

* Python 3.8+
* PyTorch 1.8+
* PyTorch-Geometric 1.7
* DGL 0.7+
* Scikit-learn 0.24+
* Numpy
* tqdm
* NetworkX

# Getting Started

## Package Overview

Our PyGCL implements four main components of graph contrastive learning algorithms:

* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting modes: specifies positive and negative pairs.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

We also implement utilities for training models, evaluating model performance, and managing experiments.

## Building Your Own GCL Algorithms

Besides try the above examples for node and graph classification tasks, you can also build your own graph contrastive learning algorithms straightforwardly.

### Graph Augmentation

In `GCL.augmentors`, PyGCL provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, PyGCL implements the following augmentation functions:

| Augmentation                             | Class name        |
| ---------------------------------------- | ----------------- |
| Edge Adding (EA)                         | `EdgeAdding`      |
| Edge Removing (ER)                       | `EdgeRemoving`    |
| Feature Masking (FM)                     | `FeatureMasking`  |
| Feature Dropout (FD)                     | `FeatureDropout`  |
| Personalized PageRank (PPR)              | `PPRDiffusion`    |
| Markov Diffusion Kernel (MDK)            | `MarkovDiffusion` |
| Node Dropping (ND)                       | `NodeDropping`    |
| Subgraphs induced by Random Walks (RWS)  | `RWSampling`      |
| Ego-net Sampling (ES)                    | `Identity`        |

Call these augmentation functions by feeding with a graph of in a tuple form of node features, edge index, and edge features `x, edge_index, edge_weights`will produce corresponding augmented graphs.

PyGCL also supports composing arbitrary number of augmentations together. To compose a list of augmentation instances `augmentors`, you only need to use the right shift operator `>>`:

```python
aug = augmentors[0]
for a in augs[1:]:
    aug = aug >> a
```

You can also write your own augmentation functions by defining the `augment` function.

### Contrasting Modes

PyGCL implements three contrasting modes: (a) local-local, (b) global-local, and (c) global-global modes. You can refer to the `models` folder for details. Note that the bootstrapping latent loss involves some special model design (asymmetric online/offline encoders and momentum weight updates) and thus we implement contrasting modes involving this contrastive objective in a separate `BGRL` model.

### Contrastive Objectives

In `GCL.losses`, PyGCL implements the following contrastive objectives:

| Contrastive objectives                | Class name      |
| ------------------------------------- | --------------- |
| InfoNCE loss                          | `InfoNCELoss`   |
| Jensen-Shannon Divergence (JSD) loss  | `JSDLoss`       |
| Triplet Margin (TM) loss              | `TripletLoss`   |
| Bootstrapping Latent (BL) loss        | `BootstrapLoss` |
| Barlow Twins (BT) loss                | `BTLoss`        |
| VICReg loss                           | `VICRegLoss`    |

All these objectives are for contrasting positive and negative pairs at the same scale (i.e. local-local and global-global modes). For global-local modes, we offer `G2L` variants except for Barlow Twins and VICReg losses. Moreover, for InfoNCE, JSD, and Triplet losses, we further provide `G2LEN` variants, primarily for node-level tasks, which involve explicit construction of negative samples. You can find their examples in the root folder.

### Negative Mining Strategies

In `GCL.losses`, PyGCL further implements four negative mining strategies that are build upon the InfoNCE contrastive objective:

| Hard negative mining strategies   | Class name                           |
| --------------------------------- | ------------------------------------ |
| Hard negative mixing              | `HardMixingLoss`                     |
| Conditional negative sampling     | `RingLoss`                           |
| Debiased contrastive objective    | `InfoNCELoss(debiased_nt_xent_loss)` |
| Hardness-biased negative sampling | `InfoNCELoss(hardness_nt_xent_loss)` |

# Examples

For a quick start, please check out the `examples` folder. We currently implemented the following methods:

* **DGI** (P. Veličković et al., Deep Graph Infomax, ICLR, 2019) [[Example1](examples/DGI_transductive.py), [Example2](DGI_inductive.py)]
* **InfoGraph** (F.-Y. Sun et al., InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization, ICLR, 2020) [[Example](examples/InfoGraph.py)]
* **MVGRL** (K. Hassani et al., Contrastive Multi-View Representation Learning on Graphs, ICML, 2020) [[Example1](examples/MVGRL_node.py), [Example2](examples/MVGRL_graph.py)]
* **GRACE** (Y. Zhu et al., Deep Graph Contrastive Representation Learning, GRL+@ICML, 2020) [[Example](examples/GRACE.py)]
* **BGRL** (S. Thakoor et al., Bootstrapped Representation Learning on Graphs, arXiv, 2021) [[Example1](examples/BGRL_L2L.py), [Example2](examples/BGRL_G2L.py)]
* **G-BT** (P. Bielak et al., Graph Barlow Twins: A Self-Supervised Representation Learning Framework for Graphs, arXiv, 2021) [[Example](examples/GBT.py)]
* **GCC** (J. Qiu et al., GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training, KDD, 2020)
* **SupCon** (P. Khosla et al., Supervised Contrastive Learning, NeurIPS, 2020)
* **HardMixing** (Y. Kalantidis et al., Hard Negative Mixing for Contrastive Learning, NeurIPS, 2020)
* **DCL** (C.-Y. Chuang et al., Debiased Contrastive Learning, NeurIPS, 2020)
* **HCL** (J. Robinson et al., Contrastive Learning with Hard Negative Samples, ICLR, 2021)
* **Ring** (M. Wu et al., Conditional Negative Sampling for Contrastive Learning of Visual Representations, ICLR, 2021)

