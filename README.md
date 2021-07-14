# PyGCL: Graph Contrastive Learning for PyTorch

PyGCL is an open-source library for graph contrastive learning (GCL), which features modularized GCL components from published papers, standardized evaluation, and experiment management. 

---

# Prerequisites

PyGCL needs the following packages to be installed beforehand:

* Python 3.8+
* PyTorch 1.7+
* PyTorch-Geometric 1.7
* DGL 0.5+
* Scikit-learn 0.24+

# Getting Started

Take a look at various examples located at the root directory. For example, try the following command to train a simple GCN for node classification on the WikiCS dataset using the local-local contrasting mode:

````
python train_node_l2l.py --dataset WikiCS --param_path params/GRACE/wikics@current.json --base_model GCNConv
````

For detailed parameter settings, please refer to `wikics@current.json`. These examples are mainly for reproducing experiments in our [benchmarking study](https://openreview.net/forum?id=fYxEnpY-__G). You can find more details regarding general practices of graph contrastive learning in the paper.

# Usage

## Package Overview

Our PyGCL implements four main components of graph contrastive learning algorithms:

* graph augmentation: transforms input graphs into congruent graph views.
* contrasting modes: specifies positive and negative pairs.
* contrastive objectives: computes the likelihood score for positive and negative pairs.
* negative mining strategies: improves the negative sample set by considering hardness of negative sample.

We also implement utilities for loading datasets, training models, and running experiments.

## Building Your Own GCL Algorithms

Besides try the above examples for node and graph classification tasks, you can also build your own graph contrastive learning algorithms straightforwardly.

### Graph Augmentation

In `GCL.augmentors`, PyGCL provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, PyGCL implements the following augmentation functions:

| Augmentation                             | Class name          |
| ---------------------------------------- | ------------------- |
| Edge Adding (EA)                        | `EdgeAdding`      |
| Edge Removing (ER)                      | `EdgeRemoving`    |
| Feature Masking (FM)                    | `FeatureMasking`  |
| Feature Dropout (FD)                    | `FeatureDropout`  |
| Personalized PageRank (PPR)             | `PPRDiffusion`    |
| Markov Diffusion Kernel (MDK)           | `MarkovDiffusion` |
| Node Dropping (ND)                      | `NodeDropping`    |
| Subgraphs induced by Random Walks (RWS) | `RWSampling`      |
| Ego-net Sampling (ES)                   | `Identity`        |

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

### Contrastive Objectives and Hard Negative Mining Strategies

In `GCL.losses`, PyGCL implements the following contrastive objectives:

| Contrastive objectives                   | Class name          |
| ---------------------------------------- | ------------------- |
| InfoNCE loss                          | `InfoNCELoss`       |
| Jensen-Shannon Divergence (JSD) loss     | `JSDLoss`           |
| Triplet Margin (TM) loss | `TripletLoss` |
| Bootstrapping Latent (BL) loss       | `BootstrapLoss` |
| Barlow Twins (BT) loss | `BTLoss` |
| VICReg loss                          | `VICRegLoss` |

All these objectives are for contrasting positive and negative pairs at the same scale (i.e. local-local and global-global modes). For global-local modes, we offer `G2L` variants except for Barlow Twins and VICReg losses. Moreover, for InfoNCE, JSD, and Triplet losses, we further provide `G2LEN` variants, primarily for node-level tasks, which involve explicit construction of negative samples. You can find their examples in the root folder.

## Utilities

PyGCL provides various utilities for data loading, model training, and experiment execution.

In `GCL.util` you can use the following utilities:

* `split_dataset`: splits the dataset into train/test/validation sets according to public or random splits. Currently, four split modes are supported: [`rand`, `ogb`, `wikics`, `preload`] .
* `seed_everything`: manually sets the seed to numpy and PyTorch environments to ensure better reproducebility.
* `SimpleParam`: provides a simple parameter configuration class to manage parameters from microsoft-nni, JSON, and YAML files.

We also implement two downstream classifiers`LR_classification` and `SVM_classification` in `GCL.eval` based on PyTorch and Scikit-learn respectively.

Moreover, based on PyTorch Geometric, we provide functions for loading common node and graph datasets. You can use`load_node_dataset` and `load_graph_dataset` in `utils.py`.



