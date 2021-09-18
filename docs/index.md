---
github_url: https://github.com/GraphCL/PyGCL/
---

# PyGCL Documentation

**PyGCL** is an open-source Graph Contrastive Learning (GCL) library for [PyTorch](https://pytorch.org/), which features modularized GCL components from published papers, standardized evaluation, and experiment management.

It implements four main components of graph contrastive learning algorithms:
* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting architectures and modes: generate positive and negative pairs according to node and graph embeddings.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

It also implements utilities for training models, evaluating model performance, and managing experiments.

```{toctree}
:maxdepth: 1
:caption: "Contents"
:glob:

notes/installation
```

# Package Reference

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
