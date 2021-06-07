from GCL.augmentations.GraphAug import Graph, GraphAug


class Identity(GraphAug):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g
