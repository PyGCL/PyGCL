import torch


class ConModel(torch.nn.Module):
    def __init__(self, loss_fn, sampler, *args, **kwargs):
        super(ConModel, self).__init__()
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.kwargs = kwargs

    def forward(self, h1, h2, g1=None, g2=None, batch=None, h3=None, h4=None):
        if batch is None:
            if h3 is None and h4 is None:  # same-scale contrasting
                anchor, sample, pos_mask, neg_mask = self.sampler(anchor=h1, sample=h2)
                return self.loss_fn(anchor=anchor, sample=sample, pos_mask=pos_mask, neg_mask=neg_mask, **self.kwargs)
            else:  # global to local, only one graph
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
        else:  # global to local, multiple graphs
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        l1 = self.loss_fn(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss_fn(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)
        return (l1 + l2) * 0.5
