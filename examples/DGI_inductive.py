import torch
import os.path as osp
import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import uniform
from torch_geometric.data import NeighborSampler
from torch_geometric.datasets import Reddit


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.layers[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, dataloader, optimizer):
    encoder_model.train()
    total_loss = total_examples = 0
    for batch_size, node_id, adjs in dataloader:
        adjs = [adj.to('cuda') for adj in adjs]
        optimizer.zero_grad()
        z, g, zn = encoder_model(data.x[node_id], adjs)
        loss = contrast_model(h=z, g=g, hn=zn)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * z.shape[0]
        total_examples += z.shape[0]
    return total_loss / total_examples


def test(encoder_model, data, dataloader):
    encoder_model.eval()
    zs = []
    for i, (batch_size, node_id, adjs) in enumerate(dataloader):
        adjs = [adj.to('cuda') for adj in adjs]
        z, _, _ = encoder_model(data.x[node_id], adjs)
        zs.append(z)
    x = torch.cat(zs, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(x, data.y, split)
    return result


def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0].to(device)

    train_loader = NeighborSampler(data.edge_index, node_idx=None,
                                   sizes=[10, 10, 25], batch_size=128,
                                   shuffle=True, num_workers=32)
    test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                  sizes=[10, 10, 25], batch_size=128,
                                  shuffle=False, num_workers=32)

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=3).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.0001)

    with tqdm(total=30, desc='(T)') as pbar:
        for epoch in range(1, 31):
            loss = train(encoder_model, contrast_model, data, train_loader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data, test_loader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
