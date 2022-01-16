import torch
import os.path as osp
import GCL.loss as L
import GCL.augmentor as A
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from functools import partial
from torch.optim import Adam
from GCL.eval import SVMEvaluator
from GCL.model import DualBranchContrast
from GCL.utils import sinkhorn
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, num_clusters):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.C = torch.nn.Parameter(torch.FloatTensor(64, num_clusters), requires_grad=True).to('cuda')  # prototypes
        torch.nn.init.xavier_uniform_(self.C)

    def forward(self, data):
        aug1, aug2 = self.augmentor
        batch = data.batch
        data1 = aug1(data)
        data2 = aug2(data)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_weight1 = data1.x, data1.edge_index, data1.edge_attr
        x2, edge_index2, edge_weight2 = data2.x, data2.edge_index, data2.edge_attr
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer, temperature=1.0, coef=1.0):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]

        scores1 = g1 @ encoder_model.C
        scores2 = g2 @ encoder_model.C

        with torch.no_grad():
            q1 = sinkhorn(scores1)
            q2 = sinkhorn(scores2)

        p1 = F.softmax(scores1 / temperature, dim=1)
        p2 = F.softmax(scores2 / temperature, dim=1)

        loss_consistency = -0.5 * (q2 * torch.log(p1) + q1 * torch.log(p2)).mean()

        loss = contrast_model(g1=g1, g2=g2, batch=data.batch) + coef * loss_consistency
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder_model.C = F.normalize(encoder_model.C, dim=0, p=2)

        epoch_loss += loss.item()
    return epoch_loss


@ignore_warnings(category=ConvergenceWarning)
def eval(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    evaluator = SVMEvaluator(
        linear=True, split=split,
        metrics={'micro_f1': partial(f1_score, average='micro'), 'macro_f1': partial(f1_score, average='macro')})
    result = evaluator(x, y)
    return result


def main():
    batch_size = 128
    device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = TUDataset(path, name='PTC_MR')
    dataloader = DataLoader(dataset, batch_size=batch_size)
    input_dim = max(dataset.num_features, 1)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([
        A.RWSampling(num_seeds=1000, walk_length=10),
        A.NodeDropping(pn=0.1),
        A.FeatureMasking(pf=0.1),
        A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), num_clusters=20).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = eval(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]["mean"]:.4f}±{test_result["micro_f1"]["std"]:.4f},'
          f' F1Ma={test_result["macro_f1"]["mean"]:.4f}±{test_result["macro_f1"]["std"]:.4f}')


if __name__ == '__main__':
    main()
