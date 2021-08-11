import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from GCL.utils import split_dataset


def get_split(name, dataset, num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    if name.startswith('ogb'):
        split = dataset.get_idx_split()
        if 'valid' in split:
            split['val'] = split['valid']
        return split

    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'val': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


class LogisticRegression(nn.Module):
    def __init__(self, ft_in, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(ft_in, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(object):
    def __init__(self, num_epochs: int = 5000, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        for key in ['train', 'test', 'val']:
            assert key in split

        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 1
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')

                    y_val = y[split['val']].detach().cpu().numpy()
                    y_pred = classifier(x[split['val']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_val_micro
        }


def LR_classification(
        z, dataset, num_epochs: int = 5000, test_interval: int = 20,
        split_mode: str = 'rand', *args, **kwargs):
    device = z.device
    z = z.detach().to(device)
    num_hidden = z.size(1)
    y = dataset.y.view(-1).to(device)
    num_classes = dataset.y.max().item() + 1
    classifier = LogisticRegression(num_hidden, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    split = split_dataset(dataset, split_mode, *args, **kwargs)
    split = {k: v.to(device) for k, v in split.items()}
    if 'valid' in split:
        split['val'] = split['valid']
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    best_val_micro = 0
    best_test_micro = 0
    best_test_macro = 0
    best_epoch = 0

    with tqdm(total=num_epochs, desc='(LR)',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
        for epoch in range(num_epochs):
            classifier.train()
            optimizer.zero_grad()

            output = classifier(z[split['train']])
            loss = nll_loss(f(output), y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % test_interval == 0:
                classifier.eval()
                y_test = y[split['test']].detach().cpu().numpy()
                y_pred = classifier(z[split['test']]).argmax(-1).detach().cpu().numpy()
                test_micro = f1_score(y_test, y_pred, average='micro')
                test_macro = f1_score(y_test, y_pred, average='macro')

                y_val = y[split['val']].detach().cpu().numpy()
                y_pred = classifier(z[split['val']]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average='micro')

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_epoch = epoch

                pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                pbar.update(test_interval)

    return {
        'F1Mi': best_test_micro,
        'F1Ma': best_test_macro
    }


def SVM_classification(z, y, seed):
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accuracies = []
    macro_scores = []
    for train_index, test_index in kf.split(z, y):
        x_train, x_test = z[train_index], z[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        macro_scores.append(f1_score(y_test, classifier.predict(x_test), average='macro'))

    return {
        'F1Mi': [np.mean(accuracies), np.std(accuracies)],
        'F1Ma': [np.mean(macro_scores), np.std(macro_scores)]
    }
