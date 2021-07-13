from collections import namedtuple
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn

from torch.optim import Adam
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from GCL.utils import split_dataset


class LogisticRegression(nn.Module):
    def __init__(self, ft_in, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(ft_in, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def LR_classification(
        z, dataset, num_epochs: int = 5000, test_interval: int = 20,
        split_mode: str = 'rand', verbose: bool = False, *args, **kwargs):
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
            if verbose:
                print(f'\r(LR) | Epoch={epoch:03d}, '
                      f'best test F1Mi={best_test_micro:.4f}, '
                      f'F1Ma={best_test_macro:.4f}', end='')
    if verbose: print()
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


def MLP_regression(z: torch.FloatTensor, y: torch.FloatTensor, target,
                   evaluator=None, batch_size: int = None, split: dict = None,
                   num_epochs: int = 2000, hidden_dim: int = 256):
    device = z.device
    input_dim = z.size()[1]
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    ).to(device)
    optimizer = Adam(net.parameters(), lr=0.01, weight_decay=0.0)

    if split is None:
        dataset = namedtuple('Dataset', ['x'])(x=z)
        split = split_dataset(dataset, split_mode='rand', train_ratio=0.8, test_ratio=0.1)

    loss_fn = nn.L1Loss()

    best_val_mae = 1e10
    best_test_mae = 1e10
    best_epoch = 0

    pbar = tqdm(total=num_epochs)
    target_id, target_name = target
    pbar.set_description(f'({target_id}) {target_name}')
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        output = net(z).view(-1)
        train_error = loss_fn(output[split['train']], y[split['train']])

        train_error.backward()
        optimizer.step()

        if evaluator is None:
            train_mae = (output[split['train']] - y[split['train']]).mean().abs().item()
            val_mae = (output[split['val']] - y[split['val']]).mean().abs().item()
            test_mae = (output[split['test']] - y[split['test']]).mean().abs().item()
        else:
            def evaluate(mask_name):
                mask = split[mask_name]
                y_pred = output[mask]
                y_true = y[mask]
                non_nan_mask = y_true.isnan().logical_not()
                y_pred = y_pred[non_nan_mask]
                y_true = y_true[non_nan_mask]
                input_dict = {'y_pred': y_pred, 'y_true': y_true}
                result_dict = evaluator.eval(input_dict)
                return result_dict['mae']
            train_mae = evaluate('train')
            val_mae = evaluate('val')
            test_mae = evaluate('test')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_test_mae = test_mae
            best_epoch = epoch

        pbar.update(1)
        pbar.set_postfix(
            {'tr': train_mae, 'v': val_mae, 't': test_mae,
             'v*': best_val_mae, 't*': best_test_mae}
        )
    pbar.close()

    return {
        'mae': best_test_mae
    }


def LR_binary_classification(z: torch.FloatTensor, y: torch.LongTensor, dataset, evaluator, hidden_dim: int = 128, num_epochs: int = 1000):
    device = z.device
    input_dim = z.size()[1]
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, 1)
    ).to(device)

    split = split_dataset(dataset, split_mode='ogb')
    if 'valid' in split:
        split['val'] = split['valid']

    optimizer = Adam(net.parameters(), lr=0.01, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_metric = -1
    best_test_metric = -1
    best_epoch = 0

    y = y.to(torch.float32)

    pbar = tqdm(total=num_epochs, desc='LR')
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()

        output = net(z).view(-1)
        train_error = loss_fn(output[split['train']], y[split['train']].view(-1))

        train_error.backward()
        optimizer.step()

        def evaluate(split_name: str):
            mask = split[split_name]
            pred = (output[mask] > 0).view(-1, 1).to(torch.long)
            input_dict = {'y_true': y[mask], 'y_pred': pred}
            output_dict = evaluator.eval(input_dict)
            return output_dict['rocauc']

        train_metric = evaluate('train')
        val_metric = evaluate('val')
        test_metric = evaluate('test')

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test_metric
            best_epoch = epoch

        pbar.update(1)
        pbar.set_postfix(
            {'tr': train_metric, 'v': val_metric, 't': test_metric,
             'v*': best_val_metric, 't*': best_test_metric}
        )
    pbar.close()

    return {
        'rocauc': best_test_metric
    }
