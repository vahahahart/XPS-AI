import os
import random
from pathlib import Path
from time import time

import numpy as np
from matplotlib import pyplot as plt
from dvclive import Live
from ruamel.yaml import YAML

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model.train.metrics import IoULoss, Accuracy, Precision, Recall
from model.train.dataset import XPSDataset
from model.model import XPSModel


def load_params():
    yaml_loader = YAML(typ='safe', pure=True)
    params = yaml_loader.load(Path('model/params.yaml'))

    seed = params['seed']
    path_to_data = params['data_path']

    return seed, path_to_data, params['train']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model, val_loader, criterion, metrics=()):
    with torch.no_grad():

        results = {
            'val_loss': 0,
            'peak_loss': 0,
            'max_loss': 0
        }
        results.update([(f'peak_{name}', 0) for name, f in metrics])
        results.update([(f'max_{name}', 0) for name, f in metrics])

        l = len(val_loader)
        for x, peak_mask, max_mask in val_loader:
            pred_peak_mask, pred_max_mask = model(x)
            peak_targets = peak_mask.view(-1)
            max_targets = max_mask.view(-1)

            peak_inputs = pred_peak_mask.view(-1)
            max_inputs = pred_max_mask.view(-1)

            results['peak_loss'] += criterion(peak_inputs, peak_targets).detach().numpy() / l
            results['max_loss'] += criterion(max_inputs, max_targets).detach().numpy() / l

            for name, func in metrics:
                results[f'peak_{name}'] += func(peak_inputs, peak_targets).detach().numpy() / l
                results[f'max_{name}'] += func(max_inputs, max_targets).detach().numpy() / l

        results['val_loss'] = (results['peak_loss'] + results['max_loss']) / 2

    return results

def train_one_epoch(model, train_loader, optimizer, criterion):
    sum_train_loss = 0

    for x, peak_mask, max_mask in train_loader:
        pred_peak_mask, pred_max_mask = model(x)

        gt = torch.cat((peak_mask, max_mask)).view(-1)
        pred = torch.cat((pred_peak_mask, pred_max_mask)).view(-1)

        train_loss = criterion(pred, gt)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        sum_train_loss += train_loss.detach().numpy()

    return sum_train_loss 


if __name__ == '__main__':
    # init train params
    seed, path_to_data, params = load_params()
    num_epoch = params['num_epoch']
    batch_size = params['batch_size']
    split = params['train_test_split']
    lr = params['learning_rate']

    # fixing random seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # load model and data
    model = XPSModel()
    data = XPSDataset(path_to_data)

    train_data, val_data = random_split(data, (split, 1-split), gen)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        worker_init_fn=seed_worker,
        generator=gen
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        worker_init_fn=seed_worker,
        generator=gen
    )

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = IoULoss()
    metrics = (
        ('accuracy', Accuracy()),
        ('precison', Precision()),
        ('Recall', Recall())
    )

    with Live() as live:
        path = 'model/train/trained_models/model_v1.pth'

        for epoch in range(num_epoch):
            model.train()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            model.eval()
            out_metrics = evaluate(model, val_loader, criterion, metrics)
            for metric_name, val in out_metrics.items():
                live.log_metric(metric_name, val, timestamp=True)
            live.next_step()
        
        torch.save(model, path)
        live.log_artifact(
            path,
            type='model',
            name='cnn_v1.0',
            desc='cnn trained on new dataset'
        )
