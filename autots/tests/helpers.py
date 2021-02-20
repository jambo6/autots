""" Reusable helper functions for testing. """
import torch
from torch import nn, optim

from autots.models.utils import NanLossWrapper

# Define criterions and NaN criterion
CRITERIONS = {
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "nan_bce": NanLossWrapper(nn.BCEWithLogitsLoss()),
    "nan_ce": NanLossWrapper(nn.CrossEntropyLoss()),
    "nan_mse": NanLossWrapper(nn.MSELoss()),
}


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def training_loop(model, data, labels, n_epochs=5, loss_str="bce", lr=0.1):
    # Setup
    labels = labels.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CRITERIONS[loss_str]

    # Get loopy
    for _ in range(n_epochs):
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

    # Evaluate on train
    preds = torch.sigmoid(model(data))

    # Non-nan eval
    if "bce" in loss_str:
        mask = ~torch.isnan(labels)
        metric = (
            (labels[mask].view(-1) == torch.round(preds[mask]).view(-1)).sum()
            / mask.sum()
        ).item()
    elif "mse" in loss_str:
        metric = criterion(preds, labels).item()
    else:
        raise NotImplementedError

    return preds, metric
