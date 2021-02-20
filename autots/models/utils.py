""" Helpful functions for running torch models. """
import numpy as np
import torch
from torch import nn


class NanLossWrapper(nn.Module):
    """Applies a given loss function along the length of the outputs removing any nans in the given labels.

    Given a PyTorch loss function and target labels of shape [N, L, C] (as opposed to [N, C]), this removes any
    values where labels are nan (corresponding to a finished time series) and computes the loss against the labels
    and predictions in the remaining non nan locations.
    """

    def __init__(self, criterion):
        """
        Args:
            criterion (nn.Module): A pytorch loss function.
        """
        super().__init__()

        assert isinstance(criterion, nn.Module)
        self.criterion = criterion

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        return self.criterion(preds[mask].reshape(-1), labels[mask].reshape(-1))


def get_number_of_parameters(model):
    """Returns the number of trainable parameters in a pytorch model.

    Args:
        model (nn.Module): Network module.

    Returns:
        An integer denoting the number of trainable parameters in the model.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def tune_number_of_parameters(model_builder, desired_num_params):
    """Tunes a model parameter to make the total model params as close to desired_num_params as possible.

    Given a function that accepts an integer argument and outputs an initialised nn.Module, increments the argument
    until the number of parameters in the output model is as close to `desired_num_params` as possible.

    Args:
        model_builder (function): A function that accepts an integer argument and outputs a model.
        desired_num_params (int): The number of parameters wanted in the output model.

    Returns:
        An initialised model with number of params close to `desired_num_params`.
    """
    min_params = get_number_of_parameters(model_builder(1))
    if min_params > desired_num_params:
        raise ValueError(
            "Minimum number of model params ({}) is more than desired ({})".format(
                min_params, desired_num_params
            )
        )

    model = None
    num_params = 0
    hidden_dim = 1
    while num_params < desired_num_params:
        model = model_builder(hidden_dim)
        num_params = get_number_of_parameters(model)
        hidden_dim += 1

    # Revert back one if it is closer
    if hidden_dim - 2 > 0:
        prev_build = model_builder(hidden_dim - 2)
        if (
            desired_num_params - get_number_of_parameters(prev_build)
            < desired_num_params - num_params
        ):
            model = prev_build

    return model
