""" Helpful utilites used around the project. """
import random

import numpy as np
import pandas as pd
import torch


def make_time_series_problem(
    n_samples=50,
    length=10,
    n_channels=5,
    static_dim=3,
    problem="oneshot",
    n_classes=2,
    ragged=False,
    masking=False,
    pandas_format=False,
):
    """Makes a simple TSC problem.

    Note:
        - Automatically concatenates a time channel.

    Args:
        n_samples (int): Number of data samples.
        length (int): Length of the dataset.
        n_channels (int): Number of input channels. Must be > 1 to allow time concatenation.
        static_dim (int or None): Dimension of the static data, set False for no static data.
        problem (str): Any of ('oneshot', 'online', 'regression').
        n_classes (int): Number of classes (inactive if regression).
        ragged (bool): Create ragged tensors.
        masking (bool): Mask with some zeros
        pandas_format (bool): Set True to output in pandas format.

    Returns:
        Tensors (or dataframes) in the format (data, stratify) where data is either a single dataframe or tensor, or a
        list of (static_data, temporal_data) depending on the specification of static dim.
    """
    # Setup the data
    temporal_data = torch.randn(n_samples, length, n_channels - 1)
    times = torch.arange(length).reshape(1, -1, 1).repeat(n_samples, 1, 1)
    temporal_data = torch.cat([times, temporal_data], axis=-1)

    # Make some stratify and modify the data to make it predictable
    if problem == "regression":
        labels = torch.randn(len(temporal_data), 1, dtype=torch.float)
        temporal_data = temporal_data * labels.reshape(-1, 1, 1)
    elif problem == "online":
        assert n_classes == 2
        labels = torch.randint(0, n_classes, temporal_data.shape[:2], dtype=torch.float)
        # Make some rows all zero else oevery sample has a 1
        labels[0::10] = torch.zeros_like(labels[0::10], dtype=torch.float)
    elif problem == "oneshot":
        labels = torch.randint(0, n_classes, [len(temporal_data), 1], dtype=torch.float)
        temporal_data = temporal_data * labels.reshape(-1, 1, 1)
    else:
        raise NotImplementedError

    # Some data modifications
    if masking:
        temporal_data = temporal_data * torch.randint(0, 2, temporal_data.shape)
    if ragged:
        temporal_data = np.array(
            [d[: random.randint(2, 10)] for i, d in enumerate(temporal_data)],
            dtype=object,
        )
        if labels.dim() > 1:
            labels = np.array(
                [ls[: len(d)] for ls, d in zip(labels, temporal_data)], dtype=object
            )

    # Format data for outputting
    if pandas_format:
        temporal_data = _convert_temporal_to_pandas(temporal_data)
        if problem == "online":
            labels = _convert_online_labels_to_pandas(labels)
        else:
            labels = pd.Series(labels)

    # Static data
    static_data = None
    if static_dim is not None:
        static_data = torch.randn(n_samples, static_dim)
        if pandas_format:
            static_data = pd.DataFrame(index=range(n_samples), data=static_data.numpy())

    # We will always handle as (tensors, stratify)
    if static_data is None:
        data = temporal_data
    else:
        data = (static_data, temporal_data)

    return data, labels


def _convert_online_labels_to_pandas(labels):
    """ Converts a list of online labels to pandas format. """
    to_convert = [
        torch.cat((torch.arange(len(lb)).view(-1, 1), lb.view(-1, 1)), dim=-1)
        for lb in labels
    ]
    labels = _convert_temporal_to_pandas(to_convert)
    return labels


def _convert_temporal_to_pandas(data):
    """ Converts temporal data to a pandas dataframe. """
    # Convert tensor to list so the rest of the code is the same
    if isinstance(data, torch.Tensor):
        assert data.dim() == 3, "Function works only for 3D tensors."
        data = [d for d in data]
    num_channels = data[0].shape[-1]

    # Make a frame for each list index and concatenate
    frames = []
    for i, d in enumerate(data):
        ids = i * torch.ones(len(d), 1)
        d = torch.cat([ids, d], axis=-1)
        frames.append(pd.DataFrame(data=d.numpy()))
    frame = pd.concat(frames)
    frame.columns = ["id", "time"] + [
        "feature_{}".format(x) for x in range(1, num_channels)
    ]

    return frame
