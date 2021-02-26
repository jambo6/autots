""" Imputation and interpolation. """
import torch
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer as _SimpleImputer
from torchcde import linear_interpolation_coeffs

from autots.base.mixins import NoFitTransformerMixin

from .mixin import (apply_fit_to_channels, apply_transform_to_channel_subset,
                    apply_transform_to_channels)


class SimpleImputer(TransformerMixin):
    """Basic imputation for tensors. Simply borrows from sklearns SimpleImputer.

    Assumes the size is (..., num_samples, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.

    Arguments:
        strategy (str): One of ('mean', 'median', 'most_frequent', 'constant').
        fill_value (float): The value to fill nans with, this is active only if `strategy = 'constant'`.
    """

    def __init__(self, strategy, fill_value, channel_indices=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.channel_indices = channel_indices

        self.imputer = _SimpleImputer(strategy=strategy, fill_value=fill_value)

    def __repr__(self):
        string = (
            "{}={}".format(self.strategy, self.fill_value)
            if self.strategy == "constant"
            else self.strategy
        )
        return "Impute {}".format(string)

    @apply_fit_to_channels
    def fit(self, data, labels=None):
        self.imputer.fit(data)
        return self

    @apply_transform_to_channel_subset
    @apply_transform_to_channels
    def transform(self, data):
        output_data = torch.Tensor(self.imputer.transform(data))
        return output_data


class NegativeFilter(NoFitTransformerMixin):
    """Replace negative values with zero.

    Arguments:
        fill_value (float): The values to replace the negative values with.
    """

    def __init__(self, fill_value=float("nan"), channel_indices=None):
        self.fill_value = fill_value
        self.channel_indices = channel_indices

    def __repr__(self):
        return "Negative filter"

    @apply_transform_to_channel_subset
    def transform(self, data):
        data[data < 0] = self.fill_value
        return data


class Interpolation(NoFitTransformerMixin):
    """Perform linear (or rectilinear) interpolation on the missing values.

    Arguments:
        method (str): One of 'linear' or 'rectilinear'.
    """

    def __init__(self, method="linear", channel_indices=None):
        assert method in [
            "linear",
            "rectilinear",
        ], "Got method {} which is not recognised".format(method)
        self.method = method
        self.channel_indices = channel_indices

        # Linear interpolation function requires the channel index of times
        self._rectilinear = 0 if self.method == "rectilinear" else None

    def __repr__(self):
        return "{} Interpolation".format(self.method.title())

    @apply_transform_to_channel_subset
    def transform(self, data):
        return linear_interpolation_coeffs(data, rectilinear=self._rectilinear)


class ForwardFill(NoFitTransformerMixin):
    """Forward fills data in a torch tensor of shape (..., num_samples, input_channels) along the num_samples dim.

    Arguments:
        fill_index (int): Denotes the index to fill down. Default is -2 as we tend to use the convention (...,
            num_samples, input_channels) filling down the num_samples dimension.
        backwards (bool): Set True to first flip the tensor along the num_samples axis so as to perform a backfill.
        channel_indices (list or None): Leave as None to apply to the whole dataset, else apply a list of indices to
            apply only to that subset of channels, e.g. [0, 2, 5] will apply to the first third and fifth channel.

    Example:
        >>> x = torch.tensor([[1, 2], [float('nan'), 1], [2, float('nan')]], dtype=torch.float)
        >>> ForwardFill().transform(x, fill_index=-2, backwards=False)
        tensor([
            [1., 2.],
            [1., 1.],
            [2., 1.]
        ])

    Returns:
        A tensor with forward filled data.
    """

    def __init__(self, fill_index=-2, backwards=False, channel_indices=None):
        self.fill_index = fill_index
        self.backwards = backwards
        self.channel_indices = channel_indices

    def __repr__(self):
        return "Forward fill"

    @apply_transform_to_channel_subset
    def transform(self, data):
        return _forward_fill(data, self.fill_index, self.backwards)


def _forward_fill(x, fill_index=-2, backwards=False):
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    # flipping if a backwards fill
    def backflip(x):
        x = x.flip(fill_index) if backwards else x
        return x

    x = backflip(x)

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    # Re-flip if backwards
    x = backflip(x)

    return x
