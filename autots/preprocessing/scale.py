""" Data scaling transformer. """
import torch
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (FunctionTransformer, MaxAbsScaler,
                                   MinMaxScaler, StandardScaler)

from .mixin import apply_fit_to_channels, apply_transform_to_channels

SCALERS = {
    "stdsc": StandardScaler,
    "ma": MaxAbsScaler,
    "mms": MinMaxScaler,
}


class TensorScaler(TransformerMixin):
    """Scaling for 3D tensors.

    Assumes the size is (..., num_samples, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.

    Arguments:
        method (str): Scaling method, one of ('stdsc', 'ma', 'mms').
        scaling_function (transformer): Specification of an sklearn transformer that performs a method operation.
            Only one of this or method can be specified.
    """

    def __init__(self, method="stdsc", scaling_function=None):
        self.method = method

        if all([method is None, scaling_function is None]):
            self.scaler = FunctionTransformer(func=None)
        elif isinstance(method, str):
            self.scaler = SCALERS.get(method)()
            assert (
                self.scaler is not None
            ), "Scalings allowed are {}, recieved {}.".format(SCALERS.keys(), method)
        else:
            self.scaler = scaling_function

    def __repr__(self):
        return "{} scaling".format(self.method.upper())

    @apply_fit_to_channels
    def fit(self, data, labels=None):
        self.scaler.fit(data)
        return self

    @apply_transform_to_channels
    def transform(self, data):
        output_data = torch.Tensor(self.scaler.transform(data))
        return output_data
