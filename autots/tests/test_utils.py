import pytest
import torch
import pandas as pd
from autots import utils


@pytest.mark.parametrize(
    "static_dim, ragged, pandas_format", [(None, False, False), (None, False, True), (5, True, True)]
)
def test_make_time_series_problem(static_dim, ragged, pandas_format):
    data, labels = utils.make_time_series_problem(static_dim=static_dim, ragged=ragged, pandas_format=pandas_format)

    dtype = pd.DataFrame if pandas_format else torch.Tensor
    if static_dim is None:
        assert isinstance(data, dtype)
    else:
        assert all([isinstance(x, dtype) for x in data])


@pytest.mark.parametrize("ragged, pandas_format", [(False, False), (True, False), (True, True)])
def test_make_online_labels(ragged, pandas_format):
    # Test online labels are generated properly
    data, labels = utils.make_time_series_problem(
        static_dim=None, online=True, ragged=ragged, pandas_format=pandas_format
    )
    assert len(data) == len(labels)
