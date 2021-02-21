import pandas as pd
import pytest
import torch

from autots import utils
from autots.models import RNN
from autots.tests.helpers import set_seed, training_loop


@pytest.mark.parametrize(
    "static_dim, ragged, pandas_format",
    [(None, False, False), (None, False, True), (5, True, True)],
)
def test_make_time_series_problem(static_dim, ragged, pandas_format):
    data, labels = utils.make_time_series_problem(
        static_dim=static_dim, ragged=ragged, pandas_format=pandas_format
    )

    dtype = pd.DataFrame if pandas_format else torch.Tensor
    if static_dim is None:
        assert isinstance(data, dtype)
    else:
        assert all([isinstance(x, dtype) for x in data])


@pytest.mark.parametrize(
    "ragged, pandas_format", [(False, False), (True, False), (True, True)]
)
def test_make_online_labels(ragged, pandas_format):
    # Test online labels are generated properly
    data, labels = utils.make_time_series_problem(
        static_dim=None, problem="online", ragged=ragged, pandas_format=pandas_format
    )
    assert len(data) == len(labels)


def setup_nan_loss_problem(problem="oneshot"):
    input_dim = 3
    data, labels = utils.make_time_series_problem(
        n_channels=3,
        static_dim=None,
        n_classes=2,
        problem=problem,
        ragged=False,
    )

    # If online mask the labels to get nans
    if problem == "online":
        mask = torch.randint(0, 2, labels.shape)
        labels[mask == 1] = float("nan")

    model = RNN(
        input_dim=input_dim,
        output_dim=labels.size(-1),
        hidden_dim=10,
        return_sequences=False,
    )

    return model, data, labels


def test_nan_loss_equals_non_nan_loss():
    # Check nan loss achieves same value
    for seed in [1, 2, 3]:
        accs = []
        for loss_str in ["bce", "nan_bce"]:
            set_seed(seed)
            model, data, labels = setup_nan_loss_problem()
            _, acc = training_loop(
                model, data, labels, n_epochs=1, loss_str=loss_str, lr=0.001
            )
            accs.append(acc)
        assert accs[0] == accs[1]


def test_nan_loss_works_on_nans():
    model, data, labels = setup_nan_loss_problem(problem="online")
    labels = labels[:, -1]
    _, acc = training_loop(model, data, labels, n_epochs=1, loss_str="nan_bce")
    assert 0 < acc <= 1
