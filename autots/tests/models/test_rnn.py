import pytest

from autots.models import RNN
from autots.tests.helpers import set_seed, training_loop
from autots.utils import make_time_series_problem

set_seed(0)


@pytest.mark.parametrize(
    "model_string, static_dim",
    [("rnn", None), ("rnn", 5), ("gru", None), ("gru", 4)],
)
def test_rnn_handles_static(model_string, static_dim):
    input_dim = 3
    data, labels = make_time_series_problem(
        n_channels=3,
        static_dim=static_dim,
        n_classes=2,
    )

    # Simple training loop
    model = RNN(
        input_dim=input_dim,
        static_dim=static_dim,
        output_dim=1,
        hidden_dim=10,
        model_string=model_string,
        return_sequences=False,
    )
    _, acc = training_loop(model, data, labels, n_epochs=5)

    assert 0 < acc <= 1
