import pytest

from autots.models import NeuralCDE
from autots.tests import helpers
from autots.utils import make_time_series_problem


def setup_ncde_problem(static_dim=None, use_initial=True):
    # Simple problem
    input_dim = 4
    output_dim = 1
    data, labels = make_time_series_problem(
        n_channels=input_dim, n_classes=output_dim, static_dim=static_dim
    )

    # Setup and NCDE
    hidden_dim = 15
    model = NeuralCDE(
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=static_dim,
        interpolation="linear",
        use_initial=use_initial,
    )

    return model, data, labels


@pytest.mark.parametrize(
    "static_dim, use_initial",
    [(None, True), (None, False), (5, True), (5, False)],
)
def test_ncde_static_initial(static_dim, use_initial):
    # Test the model runs and gets a normal accuracy
    model, data, labels = setup_ncde_problem(
        static_dim=static_dim, use_initial=use_initial
    )
    _, acc = helpers.training_loop(model, data, labels, n_epochs=10)
    assert 0 <= acc <= 1
