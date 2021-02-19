from autots.utils import make_time_series_problem
from autots.models import utils
from autots.models import RNN


def test_tune_number_of_parameters():
    # Load a basic classification problem
    input_dim = 5
    output_dim = 1
    data, labels = make_time_series_problem(n_channels=input_dim, static_dim=None, n_classes=output_dim)

    # Create a model builder
    def model_builder(x):
        return RNN(
            input_dim=input_dim,
            hidden_dim=x,
            output_dim=output_dim,
            return_sequences=False,
        )

    # Check works
    for num_params in [1000, 50000, 100000]:
        model = utils.tune_number_of_parameters(model_builder, num_params)
        assert 0.9 * num_params < utils.get_number_of_parameters(model) < 1.1 * num_params


