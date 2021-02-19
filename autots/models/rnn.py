""" RNN/GRU/LSTM implementation. """
from torch import nn

MODELS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}


class RNN(nn.Module):
    """ Standard RNN. """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=None,
        num_layers=1,
        model_string="rnn",
        bias=True,
        dropout=0,
        return_sequences=True,
        apply_final_linear=True,
    ):
        """
        Args:
            input_dim (int): The dimension of the path.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
                net with the given density. Hidden and hidden hidden dims must be multiples of 32.
            model_string (str): Any of ('rnn', 'gru', 'lstm')
            bias (bool): Whether to add a bias term.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
            apply_final_linear (bool): Set False for no final linear layer to be applied to the hidden state.
        """
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.static_dim = static_dim
        self.num_layers = num_layers
        self.model_string = model_string
        self.bias = bias
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear

        # Get the model class
        assert (
            model_string in MODELS.keys()
        ), "model_string must be one of ('rnn', 'lstm', 'gru')"
        model = MODELS[model_string]

        # Network is static dim is set
        if self.static_dim is not None:
            self.initial_net = nn.Sequential(
                nn.Linear(self.static_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Initialise
        self.rnn = model(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            batch_first=True,
        )

        # Output layer
        self.total_hidden_size = num_layers * hidden_dim
        self.final_linear = (
            nn.Linear(self.total_hidden_size, output_dim)
            if self.apply_final_linear
            else lambda x: x
        )

    def _setup_h0(self, inputs):
        """ Puts static data through a small network if it is set. """
        if self.static_dim is not None:
            assert (
                len(inputs) == 2
            ), "Inputs must be a 2-tuple of (static_data, temporal_data)."
            static_data, temporal_data = inputs
            h0 = self.initial_net(static_data).unsqueeze(0)
        else:
            temporal_data = inputs
            h0 = None
        return temporal_data, h0

    def forward(self, x):
        # Handle inputs and setup h0
        x, h0 = self._setup_h0(x)

        # Run the RNN
        h_full, _ = self.rnn(x, h0)

        # Terminal output if classifcation else return all outputs
        if self.return_sequences:
            outputs = self.final_linear(h_full)
        else:
            outputs = self.final_linear(h_full[:, -1, :])

        return outputs
