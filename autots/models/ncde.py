import torch
import torchcde
from torch import nn

SPLINES = {
    "cubic": torchcde.NaturalCubicSpline,
    "linear": torchcde.LinearInterpolation,
    "rectilinear": torchcde.LinearInterpolation,
}


class NeuralCDE(nn.Module):
    """ Performs the Neural CDE training process over a batch of time series. """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        static_dim=None,
        hidden_hidden_dim=15,
        num_layers=3,
        use_initial=True,
        interpolation="linear",
        adjoint=True,
        solver="rk4",
        return_sequences=False,
        apply_final_linear=True,
        return_filtered_rectilinear=True,
    ):
        """
        Args:
            input_dim (int): The dimension of the path.
            hidden_dim (int): The dimension of the hidden state.
            output_dim (int): The dimension of the output.
            static_dim (int): The dimension of any static values, these will be concatenated to the initial values and
                put through a network to build h0.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
                net with the given density. Hidden and hidden hidden dims must be multiples of 32.
            use_initial (bool): Set True to use the initial absolute values to generate h0.
            adjoint (bool): Set True to use odeint_adjoint.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
            apply_final_linear (bool): Set False for no final linear layer to be applied to the hidden state.
            return_filtered_rectilinear (bool): Set True to return every other output if the interpolation scheme chosen
                is rectilinear, this is because rectilinear doubles the input length. False will return the full output.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.static_dim = static_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.use_initial = use_initial
        self.interpolation = interpolation
        self.adjoint = adjoint
        self.solver = solver
        self.return_sequences = return_sequences
        self.apply_final_linear = apply_final_linear
        self.return_filtered_rectilinear = return_filtered_rectilinear

        # Set initial linear layer
        if self.initial_dim > 0:
            self.initial_linear = nn.Linear(self.initial_dim, self.hidden_dim)

        # Interpolation function
        assert (
            self.interpolation in SPLINES.keys()
        ), "Unrecognised interpolation scheme {}".format(self.interpolation)
        self.spline = SPLINES.get(self.interpolation)

        # The net that is applied to h_{t-1}
        self.func = self._setup_vector_field()

        # Linear classifier to apply to final layer
        self.final_linear = (
            nn.Linear(self.hidden_dim, self.output_dim)
            if apply_final_linear
            else lambda x: x
        )

    @property
    def initial_dim(self):
        # Setup initial dim dependent on `use_initial` and `static_dim` options
        initial_dim = 0
        if self.use_initial:
            initial_dim += self.input_dim
        if self.static_dim is not None:
            initial_dim += self.static_dim
        return initial_dim

    def _setup_vector_field(self):
        # Now the model can be used as a subclass with this function replaced to use different vector fields
        return _NCDEFunc(
            self.input_dim, self.hidden_dim, self.hidden_hidden_dim, self.num_layers
        )

    def _setup_h0(self, inputs):
        """Sets up the initial value of the hidden state.

        The hidden state depends on the options `use_initial` and `static_dim`. If either of these are specified the
        hidden state will be generated via a network applied to either a concatenation of the initial and static data,
        or a network applied to just initial/static depending on options. If neither are specified then a zero initial
        hidden state is used.
        """
        if self.static_dim is None:
            coeffs = inputs
            if self.use_initial:
                h0 = self.initial_linear(inputs[..., 0, :])
            else:
                h0 = torch.autograd.Variable(
                    torch.zeros(inputs.size(0), self.hidden_dim)
                ).to(inputs.device)
        else:
            assert (
                len(inputs) == 2
            ), "Inputs must be a 2-tuple of (static_data, temporal_data)"
            static, coeffs = inputs
            if self.use_initial:
                h0 = self.initial_linear(torch.cat((static, coeffs[..., 0, :]), dim=-1))
            else:
                h0 = self.initial_linear(static)

        return coeffs, h0

    def _make_outputs(self, hidden):
        """ Hidden state to output format depending on `return_sequences` and rectilinear (return every other). """
        if self.return_sequences:
            outputs = self.final_linear(hidden)

            # If rectilinear and return sequences, return every other value
            if (
                self.interpolation == "rectilinear"
            ) and self.return_filtered_rectilinear:
                outputs = outputs[:, ::2]
        else:
            outputs = self.final_linear(hidden[:, -1, :])
        return outputs

    def forward(self, inputs):
        # Handle h0 and inputs
        coeffs, h0 = self._setup_h0(inputs)

        # Make lin int
        data = self.spline(coeffs)

        # Perform the adjoint operation
        hidden = torchcde.cdeint(
            data,
            self.func,
            h0,
            data.grid_points,
            adjoint=self.adjoint,
            method=self.solver,
        )

        # Convert to outputs
        outputs = self._make_outputs(hidden)

        return outputs


class _NCDEFunc(nn.Module):
    """The function applied to the hidden state in the NCDE model.

    This creates a simple RNN-like block to be used as the computation function f in:
        dh/dt = f(h) dX/dt
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_hidden_dim=15,
        num_layers=1,
        density=0.0,
        rank=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.sparsity = density
        self.rank = rank

        # Additional layers are just hidden to hidden with relu activation
        layers = [nn.Linear(hidden_dim, hidden_hidden_dim), nn.ReLU()]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_hidden_dim, hidden_hidden_dim), nn.ReLU()]

        # Add on final layer and Tanh and build net
        layers += [nn.Linear(hidden_hidden_dim, hidden_dim * input_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, t, h):
        return self.net(h).view(-1, self.hidden_dim, self.input_dim)
