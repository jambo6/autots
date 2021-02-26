import torch


def rolling_window(x, dimension, window_size, step_size=1, return_same_size=True):
    """Outputs an expanded tensor to perform rolling window operations on a pytorch tensor.

    Given an input tensor of shape (batch, length, channels), a window size, and a step size, outputs a tensor of shape
    (batch, length - step_size, channels, window_size) where the final dimension contains the time steps from that index
    to window_size num time points before.

    Notes:
        - The number of values changes from (batch * length * channels) -> (batch * (length / step) * channels * window)
        which can cause memory issues for a large tensor
        - The first time steps will contain nans as there are insufficient numbers of time points previously. If
        statistics are desired over these times, the function must be able to handle nan values.

    Args:
        x (tensor): Tensor of shape (batch, length, channels).
        dimension (int): Dimension to open.
        window_size (int): Length of the rolling window.
        step_size (int): Window step, defaults to 1.
        return_same_size (bool): Set True to return a tensor of the same size as the input tensor with nan values filled
            where insufficient prior window lengths existed. Otherwise returns a reduced size tensor from the paths
            that had sufficient data.

    Returns:
        A tensor of shape (batch, length / step_size, channels, window_size) where the final dim contains the rolling
        window of time values.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if return_same_size:
        x_dims = list(x.size())
        x_dims[dimension] = window_size - 1
        nans = float("nan") * torch.zeros(x_dims)
        x = torch.cat((nans, x), dim=dimension)

    # Unfold ready for mean calculations
    unfolded = x.unfold(dimension, window_size, step_size)

    return unfolded
