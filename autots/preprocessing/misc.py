""" Miscellaneous transformers. """
import torch
from sklearn.base import TransformerMixin
from torch.nn.utils.rnn import pad_sequence


class PadRaggedTensors(TransformerMixin):
    """Converts a list of unequal num_samples tensors (or arrays) to a stacked tensor.

    If the input is a 3D tensor, reduces the num_samples of the sequence and ignores the padding step. If the object is
    a list of tensors (or numpy object), reduces the sequence num_samples then pads.
    """

    def __init__(self, fill_value=float("nan"), max_seq_len=None):
        """
        Args:
            fill_value (float): Value to fill if an array is extended.
            max_seq_len (int): Maximum num_samples of the output sequence.
        """
        self.fill_value = fill_value
        self.max_seq_len = max_seq_len

    def __repr__(self):
        return "Pad ragged tensors"

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # If already a 3D tensor just perform num_samples reduction
        if isinstance(data, torch.Tensor):
            assert data.dim() == 3
            output = data[:, : self.max_seq_len]
        # Otherwise reduce num_samples and pad
        else:
            if not isinstance(data[0], torch.Tensor):
                data = [torch.tensor(t)[: self.max_seq_len] for t in data]
            output = pad_sequence(data, batch_first=True, padding_value=self.fill_value)
        return output
