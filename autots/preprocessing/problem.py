""" Methods for defining the labels. """
import numpy as np
import torch
from sklearn.base import TransformerMixin

from .misc import PadRaggedTensors


class LabelMaker(TransformerMixin):
    """ Builds the labels. """

    def __init__(
        self, problem="oneshot", lookback=0, lookforward=0, pad_sequence=False
    ):
        self.problem = problem
        self.lookback = lookback
        self.lookforward = lookforward
        self.pad_sequence = pad_sequence

        self._assertions()

    def _assertions(self):
        assert self.problem in [
            "oneshot",
            "online",
            "regression",
        ], "Unrecognized problem type {}".format(self.problem)

        if self.problem != "online":
            if not all([x == 0 for x in [self.lookback, self.lookforward]]):
                raise NotImplementedError(
                    "lookback/lookforward/mask_after_first_occurrence only for problem type online"
                )

    def fit(self, labels):
        return self

    def get_stratification_labels(self, labels):
        """ Method to return the labels to stratify on, uses max label per sample online. """
        stratification_labels = labels
        if self.problem == "online":
            stratification_labels = torch.tensor([lab.max() for lab in labels]).view(
                -1, 1
            )
        elif self.problem == "regression":
            stratification_labels = None
        return stratification_labels

    def transform(self, labels):
        # Handle for online and padding
        if self.problem == "online":
            labels = _create_online_labels(
                labels, lookback=self.lookback, lookforward=self.lookforward
            )
        if self.pad_sequence:
            labels = PadRaggedTensors(fill_value=float("nan")).transform(labels)
        return labels


def _create_online_labels(labels, lookback, lookforward):
    """ Loop over the ids and fill in labels according to the parameters. """
    assert all([not torch.isnan(x).any() for x in labels])
    labels = [lab.clone() for lab in labels]
    for i in range(len(labels)):
        labels_ = labels[i].clone()
        one_locations = np.argwhere(labels_.numpy() == 1).reshape(-1)
        if one_locations.shape[0] > 0:
            for j in one_locations:
                start = max(0, j - lookback)
                end = min(j + lookforward + 1, len(labels_))
                labels_[start:end] = 1
        labels[i] = labels_
    return labels
