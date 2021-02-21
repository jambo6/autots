""" Methods for defining the labels. """
import numpy as np
import torch
from sklearn.base import TransformerMixin

from .misc import PadRaggedTensors


class LabelMaker(TransformerMixin):
    """Useful for setting up the labels for a given problem.

    Allows three kinds of problem ('oneshot', 'online', 'regression'). For online and regression, currently this
    class provides little functionality. However for online we provide `lookback`, `lookforward`,
    and `return_online_time` arguments (see __init__ method for explanations).

    Notes:
        - For online labels, each label sample **must** be a 2D tensor with shape (length sample i, 2) where the first
        channel is time and the second is a binary label.

    For online problems, uppose we set lookback=2, lookforward=3 and have data:
        >>> times = [0, 1.4, 2.3, 3.2, 5.8]
        >>> labels = [0, 0, 1, 0, 0]
    Then with times and labels stacked in an appropriate 2D tensor, this method will return
        >>> [0, 1, 1, 1, 0]
    as the new labels.
    """

    def __init__(
        self,
        problem="oneshot",
        lookback=0,
        lookforward=0,
        return_online_time=None,
        pad_sequence=False,
    ):
        """
        Args:
            problem (str): One of ('oneshot', 'online', 'regression').
            lookback (float): The lookback time from any 1 to still mark the patient as 1.
            lookforward (float): The lookforward time from any 1 to still mark the patient as 1.
            return_online_time (bool): If False will return just the labels in the transform method. If True will
                additionally return the times of each label occurrence, this is necessary when using
                SimplePandasPipeline.
            pad_sequence (bool): Set True to nan pad to max length in the transform.
        """
        self.problem = problem
        self.lookback = lookback
        self.lookforward = lookforward
        self.pad_sequence = pad_sequence
        self.return_online_time = return_online_time

        self._assertions()

    def _assertions(self):
        assert self.problem in [
            "oneshot",
            "online",
            "regression",
        ], "Unrecognized problem type {}".format(self.problem)
        if self.problem == "online":
            if self.return_online_time not in [True, False]:
                raise ValueError(
                    "For online problems you must specify whether to return the online time values."
                )

    def fit(self, labels, y=None):
        return self

    def get_stratification_labels(self, labels):
        """Method to get the labels to stratify on, to specify a different methods overwrite this.

        Returns:
            - 'oneshot' will return the raw labels.
            - 'online' returns a vector denoting whether there was a 1 in the entire series of each label sample.
            - 'regression' returns None.

        """
        stratification_labels = labels
        if self.problem == "online":
            stratification_labels = torch.tensor(
                [lab[:, 1].max() for lab in labels]
            ).view(-1, 1)
        elif self.problem == "regression":
            stratification_labels = None
        return stratification_labels

    def transform(self, labels):
        # Handle for online and padding
        if self.problem == "online":
            assert labels[0].size(-1) == 2, "labels must have [time, labels] channels."
            labels = _prepare_online_labels(
                labels, lookback=self.lookback, lookforward=self.lookforward
            )
            if not self.return_online_time:
                labels = [lab[:, -1] for lab in labels]
        if self.pad_sequence:
            labels = PadRaggedTensors(fill_value=float("nan")).transform(labels)
        return labels


def _prepare_online_labels(labels, lookback, lookforward):
    """ Implements lookback and lookforward for the online labels. """
    assert all(
        [lookback >= 0, lookforward >= 0]
    ), "Cannot have negative lookback/lookforward"
    assert labels[0].size(-1) == 2, "labels must have columns ['time', 'labels']."
    labels = [lab.clone() for lab in labels]
    for i in range(len(labels)):
        times_, labels_ = labels[i][:, 0], labels[i][:, 1]
        times_ = times_[~labels_.isnan()]
        one_locations = np.argwhere(labels_.numpy() == 1).reshape(-1)
        if one_locations.shape[0] > 0:
            for j in one_locations:
                time_now = times_[j].item()
                start = np.argwhere(times_.numpy() >= time_now - lookback).reshape(-1)[
                    0
                ]
                end = np.argwhere(times_.numpy() <= time_now + lookforward).reshape(-1)[
                    -1
                ]
                labels[i][start : end + 1, 1] = 1
    return labels
