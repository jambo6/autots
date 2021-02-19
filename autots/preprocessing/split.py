""" Methods for train/val/test splitting the data. """
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def train_val_test_split(
    tensors,
    val_frac=0.15,
    test_frac=0.15,
    stratify_idx=None,
    shuffle=True,
    random_state=None,
):
    """Train test split method for an arbitrary number of tensors.

    Given a list of tensor in the variable `tensors`, splits each randomly into (train_frac, val_frac, test_frac)
    proportions.

    Arguments:
        tensors (list): A list of torch tensors.
        val_frac (float): The fraction to use as validation data.
        test_frac (float): The fraction to use as test data.
        stratify_idx (int): The index of the `tensors` variable to use as stratification stratify.
        shuffle (bool): Set True to shuffle first.
        random_state (int): Random seed.

    Returns:
        A tuple containing three lists corresponding to the train/val/test split of `tensors`.
    """
    # Set random_state
    if random_state is not None:
        np.random.seed(random_state)

    # Check all tensors have the same num_samples
    num_samples = tensors[0].size(0)
    assert [t.size(0) == num_samples for t in tensors]

    # Stratification stratify
    stratification_labels = None
    if stratify_idx is not None:
        stratification_labels = (
            tensors[stratify_idx] if tensors[stratify_idx].dim() <= 2 else None
        )

    # Get a train+val/test split followed by a train/val split.
    train_val_data, test_data = tensor_train_test_split(
        tensors, test_frac, stratify=stratification_labels, shuffle=shuffle
    )

    # Split out train and val
    if stratify_idx is not None:
        stratification_labels = train_val_data[stratify_idx]
    new_test_frac = val_frac / (1 - test_frac)
    train_data, val_data = tensor_train_test_split(
        train_val_data, new_test_frac, stratify=stratification_labels, shuffle=shuffle
    )

    # Return either of the indices or the full tensors
    tensors = [train_data, val_data, test_data]

    return tensors


def tensor_train_test_split(
    tensors, test_frac, stratify=None, shuffle=True, random_state=None
):
    """Splits a list of tensors into two parts according to the test_frac.

    Arguments:
        tensors (list): A list of tensors.
        test_frac (float): The fraction the test set.
        stratify (tensor): Stratification stratify.
        shuffle (bool): Set True to shuffle first.
        random_state (int): Random state.

    Returns:
        Two lists, the first list contains the training tensors, the second the test tensors.
    """
    split_tensors = train_test_split(
        *tensors,
        stratify=stratify,
        shuffle=shuffle,
        test_size=test_frac,
        random_state=random_state
    )
    return split_tensors[0::2], split_tensors[1::2]


class TrainValTestSplit:
    """ Much like sklearns functions, but also allows for validation splits. """

    def __init__(
        self,
        val_frac=0.15,
        test_frac=0.15,
        stratify=False,
        shuffle=True,
        random_state=None,
    ):
        """
        Args:
            val_frac (float or None): Float in (0, 1) to determine the proportion of validation data. None will
                result in train/test split only.
            test_frac (float): Float in (0, 1) to determine the proportion of test data.
            stratify (bool): Set True to allow stratification labels.
            shuffle:
            random_state:
        """
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.stratify = stratify
        self.shuffle = shuffle
        self.random_state = random_state

        # Use None rather than 0
        if val_frac == 0:
            self.val_frac = None

        self._assertions()

    def _assertions(self):
        assert self.test_frac is not None, "test_frac cannot be None"
        assert 0 < self.test_frac < 1, "test_frac must be in (0, 1)"
        if self.val_frac is not None:
            assert 0 < self.val_frac < 1, "val_frac must be in (0, 1)"
            assert (
                self.val_frac + self.test_frac < 1
            ), "val_frac and test_frac must sum less than 1"

    def split(self, num_samples, stratification_tensor=None):
        """Generate train/(val)/test indices.

        Args:
            num_samples (int): Number of samples, should be the same length as labels if specified.
            stratification_tensor (tensor or None): If this is provided, and stratify is set to True in initialisation,
                then will split using stratification_tensor as the stratify labels. NOTE: This will only work if
                stratify were set to True in the model construction.

        Todo:
            - Logging warning if stratify not set but stratification tensor is given.

        Returns:
            A list of indices referring to [train_indices, test_indices] or [train_indices, val_indices, test_indices]
            depending on whether val_frac is specified.
        """
        assert isinstance(
            num_samples, int
        ), "num_samples of the data must be inserted as an integer"

        # Both self.stratify and stratification tensor must be set
        all_indices = torch.arange(num_samples).view(-1, 1)
        tensors, stratify_index, = (
            ([all_indices, stratification_tensor], 1)
            if stratification_tensor is not None
            else ([all_indices], None)
        )

        # Split
        if self.val_frac is not None:
            split_indices = train_val_test_split(
                tensors,
                self.val_frac,
                self.test_frac,
                stratify_idx=stratify_index,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            split_indices = tensor_train_test_split(
                tensors,
                self.test_frac,
                stratify=tensors[stratify_index] if stratify_index else None,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )

        # Get indices only
        split_indices = [s[0].reshape(-1) for s in split_indices]

        return split_indices
