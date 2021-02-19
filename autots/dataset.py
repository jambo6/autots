""" Contains classes for packaging time series data into usable torch datasets. """
import torch


class SupervisedLearningDataset:
    """Dataset for a collection of input tensors and corresponding stratify.

    Assumes a set of inputs (tensors, stratify) where tensors can be either a list of tensors or a single 3D tensor. If
    the tensor is a list the __getitem__ method will return ((tensor[item] for tensor in tensors), stratify).

    This dataset is useful when we are following the (tensors, label) convention to train models with multiple data
    inputs (e.g. static + temporal). With this approach we force model.forward to accept a single input, the method of
    handling this input should be determined in the model class itself.
    """

    def __init__(self, tensors, labels):
        """
        Args:
            tensors (tensor or list of tensors): The data tensors.
            labels (tensor): A tensor of labels.
        """
        super(SupervisedLearningDataset, self).__init__()

        self._assert(tensors, labels)

        self.tensors = tensors
        self.labels = labels

    def _assert(self, tensors, labels):
        assert any(
            [isinstance(tensors, x) for x in [list, torch.Tensor]]
        ), "tensors must be a list or tensor"
        if isinstance(tensors, list):
            self.is_list = True
            assert [len(x) == len(labels) for x in tensors]
        else:
            self.is_list = False
            assert len(tensors) == len(labels)

    def __getitem__(self, index):
        if self.is_list:
            return [t[index] for t in self.tensors], self.labels[index]
        else:
            return self.tensors[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
