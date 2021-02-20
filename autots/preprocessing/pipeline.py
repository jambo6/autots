""" Pipeline classes for transformation automation. """
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from autots.dataset import SupervisedLearningDataset

from .mixin import NullTransformer


class SimplePipeline(Pipeline):
    """This class simply extends the sklearn Pipeline class to allow for unnamed transformers to be passed.

    In the sklearn.pipeline.Pipeline class one must specify:
        >>> steps = [(name_1, transformer_1), ..., (name_n, transformer_n)]
    here we allow specification of
        >>> steps = [transformer_1, ..., transformer_n]
    and the name is inferred from the class __repr__ method.
    """

    def __init__(self, steps, verbose=False):
        """
        Args:
            steps (list): List of sklearn transformers to go into the pipeline. These do not have to be specified as
                tuples (name, class).
            verbose (bool): Set True to print out individual steps and the time taken.
        """
        steps = self._setup_steps(steps)
        super().__init__(steps, verbose=verbose)

    def __repr__(self):
        string = "Pipeline with steps:"
        for i, (name, _) in enumerate(self.steps):
            string += "\n\t{}. {}".format(i + 1, name)
        return string

    @staticmethod
    def _setup_steps(steps):
        # Gives a name to each of the steps if they are not already in tuple format
        if not isinstance(steps[0], tuple):
            steps = [(repr(s), s) for s in steps]
        return steps


class SupervisedLearningDataPipeline:
    """The data processing pipeline for supervised learning.

    This assumes the input will be of the form (tensors, stratify) where tensors is a list of an arbitrary amount of
    tensors. The class must be initialised with (tensor_pipelines, label_pipeline) where tensor_pipelines corresponds
    to an equal num_samples list to tensors and each tensor and tensor_pipeline shares a corresponding index in each
    list.
    """

    def __init__(self, tensor_pipelines, label_pipeline=None, cv=None, batch_size=None):
        """
        Args:
            tensor_pipelines (list or pipeline): A list of pipelines corresponding to multiple data source, or a single
                pipeline for a single data source.
            label_pipeline (pipeline or None): A label pipeline object or None.
            cv (cv or None): This must be a cross validation class with a split method that returns a list of
                indices. If this is not set sets self.train_indices to all indices.
            batch_size (int or None): If this is specified, will convert the split data into dataloaders in the
                transform method before outputting.
        """
        self.tensor_pipelines = tensor_pipelines
        self.label_pipeline = label_pipeline
        self.cv = cv
        self.batch_size = batch_size

        # Make label pipeline be a null transformer to avoid if statements later
        if label_pipeline is None:
            self.label_pipeline = NullTransformer()

        self.num_data_pipelines = (
            len(self.tensor_pipelines) if isinstance(self.tensor_pipelines, list) else 1
        )
        self.is_fitted = False
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def __repr__(self):
        # Print information on the contained pipelines
        string = "{} data pipelines:".format(self.num_data_pipelines)
        for i, pipeline in enumerate(self.tensor_pipelines):
            string += "\n{}. {}".format(i + 1, repr(pipeline))
        if self.label_pipeline:
            string += "\nand a label pipeline:\n\t{}".format(repr(self.label_pipeline))
        return string

    @property
    def indices(self):
        # Return the indices if any exist, otherwise return none
        indices = [
            x
            for x in [self.train_indices, self.val_indices, self.test_indices]
            if x is not None
        ]
        if len(indices) == 0:
            indices = None
        return indices

    def _generate_indices(self, labels):
        """ Splits the data, if no cv is set train indices becomes all indices. """
        # Method to get stratification laebsl if one exists
        stratification_labels = labels
        if hasattr(self.label_pipeline, "get_stratification_labels"):
            stratification_labels = self.label_pipeline.get_stratification_labels(
                labels
            )

        if self.cv:
            indices = self.cv.split(len(labels), stratification_labels)
            if len(indices) == 2:
                self.train_indices, self.test_indices = indices
            else:
                self.train_indices, self.val_indices, self.test_indices = indices
        else:
            self.train_indices = torch.arange(len(labels))

    def fit(self, tensors, labels):
        # Split train/val/test
        self._generate_indices(labels)

        # Fit the data
        if self.num_data_pipelines > 1:
            for tensor, pipeline in zip(tensors, self.tensor_pipelines):
                pipeline.fit(tensor[self.train_indices])
        else:
            self.tensor_pipelines.fit(tensors[self.train_indices])

        # Assert fitted
        self.is_fitted = True

        return self

    def _to_dataloader(self, tensors, labels):
        # Convert into a dataloader
        dataset = SupervisedLearningDataset(tensors, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def transform(self, tensors, labels):
        assert self.is_fitted, "Class has not been fit."
        assert self.indices, "No indices are set, check the CV method."

        # Transform data
        outputs = []
        for indices in self.indices:
            # Transform the tensors for the given indices
            if self.num_data_pipelines > 1:
                split_tensors = []
                for i, (tensor, pipeline) in enumerate(
                    zip(tensors, self.tensor_pipelines)
                ):
                    split_tensors.append(pipeline.transform(tensor[indices]))
            else:
                split_tensors = self.tensor_pipelines.transform(tensors[indices])

            # Transform the labels
            split_labels = labels[indices]
            split_labels = self.label_pipeline.transform(split_labels)

            outputs.append([split_tensors, split_labels])

        # Further processing
        if self.batch_size is not None:
            outputs = [
                self._to_dataloader(tensors, labels) for tensors, labels in outputs
            ]
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def fit_transform(self, tensors, labels):
        """ Differs from sklearn fit_transform as labels must be passed to the fit method. """
        self.fit(tensors, labels)
        return self.transform(tensors, labels)
