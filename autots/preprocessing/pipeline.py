""" Pipeline classes for transformation automation. """
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from autots.dataset import SupervisedLearningDataset

from .impute import ForwardFill
from .misc import PadRaggedTensors
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


class SimplePandasPipeline:
    """Mimics SimplePipeline but can be used on pandas format data.

    Given temporal or static pandas data and a transformation pipeline, first converts the data onto ragged tensor
    format, then applies the pipeline

    Notes:
        - For temporal data, automatically pads the data and forward fills the time value. The forward filling of time
        enables the method to reduce the padded tensors onto the original times after transformations.
        - When using LabelMaker with online labels, the 'return_online_times' flag **must** be set to True so that the
        method can reconstruct the labels with the relevant times after the transform.
    """

    def __init__(self, data_type, steps, verbose=False):
        """
        Args:
            data_type (str): One of ('static', 'temporal').
            steps (list): List of transformers.
            verbose (bool): Set True for verbose pipeline output.
        """
        assert data_type in [
            "static",
            "temporal",
        ], "data_type must be one of static or temporal."
        self.data_type = data_type

        steps = self.additional_setup(steps)
        self.pipeline = SimplePipeline(steps, verbose)

    def additional_setup(self, steps):
        initial_steps = []
        if self.data_type == "temporal":
            initial_steps = [
                PadRaggedTensors(fill_value=float("nan")),
                ForwardFill(channel_indices=[0]),
            ]
        steps = initial_steps + steps
        return steps

    def pre_convert(self, frame):
        """ Convert temporal pandas to list of tensors. """
        if self.data_type == "static":
            ids = list(frame.index)
            data = torch.tensor(frame.values)
        else:
            assert list(frame.columns[0:2]) == ["id", "time"], (
                "Temporal frame must have id and time as the first two " "columns."
            )
            ids = frame["id"].unique()
            data = [
                torch.tensor(frame[frame["id"] == id_].values[:, 1:]) for id_ in ids
            ]
        return data, ids

    def post_convert(self, transform_outputs, ids, original_frame):
        # Remove up to the max time
        if self.data_type == "static":
            if isinstance(original_frame, pd.Series):
                frame = pd.Series(index=ids, data=transform_outputs.numpy())
            else:
                frame = pd.DataFrame(
                    index=ids,
                    data=transform_outputs.numpy(),
                    columns=original_frame.columns,
                )
        else:
            tensor_list = [
                t[: np.argwhere(t[..., 0] == t[..., 0].max()).reshape(-1)[0] + 1]
                for t in transform_outputs
            ]
            flist = [
                pd.DataFrame(index=[id_] * len(t), data=t.numpy()).reset_index()
                for id_, t in zip(ids, tensor_list)
            ]
            frame = pd.concat(flist, axis=0)
            frame.columns = original_frame.columns
        return frame

    def fit(self, frame, labels=None):
        data, _ = self.pre_convert(frame)
        return self.pipeline.fit(data, labels)

    def transform(self, frame):
        data, ids = self.pre_convert(frame)
        transform_outputs = self.pipeline.transform(data)
        outputs = self.post_convert(transform_outputs, ids, frame)
        return outputs

    def fit_transform(self, frame, labels=None):
        self.fit(frame, labels)
        return self.transform(frame)


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
