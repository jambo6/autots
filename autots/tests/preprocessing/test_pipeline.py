import numpy as np
import pandas as pd
import pytest
import torch

from autots import preprocessing
from autots.utils import make_time_series_problem


def test_simple_pipeline():
    # Check it runs with non names steps
    data, _ = make_time_series_problem(static_dim=None, masking=True)
    pipeline = preprocessing.SimplePipeline(
        steps=[
            preprocessing.PadRaggedTensors(),
            preprocessing.TensorScaler(),
            preprocessing.ForwardFill(),
        ]
    )
    assert not torch.any(torch.isnan(pipeline.fit_transform(data)))


@pytest.mark.parametrize("val_frac, batch_size", [(0.1, None), (None, 64)])
def test_supervised_learning_pipeline(val_frac, batch_size):
    # Data and pipelines
    tensors, labels = make_time_series_problem()
    static_pipeline = preprocessing.SimplePipeline(
        [preprocessing.SimpleImputer(strategy="constant", fill_value=0.0)]
    )
    temporal_pipeline = preprocessing.SimplePipeline(
        [preprocessing.SimpleImputer(strategy="constant", fill_value=0.0)]
    )

    # SL pipeline
    cv = preprocessing.split.TrainValTestSplit(
        val_frac=val_frac, test_frac=0.15, shuffle=True
    )
    pipeline = preprocessing.SupervisedLearningDataPipeline(
        tensor_pipelines=[static_pipeline, temporal_pipeline],
        label_pipeline=None,
        cv=cv,
        batch_size=batch_size,
    )
    outputs = pipeline.fit_transform(tensors, labels)

    # Just ensure correct number of outputs
    assert len(outputs) == 3 if val_frac is not None else 2


def test_simple_pandas_pipeline():
    # Test pandas pipeline returns dataframes for static, temporal, labels, online labels input dataframes
    frames, labels = make_time_series_problem(pandas_format=True, ragged=True)
    _, labels_online = make_time_series_problem(
        pandas_format=True, ragged=True, problem="online"
    )

    # Setup pipelines
    impute = preprocessing.SimpleImputer(strategy="constant", fill_value=0.0)

    def wrap(data_type, steps):
        return preprocessing.SimplePandasPipeline(data_type, steps)

    static_pipeline = wrap("static", [impute])
    temporal_pipeline = wrap("temporal", [impute])
    labels_pipeline = wrap("static", [preprocessing.LabelMaker(problem="oneshot")])
    labels_online_pipeline = wrap(
        "temporal",
        [preprocessing.LabelMaker(problem="online", return_online_time=True)],
    )

    # Check we return the desired types
    for frame, pipeline in [
        (frames[0], static_pipeline),
        (frames[1], temporal_pipeline),
        (labels, labels_pipeline),
        (labels_online, labels_online_pipeline),
    ]:
        out = pipeline.fit_transform(frame)
        assert any([isinstance(out, pd.DataFrame), isinstance(out, pd.Series)])


def test_pandas_pipeline_returns_expected_labels():
    # Make sure the pandas pipeline returns the expected format of the labels
    _, labels = make_time_series_problem(
        problem="online", irregular=True, pandas_format=True, ragged=True, length=30
    )
    labels = labels.sort_values(["id", "time"])

    # Get expected values
    for lookback, lookforward in [(0, 0), (1, 2), (6, 3)]:
        expected_labels = labels.copy()
        for id_ in labels["id"].unique():
            id_frame = expected_labels[expected_labels["id"] == id_]
            times_, labels_ = (
                id_frame["time"].values,
                id_frame[id_frame.columns[-1]].values,
            )
            new_labels = labels_.copy()
            for time, label in zip(times_, labels_):
                if label == 1:
                    start = np.argwhere(times_ >= time - lookback).reshape(-1)[0]
                    end = np.argwhere(times_ <= time + lookforward).reshape(-1)[-1]
                    new_labels[start : end + 1] = 1
            expected_labels.loc[expected_labels["id"] == id_, "label"] = new_labels

        # Run pipeline and check
        pipeline = preprocessing.SimplePandasPipeline(
            "temporal",
            steps=[
                preprocessing.LabelMaker(
                    problem="online",
                    lookback=lookback,
                    lookforward=lookforward,
                    return_online_time=True,
                )
            ],
        )
        output_labels = pipeline.fit_transform(labels)
        assert (output_labels["label"] != expected_labels["label"]).sum() == 0
