import pytest
import torch
from autots.utils import make_time_series_problem
from autots import preprocessing


def test_simple_pipeline():
    # Check it runs with non names steps
    data, _ = make_time_series_problem(static_dim=None, masking=True)
    pipeline = preprocessing.SimplePipeline(
        steps=[preprocessing.PadRaggedTensors(), preprocessing.TensorScaler(), preprocessing.ForwardFill()]
    )
    assert not torch.any(torch.isnan(pipeline.fit_transform(data)))


@pytest.mark.parametrize(
    "val_frac, batch_size", [(0.1, None), (None, 64)]
)
def test_supervised_learning_pipeline(val_frac, batch_size):
    # Data and pipelines
    tensors, labels = make_time_series_problem()
    static_pipeline = preprocessing.SimplePipeline([preprocessing.SimpleImputer(strategy="constant", fill_value=0.0)])
    temporal_pipeline = preprocessing.SimplePipeline([preprocessing.SimpleImputer(strategy="constant", fill_value=0.0)])

    # SL pipeline
    cv = preprocessing.split.TrainValTestSplit(val_frac=val_frac, test_frac=0.15, shuffle=True)
    pipeline = preprocessing.SupervisedLearningDataPipeline(
        tensor_pipelines=[static_pipeline, temporal_pipeline], label_pipeline=None, cv=cv, batch_size=batch_size
    )
    outputs = pipeline.fit_transform(tensors, labels)

    # Just ensure correct number of outputs
    assert len(outputs) == 3 if val_frac is not None else 2

