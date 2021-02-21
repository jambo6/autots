from autots import preprocessing
from autots.models import RNN
from autots.tests import helpers
from autots.utils import make_time_series_problem

helpers.set_seed(0)


def test_label_maker_in_pipeline():
    # Check label maker works as expected for online labels in a full pipeline
    input_dim = 3
    output_dim = 1
    static_dim = None
    data, labels = make_time_series_problem(
        n_channels=input_dim,
        n_classes=output_dim + 1,
        problem="online",
        static_dim=static_dim,
        ragged=True,
    )

    # Setup pipelines
    tensor_pipelines = preprocessing.SimplePipeline(
        steps=[
            preprocessing.PadRaggedTensors(),
            preprocessing.SimpleImputer(strategy="constant", fill_value=0.0),
        ]
    )
    label_pipeline = preprocessing.LabelMaker(
        problem="online",
        lookback=2,
        lookforward=3,
        return_online_time=False,
        pad_sequence=True,
    )

    # CV and main pipe
    cv = preprocessing.TrainValTestSplit(stratify=True)
    pipeline = preprocessing.SupervisedLearningDataPipeline(
        tensor_pipelines=tensor_pipelines, label_pipeline=label_pipeline, cv=cv
    )

    (train_data, train_labels), _, _ = pipeline.fit_transform(data, labels)
    model = RNN(input_dim, 5, output_dim, static_dim, return_sequences=True)
    _, acc = helpers.training_loop(model, train_data, train_labels, loss_str="nan_bce")
    assert 0. < acc <= 1.0
