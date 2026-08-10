import pytest


def test_cnn_output_shape_and_normalization():
    tf = pytest.importorskip("tensorflow")
    from src.models.cnn import build_cnn

    model = build_cnn(num_classes=36, image_size=32, dropout=0.2)
    assert model.output_shape == (None, 36)
    assert isinstance(model.get_layer("rescale"), tf.keras.layers.Rescaling)


def test_model_factory_supports_cnn():
    pytest.importorskip("tensorflow")
    from src.models.transfer import build_transfer_model

    model = build_transfer_model("CNN", num_classes=3, image_size=32, dropout=0.2)
