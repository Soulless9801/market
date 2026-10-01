"""Architecture checks; shared pipeline and transport tests cover checkpoint use."""

import pytest
import torch
from torch import nn

from ml.data.dataset import MarketDataset
from ml.models import build_model, configure_model, model_names
from ml.utils.reproducibility import seed_everything


@pytest.mark.parametrize("layout", ["cnn", "mlp"])
def test_cnn_derives_dimensions_without_coupling_to_layout_name(exports, layout):
    metadata = MarketDataset(exports / layout).metadata
    config = configure_model("cnn", metadata, {})
    assert {"cnn", "mlp"} <= set(model_names())
    seed_everything(42)
    model = build_model("cnn", 1, config).eval()
    seed_everything(42)
    repeated = build_model("cnn", 1, config).eval()
    inputs = torch.randn(4, metadata["inputShape"][0])
    outputs = model(inputs)
    assert outputs.shape == (4, metadata["numClasses"])
    assert torch.isfinite(outputs).all()
    assert torch.equal(outputs, repeated(inputs))
    assert sum(p.numel() for p in model.parameters()) < 1000

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    nn.CrossEntropyLoss()(outputs, torch.tensor([0, 1, 2, 0])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    # Both the temporal filters and classification head must participate in learning.
    assert not torch.equal(before["convolutions.0.weight"], model.convolutions[0].weight)
    assert not torch.equal(before["classifier.3.weight"], model.classifier[3].weight)


def test_cnn_preserves_feature_order_and_dropout_is_disabled_for_inference():
    config = configure_model("cnn", {"inputShape": [7], "numClasses": 5}, {"dropout": 0.5})
    seed_everything(42)
    model = build_model("cnn", 1, config)
    inputs = torch.arange(56, dtype=torch.float32).reshape(8, 7)
    observed = []
    hook = model.convolutions[0].register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    first, second = model(inputs), model(inputs)
    hook.remove()
    assert first.shape == (8, 5)
    assert observed[0].shape == (8, 1, 7)
    assert torch.equal(observed[0][:, 0, :], inputs)
    assert not torch.equal(first, second)
    model.eval()
    assert torch.equal(model(inputs), model(inputs))
    # Valid narrow layouts and singleton batches need no special trainer handling.
    narrow = build_model(
        "cnn", 1, configure_model("cnn", {"inputShape": [1], "numClasses": 2}, {})
    ).eval()
    assert narrow(torch.ones(1, 1)).shape == (1, 2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"channels": []}, {"channels": [0]}, {"channels": [True]}, {"channels": "8"},
        {"kernel_size": 0}, {"kernel_size": 2}, {"kernel_size": True},
        {"pool_size": 0}, {"pool_size": 31}, {"pool_size": 1.5},
        {"dropout": -0.1}, {"dropout": 1}, {"dropout": True}, {"dropout": "0.2"},
        {"dropout": float("nan")}, {"dropout": float("inf")},
        {"input_shape": [30, 1]}, {"input_shape": [0]}, {"num_classes": 1},
        {"extra": 1},
    ],
)
def test_cnn_rejects_invalid_saved_configuration(overrides):
    config = configure_model("cnn", {"inputShape": [30], "numClasses": 3}, {})
    with pytest.raises(ValueError):
        build_model("cnn", 1, {**config, **overrides})


def test_cnn_rejects_input_shape_overrides_and_wrong_tensor_dimensions():
    metadata = {"inputShape": [30], "numClasses": 3}
    with pytest.raises(ValueError, match="Unknown CNN options"):
        configure_model("cnn", metadata, {"input_shape": [40]})
    model = build_model("cnn", 1, configure_model("cnn", metadata, {}))
    for shape in ((30,), (2, 29), (2, 1, 30)):
        with pytest.raises(ValueError, match="Expected"):
            model(torch.zeros(shape))
