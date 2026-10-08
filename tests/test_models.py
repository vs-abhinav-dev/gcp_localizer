import torch
from gcp_localizer.models.localizer import GCPLocalizer
from gcp_localizer.models.classifier import GCPClassifier


def test_localizer_forward():
    model = GCPLocalizer(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()


def test_classifier_forward():
    binary_model = GCPClassifier(num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = binary_model(dummy_input)

    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()

    shape_model = GCPClassifier(num_classes=3, pretrained=False)
    shape_output = shape_model(dummy_input)
    assert shape_output.shape == (2, 3)
