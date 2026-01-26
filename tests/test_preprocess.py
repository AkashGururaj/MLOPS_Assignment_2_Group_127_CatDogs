import torch
from src.data.preprocess import prepare_data

def test_prepare_data_loaders():
    train_loader, val_loader, test_loader = prepare_data(batch_size=2, augment=False)
    images, labels = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.shape[1:] == (3, 224, 224)
    assert ((labels == 0) | (labels == 1)).all()
