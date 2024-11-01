import click
import torch
import segmentation_models_pytorch as smp

from utils.other import set_seed
from utils.train import train_model

set_seed()


@click.command()
@click.option("--n_epoch", default=50)
@click.option("--batch_size", default=2)
def train(n_epoch, batch_size):

    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=4,
    )
    train_model(model, n_epoch, batch_size)
    model.load_state_dict(torch.load('models/best_model.pth'))

    torch.onnx.export(
        model,
        torch.rand((1, 3, 384, 384)),
        "models/segm_model.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
    )


if __name__ == '__main__':
    train()
