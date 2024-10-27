import torch
import time
from .losses import ComboLoss
from .dataset import get_dataloaders


def dice_channels(
    prob: torch.Tensor, truth: torch.Tensor,
    threshold: float = 0.5, eps: float = 1e-9
) -> torch.Tensor:
    """_summary_

    Args:
        prob (torch.Tensor): Predicted probabilities (N, C, H, W).
        truth (torch.Tensor): Ground truth binary values (N, C, H, W).
        threshold (float, optional): Threshold to binarize `prob`.
            Defaults to 0.5.
        eps (float, optional): Small value to prevent division by zero.
            Defaults to 1E-9.

    Returns:
        torch.Tensor: Mean Dice coefficient for each channel (C,).
    """
    num_imgs = prob.size(0)
    num_channels = prob.size(1)
    prob = (prob > threshold).float()
    truth = (truth > 0.5).float()

    prob = prob.view(num_imgs, num_channels, -1)
    truth = truth.view(num_imgs, num_channels, -1)

    intersection = prob * truth
    score = (2.0 * intersection.sum(2) + eps) /\
        (prob.sum(2) + truth.sum(2) + eps)
    score[score >= 1] = 1

    return score.mean()


def train_epoch(
    loader: torch.utils.data.DataLoader, model,
    loss_function_seg, optimizer, device
) -> torch.Tensor:
    """Train the model for one epoch.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (_type_): The model to train.
        loss_function_seg (_type_): Loss function for segmentation.
        optimizer (_type_): Optimizer for model parameters.

    Returns:
        torch.Tensor:  Average loss for the epoch.
    """
    model.train()
    avg_loss = 0.0
    optimizer.zero_grad()

    for i, (image, mask) in enumerate(loader):
        x = image.to(device)
        y = mask.to(device)
        prediction_seg = model(x)

        loss = loss_function_seg(prediction_seg, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item()

    avg_loss /= i + 1
    return avg_loss


def valid_epoch(loader, model, device):
    """Validate the model for one epoch.

    Args:
        loader (_type_): DataLoader for validation data.
        model (_type_): The model to evaluate.
        device (_type_):  Device to perform computations (CPU or GPU).

    Returns:
        _type_:  Average Dice score for the epoch.
    """
    model = model.to(device)
    model.eval()
    scores = []

    with torch.no_grad():
        for image, mask in loader:
            x = image.to(device)
            y = mask.to(device)
            probs = torch.sigmoid(model(x))
            scores.append(dice_channels(probs, y))

    return torch.stack(scores).mean().item()


def train_model(model, n_epoch, batch_size=1):

    def lambdalr(epoch):
        return (1 - epoch / n_epoch) ** 0.9

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loss_fn = ComboLoss(
        weights={"bce": 0.9, "dice": 0.1},
        channel_weights=[1] * 4
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdalr)

    train_loader, valid_loader = get_dataloaders(batch_size)

    best_score = 0
    t0 = time.time()
    print("start train")
    for epoch in range(n_epoch):
        train_loss = train_epoch(train_loader, model,
                                 loss_fn, optimizer, device)
        valid_score = valid_epoch(valid_loader, model, device)
        scheduler.step()
        if valid_score > best_score:
            torch.save(model.state_dict(), "models/best_model.pth")
            best_score = valid_score
        d_t = time.time() - t0
        t0 = time.time()
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}, train_loss:{train_loss:.3f}, "
                + f"val_score: {valid_score:.3f}, time: {d_t:.0f}, "
                + f"lr: {scheduler.get_last_lr()[0]:.6f})"
            )
    return
