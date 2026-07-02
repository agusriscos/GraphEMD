"""
Script to load the trained GraphEMD model and run it locally on a validation dataloader.

Uses project libraries (GraphEMD.conf, GraphEMD.data, GraphEMD.model) to build
the model, load weights from the checkpoint, and evaluate on VisualGraphDataloader.
"""
import os
from pathlib import Path

import torch

from GraphEMD.conf import (
    DATA_PATH,
    GraphEncoderConfig,
    InverseFunnelDecoderConfig,
    TransformConfig,
)
from GraphEMD.conf.paths import REPO_ROOT
from GraphEMD.data import VisualGraphDataloader
from GraphEMD.model import GraphEMDAutoEncoder, LightningAutoEncoder


# Paths and model name (consistent with train.py)
CKPT_DIR = str(REPO_ROOT / "outputs" / "autoencoder" / "ckpt")
MODEL_NAME = "GraphEMDAutoEncoder_251115"
CKPT_SUFFIX = "best_state_dict"


def get_checkpoint_path(
    ckpt_dir: str = CKPT_DIR,
    model_name: str = MODEL_NAME,
    suffix: str = CKPT_SUFFIX,
) -> str:
    """
    Return the path to the checkpoint .pt file.

    Parameters
    ----------
    ckpt_dir : str
        Directory where checkpoints were saved.
    model_name : str
        Model name (filename without extension).
    suffix : str
        Checkpoint suffix (e.g. best_state_dict).

    Returns
    -------
    str
        Absolute path to the .pt file.

    Examples
    --------
    >>> path = get_checkpoint_path()
    >>> os.path.isfile(path)
    True
    """
    filename = f"{model_name}_{suffix}.pt"
    return os.path.join(ckpt_dir, filename)


def build_dataloader(
    data_dir: str,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    max_num_nodes: int | None = None,
    num_workers: int = 0,
) -> VisualGraphDataloader:
    """
    Build a VisualGraphDataloader with project configuration.

    Parameters
    ----------
    data_dir : str
        Data directory (e.g. val_data or train_data).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the data.
    drop_last : bool
        Whether to drop the last incomplete batch.
    max_num_nodes : int, optional
        Maximum number of nodes per graph. If None, uses TransformConfig.
    num_workers : int
        Number of DataLoader workers.

    Returns
    -------
    VisualGraphDataloader
        Configured dataloader.
    """
    mn = (
        max_num_nodes
        if max_num_nodes is not None
        else TransformConfig.to_dict()["MAX_WINDOW_SIZE"]
    )
    return VisualGraphDataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        max_num_nodes=mn,
        num_workers=num_workers,
    )


def load_trained_model(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> LightningAutoEncoder:
    """
    Load the LightningAutoEncoder model with checkpoint weights.

    The checkpoint saved by PtModelCheckpoint is a state_dict of the LightningModule.
    Instantiate LightningAutoEncoder with the same configuration as in train.py
    and load weights from the .pt file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint .pt file.
    device : torch.device, optional
        Device (cuda/cpu). Defaults to automatic selection.

    Returns
    -------
    LightningAutoEncoder
        Model with loaded weights, in eval mode.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at checkpoint_path.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = GraphEncoderConfig.to_dict()
    decoder_config = InverseFunnelDecoderConfig.to_dict()
    model_configuration = {
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
    }
    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    lightning_model = LightningAutoEncoder(
        model_gen=GraphEMDAutoEncoder,
        model_configuration=model_configuration,
        criterion=criterion,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    lightning_model.load_state_dict(state_dict, strict=True)
    lightning_model.to(device)
    lightning_model.eval()
    return lightning_model


def run_validation(
    model: LightningAutoEncoder,
    dataloader: VisualGraphDataloader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Run model validation on the dataloader and return mean loss.

    Parameters
    ----------
    model : LightningAutoEncoder
        Model in eval mode.
    dataloader : VisualGraphDataloader
        Validation dataloader.
    device : torch.device
        Device where the model resides.
    max_batches : int, optional
        Maximum number of batches to evaluate. None = all.

    Returns
    -------
    float
        Mean loss (MSE) over evaluated batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    max_num_nodes = TransformConfig.to_dict()["MAX_WINDOW_SIZE"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            y = batch.x.reshape(-1, int(batch.num_nodes / batch.num_graphs))
            y_filter = y[y != 0]
            output_filter = output[y != 0]
            loss = torch.nn.functional.mse_loss(output_filter, y_filter, reduction="sum")
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main() -> None:
    """
    Entry point: load the model, build the dataloader, and run local validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    checkpoint_path = get_checkpoint_path()
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, device=device)

    val_data_dir = os.path.join(DATA_PATH, "val_data")
    max_num_nodes = TransformConfig.to_dict()["MAX_WINDOW_SIZE"]
    val_dataloader = build_dataloader(
        data_dir=val_data_dir,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        max_num_nodes=max_num_nodes,
        num_workers=0,
    )

    print("Running validation (first 10 batches)...")
    mean_loss = run_validation(model, val_dataloader, device, max_batches=10)
    print(f"Mean loss (MSE) on validation: {mean_loss:.6f}")

    # Example batch as in run_autoencoder.py
    batch = next(iter(val_dataloader))
    num_time_samples = batch.x.shape[0]
    print(f"Number of samples in the first batch: {num_time_samples}")

    with torch.no_grad():
        batch = batch.to(device)
        output = model(batch)
    x = batch.x.reshape(-1, max_num_nodes)
    x_filter = x[x != 0]
    output_filter = output[x != 0]
    loss_fn = torch.nn.MSELoss()
    loss_one_batch = loss_fn(output_filter, x_filter)
    print(f"Loss on one example batch: {loss_one_batch.item():.6f}")
    print("Listo.")


if __name__ == "__main__":
    main()
