"""
Script para cargar el modelo GraphEMD entrenado y ejecutarlo en local sobre un dataloader de validación.

Usa las librerías del proyecto (GraphEMD.conf, GraphEMD.data, GraphEMD.model) para construir
el modelo, cargar pesos desde el checkpoint y evaluar sobre VisualGraphDataloader.
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


# Rutas y nombre del modelo (coherentes con train.py)
CKPT_DIR = str(REPO_ROOT / "outputs" / "autoencoder" / "ckpt")
MODEL_NAME = "GraphEMDAutoEncoder_251115"
CKPT_SUFFIX = "best_state_dict"


def get_checkpoint_path(
    ckpt_dir: str = CKPT_DIR,
    model_name: str = MODEL_NAME,
    suffix: str = CKPT_SUFFIX,
) -> str:
    """
    Devuelve la ruta al archivo .pt del checkpoint.

    Parameters
    ----------
    ckpt_dir : str
        Directorio donde se guardaron los checkpoints.
    model_name : str
        Nombre del modelo (nombre del archivo sin extensión).
    suffix : str
        Sufijo del checkpoint (ej. best_state_dict).

    Returns
    -------
    str
        Ruta absoluta al archivo .pt.

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
    Construye un VisualGraphDataloader con la configuración del proyecto.

    Parameters
    ----------
    data_dir : str
        Directorio con los datos (ej. val_data o train_data).
    batch_size : int
        Tamaño del batch.
    shuffle : bool
        Si se barajan los datos.
    drop_last : bool
        Si se descarta el último batch incompleto.
    max_num_nodes : int, optional
        Máximo número de nodos por grafo. Si es None se usa TransformConfig.
    num_workers : int
        Número de workers del DataLoader.

    Returns
    -------
    VisualGraphDataloader
        Dataloader configurado.
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
    Carga el modelo LightningAutoEncoder con los pesos del checkpoint.

    El checkpoint guardado por PtModelCheckpoint es un state_dict del LightningModule.
    Se instancia el LightningAutoEncoder con la misma configuración que en train.py
    y se cargan los pesos desde el .pt.

    Parameters
    ----------
    checkpoint_path : str
        Ruta al archivo .pt del checkpoint.
    device : torch.device, optional
        Dispositivo (cuda/cpu). Por defecto se elige automáticamente.

    Returns
    -------
    LightningAutoEncoder
        Modelo con pesos cargados, en modo eval.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo en checkpoint_path.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")

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
    Ejecuta validación del modelo sobre el dataloader y devuelve la pérdida media.

    Parameters
    ----------
    model : LightningAutoEncoder
        Modelo en modo eval.
    dataloader : VisualGraphDataloader
        Dataloader de validación.
    device : torch.device
        Dispositivo donde está el modelo.
    max_batches : int, optional
        Número máximo de batches a evaluar. None = todos.

    Returns
    -------
    float
        Pérdida media (MSE) sobre los batches evaluados.
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
    Punto de entrada: carga el modelo, construye el dataloader y ejecuta validación local.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    checkpoint_path = get_checkpoint_path()
    print(f"Cargando checkpoint: {checkpoint_path}")
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

    print("Ejecutando validación (primeros 10 batches)...")
    mean_loss = run_validation(model, val_dataloader, device, max_batches=10)
    print(f"Pérdida media (MSE) en validación: {mean_loss:.6f}")

    # Un batch de ejemplo como en run_autoencoder.py
    batch = next(iter(val_dataloader))
    num_time_samples = batch.x.shape[0]
    print(f"Número de muestras en el primer batch: {num_time_samples}")

    with torch.no_grad():
        batch = batch.to(device)
        output = model(batch)
    x = batch.x.reshape(-1, max_num_nodes)
    x_filter = x[x != 0]
    output_filter = output[x != 0]
    loss_fn = torch.nn.MSELoss()
    loss_one_batch = loss_fn(output_filter, x_filter)
    print(f"Pérdida en un batch de ejemplo: {loss_one_batch.item():.6f}")
    print("Listo.")


if __name__ == "__main__":
    main()
