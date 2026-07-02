import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from GraphEMD.conf import DATA_PATH, GraphEncoderConfig, InverseFunnelDecoderConfig, TransformConfig
from GraphEMD.conf.paths import REPO_ROOT
from GraphEMD.data import VisualGraphDataloader
from GraphEMD.model import GraphEMDAutoEncoder, LightningAutoEncoder
from GraphEMD.utils import ProgressBar, PtModelCheckpoint

if __name__ == '__main__':
    dir_ = str(REPO_ROOT / "outputs" / "autoencoder")
    Path(dir_).mkdir(parents=True, exist_ok=True)
    model_name = "GraphEMDAutoEncoder_251115"
    verbose = True
    logger = TensorBoardLogger(
        save_dir="{}/logs".format(dir_), name=model_name,
        log_graph=False
    )
    callbacks = [
        ProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=verbose),
        PtModelCheckpoint(
            dirpath=r"{}/ckpt".format(dir_), filename=model_name, suffix="best_state_dict", monitor="val_loss",
            mode="min",
            save_weights_only=False, verbose=verbose
        )
    ]
    trainer = pl.Trainer(
        accelerator="cuda", devices="auto", max_epochs=1001, enable_progress_bar=True, log_every_n_steps=1,
        num_sanity_val_steps=0, limit_train_batches=None, limit_val_batches=None, limit_test_batches=None, logger=logger,
        callbacks=callbacks
    )

    MAX_NUM_NODES = TransformConfig.to_dict()["MAX_WINDOW_SIZE"]
    encoder_config = GraphEncoderConfig.to_dict()
    decoder_config = InverseFunnelDecoderConfig.to_dict()
    num_workers = max(1, (os.cpu_count() or 4) - 2)
    val_dataloader = VisualGraphDataloader(
        data_dir="{}/val_data".format(DATA_PATH), batch_size=100,
        shuffle=False, drop_last=True, max_num_nodes=MAX_NUM_NODES, num_workers=num_workers
    )
    train_dataloader = VisualGraphDataloader(
        data_dir="{}/train_data".format(DATA_PATH), batch_size=100,
        shuffle=True, drop_last=True, max_num_nodes=MAX_NUM_NODES, num_workers=num_workers
    )
    model_gen = GraphEMDAutoEncoder
    lightning_model = LightningAutoEncoder(
        model_gen=model_gen, model_configuration={"encoder_config": encoder_config, "decoder_config": decoder_config},
        criterion=torch.nn.MSELoss(reduction="sum").to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    )
    lightning_model.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    lightning_model.forward(val_dataloader.dataset[0])

    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)