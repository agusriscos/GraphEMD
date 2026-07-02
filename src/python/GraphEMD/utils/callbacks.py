import sys
import os
import torch
from typing import Dict, Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar


class ProgressBar(TQDMProgressBar):
    """
    Control abling and disabling Progress Bar in Lightning Trainer.
    """
    def init_validation_tqdm(self) -> TQDMProgressBar.mro():
        """
        Disable at beggining validation
        :return: tqdm
        """
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self) -> TQDMProgressBar.mro():
        """
        Disable at beggining predict
        :return: tqdm
        """
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self) -> TQDMProgressBar.mro():
        """
        Disable at beggining test
        :return: tqdm
        """
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class PtModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """
    Checkpoint pt model.
    """
    def __init__(self, dirpath: Optional[str], filename: Optional[str],
                 monitor: Optional[str], mode: Optional[str],
                 save_weights_only: Optional[bool] = False, verbose: Optional[bool] = False,
                 suffix: Optional[str] = None):
        """
        Constructor
        :param dirpath: Directory path
        :param filename: Filename
        :param monitor: monitored loss
        :param mode: monitoring mode
        :param save_weights_only: save whole model or weights
        :param verbose: verbosity
        :param suffix: Suffix to add to filename.
        """
        super().__init__(dirpath=dirpath, filename=filename, monitor=monitor,
                         mode=mode, save_weights_only=save_weights_only,
                         verbose=verbose)
        self.suffix = '_' + suffix if suffix is not None else ''

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           checkpoint: Dict[str, Any]) -> None:
        """
        Saves model when checkpoint is triggered
        :param trainer: Trainer instance
        :param pl_module: Lightning Module to save
        :param checkpoint: Checkpoint
        """
        torch.save(pl_module.state_dict(),
                   os.path.join(self.dirpath,
                                self.filename+self.suffix+'.pt'))
