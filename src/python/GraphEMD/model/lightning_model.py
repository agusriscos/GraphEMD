from .lightning_base import LightningModel
from typing import Union, Optional, Any, Dict
import torch
import torch_geometric as tg

class LightningAutoEncoder(LightningModel):
    def __init__(
            self, model_gen: Union[type, torch.nn.Module], model_configuration: Dict[str, Any],
            criterion: torch.nn.Module,
            learning_rate: float = 1e-3, use_rlr: Optional[bool] = False, rlr_patience: int = 2,
            rlr_factor: float = .1, min_rlr: float = 1e-6, rlr_min_delta: float = 1e-3,
    ):
        super().__init__(
            model_gen, model_configuration=model_configuration, use_rlr=use_rlr, rlr_patience=rlr_patience,
            rlr_factor=rlr_factor, min_lr=min_rlr, rlr_min_delta=rlr_min_delta, optimizer="Adam",
            learning_rate=learning_rate
        )
        self.criterion = criterion

    def training_step(
            self, batch: Union[tg.data.Batch, tg.data.Data], batch_idx: torch.Tensor
    ) -> torch.Tensor:
        y_hat = self.forward(batch)
        y = batch.x.reshape(-1, int(batch.num_nodes / batch.num_graphs))
        y_filter = y[y != 0]
        y_hat_filter = y_hat[y != 0]
        loss = self.calculate_loss(y_hat_filter, y_filter, self.criterion)
        log_dict = {'loss': loss}
        for k, v in log_dict.items():
            if k == 'loss':
                self.log(
                    'train_' + k, v, on_step=True, on_epoch=True, prog_bar=True,
                    batch_size=batch.size(0)
                )
        return log_dict['loss']

    @torch.no_grad()
    def validation_step(self, batch: Union[tg.data.Batch, tg.data.Data], batch_idx: torch.Tensor):
        y_hat = self.forward(batch)
        y = batch.x.reshape(-1, int(batch.num_nodes / batch.num_graphs))
        y_filter = y[y != 0]
        y_hat_filter = y_hat[y != 0]
        loss = self.calculate_loss(y_hat_filter, y_filter, self.criterion)
        log_dict = {'loss': loss}
        for k, v in log_dict.items():
            if k == 'loss':
                self.log(
                    'val_' + k, v, on_step=True, on_epoch=True, prog_bar=True,
                    batch_size=batch.size(0)
                )
        return log_dict['loss']

    @torch.no_grad()
    def test_step(self, batch: Union[tg.data.Batch, tg.data.Data], batch_idx: torch.Tensor):
        y_hat = self.forward(batch)
        y = batch.x.reshape(-1, int(batch.num_nodes / batch.num_graphs))
        loss = self.calculate_loss(y_hat, y, self.criterion)
        log_dict = {'loss': loss}
        for k, v in log_dict.items():
            if k == 'loss':
                self.log(
                    'test_' + k, v, on_step=True, on_epoch=True, prog_bar=True,
                )
        return log_dict['loss']

    def calculate_loss(self, y_hat: torch.Tensor, y_true: Optional[torch.Tensor] = None,
                       criterion: Optional[Any] = None) -> torch.Tensor:
        return criterion(y_hat, y_true)
