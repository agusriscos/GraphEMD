import os
from typing import Union, Dict, Any, Optional, Tuple, List
import torch
import lightning.pytorch as pl
import torch_geometric as tg
import torch.utils.data as torch_data


class LightningModel(pl.LightningModule):
    """
        Wrapper class for converting a classifier model into a pl.LightningModule
        """

    def __init__(
            self, model_gen: Union[type, torch.nn.Module],
            model_configuration: Dict[str, Any],
            save_hyperparameters: Optional[bool] = True,
            description: Optional[str] = None,
            optimizer: Union[torch.optim.Optimizer, str] = 'Adam',
            learning_rate: float = 1e-3,
            use_rlr: bool = True, rlr_patience: int = 3, rlr_factor: float = 0.1,
            min_lr: float = 1e-6, rlr_min_delta: float = 1e-3,
            weight_decay: float = 0.,
            verbose: int = 1
    ):
        """
        Initialization.
        :param model_gen: Initializer of a torch.nn.Module Class or the torch.nn.Module initialized.
        :param save_hyperparameters: Whether to seve hyperparms or not.
        :param description: Model description.
        :param gnn_kwargs: keyword arguments for building the GNN torch.nn.Module.
            Only used if model_gen is not initialized.
        :param metadata: ([Node types], [edge indexes]).
            Only used if model_gen is not initialized.
        :param embed_vars: Dictionary as {node_type: [0,14,0,0,7,...]}
            where 0 means continuous and N is the dictionary size for categorical variables
            Only used if model_gen is not initialized.
        :param optimizer: Optimizer to use in training
        :param learning_rate: Learning rate
        :param use_rlr: Whether to use or not ReduceLROnPlateau while training
        :param rlr_patience: Patience of ReduceLROnPlateau
        :param rlr_factor: Decrease factor for learning rate in ReduceLROnPlateau
        :param min_lr: Minimum learning rate to reach
        :param rlr_min_delta: Minimum change in loss to consider it as a non-learning step.
        :param weight_decay: Weight decay regularization.
        :param verbose: Verbosity level.
        """
        super().__init__()
        if save_hyperparameters:
            if isinstance(model_gen, type):
                self.save_hyperparameters(ignore=["criterion"])
            else:
                self.save_hyperparameters(ignore=["model_gen", "criterion"])
        self.descr = description
        self.device_ = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = model_gen(model_configuration).to(self.device_)
        self.optimizer = optimizer
        self.lr = learning_rate
        self.use_rlr = use_rlr
        self.rlr_patience = rlr_patience
        self.reduce_lr_factor = rlr_factor
        self.min_lr = min_lr
        self.rlr_min_delta = rlr_min_delta
        self.weight_decay = weight_decay
        self.optim = None
        self.scheduler = None
        self.verbose = verbose

    @classmethod
    def load(cls, ckpt_dir: str, model_name: str, model_gen: type, gnn_kwargs: dict) -> pl.LightningModule:
        """
        Load a model from checkpoint (.pt) file
        :param ckpt_dir: checkpoint directory
        :param model_name: model name identifier
        :param model_gen: model generator
        :param gnn_kwargs: keyword arguments for model generator
        :return: model
        """
        pretrained_filename = os.path.join(ckpt_dir, model_name + '.ckpt')
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model: {os.path.basename(pretrained_filename)}")
            return cls.load_from_checkpoint(pretrained_filename, model_gen=model_gen, gnn_kwargs=gnn_kwargs)
        else:
            raise ValueError(f"Couldn't find pretrained_checkpoint: {pretrained_filename}")

    @classmethod
    def __cls__(cls) -> type:
        """
        Return non-initialized class
        :return: cls
        """
        return cls

    def configure_optimizers(self) -> \
            Union[Tuple[List[torch.optim.Optimizer], List[dict]], List[torch.optim.Optimizer]]:
        """
        Optimizer configuration. Method overriden from pl.LightningModule.
        :return: ()
            optimizer
        """
        self.optim = None
        if isinstance(self.optimizer, str):
            if self.optimizer == 'Adam':
                self.optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == 'RMSProp':
                self.optim = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == 'SGD':
                self.optim = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                raise ValueError('optimizer given must be a valid str ["Adam", "RMSProp", "SGD"]. Got {self.optimizer}')
        elif isinstance(self.optimizer, torch.optim.Optimizer.__class__):
            self.optim = self.optimizer
        else:
            raise (ValueError('optimizer given must be a valid str or torch.optim.Optimizer instance'))
        if self.use_rlr:
            self.scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optim,
                    patience=self.rlr_patience - 1,
                    mode='min',
                    min_lr=self.min_lr,
                    threshold=self.rlr_min_delta,
                    verbose=True,
                    factor=self.reduce_lr_factor
                ),
                'monitor': 'val_loss',
                'frequency': 1,
                'interval': 'epoch'
            }
            return [self.optim], [self.scheduler]
        else:
            return [self.optim]

    def forward(self, data: Union[tg.data.Batch, tg.data.Data]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Make the forward pass of the GNN
        :param data: data to pass
        :return: prediction
        """
        y_hat = self.model(data.to(self.device_))
        return y_hat

    def calculate_loss(self, y_hat: torch.Tensor, y_true: Optional[torch.Tensor] = None,
                       criterion: Optional[Any] = None) -> torch.Tensor:
        """
        Calculates the loss using the criterion
        :param y_hat: predicted value
        :param y_true: real value
        :param criterion: criterion to use
        :return: loss
        """
        raise NotImplementedError("Implement this function in subclass.")

    def calculate_metrics(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the metrics
        :param y_hat: predicted value
        :param y_true: real value
        :return: loss
        """
        raise NotImplementedError("Implement this function in subclass.")

    def training_step(self, batch: Union[tg.data.Batch, tg.data.Data], batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Train step. Method overriden from pl.LightningModule.
        :param batch: batch
        :param batch_idx: identifier
        :return: loss
        """
        raise NotImplementedError("Implement this function in subclass.")

    @torch.no_grad()
    def validation_step(self, batch: Union[tg.data.Batch, tg.data.HeteroData], batch_idx: torch.Tensor) -> None:
        """
        Validation step. Method overriden from pl.LightningModule.
        :param batch: batch
        :param batch_idx: identifier
        :return: loss
        """
        raise NotImplementedError("Implement this function in subclass.")

    @torch.no_grad()
    def test_step(self, batch: Union[tg.data.Batch, tg.data.HeteroData], batch_idx: torch.Tensor) -> None:
        """
        Test step. Method overriden from pl.LightningModule.
        :param batch: batch
        :param batch_idx: identifier
        :return: loss
        """
        raise NotImplementedError("Implement this function in subclass.")

    @torch.no_grad()
    def predict_step(
            self, batch: Union[tg.data.Batch, tg.data.Data], batch_idx: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Any]:
        """
        :param batch:  batch
        :param batch_idx:  batch identifier
        :return: prediction
        """
        raise NotImplementedError("Implement this function in subclass.")

    @torch.no_grad()
    def predict(self, data: Union[torch_data.DataLoader]) -> Union[torch.Tensor, List[torch.Tensor], Any]:
        """
        Makes a prediction
        :param data: Data to predict
        :return: prediction
        """
        raise NotImplementedError("Implement this function in subclass.")
