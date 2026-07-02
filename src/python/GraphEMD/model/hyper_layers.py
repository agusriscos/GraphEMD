from typing import Union, Optional, Dict, Any
from inspect import signature

import torch
import torch_geometric as pyg


class HyperConvLayer(torch.nn.Module):
    def __init__(
            self, conv_layer: Union[type, pyg.nn.MessagePassing],
            **convolution_kwargs
    ):
        super().__init__()
        self.conv_layer = None
        if issubclass(
                conv_layer, (pyg.nn.SAGEConv, pyg.nn.GATv2Conv, pyg.nn.GCNConv, pyg.nn.GraphConv, pyg.nn.ARMAConv)
        ):
            self.set_convolution_layer(conv_layer, **convolution_kwargs)
        else:
            raise ValueError("Tipo de capa no soportada.")

    def set_convolution_layer(self, convolution_type: pyg.nn.MessagePassing, **convolution_kwargs):
        kw = {
            k: v for k, v in convolution_kwargs.items()
            if k in list(signature(convolution_type.__init__).parameters)
        }
        self.conv_layer = convolution_type(**kw)

    def forward(
            self, x: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.conv_layer(x, edge_index)


class HyperActivationLayer(torch.nn.Module):
    def __init__(self, activation_layer: Union[type, pyg.nn.MessagePassing]):
        super().__init__()
        if issubclass(activation_layer, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.GELU)):
            self.activation_layer = activation_layer()
        else:
            raise ValueError("Tipo de capa no soportada.")

    def forward(self, data: Union[torch.Tensor, pyg.data.Batch, pyg.data.Data]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return self.activation_layer(data)
        else:
            return self.activation_layer(data.x)


class HyperAggregationLayer(torch.nn.Module):
    def __init__(self, aggregation_type: str, **aggregation_kwargs):
        super().__init__()
        self.get_aggregation_dict: Dict = {
            "sum": self.get_aggregation_layer(pyg.nn.SumAggregation, **aggregation_kwargs),
            "mean": self.get_aggregation_layer(pyg.nn.MeanAggregation, **aggregation_kwargs),
            "median": self.get_aggregation_layer(pyg.nn.MedianAggregation, **aggregation_kwargs),
            "max": self.get_aggregation_layer(pyg.nn.MaxAggregation, **aggregation_kwargs),
            "mul": self.get_aggregation_layer(pyg.nn.MulAggregation, **aggregation_kwargs),
            "std": self.get_aggregation_layer(pyg.nn.StdAggregation, **aggregation_kwargs),
            "var": self.get_aggregation_layer(pyg.nn.VarAggregation, **aggregation_kwargs),
            "attn": self.get_aggregation_layer(pyg.nn.AttentionalAggregation, **aggregation_kwargs),
            "deepsets": self.get_aggregation_layer(pyg.nn.DeepSetsAggregation, **aggregation_kwargs),
            "graphmultiset": self.get_aggregation_layer(pyg.nn.GraphMultisetTransformer, **aggregation_kwargs),
            "settransformer": self.get_aggregation_layer(pyg.nn.SetTransformerAggregation, **aggregation_kwargs)
        }
        self.aggr = None
        if aggregation_type in self.get_aggregation_dict.keys():
           self.aggr = self.get_aggregation_dict[aggregation_type]
        else:
            raise ValueError("Tipo de agregación no soportada.")

    @staticmethod
    def get_aggregation_layer(aggregation_layer: pyg.nn.Aggregation, **aggregation_kwargs):
        try:
            kw = {k: v for k, v in aggregation_kwargs.items() if k in list(signature(aggregation_layer).parameters)}
            return aggregation_layer(**kw)
        except TypeError:
            pass

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        return self.aggr(x, batch_index)
