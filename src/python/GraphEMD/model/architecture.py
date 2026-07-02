from typing import Dict, Any, Union

import math
import torch
import torch_geometric as pyg
from .hyper_layers import HyperConvLayer, HyperAggregationLayer, HyperActivationLayer


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, mode: int):
        """
        Positional Embedding Module.
        :param mode: For mode == 1, encoding uses only the sine function for all positions. For mode == 2,
        encoding uses sine for odd positions and cosine for even positions.
        """
        super().__init__()
        self.div_term_const = 10000
        self.mode = mode

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if self.mode == 1:
            position_encoding = torch.sin(
                position / self.div_term_const ** (2 * (torch.arange(x.size(-1)) // 2) / x.size(-1))
            )
        else:
            div_term = torch.exp(torch.arange(0, x.size(-1), 2).float() * (-math.log(10000.0) / x.size(-1)))
            position_encoding = torch.zeros(x.size())
            position_encoding[:, 0::2] = torch.sin(position * div_term)
            position_encoding[:, 1::2] = torch.cos(position * div_term)

        return x + position_encoding


class NodeEmbedding(torch.nn.Module):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.configuration = configuration
        conv_config = configuration["CONVOLUTION"]

        # Message propagation and aggregation block (INPUT GRAPH EMBEDDINGS)
        self.conv_layers = torch.nn.ModuleList(
            [HyperConvLayer(**conv_config) for _ in range(configuration["CONVOLUTION_NUM_LAYERS"])]
        )
        self.dropout_layers = torch.nn.ModuleList(
            [torch.nn.Dropout(configuration["DROPOUT"]) for _ in range(len(self.conv_layers))]
        )
        self.activation_layers = torch.nn.ModuleList(
            [HyperActivationLayer(configuration["ACTIVATION_LAYER"]) for _ in range(len(self.conv_layers))]
        )
        if configuration["USE_POSITIONAL_EMBEDDING"]:
            self.pos_embedding = PositionalEmbedding(configuration["POSITIONAL_EMBEDDING_MODE"])

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]) -> torch.Tensor:
        out_x = data.x
        for conv_i, layer in enumerate(self.conv_layers):
            out_x = layer.forward(x=out_x, edge_index=data.edge_index)
            out_x = self.activation_layers[conv_i].forward(out_x)
            if conv_i < len(self.conv_layers) - 1:
                out_x = self.dropout_layers[conv_i].forward(out_x)
        if self.configuration["USE_POSITIONAL_EMBEDDING"]:
            out_x = self.pos_embedding.forward(out_x, data.position)
        out_x = self.dropout_layers[-1].forward(out_x)
        if self.configuration["APPLY_GRAPH_MASK"]:
            out_x = data.graph_mask * out_x
        return out_x


class GraphEncoder(torch.nn.Module):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.configuration = configuration
        conv_config, aggr_config = configuration["CONVOLUTION"], configuration["AGGREGATION"]

        self.node_embedding = NodeEmbedding(configuration)
        self.aggr_layer = HyperAggregationLayer(**aggr_config)
        self.linear_layer = torch.nn.Linear(
            in_features=configuration["LATENT_DIM"], out_features=configuration["LATENT_DIM"], bias=True
        )
        if configuration["USE_ACTIVATION_OUTPUT"]:
            self.activation_layers.append(torch.nn.Sigmoid())

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]) -> torch.Tensor:
        # Node embedding computation
        out_x = self.node_embedding.forward(data)
        # Graph embedding computation
        out = self.aggr_layer.forward(x=out_x, batch_index=data.batch)
        out = self.linear_layer.forward(out)

        if self.configuration["USE_ACTIVATION_OUTPUT"]:
            out = self.activation_layers[-1].forward(out)
        return out


class InverseFunnelDecoder(torch.nn.Module):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.configuration = configuration
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(configuration["ENCODER_LATENT_DIM"], configuration["LINEAR_OUTPUT_DIMS"][0])]
        )
        self.activation_layers = torch.nn.ModuleList([HyperActivationLayer(configuration["ACTIVATION_LAYER"])])
        self.dropout_layers = torch.nn.ModuleList([torch.nn.Dropout(configuration["DROPOUT"])])

        for layer_i in range(1, len(configuration["LINEAR_OUTPUT_DIMS"])):
            self.linear_layers.append(
                torch.nn.Linear(self.linear_layers[-1].out_features, configuration["LINEAR_OUTPUT_DIMS"][layer_i])
            )
            if layer_i < len(configuration["LINEAR_OUTPUT_DIMS"]) - 1:
                self.activation_layers.append(HyperActivationLayer(configuration["ACTIVATION_LAYER"]))
                self.dropout_layers.append(torch.nn.Dropout(configuration["DROPOUT"]))

        self.activation_layers.append(HyperActivationLayer(torch.nn.Sigmoid))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.dropout_layers[0](self.activation_layers[0](self.linear_layers[0](data)))
        for layer_index in range(1, len(self.linear_layers)):
            out = self.linear_layers[layer_index].forward(out)
            out = self.activation_layers[layer_index].forward(out)
            if layer_index < len(self.linear_layers) - 1:
                out = self.dropout_layers[layer_index].forward(out)
        return out


class GraphEMDAutoEncoder(torch.nn.Module):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.configuration = configuration
        self.encoder = GraphEncoder(configuration["encoder_config"])
        self.decoder = InverseFunnelDecoder(configuration["decoder_config"])

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]) -> torch.Tensor:
        out = self.encoder(data)
        out = self.decoder(out)
        return out


class GraphEMDVariationalAutoEncoder(torch.nn.Module):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.configuration = configuration
        self.encoder = GraphEncoder(configuration["encoder_config"])
        self.decoder = InverseFunnelDecoder(configuration["decoder_config"])
        self.mu_dim = self.configuration.get("MU_DIM", int(self.configuration["encoder_config"]["LATENT_DIM"] // 2))

    @staticmethod
    def parametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]) -> torch.Tensor:
        out = self.encoder(data)
        mu, logvar = out[:, :self.configuration["MU_DIM"]], out[:, self.configuration["MU_DIM"]:]
        z = self.parametrize(mu, logvar)
        return self.decoder(z), mu, logvar
