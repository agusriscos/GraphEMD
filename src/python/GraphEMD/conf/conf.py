from typing import Dict, Any, Iterable
import torch
import torch_geometric as pyg
from CommonUtils.data import DictClass

from GraphEMD.conf.paths import DATA_DIR, REPO_ROOT

# PATH
DATA_PATH: str = str(DATA_DIR)
FILE_PATH: str = "{}/dcoilwtico.csv".format(DATA_PATH)
ARTIFACTS_DIR: str = str(REPO_ROOT / "outputs")


# OverlapTransformConfig
class TransformConfig(DictClass):  # type: ignore[misc]
    MIN_VAL: int = -500  # type: ignore[assignment]
    MAX_VAL: int = 500  # type: ignore[assignment]
    MIN_IMF_VAL: int = -40  # type: ignore[assignment]
    MAX_IMF_VAL: int = 80  # type: ignore[assignment]
    MIN_WINDOW_SIZE: int = 100  # type: ignore[assignment]
    MAX_WINDOW_SIZE: int = 2500  # type: ignore[assignment]
    WINDOW_NUM: int = 10000  # type: ignore[assignment]
    WINDOW_MODE: int = 2  # type: ignore[assignment]


class GraphEncoderConvolutionConfig(DictClass):  # type: ignore[misc]
    conv_layer: pyg.nn.MessagePassing = pyg.nn.SAGEConv  # type: ignore[assignment]
    aggr: str = "mean"  # type: ignore[assignment]

    # NO TOUCH! ALREADY FIXED
    in_channels: int = -1  # type: ignore[assignment]
    project: bool = False  # type: ignore[assignment]
    normalize: bool = False  # type: ignore[assignment]
    dropout: int = 0  # type: ignore[assignment]
    bias: bool = True  # type: ignore[assignment]


class GraphEncoderAggregationConfig(DictClass):  # type: ignore[misc]
    aggregation_type: str = "graphmultiset"  # == pyg.nn.GraphMultisetTransformer  # type: ignore[assignment]
    k: int = 3  # type: ignore[assignment]
    num_encoder_blocks: int = 4  # type: ignore[assignment]
    heads: int = 2  # type: ignore[assignment]

    # NO TOUCH! FIXED!!
    num_decoder_blocks: int = 0  # type: ignore[assignment]
    norm: bool = False  # type: ignore[assignment]
    layer_norm: bool = False  # type: ignore[assignment]
    dropout: float = 0  # type: ignore[assignment]


class GraphEncoderConfig(DictClass):  # type: ignore[misc]
    # General parameters
    LATENT_DIM: int = 250  # type: ignore[assignment]
    CONVOLUTION_NUM_LAYERS: int = 3  # type: ignore[assignment]
    DROPOUT: float = 0  # type: ignore[assignment]
    USE_POSITIONAL_EMBEDDING: bool = False  # type: ignore[assignment]
    POSITIONAL_EMBEDDING_MODE: int = None  # type: ignore[assignment]
    ACTIVATION_LAYER: torch.nn.Module = torch.nn.ReLU  # type: ignore[assignment]
    USE_ACTIVATION_OUTPUT: bool = False  # type: ignore[assignment]
    APPLY_GRAPH_MASK: bool = True  # type: ignore[assignment]

    # NO TOUCH! FIXED!!
    # Message-passing parameters between nodes (node embeddings)
    CONVOLUTION: Dict[str, Any] = GraphEncoderConvolutionConfig.to_dict()  # type: ignore[assignment]
    CONVOLUTION.update({"out_channels": LATENT_DIM})
    # Aggregation parameters (readout(node_embeddings) -> graph_embeddings) for nodes
    AGGREGATION: Dict[str, Any] = GraphEncoderAggregationConfig.to_dict()  # type: ignore[assignment]
    AGGREGATION.update({"channels": LATENT_DIM})


class InverseFunnelDecoderConfig(DictClass):  # type: ignore[misc]
    DROPOUT: float = 0  # type: ignore[assignment]
    ACTIVATION_LAYER: torch.nn.Module = torch.nn.ReLU  # type: ignore[assignment]
    APPLY_GRAPH_MASK: bool = False  # type: ignore[assignment]
    LINEAR_OUTPUT_DIMS: Iterable[int] = [500, 1000, TransformConfig.to_dict()["MAX_WINDOW_SIZE"]]  # type: ignore[assignment]

    # NO TOUCH! FIXED!!
    ENCODER_LATENT_DIM: int = GraphEncoderConfig.to_dict()["LATENT_DIM"]  # type: ignore[assignment]