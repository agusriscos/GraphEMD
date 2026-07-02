from os import listdir
from typing import Iterable

import numpy as np
from pandas import read_parquet

import torch
from torch.utils.data import Dataset, DataLoader

from ts2vg import NaturalVG
from torch_geometric.data import Data, Batch
from torch_geometric.loader.dataloader import Collater


class VisualGraphCollater(Collater):
    def __init__(self, follow_batch, exclude_keys):
        super().__init__(follow_batch, exclude_keys)

    def __call__(self, batch: Iterable[Data]):
        return Batch.from_data_list(list(batch))


class VisualGraphDataset(Dataset):
    def __init__(self, data_dir: str, max_num_nodes: int):
        self.data_dir = data_dir
        self.max_num_nodes = max_num_nodes
        self.visual_graph_builder = NaturalVG(directed="left_to_right")

    def __len__(self):
        return len(listdir(self.data_dir))

    def __getitem__(self, idx: int) -> Data:
        file_path = self.data_dir + "/" + listdir(self.data_dir)[idx]
        raw = read_parquet(file_path, engine="pyarrow").values[:, 0]
        # Writable copy: ts2vg expects a writable buffer (pyarrow returns read-only).
        data = np.asarray(raw, dtype=np.float64).copy()

        vg_build = self.visual_graph_builder.build(data)
        edges = torch.as_tensor(vg_build.edges, dtype=torch.int64).T

        node_embeddings = torch.as_tensor(data, dtype=torch.float).unsqueeze(dim=-1)
        num_nodes = node_embeddings.size(0)
        num_padding_rows = self.max_num_nodes - num_nodes
        padding_tensor = torch.zeros((num_padding_rows, node_embeddings.size(1)), dtype=torch.float)
        node_embeddings = torch.cat([node_embeddings, padding_tensor], dim=0)

        graph = Data(x=node_embeddings, edge_index=edges)
        graph.position = torch.arange(node_embeddings.size(0)).unsqueeze(1)
        graph.graph_mask = torch.cat(
            (torch.ones(num_nodes, node_embeddings.size(1)),
             torch.zeros(num_padding_rows, node_embeddings.size(1))), dim=0
        )
        graph.validate(raise_on_error=True)

        return graph


class VisualGraphDataloader(DataLoader):
    def __init__(self, data_dir: str, max_num_nodes: int, batch_size: int, **kwargs):
        self.dataset = VisualGraphDataset(data_dir, max_num_nodes)
        self.collator = VisualGraphCollater(follow_batch=None, exclude_keys=None)
        super().__init__(
            self.dataset,
            batch_size,
            collate_fn=self.collator,
            **kwargs
        )
