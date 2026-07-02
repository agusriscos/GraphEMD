# Guide for Adding a New IMF-to-Graph Transformation

This guide explains the steps required to add a new Intrinsic Mode Function (IMF) to graph transformation in the GraphEMD project.

## Project Structure

The project currently supports three types of transformations:
- **HVG (Horizontal Visibility Graph)**: Implemented in `build_hvg_imf_graph()`
- **NVG (Natural Visibility Graph)**: Implemented in `build_nvg_imf_graph()`
- **Recurrence Graph**: Implemented in `build_recurrence_imf_graph()`

All transformations are in the file:
```
src/python/GraphEMD/data/graph_imf_transform_utils.py
```

## Steps to Add a New Transformation

### Step 1: Implement the Transformation Function

Create a new function in `graph_imf_transform_utils.py` that follows the pattern of the existing functions.

#### Base Function Structure

```python
def build_new_type_imf_graph(
    archivo_imfs: str,
    id_imf: str,
    # Add transformation-specific parameters here
) -> Data:
    """
    Transforms an IMF into a [GRAPH_TYPE_NAME] graph and returns a PyTorch Geometric Data object.

    [Detailed description of what the transformation does]

    Parameters
    ----------
    archivo_imfs : str
        Path to the parquet file with IMFs (must contain columns IMF_1, IMF_2, etc.).
    id_imf : str
        Identifier of the IMF to transform (e.g. "IMF_1", "IMF_2", "Residuo").
    # Document transformation-specific parameters here

    Returns
    -------
    Data
        PyTorch Geometric Data object with the [GRAPH_TYPE_NAME] graph of the IMF.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = build_new_type_imf_graph(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodes: {grafo.num_nodes}, Edges: {grafo.num_edges}")
    """
    # 1. Load IMF data
    print(f"Loading IMFs from: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"DataFrame shape: {df_imfs.shape}")
    print(f"Available columns: {list(df_imfs.columns)}")

    # 2. Verify that the specified IMF exists
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"IMF '{id_imf}' does not exist in the DataFrame. "
            f"Available columns: {list(df_imfs.columns)}"
        )

    # 3. Extract the selected IMF
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nSelected IMF: {id_imf}")
    print(f"Shape of {id_imf}: {imf_valores.shape}")
    print(
        f"Values - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # 4. Build the graph using your specific algorithm
    print("\nBuilding [GRAPH_TYPE_NAME] graph...")
    # YOUR SPECIFIC LOGIC TO BUILD THE GRAPH GOES HERE
    # You must obtain:
    # - nodos: array with node indices
    # - edges: array of shape (2, num_edges) or list of tuples (source, target)

    # 5. Convert to PyTorch Geometric format
    print("Converting to PyTorch Geometric Data object...")
    # Convert edges to edge_index (format [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Create node features
    # Option 1: If each node has a single feature (series value)
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)
    
    # Option 2: If each node has multiple features (e.g. embedding)
    # node_features = torch.tensor(embedding, dtype=torch.float)

    # 6. Create the Data object
    data = Data(x=node_features, edge_index=edge_index)

    # 7. (Optional) Save metadata in the Data object
    # If your transformation has important parameters, save them as attributes
    # data.parametro_importante = valor_parametro

    # 8. Print information about the created graph
    print(f"\nData object created:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    if data.x is not None:
        print(f"  - Node features shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data
```

#### Important Points

1. **edge_index format**: Must be a tensor of shape `[2, num_edges]` where the first row contains source nodes and the second row contains target nodes.

2. **Node features**: Must be a tensor of shape `[num_nodes, num_features]`. If each node has a single feature, use `.unsqueeze(1)` to add the dimension.

3. **Optional metadata**: If your transformation has important parameters (such as `tau` and `dim_embedding` in the recurrence graph), save them as attributes of the `Data` object so they are included in metadata when saving.

4. **Error handling**: Include appropriate validations and clear error messages.

5. **Logging**: Use `print()` to provide progress information (the project does not use the standard logging system in these functions).

### Step 2: Integrate into `build_all_imf_graphs()`

Modify the `build_all_imf_graphs()` function in the same file to include your new transformation.

#### Location in the Code

Find the section where the different graph types are processed (around line 776) and add a new block:

```python
# 4. [GRAPH_TYPE_NAME] graph
try:
    print(f"\n--- Generating [GRAPH_TYPE_NAME] graph for {id_imf} ---")
    carpeta_nuevo_tipo = carpeta_salida_base / "[nombre_tipo]" / id_imf.lower()
    carpeta_nuevo_tipo.mkdir(parents=True, exist_ok=True)
    archivo_salida_nuevo_tipo = str(
        carpeta_nuevo_tipo / f"grafo_[nombre_tipo]_{id_imf.lower()}"
    )

    grafo_nuevo_tipo = build_new_type_imf_graph(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
        # Pass transformation-specific parameters here
    )
    save_graph_data(
        data=grafo_nuevo_tipo,
        archivo_salida=archivo_salida_nuevo_tipo,
        id_imf=id_imf,
    )

    resultados_imf["[nombre_tipo]"] = {
        "archivo": archivo_salida_nuevo_tipo,
        "num_nodes": grafo_nuevo_tipo.num_nodes,
        "num_edges": grafo_nuevo_tipo.num_edges,
        # Add specific metadata here if you saved it in the Data object
        "exito": True,
    }
    print(f"✓ [GRAPH_TYPE_NAME] graph generated successfully for {id_imf}")

except Exception as e:
    print(f"✗ Error generating [GRAPH_TYPE_NAME] graph for {id_imf}: {e}")
    resultados_imf["[nombre_tipo]"] = {"exito": False, "error": str(e)}
```

#### Parameters of `build_all_imf_graphs()`

If your transformation requires additional parameters, add them to the signature of `build_all_imf_graphs()` and pass them to your transformation function.

### Step 3: Update the Module Documentation

Update the module docstring at the top of `graph_imf_transform_utils.py` to include your new transformation:

```python
"""
Utilities for transforming IMFs into graphs as PyTorch Geometric Data objects.

This module contains functions for transforming Intrinsic Mode Functions (IMFs) into different
types of graphs: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG),
recurrence graph, and [GRAPH_TYPE_NAME].
"""
```

### Step 4: Create a Test Script (Optional but Recommended)

Create a test script in `scripts/` to test your new transformation independently. You can use the existing scripts as reference:

- `scripts/16nov25/build_hvg_imf_graph.py`
- `scripts/16nov25/build_nvg_imf_graph.py`
- `scripts/16nov25/build_recurrence_imf_graph.py`

#### Example Test Script

```python
"""
Script to transform an IMF into a [GRAPH_TYPE_NAME] graph as a PyTorch Geometric Data object.

This script loads an IMF from a parquet file and transforms it into a [GRAPH_TYPE_NAME] graph
as a PyTorch Geometric Data object.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import build_new_type_imf_graph
from GraphEMD.data.python_utils import save_graph_data


if __name__ == "__main__":
    
    # Configuration variables to test the method
    proyecto_root = Path(__file__).parent.parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    id_imf = "IMF_1"

    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos" / "[nombre_tipo]" / id_imf.lower()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = str(carpeta_salida / f"grafo_[nombre_tipo]_{id_imf.lower()}")

    # Verify that the file exists
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"File {archivo_imfs} does not exist. "
            "Make sure the file is in the correct location."
        )

    # Test the method
    print("=" * 60)
    print("IMF TO [GRAPH_TYPE_NAME] GRAPH TRANSFORMATION")
    print("=" * 60)

    # Build the graph
    grafo = build_new_type_imf_graph(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
        # Pass transformation-specific parameters here
    )

    # Save the graph
    save_graph_data(
        data=grafo,
        archivo_salida=archivo_salida,
        id_imf=id_imf,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"[GRAPH_TYPE_NAME] graph created successfully for {id_imf}")
    print(f"  - Nodes: {grafo.num_nodes}")
    print(f"  - Edges: {grafo.num_edges}")
    if grafo.x is not None:
        print(f"  - Node features: {grafo.x.shape}")
```

### Step 5: Verify the Output Format

Make sure your function returns a PyTorch Geometric `Data` object with:
- `x`: Node features tensor of shape `[num_nodes, num_features]`
- `edge_index`: Edge tensor of shape `[2, num_edges]`

Optionally, you can add custom attributes to the `Data` object to store metadata.

### Step 6: Test the Integration

Run `build_all_imf_graphs()` with your new transformation to verify that:
1. Graphs are generated correctly
2. They are saved in the expected folder structure
3. Metadata is saved correctly
4. There are no errors during processing

## Output Folder Structure

Graphs are saved in the following structure:

```
{carpeta_salida_base}/
├── hvg/
│   └── {id_imf}/
│       └── grafo_hvg_{id_imf}.*
├── nvg/
│   └── {id_imf}/
│       └── grafo_nvg_{id_imf}.*
├── recurrencia/
│   └── {id_imf}/
│       └── grafo_recurrencia_{id_imf}.*
└── [nombre_tipo]/
    └── {id_imf}/
        └── grafo_[nombre_tipo]_{id_imf}.*
```

Each graph is saved in multiple formats:
- `*_features.parquet`: Node features
- `*_edges.parquet`: Edge list
- `*_metadata.csv`: Graph metadata
- `*.pt`: Full serialized Data object with torch

## Reference Examples

### Simple Example: HVG and NVG

The `build_hvg_imf_graph()` and `build_nvg_imf_graph()` functions are simple examples that:
- Use the `ts2vg` library to build the graph
- Each node has a single feature (the time series value)
- Do not require complex additional parameters

### Complex Example: Recurrence Graph

The `build_recurrence_imf_graph()` function is a more complex example that:
- Requires multiple steps (tau selection, dim selection, embedding construction, recurrence matrix computation)
- Has multiple configurable parameters
- Saves additional metadata in the Data object
- Uses helper functions for intermediate calculations

## Code Conventions

Follow the conventions established in the project:

1. **Function names**: Use `snake_case` and the prefix `build_` followed by the graph type and `_imf`
2. **Docstrings**: Use NumPy/SciPy format in English
3. **Type hints**: Include type hints in all functions
4. **Informative messages**: Use `print()` to provide progress information
5. **Error handling**: Include validations and clear error messages

## Final Checklist

Before considering your contribution complete, verify:

- [ ] The transformation function is implemented and documented
- [ ] The function returns a valid PyTorch Geometric `Data` object
- [ ] The function is integrated into `build_all_imf_graphs()`
- [ ] The module documentation is updated
- [ ] A test script has been created (optional but recommended)
- [ ] The transformation has been tested with real data
- [ ] Graphs are saved correctly in the expected folder structure
- [ ] Metadata is saved correctly (if applicable)
- [ ] The code follows project conventions
- [ ] There are no linting errors (verify with black)

## Additional Notes

- If your transformation requires new dependencies, make sure to add them to the project configuration files (requirements.txt, setup.py, etc.)
- If your transformation is computationally intensive, consider adding options for parallelization or batch processing
- If your transformation has parameters that significantly affect results, clearly document their effects and recommended values

## Support

If you have questions or problems when implementing a new transformation, review:
1. The existing functions as reference
2. The PyTorch Geometric documentation on `Data` objects
3. The existing test scripts
