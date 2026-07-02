# Guía para Añadir una Nueva Transformación de IMF a Grafo

Esta guía explica los pasos necesarios para añadir una nueva transformación de Intrinsic Mode Function (IMF) a grafo en el proyecto GraphEMD.

## Estructura del Proyecto

El proyecto actualmente soporta tres tipos de transformaciones:
- **HVG (Horizontal Visibility Graph)**: Implementado en `obtener_grafo_hvg_imf()`
- **NVG (Natural Visibility Graph)**: Implementado en `obtener_grafo_nvg_imf()`
- **Grafo de Recurrencia**: Implementado en `obtener_grafo_recurrencia_imf()`

Todas las transformaciones están en el archivo:
```
src/python/GraphEMD/data/graph_imf_transform_utils.py
```

## Pasos para Añadir una Nueva Transformación

### Paso 1: Implementar la Función de Transformación

Crea una nueva función en `graph_imf_transform_utils.py` que siga el patrón de las funciones existentes.

#### Estructura Base de la Función

```python
def obtener_grafo_nuevo_tipo_imf(
    archivo_imfs: str,
    id_imf: str,
    # Añade aquí parámetros específicos de tu transformación
) -> Data:
    """
    Transforma una IMF a grafo [NOMBRE_DEL_TIPO] y retorna un objeto Data de PyTorch Geometric.

    [Descripción detallada de lo que hace la transformación]

    Parameters
    ----------
    archivo_imfs : str
        Ruta al archivo parquet con las IMFs (debe contener columnas IMF_1, IMF_2, etc.).
    id_imf : str
        Identificador de la IMF a transformar (ej: "IMF_1", "IMF_2", "Residuo").
    # Documenta aquí los parámetros específicos de tu transformación

    Returns
    -------
    Data
        Objeto Data de PyTorch Geometric con el grafo [NOMBRE_DEL_TIPO] de la IMF.

    Examples
    --------
    >>> from pathlib import Path
    >>> proyecto_root = Path(__file__).parent.parent.parent.parent.parent
    >>> archivo_imfs = proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet"
    >>> grafo = obtener_grafo_nuevo_tipo_imf(str(archivo_imfs), "IMF_1")
    >>> print(f"Nodos: {grafo.num_nodes}, Enlaces: {grafo.num_edges}")
    """
    # 1. Cargar datos de IMFs
    print(f"Cargando IMFs desde: {archivo_imfs}")
    df_imfs = pd.read_parquet(archivo_imfs, engine="pyarrow")
    print(f"Shape del DataFrame: {df_imfs.shape}")
    print(f"Columnas disponibles: {list(df_imfs.columns)}")

    # 2. Verificar que existe la IMF especificada
    if id_imf not in df_imfs.columns:
        raise ValueError(
            f"La IMF '{id_imf}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df_imfs.columns)}"
        )

    # 3. Extraer la IMF seleccionada
    imf_valores = np.array(df_imfs[id_imf].values)
    print(f"\nIMF seleccionada: {id_imf}")
    print(f"Shape de {id_imf}: {imf_valores.shape}")
    print(
        f"Valores - Min: {np.min(imf_valores):.4f}, Max: {np.max(imf_valores):.4f}, "
        f"Mean: {np.mean(imf_valores):.4f}"
    )

    # 4. Construir el grafo usando tu algoritmo específico
    print("\nConstruyendo grafo [NOMBRE_DEL_TIPO]...")
    # AQUÍ VA TU LÓGICA ESPECÍFICA PARA CONSTRUIR EL GRAFO
    # Debes obtener:
    # - nodos: array con los índices de los nodos
    # - enlaces: array de forma (2, num_edges) o lista de tuplas (source, target)

    # 5. Convertir a formato PyTorch Geometric
    print("Convirtiendo a objeto Data de PyTorch Geometric...")
    # Convertir enlaces a edge_index (formato [2, num_edges])
    edge_index = torch.tensor(enlaces.T, dtype=torch.long)

    # Crear features de nodos
    # Opción 1: Si cada nodo tiene una sola feature (valor de la serie)
    node_features = torch.tensor(imf_valores, dtype=torch.float).unsqueeze(1)
    
    # Opción 2: Si cada nodo tiene múltiples features (ej: embedding)
    # node_features = torch.tensor(embedding, dtype=torch.float)

    # 6. Crear el objeto Data
    data = Data(x=node_features, edge_index=edge_index)

    # 7. (Opcional) Guardar metadatos en el objeto Data
    # Si tu transformación tiene parámetros importantes, guárdalos como atributos
    # data.parametro_importante = valor_parametro

    # 8. Imprimir información del grafo creado
    print(f"\nObjeto Data creado:")
    print(f"  - Número de nodos: {data.num_nodes}")
    print(f"  - Número de enlaces: {data.num_edges}")
    if data.x is not None:
        print(f"  - Features de nodos shape: {data.x.shape}")
    if data.edge_index is not None:
        print(f"  - Edge index shape: {data.edge_index.shape}")

    return data
```

#### Puntos Importantes

1. **Formato de edge_index**: Debe ser un tensor de forma `[2, num_edges]` donde la primera fila son los nodos fuente y la segunda los nodos destino.

2. **Features de nodos**: Debe ser un tensor de forma `[num_nodes, num_features]`. Si cada nodo tiene una sola feature, usa `.unsqueeze(1)` para añadir la dimensión.

3. **Metadatos opcionales**: Si tu transformación tiene parámetros importantes (como `tau` y `dim_embedding` en el grafo de recurrencia), guárdalos como atributos del objeto `Data` para que se incluyan en los metadatos al guardar.

4. **Manejo de errores**: Incluye validaciones apropiadas y mensajes de error claros.

5. **Logging**: Usa `print()` para proporcionar información sobre el progreso (el proyecto no usa el sistema de logging estándar en estas funciones).

### Paso 2: Integrar en `obtener_grafos_all_imf()`

Modifica la función `obtener_grafos_all_imf()` en el mismo archivo para incluir tu nueva transformación.

#### Ubicación en el Código

Busca la sección donde se procesan los diferentes tipos de grafos (alrededor de la línea 776) y añade un nuevo bloque:

```python
# 4. Grafo [NOMBRE_DEL_TIPO]
try:
    print(f"\n--- Generando grafo [NOMBRE_DEL_TIPO] para {id_imf} ---")
    carpeta_nuevo_tipo = carpeta_salida_base / "[nombre_tipo]" / id_imf.lower()
    carpeta_nuevo_tipo.mkdir(parents=True, exist_ok=True)
    archivo_salida_nuevo_tipo = str(
        carpeta_nuevo_tipo / f"grafo_[nombre_tipo]_{id_imf.lower()}"
    )

    grafo_nuevo_tipo = obtener_grafo_nuevo_tipo_imf(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
        # Pasa aquí los parámetros específicos de tu transformación
    )
    guardar_grafo_data(
        data=grafo_nuevo_tipo,
        archivo_salida=archivo_salida_nuevo_tipo,
        id_imf=id_imf,
    )

    resultados_imf["[nombre_tipo]"] = {
        "archivo": archivo_salida_nuevo_tipo,
        "num_nodes": grafo_nuevo_tipo.num_nodes,
        "num_edges": grafo_nuevo_tipo.num_edges,
        # Añade aquí metadatos específicos si los guardaste en el objeto Data
        "exito": True,
    }
    print(f"✓ Grafo [NOMBRE_DEL_TIPO] generado exitosamente para {id_imf}")

except Exception as e:
    print(f"✗ Error al generar grafo [NOMBRE_DEL_TIPO] para {id_imf}: {e}")
    resultados_imf["[nombre_tipo]"] = {"exito": False, "error": str(e)}
```

#### Parámetros de `obtener_grafos_all_imf()`

Si tu transformación requiere parámetros adicionales, añádelos a la firma de `obtener_grafos_all_imf()` y pásalos a tu función de transformación.

### Paso 3: Actualizar la Documentación del Módulo

Actualiza el docstring del módulo al inicio de `graph_imf_transform_utils.py` para incluir tu nueva transformación:

```python
"""
Utilidades para transformar IMFs a grafos como objetos Data de PyTorch Geometric.

Este módulo contiene funciones para transformar Intrinsic Mode Functions (IMFs) a diferentes
tipos de grafos: Horizontal Visibility Graph (HVG), Natural Visibility Graph (NVG),
grafo de recurrencia y [NOMBRE_DEL_TIPO].
"""
```

### Paso 4: Crear un Script de Prueba (Opcional pero Recomendado)

Crea un script de prueba en `scripts/` para probar tu nueva transformación de forma independiente. Puedes usar como referencia los scripts existentes:

- `scripts/16nov25/obtener_grafo_hvg_imf.py`
- `scripts/16nov25/obtener_grafo_nvg_imf.py`
- `scripts/16nov25/obtener_grafo_recurrencia_imf.py`

#### Ejemplo de Script de Prueba

```python
"""
Script para transformar una IMF a grafo [NOMBRE_DEL_TIPO] como objeto Data de PyTorch Geometric.

Este script carga una IMF desde un archivo parquet y la transforma a grafo [NOMBRE_DEL_TIPO]
como un objeto Data de PyTorch Geometric.
"""

from pathlib import Path

from GraphEMD.data.graph_imf_transform_utils import obtener_grafo_nuevo_tipo_imf
from GraphEMD.data.python_utils import guardar_grafo_data


if __name__ == "__main__":
    
    # Variables de configuración para probar el método
    proyecto_root = Path(__file__).parent.parent.parent

    archivo_imfs = str(proyecto_root / "data" / "16dic25" / "msci_world_imfs.parquet")
    id_imf = "IMF_1"

    carpeta_salida = proyecto_root / "data" / "16dic25" / "grafos" / "[nombre_tipo]" / id_imf.lower()
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    archivo_salida = str(carpeta_salida / f"grafo_[nombre_tipo]_{id_imf.lower()}")

    # Verificar que el archivo existe
    if not Path(archivo_imfs).exists():
        raise FileNotFoundError(
            f"El archivo {archivo_imfs} no existe. "
            "Asegúrate de que el archivo esté en la ubicación correcta."
        )

    # Probar el método
    print("=" * 60)
    print("TRANSFORMACIÓN DE IMF A GRAFO [NOMBRE_DEL_TIPO]")
    print("=" * 60)

    # Construir el grafo
    grafo = obtener_grafo_nuevo_tipo_imf(
        archivo_imfs=archivo_imfs,
        id_imf=id_imf,
        # Pasa aquí los parámetros específicos
    )

    # Guardar el grafo
    guardar_grafo_data(
        data=grafo,
        archivo_salida=archivo_salida,
        id_imf=id_imf,
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Grafo [NOMBRE_DEL_TIPO] creado exitosamente para {id_imf}")
    print(f"  - Nodos: {grafo.num_nodes}")
    print(f"  - Enlaces: {grafo.num_edges}")
    if grafo.x is not None:
        print(f"  - Features de nodos: {grafo.x.shape}")
```

### Paso 5: Verificar el Formato de Salida

Asegúrate de que tu función retorna un objeto `Data` de PyTorch Geometric con:
- `x`: Tensor de features de nodos de forma `[num_nodes, num_features]`
- `edge_index`: Tensor de enlaces de forma `[2, num_edges]`

Opcionalmente, puedes añadir atributos personalizados al objeto `Data` para guardar metadatos.

### Paso 6: Probar la Integración

Ejecuta `obtener_grafos_all_imf()` con tu nueva transformación para verificar que:
1. Se generan los grafos correctamente
2. Se guardan en la estructura de carpetas esperada
3. Los metadatos se guardan correctamente
4. No hay errores en el procesamiento

## Estructura de Carpetas de Salida

Los grafos se guardan en la siguiente estructura:

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

Cada grafo se guarda en múltiples formatos:
- `*_features.parquet`: Features de nodos
- `*_edges.parquet`: Lista de enlaces
- `*_metadata.csv`: Metadatos del grafo
- `*.pt`: Objeto Data completo serializado con torch

## Ejemplos de Referencia

### Ejemplo Simple: HVG y NVG

Las funciones `obtener_grafo_hvg_imf()` y `obtener_grafo_nvg_imf()` son ejemplos simples que:
- Usan la librería `ts2vg` para construir el grafo
- Cada nodo tiene una sola feature (el valor de la serie temporal)
- No requieren parámetros adicionales complejos

### Ejemplo Complejo: Grafo de Recurrencia

La función `obtener_grafo_recurrencia_imf()` es un ejemplo más complejo que:
- Requiere múltiples pasos (selección de tau, selección de dim, construcción de embedding, cálculo de matriz de recurrencia)
- Tiene múltiples parámetros configurables
- Guarda metadatos adicionales en el objeto Data
- Usa funciones auxiliares para cálculos intermedios

## Convenciones de Código

Sigue las convenciones establecidas en el proyecto:

1. **Nombres de funciones**: Usa `snake_case` y el prefijo `obtener_grafo_` seguido del tipo de grafo y `_imf`
2. **Docstrings**: Usa formato NumPy/SciPy en español
3. **Type hints**: Incluye type hints en todas las funciones
4. **Mensajes informativos**: Usa `print()` para proporcionar información sobre el progreso
5. **Manejo de errores**: Incluye validaciones y mensajes de error claros

## Checklist Final

Antes de considerar tu contribución completa, verifica:

- [ ] La función de transformación está implementada y documentada
- [ ] La función retorna un objeto `Data` válido de PyTorch Geometric
- [ ] La función está integrada en `obtener_grafos_all_imf()`
- [ ] La documentación del módulo está actualizada
- [ ] Se ha creado un script de prueba (opcional pero recomendado)
- [ ] Se ha probado la transformación con datos reales
- [ ] Los grafos se guardan correctamente en la estructura de carpetas esperada
- [ ] Los metadatos se guardan correctamente (si aplica)
- [ ] El código sigue las convenciones del proyecto
- [ ] No hay errores de linting (verifica con black)

## Notas Adicionales

- Si tu transformación requiere nuevas dependencias, asegúrate de añadirlas a los archivos de configuración del proyecto (requirements.txt, setup.py, etc.)
- Si tu transformación es computacionalmente intensiva, considera añadir opciones para paralelización o procesamiento por lotes
- Si tu transformación tiene parámetros que afectan significativamente los resultados, documenta claramente sus efectos y valores recomendados

## Soporte

Si tienes dudas o problemas al implementar una nueva transformación, revisa:
1. Las funciones existentes como referencia
2. La documentación de PyTorch Geometric sobre objetos `Data`
3. Los scripts de prueba existentes

