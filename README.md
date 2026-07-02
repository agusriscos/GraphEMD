# GraphEMD

Repositorio del artículo *Empirical Mode Decomposition and Graph Transformation of Financial Series* (Agustín M. de los Riscos, Ana Lazcano, Julio E. Sandubete).

Código migrado desde [ARPTools](https://github.com/agusriscos/ARPTools) sin modificar el origen. El manuscrito LaTeX vive en el repositorio hermano `PAPER/`.

## Contenido

| Directorio | Descripción |
|------------|-------------|
| `src/python/GraphEMD/` | Librería: transformación IMF→grafo, señales sintéticas, modelo autoencoder |
| `src/python/CommonUtils/` | Utilidades mínimas compartidas (`DictClass`) |
| `scripts/GraphEMD/` | Pipelines del paper: panel empírico, emdsynth, exploración MSCI |
| `scripts/16dic25/` | Scripts legacy MSCI World (dic-2025) |
| `analysis/` | Cuadernos Jupyter de experimentos |
| `docs/` | Informes técnicos LaTeX (`16dic25`, `20abr26`) |
| `data/` | Datos intermedios (no versionados; ver `data/README.md`) |

## Instalación

```bash
cd GraphEMD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e src/python
```

## Pipelines principales del paper

```bash
# Panel empírico completo (MSCI + XLE/XLP/XLV/XAUUSD)
PYTHONPATH=src/python python scripts/GraphEMD/ejecutar_vmd_todos_activos.py

# Benchmark sintético (EMD/EEMD/CEEMDAN/VMD)
PYTHONPATH=src/python python scripts/GraphEMD/emdsynth/ejecutar_descomposiciones_emdsynth.py

# Figuras para el artículo (salida en PAPER/figures/)
PYTHONPATH=src/python python scripts/GraphEMD/generar_figura_descomposicion_panel_activos.py
PYTHONPATH=src/python python scripts/GraphEMD/generar_figura_ica_panel_activos.py
```

## Datos existentes en ARPTools

Si ya tienes artefactos generados en ARPTools, puedes copiarlos sin tocar el origen:

```bash
rsync -a ARPTools/data/20abr26/ GraphEMD/data/20abr26/
rsync -a ARPTools/data/GraphEMD/ GraphEMD/data/GraphEMD/
```

## Relación con PAPER

- Figuras del artículo: `../PAPER/figures/`
- Scripts de póster: `../PAPER/scripts/` (apuntan a `GraphEMD/data/` y `GraphEMD/docs/`)

## Licencia

Apache-2.0 — ver [LICENSE](LICENSE).
