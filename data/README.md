# Datos del proyecto GraphEMD

Los artefactos binarios (Parquet, CSV, PNG, HTML, `.pt`) no se versionan en Git.
Regenera los datos con los scripts en `scripts/GraphEMD/` o copia desde ARPTools.

## Estructura esperada

```
data/
├── 20abr26/                    # MSCI World (ventana principal del paper)
│   ├── msci_world.parquet
│   ├── msci_world_imfs_ceemdan.parquet
│   ├── imfs_ceemdan_dim_red/   # salidas ICA/PCA
│   └── grafos_ica/             # metadatos de grafos sobre componentes ICA
└── GraphEMD/                   # Panel sectorial + oro
    ├── xle_etf_analysis/
    ├── xlp_analysis/
    ├── xlv_analysis/
    └── xauusd_analysis/
```

## Ventana temporal del paper

- Inicio: 2012-01-12
- Fin: 2026-04-20
- Activos: MSCI World, XLE, XLP, XLV, XAU/USD

## Copiar desde ARPTools (sin modificar origen)

```bash
rsync -a ../ARPTools/data/20abr26/ ./20abr26/
rsync -a ../ARPTools/data/GraphEMD/ ./GraphEMD/
```
