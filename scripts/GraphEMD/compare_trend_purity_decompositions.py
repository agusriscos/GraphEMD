#!/usr/bin/env python3
"""
Compara la pureza del componente de tendencia entre CEEMDAN, EEMD y VMD.

Carga las descomposiciones persistidas de cada señal financiera (y, opcionalmente,
los escenarios sintéticos emdsynth) y calcula, por método:

- ``r2_regresion_lineal``: bondad de ajuste de una recta temporal.
- Fracciones de incrementos/decrementos (monotonía blanda).
- ``rugosidad_d2``: desviación típica de la segunda diferencia normalizada
  (notación científica, p. ej. ``4.8333e-02``).

Para CEEMDAN y EEMD el componente evaluado es el residuo implícito
(``Residuo`` o ``|x - Σ IMF|``). En VMD con ``DC=1`` se evalúa ``IMF_1``,
que concentra la tendencia.

Ejemplo::

    python scripts/GraphEMD/comparar_pureza_tendencia_descomposiciones.py

    python scripts/GraphEMD/comparar_pureza_tendencia_descomposiciones.py \\
        --activos msci_world,xauusd --incluir-emdsynth
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EMDSYNTH_OUT = _REPO_ROOT / "scripts" / "GraphEMD" / "emdsynth" / "out"
_OUT_DIR = _REPO_ROOT / "scripts" / "GraphEMD" / "out"

TOL_DIFF_MONOTONA: float = 1e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigSenal:
    """
    Metadatos de una señal con descomposiciones CEEMDAN, EEMD y VMD.

    Parameters
    ----------
    id_senal : str
        Identificador corto.
    nombre : str
        Etiqueta legible.
    dir_datos : Path
        Directorio con los parquet ``{prefijo}_imfs_*.parquet``.
    prefijo : str
        Prefijo de archivos IMF.
    ruta_precios : Path | None
        Parquet de precios; si es ``None`` se reconstruye la señal desde IMFs.
    columna_precio : str
        Nombre de la columna de cierre.
    tipo : str
        ``empirica`` o ``sintetica``.
    """

    id_senal: str
    nombre: str
    dir_datos: Path
    prefijo: str
    ruta_precios: Path | None = None
    columna_precio: str = "Close"
    tipo: str = "empirica"


SENALES_EMPRICAS: tuple[ConfigSenal, ...] = (
    ConfigSenal(
        id_senal="msci_world",
        nombre="MSCI World",
        dir_datos=_REPO_ROOT / "data" / "20abr26",
        prefijo="msci_world",
        ruta_precios=_REPO_ROOT / "data" / "20abr26" / "msci_world.parquet",
    ),
    ConfigSenal(
        id_senal="xle",
        nombre="XLE",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xle_etf_analysis",
        prefijo="xle",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xle_etf_analysis"
        / "xle.parquet",
    ),
    ConfigSenal(
        id_senal="xlp",
        nombre="XLP",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis",
        prefijo="xlp",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlp_analysis" / "xlp.parquet",
    ),
    ConfigSenal(
        id_senal="xlv",
        nombre="XLV",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis",
        prefijo="xlv",
        ruta_precios=_REPO_ROOT / "data" / "GraphEMD" / "xlv_analysis" / "xlv.parquet",
    ),
    ConfigSenal(
        id_senal="xauusd",
        nombre="XAU/USD",
        dir_datos=_REPO_ROOT / "data" / "GraphEMD" / "xauusd_analysis",
        prefijo="xauusd",
        ruta_precios=_REPO_ROOT
        / "data"
        / "GraphEMD"
        / "xauusd_analysis"
        / "xauusd.parquet",
    ),
)


def _columnas_imf(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas ``IMF_*`` ordenadas por índice.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla de descomposición.

    Returns
    -------
    list[str]
        Nombres de columnas IMF ordenados.
    """
    return sorted(
        [c for c in df.columns if c.startswith("IMF_")],
        key=lambda nombre: int(nombre.split("_")[1]),
    )


def _alinear_longitud(df: pd.DataFrame, n_objetivo: int) -> pd.DataFrame:
    """
    Recorta un DataFrame al número de observaciones común.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla a alinear.
    n_objetivo : int
        Longitud deseada.

    Returns
    -------
    pd.DataFrame
        Primeras ``n_objetivo`` filas.
    """
    if len(df) == n_objetivo:
        return df
    if len(df) < n_objetivo:
        raise ValueError(
            f"DataFrame con {len(df)} filas; se requieren al menos {n_objetivo}."
        )
    return df.iloc[:n_objetivo].reset_index(drop=True)


def _cargar_serie_original(
    cfg: ConfigSenal,
    df_referencia: pd.DataFrame,
) -> np.ndarray:
    """
    Obtiene la señal original desde precios o reconstrucción IMF+residuo.

    Parameters
    ----------
    cfg : ConfigSenal
        Configuración de la señal.
    df_referencia : pd.DataFrame
        Descomposición de referencia (CEEMDAN) ya alineada.

    Returns
    -------
    np.ndarray
        Vector de la señal original.
    """
    if cfg.ruta_precios is not None and cfg.ruta_precios.is_file():
        df_precios = pd.read_parquet(cfg.ruta_precios, engine="pyarrow")
        if cfg.columna_precio not in df_precios.columns:
            raise ValueError(
                f"Columna {cfg.columna_precio!r} ausente en {cfg.ruta_precios}"
            )
        serie = np.asarray(
            _alinear_longitud(df_precios, len(df_referencia))[
                cfg.columna_precio
            ].values,
            dtype=np.float64,
        )
        return serie

    cols_imf = _columnas_imf(df_referencia)
    suma = np.sum(
        [np.asarray(df_referencia[c].values, dtype=np.float64) for c in cols_imf],
        axis=0,
    )
    if "Residuo" in df_referencia.columns:
        suma += np.asarray(df_referencia["Residuo"].values, dtype=np.float64)
    return suma


def _brecha_sin_residuo(serie: np.ndarray, df_imfs: pd.DataFrame) -> np.ndarray:
    """
    Calcula ``|x - Σ IMF|`` (brecha de tendencia sin columna Residuo).

    Parameters
    ----------
    serie : np.ndarray
        Señal original.
    df_imfs : pd.DataFrame
        Descomposición con columnas ``IMF_*``.

    Returns
    -------
    np.ndarray
        Magnitud de la brecha.
    """
    cols = _columnas_imf(df_imfs)
    suma_imfs = np.sum(
        [np.asarray(df_imfs[c].values, dtype=np.float64) for c in cols],
        axis=0,
    )
    return np.abs(serie - suma_imfs)


def _leer_vmd_dc(cfg: ConfigSenal) -> int:
    """
    Lee el valor ``DC`` usado en la calibración VMD del activo.

    Parameters
    ----------
    cfg : ConfigSenal
        Señal empírica.

    Returns
    -------
    int
        ``DC`` calibrado; por defecto 1 si no hay JSON.
    """
    ruta_json = cfg.dir_datos / f"{cfg.prefijo}_vmd_parametros.json"
    if not ruta_json.is_file():
        return 1
    with ruta_json.open(encoding="utf-8") as fichero:
        datos = json.load(fichero)
    return int(datos.get("calibracion", {}).get("mejor", {}).get("DC", 1))


def _extraer_componente_tendencia(
    metodo: str,
    serie: np.ndarray,
    df_imfs: pd.DataFrame,
    vmd_dc: int = 1,
) -> tuple[np.ndarray, str]:
    """
    Extrae el componente lento evaluable para cada método.

    Parameters
    ----------
    metodo : str
        ``ceemdan``, ``eemd`` o ``vmd``.
    serie : np.ndarray
        Señal original.
    df_imfs : pd.DataFrame
        Descomposición del método.
    vmd_dc : int
        Parámetro ``DC`` de VMD.

    Returns
    -------
    tuple[np.ndarray, str]
        Serie de tendencia y descripción de la definición usada.
    """
    metodo_norm = metodo.lower()
    if metodo_norm == "ceemdan":
        if "Residuo" in df_imfs.columns:
            componente = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
            definicion = "columna Residuo"
        else:
            componente = _brecha_sin_residuo(serie, df_imfs)
            definicion = "|x - Σ IMF|"
        return componente, definicion

    if metodo_norm == "eemd":
        componente = _brecha_sin_residuo(serie, df_imfs)
        return componente, "|x - Σ IMF|"

    if metodo_norm == "vmd":
        cols = _columnas_imf(df_imfs)
        if vmd_dc == 1 and len(cols) >= 1:
            componente = np.asarray(df_imfs[cols[0]].values, dtype=np.float64)
            return componente, f"{cols[0]} (VMD DC=1)"
        if "Residuo" in df_imfs.columns:
            componente = np.asarray(df_imfs["Residuo"].values, dtype=np.float64)
            return componente, "columna Residuo"
        componente = _brecha_sin_residuo(serie, df_imfs)
        return componente, "|x - Σ IMF|"

    raise ValueError(f"Método no soportado: {metodo!r}")


def calcular_metricas_monotonia(serie: np.ndarray) -> dict[str, Any]:
    """
    Calcula fracciones de monotonía del componente de tendencia.

    Parameters
    ----------
    serie : np.ndarray
        Componente lento a evaluar.

    Returns
    -------
    dict
        Fracciones de diferencias positivas, negativas y de mismo signo.
    """
    y = np.asarray(serie, dtype=np.float64)
    diffs = np.diff(y)
    frac_creciente = float(np.mean(diffs > TOL_DIFF_MONOTONA))
    frac_decreciente = float(np.mean(diffs < -TOL_DIFF_MONOTONA))
    frac_mismo_signo = max(frac_creciente, frac_decreciente)
    return {
        "frac_diff_creciente": frac_creciente,
        "frac_diff_decreciente": frac_decreciente,
        "frac_diff_mismo_signo": frac_mismo_signo,
    }


def calcular_rugosidad_d2(serie: np.ndarray) -> float:
    """
    Mide rugosidad como ``std(Δ²y) / std(y)``.

    Parameters
    ----------
    serie : np.ndarray
        Componente lento.

    Returns
    -------
    float
        Rugosidad normalizada; valores bajos indican tendencia más lisa.
    """
    y = np.asarray(serie, dtype=np.float64)
    if len(y) < 3:
        return float("nan")
    segunda_diff = np.diff(y, n=2)
    return float(np.std(segunda_diff) / (np.std(y) + 1e-15))


def formatear_notacion_cientifica(valor: float, decimales: int = 4) -> str:
    """
    Formatea un escalar en notación científica (p. ej. ``1.2786e-06``).

    Parameters
    ----------
    valor : float
        Número a formatear.
    decimales : int
        Cifras decimales en la mantisa.

    Returns
    -------
    str
        Representación en notación científica o ``"nan"`` si no es finito.
    """
    if not np.isfinite(valor):
        return "nan"
    return f"{valor:.{decimales}e}"


def calcular_metricas_pureza_tendencia(
    componente: np.ndarray,
    serie_original: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calcula R² lineal, monotonía y rugosidad de un componente de tendencia.

    Parameters
    ----------
    componente : np.ndarray
        Serie del componente lento.
    serie_original : np.ndarray | None
        Señal de precios para correlación opcional.

    Returns
    -------
    dict
        Métricas de pureza de tendencia.
    """
    y = np.asarray(componente, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, y, 1)
    ajuste_lineal = coef[0] * t + coef[1]
    ss_res = float(np.sum((y - ajuste_lineal) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-15
    r2_lineal = 1.0 - ss_res / ss_tot
    rho, p_spearman = spearmanr(t, y)

    rugosidad = calcular_rugosidad_d2(y)
    metricas = {
        "n_observaciones": int(n),
        "media": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)),
        "r2_regresion_lineal": float(r2_lineal),
        "pendiente_regresion_temporal": float(coef[0]),
        "spearman_tiempo": float(rho),
        "p_valor_spearman": float(p_spearman),
        "rugosidad_d2": formatear_notacion_cientifica(rugosidad),
    }
    metricas.update(calcular_metricas_monotonia(y))

    if serie_original is not None:
        close = np.asarray(serie_original, dtype=np.float64)
        metricas["correlacion_con_senal_original"] = float(np.corrcoef(y, close)[0, 1])

    return metricas


def _ruta_parquet_imfs(cfg: ConfigSenal, metodo: str) -> Path:
    """
    Construye la ruta al parquet de IMFs de un método.

    Parameters
    ----------
    cfg : ConfigSenal
        Señal.
    metodo : str
        ``ceemdan``, ``eemd`` o ``vmd``.

    Returns
    -------
    Path
        Ruta esperada del parquet.
    """
    return cfg.dir_datos / f"{cfg.prefijo}_imfs_{metodo}.parquet"


def evaluar_senal(cfg: ConfigSenal) -> dict[str, Any]:
    """
    Evalúa pureza de tendencia para CEEMDAN, EEMD y VMD de una señal.

    Parameters
    ----------
    cfg : ConfigSenal
        Configuración de la señal.

    Returns
    -------
    dict
        Informe con métricas por método y metadatos.
    """
    rutas = {m: _ruta_parquet_imfs(cfg, m) for m in ("ceemdan", "eemd", "vmd")}
    faltantes = [m for m, ruta in rutas.items() if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan descomposiciones {faltantes} para {cfg.id_senal} en {cfg.dir_datos}"
        )

    dataframes = {
        metodo: pd.read_parquet(ruta, engine="pyarrow")
        for metodo, ruta in rutas.items()
    }
    n_comun = min(len(df) for df in dataframes.values())
    dataframes = {
        metodo: _alinear_longitud(df, n_comun) for metodo, df in dataframes.items()
    }

    serie = _cargar_serie_original(cfg, dataframes["ceemdan"])
    if len(serie) != n_comun:
        serie = serie[:n_comun]

    vmd_dc = _leer_vmd_dc(cfg) if cfg.tipo == "empirica" else 1
    resultados_metodos: dict[str, Any] = {}

    for metodo, df_imfs in dataframes.items():
        componente, definicion = _extraer_componente_tendencia(
            metodo,
            serie,
            df_imfs,
            vmd_dc=vmd_dc,
        )
        metricas = calcular_metricas_pureza_tendencia(componente, serie)
        resultados_metodos[metodo] = {
            "definicion_componente": definicion,
            "metricas": metricas,
        }

    return {
        "id_senal": cfg.id_senal,
        "nombre": cfg.nombre,
        "tipo": cfg.tipo,
        "n_observaciones": n_comun,
        "vmd_dc": vmd_dc,
        "metodos": resultados_metodos,
    }


def _cargar_escenarios_emdsynth() -> list[ConfigSenal]:
    """
    Construye configuraciones para escenarios sintéticos con los tres métodos.

    Returns
    -------
    list[ConfigSenal]
        Lista de señales sintéticas detectadas en ``emdsynth/out``.
    """
    manifest = _EMDSYNTH_OUT / "manifest_emdsynth.json"
    if not manifest.is_file():
        logger.warning(
            "No se encontró manifest emdsynth; se omiten señales sintéticas."
        )
        return []

    with manifest.open(encoding="utf-8") as fichero:
        datos = json.load(fichero)

    escenarios: list[ConfigSenal] = []
    for item in datos.get("escenarios", []):
        id_escenario = item["id"]
        ruta_ceemdan = _EMDSYNTH_OUT / f"{id_escenario}_imfs_ceemdan.parquet"
        if not ruta_ceemdan.is_file():
            continue
        escenarios.append(
            ConfigSenal(
                id_senal=id_escenario,
                nombre=item.get("descripcion", id_escenario),
                dir_datos=_EMDSYNTH_OUT,
                prefijo=id_escenario,
                ruta_precios=None,
                tipo="sintetica",
            )
        )
    return escenarios


def _informe_a_filas(informe: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Aplana un informe por señal a filas tabulares.

    Parameters
    ----------
    informe : dict
        Salida de ``evaluar_senal``.

    Returns
    -------
    list[dict]
        Filas listas para DataFrame.
    """
    filas: list[dict[str, Any]] = []
    for metodo, bloque in informe["metodos"].items():
        fila = {
            "id_senal": informe["id_senal"],
            "nombre": informe["nombre"],
            "tipo": informe["tipo"],
            "metodo": metodo,
            "definicion_componente": bloque["definicion_componente"],
            "n_observaciones": informe["n_observaciones"],
            "vmd_dc": informe.get("vmd_dc"),
        }
        fila.update(bloque["metricas"])
        filas.append(fila)
    return filas


def ejecutar_comparativa(
    senales: list[ConfigSenal],
    ruta_json: Path,
    ruta_csv: Path,
) -> dict[str, Any]:
    """
    Ejecuta la comparativa para todas las señales y persiste JSON y CSV.

    Parameters
    ----------
    senales : list[ConfigSenal]
        Señales a evaluar.
    ruta_json : Path
        Destino del informe agregado.
    ruta_csv : Path
        Destino de la tabla plana.

    Returns
    -------
    dict
        Informe completo.
    """
    informes: list[dict[str, Any]] = []
    filas: list[dict[str, Any]] = []
    errores: list[dict[str, str]] = []

    for cfg in senales:
        logger.info("Evaluando %s (%s)...", cfg.nombre, cfg.id_senal)
        try:
            informe = evaluar_senal(cfg)
            informes.append(informe)
            filas.extend(_informe_a_filas(informe))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en %s: %s", cfg.id_senal, exc)
            errores.append({"id_senal": cfg.id_senal, "error": str(exc)})

    salida = {
        "descripcion": (
            "Métricas de pureza de tendencia: R² lineal, fracción de incrementos "
            "y rugosidad_d2 para CEEMDAN (Residuo), EEMD (|x-ΣIMF|) y VMD (IMF_1 si DC=1)."
        ),
        "n_senales_ok": len(informes),
        "n_errores": len(errores),
        "senales": informes,
        "errores": errores,
    }

    ruta_json.parent.mkdir(parents=True, exist_ok=True)
    with ruta_json.open("w", encoding="utf-8") as fichero:
        json.dump(salida, fichero, indent=2, ensure_ascii=False)

    if filas:
        pd.DataFrame(filas).to_csv(ruta_csv, index=False)
        logger.info("CSV guardado: %s", ruta_csv)

    logger.info("JSON guardado: %s", ruta_json)
    return salida


def _parsear_argumentos() -> argparse.Namespace:
    """
    Define y parsea argumentos de línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compara pureza de tendencia (R², fracción de incrementos, rugosidad) entre "
            "CEEMDAN, EEMD y VMD para todas las señales del proyecto."
        )
    )
    parser.add_argument(
        "--activos",
        type=str,
        default="",
        help="IDs separados por coma (p. ej. msci_world,xauusd). Vacío = todos.",
    )
    parser.add_argument(
        "--incluir-emdsynth",
        action="store_true",
        help="Incluye escenarios sintéticos de emdsynth/out.",
    )
    parser.add_argument(
        "--salida-json",
        type=Path,
        default=_OUT_DIR / "metricas_pureza_tendencia_descomposiciones.json",
        help="Ruta del informe JSON.",
    )
    parser.add_argument(
        "--salida-csv",
        type=Path,
        default=_OUT_DIR / "metricas_pureza_tendencia_descomposiciones.csv",
        help="Ruta de la tabla CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada: evalúa señales y escribe resultados.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parsear_argumentos()

    senales = list(SENALES_EMPRICAS)
    if args.activos.strip():
        ids = {s.strip() for s in args.activos.split(",") if s.strip()}
        senales = [s for s in senales if s.id_senal in ids]
        desconocidos = ids - {s.id_senal for s in senales}
        if desconocidos:
            raise ValueError(f"IDs de activo no reconocidos: {sorted(desconocidos)}")

    if args.incluir_emdsynth:
        senales.extend(_cargar_escenarios_emdsynth())

    if not senales:
        raise ValueError("No hay señales que evaluar.")

    ejecutar_comparativa(senales, args.salida_json, args.salida_csv)


if __name__ == "__main__":
    main()
