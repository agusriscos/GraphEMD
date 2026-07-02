"""
Script para extraer texto de un archivo LaTeX y guardarlo en formato .txt.

Este script lee un archivo LaTeX, elimina los comandos de LaTeX y extrae
el contenido textual, guardándolo en un archivo .txt con formato legible.
"""

import re
import sys
from pathlib import Path


def limpiar_comandos_latex(texto: str) -> str:
    """
    Elimina comandos de LaTeX del texto, manteniendo solo el contenido textual.

    Parameters
    ----------
    texto : str
        Texto con comandos de LaTeX a limpiar.

    Returns
    -------
    str
        Texto limpio sin comandos de LaTeX.
    """
    # Eliminar todo el preámbulo (hasta \begin{document})
    texto = re.sub(r".*?\\begin\{document\}", "", texto, flags=re.DOTALL)

    # Eliminar \end{document} y todo lo que sigue
    texto = re.sub(r"\\end\{document\}.*", "", texto, flags=re.DOTALL)

    # Eliminar comandos de sección, subsección, etc. pero mantener el título
    texto = re.sub(r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}", r"\2", texto)

    # Eliminar comandos de formato pero mantener el contenido
    texto = re.sub(r"\\(textbf|textit|emph|texttt)\{([^}]+)\}", r"\2", texto)

    # Eliminar referencias y citas
    texto = re.sub(r"\\(citep|citet|ref)\{[^}]+\}", "", texto)
    texto = re.sub(r"~\\ref\{[^}]+\}", "", texto)

    # Eliminar comandos de label
    texto = re.sub(r"\\label\{[^}]+\}", "", texto)

    # Eliminar entornos de ecuaciones (mantener solo el contenido básico)
    texto = re.sub(r"\\begin\{equation\}[^\\]*\\end\{equation\}", "", texto, flags=re.DOTALL)

    # Eliminar entornos de itemize, enumerate, etc. pero mantener los items
    texto = re.sub(r"\\begin\{(itemize|enumerate|description)\}", "", texto)
    texto = re.sub(r"\\end\{(itemize|enumerate|description)\}", "", texto)
    texto = re.sub(r"\\item\s*\\(textbf)?\{([^}]+)\}", r"- \2", texto)
    texto = re.sub(r"\\item\s+", "- ", texto)

    # Eliminar comandos de tabla
    texto = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", texto, flags=re.DOTALL)

    # Eliminar comandos de bibliografía
    texto = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\\bibitem\{[^}]+\}", "", texto)
    texto = re.sub(r"\\newblock", "", texto)

    # Eliminar comandos de formato adicionales
    texto = re.sub(r"\\(maketitle|tableofcontents|newpage|centering|caption\{[^}]+\})", "", texto)

    # Eliminar comandos con llaves pero sin contenido útil
    texto = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", texto)

    # Eliminar comandos simples sin argumentos
    texto = re.sub(r"\\([a-zA-Z]+)", "", texto)

    # Limpiar espacios múltiples y saltos de línea excesivos
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r" +", " ", texto)

    # Limpiar espacios al inicio y final de líneas
    lineas = [linea.strip() for linea in texto.split("\n")]
    texto = "\n".join(linea for linea in lineas if linea)

    return texto


def extraer_texto_latex(archivo_entrada: str, archivo_salida: str) -> None:
    """
    Extrae el texto de un archivo LaTeX y lo guarda en un archivo .txt.

    Parameters
    ----------
    archivo_entrada : str
        Ruta al archivo LaTeX de entrada.
    archivo_salida : str
        Ruta al archivo .txt de salida.

    Examples
    --------
    >>> extraer_texto_latex("main.tex", "main.txt")
    """
    ruta_entrada = Path(archivo_entrada)

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"El archivo {archivo_entrada} no existe.")

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        contenido = f.read()

    texto_limpio = limpiar_comandos_latex(contenido)

    ruta_salida = Path(archivo_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(texto_limpio)

    print(f"Texto extraído exitosamente de {archivo_entrada}")
    print(f"Guardado en {archivo_salida}")


def main() -> None:
    """
    Función principal del script.

    Lee los argumentos de línea de comandos y ejecuta la extracción de texto.
    """
    if len(sys.argv) < 2:
        print("Uso: python extraer_texto_latex.py <archivo_latex> [archivo_salida]")
        print("Ejemplo: python extraer_texto_latex.py main.tex main.txt")
        sys.exit(1)

    archivo_entrada = sys.argv[1]

    if len(sys.argv) >= 3:
        archivo_salida = sys.argv[2]
    else:
        # Si no se especifica salida, usar el mismo nombre con extensión .txt
        ruta_entrada = Path(archivo_entrada)
        archivo_salida = ruta_entrada.with_suffix(".txt")

    try:
        extraer_texto_latex(archivo_entrada, str(archivo_salida))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()



