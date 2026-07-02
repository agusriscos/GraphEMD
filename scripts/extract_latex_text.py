"""
Script to extract text from a LaTeX file and save it in .txt format.

This script reads a LaTeX file, removes LaTeX commands, and extracts
the textual content, saving it to a .txt file in a readable format.
"""

import re
import sys
from pathlib import Path


def limpiar_comandos_latex(texto: str) -> str:
    """
    Remove LaTeX commands from text, keeping only the textual content.

    Parameters
    ----------
    texto : str
        Text containing LaTeX commands to clean.

    Returns
    -------
    str
        Clean text without LaTeX commands.
    """
    # Remove the entire preamble (up to \begin{document})
    texto = re.sub(r".*?\\begin\{document\}", "", texto, flags=re.DOTALL)

    # Remove \end{document} and everything that follows
    texto = re.sub(r"\\end\{document\}.*", "", texto, flags=re.DOTALL)

    # Remove section, subsection, etc. commands but keep the title
    texto = re.sub(r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}", r"\2", texto)

    # Remove formatting commands but keep the content
    texto = re.sub(r"\\(textbf|textit|emph|texttt)\{([^}]+)\}", r"\2", texto)

    # Remove references and citations
    texto = re.sub(r"\\(citep|citet|ref)\{[^}]+\}", "", texto)
    texto = re.sub(r"~\\ref\{[^}]+\}", "", texto)

    # Remove label commands
    texto = re.sub(r"\\label\{[^}]+\}", "", texto)

    # Remove equation environments (keep only basic content)
    texto = re.sub(r"\\begin\{equation\}[^\\]*\\end\{equation\}", "", texto, flags=re.DOTALL)

    # Remove itemize, enumerate, etc. environments but keep the items
    texto = re.sub(r"\\begin\{(itemize|enumerate|description)\}", "", texto)
    texto = re.sub(r"\\end\{(itemize|enumerate|description)\}", "", texto)
    texto = re.sub(r"\\item\s*\\(textbf)?\{([^}]+)\}", r"- \2", texto)
    texto = re.sub(r"\\item\s+", "- ", texto)

    # Remove table commands
    texto = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", texto, flags=re.DOTALL)

    # Remove bibliography commands
    texto = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"\\bibitem\{[^}]+\}", "", texto)
    texto = re.sub(r"\\newblock", "", texto)

    # Remove additional formatting commands
    texto = re.sub(r"\\(maketitle|tableofcontents|newpage|centering|caption\{[^}]+\})", "", texto)

    # Remove commands with braces but no useful content
    texto = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", texto)

    # Remove simple commands without arguments
    texto = re.sub(r"\\([a-zA-Z]+)", "", texto)

    # Clean up multiple spaces and excessive line breaks
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r" +", " ", texto)

    # Trim leading and trailing whitespace on each line
    lineas = [linea.strip() for linea in texto.split("\n")]
    texto = "\n".join(linea for linea in lineas if linea)

    return texto


def extract_latex_text(archivo_entrada: str, archivo_salida: str) -> None:
    """
    Extract text from a LaTeX file and save it to a .txt file.

    Parameters
    ----------
    archivo_entrada : str
        Path to the input LaTeX file.
    archivo_salida : str
        Path to the output .txt file.

    Examples
    --------
    >>> extract_latex_text("main.tex", "main.txt")
    """
    ruta_entrada = Path(archivo_entrada)

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"File {archivo_entrada} does not exist.")

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        contenido = f.read()

    texto_limpio = limpiar_comandos_latex(contenido)

    ruta_salida = Path(archivo_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(texto_limpio)

    print(f"Text successfully extracted from {archivo_entrada}")
    print(f"Saved to {archivo_salida}")


def main() -> None:
    """
    Main entry point for the script.

    Reads command-line arguments and runs the text extraction.
    """
    if len(sys.argv) < 2:
        print("Usage: python extract_latex_text.py <latex_file> [output_file]")
        print("Example: python extract_latex_text.py main.tex main.txt")
        sys.exit(1)

    archivo_entrada = sys.argv[1]

    if len(sys.argv) >= 3:
        archivo_salida = sys.argv[2]
    else:
        # If no output is specified, use the same name with a .txt extension
        ruta_entrada = Path(archivo_entrada)
        archivo_salida = ruta_entrada.with_suffix(".txt")

    try:
        extract_latex_text(archivo_entrada, str(archivo_salida))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
