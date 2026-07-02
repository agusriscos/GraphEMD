"""
Script para convertir un archivo Markdown a PDF.

Este script convierte un archivo Markdown a formato PDF utilizando
librerías de Python para procesamiento de markdown y generación de PDFs.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import markdown
except ImportError:
    markdown = None

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False


def convertir_markdown_a_html(archivo_markdown: str) -> str:
    """
    Convierte un archivo Markdown a HTML.

    Parameters
    ----------
    archivo_markdown : str
        Ruta al archivo Markdown de entrada.

    Returns
    -------
    str
        Contenido HTML generado desde el Markdown.

    Examples
    --------
    >>> html = convertir_markdown_a_html("documento.md")
    """
    ruta_archivo = Path(archivo_markdown)

    if not ruta_archivo.exists():
        raise FileNotFoundError(f"El archivo {archivo_markdown} no existe.")

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido_md = f.read()

    if markdown is None:
        # Si markdown no está disponible, usar conversión básica
        html = f"<html><head><meta charset='utf-8'><title>Document</title></head><body><pre>{contenido_md}</pre></body></html>"
    else:
        # Convertir markdown a HTML con extensiones
        md = markdown.Markdown(extensions=["extra", "codehilite", "tables"])
        html_body = md.convert(contenido_md)

        # Crear HTML completo con estilos
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Document</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            line-height: 1.6;
        }}
        h1 {{
            font-size: 24pt;
            margin-top: 24pt;
            margin-bottom: 12pt;
        }}
        h2 {{
            font-size: 18pt;
            margin-top: 18pt;
            margin-bottom: 9pt;
        }}
        h3 {{
            font-size: 14pt;
            margin-top: 14pt;
            margin-bottom: 7pt;
        }}
        p {{
            margin-bottom: 12pt;
            text-align: justify;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

    return html


def convertir_html_a_pdf_con_weasyprint(html: str, archivo_pdf: str) -> None:
    """
    Convierte HTML a PDF usando WeasyPrint.

    Parameters
    ----------
    html : str
        Contenido HTML a convertir.
    archivo_pdf : str
        Ruta al archivo PDF de salida.

    Examples
    --------
    >>> convertir_html_a_pdf_con_weasyprint(html, "documento.pdf")
    """
    html_doc = HTML(string=html)
    html_doc.write_pdf(archivo_pdf)


def convertir_html_a_pdf_con_pdfkit(html: str, archivo_pdf: str) -> None:
    """
    Convierte HTML a PDF usando pdfkit (requiere wkhtmltopdf).

    Parameters
    ----------
    html : str
        Contenido HTML a convertir.
    archivo_pdf : str
        Ruta al archivo PDF de salida.

    Examples
    --------
    >>> convertir_html_a_pdf_con_pdfkit(html, "documento.pdf")
    """
    options = {
        "page-size": "A4",
        "margin-top": "0.75in",
        "margin-right": "0.75in",
        "margin-bottom": "0.75in",
        "margin-left": "0.75in",
        "encoding": "UTF-8",
    }
    pdfkit.from_string(html, archivo_pdf, options=options)


def convertir_markdown_a_pdf(
    archivo_markdown: str, archivo_pdf: Optional[str] = None
) -> None:
    """
    Convierte un archivo Markdown a PDF.

    Parameters
    ----------
    archivo_markdown : str
        Ruta al archivo Markdown de entrada.
    archivo_pdf : str, optional
        Ruta al archivo PDF de salida. Si no se especifica,
        se usa el mismo nombre con extensión .pdf.

    Examples
    --------
    >>> convertir_markdown_a_pdf("documento.md", "documento.pdf")
    """
    if archivo_pdf is None:
        ruta_entrada = Path(archivo_markdown)
        archivo_pdf = str(ruta_entrada.with_suffix(".pdf"))

    # Convertir markdown a HTML
    html = convertir_markdown_a_html(archivo_markdown)

    # Convertir HTML a PDF
    if WEASYPRINT_AVAILABLE:
        convertir_html_a_pdf_con_weasyprint(html, archivo_pdf)
        print(f"PDF generado exitosamente usando WeasyPrint: {archivo_pdf}")
    elif PDFKIT_AVAILABLE:
        try:
            convertir_html_a_pdf_con_pdfkit(html, archivo_pdf)
            print(f"PDF generado exitosamente usando pdfkit: {archivo_pdf}")
        except Exception as e:
            raise RuntimeError(
                f"Error al generar PDF con pdfkit: {e}. "
                "Asegúrate de que wkhtmltopdf esté instalado."
            ) from e
    else:
        raise RuntimeError(
            "No se encontró ninguna librería para generar PDFs. "
            "Instala una de las siguientes opciones:\n"
            "  - weasyprint: pip install weasyprint\n"
            "  - pdfkit: pip install pdfkit (requiere wkhtmltopdf)"
        )


def main() -> None:
    """
    Función principal del script.

    Lee los argumentos de línea de comandos y ejecuta la conversión.
    """
    if len(sys.argv) < 2:
        print("Uso: python convertir_markdown_a_pdf.py <archivo_markdown> [archivo_pdf]")
        print("Ejemplo: python convertir_markdown_a_pdf.py documento.md documento.pdf")
        sys.exit(1)

    archivo_markdown = sys.argv[1]

    if len(sys.argv) >= 3:
        archivo_pdf = sys.argv[2]
    else:
        archivo_pdf = None

    try:
        convertir_markdown_a_pdf(archivo_markdown, archivo_pdf)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


