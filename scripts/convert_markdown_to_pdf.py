"""
Script to convert a Markdown file to PDF.

This script converts a Markdown file to PDF format using
Python libraries for markdown processing and PDF generation.
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


def convert_markdown_to_html(archivo_markdown: str) -> str:
    """
    Convert a Markdown file to HTML.

    Parameters
    ----------
    archivo_markdown : str
        Path to the input Markdown file.

    Returns
    -------
    str
        HTML content generated from the Markdown.

    Examples
    --------
    >>> html = convert_markdown_to_html("document.md")
    """
    ruta_archivo = Path(archivo_markdown)

    if not ruta_archivo.exists():
        raise FileNotFoundError(f"File {archivo_markdown} does not exist.")

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido_md = f.read()

    if markdown is None:
        # If markdown is not available, use basic conversion
        html = f"<html><head><meta charset='utf-8'><title>Document</title></head><body><pre>{contenido_md}</pre></body></html>"
    else:
        # Convert markdown to HTML with extensions
        md = markdown.Markdown(extensions=["extra", "codehilite", "tables"])
        html_body = md.convert(contenido_md)

        # Build full HTML with styles
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
    Convert HTML to PDF using WeasyPrint.

    Parameters
    ----------
    html : str
        HTML content to convert.
    archivo_pdf : str
        Path to the output PDF file.

    Examples
    --------
    >>> convertir_html_a_pdf_con_weasyprint(html, "document.pdf")
    """
    html_doc = HTML(string=html)
    html_doc.write_pdf(archivo_pdf)


def convertir_html_a_pdf_con_pdfkit(html: str, archivo_pdf: str) -> None:
    """
    Convert HTML to PDF using pdfkit (requires wkhtmltopdf).

    Parameters
    ----------
    html : str
        HTML content to convert.
    archivo_pdf : str
        Path to the output PDF file.

    Examples
    --------
    >>> convertir_html_a_pdf_con_pdfkit(html, "document.pdf")
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


def convert_markdown_to_pdf(
    archivo_markdown: str, archivo_pdf: Optional[str] = None
) -> None:
    """
    Convert a Markdown file to PDF.

    Parameters
    ----------
    archivo_markdown : str
        Path to the input Markdown file.
    archivo_pdf : str, optional
        Path to the output PDF file. If not specified,
        the same name with a .pdf extension is used.

    Examples
    --------
    >>> convert_markdown_to_pdf("document.md", "document.pdf")
    """
    if archivo_pdf is None:
        ruta_entrada = Path(archivo_markdown)
        archivo_pdf = str(ruta_entrada.with_suffix(".pdf"))

    # Convert markdown to HTML
    html = convert_markdown_to_html(archivo_markdown)

    # Convert HTML to PDF
    if WEASYPRINT_AVAILABLE:
        convertir_html_a_pdf_con_weasyprint(html, archivo_pdf)
        print(f"PDF successfully generated using WeasyPrint: {archivo_pdf}")
    elif PDFKIT_AVAILABLE:
        try:
            convertir_html_a_pdf_con_pdfkit(html, archivo_pdf)
            print(f"PDF successfully generated using pdfkit: {archivo_pdf}")
        except Exception as e:
            raise RuntimeError(
                f"Error generating PDF with pdfkit: {e}. "
                "Make sure wkhtmltopdf is installed."
            ) from e
    else:
        raise RuntimeError(
            "No PDF generation library found. "
            "Install one of the following options:\n"
            "  - weasyprint: pip install weasyprint\n"
            "  - pdfkit: pip install pdfkit (requires wkhtmltopdf)"
        )


def main() -> None:
    """
    Main entry point for the script.

    Reads command-line arguments and runs the conversion.
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_markdown_to_pdf.py <markdown_file> [pdf_file]")
        print("Example: python convert_markdown_to_pdf.py document.md document.pdf")
        sys.exit(1)

    archivo_markdown = sys.argv[1]

    if len(sys.argv) >= 3:
        archivo_pdf = sys.argv[2]
    else:
        archivo_pdf = None

    try:
        convert_markdown_to_pdf(archivo_markdown, archivo_pdf)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
