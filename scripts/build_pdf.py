"""Convert docs/LEARNING_GUIDE.md into docs/LEARNING_GUIDE.pdf.

Pipeline:  markdown --pandoc--> Typst source --typst--> PDF

* pandoc (installed by ``pip install pypandoc_binary``) converts the markdown,
  including the maths, into Typst markup, using docs/pdf/template.typ for the
  title page, table of contents, page numbers and styling.
* Typst (installed by ``pip install typst``) is a modern typesetting system,
  a small and fast alternative to LaTeX. It highlights the code and lays out the pages.

Usage (from the repository root):

    python scripts/build_guide.py      # first refresh the markdown from docs/guide_src/
    python scripts/build_pdf.py        # then build the PDF
    python scripts/build_pdf.py --png  # also render every page to docs/build/pages/ for checking
"""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

import pypandoc
import typst
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BUILD = DOCS / "build"
TITLE = "The Mini-GPT Learning Guide"


def prepare_markdown(text: str) -> str:
    """Remove the parts the PDF template provides itself and fix image paths."""
    # The PDF has its own title page and table of contents.
    text = re.sub(r"<!-- toc:start -->.*?<!-- toc:end -->", "", text, flags=re.DOTALL)
    text = re.sub(r"<!-- GENERATED FILE.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(rf"^# {re.escape(TITLE)}\n+\*[^\n]*\*\n", "", text, count=1, flags=re.MULTILINE)

    # Image paths in the markdown are relative to docs/. Typst wants them relative
    # to the project root (written with a leading slash).
    def absolute(match: re.Match) -> str:
        """Rewrite one image link so that its path starts at the repository root."""
        target = (DOCS / match.group(2)).resolve().relative_to(ROOT).as_posix()
        return f"{match.group(1)}(/{target})"

    return re.sub(r"(!\[[^\]]*\])\((?!https?:)([^)]+)\)", absolute, text)


def build(render_png: bool = False) -> Path:
    """Build the PDF and return its path."""
    BUILD.mkdir(parents=True, exist_ok=True)
    markdown = prepare_markdown((DOCS / "LEARNING_GUIDE.md").read_text(encoding="utf-8"))

    typst_source = pypandoc.convert_text(
        markdown,
        to="typst",
        format="gfm+tex_math_dollars",  # GitHub-flavoured markdown plus $maths$
        extra_args=[
            "--standalone",
            f"--template={DOCS / 'pdf' / 'template.typ'}",
            f"--lua-filter={DOCS / 'pdf' / 'filter.lua'}",
            "--wrap=none",
            "--columns=1000",  # let Typst choose table column widths
            "-V", f"title={TITLE}",
            "-V", f"date={datetime.date.today():%B %Y}",
        ],
    )
    # On Windows pandoc returns CRLF line endings. Keep only LF, and stop Python
    # from converting them again (a doubled line ending would look like a blank
    # line to Typst, i.e. a paragraph break after every line).
    typst_source = "\n".join(typst_source.splitlines()) + "\n"
    typ_path = BUILD / "LEARNING_GUIDE.typ"
    with open(typ_path, "w", encoding="utf-8", newline="") as f:
        f.write(typst_source)

    pdf_path = DOCS / "LEARNING_GUIDE.pdf"
    typst.compile(str(typ_path), output=str(pdf_path), root=str(ROOT))

    if render_png:
        pages_dir = BUILD / "pages"
        pages_dir.mkdir(exist_ok=True)
        for old in pages_dir.glob("*.png"):
            old.unlink()
        typst.compile(str(typ_path), output=str(pages_dir / "page-{0p}.png"), root=str(ROOT),
                      format="png", ppi=80)
    return pdf_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the learning-guide PDF.")
    parser.add_argument("--png", action="store_true", help="also render each page as a PNG")
    args = parser.parse_args()

    pdf_path = build(render_png=args.png)
    reader = PdfReader(str(pdf_path))
    size_mb = pdf_path.stat().st_size / 1e6
    print(f"Wrote {pdf_path.relative_to(ROOT)}: {len(reader.pages)} pages, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
