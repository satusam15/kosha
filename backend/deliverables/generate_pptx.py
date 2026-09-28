"""python-pptx renderer for the Presentation schema.

Supports three slide layouts: title, bullets, table.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation as PptxPresentation
from pptx.util import Inches, Pt

from backend.deliverables.schemas import Presentation, Slide


def _add_title_slide(prs, slide: Slide) -> None:
    layout = prs.slide_layouts[0]  # Title layout
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = slide.title
    for shape in s.shapes:
        if shape.has_text_frame and shape is not s.shapes.title:
            tf = shape.text_frame
            tf.text = ""
            if slide.bullets:
                tf.text = slide.bullets[0]
                for b in slide.bullets[1:]:
                    p = tf.add_paragraph()
                    p.text = b


def _add_bullets_slide(prs, slide: Slide) -> None:
    layout = prs.slide_layouts[1]  # Title + Content
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = slide.title
    body = s.placeholders[1]
    tf = body.text_frame
    tf.text = slide.bullets[0] if slide.bullets else ""
    for b in slide.bullets[1:]:
        p = tf.add_paragraph()
        p.text = b


def _add_table_slide(prs, slide: Slide) -> None:
    layout = prs.slide_layouts[5]  # Title Only
    s = prs.slides.add_slide(layout)
    s.shapes.title.text = slide.title

    rows = slide.table or []
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    nrows = len(rows)

    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(0.4 * nrows)
    table_shape = s.shapes.add_table(nrows, ncols, left, top, width, height)
    tbl = table_shape.table

    for i, row in enumerate(rows):
        for j in range(ncols):
            val = row[j] if j < len(row) else ""
            tbl.cell(i, j).text = str(val)


_RENDERERS = {
    "title": _add_title_slide,
    "bullets": _add_bullets_slide,
    "table": _add_table_slide,
}


def render_presentation(pres: Presentation, out_path: Path | str) -> str:
    """Render a validated Presentation to a .pptx file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prs = PptxPresentation()

    # Cover slide
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = pres.title
    if pres.subtitle:
        sub = cover.placeholders[1]
        sub.text = pres.subtitle

    for slide in pres.slides:
        _RENDERERS.get(slide.layout, _add_bullets_slide)(prs, slide)

    prs.save(str(out_path))
    return str(out_path.resolve())
