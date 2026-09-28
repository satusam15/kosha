"""python-docx renderer for the ApprovalNote schema."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

from backend.deliverables.schemas import ApprovalNote


def render_approval_note(note: ApprovalNote, out_path: Path | str) -> str:
    """Render a validated ApprovalNote to a .docx file.

    Args:
        note: a validated ApprovalNote schema instance.
        out_path: where to write the file (e.g. ".../approval_note.docx").

    Returns:
        The absolute path of the written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Title
    h = doc.add_heading(note.title, level=0)

    # Meta line
    meta = doc.add_paragraph()
    meta_run = meta.add_run(
        f"Reference: {note.reference or '—'}   |   Date: {note.date or '—'}"
    )
    meta_run.italic = True
    meta_run.font.size = Pt(9)

    # Summary
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(note.summary or "(no summary provided)")

    # Preparation
    doc.add_heading("Prepared By", level=1)
    doc.add_paragraph(note.prepared_by)

    # Approval block
    doc.add_heading("Approval", level=1)
    if note.approval_required:
        doc.add_paragraph(f"Approver: {note.approver_name or 'TBD'}")
        if note.signature_required:
            doc.add_paragraph("Signature: ______________________")
    else:
        doc.add_paragraph("No approval required for this document.")

    doc.save(str(out_path))
    return str(out_path.resolve())
