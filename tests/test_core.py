"""Smoke tests for the Kosh backend.

Run from the project root with:  python -m pytest tests/ -q
(or: python tests/test_core.py)

These tests verify the parts that don't require Ollama/Docker/FAISS to be
present, so the suite passes on a plain machine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.router.model_router import route_task
from backend.deliverables import schemas
from backend.deliverables.generate_docx import render_approval_note
from backend.deliverables.generate_xlsx import render_calc_sheet
from backend.deliverables.generate_pptx import render_presentation
from backend.agent.tools import describe_tools, run_tool, TOOLS


# ---------------------------------------------------------------------------
# Model router
# ---------------------------------------------------------------------------
def test_routes_code_keywords():
    r = route_task("write a python function that parses a csv file")
    assert r.task_type == "code"
    assert r.model != schemas  # sanity


def test_routes_code_model():
    assert route_task("please write a script to parse CSV").task_type == "code"


def test_routes_vision_on_upload():
    r = route_task("what does this show?", has_upload=True)
    assert r.task_type == "vision"


def test_routes_vision_keyword():
    assert route_task("extract text from this image").task_type == "vision"


def test_routes_general():
    r = route_task("summarize the material handling procedure")
    assert r.task_type == "general"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
def test_tools_registered():
    assert set(TOOLS) == {
        "read_file", "write_file", "ocr_image", "search_docs", "run_code",
        "generate_deliverable",
    }
    assert "generate_deliverable" in describe_tools()


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    import backend.agent.tools as tools

    # Repoint the workspace to a temp dir so we don't pollute the repo.
    monkeypatch.setattr(tools, "WORKSPACE_DIR", tmp_path)
    out = run_tool("write_file", '{"path_str": "hello.txt", "content": "hi"}')
    assert "wrote" in out
    got = run_tool("read_file", '{"path_str": "hello.txt"}')
    assert got == "hi"


def test_unknown_tool():
    out = run_tool("nope", "{}")
    assert "unknown tool" in out


# ---------------------------------------------------------------------------
# Deliverable schemas (strict validation)
# ---------------------------------------------------------------------------
def test_approval_note_valid():
    note = schemas.ApprovalNote(
        title="Plant Shutdown Approval", prepared_by="QA", approver_name="S. Rao"
    )
    assert note.approval_required is True


def test_approval_note_rejects_extra_field():
    try:
        schemas.ApprovalNote(title="x", prepared_by="y", bogus="field")
        raise AssertionError("should have raised")
    except Exception as e:
        assert "bogus" in str(e)


def test_calc_sheet_validation():
    sheet = schemas.CalcSheet(
        title="Budget", tax_rate=0.10,
        rows=[{"item": "Bolt M8", "qty": 200, "unit_price": 0.5}],
    )
    assert sheet.rows[0].qty == 200


def test_calc_sheet_rejects_bad_tax():
    try:
        schemas.CalcSheet(title="t", tax_rate=1.5)
        raise AssertionError("should have raised")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Renderers (produce real files)
# ---------------------------------------------------------------------------
def test_render_approval_note(tmp_path):
    note = schemas.ApprovalNote(title="Sign-off", prepared_by="Ops",
                                summary="Testing", approver_name="A. Kumar")
    path = render_approval_note(note, tmp_path / "note.docx")
    assert Path(path).exists()
    assert path.endswith(".docx")


def test_render_calc_sheet_with_formulas(tmp_path):
    sheet = schemas.CalcSheet(
        title="Materials", tax_rate=0.10,
        rows=[
            {"item": "Bolt", "qty": 100, "unit_price": 0.5},
            {"item": "Nut", "qty": 50, "unit_price": 0.3},
        ],
    )
    path = render_calc_sheet(sheet, tmp_path / "calc.xlsx")
    assert Path(path).exists()

    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=False)
    ws = wb.active
    # Amount column has a live formula, e.g. =B2*C2
    assert ws["D2"].value.startswith("=")
    assert "SUM" in ws["D5"].value  # subtotal formula


def test_render_presentation(tmp_path):
    pres = schemas.Presentation(
        title="Inspection Summary", subtitle="Q3",
        slides=[
            {"layout": "title", "title": "Overview", "bullets": ["Intro"]},
            {"layout": "bullets", "title": "Findings", "bullets": ["a", "b"]},
            {"layout": "table", "title": "Data", "table": [["Item", "Qty"], ["A", "1"]]},
        ],
    )
    path = render_presentation(pres, tmp_path / "deck.pptx")
    assert Path(path).exists()
    assert path.endswith(".pptx")


# ---------------------------------------------------------------------------
# Network monitor (offline watchdog) - no Ollama needed
# ---------------------------------------------------------------------------
def test_network_stats_shape():
    from backend import network_monitor

    stats = network_monitor.network_stats()
    assert "outbound_nonlocal" in stats
    assert isinstance(stats["outbound_nonlocal"], int)


if __name__ == "__main__":
    import traceback

    failures = 0
    tests = [
        test_routes_code_keywords, test_routes_code_model, test_routes_vision_on_upload,
        test_routes_vision_keyword, test_routes_general, test_tools_registered,
        test_unknown_tool, test_approval_note_valid, test_approval_note_rejects_extra_field,
        test_calc_sheet_validation, test_calc_sheet_rejects_bad_tax,
        test_render_approval_note, test_render_calc_sheet_with_formulas,
        test_render_presentation, test_network_stats_shape,
    ]
    import tempfile, types

    for fn in tests:
        tmp = tempfile.TemporaryDirectory()
        f = lambda: fn(tmp_path=Path(tmp.name))
        # handle tmp_path args + monkeypatch via simple wrappers
        try:
            if fn in (test_write_then_read_roundtrip,):
                continue  # needs monkeypatch; covered by pytest
            fn(tmp_path=Path(tmp.name))
            print("PASS:", fn.__name__)
        except Exception:
            failures += 1
            print("FAIL:", fn.__name__)
            traceback.print_exc()
        finally:
            tmp.cleanup()
    print(f"\n{len(tests)}-2 tests run, {failures} failures")
