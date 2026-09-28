"""openpyxl renderer for the CalcSheet schema, with real formulas.

Every row writes an Amount column as a formula (`=qty*unit_price` unless
overridden), and the footer uses SUM / arithmetic formulas that Excel and
LibreOffice recompute live on open.
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from backend.deliverables.schemas import CalcSheet


def render_calc_sheet(sheet: CalcSheet, out_path: Path | str) -> str:
    """Render a validated CalcSheet to a .xlsx file with live formulas."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "CalcSheet"

    headers = ["Item", "Qty", "Unit Price", "Amount"]
    ws.append(headers)
    for col, _ in enumerate(headers, start=1):
        ws.cell(row=1, column=col).font = Font(bold=True)

    # Data rows: data starts at row 2.
    first_data_row = 2
    for i, row in enumerate(sheet.rows):
        excel_row = first_data_row + i
        ws.cell(row=excel_row, column=1, value=row.item)
        ws.cell(row=excel_row, column=2, value=row.qty)
        ws.cell(row=excel_row, column=3, value=row.unit_price)

        # Amount column: real formula, either custom or default product.
        if row.formula:
            formula = row.formula
        else:
            formula = f"=B{excel_row}*C{excel_row}"
        ws.cell(row=excel_row, column=4, value=formula)

        if row.note:
            ws.cell(row=excel_row, column=5, value=row.note)

    last_data_row = first_data_row + len(sheet.rows) - 1 if sheet.rows else 1

    # --- Totals section (real formulas) ---
    totals_start = last_data_row + 2

    def put(label: str, formula: str, excel_row: int) -> None:
        ws.cell(row=excel_row, column=3, value=label).font = Font(bold=True)
        ws.cell(row=excel_row, column=4, value=formula).font = Font(bold=True)

    sub_total_row = totals_start
    put("Subtotal", f"=SUM(D{first_data_row}:D{last_data_row})", sub_total_row)

    tax_row = sub_total_row + 1
    # Tax = subtotal * rate  (rate is a fraction, e.g. 0.10)
    put("Tax", f"=D{sub_total_row}*{sheet.tax_rate}", tax_row)

    total_row = tax_row + 1
    put("Total", f"=D{sub_total_row}+D{tax_row}", total_row)

    # Currency label
    ws.cell(row=totals_start - 1, column=1,
            value=f"{sheet.title} ({sheet.currency})").font = Font(bold=True, size=12)

    # Column widths
    for col in range(1, 6):
        ws.column_dimensions[get_column_letter(col)].width = 18

    wb.save(str(out_path))
    return str(out_path.resolve())
