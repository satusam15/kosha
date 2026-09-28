"""Strict pydantic schemas for deliverables.

Generation output is validated against these models BEFORE rendering, so a
malformed payload can never reach the docx/xlsx/pptx renderers.

All models use `extra="forbid"` and `strict=True` so unknown fields and
loose types are rejected loudly rather than silently coerced.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# ---------------------------------------------------------------------------
# Approval Note
# ---------------------------------------------------------------------------
class ApprovalNote(_Strict):
    """A one-page approval / sign-off document."""

    title: str = Field(min_length=1, max_length=200)
    reference: str = Field(default="", max_length=100)       # e.g. doc no.
    prepared_by: str = Field(min_length=1)
    date: str = Field(default="")                             # ISO date string
    summary: str = Field(default="")
    approval_required: bool = True
    approver_name: str = Field(default="")
    signature_required: bool = True


# ---------------------------------------------------------------------------
# Calculation Sheet (spreadsheet with real formulas)
# ---------------------------------------------------------------------------
class CalcRow(_Strict):
    """One row of the calculation sheet. `value` may be numeric or a formula
    string (e.g. "=B2*C2") — renderers write formulas verbatim into cells."""

    item: str = Field(min_length=1)
    qty: float = 0.0
    unit_price: float = 0.0
    formula: Optional[str] = None     # overrides the default product formula
    note: str = ""


class CalcSheet(_Strict):
    """A table of line items plus computed totals (subtotal / tax / total)."""

    title: str = Field(min_length=1, max_length=200)
    currency: str = "USD"
    tax_rate: float = Field(default=0.0, ge=0, le=1.0)      # e.g. 0.10 == 10%
    rows: list[CalcRow] = Field(default_factory=list)

    @field_validator("rows")
    @classmethod
    def _limit_rows(cls, v: list[CalcRow]) -> list[CalcRow]:
        if len(v) > 500:
            raise ValueError("too many rows (max 500)")
        return v


# ---------------------------------------------------------------------------
# Presentation (slides)
# ---------------------------------------------------------------------------
class Slide(_Strict):
    """One slide. `layout` picks the rendering template in the pptx renderer."""

    layout: Literal["title", "bullets", "table"] = "bullets"
    title: str = Field(min_length=1, max_length=200)
    bullets: list[str] = Field(default_factory=list)
    table: Optional[list[list[str]]] = None   # rows x cols of strings


class Presentation(_Strict):
    """A deck composed of slides."""

    title: str = Field(min_length=1, max_length=200)
    subtitle: str = ""
    author: str = ""
    slides: list[Slide] = Field(default_factory=list)

    @field_validator("slides")
    @classmethod
    def _limit_slides(cls, v: list[Slide]) -> list[Slide]:
        if len(v) > 100:
            raise ValueError("too many slides (max 100)")
        return v


# ---------------------------------------------------------------------------
# Kind -> schema map used by the generate_deliverable tool
# ---------------------------------------------------------------------------
SCHEMA_BY_KIND: dict[str, type[_Strict]] = {
    "approval_note": ApprovalNote,
    "calc_sheet": CalcSheet,
    "presentation": Presentation,
}
