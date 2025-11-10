"""FastAPI application exposing schema reports produced by :mod:`backend.reporting`."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from backend import SchemaAnalyzer, SchemaReport

DATA_ROOT = Path(__file__).parent / "sample_data"

app = FastAPI(title="Schema Reporting Service")
_analyzer = SchemaAnalyzer()
_cached_report: Optional[SchemaReport] = None


def _discover_files() -> List[Path]:
    return sorted(DATA_ROOT.glob("*.csv"))


def _refresh_report() -> SchemaReport:
    files = _discover_files()
    if not files:
        raise HTTPException(status_code=404, detail="No CSV files available for schema inference")
    return _analyzer.generate_report(files)


@app.on_event("startup")
def _load_initial_report() -> None:
    global _cached_report
    try:
        _cached_report = _refresh_report()
    except HTTPException:
        _cached_report = None


@app.get("/reports/schema")
def get_schema_report() -> dict:
    if _cached_report is None:
        raise HTTPException(status_code=404, detail="Schema report has not been generated")
    return _cached_report.to_dict()


@app.post("/reports/schema/refresh")
def refresh_schema_report() -> dict:
    global _cached_report
    _cached_report = _refresh_report()
    return _cached_report.to_dict()
