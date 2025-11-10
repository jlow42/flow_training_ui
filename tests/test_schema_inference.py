from pathlib import Path

import pytest

from backend import SchemaAnalyzer


@pytest.fixture(scope="module")
def sample_files() -> list[Path]:
    data_dir = Path(__file__).resolve().parents[1] / "sample_data"
    return sorted(data_dir.glob("demo_flow_*.csv"))


def test_schema_inference_identifies_numeric_columns(sample_files: list[Path]) -> None:
    analyzer = SchemaAnalyzer()
    report = analyzer.generate_report(sample_files[:1])
    assert report.summary["total_files"] == 1

    file_report = report.files[0]
    numeric_columns = {
        column.name: column
        for column in file_report.columns
        if column.inferred_type == "numeric"
    }
    assert "FSC_A" in numeric_columns
    stats = numeric_columns["FSC_A"].stats
    assert stats["mean"] is not None
    assert stats["max"] is not None
    assert stats["min"] is not None


def test_schema_inference_detects_categorical_anomalies(tmp_path) -> None:
    csv_path = tmp_path / "high_cardinality.csv"
    values = [f"value_{idx}" for idx in range(10)]
    csv_path.write_text("category\n" + "\n".join(values))

    analyzer = SchemaAnalyzer(high_cardinality_threshold=0.5)
    report = analyzer.generate_report([csv_path])

    file_report = report.files[0]
    column = file_report.columns[0]
    assert column.inferred_type == "categorical"
    assert "high_cardinality" in column.anomalies
