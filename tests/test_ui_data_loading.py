from Applications.schema_dashboard import SchemaDashboard


def test_dashboard_loads_summary_from_fetcher() -> None:
    payload = {
        "summary": {
            "total_files": 2,
            "columns": {
                "FSC_A": {"coverage_pct": 100.0, "anomalies": []},
                "CellType": {"coverage_pct": 100.0, "anomalies": ["high_cardinality"]},
            },
            "warnings": ["Column 'SSC_A' is only present in 50.0% of files"],
        },
        "files": [
            {"path": "demo_flow_1.csv", "row_count": 100, "warnings": ["high_missing_rate"]},
            {"path": "demo_flow_2.csv", "row_count": 80, "warnings": []},
        ],
    }

    def fake_fetch(url: str):
        return payload

    dashboard = SchemaDashboard(fetcher=fake_fetch)
    dashboard.load()

    summary_text = dashboard.render_summary()
    assert "Total files analysed: 2" in summary_text
    assert "CellType" in summary_text
    assert "Warnings" in summary_text

    table = dashboard.render_file_table()
    assert table[0]["warnings"] == "high_missing_rate"
    assert table[1]["row_count"] == 80
