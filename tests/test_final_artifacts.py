from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_FIGURES = [
    Path("outputs/figures/figure2b_exact_mouse53_pm_blockA.png"),
    Path("outputs/figures/figure2c_exact_mouse53_pm_blockA.png"),
    Path("outputs/figures/figure2e_pm_only_pv_across_mice.png"),
    Path("outputs/figures/figure2h_pm_only_ensemble_rate_across_mice.png"),
]

FINAL_TABLES = [
    Path("outputs/tables/figure2b_exact_mouse53_pm_blockA_matrix.csv"),
    Path("outputs/tables/figure2b_exact_mouse53_pm_blockA_plot_matrix.csv"),
    Path("outputs/tables/figure2c_exact_mouse53_pm_blockA_raw_points.csv"),
    Path("outputs/tables/figure2c_exact_mouse53_pm_blockA_lag_means.csv"),
    Path("outputs/tables/figure2e_pm_only_valid_mice.csv"),
    Path("outputs/tables/figure2e_pm_only_mouse_by_lag.csv"),
    Path("outputs/tables/figure2e_pm_only_summary_by_lag.csv"),
    Path("outputs/tables/figure2h_pm_only_valid_mice.csv"),
    Path("outputs/tables/figure2h_pm_only_mouse_by_lag.csv"),
    Path("outputs/tables/figure2h_pm_only_summary_by_lag.csv"),
]

LEGACY_OUTPUTS = [
    Path("outputs/figures/figure2bc_single_mouse_pm_block1.png"),
    Path("outputs/figures/figure2bc_original_mouse53_pm_block1.png"),
    Path("outputs/figures/figure2bc_control_session847657808_pm_block1.png"),
    Path("outputs/tables/figure2c_control_session847657808_pm_block1_lag_curve.csv"),
    Path("outputs/tables/figure2c_original_mouse53_pm_block1_lag_curve.csv"),
    Path("outputs/tables/figure2c_single_mouse_pm_block1_lag_curve.csv"),
    Path("outputs/tables/figure2c_single_mouse_pm_lag_curve.csv"),
]


def read_csv_rows(relative_path: Path) -> list[dict[str, str]]:
    with (PROJECT_ROOT / relative_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_nonempty_file(relative_path: Path) -> None:
    path = PROJECT_ROOT / relative_path
    assert path.exists(), f"Missing file: {relative_path}"
    assert path.stat().st_size > 0, f"Empty file: {relative_path}"


def test_final_figures_exist_and_are_nonempty() -> None:
    for figure in FINAL_FIGURES:
        assert_nonempty_file(figure)


def test_final_tables_exist_and_are_nonempty() -> None:
    for table in FINAL_TABLES:
        assert_nonempty_file(table)


def test_figure2c_raw_points_and_lag_means() -> None:
    raw_rows = read_csv_rows(Path("outputs/tables/figure2c_exact_mouse53_pm_blockA_raw_points.csv"))
    assert len(raw_rows) == 435
    assert {"lag", "repeat1", "repeat2", "pv_correlation"} <= set(raw_rows[0])
    raw_lags = {int(row["lag"]) for row in raw_rows}
    assert raw_lags == set(range(1, 30))
    for row in raw_rows:
        lag = int(row["lag"])
        assert lag != 0
        assert int(row["repeat2"]) - int(row["repeat1"]) == lag

    lag_rows = read_csv_rows(Path("outputs/tables/figure2c_exact_mouse53_pm_blockA_lag_means.csv"))
    assert len(lag_rows) == 29
    assert [int(row["lag"]) for row in lag_rows] == list(range(1, 30))
    assert [int(row["n_pairs"]) for row in lag_rows] == list(range(29, 0, -1))


def test_figure2e_and_2h_summaries() -> None:
    summary_specs = [
        (Path("outputs/tables/figure2e_pm_only_summary_by_lag.csv"), "pv_correlation_mean"),
        (Path("outputs/tables/figure2h_pm_only_summary_by_lag.csv"), "ensemble_rate_correlation_mean"),
    ]
    for table, mean_column in summary_specs:
        rows = read_csv_rows(table)
        assert len(rows) == 29
        assert [int(row["lag"]) for row in rows] == list(range(1, 30))
        assert "n_mice" in rows[0]
        assert all(int(row["n_mice"]) > 0 for row in rows)
        by_lag = {int(row["lag"]): row for row in rows}
        assert float(by_lag[1][mean_column]) > float(by_lag[29][mean_column])


def test_valid_mice_match_between_figure2e_and_figure2h() -> None:
    figure2e_rows = read_csv_rows(Path("outputs/tables/figure2e_pm_only_valid_mice.csv"))
    figure2h_rows = read_csv_rows(Path("outputs/tables/figure2h_pm_only_valid_mice.csv"))
    figure2e_sessions = {row["session_id"] for row in figure2e_rows}
    figure2h_sessions = {row["session_id"] for row in figure2h_rows if row.get("included", "True") == "True"}
    assert figure2e_sessions
    assert figure2h_sessions
    assert figure2e_sessions == figure2h_sessions


def test_legacy_outputs_are_not_in_final_output_folders() -> None:
    for output in LEGACY_OUTPUTS:
        assert not (PROJECT_ROOT / output).exists(), f"Legacy output still present: {output}"


def test_main_py_check_mode() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--check"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = result.stdout + result.stderr
    assert result.returncode == 0
    assert "all_final_artifacts_present=True" in combined_output
