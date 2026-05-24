from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "ZIV_VISUAL_DRIFT_ROOT",
        r"C:\ziv_drift_workspace\visual_drift-main\visual_drift-main",
    )
)
SESSION_PATH = SOURCE_ROOT / "data" / "neuropixels" / "session_831882777.mat"
COLORMAP_PATH = SOURCE_ROOT / "data" / "colormaps" / "newmap3.mat"

FIGURE2B_MATRIX_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2b_exact_mouse53_pm_blockA_matrix.csv"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "figure2c_exact_mouse53_pm_blockA.png"
RAW_POINTS_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2c_exact_mouse53_pm_blockA_raw_points.csv"
LAG_MEANS_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2c_exact_mouse53_pm_blockA_lag_means.csv"

NAT_MOVIE_INDEX = 0
AREA_INDEX = 3
FRAMES_PER_REPEAT = 900
N_REPEATS = 30
N_BLOCKS = 2
N_TIME_BINS = 30
FRAMES_PER_BIN = 30
ACTIVITY_SHAPE = (72, 27000, 2)
POPULATION_SHAPE = (72, 30, 60)
CURRENT_MOUSE_SHAPE = (72, 30, 30)
MATRIX_SHAPE = (30, 30)
RAW_POINT_COUNT = sum(range(1, N_REPEATS))


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def load_or_recompute_matrix() -> np.ndarray:
    if FIGURE2B_MATRIX_PATH.is_file():
        matrix = np.loadtxt(FIGURE2B_MATRIX_PATH, delimiter=",")
        if is_valid_source_matrix(matrix):
            return matrix
        LOGGER.warning("Existing Figure 2B matrix failed validation; recomputing from MAT file.")

    matrix = recompute_matrix_from_mat()
    FIGURE2B_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(FIGURE2B_MATRIX_PATH, matrix, delimiter=",", fmt="%.17g")
    return matrix


def is_valid_source_matrix(matrix: np.ndarray) -> bool:
    if matrix.shape != MATRIX_SHAPE:
        return False
    if not np.allclose(matrix, matrix.T, atol=1e-12, equal_nan=True):
        return False
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        return False
    return True


def recompute_matrix_from_mat() -> np.ndarray:
    mat_data = loadmat(
        str(SESSION_PATH),
        appendmat=False,
        simplify_cells=False,
        squeeze_me=False,
        struct_as_record=False,
        verify_compressed_data_integrity=True,
    )
    activity = unwrap_scalar_object(mat_data["informative_rater_mat"][NAT_MOVIE_INDEX, AREA_INDEX])
    if not isinstance(activity, np.ndarray):
        raise TypeError(f"extracted activity should be ndarray, got {type(activity).__name__}")
    if activity.shape != ACTIVITY_SHAPE:
        raise ValueError(f"Activity shape mismatch: {activity.shape}")

    population_vectors = build_population_vectors_explicit(activity)
    current_mouse = population_vectors[:, :, :N_REPEATS]
    if current_mouse.shape != CURRENT_MOUSE_SHAPE:
        raise ValueError(f"current_mouse shape mismatch: {current_mouse.shape}")
    return compute_figure2b_matrix(current_mouse)


def unwrap_scalar_object(value: object) -> object:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def build_population_vectors_explicit(activity: np.ndarray) -> np.ndarray:
    population_vectors = np.empty(POPULATION_SHAPE, dtype=np.float64)
    sub_index = 0
    for block_index in range(N_BLOCKS):
        for repeat_index in range(N_REPEATS):
            repeat_start = repeat_index * FRAMES_PER_REPEAT
            repeat_stop = repeat_start + FRAMES_PER_REPEAT
            current_repeat = activity[:, repeat_start:repeat_stop, block_index]
            for bin_index in range(N_TIME_BINS):
                bin_start = bin_index * FRAMES_PER_BIN
                bin_stop = bin_start + FRAMES_PER_BIN
                population_vectors[:, bin_index, sub_index] = np.nanmean(
                    current_repeat[:, bin_start:bin_stop],
                    axis=1,
                )
            sub_index += 1
    return population_vectors


def compute_figure2b_matrix(current_mouse: np.ndarray) -> np.ndarray:
    matrix = np.empty(MATRIX_SHAPE, dtype=np.float64)
    for repeat1 in range(N_REPEATS):
        for repeat2 in range(N_REPEATS):
            current_pv = pearson_corr_columns(current_mouse[:, :, repeat1], current_mouse[:, :, repeat2])
            matrix[repeat1, repeat2] = np.nanmean(np.diag(current_pv))
    return matrix


def pearson_corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    numerator = x_centered.T @ y_centered
    x_norm = np.sqrt(np.sum(x_centered * x_centered, axis=0))
    y_norm = np.sqrt(np.sum(y_centered * y_centered, axis=0))
    denominator = np.outer(x_norm, y_norm)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def extract_raw_points_and_lag_means(matrix: np.ndarray) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    raw_rows: list[dict[str, float | int]] = []
    lag_rows: list[dict[str, float | int]] = []

    for lag in range(1, N_REPEATS):
        values = np.diagonal(matrix, offset=lag).astype(np.float64, copy=False)
        finite_values = values[np.isfinite(values)]
        for offset_index, value in enumerate(values):
            repeat1 = offset_index + 1
            repeat2 = repeat1 + lag
            raw_rows.append(
                {
                    "lag": lag,
                    "repeat1": repeat1,
                    "repeat2": repeat2,
                    "pv_correlation": float(value),
                    "source_diagonal_offset": lag,
                }
            )

        sem = float("nan")
        if finite_values.size > 1:
            sem = float(np.std(finite_values, ddof=1) / np.sqrt(finite_values.size))
        lag_rows.append(
            {
                "lag": lag,
                "pv_correlation_mean": float(np.nanmean(values)),
                "pv_correlation_sem": sem,
                "n_pairs": int(finite_values.size),
            }
        )

    return raw_rows, lag_rows


def lag_mean(matrix: np.ndarray, lag: int) -> float:
    return float(np.nanmean(np.diagonal(matrix, offset=lag)))


def validate_panel_c(matrix: np.ndarray, raw_rows: list[dict[str, float | int]], lag_rows: list[dict[str, float | int]]) -> None:
    if matrix.shape != MATRIX_SHAPE:
        raise AssertionError(f"Matrix shape mismatch: {matrix.shape}")
    if len(raw_rows) != RAW_POINT_COUNT:
        raise AssertionError(f"Raw point count mismatch: {len(raw_rows)}")
    if len(lag_rows) != N_REPEATS - 1:
        raise AssertionError(f"Lag row count mismatch: {len(lag_rows)}")

    pair_counts = list(range(29, 0, -1))
    observed_n_pairs = [int(row["n_pairs"]) for row in lag_rows]
    if observed_n_pairs != pair_counts:
        raise AssertionError(f"n_pairs mismatch: {observed_n_pairs}")

    for row in raw_rows:
        lag = int(row["lag"])
        repeat1 = int(row["repeat1"])
        repeat2 = int(row["repeat2"])
        if lag == 0:
            raise AssertionError("Lag 0 found in raw rows")
        if repeat1 == repeat2:
            raise AssertionError("Self-pair found in raw rows")
        if repeat2 - repeat1 != lag:
            raise AssertionError(f"Invalid pair row: repeat2-repeat1 != lag for {row}")

    lag1 = float(lag_rows[0]["pv_correlation_mean"])
    lag29 = float(lag_rows[-1]["pv_correlation_mean"])
    if lag1 <= lag29:
        raise AssertionError("Lag 1 mean is not greater than lag 29 mean")


def write_csv(path: Path, rows: list[dict[str, float | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_panel_c_figure(raw_rows: list[dict[str, float | int]], lag_rows: list[dict[str, float | int]]) -> None:
    colormap_data = loadmat(str(COLORMAP_PATH), appendmat=False)
    if "newmap3" not in colormap_data:
        raise KeyError(f"Missing newmap3 variable in {COLORMAP_PATH}")
    newmap3 = colormap_data["newmap3"]
    if not isinstance(newmap3, np.ndarray) or newmap3.ndim != 2 or newmap3.shape[1] != 3:
        raise ValueError(f"newmap3 shape mismatch: {getattr(newmap3, 'shape', None)}")

    scatter_x = [int(row["lag"]) for row in raw_rows]
    scatter_y = [float(row["pv_correlation"]) for row in raw_rows]
    lag_x = [int(row["lag"]) for row in lag_rows]
    lag_y = [float(row["pv_correlation_mean"]) for row in lag_rows]

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.scatter(scatter_x, scatter_y, s=25, color=(0.7, 0.7, 0.7), edgecolors="none", alpha=0.75)
    ax.plot(lag_x, lag_y, color=newmap3[0, :], linewidth=4)
    ax.text(0.05, 0.075, "1 repeat = 30 seconds", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Elapsed time (# of movie repeats)")
    ax.set_ylabel("PV correlation")
    ax.set_title("Single animal example")
    ax.set_xticks([1, 10, 20, 29])
    ax.set_yticks([0.72, 0.76, 0.80, 0.84, 0.88, 0.92])
    ax.set_xlim(0.5, 29.5)
    ax.set_ylim(0.72, 0.92)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_logging()
    matrix = load_or_recompute_matrix()
    raw_rows, lag_rows = extract_raw_points_and_lag_means(matrix)
    validate_panel_c(matrix, raw_rows, lag_rows)

    write_csv(
        RAW_POINTS_CSV_PATH,
        raw_rows,
        ["lag", "repeat1", "repeat2", "pv_correlation", "source_diagonal_offset"],
    )
    write_csv(
        LAG_MEANS_CSV_PATH,
        lag_rows,
        ["lag", "pv_correlation_mean", "pv_correlation_sem", "n_pairs"],
    )
    save_panel_c_figure(raw_rows, lag_rows)

    LOGGER.info("figure=%s", FIGURE_PATH)
    LOGGER.info("raw_points=%s", RAW_POINTS_CSV_PATH)
    LOGGER.info("lag_means=%s", LAG_MEANS_CSV_PATH)


if __name__ == "__main__":
    main()
