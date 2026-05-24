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
NEUROPIXELS_DIR = SOURCE_ROOT / "data" / "neuropixels"

NAT_MOVIE_INDEX = 0
AREA_INDEX = 3
AREA_NAME = "VISpm / PM"
FRAME_RATE_HZ = 30
MOVIE_SECONDS = 30
FRAMES_PER_REPEAT = FRAME_RATE_HZ * MOVIE_SECONDS
N_TIME_BINS = 30
FRAMES_PER_BIN = FRAMES_PER_REPEAT // N_TIME_BINS
N_BLOCKS = 2
N_REPEATS = 30
CELL_CUTOFF = 15

FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "figure2e_pm_only_pv_across_mice.png"
VALID_MICE_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2e_pm_only_valid_mice.csv"
MOUSE_BY_LAG_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2e_pm_only_mouse_by_lag.csv"
SUMMARY_BY_LAG_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2e_pm_only_summary_by_lag.csv"

PM_FILL_COLOR = np.array([0.9, 0.6, 0.2])
PM_LINE_COLOR = np.array([0.7, 0.4, 0.0])


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def unwrap_scalar_object(value: object) -> object:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def load_session(path: Path) -> dict:
    mat_data = loadmat(
        str(path),
        appendmat=False,
        simplify_cells=False,
        squeeze_me=False,
        struct_as_record=False,
        verify_compressed_data_integrity=True,
    )
    required = ("informative_rater_mat", "mean_running_speed_repeats", "cell_num")
    missing = [key for key in required if key not in mat_data]
    if missing:
        raise KeyError(f"{path.name} is missing MAT variables: {missing}")
    return {key: value for key, value in mat_data.items() if not key.startswith("__")}


def natural_movie1_repeats(mat_data: dict) -> int:
    repeats = unwrap_scalar_object(mat_data["mean_running_speed_repeats"][0, NAT_MOVIE_INDEX])
    if not isinstance(repeats, np.ndarray) or repeats.ndim < 1:
        raise ValueError("mean_running_speed_repeats[0, 0] does not contain an array")
    return int(repeats.shape[0])


def pm_cell_count(mat_data: dict) -> int:
    cell_num = np.asarray(mat_data["cell_num"])
    if cell_num.ndim != 2 or cell_num.shape[1] <= AREA_INDEX:
        raise ValueError(f"cell_num shape mismatch: {cell_num.shape}")
    return int(cell_num[0, AREA_INDEX])


def extract_pm_movie1_activity(mat_data: dict) -> np.ndarray | None:
    informative = mat_data["informative_rater_mat"]
    if not isinstance(informative, np.ndarray):
        raise TypeError(f"informative_rater_mat should be ndarray, got {type(informative).__name__}")
    activity = unwrap_scalar_object(informative[NAT_MOVIE_INDEX, AREA_INDEX])
    if not isinstance(activity, np.ndarray) or activity.size == 0:
        return None
    if activity.ndim != 3:
        raise ValueError(f"Activity should be 3D, got shape {activity.shape}")
    return activity


def build_population_vectors_explicit(activity: np.ndarray) -> np.ndarray:
    frames_per_block = N_REPEATS * FRAMES_PER_REPEAT
    if activity.shape[1:] != (frames_per_block, N_BLOCKS):
        raise ValueError(f"Activity shape mismatch: {activity.shape}")

    population = np.empty((activity.shape[0], N_TIME_BINS, N_REPEATS * N_BLOCKS), dtype=np.float64)
    sub_index = 0
    for block_index in range(N_BLOCKS):
        for repeat_index in range(N_REPEATS):
            repeat_start = repeat_index * FRAMES_PER_REPEAT
            repeat_stop = repeat_start + FRAMES_PER_REPEAT
            current_repeat = activity[:, repeat_start:repeat_stop, block_index]
            for bin_index in range(N_TIME_BINS):
                bin_start = bin_index * FRAMES_PER_BIN
                bin_stop = bin_start + FRAMES_PER_BIN
                population[:, bin_index, sub_index] = np.nanmean(current_repeat[:, bin_start:bin_stop], axis=1)
            sub_index += 1
    return population


def pearson_corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)
    y_centered = y - np.nanmean(y, axis=0, keepdims=True)
    numerator = x_centered.T @ y_centered
    x_norm = np.sqrt(np.nansum(x_centered * x_centered, axis=0))
    y_norm = np.sqrt(np.nansum(y_centered * y_centered, axis=0))
    denominator = np.outer(x_norm, y_norm)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def repeat_similarity_matrix(block_vectors: np.ndarray) -> np.ndarray:
    if block_vectors.shape[1:] != (N_TIME_BINS, N_REPEATS):
        raise ValueError(f"Block vector shape mismatch: {block_vectors.shape}")

    matrix = np.empty((N_REPEATS, N_REPEATS), dtype=np.float64)
    for repeat1 in range(N_REPEATS):
        for repeat2 in range(N_REPEATS):
            current_pv = pearson_corr_columns(block_vectors[:, :, repeat1], block_vectors[:, :, repeat2])
            matrix[repeat1, repeat2] = np.nanmean(np.diag(current_pv))
    return matrix


def collapse_matrix_by_positive_lag(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (N_REPEATS, N_REPEATS):
        raise ValueError(f"Repeat matrix shape mismatch: {matrix.shape}")
    return np.array([np.nanmean(np.diagonal(matrix, offset=lag)) for lag in range(1, N_REPEATS)], dtype=np.float64)


def compute_session_lag_curves(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    population = build_population_vectors_explicit(activity)
    block_a = population[:, :, :N_REPEATS]
    block_b = population[:, :, N_REPEATS:]
    matrix_a = repeat_similarity_matrix(block_a)
    matrix_b = repeat_similarity_matrix(block_b)
    lag_a = collapse_matrix_by_positive_lag(matrix_a)
    lag_b = collapse_matrix_by_positive_lag(matrix_b)
    lag_mean = np.nanmean(np.vstack([lag_a, lag_b]), axis=0)
    return lag_a, lag_b, lag_mean


def sem(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return float("nan")
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def collect_pm_valid_mice() -> tuple[list[dict], list[dict]]:
    valid_rows: list[dict] = []
    mouse_lag_rows: list[dict] = []

    mat_files = sorted(NEUROPIXELS_DIR.glob("*.mat"))
    for python_index, path in enumerate(mat_files):
        mat_data = load_session(path)
        repeats = natural_movie1_repeats(mat_data)
        cell_count = pm_cell_count(mat_data)
        activity = extract_pm_movie1_activity(mat_data)

        reason = ""
        if repeats != N_REPEATS:
            reason = f"movie1_repeats={repeats}"
        elif cell_count < CELL_CUTOFF:
            reason = f"pm_cell_count={cell_count}"
        elif activity is None:
            reason = "missing_pm_activity"
        elif activity.shape != (cell_count, N_REPEATS * FRAMES_PER_REPEAT, N_BLOCKS):
            reason = f"activity_shape={activity.shape}"

        if reason:
            continue

        lag_a, lag_b, lag_mean = compute_session_lag_curves(activity)
        valid_rows.append(
            {
                "session_order_index": python_index + 1,
                "python_index": python_index,
                "session_file": path.name,
                "session_id": path.stem.replace("session_", ""),
                "area": AREA_NAME,
                "movie1_repeats_per_block": repeats,
                "pm_cell_count": cell_count,
                "activity_shape": str(tuple(activity.shape)),
            }
        )
        for lag_index, lag in enumerate(range(1, N_REPEATS)):
            mouse_lag_rows.append(
                {
                    "session_order_index": python_index + 1,
                    "python_index": python_index,
                    "session_file": path.name,
                    "session_id": path.stem.replace("session_", ""),
                    "lag": lag,
                    "block_a_pv_correlation": lag_a[lag_index],
                    "block_b_pv_correlation": lag_b[lag_index],
                    "mean_block_pv_correlation": lag_mean[lag_index],
                }
        )
        LOGGER.info("valid_session=%s pm_units=%s", path.name, cell_count)

    return valid_rows, mouse_lag_rows


def make_summary(mouse_lag_rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    for lag in range(1, N_REPEATS):
        values = np.array(
            [float(row["mean_block_pv_correlation"]) for row in mouse_lag_rows if int(row["lag"]) == lag],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "lag": lag,
                "n_mice": int(finite.size),
                "pv_correlation_mean": float(np.nanmean(finite)),
                "pv_correlation_sem": sem(finite),
                "pv_correlation_std": float(np.nanstd(finite, ddof=1)) if finite.size > 1 else float("nan"),
            }
        )
    return summary_rows


def validate_results(valid_rows: list[dict], mouse_lag_rows: list[dict], summary_rows: list[dict]) -> None:
    session_files = {row["session_file"] for row in valid_rows}
    if not session_files:
        raise AssertionError("No valid PM sessions found")
    if len(summary_rows) != N_REPEATS - 1:
        raise AssertionError(f"Summary lag count mismatch: {len(summary_rows)}")

    n_valid_mice = len(valid_rows)
    for row in summary_rows:
        if int(row["n_mice"]) != n_valid_mice:
            raise AssertionError(f"Lag {row['lag']} has n_mice={row['n_mice']}, n_valid_mice={n_valid_mice}")
    lag1 = float(summary_rows[0]["pv_correlation_mean"])
    lag29 = float(summary_rows[-1]["pv_correlation_mean"])
    if lag1 <= lag29:
        raise AssertionError(f"PM lag trend failed: lag1={lag1}, lag29={lag29}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(summary_rows: list[dict]) -> None:
    lags = np.array([int(row["lag"]) for row in summary_rows], dtype=np.int64)
    means = np.array([float(row["pv_correlation_mean"]) for row in summary_rows], dtype=np.float64)
    sems = np.array([float(row["pv_correlation_sem"]) for row in summary_rows], dtype=np.float64)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    ax.fill_between(lags, means - sems, means + sems, color=PM_FILL_COLOR, alpha=0.4, linewidth=0)
    ax.plot(lags, means, color=PM_LINE_COLOR, linewidth=3)
    ax.text(0.05, 0.075, "1 repeat = 30 seconds", transform=ax.transAxes, fontsize=10)
    ax.set_title("Average across animals")
    ax.set_xlabel("Elapsed time (# of movie repeats)")
    ax.set_ylabel("PV correlation")
    ax.set_xlim(1, 29)
    ax.set_xticks([1, 10, 20, 29])
    ax.set_yticks(np.arange(0.66, 0.841, 0.04))
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_logging()
    valid_rows, mouse_lag_rows = collect_pm_valid_mice()
    summary_rows = make_summary(mouse_lag_rows)
    validate_results(valid_rows, mouse_lag_rows, summary_rows)

    write_csv(VALID_MICE_CSV, valid_rows)
    write_csv(MOUSE_BY_LAG_CSV, mouse_lag_rows)
    write_csv(SUMMARY_BY_LAG_CSV, summary_rows)
    save_figure(summary_rows)

    LOGGER.info("valid_mice_count=%s", len(valid_rows))
    LOGGER.info("figure=%s", FIGURE_PATH)
    LOGGER.info("valid_mice_csv=%s", VALID_MICE_CSV)
    LOGGER.info("mouse_by_lag_csv=%s", MOUSE_BY_LAG_CSV)
    LOGGER.info("summary_by_lag_csv=%s", SUMMARY_BY_LAG_CSV)


if __name__ == "__main__":
    main()
