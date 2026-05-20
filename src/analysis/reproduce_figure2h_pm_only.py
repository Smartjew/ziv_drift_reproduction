from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\ziv_drift_workspace\ziv_drift_reproduction")
SOURCE_ROOT = Path(r"C:\ziv_drift_workspace\visual_drift-main\visual_drift-main")
NEUROPIXELS_DIR = SOURCE_ROOT / "data" / "neuropixels"

FIGURE2E_VALID_MICE_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2e_pm_only_valid_mice.csv"

NAT_MOVIE_INDEX = 0
AREA_INDEX = 3
FRAME_RATE_HZ = 30
MOVIE_SECONDS = 30
FRAMES_PER_REPEAT = FRAME_RATE_HZ * MOVIE_SECONDS
N_REPEATS = 30
N_TIME_BINS = 30
FRAMES_PER_BIN = FRAMES_PER_REPEAT // N_TIME_BINS
N_BLOCKS = 2
CELL_CUTOFF = 15

FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "figure2h_pm_only_ensemble_rate_across_mice.png"
VALID_MICE_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2h_pm_only_valid_mice.csv"
MOUSE_BY_LAG_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2h_pm_only_mouse_by_lag.csv"
SUMMARY_BY_LAG_CSV = PROJECT_ROOT / "outputs" / "tables" / "figure2h_pm_only_summary_by_lag.csv"

PM_FILL_COLOR = np.array([0.9, 0.6, 0.2])
PM_LINE_COLOR = np.array([0.7, 0.4, 0.0])


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def unwrap_scalar_object(value: object) -> object:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        raise KeyError(f"{path.name} missing MAT variables: {missing}")
    return {key: value for key, value in mat_data.items() if not key.startswith("__")}


def extract_pm_activity(mat_data: dict) -> np.ndarray | None:
    informative = mat_data["informative_rater_mat"]
    if not isinstance(informative, np.ndarray):
        raise TypeError(f"informative_rater_mat should be ndarray, got {type(informative).__name__}")
    activity = unwrap_scalar_object(informative[NAT_MOVIE_INDEX, AREA_INDEX])
    if not isinstance(activity, np.ndarray) or activity.size == 0:
        return None
    if activity.ndim != 3:
        raise ValueError(f"Activity should be 3D, got shape {activity.shape}")
    return activity


def movie1_repeat_count(mat_data: dict) -> int:
    repeats = unwrap_scalar_object(mat_data["mean_running_speed_repeats"][0, NAT_MOVIE_INDEX])
    if not isinstance(repeats, np.ndarray) or repeats.ndim < 1:
        raise ValueError("mean_running_speed_repeats[0, 0] does not contain an array")
    return int(repeats.shape[0])


def pm_unit_count(mat_data: dict) -> int:
    cell_num = np.asarray(mat_data["cell_num"])
    if cell_num.ndim != 2 or cell_num.shape[1] <= AREA_INDEX:
        raise ValueError(f"cell_num shape mismatch: {cell_num.shape}")
    return int(cell_num[0, AREA_INDEX])


def valid_activity_shape(activity: np.ndarray, n_units: int) -> bool:
    return activity.shape == (n_units, N_REPEATS * FRAMES_PER_REPEAT, N_BLOCKS)


def load_valid_mice() -> list[dict]:
    if FIGURE2E_VALID_MICE_CSV.exists():
        source_rows = read_csv(FIGURE2E_VALID_MICE_CSV)
        valid_rows = []
        for row in source_rows:
            filename = row["session_file"]
            n_units = int(row["pm_cell_count"])
            valid_rows.append(
                {
                    "filename": filename,
                    "session_id": row["session_id"],
                    "n_units_pm": n_units,
                    "activity_shape": row["activity_shape"],
                    "included": True,
                    "exclusion_reason": "",
                }
            )
        return valid_rows

    rows = []
    mat_files = sorted(NEUROPIXELS_DIR.glob("*.mat"))
    for path in mat_files:
        mat_data = load_session(path)
        repeats = movie1_repeat_count(mat_data)
        n_units = pm_unit_count(mat_data)
        activity = extract_pm_activity(mat_data)
        activity_shape = "" if activity is None else str(tuple(activity.shape))
        included = False
        reason = ""

        if activity is None:
            reason = "missing_pm_activity"
        elif repeats != N_REPEATS:
            reason = f"movie1_repeats={repeats}"
        elif n_units < CELL_CUTOFF:
            reason = f"n_units_pm={n_units}"
        elif not valid_activity_shape(activity, n_units):
            reason = f"activity_shape={activity_shape}"
        else:
            included = True

        rows.append(
            {
                "filename": path.name,
                "session_id": path.stem.replace("session_", ""),
                "n_units_pm": n_units,
                "activity_shape": activity_shape,
                "included": included,
                "exclusion_reason": reason,
            }
        )

    return [row for row in rows if row["included"] is True]


def build_binned_blocks(activity: np.ndarray) -> np.ndarray:
    n_units = activity.shape[0]
    binned = np.empty((N_BLOCKS, n_units, N_TIME_BINS, N_REPEATS), dtype=np.float64)
    for block_index in range(N_BLOCKS):
        for repeat_index in range(N_REPEATS):
            repeat_start = repeat_index * FRAMES_PER_REPEAT
            repeat_stop = repeat_start + FRAMES_PER_REPEAT
            current_repeat = activity[:, repeat_start:repeat_stop, block_index]
            for bin_index in range(N_TIME_BINS):
                bin_start = bin_index * FRAMES_PER_BIN
                bin_stop = bin_start + FRAMES_PER_BIN
                binned[block_index, :, bin_index, repeat_index] = np.nanmean(
                    current_repeat[:, bin_start:bin_stop],
                    axis=1,
                )
    return binned


def pearson_corr_columns(x: np.ndarray) -> np.ndarray:
    centered = x - np.nanmean(x, axis=0, keepdims=True)
    numerator = centered.T @ centered
    norms = np.sqrt(np.nansum(centered * centered, axis=0))
    denominator = np.outer(norms, norms)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def collapse_rate_matrix_by_lag(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape != (N_REPEATS, N_REPEATS):
        raise ValueError(f"Ensemble-rate matrix shape mismatch: {matrix.shape}")
    lag_means = np.empty(N_REPEATS - 1, dtype=np.float64)
    n_pairs = np.empty(N_REPEATS - 1, dtype=np.int64)
    for lag in range(1, N_REPEATS):
        values = np.diagonal(matrix, offset=lag)
        finite = values[np.isfinite(values)]
        lag_means[lag - 1] = np.nanmean(finite)
        n_pairs[lag - 1] = finite.size
    return lag_means, n_pairs


def compute_mouse_ensemble_rate(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binned = build_binned_blocks(activity)
    block_lag_means = []
    block_n_pairs = []
    for block_index in range(N_BLOCKS):
        ensemble_rate_by_repeat = np.nanmean(binned[block_index], axis=1)
        if ensemble_rate_by_repeat.shape != (activity.shape[0], N_REPEATS):
            raise ValueError(f"Ensemble-rate shape mismatch: {ensemble_rate_by_repeat.shape}")
        rate_matrix = pearson_corr_columns(ensemble_rate_by_repeat)
        lag_means, n_pairs = collapse_rate_matrix_by_lag(rate_matrix)
        block_lag_means.append(lag_means)
        block_n_pairs.append(n_pairs)
    return block_lag_means[0], block_lag_means[1], block_n_pairs[0], block_n_pairs[1]


def compute_mouse_rows(valid_rows: list[dict]) -> list[dict]:
    mouse_rows = []
    for row in valid_rows:
        path = NEUROPIXELS_DIR / row["filename"]
        mat_data = load_session(path)
        activity = extract_pm_activity(mat_data)
        n_units = int(row["n_units_pm"])
        if activity is None or not valid_activity_shape(activity, n_units):
            raise ValueError(f"Invalid activity for included session {path.name}: {None if activity is None else activity.shape}")

        block_a, block_b, block_a_n, block_b_n = compute_mouse_ensemble_rate(activity)
        for lag in range(1, N_REPEATS):
            idx = lag - 1
            mouse_rows.append(
                {
                    "filename": row["filename"],
                    "session_id": row["session_id"],
                    "n_units_pm": n_units,
                    "lag": lag,
                    "blockA_mean": block_a[idx],
                    "blockB_mean": block_b[idx],
                    "mouse_mean": float(np.nanmean([block_a[idx], block_b[idx]])),
                    "blockA_n_pairs": int(block_a_n[idx]),
                    "blockB_n_pairs": int(block_b_n[idx]),
                }
            )
        LOGGER.info("computed_session=%s n_units_pm=%s", row["filename"], n_units)
    return mouse_rows


def sem(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return float("nan")
    return float(np.nanstd(finite, ddof=1) / np.sqrt(finite.size))


def summarize_by_lag(mouse_rows: list[dict]) -> list[dict]:
    summary_rows = []
    for lag in range(1, N_REPEATS):
        values = np.array([float(row["mouse_mean"]) for row in mouse_rows if int(row["lag"]) == lag], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "lag": lag,
                "ensemble_rate_correlation_mean": float(np.nanmean(finite)),
                "ensemble_rate_correlation_sem": sem(finite),
                "n_mice": int(finite.size),
            }
        )
    return summary_rows


def validate_outputs(valid_rows: list[dict], mouse_rows: list[dict], summary_rows: list[dict]) -> None:
    if not valid_rows:
        raise AssertionError("valid mice count is zero")
    required_lags = list(range(1, N_REPEATS))
    mouse_lags = sorted({int(row["lag"]) for row in mouse_rows})
    summary_lags = [int(row["lag"]) for row in summary_rows]
    if mouse_lags != required_lags or summary_lags != required_lags:
        raise AssertionError(f"Lags are not exactly 1..29: mouse={mouse_lags}, summary={summary_lags}")
    if len(mouse_rows) != len(valid_rows) * (N_REPEATS - 1):
        raise AssertionError(f"mouse-by-lag row count mismatch: {len(mouse_rows)}")
    if len(summary_rows) != N_REPEATS - 1:
        raise AssertionError(f"summary row count mismatch: {len(summary_rows)}")
    for row in mouse_rows:
        lag = int(row["lag"])
        pair_count = N_REPEATS - lag
        if lag == 0:
            raise AssertionError("lag 0 appears in mouse-by-lag table")
        if int(row["blockA_n_pairs"]) != pair_count or int(row["blockB_n_pairs"]) != pair_count:
            raise AssertionError(f"n_pairs mismatch at lag {lag} for {row['filename']}")
    if FIGURE_PATH.name != "figure2h_pm_only_ensemble_rate_across_mice.png":
        raise AssertionError(f"Wrong figure path: {FIGURE_PATH}")
    output_csv_names = {
        VALID_MICE_CSV.name,
        MOUSE_BY_LAG_CSV.name,
        SUMMARY_BY_LAG_CSV.name,
    }
    if output_csv_names != {
        "figure2h_pm_only_valid_mice.csv",
        "figure2h_pm_only_mouse_by_lag.csv",
        "figure2h_pm_only_summary_by_lag.csv",
    }:
        raise AssertionError("Wrong CSV output paths")


def save_figure(summary_rows: list[dict]) -> None:
    lags = np.array([int(row["lag"]) for row in summary_rows], dtype=np.int64)
    means = np.array([float(row["ensemble_rate_correlation_mean"]) for row in summary_rows], dtype=np.float64)
    sems = np.array([float(row["ensemble_rate_correlation_sem"]) for row in summary_rows], dtype=np.float64)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    ax.fill_between(lags, means - sems, means + sems, color=PM_FILL_COLOR, alpha=0.4, linewidth=0)
    ax.plot(lags, means, color=PM_LINE_COLOR, linewidth=3)
    ax.text(0.05, 0.075, "1 repeat = 30 seconds", transform=ax.transAxes, fontsize=10)
    ax.set_title("PM only - Ensemble rate correlation")
    ax.set_xlabel("Elapsed time (# of movie repeats)")
    ax.set_ylabel("Ensemble rate correlation")
    ax.set_xlim(1, 29)
    ax.set_xticks([1, 10, 20, 29])
    ax.set_yticks(np.arange(0.86, 0.981, 0.04))
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_logging()
    valid_rows = load_valid_mice()
    mouse_rows = compute_mouse_rows(valid_rows)
    summary_rows = summarize_by_lag(mouse_rows)
    validate_outputs(valid_rows, mouse_rows, summary_rows)

    valid_fieldnames = [
        "filename",
        "session_id",
        "n_units_pm",
        "activity_shape",
        "included",
        "exclusion_reason",
    ]
    write_csv(VALID_MICE_CSV, valid_rows, valid_fieldnames)
    write_csv(MOUSE_BY_LAG_CSV, mouse_rows)
    write_csv(SUMMARY_BY_LAG_CSV, summary_rows)
    save_figure(summary_rows)

    LOGGER.info("valid_mice_count=%s", len(valid_rows))
    LOGGER.info("figure=%s", FIGURE_PATH)
    LOGGER.info("valid_mice_csv=%s", VALID_MICE_CSV)
    LOGGER.info("mouse_by_lag_csv=%s", MOUSE_BY_LAG_CSV)
    LOGGER.info("summary_by_lag_csv=%s", SUMMARY_BY_LAG_CSV)


if __name__ == "__main__":
    main()
