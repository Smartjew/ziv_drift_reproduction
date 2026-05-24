from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
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
COLORMAP_PATH = SOURCE_ROOT / "data" / "colormaps" / "newmap3.mat"

SESSION_FILENAME = "session_831882777.mat"
SESSION_PATH = NEUROPIXELS_DIR / SESSION_FILENAME

EXAMPLE_MOUSE = "mouse 53"
SORTED_SESSION_INDEX = 52
NAT_MOVIE_INDEX = 0
AREA_INDEX = 3
BLOCK_A_INDEX = 0

FRAMES_PER_REPEAT = 900
N_REPEATS = 30
N_BLOCKS = 2
N_TIME_BINS = 30
FRAMES_PER_BIN = 30
ACTIVITY_SHAPE = (72, 27000, 2)
POPULATION_SHAPE = (72, 30, 60)
CURRENT_MOUSE_SHAPE = (72, 30, 30)
MATRIX_SHAPE = (30, 30)

FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "figure2b_exact_mouse53_pm_blockA.png"
MATRIX_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2b_exact_mouse53_pm_blockA_matrix.csv"
PLOT_MATRIX_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2b_exact_mouse53_pm_blockA_plot_matrix.csv"
DEBUG_CURRENT_MOUSE_PATH = PROJECT_ROOT / "outputs" / "debug" / "figure2b_current_mouse_mouse53_pm_blockA.npy"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def verify_mouse_file_order() -> list[str]:
    filenames = sorted(path.name for path in NEUROPIXELS_DIR.glob("*.mat"))
    if len(filenames) <= SORTED_SESSION_INDEX:
        raise RuntimeError(f"Need at least {SORTED_SESSION_INDEX + 1} Neuropixels files, found {len(filenames)}")
    resolved_file = filenames[SORTED_SESSION_INDEX]
    if resolved_file != SESSION_FILENAME:
        raise RuntimeError(
            f"{EXAMPLE_MOUSE} did not resolve to {SESSION_FILENAME}; "
            f"sorted session index {SORTED_SESSION_INDEX} resolved to {resolved_file}"
        )
    return filenames


def load_required_mat_data() -> dict:
    mat_data = loadmat(
        str(SESSION_PATH),
        appendmat=False,
        simplify_cells=False,
        squeeze_me=False,
        struct_as_record=False,
        verify_compressed_data_integrity=True,
    )
    required = ("informative_rater_mat", "mean_running_speed_repeats", "cell_num")
    missing = [key for key in required if key not in mat_data]
    if missing:
        raise KeyError(f"Missing required MAT variables: {missing}")
    return {key: value for key, value in mat_data.items() if not key.startswith("__")}


def unwrap_scalar_object(value: object) -> object:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.reshape(-1)[0]
    return current


def extract_activity(mat_data: dict) -> np.ndarray:
    informative_rater_mat = mat_data["informative_rater_mat"]
    if not isinstance(informative_rater_mat, np.ndarray):
        raise TypeError(f"informative_rater_mat should be ndarray, got {type(informative_rater_mat).__name__}")
    activity = unwrap_scalar_object(informative_rater_mat[NAT_MOVIE_INDEX, AREA_INDEX])
    if not isinstance(activity, np.ndarray):
        raise TypeError(f"activity should be ndarray, got {type(activity).__name__}")
    if activity.shape != ACTIVITY_SHAPE:
        raise ValueError(f"Activity shape mismatch: {activity.shape}")
    return activity


def build_population_vectors_explicit(activity: np.ndarray) -> np.ndarray:
    n_units = activity.shape[0]
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

    if population_vectors.shape != (n_units, N_TIME_BINS, N_REPEATS * N_BLOCKS):
        raise ValueError(f"Population vector shape mismatch: {population_vectors.shape}")
    return population_vectors


def build_population_vectors_vectorized(activity: np.ndarray) -> np.ndarray:
    blocks = []
    for block_index in range(N_BLOCKS):
        block = activity[:, :, block_index]
        block_repeats = block.reshape(activity.shape[0], N_REPEATS, N_TIME_BINS, FRAMES_PER_BIN)
        blocks.append(block_repeats.mean(axis=3).transpose(0, 2, 1))
    return np.concatenate(blocks, axis=2)


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


def compute_figure2b_matrix(current_mouse: np.ndarray) -> np.ndarray:
    mean_current_movie_pv_corr = np.empty(MATRIX_SHAPE, dtype=np.float64)
    for repeat1 in range(N_REPEATS):
        for repeat2 in range(N_REPEATS):
            x = current_mouse[:, :, repeat1]
            y = current_mouse[:, :, repeat2]
            current_pv = pearson_corr_columns(x, y)
            mean_current_movie_pv_corr[repeat1, repeat2] = np.nanmean(np.diag(current_pv))
    return mean_current_movie_pv_corr


def create_plot_matrix(matrix: np.ndarray) -> np.ndarray:
    plot_matrix = matrix.copy()
    off_diagonal = plot_matrix.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    max_off_diagonal = float(np.nanmax(off_diagonal))
    np.fill_diagonal(plot_matrix, max_off_diagonal)
    return plot_matrix


def lag_mean(matrix: np.ndarray, lag: int) -> float:
    return float(np.nanmean(np.diagonal(matrix, offset=lag)))


def validate_outputs(
    activity: np.ndarray,
    population_vectors: np.ndarray,
    current_mouse: np.ndarray,
    matrix: np.ndarray,
    vectorized_diff: float,
) -> None:
    if activity.shape != ACTIVITY_SHAPE:
        raise AssertionError(f"activity shape mismatch: {activity.shape}")
    if population_vectors.shape != POPULATION_SHAPE:
        raise AssertionError(f"population_vectors shape mismatch: {population_vectors.shape}")
    if current_mouse.shape != CURRENT_MOUSE_SHAPE:
        raise AssertionError(f"current_mouse shape mismatch: {current_mouse.shape}")
    if matrix.shape != MATRIX_SHAPE:
        raise AssertionError(f"matrix shape mismatch: {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-12, equal_nan=True):
        raise AssertionError("Figure 2B matrix is not symmetric within tolerance")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise AssertionError("Pre-plot matrix diagonal is not close to 1")
    if vectorized_diff > 1e-12:
        raise AssertionError(f"Explicit/vectorized population vectors differ: {vectorized_diff}")
    if lag_mean(matrix, 1) <= lag_mean(matrix, 29):
        raise AssertionError("Lag 1 mean is not greater than lag 29 mean")


def save_figure(plot_matrix: np.ndarray) -> None:
    colormap_data = loadmat(str(COLORMAP_PATH), appendmat=False)
    if "newmap3" not in colormap_data:
        raise KeyError(f"Missing newmap3 variable in {COLORMAP_PATH}")
    newmap3 = colormap_data["newmap3"]
    if not isinstance(newmap3, np.ndarray) or newmap3.ndim != 2 or newmap3.shape[1] != 3:
        raise ValueError(f"newmap3 shape mismatch: {getattr(newmap3, 'shape', None)}")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    image = ax.imshow(plot_matrix, cmap=ListedColormap(newmap3), aspect="equal", interpolation="nearest")
    ax.set_title("Single animal example")
    ax.set_xlabel("Movie repeat")
    ax.set_ylabel("Movie repeat")
    ax.set_xticks([0, 4, 9, 14, 19, 24, 29], labels=[1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0, 4, 9, 14, 19, 24, 29], labels=[1, 5, 10, 15, 20, 25, 30])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("PV correlation")
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_logging()
    filenames = verify_mouse_file_order()
    mat_data = load_required_mat_data()
    activity = extract_activity(mat_data)

    population_vectors = build_population_vectors_explicit(activity)
    vectorized_vectors = build_population_vectors_vectorized(activity)
    vectorized_diff = float(np.max(np.abs(population_vectors - vectorized_vectors)))

    current_mouse = population_vectors[:, :, :N_REPEATS]
    matrix = compute_figure2b_matrix(current_mouse)
    plot_matrix = create_plot_matrix(matrix)

    validate_outputs(activity, population_vectors, current_mouse, matrix, vectorized_diff)

    DEBUG_CURRENT_MOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(DEBUG_CURRENT_MOUSE_PATH, current_mouse)
    MATRIX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(MATRIX_CSV_PATH, matrix, delimiter=",", fmt="%.17g")
    np.savetxt(PLOT_MATRIX_CSV_PATH, plot_matrix, delimiter=",", fmt="%.17g")
    save_figure(plot_matrix)

    LOGGER.info("figure=%s", FIGURE_PATH)
    LOGGER.info("matrix_csv=%s", MATRIX_CSV_PATH)


if __name__ == "__main__":
    main()
