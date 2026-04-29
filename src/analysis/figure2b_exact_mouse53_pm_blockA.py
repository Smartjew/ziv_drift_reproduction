from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"C:\ziv_drift_workspace\ziv_drift_reproduction")
SOURCE_ROOT = Path(r"C:\ziv_drift_workspace\visual_drift-main\visual_drift-main")
NEUROPIXELS_DIR = SOURCE_ROOT / "data" / "neuropixels"
MATLAB_SCRIPT_PATH = SOURCE_ROOT / "scripts" / "visual_drift_analysis.m"
COLORMAP_PATH = SOURCE_ROOT / "data" / "colormaps" / "newmap3.mat"

SESSION_FILENAME = "session_831882777.mat"
SESSION_PATH = NEUROPIXELS_DIR / SESSION_FILENAME

MATLAB_MOUSE_INDEX = 53
PYTHON_MOUSE_INDEX = 52
NAT_MOVIE_INDEX = 0
AREA_INDEX = 3
BLOCK_A_INDEX = 0

FRAMES_PER_REPEAT = 900
N_REPEATS = 30
N_BLOCKS = 2
N_TIME_BINS = 30
FRAMES_PER_BIN = 30
EXPECTED_ACTIVITY_SHAPE = (72, 27000, 2)
EXPECTED_POPULATION_SHAPE = (72, 30, 60)
EXPECTED_CURRENT_MOUSE_SHAPE = (72, 30, 30)
EXPECTED_MATRIX_SHAPE = (30, 30)

FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "figure2b_exact_mouse53_pm_blockA.png"
MATRIX_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2b_exact_mouse53_pm_blockA_matrix.csv"
PLOT_MATRIX_CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "figure2b_exact_mouse53_pm_blockA_plot_matrix.csv"
DIAGNOSTICS_PATH = PROJECT_ROOT / "outputs" / "reports" / "figure2b_exact_mouse53_pm_blockA_diagnostics.txt"
DEBUG_CURRENT_MOUSE_PATH = PROJECT_ROOT / "outputs" / "debug" / "figure2b_current_mouse_mouse53_pm_blockA.npy"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def verify_mouse_file_order() -> list[str]:
    filenames = sorted(path.name for path in NEUROPIXELS_DIR.glob("*.mat"))
    if len(filenames) <= PYTHON_MOUSE_INDEX:
        raise RuntimeError(f"Expected at least {PYTHON_MOUSE_INDEX + 1} Neuropixels files, found {len(filenames)}")
    resolved_file = filenames[PYTHON_MOUSE_INDEX]
    if resolved_file != SESSION_FILENAME:
        raise RuntimeError(
            f"MATLAB mouse {MATLAB_MOUSE_INDEX} did not resolve to {SESSION_FILENAME}; "
            f"Python index {PYTHON_MOUSE_INDEX} resolved to {resolved_file}"
        )
    return filenames


def extract_matlab_source_lines() -> list[str]:
    wanted_lines = set(range(756, 763)) | set(range(767, 771)) | set(range(774, 777))
    lines = MATLAB_SCRIPT_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return [f"{line_no}: {lines[line_no - 1]}" for line_no in sorted(wanted_lines)]


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
        raise TypeError(f"Expected informative_rater_mat ndarray, got {type(informative_rater_mat).__name__}")
    activity = unwrap_scalar_object(informative_rater_mat[NAT_MOVIE_INDEX, AREA_INDEX])
    if not isinstance(activity, np.ndarray):
        raise TypeError(f"Expected activity ndarray, got {type(activity).__name__}")
    if activity.shape != EXPECTED_ACTIVITY_SHAPE:
        raise ValueError(f"Expected activity shape {EXPECTED_ACTIVITY_SHAPE}, got {activity.shape}")
    return activity


def build_population_vectors_explicit(activity: np.ndarray) -> np.ndarray:
    n_units = activity.shape[0]
    population_vectors = np.empty(EXPECTED_POPULATION_SHAPE, dtype=np.float64)
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
        raise ValueError(f"Unexpected population vector shape: {population_vectors.shape}")
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
    mean_current_movie_pv_corr = np.empty(EXPECTED_MATRIX_SHAPE, dtype=np.float64)
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
    if SESSION_FILENAME != "session_831882777.mat":
        raise AssertionError(f"Forbidden session filename for exact Figure 2B output: {SESSION_FILENAME}")
    if activity.shape != EXPECTED_ACTIVITY_SHAPE:
        raise AssertionError(f"activity shape mismatch: {activity.shape}")
    if population_vectors.shape != EXPECTED_POPULATION_SHAPE:
        raise AssertionError(f"population_vectors shape mismatch: {population_vectors.shape}")
    if current_mouse.shape != EXPECTED_CURRENT_MOUSE_SHAPE:
        raise AssertionError(f"current_mouse shape mismatch: {current_mouse.shape}")
    if matrix.shape != EXPECTED_MATRIX_SHAPE:
        raise AssertionError(f"matrix shape mismatch: {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-12, equal_nan=True):
        raise AssertionError("Figure 2B matrix is not symmetric within tolerance")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise AssertionError("Pre-plot matrix diagonal is not close to 1")
    if vectorized_diff > 1e-12:
        raise AssertionError(f"Explicit/vectorized population vectors differ: {vectorized_diff}")
    if lag_mean(matrix, 1) <= lag_mean(matrix, 29):
        raise AssertionError("Lag 1 mean is not greater than lag 29 mean")
    if abs(lag_mean(matrix, 1) - 0.8791) > 1e-3:
        raise AssertionError(f"Lag 1 sanity check failed: {lag_mean(matrix, 1)}")
    if abs(lag_mean(matrix, 29) - 0.7623) > 1e-3:
        raise AssertionError(f"Lag 29 sanity check failed: {lag_mean(matrix, 29)}")


def save_figure(plot_matrix: np.ndarray) -> None:
    colormap_data = loadmat(str(COLORMAP_PATH), appendmat=False)
    if "newmap3" not in colormap_data:
        raise KeyError(f"Missing newmap3 variable in {COLORMAP_PATH}")
    newmap3 = colormap_data["newmap3"]
    if not isinstance(newmap3, np.ndarray) or newmap3.ndim != 2 or newmap3.shape[1] != 3:
        raise ValueError(f"Expected newmap3 shape (n, 3), got {getattr(newmap3, 'shape', None)}")

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


def write_diagnostics(
    mat_data: dict,
    filenames: list[str],
    matlab_lines: list[str],
    activity: np.ndarray,
    population_vectors: np.ndarray,
    current_mouse: np.ndarray,
    matrix: np.ndarray,
    plot_matrix: np.ndarray,
    vectorized_diff: float,
) -> None:
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    symmetry_error = float(np.nanmax(np.abs(matrix - matrix.T)))

    lines = [
        "Figure 2B exact forensic reproduction diagnostics",
        "",
        f"source_matlab_script = {MATLAB_SCRIPT_PATH}",
        "matlab_source_lines:",
        *matlab_lines,
        "",
        f"neuropixels_dir = {NEUROPIXELS_DIR}",
        f"matlab_mouse_index = {MATLAB_MOUSE_INDEX}",
        f"python_mouse_index = {PYTHON_MOUSE_INDEX}",
        f"resolved_file = {filenames[PYTHON_MOUSE_INDEX]}",
        f"session_path = {SESSION_PATH}",
        f"sorted_file_count = {len(filenames)}",
        "",
        "target_parameters:",
        f"nat_movie_matlab = 1",
        f"nat_movie_index_python = {NAT_MOVIE_INDEX}",
        f"area_matlab = 4",
        f"area_index_python = {AREA_INDEX}",
        f"block = A",
        f"block_index_python = {BLOCK_A_INDEX}",
        f"frames_per_repeat = {FRAMES_PER_REPEAT}",
        f"n_repeats = {N_REPEATS}",
        f"n_time_bins = {N_TIME_BINS}",
        f"frames_per_bin = {FRAMES_PER_BIN}",
        "",
        "required_variables:",
        f"informative_rater_mat_shape = {mat_data['informative_rater_mat'].shape}",
        f"mean_running_speed_repeats_shape = {mat_data['mean_running_speed_repeats'].shape}",
        f"cell_num_shape = {mat_data['cell_num'].shape}",
        "",
        "activity:",
        f"shape = {activity.shape}",
        f"dtype = {activity.dtype}",
        f"min = {np.nanmin(activity)}",
        f"max = {np.nanmax(activity)}",
        f"nan_count = {int(np.isnan(activity).sum())}",
        f"finite_count = {int(np.isfinite(activity).sum())}",
        "",
        "population_vectors:",
        f"explicit_shape = {population_vectors.shape}",
        f"current_mouse_shape = {current_mouse.shape}",
        f"debug_current_mouse_path = {DEBUG_CURRENT_MOUSE_PATH}",
        f"explicit_vs_vectorized_max_abs_diff = {vectorized_diff}",
        "",
        "figure2b_matrix_before_plot_diagonal_replacement:",
        f"shape = {matrix.shape}",
        f"diagonal_min = {float(np.nanmin(np.diag(matrix)))}",
        f"diagonal_max = {float(np.nanmax(np.diag(matrix)))}",
        f"off_diagonal_min = {float(np.nanmin(off_diagonal))}",
        f"off_diagonal_max = {float(np.nanmax(off_diagonal))}",
        f"off_diagonal_mean = {float(np.nanmean(off_diagonal))}",
        f"off_diagonal_median = {float(np.nanmedian(off_diagonal))}",
        f"symmetry_max_abs_error = {symmetry_error}",
        f"nan_count = {int(np.isnan(matrix).sum())}",
        f"lag1 = {lag_mean(matrix, 1)}",
        f"lag2 = {lag_mean(matrix, 2)}",
        f"lag5 = {lag_mean(matrix, 5)}",
        f"lag10 = {lag_mean(matrix, 10)}",
        f"lag20 = {lag_mean(matrix, 20)}",
        f"lag29 = {lag_mean(matrix, 29)}",
        f"lag1_minus_lag29 = {lag_mean(matrix, 1) - lag_mean(matrix, 29)}",
        "",
        "plot_matrix_after_diagonal_replacement:",
        f"diagonal_value = {float(np.diag(plot_matrix)[0])}",
        f"plot_matrix_csv = {PLOT_MATRIX_CSV_PATH}",
        "",
        "outputs:",
        f"figure = {FIGURE_PATH}",
        f"matrix_csv = {MATRIX_CSV_PATH}",
        f"diagnostics = {DIAGNOSTICS_PATH}",
        "",
        "final_source_confirmation:",
        "session = session_831882777.mat",
        "stimulus = Natural Movie 1",
        "area = VISpm / PM",
        "block = A",
        "repeats = 1-30",
        "units = 72",
        "time_bins = 30",
    ]

    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_logging()
    filenames = verify_mouse_file_order()
    matlab_lines = extract_matlab_source_lines()
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
    write_diagnostics(
        mat_data=mat_data,
        filenames=filenames,
        matlab_lines=matlab_lines,
        activity=activity,
        population_vectors=population_vectors,
        current_mouse=current_mouse,
        matrix=matrix,
        plot_matrix=plot_matrix,
        vectorized_diff=vectorized_diff,
    )

    LOGGER.info("figure=%s", FIGURE_PATH)
    LOGGER.info("matrix_csv=%s", MATRIX_CSV_PATH)
    LOGGER.info("diagnostics=%s", DIAGNOSTICS_PATH)


if __name__ == "__main__":
    main()
