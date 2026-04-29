from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

FINAL_SCRIPTS = [
    Path("src/analysis/figure2b_exact_mouse53_pm_blockA.py"),
    Path("src/analysis/reproduce_figure2c_exact.py"),
    Path("src/analysis/reproduce_figure2e_pm_only.py"),
    Path("src/analysis/reproduce_figure2h_pm_only.py"),
]

FINAL_ARTIFACTS = [
    Path("outputs/figures/figure2b_exact_mouse53_pm_blockA.png"),
    Path("outputs/figures/figure2c_exact_mouse53_pm_blockA.png"),
    Path("outputs/figures/figure2e_pm_only_pv_across_mice.png"),
    Path("outputs/figures/figure2h_pm_only_ensemble_rate_across_mice.png"),
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


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def check_artifacts_exist() -> bool:
    required = FINAL_SCRIPTS + FINAL_ARTIFACTS
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    logging.info("final_scripts=%s", len(FINAL_SCRIPTS))
    logging.info("final_artifacts=%s", len(FINAL_ARTIFACTS))
    if missing:
        for path in missing:
            logging.error("missing=%s", path)
        return False
    logging.info("all_final_artifacts_present=True")
    return True


def run_final_scripts() -> None:
    for script in FINAL_SCRIPTS:
        logging.info("running=%s", script)
        subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, check=True)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Final artifact checker for the Ziv drift reproduction.")
    parser.add_argument("--check", action="store_true", help="Check final scripts and artifacts exist.")
    parser.add_argument("--run", action="store_true", help="Regenerate final outputs by running final scripts in order.")
    args = parser.parse_args()

    if args.run:
        run_final_scripts()
    return 0 if check_artifacts_exist() else 1


if __name__ == "__main__":
    raise SystemExit(main())
