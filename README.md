# Representational Drift Reproduction - Figure 2 Selected Panels

## Project Objective

This project is a focused Python reproduction of selected results from Deitch, Rubin, and Ziv, 2021, "Representational drift in the mouse visual cortex." The goal is to reproduce a small, auditable subset of the paper that demonstrates representational drift in mouse visual cortex.

The project uses processed Neuropixels data from the official Ziv lab `visual_drift` repository. The analysis is limited to Natural Movie 1 and area PM / VISpm. The final outputs are Figure 2B, Figure 2C, Figure 2E PM-only, and Figure 2H PM-only. Figure 2E and Figure 2H are PM-only reproductions, not full all-area reproductions.

## Scientific Background

Representational drift means that neural responses to the same stimulus gradually change over time. In this project, drift is measured by comparing responses to repeated presentations of Natural Movie 1.

Population-vector correlation measures similarity between population activity patterns. For Figure 2B and Figure 2E, each repeat is kept as a matrix of units by time bins, and corresponding time-bin correlations are averaged. Ensemble rate correlation is different: for Figure 2H, each movie repeat is collapsed into one activity-rate vector per unit, and repeat vectors are correlated.

Figure 2B and Figure 2C are single-mouse example analyses. Figure 2E and Figure 2H extend the PM / VISpm analysis across valid Neuropixels mice.

## Data

Data source: official processed data from the Ziv lab `visual_drift` repository.

Local Neuropixels data path used by the scripts:

```text
C:\ziv_drift_workspace\visual_drift-main\visual_drift-main\data\neuropixels
```

Main processed data variable:

```text
informative_rater_mat
```

Analysis scope:

- Stimulus: Natural Movie 1
- Area: PM / VISpm
- Inclusion rule: Functional Connectivity sessions with 30 repeats per block and at least 15 PM units

## Project Structure

```text
main.py                 Lightweight final artifact checker
pyproject.toml          Project metadata and dependencies
src\analysis\           Final reproduction scripts
tests\                  Final artifact tests
outputs\figures\        Final output figures
outputs\tables\         Final output tables
```

## Final Scripts

- `src\analysis\figure2b_exact_mouse53_pm_blockA.py`: reproduces the single-mouse Figure 2B PM block A heatmap.
- `src\analysis\reproduce_figure2c_exact.py`: reproduces the single-mouse Figure 2C lag collapse from the Figure 2B matrix.
- `src\analysis\reproduce_figure2e_pm_only.py`: reproduces PM-only Figure 2E PV correlation across valid Neuropixels mice.
- `src\analysis\reproduce_figure2h_pm_only.py`: reproduces PM-only Figure 2H ensemble rate correlation across valid Neuropixels mice.

## How To Run

Run commands from the project root:

```powershell
cd C:\ziv_drift_workspace\ziv_drift_reproduction
```

Create a fresh Python environment and install the project dependencies:

```powershell
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
```

If `py -3.11` is not available, use any Python executable that satisfies `requires-python = ">=3.11"` in `pyproject.toml`.

Lightweight artifact check:

```powershell
& '.\.venv\Scripts\python.exe' '.\main.py' --check
```

Tests:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest
```

Reproduce Figure 2B:

```powershell
& '.\.venv\Scripts\python.exe' '.\src\analysis\figure2b_exact_mouse53_pm_blockA.py'
```

Reproduce Figure 2C:

```powershell
& '.\.venv\Scripts\python.exe' '.\src\analysis\reproduce_figure2c_exact.py'
```

Reproduce Figure 2E PM-only:

```powershell
& '.\.venv\Scripts\python.exe' '.\src\analysis\reproduce_figure2e_pm_only.py'
```

Reproduce Figure 2H PM-only:

```powershell
& '.\.venv\Scripts\python.exe' '.\src\analysis\reproduce_figure2h_pm_only.py'
```

Optional regeneration:

```powershell
& '.\.venv\Scripts\python.exe' '.\main.py' --run
```

The `--run` option runs all four final scripts in order. It is not needed for a lightweight check.

## Final Outputs

Final figures:

- `outputs\figures\figure2b_exact_mouse53_pm_blockA.png`
- `outputs\figures\figure2c_exact_mouse53_pm_blockA.png`
- `outputs\figures\figure2e_pm_only_pv_across_mice.png`
- `outputs\figures\figure2h_pm_only_ensemble_rate_across_mice.png`

Final tables:

- `outputs\tables\figure2b_exact_mouse53_pm_blockA_matrix.csv`
- `outputs\tables\figure2b_exact_mouse53_pm_blockA_plot_matrix.csv`
- `outputs\tables\figure2c_exact_mouse53_pm_blockA_raw_points.csv`
- `outputs\tables\figure2c_exact_mouse53_pm_blockA_lag_means.csv`
- `outputs\tables\figure2e_pm_only_valid_mice.csv`
- `outputs\tables\figure2e_pm_only_mouse_by_lag.csv`
- `outputs\tables\figure2e_pm_only_summary_by_lag.csv`
- `outputs\tables\figure2h_pm_only_valid_mice.csv`
- `outputs\tables\figure2h_pm_only_mouse_by_lag.csv`
- `outputs\tables\figure2h_pm_only_summary_by_lag.csv`

## Validation

`main.py` checks that the final scripts and final artifacts exist. It does not rerun the analyses by default.

The pytest suite validates:

- final figures and tables exist and are non-empty
- Figure 2C raw point count and lag structure
- Figure 2C lag-pair counts
- Figure 2E and Figure 2H summary lag ranges and basic lag 1 > lag 29 trend
- valid mice consistency between Figure 2E and Figure 2H
- legacy outputs are absent from final output folders
- `main.py --check` succeeds

## Limitations

This is not a full reproduction of all figures in the paper. It focuses only on selected Figure 2 panels.

Figure 2E and Figure 2H were reproduced only for PM / VISpm, not for all visual areas. The project uses processed data from the authors' repository rather than rebuilding the dataset from raw AllenSDK NWB files. Full all-area reproduction, tuning-curve correlation, calcium imaging analyses, and additional supplementary controls are possible extensions.

## References

- Deitch, Rubin, and Ziv, 2021, Current Biology, "Representational drift in the mouse visual cortex."
- Official Ziv lab `visual_drift` GitHub repository.
- Allen Brain Observatory / AllenSDK as the original data source.
