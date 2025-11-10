# Flow Cytometry Data Preparation UI

This desktop app helps you explore and curate large collections of flow cytometry (or other tabular) CSV files before model training.

## Features

- Multi-file CSV loader with per-file row/column summaries.
- Column coverage explorer highlighting overlap across files.
- Training feature selection with quick awareness of missing columns.
- Target column selection (categorical) with total and per-file category counts plus diagnostic plots.
- Visual dashboards for column coverage, target distributions, confusion matrices, and feature importance.
- Built-in configurable downsampling strategies (total, per file, per class, per file + class).
- DuckDB-powered out-of-core data engine with reusable parquet cache, enabling multi-GB CSV collections without exhausting RAM.
- Multi-model training with hyperparameter controls, cross-validation, multi-core execution, and rich evaluation metrics (Random Forest, LDA, SVM, Logistic Regression, Naive Bayes, XGBoost, LightGBM, k-means with label mapping, and a PyTorch-based neural network with optional GPU support). Built-in class balancing lets you apply per-class weights or sample weighting before fitting.
- Save-time controls for trimming Random Forest out-of-bag caches to keep exported bundles lightweight when needed.
- Clustering module with feature selection, class-aware downsampling, and configurable KMeans, Leiden, Louvain, and SOM+metaclustering pipelines.
- Clustering visualizations including UMAP projections (colored by clusters or marker intensity) and configurable cluster/marker heatmaps.
- Environment snapshot export and PyInstaller-based desktop bundle generation tools for reproducible deployments.

## Getting Started

1. Install requirements (Tkinter ships with the standard Python installer):
   ```bash
   python3 -m pip install -r requirements.txt
   ```
   *Optional but recommended:* install [DuckDB](https://duckdb.org/) + Polars from the requirements file to unlock the out-of-core loader. Install PyTorch separately using the wheel that matches your platform (see [pytorch.org](https://pytorch.org/get-started/locally/)) to enable the neural-network trainer, and install `xgboost`/`lightgbm` if you plan to use the boosted-tree models.
2. Launch the app:
   ```bash
   python3 app.py
   ```

## Packaging & Deployment

Use the built-in packaging helpers when you need to hand off the application or
capture the exact Python environment that produced a model export.

### Environment snapshots

- **UI:** Click **Export Environment Snapshot** from the top control bar to save a
  JSON file containing the Python version, dependency list (via `pip freeze`), and
  relevant configuration metadata (cached session state and optional application
  resources).
- **CLI:**
  ```bash
  python -m packaging_tools snapshot snapshots/flow_environment.json
  ```
  Supply `--config` multiple times to include additional configuration paths and
  `--env-prefix` to capture custom environment variables.

### Building desktop bundles

- **UI:** Select **Build Desktop Bundle** to run PyInstaller from within the
  application. The bundle is created in the directory you choose and is validated
  for the current platform. Each build also stores an accompanying environment
  snapshot for auditing.
- **CLI:**
  ```bash
  python -m packaging_tools bundle --output-dir dist_bundle --validate
  ```
  Customize the build with flags such as `--onefile`, `--icon`, or
  `--add-data path:target` to ship additional resources. Use `python -m
  packaging_tools validate dist_bundle/dist/FlowTrainingUI` to re-run validation
  on any generated artifact, including bundles produced on other platforms.

> **Testing bundles across platforms:** The `validate` command enforces sensible
> defaults for Linux, macOS, and Windows artifacts (executable permissions,
> `.app` folder structure, and `.exe` naming). Combine it with your CI matrix to
> automatically verify the integrity of bundles produced on different operating
> systems.

## Usage Tips

1. Select one or more CSV files. The summary labels update with total files, combined row counts, and unique columns.
2. Use the *Columns* tab to review column overlap and identify common fields.
3. In *Training Setup*, choose your training features. Columns missing from any file appear in red.
4. Pick a categorical target column to see global and per-file class balances.
5. Try the downsampling preview to check the effect of different balancing strategies. The preview tables show per-file and per-class row counts after sampling.
6. Pick a model type (Random Forest, LDA, SVM, KMeans, or Neural Network), adjust its hyperparameters, choose the test split, cross-validation folds, and CPU usage, then click **Train Model**.
7. Review the *Model Results* tab for accuracy, F1 scores, classification report, confusion matrix heatmap, and feature importance plots (where supported). Use **Save Model…** to persist the trained model package (includes metadata, metrics, and any required artifacts such as scalers). The *Save Options* group lets you drop large Random Forest OOB caches before exporting when disk size matters.
8. Move to the *Clustering Module* tab to explore unsupervised setups: select clustering features, optional class-balanced downsampling, choose algorithms (KMeans, Leiden, Louvain, SOM + metaclustering), configure their parameters, choose CPU usage, and run clustering to review per-method summaries and cluster membership counts.
9. Use the *Visualization* sub-tab to generate UMAP embeddings (colored by cluster labels or marker expression) and cluster/marker heatmaps with customizable normalization.
10. Save clustering results to export both the annotated dataset (CSV/Parquet) and a companion JSON file capturing run metadata (files, features, parameters, and cluster sizes per method).

> The preview keeps a `__source_file` column so you can trace rows back to their origin. Drop it if you don't need it for downstream work.
