import copy
import re
import sys
import tkinter as tk
import multiprocessing
import queue
import time
from collections import defaultdict
import random
import uuid
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Set, TYPE_CHECKING

from tkinter import filedialog, messagebox, ttk, simpledialog

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle
from pandas.api.types import is_numeric_dtype

from data_engine import DataEngine, DataEngineError
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    adjusted_rand_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
try:
    from scipy import sparse  # type: ignore[import]
except ImportError:
    sparse = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix as CSRMatrix  # type: ignore[import]
else:  # pragma: no cover - runtime fallback when SciPy is unavailable
    CSRMatrix = Any

try:
    import igraph as ig  # type: ignore[import]
except ImportError:
    ig = None  # type: ignore[assignment]

try:
    import leidenalg  # type: ignore[import]
except ImportError:
    leidenalg = None  # type: ignore[assignment]

try:
    import networkx as nx  # type: ignore[import]
except ImportError:
    nx = None  # type: ignore[assignment]

try:
    import community as community_louvain  # type: ignore[import]
except ImportError:
    community_louvain = None  # type: ignore[assignment]

try:
    from minisom import MiniSom  # type: ignore[import]
except ImportError:
    MiniSom = None  # type: ignore[assignment]

try:
    import umap  # type: ignore[import]
except ImportError:
    umap = None  # type: ignore[assignment]

try:
    import torch  # type: ignore[import]
    from torch import nn  # type: ignore[import]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import]
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


RANDOM_STATE = 42
MAX_UNIQUE_CATEGORY_SAMPLE = 200
MAX_CENTROID_LABELS = 40
MAX_CLUSTER_COMPARE_CATEGORIES = 30
CSV_METADATA_SAMPLE_ROWS = 5000
DEFAULT_CSV_CHUNKSIZE = 200_000
IS_DARWIN = sys.platform == "darwin"


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window_id, width=event.width),
        )

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

        self.scrollable_frame.bind("<Enter>", self._bind_to_mousewheel)
        self.scrollable_frame.bind("<Leave>", self._unbind_from_mousewheel)
        self.canvas.bind("<Enter>", self._bind_to_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_from_mousewheel)

    def _bind_to_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_from_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Shift-MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
        elif event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        if event.delta:
            self.canvas.xview_scroll(-int(event.delta / 120), "units")
        elif event.num == 4:
            self.canvas.xview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.xview_scroll(1, "units")


@dataclass
class ColumnProfile:
    column: str
    is_numeric: Optional[bool] = None
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    categories: Optional[List[str]] = None
    categories_capped: bool = False
    unique_count: Optional[int] = None
    unique_capped: bool = False


@dataclass
class DataFile:
    path: Path
    columns: List[str]
    dtype_hints: Dict[str, str]
    _row_count: Optional[int] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def row_count(self) -> int:
        if self._row_count is None:
            self._row_count = self._compute_row_count()
        return self._row_count

    def _compute_row_count(self) -> int:
        if not self.columns:
            return 0
        total = 0
        try:
            reader = pd.read_csv(
                self.path,
                usecols=[self.columns[0]],
                chunksize=DEFAULT_CSV_CHUNKSIZE,
                low_memory=False,
            )
            for chunk in reader:
                total += len(chunk.index)
        except Exception:
            total = 0
        return total

    def has_column(self, column: str) -> bool:
        return column in self.columns

    def dtype_hint(self, column: str) -> Optional[str]:
        return self.dtype_hints.get(column)

    def iter_chunks(
        self,
        usecols: Optional[List[str]] = None,
        chunksize: int = DEFAULT_CSV_CHUNKSIZE,
    ) -> Iterator[pd.DataFrame]:
        reader = pd.read_csv(
            self.path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            yield chunk


if torch is not None and nn is not None and TensorDataset is not None:

    class SimpleMLP(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, activation: str, dropout: float) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            act_layer_factory = nn.ReLU if activation == "relu" else nn.Tanh
            for units in hidden_layers:
                layers.append(nn.Linear(prev_dim, units))
                layers.append(act_layer_factory())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = units
            layers.append(nn.Linear(prev_dim, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.model(x)


    class TorchNeuralNetClassifier:
        def __init__(
            self,
            input_dim: int,
            hidden_layers: List[int],
            output_dim: int,
            activation: str,
            dropout: float,
            learning_rate: float,
            weight_decay: float,
            epochs: int,
            batch_size: int,
            device: str,
            class_weights: Optional[Dict[object, float]] = None,
        ) -> None:
            self.device = torch.device(device)
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = SimpleMLP(input_dim, hidden_layers, output_dim, activation, dropout).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.criterion: Optional[nn.Module] = None
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.classes_: List[str] = []
            self.class_weight_mapping = class_weights or {}

        def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            X_scaled = self.scaler.fit_transform(X_train)
            y_encoded = self.label_encoder.fit_transform(y_train)
            self.classes_ = list(self.label_encoder.classes_)
            if self.class_weight_mapping:
                ordered = [self.class_weight_mapping.get(cls, 1.0) for cls in self.classes_]
                weight_tensor = torch.tensor(ordered, dtype=torch.float32, device=self.device)
            else:
                weight_tensor = None
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            dataset = TensorDataset(
                torch.from_numpy(X_scaled.astype(np.float32)),
                torch.from_numpy(y_encoded.astype(np.int64)),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for _ in range(self.epochs):
                self.model.train()
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    self.optimizer.zero_grad()
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
                    loss.backward()
                    self.optimizer.step()

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_scaled = self.scaler.transform(X)
            tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = self.label_encoder.inverse_transform(preds)
            return labels

        def to_cpu(self) -> None:
            self.model.to(torch.device("cpu"))
            self.device = torch.device("cpu")

else:

    class TorchNeuralNetClassifier:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN001
            raise ImportError("PyTorch is required for neural network training.")

        def fit(self, *args, **kwargs) -> None:  # noqa: ANN001, D401
            raise ImportError("PyTorch is required for neural network training.")

        def predict(self, *args, **kwargs):  # noqa: ANN001, D401
            raise ImportError("PyTorch is required for neural network training.")

        def to_cpu(self) -> None:
            return


class FlowDataApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Flow Cytometry Data Preparation")
        self.root.geometry("1100x700")

        self.app_dir = Path(__file__).resolve().parent
        self.cache_dir = self.app_dir / ".flow_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.data_engine = DataEngine(self.cache_dir)
        self.schema_report: List[Dict[str, object]] = []
        self.quality_overview: Dict[str, object] = {}
        self.session_path = self.cache_dir / "session_state.json"
        self.session_dirty = False
        self.run_registry_path = self.cache_dir / "run_registry.json"
        self.run_registry: List[Dict[str, object]] = []
        self._pending_training_restore: Optional[Dict[str, object]] = None
        self.quality_tree: Optional[ttk.Treeview] = None
        self.quality_summary_text: Optional[tk.Text] = None
        self.run_registry_tree: Optional[ttk.Treeview] = None
        self.run_detail_text: Optional[tk.Text] = None
        self._loading_session = False

        self.data_files: List[DataFile] = []
        self.column_presence: Dict[str, int] = {}
        self.common_columns: List[str] = []
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.column_numeric_hints: Dict[str, bool] = {}
        self.available_training_columns: List[str] = []
        self.training_selection: List[str] = []
        self.target_column_var = tk.StringVar()
        self.target_unique_counts: Dict[str, int] = {}
        self.target_unique_capped: Dict[str, bool] = {}
        self.target_counts_total: Dict[str, int] = {}
        self.target_counts_per_file: Dict[str, Dict[str, int]] = {}
        self.target_missing_files: List[str] = []
        self.training_missing_var = tk.StringVar(value="")
        self.target_info_var = tk.StringVar(
            value="Select a classification column to view category statistics."
        )
        self.target_missing_var = tk.StringVar(value="")
        self.training_hint_var = tk.StringVar(
            value="Load files to populate feature options."
        )

        self.training_downsample_method_var = tk.StringVar(value="None")
        self.training_downsample_value_var = tk.StringVar(value="")
        self.training_downsample_message_var = tk.StringVar(
            value="Configure a downsampling strategy to preview its effect."
        )
        self.training_downsample_value_label_var = tk.StringVar(value="Target size")
        self.training_downsampled_df: Optional[pd.DataFrame] = None

        total_cpu_cores = multiprocessing.cpu_count()
        default_jobs = max(total_cpu_cores - 1, 1)
        self.total_cpu_cores = total_cpu_cores
        self.training_queue: "queue.Queue[dict]" = queue.Queue()
        self.training_thread: Optional[Thread] = None
        self.training_in_progress = False
        self.trained_model: Optional[object] = None
        self.training_results: Dict[str, object] = {}
        self.csv_chunksize = DEFAULT_CSV_CHUNKSIZE

        self.training_model_var = tk.StringVar(value="Random Forest")
        self.n_estimators_var = tk.IntVar(value=300)
        self.max_depth_var = tk.StringVar(value="")
        self.max_features_var = tk.StringVar(value="sqrt")
        self.min_samples_leaf_var = tk.IntVar(value=1)
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.cv_folds_var = tk.IntVar(value=5)
        self.n_jobs_var = tk.IntVar(value=default_jobs)
        self.training_status_var = tk.StringVar(value="Training not started.")
        # Model-specific hyperparameters
        self.lda_solver_var = tk.StringVar(value="svd")
        self.lda_shrinkage_var = tk.StringVar(value="")
        self.svm_kernel_var = tk.StringVar(value="rbf")
        self.svm_c_var = tk.DoubleVar(value=1.0)
        self.svm_gamma_var = tk.StringVar(value="scale")
        self.svm_degree_var = tk.IntVar(value=3)
        self.kmeans_clusters_var = tk.IntVar(value=10)
        self.kmeans_max_iter_var = tk.IntVar(value=300)
        self.kmeans_n_init_var = tk.IntVar(value=10)
        self.nn_hidden_layers_var = tk.StringVar(value="128,64")
        self.nn_activation_var = tk.StringVar(value="relu")
        self.nn_dropout_var = tk.DoubleVar(value=0.2)
        self.nn_epochs_var = tk.IntVar(value=30)
        self.nn_batch_var = tk.IntVar(value=64)
        self.nn_lr_var = tk.DoubleVar(value=1e-3)
        self.nn_weight_decay_var = tk.DoubleVar(value=0.0)
        self.nn_use_gpu_var = tk.BooleanVar(value=True)
        self.class_balance_var = tk.StringVar(value="None")
        self.run_tags_var = tk.StringVar(value="")
        self.run_notes_var = tk.StringVar(value="")
        self.lr_solver_var = tk.StringVar(value="lbfgs")
        self.lr_penalty_var = tk.StringVar(value="l2")
        self.lr_c_var = tk.DoubleVar(value=1.0)
        self.lr_max_iter_var = tk.IntVar(value=400)
        self.lr_l1_ratio_var = tk.DoubleVar(value=0.5)
        self.nb_var_smoothing_var = tk.DoubleVar(value=1e-9)
        self.xgb_estimators_var = tk.IntVar(value=300)
        self.xgb_learning_rate_var = tk.DoubleVar(value=0.1)
        self.xgb_max_depth_var = tk.IntVar(value=6)
        self.xgb_subsample_var = tk.DoubleVar(value=0.8)
        self.xgb_colsample_var = tk.DoubleVar(value=0.8)
        self.lgb_estimators_var = tk.IntVar(value=400)
        self.lgb_learning_rate_var = tk.DoubleVar(value=0.05)
        self.lgb_max_depth_var = tk.IntVar(value=-1)
        self.lgb_num_leaves_var = tk.IntVar(value=31)
        self.lgb_subsample_var = tk.DoubleVar(value=0.8)
        self.keep_rf_oob_var = tk.BooleanVar(value=True)
        self.training_model_description_var = tk.StringVar(value="")
        self.class_balance_var.trace_add("write", lambda *_: self._mark_session_dirty())
        self.training_model_configs = {
            "Random Forest": {
                "key": "rf",
                "type": "supervised",
                "description": "Ensemble of decision trees; robust to mixed feature scales and handles feature importance.",
            },
            "LDA": {
                "key": "lda",
                "type": "supervised",
                "description": "Linear Discriminant Analysis (generative, assumes Gaussian classes). Works best on scaled numeric features.",
            },
            "SVM": {
                "key": "svm",
                "type": "supervised",
                "description": "Support Vector Machine classifier with optional kernels and regularization.",
            },
            "Logistic Regression": {
                "key": "logreg",
                "type": "supervised",
                "description": "Scaled logistic regression with configurable solvers and penalties.",
            },
            "Naive Bayes": {
                "key": "nb",
                "type": "supervised",
                "description": "Gaussian Naive Bayes for fast baselines (supports sample weighting).",
            },
            "XGBoost": {
                "key": "xgb",
                "type": "supervised",
                "description": "Gradient boosted trees via XGBoost (requires xgboost package).",
            },
            "LightGBM": {
                "key": "lgbm",
                "type": "supervised",
                "description": "Gradient boosted trees via LightGBM (requires lightgbm package).",
            },
            "KMeans": {
                "key": "kmeans",
                "type": "unsupervised",
                "description": "Unsupervised k-means clustering with majority-vote mapping back to class labels for evaluation.",
            },
            "Neural Network": {
                "key": "nn",
                "type": "supervised",
                "description": "Feed-forward neural network (PyTorch) with configurable hidden layers and optional GPU acceleration.",
            },
        }

        # Clustering module state
        self.clustering_methods = {
            "kmeans": {
                "label": "KMeans",
                "selected": tk.BooleanVar(value=True),
                "params": {
                    "n_clusters": {
                        "label": "Clusters (k)",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 2,
                        "to": 200,
                        "increment": 1,
                        "token": "k{}",
                        "display_token": "k={}",
                    },
                    "max_iter": {
                        "label": "Max iterations",
                        "var": tk.StringVar(value="300"),
                        "type": "int",
                        "from": 10,
                        "to": 1000,
                        "increment": 10,
                        "display_token": "iter={}",
                    },
                    "n_init": {
                        "label": "Initializations",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 1,
                        "to": 50,
                        "increment": 1,
                        "display_token": "n_init={}",
                    },
                },
            },
            "leiden": {
                "label": "Leiden",
                "selected": tk.BooleanVar(value=False),
                "params": {
                    "n_neighbors": {
                        "label": "Neighbors",
                        "var": tk.StringVar(value="15"),
                        "type": "int",
                        "from": 5,
                        "to": 200,
                        "increment": 1,
                        "token": "nn{}",
                        "display_token": "nn={}",
                    },
                    "resolution": {
                        "label": "Resolution",
                        "var": tk.StringVar(value="1.0"),
                        "type": "float",
                        "from": 0.1,
                        "to": 5.0,
                        "increment": 0.1,
                        "token": "res{}",
                        "display_token": "res={}",
                    },
                },
            },
            "louvain": {
                "label": "Louvain",
                "selected": tk.BooleanVar(value=False),
                "params": {
                    "n_neighbors": {
                        "label": "Neighbors",
                        "var": tk.StringVar(value="15"),
                        "type": "int",
                        "from": 5,
                        "to": 200,
                        "increment": 1,
                        "token": "nn{}",
                        "display_token": "nn={}",
                    },
                    "resolution": {
                        "label": "Resolution",
                        "var": tk.StringVar(value="1.0"),
                        "type": "float",
                        "from": 0.1,
                        "to": 5.0,
                        "increment": 0.1,
                        "token": "res{}",
                        "display_token": "res={}",
                    },
                },
            },
            "som_metacluster": {
                "label": "SOM + Metaclustering",
                "selected": tk.BooleanVar(value=False),
                "params": {
                    "grid_x": {
                        "label": "Grid X",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 2,
                        "to": 50,
                        "increment": 1,
                        "display_token": "grid_x={}",
                    },
                    "grid_y": {
                        "label": "Grid Y",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 2,
                        "to": 50,
                        "increment": 1,
                        "display_token": "grid_y={}",
                    },
                    "iterations": {
                        "label": "Iterations",
                        "var": tk.StringVar(value="2000"),
                        "type": "int",
                        "from": 100,
                        "to": 20000,
                        "increment": 100,
                        "display_token": "iter={}",
                    },
                    "meta_clusters": {
                        "label": "Meta clusters",
                        "var": tk.StringVar(value="15"),
                        "type": "int",
                        "from": 2,
                        "to": 200,
                        "increment": 1,
                        "token": "meta{}",
                        "display_token": "meta={}",
                    },
                },
            },
            "som_louvain": {
                "label": "SOM + Louvain",
                "selected": tk.BooleanVar(value=False),
                "params": {
                    "grid_x": {
                        "label": "Grid X",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 2,
                        "to": 50,
                        "increment": 1,
                        "display_token": "grid_x={}",
                    },
                    "grid_y": {
                        "label": "Grid Y",
                        "var": tk.StringVar(value="10"),
                        "type": "int",
                        "from": 2,
                        "to": 50,
                        "increment": 1,
                        "display_token": "grid_y={}",
                    },
                    "n_neighbors": {
                        "label": "Neighbors",
                        "var": tk.StringVar(value="15"),
                        "type": "int",
                        "from": 5,
                        "to": 200,
                        "increment": 1,
                        "token": "nn{}",
                        "display_token": "nn={}",
                    },
                    "resolution": {
                        "label": "Resolution",
                        "var": tk.StringVar(value="1.0"),
                        "type": "float",
                        "from": 0.1,
                        "to": 5.0,
                        "increment": 0.1,
                        "token": "res{}",
                        "display_token": "res={}",
                    },
                },
            },
        }
        self.clustering_available_columns: List[str] = []
        self.clustering_selection: List[str] = []
        self.clustering_hint_var = tk.StringVar(
            value="Load files to populate clustering features."
        )
        self.clustering_missing_var = tk.StringVar(value="")
        self.clustering_total_rows_var = tk.StringVar(value="Total cells: 0")
        self.clustering_class_var = tk.StringVar()
        self.clustering_subset_name_var = tk.StringVar(value="")
        self.clustering_subset_rows_var = tk.StringVar(value="Filtered rows: all")
        self.clustering_filters: List[Dict[str, Any]] = []
        self.clustering_filter_counter = 0
        self.clustering_filter_column_var = tk.StringVar()
        self.clustering_filter_min_var = tk.StringVar()
        self.clustering_filter_max_var = tk.StringVar()
        self.clustering_categorical_options: Dict[str, List[Any]] = {}
        self.clustering_filter_current_values: List[Any] = []
        self.clustering_filter_values_label_var = tk.StringVar(value="Select categories")
        self.clustering_numeric_ranges: Dict[str, Dict[str, float]] = {}
        self.clustering_filter_mode_var = tk.StringVar(value="AND")
        self.clustering_downsample_method_var = tk.StringVar(value="None")
        self.clustering_downsample_value_var = tk.StringVar(value="")
        self.clustering_downsample_value_label_var = tk.StringVar(value="Target size")
        self.clustering_downsample_message_var = tk.StringVar(
            value="Configure a downsampling strategy to preview its effect."
        )
        self.clustering_downsampled_df: Optional[pd.DataFrame] = None
        self.clustering_method_param_widgets: Dict[str, List[tk.Widget]] = {}
        self.clustering_queue: "queue.Queue[dict]" = queue.Queue()
        self.clustering_thread: Optional[Thread] = None
        self.clustering_in_progress = False
        self.clustering_results: Dict[str, pd.DataFrame] = {}
        self.clustering_status_var = tk.StringVar(value="Clustering not started.")
        self.clustering_n_jobs_var = tk.IntVar(value=default_jobs)
        self.clustering_features_used: List[str] = []
        self.clustering_dataset_cache: Optional[pd.DataFrame] = None
        self.clustering_feature_matrix: Optional[np.ndarray] = None
        self.base_clustering_method_labels = {
            key: cfg["label"] for key, cfg in self.clustering_methods.items()
        }
        self.clustering_method_labels = dict(self.base_clustering_method_labels)
        self.clustering_umap_cache: Dict[tuple, np.ndarray] = {}
        self.clustering_umap_colorbar = None
        self.clustering_umap_show_labels_var = tk.BooleanVar(value=False)
        self.clustering_umap_downsample_var = tk.BooleanVar(value=False)
        self.clustering_umap_max_cells_var = tk.IntVar(value=20000)
        self.clustering_umap_jobs_var = tk.IntVar(value=max(1, min(self.total_cpu_cores, 4)))
        self.clustering_umap_dot_size_var = tk.DoubleVar(value=6.0)
        self.clustering_umap_alpha_var = tk.DoubleVar(value=0.8)
        self.clustering_umap_marker_var = tk.StringVar()
        self.clustering_heatmap_colorbar = None
        self.clustering_heatmap_cluster_dendro_var = tk.BooleanVar(value=False)
        self.clustering_heatmap_marker_dendro_var = tk.BooleanVar(value=False)
        self.clustering_metadata: Dict[str, Dict[str, object]] = {}
        self.clustering_run_metadata_base: Dict[str, object] = {}
        self.cluster_explorer_method_var = tk.StringVar()
        self.cluster_explorer_plots: List[Dict[str, Any]] = []
        self.cluster_explorer_feature_options: List[str] = []
        self.cluster_explorer_cluster_listbox: Optional[tk.Listbox] = None
        self.cluster_explorer_select_all_button: Optional[ttk.Button] = None
        self.cluster_explorer_clear_button: Optional[ttk.Button] = None
        self.cluster_explorer_status_var = tk.StringVar(value="Load clustering results to explore clusters.")
        self.cluster_explorer_sample_limit = 20000
        self.cluster_explorer_dot_size_var = tk.DoubleVar(value=12.0)
        self.cluster_explorer_alpha_var = tk.DoubleVar(value=0.8)
        self.cluster_explorer_feature_menu: Optional[tk.Menu] = None
        self.pending_clustering_labels: Dict[str, str] = {}
        self.cluster_annotation_method_var = tk.StringVar()
        self.cluster_annotations: Dict[str, pd.DataFrame] = {}
        self.cluster_annotation_recent_terms: Dict[str, Set[str]] = defaultdict(set)
        self.current_annotation_run_key: Optional[str] = None
        self.cluster_annotation_info_var = tk.StringVar(
            value="Select a clustering run to begin annotating clusters."
        )
        self.cluster_annotation_status_var = tk.StringVar(value="")
        self.annotation_tree: Optional[ttk.Treeview] = None
        self.current_annotation_columns: List[str] = ["cluster"]
        self.annotation_edit_widget: Optional[ttk.Combobox] = None
        self.annotation_edit_info: Optional[Dict[str, Any]] = None
        self.cluster_compare_method_var = tk.StringVar()
        self.cluster_compare_category_var = tk.StringVar()
        self.cluster_compare_category_map: Dict[str, tuple[str, str]] = {}
        self.cluster_compare_results: List[Dict[str, object]] = []
        self.cluster_compare_result_counter = 0

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_controls()
        self._build_summary()
        self._build_notebook()
        self._build_status_bar()

        self._load_run_registry()
        self.root.after(400, self._load_session_state)
        self._reset_results_view()

    def _build_controls(self) -> None:
        control_frame = ttk.Frame(self.root, padding=(12, 8))
        control_frame.pack(fill="x")

        ttk.Button(
            control_frame,
            text="Select CSV Files",
            command=self.select_files,
        ).pack(side="left")

        self.files_loaded_var = tk.StringVar(value="No files loaded")
        ttk.Label(control_frame, textvariable=self.files_loaded_var).pack(
            side="left", padx=(12, 0)
        )

    def _build_summary(self) -> None:
        summary_frame = ttk.Frame(self.root, padding=(12, 0))
        summary_frame.pack(fill="x")

        self.total_files_var = tk.StringVar(value="Files: 0")
        self.total_rows_var = tk.StringVar(value="Total cells (rows): 0")
        self.total_columns_var = tk.StringVar(value="Unique columns: 0")

        ttk.Label(summary_frame, textvariable=self.total_files_var).pack(
            side="left", padx=(0, 24)
        )
        ttk.Label(summary_frame, textvariable=self.total_rows_var).pack(
            side="left", padx=(0, 24)
        )
        ttk.Label(summary_frame, textvariable=self.total_columns_var).pack(
            side="left"
        )

    def _build_notebook(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        self.notebook = notebook

        # Data module with Files / Columns
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Data Overview")
        data_notebook = ttk.Notebook(data_tab)
        data_notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self._init_files_tab(data_notebook)
        self._init_columns_tab(data_notebook)
        self._init_quality_tab(data_notebook)

        # Training module with setup/run/results/visuals
        training_tab = ttk.Frame(notebook)
        notebook.add(training_tab, text="Training Module")
        training_notebook = ttk.Notebook(training_tab)
        training_notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self._init_training_setup_tab(training_notebook)
        self._init_training_module_tab(training_notebook)
        self._init_training_visuals_tab(training_notebook)
        self._init_results_tab(training_notebook)
        self._init_run_registry_tab(training_notebook)

        # Clustering module with setup/results/visualization
        clustering_tab = ttk.Frame(notebook)
        notebook.add(clustering_tab, text="Clustering Module")
        clustering_notebook = ttk.Notebook(clustering_tab)
        clustering_notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self._init_clustering_setup_tab(clustering_notebook)
        self._init_clustering_results_tab(clustering_notebook)
        self._init_clustering_visuals_tab(clustering_notebook)
        self._init_clustering_explorer_tab(clustering_notebook)
        self._init_clustering_annotation_tab(clustering_notebook)
        self._init_clustering_comparison_tab(clustering_notebook)

    def _init_files_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Files")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        files_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        files_tab.pack(fill="both", expand=True)

        columns = ("file", "rows", "columns")
        tree = ttk.Treeview(
            files_tab,
            columns=columns,
            show="headings",
            height=18,
        )
        tree.heading("file", text="File")
        tree.heading("rows", text="Rows")
        tree.heading("columns", text="Columns")

        tree.column("file", width=400, anchor="w")
        tree.column("rows", width=120, anchor="center")
        tree.column("columns", width=120, anchor="center")

        tree.pack(fill="both", expand=True)
        self.files_tree = tree

        scrollbar = ttk.Scrollbar(
            files_tab,
            orient="vertical",
            command=self.files_tree.yview,
        )
        self.files_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        ttk.Button(
            files_tab,
            text="Remove Selected Files",
            command=self.remove_selected_files,
        ).pack(anchor="w", pady=(8, 0))

    def _init_columns_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Columns")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        columns_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        columns_tab.pack(fill="both", expand=True)

        top_frame = ttk.Frame(columns_tab)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Column overview across files").pack(
            anchor="w"
        )

        tree = ttk.Treeview(
            columns_tab,
            columns=("column", "files_present"),
            show="headings",
            height=18,
        )
        tree.heading("column", text="Column")
        tree.heading("files_present", text="Files containing column")
        tree.column("column", width=400, anchor="w")
        tree.column("files_present", width=200, anchor="center")
        tree.pack(fill="both", expand=True, pady=(8, 0))
        self.columns_tree = tree

        column_scrollbar = ttk.Scrollbar(
            columns_tab, orient="vertical", command=self.columns_tree.yview
        )
        self.columns_tree.configure(yscroll=column_scrollbar.set)
        column_scrollbar.pack(side="right", fill="y")

    def _init_quality_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Data Quality")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        quality_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        quality_tab.pack(fill="both", expand=True)

        columns = (
            "column",
            "dtype",
            "non_null",
            "missing",
            "unique",
            "min",
            "max",
            "notes",
        )
        tree = ttk.Treeview(
            quality_tab,
            columns=columns,
            show="headings",
            height=16,
        )
        for name, label, width, anchor in [
            ("column", "Column", 220, "w"),
            ("dtype", "Type", 90, "center"),
            ("non_null", "Non-null %", 100, "center"),
            ("missing", "Missing %", 100, "center"),
            ("unique", "Unique", 100, "center"),
            ("min", "Min", 140, "w"),
            ("max", "Max", 140, "w"),
            ("notes", "Notes", 260, "w"),
        ]:
            tree.heading(name, text=label)
            tree.column(name, width=width, anchor=anchor)
        tree.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            quality_tab,
            orient="vertical",
            command=tree.yview,
        )
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.quality_tree = tree

        summary_frame = ttk.LabelFrame(quality_tab, text="Summary", padding=8)
        summary_frame.pack(fill="both", expand=False, pady=(12, 0))
        summary_text = tk.Text(
            summary_frame,
            height=6,
            wrap="word",
            state="disabled",
            background=self.root.cget("background"),
            relief="flat",
        )
        summary_text.pack(fill="both", expand=True)
        self.quality_summary_text = summary_text

    def _init_training_setup_tab(self, notebook: ttk.Notebook) -> None:
        setup_container = ttk.Frame(notebook)
        notebook.add(setup_container, text="Setup")
        setup_scroll = ScrollableFrame(setup_container)
        setup_scroll.pack(fill="both", expand=True)
        setup_tab = ttk.Frame(setup_scroll.scrollable_frame, padding=12)
        setup_tab.pack(fill="both", expand=True)

        feature_frame = ttk.LabelFrame(setup_tab, text="Feature Columns", padding=12)
        feature_frame.pack(fill="x", expand=False)

        ttk.Label(
            feature_frame,
            text=(
                "Select the columns to use as features during training. Columns present in all files are shown in black; "
                "columns missing in at least one file appear in red."
            ),
            wraplength=650,
            justify="left",
        ).pack(anchor="w")

        listbox_frame = ttk.Frame(feature_frame)
        listbox_frame.pack(fill="x", pady=(8, 0))

        self.training_listbox = tk.Listbox(
            listbox_frame,
            listvariable=tk.StringVar(value=[]),
            selectmode="extended",
            exportselection=False,
            height=10,
        )
        self.training_listbox.pack(side="left", fill="both", expand=True)
        self.training_listbox.bind(
            "<<ListboxSelect>>", lambda _evt: self._on_training_selection_changed()
        )
        self.training_listbox.bind("<Control-Button-1>", self._toggle_training_listbox_selection)
        self.training_listbox.bind("<Command-Button-1>", self._toggle_training_listbox_selection)

        feature_scroll = ttk.Scrollbar(
            listbox_frame,
            orient="vertical",
            command=self.training_listbox.yview,
        )
        feature_scroll.pack(side="right", fill="y")
        self.training_listbox.configure(yscrollcommand=feature_scroll.set)

        button_row = ttk.Frame(feature_frame)
        button_row.pack(anchor="w", pady=(8, 0))
        ttk.Button(
            button_row,
            text="Select All Common",
            command=self._select_all_common_features,
        ).pack(side="left")
        ttk.Button(
            button_row,
            text="Clear Selection",
            command=lambda: self.training_listbox.selection_clear(0, tk.END),
        ).pack(side="left", padx=(8, 0))

        ttk.Label(feature_frame, textvariable=self.training_hint_var).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            feature_frame,
            textvariable=self.training_missing_var,
            foreground="red",
        ).pack(anchor="w")

        target_frame = ttk.LabelFrame(setup_tab, text="Classification Target", padding=12)
        target_frame.pack(fill="both", expand=True, pady=(12, 0))

        ttk.Label(
            target_frame,
            text="Choose the categorical column to use as the target for classification.",
        ).pack(anchor="w")

        self.target_combo = ttk.Combobox(
            target_frame,
            textvariable=self.target_column_var,
            state="readonly",
            values=[],
        )
        self.target_combo.pack(anchor="w", pady=(6, 0), fill="x")
        self.target_combo.bind("<<ComboboxSelected>>", self.on_target_column_selected)

        ttk.Label(target_frame, textvariable=self.target_info_var, wraplength=650).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            target_frame,
            textvariable=self.target_missing_var,
            foreground="red",
            wraplength=650,
        ).pack(anchor="w")

        totals_frame = ttk.Frame(target_frame)
        totals_frame.pack(fill="both", expand=True, pady=(8, 0))

        totals_left = ttk.Frame(totals_frame)
        totals_left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(totals_left, text="Overall category counts").pack(anchor="w")

        self.category_totals_tree = ttk.Treeview(
            totals_left,
            columns=("category", "count"),
            show="headings",
            height=6,
        )
        self.category_totals_tree.heading("category", text="Category")
        self.category_totals_tree.heading("count", text="Count")
        self.category_totals_tree.column("category", anchor="w", width=180)
        self.category_totals_tree.column("count", anchor="center", width=100)
        self.category_totals_tree.pack(fill="both", expand=True, pady=(4, 0))

        totals_scroll = ttk.Scrollbar(
            totals_left, orient="vertical", command=self.category_totals_tree.yview
        )
        self.category_totals_tree.configure(yscroll=totals_scroll.set)
        totals_scroll.pack(side="right", fill="y")

        totals_right = ttk.Frame(totals_frame)
        totals_right.pack(side="left", fill="both", expand=True)
        ttk.Label(totals_right, text="Per-file category counts").pack(anchor="w")

        self.category_per_file_tree = ttk.Treeview(
            totals_right,
            columns=("file", "category", "count"),
            show="headings",
            height=6,
        )
        self.category_per_file_tree.heading("file", text="File")
        self.category_per_file_tree.heading("category", text="Category")
        self.category_per_file_tree.heading("count", text="Count")
        self.category_per_file_tree.column("file", anchor="w", width=200)
        self.category_per_file_tree.column("category", anchor="w", width=140)
        self.category_per_file_tree.column("count", anchor="center", width=80)
        self.category_per_file_tree.pack(fill="both", expand=True, pady=(4, 0))

        per_file_scroll = ttk.Scrollbar(
            totals_right,
            orient="vertical",
            command=self.category_per_file_tree.yview,
        )
        self.category_per_file_tree.configure(yscroll=per_file_scroll.set)
        per_file_scroll.pack(side="right", fill="y")

    def _init_training_module_tab(self, notebook: ttk.Notebook) -> None:
        module_container = ttk.Frame(notebook)
        notebook.add(module_container, text="Training")
        module_scroll = ScrollableFrame(module_container)
        module_scroll.pack(fill="both", expand=True)
        module_tab = ttk.Frame(module_scroll.scrollable_frame, padding=12)
        module_tab.pack(fill="both", expand=True)

        downsample_frame = ttk.LabelFrame(module_tab, text="Downsampling", padding=12)
        downsample_frame.pack(fill="both", expand=False, pady=(0, 12))

        controls_row = ttk.Frame(downsample_frame)
        controls_row.pack(fill="x")

        ttk.Label(controls_row, text="Method").pack(side="left")
        downsample_methods = [
            "None",
            "Total Count",
            "Per File",
            "Per Class",
            "Per File + Class",
        ]
        self.training_downsample_method_combo = ttk.Combobox(
            controls_row,
            state="readonly",
            textvariable=self.training_downsample_method_var,
            values=downsample_methods,
            width=18,
        )
        self.training_downsample_method_combo.pack(side="left", padx=(6, 12))
        self.training_downsample_method_combo.bind(
            "<<ComboboxSelected>>",
            lambda _evt: self._on_training_downsample_method_changed(),
        )

        self.training_downsample_value_label = ttk.Label(
            controls_row, textvariable=self.training_downsample_value_label_var
        )
        self.training_downsample_value_label.pack(side="left")

        self.training_downsample_value_entry = ttk.Entry(
            controls_row, textvariable=self.training_downsample_value_var, width=12
        )
        self.training_downsample_value_entry.pack(side="left", padx=(6, 12))

        ttk.Button(
            controls_row,
            text="Preview Downsampling",
            command=self._training_preview_downsampling,
        ).pack(side="left")

        ttk.Label(
            downsample_frame,
            textvariable=self.training_downsample_message_var,
            wraplength=650,
        ).pack(anchor="w", pady=(10, 4))

        tree_frame = ttk.Frame(downsample_frame)
        tree_frame.pack(fill="both", expand=True)

        self.training_downsample_file_tree = ttk.Treeview(
            tree_frame,
            columns=("file", "rows"),
            show="headings",
            height=6,
        )
        self.training_downsample_file_tree.heading("file", text="File")
        self.training_downsample_file_tree.heading("rows", text="Rows")
        self.training_downsample_file_tree.column("file", anchor="w", width=220)
        self.training_downsample_file_tree.column("rows", anchor="center", width=100)
        self.training_downsample_file_tree.pack(side="left", fill="both", expand=True)

        downsample_file_scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.training_downsample_file_tree.yview,
        )
        self.training_downsample_file_tree.configure(yscroll=downsample_file_scroll.set)
        downsample_file_scroll.pack(side="left", fill="y")

        self.training_downsample_category_tree = ttk.Treeview(
            tree_frame,
            columns=("category", "rows"),
            show="headings",
            height=6,
        )
        self.training_downsample_category_tree.heading("category", text="Category")
        self.training_downsample_category_tree.heading("rows", text="Rows")
        self.training_downsample_category_tree.column("category", anchor="w", width=140)
        self.training_downsample_category_tree.column("rows", anchor="center", width=80)
        self.training_downsample_category_tree.pack(
            side="left", fill="both", expand=True, padx=(12, 0)
        )

        downsample_category_scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.training_downsample_category_tree.yview,
        )
        self.training_downsample_category_tree.configure(
            yscroll=downsample_category_scroll.set
        )
        downsample_category_scroll.pack(side="left", fill="y")

        training_frame = ttk.LabelFrame(
            module_tab, text="Model Training", padding=12
        )
        training_frame.pack(fill="both", expand=True)

        ttk.Label(
            training_frame,
            text=(
                "Select a model family, configure its hyperparameters, and adjust evaluation settings before starting training."
            ),
            wraplength=700,
            justify="left",
        ).pack(anchor="w")

        model_select = ttk.Frame(training_frame)
        model_select.pack(fill="x", pady=(10, 4))
        ttk.Label(model_select, text="Model type").grid(row=0, column=0, sticky="w")
        self.training_model_combo = ttk.Combobox(
            model_select,
            state="readonly",
            values=list(self.training_model_configs.keys()),
            textvariable=self.training_model_var,
            width=24,
        )
        self.training_model_combo.grid(row=1, column=0, sticky="w")
        self.training_model_combo.bind(
            "<<ComboboxSelected>>", self._handle_training_model_selected
        )

        ttk.Label(
            training_frame,
            textvariable=self.training_model_description_var,
            wraplength=700,
            justify="left",
            foreground="#444444",
        ).pack(anchor="w", pady=(4, 6))

        self.training_model_param_container = ttk.Frame(training_frame)
        self.training_model_param_container.pack(fill="x", pady=(4, 0))
        self.training_model_param_frames: Dict[str, ttk.Frame] = {}
        self._build_training_model_frames()

        eval_frame = ttk.LabelFrame(
            training_frame, text="Evaluation Settings", padding=8
        )
        eval_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(eval_frame, text="Test split (fraction)").grid(row=0, column=0, sticky="w")
        self.test_size_combo = ttk.Combobox(
            eval_frame,
            state="readonly",
            values=["0.1", "0.15", "0.2", "0.25", "0.3"],
            textvariable=self.test_size_var,
            width=8,
        )
        self.test_size_combo.grid(row=1, column=0, sticky="w")

        ttk.Label(eval_frame, text="CV folds").grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        self.cv_folds_spin = ttk.Spinbox(
            eval_frame,
            from_=2,
            to=10,
            increment=1,
            textvariable=self.cv_folds_var,
            width=6,
        )
        self.cv_folds_spin.grid(row=1, column=1, sticky="w", padx=(12, 0))

        ttk.Label(eval_frame, text="CPU cores").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        self.training_n_jobs_spin = ttk.Spinbox(
            eval_frame,
            from_=1,
            to=self.total_cpu_cores,
            increment=1,
            textvariable=self.n_jobs_var,
            width=6,
        )
        self.training_n_jobs_spin.grid(row=1, column=2, sticky="w", padx=(12, 0))

        ttk.Label(eval_frame, text="Class balance").grid(
            row=0, column=3, sticky="w", padx=(12, 0)
        )
        ttk.Combobox(
            eval_frame,
            state="readonly",
            values=["None", "Balanced"],
            textvariable=self.class_balance_var,
            width=12,
        ).grid(row=1, column=3, sticky="w", padx=(12, 0))

        run_meta_frame = ttk.Frame(training_frame)
        run_meta_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(run_meta_frame, text="Run tags (comma-separated)").grid(row=0, column=0, sticky="w")
        ttk.Entry(run_meta_frame, textvariable=self.run_tags_var, width=40).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(run_meta_frame, text="Notes").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Entry(run_meta_frame, textvariable=self.run_notes_var, width=50).grid(
            row=1, column=1, sticky="w", padx=(12, 0)
        )

        save_frame = ttk.LabelFrame(
            training_frame, text="Save Options", padding=8
        )
        save_frame.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(
            save_frame,
            text="Keep Random Forest OOB caches when saving",
            variable=self.keep_rf_oob_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            save_frame,
            text="Uncheck to strip out-of-bag prediction matrices from Random Forest models before exporting to shrink file size.",
            wraplength=680,
            justify="left",
            foreground="#555555",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls_frame = ttk.Frame(training_frame)
        controls_frame.pack(fill="x", pady=(12, 0))

        self.train_button = ttk.Button(
            controls_frame,
            text="Train Model",
            command=self.start_training,
        )
        self.train_button.pack(side="left")

        self.training_progress = ttk.Progressbar(
            controls_frame,
            mode="indeterminate",
            length=220,
        )
        self.training_progress.pack(side="left", padx=(12, 0))
        self.test_size_combo.set(str(self.test_size_var.get()))

        ttk.Label(
            training_frame,
            text=(
                "Training uses the downsampled dataset when available, otherwise the full combined data."
            ),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(12, 0))
        self.training_downsample_method_combo.current(0)
        self._on_training_downsample_method_changed()
        self._on_training_model_changed()

    def _build_training_model_frames(self) -> None:
        for child in self.training_model_param_container.winfo_children():
            child.destroy()
        self.training_model_param_frames.clear()

        # Random Forest
        rf_frame = ttk.LabelFrame(
            self.training_model_param_container, text="Random Forest Parameters", padding=8
        )
        ttk.Label(rf_frame, text="Trees (n_estimators)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            rf_frame,
            from_=50,
            to=2000,
            increment=50,
            textvariable=self.n_estimators_var,
            width=8,
        ).grid(row=1, column=0, sticky="w")

        ttk.Label(rf_frame, text="Max depth (blank=auto)").grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        ttk.Entry(
            rf_frame,
            textvariable=self.max_depth_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))

        ttk.Label(rf_frame, text="Max features (mtry)").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        ttk.Combobox(
            rf_frame,
            state="readonly",
            values=["auto", "sqrt", "log2", "None"],
            textvariable=self.max_features_var,
            width=8,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))

        ttk.Label(rf_frame, text="Min samples per leaf").grid(
            row=0, column=3, sticky="w", padx=(12, 0)
        )
        ttk.Spinbox(
            rf_frame,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.min_samples_leaf_var,
            width=6,
        ).grid(row=1, column=3, sticky="w", padx=(12, 0))
        self.training_model_param_frames["Random Forest"] = rf_frame

        # LDA
        lda_frame = ttk.LabelFrame(
            self.training_model_param_container, text="LDA Parameters", padding=8
        )
        ttk.Label(lda_frame, text="Solver").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            lda_frame,
            state="readonly",
            values=["svd", "lsqr", "eigen"],
            textvariable=self.lda_solver_var,
            width=10,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(lda_frame, text="Shrinkage (blank/auto/value)").grid(
            row=0, column=1, sticky="w", padx=(12, 0)
        )
        ttk.Entry(
            lda_frame,
            textvariable=self.lda_shrinkage_var,
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))
        self.training_model_param_frames["LDA"] = lda_frame

        # SVM
        svm_frame = ttk.LabelFrame(
            self.training_model_param_container, text="SVM Parameters", padding=8
        )
        ttk.Label(svm_frame, text="Kernel").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            svm_frame,
            state="readonly",
            values=["rbf", "linear", "poly"],
            textvariable=self.svm_kernel_var,
            width=10,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(svm_frame, text="C").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            svm_frame,
            from_=0.01,
            to=1000,
            increment=0.01,
            textvariable=self.svm_c_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(svm_frame, text="Gamma (scale/auto/value)").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        ttk.Entry(
            svm_frame,
            textvariable=self.svm_gamma_var,
            width=10,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Label(svm_frame, text="Degree (poly)").grid(
            row=0, column=3, sticky="w", padx=(12, 0)
        )
        ttk.Spinbox(
            svm_frame,
            from_=2,
            to=10,
            increment=1,
            textvariable=self.svm_degree_var,
            width=6,
        ).grid(row=1, column=3, sticky="w", padx=(12, 0))
        self.training_model_param_frames["SVM"] = svm_frame

        # Logistic Regression
        lr_frame = ttk.LabelFrame(
            self.training_model_param_container,
            text="Logistic Regression Parameters",
            padding=8,
        )
        ttk.Label(lr_frame, text="Solver").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            lr_frame,
            state="readonly",
            values=["lbfgs", "saga", "liblinear"],
            textvariable=self.lr_solver_var,
            width=10,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(lr_frame, text="Penalty").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Combobox(
            lr_frame,
            state="readonly",
            values=["l2", "l1", "elasticnet"],
            textvariable=self.lr_penalty_var,
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(lr_frame, text="C (regularization)").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            lr_frame,
            from_=0.0001,
            to=1000.0,
            increment=0.1,
            textvariable=self.lr_c_var,
            width=10,
            format="%.4f",
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Label(lr_frame, text="Max iter").grid(row=0, column=3, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            lr_frame,
            from_=100,
            to=2000,
            increment=50,
            textvariable=self.lr_max_iter_var,
            width=8,
        ).grid(row=1, column=3, sticky="w", padx=(12, 0))
        ttk.Label(lr_frame, text="L1 ratio (elasticnet)").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            lr_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.lr_l1_ratio_var,
            width=8,
            format="%.2f",
        ).grid(row=1, column=4, sticky="w", padx=(12, 0))
        self.training_model_param_frames["Logistic Regression"] = lr_frame

        # Naive Bayes
        nb_frame = ttk.LabelFrame(
            self.training_model_param_container,
            text="Naive Bayes Parameters",
            padding=8,
        )
        ttk.Label(nb_frame, text="Var smoothing").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            nb_frame,
            from_=1e-12,
            to=1e-3,
            increment=1e-4,
            textvariable=self.nb_var_smoothing_var,
            width=12,
            format="%.5e",
        ).grid(row=1, column=0, sticky="w")
        self.training_model_param_frames["Naive Bayes"] = nb_frame

        # XGBoost
        xgb_frame = ttk.LabelFrame(
            self.training_model_param_container,
            text="XGBoost Parameters",
            padding=8,
        )
        if XGBClassifier is None:
            ttk.Label(
                xgb_frame,
                text="Install the 'xgboost' package to enable this model.",
                foreground="red",
            ).grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(xgb_frame, text="Trees").grid(row=0, column=0, sticky="w")
            ttk.Spinbox(
                xgb_frame,
                from_=50,
                to=2000,
                increment=50,
                textvariable=self.xgb_estimators_var,
                width=8,
            ).grid(row=1, column=0, sticky="w")
            ttk.Label(xgb_frame, text="Learning rate").grid(row=0, column=1, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                xgb_frame,
                from_=0.001,
                to=1.0,
                increment=0.01,
                textvariable=self.xgb_learning_rate_var,
                width=8,
                format="%.3f",
            ).grid(row=1, column=1, sticky="w", padx=(12, 0))
            ttk.Label(xgb_frame, text="Max depth").grid(row=0, column=2, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                xgb_frame,
                from_=1,
                to=15,
                increment=1,
                textvariable=self.xgb_max_depth_var,
                width=6,
            ).grid(row=1, column=2, sticky="w", padx=(12, 0))
            ttk.Label(xgb_frame, text="Subsample").grid(row=0, column=3, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                xgb_frame,
                from_=0.5,
                to=1.0,
                increment=0.05,
                textvariable=self.xgb_subsample_var,
                width=6,
                format="%.2f",
            ).grid(row=1, column=3, sticky="w", padx=(12, 0))
            ttk.Label(xgb_frame, text="Colsample by tree").grid(row=0, column=4, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                xgb_frame,
                from_=0.5,
                to=1.0,
                increment=0.05,
                textvariable=self.xgb_colsample_var,
                width=6,
                format="%.2f",
            ).grid(row=1, column=4, sticky="w", padx=(12, 0))
        self.training_model_param_frames["XGBoost"] = xgb_frame

        # LightGBM
        lgb_frame = ttk.LabelFrame(
            self.training_model_param_container,
            text="LightGBM Parameters",
            padding=8,
        )
        if lgb is None:
            ttk.Label(
                lgb_frame,
                text="Install the 'lightgbm' package to enable this model.",
                foreground="red",
            ).grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(lgb_frame, text="Trees").grid(row=0, column=0, sticky="w")
            ttk.Spinbox(
                lgb_frame,
                from_=50,
                to=4000,
                increment=50,
                textvariable=self.lgb_estimators_var,
                width=8,
            ).grid(row=1, column=0, sticky="w")
            ttk.Label(lgb_frame, text="Learning rate").grid(row=0, column=1, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                lgb_frame,
                from_=0.001,
                to=1.0,
                increment=0.01,
                textvariable=self.lgb_learning_rate_var,
                width=8,
                format="%.3f",
            ).grid(row=1, column=1, sticky="w", padx=(12, 0))
            ttk.Label(lgb_frame, text="Max depth (-1=auto)").grid(row=0, column=2, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                lgb_frame,
                from_=-1,
                to=32,
                increment=1,
                textvariable=self.lgb_max_depth_var,
                width=6,
            ).grid(row=1, column=2, sticky="w", padx=(12, 0))
            ttk.Label(lgb_frame, text="Num leaves").grid(row=0, column=3, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                lgb_frame,
                from_=8,
                to=256,
                increment=1,
                textvariable=self.lgb_num_leaves_var,
                width=6,
            ).grid(row=1, column=3, sticky="w", padx=(12, 0))
            ttk.Label(lgb_frame, text="Subsample").grid(row=0, column=4, sticky="w", padx=(12, 0))
            ttk.Spinbox(
                lgb_frame,
                from_=0.5,
                to=1.0,
                increment=0.05,
                textvariable=self.lgb_subsample_var,
                width=6,
                format="%.2f",
            ).grid(row=1, column=4, sticky="w", padx=(12, 0))
        self.training_model_param_frames["LightGBM"] = lgb_frame

        # KMeans
        km_frame = ttk.LabelFrame(
            self.training_model_param_container, text="KMeans Parameters", padding=8
        )
        ttk.Label(km_frame, text="Clusters (k)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            km_frame,
            from_=2,
            to=200,
            increment=1,
            textvariable=self.kmeans_clusters_var,
            width=8,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(km_frame, text="Max iterations").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            km_frame,
            from_=10,
            to=1000,
            increment=10,
            textvariable=self.kmeans_max_iter_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(km_frame, text="Initializations").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            km_frame,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.kmeans_n_init_var,
            width=8,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))
        self.training_model_param_frames["KMeans"] = km_frame

        # Neural Network
        nn_frame = ttk.LabelFrame(
            self.training_model_param_container, text="Neural Network Parameters", padding=8
        )
        ttk.Label(nn_frame, text="Hidden layers (comma-separated)").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            nn_frame,
            textvariable=self.nn_hidden_layers_var,
            width=18,
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(nn_frame, text="Activation").grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Combobox(
            nn_frame,
            state="readonly",
            values=["relu", "tanh"],
            textvariable=self.nn_activation_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=(12, 0))
        ttk.Label(nn_frame, text="Dropout").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(
            nn_frame,
            from_=0.0,
            to=0.9,
            increment=0.05,
            textvariable=self.nn_dropout_var,
            width=6,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Label(nn_frame, text="Epochs").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            nn_frame,
            from_=5,
            to=500,
            increment=5,
            textvariable=self.nn_epochs_var,
            width=8,
        ).grid(row=3, column=0, sticky="w")
        ttk.Label(nn_frame, text="Batch size").grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(
            nn_frame,
            from_=8,
            to=512,
            increment=8,
            textvariable=self.nn_batch_var,
            width=8,
        ).grid(row=3, column=1, sticky="w", padx=(12, 0))
        ttk.Label(nn_frame, text="Learning rate").grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(
            nn_frame,
            from_=0.0001,
            to=0.1,
            increment=0.0001,
            textvariable=self.nn_lr_var,
            width=8,
        ).grid(row=3, column=2, sticky="w", padx=(12, 0))
        ttk.Label(nn_frame, text="Weight decay").grid(row=2, column=3, sticky="w", padx=(12, 0), pady=(8, 0))
        ttk.Spinbox(
            nn_frame,
            from_=0.0,
            to=0.1,
            increment=0.0005,
            textvariable=self.nn_weight_decay_var,
            width=8,
        ).grid(row=3, column=3, sticky="w", padx=(12, 0))
        ttk.Checkbutton(
            nn_frame,
            text="Use GPU (if available)",
            variable=self.nn_use_gpu_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.training_model_param_frames["Neural Network"] = nn_frame

    def _on_training_model_changed(self) -> None:
        selected = self.training_model_var.get()
        desc = self.training_model_configs.get(selected, {}).get("description", "")
        self.training_model_description_var.set(desc)
        for name, frame in self.training_model_param_frames.items():
            frame.pack_forget()
        frame = self.training_model_param_frames.get(selected)
        if frame is not None:
            frame.pack(fill="x", pady=(4, 0))

    def _handle_training_model_selected(self, _event: Optional[tk.Event] = None) -> None:
        self._on_training_model_changed()
        self._mark_session_dirty()

    def _init_clustering_setup_tab(self, notebook: ttk.Notebook) -> None:
        module_container = ttk.Frame(notebook)
        notebook.add(module_container, text="Setup")
        module_scroll = ScrollableFrame(module_container)
        module_scroll.pack(fill="both", expand=True)
        module_tab = ttk.Frame(module_scroll.scrollable_frame, padding=12)
        module_tab.pack(fill="both", expand=True)

        overview_frame = ttk.LabelFrame(module_tab, text="Dataset Overview", padding=12)
        overview_frame.pack(fill="x", expand=False)

        ttk.Label(
            overview_frame,
            textvariable=self.clustering_total_rows_var,
        ).pack(anchor="w")

        ttk.Label(
            overview_frame,
            text=(
                "Optional: select a categorical column to preserve class balance during downsampling."
            ),
            wraplength=650,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self.clustering_class_combo = ttk.Combobox(
            overview_frame,
            textvariable=self.clustering_class_var,
            state="readonly",
            values=[],
        )
        self.clustering_class_combo.pack(anchor="w", pady=(4, 0), fill="x")

        subset_frame = ttk.LabelFrame(module_tab, text="Subsetting (optional)", padding=12)
        subset_frame.pack(fill="x", expand=False, pady=(12, 0))

        ttk.Label(subset_frame, text="Subset name").grid(row=0, column=0, sticky="w")
        ttk.Entry(
            subset_frame,
            textvariable=self.clustering_subset_name_var,
            width=30,
        ).grid(row=0, column=1, sticky="w", padx=(6, 12))
        ttk.Label(
            subset_frame,
            textvariable=self.clustering_subset_rows_var,
        ).grid(row=0, column=2, sticky="w")
        logic_frame = ttk.LabelFrame(subset_frame, text="Filter logic", padding=(6, 4))
        logic_frame.grid(row=0, column=3, rowspan=2, sticky="nw", padx=(12, 0))
        ttk.Radiobutton(
            logic_frame,
            text="Match ALL (AND)",
            value="AND",
            variable=self.clustering_filter_mode_var,
            command=self._on_clustering_filter_logic_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            logic_frame,
            text="Match ANY (OR)",
            value="OR",
            variable=self.clustering_filter_mode_var,
            command=self._on_clustering_filter_logic_changed,
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(subset_frame, text="Filter column").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.clustering_filter_column_combo = ttk.Combobox(
            subset_frame,
            textvariable=self.clustering_filter_column_var,
            state="readonly",
            values=[],
            width=28,
        )
        self.clustering_filter_column_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.clustering_filter_column_combo.bind(
            "<<ComboboxSelected>>", self._on_clustering_filter_column_selected
        )

        self.clustering_filter_value_container = ttk.Frame(subset_frame)
        self.clustering_filter_value_container.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.clustering_filter_categorical_frame = ttk.Frame(self.clustering_filter_value_container)
        ttk.Label(
            self.clustering_filter_categorical_frame,
            textvariable=self.clustering_filter_values_label_var,
        ).pack(anchor="w")
        self.clustering_filter_values_listbox = tk.Listbox(
            self.clustering_filter_categorical_frame,
            listvariable=tk.StringVar(value=[]),
            selectmode="extended",
            exportselection=False,
            height=6,
            width=40,
        )
        self.clustering_filter_values_listbox.pack(side="left", fill="both", expand=True)
        cat_scroll = ttk.Scrollbar(
            self.clustering_filter_categorical_frame,
            orient="vertical",
            command=self.clustering_filter_values_listbox.yview,
        )
        cat_scroll.pack(side="right", fill="y")
        self.clustering_filter_values_listbox.configure(yscrollcommand=cat_scroll.set)

        self.clustering_filter_numeric_frame = ttk.Frame(self.clustering_filter_value_container)
        ttk.Label(self.clustering_filter_numeric_frame, text="Numeric range").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(self.clustering_filter_numeric_frame, text="Min").grid(row=1, column=0, sticky="w")
        self.clustering_filter_min_entry = ttk.Spinbox(
            self.clustering_filter_numeric_frame,
            textvariable=self.clustering_filter_min_var,
            width=10,
        )
        self.clustering_filter_min_entry.grid(row=1, column=1, sticky="w", padx=(4, 12))
        ttk.Label(self.clustering_filter_numeric_frame, text="Max").grid(row=1, column=2, sticky="w")
        self.clustering_filter_max_entry = ttk.Spinbox(
            self.clustering_filter_numeric_frame,
            textvariable=self.clustering_filter_max_var,
            width=10,
        )
        self.clustering_filter_max_entry.grid(row=1, column=3, sticky="w", padx=(4, 0))
        self.clustering_filter_min_var.set("")
        self.clustering_filter_max_var.set("")

        ttk.Button(
            subset_frame,
            text="Add filter",
            command=self._add_clustering_filter,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(8, 0))

        self.clustering_filter_tree = ttk.Treeview(
            subset_frame,
            columns=("column", "type", "criteria"),
            show="headings",
            height=5,
        )
        self.clustering_filter_tree.heading("column", text="Column")
        self.clustering_filter_tree.heading("type", text="Type")
        self.clustering_filter_tree.heading("criteria", text="Criteria")
        self.clustering_filter_tree.column("column", width=180, anchor="w")
        self.clustering_filter_tree.column("type", width=80, anchor="center")
        self.clustering_filter_tree.column("criteria", width=260, anchor="w")
        self.clustering_filter_tree.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        subset_frame.grid_rowconfigure(3, weight=1)
        subset_frame.grid_columnconfigure(0, weight=0)
        subset_frame.grid_columnconfigure(1, weight=0)
        subset_frame.grid_columnconfigure(2, weight=1)
        subset_frame.grid_columnconfigure(3, weight=0)

        ttk.Button(
            subset_frame,
            text="Remove selected",
            command=self._remove_selected_clustering_filter,
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self._refresh_clustering_filter_tree()

        feature_frame = ttk.LabelFrame(
            module_tab, text="Feature Columns for Clustering", padding=12
        )
        feature_frame.pack(fill="x", expand=False, pady=(12, 0))

        ttk.Label(
            feature_frame,
            text=(
                "Select numeric columns to feed into clustering algorithms. Columns missing in any file are highlighted."
            ),
            wraplength=650,
            justify="left",
        ).pack(anchor="w")

        listbox_frame = ttk.Frame(feature_frame)
        listbox_frame.pack(fill="x", pady=(8, 0))

        self.clustering_listbox = tk.Listbox(
            listbox_frame,
            listvariable=tk.StringVar(value=[]),
            selectmode="extended",
            exportselection=False,
            height=10,
        )
        self.clustering_listbox.pack(side="left", fill="both", expand=True)
        self.clustering_listbox.bind(
            "<<ListboxSelect>>", lambda _evt: self._on_clustering_selection_changed()
        )
        self.clustering_listbox.bind(
            "<Control-Button-1>", self._toggle_clustering_listbox_selection
        )
        self.clustering_listbox.bind(
            "<Command-Button-1>", self._toggle_clustering_listbox_selection
        )

        clustering_feature_scroll = ttk.Scrollbar(
            listbox_frame,
            orient="vertical",
            command=self.clustering_listbox.yview,
        )
        clustering_feature_scroll.pack(side="right", fill="y")
        self.clustering_listbox.configure(yscrollcommand=clustering_feature_scroll.set)

        action_row = ttk.Frame(feature_frame)
        action_row.pack(anchor="w", pady=(8, 0))
        ttk.Button(
            action_row,
            text="Select All Common",
            command=self._select_all_clustering_common_features,
        ).pack(side="left")
        ttk.Button(
            action_row,
            text="Clear Selection",
            command=lambda: self.clustering_listbox.selection_clear(0, tk.END),
        ).pack(side="left", padx=(8, 0))

        ttk.Label(feature_frame, textvariable=self.clustering_hint_var).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            feature_frame,
            textvariable=self.clustering_missing_var,
            foreground="red",
        ).pack(anchor="w")

        downsample_frame = ttk.LabelFrame(module_tab, text="Downsampling", padding=12)
        downsample_frame.pack(fill="both", expand=False, pady=(12, 0))

        controls_row = ttk.Frame(downsample_frame)
        controls_row.pack(fill="x")

        ttk.Label(controls_row, text="Method").pack(side="left")
        downsample_methods = [
            "None",
            "Total Count",
            "Per File",
            "Per Class",
            "Per File + Class",
        ]
        self.clustering_downsample_method_combo = ttk.Combobox(
            controls_row,
            state="readonly",
            textvariable=self.clustering_downsample_method_var,
            values=downsample_methods,
            width=18,
        )
        self.clustering_downsample_method_combo.pack(side="left", padx=(6, 12))
        self.clustering_downsample_method_combo.bind(
            "<<ComboboxSelected>>",
            lambda _evt: self._on_clustering_downsample_method_changed(),
        )

        self.clustering_downsample_value_label = ttk.Label(
            controls_row, textvariable=self.clustering_downsample_value_label_var
        )
        self.clustering_downsample_value_label.pack(side="left")

        self.clustering_downsample_value_entry = ttk.Entry(
            controls_row, textvariable=self.clustering_downsample_value_var, width=12
        )
        self.clustering_downsample_value_entry.pack(side="left", padx=(6, 12))

        ttk.Button(
            controls_row,
            text="Preview Downsampling",
            command=self._clustering_preview_downsampling,
        ).pack(side="left")

        ttk.Label(
            downsample_frame,
            textvariable=self.clustering_downsample_message_var,
            wraplength=650,
        ).pack(anchor="w", pady=(10, 4))

        clustering_tree_frame = ttk.Frame(downsample_frame)
        clustering_tree_frame.pack(fill="both", expand=True)

        self.clustering_downsample_file_tree = ttk.Treeview(
            clustering_tree_frame,
            columns=("file", "rows"),
            show="headings",
            height=6,
        )
        self.clustering_downsample_file_tree.heading("file", text="File")
        self.clustering_downsample_file_tree.heading("rows", text="Rows")
        self.clustering_downsample_file_tree.column("file", anchor="w", width=220)
        self.clustering_downsample_file_tree.column("rows", anchor="center", width=100)
        self.clustering_downsample_file_tree.pack(side="left", fill="both", expand=True)

        clustering_downsample_file_scroll = ttk.Scrollbar(
            clustering_tree_frame,
            orient="vertical",
            command=self.clustering_downsample_file_tree.yview,
        )
        self.clustering_downsample_file_tree.configure(
            yscroll=clustering_downsample_file_scroll.set
        )
        clustering_downsample_file_scroll.pack(side="left", fill="y")

        self.clustering_downsample_category_tree = ttk.Treeview(
            clustering_tree_frame,
            columns=("category", "rows"),
            show="headings",
            height=6,
        )
        self.clustering_downsample_category_tree.heading("category", text="Category")
        self.clustering_downsample_category_tree.heading("rows", text="Rows")
        self.clustering_downsample_category_tree.column(
            "category", anchor="w", width=140
        )
        self.clustering_downsample_category_tree.column(
            "rows", anchor="center", width=80
        )
        self.clustering_downsample_category_tree.pack(
            side="left", fill="both", expand=True, padx=(12, 0)
        )

        clustering_downsample_category_scroll = ttk.Scrollbar(
            clustering_tree_frame,
            orient="vertical",
            command=self.clustering_downsample_category_tree.yview,
        )
        self.clustering_downsample_category_tree.configure(
            yscroll=clustering_downsample_category_scroll.set
        )
        clustering_downsample_category_scroll.pack(side="left", fill="y")

        methods_frame = ttk.LabelFrame(
            module_tab, text="Clustering Methods", padding=12
        )
        methods_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.clustering_method_param_widgets = {}
        for method_key, method_config in self.clustering_methods.items():
            method_panel = ttk.Frame(methods_frame)
            method_panel.pack(fill="x", expand=True, pady=(0, 12))

            ttk.Checkbutton(
                method_panel,
                text=method_config["label"],
                variable=method_config["selected"],
            ).pack(anchor="w")

            params_frame = ttk.Frame(method_panel, padding=(24, 4))
            params_frame.pack(fill="x", expand=True)
            widgets = []
            for param_key, param_config in method_config["params"].items():
                label_text = param_config["label"]
                var = param_config["var"]
                param_type = param_config["type"]
                frame = ttk.Frame(params_frame)
                frame.pack(fill="x", pady=2)
                ttk.Label(frame, text=label_text).pack(side="left")
                entry = ttk.Entry(frame, textvariable=var, width=14)
                entry.pack(side="left", padx=(8, 0))
                hint = ttk.Label(frame, text="(comma separated)", foreground="#666666")
                hint.pack(side="left", padx=(8, 0))
                widgets.append(entry)
            self.clustering_method_param_widgets[method_key] = widgets

        self.clustering_downsample_method_combo.current(0)
        self._on_clustering_downsample_method_changed()

        action_frame = ttk.Frame(module_tab)
        action_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(action_frame, text="CPU cores").pack(side="left")
        self.clustering_n_jobs_spin = ttk.Spinbox(
            action_frame,
            from_=1,
            to=self.total_cpu_cores,
            increment=1,
            textvariable=self.clustering_n_jobs_var,
            width=6,
        )
        self.clustering_n_jobs_spin.pack(side="left", padx=(6, 18))

        self.run_clustering_button = ttk.Button(
            action_frame, text="Run Clustering", command=self.start_clustering
        )
        self.run_clustering_button.pack(side="left")

        self.clustering_progress = ttk.Progressbar(
            action_frame, mode="indeterminate", length=220
        )
        self.clustering_progress.pack(side="left", padx=(12, 0))

        ttk.Label(
            action_frame,
            textvariable=self.clustering_status_var,
            wraplength=650,
            justify="left",
        ).pack(side="left", padx=(12, 0))
        ttk.Button(
            action_frame,
            text="Save Clustering Output…",
            command=self.save_clustering_output,
        ).pack(side="left", padx=(12, 0))

    def _init_clustering_results_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Results")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        results_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        results_tab.pack(fill="both", expand=True)

        summary_frame = ttk.Frame(results_tab)
        summary_frame.pack(fill="x")
        ttk.Label(summary_frame, text="Method summaries").pack(anchor="w")

        self.clustering_summary_tree = ttk.Treeview(
            summary_frame,
            columns=("method", "clusters", "rows"),
            show="headings",
            height=6,
        )
        self.clustering_summary_tree.heading("method", text="Method")
        self.clustering_summary_tree.heading("clusters", text="Clusters")
        self.clustering_summary_tree.heading("rows", text="Rows used")
        self.clustering_summary_tree.column("method", anchor="w", width=220)
        self.clustering_summary_tree.column("clusters", anchor="center", width=100)
        self.clustering_summary_tree.column("rows", anchor="center", width=100)
        self.clustering_summary_tree.pack(fill="x", pady=(4, 8))

        clusters_frame = ttk.Frame(results_tab)
        clusters_frame.pack(fill="both", expand=True)
        ttk.Label(clusters_frame, text="Cluster membership counts").pack(anchor="w")

        self.clustering_clusters_tree = ttk.Treeview(
            clusters_frame,
            columns=("method", "cluster", "count"),
            show="headings",
            height=12,
        )
        self.clustering_clusters_tree.heading("method", text="Method")
        self.clustering_clusters_tree.heading("cluster", text="Cluster")
        self.clustering_clusters_tree.heading("count", text="Count")
        self.clustering_clusters_tree.column("method", anchor="w", width=220)
        self.clustering_clusters_tree.column("cluster", anchor="center", width=120)
        self.clustering_clusters_tree.column("count", anchor="center", width=100)
        self.clustering_clusters_tree.pack(fill="both", expand=True, pady=(4, 0))

        clusters_scroll = ttk.Scrollbar(
            clusters_frame,
            orient="vertical",
            command=self.clustering_clusters_tree.yview,
        )
        self.clustering_clusters_tree.configure(yscroll=clusters_scroll.set)
        clusters_scroll.pack(side="right", fill="y")

    def _init_clustering_visuals_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Visualization")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        viz_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        viz_tab.pack(fill="both", expand=True)

        # UMAP controls
        umap_frame = ttk.LabelFrame(viz_tab, text="UMAP Projection", padding=12)
        umap_frame.pack(fill="both", expand=False)

        self.clustering_umap_method_var = tk.StringVar()
        ttk.Label(umap_frame, text="Clustering method").grid(row=0, column=0, sticky="w")
        self.clustering_umap_method_combo = ttk.Combobox(
            umap_frame,
            textvariable=self.clustering_umap_method_var,
            state="readonly",
            values=[],
            width=22,
        )
        self.clustering_umap_method_combo.grid(row=1, column=0, sticky="w")

        self.clustering_umap_color_mode_var = tk.StringVar(value="cluster")
        color_frame = ttk.Frame(umap_frame)
        color_frame.grid(row=0, column=1, rowspan=2, padx=(18, 0), sticky="w")
        ttk.Label(color_frame, text="Color mode").pack(anchor="w")
        ttk.Radiobutton(
            color_frame,
            text="Clusters",
            value="cluster",
            variable=self.clustering_umap_color_mode_var,
            command=self._on_umap_color_mode_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            color_frame,
            text="Marker expression",
            value="marker",
            variable=self.clustering_umap_color_mode_var,
            command=self._on_umap_color_mode_changed,
        ).pack(anchor="w")

        ttk.Label(umap_frame, text="Marker for heatmap coloring").grid(
            row=0, column=2, sticky="w"
        )
        self.clustering_umap_marker_combo = ttk.Combobox(
            umap_frame,
            textvariable=self.clustering_umap_marker_var,
            state="disabled",
            values=[],
            width=22,
        )
        self.clustering_umap_marker_combo.grid(row=1, column=2, sticky="w")

        params_frame = ttk.Frame(umap_frame)
        params_frame.grid(row=2, column=0, columnspan=3, pady=(12, 0), sticky="w")
        self.clustering_umap_neighbors_var = tk.IntVar(value=15)
        self.clustering_umap_min_dist_var = tk.DoubleVar(value=0.1)
        self.clustering_umap_metric_var = tk.StringVar(value="euclidean")

        ttk.Label(params_frame, text="Neighbors").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            params_frame,
            from_=5,
            to=200,
            increment=1,
            textvariable=self.clustering_umap_neighbors_var,
            width=6,
        ).grid(row=1, column=0, sticky="w")

        ttk.Label(params_frame, text="Min dist").grid(row=0, column=1, padx=(12, 0), sticky="w")
        ttk.Spinbox(
            params_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.clustering_umap_min_dist_var,
            width=6,
            format="%.2f",
        ).grid(row=1, column=1, padx=(12, 0), sticky="w")

        ttk.Label(params_frame, text="Metric").grid(row=0, column=2, padx=(12, 0), sticky="w")
        ttk.Combobox(
            params_frame,
            textvariable=self.clustering_umap_metric_var,
            state="readonly",
            values=["euclidean", "manhattan", "cosine"],
            width=10,
        ).grid(row=1, column=2, padx=(12, 0), sticky="w")

        ttk.Label(params_frame, text="Dot size").grid(row=0, column=3, padx=(12, 0), sticky="w")
        self.clustering_umap_dot_size_var = tk.DoubleVar(value=6.0)
        ttk.Spinbox(
            params_frame,
            from_=1.0,
            to=50.0,
            increment=0.5,
            textvariable=self.clustering_umap_dot_size_var,
            width=6,
            format="%.1f",
        ).grid(row=1, column=3, padx=(12, 0), sticky="w")

        ttk.Label(params_frame, text="Alpha").grid(row=0, column=4, padx=(12, 0), sticky="w")
        self.clustering_umap_alpha_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(
            params_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            textvariable=self.clustering_umap_alpha_var,
            width=6,
            format="%.2f",
        ).grid(row=1, column=4, padx=(12, 0), sticky="w")

        ttk.Button(
            params_frame,
            text="Generate UMAP",
            command=self._generate_clustering_umap,
        ).grid(row=1, column=5, padx=(18, 0), sticky="w")

        self.clustering_umap_fig = Figure(figsize=(7, 5.5), dpi=100)
        self.clustering_umap_ax = self.clustering_umap_fig.add_subplot(111)
        self.clustering_umap_ax.set_title("Run clustering to generate UMAP")
        self.clustering_umap_ax.set_xticks([])
        self.clustering_umap_ax.set_yticks([])
        button_frame = ttk.Frame(umap_frame)
        button_frame.grid(row=2, column=3, padx=(18, 0), sticky="nw")
        ttk.Label(button_frame, text="Downsample cells").pack(anchor="w")
        ttk.Checkbutton(
            button_frame,
            text="Enable",
            variable=self.clustering_umap_downsample_var,
        ).pack(anchor="w")
        ttk.Label(button_frame, text="Max cells").pack(anchor="w", pady=(4, 0))
        ttk.Spinbox(
            button_frame,
            from_=1000,
            to=200000,
            increment=1000,
            textvariable=self.clustering_umap_max_cells_var,
            width=10,
        ).pack(anchor="w")
        ttk.Label(button_frame, text="Parallel workers").pack(anchor="w", pady=(4, 0))
        ttk.Spinbox(
            button_frame,
            from_=1,
            to=self.total_cpu_cores,
            increment=1,
            textvariable=self.clustering_umap_jobs_var,
            width=10,
        ).pack(anchor="w")
        ttk.Checkbutton(
            button_frame,
            text="Label centroids",
            variable=self.clustering_umap_show_labels_var,
        ).pack(anchor="w", pady=(4, 0))

        umap_canvas = FigureCanvasTkAgg(self.clustering_umap_fig, master=umap_frame)
        umap_canvas.get_tk_widget().grid(
            row=3, column=0, columnspan=4, pady=(12, 0), sticky="nsew"
        )
        self._make_canvas_responsive(umap_canvas, self.clustering_umap_fig, min_height=260)
        self.clustering_umap_canvas = umap_canvas

        umap_frame.grid_columnconfigure(3, weight=1)
        umap_frame.grid_rowconfigure(3, weight=1)
        ttk.Button(
            button_frame,
            text="Save Figure…",
            command=self._save_clustering_umap_figure,
        ).pack(anchor="w", pady=(8, 0))

        # Heatmap controls
        heatmap_frame = ttk.LabelFrame(viz_tab, text="Cluster Heatmap", padding=12)
        heatmap_frame.pack(fill="both", expand=True, pady=(12, 0))

        ttk.Label(heatmap_frame, text="Clustering method").grid(row=0, column=0, sticky="w")
        self.clustering_heatmap_method_var = tk.StringVar()
        self.clustering_heatmap_method_combo = ttk.Combobox(
            heatmap_frame,
            textvariable=self.clustering_heatmap_method_var,
            state="readonly",
            values=[],
            width=22,
        )
        self.clustering_heatmap_method_combo.grid(row=1, column=0, sticky="w")

        markers_frame = ttk.Frame(heatmap_frame)
        markers_frame.grid(row=0, column=1, rowspan=2, padx=(18, 0), sticky="nsew")
        ttk.Label(markers_frame, text="Markers").pack(anchor="w")
        self.clustering_heatmap_markers_listbox = tk.Listbox(
            markers_frame,
            listvariable=tk.StringVar(value=[]),
            selectmode="extended",
            exportselection=False,
            height=8,
        )
        self.clustering_heatmap_markers_listbox.pack(side="left", fill="both", expand=True)
        markers_scroll = ttk.Scrollbar(
            markers_frame,
            orient="vertical",
            command=self.clustering_heatmap_markers_listbox.yview,
        )
        markers_scroll.pack(side="right", fill="y")
        self.clustering_heatmap_markers_listbox.configure(yscrollcommand=markers_scroll.set)

        norm_frame = ttk.Frame(heatmap_frame)
        norm_frame.grid(row=0, column=2, rowspan=2, padx=(18, 0), sticky="nw")
        ttk.Label(norm_frame, text="Normalization").pack(anchor="w")
        self.clustering_heatmap_norm_var = tk.StringVar(value="raw")
        ttk.Radiobutton(
            norm_frame,
            text="Raw",
            value="raw",
            variable=self.clustering_heatmap_norm_var,
            command=self._on_heatmap_norm_changed,
        ).pack(anchor="w")
        ttk.Radiobutton(
            norm_frame,
            text="Min-max (0-1)",
            value="minmax",
            variable=self.clustering_heatmap_norm_var,
            command=self._on_heatmap_norm_changed,
        ).pack(anchor="w")
        range_frame = ttk.Frame(norm_frame)
        range_frame.pack(anchor="w", pady=(4, 0))
        ttk.Radiobutton(
            range_frame,
            text="Fixed range",
            value="fixed",
            variable=self.clustering_heatmap_norm_var,
            command=self._on_heatmap_norm_changed,
        ).pack(anchor="w")
        range_values = ttk.Frame(norm_frame)
        range_values.pack(anchor="w")
        ttk.Label(range_values, text="Min").grid(row=0, column=0, sticky="w")
        self.clustering_heatmap_min_var = tk.DoubleVar(value=0.0)
        self.clustering_heatmap_min_entry = ttk.Spinbox(
            range_values,
            from_=-1000.0,
            to=1000.0,
            increment=0.5,
            textvariable=self.clustering_heatmap_min_var,
            width=6,
            format="%.2f",
            state="disabled",
        )
        self.clustering_heatmap_min_entry.grid(row=1, column=0, sticky="w")
        ttk.Label(range_values, text="Max").grid(row=0, column=1, padx=(8, 0), sticky="w")
        self.clustering_heatmap_max_var = tk.DoubleVar(value=5.0)
        self.clustering_heatmap_max_entry = ttk.Spinbox(
            range_values,
            from_=-1000.0,
            to=1000.0,
            increment=0.5,
            textvariable=self.clustering_heatmap_max_var,
            width=6,
            format="%.2f",
            state="disabled",
        )
        self.clustering_heatmap_max_entry.grid(row=1, column=1, padx=(8, 0), sticky="w")

        ttk.Button(
            heatmap_frame,
            text="Generate Heatmap",
            command=self._generate_clustering_heatmap,
        ).grid(row=1, column=3, padx=(18, 0), sticky="nw")

        dendro_frame = ttk.Frame(heatmap_frame)
        dendro_frame.grid(row=0, column=3, sticky="nw")
        ttk.Label(dendro_frame, text="Dendrograms").pack(anchor="w")
        ttk.Checkbutton(
            dendro_frame,
            text="Clusters",
            variable=self.clustering_heatmap_cluster_dendro_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            dendro_frame,
            text="Markers",
            variable=self.clustering_heatmap_marker_dendro_var,
        ).pack(anchor="w")

        self.clustering_heatmap_fig = Figure(figsize=(6, 4), dpi=100)
        self.clustering_heatmap_ax = self.clustering_heatmap_fig.add_subplot(111)
        self.clustering_heatmap_ax.set_title("Run clustering to view heatmap")
        heatmap_canvas = FigureCanvasTkAgg(
            self.clustering_heatmap_fig, master=heatmap_frame
        )
        heatmap_canvas.get_tk_widget().grid(
            row=2, column=0, columnspan=4, pady=(12, 0), sticky="nsew"
        )
        self._make_canvas_responsive(heatmap_canvas, self.clustering_heatmap_fig, min_height=240)
        self.clustering_heatmap_canvas = heatmap_canvas

        heatmap_frame.grid_columnconfigure(0, weight=1)
        heatmap_frame.grid_columnconfigure(1, weight=1)
        heatmap_frame.grid_rowconfigure(2, weight=1)
        ttk.Button(
            heatmap_frame,
            text="Save Figure…",
            command=self._save_clustering_heatmap_figure,
        ).grid(row=1, column=4, padx=(12, 0), sticky="nw")
        heatmap_frame.grid_columnconfigure(4, weight=0)
        self._update_clustering_visual_controls()
        self._refresh_cluster_explorer_controls()

    def _init_clustering_explorer_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Cluster Explorer")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        explorer_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        explorer_tab.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(explorer_tab, text="Explorer Controls", padding=12)
        controls.pack(fill="x", expand=False)
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(2, weight=0)
        controls.grid_columnconfigure(3, weight=0)

        ttk.Label(controls, text="Clustering method").grid(row=0, column=0, sticky="w")
        self.cluster_explorer_method_combo = ttk.Combobox(
            controls,
            textvariable=self.cluster_explorer_method_var,
            state="readonly",
            values=[],
            width=28,
        )
        self.cluster_explorer_method_combo.grid(row=1, column=0, sticky="w")
        self.cluster_explorer_method_combo.bind(
            "<<ComboboxSelected>>", self._on_cluster_explorer_method_changed
        )

        cluster_box = ttk.Frame(controls)
        cluster_box.grid(row=0, column=1, rowspan=2, padx=(18, 0), sticky="nsew")
        ttk.Label(cluster_box, text="Clusters to display").pack(anchor="w")
        cluster_list_height = 8
        cluster_list = tk.Listbox(
            cluster_box,
            selectmode="extended",
            exportselection=False,
            height=cluster_list_height,
        )
        cluster_list.pack(side="left", fill="y", expand=False)
        cluster_scroll = ttk.Scrollbar(
            cluster_box, orient="vertical", command=cluster_list.yview
        )
        cluster_scroll.pack(side="right", fill="y")
        cluster_list.configure(yscrollcommand=cluster_scroll.set)
        cluster_list.bind("<<ListboxSelect>>", self._on_cluster_explorer_clusters_changed)
        cluster_list.bind("<Control-Button-1>", self._toggle_cluster_explorer_selection)
        cluster_list.bind("<Command-Button-1>", self._toggle_cluster_explorer_selection)
        self.cluster_explorer_cluster_listbox = cluster_list

        button_box = ttk.Frame(controls)
        button_box.grid(row=0, column=2, rowspan=2, padx=(18, 0), sticky="n")
        self.cluster_explorer_select_all_button = ttk.Button(
            button_box,
            text="Select All",
            command=self._select_all_cluster_explorer_clusters,
            state="disabled",
        )
        self.cluster_explorer_select_all_button.pack(anchor="w")
        self.cluster_explorer_clear_button = ttk.Button(
            button_box,
            text="Clear Selection",
            command=self._clear_cluster_explorer_clusters,
            state="disabled",
        )
        self.cluster_explorer_clear_button.pack(anchor="w", pady=(4, 0))
        ttk.Button(
            button_box,
            text="Randomize Axes",
            command=self._randomize_cluster_explorer_features,
        ).pack(anchor="w", pady=(12, 0))

        ttk.Label(
            controls,
            textvariable=self.cluster_explorer_status_var,
            wraplength=600,
            justify="left",
            foreground="#444444",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        style_frame = ttk.Frame(controls)
        style_frame.grid(row=0, column=3, rowspan=2, padx=(18, 0), sticky="nw")
        ttk.Label(style_frame, text="Dot size").pack(anchor="w")
        dot_spin = ttk.Spinbox(
            style_frame,
            from_=1.0,
            to=200.0,
            increment=1.0,
            textvariable=self.cluster_explorer_dot_size_var,
            width=6,
            format="%.0f",
            command=self._on_cluster_explorer_style_change,
        )
        dot_spin.pack(anchor="w")
        dot_spin.bind("<FocusOut>", lambda _e: self._on_cluster_explorer_style_change())
        dot_spin.bind("<Return>", lambda _e: self._on_cluster_explorer_style_change())
        ttk.Label(style_frame, text="Alpha").pack(anchor="w", pady=(6, 0))
        alpha_spin = ttk.Spinbox(
            style_frame,
            from_=0.05,
            to=1.0,
            increment=0.05,
            textvariable=self.cluster_explorer_alpha_var,
            width=6,
            format="%.2f",
            command=self._on_cluster_explorer_style_change,
        )
        alpha_spin.pack(anchor="w")
        alpha_spin.bind("<FocusOut>", lambda _e: self._on_cluster_explorer_style_change())
        alpha_spin.bind("<Return>", lambda _e: self._on_cluster_explorer_style_change())

        plots_frame = ttk.Frame(explorer_tab)
        plots_frame.pack(fill="both", expand=True, pady=(12, 0))
        plots_frame.grid_columnconfigure(0, weight=1)
        plots_frame.grid_columnconfigure(1, weight=1)
        plots_frame.grid_columnconfigure(2, weight=1)

        self.cluster_explorer_plots.clear()
        total_plots = 6
        for index in range(total_plots):
            plot_container = ttk.LabelFrame(
                plots_frame,
                text=f"Scatter {index + 1}",
                padding=8,
            )
            row = index // 3
            column = index % 3
            plot_container.grid(
                row=row,
                column=column,
                padx=6,
                pady=6,
                sticky="nsew",
            )
            plots_frame.grid_rowconfigure(row, weight=1)

            fig = Figure(figsize=(4, 3), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title("Select method to view clusters")
            ax.set_xlabel("Feature X")
            ax.set_ylabel("Feature Y")
            canvas = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            self._make_canvas_responsive(canvas, fig, min_height=220)
            cid = canvas.mpl_connect(
                "button_press_event",
                partial(self._handle_cluster_explorer_axis_click, index),
            )
            plot_info = {
                "figure": fig,
                "ax": ax,
                "canvas": canvas,
                "index": index,
                "x_feature": None,
                "y_feature": None,
                "connection": cid,
            }
            self.cluster_explorer_plots.append(plot_info)

    def _refresh_cluster_explorer_controls(self) -> None:
        if not hasattr(self, "cluster_explorer_method_combo"):
            return
        method_labels = [
            self.clustering_method_labels.get(key, key)
            for key in self.clustering_results.keys()
        ]
        if not method_labels:
            self.cluster_explorer_method_combo["values"] = []
            self.cluster_explorer_method_var.set("")
            self.cluster_explorer_status_var.set(
                "Run clustering to explore clusters in this view."
            )
            if self.cluster_explorer_cluster_listbox is not None:
                self.cluster_explorer_cluster_listbox.delete(0, tk.END)
            if self.cluster_explorer_select_all_button is not None:
                self.cluster_explorer_select_all_button.configure(state="disabled")
            if self.cluster_explorer_clear_button is not None:
                self.cluster_explorer_clear_button.configure(state="disabled")
            for plot in self.cluster_explorer_plots:
                ax = plot["ax"]
                ax.clear()
                ax.set_title("No clustering results")
                ax.set_xlabel("Feature X")
                ax.set_ylabel("Feature Y")
                plot["canvas"].draw_idle()
            return

        self.cluster_explorer_method_combo["values"] = method_labels
        if self.cluster_explorer_method_var.get() not in method_labels:
            self.cluster_explorer_method_var.set(method_labels[0])

        dataset, _ = self._get_cluster_explorer_dataset()
        if dataset is None or dataset.empty:
            self.cluster_explorer_feature_options = []
            self.cluster_explorer_status_var.set(
                "Selected clustering result is empty."
            )
        else:
            feature_options = [
                feature
                for feature in self.clustering_features_used
                if feature in dataset.columns
            ]
            if not feature_options:
                feature_options = [
                    column
                    for column in dataset.columns
                    if column not in {"cluster", "__source_file"}
                    and is_numeric_dtype(dataset[column])
                ]
            self.cluster_explorer_feature_options = feature_options
        self.cluster_explorer_status_var.set(
            "Click an axis label to choose a feature. Coloring is based on cluster."
        )

        self._update_cluster_explorer_clusters()
        self._randomize_cluster_explorer_features()

    def _get_training_model_params(self, model_name: str) -> Dict[str, object]:
        if model_name == "Random Forest":
            n_estimators = max(int(self.n_estimators_var.get()), 1)
            max_depth_raw = self.max_depth_var.get().strip()
            if max_depth_raw:
                try:
                    max_depth = int(max_depth_raw)
                    if max_depth <= 0:
                        raise ValueError
                except ValueError:
                    raise ValueError("Max depth must be a positive integer or blank.")
            else:
                max_depth = None
            max_features_value = self.max_features_var.get()
            if max_features_value == "None":
                max_features = None
            elif max_features_value in {"auto", "sqrt", "log2"}:
                max_features = max_features_value
            else:
                max_features = None
            min_samples_leaf = max(int(self.min_samples_leaf_var.get()), 1)
            return {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
            }
        if model_name == "LDA":
            solver = self.lda_solver_var.get() or "svd"
            shrinkage_raw = (self.lda_shrinkage_var.get() or "").strip()
            shrinkage = None
            if shrinkage_raw:
                if solver not in {"lsqr", "eigen"}:
                    raise ValueError("Shrinkage is only supported for 'lsqr' or 'eigen' solvers.")
                if shrinkage_raw.lower() == "auto":
                    shrinkage = "auto"
                else:
                    try:
                        shrinkage = float(shrinkage_raw)
                    except ValueError:
                        raise ValueError("Shrinkage must be 'auto' or a numeric value.")
            return {"solver": solver, "shrinkage": shrinkage}
        if model_name == "SVM":
            kernel = self.svm_kernel_var.get() or "rbf"
            try:
                c_value = float(self.svm_c_var.get())
                if c_value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("SVM C must be a positive number.")
            gamma_text = (self.svm_gamma_var.get() or "scale").strip()
            if gamma_text not in {"scale", "auto"}:
                try:
                    gamma_value = float(gamma_text)
                except ValueError:
                    raise ValueError("SVM gamma must be 'scale', 'auto', or a numeric value.")
            else:
                gamma_value = gamma_text
            degree = max(int(self.svm_degree_var.get()), 1)
            return {
                "kernel": kernel,
                "C": c_value,
                "gamma": gamma_value,
                "degree": degree,
            }
        if model_name == "Logistic Regression":
            solver = self.lr_solver_var.get() or "lbfgs"
            penalty = self.lr_penalty_var.get() or "l2"
            if penalty == "l1" and solver not in {"saga", "liblinear"}:
                raise ValueError("L1 penalty requires saga or liblinear solver.")
            if penalty == "elasticnet" and solver != "saga":
                raise ValueError("Elastic net penalty requires the saga solver.")
            try:
                c_value = float(self.lr_c_var.get())
                if c_value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("Logistic regression C must be positive.")
            max_iter = max(int(self.lr_max_iter_var.get()), 50)
            l1_ratio = float(self.lr_l1_ratio_var.get())
            l1_ratio = min(max(l1_ratio, 0.0), 1.0)
            return {
                "solver": solver,
                "penalty": penalty,
                "C": c_value,
                "max_iter": max_iter,
                "l1_ratio": l1_ratio,
            }
        if model_name == "Naive Bayes":
            try:
                smoothing = float(self.nb_var_smoothing_var.get())
                if smoothing <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("Var smoothing must be positive.")
            return {"var_smoothing": smoothing}
        if model_name == "XGBoost":
            if XGBClassifier is None:
                raise ValueError("Install the 'xgboost' package to use this model.")
            params = {
                "n_estimators": max(int(self.xgb_estimators_var.get()), 10),
                "learning_rate": float(self.xgb_learning_rate_var.get()),
                "max_depth": max(int(self.xgb_max_depth_var.get()), 1),
                "subsample": float(self.xgb_subsample_var.get()),
                "colsample_bytree": float(self.xgb_colsample_var.get()),
            }
            return params
        if model_name == "LightGBM":
            if lgb is None:
                raise ValueError("Install the 'lightgbm' package to use this model.")
            params = {
                "n_estimators": max(int(self.lgb_estimators_var.get()), 10),
                "learning_rate": float(self.lgb_learning_rate_var.get()),
                "max_depth": int(self.lgb_max_depth_var.get()),
                "num_leaves": max(int(self.lgb_num_leaves_var.get()), 2),
                "subsample": float(self.lgb_subsample_var.get()),
            }
            return params
        if model_name == "KMeans":
            n_clusters = max(int(self.kmeans_clusters_var.get()), 2)
            max_iter = max(int(self.kmeans_max_iter_var.get()), 10)
            n_init = max(int(self.kmeans_n_init_var.get()), 1)
            return {
                "n_clusters": n_clusters,
                "max_iter": max_iter,
                "n_init": n_init,
            }
        if model_name == "Neural Network":
            layers_text = self.nn_hidden_layers_var.get()
            try:
                hidden_layers = [
                    int(value.strip())
                    for value in layers_text.split(",")
                    if value.strip()
                ]
            except ValueError:
                raise ValueError("Hidden layers must be a comma-separated list of integers.")
            if not hidden_layers:
                hidden_layers = [128, 64]
            activation = self.nn_activation_var.get() or "relu"
            try:
                dropout = float(self.nn_dropout_var.get())
            except (TypeError, ValueError):
                raise ValueError("Dropout must be numeric.")
            epochs = max(int(self.nn_epochs_var.get()), 1)
            batch_size = max(int(self.nn_batch_var.get()), 8)
            try:
                learning_rate = float(self.nn_lr_var.get())
                weight_decay = float(self.nn_weight_decay_var.get())
            except (TypeError, ValueError):
                raise ValueError("Learning rate and weight decay must be numeric.")
            use_gpu = bool(self.nn_use_gpu_var.get())
            return {
                "hidden_layers": hidden_layers,
                "activation": activation,
                "dropout": max(0.0, min(0.9, dropout)),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": max(0.0, weight_decay),
                "use_gpu": use_gpu,
            }
        raise ValueError(f"Unsupported model '{model_name}'.")

    def _on_cluster_explorer_method_changed(self, _event: Optional[tk.Event] = None) -> None:
        self._update_cluster_explorer_clusters()
        self._randomize_cluster_explorer_features()

    def _on_cluster_explorer_style_change(self) -> None:
        self._update_all_cluster_explorer_plots()

    def _select_all_cluster_explorer_clusters(self) -> None:
        if not self.cluster_explorer_cluster_listbox:
            return
        self.cluster_explorer_cluster_listbox.selection_set(0, tk.END)
        self._update_all_cluster_explorer_plots()

    def _clear_cluster_explorer_clusters(self) -> None:
        if not self.cluster_explorer_cluster_listbox:
            return
        self.cluster_explorer_cluster_listbox.selection_clear(0, tk.END)
        self._update_all_cluster_explorer_plots()

    def _on_cluster_explorer_clusters_changed(self, _event: Optional[tk.Event] = None) -> None:
        self._update_all_cluster_explorer_plots()

    def _toggle_cluster_explorer_selection(self, event: tk.Event) -> str:
        widget = event.widget
        if not isinstance(widget, tk.Listbox):
            return "break"
        index = widget.nearest(event.y)
        if index >= 0:
            if widget.selection_includes(index):
                widget.selection_clear(index)
            else:
                widget.selection_set(index)
        self._update_all_cluster_explorer_plots()
        return "break"

    def _randomize_cluster_explorer_features(self) -> None:
        dataset, _ = self._get_cluster_explorer_dataset()
        if dataset is None or dataset.empty:
            for plot in self.cluster_explorer_plots:
                plot["x_feature"] = None
                plot["y_feature"] = None
                self._update_cluster_explorer_plot(plot)
            return

        features = self._get_cluster_explorer_features(dataset)
        if len(features) < 1:
            self.cluster_explorer_status_var.set(
                "No numeric features available for the explorer view."
            )
            return

        rng = random.Random(RANDOM_STATE + int(time.time()))
        for plot in self.cluster_explorer_plots:
            if len(features) >= 2:
                x_feature, y_feature = rng.sample(features, 2)
            else:
                x_feature = y_feature = features[0]
            plot["x_feature"] = x_feature
            plot["y_feature"] = y_feature
            self._update_cluster_explorer_plot(plot)

    def _update_cluster_explorer_clusters(self) -> None:
        if not self.cluster_explorer_cluster_listbox:
            return
        dataset, method_key = self._get_cluster_explorer_dataset()
        listbox = self.cluster_explorer_cluster_listbox
        listbox.delete(0, tk.END)

        if dataset is None or dataset.empty or method_key is None:
            if self.cluster_explorer_select_all_button is not None:
                self.cluster_explorer_select_all_button.configure(state="disabled")
            if self.cluster_explorer_clear_button is not None:
                self.cluster_explorer_clear_button.configure(state="disabled")
            return

        cluster_values = sorted(
            {str(value) for value in dataset["cluster"].unique()}
        )
        for value in cluster_values:
            listbox.insert(tk.END, value)

        if self.cluster_explorer_select_all_button is not None:
            self.cluster_explorer_select_all_button.configure(
                state="normal" if cluster_values else "disabled"
            )
        if self.cluster_explorer_clear_button is not None:
            self.cluster_explorer_clear_button.configure(
                state="normal" if cluster_values else "disabled"
            )

        listbox.selection_set(0, tk.END)
        self._update_all_cluster_explorer_plots()

    def _update_all_cluster_explorer_plots(self) -> None:
        for plot in self.cluster_explorer_plots:
            self._update_cluster_explorer_plot(plot)

    def _get_cluster_explorer_dataset(self) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        method_label = self.cluster_explorer_method_var.get()
        method_key = (
            self._get_method_key_from_label(method_label)
            if method_label
            else None
        )
        if not method_key or method_key not in self.clustering_results:
            return None, None
        dataset = self.clustering_results[method_key]
        if dataset is None or dataset.empty:
            return None, method_key
        return dataset, method_key

    def _get_cluster_explorer_features(self, dataset: pd.DataFrame) -> List[str]:
        if not self.cluster_explorer_feature_options:
            feature_candidates = [
                column
                for column in dataset.columns
                if column not in {"cluster", "__source_file"}
                and is_numeric_dtype(dataset[column])
            ]
            return feature_candidates
        return [
            feature
            for feature in self.cluster_explorer_feature_options
            if feature in dataset.columns and is_numeric_dtype(dataset[feature])
        ]

    def _parse_param_values(
        self,
        raw_value: str,
        param_config: Dict[str, object],
        method_label: str,
    ) -> List[object]:
        value_type = param_config.get("type", "str")
        tokens = [token.strip() for token in re.split(r"[,\s]+", raw_value) if token.strip()]
        if not tokens:
            return []
        parsed: List[object] = []
        for token in tokens:
            try:
                if value_type == "int":
                    parsed.append(int(token))
                elif value_type == "float":
                    parsed.append(float(token))
                else:
                    parsed.append(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value '{token}' for {param_config.get('label', 'parameter')} "
                    f"in {method_label}."
                ) from exc
        return parsed

    @staticmethod
    def _format_param_value(value: object, value_type: str, for_key: bool = False) -> str:
        if value_type == "float":
            formatted = f"{float(value):g}"
        elif value_type == "int":
            formatted = f"{int(value)}"
        else:
            formatted = str(value)
        if for_key:
            formatted = formatted.replace("-", "m").replace(".", "p")
        return formatted

    def _build_clustering_run_key(
        self,
        method_key: str,
        params: Dict[str, object],
        method_info: Dict[str, object],
    ) -> str:
        tokens: List[str] = []
        for param_key, value in params.items():
            param_cfg = method_info["params"].get(param_key, {})  # type: ignore[index]
            token_tpl = param_cfg.get("token")
            value_type = param_cfg.get("type", "str")
            formatted_value = self._format_param_value(value, value_type, for_key=True)
            if token_tpl:
                tokens.append(token_tpl.format(formatted_value))
            else:
                tokens.append(f"{param_key}{formatted_value}")
        suffix = "_".join(tokens) if tokens else "default"
        return f"{method_key}:{suffix}"

    def _build_clustering_run_label(
        self,
        method_info: Dict[str, object],
        params: Dict[str, object],
    ) -> str:
        base_label = str(method_info.get("label", "Method"))
        display_tokens: List[str] = []
        for param_key, value in params.items():
            param_cfg = method_info["params"].get(param_key, {})  # type: ignore[index]
            display_tpl = param_cfg.get("display_token")
            value_type = param_cfg.get("type", "str")
            formatted_value = self._format_param_value(value, value_type, for_key=False)
            if display_tpl:
                display_tokens.append(display_tpl.format(formatted_value))
            else:
                display_tokens.append(f"{param_cfg.get('label', param_key)}={formatted_value}")
        if display_tokens:
            return f"{base_label} ({', '.join(display_tokens)})"
        return base_label

    def _update_cluster_explorer_plot(self, plot_info: Dict[str, Any]) -> None:
        ax: Axes = plot_info["ax"]  # type: ignore[assignment]
        canvas: FigureCanvasTkAgg = plot_info["canvas"]  # type: ignore[name-defined]
        dataset, _method_key = self._get_cluster_explorer_dataset()
        x_feature = plot_info.get("x_feature")
        y_feature = plot_info.get("y_feature")

        if dataset is None or dataset.empty or not x_feature or not y_feature:
            ax.clear()
            ax.set_title("Cluster explorer")
            ax.set_xlabel("Feature X")
            ax.set_ylabel("Feature Y")
            canvas.draw_idle()
            return

        available_features = self._get_cluster_explorer_features(dataset)
        if x_feature not in available_features or y_feature not in available_features:
            self.cluster_explorer_status_var.set(
                "Some selected features are unavailable; refreshing selections."
            )
            self._randomize_cluster_explorer_features()
            return

        df = dataset
        selected_clusters: List[str] = []
        if self.cluster_explorer_cluster_listbox is not None:
            selected_indices = self.cluster_explorer_cluster_listbox.curselection()
            selected_clusters = [
                self.cluster_explorer_cluster_listbox.get(idx)
                for idx in selected_indices
            ]
        if selected_clusters:
            df = df[df["cluster"].astype(str).isin(selected_clusters)]

        if df.empty:
            ax.clear()
            ax.set_title("No data for selected clusters")
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            canvas.draw_idle()
            return

        if len(df) > self.cluster_explorer_sample_limit:
            df = df.sample(
                n=self.cluster_explorer_sample_limit, random_state=RANDOM_STATE
            )

        ax.clear()
        clusters = df["cluster"].astype(str)
        unique_clusters = sorted(pd.unique(clusters))
        cmap = cm.get_cmap("tab20", max(1, len(unique_clusters)))
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        cluster_codes = np.array([cluster_to_index[c] for c in clusters], dtype=float)

        dot_size = max(1.0, float(self.cluster_explorer_dot_size_var.get()))
        alpha = max(0.05, min(1.0, float(self.cluster_explorer_alpha_var.get())))

        ax.scatter(
            df[x_feature],
            df[y_feature],
            c=cluster_codes,
            cmap=cmap,
            s=dot_size,
            alpha=alpha,
            linewidths=0,
            vmin=0,
            vmax=max(0, len(unique_clusters) - 1),
        )
        ax.set_xlabel(f"{x_feature} ▼")
        ax.set_ylabel(f"{y_feature} ▼")
        ax.set_title(f"{len(df)} cells | {len(unique_clusters)} clusters")
        ax.margins(0.05)

        existing_legend = ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                color=(
                    cmap(cluster_to_index[c] / max(1, len(unique_clusters) - 1))
                    if len(unique_clusters) > 1
                    else cmap(0.5)
                ),
                label=c,
            )
            for c in unique_clusters[:20]
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="Cluster",
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
                borderaxespad=0.0,
            fontsize=8,
            )
            plot_info["figure"].subplots_adjust(left=0.18, right=0.84, top=0.9, bottom=0.22)
        else:
            plot_info["figure"].subplots_adjust(left=0.18, right=0.96, top=0.9, bottom=0.22)
        canvas.draw_idle()

    def _handle_cluster_explorer_axis_click(
        self, plot_index: int, event: "matplotlib.backend_bases.MouseEvent"
    ) -> None:
        if event is None:
            return
        if not (0 <= plot_index < len(self.cluster_explorer_plots)):
            return
        plot_info = self.cluster_explorer_plots[plot_index]
        ax = plot_info["ax"]
        if ax is None:
            return
        contains_x, _ = ax.xaxis.label.contains(event)
        contains_y, _ = ax.yaxis.label.contains(event)
        if contains_x:
            self._open_cluster_explorer_feature_picker(plot_index, "x", event)
        elif contains_y:
            self._open_cluster_explorer_feature_picker(plot_index, "y", event)

    def _open_cluster_explorer_feature_picker(
        self,
        plot_index: int,
        axis: str,
        event: "matplotlib.backend_bases.MouseEvent",
    ) -> None:
        dataset, _ = self._get_cluster_explorer_dataset()
        if dataset is None or dataset.empty:
            messagebox.showinfo("No data", "Run clustering to populate the explorer.")
            return
        features = self._get_cluster_explorer_features(dataset)
        if not features:
            messagebox.showinfo(
                "No features",
                "No numeric features are available for selection.",
            )
            return
        plot_info = self.cluster_explorer_plots[plot_index]
        if self.cluster_explorer_feature_menu is not None:
            try:
                self.cluster_explorer_feature_menu.unpost()
            except Exception:
                pass
            self.cluster_explorer_feature_menu = None

        menu = tk.Menu(self.root, tearoff=0)
        for feature in features:
            menu.add_command(
                label=feature,
                command=lambda f=feature: self._set_cluster_explorer_feature(plot_index, axis, f),
            )
        widget = event.canvas.get_tk_widget()
        x_root = widget.winfo_pointerx()
        y_root = widget.winfo_pointery()
        self.cluster_explorer_feature_menu = menu
        try:
            menu.tk_popup(x_root, y_root)
        finally:
            menu.grab_release()

    def _set_cluster_explorer_feature(self, plot_index: int, axis: str, feature: str) -> None:
        if not (0 <= plot_index < len(self.cluster_explorer_plots)):
            return
        plot_info = self.cluster_explorer_plots[plot_index]
        if axis == "x":
            plot_info["x_feature"] = feature
        else:
            plot_info["y_feature"] = feature
        self._update_cluster_explorer_plot(plot_info)
        if self.cluster_explorer_feature_menu is not None:
            try:
                self.cluster_explorer_feature_menu.unpost()
            except Exception:
                pass
            self.cluster_explorer_feature_menu = None

    def _init_clustering_annotation_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Annotation Wizard")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        annotation_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        annotation_tab.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(annotation_tab, text="Annotation Controls", padding=12)
        controls.pack(fill="x", expand=False)
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(2, weight=0)

        ttk.Label(controls, text="Clustering run").grid(row=0, column=0, sticky="w")
        self.cluster_annotation_method_combo = ttk.Combobox(
            controls,
            textvariable=self.cluster_annotation_method_var,
            state="readonly",
            width=30,
            values=[],
        )
        self.cluster_annotation_method_combo.grid(row=1, column=0, sticky="w")
        self.cluster_annotation_method_combo.bind(
            "<<ComboboxSelected>>", self._on_annotation_method_changed
        )

        button_box = ttk.Frame(controls)
        button_box.grid(row=0, column=1, rowspan=2, padx=(18, 0), sticky="w")
        ttk.Button(
            button_box,
            text="Add Annotation Column",
            command=self._add_annotation_column,
        ).pack(anchor="w")
        ttk.Button(
            button_box,
            text="Remove Annotation Column",
            command=self._remove_annotation_column,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(
            button_box,
            text="Clear Annotations",
            command=self._clear_annotation_values,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Button(
            controls,
            text="Save Annotation Table…",
            command=self._save_annotation_table,
        ).grid(row=1, column=2, sticky="e")

        ttk.Label(
            annotation_tab,
            textvariable=self.cluster_annotation_info_var,
            foreground="#444444",
            wraplength=750,
            justify="left",
        ).pack(anchor="w", pady=(12, 4))

        tree_frame = ttk.Frame(annotation_tab)
        tree_frame.pack(fill="both", expand=True)
        self.annotation_tree = ttk.Treeview(
            tree_frame,
            columns=("cluster",),
            show="headings",
            selectmode="browse",
        )
        self.annotation_tree.heading("cluster", text="Cluster")
        self.annotation_tree.column("cluster", width=160, anchor="center")
        self.annotation_tree.pack(side="left", fill="both", expand=True)
        self.annotation_tree.bind("<Double-1>", self._on_annotation_cell_double_click)

        annotation_scroll_y = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.annotation_tree.yview
        )
        annotation_scroll_y.pack(side="left", fill="y")
        annotation_scroll_x = ttk.Scrollbar(
            tree_frame, orient="horizontal", command=self.annotation_tree.xview
        )
        annotation_scroll_x.pack(side="bottom", fill="x")
        self.annotation_tree.configure(
            yscroll=annotation_scroll_y.set, xscroll=annotation_scroll_x.set
        )

        ttk.Label(
            annotation_tab,
            textvariable=self.cluster_annotation_status_var,
            foreground="#666666",
        ).pack(anchor="w", pady=(6, 0))
        self._refresh_annotation_method_choices()
        self._refresh_cluster_comparison_controls()

    def _init_clustering_comparison_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Cluster Comparison")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        comparison_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        comparison_tab.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(comparison_tab, text="Comparison Controls", padding=12)
        controls.pack(fill="x", expand=False)
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=0)
        controls.grid_columnconfigure(2, weight=0)
        controls.grid_columnconfigure(3, weight=1)

        ttk.Label(controls, text="Clustering run").grid(row=0, column=0, sticky="w")
        self.cluster_compare_method_combo = ttk.Combobox(
            controls,
            textvariable=self.cluster_compare_method_var,
            state="readonly",
            width=28,
            values=[],
        )
        self.cluster_compare_method_combo.grid(row=1, column=0, sticky="w")
        self.cluster_compare_method_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._refresh_cluster_comparison_controls()
        )

        ttk.Label(controls, text="Baseline category").grid(row=0, column=1, sticky="w")
        self.cluster_compare_category_combo = ttk.Combobox(
            controls,
            textvariable=self.cluster_compare_category_var,
            state="readonly",
            width=28,
            values=[],
        )
        self.cluster_compare_category_combo.grid(row=1, column=1, sticky="w", padx=(12, 0))

        ttk.Button(
            controls,
            text="Add Comparison",
            command=self._generate_cluster_comparison,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))

        ttk.Label(
            controls,
            text=(
                "Compare new clusters against existing categorical labels. Each run adds stacked composition bars, "
                "top-class summaries, and sankey plots below."
            ),
            foreground="#444444",
            justify="left",
            wraplength=600,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        results_scroll = ScrollableFrame(comparison_tab)
        results_scroll.pack(fill="both", expand=True, pady=(12, 0))
        self.cluster_compare_results_frame = ttk.Frame(results_scroll.scrollable_frame)
        self.cluster_compare_results_frame.pack(fill="both", expand=True)

        ttk.Label(
            comparison_tab,
            text="Each new comparison is appended below. Use per-chart controls to save figures or tables.",
            foreground="#666666",
        ).pack(anchor="w", pady=(8, 0))

        self._refresh_cluster_comparison_controls()

    def _refresh_annotation_method_choices(self) -> None:
        if not hasattr(self, "cluster_annotation_method_combo"):
            return
        available_labels = [
            self.clustering_method_labels.get(key, key)
            for key in self.clustering_results.keys()
        ]
        current_label = self.cluster_annotation_method_var.get()
        self.cluster_annotation_method_combo["values"] = available_labels
        if current_label not in available_labels:
            if available_labels:
                self.cluster_annotation_method_var.set(available_labels[0])
            else:
                self.cluster_annotation_method_var.set("")
        self._load_annotation_table()

    def _on_annotation_method_changed(self, _event: Optional[tk.Event] = None) -> None:
        self._load_annotation_table()

    def _load_annotation_table(self) -> None:
        if self.annotation_edit_widget is not None:
            self._cancel_annotation_edit()

        label = self.cluster_annotation_method_var.get()
        run_key = self._get_method_key_from_label(label) if label else None
        self.current_annotation_run_key = run_key
        if not run_key or run_key not in self.clustering_results:
            self.cluster_annotation_info_var.set(
                "Select a clustering run to begin annotating clusters."
            )
            self.cluster_annotation_status_var.set("")
            self.current_annotation_columns = ["cluster"]
            if self.annotation_tree is not None:
                self.annotation_tree.delete(*self.annotation_tree.get_children())
                self.annotation_tree["columns"] = ("cluster",)
                self.annotation_tree.heading("cluster", text="Cluster")
            return

        dataset = self.clustering_results[run_key]
        clusters = sorted(dataset["cluster"].astype(str).unique(), key=str)
        existing = self.cluster_annotations.get(run_key)
        if existing is None:
            df = pd.DataFrame({"cluster": clusters})
        else:
            df = existing.copy()
            df["cluster"] = df["cluster"].astype(str)
            df = df.drop_duplicates(subset="cluster", keep="first")
            df = df.set_index("cluster").reindex(clusters)
            df = df.reset_index()
            df = df.fillna("")

        self.cluster_annotations[run_key] = df
        self.current_annotation_columns = list(df.columns)
        if self.current_annotation_columns[0] != "cluster":
            self.current_annotation_columns = ["cluster"] + [
                col for col in self.current_annotation_columns if col != "cluster"
            ]
            df = df[self.current_annotation_columns]
            self.cluster_annotations[run_key] = df
        self.cluster_annotation_recent_terms.setdefault(run_key, set())
        self.cluster_annotation_info_var.set(
            f"Annotating clustering run: {label}. Double-click a cell to edit."
        )
        self.cluster_annotation_status_var.set("")
        self._populate_annotation_tree(df)
        self._refresh_cluster_comparison_controls()

    def _populate_annotation_tree(self, dataframe: pd.DataFrame) -> None:
        if self.annotation_tree is None:
            return
        columns = list(dataframe.columns)
        self.annotation_tree.delete(*self.annotation_tree.get_children())
        self.annotation_tree["columns"] = columns
        for column in columns:
            heading = "Cluster" if column == "cluster" else column
            anchor = "center" if column == "cluster" else "w"
            width = 160 if column == "cluster" else 200
            self.annotation_tree.heading(column, text=heading)
            self.annotation_tree.column(column, anchor=anchor, width=width, stretch=True)

        for _, row in dataframe.iterrows():
            cluster_id = str(row["cluster"])
            values = [str(row.get(column, "")) for column in columns]
            self.annotation_tree.insert("", "end", iid=cluster_id, values=values)

        run_key = self.current_annotation_run_key
        if run_key:
            suggestions = self.cluster_annotation_recent_terms.setdefault(run_key, set())
            for column in columns[1:]:
                suggestions.update(
                    {
                        str(value)
                        for value in dataframe[column].dropna().unique()
                        if str(value).strip()
                    }
                )

    def _add_annotation_column(self) -> None:
        run_key = self.current_annotation_run_key
        if not run_key or run_key not in self.cluster_annotations:
            messagebox.showinfo("Select run", "Select a clustering run before adding columns.")
            return
        column_name = simpledialog.askstring(
            "Annotation column",
            "Enter the name of the new annotation column:",
            parent=self.root,
        )
        if not column_name:
            return
        column_name = column_name.strip()
        if not column_name:
            return
        if column_name.lower() == "cluster" or column_name in self.current_annotation_columns:
            messagebox.showerror(
                "Column exists",
                f"The column '{column_name}' already exists.",
            )
            return
        df = self.cluster_annotations[run_key]
        df[column_name] = ""
        self.cluster_annotations[run_key] = df
        self.current_annotation_columns.append(column_name)
        self._populate_annotation_tree(df)
        self.cluster_annotation_status_var.set(f"Added column '{column_name}'.")
        self._refresh_cluster_comparison_controls()

    def _remove_annotation_column(self) -> None:
        run_key = self.current_annotation_run_key
        if not run_key or run_key not in self.cluster_annotations:
            return
        existing_columns = [col for col in self.current_annotation_columns if col != "cluster"]
        if not existing_columns:
            messagebox.showinfo("No columns", "There are no annotation columns to remove.")
            return
        column_name = simpledialog.askstring(
            "Remove column",
            "Enter the name of the column to remove:",
            parent=self.root,
        )
        if not column_name:
            return
        column_name = column_name.strip()
        if column_name == "cluster" or column_name not in existing_columns:
            messagebox.showerror(
                "Unknown column",
                f"The column '{column_name}' cannot be removed.",
            )
            return
        df = self.cluster_annotations[run_key].drop(columns=[column_name])
        self.cluster_annotations[run_key] = df
        self.current_annotation_columns.remove(column_name)
        self._populate_annotation_tree(df)
        self.cluster_annotation_status_var.set(f"Removed column '{column_name}'.")
        self._refresh_cluster_comparison_controls()

    def _clear_annotation_values(self) -> None:
        run_key = self.current_annotation_run_key
        if not run_key or run_key not in self.cluster_annotations:
            return
        df = self.cluster_annotations[run_key]
        for column in self.current_annotation_columns[1:]:
            df[column] = ""
        self.cluster_annotations[run_key] = df
        self._populate_annotation_tree(df)
        self.cluster_annotation_status_var.set("Cleared annotation values.")
        self._refresh_cluster_comparison_controls()

    def _save_annotation_table(self) -> None:
        run_key = self.current_annotation_run_key
        if not run_key or run_key not in self.cluster_annotations:
            messagebox.showinfo("Select run", "Select a clustering run to save annotations.")
            return
        df = self.cluster_annotations[run_key]
        if df.empty:
            messagebox.showinfo("No data", "No annotations to save.")
            return
        label = self.cluster_annotation_method_var.get() or run_key
        default_filename = (
            re.sub(r"[^A-Za-z0-9]+", "_", label.lower()).strip("_") + "_annotations.csv"
        )
        path = filedialog.asksaveasfilename(
            title="Save annotations",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", f"Failed to save annotations: {exc}")
            return
        self.cluster_annotation_status_var.set(f"Annotations saved to {path}.")

    def _on_annotation_cell_double_click(self, event: tk.Event) -> None:
        if self.annotation_tree is None:
            return
        if self.annotation_edit_widget is not None:
            self._cancel_annotation_edit()

        region = self.annotation_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.annotation_tree.identify_row(event.y)
        column_id = self.annotation_tree.identify_column(event.x)
        if not item_id or column_id == "#0":
            return
        column_index = int(column_id.replace("#", ""))
        columns = list(self.annotation_tree["columns"])
        if column_index <= 1 or column_index > len(columns):
            return
        column_name = columns[column_index - 1]
        if column_name == "cluster":
            return

        bbox = self.annotation_tree.bbox(item_id, column_name)
        if not bbox:
            return
        x, y, width, height = bbox
        value = self.annotation_tree.set(item_id, column_name)
        run_key = self.current_annotation_run_key
        suggestions = self._get_annotation_suggestions(run_key, column_name)

        combo = ttk.Combobox(
            self.annotation_tree,
            values=sorted(suggestions, key=str.lower),
            width=max(12, int(width / 8)),
        )
        combo.place(x=x, y=y, width=width, height=height)
        combo.insert(0, value)
        combo.focus_set()
        combo.bind("<Return>", lambda _e: self._commit_annotation_edit(combo))
        combo.bind("<FocusOut>", lambda _e: self._commit_annotation_edit(combo))
        combo.bind("<Escape>", lambda _e: self._cancel_annotation_edit())
        self.annotation_edit_widget = combo
        self.annotation_edit_info = {
            "item": item_id,
            "column": column_name,
            "run_key": run_key,
        }

    def _get_annotation_suggestions(self, run_key: Optional[str], column_name: str) -> Set[str]:
        suggestions: Set[str] = set()
        if run_key and run_key in self.cluster_annotations:
            df = self.cluster_annotations[run_key]
            if column_name in df.columns:
                suggestions.update(
                    {
                        str(value)
                        for value in df[column_name].dropna().unique()
                        if str(value).strip()
                    }
                )
        if run_key:
            suggestions.update(self.cluster_annotation_recent_terms.get(run_key, set()))
        return {s for s in suggestions if s}

    def _commit_annotation_edit(self, widget: ttk.Combobox) -> None:
        if self.annotation_edit_info is None:
            widget.destroy()
            self.annotation_edit_widget = None
            return
        new_value = widget.get().strip()
        info = self.annotation_edit_info
        item_id = info["item"]
        column_name = info["column"]
        run_key = info["run_key"]
        widget.destroy()
        self.annotation_edit_widget = None
        self.annotation_edit_info = None
        if not run_key or run_key not in self.cluster_annotations:
            return
        df = self.cluster_annotations[run_key]
        mask = df["cluster"].astype(str) == str(item_id)
        if column_name not in df.columns:
            df[column_name] = ""
        df.loc[mask, column_name] = new_value
        self.cluster_annotations[run_key] = df
        self.annotation_tree.set(item_id, column_name, new_value)
        if new_value:
            self.cluster_annotation_recent_terms.setdefault(run_key, set()).add(new_value)
        self.cluster_annotation_status_var.set(
            f"Updated cluster {item_id} → {column_name} = {new_value or '(blank)'}"
        )

    def _refresh_cluster_comparison_controls(self) -> None:
        if not hasattr(self, "cluster_compare_method_combo"):
            return
        method_labels = [
            self.clustering_method_labels.get(key, key)
            for key in self.clustering_results.keys()
        ]
        current_method = self.cluster_compare_method_var.get()
        self.cluster_compare_method_combo["values"] = method_labels
        if current_method not in method_labels:
            if method_labels:
                self.cluster_compare_method_var.set(method_labels[0])
            else:
                self.cluster_compare_method_var.set("")

        self.cluster_compare_category_map = {}
        category_options: List[str] = []
        run_key = self._get_method_key_from_label(self.cluster_compare_method_var.get())
        if run_key and run_key in self.clustering_results:
            dataset = self.clustering_results[run_key]
            dataset_columns = self._get_dataset_categorical_columns(dataset)
            for column in dataset_columns:
                category_options.append(column)
                self.cluster_compare_category_map[column] = ("dataset", column)

            annotation_df = self.cluster_annotations.get(run_key)
            if annotation_df is not None:
                for column in annotation_df.columns:
                    if column == "cluster":
                        continue
                    display = f"Annotation: {column}"
                    category_options.append(display)
                    self.cluster_compare_category_map[display] = ("annotation", column)

        current_category = self.cluster_compare_category_var.get()
        self.cluster_compare_category_combo["values"] = category_options
        if current_category not in category_options:
            if category_options:
                self.cluster_compare_category_var.set(category_options[0])
            else:
                self.cluster_compare_category_var.set("")

    def _get_dataset_categorical_columns(self, dataset: pd.DataFrame) -> List[str]:
        columns: List[str] = []
        for column in dataset.columns:
            if column in {"cluster", "__source_file"}:
                continue
            series = dataset[column]
            unique_count = series.nunique(dropna=True)
            if unique_count == 0:
                continue
            if (
                series.dtype == object
                or unique_count <= MAX_CLUSTER_COMPARE_CATEGORIES
            ):
                columns.append(column)
        return sorted(columns, key=str.lower)

    def _generate_cluster_comparison(self) -> None:
        if not self.clustering_results:
            messagebox.showinfo("No clustering", "Run clustering before generating comparisons.")
            return
        run_key = self._get_method_key_from_label(self.cluster_compare_method_var.get())
        if not run_key or run_key not in self.clustering_results:
            messagebox.showerror("Select run", "Select a clustering run to compare.")
            return
        category_label = self.cluster_compare_category_var.get()
        if not category_label or category_label not in self.cluster_compare_category_map:
            messagebox.showerror("Select category", "Choose a baseline categorical variable to compare against.")
            return

        source_type, column_name = self.cluster_compare_category_map[category_label]
        assignment_df = self.clustering_results[run_key]
        dataset_label = self.clustering_method_labels.get(run_key, run_key)

        if source_type == "dataset":
            if column_name not in assignment_df.columns:
                messagebox.showerror(
                    "Missing column",
                    f"Column '{column_name}' is unavailable in the selected clustering dataset.",
                )
                return
            category_series = assignment_df[column_name]
        else:
            annotation_df = self.cluster_annotations.get(run_key)
            if annotation_df is None or column_name not in annotation_df.columns:
                messagebox.showerror(
                    "Missing annotations",
                    f"Annotation column '{column_name}' is not available for this clustering run.",
                )
                return
            mapping = (
                annotation_df.set_index(annotation_df["cluster"].astype(str))[column_name]
            )
            category_series = assignment_df["cluster"].astype(str).map(mapping)

        comparison_df = pd.DataFrame(
            {
                "cluster": assignment_df["cluster"].astype(str),
                "category": category_series.astype(str),
            }
        )
        comparison_df = comparison_df.replace({"": np.nan}).dropna()
        if comparison_df.empty:
            messagebox.showerror(
                "No overlap",
                "No overlapping data between the selected cluster run and category column.",
            )
            return

        counts = (
            comparison_df.groupby(["cluster", "category"]).size().unstack(fill_value=0)
        )
        if counts.shape[1] > MAX_CLUSTER_COMPARE_CATEGORIES:
            messagebox.showerror(
                "Too many categories",
                f"The selected category column has more than {MAX_CLUSTER_COMPARE_CATEGORIES} unique values."
                " Please choose a column with fewer categories.",
            )
            return
        percents = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100
        clusters = counts.index.tolist()
        if len(clusters) < 1:
            messagebox.showerror("No clusters", "No clusters found for comparison.")
            return

        result_index = self.cluster_compare_result_counter + 1
        title = f"Comparison {result_index}: {dataset_label} vs {category_label}"
        container = ttk.LabelFrame(
            self.cluster_compare_results_frame,
            text=title,
            padding=10,
        )
        container.pack(fill="both", expand=True, pady=10)

        ttk.Label(
            container,
            text="Stacked bars show the percent of each cluster belonging to existing categories.",
            foreground="#444444",
        ).pack(anchor="w")

        bar_fig = self._create_cluster_comparison_bar_figure(
            percents,
            dataset_label,
            category_label,
        )
        bar_canvas = FigureCanvasTkAgg(bar_fig, master=container)
        bar_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        self._make_canvas_responsive(bar_canvas, bar_fig)
        ttk.Button(
            container,
            text="Save Bar Chart…",
            command=lambda fig=bar_fig: self._save_figure(fig, f"cluster_compare_bar_{result_index}.png"),
        ).pack(anchor="e", pady=(4, 8))

        summary_df = pd.DataFrame(
            {
                "cluster": counts.index,
                "top_class": percents.idxmax(axis=1),
                "percent": percents.max(axis=1).round(1),
            }
        )

        summary_frame = ttk.LabelFrame(container, text="Top category per cluster", padding=8)
        summary_frame.pack(fill="x", expand=False, pady=(4, 4))
        tree = ttk.Treeview(
            summary_frame,
            columns=("cluster", "category", "percent"),
            show="headings",
            height=min(len(summary_df), 6),
        )
        tree.heading("cluster", text="Cluster")
        tree.heading("category", text="Top category")
        tree.heading("percent", text="Percent")
        tree.column("cluster", width=120, anchor="center")
        tree.column("category", width=200, anchor="w")
        tree.column("percent", width=100, anchor="center")
        for _, row in summary_df.iterrows():
            tree.insert("", "end", values=(row["cluster"], row["top_class"], f"{row['percent']:.1f}%"))
        tree.pack(side="left", fill="x", expand=True)
        ttk.Button(
            summary_frame,
            text="Save Summary…",
            command=lambda df=summary_df.copy(): self._save_cluster_compare_summary(
                df, f"cluster_compare_summary_{result_index}.csv"
            ),
        ).pack(side="left", padx=(8, 0))

        sankey_fig = self._create_cluster_comparison_sankey(
            counts,
            dataset_label,
            category_label,
        )
        sankey_canvas = FigureCanvasTkAgg(sankey_fig, master=container)
        sankey_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self._make_canvas_responsive(sankey_canvas, sankey_fig, min_height=260)
        ttk.Button(
            container,
            text="Save Sankey Plot…",
            command=lambda fig=sankey_fig: self._save_figure(fig, f"cluster_compare_sankey_{result_index}.png"),
        ).pack(anchor="e", pady=(4, 0))

        self.cluster_compare_results.append(
            {
                "frame": container,
                "bar_fig": bar_fig,
                "sankey_fig": sankey_fig,
                "summary": summary_df,
            }
        )
        self.cluster_compare_result_counter = result_index
        self.cluster_compare_results_frame.update_idletasks()

    def _create_cluster_comparison_bar_figure(
        self,
        percents: pd.DataFrame,
        run_label: str,
        category_label: str,
    ) -> Figure:
        clusters = list(percents.index)
        categories = list(percents.columns)
        height = self._compute_dynamic_height(len(clusters), base_height=4.0, per_row=0.35)
        width = max(6.5, min(16.0, self._get_available_plot_width()))
        fig = Figure(figsize=(width, height), dpi=100)
        ax = fig.add_subplot(111)
        bottoms = np.zeros(len(clusters))
        palette = sns.color_palette("tab20", len(categories))
        for idx, category in enumerate(categories):
            values = percents[category].to_numpy()
            ax.barh(
                clusters,
                values,
                left=bottoms,
                color=palette[idx],
                label=category,
            )
            bottoms += values
        ax.set_xlabel("Percent of cluster (%)")
        ax.set_ylabel("Cluster")
        ax.set_title(f"{run_label}: composition vs {category_label}")
        ax.set_xlim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        self._finalize_figure_layout(fig)
        return fig

    def _create_cluster_comparison_sankey(
        self,
        counts: pd.DataFrame,
        run_label: str,
        category_label: str,
    ) -> Figure:
        clusters = list(counts.index)
        categories = list(counts.columns)
        total = counts.values.sum()
        height = self._compute_dynamic_height(len(clusters), base_height=4.5, per_row=0.4)
        width = max(7.0, min(16.0, self._get_available_plot_width()))
        fig = Figure(figsize=(width, height), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        cluster_totals = counts.sum(axis=1)
        category_totals = counts.sum(axis=0)

        def compute_positions(index, totals):
            positions = {}
            current = 0.0
            for key in index:
                height = totals.loc[key] / total
                positions[key] = (current, current + height)
                current += height
            return positions

        cluster_pos = compute_positions(counts.index, cluster_totals)
        category_pos = compute_positions(counts.columns, category_totals)
        cluster_offsets = {key: start for key, (start, _) in cluster_pos.items()}
        category_offsets = {key: start for key, (start, _) in category_pos.items()}

        left_x = 0.1
        right_x = 0.9
        rect_width = 0.03

        for cluster in clusters:
            start, end = cluster_pos[cluster]
            ax.add_patch(
                Rectangle((left_x - rect_width, start), rect_width, end - start, color="#555555", alpha=0.5)
            )
            ax.text(left_x - rect_width - 0.01, (start + end) / 2, cluster, ha="right", va="center")

        palette = sns.color_palette("tab20", len(categories))
        category_colors = {category: palette[idx] for idx, category in enumerate(categories)}

        for category in categories:
            start, end = category_pos[category]
            ax.add_patch(
                Rectangle((right_x, start), rect_width, end - start, color=category_colors[category], alpha=0.6)
            )
            ax.text(right_x + rect_width + 0.01, (start + end) / 2, category, ha="left", va="center")

        for cluster in clusters:
            for category in categories:
                value = counts.loc[cluster, category]
                if value <= 0:
                    continue
                height = value / total
                cluster_start = cluster_offsets[cluster]
                category_start = category_offsets[category]
                cluster_offsets[cluster] += height
                category_offsets[category] += height
                points = [
                    (left_x, cluster_start),
                    (left_x, cluster_start + height),
                    (right_x, category_start + height),
                    (right_x, category_start),
                ]
                polygon = Polygon(points, closed=True, facecolor=category_colors[category], alpha=0.35, edgecolor="none")
                ax.add_patch(polygon)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Sankey: {run_label} → {category_label}")
        self._finalize_figure_layout(fig)
        return fig

    def _save_cluster_compare_summary(self, dataframe: pd.DataFrame, default_filename: str) -> None:
        path = filedialog.asksaveasfilename(
            title="Save summary",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            dataframe.to_csv(path, index=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", f"Failed to save summary: {exc}")
            return
    def _compute_dynamic_height(
        self,
        rows: int,
        base_height: float = 4.0,
        per_row: float = 0.3,
        min_height: float = 3.5,
        max_height: float = 10.0,
    ) -> float:
        height = max(min_height, base_height + per_row * max(0, rows - 5))
        try:
            window_inches = self.root.winfo_height() / 120.0
            dynamic_max = max(min_height, min(max_height, window_inches))
        except Exception:
            dynamic_max = max_height
        return min(height, dynamic_max)

    def _get_available_plot_width(self, fallback: float = 10.0) -> float:
        try:
            width_pixels = self.root.winfo_width()
            if hasattr(self, "cluster_compare_results_frame"):
                width_pixels = max(width_pixels, self.cluster_compare_results_frame.winfo_width())
            if width_pixels <= 0:
                return fallback
            inches = max(6.0, min(18.0, width_pixels / 110.0))
            return inches
        except Exception:
            return fallback

    def _cancel_annotation_edit(self) -> None:
        if self.annotation_edit_widget is not None:
            try:
                self.annotation_edit_widget.destroy()
            except Exception:
                pass
        self.annotation_edit_widget = None
        self.annotation_edit_info = None

    def _init_training_visuals_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Visualizations")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        visuals_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        visuals_tab.pack(fill="both", expand=True)

        coverage_frame = ttk.LabelFrame(
            visuals_tab, text="Column coverage across files", padding=8
        )
        coverage_frame.pack(fill="both", expand=True)

        columns_fig = Figure(figsize=(6, 4), dpi=100)
        self.columns_ax = columns_fig.add_subplot(111)
        self.columns_ax.set_ylabel("Files")
        self.columns_ax.set_xlabel("Column")
        self.columns_ax.set_title("Load data to view column coverage")
        columns_canvas = FigureCanvasTkAgg(columns_fig, master=coverage_frame)
        columns_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._make_canvas_responsive(columns_canvas, columns_fig, min_height=240)
        self.columns_fig = columns_fig
        self.columns_canvas = columns_canvas
        ttk.Button(
            coverage_frame,
            text="Save Figure…",
            command=self._save_training_columns_figure,
        ).pack(anchor="e", pady=(6, 0))

        category_frame = ttk.LabelFrame(
            visuals_tab,
            text="Target category distribution",
            padding=8,
        )
        category_frame.pack(fill="both", expand=True, pady=(12, 0))

        category_fig = Figure(figsize=(6, 4), dpi=100)
        self.category_ax = category_fig.add_subplot(111)
        self.category_ax.set_ylabel("Rows")
        self.category_ax.set_xlabel("Category")
        self.category_ax.set_title("Select a target column to view category counts")
        self.category_ax.tick_params(axis="x", labelrotation=45)
        category_canvas = FigureCanvasTkAgg(category_fig, master=category_frame)
        category_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._make_canvas_responsive(category_canvas, category_fig, min_height=240)
        self.category_fig = category_fig
        self.category_canvas = category_canvas
        ttk.Button(
            category_frame,
            text="Save Figure…",
            command=self._save_training_category_figure,
        ).pack(anchor="e", pady=(6, 0))

        importance_frame = ttk.LabelFrame(
            visuals_tab,
            text="Feature importance (post-training)",
            padding=8,
        )
        importance_frame.pack(fill="both", expand=True, pady=(12, 0))

        viz_importance_fig = Figure(figsize=(6, 4), dpi=100)
        self.visual_importance_ax = viz_importance_fig.add_subplot(111)
        self.visual_importance_ax.set_title("Train a model to view feature importance")
        self.visual_importance_ax.set_xlabel("Importance")
        self.visual_importance_ax.set_ylabel("Feature")
        viz_importance_canvas = FigureCanvasTkAgg(
            viz_importance_fig, master=importance_frame
        )
        viz_importance_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._make_canvas_responsive(viz_importance_canvas, viz_importance_fig, min_height=240)
        self.visual_importance_fig = viz_importance_fig
        self.visual_importance_canvas = viz_importance_canvas
        ttk.Button(
            importance_frame,
            text="Save Figure…",
            command=self._save_training_visual_importance_figure,
        ).pack(anchor="e", pady=(6, 0))

    def _init_results_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Results")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        results_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        results_tab.pack(fill="both", expand=True)

        metrics_frame = ttk.LabelFrame(
            results_tab, text="Evaluation Summary", padding=8
        )
        metrics_frame.pack(fill="both", expand=False, padx=0, pady=(0, 12))

        ttk.Label(
            metrics_frame,
            textvariable=self.training_status_var,
            wraplength=700,
            justify="left",
        ).pack(anchor="w")

        self.cv_summary_var = tk.StringVar(
            value="Train a model to view cross-validation results."
        )
        ttk.Label(
            metrics_frame,
            textvariable=self.cv_summary_var,
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 4))

        text_frame = ttk.Frame(metrics_frame)
        text_frame.pack(fill="both", expand=True)

        self.metrics_text = tk.Text(
            text_frame,
            height=12,
            wrap="word",
            state="disabled",
            background=self.root.cget("background"),
            relief="flat",
        )
        self.metrics_text.pack(side="left", fill="both", expand=True)

        metrics_scroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.metrics_text.yview
        )
        metrics_scroll.pack(side="right", fill="y")
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)

        button_row = ttk.Frame(metrics_frame)
        button_row.pack(fill="x", pady=(6, 0))
        self.save_model_button = ttk.Button(
            button_row,
            text="Save Model…",
            command=self.save_model,
            state="disabled",
        )
        self.save_model_button.pack(side="left")

        plots_frame = ttk.Frame(results_tab)
        plots_frame.pack(fill="both", expand=True)

        confusion_frame = ttk.LabelFrame(
            plots_frame, text="Confusion Matrix", padding=8
        )
        confusion_frame.pack(fill="both", expand=True)

        confusion_fig = Figure(figsize=(7, 6), dpi=100)
        self.confusion_ax = confusion_fig.add_subplot(111)
        self.confusion_ax.set_title("Train a model to view the confusion matrix")
        self.confusion_ax.set_xlabel("Predicted")
        self.confusion_ax.set_ylabel("Actual")
        self.confusion_canvas = FigureCanvasTkAgg(confusion_fig, master=confusion_frame)
        self.confusion_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._make_canvas_responsive(self.confusion_canvas, confusion_fig, min_height=260)
        self.confusion_fig = confusion_fig
        ttk.Button(
            confusion_frame,
            text="Save Figure…",
            command=self._save_confusion_figure,
        ).pack(anchor="e", pady=(4, 0))

        importance_frame = ttk.LabelFrame(
            plots_frame, text="Feature Importance", padding=8
        )
        importance_frame.pack(fill="both", expand=True, pady=(12, 0))

        importance_fig = Figure(figsize=(6, 4), dpi=100)
        self.importance_ax = importance_fig.add_subplot(111)
        self.importance_ax.set_title("Train a model to view feature importance")
        self.importance_ax.set_xlabel("Importance")
        self.importance_ax.set_ylabel("Feature")
        self.importance_canvas = FigureCanvasTkAgg(
            importance_fig, master=importance_frame
        )
        self.importance_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._make_canvas_responsive(self.importance_canvas, importance_fig, min_height=240)
        self.importance_fig = importance_fig
        ttk.Button(
            importance_frame,
            text="Save Figure…",
            command=self._save_training_importance_figure,
        ).pack(anchor="e", pady=(4, 0))

    def _init_run_registry_tab(self, notebook: ttk.Notebook) -> None:
        container = ttk.Frame(notebook)
        notebook.add(container, text="Run Registry")
        scroll = ScrollableFrame(container)
        scroll.pack(fill="both", expand=True)
        registry_tab = ttk.Frame(scroll.scrollable_frame, padding=12)
        registry_tab.pack(fill="both", expand=True)

        ttk.Label(
            registry_tab,
            text=(
                "Each completed training run is captured here with accuracy, configuration, "
                "and any tags/notes you supply. Select a run to view details or restore "
                "its configuration."
            ),
            wraplength=800,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        columns = ("timestamp", "model", "target", "accuracy", "tags")
        tree = ttk.Treeview(
            registry_tab,
            columns=columns,
            show="headings",
            height=12,
        )
        tree.heading("timestamp", text="Timestamp")
        tree.heading("model", text="Model")
        tree.heading("target", text="Target")
        tree.heading("accuracy", text="Accuracy")
        tree.heading("tags", text="Tags")
        tree.column("timestamp", width=180, anchor="w")
        tree.column("model", width=140, anchor="w")
        tree.column("target", width=140, anchor="w")
        tree.column("accuracy", width=100, anchor="center")
        tree.column("tags", width=200, anchor="w")
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", lambda _e: self._on_run_registry_select())
        scrollbar = ttk.Scrollbar(registry_tab, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.run_registry_tree = tree

        detail_frame = ttk.LabelFrame(registry_tab, text="Run Details", padding=8)
        detail_frame.pack(fill="both", expand=True, pady=(12, 0))
        detail_text = tk.Text(
            detail_frame,
            height=8,
            wrap="word",
            state="disabled",
            background=self.root.cget("background"),
            relief="flat",
        )
        detail_text.pack(fill="both", expand=True)
        self.run_detail_text = detail_text

        button_row = ttk.Frame(detail_frame)
        button_row.pack(fill="x", pady=(6, 0))
        ttk.Button(
            button_row,
            text="Restore Configuration",
            command=self._restore_selected_run,
        ).pack(side="left")

    def _load_run_registry(self) -> None:
        try:
            if self.run_registry_path.exists():
                with self.run_registry_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.run_registry = data.get("runs", [])
            else:
                self.run_registry = []
        except Exception:
            self.run_registry = []
        self._update_run_registry_view()

    def _save_run_registry(self) -> None:
        try:
            with self.run_registry_path.open("w", encoding="utf-8") as handle:
                json.dump({"runs": self.run_registry}, handle, indent=2)
        except Exception:
            pass

    def _update_run_registry_view(self) -> None:
        if not hasattr(self, "run_registry_tree") or self.run_registry_tree is None:
            return
        tree = self.run_registry_tree
        tree.delete(*tree.get_children())
        for record in sorted(self.run_registry, key=lambda item: item.get("timestamp", 0), reverse=True):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.get("timestamp", 0)))
            accuracy = record.get("accuracy")
            accuracy_str = f"{accuracy:.3f}" if isinstance(accuracy, (int, float)) else "-"
            tags = ", ".join(record.get("tags", []))
            tree.insert(
                "",
                "end",
                iid=record.get("id"),
                values=(timestamp, record.get("model_name"), record.get("target"), accuracy_str, tags),
            )
        if hasattr(self, "run_detail_text") and self.run_detail_text is not None:
            self.run_detail_text.configure(state="normal")
            self.run_detail_text.delete("1.0", tk.END)
            self.run_detail_text.insert("1.0", "Select a run to view details.")
            self.run_detail_text.configure(state="disabled")

    def _on_run_registry_select(self) -> None:
        if not self.run_registry_tree or self.run_detail_text is None:
            return
        selection = self.run_registry_tree.selection()
        if not selection:
            return
        record = self._find_run_record(selection[0])
        if not record:
            return
        lines = [
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.get('timestamp', 0)))}",
            f"Model: {record.get('model_name')}",
            f"Target: {record.get('target')}",
            f"Accuracy: {record.get('accuracy'):.3f}" if isinstance(record.get('accuracy'), (int, float)) else "Accuracy: -",
            f"Macro F1: {record.get('f1_macro'):.3f}" if isinstance(record.get('f1_macro'), (int, float)) else "Macro F1: -",
            f"Class balance: {record.get('class_balance', 'None')}",
            f"Tags: {', '.join(record.get('tags', [])) or '-'}",
            f"Notes: {record.get('notes') or '-'}",
            f"Features ({len(record.get('features', []))}): {', '.join(record.get('features', [])[:25])}" + ("…" if len(record.get('features', [])) > 25 else ""),
        ]
        self.run_detail_text.configure(state="normal")
        self.run_detail_text.delete("1.0", tk.END)
        self.run_detail_text.insert("1.0", "\n".join(lines))
        self.run_detail_text.configure(state="disabled")

    def _restore_selected_run(self) -> None:
        if not self.run_registry_tree:
            return
        selection = self.run_registry_tree.selection()
        if not selection:
            messagebox.showinfo("Select run", "Select a run to restore its configuration.")
            return
        record = self._find_run_record(selection[0])
        if not record:
            messagebox.showerror("Missing run", "Could not find the selected run in the registry.")
            return
        self._restore_run_configuration(record)

    def _find_run_record(self, run_id: str) -> Optional[Dict[str, object]]:
        for record in self.run_registry:
            if record.get("id") == run_id:
                return record
        return None

    def _restore_run_configuration(self, record: Dict[str, object]) -> None:
        features = record.get("features", [])
        available_map = {name: idx for idx, name in enumerate(self.available_training_columns)}
        if features and hasattr(self, "training_listbox"):
            self.training_listbox.selection_clear(0, tk.END)
            for feature in features:
                idx = available_map.get(feature)
                if idx is not None:
                    self.training_listbox.selection_set(idx)
            self._on_training_selection_changed()
        target = record.get("target")
        if target and target in (self.target_combo["values"] or []):
            self.target_column_var.set(target)
            self.on_target_column_selected()
        class_balance = record.get("class_balance")
        if class_balance:
            self.class_balance_var.set(class_balance)
        model_name = record.get("model_name")
        if model_name and model_name in self.training_model_configs:
            self.training_model_var.set(model_name)
            self._on_training_model_changed()
            self._apply_model_params(model_name, record.get("model_params", {}))
        self._mark_session_dirty()

    def _apply_model_params(self, model_name: str, params: Dict[str, object]) -> None:
        if model_name == "Random Forest":
            self.n_estimators_var.set(params.get("n_estimators", self.n_estimators_var.get()))
            self.max_depth_var.set("" if params.get("max_depth") is None else str(params.get("max_depth")))
            self.max_features_var.set(str(params.get("max_features", self.max_features_var.get())))
            self.min_samples_leaf_var.set(params.get("min_samples_leaf", self.min_samples_leaf_var.get()))
        elif model_name == "LDA":
            self.lda_solver_var.set(params.get("solver", self.lda_solver_var.get()))
            self.lda_shrinkage_var.set("" if params.get("shrinkage") in (None, "") else str(params.get("shrinkage")))
        elif model_name == "SVM":
            self.svm_kernel_var.set(params.get("kernel", self.svm_kernel_var.get()))
            self.svm_c_var.set(params.get("C", self.svm_c_var.get()))
            self.svm_gamma_var.set(str(params.get("gamma", self.svm_gamma_var.get())))
            self.svm_degree_var.set(params.get("degree", self.svm_degree_var.get()))
        elif model_name == "Logistic Regression":
            self.lr_solver_var.set(params.get("solver", self.lr_solver_var.get()))
            self.lr_penalty_var.set(params.get("penalty", self.lr_penalty_var.get()))
            self.lr_c_var.set(params.get("C", self.lr_c_var.get()))
            self.lr_max_iter_var.set(params.get("max_iter", self.lr_max_iter_var.get()))
            self.lr_l1_ratio_var.set(params.get("l1_ratio", self.lr_l1_ratio_var.get()))
        elif model_name == "Naive Bayes":
            self.nb_var_smoothing_var.set(params.get("var_smoothing", self.nb_var_smoothing_var.get()))
        elif model_name == "XGBoost":
            self.xgb_estimators_var.set(params.get("n_estimators", self.xgb_estimators_var.get()))
            self.xgb_learning_rate_var.set(params.get("learning_rate", self.xgb_learning_rate_var.get()))
            self.xgb_max_depth_var.set(params.get("max_depth", self.xgb_max_depth_var.get()))
            self.xgb_subsample_var.set(params.get("subsample", self.xgb_subsample_var.get()))
            self.xgb_colsample_var.set(params.get("colsample_bytree", self.xgb_colsample_var.get()))
        elif model_name == "LightGBM":
            self.lgb_estimators_var.set(params.get("n_estimators", self.lgb_estimators_var.get()))
            self.lgb_learning_rate_var.set(params.get("learning_rate", self.lgb_learning_rate_var.get()))
            self.lgb_max_depth_var.set(params.get("max_depth", self.lgb_max_depth_var.get()))
            self.lgb_num_leaves_var.set(params.get("num_leaves", self.lgb_num_leaves_var.get()))
            self.lgb_subsample_var.set(params.get("subsample", self.lgb_subsample_var.get()))

    def _record_training_run(self, payload: Dict[str, object]) -> None:
        metrics = payload.get("metrics", {})
        record = {
            "id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "model_name": payload.get("model_name"),
            "target": payload.get("target"),
            "features": payload.get("features", []),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "model_params": payload.get("config", {}).get("model_params", {}),
            "class_balance": self.class_balance_var.get(),
            "tags": self._parse_tags(self.run_tags_var.get()),
            "notes": self.run_notes_var.get().strip(),
            "seed": RANDOM_STATE,
        }
        self.run_registry.append(record)
        self._save_run_registry()
        self._update_run_registry_view()
        self.run_notes_var.set("")

    @staticmethod
    def _parse_tags(raw: str) -> List[str]:
        return [tag.strip() for tag in raw.split(",") if tag.strip()]
    def _update_training_controls(self) -> None:
        if not hasattr(self, "training_listbox"):
            return

        self.training_listbox.delete(0, tk.END)
        self.available_training_columns.clear()
        self.training_missing_var.set("")
        self.training_selection = []
        self.target_unique_counts.clear()
        self.target_unique_capped.clear()

        if not self.data_files:
            self.training_listbox.configure(state="disabled")
            self.training_hint_var.set("Load files to populate feature options.")
            self.target_combo["values"] = []
            self.target_column_var.set("")
            self.target_info_var.set(
                "Select a classification column to view category statistics."
            )
            self.target_missing_var.set("")
            self._clear_category_tables()
            self._update_category_chart()
            self._clear_training_downsampling_preview()
            return

        total_files = len(self.data_files)
        self.training_listbox.configure(state="normal")

        sorted_columns = sorted(
            self.column_presence.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )

        for column_name, presence in sorted_columns:
            self.available_training_columns.append(column_name)
            self.training_listbox.insert(tk.END, column_name)
            index = self.training_listbox.size() - 1
            if presence < total_files:
                self.training_listbox.itemconfig(index, foreground="#B22222")

        self.training_hint_var.set(
            f"{len(self.common_columns)} columns are available across all {total_files} files."
        )

        # Default to selecting all common columns.
        self.training_listbox.selection_clear(0, tk.END)
        for index, column_name in enumerate(self.available_training_columns):
            if column_name in self.common_columns:
                self.training_listbox.selection_set(index)
        self._on_training_selection_changed()

        candidate_names = sorted(self.common_columns, key=str.lower)
        self.target_combo["values"] = candidate_names

        previous_target = self.target_column_var.get()
        if previous_target in candidate_names:
            # Re-run statistics for the retained target.
            self.on_target_column_selected()
        else:
            self.target_column_var.set("")
            self.target_info_var.set(
                "Select a classification column to view category statistics."
            )
            self.target_missing_var.set("")
            self._clear_category_tables()
            self._update_category_chart()

        self._clear_training_downsampling_preview()

    def _update_clustering_controls(self) -> None:
        if not hasattr(self, "clustering_listbox"):
            return

        self.clustering_listbox.delete(0, tk.END)
        self.clustering_available_columns.clear()
        self.clustering_missing_var.set("")
        self.clustering_selection = []
        self.clustering_categorical_options = {}
        self.clustering_numeric_ranges = {}

        total_rows = sum(data_file.row_count for data_file in self.data_files)
        self.clustering_total_rows_var.set(f"Total cells: {total_rows}")

        if not self.data_files:
            self.clustering_listbox.configure(state="disabled")
            self.clustering_hint_var.set("Load files to populate clustering features.")
            self.clustering_class_combo["values"] = []
            self.clustering_class_var.set("")
            if hasattr(self, "clustering_filter_column_combo"):
                self.clustering_filter_column_combo["values"] = []
                self.clustering_filter_column_var.set("")
            self.clustering_filters.clear()
            self._refresh_clustering_filter_tree()
            self._update_clustering_subset_summary()
            self._clear_clustering_downsampling_preview()
            return

        self.clustering_listbox.configure(state="normal")

        sorted_columns = sorted(
            self.column_presence.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )

        total_files = len(self.data_files)
        numeric_common: List[str] = [
            column_name
            for column_name in self.common_columns
            if self.column_numeric_hints.get(column_name, False)
        ]

        for column_name, presence in sorted_columns:
            self.clustering_available_columns.append(column_name)
            self.clustering_listbox.insert(tk.END, column_name)
            index = self.clustering_listbox.size() - 1
            if presence < total_files:
                self.clustering_listbox.itemconfig(index, foreground="#B22222")

        if numeric_common:
            self.clustering_hint_var.set(
                f"{len(numeric_common)} numeric columns available across all {total_files} files."
            )
        else:
            self.clustering_hint_var.set(
                "No numeric columns are shared across every file. Select available features carefully."
            )

        self.clustering_listbox.selection_clear(0, tk.END)
        for index, column_name in enumerate(self.clustering_available_columns):
            if column_name in numeric_common:
                self.clustering_listbox.selection_set(index)
        self._on_clustering_selection_changed()

        filter_columns = sorted(self.column_presence.keys(), key=str.lower)
        if hasattr(self, "clustering_filter_column_combo"):
            self.clustering_filter_column_combo["values"] = filter_columns
            if self.clustering_filter_column_var.get() not in filter_columns:
                default_column = filter_columns[0] if filter_columns else ""
                self.clustering_filter_column_var.set(default_column)
            if not filter_columns:
                self.clustering_filter_current_values = []
                if hasattr(self, "clustering_filter_values_listbox"):
                    self.clustering_filter_values_listbox.delete(0, tk.END)

        available_filter_cols = set(filter_columns)
        if available_filter_cols:
            self.clustering_filters = [
                filt for filt in self.clustering_filters if filt["column"] in available_filter_cols
            ]
        else:
            self.clustering_filters.clear()
        self._refresh_clustering_filter_tree()

        categorical_candidates = [
            column_name
            for column_name in self.common_columns
            if not self.column_numeric_hints.get(column_name, False)
        ]
        categorical_candidates = sorted(categorical_candidates, key=str.lower)
        self.clustering_class_combo["values"] = categorical_candidates

        previous_selection = self.clustering_class_var.get()
        if previous_selection not in categorical_candidates:
            self.clustering_class_var.set(
                categorical_candidates[0] if categorical_candidates else ""
            )

        self._on_clustering_downsample_method_changed()
        self._on_clustering_filter_column_selected()
        self._update_clustering_subset_summary()
        self._clear_clustering_downsampling_preview()

    def _refresh_clustering_filter_tree(self) -> None:
        if not hasattr(self, "clustering_filter_tree"):
            return
        self.clustering_filter_tree.delete(*self.clustering_filter_tree.get_children())
        for filt in self.clustering_filters:
            if filt["type"] == "categorical":
                values = filt.get("values", [])
                labels = [self._format_filter_value(value) for value in values]
                if len(labels) > 6:
                    criteria = ", ".join(labels[:6]) + f", +{len(labels) - 6}"
                else:
                    criteria = ", ".join(labels)
            else:
                min_val = filt.get("min")
                max_val = filt.get("max")
                if min_val is None and max_val is None:
                    criteria = "All values"
                elif min_val is None:
                    criteria = f"<= {max_val}"
                elif max_val is None:
                    criteria = f">= {min_val}"
                else:
                    criteria = f"{min_val} to {max_val}"
            self.clustering_filter_tree.insert(
                "",
                "end",
                iid=str(filt["id"]),
                values=(filt["column"], filt["type"], criteria),
            )

    def _update_clustering_subset_summary(self) -> None:
        if not self.data_files:
            self.clustering_subset_rows_var.set("Filtered rows: n/a")
            return
        try:
            combined = self._combined_dataframe()
            filtered = self._get_filtered_dataframe_base(use_cached_combined=combined)
        except ValueError:
            self.clustering_subset_rows_var.set("Filtered rows: error")
            return
        total = len(combined)
        remaining = len(filtered)
        if remaining == total:
            self.clustering_subset_rows_var.set(f"Filtered rows: {remaining} (all rows)")
        else:
            self.clustering_subset_rows_var.set(
                f"Filtered rows: {remaining} (out of {total})"
            )

    def _get_filtered_dataframe_base(
        self, use_cached_combined: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        combined = (
            use_cached_combined.copy()
            if use_cached_combined is not None
            else self._combined_dataframe()
        )
        if not self.clustering_filters:
            return combined
        mode = (self.clustering_filter_mode_var.get() or "AND").upper()
        if mode not in {"AND", "OR"}:
            mode = "AND"

        if mode == "OR":
            combined_mask = pd.Series(False, index=combined.index)
        else:
            combined_mask = pd.Series(True, index=combined.index)

        applied = False
        for filt in self.clustering_filters:
            mask = self._evaluate_clustering_filter_mask(combined, filt)
            if mask is None:
                continue
            mask = mask.reindex(combined.index, fill_value=False)
            applied = True
            if mode == "OR":
                combined_mask = combined_mask | mask
            else:
                combined_mask = combined_mask & mask

        if not applied:
            return combined

        return combined[combined_mask]

    def _evaluate_clustering_filter_mask(
        self, dataframe: pd.DataFrame, filt: Dict[str, Any]
    ) -> Optional[pd.Series]:
        column = filt.get("column")
        if not column or column not in dataframe.columns:
            return None
        series = dataframe[column]
        if filt.get("type") == "categorical":
            values = filt.get("values") or []
            if not values:
                return None
            mask = series.isin(values)
            return mask.fillna(False)

        if filt.get("type") == "numeric":
            working = series
            if not is_numeric_dtype(working):
                working = pd.to_numeric(working, errors="coerce")
            mask = pd.Series(True, index=series.index)
            min_val = filt.get("min")
            max_val = filt.get("max")
            if min_val is not None:
                mask &= working >= min_val
            if max_val is not None:
                mask &= working <= max_val
            mask &= working.notna()
            return mask.fillna(False)

        return None

    def _on_clustering_filter_column_selected(self, _event: Optional[tk.Event] = None) -> None:
        if not hasattr(self, "clustering_filter_categorical_frame"):
            return
        column = self.clustering_filter_column_var.get()
        self.clustering_filter_categorical_frame.grid_forget()
        self.clustering_filter_numeric_frame.grid_forget()
        if not column:
            return
        self.clustering_filter_current_values = []
        prefers_numeric = self.column_numeric_hints.get(column, False)
        showed_ui = False
        if prefers_numeric and self._ensure_numeric_range(column):
            ranges = self.clustering_numeric_ranges.get(column, {})
            min_val = ranges.get("min")
            max_val = ranges.get("max")
            self.clustering_filter_min_entry.configure(
                from_=(min_val if min_val is not None else -1e12),
                to=(max_val if max_val is not None else 1e12),
                increment=0.1,
            )
            self.clustering_filter_max_entry.configure(
                from_=(min_val if min_val is not None else -1e12),
                to=(max_val if max_val is not None else 1e12),
                increment=0.1,
            )
            self.clustering_filter_min_var.set("" if min_val is None else f"{min_val:.6g}")
            self.clustering_filter_max_var.set("" if max_val is None else f"{max_val:.6g}")
            self.clustering_filter_values_label_var.set("Select categories")
            self.clustering_filter_numeric_frame.grid(row=0, column=0, sticky="w")
            showed_ui = True
        if not showed_ui and self._ensure_categorical_options(column):
            values = list(self.clustering_categorical_options.get(column, []))
            self.clustering_filter_current_values = values
            self.clustering_filter_values_listbox.delete(0, tk.END)
            for value in values:
                self.clustering_filter_values_listbox.insert(
                    tk.END, self._format_filter_value(value)
                )
            shown = len(values)
            if shown >= MAX_UNIQUE_CATEGORY_SAMPLE:
                self.clustering_filter_values_label_var.set(
                    f"Select categories (showing first {shown})"
                )
            else:
                self.clustering_filter_values_label_var.set(
                    f"Select categories ({shown} total)"
                )
            self.clustering_filter_values_listbox.selection_clear(0, tk.END)
            self.clustering_filter_categorical_frame.grid(row=0, column=0, sticky="w")
            showed_ui = True
        if not showed_ui and self._ensure_numeric_range(column):
            ranges = self.clustering_numeric_ranges.get(column, {})
            min_val = ranges.get("min")
            max_val = ranges.get("max")
            self.clustering_filter_min_entry.configure(
                from_=(min_val if min_val is not None else -1e12),
                to=(max_val if max_val is not None else 1e12),
                increment=0.1,
            )
            self.clustering_filter_max_entry.configure(
                from_=(min_val if min_val is not None else -1e12),
                to=(max_val if max_val is not None else 1e12),
                increment=0.1,
            )
            self.clustering_filter_min_var.set("" if min_val is None else f"{min_val:.6g}")
            self.clustering_filter_max_var.set("" if max_val is None else f"{max_val:.6g}")
            self.clustering_filter_values_label_var.set("Select categories")
            self.clustering_filter_numeric_frame.grid(row=0, column=0, sticky="w")
            showed_ui = True
        if not showed_ui:
            self.clustering_filter_values_listbox.delete(0, tk.END)
            self.clustering_filter_values_label_var.set("Select categories")

    def _add_clustering_filter(self) -> None:
        column = self.clustering_filter_column_var.get()
        if not column:
            messagebox.showerror("Select column", "Choose a column to filter.")
            return

        has_categorical = column in self.clustering_categorical_options or self._ensure_categorical_options(column)
        has_numeric = column in self.clustering_numeric_ranges or self._ensure_numeric_range(column)

        if has_categorical and self.clustering_filter_current_values:
            selections = self.clustering_filter_values_listbox.curselection()
            if not selections:
                messagebox.showerror("Select values", "Select at least one category.")
                return
            if column == self.clustering_filter_column_var.get() and self.clustering_filter_current_values:
                available_values = self.clustering_filter_current_values
            else:
                available_values = self.clustering_categorical_options.get(column, [])
            values = [
                available_values[idx]
                for idx in selections
                if 0 <= idx < len(available_values)
            ]
            if not values:
                messagebox.showerror("Select values", "Selected categories are unavailable.")
                return
            existing_match = any(
                filt["column"] == column
                and filt["type"] == "categorical"
                and sorted(filt.get("values", []), key=str)
                == sorted(values, key=str)
                for filt in self.clustering_filters
            )
            if existing_match:
                messagebox.showinfo(
                    "Duplicate filter",
                    "An identical categorical filter already exists.",
                )
                return
            filter_entry = {
                "id": self.clustering_filter_counter,
                "column": column,
                "type": "categorical",
                "values": values,
            }
        elif has_numeric:
            min_text = self.clustering_filter_min_var.get().strip()
            max_text = self.clustering_filter_max_var.get().strip()
            min_val = None
            max_val = None
            try:
                if min_text:
                    min_val = float(min_text)
                if max_text:
                    max_val = float(max_text)
            except ValueError:
                messagebox.showerror("Invalid range", "Numeric thresholds must be valid numbers.")
                return
            if min_val is not None and max_val is not None and max_val < min_val:
                messagebox.showerror("Invalid range", "Max value must be greater than or equal to min value.")
                return
            existing_match = any(
                filt["column"] == column
                and filt["type"] == "numeric"
                and filt.get("min") == min_val
                and filt.get("max") == max_val
                for filt in self.clustering_filters
            )
            if existing_match:
                messagebox.showinfo(
                    "Duplicate filter",
                    "An identical numeric filter already exists.",
                )
                return
            filter_entry = {
                "id": self.clustering_filter_counter,
                "column": column,
                "type": "numeric",
                "min": min_val,
                "max": max_val,
            }
        else:
            messagebox.showerror(
                "Unsupported column",
                f"Column '{column}' is not available for filtering.",
            )
            return

        self.clustering_filter_counter += 1
        self.clustering_filters.append(filter_entry)
        self.clustering_filter_values_listbox.selection_clear(0, tk.END)
        self.clustering_filter_min_var.set("")
        self.clustering_filter_max_var.set("")
        self._refresh_clustering_filter_tree()
        self._update_clustering_subset_summary()
        self._clear_clustering_downsampling_preview()

    def _remove_selected_clustering_filter(self) -> None:
        if not hasattr(self, "clustering_filter_tree"):
            return
        selected = self.clustering_filter_tree.selection()
        if not selected:
            return
        selected_id = selected[0]
        self.clustering_filters = [
            filt for filt in self.clustering_filters if str(filt["id"]) != selected_id
        ]
        self._refresh_clustering_filter_tree()
        self._update_clustering_subset_summary()
        self._clear_clustering_downsampling_preview()

    def _on_clustering_filter_logic_changed(self) -> None:
        self._update_clustering_subset_summary()
        self._clear_clustering_downsampling_preview()

    def _select_all_common_features(self) -> None:
        if not self.available_training_columns:
            return
        self.training_listbox.selection_clear(0, tk.END)
        for index, column_name in enumerate(self.available_training_columns):
            if column_name in self.common_columns:
                self.training_listbox.selection_set(index)
        self._on_training_selection_changed()

    def _toggle_training_listbox_selection(self, event: tk.Event) -> str:
        widget = event.widget
        if not isinstance(widget, tk.Listbox):
            return "break"
        index = widget.nearest(event.y)
        if index >= 0:
            if widget.selection_includes(index):
                widget.selection_clear(index)
            else:
                widget.selection_set(index)
        self._on_training_selection_changed()
        return "break"

    def _on_training_selection_changed(self) -> None:
        if not self.data_files:
            self.training_missing_var.set("")
            self._mark_session_dirty()
            return

        selected_indices = self.training_listbox.curselection()
        self.training_selection = [
            self.available_training_columns[index] for index in selected_indices
        ]

        if not self.training_selection:
            self.training_missing_var.set("No feature columns selected.")
            self._mark_session_dirty()
            return

        missing_messages: List[str] = []
        for data_file in self.data_files:
            missing = [
                column for column in self.training_selection if column not in data_file.columns
            ]
            if missing:
                preview = ", ".join(missing[:5])
                if len(missing) > 5:
                    preview += f", +{len(missing) - 5} more"
                missing_messages.append(f"{data_file.name}: {preview}")

        if missing_messages:
            display = "Files missing selected columns:\n" + "\n".join(
                missing_messages[:8]
            )
            if len(missing_messages) > 8:
                display += f"\n...and {len(missing_messages) - 8} more."
            self.training_missing_var.set(display)
        else:
            self.training_missing_var.set("")
        self._mark_session_dirty()

    def _select_all_clustering_common_features(self) -> None:
        if not self.clustering_available_columns:
            return
        self.clustering_listbox.selection_clear(0, tk.END)
        for index, column_name in enumerate(self.clustering_available_columns):
            if column_name in self.common_columns and self.column_numeric_hints.get(column_name, False):
                self.clustering_listbox.selection_set(index)
        self._on_clustering_selection_changed()

    def _toggle_clustering_listbox_selection(self, event: tk.Event) -> str:
        widget = event.widget
        if not isinstance(widget, tk.Listbox):
            return "break"
        index = widget.nearest(event.y)
        if index >= 0:
            if widget.selection_includes(index):
                widget.selection_clear(index)
            else:
                widget.selection_set(index)
        self._on_clustering_selection_changed()
        return "break"

    def _on_clustering_selection_changed(self) -> None:
        if not self.data_files:
            self.clustering_missing_var.set("")
            return

        selected_indices = self.clustering_listbox.curselection()
        self.clustering_selection = [
            self.clustering_available_columns[index] for index in selected_indices
        ]

        if not self.clustering_selection:
            self.clustering_missing_var.set("No feature columns selected.")
            return

        missing_messages: List[str] = []
        for data_file in self.data_files:
            missing = [
                column
                for column in self.clustering_selection
                if column not in data_file.columns
            ]
            if missing:
                preview = ", ".join(missing[:5])
                if len(missing) > 5:
                    preview += f", +{len(missing) - 5} more"
                missing_messages.append(f"{data_file.name}: {preview}")

        if missing_messages:
            display = "Files missing selected columns:\n" + "\n".join(
                missing_messages[:8]
            )
            if len(missing_messages) > 8:
                display += f"\n...and {len(missing_messages) - 8} more."
            self.clustering_missing_var.set(display)
        else:
            self.clustering_missing_var.set("")

    def on_target_column_selected(self, _event: Optional[tk.Event] = None) -> None:
        if not self.data_files:
            return

        target_column = self.target_column_var.get()
        if not target_column:
            self._clear_category_tables()
            self._update_category_chart()
            return

        self._compute_target_stats(target_column)

        unique_count = self.target_unique_counts.get(target_column)
        capped = self.target_unique_capped.get(target_column, False)
        if unique_count is None:
            unique_display = "?"
        elif capped:
            unique_display = f"≥ {unique_count}"
        else:
            unique_display = str(unique_count)
        total_rows = sum(self.target_counts_total.values())
        info_parts = [
            f"Column '{target_column}' has {unique_display} unique values "
            f"across {total_rows} rows."
        ]
        if unique_count and (unique_count > 50 or capped):
            info_parts.append("This is a high number of categories; ensure it is appropriate for classification.")
        self.target_info_var.set(" ".join(info_parts))

        if self.target_missing_files:
            missing_preview = ", ".join(self.target_missing_files[:5])
            if len(self.target_missing_files) > 5:
                missing_preview += f", +{len(self.target_missing_files) - 5} more"
            self.target_missing_var.set(
                f"The target column is missing in: {missing_preview}"
            )
        else:
            self.target_missing_var.set("")

        self._populate_category_tables()
        self._update_category_chart()
        self._clear_training_downsampling_preview()
        self._mark_session_dirty()

    def _compute_target_stats(self, target_column: str) -> None:
        total_counts: Dict[str, int] = defaultdict(int)
        per_file_counts: Dict[str, Dict[str, int]] = {}
        missing_files: List[str] = []
        unique_values: Set[str] = set()
        unique_capped = False

        for data_file in self.data_files:
            if not data_file.has_column(target_column):
                missing_files.append(data_file.name)
                continue

            per_file_counter: Dict[str, int] = defaultdict(int)
            for series in self._stream_column_series(target_column, data_files=[data_file]):
                counts_series = series.value_counts(dropna=False)
                for value, count in counts_series.items():
                    label = self._format_filter_value(value)
                    per_file_counter[label] += int(count)
                    total_counts[label] += int(count)
                    if not unique_capped:
                        unique_values.add(label)
                        if len(unique_values) > MAX_UNIQUE_CATEGORY_SAMPLE:
                            unique_capped = True

            per_file_counts[data_file.name] = dict(
                sorted(per_file_counter.items(), key=lambda item: (-item[1], item[0]))
            )

        self.target_counts_total = dict(
            sorted(total_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        self.target_counts_per_file = per_file_counts
        self.target_missing_files = missing_files
        unique_count = len(unique_values)
        if unique_capped and unique_count > MAX_UNIQUE_CATEGORY_SAMPLE:
            unique_count = MAX_UNIQUE_CATEGORY_SAMPLE
        self.target_unique_counts[target_column] = unique_count
        self.target_unique_capped[target_column] = unique_capped

    def _populate_category_tables(self) -> None:
        self.category_totals_tree.delete(*self.category_totals_tree.get_children())
        for category, count in self.target_counts_total.items():
            self.category_totals_tree.insert("", "end", values=(category, count))

        self.category_per_file_tree.delete(
            *self.category_per_file_tree.get_children()
        )
        for file_name, counts in self.target_counts_per_file.items():
            for category, count in counts.items():
                self.category_per_file_tree.insert(
                    "",
                    "end",
                    values=(file_name, category, count),
                )

    def _update_category_chart(self) -> None:
        self.category_ax.clear()

        if not self.target_counts_total:
            self.category_ax.set_title("Select a target column to view category counts")
            self.category_ax.set_xlabel("Category")
            self.category_ax.set_ylabel("Rows")
            self.category_canvas.draw_idle()
            return

        categories = list(self.target_counts_total.keys())
        counts = [self.target_counts_total[category] for category in categories]
        self.category_ax.bar(categories, counts, color="#55A868")
        self.category_ax.set_ylabel("Rows")
        self.category_ax.set_xlabel("Category")
        self.category_ax.set_title(f"{self.target_column_var.get()} distribution")
        self.category_ax.tick_params(axis="x", labelrotation=35)
        self._finalize_figure_layout(self.category_fig, bottom=0.32)
        self.category_canvas.draw_idle()

    def _clear_category_tables(self) -> None:
        self.target_counts_total = {}
        self.target_counts_per_file = {}
        self.target_missing_files = []
        self.category_totals_tree.delete(*self.category_totals_tree.get_children())
        self.category_per_file_tree.delete(*self.category_per_file_tree.get_children())

    def _on_training_downsample_method_changed(self) -> None:
        method = self.training_downsample_method_var.get()
        method_to_label = {
            "None": "Target size",
            "Total Count": "Total rows",
            "Per File": "Rows per file",
            "Per Class": "Rows per category",
            "Per File + Class": "Rows per file/category",
        }
        label = method_to_label.get(method, "Target size")
        self.training_downsample_value_label_var.set(label)

        entry_state = "disabled" if method == "None" else "normal"
        self.training_downsample_value_entry.configure(state=entry_state)
        if method == "None":
            self.training_downsample_value_var.set("")

    def _training_preview_downsampling(self) -> None:
        if not self.data_files:
            messagebox.showinfo("No data", "Load files before configuring downsampling.")
            return

        method = self.training_downsample_method_var.get()
        if method != "None":
            value_str = self.training_downsample_value_var.get().strip()
            if not value_str.isdigit():
                messagebox.showerror(
                    "Invalid value",
                    "Enter a positive integer for the downsampling target.",
                )
                return
            target_value = int(value_str)
            if target_value <= 0:
                messagebox.showerror(
                    "Invalid value",
                    "The target value must be greater than zero.",
                )
                return
        else:
            target_value = 0

        if method in {"Per Class", "Per File + Class"}:
            target_column = self.target_column_var.get()
            if not target_column:
                messagebox.showerror(
                    "Target required",
                    "Select a classification target before using class-based downsampling.",
                )
                return
            if self.target_missing_files:
                messagebox.showerror(
                    "Target missing",
                    "The target column is not present in all files. Resolve missing columns before using this downsampling strategy.",
                )
                return

        try:
            downsampled, message = self._downsample_dataset(
                method=method,
                target_value=target_value,
                class_column=self.target_column_var.get()
                if method in {"Per Class", "Per File + Class"}
                else None,
            )
        except ValueError as exc:
            messagebox.showerror("Downsampling error", str(exc))
            return

        self.training_downsampled_df = downsampled
        self.training_downsample_message_var.set(message)
        self._populate_training_downsample_tables(downsampled)

    def _clear_training_downsampling_preview(self) -> None:
        self.training_downsampled_df = None
        if hasattr(self, "training_downsample_file_tree"):
            self.training_downsample_file_tree.delete(
                *self.training_downsample_file_tree.get_children()
            )
        if hasattr(self, "training_downsample_category_tree"):
            self.training_downsample_category_tree.delete(
                *self.training_downsample_category_tree.get_children()
            )
        self.training_downsample_message_var.set(
            "Configure a downsampling strategy to preview its effect."
        )

    def _populate_training_downsample_tables(self, dataframe: pd.DataFrame) -> None:
        self.training_downsample_file_tree.delete(
            *self.training_downsample_file_tree.get_children()
        )
        self.training_downsample_category_tree.delete(
            *self.training_downsample_category_tree.get_children()
        )

        if dataframe.empty:
            return

        file_counts = dataframe["__source_file"].value_counts().to_dict()
        for file_name, count in sorted(file_counts.items(), key=lambda item: item[0].lower()):
            self.training_downsample_file_tree.insert(
                "", "end", values=(file_name, int(count))
            )

        target_column = self.target_column_var.get()
        if target_column and target_column in dataframe.columns:
            counts_series = dataframe[target_column].value_counts(dropna=False)
            for value, count in counts_series.items():
                label = "<NA>" if pd.isna(value) else str(value)
                self.training_downsample_category_tree.insert(
                    "",
                    "end",
                    values=(label, int(count)),
                )

    def _on_clustering_downsample_method_changed(self) -> None:
        method = self.clustering_downsample_method_var.get()
        method_to_label = {
            "None": "Target size",
            "Total Count": "Total rows",
            "Per File": "Rows per file",
            "Per Class": "Rows per category",
            "Per File + Class": "Rows per file/category",
        }
        label = method_to_label.get(method, "Target size")
        self.clustering_downsample_value_label_var.set(label)

        entry_state = "disabled" if method == "None" else "normal"
        self.clustering_downsample_value_entry.configure(state=entry_state)
        if method == "None":
            self.clustering_downsample_value_var.set("")

    def _clustering_preview_downsampling(self) -> None:
        if not self.data_files:
            messagebox.showinfo("No data", "Load files before configuring downsampling.")
            return

        method = self.clustering_downsample_method_var.get()
        if method != "None":
            value_str = self.clustering_downsample_value_var.get().strip()
            try:
                target_value = int(value_str)
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    "Enter a positive integer for the downsampling target.",
                )
                return
            if target_value <= 0:
                messagebox.showerror(
                    "Invalid value",
                    "The target value must be greater than zero.",
                )
                return
        else:
            target_value = 0

        class_column = None
        if method in {"Per Class", "Per File + Class"}:
            class_column = self.clustering_class_var.get()
            if not class_column:
                messagebox.showerror(
                    "Class required",
                    "Select a categorical column before using class-based downsampling.",
                )
                return

        try:
            filtered = self._get_filtered_dataframe_base()
        except ValueError as exc:
            messagebox.showerror("Data error", str(exc))
            return

        if filtered.empty:
            messagebox.showerror(
                "Empty subset",
                "The current filters produced an empty dataset.",
            )
            return

        try:
            downsampled, message = self._downsample_dataset(
                method=method,
                target_value=target_value,
                class_column=class_column,
                dataset=filtered,
            )
        except ValueError as exc:
            messagebox.showerror("Downsampling error", str(exc))
            return

        self.clustering_downsampled_df = downsampled
        self.clustering_downsample_message_var.set(message)
        self._populate_clustering_downsample_tables(
            downsampled, class_column=class_column
        )

    def _clear_clustering_downsampling_preview(self) -> None:
        self.clustering_downsampled_df = None
        if hasattr(self, "clustering_downsample_file_tree"):
            self.clustering_downsample_file_tree.delete(
                *self.clustering_downsample_file_tree.get_children()
            )
        if hasattr(self, "clustering_downsample_category_tree"):
            self.clustering_downsample_category_tree.delete(
                *self.clustering_downsample_category_tree.get_children()
            )
        self.clustering_downsample_message_var.set(
            "Configure a downsampling strategy to preview its effect."
        )

    def _populate_clustering_downsample_tables(
        self, dataframe: pd.DataFrame, class_column: Optional[str]
    ) -> None:
        self.clustering_downsample_file_tree.delete(
            *self.clustering_downsample_file_tree.get_children()
        )
        self.clustering_downsample_category_tree.delete(
            *self.clustering_downsample_category_tree.get_children()
        )

        if dataframe.empty:
            return

        file_counts = dataframe["__source_file"].value_counts().to_dict()
        for file_name, count in sorted(file_counts.items(), key=lambda item: item[0].lower()):
            self.clustering_downsample_file_tree.insert(
                "", "end", values=(file_name, int(count))
            )

        target_column = class_column
        if target_column and target_column in dataframe.columns:
            counts_series = dataframe[target_column].value_counts(dropna=False)
            for value, count in counts_series.items():
                label = "<NA>" if pd.isna(value) else str(value)
                self.clustering_downsample_category_tree.insert(
                    "",
                    "end",
                    values=(label, int(count)),
                )

    def _on_umap_color_mode_changed(self) -> None:
        mode = self.clustering_umap_color_mode_var.get()
        if mode == "marker":
            state = "readonly" if self.clustering_umap_marker_combo["values"] else "disabled"
            self.clustering_umap_marker_combo.configure(state=state)
        else:
            self.clustering_umap_marker_combo.configure(state="disabled")

    def _on_heatmap_norm_changed(self) -> None:
        mode = self.clustering_heatmap_norm_var.get()
        state = "normal" if mode == "fixed" else "disabled"
        self.clustering_heatmap_min_entry.configure(state=state)
        self.clustering_heatmap_max_entry.configure(state=state)

    def _update_clustering_visual_controls(self) -> None:
        method_labels = [
            self.clustering_method_labels.get(key, key) for key in self.clustering_results.keys()
        ]
        for combo in [
            getattr(self, "clustering_umap_method_combo", None),
            getattr(self, "clustering_heatmap_method_combo", None),
        ]:
            if combo is not None:
                combo["values"] = method_labels
        if method_labels:
            if self.clustering_umap_method_var.get() not in method_labels:
                self.clustering_umap_method_var.set(method_labels[0])
            if self.clustering_heatmap_method_var.get() not in method_labels:
                self.clustering_heatmap_method_var.set(method_labels[0])
        else:
            self.clustering_umap_method_var.set("")
            self.clustering_heatmap_method_var.set("")

        markers = self.clustering_features_used or []
        if getattr(self, "clustering_umap_marker_combo", None):
            self.clustering_umap_marker_combo["values"] = markers
            if markers and self.clustering_umap_marker_var.get() not in markers:
                self.clustering_umap_marker_var.set(markers[0])
            elif not markers:
                self.clustering_umap_marker_var.set("")
        if getattr(self, "clustering_heatmap_markers_listbox", None):
            self.clustering_heatmap_markers_listbox.delete(0, tk.END)
            for marker in markers:
                self.clustering_heatmap_markers_listbox.insert(tk.END, marker)
            if markers:
                self.clustering_heatmap_markers_listbox.selection_set(0, tk.END)
        self._on_umap_color_mode_changed()
        self._on_heatmap_norm_changed()
        self._refresh_cluster_explorer_controls()
        self._refresh_annotation_method_choices()

    def _get_method_key_from_label(self, label: str) -> Optional[str]:
        for key, display in self.clustering_method_labels.items():
            if display == label:
                return key
        return None

    def _generate_clustering_umap(self) -> None:
        if umap is None:
            messagebox.showerror(
                "UMAP unavailable",
                "The umap-learn package is not installed. Install 'umap-learn' to enable this feature.",
            )
            return

        method_label = self.clustering_umap_method_var.get()
        method_key = self._get_method_key_from_label(method_label) if method_label else None
        if not method_key or method_key not in self.clustering_results:
            messagebox.showerror("No clustering results", "Run clustering before generating UMAP.")
            return

        assignment_df = self.clustering_results[method_key]
        if assignment_df.empty:
            messagebox.showerror("No data", "Selected clustering result is empty.")
            return

        features = self.clustering_features_used
        if not features:
            messagebox.showerror("No features", "Clustering features are not available.")
            return

        if (
            self.clustering_feature_matrix is not None
            and self.clustering_dataset_cache is not None
            and len(self.clustering_dataset_cache) == len(self.clustering_feature_matrix)
        ):
            scaled = self.clustering_feature_matrix
        else:
            scaled = StandardScaler().fit_transform(
                assignment_df[features].to_numpy(dtype=float, copy=False)
            )

        selected_df = assignment_df
        use_cache = True
        if self.clustering_umap_downsample_var.get():
            max_cells = max(1000, int(self.clustering_umap_max_cells_var.get()))
            if len(assignment_df) > max_cells:
                selected_df = assignment_df.sample(
                    n=max_cells, random_state=RANDOM_STATE
                ).sort_index()
                use_cache = False

        indices = selected_df.index.to_numpy()
        scaled_subset = scaled[indices]

        n_neighbors = max(5, int(self.clustering_umap_neighbors_var.get()))
        min_dist = max(0.0, float(self.clustering_umap_min_dist_var.get()))
        metric = self.clustering_umap_metric_var.get() or "euclidean"
        umap_jobs = max(
            1, min(int(self.clustering_umap_jobs_var.get()), self.total_cpu_cores)
        )
        cache_key = (n_neighbors, round(min_dist, 4), metric)
        embedding: Optional[np.ndarray] = None
        if use_cache:
            embedding = self.clustering_umap_cache.get(cache_key)
            if embedding is not None and len(embedding) != scaled.shape[0]:
                embedding = None

        if embedding is None:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=RANDOM_STATE,
                n_components=2,
                n_jobs=umap_jobs,
            )
            embedding_full = reducer.fit_transform(scaled)
            if use_cache:
                self.clustering_umap_cache[cache_key] = embedding_full
            if use_cache:
                embedding = embedding_full
            else:
                embedding = embedding_full[indices]
        elif not use_cache:
            embedding = embedding[indices]

        ax = self.clustering_umap_ax
        if self.clustering_umap_colorbar is not None:
            try:
                self.clustering_umap_colorbar.remove()
            except Exception:
                pass
            finally:
                self.clustering_umap_colorbar = None
        ax.clear()
        mode = self.clustering_umap_color_mode_var.get()
        if mode == "marker":
            marker = self.clustering_umap_marker_var.get()
            if not marker:
                messagebox.showerror("No marker", "Select a marker to color by expression.")
                return
            if marker not in assignment_df.columns:
                messagebox.showerror("Marker missing", f"Marker '{marker}' not found in dataset.")
                return
            values = selected_df[marker].to_numpy(dtype=float, copy=False)
            n_points = len(selected_df)
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=values,
                cmap="viridis",
                s=max(1.0, float(self.clustering_umap_dot_size_var.get())),
                linewidths=0,
                alpha=max(0.05, min(1.0, float(self.clustering_umap_alpha_var.get()))),
            )
            cbar = self.clustering_umap_fig.colorbar(scatter, ax=ax)
            cbar.set_label(marker)
            self.clustering_umap_colorbar = cbar
            ax.set_title(f"UMAP colored by {marker} (n={n_points})")
        else:
            clusters = selected_df["cluster"].astype(str).to_numpy()
            unique_clusters = sorted(pd.unique(clusters))
            cmap = ListedColormap(cm.tab20.colors[: len(unique_clusters)])
            cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
            colors = [cluster_to_index[c] for c in clusters]
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                cmap=cmap,
                s=max(1.0, float(self.clustering_umap_dot_size_var.get())),
                linewidths=0,
                alpha=max(0.05, min(1.0, float(self.clustering_umap_alpha_var.get()))),
            )
            ax.set_title(f"UMAP colored by clusters (n={len(selected_df)})")
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    color=cmap(cluster_to_index[c]),
                    label=c,
                )
                for c in unique_clusters
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Cluster")
            if self.clustering_umap_show_labels_var.get():
                if len(unique_clusters) > MAX_CENTROID_LABELS:
                    if hasattr(self, "status_var"):
                        self.status_var.set(
                            f"Centroid labels hidden: {len(unique_clusters)} clusters exceeds limit ({MAX_CENTROID_LABELS})."
                        )
                else:
                    for cluster_label in unique_clusters:
                        mask = clusters == cluster_label
                        if mask.any():
                            centroid = embedding[mask].mean(axis=0)
                            ax.text(
                                centroid[0],
                                centroid[1],
                                cluster_label,
                                fontsize=9,
                                ha="center",
                                va="center",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])
        self._finalize_figure_layout(self.clustering_umap_fig)
        self.clustering_umap_canvas.draw_idle()

    def _generate_clustering_heatmap(self) -> None:
        method_label = self.clustering_heatmap_method_var.get()
        method_key = self._get_method_key_from_label(method_label) if method_label else None
        if not method_key or method_key not in self.clustering_results:
            messagebox.showerror("No clustering results", "Run clustering before generating heatmaps.")
            return

        assignment_df = self.clustering_results[method_key]
        if assignment_df.empty:
            messagebox.showerror("No data", "Selected clustering result is empty.")
            return

        markers = self.clustering_features_used or []
        selected_indices = self.clustering_heatmap_markers_listbox.curselection()
        if selected_indices:
            markers = [self.clustering_features_used[i] for i in selected_indices]
        if not markers:
            messagebox.showerror("No markers", "Select at least one marker for the heatmap.")
            return

        missing_markers = [m for m in markers if m not in assignment_df.columns]
        if missing_markers:
            messagebox.showerror(
                "Markers missing",
                "The following markers are not available in the result dataset:\n"
                + ", ".join(missing_markers),
            )
            return

        cluster_means = (
            assignment_df.groupby("cluster")[markers].mean().sort_index()
        )
        norm_mode = self.clustering_heatmap_norm_var.get()
        data = cluster_means.copy()
        if norm_mode == "minmax":
            min_vals = data.min()
            max_vals = data.max()
            data = (data - min_vals) / (max_vals - min_vals + 1e-9)
        elif norm_mode == "fixed":
            min_val = float(self.clustering_heatmap_min_var.get())
            max_val = float(self.clustering_heatmap_max_var.get())
            if max_val <= min_val:
                messagebox.showerror(
                    "Invalid range", "Heatmap max must be greater than min."
                )
                return
            data = data.clip(lower=min_val, upper=max_val)

        cluster_dendro = self.clustering_heatmap_cluster_dendro_var.get()
        marker_dendro = self.clustering_heatmap_marker_dendro_var.get()

        n_clusters = max(1, data.shape[0])
        n_markers = max(1, data.shape[1])
        base_width = max(6.0, min(n_markers * 0.6, 24.0))
        base_height = max(4.0, min(n_clusters * 0.45, 24.0))
        width = base_width + (2.0 if marker_dendro else 0.0)
        height = base_height + (2.0 if cluster_dendro else 0.0)
        linewidth = 0.2
        linecolor = "#333333"
        self.clustering_heatmap_colorbar = None

        if cluster_dendro or marker_dendro:
            try:
                dendrogram_ratio = (
                    0.18 if cluster_dendro else 0.05,
                    0.18 if marker_dendro else 0.05,
                )
                g = sns.clustermap(
                    data,
                    cmap="viridis",
                    row_cluster=cluster_dendro,
                    col_cluster=marker_dendro,
                    method="average",
                    figsize=(width, height),
                    linewidths=linewidth,
                    linecolor=linecolor,
                    dendrogram_ratio=dendrogram_ratio,
                    cbar_kws={"label": "Expression"},
                )
            except Exception as exc:
                messagebox.showwarning(
                    "Heatmap clustering failed",
                    f"Failed to compute dendrogram heatmap: {exc}\nFalling back to standard heatmap.",
                )
            else:
                g.ax_heatmap.set_title(f"{method_label} cluster heatmap")
                g.ax_heatmap.set_xlabel("Marker")
                g.ax_heatmap.set_ylabel("Cluster")
                g.ax_heatmap.tick_params(axis="x", rotation=45)
                self._update_heatmap_canvas(g.fig)
                self.clustering_heatmap_ax = g.ax_heatmap
                if g.ax_heatmap.collections:
                    self.clustering_heatmap_colorbar = (
                        g.ax_heatmap.collections[0].colorbar
                    )
                else:
                    self.clustering_heatmap_colorbar = None
                if self.clustering_heatmap_colorbar is not None:
                    self.clustering_heatmap_colorbar.set_label("Expression")
                self.clustering_heatmap_canvas.draw_idle()
                return

        figure = Figure(figsize=(width, height), dpi=100)
        ax = figure.add_subplot(111)
        heatmap = sns.heatmap(
            data,
            ax=ax,
            cmap="viridis",
            cbar=True,
            linewidths=linewidth,
            linecolor=linecolor,
        )

        ax.set_xlabel("Marker")
        ax.set_ylabel("Cluster")
        ax.set_title(f"{method_label} cluster heatmap")
        ax.tick_params(axis="x", rotation=45)
        if heatmap.collections:
            self.clustering_heatmap_colorbar = heatmap.collections[0].colorbar
        if self.clustering_heatmap_colorbar is not None:
            self.clustering_heatmap_colorbar.set_label("Expression")
        self._finalize_figure_layout(figure, bottom=0.32)
        self._update_heatmap_canvas(figure)
        self.clustering_heatmap_ax = ax
        self.clustering_heatmap_canvas.draw_idle()

    @staticmethod
    def _sanitize_for_column(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
        return sanitized.strip("_") or "value"

    def _clustering_worker(
        self,
        dataset: pd.DataFrame,
        features_scaled: np.ndarray,
        selected_methods: List[
            tuple[str, str, Dict[str, object], Dict[str, object], str]
        ],
        n_jobs: int,
        base_metadata: Dict[str, object],
    ) -> None:
        try:
            base_metadata = dict(base_metadata)
            results_summary: List[Dict[str, object]] = []
            cluster_breakdown: List[Dict[str, object]] = []
            assignments: Dict[str, pd.DataFrame] = {}
            errors: List[str] = []
            metadata: Dict[str, Dict[str, object]] = {}
            label_map: Dict[str, str] = {}

            max_workers = max(1, min(len(selected_methods), n_jobs))

            def run_single(method_tuple):
                run_key, method_key, method_info, params, run_label = method_tuple
                label = run_label
                start_time = time.time()
                labels = self._run_clustering_method(
                    method_key, features_scaled, params, n_jobs
                )
                elapsed = time.time() - start_time
                labels_array = np.asarray(labels)
                if labels_array.shape[0] != dataset.shape[0]:
                    raise RuntimeError(
                        f"{label}: output size mismatch ({labels_array.shape[0]} vs {dataset.shape[0]})."
                    )

                label_series = pd.Series(labels_array, name="cluster")
                counts = (
                    label_series.value_counts(dropna=False)
                    .sort_index()
                    .to_dict()
                )
                summary_entry = {
                    "method_key": method_key,
                    "method_label": label,
                    "run_key": run_key,
                    "cluster_count": len(counts),
                    "rows": len(dataset),
                }
                cluster_entries = [
                    {
                        "run_key": run_key,
                        "method_key": method_key,
                        "method_label": label,
                        "cluster": str(cluster_value),
                        "count": int(cluster_size),
                    }
                    for cluster_value, cluster_size in counts.items()
                ]
                assignment_df = dataset.copy()
                assignment_df["cluster"] = labels_array

                cluster_sizes = {
                    str(cluster_value): int(cluster_size)
                    for cluster_value, cluster_size in counts.items()
                }
                meta_entry = {
                    "run_key": run_key,
                    "method_key": method_key,
                    "method_label": label,
                    "params": params.copy(),
                    "cluster_count": len(cluster_sizes),
                    "rows_used": len(dataset),
                    "elapsed_sec": elapsed,
                    "cluster_sizes": cluster_sizes,
                }
                meta_entry["params"] = self._to_serializable(meta_entry["params"])
                return summary_entry, cluster_entries, assignment_df, meta_entry

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single, method_tuple): method_tuple
                    for method_tuple in selected_methods
                }
                for future in as_completed(futures):
                    run_key, method_key, method_info, params, run_label = futures[future]
                    label = run_label
                    try:
                        (
                            summary_entry,
                            cluster_entries,
                            assignment_df,
                            meta_entry,
                        ) = future.result()
                        results_summary.append(summary_entry)
                        cluster_breakdown.extend(cluster_entries)
                        assignments[run_key] = assignment_df
                        metadata_entry = base_metadata.copy()
                        metadata_entry.update(meta_entry)
                        metadata[run_key] = self._to_serializable(metadata_entry)
                        label_map[run_key] = run_label
                    except ImportError as exc:
                        errors.append(f"{label}: missing dependency ({exc}).")
                        params_serializable = self._to_serializable(params)
                        metadata_entry = base_metadata.copy()
                        metadata_entry.update(
                            {
                                "run_key": run_key,
                                "method_key": method_key,
                                "method_label": label,
                                "params": params_serializable,
                                "error": str(exc),
                            }
                        )
                        metadata[run_key] = self._to_serializable(metadata_entry)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{label}: {exc}")
                        params_serializable = self._to_serializable(params)
                        metadata_entry = base_metadata.copy()
                        metadata_entry.update(
                            {
                                "run_key": run_key,
                                "method_key": method_key,
                                "method_label": label,
                                "params": params_serializable,
                                "error": str(exc),
                            }
                        )
                        metadata[run_key] = self._to_serializable(metadata_entry)

            payload = {
                "summary": results_summary,
                "clusters": cluster_breakdown,
                "assignments": assignments,
                "errors": errors,
                "features": list(self.clustering_selection),
                "metadata": metadata,
                "base_metadata": self._to_serializable(base_metadata),
                "labels": label_map,
            }
            self.clustering_queue.put({"status": "success", "payload": payload})
        except Exception as exc:  # noqa: BLE001
            self.clustering_queue.put({"status": "error", "message": str(exc)})

    def _run_clustering_method(
        self,
        method_key: str,
        features_scaled: np.ndarray,
        params: Dict[str, object],
        n_jobs: int,
    ) -> np.ndarray:
        if method_key == "kmeans":
            n_clusters = max(2, int(params["n_clusters"]))
            max_iter = max(10, int(params["max_iter"]))
            n_init = max(1, int(params["n_init"]))
            model = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                n_init=n_init,
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(features_scaled)
            return labels

        if method_key == "leiden":
            if ig is None or leidenalg is None:
                raise ImportError("leidenalg (with python-igraph)")
            n_neighbors = max(2, int(params["n_neighbors"]))
            resolution = float(params["resolution"])
            graph = self._build_knn_graph(features_scaled, n_neighbors, n_jobs)
            ig_graph = self._igraph_from_sparse(graph)
            partition = leidenalg.RBConfigurationVertexPartition(
                ig_graph,
                weights=ig_graph.es["weight"],
                resolution_parameter=resolution,
            )
            optimiser = leidenalg.Optimiser()
            optimiser.optimise_partition(partition)
            return np.array(partition.membership, dtype=int)

        if method_key == "louvain":
            if nx is None or community_louvain is None:
                raise ImportError("networkx and python-louvain")
            n_neighbors = max(2, int(params["n_neighbors"]))
            resolution = float(params["resolution"])
            graph = self._build_knn_graph(features_scaled, n_neighbors, n_jobs)
            G = nx.from_scipy_sparse_array(graph, edge_attribute="weight")
            partition = community_louvain.best_partition(
                G, weight="weight", resolution=resolution, random_state=RANDOM_STATE
            )
            labels = [partition[node] for node in range(features_scaled.shape[0])]
            return np.array(labels, dtype=int)

        if method_key == "som_metacluster":
            if MiniSom is None:
                raise ImportError("MiniSom")
            grid_x = max(2, int(params["grid_x"]))
            grid_y = max(2, int(params["grid_y"]))
            iterations = max(100, int(params["iterations"]))
            meta_clusters = max(2, int(params["meta_clusters"]))

            som = MiniSom(
                grid_x,
                grid_y,
                features_scaled.shape[1],
                sigma=max(grid_x, grid_y) / 2.0,
                learning_rate=0.5,
                random_seed=RANDOM_STATE,
            )
            som.train_random(features_scaled, iterations)

            bmus = np.array([som.winner(x) for x in features_scaled])
            bmu_indices = np.array([coord[0] * grid_y + coord[1] for coord in bmus])

            codebook = som.get_weights().reshape(grid_x * grid_y, -1)
            kmeans_meta = KMeans(
                n_clusters=meta_clusters, random_state=RANDOM_STATE, n_init=10
            )
            meta_labels = kmeans_meta.fit_predict(codebook)
            labels = meta_labels[bmu_indices]
            return labels

        raise ValueError(f"Clustering method '{method_key}' is not implemented yet.")

    def _build_knn_graph(
        self, X: np.ndarray, n_neighbors: int, n_jobs: int
    ) -> CSRMatrix:
        if sparse is None:
            raise ImportError("scipy")
        n_neighbors = min(max(2, n_neighbors), max(2, X.shape[0] - 1))
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric="euclidean",
            n_jobs=max(1, n_jobs),
        )
        nbrs.fit(X)
        graph = nbrs.kneighbors_graph(mode="distance")
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph

    def _igraph_from_sparse(self, matrix: CSRMatrix) -> "ig.Graph":
        if ig is None:
            raise ImportError("python-igraph")
        coo = matrix.tocoo()
        mask = coo.row < coo.col
        sources = coo.row[mask]
        targets = coo.col[mask]
        weights = coo.data[mask]
        graph = ig.Graph(
            n=matrix.shape[0], edges=list(zip(sources, targets)), directed=False
        )
        graph.es["weight"] = weights.tolist()
        return graph

    def _check_clustering_queue(self) -> None:
        if not self.clustering_in_progress:
            return

        try:
            message = self.clustering_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._check_clustering_queue)
            return

        if message["status"] == "success":
            self._handle_clustering_success(message["payload"])
        else:
            self._handle_clustering_failure(message["message"])

    def _handle_clustering_success(self, payload: Dict[str, object]) -> None:
        self.clustering_in_progress = False
        self.clustering_progress.stop()
        self.run_clustering_button.configure(state="normal")

        errors: List[str] = payload.get("errors", [])  # type: ignore[assignment]
        summary: List[Dict[str, object]] = payload.get("summary", [])  # type: ignore[assignment]
        clusters: List[Dict[str, object]] = payload.get("clusters", [])  # type: ignore[assignment]
        assignments: Dict[str, pd.DataFrame] = payload.get("assignments", {})  # type: ignore[assignment]
        labels_map: Dict[str, str] = payload.get("labels", {})  # type: ignore[assignment]

        self.clustering_results = assignments
        self.clustering_metadata = payload.get("metadata", {})
        base_meta = payload.get("base_metadata")
        if isinstance(base_meta, dict):
            self.clustering_run_metadata_base = base_meta
        else:
            self.clustering_run_metadata_base = {}
        features_meta = payload.get("features")
        if isinstance(features_meta, list) and features_meta:
            self.clustering_features_used = [str(f) for f in features_meta]

        if hasattr(self, "clustering_summary_tree"):
            self.clustering_summary_tree.delete(
                *self.clustering_summary_tree.get_children()
            )
            for record in summary:
                run_key = record.get("run_key")
                self.clustering_summary_tree.insert(
                    "",
                    "end",
                    iid=run_key if run_key else "",
                    values=(
                        record.get("method_label"),
                        record.get("cluster_count"),
                        record.get("rows"),
                    ),
                )

        if hasattr(self, "clustering_clusters_tree"):
            self.clustering_clusters_tree.delete(
                *self.clustering_clusters_tree.get_children()
            )
            for record in clusters:
                run_key = record.get("run_key")
                self.clustering_clusters_tree.insert(
                    "",
                    "end",
                    iid=f"{run_key}:{record.get('cluster')}" if run_key else "",
                    values=(
                        record.get("method_label"),
                        record.get("cluster"),
                        record.get("count"),
                    ),
                )

        if summary:
            methods_list = ", ".join(
                str(item.get("method_label")) for item in summary
            )
            self.clustering_status_var.set(
                f"Clustering complete for: {methods_list}."
            )
        else:
            self.clustering_status_var.set(
                "Clustering finished with no successful results."
            )

        if labels_map:
            self.clustering_method_labels = dict(labels_map)
        elif self.pending_clustering_labels:
            self.clustering_method_labels = dict(self.pending_clustering_labels)
        else:
            self.clustering_method_labels = dict(self.base_clustering_method_labels)
        self.pending_clustering_labels = {}

        self._update_clustering_visual_controls()

        if errors:
            messagebox.showwarning(
                "Clustering warnings",
                "Some clustering methods failed:\n" + "\n".join(errors),
            )

    def _handle_clustering_failure(self, message: str) -> None:
        self.clustering_in_progress = False
        self.clustering_progress.stop()
        self.run_clustering_button.configure(state="normal")
        self.clustering_status_var.set(f"Clustering failed: {message}")
        messagebox.showerror("Clustering error", message)
        self.pending_clustering_labels = {}
        if hasattr(self, "clustering_umap_method_combo"):
            self._update_clustering_visual_controls()

    def save_clustering_output(self) -> None:
        if not self.clustering_results:
            messagebox.showerror(
                "No results", "Run clustering before saving assignments."
            )
            return
        base_df = self.clustering_dataset_cache
        if base_df is None or base_df.empty:
            messagebox.showerror(
                "No data",
                "Original clustering dataset is unavailable.",
            )
            return

        export_df = base_df.copy()
        added_column = False
        annotation_added = False
        for method_key, result_df in self.clustering_results.items():
            if result_df.empty or "cluster" not in result_df.columns:
                continue
            label = self.clustering_method_labels.get(method_key, method_key)
            sanitized_method = self._sanitize_for_column(label)
            col_name = f"cluster_{sanitized_method or method_key}"
            if len(result_df) != len(export_df):
                messagebox.showwarning(
                    "Length mismatch",
                    f"Cluster results for '{label}' do not match dataset size and were skipped.",
                )
                continue
            export_df[col_name] = result_df["cluster"].values
            added_column = True

            annotation_df = self.cluster_annotations.get(method_key)
            if annotation_df is not None and not annotation_df.empty:
                annotation_df = annotation_df.copy()
                annotation_df["cluster"] = annotation_df["cluster"].astype(str)
                annotation_df = annotation_df.drop_duplicates(subset="cluster", keep="first")
                annotation_df = annotation_df.set_index("cluster")
                annotation_columns = [
                    column for column in annotation_df.columns if column != "cluster"
                ]
                if annotation_columns:
                    cluster_series = result_df["cluster"].astype(str)
                    for annotation_column in annotation_columns:
                        export_column = (
                            f"annotation_{sanitized_method}_"
                            f"{self._sanitize_for_column(annotation_column)}"
                        )
                        try:
                            values = cluster_series.map(annotation_df[annotation_column])
                        except KeyError:
                            continue
                        export_df[export_column] = values.fillna("")
                        annotation_added = True

        if annotation_added:
            self.cluster_annotation_status_var.set(
                "Annotations were included in the exported clustering file."
            )

        if not added_column:
            messagebox.showerror(
                "No clusters",
                "No clustering columns were available to export.",
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Save clustering output",
            defaultextension=".csv",
            initialfile="clustering_results.csv",
            filetypes=[
                ("CSV file", "*.csv"),
                ("Parquet file", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".parquet"):
                export_df.to_parquet(file_path, index=False)
            else:
                export_df.to_csv(file_path, index=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", f"Failed to save clustering output: {exc}")
            return
        base_meta = dict(self.clustering_run_metadata_base)
        if "features" not in base_meta:
            base_meta["features"] = list(self.clustering_features_used)
        if "files" not in base_meta:
            base_meta["files"] = [str(data_file.path) for data_file in self.data_files]
        metadata_export = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base": base_meta,
            "methods": self.clustering_metadata,
        }
        metadata_export = self._to_serializable(metadata_export)
        metadata_path = Path(file_path).with_name(
            Path(file_path).stem + "_metadata.json"
        )
        try:
            metadata_path.write_text(
                json.dumps(metadata_export, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning(
                "Partial save",
                f"Clustering data saved, but failed to write metadata: {exc}",
            )
            metadata_path = None
        if metadata_path:
            messagebox.showinfo(
                "Saved",
                f"Clustering output saved to {file_path}\nMetadata saved to {metadata_path}",
            )
        else:
            messagebox.showinfo("Saved", f"Clustering output saved to {file_path}")

    def _save_figure(self, figure: Optional[Figure], default_filename: str) -> None:
        if figure is None:
            messagebox.showerror("Save figure", "Figure is not available yet.")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            figure.savefig(file_path, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", f"Failed to save figure: {exc}")
            return
        messagebox.showinfo("Figure saved", f"Figure saved to {file_path}")

    def _save_training_columns_figure(self) -> None:
        self._save_figure(self.columns_fig, "columns_overview.png")

    def _save_training_category_figure(self) -> None:
        self._save_figure(self.category_fig, "target_distribution.png")

    def _save_training_visual_importance_figure(self) -> None:
        self._save_figure(self.visual_importance_fig, "feature_importance_visuals.png")

    def _save_confusion_figure(self) -> None:
        self._save_figure(self.confusion_fig, "confusion_matrix.png")

    def _save_training_importance_figure(self) -> None:
        self._save_figure(self.importance_fig, "model_feature_importance.png")

    def _save_clustering_umap_figure(self) -> None:
        self._save_figure(self.clustering_umap_fig, "clustering_umap.png")

    def _save_clustering_heatmap_figure(self) -> None:
        self._save_figure(self.clustering_heatmap_fig, "clustering_heatmap.png")

    def _update_heatmap_canvas(self, figure: Figure) -> None:
        if not hasattr(self, "clustering_heatmap_canvas"):
            return
        widget = self.clustering_heatmap_canvas.get_tk_widget()
        grid_info = widget.grid_info()
        master = widget.master
        widget.destroy()
        info = {k: v for k, v in grid_info.items() if k != "in"}
        row = int(info.pop("row", 0))
        column = int(info.pop("column", 0))
        rowspan = int(info.pop("rowspan", 1))
        columnspan = int(info.pop("columnspan", 1))
        sticky = info.pop("sticky", "")
        padx = info.pop("padx", 0)
        pady = info.pop("pady", 0)
        ipadx = info.pop("ipadx", 0)
        ipady = info.pop("ipady", 0)
        self.clustering_heatmap_canvas = FigureCanvasTkAgg(figure, master=master)
        new_widget = self.clustering_heatmap_canvas.get_tk_widget()
        new_widget.grid(
            row=row,
            column=column,
            rowspan=rowspan,
            columnspan=columnspan,
            sticky=sticky,
            padx=padx,
            pady=pady,
            ipadx=ipadx,
            ipady=ipady,
        )
        self._make_canvas_responsive(self.clustering_heatmap_canvas, figure, min_height=240)
        self.clustering_heatmap_fig = figure

    @staticmethod
    def _normalize_filter_value(value: object) -> object:
        if isinstance(value, np.generic):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _format_filter_value(value: object) -> str:
        try:
            if pd.isna(value):
                return "<NA>"
        except TypeError:
            # pd.isna raises for containers; fall back to default string conversion.
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    def _stream_column_series(
        self,
        column: str,
        data_files: Optional[List[DataFile]] = None,
        chunksize: Optional[int] = None,
    ) -> Iterator[pd.Series]:
        files = data_files or self.data_files
        for data_file in files:
            if not data_file.has_column(column):
                continue
            reader = data_file.iter_chunks(
                usecols=[column],
                chunksize=chunksize or self.csv_chunksize,
            )
            for chunk in reader:
                yield chunk[column]

    def _get_or_create_column_profile(self, column: str) -> ColumnProfile:
        profile = self.column_profiles.get(column)
        if profile is None:
            profile = ColumnProfile(column=column, is_numeric=self.column_numeric_hints.get(column))
            self.column_profiles[column] = profile
        return profile

    def _collect_column_categories(self, column: str) -> tuple[List[object], bool]:
        values: List[object] = []
        seen: Set[object] = set()
        truncated = False
        for series in self._stream_column_series(column):
            unique_values = pd.unique(series)
            for value in unique_values:
                normalized = self._normalize_filter_value(value)
                try:
                    hashable = normalized
                except Exception:
                    hashable = str(normalized)
                if hashable in seen:
                    continue
                seen.add(hashable)
                values.append(normalized)
                if len(values) >= MAX_UNIQUE_CATEGORY_SAMPLE:
                    truncated = True
                    values = values[:MAX_UNIQUE_CATEGORY_SAMPLE]
                    return sorted(values, key=lambda val: str(val).lower()), truncated
        values.sort(key=lambda val: str(val).lower())
        return values, truncated

    def _collect_numeric_range(self, column: str) -> Optional[Dict[str, float]]:
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        found = False
        for series in self._stream_column_series(column):
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_series = numeric_series.dropna()
            if numeric_series.empty:
                continue
            found = True
            series_min = float(numeric_series.min())
            series_max = float(numeric_series.max())
            min_val = series_min if min_val is None else min(min_val, series_min)
            max_val = series_max if max_val is None else max(max_val, series_max)
        if not found or min_val is None or max_val is None:
            return None
        return {"min": min_val, "max": max_val}

    def _ensure_categorical_options(self, column: str) -> bool:
        if column in self.clustering_categorical_options:
            return True
        values, truncated = self._collect_column_categories(column)
        if not values and not truncated:
            self.clustering_categorical_options[column] = []
        else:
            self.clustering_categorical_options[column] = values
        profile = self._get_or_create_column_profile(column)
        profile.categories = self.clustering_categorical_options.get(column, [])
        profile.categories_capped = truncated
        profile.unique_count = len(profile.categories)
        profile.unique_capped = truncated
        return True

    def _ensure_numeric_range(self, column: str) -> bool:
        if column in self.clustering_numeric_ranges:
            return True
        range_values = self._collect_numeric_range(column)
        if range_values is None:
            return False
        self.clustering_numeric_ranges[column] = range_values
        profile = self._get_or_create_column_profile(column)
        profile.numeric_min = range_values["min"]
        profile.numeric_max = range_values["max"]
        profile.is_numeric = True
        return True

    def _combined_dataframe(self, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.data_files:
            raise ValueError("No data available.")
        try:
            return self.data_engine.fetch_dataframe(columns=required_columns)
        except DataEngineError as exc:  # pragma: no cover - surfaced in UI
            raise ValueError(str(exc)) from exc

    def _downsample_dataset(
        self,
        method: str,
        target_value: int,
        class_column: Optional[str],
        dataset: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, str]:
        method = method or "None"
        if dataset is None:
            dataset = self._combined_dataframe()
        else:
            dataset = dataset.copy()

        if dataset.empty and method != "None":
            raise ValueError("No data available for downsampling.")

        if "__source_file" not in dataset.columns:
            dataset["__source_file"] = "source"

        if method == "None":
            return dataset, f"No downsampling applied. Combined rows: {len(dataset)}."

        if method == "Total Count":
            if target_value >= len(dataset):
                return dataset, (
                    f"Target ({target_value}) meets/exceeds total rows; returning full dataset of {len(dataset)} rows."
                )
            sampled = dataset.sample(n=target_value, random_state=RANDOM_STATE)
            return (
                sampled,
                f"Total count downsampling: {len(sampled)} rows (target {target_value}).",
            )

        if method == "Per File":
            frames = []
            for _source, group in dataset.groupby("__source_file", dropna=False):
                sample_size = min(target_value, len(group))
                if sample_size == len(group):
                    frames.append(group)
                elif sample_size > 0:
                    frames.append(
                        group.sample(n=sample_size, random_state=RANDOM_STATE)
                    )
            if not frames:
                raise ValueError("No data available for downsampling.")
            sampled = pd.concat(frames, ignore_index=True)
            return (
                sampled,
                f"Per-file downsampling applied with target {target_value} rows per file. Combined rows: {len(sampled)}.",
            )

        if method == "Per Class":
            if not class_column:
                raise ValueError(
                    "Select a categorical column before using class-based downsampling."
                )
            if class_column not in dataset.columns:
                raise ValueError("Selected class column is missing from the dataset.")

            frames = []
            for _value, group in dataset.groupby(class_column, dropna=False):
                sample_size = min(target_value, len(group))
                if sample_size == len(group):
                    frames.append(group)
                elif sample_size > 0:
                    frames.append(
                        group.sample(n=sample_size, random_state=RANDOM_STATE)
                    )

            if not frames:
                raise ValueError("No data available after class-based grouping.")
            sampled = pd.concat(frames, ignore_index=True)
            return (
                sampled,
                f"Per-class downsampling applied with target {target_value} rows per category. Combined rows: {len(sampled)}.",
            )

        if method == "Per File + Class":
            if not class_column:
                raise ValueError(
                    "Select a categorical column before using file+class downsampling."
                )
            if class_column not in dataset.columns:
                raise ValueError("Selected class column is missing from the dataset.")

            frames = []
            grouped = dataset.groupby(["__source_file", class_column], dropna=False)
            for _keys, group in grouped:
                sample_size = min(target_value, len(group))
                if sample_size == len(group):
                    frames.append(group)
                elif sample_size > 0:
                    frames.append(
                        group.sample(n=sample_size, random_state=RANDOM_STATE)
                    )

            if not frames:
                raise ValueError("No data available after per file + class downsampling.")
            sampled = pd.concat(frames, ignore_index=True)
            return (
                sampled,
                f"Per file + class downsampling applied with target {target_value} rows per file/category. Combined rows: {len(sampled)}.",
            )

        raise ValueError(f"Unsupported downsampling method '{method}'.")

    def _prepare_training_dataframe(self) -> pd.DataFrame:
        if (
            self.training_downsampled_df is not None
            and not self.training_downsampled_df.empty
        ):
            return self.training_downsampled_df.copy()

        if not self.training_selection:
            raise ValueError("Select feature columns before preparing the dataset.")
        target_column = self.target_column_var.get()
        if not target_column:
            raise ValueError("Select a target column before preparing the dataset.")

        required_columns = list(dict.fromkeys(self.training_selection + [target_column]))
        signature = self.data_engine.build_signature(
            columns=required_columns,
            extra={"mode": "training"},
        )
        cache_path = self.data_engine.ensure_cached_dataset(
            signature=signature,
            columns=required_columns,
        )
        return pd.read_parquet(cache_path)

    @staticmethod
    def _compute_class_weight_dict(y: pd.Series) -> Dict[object, float]:
        counts = y.value_counts(dropna=False)
        total = len(y)
        n_classes = len(counts)
        weights: Dict[object, float] = {}
        for cls, count in counts.items():
            if count == 0:
                continue
            weights[cls] = total / (n_classes * count)
        return weights

    @staticmethod
    def _sample_weight_array(y: pd.Series, class_weights: Dict[object, float]) -> np.ndarray:
        mapped = y.map(class_weights).astype(float)
        return mapped.to_numpy(dtype=float, copy=False)

    def _reset_results_view(self) -> None:
        if not hasattr(self, "metrics_text"):
            return
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.configure(state="disabled")
        self.training_status_var.set("Training not started.")
        if hasattr(self, "cv_summary_var"):
            self.cv_summary_var.set("Train a model to view cross-validation results.")

        if hasattr(self, "confusion_ax"):
            self.confusion_ax.clear()
            self.confusion_ax.set_title("Train a model to view the confusion matrix")
            self.confusion_ax.set_xlabel("Predicted")
            self.confusion_ax.set_ylabel("Actual")
            self.confusion_canvas.draw_idle()

        if hasattr(self, "importance_ax"):
            self.importance_ax.clear()
            self.importance_ax.set_title("Train a model to view feature importance")
            self.importance_ax.set_xlabel("Importance")
            self.importance_ax.set_ylabel("Feature")
            self.importance_canvas.draw_idle()

        if hasattr(self, "visual_importance_ax"):
            self.visual_importance_ax.clear()
            self.visual_importance_ax.set_title("Train a model to view feature importance")
            self.visual_importance_ax.set_xlabel("Importance")
            self.visual_importance_ax.set_ylabel("Feature")
            self.visual_importance_canvas.draw_idle()

        if hasattr(self, "save_model_button"):
            self.save_model_button.configure(state="disabled")

    def _clear_training_state(self) -> None:
        self.trained_model = None
        self.training_results = {}
        self.training_in_progress = False
        self._reset_results_view()

    def _clear_clustering_state(self) -> None:
        self.clustering_results = {}
        self.clustering_in_progress = False
        self.clustering_status_var.set("Clustering not started.")
        self.clustering_features_used = []
        self.clustering_dataset_cache = None
        self.clustering_feature_matrix = None
        self.clustering_umap_cache.clear()
        self.clustering_metadata = {}
        self.clustering_run_metadata_base = {}
        self.clustering_method_labels = dict(self.base_clustering_method_labels)
        self.clustering_subset_name_var.set("")
        self.clustering_filters.clear()
        self.clustering_filter_counter = 0
        self.clustering_filter_column_var.set("")
        self.clustering_filter_min_var.set("")
        self.clustering_filter_max_var.set("")
        self.clustering_filter_current_values = []
        self.clustering_filter_values_label_var.set("Select categories")
        self.clustering_filter_mode_var.set("AND")
        self.cluster_explorer_method_var.set("")
        self.cluster_explorer_feature_options = []
        self.cluster_explorer_status_var.set(
            "Load clustering results to explore clusters."
        )
        self.cluster_explorer_feature_menu = None
        self.pending_clustering_labels = {}
        self._refresh_clustering_filter_tree()
        self._on_clustering_filter_column_selected()
        self._update_clustering_subset_summary()
        if hasattr(self, "clustering_summary_tree"):
            self.clustering_summary_tree.delete(
                *self.clustering_summary_tree.get_children()
            )
        if hasattr(self, "clustering_clusters_tree"):
            self.clustering_clusters_tree.delete(
                *self.clustering_clusters_tree.get_children()
            )
        if hasattr(self, "clustering_umap_method_combo"):
            self.clustering_umap_method_combo["values"] = []
            self.clustering_umap_method_var.set("")
        if hasattr(self, "clustering_heatmap_method_combo"):
            self.clustering_heatmap_method_combo["values"] = []
            self.clustering_heatmap_method_var.set("")
        if hasattr(self, "clustering_umap_marker_combo"):
            self.clustering_umap_marker_combo["values"] = []
            self.clustering_umap_marker_var.set("")
            self.clustering_umap_marker_combo.configure(state="disabled")
        if hasattr(self, "clustering_heatmap_markers_listbox"):
            self.clustering_heatmap_markers_listbox.delete(0, tk.END)
        if hasattr(self, "clustering_umap_ax"):
            self.clustering_umap_ax.clear()
            self.clustering_umap_ax.set_title("Run clustering to generate UMAP")
            self.clustering_umap_ax.set_xticks([])
            self.clustering_umap_ax.set_yticks([])
            if hasattr(self, "clustering_umap_canvas"):
                self.clustering_umap_canvas.draw_idle()
        self.clustering_umap_colorbar = None
        if self.clustering_heatmap_colorbar is not None:
            try:
                self.clustering_heatmap_colorbar.remove()
            except Exception:
                pass
            finally:
                self.clustering_heatmap_colorbar = None
        if hasattr(self, "clustering_heatmap_canvas"):
            base_fig = Figure(figsize=(6, 4), dpi=100)
            base_ax = base_fig.add_subplot(111)
            base_ax.set_title("Run clustering to view heatmap")
            base_ax.set_xlabel("")
            base_ax.set_ylabel("")
            self._update_heatmap_canvas(base_fig)
            self.clustering_heatmap_ax = base_ax
            self.clustering_heatmap_canvas.draw_idle()
        if self.cluster_explorer_cluster_listbox is not None:
            self.cluster_explorer_cluster_listbox.delete(0, tk.END)
        for plot in self.cluster_explorer_plots:
            ax = plot["ax"]
            ax.clear()
            ax.set_title("Select clustering results")
            ax.set_xlabel("Feature X")
            ax.set_ylabel("Feature Y")
            plot["canvas"].draw_idle()
        self.cluster_annotation_method_var.set("")
        self.cluster_annotations = {}
        self.cluster_annotation_recent_terms = defaultdict(set)
        self.current_annotation_run_key = None
        self.cluster_annotation_info_var.set(
            "Select a clustering run to begin annotating clusters."
        )
        self.cluster_annotation_status_var.set("")
        if self.annotation_tree is not None:
            self.annotation_tree.delete(*self.annotation_tree.get_children())
            self.annotation_tree["columns"] = ("cluster",)
            self.annotation_tree.heading("cluster", text="Cluster")
        if self.annotation_edit_widget is not None:
            try:
                self.annotation_edit_widget.destroy()
            except Exception:
                pass
            self.annotation_edit_widget = None
            self.annotation_edit_info = None
        if hasattr(self, "clustering_umap_method_combo"):
            self._update_clustering_visual_controls()
        self._refresh_cluster_explorer_controls()
        self._refresh_annotation_method_choices()

    def start_clustering(self) -> None:
        if self.clustering_in_progress:
            return

        if not self.data_files:
            messagebox.showerror("No data", "Load CSV files before clustering.")
            return

        if not self.clustering_selection:
            messagebox.showerror(
                "No features",
                "Select at least one feature column for clustering.",
            )
            return

        if self.clustering_missing_var.get():
            messagebox.showerror(
                "Missing columns",
                "Some files are missing the selected clustering columns. "
                "Adjust the selection or address missing columns before clustering.",
            )
            return

        try:
            combined_before = self._combined_dataframe()
            base_dataset = self._get_filtered_dataframe_base(use_cached_combined=combined_before)
        except ValueError as exc:
            messagebox.showerror("Data error", str(exc))
            return

        if base_dataset.empty:
            messagebox.showerror(
                "Empty dataset",
                "The current filters produced an empty dataset.",
            )
            return

        if (
            self.clustering_downsampled_df is not None
            and not self.clustering_downsampled_df.empty
        ):
            dataset = self.clustering_downsampled_df.copy()
        else:
            dataset = base_dataset.copy()

        required_columns = list(self.clustering_selection)
        missing_in_dataset = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing_in_dataset:
            messagebox.showerror(
                "Missing data",
                "The prepared dataset is missing required columns:\n"
                + ", ".join(missing_in_dataset),
            )
            return

        rows_after_downsampling = len(dataset)
        dataset = dataset.dropna(subset=required_columns)
        rows_after_dropna = len(dataset)
        if dataset.empty:
            messagebox.showerror(
                "Empty dataset",
                "No rows remain after dropping records with missing values.",
            )
            return

        non_numeric = [
            column
            for column in required_columns
            if not is_numeric_dtype(dataset[column])
        ]
        if non_numeric:
            messagebox.showerror(
                "Non-numeric features",
                "Clustering requires numeric feature columns. "
                "The following selections are not numeric:\n"
                + ", ".join(non_numeric),
            )
            return

        max_combinations = 100
        selected_methods: List[
            tuple[str, str, Dict[str, object], Dict[str, object], str]
        ] = []
        pending_labels: Dict[str, str] = {}

        for method_key, method_info in self.clustering_methods.items():
            if not method_info["selected"].get():
                continue

            param_values_map: Dict[str, List[object]] = {}
            for param_key, param_cfg in method_info["params"].items():
                raw_value = str(param_cfg.get("var").get()).strip()
                parsed_values = self._parse_param_values(
                    raw_value, param_cfg, method_info["label"]
                )
                if not parsed_values:
                    messagebox.showerror(
                        "Missing parameter values",
                        f"Provide at least one value for '{param_cfg.get('label', param_key)}' "
                        f"in {method_info['label']}.",
                    )
                    return
                param_values_map[param_key] = parsed_values

            param_keys = list(param_values_map.keys())
            value_product = list(
                product(*(param_values_map[param_key] for param_key in param_keys))
            )

            if len(value_product) > max_combinations:
                messagebox.showerror(
                    "Too many combinations",
                    f"{method_info['label']} would run {len(value_product)} combinations. "
                    f"Reduce the number of parameter values (limit {max_combinations}).",
                )
                return

            for combo in value_product:
                params = {param_key: combo[idx] for idx, param_key in enumerate(param_keys)}
                run_key = self._build_clustering_run_key(method_key, params, method_info)
                run_label = self._build_clustering_run_label(method_info, params)
                if run_key in pending_labels:
                    continue
                selected_methods.append((run_key, method_key, method_info, params, run_label))
                pending_labels[run_key] = run_label

        if not selected_methods:
            messagebox.showerror(
                "No methods",
                "Select at least one clustering method to run.",
            )
            return

        self.pending_clustering_labels = pending_labels

        dataset = dataset.reset_index(drop=True)
        features = dataset[required_columns].to_numpy(dtype=float, copy=False)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        self.clustering_features_used = list(required_columns)
        self.clustering_dataset_cache = dataset.copy()
        self.clustering_feature_matrix = features_scaled.copy()
        self.clustering_umap_cache.clear()

        downsample_method = self.clustering_downsample_method_var.get()
        downsample_value_raw = self.clustering_downsample_value_var.get().strip()
        downsample_value: Optional[object]
        if downsample_value_raw:
            try:
                downsample_value = int(downsample_value_raw)
            except ValueError:
                try:
                    downsample_value = float(downsample_value_raw)
                except ValueError:
                    downsample_value = downsample_value_raw
        else:
            downsample_value = None

        total_rows_files = len(combined_before)
        subset_filters_metadata = [
            {
                "column": filt.get("column"),
                "type": filt.get("type"),
                "values": filt.get("values"),
                "min": filt.get("min"),
                "max": filt.get("max"),
            }
            for filt in self.clustering_filters
        ]
        timestamp = time.time()
        base_metadata = {
            "timestamp": timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "files": [str(data_file.path) for data_file in self.data_files],
            "features": list(required_columns),
            "total_rows_files": total_rows_files,
            "rows_after_downsampling": rows_after_downsampling,
            "rows_used": rows_after_dropna,
            "downsampling": {
                "method": downsample_method,
                "value": downsample_value,
                "class_column": self.clustering_class_var.get() or None,
            },
            "n_jobs": int(self.clustering_n_jobs_var.get()),
            "subset": {
                "name": self.clustering_subset_name_var.get().strip() or None,
                "filters": self._to_serializable(subset_filters_metadata),
                "rows_after_filters": len(base_dataset),
                "logic": (self.clustering_filter_mode_var.get() or "AND").upper(),
            },
        }
        self.clustering_run_metadata_base = base_metadata.copy()

        self.clustering_queue = queue.Queue()
        self.clustering_in_progress = True
        self.run_clustering_button.configure(state="disabled")
        self.clustering_progress.start(12)
        self.clustering_status_var.set("Clustering in progress…")

        clustering_jobs = max(
            1, min(self.clustering_n_jobs_var.get(), self.total_cpu_cores)
        )
        self.clustering_thread = Thread(
            target=self._clustering_worker,
            args=(
                dataset,
                features_scaled,
                selected_methods,
                clustering_jobs,
                base_metadata,
            ),
            daemon=True,
        )
        self.clustering_thread.start()
        self.root.after(150, self._check_clustering_queue)

    def start_training(self) -> None:
        if self.training_in_progress:
            return

        if not self.data_files:
            messagebox.showerror("No data", "Load CSV files before training.")
            return

        if not self.training_selection:
            messagebox.showerror(
                "No features",
                "Select at least one feature column for training.",
            )
            return

        if self.training_missing_var.get():
            messagebox.showerror(
                "Missing columns",
                "Some files are missing the selected feature columns. "
                "Adjust the selection or address missing columns before training.",
            )
            return

        target_column = self.target_column_var.get()
        if not target_column:
            messagebox.showerror(
                "No target",
                "Select a classification target column before training.",
            )
            return

        if self.target_missing_var.get():
            messagebox.showerror(
                "Target missing",
                "The selected target column is missing from at least one file. "
                "Please resolve this before training.",
            )
            return

        try:
            dataset = self._prepare_training_dataframe()
        except ValueError as exc:
            messagebox.showerror("Data error", str(exc))
            return

        required_columns = self.training_selection + [target_column]
        missing_in_dataset = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing_in_dataset:
            messagebox.showerror(
                "Missing data",
                "The prepared dataset is missing required columns:\n"
                + ", ".join(missing_in_dataset),
            )
            return

        dataset = dataset[required_columns].copy()
        dataset = dataset.dropna()
        if dataset.empty:
            messagebox.showerror(
                "Empty dataset",
                "No rows remain after dropping records with missing feature/target values.",
            )
            return

        non_numeric_features = [
            column
            for column in self.training_selection
            if not is_numeric_dtype(dataset[column])
        ]
        if non_numeric_features:
            messagebox.showerror(
                "Non-numeric features",
                "Random forest training requires numeric feature columns. "
                "The following selections are not numeric:\n"
                + ", ".join(non_numeric_features),
            )
            return

        unique_classes = dataset[target_column].nunique(dropna=True)
        if unique_classes < 2:
            messagebox.showerror(
                "Insufficient classes",
                "At least two distinct classes are required in the target column.",
            )
            return

        model_name = self.training_model_var.get()
        if not model_name or model_name not in self.training_model_configs:
            messagebox.showerror("Select model", "Choose a training model before starting.")
            return
        try:
            model_params = self._get_training_model_params(model_name)
        except ValueError as exc:
            messagebox.showerror("Invalid parameter", str(exc))
            return

        try:
            test_size = float(self.test_size_var.get())
        except (TypeError, ValueError):
            messagebox.showerror(
                "Invalid split",
                "Enter a numeric value for the test split.",
            )
            return
        if not 0.05 <= test_size <= 0.5:
            messagebox.showerror(
                "Split out of range",
                "Test split must be between 0.05 and 0.5.",
            )
            return

        cv_folds = max(int(self.cv_folds_var.get()), 2)

        n_jobs = int(self.n_jobs_var.get())
        if n_jobs < 1:
            n_jobs = 1
        if n_jobs > self.total_cpu_cores:
            n_jobs = self.total_cpu_cores
        self.n_jobs_var.set(n_jobs)

        config = {
            "model_name": model_name,
            "model_params": model_params,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "n_jobs": n_jobs,
            "features": list(self.training_selection),
            "target": target_column,
        }

        self.training_results = {}
        self.training_queue = queue.Queue()
        self.training_in_progress = True
        self.train_button.configure(state="disabled")
        self.training_progress.start(12)
        self.training_status_var.set("Training in progress…")
        if hasattr(self, "metrics_text"):
            self.metrics_text.configure(state="normal")
            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert(
                "1.0",
                "Training in progress… results will appear here once complete.",
            )
            self.metrics_text.configure(state="disabled")
        self.save_model_button.configure(state="disabled")

        self.training_thread = Thread(
            target=self._train_model_worker,
            args=(config, dataset),
            daemon=True,
        )
        self.training_thread.start()
        self.root.after(150, self._check_training_queue)

    def _train_model_worker(self, config: Dict[str, object], dataset: pd.DataFrame) -> None:
        try:
            start_time = time.time()
            model_name = config["model_name"]
            params = config["model_params"]
            features: List[str] = config["features"]  # type: ignore[assignment]
            target: str = config["target"]  # type: ignore[assignment]
            test_size = float(config["test_size"])  # type: ignore[arg-type]
            cv_folds = int(config["cv_folds"])  # type: ignore[arg-type]
            n_jobs = int(config["n_jobs"])  # type: ignore[arg-type]

            X = dataset[features]
            y = dataset[target]
            stratify = y if y.nunique() > 1 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=stratify,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=None,
                )

            class_balance_mode = self.class_balance_var.get()
            class_weight_dict: Optional[Dict[object, float]] = None
            fit_sample_weight: Optional[np.ndarray] = None
            if class_balance_mode != "None" and len(y_train) > 0:
                class_weight_dict = self._compute_class_weight_dict(y_train)
                fit_sample_weight = self._sample_weight_array(y_train, class_weight_dict)

            if model_name == "Random Forest":
                payload = self._train_random_forest_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    cv_folds,
                    n_jobs,
                    class_weight_dict,
                    fit_sample_weight,
                )
            elif model_name == "LDA":
                payload = self._train_lda_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    cv_folds,
                    n_jobs,
                    fit_sample_weight,
                )
            elif model_name == "SVM":
                payload = self._train_svm_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    cv_folds,
                    n_jobs,
                    class_weight_dict,
                    fit_sample_weight,
                )
            elif model_name == "Logistic Regression":
                payload = self._train_logistic_regression_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    cv_folds,
                    n_jobs,
                    class_weight_dict,
                    fit_sample_weight,
                )
            elif model_name == "Naive Bayes":
                payload = self._train_naive_bayes_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    fit_sample_weight,
                )
            elif model_name == "XGBoost":
                payload = self._train_xgboost_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    class_weight_dict,
                    fit_sample_weight,
                    n_jobs,
                )
            elif model_name == "LightGBM":
                payload = self._train_lightgbm_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    class_weight_dict,
                    fit_sample_weight,
                    n_jobs,
                )
            elif model_name == "KMeans":
                payload = self._train_kmeans_model(X_train, X_test, y_train, y_test, params)
            elif model_name == "Neural Network":
                payload = self._train_neural_network_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    params,
                    class_weight_dict,
                )
            else:
                raise ValueError(f"Unsupported model '{model_name}'.")

            payload.setdefault("training_time", time.time() - start_time)
            payload.setdefault("artifacts", {})
            payload.update(
                {
                    "model_name": model_name,
                    "features": features,
                    "target": target,
                }
            )
            payload["config"] = {
                "model_params": dict(params),
                "test_size": test_size,
                "cv_folds": cv_folds,
                "n_jobs": n_jobs,
            }
            self.training_queue.put({"status": "success", "payload": payload})
        except Exception as exc:  # noqa: BLE001
            self.training_queue.put({"status": "error", "message": str(exc)})

    def _classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray, class_labels: Optional[List[str]] = None) -> tuple[Dict[str, object], np.ndarray, List[str]]:
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report_text = classification_report(y_true, y_pred, digits=3, zero_division=0)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        if class_labels is None:
            class_labels = sorted(pd.unique(pd.Series(list(y_true) + list(y_pred))))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "report_text": report_text,
            "report_dict": report_dict,
        }
        return metrics, conf_matrix, class_labels

    def _perform_cross_validation(self, estimator, X_train, y_train, cv_folds: int, n_jobs: int) -> tuple[Optional[np.ndarray], str]:
        if cv_folds < 2 or len(y_train) < cv_folds:
            return None, ""
        try:
            scores = cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="f1_macro",
                n_jobs=n_jobs,
            )
            return scores, ""
        except ValueError as exc:
            return None, str(exc)

    def _train_random_forest_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        cv_folds: int,
        n_jobs: int,
        class_weights: Optional[Dict[object, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            class_weight=class_weights,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions, list(model.classes_))
        cv_scores, cv_warning = self._perform_cross_validation(
            RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                min_samples_leaf=params["min_samples_leaf"],
                n_jobs=n_jobs,
                random_state=RANDOM_STATE,
                class_weight=class_weights,
            ),
            X_train,
            y_train,
            cv_folds,
            n_jobs,
        )
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": model.feature_importances_,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": cv_scores,
            "cv_warning": cv_warning,
            "artifacts": {},
        }

    def _train_lda_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        cv_folds: int,
        n_jobs: int,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        lda = LinearDiscriminantAnalysis(solver=params["solver"], shrinkage=params["shrinkage"])
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", lda),
        ])
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["lda__sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        cv_scores, cv_warning = self._perform_cross_validation(model, X_train, y_train, cv_folds, n_jobs)
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": cv_scores,
            "cv_warning": cv_warning,
            "artifacts": {},
        }

    def _train_svm_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        cv_folds: int,
        n_jobs: int,
        class_weights: Optional[Dict[object, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        svm = SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            degree=params["degree"],
            class_weight=class_weights,
        )
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svm),
        ])
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["svm__sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        cv_scores, cv_warning = self._perform_cross_validation(model, X_train, y_train, cv_folds, n_jobs)
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": cv_scores,
            "cv_warning": cv_warning,
            "artifacts": {},
        }

    def _train_logistic_regression_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        cv_folds: int,
        n_jobs: int,
        class_weights: Optional[Dict[object, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        lr_kwargs = {
            "solver": params["solver"],
            "penalty": params["penalty"],
            "C": params["C"],
            "max_iter": params["max_iter"],
            "class_weight": class_weights,
            "n_jobs": n_jobs,
            "random_state": RANDOM_STATE,
        }
        if params["penalty"] == "elasticnet":
            lr_kwargs["l1_ratio"] = params.get("l1_ratio", 0.5)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(**lr_kwargs)),
            ]
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["logreg__sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        cv_scores, cv_warning = self._perform_cross_validation(model, X_train, y_train, cv_folds, n_jobs)
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": cv_scores,
            "cv_warning": cv_warning,
            "artifacts": {},
        }

    def _train_naive_bayes_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        model = GaussianNB(var_smoothing=params["var_smoothing"])
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train, **fit_kwargs)
        predictions = model.predict(X_test)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {},
        }

    def _train_xgboost_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        class_weights: Optional[Dict[object, float]],
        sample_weight: Optional[np.ndarray],
        n_jobs: int,
    ) -> Dict[str, object]:
        if XGBClassifier is None:
            raise ValueError("Install the 'xgboost' package to use this model.")
        X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
        X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
        num_classes = y_train.nunique()
        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective=objective,
            eval_metric="mlogloss",
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
        model.fit(X_train_np, y_train, sample_weight=sample_weight)
        predictions = model.predict(X_test_np)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        feature_importances = list(model.feature_importances_.tolist()) if hasattr(model, "feature_importances_") else []
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": feature_importances,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {},
        }

    def _train_lightgbm_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        class_weights: Optional[Dict[object, float]],
        sample_weight: Optional[np.ndarray],
        n_jobs: int,
    ) -> Dict[str, object]:
        if lgb is None:
            raise ValueError("Install the 'lightgbm' package to use this model.")
        X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
        X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
        model = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            subsample=params["subsample"],
            class_weight=class_weights,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_np, y_train, sample_weight=sample_weight)
        predictions = model.predict(X_test_np)
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        feature_importances = list(model.feature_importances_.tolist()) if hasattr(model, "feature_importances_") else []
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": feature_importances,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {},
        }

    def _build_cluster_label_map(self, clusters: np.ndarray, labels: pd.Series) -> tuple[Dict[int, object], object]:
        mapping: Dict[int, object] = {}
        fallback = labels.mode().iloc[0]
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            if mask.any():
                majority = labels[mask].mode()
                mapping[cluster_id] = majority.iloc[0] if not majority.empty else fallback
            else:
                mapping[cluster_id] = fallback
        return mapping, fallback

    def _train_kmeans_model(self, X_train, X_test, y_train, y_test, params: Dict[str, object]) -> Dict[str, object]:
        X_train_np = X_train.to_numpy(dtype=np.float64)
        X_test_np = X_test.to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)
        model = KMeans(
            n_clusters=params["n_clusters"],
            max_iter=params["max_iter"],
            n_init=params["n_init"],
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_scaled)
        train_clusters = model.predict(X_train_scaled)
        cluster_map, fallback = self._build_cluster_label_map(train_clusters, y_train.reset_index(drop=True))
        test_clusters = model.predict(X_test_scaled)
        predictions = np.array([cluster_map.get(cluster, fallback) for cluster in test_clusters])
        metrics, conf_matrix, classes = self._classification_metrics(y_test, predictions)
        ari = adjusted_rand_score(y_test, predictions)
        metrics.setdefault("extra_lines", []).append(f"Adjusted Rand Index: {ari:.3f}")
        metrics["extra_status"] = f"Adjusted Rand: {ari:.3f}."
        return {
            "model": model,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {"scaler": scaler, "cluster_label_map": cluster_map},
        }

    def _select_torch_device(self, prefer_gpu: bool) -> str:
        if torch is None or not prefer_gpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _train_neural_network_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        params: Dict[str, object],
        class_weights: Optional[Dict[object, float]] = None,
    ) -> Dict[str, object]:
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Install torch to enable neural network training."
            )
        X_train_np = X_train.to_numpy(dtype=np.float32)
        X_test_np = X_test.to_numpy(dtype=np.float32)
        classes = sorted(pd.unique(pd.concat([y_train, y_test])))
        device = self._select_torch_device(params.get("use_gpu", True))
        classifier = TorchNeuralNetClassifier(
            input_dim=X_train_np.shape[1],
            hidden_layers=params["hidden_layers"],
            output_dim=len(classes),
            activation=params["activation"],
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            device=device,
            class_weights=class_weights,
        )
        classifier.fit(X_train_np, y_train.to_numpy())
        predictions = classifier.predict(X_test_np)
        classifier.to_cpu()
        metrics, conf_matrix, class_labels = self._classification_metrics(y_test, predictions)
        return {
            "model": classifier,
            "metrics": metrics,
            "confusion_matrix": conf_matrix,
            "classes": class_labels,
            "feature_importances": [],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {},
        }

    @staticmethod
    def _strip_random_forest_oob(model: RandomForestClassifier) -> bool:
        removed = False
        for attr in ("oob_score_", "oob_decision_function_", "oob_prediction_"):
            if hasattr(model, attr):
                delattr(model, attr)
                removed = True
        return removed

    def _check_training_queue(self) -> None:
        if not self.training_in_progress:
            return

        try:
            message = self.training_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._check_training_queue)
            return

        if message["status"] == "success":
            self._handle_training_success(message["payload"])
        else:
            self._handle_training_failure(message["message"])

    def _handle_training_success(self, payload: Dict[str, object]) -> None:
        self.training_in_progress = False
        self.training_progress.stop()
        self.train_button.configure(state="normal")

        self.trained_model = payload["model"]  # type: ignore[assignment]
        self.training_results = payload
        metrics = payload["metrics"]  # type: ignore[assignment]

        accuracy = metrics["accuracy"]  # type: ignore[index]
        f1_macro = metrics["f1_macro"]  # type: ignore[index]
        elapsed = payload.get("training_time", 0.0)
        status_message = (
            f"Training complete in {elapsed:.2f}s. "
            f"Test accuracy: {accuracy:.3f}, macro F1: {f1_macro:.3f}."
        )
        extra_status = metrics.get("extra_status")  # type: ignore[attr-defined]
        if extra_status:
            status_message += f" {extra_status}"
        self.training_status_var.set(status_message)

        cv_scores = payload.get("cv_scores")
        cv_warning = payload.get("cv_warning", "")
        if isinstance(cv_scores, np.ndarray):
            cv_summary = (
                f"{len(cv_scores)}-fold CV (macro F1) — "
                f"mean: {cv_scores.mean():.3f}, std: {cv_scores.std():.3f}."
            )
        elif cv_scores is not None:
            cv_array = np.array(cv_scores)
            cv_summary = (
                f"{len(cv_array)}-fold CV (macro F1) — "
                f"mean: {cv_array.mean():.3f}, std: {cv_array.std():.3f}."
            )
        else:
            cv_summary = "Cross-validation skipped."
        if cv_warning:
            cv_summary += f" (CV warning: {cv_warning})"
        self.cv_summary_var.set(cv_summary)

        self._update_metrics_display(payload)
        self._update_confusion_matrix_plot(payload)
        self._update_importance_plot(payload)
        self.save_model_button.configure(state="normal")
        self._record_training_run(payload)

    def _handle_training_failure(self, message: str) -> None:
        self.training_in_progress = False
        self.training_progress.stop()
        self.train_button.configure(state="normal")
        self.training_status_var.set(f"Training failed: {message}")
        messagebox.showerror("Training error", message)

    def _update_metrics_display(self, payload: Dict[str, object]) -> None:
        if not hasattr(self, "metrics_text"):
            return
        metrics = payload["metrics"]  # type: ignore[index]
        train_rows = payload.get("train_rows", 0)
        test_rows = payload.get("test_rows", 0)
        accuracy = metrics["accuracy"]
        f1_macro = metrics["f1_macro"]
        f1_weighted = metrics["f1_weighted"]
        report_text = metrics["report_text"]

        summary_lines = [
            f"Training rows: {train_rows}",
            f"Test rows: {test_rows}",
            f"Accuracy: {accuracy:.3f}",
            f"Macro F1: {f1_macro:.3f}",
            f"Weighted F1: {f1_weighted:.3f}",
            "",
            "Classification report:",
            report_text,
        ]
        extra_lines = metrics.get("extra_lines") if isinstance(metrics, dict) else None
        if extra_lines:
            summary_lines.extend(["", *extra_lines])

        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", "\n".join(summary_lines))
        self.metrics_text.configure(state="disabled")

    def _update_confusion_matrix_plot(self, payload: Dict[str, object]) -> None:
        if not hasattr(self, "confusion_ax"):
            return
        conf_matrix = np.array(payload["confusion_matrix"])
        class_labels = payload["classes"]

        self.confusion_ax.clear()
        if conf_matrix.size == 0:
            self.confusion_ax.set_title("Confusion matrix unavailable")
            self.confusion_canvas.draw_idle()
            return

        sns.heatmap(
            conf_matrix,
            annot=False,
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=self.confusion_ax,
            cbar_kws={"shrink": 0.7},
        )
        self.confusion_ax.set_xlabel("Predicted")
        self.confusion_ax.set_ylabel("Actual")
        self.confusion_ax.set_title("Confusion matrix (counts)")
        self.confusion_ax.tick_params(labelsize=10)
        colorbar = self.confusion_ax.collections[0].colorbar if self.confusion_ax.collections else None
        if colorbar is not None:
            colorbar.ax.tick_params(labelsize=8)
        self._finalize_figure_layout(self.confusion_fig)
        self.confusion_canvas.draw_idle()

    def _update_importance_plot(self, payload: Dict[str, object]) -> None:
        if not hasattr(self, "importance_ax"):
            return
        importances = np.array(payload["feature_importances"])
        feature_names = payload["features"]

        self.importance_ax.clear()
        if hasattr(self, "visual_importance_ax"):
            self.visual_importance_ax.clear()

        if importances.size == 0:
            self.importance_ax.set_title("Feature importance unavailable")
            self.importance_canvas.draw_idle()
            if hasattr(self, "visual_importance_canvas"):
                self.visual_importance_ax.set_title("Feature importance unavailable")
                self.visual_importance_canvas.draw_idle()
            return

        sorted_pairs = sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
        top_n = min(20, len(sorted_pairs))
        top_features, top_values = zip(*sorted_pairs[:top_n])
        top_features_arr = list(top_features)
        top_values_arr = np.array(top_values)

        self.importance_ax.barh(
            top_features_arr[::-1], top_values_arr[::-1], color="#dd8452"
        )
        self.importance_ax.set_xlabel("Importance")
        self.importance_ax.set_ylabel("Feature")
        self.importance_ax.set_title("Top feature importances")
        self._finalize_figure_layout(self.importance_fig)
        self.importance_canvas.draw_idle()

        if hasattr(self, "visual_importance_ax"):
            self.visual_importance_ax.barh(
                top_features_arr[::-1], top_values_arr[::-1], color="#dd8452"
            )
            self.visual_importance_ax.set_xlabel("Importance")
            self.visual_importance_ax.set_ylabel("Feature")
            self.visual_importance_ax.set_title("Top feature importances")
            self._finalize_figure_layout(self.visual_importance_fig)
            self.visual_importance_canvas.draw_idle()

    def save_model(self) -> None:
        if self.trained_model is None:
            messagebox.showinfo("No model", "Train a model before saving.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib file", "*.joblib"), ("All files", "*.*")],
            title="Save trained model",
        )
        if not file_path:
            return

        model_to_save = self.trained_model
        save_options = {"keep_rf_oob": bool(self.keep_rf_oob_var.get())}
        if (
            isinstance(model_to_save, RandomForestClassifier)
            and not save_options["keep_rf_oob"]
        ):
            model_copy = copy.deepcopy(model_to_save)
            stripped = self._strip_random_forest_oob(model_copy)
            if stripped:
                save_options["rf_oob_stripped"] = True
                model_to_save = model_copy
            else:
                save_options["rf_oob_stripped"] = False

        payload = {
            "model_name": self.training_results.get("model_name"),
            "model": model_to_save,
            "features": self.training_results.get("features"),
            "target": self.training_results.get("target"),
            "metrics": self.training_results.get("metrics"),
            "classes": self.training_results.get("classes"),
            "confusion_matrix": self.training_results.get("confusion_matrix"),
            "cv_scores": self.training_results.get("cv_scores"),
            "cv_warning": self.training_results.get("cv_warning"),
            "artifacts": self.training_results.get("artifacts", {}),
            "config": self.training_results.get("config", {}),
            "training_time": self.training_results.get("training_time"),
            "timestamp": time.time(),
            "save_options": save_options,
        }
        try:
            joblib.dump(payload, file_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save error", f"Failed to save model: {exc}")
            return
        messagebox.showinfo("Model saved", f"Model saved to {file_path}")

    def _finalize_figure_layout(self, fig: Figure, **adjust_kwargs) -> None:
        """Apply a layout adjustment that avoids macOS/Tk tight_layout crashes."""
        if not adjust_kwargs:
            adjust_kwargs = {"bottom": 0.18}
        if IS_DARWIN:
            fig.subplots_adjust(**adjust_kwargs)
            return
        try:
            fig.tight_layout()
        except Exception:
            fig.subplots_adjust(**adjust_kwargs)

    def _make_canvas_responsive(
        self,
        canvas: FigureCanvasTkAgg,
        figure: Figure,
        min_width: int = 200,
        min_height: int = 160,
    ) -> None:
        widget = canvas.get_tk_widget()

        def _resize(event: tk.Event, fig=figure, canv=canvas) -> None:
            width = max(int(getattr(event, "width", widget.winfo_width())), min_width)
            height = max(int(getattr(event, "height", widget.winfo_height())), min_height)
            dpi = fig.get_dpi() or 100.0
            new_width = max(width / dpi, min_width / dpi)
            new_height = max(height / dpi, min_height / dpi)
            current_width, current_height = fig.get_size_inches()
            if (
                abs(current_width - new_width) < 0.05
                and abs(current_height - new_height) < 0.05
            ):
                return
            fig.set_size_inches(new_width, new_height, forward=False)
            canv.draw_idle()

        widget.bind("<Configure>", _resize, add="+")

    def _build_status_bar(self) -> None:
        status_frame = ttk.Frame(self.root, relief="sunken")
        status_frame.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, padding=(8, 2)).pack(
            side="left"
        )

    def select_files(self) -> None:
        file_paths = filedialog.askopenfilenames(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_paths:
            return
        paths = [Path(path_str) for path_str in file_paths]
        errors = self.load_files_from_paths(paths)

        if errors:
            messagebox.showwarning(
                "Some files failed",
                "Some files could not be loaded.\n" + "\n".join(errors[:10]),
            )

    def load_files_from_paths(self, paths: List[Path]) -> List[str]:
        loaded: List[DataFile] = []
        errors: List[str] = []

        for path in paths:
            try:
                metadata = self._build_data_file_metadata(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}: {exc}")
                continue

            loaded.append(metadata)

        if not loaded:
            self.status_var.set("Failed to load files")
            self.files_loaded_var.set("No files loaded")
            if not errors:
                errors.append("No files provided.")
            return errors

        self.data_files = loaded
        self.status_var.set(f"Loaded {len(self.data_files)} files")
        self.files_loaded_var.set(f"{len(self.data_files)} files loaded")

        self._refresh_views()
        self._mark_session_dirty()
        return errors

    def _build_data_file_metadata(self, path: Path) -> DataFile:
        sample = pd.read_csv(
            path,
            nrows=CSV_METADATA_SAMPLE_ROWS,
            low_memory=False,
        )
        columns = list(sample.columns)
        dtype_hints: Dict[str, str] = {
            column: sample[column].dtype.kind for column in columns
        }
        return DataFile(path=path, columns=columns, dtype_hints=dtype_hints)

    def remove_selected_files(self) -> None:
        if not self.data_files:
            messagebox.showinfo("No files", "No files are currently loaded.")
            return
        if not hasattr(self, "files_tree"):
            return
        selected_items = self.files_tree.selection()
        if not selected_items:
            messagebox.showinfo("Select files", "Select one or more files to remove.")
            return

        selected_ids = {str(item) for item in selected_items}
        remaining_files = [
            data_file for data_file in self.data_files if str(data_file.path) not in selected_ids
        ]
        removed_count = len(self.data_files) - len(remaining_files)
        if removed_count <= 0:
            messagebox.showinfo(
                "No files removed",
                "Could not match the selected entries to loaded files.",
            )
            return

        self.data_files = remaining_files
        self._refresh_views()
        self._mark_session_dirty()

        if self.data_files:
            self.status_var.set(f"Removed {removed_count} file(s).")
            self.files_loaded_var.set(f"{len(self.data_files)} files loaded")
        else:
            self.status_var.set("All files removed.")
            self.files_loaded_var.set("No files loaded")

    def _refresh_views(self) -> None:
        self._clear_training_state()
        self._clear_clustering_state()
        self.data_engine.sync_files(self.data_files)
        self.column_profiles.clear()
        self.column_numeric_hints.clear()
        self._update_summary()
        self._populate_files_tree()
        self._populate_columns_tree()
        self._update_training_controls()
        self._update_clustering_controls()
        self._update_columns_chart()
        self._update_category_chart()

    def _update_summary(self) -> None:
        total_files = len(self.data_files)
        total_rows = sum(data_file.row_count for data_file in self.data_files)

        unique_columns = set()
        for data_file in self.data_files:
            unique_columns.update(data_file.columns)

        self.total_files_var.set(f"Files: {total_files}")
        self.total_rows_var.set(f"Total cells (rows): {total_rows}")
        self.total_columns_var.set(f"Unique columns: {len(unique_columns)}")

    def _populate_files_tree(self) -> None:
        self.files_tree.delete(*self.files_tree.get_children())

        for data_file in self.data_files:
            item_id = str(data_file.path)
            self.files_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(data_file.name, data_file.row_count, len(data_file.columns)),
            )

    def _populate_columns_tree(self) -> None:
        self.columns_tree.delete(*self.columns_tree.get_children())
        self.column_presence.clear()
        self.common_columns = []

        for data_file in self.data_files:
            for column in data_file.columns:
                self.column_presence[column] = self.column_presence.get(column, 0) + 1

        sorted_columns = sorted(
            self.column_presence.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )

        total_files = len(self.data_files)
        if total_files:
            self.common_columns = sorted(
                [
                    column_name
                    for column_name, count in self.column_presence.items()
                    if count == total_files
                ],
                key=str.lower,
            )

        for column_name, count in sorted_columns:
            self.columns_tree.insert("", "end", values=(column_name, count))

        self._recompute_column_numeric_hints()
        self._compute_schema_report()

    def _compute_schema_report(self) -> None:
        if not self.data_files:
            self.schema_report = []
            self.quality_overview = {}
            self._update_quality_report_view()
            return

        report: List[Dict[str, object]] = []
        for column_name in sorted(self.column_presence.keys(), key=str.lower):
            numeric = self.column_numeric_hints.get(column_name, False)
            try:
                stats = self.data_engine.column_stats(column_name, numeric)
            except DataEngineError as exc:
                report.append(
                    {
                        "column": column_name,
                        "dtype": "Numeric" if numeric else "Categorical",
                        "non_null_pct": 0.0,
                        "missing_pct": 0.0,
                        "unique": 0,
                        "min": None,
                        "max": None,
                        "notes": str(exc),
                    }
                )
                continue

            total_rows = stats.get("total_rows") or 0
            non_null = stats.get("non_null") or 0
            missing = stats.get("missing") or 0
            unique_vals = stats.get("unique_vals") or 0
            non_null_pct = (non_null / total_rows * 100) if total_rows else 0.0
            missing_pct = (missing / total_rows * 100) if total_rows else 0.0
            min_value = stats.get("min_value")
            max_value = stats.get("max_value")
            notes: List[str] = []
            if missing_pct > 20:
                notes.append("High missing rate")
            if not numeric and total_rows and unique_vals > total_rows * 0.8:
                notes.append("High cardinality")
            if numeric:
                avg_value = stats.get("avg_value")
                std_value = stats.get("std_value")
                q1 = stats.get("q1_value")
                q3 = stats.get("q3_value")
                if std_value is not None and std_value not in (0, "nan") and std_value:
                    try:
                        upper_z = (float(max_value) - float(avg_value)) / float(std_value)
                        if upper_z > 6:
                            notes.append("Extreme upper values")
                    except (TypeError, ZeroDivisionError):
                        pass
                if q1 is not None and q3 is not None:
                    try:
                        iqr = float(q3) - float(q1)
                        if iqr > 0 and max_value is not None and min_value is not None:
                            upper = float(q3) + 1.5 * iqr
                            lower = float(q1) - 1.5 * iqr
                            if float(max_value) > upper:
                                notes.append("Upper outliers")
                            if float(min_value) < lower:
                                notes.append("Lower outliers")
                    except (TypeError, ValueError):
                        pass
                profile = self._get_or_create_column_profile(column_name)
                profile.numeric_min = min_value if min_value is None or isinstance(min_value, (int, float)) else profile.numeric_min
                profile.numeric_max = max_value if max_value is None or isinstance(max_value, (int, float)) else profile.numeric_max
                profile.is_numeric = True
            else:
                profile = self._get_or_create_column_profile(column_name)
                profile.categories = profile.categories or []
            report.append(
                {
                    "column": column_name,
                    "dtype": "Numeric" if numeric else "Categorical",
                    "non_null_pct": non_null_pct,
                    "missing_pct": missing_pct,
                    "unique": unique_vals,
                    "min": min_value,
                    "max": max_value,
                    "notes": ", ".join(notes),
                }
            )

        try:
            overview = self.data_engine.overall_row_stats()
        except DataEngineError:
            overview = {}
        self.schema_report = report
        self.quality_overview = overview
        self._update_quality_report_view()

    def _update_quality_report_view(self) -> None:
        if not hasattr(self, "quality_tree") or self.quality_tree is None:
            return
        tree = self.quality_tree
        tree.delete(*tree.get_children())
        for row in self.schema_report:
            tree.insert(
                "",
                "end",
                values=(
                    row.get("column", ""),
                    row.get("dtype", ""),
                    f"{row.get('non_null_pct', 0.0):.1f}%",
                    f"{row.get('missing_pct', 0.0):.1f}%",
                    row.get("unique", 0),
                    self._format_quality_value(row.get("min")),
                    self._format_quality_value(row.get("max")),
                    row.get("notes", ""),
                ),
            )
        if not hasattr(self, "quality_summary_text") or self.quality_summary_text is None:
            return
        overview = self.quality_overview or {}
        lines = []
        total_rows = overview.get("total_rows")
        if total_rows is not None:
            lines.append(f"Total rows: {total_rows:,}")
        duplicates = overview.get("duplicate_rows")
        if duplicates is not None:
            lines.append(f"Duplicate rows: {duplicates:,}")
        flagged = [row for row in self.schema_report if row.get("notes")]
        if flagged:
            lines.append("")
            lines.append("Columns needing attention:")
            for entry in flagged[:6]:
                lines.append(f"- {entry['column']}: {entry['notes']}")
            if len(flagged) > 6:
                lines.append(f"…and {len(flagged) - 6} more.")
        if not lines:
            lines = ["No issues detected so far."]
        summary = "\n".join(lines)
        text_widget = self.quality_summary_text
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", summary)
        text_widget.configure(state="disabled")

    @staticmethod
    def _format_quality_value(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value)

    def _load_session_state(self) -> None:
        if not self.session_path.exists():
            return
        try:
            with self.session_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception:
            return
        file_paths = [Path(p) for p in state.get("files", []) if Path(p).exists()]
        if file_paths:
            self._loading_session = True
            try:
                self.load_files_from_paths(file_paths)
            finally:
                self._loading_session = False
        training_state = state.get("training") or {}
        if training_state:
            self._pending_training_restore = training_state
            self.root.after(300, self._apply_training_restore)

    def _apply_training_restore(self) -> None:
        state = self._pending_training_restore
        if not state:
            return
        self._pending_training_restore = None
        features = state.get("features", [])
        if features and hasattr(self, "training_listbox"):
            self.training_listbox.selection_clear(0, tk.END)
            available_map = {name: idx for idx, name in enumerate(self.available_training_columns)}
            for feature in features:
                idx = available_map.get(feature)
                if idx is not None:
                    self.training_listbox.selection_set(idx)
            self._on_training_selection_changed()
        target = state.get("target")
        if target and target in (self.target_combo["values"] or []):
            self.target_column_var.set(target)
            self.on_target_column_selected()
        balance = state.get("class_balance")
        if balance:
            self.class_balance_var.set(balance)
        model = state.get("model")
        if model and model in self.training_model_configs:
            self.training_model_var.set(model)
            self._on_training_model_changed()
        self._mark_session_dirty()

    def _mark_session_dirty(self) -> None:
        if self._loading_session:
            return
        if self.session_dirty:
            return
        self.session_dirty = True
        self.root.after(800, self._flush_session_state)

    def _flush_session_state(self) -> None:
        if not self.session_dirty:
            return
        state = {
            "files": [str(data_file.path) for data_file in self.data_files],
            "training": {
                "features": list(self.training_selection),
                "target": self.target_column_var.get(),
                "model": self.training_model_var.get(),
                "class_balance": self.class_balance_var.get(),
            },
        }
        try:
            with self.session_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
        except Exception:
            pass
        self.session_dirty = False

    def _recompute_column_numeric_hints(self) -> None:
        hints: Dict[str, bool] = {}
        for column_name in self.column_presence.keys():
            files_with_column = [
                data_file for data_file in self.data_files if data_file.has_column(column_name)
            ]
            if not files_with_column:
                continue
            hints[column_name] = all(
                self._is_column_numeric_in_file(data_file, column_name)
                for data_file in files_with_column
            )
        self.column_numeric_hints = hints

    @staticmethod
    def _is_column_numeric_in_file(data_file: DataFile, column: str) -> bool:
        dtype_kind = data_file.dtype_hint(column)
        if dtype_kind is None:
            return False
        return dtype_kind in {"b", "i", "u", "f", "c"}

    def _update_columns_chart(self) -> None:
        self.columns_ax.clear()

        if not self.column_presence:
            self.columns_ax.set_title("Load data to view column coverage")
            self.columns_ax.set_xlabel("")
            self.columns_ax.set_ylabel("")
            self.columns_canvas.draw_idle()
            return

        columns = list(self.column_presence.keys())
        counts = [self.column_presence[column] for column in columns]

        self.columns_ax.barh(columns, counts, color="#4C72B0")
        self.columns_ax.set_xlabel("Number of files containing column")
        self.columns_ax.set_ylabel("Column")
        self.columns_ax.invert_yaxis()
        self.columns_ax.set_title("Column coverage across loaded files")
        self._finalize_figure_layout(self.columns_fig, left=0.22)
        self.columns_canvas.draw_idle()

    @classmethod
    def _to_serializable(cls, value: object) -> object:
        if isinstance(value, dict):
            return {str(k): cls._to_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._to_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [cls._to_serializable(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return value


def main() -> None:
    root = tk.Tk()
    FlowDataApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
try:
    from xgboost import XGBClassifier  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[assignment]

try:
    import lightgbm as lgb  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore[assignment]
