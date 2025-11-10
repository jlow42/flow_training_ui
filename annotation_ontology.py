"""Ontology helpers for cluster annotation suggestions and recipes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class MarkerOntologyEntry:
    """Represents a single ontology entry with marker requirements."""

    label: str
    required_markers: Set[str] = field(default_factory=set)
    supporting_markers: Set[str] = field(default_factory=set)
    forbidden_markers: Set[str] = field(default_factory=set)
    synonyms: Set[str] = field(default_factory=set)
    columns: Set[str] = field(default_factory=set)
    description: str = ""


DEFAULT_MARKER_ENTRIES: Sequence[MarkerOntologyEntry] = (
    MarkerOntologyEntry(
        label="Naive T cell",
        required_markers={"CD3"},
        supporting_markers={"CD45RA", "CCR7", "CD62L"},
        forbidden_markers={"CD19", "CD14"},
        synonyms={"T cell naive", "Naive T"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD3 positive, CD45RA/CCR7 high, lacks myeloid and B cell markers.",
    ),
    MarkerOntologyEntry(
        label="Memory T cell",
        required_markers={"CD3"},
        supporting_markers={"CD45RO", "CCR7", "CD27"},
        forbidden_markers={"CD19", "CD14"},
        synonyms={"T cell memory", "Memory T"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD3 positive with memory-associated markers (CD45RO/CCR7).",
    ),
    MarkerOntologyEntry(
        label="Cytotoxic T cell",
        required_markers={"CD3", "CD8"},
        supporting_markers={"Granzyme B", "Perforin", "CD16"},
        forbidden_markers={"CD19"},
        synonyms={"CD8 T cell", "CTL"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD3/CD8 positive with cytotoxic effector molecules.",
    ),
    MarkerOntologyEntry(
        label="Helper T cell",
        required_markers={"CD3", "CD4"},
        supporting_markers={"CXCR5", "CCR4"},
        forbidden_markers={"CD19"},
        synonyms={"CD4 T cell", "Th cell"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD3/CD4 positive with helper-associated chemokine receptors.",
    ),
    MarkerOntologyEntry(
        label="Regulatory T cell",
        required_markers={"CD3", "CD4"},
        supporting_markers={"FOXP3", "CD25", "CTLA4"},
        forbidden_markers={"CD19"},
        synonyms={"Treg", "Reg T cell"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD4 positive with FOXP3/CD25/CTLA4 expression.",
    ),
    MarkerOntologyEntry(
        label="B cell",
        required_markers={"CD19"},
        supporting_markers={"CD20", "CD79A", "CD79B"},
        forbidden_markers={"CD3", "CD14"},
        synonyms={"B lymphocyte", "B-cell"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD19/CD20 positive without T cell or myeloid markers.",
    ),
    MarkerOntologyEntry(
        label="Plasma cell",
        required_markers={"CD38"},
        supporting_markers={"CD138", "XBP1"},
        forbidden_markers={"CD3"},
        synonyms={"Plasmablast"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="High CD38 with CD138/XBP1 consistent with plasma cells.",
    ),
    MarkerOntologyEntry(
        label="NK cell",
        required_markers={"CD56"},
        supporting_markers={"CD16", "NKG2D", "Granzyme B"},
        forbidden_markers={"CD3", "CD19"},
        synonyms={"Natural killer", "NK"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD56 positive, lacks CD3/CD19, often expresses cytotoxic markers.",
    ),
    MarkerOntologyEntry(
        label="Monocyte",
        required_markers={"CD14"},
        supporting_markers={"CD16", "CD11b", "HLA-DR"},
        forbidden_markers={"CD3"},
        synonyms={"CD14 monocyte", "Mono"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="CD14 positive myeloid population, often CD16/CD11b high.",
    ),
    MarkerOntologyEntry(
        label="Dendritic cell",
        required_markers={"CD11c"},
        supporting_markers={"HLA-DR", "CD86", "CD123"},
        forbidden_markers={"CD3", "CD19"},
        synonyms={"DC", "Dendritic"},
        columns={"cell_type", "annotation", "cell_identity"},
        description="Antigen-presenting cells expressing CD11c/HLA-DR.",
    ),
)


DEFAULT_COLUMN_ALIASES: Dict[str, str] = {
    "celltype": "cell_type",
    "cell_type": "cell_type",
    "cell type": "cell_type",
    "cell identity": "cell_identity",
    "cell_identity": "cell_identity",
    "identity": "cell_identity",
    "annotation": "annotation",
    "cluster annotation": "annotation",
    "label": "annotation",
    "cluster_label": "annotation",
    "cellstate": "cell_state",
    "cell_state": "cell_state",
    "cell state": "cell_state",
    "state": "cell_state",
    "phenotype": "phenotype",
}


DEFAULT_COLUMN_VALUES: Dict[str, Set[str]] = {
    "cell_state": {
        "Naive",
        "Memory",
        "Activated",
        "Exhausted",
        "Effector",
        "Resting",
    },
    "phenotype": {
        "CD4+",
        "CD8+",
        "CD19+",
        "Double negative",
        "Double positive",
        "Gamma delta",
    },
}


DEFAULT_RECIPES: Sequence[Dict[str, object]] = (
    {
        "name": "Naive T cell labeling",
        "description": "Assign naive T cell identity and naive state.",
        "values": {"cell_type": "Naive T cell", "cell_state": "Naive", "phenotype": "CD4+"},
    },
    {
        "name": "B cell labeling",
        "description": "Mark clusters with strong CD19/CD20 expression as B cells.",
        "values": {"cell_type": "B cell", "phenotype": "CD19+"},
    },
    {
        "name": "NK cell cytotoxic",
        "description": "Label NK clusters and set effector state.",
        "values": {"cell_type": "NK cell", "cell_state": "Effector"},
    },
)


class AnnotationOntology:
    """Provides marker-based suggestions, validation, and recipes."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.marker_entries: Tuple[MarkerOntologyEntry, ...] = tuple(DEFAULT_MARKER_ENTRIES)
        self.column_aliases: Dict[str, str] = dict(DEFAULT_COLUMN_ALIASES)
        self.base_values: Dict[str, Set[str]] = {
            key: set(values) for key, values in DEFAULT_COLUMN_VALUES.items()
        }
        self.allowed_lookup: Dict[str, Dict[str, str]] = {}
        self._build_allowed_lookup()

        self.storage_path = storage_path or (Path.home() / ".flow_training_ui" / "annotation_recipes.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.recipes: Dict[str, Dict[str, object]] = {}
        self.custom_values: Dict[str, Set[str]] = {}
        self._load_defaults()
        self._load_user_state()

    # ------------------------------------------------------------------
    # Public API
    def normalize_column(self, column_name: str) -> Optional[str]:
        key = column_name.strip().lower()
        return self.column_aliases.get(key, key if key in self.base_values else None)

    def suggest_for_cluster(
        self,
        column_name: str,
        marker_profile: Optional[Dict[str, float]],
        limit: int = 8,
    ) -> List[Tuple[str, int]]:
        column_key = self.normalize_column(column_name)
        if column_key is None:
            return []

        if not marker_profile:
            return [
                (entry.label, 1)
                for entry in self.marker_entries
                if column_key in entry.columns
            ][:limit]

        weights = self._normalise_marker_profile(marker_profile)
        expressed_high = {marker for marker, weight in weights.items() if weight >= 0.55}
        expressed_any = {marker for marker, weight in weights.items() if weight >= 0.18}
        expressed_top = set(list(sorted(weights, key=weights.get, reverse=True))[:6])

        suggestions: List[Tuple[str, int]] = []
        for entry in self.marker_entries:
            if column_key not in entry.columns:
                continue
            if entry.forbidden_markers & expressed_high:
                continue
            if entry.required_markers and not entry.required_markers <= expressed_any:
                continue

            score = 0
            score += 6 * len(entry.required_markers & expressed_high)
            score += 4 * len(entry.required_markers & expressed_any)
            score += 2 * len(entry.supporting_markers & expressed_top)
            score += 1 * len(entry.supporting_markers & expressed_any)

            if score == 0 and entry.required_markers:
                continue

            suggestions.append((entry.label, score or 1))

        suggestions.sort(key=lambda item: (-item[1], item[0].lower()))
        return suggestions[:limit]

    def canonicalize(self, column_name: str, value: str) -> str:
        column_key = self.normalize_column(column_name)
        if column_key is None:
            return value

        trimmed = value.strip()
        if not trimmed:
            return trimmed

        lookup = self.allowed_lookup.get(column_key, {})
        lower_value = trimmed.lower()
        if lower_value in lookup:
            return lookup[lower_value]

        for entry in self.marker_entries:
            if column_key not in entry.columns:
                continue
            if lower_value == entry.label.lower():
                return entry.label
            for synonym in entry.synonyms:
                if lower_value == synonym.lower():
                    return entry.label
        return trimmed

    def is_value_allowed(self, column_name: str, value: str) -> bool:
        if not value.strip():
            return True
        column_key = self.normalize_column(column_name)
        if column_key is None:
            return True
        lookup = self.allowed_lookup.get(column_key, {})
        lower_value = value.strip().lower()
        if lower_value in lookup:
            return True
        custom = self.custom_values.get(column_key, set())
        return lower_value in {val.lower() for val in custom}

    def allowed_values(self, column_name: str) -> List[str]:
        column_key = self.normalize_column(column_name)
        if column_key is None:
            return []
        values = set(self.allowed_lookup.get(column_key, {}).values())
        values.update(self.custom_values.get(column_key, set()))
        return sorted(values)

    def register_custom_value(self, column_name: str, value: str) -> None:
        column_key = self.normalize_column(column_name)
        if column_key is None:
            return
        self.custom_values.setdefault(column_key, set()).add(value)
        self._save_user_state()

    # Recipes ----------------------------------------------------------
    def list_recipes(self) -> List[str]:
        return sorted(self.recipes.keys(), key=str.lower)

    def get_recipe(self, name: str) -> Optional[Dict[str, object]]:
        return self.recipes.get(name)

    def add_recipe(
        self,
        name: str,
        description: str,
        values: Dict[str, str],
    ) -> None:
        canonical_values = {}
        for column, value in values.items():
            column_key = self.normalize_column(column)
            if column_key is None:
                continue
            canonical_values[column_key] = self.canonicalize(column, value)
        self.recipes[name] = {
            "name": name,
            "description": description,
            "values": canonical_values,
        }
        self._save_user_state()

    def remove_recipe(self, name: str) -> None:
        if name in self.recipes:
            del self.recipes[name]
            self._save_user_state()

    # ------------------------------------------------------------------
    # Internal helpers
    def _build_allowed_lookup(self) -> None:
        allowed_lookup: Dict[str, Dict[str, str]] = {}
        for entry in self.marker_entries:
            for column in entry.columns:
                column_key = self.normalize_column(column) or column
                allowed_lookup.setdefault(column_key, {})[entry.label.lower()] = entry.label
                for synonym in entry.synonyms:
                    allowed_lookup[column_key][synonym.lower()] = entry.label

        for column, values in self.base_values.items():
            column_key = self.normalize_column(column) or column
            for value in values:
                allowed_lookup.setdefault(column_key, {})[value.lower()] = value

        self.allowed_lookup = allowed_lookup

    def _normalise_marker_profile(self, profile: Dict[str, float]) -> Dict[str, float]:
        if not profile:
            return {}
        maximum = max(profile.values()) if profile else 0.0
        if maximum <= 0:
            return {marker: 0.0 for marker in profile}
        return {marker: value / maximum for marker, value in profile.items()}

    def _load_defaults(self) -> None:
        for recipe in DEFAULT_RECIPES:
            values = {
                self.normalize_column(column) or column: self.canonicalize(column, str(value))
                for column, value in recipe["values"].items()
            }
            self.recipes[recipe["name"]] = {
                "name": recipe["name"],
                "description": recipe.get("description", ""),
                "values": values,
            }

    def _load_user_state(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
        except Exception:  # noqa: BLE001
            return

        for recipe in data.get("recipes", []):
            name = recipe.get("name")
            values = recipe.get("values", {})
            if not isinstance(name, str) or not isinstance(values, dict):
                continue
            normalized = {
                self.normalize_column(column) or column: self.canonicalize(column, str(value))
                for column, value in values.items()
            }
            self.recipes[name] = {
                "name": name,
                "description": recipe.get("description", ""),
                "values": normalized,
            }

        custom_values = data.get("custom_values", {})
        if isinstance(custom_values, dict):
            for column, values in custom_values.items():
                column_key = self.normalize_column(column) or column
                if not isinstance(values, Iterable):
                    continue
                self.custom_values[column_key] = {str(value) for value in values}

    def _save_user_state(self) -> None:
        payload = {
            "recipes": list(self.recipes.values()),
            "custom_values": {
                column: sorted(values)
                for column, values in self.custom_values.items()
            },
        }
        try:
            self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:  # noqa: BLE001
            # Saving is a convenience feature; ignore failures silently.
            pass

