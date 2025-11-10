"""UI theming and plotting style helpers for Flow Cytometry UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import tkinter as tk
import tkinter.ttk as ttk
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import seaborn as sns

DEFAULT_THEME = "light"
DEFAULT_FIGURE_STYLE = "Journal"


@dataclass(frozen=True)
class ThemeColors:
    background: str
    surface: str
    surface_alt: str
    text: str
    muted_text: str
    accent: str
    accent_hover: str
    border: str
    figure_face: str
    axes_face: str
    axes_edge: str
    grid: str
    error: str


APP_THEMES: Dict[str, Dict[str, object]] = {
    "light": {
        "label": "Light",
        "base_theme": "clam",
        "colors": ThemeColors(
            background="#F5F7FA",
            surface="#FFFFFF",
            surface_alt="#F0F4F8",
            text="#1F2933",
            muted_text="#52606D",
            accent="#0B6CFF",
            accent_hover="#0845A3",
            border="#CBD2D9",
            figure_face="#FFFFFF",
            axes_face="#FFFFFF",
            axes_edge="#9AA5B1",
            grid="#D2D6DC",
            error="#B91C1C",
        ),
    },
    "dark": {
        "label": "Dark",
        "base_theme": "clam",
        "colors": ThemeColors(
            background="#0F172A",
            surface="#111827",
            surface_alt="#1F2937",
            text="#F9FAFB",
            muted_text="#CBD5F5",
            accent="#3B82F6",
            accent_hover="#2563EB",
            border="#374151",
            figure_face="#0F172A",
            axes_face="#111827",
            axes_edge="#4B5563",
            grid="#374151",
            error="#F87171",
        ),
    },
}


FIGURE_STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "Journal": {
        "label": "Journal",
        "font_size": 10,
        "title_size": 12,
        "grid": True,
        "grid_style": ("--", 0.6),
        "color_cycle": [
            "#0B6CFF",
            "#D62E2E",
            "#14866D",
            "#B7791F",
            "#6B4CE6",
            "#C2410C",
        ],
    },
    "Web": {
        "label": "Web",
        "font_size": 11,
        "title_size": 13,
        "grid": True,
        "grid_style": ("-", 0.4),
        "color_cycle": [
            "#22D3EE",
            "#FBBF24",
            "#F472B6",
            "#60A5FA",
            "#34D399",
            "#F97316",
        ],
    },
}


def get_theme_colors(theme_name: str) -> ThemeColors:
    theme = APP_THEMES.get(theme_name)
    if not theme:
        theme = APP_THEMES[DEFAULT_THEME]
    return theme["colors"]  # type: ignore[return-value]


def get_color_cycle(preset_name: str, _theme_name: str) -> List[str]:
    preset = FIGURE_STYLE_PRESETS.get(preset_name)
    if not preset:
        preset = FIGURE_STYLE_PRESETS[DEFAULT_FIGURE_STYLE]
    return list(preset["color_cycle"])  # type: ignore[return-value]


def get_sequential_cmap(theme_name: str) -> ListedColormap:
    colors = get_theme_colors(theme_name)
    if theme_name == "dark":
        palette = [colors.surface_alt, colors.accent, "#FFFFFF"]
    else:
        palette = ["#FFFFFF", colors.accent, colors.text]
    return ListedColormap(palette)


def configure_global_palette(preset_name: str, theme_name: str) -> None:
    colors = get_theme_colors(theme_name)
    preset = FIGURE_STYLE_PRESETS.get(preset_name, FIGURE_STYLE_PRESETS[DEFAULT_FIGURE_STYLE])
    cycle = get_color_cycle(preset_name, theme_name)
    linestyle, linewidth = preset.get("grid_style", ("--", 0.6))
    sns.set_theme(
        style="whitegrid" if preset.get("grid", True) else "white",
        palette=cycle,
        rc={
            "figure.facecolor": colors.figure_face,
            "axes.facecolor": colors.axes_face,
            "axes.edgecolor": colors.axes_edge,
            "axes.labelcolor": colors.text,
            "axes.prop_cycle": cycler("color", cycle),
            "text.color": colors.text,
            "axes.grid": preset.get("grid", True),
            "grid.color": colors.grid,
            "grid.linestyle": linestyle,
            "grid.linewidth": linewidth,
            "xtick.color": colors.text,
            "ytick.color": colors.text,
            "axes.titlesize": preset.get("title_size", 12),
            "axes.labelsize": preset.get("font_size", 10),
            "font.size": preset.get("font_size", 10),
        },
    )


def apply_theme(root: tk.Tk, style: ttk.Style, theme_name: str) -> None:
    colors = get_theme_colors(theme_name)
    base_theme = APP_THEMES.get(theme_name, APP_THEMES[DEFAULT_THEME]).get("base_theme", "clam")
    if base_theme not in style.theme_names():
        base_theme = "clam"
    style.theme_use(base_theme)
    root.configure(bg=colors.background)
    style.configure("TFrame", background=colors.background)
    style.configure(
        "TLabelframe",
        background=colors.surface,
        foreground=colors.text,
        bordercolor=colors.border,
    )
    style.configure("TLabelframe.Label", background=colors.surface, foreground=colors.text)
    style.configure("TLabel", background=colors.surface, foreground=colors.text)
    style.configure(
        "TButton",
        background=colors.surface_alt,
        foreground=colors.text,
        bordercolor=colors.border,
        focusthickness=1,
    )
    style.map(
        "TButton",
        background=[("active", colors.accent_hover), ("pressed", colors.accent)],
        foreground=[("active", colors.surface), ("pressed", colors.surface)],
    )
    style.configure("TNotebook", background=colors.background, bordercolor=colors.border)
    style.configure(
        "TNotebook.Tab",
        background=colors.surface_alt,
        foreground=colors.muted_text,
        padding=(12, 6),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors.surface)],
        foreground=[("selected", colors.text)],
    )
    style.configure(
        "Treeview",
        background=colors.surface,
        fieldbackground=colors.surface,
        foreground=colors.text,
        bordercolor=colors.border,
        rowheight=22,
    )
    style.map(
        "Treeview",
        background=[("selected", colors.accent)],
        foreground=[("selected", colors.surface)],
    )
    style.configure(
        "TCombobox",
        fieldbackground=colors.surface,
        background=colors.surface,
        foreground=colors.text,
        bordercolor=colors.border,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", colors.surface)],
        foreground=[("readonly", colors.text)],
    )
    style.configure(
        "TEntry",
        fieldbackground=colors.surface,
        foreground=colors.text,
        bordercolor=colors.border,
    )
    style.configure("TMenubutton", background=colors.surface, foreground=colors.text)
    style.configure("Horizontal.TSeparator", background=colors.border)


def _apply_text_collection(texts: Iterable[Text], color: str) -> None:
    for text in texts:
        try:
            text.set_color(color)
        except Exception:
            continue


def _apply_legend_style(legend: Legend, colors: ThemeColors) -> None:
    legend.get_frame().set_facecolor(colors.surface)
    legend.get_frame().set_edgecolor(colors.border)
    _apply_text_collection(legend.get_texts(), colors.text)


def _apply_line_style(lines: Iterable[Line2D], colors: ThemeColors) -> None:
    for line in lines:
        line.set_linewidth(max(1.5, line.get_linewidth()))
        line.set_markeredgecolor(colors.axes_edge)


def _apply_patch_style(patches: Iterable[Patch], colors: ThemeColors) -> None:
    for patch in patches:
        patch.set_linewidth(0.8)
        patch.set_edgecolor(colors.axes_edge)


def _style_axes(ax: Axes, colors: ThemeColors, preset: Dict[str, object], cycle: List[str]) -> None:
    ax.set_facecolor(colors.axes_face)
    for spine in ax.spines.values():
        spine.set_color(colors.axes_edge)
        spine.set_linewidth(0.9)
    linestyle, linewidth = preset.get("grid_style", ("--", 0.6))
    if preset.get("grid", True):
        ax.grid(True, color=colors.grid, linestyle=linestyle, linewidth=linewidth)
    else:
        ax.grid(False)
    _apply_text_collection([ax.title, ax.xaxis.label, ax.yaxis.label], colors.text)
    ax.tick_params(colors=colors.text, which="both")
    _apply_text_collection(ax.get_xticklabels() + ax.get_yticklabels(), colors.text)
    ax.set_prop_cycle(cycler("color", cycle))


def apply_figure_style(figure: Figure, preset_name: str, theme_name: str) -> None:
    colors = get_theme_colors(theme_name)
    preset = FIGURE_STYLE_PRESETS.get(preset_name, FIGURE_STYLE_PRESETS[DEFAULT_FIGURE_STYLE])
    cycle = get_color_cycle(preset_name, theme_name)
    figure.set_facecolor(colors.figure_face)
    figure.patch.set_facecolor(colors.figure_face)
    for ax in figure.axes:
        _style_axes(ax, colors, preset, cycle)
        if ax.collections:
            for collection in ax.collections:
                try:
                    collection.set_edgecolor(colors.axes_edge)
                except Exception:
                    continue
        legend = ax.get_legend()
        if legend is not None:
            _apply_legend_style(legend, colors)
        _apply_line_style(ax.lines, colors)
        _apply_patch_style(ax.patches, colors)
    if figure.legends:
        for legend in figure.legends:
            _apply_legend_style(legend, colors)


__all__ = [
    "APP_THEMES",
    "DEFAULT_THEME",
    "DEFAULT_FIGURE_STYLE",
    "FIGURE_STYLE_PRESETS",
    "apply_theme",
    "apply_figure_style",
    "configure_global_palette",
    "get_theme_colors",
    "get_color_cycle",
    "get_sequential_cmap",
]
