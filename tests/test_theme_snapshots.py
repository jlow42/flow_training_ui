import hashlib
import io
import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from theme import (
    APP_THEMES,
    FIGURE_STYLE_PRESETS,
    apply_figure_style,
    get_theme_colors,
)


class ThemeConfigurationTests(unittest.TestCase):
    def test_theme_color_fields_present(self) -> None:
        required_fields = {
            "background",
            "surface",
            "surface_alt",
            "text",
            "muted_text",
            "accent",
            "accent_hover",
            "border",
            "figure_face",
            "axes_face",
            "axes_edge",
            "grid",
            "error",
        }
        for theme_name in APP_THEMES:
            colors = get_theme_colors(theme_name)
            for field in required_fields:
                self.assertTrue(hasattr(colors, field), f"Missing {field} for {theme_name}")


class ThemeSnapshotTests(unittest.TestCase):
    SNAPSHOT_HASHES = {
        ("Journal", "light"): "9f0557312458118d7e652b32ec235961232b2807e88e75c8fef9733c2a1a01e0",
        ("Journal", "dark"): "d6e814fad9c30b6b199635e8a656a8429bd16e143eb465736a48f446ff6038ee",
        ("Web", "light"): "f10cba34c102dc8850cbcff383687354d9a4c43440d1682d956886d4e464204b",
        ("Web", "dark"): "e0be59e6d98571ce8c1b927ceda67cba0fbfe19d386c23b9ff8747a8d6433538",
    }

    def test_figure_snapshots(self) -> None:
        for style_name in FIGURE_STYLE_PRESETS:
            for theme_name in APP_THEMES:
                fig = Figure(figsize=(2.5, 2.5), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot([0, 1], [0, 1])
                ax.set_title("Snapshot")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                apply_figure_style(fig, style_name, theme_name)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png")
                digest = hashlib.sha256(buffer.getvalue()).hexdigest()
                expected = self.SNAPSHOT_HASHES[(style_name, theme_name)]
                self.assertNotEqual(
                    expected,
                    "",
                    f"Snapshot hash missing for {(style_name, theme_name)}",
                )
                self.assertEqual(
                    digest,
                    expected,
                    f"Snapshot mismatch for {(style_name, theme_name)}",
                )


if __name__ == "__main__":
    unittest.main()
