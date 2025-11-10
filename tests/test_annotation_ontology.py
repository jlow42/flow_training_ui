import json
import tempfile
import unittest
from pathlib import Path

from annotation_ontology import AnnotationOntology


class AnnotationOntologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        storage_path = Path(self.tmpdir.name) / "state.json"
        self.ontology = AnnotationOntology(storage_path=storage_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_suggest_for_cluster_prioritises_required_markers(self) -> None:
        profile = {"CD3": 1.0, "CD45RA": 0.8, "CCR7": 0.7, "CD19": 0.0}
        suggestions = self.ontology.suggest_for_cluster("cell_type", profile)
        labels = [label for label, _score in suggestions]
        self.assertIn("Naive T cell", labels[:3])

    def test_canonicalise_maps_synonyms(self) -> None:
        result = self.ontology.canonicalize("CellType", "b-cell")
        self.assertEqual(result, "B cell")

    def test_custom_value_registration(self) -> None:
        column = "CellType"
        self.assertFalse(self.ontology.is_value_allowed(column, "My custom type"))
        self.ontology.register_custom_value(column, "My custom type")
        self.assertTrue(self.ontology.is_value_allowed(column, "My custom type"))

    def test_recipe_persistence(self) -> None:
        values = {"CellType": "NK cell", "cell_state": "Effector"}
        self.ontology.add_recipe("NK Effector", "Test recipe", values)
        storage_path = self.ontology.storage_path
        self.assertTrue(storage_path.exists())
        data = json.loads(storage_path.read_text())
        self.assertTrue(any(r["name"] == "NK Effector" for r in data["recipes"]))
        reloaded = AnnotationOntology(storage_path=storage_path)
        self.assertIn("NK Effector", reloaded.list_recipes())


if __name__ == "__main__":
    unittest.main()
