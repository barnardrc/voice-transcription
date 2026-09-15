import json
import tempfile
import unittest
from pathlib import Path

from utils import ConfigLoader


class ConfigLoaderTests(unittest.TestCase):
    def test_load_returns_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, "config.json")
            expected = {"audio": {"sample_rate": 16_000}}
            path.write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(ConfigLoader.load(path), expected)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load("does-not-exist.json")


if __name__ == "__main__":
    unittest.main()
