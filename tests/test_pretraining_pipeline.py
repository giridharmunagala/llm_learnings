from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.data.pretraining_pipeline import (
    build_default_pipeline_config,
    run_pipeline,
)


class PretrainingPipelineTests(unittest.TestCase):
    def test_pipeline_runs_and_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_root = Path(tmpdir) / "processed"
            records = [
                {"text": "This is a valid document about machine learning and the data pipeline."},
                {"text": "This is a valid document about machine learning and the data pipeline."},
                {"text": "Tiny"},
            ]
            with input_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            config = build_default_pipeline_config(
                input_path=str(input_path),
                output_root=str(output_root),
                run_name="test_run",
            )
            config.shard_size = 2
            config.quality.min_chars = 20
            summary = run_pipeline(config)

            export_path = output_root / "test_run" / "stage=tokenize_pack_export" / "source=pile_uncopyrighted" / "shard=00000.jsonl"
            exported = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(len(exported), 1)
            self.assertIn("pile_uncopyrighted", summary["sources"])


if __name__ == "__main__":
    unittest.main()
