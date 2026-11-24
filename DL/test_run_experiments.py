import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_experiments import MODEL_SCRIPTS, make_plan, run_schedule, validate_config


class ExperimentRunnerTests(unittest.TestCase):
    def test_schedule_is_blocked_reproducible_and_seeds_are_configurable(self):
        config = validate_config({
            "repetitions": 3, "order_seed": 123, "model_seed": 40,
            "seed_mode": "increment",
            "experiments": [
                {"id": "a", "model": "cnn", "args": []},
                {"id": "b", "model": "bilstm", "args": []},
                {"id": "c", "model": "ggsatd", "args": []},
            ],
        })
        root = Path("test-output").resolve()
        plan = make_plan(config, root)
        self.assertEqual(plan, make_plan(config, root))
        for repetition in range(1, 4):
            block = [run for run in plan if run["repetition"] == repetition]
            self.assertEqual({run["experiment"] for run in block}, {"a", "b", "c"})
            self.assertEqual({run["seed"] for run in block}, {39 + repetition})
        self.assertEqual(len({run["output_dir"] for run in plan}), 9)

    def test_rejects_conflicting_options_and_duplicate_ids(self):
        for option in ("--seed=3", "--se", "--output-dir", "--out", "--help", "-h"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_config({"experiments": [{"id": "a", "model": "cnn", "args": [option]}]})
        with self.assertRaises(ValueError):
            validate_config({"experiments": [
                {"id": "A", "model": "cnn", "args": []},
                {"id": "a", "model": "cnn", "args": []},
            ]})

    def test_nonrandom_schedule_groups_repetitions_by_experiment(self):
        config = validate_config({
            "repetitions": 3, "randomize_order": False, "model_seed": 40,
            "seed_mode": "increment",
            "experiments": [
                {"id": "a", "model": "cnn", "args": []},
                {"id": "b", "model": "bilstm", "args": []},
            ],
        })
        plan = make_plan(config, Path("test-output"))
        self.assertEqual([(r["experiment"], r["repetition"], r["seed"]) for r in plan], [
            ("a", 1, 40), ("a", 2, 41), ("a", 3, 42),
            ("b", 1, 40), ("b", 2, 41), ("b", 3, 42),
        ])
        self.assertEqual(len({run["output_dir"] for run in plan}), 6)

    def test_runs_real_subprocesses_with_isolated_outputs_and_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_model.py"
            script.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--seed', required=True)\n"
                "p.add_argument('--output-dir', required=True)\n"
                "a = p.parse_args()\n"
                "Path(a.output_dir, 'result.txt').write_text(a.seed)\n"
                "print('finished')\n", encoding="utf-8")
            config = validate_config({
                "repetitions": 2, "delay_seconds": 3, "randomize_order": False,
                "experiments": [
                    {"id": "one", "model": "cnn", "args": []},
                    {"id": "two", "model": "bilstm", "args": []},
                ],
            })
            session = root / "session"
            with patch.dict(MODEL_SCRIPTS, {"cnn": script, "bilstm": script}), \
                    patch("run_experiments.time.sleep") as sleep:
                self.assertEqual(run_schedule(config, session), 0)
            self.assertEqual(sleep.call_count, 3)
            self.assertTrue(all(call.args == (3,) for call in sleep.call_args_list))
            manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual([run["experiment"] for run in manifest["runs"]],
                             ["one", "one", "two", "two"])
            for run in manifest["runs"]:
                output = Path(run["output_dir"])
                self.assertEqual(run["status"], "completed")
                self.assertEqual((output / "result.txt").read_text(), "6666")
                self.assertIn("finished", (output / "run.log").read_text())

    def test_failure_is_recorded_and_stops_before_next_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fail.py"
            script.write_text("import sys\nprint('failure')\nsys.exit(7)\n", encoding="utf-8")
            config = validate_config({
                "repetitions": 2,
                "experiments": [{"id": "fail", "model": "cnn", "args": []}],
            })
            session = root / "session"
            with patch.dict(MODEL_SCRIPTS, {"cnn": script}), patch("run_experiments.time.sleep") as sleep:
                self.assertEqual(run_schedule(config, session), 1)
            sleep.assert_not_called()
            manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual([run["status"] for run in manifest["runs"]], ["failed", "pending"])
            self.assertEqual(manifest["runs"][0]["exit_code"], 7)
            self.assertIn("failure", Path(manifest["runs"][0]["output_dir"], "run.log").read_text())

    def test_interruption_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            config = validate_config({
                "repetitions": 2,
                "experiments": [{"id": "test", "model": "cnn", "args": []}],
            })
            with patch("run_experiments.subprocess.run", side_effect=KeyboardInterrupt):
                self.assertEqual(run_schedule(config, session), 130)
            manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual([run["status"] for run in manifest["runs"]], ["interrupted", "pending"])

    def test_cli_runs_cnn_on_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "data"
            dataset_dir.mkdir()
            with (dataset_dir / "maldonado.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["comment_text", "satd"])
                writer.writerows((f"todo fix word{i}", i % 2) for i in range(24))
            config = {
                "repetitions": 1, "output_dir": str(root / "outputs"),
                "experiments": [{
                    "id": "cnn", "model": "cnn",
                    "args": ["--task", "binary", "--dataset", "maldonado", "--dataset-dir",
                             str(dataset_dir), "--folds", "2", "--epochs", "1",
                             "--embedding-epochs", "1", "--filters", "2", "--batch-size", "16",
                             "--no-codecarbon"],
                }],
            }
            config_file = root / "config.json"
            config_file.write_text(json.dumps(config), encoding="utf-8")
            runner = Path(__file__).with_name("run_experiments.py")
            result = subprocess.run([sys.executable, str(runner), str(config_file)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifests = [path / "manifest.json" for path in (root / "outputs").iterdir()
                         if (path / "manifest.json").is_file()]
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            output = Path(manifest["runs"][0]["output_dir"])
            self.assertTrue((output / "CNN_SATD_fold_metrics.csv").is_file())
            self.assertTrue((output / "CNN_SATD_aggregated_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
