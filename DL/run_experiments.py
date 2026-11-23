"""Run configured DL experiments sequentially with reproducible scheduling."""

import argparse
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_SCRIPTS = {
    "ggsatd": ROOT / "DL" / "GGSATD" / "ggsatd.py",
    "bilstm": ROOT / "DL" / "BiLSTM_Attention" / "bilstm_attention.py",
    "cnn": ROOT / "DL" / "CNN_SATD" / "cnn_satd.py",
    "ffnn": ROOT / "DL" / "FF" / "ffnn.py",
    "mt-bert": ROOT / "DL" / "2023-MT-BERT-SATD" / "mt-bert-satd" / "run_project_mt_bert.py",
}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = {"experiments", "repetitions", "delay_seconds", "randomize_order",
               "order_seed", "model_seed", "seed_mode", "output_dir"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    repetitions = config.get("repetitions", 10)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    delay = config.get("delay_seconds", 0)
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ValueError("delay_seconds must be a finite nonnegative number")
    randomize = config.get("randomize_order", True)
    if type(randomize) is not bool:
        raise ValueError("randomize_order must be a boolean")
    order_seed = config.get("order_seed", 6666)
    model_seed = config.get("model_seed", 6666)
    if type(order_seed) is not int or type(model_seed) is not int or model_seed < 0:
        raise ValueError("order_seed must be an integer and model_seed a nonnegative integer")
    seed_mode = config.get("seed_mode", "fixed")
    if seed_mode not in ("fixed", "increment"):
        raise ValueError("seed_mode must be fixed or increment")
    if model_seed + (repetitions - 1 if seed_mode == "increment" else 0) > 2**32 - 1:
        raise ValueError("model seeds must fit in the 32-bit range")
    output_dir = config.get("output_dir", str(Path("DL") / "experiment_runs"))
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a nonempty path string")
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a nonempty list")
    seen = set()
    for experiment in experiments:
        if not isinstance(experiment, dict) or set(experiment) != {"id", "model", "args"}:
            raise ValueError("Each experiment must contain only id, model and args")
        name, model, args = experiment["id"], experiment["model"], experiment["args"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Experiment id must contain only letters, digits, underscores or hyphens")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate experiment id: {name}")
        seen.add(name.casefold())
        if not isinstance(model, str) or model not in MODEL_SCRIPTS:
            raise ValueError(f"Unknown model: {model!r}")
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ValueError(f"{name}: args must be a list of strings")
        for arg in args:
            option = arg.split("=", 1)[0]
            if option in ("--", "-h") or (option.startswith("--") and any(
                    reserved.startswith(option) for reserved in ("--seed", "--output-dir", "--help"))):
                raise ValueError(f"{name}: {option} is managed by the experiment runner")
    return {
        "experiments": experiments, "repetitions": repetitions, "delay_seconds": delay,
        "randomize_order": randomize, "order_seed": order_seed, "model_seed": model_seed,
        "seed_mode": seed_mode, "output_dir": output_dir,
    }


def make_plan(config, session_dir):
    rng = random.Random(config["order_seed"])
    plan = []
    for repetition in range(1, config["repetitions"] + 1):
        experiments = list(config["experiments"])
        if config["randomize_order"]:
            rng.shuffle(experiments)
        seed = config["model_seed"] + (repetition - 1 if config["seed_mode"] == "increment" else 0)
        for experiment in experiments:
            run_dir = session_dir / f"rep-{repetition:03d}" / f"experiment-{experiment['id']}"
            command = [sys.executable, "-u", str(MODEL_SCRIPTS[experiment["model"]]),
                       *experiment["args"], "--seed", str(seed), "--output-dir", str(run_dir)]
            plan.append({
                "repetition": repetition, "experiment": experiment["id"],
                "model": experiment["model"], "seed": seed, "output_dir": str(run_dir),
                "command": command, "status": "pending",
            })
    return plan


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_schedule(config, session_dir):
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = session_dir / "manifest.json"
    manifest = {
        "created_at_utc": timestamp(), "working_directory": str(ROOT),
        "python_version": sys.version, "platform": platform.platform(),
        "configuration": config, "status": "running", "runs": make_plan(config, session_dir),
    }
    save_manifest(manifest_path, manifest)
    current = None
    try:
        for index, run in enumerate(manifest["runs"]):
            if index:
                time.sleep(config["delay_seconds"])
            current = run
            run_dir = Path(run["output_dir"])
            run_dir.mkdir(parents=True, exist_ok=False)
            run["status"] = "running"
            run["started_at_utc"] = timestamp()
            save_manifest(manifest_path, manifest)
            print(f"[{index + 1}/{len(manifest['runs'])}] {run['experiment']} "
                  f"(repetition {run['repetition']}, seed {run['seed']})", flush=True)
            with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(run["command"], cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            run["exit_code"] = result.returncode
            run["finished_at_utc"] = timestamp()
            run["status"] = "completed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                manifest["status"] = "failed"
                save_manifest(manifest_path, manifest)
                print(f"Experiment failed with exit code {result.returncode}; see {run_dir / 'run.log'}",
                      file=sys.stderr)
                return 1
            save_manifest(manifest_path, manifest)
            current = None
    except KeyboardInterrupt:
        if current is not None and current["status"] == "running":
            current["status"] = "interrupted"
            current["finished_at_utc"] = timestamp()
        manifest["status"] = "interrupted"
        save_manifest(manifest_path, manifest)
        print(f"Experiment run interrupted; see {manifest_path}", file=sys.stderr)
        return 130
    except OSError as exc:
        if current is not None and current["status"] == "running":
            current["status"] = "failed"
            current["finished_at_utc"] = timestamp()
            current["error"] = str(exc)
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        save_manifest(manifest_path, manifest)
        print(f"Experiment run failed: {exc}", file=sys.stderr)
        return 1
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = timestamp()
    save_manifest(manifest_path, manifest)
    print(f"Results saved in {session_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    args = parser.parse_args()
    try:
        config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    session_dir = output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    return run_schedule(config, session_dir)


if __name__ == "__main__":
    sys.exit(main())
