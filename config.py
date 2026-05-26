import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG = {
    "data": "./datasets/BCCD/data.yaml",
    "model": "./yolov8n.pt",
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "device": "0",
    "project": "./runs",
    "name": "bccd_simple",
}


def load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration from the given path or from CONFIG_PATH environment variable."""
    config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML config files. Install with `pip install pyyaml`."
        )

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping/dictionary.")

    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in data.items() if v is not None})
    return config


if __name__ == "__main__":
    import json

    try:
        config = load_config()
        print(json.dumps(config, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(exc)
        raise
