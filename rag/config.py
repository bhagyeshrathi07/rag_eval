"""Load and expose the YAML config as a typed, dotted-access object."""
from pathlib import Path
import yaml

_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    """Lightweight dotted-access wrapper around the config dict."""

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self):
        return self._data


def load_config(path: str | Path = _DEFAULT) -> Config:
    with open(path, "r") as f:
        return Config(yaml.safe_load(f))


# Convenience singleton for scripts that just want the defaults.
cfg = load_config()
