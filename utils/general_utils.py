from pathlib import Path
import yaml


# Resolves to the root of the project (two levels up from utils/)
project_root = Path(__file__).resolve().parents[1]


def load_config(path: Path):
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        path = project_root.joinpath(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)


