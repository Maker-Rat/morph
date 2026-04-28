from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunContext:
    run_name: str
    output_dir: Path
    seed: int = 42
