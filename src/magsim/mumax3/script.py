from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Script:
    commands: list[str]

    def to_script(self) -> str:
        return "\n".join(self.commands)

    def save(self, filepath: Path | str) -> None:
        Path(filepath).write_text(self.to_script())

    def print(self) -> None:
        print(self.to_script())
